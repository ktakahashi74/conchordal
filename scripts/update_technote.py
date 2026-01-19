#!/usr/bin/env python3
"""
scripts/update_technote.py

Auto-update technote.md based on current codebase using Claude.
Also generates Japanese translation (technote.ja.md).

Output:
  - docs/generated/technote.{timestamp}.md     (English, versioned)
  - docs/generated/technote.ja.{timestamp}.md  (Japanese, versioned)
  - web/content/technote.md                    (English, production)
  - web/content/technote.ja.md                 (Japanese, production)

Usage:
  python scripts/update_technote.py                    # Default (Claude CLI, opus)
  python scripts/update_technote.py --model sonnet     # Use different model
  python scripts/update_technote.py --skip-ja          # English only
  python scripts/update_technote.py --ja-only          # Japanese only (from existing English)
  python scripts/update_technote.py --dry-run          # Preview only (shows diff)
  python scripts/update_technote.py --sections 4 6.3   # Update only specific sections
"""

import argparse
import difflib
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# === Paths ===
REPO_ROOT = Path(__file__).resolve().parent.parent
CARGO_TOML = REPO_ROOT / "Cargo.toml"
TECHNOTE_PATH = REPO_ROOT / "web" / "content" / "technote.md"
GENERATED_DIR = REPO_ROOT / "docs" / "generated"
PROMPT_PATH = REPO_ROOT / "docs" / "maintenance" / "update_technote_prompt.md"
CONTEXT_FILE = REPO_ROOT / "repomix-output.xml"

# === Defaults ===
DEFAULT_MODEL = "opus"  # CLI alias for latest opus
MAX_TOKENS = 32768

# === Model ID Mapping ===
# Maps CLI aliases to full model IDs for accurate metadata recording
MODEL_IDS = {
    "opus": "claude-opus-4-5-20251101",
    "sonnet": "claude-sonnet-4-20250514",
}

# Maps CLI aliases to API model names
MODEL_API_NAMES = {
    "opus": "claude-opus-4-5-20251101",
    "sonnet": "claude-sonnet-4-20250514",
}

# Required sections for validation
REQUIRED_SECTIONS = ["# 1.", "# 2.", "# 3.", "# 4.", "# 5.", "# 6.", "# 7.", "# 8."]


# =============================================================================
# Utilities
# =============================================================================

def get_cargo_version() -> str | None:
    """Extract version from Cargo.toml."""
    if not CARGO_TOML.exists():
        return None
    content = CARGO_TOML.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    return match.group(1) if match else None


def get_git_short_commit() -> str | None:
    """Get short commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_commit_date() -> str | None:
    """Get commit date in ISO 8601 format."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cI"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def update_frontmatter_field(content: str, key: str, value: str) -> str:
    """Update or add a field in [extra] section of TOML frontmatter."""
    fm_pattern = r"^(\+\+\+\n)(.*?)(\n\+\+\+\n)"
    fm_match = re.match(fm_pattern, content, re.DOTALL)
    if not fm_match:
        return content

    frontmatter = fm_match.group(2)
    body = content[fm_match.end():]

    field_pattern = rf'^({key}\s*=\s*)"[^"]*"'
    if re.search(field_pattern, frontmatter, re.MULTILINE):
        frontmatter = re.sub(
            field_pattern, rf'\1"{value}"', frontmatter, flags=re.MULTILINE
        )
    else:
        extra_pattern = r"(\[extra\]\n)"
        if re.search(extra_pattern, frontmatter):
            frontmatter = re.sub(
                extra_pattern, rf'\1{key} = "{value}"\n', frontmatter
            )
        else:
            frontmatter = frontmatter.rstrip() + f'\n[extra]\n{key} = "{value}"\n'

    return f"+++\n{frontmatter}\n+++\n{body}"


def inject_metadata(content: str, model_name: str) -> str:
    """Inject version and commit info into frontmatter."""
    version = get_cargo_version()
    commit = get_git_short_commit()
    commit_date = get_git_commit_date()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if version:
        content = update_frontmatter_field(content, "source_version", version)
    if commit:
        content = update_frontmatter_field(content, "source_commit", commit)
    if commit_date:
        content = update_frontmatter_field(content, "source_snapshot", commit_date)

    content = update_frontmatter_field(content, "last_updated", now)

    # Use full model ID instead of alias
    resolved_model_id = MODEL_IDS.get(model_name, model_name)
    content = update_frontmatter_field(content, "generated_by", resolved_model_id)

    return content


def read_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: Path, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def show_diff(old_content: str, new_content: str, filename: str = "technote.md") -> None:
    """Display unified diff between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    )

    diff_text = ''.join(diff)
    if diff_text:
        print("\n" + "=" * 60)
        print(f"DIFF ({filename}):")
        print("=" * 60)
        # Colorize diff output if terminal supports it
        for line in diff_text.splitlines():
            if line.startswith('+') and not line.startswith('+++'):
                print(f"\033[32m{line}\033[0m")  # Green for additions
            elif line.startswith('-') and not line.startswith('---'):
                print(f"\033[31m{line}\033[0m")  # Red for deletions
            elif line.startswith('@@'):
                print(f"\033[36m{line}\033[0m")  # Cyan for line numbers
            else:
                print(line)
    else:
        print(f"\nNo changes in {filename}")


# =============================================================================
# LLM Calls
# =============================================================================

def run_repomix(output_file: Path) -> None:
    """Pack codebase with repomix."""
    print(">> Running repomix...")
    try:
        subprocess.run(
            ["npx", "repomix", "--output", str(output_file)],
            check=True,
            cwd=REPO_ROOT,
        )
    except subprocess.CalledProcessError:
        print("Error: repomix failed.", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: npx not found. Install Node.js.", file=sys.stderr)
        sys.exit(1)

    # Check file size and warn if too large
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f">> Context file size: {size_mb:.1f} MB")
        if size_mb > 5:
            print(f"Warning: Context file is {size_mb:.1f}MB, may exceed token limits",
                  file=sys.stderr)
        if size_mb > 10:
            print("Error: Context file too large (>10MB). Consider filtering with repomix config.",
                  file=sys.stderr)
            sys.exit(1)


def call_llm_cli(model: str, system_prompt: str, user_message: str) -> str:
    """Call Claude Code CLI (subscription-based)."""
    print(f">> Calling Claude CLI ({model})...")

    # Combine prompts for stdin (avoids shell escaping issues with long --system-prompt)
    full_prompt = f"""<system>
{system_prompt}
</system>

{user_message}"""

    try:
        result = subprocess.run(
            [
                "claude",
                "--print",
                "--model", model,
                "--tools", "Read",  # Enable file reading
            ],
            input=full_prompt,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: CLI failed (exit code {e.returncode})", file=sys.stderr)
        if e.stderr:
            print(f"stderr: {e.stderr}", file=sys.stderr)
        if e.stdout:
            print(f"stdout: {e.stdout[:500]}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'claude' not found.", file=sys.stderr)
        print("  npm install -g @anthropic-ai/claude-code", file=sys.stderr)
        print("  claude auth login", file=sys.stderr)
        sys.exit(1)


def call_llm_api(model: str, system_prompt: str, user_message: str) -> str:
    """Call Anthropic API."""
    try:
        import anthropic
    except ImportError:
        print("Error: pip install anthropic", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    # Resolve CLI aliases to full model names for API
    resolved_model = MODEL_API_NAMES.get(model, model)

    print(f">> Calling API ({resolved_model})...")
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=resolved_model,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text


def validate_output(content: str, check_sections: bool = True) -> bool:
    """Validate generated markdown structure."""
    is_valid = True

    # Check frontmatter
    if not re.match(r"^\+\+\+\n.*?\n\+\+\+\n", content, re.DOTALL):
        print("Warning: Missing TOML frontmatter", file=sys.stderr)
        is_valid = False

    # Check minimum length
    if len(content) < 1000:
        print("Warning: Output too short", file=sys.stderr)
        is_valid = False

    # Check required sections
    if check_sections:
        for sec in REQUIRED_SECTIONS:
            if sec not in content:
                print(f"Warning: Missing section {sec}", file=sys.stderr)
                is_valid = False

    # Check for common LLM artifacts
    artifacts = [
        "I'll help you",
        "Here's the updated",
        "```markdown",
        "Let me ",
        "I've updated",
    ]
    for artifact in artifacts:
        if artifact.lower() in content[:500].lower():
            print(f"Warning: Possible LLM preamble detected: '{artifact}'", file=sys.stderr)
            is_valid = False

    return is_valid


def clean_llm_output(content: str) -> str:
    """Clean LLM output to extract pure markdown."""
    content = content.strip()

    # Find frontmatter start and discard preamble
    if "+++" in content:
        start = content.find("+++")
        content = content[start:]

    # Remove wrapping code fences
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    # Remove trailing code fence if present
    content = re.sub(r'\n```\s*$', '', content)

    return content


# =============================================================================
# Section-based Update
# =============================================================================

def extract_sections(content: str) -> dict[str, str]:
    """Extract sections from markdown by heading."""
    sections = {}
    current_section = None
    current_lines = []

    # Skip frontmatter
    fm_match = re.match(r"^\+\+\+\n.*?\n\+\+\+\n", content, re.DOTALL)
    if fm_match:
        sections["_frontmatter"] = fm_match.group(0)
        content = content[fm_match.end():]

    for line in content.split("\n"):
        heading_match = re.match(r"^(#{1,2})\s+(\d+(?:\.\d+)?)\.", line)
        if heading_match:
            # Save previous section
            if current_section:
                sections[current_section] = "\n".join(current_lines)
            current_section = heading_match.group(2)
            current_lines = [line]
        else:
            current_lines.append(line)

    # Save last section
    if current_section:
        sections[current_section] = "\n".join(current_lines)

    return sections


def merge_sections(old_content: str, new_sections: dict[str, str],
                   sections_to_update: list[str]) -> str:
    """Merge specific new sections into old content."""
    old_sections = extract_sections(old_content)

    for sec_num in sections_to_update:
        if sec_num in new_sections:
            old_sections[sec_num] = new_sections[sec_num]
            print(f">> Updated section {sec_num}")
        else:
            print(f"Warning: Section {sec_num} not found in new content", file=sys.stderr)

    # Reconstruct document
    result = old_sections.get("_frontmatter", "")

    # Sort sections by number
    section_nums = sorted(
        [k for k in old_sections.keys() if k != "_frontmatter"],
        key=lambda x: [int(n) for n in x.split(".")]
    )

    for sec_num in section_nums:
        result += old_sections[sec_num] + "\n"

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Update technote.md via LLM")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name (opus, sonnet, or full ID)")
    parser.add_argument("--use-api", action="store_true", help="Use API instead of CLI")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes with diff")
    parser.add_argument("--skip-repomix", action="store_true", help="Skip repomix")
    parser.add_argument("--skip-ja", action="store_true", help="Skip Japanese translation")
    parser.add_argument("--ja-only", action="store_true", help="Only generate Japanese (from existing English)")
    parser.add_argument("--sections", nargs="+", help="Update only specific sections (e.g., --sections 4 6.3)")
    parser.add_argument("--context", type=Path, default=CONTEXT_FILE, help="Context file")
    parser.add_argument("--technote", type=Path, default=TECHNOTE_PATH, help="Source technote.md path")
    args = parser.parse_args()

    if not PROMPT_PATH.exists():
        print(f"Error: {PROMPT_PATH} not found", file=sys.stderr)
        sys.exit(1)
    if not args.technote.exists():
        print(f"Error: {args.technote} not found", file=sys.stderr)
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    # Store original content for diff
    original_content = read_file(args.technote)

    # === English Version ===
    if args.ja_only:
        # Skip English generation, use existing file
        print(">> Using existing English version...")
        updated_content = original_content
    else:
        # 1. Pack codebase
        if not args.skip_repomix:
            run_repomix(args.context)
        if not args.context.exists():
            print(f"Error: {args.context} not found", file=sys.stderr)
            sys.exit(1)

        # 2. Load system prompt
        print(">> Loading system prompt...")
        system_prompt = read_file(PROMPT_PATH)

        # 3. Build prompt
        if args.sections:
            section_list = ", ".join(args.sections)
            user_message = f"""Update ONLY sections {section_list} of technote.md.

Source files:
1. Codebase context: {args.context.resolve()}
2. Current technote to update: {args.technote.resolve()}

Read both files, then output the COMPLETE updated technote.md.
Only modify sections {section_list}, keep all other sections unchanged.

CRITICAL: Your response must start EXACTLY with "+++" (the TOML frontmatter delimiter).
Do NOT include any explanation, thinking, or preamble before the frontmatter.
Output ONLY the raw markdown file content.
"""
        else:
            user_message = f"""Update technote.md with minimal changes.

Source files:
1. Codebase context: {args.context.resolve()}
2. Current technote to update: {args.technote.resolve()}

Read both files, then output the updated technote.md.
Make only necessary changes to match current codebase.

CRITICAL: Your response must start EXACTLY with "+++" (the TOML frontmatter delimiter).
Do NOT include any explanation, thinking, or preamble before the frontmatter.
Output ONLY the raw markdown file content.
"""

        # 4. Call LLM
        try:
            if args.use_api:
                # API mode needs content embedded
                context_xml = read_file(args.context)
                current_technote = read_file(args.technote)
                api_message = f"""
<codebase_context>
{context_xml}
</codebase_context>

<current_technote>
{current_technote}
</current_technote>

Output the fully updated technote.md. ONLY markdown with TOML frontmatter, no filler.
"""
                updated_content = call_llm_api(args.model, system_prompt, api_message)
            else:
                updated_content = call_llm_cli(args.model, system_prompt, user_message)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        # 5. Clean output
        updated_content = clean_llm_output(updated_content)

        # 6. Handle section-specific updates
        if args.sections:
            new_sections = extract_sections(updated_content)
            updated_content = merge_sections(original_content, new_sections, args.sections)

        # 7. Validate
        if not validate_output(updated_content, check_sections=not args.sections):
            print("Warning: Validation failed. Review carefully.", file=sys.stderr)

        # 8. Inject metadata
        updated_content = inject_metadata(updated_content, args.model)

        # 9. Write English version
        if args.dry_run:
            show_diff(original_content, updated_content, "technote.md")
            print(f"\n>> Would write {len(updated_content)} chars to {TECHNOTE_PATH}")
        else:
            generated_path = GENERATED_DIR / f"technote.{timestamp}.md"
            write_file(generated_path, updated_content)
            print(f">> Generated: {generated_path}")

            write_file(TECHNOTE_PATH, updated_content)
            print(f">> Updated: {TECHNOTE_PATH}")

    # === Japanese Version ===
    if not args.skip_ja:
        print(">> Translating to Japanese...")
        ja_prompt = """Translate this technical document to Japanese.

Rules:
- Keep TOML frontmatter as-is (do not translate field names)
- Keep LaTeX equations as-is
- Keep code identifiers (function names, file paths, struct names) as-is
- Translate prose naturally, not literally
- Use appropriate technical Japanese terminology
- Output ONLY the translated markdown, starting with +++

Document to translate:
"""

        try:
            if args.use_api:
                ja_content = call_llm_api(args.model, ja_prompt, updated_content)
            else:
                ja_content = call_llm_cli(args.model, ja_prompt, updated_content)
        except Exception as e:
            print(f"Error translating: {e}", file=sys.stderr)
            sys.exit(1)

        # Clean Japanese output
        ja_content = clean_llm_output(ja_content)

        # Validate Japanese output
        if not validate_output(ja_content, check_sections=False):
            print("Warning: Japanese validation failed.", file=sys.stderr)

        # Write Japanese version
        ja_technote_path = TECHNOTE_PATH.with_suffix(".ja.md")
        ja_original = read_file(ja_technote_path) if ja_technote_path.exists() else ""

        if args.dry_run:
            if ja_original:
                show_diff(ja_original, ja_content, "technote.ja.md")
            else:
                print(f"\n>> Would create new file: {ja_technote_path}")
                print(f">> Content length: {len(ja_content)} chars")
        else:
            ja_generated_path = GENERATED_DIR / f"technote.ja.{timestamp}.md"
            write_file(ja_generated_path, ja_content)
            print(f">> Generated: {ja_generated_path}")

            write_file(ja_technote_path, ja_content)
            print(f">> Updated: {ja_technote_path}")

    if not args.dry_run:
        print(">> Review with: git diff")

    # Print model info
    resolved_id = MODEL_IDS.get(args.model, args.model)
    print(f">> Model used: {resolved_id}")


if __name__ == "__main__":
    main()
