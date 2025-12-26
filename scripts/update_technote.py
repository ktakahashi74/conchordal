#!/usr/bin/env python3
"""
scripts/update_technote.py

Auto-update technote.md based on current codebase using Claude.

Output:
  - docs/generated/technote.{timestamp}.md  (versioned history)
  - web/content/technote.md                 (production copy)

Usage:
  python scripts/update_technote.py                    # Default (Claude CLI, opus)
  python scripts/update_technote.py --model sonnet     # Use different model
  python scripts/update_technote.py --use-api          # Use Anthropic API
  python scripts/update_technote.py --dry-run          # Preview only
  python scripts/update_technote.py --skip-repomix     # Skip repomix step
  python scripts/update_technote.py --technote PATH    # Specify source file
"""

import argparse
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
MAX_TOKENS = 16384


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
    content = update_frontmatter_field(content, "generated_by", model_name)

    return content


def read_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: Path, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# =============================================================================
# LLM Calls
# =============================================================================

def run_repomix() -> None:
    """Pack codebase with repomix."""
    print(">> Running repomix...")
    try:
        subprocess.run(["npx", "repomix"], check=True, cwd=REPO_ROOT)
    except subprocess.CalledProcessError:
        print("Error: repomix failed.", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: npx not found. Install Node.js.", file=sys.stderr)
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
    model_aliases = {
        "opus": "claude-opus-4-20250514",
        "sonnet": "claude-sonnet-4-20250514",
    }
    resolved_model = model_aliases.get(model, model)

    print(f">> Calling API ({resolved_model})...")
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=resolved_model,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text


def validate_output(content: str) -> bool:
    """Basic validation of generated markdown."""
    if not re.match(r"^\+\+\+\n.*?\n\+\+\+\n", content, re.DOTALL):
        print("Warning: Missing TOML frontmatter", file=sys.stderr)
        return False
    if len(content) < 1000:
        print("Warning: Output too short", file=sys.stderr)
        return False
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Update technote.md via LLM")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--use-api", action="store_true", help="Use API instead of CLI")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--skip-repomix", action="store_true", help="Skip repomix")
    parser.add_argument("--context", type=Path, default=CONTEXT_FILE, help="Context file")
    parser.add_argument("--technote", type=Path, default=TECHNOTE_PATH, help="Source technote.md path")
    args = parser.parse_args()

    if not PROMPT_PATH.exists():
        print(f"Error: {PROMPT_PATH} not found", file=sys.stderr)
        sys.exit(1)
    if not args.technote.exists():
        print(f"Error: {args.technote} not found", file=sys.stderr)
        sys.exit(1)

    # 1. Pack codebase
    if not args.skip_repomix:
        run_repomix()
    if not args.context.exists():
        print(f"Error: {args.context} not found", file=sys.stderr)
        sys.exit(1)

    # 2. Load system prompt
    print(">> Loading system prompt...")
    system_prompt = read_file(PROMPT_PATH)

    # 3. Build prompt (reference files by path, let Claude read them)
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

    # 5. Clean output - strip everything before frontmatter
    updated_content = updated_content.strip()
    
    # Find frontmatter start and discard preamble
    if "+++" in updated_content:
        start = updated_content.find("+++")
        updated_content = updated_content[start:]
    
    # Remove trailing code fences
    if updated_content.startswith("```"):
        lines = updated_content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        updated_content = "\n".join(lines)

    # 6. Validate
    if not validate_output(updated_content):
        print("Warning: Validation failed. Review carefully.", file=sys.stderr)

    # 7. Inject metadata
    updated_content = inject_metadata(updated_content, args.model)

    # 8. Write
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN OUTPUT:")
        print("=" * 60)
        print(updated_content[:2000])
        if len(updated_content) > 2000:
            print(f"\n... ({len(updated_content)} chars)")
    else:
        # Create timestamped version in generated dir
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generated_path = GENERATED_DIR / f"technote.{timestamp}.md"
        write_file(generated_path, updated_content)
        print(f">> Generated: {generated_path}")

        # Copy to production location
        write_file(TECHNOTE_PATH, updated_content)
        print(f">> Updated: {TECHNOTE_PATH}")
        print(">> Review with: git diff")


if __name__ == "__main__":
    main()
