#!/usr/bin/env python3
"""Freeze the inputs of the I11 stage-1 measurement into plan.json.

Copies the 12 I10 scenario scripts and builds the three config variants
(none / body / proxy) under <root>/inputs/, then records the SHA-256 of every
input, the baseline commit and the SHA-256 of the two binaries into plan.json.

Registration: docs/roadmap/temporal-dcc/i11-onset-comparison.md SS4.10, SS5.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def repo_root(start=None):
    """Walk up to the directory holding Cargo.toml, so the script runs from any location."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "Cargo.toml").is_file():
            return candidate
    raise SystemExit("cannot locate the repository root (no Cargo.toml above this script)")

REPO = repo_root()
I10 = REPO / "target" / "i10-reduced-resources-20260921"

# SS4.10: the Some form of the section. arrival / arrival_weight stay at their
# registered defaults and are not written, because stage 1 does not deliver the
# arrival tables (SS4.5, SS7).
VARIANT_SUFFIX = {
    "none": "",
    "body": '\n[temporal_onset_comparison]\nfootprint = "body"\n',
    "proxy": '\n[temporal_onset_comparison]\nfootprint = "proxy"\n',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=ROOT, help="measurement directory (default: this script's directory)")
    ap.add_argument("--i10-dir", type=Path, default=I10, help="I10 measurement directory holding inputs/ (default: %(default)s)")
    ap.add_argument("--baseline", metavar="SHA", help="baseline commit recorded as plan.baseline_commit (SS5.4a)")
    ap.add_argument(
        "--binaries",
        nargs=2,
        metavar=("INSTRUMENT", "RENDER"),
        help="paths to the conchordal and conchordal-render binaries whose SHA-256 is recorded",
    )
    ap.add_argument("--force", action="store_true", help="overwrite an already filled plan.json")
    args = ap.parse_args()

    root = args.root.resolve()
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text())
    if plan.get("inputs_sha256") and not args.force:
        raise SystemExit(f"{plan_path} is already filled; pass --force to overwrite")

    src_inputs = (args.i10_dir / "inputs").resolve()
    dst_inputs = root / "inputs"
    dst_inputs.mkdir(parents=True, exist_ok=True)

    for case in plan["cases"]:
        shutil.copy2(src_inputs / f"{case}.rhai", dst_inputs / f"{case}.rhai")

    base_config = (src_inputs / "config.toml").read_text()
    if "[temporal_onset_comparison]" in base_config:
        raise SystemExit("the I10 config already carries [temporal_onset_comparison]")
    if not base_config.endswith("\n"):
        base_config += "\n"
    for variant, suffix in VARIANT_SUFFIX.items():
        (dst_inputs / f"config-{variant}.toml").write_text(base_config + suffix)

    files = sorted(dst_inputs.iterdir())
    plan["inputs_sha256"] = {str(p.relative_to(REPO)): sha256(p) for p in files}
    plan["inputs_source"] = {
        "scenario_scripts": f"byte copies of {src_inputs.relative_to(REPO)}/*.rhai",
        "config": f"{(src_inputs / 'config.toml').relative_to(REPO)} plus the SS4.10 section per variant",
        "config_sha256_upstream": sha256(src_inputs / "config.toml"),
    }

    if args.baseline:
        rev = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", args.baseline],
            capture_output=True, text=True, stdin=subprocess.DEVNULL,
        )
        plan["baseline_commit"] = {
            "given": args.baseline,
            "resolved": rev.stdout.strip() or None,
            "resolved_ok": rev.returncode == 0,
        }

    if args.binaries:
        entry = {}
        for name, given in zip(("instrument", "render"), args.binaries):
            p = Path(given)
            if p.exists():
                entry[name] = {"path": str(p), "sha256": sha256(p), "size": p.stat().st_size,
                               "mtime": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).astimezone().isoformat(timespec="seconds")}
            else:
                entry[name] = {"path": str(p), "sha256": None, "missing": True}
        entry["note"] = (
            "SHA-256 of the binaries present when the inputs were frozen. These may predate the stage-1 "
            "implementation; acquire.py re-records the binaries it actually ran into binaries.json and flags "
            "any drift from this entry."
        )
        plan["binaries"] = entry

    plan["generated_at"] = now_iso()
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")

    print(f"inputs: {len(files)} files under {dst_inputs}")
    for p in files:
        print(f"  {p.name} {sha256(p)}")
    print(f"baseline_commit: {json.dumps(plan.get('baseline_commit'), ensure_ascii=False)}")
    if plan.get("binaries"):
        for name in ("instrument", "render"):
            print(f"  {name}: {plan['binaries'][name].get('sha256')}")
    print(f"wrote {plan_path}")


if __name__ == "__main__":
    main()
