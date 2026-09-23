#!/usr/bin/env python3
"""Run the I11 stage-1 measurement: 12 cases x 3 config variants x 3 modes.

For every case and every config variant (none / body / proxy) this runs

  * conchordal-render once, offline, with the report on (deterministic source
    for SS5.4a and SS5.5);
  * the conchordal instrument with --report, 3 repetitions;
  * the conchordal instrument without --report, 3 repetitions.

Within each repetition the variant order is rotated so that machine drift does
not align with a single variant (SS5.7, the same convention as
target/i10-reduced-resources-20260921/hop-path/acquire.py).

Outputs are named <case>-<variant>-<mode>-<rep>.{log,wav,jsonl} and
<case>-<variant>-<mode>-<rep>-profile.json. runs.json is rewritten after every
run so that an interrupted acquisition still leaves a usable ledger.

The SHA-256 of every input registered by register.py is verified first; a
mismatch aborts before anything is run.

Warning: the report jsonl streams are large (the I10 runs reached ~200 MB for a
single 64-Voice case). Use --case / --variant / --mode to acquire in slices.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import time
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
VARIANTS = ["none", "body", "proxy"]
MODES = ["render", "report", "no-report"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def verify_inputs(plan):
    registered = plan.get("inputs_sha256") or {}
    if not registered:
        raise SystemExit("plan.json carries no inputs_sha256; run register.py first")
    bad = []
    for rel, digest in registered.items():
        p = REPO / rel
        if not p.exists():
            bad.append(f"{rel}: missing")
        elif sha256(p) != digest:
            bad.append(f"{rel}: sha256 mismatch")
    if bad:
        raise SystemExit("registered inputs changed since register.py:\n  " + "\n  ".join(bad))
    return len(registered)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=ROOT, help="measurement directory (default: this script's directory)")
    ap.add_argument("--inputs", type=Path, help="directory holding the registered inputs (default: <plan directory>/inputs)")
    ap.add_argument("--plan", type=Path, help="registration read (default: <root>/plan.json)")
    ap.add_argument("--instrument", type=Path, default=REPO / "target" / "release" / "conchordal", help="instrument binary (default: %(default)s)")
    ap.add_argument("--render", type=Path, default=REPO / "target" / "release" / "conchordal-render", help="offline render binary (default: %(default)s)")
    ap.add_argument("--case", action="append", help="restrict to this case (repeatable; default: every case in plan.json)")
    ap.add_argument("--variant", action="append", choices=VARIANTS, help="restrict to this config variant (repeatable)")
    ap.add_argument("--mode", action="append", choices=MODES, help="restrict to this mode (repeatable)")
    ap.add_argument("--reps", type=int, default=3, help="instrument repetitions (default: %(default)s)")
    ap.add_argument("--runs-name", default="runs.json", help="ledger file name inside the root (default: %(default)s)")
    ap.add_argument("--keep-going", action="store_true", help="record a non-zero exit and continue instead of aborting")
    args = ap.parse_args()

    root = args.root.resolve()
    plan_path = (args.plan or (root / "plan.json")).resolve()
    plan = json.loads(plan_path.read_text())
    n_inputs = verify_inputs(plan)
    print(f"verified {n_inputs} registered inputs", flush=True)

    cases = args.case or plan["cases"]
    variants = args.variant or VARIANTS
    modes = args.mode or MODES
    inputs = (args.inputs or (plan_path.parent / "inputs")).resolve()

    # Copy the binaries into the measurement directory, as the I10 acquisition
    # does, so the artifacts stay readable after target/release is rebuilt.
    binaries = {}
    for name, source in (("instrument", args.instrument), ("render", args.render)):
        dst = root / name
        shutil.copy2(source, dst)
        binaries[name] = {"source": str(source), "sha256": sha256(dst)}
    registered = plan.get("binaries") or {}
    for name in binaries:
        was = (registered.get(name) or {}).get("sha256")
        binaries[name]["registered_sha256"] = was
        binaries[name]["matches_plan"] = (was == binaries[name]["sha256"])
        if was and not binaries[name]["matches_plan"]:
            print(f"note: {name} differs from the binary registered by register.py "
                  f"(expected for a rebuilt stage-1 binary)", flush=True)
    (root / "binaries.json").write_text(json.dumps(
        {"schema": "conchordal/i11-stage1-binaries/1", "generated_at": now_iso(), "binaries": binaries},
        ensure_ascii=False, indent=2) + "\n")
    (root / "source-sha256.json").write_text(json.dumps(
        {"schema": "conchordal/i11-stage1-source-sha256/1", "generated_at": now_iso(),
         "files": {str(p.relative_to(REPO)): sha256(p) for p in sorted((REPO / "src").rglob("*.rs"))}},
        ensure_ascii=False, indent=2) + "\n")

    jobs = []
    for case in cases:
        if "render" in modes:
            for variant in variants:
                jobs.append((case, variant, "render", 0))
        for rep in range(args.reps):
            rotated = variants[rep % len(variants):] + variants[: rep % len(variants)]
            for variant in rotated:
                for mode in modes:
                    if mode != "render":
                        jobs.append((case, variant, mode, rep))

    runs = []
    ledger = root / args.runs_name
    failures = 0
    for case, variant, mode, rep in jobs:
        label = f"{case}-{variant}-{mode}-{rep}"
        script = str(inputs / f"{case}.rhai")
        config = str(inputs / f"config-{variant}.toml")
        if mode == "render":
            cmd = [str(root / "render"), script, "--config", config,
                   "-o", str(root / f"{label}.wav"), "--report", str(root / f"{label}.jsonl")]
        else:
            cmd = [str(root / "instrument"), script, "--config", config, "--nogui",
                   "--play=false", "--wait-user-start=false", "--wait-user-exit=false",
                   "--profile", str(root / f"{label}-profile.json")]
            if mode == "report":
                cmd += ["--report", str(root / f"{label}.jsonl")]
        start = time.monotonic()
        with (root / f"{label}.log").open("w") as log:
            result = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        runs.append(dict(label=label, case=case, variant=variant, mode=mode, rep=rep,
                         command=cmd, exit=result.returncode,
                         elapsed_sec=round(time.monotonic() - start, 3), finished_at=now_iso()))
        ledger.write_text(json.dumps(
            {"schema": "conchordal/i11-stage1-runs/1", "generated_at": now_iso(),
             "binaries": binaries, "runs": runs}, ensure_ascii=False, indent=2) + "\n")
        print(label, result.returncode, f"{runs[-1]['elapsed_sec']:.1f}s", flush=True)
        if result.returncode:
            failures += 1
            if not args.keep_going:
                raise SystemExit(result.returncode)
    print(f"{len(runs)} runs, {failures} failures, ledger {ledger}")


if __name__ == "__main__":
    main()
