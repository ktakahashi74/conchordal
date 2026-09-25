#!/usr/bin/env python3
"""Validate the frozen I11-2 registration before using any input bytes."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registered_inputs(registration=HERE, repo=REPO, require_baselines=True):
    registration = Path(registration).resolve()
    repo = Path(repo).resolve()
    freeze = json.loads((registration / "freeze.json").read_text())
    plan = json.loads((registration / "plan.json").read_text())
    if freeze.get("schema") != "conchordal/i11-stage2-preregistration-freeze/1":
        raise ValueError("unexpected freeze schema")
    if plan.get("schema") != "conchordal/i11-stage2-registration/1":
        raise ValueError("unexpected plan schema")
    for relative, expected in freeze["files_sha256"].items():
        path = repo / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"frozen file differs: {relative}")
    if len(plan["run_order"]) != 852:
        raise ValueError("frozen run count differs")
    for sequence, row in enumerate(plan["run_order"], 1):
        if row["sequence"] != sequence:
            raise ValueError(f"run order differs at sequence {sequence}")
        phase = row["phase"]
        expected_phase = ("offline" if sequence <= 96 else
                          "performance_aa" if sequence <= 600 else "performance_comparison")
        if not (phase.startswith("offline_") if expected_phase == "offline" else phase == expected_phase):
            raise ValueError(f"run phase differs at sequence {sequence}")
    inputs = {}
    for relative, expected in plan["input_sha256"].items():
        path = registration / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"registered input differs: {relative}")
        inputs[path.name] = path
    for case, entry in plan["existing_stage1_inputs"].items():
        path = repo / entry["registered_path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"stage-1 input differs: {case}")
        inputs[path.name] = path
    for row in plan["run_order"]:
        if f"{row['case']}.rhai" not in inputs or f"config-{row['config']}.toml" not in inputs:
            raise ValueError(f"unregistered run input: {row['sequence']}")
    if require_baselines:
        for baseline in plan["baseline"].values():
            if baseline.get("manifest_path"):
                path = repo / baseline["manifest_path"]
                if not path.is_file() or sha256(path) != baseline["manifest_sha256"]:
                    raise ValueError(f"baseline manifest differs: {path}")
    return plan, inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=HERE)
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args(argv)
    plan, inputs = registered_inputs(args.registration, args.repo)
    print(json.dumps({"runs": len(plan["run_order"]), "inputs": len(inputs), "frozen": True}))


if __name__ == "__main__":
    main()
