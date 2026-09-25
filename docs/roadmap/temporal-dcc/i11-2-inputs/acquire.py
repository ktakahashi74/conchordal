#!/usr/bin/env python3
"""Sequential I11-2 acquisition with frozen inputs, provenance, and an A/A floor gate."""

import argparse
import hashlib
import json
import math
import platform
import re
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

from validate_inputs import HERE, REPO, registered_inputs, sha256

FIELDS = ("population_us", "synthesis_us", "elapsed_us")
STATS = ("median", "p99", "max")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def verified_binary(path, expected):
    path = Path(path).resolve()
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"binary SHA-256 differs: {path}")
    return path


def binary_sources(plan, candidate_manifest):
    original_path = REPO / plan["baseline"]["original_none"]["manifest_path"]
    pre_path = REPO / plan["baseline"]["pre_stage2_off"]["manifest_path"]
    original = json.loads(original_path.read_text())
    pre = json.loads(pre_path.read_text())
    candidate = json.loads(Path(candidate_manifest).read_text())
    if candidate.get("schema") != "conchordal/i11-stage2-candidate-provenance/1":
        raise ValueError("unexpected candidate provenance schema")
    if not candidate.get("source_commit") or not candidate.get("source_files_sha256"):
        raise ValueError("candidate provenance has no source commit and file hashes")
    required_source = {str(path.relative_to(REPO)) for path in (REPO / "src").rglob("*.rs")}
    required_source.update(("Cargo.toml", "Cargo.lock"))
    if not required_source <= set(candidate["source_files_sha256"]):
        raise ValueError("candidate provenance omits source or Cargo files")
    for relative, digest in candidate["source_files_sha256"].items():
        path = (REPO / relative).resolve()
        if not path.is_relative_to(REPO):
            raise ValueError(f"candidate source path escapes repository: {relative}")
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"candidate source differs: {relative}")
    binaries = {}
    for key, data in (("original_none", original), ("pre_stage2_off", pre),
                      ("stage2_candidate", candidate)):
        binaries[key] = {}
        for executable, kind in (("conchordal", "instrument"), ("conchordal-render", "render")):
            entry = data["binaries"][executable]
            if Path(entry["path"]).is_absolute():
                path = Path(entry["path"])
            elif key == "pre_stage2_off":
                path = pre_path.parent / entry["path"]
            elif key == "stage2_candidate":
                path = Path(candidate_manifest).resolve().parent / entry["path"]
            else:
                path = REPO / entry["path"]
            expected = (plan["baseline"][key].get(f"{kind}_binary_sha256")
                        if key != "stage2_candidate" else entry["sha256"])
            if key != "stage2_candidate" and entry["sha256"] != expected:
                raise ValueError(f"baseline manifest binary differs: {key}/{executable}")
            binaries[key][executable] = {"path": str(verified_binary(path, expected)), "sha256": expected}
    return binaries, candidate


def prepare_configs(plan, inputs, output, profile_file):
    if tomllib is None:
        raise ValueError("I11-2 acquisition requires Python 3.11 or newer (tomllib)")
    source = Path(profile_file).resolve()
    expected = plan["source_profile_sha256"]
    if not source.is_file() or sha256(source) != expected:
        raise ValueError("profile is missing or differs from frozen SHA-256")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "action-profiles.bin"
    if destination.exists() and sha256(destination) != expected:
        raise ValueError("stored profile SHA-256 differs")
    if not destination.exists() and source != destination:
        shutil.copy2(source, destination)
    if sha256(destination) != expected:
        raise ValueError("copied profile SHA-256 differs")
    configs = {}
    config_dir = output / "run-configs"
    config_dir.mkdir(exist_ok=True)
    for config in sorted({r["config"] for r in plan["run_order"]}):
        original = inputs[f"config-{config}.toml"]
        text = original.read_text(encoding="utf-8")
        parsed = tomllib.loads(text)
        if parsed["temporal_action_profiles"]["sha256"] != expected:
            raise ValueError(f"config profile SHA-256 differs: {original}")
        lines = text.splitlines(keepends=True)
        section = False
        replaced = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                section = stripped == "[temporal_action_profiles]"
            elif section:
                match = re.fullmatch(r"([ \t]*file[ \t]*=[ \t]*)[^\r\n]*(\r?\n?)", line)
                if match:
                    lines[i] = match[1] + json.dumps(str(destination), ensure_ascii=False) + match[2]
                    replaced += 1
        if replaced != 1:
            raise ValueError(f"profile file key count differs: {original}")
        derived = "".join(lines)
        parsed["temporal_action_profiles"]["file"] = str(destination)
        if tomllib.loads(derived) != parsed:
            raise ValueError(f"derived config changed another value: {original}")
        path = config_dir / original.name
        if path.exists() and path.read_text(encoding="utf-8") != derived:
            raise ValueError(f"existing derived config differs: {path}")
        path.write_text(derived, encoding="utf-8")
        configs[config] = {"original": str(original), "original_sha256": sha256(original),
                           "path": str(path), "sha256": sha256(path)}
    return configs, {"source_path": str(source), "path": str(destination), "sha256": expected}


def prepare_binaries(binaries, output):
    copied = {}
    root = output / "binaries"
    root.mkdir(exist_ok=True)
    for version, entries in binaries.items():
        copied[version] = {}
        for executable, entry in entries.items():
            path = root / f"{version}-{executable}"
            if path.exists():
                if sha256(path) != entry["sha256"]:
                    raise ValueError(f"stored binary differs: {path}")
            else:
                shutil.copy2(entry["path"], path)
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"copied binary differs: {path}")
            copied[version][executable] = {"source_path": entry["path"],
                                           "path": str(path), "sha256": entry["sha256"]}
    return copied


def output_paths(output, row):
    label = f"{row['sequence']:03d}-{row['case']}-{row['config']}-{row['mode']}-{row['rep']}"
    if "arm" in row:
        label += f"-{row['arm']}"
    if "pass_index" in row:
        label += f"-pass{row['pass_index']}"
    artifacts = {"log": output / f"{label}.log"}
    if row["mode"] == "render":
        artifacts.update(wav=output / f"{label}.wav", report=output / f"{label}.jsonl")
    else:
        artifacts["profile"] = output / f"{label}-profile.json"
        if row["mode"] == "report":
            artifacts["report"] = output / f"{label}.jsonl"
    return artifacts


def command(row, inputs, configs, binaries, artifacts):
    executable = "conchordal-render" if row["mode"] == "render" else "conchordal"
    binary = binaries[row["binary"]][executable]["path"]
    script = str(inputs[f"{row['case']}.rhai"])
    config = configs[row["config"]]["path"]
    if row["mode"] == "render":
        return [binary, script, "--config", config, "-o", str(artifacts["wav"]),
                "--report", str(artifacts["report"])]
    cmd = [binary, script, "--config", config, "--nogui", "--play=false",
           "--wait-user-start=false", "--wait-user-exit=false", "--profile", str(artifacts["profile"])]
    if row["mode"] == "report":
        cmd.extend(("--report", str(artifacts["report"])))
    return cmd


def profile_stats(path):
    profile = json.loads(Path(path).read_text())
    hops = profile.get("hops") or []
    if (not hops or profile.get("truncated") is not False or profile.get("dropped_hops") != 0
            or profile["summary"].get("hop_count") != len(hops)):
        raise ValueError(f"incomplete hop profile: {path}")
    result = {}
    for field in FIELDS:
        values = [h[field] for h in hops]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"invalid {field} timing: {path}")
        values.sort()
        result.update({f"{field}_{stat}": values[int(frac * (len(values) - 1))]
                       for stat, frac in (("median", 0.5), ("p99", 0.99), ("max", 1.0))})
    budget = profile["hop_budget_us"]
    over = profile["summary"]["over_budget_hops"]
    if (type(over) is not int or isinstance(budget, bool) or not isinstance(budget, (int, float))
            or not math.isfinite(budget) or budget <= 0
            or over != sum(hop["elapsed_us"] > budget for hop in hops)):
        raise ValueError(f"invalid over-budget count: {path}")
    result["over_budget_hops"] = over
    return result


def aa_floor(plan, output, runs):
    if len(runs) != 600:
        raise ValueError("A/A floor requires all first 600 runs")
    groups = {}
    for row, run in zip(plan["run_order"][96:600], runs[96:600]):
        key = (row["pass_index"], row["case"], row["mode"])
        groups.setdefault(key, {}).setdefault(row["arm"], []).append(
            profile_stats(output / run["artifacts"]["profile"]["path"]))
    if len(groups) != 84:
        raise ValueError("A/A cell count differs")
    differences = []
    fields = [f"{field}_{stat}" for field in FIELDS for stat in STATS] + ["over_budget_hops"]
    floor = {key: 0 for key in fields}
    first_two = {key: 0 for key in fields}
    for (pass_index, case, mode), arms in sorted(groups.items()):
        if set(arms) != {"none-A", "none-B"} or any(len(v) != 3 for v in arms.values()):
            raise ValueError("A/A repetition count differs")
        diff = {}
        for key in fields:
            value = (lambda reps: sum(r[key] for r in reps) if key == "over_budget_hops"
                     else statistics.median(r[key] for r in reps))
            diff[key] = abs(value(arms["none-A"]) - value(arms["none-B"]))
            floor[key] = max(floor[key], diff[key])
            if pass_index < 3:
                first_two[key] = max(first_two[key], diff[key])
        differences.append({"pass_index": pass_index, "case": case, "mode": mode, "absolute_difference": diff})
    return {"schema": "conchordal/i11-stage2-aa-floor/1", "created_at": timestamp(),
            "source_ledger_prefix_sha256": hashlib.sha256(json.dumps(runs, sort_keys=True).encode()).hexdigest(),
            "floor": floor, "pass3_exceedances_against_pass1_2":
            {key: sum(item["pass_index"] == 3 and item["absolute_difference"][key] > first_two[key]
                      for item in differences) for key in fields}, "differences": differences}


def verify_ledger(plan, output, ledger, provenance_sha):
    if ledger["schema"] != "conchordal/i11-stage2-runs/1" or ledger["provenance_sha256"] != provenance_sha:
        raise ValueError("ledger provenance differs")
    runs = ledger["runs"]
    if len(runs) > len(plan["run_order"]):
        raise ValueError("ledger has more runs than plan")
    for row, run in zip(plan["run_order"], runs):
        if run["sequence"] != row["sequence"] or run["registration"] != row or run["exit"] != 0:
            raise ValueError(f"ledger entry differs: {row['sequence']}")
        if run.get("arrival_eligibility") is False:
            raise ValueError(f"run {row['sequence']} had no valid arrival consumption")
        expected = output_paths(output, row)
        if set(run["artifacts"]) != set(expected):
            raise ValueError(f"ledger artifact set differs: {row['sequence']}")
        for kind, artifact in run["artifacts"].items():
            if artifact["path"] != expected[kind].name:
                raise ValueError(f"ledger artifact path differs: {row['sequence']}/{kind}")
            path = output / artifact["path"]
            if (not path.is_file() or not path.stat().st_size
                    or path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]):
                raise ValueError(f"ledger artifact differs: {path}")
    return runs


def ensure_floor(plan, output, runs, ledger, ledger_path):
    floor_path = output / "aa-floor.json"
    expected = aa_floor(plan, output, runs)
    if floor_path.exists():
        actual = json.loads(floor_path.read_text())
        for key in ("schema", "source_ledger_prefix_sha256", "floor",
                    "pass3_exceedances_against_pass1_2", "differences"):
            if actual.get(key) != expected[key]:
                raise ValueError("existing A/A floor differs from measured profiles")
    else:
        write_json(floor_path, expected)
    digest = sha256(floor_path)
    if ledger.get("aa_floor_sha256") not in (None, digest):
        raise ValueError("A/A floor SHA-256 differs")
    ledger["aa_floor_sha256"] = digest
    write_json(ledger_path, ledger)


def acquire(plan, inputs, configs, binaries, output, provenance_sha, through_sequence):
    ledger_path = output / "runs.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        runs = verify_ledger(plan, output, ledger, provenance_sha)
        for row, run in zip(plan["run_order"], runs):
            if run.get("command") != command(row, inputs, configs, binaries, output_paths(output, row)):
                raise ValueError(f"ledger command differs: {row['sequence']}")
        if len(runs) >= 600 and ledger.get("aa_floor_sha256"):
            floor_path = output / "aa-floor.json"
            if not floor_path.exists() or sha256(floor_path) != ledger.get("aa_floor_sha256"):
                raise ValueError("saved A/A floor differs")
        elif len(runs) > 600:
            raise ValueError("comparison runs lack a frozen A/A floor")
    else:
        ledger = {"schema": "conchordal/i11-stage2-runs/1", "provenance_sha256": provenance_sha,
                  "created_at": timestamp(), "runs": []}
        runs = ledger["runs"]
    for row in plan["run_order"][len(runs):through_sequence]:
        if row["sequence"] == 601:
            ensure_floor(plan, output, runs, ledger, ledger_path)
        artifacts = output_paths(output, row)
        if any(path.exists() for path in artifacts.values()):
            raise ValueError(f"unregistered output exists for run {row['sequence']}")
        cmd = command(row, inputs, configs, binaries, artifacts)
        start = time.monotonic()
        with artifacts["log"].open("wb") as log:
            result = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise ValueError(f"run {row['sequence']} exit={result.returncode}; inspect {artifacts['log']}")
        recorded = {}
        for kind, path in artifacts.items():
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"run {row['sequence']} missing or empty {kind}: {path}")
            recorded[kind] = {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
        eligibility = None
        if "profile" in artifacts:
            profile_stats(artifacts["profile"])
        if (row["phase"] == "performance_comparison" and row["config"] == "body-on"
                and row["case"] in plan["new_pulse_cases"]):
            profile = json.loads(artifacts["profile"].read_text())
            count = (profile.get("background") or {}).get("arrival_known_decisions")
            eligibility = isinstance(count, int) and count >= 1
        runs.append({"sequence": row["sequence"], "registration": row, "command": cmd,
                     "exit": result.returncode, "elapsed_seconds": time.monotonic() - start,
                     "finished_at": timestamp(), "artifacts": recorded,
                     "arrival_eligibility": eligibility})
        write_json(ledger_path, ledger)
        print(f"{row['sequence']}/852 {row['phase']} {row['case']} {row['config']} exit=0", flush=True)
        if eligibility is False:
            raise ValueError(f"run {row['sequence']} had no valid arrival consumption")
    if len(runs) == 600:
        ensure_floor(plan, output, runs, ledger, ledger_path)
    return len(runs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profile-file", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--execute", action="store_true", help="start the registered sequential runs")
    parser.add_argument("--through-sequence", type=int, default=852)
    args = parser.parse_args(argv)
    if not 1 <= args.through_sequence <= 852:
        raise ValueError("through-sequence must be within 1..852")
    plan, inputs = registered_inputs()
    candidate_manifest = args.candidate_manifest.resolve()
    if sha256(candidate_manifest) != args.candidate_manifest_sha256:
        raise ValueError("candidate provenance manifest SHA-256 differs")
    binaries, candidate = binary_sources(plan, candidate_manifest)
    output = args.root.resolve()
    configs, profile = prepare_configs(plan, inputs, output, args.profile_file)
    binaries = prepare_binaries(binaries, output)
    provenance = {"schema": "conchordal/i11-stage2-acquisition-provenance/1",
                  "registration_freeze_sha256": sha256(HERE / "freeze.json"),
                  "plan_sha256": sha256(HERE / "plan.json"),
                  "candidate_manifest": str(candidate_manifest),
                  "candidate_manifest_sha256": args.candidate_manifest_sha256,
                  "candidate_source_commit": candidate["source_commit"],
                  "binaries": binaries, "configs": configs, "profile": profile,
                  "machine": {"platform": platform.platform(), "architecture": platform.machine(),
                              "python": platform.python_version()}}
    provenance_path = output / "acquisition-provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if previous != provenance:
            raise ValueError("existing acquisition provenance differs")
    else:
        write_json(provenance_path, provenance)
    if args.execute:
        print(f"completed {acquire(plan, inputs, configs, binaries, output, sha256(provenance_path), args.through_sequence)} runs")
    else:
        print("registration and acquisition provenance verified; no run started")


if __name__ == "__main__":
    main()
