#!/usr/bin/env python3
"""Audit all 96 registered I11-2 offline renders and their acquisition provenance."""

import argparse
import json
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTRATION = REPO / "docs/roadmap/temporal-dcc/i11-2-inputs"
sys.path.insert(0, str(REPO / "docs/roadmap/temporal-dcc/i11-inputs"))
sys.path.insert(0, str(REGISTRATION))

import acquire  # noqa: E402
import bit_identity  # noqa: E402
import validate_inputs  # noqa: E402
from compare_i11_candidate_records import (  # noqa: E402
    KEY as CANDIDATE_KEY, compare as compare_candidates, load as load_candidates,
)
from locate_i11_first_divergence import (  # noqa: E402
    CANDIDATE_INPUTS, DECISION_INPUTS, decisions, locate,
)
from verify_i11_stage1 import load as load_stage1, verify as verify_stage1  # noqa: E402
from verify_i11_stage2 import (  # noqa: E402
    T2_CANDIDATE_INPUTS, T2_DECISION_INPUTS, locate_arrival_divergence, same_value,
    verify_report,
)

REALTIME_EXACT = {"processing_us", "max_processing_us", "worker_resources", "delivery_delay_us"}
CONTEXT_STAGE1_ADDITIONS = {
    "footprint_source", "footprint_bins", "footprint_truncated", "reference_cost",
    "selected_cost", "selected_offset", "footprint_requested_at", "footprint_received_at",
}
LEARNING_TYPES = tuple(bit_identity.LEARNING_TYPES)


def digest(path):
    return validate_inputs.sha256(path)


def require_artifact(root, artifact, expected_name):
    if set(artifact) != {"path", "sha256", "bytes"} or artifact["path"] != expected_name:
        raise ValueError(f"artifact declaration differs: {expected_name}")
    path = root / expected_name
    if (not path.is_file() or artifact["bytes"] <= 0
            or path.stat().st_size != artifact["bytes"] or digest(path) != artifact["sha256"]):
        raise ValueError(f"artifact missing or corrupt: {path}")
    return path


def require_file(path, expected, label):
    path = Path(path)
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"{label} missing or SHA-256 differs: {path}")
    return path


def provenance_inputs(root, plan, registered):
    path = root / "acquisition-provenance.json"
    provenance = json.loads(path.read_text())
    if provenance.get("schema") != "conchordal/i11-stage2-acquisition-provenance/1":
        raise ValueError("acquisition provenance schema differs")
    if provenance.get("registration_freeze_sha256") != digest(REGISTRATION / "freeze.json"):
        raise ValueError("registration freeze SHA-256 differs")
    if provenance.get("plan_sha256") != digest(REGISTRATION / "plan.json"):
        raise ValueError("registration plan SHA-256 differs")
    candidate_path = require_file(provenance["candidate_manifest"],
                                  provenance["candidate_manifest_sha256"], "candidate manifest")
    candidate = json.loads(candidate_path.read_text())
    if (candidate.get("schema") != "conchordal/i11-stage2-candidate-provenance/1"
            or candidate.get("source_commit") != provenance.get("candidate_source_commit")
            or not candidate.get("source_files_sha256")):
        raise ValueError("candidate source provenance differs")
    profile = provenance["profile"]
    if profile["sha256"] != plan["source_profile_sha256"]:
        raise ValueError("source profile SHA-256 differs from plan")
    require_file(profile["path"], profile["sha256"], "copied action profile")
    if Path(profile["path"]).resolve() != root / "action-profiles.bin":
        raise ValueError("profile copy is outside acquisition root")

    configs = provenance["configs"]
    required_configs = {row["config"] for row in plan["run_order"][:96]}
    if not required_configs <= configs.keys():
        raise ValueError("acquisition provenance lacks a registered config")
    for name, entry in configs.items():
        original = registered.get(f"config-{name}.toml")
        if original is None or Path(entry["original"]).resolve() != original.resolve():
            raise ValueError(f"unregistered config: {name}")
        require_file(original, entry["original_sha256"], f"original config {name}")
        if entry["original_sha256"] != plan["input_sha256"][f"inputs/config-{name}.toml"]:
            raise ValueError(f"frozen config SHA-256 differs: {name}")
        derived = require_file(entry["path"], entry["sha256"], f"derived config {name}")
        if derived.resolve() != root / "run-configs" / original.name:
            raise ValueError(f"derived config is outside acquisition root: {name}")
        old = tomllib.loads(original.read_text())
        new = tomllib.loads(derived.read_text())
        old["temporal_action_profiles"]["file"] = profile["path"]
        if new != old or new["temporal_action_profiles"]["sha256"] != profile["sha256"]:
            raise ValueError(f"derived config changed another setting: {name}")

    binaries = provenance["binaries"]
    if set(binaries) != {"original_none", "pre_stage2_off", "stage2_candidate"}:
        raise ValueError("binary provenance set differs")
    for version, pair in binaries.items():
        if set(pair) != {"conchordal", "conchordal-render"}:
            raise ValueError(f"binary pair incomplete: {version}")
        for executable, entry in pair.items():
            expected = (candidate["binaries"][executable]["sha256"] if version == "stage2_candidate"
                        else plan["baseline"][version][
                            "render_binary_sha256" if executable == "conchordal-render"
                            else "instrument_binary_sha256"])
            if entry["sha256"] != expected:
                raise ValueError(f"binary digest differs from provenance: {version}/{executable}")
            copied = require_file(entry["path"], expected, f"binary {version}/{executable}")
            if copied.resolve() != root / "binaries" / f"{version}-{executable}":
                raise ValueError(f"binary copy is outside acquisition root: {version}/{executable}")
    return provenance, configs, binaries


def offline_runs(root, plan, registered, configs, binaries):
    ledger_path = root / "runs.json"
    ledger = json.loads(ledger_path.read_text())
    if (ledger.get("schema") != "conchordal/i11-stage2-runs/1"
            or ledger.get("provenance_sha256") != digest(root / "acquisition-provenance.json")):
        raise ValueError("run ledger provenance differs")
    runs = ledger.get("runs")
    if not isinstance(runs, list) or len(runs) < 96:
        raise ValueError(f"offline acquisition incomplete: {0 if not isinstance(runs, list) else len(runs)}/96")
    output = {}
    for row, run in zip(plan["run_order"][:96], runs[:96]):
        sequence = row["sequence"]
        if (row["mode"] != "render" or row["rep"] != 0 or run.get("sequence") != sequence
                or run.get("registration") != row or run.get("exit") != 0):
            raise ValueError(f"run registration or exit differs: {sequence}")
        paths = acquire.output_paths(root, row)
        if set(run.get("artifacts", {})) != set(paths):
            raise ValueError(f"run artifact set differs: {sequence}")
        for kind, path in paths.items():
            require_artifact(root, run["artifacts"][kind], path.name)
        expected_command = acquire.command(row, registered, configs, binaries, paths)
        if run.get("command") != expected_command:
            raise ValueError(f"run command differs: {sequence}")
        key = (row["binary"], row["case"], row["config"])
        if key in output:
            raise ValueError(f"duplicate offline run: {key}")
        output[key] = {"sequence": sequence, "row": row,
                       "wav": paths["wav"], "report": paths["report"]}
    if len(output) != 96:
        raise ValueError("offline run matrix has duplicate or missing entries")
    return output


def stripped(value, removed, type_name, path=""):
    if isinstance(value, dict):
        result = {}
        for key, member in value.items():
            field = f"{path}.{key}" if path else key
            if (key in REALTIME_EXACT or key.endswith(("_us", "_ns"))
                    or type_name == "participation_context" and key in CONTEXT_STAGE1_ADDITIONS):
                removed.add(field)
            else:
                result[key] = stripped(member, removed, type_name, field)
        return result
    if isinstance(value, list):
        return [stripped(member, removed, type_name, f"{path}[]") for member in value]
    return value


def record_stream(path, kinds, allow_context_legacy=False):
    records = {kind: [] for kind in kinds}
    removed = set()
    candidate_count = 0
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            kind = row.get("type")
            if kind == "body_candidate_energy":
                candidate_count += 1
            if kind in records:
                label = kind if allow_context_legacy else ""
                value = stripped(row, removed, label)
                records[kind].append(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return records, candidate_count, removed


def exact_report_pair(left, right, kinds, allow_context_legacy=False):
    a, a_candidates, a_removed = record_stream(left, kinds, allow_context_legacy)
    b, b_candidates, b_removed = record_stream(right, kinds, allow_context_legacy)
    types = {kind: {"left": len(a[kind]), "right": len(b[kind]), "equal": a[kind] == b[kind]}
             for kind in kinds}
    return {"passed": all(item["equal"] for item in types.values()), "types": types,
            "excluded_paths": sorted(a_removed | b_removed),
            "candidate_counts": {"left": a_candidates, "right": b_candidates}}


def compare_candidates_bits(left, right):
    result = compare_candidates(left, right)
    a, b = load_candidates(left), load_candidates(right)
    differing = [dict(zip(CANDIDATE_KEY, key)) for key in sorted(a.keys() & b.keys(), key=str)
                 if not same_value(a[key], b[key])]
    return {**result, "bit_differing": len(differing), "bit_examples": differing[:5]}


def footprint_records_strict(path):
    records = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("type") != "body_footprint":
                continue
            key = (json.dumps(row["identity"], sort_keys=True), row["received_at"])
            if key in records:
                raise ValueError(f"duplicate body footprint key: {path}: {key}")
            records[key] = row
    return records


def verify_stage1_report(path, arrival_on=False):
    rows, footprints, contexts = load_stage1(path)
    if len(footprints) != len(footprint_records_strict(path)):
        raise ValueError(f"body footprint record indexing differs: {path}")
    findings, intervention = verify_stage1(rows, footprints, contexts)
    replaced = {"5.1 cost", "5.1 skip rule"} if arrival_on else set()
    failures = {name: value for name, value in findings.summary().items()
                if value["failed"] and name not in replaced}
    result = {"decisions": len(rows), "failed_checks": failures,
              "checks_replaced_by_stage2": sorted(replaced)}
    if not arrival_on:
        result["intervention"] = intervention
    return result


def referenced_footprints_match(left, right, first_index):
    left_rows, _, _ = load_stage1(left)
    right_rows, _, _ = load_stage1(right)
    left_records = footprint_records_strict(left)
    right_records = footprint_records_strict(right)
    left_rows.sort(key=lambda row: (row["now"], row["voice_id"]))
    right_rows.sort(key=lambda row: (row["now"], row["voice_id"]))
    limit = min(len(left_rows), len(right_rows)) if first_index is None else first_index + 1
    compared = 0
    for a, b in zip(left_rows[:limit], right_rows[:limit]):
        for row in (a, b):
            identity, received = row["footprint_identity"], row["footprint_received_at"]
            if identity is None:
                continue
            key = (json.dumps(identity, sort_keys=True), received)
            if key not in left_records or key not in right_records:
                return {"passed": False, "reason": "referenced footprint missing", "key": key}
            left_value = {k: v for k, v in left_records[key].items() if k != "computed_at"}
            right_value = {k: v for k, v in right_records[key].items() if k != "computed_at"}
            if not same_value(left_value, right_value):
                return {"passed": False, "reason": "referenced footprint differs", "key": key}
            compared += 1
    return {"passed": True, "referenced": compared}


def strict_prefix_inputs(left, right, first_index, decision_keys, candidate_keys):
    a, b = decisions(left), decisions(right)
    limit = min(len(a), len(b)) if first_index is None else first_index + 1
    for index, (first, second) in enumerate(zip(a[:limit], b[:limit])):
        if (first["now"], first["voice_id"]) != (second["now"], second["voice_id"]):
            return {"passed": False, "index": index, "field": "decision_identity"}
        for field in decision_keys:
            if not same_value(first[field], second[field]):
                return {"passed": False, "index": index, "field": field}
        if len(first["candidates"]) != 23 or len(second["candidates"]) != 23:
            return {"passed": False, "index": index, "field": "candidate_count"}
        for slot, (ca, cb) in enumerate(zip(first["candidates"], second["candidates"])):
            if (ca is None) != (cb is None):
                return {"passed": False, "index": index, "field": f"candidates[{slot}].present"}
            if ca is None:
                continue
            for field in ("offset",) + tuple(candidate_keys):
                if not same_value(ca[field], cb[field]):
                    return {"passed": False, "index": index,
                            "field": f"candidates[{slot}].{field}"}
    if first_index is None and len(a) != len(b):
        return {"passed": False, "index": limit, "field": "decision_count"}
    return {"passed": True, "compared": limit}


def evaluate(root):
    root = Path(root).resolve()
    plan, registered = validate_inputs.registered_inputs()
    provenance, configs, binaries = provenance_inputs(root, plan, registered)
    runs = offline_runs(root, plan, registered, configs, binaries)
    old_cases = tuple(plan["existing_stage1_cases"])
    new_cases = tuple(plan["new_pulse_cases"])
    if len(old_cases) != 12 or len(new_cases) != 6:
        raise ValueError("registered case populations differ")
    get = lambda binary, case, config: runs[(binary, case, config)]
    failures = []
    summary = {"schema": "conchordal/i11-stage2-offline-evaluation/1",
               "acquisition_root": str(root), "plan_sha256": digest(REGISTRATION / "plan.json"),
               "provenance_sha256": digest(root / "acquisition-provenance.json"),
               "ledger_sha256": digest(root / "runs.json"), "offline_runs": 96,
               "excluded_realtime_fields": sorted(REALTIME_EXACT),
               "excluded_realtime_suffixes": ["_us", "_ns"],
               "excluded_legacy_context_fields": sorted(CONTEXT_STAGE1_ADDITIONS)}

    report_checks = []
    for run in runs.values():
        if run["row"]["binary"] == "stage2_candidate" and run["row"]["config"] == "body-on":
            result = {"arrival": verify_report(run["report"]),
                      "body": verify_stage1_report(run["report"], arrival_on=True)}
            bad = (result["arrival"]["failed_decisions"] > 0
                   or bool(result["body"]["failed_checks"])
                   or result["arrival"]["decisions"] != result["body"]["decisions"])
            count = result["arrival"]["decisions"]
            kind = "stage2"
        elif run["row"]["config"] != "none":
            result = verify_stage1_report(run["report"])
            bad = bool(result["failed_checks"])
            count = result["decisions"]
            kind = "stage1"
        else:
            rows = decisions(run["report"])
            result = {"decisions": len(rows), "failed_checks": {}}
            bad = bool(rows)
            count = len(rows)
            kind = "none_report_parse"
        allow_empty = (run["row"]["case"] in {"sine-hold-4", "sine-hold-16", "sine-hold-64"}
                       and run["row"]["config"] != "none")
        if run["row"]["config"] != "none" and count == 0 and not allow_empty:
            bad = True
        result["zero_decisions_allowed"] = allow_empty
        if bad:
            failures.append(f"report verifier failed: {run['sequence']} {run['row']['case']}")
        report_checks.append({"sequence": run["sequence"], "verifier": kind, **result})
    summary["report_verifiers"] = report_checks

    original = []
    candidate_checks = []
    for case in old_cases:
        a, b = get("original_none", case, "none"), get("stage2_candidate", case, "none")
        comparison = exact_report_pair(a["report"], b["report"], LEARNING_TYPES, True)
        same_wav = digest(a["wav"]) == digest(b["wav"])
        result = compare_candidates_bits(a["report"], b["report"])
        original.append({"case": case, "wav_equal": same_wav, "learning": comparison})
        candidate_checks.append({"case": case, "config": "none", **result})
        if not same_wav or not comparison["passed"]:
            failures.append(f"5.4a original None differs: {case}")
    summary["original_none_bit_identity"] = original

    off_regression = []
    for case in old_cases:
        for config in ("body-off", "proxy-off"):
            a, b = get("pre_stage2_off", case, config), get("stage2_candidate", case, config)
            comparison = exact_report_pair(
                a["report"], b["report"], LEARNING_TYPES + ("participation_decision",))
            same_wav = digest(a["wav"]) == digest(b["wav"])
            result = compare_candidates_bits(a["report"], b["report"])
            off_regression.append({"case": case, "config": config, "wav_equal": same_wav,
                                   "decision_and_learning": comparison})
            candidate_checks.append({"case": case, "config": config, **result})
            if not same_wav or not comparison["passed"]:
                failures.append(f"pre-stage2 off regression differs: {case}/{config}")
    summary["pre_stage2_off_regression"] = off_regression
    summary["candidate_records"] = {
        "pairs": candidate_checks,
        "matched": sum(row["matched"] for row in candidate_checks),
        "differing": sum(row["differing"] for row in candidate_checks),
        "bit_differing": sum(row["bit_differing"] for row in candidate_checks),
        "base_only": sum(row["base_only"] for row in candidate_checks),
        "new_only": sum(row["new_only"] for row in candidate_checks),
        "frozen_packet_replay": "not_performed",
    }
    for row in candidate_checks:
        if row["matched"] == 0 or row["differing"] or row["bit_differing"]:
            failures.append(f"5.4b candidate records unmatched or differ: {row['case']}/{row['config']}")

    t1 = []
    for case in old_cases:
        body, proxy = get("stage2_candidate", case, "body-off"), get("stage2_candidate", case, "proxy-off")
        result = locate(body["report"], proxy["report"])
        wav_differs = digest(body["wav"]) != digest(proxy["wav"])
        first_index = result.get("index") if result["status"] == "diverged" else None
        prefix = referenced_footprints_match(body["report"], proxy["report"],
                                             first_index)
        strict = strict_prefix_inputs(body["report"], proxy["report"], first_index,
                                      DECISION_INPUTS, CANDIDATE_INPUTS)
        required_effect = case.startswith(("harmonic-", "modal-"))
        valid = (result["status"] in {"diverged", "no_divergence"}
                 and result.get("first_prior_input_difference") is None
                 and prefix["passed"] and strict["passed"]
                 and (result["status"] != "diverged" or result["sources_valid"]
                      and result["powers_differ"] and not result["other_differing_inputs"])
                 and (result["status"] != "no_divergence" or not wav_differs)
                 and (not required_effect or result["status"] == "diverged" and wav_differs))
        t1.append({"case": case, "wav_differs": wav_differs, "prefix_footprints": prefix,
                   "bit_strict_prefix": strict,
                   "required_effect": required_effect, "passed": valid, **result})
        if not valid:
            failures.append(f"T1 first-divergence contract failed: {case}")
    summary["T1_body_vs_proxy"] = t1

    t2 = []
    body_effect = {body: False for body in ("sine", "harmonic", "modal")}
    for case in new_cases:
        off, on = get("stage2_candidate", case, "body-off"), get("stage2_candidate", case, "body-on")
        result = locate_arrival_divergence(off["report"], on["report"])
        wav_differs = digest(off["wav"]) != digest(on["wav"])
        first_index = result.get("index") if result["status"] == "diverged" else None
        prefix = referenced_footprints_match(off["report"], on["report"], first_index)
        strict = strict_prefix_inputs(off["report"], on["report"], first_index,
                                      T2_DECISION_INPUTS, T2_CANDIDATE_INPUTS)
        on_decisions = decisions(on["report"])
        known = sum(bool(row.get("arrival", {}).get("known_decision")) for row in on_decisions)
        valid_prefix = (result["status"] in {"diverged", "no_divergence"}
                        and result.get("first_prior_input_difference") is None
                        and prefix["passed"] and strict["passed"]
                        and (result["status"] != "no_divergence" or not wav_differs)
                        and (result["status"] != "diverged" or not result["nonarrival_differences"]
                             and not result["on_arrival_failures"] and result["arrival_used"]))
        effect = result["status"] == "diverged" and wav_differs and valid_prefix
        body = case.split("-", 1)[0]
        body_effect[body] |= effect
        t2.append({"case": case, "wav_differs": wav_differs, "known_arrival_decisions": known,
                   "prefix_footprints": prefix, "bit_strict_prefix": strict,
                   "prefix_passed": valid_prefix, "body_effect": effect, **result})
        if not valid_prefix or known == 0:
            failures.append(f"T2 prefix or arrival use failed: {case}")
    summary["T2_body_arrival_intervention"] = {"cases": t2, "body_effect": body_effect}
    for body, effect in body_effect.items():
        if not effect:
            failures.append(f"T2 no selection/skip and WAV effect for body: {body}")
    summary["failures"] = failures
    summary["passed"] = not failures
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="acquire.py --root directory")
    parser.add_argument("--output", type=Path, help="JSON result (default: ROOT/offline-evaluation.json)")
    args = parser.parse_args(argv)
    output = args.output or args.root / "offline-evaluation.json"
    try:
        result = evaluate(args.root)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        result = {"schema": "conchordal/i11-stage2-offline-evaluation/1", "passed": False,
                  "failures": [f"{type(error).__name__}: {error}"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "failures": result["failures"],
                      "output": str(output)}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
