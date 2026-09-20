"""Check shared window arithmetic against frozen actual-audio feature rows.

Future targets are tagged hypothetical solely for a numerical fixture. They are
not issue-time forecasts. This checks neither prediction fidelity nor calibration.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(rows, start, end, issue):
    left = np.array([r["raw"]["start"] for r in rows], dtype=np.int64)
    right = np.array([r["raw"]["end"] for r in rows], dtype=np.int64)
    raw = np.array([r["raw"]["values"] for r in rows], dtype=float)
    energy = np.array([r["energy"] for r in rows], dtype=float)
    weights = np.maximum(0, np.minimum(end, right)-np.maximum(start, left))
    values = np.column_stack((raw[:, 3:7], np.where(np.isnan(raw[:, 2]), np.nan, np.exp2(raw[:, 2]) <= .01), np.sqrt(energy)))
    known = ~np.isnan(values)
    valid = np.where(known, weights[:, None], 0)
    totals = np.sum(np.nan_to_num(values)*valid, axis=0)
    counts = valid.sum(axis=0)
    expected = np.full(7, np.nan)
    observed = np.zeros(7)
    projected = np.zeros(7)
    duration = end-start
    if duration <= 0:
        return expected, observed, projected
    observed[:6] = valid[left < issue].sum(axis=0) / duration
    projected[:6] = valid[left >= issue].sum(axis=0) / duration
    for i in range(6):
        covered = counts[i] >= .9*duration if i < 4 else counts[i]/duration >= .9
        if counts[i] and covered:
            expected[i] = totals[i]/counts[i]
    endpoints = np.flatnonzero((weights > 0) & known[:, 4])
    if len(endpoints) >= 2 and not np.isnan(expected[4]):
        first, last = endpoints[0], endpoints[-1]
        a, b = max(start, int(left[first])), min(end, int(right[first]))
        c, d = max(start, int(left[last])), min(end, int(right[last]))
        if a == start and d == end and c > a:
            expected[6] = (raw[last, 2]-raw[first, 2])*96000/((c-a)+(d-b))
            observed[6], projected[6] = observed[4], projected[4]
    return expected, observed, projected


def verify(root, results):
    manifest = json.loads((root / "manifest.json").read_text())
    names = manifest["source_manifest"]["cases"]
    issue = manifest["source_manifest"]["issue_sample"]
    valid_streams = {(name, p.stem) for name in names for p in (root/name).glob("*.jsonl") if not p.name.startswith("prefix-")}
    seen = set()
    loaded = None
    rows = None
    error = 0.
    supported = observed_count = projected_count = missing = 0
    for line in results.open():
        item = json.loads(line)
        key = item["case"], item["stream"]
        cell, width = item["cell"], item["width_samples"]
        if key not in valid_streams or cell not in range(32) or width not in (12000, 96000, 384000):
            raise ValueError("unregistered window input")
        identity = (*key, cell, width)
        if identity in seen:
            raise ValueError("repeated window input")
        seen.add(identity)
        end = issue + (192000*cell+15)//31
        start = max(0, end-width)
        if (item["start"], item["end"], item["issue"]) != (start, end, issue):
            raise ValueError("shifted window or issue-time clock")
        if key != loaded:
            name, stream = key
            prefix = root/name/f"prefix-bus{stream[-1]}.jsonl"
            rows = [json.loads(line) for path in (prefix, root/name/f"{stream}.jsonl") for line in path.read_text().splitlines()]
            loaded = key
        expected, observed, projected = reference(rows, start, end, issue)
        summary = item["summary"]
        if not np.array_equal(summary["observed_fraction"], observed) or not np.array_equal(summary["projected_fraction"], projected):
            raise ValueError(f"wrong physical provenance fractions: {identity}")
        for i, result in enumerate(summary["values"]):
            origin = "unsupported" if np.isnan(expected[i]) else "projected" if projected[i] > 0 else "observed"
            if result["origin"] != origin or (origin == "unsupported" and "value" in result):
                raise ValueError(f"wrong window feature origin: {identity}/{i}")
            if origin != "unsupported":
                actual = result["value"]
                if not np.isfinite(actual) or not np.isclose(actual, expected[i], rtol=2e-12, atol=1e-13):
                    raise ValueError(f"window numerical mismatch: {identity}/{i}")
                error = max(error, abs(actual-expected[i]))
                supported += 1
                observed_count += origin == "observed"
                projected_count += origin == "projected"
            else:
                missing += 1
    if len(valid_streams) != 792 or len(seen) != 76032:
        raise ValueError("incomplete window denominator")
    return dict(schema="i10-feature-window-arithmetic-verification-v1", windows=len(seen),
                coordinates=supported+missing, observed_coordinates=observed_count,
                projected_coordinates=projected_count, unsupported_coordinates=missing,
                maximum_absolute_error=error, results_sha256=digest(results),
                claim="numerical hypothetical-tag replay of actual targets; not candidate forecasts or ordinal transfer acceptance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    record = json.loads(Path("docs/roadmap/temporal-dcc/i10-action-feature-targets.json").read_text())
    root = Path(record["directory"])
    if args.replay:
        args.output.mkdir(parents=True, exist_ok=False)
    for phase in ("before", "after"):
        if digest(root/"manifest.json") != record["manifest_sha256"] or digest(root/"verification.json") != record["verification_sha256"]:
            raise ValueError("changed target manifest or verification")
        source_hashes = json.loads((root/"verification.json").read_text())["files_sha256"]
        if len(source_hashes) != record["verified_files"] or any(digest(root/p) != h for p, h in source_hashes.items()):
            raise ValueError("changed actual-audio targets")
        if phase == "before":
            results = args.output/"windows.jsonl"
            if args.replay:
                subprocess.run(["cargo", "test", "--lib", "temporal_cognition::feature_projection::window::tests::replay_actual_targets_as_hypothetical_window_inputs", "--", "--ignored", "--exact", "--nocapture"],
                               env=dict(os.environ, CONCHORDAL_I10_FEATURE_DIR=str(root.resolve()),
                                        CONCHORDAL_I10_WINDOW_OUTPUT=str(results.resolve())), check=True)
            summary = verify(root, results)
    summary["input_verification_sha256"] = record["verification_sha256"]
    (args.output/"verification.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))
