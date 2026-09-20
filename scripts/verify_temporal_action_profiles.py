"""Verify all seven-class offline PCM branches without trusting reported energy.

The renderer executable checks cloned versus independently replayed owned state.
This verifier separately checks the saved matrix, routing, abstention tails,
class equivalences, timed onsets, and every reported physical-hop energy coordinate.
Version 2 also preserves the owned PCM prefix on the original analysis grid.
"""

import argparse
import hashlib
import itertools
import json
import math
import struct
from pathlib import Path


def read_pcm(path, samples):
    pcm = path.read_bytes()
    if len(pcm) != samples * 4:
        raise ValueError("incomplete PCM support")
    values = struct.unpack(f"<{samples}f", pcm)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("nonfinite PCM")
    return pcm, values


def check_energy(reported, samples, paths, hop, label):
    if len(reported) != 2:
        raise ValueError("missing bus energy")
    count, largest = 0, 0.
    for bus, values in enumerate(reported):
        if len(values) != len(samples) // hop or len(samples) % hop:
            raise ValueError("wrong energy grid")
        for i, actual in enumerate(values):
            expected = math.fsum(x*x for x in samples[i*hop:(i+1)*hop]) / hop if paths[bus] else 0.
            if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=2e-14, abs_tol=1e-20):
                raise ValueError(f"energy mismatch: {label}/{bus}/{i}")
            count += 1
            largest = max(largest, abs(actual-expected))
    return count, largest


def verify(root):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] not in ("i10-actual-action-profiles-v1", "i10-actual-action-profiles-v2"):
        raise ValueError("unsupported profile schema")
    has_prefix = manifest["schema"].endswith("v2")
    rate, hop = manifest["sample_rate"], manifest["hop_samples"]
    issue, horizon = manifest["issue_sample"], manifest["horizon_samples"]
    period = manifest["intrinsic_period_samples"]
    expected_issue = ((14400 + hop - 1) // hop) * hop if hop in (480, 512) else None
    if ((rate, horizon, period) != (48000, 192000, 9600)
            or hop not in ((480, 512) if has_prefix else (480,)) or issue != expected_issue):
        raise ValueError("acquisition differs from the registered matrix")
    if has_prefix and (manifest["requested_issue_sample"] != 14400 or manifest["prefix_samples"] != issue
            or manifest["prefix_support"] != [0, issue]
            or manifest["future_support"] != [issue, issue + horizon]):
        raise ValueError("prefix/future physical support mismatch")
    energy_key = "energy_per_hop" if has_prefix else "energy_10ms"
    if manifest["pending_intrinsic_opportunity_sample"] != issue:
        raise ValueError("skip requires the registered pending opportunity")
    names = [f"{b}-{r}-{p}-{m}" for b, r, p, m in itertools.product(
        ("sine", "harmonic", "modal"), ("both", "habitat", "presentation"),
        ("silent", "two_tones"), ("gate", "sway"))]
    if manifest["cases"] != names:
        raise ValueError("missing, repeated, reordered or unregistered case")
    offsets = {"wait": [6000, 24000], "skip": [0], "continue": [0], "onset_now": [0],
               "delayed_onset": [6000, 24000], "release": [0, 6000, 24000], "gap": [0, 6000, 24000]}
    compared = ineligible = coordinates = tail_cases = prefix_coordinates = 0
    largest_error = 0.
    hashes = {"manifest.json": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    for name in names:
        path = root / name / "profiles.json"
        profiles = json.loads(path.read_text())
        hashes[f"{name}/profiles.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        body, route, population, modulator = name.split("-")
        if (profiles["body"] != body or profiles["routing"] != route
                or profiles["active_tones"] != (2 if population == "two_tones" else 0)
                or profiles["modulator"] != ("seq_gate" if modulator == "gate" else "drone_sway")):
            raise ValueError("profile identity differs from the registered case")
        if has_prefix:
            paths = [None if route == "presentation" else "prefix.f32le",
                     None if route == "habitat" else "prefix.f32le"]
            if profiles["prefix_pcm"] != paths:
                raise ValueError("wrong prefix bus projection")
            prefix, prefix_samples = read_pcm(root / name / "prefix.f32le", issue)
            if (population == "silent") != (not any(prefix_samples)):
                raise ValueError("prefix activity differs from registered body state")
            count, error = check_energy(profiles["prefix_energy_per_hop"], prefix_samples, paths, hop, name+"/prefix")
            prefix_coordinates += count
            largest_error = max(largest_error, error)
            hashes[f"{name}/prefix.f32le"] = hashlib.sha256(prefix).hexdigest()
        active = profiles["active_tones"] == 2
        if profiles["active_tones"] not in (0, 2):
            raise ValueError(f"unexpected active body: {name}")
        natural = None
        rows = profiles["branches"]
        expected = [(c, o) for c, times in offsets.items() for o in times]
        if [(r["class"], r["candidate_sample"] - issue) for r in rows] != expected:
            raise ValueError(f"missing or duplicate candidate: {name}")
        releases = {}
        for row in rows:
            cls, offset = row["class"], row["candidate_sample"] - issue
            expected_eligibility = active or cls not in ("continue", "release")
            if row["eligible"] != expected_eligibility:
                raise ValueError(f"eligibility differs: {name}/{cls}/{offset}")
            if row["consumes_due_opportunity"] != (cls == "skip"):
                raise ValueError("opportunity bookkeeping mismatch")
            if row["withhold_until"] != (issue + offset + period if cls == "gap" else None):
                raise ValueError("gap endpoint was not frozen at the decision")
            if row["reconsider_at"] != (issue + offset if cls == "wait" else None):
                raise ValueError("wait reconsideration time mismatch")
            if not row["eligible"]:
                if "pcm" in row or energy_key in row:
                    raise ValueError("ineligible candidate received fabricated audio")
                ineligible += 1
                continue
            compared += 1
            if not row["snapshot_matches_independent_replay"]:
                raise ValueError("renderer did not pass owned-state comparison")
            if has_prefix and not row["prefix_matches_independent_replay"]:
                raise ValueError("branch lost the common private prefix")
            path_name = f"{cls}-{offset}.f32le"
            expected_paths = [None if profiles["routing"] == "presentation" else path_name,
                              None if profiles["routing"] == "habitat" else path_name]
            if row["pcm"] != expected_paths:
                raise ValueError("wrong bus projection")
            pcm, samples = read_pcm(root / name / path_name, horizon)
            hashes[f"{name}/{path_name}"] = hashlib.sha256(pcm).hexdigest()
            if cls == "wait" and natural is None:
                natural = pcm
            if cls in ("wait", "skip", "continue") and pcm != natural:
                raise ValueError("no-added-excitation classes lost natural continuation")
            if cls == "delayed_onset" and pcm[:offset*4] != natural[:offset*4]:
                raise ValueError("delayed excitation changed its preceding audio")
            if cls in ("onset_now", "delayed_onset") and pcm == natural:
                raise ValueError("candidate excitation had no acoustic effect")
            if active and cls in ("release", "gap"):
                if pcm == natural or not any(abs(x) > 1e-6 for x in samples[offset:offset+period]):
                    raise ValueError("release was ignored or abstention erased its tail")
                tail_cases += 1
            if cls == "release":
                releases[offset] = pcm
            if active and cls == "gap" and pcm != releases[offset]:
                raise ValueError("gap changed release acoustics without another excitation")
            if not active and cls not in ("onset_now", "delayed_onset") and any(samples):
                raise ValueError("silent body acquired activity")
            count, error = check_energy(row[energy_key], samples, expected_paths, hop, f"{name}/{cls}/{offset}")
            coordinates += count
            largest_error = max(largest_error, error)
        print(name, flush=True)
    if (compared, ineligible) != (396, 72) or (manifest["realized_branches"], manifest["ineligible_branches"]) != (compared, ineligible):
        raise ValueError("incorrect acquisition denominator")
    return dict(schema="i10-actual-action-profiles-verification-v2" if has_prefix else "i10-actual-action-profiles-verification-v1", cases=len(names),
                sample_rate=rate, hop_samples=hop, issue_sample=issue, prefix_energy_coordinates=prefix_coordinates,
                realized_branches=compared, ineligible_branches=ineligible,
                energy_coordinates=coordinates, tail_cases=tail_cases,
                max_energy_absolute_error=largest_error, files_sha256=hashes,
                claim="PCM and registered class semantics only; no live projection accuracy, ordinal calibration or resource acceptance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    (args.directory / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files_sha256"}))
