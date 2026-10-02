#!/usr/bin/env python3
"""Compare frozen I11-2 arrival ON/OFF reports through their first choice split."""

import argparse
import hashlib
import json
from pathlib import Path

DECISION_INPUTS = (
    "sample_rate", "due_frame", "period_frames", "width", "earliest", "coupling",
    "overlap_sensitivity", "onset_allowed", "memory", "own_band_energy",
    "sound_hold_sec", "sound_adsr", "footprint_source", "footprint_identity",
    "current_identity", "footprint_requested_at", "footprint_received_at",
    "footprint_d_samples", "footprint_truncated", "footprint_delay",
    "footprint_power", "forecast_observed_frame",
    "forecast_available_through_frame", "skipped_cycles",
)
CANDIDATE_INPUTS = (
    "offset", "at", "displacement_sq", "context_distance",
    "pred_external_band_energy", "overlap", "external_energy",
)
TOLERANCE = 1e-5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decisions(path):
    with open(path) as source:
        rows = [json.loads(line) for line in source
                if '"participation_decision"' in line]
    return sorted(rows, key=lambda row: (row["now"], row["voice_id"]))


def compare(on_report, off_report, on_wav, off_wav):
    on, off = decisions(on_report), decisions(off_report)
    result = {
        "on_decisions": len(on), "off_decisions": len(off),
        "compared_prefix": 0, "eligible_known": 0,
        "arrival_states": {}, "status": "no_divergence",
        "wav_on_sha256": sha256(on_wav), "wav_off_sha256": sha256(off_wav),
    }
    result["wav_differs"] = result["wav_on_sha256"] != result["wav_off_sha256"]
    for index, (arrival, control) in enumerate(zip(on, off)):
        state = arrival.get("arrival_state")
        result["arrival_states"][str(state)] = result["arrival_states"].get(str(state), 0) + 1
        if any(key.startswith("arrival_") for key in control):
            result.update(status="off_has_arrival_fields", index=index)
            return result
        if (arrival["now"], arrival["voice_id"]) != (control["now"], control["voice_id"]):
            result.update(status="schedule_differs_before_choice", index=index)
            return result
        ac, bc = arrival["candidates"], control["candidates"]
        if len(ac) != 23 or len(bc) != 23 or any((a is None) != (b is None)
                                                      for a, b in zip(ac, bc)):
            result.update(status="candidate_structure_differs", index=index)
            return result
        differences = [key for key in DECISION_INPUTS
                       if arrival.get(key) != control.get(key)]
        for slot, (a, b) in enumerate(zip(ac, bc)):
            if a is None:
                continue
            if a["offset"] != slot - 2 or b["offset"] != slot - 2:
                differences.append(f"candidates[{slot}].offset_order")
            differences += [f"candidates[{slot}].{key}" for key in CANDIDATE_INPUTS
                            if a.get(key) != b.get(key)]
            probability, term = a.get("arrival_probability"), a.get("arrival_cost")
            if state == "known":
                if (arrival.get("arrival_groups", 0) <= 0 or probability is None
                        or term is None or not 0.0 <= probability <= 1.0
                        or abs(term - arrival["coupling"] * 4.0 * (1.0 - probability)) > TOLERANCE):
                    differences.append(f"candidates[{slot}].arrival_term")
            elif probability is not None or term is not None:
                differences.append(f"candidates[{slot}].unknown_as_known")
            base = a["cost"] - (term or 0.0)
            if abs(base - b["cost"]) > TOLERANCE:
                differences.append(f"candidates[{slot}].base_cost")
        if differences:
            result.update(status="nonarrival_inputs_differ", index=index,
                          differing_fields=differences[:25])
            return result
        result["compared_prefix"] += 1
        if state == "known":
            result["eligible_known"] += 1
        diverged = (arrival["selected_offset"] != control["selected_offset"]
                    or arrival["selected_at"] != control["selected_at"]
                    or arrival["skipped"] != control["skipped"])
        if diverged:
            result.update(index=index, voice_id=arrival["voice_id"], now=arrival["now"],
                          selected_on=arrival["selected_at"],
                          selected_off=control["selected_at"],
                          skipped_on=arrival["skipped"], skipped_off=control["skipped"])
            result["status"] = ("arrival_choice_effect"
                                if state == "known" and not arrival["skipped"]
                                and not control["skipped"]
                                and arrival["selected_offset"] != control["selected_offset"]
                                and result["wav_differs"] else "invalid_or_unproved_split")
            return result
    if len(on) != len(off):
        result.update(status="schedule_length_differs", index=min(len(on), len(off)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", action="append", required=True)
    args = parser.parse_args()
    rows = {}
    for case in args.case:
        stem = args.root / case
        rows[case] = compare(f"{stem}-on.jsonl", f"{stem}-off.jsonl",
                             f"{stem}-on.wav", f"{stem}-off.wav")
    accepted = {"arrival_choice_effect", "no_divergence"}
    verdict = all(row["status"] in accepted for row in rows.values()) and any(
        row["status"] == "arrival_choice_effect" for row in rows.values())
    output = {"schema": "conchordal/i11-stage2-arrival-effect/1",
              "cases": rows, "local_effect_pass": verdict}
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
