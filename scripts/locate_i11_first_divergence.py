#!/usr/bin/env python3
"""I11-1 §5.5(a): where a `body` run and a `proxy` run first choose differently, and why.

Takes report pairs (body, proxy) rendered from the same scenario with the offline
deterministic delivery. Traced decisions are ordered by (now, voice_id). The script
checks every compared decision through the first different selected offset or skip.
A valid first divergence has different footprint powers and identical other inputs,
candidate structure, and preceding decisions' non-power inputs.
"""

import argparse
import json
import sys
from pathlib import Path

DECISION_INPUTS = (
    "sample_rate", "due_frame", "period_frames", "width", "earliest", "coupling",
    "overlap_sensitivity", "onset_allowed", "memory", "own_band_energy",
    "sound_hold_sec", "sound_adsr",
    "footprint_identity", "current_identity", "footprint_requested_at",
    "footprint_received_at", "footprint_d_samples",
    "footprint_truncated", "footprint_delay", "forecast_observed_frame",
    "forecast_available_through_frame", "skipped_cycles",
)
CANDIDATE_INPUTS = (
    "at", "displacement_sq", "context_distance", "pred_external_band_energy", "external_energy",
)


def decisions(path):
    with open(path) as handle:
        records = [json.loads(line) for line in handle if '"participation_decision"' in line]
    return sorted(records, key=lambda r: (r["now"], r["voice_id"]))


def locate(body_path, proxy_path):
    body, proxy = decisions(body_path), decisions(proxy_path)
    first_prior_input_difference = None
    for index, (b, p) in enumerate(zip(body, proxy)):
        if (b["now"], b["voice_id"]) != (p["now"], p["voice_id"]):
            return {"status": "schedules_part_before_a_choice_differs", "index": index}
        b_offsets = [None if c is None else c.get("offset") for c in b["candidates"]]
        p_offsets = [None if c is None else c.get("offset") for c in p["candidates"]]
        if (len(b_offsets) != 23 or len(p_offsets) != 23 or b_offsets != p_offsets
                or any((bc is None) != (pc is None)
                       for bc, pc in zip(b["candidates"], p["candidates"]))
                or any(c is not None and c.get("offset") != i - 2
                       for candidates in (b["candidates"], p["candidates"])
                       for i, c in enumerate(candidates))):
            return {"status": "candidate_structure_differs", "index": index,
                    "body_candidate_offsets": b_offsets, "proxy_candidate_offsets": p_offsets}
        differing = [k for k in DECISION_INPUTS if b[k] != p[k]]
        differing += [f"candidates.{k}" for k in CANDIDATE_INPUTS
                      if any(cb is not None and cb[k] != cp[k]
                             for cb, cp in zip(b["candidates"], p["candidates"]))]
        diverged_outputs = [k for k in ("selected_offset", "skipped") if b[k] != p[k]]
        if not diverged_outputs:
            if differing and first_prior_input_difference is None:
                first_prior_input_difference = {"index": index, "inputs": differing}
            continue
        return {
            "status": "diverged", "index": index, "voice_id": b["voice_id"], "now": b["now"],
            "sources": [b["footprint_source"], p["footprint_source"]],
            "sources_valid": b["footprint_source"] == "body"
                             and p["footprint_source"] == "proxy(setting)",
            "diverged_outputs": diverged_outputs,
            "powers_differ": b["footprint_power"] != p["footprint_power"],
            "other_differing_inputs": differing,
            "first_prior_input_difference": first_prior_input_difference,
        }
    if len(body) != len(proxy):
        return {"status": "schedules_part_before_a_choice_differs", "index": min(len(body), len(proxy)),
                "body_decisions": len(body), "proxy_decisions": len(proxy)}
    return {"status": "no_divergence", "compared": len(body),
            "first_prior_input_difference": first_prior_input_difference}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", type=Path, help="body and proxy report pairs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if len(args.reports) % 2:
        parser.error("reports come in (body, proxy) pairs")
    results = [{"body": str(args.reports[i]), "proxy": str(args.reports[i + 1]),
                **locate(args.reports[i], args.reports[i + 1])}
               for i in range(0, len(args.reports), 2)]
    text = json.dumps(results, indent=1)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 1 if any(
        r["status"] not in ("diverged", "no_divergence")
        or r["first_prior_input_difference"] is not None
        or (r["status"] == "diverged"
            and (not r["sources_valid"] or not r["powers_differ"]
                 or r["other_differing_inputs"]))
        for r in results
    ) else 0


if __name__ == "__main__":
    sys.exit(main())
