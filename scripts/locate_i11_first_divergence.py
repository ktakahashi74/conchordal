#!/usr/bin/env python3
"""I11-1 §5.5(a): where a `body` run and a `proxy` run first choose differently, and why.

Takes report pairs (body, proxy) rendered from the same scenario with the offline
deterministic delivery. Traced decisions are ordered by (now, voice_id). Until the first
decision whose selected offset differs, both runs share one state; at that decision the
script lists every input that differs besides the footprint powers. An empty list shows
that the acoustic difference enters through `power_k` alone.
"""

import argparse
import json
import sys
from pathlib import Path

DECISION_INPUTS = (
    "due_frame", "period_frames", "width", "earliest", "coupling", "overlap_sensitivity",
    "onset_allowed", "memory", "own_band_energy", "footprint_d_samples",
    "forecast_observed_frame", "forecast_available_through_frame",
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
    for index, (b, p) in enumerate(zip(body, proxy)):
        if (b["now"], b["voice_id"]) != (p["now"], p["voice_id"]):
            return {"status": "schedules_part_before_a_choice_differs", "index": index}
        if b["selected_offset"] == p["selected_offset"]:
            continue
        differing = [k for k in DECISION_INPUTS if b[k] != p[k]]
        differing += [f"candidates.{k}" for k in CANDIDATE_INPUTS
                      if any((cb or {}).get(k) != (cp or {}).get(k)
                             for cb, cp in zip(b["candidates"], p["candidates"]))]
        return {
            "status": "diverged", "index": index, "voice_id": b["voice_id"], "now": b["now"],
            "sources": [b["footprint_source"], p["footprint_source"]],
            "powers_differ": b["footprint_power"] != p["footprint_power"],
            "other_differing_inputs": differing,
        }
    return {"status": "no_divergence", "compared": min(len(body), len(proxy))}


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
    return 1 if any(r["status"] == "diverged" and r["other_differing_inputs"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
