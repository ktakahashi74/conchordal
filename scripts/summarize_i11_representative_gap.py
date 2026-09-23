#!/usr/bin/env python3
"""Distribution report for I11-1 §5.9: representative against actual onset footprints.

Reads the directory written by the ignored Rust acquisition
`runtime::body_profiles::representative_gap::acquire_representative_gap`. For every
granted onset it takes the actual tone's sixteen coherent energies over
`[onset, onset + D_rep)` and the representative powers of the decision that chose the
onset, normalizes the actual energies to their peak, and reports

- `power_gap`: max_k |power_k^actual - power_k^rep|, and
- `overlap_gap`: |Ov^actual - Ov^rep| on the selected candidate's external energies, with
  `term_gap` the same difference after `coupling * 6 * s / norm`.

Onsets with an unsupported coherent bin are excluded and counted. When both sides are
known silence the gap is zero. The result is a distribution (median, p95, maximum by
nearest rank), not a pass/fail judgment.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_i11_stage1 import overlap  # noqa: E402


def nearest_rank(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def distribution(values):
    if not values:
        return {"count": 0}
    return {"count": len(values), "median": nearest_rank(values, 0.5),
            "p95": nearest_rank(values, 0.95), "max": max(values)}


def normalized(energies):
    peak = max(energies)
    return [0.0] * len(energies) if peak <= 0.0 else [e / peak for e in energies]


def summarize_case(directory):
    decisions = {}
    with open(directory / "report.jsonl") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("type") == "participation_decision" and not record["skipped"]:
                decisions[(record["voice_id"], record["now"])] = record
    counts = defaultdict(int)
    metrics = defaultdict(lambda: defaultdict(list))
    with open(directory / "gap.jsonl") as handle:
        for line in handle:
            row = json.loads(line)
            counts[row["status"]] += 1
            if row["status"] != "projected":
                continue
            energies = row["coherent_energies"]
            if any(e is None for e in energies):
                counts["excluded_unsupported_bin"] += 1
                continue
            decision = decisions[(row["key"]["source_id"], row["decided_at"])]
            source = decision["footprint_source"]
            rep = decision["footprint_power"]
            actual = normalized(energies)
            silent = max(energies) <= 0.0 and max(rep) <= 0.0
            power_gap = 0.0 if silent else max(abs(a - r) for a, r in zip(actual, rep))
            metrics[source]["power_gap"].append(power_gap)
            candidate = decision["candidates"][decision["selected_offset"] + 2]
            if candidate["overlap"] is None:
                counts["overlap_not_applied"] += 1
                continue
            own = decision["own_band_energy"]
            external = candidate["external_energy"]
            ov_actual, ov_rep = overlap(own, actual, external), overlap(own, rep, external)
            scale = decision["coupling"] * 6.0 * decision["overlap_sensitivity"]
            norm_actual = max(sum(own) * sum(actual), 1e-12)
            norm_rep = max(sum(own) * sum(rep), 1e-12)
            metrics[source]["overlap_gap"].append(abs(ov_actual - ov_rep))
            metrics[source]["term_gap"].append(abs(scale * (ov_actual / norm_actual - ov_rep / norm_rep)))
    return {
        "counts": dict(sorted(counts.items())),
        "by_footprint_source": {
            source: {name: distribution(values) for name, values in sorted(named.items())}
            for source, named in sorted(metrics.items())
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("acquisition", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    cases = sorted(p for p in args.acquisition.iterdir() if (p / "gap.jsonl").exists())
    out = {case.name: summarize_case(case) for case in cases}
    text = json.dumps(out, indent=1)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0 if cases else 1


if __name__ == "__main__":
    sys.exit(main())
