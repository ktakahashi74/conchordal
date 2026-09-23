#!/usr/bin/env python3
"""I11-1 §5.4(b): candidate records of a baseline and a new build on the same runs.

Takes report pairs (baseline, new) rendered from the same scenario and configuration
with `[temporal_onset_comparison]` absent. Candidate records (`body_candidate_energy`)
are matched on (source_id, source_generation, tone_id, issued_at, decision_at, scope);
a matched pair must be equal in every field except the wall-clock `processing_us`.
Records present on one side only are counted, not compared: the live candidate worker
drops packets under saturation, so their number varies between runs.
"""

import argparse
import json
import sys
from pathlib import Path

WALL_CLOCK = ("processing_us",)
KEY = ("source_id", "source_generation", "tone_id", "issued_at", "decision_at", "scope")
EXAMPLES = 5


def load(path):
    records = {}
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("type") != "body_candidate_energy":
                continue
            key = tuple(record[name] for name in KEY)
            if key in records:
                raise ValueError(f"{path}: duplicate candidate key {key}")
            records[key] = {name: value for name, value in record.items() if name not in WALL_CLOCK}
    return records


def compare(base_path, new_path):
    base, new = load(base_path), load(new_path)
    matched = base.keys() & new.keys()
    differing = []
    for key in sorted(matched, key=str):
        if base[key] != new[key]:
            fields = sorted(n for n in base[key].keys() | new[key].keys()
                            if base[key].get(n) != new[key].get(n))
            differing.append({"key": dict(zip(KEY, key)), "fields": fields})
    return {
        "base": str(base_path), "new": str(new_path),
        "matched": len(matched), "identical": len(matched) - len(differing),
        "differing": len(differing), "examples": differing[:EXAMPLES],
        "base_only": len(base.keys() - new.keys()), "new_only": len(new.keys() - base.keys()),
        "excluded_fields": list(WALL_CLOCK),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", type=Path, help="baseline and new report pairs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if len(args.reports) % 2:
        parser.error("reports come in (baseline, new) pairs")
    results = [compare(args.reports[i], args.reports[i + 1]) for i in range(0, len(args.reports), 2)]
    text = json.dumps(results, indent=1)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 1 if any(r["differing"] or not r["matched"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
