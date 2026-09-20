#!/usr/bin/env python3
"""Independent exhaustive grouping admission with linear boundary searches."""

import argparse
import json
import random
from pathlib import Path

STEPS = [1 / 4, 1 / 3, 1 / 2, 2 / 3, 1, 3 / 2, 2, 3, 4]


def reference(case):
    events = case["events"]
    controls = case["controls"]
    tol = controls["tolerance"]
    found = {}
    admitted = 0

    def insert(peak, kind, shape, indices, skipped=None):
        nonlocal admitted
        middle = len(indices) // 2
        ends = [events[i][0] for i in indices]
        duration = peak[1] * case["sample_rate"] * (
            shape if kind == "integer" else sum(STEPS[b] for b in shape)
        )
        durations = [ends[middle] - ends[0], ends[-1] - ends[middle]]
        prefixes = [events[i][1] for i in indices]
        support = [prefixes[middle] - prefixes[0], prefixes[-1] - prefixes[middle]]
        if any(abs(d - duration) > tol * duration for d in durations):
            return
        if any(10 * s < 9 * d for s, d in zip(support, durations)):
            return
        weight = sum(events[i][2] for i in indices)
        nominal_steps = (
            [duration, duration]
            if kind == "integer"
            else [STEPS[b] * peak[1] * case["sample_rate"] for b in shape * 2]
        )
        residuals = [(b - a - n) / n for a, b, n in zip(ends, ends[1:], nominal_steps)]
        shape_key = (0, shape) if kind == "integer" else (1, len(shape), tuple(shape))
        key = (peak[0], shape_key, tuple((end - 16, end) for end in ends))
        proposal = {
            "period_bin": peak[0], "period_seconds": peak[1], "kind": kind,
            "shape": shape, "anchors": ends,
            "nominal_duration_seconds": duration / case["sample_rate"],
            "observed_duration_samples": durations, "coverage": list(zip(support, durations)),
            "timing_residuals": residuals, "endpoint_weight": weight,
            "mean_endpoint_weight": weight / len(indices),
            "source_start": ends[0] - 48, "source_end": ends[-1] + 16,
            "available_end": ends[-1] + 32,
            "skipped_accent": None if skipped is None else events[skipped][0],
        }
        admitted += 1
        found.setdefault(key, proposal)

    def word(peak, indices, length, skipped=None):
        intervals = [events[b][0] - events[a][0] for a, b in zip(indices, indices[1:])]
        expected = [step * peak[1] * case["sample_rate"] for step in STEPS]
        symbols = [min(range(9), key=lambda b: (abs(interval - expected[b]), b)) for interval in intervals[:length]]
        if sum(STEPS[b] for b in symbols) > 16:
            return
        if any(abs(interval - expected[b]) > tol * expected[b] for interval, b in zip(intervals, symbols * 2)):
            return
        insert(peak, "word", symbols, indices, skipped)

    for peak in case["peaks"]:
        for start, event in enumerate(events):
            for length in range(2, 5 if controls["integers_234_only"] else 17):
                duration = length * peak[1] * case["sample_rate"]
                indices = [start]
                for repeat in (1, 2):
                    target = repeat * duration
                    candidates = [i for i in range(indices[-1] + 1, len(events)) if abs((events[i][0] - event[0]) - target) <= tol * duration]
                    if not candidates:
                        break
                    indices.append(min(candidates, key=lambda i: (abs((events[i][0] - event[0]) - target), events[i][0], events[i][0] - 16)))
                if len(indices) == 3 and (not controls["strict_integer"] or indices == list(range(start, start + 3))):
                    insert(peak, "integer", length, indices)
            if controls["integers_234_only"]:
                continue
            for length in range(2, 9):
                if start + 2 * length >= len(events):
                    continue
                word(peak, list(range(start, start + 2 * length + 1)), length)
                if controls["one_skip_words"] and start + 2 * length + 1 < len(events):
                    for skipped in range(start + 1, start + 2 * length + 1):
                        word(peak, [i for i in range(start, start + 2 * length + 2) if i != skipped], length, skipped)
    return {"admitted_cases": admitted, "unique_cases": len(found), "proposals": [p for _, p in sorted(found.items(), key=lambda kv: (-kv[1]["endpoint_weight"], kv[0]))[:16]]}


def generate():
    cases = []
    baseline = {"tolerance": 0.10, "integers_234_only": False, "strict_integer": False, "one_skip_words": False}

    def add(name, offsets, *, peaks=None, controls=None, coverage=None, weights=None, origin=1000):
        prefixes = [origin]
        for j, (a, b) in enumerate(zip(offsets, offsets[1:])):
            prefixes.append(prefixes[-1] + (b - a) * (100 if coverage is None else coverage[j]) // 100)
        case = {"name": name, "sample_rate": 1000, "events": [[origin + t, prefix, 1 if weights is None else weights[i]] for i, (t, prefix) in enumerate(zip(offsets, prefixes))], "peaks": [[96, 0.5]] if peaks is None else peaks, "controls": {**baseline, **(controls or {})}}
        case["expected"] = reference(case)
        cases.append(case)

    add("both_boundaries_precede_predictions", [0, 1900, 3900])
    add("second_absolute_window_not_doubled", [0, 2000, 4350])
    add("successive_errors_cannot_cancel", [0, 1800, 4200])
    add("equidistant_boundary_earlier_wins", [0, 1800, 2200, 4000])
    add("integer_allows_extra_accents", [0, 250, 2000, 2250, 4000])
    add("strict_integer_control", [0, 250, 2000, 2250, 4000], controls={"strict_integer": True})
    add("unequal_word", [0, 500, 1250, 1750, 2500])
    add("word_total_without_order", [0, 500, 1250, 2000, 2500])
    add("extra_accent_breaks_consecutive_word", [0, 250, 500, 1250, 1750, 2500])
    add("explicit_one_skip_control", [0, 250, 500, 1250, 1750, 2500], controls={"one_skip_words": True})
    add("original_support_ninety", [0, 2000, 4000], coverage=[90, 90])
    add("original_support_eighty_nine", [0, 2000, 4000], coverage=[89, 100])
    add("no_period_no_quantized_word", [0, 500, 1250, 1750, 2500], peaks=[])
    add("empty_bank", [], peaks=[])
    add("large_epoch_same_intervals", [0, 1900, 3900], origin=2**60)
    add("rotations_retained", [0, 500, 1250, 1750, 2500, 3000, 3750])
    add("word_exceeds_sixteen_beats", [i * 2000 for i in range(13)])
    add("integer_234_inventory_control", list(range(0, 10001, 500)), controls={"integers_234_only": True})
    add("quantization_tie_smaller_step", [0, 625, 1125, 1750, 2250], controls={"tolerance": .5})
    for cap in (64, 128, 256):
        add(f"saturated_{cap}", [i * 64 for i in range(cap)], peaks=[[b, .125 * 2**(b/48)] for b in (0, 32, 48, 64, 80, 96, 112, 144)], weights=[(.25, .5, .75, 1)[i % 4] for i in range(cap)])
    rng = random.Random(20260912)
    for i in range(32):
        offsets = [0]
        for _ in range(23):
            offsets.append(offsets[-1] + rng.choice([125, 250, 333, 500, 666, 750, 1000, 1500, 2000]))
        add(f"mixed_{i}", offsets, peaks=[[48, .25], [96, .5], [144, 1.]], coverage=[rng.choice([100, 100, 90, 89]) for _ in offsets[1:]], weights=[rng.choice([.25, .5, .75, 1.]) for _ in offsets], controls={"tolerance": [.05, .1, .2][i % 3], "one_skip_words": i % 7 == 0})
    return {"schema": "temporal-grouping-inventory-fixtures-v1", "source": "independent linear search and complete candidate collection/sort; no Rust cache or admission reuse", "cases": cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/temporal_cognition/grouping-inventory.json"))
    args = parser.parse_args()
    result = generate()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(result["cases"]), "events": sum(len(c["events"]) for c in result["cases"]), "admitted_cases": sum(c["expected"]["admitted_cases"] for c in result["cases"]), "retained_proposals": sum(len(c["expected"]["proposals"]) for c in result["cases"]), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
