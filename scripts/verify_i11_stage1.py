#!/usr/bin/env python3
"""Independent reference for I11-1 stage 1 (docs/roadmap/temporal-dcc/i11-onset-comparison.md §5.1-5.3).

Reads a render report (JSONL) written with `[temporal_onset_comparison]` configured and
recomputes every traced participation decision from the inputs it carries:

- §5.1 input match: the footprint a decision used equals the record the worker delivered,
  the proxy powers and span follow §4.2/§4.3, every cost term and the selection are
  recomputed, and the later `participation_context` restates the same decision.
- §5.2 state intervention: body powers are replaced by the proxy on the same observation;
  only the overlap term moves, and the selections that change are counted.
- §5.3 deadlines: no footprint from the future, no stale identity used as a body, no
  superseded record used, no external energy before `available_through_frame`, no
  reselection while a reservation stands.

The Rust side is not imported; the formulas are restated here from the registration.
"""

import argparse
import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

BINS = 16
OFFSETS = range(-2, 21)
REL_TOL = 1e-5
ABS_TOL = 1e-7
EXAMPLES = 5


def f32(value):
    """Round to f32; serde writes f32 in its shortest form, which parses to a nearby f64."""
    return struct.unpack("f", struct.pack("f", value))[0]


def close(a, b):
    return abs(a - b) <= ABS_TOL + REL_TOL * max(abs(a), abs(b))


def rust_round(value):
    """`f64::round`: halves away from zero, unlike Python's banker's rounding."""
    return math.floor(value + 0.5) if value >= 0 else -math.floor(-value + 0.5)


def key(identity):
    return None if identity is None else json.dumps(identity, sort_keys=True)


def proxy_power(age, hold, adsr, fs):
    """Squared ADSR proxy gain at `age` seconds (§4.3)."""
    if adsr is None:
        return 1.0
    attack_sec, decay_sec, sustain, release_sec = adsr
    release = max(release_sec, 0.0)
    duration = f32(hold + release)
    attack = min(max(attack_sec, f32(1.0 / fs)), hold)
    if age < attack:
        before = age / attack
    elif decay_sec > 0.0:
        before = sustain + (1.0 - sustain) * math.exp(-6.908 * (age - attack) / decay_sec)
    else:
        before = sustain
    tail = 1.0 if age < hold else min(max((duration - age) / max(release, 1e-12), 0.0), 1.0)
    gain = before * tail
    return gain * gain


def proxy_span(decision, fs):
    release = 0.0 if decision["sound_adsr"] is None else max(decision["sound_adsr"][3], 0.0)
    duration = f32(decision["sound_hold_sec"] + release)
    return min(duration, 4.0) * fs


def hellinger(memory, predicted):
    a = [v for row in memory for v in row]
    b = [v for row in predicted for v in row]
    a_mass, b_mass = sum(a), sum(b)
    if a_mass <= 1e-12 or b_mass <= 1e-12:
        return float((a_mass > 1e-12) != (b_mass > 1e-12))
    return 0.5 * sum((math.sqrt(x / a_mass) - math.sqrt(y / b_mass)) ** 2 for x, y in zip(a, b))


def overlap(own, powers, energies):
    total = 0.0
    for power, energy in zip(powers, energies):
        if energy is None:
            continue
        for own_band, other in zip(own, energy):
            scaled = own_band * power
            total += scaled * other / (scaled + other + 1e-12)
    return total


def cost_of(decision, candidate, distance, overlap_sum, powers):
    cost = candidate["displacement_sq"]
    if distance is not None:
        cost += decision["coupling"] * distance
    if overlap_sum is not None:
        norm = max(sum(decision["own_band_energy"]) * sum(powers), 1e-12)
        cost += decision["coupling"] * 6.0 * decision["overlap_sensitivity"] * overlap_sum / norm
    return cost


def first_minimum(costs):
    best = None
    for offset, cost in costs:
        if best is None or cost < best[1]:
            best = (offset, cost)
    return best


class Findings:
    def __init__(self):
        self.checked = defaultdict(int)
        self.failed = defaultdict(list)

    def check(self, name, ok, example):
        self.checked[name] += 1
        if not ok:
            self.failed[name].append(example)

    def summary(self):
        return {
            name: {
                "checked": self.checked[name],
                "failed": len(self.failed[name]),
                "examples": self.failed[name][:EXAMPLES],
            }
            for name in sorted(self.checked)
        }


def load(path):
    decisions, footprints, contexts = [], {}, {}
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            kind = record.get("type")
            if kind == "participation_decision":
                decisions.append(record)
            elif kind == "body_footprint":
                footprints[(key(record["identity"]), record["received_at"])] = record
            elif kind == "participation_context":
                contexts[(record["voice_id"], record["onset_frame"])] = record
    return decisions, footprints, contexts


def verify(decisions, footprints, contexts):
    f = Findings()
    intervention = {"body_decisions": 0, "changed_selections": 0}
    last_selected = {}
    for decision in sorted(decisions, key=lambda d: (d["voice_id"], d["now"])):
        fs = decision["sample_rate"]
        where = {"voice_id": decision["voice_id"], "now": decision["now"]}
        source = decision["footprint_source"]
        identity = key(decision["footprint_identity"])
        current = key(decision["current_identity"])
        received = decision["footprint_received_at"]
        record = footprints.get((identity, received)) if identity is not None else None
        forecast = decision["forecast_observed_frame"] is not None

        # §5.1 footprint: the delivered record, or the proxy on the span §4.3 prescribes.
        if identity is not None:
            f.check("5.1 footprint record delivered", record is not None, where)
        span = 0
        if record is not None and record["state"] in ("body", "body_silent"):
            span = record["d_samples"]
        d = float(span) if span > 0 else proxy_span(decision, fs)
        f.check("5.1 footprint span", close(decision["footprint_d_samples"], d),
                {**where, "reported": decision["footprint_d_samples"], "expected": d})
        delays = [(k + 0.5) * decision["footprint_d_samples"] / 16.0 for k in range(BINS)]
        f.check("5.1 footprint delays", delays == decision["footprint_delay"], where)
        proxy = [proxy_power(delay / fs, decision["sound_hold_sec"], decision["sound_adsr"], fs)
                 for delay in delays]
        if source == "body":
            expected = [f32(p) for p in record["power"]] if record else None
            f.check("5.1 body powers", expected == [f32(p) for p in decision["footprint_power"]], where)
        else:
            f.check("5.1 proxy powers",
                    all(close(a, b) for a, b in zip(proxy, decision["footprint_power"])), where)

        # §5.3 footprint deadlines and identity.
        if received is not None:
            f.check("5.3 footprint received before the decision", received <= decision["now"], where)
        if source == "body":
            f.check("5.3 body identity is current", identity == current, where)
        if record is not None:
            f.check("5.3 superseded record unused", not record["superseded"], where)
            if identity != current:
                f.check("5.3 stale identity falls back", source == "proxy(stale)", where)

        # §5.1 every candidate and every term.
        due, period, width = decision["due_frame"], decision["period_frames"], decision["width"]
        earliest = decision["earliest"]
        powers = decision["footprint_power"]
        costs, proxy_costs = [], []
        for slot, offset in enumerate(OFFSETS):
            shift = offset * width / 2.0 if offset < 0 else offset * period / 20.0
            at = max(due, earliest) if offset == 0 else due + shift
            candidate = decision["candidates"][slot]
            f.check("5.1 candidate grid", (candidate is not None) == (at >= earliest),
                    {**where, "offset": offset})
            if candidate is None:
                continue
            here = {**where, "offset": offset}
            f.check("5.1 candidate time", candidate["at"] == at and candidate["offset"] == offset, here)
            displacement = (at - due) / period
            f.check("5.1 displacement",
                    f32(candidate["displacement_sq"]) == f32(displacement * displacement), here)
            predicted = candidate["pred_external_band_energy"]
            applies_context = decision["onset_allowed"] and decision["memory"] is not None and predicted is not None
            distance = hellinger(decision["memory"], predicted) if applies_context else None
            f.check("5.1 context term",
                    (distance is None) == (candidate["context_distance"] is None)
                    and (distance is None or close(distance, candidate["context_distance"])), here)
            applies_overlap = decision["onset_allowed"] and decision["overlap_sensitivity"] != 0.0 and forecast
            energies = candidate["external_energy"]
            overlap_sum = overlap(decision["own_band_energy"], powers, energies) if applies_overlap else None
            f.check("5.1 overlap term",
                    (overlap_sum is None) == (candidate["overlap"] is None)
                    and (overlap_sum is None or close(overlap_sum, candidate["overlap"])), here)
            cost = cost_of(decision, candidate, candidate["context_distance"], candidate["overlap"], powers)
            f.check("5.1 cost", close(cost, candidate["cost"]),
                    {**here, "reported": candidate["cost"], "expected": cost})
            available = decision["forecast_available_through_frame"]
            for delay, energy in zip(delays, energies):
                if energy is not None:
                    f.check("5.3 external energy after availability", at + delay >= available, here)
            costs.append((offset, candidate["cost"]))
            if source == "body" and applies_overlap:
                swapped = overlap(decision["own_band_energy"], proxy, energies)
                proxy_costs.append((offset, cost_of(decision, candidate, candidate["context_distance"],
                                                    swapped, proxy)))

        # §5.1 selection and skip, on the reported costs.
        best_offset, best_cost = first_minimum(costs)
        f.check("5.1 selection", best_offset == decision["selected_offset"]
                and best_cost == decision["selected_cost"], where)
        reference = dict(costs).get(0)
        f.check("5.1 reference cost", reference == decision["reference_cost"], where)
        skip = (decision["onset_allowed"] and forecast and sum(decision["own_band_energy"]) > 1e-12
                and decision["overlap_sensitivity"] > 0.0
                and best_cost > f32(1.0 + decision["skipped_cycles"]))
        f.check("5.1 skip rule", skip == decision["skipped"], where)

        # §5.2 the same observation with proxy powers.
        if proxy_costs:
            intervention["body_decisions"] += 1
            if first_minimum(proxy_costs)[0] != best_offset:
                intervention["changed_selections"] += 1

        if decision["skipped"]:
            continue
        # §5.3 no reselection while the previous reservation stands.
        previous = last_selected.get(decision["voice_id"])
        if previous is not None:
            f.check("5.3 no reselection before the reservation", decision["now"] >= rust_round(previous),
                    {**where, "previous_selected_at": previous})
        last_selected[decision["voice_id"]] = decision["selected_at"]

        # §5.1 the context written after the outcome restates the decision.
        context = contexts.get((decision["voice_id"], rust_round(decision["selected_at"])))
        if forecast and context is not None:
            f.check("5.1 context restates the decision",
                    context["selected_cost"] == decision["selected_cost"]
                    and context["reference_cost"] == decision["reference_cost"]
                    and context["selected_offset"] == decision["selected_offset"]
                    and context["footprint_source"] == source
                    and context["footprint_received_at"] == received
                    and context["forecast_observed_frame"] == decision["forecast_observed_frame"], where)
            f.check("5.3 context windows after availability",
                    context["target_start_frames"][0] >= decision["forecast_available_through_frame"], where)
    return f, intervention


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    out, failed = {}, False
    for path in args.reports:
        decisions, footprints, contexts = load(path)
        findings, intervention = verify(decisions, footprints, contexts)
        summary = findings.summary()
        failed |= any(entry["failed"] for entry in summary.values()) or not decisions
        out[str(path)] = {
            "decisions": len(decisions),
            "skipped": sum(d["skipped"] for d in decisions),
            "footprint_sources": dict(sorted(
                {s: sum(d["footprint_source"] == s for d in decisions)
                 for s in {d["footprint_source"] for d in decisions}}.items())),
            "delivered_footprints": len(footprints),
            "contexts": len(contexts),
            "checks": summary,
            "intervention_5_2": intervention,
        }
    text = json.dumps(out, indent=1)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
