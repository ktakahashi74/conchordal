"""Freeze the pre-existing private timing oracle for the bounded native filter."""

import hashlib
import json
import struct
from pathlib import Path

from temporal_cognition_reference import (
    PrivateTimingTrace,
    integrate_retained_timing_bins,
    integrate_timing_bins,
    timing_lookup,
)


def encoded(value):
    if isinstance(value, float):
        return {"f64": struct.unpack("<Q", struct.pack("<d", value))[0]}
    if isinstance(value, dict):
        return {key: encoded(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encoded(item) for item in value]
    return value


def generate():
    timings = []
    supports = [([0.25, 0.25], [0., 0.]), ([0.25, 0.25], [0., 0.1]),
                ([0.2, 0.3], [0., 0.]), ([0.2, 0.3], [0., 0.1]),
                ([-1., 2.], [-0.3, 0.2]), ([3.9, 4.2], [0., 0.1]),
                ([-4., -3.], [0., 0.]), ([1., 1.000000001], [0., 0.1]),
                ([1e7, 1e7 + .01], [1e7 - .3, 1e7 - .2]),
                ([-1e-20, -1e-20], [0., 0.])]
    for periodic in (False, True):
        for interval, anchor in supports:
            for tau in (.01, 20., 1e8):
                anchors = [dict(weight=.6, interval=anchor, period_sec=1.),
                           dict(weight=.25, interval=[x + .125 for x in anchor], period_sec=.7),
                           dict(weight=.1, interval=None, period_sec=1.)]
                timings.append(dict(periodic=periodic, interval=interval, anchors=anchors, tau=tau,
                                    support=integrate_timing_bins(interval, anchors, periodic),
                                    retained=integrate_retained_timing_bins(interval, anchors, periodic, tau)))
    cases = []
    for capacity in (2, 16):
        for strength in (.3, 1.2):
            bank = PrivateTimingTrace(1, 20., .7, strength, capacity)
            operations = []
            keys = [(1, (i, 1), "periodic" if i % 2 else "nonperiodic") for i in range(1, 5)]
            for step in range(28):
                end = step * .2 + .1
                head = "onset" if step % 3 else "release"
                interval = [end - .04, end]
                entries = []
                for j, key in enumerate(keys):
                    entries.append(dict(key=key, weight=[.35, .3, .2, .1][j], anchors=[
                        dict(weight=.7, interval=[end - .5 - .1 * (step % 2), end - .4], period_sec=1. + j / 4),
                        dict(weight=.2, interval=[end - .9, end - .85], period_sec=1. + j / 4),
                        dict(weight=.1, interval=None, period_sec=1. + j / 4)]))
                confirmed = step not in (3, 11)
                observed_fraction = 0. if step == 9 else .8
                retained = keys if step < 15 or step >= 20 else keys[1:]
                if step == 20:
                    keys[0] = (1, (1, 2), "periodic")
                    retained = keys
                before = [{"key": key, "probabilities": bank.probabilities(key, head)} for key in keys]
                receipt = bank.observe(step, head, interval, entries, observed_fraction, retained, confirmed)
                operations.append(dict(id=step, head=head, interval=interval, entries=entries,
                                       observed_fraction=observed_fraction, retained=list(retained), confirmed=confirmed,
                                       applied=receipt["applied"], credits=[receipt["credits"][e["key"]] for e in entries],
                                       unassigned=receipt["unassigned"], removed=receipt["removed"], evicted=receipt["evicted"],
                                       before=before, traces=[dict(key=key, onset=list(value["onset"]), release=list(value["release"]),
                                                                 end=value["end"]) for key, value in bank.traces.items()]))
            cases.append(dict(capacity=capacity, strength=strength, operations=operations))
    lookups = []
    for periodic in (False, True):
        masses = [0. if i % 3 == 0 else float(i + 1) for i in range(32 + (not periodic))]
        for at in [-1., 0., .01, .015625, .1, .5, .99, 1., 3.9375, 4., 4.01]:
            lookups.append(dict(periodic=periodic, masses=masses, position=at,
                                expected=timing_lookup(masses, at, periodic)))
    root = Path(__file__).resolve().parents[1]
    source = root / "scripts/temporal_cognition_reference.py"
    result = dict(schema="private-trace-native-oracle-v1", source="scripts/temporal_cognition_reference.py",
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), timings=timings,
                  cases=cases, lookups=lookups)
    target = root / "tests/fixtures/temporal_cognition/private_trace.json"
    target.write_text(json.dumps(encoded(result), separators=(",", ":")) + "\n")
    print(f"{len(timings)} timing cases, {sum(len(c['operations']) for c in cases)} state transitions, {len(lookups)} lookups")


if __name__ == "__main__":
    generate()
