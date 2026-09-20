"""Independent rational peak/ownership reference for the first acoustic stage."""

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import sys


def reference(scan, bus_energy):
    values = [Fraction(v) for v in scan]
    maximum = max(values)
    candidates = []
    for i, v in enumerate(values):
        neighbors = [values[j] for j in (i - 2, i + 2) if 0 <= j < len(values)]
        if (v > 0 and v >= maximum / 100 and neighbors
                and (i == 0 or v > values[i - 1])
                and (i == len(values) - 1 or v >= values[i + 1])
                and v - max(neighbors) >= v / 10):
            candidates.append(i)
    candidates = sorted(candidates, key=lambda i: (-values[i], i))[:7]
    owners = []
    for i in range(len(values)):
        nearby = [p for p in candidates if abs(p - i) <= 2]
        owners.append(min(nearby, key=lambda p: (abs(p - i), p)) if nearby else None)
    masses = [sum(v for v, owner in zip(values, owners) if owner == p)
              for p in candidates]
    masses += [Fraction(0)] * (7 - len(candidates))
    masses += [sum(v for v, owner in zip(values, owners) if owner is None)]
    total = sum(values)
    energy = ([Fraction(bus_energy) * m / total for m in masses] if total
              else [Fraction(0)] * 7 + [Fraction(bus_energy)])
    assert sum(energy) == Fraction(bus_energy)
    return {"power_scan": scan, "mono_energy": bus_energy,
            "peak_bins": candidates + [None] * (7 - len(candidates)),
            "energy": [float(e) for e in energy],
            "spectral_shape_supported": bool(total or bus_energy == 0)}


if __name__ == "__main__":
    rng = random.Random(9122032)
    examples = [([0] * 9, 0), ([0] * 9, 1), ([1] * 9, 1),
                ([4, 0, 0, 1, 1, 0, 0, 0, 2], 8),
                ([0, 1, 0, 0.5, 0, 2, 0, 0, 0], 7),
                ([100, 0, 0, 0, 1, 0, 0, 0, 0.5], 1),
                ([9, 0, 10, 0, 0], 1), ([10, 0, 10, 0, 0], 1),
                ([1], 1), ([1, 0], 1)]
    for size in (9, 33, 129):
        for _ in range(32):
            examples.append(([rng.randrange(65) / 8 for _ in range(size)], rng.randrange(65) / 16))
    out = {"schema": "temporal-trajectory-rational-oracle-v1", "seed": 9122032,
           "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "cases": [reference(*case) for case in examples]}
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/temporal_cognition/trajectories.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
