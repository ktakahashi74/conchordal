"""Independent high-precision CDF oracle for the unfitted ordinal link."""

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import random


def generate():
    rng = random.Random(9122030)
    cases = [(eta, [-2.0, -0.5, 0.5, 2.0]) for eta in [-1000.0, -50.0, 0.0, 50.0, 1000.0]]
    cases += [(eta, [0.0, 1e-12, 2e-12, 3e-12]) for eta in [-1000.0, 0.0, 1000.0]]
    for _ in range(48):
        cases.append((rng.uniform(-25, 25), sorted(rng.uniform(-10, 10) for _ in range(4))))
    output = []
    with localcontext() as ctx:
        # Direct CDF subtraction remains resolved even in the ±1000 tails.
        ctx.prec = 750
        one = Decimal(1)
        for eta, cuts in cases:
            cumulative = [Decimal(0)]
            for cut in cuts:
                x = Decimal.from_float(cut) - Decimal.from_float(eta)
                cumulative.append(one / (one + (-x).exp()))
            cumulative.append(one)
            log_p = [float((hi - lo).ln()) for lo, hi in zip(cumulative, cumulative[1:])]
            output.append({"eta": eta, "cutpoints": cuts, "log_probabilities": log_p})
    mixtures = []
    first = [0.8, 0.05, 0.05, 0.05, 0.05]
    last = list(reversed(first))
    prior = [0.5, 0.2, 0.1, 0.1, 0.1]
    inputs = [
        ([(0.25, first), (0.25, last), (0.5, None)], 0.8, prior, t)
        for t in [0.5, 1.0, 2.0, 10.0]
    ]
    inputs += [([], 1.0, prior, 2.0), ([(1.0, None)], 1.0, prior, 2.0),
               ([(1.0, first)], 0.0, prior, 2.0),
               ([(0.125, first), (0.125, first), (0.25, last)], 0.8, prior, 2.0)]
    for _ in range(32):
        masses = [rng.randrange(1, 10) for _ in range(5)]
        rows = []
        for i, mass in enumerate(masses):
            counts = [rng.randrange(1, 10) for _ in range(5)]
            rows.append((mass / sum(masses), None if i == 4 else [c / sum(counts) for c in counts]))
        inputs.append((rows, rng.random(), prior, math.exp(rng.uniform(-2, 2))))
    with localcontext() as ctx:
        ctx.prec = 100
        decimal = Decimal.from_float
        for rows, coverage, base, temperature in inputs:
            supported = sum((decimal(w) for w, p in rows if p is not None), Decimal(0))
            reliability = decimal(coverage) * supported
            mixture = [sum((decimal(w) * decimal(p[k]) for w, p in rows if p is not None), Decimal(0)) for k in range(5)]
            if reliability:
                powers = [(p / supported) ** (1 / decimal(temperature)) for p in mixture]
                norm = sum(powers)
                final = [reliability * p / norm + (1 - reliability) * decimal(b) for p, b in zip(powers, base)]
            else:
                final = list(map(decimal, base))
            mixtures.append({
                "rows": [{"weight": w, "probabilities": p} for w, p in rows],
                "observed_coverage": coverage, "training_prior": base, "temperature": temperature,
                "supported_mass": float(supported), "reported_support_mass": float(reliability),
                "log_probabilities": [float(p.ln()) for p in final],
            })
    return {
        "schema": "temporal-rating-decimal-oracle-v2",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "decimal_precision": {"ordinal": 750, "mixtures": 100},
        "seed": 9122030,
        "fitted_human_parameters": False,
        "ordinal": output,
        "mixtures": mixtures,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(generate(), indent=2, allow_nan=False) + "\n")
