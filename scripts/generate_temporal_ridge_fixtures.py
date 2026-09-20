"""Decimal oracle for masked ridge residuals and candidate secants."""

from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import random
import sys


def reference(row):
    d = lambda x: None if x is None else Decimal(x)
    old, new = row["old"], row["new"]
    f0, f1 = d(old[0]), d(new[0])
    ell0, ell1 = d(old[1]), d(new[1])
    slope = d(row["slope"])
    with localcontext() as context:
        context.prec = 100
        interval = row["end_sample"] - row["old_end_sample"]
        dt = Decimal(interval) / Decimal(row["sample_rate"])
        secant = (f1-f0)/dt if f0 is not None and f1 is not None and interval == row["hop"] else None
        residuals = [(f1-f0-(slope or Decimal(0))*dt) if f0 is not None and f1 is not None else None,
                     secant-slope if secant is not None and slope is not None else None,
                     ell1-ell0 if ell1 is not None and ell0 is not None else None]
        z = [(r-Decimal(mean))/max(Decimal(sd), Decimal("0.000001"))
             for r, mean, sd in zip(residuals, row["means"], row["deviations"]) if r is not None]
        rms = (sum(v*v for v in z)/Decimal(len(z))).sqrt() if z else None
        return {"distance": None if rms is None else float(rms),
                "secant": None if not z or secant is None else float(secant),
                "available_coordinates": len(z)}


if __name__ == "__main__":
    rng = random.Random(9122033)
    rows = []
    for i in range(160):
        value = lambda: rng.randrange(-80, 81)/8
        row = {"old": [value(), value()], "new": [value(), value()],
               "slope": None if i % 4 == 0 else value(),
               "sample_rate": 48000, "hop": 512, "old_end_sample": 512,
               "end_sample": 1024 if i % 3 else 4096,
               "means": [rng.randrange(-4, 5)/8 for _ in range(3)],
               "deviations": [rng.randrange(0, 17)/8 for _ in range(3)]}
        if i % 5 == 0: row["old"][0] = None
        if i % 7 == 0: row["new"][1] = None
        row["expected"] = reference(row)
        rows.append(row)
    output = {"schema": "temporal-ridge-decimal-oracle-v1", "precision": 100, "seed": 9122033,
              "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "individual continuity distances/secants, independent of lifecycle and human calibration",
              "cases": rows}
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/temporal_cognition/ridges.json")
    path.write_text(json.dumps(output, indent=2) + "\n")
