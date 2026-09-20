"""Independent Decimal oracle for saved-member maxima and fractional assignment."""

from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import random
import sys


def evaluate(case):
    with localcontext() as context:
        context.prec = 100
        output = []
        for trajectory in case["current"]:
            if trajectory is None or all(x is None for x in trajectory["point"]):
                output.append(None)
                continue
            scores, provenance = [], []
            for group in case["groups"]:
                choices = []
                if group is not None and group["eligible"]:
                    for member in group["members"]:
                        if member["weight"] <= 0:
                            continue
                        old = member["trajectory"]
                        interval = case["end_sample"] - member["end_sample"]
                        dt = Decimal(interval)/Decimal(case["sample_rate"])
                        f0, e0 = old["point"]
                        f1, e1 = trajectory["point"]
                        distances = []
                        for slope_index, slope in enumerate(old["slopes"][:2 if old["slopes"][1] is not None else 1]):
                            residuals = [None]*3
                            if f0 is not None and f1 is not None:
                                residuals[0] = Decimal(f1)-Decimal(f0)-Decimal(slope or 0)*dt
                                if interval == case["hop"] and slope is not None:
                                    residuals[1] = (Decimal(f1)-Decimal(f0))/dt-Decimal(slope)
                            if e0 is not None and e1 is not None:
                                residuals[2] = Decimal(e1)-Decimal(e0)
                            z = [(r-Decimal(mean))/max(Decimal(sd), Decimal("0.000001"))
                                 for r, mean, sd in zip(residuals, case["means"], case["deviations"]) if r is not None]
                            if z:
                                distance = (sum(v*v for v in z)/Decimal(len(z))).sqrt()
                                if distance <= Decimal(case["distance_limit"]):
                                    distances.append((distance, slope_index))
                        if distances:
                            distance, slope_index = min(distances)
                            score = Decimal(member["weight"])*(-distance*distance).exp()
                            choices.append((score, old["generation"], slope_index, distance, member["end_sample"]))
                if choices:
                    best = min(choices, key=lambda c: (-c[0], c[1], c[2]))
                    scores.append(best[0])
                    provenance.append({"generation": best[1], "slope_index": best[2], "distance": float(best[3]), "end_sample": best[4]})
                else:
                    scores.append(Decimal(0)); provenance.append(None)
            scores.append(Decimal(case["residual_raw"]))
            total = sum(scores)
            output.append({"weights": [float(s/total) for s in scores], "matched_members": provenance})
        return output


if __name__ == "__main__":
    rng = random.Random(9122034)

    def trajectory(generation):
        return {"generation": generation,
                "point": [None if rng.randrange(5) == 0 else 8+rng.randrange(-8, 9)/1024,
                          None if rng.randrange(7) == 0 else -2+rng.randrange(-4, 5)/16],
                "slopes": [None if rng.randrange(4) == 0 else rng.randrange(-4, 5)/16,
                           None if rng.randrange(3) == 0 else rng.randrange(-4, 5)/16]}

    cases = []
    for index in range(64):
        case = {"sample_rate": 1000, "hop": 10, "end_sample": 40,
                "means": [0.0, 0.125, -0.125], "deviations": [0.125, 2.0, 1.0],
                "distance_limit": 1.0, "residual_raw": [0.25, 0.5, 1.0][index % 3],
                "current": [trajectory(1000+i) if rng.randrange(7) else None for i in range(8)],
                "groups": []}
        for group_id in range(7):
            members = [{"trajectory": trajectory(group_id*10+i+1),
                        "end_sample": [10, 20, 30][rng.randrange(3)],
                        "weight": [0.0, 0.25, 0.5, 1.0][rng.randrange(4)]} for i in range(8)]
            case["groups"].append({"generation": 100+group_id, "eligible": rng.randrange(8) > 0, "members": members})
        case["expected"] = evaluate(case)
        cases.append(case)
    output = {"schema": "temporal-group-decimal-oracle-v1", "precision": 100,
              "seed": 9122034, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "group assignment only; constructed coordinates and scales", "cases": cases}
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/temporal_cognition/groups.json")
    target.write_text(json.dumps(output, indent=2)+"\n")
