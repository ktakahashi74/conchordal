"""Independent high-precision oracle for I10's hypothetical consequence mixture.

Only frozen numerical inputs are used. No audio, fit or passive-state update is
implied. Decimal probability-space normalization differs from the Rust log-space
implementation; fixed cases additionally pin the support and backoff semantics.
"""

import hashlib
import json
import random
from decimal import Decimal, localcontext
from pathlib import Path


def project(case):
    with localcontext() as context:
        context.prec = 80
        d = lambda x: Decimal(str(x))
        rows = case["rows"]
        eligible = [row["weight"] > 0 and row["delta"] is not None for row in rows]
        mass = sum((d(row["weight"]) for row, ok in zip(rows, eligible) if ok), Decimal(0))
        count = sum(eligible)
        largest = max((d(row["delta"]) for row, ok in zip(rows, eligible) if ok), default=Decimal(0))
        weights = []
        for row, ok in zip(rows, eligible):
            delta = d(row["delta"]) - largest if ok else Decimal(-10001)
            weights.append(d(row["weight"]) * delta.exp() if ok and delta > -10000 else Decimal(0))
        total = sum(weights)
        p = [w / total if total else Decimal(0) for w in weights]
        entropy = -sum((value * value.ln() for value in p if value), Decimal(0))
        uncertainty = 1 - mass + (mass * entropy / d(count).ln() if count > 1 else 0)
        heads = []
        for head in range(3):
            groups = [row["head_groups"][head] if "head_groups" in row
                      else [dict(weight=1., probabilities=row["heads"][head])] for row in rows]
            supported = sum((mass * value * d(group["weight"])
                             for value, entries in zip(p, groups) for group in entries
                             if group["probabilities"] is not None), Decimal(0))
            coverage = case["coverage"][head] if isinstance(case["coverage"], list) else case["coverage"]
            reported = supported * d(coverage)
            prior = list(map(d, case["priors"][head]))
            prediction = prior
            if supported:
                mixture = [sum((mass * value * d(group["weight"]) * d(group["probabilities"][category])
                                for value, entries in zip(p, groups) for group in entries
                                if group["probabilities"] is not None), Decimal(0)) / supported
                           for category in range(5)]
                tempered = [value ** (1 / d(case["temperatures"][head])) if value else Decimal(0) for value in mixture]
                prediction = [reported * value / sum(tempered) + (1 - reported) * base
                              for value, base in zip(tempered, prior)]
            heads.append(dict(probabilities=list(map(float, prediction)), supported=float(supported),
                              reported=float(reported), expected=float(sum(d(i) / 4 * value for i, value in enumerate(prediction)))))
        return dict(weights=list(map(float, p)), retained=float(mass), unknown=float(1-mass), count=count,
                    entropy=float(entropy), uncertainty=float(uncertainty), heads=heads)


def generate():
    uniform = [.2] * 5
    low = [1., 0., 0., 0., 0.]
    high = [0., 0., 0., 0., 1.]
    cases = []

    def add(name, rows, coverage=.8, temperatures=(1., 2., .5)):
        case = dict(name=name, rows=[dict(weight=w, delta=score, heads=heads) for w, score, heads in rows],
                    coverage=coverage, priors=[uniform, [.1, .2, .4, .2, .1], [.3, .2, .2, .2, .1]],
                    temperatures=list(temperatures))
        case["expected"] = project(case)
        cases.append(case)

    add("empty", [])
    add("equal", [(.4, 0., [low, high, uniform]), (.4, 0., [high, low, uniform])])
    add("concentrated", [(.4, 1000., [high]*3), (.4, -1000., [low]*3)])
    add("one_resolved", [(.4, 0., [high]*3), (.5, None, [low]*3)])
    add("all_unknown", [(.4, None, [high]*3), (.5, None, [low]*3)])
    add("head_missing", [(.4, 0., [high, None, None]), (.4, 1., [None, low, None])])
    add("zero_weight", [(0., 1e308, [high]*3), (.6, 0., [low]*3)])
    add("large_common_score", [(.1, 1e308, [high]*3), (.7, 1e308, [low]*3)])
    add("opposite_extreme_score", [(.1, 1e308, [high]*3), (.7, -1e308, [low]*3)])
    add("tiny_retained", [(1e-250, 0., [high]*3)])
    add("coverage_zero", [(.4, 0., [high]*3)], coverage=0.)
    add("full_mass_full_coverage", [(1., 0., [high]*3)], coverage=1.)
    rng = random.Random(101014)
    for index in range(48):
        count = [1, 2, 7, 16, 64, 128][index % 6]
        units = [rng.randrange(1, 10) for _ in range(count)]
        denominator = sum(units) + rng.randrange(1, 20)
        rows = []
        for unit in units:
            heads = []
            for _ in range(3):
                probabilities = [rng.randrange(1, 20) for _ in range(5)]
                heads.append([value / sum(probabilities) for value in probabilities] if rng.random() > .25 else None)
            rows.append((unit / denominator, rng.uniform(-80, 80) if rng.random() > .25 else None, heads))
        add(f"seeded_{index}", rows, coverage=rng.random(), temperatures=(.3, 1., 3.))
    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests/fixtures/temporal_cognition/consequence.json"
    data = (json.dumps(dict(version=1, oracle="80-digit decimal probability-space", cases=cases),
                       indent=2, allow_nan=False) + "\n").encode()
    fixture.write_bytes(data)
    print(f"{len(cases)} cases; sha256={hashlib.sha256(data).hexdigest()}")


def generate_grouped():
    rng = random.Random(101018)
    cases = []
    for case_index in range(10):
        rows = []
        for row_index, weight in enumerate([.35, .2, .1, .05]):
            heads = []
            for head in range(3):
                groups = []
                for group, (group_weight, local_support) in enumerate(zip([.6, .3, .1], [1., .5, .25])):
                    units = [rng.randrange(1, 20) for _ in range(5)]
                    missing = (case_index + row_index + head + group) % 4 == 0 or case_index == 9
                    groups.append(dict(weight=group_weight * local_support,
                                       probabilities=None if missing else [u / sum(units) for u in units]))
                heads.append(groups)
            rows.append(dict(weight=weight, delta=None if row_index == 2 else [-.4, 1.1, 0., .3][row_index],
                             head_groups=heads))
        case = dict(name=f'grouped-{case_index}', rows=rows,
                    issue_group_weights=[.6, .3, .1], conditional_group_support=[1., .5, .25],
                    coverage=[0. if case_index == 0 else .8, .3, 1.],
                    priors=[[.2]*5, [.1, .2, .4, .2, .1], [.3, .2, .2, .2, .1]],
                    temperatures=[.3, 1., 3.])
        case['expected'] = project(case)
        cases.append(case)
    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests/fixtures/temporal_cognition/consequence_groups.json"
    data = (json.dumps(dict(version=1, oracle="80-digit decimal flattened group/path probability-space", cases=cases),
                       indent=2, allow_nan=False) + "\n").encode()
    fixture.write_bytes(data)
    print(f"{len(cases)} grouped cases; sha256={hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    generate()
    generate_grouped()
