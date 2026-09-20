"""Decimal probability-space reference for the four shared proposal slots."""

from decimal import Decimal, localcontext
import itertools
import json
from pathlib import Path


def reference(case):
    admitted = {}
    for slot, item in zip([0, 2, 3], case["inputs"]):
        if item is None or item["raw_score"] == 0:
            continue
        if not case["observed"] and case["parent"] is None:
            continue
        admitted.setdefault(item["path"], (slot, Decimal(str(item["raw_score"]))))
    total = sum((score for _, score in admitted.values()), Decimal(0))
    keep = (-Decimal(str(case["dt"])) / 120).exp()
    result = [None] * 4
    for path, (slot, score) in admitted.items():
        result[slot] = dict(path=path, probability=float(keep * score / total))
    result[1] = dict(path=None, probability=float(1 - keep) if admitted else 1.)
    return result


def cases():
    result = []
    for parent, observed, mask in itertools.product([None, 7], [False, True], range(8)):
        inputs = [dict(path=7, raw_score=2.) if parent is not None and mask & 1 else None,
                  dict(path=11, raw_score=3.) if mask & 2 else None,
                  dict(path=13, raw_score=5.) if mask & 4 else None]
        result.append(dict(name=f"support-{parent}-{observed}-{mask}", parent=parent,
                           observed=observed, dt=512 / 48000, inputs=inputs))
    for scores, ids, dt in itertools.product(
        [[1., 1., 1.], [1e308, 1e308, 1e308], [5e-324, 1., 1e308], [0., 0., 0.]],
        [[7, 11, 13], [7, 7, 7], [7, 11, 11]],
        [0., 1e-12, 60., 1e6],
    ):
        result.append(dict(name=f"numeric-{len(result)}", parent=7, observed=True, dt=dt,
                           inputs=[dict(path=key, raw_score=score) for key, score in zip(ids, scores)]))
    return result


def main():
    with localcontext() as ctx:
        ctx.prec = 80
        rows = cases()
        for row in rows:
            row["expected"] = reference(row)
    output = dict(schema="temporal-shared-proposals-v1", reference="80-digit Decimal probability arithmetic",
                  roles=["stay", "unknown", "retrieved", "new_or_contrasting"], tau_seconds=120,
                  cases=rows)
    path = Path(__file__).resolve().parents[1] / "tests/fixtures/temporal_cognition/shared_proposals.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"{len(rows)} cases -> {path}")


if __name__ == "__main__":
    main()
