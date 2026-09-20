"""Fit frozen descriptive body medoids from registered, actually rendered records.

This is an offline diagnostic model. It does not certify counterfactual ranges or
ordinal transfer. No model update runs in the instrument.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def fit(corpus, count):
    records = sorted(corpus["records"], key=lambda row: row["record_id"])
    if not 1 <= count <= min(16, len(records)):
        raise ValueError("1..16 medoids and at least that many records required")
    ids = [row["record_id"] for row in records]
    if len(set(ids)) != len(ids) or any(not key or len(key) > 128 or not key.isascii() for key in ids):
        raise ValueError("unique nonempty ASCII record IDs of at most 128 bytes required")
    for row in records:
        values = row["descriptor"]["raw_values"]
        mask = row["descriptor"]["mask"]
        if len(values) != 6 or not 1 <= mask <= 63 or any(not math.isfinite(value) for value in values):
            raise ValueError("each record needs finite values and at least one supported coordinate")
    columns = [[row["descriptor"]["raw_values"][i] for row in records if row["descriptor"]["mask"] & (1 << i)] for i in range(6)]
    if any(not column for column in columns):
        raise ValueError("the registered corpus must support every scale coordinate")
    means = [math.fsum(column) / len(column) for column in columns]
    deviations = [math.sqrt(math.fsum((value - mean)**2 for value in column) / len(column))
                  for column, mean in zip(columns, means)]
    values = [[(value - means[i]) / max(deviations[i], 1e-6) if row["descriptor"]["mask"] & (1 << i) else None
               for i, value in enumerate(row["descriptor"]["raw_values"])] for row in records]
    if any(not math.isfinite(value) for vector in values for value in vector if value is not None):
        raise ValueError("non-finite standardized coordinate")
    distances = []
    common = []
    for left in values:
        distances.append([])
        common.append([])
        for right in values:
            differences = [a-b for a, b in zip(left, right) if a is not None and b is not None]
            distances[-1].append(math.hypot(*differences) / math.sqrt(len(differences)) if differences else math.inf)
            common[-1].append(len(differences))
    medoids = [0]
    while len(medoids) < count:
        remaining = [i for i in range(len(records)) if i not in medoids]
        medoids.append(min(remaining, key=lambda i: (-min(distances[i][m] for m in medoids), ids[i])))
    medoids.sort()
    initial = medoids.copy()
    objective = math.fsum(min(row[m] for m in medoids) for row in distances)
    history = [objective]
    passes = 0
    stop = "pass_limit"
    for _ in range(100):
        passes += 1
        selected = set(medoids)
        best = objective
        best_set = medoids
        # Cache the objective without each removed medoid; no approximation or sampling.
        for removed in medoids:
            others = [m for m in medoids if m != removed]
            without = [min((row[m] for m in others), default=math.inf) for row in distances]
            for added in range(len(records)):
                if added in selected:
                    continue
                proposed = sorted(others + [added])
                value = math.fsum(min(old, distances[i][added]) for i, old in enumerate(without))
                if value < best or (value == best and proposed < best_set):
                    best, best_set = value, proposed
        if not best < objective or objective - best <= 1e-6:
            stop = "improvement_at_most_1e-6"
            break
        medoids, objective = best_set, best
        history.append(objective)
    if not math.isfinite(objective):
        raise ValueError("medoids cannot cover all records through a common observable coordinate")
    assignments = []
    for i, row in enumerate(distances):
        selected = min(medoids, key=lambda m: (row[m], -common[i][m], ids[m]))
        assignments.append(dict(record_id=ids[i], medoid_id=ids[selected], distance=row[selected],
                                common_coordinates=common[i][selected], compatible=row[selected] <= .25))
    model = dict(schema="temporal-body-prototypes-v1", status="diagnostic_not_counterfactual_or_ordinal_calibration",
                 corpus_sha256=hashlib.sha256(json.dumps(corpus, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
                 acquisition=corpus["acquisition"], accent_scales=corpus["accent_scales"],
                 means=means, deviations=deviations, coordinate_counts=list(map(len, columns)),
                 distance="RMS over common observed coordinates; distance, descending common count, lexical record ID",
                 compatibility_threshold=.25, initial_medoids=[ids[i] for i in initial],
                 medoids=[dict(record_id=ids[i], raw_values=records[i]["descriptor"]["raw_values"], mask=records[i]["descriptor"]["mask"]) for i in medoids],
                 objective=objective, objective_history=[value if math.isfinite(value) else None for value in history],
                 passes=passes, stop_reason=stop, assignments=assignments, records=records)
    model["model_version"] = hashlib.sha256(json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return model


def configuration(model):
    if len(model["medoids"]) > 8:
        raise ValueError("more than eight medoids is an offline sensitivity assay, not a runtime model")
    encoded = lambda value: json.dumps(value, allow_nan=False)
    lines = ["[temporal_body]", "means = " + encoded(model["means"]), "deviations = " + encoded(model["deviations"]),
             "accent_means = " + encoded(model["accent_scales"]["means"]),
             "accent_deviations = " + encoded(model["accent_scales"]["deviations"]), "", "[temporal_body_prototypes]",
             "model_version = " + encoded(model["model_version"])]
    lines += [f"{key} = {encoded(model['acquisition'][key])}" for key in ("sample_rate", "nfft", "hop_size")]
    lines += [f"{key} = {encoded(model[key])}" for key in ("means", "deviations")]
    lines += ["accent_means = " + encoded(model["accent_scales"]["means"]), "accent_deviations = " + encoded(model["accent_scales"]["deviations"])]
    for medoid in model["medoids"]:
        lines += ["", "[[temporal_body_prototypes.medoids]]"]
        lines += [f"{key} = {encoded(medoid[key])}" for key in ("record_id", "raw_values", "mask")]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    model = fit(json.loads(args.corpus.read_text()), args.count)
    text = configuration(model) if args.config else None
    args.output.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")
    if args.config:
        args.config.write_text(text)
    print(json.dumps({key: model[key] for key in ("model_version", "objective", "passes", "stop_reason")}))
