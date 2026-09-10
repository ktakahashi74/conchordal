#!/usr/bin/env python3
"""Reanalyse the published PPM-Decay task without treating it as an interval-memory fit."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np


def evaluate(response_path, models_path, participant_statistic="median"):
    if participant_statistic not in ("median", "mean"):
        raise ValueError("expected participant statistic median or mean")
    with Path(response_path).open(newline="") as source:
        rows = [{key.strip(): value.strip() for key, value in row.items()}
                for row in csv.DictReader(source, delimiter="\t")]
    required = {"subj", "block", "speed", "condi", "RTs", "response", "correct"}
    if not rows or not required <= rows[0].keys():
        raise ValueError("expected the published response-table columns")
    counts = {"raw_trials": len(rows), "raw_participants": len({r['subj'] for r in rows})}
    data = []
    for row in rows:
        subject, speed = int(row["subj"]), int(row["speed"])
        if speed not in range(1, 7):
            raise ValueError("expected speed condition in 1..6")
        if subject in (7, 25):
            continue
        data.append({"subject": subject, "block": int(row["block"]),
                     "cycle_tones": 10 if speed <= 3 else 20,
                     "tone_ms": (25, 50, 75)[(speed - 1) % 3],
                     "condition": int(row["condi"]), "response": int(row["response"]),
                     "included": bool(int(row["correct"])), "rt_sec": float(row["RTs"])})
    counts["after_subject_exclusions"] = len(data)
    data = [row for row in data if row["included"]]
    counts["after_trial_flags"] = len(data)
    steps = defaultdict(list)
    for row in data:
        if row["condition"] == 4 and row["response"] == 1:
            steps[row["subject"], row["block"]].append(row["rt_sec"])
    groups = defaultdict(list)
    for row in data:
        if row["condition"] == 2:
            reference = steps[row["subject"], row["block"]]
            if not reference or not np.all(np.isfinite(reference)):
                raise ValueError("missing finite STEP baseline for a participant block")
            row["rt_norm_sec"] = row["rt_sec"] - float(np.mean(reference))
        else:
            row["rt_norm_sec"] = row["rt_sec"]
        groups[row["subject"], row["cycle_tones"], row["tone_ms"], row["condition"]].append(row)
    retained = []
    for group in groups.values():
        finite = [r["rt_norm_sec"] for r in group if math.isfinite(r["rt_norm_sec"])]
        mean = float(np.mean(finite)) if finite else math.nan
        sd = float(np.std(finite, ddof=1)) if len(finite) >= 2 else math.nan
        for row in group:
            value = row["rt_norm_sec"]
            if math.isnan(value) or sd > 0 and abs((value - mean) / sd) < 2:
                retained.append(row)
    counts["after_outlier_exclusions"] = len(retained)
    hit_groups = defaultdict(list)
    for row in retained:
        if row["condition"] == 2 and row["response"] == 1:
            lag = 1000 * row["rt_norm_sec"] / row["tone_ms"] - row["cycle_tones"]
            hit_groups[row["subject"], row["cycle_tones"], row["tone_ms"]].append(lag)
    counts["retained_randreg_hits"] = sum(len(values) for values in hit_groups.values())
    participants = sorted({key[0] for key in hit_groups})
    conditions = [(n, t) for n in (10, 20) for t in (25, 50, 75)]
    if len(participants) < 2 or set(hit_groups) != {(p, n, t) for p in participants for n, t in conditions}:
        raise ValueError("incomplete participant-by-condition design")
    aggregate = np.median if participant_statistic == "median" else np.mean
    matrix = np.array([[aggregate(hit_groups[p, n, t]) for n, t in conditions] for p in participants])
    means = matrix.mean(axis=0)
    summary = [{"cycle_tones": n, "tone_ms": t, "cycle_sec": n * t / 1000,
                "participants": len(participants), "trials": sum(len(hit_groups[p, n, t]) for p in participants),
                "mean_participant_lag_tones": float(means[i]),
                "mean_participant_lag_sec": float(means[i] * t / 1000),
                "median_participant_lag_tones": float(np.median(matrix[:, i])),
                "se_participant_lags": float(np.std(matrix[:, i], ddof=1) / math.sqrt(len(participants)))}
               for i, (n, t) in enumerate(conditions)]

    # Paired contrasts retain each participant as the sampling unit.
    contrasts = {}
    for unit in ("tones", "seconds"):
        values = matrix if unit == "tones" else matrix * np.array([t / 1000 for _, t in conditions])
        contrasts.update({(name, unit): value for name, value in {
            "equal_cycle_duration_20x25_minus_10x50": values[:, 3] - values[:, 1],
            "rate_effect_10_tones_75_minus_25": values[:, 2] - values[:, 0],
            "rate_effect_20_tones_75_minus_25": values[:, 5] - values[:, 3],
            "rate_by_load_interaction": (values[:, 5] - values[:, 3]) - (values[:, 2] - values[:, 0])}.items()})
    rng = np.random.default_rng(20260907)
    indices = rng.integers(0, len(participants), size=(100_000, len(participants)))
    normal = NormalDist()
    contrast_results = {}
    for (name, unit), values in contrasts.items():
        estimate = float(np.mean(values))
        bootstrap = values[indices].mean(axis=1)
        rank = float(np.mean(bootstrap < estimate))
        z0 = normal.inv_cdf(min(1 - 1 / (2 * len(bootstrap)), max(1 / (2 * len(bootstrap)), rank)))
        jackknife = (np.sum(values) - values) / (len(values) - 1)
        centered = jackknife.mean() - jackknife
        denominator = 6 * float(np.sum(centered ** 2)) ** 1.5
        acceleration = float(np.sum(centered ** 3)) / denominator if denominator > 0 else 0.
        quantiles = [normal.cdf(z0 + (z0 + normal.inv_cdf(q)) /
                               (1 - acceleration * (z0 + normal.inv_cdf(q)))) for q in (.025, .975)]
        contrast_results[name + "_" + unit] = {"unit": unit, "mean_lag_difference": estimate,
                                               "bca_95": np.quantile(bootstrap, quantiles).tolist()}

    with Path(models_path).open(newline="") as source:
        model_groups = defaultdict(list)
        for row in csv.DictReader(source):
            model_groups[int(row["i"])].append(row)
    comparisons = []
    for number, model_rows in sorted(model_groups.items()):
        lookup = {(int(row["alphabet_size"]), int(row["tone_len_ms"])): row for row in model_rows}
        if len(lookup) != len(model_rows) or set(lookup) != set(conditions):
            raise ValueError("expected exactly six published conditions per model")
        predicted = np.array([float(lookup[key]["mean"]) for key in conditions])
        comparison = {"model": number, "label": model_rows[0]["label"],
                      "rmse_lag_tones": float(np.sqrt(np.mean((predicted - means) ** 2))),
                      "mean_predicted_lag_tones": predicted.tolist(),
                      "excluded_model_lags": sum(int(row["error_count"]) for row in model_rows)}
        comparisons.append(comparison)
    return {"scope": "Reconstructed published participant preprocessing and compared archived model means; no new model simulation, independent validation, or interval-memory fit.",
            "counts": counts, "conditions": summary,
            "participant_values": [{"subject": p, "lag_tones": matrix[i].tolist()} for i, p in enumerate(participants)],
            "contrasts": contrast_results,
            "bootstrap": {"unit": "participant", "samples": 100_000, "seed": 20260907, "method": "BCa; NumPy draws differ from the original R draws"},
            "archived_model_comparisons": comparisons,
            "participant_statistic": participant_statistic,
            "aggregation": "Mean of participant summaries. Upstream fit code uses within-participant medians; the paper describes means. These routes are reported separately.",
            "inputs": {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (response_path, models_path)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", type=Path)
    parser.add_argument("model_summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--participant-statistic", choices=("median", "mean"), default="median")
    args = parser.parse_args()
    result = evaluate(args.responses, args.model_summary, args.participant_statistic)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("counts", "conditions", "contrasts", "archived_model_comparisons")}, indent=2))


if __name__ == "__main__":
    main()
