#!/usr/bin/env python3
"""Score observed waiting with the existing interval forecast, without inferring closure."""

import argparse
import csv
import json
import math
from pathlib import Path

from evaluate_interval_order import IntervalPredictor, MODELS, STEP_SEC, TARGET_WIDTH


def waiting_score(prediction, start_sec, end_sec, fired):
    """Condition one immutable log-interval forecast on survival to start_sec."""
    if (not math.isfinite(start_sec) or not math.isfinite(end_sec)
            or not 0 <= start_sec < end_sec or not isinstance(fired, bool)):
        raise ValueError("expected an ordered finite elapsed-time interval and boolean onset")
    # These constants reproduce the engineering baseline, not a fitted cognitive model.
    means = [-1., *prediction["means_log2_sec"], prediction["context_log2_sec"][-1]]
    widths = [2., *([TARGET_WIDTH] * len(prediction["means_log2_sec"])), TARGET_WIDTH]
    tails = []
    for time in (start_sec, end_sec):
        tails.append([1. if time == 0 else .5 * math.erfc(
            (math.log2(time) - mean) / (width * math.sqrt(2)))
            for mean, width in zip(means, widths)])
    result = {}
    for name in MODELS:
        weights = ([.05, *([0.] * len(prediction["means_log2_sec"])), .95]
                   if name == "persistence" else [.05, *prediction["weights"][name], 0.])
        before, after = [math.fsum(w * s for w, s in zip(weights, tail)) for tail in tails]
        probability = (max(0., before - after) if fired else after) / before if before > 0 else 0.
        probability = min(1., probability)
        result[name] = {"survival_before": before, "survival_after": after,
                        "probability": probability if probability > 0 else None,
                        "loss_bits": -math.log2(probability) if probability > 0 else None,
                        "unavailable_reason": None if probability > 0 else "numerical_tail_resolution"}
    return result


def evaluate(rows):
    model = IntervalPredictor()
    scores, gaps = [], []
    previous_time = None
    pending_origin = None
    for row in rows:
        time = float(row["time_sec"])
        band_value = float(row["observed_band"])
        if not math.isfinite(band_value) or band_value not in (0, 1, 2, 3):
            raise ValueError("expected observed_band in 0, 1, 2, 3")
        fired = band_value != 0
        prediction = model.pending
        event = model.process(time, fired)
        if event is not None and event["kind"] == "gap":
            gaps.append({"available_sec": time, "last_valid_sec": previous_time,
                         "censored_origin_sec": pending_origin})
            pending_origin = None
        elif prediction is not None:
            start = previous_time - prediction["issued_sec"]
            end = time - prediction["issued_sec"]
            scores.append({"available_sec": time, "issued_sec": prediction["issued_sec"],
                           "elapsed_start_sec": start, "elapsed_end_sec": end,
                           "kind": "onset" if fired else "no_detected_onset",
                           "models": waiting_score(prediction, start, end, fired)})
        if event is not None and event["kind"] == "onset":
            pending_origin = model.pending["issued_sec"] if model.pending else None
        previous_time = time
    return {"scope": "Conditional arrival and nonarrival likelihood of detected onsets. Inherits an engineering interval model; not a cognitive fit or closure detector.",
            "observation_step_sec": STEP_SEC,
            "scores": scores, "gaps": gaps,
            "right_censored": {"issued_sec": pending_origin, "last_valid_sec": previous_time}
                              if pending_origin is not None else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.observations.open(newline="") as source:
        result = evaluate(csv.DictReader(source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
