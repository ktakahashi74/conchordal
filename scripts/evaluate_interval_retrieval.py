#!/usr/bin/env python3
"""Research-only continuous interval retrieval with explicit, unfitted parameters.

Time/item expiry follows the PPM-Decay mechanism, but continuous cue kernels and
prior competition are new hypotheses, not a reproduction of its symbolic model.
No score labels or future arrival times enter an issued forecast. Stored traces
are unbounded; this is not a real-time implementation or a closure detector.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def score_waiting(prediction, start_sec, end_sec, fired):
    if (not math.isfinite(start_sec) or not math.isfinite(end_sec)
            or not 0 <= start_sec < end_sec or not isinstance(fired, bool)):
        raise ValueError("expected ordered finite elapsed times and boolean onset")
    tails = []
    for elapsed in (start_sec, end_sec):
        tails.append([1. if elapsed == 0 else .5 * math.erfc(
            (math.log2(elapsed) - mean) / (width * math.sqrt(2)))
            for mean, width in zip(prediction["means_log2_sec"], prediction["widths_log2_sec"])])
    scores = {}
    for name, weights in prediction["weights"].items():
        before, after = [math.fsum(weight * tail for weight, tail in zip(weights, values)) for values in tails]
        probability = (max(0., before - after) if fired else after) / before if before > 0 else 0.
        probability = min(1., probability)
        scores[name] = {"probability": probability if probability > 0 else None,
                        "loss_bits": -math.log2(probability) if probability > 0 else None,
                        "unavailable_reason": None if probability > 0 else "numerical_tail_resolution"}
    return scores


class IntervalRetrieval:
    def __init__(self, parameters, *, step_sec):
        p = dict(parameters)
        required = {"max_context", "buffer_items", "buffer_sec", "post_buffer_weight",
                    "decay_half_life_sec", "context_width_log2", "target_width_log2",
                    "prior_mean_log2", "prior_width_log2", "prior_mass"}
        if set(p) != required or any(not math.isfinite(float(value)) for value in p.values()):
            raise ValueError("expected all explicit finite model parameters")
        if (not isinstance(p["max_context"], int) or p["max_context"] < 0
                or not isinstance(p["buffer_items"], int) or p["buffer_items"] < 1
                or not 0 < p["post_buffer_weight"] <= 1
                or any(p[key] <= 0 for key in ("buffer_sec", "decay_half_life_sec", "context_width_log2",
                                              "target_width_log2", "prior_width_log2", "prior_mass"))
                or not math.isfinite(step_sec) or step_sec <= 0):
            raise ValueError("invalid memory, kernel, prior or observation parameters")
        self.parameters, self.step_sec = p, step_sec
        self.intervals, self.times = [], []
        self.last_observation = self.last_onset = self.pending = None

    def retrieve(self, time_sec):
        if (not math.isfinite(time_sec) or time_sec < 0
                or self.last_observation is not None and time_sec < self.last_observation):
            raise ValueError("retrieval cannot precede available observations")
        if not self.intervals:
            return None
        p = self.parameters
        values, times = np.asarray(self.intervals), np.asarray(self.times)
        count = len(values)
        weights = np.zeros(count + 1)
        weights[0] = 1.
        result = {"issued_sec": time_sec, "available_intervals": count,
                  "means_log2_sec": [p["prior_mean_log2"], *self.intervals],
                  "widths_log2_sec": [p["prior_width_log2"], *([p["target_width_log2"]] * count)],
                  "weights": {}, "support": {}}
        for order in range(p["max_context"] + 1):
            evidence = np.zeros(count + 1)
            if order < count and order < p["buffer_items"]:
                ends = np.arange(order, count)
                begins = ends - order
                expiry = times[begins] + p["buffer_sec"]
                item_expiry = begins + p["buffer_items"]
                known = item_expiry < count
                expiry[known] = np.minimum(expiry[known], times[item_expiry[known]])
                elapsed = time_sec - expiry
                retention = np.where(elapsed < 0, 1., p["post_buffer_weight"]
                                     * np.exp2(-np.maximum(elapsed, 0) / p["decay_half_life_sec"]))
                # An episode must fit in the strong buffer when it is encoded.
                retention[times[ends] - times[begins] >= p["buffer_sec"]] = 0.
                if order:
                    context = values[-order:]
                    distance = np.zeros(len(ends))
                    for offset in range(order):
                        distance += ((values[begins + offset] - context[offset]) / p["context_width_log2"]) ** 2
                    retention *= np.exp(-.5 * distance)
                    if time_sec - times[-order] > p["buffer_sec"]:
                        retention[:] = 0.
                evidence[ends + 1] = retention
            support = float(evidence.sum())
            weights = (evidence + p["prior_mass"] * weights) / (support + p["prior_mass"])
            name = f"context_{order}"
            result["weights"][name] = weights.tolist()
            result["support"][name] = support
        return result

    def process(self, time_sec, fired):
        if (not math.isfinite(time_sec) or time_sec < 0 or not isinstance(fired, bool)
                or self.last_observation is not None and time_sec <= self.last_observation):
            raise ValueError("expected increasing finite observation times and boolean onset")
        previous = self.last_observation
        self.last_observation = time_sec
        if previous is not None and not math.isclose(time_sec - previous, self.step_sec, rel_tol=0, abs_tol=1e-7):
            censored = self.pending
            self.intervals.clear()
            self.times.clear()
            self.last_onset = self.pending = None
            return {"kind": "gap", "available_sec": time_sec, "last_valid_sec": previous,
                    "censored_forecast": censored}
        prediction = self.pending
        score = (score_waiting(prediction, previous - prediction["issued_sec"], time_sec - prediction["issued_sec"], fired)
                 if prediction is not None else None)
        result = {"kind": "onset" if fired else "no_detected_onset", "available_sec": time_sec,
                  "issued_sec": prediction["issued_sec"] if prediction else None, "score": score}
        if fired:
            interval = time_sec - self.last_onset if self.last_onset is not None else None
            if interval is not None:
                self.intervals.append(math.log2(interval))
                self.times.append(time_sec)
            self.last_onset = time_sec
            self.pending = self.retrieve(time_sec)
            result.update(interval_sec=interval, next_forecast=self.pending)
        return result


def evaluate(rows, parameters, *, step_sec):
    model = IntervalRetrieval(parameters, step_sec=step_sec)
    records, cumulative = [], {}
    waiting_windows = 0
    for row in rows:
        band = float(row["observed_band"])
        if band not in (0, 1, 2, 3):
            raise ValueError("expected observed_band in 0, 1, 2, 3")
        event = model.process(float(row["time_sec"]), bool(band))
        if event["kind"] == "gap":
            event["censored_loss_bits"] = cumulative
            cumulative = {}
        elif event["score"] is not None:
            for name, score in event["score"].items():
                old, increment = cumulative.get(name, 0.), score["loss_bits"]
                cumulative[name] = old + increment if old is not None and increment is not None else None
            if event["kind"] == "onset":
                event["interval_loss_bits"] = cumulative
                cumulative = {}
            else:
                waiting_windows += 1
        if event["kind"] != "no_detected_onset":
            records.append(event)
    return {"scope": "Continuous acoustic-interval retrieval hypothesis; explicit unfitted parameters, unbounded research storage, no closure or generation coupling.",
            "parameters": dict(parameters), "step_sec": step_sec, "events": records,
            "scored_waiting_windows": waiting_windows,
            "right_censored": {"forecast": model.pending, "last_valid_sec": model.last_observation,
                               "observed_loss_bits": cumulative} if model.pending else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("parameters", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--step-sec", type=float, required=True)
    args = parser.parse_args()
    with args.observations.open(newline="") as source:
        result = evaluate(csv.DictReader(source), json.loads(args.parameters.read_text()), step_sec=args.step_sec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
