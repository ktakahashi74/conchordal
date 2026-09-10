#!/usr/bin/env python3
"""Test causal acoustic-event recurrence prediction without driving production."""

import argparse
import collections
import csv
import json
import math
from pathlib import Path
import shutil

import evaluate_rhythm_perception as assay


class RecurrencePredictor:
    """Experimental delay hypotheses; recurrence period is not a tactus label."""

    def __init__(self, dt=assay.HOP / assay.FS, adaptation="forgetting"):
        if adaptation not in ("forgetting", "switching"):
            raise ValueError("unknown adaptation model")
        self.adaptation = adaptation
        self.dt = dt
        self.lags = list(range(math.ceil(.25 / dt), math.floor(2 / dt) + 1))
        self.scores = [0.0] * len(self.lags)
        total = math.fsum(1 / lag for lag in self.lags)
        self.log_priors = [math.log(.5 / lag / total) for lag in self.lags]
        self.prior_weights = [.5] + [.5 / lag / total for lag in self.lags]
        self.weights = self.prior_weights.copy()
        self.score_decay = math.exp(-dt / 6)
        self.count_decay = math.exp(-dt / 4)
        self.counts = [1.0, .001, .001, .001]
        self.history = []
        self.smoothed = []
        self.kernel = [.08, .24, .36, .24, .08]
        self.recent_onsets = 0.0

    def step(self, observed_band):
        if observed_band not in (0, 1, 2, 3):
            raise ValueError("expected silence or one of three acoustic onset bands")
        count = math.fsum(self.counts)
        # A small symmetric prior keeps finite scores before the first onset.
        null = [(x + .01) / (count + .04) for x in self.counts]
        now = len(self.history)
        predictions = []
        for lag in self.lags:
            if now < 3 * lag + 2 or self.recent_onsets < 1.0:
                predictions.append(null)
                continue
            a, b, c = (self.smoothed[now - lag * n] for n in (1, 2, 3))
            mass = [.6 * a[k] + .3 * b[k] + .1 * c[k] for k in range(3)]
            recurrence = [1 - math.fsum(mass)] + mass
            predictions.append([.9 * p + .1 * base for p, base in zip(recurrence, null)])
        if self.adaptation == "forgetting":
            logs = [math.log(.5)] + [prior + score for prior, score in zip(self.log_priors, self.scores)]
            maximum = max(logs)
            weights = [math.exp(x - maximum) for x in logs]
            norm = math.fsum(weights)
            weights = [x / norm for x in weights]
        else:
            weights = self.weights
        mixture = [weights[0] * p for p in null]
        for weight, prediction in zip(weights[1:], predictions):
            for k in range(4):
                mixture[k] += weight * prediction[k]
        best = max(range(len(self.lags)), key=lambda i: weights[i + 1])
        result = {
            "event_probability": 1 - mixture[0],
            "null_event_probability": 1 - null[0],
            "recurrence_weight": 1 - weights[0],
            "best_period_sec": (self.lags[best] * self.dt
                                if self.recent_onsets >= 1.0 and now >= 3 * self.lags[best] + 2
                                and weights[0] < .5 else None),
            "recent_onsets": self.recent_onsets,
            "gain_bits": math.log2(mixture[observed_band] / null[observed_band]),
            "observed_band": observed_band,
        }
        if self.adaptation == "forgetting":
            null_log = math.log(null[observed_band])
            for i, prediction in enumerate(predictions):
                self.scores[i] = (self.scores[i] * self.score_decay
                                  + math.log(prediction[observed_band]) - null_log)
        else:
            posterior = [weights[0] * null[observed_band]] + [
                w * p[observed_band] for w, p in zip(weights[1:], predictions)]
            norm = math.fsum(posterior)
            self.weights = [self.score_decay * p / norm + (1 - self.score_decay) * prior
                            for p, prior in zip(posterior, self.prior_weights)]
        self.counts = [x * self.count_decay + int(i == observed_band)
                       for i, x in enumerate(self.counts)]
        self.recent_onsets = (self.recent_onsets * self.count_decay
                              + int(observed_band != 0))
        self.history.append(observed_band)
        self.smoothed.append([0.0, 0.0, 0.0])
        # Only past centers are finalized. Every queried lag is at least 24 hops.
        center = now - 2
        if center >= 0:
            for offset, weight in zip(range(-2, 3), self.kernel):
                source = center + offset
                if source >= 0 and (band := self.history[source]):
                    self.smoothed[center][band - 1] += weight
        return result


def predict(rows, adaptation="forgetting"):
    predictor = RecurrencePredictor(adaptation=adaptation)
    output = []
    for row in rows:
        band = 0
        if row["onset_time_sec"] is not None:
            band = 1 + max(range(3), key=lambda k: row[("e_low", "e_mid", "e_high")[k]])
        output.append({"time_sec": row["time_sec"], **predictor.step(band)})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adaptation", choices=("forgetting", "switching"), default="forgetting")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for path in (Path(__file__), Path(assay.__file__)):
        sources[path.name] = assay.sha256(path)
        shutil.copy2(path, args.output / path.name)
    manifest = json.loads((args.baseline / "manifest.json").read_text())
    summary = {"status": "experimental_observer_only", "production_connected": False,
               "adaptation": args.adaptation, "adaptation_tau_sec": 6,
               "source_sha256": sources, "baseline": str(args.baseline.resolve()),
               "baseline_manifest_sha256": assay.sha256(args.baseline / "manifest.json"),
               "period_semantics": "recurrence hypothesis, not unique beat/tactus",
               "weight_semantics": "mixture model weight, not perceptual beat confidence",
               "scoring": "each hop predicted before observing its event; gain against adaptive categorical rate",
               "cases": []}
    baseline_summary = json.loads((args.baseline / "summary.json").read_text())
    for case in manifest["cases"]:
        source = args.baseline / case["id"] / "unshaped.csv"
        expected = next(c for c in baseline_summary["cases"] if c["id"] == case["id"])
        if assay.sha256(source) != expected["priors"]["unshaped"]["observation_sha256"]:
            raise ValueError("baseline observations changed")
        rows = predict(assay.read_observations(source), args.adaptation)
        target = args.output / f"{case['id']}.csv"
        with target.open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        windows = []
        for start, end in assay.WINDOWS:
            part = [r for r in rows if start <= r["time_sec"] < end]
            onsets = sum(r["observed_band"] != 0 for r in part)
            gain = math.fsum(r["gain_bits"] for r in part)
            windows.append({"start_sec": start, "end_sec": end, "onsets": onsets,
                            "gain_bits": gain, "gain_bits_per_sec": gain / (end - start),
                            "gain_bits_per_onset": gain / onsets if onsets else None,
                            "mean_recurrence_weight": math.fsum(r["recurrence_weight"] for r in part) / len(part),
                            "leading_period_counts": dict(collections.Counter(
                                "none" if r["best_period_sec"] is None else f'{r["best_period_sec"]:.6f}'
                                for r in part))})
        summary["cases"].append({"id": case["id"], "case": case["case"], "seed": case["seed"],
                                 "input_sha256": assay.sha256(source),
                                 "trace_sha256": assay.sha256(target), "windows": windows})
        print(case["id"], f"late gain={windows[-1]['gain_bits_per_sec']:.3f} bits/s", flush=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
