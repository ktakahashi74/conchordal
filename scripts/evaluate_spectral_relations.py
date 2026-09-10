#!/usr/bin/env python3
"""Prospective regional NSGT expectations conditioned on native frequency history.

This compares point forecasts. Regions are observations, not identified sources;
conditional prediction gain is not a causal effect or perceptual grouping score.
"""

import argparse
import copy
from collections import deque
import gzip
import itertools
import json
import math
from pathlib import Path

import numpy as np

from evaluate_prediction_residuals import CenteredRidge


VARIANTS = ("own", "partner", "background", "both", "partner_current", "both_current")


def regions(frequencies, mean_power, maximum):
    frequencies = np.asarray(frequencies, dtype=float)
    power = np.asarray(mean_power, dtype=float)
    if (frequencies.ndim != 1 or power.shape != frequencies.shape
            or not np.isfinite(frequencies).all() or np.any(np.diff(frequencies) <= 0)
            or np.any(frequencies <= 0) or not np.isfinite(power).all() or np.any(power < 0)
            or not isinstance(maximum, int) or maximum < 2):
        raise ValueError("expected an ordered frequency grid and nonnegative observed power")
    # Peak candidates and spacing depend only on the observed prefix.
    peaks = np.flatnonzero((power > np.r_[0., power[:-1]])
                          & (power >= np.r_[power[1:], 0.]))
    selected = []
    for index in sorted(peaks, key=lambda i: (-power[i], i)):
        width = 24.7 + frequencies[index] / 9.265
        if all(abs(frequencies[index] - frequencies[j])
               > width + 24.7 + frequencies[j] / 9.265 for j in selected):
            selected.append(int(index))
        if len(selected) == maximum:
            break
    selected.sort()
    if len(selected) < 2:
        return None
    masks = np.array([np.abs(frequencies - frequencies[i])
                      <= 24.7 + frequencies[i] / 9.265 for i in selected])
    return selected, masks / masks.sum(axis=1, keepdims=True)


class SpectralRelations:
    def __init__(self, weights, reference_rms, horizon_samples, hop_samples, ridge=1.):
        self.weights = np.array(weights, dtype=float, copy=True)
        if (self.weights.ndim != 2 or len(self.weights) < 2
                or not np.isfinite(self.weights).all() or np.any(self.weights < 0)
                or not np.allclose(self.weights.sum(axis=1), 1.)
                or not math.isfinite(reference_rms) or reference_rms <= 0
                or not isinstance(hop_samples, int) or hop_samples <= 0
                or not isinstance(horizon_samples, int) or horizon_samples < hop_samples
                or horizon_samples % hop_samples):
            raise ValueError("invalid regions, observed reference or query clock")
        self.reference_rms = reference_rms
        self.horizon_samples, self.hop_samples = horizon_samples, hop_samples
        self.pairs = list(itertools.permutations(range(len(self.weights)), 2))
        background = np.array([~((self.weights[a] > 0) | (self.weights[b] > 0))
                               for a, b in self.pairs], dtype=float)
        if np.any(background.sum(axis=1) == 0):
            raise ValueError("each pair requires observed outside frequencies")
        self.background = background / background.sum(axis=1, keepdims=True)
        self.models = [{name: CenteredRidge(width, ridge, outputs=1)
                        for name, width in zip(VARIANTS, (17, 26, 26, 35, 26, 35))}
                       for _ in self.pairs]
        self.pending = deque()
        self.last_sample = None
        self.next_query = 0

    def observe(self, end_sample, power_scan, history_scan=None, coverage=None):
        if (not isinstance(end_sample, int) or end_sample < 0
                or self.last_sample is not None and end_sample <= self.last_sample):
            raise ValueError("observation clock must advance")
        events = []
        gap = (power_scan is None or self.last_sample is not None
               and end_sample != self.last_sample + self.hop_samples)
        if gap:
            while self.pending:
                query = self.pending.popleft()
                events.append({"kind": "censored", "query_id": query["query_id"],
                               "target_sample": query["target_sample"], "reason": "input_gap"})
        self.last_sample = end_sample
        if power_scan is None:
            return events
        power = np.asarray(power_scan, dtype=float)
        if (power.shape != (self.weights.shape[1],) or not np.isfinite(power).all()
                or np.any(power < 0)):
            raise ValueError("expected nonnegative native power on the declared grid")
        current = np.log1p(np.sqrt(self.weights @ power) / self.reference_rms)
        while self.pending and self.pending[0]["target_sample"] <= end_sample:
            query = self.pending.popleft()
            if query["target_sample"] != end_sample:
                raise ValueError("an unobserved target cannot become a training example")
            observed = np.array([current[a] for a, _ in self.pairs])
            for pair, models in enumerate(self.models):
                for name, model in models.items():
                    model.update(query["features"][name][pair], observed[pair:pair + 1])
            events.append({"kind": "completed", "query_id": query["query_id"],
                           "issued_sample": query["issued_sample"],
                           "target_sample": end_sample, "observed": observed.tolist(),
                           "predictions": query["predictions"],
                           "completed_before_issue": query["completed_before_issue"],
                           "features": {name: x.tolist() for name, x in query["features"].items()}})
        if history_scan is None:
            if coverage is not None:
                raise ValueError("coverage requires a history snapshot")
            return events
        history = np.asarray(history_scan, dtype=float)
        known = np.asarray(coverage, dtype=float)
        if (history.shape != (len(power), 8) or known.shape != (8,)
                or not np.isfinite(history).all() or np.any(history < 0)
                or not np.isfinite(known).all() or np.any(known < 0) or np.any(known > 1)):
            raise ValueError("expected native history and separate known coverage")
        retained = np.log1p(np.sqrt(self.weights @ history**2) / self.reference_rms)
        regional = np.column_stack((current, retained)) / math.sqrt(9)
        outside = np.log1p(np.column_stack((np.sqrt(self.background @ power),
                                           np.sqrt(self.background @ history**2)))
                           / self.reference_rms) / math.sqrt(9)
        own = np.array([np.r_[regional[a], known / math.sqrt(8)] for a, _ in self.pairs])
        partner = np.array([regional[b] for _, b in self.pairs])
        features = {"own": own, "partner": np.c_[own, partner],
                    "background": np.c_[own, outside], "both": np.c_[own, outside, partner]}
        current_partner = np.repeat(partner[:, :1], 9, axis=1)
        features["partner_current"] = np.c_[own, current_partner]
        features["both_current"] = np.c_[own, outside, current_partner]
        completed = [m["own"].n for m in self.models]
        predictions = {"persistence": [float(current[a]) for a, _ in self.pairs],
                       "mean": [float(m["own"].mean_y[0]) if m["own"].n else None
                                for m in self.models]}
        for name in VARIANTS:
            predictions[name] = []
            for index, models in enumerate(self.models):
                predicted = models[name].predict(features[name][index])
                predictions[name].append(None if predicted is None else float(max(0., predicted[0])))
        query = {"query_id": self.next_query, "issued_sample": end_sample,
                 "target_sample": end_sample + self.horizon_samples,
                 "completed_before_issue": completed, "predictions": predictions}
        events.append(dict(copy.deepcopy(query), kind="issued"))
        # Issuance data remain fixed while subsequent observations update coefficients.
        self.pending.append(dict(query, features=features))
        assert len(self.pending) <= self.horizon_samples // self.hop_samples + 1
        self.next_query += 1
        return events


def run(native_path, output, *, prefix_sec=2., maximum_regions=3, horizon_sec=1., ridge=1.,
        include_issued_features=False):
    if not math.isfinite(prefix_sec) or prefix_sec <= 0 or not math.isfinite(horizon_sec) or horizon_sec <= 0:
        raise ValueError("expected explicit positive prefix and query durations")
    opener = gzip.open if str(native_path).endswith(".gz") else open
    output.parent.mkdir(parents=True, exist_ok=True)
    with opener(native_path, "rt") as source, gzip.open(output, "xt") as target:
        contract = json.loads(next(source))
        fs, hop = int(contract["sample_rate"]), int(contract["hop_samples"])
        horizon = math.ceil(horizon_sec * fs / hop) * hop
        if horizon < contract["window_samples"]:
            raise ValueError("query lead must cover at least one FFT container")
        frequencies = np.array(contract["frequency_hz"], dtype=float)
        prefix_power = np.zeros(len(frequencies))
        prefix_count = 0
        model = None
        for line in source:
            row = json.loads(line)
            end = round(row["available_sec"] * fs)
            power = np.array(row["nsgt_power_scan"], dtype=float)
            if power.shape != frequencies.shape:
                raise ValueError("native power must align with the frequency grid")
            if model is None:
                if not row["full_window"]:
                    continue
                prefix_power += power
                prefix_count += 1
                if end < prefix_sec * fs or row["known_rms_by_age_scan"] is None:
                    continue
                selection = regions(frequencies, prefix_power / prefix_count, maximum_regions)
                if selection is None:
                    continue
                indices, weights = selection
                reference = float(np.sqrt(np.mean(prefix_power / prefix_count)))
                model = SpectralRelations(weights, reference, horizon, hop, ridge)
                target.write(json.dumps({"kind": "contract", "sample_rate": fs,
                    "hop_samples": hop, "window_samples": contract["window_samples"],
                    "horizon_samples": horizon, "prefix_end_sample": end,
                    "selected_indices": indices, "selected_frequencies_hz": frequencies[indices].tolist(),
                    "region_weights": weights.tolist(), "reference_rms": reference,
                    "pairs": model.pairs, "ridge": ridge,
                    "scope": "Regional log1p NSGT RMS point forecasts from current and native retained evidence. Cumulative ridge statistics, not cognitive forgetting, source labels, causal effects, grouping probabilities, or action rewards."}) + "\n")
            history = row["known_rms_by_age_scan"]
            if history is not None and row["history_observed_through_sample"] != end:
                raise ValueError("history availability differs from the observation endpoint")
            for event in model.observe(end, power if row["full_window"] else None,
                                       history, row["known_coverage_by_age"] if history is not None else None):
                if include_issued_features and event["kind"] == "issued":
                    event["features"] = {name: x.tolist()
                                         for name, x in model.pending[-1]["features"].items()}
                target.write(json.dumps(event, allow_nan=False) + "\n")
        target.write(json.dumps({"kind": "eof", "queries": model.next_query if model else 0,
                                "pending": len(model.pending) if model else 0,
                                "region_selection_available": model is not None}) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.native, args.output)
