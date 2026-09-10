#!/usr/bin/env python3
"""Research: causal continuous-feature expectations over changing contexts.

This is a Bayesian autoregression with a mixture over context starts, not a
D-REX port or a source/phrase classifier. Every feature has its own contexts.
Priors, lag offsets, observation cadence, change rate, and computation budget
are explicit. A budget prunes hypotheses; it does not set a memory lifetime.
"""

import argparse
from collections import deque
import datetime as dt
import gzip
import json
import math
from pathlib import Path

import numpy as np

from evaluate_phrase_expectation import sha256


class ContextExpectation:
    def __init__(self, features, lags, step_sec, change_rate_hz, max_hypotheses,
                 prior_mean, prior_precision, prior_shape, prior_scale, zero_mass_prior=None,
                 conditioning_sources=None):
        lags = list(lags)
        if not isinstance(features, int) or isinstance(features, bool) or features < 1:
            raise ValueError("expected a positive feature count")
        sources = (np.arange(features)[:, None] if conditioning_sources is None
                   else np.array(conditioning_sources, copy=True))
        if (sources.ndim != 2 or sources.shape[0] != features or sources.shape[1] < 1
                or sources.dtype.kind not in "iu" or np.any(sources < 0) or np.any(sources >= features)
                or any(len(set(row)) != len(row) for row in sources)):
            raise ValueError("expected distinct in-range conditioning sources for each feature")
        mean = np.array(prior_mean, dtype=float, copy=True)
        precision = np.array(prior_precision, dtype=float, copy=True)
        if (not isinstance(features, int) or isinstance(features, bool) or features < 1
                or any(not isinstance(lag, int) or isinstance(lag, bool) or lag < 1 for lag in lags)
                or lags != sorted(set(lags)) or not math.isfinite(step_sec) or step_sec <= 0
                or not math.isfinite(change_rate_hz) or change_rate_hz < 0
                or not isinstance(max_hypotheses, int) or max_hypotheses < 1
                or mean.shape != (len(lags) * sources.shape[1] + 1,)
                or precision.shape != (len(mean), len(mean))
                or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(precision))
                or not np.allclose(precision, precision.T, rtol=0, atol=1e-14)
                or not math.isfinite(prior_shape) or prior_shape <= 0
                or not math.isfinite(prior_scale) or prior_scale <= 0):
            raise ValueError("expected explicit valid dimensions, priors, clock, rate and budget")
        try:
            np.linalg.cholesky(precision)
        except np.linalg.LinAlgError as error:
            raise ValueError("prior precision must be positive definite") from error
        if zero_mass_prior is not None:
            zero_mass_prior = np.array(zero_mass_prior, dtype=float, copy=True)
            if (zero_mass_prior.shape != (2,) or not np.all(np.isfinite(zero_mass_prior))
                    or np.any(zero_mass_prior <= 0)):
                raise ValueError("expected positive prior counts for positive values and exact zeros")
        self.zero_mass_prior = zero_mass_prior
        self.conditioning_sources = sources
        self.features, self.lags = features, lags
        self.step_sec, self.change_rate_hz = step_sec, change_rate_hz
        self.max_hypotheses = max_hypotheses
        self.prior_mean, self.prior_cov = mean, np.linalg.inv(precision)
        self.prior_shape, self.prior_scale = prior_shape, prior_scale
        self.time_sec = 0.
        self.posterior_time_sec = None
        self.recent = deque(maxlen=max(lags, default=0))
        self.mean = np.broadcast_to(mean, (features, 1, len(mean))).copy()
        self.cov = np.broadcast_to(self.prior_cov, (features, 1, len(mean), len(mean))).copy()
        self.shape = np.full((features, 1), prior_shape, dtype=float)
        self.scale = np.full((features, 1), prior_scale, dtype=float)
        self.log_weights = np.zeros((features, 1))
        self.starts_sec = np.zeros((features, 1))
        if zero_mass_prior is not None:
            self.positive_count = np.full((features, 1), zero_mass_prior[0])
            self.zero_count = np.full((features, 1), zero_mass_prior[1])

    def _components(self):
        if len(self.recent) < max(self.lags, default=0):
            return None
        design = np.column_stack([np.ones(self.features),
                                  *[self.recent[-lag][self.conditioning_sources] for lag in self.lags]])
        target = self.time_sec + self.step_sec
        mean, cov, shape, scale = self.mean, self.cov, self.shape, self.scale
        weights, starts = self.log_weights, self.starts_sec
        positive_count = self.positive_count if self.zero_mass_prior is not None else None
        zero_count = self.zero_count if self.zero_mass_prior is not None else None
        if self.posterior_time_sec is not None and self.change_rate_hz > 0:
            log_survival = -self.change_rate_hz * (target - self.posterior_time_sec)
            weights = np.column_stack([weights + log_survival,
                                       np.full(self.features, math.log(-math.expm1(log_survival)))])
            mean = np.concatenate([mean, np.broadcast_to(self.prior_mean, (self.features, 1, len(self.prior_mean)))], axis=1)
            cov = np.concatenate([cov, np.broadcast_to(self.prior_cov, (self.features, 1, *self.prior_cov.shape))], axis=1)
            shape = np.column_stack([shape, np.full(self.features, self.prior_shape)])
            scale = np.column_stack([scale, np.full(self.features, self.prior_scale)])
            # After a gap, this is a reset interval, not a dated boundary.
            starts = np.column_stack([starts, np.full(self.features, self.posterior_time_sec)])
            if self.zero_mass_prior is not None:
                positive_count = np.column_stack([positive_count, np.full(self.features, self.zero_mass_prior[0])])
                zero_count = np.column_stack([zero_count, np.full(self.features, self.zero_mass_prior[1])])
        vx = np.einsum("bhij,bj->bhi", cov, design)
        q = 1 + np.einsum("bi,bhi->bh", design, vx)
        location = np.einsum("bhi,bi->bh", mean, design)
        scale2 = scale / shape * q
        return {"design": design, "mean": mean, "cov": cov, "shape": shape,
                "scale": scale, "log_weights": weights, "starts_sec": starts,
                "vx": vx, "q": q, "location": location, "scale2": scale2,
                "positive_count": positive_count, "zero_count": zero_count}

    def forecast(self):
        components = self._components()
        if components is None:
            return None
        result = {"issued_sec": self.time_sec, "target_sec": self.time_sec + self.step_sec,
                "location": components["location"].copy(),
                "scale2": components["scale2"].copy(),
                "degrees_freedom": 2 * components["shape"].copy(),
                "log_weights": components["log_weights"].copy()}
        if self.zero_mass_prior is not None:
            result["positive_probability"] = (components["positive_count"]
                                              / (components["positive_count"] + components["zero_count"]))
        return result

    def observe(self, end_sec, values):
        if not math.isfinite(end_sec) or end_sec <= self.time_sec:
            raise ValueError("expected a strictly increasing finite observation endpoint")
        if values is None:
            self.time_sec = end_sec
            self.recent.clear()
            return {"kind": "input_gap", "available_sec": end_sec}
        values = np.array(values, dtype=float, copy=True)
        if (values.shape != (self.features,) or not np.all(np.isfinite(values))
                or self.zero_mass_prior is not None and np.any(values < 0)
                or not math.isclose(end_sec - self.time_sec, self.step_sec, rel_tol=0, abs_tol=1e-8)):
            raise ValueError("expected one finite feature vector per declared step; declare gaps explicitly")
        comp = self._components()
        result = {"kind": "warmup", "available_sec": end_sec}
        if comp is not None:
            learned = np.ones(self.features, dtype=bool) if self.zero_mass_prior is None else values > 0
            response = values.copy()
            if self.zero_mass_prior is not None:
                response[learned] = np.log(values[learned])
            error = response[:, None] - comp["location"]
            degrees = 2 * comp["shape"]
            lgamma = np.vectorize(math.lgamma, otypes=[float])
            log_likelihood = (lgamma((degrees + 1) / 2) - lgamma(degrees / 2)
                              - .5 * np.log(degrees * np.pi * comp["scale2"])
                              - (degrees + 1) / 2 * np.log1p(error ** 2 / (degrees * comp["scale2"])))
            if self.zero_mass_prior is not None:
                total_count = comp["positive_count"] + comp["zero_count"]
                # Density is with respect to a zero atom plus Lebesgue measure.
                log_likelihood[learned] += (np.log(comp["positive_count"][learned] / total_count[learned])
                                            - response[learned, None])
                log_likelihood[~learned] = np.log(comp["zero_count"][~learned] / total_count[~learned])
            log_joint = comp["log_weights"] + log_likelihood
            log_density = np.logaddexp.reduce(log_joint, axis=1)
            posterior = log_joint - log_density[:, None]
            count = min(self.max_hypotheses, posterior.shape[1])
            keep = np.argsort(-posterior, axis=1, kind="stable")[:, :count]
            band = np.arange(self.features)[:, None]
            retained = posterior[band, keep]
            log_kept_mass = np.logaddexp.reduce(retained, axis=1)
            gain = learned[:, None] / comp["q"]
            updated_mean = comp["mean"] + comp["vx"] * (error * gain)[:, :, None]
            updated_cov = comp["cov"] - (comp["vx"][:, :, :, None] * comp["vx"][:, :, None, :]
                                         * gain[:, :, None, None])
            self.mean = updated_mean[band, keep]
            self.cov = updated_cov[band, keep]
            self.shape = (comp["shape"] + .5 * learned[:, None])[band, keep]
            self.scale = (comp["scale"] + .5 * error ** 2 * gain)[band, keep]
            if self.zero_mass_prior is not None:
                self.positive_count = (comp["positive_count"] + learned[:, None])[band, keep]
                self.zero_count = (comp["zero_count"] + (~learned)[:, None])[band, keep]
            self.log_weights = retained - log_kept_mass[:, None]
            self.starts_sec = comp["starts_sec"][band, keep]
            result = {"kind": "score", "available_sec": end_sec,
                      "log_density": log_density.copy(),
                      "reset_interval_probability": (np.exp(posterior[:, -1]) if self.posterior_time_sec is not None
                                                     and self.change_rate_hz > 0 else np.zeros(self.features)),
                      "reset_interval_start_sec": self.posterior_time_sec,
                      "pruned_mass": np.maximum(0, -np.expm1(log_kept_mass)),
                      "context_starts_sec": self.starts_sec.copy(),
                      "context_log_weights": self.log_weights.copy()}
            self.posterior_time_sec = end_sec
        self.recent.append(values)
        self.time_sec = end_sec
        return result


def envelope_frames(path, input_step_sec, step_sec, amplitude_scale):
    """Aggregate complete observed intervals, never join evidence across gaps."""
    ratio = round(step_sec / input_step_sec)
    if (input_step_sec <= 0 or ratio < 1 or not math.isclose(ratio * input_step_sec, step_sec)
            or not math.isfinite(amplitude_scale) or amplitude_scale <= 0):
        raise ValueError("expected a positive scale and an integer aggregation ratio")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as source:
        total, count, previous = None, 0, 0.
        for line in source:
            row = json.loads(line)
            time = float(row["available_sec"])
            if row["kind"] == "input_gap":
                if time <= previous:
                    raise ValueError("expected an increasing gap endpoint")
                yield time, None
                total, count, previous = None, 0, time
                continue
            value = np.asarray(row["envelope_scan"], dtype=float)
            if (row["kind"] != "auditory_envelope" or value.ndim != 1 or not len(value)
                    or np.any(value < 0) or not np.all(np.isfinite(value))
                    or not math.isclose(time - previous, input_step_sec, rel_tol=0, abs_tol=1e-8)):
                raise ValueError("expected contiguous nonnegative auditory envelopes")
            if total is None:
                total = np.zeros_like(value)
            if total.shape != value.shape:
                raise ValueError("feature grid changed")
            total += value ** 2
            count += 1
            previous = time
            if count == ratio:
                yield time, np.log1p(np.sqrt(total / ratio) / amplitude_scale)
                total.fill(0)
                count = 0


def run(manifest_path, config_path, output):
    cases = json.loads(manifest_path.read_text())
    config = json.loads(config_path.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("expected a nonempty input manifest")
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__)
    plan = {"created_at": dt.datetime.now().astimezone().isoformat(), "config": config,
            "cases": cases, "source_sha256": sha256(source),
            "manifest_sha256": sha256(manifest_path), "config_sha256": sha256(config_path),
            "scope": "Independent continuous-feature contexts. Log likelihood in declared log1p-amplitude units, not onset probability, source identity, phrase, or cognitive parameter identification. With zero_mass_prior, exact zero has a probability mass and positive values a log-Student density with its Jacobian; forecast locations describe log positive values, not mean amplitudes. Without it, Student predictions extend over the real line. Warmup only fills lags. Missing input censors a target; elapsed time affects the reset prior, never a silence likelihood. No model learns the unobserved inputs."}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    summaries = []
    for index, case in enumerate(cases):
        path = Path(case["input"])
        if sha256(path) != case["sha256"]:
            raise ValueError(f"input changed: {path}")
        model = ContextExpectation(**config["model"])
        stream = iter(envelope_frames(path, config["input_step_sec"], model.step_sec, config["amplitude_scale"]))
        target = output / f"{index:03}.jsonl.gz"
        counts = {"score": 0, "input_gap": 0, "warmup": 0}
        loss = np.zeros(model.features)
        worst_pruned = np.zeros(model.features)
        with gzip.open(target, "wt") as sink:
            def write(record):
                sink.write(json.dumps(record, allow_nan=False,
                                      default=lambda value: value.tolist()) + "\n")
            while True:
                prediction = model.forecast()
                if prediction is not None:
                    write({"kind": "forecast", **prediction})
                try:
                    time, value = next(stream)
                except StopIteration:
                    write({"kind": "right_censored", "target_sec": prediction["target_sec"] if prediction else None})
                    break
                result = model.observe(time, value)
                counts[result["kind"]] += 1
                if result["kind"] == "score":
                    loss -= result["log_density"] / math.log(2)
                    worst_pruned = np.maximum(worst_pruned, result["pruned_mass"])
                write({**result, "observed": value})
        summaries.append({"case": case, "output": str(target.resolve()), "sha256": sha256(target),
                          "counts": counts, "loss_bits_by_feature": loss.tolist(),
                          "maximum_pruned_mass_by_feature": worst_pruned.tolist()})
        print(json.dumps({"completed": index + 1, "total": len(cases), "counts": counts}), flush=True)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.manifest, args.config, args.output)
