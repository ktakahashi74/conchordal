#!/usr/bin/env python3
"""Research temporal history with scale-dependent temporal precision.

This implements the continuous-history kernel implied by the Post inverse in
Shankar & Howard (2012), not its proposed neural circuit or a fitted listener.
The kernel is realized by an equivalent cascade with exact constant-input
updates. Known-input coverage is an additional observation contract: a missing
interval has no observed contribution, but is not classified as silence.
"""

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np


class TemporalHistory:
    def __init__(self, ages_sec, post_order, channels):
        ages = np.array(ages_sec, dtype=float, copy=True)
        if (ages.ndim != 1 or not len(ages) or not np.all(np.isfinite(ages))
                or np.any(ages <= 0) or np.any(np.diff(ages) <= 0)
                or not isinstance(post_order, int) or post_order < 1
                or not isinstance(channels, int) or channels < 1):
            raise ValueError("expected positive ordered ages, positive integer Post order and channel count")
        self.ages_sec, self.post_order, self.channels = ages, post_order, channels
        self.rates = post_order / ages
        if not np.all(np.isfinite(self.rates)):
            raise ValueError("unrepresentable history rates")
        self.state = np.zeros((len(ages), post_order + 1, channels + 1))
        self.previous_dt = self.transition = self.input_gain = None
        self.mean_transition = self.mean_input_gain = None

    def advance(self, dt_sec, envelope):
        if not math.isfinite(dt_sec) or dt_sec <= 0:
            raise ValueError("expected a positive finite elapsed interval")
        if envelope is None:
            observed = np.zeros(self.channels + 1)
        else:
            envelope = np.asarray(envelope, dtype=float)
            if envelope.shape != (self.channels,) or not np.all(np.isfinite(envelope)) or np.any(envelope < 0):
                raise ValueError("expected a finite nonnegative aligned envelope or unknown input")
            observed = np.r_[envelope, 1.]
        if dt_sec != self.previous_dt:
            order = self.post_order
            with np.errstate(over="ignore"):
                x = self.rates * dt_sec
            if np.any(x == 0):
                raise ValueError("unrepresentable elapsed history interval")
            terms = np.zeros((len(x), order + 1))
            finite = np.isfinite(x)
            degrees = np.arange(order + 1)
            factorials = np.array([math.lgamma(n + 1) for n in degrees])
            terms[finite] = np.exp(-x[finite, None] + np.log(x[finite, None]) * degrees - factorials)
            gains = np.maximum(0., 1 - np.cumsum(terms, axis=1))
            # Sum the positive Poisson tail where complement subtraction loses it.
            small = x < order + 1
            if np.any(small):
                tail = np.exp(-x[small, None] + np.log(x[small, None]) * (degrees + 1)
                              - np.array([math.lgamma(n + 2) for n in degrees]))
                total = tail.copy()
                extra = 1
                while np.any(tail > np.finfo(float).eps * total):
                    tail *= x[small, None] / (degrees + 1 + extra)
                    total += tail
                    extra += 1
                gains[small] = total
            differences = degrees[:, None] - degrees[None, :]
            self.transition = terms[:, np.maximum(differences, 0)] * (differences >= 0)
            self.input_gain = gains
            next_gain = np.ones(len(x))
            next_term = np.zeros(len(x))
            next_term[finite] = np.exp(-x[finite] + np.log(x[finite]) * (order + 1)
                                       - math.lgamma(order + 2))
            next_gain[finite] = np.maximum(0., gains[finite, -1] - next_term[finite])
            if np.any(small):
                tail = next_term[small] * x[small] / (order + 2)
                total = tail.copy()
                extra = 1
                while np.any(tail > np.finfo(float).eps * total):
                    tail *= x[small] / (order + 2 + extra)
                    total += tail
                    extra += 1
                next_gain[small] = total
            # Integrate the last cascade stage before updating its endpoint.
            self.mean_transition = gains[:, ::-1] / x[:, None]
            self.mean_input_gain = gains[:, -1] - (order + 1) * (next_gain / x)
            self.previous_dt = dt_sec
        mean = np.einsum("aj,ajc->ac", self.mean_transition, self.state)
        mean += self.mean_input_gain[:, None] * observed
        self.state = np.einsum("aij,ajc->aic", self.transition, self.state) + self.input_gain[:, :, None] * observed
        return mean

    def snapshot(self):
        known = self.state[:, -1, :-1]
        coverage = self.state[:, -1, -1]
        assert known.shape == (len(self.ages_sec), self.channels)
        return {"ages_sec": self.ages_sec.tolist(), "post_order": self.post_order,
                "known_envelope_by_age": known.tolist(),
                "known_coverage_by_age": np.clip(coverage, 0., 1.).tolist()}


def run(observations, parameters, output):
    parameters = dict(parameters)
    required = {"ages_sec", "post_order", "channels", "step_sec", "report_steps"}
    if set(parameters) != required:
        raise ValueError("expected explicit history, step and reporting parameters")
    step, report_steps = parameters["step_sec"], parameters["report_steps"]
    if not math.isfinite(step) or step <= 0 or not isinstance(report_steps, int) or report_steps < 1:
        raise ValueError("invalid observation or report interval")
    model = TemporalHistory(parameters["ages_sec"], parameters["post_order"], parameters["channels"])
    output.parent.mkdir(parents=True, exist_ok=True)
    last_time, count = 0., 0
    with gzip.open(observations, "rt") as source, gzip.open(output, "xt") as target:
        target.write(json.dumps({"kind": "history_contract", "parameters": parameters,
                                "source": str(observations),
                                "scope": "A gamma-kernel realization of the continuous Post history. Piecewise-constant approximation of observed envelopes. Coverage is known kernel mass, not perceptual confidence. Explicit unfitted ages and precision; no item interference, grouping, forecast or generation coupling."}) + "\n")
        for line in source:
            row = json.loads(line)
            time = row["available_sec"]
            if not math.isfinite(time) or time <= last_time:
                raise ValueError("expected increasing finite observation times")
            if row["kind"] == "input_gap":
                model.advance(time - last_time, None)
                last_time = time
                target.write(json.dumps(dict(model.snapshot(), kind="input_gap", available_sec=time), allow_nan=False) + "\n")
                continue
            if row["kind"] != "auditory_envelope":
                raise ValueError("expected an auditory envelope or explicit gap")
            if not math.isclose(time - last_time, step, rel_tol=0, abs_tol=1e-9):
                raise ValueError("unreported input gap or a mismatched observation interval")
            model.advance(step, row["envelope_scan"])
            count += 1
            last_time = time
            if count % report_steps == 0:
                target.write(json.dumps(dict(model.snapshot(), kind="temporal_history", available_sec=time), allow_nan=False) + "\n")
    return {"observations": count, "last_available_sec": last_time,
            "state_scalars": model.state.size, "state_bytes": model.state.nbytes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("parameters", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.observations, json.loads(args.parameters.read_text()), args.output)))
