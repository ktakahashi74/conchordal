#!/usr/bin/env python3
"""Research Hebbian associations from a continuous auditory history.

Shankar & Howard (2012), equations 2.11-2.14, with g=1 and finite-domain
trapezoidal quadrature. Recall is unnormalized activation, not an amplitude
forecast, event probability, stream identity, or musical reward. No forgetting
or competitive learning is added to the original additive rule.
"""

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

from evaluate_temporal_history import TemporalHistory


class TemporalAssociations:
    def __init__(self, ages_sec, post_order, channels):
        self.history = TemporalHistory(ages_sec, post_order, channels)
        ages = self.history.ages_sec
        if len(ages) < 2:
            raise ValueError("at least two ages are required for finite-domain quadrature")
        self.age_weights = np.diff(ages, prepend=ages[0], append=ages[-1])
        self.age_weights = (self.age_weights[:-1] + self.age_weights[1:]) / 2
        self.associations = np.zeros((len(ages), channels, channels))
        self.update_scratch = np.empty_like(self.associations)
        self.cross_mask = ~np.eye(channels, dtype=bool)
        self.pair_exposure_sec = np.zeros(len(ages))
        self.known_current_sec = self.time_sec = 0.

    def observe(self, dt_sec, envelope):
        mean = self.history.advance(dt_sec, envelope)
        if envelope is not None:
            # The mean integrates history throughout the observed interval.
            np.multiply(mean[:, None, :-1], np.asarray(envelope)[None, :, None],
                        out=self.update_scratch)
            self.associations += dt_sec * self.update_scratch
            self.pair_exposure_sec += dt_sec * mean[:, -1]
            self.known_current_sec += dt_sec
        self.time_sec += dt_sec

    def predict(self):
        history = self.history.state[:, -1, :-1]
        within = np.einsum("ai,ai,a->i", np.diagonal(self.associations, axis1=1, axis2=2),
                           history, self.age_weights)
        cross = np.einsum("aij,aj,a,ij->i", self.associations, history,
                          self.age_weights, self.cross_mask)
        return {"issued_sec": self.time_sec,
                "activation": (within + cross).tolist(),
                "within_band_activation": within.tolist(),
                "cross_band_activation": cross.tolist(),
                "known_coverage_by_age": np.clip(self.history.state[:, -1, -1], 0., 1.).tolist(),
                "pair_exposure_sec_by_age": self.pair_exposure_sec.tolist(),
                "known_current_sec": self.known_current_sec}


def run(observations, parameters, output):
    required = {"ages_sec", "post_order", "channels", "step_sec", "report_steps"}
    if set(parameters) != required:
        raise ValueError("expected explicit history, observation and report parameters")
    step, report_steps = parameters["step_sec"], parameters["report_steps"]
    if not math.isfinite(step) or step <= 0 or not isinstance(report_steps, int) or report_steps < 1:
        raise ValueError("invalid observation or report interval")
    model = TemporalAssociations(parameters["ages_sec"], parameters["post_order"], parameters["channels"])
    output.parent.mkdir(parents=True, exist_ok=True)
    count = forecasts = outcomes = censored = 0
    issued = None
    last_time = 0.
    with gzip.open(observations, "rt") as source, gzip.open(output, "xt") as target:
        target.write(json.dumps({"kind": "association_contract", "parameters": parameters,
                                "source": str(observations), "node_density": "g=1",
                                "age_quadrature_weights_sec": model.age_weights.tolist(),
                                "scope": "Unnormalized recall before the next observation. Exact constant-input Hebbian time integral, finite-age quadrature. Missing contributions are not normalized by coverage. No forgetting, probability calibration, future-arrival distribution, phrase classification or generation coupling."}) + "\n")
        while True:
            issued = None
            if count % report_steps == 0:
                issued = dict(model.predict(), kind="forecast", forecast_index=forecasts,
                              observation_index=count, target_start_sec=last_time,
                              target_end_sec=last_time + step)
                target.write(json.dumps(issued, allow_nan=False) + "\n")
                forecasts += 1
            # Issue before reading even the kind or timestamp of the next input.
            line = source.readline()
            if not line:
                break
            row = json.loads(line)
            time = row["available_sec"]
            if not math.isfinite(time) or time <= last_time:
                raise ValueError("expected increasing finite observation times")
            if row["kind"] == "input_gap":
                if issued is not None:
                    target.write(json.dumps({"kind": "censored", "forecast_index": issued["forecast_index"],
                                             "available_sec": time, "reason": "input_gap"}) + "\n")
                    censored += 1
                model.observe(time - last_time, None)
                model.time_sec = last_time = time
                target.write(json.dumps(dict(model.predict(), kind="input_gap"), allow_nan=False) + "\n")
                continue
            if row["kind"] != "auditory_envelope":
                raise ValueError("expected an auditory envelope or explicit gap")
            if not math.isclose(time - last_time, step, rel_tol=0, abs_tol=1e-9):
                raise ValueError("unreported input gap or a mismatched observation interval")
            model.observe(step, row["envelope_scan"])
            model.time_sec = last_time = time
            if issued is not None:
                target.write(json.dumps({"kind": "outcome", "observation_index": count,
                                         "forecast_index": issued["forecast_index"],
                                         "available_sec": time, "envelope_scan": row["envelope_scan"]},
                                        allow_nan=False) + "\n")
                outcomes += 1
            count += 1
    return {"observations": count, "forecasts": forecasts, "observed_outcomes": outcomes,
            "censored_forecasts": censored, "pending_forecasts": int(issued is not None),
            "last_available_sec": last_time,
            "association_scalars": model.associations.size,
            "association_bytes": model.associations.nbytes,
            "final_known_current_sec": model.known_current_sec,
            "final_pair_exposure_sec_by_age": model.pair_exposure_sec.tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("parameters", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.observations, json.loads(args.parameters.read_text()), args.output)))
