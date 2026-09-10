#!/usr/bin/env python3
"""Research: past-partner prediction gains conditional on own and background history.

Directed predictive gain is neither causal influence nor a grouping probability.
The outside-band envelope summary is an observed covariate, not a known cause.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from evaluate_context_expectation import ContextExpectation
from evaluate_phrase_expectation import sha256


VARIANTS = ("own", "partner", "background", "both")


class PredictiveRelations:
    def __init__(self, centers_log2, pairs, lags, step_sec, change_rate_hz,
                 max_hypotheses, prior_shape, prior_scale, zero_mass_prior):
        centers = np.array(centers_log2, dtype=float, copy=True)
        pairs = np.array(pairs, copy=True)
        if (centers.ndim != 1 or not len(centers) or not np.all(np.isfinite(centers))
                or np.any(np.diff(centers) <= 0) or pairs.ndim != 2 or pairs.shape[1] != 2
                or len(pairs) == 0 or pairs.dtype.kind not in "iu" or np.any(pairs < 0)
                or np.any(pairs >= len(centers)) or np.any(pairs[:,0] >= pairs[:,1])
                or len(set(map(tuple,pairs))) != len(pairs)):
            raise ValueError("expected ordered frequency bins and distinct ascending band pairs")
        self.centers_log2, self.pairs = centers, pairs
        frequency = np.exp2(centers)
        erb = 24.7 + frequency / 9.265
        mask = np.ones((len(pairs), len(centers)), dtype=float)
        for p,(a,b) in enumerate(pairs):
            mask[p] = ((np.abs(frequency-frequency[a]) > erb[a])
                       & (np.abs(frequency-frequency[b]) > erb[b]))
        if np.any(mask.sum(axis=1) == 0):
            raise ValueError("each pair needs observed bands outside its ERB neighborhoods")
        self.background_weights = mask / mask.sum(axis=1)[:,None]
        self.models = {}
        triple = np.arange(3*len(pairs)).reshape(-1,3)
        own = triple.reshape(-1,1)
        partner = triple[:,[1,0,0]].reshape(-1,1)
        background = triple[:,[2,2,1]].reshape(-1,1)
        for name in VARIANTS:
            sources = np.concatenate([own, *([background] if name in ("background","both") else []),
                                      *([partner] if name in ("partner","both") else [])],axis=1)
            width = 1+len(lags)*sources.shape[1]
            self.models[name] = ContextExpectation(3*len(pairs), lags, step_sec, change_rate_hz,
                    max_hypotheses, np.zeros(width), np.eye(width), prior_shape, prior_scale,
                    zero_mass_prior, sources)
        self.step_sec = step_sec

    def forecast(self):
        result = {name:model.forecast() for name,model in self.models.items()}
        return None if result["own"] is None else result

    def observe(self, end_sec, log1p_envelope):
        values = None
        if log1p_envelope is not None:
            row = np.asarray(log1p_envelope,dtype=float)
            if row.shape != self.centers_log2.shape or np.any(row < 0) or not np.all(np.isfinite(row)):
                raise ValueError("expected a nonnegative envelope on the declared frequency grid")
            amplitude = np.expm1(row)
            background = np.log1p(np.sqrt(self.background_weights @ (amplitude**2)))
            values = np.column_stack([row[self.pairs[:,0]], row[self.pairs[:,1]], background]).ravel()
        results = {name:model.observe(end_sec,values) for name,model in self.models.items()}
        first = results["own"]
        output = {"kind":first["kind"],"available_sec":end_sec}
        if first["kind"] == "score":
            log_density = {name:row["log_density"].reshape(-1,3)[:,:2].copy()
                           for name,row in results.items()}
            output.update({"log_density":log_density,
                "partner_gain_bits":(log_density["partner"]-log_density["own"])/math.log(2),
                "conditional_partner_gain_bits":(log_density["both"]-log_density["background"])/math.log(2),
                "maximum_pruned_mass":{name:float(row["pruned_mass"].max()) for name,row in results.items()}})
        return output


def run(inputs_path, parameters_path, output):
    parameters = json.loads(parameters_path.read_text())
    cases = json.loads(inputs_path.read_text())
    if not isinstance(cases,list) or not cases:
        raise ValueError("expected a nonempty input manifest")
    output.mkdir(parents=True,exist_ok=False)
    (output/"plan.json").write_text(json.dumps({"parameters":parameters,"inputs":cases,
        "sources":{str(Path(__file__)):sha256(Path(__file__)),
                   str(Path(__file__).with_name("evaluate_context_expectation.py")):sha256(Path(__file__).with_name("evaluate_context_expectation.py"))},
        "scope":"Causal directional predictive comparison only. Inputs are declared log1p envelopes. The observed outside-band summary is not an identified common cause; no causal or grouping probability is inferred. Pairwise scores share observations and cannot be summed as independent evidence."},indent=2)+"\n")
    summary=[]
    for case in cases:
        path=Path(case["input"])
        if sha256(path) != case["sha256"]:
            raise ValueError(f"input changed: {path}")
        with np.load(path) as source:
            values=source["values"]
        model=PredictiveRelations(**parameters)
        rows={key:[] for key in ["target_sec","log_density","partner_gain_bits","conditional_partner_gain_bits"]}
        max_pruned={name:0. for name in VARIANTS}
        for i in range(len(values)):
            # Only past values enter these copied forecasts.
            prediction=model.forecast()
            result=model.observe((i+1)*model.step_sec,values[i])
            if prediction is None:
                continue
            rows["target_sec"].append(prediction["own"]["target_sec"])
            rows["log_density"].append(np.stack([result["log_density"][name] for name in VARIANTS]))
            for key in ["partner_gain_bits","conditional_partner_gain_bits"]:
                rows[key].append(result[key])
            for name in VARIANTS:
                max_pruned[name]=max(max_pruned[name],result["maximum_pruned_mass"][name])
        target=output/f"{case['name']}.npz"
        np.savez_compressed(target,**{key:np.array(value) for key,value in rows.items()})
        row={"case":case["name"],"output":str(target.resolve()),"sha256":sha256(target),
             "scored_targets":len(rows["target_sec"]),"pairs":len(model.pairs),"maximum_pruned_mass":max_pruned,
             "mean_partner_gain_bits":np.mean(rows["partner_gain_bits"],axis=0).tolist(),
             "mean_conditional_partner_gain_bits":np.mean(rows["conditional_partner_gain_bits"],axis=0).tolist()}
        summary.append(row)
        print(json.dumps({"case":case["name"],"scored_targets":row["scored_targets"]}),flush=True)
    (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs",type=Path)
    parser.add_argument("parameters",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    run(args.inputs,args.parameters,args.output)
