"""Distributional audit of bounded native perceptual fields.

Energy loss is the negative of Gneiting and Raftery (2007), equation 22 with
beta=1: E||X-y|| - E||X-X'||/2. Divide the Euclidean norm by sqrt(dimension).
https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf
Numerical bounds use the bounded-differences inequality (Warnke, theorem 1):
https://arxiv.org/pdf/1212.5796
This scores forecasts, not consonance, musical quality or cognitive validity.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from evaluate_perceptual_action import FIELDS, read_results, write_json


def stratified_energy_loss(values, target, record, *, alpha=.05):
    """Use independent within-stratum pairs; do not renormalize omitted mass."""
    values, target = np.asarray(values,dtype=float), np.asarray(target,dtype=float)
    counts = np.asarray(record['draw_counts'])
    weights = np.asarray(record['component_weights'],dtype=float)
    omitted = record['omitted_mass']
    if (values.ndim!=2 or target.shape!=values.shape[1:] or not len(target)
            or counts.ndim!=1 or counts.dtype.kind not in 'iu' or not len(counts)
            or weights.shape!=counts.shape or np.any(counts<2) or int(counts.sum())!=len(values)
            or not np.isfinite(weights).all() or np.any(weights<0)
            or not np.isfinite(omitted) or not 0<=omitted<=1
            or not np.isclose(math.fsum(weights)+omitted,1.,rtol=0.,atol=1e-12)
            or not np.isfinite(values).all() or not np.isfinite(target).all()
            or np.any(values<0) or np.any(values>1) or np.any(target<0) or np.any(target>1)
            or not np.isfinite(alpha) or not 0<alpha<1):
        raise ValueError('expected bounded vectors and independent strata with conserved mass')
    point_weights = np.repeat(weights/counts,counts)
    distances = np.zeros((len(values),len(values)))
    # Direct differences preserve small errors around a common nonzero baseline.
    for index,value in enumerate(values[:-1]):
        delta = values[index+1:]-value
        distances[index,index+1:] = np.sqrt(np.mean(delta*delta,axis=1))
    distances += distances.T
    pair_weights = np.outer(point_weights,point_weights)
    start = 0
    for count in counts:
        block = slice(start,start+count)
        pair_weights[block,block] *= count/(count-1)
        start += count
    np.fill_diagonal(pair_weights,0.)
    delta = values-target
    first = float(point_weights @ np.sqrt(np.mean(delta*delta,axis=1)))
    second = .5*float(np.sum(pair_weights*distances))
    loss = first-second
    mass = math.fsum(weights)
    # Replacing one draw changes the loss by at most (1+mass)*weight/count.
    radius = (1+mass)*math.sqrt(.5*float(np.sum(weights**2/counts))*math.log(2/alpha))
    missing_lower, missing_upper = -.5*omitted*(2-omitted), omitted
    return dict(energy_loss=loss,expected_distance_estimate=first,pair_dispersion_estimate=second,
        omitted_mass=omitted,omission_lower_adjustment=missing_lower,omission_upper_adjustment=missing_upper,
        numerical_radius=radius,numerical_lower=float(np.clip(loss+missing_lower-radius,0.,1.)),
        numerical_upper=float(np.clip(loss+missing_upper+radius,0.,1.)),marginal_alpha=alpha)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir()
    results=[]
    for path in sorted(args.input.iterdir()):
        if not (path/'result.json').exists():
            continue
        original=json.loads((path/'result.json').read_text())
        issued=json.loads((path/'issued.json').read_text())
        digest=hashlib.sha256((path/'predicted.jsonl').read_bytes()).hexdigest()
        if digest!=json.loads((path/'prediction-seal.json').read_text())['sha256']:
            raise ValueError('issued native forecasts changed before distributional audit')
        contract,predictions=read_results(path/'predicted.jsonl')
        actual_contract,actual=read_results(path/'actual.jsonl')
        if actual_contract!=contract:
            raise ValueError('distributional audit requires matching native contracts')
        truth=actual['actual']['observation']
        scores={}
        for name,forecast in issued['forecasts'].items():
            record=forecast['sampling']
            scores[name]={}
            for field in FIELDS:
                values=np.asarray([predictions[f'{name}.draw.{i}']['observation'][field]
                                   for i in range(record['actual_draws'])])
                scores[name][field]=stratified_energy_loss(values,truth[field],record)
        result=dict(case=original['case'],chosen_action=original['chosen_action'],scores=scores,
                    native_prediction_sha256=digest)
        write_json(args.output/(path.name+'.json'),result)
        results.append(result)
        print('scored',path.name,flush=True)
    if not results:
        raise ValueError('no completed forecasts found for distributional audit')
    write_json(args.output/'summary.json',results)


if __name__=='__main__':
    main()
