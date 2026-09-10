#!/usr/bin/env python3
"""Bounded distribution distances for frozen regional context forecasts.

CRPS after u=y/(y+s) is a weighted integral of Brier scores (Gneiting and
Raftery 2007, equations 20 and 49). The scale is an evaluation choice, not
auditory noise, a hearing threshold, musical value or a change to inference.
"""

import argparse
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import expit, stdtr, stdtrit


_quadrature = lru_cache(maxsize=8)(np.polynomial.legendre.leggauss)


@lru_cache(maxsize=4096)
def _student_nodes(degree):
    return stdtrit(degree, (_quadrature(16)[0]+1)/2)


def bounded_cdf(forecast, points, *, scale=1., survival=False):
    """CDF or strict survival, including the zero atom; never exponentiate a mean."""
    u = np.asarray(points, dtype=float)
    names = ('location', 'scale2', 'degrees_freedom', 'positive_probability', 'log_weights')
    location, variance, degrees, positive, log_weights = (
        np.asarray(forecast[k], dtype=float) for k in names)
    weights = np.exp(log_weights)
    if (not math.isfinite(scale) or scale <= 0
            or not np.isfinite(u).all() or np.any((u < 0) | (u > 1))
            or location.ndim != 1 or not len(location)
            or any(a.shape != location.shape for a in (variance, degrees, positive, weights))
            or any(not np.isfinite(a).all() for a in (location, variance, degrees, positive, weights))
            or np.any(variance <= 0) or np.any(degrees <= 0)
            or np.any((positive < 0) | (positive > 1))
            or not math.isclose(float(weights.sum()), 1., rel_tol=0., abs_tol=1e-10)):
        raise ValueError('expected a normalized mixture and bounded evaluation coordinates')
    flat = u.ravel()
    interior = (flat > 0) & (flat < 1)
    result = np.empty(len(flat))
    mass = weights * positive
    result[flat == 0] = mass.sum() if survival else weights @ (1-positive)
    result[flat == 1] = 0. if survival else 1.
    if interior.any():
        z = np.log(flat[interior]) - np.log1p(-flat[interior]) + math.log(scale)
        t = (z[:, None] - location) / np.sqrt(variance)
        result[interior] = (stdtr(degrees, -t) @ mass if survival else
                            weights @ (1-positive) + stdtr(degrees, t) @ mass)
    return result.reshape(u.shape)


def bounded_crps(forecast, value, *, scale=1., order=8):
    """Partition near mixture quantiles, then integrate the actual CDF on each side."""
    if (not math.isfinite(value) or value < 0 or not math.isfinite(scale) or scale <= 0
            or not isinstance(order, int) or order < 2):
        raise ValueError('expected nonnegative observation, positive scale and quadrature order')
    at = 0. if value == 0 else float(expit(math.log(value)-math.log(scale)))
    atom = float(bounded_cdf(forecast, 0., scale=scale))
    location, variance, positive, log_weights = (np.asarray(forecast[k], dtype=float)
                                               for k in ('location', 'scale2', 'positive_probability', 'log_weights'))
    standard = np.array([_student_nodes(df) for df in forecast['degrees_freedom']])
    approximate = expit(location[:, None]+np.sqrt(variance)[:, None]*standard-math.log(scale))
    masses = np.exp(log_weights)[:, None]*positive[:, None]*(_quadrature(16)[1]/2)
    values, masses = np.r_[0., approximate.ravel()], np.r_[atom, masses.ravel()]
    values, inverse = np.unique(values, return_inverse=True)
    masses = np.bincount(inverse, weights=masses)
    present = masses > 0
    knots = np.interp(np.arange(1, 16)/16, np.cumsum(masses[present]), values[present])
    edges = np.unique(np.r_[0., knots, at, 1.])
    widths = np.diff(edges)
    nodes, weights = _quadrature(order)
    points = edges[:-1, None]+widths[:, None]*(nodes+1)/2
    left = edges[1:] <= at
    probabilities = np.empty_like(points)
    probabilities[left] = bounded_cdf(forecast, points[left], scale=scale)
    probabilities[~left] = bounded_cdf(forecast, points[~left], scale=scale, survival=True)
    return float(widths/2 @ (probabilities**2 @ weights))


def run(source_path, output, *, scales=(.25, 1., 4.), order=8):
    source_path, output = Path(source_path), Path(output)
    if (not scales or any(not math.isfinite(s) or s <= 0 for s in scales)
            or len(set(scales)) != len(scales)):
        raise ValueError('expected distinct positive evaluation scales')
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = {}
    completed = censored = issued = 0
    with gzip.open(source_path, 'rt') as source, gzip.open(output, 'xt') as target:
        def write(row):
            target.write(json.dumps(row, allow_nan=False)+'\n')
        contract = json.loads(next(source))
        if contract['kind'] != 'contract':
            raise ValueError('expected a frozen context forecast contract')
        score=bounded_crps
        integration_options={}
        if contract.get('observation_model')=='folded_gaussian':
            from uncertain_regression import bounded_crps as magnitude_crps
            score=magnitude_crps
            integration_options=dict(outer_component_sd=8)
        elif contract.get('observation_model') == 'folded_student':
            from residual_regression import bounded_crps as residual_crps
            score = residual_crps
            integration_options = dict(outer_signed_component_quantile=1-1e-6,
                                       narrow_partition_width_ratio=8)
        write(dict(kind='contract', source_contract=contract, scales=list(scales), order=order,
                   integration='mixture_quantile_partitions', partition_quantile_order=16,
                   source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                   scope='CRPS of u=y/(y+s) and current-value point persistence. Evaluation only; no cognitive resolution, inference update or musical value.',**integration_options))
        for line in source:
            row = json.loads(line)
            if row['kind'] == 'issued':
                assert row['query_id'] not in pending
                assert all(len(x) >= 9 for x in row['features']['background'])
                pending[row['query_id']] = row
                issued += 1
            elif row['kind'] == 'completed':
                query = pending.pop(row['query_id'])
                assert query['target_sample'] == row['target_sample']
                persistence = [x[0]*math.sqrt(9) for x in query['features']['background']]
                scores = []
                for scale in scales:
                    losses = {name: [score(f, y, scale=scale, order=order) for f, y in
                                     zip(forecasts, row['observed'], strict=True)]
                              for name, forecasts in query['forecasts'].items()}
                    losses['persistence'] = [abs((0. if a == 0 else float(expit(math.log(a)-math.log(scale))))
                                                 -(0. if b == 0 else float(expit(math.log(b)-math.log(scale)))))
                                              for a, b in zip(persistence, row['observed'], strict=True)]
                    scores.append(dict(scale=scale, loss=losses))
                write(dict(kind='completed', query_id=row['query_id'], issued_sample=query['issued_sample'],
                           target_sample=row['target_sample'], observed=row['observed'],
                           persistence=persistence, scores=scores, issued_log_density=row['issued_log_density']))
                completed += 1
            elif row['kind'] == 'censored':
                pending.pop(row['query_id'])
                censored += 1
                write(row)
            elif row['kind'] == 'eof':
                assert row['pending'] == len(pending) and row['completed'] == completed
                assert issued == completed+censored+len(pending)
                write(dict(kind='eof', issued=issued, completed=completed, censored=censored,
                           pending=len(pending), pending_targets=[q['target_sample'] for q in pending.values()]))
            else:
                raise ValueError('unexpected frozen forecast event')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--order', type=int, default=8)
    parser.add_argument('--scales', type=float, nargs='+', default=[.25, 1., 4.])
    args = parser.parse_args()
    run(args.source, args.output, scales=args.scales, order=args.order)
