#!/usr/bin/env python3
"""Learn total residual variation from matching outcomes of frozen regional forecasts."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from residual_regression import ResidualRegression, log_density


def run(source_path, output, *, max_paths=128, state_drift_per_sec=None):
    source_path, output = Path(source_path), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending, banks = {}, {}
    completed = issued = censored = 0
    parameters = dict(prior_shape=3., coefficient_residual_variance=.02,
                      coefficient_relative_covariance=.25, continuation_residual_variance=.03)
    if state_drift_per_sec is not None:
        from evolving_regression import EvolvingResidualRegression
        if not np.isfinite(state_drift_per_sec) or state_drift_per_sec < 0:
            raise ValueError('expected a finite nonnegative state diffusion ratio')
        parameters.update(state_drift_per_sec=state_drift_per_sec,
                          state_initialization='zero deviation at first issued sample')
    baseline = ResidualRegression([], np.zeros((0, 0)), prior_shape=3.,
                                  prior_residual_variance=.03, max_paths=max_paths)
    with gzip.open(source_path, 'rt') as source, gzip.open(output, 'xt') as target:
        def write(row):
            target.write(json.dumps(row, allow_nan=False,
                                    default=lambda value: value.tolist())+'\n')
        contract = json.loads(next(source))
        if contract['kind'] != 'contract':
            raise ValueError('expected frozen native region observations')
        pairs = len(contract['pairs'])
        scope = ('Stationary folded offset regression with integrated unknown total residual variance. '
                 'Frozen inputs, explicit signed paths and finite path approximation. '
                 'No measurement/process split, cognitive retention, source identity or action value. '
                 'Observed numerical zero, missing observation and pending future are distinct.')
        if state_drift_per_sec is not None:
            scope = ('Folded offset regression with static coefficients and a continuous-time diffusing deviation. '
                     'Joint coefficient/state covariance and a shared unknown variance scale are integrated. '
                     'The declared diffusion ratio specifies temporal covariance, not identified auditory noise '
                     'or cognitive retention. Frozen predictions, zero, missing input and pending future remain distinct.')
        write(dict(kind='contract', observation_model='folded_student', native_contract=contract,
                   parameters=parameters, max_paths=max_paths, source=str(source_path.resolve()),
                   source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                   scope=scope))
        for line in source:
            row = json.loads(line)
            if row['kind'] == 'issued':
                conditioning = {}
                for family, size in (('background', 26), ('both', 35)):
                    conditioning[family] = []
                    assert len(row['features'][family]) == pairs
                    for raw in row['features'][family]:
                        x = np.asarray(raw, dtype=float)
                        if x.shape != (size,) or not np.isfinite(x).all() or x[0] < 0:
                            raise ValueError('expected frozen native target and environment features')
                        own, relation = np.r_[1., x[:17]], np.r_[1., x[17:]]
                        conditioning[family].append(dict(
                            offset=float(3*x[0]),
                            design=np.r_[own/np.linalg.norm(own), relation/np.linalg.norm(relation)]))
                conditioning['continuation'] = [dict(offset=x['offset'], design=np.empty(0))
                                                for x in conditioning['background']]
                assert [x['offset'] for x in conditioning['background']] == [
                    x['offset'] for x in conditioning['both']]
                if not banks:
                    model_type, extra = ResidualRegression, {}
                    if state_drift_per_sec is not None:
                        model_type = EvolvingResidualRegression
                        extra = dict(drift_per_sec=state_drift_per_sec,
                                     initial_time_sec=row['issued_sample']/contract['sample_rate'])
                    banks = {family: [model_type(
                        np.zeros(len(x['design'])), .25*np.eye(len(x['design'])),
                        prior_shape=3., prior_residual_variance=.03 if family == 'continuation' else .02,
                        max_paths=max_paths, **extra) for x in values] for family, values in conditioning.items()}
                assert row['completed_before_issue'] == [completed]*pairs
                forecasts = {}
                for family, models in banks.items():
                    assert all(m.completed == completed for m in models)
                    target_args = () if state_drift_per_sec is None else (row['target_sample']/contract['sample_rate'],)
                    forecasts[family] = [m.forecast(*target_args, x['design'], offset=x['offset'])
                                         for m, x in zip(models, conditioning[family], strict=True)]
                forecasts['unlearned_continuation'] = [baseline.forecast([], offset=x['offset'])
                                                       for x in conditioning['continuation']]
                query = dict(kind='issued', query_id=row['query_id'], issued_sample=row['issued_sample'],
                             target_sample=row['target_sample'], completed_before_issue=completed,
                             features={f: row['features'][f] for f in ('background', 'both')},
                             conditioning=conditioning, forecasts=forecasts)
                assert query['query_id'] not in pending
                pending[query['query_id']] = query
                issued += 1
                write(query)
            elif row['kind'] == 'completed':
                query = pending.pop(row['query_id'])
                assert row['target_sample'] == query['target_sample']
                assert all(row['features'][f] == query['features'][f] for f in ('background', 'both'))
                scores = {name: [log_density(f, y) for f, y in zip(values, row['observed'], strict=True)]
                          for name, values in query['forecasts'].items()}
                target_args = () if state_drift_per_sec is None else (row['target_sample']/contract['sample_rate'],)
                updates = {family: [m.observe(*target_args, x['design'], y, offset=x['offset']) for m, x, y in
                                    zip(models, query['conditioning'][family], row['observed'], strict=True)]
                           for family, models in banks.items()}
                completed += 1
                write(dict(kind='completed', query_id=row['query_id'], issued_sample=query['issued_sample'],
                           target_sample=row['target_sample'], observed=row['observed'],
                           issued_log_density=scores, filter_updates=updates))
            elif row['kind'] == 'censored':
                pending.pop(row['query_id'])
                censored += 1
                write(row)
            elif row['kind'] == 'eof':
                assert len(pending) == row['pending']
                assert issued == completed+censored+len(pending)
                write(dict(kind='eof', completed=completed, pending=len(pending),
                           pending_targets=[q['target_sample'] for q in pending.values()]))
            else:
                raise ValueError('unexpected native query event')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--max-paths', type=int, default=128)
    parser.add_argument('--state-drift-per-sec', type=float)
    args = parser.parse_args()
    run(args.source, args.output, max_paths=args.max_paths, state_drift_per_sec=args.state_drift_per_sec)
