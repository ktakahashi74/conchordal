#!/usr/bin/env python3
"""Score frozen context forecasts when their matching native sound becomes observed."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from relational_contexts import RegressionContexts, log_density


FAMILIES = ('background', 'both')
MODES = ('stationary', 'renewal', 'recurrent')
PARAMETERS = dict(contexts=3, change_rate_hz=.2, intercept_precision=.01,
                  slope_precision=1., prior_shape=2., prior_scale=.5, zero_prior=(.5,.5))


def run(source_path, output, *, max_paths=32, observation_sd=None, residual_sd=.1,
        continuation_layout=None, correction_sd=.1):
    if continuation_layout not in (None, 'reset_all', 'shared_continuation'):
        raise ValueError('expected an explicit continuation layout')
    if continuation_layout is not None and observation_sd is None:
        raise ValueError('continuation requires explicit observation noise')
    source_path, output = Path(source_path), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending, banks = {}, {}
    completed = 0
    parameters=PARAMETERS
    model_type,score=RegressionContexts,log_density
    observation_model={}
    scope='Conditional nonnegative regional log1p-RMS density; zero atom and positive log-Student. Fixed finite context/transition/prior hypotheses, no source identity or action value. Issued density and current-filter density are distinct.'
    if observation_sd is not None:
        from uncertain_contexts import MagnitudeContexts
        from uncertain_regression import log_density as magnitude_log_density
        model_type,score=MagnitudeContexts,magnitude_log_density
        parameters={k:PARAMETERS[k] for k in ('contexts','change_rate_hz','intercept_precision','slope_precision')}
        parameters.update(residual_sd=residual_sd,observation_sd=observation_sd)
        observation_model=dict(observation_model='folded_gaussian')
        scope='Conditional nonnegative regional log1p-RMS density with folded Gaussian residual and observation noise. Context and sign uncertainty are distinct. Variance sum, priors and finite path budget are explicit engineering hypotheses, not identified auditory parameters or action value. Zero has a boundary density, not an atom. Issued and current-filter densities are distinct.'
    if continuation_layout is not None:
        from continuation_contexts import ContinuationContexts
        model_type = ContinuationContexts
        parameters = dict(contexts=3, change_rate_hz=.2, correction_sd=correction_sd,
                          residual_sd=residual_sd, observation_sd=observation_sd,
                          shared_continuation=continuation_layout == 'shared_continuation')
        observation_model.update(continuation_layout=continuation_layout,
                                 conditioning='current offset; separate unit-norm target and relational blocks with intercepts')
        scope='Folded Gaussian correction conditional on currently observed regional log1p RMS. Full joint continuation/context covariance; explicit latent signs. Renewal replaces local coefficients and preserves the retained marginal. Priors, variance sum and context/path limits are engineering hypotheses, not identified cognitive mechanisms. Compare unlearned continuation and point persistence; issued forecasts remain immutable.'
    with gzip.open(source_path, 'rt') as source, gzip.open(output, 'xt') as target:
        def write(row):
            target.write(json.dumps(row, allow_nan=False,
                                    default=lambda value: value.tolist())+'\n')

        contract = json.loads(next(source))
        if contract['kind'] != 'contract':
            raise ValueError('expected available frozen native region observations')
        fs = contract['sample_rate']
        write(dict(kind='contract', native_contract=contract, max_paths=max_paths, parameters=parameters,
                   source=str(source_path.resolve()),
                   source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                   scope=scope,**observation_model))
        for line in source:
            row = json.loads(line)
            if row['kind'] == 'issued':
                conditioning = {}
                if continuation_layout is not None:
                    for family, size in (('background', 26), ('both', 35)):
                        conditioning[family] = []
                        for raw in row['features'][family]:
                            x = np.asarray(raw, dtype=float)
                            if x.shape != (size,) or not np.isfinite(x).all() or x[0] < 0:
                                raise ValueError('expected frozen native target and environment features')
                            own, relation = np.r_[1., x[:17]], np.r_[1., x[17:]]
                            conditioning[family].append(dict(
                                offset=float(3*x[0]), target=own/np.linalg.norm(own),
                                relation=relation/np.linalg.norm(relation)))
                if not banks:
                    if continuation_layout is not None:
                        banks = {f'{family}_{mode}': [model_type(len(x['target']), len(x['relation']),
                                                               mode=mode, max_paths=max_paths, **parameters)
                                                     for x in conditioning[family]]
                                 for family in FAMILIES for mode in MODES}
                    else:
                        banks = {f'{family}_{mode}': [model_type(len(x), mode=mode, max_paths=max_paths, **parameters)
                                                 for x in row['features'][family]]
                             for family in FAMILIES for mode in MODES}
                forecasts = {}
                for family in FAMILIES:
                    for mode in MODES:
                        name = f'{family}_{mode}'
                        assert all(m.completed == completed for m in banks[name])
                        if continuation_layout is None:
                            forecasts[name] = [m.forecast(row['target_sample']/fs, x)
                                               for m,x in zip(banks[name],row['features'][family],strict=True)]
                        else:
                            forecasts[name] = [m.forecast(row['target_sample']/fs, x['target'], x['relation'], x['offset'])
                                               for m,x in zip(banks[name],conditioning[family],strict=True)]
                if continuation_layout is not None:
                    forecasts['unlearned_continuation'] = [dict(
                        target_sec=row['target_sample']/fs, completed=0, observed_offset=x['offset'],
                        location=np.array([x['offset']]), log_weights=np.array([0.]),
                        within_component_parameter_variance=np.array([correction_sd**2]),
                        residual_variance=residual_sd**2, observation_noise_variance=observation_sd**2,
                        component_observation_variance=np.array([correction_sd**2+residual_sd**2+observation_sd**2]))
                        for x in conditioning['background']]
                assert row['completed_before_issue'] == [completed]*len(contract['pairs'])
                query = dict(kind='issued', query_id=row['query_id'], issued_sample=row['issued_sample'],
                             target_sample=row['target_sample'], completed_before_issue=completed,
                             features={family: row['features'][family] for family in FAMILIES},
                             forecasts=forecasts)
                if continuation_layout is not None:
                    query['conditioning'] = conditioning
                assert row['query_id'] not in pending
                pending[row['query_id']] = query
                write(query)
            elif row['kind'] == 'completed':
                query = pending.pop(row['query_id'])
                assert row['target_sample'] == query['target_sample']
                assert all(row['features'][family] == query['features'][family] for family in FAMILIES)
                scores = {name: [score(f, y) for f, y in zip(forecasts, row['observed'], strict=True)]
                          for name, forecasts in query['forecasts'].items()}
                updates = {}
                for family in FAMILIES:
                    for mode in MODES:
                        name = f'{family}_{mode}'
                        if continuation_layout is None:
                            updates[name] = [m.observe(row['target_sample']/fs, x, y) for m,x,y in
                                             zip(banks[name],query['features'][family],row['observed'],strict=True)]
                        else:
                            updates[name] = [m.observe(row['target_sample']/fs, x['target'], x['relation'], x['offset'], y)
                                             for m,x,y in zip(banks[name],query['conditioning'][family],row['observed'],strict=True)]
                completed += 1
                write(dict(kind='completed', query_id=row['query_id'], issued_sample=query['issued_sample'],
                           target_sample=row['target_sample'], observed=row['observed'],
                           issued_log_density=scores, filter_updates=updates))
            elif row['kind'] == 'censored':
                pending.pop(row['query_id'])
                write(row)
            elif row['kind'] == 'eof':
                assert len(pending) == row['pending']
                write(dict(kind='eof', completed=completed, pending=len(pending),
                           pending_targets=[q['target_sample'] for q in pending.values()]))
            else:
                raise ValueError('unexpected native query event')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--max-paths', type=int, default=32)
    parser.add_argument('--observation-sd',type=float)
    parser.add_argument('--residual-sd',type=float,default=.1)
    parser.add_argument('--continuation-layout', choices=('reset_all', 'shared_continuation'))
    parser.add_argument('--correction-sd', type=float, default=.1)
    args = parser.parse_args()
    run(args.source, args.output, max_paths=args.max_paths,
        observation_sd=args.observation_sd,residual_sd=args.residual_sd,
        continuation_layout=args.continuation_layout,correction_sd=args.correction_sd)
