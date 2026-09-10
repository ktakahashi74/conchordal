"""Compare retained states and selected-evidence proposals on common future audio.

The preceding prospective campaign supplies inferred pre-action parameters only.
The original conditional forecast is reconstructed exactly from past evidence.
Only the chosen adaptation is visible during proposal; common comparison and
evaluation targets become available after their predictions have been recorded.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from acoustic_revision import revise_body_bank
from evaluate_perceptual_action import FIELDS, read_results, write_json
from evaluate_prospective_actions import native
from evaluate_structural_predictions import build_driven_bank
from perceptual_sampling import bounded_summary, stratified_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--name', action='append')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    old_plan = json.loads((args.previous/'plan.json').read_text())
    if plan['driven_body'] != old_plan['driven_body']:
        raise ValueError('revision controls require the original scale and state priors')
    actions = plan['chosen_actions']
    if not actions or len(set(actions)) != len(actions) or any(a not in ('wait', 'sound') for a in actions):
        raise ValueError('distinct prescribed chosen actions must be sound or wait')
    previous = {f.parent.name: f for f in args.previous.glob('group-*/*/issued.json')}
    cases = json.loads((args.input/'inputs.json').read_text())
    if args.name:
        cases = [c for c in cases if c['name'] in args.name]
        if {c['name'] for c in cases} != set(args.name):
            raise ValueError('unknown requested case')
    args.output.mkdir()
    write_json(args.output/'plan.json', plan)
    summaries = []
    for case in cases:
        old = json.loads(previous[case['name']].read_text())
        fs, issue = case['fs'], case['issue_sample']
        source = args.input/case['name']
        past, own_past = [np.fromfile(source/(k+'.f32le'), dtype='<f4').astype(float) for k in ('past_mix', 'past_own')]
        if len(past) != issue or own_past.shape != past.shape:
            raise ValueError('complete pre-action evidence is required')
        external = past-own_past
        old_adapt, old_compare, old_horizon = [round(fs*old_plan['hypotheses'][k])
            for k in ('adaptation_sec', 'selection_sec', 'horizon_sec')]
        start = issue-old_adapt-old_compare
        pending = []
        for row in old['body_candidates']:
            fit = SimpleNamespace(frequency_hz=np.asarray(row['frequency_hz']),
                log_gain_per_sec=np.asarray(row['log_gain_per_sec']), fit_end_sample=issue-old_compare)
            pending.append(('refitted_passive', SimpleNamespace(trajectory=fit), dict(adaptation_start=row['adaptation_start'])))
        original, rows = build_driven_bank(pending, external[start:issue-old_compare], fs, start, plan['driven_body'])
        if rows != old['body_candidates']:
            raise ValueError('pre-action inferred candidates changed during replay')
        original.observe(issue-old_compare, external[issue-old_compare:])
        original_digest = original.forecast(old_horizon).digest()
        if original_digest != old['forecast_sha256']['body_mixture']:
            raise ValueError('pre-action conditional model does not reproduce the previous issued forecast')
        adapt, compare, horizon = [round(fs*plan[k]) for k in ('adaptation_sec', 'comparison_sec', 'horizon_sec')]
        if min(adapt, compare, horizon) < 1 or adapt+compare+horizon != old_horizon:
            raise ValueError('the chosen revision must preserve the original target interval')
        for action in actions:
            directory = args.output/(case['name']+'.'+action)
            directory.mkdir()
            owned = np.fromfile(source/('own_'+action+'.f32le'), dtype='<f4', count=old_horizon).astype(float)
            adaptation = np.fromfile(source/('truth_'+action+'.f32le'), dtype='<f4', count=adapt).astype(float)
            if len(owned) != old_horizon or len(adaptation) != adapt or not np.isfinite(owned).all():
                raise ValueError('complete known owned future and chosen adaptation are required')
            banks, revision = revise_body_bank(original, rows, adaptation-owned[:adapt], fs, plan['driven_body'],
                counts=plan['fresh_counts'], offsets_sec=plan['fresh_offsets_sec'], prior_mass=plan['revision_prior_mass'])
            forecasts = {name: bank.forecast(compare) for name, bank in banks.items()}
            write_json(directory/'proposed.json', dict(case=case, chosen_action=action, revision=revision,
                original_forecast_sha256=original_digest, comparison_forecast_sha256={n:f.digest() for n,f in forecasts.items()},
                before_comparison_log_weights={n:b.log_weights.tolist() for n,b in banks.items()},
                adaptation_sha256=hashlib.sha256(adaptation.tobytes()).hexdigest()))
            comparison = np.fromfile(source/('truth_'+action+'.f32le'), dtype='<f4', count=compare, offset=adapt*4).astype(float)
            if len(comparison) != compare or not np.isfinite(comparison).all():
                raise ValueError('complete common chosen comparison is required')
            scores = {}
            for name, bank in banks.items():
                density, components = bank.observe(issue+adapt, comparison-owned[adapt:adapt+compare])
                scores[name] = dict(comparison_log_density=density, component_comparison_log_density=components.tolist(),
                    after_comparison_log_weights=bank.log_weights.tolist())
            if {n:f.digest() for n,f in forecasts.items()} != json.loads((directory/'proposed.json').read_text())['comparison_forecast_sha256']:
                raise ValueError('comparison predictions changed after observing their targets')
            one_block = copy.deepcopy(original)
            one_block.observe(issue, np.r_[adaptation,comparison]-owned[:adapt+compare])
            reference = one_block.forecast(horizon)
            retained = banks['retained'].forecast(horizon)
            np.testing.assert_allclose(retained.mean,reference.mean,rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose(retained.diagonal_variance(),reference.diagonal_variance(),rtol=1e-10,atol=1e-12)
            chunking = dict(max_mean_difference=float(np.max(np.abs(retained.mean-reference.mean))),
                max_variance_difference=float(np.max(np.abs(retained.diagonal_variance()-reference.diagonal_variance()))),
                max_log_weight_difference=float(np.max(np.abs(one_block.log_weights-banks['retained'].log_weights))))
            prefix = np.r_[past, adaptation, comparison]
            prefix_path = directory/'observed-past.f32le'
            np.asarray(prefix, dtype='<f4').tofile(prefix_path)
            branches, records, moments = [], {}, {}
            sampling = plan['perceptual_sampling']
            for name, bank in banks.items():
                forecast = bank.forecast(horizon).with_known_waveform(owned[adapt+compare:])
                draws, record = stratified_samples(forecast, np.random.default_rng([sampling['seed'], case['seed'], fs]),
                    target_draws=sampling['target_draws'], min_per_component=sampling['min_per_component'],
                    max_omitted_mass=sampling['max_omitted_mass'])
                records[name] = dict(forecast_sha256=forecast.digest(), sampling=record,
                    log_weights=bank.log_weights.tolist(), **scores[name])
                moments[name+'.mean'] = forecast.mean
                moments[name+'.variance'] = forecast.diagonal_variance()
                for i, audio in enumerate(draws):
                    path = directory/f'{name}.draw.{i}.f32le'
                    with np.errstate(over='raise', invalid='raise'):
                        value = np.asarray(audio, dtype='<f4')
                    if not np.isfinite(value).all():
                        raise ArithmeticError('nonfinite revised native forecast')
                    value.tofile(path)
                    branches.append(dict(id=f'{name}.draw.{i}', origin='predictive', audio=str(path.resolve())))
            np.savez_compressed(directory/'forecast-moments.npz', **moments)
            write_json(directory/'issued.json', dict(issued_sample=issue+adapt+compare, target_end_sample=issue+old_horizon,
                forecasts=records, retained_chunking_control=chunking,
                chosen_prefix_sha256=hashlib.sha256(np.r_[adaptation,comparison].tobytes()).hexdigest(),
                revision_group_log_mass={n:float(np.logaddexp.reduce(banks['revision'].log_weights[s])) for n,s in
                    [('retained',slice(0,len(rows))),('fresh',slice(len(rows),None))]}))
            native_case = dict(id=directory.name, sample_rate=fs, issued_sample=issue+adapt+compare, bus='habitat',
                past=str(prefix_path.resolve()), output=str((directory/'predicted.jsonl').resolve()), branches=branches)
            native(args.config, [native_case], directory, 'predicted')
            contract, predictions = read_results(directory/'predicted.jsonl')
            seal = hashlib.sha256((directory/'predicted.jsonl').read_bytes()).hexdigest()
            write_json(directory/'prediction-seal.json', dict(sha256=seal))
            target = np.fromfile(source/('truth_'+action+'.f32le'), dtype='<f4', count=horizon, offset=(adapt+compare)*4).astype(float)
            if len(target) != horizon or not np.isfinite(target).all():
                raise ValueError('complete selected evaluation target is required')
            target_path = directory/'actual.f32le'
            np.asarray(target,dtype='<f4').tofile(target_path)
            native(args.config, [dict(native_case, output=str((directory/'actual.jsonl').resolve()),
                branches=[dict(id='actual',origin='perceptual',audio=str(target_path.resolve()))])], directory, 'actual')
            actual_contract, actual = read_results(directory/'actual.jsonl')
            truth = actual['actual']['observation']
            if actual_contract != contract or truth is None or any(r['observation'] is None or r['observation']['end_sample'] != truth['end_sample'] for r in predictions.values()):
                raise ValueError('revised predictions must share the actual complete native target')
            metrics, integrated = {}, {}
            for name, record in records.items():
                forecast = banks[name].forecast(horizon).with_known_waveform(owned[adapt+compare:])
                if forecast.digest() != record['forecast_sha256']:
                    raise ValueError('issued revised model changed during evaluation')
                fields = {}
                for field in FIELDS:
                    values = np.asarray([predictions[f'{name}.draw.{i}']['observation'][field] for i in range(record['sampling']['actual_draws'])])
                    estimate = bounded_summary(values, record['sampling'], alpha=sampling['alpha'])
                    for key in ('omission_midpoint','monte_carlo_se','numerical_lower','numerical_upper'):
                        integrated[f'{name}.{field}.{key}'] = estimate[key]
                    fields[field] = dict(mse=float(np.mean((estimate['omission_midpoint']-truth[field])**2)),
                        mean_numerical_se=float(np.mean(estimate['monte_carlo_se'])),
                        max_numerical_se=float(np.max(estimate['monte_carlo_se'])))
                metrics[name] = dict(pcm_mse=float(np.mean((forecast.mean-target)**2)),
                    target_joint_log_density=forecast.log_density(target), fields=fields)
            if hashlib.sha256((directory/'predicted.jsonl').read_bytes()).hexdigest() != seal:
                raise ValueError('issued native predictions changed after target access')
            if original.next_sample != issue or original.forecast(old_horizon).digest() != original_digest:
                raise ValueError('an independent chosen episode modified the original model')
            np.savez_compressed(directory/'integration-moments.npz', **integrated)
            result = dict(case=case, chosen_action=action, issued_sample=issue+adapt+compare,
                end_sample=truth['end_sample'], metrics=metrics)
            write_json(directory/'result.json', result)
            summaries.append(result)
            print('scored', directory.name, flush=True)
    write_json(args.output/'summary.json', summaries)


if __name__ == '__main__':
    main()
