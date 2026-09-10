#!/usr/bin/env python3
"""Issue owned-action trajectories, observe native perceptual consequences, then score.

The Rust test assay owns the auditory transform. These commands only prepare
finite trajectories and compare its outputs; no R/H kernel is reimplemented.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_action_conditioned import forecast_branches


FIELDS = ('r_state01_scan', 'h_state01_scan', 'c_field_level_scan')


def write_json(path, value):
    with path.open('x') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')


def read_results(path):
    with path.open() as source:
        contract = json.loads(next(source))
        rows = [json.loads(line) for line in source]
    if contract['type'] != 'contract' or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('invalid native results')
    return contract, {row['id']: row for row in rows}


def prepare(args):
    plan = json.loads(args.plan.read_text())
    if plan['draws'] < 4:
        raise ValueError('at least four draws are required')
    args.output.mkdir()
    inputs = json.loads((args.input / 'inputs.json').read_text())
    cases, records = [], []
    for case in inputs:
        source = args.input / case['name']
        output = args.output / case['name']
        output.mkdir()
        history = {name: np.fromfile(source / (name + '.f32le'), dtype='<f4').astype(float)
                   for name in ('past_mix', 'past_own', 'own_wait', 'own_sound')}
        if len(history['past_mix']) != case['issue_sample']:
            raise ValueError('complete history from sample zero is required')
        external, forecasts, selection = forecast_branches(
            **history, fs=case['fs'], issued=case['issue_sample'], plan=plan['forecast'])
        rng = np.random.default_rng([plan['sampling_seed'], case['seed'], case['fs']])
        # Shared environmental draws preserve the paired action comparison.
        draws = external.sample(rng, plan['draws'])
        branches = []
        digests = {}
        for action, forecast in forecasts.items():
            trajectories = [('mean', forecast.mean)] + [
                (f'draw.{index}', draw + history['own_' + action]) for index, draw in enumerate(draws)]
            for label, audio in trajectories:
                branch_id = action + '.' + label
                destination = output / (branch_id + '.f32le')
                values = np.asarray(audio, dtype='<f4')
                if not np.isfinite(values).all():
                    raise ValueError('non-finite forward trajectory')
                with destination.open('xb') as stream:
                    stream.write(values.tobytes())
                digests[branch_id] = hashlib.sha256(values.tobytes()).hexdigest()
                branches.append(dict(id=branch_id, origin='predictive', audio=str(destination.resolve())))
        record = dict(case=case, selection=selection, samples=plan['draws'],
                      innovation_variance=external.innovation_variance,
                      trajectory_sha256=digests,
                      evidence_sha256={name: hashlib.sha256(values.tobytes()).hexdigest()
                                       for name, values in history.items()})
        write_json(output / 'issued.json', record)
        # Preserve the joint distribution, not just the finite Monte Carlo draws.
        np.savez_compressed(output / 'distribution.npz', external_mean=external.mean,
                            impulse=external.impulse, amplitude_factor=external.amplitude_factor,
                            reflection=external.reflection)
        cases.append(dict(id=case['name'], sample_rate=case['fs'], issued_sample=case['issue_sample'],
                          bus='habitat', past=str((source / 'past_mix.f32le').resolve()),
                          output=str((output / 'predicted.jsonl').resolve()), branches=branches))
        records.append(record)
        print('issued', case['name'], flush=True)
    write_json(args.output / 'plan.json', plan)
    write_json(args.output / 'issued.json', records)
    write_json(args.output / 'predicted-plan.json', dict(config=str(args.config.resolve()), cases=cases))


def prepare_actual(args):
    plan = json.loads((args.output / 'predicted-plan.json').read_text())
    seals = {}
    # Every prediction must have passed the native observer before actual paths are released.
    for case in plan['cases']:
        path = Path(case['output'])
        contract, results = read_results(path)
        if (contract['issued_sample'] != case['issued_sample']
                or set(results) != {branch['id'] for branch in case['branches']}):
            raise ValueError('incomplete issued observations')
        seals[case['id']] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(args.output / 'prediction-seal.json', seals)
    cases = []
    for case in plan['cases']:
        actual = dict(case)
        actual['output'] = str((args.output / case['id'] / 'actual.jsonl').resolve())
        actual['branches'] = [dict(id=action, origin='perceptual',
            audio=str((args.input / case['id'] / ('truth_' + action + '.f32le')).resolve()))
            for action in ('wait', 'sound')]
        cases.append(actual)
    write_json(args.output / 'actual-plan.json', dict(config=plan['config'], cases=cases))


def compare(contract, predicted, actual, samples):
    """Score the last complete native hop without inventing pending evidence."""
    result = dict(branches={}, action_difference={})
    for action in ('wait', 'sound'):
        observed = actual[action]['observation']
        if (observed is None or observed['end_sample'] - contract['issued_sample'] < contract['hop_samples']):
            return dict(status='no_complete_future_hop')
        model = predicted[action + '.mean']['observation']
        draws = [predicted[f'{action}.draw.{index}']['observation'] for index in range(samples)]
        if model is None or any(draw is None for draw in draws):
            raise ValueError('prediction is missing the observed target')
        if any(row['end_sample'] != observed['end_sample'] for row in [model] + draws):
            raise ValueError('prediction and observation times differ')
        scores = {}
        for field in FIELDS:
            target = np.asarray(observed[field], dtype=float)
            distribution = np.array([draw[field] for draw in draws], dtype=float)
            mean = distribution.mean(axis=0)
            lower, upper = np.quantile(distribution, [.05, .95], axis=0)
            baseline = np.array(contract['observed_' + field])
            scores[field] = dict(
                expected_field_mse=float(np.mean((mean-target)**2)),
                mean_waveform_field_mse=float(np.mean((np.array(model[field])-target)**2)),
                persistence_mse=float(np.mean((baseline-target)**2)),
                marginal_90_coverage=float(np.mean((target>=lower)&(target<=upper))),
                nonlinear_mean_gap_rms=float(np.sqrt(np.mean((mean-np.array(model[field]))**2))),
                half_draw_mean_change_rms=float(np.sqrt(np.mean((distribution[:samples//2].mean(axis=0)-mean)**2))))
        result['branches'][action] = scores
    for field in FIELDS:
        targets = {a: np.array(actual[a]['observation'][field]) for a in ('wait', 'sound')}
        expected = {a: np.array([predicted[f'{a}.draw.{i}']['observation'][field]
                                 for i in range(samples)]).mean(axis=0) for a in ('wait', 'sound')}
        delta = targets['sound']-targets['wait']
        predicted_delta = expected['sound']-expected['wait']
        result['action_difference'][field] = dict(
            mse=float(np.mean((predicted_delta-delta)**2)),
            zero_difference_mse=float(np.mean(delta**2)),
            predicted_rms=float(np.sqrt(np.mean(predicted_delta**2))),
            observed_rms=float(np.sqrt(np.mean(delta**2))))
    result.update(status='observed', end_sample=observed['end_sample'],
                  startup_zero_samples=observed['startup_zero_samples'])
    return result


def score(args):
    seals = json.loads((args.output / 'prediction-seal.json').read_text())
    records = json.loads((args.output / 'issued.json').read_text())
    results = []
    for record in records:
        case = record['case']
        directory = args.output / case['name']
        if hashlib.sha256((directory / 'predicted.jsonl').read_bytes()).hexdigest() != seals[case['name']]:
            raise ValueError('issued perceptual predictions changed')
        contract, predicted = read_results(directory / 'predicted.jsonl')
        actual_contract, actual = read_results(directory / 'actual.jsonl')
        if actual_contract != contract:
            raise ValueError('different observed history or analysis contract')
        result = compare(contract, predicted, actual, record['samples'])
        result.update(name=case['name'], body=case['body'], route=case['route'],
                      external_condition=case['external_condition'], sample_rate=case['fs'])
        if case['route'] == 'presentation' and result['status'] == 'observed':
            for metric in result['action_difference'].values():
                if metric['predicted_rms'] != 0. or metric['observed_rms'] != 0.:
                    raise ValueError('presentation-only action changed habitat evidence')
        results.append(result)
    write_json(args.output / 'summary.json', results)
    print('scored', len(results), 'cases', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'prepare-actual', 'score'))
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--config', type=Path, default=Path('config.toml'))
    args = parser.parse_args()
    if args.command == 'prepare' and args.plan is None:
        parser.error('prepare requires --plan')
    {'prepare': prepare, 'prepare-actual': prepare_actual, 'score': score}[args.command](args)


if __name__ == '__main__':
    main()
