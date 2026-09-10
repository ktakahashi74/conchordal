"""Prospective paired sound/wait consequences and isolated chosen-outcome updates.

The shared input construction extends evaluate_perceptual_action.py. Candidate
banks and stratified native integration come from the later continuation assays.
No future outcome enters pre-action fitting, and offline unchosen outcomes do not
update a model. Ecological preferences and action-value learning remain separate.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from acoustic_hypotheses import propose_hypotheses
from evaluate_perceptual_action import FIELDS, read_results, write_json
from evaluate_structural_predictions import build_bank, build_driven_bank
from perceptual_sampling import bounded_summary, stratified_samples


def native(config, cases, directory, label):
    path = directory/(label+'-plan.json')
    write_json(path, dict(config=str(config.resolve()), cases=cases))
    command = ['cargo', 'test', '--lib', 'runtime::perceptual_action_assay::export_perceptual_action_assay',
               '--', '--ignored', '--exact', '--nocapture']
    with (directory/(label+'-native.log')).open('x') as output:
        subprocess.run(command, env=dict(os.environ, CONCHORDAL_PERCEPTUAL_ACTION_PLAN=str(path.resolve())),
                       stdout=output, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--name', action='append')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    choices = plan['chosen_actions']
    if not choices or len(set(choices)) != len(choices) or any(c not in ('wait', 'sound') for c in choices):
        raise ValueError('distinct chosen actions must be wait or sound')
    cases = json.loads((args.input/'inputs.json').read_text())
    if args.name:
        cases = [c for c in cases if c['name'] in args.name]
        if {c['name'] for c in cases} != set(args.name):
            raise ValueError('unknown requested case')
    args.output.mkdir()
    write_json(args.output/'plan.json', plan)
    summaries = []
    for case in cases:
        source, directory = args.input/case['name'], args.output/case['name']
        directory.mkdir()
        fs, issued = case['fs'], case['issue_sample']
        hypothesis, sampling = plan['hypotheses'], plan['perceptual_sampling']
        adapt, compare, horizon = [round(hypothesis[k]*fs) for k in ('adaptation_sec', 'selection_sec', 'horizon_sec')]
        update = round(plan['update_sec']*fs)
        if not 0 < update < horizon:
            raise ValueError('chosen observation must end inside the forecast horizon')
        past, own_past = [np.fromfile(source/(key+'.f32le'), dtype='<f4').astype(float)
                          for key in ('past_mix', 'past_own')]
        owned = {action: np.fromfile(source/('own_'+action+'.f32le'), dtype='<f4', count=horizon).astype(float)
                 for action in ('wait', 'sound')}
        if (len(past) != issued or past.shape != own_past.shape
                or any(len(x) != horizon or not np.isfinite(x).all() for x in owned.values())):
            raise ValueError('complete pre-action history and owned forward branches are required')
        external_history = past-own_past
        start = issued-adapt-compare
        pending, unavailable, proposal = propose_hypotheses(
            external_history[:start], external_history[start:start+adapt], fs, start, hypothesis, learn_variance=True)
        structural, structural_record = build_bank(pending, unavailable, plan['family_prior_mass'], compare+horizon)
        body, body_rows = build_driven_bank(pending, external_history[start:start+adapt], fs, start, plan['driven_body'])
        banks = dict(mixture=structural, body_mixture=body)
        comparison_hashes = {name: bank.forecast(compare).digest() for name, bank in banks.items()}
        for bank in banks.values():
            bank.observe(start+adapt, external_history[start+adapt:issued])
        if any(bank.next_sample != issued for bank in banks.values()):
            raise ValueError('pre-action models must stop at the shared issue boundary')
        forecasts = {name: bank.forecast(horizon) for name, bank in banks.items()}
        before = {name: f.digest() for name, f in forecasts.items()}
        records, branches, forecast_arrays = {}, [], {}
        for name, forecast in forecasts.items():
            draws, record = stratified_samples(forecast, np.random.default_rng([sampling['seed'], case['seed'], fs]),
                target_draws=sampling['target_draws'], min_per_component=sampling['min_per_component'],
                max_omitted_mass=sampling['max_omitted_mass'])
            records[name] = record
            forecast_arrays[name+'.external_mean'] = forecast.mean
            forecast_arrays[name+'.external_variance'] = forecast.diagonal_variance()
            record['environment_draws_sha256'] = hashlib.sha256(draws.tobytes()).hexdigest()
            for action in ('wait', 'sound'):
                forecast_arrays[name+'.'+action+'.mean'] = forecast.mean+owned[action]
                for index, audio in enumerate(draws+owned[action]):
                    path = directory/f'{name}.{action}.draw.{index}.f32le'
                    with np.errstate(over='raise', invalid='raise'):
                        values = np.asarray(audio, dtype='<f4')
                    if not np.isfinite(values).all():
                        raise ArithmeticError('nonfinite prospective native trajectory')
                    values.tofile(path)
                    branches.append(dict(id=f'{name}.{action}.draw.{index}', origin='predictive', audio=str(path.resolve())))
        np.savez_compressed(directory/'forecast-moments.npz', **forecast_arrays)
        write_json(directory/'issued.json', dict(case=case, issued_sample=issued, target_end_sample=issued+horizon,
            evidence_end_sample=issued, proposal=proposal, structural_candidates=structural_record,
            body_candidates=body_rows, comparison_forecast_sha256=comparison_hashes,
            forecast_sha256=before, sampling=records,
            log_weights={name: bank.log_weights.tolist() for name, bank in banks.items()},
            evidence_sha256={key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in
                             dict(past=past, own_past=own_past, own_wait=owned['wait'], own_sound=owned['sound']).items()}))
        native_case = dict(id=case['name'], sample_rate=fs, issued_sample=issued, bus='habitat',
            past=str((source/'past_mix.f32le').resolve()), output=str((directory/'predicted.jsonl').resolve()), branches=branches)
        native(args.config, [native_case], directory, 'predicted')
        contract, observations = read_results(directory/'predicted.jsonl')
        first = observations[branches[0]['id']]['observation']
        if first is None:
            raise ValueError('prospective target has no complete future hop')
        end = first['end_sample']
        if (end-issued < contract['hop_samples'] or any(row['observation'] is None or
                row['observation']['end_sample'] != end for row in observations.values())):
            raise ValueError('prospective observations must share a complete future hop')
        estimates, numeric = {}, {}
        for name, record in records.items():
            numeric[name] = {}
            for field in FIELDS:
                values = {action: np.array([observations[f'{name}.{action}.draw.{i}']['observation'][field]
                                            for i in range(record['actual_draws'])]) for action in ('wait', 'sound')}
                paired = bounded_summary(values['sound']-values['wait'], record, alpha=sampling['alpha'], bounds=(-1., 1.))
                summaries_by_action = {a: bounded_summary(v, record, alpha=sampling['alpha']) for a, v in values.items()}
                for action, estimate in [*summaries_by_action.items(), ('difference', paired)]:
                    for key in ('omission_midpoint', 'monte_carlo_se', 'numerical_lower', 'numerical_upper'):
                        estimates[f'{name}.{action}.{field}.{key}'] = estimate[key]
                numeric[name][field] = dict(omitted_mass=paired['omitted_mass'],
                    mean_paired_numerical_variance=float(np.mean(paired['monte_carlo_se']**2)),
                    mean_variance_if_independent=float(np.mean(sum(e['monte_carlo_se']**2 for e in summaries_by_action.values()))),
                    max_paired_numerical_se=float(np.max(paired['monte_carlo_se'])),
                    hoeffding_radius=paired['hoeffding_radius'])
        np.savez_compressed(directory/'prospective-estimates.npz', **estimates)
        write_json(directory/'numerical.json', numeric)
        seal = hashlib.sha256((directory/'predicted.jsonl').read_bytes()).hexdigest()
        write_json(directory/'prediction-seal.json', dict(sha256=seal))
        updates = {}
        # Each chosen episode starts from the identical pre-action state.
        for action in choices:
            prefix = np.fromfile(source/('truth_'+action+'.f32le'), dtype='<f4', count=update).astype(float)
            if len(prefix) != update or not np.isfinite(prefix).all():
                raise ValueError('incomplete chosen outcome prefix')
            updated_branches, updated_records = [], {}
            for name, original in banks.items():
                bank = copy.deepcopy(original)
                density, component_scores = bank.observe(issued, prefix-owned[action][:update])
                future = bank.forecast(horizon-update).with_known_waveform(owned[action][update:])
                draws, record = stratified_samples(future,
                    np.random.default_rng([sampling['seed']+1, case['seed'], fs]),
                    target_draws=sampling['target_draws'], min_per_component=sampling['min_per_component'],
                    max_omitted_mass=sampling['max_omitted_mass'])
                updated_records[name] = dict(forecast_sha256=future.digest(), chosen_prefix_log_density=density,
                    component_prefix_log_density=component_scores.tolist(), log_weights=bank.log_weights.tolist(), sampling=record)
                for index, audio in enumerate(draws):
                    path = directory/f'updated.{action}.{name}.draw.{index}.f32le'
                    with np.errstate(over='raise', invalid='raise'):
                        values = np.asarray(audio, dtype='<f4')
                    if not np.isfinite(values).all():
                        raise ArithmeticError('nonfinite chosen continuation trajectory')
                    values.tofile(path)
                    updated_branches.append(dict(id=f'{name}.draw.{index}', origin='predictive', audio=str(path.resolve())))
            write_json(directory/f'updated-{action}-issued.json', dict(chosen_action=action, evidence_end_sample=issued+update,
                target_end_sample=issued+horizon, chosen_prefix_sha256=hashlib.sha256(prefix.tobytes()).hexdigest(), models=updated_records))
            past_path = directory/f'updated-{action}-past.f32le'
            np.asarray(np.r_[past, prefix], dtype='<f4').tofile(past_path)
            update_case = dict(native_case, id=case['name']+'.'+action, issued_sample=issued+update,
                past=str(past_path.resolve()), output=str((directory/f'updated-{action}.jsonl').resolve()), branches=updated_branches)
            native(args.config, [update_case], directory, 'updated-'+action)
            updated_contract, updated_observations = read_results(directory/f'updated-{action}.jsonl')
            if any(r['observation'] is None or r['observation']['end_sample'] != end for r in updated_observations.values()):
                raise ValueError('chosen continuation changed the native target boundary')
            updated_arrays = {}
            for name, record in updated_records.items():
                for field in FIELDS:
                    values = np.array([updated_observations[f'{name}.draw.{i}']['observation'][field]
                                       for i in range(record['sampling']['actual_draws'])])
                    estimate = bounded_summary(values, record['sampling'], alpha=sampling['alpha'])
                    for key in ('omission_midpoint', 'monte_carlo_se', 'numerical_lower', 'numerical_upper'):
                        updated_arrays[f'{name}.{field}.{key}'] = estimate[key]
            np.savez_compressed(directory/f'updated-{action}-estimates.npz', **updated_arrays)
            updates[action] = (updated_records, updated_arrays, update_case)
        update_seals = {action: hashlib.sha256((directory/f'updated-{action}.jsonl').read_bytes()).hexdigest() for action in choices}
        write_json(directory/'updates-issued.json', dict(chosen_actions=choices, sha256=update_seals))
        assert {name: f.digest() for name, f in forecasts.items()} == before
        assert all(bank.next_sample == issued for bank in banks.values())
        # Both realized futures are now available only for offline evaluation.
        actual_branches, actual_cases = [], []
        for action in ('wait', 'sound'):
            audio = np.fromfile(source/('truth_'+action+'.f32le'), dtype='<f4', count=horizon)
            if len(audio) != horizon or not np.isfinite(audio).all():
                raise ValueError('incomplete offline branch target')
            path = directory/f'actual-{action}.f32le'
            audio.tofile(path)
            actual_branches.append(dict(id=action, origin='perceptual', audio=str(path.resolve())))
            if action in updates:
                tail = directory/f'actual-{action}-tail.f32le'
                audio[update:].tofile(tail)
                actual_cases.append(dict(updates[action][2], output=str((directory/f'actual-updated-{action}.jsonl').resolve()),
                    branches=[dict(id=action, origin='perceptual', audio=str(tail.resolve()))]))
        actual_cases.insert(0, dict(native_case, output=str((directory/'actual.jsonl').resolve()), branches=actual_branches))
        native(args.config, actual_cases, directory, 'actual')
        actual_contract, actual = read_results(directory/'actual.jsonl')
        if actual_contract != contract or hashlib.sha256((directory/'predicted.jsonl').read_bytes()).hexdigest() != seal:
            raise ValueError('prospective observed state or issued prediction changed')
        for action in choices:
            if hashlib.sha256((directory/f'updated-{action}.jsonl').read_bytes()).hexdigest() != update_seals[action]:
                raise ValueError('issued chosen-continuation observation changed')
            _, continued = read_results(directory/f'actual-updated-{action}.jsonl')
            if continued[action]['observation'] != actual[action]['observation']:
                raise ValueError('observing a prefix must reproduce the same actual target')
        metrics = {}
        for name in banks:
            metrics[name] = {}
            for field in FIELDS:
                truth = {a: np.asarray(actual[a]['observation'][field]) for a in ('wait', 'sound')}
                if any(actual[a]['observation']['end_sample'] != end for a in truth):
                    raise ValueError('offline target differs from the issued target')
                delta = truth['sound']-truth['wait']
                predicted = estimates[f'{name}.difference.{field}.omission_midpoint']
                metrics[name][field] = dict(difference_mse=float(np.mean((predicted-delta)**2)),
                    zero_difference_mse=float(np.mean(delta**2)), predicted_difference_rms=float(np.sqrt(np.mean(predicted**2))),
                    observed_difference_rms=float(np.sqrt(np.mean(delta**2))),
                    branch_mse={a: float(np.mean((estimates[f'{name}.{a}.{field}.omission_midpoint']-truth[a])**2)) for a in truth},
                    updated_mse={a: float(np.mean((updates[a][1][f'{name}.{field}.omission_midpoint']-truth[a])**2)) for a in choices})
        result = dict(case=case, issued_sample=issued, end_sample=end, numerical=numeric, metrics=metrics,
                      chosen_actions=choices, scope='Prospective consequences and isolated chosen-outcome updates; no action-value policy.')
        write_json(directory/'result.json', result)
        summaries.append(result)
        print('scored', case['name'], flush=True)
    write_json(args.output/'summary.json', summaries)


if __name__ == '__main__':
    main()
