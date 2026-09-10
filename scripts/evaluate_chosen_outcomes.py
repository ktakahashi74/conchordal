#!/usr/bin/env python3
"""Learn acoustic innovation scale from only the selected action's actual PCM.

Fixed and learned variance controls start with identical means and covariances.
The native auditory assay transforms their updated joint predictions. Other
actors are nonreactive controls; this is not policy learning or causal attribution.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from acoustic_posterior import AcousticPosterior
from evaluate_action_conditioned import fit_external_posterior
from evaluate_perceptual_action import FIELDS, read_results, write_json


def build_models(past, own, fs, issued, horizon, plan):
    template, selection = fit_external_posterior(past, own, fs, issued, plan, horizon)
    start = selection['external_evidence_start']
    external = (past-own)[start-(issued-len(past)):]
    learned = AcousticPosterior(template.trajectory, template.residual, external, start, learn_variance=True)
    initial_variance = learned.forecast(1).innovation_variance
    fixed = AcousticPosterior(template.trajectory, replace(template.residual, drive_variance=initial_variance),
                              external, start)
    return dict(fixed=fixed, learned=learned), selection


def observe_chosen(models, start, actual, owned, chunk_samples):
    """Only a contiguous chosen mixture and its known own contribution are accepted."""
    actual, owned = np.asarray(actual, dtype=float), np.asarray(owned, dtype=float)
    if (actual.ndim != 1 or actual.shape != owned.shape or not len(actual)
            or not np.isfinite(actual).all() or not np.isfinite(owned).all()
            or chunk_samples < 1 or any(model.next_sample != start for model in models.values())):
        raise ValueError('expected contiguous chosen audio and matching known owned contribution')
    external = actual-owned
    ledger = []
    for begin in range(0, len(actual), chunk_samples):
        end = min(begin+chunk_samples, len(actual))
        states = {}
        for name, model in models.items():
            forecast = model.forecast(end-begin)
            states[name] = dict(issued_sample=start+begin, observed_end_sample=start+end,
                forecast_sha256=forecast.digest(), variance_before=forecast.innovation_variance,
                pre_update_log_density=model.observe(start+begin, external[begin:end]),
                variance_after=model.forecast(1).innovation_variance,
                variance_shape=model.variance_shape, variance_scale=model.variance_scale)
        ledger.append(states)
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config.toml'))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan['draws'] < 4 or plan['observe_sec'] <= 0:
        raise ValueError('positive observation and at least four draws required')
    args.output.mkdir()
    write_json(args.output/'plan.json', plan)
    cases = json.loads((args.input/'inputs.json').read_text())
    episodes = []
    # Issue every initial distribution before opening any action result.
    for case in cases:
        source = args.input/case['name']
        fs, issued = case['fs'], case['issue_sample']
        history = {key: np.fromfile(source/(key+'.f32le'), dtype='<f4').astype(float)
                   for key in ('past_mix', 'past_own', 'own_wait', 'own_sound')}
        horizon = len(history['own_wait'])
        if len(history['own_sound']) != horizon or len(history['past_mix']) != issued:
            raise ValueError('incomplete prior observations or owned forecasts')
        models, selection = build_models(history['past_mix'], history['past_own'], fs, issued, horizon, plan['forecast'])
        for action in ('wait', 'sound'):
            directory = args.output/(case['name']+'.'+action)
            directory.mkdir()
            owned = history['own_'+action]
            priors = {name: model.forecast(horizon).with_known_waveform(owned) for name, model in models.items()}
            np.testing.assert_array_equal(priors['fixed'].mean, priors['learned'].mean)
            np.testing.assert_array_equal(priors['fixed'].diagonal_variance(), priors['learned'].diagonal_variance())
            arrays = {}
            for name, forecast in priors.items():
                for field in ('mean', 'impulse', 'amplitude_factor', 'reflection'):
                    arrays[name+'_'+field] = getattr(forecast, field)
            np.savez_compressed(directory/'initial.npz', **arrays)
            record = dict(case=case, chosen_action=action, selection=selection,
                forecast_sha256={name: forecast.digest() for name, forecast in priors.items()},
                initial_variance=priors['learned'].innovation_variance,
                initial_variance_shape=priors['learned'].variance_shape,
                initial_variance_scale=priors['learned'].variance_scale,
                evidence_sha256={key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in history.items()})
            write_json(directory/'initial.json', record)
            episodes.append(dict(directory=directory, source=source, case=case, action=action,
                past=history['past_mix'], owned=owned, models=copy.deepcopy(models), priors=priors, record=record))
    predicted_cases = []
    for episode in episodes:
        directory, source, case = episode['directory'], episode['source'], episode['case']
        fs, issued, action = case['fs'], case['issue_sample'], episode['action']
        observed_count = round(plan['observe_sec']*fs)
        horizon = len(episode['owned'])
        if not 0 < observed_count < horizon:
            raise ValueError('observation must leave an unobserved target')
        actual = np.fromfile(source/('truth_'+action+'.f32le'), dtype='<f4', count=observed_count).astype(float)
        if len(actual) != observed_count:
            raise ValueError('incomplete selected prefix')
        ledger = observe_chosen(episode['models'], issued, actual, episode['owned'][:observed_count], fs//1000)
        write_json(directory/'updates.json', dict(chosen_action=action, ledger=ledger,
            observed_prefix_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
        np.asarray(np.r_[episode['past'], actual], dtype='<f4').tofile(directory/'observed-past.f32le')
        forecasts = {name: model.forecast(horizon-observed_count).with_known_waveform(episode['owned'][observed_count:])
                     for name, model in episode['models'].items()}
        np.testing.assert_array_equal(forecasts['fixed'].mean, forecasts['learned'].mean)
        episode.update(forecasts=forecasts, observed_count=observed_count)
        arrays, branches, metadata = {}, [], {}
        for name, forecast in forecasts.items():
            for field in ('mean', 'impulse', 'amplitude_factor', 'reflection'):
                arrays[name+'_'+field] = getattr(forecast, field)
            rng = np.random.default_rng([plan['sampling_seed'], case['seed'], fs])
            draws = forecast.sample(rng, plan['draws'])
            for label, audio in [('mean', forecast.mean)]+[(f'draw.{i}', row) for i, row in enumerate(draws)]:
                path = directory/(name+'.'+label+'.f32le')
                values = np.asarray(audio, dtype='<f4')
                if not np.isfinite(values).all():
                    raise ValueError('nonfinite predicted trajectory')
                values.tofile(path)
                branches.append(dict(id=name+'.'+label, origin='predictive', audio=str(path.resolve())))
            metadata[name] = dict(forecast_sha256=forecast.digest(), innovation_variance=forecast.innovation_variance,
                                  variance_shape=forecast.variance_shape, variance_scale=forecast.variance_scale)
        np.savez_compressed(directory/'updated.npz', **arrays)
        write_json(directory/'updated.json', dict(issued_sample=issued+observed_count,
            target_end_sample=issued+horizon, forecasts=metadata))
        predicted_cases.append(dict(id=directory.name, sample_rate=fs, issued_sample=issued+observed_count,
            bus='habitat', past=str((directory/'observed-past.f32le').resolve()),
            output=str((directory/'predicted.jsonl').resolve()), branches=branches))
        print('updated', directory.name, flush=True)
    prediction_plan = dict(config=str(args.config.resolve()), cases=predicted_cases)
    write_json(args.output/'predicted-plan.json', prediction_plan)
    command = ['cargo', 'test', '--lib', 'runtime::perceptual_action_assay::export_perceptual_action_assay',
               '--', '--ignored', '--exact', '--nocapture']
    with (args.output/'predicted-native.log').open('x') as output:
        subprocess.run(command, env=dict(os.environ, CONCHORDAL_PERCEPTUAL_ACTION_PLAN=str(args.output/'predicted-plan.json')),
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    seals = {episode['directory'].name: hashlib.sha256((episode['directory']/'predicted.jsonl').read_bytes()).hexdigest()
             for episode in episodes}
    write_json(args.output/'prediction-seal.json', seals)
    actual_cases = []
    for episode, case in zip(episodes, predicted_cases):
        count = episode['observed_count']
        horizon = len(episode['owned'])-count
        # Neither another action's actual audio nor this remaining future was used for updating.
        target = np.fromfile(episode['source']/('truth_'+episode['action']+'.f32le'), dtype='<f4', offset=count*4, count=horizon)
        if len(target) != horizon:
            raise ValueError('incomplete actual target')
        target_path = episode['directory']/'actual-tail.f32le'
        target.tofile(target_path)
        episode['target'] = target.astype(float)
        actual = dict(case, output=str((episode['directory']/'actual.jsonl').resolve()),
                      branches=[dict(id='chosen', origin='perceptual', audio=str(target_path.resolve()))])
        actual_cases.append(actual)
    write_json(args.output/'actual-plan.json', dict(config=prediction_plan['config'], cases=actual_cases))
    with (args.output/'actual-native.log').open('x') as output:
        subprocess.run(command, env=dict(os.environ, CONCHORDAL_PERCEPTUAL_ACTION_PLAN=str(args.output/'actual-plan.json')),
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    results = []
    for episode in episodes:
        directory, case = episode['directory'], episode['case']
        if hashlib.sha256((directory/'predicted.jsonl').read_bytes()).hexdigest() != seals[directory.name]:
            raise ValueError('issued predictions changed')
        contract, predictions = read_results(directory/'predicted.jsonl')
        actual_contract, actual = read_results(directory/'actual.jsonl')
        if contract != actual_contract:
            raise ValueError('different observed auditory state')
        observed = actual['chosen']['observation']
        if observed is None or observed['end_sample']-contract['issued_sample'] < contract['hop_samples']:
            raise ValueError('target contains no complete future hop')
        metrics = {}
        for name, forecast in episode['forecasts'].items():
            fields = {}
            for field in FIELDS:
                target = np.asarray(observed[field])
                samples = np.array([predictions[f'{name}.draw.{i}']['observation'][field] for i in range(plan['draws'])])
                lower, upper = np.quantile(samples, [.05, .95], axis=0)
                fields[field] = dict(mse=float(np.mean((samples.mean(axis=0)-target)**2)),
                    marginal_90_coverage=float(np.mean((target>=lower)&(target<=upper))),
                    interval_width=float(np.mean(upper-lower)))
            metrics[name] = dict(innovation_variance=forecast.innovation_variance,
                joint_log_density=forecast.log_density(episode['target']),
                pcm_mse=float(np.mean((forecast.mean-episode['target'])**2)), fields=fields)
        results.append(dict(name=directory.name, body=case['body'], route=case['route'], fs=case['fs'],
            external_condition=case['external_condition'], chosen_action=episode['action'],
            initial_variance=episode['record']['initial_variance'], metrics=metrics))
    write_json(args.output/'summary.json', results)
    print('scored', len(results), 'chosen-action episodes', flush=True)


if __name__ == '__main__':
    main()
