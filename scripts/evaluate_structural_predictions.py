#!/usr/bin/env python3
"""Compare retained acoustic hypotheses after a chosen action, then score real audio.

Family masses are specified before the common comparison block. They weight
conditional predictions, not perceived source identities or fitted-model evidence.
The native test assay transforms each whole sampled trajectory to R/H/C.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from acoustic_body import DrivenBodyPosterior
from acoustic_hypotheses import propose_hypotheses
from acoustic_mixture import AcousticMixtureForecast, AcousticModelBank
from acoustic_parameters import parameter_forecast, sample_passive_parameters
from acoustic_posterior import AcousticPosterior
from acoustic_trajectory import TrajectoryFit
from evaluate_perceptual_action import FIELDS, read_results, write_json
from perceptual_sampling import bounded_summary, stratified_samples


def build_bank(pending, unavailable, family_prior, count):
    """Admit finite predictions and divide a fixed family mass over its candidates."""
    if (not family_prior or any(not np.isfinite(mass) or mass <= 0 for mass in family_prior.values())
            or any(family not in family_prior for family, _, _ in pending)):
        raise ValueError('positive prior mass is required for every proposed family')
    models, rows = [], []
    rejected = list(unavailable)
    for family, model, meta in pending:
        try:
            forecast = model.forecast(count)
            variance = forecast.diagonal_variance()
            if (not np.isfinite(forecast.mean).all() or not np.isfinite(variance).all()
                    or np.any(variance <= 0) or np.max(np.abs(forecast.mean)) > np.finfo(np.float32).max):
                raise ArithmeticError('unavailable prior forecast moments')
        except (ArithmeticError, np.linalg.LinAlgError) as error:
            rejected.append(dict(family=family, **meta, unavailable=str(error)))
            continue
        models.append(model)
        rows.append(dict(family=family, **meta))
    counts = Counter(row['family'] for row in rows)
    if counts.get('continued_state') != 1:
        raise ArithmeticError('the continued-state control must be available')
    mass = [family_prior[row['family']]/counts[row['family']] for row in rows]
    bank = AcousticModelBank(models, mass)
    for row, weight in zip(rows, bank.log_weights):
        row['prior_log_weight'] = float(weight)
    return bank, dict(candidates=rows, unavailable=rejected, family_counts=dict(counts),
                     family_prior_mass=family_prior)


def build_driven_bank(pending, observed, fs, start_sample, settings):
    """Reuse fitted passive supports, counting each support once across AR orders."""
    observed = np.asarray(observed, dtype=float)
    if observed.ndim != 1 or not len(observed) or not np.isfinite(observed).all():
        raise ValueError('expected a finite observed adaptation block')
    intensities = settings['drive_per_sec']
    if (not intensities or len(set(intensities)) != len(intensities) or 0. not in intensities
            or any(not np.isfinite(d) or d < 0 for d in intensities)):
        raise ValueError('distinct nonnegative drive intensities must include the passive control')
    models, rows, seen = [], [], set()
    available = start_sample + len(observed)
    for family, candidate, meta in pending:
        if family != 'refitted_passive':
            continue
        fit, begin = candidate.trajectory, meta['adaptation_start']
        if not start_sample <= begin < available or fit.fit_end_sample > available:
            raise ValueError('body parameters must come from the observed adaptation')
        key = (begin, tuple(fit.frequency_hz), tuple(fit.log_gain_per_sec))
        if key in seen:
            continue
        seen.add(key)
        for intensity in intensities:
            body = DrivenBodyPosterior(fs, fit.frequency_hz, fit.log_gain_per_sec,
                drive_per_sec=intensity, initial_precision=settings['initial_precision'],
                variance_prior=settings['variance_prior'], start_sample=begin,
                parameters_available_sample=available)
            body.observe(begin, observed[begin-start_sample:])
            models.append(body)
            rows.append(dict(adaptation_start=begin, frequency_hz=fit.frequency_hz.tolist(),
                             log_gain_per_sec=fit.log_gain_per_sec.tolist(), drive_per_sec=intensity))
    return AcousticModelBank(models, np.ones(len(models))), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--name', action='append')
    parser.add_argument('--action', action='append', choices=('wait', 'sound'))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not isinstance(plan['draws'], int) or plan['draws'] < 4:
        raise ValueError('at least four trajectory draws are required')
    cases = json.loads((args.input/'inputs.json').read_text())
    if args.name:
        cases = [case for case in cases if case['name'] in args.name]
        if {case['name'] for case in cases} != set(args.name):
            raise ValueError('unknown requested case')
    args.output.mkdir()
    write_json(args.output/'plan.json', plan)
    results = []
    command = ['cargo', 'test', '--lib', 'runtime::perceptual_action_assay::export_perceptual_action_assay',
               '--', '--ignored', '--exact', '--nocapture']
    # Each episode is independent. No result changes the plan or another episode's state.
    for case in cases:
        source = args.input/case['name']
        fs, issued = case['fs'], case['issue_sample']
        hypothesis = plan['hypotheses']
        adapt, compare, horizon = [round(hypothesis[key]*fs) for key in
                                    ('adaptation_sec', 'selection_sec', 'horizon_sec')]
        total = adapt+compare+horizon
        past, own = [np.fromfile(source/(key+'.f32le'), dtype='<f4').astype(float)
                     for key in ('past_mix', 'past_own')]
        if len(past) != issued or own.shape != past.shape:
            raise ValueError('complete observed history and known own contribution required')
        for action in dict.fromkeys(args.action or ('wait', 'sound')):
            directory = args.output/(case['name']+'.'+action)
            directory.mkdir()
            owned = np.fromfile(source/('own_'+action+'.f32le'), dtype='<f4', count=total).astype(float)
            actual_path = source/('truth_'+action+'.f32le')
            adaptation = np.fromfile(actual_path, dtype='<f4', count=adapt).astype(float)
            if len(owned) != total or len(adaptation) != adapt:
                raise ValueError('incomplete known prediction or chosen adaptation block')
            pending, unavailable, proposal = propose_hypotheses(
                past-own, adaptation-owned[:adapt], fs, issued, hypothesis, learn_variance=True)
            bank, record = build_bank(pending, unavailable, plan['family_prior_mass'], compare+horizon)
            prior = bank.forecast(compare)
            write_json(directory/'proposed.json', dict(case=case, chosen_action=action, proposal=proposal,
                **record, comparison_forecast_sha256=prior.digest(),
                component_forecast_sha256=[component.digest() for component in prior.components],
                evidence_sha256={key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in
                    dict(past=past, own=own, owned_future=owned, adaptation=adaptation).items()}))
            np.savez_compressed(directory/'comparison.npz', mean=prior.mean, variance=prior.diagonal_variance(),
                                log_weights=bank.log_weights)
            if 'driven_body' in plan:
                settings = plan['driven_body']
                body_bank, body_rows = build_driven_bank(pending, adaptation-owned[:adapt], fs, issued, settings)
                body_prior = body_bank.forecast(compare)
                write_json(directory/'body-proposed.json', dict(settings=settings, candidates=body_rows,
                    prior_log_weights=body_bank.log_weights.tolist(), comparison_forecast_sha256=body_prior.digest(),
                    component_forecast_sha256=[c.digest() for c in body_prior.components]))
            # Common comparison evidence cannot enter fitting, admission, or prior assignment.
            comparison = np.fromfile(actual_path, dtype='<f4', count=compare, offset=adapt*4).astype(float)
            if len(comparison) != compare:
                raise ValueError('incomplete chosen comparison block')
            density, scores = bank.observe(issued+adapt, comparison-owned[adapt:adapt+compare])
            if bank.forecast(1).issued_sample != issued+adapt+compare:
                raise ValueError('incorrect posterior observation boundary')
            write_json(directory/'updated.json', dict(issued_sample=bank.next_sample,
                target_end_sample=issued+total, comparison_log_density=density,
                component_comparison_log_density=scores.tolist(), log_weights=bank.log_weights.tolist(),
                comparison_sha256=hashlib.sha256(comparison.tobytes()).hexdigest()))
            mixture = bank.forecast(horizon).with_known_waveform(owned[adapt+compare:])
            control_index = next(i for i, row in enumerate(record['candidates']) if row['family'] == 'continued_state')
            best_index = int(np.argmax(bank.log_weights))
            forecasts = dict(mixture=mixture, continued=mixture.components[control_index],
                             best=mixture.components[best_index])
            if 'driven_body' in plan:
                body_density, body_scores = body_bank.observe(issued+adapt, comparison-owned[adapt:adapt+compare])
                body_forecast = body_bank.forecast(horizon).with_known_waveform(owned[adapt+compare:])
                passive = [i for i, row in enumerate(body_rows) if row['drive_per_sec'] == 0.]
                weights = body_bank.log_weights[passive].copy()
                weights -= np.logaddexp.reduce(weights)
                forecasts['body_mixture'] = body_forecast
                forecasts['body_passive'] = AcousticMixtureForecast(tuple(body_forecast.components[i] for i in passive), weights)
                forecasts['body_best'] = body_forecast.components[int(np.argmax(body_bank.log_weights))]
                write_json(directory/'body-updated.json', dict(comparison_log_density=body_density,
                    component_comparison_log_density=body_scores.tolist(), log_weights=body_bank.log_weights.tolist(),
                    variance_shape=[m.variance_shape for m in body_bank.models],
                    variance_scale=[m.variance_scale for m in body_bank.models],
                    component_forecast_sha256=[c.digest() for c in body_forecast.components]))
            if 'comparison_family_subset' in plan:
                indices = [i for i, row in enumerate(record['candidates'])
                           if row['family'] in plan['comparison_family_subset']]
                if not indices:
                    raise ValueError('the comparison family subset must be available')
                weights = bank.log_weights[indices].copy()
                weights -= np.logaddexp.reduce(weights)
                forecasts['retained'] = AcousticMixtureForecast(tuple(mixture.components[i] for i in indices), weights)
            if 'parameter_uncertainty' in plan:
                settings = plan['parameter_uncertainty']
                indices = [i for i, row in enumerate(record['candidates']) if row['family'] == 'refitted_passive']
                if not indices:
                    raise ArithmeticError('continuous parameters require an observed passive candidate')
                selected = max(indices, key=lambda i: bank.log_weights[i])
                base = bank.models[selected]
                begin = record['candidates'][selected]['adaptation_start']
                evidence = np.r_[adaptation, comparison]-owned[:adapt+compare]
                evidence = evidence[begin-issued:]
                band = np.asarray(settings['frequency_range'], dtype=float)
                span = np.log2(band[1]/band[0])
                unit = np.column_stack(((np.log2(base.trajectory.frequency_hz)-np.log2(band[0]))/span,
                                         -base.trajectory.log_gain_per_sec/settings['max_decay']))
                if np.any(unit < 0) or np.any(unit > 1):
                    raise ValueError('selected candidate is outside the declared parameter prior')
                rng = np.random.default_rng([settings['seed'], case['seed'], fs])
                initial = np.broadcast_to(unit, (4,)+unit.shape).copy()
                initial[1] += rng.normal(0., .03, unit.shape)
                initial[2] += rng.normal(0., .15, unit.shape)
                initial[3] = rng.random(unit.shape)
                initial = 1-abs(initial % 2-1)
                initial[:, :, 0] = band[0]*2**(span*initial[:, :, 0])
                initial[:, :, 1] *= -settings['max_decay']
                parameter_bank, point, parameter_report = sample_passive_parameters(
                    evidence, begin, fs, base.residual, initial,
                    frequency_range=band, max_decay=settings['max_decay'],
                    amplitude_precision=settings['amplitude_precision'], variance_prior=settings['variance_prior'],
                    rng=rng, warmup=settings['warmup'], draws=settings['draws'], thin=settings['thin'])
                fit = TrajectoryFit(fs, float(begin), base.trajectory.frequency_hz, np.zeros(len(unit)),
                    base.trajectory.log_gain_per_sec, np.zeros(len(unit), complex), begin, bank.next_sample)
                initial_point = AcousticPosterior(fit, base.residual, evidence, begin, learn_variance=True,
                    amplitude_precision=settings['amplitude_precision'], variance_prior=settings['variance_prior'])
                forecasts['parameter_mixture'] = parameter_forecast(parameter_bank, horizon).with_known_waveform(
                    owned[adapt+compare:])
                for label, posterior in [('parameter_point', point), ('parameter_initial', initial_point)]:
                    forecasts[label] = posterior.forecast(horizon).with_known_waveform(owned[adapt+compare:])
                np.savez_compressed(directory/'parameter-chains.npz', initial=initial,
                    parameters=parameter_report.pop('parameters'), log_density=parameter_report.pop('log_density'),
                    acceptance=parameter_report.pop('acceptance'), frozen_step_size=parameter_report.pop('frozen_step_size'))
                write_json(directory/'parameters.json', dict(settings=settings, selected_candidate=selected,
                    evidence_start=begin, evidence_end=bank.next_sample, conditioned_initial_samples=len(base.residual.reflection),
                    reflection=base.residual.reflection.tolist(), report=parameter_report,
                    initial_log_density=initial_point.log_evidence,
                    point_frequency_hz=point.trajectory.frequency_hz.tolist(),
                    point_log_gain_per_sec=point.trajectory.log_gain_per_sec.tolist(),
                    component_forecast_sha256=[component.digest() for component in forecasts['parameter_mixture'].components]))
            if 'forecast_names' in plan:
                forecasts = {name: forecasts[name] for name in plan['forecast_names']}
            family_weights = {family: float(np.logaddexp.reduce([weight for row, weight in
                zip(record['candidates'], bank.log_weights) if row['family'] == family]))
                for family in record['family_counts']}
            # Full immutable forecasts remain in memory for scoring. Saved moments and hashes
            # describe them; source plus observed inputs reproduce their joint distributions.
            metadata, arrays, branches, integration_records = {}, {}, [], {}
            for name, forecast in forecasts.items():
                metadata[name] = dict(forecast_sha256=forecast.digest())
                arrays[name+'_mean'] = forecast.mean
                arrays[name+'_variance'] = forecast.diagonal_variance()
                draws = forecast.sample(np.random.default_rng([plan['sampling_seed'], case['seed'], fs]), plan['draws'])
                for label, audio in [('mean', forecast.mean)]+[(f'draw.{i}', row) for i, row in enumerate(draws)]:
                    path = directory/(name+'.'+label+'.f32le')
                    with np.errstate(over='raise', invalid='raise'):
                        values = np.asarray(audio, dtype='<f4')
                    if not np.isfinite(values).all():
                        raise ArithmeticError('nonfinite predicted native trajectory')
                    values.tofile(path)
                    branches.append(dict(id=name+'.'+label, origin='predictive', audio=str(path.resolve())))
                if 'perceptual_sampling' in plan:
                    settings = plan['perceptual_sampling']
                    stratified, sampling = stratified_samples(forecast,
                        np.random.default_rng([settings['seed'], case['seed'], fs]),
                        target_draws=settings['target_draws'], min_per_component=settings['min_per_component'],
                        max_omitted_mass=settings['max_omitted_mass'])
                    integration_records[name] = sampling
                    metadata[name]['stratified_sampling'] = sampling
                    for index, audio in enumerate(stratified):
                        path = directory / f'{name}.stratum.{index}.f32le'
                        with np.errstate(over='raise', invalid='raise'):
                            values = np.asarray(audio, dtype='<f4')
                        if not np.isfinite(values).all():
                            raise ArithmeticError('nonfinite stratified native trajectory')
                        values.tofile(path)
                        branches.append(dict(id=f'{name}.stratum.{index}', origin='predictive', audio=str(path.resolve())))
            np.savez_compressed(directory/'forecast-moments.npz', **arrays)
            write_json(directory/'issued.json', dict(issued_sample=bank.next_sample, target_end_sample=issued+total,
                forecasts=metadata, family_log_weights=family_weights, best_index=best_index,
                component_forecast_sha256=[component.digest() for component in mixture.components]))
            np.asarray(np.r_[past, adaptation, comparison], dtype='<f4').tofile(directory/'observed-past.f32le')
            native_case = dict(id=directory.name, sample_rate=fs, issued_sample=bank.next_sample, bus='habitat',
                past=str((directory/'observed-past.f32le').resolve()),
                output=str((directory/'predicted.jsonl').resolve()), branches=branches)
            write_json(directory/'predicted-plan.json', dict(config=str(args.config.resolve()), cases=[native_case]))
            with (directory/'predicted-native.log').open('x') as output:
                subprocess.run(command, env=dict(os.environ,
                    CONCHORDAL_PERCEPTUAL_ACTION_PLAN=str((directory/'predicted-plan.json').resolve())),
                    stdout=output, stderr=subprocess.STDOUT, check=True)
            seal = hashlib.sha256((directory/'predicted.jsonl').read_bytes()).hexdigest()
            write_json(directory/'prediction-seal.json', dict(sha256=seal))
            # Only this chosen action's remaining future enters scoring, after native predictions.
            target = np.fromfile(actual_path, dtype='<f4', count=horizon, offset=(adapt+compare)*4)
            if len(target) != horizon or not np.isfinite(target).all():
                raise ValueError('incomplete finite chosen target')
            target.tofile(directory/'actual-tail.f32le')
            native_case.update(output=str((directory/'actual.jsonl').resolve()),
                branches=[dict(id='chosen', origin='perceptual', audio=str((directory/'actual-tail.f32le').resolve()))])
            write_json(directory/'actual-plan.json', dict(config=str(args.config.resolve()), cases=[native_case]))
            with (directory/'actual-native.log').open('x') as output:
                subprocess.run(command, env=dict(os.environ,
                    CONCHORDAL_PERCEPTUAL_ACTION_PLAN=str((directory/'actual-plan.json').resolve())),
                    stdout=output, stderr=subprocess.STDOUT, check=True)
            if hashlib.sha256((directory/'predicted.jsonl').read_bytes()).hexdigest() != seal:
                raise ValueError('issued native predictions changed')
            contract, predictions = read_results(directory/'predicted.jsonl')
            actual_contract, actual = read_results(directory/'actual.jsonl')
            observed = actual['chosen']['observation']
            if (contract != actual_contract or observed is None
                    or observed['end_sample']-contract['issued_sample'] < contract['hop_samples']):
                raise ValueError('mismatched observed state or missing complete future hop')
            if any(row['observation'] is None or row['observation']['end_sample'] != observed['end_sample']
                   for row in predictions.values()):
                raise ValueError('predictions and target must share their native time boundary')
            metrics, integration_arrays = {}, {}
            for name, forecast in forecasts.items():
                fields = {}
                for field in FIELDS:
                    observed_field = np.asarray(observed[field])
                    samples = np.array([predictions[f'{name}.draw.{i}']['observation'][field] for i in range(plan['draws'])])
                    lower, upper = np.quantile(samples, [.05, .95], axis=0)
                    fields[field] = dict(mse=float(np.mean((samples.mean(axis=0)-observed_field)**2)),
                        marginal_90_coverage=float(np.mean((observed_field >= lower)&(observed_field <= upper))),
                        interval_width=float(np.mean(upper-lower)))
                metrics[name] = dict(pcm_mse=float(np.mean((forecast.mean-target)**2)),
                    joint_log_density=forecast.log_density(target), fields=fields)
                if name in integration_records:
                    sampling = integration_records[name]
                    integrated = {}
                    for field in FIELDS:
                        values = np.array([predictions[f'{name}.stratum.{i}']['observation'][field]
                                           for i in range(sampling['actual_draws'])])
                        estimate = bounded_summary(values, sampling, alpha=plan['perceptual_sampling']['alpha'])
                        for key in ('retained_mean', 'omission_midpoint', 'monte_carlo_se', 'numerical_lower', 'numerical_upper'):
                            integration_arrays[f'{name}.{field}.{key}'] = estimate[key]
                        integrated[field] = dict(midpoint_mse=float(np.mean((estimate['omission_midpoint']-observed[field])**2)),
                            mean_monte_carlo_se=float(np.mean(estimate['monte_carlo_se'])),
                            max_monte_carlo_se=float(np.max(estimate['monte_carlo_se'])),
                            omitted_mass=estimate['omitted_mass'], hoeffding_radius=estimate['hoeffding_radius'],
                            marginal_alpha=estimate['marginal_alpha'])
                    metrics[name]['stratified_fields'] = integrated
            if integration_arrays:
                np.savez_compressed(directory/'integration-moments.npz', **integration_arrays)
            result = dict(name=directory.name, body=case['body'], route=case['route'], fs=fs,
                external_condition=case['external_condition'], chosen_action=action,
                family_log_weights=family_weights, best_family=record['candidates'][best_index]['family'],
                model_count=len(bank.models), unavailable_count=len(record['unavailable']), metrics=metrics)
            write_json(directory/'result.json', result)
            results.append(result)
            print('scored', directory.name, result['best_family'], flush=True)
    write_json(args.output/'summary.json', results)


if __name__ == '__main__':
    main()
