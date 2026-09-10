#!/usr/bin/env python3
"""Owned-action acoustic forecasts, with realized branches read only after issue."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from acoustic_posterior import AcousticForecast, posterior_auditory_power, select_acoustic_posterior
from acoustic_trajectory import fit_trajectory_candidates
from evaluate_auditory_envelope import AuditoryEnvelopeObserver
from evaluate_driven_acoustic_state import auditory_power_moments


def fit_external_posterior(past_mix,past_own,fs,issued,plan,forecast_samples):
    """Fit the external state using only observations and known owned sound."""
    past_mix,past_own=np.asarray(past_mix,dtype=float),np.asarray(past_own,dtype=float)
    if (past_mix.ndim!=1 or past_mix.shape!=past_own.shape
            or issued<len(past_mix) or not np.isfinite(past_mix).all() or not np.isfinite(past_own).all()):
        raise ValueError("expected matching finite histories")
    fit_count=round(plan['fit_sec']*fs)
    validation_count=round(plan['validation_sec']*fs)
    count=fit_count+validation_count
    if len(past_mix)<count:
        raise ValueError("insufficient observed evidence")
    # Subtract waveforms before nonlinear energy analysis; powers do not add coherently.
    external=(past_mix-past_own)[-count:]
    start=issued-count
    stationary=fit_trajectory_candidates(external[:fit_count],start,fs,
                                        counts=tuple(plan['trajectory_counts']),glide=False)
    glide=fit_trajectory_candidates(external[:fit_count],start,fs,
                                   counts=tuple(plan['trajectory_counts']),initial=stationary)
    candidates=[('stationary',m) for m in stationary]+[('glide',m) for m in glide]
    selected=select_acoustic_posterior(candidates,external[:fit_count],external[fit_count:],start,
                                     noise_orders=tuple(plan['residual_orders']),
                                     variance_floor=plan['variance_floor'],forecast_samples=forecast_samples)
    posterior=selected['posterior']
    return posterior,dict(
                         selected=selected['candidates'][selected['posterior_index']],
                         candidates=selected['candidates'],external_evidence_start=start,
                         trajectory_fit_end=posterior.trajectory.fit_end_sample,
                         external_evidence_sha256=hashlib.sha256(external.tobytes()).hexdigest())


def forecast_branches(past_mix,past_own,own_wait,own_sound,fs,issued,plan):
    """Only prior observations and owned forward simulations enter inference."""
    own_wait,own_sound=np.asarray(own_wait,dtype=float),np.asarray(own_sound,dtype=float)
    if own_wait.ndim!=1 or own_wait.shape!=own_sound.shape or not len(own_wait):
        raise ValueError("expected complete owned forecasts")
    posterior,selection=fit_external_posterior(past_mix,past_own,fs,issued,plan,len(own_wait))
    forecast=posterior.forecast(len(own_wait))
    return forecast,{name:forecast.with_known_waveform(signal) for name,signal in
                     [('wait',own_wait),('sound',own_sound)]},selection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    args.output.mkdir()
    cases=json.loads((args.input/'inputs.json').read_text())
    summary=[]
    for case in cases:
        directory=args.input/case['name']
        evidence={key:np.fromfile(directory/(key+'.f32le'),dtype='<f4').astype(float)
                  for key in ('past_mix','past_own','own_wait','own_sound')}
        fs=case['fs'];issued=case['issue_sample']
        external,forecasts,selection=forecast_branches(**evidence,fs=fs,issued=issued,plan=plan)
        observer=AuditoryEnvelopeObserver(fs,np.linspace(np.log2(55),np.log2(7040),113),stride_samples=fs//1000)
        observer.process(issued-len(evidence['past_mix']),evidence['past_mix'])
        horizon=len(external.mean)
        arrays=dict(external_mean=external.mean,impulse=external.impulse,amplitude_factor=external.amplitude_factor,
                    reflection=external.reflection,variance=external.diagonal_variance())
        external_power=posterior_auditory_power(observer,external)[2]
        zeros=np.zeros(horizon)
        empty_observer=AuditoryEnvelopeObserver(fs,observer.centers_log2,stride_samples=observer.stride)
        for branch,forecast in forecasts.items():
            arrays[branch+'_mean']=forecast.mean
            arrays[branch+'_power']=posterior_auditory_power(observer,forecast)[2]
            # Ablation: retain both power contributions but deliberately omit their cross term.
            own_power=auditory_power_moments(empty_observer,evidence['own_'+branch],zeros,0.)[0]
            arrays[branch+'_power_without_cross']=external_power+own_power
        metadata=dict(name=case['name'],fs=fs,issued_sample=issued,target_end_sample=issued+horizon,
                      selection=selection,innovation_variance=external.innovation_variance,
                      forecast_sha256={key:value.digest() for key,value in forecasts.items()},
                      evidence_sha256={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in evidence.items()},
                      body=case['body'],route=case['route'],external_condition=case['external_condition'])
        # Seal issued predictions before opening either realized future.
        np.savez_compressed(args.output/(case['name']+'-forecast.npz'),**arrays)
        (args.output/(case['name']+'-forecast.json')).write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
        update=None
        if 'reobserve_sec' in plan:
            observed_count=round(plan['reobserve_sec']*fs)
            chosen=plan['chosen_branch']
            if chosen not in forecasts or not 0<observed_count<horizon or observed_count%observer.stride:
                raise ValueError("expected one chosen branch and an interior observation boundary")
            # Read only the executed prefix. The unchosen result and remaining future
            # are still withheld; the owned continuation is already known at issue.
            observed=np.fromfile(directory/('truth_'+chosen+'.f32le'),dtype='<f4',count=observed_count).astype(float)
            if len(observed)!=observed_count:
                raise ValueError("chosen outcome prefix is incomplete")
            owned=evidence['own_'+chosen]
            updated_external,updated_branches,updated_selection=forecast_branches(
                np.r_[evidence['past_mix'],observed],np.r_[evidence['past_own'],owned[:observed_count]],
                owned[observed_count:],owned[observed_count:],fs,issued+observed_count,plan)
            updated=updated_branches['wait']
            updated_observer=AuditoryEnvelopeObserver(fs,observer.centers_log2,stride_samples=observer.stride)
            updated_observer.process(issued-len(evidence['past_mix']),evidence['past_mix'])
            updated_observer.process(issued,observed)
            updated_power=posterior_auditory_power(updated_observer,updated)[2]
            initial=forecasts[chosen]
            prefix_forecast=AcousticForecast(issued,initial.mean[:observed_count],initial.impulse[:observed_count],
                initial.amplitude_factor[:observed_count],initial.reflection,initial.innovation_variance)
            update=dict(chosen_branch=chosen,observed_end_sample=issued+observed_count,
                        target_end_sample=issued+horizon,selection=updated_selection,
                        forecast_sha256=updated.digest(),innovation_variance=updated.innovation_variance,
                        chosen_prefix_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),
                        initial_prefix_log_density=prefix_forecast.log_density(observed))
            np.savez_compressed(args.output/(case['name']+'-updated.npz'),mean=updated.mean,
                                impulse=updated.impulse,amplitude_factor=updated.amplitude_factor,
                                reflection=updated.reflection,variance=updated.diagonal_variance(),power=updated_power)
            (args.output/(case['name']+'-updated.json')).write_text(json.dumps(update,indent=2,allow_nan=False)+'\n')
        truths={key:np.fromfile(directory/('truth_'+key+'.f32le'),dtype='<f4').astype(float) for key in forecasts}
        realized_power={key:auditory_power_moments(observer,value,zeros,0.)[0] for key,value in truths.items()}
        metrics=[]
        for branch,forecast in forecasts.items():
            metrics.append(dict(branch=branch,
                pcm_mse=float(np.mean((forecast.mean-truths[branch])**2)),
                power_mse=float(np.mean((arrays[branch+'_power']-realized_power[branch])**2)),
                power_mse_without_cross=float(np.mean((arrays[branch+'_power_without_cross']-realized_power[branch])**2)),
                joint_log_density=forecast.log_density(truths[branch]),
                marginal_95_coverage=float(np.mean(np.abs(forecast.mean-truths[branch])
                    <=1.959963984540054*np.sqrt(arrays['variance'])))))
        delta_pred=forecasts['sound'].mean-forecasts['wait'].mean
        delta_actual=truths['sound']-truths['wait']
        if update is not None:
            target=truths[chosen][observed_count:]
            remaining=realized_power[chosen][observed_count//observer.stride:]
            update['metrics']=dict(
                pcm_mse=float(np.mean((updated.mean-target)**2)),
                frozen_pcm_mse=float(np.mean((forecasts[chosen].mean[observed_count:]-target)**2)),
                power_mse=float(np.mean((updated_power-remaining)**2)),
                frozen_power_mse=float(np.mean((arrays[chosen+'_power'][observed_count//observer.stride:]-remaining)**2)),
                joint_log_density=updated.log_density(target),
                marginal_95_coverage=float(np.mean(np.abs(updated.mean-target)
                    <=1.959963984540054*np.sqrt(updated.diagonal_variance()))))
        summary.append(dict(name=case['name'],body=case['body'],route=case['route'],external_condition=case['external_condition'],
                            metrics=metrics,action_difference_max_error=float(np.max(np.abs(delta_pred-delta_actual))),
                            own_difference_rms=float(np.sqrt(np.mean(delta_pred**2))),
                            selected=selection['selected'],forecast_sha256=metadata['forecast_sha256'],update=update))
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        print(case['name'],summary[-1]['action_difference_max_error'],flush=True)


if __name__=='__main__':
    main()
