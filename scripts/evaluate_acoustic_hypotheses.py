#!/usr/bin/env python3
"""Observe the chosen action prefix, compare acoustic hypotheses, then score futures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from acoustic_hypotheses import compare_hypotheses
from acoustic_posterior import posterior_auditory_power
from evaluate_action_conditioned import forecast_branches
from evaluate_auditory_envelope import AuditoryEnvelopeObserver
from evaluate_driven_acoustic_state import auditory_power_moments


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--name',action='append')
    args=parser.parse_args();plan=json.loads(args.plan.read_text());args.output.mkdir()
    cases=json.loads((args.input/'inputs.json').read_text())
    if args.name:
        cases=[c for c in cases if c['name'] in args.name]
        if {c['name'] for c in cases}!=set(args.name):raise ValueError("unknown requested case")
    results=[]
    for case in cases:
        fs=case['fs'];issue=case['issue_sample'];directory=args.input/case['name']
        count=round((plan['adaptation_sec']+plan['selection_sec'])*fs)
        horizon=round(plan['horizon_sec']*fs)
        past=np.fromfile(directory/'past_mix.f32le',dtype='<f4').astype(float)
        own=np.fromfile(directory/'past_own.f32le',dtype='<f4').astype(float)
        owned_future=np.fromfile(directory/'own_sound.f32le',dtype='<f4').astype(float)
        observed=np.fromfile(directory/'truth_sound.f32le',dtype='<f4',count=count).astype(float)
        if len(observed)!=count or len(owned_future)<count+horizon:
            raise ValueError("incomplete observation or owned forecast")
        owned_future=owned_future[:count+horizon]
        bank,meta=compare_hypotheses(past-own,observed-owned_future[:count],fs,issue,plan)
        forecasts={name:f.with_known_waveform(owned_future[count:]) for name,f in bank.items()}
        old_plan=dict(fit_sec=plan['base_fit_sec'],validation_sec=plan['base_validation_sec'],
            trajectory_counts=plan['counts'],residual_orders=plan['orders'],variance_floor=plan['variance_floor'])
        _,baseline,_=forecast_branches(np.r_[past,observed],np.r_[own,owned_future[:count]],
            owned_future[count:],owned_future[count:],fs,issue+count,old_plan)
        forecasts['refit']=baseline['wait']
        observer=AuditoryEnvelopeObserver(fs,np.linspace(np.log2(55),np.log2(7040),113),stride_samples=fs//1000)
        observer.process(issue-len(past),np.r_[past,observed])
        arrays={};params={}
        for name,forecast in forecasts.items():
            for key in ['mean','impulse','amplitude_factor','reflection']:
                arrays[name+'_'+key]=getattr(forecast,key)
            arrays[name+'_variance']=forecast.diagonal_variance()
            arrays[name+'_power']=posterior_auditory_power(observer,forecast)[2]
            params[name]=dict(innovation_variance=forecast.innovation_variance,forecast_sha256=forecast.digest())
        evidence={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in
                  dict(past_mix=past,past_own=own,owned_future=owned_future,observed_prefix=observed).items()}
        record=dict(case=case,inference=meta,forecasts=params,evidence_sha256=evidence)
        np.savez_compressed(args.output/(case['name']+'.npz'),**arrays)
        (args.output/(case['name']+'.json')).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        # Only after every forecast is saved may the actual target enter scoring.
        target=np.fromfile(directory/'truth_sound.f32le',dtype='<f4',count=horizon,offset=count*4).astype(float)
        if len(target)!=horizon:raise ValueError("incomplete future target")
        actual_power=auditory_power_moments(observer,target,np.zeros(horizon),0.)[0]
        metrics={}
        for name,forecast in forecasts.items():
            metrics[name]=dict(pcm_mse=float(np.mean((forecast.mean-target)**2)),
                power_mse=float(np.mean((arrays[name+'_power']-actual_power)**2)),
                joint_log_density=forecast.log_density(target),
                marginal_95_coverage=float(np.mean(np.abs(forecast.mean-target)
                    <=1.959963984540054*np.sqrt(forecast.diagonal_variance()))))
        selected=meta['selected_family']
        results.append(dict(name=case['name'],external_condition=case['external_condition'],
                            selected_family=selected,metrics=metrics))
        (args.output/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
        print(case['name'],selected,metrics[selected]['pcm_mse']/metrics['refit']['pcm_mse'],flush=True)


if __name__=='__main__':main()
