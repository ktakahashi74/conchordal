"""Competing local acoustic supports and amplitude evolutions.

Nonstationary partials distinguish amplitude and phase evolution (Marchand,
DAFx 2012, section 2). Affine complex amplitudes here are a local approximation,
not that paper's estimator or a cognitive model of excitation/source identity.
"""
from dataclasses import dataclass, replace

import numpy as np

from acoustic_posterior import AcousticPosterior, select_acoustic_posterior
from acoustic_trajectory import TrajectoryFit, fit_trajectory_candidates
from evaluate_driven_acoustic_state import fit_burg


@dataclass(frozen=True)
class AmplitudeBasis:
    retained: TrajectoryFit
    added: TrajectoryFit | None
    origin_sample: float
    scale_samples: int
    degree: int

    @property
    def fit_end_sample(self):
        return max(self.retained.fit_end_sample,
                   self.added.fit_end_sample if self.added is not None else 0)

    def basis(self, start_sample, count):
        if self.degree not in (0,1) or self.scale_samples<1:
            raise ValueError("expected constant or affine amplitudes with a positive scale")
        # These hypotheses estimate amplitude anew; importing a short-fit
        # exponential gain would silently add another envelope evolution.
        base=np.column_stack([replace(carrier,log_gain_per_sec=np.zeros_like(carrier.log_gain_per_sec))
                              .basis(start_sample,count) for carrier in [self.retained,self.added]
                              if carrier is not None])
        if self.degree:
            time=(np.arange(start_sample,start_sample+count)-self.origin_sample)/self.scale_samples
            base=np.column_stack((base,time[:,None]*base))
        return base


def propose_hypotheses(past,observed,fs,issued,plan,*,learn_variance=False):
    """Propose conditional models before seeing their common comparison block.

    The second input is adaptation evidence only. Source labels, true change
    times and the subsequent comparison/forecast targets are not accepted.
    """
    past,observed=np.asarray(past,dtype=float),np.asarray(observed,dtype=float)
    fit=round(plan['base_fit_sec']*fs);validation=round(plan['base_validation_sec']*fs)
    adaptation=round(plan['adaptation_sec']*fs);selection=round(plan['selection_sec']*fs)
    horizon=round(plan['horizon_sec']*fs)
    if (past.ndim!=1 or observed.ndim!=1 or len(past)<fit+validation
            or len(observed)!=adaptation or issued<len(past)
            or min(fit,validation,adaptation,selection,horizon)<1
            or not np.isfinite(past).all() or not np.isfinite(observed).all()):
        raise ValueError("expected finite prior evidence and exactly the observed update block")
    history=past[-fit-validation:];start=issued-len(history)
    stationary=fit_trajectory_candidates(history[:fit],start,fs,counts=tuple(plan['counts']),glide=False)
    gliding=fit_trajectory_candidates(history[:fit],start,fs,counts=tuple(plan['counts']),initial=stationary)
    base=select_acoustic_posterior([('stationary',m) for m in stationary]+[('glide',m) for m in gliding],
        history[:fit],history[fit:],start,noise_orders=tuple(plan['orders']),
        variance_floor=plan['variance_floor'],forecast_samples=adaptation+selection+horizon)['posterior']
    if learn_variance:
        base=AcousticPosterior(base.trajectory,base.residual,history,start,learn_variance=True)
    support_scores=[]
    for trajectory in stationary+gliding:
        try:
            error=trajectory.predict(start+fit,validation)-history[fit:]
            score=float(np.mean(error**2))
        except ArithmeticError:
            score=float('inf')
        support_scores.append(score)
    # A residual AR model may carry a periodic signal with no explicit partials.
    # Propose explicit support using its own already-observed waveform continuation.
    retained=(stationary+gliding)[int(np.argmin(support_scores))]
    selection_start=issued+adaptation
    target_start=selection_start+selection
    candidates=[]
    base.observe(issued,observed)
    pending=[('continued_state',base,dict(adaptation_start=issued,degree=None,added_components=0))]
    for offset_sec in plan['adaptation_offsets_sec']:
        offset=round(offset_sec*fs)
        if not 0<=offset<adaptation:
            raise ValueError("adaptation offsets must precede the selection block")
        audio=observed[offset:adaptation];begin=issued+offset
        if 'refitted_passive_counts' in plan:
            counts=tuple(n for n in plan['refitted_passive_counts'] if 8*n<len(audio))
            fresh=fit_trajectory_candidates(audio,begin,fs,counts=counts,glide=False,passive=True)
            for trajectory in fresh:
                if not len(trajectory.frequency_hz):
                    continue
                residual=audio-trajectory.predict(begin,len(audio))
                orders=tuple(p for p in plan['orders'] if 2*p<len(audio))
                for noise in fit_burg(residual,begin,orders=orders,variance_floor=plan['variance_floor']):
                    meta=dict(adaptation_start=begin,degree=None,added_components=None,
                              frequency_hz=trajectory.frequency_hz.tolist(),
                              log_gain_per_sec=trajectory.log_gain_per_sec.tolist(),
                              components=len(trajectory.frequency_hz),order=len(noise.reflection))
                    try:
                        posterior=AcousticPosterior(trajectory,noise,audio,begin,learn_variance=learn_variance)
                        pending.append(('refitted_passive',posterior,meta))
                    except (ArithmeticError,np.linalg.LinAlgError) as error:
                        candidates.append(dict(family='refitted_passive',**meta,unavailable=str(error)))
        for degree in plan['amplitude_degrees']:
            local=AmplitudeBasis(retained,None,begin+(len(audio)-1)/2,len(audio),degree)
            raw=local.basis(begin,len(audio));matrix=np.column_stack((raw.real,raw.imag))
            residual=audio-matrix@np.linalg.lstsq(matrix,audio,rcond=1e-10)[0]
            additions=fit_trajectory_candidates(residual,begin,fs,counts=tuple(plan['added_counts']),glide=False)
            for added in [None]+[m for m in additions if len(m.frequency_hz)]:
                family='changed_amplitude' if added is None else 'added_components'
                if added is None and not len(retained.frequency_hz):
                    continue
                basis=AmplitudeBasis(retained,added,local.origin_sample,local.scale_samples,degree)
                raw=basis.basis(begin,len(audio));matrix=np.column_stack((raw.real,raw.imag))
                residual=audio-matrix@np.linalg.lstsq(matrix,audio,rcond=1e-10)[0]
                orders=tuple(p for p in plan['orders'] if 2*p<len(audio))
                for noise in fit_burg(residual,begin,orders=orders,variance_floor=plan['variance_floor']):
                    meta=dict(adaptation_start=begin,degree=degree,added_components=0 if added is None else len(added.frequency_hz),
                              added_frequency_hz=[] if added is None else added.frequency_hz.tolist(),order=len(noise.reflection))
                    try:
                        posterior=AcousticPosterior(basis,noise,audio,begin,learn_variance=learn_variance)
                        pending.append((family,posterior,meta))
                    except (ArithmeticError,np.linalg.LinAlgError) as error:
                        candidates.append(dict(family=family,**meta,unavailable=str(error)))
    return pending,candidates,dict(issued_sample=target_start,target_end_sample=target_start+horizon,
        base_fit_end=retained.fit_end_sample,selection_start=selection_start,selection_end=target_start,
        retained_frequency_hz=retained.frequency_hz.tolist(),retained_rate_hz_per_sec=retained.rate_hz_per_sec.tolist(),
        retained_log_gain_per_sec=retained.log_gain_per_sec.tolist(),
        support_validation_mse=[s if np.isfinite(s) else None for s in support_scores],
        support_index=int(np.argmin(support_scores)))


def compare_hypotheses(past,observed,fs,issued,plan):
    """Legacy fixed-variance family winners, using only already observed audio."""
    observed=np.asarray(observed,dtype=float)
    adaptation=round(plan['adaptation_sec']*fs);selection=round(plan['selection_sec']*fs)
    horizon=round(plan['horizon_sec']*fs)
    if observed.ndim!=1 or len(observed)!=adaptation+selection or not np.isfinite(observed).all():
        raise ValueError("expected exactly the observed adaptation and selection blocks")
    pending,candidates,record=propose_hypotheses(past,observed[:adaptation],fs,issued,plan)
    selected_audio=observed[adaptation:]
    selection_start=record['selection_start']
    family_best={};family_indices={};family_scores={}
    for family,posterior,meta in pending:
        row=dict(family=family,**meta)
        try:
            validation_forecast=posterior.forecast(selection)
            score=validation_forecast.log_density(selected_audio)
            if not np.isfinite(score):
                raise ArithmeticError("nonfinite selection density")
            row.update(selection_log_density=score,selection_mse=float(np.mean((validation_forecast.mean-selected_audio)**2)),
                       selection_forecast_sha256=validation_forecast.digest(),innovation_variance=posterior.residual.drive_variance)
            posterior.observe(selection_start,selected_audio)
            forecast=posterior.forecast(horizon)
            if not np.isfinite(score) or not np.isfinite(forecast.mean).all() or not np.isfinite(forecast.diagonal_variance()).all():
                raise ArithmeticError("nonfinite conditional forecast")
            row['forecast_sha256']=forecast.digest()
            if score>family_scores.get(family,float('-inf')):
                family_best[family]=forecast;family_scores[family]=score;family_indices[family]=len(candidates)
        except (ArithmeticError,np.linalg.LinAlgError) as error:
            row['unavailable']=str(error)
        candidates.append(row)
    if not family_best:
        raise ArithmeticError("no supported conditional forecast")
    winner=max(family_best,key=family_scores.get)
    return family_best,dict(record,candidates=candidates,family_best_indices=family_indices,selected_family=winner)
