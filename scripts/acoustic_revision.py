"""Retained and freshly proposed conditional acoustic states.

All alternatives share a chosen adaptation interval and stop before common new
comparison evidence. Acoustic proposal masses are not cognitive forgetting or
musical values. Fresh parameters may use a suffix; every state conditions on the
full adaptation so reset/support comparisons do not change that interval.
"""
import copy

import numpy as np

from acoustic_body import DrivenBodyPosterior
from acoustic_mixture import AcousticModelBank
from acoustic_trajectory import fit_trajectory_candidates


def revise_body_bank(original, rows, audio, fs, settings, *, counts, offsets_sec, prior_mass):
    """Prepare state/scale/weight controls and a retained-plus-fresh mixture."""
    audio = np.asarray(audio, dtype=float)
    start = original.next_sample
    if (audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all()
            or len(rows) != len(original.models) or any(not isinstance(m, DrivenBodyPosterior) for m in original.models)
            or set(prior_mass) != {'retained', 'fresh'}
            or any(not np.isfinite(v) or v <= 0 for v in prior_mass.values())):
        raise ValueError('expected contemporaneous modal candidates, finite adaptation and positive group masses')
    intensities = settings['drive_per_sec']
    if (not intensities or 0. not in intensities or len(set(intensities)) != len(intensities)
            or any(not np.isfinite(d) or d < 0 for d in intensities)):
        raise ValueError('distinct nonnegative drive intensities must include the passive control')
    offsets = [round(value*fs) for value in offsets_sec]
    if not offsets or len(set(offsets)) != len(offsets) or any(not 0 <= v < len(audio) for v in offsets):
        raise ValueError('distinct proposal offsets must lie inside the chosen adaptation')
    if (not counts or 0 not in counts or len(set(counts)) != len(counts)
            or any(not isinstance(n, int) or n < 0 for n in counts)):
        raise ValueError('distinct nonnegative component counts must include zero')

    retained = copy.deepcopy(original)
    retained.observe(start, audio)
    banks = dict(retained=retained)
    for name in ('state_reset', 'state_scale_reset'):
        models = []
        for old, row in zip(original.models, rows):
            variance_prior = ((old.variance_shape, old.variance_scale) if name == 'state_reset'
                              else settings['variance_prior'])
            model = DrivenBodyPosterior(fs, row['frequency_hz'], row['log_gain_per_sec'],
                drive_per_sec=row['drive_per_sec'], initial_precision=settings['initial_precision'],
                variance_prior=variance_prior, start_sample=start, parameters_available_sample=start)
            if (not np.array_equal(model.transition, old.transition)
                    or not np.array_equal(model.drive, old.drive)):
                raise ValueError('reset controls must preserve the original modal dynamics')
            models.append(model)
        bank = AcousticModelBank(models, np.ones(len(models)))
        bank.log_weights = original.log_weights.copy()
        bank.observe(start, audio)
        banks[name] = bank
    banks['uniform_reset'] = copy.deepcopy(banks['state_scale_reset'])
    banks['uniform_reset'].log_weights = np.full(len(rows), -np.log(len(rows)))

    fresh_models, fresh_rows, seen = [], [], set()
    for offset in offsets:
        local = audio[offset:]
        fits = fit_trajectory_candidates(local, start+offset, fs,
            counts=tuple(n for n in counts if 8*n < len(local)), glide=False, passive=True)
        for fit in fits:
            key = (tuple(fit.frequency_hz), tuple(fit.log_gain_per_sec))
            if key in seen:
                continue
            seen.add(key)
            for intensity in (intensities if len(fit.frequency_hz) else (0.,)):
                model = DrivenBodyPosterior(fs, fit.frequency_hz, fit.log_gain_per_sec,
                    drive_per_sec=intensity, initial_precision=settings['initial_precision'],
                    variance_prior=settings['variance_prior'], start_sample=start,
                    parameters_available_sample=start+len(audio))
                model.observe(start, audio)
                fresh_models.append(model)
                fresh_rows.append(dict(fit_start_sample=start+offset, fit_end_sample=start+len(audio),
                    condition_start_sample=start, frequency_hz=fit.frequency_hz.tolist(),
                    log_gain_per_sec=fit.log_gain_per_sec.tolist(), drive_per_sec=intensity))
    fresh = AcousticModelBank(fresh_models, np.ones(len(fresh_models)))
    banks['fresh'] = fresh
    mixture_models = copy.deepcopy(retained.models+fresh.models)
    mixture = AcousticModelBank(mixture_models, np.ones(len(mixture_models)))
    group_log_mass = np.log([prior_mass['retained'], prior_mass['fresh']])
    group_log_mass -= np.logaddexp.reduce(group_log_mass)
    mixture.log_weights = np.r_[retained.log_weights+group_log_mass[0], fresh.log_weights+group_log_mass[1]]
    banks['revision'] = mixture
    return banks, dict(fresh_candidates=fresh_rows, retained_count=len(rows),
        revision_prior_mass=prior_mass, adaptation_start_sample=start, comparison_start_sample=start+len(audio))
