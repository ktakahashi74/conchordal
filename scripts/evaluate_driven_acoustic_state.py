#!/usr/bin/env python3
"""Causal Gaussian-driven acoustic candidates, not source or cognitive labels.

Burg (1975), chapter II-C, eq. II-64, supplies the reflection estimate:
https://sepwww.stanford.edu/data/media/public/oldreports/sep06/06_09.pdf
Burg forward/backward prediction-error fitting proposes all-pole responses.
Lattice states avoid expanding high-order polynomials during synthesis.
"""

from dataclasses import dataclass
import hashlib

import numpy as np

from evaluate_auditory_envelope import AuditoryEnvelopeObserver


def analyze_lattice(audio, reflection, state=None):
    """Return innovations and backward-error state; matrix columns are independent."""
    reflection = np.asarray(reflection, dtype=float)
    forward = np.array(audio, dtype=float, copy=True)
    if forward.ndim not in (1, 2):
        raise ValueError('expected a vector or a matrix of independent columns')
    old = np.zeros((len(reflection),)+forward.shape[1:]) if state is None else np.array(state, copy=True)
    if old.shape != (len(reflection),)+forward.shape[1:]:
        raise ValueError('lattice state must match reflection order and audio columns')
    if not len(forward):
        return forward, old
    backward = forward.copy()
    new = np.empty_like(old)
    if len(old):
        new[0] = forward[-1]
    for m, coefficient in enumerate(reflection):
        shifted = np.concatenate((old[m:m+1], backward[:-1]), axis=0)
        backward = shifted + coefficient * forward
        forward = forward + coefficient * shifted
        if m + 1 < len(old):
            new[m + 1] = backward[-1]
    return forward, new


def synthesize_lattice(innovations, reflection, state):
    innovations, reflection = np.asarray(innovations, dtype=float), np.asarray(reflection, dtype=float)
    old = np.array(state, dtype=float, copy=True)
    if innovations.ndim not in (1, 2) or old.shape != (len(reflection),)+innovations.shape[1:]:
        raise ValueError('lattice state must match reflection order and innovation columns')
    if not len(reflection):
        return np.array(innovations, dtype=float, copy=True), old
    coefficients = reflection.reshape((len(reflection),)+(1,)*(innovations.ndim-1))
    output = np.empty_like(innovations)
    for n, innovation in enumerate(innovations):
        forward = innovation - np.cumsum((coefficients * old)[::-1], axis=0)[::-1]
        new = np.empty_like(old)
        new[1:] = old[:-1] + coefficients[:-1] * forward[:-1]
        new[0] = forward[0]
        old = new
        output[n] = forward[0]
    return output, old


@dataclass(frozen=True)
class DrivenFit:
    reflection: np.ndarray
    drive_variance: float
    origin_sample: int
    fit_end_sample: int

    def state(self, audio):
        # A backward error of order p-1 depends only on the last p observations.
        return analyze_lattice(audio[-len(self.reflection):] if len(self.reflection) else [],
                               self.reflection)[1]

    def forecast(self, observed, count):
        mean, _ = synthesize_lattice(np.zeros(count), self.reflection, self.state(observed))
        impulse = np.zeros(count)
        if count:
            impulse[0] = 1.
        response, _ = synthesize_lattice(impulse, self.reflection, np.zeros(len(self.reflection)))
        return mean, response

    def log_density(self, past, observed):
        error, _ = analyze_lattice(observed, self.reflection, self.state(past))
        return float(-.5 * (len(error) * np.log(2 * np.pi * self.drive_variance)
                            + (error @ error) / self.drive_variance))


def fit_burg(audio, start_sample, *, orders, variance_floor):
    audio = np.asarray(audio, dtype=float)
    if (audio.ndim != 1 or not np.all(np.isfinite(audio)) or not orders or 0 not in orders
            or any(not isinstance(p, int) or p < 0 for p in orders)
            or len(audio) <= 2 * max(orders) or start_sample < 0
            or not np.isfinite(variance_floor) or variance_floor <= 0):
        raise ValueError("expected finite PCM, admissible orders and positive variance floor")
    candidates = []
    reflection = []
    forward, backward = audio[1:].copy(), audio[:-1].copy()
    power = float(np.mean(audio ** 2))
    for order in range(max(orders) + 1):
        if order in orders:
            coefficients = np.array(reflection)
            coefficients.setflags(write=False)
            candidates.append(DrivenFit(coefficients, max(power, variance_floor),
                                        start_sample, start_sample + len(audio)))
        if order == max(orders):
            break
        denominator = float(forward @ forward + backward @ backward)
        if denominator == 0:
            break
        coefficient = float(-2 * (forward @ backward) / denominator)
        if abs(coefficient) > 1. + 32 * np.finfo(float).eps:
            raise ArithmeticError("reflection coefficient violates the energy bound")
        coefficient = float(np.clip(coefficient, -1., 1.))
        reflection.append(coefficient)
        power *= max(0., 1. - coefficient * coefficient)
        next_forward = forward + coefficient * backward
        next_backward = backward + coefficient * forward
        forward, backward = next_forward[1:], next_backward[:-1]
    return candidates


class DrivenObserver:
    """Retain competing responses and update their observed states causally."""

    def __init__(self, fs, *, sample_quantum, fit_sec=.04, horizon_sec=.01,
                 step_sec=.05, orders=(0, 4, 8, 16, 32, 64), capacity=4):
        if (not isinstance(fs, int) or fs < 16000 or not np.isfinite(sample_quantum)
                or sample_quantum <= 0 or not isinstance(capacity, int) or capacity < 1
                or not all(np.isfinite(t) and t > 0 for t in (fit_sec, horizon_sec, step_sec))):
            raise ValueError("expected sample rate, precision, positive intervals and capacity")
        self.fs = fs
        self.fit_samples, self.horizon, self.step = [round(t * fs) for t in (fit_sec, horizon_sec, step_sec)]
        if (not orders or 0 not in orders or any(not isinstance(p, int) or p < 0 for p in orders)
                or self.fit_samples <= 2 * max(orders) or self.step < self.horizon
                or self.horizon < 1):
            raise ValueError("inadmissible fitting orders or numerical windows")
        # Uniform quantization variance is a numerical floor, not exact PCM-bin likelihood.
        self.variance_floor = sample_quantum ** 2 / 12.
        self.orders = tuple(orders)
        self.capacity = capacity
        self.history_size = self.fit_samples + self.horizon
        self.history = np.empty(0)
        self.bank = []
        self.next_sample = None
        self.next_issue = None

    def process(self, start_sample, audio):
        audio = np.asarray(audio, dtype=float)
        if (not isinstance(start_sample, int) or start_sample < 0 or audio.ndim != 1
                or not np.all(np.isfinite(audio))
                or self.next_sample is not None and start_sample < self.next_sample):
            raise ValueError("expected ordered finite mono PCM")
        if not len(audio):
            return []
        rows = []
        if self.next_sample is None or self.next_sample != start_sample:
            if self.next_sample is not None:
                rows.append({"kind": "input_gap", "missing_start_sample": self.next_sample,
                             "available_sample": start_sample})
            self.history = np.empty(0)
            self.bank = []
            self.next_issue = start_sample + self.history_size
        cursor, end = start_sample, start_sample + len(audio)
        while cursor < end:
            stop = min(end, self.next_issue)
            self.history = np.concatenate((self.history, audio[cursor-start_sample:stop-start_sample]))[-self.history_size:]
            cursor = stop
            if cursor != self.next_issue:
                continue
            evidence_start = cursor - len(self.history)
            train, validation = self.history[:-self.horizon], self.history[-self.horizon:]
            candidates = fit_burg(train, evidence_start, orders=self.orders, variance_floor=self.variance_floor)
            kinds = ['fresh'] * len(candidates)
            for previous in self.bank:
                candidates.append(previous)
                kinds.append('retained')
                errors, _ = analyze_lattice(train, previous.reflection)
                power = float(np.mean(errors[max(self.orders):] ** 2))
                candidates.append(DrivenFit(previous.reflection, max(power, self.variance_floor),
                                            previous.origin_sample, evidence_start + len(train)))
                kinds.append('retained_drive_updated')
            log_densities = np.array([c.log_density(train, validation) for c in candidates])
            order = np.argsort(-log_densities, kind='stable')
            self.bank = []
            seen_responses = set()
            kept = []
            for index in order:
                response_id = hashlib.sha256(candidates[index].reflection.tobytes()).hexdigest()
                if response_id not in seen_responses:
                    seen_responses.add(response_id)
                    self.bank.append(candidates[index])
                    kept.append(int(index))
                if len(self.bank) == self.capacity:
                    break
            best = candidates[order[0]]
            mean, impulse = best.forecast(self.history, self.horizon)
            for values in (mean, impulse):
                values.setflags(write=False)
            rows.append({'kind': 'forecast', 'issued_sample': cursor,
                         'target_end_sample': cursor + self.horizon,
                         'evidence_start_sample': evidence_start,
                         'mean': mean, 'impulse': impulse,
                         'drive_variance': best.drive_variance,
                         'reflection': best.reflection,
                         'response_origin_sample': best.origin_sample,
                         'fit_end_sample': best.fit_end_sample,
                         'selected_kind': kinds[order[0]],
                         'candidate_kinds': kinds,
                         'candidate_orders': [len(c.reflection) for c in candidates],
                         'past_log_density': log_densities.tolist(), 'retained_indices': kept,
                         'mean_sha256': hashlib.sha256(mean.tobytes()).hexdigest(),
                         'impulse_sha256': hashlib.sha256(impulse.tobytes()).hexdigest()})
            self.next_issue += self.step
        self.next_sample = end
        return rows


def auditory_power_moments(observer, mean, impulse, drive_variance):
    """Expected future band power, with fixed observed filter state.

This is E[|Z|^2], not E[|Z|] or a calibrated perceptual likelihood.
All future innovations are independent in this conditional acoustic model.
"""
    if (observer.pending != 0 or len(mean) != len(impulse)
            or len(mean) % observer.stride or drive_variance < 0):
        raise ValueError("expected aligned future blocks and a nonnegative variance")
    # Preserve the current listener state; each forecast starts from the same evidence.
    state = observer.state.copy()
    mean_power = np.empty((len(mean), len(observer.pole)))
    for n, sample in enumerate(mean):
        state *= observer.pole
        state[0] += sample * observer.gain
        state[1] += state[0]
        state[2] += state[1]
        state[3] += state[2]
        mean_power[n] = np.abs(state[3]) ** 2
    # Compose the AR impulse with the auditory filter before taking power.
    impulse_observer = AuditoryEnvelopeObserver(observer.fs, observer.centers_log2, stride_samples=1)
    rows = impulse_observer.process(0, impulse)
    combined_power = np.array([r['envelope_scan'] for r in rows]) ** 2
    variance = drive_variance * np.cumsum(combined_power, axis=0)
    shape = (-1, observer.stride, len(observer.pole))
    return (mean_power.reshape(shape).mean(axis=1),
            (mean_power + variance).reshape(shape).mean(axis=1))
