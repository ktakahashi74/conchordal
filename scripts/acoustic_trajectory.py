#!/usr/bin/env python3
"""Causal acoustic chirp candidates, not identified perceptual sources.

The local signal family follows Neri, Depalle and Badeau (2021), eq. 1:
https://www.dafx.de/paper-archive/2021/proceedings/papers/DAFx20in21_paper_30.pdf
The estimator here is time-domain least squares, not their Bayesian algorithm.
"""

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class TrajectoryFit:
    fs: int
    origin_sample: float
    frequency_hz: np.ndarray
    rate_hz_per_sec: np.ndarray
    log_gain_per_sec: np.ndarray
    amplitude: np.ndarray
    fit_start_sample: int
    fit_end_sample: int
    iterations: int = 0

    def basis(self, start_sample, count):
        if count < 0 or start_sample < 0:
            raise ValueError("expected nonnegative sample coordinates")
        t = (np.arange(start_sample, start_sample + count) - self.origin_sample) / self.fs
        if not len(t) or not len(self.frequency_hz):
            return np.zeros((count, len(self.frequency_hz)), dtype=complex)
        frequency = self.frequency_hz + t[[0, -1], None] * self.rate_hz_per_sec
        if np.any(frequency <= 0) or np.any(frequency >= self.fs / 2):
            raise ArithmeticError("trajectory leaves the represented frequency band")
        phase = 2 * np.pi * (t[:, None] * self.frequency_hz
                             + .5 * t[:, None] ** 2 * self.rate_hz_per_sec)
        with np.errstate(over='raise', invalid='raise'):
            try:
                return np.exp(t[:, None] * self.log_gain_per_sec + 1j * phase)
            except FloatingPointError as error:
                raise ArithmeticError("trajectory extrapolation is not finite") from error

    def predict(self, start_sample, count):
        return (self.basis(start_sample, count) @ self.amplitude).real


def _basis(u, parameters):
    phase = u[:, None] * parameters[:, 0] + u[:, None] ** 2 * parameters[:, 1]
    envelope = np.exp(u[:, None] * parameters[:, 2])
    cosine, sine = envelope * np.cos(phase), envelope * np.sin(phase)
    return np.column_stack((cosine, sine))


def _refine(audio, u, parameters, *, glide, max_iterations, nyquist, passive=False):
    """Joint Gauss-Newton directions with linear coefficients re-solved exactly."""
    parameters = parameters.copy()
    if passive:
        parameters[:, 2] = np.minimum(parameters[:, 2], 0.)
    columns = [0, 1, 2] if glide else [0, 2]
    iterations = 0
    for iteration in range(max_iterations):
        basis = _basis(u, parameters)
        coefficients = np.linalg.lstsq(basis, audio, rcond=1e-10)[0]
        residual = audio - basis @ coefficients
        loss = float(residual @ residual)
        count = len(parameters)
        cosine, sine = basis[:, :count], basis[:, count:]
        phase_derivative = (-sine * coefficients[:count] + cosine * coefficients[count:])
        component = cosine * coefficients[:count] + sine * coefficients[count:]
        derivatives = [phase_derivative * u[:, None],
                       phase_derivative * u[:, None] ** 2, component * u[:, None]]
        jacobian = np.column_stack([basis] + [derivatives[c] for c in columns])
        scales = np.linalg.norm(jacobian, axis=0)
        scales[scales == 0] = 1.
        step = np.linalg.lstsq(jacobian / scales, residual, rcond=1e-10)[0] / scales
        delta = np.zeros_like(parameters)
        delta[:, columns] = step[2 * count:].reshape(len(columns), count).T
        accepted = False
        for exponent in range(16):
            proposal = parameters + delta * 2. ** -exponent
            if passive:
                # Constrain each trial before solving its amplitudes and assessing loss.
                proposal[:, 2] = np.minimum(proposal[:, 2], 0.)
            # Numerical search domain, not a physiological or source prior.
            frequencies = proposal[:, 0] + 2 * u[[0, -1], None] * proposal[:, 1]
            if (np.any(frequencies <= 0) or np.any(frequencies >= nyquist)
                    or np.max(np.abs(proposal[:, 2]), initial=0.) > 64):
                continue
            proposed_basis = _basis(u, proposal)
            proposed_coefficients = np.linalg.lstsq(proposed_basis, audio, rcond=1e-10)[0]
            proposed_residual = audio - proposed_basis @ proposed_coefficients
            proposed_loss = float(proposed_residual @ proposed_residual)
            if proposed_loss < loss:
                parameters = proposal
                accepted = True
                break
        iterations = iteration + 1
        if not accepted or loss - proposed_loss <= 1e-11 * max(loss, np.finfo(float).tiny):
            break
    coefficients = np.linalg.lstsq(_basis(u, parameters), audio, rcond=1e-10)[0]
    return parameters, coefficients, iterations


def fit_trajectory_candidates(audio, start_sample, fs, *, counts=(0, 1, 2, 4, 8),
                              glide=True, max_iterations=30, initial=None, passive=False):
    """Propose component counts from residual PCM; no true frequencies enter."""
    audio = np.asarray(audio, dtype=float)
    if (audio.ndim != 1 or not np.all(np.isfinite(audio)) or start_sample < 0
            or not isinstance(fs, int) or fs < 16000 or not counts or 0 not in counts
            or any(not isinstance(n, int) or n < 0 for n in counts)
            or len(audio) <= max(8, 8 * max(counts)) or max_iterations < 1):
        raise ValueError("expected finite PCM and admissible numerical search limits")
    count = len(audio)
    duration = count / fs
    origin = start_sample + (count - 1) / 2
    u = (np.arange(count) - (count - 1) / 2) / count
    scale = np.max(np.abs(audio))
    normalized = audio / scale if scale else audio
    parameters = np.empty((0, 3))
    coefficients = np.empty(0)
    candidates = []
    nfft = 4 * 2 ** int(np.ceil(np.log2(count)))
    iterations = 0
    initial_by_count = {len(m.frequency_hz): m for m in initial or []}
    for components in range(max(counts) + 1):
        if components in counts:
            chosen_parameters, chosen_coefficients, chosen_iterations = parameters, coefficients, iterations
            if glide and components and components in initial_by_count:
                previous = initial_by_count[components]
                if previous.fs != fs or previous.fit_start_sample != start_sample or previous.fit_end_sample != start_sample+count:
                    raise ValueError("initial candidate must describe the same fitting evidence")
                elapsed = (origin-previous.origin_sample)/fs
                seed = np.column_stack((2*np.pi*duration*(previous.frequency_hz+elapsed*previous.rate_hz_per_sec),
                                        np.pi*duration**2*previous.rate_hz_per_sec,
                                        duration*previous.log_gain_per_sec))
                seeded, seeded_coefficients, seeded_iterations = _refine(
                    normalized, u, seed, glide=True, max_iterations=max_iterations, nyquist=np.pi*count,
                    passive=passive)
                seeded_error = normalized-_basis(u, seeded)@seeded_coefficients
                error = normalized-_basis(u, parameters)@coefficients
                if seeded_error@seeded_error <= error@error:
                    chosen_parameters, chosen_coefficients, chosen_iterations = seeded, seeded_coefficients, seeded_iterations
            values = [chosen_parameters[:, 0] / (2 * np.pi * duration),
                      chosen_parameters[:, 1] / (np.pi * duration ** 2),
                      chosen_parameters[:, 2] / duration,
                      scale * (chosen_coefficients[:components] - 1j * chosen_coefficients[components:])]
            for array in values:
                array.setflags(write=False)
            candidates.append(TrajectoryFit(fs, origin, *values, start_sample,
                                            start_sample + count, chosen_iterations))
        if components == max(counts) or scale == 0:
            break
        residual = normalized - _basis(u, parameters) @ coefficients
        spectrum = np.abs(np.fft.rfft(residual * np.hanning(count), n=nfft))
        # Interior positive frequencies; DC and Nyquist need different real bases.
        index = int(np.argmax(spectrum[1:-1])) + 1
        initial = [2 * np.pi * index * count / nfft, 0., 0.]
        parameters = np.vstack((parameters, initial))
        parameters, coefficients, iterations = _refine(
            normalized, u, parameters, glide=glide, max_iterations=max_iterations,
            nyquist=np.pi * count, passive=passive)
    return candidates


def update_trajectory_state(audio, start_sample, previous):
    """Keep frequency and gain trajectories; estimate a new complex state."""
    duration = len(audio) / previous.fs
    origin = start_sample + (len(audio) - 1) / 2
    elapsed = (origin - previous.origin_sample) / previous.fs
    frequency = previous.frequency_hz + elapsed * previous.rate_hz_per_sec
    u = (np.arange(len(audio)) - (len(audio) - 1) / 2) / len(audio)
    parameters = np.column_stack((2 * np.pi * duration * frequency,
                                  np.pi * duration ** 2 * previous.rate_hz_per_sec,
                                  duration * previous.log_gain_per_sec))
    endpoints = frequency + u[[0, -1], None] * duration * previous.rate_hz_per_sec
    if np.any(endpoints <= 0) or np.any(endpoints >= previous.fs / 2):
        raise ArithmeticError("retained trajectory does not cover the fitting interval")
    coefficients = np.linalg.lstsq(_basis(u, parameters), audio, rcond=1e-10)[0]
    count = len(frequency)
    amplitude = coefficients[:count] - 1j * coefficients[count:]
    frequency.setflags(write=False)
    amplitude.setflags(write=False)
    return TrajectoryFit(previous.fs, origin, frequency, previous.rate_hz_per_sec,
                         previous.log_gain_per_sec, amplitude, start_sample,
                         start_sample + len(audio))


class TrajectoryObserver:
    """Compare continuation, state update, stationary and gliding hypotheses."""

    variants = ('continue', 'state_update', 'stationary', 'glide')

    def __init__(self, fs, *, fit_sec=.04, horizon_sec=.01, step_sec=.05,
                 counts=(0, 1, 2, 4, 8), max_iterations=30):
        if (not isinstance(fs, int) or fs < 16000
                or not all(np.isfinite(t) and t > 0 for t in (fit_sec, horizon_sec, step_sec))
                or not counts or 0 not in counts
                or any(not isinstance(n, int) or n < 0 for n in counts)):
            raise ValueError("expected sample rate, positive intervals and component counts")
        self.fs = fs
        self.fit_samples, self.horizon, self.step = [round(t * fs) for t in (fit_sec, horizon_sec, step_sec)]
        if (self.fit_samples <= max(8, 8 * max(counts)) or self.horizon < 1
                or self.step < self.horizon or max_iterations < 1):
            raise ValueError("inadmissible fitting windows")
        self.counts, self.max_iterations = tuple(counts), max_iterations
        self.history_size = self.fit_samples + self.horizon
        self.history = np.empty(0)
        self.previous = None
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
        if self.next_sample is None or start_sample != self.next_sample:
            if self.next_sample is not None:
                rows.append(dict(kind='input_gap', missing_start_sample=self.next_sample,
                                 available_sample=start_sample))
            self.history = np.empty(0)
            self.previous = None
            self.next_issue = start_sample + self.history_size
        cursor, end = start_sample, start_sample + len(audio)
        while cursor < end:
            stop = min(end, self.next_issue)
            self.history = np.concatenate((self.history, audio[cursor-start_sample:stop-start_sample]))[-self.history_size:]
            cursor = stop
            if cursor != self.next_issue:
                continue
            evidence_start = cursor - self.history_size
            train, validation = self.history[:-self.horizon], self.history[-self.horizon:]
            groups = [[], [], [], []]
            if self.previous is not None:
                groups[0] = [self.previous]
                try:
                    groups[1] = [update_trajectory_state(train, evidence_start, self.previous)]
                except (ArithmeticError, np.linalg.LinAlgError):
                    pass
            for index, glide in [(2, False), (3, True)]:
                groups[index] = fit_trajectory_candidates(
                    train, evidence_start, self.fs, counts=self.counts,
                    glide=glide, max_iterations=self.max_iterations,
                    initial=groups[2] if glide else None)
            models, predictions, scores, candidate_scores = [], [], [], []
            for group in groups:
                past_scores = []
                forecasts = []
                for candidate in group:
                    try:
                        prediction = candidate.predict(cursor, self.horizon)
                        error = candidate.predict(cursor-self.horizon, self.horizon) - validation
                        score = float(np.mean(error ** 2))
                    except ArithmeticError:
                        prediction, score = np.zeros(self.horizon), float('inf')
                    forecasts.append(prediction)
                    past_scores.append(score)
                index = int(np.argmin(past_scores)) if past_scores else None
                available = index is not None and np.isfinite(past_scores[index])
                models.append(group[index] if available else None)
                predictions.append(forecasts[index] if available else np.zeros(self.horizon))
                scores.append(past_scores[index] if available else float('inf'))
                candidate_scores.append([s if np.isfinite(s) else None for s in past_scores])
            selected = int(np.argmin(scores))
            self.previous = models[selected]
            predictions = np.array(predictions)
            predictions.setflags(write=False)
            rows.append(dict(kind='forecast', issued_sample=cursor,
                             target_end_sample=cursor+self.horizon, models=models,
                             predictions=predictions, available=[m is not None for m in models],
                             selected_variant=self.variants[selected],
                             past_mse=[s if np.isfinite(s) else None for s in scores],
                             candidate_past_mse=candidate_scores,
                             prediction_sha256=hashlib.sha256(predictions.tobytes()).hexdigest()))
            self.next_issue += self.step
        self.next_sample = end
        return rows
