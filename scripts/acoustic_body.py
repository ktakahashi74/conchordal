"""Conditional driven modal state, without a second resonant residual state.

Each damped rotation has isotropic continuous Gaussian drive. Measurement
noise has variance q; drive and initial covariance have a declared ratio to q.
A common inverse-Gamma q gives joint Student predictions, not independent
samplewise intervals. Frequencies, damping and variance ratios stay fixed.
Linear SDE discretization and filtering: Särkkä and Solin (2019), §§6.2, 10.6,
https://users.aalto.fi/~asolin/sde-book/sde-book.pdf
These acoustic hypotheses do not identify perceptual bodies or cognitive memory.
"""
from dataclasses import dataclass, replace
import hashlib
import math

import numpy as np


def _condition(transition, drive, observation, mean, covariance, audio):
    mean, covariance = mean.copy(), covariance.copy()
    square, logdet = 0., 0.
    identity = np.eye(len(mean))
    for value in audio:
        cross = covariance @ observation
        variance = float(observation @ cross + 1.)
        error = value - observation @ mean
        gain = cross / variance
        mean += gain * error
        # Joseph form avoids subtracting almost equal covariance matrices.
        residual = identity - np.outer(gain, observation)
        covariance = residual @ covariance @ residual.T + np.outer(gain, gain)
        mean = transition @ mean
        covariance = transition @ covariance @ transition.T + drive
        covariance = (covariance + covariance.T) / 2
        square += error * error / variance
        logdet += np.log(variance)
    return mean, covariance, float(square), float(logdet)


def _density(shape, scale, count, square, logdet):
    return float(math.lgamma(shape + count / 2) - math.lgamma(shape)
                 - .5 * (count * np.log(2 * np.pi * scale) + logdet)
                 - (shape + count / 2) * np.log1p(square / (2 * scale)))


@dataclass(frozen=True)
class DrivenBodyForecast:
    issued_sample: int
    mean: np.ndarray
    transition: np.ndarray
    drive: np.ndarray
    observation: np.ndarray
    state_mean: np.ndarray
    state_covariance: np.ndarray
    variance_shape: float
    variance_scale: float
    known_waveform: np.ndarray

    def __post_init__(self):
        for field in ('mean', 'transition', 'drive', 'observation', 'state_mean',
                      'state_covariance', 'known_waveform'):
            value = getattr(self, field).copy()
            value.setflags(write=False)
            object.__setattr__(self, field, value)

    def with_known_waveform(self, audio):
        audio = np.asarray(audio, dtype=float)
        if audio.shape != self.mean.shape or not np.isfinite(audio).all():
            raise ValueError('known signal must cover the complete forecast window')
        return replace(self, mean=self.mean + audio, known_waveform=self.known_waveform + audio)

    def log_density(self, audio):
        audio = np.asarray(audio, dtype=float)
        if audio.shape != self.mean.shape or not np.isfinite(audio).all():
            raise ValueError('expected one complete finite forecast target')
        _, _, square, logdet = _condition(
            self.transition, self.drive, self.observation, self.state_mean,
            self.state_covariance, audio - self.known_waveform)
        return _density(self.variance_shape, self.variance_scale, len(audio), square, logdet)

    def diagonal_variance(self):
        covariance = self.state_covariance.copy()
        result = np.empty(len(self.mean))
        for index in range(len(result)):
            result[index] = self.observation @ covariance @ self.observation + 1.
            covariance = self.transition @ covariance @ self.transition.T + self.drive
        return result * self.variance_scale / (self.variance_shape - 1.)

    def sample(self, rng, count):
        if not isinstance(count, int) or count < 1:
            raise ValueError('expected a positive trajectory count')
        # One q per whole trajectory preserves joint scale uncertainty.
        scales = np.sqrt(self.variance_scale / rng.gamma(self.variance_shape, size=count))
        values, vectors = np.linalg.eigh(self.state_covariance)
        tolerance = 64 * np.finfo(float).eps * max(1., np.max(np.abs(values), initial=0.))
        if np.min(values, initial=0.) < -tolerance:
            raise ArithmeticError('conditional state covariance is not positive semidefinite')
        root = vectors * np.sqrt(np.maximum(values, 0.))
        states = rng.standard_normal((count, len(values))) @ root.T
        drive_root = np.sqrt(np.diag(self.drive))
        result = np.empty((count, len(self.mean)))
        for index in range(len(self.mean)):
            result[:, index] = (states @ self.observation + rng.standard_normal(count)) * scales + self.mean[index]
            states = states @ self.transition.T + rng.standard_normal(states.shape) * drive_root
        return result

    def digest(self):
        digest = hashlib.sha256(np.asarray([self.issued_sample], dtype=np.int64).tobytes())
        for array in (self.mean, self.transition, self.drive, self.observation,
                      self.state_mean, self.state_covariance, self.known_waveform):
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        digest.update(np.asarray([self.variance_shape, self.variance_scale], dtype=float).tobytes())
        return digest.hexdigest()


class DrivenBodyPosterior:
    """Filter a fixed modal hypothesis, carrying vibration through new evidence.

The state is immediately before observation at next_sample. A subsequent
transition includes all drive accumulated over one sampling interval.
drive_per_sec is the quadrature drive intensity / measurement variance;
initial_precision is inverse normalized covariance per real quadrature.
"""

    def __init__(self, fs, frequency_hz, log_gain_per_sec, *, drive_per_sec,
                 initial_precision, variance_prior, start_sample=0, parameters_available_sample=0):
        frequency = np.asarray(frequency_hz, dtype=float)
        gain = np.asarray(log_gain_per_sec, dtype=float)
        intensity = np.asarray(drive_per_sec, dtype=float)
        if (not isinstance(fs, int) or fs <= 0 or frequency.ndim != 1
                or gain.shape != frequency.shape or not np.isfinite(frequency).all()
                or not np.isfinite(gain).all() or np.any(frequency <= 0)
                or np.any(frequency >= fs / 2) or np.any(gain > 0)
                or intensity.ndim > 1 or (intensity.ndim == 1 and intensity.shape != frequency.shape)
                or not np.isfinite(intensity).all() or np.any(intensity < 0)
                or not np.isfinite(initial_precision) or initial_precision <= 0
                or len(variance_prior) != 2 or not np.isfinite(variance_prior).all()
                or variance_prior[0] <= 1 or variance_prior[1] <= 0
                or not isinstance(start_sample, int) or start_sample < 0
                or not isinstance(parameters_available_sample, int) or parameters_available_sample < 0):
            raise ValueError('expected passive modal parameters and proper positive scale priors')
        width = 2 * len(frequency)
        self.transition = np.zeros((width, width))
        self.observation = np.tile([1., 0.], len(frequency))
        self.drive = np.zeros_like(self.transition)
        intensity = np.broadcast_to(intensity, frequency.shape)
        for index, (f, g, d) in enumerate(zip(frequency, gain, intensity)):
            angle = 2 * np.pi * f / fs
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            block = slice(2 * index, 2 * index + 2)
            self.transition[block, block] = np.exp(g / fs) * rotation
            integral = 1. / fs if g == 0 else np.expm1(2 * g / fs) / (2 * g)
            self.drive[block, block] = np.eye(2) * d * integral
        self.state_mean = np.zeros(width)
        self.state_covariance = np.eye(width) / initial_precision
        self.variance_shape, self.variance_scale = map(float, variance_prior)
        self.next_sample = start_sample
        self.parameters_available_sample = parameters_available_sample
        self.log_evidence = 0.
        self.observed_samples = 0
        self.last_observed_sample = None

    def advance_to(self, sample):
        """Propagate missing time without conditioning on invented observations."""
        if not isinstance(sample, int) or sample < self.next_sample:
            raise ValueError('expected a nondecreasing integer state clock')
        steps = sample - self.next_sample
        if not steps:
            return
        mean, covariance = self.state_mean.copy(), self.state_covariance.copy()
        transition, drive = self.transition, self.drive
        # Compose transitions by squaring; cost is logarithmic in missing duration.
        while steps:
            if steps & 1:
                mean = transition @ mean
                covariance = transition @ covariance @ transition.T + drive
            steps >>= 1
            if steps:
                drive = transition @ drive @ transition.T + drive
                transition = transition @ transition
        self.state_mean = mean
        self.state_covariance = (covariance + covariance.T) / 2
        self.next_sample = sample

    def observe(self, start_sample, audio):
        audio = np.asarray(audio, dtype=float)
        if (start_sample != self.next_sample or audio.ndim != 1 or not np.isfinite(audio).all()):
            raise ValueError('expected contiguous finite observations')
        mean, covariance, square, logdet = _condition(
            self.transition, self.drive, self.observation, self.state_mean, self.state_covariance, audio)
        density = _density(self.variance_shape, self.variance_scale, len(audio), square, logdet)
        self.state_mean, self.state_covariance = mean, covariance
        self.variance_shape += len(audio) / 2
        self.variance_scale += square / 2
        self.log_evidence += density
        self.next_sample += len(audio)
        self.observed_samples += len(audio)
        if len(audio):
            self.last_observed_sample = self.next_sample
        return density

    def forecast(self, count):
        if not isinstance(count, int) or count < 1:
            raise ValueError('expected a positive forecast sample count')
        if self.next_sample < self.parameters_available_sample:
            raise ValueError('parameters were not available at this forecast origin')
        mean, state = np.empty(count), self.state_mean.copy()
        for index in range(count):
            mean[index] = self.observation @ state
            state = self.transition @ state
        return DrivenBodyForecast(self.next_sample, mean, self.transition, self.drive, self.observation,
                                  self.state_mean, self.state_covariance, self.variance_shape,
                                  self.variance_scale, np.zeros(count))
