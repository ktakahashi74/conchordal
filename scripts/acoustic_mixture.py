"""Conditional predictive mixtures, not probabilities of perceived source identity."""
from dataclasses import dataclass
import hashlib

import numpy as np

from acoustic_posterior import AcousticForecast


@dataclass(frozen=True)
class AcousticMixtureForecast:
    components: tuple[AcousticForecast, ...]
    log_weights: np.ndarray

    def __post_init__(self):
        if (not self.components or self.log_weights.shape != (len(self.components),)
                or not np.isclose(np.logaddexp.reduce(self.log_weights), 0., atol=1e-10)
                or any(c.issued_sample != self.components[0].issued_sample
                       or c.mean.shape != self.components[0].mean.shape for c in self.components)):
            raise ValueError('components must share a target and normalized log weights')
        self.log_weights.setflags(write=False)

    @property
    def issued_sample(self):
        return self.components[0].issued_sample

    @property
    def mean(self):
        result = sum(weight*c.mean for weight, c in zip(np.exp(self.log_weights), self.components))
        result.setflags(write=False)
        return result

    def diagonal_variance(self):
        mean = self.mean
        return sum(weight*(c.diagonal_variance()+(c.mean-mean)**2)
                   for weight, c in zip(np.exp(self.log_weights), self.components))

    def with_known_waveform(self, audio):
        return AcousticMixtureForecast(tuple(c.with_known_waveform(audio) for c in self.components),
                                       self.log_weights.copy())

    def log_density(self, audio):
        return float(np.logaddexp.reduce(self.log_weights +
                                        np.array([c.log_density(audio) for c in self.components])))

    def sample(self, rng, count):
        if not isinstance(count, int) or count < 1:
            raise ValueError('expected a positive trajectory count')
        weights = np.exp(self.log_weights)
        chosen = rng.choice(len(weights), size=count, p=weights/weights.sum())
        draws = np.empty((count, len(self.components[0].mean)))
        for index, component in enumerate(self.components):
            positions = np.flatnonzero(chosen == index)
            if len(positions):
                # The same model generates a whole trajectory, not one sample at a time.
                draws[positions] = component.sample(rng, len(positions))
        return draws

    def digest(self):
        digest = hashlib.sha256(self.log_weights.tobytes())
        for component in self.components:
            digest.update(component.digest().encode())
        return digest.hexdigest()


class AcousticModelBank:
    """Retain all admitted conditional models and update their predictive weights."""

    def __init__(self, models, prior_mass):
        self.models = tuple(models)
        prior_mass = np.asarray(prior_mass, dtype=float)
        if (not self.models or prior_mass.shape != (len(self.models),)
                or not np.isfinite(prior_mass).all() or np.any(prior_mass < 0.)
                or not prior_mass.sum() > 0.
                or any(m.next_sample != self.models[0].next_sample for m in self.models)):
            raise ValueError('expected contemporaneous models with nonnegative prior mass')
        self.log_weights = np.full(len(prior_mass), -np.inf)
        np.log(prior_mass, out=self.log_weights, where=prior_mass > 0.)
        self.log_weights -= np.logaddexp.reduce(self.log_weights)

    @property
    def next_sample(self):
        return self.models[0].next_sample

    def forecast(self, count):
        return AcousticMixtureForecast(tuple(model.forecast(count) for model in self.models),
                                       self.log_weights.copy())

    def observe(self, start, audio):
        audio = np.asarray(audio, dtype=float)
        if (start != self.next_sample or audio.ndim != 1 or not len(audio)
                or not np.isfinite(audio).all()):
            raise ValueError('expected contiguous finite observations')
        prediction = self.forecast(len(audio))
        scores = np.array([component.log_density(audio) for component in prediction.components])
        if not np.isfinite(scores).all():
            raise ArithmeticError('a conditional model returned a nonfinite predictive density')
        combined = self.log_weights+scores
        density = float(np.logaddexp.reduce(combined))
        for model in self.models:
            model.observe(start, audio)
        # Keep log weights even when exp(weight) underflows, so later evidence can revive a model.
        self.log_weights = combined-density
        return density, scores
