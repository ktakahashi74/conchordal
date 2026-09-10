"""Stratified integration of bounded auditory observations of whole trajectories.

Candidate weights remain conditional acoustic weights, not source identities.
See Owen, Monte Carlo theory, methods and examples, section 8.4:
https://artowen.su.domains/mc/Ch-var-basic.pdf
"""
import math

import numpy as np

from acoustic_mixture import AcousticMixtureForecast


def stratified_samples(forecast, rng, *, target_draws, min_per_component, max_omitted_mass):
    """Guarantee draws for retained components; bound only omitted bounded values.

    The target may be exceeded to provide the minimum per component while
    keeping the bounded-observable variance bound no worse than target_draws
    independent draws. This does not guarantee lower actual variance.
    Omitted mass bounds contributions only for a declared bounded observable,
    never for unbounded PCM.
"""
    if (not isinstance(target_draws, int) or target_draws < 2
            or not isinstance(min_per_component, int) or min_per_component < 2
            or not np.isfinite(max_omitted_mass) or not 0 <= max_omitted_mass < 1):
        raise ValueError('expected at least two draws per component and an omission budget in [0, 1)')
    if isinstance(forecast, AcousticMixtureForecast):
        components, log_weights = forecast.components, forecast.log_weights.copy()
    else:
        components, log_weights = (forecast,), np.array([0.])
    log_weights -= np.logaddexp.reduce(log_weights)
    order = np.argsort(-log_weights, kind='stable')
    ordered = log_weights[order]
    tails = np.r_[np.logaddexp.accumulate(ordered[::-1])[::-1], -np.inf]
    threshold = -np.inf if max_omitted_mass == 0 else np.log(max_omitted_mass)
    keep = next(n for n in range(1, len(order)+1) if tails[n] <= threshold)
    indices = order[:keep]
    weights = np.exp(log_weights[indices])
    total = max(target_draws, min_per_component * keep)
    allocation = (total - min_per_component * keep) * weights / weights.sum()
    counts = min_per_component + np.floor(allocation).astype(int)
    remainder = total - int(counts.sum())
    counts[np.argsort(-(allocation - np.floor(allocation)), kind='stable')[:remainder]] += 1
    # Minimum allocations to tiny components must not starve dominant ones.
    while float(np.sum(weights**2 / counts)) > (1. / target_draws) * (1. + 1e-14):
        counts[int(np.argmax(weights**2 / (counts * (counts+1))))] += 1
    total = int(counts.sum())
    draws = np.concatenate([components[index].sample(rng, int(count)) for index, count in zip(indices, counts)])
    log_tail = float(tails[keep])
    record = dict(component_indices=indices.tolist(), component_weights=weights.tolist(),
                  component_log_weights=log_weights[indices].tolist(), draw_counts=counts.tolist(),
                  actual_draws=total, target_draws=target_draws, min_per_component=min_per_component,
                  max_omitted_mass=max_omitted_mass, omitted_mass=float(np.exp(log_tail)),
                  omitted_log_mass=None if log_tail == -np.inf else log_tail)
    return draws, record


def bounded_summary(values, record, *, alpha=.05, bounds=(0., 1.)):
    """Integrate bounded values and report numerical, not predictive, uncertainty.

The empirical standard error estimates within-component Monte Carlo error.
Hoeffding bounds are marginal for each output coordinate, conditional on the
model and independent draws. They do not test calibration against actual sound.
    Omitted mass contributes mass * bounds, independently of the sampling bound.
"""
    values = np.asarray(values, dtype=float)
    counts = np.asarray(record['draw_counts'], dtype=int)
    weights = np.asarray(record['component_weights'], dtype=float)
    omitted = record['omitted_mass']
    if len(bounds) != 2 or not np.isfinite(bounds).all() or not bounds[0] < bounds[1]:
        raise ValueError('expected a finite nonempty observation range')
    lower, upper = bounds
    if (values.ndim < 1 or len(values) != int(counts.sum()) or np.any(counts < 2)
            or not len(counts) or weights.shape != counts.shape or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isfinite(omitted) or not 0 <= omitted <= 1
            or not np.isclose(math.fsum(weights)+omitted, 1., rtol=0., atol=1e-12)
            or not np.isfinite(values).all() or np.any(values < lower) or np.any(values > upper)
            or not np.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError('expected bounded independent component samples with conserved mass')
    mean, variance = np.zeros(values.shape[1:]), np.zeros(values.shape[1:])
    start = 0
    for weight, count in zip(weights, counts):
        sample = values[start:start+count]
        mean += weight * sample.mean(axis=0)
        variance += weight**2 * sample.var(axis=0, ddof=1) / count
        start += count
    radius = (upper-lower) * math.sqrt(.5 * float(np.sum(weights**2 / counts)) * math.log(2 / alpha))
    return dict(retained_mean=mean, omitted_mass=omitted, monte_carlo_se=np.sqrt(variance),
                omission_midpoint=mean+.5*(lower+upper)*omitted, hoeffding_radius=radius,
                numerical_lower=np.maximum(lower, mean+lower*omitted-radius),
                numerical_upper=np.minimum(upper, mean+upper*omitted+radius),
                marginal_alpha=alpha)
