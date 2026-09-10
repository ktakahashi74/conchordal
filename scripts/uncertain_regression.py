"""Nonnegative observations with explicit Gaussian residual and observation noise.

The folded observation law is an engineering hypothesis, not an auditory fit.
Latent-sign mixtures are exact before the reported numerical path truncation.
"""

import math
from functools import lru_cache

import numpy as np
from scipy.special import logsumexp, ndtr, ndtri


_quadrature = lru_cache(maxsize=8)(np.polynomial.legendre.leggauss)


def log_density(forecast, value):
    """Continuous density on [0, infinity); zero is not a separate atom."""
    if not math.isfinite(value) or value < 0:
        raise ValueError('expected a nonnegative finite observed value')
    location = np.asarray(forecast['location'])
    variance = np.asarray(forecast['component_observation_variance'])
    signs = np.array([value, -value])[:, None]
    log_normal = -.5 * ((signs-location)**2 / variance + np.log(2*math.pi*variance))
    return float(logsumexp(log_normal + forecast['log_weights']))


def cdf(forecast, values, *, survival=False):
    """CDF of the magnitude, including no probability atom at zero."""
    points = np.asarray(values, dtype=float)
    if not np.isfinite(points).all() or (points < 0).any():
        raise ValueError('expected nonnegative finite CDF positions')
    location = np.asarray(forecast['location'])
    sd = np.sqrt(forecast['component_observation_variance'])
    upper = (points[..., None]-location) / sd
    lower = (-points[..., None]-location) / sd
    # Choose a tail orientation that avoids subtracting two values near one.
    mass = (ndtr(lower)+ndtr(-upper) if survival else
            np.where(location >= 0, ndtr(upper)-ndtr(lower), ndtr(-lower)-ndtr(-upper)))
    return mass @ np.exp(forecast['log_weights'])


def bounded_crps(forecast,value,*,scale=1.,order=8):
    """CRPS in y/(y+scale), partitioning near mixture quantiles before integration."""
    if (not math.isfinite(value) or value<0 or not math.isfinite(scale) or scale<=0
            or not isinstance(order,int) or order<2):
        raise ValueError('expected nonnegative observation, positive scale and quadrature order')
    location=np.asarray(forecast['location'],dtype=float)
    variance=np.broadcast_to(forecast['component_observation_variance'],location.shape)
    weights=np.exp(forecast['log_weights'])
    if (location.ndim!=1 or not len(location) or weights.shape!=location.shape
            or not np.isfinite(location).all() or not np.isfinite(variance).all()
            or (variance<=0).any() or not np.isfinite(weights).all()
            or not math.isclose(float(weights.sum()),1.,abs_tol=1e-10,rel_tol=0)):
        raise ValueError('expected a normalized finite folded Gaussian mixture')
    nodes,masses=_quadrature(16)
    samples=abs(location[:,None]+np.sqrt(variance)[:,None]*ndtri((nodes+1)/2))
    samples=(samples/(samples+scale)).ravel()
    masses=(weights[:,None]*masses/2).ravel()
    samples,inverse=np.unique(samples,return_inverse=True)
    masses=np.bincount(inverse,weights=masses)
    present=masses>0
    knots=np.interp(np.arange(1,16)/16,np.cumsum(masses[present]),samples[present])
    at=value/(value+scale)
    outer=np.array([max(0.,float(np.min(abs(location)-8*np.sqrt(variance)))),
                    float(np.max(abs(location)+8*np.sqrt(variance)))])
    edges=np.unique(np.r_[0.,knots,outer/(outer+scale),at,1.])
    widths=np.diff(edges)
    nodes,node_weights=_quadrature(order)
    points=edges[:-1,None]+widths[:,None]*(nodes+1)/2
    left=edges[1:]<=at
    probabilities=np.zeros_like(points)
    for selected,survival in [(left,False),(~left,True)]:
        u=points[selected]
        finite=u<1
        part=np.full(u.shape,0. if survival else 1.)
        part[finite]=cdf(forecast,scale*u[finite]/(1-u[finite]),survival=survival)
        probabilities[selected]=part
    return float(widths/2@(probabilities**2@node_weights))


class MagnitudeRegression:
    def __init__(self, prior_mean, prior_covariance, *, residual_sd, max_paths):
        mean = np.asarray(prior_mean, dtype=float)
        covariance = np.asarray(prior_covariance, dtype=float)
        if (mean.ndim != 1 or not mean.size or not np.isfinite(mean).all()
                or covariance.shape != (mean.size, mean.size)
                or not np.isfinite(covariance).all()
                or not np.allclose(covariance, covariance.T, rtol=0, atol=1e-14)
                or not math.isfinite(residual_sd) or residual_sd < 0
                or not isinstance(max_paths, int) or max_paths < 1):
            raise ValueError('expected a finite Gaussian prior and explicit variance/budget')
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError('expected positive definite prior covariance') from error
        self.precision = np.linalg.inv(covariance)
        self.information = (self.precision @ mean)[None, :]
        self.residual_variance = residual_sd**2
        self.max_paths = max_paths
        self.log_weights = np.zeros(1)
        self.completed = 0

    def forecast(self, design, *, observation_sd):
        x = np.asarray(design, dtype=float)
        variance = self.residual_variance + observation_sd**2
        if (x.shape != (self.precision.shape[0],) or not np.isfinite(x).all()
                or not math.isfinite(observation_sd) or observation_sd < 0
                or not math.isfinite(variance) or variance <= 0):
            raise ValueError('expected finite design and strictly positive total variance')
        cov_x = np.linalg.solve(self.precision, x)
        parameter_variance = float(x @ cov_x)
        return dict(completed=self.completed, location=self.information @ cov_x,
                    log_weights=self.log_weights.copy(),
                    within_component_parameter_variance=parameter_variance,
                    residual_variance=self.residual_variance,
                    observation_noise_variance=observation_sd**2,
                    component_observation_variance=parameter_variance + variance)

    def observe(self, design, value, *, observation_sd):
        forecast = self.forecast(design, observation_sd=observation_sd)
        if not math.isfinite(value) or value < 0:
            raise ValueError('expected a nonnegative finite observed value')
        x = np.asarray(design, dtype=float)
        variance = self.residual_variance + observation_sd**2
        predicted_variance = forecast['component_observation_variance']
        signs = np.array([value, -value]) if value > 0 else np.array([0.])
        log_normal = -.5 * ((signs[:, None]-forecast['location'])**2 / predicted_variance
                           + math.log(2*math.pi*predicted_variance))
        if value == 0:
            log_normal += math.log(2.)
        joint = (log_normal + self.log_weights).reshape(-1)
        evidence = float(logsumexp(joint))
        information = (self.information[None, :, :]
                       + signs[:, None, None] * (x/variance)).reshape(-1, x.size)
        # Identical sufficient statistics represent the same posterior component.
        information, inverse = np.unique(information, axis=0, return_inverse=True)
        merged = np.full(len(information), -np.inf)
        np.logaddexp.at(merged, inverse, joint-evidence)
        keep = np.argsort(-merged, kind='stable')[:self.max_paths]
        retained = float(logsumexp(merged[keep]))
        self.precision += np.outer(x, x)/variance
        self.information = information[keep].copy()
        self.log_weights = merged[keep]-retained
        self.completed += 1
        return dict(filter_log_density=evidence, pruned_mass=max(0., -math.expm1(retained)),
                    merged_paths=len(information), retained_paths=len(keep))
