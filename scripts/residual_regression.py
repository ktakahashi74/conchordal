"""Folded offset regression with learned, uncertain total predictive residual variance.

The normal-inverse-Gamma law is a statistical hypothesis, not an auditory fit.
Coefficient covariance is conditional on variance; Student scale is not variance.
"""

import math
from functools import lru_cache

import numpy as np
from scipy.special import gammaln, logsumexp, stdtr, stdtrit


_quadrature = lru_cache(maxsize=8)(np.polynomial.legendre.leggauss)


@lru_cache(maxsize=4096)
def _student_nodes(degree):
    return stdtrit(degree, (_quadrature(16)[0]+1)/2), stdtrit(degree, 1-1e-6)


def log_density(forecast, value):
    if not math.isfinite(value) or value < 0:
        raise ValueError('expected a finite nonnegative observed target')
    location = np.asarray(forecast['location'])
    scale2 = np.asarray(forecast['scale2'])
    degrees = forecast['degrees_freedom']
    signed = np.array([value, -value])[:, None]
    terms = (gammaln((degrees+1)/2)-gammaln(degrees/2)
             -.5*np.log(degrees*math.pi*scale2)
             -(degrees+1)/2*np.log1p((signed-location)**2/(degrees*scale2)))
    return float(logsumexp(terms+forecast['log_weights']))


def cdf(forecast, values, *, survival=False):
    points = np.asarray(values, dtype=float)
    if not np.isfinite(points).all() or (points < 0).any():
        raise ValueError('expected finite nonnegative evaluation points')
    location = np.asarray(forecast['location'])
    scale = np.sqrt(forecast['scale2'])
    degrees = forecast['degrees_freedom']
    upper, lower = (points[..., None]-location)/scale, (-points[..., None]-location)/scale
    mass = (stdtr(degrees, lower)+stdtr(degrees, -upper) if survival else
            np.where(location >= 0, stdtr(degrees, upper)-stdtr(degrees, lower),
                     stdtr(degrees, -lower)-stdtr(degrees, -upper)))
    return mass @ np.exp(forecast['log_weights'])


def bounded_crps(forecast, value, *, scale=1., order=8):
    """Integrate the folded Student CDF over all of u=y/(y+scale)."""
    if (not math.isfinite(value) or value < 0 or not math.isfinite(scale) or scale <= 0
            or not isinstance(order, int) or order < 2):
        raise ValueError('expected nonnegative observation, positive scale and quadrature order')
    location, variance, weights = (np.asarray(forecast[k], dtype=float)
                                   for k in ('location', 'scale2', 'log_weights'))
    weights = np.exp(weights)
    degree = forecast['degrees_freedom']
    if (location.ndim != 1 or not len(location) or variance.shape != location.shape
            or weights.shape != location.shape or not math.isfinite(degree) or degree <= 2
            or not np.isfinite(location).all() or not np.isfinite(variance).all()
            or (variance <= 0).any() or not np.isfinite(weights).all()
            or not math.isclose(float(weights.sum()), 1., rel_tol=0, abs_tol=1e-10)):
        raise ValueError('expected a normalized finite folded Student mixture with finite variance')
    standard, outer_standard = _student_nodes(degree)
    samples = abs(location[:, None]+np.sqrt(variance)[:, None]*standard)
    samples = (samples/(samples+scale)).ravel()
    masses = (weights[:, None]*_quadrature(16)[1]/2).ravel()
    samples, inverse = np.unique(samples, return_inverse=True)
    masses = np.bincount(inverse, weights=masses)
    present = masses > 0
    knots = np.interp(np.arange(1, 16)/16, np.cumsum(masses[present]), samples[present])
    at = value/(value+scale)
    # Tail landmarks partition the integral; neither tail is discarded.
    outer = np.array([max(0., float(np.min(abs(location)-outer_standard*np.sqrt(variance)))),
                      float(np.max(abs(location)+outer_standard*np.sqrt(variance)))])
    edges = np.unique(np.r_[0., knots, outer/(outer+scale), at, 1.])
    centers = abs(location)/(abs(location)+scale)
    brackets = np.clip(np.searchsorted(edges, centers, side='right')-1, 0, len(edges)-2)
    near = np.maximum(0., abs(location)[:, None]+np.sqrt(variance)[:, None]*np.array([-1., 1.]))
    bounded_near = near/(near+scale)
    narrow = np.diff(bounded_near, axis=1)[:, 0] < np.diff(edges)[brackets]/8
    # Low-mass narrow components can lie between all mixture quantile landmarks.
    if narrow.any():
        landmarks = np.maximum(0., abs(location[narrow, None])
                               + np.sqrt(variance[narrow, None])*np.array([-8., -1., 0., 1., 8.]))
        edges = np.unique(np.r_[edges, (landmarks/(landmarks+scale)).ravel()])
    widths = np.diff(edges)
    nodes, node_weights = _quadrature(order)
    points = edges[:-1, None]+widths[:, None]*(nodes+1)/2
    left = edges[1:] <= at
    probabilities = np.empty_like(points)
    for selected, survival in ((left, False), (~left, True)):
        u = points[selected]
        finite = u < 1
        part = np.full(u.shape, 0. if survival else 1.)
        part[finite] = cdf(forecast, scale*u[finite]/(1-u[finite]), survival=survival)
        probabilities[selected] = part
    return float(widths/2 @ (probabilities**2 @ node_weights))


class ResidualRegression:
    def __init__(self, prior_mean, relative_covariance, *, prior_shape,
                 prior_residual_variance, max_paths):
        mean = np.asarray(prior_mean, dtype=float)
        relative = np.asarray(relative_covariance, dtype=float)
        if (mean.ndim != 1 or not np.isfinite(mean).all()
                or relative.shape != (mean.size, mean.size)
                or not np.isfinite(relative).all()
                or not np.allclose(relative, relative.T, rtol=0, atol=1e-14)
                or not math.isfinite(prior_shape) or prior_shape <= 1
                or not math.isfinite(prior_residual_variance) or prior_residual_variance <= 0
                or not isinstance(max_paths, int) or max_paths < 1):
            raise ValueError('expected a proper conditional prior with finite mean residual variance')
        if mean.size:
            try:
                np.linalg.cholesky(relative)
            except np.linalg.LinAlgError as error:
                raise ValueError('expected positive definite relative coefficient covariance') from error
        self.mean = mean[None, :].copy()
        self.relative_covariance = relative.copy()
        self.shape = float(prior_shape)
        self.scale = np.array([(prior_shape-1)*prior_residual_variance])
        self.log_weights = np.zeros(1)
        self.max_paths = max_paths
        self.completed = 0

    def forecast(self, design, *, offset):
        x = np.asarray(design, dtype=float)
        if (x.shape != (self.mean.shape[1],) or not np.isfinite(x).all()
                or not math.isfinite(offset) or offset < 0):
            raise ValueError('expected finite frozen design and nonnegative observed offset')
        parameter_factor = float(x @ self.relative_covariance @ x)
        residual = self.scale/(self.shape-1)
        return dict(completed=self.completed, observed_offset=offset,
                    location=offset+self.mean @ x, log_weights=self.log_weights.copy(),
                    degrees_freedom=2*self.shape, scale2=self.scale/self.shape*(1+parameter_factor),
                    residual_variance_mean=residual,
                    within_component_parameter_variance=residual*parameter_factor,
                    signed_component_variance=residual*(1+parameter_factor))

    def observe(self, design, value, *, offset):
        if not math.isfinite(value) or value < 0:
            raise ValueError('expected a finite nonnegative observed target')
        forecast = self.forecast(design, offset=offset)
        x = np.asarray(design, dtype=float)
        vx = self.relative_covariance @ x
        q = 1+float(x @ vx)
        signed = np.array([value, -value]) if value > 0 else np.array([0.])
        degrees = forecast['degrees_freedom']
        errors = signed[:, None]-forecast['location']
        terms = (gammaln((degrees+1)/2)-gammaln(degrees/2)
                 -.5*np.log(degrees*math.pi*forecast['scale2'])
                 -(degrees+1)/2*np.log1p(errors**2/(degrees*forecast['scale2'])))
        if value == 0:
            terms += math.log(2.)
        joint = (terms+self.log_weights).ravel()
        evidence = float(logsumexp(joint))
        posterior = joint-evidence
        indices = np.tile(np.arange(len(self.scale)), len(signed))
        error = errors.ravel()
        means = self.mean[indices]+error[:, None]/q*vx
        scales = self.scale[indices]+.5*error**2/q
        means[means == 0] = 0.
        groups, representatives, merged = {}, [], []
        for i, (mean, scale) in enumerate(zip(means, scales, strict=True)):
            key = (mean.tobytes(), float(scale))
            if key in groups:
                group = groups[key]
                merged[group] = np.logaddexp(merged[group], posterior[i])
            else:
                groups[key] = len(merged)
                representatives.append(i)
                merged.append(posterior[i])
        merged = np.array(merged)
        keep = np.argsort(-merged, kind='stable')[:self.max_paths]
        retained = float(logsumexp(merged[keep]))
        selected = np.array(representatives)[keep]
        self.mean, self.scale = means[selected].copy(), scales[selected].copy()
        self.log_weights = merged[keep]-retained
        self.relative_covariance -= np.outer(vx, vx)/q
        self.shape += .5
        self.completed += 1
        return dict(filter_log_density=evidence, pruned_mass=max(0., -math.expm1(retained)),
                    retained_paths=len(keep), merged_paths=len(merged))
