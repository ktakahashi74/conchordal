"""Conditional positive/zero forecasts with stationary, renewed or recurring contexts.

Finite contexts and a path budget are engineering hypotheses, not source identities
or cognitive retention constants. Positive log-Student means are not finite.
"""

import math

import numpy as np

from context_paths import branches, transition


def log_density(forecast, value):
    """Density against a zero atom plus positive Lebesgue measure."""
    if not math.isfinite(value) or value < 0:
        raise ValueError("expected an observed nonnegative target")
    probability = forecast["positive_probability"]
    if value == 0:
        likelihood = np.log1p(-probability)
    else:
        response = math.log(value)
        degrees = forecast["degrees_freedom"]
        lgamma = np.vectorize(math.lgamma, otypes=[float])
        likelihood = (np.log(probability) - response
                      + lgamma((degrees + 1) / 2) - lgamma(degrees / 2)
                      - .5 * np.log(degrees * np.pi * forecast["scale2"])
                      - (degrees + 1) / 2 * np.log1p(
                          (response - forecast["location"]) ** 2
                          / (degrees * forecast["scale2"])))
    return float(np.logaddexp.reduce(forecast["log_weights"] + likelihood))


class RegressionContexts:
    def __init__(self, dimensions, *, mode, contexts=3, change_rate_hz=.2,
                 max_paths=32, intercept_precision=.01, slope_precision=1.,
                 prior_shape=2., prior_scale=.5, zero_prior=(.5, .5)):
        if (not isinstance(dimensions, int) or dimensions < 0
                or mode not in ("stationary", "renewal", "recurrent")
                or not isinstance(contexts, int) or contexts < 2
                or not isinstance(max_paths, int) or max_paths < 1
                or not math.isfinite(change_rate_hz) or change_rate_hz < 0
                or any(not math.isfinite(v) or v <= 0 for v in
                       (intercept_precision, slope_precision, prior_shape, prior_scale, *zero_prior))
                or len(zero_prior) != 2):
            raise ValueError("invalid explicit regression, transition or budget parameters")
        self.dimensions, self.mode = dimensions, mode
        self.contexts = contexts if mode == "recurrent" else 1
        self.rate, self.max_paths = change_rate_hz, max_paths
        self.prior_cov = np.diag(1 / np.r_[intercept_precision,
                                           np.full(dimensions, slope_precision)])
        self.prior_shape, self.prior_scale = prior_shape, prior_scale
        self.zero_prior = tuple(zero_prior)
        k, d = self.contexts, dimensions + 1
        self.mean = np.zeros((1, k, d))
        self.cov = np.broadcast_to(self.prior_cov, (1, k, d, d)).copy()
        self.shape = np.full((1, k), prior_shape)
        self.scale = np.full((1, k), prior_scale)
        self.positive = np.full((1, k), zero_prior[0])
        self.zero = np.full((1, k), zero_prior[1])
        self.used = np.zeros(1, dtype=int)
        self.active = np.zeros(1, dtype=int)
        self.log_weights = np.zeros(1)
        self.clock_sec = float("-inf")
        self.last_target_sec = None
        self.completed = 0

    def transition(self, elapsed_sec):
        return transition(self.contexts, self.rate, elapsed_sec)

    def _components(self, target_sec, design):
        x = np.asarray(design, dtype=float)
        if (not math.isfinite(target_sec) or target_sec <= self.clock_sec
                or x.shape != (self.dimensions,) or not np.isfinite(x).all()):
            raise ValueError("expected future target time and finite frozen design")
        x = np.r_[1., x]
        elapsed = 0. if self.last_target_sec is None else target_sec - self.last_target_sec
        route = branches(self.mode, self.contexts, self.rate, self.active, self.used,
                         elapsed, self.last_target_sec is None)
        parent, state, reset = route['parent'], route['state'], route['reset']
        mean, cov = self.mean[parent, state].copy(), self.cov[parent, state].copy()
        shape, scale = self.shape[parent, state].copy(), self.scale[parent, state].copy()
        positive, zero = self.positive[parent, state].copy(), self.zero[parent, state].copy()
        mean[reset] = 0.; cov[reset] = self.prior_cov
        shape[reset], scale[reset] = self.prior_shape, self.prior_scale
        positive[reset], zero[reset] = self.zero_prior
        vx = cov @ x
        q = 1 + vx @ x
        location = mean @ x
        return dict(parent=parent, state=state, fresh=route['fresh'], returning=route['returning'],
                    mean=mean, cov=cov, shape=shape, scale=scale, positive=positive, zero=zero,
                    vx=vx, q=q, location=location, scale2=scale / shape * q,
                    log_weights=self.log_weights[parent] + route['log_transition'])

    def forecast(self, target_sec, design):
        c = self._components(target_sec, design)
        return dict(target_sec=target_sec, completed=self.completed,
                    location=c["location"].copy(), scale2=c["scale2"].copy(),
                    degrees_freedom=2*c["shape"], log_weights=c["log_weights"].copy(),
                    positive_probability=c["positive"] / (c["positive"] + c["zero"]))

    def observe(self, target_sec, design, value):
        if not math.isfinite(value) or value < 0:
            raise ValueError("expected an observed nonnegative target")
        c = self._components(target_sec, design)
        forecast = dict(location=c["location"], scale2=c["scale2"],
                        degrees_freedom=2*c["shape"], positive_probability=c["positive"] / (c["positive"] + c["zero"]))
        # Keep the component likelihoods distinct from the issued mixture's score.
        response = math.log(value) if value > 0 else 0.
        degrees = forecast["degrees_freedom"]
        if value > 0:
            lgamma = np.vectorize(math.lgamma, otypes=[float])
            likelihood = (np.log(forecast["positive_probability"]) - response
                          + lgamma((degrees + 1)/2) - lgamma(degrees/2)
                          - .5*np.log(degrees*np.pi*c["scale2"])
                          - (degrees + 1)/2*np.log1p((response-c["location"])**2
                                                    / (degrees*c["scale2"])))
        else:
            likelihood = np.log1p(-forecast["positive_probability"])
        joint = c["log_weights"] + likelihood
        evidence = float(np.logaddexp.reduce(joint))
        posterior = joint - evidence
        keep = np.argsort(-posterior, kind="stable")[:self.max_paths]
        log_kept = float(np.logaddexp.reduce(posterior[keep]))
        parent, state = c["parent"][keep], c["state"][keep]
        for name in ("mean", "cov", "shape", "scale", "positive", "zero"):
            setattr(self, name, getattr(self, name)[parent].copy())
        row = np.arange(len(keep))
        positive = float(value > 0)
        gain = positive/c["q"][keep]
        error = response-c["location"][keep]
        vx = c["vx"][keep]
        self.mean[row, state] = c["mean"][keep] + vx*(gain*error)[:, None]
        self.cov[row, state] = c["cov"][keep] - vx[:, :, None]*vx[:, None, :]*gain[:, None, None]
        self.shape[row, state] = c["shape"][keep] + .5*positive
        self.scale[row, state] = c["scale"][keep] + .5*error**2*gain
        self.positive[row, state] = c["positive"][keep] + positive
        self.zero[row, state] = c["zero"][keep] + (1-positive)
        self.used = np.maximum(self.used[parent], state+1)
        self.active = state.copy()
        self.log_weights = posterior[keep]-log_kept
        self.last_target_sec = self.clock_sec = target_sec
        self.completed += 1
        return dict(filter_log_density=evidence, pruned_mass=max(0., -math.expm1(log_kept)),
                    new_context_mass=float(np.exp(posterior[c["fresh"]]).sum()),
                    returning_context_mass=float(np.exp(posterior[c["returning"]]).sum()),
                    retained_paths=len(keep))

    def skip_to(self, end_sec):
        if not math.isfinite(end_sec) or end_sec <= self.clock_sec:
            raise ValueError("expected an advancing observation clock")
        self.clock_sec = end_sec
