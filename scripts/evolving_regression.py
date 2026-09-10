"""Continuous-time predictive deviation with retained coefficient uncertainty.

Conditional on v, dr=sqrt(q*v)dW while regression coefficients are static.
The diffusion ratio is a declared hypothesis, not a cognitive retention rate.
"""

import copy
import math

import numpy as np

from residual_regression import ResidualRegression


class EvolvingResidualRegression:
    def __init__(self, prior_mean, relative_covariance, *, drift_per_sec,
                 initial_time_sec, **parameters):
        if (not math.isfinite(drift_per_sec) or drift_per_sec < 0
                or not math.isfinite(initial_time_sec)):
            raise ValueError('expected finite nonnegative diffusion and an initial clock')
        self.posterior = ResidualRegression(prior_mean, relative_covariance, **parameters)
        self.dimensions = len(prior_mean)
        self.drift_per_sec, self.clock_sec = drift_per_sec, initial_time_sec
        self.last_outcome_time_sec = None
        if drift_per_sec > 0:
            # The extra deviation starts deterministically at zero, independently of beta.
            self.posterior.mean = np.pad(self.posterior.mean, ((0, 0), (0, 1)))
            self.posterior.relative_covariance = np.pad(self.posterior.relative_covariance, (0, 1))

    @property
    def completed(self):
        return self.posterior.completed

    def _project(self, time_sec):
        if not math.isfinite(time_sec) or time_sec < self.clock_sec:
            raise ValueError('cannot project before the current evidence clock')
        projected = copy.copy(self.posterior)
        projected.relative_covariance = self.posterior.relative_covariance.copy()
        if self.drift_per_sec > 0:
            projected.relative_covariance[-1, -1] += self.drift_per_sec*(time_sec-self.clock_sec)
        return projected

    def advance_to(self, time_sec):
        self.posterior = self._project(time_sec)
        self.clock_sec = time_sec

    def forecast(self, target_sec, design, *, offset):
        x = np.asarray(design, dtype=float)
        if x.shape != (self.dimensions,) or not np.isfinite(x).all() or target_sec <= self.clock_sec:
            raise ValueError('expected finite frozen features and a future target')
        projected = self._project(target_sec)
        full_design = np.r_[x, 1.] if self.drift_per_sec > 0 else x
        result = projected.forecast(full_design, offset=offset)
        if self.drift_per_sec > 0:
            relative = projected.relative_covariance
            residual = result['residual_variance_mean']
            result['within_component_parameter_variance'] = residual*float(x@relative[:-1, :-1]@x)
            result['within_component_state_variance'] = residual*relative[-1, -1]
            result['within_component_parameter_state_covariance'] = residual*float(x@relative[:-1, -1])
            result['state_mean'] = projected.mean[:, -1].copy()
        else:
            for name in ('within_component_state_variance',
                         'within_component_parameter_state_covariance', 'state_mean'):
                result[name] = np.zeros(len(projected.scale))
        result.update(target_sec=target_sec, state_time_sec=self.clock_sec,
                      last_outcome_time_sec=self.last_outcome_time_sec,
                      state_drift_per_sec=self.drift_per_sec)
        return result

    def observe(self, target_sec, design, value, *, offset):
        x = np.asarray(design, dtype=float)
        if x.shape != (self.dimensions,) or not np.isfinite(x).all() or target_sec <= self.clock_sec:
            raise ValueError('expected a new observed target and its frozen features')
        projected = self._project(target_sec)
        result = projected.observe(np.r_[x, 1.] if self.drift_per_sec > 0 else x,
                                   value, offset=offset)
        self.posterior, self.clock_sec = projected, target_sec
        self.last_outcome_time_sec = target_sec
        return result
