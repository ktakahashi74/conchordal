"""Research credit assignment from Goh, Ursekar & Howard (2022), section 2.

Inputs are projected histories of counted events and pretrained pairwise log
rates. They are not auditory amplitudes. The additive cue model assumes stable
event statistics; it does not implement extinction, event extraction or policy
learning. Time integrals use a finite age domain with trapezoidal quadrature.
"""

import math

import numpy as np


class TemporalCredit:
    def __init__(self, ages_sec, post_order, base_rates, pair_log_rates, learning_rate):
        ages = np.array(ages_sec, dtype=float, copy=True)
        base = np.array(base_rates, dtype=float, copy=True)
        pair = np.array(pair_log_rates, dtype=float, copy=True)
        if (ages.ndim != 1 or len(ages) < 2 or not np.all(np.isfinite(ages))
                or np.any(ages <= 0) or np.any(np.diff(ages) <= 0)
                or not isinstance(post_order, int) or post_order < 1
                or base.ndim != 1 or not len(base) or not np.all(np.isfinite(base)) or np.any(base <= 0)
                or pair.shape != (len(ages), len(base), len(base)) or not np.all(np.isfinite(pair))
                or not math.isfinite(learning_rate) or not 0 <= learning_rate <= 1):
            raise ValueError("expected ordered positive ages, an integer order, positive base rates, finite aligned pair log rates and a learning rate in [0,1]")
        self.ages_sec, self.post_order = ages, post_order
        self.base_log_rates, self.pair_log_rates = np.log(base), pair
        widths = np.diff(ages, prepend=ages[0], append=ages[-1])
        self.age_weights = (widths[:-1] + widths[1:]) / 2
        self.log_credit = np.zeros_like(pair)
        self.learning_rate = learning_rate
        digamma = sum(1 / n for n in range(1, post_order)) - 0.5772156649015329
        self.log_kappa = (post_order + 1) * (math.log(post_order) - digamma)

    def event_kernel(self, delays_sec):
        delays = np.asarray(delays_sec, dtype=float)
        if delays.ndim != 1 or not len(delays) or not np.all(np.isfinite(delays)) or np.any(delays <= 0):
            raise ValueError("expected positive finite delays for projected unit events")
        k = self.post_order
        ratio = delays[:, None] / self.ages_sec
        return np.exp((k + 1) * math.log(k) - math.lgamma(k + 1)
                      + k * np.log(ratio) - k * ratio - np.log(self.ages_sec))

    def pair_prediction(self, cues, delays_sec):
        cues = np.asarray(cues)
        if (cues.ndim != 1 or not len(cues) or not np.issubdtype(cues.dtype, np.integer)
                or np.any(cues < 0) or np.any(cues >= len(self.base_log_rates))
                or len(np.unique(cues)) != len(cues)):
            raise ValueError("expected distinct event-type indices")
        kernel = self.event_kernel(delays_sec)
        combined = np.mean(self.pair_log_rates[:, :, cues], axis=2)
        return {"log_rate": self.log_kappa + np.einsum("da,ai,a->di", kernel, combined, self.age_weights),
                "unit_event_mass": kernel @ self.age_weights}

    def predict_log_rates(self, projected_history):
        history = np.asarray(projected_history, dtype=float)
        expected = (len(self.ages_sec), len(self.base_log_rates))
        if (history.ndim != 3 or history.shape[1:] != expected
                or not np.all(np.isfinite(history)) or np.any(history < 0)):
            raise ValueError("expected finite nonnegative projected event history indexed by future, age, event type")
        return self.base_log_rates + np.einsum("aij,daj,a->di", self.log_credit, history, self.age_weights)

    def update(self, cue, projected_history_before_event):
        if not isinstance(cue, (int, np.integer)) or not 0 <= cue < len(self.base_log_rates):
            raise ValueError("expected an event-type index")
        prior = self.predict_log_rates(projected_history_before_event)
        if prior.shape != (len(self.ages_sec), len(self.base_log_rates)):
            raise ValueError("credit updates require projection to the declared age grid")
        due = self.pair_prediction([cue], self.ages_sec)
        before = self.log_credit[:, :, cue].copy()
        rate = self.learning_rate
        if rate == 1:
            after = due["log_rate"] - prior
        elif rate == 0:
            after = before.copy()
        else:
            # The paper updates exp(C) toward a rate ratio, not C toward a log ratio.
            after = np.logaddexp(math.log1p(-rate) + before,
                                 math.log(rate) + due["log_rate"] - prior)
        self.log_credit[:, :, cue] = after
        return {"log_prior_rate": prior, "log_due_rate": due["log_rate"],
                "log_credit_before": before, "log_credit_after": after.copy(),
                "unit_event_mass": due["unit_event_mass"]}
