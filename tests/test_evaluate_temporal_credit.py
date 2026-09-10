import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from temporal_credit_reference import TemporalCredit


class TemporalCreditTests(unittest.TestCase):
    def test_neutral_credit_returns_base_rates_for_any_history(self):
        model = TemporalCredit([.1, .5, 2.], 8, [.2, .3], np.zeros((3, 2, 2)), .1)
        history = np.arange(24).reshape(4, 3, 2) / 10
        before = model.log_credit.copy()
        result = model.predict_log_rates(history)
        np.testing.assert_array_equal(result, np.tile(np.log([.2, .3]), (4, 1)))
        np.testing.assert_array_equal(model.log_credit, before)

    def test_unit_event_kernel_has_unit_age_mass_in_the_infinite_domain_limit(self):
        ages = np.geomspace(.0001, 1000., 2049)
        model = TemporalCredit(ages, 8, [1.], np.zeros((len(ages), 1, 1)), .1)
        mass = model.event_kernel([.1, 1., 10.]) @ model.age_weights
        np.testing.assert_allclose(mass, np.ones(3), rtol=2e-5)
        edge_mass = model.event_kernel([ages[0], ages[-1]]) @ model.age_weights
        self.assertTrue(np.all(edge_mass < .7))

    def test_pairwise_log_convolution_matches_fixed_delay_analytic_result(self):
        errors = []
        for nodes in [129, 257, 513]:
            ages = np.geomspace(.001, 100., nodes)
            k, delay = 8, .7
            pair_log = ((k + 1) * math.log(k) - math.lgamma(k + 1) + k * math.log(delay)
                        - (k + 1) * np.log(ages) - k * delay / ages)
            model = TemporalCredit(ages, k, [.1], pair_log[:, None, None], .1)
            future = np.array([.2, .7, 1.5, 3.])
            exact = ((k + 1) * math.log(k) - math.lgamma(k + 1) + k * math.log(delay)
                     - (k + 1) * np.log(future) - k * delay / future)
            error = np.max(np.abs(model.pair_prediction([0], future)["log_rate"][:, 0] - exact))
            errors.append(error)
        self.assertLess(errors[1], errors[0] / 3)
        self.assertLess(errors[2], errors[1] / 3)
        self.assertLess(errors[2], .002)

    def test_simultaneous_pair_cues_use_a_geometric_mean(self):
        ages = [.1, .2, .4, .8, 1.6]
        pair = np.empty((5, 2, 2))
        pair[:, :, 0] = np.log([.2, .4])
        pair[:, :, 1] = np.log([.8, .1])
        model = TemporalCredit(ages, 8, [.1, .1], pair, .1)
        a = model.pair_prediction([0], [.2, .5])["log_rate"]
        b = model.pair_prediction([1], [.2, .5])["log_rate"]
        both = model.pair_prediction([0, 1], [.2, .5])["log_rate"]
        np.testing.assert_allclose(both, (a + b) / 2, atol=1e-14)
        np.testing.assert_array_equal(both, model.pair_prediction([1, 0], [.2, .5])["log_rate"])

    def test_credit_uses_arithmetic_update_of_gain_not_log_gain(self):
        ages = [.1, .3, 1.]
        model = TemporalCredit(ages, 8, [.2, .1], np.log(np.full((3, 2, 2), .4)), .25)
        model.log_credit[:, :, 0] = np.log(2.)
        history = np.zeros((3, 3, 2))
        result = model.update(0, history)
        expected = .75 * 2 + .25 * np.exp(result["log_due_rate"] - result["log_prior_rate"])
        np.testing.assert_allclose(np.exp(result["log_credit_after"]), expected, rtol=1e-14)
        self.assertGreater(np.max(np.abs(result["log_credit_after"]
                                        - (.75 * np.log(2) + .25 * (result["log_due_rate"] - result["log_prior_rate"])))), .01)

    def test_preexisting_prediction_reduces_credit_for_a_redundant_cue(self):
        ages = np.geomspace(.05, 4., 33)
        pair = np.log(np.full((33, 3, 3), .4))
        empty = TemporalCredit(ages, 8, [.1, .1, .1], pair, 1.)
        informed = copy.deepcopy(empty)
        informed.log_credit[:, 2, 0] = math.log(4)
        zero = np.zeros((33, 33, 3))
        history = zero.copy()
        history[:, :, 0] = informed.event_kernel(ages + .5)
        uninformed = empty.update(1, zero)
        informed_result = informed.update(1, history)
        self.assertTrue(np.all(informed_result["log_credit_after"][:, 2] < uninformed["log_credit_after"][:, 2]))
        np.testing.assert_allclose(informed_result["log_credit_after"][:, 2] - uninformed["log_credit_after"][:, 2],
                                   -(informed_result["log_prior_rate"][:, 2] - uninformed["log_prior_rate"][:, 2]))

    def test_log_domain_preserves_extreme_finite_rate_ratios(self):
        model = TemporalCredit([.1, .2, .4], 8, [1.], np.full((3, 1, 1), -1000.), .5)
        model.log_credit.fill(-1000.)
        history = np.ones((3, 3, 1)) * 10_000
        result = model.update(0, history)
        self.assertTrue(np.all(np.isfinite(result["log_credit_after"])))
        self.assertTrue(np.all(result["log_credit_after"] > 1000))

    def test_outputs_are_immutable_and_invalid_input_does_not_learn(self):
        model = TemporalCredit([.1, .3, 1.], 8, [.1], np.zeros((3, 1, 1)), .2)
        history = np.ones((3, 3, 1))
        first = model.update(0, history)
        saved = copy.deepcopy(first)
        model.update(0, history)
        for key in first:
            np.testing.assert_array_equal(first[key], saved[key])
        before = model.log_credit.copy()
        for cue, bad in [(1, history), (0, np.zeros((2, 3, 1))), (0, -history)]:
            with self.assertRaises(ValueError):
                model.update(cue, bad)
        np.testing.assert_array_equal(model.log_credit, before)


if __name__ == "__main__":
    unittest.main()
