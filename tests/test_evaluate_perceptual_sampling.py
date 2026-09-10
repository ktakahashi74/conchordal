from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from acoustic_mixture import AcousticMixtureForecast
from perceptual_sampling import bounded_summary, stratified_samples


@dataclass
class ConstantForecast:
    mean: np.ndarray
    issued_sample: int = 0

    def sample(self, rng, count):
        return np.broadcast_to(self.mean, (count, len(self.mean))).copy()


class PerceptualSamplingTests(unittest.TestCase):
    def test_rare_component_is_integrated_instead_of_lost_in_thirty_two_choices(self):
        mixture = AcousticMixtureForecast((ConstantForecast(np.zeros(3)), ConstantForecast(np.ones(3))),
                                          np.log([.999, .001]))
        ordinary = mixture.sample(np.random.default_rng(185211), 32)
        np.testing.assert_array_equal(ordinary, 0.)
        draws, record = stratified_samples(mixture, np.random.default_rng(185211), target_draws=32,
                                           min_per_component=4, max_omitted_mass=1e-6)
        self.assertEqual(record['draw_counts'], [32, 4])
        self.assertLessEqual(sum(w*w/n for w, n in zip(record['component_weights'], record['draw_counts'])), 1/32)
        result = bounded_summary(draws, record)
        np.testing.assert_allclose(result['retained_mean'], .001, atol=1e-16)
        np.testing.assert_array_equal(result['monte_carlo_se'], 0.)
        self.assertEqual(record['omitted_mass'], 0.)

    def test_omitted_mass_is_not_renormalized_and_bounds_any_bounded_tail(self):
        components = tuple(ConstantForecast(np.array([x])) for x in [.25, .6, 1.])
        mixture = AcousticMixtureForecast(components, np.log([.97, .02, .01]))
        draws, record = stratified_samples(mixture, np.random.default_rng(4), target_draws=16,
                                           min_per_component=2, max_omitted_mass=.011)
        self.assertEqual(record['component_indices'], [0, 1])
        self.assertAlmostEqual(record['omitted_mass'], .01)
        result = bounded_summary(draws, record)
        expected = .97*.25 + .02*.6
        self.assertAlmostEqual(float(result['retained_mean'][0]), expected)
        for tail_value in (0., .3, 1.):
            truth = expected+.01*tail_value
            self.assertLessEqual(float(result['retained_mean'][0])-1e-15, truth)
            self.assertGreaterEqual(float(result['retained_mean'][0])+record['omitted_mass']+1e-15, truth)

    def test_minimum_allocation_underflow_and_single_forecasts(self):
        mixture = AcousticMixtureForecast(tuple(ConstantForecast(np.zeros(2)) for _ in range(3)),
                                          np.array([0., -1000., -np.inf]))
        draws, record = stratified_samples(mixture, np.random.default_rng(3), target_draws=2,
                                           min_per_component=2, max_omitted_mass=0.)
        self.assertEqual(record['component_indices'], [0, 1])
        self.assertEqual(record['actual_draws'], 4)
        self.assertEqual(len(draws), 4)
        self.assertIsNone(record['omitted_log_mass'])
        draws, record = stratified_samples(ConstantForecast(np.array([.4, .7])), np.random.default_rng(3),
                                           target_draws=8, min_per_component=2, max_omitted_mass=1e-6)
        np.testing.assert_allclose(bounded_summary(draws, record)['retained_mean'], [.4, .7])

    def test_standard_error_and_bounds_are_for_integration_not_predictive_spread(self):
        values = np.array([[0., .2], [1., .4], [.4, .5], [.6, .9]])
        record = dict(draw_counts=[2, 2], component_weights=[.8, .2], omitted_mass=0.)
        result = bounded_summary(values, record, alpha=.1)
        expected_variance = .8**2*np.array([.5, .02])/2 + .2**2*np.array([.02, .08])/2
        np.testing.assert_allclose(result['monte_carlo_se']**2, expected_variance)
        self.assertAlmostEqual(result['hoeffding_radius'], np.sqrt(.5*(.8**2/2+.2**2/2)*np.log(20)))
        self.assertTrue(np.all(result['numerical_lower'] <= result['retained_mean']))
        self.assertTrue(np.all(result['numerical_upper'] >= result['retained_mean']))
        with self.assertRaises(ValueError):
            bounded_summary(values+1., record)
        with self.assertRaises(ValueError):
            bounded_summary(values, dict(record, component_weights=[.8, .8]))
        with self.assertRaises(ValueError):
            stratified_samples(ConstantForecast(np.zeros(1)), np.random.default_rng(1),
                               target_draws=1, min_per_component=1, max_omitted_mass=0.)

    def test_signed_paired_effect_keeps_zero_exact_and_symmetric_omitted_mass(self):
        record = dict(draw_counts=[2, 2], component_weights=[.7, .2], omitted_mass=.1)
        zero = bounded_summary(np.zeros((4, 3)), record, bounds=(-1., 1.))
        np.testing.assert_array_equal(zero['omission_midpoint'], 0.)
        np.testing.assert_array_equal(zero['monte_carlo_se'], 0.)
        values = np.array([.2, .6, -.4, -.6])
        result = bounded_summary(values, record, bounds=(-1., 1.))
        self.assertAlmostEqual(float(result['omission_midpoint']), .18)
        self.assertAlmostEqual(float(result['monte_carlo_se'])**2, .02)
        ordinary = bounded_summary((values+1)/2, record)
        self.assertAlmostEqual(result['hoeffding_radius'], 2*ordinary['hoeffding_radius'])
        for bounds in ((1., 1.), (1., -1.), (0., float('inf'))):
            with self.assertRaises(ValueError):
                bounded_summary(values, record, bounds=bounds)


if __name__ == '__main__':
    unittest.main()
