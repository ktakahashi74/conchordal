import copy
import itertools
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from uncertain_regression import MagnitudeRegression, cdf, log_density


class UncertainRegressionTests(unittest.TestCase):
    def test_updates_match_exhaustive_batch_sign_marginals(self):
        mean = np.array([.5, -.1])
        covariance = np.array([[1., .2], [.2, .7]])
        design = np.array([[1., -.3], [1., .8], [1., .2], [1., -.7], [1., .1]])
        values = np.array([.4, 0., .9, .2, .5])
        noises = np.array([.15, .3, .2, .1, .4])
        model = MagnitudeRegression(mean, covariance, residual_sd=.2, max_paths=64)
        previous = 0.
        for count in range(1, len(values)+1):
            i = count-1
            result = model.observe(design[i], values[i], observation_sd=noises[i])
            x, y = design[:count], values[:count]
            variance = .2**2 + noises[:count]**2
            marginal_cov = np.diag(variance) + x@covariance@x.T
            logdet = np.linalg.slogdet(marginal_cov)[1]
            weights, batch_means = [], []
            precision = np.linalg.inv(covariance) + x.T @ np.diag(1/variance) @ x
            for signs in itertools.product((-1., 1.), repeat=count):
                z = y*np.array(signs)
                error = z-x@mean
                weights.append(-.5*(count*math.log(2*math.pi)+logdet
                                    + error@np.linalg.solve(marginal_cov, error)))
                eta = np.linalg.solve(covariance, mean)+x.T@(z/variance)
                batch_means.append(np.linalg.solve(precision, eta))
            total = float(logsumexp(weights))
            self.assertAlmostEqual(result['filter_log_density'], total-previous, places=11)
            self.assertLess(result['pruned_mass'], 1e-13)
            next_x = np.array([1., .4])
            forecast = model.forecast(next_x, observation_sd=.3)
            locations = np.array(batch_means)@next_x
            batch = dict(location=locations, log_weights=np.array(weights)-total,
                         component_observation_variance=.2**2+.3**2+next_x@np.linalg.solve(precision,next_x))
            np.testing.assert_allclose(cdf(forecast, np.linspace(0, 3, 31)),
                                       cdf(batch, np.linspace(0, 3, 31)), atol=1e-13)
            previous = total

    def test_density_normalizes_and_cdf_matches_independent_integration(self):
        model = MagnitudeRegression([1.2], [[.03]], residual_sd=.1, max_paths=32)
        model.observe([1.], .8, observation_sd=.2)
        forecast = model.forecast([1.], observation_sd=.3)
        pdf = lambda y: math.exp(log_density(forecast, y))
        self.assertAlmostEqual(quad(pdf, 0., np.inf, epsabs=1e-11)[0], 1., places=11)
        self.assertEqual(float(cdf(forecast, 0.)), 0.)
        for point in [1e-4, .1, .8, 2., 8.]:
            self.assertAlmostEqual(float(cdf(forecast, point)), quad(pdf,0.,point)[0], places=11)
        negative = dict(forecast, location=-forecast['location'])
        np.testing.assert_allclose(cdf(negative,[0.,.1,1.,8.]),cdf(forecast,[0.,.1,1.,8.]),atol=1e-15)

    def test_zero_limit_is_continuous_in_likelihood_and_learning(self):
        for noise in [.01, .1, 1.]:
            model = MagnitudeRegression([0.], [[100.]], residual_sd=.1, max_paths=512)
            for _ in range(20):
                model.observe([1.], 1., observation_sd=noise)
            issued = model.forecast([1.], observation_sd=noise)
            zero = copy.deepcopy(model)
            zero.observe([1.], 0., observation_sd=noise)
            reference = zero.forecast([1.], observation_sd=noise)
            for value in [1e-6, 1e-12, 1e-24, 1e-48]:
                branch = copy.deepcopy(model)
                update = branch.observe([1.], value, observation_sd=noise)
                self.assertLess(update['pruned_mass'], 1e-10)
                self.assertAlmostEqual(log_density(issued,value),log_density(issued,0.),delta=1e-8)
                np.testing.assert_allclose(cdf(branch.forecast([1.],observation_sd=noise),[.1,.5,1.,2.]),
                                           cdf(reference,[.1,.5,1.,2.]),atol=1e-8,rtol=0)
            self.assertEqual(model.completed, 20)
            self.assertEqual(zero.completed, 21)
            self.assertLess(reference['within_component_parameter_variance'],
                            issued['within_component_parameter_variance'])

    def test_observation_noise_changes_learning_and_predictive_precision(self):
        accurate = MagnitudeRegression([3.], [[.1]], residual_sd=.1, max_paths=16)
        uncertain = copy.deepcopy(accurate)
        accurate.observe([1.], 4., observation_sd=.01)
        uncertain.observe([1.], 4., observation_sd=1.)
        a = accurate.forecast([1.], observation_sd=.01)
        b = uncertain.forecast([1.], observation_sd=1.)
        self.assertGreater(b['within_component_parameter_variance'],a['within_component_parameter_variance'])
        self.assertGreater(b['component_observation_variance'],a['component_observation_variance'])
        self.assertGreater(np.exp(a['log_weights'])@a['location'],
                           np.exp(b['log_weights'])@b['location'])
        self.assertAlmostEqual(a['component_observation_variance'],
                               a['within_component_parameter_variance']+.1**2+.01**2)

    def test_units_and_frozen_predictions(self):
        model = MagnitudeRegression([.5], [[.3]], residual_sd=.1, max_paths=32)
        factor = 17.
        scaled = MagnitudeRegression([.5*factor], [[.3*factor**2]],
                                     residual_sd=.1*factor, max_paths=32)
        issued = model.forecast([1.], observation_sd=.2)
        frozen = copy.deepcopy(issued)
        for value in [1., 0., .3, .8]:
            a = model.observe([1.], value, observation_sd=.2)
            b = scaled.observe([1.], value*factor, observation_sd=.2*factor)
            self.assertAlmostEqual(a['filter_log_density'], b['filter_log_density']+math.log(factor),places=12)
        a = model.forecast([1.],observation_sd=.2)
        b = scaled.forecast([1.],observation_sd=.2*factor)
        points = np.array([0., .1, .5, 2.])
        np.testing.assert_allclose(cdf(a, points),cdf(b, points*factor),atol=1e-13)
        for key in issued:
            np.testing.assert_array_equal(issued[key], frozen[key])

    def test_residual_and_observation_variance_are_not_separately_identified(self):
        first = MagnitudeRegression([.5], [[.3]], residual_sd=.3, max_paths=32)
        second = MagnitudeRegression([.5], [[.3]], residual_sd=.4, max_paths=32)
        for value in [1., 0., .3, .8]:
            a = first.observe([1.],value,observation_sd=.4)
            b = second.observe([1.],value,observation_sd=.3)
            self.assertAlmostEqual(a['filter_log_density'], b['filter_log_density'], places=14)
        np.testing.assert_array_equal(first.precision,second.precision)
        np.testing.assert_array_equal(first.information,second.information)
        np.testing.assert_array_equal(first.log_weights,second.log_weights)
        a = first.forecast([1.],observation_sd=.4)
        b = second.forecast([1.],observation_sd=.3)
        self.assertNotEqual(a['residual_variance'],b['residual_variance'])
        np.testing.assert_array_equal(cdf(a,[0.,.5,1.,2.]),cdf(b,[0.,.5,1.,2.]))

    def test_pruning_reports_lost_mass_and_zero_merges_identical_branches(self):
        small = MagnitudeRegression([0.], [[1.]], residual_sd=.1, max_paths=1)
        full = MagnitudeRegression([0.], [[1.]], residual_sd=.1, max_paths=8)
        full.observe([1.],1.,observation_sd=.1)
        result = small.observe([1.],1.,observation_sd=.1)
        self.assertAlmostEqual(result['pruned_mass'],.5)
        zero = MagnitudeRegression([0.], [[1.]], residual_sd=.1, max_paths=1)
        result = zero.observe([1.],0.,observation_sd=.1)
        self.assertEqual(result['merged_paths'],1)
        self.assertAlmostEqual(result['pruned_mass'],0.)
        self.assertEqual(small.completed,full.completed)

    def test_missing_input_and_invalid_values_do_not_learn_silence(self):
        model = MagnitudeRegression([0.], [[1.]], residual_sd=0., max_paths=8)
        original = copy.deepcopy(model)
        for value, noise in [(-1.,.1),(float('nan'),.1),(0.,-1.),(0.,0.)]:
            with self.assertRaises(ValueError): model.observe([1.],value,observation_sd=noise)
        with self.assertRaises(ValueError): model.forecast([],observation_sd=.1)
        np.testing.assert_array_equal(model.precision,original.precision)
        self.assertEqual(model.completed,0)
        for covariance in [[[0.]], [[-1.]]]:
            with self.assertRaises(ValueError):
                MagnitudeRegression([0.],covariance,residual_sd=.1,max_paths=8)


if __name__ == '__main__':
    unittest.main()
