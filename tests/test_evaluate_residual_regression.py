import copy
import itertools
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln, logsumexp
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from residual_regression import ResidualRegression, bounded_crps, cdf, log_density


class ResidualRegressionTests(unittest.TestCase):
    def test_signed_sequential_updates_match_independent_batch_integrals(self):
        x = np.array([[1., .2], [1., -.5], [.3, .7], [.2, -.1]])
        offsets, values = np.array([.1, .4, .2, .3]), np.array([.7, .3, .2, .6])
        prior_mean = np.array([.2, -.1])
        relative = np.array([[.4, .05], [.05, .3]])
        for shape in (1.2, 3., 10.):
            model = ResidualRegression(prior_mean, relative, prior_shape=shape,
                                       prior_residual_variance=.02, max_paths=65536)
            b0, previous = (shape-1)*.02, 0.
            for i, value in enumerate(values):
                forecast = model.forecast(x[i], offset=offsets[i])
                update = model.observe(x[i], value, offset=offsets[i])
                n = i+1
                design = x[:n]
                response = np.array(list(itertools.product((-1., 1.), repeat=n)))*values[:n]-offsets[:n]
                centered = response-design @ prior_mean
                kernel = np.eye(n)+design @ relative @ design.T
                bn = b0+.5*np.einsum('ij,ji->i', centered, np.linalg.solve(kernel, centered.T))
                scores = (gammaln(shape+n/2)-gammaln(shape)-n/2*math.log(2*math.pi)
                          -.5*np.linalg.slogdet(kernel)[1]+shape*math.log(b0)-(shape+n/2)*np.log(bn))
                total = float(logsumexp(scores))
                self.assertAlmostEqual(update['filter_log_density'], total-previous, places=11)
                self.assertAlmostEqual(log_density(forecast, value), total-previous, places=11)
                self.assertLess(update['pruned_mass'], 1e-12)
                precision = np.linalg.inv(relative)+design.T @ design
                expected_cov = np.linalg.inv(precision)
                expected_mean = np.linalg.solve(precision, np.linalg.solve(relative, prior_mean)[:, None]
                                                + design.T @ response.T).T
                order = np.argsort(-(scores-total), kind='stable')
                np.testing.assert_allclose(model.mean, expected_mean[order], atol=1e-13)
                np.testing.assert_allclose(model.scale, bn[order], atol=1e-13)
                np.testing.assert_allclose(model.log_weights, (scores-total)[order], atol=1e-12)
                np.testing.assert_allclose(model.relative_covariance, expected_cov, atol=1e-14)
                previous = total

    def test_continuation_control_matches_initial_coefficient_predictive_distribution(self):
        full = ResidualRegression(np.zeros(4), .25*np.eye(4), prior_shape=3.,
                                  prior_residual_variance=.02, max_paths=128)
        control = ResidualRegression([], np.zeros((0, 0)), prior_shape=3.,
                                     prior_residual_variance=.03, max_paths=128)
        a = full.forecast([.8, .6, .6, -.8], offset=.5)
        b = control.forecast([], offset=.5)
        np.testing.assert_allclose(a['signed_component_variance'], [.03], atol=1e-15)
        np.testing.assert_allclose(a['within_component_parameter_variance'], [.01], atol=1e-15)
        np.testing.assert_allclose(a['scale2'], b['scale2'], atol=1e-15)
        np.testing.assert_allclose(cdf(a, [0., .2, .5, 1., 5.]), cdf(b, [0., .2, .5, 1., 5.]), atol=1e-15)

    def test_zero_observations_reduce_variance_and_later_error_reopens_uncertainty(self):
        model = ResidualRegression([], np.zeros((0, 0)), prior_shape=3.,
                                   prior_residual_variance=.03, max_paths=128)
        for n in range(1, 1025):
            model.observe([], 0., offset=0.)
            expected = .06/(2+n/2)
            np.testing.assert_allclose(model.forecast([], offset=0.)['signed_component_variance'], [expected])
            self.assertEqual(len(model.log_weights), 1)
        before = model.forecast([], offset=0.)['signed_component_variance'][0]
        self.assertLess(before, .00012)
        model.observe([], 1., offset=0.)
        after = model.forecast([], offset=0.)['signed_component_variance'][0]
        self.assertGreater(after, before*9)
        self.assertLess(after, .002)
        self.assertEqual(len(model.log_weights), 1)

    def test_units_zero_continuity_and_immutable_issuance(self):
        scale = 7.
        a = ResidualRegression([0., 0.], .25*np.eye(2), prior_shape=3.,
                               prior_residual_variance=.02, max_paths=4096)
        b = ResidualRegression([0., 0.], .25*np.eye(2), prior_shape=3.,
                               prior_residual_variance=.02*scale**2, max_paths=4096)
        for value in [.6, .3, .4]:
            left = a.observe([.8, .6], value, offset=.5)
            right = b.observe([.8, .6], value*scale, offset=.5*scale)
            self.assertAlmostEqual(left['filter_log_density']-math.log(scale), right['filter_log_density'], places=12)
        issued = a.forecast([.6, .8], offset=.4)
        saved = copy.deepcopy(issued)
        points = np.array([0., .2, .5, 1., 2.])
        np.testing.assert_allclose(cdf(issued, points), cdf(b.forecast([.6, .8], offset=.4*scale), points*scale), atol=1e-13)
        zero = copy.deepcopy(a)
        zero.observe([.8, .6], 0., offset=.5)
        for tiny in (1e-8, 1e-24, 1e-48):
            other = copy.deepcopy(a)
            other.observe([.8, .6], tiny, offset=.5)
            np.testing.assert_allclose(cdf(other.forecast([.6, .8], offset=.4), points),
                                       cdf(zero.forecast([.6, .8], offset=.4), points), atol=1e-10)
        a.observe([.8, .6], 0., offset=.5)
        for key in issued: np.testing.assert_array_equal(issued[key], saved[key])

    def test_folded_density_cdf_and_variance_follow_student_scale_convention(self):
        model = ResidualRegression([.2], [[.4]], prior_shape=1.2,
                                   prior_residual_variance=.03, max_paths=128)
        forecast = model.forecast([1.], offset=.1)
        mu, scale, df = forecast['location'][0], np.sqrt(forecast['scale2'][0]), forecast['degrees_freedom']
        self.assertAlmostEqual(forecast['signed_component_variance'][0], scale**2*df/(df-2))
        for value in (0., 1e-8, .1, 1., 10.):
            expected = t.pdf(value, df, loc=mu, scale=scale)+t.pdf(-value, df, loc=mu, scale=scale)
            self.assertAlmostEqual(math.exp(log_density(forecast, value)), expected, places=11)
            integrated = quad(lambda y: t.pdf(y, df, loc=mu, scale=scale)
                              + t.pdf(-y, df, loc=mu, scale=scale), 0., value, epsabs=1e-12)[0]
            self.assertAlmostEqual(float(cdf(forecast, value)), integrated, places=11)
            self.assertAlmostEqual(float(cdf(forecast, value)+cdf(forecast, value, survival=True)), 1., places=14)

    def test_bounded_score_matches_independent_integrals_with_heavy_and_narrow_tails(self):
        cases = [(2.01, [0.], [1e-8], [1.], 0.),
                 (2.4, [.3], [.04], [1.], .2),
                 (6., [-.8, .4, 3.], [.03, .01, 2.], [.3, .4, .3], 1.),
                 (600., [1.], [1e-16], [1.], 0.),
                 (3., [-2., .01], [1e-5, .2], [.02, .98], 1e-24)]
        for df, location, variance, weights, observed in cases:
            forecast = dict(location=np.array(location), scale2=np.array(variance),
                            log_weights=np.log(weights), degrees_freedom=df)
            for scale in (.25, 1., 4.):
                at = observed/(observed+scale)
                def integrand(u, survival):
                    y = scale*u/(1-u)
                    mu, sd = np.array(location), np.sqrt(variance)
                    mass = (t.cdf(-y, df, loc=mu, scale=sd)+t.sf(y, df, loc=mu, scale=sd)
                            if survival else t.cdf(y, df, loc=mu, scale=sd)-t.cdf(-y, df, loc=mu, scale=sd))
                    return float(mass @ weights)**2
                landmarks = np.maximum(0., abs(np.array(location))[:, None]
                                       + np.sqrt(variance)[:, None]*np.array([-100., -8., -1., 0., 1., 8., 100.]))
                edges = np.unique(np.r_[0., at, 1., (landmarks/(landmarks+scale)).ravel()])
                reference = sum(quad(lambda u: integrand(u, right > at), left, right,
                                     epsabs=1e-10, epsrel=1e-10, limit=300)[0]
                                for left, right in zip(edges[:-1], edges[1:], strict=True))
                primary = bounded_crps(forecast, observed, scale=scale)
                refined = bounded_crps(forecast, observed, scale=scale, order=32)
                self.assertAlmostEqual(primary, reference, delta=1e-4)
                self.assertAlmostEqual(refined, reference, delta=2e-6)

    def test_variance_changes_expose_stationarity_and_covariance_stays_positive(self):
        model = ResidualRegression([], np.zeros((0, 0)), prior_shape=3.,
                                   prior_residual_variance=.03, max_paths=128)
        for _ in range(256): model.observe([], .01, offset=0.)
        narrow = float(model.forecast([], offset=0.)['residual_variance_mean'][0])
        for _ in range(16): model.observe([], 1., offset=0.)
        widened = float(model.forecast([], offset=0.)['residual_variance_mean'][0])
        self.assertGreater(widened, narrow*50)
        self.assertLess(widened, .1)
        for _ in range(16): model.observe([], .01, offset=0.)
        self.assertGreater(float(model.forecast([], offset=0.)['residual_variance_mean'][0]), narrow*50)
        rng = np.random.default_rng(192071)
        designs = rng.normal(size=(512, 5))
        designs /= np.linalg.norm(designs, axis=1)[:, None]
        model = ResidualRegression(np.zeros(5), .25*np.eye(5), prior_shape=3.,
                                   prior_residual_variance=.02, max_paths=4)
        for x in designs: model.observe(x, .2, offset=.2)
        self.assertGreater(np.linalg.eigvalsh(model.relative_covariance).min(), 0.)
        np.testing.assert_allclose(model.relative_covariance, np.linalg.inv(4*np.eye(5)+designs.T@designs), atol=1e-14)


if __name__ == '__main__': unittest.main()
