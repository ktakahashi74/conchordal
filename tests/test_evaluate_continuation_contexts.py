import copy
import itertools
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from continuation_contexts import ContinuationContexts
from uncertain_regression import cdf, log_density


def batch_evidence(own, relation, offsets, values, times, *, shared, mode, contexts, sd):
    """Integrate all labelled contexts and signs using observation covariance."""
    n = len(values)
    if mode == 'recurrent':
        assignments = itertools.product(range(contexts), repeat=n)
    elif mode == 'renewal':
        assignments = (np.r_[0, np.cumsum(bits)]
                       for bits in itertools.product((0, 1), repeat=n-1))
    else:
        assignments = [np.zeros(n, dtype=int)]
    signed = np.array(list(itertools.product((-1., 1.), repeat=n))) * values - offsets
    scores = []
    for states in assignments:
        states = np.asarray(states)
        same = states[:, None] == states[None, :]
        covariance = (.02*np.eye(n) + sd**2/2 * (
            (own @ own.T) * (1 if shared else same) + (relation @ relation.T) * same))
        weight = -math.log(contexts) if mode == 'recurrent' else 0.
        for i, elapsed in enumerate(np.diff(times)):
            if mode == 'recurrent':
                survival = math.exp(-.4*contexts/(contexts-1)*elapsed)
                probability = (1-survival)/contexts + (survival if states[i] == states[i+1] else 0.)
                weight += math.log(probability)
            elif mode == 'renewal':
                weight += (-.4*elapsed if states[i] == states[i+1]
                           else math.log(-math.expm1(-.4*elapsed)))
        logdet = np.linalg.slogdet(covariance)[1]
        quadratic = np.einsum('ij,ji->i', signed, np.linalg.solve(covariance, signed.T))
        scores.append(weight + logsumexp(-.5*(n*math.log(2*math.pi)+logdet+quadratic)))
    return float(logsumexp(scores))


class ContinuationContextTests(unittest.TestCase):
    def model(self, shared=True, mode='recurrent', sd=.3, max_paths=65536, contexts=3):
        return ContinuationContexts(2, 2, mode=mode, shared_continuation=shared,
                                    correction_sd=sd, residual_sd=.1, observation_sd=.1,
                                    max_paths=max_paths, contexts=contexts, change_rate_hz=.4)

    def test_joint_context_and_sign_updates_match_independent_batch_marginals(self):
        own = np.array([[1., .4], [1., -.2], [1., .6], [1., .1]])
        relation = np.array([[1., -.1], [1., .7], [1., -.3], [1., .5]])
        own /= np.linalg.norm(own, axis=1)[:, None]
        relation /= np.linalg.norm(relation, axis=1)[:, None]
        offsets, values = np.array([.6, .8, .2, .5]), np.array([.7, 0., 1.2, .4])
        times = np.array([.2, .4, .9, 1.2])
        for shared, sd, (mode, contexts) in itertools.product(
                (False, True), (.03, .1, .3),
                (('stationary', 2), ('renewal', 2), ('recurrent', 2), ('recurrent', 3))):
            with self.subTest(shared=shared, sd=sd, mode=mode, contexts=contexts):
                model = self.model(shared, mode, sd, contexts=contexts)
                previous = 0.
                for i, value in enumerate(values):
                    forecast = model.forecast(times[i], own[i], relation[i], offsets[i])
                    update = model.observe(times[i], own[i], relation[i], offsets[i], value)
                    total = batch_evidence(own[:i+1], relation[:i+1], offsets[:i+1],
                                           values[:i+1], times[:i+1], shared=shared,
                                           mode=mode, contexts=contexts, sd=sd)
                    self.assertAlmostEqual(update['filter_log_density'], total-previous, places=10)
                    self.assertAlmostEqual(log_density(forecast, value), total-previous, places=10)
                    self.assertLess(update['pruned_mass'], 1e-12)
                    previous = total

    def test_renewal_preserves_shared_marginal_and_replaces_only_local_block(self):
        for shared in (False, True):
            model = self.model(shared, 'renewal')
            model.last_target_sec = model.clock_sec = 1.
            model.used[0] = 1
            model.mean[0] = [.2, -.1, .4, -.3]
            matrix = np.array([[1., .2, .3, 0.], [.2, 1., -.1, .2],
                               [.3, -.1, 1., .4], [0., .2, .4, 1.]])
            model.cov[0] = matrix @ matrix.T * .01
            before = copy.deepcopy(model)
            c = model._components(2., [.8, .6], [.6, -.8], .5)
            reset = np.flatnonzero(c['reset'])[0]
            if shared:
                np.testing.assert_array_equal(c['mean'][reset, :2], before.mean[0, :2])
                np.testing.assert_array_equal(c['cov'][reset, :2, :2], before.cov[0, :2, :2])
                np.testing.assert_array_equal(c['cov'][reset, :2, 2:], 0.)
                np.testing.assert_array_equal(c['cov'][reset, 2:, :2], 0.)
                np.testing.assert_array_equal(c['mean'][reset, 2:], 0.)
                np.testing.assert_array_equal(c['cov'][reset, 2:, 2:], np.eye(2)*.045)
            else:
                np.testing.assert_array_equal(c['mean'][reset], 0.)
                np.testing.assert_array_equal(c['cov'][reset], np.eye(4)*.045)
            np.testing.assert_array_equal(model.mean, before.mean)
            np.testing.assert_array_equal(model.cov, before.cov)

    def test_stationary_layouts_agree_and_shared_inactive_relations_remain_correlated(self):
        a, b = self.model(False, 'stationary'), self.model(True, 'stationary')
        for i, y in enumerate([.4, .6, 0., 1.1, .3]):
            args = (i+1., [.8, .6], [.6, -.8], .5)
            self.assertEqual(a.observe(*args, y), b.observe(*args, y))
            np.testing.assert_array_equal(a.mean, b.mean)
            np.testing.assert_array_equal(a.cov, b.cov)
        model = self.model(contexts=2)
        model.observe(1., [.8, .6], [.6, -.8], .5, .7)
        model.observe(2., [.6, .8], [.8, .6], .6, .3)
        # A path that has visited both contexts couples their coefficients through the shared block.
        visited = np.flatnonzero(model.used == 2)
        self.assertGreater(len(visited), 0)
        self.assertGreater(np.max(abs(model.cov[visited, 2:4, 4:6])), 1e-5)
        self.assertGreater(np.max(abs(model.cov[visited, :2, 4:6])), 1e-5)

    def test_offset_removes_sign_symmetry_and_zero_updates_are_continuous(self):
        for shared, mode in itertools.product((False, True), ('stationary', 'renewal', 'recurrent')):
            model = self.model(shared, mode)
            for i, y in enumerate([.6, .3]):
                model.observe(i+1., [.8, .6], [.6, -.8], .5, y)
            flipped = copy.deepcopy(model)
            flipped.mean *= -1
            args = (3., [.8, .6], [.6, -.8], .5)
            points = [0., .2, .5, 1., 2.]
            self.assertGreater(np.max(abs(cdf(model.forecast(*args), points)
                                          - cdf(flipped.forecast(*args), points))), .01)
            zero = copy.deepcopy(model)
            zero.observe(*args, 0.)
            for tiny in (1e-8, 1e-24, 1e-48):
                other = copy.deepcopy(model)
                other.observe(*args, tiny)
                future = (4., [.6, .8], [.8, .6], .4)
                np.testing.assert_allclose(cdf(other.forecast(*future), points),
                                           cdf(zero.forecast(*future), points), atol=1e-10)

    def test_units_prior_width_immutable_queries_and_missing_input(self):
        model = self.model()
        scale = 7.
        scaled = ContinuationContexts(2, 2, mode='recurrent', shared_continuation=True,
                                      correction_sd=.3*scale, residual_sd=.1*scale,
                                      observation_sd=.1*scale, max_paths=65536, change_rate_hz=.4)
        initial = model.forecast(1., [.8, .6], [.6, -.8], .5)
        np.testing.assert_allclose(initial['within_component_parameter_variance'], .3**2)
        np.testing.assert_allclose(initial['component_observation_variance'], .3**2+.02)
        for i, y in enumerate([.6, 0., .3]):
            a = model.observe(i+1., [.8, .6], [.6, -.8], .5, y)
            b = scaled.observe(i+1., [.8, .6], [.6, -.8], .5*scale, y*scale)
            self.assertAlmostEqual(a['filter_log_density']-math.log(scale), b['filter_log_density'], places=11)
        issued = model.forecast(6., [.8, .6], [.6, -.8], .5)
        saved, before = copy.deepcopy(issued), copy.deepcopy(model)
        model.skip_to(4.)
        for name in ('mean', 'cov', 'active', 'used', 'log_weights'):
            np.testing.assert_array_equal(getattr(model, name), getattr(before, name))
        points = np.array([0., .2, .5, 1., 2.])
        np.testing.assert_allclose(cdf(issued, points), cdf(scaled.forecast(
            6., [.8, .6], [.6, -.8], .5*scale), points*scale), atol=1e-13)
        for name in issued:
            np.testing.assert_array_equal(model.forecast(6., [.8, .6], [.6, -.8], .5)[name], saved[name])
        model.observe(5., [.8, .6], [.6, -.8], .5, .4)
        for name in issued:
            np.testing.assert_array_equal(issued[name], saved[name])
        with self.assertRaises(ValueError): model.forecast(5., [.8, .6], [.6, -.8], .5)

    def test_path_budget_reports_actual_discarded_probability(self):
        full, small = self.model(max_paths=128), self.model(max_paths=1)
        args = (1., [.8, .6], [.6, -.8], .1, .2)
        full.observe(*args)
        update = small.observe(*args)
        self.assertEqual(len(full.log_weights), 2)
        self.assertAlmostEqual(update['pruned_mass'], 1-np.exp(full.log_weights).max(), places=13)
        self.assertGreater(update['pruned_mass'], .01)


if __name__ == '__main__': unittest.main()
