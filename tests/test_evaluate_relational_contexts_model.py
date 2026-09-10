import copy
import itertools
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from relational_contexts import RegressionContexts, log_density


def batch_evidence(design, values):
    positive = values > 0
    x = np.column_stack([np.ones(positive.sum()), design[positive]])
    y = np.log(values[positive])
    prior = np.diag(np.r_[.01, np.ones(design.shape[1])])
    precision = prior + x.T @ x
    mean = np.linalg.solve(precision, x.T @ y)
    a = 2 + len(y)/2
    b = .5 + .5*(y@y-mean@precision@mean)
    regression = (.5*(np.linalg.slogdet(prior)[1]-np.linalg.slogdet(precision)[1])
                  + 2*math.log(.5)-a*math.log(b)+math.lgamma(a)-math.lgamma(2)
                  - len(y)/2*math.log(2*math.pi)-y.sum())
    zeros = len(values)-len(y)
    atom = (math.lgamma(.5+len(y))+math.lgamma(.5+zeros)
            - math.lgamma(1+len(values))-2*math.lgamma(.5))
    return regression+atom


class RelationalContextTests(unittest.TestCase):
    def test_recurrent_filter_matches_exhaustive_latent_assignments(self):
        values = np.array([.8, 1.2, 0., 3., 2., .7, .9])
        x = np.sin(np.arange(len(values))[:, None]*.7)
        for contexts in (2, 3):
            m = RegressionContexts(1, mode='recurrent', contexts=contexts, max_paths=512, change_rate_hz=.4)
            previous = 0.
            for i, value in enumerate(values):
                result = m.observe((i+1)*.2, x[i], value)
                evidence = []
                for assignment in itertools.product(range(contexts), repeat=i+1):
                    states = np.array(assignment)
                    retained = math.exp(-.4*contexts/(contexts-1)*.2)
                    stay = 1/contexts+(1-1/contexts)*retained
                    other = (1-retained)/contexts
                    score = -math.log(contexts)
                    score += sum(math.log(stay if a == b else other)
                                 for a, b in zip(states[:-1], states[1:]))
                    for context in range(contexts):
                        selected = states == context
                        score += batch_evidence(x[:i+1][selected], values[:i+1][selected])
                    evidence.append(score)
                total = float(np.logaddexp.reduce(evidence))
                self.assertAlmostEqual(result['filter_log_density'], total-previous, places=10)
                self.assertAlmostEqual(result['pruned_mass'], 0., places=13)
                previous = total

    def test_stationary_and_renewal_match_batch_segment_marginals(self):
        y = np.array([.8, 0., .9, 1.5, .2, .3])
        x = np.cos(np.arange(len(y))[:, None])
        for mode in ('stationary', 'renewal'):
            m = RegressionContexts(1, mode=mode, max_paths=128, change_rate_hz=.4)
            previous = 0.
            for i, value in enumerate(y):
                result = m.observe((i+1)*.2, x[i], value)
                scores = []
                partitions = (itertools.product((0, 1), repeat=i) if mode == 'renewal'
                              else [tuple(0 for _ in range(i))])
                for bits in partitions:
                    bounds = [0, *[j+1 for j, bit in enumerate(bits) if bit], i+1]
                    score = (sum(bits)*math.log(-math.expm1(-.08))-(i-sum(bits))*.08
                             if mode == 'renewal' else 0.)
                    score += sum(batch_evidence(x[a:b], y[a:b]) for a,b in zip(bounds[:-1],bounds[1:]))
                    scores.append(score)
                total = float(np.logaddexp.reduce(scores))
                self.assertAlmostEqual(result['filter_log_density'], total-previous, places=10)
                previous = total

    def test_transitions_compose_across_unobserved_time_and_rescale_units(self):
        m = RegressionContexts(0, mode='recurrent', change_rate_hz=.4)
        np.testing.assert_allclose(m.transition(.3)@m.transition(1.7), m.transition(2.), atol=1e-15)
        scaled = RegressionContexts(0, mode='recurrent', change_rate_hz=.04)
        np.testing.assert_allclose(m.transition(2.), scaled.transition(20.), atol=1e-15)
        m.observe(1., [], 1.)
        forecast = m.forecast(3., [])
        self.assertAlmostEqual(np.exp(forecast['log_weights']).sum(), 1.)
        self.assertAlmostEqual(np.exp(forecast['log_weights'])[0], m.transition(2.)[0,0])
        self.assertAlmostEqual(np.exp(forecast['log_weights'])[1], m.transition(2.)[0,1]*2)

    def test_inactive_relationship_parameters_survive_and_can_return(self):
        m = RegressionContexts(0, mode='recurrent', max_paths=1, prior_scale=.005)
        for i in range(40):
            m.observe((i+1)*.1, [], 1.)
        m.observe(4.1, [], 100.)
        self.assertEqual(m.active[0], 1)
        inactive = {n: getattr(m,n)[0,0].copy() for n in ('mean','cov','shape','scale','positive','zero')}
        for i in range(42,61):
            m.observe(i*.1, [], 100.)
        for name, original in inactive.items():
            np.testing.assert_array_equal(getattr(m,name)[0,0], original)
        result = m.observe(6.1, [], 1.)
        self.assertEqual(m.active[0], 0)
        self.assertGreater(result['returning_context_mass'], .9)

    def test_issued_score_stays_frozen_while_filter_uses_intervening_outcomes(self):
        m = RegressionContexts(0, mode='stationary')
        m.observe(.1, [], 1.)
        issued = m.forecast(1., [])
        saved = copy.deepcopy(issued)
        for i in range(2,10):
            m.observe(i*.1, [], 4.)
        result = m.observe(1., [], 4.)
        for name, value in saved.items():
            np.testing.assert_array_equal(issued[name], value)
        self.assertNotAlmostEqual(log_density(issued,4.), result['filter_log_density'], places=4)

    def test_missing_time_preserves_learning_but_silence_updates_zero_mass(self):
        m = RegressionContexts(0, mode='recurrent')
        for i in range(8):
            m.observe((i+1)*.1, [], 1.)
        saved = copy.deepcopy(m)
        m.skip_to(3.)
        for name in ('mean','cov','shape','scale','positive','zero','log_weights'):
            np.testing.assert_array_equal(getattr(m,name), getattr(saved,name))
        self.assertEqual(m.completed, saved.completed)
        a, b = m.forecast(3.1, []), saved.forecast(3.1, [])
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
        stable = RegressionContexts(0, mode='stationary')
        before = log_density(stable.forecast(.1, []),0.)
        stable.observe(.1, [], 0.)
        self.assertGreater(log_density(stable.forecast(.2, []),0.), before)
        self.assertEqual(stable.shape[0,0], 2.)

    def test_path_budget_reports_actual_discarded_mass(self):
        full = RegressionContexts(0, mode='recurrent', max_paths=100)
        small = RegressionContexts(0, mode='recurrent', max_paths=2)
        for i, y in enumerate([.9, 1.1]):
            full.observe(i+1., [], y); small.observe(i+1., [], y)
        full.observe(3., [], 1.)
        result = small.observe(3., [], 1.)
        lost = 1-sum(sorted(np.exp(full.log_weights),reverse=True)[:2])
        self.assertAlmostEqual(result['pruned_mass'], lost, places=13)

    def test_prior_density_support_and_invalid_observations(self):
        m = RegressionContexts(0, mode='stationary')
        forecast = m.forecast(.1, [])
        self.assertAlmostEqual(math.exp(log_density(forecast,0.)), .5)
        u = np.linspace(-1000, 1000, 200001)
        scale = math.sqrt(forecast['scale2'][0])
        t4 = math.gamma(2.5)/(math.gamma(2)*math.sqrt(4*math.pi)*scale)*(1+(u/scale)**2/4)**-2.5
        self.assertAlmostEqual(.5+.5*np.trapezoid(t4,u), 1., places=8)
        expected = .5*math.gamma(2.5)/(math.gamma(2)*math.sqrt(4*math.pi)*scale)
        self.assertAlmostEqual(math.exp(log_density(forecast,1.)), expected)
        for y in (-1., float('nan')):
            with self.assertRaises(ValueError): m.observe(.1, [], y)
        with self.assertRaises(ValueError): m.forecast(.1, [1.])


if __name__ == '__main__':
    unittest.main()
