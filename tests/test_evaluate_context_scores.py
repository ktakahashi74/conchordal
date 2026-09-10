import copy
import gzip
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_context_scores import bounded_cdf, bounded_crps, run
from relational_contexts import RegressionContexts, log_density


class BoundedContextScoreTests(unittest.TestCase):
    def setUp(self):
        self.forecast = dict(location=[-.7, 1.2], scale2=[.2, 1.3], degrees_freedom=[4., 21.],
                             positive_probability=[.8, .6], log_weights=np.log([.3, .7]).tolist())

    def test_cdf_matches_independent_student_density_integrals_and_zero_atom(self):
        f = self.forecast
        points = np.array([0., .01, .2, .5, .9, 1.])
        for scale in (.25, 1., 4.):
            actual = bounded_cdf(f, points, scale=scale)
            expected = []
            for u in points:
                total = 0.
                for location, variance, degree, positive, log_weight in zip(*f.values(), strict=True):
                    if u in (0., 1.):
                        cdf = float(u)
                    else:
                        end = (math.log(u/(1-u))+math.log(scale)-location)/math.sqrt(variance)
                        norm = math.gamma((degree+1)/2)/(math.gamma(degree/2)*math.sqrt(degree*math.pi))
                        cdf = quad(lambda t: norm*(1+t*t/degree)**(-(degree+1)/2),
                                   -np.inf, end, epsabs=1e-12)[0]
                    total += math.exp(log_weight)*(1-positive+positive*cdf)
                expected.append(1. if u == 1. else total)
            np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-12)
            np.testing.assert_allclose(actual+bounded_cdf(f, points, scale=scale, survival=True), 1., atol=1e-14)
            self.assertTrue(np.all(np.diff(actual) >= 0))

    def test_loss_matches_adaptive_integration_including_silence_and_tails(self):
        for value in (0., 1e-100, .3, 3., 1e100):
            for scale in (.25, 1., 4.):
                at = value/(value+scale)
                reference = (quad(lambda u: float(bounded_cdf(self.forecast, u, scale=scale))**2,
                                  0., at, epsabs=1e-11)[0]
                             + quad(lambda u: float(bounded_cdf(self.forecast, u, scale=scale, survival=True))**2,
                                    at, 1., epsabs=1e-11)[0])
                estimate = bounded_crps(self.forecast, value, scale=scale, order=128)
                self.assertAlmostEqual(estimate, reference, delta=2e-6)
                self.assertTrue(0 <= estimate <= 1)

    def test_zero_forecast_has_exact_absolute_error_and_tiny_targets_are_continuous(self):
        f = dict(self.forecast, positive_probability=[0., 0.])
        for value in (0., .2, 3.):
            self.assertAlmostEqual(bounded_crps(f, value), value/(1+value), places=14)
        zero = bounded_crps(self.forecast, 0.)
        for value in (1e-10, 1e-50, 1e-300):
            self.assertLessEqual(abs(bounded_crps(self.forecast, value)-zero), value+1e-14)

    def test_narrow_and_broad_distributions_match_resolved_reference(self):
        for variance in (1e-3, 1e-5, 1e-9, 50.):
            f = dict(location=[0.], scale2=[variance], degrees_freedom=[8.],
                     positive_probability=[1.], log_weights=[0.])
            at = 1/6
            points = [float(expit(z*math.sqrt(variance))) for z in (-8, -2, 0, 2, 8)]
            reference = (quad(lambda u: float(bounded_cdf(f, u))**2, 0., at, epsabs=1e-12)[0]
                         + quad(lambda u: float(bounded_cdf(f, u, survival=True))**2, at, 1.,
                                points=[p for p in points if at < p < 1], epsabs=1e-12)[0])
            self.assertAlmostEqual(bounded_crps(f, .2), reference, delta=5e-5)
            self.assertAlmostEqual(bounded_crps(f, .2, order=64), reference, delta=1e-6)

    def test_units_and_component_splitting_do_not_change_the_distribution_score(self):
        expected = bounded_crps(self.forecast, .7, scale=.25, order=64)
        for factor in (1e-50, 1e50):
            scaled = dict(self.forecast, location=(np.array(self.forecast['location'])+math.log(factor)).tolist())
            self.assertAlmostEqual(bounded_crps(scaled, .7*factor, scale=.25*factor, order=64), expected, places=13)
        split = {k: np.repeat(v, 2).tolist() for k, v in self.forecast.items()}
        split['log_weights'] = (np.array(split['log_weights'])-math.log(2)).tolist()
        self.assertAlmostEqual(bounded_crps(split, .7, scale=.25, order=64), expected, places=14)
        reordered = {k: list(reversed(v)) for k, v in self.forecast.items()}
        self.assertAlmostEqual(bounded_crps(reordered, .7, scale=.25, order=64), expected, places=14)

    def test_invalid_mixtures_and_targets_are_rejected(self):
        for fields in ({'scale2': [-1., 1.]}, {'degrees_freedom': [0., 1.]},
                       {'positive_probability': [1.1, .5]}, {'log_weights': [0., 0.]},
                       {'location': [0.]}, {'location': [float('nan'), 1.]}):
            with self.assertRaises(ValueError):
                bounded_cdf(dict(self.forecast, **fields), [.5])
        for value in (-1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                bounded_crps(self.forecast, value)
        with self.assertRaises(ValueError): bounded_crps(self.forecast, 0., scale=0.)
        with self.assertRaises(ValueError): bounded_crps(self.forecast, 0., order=1)

    def test_stream_uses_frozen_forecasts_and_preserves_censoring_and_eof(self):
        m = RegressionContexts(26, mode='stationary')
        x = [1/3, *[0.]*25]
        m.observe(.1, x, 1.)
        issued = m.forecast(1., x)
        m.observe(.2, x, 5.)
        self.assertNotAlmostEqual(log_density(issued, .5), log_density(m.forecast(1., x), .5))
        query = dict(kind='issued', query_id=0, issued_sample=2, target_sample=10,
                     features=dict(background=[x]), forecasts=dict(both_stationary=[issued]))
        rows = [dict(kind='contract', native_contract=dict(sample_rate=10)), query,
                dict(copy.deepcopy(query), query_id=1, target_sample=20),
                dict(kind='completed', query_id=0, target_sample=10, observed=[.5],
                     issued_log_density=dict(both_stationary=[log_density(issued, .5)])),
                dict(copy.deepcopy(query), query_id=2, target_sample=30),
                dict(kind='censored', query_id=2, target_sample=30, reason='input_gap'),
                dict(kind='eof', pending=1, completed=1)]
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'input.gz', Path(directory)/'output.gz'
            with gzip.open(source, 'wt') as target:
                for row in rows:
                    target.write(json.dumps(row, default=lambda a: a.tolist())+'\n')
            original = source.read_bytes()
            run(source, output)
            self.assertEqual(source.read_bytes(), original)
            with gzip.open(output, 'rt') as result:
                scored = [json.loads(line) for line in result]
            self.assertEqual(scored[-1], dict(kind='eof', issued=3, completed=1, censored=1,
                                              pending=1, pending_targets=[20]))
            self.assertEqual(scored[-2], rows[-2])
            row = scored[1]
            self.assertEqual(row['persistence'], [1.])
            self.assertEqual(row['issued_log_density'], rows[3]['issued_log_density'])
            for score in row['scores']:
                s = score['scale']
                self.assertAlmostEqual(score['loss']['persistence'][0], abs(1/(1+s)-.5/(.5+s)))
                self.assertAlmostEqual(score['loss']['both_stationary'][0], bounded_crps(issued, .5, scale=s))


if __name__ == '__main__':
    unittest.main()
