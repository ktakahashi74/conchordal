from pathlib import Path
import copy
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from acoustic_body import DrivenBodyPosterior
from evaluate_continuous_acoustic_state import stream_predictions


class ContinuousAcousticEvidenceTests(unittest.TestCase):
    def test_request_cadence_and_repeated_targets_do_not_train_state(self):
        initial = DrivenBodyPosterior(24000, [317., 820.], [-3., 0.], drive_per_sec=100.,
                                      initial_precision=.1, variance_prior=(2., .004))
        audio = np.random.default_rng(192103).normal(0., .1, 420)
        sparse = [(100, 150), (300, 150)]
        dense = sorted(sparse+[(n, 150) for n in range(0, 420, 10)]+sparse)
        outputs, models = [], []
        for queries, ends in [(sparse, [420]), (dense, [31, 100, 201, 309, 420]), ([], [17, 420])]:
            model = copy.deepcopy(initial)
            starts = [0]+ends[:-1]
            rows = list(stream_predictions(model, ((a,b,audio[a:b]) for a,b in zip(starts,ends)), queries))
            outputs.append(rows)
            models.append(model)
            self.assertEqual(rows[-1]['observed_samples'], 420)
            self.assertEqual(sum(r['end_sample']-r['start_sample'] for r in rows if r['kind']=='observed'), 420)
            self.assertAlmostEqual(sum(r['conditional_log_density'] for r in rows if r['kind']=='observed'),
                                   model.log_evidence, places=10)
            for r in rows:
                if r['kind']=='issued':
                    self.assertEqual(r['observed_samples'], r['issued_sample'])
                    self.assertEqual(r['forecast'].digest(), r['issued_sha256'])
                if r['kind']=='completed':
                    issued = next(x for x in rows if x['kind']=='issued' and x['query_id']==r['query_id'])
                    self.assertEqual(r['issued_sha256'], issued['issued_sha256'])
                    self.assertAlmostEqual(r['joint_log_density'], issued['forecast'].log_density(
                        audio[r['issued_sample']:r['target_sample']]), places=12)
        for model in models[1:]:
            np.testing.assert_array_equal(model.state_mean, models[0].state_mean)
            np.testing.assert_array_equal(model.state_covariance, models[0].state_covariance)
            self.assertAlmostEqual(model.variance_scale, models[0].variance_scale, places=10)
            self.assertEqual(model.variance_shape, models[0].variance_shape)
        for r in outputs[0]:
            if r['kind']=='issued':
                for other in outputs[1]:
                    if other['kind']=='issued' and other['issued_sample']==r['issued_sample']:
                        np.testing.assert_array_equal(r['forecast'].mean, other['forecast'].mean)
                        np.testing.assert_allclose(r['forecast'].diagonal_variance(),
                                                   other['forecast'].diagonal_variance(), rtol=1e-13)
        self.assertEqual(outputs[0][-1]['completed'], 1)
        self.assertEqual(outputs[0][-1]['pending_targets'], [450])

    def test_gap_censors_targets_without_observations_or_eof_padding(self):
        initial = DrivenBodyPosterior(24000, [417.], [-2.], drive_per_sec=100.,
                                      initial_precision=.1, variance_prior=(2., .004))
        audio = np.random.default_rng(192107).normal(0., .1, 120)
        chunks = [(0,40,audio[:40]), (40,80,None), (80,120,audio[80:])]
        queries = [(20,20), (30,50), (50,10), (80,20), (110,20), (150,10)]
        model = copy.deepcopy(initial)
        rows = list(stream_predictions(model, chunks, queries))
        self.assertEqual(rows[-1], dict(kind='eof', state_sample=120, observed_samples=80,
            last_observed_sample=120, completed=2, censored=2, pending_targets=[130], unissued_queries=1))
        in_gap = next(r for r in rows if r['kind']=='issued' and r['issued_sample']==50)
        self.assertEqual(in_gap['last_observed_sample'], 40)
        self.assertEqual(in_gap['state_sample'], 50)
        self.assertEqual(in_gap['observed_samples'], 40)
        direct = copy.deepcopy(initial)
        direct.observe(0,audio[:40]); direct.advance_to(80); direct.observe(80,audio[80:])
        np.testing.assert_allclose(model.state_mean, direct.state_mean, atol=1e-13)
        np.testing.assert_allclose(model.state_covariance, direct.state_covariance, rtol=1e-12)
        self.assertAlmostEqual(model.variance_scale, direct.variance_scale, places=12)
        for bad_chunks, bad_queries in [([(1,40,audio[:39])], []), ([(0,40,audio[:39])], []),
                                         (chunks, [(10,0)]), (chunks, [(20,1),(10,1)])]:
            fresh = copy.deepcopy(initial)
            with self.assertRaises(ValueError):
                list(stream_predictions(fresh,bad_chunks,bad_queries))
            self.assertEqual(fresh.observed_samples, 0)
            self.assertEqual(fresh.next_sample, 0)
