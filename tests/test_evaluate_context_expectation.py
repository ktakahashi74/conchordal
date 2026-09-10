import copy
import gzip
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_context_expectation import ContextExpectation, envelope_frames, run
from evaluate_phrase_expectation import sha256


def model(lags=(1,), features=1, rate=.2, budget=64, step=.1):
    return ContextExpectation(features, lags, step, rate, budget,
                              np.zeros(len(lags) + 1), np.eye(len(lags) + 1), 2., .5)


def batch_log_evidence(x, y):
    precision = np.eye(x.shape[1]) + x.T @ x
    mean = np.linalg.solve(precision, x.T @ y)
    shape = 2 + len(y) / 2
    scale = .5 + .5 * (y @ y - mean @ precision @ mean)
    return (-.5 * np.linalg.slogdet(precision)[1] + 2 * math.log(.5)
            - shape * math.log(scale) + math.lgamma(shape) - math.lgamma(2)
            - len(y) / 2 * math.log(2 * math.pi))


class ContextExpectationTests(unittest.TestCase):
    def test_online_filter_matches_exhaustive_segment_marginal_likelihoods(self):
        source = np.array([.1, .2, -.3, .5, 1.2, .8, -.1, -.2])
        for lags in [(), (1,), (1, 2)]:
            observer = model(lags=lags, rate=1.5)
            warmup = max(lags, default=0)
            x = np.array([[1., *[source[i-lag] for lag in lags]]
                          for i in range(warmup, len(source))])
            y = source[warmup:]
            previous_log_evidence = 0.
            for i, value in enumerate(source):
                result = observer.observe((i + 1) * .1, [value])
                if i < warmup:
                    self.assertEqual(result["kind"], "warmup")
                    continue
                n = i + 1 - warmup
                by_start = {}
                for bits in itertools.product((0, 1), repeat=n - 1):
                    boundaries = [0, *[j + 1 for j, bit in enumerate(bits) if bit], n]
                    logp = sum(bits) * math.log(-math.expm1(-.15)) - (n - 1 - sum(bits)) * .15
                    for a, b in zip(boundaries, boundaries[1:]):
                        logp += batch_log_evidence(x[a:b], y[a:b])
                    start = boundaries[-2]
                    by_start[start] = np.logaddexp(by_start.get(start, -np.inf), logp)
                total = np.logaddexp.reduce(list(by_start.values()))
                self.assertAlmostEqual(result["log_density"][0], total - previous_log_evidence, places=11)
                previous_log_evidence = total
                actual = {round(t, 8): p for t, p in zip(observer.starts_sec[0], np.exp(observer.log_weights[0]))}
                for start, logp in by_start.items():
                    key = 0. if start == 0 else round((warmup + start) * .1, 8)
                    self.assertAlmostEqual(actual[key], math.exp(logp - total), places=11)

    def test_predictive_density_normalizes_and_can_exceed_one(self):
        observer = ContextExpectation(1, [], .1, 0., 1, [0.], [[1.]], 1., .0001)
        prediction = observer.forecast()
        # A t_2 density has an elementary CDF, independent of the update equations.
        scale = math.sqrt(prediction["scale2"][0, 0])
        x = np.linspace(-1000 * scale, 1000 * scale, 100_001)
        density = (1 / (2 * math.sqrt(2) * scale)) * (1 + (x / scale) ** 2 / 2) ** -1.5
        self.assertAlmostEqual(np.trapezoid(density, x), 1., places=5)
        result = observer.observe(.1, [0.])
        self.assertAlmostEqual(math.exp(result["log_density"][0]), 1 / (2 * math.sqrt(2) * scale))
        self.assertGreater(result["log_density"][0], 0.)

    def test_zero_atom_and_positive_density_match_exhaustive_evidence(self):
        source = np.array([0., .2, .4, 0., .5, 0., 0.])
        observer = ContextExpectation(1, [1], .1, 1.5, 64, [0., 0.], np.eye(2), 2., .5, [.5, .5])
        x, y = np.column_stack([np.ones(len(source)-1), source[:-1]]), source[1:]
        previous = 0.
        for i, value in enumerate(source):
            prediction = observer.forecast()
            result = observer.observe((i + 1) * .1, [value])
            if i == 0:
                continue
            evidence = []
            for bits in itertools.product((0, 1), repeat=i-1):
                boundaries = [0, *[j+1 for j, bit in enumerate(bits) if bit], i]
                logp = sum(bits) * math.log(-math.expm1(-.15)) - (i-1-sum(bits)) * .15
                for a, b in zip(boundaries, boundaries[1:]):
                    positive = y[a:b] > 0
                    count = int(positive.sum())
                    logp += (math.lgamma(.5 + count) + math.lgamma(.5 + b-a-count)
                             - math.lgamma(1 + b-a) - 2 * math.lgamma(.5))
                    logy = np.log(y[a:b][positive])
                    logp += batch_log_evidence(x[a:b][positive], logy) - logy.sum()
                evidence.append(logp)
            total = np.logaddexp.reduce(evidence)
            self.assertAlmostEqual(result["log_density"][0], total-previous, places=11)
            previous = total
            if value == 0:
                expected_zero = np.sum(np.exp(prediction["log_weights"])
                                       * (1-prediction["positive_probability"]))
                self.assertAlmostEqual(math.exp(result["log_density"][0]), expected_zero)

    def test_zero_atom_is_normalized_and_zeros_do_not_fit_log_regression(self):
        observer = ContextExpectation(1, [], .1, 0., 8, [0.], [[1.]], 1., .5, [2., 3.])
        prediction = observer.forecast()
        positive = prediction["positive_probability"][0, 0]
        log_values = np.linspace(-1000., 1000., 100_001)
        # Integrating f(log(y))/y in y equals integrating f(z) in z.
        scale = math.sqrt(prediction["scale2"][0, 0])
        density_log = 1 / (2*math.sqrt(2)*scale) * (1 + (log_values/scale)**2/2)**-1.5
        self.assertAlmostEqual(1-positive + positive*np.trapezoid(density_log, log_values), 1., places=5)
        before = {key: getattr(observer, key).copy() for key in ["mean", "cov", "shape", "scale"]}
        for i in range(20):
            observer.observe((i+1)*.1, [0.])
        for key, value in before.items():
            np.testing.assert_array_equal(getattr(observer, key), value)
        self.assertAlmostEqual(observer.forecast()["positive_probability"][0,0], 2/25)
        with self.assertRaises(ValueError):
            observer.observe(2.1, [-.1])

    def test_silence_learns_but_a_gap_does_not_invent_observations(self):
        observer = model(lags=())
        for i in range(30):
            observer.observe((i + 1) * .1, [1.])
        silent, missing = copy.deepcopy(observer), copy.deepcopy(observer)
        saved = {key: value.copy() for key, value in vars(missing).items() if isinstance(value, np.ndarray)}
        before = silent.forecast()
        for i in range(30, 60):
            silent.observe((i + 1) * .1, [0.])
        missing.observe(6., None)
        for key, value in saved.items():
            np.testing.assert_array_equal(getattr(missing, key), value)
        after = silent.forecast()
        self.assertLess(np.sum(np.exp(after["log_weights"]) * after["location"]), .1)
        self.assertGreater(np.sum(np.exp(before["log_weights"]) * before["location"]), .9)
        # Missing time changes the prior survival, not the learned sufficient statistics.
        weights = np.exp(missing.forecast()["log_weights"])
        self.assertAlmostEqual(weights[0, -1], -math.expm1(-.2 * 3.1))

    def test_gap_clears_conditioning_and_requires_observed_lag_refill(self):
        observer = model(lags=(1, 3))
        for i in range(6):
            observer.observe((i + 1) * .1, [i / 10])
        observer.observe(2., None)
        for i in range(3):
            self.assertIsNone(observer.forecast())
            self.assertEqual(observer.observe(2. + (i + 1) * .1, [.2])["kind"], "warmup")
        self.assertIsNotNone(observer.forecast())

    def test_independent_features_and_time_unit_scaling(self):
        values = np.random.default_rng(821).normal(size=(40, 3))
        joint = model(features=3, budget=4)
        singles = [model(budget=4) for _ in range(3)]
        slowed = model(features=3, budget=4, step=1., rate=.02)
        for i, row in enumerate(values):
            actual = joint.observe((i + 1) * .1, row)
            scaled = slowed.observe(i + 1., row)
            for j, single in enumerate(singles):
                expected = single.observe((i + 1) * .1, [row[j]])
                if actual["kind"] == "score":
                    self.assertAlmostEqual(actual["log_density"][j], expected["log_density"][0], places=12)
                    self.assertAlmostEqual(actual["log_density"][j], scaled["log_density"][j], places=12)

    def test_cross_feature_conditioning_matches_batch_regression_evidence(self):
        values = np.random.default_rng(9143).normal(size=(24,3))
        sources = np.array([[0,2],[1,0],[2,1]])
        observer = ContextExpectation(3,[1,3],.1,0.,1,np.zeros(5),np.eye(5),2.,.5,
                                      conditioning_sources=sources)
        previous = np.zeros(3)
        designs = [[] for _ in range(3)]
        for i,value in enumerate(values):
            result = observer.observe((i+1)*.1,value)
            if i < 3:
                continue
            for band in range(3):
                designs[band].append([1.,*[values[i-lag,source] for lag in (1,3) for source in sources[band]]])
                evidence = batch_log_evidence(np.array(designs[band]),values[3:i+1,band])
                self.assertAlmostEqual(result["log_density"][band],evidence-previous[band],places=11)
                previous[band] = evidence

    def test_relabeling_conditioned_features_preserves_predictions(self):
        values = np.random.default_rng(9149).uniform(.1,1.,size=(30,3))
        sources = np.array([[0,2],[1,0],[2,1]])
        permutation = np.array([2,0,1])
        inverse = np.argsort(permutation)
        params = (3,[1],.1,.2,8,np.zeros(3),np.eye(3),2.,.5,[.5,.5])
        original = ContextExpectation(*params,conditioning_sources=sources)
        changed = ContextExpectation(*params,conditioning_sources=inverse[sources[permutation]])
        for i,value in enumerate(values):
            a = original.observe((i+1)*.1,value)
            b = changed.observe((i+1)*.1,value[permutation])
            if a["kind"] == "score":
                np.testing.assert_array_equal(a["log_density"][permutation],b["log_density"])
            for key in ["location","scale2","degrees_freedom","log_weights","positive_probability"]:
                np.testing.assert_array_equal(original.forecast()[key][permutation],changed.forecast()[key])

    def test_conditioning_sources_validate_indices_and_prior_width(self):
        for sources in [[[0,0],[1,0]], [[0,2],[1,0]], [[0.,1.],[1.,0.]], [[0],[1]]]:
            with self.assertRaises(ValueError):
                ContextExpectation(2,[1],.1,.2,8,np.zeros(3),np.eye(3),2.,.5,
                                   conditioning_sources=sources)

    def test_budget_reports_discarded_mass_without_forcing_an_age_cutoff(self):
        full, small = model(lags=(), rate=1., budget=20), model(lags=(), rate=1., budget=2)
        for i, value in enumerate([.1, .2]):
            full.observe((i + 1) * .1, [value])
            small.observe((i + 1) * .1, [value])
        result = full.observe(.3, [.15])
        pruned = small.observe(.3, [.15])
        self.assertAlmostEqual(pruned["pruned_mass"][0], min(np.exp(result["context_log_weights"][0])))
        for i in range(3, 100):
            small.observe((i + 1) * .1, [.15])
            self.assertLessEqual(small.log_weights.shape[1], 2)
        self.assertIn(0., small.starts_sec[0])

    def test_queries_are_pure_and_validation_does_not_modify_state(self):
        observer = model()
        observer.observe(.1, [.2])
        prediction = observer.forecast()
        saved = copy.deepcopy(prediction)
        observer.forecast()["location"].fill(100.)
        for time, value in [(.1, [.2]), (.3, [.2]), (.2, [float("nan")]), (.2, [1., 2.])]:
            with self.assertRaises(ValueError):
                observer.observe(time, value)
        for key, value in prediction.items():
            np.testing.assert_array_equal(value, saved[key])
            np.testing.assert_array_equal(observer.forecast()[key], saved[key])
        observer.observe(.2, [.3])
        for key in prediction:
            np.testing.assert_array_equal(prediction[key], saved[key])

    def test_stream_aggregation_scores_zero_and_censors_eof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"kind": "auditory_envelope", "available_sec": (i + 1) * .01,
                     "envelope_scan": [0. if i >= 6 else .2]} for i in range(12)]
            source = root / "envelopes.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            frames = list(envelope_frames(source, .01, .03, .1))
            self.assertEqual(len(frames), 4)
            np.testing.assert_array_equal(frames[-1][1], [0.])
            manifest, config = root / "manifest.json", root / "config.json"
            manifest.write_text(json.dumps([{"input": str(source), "sha256": sha256(source)}]))
            config.write_text(json.dumps({"input_step_sec": .01, "amplitude_scale": .1,
                 "model": {"features": 1, "lags": [1], "step_sec": .03, "change_rate_hz": .1,
                           "max_hypotheses": 8, "prior_mean": [0., 0.], "prior_precision": [[1., 0.], [0., 1.]],
                           "prior_shape": 2., "prior_scale": .5}}))
            run(manifest, config, root / "result")
            with gzip.open(root / "result/000.jsonl.gz", "rt") as stream:
                result = [json.loads(line) for line in stream]
            self.assertEqual(result[-1]["kind"], "right_censored")
            scores = [row for row in result if row["kind"] == "score"]
            self.assertEqual(len(scores), 3)
            for i, row in enumerate(result):
                if row["kind"] == "score":
                    self.assertEqual(result[i - 1]["kind"], "forecast")
                    self.assertAlmostEqual(result[i - 1]["target_sec"], row["available_sec"])


if __name__ == "__main__":
    unittest.main()
