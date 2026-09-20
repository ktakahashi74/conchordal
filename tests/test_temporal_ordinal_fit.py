"""Mixture-likelihood, gradient and finite-fit checks on constructed responses."""

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import temporal_ordinal_fit as ref


class OrdinalFitTests(unittest.TestCase):
    def test_crossed_folds_exclude_both_axes_and_score_every_target_once(self):
        source = np.repeat(np.arange(5), 5)
        participant = np.tile(np.arange(5), 5)
        heldout_counts = np.zeros(25, dtype=int)
        for train, heldout in ref.crossed_partitions(source, participant):
            self.assertEqual(len(train), 16)
            self.assertEqual(len(heldout), 1)
            heldout_counts[heldout] += 1
            self.assertFalse(set(source[train]) & set(source[heldout]))
            self.assertFalse(set(participant[train]) & set(participant[heldout]))
        np.testing.assert_array_equal(heldout_counts, np.ones(25))
        with self.assertRaises(ValueError): ref.crossed_partitions([0, 1], [0, 1])
        with self.assertRaises(ValueError): ref.crossed_partitions([0.5], [0])

    def test_whole_mixture_loss_and_gradient_against_direct_cdf(self):
        x = np.array([[1, -1], [1, 2], [1, -0.5], [1, 1], [1, 0.25]], dtype=float)
        data = ref.prepare(x, [0, 0, 1, 2, 2], [0.25, 0.25, 0.8, 0.1, 0.6], [0.8, 0, 1], [0, 2, 4])
        parameters = np.array([0.3, -1.5, 0.2, -0.1, 0.4])

        def direct(theta):
            cuts = [theta[1]]
            for value in theta[2:]:
                cuts.append(cuts[-1] + math.log1p(math.exp(value)) + 1e-8)
            probs = []
            for row in x:
                cumulative = [0.0] + [1 / (1 + math.exp(-(cut - row[1] * theta[0]))) for cut in cuts] + [1.0]
                probs.append([b - a for a, b in zip(cumulative, cumulative[1:])])
            losses = []
            for target, label in enumerate(data["y"]):
                prediction = ((1 - data["coverage"][target] * data["support"][target]) * data["prior"][label]
                              + sum(data["coverage"][target] * w * p[label] for i, w, p in
                                    zip(data["index"], data["weights"], probs) if i == target))
                losses.append(-math.log(prediction))
            return sum(losses) / len(losses) + 0.05 * theta[0] ** 2

        loss, gradient = ref.objective(parameters, data, 0.1)
        self.assertAlmostEqual(loss, direct(parameters), places=13)
        for index in range(len(parameters)):
            delta = np.zeros(len(parameters)); delta[index] = 1e-5
            numerical = (direct(parameters + delta) - direct(parameters - delta)) / 2e-5
            self.assertAlmostEqual(gradient[index], numerical, places=8)
        # The unsupported middle response still contributes its fixed prior loss.
        self.assertGreater(loss, -math.log(data["prior"][2]) / 3)

    def test_intercept_only_fit_recovers_empirical_category_probabilities(self):
        counts = [10, 20, 30, 25, 15]
        y = np.repeat(np.arange(5), counts)
        data = ref.prepare(np.ones((len(y), 1)), np.arange(len(y)), np.ones(len(y)), np.ones(len(y)), y)
        result = ref.fit(data, 0.1)
        self.assertTrue(result["fit_complete"], result)
        self.assertFalse(result["calibrated"])
        self.assertEqual(result["beta"], [0.0])
        cuts = np.array(result["cutpoints"])
        empirical = np.cumsum(counts)[:4] / sum(counts)
        np.testing.assert_allclose(cuts, np.log(empirical / (1 - empirical)), atol=1e-5, rtol=0)
        self.assertLessEqual(result["evaluations"], ref.PLAN["solver"]["max_evaluations"])
        np.testing.assert_allclose(result["training_prior"], (np.array(counts) + 0.5) / (sum(counts) + 2.5))

    def test_budget_exhaustion_and_all_unknown_are_not_completed_fits(self):
        y = np.repeat(np.arange(5), [10, 20, 30, 25, 15])
        data = ref.prepare(np.ones((len(y), 1)), np.arange(len(y)), np.ones(len(y)), np.ones(len(y)), y)
        result = ref.fit(data, 0.1, max_evaluations=1)
        self.assertEqual(result["status"], "evaluation_budget")
        self.assertFalse(result["fit_complete"])
        self.assertEqual(result["evaluations"], 1)
        self.assertIsNotNone(result["parameters"])
        unknown = ref.prepare(np.empty((0, 1)), [], [], np.ones(5), np.arange(5))
        self.assertEqual(ref.fit(unknown, 0.1)["status"], "unsupported")
        with self.assertRaises(ValueError): ref.fit(data, 0.12345)

    def test_input_validation_and_owned_snapshot(self):
        x = np.ones((5, 1))
        data = ref.prepare(x, np.arange(5), np.ones(5), np.ones(5), np.arange(5))
        x[0, 0] = 50
        self.assertEqual(data["x"][0, 0], 1)
        with self.assertRaises(ValueError): data["x"][0, 0] = 50
        with self.assertRaises(ValueError): ref.prepare(np.ones((2, 1)), [0, 0], [0.6, 0.6], [1], [0])
        with self.assertRaises(ValueError): ref.prepare(np.ones((1, 1)), [0], [1], [1], [0.5])
        with self.assertRaises(ValueError): ref.prepare(np.ones((1, 1)), [0.5], [1], [1], [0])
        with self.assertRaises(ValueError): ref.prepare([[1, 0], [1, 1]], [0, 1], [1, 1], [1, 1], [0, 1], fixed_columns=(0, 1))

    def test_random_mixtures_cover_all_category_gradient_branches(self):
        rng = np.random.default_rng(9122031)
        for _ in range(12):
            x = np.column_stack([np.ones(10), rng.normal(size=(10, 2))])
            data = ref.prepare(x, np.repeat(np.arange(5), 2), np.tile([0.25, 0.4], 5), rng.random(5), np.arange(5))
            parameters = rng.normal(size=6)
            _, gradient = ref.objective(parameters, data, 0.01)
            for index in range(len(parameters)):
                delta = np.zeros(6); delta[index] = 1e-5
                plus = ref.objective(parameters + delta, data, 0.01)[0]
                minus = ref.objective(parameters - delta, data, 0.01)[0]
                self.assertAlmostEqual(gradient[index], (plus-minus)/2e-5, places=7)

    def test_elapsed_budget_keeps_last_full_evaluation_as_incomplete(self):
        data = ref.prepare(np.ones((5, 1)), np.arange(5), np.ones(5), np.ones(5), np.arange(5))
        with patch.object(ref.time, "perf_counter", side_effect=[0, 0.01, 301, 302]):
            result = ref.fit(data, 0.1)
        self.assertEqual(result["status"], "elapsed_budget")
        self.assertFalse(result["fit_complete"])
        self.assertEqual(result["evaluations"], 1)
        self.assertIsNotNone(result["parameters"])

    def test_heldout_backoff_uses_frozen_training_counts(self):
        y = np.repeat(np.arange(5), [10, 20, 30, 25, 15])
        training = ref.prepare(np.ones((len(y), 1)), np.arange(len(y)), np.ones(len(y)), np.ones(len(y)), y)
        heldout = ref.prepare(np.empty((0, 1)), [], [], [1], [4], training_prior=training["prior"])
        loss, gradient = ref.objective([-1, 0, 0, 0], heldout, 0)
        self.assertAlmostEqual(loss, -math.log(training["prior"][4]), places=14)
        np.testing.assert_array_equal(gradient, np.zeros(4))
        self.assertEqual(heldout["prior_source"], "frozen_training_prior")
        with self.assertRaises(ValueError): ref.fit(heldout, 0.1)


if __name__ == "__main__":
    unittest.main()
