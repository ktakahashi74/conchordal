"""Check the recovery likelihood against scalar and derivative references."""

from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_temporal_memory_recovery as recovery
import temporal_cognition_reference as ref


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.design = recovery.registration()

    def test_factorial_and_assignment_preserve_independent_factors(self):
        rows = recovery.cells(self.design)
        self.assertEqual(rows.shape, (720, 5))
        reordered = copy.deepcopy(self.design)
        reordered["factorial"] = dict(reversed(list(reordered["factorial"].items())))
        np.testing.assert_array_equal(recovery.cells(reordered), rows)
        assignments = recovery.assignment(self.design)
        self.assertEqual(assignments, recovery.assignment(self.design))
        self.assertEqual(len(assignments), 2880)
        families = Counter(a["source_family"] for a in assignments)
        self.assertEqual(set(families.values()), {4})
        per_person = defaultdict(list)
        for a in assignments:
            per_person[a["synthetic_participant"]].append(a)
            self.assertGreaterEqual(a["prefix_sec_estimate"], 16)
        self.assertEqual(len(per_person), 960)
        for person in per_person.values():
            self.assertEqual(len({a["source_family"] for a in person}), 3)
            self.assertEqual({a["order"] for a in person}, {0, 1, 2})

    def test_full_bank_probabilities_match_independent_scalar_reference(self):
        truth = [20, 4, 2.7, -0.4]
        theta = np.asarray([*np.log(truth[:3]), truth[3]])
        for condition in self.design["conditions"]:
            inputs = recovery.prepared_inputs(self.design, condition)
            logit, _ = recovery.forward(theta, inputs)
            support, ages, interference, scores = inputs
            for i in range(0, len(support), 29):
                strengths = [min(truth[2], support[i])] + [1] * (ages.shape[1] - 1)
                available = [ref.episode_log_availability(s, a, j, truth[0], truth[1])
                             for s, a, j in zip(strengths, ages[i], interference[i])]
                expected = ref.recognition_probability(available, scores[i], truth[3])
                self.assertAlmostEqual(expit(logit[i]), expected, places=14)

    def test_gradient_matches_central_difference_with_competition_and_floor(self):
        inputs = recovery.prepared_inputs(self.design, self.design["conditions"][-1])
        theta = np.asarray([math.log(7), math.log(5), math.log(2.7), 0.4])
        usable = np.full(720, 4)
        yes = np.tile(np.arange(5), 144)
        for floor in [1e-300, 0.01]:
            value, gradient = recovery.objective(theta, inputs, usable, yes, 1e-6, floor)
            self.assertTrue(math.isfinite(value))
            for i in range(4):
                step = np.zeros(4)
                step[i] = 1e-5
                hi = recovery.objective(theta + step, inputs, usable, yes, 1e-6, floor)[0]
                lo = recovery.objective(theta - step, inputs, usable, yes, 1e-6, floor)[0]
                self.assertAlmostEqual(gradient[i], (hi - lo) / 2e-5, delta=1e-5)

    def test_fit_recovers_parameters_from_expected_counts_without_response_noise(self):
        inputs = recovery.prepared_inputs(self.design, self.design["conditions"][-1])
        truth = [32, 5, 2.7, -0.4]
        theta = np.asarray([*np.log(truth[:3]), truth[3]])
        logits, _ = recovery.forward(theta, inputs)
        usable = np.full(720, 2000)
        result = recovery.fit_counts(self.design, inputs, usable, usable * expit(logits))
        self.assertEqual(result["status"], "fitted", result)
        np.testing.assert_allclose(result["estimate"], truth, rtol=1e-4, atol=1e-4)

    def test_no_observations_cannot_be_identified_by_ridge(self):
        inputs = recovery.prepared_inputs(self.design, self.design["conditions"][0])
        result = recovery.fit_counts(self.design, inputs, np.zeros(720), np.zeros(720))
        self.assertEqual(result["status"], "failed")

    def test_short_retention_oracle_is_identifiable_with_sufficient_expected_counts(self):
        inputs = recovery.prepared_inputs(self.design, self.design["conditions"][-1])
        truth = [2, 4, 3, 0]
        theta = np.asarray([*np.log(truth[:3]), truth[3]])
        logits, _ = recovery.forward(theta, inputs)
        usable = np.full(720, 100000)
        result = recovery.fit_counts(self.design, inputs, usable, usable * expit(logits))
        self.assertEqual(result["status"], "fitted", result)
        np.testing.assert_allclose(result["estimate"], truth, rtol=1e-4, atol=1e-4)

    def test_budget_exhaustion_cannot_be_a_recovery_pass(self):
        self.design["compute_envelope"]["wall_hours_max"] = 0
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            design_path = root / "design.json"
            design_path.write_text(json.dumps(self.design))
            recovery.run(design_path, root / "run", 1)
            result = json.loads((root / "run/result.json").read_text())
            self.assertEqual(result["status"], "incomplete_budget_exhausted")
            self.assertEqual(result["cells"], [])
            self.assertEqual((root / "run/fits.jsonl").read_text(), "")

    def test_failed_fits_remain_in_quantiles(self):
        records = [{"condition": "test", "grid_index": 0, "truth": [20, 4, 3, 0],
                    "status": "fitted" if i < 8 else "failed",
                    "absolute_errors": [0] * 4 if i < 8 else ["infinite"] * 4} for i in range(10)]
        result = recovery.recovery_summary(records, self.design)
        self.assertFalse(result["pass"])
        self.assertEqual(result["failed_fits"], 2)
        for param in result["parameters"].values():
            self.assertEqual(param["median"], 0)
            self.assertEqual(param["p90"], "infinite")
            self.assertFalse(param["pass"])

    def test_auxiliary_queries_retain_every_original_cell_and_observation_age(self):
        proposed = recovery.short_query_registration()
        rows = recovery.cells(proposed)
        self.assertEqual(rows.shape, (2640, 6))
        row_set = set(map(tuple, rows))
        self.assertTrue(all((*row, 8) in row_set for row in recovery.cells(self.design)))
        self.assertTrue(all(row[1] <= 1 for row in rows if row[0] == 1.25))
        self.assertTrue(all(row[1] <= 4 for row in rows if row[0] == 2.25))
        inputs = recovery.prepared_inputs(proposed, proposed["conditions"][0])
        np.testing.assert_array_equal(inputs[1][:, 0], rows[:, 0].astype(float) + rows[:, 5].astype(float))
        self.assertEqual(inputs[1].min(), 2.25)

    def test_auxiliary_assignment_preserves_one_prefix_per_family(self):
        proposed = recovery.short_query_registration()
        proposed["allocation"].update(ratings_per_cell=4, candidate_participants=960, assigned_targets=10560)
        assignments = recovery.assignment(proposed)
        self.assertEqual(len(assignments), 10560)
        families = Counter(a["source_family"] for a in assignments)
        self.assertEqual(set(families.values()), {4})
        per_person = defaultdict(list)
        for a in assignments:
            per_person[a["synthetic_participant"]].append(a)
        self.assertEqual(len(per_person), 960)
        for person in per_person.values():
            self.assertEqual(len(person), 11)
            self.assertEqual(len({a["source_family"] for a in person}), 11)
            self.assertEqual({a["order"] for a in person}, set(range(11)))

    def test_planning_information_matches_expected_likelihood_curvature(self):
        proposed = recovery.short_query_registration()
        inputs = recovery.prepared_inputs(proposed, proposed["conditions"][-1])
        theta = np.asarray([math.log(3), math.log(7), math.log(2.7), 0.4])
        eta, jac = recovery.forward(theta, inputs)
        p = expit(eta)
        usable = np.full(len(p), 4)
        fisher = jac.T @ ((usable * p * (1 - p))[:, None] * jac)
        numeric = np.zeros((4, 4))
        for i in range(4):
            step = np.zeros(4)
            step[i] = 1e-5
            hi = recovery.objective(theta + step, inputs, usable, usable * p, 0, 1e-300)[1]
            lo = recovery.objective(theta - step, inputs, usable, usable * p, 0, 1e-300)[1]
            numeric[:, i] = (hi - lo) / 2e-5
        np.testing.assert_allclose(fisher, numeric, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
