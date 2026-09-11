"""Independent checks of the registered private physical-record recovery."""

import math
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_temporal_trace_recovery as recovery
import optimize_temporal_trace_allocation as allocation
import optimize_temporal_trace_separation as cut_allocation
import plan_temporal_trace_separation as separation
import temporal_cognition_reference as reference


class PhysicalRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = recovery.registration()
        cls.geometries = [recovery.emission_geometry(cls.design, condition) for condition in cls.design["conditions"]]

    def test_alternative_cuts_preserve_baseline_and_even_counts_with_known_optimum(self):
        baseline = np.array([2, 2, 2])
        cost = np.array([100.0, 1.0, 1.0])
        distances = np.array([[0, 0.5, 0.1], [0, 0.1, 0.5]])
        counts = cut_allocation.allocate_counts(cost, baseline, np.array([1, 2]), distances, 3.2)
        np.testing.assert_array_equal(counts, [2, 6, 6])
        self.assertTrue(np.all(distances @ counts >= 3.2))
        feasible = [200 + a + b for a in range(2, 20, 2) for b in range(2, 20, 2)
                    if 0.5 * a + 0.1 * b >= 3.2 and 0.1 * a + 0.5 * b >= 3.2]
        self.assertEqual(cost @ counts, min(feasible))
        with self.assertRaises(ValueError):
            cut_allocation.allocate_counts(cost, baseline, np.array([1, 2]), np.zeros((1, 3)), 3.2)

    def test_forward_gradient_and_complete_physical_mass(self):
        parameters = np.log([7.3, 5.7, 2.6])
        design = dict(self.design, second_prefix_counts=[1, 4, 16],
                      fully_supported_first_prefix={"first_counts": [64, 128], "delays_sec": [0.125, 1, 4, 16],
                                                    "second_counts": [1, 4, 16]})
        for condition, geometry in zip(self.design["conditions"], self.geometries):
            rows = recovery.cells(design, condition)
            self.assertEqual(len(rows), 492)
            probability, gradient = recovery.forward(parameters, design, condition, rows, geometry)
            np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-10, rtol=0)
            np.testing.assert_allclose(gradient.sum(axis=1), 0, atol=1e-10, rtol=0)
            for index in range(3):
                delta = np.eye(3)[index] * 1e-5
                high = recovery.forward(parameters + delta, design, condition, rows, geometry)[0]
                low = recovery.forward(parameters - delta, design, condition, rows, geometry)[0]
                np.testing.assert_allclose(gradient[:, :, index], (high - low) / 2e-5, atol=2e-10, rtol=1e-5)
            original = recovery.forward(parameters, self.design, condition, recovery.cells(self.design, condition), geometry)[0]
            np.testing.assert_array_equal(probability[(rows[:, 3] == 1) & (rows[:, 4] == 0)], original)

    def test_population_likelihood_recovers_all_three_parameters(self):
        condition, geometry = self.design["conditions"][-1], self.geometries[-1]
        rows = recovery.cells(self.design, condition)
        truth = np.asarray([20.0, 4.0, 3.0])
        probability = recovery.forward(np.log(truth), self.design, condition, rows, geometry)[0]
        result = recovery.fit(probability * 1000, self.design, condition, rows, geometry)
        self.assertTrue(result["success"], result)
        np.testing.assert_allclose(np.log(np.asarray(result["estimate"]) / truth), 0, atol=1e-4, rtol=0)

    def test_registered_interference_allocation_fits_physical_delays(self):
        for condition in self.design["conditions"]:
            credit = condition["observed_fraction"] * condition["anchor_coverage"]
            rows = recovery.cells(self.design, condition)
            self.assertEqual(len(rows), 140)
            for count, delay, interference in rows:
                events = 2 * math.ceil(interference / (2 * credit))
                self.assertLessEqual((events + 1) * 0.0625, delay)
                self.assertIn(count, self.design["prefix_counts"])

    def test_reduced_forward_matches_sequential_two_head_trace_with_actual_credit(self):
        width = self.design["prefix_timestamp_width_sec"]
        for condition, geometry in zip(self.design["conditions"], self.geometries):
            for head in ("onset", "release"):
                for truth, row in [([20, 4, 3], [4, 1, 4]), ([2, 1, 6], [128, 4, 16]),
                                   ([2, 4, 6], [128, 0.125, 0, 16]), ([20, 16, 1.5], [16, 1, 4, 4]),
                                   ([1200, 4, 3], [64, 64, 16, 16]),
                                   ([2, 4, 6], [128, 1, 4, 4, 1]), ([20, 16, 3], [64, 4, 16, 16, 1])]:
                    state = reference.PrivateTimingTrace(1, *truth)
                    keys = {family: (1, index, family) for index, family in enumerate(("nonperiodic", "periodic"))}
                    competing = (1, 3, "nonperiodic")
                    retained = set(keys.values()) | {competing}
                    event_id, clock = 0, 0.0

                    def observe(time, event_head, key, fraction, anchors, confirmed=True):
                        nonlocal event_id
                        state.observe(event_id, event_head, (time - width, time),
                            [{"key": key, "weight": 1, "anchors": anchors}], fraction, retained, confirmed)
                        event_id += 1

                    for family in ("nonperiodic", "periodic"):
                        cfg = self.design["prefix_geometry"][family]

                        def anchors_at(time, index, full_support):
                            bin_width = cfg["period_sec"] * (1 if family == "periodic" else 4) / 32
                            coordinate = 4.5 * cfg["period_sec"] if index == 32 else (index + 0.5) * bin_width
                            center = time - width / 2 - coordinate
                            return [{"weight": (1 if full_support else condition["anchor_coverage"]) / condition["modes"],
                                     "interval": (center - mode * self.design["alternative_anchor_shift_sec"][family] - width / 2,
                                                  center - mode * self.design["alternative_anchor_shift_sec"][family] + width / 2),
                                     "period_sec": cfg["period_sec"]} for mode in range(condition["modes"])]

                        def target(time, index, confirmed=True, full_support=False):
                            anchors = anchors_at(time, index, full_support)
                            fraction = 1 if full_support else condition["observed_fraction"]
                            if head == "release":
                                observe(time - 0.025, "onset", keys[family], fraction, anchors)
                            observe(time, head, keys[family], fraction, anchors, confirmed)
                            if head == "onset":
                                observe(time + 0.025, "release", keys[family], fraction, anchors)

                        for step in range(1, int(row[0]) + 1):
                            clock += cfg["repeat_sec"]
                            target(clock, cfg["first_bin"], not condition["skip_every"] or step % condition["skip_every"] != 0,
                                   len(row) == 5 and row[4] == 1)
                        credit = condition["observed_fraction"] * condition["anchor_coverage"]
                        events = 2 * math.ceil(row[2] / (2 * credit))
                        for index in range(events):
                            time = clock + row[1] * (index + 1) / (events + 1)
                            observe(time, "onset" if index % 2 == 0 else "release", competing, row[2] / events,
                                    [{"weight": 1, "interval": (time - 0.5, time - 0.5), "period_sec": 1}])
                        clock += row[1]
                        for step in range(1, (int(row[3]) if len(row) >= 4 else 1) + 1):
                            if step > 1:
                                clock += cfg["repeat_sec"]
                            target(clock, cfg["second_bin"], not condition["skip_every"] or step % condition["skip_every"] != 0)
                        clock += 0.025
                    kernels = {keys[family]: geometry["full"][family]["kernel"] for family in keys}
                    inventory = [{"key": keys[family], "weight": self.design["probe_weights"][family],
                                  "probabilities": state.probabilities(keys[family], head)} for family in keys]
                    actual = reference.private_outcome_distribution(kernels, inventory, 1, retained)["probabilities"]
                    expected = recovery.forward(np.log(truth), self.design, condition, np.asarray([row], float), geometry)[0][0]
                    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-9,
                                               err_msg=f"{condition['name']} {head} {truth} {row}")
                    cost = recovery.replay_cost_seconds(self.design, np.asarray([row], float))[0]
                    self.assertAlmostEqual(cost, clock + 4, places=9)

    def test_failed_fits_remain_in_tail_and_duplicate_records_are_rejected(self):
        truth = self.design["truth_grid"][0]
        records = [{"condition": 0, "truth_id": 0, "replicate": index, "truth": truth,
                    "success": index < 90, "estimate": truth if index < 90 else None,
                    "absolute_log_errors": [0, 0, 0] if index < 90 else None} for index in range(100)]
        result = recovery.summarize(records, self.design)[0]
        json.dumps(result)
        self.assertEqual(result["failed_fits"], 10)
        self.assertEqual(result["parameters"][0]["median"], 0)
        self.assertIsNone(result["parameters"][0]["p90"])
        self.assertFalse(result["passed"])
        with self.assertRaises(ValueError):
            recovery.summarize(records + [records[0]], self.design)

    def test_missing_observation_is_one_categorical_record_and_empty_data_fail(self):
        condition, geometry = self.design["conditions"][-1], self.geometries[-1]
        rows = recovery.cells(self.design, condition)
        p = recovery.forward(np.log([20, 4, 3]), self.design, condition, rows, geometry)[0]
        rng = np.random.default_rng(1827)
        counts = rng.multinomial(16, p / p.sum(axis=1, keepdims=True))
        np.testing.assert_array_equal(counts.sum(axis=1), 16)
        self.assertGreater(counts[:, -1].sum(), 0)
        loss, _ = recovery.objective(np.log([20, 4, 3]), counts * 0, self.design, condition, rows, geometry)
        self.assertTrue(math.isinf(loss))

    def test_registered_nonsmooth_fallback_refines_the_failed_case(self):
        condition, geometry = self.design["conditions"][-1], self.geometries[-1]
        rows = recovery.cells(self.design, condition)
        p = recovery.forward(np.log([2, 1, 3]), self.design, condition, rows, geometry)[0]
        rng = np.random.default_rng(np.random.SeedSequence([2026091101, 2, 1, 25]))
        counts = rng.multinomial(1024, p / p.sum(axis=1, keepdims=True))
        revised = json.loads(json.dumps(self.design))
        revision = Path(__file__).resolve().parents[1] / "docs/roadmap/temporal-dcc/private-recovery-fit-v2.json"
        revised["fit"]["nonsmooth_fallback"] = json.loads(revision.read_text())
        result = recovery.fit(counts, revised, condition, rows, geometry)
        self.assertTrue(result["success"], result)
        self.assertLess(np.max(np.abs(np.log(np.asarray(result["estimate"]) / [2, 1, 3]))), 0.1)
        if result["solver"] == "Nelder-Mead":
            self.assertGreaterEqual(result["local_probe_min_loss_change"], -1e-10)

    def test_covariance_allocation_derivative_matches_finite_information_changes(self):
        matrices = np.asarray([[np.diag([1, 2, 3]), [[2, 1, 0], [1, 2, 0], [0, 0, 1]],
                                [[1, 0, 0.5], [0, 1, 0], [0.5, 0, 2]]]], float)
        counts = np.asarray([2.0, 3.0, 5.0])
        covariance, derivative = allocation.covariance_constraints(counts, matrices)
        self.assertTrue(np.all(covariance > 0))
        for index in range(3):
            delta = np.eye(3)[index] * 1e-5
            hi = allocation.covariance_constraints(counts + delta, matrices)[0]
            lo = allocation.covariance_constraints(counts - delta, matrices)[0]
            np.testing.assert_allclose(derivative[:, :, index], (hi - lo) / 2e-5, atol=1e-10, rtol=1e-7)

    def test_lower_loss_cap_corner_is_refined_despite_a_converged_flat_branch(self):
        root = Path(__file__).resolve().parents[1] / "docs/roadmap/temporal-dcc"
        design = json.loads((root / "private-recovery-design-v4.json").read_text())
        condition, geometry = design["conditions"][-1], self.geometries[-1]
        rows = recovery.cells(design, condition)
        truth = np.asarray([2, 16, 6])
        probability = recovery.forward(np.log(truth), design, condition, rows, geometry)[0]
        rng = np.random.default_rng(np.random.SeedSequence([2026091104, 2, 8, 19]))
        counts = rng.multinomial(design["probes_per_cell"], probability / probability.sum(axis=1, keepdims=True))
        old = recovery.fit(counts, design, condition, rows, geometry)
        self.assertEqual(old.get("failure"), "rank_deficient_fitted_timing_model")
        design["fit"]["refine_lower_loss_attempts"] = True
        result = recovery.fit(counts, design, condition, rows, geometry)
        self.assertTrue(result["success"], result)
        self.assertEqual(result["solver"], "Nelder-Mead")
        self.assertLess(result["loss"], 4.506029216869672 - 1e-6)
        self.assertLess(np.max(np.abs(np.log(np.asarray(result["estimate"]) / truth))), math.log(1.25))

    def test_high_cap_profile_gradient_and_unidentifiable_self_distance(self):
        parameters = np.log([2.3, 5.7])
        for condition, geometry in zip(self.design["conditions"], self.geometries):
            rows = recovery.cells(self.design, condition)
            truth = recovery.forward(np.log([2, 4, 6]), self.design, condition, rows, geometry)[0]
            counts = np.arange(2, 2 * len(rows) + 1, 2)
            value, gradient = separation.profile_distance(parameters, truth, counts, self.design, condition, rows, geometry)
            self.assertGreater(value, 0)
            for index in range(2):
                delta = np.eye(2)[index] * 1e-5
                hi = separation.profile_distance(parameters + delta, truth, counts, self.design, condition, rows, geometry)[0]
                lo = separation.profile_distance(parameters - delta, truth, counts, self.design, condition, rows, geometry)[0]
                self.assertAlmostEqual(gradient[index], (hi - lo) / 2e-5, delta=1e-5)
            unsaturated = recovery.forward(np.r_[parameters, math.log(512)], self.design, condition, rows, geometry)[0]
            value, gradient = separation.profile_distance(parameters, unsaturated, counts, self.design, condition, rows, geometry)
            self.assertAlmostEqual(value, 0, delta=1e-9)
            np.testing.assert_allclose(gradient, 0, atol=1e-9, rtol=0)

    def test_low_cap_cancellation_is_not_an_identified_fit(self):
        condition, geometry = self.design["conditions"][0], self.geometries[0]
        rows = recovery.cells(self.design, condition)
        probabilities, derivative = recovery.forward(np.log([20, 1, 0.7]), self.design, condition, rows, geometry)
        np.testing.assert_array_equal(derivative[:, :, 2], 0)
        alternative = recovery.forward(np.log([20, 1, 0.9]), self.design, condition, rows, geometry)[0]
        np.testing.assert_allclose(probabilities, alternative, atol=1e-14, rtol=0)
        revised = json.loads(json.dumps(self.design))
        revised["fit"]["require_full_rank_information"] = True
        fit = recovery.fit(probabilities * 1000, revised, condition, rows, geometry)
        self.assertFalse(fit["success"])

    def test_complete_low_cap_branch_cannot_exceed_half_first_pattern_mass(self):
        condition, geometry = self.design["conditions"][0], self.geometries[0]
        row = np.asarray([[4, 0.125, 0]])
        truth = recovery.forward(np.log([2, 1, 1.5]), self.design, condition, row, geometry)[0][0]
        corner = geometry["fixed"] + np.asarray([0.5, 0.5]) @ geometry["differences"]
        distance = -np.log(np.sqrt(truth * corner).sum())
        self.assertGreater(distance, 0)
        for tau in [0.5, 2, 20, 1200, 4800]:
            candidate = recovery.forward(np.log([tau, 4, 0.6]), self.design, condition, row, geometry)[0][0]
            fraction = 1 / (1 + math.exp(0.125 / tau))
            expected = geometry["fixed"] + np.asarray([fraction, fraction]) @ geometry["differences"]
            np.testing.assert_allclose(candidate, expected, atol=1e-13, rtol=0)

    def test_registered_global_separation_and_smallest_count_have_independent_corner_certificate(self):
        root = Path(__file__).resolve().parents[1] / "docs/roadmap/temporal-dcc"
        plan = json.loads((root / "private-recovery-planning-v3.json").read_text())
        chosen = json.loads((root / "private-recovery-design-v3.json").read_text())
        condition, geometry = self.design["conditions"][0], self.geometries[0]
        rows = recovery.cells(self.design, condition)
        index = np.flatnonzero(np.all(rows == [4, 0.125, 0], axis=1)).item()
        alternative = geometry["fixed"] + np.asarray([0.5, 0.5]) @ geometry["differences"]
        distances = []
        for truth in self.design["truth_grid"]:
            p = recovery.forward(np.log(truth), self.design, condition, rows[[index]], geometry)[0][0]
            mask = (p > 0) & (alternative > 0)
            gradient = 0.5 * (geometry["differences"][:, mask] @ np.sqrt(p[mask] / alternative[mask]))
            # Affinity is concave; this sign certifies its box maximum at the upper corner.
            self.assertTrue(np.all(gradient >= 0))
            distances.append(-math.log(np.sqrt(p[mask] * alternative[mask]).sum()))
        minimum = min(distances)
        count = next(n for n in plan["count_candidates"] if minimum * n >= 8)
        self.assertEqual(chosen["probes_per_cell"][index], count)
        self.assertAlmostEqual(minimum, 0.0013318198749238147, places=12)


if __name__ == "__main__":
    unittest.main()
