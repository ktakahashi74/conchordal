import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_prediction_residuals import (
    CenteredRidge, ResidualComparison, evaluate_group, known_query, run,
)


def records(count=30):
    rows = []
    for i in range(count):
        a = np.array([.2, .3, .4]) * (1 + .1 * np.sin(i))
        b = a + [.1, -.02, .05]
        features = np.zeros(57)
        features[0] = 1
        features[1:28] = .1
        features[28:36] = .7
        features[37] = .4 if i % 2 else -.4
        rows.append(dict(
            type="local_prediction_match", voice_id=7, sample_rate=1000,
            issued_step=i * 10, target_step=i * 10 + 10,
            forecast_observed_frame=i * 100, requested_frame=i * 100,
            target_end_frame=i * 100 + 110, completed_before_issue=max(0, i - 1),
            recurrence=a.tolist(), history=b.tolist(), history_weight=[.4] * 3,
            mixed=(a + .4 * (b - a)).tolist(),
            observed=(a + features[37] * np.array([.2, .1, -.1])).tolist(),
            issued_features=features.tolist(),
        ))
    return rows


class PredictionResidualTests(unittest.TestCase):
    def test_scalar_response_matches_independent_centered_batch_fit(self):
        rng = np.random.default_rng(6029)
        x = rng.normal(size=(160, 65))
        x[:, 0] = 1
        y = (x[:, 5] * .4 - x[:, 6] * .2 + rng.normal(size=160) * .1)[:, None]
        ridge = CenteredRidge(65, outputs=1)
        for features, target in zip(x, y):
            ridge.update(features, target)
        centered = x - x.mean(axis=0)
        weights = np.linalg.solve(centered.T @ centered + np.eye(65),
                                  centered.T @ (y - y.mean(axis=0)))
        query = rng.normal(size=65)
        expected = y.mean(axis=0) + (query - x.mean(axis=0)) @ weights
        np.testing.assert_allclose(ridge.predict(query), expected, rtol=1e-11, atol=1e-13)
        self.assertEqual(ridge.predict(query).shape, (1,))

    def test_constant_correction_gives_the_same_linear_comparison_target(self):
        rows = records(100)
        for row in rows:
            row["recurrence"] = [1., 1., 1.]
            row["history"] = [1.5, 1., 1.]
            row["mixed"] = [1.25, 1., 1.]
        for row in evaluate_group(rows):
            if row["predicted"] is None:
                continue
            loss = row["predicted"]["loss_change"]
            self.assertAlmostEqual(loss["loss_geometry"], loss["geometry"], places=13)
            self.assertAlmostEqual(loss["loss_context"], loss["context"], places=13)

    def test_scalar_comparison_preserves_known_equality_after_nonzero_training(self):
        learner = ResidualComparison()
        source = records(1)[0]
        query = known_query(source)
        learner.update(query, query["m"])
        self.assertLess(learner.predict(**query)["loss_change"]["mean_loss"], 0)
        query["m"] = query["a"].copy()
        result = learner.predict(**query)["loss_change"]
        self.assertEqual(result["loss_geometry"], 0)
        self.assertEqual(result["loss_context"], 0)
        self.assertLess(result["mean_loss"], 0)

    def test_centered_incremental_ridge_matches_independent_batch_solution(self):
        rng = np.random.default_rng(4051)
        for dimensions in (9, 65):
            x = rng.normal(size=(120, dimensions))
            x[:, 0] = 1
            x[:, 1] = x[:, 2]
            y = rng.normal(size=(120, 3)) * .03
            ridge = CenteredRidge(dimensions)
            self.assertIsNone(ridge.predict(x[0]))
            for i, (features, target) in enumerate(zip(x, y), 1):
                ridge.update(features, target)
                if i not in (1, 2, 11, 120):
                    continue
                mx, my = x[:i].mean(axis=0), y[:i].mean(axis=0)
                centered = x[:i] - mx
                weights = np.linalg.solve(centered.T @ centered + np.eye(dimensions),
                                          centered.T @ (y[:i] - my))
                query = x[-1] * .3
                expected = my + (query - mx) @ weights
                np.testing.assert_allclose(ridge.predict(query), expected,
                                           rtol=1e-10, atol=1e-13)

    def test_future_outcomes_cannot_change_earlier_issued_predictions(self):
        rows = records()
        baseline = list(evaluate_group(rows))
        altered = copy.deepcopy(rows)
        for row in altered[12:]:
            row["observed"] = [10., 20., 30.]
        changed = list(evaluate_group(altered))
        for before, after in zip(baseline, changed):
            if before["requested_frame"] < rows[12]["target_end_frame"]:
                self.assertEqual(before["predicted"], after["predicted"])
                self.assertEqual({k: v["forecast"] for k, v in before["choices"].items()},
                                 {k: v["forecast"] for k, v in after["choices"].items()})
        self.assertNotEqual(baseline[-1]["predicted"], changed[-1]["predicted"])

    def test_exact_completion_boundary_and_false_counts(self):
        rows = records()
        rows[2]["requested_frame"] = rows[0]["target_end_frame"]
        rows[2]["forecast_observed_frame"] = rows[2]["requested_frame"]
        actual = list(evaluate_group(rows))
        self.assertEqual(actual[2]["completed"], 1)
        self.assertEqual(actual[2]["last_completed_frame"], rows[2]["requested_frame"])
        self.assertEqual(actual[0]["training_count"], 0)
        self.assertIsNone(actual[0]["predicted"])
        rows[2]["completed_before_issue"] = 2
        with self.assertRaisesRegex(ValueError, "prior completion count"):
            list(evaluate_group(rows))

    def test_zero_correction_is_known_equality_and_mean_residual_is_actual_mean(self):
        rows = records()
        for row in rows:
            row["mixed"] = row["recurrence"][:]
        actual = list(evaluate_group(rows))
        for row in actual:
            self.assertEqual(row["actual_loss_change"], 0)
            self.assertFalse(row["nonzero_delta"])
            if row["predicted"] is not None:
                self.assertEqual(set(row["predicted"]["loss_change"].values()), {0.})
                eligible = rows[:row["completed"]]
                residuals = [np.array(r["observed"], dtype=np.float32).astype(float)
                             - np.array(r["recurrence"], dtype=np.float32).astype(float)
                             for r in eligible]
                np.testing.assert_allclose(row["predicted"]["residual"]["mean_residual"],
                                           np.mean(residuals, axis=0), atol=1e-15)

    def test_common_energy_gain_preserves_features_and_scales_physical_losses(self):
        rows = records()
        scaled = copy.deepcopy(rows)
        for row in scaled:
            for name in ("recurrence", "history", "mixed", "observed"):
                row[name] = [4 * x for x in row[name]]
            row["issued_features"][1:28] = [2 * x for x in row["issued_features"][1:28]]
        for a, b in zip(rows, scaled):
            np.testing.assert_array_equal(known_query(a)["context"], known_query(b)["context"])
        for a, b in zip(evaluate_group(rows), evaluate_group(scaled)):
            self.assertAlmostEqual(16 * a["actual_loss_change"], b["actual_loss_change"])
            if a["predicted"] is not None:
                for name in a["predicted"]["loss_change"]:
                    self.assertAlmostEqual(16 * a["predicted"]["loss_change"][name],
                                           b["predicted"]["loss_change"][name])
                for name in a["predicted"]["residual"]:
                    np.testing.assert_allclose(4 * np.array(a["predicted"]["residual"][name]),
                                               b["predicted"]["residual"][name])

    def test_missing_context_is_unavailable_not_an_observed_zero(self):
        rows = records()
        rows[0]["issued_features"] = None
        rows[4]["issued_features"] = None
        result = list(evaluate_group(rows))
        self.assertEqual(result[2]["completed"], 1)
        self.assertEqual(result[2]["training_count"], 0)
        self.assertIsNone(result[4]["predicted"])
        self.assertGreater(result[4]["training_count"], 0)
        self.assertEqual(result[-1]["training_count"], result[-1]["completed"] - 2)

    def test_context_can_distinguish_a_control_with_identical_forecast_geometry(self):
        learner = ResidualComparison()
        queries = []
        for sign in (-1, 1):
            row = records(1)[0]
            row["issued_features"][37] = sign * .4
            queries.append(known_query(row))
        for _ in range(200):
            for sign, query in zip((-1, 1), queries):
                learner.update(query, query["a"] + sign * np.array([.05, .02, 0.]))
        errors = dict(mean_residual=0., geometry=0., context=0.)
        for sign, query in zip((-1, 1), queries):
            result = learner.predict(**query)
            expected = sign * np.array([.05, .02, 0.])
            for name, prediction in result["residual"].items():
                errors[name] += float(np.sum((prediction - expected) ** 2))
        self.assertLess(errors["context"], .1 * errors["geometry"])
        self.assertAlmostEqual(errors["geometry"], errors["mean_residual"])

    def test_runner_separates_cases_horizons_roles_and_preserves_unavailable_loss(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "report.jsonl"
            rows = records()
            source.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            plan = root / "plan.json"
            plan.write_text(json.dumps(dict(ridge=1., cases=[
                dict(case="fresh", role="validation", input="report.jsonl"),
                dict(case="reused", role="development", input="report.jsonl")
            ])))
            result = run(plan, root / "out")
            self.assertEqual(len(result["cases"]), 2)
            self.assertEqual(result["cases"][0]["comparisons"],
                             result["cases"][1]["comparisons"])
            for role in ("validation", "development"):
                whole = next(r for r in result["pooled"] if r["role"] == role
                             and r["request_cutoff_sec"] == 0 and r["stratum"] == "all")
                self.assertEqual(whole["n"], len(rows))
                self.assertEqual(whole["unavailable"], 2)
                for model in whole["models"].values():
                    self.assertAlmostEqual(model["selected_sse"],
                                           model["eligible_selected_sse"] + whole["unavailable_mixed_sse"])
            with gzip.open(root / "out/fresh.jsonl.gz", "rt") as stream:
                trace = [json.loads(line) for line in stream]
            self.assertEqual(trace[2]["training_count"], 1)
            with self.assertRaises(FileExistsError):
                run(plan, root / "out")


if __name__ == "__main__":
    unittest.main()
