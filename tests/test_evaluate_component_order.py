"""Set-valued observations, causal forecasts, and separate time/frequency evidence."""

import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_component_order import (ComponentOrderPredictor, MEMORY, control_audio,
                                      densities, evaluate, normal_density, summarize)
from evaluate_frequency_events import FrequencyEventObserver


def frames(count=25, simultaneous=False):
    result = []
    for tick in range(30 * count + 20):
        index = (tick - 10) // 30
        active = tick >= 10 and (tick - 10) % 30 == 0 and index < count
        frequencies = [8.3 + .65 * ((index % 4) // 2)] if active else []
        if active and simultaneous:
            frequencies.append(10.0)
        result.append({"kind": "components", "available_sec": (tick + 1) / 100,
                       "components": [{"frequency_log2": f, "power": .02, "rise": True}
                                      for f in frequencies]})
    return result


class ComponentOrderTests(unittest.TestCase):
    def test_simultaneous_order_and_overall_gain_do_not_change_predictions(self):
        rows = frames(simultaneous=True)
        permuted = copy.deepcopy(rows)
        for row in permuted:
            row["components"].reverse()
            for c in row["components"]:
                c["power"] *= .25
        self.assertEqual(evaluate(rows), evaluate(permuted))
        groups = evaluate(rows)["records"]
        self.assertEqual(len(groups), 25)
        self.assertTrue(all(g["component_count"] == 2 for g in groups))

    def test_future_changes_cannot_rewrite_issued_forecasts_or_prefix_records(self):
        rows = frames()
        model = ComponentOrderPredictor()
        prefix = []
        for row in rows[:490]:
            result = model.process(row)
            if result:
                prefix.append(result)
        pending = copy.deepcopy(model.pending)
        full = evaluate(rows)
        self.assertEqual(prefix, [r for r in full["records"] if r["available_sec"] <= 4.9])
        altered = copy.deepcopy(rows[490:])
        for row in altered:
            for c in row["components"]:
                c["frequency_log2"] += 1.2
        first_score = None
        for row in altered:
            result = model.process(row)
            if result and result["score"]:
                first_score = result["score"]
                break
        self.assertEqual(first_score["forecast_issued_sec"], pending["issued_sec"])
        for r in full["records"]:
            if r["score"]:
                self.assertLess(r["score"]["forecast_issued_sec"], r["start_sec"])
        self.assertNotEqual(first_score, next(r["score"] for r in full["records"] if r["available_sec"] > 4.9))

    def test_gap_and_eof_censor_pending_evidence_without_making_a_boundary(self):
        rows = frames()
        model = ComponentOrderPredictor()
        for row in rows[:491]:
            model.process(row)
        self.assertIsNotNone(model.group)
        self.assertIsNotNone(model.pending)
        gap = model.process({"kind": "input_gap", "available_sec": 5.3})
        self.assertIsNotNone(gap["censored_group"])
        self.assertIsNotNone(gap["censored_forecast_issued_sec"])
        self.assertIsNone(model.pending)
        self.assertEqual(len(model.context), 0)
        self.assertIsNone(model.process({"kind": "components", "available_sec": 5.39,
                                         "initial_observation": True, "components": []}))
        partial = evaluate(rows[:491])
        self.assertIsNotNone(partial["right_censored_group"])
        self.assertTrue(all(r["kind"] == "group" for r in partial["records"]))
        self.assertNotIn("closure", partial)

    def test_joint_density_integrates_to_its_time_marginal_and_unit_total(self):
        model = ComponentOrderPredictor()
        for row in frames(8, simultaneous=True):
            model.process(row)
        prediction = model.pending
        frequencies = np.linspace(-15, 33, 4801)
        for time in (-2, math.log2(.3), 0):
            for value in densities(prediction, time, frequencies).values():
                self.assertAlmostEqual(float(np.trapezoid(value["joint"], frequencies)), value["time"], places=7)
        times = np.linspace(-17, 15, 3201)
        values = [densities(prediction, t, [9]) for t in times]
        for name in values[0]:
            self.assertAlmostEqual(float(np.trapezoid([v[name]["time"] for v in values], times)), 1, places=7)

    def test_frequency_order_adds_information_beyond_time_and_single_group(self):
        result = evaluate(frames(65, simultaneous=True))
        scores = [r["score"] for r in result["records"] if r["score"] and r["start_sec"] > 10]
        for name in ("time_only", "one_group", "marginal", "persistence"):
            self.assertGreater(np.mean([s["gain_bits"][name]["joint"] for s in scores]), .3)
        self.assertAlmostEqual(np.mean([s["gain_bits"]["time_only"]["time"] for s in scores]), 0)

    def test_audio_controls_keep_both_interval_and_frequency_order_evidence(self):
        for case, control, part in (("frequency_repeat", "time_only", "frequency_given_time"),
                                    ("simultaneous_repeat", "one_group", "frequency_given_time"),
                                    ("interval_repeat", "one_group", "time")):
            with self.subTest(case=case):
                audio, truth = control_audio(case, 21)
                result = evaluate(FrequencyEventObserver(24_000).process(0, audio))
                summary = summarize(result, truth)
                self.assertTrue(summary["observation_passed"], summary)
                self.assertGreater(summary["windows"]["trained_before_change"]["gain_bits"][control][part], .1)

    def test_history_and_open_group_remain_bounded(self):
        model = ComponentOrderPredictor()
        for row in frames(180, simultaneous=True):
            model.process(row)
            if model.group:
                self.assertLessEqual(len(model.group["components"]), 128)
        self.assertEqual(len(model.history), MEMORY)
        self.assertEqual(len(model.context), 2)
        self.assertLessEqual(len(model.pending["targets"]), MEMORY)

    def test_invalid_input_does_not_mutate_the_model(self):
        model = ComponentOrderPredictor()
        for row in frames(5):
            model.process(row)
        snapshot = copy.deepcopy(model.__dict__)
        for row in ({"kind": "components", "available_sec": 0, "components": []},
                    {"kind": "components", "available_sec": 2, "components": [
                        {"frequency_log2": math.nan, "power": .1, "rise": True}]},
                    {"kind": "components", "available_sec": 2, "components": [
                        {"frequency_log2": 9, "power": 0, "rise": True}]}):
            with self.assertRaises(ValueError):
                model.process(row)
            self.assertEqual(model.__dict__, snapshot)

    def test_two_movements_remain_predictable_as_the_register_changes(self):
        rows = frames(65)
        index = -1
        for row in rows:
            if row["components"]:
                index += 1
                for c in row["components"]:
                    c["frequency_log2"] += .14 * (index // 4)
        scores = {}
        for coordinates in ("absolute", "relative"):
            for count in (2, 3):
                result = evaluate(rows, coordinates, count)
                scores[coordinates, count] = [g["score"] for g in result["records"]
                                               if g["score"] and g["start_sec"] > 10]
        losses = {key: np.mean([s["loss_bits"]["ordered"]["joint"] for s in values])
                  for key, values in scores.items()}
        self.assertLess(losses["relative", 3], losses["absolute", 3] - .5)
        self.assertLess(losses["relative", 3], losses["relative", 2] - .1)
        for name in ("time_only", "one_group", "marginal", "persistence"):
            self.assertGreater(np.mean([s["gain_bits"][name]["joint"] for s in scores["relative", 3]]), .3)

    def test_transposition_moves_every_relative_baseline_and_keeps_the_same_prior(self):
        rows = frames(simultaneous=True)
        shifted = copy.deepcopy(rows)
        for row in shifted:
            for c in row["components"]:
                c["frequency_log2"] += 1
        predictions = []
        for sequence in (rows, shifted):
            model = ComponentOrderPredictor("relative", 3)
            for row in sequence:
                model.process(row)
            predictions.append(model.pending)
        first, second = predictions
        for name in first["weights"]:
            np.testing.assert_allclose(first["weights"][name], second["weights"][name], atol=1e-12)
        for a, b in zip(first["targets"], second["targets"]):
            np.testing.assert_allclose(np.asarray(a["frequencies_log2"]) + 1, b["frequencies_log2"], atol=1e-12)
        x = np.linspace(7, 12, 201)
        time = math.log2(.3)
        a, b = densities(first, time, x), densities(second, time, x + 1)
        for name in a:
            np.testing.assert_allclose(a[name]["joint"] - .05 * normal_density(time, -1, 2) * normal_density(x, 9, 3),
                                       b[name]["joint"] - .05 * normal_density(time, -1, 2) * normal_density(x + 1, 9, 3), atol=1e-10)

    def test_relative_comparison_does_not_weaken_persistence(self):
        predictions = []
        for coordinates in ("absolute", "relative"):
            model = ComponentOrderPredictor(coordinates, 3)
            for row in frames(12, simultaneous=True):
                model.process(row)
            predictions.append(model.pending)
        a, b = [densities(p, math.log2(.27), np.linspace(7, 12, 251))["persistence"] for p in predictions]
        self.assertEqual(a["time"], b["time"])
        np.testing.assert_array_equal(a["joint"], b["joint"])

    def test_relative_frequency_reference_is_frozen_before_the_target_arrives(self):
        rows = frames(30, simultaneous=True)
        model = ComponentOrderPredictor("relative", 3)
        for row in rows[:650]:
            model.process(row)
        issued = model.pending
        frozen = copy.deepcopy(issued)
        result = None
        for row in rows[650:]:
            row = copy.deepcopy(row)
            for c in row["components"]:
                c["frequency_log2"] += 2
            result = model.process(row)
            if result and result["score"]:
                break
        self.assertEqual(issued, frozen)
        self.assertLess(frozen["issued_sec"], result["start_sec"])
        target = result["distribution"]
        expected = densities(frozen, target["interval_log2_sec"], target["frequencies_log2"])
        for name, density in expected.items():
            loss = -float(np.dot(target["mass"], np.log2(density["joint"])))
            self.assertAlmostEqual(result["score"]["loss_bits"][name]["joint"], loss)

    def test_relative_three_group_memory_remains_bounded(self):
        model = ComponentOrderPredictor("relative", 3)
        for row in frames(180, simultaneous=True):
            model.process(row)
        self.assertEqual(len(model.context), 3)
        self.assertEqual(len(model.history), MEMORY)
        self.assertEqual(len(model.pending["targets"]), MEMORY)

    def test_register_timbre_and_gain_controls_remain_observable(self):
        for case in ("frequency_transposed", "simultaneous_transposed", "frequency_quieter",
                     "frequency_timbre", "frequency_random"):
            with self.subTest(case=case):
                audio, truth = control_audio(case, 21)
                observed = FrequencyEventObserver(24_000).process(0, audio)
                result = summarize(evaluate(observed, "relative", 3), truth)
                self.assertTrue(result["observation_passed"], result)


if __name__ == "__main__":
    unittest.main()
