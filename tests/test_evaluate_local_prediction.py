"""Future targets, local order, and bounded state for the envelope comparison."""

import copy
import math
from pathlib import Path
import pickle
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_local_prediction import (CENTERS_LOG2, HORIZON, LAGS, MEMORY, MIN_TRAIN,
                                       LocalPredictor, control_audio, evaluate, summarize)
from evaluate_component_relations import ComponentRelations
from evaluate_frequency_events import FrequencyEventObserver


def observations(count=220):
    for tick in range(count):
        amplitude = np.zeros(len(CENTERS_LOG2))
        amplitude[30] = .08 + .06 * math.sin(math.tau * tick / 20)
        amplitude[55] = .05 + .04 * math.sin(math.tau * tick / 10 + .7)
        yield (tick + 1) / 10, amplitude


class LocalPredictionTests(unittest.TestCase):
    def test_local_order_predicts_phase_beyond_the_same_unordered_history(self):
        model, rows = LocalPredictor(), []
        for time, amplitude in observations(240):
            rows.extend(model.process(time, amplitude))
        summary = summarize({"records": rows, "right_censored_forecasts": 3}, 24)
        self.assertTrue(summary["normal_screen_passed"], summary)
        self.assertGreater(summary["relative_error_reduction"]["unordered"], .8)

    def test_future_targets_do_not_enter_training_or_rewrite_forecasts(self):
        model = LocalPredictor()
        data = list(observations(90))
        rows = []
        for time, amplitude in data[:70]:
            rows.extend(model.process(time, amplitude))
        first = next(row for row in rows if row["kind"] == "forecast")
        self.assertEqual(first["training_examples"], MIN_TRAIN)
        self.assertAlmostEqual(first["available_sec"], (LAGS + HORIZON + MIN_TRAIN - 1) / 10)
        pending = copy.deepcopy(model.pending)
        snapshot = copy.deepcopy(rows)
        for time, amplitude in data[70:]:
            rows.extend(model.process(time, amplitude * 2))
        self.assertEqual(snapshot, rows[:len(snapshot)])
        score = next(row for row in rows[len(snapshot):] if row["kind"] == "score")
        self.assertEqual(score["origin_sec"], pending[0]["origin_sec"])
        actual = np.asarray(score["actual"])
        self.assertAlmostEqual(score["squared_error"]["local"],
                               float(np.sum((pending[0]["prediction"]["local"] - actual) ** 2)))
        self.assertTrue(all(row["origin_sec"] < row["available_sec"] for row in rows if row["kind"] == "score"))

    def test_silence_and_constant_energy_do_not_fabricate_predictive_gain(self):
        for value in (0, .08):
            model, rows = LocalPredictor(), []
            for time, amplitude in observations(140):
                rows.extend(model.process(time, np.full_like(amplitude, value)))
            summary = summarize({"records": rows, "right_censored_forecasts": 3}, 14)
            self.assertFalse(summary["normal_screen_passed"])
            self.assertTrue(all(value < 1e-12 for value in summary["squared_error_sum"].values()))

    def test_input_gap_censors_and_forgets_completed_examples(self):
        model = LocalPredictor()
        for time, amplitude in observations(90):
            model.process(time, amplitude)
        gap = model.process(10, None)
        self.assertEqual(gap[0]["censored_forecasts"], HORIZON)
        self.assertFalse(model.training)
        self.assertFalse(model.pending)
        self.assertEqual(model.process(10.18, np.ones(len(CENTERS_LOG2)) * .1), [])
        for tick in range(1, LAGS + HORIZON + MIN_TRAIN - 1):
            rows = model.process(10.18 + tick / 10, np.ones(len(CENTERS_LOG2)) * .1)
            if tick < LAGS + HORIZON + MIN_TRAIN - 2:
                self.assertFalse(rows)
        self.assertEqual(rows[0]["kind"], "forecast")

    def test_audio_packetization_and_prefix_leave_issued_predictions_identical(self):
        audio, _ = control_audio("independent", 1)
        audio = audio[:7 * 24_000]
        expected = evaluate(audio, 24_000)["records"]
        front, relations, model, rows = FrequencyEventObserver(24_000), ComponentRelations(), LocalPredictor(), []
        for start in range(0, len(audio), 347):
            for frame in front.process(start, audio[start:start + 347]):
                report = relations.process(frame)
                if report is not None:
                    rows.extend(model.process(report["available_sec"], relations.fast))
        self.assertEqual(expected, rows)
        prefix = evaluate(audio[:round(6.417 * 24_000)], 24_000)["records"]
        self.assertEqual(prefix, [r for r in expected if r["available_sec"] <= 6.417])

    def test_invalid_inputs_are_atomic_and_all_histories_are_bounded(self):
        model = LocalPredictor()
        for time, amplitude in observations(240):
            model.process(time, amplitude)
        self.assertEqual(len(model.training), MEMORY)
        self.assertEqual(len(model.recent), LAGS)
        self.assertEqual(len(model.pending), HORIZON)
        state = pickle.dumps(model)
        for time, amplitude in ((23, np.zeros(len(CENTERS_LOG2))),
                                (25, np.zeros(len(CENTERS_LOG2) - 1)),
                                (25, np.full(len(CENTERS_LOG2), math.nan)),
                                (25, -np.ones(len(CENTERS_LOG2)))):
            with self.assertRaises(ValueError):
                model.process(time, amplitude)
            self.assertEqual(state, pickle.dumps(model))

    def test_shared_neighborhood_keeps_time_order_and_observation_boundaries(self):
        model, rows = LocalPredictor(2, "neighborhood"), []
        position = np.arange(len(CENTERS_LOG2))
        for tick in range(240):
            amplitude = .08 + .06 * np.cos(math.tau * position / 20) * np.sin(math.tau * tick / 16)
            rows.extend(model.process((tick + 1) / 10, amplitude))
        summary = summarize({"records": rows, "right_censored_forecasts": 3}, 24)
        self.assertGreater(summary["relative_error_reduction"]["unordered"], .5)
        self.assertGreater(summary["relative_error_reduction"]["one_frame"], .5)
        for name in ("local", "unordered"):
            self.assertEqual(model.sums[name]["x"].shape, (len(CENTERS_LOG2), 10))
        model.process(25, None)
        self.assertEqual((model.mode, model.lags), ("neighborhood", 2))
        self.assertFalse(model.training)
        self.assertEqual(model.process(25.18, np.zeros(len(CENTERS_LOG2))), [])

    def test_short_channel_mode_preserves_the_fair_same_target_baselines(self):
        short, long = LocalPredictor(2), LocalPredictor(20)
        first, second = [], []
        for time, amplitude in observations(220):
            first.extend(short.process(time, amplitude))
            second.extend(long.process(time, amplitude))
        first = {r["available_sec"]: r for r in first if r["kind"] == "forecast" and r["training_examples"] == MEMORY}
        for r in second:
            if r["kind"] == "forecast" and r["training_examples"] == MEMORY:
                for name in ("marginal", "persistence", "one_frame"):
                    np.testing.assert_allclose(r["prediction"][name], first[r["available_sec"]]["prediction"][name], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
