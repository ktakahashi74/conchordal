"""Verify causal forecasts and distinguish repetition from onset concentration."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_rhythm_recurrence import RecurrencePredictor


class RecurrenceTests(unittest.TestCase):
    def test_current_event_does_not_change_its_own_forecast(self):
        a, b = RecurrencePredictor(), RecurrencePredictor()
        for i in range(900):
            band = 1 if i % 47 == 0 else 0
            self.assertEqual(a.step(band), b.step(band))
        left, right = a.step(0), b.step(2)
        for key in ("event_probability", "null_event_probability", "recurrence_weight",
                    "best_period_sec", "recent_onsets"):
            self.assertEqual(left[key], right[key])
        self.assertNotEqual(left["gain_bits"], right["gain_bits"])

    def test_alternating_marks_with_omissions_have_predictive_gain(self):
        model = RecurrencePredictor()
        gain = 0
        for i in range(2400):
            phase = i % 94
            band = 1 if phase == 0 else 2 if phase == 70 else 0
            result = model.step(band)
            if i >= 1500:
                gain += result["gain_bits"]
        self.assertGreater(gain, 30)
        self.assertAlmostEqual(result["best_period_sec"], 94 * model.dt, delta=.04)

    def test_silence_has_no_predictive_evidence_for_a_period(self):
        model = RecurrencePredictor()
        for _ in range(700):
            result = model.step(0)
        self.assertEqual(result["recent_onsets"], 0)
        self.assertIsNone(result["best_period_sec"])
        self.assertAlmostEqual(result["gain_bits"], 0)
        self.assertAlmostEqual(result["recurrence_weight"], .5)

    def test_switching_keeps_probability_mass_and_forecasts_before_update(self):
        a = RecurrencePredictor(adaptation="switching")
        b = RecurrencePredictor(adaptation="switching")
        for i in range(950):
            band = 1 + (i // 47) % 2 if i % 47 == 0 else 0
            self.assertEqual(a.step(band), b.step(band))
            self.assertAlmostEqual(sum(a.weights), 1)
            self.assertTrue(all(0 <= w <= 1 for w in a.weights))
        left, right = a.step(0), b.step(3)
        for key in ("event_probability", "recurrence_weight", "best_period_sec"):
            self.assertEqual(left[key], right[key])
        self.assertNotEqual(a.weights, b.weights)


if __name__ == "__main__":
    unittest.main()
