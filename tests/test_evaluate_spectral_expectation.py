"""Independent input, forecasting and evidence boundaries of the spectral assay."""

import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_spectral_expectation import (
    CASES, LOG2_HZ, MEMORY, SpectralObserver, SpectralPredictor,
    assess_manifest, relative_mass,
)


class SpectralExpectationTests(unittest.TestCase):
    def test_overlapping_audio_has_causal_features_independent_of_call_boundaries(self):
        for fs in (24_000, 44_100, 48_000):
            t = np.arange(round(1.317 * fs)) / fs
            audio = .12 * np.sin(math.tau * 273.7 * t)
            audio += .1 * np.sin(math.tau * 411.3 * t) * (t >= .4)
            whole = SpectralObserver(fs).process(0, audio)
            observer = SpectralObserver(fs)
            chunks = []
            for start in range(0, len(audio), 511):
                chunks.extend(observer.process(start, audio[start:start + 511]))
            self.assertEqual(len(whole), len(chunks))
            for a, b in zip(whole, chunks):
                np.testing.assert_array_equal(a.pop("mass"), b.pop("mass"))
                self.assertEqual(a, b)
            previous = 0
            for row in whole:
                self.assertGreater(row["window_start_sec"], previous)
                previous = row["available_sec"]
            cutoff = round(.837 * fs)
            prefix = SpectralObserver(fs).process(0, audio[:cutoff])
            self.assertEqual([r["available_sec"] for r in prefix],
                             [r["available_sec"] for r in whole if r["available_sec"] <= cutoff / fs])
            self.assertEqual(len(observer.window), observer.window_size)
            self.assertLess(observer.pending_count, observer.stride)

    def test_scaling_audible_audio_does_not_change_spectral_mass(self):
        fs = 24_000
        t = np.arange(fs) / fs
        audio = .15 * (np.sin(math.tau * 287.3 * t) + .7 * np.sin(math.tau * 463.1 * t))
        a = SpectralObserver(fs).process(0, audio)
        b = SpectralObserver(fs).process(0, .31 * audio)
        for left, right in zip(a, b):
            self.assertAlmostEqual(np.sum(left["mass"]), 1)
            np.testing.assert_allclose(left["mass"], right["mass"], atol=2e-15)
            self.assertGreater(left["rms"], right["rms"])

    def test_missing_input_resets_context_and_requires_a_full_new_window(self):
        fs = 24_000
        t = np.arange(fs) / fs
        audio = .12 * np.sin(math.tau * 287.3 * t)
        observer, predictor = SpectralObserver(fs), SpectralPredictor()
        for row in observer.process(0, audio[:17_513]):
            predictor.step(row["mass"])
        rows = observer.process(fs, audio)
        self.assertEqual(rows[0], {"kind": "input_gap", "available_sec": 1,
                                  "missing_start_sec": 17_513 / fs, "mass": None})
        self.assertGreaterEqual(rows[1]["window_start_sec"], 1)
        self.assertIsNone(predictor.step(rows[0]["mass"]))
        self.assertIsNone(predictor.forecast())
        self.assertIsNone(predictor.step(rows[1]["mass"]))
        self.assertIsNone(predictor.step(rows[2]["mass"]))
        self.assertIsNotNone(predictor.step(rows[3]["mass"]))

    def test_unavailable_audio_and_invalid_input_do_not_invent_observations(self):
        fs = 24_000
        t = np.arange(fs) / fs
        for audio in (np.zeros(fs), .12 * np.sin(math.tau * 10_000 * t)):
            rows = SpectralObserver(fs).process(0, audio)
            self.assertTrue(rows)
            self.assertTrue(all(r["mass"] is None for r in rows))
        observer = SpectralObserver(fs)
        observer.process(0, np.ones(100))
        for start, bad in ((99, [0]), (100, [float("nan")]), (100, [[0]]), (-1, [0])):
            with self.assertRaises(ValueError):
                observer.process(start, bad)
            self.assertEqual(observer.next_sample, 100)
            self.assertEqual(observer.pending_count, 100)
        self.assertEqual(observer.process(999, []), [])
        self.assertEqual(observer.next_sample, 100)

    def test_relative_mass_preserves_distribution_under_log_frequency_translation(self):
        mass = np.zeros(len(LOG2_HZ))
        mass[110], mass[137], mass[155] = .4, .35, .25
        shifted = np.roll(mass, 21)
        a = relative_mass(mass, float(mass @ LOG2_HZ))
        b = relative_mass(shifted, float(shifted @ LOG2_HZ))
        self.assertAlmostEqual(np.sum(a), 1)
        np.testing.assert_allclose(a, b, atol=3e-14)

    def test_forecasts_use_only_past_targets_and_memory_remains_bounded(self):
        model = SpectralPredictor()
        for index in range(360):
            mass = np.zeros(len(LOG2_HZ))
            mass[140 + (0, 9, 3, 15, 7, 19, 11, 22)[index % 8]] = 1
            model.step(mass)
        forecast = model.forecast()
        for name in ("ordered", "one_frame", "marginal", "persistence"):
            self.assertAlmostEqual(np.sum(forecast[name]), 1)
            self.assertTrue(np.all(forecast[name] > 0))
        self.assertEqual(forecast["history_events"], MEMORY)
        self.assertEqual(len(model.history), MEMORY)
        alternative = copy.deepcopy(model)
        left, right = np.zeros(len(LOG2_HZ)), np.zeros(len(LOG2_HZ))
        left[140], right[222] = 1, 1
        a, b = model.step(left), alternative.step(right)
        self.assertEqual(a["support"], b["support"])
        self.assertEqual(a["history_events"], b["history_events"])
        self.assertNotEqual(a["loss_bits"], b["loss_bits"])
        self.assertLess(a["loss_bits"]["ordered"], b["loss_bits"]["ordered"])
        before = len(model.history), len(model.recent)
        for bad in (np.ones(3), np.zeros(len(LOG2_HZ)), np.full(len(LOG2_HZ), math.nan)):
            with self.assertRaises(ValueError):
                model.step(bad)
            self.assertEqual((len(model.history), len(model.recent)), before)

    def test_successful_execution_does_not_convert_failed_baselines_into_acceptance(self):
        manifest = {"status": "complete", "seeds": [1],
                    "cases": [{"seed": 1, "duration_sec": d, "case": c}
                              for d in (.22, .50) for c in CASES],
                    "samples": [{"source": "test.wav", "evaluation": {
                        "scored_windows": 100,
                        "gain_bits": {"marginal": .1, "one_frame": -.02, "persistence": -2}}}]}
        result = assess_manifest(manifest)
        self.assertEqual(result["normal_order_advantage_conditions"], 0)
        self.assertFalse(result["normal_samples"][0]["beats_all_baselines"])
        self.assertEqual(result["runtime_attachment"], "research_only")
        manifest["cases"].pop()
        with self.assertRaises(ValueError):
            assess_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
