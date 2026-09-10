"""Distribution identity, settling and causal evidence of acoustic-state comparisons."""

import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_acoustic_state import (
    AcousticStateObserver, KERNEL_WIDTH, LOG2_HZ, STATE_FRAMES, STRIDE_SEC,
    SpectralObserver, assess, evaluate_audio, state_audio,
)


def point(index):
    mass = np.zeros(len(LOG2_HZ))
    mass[index] = 1
    return mass


class AcousticStateTests(unittest.TestCase):
    def test_incremental_kernel_matches_independent_full_pairwise_recalculation(self):
        model = AcousticStateObserver()
        masses = []
        for i in range(105):
            mass = point(110 + i % 7)
            masses.append(mass)
            row = model.process((i + 1) * STRIDE_SEC, mass)
            if not row["ready"]:
                continue
            roots = np.sqrt(masses[-2 * STATE_FRAMES:])
            distances = np.sum((roots[:, None, :] - roots[None, :, :]) ** 2, axis=2) / 2
            kernel = np.exp(-distances / (2 * KERNEL_WIDTH ** 2))
            np.testing.assert_allclose(model.kernel, kernel, atol=1e-14)
            n = STATE_FRAMES
            statistic = np.mean(kernel[:n, :n]) + np.mean(kernel[n:, n:]) - 2 * np.mean(kernel[:n, n:])
            self.assertAlmostEqual(row["distance"]["distribution"], math.sqrt(max(0, statistic) / 2))
        self.assertEqual(len(model.roots), 2 * STATE_FRAMES)

    def test_identical_means_do_not_hide_alternation_versus_simultaneous_occupancy(self):
        masses = [point(i) for i in (110, 140, 170, 200)]
        mixture = np.mean(masses, axis=0)
        model = AcousticStateObserver()
        rows = []
        for i in range(5 * STATE_FRAMES):
            mass = masses[i % 4] if i < 3 * STATE_FRAMES else mixture
            rows.append(model.process((i + 1) * STRIDE_SEC, mass))
        exact = rows[4 * STATE_FRAMES - 1]
        self.assertAlmostEqual(exact["distance"]["mean_spectrum"], 0)
        self.assertGreater(exact["distance"]["distribution"], .7)
        self.assertTrue(any("distribution" in r["candidates"] for r in rows))
        self.assertFalse(any("mean_spectrum" in r["candidates"] for r in rows))

    def test_order_within_the_two_windows_is_deliberately_unidentified(self):
        a = [point(110 + i % 4 * 20) for i in range(2 * STATE_FRAMES)]
        b = a[:STATE_FRAMES][::-1] + a[STATE_FRAMES:][::-1]
        outcomes = []
        for sequence in (a, b):
            observer = AcousticStateObserver()
            for i, mass in enumerate(sequence):
                row = observer.process((i + 1) * STRIDE_SEC, mass)
            outcomes.append(row)
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertAlmostEqual(outcomes[0]["distance"]["distribution"], 0)

    def test_brief_excursion_is_ignored_and_persistent_change_needs_observed_settling(self):
        for duration in (4, 65):
            model, rows = AcousticStateObserver(), []
            for i in range(190):
                mass = point(190 if 60 <= i < 60 + duration else 110)
                rows.append(model.process((i + 1) * STRIDE_SEC, mass))
            expected = [60 * STRIDE_SEC, (60 + duration) * STRIDE_SEC] if duration == 65 else []
            self.assertTrue(assess(rows, expected)["passed"])
            if duration == 65:
                summary = assess(rows, expected)
                self.assertEqual(len(summary["settled_sec"]["distribution"]), 2)
                for start, end in zip(summary["candidate_sec"]["distribution"], summary["settled_sec"]["distribution"]):
                    self.assertGreater(end, start)
                self.assertFalse(summary["active_at_last_ready_window"]["distribution"])

    def test_gaps_censor_active_differences_instead_of_inventing_settling(self):
        model = AcousticStateObserver()
        for i in range(100):
            row = model.process((i + 1) * STRIDE_SEC, point(110 if i < 60 else 190))
            if "distribution" in row["candidates"]:
                break
        self.assertTrue(row["active_difference"]["distribution"])
        state = copy.deepcopy(model)
        for invalid_time, mass in ((0, point(110)), (math.nan, point(110)),
                                   (100, np.zeros(len(LOG2_HZ))), (100, [1])):
            with self.assertRaises(ValueError):
                model.process(invalid_time, mass)
            self.assertEqual(model.last_time, state.last_time)
            np.testing.assert_array_equal(model.kernel, state.kernel)
        gap_time = model.last_time + 3 * STRIDE_SEC
        gap = model.process(gap_time, None)
        self.assertTrue(gap["input_gap"])
        self.assertIn("distribution", gap["censored_difference"])
        self.assertFalse(gap["settled"])
        for i in range(39):
            self.assertFalse(model.process(gap_time + (i + 1) * STRIDE_SEC, point(190))["ready"])
        resumed = model.process(gap_time + 40 * STRIDE_SEC, point(190))
        self.assertTrue(resumed["ready"])
        self.assertFalse(resumed["candidates"])

    def test_block_calibration_does_not_treat_a_common_allocation_as_exceptional(self):
        model = AcousticStateObserver()
        blocks = (110, 110, 110, 111, 111, 110, 111, 111, 111, 111)
        for i in range(40):
            row = model.process((i + 1) * STRIDE_SEC, point(blocks[i // 4]))
        self.assertEqual(model.partitions.shape, (126, 10))
        self.assertTrue(np.all(model.partitions.sum(axis=1) == 5))
        for threshold in row["thresholds"].values():
            self.assertGreaterEqual(threshold["enter"], threshold["leave"])
        self.assertGreater(row["distance"]["distribution"], .25)
        self.assertGreater(row["thresholds"]["distribution"]["enter"], .25)
        self.assertLessEqual(row["distance"]["distribution"], row["thresholds"]["distribution"]["enter"] + 1e-10)

    def test_initial_motion_is_not_a_transition_from_an_unobserved_stable_state(self):
        model, rows = AcousticStateObserver(), []
        for i in range(160):
            rows.append(model.process((i + 1) * STRIDE_SEC, point(100 + i)))
        self.assertFalse(any("combined" in row["candidates"] for row in rows))
        self.assertEqual(rows[-1]["state"]["combined"], "unestablished")
        self.assertIsNone(rows[-1]["active_difference"]["combined"])
        for i in range(160, 225):
            row = model.process((i + 1) * STRIDE_SEC, point(259))
        self.assertEqual(row["state"]["combined"], "stable")
        later = [model.process((i + 1) * STRIDE_SEC, point(130)) for i in range(225, 275)]
        self.assertEqual(sum("combined" in r["candidates"] for r in later), 1)

    def test_frequency_transport_respects_distance_on_the_log_frequency_axis(self):
        model = AcousticStateObserver()
        for i in range(60):
            model.process((i + 1) * STRIDE_SEC, point(120))
        for i in range(60, 80):
            row = model.process((i + 1) * STRIDE_SEC, point(140))
        self.assertAlmostEqual(row["distance"]["frequency_transport"], 20 / 48)

    def test_audio_prefix_and_gain_changes_preserve_observations(self):
        fs = 24_000
        t = np.arange(8 * fs) / fs
        audio = .12 * (np.sin(math.tau * 277.3 * t) + .7 * np.sin(math.tau * 411.7 * t))
        rows = evaluate_audio(audio, fs)
        cutoff = round(6.731 * fs)
        prefix = evaluate_audio(audio[:cutoff], fs)
        self.assertEqual(prefix, [r for r in rows if r["available_sec"] <= cutoff / fs])
        scaled = evaluate_audio(.4 * audio, fs)
        self.assertEqual([r["candidates"] for r in rows], [r["candidates"] for r in scaled])
        self.assertFalse(assess(rows, [6])["passed"])
        self.assertTrue(assess(rows, [])["passed"])

    def test_changed_state_retains_motion_without_an_added_attack_or_gain_step(self):
        moving, truth = state_audio("moving_texture", 21)
        brief, _ = state_audio("moving_excursion", 21)
        steady, _ = state_audio("steady", 21)
        static, _ = state_audio("texture_change", 21)
        fs = 24_000
        np.testing.assert_allclose(moving[:12 * fs], steady[:12 * fs], atol=1e-15, rtol=0)
        np.testing.assert_allclose(brief[:12 * fs], steady[:12 * fs], atol=1e-15, rtol=0)
        np.testing.assert_allclose(brief[13 * fs:], steady[13 * fs:], atol=1e-15)
        self.assertEqual(truth["sustained_transition_sec"], [12])
        motion = []
        for audio in (moving, static):
            spectra = SpectralObserver(fs).process(0, audio)
            late = [r["mass"] for r in spectra if 16 <= r["available_sec"] <= 28]
            self.assertTrue(all(mass is not None for mass in late))
            steps = np.linalg.norm(np.diff(np.sqrt(late), axis=0), axis=1) / math.sqrt(2)
            motion.append(np.quantile(steps, .75))
        self.assertGreater(motion[0], .25)
        self.assertLess(motion[1], .02)
        rms = [np.sqrt(np.mean(audio[16 * fs:28 * fs] ** 2)) for audio in (moving, steady)]
        self.assertLess(abs(20 * np.log10(rms[0] / rms[1])), .5)
        boundary = moving[12 * fs - 1:12 * fs + round(.06 * fs) + 1]
        self.assertLess(np.max(np.abs(np.diff(boundary))), .06)


if __name__ == "__main__":
    unittest.main()
