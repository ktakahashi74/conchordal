"""Control identity, causal density forecasts, and missing-evidence semantics."""

import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_interval_order import IntervalPredictor, PATTERNS, STEP_SEC, confirm_attacks, densities, evaluate, interval_audio, summarize


def observations(intervals, gap_at=None):
    ticks = [50]
    for interval in intervals:
        ticks.append(ticks[-1] + round(interval / STEP_SEC))
    selected = set(ticks)
    return [{"time_sec": tick * STEP_SEC, "observed_band": 1 if tick in selected else 0}
            for tick in range(ticks[-1] + 40) if gap_at is None or not gap_at[0] <= tick < gap_at[1]]


class IntervalOrderTests(unittest.TestCase):
    def test_stimuli_preserve_duration_interval_multiset_and_prechange_pcm(self):
        baseline, truth = interval_audio("repeat", 21)
        original = np.diff([round(r["onset_sec"] * 24_000) for r in truth["events"]])
        for case in ("reordered", "rotated", "shuffled", "quieter", "timbre", "overlap"):
            audio, spec = interval_audio(case, 21)
            intervals = np.diff([round(r["onset_sec"] * 24_000) for r in spec["events"]])
            self.assertEqual(len(audio), len(baseline))
            np.testing.assert_array_equal(np.sort(intervals), np.sort(original))
            self.assertEqual(len(spec["events"]), len(truth["events"]))
            if case not in ("rotated", "shuffled"):
                split = round(spec["split_sec"] * 24_000)
                np.testing.assert_array_equal(np.rint(audio[:split] * 32767), np.rint(baseline[:split] * 32767))
        reordered = truth["after_interval_frames"]
        before = truth["before_interval_frames"]
        self.assertTrue(all(reordered != before[i:] + before[:i] for i in range(len(before))))
        changed, spec = interval_audio("reordered", 21)
        self.assertFalse(np.array_equal(changed, baseline))
        self.assertAlmostEqual(np.sum(changed ** 2), np.sum(baseline ** 2), places=9)

    def test_context_adds_information_beyond_a_marginal_and_last_interval(self):
        rows = evaluate(observations([.3, .3, .6, .6] * 35))
        summary = summarize(rows)
        gains = summary["windows"]["all_after_10_sec"]["gain_bits"]
        self.assertGreater(gains["marginal"], .4)
        self.assertGreater(gains["one_interval"], .4)
        self.assertGreater(gains["persistence"], 0)
        self.assertFalse(summary["error_candidates_sec"])

    def test_different_length_patterns_preserve_the_comparison_and_overlap_seam(self):
        for pattern in PATTERNS:
            with self.subTest(pattern=pattern):
                baseline, truth = interval_audio("repeat", 121003, pattern)
                changed, spec = interval_audio("reordered", 121003, pattern)
                overlap, mixed = interval_audio("overlap", 121003, pattern)
                split = round(truth["split_sec"] * 24_000)
                np.testing.assert_array_equal(baseline[:split], changed[:split])
                np.testing.assert_array_equal(baseline[:split], overlap[:split])
                self.assertFalse(np.array_equal(baseline, changed))
                self.assertFalse(np.array_equal(baseline, overlap))
                self.assertAlmostEqual(float(np.sum(baseline ** 2)), float(np.sum(changed ** 2)), places=9)
                before = np.diff([round(e["onset_sec"] * 24_000) for e in truth["events"]])
                after = np.diff([round(e["onset_sec"] * 24_000) for e in spec["events"]])
                np.testing.assert_array_equal(np.sort(before), np.sort(after))
                np.testing.assert_array_equal(np.sort(before),
                                              np.sort(np.rint(np.asarray(truth["intervals_sec"]) * 24_000)))
                self.assertEqual(len(truth["events"]), 16 * len(PATTERNS[pattern][0]) + 1)
                self.assertEqual(truth["events"][-1]["onset_sec"], spec["events"][-1]["onset_sec"])
                for first, second in zip(truth["events"], mixed["events"]):
                    self.assertEqual(first["onset_sec"], second["onset_sec"])

    def test_attack_confirmation_rejects_steady_and_falling_energy_without_future_audio(self):
        fs = 24_000
        time = np.arange(round(.4 * fs)) / fs
        audio = .2 * np.sin(math.tau * 300 * time) * ((time >= .12) & (time < .27))
        candidates = [{"time_sec": t, "observed_band": 1} for t in (.14, .22, .28, .34)]
        rows = confirm_attacks(candidates, audio, fs)
        self.assertEqual([r["observed_band"] for r in rows], [1, 0, 0, 0])
        prefix = confirm_attacks(candidates[:2], audio[:round(.22 * fs)], fs)
        self.assertEqual(prefix, rows[:2])
        with self.assertRaises(ValueError):
            confirm_attacks(candidates, audio[:round(.22 * fs)], fs)

    def test_steady_two_tone_energy_ripples_do_not_create_attacks(self):
        fs = 24_000
        time = np.arange(fs) / fs
        audio = .2 * (np.sin(math.tau * 277.3 * time) + np.sin(math.tau * 277.3 * math.sqrt(2) * time))
        candidates = [{"time_sec": tick / 100, "observed_band": 1} for tick in range(20, 100)]
        rows = confirm_attacks(candidates, audio, fs)
        self.assertFalse(any(r["observed_band"] for r in rows))

    def test_forecast_densities_integrate_to_one_in_the_declared_log_coordinate(self):
        rows = evaluate(observations([.3, .3, .6, .6] * 12))
        forecast = rows["right_censored_forecast"]
        grid = np.linspace(-15, 15, 10_001)
        values = [densities(forecast, 2 ** x) for x in grid]
        for name in values[0]:
            self.assertAlmostEqual(np.trapezoid([v[name] for v in values], grid), 1, places=7)

    def test_forecasts_use_only_the_prefix_and_never_score_the_eof(self):
        same = observations([.3, .3, .6, .6] * 16)
        prefix = same[:1703]
        short = evaluate(prefix)
        full = evaluate(same)
        self.assertEqual(short["events"], [r for r in full["events"] if r["available_sec"] <= prefix[-1]["time_sec"]])
        self.assertIsNotNone(short["right_censored_forecast"])
        self.assertEqual(short["right_censored_forecast"], short["events"][-1]["next_forecast"])
        for row in full["events"]:
            if row["score"] is not None:
                self.assertLess(row["score"]["forecast"]["issued_sec"], row["available_sec"])

    def test_gap_censors_the_forecast_and_restarts_interval_context(self):
        stream = observations([.3, .3, .6, .6] * 15, gap_at=(1700, 1770))
        rows = evaluate(stream)
        gap = next(r for r in rows["events"] if r["kind"] == "gap")
        self.assertIsNotNone(gap["censored_forecast"])
        post = next(r for r in rows["events"] if r["kind"] == "onset" and r["available_sec"] > gap["available_sec"])
        self.assertIsNone(post["interval_sec"])
        self.assertIsNone(post["score"])
        self.assertIsNone(post["next_forecast"])

    def test_missing_onsets_fail_observation_checks_and_silence_stays_unscored(self):
        _, spec = interval_audio("repeat", 1)
        silence = evaluate([{"time_sec": i * STEP_SEC, "observed_band": 0} for i in range(1000)])
        self.assertFalse(summarize(silence, spec)["observation_passed"])
        self.assertIsNone(silence["right_censored_forecast"])
        _, empty = interval_audio("silence", 1)
        self.assertTrue(summarize(silence, empty)["observation_passed"])

    def test_unavailable_change_judgments_remain_distinct_from_negative_ones(self):
        observed = evaluate(observations([.3, .4, .5] * 8))
        truth = {"split_sec": 5, "duration_sec": 11, "events": []}
        report = summarize(observed, truth)
        self.assertFalse(report["error_candidates_sec"])
        for window in report["windows"].values():
            self.assertEqual(window["eligible_error_checks"] + window["unavailable_error_checks"],
                             window["scored_intervals"])
        window = report["windows"]["first_six_seconds_after_change"]
        self.assertGreater(window["scored_intervals"], 0)
        self.assertEqual(window["eligible_error_checks"], 0)
        self.assertEqual(window["unavailable_error_checks"], window["scored_intervals"])

    def test_invalid_input_does_not_mutate_state_and_memory_is_bounded(self):
        model = IntervalPredictor()
        for row in observations([.3, .3, .6, .6] * 70):
            model.process(row["time_sec"], bool(row["observed_band"]))
        self.assertEqual(len(model.history), 128)
        self.assertEqual(len(model.losses), 128)
        before = copy.deepcopy(model.__dict__)
        for time, fired in ((model.last_observation, True), (math.nan, False), (-1, False), (1000, 3)):
            with self.assertRaises(ValueError):
                model.process(time, fired)
            self.assertEqual(model.__dict__, before)


if __name__ == "__main__":
    unittest.main()
