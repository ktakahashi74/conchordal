"""First-passage timing, causal learning and censoring of spectral forecasts."""

import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_spectral_change import (
    CASES, CONTROLS, HORIZON_STEPS, LOG2_HZ, MEMORY, STRIDE_SEC,
    ChangePredictor, assess, evaluate_audio, summarize,
)


def point(index):
    mass = np.zeros(len(LOG2_HZ))
    mass[index] = 1
    return mass


class SpectralChangeTests(unittest.TestCase):
    def test_first_passage_bins_and_direction_are_scored_at_the_fixed_deadline(self):
        model, records = ChangePredictor(), []
        for index in range(1, 41):
            records.extend(model.process(index * STRIDE_SEC, point(130 if index < 16 else 155)))
        scores = {round(row["origin_sec"] / STRIDE_SEC): row for row in records if row["kind"] == "score"}
        self.assertEqual(scores[3]["target"], 9)
        for origin, expected in ((4, 8), (10, 5), (13, 2)):
            row = scores[origin]
            self.assertEqual(row["target"], expected)
            self.assertEqual(row["first_change_step"], 16 - origin)
            self.assertAlmostEqual(row["time_sec"], (origin + 12) * STRIDE_SEC)
            self.assertAlmostEqual(row["center_delta_oct"], 25 / 48)
            for loss in row["loss_bits"].values():
                self.assertAlmostEqual(loss["joint"], loss["time"] + loss["direction"])
                self.assertEqual(len(loss["occurrence"]), 3)
        self.assertIsNone(scores[3]["loss_bits"]["ordered"]["direction"])
        self.assertEqual(len(model.pending), HORIZON_STEPS[-1])

    def test_forecasts_and_resolved_prefix_do_not_depend_on_future_changes(self):
        def run(change_at):
            model, records = ChangePredictor(), []
            for index in range(1, 65):
                records.extend(model.process(index * STRIDE_SEC, point(130 if index < change_at else 155)))
            return records
        left, right = run(40), run(100)
        cutoff = 39 * STRIDE_SEC
        self.assertEqual([r for r in left if r["time_sec"] <= cutoff],
                         [r for r in right if r["time_sec"] <= cutoff])
        for row in left:
            if row["kind"] != "forecast":
                continue
            for p in row["probabilities"].values():
                self.assertAlmostEqual(sum(p), 1)
                self.assertTrue(all(probability > 0 for probability in p))
                cumulative = np.cumsum(np.array(p[:9]).reshape(3, 3).sum(axis=1))
                self.assertTrue(np.all(np.diff(cumulative) >= 0))
                self.assertLess(cumulative[-1], 1)
        first_changed = next(r for r in left if r["kind"] == "score" and r["target"] != 9)
        self.assertGreaterEqual(first_changed["time_sec"], 40 * STRIDE_SEC)

    def test_missing_or_unavailable_input_censors_even_an_early_known_crossing(self):
        for gap in (False, True):
            model = ChangePredictor()
            for index in range(1, 8):
                rows = model.process(index * STRIDE_SEC, point(130 if index < 5 else 155))
                self.assertFalse(any(r["kind"] == "score" for r in rows))
            self.assertTrue(any(p["target"] is not None for p in model.pending))
            rows = (model.process(10 * STRIDE_SEC, point(155)) if gap else
                    model.process(8 * STRIDE_SEC, None))
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(r["kind"] == "censored" for r in rows))
            self.assertEqual(len(model.history), 0)
            self.assertEqual(len(model.pending), 0)
            self.assertLess(len(model.recent), 2)

    def test_spectral_redistribution_can_change_without_moving_the_center(self):
        narrow, wide = np.zeros(len(LOG2_HZ)), np.zeros(len(LOG2_HZ))
        narrow[130], narrow[150] = .5, .5
        wide[110], wide[170] = .5, .5
        model, records = ChangePredictor(), []
        for index in range(1, 32):
            records.extend(model.process(index * STRIDE_SEC, narrow if index < 12 else wide))
        changes = [r for r in records if r["kind"] == "score" and r["target"] != 9]
        self.assertTrue(changes)
        self.assertTrue(all(r["target"] % 3 == 1 for r in changes))
        self.assertTrue(all(abs(r["center_delta_oct"]) < 1e-12 for r in changes))

    def test_audio_prefixes_and_level_changes_preserve_forecasts_and_no_eof_ending(self):
        fs = 24_000
        t = np.arange(4 * fs) / fs
        audio = .12 * (np.sin(math.tau * 277.3 * t) + .7 * np.sin(math.tau * 401.9 * t))
        rows, available, pending = evaluate_audio(audio, fs)
        prefix, _, prefix_pending = evaluate_audio(audio[:round(2.937 * fs)], fs)
        self.assertEqual(prefix, [r for r in rows if r["time_sec"] <= 2.937])
        self.assertEqual(prefix_pending, 12)
        summary = summarize(rows, available, pending, 0)
        self.assertEqual(summary["changed_origins"], 0)
        self.assertEqual(summary["pending_at_eof"], 12)
        self.assertEqual(summary["forecast_count"], summary["resolved_count"] + 12)
        quiet, _, _ = evaluate_audio(.4 * audio, fs)
        self.assertEqual([r["target"] for r in rows if r["kind"] == "score"],
                         [r["target"] for r in quiet if r["kind"] == "score"])
        a = next(r for r in reversed(rows) if r["kind"] == "forecast")
        b = next(r for r in reversed(quiet) if r["kind"] == "forecast")
        for name in a["probabilities"]:
            np.testing.assert_allclose(a["probabilities"][name], b["probabilities"][name], atol=1e-13)

    def test_memory_is_bounded_and_invalid_input_does_not_advance_it(self):
        model = ChangePredictor()
        for index in range(1, 370):
            model.process(index * STRIDE_SEC, point(130 + index % 9))
        self.assertEqual(len(model.history), MEMORY)
        self.assertEqual(len(model.pending), 12)
        reference = copy.deepcopy(model)
        for time, mass in ((0, point(130)), (math.nan, point(130)),
                           (400, np.zeros(len(LOG2_HZ))), (400, np.ones(3))):
            with self.assertRaises(ValueError):
                model.process(time, mass)
            self.assertEqual(model.last_time, reference.last_time)
            self.assertEqual(len(model.pending), len(reference.pending))
            for name, p in model.forecast()["probabilities"].items():
                np.testing.assert_array_equal(p, reference.forecast()["probabilities"][name])

    def test_controls_and_predicted_utility_remain_separate_from_execution(self):
        rows = []
        for case in (*CASES, *CONTROLS):
            summary = {"scored_origins": 150, "changed_origins": 80,
                       "target_counts": [0, 80, 0, 0, 0, 0, 0, 0, 0, 70],
                       "center_delta_range_oct": [.01, .02]}
            if case in ("steady", "gain_only"):
                summary["changed_origins"] = 0
            if case == "falling":
                summary["center_delta_range_oct"] = [-.02, -.01]
            rows.append({"group": "controlled", "case": case, "seed": 1, "summary": summary})
        self.assertTrue(assess(rows, [1])["controls_passed"])
        rows[0]["summary"]["scored_origins"] = 0
        steady = next(r for r in rows if r["case"] == "steady")
        steady["summary"]["changed_origins"] = 1
        self.assertFalse(assess(rows, [1])["controls_passed"])
        self.assertEqual(assess(rows, [1])["runtime_attachment"], "research_only")
        with self.assertRaises(ValueError):
            assess(rows[:-1], [1])


if __name__ == "__main__":
    unittest.main()
