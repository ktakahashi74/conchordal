import csv
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_memory_constraints import evaluate


class MemoryConstraintTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.responses, self.models = root / "responses.csv", root / "models.csv"
        self.rows = []
        for subject in (1, 2, 7, 25):
            for speed in range(1, 7):
                duration = (25, 50, 75)[(speed - 1) % 3] / 1000
                cycle = 10 if speed <= 3 else 20
                baseline = .5 + speed / 10
                for condition, response, rt, include in [
                    *[(4, 1, baseline + offset, 1) for offset in (-.05, 0, .05)],
                    *[(2, 1, baseline + (cycle + subject + 2 + offset) * duration, 1)
                      for offset in (-.1, 0, .1)],
                    (4, 1, 99., 0), (1, 2, float("nan"), 1)]:
                    self.rows.append({"subj": subject, "block": speed, "speed": speed,
                                      "condi": condition, "RTs": rt, "response": response, "correct": include})
        with self.responses.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(self.rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(self.rows)
        with self.models.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=("i", "label", "alphabet_size", "tone_len_ms", "mean", "error_count"))
            writer.writeheader()
            for count in (10, 20):
                for duration in (25, 50, 75):
                    writer.writerow({"i": 1, "label": "Known synthetic lag", "alphabet_size": count,
                                     "tone_len_ms": duration, "mean": 3.5, "error_count": 0})

    def test_subject_and_trial_filters_and_per_block_baseline(self):
        result = evaluate(self.responses, self.models)
        self.assertEqual(result["counts"]["raw_trials"], 192)
        self.assertEqual(result["counts"]["after_subject_exclusions"], 96)
        self.assertEqual(result["counts"]["after_trial_flags"], 84)
        self.assertEqual(result["counts"]["retained_randreg_hits"], 36)
        self.assertEqual([row["subject"] for row in result["participant_values"]], [1, 2])
        for row in result["conditions"]:
            self.assertAlmostEqual(row["mean_participant_lag_tones"], 3.5)
            self.assertAlmostEqual(row["mean_participant_lag_sec"], 3.5 * row["tone_ms"] / 1000)
        self.assertAlmostEqual(result["archived_model_comparisons"][0]["rmse_lag_tones"], 0.)
        json.dumps(result, allow_nan=False)

    def test_same_tone_lag_does_not_mean_same_elapsed_time(self):
        result = evaluate(self.responses, self.models)
        tone = result["contrasts"]["equal_cycle_duration_20x25_minus_10x50_tones"]
        seconds = result["contrasts"]["equal_cycle_duration_20x25_minus_10x50_seconds"]
        self.assertAlmostEqual(tone["mean_lag_difference"], 0.)
        self.assertAlmostEqual(seconds["mean_lag_difference"], -.0875)

    def test_incomplete_design_is_not_silently_averaged(self):
        remaining = [row for row in self.rows if not (row["subj"] == 2 and row["speed"] == 6)]
        with self.responses.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(remaining[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(remaining)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            evaluate(self.responses, self.models)

    def test_trial_order_does_not_change_participant_summary(self):
        original = evaluate(self.responses, self.models)
        random.Random(15).shuffle(self.rows)
        with self.responses.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(self.rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(self.rows)
        shuffled = evaluate(self.responses, self.models)
        for before, after in zip(original["conditions"], shuffled["conditions"]):
            self.assertAlmostEqual(before["mean_participant_lag_tones"], after["mean_participant_lag_tones"])
        self.assertEqual(original["counts"], shuffled["counts"])


if __name__ == "__main__":
    unittest.main()
