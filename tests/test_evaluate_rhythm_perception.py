"""Guard matched acoustic controls and keep missing evidence distinct from zero."""

import collections
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_rhythm_perception as assay


class PerceptionAssayTests(unittest.TestCase):
    def test_shuffled_control_preserves_intervals_and_event_content(self):
        for seed in (1, 21, 42):
            original = assay.events_for("syncopated", seed)
            shuffled = assay.events_for("shuffled_syncopated", seed)
            self.assertNotEqual(original, shuffled)
            self.assertEqual(original[0]["frame"], shuffled[0]["frame"])
            self.assertEqual(original[-1]["frame"], shuffled[-1]["frame"])
            gaps = [collections.Counter(b["frame"] - a["frame"] for a, b in zip(e, e[1:]))
                    for e in (original, shuffled)]
            self.assertEqual(*gaps)
            content = [[{k: v for k, v in e.items() if k != "frame"} for e in seq]
                       for seq in (original, shuffled)]
            self.assertEqual(*content)

    def test_stimulus_boundaries_and_silent_gaps(self):
        for name in assay.CASES:
            events = assay.events_for(name, 21)
            for a, b in zip(events, events[1:]):
                self.assertLess(a["frame"], b["frame"])
            self.assertTrue(all(0 <= e["frame"] < 39 * assay.FS for e in events))
        for e in assay.events_for("gaps", 21):
            time = e["frame"] / assay.FS
            self.assertFalse(12 <= time < 14 or 22 <= time < 31)
        self.assertEqual(assay.events_for("silence", 21), [])

    def test_no_onsets_leave_phase_statistics_absent(self):
        row = dict(time_sec=1, beat_hz=2, beat_confidence=0, subdivision_confidence=0,
                   measure_confidence=0, onset_time_sec=None, onset_phase=None)
        stats = assay.window_stats([row], 0, 2)
        self.assertIsNone(stats["onset_r1"])
        self.assertIsNone(stats["onset_r2"])
        self.assertEqual(stats["mean_beat_confidence"], 0)
        with self.assertRaises(ValueError):
            assay.window_stats([row], 2, 3)


if __name__ == "__main__":
    unittest.main()
