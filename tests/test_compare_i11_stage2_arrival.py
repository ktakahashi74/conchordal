"""The arrival comparison rejects non-arrival changes before a choice split."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_i11_stage2_arrival import compare


class ArrivalComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.on_report = self.root / "on.jsonl"
        self.off_report = self.root / "off.jsonl"
        self.on_wav = self.root / "on.wav"
        self.off_wav = self.root / "off.wav"
        self.on_wav.write_bytes(b"on")
        self.off_wav.write_bytes(b"off")
        candidates = [dict(offset=i - 2, at=float(i), displacement_sq=0.0,
                           context_distance=None, pred_external_band_energy=None,
                           overlap=None, external_energy=[], cost=0.1) for i in range(23)]
        candidates[2]["cost"] = 0.0
        self.off = dict(type="participation_decision", now=0, voice_id=1,
                        candidates=candidates, selected_offset=0, selected_at=2.0,
                        skipped=False, coupling=1.0)
        self.on = copy.deepcopy(self.off)
        self.on.update(arrival_state="known", arrival_groups=1,
                       selected_offset=1, selected_at=3.0)
        for i, candidate in enumerate(self.on["candidates"]):
            probability = float(i == 3)
            candidate.update(arrival_probability=probability,
                             arrival_cost=4.0 * (1.0 - probability))
            candidate["cost"] += candidate["arrival_cost"]

    def check(self):
        self.on_report.write_text(json.dumps(self.on) + "\n")
        self.off_report.write_text(json.dumps(self.off) + "\n")
        return compare(self.on_report, self.off_report, self.on_wav, self.off_wav)

    def test_known_arrival_choice_effect(self):
        row = self.check()
        self.assertEqual(row["status"], "arrival_choice_effect")
        self.assertEqual(row["eligible_known"], 1)

    def test_nonarrival_input_change_is_rejected(self):
        self.on["memory"] = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        row = self.check()
        self.assertEqual(row["status"], "nonarrival_inputs_differ")
        self.assertIn("memory", row["differing_fields"])

    def test_unknown_cannot_explain_choice_effect(self):
        self.on["arrival_state"] = "no_eligible_group"
        for candidate in self.on["candidates"]:
            candidate["cost"] -= candidate.pop("arrival_cost")
            candidate.pop("arrival_probability")
        row = self.check()
        self.assertEqual(row["status"], "invalid_or_unproved_split")


if __name__ == "__main__":
    unittest.main()
