"""Checks of the I11-1 §5.9 distribution report; no renderer or Rust build required."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "summarize_i11_representative_gap", SCRIPTS / "summarize_i11_representative_gap.py")
gap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gap)

REP = [1.0 - k / 32 for k in range(16)]


def decision(overlap):
    candidate = dict(overlap=overlap, external_energy=[[0.02, 0.01, 0.005]] * 16)
    return dict(type="participation_decision", voice_id=3, now=100, skipped=False,
                footprint_source="body", footprint_power=REP, selected_offset=0,
                candidates=[None, None, candidate] + [None] * 20, own_band_energy=[1.0, 0.5, 0.25],
                coupling=1.0, overlap_sensitivity=0.8)


def row(energies, status="projected"):
    return dict(key=dict(source_id=3, tone_id=1, onset=400, issued_at=100, kick=1.0),
                status=status, decided_at=100, d_rep=300.0, coherent_energies=energies)


def summarize(decisions, rows):
    with tempfile.TemporaryDirectory() as tmp:
        case = Path(tmp)
        (case / "report.jsonl").write_text("".join(json.dumps(d) + "\n" for d in decisions))
        (case / "gap.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return gap.summarize_case(case)


class RepresentativeGapTest(unittest.TestCase):
    def test_gaps_follow_the_normalized_actual_energies(self):
        actual = [2.0 * p for p in REP]
        actual[5] = 0.0
        summary = summarize([decision(0.1)], [row(actual)])
        body = summary["by_footprint_source"]["body"]
        self.assertAlmostEqual(body["power_gap"]["max"], REP[5])
        self.assertGreater(body["overlap_gap"]["max"], 0.0)
        self.assertEqual(summary["counts"], {"projected": 1})

    def test_unsupported_bins_and_missing_decisions_are_counted_not_scored(self):
        unsupported = [1.0] * 15 + [None]
        summary = summarize([decision(None)], [
            row(unsupported), row(None, status="no_decision"), row([2.0 * p for p in REP])])
        self.assertEqual(summary["counts"], {
            "excluded_unsupported_bin": 1, "no_decision": 1, "overlap_not_applied": 1, "projected": 2})
        body = summary["by_footprint_source"]["body"]
        self.assertEqual(body["power_gap"], {"count": 1, "median": 0.0, "p95": 0.0, "max": 0.0})
        self.assertNotIn("overlap_gap", body)

    def test_known_silence_on_both_sides_is_no_gap(self):
        silent = decision(0.0)
        silent["footprint_power"] = [0.0] * 16
        summary = summarize([silent], [row([0.0] * 16)])
        self.assertEqual(summary["by_footprint_source"]["body"]["power_gap"]["max"], 0.0)


if __name__ == "__main__":
    unittest.main()
