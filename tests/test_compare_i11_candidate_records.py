"""Checks of the I11-1 §5.4(b) candidate-record comparison; no renderer required."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "compare_i11_candidate_records", SCRIPTS / "compare_i11_candidate_records.py")
cmp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cmp)


def record(issued_at, energy=1.0, processing_us=10, tone_id=None):
    return dict(type="body_candidate_energy", source_id=1, source_generation=0, tone_id=tone_id,
                issued_at=issued_at, decision_at=issued_at, scope="onset", candidates=[energy],
                processing_us=processing_us)


def compare(base, new):
    with tempfile.TemporaryDirectory() as tmp:
        paths = [Path(tmp) / "base.jsonl", Path(tmp) / "new.jsonl"]
        for path, records in zip(paths, (base, new)):
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return cmp.compare(*paths)


class CompareCandidateRecordsTest(unittest.TestCase):
    def test_wall_clock_is_ignored_and_one_sided_records_are_counted(self):
        result = compare([record(0, processing_us=3), record(512), record(1024, tone_id=7)],
                         [record(0, processing_us=9), record(1024, tone_id=7), record(1536)])
        self.assertEqual((result["matched"], result["identical"], result["differing"]), (2, 2, 0))
        self.assertEqual((result["base_only"], result["new_only"]), (1, 1))

    def test_a_changed_field_is_reported(self):
        result = compare([record(0, energy=1.0)], [record(0, energy=1.5)])
        self.assertEqual(result["differing"], 1)
        self.assertEqual(result["examples"][0]["fields"], ["candidates"])

    def test_a_duplicate_key_is_an_error(self):
        with self.assertRaises(ValueError):
            compare([record(0), record(0)], [record(0)])


if __name__ == "__main__":
    unittest.main()
