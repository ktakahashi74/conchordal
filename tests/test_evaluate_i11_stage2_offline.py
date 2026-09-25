"""Fail-closed checks for the registered I11-2 offline audit."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_i11_stage2_offline as offline


class OfflineAuditTests(unittest.TestCase):
    def test_imports_stage2_acquisition_helpers(self):
        self.assertEqual(offline.acquire.__file__,
                         str(offline.REGISTRATION / "acquire.py"))

    def test_artifact_requires_declared_name_bytes_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "001-test.jsonl"
            path.write_bytes(b'{"type":"onset"}\n')
            entry = {"path": path.name, "bytes": path.stat().st_size,
                     "sha256": offline.digest(path)}
            self.assertEqual(offline.require_artifact(root, entry, path.name), path)
            with self.assertRaisesRegex(ValueError, "declaration differs"):
                offline.require_artifact(root, {**entry, "path": "other.jsonl"}, path.name)
            path.write_bytes(b'{"type":"skip"}\n')
            with self.assertRaisesRegex(ValueError, "missing or corrupt"):
                offline.require_artifact(root, entry, path.name)

    def test_incomplete_ledger_cannot_pass_as_offline_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provenance = root / "acquisition-provenance.json"
            provenance.write_text("{}\n")
            (root / "runs.json").write_text(json.dumps({
                "schema": "conchordal/i11-stage2-runs/1",
                "provenance_sha256": offline.digest(provenance), "runs": [],
            }))
            with self.assertRaisesRegex(ValueError, "offline acquisition incomplete: 0/96"):
                offline.offline_runs(root, {"run_order": []}, {}, {}, {})

    def test_complete_ledger_checks_every_registered_command_and_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provenance = root / "acquisition-provenance.json"
            provenance.write_text("{}\n")
            rows, runs, registered = [], [], {}
            configs = {"none": {"path": str(root / "config.toml")}}
            binaries = {"original_none": {"conchordal-render": {"path": str(root / "render")}}}
            for sequence in range(1, 97):
                case = f"case-{sequence}"
                row = {"sequence": sequence, "phase": "offline_original_none", "case": case,
                       "config": "none", "binary": "original_none", "mode": "render", "rep": 0}
                rows.append(row)
                registered[f"{case}.rhai"] = root / f"{case}.rhai"
                paths = offline.acquire.output_paths(root, row)
                artifacts = {}
                for kind, path in paths.items():
                    path.write_bytes(f"{sequence}-{kind}\n".encode())
                    artifacts[kind] = {"path": path.name, "sha256": offline.digest(path),
                                       "bytes": path.stat().st_size}
                runs.append({"sequence": sequence, "registration": row, "exit": 0,
                             "command": offline.acquire.command(row, registered, configs,
                                                                 binaries, paths),
                             "artifacts": artifacts})
            ledger = {"schema": "conchordal/i11-stage2-runs/1",
                      "provenance_sha256": offline.digest(provenance), "runs": runs}
            (root / "runs.json").write_text(json.dumps(ledger))
            self.assertEqual(len(offline.offline_runs(
                root, {"run_order": rows}, registered, configs, binaries)), 96)
            runs[12]["command"][0] = "wrong-binary"
            (root / "runs.json").write_text(json.dumps(ledger))
            with self.assertRaisesRegex(ValueError, "run command differs: 13"):
                offline.offline_runs(root, {"run_order": rows}, registered, configs, binaries)
            runs[12]["command"][0] = binaries["original_none"]["conchordal-render"]["path"]
            (root / "runs.json").write_text(json.dumps(ledger))
            paths["report"].write_bytes(b"corrupt\n")
            with self.assertRaisesRegex(ValueError, "artifact missing or corrupt"):
                offline.offline_runs(root, {"run_order": rows}, registered, configs, binaries)

    def test_legacy_context_exclusions_do_not_hide_other_record_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right = root / "left.jsonl", root / "right.jsonl"
            left.write_text(json.dumps({"type": "participation_context", "footprint_source": "body",
                                        "processing_us": 1}) + "\n")
            right.write_text(json.dumps({"type": "participation_context", "footprint_source": "proxy",
                                         "processing_us": 2}) + "\n")
            self.assertTrue(offline.exact_report_pair(
                left, right, ("participation_context",), True)["passed"])
            left.write_text(json.dumps({"type": "onset", "footprint_source": "body"}) + "\n")
            right.write_text(json.dumps({"type": "onset", "footprint_source": "proxy"}) + "\n")
            self.assertFalse(offline.exact_report_pair(left, right, ("onset",), True)["passed"])

    def test_strict_prefix_distinguishes_signed_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right = root / "left.jsonl", root / "right.jsonl"
            row = {"type": "participation_decision", "now": 1, "voice_id": 2,
                   "due_frame": 0.0, "candidates": [None] * 23}
            row["candidates"][2] = {"offset": 0, "at": 0.0}
            left.write_text(json.dumps(row) + "\n")
            row["due_frame"] = -0.0
            right.write_text(json.dumps(row) + "\n")
            result = offline.strict_prefix_inputs(
                left, right, 0, ("due_frame",), ("at",))
            self.assertEqual(result["field"], "due_frame")
            self.assertFalse(result["passed"])

    def test_candidate_payload_compares_float_bits_after_processing_time_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right = root / "left.jsonl", root / "right.jsonl"
            row = {"type": "body_candidate_energy", "source_id": 1,
                   "source_generation": 2, "tone_id": 3, "issued_at": 4,
                   "decision_at": 5, "scope": "body", "processing_us": 10,
                   "power": [0.0]}
            left.write_text(json.dumps(row) + "\n")
            row["processing_us"] = 20
            row["power"] = [-0.0]
            right.write_text(json.dumps(row) + "\n")
            result = offline.compare_candidates_bits(left, right)
            self.assertEqual(result["matched"], 1)
            self.assertEqual(result["differing"], 0)
            self.assertEqual(result["bit_differing"], 1)

    def test_duplicate_body_footprint_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.jsonl"
            row = {"type": "body_footprint", "identity": {"source_id": 1},
                   "received_at": 5}
            report.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate body footprint key"):
                offline.footprint_records_strict(report)


if __name__ == "__main__":
    unittest.main()
