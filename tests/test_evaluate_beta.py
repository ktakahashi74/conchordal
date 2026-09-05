"""Artifact and failure-path checks; no renderer or Rust build required."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave


SPEC = importlib.util.spec_from_file_location(
    "evaluate_beta", Path(__file__).resolve().parents[1] / "scripts" / "evaluate_beta.py")
beta = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(beta)


def report_fixture():
    listener = {key: 0.5 for key in beta.LISTENER_FLOATS}
    listener.update({key: 0 for key in beta.LISTENER_INTS})
    listener.update(type="listener_state", time_sec=1.0)
    summary = {key: 0.0 for key in beta.RHYTHM_FIELDS}
    summary.update({key: None for key in beta.RHYTHM_OPTIONAL})
    summary.update(type="rhythm_summary", population_id=None, onset_count=0)
    timings = [dict(type="hop_timing", frame_idx=i, time_sec=float(i), elapsed_us=float(i * 10 + 10),
                    analysis_wait_us=1.0, listener_wait_us=2.0, hop_budget_us=1_000_000.0,
                    audio_output="no_device", underrun_frames_total=None) for i in range(4)]
    return [dict(type="meta", seed=21, hop_timing_scope="process_hop excluding timing write"), listener,
            dict(type="listener_confidence_summary", window_start_sec=0.0, window_end_sec=3.0,
                 sample_count=1, beat_confidence_peak=0.5, beat_confidence_late_mean=0.5),
            *timings, summary]


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "report.jsonl"

    def parse(self, records):
        self.path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        return beta.parse_report(self.path, 21, 2)

    def test_warmup_percentiles_and_null_underruns(self):
        metrics = self.parse(report_fixture())
        timing = metrics["hop_timing"]
        self.assertEqual(timing["excluded_count"], 2)
        self.assertEqual(timing["retained_count"], 2)
        self.assertEqual(timing["stats_us"]["elapsed_us"]["p95"], 39.5)
        self.assertEqual(timing["stats_us"]["elapsed_us"]["p99"], 39.9)
        self.assertIsNone(timing["underrun_frames_total"])
        self.assertEqual(metrics["listener"]["stats"]["tension_level"]["count"], 1)

    def test_rejects_missing_seed_scope_and_nullable_fields(self):
        for record_index, field in [(0, "seed"), (0, "hop_timing_scope"), (3, "underrun_frames_total"),
                                    (-1, "ioi_cv")]:
            records = report_fixture()
            del records[record_index][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "missing"):
                self.parse(records)

    def test_rejects_seed_mismatch_missing_tail_and_timing_gaps(self):
        original = report_fixture()
        variants = []
        wrong_seed = copy.deepcopy(original)
        wrong_seed[0]["seed"] = 1
        variants.extend([wrong_seed, original[:-1], original[:2] + original[3:], original[:3] + original[4:]])
        missing_hop = copy.deepcopy(original)
        missing_hop[4]["frame_idx"] = 7
        variants.append(missing_hop)
        confidence = copy.deepcopy(original)
        confidence[2]["sample_count"] = 2
        variants.append(confidence)
        for records in variants:
            with self.subTest(records=records), self.assertRaises(ValueError):
                self.parse(records)

    def test_rejects_duplicate_nonfinite_and_blank_json_lines(self):
        for line in ['{"type":"meta","seed":21,"seed":21}',
                     '{"type":"meta","seed":21,"extra":NaN}',
                     '{"type":"meta","seed":21,"extra":1e999}', "", "{broken"]:
            self.path.write_text(line + "\n", encoding="utf-8")
            with self.subTest(line=line), self.assertRaises(ValueError):
                beta.parse_report(self.path, 21, 2)

    def test_rejects_inconsistent_population_summary(self):
        records = report_fixture()
        records[-1]["onset_count"] = 1
        with self.assertRaisesRegex(ValueError, "onset counts"):
            self.parse(records)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def wav(self, samples, channels=1):
        path = self.directory / "test.wav"
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(channels)
            stream.setsampwidth(2)
            stream.setframerate(8000)
            stream.writeframes(struct.pack("<" + "h" * len(samples), *samples))
        return path

    def test_pcm_metrics_use_symmetric_renderer_endpoints(self):
        result = beta.wav_metrics(self.wav([-32767, 32767, 0, 1]))
        self.assertEqual(result["frames"], 4)
        self.assertEqual(result["clipping_fraction"], 0.5)
        self.assertEqual(result["silence_fraction"], 0.5)
        self.assertAlmostEqual(result["duration_sec"], 4 / 8000)
        self.assertAlmostEqual(result["peak"], 32767 / 32768)
        self.assertFalse(result["all_silent"])

    def test_rejects_stereo_empty_and_truncated_wav(self):
        for samples, channels in [([1, 2], 2), ([], 1)]:
            with self.subTest(channels=channels), self.assertRaises(ValueError):
                beta.wav_metrics(self.wav(samples, channels))
        path = self.wav([1, 2, 3, 4])
        path.write_bytes(path.read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError, "truncated"):
            beta.wav_metrics(path)

    def test_source_allowlist_excludes_credentials_and_build_outputs(self):
        for name in [".env", "src/.env", ".claude/private.rs", ".codex/config.toml", "target/debug/main.rs",
                     "src/key.pem", "tests/__pycache__/secret.py", "samples/.private.rhai", "../src/main.rs"]:
            with self.subTest(name=name):
                self.assertFalse(beta.source_allowed(Path(name)))
        for name in ["Cargo.lock", "src/main.rs", "tests/scripts/probe.rhai", "scripts/evaluate_beta.py"]:
            self.assertTrue(beta.source_allowed(Path(name)))

    def test_nonzero_exit_timeout_and_limiter_override(self):
        failure = beta.execute([sys.executable, "-c", "raise SystemExit(7)"], self.directory,
                               self.directory / "failure.log", 2)
        self.assertEqual((failure["status"], failure["returncode"]), ("error", 7))
        timeout = beta.execute([sys.executable, "-c", "import time; time.sleep(5)"], self.directory,
                               self.directory / "timeout.log", 0.05)
        self.assertEqual(timeout["status"], "timeout")
        with patch.dict(os.environ, {"CONCHORDAL_LIMITER": "hard-clip"}):
            result = beta.execute([sys.executable, "-c", "import os; assert 'CONCHORDAL_LIMITER' not in os.environ"],
                                  self.directory, self.directory / "env.log", 2)
        self.assertEqual(result["status"], "ok")

    def test_existing_campaign_directory_is_not_modified(self):
        sentinel = self.directory / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(SystemExit):
            beta.main(["--output", str(self.directory)])
        self.assertEqual(sentinel.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
