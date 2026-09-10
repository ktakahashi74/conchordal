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


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("evaluate_beta", SCRIPTS / "evaluate_beta.py")
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

    def test_contour_report_preserves_unknown_evidence_and_signed_scores(self):
        contour = dict.fromkeys(beta.CONTOUR_FLOATS)
        contour.update(type="listener_contour", event="periodic_episode", time_sec=1.0,
                       generated_frame_id=93, onset_sec=.89, periodic_frequency_hz=220.0,
                       periodicity=.99, context_support=3.0, calibration_events=20,
                       error_candidate=False, delta_log2=-.2, loss_bits=-2.0, gain_bits=.8)
        records = report_fixture()
        records.insert(1, contour)
        self.assertEqual(self.parse(records)["record_counts"]["listener_contour"], 1)
        for changes in (dict(periodic_frequency_hz=None), dict(error_candidate=0),
                        dict(error_candidate=True), dict(gain_bits=None), dict(event="closure"),
                        dict(onset_sec=2.0), dict(periodicity=1.1), dict(context_support=None)):
            invalid = copy.deepcopy(records)
            invalid[1].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.parse(invalid)
        gap = dict.fromkeys(beta.CONTOUR_FLOATS)
        gap.update(type="listener_contour", event="input_gap", time_sec=1.0,
                   generated_frame_id=93, missing_start_sec=.5,
                   calibration_events=None, error_candidate=False)
        records[1] = gap
        self.assertEqual(self.parse(records)["record_counts"]["listener_contour"], 1)
        del gap["missing_start_sec"]
        with self.assertRaisesRegex(ValueError, "missing"):
            self.parse(records)

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

    def test_habituation_scans_keep_log2_dimensions_and_state_bounds(self):
        scan = dict(type="habituation_scan", time_sec=1.0, fmin_hz=55.0, bins_per_octave=96, n_bins=2,
                    state_scan=[0.0, 0.5], raw_score_scan=[-0.2, 0.8], eff_score_scan=[-0.2, 0.4])
        records = report_fixture()
        records.insert(1, scan)
        self.assertEqual(self.parse(records)["record_counts"]["habituation_scan"], 1)
        for changes in (dict(n_bins=3), dict(state_scan=[0.0, 1.1]), dict(fmin_hz=0),
                        dict(bins_per_octave=0), dict(state_scan=[False, 0.5]),
                        dict(raw_score_scan=[]), dict(eff_score_scan=[0.0, float("nan")])):
            broken = copy.deepcopy(records)
            broken[1].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.parse(broken)

    def test_local_prediction_matches_keep_request_and_observation_times_distinct(self):
        record = dict(type="local_prediction_match", voice_id=1, sample_rate=1000,
                      window_frames=10, forecast_observed_frame=10, requested_frame=13,
                      target_start_frame=110, target_end_frame=120, issued_step=1, target_step=11,
                      completed_before_issue=0, history_weight=[0.0]*3,
                      recurrence=[0.1]*3, history=[0.2]*3, mixed=[0.1]*3, observed=[0.15]*3,
                      issued_features=[1.0]+[0.0]*56)
        records = report_fixture()
        records.insert(1, record)
        self.assertEqual(self.parse(records)["record_counts"]["local_prediction_match"], 1)
        missing = copy.deepcopy(records)
        missing[1]["issued_features"] = None
        self.assertEqual(self.parse(missing)["record_counts"]["local_prediction_match"], 1)
        for change in (dict(requested_frame=9), dict(requested_frame=120),
                       dict(target_step=10), dict(target_end_frame=119), dict(window_frames=0),
                       dict(history_weight=[1.1]*3), dict(issued_features=[0.0]*57),
                       dict(issued_features=[1.0]+[0.0]*27+[1.1]*8+[0.0]*21),
                       dict(issued_features=[1.0]+[0.0]*35+[0.6]*21)):
            broken = copy.deepcopy(records)
            broken[1].update(change)
            with self.assertRaises(ValueError):
                self.parse(broken)

    def test_local_prediction_errors_require_completed_observation_windows(self):
        record = dict(type="local_prediction_error", voice_id=1, sample_rate=1000,
                      issued=0,
                      observed_from_frame=10, observed_through_frame=20, window_frames=10,
                      horizon_frames=[0, 100, 200, 500, 1000, 2000, 4000],
                      completed=[1, 0, 0, 0, 0, 0, 0],
                      recurrence_squared_error=[[.1]*3]+[[0.]*3]*6,
                      history_squared_error=[[.2]*3]+[[0.]*3]*6,
                      mixed_squared_error=[[.1]*3]+[[0.]*3]*6)
        records = report_fixture()
        records.insert(1, record)
        self.assertEqual(self.parse(records)["record_counts"]["local_prediction_error"], 1)
        pending = copy.deepcopy(records)
        pending[1].update(issued=1, observed_through_frame=10, completed=[0]*7,
                          recurrence_squared_error=[[0.]*3]*7, history_squared_error=[[0.]*3]*7,
                          mixed_squared_error=[[0.]*3]*7)
        self.assertEqual(self.parse(pending)["record_counts"]["local_prediction_error"], 1)
        for changes in (dict(observed_through_frame=21), dict(window_frames=0),
                        dict(completed=[0]*7), dict(completed=[2]+[0]*6),
                        dict(completed=[True]+[0]*6), dict(horizon_frames=[0]*7),
                        dict(mixed_squared_error=[[.1]*3]*7),
                        dict(history_squared_error=[[-.1]*3]+[[0.]*3]*6)):
            broken = copy.deepcopy(records)
            broken[1].update(changes)
            with self.assertRaises(ValueError):
                self.parse(broken)

    def test_participation_reports_keep_observations_distinct_from_predictions(self):
        outcome = dict(type="participation_outcome", voice_id=1, sample_rate=1000,
                       issued_frame=10, onset_frame=12, forecast_observed_frame=10,
                       target_start_frame=20, target_end_frame=30,
                       status="observed", pred_continuation_habitat_band_energy=[0.1]*3,
                       observed_start_frame=20, observed_end_frame=30, observed_habitat_band_energy=[0.2]*3)
        context = dict(type="participation_context", voice_id=1, sample_rate=1000,
                       energy_prediction_model="local_history_mix",
                       onset_frame=12, forecast_observed_frame=10, observed_through_frame=110,
                       target_start_frames=[20, 100], target_end_frames=[30, 110], status="observed",
                       pred_external_band_energy=[[0.1]*3]*2, observed_external_band_energy=[[0.2]*3]*2,
                       memory_external_band_energy=[[0.2]*3]*2,
                       decision_external_history=dict(ages_sec=[.125, .25, .5, 1, 2, 4, 8, 16], post_order=12,
                                                      known_band_rms_by_age=[[.1]*3]*8,
                                                      known_coverage_by_age=[.5]*8))
        records = report_fixture()
        records[1:1] = [outcome, context]
        parsed = self.parse(records)
        self.assertEqual(parsed["record_counts"]["participation_outcome"], 1)
        self.assertEqual(parsed["record_counts"]["participation_context"], 1)
        for index, changes in ((1, dict(observed_start_frame=21)), (1, dict(status="input_gap")),
                               (1, dict(pred_continuation_habitat_band_energy=[True, 0, 0])),
                               (1, dict(issued_frame=13)), (1, dict(sample_rate=0)),
                               (2, dict(observed_through_frame=109)), (2, dict(memory_external_band_energy=None)),
                               (2, dict(energy_prediction_model="unknown")),
                               (2, dict(target_end_frames=[30, 111])), (2, dict(target_start_frames=[20.0, 100])),
                               (2, dict(observed_external_band_energy=[[0.0]*2]*2)),
                               (2, dict(pred_external_band_energy=[[-0.1]*3]*2))):
            broken = copy.deepcopy(records)
            broken[index].update(changes)
            with self.subTest(index=index, changes=changes), self.assertRaises(ValueError):
                self.parse(broken)
        missing = copy.deepcopy(records)
        missing[1].update(status="end_of_input", observed_start_frame=None,
                          observed_end_frame=None, observed_habitat_band_energy=None)
        missing[2].update(status="outside_observed_history", observed_external_band_energy=None,
                          memory_external_band_energy=None)
        self.parse(missing)
        for changes in (dict(ages_sec=[1, .5]), dict(post_order=True),
                        dict(known_band_rms_by_age=[[0]*3]*7), dict(known_coverage_by_age=[1.1]*8),
                        dict(known_band_rms_by_age=[[False, 0, 0]]*8), dict(ages_sec="ages")):
            broken = copy.deepcopy(records)
            broken[2]["decision_external_history"].update(changes)
            with self.subTest(history=changes), self.assertRaises(ValueError):
                self.parse(broken)


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

    def test_sample12_windows_follow_current_interventions_and_reject_drift(self):
        source = self.directory / "12_emergence_and_resolution.rhai"
        original = (SCRIPTS.parent / "samples" / source.name).read_text()
        source.write_text(original)

        def fake_render(command, _cwd, _log, _timeout):
            report = Path(command[command.index("--report") + 1])
            audio = Path(command[command.index("-o") + 1])
            fixture = report_fixture()
            # Sample both sides of the resolution boundary and the exclusive end.
            times = [2.3, 11.69, 11.7, 14.99, 15.0, 20.29, 20.3, 23.59, 23.6, 24.9, 30.19, 30.2]
            listener = [dict(fixture[1], time_sec=t, tension_level=t) for t in times]
            confidence = dict(fixture[2], sample_count=len(listener))
            timings = [dict(fixture[3], frame_idx=i, time_sec=float(i)) for i in range(40)]
            report.write_text("".join(json.dumps(r) + "\n" for r in
                                      [fixture[0], *listener, confidence, *timings, fixture[-1]]))
            with wave.open(str(audio), "wb") as stream:
                stream.setparams((1, 2, 10, 0, "NONE", "not compressed"))
                stream.writeframes(struct.pack("<400h", *([100] * 400)))
            return {"status": "ok", "returncode": 0}

        output = self.directory / "current"
        args = ["--samples", str(source), "--seeds", "21", "--skip-build", "--binary", sys.executable]
        with patch.object(beta, "snapshot", return_value={"source_files": {}}), \
                patch.object(beta, "execute", side_effect=fake_render):
            self.assertEqual(beta.main([*args, "--output", str(output)]), 0)
        metrics = json.loads((output / "00_12_emergence_and_resolution/seed-21/metrics.json").read_text())
        windows = metrics["report"]["script_operation_windows"]["windows"]
        self.assertEqual([(r["start_sec"], r["end_sec"]) for r in windows],
                         [(2.3, 11.7), (11.7, 15.0), (15.0, 20.3), (20.3, 23.6), (24.9, 30.2)])
        self.assertEqual([r["count"] for r in windows], [2] * 5)
        self.assertAlmostEqual(windows[3]["stats"]["tension_level"]["mean"], (20.3 + 23.59) / 2)

        source.write_text(original.replace("wait(9.4)", "wait(9.5)"))
        failed = self.directory / "drifted"
        with patch.object(beta, "snapshot", return_value={"source_files": {}}), \
                patch.object(beta, "execute") as execute:
            self.assertEqual(beta.main([*args, "--output", str(failed)]), 1)
            execute.assert_not_called()
        self.assertEqual(json.loads((failed / "manifest.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
