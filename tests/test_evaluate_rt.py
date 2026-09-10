"""Profile artifact and campaign failure checks without Rust or an audio device."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "evaluate_rt", Path(__file__).resolve().parents[1] / "scripts" / "evaluate_rt.py")
rt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rt)


def profile_fixture(mode="device", report=False, dcc=0.0):
    device = mode == "device"
    hops = [dict(frame_idx=i, time_sec=i * .25, elapsed_us=100.0 + i, alive_voice_count=4,
                 analysis_wait_us=10.0, listener_wait_us=0.0,
                 landscape_update_us=1.0, population_us=20.0, reports_us=5.0,
                 render_route_us=40.0, post_render_us=10.0,
                 synthesis_us=30.0, rendered_tone_count=4,
                 worker_allocations=dict(count=i + 1, bytes=(i + 1) * 16),
                 underrun_frames_total=(9 if i < 12 else 99) if device else None)
            for i in range(20)]
    return dict(schema_version=2, scope="process_hop", allocation_scope="worker thread Rust requests",
                seed=1, report_enabled=report, dcc_coupling_strength=dcc, listener_enabled=report or dcc > 0,
                sample_rate=4, hop_size=1, hop_budget_us=250000.0, allocation_instrumented=True,
                truncated=False, dropped_hops=0, hop_capacity=100000, audio_output="device" if device else "no_device",
                audio=dict(backend="fixture", device_name="fixture device", sample_rate=4, channels=2,
                           ring_capacity_frames=8, callback_count=50, callback_frames_total=25,
                           underrun_frames_total=99, callback_errors_total=0) if device else None,
                summary=dict(hop_count=len(hops), elapsed_p99_us=118.81, elapsed_max_us=119.0, over_budget_hops=0),
                hops=hops)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "profile.json"

    def parse(self, data, mode="device", warmup=1, duration=2):
        self.path.write_text(json.dumps(data), encoding="utf-8")
        return rt.parse_profile(self.path, mode=mode, voices=4, seed=1, report_enabled=False, dcc=0.0,
                                warmup_sec=warmup, duration_sec=duration, wall_sec=5)

    def test_window_excludes_startup_and_tail_counters_allocations(self):
        metrics = self.parse(profile_fixture())
        self.assertEqual((metrics["window_start_sec"], metrics["window_end_sec"], metrics["hop_count"]), (1, 3, 8))
        self.assertEqual(metrics["underrun_frames_delta"], 0)
        self.assertEqual((metrics["underrun_counter_start"], metrics["underrun_counter_end"]), (9, 9))
        self.assertEqual(metrics["worker_alloc_count"], sum(range(5, 13)))
        self.assertEqual(metrics["worker_alloc_bytes"], sum(range(5, 13)) * 16)
        self.assertAlmostEqual(metrics["elapsed_p95_us"], 110.65)
        self.assertAlmostEqual(metrics["elapsed_p99_us"], 110.93)
        self.assertTrue(metrics["device_rt_pass"])

    def test_non_aligned_window_uses_only_contained_hops(self):
        metrics = self.parse(profile_fixture(), warmup=1.1, duration=1.8)
        self.assertEqual((metrics["window_start_sec"], metrics["window_end_sec"]), (1.25, 2.75))
        self.assertEqual(metrics["hop_count"], 6)
        self.assertEqual(metrics["underrun_counter_start"], 9)

    def test_phase_timings_are_complete_nonoverlapping_and_summarized(self):
        data = profile_fixture()
        self.assertEqual(self.parse(data)["render_route_us"]["max"], 40.0)
        for key in rt.PHASE_FIELDS:
            missing = copy.deepcopy(data)
            del missing["hops"][0][key]
            with self.assertRaisesRegex(ValueError, "missing"):
                self.parse(missing)
        data["hops"][0]["render_route_us"] = 100.0
        with self.assertRaisesRegex(ValueError, "phase durations"):
            self.parse(data)
        data = profile_fixture()
        data["hops"][0]["synthesis_us"] = 50.0
        with self.assertRaisesRegex(ValueError, "synthesis duration"):
            self.parse(data)

    def test_fixed_population_required_only_during_measurement(self):
        data = profile_fixture()
        data["hops"][0]["alive_voice_count"] = 0
        data["hops"][-1]["alive_voice_count"] = 0
        self.assertTrue(self.parse(data)["device_rt_pass"])
        data["hops"][7]["alive_voice_count"] = 3
        with self.assertRaisesRegex(ValueError, "Voice count"):
            self.parse(data)

    def test_virtual_device_and_implausibly_fast_execution_never_pass(self):
        for name in ("null", "Dummy Output", "ALSA Loopback", "virtual sink", "no sound"):
            data = profile_fixture()
            data["audio"]["device_name"] = name
            metrics = self.parse(data)
            self.assertIsNone(metrics["device_rt_pass"])
            self.assertEqual(metrics["physical_output_verification"], "known_virtual")
        self.path.write_text(json.dumps(profile_fixture()), encoding="utf-8")
        for wall in (None, .01):
            result = rt.parse_profile(self.path, mode="device", voices=4, seed=1, report_enabled=False, dcc=0,
                                      warmup_sec=1, duration_sec=2, wall_sec=wall)
            self.assertIsNone(result["device_rt_pass"])
        data = profile_fixture()
        data["audio"].update(backend="ALSA", device_name="default")
        self.assertEqual(self.parse(data)["physical_output_verification"], "not_independently_verified")

    def test_large_ring_cannot_pass_with_measurement_audio_still_queued(self):
        data = profile_fixture()
        data["audio"].update(ring_capacity_frames=80, callback_count=1, callback_frames_total=1)
        self.path.write_text(json.dumps(data), encoding="utf-8")
        metrics = rt.parse_profile(self.path, mode="device", voices=4, seed=1, report_enabled=False, dcc=0,
                                   warmup_sec=1, duration_sec=2, wall_sec=.1)
        self.assertIsNone(metrics["device_rt_pass"])
        self.assertEqual(metrics["assessment_reason"], "measurement_window_not_drained")
        self.assertEqual(metrics["generated_frames"], 20)
        self.assertEqual(metrics["minimum_dequeued_frames"], 0)

    def test_queue_drain_requires_callback_and_wall_evidence(self):
        data = profile_fixture()
        metrics = self.parse(data)
        self.assertEqual(metrics["minimum_dequeued_frames"], 12)
        self.assertEqual(metrics["minimum_device_wall_sec"], 2.5)
        self.assertTrue(metrics["device_rt_pass"])
        self.path.write_text(json.dumps(data), encoding="utf-8")
        metrics = rt.parse_profile(self.path, mode="device", voices=4, seed=1, report_enabled=False, dcc=0,
                                   warmup_sec=1, duration_sec=2, wall_sec=1.0)
        self.assertIsNone(metrics["device_rt_pass"])
        self.assertEqual(metrics["assessment_reason"], "device_pacing_not_demonstrated")
        data["audio"]["callback_frames_total"] = 11
        metrics = self.parse(data)
        self.assertIsNone(metrics["device_rt_pass"])
        self.assertEqual(metrics["assessment_reason"], "callback_frames_below_dequeue_bound")

    def test_underrun_inside_window_and_callback_error_cannot_pass(self):
        data = profile_fixture()
        for row in data["hops"][8:12]:
            row["underrun_frames_total"] += 2
        result = self.parse(data)
        self.assertEqual(result["underrun_frames_delta"], 2)
        self.assertFalse(result["device_rt_pass"])
        data = profile_fixture()
        data["audio"]["callback_errors_total"] = 1
        self.assertFalse(self.parse(data)["device_rt_pass"])

    def test_p99_over_budget_cannot_pass(self):
        data = profile_fixture()
        for row in data["hops"][4:12]:
            row["elapsed_us"] = 300000.0
        data["summary"].update(elapsed_max_us=300000.0, elapsed_p99_us=300000.0, over_budget_hops=8)
        self.assertFalse(self.parse(data)["device_rt_pass"])

    def test_offline_and_unobserved_callback_and_allocations_never_pass(self):
        offline = self.parse(profile_fixture("offline-check"), mode="offline-check")
        self.assertIsNone(offline["device_rt_pass"])
        self.assertIsNone(offline["underrun_frames_delta"])
        for missing in ("callback_count", "callback_frames_total"):
            data = profile_fixture()
            data["audio"][missing] = 0
            self.assertIsNone(self.parse(data)["device_rt_pass"])
        data = profile_fixture()
        data["allocation_instrumented"] = False
        for row in data["hops"]:
            row["worker_allocations"] = None
        metrics = self.parse(data)
        self.assertIsNone(metrics["device_rt_pass"])
        self.assertIsNone(metrics["worker_alloc_count"])

    def test_mode_seed_report_dcc_and_listener_must_match_request(self):
        for key, value in [("seed", 2), ("report_enabled", True), ("dcc_coupling_strength", .25),
                           ("listener_enabled", True), ("schema_version", 1), ("audio_output", "no_device")]:
            data = profile_fixture()
            data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.parse(data)
        with self.assertRaises(ValueError):
            self.parse(profile_fixture(), mode="offline-check")

    def test_missing_nullable_or_counter_fields_rejected(self):
        for scope, key in [("top", "allocation_scope"), ("top", "audio"), ("hop", "alive_voice_count"), ("hop", "worker_allocations"),
                           ("hop", "underrun_frames_total"), ("audio", "callback_errors_total")]:
            data = profile_fixture()
            record = data if scope == "top" else data["hops"][0] if scope == "hop" else data["audio"]
            del record[key]
            with self.subTest(scope=scope, key=key), self.assertRaisesRegex(ValueError, "missing"):
                self.parse(data)
        data = profile_fixture()
        data["hops"][4]["underrun_frames_total"] = None
        with self.assertRaises(ValueError):
            self.parse(data)

    def test_missing_unordered_truncated_and_short_profiles_rejected(self):
        variants = []
        data = profile_fixture()
        del data["hops"][6]
        variants.append(data)
        for key, value in [("truncated", True), ("dropped_hops", 1)]:
            data = profile_fixture()
            data[key] = value
            variants.append(data)
        data = profile_fixture()
        data["summary"]["hop_count"] = 21
        variants.append(data)
        data = profile_fixture()
        data["hops"][5]["time_sec"] = 1.0
        variants.append(data)
        data = profile_fixture()
        data["hops"][5]["underrun_frames_total"] = 8
        variants.append(data)
        for data in variants:
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.parse(data)
        with self.assertRaisesRegex(ValueError, "cover"):
            self.parse(profile_fixture(), duration=10)

    def test_nonfinite_duplicate_and_wrong_types_rejected(self):
        for payload in ['{"schema_version":1,"schema_version":1}', '{"bad":NaN}', '{"bad":1e999}', "[]", ""]:
            self.path.write_text(payload, encoding="utf-8")
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                rt.parse_profile(self.path, mode="device", voices=4, seed=1, report_enabled=False, dcc=0,
                                 warmup_sec=1, duration_sec=2)
        for key, value in [("sample_rate", True), ("hop_budget_us", -1), ("allocation_instrumented", 1)]:
            data = profile_fixture()
            data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.parse(data)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.config = self.directory / "input.toml"
        self.config.write_text('[audio]\nsample_rate=48000\n[dcc]\ncoupling_strength=0.9\nmax_temperature_bonus=0.07\n', encoding="utf-8")
        self.binary = self.directory / "conchordal"
        self.binary.write_bytes(b"fixture binary")

    def args(self, output, mode="device"):
        return ["--output", str(output), "--config", str(self.config), "--skip-build", "--binary", str(self.binary),
                "--mode", mode, "--voices", "4", "--bodies", "sine", "--warmup-sec", "1", "--duration-sec", "2"]

    def snapshot(self, root, output):
        (output / "source").mkdir()
        return {"head": "fixture", "source_files": {}}

    def test_existing_output_preserved_before_any_external_call(self):
        sentinel = self.directory / "keep"
        sentinel.write_text("unchanged", encoding="utf-8")
        with patch.object(rt.beta, "snapshot") as snapshot, self.assertRaises(SystemExit):
            rt.main(self.args(self.directory))
        snapshot.assert_not_called()
        self.assertEqual(sentinel.read_text(), "unchanged")

    def test_dangling_output_symlink_is_not_followed(self):
        destination = self.directory / "not-created"
        link = self.directory / "existing-link"
        link.symlink_to(destination, target_is_directory=True)
        with self.assertRaises(SystemExit):
            rt.main(self.args(link))
        self.assertFalse(destination.exists())

    def test_config_overrides_only_campaign_controls(self):
        original = tomllib.loads(self.config.read_text())
        saved = copy.deepcopy(original)
        parsed = tomllib.loads(rt.config_text(original, .25))
        self.assertEqual(original, saved)
        self.assertEqual(parsed["dcc"], {"coupling_strength": .25, "max_temperature_bonus": .07})
        self.assertEqual(parsed["audio"]["sample_rate"], 48000)
        self.assertFalse(parsed["playback"]["wait_user_exit"])

    def test_audio_initialization_failure_stops_device_campaign(self):
        output = self.directory / "device"

        def execute(command, cwd, log, timeout):
            log.write_text("Audio init failed: no device\n", encoding="utf-8")
            return {"status": "error", "returncode": 1, "command": list(map(str, command))}

        with patch.object(rt.beta, "snapshot", side_effect=self.snapshot), \
                patch.object(rt.subprocess, "check_output", return_value="fixture"), \
                patch.object(rt.beta, "execute", side_effect=execute) as run:
            args = self.args(output) + ["--voices", "4", "16", "64", "--bodies", "sine", "harmonic", "modal"]
            self.assertEqual(rt.main(args), 1)
        self.assertEqual(run.call_count, 1)
        self.assertIn("--play=true", run.call_args.args[0])
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["status"], "device_blocked")
        self.assertEqual([c["status"] for c in summary["cases"]], ["error"] + ["skipped_device_unavailable"] * 35)
        self.assertTrue((output / "device_blocker.json").is_file())
        self.assertFalse(any(c["device_rt_pass"] for c in summary["cases"]))

    def test_explicit_offline_campaign_records_all_report_dcc_pairs(self):
        output = self.directory / "offline"

        def execute(command, cwd, log, timeout):
            self.assertIn("--play=false", command)
            report = "--report" in command
            config = tomllib.loads(Path(command[command.index("--config") + 1]).read_text())
            profile = Path(command[command.index("--profile") + 1])
            profile.write_text(json.dumps(profile_fixture("offline-check", report, config["dcc"]["coupling_strength"])), encoding="utf-8")
            if report:
                Path(command[command.index("--report") + 1]).write_text('{"fixture":true}\n', encoding="utf-8")
            log.write_text("", encoding="utf-8")
            return {"status": "ok", "returncode": 0}

        with patch.object(rt.beta, "snapshot", side_effect=self.snapshot), \
                patch.object(rt.subprocess, "check_output", return_value="fixture"), \
                patch.object(rt.beta, "execute", side_effect=execute) as run:
            self.assertEqual(rt.main(self.args(output, "offline-check")), 0)
        self.assertEqual(run.call_count, 4)
        cases = json.loads((output / "summary.json").read_text())["cases"]
        self.assertEqual({(c["report_enabled"], c["dcc_coupling_strength"]) for c in cases},
                         {(False, 0.0), (False, .25), (True, 0.0), (True, .25)})
        self.assertTrue(all(c["device_rt_pass"] is None and c["status"] == "ok" for c in cases))
        self.assertEqual([c["listener_enabled"] for c in cases], [False, True, True, True])
        self.assertFalse(list(output.rglob("*.wav")))

    def test_recovery_campaign_checks_three_populations_and_actual_onsets(self):
        output = self.directory / "recovery"

        def execute(command, cwd, log, timeout):
            data = profile_fixture(report=True)
            data["hops"] = [dict(frame_idx=i, time_sec=i * .25,
                elapsed_us=300000.0 if 12 <= i < 16 else 100.0,
                alive_voice_count=8 if 12 <= i < 16 else 4,
                analysis_wait_us=0.0, listener_wait_us=0.0,
                landscape_update_us=0.0, population_us=20.0, reports_us=0.0,
                render_route_us=40.0, post_render_us=0.0,
                synthesis_us=30.0, rendered_tone_count=4,
                worker_allocations=dict(count=1, bytes=16),
                underrun_frames_total=max(0, min(i - 11, 4)) * 10) for i in range(40)]
            data["summary"].update(hop_count=40, elapsed_p99_us=300000.0,
                                   elapsed_max_us=300000.0, over_budget_hops=4)
            data["audio"].update(callback_frames_total=48, underrun_frames_total=40)
            Path(command[command.index("--profile") + 1]).write_text(json.dumps(data))
            events = [dict(type="onset", time_sec=i / 8, voice_id=v)
                      for i in range(80) for v in range(8 if 24 <= i < 32 else 4)]
            Path(command[command.index("--report") + 1]).write_text(
                "\n".join(json.dumps(e) for e in events) + "\n")
            log.write_text("")
            return {"status": "ok", "returncode": 0, "wall_sec": 10}

        with patch.object(rt.beta, "snapshot", side_effect=self.snapshot), \
                patch.object(rt.subprocess, "check_output", return_value="fixture"), \
                patch.object(rt.beta, "execute", side_effect=execute) as run:
            args = self.args(output) + ["--workload", "recovery", "--reports", "1", "--dcc", "0",
                                        "--stress-sec", "1", "--overload-voices", "8"]
            self.assertEqual(rt.main(args), 0)
        self.assertEqual(run.call_count, 1)
        case = json.loads((output / "summary.json").read_text())["cases"][0]
        self.assertTrue(case["recovery_pass"])
        self.assertTrue(case["overload_observed"])
        self.assertIsNone(case["device_rt_pass"])
        self.assertFalse(case["phases"]["overloaded"]["device_rt_pass"])
        self.assertEqual([p["onsets_per_voice_sec"] for p in case["phases"].values()], [8, 8, 8])
        self.assertEqual(case["phases"]["recovered"]["underrun_frames_delta"], 0)

    def test_missing_attacks_or_voice_coverage_reject_dense_label(self):
        path = self.directory / "report.jsonl"
        path.write_text(json.dumps(dict(type="onset", time_sec=1, voice_id=1)) + "\n")
        with self.assertRaisesRegex(ValueError, "density/Voice"):
            rt.onset_windows(path, [(1, 3, 4)])
        path.write_text("\n".join(json.dumps(dict(type="onset", time_sec=1+i/32, voice_id=1))
                                  for i in range(64)) + "\n")
        with self.assertRaisesRegex(ValueError, "density/Voice"):
            rt.onset_windows(path, [(1, 3, 4)])

    def test_overload_must_occur_and_recovery_must_pass(self):
        good = dict(device_rt_pass=True, underrun_frames_delta=0, elapsed_p99_us=100, hop_budget_us=200)
        overloaded = dict(good, device_rt_pass=False, underrun_frames_delta=50)
        self.assertIsNone(rt.recovery_assessment(good, good, good)["recovery_pass"])
        self.assertFalse(rt.recovery_assessment(good, overloaded, overloaded)["recovery_pass"])
        self.assertFalse(rt.recovery_assessment(overloaded, overloaded, good)["recovery_pass"])
        self.assertTrue(rt.recovery_assessment(good, overloaded, good)["recovery_pass"])
        self.assertIsNone(rt.recovery_assessment(good, overloaded, dict(good, device_rt_pass=None))["recovery_pass"])

    def test_onset_bursts_cannot_substitute_for_a_dense_repeating_workload(self):
        path = self.directory / "report.jsonl"
        path.write_text("\n".join(json.dumps(dict(type="onset", time_sec=1, voice_id=v))
                                  for v in range(4) for _ in range(16)) + "\n")
        with self.assertRaisesRegex(ValueError, "one report hop"):
            rt.onset_windows(path, [(1, 3, 4)])

    def test_synchronized_assay_requires_shared_scaffold_and_regular_8_hz(self):
        path = self.directory / "report.jsonl"
        rows = [dict(type="onset", time_sec=1+i/8, voice_id=v, scaffold_mode="shared")
                for i in range(16) for v in range(4)]
        def check(events):
            path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
            return rt.onset_windows(path, [(1, 3, 4)], timing="synchronized", hop_sec=512/48000)
        self.assertEqual(check(rows)[0]["onsets_per_voice_sec"], 8)
        with self.assertRaisesRegex(ValueError, "shared scaffold"):
            check([dict(r, scaffold_mode="off") for r in rows])
        with self.assertRaisesRegex(ValueError, "8 Hz onset count and intervals"):
            check([r for i,r in enumerate(rows) if i not in (8, 12)])
        with self.assertRaisesRegex(ValueError, "8 Hz onset count and intervals"):
            check([dict(r, time_sec=r["time_sec"] + (.02 if i // 4 == 3 else 0))
                   for i,r in enumerate(rows)])

    def test_participation_requires_repeated_activity_without_a_dense_clock(self):
        path = self.directory / "participation.jsonl"
        rows = [dict(type="onset", time_sec=t + v * .02, voice_id=v)
                for v in range(4) for t in (1.0, 1.7, 2.5)]
        path.write_text("\n".join(map(json.dumps, rows)))
        for timing in ("entrained", "flow"):
            result = rt.onset_windows(path, [(1, 3, 4)], timing=timing)[0]
            self.assertEqual(result["onset_count"], 12)
            self.assertLess(result["onsets_per_voice_sec"], 4)
        with self.assertRaisesRegex(ValueError, "dense workload"):
            rt.onset_windows(path, [(1, 3, 4)])
        path.write_text("\n".join(map(json.dumps, rows[:10])))
        with self.assertRaisesRegex(ValueError, "repeated onsets"):
            rt.onset_windows(path, [(1, 3, 4)], timing="flow")

    def test_participation_presets_cannot_be_replaced_by_a_synchronized_scaffold(self):
        for mode in ("entrained", "flow"):
            script = rt.scenario_text(4, "modal", 21, 5, 10, workload=mode)
            self.assertIn(f".{mode}().cycles(1)", script)
            self.assertNotIn(".pulse(", script)
            self.assertNotIn("set_scaffold", script)
            with self.assertRaises(SystemExit):
                rt.main(self.args(self.directory / ("invalid-" + mode)) + ["--workload", mode, "--timing", "synchronized"])

    def test_synchronization_is_explicit_and_requires_onset_evidence(self):
        adaptive = rt.scenario_text(4, "modal", 1, 5, 10, workload="dense")
        synchronized = rt.scenario_text(4, "modal", 1, 5, 10, workload="dense", timing="synchronized")
        self.assertNotIn("set_scaffold_shared", adaptive)
        self.assertIn("set_scaffold_shared(16.0)", synchronized)
        self.assertIn(".pulse(32.0)", synchronized)
        self.assertIn(".pulse(8.0)", adaptive)
        with self.assertRaises(SystemExit):
            rt.main(self.args(self.directory / "invalid-sync") + ["--timing", "synchronized"])


if __name__ == "__main__":
    unittest.main()
