"""Resolution intervention and paired-window checks without rendering."""

import importlib.util
import itertools
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("evaluate_resolution", SCRIPTS / "evaluate_resolution.py")
resolution = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolution)


class ResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (SCRIPTS.parent / "samples" / "12_emergence_and_resolution.rhai").read_text(encoding="utf-8")

    def test_all_eight_variants_remove_only_disabled_operations(self):
        for values in itertools.product((0, 1), repeat=3):
            factors = dict(zip(resolution.FACTORS, values))
            generated = resolution.variant_source(self.source, factors)
            expected = self.source
            for factor, fragments in resolution.FACTORS.items():
                for fragment in fragments:
                    self.assertEqual(fragment in generated, bool(factors[factor]))
                    if not factors[factor]:
                        expected = expected.replace(fragment, "")
            self.assertEqual(generated, expected)
            self.assertIn("    colony.amp(0.034);\n", generated)
            self.assertIn("    colony.glide(0.22);\n", generated)
        self.assertEqual(resolution.variant_source(self.source, dict.fromkeys(resolution.FACTORS, 1)), self.source)

    def test_operation_and_wait_drift_fail_even_for_control(self):
        flow_line = resolution.FACTORS["flow"][0]
        for source in [self.source.replace("colony.temperature(0.85)", "colony.temperature(0.8)"),
                       self.source.replace("wait(9.4)", "wait(9.5)"),
                       self.source.replace("count(8).spacing(0.84)", "count(7).spacing(0.84)"),
                       self.source + "let extra = place(sine(), at(220.0));\n",
                       self.source.replace(flow_line + "    wait(5.3);", "    wait(5.3);\n" + flow_line, 1),
                       self.source + "    release(flow);\n"]:
            with self.assertRaises(ValueError):
                resolution.variant_source(source, dict.fromkeys(resolution.FACTORS, 1))

    def test_pre_flow_requires_identical_pcm_and_state_but_ignores_timing_and_post_flow(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            variants = {label: {"sample": label} for label in ("000", "001")}
            cases = [{"sample": label, "seed": 42, "case": label, "status": "ok"} for label in variants]
            (output / "summary.json").write_text(json.dumps({"cases": cases}))
            records = [dict(type="spawn", time_sec=0.0, voice_id=1),
                       dict(type="population_step", time_sec=6.0, alive_count=8),
                       dict(type="listener_state", time_sec=6.0, tension_level=.1),
                       dict(type="respawn", time_sec=13.7, voice_id=20)]
            for index, label in enumerate(variants):
                directory = output / label
                directory.mkdir()
                with wave.open(str(directory / "audio.wav"), "wb") as stream:
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.setframerate(10)
                    samples = [100] * 187 + [index] * 13
                    stream.writeframes(struct.pack("<200h", *samples))
                rows = records + [dict(type="hop_timing", time_sec=6.0, elapsed_us=index),
                                  dict(type="spawn", time_sec=18.7, voice_id=index)]
                (directory / "report.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.assertEqual(resolution.flow_pre_intervention_checks(output, variants, [42]), [])
            check = json.loads((output / "flow_pre_intervention.json").read_text())["checks"][0]
            self.assertTrue(check["matched"])
            self.assertEqual(check["off_prefix"]["audio_frames"], 187)
            self.assertEqual(check["off_prefix"]["respawns"][0]["voice_id"], 20)

            path = output / "001" / "report.jsonl"
            original = path.read_text()
            path.write_text(original.replace('"voice_id": 20', '"voice_id": 11'))
            self.assertTrue(resolution.flow_pre_intervention_checks(output, variants, [42]))
            path.write_text(original)
            with wave.open(str(output / "001" / "audio.wav"), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(10)
                stream.writeframes(struct.pack("<200h", 101, *([100] * 199)))
            self.assertTrue(resolution.flow_pre_intervention_checks(output, variants, [42]))
            path.write_text("".join(json.dumps(r) + "\n" for r in records if r["type"] != "listener_state"))
            with self.assertRaisesRegex(ValueError, "missing pre-flow"):
                resolution.flow_pre_intervention_checks(output, variants, [42])

    def test_window_excludes_dead_colony_scores_and_half_open_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            wav = Path(temporary) / "audio.wav"
            with wave.open(str(wav), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(10)
                stream.writeframes(struct.pack("<" + "h" * 400, *([16384] * 400)))
            listener = [dict(time_sec=t, analysis_lag_frames=1,
                             **dict.fromkeys(resolution.LISTENER_FIELDS, value))
                        for t, value in [(18.7, .2), (23.9, .4), (24.0, .9)]]
            population = [dict(time_sec=t, population_id=3, alive_count=alive,
                               mean_freq_hz=freq, mean_c_field_level=score)
                          for t, alive, freq, score in [(18.7, 2, 220, .8), (23.9, 0, 0, 0), (24., 3, 900, 1.)]]
            row = resolution.window_metrics(listener, population, wav, "tension")
            self.assertEqual(row["listener_count"], 2)
            self.assertAlmostEqual(row["listener_tension_level_mean"], .3)
            self.assertEqual(row["colony_alive_count_mean"], 1)
            self.assertEqual(row["colony_mean_freq_hz"], 220)
            self.assertEqual(row["colony_mean_c_field_level"], .8)
            self.assertEqual(row["audio_frames"], 53)
            self.assertEqual(row["audio_rms"], .5)
            self.assertEqual(row["audio_peak"], .5)
            self.assertEqual(row["audio_silence_fraction"], 0)

    def test_paired_differences_match_seeds_and_preserve_missing_scores(self):
        rows = []
        for variant, seed, delta in [("111", 1, .1), ("000", 1, .3), ("111", 42, .5), ("000", 42, .2)]:
            for window, value in [("tension", 0), ("early_resolution", delta)]:
                rows.append({"variant": variant, "seed": seed, "window": window,
                             **dict(zip(resolution.FACTORS, map(int, variant))),
                             **dict.fromkeys(resolution.MEASURES, value), "colony_mean_freq_hz": None})
        effects = {(r["variant"], r["seed"]): r for r in resolution.paired_effects(rows)}
        key = "delta_minus_control_111_listener_tension_level_mean"
        self.assertAlmostEqual(effects[("000", 1)][key], .2)
        self.assertAlmostEqual(effects[("000", 42)][key], -.3)
        self.assertEqual(effects[("111", 1)][key], 0)
        self.assertIsNone(effects[("000", 1)]["early_minus_tension_colony_mean_freq_hz"])

    def test_source_digest_is_fixed_before_rendering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "samples" / "12_emergence_and_resolution.rhai"
            source_path.parent.mkdir()
            source_path.write_text(self.source, encoding="utf-8")
            output = root / "campaign"

            def fake_beta(_args):
                reserve_index = _args.index("--reserve-runtime-ids-through")
                self.assertEqual(_args[reserve_index + 1], "19")
                output.mkdir()
                (output / "manifest.json").write_text('{"status":"complete"}', encoding="utf-8")
                source_path.write_text("changed during rendering", encoding="utf-8")
                return 0

            with patch.object(resolution, "__file__", str(root / "scripts" / "evaluate_resolution.py")), \
                    patch.object(resolution.beta, "main", side_effect=fake_beta), \
                    patch.object(resolution, "aggregate", return_value=[]):
                self.assertEqual(resolution.main(["--output", str(output)]), 0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            expected = resolution.hashlib.sha256(self.source.encode("utf-8")).hexdigest()
            self.assertEqual(manifest["resolution_experiment"]["source_sha256"], expected)
            self.assertEqual(manifest["resolution_experiment"]["variants"]["111"]["sha256"], expected)

    def test_aggregation_failure_marks_manifest_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "samples" / "12_emergence_and_resolution.rhai"
            source_path.parent.mkdir()
            source_path.write_text(self.source, encoding="utf-8")
            output = root / "campaign"

            def fake_beta(_args):
                output.mkdir()
                (output / "manifest.json").write_text('{"status":"complete","head":"original"}', encoding="utf-8")
                return 0

            with patch.object(resolution, "__file__", str(root / "scripts" / "evaluate_resolution.py")), \
                    patch.object(resolution.beta, "main", side_effect=fake_beta), \
                    patch.object(resolution, "aggregate", side_effect=ValueError("missing window")):
                self.assertEqual(resolution.main(["--output", str(output)]), 1)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["head"], "original")
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["resolution_experiment"]["status"], "failed")
            self.assertEqual(manifest["resolution_experiment"]["error"], "missing window")


if __name__ == "__main__":
    unittest.main()
