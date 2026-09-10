"""Check the isolated amplitude intervention and its pre-intervention gate."""

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
import evaluate_flow_amplitude as amplitude


class FlowAmplitudeTests(unittest.TestCase):
    def test_pair_changes_only_the_amplitude_patch_and_rejects_source_drift(self):
        source = (SCRIPTS.parent / "samples" / "12_emergence_and_resolution.rhai").read_text()
        self.assertEqual(amplitude.amplitude_source(source, "reduced"), source)
        control = amplitude.amplitude_source(source, "unchanged")
        self.assertEqual(control.replace("    flow.amp(0.020);\n", "    flow.amp(0.014);\n"), source)
        for changed in (
            source.replace("    .amp(0.020)", "    .amp(0.025)"),
            source.replace("wait(5.3)", "wait(5.4)", 1),
            source.replace("    flow.amp(0.014);\n", "") + "    flow.amp(0.014);\n",
            source + "    flow.amp(0.019);\n",
        ):
            with self.assertRaises(ValueError):
                amplitude.amplitude_source(changed, "reduced")

    def test_prefix_extends_through_flow_until_amplitude_patch(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            variants = {label: {"sample": label} for label in amplitude.AMPLITUDES}
            cases = [{"sample": label, "seed": 1, "case": label, "status": "ok"} for label in variants]
            (output / "summary.json").write_text(json.dumps({"cases": cases}))
            records = [dict(type="spawn", time_sec=0.0, population_id=1),
                       dict(type="spawn", time_sec=15.0, population_id=3),
                       dict(type="population_step", time_sec=19.0, population_id=3, alive_count=9),
                       dict(type="listener_state", time_sec=20.29, tension_level=.1)]
            for index, label in enumerate(variants):
                directory = output / label
                directory.mkdir()
                with wave.open(str(directory / "audio.wav"), "wb") as stream:
                    stream.setparams((1, 2, 10, 0, "NONE", "not compressed"))
                    stream.writeframes(struct.pack("<240h", *([100] * 203 + [index] * 37)))
                rows = records + [dict(type="listener_state", time_sec=20.3, tension_level=index)]
                (directory / "report.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            args = dict(end_sec=amplitude.INTERVENTION_SEC, pairs=[("unchanged", "reduced")],
                        filename="amplitude_pre_intervention.json")
            check = amplitude.resolution.flow_pre_intervention_checks
            self.assertEqual(check(output, variants, [1], **args), [])
            saved = json.loads((output / args["filename"]).read_text())
            self.assertEqual(saved["checks"][0]["off_prefix"]["audio_frames"], 203)
            path = output / "reduced" / "report.jsonl"
            original = path.read_text()
            path.write_text(original.replace('"alive_count": 9', '"alive_count": 8'))
            self.assertTrue(check(output, variants, [1], **args))
            path.write_text(original)
            with wave.open(str(output / "reduced" / "audio.wav"), "wb") as stream:
                stream.setparams((1, 2, 10, 0, "NONE", "not compressed"))
                stream.writeframes(struct.pack("<240h", *([100] * 202 + [101] + [0] * 37)))
            self.assertTrue(check(output, variants, [1], **args))

    def test_failed_pair_validation_marks_campaign_failed_without_audition(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "campaign"

            def fake_beta(args):
                self.assertEqual(args[args.index("--seeds") + 1], "1")
                self.assertEqual(args[args.index("--reserve-runtime-ids-through") + 1], "18")
                output.mkdir()
                (output / "manifest.json").write_text('{"status":"complete"}')
                return 0

            with patch.object(amplitude.beta, "main", side_effect=fake_beta), \
                    patch.object(amplitude, "publish_audition", side_effect=ValueError("prefix mismatch")):
                self.assertEqual(amplitude.main(["--output", str(output)]), 1)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["amplitude_experiment"]["error"], "prefix mismatch")
            self.assertFalse((output / "audition.html").exists())


if __name__ == "__main__":
    unittest.main()
