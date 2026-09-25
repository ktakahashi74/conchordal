"""Registered I11 inputs stay frozen when an action-profile path is relocated."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


SCRIPT = (Path(__file__).resolve().parents[1]
          / "docs/roadmap/temporal-dcc/i11-inputs/acquire.py")
SPEC = importlib.util.spec_from_file_location("i11_acquire", SCRIPT)
acquire = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acquire)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AcquireInputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo"
        self.registration = self.repo / "registered"
        self.inputs = self.registration / "inputs"
        self.inputs.mkdir(parents=True)
        self.output = self.base / 'output with "quotes" and spaces'
        self.profile = self.base / 'profile with "quotes" and spaces.bin'
        self.profile.write_bytes(b"registered action profile")
        self.script = self.inputs / "sample.rhai"
        self.script.write_text("// scenario\n")
        self.config = self.inputs / "config-body.toml"
        self.config.write_text(
            '[audio]\nsample_rate = 48000\n'
            '[temporal_action_profiles]\nfile = "/old/linux/profile.bin"\n'
            f'sha256 = "{digest(self.profile)}"\n'
            '[other]\nvalue = 7\n')
        self.original_config = self.config.read_bytes()
        registered = {str(path.relative_to(self.repo)): digest(path)
                      for path in (self.script, self.config)}
        self.plan = self.registration / "plan.json"
        self.plan.write_text(json.dumps({"cases": ["sample"], "inputs_sha256": registered}))
        self.instrument = self.base / "instrument"
        self.render = self.base / "render"
        self.instrument.write_bytes(b"instrument")
        self.render.write_bytes(b"render")
        self.repo.joinpath("src").mkdir()

    def run_acquire(self, *options, inputs=None, output=None):
        argv = ["--root", str(output or self.output), "--plan", str(self.plan),
                "--inputs", str(inputs or self.inputs),
                "--instrument", str(self.instrument), "--render", str(self.render),
                "--case", "sample", "--variant", "body", "--mode", "no-report",
                "--reps", "0", *options]
        with patch.object(acquire, "REPO", self.repo), patch.object(acquire, "ROOT", self.registration):
            with contextlib.redirect_stdout(io.StringIO()):
                acquire.main(argv)

    @unittest.skipIf(tomllib is None, "profile relocation requires Python 3.11+")
    def test_profile_override_changes_only_file_and_records_provenance(self):
        self.run_acquire("--profile-file", str(self.profile))
        derived = self.output / "run-configs/config-body.toml"
        actual_profile = self.output.resolve() / "action-profiles.bin"
        original = tomllib.loads(self.original_config.decode())
        expected = tomllib.loads(self.original_config.decode())
        expected["temporal_action_profiles"]["file"] = str(actual_profile)
        self.assertEqual(tomllib.loads(derived.read_text()), expected)
        self.assertEqual(
            derived.read_text(),
            self.original_config.decode().replace(
                'file = "/old/linux/profile.bin"',
                "file = " + json.dumps(str(actual_profile), ensure_ascii=False)))
        self.assertEqual(self.config.read_bytes(), self.original_config)
        self.assertEqual(actual_profile.read_bytes(), self.profile.read_bytes())
        self.assertNotEqual(derived.read_bytes(), self.original_config)
        manifest = json.loads((self.output / "acquisition-inputs.json").read_text())
        self.assertEqual(manifest["selected_inputs"]["config-body.toml"]["sha256"],
                         digest(self.config))
        self.assertEqual(manifest["selected_inputs"]["config-body.toml"]["original_sha256"],
                         digest(self.config))
        self.assertEqual(manifest["selected_inputs"]["config-body.toml"]["derived_sha256"],
                         digest(derived))
        self.assertEqual(manifest["profile"],
                         {"source_path": str(self.profile.resolve()), "actual_path": str(actual_profile),
                          "sha256": digest(self.profile)})
        self.assertEqual(original["temporal_action_profiles"]["sha256"], digest(self.profile))

    @unittest.skipIf(tomllib is None, "profile relocation requires Python 3.11+")
    def test_missing_or_wrong_profile_fails_before_binary_copy(self):
        wrong = self.base / "wrong.bin"
        wrong.write_bytes(b"different")
        for profile in (self.base / "missing.bin", wrong):
            with self.subTest(profile=profile):
                with self.assertRaises(SystemExit):
                    self.run_acquire("--profile-file", str(profile))
                self.assertFalse((self.output / "instrument").exists())
                self.assertFalse((self.output / "render").exists())

    @unittest.skipIf(tomllib is None, "profile relocation requires Python 3.11+")
    def test_render_uses_derived_config_and_copied_profile(self):
        with patch.object(acquire.subprocess, "run", return_value=Mock(returncode=0)) as run:
            self.run_acquire("--profile-file", str(self.profile), "--mode", "render")
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(Path(command[0]), self.output.resolve() / "render")
        config = Path(command[command.index("--config") + 1])
        self.assertEqual(config, self.output.resolve() / "run-configs/config-body.toml")
        profile = Path(tomllib.loads(config.read_text())["temporal_action_profiles"]["file"])
        self.assertEqual(profile, self.output.resolve() / "action-profiles.bin")
        self.assertEqual(profile.read_bytes(), self.profile.read_bytes())

    def test_unverified_inputs_are_rejected_before_binary_copy(self):
        alternate = self.base / "alternate"
        alternate.mkdir()
        for filename in ("sample.rhai", "config-body.toml"):
            shutil.copy2(self.inputs / filename, alternate / filename)
        for filename in ("sample.rhai", "config-body.toml"):
            with self.subTest(filename=filename):
                path = alternate / filename
                original = path.read_bytes()
                path.write_bytes(original + b"\n# changed\n")
                with self.assertRaisesRegex(SystemExit, "selected input is not the registered bytes"):
                    self.run_acquire(inputs=alternate)
                self.assertFalse((self.output / "instrument").exists())
                path.write_bytes(original)

    def test_old_registration_location_allows_verified_copies(self):
        old_inputs = self.repo / "old-registration" / "inputs"
        old_inputs.mkdir(parents=True)
        for filename in ("sample.rhai", "config-body.toml"):
            shutil.copy2(self.inputs / filename, old_inputs / filename)
        plan = json.loads(self.plan.read_text())
        plan["inputs_sha256"] = {str(path.relative_to(self.repo)): digest(path)
                                 for path in old_inputs.iterdir()}
        self.plan.write_text(json.dumps(plan))
        self.run_acquire(inputs=self.inputs)
        manifest = json.loads((self.output / "acquisition-inputs.json").read_text())
        self.assertEqual(
            manifest["selected_inputs"]["sample.rhai"]["registered_path"],
            "old-registration/inputs/sample.rhai")

    def test_without_override_keeps_registered_config_and_python39_path(self):
        with patch.object(acquire, "tomllib", None):
            self.run_acquire()
        self.assertEqual(self.config.read_bytes(), self.original_config)
        self.assertFalse((self.output / "run-configs").exists())
        manifest = json.loads((self.output / "acquisition-inputs.json").read_text())
        self.assertIsNone(manifest["profile"])
        self.assertEqual(manifest["selected_inputs"]["config-body.toml"]["path"],
                         str(self.config.resolve()))
        self.assertNotIn("derived_sha256", manifest["selected_inputs"]["config-body.toml"])

    def test_override_without_tomllib_fails_before_binary_copy(self):
        with patch.object(acquire, "tomllib", None):
            with self.assertRaisesRegex(SystemExit, "requires Python 3.11 or newer"):
                self.run_acquire("--profile-file", str(self.profile))
        self.assertFalse((self.output / "instrument").exists())


if __name__ == "__main__":
    unittest.main()
