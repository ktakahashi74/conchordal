"""Fail-closed tests for the frozen I11-2 acquisition path."""

import importlib.util
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = (Path(__file__).resolve().parents[1] / "docs" / "roadmap" / "temporal-dcc"
          / "i11-2-inputs" / "acquire.py")
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("i11_stage2_acquire", SCRIPT)
acquire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acquire)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_i11_stage2 as verifier


class AcquisitionTest(unittest.TestCase):
    def test_frozen_registration_is_intact(self):
        plan, inputs = acquire.registered_inputs(require_baselines=False)
        self.assertEqual(len(plan["run_order"]), 852)
        self.assertEqual(len(inputs), 25)

    @unittest.skipIf(acquire.tomllib is None, "tomllib requires Python 3.11")
    def test_profile_copy_preserves_original_and_config_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "config-body-on.toml"
            profile = root / "profile with ' quote.bin"
            profile.write_bytes(b"registered binary bytes")
            digest = acquire.sha256(profile)
            original.write_text("[temporal_action_profiles]\nfile = 'before'\nsha256 = '" + digest
                                + "'\n[temporal_onset_comparison]\narrival = true\n")
            before = original.read_bytes()
            plan = {"source_profile_sha256": digest,
                    "run_order": [{"config": "body-on"}]}
            configs, copied = acquire.prepare_configs(plan, {original.name: original}, root / "out", profile)
            self.assertEqual(original.read_bytes(), before)
            self.assertEqual(acquire.sha256(copied["path"]), digest)
            import tomllib
            base = tomllib.loads(before.decode())
            derived = tomllib.loads(Path(configs["body-on"]["path"]).read_text())
            self.assertEqual(derived["temporal_action_profiles"]["file"], copied["path"])
            derived["temporal_action_profiles"]["file"] = base["temporal_action_profiles"]["file"]
            self.assertEqual(derived, base)
            Path(configs["body-on"]["path"]).write_text("tampered")
            with self.assertRaisesRegex(ValueError, "derived config"):
                acquire.prepare_configs(plan, {original.name: original}, root / "out", profile)
            with self.assertRaisesRegex(ValueError, "profile is missing or differs"):
                acquire.prepare_configs(plan, {original.name: original}, root / "other", root / "missing")

    def test_wrong_binary_sha_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "render"
            binary.write_bytes(b"wrong binary")
            with self.assertRaisesRegex(ValueError, "binary SHA-256"):
                acquire.verified_binary(binary, "0" * 64)

    def test_candidate_source_manifest_must_cover_every_source(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "src").mkdir()
            (repo / "src" / "main.rs").write_text("fn main() {}")
            (repo / "Cargo.toml").write_text("[package]")
            (repo / "Cargo.lock").write_text("lock")
            (repo / "old.json").write_text(json.dumps({"binaries": {}}))
            (repo / "pre.json").write_text(json.dumps({"binaries": {}}))
            candidate = repo / "candidate.json"
            candidate.write_text(json.dumps({"schema": "conchordal/i11-stage2-candidate-provenance/1",
                                             "source_commit": "base", "source_files_sha256":
                                             {"Cargo.toml": acquire.sha256(repo / "Cargo.toml")},
                                             "binaries": {}}))
            plan = {"baseline": {"original_none": {"manifest_path": "old.json"},
                                 "pre_stage2_off": {"manifest_path": "pre.json"}}}
            with mock.patch.object(acquire, "REPO", repo):
                with self.assertRaisesRegex(ValueError, "omits source"):
                    acquire.binary_sources(plan, candidate)

    def one_run(self, root, *, mode="render", phase="offline_stage2_existing", config="none", case="sine-flow-4"):
        row = {"sequence": 1, "phase": phase, "case": case, "config": config,
               "binary": "stage2_candidate", "mode": mode, "rep": 0}
        plan = {"run_order": [row], "new_pulse_cases": ["sine-pulse-flow-4"]}
        inputs = {f"{case}.rhai": root / "case.rhai"}
        configs = {config: {"path": str(root / "config.toml")}}
        binaries = {"stage2_candidate": {"conchordal": {"path": str(root / "instrument")},
                                          "conchordal-render": {"path": str(root / "render")}}}
        return plan, inputs, configs, binaries

    @staticmethod
    def valid_profile(consumed=1):
        return {"hops": [{"population_us": 100.0, "synthesis_us": 200.0,
                          "elapsed_us": 300.0}], "hop_budget_us": 10000.0,
                "truncated": False, "dropped_hops": 0,
                "summary": {"hop_count": 1, "over_budget_hops": 0},
                "background": {"arrival_known_decisions": consumed}}

    @staticmethod
    def fake_run(command, **kwargs):
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_bytes(b"WAV")
        if "--report" in command:
            Path(command[command.index("--report") + 1]).write_text("report\n")
        if "--profile" in command:
            Path(command[command.index("--profile") + 1]).write_text(
                json.dumps(AcquisitionTest.valid_profile()))
        kwargs["stdout"].write(b"completed\n")
        return mock.Mock(returncode=0)

    def test_resume_rejects_replaced_artifact_and_does_not_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, inputs, configs, binaries = self.one_run(root)
            with mock.patch.object(acquire.subprocess, "run", side_effect=self.fake_run) as run:
                self.assertEqual(acquire.acquire(plan, inputs, configs, binaries, root, "sealed", 1), 1)
                self.assertEqual(acquire.acquire(plan, inputs, configs, binaries, root, "sealed", 1), 1)
                self.assertEqual(run.call_count, 1)
            ledger = json.loads((root / "runs.json").read_text())
            missing = json.loads(json.dumps(ledger))
            missing["runs"][0]["artifacts"] = {}
            with self.assertRaisesRegex(ValueError, "artifact set differs"):
                acquire.verify_ledger(plan, root, missing, "sealed")
            report = next(root.glob("*.jsonl"))
            report.write_text("replaced\n")
            with self.assertRaisesRegex(ValueError, "ledger artifact differs"):
                acquire.acquire(plan, inputs, configs, binaries, root, "sealed", 1)

    def test_nonzero_exit_and_missing_report_stop_without_ledger_entry(self):
        for outcome in ("exit", "missing"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan, inputs, configs, binaries = self.one_run(root)
                def run(command, **kwargs):
                    if outcome == "exit":
                        kwargs["stdout"].write(b"failed\n")
                        return mock.Mock(returncode=9)
                    Path(command[command.index("-o") + 1]).write_bytes(b"WAV")
                    kwargs["stdout"].write(b"no report\n")
                    return mock.Mock(returncode=0)
                with mock.patch.object(acquire.subprocess, "run", side_effect=run):
                    with self.assertRaisesRegex(ValueError, "exit=9|missing or empty report"):
                        acquire.acquire(plan, inputs, configs, binaries, root, "sealed", 1)
                self.assertFalse((root / "runs.json").exists())

    def test_no_report_pulse_requires_consumption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, inputs, configs, binaries = self.one_run(
                root, mode="no-report", phase="performance_comparison",
                config="body-on", case="sine-pulse-flow-4")
            def run(command, **kwargs):
                Path(command[command.index("--profile") + 1]).write_text(
                    json.dumps(self.valid_profile(0)))
                kwargs["stdout"].write(b"completed\n")
                return mock.Mock(returncode=0)
            with mock.patch.object(acquire.subprocess, "run", side_effect=run):
                with self.assertRaisesRegex(ValueError, "no valid arrival consumption"):
                    acquire.acquire(plan, inputs, configs, binaries, root, "sealed", 1)
            self.assertEqual(json.loads((root / "runs.json").read_text())["runs"][0]["arrival_eligibility"], False)
            with self.assertRaisesRegex(ValueError, "no valid arrival consumption"):
                acquire.acquire(plan, inputs, configs, binaries, root, "sealed", 1)

    def test_profile_rejects_nan_and_forged_budget_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            profile = self.valid_profile()
            profile["hops"][0]["elapsed_us"] = float("nan")
            path.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError, "invalid elapsed_us timing"):
                verifier.run_profile(path)
            profile = self.valid_profile()
            profile["summary"]["over_budget_hops"] = 1
            path.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError, "invalid over-budget count"):
                verifier.run_profile(path)

    def test_missing_floor_blocks_comparison_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs.json").write_text(json.dumps({"schema": "conchordal/i11-stage2-runs/1",
                                                       "provenance_sha256": "sealed", "runs": []}))
            plan, inputs, configs, binaries = self.one_run(root)
            with (mock.patch.object(acquire, "verify_ledger", return_value=[{}] * 601),
                  mock.patch.object(acquire, "command", return_value=None)):
                with self.assertRaisesRegex(ValueError, "frozen A/A floor"):
                    acquire.acquire(plan, inputs, configs, binaries, root, "sealed", 1)

    def test_existing_floor_cannot_be_adopted_without_recalculation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "aa-floor.json").write_text(json.dumps({"schema": "forged", "floor": {}}))
            expected = {"schema": "conchordal/i11-stage2-aa-floor/1",
                        "source_ledger_prefix_sha256": "measured", "floor": {"elapsed_us_max": 100},
                        "pass3_exceedances_against_pass1_2": {}, "differences": []}
            with mock.patch.object(acquire, "aa_floor", return_value=expected):
                with self.assertRaisesRegex(ValueError, "existing A/A floor differs"):
                    acquire.ensure_floor({}, root, [], {}, root / "runs.json")

    def test_performance_floor_and_all_comparison_cells(self):
        plan, inputs = acquire.registered_inputs(require_baselines=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            candidate_manifest = root / "candidate-manifest.json"
            profile_file = root / "action-profiles.bin"
            profile_file.write_bytes(b"profile")
            configs = {}
            for name in {row["config"] for row in plan["run_order"]}:
                original = inputs[f"config-{name}.toml"]
                derived = root / f"config-{name}.toml"
                derived.write_text(re.sub(r'(?m)^file = .*$',
                                          'file = ' + json.dumps(str(profile_file)),
                                          original.read_text(), count=1))
                configs[name] = {"original": str(original), "original_sha256": acquire.sha256(original),
                                 "path": str(derived), "sha256": acquire.sha256(derived)}
            binaries = {}
            simulated_hashes = {str(profile_file): plan["source_profile_sha256"]}
            for version in {row["binary"] for row in plan["run_order"]}:
                binaries[version] = {}
                for executable in ("conchordal", "conchordal-render"):
                    path = root / f"{version}-{executable}"
                    path.write_bytes(b"binary")
                    digest = acquire.sha256(path)
                    if version != "stage2_candidate":
                        kind = "render" if executable == "conchordal-render" else "instrument"
                        digest = plan["baseline"][version][f"{kind}_binary_sha256"]
                        simulated_hashes[str(path)] = digest
                    binaries[version][executable] = {"path": str(path), "sha256": digest}
            for version in ("original_none", "pre_stage2_off"):
                baseline = plan["baseline"][version]
                simulated_hashes[str(acquire.REPO / baseline["manifest_path"])] = baseline["manifest_sha256"]
            candidate_manifest.write_text(json.dumps({
                "schema": "conchordal/i11-stage2-candidate-provenance/1",
                "binaries": {name: {"sha256": entry["sha256"]}
                             for name, entry in binaries["stage2_candidate"].items()}}))
            provenance = {"schema": "conchordal/i11-stage2-acquisition-provenance/1",
                          "registration_freeze_sha256": acquire.sha256(acquire.HERE / "freeze.json"),
                          "plan_sha256": acquire.sha256(acquire.HERE / "plan.json"),
                          "candidate_manifest": str(candidate_manifest),
                          "candidate_manifest_sha256": acquire.sha256(candidate_manifest),
                          "binaries": binaries, "configs": configs,
                          "profile": {"path": str(profile_file), "sha256": plan["source_profile_sha256"]}}
            acquire.write_json(root / "acquisition-provenance.json", provenance)
            real_sha256 = verifier.sha256_file
            def verified_hash(path):
                return (simulated_hashes[str(path)] if str(path) in simulated_hashes
                        else real_sha256(path))
            def verify():
                with mock.patch.object(verifier, "sha256_file", side_effect=verified_hash):
                    return verifier.verify_performance(Path(os.path.relpath(root, Path.cwd())), plan)
            runs = []
            for row in plan["run_order"]:
                output = acquire.output_paths(root, row)
                artifacts = {}
                for kind, path in output.items():
                    if kind == "profile":
                        baseline = 1000 + (100 if row.get("arm") == "none-B" else 0)
                        if row["phase"] == "performance_comparison" and row["config"] == "body-on":
                            baseline += 50
                        profile = {"hops": [{"population_us": baseline,
                                             "synthesis_us": baseline,
                                             "elapsed_us": baseline}],
                                   "truncated": False, "dropped_hops": 0,
                                   "hop_budget_us": 10000.0,
                                   "summary": {"hop_count": 1, "over_budget_hops": 0},
                                   "background": {"arrival_known_decisions": 1}}
                        path.write_text(json.dumps(profile))
                    else:
                        path.write_bytes(b"artifact")
                    artifacts[kind] = {"path": path.name, "sha256": acquire.sha256(path),
                                       "bytes": path.stat().st_size}
                runs.append({"sequence": row["sequence"], "registration": row, "exit": 0,
                             "artifacts": artifacts,
                             "command": acquire.command(row, inputs, configs, binaries, output)})
            floor_path = root / "aa-floor.json"
            acquire.write_json(floor_path, acquire.aa_floor(plan, root, runs[:600]))
            ledger = {"schema": "conchordal/i11-stage2-runs/1", "runs": runs,
                      "provenance_sha256": acquire.sha256(root / "acquisition-provenance.json"),
                      "aa_floor_sha256": acquire.sha256(floor_path)}
            acquire.write_json(root / "runs.json", ledger)
            result = verify()
            self.assertEqual(result["checked"], 596)
            self.assertEqual(result["failed"], 0)
            rows_without_artifact = json.loads(json.dumps(ledger))
            rows_without_artifact["runs"][0]["artifacts"] = {}
            acquire.write_json(root / "runs.json", rows_without_artifact)
            with self.assertRaisesRegex(ValueError, "artifact set differs"):
                verify()
            acquire.write_json(root / "runs.json", ledger)
            wrong_command = json.loads(json.dumps(ledger))
            wrong_command["runs"][0]["command"][0] = "/other/binary"
            acquire.write_json(root / "runs.json", wrong_command)
            with self.assertRaisesRegex(ValueError, "run command differs"):
                verify()
            acquire.write_json(root / "runs.json", ledger)
            pulse = next(run for run in runs if run["registration"]["phase"] == "performance_comparison"
                         and run["registration"]["case"] == "sine-pulse-flow-4"
                         and run["registration"]["config"] == "body-on")
            path = root / pulse["artifacts"]["profile"]["path"]
            broken = json.loads(path.read_text())
            broken["background"]["arrival_known_decisions"] = 0
            path.write_text(json.dumps(broken))
            pulse["artifacts"]["profile"]["sha256"] = acquire.sha256(path)
            acquire.write_json(root / "runs.json", ledger)
            self.assertGreater(verify()["failed"], 0)


if __name__ == "__main__":
    unittest.main()
