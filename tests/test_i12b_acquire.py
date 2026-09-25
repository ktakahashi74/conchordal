"""Fixture tests for the sealed I12b acquisition boundary."""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "target/i12b-20260924/acquire.py"
SPEC = importlib.util.spec_from_file_location("i12b_acquire", SCRIPT)
acquire = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acquire)
sys.path.insert(0, str(SCRIPT.parents[2] / "scripts"))
import evaluate_i12b as evaluator  # noqa: E402


class AcquisitionTests(unittest.TestCase):
    def test_evaluator_accepts_complete_runner_ledger_and_rejects_audit_mutation(self):
        plan = acquire.registered_plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provenance = {
                "candidate_seal_sha256": "fixture-seal",
                "configs": {row["config"]: {"path": str(root / row["config"])}
                            for row in plan["run_order"]},
                "binaries": {role: {"path": role} for role in
                             ("new-render", "new-instrument", "old-instrument")},
            }
            (root / "acquisition-provenance.json").write_text(json.dumps(provenance))
            launched = []

            class Child:
                def __init__(self, cmd, **_kwargs):
                    row = plan["run_order"][len(launched)]
                    launched.append(cmd)
                    paths = acquire.artifact_paths(root, row)
                    for kind in ("wav", "report"):
                        if kind in paths:
                            paths[kind].write_bytes(b"fixture\n")
                    if "profile" in paths:
                        shared = ([{"period_accent_cut_deliveries": 1}]
                                  if row["arm"] == "backward-cut" else [{}])
                        paths["profile"].write_text(json.dumps({
                            "hops": [{}], "truncated": False, "dropped_hops": 0,
                            "summary": {"hop_count": 1},
                            "background": {"arrival_known_decisions": 1, "shared": shared},
                        }))
                    self.pid = 12345

                def wait(self):
                    return 0

            def system_command(cmd, **kwargs):
                outputs = {
                    ("pmset", "-g", "batt"): "Now drawing from 'AC Power'\n",
                    ("ioreg", "-r", "-k", "AppleClamshellState", "-d", "1"):
                        '"AppleClamshellState" = No\n',
                    ("pmset", "-g", "assertions"):
                        "pid 88(caffeinate): PreventUserIdleSystemSleep\n"
                        f" Details: caffeinate asserting on behalf of 'python' (pid {acquire.os.getpid()})\n",
                    ("ps", "-axo", "pid=,comm="): "12345 fixture-render\n",
                    ("pmset", "-g", "log"): "fixture power log without events\n",
                }
                output = outputs[tuple(cmd)]
                if "stdout" in kwargs:
                    kwargs["stdout"].write(output.encode())
                if kwargs.get("text"):
                    return acquire.subprocess.CompletedProcess(cmd, 0, output, "")
                return acquire.subprocess.CompletedProcess(cmd, 0, output.encode(), b"")

            with (mock.patch.object(acquire.subprocess, "run", side_effect=system_command),
                  mock.patch.object(acquire.subprocess, "Popen", side_effect=Child),
                  redirect_stdout(io.StringIO())):
                self.assertEqual(acquire.acquire_locked(plan, provenance, root, 192), 192)
            runs, ledger = evaluator.acquired_runs(root, plan, provenance)
            self.assertEqual(len(runs), 192)
            self.assertEqual(len(launched), 192)

            first = ledger["runs"][0]
            audit_path = root / first["pmset_log"]["path"]
            audit = json.loads(audit_path.read_text())
            audit["exit"] = 1
            audit_path.write_text(json.dumps(audit))
            first["pmset_log"].update(bytes=audit_path.stat().st_size,
                                      sha256=evaluator.digest(audit_path))
            (root / "runs.json").write_text(json.dumps(ledger))
            with self.assertRaisesRegex(ValueError, "run pmset audit differs"):
                evaluator.acquired_runs(root, plan, provenance)

    def test_registered_order_is_complete_and_phase_commands_use_correct_binary(self):
        plan = acquire.registered_plan()
        rows = plan["run_order"]
        self.assertEqual([r["sequence"] for r in rows], list(range(1, 193)))
        self.assertEqual([r["phase"] for r in rows].count("performance"), 144)
        self.assertEqual([r["phase"] for r in rows].count("offline-effect"), 36)
        self.assertEqual([r["phase"] for r in rows].count("offline-none-regression"), 12)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configs = {r["config"]: {"path": str(root / r["config"])} for r in rows}
            binaries = {role: {"path": role} for role in
                        ("new-render", "new-instrument", "old-instrument")}
            for row in rows:
                command = acquire.command(row, configs, binaries, root)
                self.assertEqual(command[0], row["binary_role"])
                if row["phase"] == "performance":
                    self.assertIn("--profile", command)
                    self.assertNotIn("-o", command)
                    self.assertNotIn("wav", acquire.artifact_paths(root, row))
                else:
                    self.assertIn("-o", command)
                    self.assertNotIn("--profile", command)
                self.assertEqual("--report" in command, row["mode"] == "report")

    def test_seal_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / "seal.json"
            file.write_text("{}")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                acquire.candidate_seal(file, "0" * 64)

    def test_sealed_runner_hash_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / "seal.json"
            required = {
                "target/i12b-20260924/acquire.py",
                "target/i12b-20260924/run-acquire.sh",
                "tests/test_i12b_acquire.py",
                "scripts/evaluate_i12b.py",
                "tests/test_evaluate_i12b.py",
                "scripts/verify_i11_stage2.py",
                "scripts/evaluate_i11_stage2_offline.py",
                "scripts/compare_i11_candidate_records.py",
                "scripts/locate_i11_first_divergence.py",
                "scripts/verify_i11_stage1.py",
                "docs/roadmap/temporal-dcc/i11-inputs/bit_identity.py",
                "docs/roadmap/temporal-dcc/i11-2-inputs/acquire.py",
                "docs/roadmap/temporal-dcc/i11-2-inputs/validate_inputs.py",
            }
            seal = {
                "registration_freeze": {"path": str(acquire.REGISTERED / "freeze.json"),
                                        "sha256": acquire.FREEZE_SHA},
                "verifier_hashes": {name: "0" * 64 for name in required},
                "candidate_manifest": {"path": "unused", "sha256": "0" * 64},
            }
            file.write_text(json.dumps(seal))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                acquire.candidate_seal(file, acquire.digest(file))

    def test_seal_rejects_missing_transitive_verifier(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / "seal.json"
            sealed = {"registration_freeze": {"path": str(acquire.REGISTERED / "freeze.json"),
                                               "sha256": acquire.FREEZE_SHA},
                      "verifier_hashes": {
                          "target/i12b-20260924/acquire.py": "0" * 64,
                          "target/i12b-20260924/run-acquire.sh": "0" * 64,
                          "tests/test_i12b_acquire.py": "0" * 64,
                          "scripts/evaluate_i12b.py": "0" * 64,
                          "tests/test_evaluate_i12b.py": "0" * 64,
                          "scripts/verify_i11_stage2.py": "0" * 64,
                          "scripts/evaluate_i11_stage2_offline.py": "0" * 64,
                          "scripts/compare_i11_candidate_records.py": "0" * 64,
                          "scripts/locate_i11_first_divergence.py": "0" * 64,
                          "scripts/verify_i11_stage1.py": "0" * 64,
                          "docs/roadmap/temporal-dcc/i11-inputs/bit_identity.py": "0" * 64,
                          "docs/roadmap/temporal-dcc/i11-2-inputs/acquire.py": "0" * 64},
                      "candidate_manifest": {"path": "unused", "sha256": "0" * 64}}
            file.write_text(json.dumps(sealed))
            with self.assertRaisesRegex(ValueError, "omits acquisition scripts"):
                acquire.candidate_seal(file, acquire.digest(file))

    def test_started_run_refuses_resume_and_reports_pid_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "active-run.json").write_text(json.dumps({"sequence": 1, "pid": 99999999}))
            with self.assertRaisesRegex(ValueError, "started run 1 PID .*not alive"):
                acquire.acquire_locked({}, {}, root, 1)

    def test_caffeinate_must_name_the_python_pid(self):
        with mock.patch.object(acquire.os, "getpid", return_value=123):
            valid = ('pid 456(caffeinate): PreventUserIdleSystemSleep\n'
                     "Details: caffeinate asserting on behalf of '/opt/homebrew/bin/python3.12' (pid 123)")
            self.assertTrue(acquire.caffeinate_present(valid))
            self.assertFalse(acquire.caffeinate_present(valid.replace("pid 123", "pid 124")))

    def test_failed_before_power_is_saved_and_cannot_resume(self):
        plan = acquire.registered_plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "acquisition-provenance.json").write_text("{}")
            provenance = {"candidate_seal_sha256": "abc", "configs": {}, "binaries": {}}

            def failed_power(_root, stage, _sequence):
                return {"stage": stage, "passed": stage != "before"}

            def fake_pmset(path):
                path.write_text("")

            with (mock.patch.object(acquire, "power_state", side_effect=failed_power),
                  mock.patch.object(acquire, "pmset_log", side_effect=fake_pmset),
                  mock.patch.object(acquire, "control_events", return_value=[]),
                  mock.patch.object(acquire, "command", return_value=["fixture"]),
                  mock.patch.object(acquire.subprocess, "Popen") as popen):
                with self.assertRaisesRegex(ValueError, "power control differs before 1"):
                    acquire.acquire_locked(plan, provenance, root, 1)
                popen.assert_not_called()
            ledger = json.loads((root / "runs.json").read_text())
            self.assertEqual(ledger["runs"][0]["status"], "failed")
            self.assertFalse(ledger["runs"][0]["power_before"]["passed"])
            self.assertTrue((root / "active-run.json").exists())
            with self.assertRaisesRegex(ValueError, "started run 1"):
                acquire.acquire_locked(plan, provenance, root, 1)

    def test_mock_runs_follow_registered_order_and_resume_checks_artifacts(self):
        plan = acquire.registered_plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "acquisition-provenance.json").write_text("{}")
            provenance = {
                "candidate_seal_sha256": "abc",
                "configs": {"inputs/config-none.toml": {"path": "fixture-config"}},
                "binaries": {"new-render": {"path": "fixture-render"}},
            }
            launched = []

            class Child:
                def __init__(self, cmd, **_kwargs):
                    launched.append(cmd[1])
                    Path(cmd[cmd.index("-o") + 1]).write_bytes(b"wav")
                    Path(cmd[cmd.index("--report") + 1]).write_bytes(b"{}\n")
                    self.pid = 12345

                def wait(self):
                    return 0

            def good_power(_root, stage, _sequence):
                result = {"stage": stage, "sequence": _sequence, "passed": True}
                with (_root / "power-states.jsonl").open("a") as stream:
                    stream.write(json.dumps(result) + "\n")
                return result

            def fake_pmset(path):
                path.write_text("")

            def fake_run_pmset(path, _start, _end):
                path.write_text(json.dumps({"exit": 0, "events": []}))
                return []

            with (mock.patch.object(acquire, "power_state", side_effect=good_power),
                  mock.patch.object(acquire, "pmset_log", side_effect=fake_pmset),
                  mock.patch.object(acquire, "run_pmset_audit", side_effect=fake_run_pmset),
                  mock.patch.object(acquire, "control_events", return_value=[]),
                  mock.patch.object(acquire.subprocess, "Popen", side_effect=Child)):
                self.assertEqual(acquire.acquire_locked(plan, provenance, root, 2), 2)
            self.assertEqual(launched, [str(acquire.REPO / r["scenario"])
                                        for r in plan["run_order"][:2]])
            ledger = json.loads((root / "runs.json").read_text())
            self.assertEqual([r["status"] for r in ledger["runs"]], ["completed", "completed"])
            self.assertEqual([r["sequence"] for r in ledger["runs"]], [1, 2])
            self.assertFalse((root / "active-run.json").exists())
            acquire.verify_prior(plan, provenance, root, ledger)
            first_wav = acquire.artifact_paths(root, plan["run_order"][0])["wav"]
            first_wav.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "prior artifact differs"):
                acquire.verify_prior(plan, provenance, root, ledger)

    def test_post_run_control_failures_retain_exit_and_raw_hashes(self):
        plan = acquire.registered_plan()
        for failing_stage in ("after-power", "pmset-audit"):
            with self.subTest(failing_stage=failing_stage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "acquisition-provenance.json").write_text("{}")
                provenance = {
                    "candidate_seal_sha256": "abc",
                    "configs": {"inputs/config-none.toml": {"path": "fixture-config"}},
                    "binaries": {"new-render": {"path": "fixture-render"}},
                }

                class Child:
                    def __init__(self, cmd, **_kwargs):
                        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"wav")
                        Path(cmd[cmd.index("--report") + 1]).write_bytes(b"{}\n")
                        self.pid = 12345

                    def wait(self):
                        return 0

                def power(_root, stage, sequence):
                    if failing_stage == "after-power" and stage == "after":
                        raise RuntimeError("ioreg unavailable")
                    result = {"stage": stage, "sequence": sequence, "passed": True}
                    with (_root / "power-states.jsonl").open("a") as stream:
                        stream.write(json.dumps(result) + "\n")
                    return result

                def pmset(path):
                    path.write_text("")

                def audit(path, _start, _end):
                    path.write_text(json.dumps({"exit": 1, "events": []}))
                    raise RuntimeError("pmset unavailable")

                with (mock.patch.object(acquire, "power_state", side_effect=power),
                      mock.patch.object(acquire, "pmset_log", side_effect=pmset),
                      mock.patch.object(acquire, "run_pmset_audit", side_effect=audit),
                      mock.patch.object(acquire, "control_events", return_value=[]),
                      mock.patch.object(acquire.subprocess, "Popen", side_effect=Child)):
                    with self.assertRaisesRegex(ValueError, "acquisition stopped; raw retained"):
                        acquire.acquire_locked(plan, provenance, root, 1)
                ledger = json.loads((root / "runs.json").read_text())
                run = ledger["runs"][0]
                self.assertEqual(run["status"], "failed")
                self.assertEqual(run["exit"], 0)
                self.assertIsNotNone(run["finished_at"])
                for kind in ("stdout", "stderr", "wav", "report"):
                    artifact = run["artifacts"][kind]
                    self.assertEqual(artifact["sha256"], acquire.digest(root / artifact["path"]))
                session = json.loads((root / "control/session-001/status.json").read_text())
                self.assertEqual(session["completed_runs"], 0)
                self.assertTrue((root / "active-run.json").exists())
                with self.assertRaisesRegex(ValueError, "started run 1"):
                    acquire.acquire_locked(plan, provenance, root, 1)


if __name__ == "__main__":
    unittest.main()
