import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_guarded.py"
GIB = 1024**3


class RunGuardedTests(unittest.TestCase):
    def test_low_ram_does_not_start_command(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "started"
            result = subprocess.run([
                sys.executable, str(SCRIPT), "--min-available-gib", "1000000000", "--",
                sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()",
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 75)
            self.assertFalse(marker.exists())

    def test_invalid_admission_threshold(self):
        result = subprocess.run([
            sys.executable, str(SCRIPT), "--min-available-gib", "0", "--", "true",
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)


@unittest.skipUnless(os.environ.get("RESOURCE_GUARD_LIVE_TESTS") == "1",
                     "requires the installed user slice and a host systemd user bus")
class LiveRunGuardedTests(unittest.TestCase):
    def test_limits_reach_descendants_and_exit_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "child.json"
            child = (
                "import json,os; from pathlib import Path; "
                "c=next(x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::')); "
                "p=Path('/sys/fs/cgroup')/c.lstrip('/'); "
                "print(json.dumps({'cgroup':c,'max':(p/'memory.max').read_text().strip(),"
                "'swap':(p/'memory.swap.max').read_text().strip(),"
                "'jobs':os.environ['CARGO_BUILD_JOBS'],'threads':os.environ['RUST_TEST_THREADS']}))"
            )
            parent = (
                "import subprocess,sys; from pathlib import Path; "
                f"r=subprocess.run([sys.executable,'-c',{child!r}],capture_output=True,text=True,check=True); "
                f"Path({str(marker)!r}).write_text(r.stdout); sys.exit(7)"
            )
            environment = os.environ.copy()
            environment.update(XDG_STATE_HOME=directory, CARGO_BUILD_JOBS="99", RUST_TEST_THREADS="99")
            result = subprocess.run([sys.executable, str(SCRIPT), "--", sys.executable, "-c", parent],
                                    capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 7, result.stderr)
            inherited = json.loads(marker.read_text())
            record = json.loads(next(Path(directory).glob("resource-guard/*/run.json")).read_text())
            self.assertEqual(inherited["cgroup"], record["cgroup"])
            self.assertEqual(inherited["max"], str(8 * GIB))
            self.assertEqual(inherited["swap"], "0")
            self.assertEqual((inherited["jobs"], inherited["threads"]), ("2", "1"))
            self.assertEqual(record["command_exit"], 7)
            self.assertEqual(record["launcher_exit"], 7)
            self.assertEqual(record["memory_events"]["oom_kill"], 0)

    def test_missing_parent_and_wrong_job_limit_refuse_before_execution(self):
        for parent, maximum in [("app.slice", "8G"), ("resource-work.slice", "256M")]:
            with self.subTest(parent=parent, maximum=maximum), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "run.json"
                path.write_text(json.dumps({"status": "launching"}))
                marker = Path(directory) / "started"
                command = [
                    "systemd-run", "--user", "--scope", "--collect",
                    f"--unit=resource-negative-{uuid.uuid4().hex}.scope", f"--slice={parent}",
                    "--property=MemoryHigh=4G", f"--property=MemoryMax={maximum}",
                    "--property=MemorySwapMax=0", "--", sys.executable, str(SCRIPT),
                    "--inside", str(path), "--", sys.executable, "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ]
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 78, result.stderr)
                self.assertFalse(marker.exists())
                self.assertEqual(json.loads(path.read_text())["status"], "refused")

    def test_success_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            environment["XDG_STATE_HOME"] = directory
            result = subprocess.run([sys.executable, str(SCRIPT), "--", "/usr/bin/true"],
                                    capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(next(Path(directory).glob("resource-guard/*/run.json")).read_text())
            self.assertEqual(record["status"], "finished")
            self.assertEqual(record["command_exit"], 0)
            self.assertGreater(record["memory_peak_bytes"], 0)

    def test_killed_wrapper_is_recorded_as_terminal_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            environment["XDG_STATE_HOME"] = directory
            result = subprocess.run([
                sys.executable, str(SCRIPT), "--", sys.executable, "-c",
                "import os,signal,time; time.sleep(.05); os.kill(os.getppid(),signal.SIGKILL)",
            ], capture_output=True, text=True, env=environment)
            self.assertNotEqual(result.returncode, 0, result.stderr)
            record = json.loads(next(Path(directory).glob("resource-guard/*/run.json")).read_text())
            self.assertEqual(record["status"], "scope_terminated")
            self.assertIsNone(record["command_exit"])
            self.assertIsNone(record["memory_events"])


if __name__ == "__main__":
    unittest.main()
