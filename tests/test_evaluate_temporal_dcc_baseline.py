"""Exercise restoration evidence, dirty worktrees and tamper detection."""

import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from capture_temporal_dcc_baseline import capture, verify


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        self.output = Path(self.temp.name) / "frozen"
        self.git("init", "-q")
        (self.root / "source.rs").write_text("old source\n")
        (self.root / "deleted.rs").write_text("deleted later\n")
        (self.root / ".gitignore").write_text("target/\n")
        self.git("add", "source.rs", "deleted.rs", ".gitignore")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "-qm", "Fixture baseline")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args])

    def test_preserves_dirty_deleted_untracked_and_ignored_evidence(self):
        (self.root / "source.rs").write_text("uncommitted source\n")
        (self.root / "deleted.rs").unlink()
        (self.root / "new.py").write_text("new research\n")
        (self.root / ".claude").mkdir()
        (self.root / ".claude" / "local.json").write_text("private settings\n")
        (self.root / "target").mkdir()
        (self.root / "target" / "evidence.json").write_text('{"measured": true}\n')
        result = capture(self.root, self.output, ["target/evidence.json"])
        rows = {r["path"]: r for r in result["files"]}
        self.assertEqual(rows["deleted.rs"]["kind"], "deleted")
        self.assertEqual(rows["target/evidence.json"]["role"], "offline_evidence")
        self.assertNotIn(".claude/local.json", rows)
        with tarfile.open(self.output / "baseline.tar.gz") as archive:
            self.assertEqual(archive.extractfile("source.rs").read(), b"uncommitted source\n")
            self.assertEqual(archive.extractfile("new.py").read(), b"new research\n")
        (self.root / "source.rs").write_text("later implementation\n")
        self.assertEqual(verify(self.output / "baseline.json"), len(rows) - 1)
        with self.assertRaises(FileExistsError):
            capture(self.root, self.output, [])

    def test_checks_contents_against_manifest_and_archive_digest(self):
        capture(self.root, self.output, [])
        path = self.output / "baseline.json"
        original = path.read_text()
        document = json.loads(original)
        document["files"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "file content mismatch"):
            verify(path)
        path.write_text(original)
        with (self.output / "baseline.tar.gz").open("ab") as stream:
            stream.write(b"altered")
        with self.assertRaisesRegex(ValueError, "archive hash mismatch"):
            verify(path)

    def test_records_symlink_without_reading_external_target(self):
        secret = Path(self.temp.name) / "external"
        secret.write_text("not part of the repository\n")
        (self.root / "link").symlink_to(secret)
        result = capture(self.root, self.output, [])
        row = next(r for r in result["files"] if r["path"] == "link")
        self.assertEqual(row["kind"], "symlink")
        self.assertNotIn("sha256", row)
        self.assertEqual(verify(self.output / "baseline.json"), len(result["files"]))

    def test_rejects_missing_evidence_and_recursive_capture(self):
        with self.assertRaises(ValueError):
            capture(self.root, self.output, ["missing"])
        (self.root / "target").mkdir()
        with self.assertRaises(ValueError):
            capture(self.root, self.root / "target" / "recursive", ["target"])
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
