#!/usr/bin/env python3
"""Freeze actual worktree bytes and explicitly selected offline evidence for M0."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def digest(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()


def capture(root, output, evidence):
    root, output = Path(root).resolve(), Path(output).resolve()
    # Explicit evidence is allowed under ignored target/, never through symlinks.
    extras = set()
    for name in evidence:
        path = root / name
        if not path.exists() or path.is_symlink():
            raise ValueError(f"missing or symlink evidence: {name}")
        path.resolve().relative_to(root)
        if output == path.resolve() or path.resolve() in output.parents:
            raise ValueError("output must not be inside selected evidence")
        extras.update(p.relative_to(root).as_posix() for p in
                      (path.rglob("*") if path.is_dir() else [path])
                      if p.is_file() or p.is_symlink())
    git = lambda *args: subprocess.check_output(["git", "-C", str(root), *args])
    names = git("ls-files", "-z", "--cached", "--others", "--exclude-standard")
    names = {os.fsdecode(n) for n in names.split(b"\0") if n}
    names = {n for n in names if n != ".codex" and not n.startswith((".claude/", ".codex/"))}
    for name in names | extras:
        if output == root / name or output in (root / name).parents:
            raise ValueError("output is already part of the input tree")
    head = git("rev-parse", "HEAD").decode().strip()
    status = git("status", "--porcelain=v1", "--untracked-files=all").decode()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    archive_path = output / "baseline.tar.gz"
    with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
        for name in sorted(names | extras):
            path = root / name
            row = {"path": name, "role": "offline_evidence" if name in extras else "worktree"}
            if path.is_symlink():
                target = os.readlink(path)
                row.update(kind="symlink", target=target)
            elif path.is_file():
                with path.open("rb") as stream:
                    row.update(kind="file", bytes=path.stat().st_size, sha256=digest(stream))
            elif not path.exists():
                row.update(kind="deleted")
            else:
                raise ValueError(f"unsupported worktree entry: {name}")
            if row["kind"] != "deleted":
                archive.add(path, arcname=name, recursive=False)
            records.append(row)
    with archive_path.open("rb") as stream:
        archive_hash = digest(stream)
    manifest = {
        "schema": "temporal-dcc-baseline-v1", "captured_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(), "git_head": head, "git_status": status,
        "scope": "Actual bytes, including uncommitted research; excludes local agent settings.",
        "archive": {"path": archive_path.name, "sha256": archive_hash}, "files": records,
    }
    manifest_path = output / "baseline.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    verify(manifest_path)
    return manifest


def verify(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    archive_path = manifest_path.parent / manifest["archive"]["path"]
    with archive_path.open("rb") as stream:
        if digest(stream) != manifest["archive"]["sha256"]:
            raise ValueError("archive hash mismatch")
    expected = {r["path"]: r for r in manifest["files"] if r["kind"] != "deleted"}
    if len({r["path"] for r in manifest["files"]}) != len(manifest["files"]):
        raise ValueError("duplicate manifest paths")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError("archive inventory mismatch")
        for member in members:
            row = expected[member.name]
            if row["kind"] == "symlink":
                if not member.issym() or member.linkname != row["target"]:
                    raise ValueError(f"symlink mismatch: {member.name}")
            elif not member.isfile() or member.size != row["bytes"]:
                raise ValueError(f"file metadata mismatch: {member.name}")
            else:
                with archive.extractfile(member) as stream:
                    if digest(stream) != row["sha256"]:
                        raise ValueError(f"file content mismatch: {member.name}")
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("capture")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--evidence", action="append", default=[])
    check = commands.add_parser("verify")
    check.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.command == "capture":
        result = capture(args.root, args.output, args.evidence)
        print(json.dumps({"git_head": result["git_head"], "files": len(result["files"]),
                          "archive": result["archive"]}))
    else:
        print(json.dumps({"verified_files": verify(args.manifest)}))


if __name__ == "__main__":
    main()
