#!/usr/bin/env python3
"""Run a command in a verified, memory-limited systemd user scope."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

GIB = 1024**3
SLICE = "resource-work.slice"


def save(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-available-gib", type=int, default=12)
    parser.add_argument("--inside", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command or args.min_available_gib < 12:
        parser.error("supply a command and a minimum available RAM of at least 12 GiB")

    available = int(
        next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
             if line.startswith("MemAvailable:"))
    ) * 1024
    if available < args.min_available_gib * GIB:
        print(f"refused: available RAM {available} bytes; command not started", file=sys.stderr)
        return 75

    if args.inside is None:
        identifier = uuid.uuid4().hex
        state = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state")))
        directory = state / "resource-guard" / identifier
        directory.mkdir(parents=True)
        path = directory / "run.json"
        unit = f"resource-job-{identifier}.scope"
        record = {
            "started_at_unix": time.time(), "unit": unit, "slice": SLICE,
            "command": command, "cwd": str(Path.cwd()),
            "available_ram_before_bytes": available, "status": "launching",
        }
        save(path, record)
        print(f"resource guard: {path}", file=sys.stderr, flush=True)
        invocation = [
            "systemd-run", "--user", "--scope", "--collect", f"--unit={unit}",
            f"--slice={SLICE}", "--property=MemoryHigh=4G",
            "--property=MemoryMax=8G", "--property=MemorySwapMax=0",
            "--property=OOMPolicy=continue", "--",
            sys.executable, str(Path(__file__).resolve()),
            "--min-available-gib", str(args.min_available_gib),
            "--inside", str(path), "--", *command,
        ]
        result = subprocess.run(invocation, check=False)
        record = json.loads(path.read_text())
        exit_code = result.returncode if result.returncode >= 0 else 128 - result.returncode
        record["launcher_exit"] = exit_code
        if record["status"] in ("launching", "admitted", "running"):
            record.update(status="scope_terminated", finished_at_unix=time.time())
            record.setdefault("command_exit", None)
            record.setdefault("memory_peak_bytes", None)
            record.setdefault("memory_events", None)
            if exit_code == 0:
                exit_code = 78
        save(path, record)
        return exit_code

    path = args.inside
    record = json.loads(path.read_text())
    try:
        relative = next(
            line.removeprefix("0::") for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        group = Path("/sys/fs/cgroup") / relative.lstrip("/")
        limits = {name: (group / name).read_text().strip()
                  for name in ("memory.high", "memory.max", "memory.swap.max")}
        if limits != {"memory.high": str(4 * GIB), "memory.max": str(8 * GIB),
                      "memory.swap.max": "0"}:
            raise ValueError(f"unexpected job limits: {limits}")
        parent = next(p for p in group.parents if p.name == SLICE)
        parent_limits = {name: (parent / name).read_text().strip()
                         for name in ("memory.max", "memory.swap.max")}
        if parent_limits != {"memory.max": str(16 * GIB), "memory.swap.max": "0"}:
            raise ValueError(f"unexpected aggregate limits: {parent_limits}")
        record.update({
            "wrapper_pid": os.getpid(), "cgroup": relative,
            "verified_job_limits": limits, "verified_aggregate_limits": parent_limits,
            "available_ram_admission_bytes": available,
            "status": "admitted",
        })
        save(path, record)
        environment = os.environ.copy()
        environment.update(CARGO_BUILD_JOBS="2", RUST_TEST_THREADS="1")
        child = subprocess.Popen(command, env=environment)
        record.update(command_pid=child.pid, status="running")
        save(path, record)
        result = child.wait()
        record.update(
            status="finished", command_exit=result, finished_at_unix=time.time(),
            memory_peak_bytes=int((group / "memory.peak").read_text()),
            memory_events={key: int(value) for key, value in
                           (line.split() for line in (group / "memory.events").read_text().splitlines())},
        )
        save(path, record)
        return result if result >= 0 else 128 - result
    except (OSError, ValueError, StopIteration) as error:
        record.update(status="failed" if "command_pid" in record else "refused",
                      error=str(error), finished_at_unix=time.time())
        save(path, record)
        print(f"resource guard refused: {error}", file=sys.stderr)
        return 78


if __name__ == "__main__":
    sys.exit(main())
