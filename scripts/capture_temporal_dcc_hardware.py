#!/usr/bin/env python3
"""Record the Linux hardware envelope without claiming runtime acceptance."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tomllib


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return {"command": args, "exit": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": args, "exit": None, "error": str(error)}


def capture(root):
    root = Path(root)
    observations = {}
    affinity = sorted(os.sched_getaffinity(0))
    paths = [Path("/proc/meminfo"), Path("/proc/asound/cards"), Path("/proc/asound/pcm"),
             Path("/proc/self/cgroup"), Path("/etc/os-release")]
    cgroup = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                  if line.startswith("0::"))
    folder = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    while folder.is_relative_to("/sys/fs/cgroup"):
        paths += [folder / name for name in ["cpu.max", "cpuset.cpus.effective", "memory.max"]]
        folder = folder.parent
    topology = []
    for cpu in affinity:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        pair = [(base / "topology" / n).read_text().strip()
                for n in ["physical_package_id", "core_id"]]
        topology.append((pair[0], pair[1]))
        paths += [base / "cpufreq" / "scaling_governor",
                  base / "cpufreq" / "energy_performance_preference"]
    for path in paths:
        try:
            observations[str(path)] = {"status": "observed", "text": path.read_text().strip()}
        except OSError as error:
            observations[str(path)] = {"status": "unavailable", "reason": str(error)}
    config = root / "config.toml"
    cargo = tomllib.loads((root / "Cargo.toml").read_text())
    return {
        "schema": "temporal-dcc-hardware-v1",
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "inventory_only_not_a_feasibility_pass",
        "platform": {"system": platform.system(), "kernel": platform.release(),
                     "architecture": platform.machine()},
        "cpu": {"affinity_logical_ids": affinity, "usable_logical_cores": len(affinity),
                "usable_physical_cores": len(set(topology)), "lscpu": command(["lscpu", "-J"]),
                "virtualization": command(["systemd-detect-virt"])},
        "observations": observations, "power_profile": command(["powerprofilesctl", "get"]),
        "compiler": command(["rustc", "-vV"]), "cargo": command(["cargo", "-V"]),
        "release_profile": cargo.get("profile", {}).get("release", {}),
        "build_flags": {name: os.environ.get(name) for name in
                        ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET"]},
        "runtime_config": {"path": "config.toml", "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                           "explicit_values": tomllib.loads(config.read_text())},
        "reference_operating_envelope": {
            "sample_rate_hz": 48000, "hop_samples": 512, "requested_latency_ms": 50,
            "callback_buffer": "cpal::BufferSize::Default; actual device size requires live profile",
            "bus_channels": {"habitat": ["mono"], "presentation": ["mono"]},
            "source_downmix": "arithmetic mean across source channels, then identical analysis/listening PCM",
            "device_mapping": "presentation mono duplicated into each negotiated output channel",
            "listener_tap": "before device guard", "guard": "peak-limiter unless explicitly overridden",
            "actual_device_profile": None, "physical_output_verified": False,
            "ordinary_workload": "64 Voice instrument and both analysis buses concurrent with both cognition workers",
            "warmup_sec": 60, "measurement_sec": 600, "worker_cycles_per_bus": 6000,
            "worker_p99_ms_max": 40, "voice_decision_p99_ms_max": 1.6,
            "combined_hop_p99_ms_max": 512 / 48000 * 1000 * 0.8,
        },
        "unverified": ["active physical output and negotiated channel/buffer configuration",
                       "ordinary-workload source and binary hashes",
                       "complete cognition and decision kernels, concurrent timing and checksums",
                       "transparent device guard and presented/analysed waveform correspondence"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = capture(args.root)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "logical_cores": result["cpu"]["usable_logical_cores"],
                      "physical_cores": result["cpu"]["usable_physical_cores"]}))


if __name__ == "__main__":
    main()
