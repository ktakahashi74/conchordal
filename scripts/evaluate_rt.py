#!/usr/bin/env python3
"""Profile sustained, dense-onset and overload/recovery workloads."""

import argparse
import copy
import csv
import datetime as dt
import itertools
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_beta as beta


PROFILE_SCHEMA = {
    **dict.fromkeys("schema_version seed sample_rate hop_size dropped_hops hop_capacity".split(), "i"),
    **dict.fromkeys("scope allocation_scope audio_output".split(), "s"),
    **dict.fromkeys("report_enabled listener_enabled allocation_instrumented truncated".split(), "b"),
    "dcc_coupling_strength": "n", "hop_budget_us": "n", "audio": "?o", "summary": "o", "hops": "l",
}
PHASE_FIELDS = ("analysis_wait_us listener_wait_us landscape_update_us population_us "
                "reports_us render_route_us post_render_us").split()
RENDER_FIELDS = ("setup_us commands_us samples_us history_us observation_us capture_delivery_us").split()
HOP_SCHEMA = {
    "synthesis_us": "n", "rendered_tone_count": "i",
    "frame_idx": "i", "alive_voice_count": "i", "worker_allocations": "?o", "underrun_frames_total": "?i",
    **dict.fromkeys(["time_sec", "elapsed_us", *PHASE_FIELDS], "n"),
}
AUDIO_SCHEMA = {
    "backend": "s", "device_name": "s",
    **dict.fromkeys(("sample_rate channels ring_capacity_frames callback_count callback_frames_total "
                     "underrun_frames_total callback_errors_total").split(), "i"),
}
POLICY = {
    "timing": "adaptive is the instrument theta clock with accumulator rate 8; synchronized explicitly uses a 16 Hz shared scaffold, accumulator drive 32 and the existing two-gate refractory interval to maintain 8 Hz simultaneous onsets despite gate weighting; no musical emergence acceptance",
    "window": "complete hops contained in [warmup, warmup + duration); release tail excluded",
    "percentiles": "linear interpolation at (n-1)*p; process_hop includes enabled report work",
    "underruns": "difference of post-hop cumulative missing mono-frame snapshots around the retained window; not hardware xruns",
    "allocations": "successful Rust allocation/reallocation requests on the worker thread; excludes other threads and native malloc; bytes are not retained memory",
    "queue_drain": "generated frames minus ring capacity is a lower bound on dequeued frames; it must cover the measurement window end, with callback frames and wall pacing supporting that bound; excludes downstream hardware latency",
    "report": "report on/off also changes listener activation when DCC is zero; not a pure report-I/O comparison",
    "acceptance": "fixed live Voice count, device callback observed, counters present, allocations instrumented, plausible wall pacing, window underruns zero, callback errors zero and hop p99 within budget; no musical acceptance",
    "device_identity": "known null/dummy/loopback sinks excluded; backend/device names alone do not independently verify the physical output, especially ALSA default",
    "offline": "offline-check never establishes device RT acceptance; device failure is not silently retried offline",
    "recovery": "temporary added population; require baseline pass, observed overload and post-release pass; allow two simulated seconds for release/queue settling; never claim the overloaded interval passed",
    "dense": "pulse accumulator on the selected theta clock, one-cycle holds and 50 ms release; reject multiple onsets for a Voice in one report hop; adaptive requires all Voices and mean >=4 onsets/Voice/sec; synchronized additionally requires each Voice's 8 Hz count and 125 ms intervals within report-hop precision",
    "participation": "entrained/flow workloads use the actual participation presets with one-cycle holds, intrinsic renewal for flow and acoustic cadence for entrained; report-enabled windows require at least two onsets from every Voice, without imposing a common onset rate or phase; report-off activity is not independently verified",
}
SUMMARY_FIELDS = ("case mode voices body seed report_enabled dcc_coupling_strength status listener_enabled "
                  "device_rt_pass assessment_reason sample_rate hop_size hop_budget_us hop_count "
                  "window_start_sec window_end_sec alive_voice_count_min alive_voice_count_max elapsed_p95_us elapsed_p99_us elapsed_max_us over_budget_hops "
                  "worker_alloc_count worker_alloc_bytes allocations_per_hop bytes_per_hop "
                  "underrun_frames_delta callback_count callback_frames_total callback_errors_total "
                  "backend device_name device_channels ring_capacity_frames generated_frames minimum_dequeued_frames execution_wall_sec minimum_device_wall_sec physical_output_verification "
                  "workload timing recovery_pass overload_observed onset_count onsets_per_voice_sec error").split()


def require(record, schema, label):
    if not isinstance(record, dict):
        raise ValueError(f"{label}: expected object")
    for key, spec in schema.items():
        if key not in record:
            raise ValueError(f"{label}: missing {key}")
        value = record[key]
        if value is None and spec.startswith("?"):
            continue
        kind = spec.lstrip("?")
        valid = ((kind == "i" and type(value) is int and 0 <= value < 2**64)
                 or (kind == "n" and type(value) in (float, int) and math.isfinite(value) and value >= 0)
                 or (kind == "b" and type(value) is bool)
                 or (kind == "s" and isinstance(value, str) and bool(value))
                 or (kind == "o" and isinstance(value, dict))
                 or (kind == "l" and isinstance(value, list)))
        if not valid:
            raise ValueError(f"{label}: invalid {key}={value!r}")


def parse_profile(path, *, mode, voices, seed, report_enabled, dcc, warmup_sec, duration_sec, wall_sec=None):
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=beta.strict_object,
                      parse_float=beta.finite_float, parse_constant=beta.finite_float)
    require(data, PROFILE_SCHEMA, "profile")
    if data["schema_version"] not in (2, 3, 4):
        raise ValueError("unsupported profile schema_version")
    # Keep schema 2 readable for the saved population baseline; it has no nested timings.
    if data["schema_version"] >= 3:
        require(data, {"rendering_scope": "s"}, "profile")
    if data["schema_version"] == 4:
        require(data, {"background": "o"}, "profile")
        require(data["background"], {"scope": "s", "shared": "l", "body": "?o", "candidate_energy": "?o"}, "background")
        if len(data["background"]["shared"]) != 2:
            raise ValueError("background shared must contain both bus slots")
    if (data["seed"] != seed or data["report_enabled"] != report_enabled
            or data["dcc_coupling_strength"] != dcc
            or data["listener_enabled"] != (report_enabled or dcc > 0)):
        raise ValueError("profile seed/report/DCC/listener conditions differ from request")
    expected_output = "device" if mode == "device" else "no_device"
    if mode not in ("device", "offline-check") or data["audio_output"] != expected_output:
        raise ValueError("profile audio_output differs from requested mode")
    if (not data["sample_rate"] or not data["hop_size"] or data["truncated"] or data["dropped_hops"]
            or not data["hop_capacity"] or len(data["hops"]) > data["hop_capacity"]):
        raise ValueError("invalid dimensions or truncated/dropped profile hops")
    hop_sec = data["hop_size"] / data["sample_rate"]
    if not math.isclose(data["hop_budget_us"], hop_sec * 1e6, rel_tol=1e-5):
        raise ValueError("hop budget disagrees with sample rate/hop size")
    audio = data["audio"]
    if expected_output == "device":
        require(audio, AUDIO_SCHEMA, "audio")
        if (audio["sample_rate"] != data["sample_rate"] or not audio["channels"]
                or audio["ring_capacity_frames"] < data["hop_size"]):
            raise ValueError("invalid audio device configuration")
    elif audio is not None:
        raise ValueError("no_device requires null audio metadata")
    hops = data["hops"]
    if not hops:
        raise ValueError("profile has no hops")
    previous_underruns = 0
    for index, row in enumerate(hops):
        require(row, HOP_SCHEMA, f"hop {index}")
        if sum(row[key] for key in PHASE_FIELDS) > row["elapsed_us"] + 1e-6:
            raise ValueError("hop phase durations exceed total elapsed time")
        if row["synthesis_us"] > row["render_route_us"] + 1e-6:
            raise ValueError("synthesis duration exceeds render/route duration")
        if data["schema_version"] >= 3:
            require(row, {"rendering": "o"}, f"hop {index}")
            require(row["rendering"], dict.fromkeys(RENDER_FIELDS, "n"), f"hop {index} rendering")
            if sum(row["rendering"][key] for key in RENDER_FIELDS) > row["synthesis_us"] + 1e-6:
                raise ValueError("renderer phase durations exceed synthesis time")
        if (row["frame_idx"] != index or not math.isclose(row["time_sec"], index * hop_sec,
                                                         rel_tol=2e-7, abs_tol=2e-6)):
            raise ValueError("hop sequence/time is incomplete or unordered")
        allocations = row["worker_allocations"]
        if data["allocation_instrumented"]:
            require(allocations, {"count": "i", "bytes": "i"}, f"hop {index} allocations")
        elif allocations is not None:
            raise ValueError("uninstrumented allocations must be null")
        underruns = row["underrun_frames_total"]
        if expected_output == "device":
            if underruns is None or underruns < previous_underruns:
                raise ValueError("device underrun counter missing or decreasing")
            previous_underruns = underruns
        elif underruns is not None:
            raise ValueError("no_device underrun counter must be null")
    if audio and audio["underrun_frames_total"] < previous_underruns:
        raise ValueError("final underrun counter precedes hop counter")
    summary = data["summary"]
    require(summary, {"hop_count": "i", "elapsed_p99_us": "?n", "elapsed_max_us": "?n",
                      "over_budget_hops": "i"}, "summary")
    if (summary["hop_count"] != len(hops) or summary["elapsed_p99_us"] is None
            or summary["elapsed_max_us"] != max(r["elapsed_us"] for r in hops)
            or summary["over_budget_hops"] != sum(r["elapsed_us"] > data["hop_budget_us"] for r in hops)):
        raise ValueError("profile summary disagrees with hop records")
    end_sec = warmup_sec + duration_sec
    if len(hops) * hop_sec + 2e-6 < end_sec:
        raise ValueError("profile does not cover complete measurement window")
    selected = [r for r in hops if r["frame_idx"] * hop_sec >= warmup_sec - 1e-9
                and (r["frame_idx"] + 1) * hop_sec <= end_sec + 1e-9]
    if not selected:
        raise ValueError("measurement window contains no complete hops")
    if any(r["alive_voice_count"] != voices for r in selected):
        raise ValueError("live Voice count differs from requested fixed population in measurement window")
    first, last = selected[0]["frame_idx"], selected[-1]["frame_idx"]
    underrun_start = (hops[first - 1]["underrun_frames_total"] if first else 0) if audio else None
    underrun_end = selected[-1]["underrun_frames_total"]
    delta = underrun_end - underrun_start if audio else None
    count = sum(r["worker_allocations"]["count"] for r in selected) if data["allocation_instrumented"] else None
    size = sum(r["worker_allocations"]["bytes"] for r in selected) if data["allocation_instrumented"] else None
    elapsed = beta.stats([r["elapsed_us"] for r in selected])
    generated_frames = len(hops) * data["hop_size"]
    minimum_dequeued = max(0, generated_frames - audio["ring_capacity_frames"]) if audio else None
    minimum_wall = max(0.0, minimum_dequeued / data["sample_rate"] - .5) if audio else None
    known_virtual = bool(audio and re.search(r"\b(null|dummy|discard|loopback)\b|virtual\s+(sink|output)|no\s+sound",
                                             audio["device_name"], re.IGNORECASE))
    if not audio:
        passed, reason = None, "offline_check_no_device_acceptance"
    elif known_virtual:
        passed, reason = None, "known_virtual_audio_sink"
    elif not audio["callback_count"] or not audio["callback_frames_total"]:
        passed, reason = None, "device_callback_not_observed"
    elif not data["allocation_instrumented"]:
        passed, reason = None, "worker_allocations_not_instrumented"
    elif minimum_dequeued < (last + 1) * data["hop_size"]:
        passed, reason = None, "measurement_window_not_drained"
    elif audio["callback_frames_total"] < minimum_dequeued:
        passed, reason = None, "callback_frames_below_dequeue_bound"
    elif wall_sec is None or not math.isfinite(wall_sec) or wall_sec < 0:
        passed, reason = None, "execution_wall_time_missing"
    elif wall_sec < minimum_wall:
        passed, reason = None, "device_pacing_not_demonstrated"
    else:
        passed = delta == 0 and audio["callback_errors_total"] == 0 and elapsed["p99"] <= data["hop_budget_us"]
        reason = "device_criteria_met" if passed else "device_criteria_failed"
    return {"device_rt_pass": passed, "assessment_reason": reason,
            "listener_enabled": data["listener_enabled"], "sample_rate": data["sample_rate"],
            "hop_size": data["hop_size"], "hop_budget_us": data["hop_budget_us"], "hop_count": len(selected),
            "window_start_sec": first * hop_sec, "window_end_sec": (last + 1) * hop_sec,
            "requested_window_start_sec": warmup_sec, "requested_window_end_sec": end_sec,
            "alive_voice_count_min": min(r["alive_voice_count"] for r in selected),
            "alive_voice_count_max": max(r["alive_voice_count"] for r in selected),
            "elapsed_p95_us": elapsed["p95"], "elapsed_p99_us": elapsed["p99"], "elapsed_max_us": elapsed["max"],
            "over_budget_hops": sum(r["elapsed_us"] > data["hop_budget_us"] for r in selected),
            "worker_alloc_count": count, "worker_alloc_bytes": size,
            "allocations_per_hop": count / len(selected) if count is not None else None,
            "bytes_per_hop": size / len(selected) if size is not None else None,
            "underrun_frames_delta": delta, "underrun_counter_start": underrun_start,
            "underrun_counter_end": underrun_end, "allocation_instrumented": data["allocation_instrumented"],
            "scope": data["scope"], "allocation_scope": data["allocation_scope"],
            "generated_frames": generated_frames, "minimum_dequeued_frames": minimum_dequeued,
            "execution_wall_sec": wall_sec, "minimum_device_wall_sec": minimum_wall,
            "physical_output_verification": "known_virtual" if known_virtual else "not_independently_verified" if audio else "no_device",
            "device_limitations": POLICY["device_identity"],
            **{key: beta.stats([r[key] for r in selected]) for key in PHASE_FIELDS},
            "synthesis_us": beta.stats([r["synthesis_us"] for r in selected]),
            "rendering_scope": data.get("rendering_scope"),
            "rendering": {key: beta.stats([r["rendering"][key] for r in selected])
                          for key in RENDER_FIELDS} if data["schema_version"] >= 3 else None,
            "background_terminal": data.get("background"),
            "rendered_tone_count": beta.stats([r["rendered_tone_count"] for r in selected]),
            **{key: audio[key] if audio else None for key in (
                "callback_count", "callback_frames_total", "callback_errors_total", "backend", "device_name", "ring_capacity_frames")},
            "device_channels": audio["channels"] if audio else None, "device": audio}


def scenario_text(voices, body, seed, warmup_sec, duration_sec, *, workload="sustain", overload_voices=512, stress_sec=5, timing="adaptive"):
    modes = ".modes(harmonic_modes().count(8))" if body == "harmonic" else ""
    accumulator_rate = 32.0 if timing == "synchronized" else 8.0
    phonation = ".sustain()" if workload == "sustain" else (
        f".pulse({accumulator_rate:.1f}).pulse_lock(0.0).social(0.0).cycles(1)\n"
        "    .attack_cost_fraction(0.0).rhythm_coupling_vitality(0.0, 1.0)")
    if workload in ("entrained", "flow"):
        phonation = f".{workload}().cycles(1).attack_cost_fraction(0.0)"
    envelope = "0.02, 0.10, 0.85, 0.25" if workload == "sustain" else "0.003, 0.025, 0.30, 0.05"
    amplitude_count = overload_voices if workload == "recovery" else voices
    scaffold = "set_scaffold_shared(16.0); // Explicit synchronization performance assay.\n" if timing == "synchronized" else ""
    stress = f'''
    let extra = place(voice, line(110.0, 880.0).count({overload_voices - voices}));
    wait({stress_sec:.9f});
    release(extra);
    wait(2.0);
    wait({duration_sec:.9f});
''' if workload == "recovery" else ""
    return f'''// Performance assay; no respawn, dedicated beat carrier or audio-file output.
seed({seed});
{scaffold}\
let voice = {body}()
    {modes}
    {phonation}
    .endurance(1000000.0)
    .sustain_drive(0.002)
    .amp({0.06 / math.sqrt(amplitude_count):.12f})
    .seek_consonance()
    .adsr({envelope});
section("RT workload", || {{
    let population = place(voice, line(110.0, 880.0).count({voices}));
    wait({warmup_sec:.9f});
    wait({duration_sec:.9f});
{stress}
    release(population);
    wait(2.0);
}});
'''


def onset_windows(path, windows, *, timing="adaptive", hop_sec=None):
    counts = [0] * len(windows)
    voices = [set() for _ in windows]
    voice_counts = [{} for _ in windows]
    last_onset_times = [{} for _ in windows]
    intervals = [[] for _ in windows]
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line, object_pairs_hook=beta.strict_object,
                             parse_float=beta.finite_float, parse_constant=beta.finite_float)
            if not isinstance(row, dict):
                raise ValueError("report record must be an object")
            if row.get("type") != "onset":
                continue
            require(row, {"time_sec": "n", "voice_id": "i"}, "onset")
            for index, (start, end, _) in enumerate(windows):
                if start <= row["time_sec"] < end:
                    if timing == "synchronized" and row.get("scaffold_mode") != "shared":
                        raise ValueError("synchronized workload did not report the shared scaffold")
                    previous = last_onset_times[index].get(row["voice_id"])
                    if previous is not None and row["time_sec"] <= previous:
                        raise ValueError("dense workload repeats a Voice onset within one report hop or has unordered onsets")
                    if previous is not None:
                        intervals[index].append(row["time_sec"] - previous)
                    last_onset_times[index][row["voice_id"]] = row["time_sec"]
                    counts[index] += 1
                    voices[index].add(row["voice_id"])
                    voice_counts[index][row["voice_id"]] = voice_counts[index].get(row["voice_id"], 0) + 1
    result = []
    for (start, end, expected_voices), count, observed, gaps, per_voice in zip(windows, counts, voices, intervals, voice_counts):
        rate = count / ((end - start) * expected_voices)
        if timing in ("entrained", "flow"):
            if len(observed) != expected_voices or min(per_voice.values(), default=0) < 2:
                raise ValueError("participation workload did not demonstrate repeated onsets from every Voice")
        elif len(observed) != expected_voices or rate < 4.0:
            raise ValueError("dense workload did not demonstrate the required onset density/Voice coverage")
        if timing == "synchronized":
            if hop_sec is None or not math.isfinite(hop_sec) or hop_sec <= 0:
                raise ValueError("synchronized onset checks require the actual hop duration")
            if (any(abs(n - 8 * (end - start)) > 1.0 for n in per_voice.values())
                    or not gaps or any(abs(gap - .125) > hop_sec + 1e-4 for gap in gaps)):
                raise ValueError("synchronized workload did not maintain 8 Hz onset count and intervals")
        result.append({"onset_count": count, "onsets_per_voice_sec": rate,
                       "onset_voice_count": len(observed), "one_onset_per_voice_per_hop": True,
                       "onset_interval_sec": beta.stats(gaps) if gaps else None})
    return result


def recovery_assessment(baseline, overloaded, recovered):
    if any(window["device_rt_pass"] is None for window in (baseline, overloaded, recovered)):
        return {"recovery_pass": None, "overload_observed": None, "assessment_reason": "recovery_device_evidence_incomplete"}
    observed = overloaded["underrun_frames_delta"] > 0 or overloaded["elapsed_p99_us"] > overloaded["hop_budget_us"]
    passed = baseline["device_rt_pass"] and recovered["device_rt_pass"] if observed else None
    return {"recovery_pass": passed, "overload_observed": observed,
            "assessment_reason": "overload_not_observed" if not observed else "recovery_criteria_met" if passed else "recovery_criteria_failed"}


def config_text(base, dcc):
    config = copy.deepcopy(base)
    config.setdefault("dcc", {})["coupling_strength"] = dcc
    config.setdefault("playback", {}).update(wait_user_start=False, wait_user_exit=False)
    lines = []

    def emit(table, prefix):
        if prefix:
            lines.append("[" + ".".join(json.dumps(p) for p in prefix) + "]")
        for key, value in table.items():
            if isinstance(value, dict):
                continue
            if type(value) not in (str, bool, int, float):
                raise ValueError(f"unsupported config value: {key}")
            lines.append(f"{json.dumps(key)} = {json.dumps(value, ensure_ascii=False, allow_nan=False)}")
        lines.append("")
        for key, value in table.items():
            if isinstance(value, dict):
                emit(value, prefix + [key])

    emit(config, [])
    return "\n".join(lines)


def write_results(output, cases, manifest):
    beta.dump_json(output / "manifest.json", manifest)
    beta.dump_json(output / "summary.json", {"status": manifest["status"], "metric_policy": POLICY, "cases": cases})
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cases)


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("device", "offline-check"), default="device")
    parser.add_argument("--voices", type=int, nargs="+", choices=(4, 16, 64), default=[4, 16, 64])
    parser.add_argument("--bodies", nargs="+", choices=("sine", "harmonic", "modal"), default=["sine", "harmonic", "modal"])
    parser.add_argument("--workload", choices=("sustain", "dense", "recovery", "entrained", "flow"), default="sustain")
    parser.add_argument("--timing", choices=("adaptive", "synchronized"), default="adaptive",
                        help="synchronized uses an explicit 16 Hz theta scaffold for 8 Hz simultaneous-onset load")
    parser.add_argument("--reports", type=int, nargs="+", choices=(0, 1), default=[0, 1])
    parser.add_argument("--dcc", type=float, nargs="+", choices=(0.0, 0.25), default=[0.0, 0.25])
    parser.add_argument("--overload-voices", type=int, default=512, help="total live Voices during recovery workload overload")
    parser.add_argument("--stress-sec", type=float, default=5, help="overloaded interval before release and two-second settling")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--warmup-sec", type=float, default=5)
    parser.add_argument("--duration-sec", type=float, default=10, help="measurement duration, excluding warmup and two-second tail")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--config", type=Path, default=root / "config.toml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--binary", type=Path, help="existing release profile-alloc instrument; requires --skip-build")
    args = parser.parse_args(argv)
    if args.binary and not args.skip_build:
        parser.error("--binary requires --skip-build")
    if args.timing == "synchronized" and (args.workload not in ("dense", "recovery") or args.reports != [1]):
        parser.error("synchronized timing requires dense/recovery and --reports 1 to verify actual onsets")
    if (args.seed < 0 or args.seed >= 2**64 or len(set(args.voices)) != len(args.voices)
            or len(set(args.bodies)) != len(args.bodies) or not math.isfinite(args.warmup_sec)
            or len(set(args.reports)) != len(args.reports) or len(set(args.dcc)) != len(args.dcc)
            or args.warmup_sec < 0 or not math.isfinite(args.duration_sec) or args.duration_sec <= 0
            or not math.isfinite(args.timeout) or args.timeout <= 0
            or args.warmup_sec + args.duration_sec >= 100000):
        parser.error("require distinct conditions, u64 seed, finite nonnegative warmup, positive duration/timeout and total < 100000 s")
    if (not math.isfinite(args.stress_sec) or args.stress_sec <= 0
            or not 1 <= args.overload_voices <= 4096
            or (args.workload == "recovery" and args.overload_voices <= max(args.voices))):
        parser.error("require positive finite stress duration and overload Voices above baseline and at most 4096")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    requested_output = args.output or root / "target" / "rt-evaluation" / stamp
    if requested_output.exists() or requested_output.is_symlink():
        parser.error(f"output already exists: {requested_output}")
    output = requested_output.resolve()
    output.mkdir(parents=True)
    cases, manifest = [], {"created_utc": stamp, "status": "running", "mode": args.mode,
                          "metric_policy": POLICY, "commands": [], "seed": args.seed,
                          "warmup_sec": args.warmup_sec, "duration_sec": args.duration_sec, "release_tail_sec": 2,
                          "voices": args.voices, "bodies": args.bodies, "report_enabled": [bool(r) for r in args.reports],
                          "dcc_coupling_strength": args.dcc, "device_blocker": None,
                          "workload": args.workload, "timing": args.timing, "overload_voices": args.overload_voices if args.workload == "recovery" else None,
                          "stress_sec": args.stress_sec if args.workload == "recovery" else None,
                          "recovery_settle_sec": 2 if args.workload == "recovery" else None}
    try:
        manifest.update(beta.snapshot(root, output))
        base_path = args.config.resolve(strict=True)
        shutil.copy2(base_path, output / "config.toml")
        base_config = tomllib.loads((output / "config.toml").read_text(encoding="utf-8"))
        manifest["config"] = {"original_path": str(base_path), "sha256": beta.sha256(output / "config.toml")}
        manifest["platform"] = {"platform": platform.platform(), "machine": platform.machine(),
                                "logical_cpus": os.cpu_count(), "python": sys.version,
                                "rustc": subprocess.check_output(["rustc", "--version", "--verbose"], text=True, timeout=30).strip()}
        cpuinfo = Path("/proc/cpuinfo")
        manifest["platform"]["cpu_model"] = next(
            (line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
             if line.startswith("model name") and ":" in line), None) if cpuinfo.is_file() else platform.processor()
        manifest["platform"]["affinity_cpus"] = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        manifest["build_environment"] = {k: os.environ.get(k) for k in (
            "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET", "CARGO_TARGET_DIR")}
        manifest["build_mode"] = "skip_build_source_match_unverified" if args.skip_build else "cargo_release_locked_profile_alloc"
        write_results(output, cases, manifest)
        if not args.skip_build:
            build = beta.execute(["cargo", "build", "--release", "--locked", "--features", "profile-alloc",
                                  "--bin", "conchordal", "--message-format=json-render-diagnostics"],
                                 root, output / "build.log", max(900, args.timeout))
            manifest["commands"].append(build)
            if build["status"] != "ok":
                raise ValueError(f"release profile build {build['status']}; see build.log")
            artifacts = []
            for line in (output / "build.log").read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if (isinstance(record, dict) and record.get("reason") == "compiler-artifact"
                        and record.get("target", {}).get("name") == "conchordal" and record.get("executable")):
                    artifacts.append(Path(record["executable"]))
            if len(artifacts) != 1:
                raise ValueError("build must identify exactly one instrument executable")
            binary = artifacts[0]
        else:
            target = Path(os.environ.get("CARGO_TARGET_DIR", "target"))
            binary = args.binary or (target if target.is_absolute() else root / target) / "release" / (
                "conchordal.exe" if os.name == "nt" else "conchordal")
        for name, entry in manifest["source_files"].items():
            if not (root / name).exists() or beta.sha256(root / name) != entry["sha256"]:
                raise ValueError(f"source changed after snapshot: {name}")
        binary = binary.resolve(strict=True)
        saved_binary = output / "bin" / binary.name
        saved_binary.parent.mkdir()
        shutil.copy2(binary, saved_binary)
        manifest["binary"] = {"original_path": str(binary), "sha256": beta.sha256(saved_binary)}
        (output / "README.md").write_text(
            "# RT負荷比較\n\nこの実行は音声ファイルを保存しない。profile.json、config.toml、scenario.rhaiとログを各条件に保存する。\n"
            "profile-allocによるworker threadの割当計測を含む。analysis threadやcallback、native mallocは集計外。\n"
            "計測は指定窓内の完全なhopだけを使い、warmupとrelease尾部を除く。callbackの累積値は最後の状態、underrun差は計測窓の境界値。\n"
            "report有無でListenerTwinの起動条件も変わる。reportのI/Oだけの効果とは読まない。\n"
            "測定窓の全hopで指定Voice数を確認する。既知のnull/dummy/loopback出力と、進行時間に対し極端に短い実行は実機合格にしない。\n"
            "生成framesからring容量を引いた最低消費量が計測窓末尾へ達することを要求し、callback framesと実行時間もその下限を満たすか確認する。尾部が短く消費を確認できない条件は未判定。\n"
            "この消費確認はringからcallbackへの取り出しを扱い、後段のハードウェア出力遅延や物理的な出力先の証明を含まない。\n"
            "backendや機器名だけでは物理的な出力先を独立に確認できない。特にALSAのdefaultは転送先の実機確認を別に残す必要がある。\n"
            "offline-checkは実機合格を示さない。deviceで機器初期化に失敗した場合は残条件を省略し、device_blocker.jsonへ保存する。\n"
            "その後の機器不要の確認は、新しい出力先と明示的な --mode offline-check で別に実行する。\n"
            "device_rt_passは記載した性能条件だけの判定であり、試聴合格ではない。実行成功と性能合格を分けて読む。\n",
            encoding="utf-8")
        with (output / "README.md").open("a", encoding="utf-8") as stream:
            stream.write(f"\n今回のworkloadは`{args.workload}`。dense/recoveryはpulseと短いenvelopeを使う。adaptiveの蓄積率は8 Hzで、実際のonsetは適応的θ時計に依存する。\n"
                         "report有効時は、各窓で全Voiceの発音とVoiceあたり平均4回/秒以上を確認する。report無効時の発音密度は未検証。\n"
                         "recoveryは測定窓と同じ長さのbaseline・回復窓を持つ。追加Voiceを指定秒数後にreleaseし、2秒待って回復窓を開始する。\n"
                         "baselineの合格、負荷区間のp99超過または出力不足、回復窓の合格が揃えばrecovery_pass=true。\n"
                         "負荷超過なしは未判定。負荷区間を含む全体のdevice_rt_passはnullとし、phasesに各窓の数値を残す。\n")
            stream.write(f"\ntimingは`{args.timing}`。synchronizedは同期が目的の性能アッセイ。16 Hzの共有θ scaffold、蓄積率32、既存の2ゲート間隔制限で8回/秒を駆動し、各Voiceの回数・間隔を検査する。音楽的な創発の合否には使わない。\n")
            if args.workload in ("entrained", "flow"):
                stream.write("\nentrained/flowでは対応する参加プリセットを使い、拍送りを追加しない。上記のpulse用4回/秒基準は適用せず、report有効時は全Voiceが窓内で2回以上発音したことを要求する。共通の間隔や位相を強制する試験ではない。report無効時の発音の継続は未検証。\n")
        for voices, body, report, dcc in itertools.product(args.voices, args.bodies, map(bool, args.reports), args.dcc):
            slug = f"v{voices:02d}_{body}_report-{int(report)}_dcc-{dcc:.2f}"
            case = {"case": slug, "mode": args.mode, "voices": voices, "body": body, "seed": args.seed,
                    "report_enabled": report, "dcc_coupling_strength": dcc, "status": "pending", "device_rt_pass": None,
                    "workload": args.workload, "timing": args.timing}
            if manifest["device_blocker"]:
                case.update(status="skipped_device_unavailable", error="see device_blocker.json")
                cases.append(case)
                continue
            directory = output / slug
            directory.mkdir()
            script, config = directory / "scenario.rhai", directory / "config.toml"
            script.write_text(scenario_text(voices, body, args.seed, args.warmup_sec, args.duration_sec,
                                           workload=args.workload, overload_voices=args.overload_voices,
                                           stress_sec=args.stress_sec, timing=args.timing), encoding="utf-8")
            config.write_text(config_text(base_config, dcc), encoding="utf-8")
            command = [saved_binary, script, "--nogui", f"--play={'true' if args.mode == 'device' else 'false'}",
                       "--config", config, "--seed", str(args.seed), "--profile", directory / "profile.json"]
            if report:
                command += ["--report", directory / "report.jsonl"]
            print(f"start {slug} ({args.workload})", flush=True)
            try:
                run = beta.execute(command, output / "source", directory / "run.log", args.timeout)
                case["execution"] = run
                manifest["commands"].append(run)
                log = (directory / "run.log").read_text(encoding="utf-8", errors="replace")
                if args.mode == "device" and "Audio init failed:" in log:
                    manifest["device_blocker"] = {"case": slug, "reason": "audio_initialization_failed", "log": str(directory / "run.log")}
                    beta.dump_json(output / "device_blocker.json", manifest["device_blocker"])
                if run["status"] != "ok":
                    raise ValueError(f"instrument {run['status']} (exit {run['returncode']}); see run.log")
                baseline = parse_profile(directory / "profile.json", mode=args.mode, voices=voices, seed=args.seed,
                                         report_enabled=report, dcc=dcc, warmup_sec=args.warmup_sec,
                                         duration_sec=args.duration_sec, wall_sec=run.get("wall_sec"))
                case.update(baseline)
                if report and (not (directory / "report.jsonl").is_file() or not (directory / "report.jsonl").stat().st_size):
                    raise ValueError("requested report missing or empty")
                windows = [(args.warmup_sec, args.warmup_sec + args.duration_sec, voices)]
                if args.workload == "recovery":
                    stress_start = args.warmup_sec + args.duration_sec
                    recovery_start = stress_start + args.stress_sec + 2
                    windows.extend([(stress_start, stress_start + args.stress_sec, args.overload_voices),
                                    (recovery_start, recovery_start + args.duration_sec, voices)])
                    phases = {"baseline": baseline}
                    for label, (start, end, count) in zip(("overloaded", "recovered"), windows[1:]):
                        phases[label] = parse_profile(directory / "profile.json", mode=args.mode, voices=count,
                            seed=args.seed, report_enabled=report, dcc=dcc, warmup_sec=start,
                            duration_sec=end - start, wall_sec=run.get("wall_sec"))
                    case.update(recovery_assessment(phases["baseline"], phases["overloaded"], phases["recovered"]))
                    case.update(device_rt_pass=None, phases=phases)
                if args.workload != "sustain" and report:
                    onset_timing = args.workload if args.workload in ("entrained", "flow") else args.timing
                    onsets = onset_windows(directory / "report.jsonl", windows, timing=onset_timing,
                                           hop_sec=baseline["hop_size"] / baseline["sample_rate"])
                    case.update(onsets[0])
                    if args.workload == "recovery":
                        for phase, onset in zip(case["phases"].values(), onsets):
                            phase.update(onset)
                case["status"] = "ok"
                if args.mode == "device" and case["assessment_reason"] in (
                        "device_callback_not_observed", "known_virtual_audio_sink", "device_pacing_not_demonstrated"):
                    manifest["device_blocker"] = {"case": slug, "reason": case["assessment_reason"], "log": str(directory / "run.log")}
                    beta.dump_json(output / "device_blocker.json", manifest["device_blocker"])
            except (OSError, ValueError, TypeError, OverflowError) as error:
                case.update(status="error", device_rt_pass=None, recovery_pass=None, error=str(error))
            case["artifact_sha256"] = {p.name: beta.sha256(p) for p in directory.iterdir() if p.is_file()}
            beta.dump_json(directory / "metrics.json", case)
            cases.append(case)
            write_results(output, cases, manifest)
            print(f"{slug}: {case['status']}", flush=True)
            if case.get("execution", {}).get("status") == "interrupted":
                raise ValueError("campaign interrupted")
        manifest["status"] = "device_blocked" if manifest["device_blocker"] else (
            "complete" if all(c["status"] == "ok" for c in cases) else "failed")
    except (OSError, ValueError, TypeError, OverflowError, subprocess.SubprocessError) as error:
        manifest.update(status="failed", error=str(error))
    write_results(output, cases, manifest)
    print(output, flush=True)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
