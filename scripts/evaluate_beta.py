#!/usr/bin/env python3
"""Render reproducible beta comparisons with the Python standard library."""

import argparse
import array
import collections
import csv
import datetime as dt
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import wave


DEFAULT_SAMPLES = ["07_heartbeat", "08_murmuration", "09_rain", "12_emergence_and_resolution"]
RHYTHM_FIELDS = "time_sec window_start_sec window_end_sec onset_density_hz".split()
RHYTHM_OPTIONAL = ("ioi_mean_sec ioi_cv beat_stability mean_plv kuramoto_order_mean "
                   "kuramoto_order_max sync_emergence_sec burstiness one_over_f_slope "
                   "ioi_one_over_f_slope").split()
LISTENER_FLOATS = ("stability_level resolvability_level tension_level attention_level beat_hz "
                   "beat_phase beat_confidence subdivision_confidence measure_hz "
                   "measure_confidence").split()
LISTENER_INTS = ("generated_frame_id analysis_frame_id analysis_lag_frames subdivision_ratio "
                 "measure_ratio").split()
# Each schema lists required fields, including required nullable fields.
SCHEMAS = {
    "meta": {"seed": "i", "hop_timing_scope": "s"},
    "scene_marker": {"time_sec": "n", "order": "i", "name": "s"},
    "spawn": dict.fromkeys("time_sec freq_hz".split(), "n") | dict.fromkeys(
        "population_id voice_id generation member_idx".split(), "i"),
    "respawn": dict.fromkeys("time_sec freq_hz".split(), "n") | dict.fromkeys(
        "population_id voice_id generation member_idx".split(), "i") | {"parent_id": "?i"},
    "death": dict.fromkeys("time_sec lifetime_sec first_k_mean".split(), "n") | dict.fromkeys(
        "population_id voice_id generation".split(), "i") | dict.fromkeys(
        "configured_endurance_sec energy_depletion_sec plv_at_death".split(), "?n"),
    "onset": dict.fromkeys("time_sec freq_hz strength".split(), "n") | dict.fromkeys(
        "population_id voice_id generation".split(), "i") | {
        "plv": "?n", "scaffold_mode": "s", "scaffold_phase_0_1": "?n"},
    "population_step": dict.fromkeys(
        "time_sec mean_freq_hz mean_c_field_score mean_c_field_level freq_entropy_log2".split(), "n")
        | dict.fromkeys("population_id alive_count".split(), "i"),
    "rhythm_observation": dict.fromkeys("time_sec env_open env_level".split(), "n")
        | dict.fromkeys("kuramoto_order_r theta_hz delta_hz".split(), "?n")
        | dict.fromkeys("kuramoto_active_count onsets_in_hop".split(), "i"),
    "listener_state": dict.fromkeys(["time_sec"] + LISTENER_FLOATS, "n")
        | dict.fromkeys(LISTENER_INTS, "i"),
    "dcc_pressure": dict.fromkeys("time_sec tension_pressure temperature_bonus".split(), "n"),
    "habituation": dict.fromkeys(
        "time_sec mean_h max_h mean_erosion tracked_h tracked_raw_score tracked_eff_score".split(), "n")
        | {"tracked_bin": "i"},
    "rhythm_summary": dict.fromkeys(RHYTHM_FIELDS, "n")
        | dict.fromkeys(RHYTHM_OPTIONAL, "?n") | {"population_id": "?i", "onset_count": "i"},
    "listener_confidence_summary": dict.fromkeys(
        "window_start_sec window_end_sec beat_confidence_peak beat_confidence_late_mean".split(), "n")
        | {"sample_count": "i"},
    "phonation_gate_open": dict.fromkeys("time_sec consonance".split(), "n")
        | dict.fromkeys("population_id voice_id".split(), "i"),
    "hop_timing": dict.fromkeys(
        "time_sec elapsed_us analysis_wait_us listener_wait_us hop_budget_us".split(), "n")
        | {"frame_idx": "i", "audio_output": "s", "underrun_frames_total": "?i"},
}
SUMMARY_FIELDS = ("case sample seed status wav_duration_sec wav_peak wav_rms clipping_fraction "
                  "silence_fraction all_silent onset_count onset_density_hz ioi_cv beat_stability mean_plv "
                  "kuramoto_order_mean sync_emergence_sec "
                  "listener_tension_mean listener_beat_confidence_peak hop_elapsed_p95_ms "
                  "hop_elapsed_p99_ms analysis_wait_p95_ms listener_wait_p95_ms error").split()
POLICY = {
    "silence": "abs(PCM16 sample) <= 1; fraction of samples, not silent segments",
    "clipping": "abs(PCM16 sample) >= 32767; renderer scales by 32767; not pre-limiter clipping",
    "hop_percentiles": "linear interpolation at (n-1)*p; exclude time_sec < warmup_sec",
    "timing_scope": "offline process_hop wall time; no audio device or real-time acceptance claim",
    "rhythm_scope": "all reported onsets, including habitat-only routing; IOI uses hop timestamps",
    "summary_scope": "rhythm window is first-to-last onset; listener late mean includes the final quarter/tail",
}


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def stats(values):
    return {"count": len(values), "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": math.fsum(values) / len(values) if values else None,
            "p95": percentile(values, .95), "p99": percentile(values, .99)}


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def finite_float(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    return value


def parse_report(path, seed, warmup_sec):
    records = collections.defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                record = json.loads(line, object_pairs_hook=strict_object, parse_float=finite_float,
                                    parse_constant=lambda value: finite_float(value))
                if not isinstance(record, dict) or record.get("type") not in SCHEMAS:
                    raise ValueError("unknown or missing record type")
                kind = record["type"]
                for name, spec in SCHEMAS[kind].items():
                    if name not in record:
                        raise ValueError(f"{kind}: missing {name}")
                    value = record[name]
                    if value is None and spec.startswith("?"):
                        continue
                    spec = spec.lstrip("?")
                    valid = ((spec == "s" and isinstance(value, str))
                             or (spec == "i" and type(value) is int and value >= 0)
                             or (spec == "n" and type(value) in (int, float) and math.isfinite(value)))
                    if not valid:
                        raise ValueError(f"{kind}: invalid {name}={value!r}")
                if number == 1 and kind != "meta":
                    raise ValueError("first record must be meta")
                records[kind].append(record)
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f"{path.name}:{number}: {error}") from error
    if len(records["meta"]) != 1 or records["meta"][0]["seed"] != seed:
        raise ValueError("missing/duplicate meta or effective seed differs from requested seed")
    summaries = records["rhythm_summary"]
    global_rows = [r for r in summaries if r["population_id"] is None]
    groups = [r for r in summaries if r["population_id"] is not None]
    if len(global_rows) != 1 or not records["listener_state"] or not records["hop_timing"]:
        raise ValueError("incomplete report: require one global rhythm summary, listener states and hop timings")
    counts = collections.Counter(r["population_id"] for r in records["onset"])
    if (global_rows[0]["onset_count"] != sum(counts.values())
            or len(groups) != len(counts)
            or {r["population_id"]: r["onset_count"] for r in groups} != dict(counts)):
        raise ValueError("rhythm summaries do not match recorded onset counts")
    confidence = records["listener_confidence_summary"]
    if len(confidence) != 1 or confidence[0]["sample_count"] != len(records["listener_state"]):
        raise ValueError("listener confidence summary does not match listener state count")
    timings = records["hop_timing"]
    if timings[0]["frame_idx"] != 0 or timings[0]["time_sec"] != 0:
        raise ValueError("hop timing must start at frame/time zero")
    for previous, current in zip(timings, timings[1:]):
        if current["frame_idx"] != previous["frame_idx"] + 1 or current["time_sec"] <= previous["time_sec"]:
            raise ValueError("hop timing sequence is incomplete or unordered")
    for row in timings:
        if row["audio_output"] != "no_device" or row["underrun_frames_total"] is not None:
            raise ValueError("offline timing must report no_device and null underrun_frames_total")
        if row["hop_budget_us"] != timings[0]["hop_budget_us"] or row["hop_budget_us"] <= 0 or any(row[key] < 0 for key in (
                "elapsed_us", "analysis_wait_us", "listener_wait_us")):
            raise ValueError("invalid hop timing duration")
    retained = [r for r in timings if r["time_sec"] >= warmup_sec]
    listener = records["listener_state"]
    return {"meta": records["meta"][0], "record_counts": {k: len(v) for k, v in records.items()},
            "rhythm": {"global": global_rows[0], "populations": groups},
            "scene_markers": records["scene_marker"],
            "listener": {"stats": {key: stats([r[key] for r in listener])
                                    for key in LISTENER_FLOATS + ["analysis_lag_frames"]},
                         "confidence_summary": records["listener_confidence_summary"], "series": listener},
            "hop_timing": {"warmup_sec": warmup_sec, "excluded_count": len(timings) - len(retained),
                           "total_count": len(timings), "hop_budget_us": timings[0]["hop_budget_us"],
                           "retained_count": len(retained), "underrun_frames_total": None,
                           "stats_us": {key: stats([r[key] for r in retained]) for key in
                                        ("elapsed_us", "analysis_wait_us", "listener_wait_us", "hop_budget_us")}}}


def wav_metrics(path):
    count = total_square = peak = clipped = silent = 0
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getcomptype() != "NONE":
            raise ValueError("WAV must be mono PCM16")
        rate, expected = stream.getframerate(), stream.getnframes()
        if rate <= 0 or expected <= 0:
            raise ValueError("WAV sample rate and frame count must be positive")
        while data := stream.readframes(65536):
            samples = array.array("h", data)
            if sys.byteorder != "little":
                samples.byteswap()
            count += len(samples)
            for value in samples:
                peak = max(peak, abs(value))
                total_square += value * value
                clipped += abs(value) >= 32767
                silent += abs(value) <= 1
        if count != expected:
            raise ValueError(f"truncated WAV: expected {expected} frames, read {count}")
    return {"sample_rate": rate, "frames": count, "duration_sec": count / rate,
            "peak": peak / 32768, "rms": math.sqrt(total_square / count) / 32768,
            "clipping_frames": clipped, "clipping_fraction": clipped / count,
            "silence_frames": silent, "silence_fraction": silent / count, "all_silent": peak <= 1}


def source_allowed(path):
    if path.is_absolute() or any(p.startswith(".") or p == "__pycache__" for p in path.parts):
        return False
    if len(path.parts) == 1:
        return path.name in {"Cargo.toml", "Cargo.lock", "build.rs", "AGENTS.md", "rust-toolchain.toml"}
    extensions = {"src": {".rs"}, "samples": {".rhai"}, "scripts": {".py"},
                  "tests": {".rs", ".rhai", ".py"}}
    return path.suffix in extensions.get(path.parts[0], set())


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, timeout=30)


def snapshot(root, output):
    tracked = {os.fsdecode(p) for p in git(root, "ls-files", "-z").split(b"\0") if p}
    untracked = {os.fsdecode(p) for p in git(root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0") if p}
    allowed = sorted(name for name in tracked | untracked if source_allowed(Path(name)))
    entries = {}
    for name in allowed:
        path = root / name
        if not path.exists():
            continue
        if path.is_symlink() or root not in path.resolve().parents or not path.is_file():
            raise ValueError(f"source snapshot rejects symlink/non-file: {name}")
        dest = output / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        entries[name] = {"sha256": sha256(dest), "tracked": name in tracked}
    for filename, arguments in [("tracked-head.patch", ["HEAD"]), ("staged.patch", ["--cached"]),
                                ("unstaged.patch", [])]:
        (output / filename).write_bytes(git(root, "diff", "--binary", *arguments, "--", *allowed))
    return {"head": git(root, "rev-parse", "HEAD").decode().strip(), "source_files": entries,
            "tracked_status": git(root, "status", "--porcelain=v1", "--untracked-files=no").decode(),
            "untracked_source_files": sorted(n for n in entries if n in untracked),
            "diff_scope": "allowlisted source paths; HEAD, staged and unstaged patches stored separately"}


def execute(command, cwd, log_path, timeout):
    started = time.monotonic()
    env = os.environ.copy()
    env["RUST_LOG"] = "warn"
    env.pop("CONCHORDAL_LIMITER", None)
    outcome = {"command": [str(v) for v in command], "cwd": str(cwd), "log": str(log_path),
               "timeout_sec": timeout, "environment_overrides": {"RUST_LOG": "warn", "CONCHORDAL_LIMITER": None}}
    with log_path.open("wb") as log:
        process = subprocess.Popen(outcome["command"], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=(os.name == "posix"))
        try:
            outcome["returncode"] = process.wait(timeout=timeout)
            outcome["status"] = "ok" if process.returncode == 0 else "error"
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            outcome.update(returncode=process.returncode,
                           status="timeout" if isinstance(error, subprocess.TimeoutExpired) else "interrupted")
    outcome["wall_sec"] = time.monotonic() - started
    return outcome


def write_tables(output, cases, status):
    rows = []
    for case in cases:
        row = {key: case.get(key, "") for key in ("case", "sample", "seed", "status", "error")}
        if "audio" in case:
            audio = case["audio"]
            row.update(wav_duration_sec=audio["duration_sec"], wav_peak=audio["peak"], wav_rms=audio["rms"],
                       clipping_fraction=audio["clipping_fraction"], silence_fraction=audio["silence_fraction"],
                       all_silent=audio["all_silent"])
        if "report" in case:
            report = case["report"]
            row.update({key: report["rhythm"]["global"][key] for key in
                        ("onset_count", "onset_density_hz", "ioi_cv", "beat_stability", "mean_plv",
                         "kuramoto_order_mean", "sync_emergence_sec")})
            row["listener_tension_mean"] = report["listener"]["stats"]["tension_level"]["mean"]
            row["listener_beat_confidence_peak"] = report["listener"]["stats"]["beat_confidence"]["max"]
            timing = report["hop_timing"]["stats_us"]
            for column, key, stat in [("hop_elapsed_p95_ms", "elapsed_us", "p95"),
                                      ("hop_elapsed_p99_ms", "elapsed_us", "p99"),
                                      ("analysis_wait_p95_ms", "analysis_wait_us", "p95"),
                                      ("listener_wait_p95_ms", "listener_wait_us", "p95")]:
                row[column] = timing[key][stat] / 1000 if timing[key][stat] is not None else None
        rows.append(row)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    dump_json(output / "summary.json", {"status": status, "metric_policy": POLICY, "cases": rows})
    judgments = "pulse texture tension_resolution release audible_dropouts notes".split()
    with (output / "listening.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "sample", "seed", "status", *judgments])
        writer.writeheader()
        writer.writerows({**{k: row[k] for k in ("case", "sample", "seed", "status")},
                          **dict.fromkeys(judgments, "未判定")} for row in rows)
    cards = []
    for row in rows:
        base = html.escape(row["case"], quote=True)
        controls = (f'<audio controls preload="none" src="{base}/audio.wav"></audio>'
                    if row["status"] == "ok" else "")
        cards.append(f'<article><h2>{html.escape(row["sample"])} / seed {row["seed"]}</h2>'
                     f'<p>{html.escape(row["status"])} {html.escape(row.get("error", ""))}</p>{controls}'
                     f'<p><a href="{base}/metrics.json">指標</a> · '
                     f'<a href="{base}/report.jsonl">JSONL</a> · <a href="{base}/run.log">実行ログ</a></p></article>')
    (output / "index.html").write_text(
        '<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>Conchordal beta 比較</title><style>body{max-width:900px;margin:2rem auto;padding:0 1rem;'
        'font:16px/1.6 sans-serif}article{border-top:1px solid #bbb;padding:1rem 0}audio{width:100%}</style>'
        '<h1>Conchordal beta 比較</h1><p>録音と指標は同一のオフライン実行から取得。実機RTの合格判定ではありません。</p>'
        '<p><a href="summary.csv">集計CSV</a> · <a href="listening.csv">試聴票（未判定）</a> · '
        '<a href="README.md">測定条件</a></p>' + "".join(cards) + '</html>\n', encoding="utf-8")
    (output / "README.md").write_text(
        "# Conchordal beta 比較\n\n"
        f"キャンペーン状態: `{status}`。各条件の `ok` は実行と成果物検証の成功を示し、音楽的な合格判定ではありません。\n\n"
        "`index.html` で試聴し、`listening.csv` の未判定欄に記入してください。`summary.csv/json` は条件別集計、"
        "各 `metrics.json` は population 別 rhythm_summary、ListenerTwin 全時系列・統計、hop 時間分布を保持します。\n\n"
        "hop の p95/p99 は各条件の `warmup_sec` より前を除外し、(n−1)×p の線形補間で算出します。"
        "除外後の標本がない場合は null です。オフライン測定なので機器underrunは null、実機のRT性能を示しません。\n\n"
        "PCM16 の正規化分母は32768。clip率はrendererの量子化倍率32767に合わせた絶対値32767以上、無音率は絶対値1以下の標本の割合です。"
        "プリリミッターのclippingや無音区間の長さを示す値ではありません。\n\n"
        "rhythm_summary は全報告onsetを含み、habitatのみのVoiceも対象です。IOIはhop時刻に量子化され、"
        "global値は同時発音の影響を受けます。分母は最初〜最後のonsetです。population別の値とWAVを併読してください。"
        "listenerのlate meanは全尺の最後25%なのでrelease尾部を含みます。12の窓はscript操作時刻であり、知覚的な相とは未確認です。\n\n"
        "`manifest.json` にHEAD、source/config/binaryのSHA256、版、実行コマンドを保存しています。"
        "`source/` と差分patchは src/Cargo/samples/scripts/tests/AGENTS の許可対象のみ。"
        "`.env`、`.claude`、`.codex` 等は保存しません。明示指定した外部Rhaiは単体で保存します。"
        "`--skip-build` 使用時は保存binaryとsourceの対応を未検証として記録します。\n",
        encoding="utf-8")


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=DEFAULT_SAMPLES, help="sample stems or .rhai paths")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 21, 42])
    parser.add_argument("--config", type=Path, default=root / "config.toml")
    parser.add_argument("--output", type=Path, help="new campaign directory; existing paths are rejected")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--binary", type=Path, help="existing release renderer path; requires --skip-build")
    parser.add_argument("--timeout", type=float, default=300, help="timeout seconds per render")
    parser.add_argument("--warmup-sec", type=float, default=2, help="exclude initial simulated seconds from timing percentiles")
    parser.add_argument("--reserve-runtime-ids-through", type=int, default=0,
                        help="reserve renderer Voice IDs for matched intervention comparisons")
    args = parser.parse_args(argv)
    if args.binary and not args.skip_build:
        parser.error("--binary requires --skip-build")
    if (not math.isfinite(args.timeout) or args.timeout <= 0 or not math.isfinite(args.warmup_sec)
            or args.warmup_sec < 0 or len(set(args.seeds)) != len(args.seeds)
            or any(seed < 0 or seed >= 2**64 for seed in args.seeds)
            or not 0 <= args.reserve_runtime_ids_through < 2**64 - 1):
        parser.error("require positive finite timeout, nonnegative finite warmup, distinct u64 seeds, and a reservation below u64::MAX")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output or root / "target" / "beta-evaluation" / stamp).resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.mkdir(parents=True)
    cases, manifest = [], {"created_utc": stamp, "status": "running", "metric_policy": POLICY, "commands": []}
    try:
        manifest.update(snapshot(root, output))
        config = args.config.resolve(strict=True)
        shutil.copy2(config, output / "config.toml")
        manifest["config"] = {"original_path": str(config), "sha256": sha256(output / "config.toml")}
        manifest["platform"] = {"platform": platform.platform(), "machine": platform.machine(),
                                "python": sys.version, "rustc": subprocess.check_output(
                                    ["rustc", "--version", "--verbose"], timeout=30, text=True).strip(),
                                "cargo": subprocess.check_output(["cargo", "--version"], timeout=30, text=True).strip()}
        selected = []
        for index, value in enumerate(args.samples):
            source = Path(value)
            if not source.is_file():
                source = root / "samples" / (value if value.endswith(".rhai") else value + ".rhai")
            source = source.resolve(strict=True)
            if source.suffix != ".rhai" or any(p.startswith(".") for p in source.parts):
                raise ValueError(f"scenario must be an explicit non-hidden .rhai file: {source}")
            relative = source.relative_to(root) if source.is_relative_to(root) else Path(
                "samples", "external", f"{index:02d}_{source.name}")
            if not source_allowed(relative):
                raise ValueError(f"scenario path is outside snapshot allowlist: {relative}")
            dest = output / "source" / relative
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
            selected.append({"original_path": str(source), "snapshot_path": str(relative),
                             "sha256": sha256(dest), "sample": source.stem,
                             "slug": f"{index:02d}_" + re.sub(r"[^a-zA-Z0-9_-]", "_", source.stem)})
        manifest["samples"], manifest["seeds"] = selected, args.seeds
        manifest["build_environment"] = {key: os.environ.get(key) for key in (
            "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET", "CARGO_TARGET_DIR")}
        manifest["warmup_sec"] = args.warmup_sec
        manifest["reserve_runtime_ids_through"] = args.reserve_runtime_ids_through
        manifest["build_mode"] = "skip_build_source_match_unverified" if args.skip_build else "cargo_release_locked"
        dump_json(output / "manifest.json", manifest)
        if not args.skip_build:
            build = execute(["cargo", "build", "--release", "--locked", "--bin", "conchordal-render",
                             "--message-format=json-render-diagnostics"],
                            root, output / "build.log", max(900, args.timeout))
            manifest["commands"].append(build)
            if build["status"] != "ok":
                raise ValueError(f"release build {build['status']}; see build.log")
            artifacts = []
            for line in (output / "build.log").read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if (isinstance(record, dict) and record.get("reason") == "compiler-artifact"
                        and record.get("target", {}).get("name") == "conchordal-render"
                        and record.get("executable")):
                    artifacts.append(Path(record["executable"]))
            if len(artifacts) != 1:
                raise ValueError("build must identify exactly one renderer executable")
            binary = artifacts[0]
        for name, entry in manifest["source_files"].items():
            if not (root / name).exists() or sha256(root / name) != entry["sha256"]:
                raise ValueError(f"source changed after snapshot: {name}")
        if args.skip_build:
            target = Path(os.environ.get("CARGO_TARGET_DIR", root / "target"))
            binary = args.binary or (target if target.is_absolute() else root / target) / "release" / (
                "conchordal-render.exe" if os.name == "nt" else "conchordal-render")
        binary = binary.resolve(strict=True)
        saved_binary = output / "bin" / binary.name
        saved_binary.parent.mkdir()
        shutil.copy2(binary, saved_binary)
        manifest["binary"] = {"original_path": str(binary), "sha256": sha256(saved_binary)}
        for sample in selected:
            for seed in args.seeds:
                case_path = Path(sample["slug"]) / f"seed-{seed}"
                directory = output / case_path
                directory.mkdir(parents=True)
                case = {"case": case_path.as_posix(), "sample": sample["sample"], "seed": seed, "status": "error"}
                print(f"render {case['sample']} seed={seed}", flush=True)
                command = [saved_binary, output / "source" / sample["snapshot_path"], "--config", output / "config.toml",
                           "--seed", str(seed), "--report", directory / "report.jsonl", "-o", directory / "audio.wav"]
                if args.reserve_runtime_ids_through:
                    command.extend(["--reserve-runtime-ids-through", str(args.reserve_runtime_ids_through)])
                try:
                    run = execute(command, output / "source", directory / "run.log", args.timeout)
                    manifest["commands"].append(run)
                    case["execution"] = run
                    if run["status"] != "ok":
                        raise ValueError(f"renderer {run['status']} (exit={run['returncode']}); see run.log")
                    case["audio"] = wav_metrics(directory / "audio.wav")
                    case["report"] = parse_report(directory / "report.jsonl", seed, args.warmup_sec)
                    timing = case["report"]["hop_timing"]
                    hop_frames = round(timing["hop_budget_us"] * case["audio"]["sample_rate"] / 1e6)
                    if hop_frames <= 0 or case["audio"]["frames"] != timing["total_count"] * hop_frames:
                        raise ValueError("WAV frame count does not match reported hop timings")
                    if sample["sample"] == "12_emergence_and_resolution":
                        windows = []
                        for label, lo, hi in [("baseline_colony", 6, 15.4), ("tension_before_flow", 15.4, 18.7),
                                               ("tension_with_flow", 18.7, 24), ("resolution_with_flow", 24, 27.3),
                                               ("resolution_after_flow", 28.6, 33.9)]:
                            series = [r for r in case["report"]["listener"]["series"] if lo <= r["time_sec"] < hi]
                            windows.append({"label": label, "start_sec": lo, "end_sec": hi, "count": len(series),
                                            "stats": {key: stats([r[key] for r in series]) for key in
                                                      ("tension_level", "beat_confidence", "analysis_lag_frames")}})
                        case["report"]["script_operation_windows"] = {
                            "scope": "script operation times, not validated perceptual phases", "windows": windows}
                    if sample["sample"] in DEFAULT_SAMPLES and case["audio"]["all_silent"]:
                        raise ValueError("standard comparison sample rendered entirely silent")
                    case["artifact_sha256"] = {name: sha256(directory / name) for name in ("audio.wav", "report.jsonl")}
                    case["status"] = "ok"
                except (ValueError, OSError, EOFError, wave.Error) as error:
                    case["error"] = str(error)
                dump_json(directory / "metrics.json", case)
                cases.append(case)
                dump_json(output / "manifest.json", manifest)
                write_tables(output, cases, "running")
                if case.get("execution", {}).get("status") == "interrupted":
                    raise ValueError("campaign interrupted")
        manifest["status"] = "complete" if all(case["status"] == "ok" for case in cases) else "failed"
    except (ValueError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        manifest.update(status="failed", error=str(error))
        print(f"evaluation failed: {error}", file=sys.stderr)
    dump_json(output / "manifest.json", manifest)
    write_tables(output, cases, manifest["status"])
    print(output, flush=True)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
