#!/usr/bin/env python3
"""Compare the three scripted resolution interventions in sample 12."""

import argparse
import array
import collections
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import wave

import evaluate_beta as beta


FACTORS = {
    "temperature": ("    colony.temperature(0.85);\n", "    colony.temperature(0.0);\n"),
    "pitch_shift": ("    root.freq(root_hz * 1.5);\n", "    pulse.freq(root_hz * 1.5);\n",
                    "    root.freq(root_hz);\n", "    pulse.freq(root_hz);\n"),
    "flow": ("    let flow = place(flow_particles, consonance(200.0, 1500.0).count(9).spacing(0.66));\n",
             "    flow.amp(0.014);\n", "    release(flow);\n"),
}
COMMON_PLACEMENTS = (
    "    let root = place(field_anchor, at(root_hz).count(1));\n",
    "    let pulse = place(metric_body, at(root_hz).count(1));\n",
    "    let colony = place(consonance_colony, consonance(80.0, 900.0).count(8).spacing(0.84));\n",
)
RESERVE_RUNTIME_IDS_THROUGH = 19
FLOW_START_SEC = 18.7
STATE_RECORDS = {"spawn", "respawn", "death", "onset", "population_step", "rhythm_observation",
                 "listener_state", "dcc_pressure", "habituation", "phonation_gate_open"}
WAIT_SEQUENCE = ["2.3", "3.7", "9.4", "3.3", "5.3", "3.3", "1.3", "5.3", "2.0", "1.3", "4.0"]
OPERATION_WAIT_COUNTS = [
    *zip(COMMON_PLACEMENTS, (0, 1, 2)),
    *zip(FACTORS["temperature"], (3, 5)),
    *zip(FACTORS["pitch_shift"], (3, 3, 5, 5)),
    *zip(FACTORS["flow"], (4, 5, 6)),
    ("    colony.amp(0.034);\n", 3),
    ('    colony.pitch_apply_mode("glide");\n', 5),
    ("    colony.glide(0.22);\n", 5),
    ("    colony.amp(0.028);\n", 7),
    ("    release(colony);\n", 8),
    ("    release(pulse);\n", 9),
    ("    release(root);\n", 10),
]
WINDOWS = {"baseline": (6.0, 15.4), "tension": (18.7, 24.0),
           "early_resolution": (24.0, 27.3), "late_resolution": (28.6, 33.9)}
LISTENER_FIELDS = ["tension_level", "stability_level", "resolvability_level", "attention_level", "beat_confidence"]
MEASURES = [f"listener_{key}_mean" for key in LISTENER_FIELDS] + [
    "audio_rms", "audio_peak", "audio_silence_fraction", "colony_alive_count_mean",
    "colony_mean_freq_hz", "colony_mean_c_field_level"]


def variant_source(source, factors):
    # Validate every intervention even when enabled, so the control cannot hide drift.
    for fragments in FACTORS.values():
        for fragment in fragments:
            if source.count(fragment) != 1:
                raise ValueError(f"sample 12 drift: expected exactly one {fragment.strip()!r}")
    # The ID reservation is tied to these four placements: 1 + 1 + 8 + 9.
    if (any(source.count(fragment) != 1 for fragment in COMMON_PLACEMENTS)
            or len(re.findall(r"\bplace\s*\(", source)) != 4):
        raise ValueError("sample 12 placements changed; review the runtime ID reservation")
    waits = re.findall(r"(?m)^\s*wait\(([^)]+)\);$", source)
    if waits != WAIT_SEQUENCE:
        raise ValueError("sample 12 wait sequence changed; review the measurement windows")
    for fragment, count in OPERATION_WAIT_COUNTS:
        if (source.count(fragment) != 1 or re.findall(r"(?m)^\s*wait\(([^)]+)\);$",
                                                     source[:source.index(fragment)]) != waits[:count]):
            raise ValueError(f"sample 12 operation timing changed: {fragment.strip()}")
    for name, fragments in FACTORS.items():
        if not factors[name]:
            for fragment in fragments:
                source = source.replace(fragment, "", 1)
    return source


def mean(values):
    return math.fsum(values) / len(values) if values else None


def window_metrics(listener, population, wav_path, label):
    lo, hi = WINDOWS[label]
    listener = [r for r in listener if lo <= r["time_sec"] < hi]
    population = [r for r in population if r["population_id"] == 3 and lo <= r["time_sec"] < hi]
    if not listener or not population:
        raise ValueError(f"{label}: missing listener or colony observations")
    alive = [r for r in population if r["alive_count"] > 0]
    row = {"window": label, "start_sec": lo, "end_sec": hi, "listener_count": len(listener),
           "colony_sample_count": len(population), "colony_alive_sample_count": len(alive),
           "colony_alive_count_mean": mean([r["alive_count"] for r in population]),
           "colony_mean_freq_hz": mean([r["mean_freq_hz"] for r in alive]),
           "colony_mean_c_field_level": mean([r["mean_c_field_level"] for r in alive]),
           "listener_analysis_lag_frames_mean": mean([r["analysis_lag_frames"] for r in listener])}
    row.update({f"listener_{key}_mean": mean([r[key] for r in listener]) for key in LISTENER_FIELDS})
    with wave.open(str(wav_path), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) != (1, 2, "NONE"):
            raise ValueError("window audio must be mono PCM16")
        start, end = math.ceil(lo * stream.getframerate()), math.ceil(hi * stream.getframerate())
        if start >= end or end > stream.getnframes():
            raise ValueError(f"{label}: audio does not cover the complete window")
        stream.setpos(start)
        samples = array.array("h", stream.readframes(end - start))
        if len(samples) != end - start:
            raise ValueError(f"{label}: truncated audio window")
        if sys.byteorder != "little":
            samples.byteswap()
    row.update(audio_frames=len(samples), audio_rms=math.sqrt(sum(v * v for v in samples) / len(samples)) / 32768,
               audio_peak=max(abs(v) for v in samples) / 32768,
               audio_silence_fraction=sum(abs(v) <= 1 for v in samples) / len(samples))
    return row


def paired_effects(rows):
    effects = []
    cases = {(r["variant"], r["seed"]) for r in rows}
    for variant, seed in sorted(cases):
        pair = {r["window"]: r for r in rows if r["variant"] == variant and r["seed"] == seed}
        before, after = pair["tension"], pair["early_resolution"]
        effect = {key: before[key] for key in ("variant", "seed", *FACTORS)}
        for key in MEASURES:
            effect[f"early_minus_tension_{key}"] = (after[key] - before[key]
                                                     if before[key] is not None and after[key] is not None else None)
        effects.append(effect)
    controls = {r["seed"]: r for r in effects if r["variant"] == "111"}
    for effect in effects:
        control = controls.get(effect["seed"])
        for key in MEASURES:
            value = effect[f"early_minus_tension_{key}"]
            reference = control[f"early_minus_tension_{key}"] if control else None
            effect[f"delta_minus_control_111_{key}"] = (value - reference
                                                        if value is not None and reference is not None else None)
    return effects


def save_table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["variant", "seed", "status"])
        writer.writeheader()
        writer.writerows(rows)


def flow_pre_intervention_checks(output, variants, seeds):
    cases = {(r["sample"], r["seed"]): r for r in
             json.loads((output / "summary.json").read_text(encoding="utf-8"))["cases"]}
    prefixes, failures = {}, []
    for label, spec in variants.items():
        for seed in seeds:
            case = cases.get((spec["sample"], seed))
            if not case or case["status"] != "ok":
                continue
            directory = output / case["case"]
            with wave.open(str(directory / "audio.wav"), "rb") as stream:
                frames = math.ceil(FLOW_START_SEC * stream.getframerate())
                audio = stream.readframes(frames)
                if len(audio) != frames * stream.getnchannels() * stream.getsampwidth():
                    raise ValueError("audio does not cover the pre-flow interval")
                audio_format = [stream.getframerate(), stream.getnchannels(), stream.getsampwidth()]
            digest, counts, respawns = hashlib.sha256(), collections.Counter(), []
            with (directory / "report.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["type"] in STATE_RECORDS and row["time_sec"] < FLOW_START_SEC:
                        digest.update((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
                        counts[row["type"]] += 1
                        if row["type"] == "respawn":
                            respawns.append(row)
            if not counts["listener_state"] or not counts["population_step"] or not counts["spawn"]:
                raise ValueError("missing pre-flow state observations")
            prefixes[(label, seed)] = {"pcm_sha256": hashlib.sha256(audio).hexdigest(),
                                       "state_sha256": digest.hexdigest(), "audio_format": audio_format,
                                       "audio_frames": frames, "record_counts": dict(counts), "respawns": respawns}
    checks = []
    for off in sorted(label for label in variants if label.endswith("0")):
        on = off[:2] + "1"
        for seed in seeds:
            a, b = prefixes.get((off, seed)), prefixes.get((on, seed))
            matched = a is not None and b is not None and a == b
            checks.append({"off": off, "on": on, "seed": seed, "matched": matched,
                           "off_prefix": a, "on_prefix": b})
            if not matched:
                failures.append({"off": off, "on": on, "seed": seed,
                                 "error": "flow comparison differs or lacks observations before intervention"})
    beta.dump_json(output / "flow_pre_intervention.json", {
        "interval": [0.0, FLOW_START_SEC], "interval_end_exclusive": True,
        "reserve_runtime_ids_through": RESERVE_RUNTIME_IDS_THROUGH,
        "state_scope": sorted(STATE_RECORDS), "excludes": "wall-clock hop timing and final summaries",
        "checks": checks, "failures": failures})
    return failures


def aggregate(output, variants, seeds):
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    cases = {(r["sample"], r["seed"]): r for r in summary["cases"]}
    rows, failures = [], []
    for variant, spec in variants.items():
        for seed in seeds:
            case = cases.get((spec["sample"], seed))
            if not case or case["status"] != "ok":
                failures.append({"variant": variant, "seed": seed, "error": case.get("error", "render failed")
                                 if case else "render not completed"})
                continue
            directory = output / case["case"]
            try:
                metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
                for filename in ("report.jsonl", "audio.wav"):
                    if beta.sha256(directory / filename) != metrics["artifact_sha256"][filename]:
                        raise ValueError(f"artifact changed after beta validation: {filename}")
                # The same bytes were strictly validated by evaluate_beta; retain its listener series.
                with (directory / "report.jsonl").open(encoding="utf-8") as stream:
                    population = [r for line in stream if (r := json.loads(line))["type"] == "population_step"]
                window_rows = [{"variant": variant, "seed": seed, **spec["factors"],
                                **window_metrics(metrics["report"]["listener"]["series"], population,
                                                 directory / "audio.wav", label)} for label in WINDOWS]
                rows.extend(window_rows)
            except (ValueError, KeyError, OSError, EOFError, wave.Error) as error:
                failures.append({"variant": variant, "seed": seed, "error": str(error)})
    failures.extend(flow_pre_intervention_checks(output, variants, seeds))
    effects = paired_effects(rows)
    save_table(output / "resolution_windows.csv", rows)
    save_table(output / "resolution_effects.csv", effects)
    beta.dump_json(output / "resolution_windows.json", {"windows": rows, "failures": failures})
    beta.dump_json(output / "resolution_effects.json", {
        "contrast": "early_resolution minus tension, then subtract matched-seed control 111 contrast",
        "effects": effects, "failures": failures})
    (output / "resolution-results.md").write_text(
        "# Sample 12 操作分解比較\n\n"
        f"集計成功: {len(effects)}/{len(variants) * len(seeds)}条件。試聴は未実施です。\n\n"
        "variantは温度・音高移動・flowの順にON=1/OFF=0を並べた値です。111が原本の対照条件です。"
        "OFFでは対象操作だけを削除し、wait、colonyのamp/glide、その他の操作を維持しています。"
        "`manifest.json` に要因値、生成元と各scriptのSHA256を保存しています。"
        "全variantでVoice ID 1〜19を予約し、flow登場前[0,18.7)秒のPCMと決定的な状態記録の一致を"
        "`flow_pre_intervention.json` で検査します。不一致や欠落はcampaign失敗です。\n\n"
        "`resolution_windows.csv/json` は4つの共通操作窓の平均、PCM音声、population 3の状態を保持します。"
        "mean_freqとmean_c_field_levelはalive_count>0の標本のみ、alive_count平均は0も含みます。"
        "窓は脚本操作時刻であり、知覚的な相とは未確認です。ListenerTwinは解析遅延を含みます。\n\n"
        "`resolution_effects.csv/json` は各seedでearly_resolution−tensionを計算し、さらに同seedの111との差を記録します。"
        "seedを跨いだ対応付け、p値の計算、未試聴の音楽的評価は行っていません。"
        "比較対象はこの実装に対する脚本操作の効果です。内部指標の変化を人の知覚への因果効果と同一視しません。"
        "介入前のID予約を揃えても、flow登場後は集団構成・配置抽選・相互作用が変わります。"
        "これはflow集団を加える効果の比較であり、flow音だけの効果や介入後も完全な共通乱数の対照ではありません。"
        "全ONと既存baseline-r2のWAV一致は別途確認が必要です。\n\n"
        "[試聴ページ](index.html) / [共通測定条件](README.md) / [窓CSV](resolution_windows.csv) / "
        "[対応差CSV](resolution_effects.csv) / [介入前の一致](flow_pre_intervention.json)\n", encoding="utf-8")
    return failures


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config.toml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 21, 42])
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output or root / "target" / "resolution-evaluation" / stamp).resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    source_path = root / "samples" / "12_emergence_and_resolution.rhai"
    source = source_path.read_text(encoding="utf-8")
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    variants = {}
    try:
        with tempfile.TemporaryDirectory(prefix="conchordal-resolution-", dir="/tmp") as temporary:
            paths = []
            for values in itertools.product((0, 1), repeat=3):
                label = "".join(map(str, values))
                factors = dict(zip(FACTORS, values))
                path = Path(temporary) / f"resolution_{label}.rhai"
                path.write_text(variant_source(source, factors), encoding="utf-8")
                variants[label] = {"sample": path.stem, "factors": factors, "sha256": beta.sha256(path)}
                paths.append(str(path))
            command = ["--samples", *paths, "--seeds", *map(str, args.seeds), "--config", str(args.config),
                       "--output", str(output), "--timeout", str(args.timeout),
                       "--reserve-runtime-ids-through", str(RESERVE_RUNTIME_IDS_THROUGH)]
            if args.skip_build:
                command.append("--skip-build")
            if args.binary:
                command.extend(["--binary", str(args.binary)])
            result = beta.main(command)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        manifest["resolution_experiment"] = {
            "source_sample": str(source_path), "source_sha256": source_sha256,
            "factor_order": list(FACTORS), "variants": variants, "control": "111",
            "reserve_runtime_ids_through": RESERVE_RUNTIME_IDS_THROUGH,
            "windows": WINDOWS, "baseline_r2_wav_match": "not_checked",
            "scope": "script interventions with identical pre-flow PCM/state required; post-flow trajectories may diverge; perceptual effects unreviewed"}
        beta.dump_json(output / "manifest.json", manifest)
        failures = aggregate(output, variants, args.seeds)
        manifest["resolution_experiment"]["status"] = "complete" if result == 0 and not failures else "failed"
        if failures:
            manifest["status"] = "failed"
        beta.dump_json(output / "manifest.json", manifest)
        print(f"resolution comparison: {output}", flush=True)
        return 0 if result == 0 and not failures else 1
    except (ValueError, KeyError, OSError, EOFError, wave.Error) as error:
        print(f"resolution evaluation failed: {error}", file=sys.stderr)
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                manifest = {}
            manifest["status"] = "failed"
            manifest.setdefault("resolution_experiment", {}).update(status="failed", error=str(error))
            try:
                beta.dump_json(manifest_path, manifest)
            except OSError as write_error:
                print(f"could not record resolution failure: {write_error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
