#!/usr/bin/env python3
"""Archive acoustic-only meter observations before changing temporal behavior."""

import argparse
import array
import csv
import datetime as dt
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import wave


FS = 48_000
DURATION = 40
HOP = 512
CASES = ("pulse", "alternating", "syncopated", "shuffled_syncopated", "renewal",
         "gaps", "tempo_drift", "pulse_slow", "pulse_fast", "silence",
         "ternary_marked", "jittered", "uneven_cycle", "shuffled_uneven_cycle")
BASELINE_CASES = CASES[:10]
WINDOWS = ((2, 10), (12, 14), (15, 21), (24, 30), (32, 39))
PRIORS = ("unshaped", "sample08_prior")
ASSAY = "core::meter::audio_assay::export_audio_perception_assay"
SOURCES = ("Cargo.lock", "src/core/meter.rs", "src/core/meter_assay.rs", "src/core/onset.rs",
           "src/core/stream/dorsal.rs", "src/core/phase.rs",
           "scripts/evaluate_rhythm_perception.py")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def events_for(case, seed):
    if case not in CASES:
        raise ValueError(f"unknown case: {case}")
    rng = random.Random(seed)
    start = round((.8 + rng.uniform(0, .35)) * FS)
    stop = 39 * FS
    if case == "silence":
        return []
    if case.startswith("shuffled_"):
        events = events_for(case.removeprefix("shuffled_"), seed)
        gaps = [b["frame"] - a["frame"] for a, b in zip(events, events[1:])]
        rng.shuffle(gaps)
        cursor = events[0]["frame"]
        for event, gap in zip(events[1:], gaps):
            cursor += gap
            event["frame"] = cursor
        return events
    if case == "jittered":
        events = events_for("pulse", seed)
        for event in events:
            event["frame"] += round(rng.uniform(-.025, .025) * FS)
        return events
    if case == "renewal":
        events = events_for("pulse", seed)
        weights = [.06 + rng.expovariate(2) for _ in events[1:]]
        span = events[-1]["frame"] - start
        total = math.fsum(weights)
        for index in range(1, len(events)):
            events[index]["frame"] = start + round(span * math.fsum(weights[:index]) / total)
        return events
    frames = []
    cursor = start / FS
    while round(cursor * FS) < stop:
        frame = round(cursor * FS)
        if case != "gaps" or not (12 <= cursor < 14 or 22 <= cursor < 31):
            frames.append(frame)
        if case == "alternating":
            rate = 4
        elif case == "ternary_marked":
            rate = 1 / .21
        elif case == "uneven_cycle":
            cursor += (.18, .35, .38, .46)[(len(frames) - 1) % 4]
            continue
        elif case == "syncopated":
            cursor += .75 if len(frames) % 2 else .25
            continue
        elif case == "tempo_drift":
            rate = 2 + .6 * min(1, max(0, (cursor - 14) / 12))
        else:
            rate = {"pulse_slow": 1.4, "pulse_fast": 2.6}.get(case, 2)
        cursor += 1 / rate
    return [{"frame": frame, "timbre": (i % 2 if case == "alternating" else
                                        i % 3 if case == "ternary_marked" else
                                        (0, 1, 2, 1)[i % 4] if case == "uneven_cycle" else 0),
             "amplitude": .4, "duration_sec": .09, "decay_sec": .018}
            for i, frame in enumerate(frames)]


def write_audio(path, events):
    audio = array.array("f", [0]) * (DURATION * FS)
    for event in events:
        freq = (180.0, 1800.0, 6000.0)[event["timbre"]]
        for j in range(round(event["duration_sec"] * FS)):
            index = event["frame"] + j
            if index >= len(audio):
                raise ValueError("truncated stimulus")
            t = j / FS
            envelope = min(1, t / .002) * math.exp(-t / event["decay_sec"])
            audio[index] += event["amplitude"] * envelope * math.sin(math.tau * freq * t)
    peak = max(map(abs, audio))
    if peak >= 1:
        raise ValueError("stimulus clips")
    pcm = array.array("h", (round(x * 32767) for x in audio))
    rms = math.sqrt(math.fsum((x / 32768) ** 2 for x in pcm) / len(pcm))
    if sys.byteorder != "little":
        pcm.byteswap()
    with wave.open(str(path), "wb") as out:
        out.setparams((1, 2, FS, 0, "NONE", "not compressed"))
        out.writeframes(pcm.tobytes())
    return {"peak": peak, "rms": rms, "frames": len(audio)}


def read_observations(path):
    with path.open(newline="") as stream:
        rows = [{key: None if value == "" else float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]
    if len(rows) != math.ceil(DURATION * FS / HOP):
        raise ValueError(f"incomplete observation: {path}")
    previous = 0
    for row in rows:
        if any(v is not None and not math.isfinite(v) for v in row.values()):
            raise ValueError(f"nonfinite observation: {path}")
        now = row["time_sec"]
        if not previous < now <= DURATION or now - previous > HOP / FS + 1e-8:
            raise ValueError(f"invalid time coverage: {path}")
        for key in ("beat_confidence", "subdivision_confidence", "measure_confidence"):
            if not 0 <= row[key] <= 1:
                raise ValueError(f"invalid {key}: {path}")
        onset = row["onset_time_sec"]
        if (onset is None) != (row["onset_phase"] is None):
            raise ValueError("onset phase without event or event without phase")
        if onset is not None and not previous - 1e-8 <= onset <= now + 1e-8:
            raise ValueError("onset outside its observed chunk")
        previous = now
    if abs(previous - DURATION) > 1e-8:
        raise ValueError("missing final observation")
    return rows


def window_stats(rows, start, end):
    subset = [r for r in rows if start <= r["time_sec"] < end]
    if not subset:
        raise ValueError("empty observation window")
    phases = [r["onset_phase"] for r in rows
              if r["onset_time_sec"] is not None and start <= r["onset_time_sec"] < end]
    result = {"start_sec": start, "end_sec": end, "detected_onsets": len(phases)}
    for name in ("beat_hz", "beat_confidence", "subdivision_confidence", "measure_confidence"):
        result["mean_" + name] = math.fsum(r[name] for r in subset) / len(subset)
    result["beat_hz_min"] = min(r["beat_hz"] for r in subset)
    result["beat_hz_max"] = max(r["beat_hz"] for r in subset)
    for harmonic in (1, 2):
        result[f"onset_r{harmonic}"] = (math.hypot(
            math.fsum(math.cos(harmonic * p) for p in phases),
            math.fsum(math.sin(harmonic * p) for p in phases)) / len(phases)) if phases else None
    return result


def summarize(output):
    manifest = json.loads((output / "manifest.json").read_text())
    cases = []
    for case in manifest["cases"]:
        folder = output / case["id"]
        if sha256(folder / "audio.wav") != case["audio_sha256"]:
            raise ValueError("input audio changed after manifest creation")
        events = json.loads((folder / "events.json").read_text())
        if sha256(folder / "events.json") != case["events_sha256"]:
            raise ValueError("ground-truth events changed")
        ground_truth = [e["frame"] / FS for e in events]
        result = {"id": case["id"], "case": case["case"], "seed": case["seed"], "priors": {}}
        for prior in PRIORS:
            rows = read_observations(folder / f"{prior}.csv")
            detected = [r["onset_time_sec"] for r in rows if r["onset_time_sec"] is not None]
            # Match one-to-one within 60 ms; event labels never enter observation.
            available = set(range(len(detected)))
            matches = []
            for onset in ground_truth:
                nearby = [i for i in available if abs(detected[i] - onset) <= .06]
                if nearby:
                    index = min(nearby, key=lambda i: abs(detected[i] - onset))
                    available.remove(index)
                    matches.append(detected[index] - onset)
            result["priors"][prior] = {
                "detected_onsets": len(detected), "source_onsets": len(events),
                "matched_onsets": len(matches), "extra_detections": len(available),
                "mean_detection_offset_sec": math.fsum(matches) / len(matches) if matches else None,
                "windows": [window_stats(rows, *w) for w in WINDOWS],
                "observation_sha256": sha256(folder / f"{prior}.csv"),
            }
        cases.append(result)
    summary = {"status": "observed", "musical_acceptance": "not_assessed",
               "prediction_performance": "not_implemented_in_baseline", "cases": cases}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = []
    for case in cases:
        for prior, record in case["priors"].items():
            late = record["windows"][-1]
            r1 = late["onset_r1"]
            table.append(f'<tr><td>{html.escape(case["id"])}</td><td>{prior}</td>'
                         f'<td>{late["mean_beat_hz"]:.3f}</td>'
                         f'<td>{late["mean_beat_confidence"]:.3f}</td>'
                         f'<td>{"—" if r1 is None else f"{r1:.3f}"}</td>'
                         f'<td>{record["matched_onsets"]}/{record["source_onsets"]}</td>'
                         f'<td><a href="{case["id"]}/{prior}.csv">CSV</a></td></tr>')
    audio = "\n".join(f'<p>{html.escape(c["id"])}<br><audio controls preload="none" '
                       f'src="{c["id"]}/audio.wav"></audio></p>' for c in cases)
    page = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Acoustic rhythm perception assay</title>'
            '<style>body{max-width:1100px;margin:2em auto;padding:1em;font:16px system-ui}'
            'td,th{padding:.4em;text-align:left;border-bottom:1px solid #ddd}audio{width:90%}</style>'
            '<h1>Acoustic rhythm perception assay</h1>'
            '<p>Baseline observations; no production feedback. Statistics below cover 32–39 s. '
            'PLV is onset concentration, not a musical score or predictive accuracy. '
            'The sample08 prior constrains tempo to 1.6–2.0 Hz; the same prior is used for every input.</p>'
            '<table><tr><th>Input</th><th>Prior</th><th>Hz</th><th>Confidence</th><th>R1</th>'
            '<th>Matched onsets</th><th>Trace</th></tr>' + "\n".join(table) + '</table>'
            '<h2>Stimuli</h2>' + audio + '</html>')
    (output / "audition.html").write_text(page)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 21, 42])
    parser.add_argument("--cases", choices=CASES, nargs="+", default=BASELINE_CASES)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.analyze_only:
        summarize(output)
        return
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    if len(set(args.cases)) != len(args.cases):
        parser.error("cases must be distinct")
    root = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "source"
    for name in SOURCES:
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    manifest = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "sample_rate": FS, "hop_size": HOP, "duration_sec": DURATION,
                "source_sha256": {name: sha256(root / name) for name in SOURCES},
                "observer_input": "mono PCM16 only; no event times, roles, seed or source labels",
                "priors": {"unshaped": {"stability": 0, "basin_hz": None},
                           "sample08_prior": {"stability": .45, "basin_hz": [1.6, 2]}},
                "cases": []}
    for seed in args.seeds:
        for name in args.cases:
            case_id = f"{name}_seed{seed}"
            folder = output / case_id
            folder.mkdir()
            events = events_for(name, seed)
            (folder / "events.json").write_text(json.dumps(events, indent=2) + "\n")
            metrics = write_audio(folder / "audio.wav", events)
            manifest["cases"].append({"id": case_id, "case": name, "seed": seed,
                                      "audio": metrics,
                                      "audio_sha256": sha256(folder / "audio.wav"),
                                      "events_sha256": sha256(folder / "events.json")})
    build = subprocess.run(["cargo", "test", "--lib", "--no-run", "--message-format=json"],
                           cwd=root, text=True, capture_output=True)
    (output / "build.jsonl").write_text(build.stdout)
    (output / "build.stderr.txt").write_text(build.stderr)
    build.check_returncode()
    binaries = [r["executable"] for line in build.stdout.splitlines()
                if (r := json.loads(line)).get("reason") == "compiler-artifact"
                and r.get("executable") and r["target"]["kind"] == ["lib"]]
    if len(binaries) != 1:
        raise RuntimeError(f"expected one library test binary, found {binaries}")
    manifest["observer_binary_sha256"] = sha256(Path(binaries[0]))
    manifest["observer_binary"] = binaries[0]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = os.environ | {"CONCHORDAL_RHYTHM_ASSAY_DIR": str(output), "RUST_BACKTRACE": "1"}
    with (output / "observer.log").open("w") as log:
        run = subprocess.run([binaries[0], ASSAY, "--exact", "--ignored", "--nocapture"],
                             cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    (output / "observer-status.json").write_text(json.dumps({"exit_code": run.returncode}) + "\n")
    run.check_returncode()
    summarize(output)
    print(f"Observed {len(manifest['cases'])} inputs under {len(PRIORS)} priors: {output}")


if __name__ == "__main__":
    main()
