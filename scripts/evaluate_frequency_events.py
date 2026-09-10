#!/usr/bin/env python3
"""Observe spectral-component power rises and trajectories in mixed audio, offline."""

import argparse
from collections import deque
import datetime as dt
import json
import math
from pathlib import Path
import shutil
import wave

import numpy as np

from evaluate_phrase_expectation import FS, sha256, synthesize, write_wav


STRIDE_SEC = .01
WINDOW_SEC = 2048 / 24_000
MAX_COMPONENTS = 32
MIN_POWER = .005 ** 2
MIN_CONCENTRATION = .80
MATCH_OCT = .12
RISE_RATIO = 1.6
REFRACTORY_SEC = .14
CASES = ("isolated", "overlapping", "simultaneous", "soft_overlay", "harmonic",
         "steady_pair", "vibrato", "glide", "silence")


class FrequencyEventObserver:
    """Bounded local ridges; their identifiers are observations, never Voice IDs."""

    def __init__(self, fs):
        if not isinstance(fs, int) or fs < 16_000:
            raise ValueError("expected an integer sample rate of at least 16000 Hz")
        self.fs = fs
        self.stride = round(STRIDE_SEC * fs)
        self.size = round(WINDOW_SEC * fs)
        self.window = deque(maxlen=self.size)
        self.taper = np.hanning(self.size)
        self.power_scale = 2 / (self.size * float(np.sum(self.taper ** 2)))
        self.next_sample = None
        self.evidence_floor = None
        self.pending = 0
        self.tracks = []
        self.next_id = 0
        self.primed = False

    def process(self, start_sample, audio):
        audio = np.asarray(audio, dtype=float)
        if (not isinstance(start_sample, int) or start_sample < 0 or audio.ndim != 1
                or not np.all(np.isfinite(audio))
                or self.next_sample is not None and start_sample < self.next_sample):
            raise ValueError("expected ordered sample indices and finite mono audio")
        if not len(audio):
            return []
        result = []
        if self.next_sample is None:
            self.evidence_floor = start_sample
        if self.next_sample is not None and start_sample != self.next_sample:
            result.append({"kind": "input_gap", "available_sec": start_sample / self.fs,
                           "missing_start_sec": self.next_sample / self.fs})
            self.window.clear()
            self.tracks.clear()
            self.pending = 0
            self.primed = False
            self.evidence_floor = start_sample
        cursor = 0
        while cursor < len(audio):
            count = min(self.stride - self.pending, len(audio) - cursor)
            self.window.extend(audio[cursor:cursor + count])
            cursor += count
            self.pending += count
            if self.pending != self.stride:
                continue
            self.pending = 0
            if len(self.window) != self.size:
                continue
            end = start_sample + cursor
            x = np.asarray(self.window)
            spectrum = np.abs(np.fft.rfft((x - np.mean(x)) * self.taper)) ** 2
            peak_indices = np.flatnonzero((spectrum[1:-1] > spectrum[:-2])
                                         & (spectrum[1:-1] >= spectrum[2:])) + 1
            candidates = []
            for index in peak_indices:
                if not 55 <= index * self.fs / self.size <= 7040:
                    continue
                local = float(np.sum(spectrum[max(0, index - 1):index + 2]))
                broad = float(np.sum(spectrum[max(0, index - 5):index + 6]))
                concentration = local / max(broad, 1e-30)
                power = float(np.sum(spectrum[max(0, index - 2):index + 3])) * self.power_scale
                if power < MIN_POWER or concentration < MIN_CONCENTRATION:
                    continue
                a, b, c = np.log(np.maximum(spectrum[index - 1:index + 2], 1e-30))
                curvature = a - 2 * b + c
                offset = float(np.clip(.5 * (a - c) / curvature, -.5, .5)) if curvature < 0 else 0
                frequency = (index + offset) * self.fs / self.size
                candidates.append({"frequency_log2": math.log2(frequency), "power": power,
                                   "concentration": concentration})
            candidates.sort(key=lambda c: c["power"], reverse=True)
            truncated = max(0, len(candidates) - MAX_COMPONENTS)
            candidates = candidates[:MAX_COMPONENTS]
            pairs = sorted((abs(c["frequency_log2"] - t["frequency_log2"]), ci, ti)
                           for ci, c in enumerate(candidates) for ti, t in enumerate(self.tracks)
                           if abs(c["frequency_log2"] - t["frequency_log2"]) <= MATCH_OCT)
            matched, used = {}, set()
            for _, ci, ti in pairs:
                if ci not in matched and ti not in used:
                    matched[ci] = ti
                    used.add(ti)
            for ti, track in enumerate(self.tracks):
                if ti not in used:
                    track["powers"].append(0.0)
                    track["consecutive"] = 0
            components, fresh = [], []
            for ci, candidate in enumerate(candidates):
                if ci in matched:
                    track = self.tracks[matched[ci]]
                else:
                    track = {"id": self.next_id, "powers": deque(
                        [0.0 if self.primed else candidate["power"]] * 6, maxlen=6),
                        "consecutive": 0, "last_rise": None}
                    self.next_id += 1
                    fresh.append(track)
                track.update(frequency_log2=candidate["frequency_log2"], last_seen=end)
                track["consecutive"] += 1
                track["powers"].append(candidate["power"])
                history = list(track["powers"])
                now, before = float(np.mean(history[-3:])), float(np.mean(history[:3]))
                separated = track["last_rise"] is None or end - track["last_rise"] >= round(REFRACTORY_SEC * self.fs)
                rise = self.primed and track["consecutive"] >= 3 and now >= MIN_POWER and now > RISE_RATIO * before and separated
                if rise:
                    track["last_rise"] = end
                components.append({"ridge_id": track["id"], "frequency_log2": candidate["frequency_log2"],
                                   "power": candidate["power"], "concentration": candidate["concentration"],
                                   "rise": bool(rise), "recent_power": now, "preceding_power": before})
            self.tracks.extend(fresh)
            self.tracks = sorted((t for t in self.tracks if end - t["last_seen"] <= 6 * self.stride),
                                 key=lambda t: t["last_seen"], reverse=True)[:MAX_COMPONENTS]
            result.append({"kind": "components", "available_sec": end / self.fs,
                           "evidence_start_sec": max(self.evidence_floor, end - self.size - 5 * self.stride) / self.fs,
                           "initial_observation": not self.primed,
                           "truncated_components": truncated, "components": components})
            self.primed = True
        self.next_sample = start_sample + len(audio)
        return result


def control_audio(case, seed):
    if case not in CASES:
        raise ValueError("unknown frequency-event control")
    rng = np.random.default_rng(seed)
    first = float(rng.uniform(260, 360))
    second = first * float(rng.uniform(1.6, 1.9))
    times = [.6 + .35 * i for i in range(24)]
    events = []
    for i, onset in enumerate(times):
        if case in ("steady_pair", "vibrato", "glide", "silence") and i:
            break
        frequencies = [first, second] if case in ("simultaneous", "steady_pair") else [first if i % 2 == 0 else second]
        if case in ("soft_overlay", "vibrato", "glide", "harmonic"):
            frequencies = [second if case == "soft_overlay" else first]
        for frequency in frequencies:
            events.append({"onset_sec": onset, "duration_sec": 9.0 if case in ("steady_pair", "vibrato", "glide") else
                           .55 if case == "overlapping" else .18,
                           "frequency_hz": frequency, "amplitude": .08 if case == "soft_overlay" else .22,
                           "timbre": "harmonic" if case == "harmonic" else "sine"})
    if case == "soft_overlay":
        events.append({"onset_sec": .25, "duration_sec": 9.5, "frequency_hz": first,
                       "amplitude": .30, "timbre": "sine"})
    if case == "silence":
        events = []
    spec = {"case": case, "seed": seed, "duration_sec": 10.2, "events": events}
    audio = synthesize(spec)
    if case in ("vibrato", "glide"):
        event = events[0]
        n = round(event["duration_sec"] * FS)
        t = np.arange(n) / FS
        frequency = first * 2 ** (.02 * np.sin(math.tau * 5 * t) if case == "vibrato" else .08 * t)
        phase = math.tau * np.cumsum(frequency) / FS
        envelope = np.minimum(1, t / .012) * np.minimum(1, (n / FS - t) / .025)
        start = round(event["onset_sec"] * FS)
        audio[start:start + n] = event["amplitude"] * np.sin(phase) * envelope
    expected = [{"onset_sec": e["onset_sec"], "frequency_log2": math.log2(e["frequency_hz"] * harmonic)}
                for e in events for harmonic in ((1, 2, 3) if e["timbre"] == "harmonic" else (1,))]
    return audio, spec | {"expected_components": sorted(expected, key=lambda e: (e["onset_sec"], e["frequency_log2"]))}


def summarize(rows, truth=None):
    frames = [r for r in rows if r["kind"] == "components"]
    rises = [{"available_sec": row["available_sec"], **c} for row in frames for c in row["components"] if c["rise"]]
    result = {"analysis_frames": len(frames), "component_rises": len(rises),
              "frames_with_components": sum(bool(r["components"]) for r in frames),
              "maximum_simultaneous_components": max((len(r["components"]) for r in frames), default=0),
              "truncated_frames": sum(r["truncated_components"] > 0 for r in frames),
              "input_gaps": sum(r["kind"] == "input_gap" for r in rows)}
    if truth is not None:
        candidates = sorted((abs(observed["frequency_log2"] - expected["frequency_log2"]), oi, ei)
                            for oi, observed in enumerate(rises) for ei, expected in enumerate(truth["expected_components"])
                            if 0 <= observed["available_sec"] - expected["onset_sec"] <= .15
                            and abs(observed["frequency_log2"] - expected["frequency_log2"]) <= .05)
        matched, used = [], set()
        for _, oi, ei in candidates:
            if oi not in used and not any(pair[1] == ei for pair in matched):
                matched.append((oi, ei)); used.add(oi)
        result.update(expected_components=len(truth["expected_components"]), matched_components=len(matched),
                      extra_rises=len(rises) - len(matched), missing_components=len(truth["expected_components"]) - len(matched),
                      maximum_latency_sec=max((rises[oi]["available_sec"] - truth["expected_components"][ei]["onset_sec"] for oi, ei in matched), default=None),
                      maximum_frequency_error_cents=max((1200 * abs(rises[oi]["frequency_log2"] - truth["expected_components"][ei]["frequency_log2"]) for oi, ei in matched), default=None))
        result["observation_passed"] = result["extra_rises"] == result["missing_components"] == 0
    return result


def run(output, seeds, sample_root=None):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    output.mkdir(parents=True, exist_ok=False)
    plan = {"created_at": dt.datetime.now().astimezone().isoformat(), "seeds": seeds, "cases": CASES,
            "window_sec": WINDOW_SEC, "stride_sec": STRIDE_SEC, "max_components": MAX_COMPONENTS,
            "minimum_power": MIN_POWER, "minimum_spectral_concentration": MIN_CONCENTRATION,
            "maximum_matching_distance_oct": MATCH_OCT, "rise_ratio": RISE_RATIO,
            "refractory_sec": REFRACTORY_SEC,
            "acceptance": "All controlled spectral-component energy rises match within 150 ms and 60 cents, with no extras. Harmonic partials are distinct components. Report normal recordings without source-note labels.",
            "scope": "Research-only spectral ridges and energy rises; not source identity, pitch, phrase, closure or a predictor. Missing input resets evidence; initial observations cannot establish rises."}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    sources = {}
    for source in (Path(__file__), Path(__file__).with_name("evaluate_phrase_expectation.py")):
        shutil.copy2(source, output / source.name)
        sources[source.name] = sha256(source)
    inputs = []
    for seed in seeds:
        for case in CASES:
            folder = output / f"seed-{seed}-{case}";folder.mkdir()
            audio, truth = control_audio(case, seed)
            write_wav(folder / "audio.wav", audio)
            (folder / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
            inputs.append((folder, case, seed, truth, None))
    if sample_root is not None:
        samples = sorted(sample_root.glob("*/seed-*/audio.wav"))
        if not samples:
            raise ValueError("sample root contains no WAV inputs")
        for source in samples:
            folder = output / f"normal-{source.parent.parent.name}-{source.parent.name}";folder.mkdir()
            shutil.copy2(source, folder / "audio.wav")
            inputs.append((folder, source.parent.parent.name, int(source.parent.name.removeprefix("seed-")), None, source))
    manifest = {"status": "running", "plan_sha256": sha256(output / "plan.json"), "sources": sources, "rows": []}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for folder, case, seed, truth, source in inputs:
        with wave.open(str(folder / "audio.wav")) as audio:
            if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                raise ValueError("expected mono PCM16")
            fs = audio.getframerate()
            signal = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(float) / 32768
        observer, rows = FrequencyEventObserver(fs), []
        for start in range(0, len(signal), 4093):
            rows.extend(observer.process(start, signal[start:start + 4093]))
        with (folder / "components.jsonl").open("w") as file:
            for row in rows:
                file.write(json.dumps(row, allow_nan=False) + "\n")
        summary = summarize(rows, truth)
        row = {"directory": folder.name, "case": case, "seed": seed,
               "group": "controlled" if truth else "normal", "summary": summary,
               "original_audio": str(source) if source else None,
               "sha256": {p.name: sha256(p) for p in folder.iterdir()}}
        manifest["rows"].append(row)
        print(json.dumps({k: row[k] for k in ("directory", "summary")}), flush=True)
    manifest["status"] = "observed" if all(r["summary"].get("observation_passed", True) for r in manifest["rows"]) else "failed_observations"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["status"] != "observed":
        raise ValueError("frequency-event controls failed; preserve this run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--sample-root", type=Path)
    args = parser.parse_args()
    run(args.output, args.seeds, args.sample_root)
