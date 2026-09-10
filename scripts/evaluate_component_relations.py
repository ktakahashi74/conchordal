#!/usr/bin/env python3
"""Observe causal temporal relations between frequency channels in mixed audio."""

import argparse
from collections import deque
import datetime as dt
import json
import math
from pathlib import Path
import shutil
import wave

import numpy as np

from evaluate_frequency_events import FrequencyEventObserver, MIN_POWER, STRIDE_SEC, control_audio as frequency_control
from evaluate_phrase_expectation import FS, sha256, synthesize, write_wav


LO_LOG2 = math.log2(55)
HI_LOG2 = math.log2(7040)
BINS_PER_OCTAVE = 16
CENTERS_LOG2 = LO_LOG2 + np.arange(round((HI_LOG2 - LO_LOG2) * BINS_PER_OCTAVE) + 1) / BINS_PER_OCTAVE
PROJECTION_WIDTH_OCT = .075
FAST_SEC = .04
SLOW_SEC = .20
WINDOW_SEC = 2.0
WINDOW_FRAMES = round(WINDOW_SEC / STRIDE_SEC)
REPORT_FRAMES = 10
MIN_MODULATION_STD = .001
MIN_RELATIVE_MODULATION = .03
MIN_OCCUPANCY = .10
MAX_CHANNELS = 32
CASES = ("synchronous", "alternating", "independent", "two_pairs", "weaker_partner",
         "steady_pair", "startup_pair", "silence", "glide", "vibrato", "beating", "noise", "common_gain")


class ComponentRelations:
    """Signed envelope-modulation correlations, without source or stream labels."""

    def __init__(self):
        self.last_time = None
        self.after_input_gap = False
        self.segment_start = None
        self.fast = None
        self.slow = None
        self.history = deque()
        self.modulation_sum = np.zeros(len(CENTERS_LOG2))
        self.cross_sum = np.zeros((len(CENTERS_LOG2), len(CENTERS_LOG2)))
        self.power_sum = np.zeros(len(CENTERS_LOG2))
        self.occupancy_sum = np.zeros(len(CENTERS_LOG2), dtype=int)
        self.frames = 0
        self.outside_since_report = 0

    def process(self, row):
        time = row["available_sec"]
        if (not math.isfinite(time) or time < 0 or row["kind"] not in ("components", "input_gap")
                or self.last_time is not None and time <= self.last_time):
            raise ValueError("expected ordered finite component frames or a gap")
        components = row.get("components", [])
        for c in components:
            if not math.isfinite(c["frequency_log2"]) or not math.isfinite(c["power"]) or c["power"] <= 0:
                raise ValueError("expected finite component frequencies and positive powers")
        resumed = self.after_input_gap and row.get("initial_observation", False)
        gap = (row["kind"] == "input_gap" or self.last_time is not None and not resumed
               and not math.isclose(time - self.last_time, STRIDE_SEC, abs_tol=1e-7))
        previous_time = self.last_time
        if gap:
            self.__init__()
        self.last_time = time
        self.after_input_gap = row["kind"] == "input_gap"
        if row["kind"] == "input_gap":
            return {"kind": "input_gap", "available_sec": time, "previous_frame_sec": previous_time}
        power = np.zeros(len(CENTERS_LOG2))
        outside = 0
        for c in sorted(components, key=lambda c: (c["frequency_log2"], c["power"])):
            frequency = c["frequency_log2"]
            if not LO_LOG2 <= frequency <= HI_LOG2:
                outside += 1
                continue
            weights = np.exp(-.5 * ((CENTERS_LOG2 - frequency) / PROJECTION_WIDTH_OCT) ** 2)
            power += c["power"] * weights / np.sum(weights)
        assert power.shape == CENTERS_LOG2.shape
        amplitude = np.sqrt(power)
        if self.fast is None:
            self.fast = amplitude.copy()
            self.slow = amplitude.copy()
            self.segment_start = row.get("evidence_start_sec", time)
        else:
            self.fast += -math.expm1(-STRIDE_SEC / FAST_SEC) * (amplitude - self.fast)
            self.slow += -math.expm1(-STRIDE_SEC / SLOW_SEC) * (amplitude - self.slow)
        modulation = self.fast - self.slow
        occupied = power >= MIN_POWER
        if len(self.history) == WINDOW_FRAMES:
            old_modulation, old_power, old_occupied = self.history.popleft()
            self.modulation_sum -= old_modulation
            self.cross_sum -= np.outer(old_modulation, old_modulation)
            self.power_sum -= old_power
            self.occupancy_sum -= old_occupied
        self.history.append((modulation, power, occupied))
        self.modulation_sum += modulation
        self.cross_sum += np.outer(modulation, modulation)
        self.power_sum += power
        self.occupancy_sum += occupied
        self.frames += 1
        self.outside_since_report += outside
        if gap:
            return {"kind": "input_gap", "available_sec": time, "previous_frame_sec": previous_time}
        if self.frames % REPORT_FRAMES:
            return None
        n = len(self.history)
        output = {"kind": "relations", "available_sec": time, "evidence_start_sec": self.segment_start,
                  "window_frames": n, "status": "warming_up", "channels": [], "pairs": [],
                  "out_of_band_components": self.outside_since_report, "truncated_channels": 0,
                  "energetic_peaks": 0}
        self.outside_since_report = 0
        if n < WINDOW_FRAMES:
            return output
        mean_power = np.maximum(0, self.power_sum / n)
        covariance = self.cross_sum / n - np.outer(self.modulation_sum / n, self.modulation_sum / n)
        variance = np.maximum(0, np.diag(covariance))
        std = np.sqrt(variance)
        padded = np.pad(mean_power, (1, 1), constant_values=-1)
        energetic = ((mean_power > padded[:-2]) & (mean_power >= padded[2:])
                     & (mean_power >= MIN_POWER) & (self.occupancy_sum / n >= MIN_OCCUPANCY))
        output["energetic_peaks"] = int(np.sum(energetic))
        peaks = np.flatnonzero(energetic & (std >= np.maximum(MIN_MODULATION_STD,
                                                           MIN_RELATIVE_MODULATION * np.sqrt(mean_power))))
        order = sorted(peaks, key=lambda i: (-mean_power[i], i))
        output["truncated_channels"] = max(0, len(order) - MAX_CHANNELS)
        selected = sorted(order[:MAX_CHANNELS])
        output["channels"] = [{"bin": int(i), "frequency_log2": float(CENTERS_LOG2[i]),
                               "mean_power": float(mean_power[i]), "modulation_std": float(std[i]),
                               "occupancy": float(self.occupancy_sum[i] / n)} for i in selected]
        output["pairs"] = [{"a_bin": int(i), "b_bin": int(j),
                            "correlation": float(np.clip(covariance[i, j] / (std[i] * std[j]), -1, 1))}
                           for ai, i in enumerate(selected) for j in selected[ai + 1:]]
        output["status"] = "available" if output["pairs"] else "no_modulated_pair"
        return output


def evaluate(rows):
    model, output = ComponentRelations(), []
    for row in rows:
        result = model.process(row)
        if result is not None:
            output.append(result)
    return output


def control_audio(case, seed):
    if case not in CASES:
        raise ValueError("unknown component-relation control")
    if case in ("silence", "glide", "vibrato"):
        return frequency_control(case, seed)
    rng = np.random.default_rng(seed)
    root = float(rng.uniform(260, 350))
    partner = root * 1.55
    period_base = float(rng.uniform(.46, .56))
    period_partner = period_base * float(rng.uniform(.70, .82))
    duration = 11.2
    events = []
    paired = case == "two_pairs"
    if case in ("steady_pair", "startup_pair", "beating", "common_gain"):
        frequencies = (root, root + 8 if case == "beating" else partner)
        onset = 0 if case == "startup_pair" else .6
        for frequency in frequencies:
            events.append({"onset_sec": onset,
                           "duration_sec": duration - .7 - onset, "frequency_hz": frequency,
                           "amplitude": .18, "timbre": "sine"})
    elif case != "noise":
        for channel in range(2):
            period = period_partner if channel and case in ("independent", "two_pairs") else period_base
            delay = .5 * period_base if channel and case == "alternating" else 0
            times = np.arange(.6 + delay, duration - .7, period)
            frequencies = [root if channel == 0 else partner]
            if paired:
                frequencies.append(frequencies[0] * 2)
            for time in times:
                for frequency in frequencies:
                    events.append({"onset_sec": float(time), "duration_sec": .24 * period_base,
                                   "frequency_hz": frequency,
                                   "amplitude": .065 if channel and case == "weaker_partner" else .18,
                                   "timbre": "sine"})
    spec = {"case": case, "seed": seed, "events": events, "duration_sec": duration,
            "period_base_sec": period_base, "period_partner_sec": period_partner}
    audio = synthesize(spec)
    if case == "noise":
        audio = rng.normal(0, .07, round(duration * FS))
    elif case == "common_gain":
        audio *= .55 + .35 * np.sin(math.tau * np.arange(len(audio)) / FS / period_base)
    relation = "positive" if case in ("synchronous", "weaker_partner") else "negative" if case == "alternating" else "uncoupled"
    pairs = []
    if case in ("synchronous", "alternating", "independent", "weaker_partner"):
        pairs = [{"frequencies_log2": [math.log2(root), math.log2(partner)], "expected": relation}]
    elif paired:
        pairs = [{"frequencies_log2": [math.log2(f), math.log2(f * 2)], "expected": "positive"} for f in (root, partner)]
        pairs += [{"frequencies_log2": [math.log2(root), math.log2(partner)], "expected": "uncoupled"}]
    return audio, spec | {"expected_pairs": pairs}


def summarize(rows, truth=None):
    frames = [r for r in rows if r["kind"] == "relations" and 4 <= r["available_sec"] < 10]
    available = [r for r in frames if r["status"] == "available"]
    result = {"late_frames": len(frames), "available_frames": len(available),
              "no_modulated_pair_frames": sum(r["status"] == "no_modulated_pair" for r in frames),
              "maximum_channels": max((len(r["channels"]) for r in frames), default=0),
              "maximum_pairs": max((len(r["pairs"]) for r in frames), default=0),
              "truncated_frames": sum(r["truncated_channels"] > 0 for r in frames)}
    if truth is not None:
        tests = []
        for pair in truth.get("expected_pairs", []):
            values = []
            a, b = sorted(pair["frequencies_log2"])
            for row in frames:
                candidates = [(abs(float(CENTERS_LOG2[p["a_bin"]]) - a) + abs(float(CENTERS_LOG2[p["b_bin"]]) - b), p)
                              for p in row["pairs"]
                              if abs(float(CENTERS_LOG2[p["a_bin"]]) - a) <= .12
                              and abs(float(CENTERS_LOG2[p["b_bin"]]) - b) <= .12]
                if candidates:
                    values.append(min(candidates, key=lambda p: p[0])[1]["correlation"])
            median = float(np.median(values)) if values else None
            passed = len(values) >= .9 * len(frames) and median is not None and (
                median >= .8 if pair["expected"] == "positive" else
                median <= -.15 if pair["expected"] == "negative" else abs(median) <= .25)
            tests.append({**pair, "observed_frames": len(values), "median_correlation": median, "passed": bool(passed)})
        result["pair_checks"] = tests
        if tests:
            result["observation_passed"] = all(t["passed"] for t in tests)
        if truth["case"] in ("steady_pair", "startup_pair", "silence"):
            result["observation_passed"] = len(available) == 0 and bool(frames)
    return result


def run(output, seeds, sample_root=None):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("expected unique nonempty seeds")
    output.mkdir(parents=True, exist_ok=False)
    plan = {"created_at": dt.datetime.now().astimezone().isoformat(), "seeds": seeds, "cases": CASES,
            "channel_centers_log2": CENTERS_LOG2.tolist(), "projection_width_oct": PROJECTION_WIDTH_OCT,
            "fast_sec": FAST_SEC, "slow_sec": SLOW_SEC, "window_sec": WINDOW_SEC,
            "minimum_modulation_std": MIN_MODULATION_STD, "minimum_relative_modulation": MIN_RELATIVE_MODULATION,
            "minimum_occupancy": MIN_OCCUPANCY, "maximum_report_channels": MAX_CHANNELS,
            "controlled_evaluation": "Use 4-10 sec. At least 90 percent pair availability, median correlation >=0.8 for common activity, <=-0.15 for alternation, abs <=0.25 for independent periods. Constant/silent late input supplies no modulated pair. Glide, vibrato, beating, noise and common gain remain ambiguity diagnostics, not source-count checks.",
            "scope": "Research-only channel modulation relations. No stream count, Voice IDs, source separation, learned sequence, phrase or closure. IIR filter state depends on the whole contiguous segment."}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    sources = {}
    for name in (Path(__file__).name, "evaluate_frequency_events.py", "evaluate_phrase_expectation.py"):
        source = Path(__file__).with_name(name); shutil.copy2(source, output / name); sources[name] = sha256(source)
    inputs = []
    for seed in seeds:
        for case in CASES:
            folder = output / f"seed-{seed}-{case}"; folder.mkdir()
            audio, truth = control_audio(case, seed)
            write_wav(folder / "audio.wav", audio)
            (folder / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
            inputs.append((folder, case, seed, truth, None))
    if sample_root is not None:
        samples = sorted(sample_root.glob("*/seed-*/audio.wav"))
        if not samples:
            raise ValueError("sample root contains no WAV inputs")
        for source in samples:
            folder = output / f"normal-{source.parent.parent.name}-{source.parent.name}"; folder.mkdir()
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
        rows = evaluate(FrequencyEventObserver(fs).process(0, signal))
        (folder / "relations.json").write_text(json.dumps(rows, allow_nan=False) + "\n")
        summary = summarize(rows, truth)
        manifest["rows"].append({"directory": folder.name, "case": case, "seed": seed,
            "group": "normal" if truth is None else "controlled" if "observation_passed" in summary else "diagnostic", "summary": summary,
            "original_audio": str(source) if source else None, "sha256": {p.name: sha256(p) for p in folder.iterdir()}})
        print(json.dumps({"directory": folder.name, "summary": summary}), flush=True)
    manifest["status"] = "observed" if all(r["summary"].get("observation_passed", True) for r in manifest["rows"]) else "failed_observations"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["status"] != "observed":
        raise ValueError("component-relation controls failed; preserve this run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--sample-root", type=Path)
    args = parser.parse_args()
    run(args.output, args.seeds, args.sample_root)
