#!/usr/bin/env python3
"""Observe and predict interval order from the existing Rust acoustic onset path."""

import argparse
from collections import deque
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import wave

import numpy as np

from evaluate_phrase_expectation import FS, sha256, synthesize, write_wav


CASES = ("repeat", "reordered", "rotated", "shuffled", "quieter", "timbre", "overlap", "silence")
PATTERNS = {
    "paired": ((0, 0, 2, 2), (0, 2, 0, 2)),
    "staggered": ((0, 1, 0, 2), (0, 0, 1, 2)),
    "mirrored": ((0, 1, 2, 2, 1, 0), (0, 1, 2, 0, 1, 2)),
    "long_runs": ((0, 0, 0, 2, 2, 2), (0, 2, 0, 2, 0, 2)),
    "asymmetric": ((0, 1, 2), (0, 2, 1)),
}
MEMORY = 128
CONTEXT_WIDTH = .10
TARGET_WIDTH = .07
STEP_SEC = .01
ENERGY_WINDOWS_SEC = (.02, .04)
ENERGY_RISE_RATIO = 1.15
MODELS = ("ordered", "one_interval", "marginal", "persistence")
ASSAY = "core::temporal_expectation::audio_assay::export_temporal_assay"


def interval_audio(case, seed, pattern="paired"):
    if case not in CASES or pattern not in PATTERNS:
        raise ValueError("unknown interval-order control or pattern")
    rng = np.random.default_rng(seed)
    short = round(rng.uniform(.23, .31) * FS)
    long = round(short * rng.uniform(1.65, 1.95))
    frequency = float(rng.uniform(260, 380))
    values = (short, round(math.sqrt(short * long)), long)
    before, after = ([values[i] for i in order] for order in PATTERNS[pattern])
    assert sorted(before) == sorted(after)
    assert all(after != before[n:] + before[:n] for n in range(len(before)))
    cursor = round(.6 * FS)
    split = (cursor + 8 * sum(before)) / FS
    events, intervals = [], []
    for cycle in range(16):
        order = before.copy()
        if case == "rotated":
            order = before[1:] + before[:1]
        elif case == "shuffled":
            rng.shuffle(order)
        elif case == "reordered" and cycle >= 8:
            order = after.copy()
        for interval in order:
            changed = cycle >= 8
            amplitude = .25 * (.55 if case == "quieter" and changed else 1)
            if case == "overlap" and changed:
                amplitude /= math.sqrt(2)
            events.append({"onset_sec": cursor / FS, "duration_sec": .12,
                           "frequency_hz": frequency, "amplitude": amplitude,
                           "timbre": "harmonic" if case == "timbre" and changed else "sine"})
            intervals.append(interval / FS)
            cursor += interval
    # Sound the endpoint so every permuted interval has an observed successor.
    events.append({**events[-1], "onset_sec": cursor / FS})
    spec = {"seed": seed, "duration_sec": cursor / FS + .5,
            "events": events if case != "silence" else []}
    audio = synthesize(spec)
    if case == "overlap":
        second = {**spec, "events": [{**event, "frequency_hz": frequency * math.sqrt(2)}
                                     for event in events[8 * len(before):]]}
        audio += synthesize(second)
    if np.max(np.abs(audio), initial=0) >= 1:
        raise ValueError("interval-order stimulus clips")
    truth = {"case": case, "seed": seed, "pattern": pattern, "events": spec["events"], "split_sec": split,
             "duration_sec": spec["duration_sec"], "intervals_sec": intervals,
             "before_interval_frames": before, "after_interval_frames": after,
             "source_frequency_hz": frequency, "note_sec": .12,
             "scope": "Intervals are permuted within equal-duration blocks; no extra beat carrier or phrase label"}
    return audio, truth


def densities(prediction, interval_sec):
    if not math.isfinite(interval_sec) or interval_sec <= 0:
        raise ValueError("expected a positive finite interval")
    x = math.log2(interval_sec)
    prior = math.exp(-.5 * ((x + 1) / 2) ** 2) / (2 * math.sqrt(math.tau))
    kernels = np.exp(-.5 * ((x - np.asarray(prediction["means_log2_sec"])) / TARGET_WIDTH) ** 2)
    kernels /= TARGET_WIDTH * math.sqrt(math.tau)
    result = {name: .05 * prior + float(np.dot(weights, kernels))
              for name, weights in prediction["weights"].items()}
    last = prediction["context_log2_sec"][-1]
    result["persistence"] = (.05 * prior + .95 * math.exp(-.5 * ((x - last) / TARGET_WIDTH) ** 2)
                             / (TARGET_WIDTH * math.sqrt(math.tau)))
    return {name: max(value, 1e-300) for name, value in result.items()}


def confirm_attacks(rows, audio, fs):
    """Require a causal energy rise at native flux candidates; expose rejected evidence."""
    audio = np.asarray(audio, dtype=float)
    if audio.ndim != 1 or not np.all(np.isfinite(audio)) or not isinstance(fs, int) or fs < 16_000:
        raise ValueError("expected finite mono audio and a supported sample rate")
    widths = [round(duration * fs) for duration in ENERGY_WINDOWS_SEC]
    result = []
    for row in rows:
        end = round(float(row["time_sec"]) * fs)
        if end < 0 or end > len(audio):
            raise ValueError("candidate time is outside available audio")
        band = int(row["observed_band"])
        if band not in range(4):
            raise ValueError("invalid acoustic onset band")
        evidence = {name: None for name in ("short_energy_now", "short_energy_before", "long_energy_now", "long_energy_before")}
        accepted = False
        if band and end >= 2 * max(widths):
            accepted = True
            for name, width in zip(("short", "long"), widths):
                now = float(np.mean(audio[end - width:end] ** 2))
                before = float(np.mean(audio[end - 2 * width:end - width] ** 2))
                evidence[name + "_energy_now"] = now
                evidence[name + "_energy_before"] = before
                accepted = accepted and now >= .005 ** 2 and now > before * ENERGY_RISE_RATIO
        result.append({"time_sec": float(row["time_sec"]), "candidate_band": band,
                       "observed_band": band if accepted else 0, **evidence})
    return result


class IntervalPredictor:
    """Forecast the next detected interval before it arrives; never infer closure."""

    def __init__(self):
        self.history = deque(maxlen=MEMORY)
        self.context = deque(maxlen=2)
        self.losses = deque(maxlen=MEMORY)
        self.last_observation = None
        self.last_onset = None
        self.pending = None

    def forecast(self, time_sec):
        if len(self.context) != 2 or not self.history:
            return None
        contexts = np.asarray([row[0] for row in self.history])
        means = [row[1] for row in self.history]
        marginal = np.full(len(means), .95 / len(means))
        weights = {"marginal": marginal.tolist()}
        support = {}
        for name, count in (("ordered", 2), ("one_interval", 1)):
            distance = np.mean((contexts[:, -count:] - np.asarray(self.context)[-count:]) ** 2, axis=1)
            kernels = np.exp(-distance / (2 * CONTEXT_WIDTH ** 2))
            support[name] = float(np.sum(kernels))
            strength = support[name] / (support[name] + 2)
            weights[name] = ((1 - strength) * marginal
                             + strength * .95 * kernels / max(support[name], 1e-300)).tolist()
        threshold = (float(np.quantile(self.losses, .95)) + 2
                     if support["ordered"] >= 2 and len(self.losses) >= 16 else None)
        return {"issued_sec": time_sec, "means_log2_sec": means, "weights": weights,
                "context_log2_sec": list(self.context), "support": support,
                "error_threshold_bits": threshold, "calibration_events": len(self.losses)}

    def process(self, time_sec, fired):
        if (not math.isfinite(time_sec) or time_sec < 0 or not isinstance(fired, bool)
                or self.last_observation is not None and time_sec <= self.last_observation):
            raise ValueError("expected strictly ordered finite observation times and a boolean onset")
        gap = (self.last_observation is not None
               and not math.isclose(time_sec - self.last_observation, STEP_SEC, abs_tol=1e-7))
        self.last_observation = time_sec
        if gap:
            censored = self.pending
            self.context.clear()
            self.last_onset = self.pending = None
            return {"kind": "gap", "available_sec": time_sec, "censored_forecast": censored}
        if not fired:
            return None
        prediction, score, interval = self.pending, None, None
        if self.last_onset is not None:
            interval = time_sec - self.last_onset
            if prediction is not None:
                values = densities(prediction, interval)
                loss = {name: -math.log2(value) for name, value in values.items()}
                threshold = prediction["error_threshold_bits"]
                score = {"forecast": prediction, "interval_sec": interval, "loss_bits": loss,
                         "gain_bits": {name: loss[name] - loss["ordered"] for name in MODELS[1:]},
                         "error_candidate": threshold is not None and loss["ordered"] > threshold}
                if prediction["support"]["ordered"] >= 2:
                    self.losses.append(loss["ordered"])
            value = math.log2(interval)
            if len(self.context) == 2:
                self.history.append((tuple(self.context), value))
            self.context.append(value)
        self.last_onset = time_sec
        self.pending = self.forecast(time_sec)
        return {"kind": "onset", "available_sec": time_sec, "interval_sec": interval,
                "score": score, "next_forecast": self.pending}


def evaluate(rows):
    model, result = IntervalPredictor(), []
    for row in rows:
        band = int(row["observed_band"])
        if band not in range(4):
            raise ValueError("invalid acoustic onset band")
        observation = model.process(float(row["time_sec"]), bool(band))
        if observation is not None:
            result.append(observation)
    return {"events": result, "right_censored_forecast": model.pending}


def summarize(observed, truth=None):
    events = [r for r in observed["events"] if r["kind"] == "onset"]
    result = {"detected_onsets": len(events), "input_gaps": sum(r["kind"] == "gap" for r in observed["events"]),
              "error_candidates_sec": [r["available_sec"] for r in events
                                       if r["score"] is not None and r["score"]["error_candidate"]]}
    windows = {"all_after_10_sec": (10, math.inf)} if truth is None else {
        "trained_before_change": (truth["split_sec"] - 6, truth["split_sec"]),
        "first_six_seconds_after_change": (truth["split_sec"], truth["split_sec"] + 6),
        "later": (truth["split_sec"] + 6, truth["duration_sec"])}
    result["windows"] = {}
    for name, (start, end) in windows.items():
        scores = [r["score"] for r in events if start <= r["available_sec"] < end and r["score"] is not None]
        result["windows"][name] = {"scored_intervals": len(scores),
            "eligible_error_checks": sum(s["forecast"]["error_threshold_bits"] is not None for s in scores),
            "unavailable_error_checks": sum(s["forecast"]["error_threshold_bits"] is None for s in scores),
            "gain_bits": {key: float(np.mean([s["gain_bits"][key] for s in scores])) if scores else None
                          for key in MODELS[1:]}}
    if truth is not None:
        times = [r["available_sec"] for r in events]
        expected = [r["onset_sec"] for r in truth["events"]]
        errors = [found - source for found, source in zip(times, expected)]
        result["expected_onsets"] = len(expected)
        result["max_onset_error_sec"] = max(map(abs, errors), default=None)
        result["observation_passed"] = len(times) == len(expected) and all(-.001 <= e <= .05 for e in errors)
    return result


def run(output, seeds, sample_root=None, patterns=("paired",)):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    if not patterns or len(set(patterns)) != len(patterns) or any(p not in PATTERNS for p in patterns):
        raise ValueError("patterns must be known, nonempty and unique")
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = {"created_at": dt.datetime.now().astimezone().isoformat(), "seeds": seeds, "cases": CASES,
            "patterns": {name: PATTERNS[name] for name in patterns},
            "pattern_scope": "Only stimulus order varies. Predictor and attack confirmation coefficients remain fixed. Some sequences require more than two intervals or only one; do not require a two-interval model to win everywhere",
            "context_width_log2_sec": CONTEXT_WIDTH, "target_width_log2_sec": TARGET_WIDTH,
            "memory_transitions": MEMORY, "step_sec": STEP_SEC,
            "attack_confirmation": {"adjacent_trailing_energy_windows_sec": ENERGY_WINDOWS_SEC,
                                    "minimum_energy_ratio": ENERGY_RISE_RATIO, "minimum_rms": .005,
                                    "rule": "Both scales must rise; short windows reject delayed plateau/release flux, long windows reduce two-tone energy ripples",
                                    "scope": "Energy attacks only; equal-level spectral changes may remain unobserved. Preserve raw flux candidates and rejected candidates"},
            "forecast": "Density of the next observed inter-onset interval in log2 seconds; issued at a detected onset, scored at the next one before updating. Density is not a discrete probability",
            "controls": "Two-interval context, last-interval context, empirical marginal, and last-interval persistence share a 5 percent broad Gaussian prior. Onset labels are measured by the existing Rust audio path; generator times enter evaluation only",
            "censoring": "Missing observation steps clear pending predictions and temporal context. The first resumed row cannot establish a fresh onset. EOF does not score the pending interval or imply closure",
            "acceptance": "All controlled onsets match within 50 ms with no extras. Report prediction gains and novelty separately; successful onset matching is not musical or model acceptance",
            "scope": "Research comparison; no production intervention, phrase label, or audible-flow claim"}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    sources = [Path(__file__).relative_to(root), Path("scripts/evaluate_phrase_expectation.py"),
               Path("Cargo.toml"), Path("Cargo.lock"),
               *[p.relative_to(root) for p in sorted((root / "src").rglob("*.rs"))]]
    hashes = {}
    for source in sources:
        target = output / "source" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / source, target)
        hashes[str(source)] = sha256(root / source)
    manifest = {"status": "running", "plan_sha256": sha256(output / "plan.json"),
                "sources": hashes, "seeds": seeds, "numpy_version": np.__version__, "rows": []}
    inputs = []
    for pattern in patterns:
        for seed in seeds:
            for case in CASES:
                folder = output / f"{pattern}-seed-{seed}-{case}"
                folder.mkdir()
                audio, truth = interval_audio(case, seed, pattern)
                write_wav(folder / "audio.wav", audio)
                (folder / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
                inputs.append((folder, case, seed, truth, None))
    if sample_root is not None:
        samples = sorted(sample_root.glob("*/seed-*/audio.wav"))
        if not samples:
            raise ValueError("sample root contains no audio")
        for source in samples:
            folder = output / f"normal-{source.parent.parent.name}-{source.parent.name}"
            folder.mkdir()
            shutil.copy2(source, folder / "audio.wav")
            inputs.append((folder, source.parent.parent.name, int(source.parent.name.removeprefix("seed-")), None, source))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    build = subprocess.run(["cargo", "test", "--lib", "--no-run", "--message-format=json"],
                           cwd=root, text=True, capture_output=True)
    (output / "build.jsonl").write_text(build.stdout)
    (output / "build.stderr.txt").write_text(build.stderr)
    build.check_returncode()
    binaries = [row["executable"] for line in build.stdout.splitlines()
                if (row := json.loads(line)).get("reason") == "compiler-artifact"
                and row.get("executable") and row["target"]["kind"] == ["lib"]]
    if len(binaries) != 1:
        raise ValueError("expected one library test binary")
    manifest["observer_binary"] = binaries[0]
    manifest["observer_binary_sha256"] = sha256(Path(binaries[0]))
    with (output / "observer.log").open("w") as log:
        process = subprocess.run([binaries[0], ASSAY, "--exact", "--ignored", "--nocapture"], cwd=root,
                                 env=os.environ | {"CONCHORDAL_TEMPORAL_ASSAY_DIR": str(output), "RUST_BACKTRACE": "1"},
                                 stdout=log, stderr=subprocess.STDOUT)
    (output / "observer-status.json").write_text(json.dumps({"exit_code": process.returncode}) + "\n")
    process.check_returncode()
    if not all((folder / "acoustic_recurrence.csv").is_file() for folder, *_ in inputs):
        manifest["status"] = "failed_observer_export"
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        raise ValueError("observer did not export all requested audio inputs")
    for folder, case, seed, truth, source in inputs:
        with (folder / "acoustic_recurrence.csv").open() as file:
            raw = list(csv.DictReader(file))
        with wave.open(str(folder / "audio.wav")) as wav:
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float) / 32768
            fs = wav.getframerate()
        confirmed = confirm_attacks(raw, audio, fs)
        with (folder / "interval_onsets.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(confirmed[0]))
            writer.writeheader()
            writer.writerows(confirmed)
        observation = evaluate(confirmed)
        (folder / "predictions.json").write_text(json.dumps(observation, indent=2, allow_nan=False) + "\n")
        result = {"directory": folder.name, "group": "controlled" if truth is not None else "normal",
                  "case": case, "seed": seed, "pattern": truth["pattern"] if truth else None,
                  "original_audio": str(source) if source else None,
                  "summary": summarize(observation, truth), "raw_summary": summarize(evaluate(raw), truth),
                  "sha256": {p.name: sha256(p) for p in folder.iterdir()}}
        manifest["rows"].append(result)
        print(json.dumps({k: result[k] for k in ("group", "case", "seed", "pattern", "summary")}), flush=True)
    controls = [row for row in manifest["rows"] if row["group"] == "controlled"]
    assert len(controls) == len(CASES) * len(seeds) * len(patterns)
    manifest["status"] = "observed" if all(r["summary"]["observation_passed"] for r in controls) else "failed_observations"
    manifest["status_scope"] = "Acoustic event matching only; prediction usefulness and author judgments are separate"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["status"] != "observed":
        raise ValueError("interval-order event observation failed; preserve this run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--sample-root", type=Path)
    parser.add_argument("--patterns", nargs="+", choices=PATTERNS, default=["paired"])
    args = parser.parse_args()
    run(args.output, args.seeds, args.sample_root, args.patterns)
