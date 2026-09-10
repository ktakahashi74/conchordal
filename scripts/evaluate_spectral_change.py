#!/usr/bin/env python3
"""Forecast the first spectral change and its direction, using only past audio."""

import argparse
from collections import deque
import datetime as dt
import json
import math
from pathlib import Path
import shutil
import wave

import numpy as np

from evaluate_phrase_expectation import FS, sha256, stimulus, synthesize, write_wav
from evaluate_spectral_expectation import (
    CASES, KERNEL_WIDTH, LOG2_HZ, MEMORY, STRIDE_SEC, SpectralObserver, relative_mass,
)


HORIZON_STEPS = (3, 6, 12)
CHANGE_DISTANCE = .25
CENTER_TOLERANCE_OCT = .025
CONTROLS = ("steady", "gain_only", "rising", "falling", "spread", "band_noise")
MODELS = ("ordered", "one_frame", "motion", "marginal", "persistence")
# Three change-time bins with three directions each, followed by no change.
PRIOR = np.array([1 / 12] * 9 + [1 / 4])


class ChangePredictor:
    """Bounded pending forecasts; learn labels only at their fixed final deadline."""

    def __init__(self):
        self.recent = deque(maxlen=2)
        self.history = deque(maxlen=MEMORY)
        self.pending = deque()
        self.last_time = None

    def forecast(self):
        if len(self.recent) < 2:
            return None
        anchor = float(self.recent[-1] @ LOG2_HZ)
        context = np.sqrt([relative_mass(p, anchor) for p in self.recent])
        motion = float(np.linalg.norm(np.sqrt(self.recent[-1]) - np.sqrt(self.recent[-2]))
                       / math.sqrt(2))
        targets = np.array([target for _, _, target in self.history], dtype=int)
        counts = np.bincount(targets, minlength=10)
        marginal = (counts + 4 * PRIOR) / (len(self.history) + 4)
        probabilities = {"marginal": marginal,
                         "persistence": np.array([.005 / 3] * 6 + [.01 / 3] * 3 + [.98])}
        supports = {}
        for name, count in (("one_frame", 1), ("ordered", 2), ("motion", 0)):
            if not self.history:
                kernels = np.empty(0)
            elif name == "motion":
                distance = (np.array([m for _, m, _ in self.history]) - motion) ** 2
                kernels = np.exp(-distance / (2 * KERNEL_WIDTH ** 2))
            else:
                past = np.array([c for c, _, _ in self.history])
                distance = np.sum((past[:, -count:] - context[-count:]) ** 2, axis=(1, 2)) / (2 * count)
                kernels = np.exp(-distance / (2 * KERNEL_WIDTH ** 2))
            support = float(np.sum(kernels))
            supports[name] = support
            probabilities[name] = (np.bincount(targets, weights=kernels, minlength=10)
                                   + 4 * marginal) / (support + 4)
        return {"probabilities": probabilities, "support": supports,
                "history_events": len(self.history), "context": context, "motion": motion}

    def process(self, time_sec, mass, *, unavailable_reason="unavailable"):
        if not math.isfinite(time_sec) or time_sec < 0:
            raise ValueError("expected a nonnegative finite time")
        if self.last_time is not None and time_sec <= self.last_time:
            raise ValueError("observations must arrive in strict time order")
        if mass is not None:
            mass = np.asarray(mass, dtype=float)
            if (mass.shape != LOG2_HZ.shape or not np.all(np.isfinite(mass))
                    or np.any(mass < 0) or not np.isclose(np.sum(mass), 1, rtol=1e-8)):
                raise ValueError("expected normalized finite Log2Space mass, or unavailable")
        discontinuity = (self.last_time is not None
                         and not math.isclose(time_sec - self.last_time, STRIDE_SEC, abs_tol=1e-7))
        output = []
        if mass is None or discontinuity:
            reason = "input_gap" if discontinuity else unavailable_reason
            for pending in self.pending:
                output.append({"kind": "censored", "time_sec": time_sec,
                               "origin_sec": pending["origin_sec"], "reason": reason})
            self.pending.clear()
            self.recent.clear()
        self.last_time = time_sec
        if mass is None:
            return output

        root = np.sqrt(mass)
        for pending in self.pending:
            pending["elapsed_steps"] += 1
            if pending["target"] is None:
                distance = float(np.linalg.norm(root - pending["reference_root"]) / math.sqrt(2))
                if distance >= CHANGE_DISTANCE:
                    delta = float(mass @ LOG2_HZ) - pending["reference_center"]
                    direction = 0 if delta < -CENTER_TOLERANCE_OCT else 2 if delta > CENTER_TOLERANCE_OCT else 1
                    time_bin = next(i for i, step in enumerate(HORIZON_STEPS)
                                    if pending["elapsed_steps"] <= step)
                    pending["target"] = time_bin * 3 + direction
                    pending["first_change_step"] = pending["elapsed_steps"]
                    pending["center_delta_oct"] = delta
        if self.pending and self.pending[0]["elapsed_steps"] == HORIZON_STEPS[-1]:
            pending = self.pending.popleft()
            target = 9 if pending["target"] is None else pending["target"]
            changed = target != 9
            time_bin = target // 3 if changed else 3
            losses = {}
            for name, probability in pending["forecast"]["probabilities"].items():
                times = np.append(probability[:9].reshape(3, 3).sum(axis=1), probability[9])
                cumulative = np.cumsum(times[:3])
                occurrence_losses = []
                for bin_index, chance in enumerate(cumulative):
                    occurred = changed and time_bin <= bin_index
                    occurrence_losses.append(-math.log2(float(chance if occurred else 1 - chance)))
                losses[name] = {"joint": -math.log2(float(probability[target])),
                                "time": -math.log2(float(times[time_bin])),
                                "direction": (-math.log2(float(probability[target] / times[time_bin]))
                                              if changed else None),
                                "occurrence": occurrence_losses}
            output.append({"kind": "score", "time_sec": time_sec,
                           "origin_sec": pending["origin_sec"], "target": target,
                           "first_change_step": pending["first_change_step"],
                           "center_delta_oct": pending["center_delta_oct"], "loss_bits": losses})
            prediction = pending["forecast"]
            self.history.append((prediction["context"], prediction["motion"], target))

        self.recent.append(mass.copy())
        prediction = self.forecast()
        if prediction is not None:
            self.pending.append({"origin_sec": time_sec, "reference_root": root,
                                 "reference_center": float(mass @ LOG2_HZ),
                                 "forecast": prediction, "elapsed_steps": 0, "target": None,
                                 "first_change_step": None, "center_delta_oct": None})
            output.append({"kind": "forecast", "time_sec": time_sec,
                           "deadline_sec": time_sec + HORIZON_STEPS[-1] * STRIDE_SEC,
                           "history_events": prediction["history_events"],
                           "support": prediction["support"],
                           "probabilities": {name: p.tolist() for name, p in prediction["probabilities"].items()}})
        assert len(self.pending) <= HORIZON_STEPS[-1]
        return output


def evaluate_audio(audio, fs):
    observer, predictor = SpectralObserver(fs), ChangePredictor()
    records, availability = [], []
    for start in range(0, len(audio), 4093):
        for row in observer.process(start, audio[start:start + 4093]):
            mass = row.pop("mass")
            availability.append({**row, "observed": mass is not None})
            records.extend(predictor.process(row["available_sec"], mass,
                                             unavailable_reason=row["kind"]))
    return records, availability, len(predictor.pending)


def summarize(records, availability, pending, start_sec):
    scores = [r for r in records if r["kind"] == "score" and r["origin_sec"] >= start_sec]
    changed = [r for r in scores if r["target"] != 9]
    losses = {}
    for name in MODELS:
        losses[name] = {}
        for part, rows in (("joint", scores), ("time", scores), ("direction", changed)):
            losses[name][part] = (float(np.mean([r["loss_bits"][name][part] for r in rows]))
                                  if rows else None)
    gains = {name: {part: losses[name][part] - losses["ordered"][part]
                   if losses[name][part] is not None else None
                   for part in ("joint", "time", "direction")}
             for name in MODELS if name != "ordered"}
    forecasts = sum(r["kind"] == "forecast" for r in records)
    censored = sum(r["kind"] == "censored" for r in records)
    resolved = sum(r["kind"] == "score" for r in records)
    assert forecasts == resolved + censored + pending
    return {"evaluation_start_sec": start_sec, "scored_origins": len(scores),
            "changed_origins": len(changed), "target_counts": np.bincount(
                [r["target"] for r in scores], minlength=10).tolist(),
            "center_delta_range_oct": ([min(r["center_delta_oct"] for r in changed),
                                         max(r["center_delta_oct"] for r in changed)] if changed else None),
            "gain_bits": gains, "mean_loss_bits": losses,
            "observed_windows": sum(r["observed"] for r in availability),
            "forecast_count": forecasts, "resolved_count": resolved,
            "censored_count": censored, "pending_at_eof": pending}


def control_audio(case, seed):
    if case not in CONTROLS:
        raise ValueError("unknown control")
    rng = np.random.default_rng(seed)
    t = np.arange(30 * FS) / FS
    base = rng.uniform(240, 340)
    phase = rng.uniform(0, math.tau)
    tone = np.sin(math.tau * base * t + phase) + .7 * np.sin(math.tau * math.sqrt(2) * base * t)
    if case == "gain_only":
        tone *= .6 + .25 * np.sin(math.tau * t / 2.3)
    elif case in ("rising", "falling"):
        slope = .09 if case == "rising" else -.09
        frequencies = base * 2 ** (slope * (t - 15))
        tone = np.sin(math.tau * np.cumsum(frequencies) / FS + phase)
    elif case == "spread":
        spread = np.where((t.astype(int) // 2) % 2 == 0, .25, .5)
        low, high = base * 2 ** -spread, base * 2 ** spread
        tone = np.sin(math.tau * np.cumsum(low) / FS + phase) + np.sin(math.tau * np.cumsum(high) / FS)
    elif case == "band_noise":
        noise = rng.normal(size=len(t))
        offset = np.arange(257) - 128
        low_pass = 2 * 4000 / FS * np.sinc(2 * 4000 / FS * offset)
        high_pass = 2 * 500 / FS * np.sinc(2 * 500 / FS * offset)
        kernel = (low_pass - high_pass) * np.hamming(len(offset))
        tone = np.convolve(noise, kernel, mode="full")[:len(t)]
        tone /= 1.5
    envelope = np.clip(t / .02, 0, 1) * np.clip((30 - t) / .02, 0, 1)
    audio = .12 * tone * envelope
    if np.max(np.abs(audio)) >= 1:
        raise ValueError("control clips")
    return audio


def assess(rows, seeds):
    expected = {(seed, case) for seed in seeds for case in (*CASES, *CONTROLS)}
    actual = [(row["seed"], row["case"]) for row in rows if row["group"] == "controlled"]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("incomplete or duplicate control conditions")
    normal = [r for r in rows if r["group"] == "normal"]
    checks = []
    for row in rows:
        if row["case"] not in CONTROLS:
            continue
        stats = row["summary"]
        counts = stats["target_counts"]
        if row["case"] in ("steady", "gain_only"):
            passed = stats["scored_origins"] > 100 and stats["changed_origins"] == 0
        elif row["case"] in ("rising", "falling"):
            delta_range = stats["center_delta_range_oct"]
            # Small true glides can remain inside the stable-center category.
            passed = stats["changed_origins"] > 30 and (
                delta_range[0] > .005 if row["case"] == "rising" else delta_range[1] < -.005)
        elif row["case"] == "spread":
            centered = sum(counts[i] for i in (1, 4, 7))
            passed = stats["changed_origins"] > 30 and centered / stats["changed_origins"] > .8
        else:
            # In-band noise must be observed; it is not a positive sequence control.
            passed = stats["scored_origins"] > 100
        checks.append({"case": row["case"], "seed": row["seed"], "passed": bool(passed)})
    advantage = []
    for row in normal:
        stats = row["summary"]
        gains = stats["gain_bits"]
        advantage.append({"source": row["source"],
                          "time_advantage": stats["scored_origins"] > 0 and all(
                              gains[name]["time"] > 0 for name in MODELS if name != "ordered"),
                          "direction_advantage": stats["changed_origins"] > 0 and all(
                              gains[name]["direction"] > 0 for name in MODELS if name != "ordered")})
    return {"control_checks": checks, "controls_passed": all(c["passed"] for c in checks) and bool(checks),
            "normal_comparison": advantage, "runtime_attachment": "research_only",
            "scope": "Descriptive acoustic prediction; overlapping origins are not independent events; no melody or closure claim"}


def run(output, seeds, sample_root=None):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    samples = sorted(sample_root.glob("*/seed-*/audio.wav")) if sample_root is not None else []
    if sample_root is not None and not samples:
        raise ValueError("sample root contains no expected audio files")
    output.mkdir(parents=True, exist_ok=False)
    plan = {"created_at": dt.datetime.now().astimezone().isoformat(), "seeds": seeds,
            "cases": CASES, "controls": CONTROLS, "note_duration_sec": .50,
            "stride_sec": STRIDE_SEC, "horizon_steps": HORIZON_STEPS,
            "hellinger_change_distance": CHANGE_DISTANCE, "center_tolerance_oct": CENTER_TOLERANCE_OCT,
            "kernel_width": KERNEL_WIDTH, "memory_events": MEMORY,
            "target": "First crossing from the frozen origin spectrum, binned by time and log-frequency center displacement",
            "censoring": "Require the entire 12-step horizon to remain observed; unknown input censors even a previously seen crossing; EOF leaves pending forecasts unresolved",
            "learning": "Labels enter memory only at the fixed 12-step deadline, before forecasting from that current observed frame",
            "controls_required": "Steady and global gain: zero changes; glide: more than 30 changed origins, each centroid displacement has the known sign beyond .005 oct (the .025 oct direction category may remain stable); spread: more than 30 changed origins, over 80 percent with stable center; in-band noise: more than 100 scored origins",
            "model_check": "Report time and direction gains separately against marginal, one-frame, motion and persistence; successful execution alone permits no attachment",
            "scope": "Research contract, not universal perceptual thresholds; no generator or runtime change"}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    sources = {}
    for name in (Path(__file__).name, "evaluate_spectral_expectation.py", "evaluate_phrase_expectation.py"):
        source = Path(__file__).with_name(name)
        shutil.copy2(source, output / name)
        sources[name] = sha256(source)
    manifest = {"status": "running", "seeds": seeds, "sources": sources,
                "numpy_version": np.__version__, "plan_sha256": sha256(output / "plan.json"), "rows": []}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    inputs = []
    for seed in seeds:
        for case in (*CASES, *CONTROLS):
            folder = output / f"seed-{seed}" / case
            folder.mkdir(parents=True)
            if case in CASES:
                spec = stimulus(case, seed)
                for event in spec["events"]:
                    event["duration_sec"] = .50
                audio = synthesize(spec)
                evaluation_start = spec["evaluation_start_sec"]
                (folder / "truth.json").write_text(json.dumps(spec, indent=2) + "\n")
            else:
                audio, evaluation_start = control_audio(case, seed), 10
            write_wav(folder / "audio.wav", audio)
            inputs.append((folder / "audio.wav", folder, "controlled", case, seed, evaluation_start))
    for path in samples:
        folder = output / "normal" / path.parent.parent.name / path.parent.name
        folder.mkdir(parents=True)
        inputs.append((path, folder, "normal", path.parent.parent.name,
                       int(path.parent.name.removeprefix("seed-")), 10))
    for path, folder, group, case, seed, start in inputs:
        with wave.open(str(path)) as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError("expected mono PCM16")
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float) / 32768
            fs = wav.getframerate()
        records, availability, pending = evaluate_audio(audio, fs)
        (folder / "records.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        (folder / "availability.json").write_text(json.dumps(availability, indent=2, allow_nan=False) + "\n")
        row = {"group": group, "case": case, "seed": seed, "source": str(path),
               "audio_sha256": sha256(path), "directory": str(folder.relative_to(output)),
               "summary": summarize(records, availability, pending, start),
               "sha256": {p.name: sha256(p) for p in folder.iterdir()}}
        manifest["rows"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k in ("group", "case", "seed", "summary")}), flush=True)
    assessment = assess(manifest["rows"], seeds)
    (output / "assessment.json").write_text(json.dumps(assessment, indent=2) + "\n")
    manifest["assessment_sha256"] = sha256(output / "assessment.json")
    manifest["status"] = "complete" if assessment["controls_passed"] else "failed_controls"
    manifest["status_scope"] = "Control and execution result, not musical or model acceptance"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    if not assessment["controls_passed"]:
        raise ValueError("control checks failed; preserve the result and review the contract")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--sample-root", type=Path)
    args = parser.parse_args()
    run(args.output, args.seeds, args.sample_root)
