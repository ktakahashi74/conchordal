#!/usr/bin/env python3
"""Research-only prediction of relative spectral mass in overlapping audio.

This target is neither a source pitch nor a separated melody. Only past audio
features enter forecasts; scenario labels belong to the evaluation side.
"""

import argparse
from collections import deque
import datetime as dt
import json
from pathlib import Path
import shutil
import wave

import numpy as np

from evaluate_phrase_expectation import sha256, stimulus, synthesize, write_wav


BINS_PER_OCT = 48
LOG2_HZ = np.log2(55.0) + np.arange(337) / BINS_PER_OCT
RELATIVE_OCT = np.arange(-336, 337) / BINS_PER_OCT
STRIDE_SEC = .12
MEMORY = 128
KERNEL_WIDTH = .15
CASES = ("repeat", "altered", "shuffled", "transposed", "timbre", "quieter",
         "renewal", "noise", "silence")


def relative_mass(mass, anchor):
    """Deposit probability mass on a relative log-frequency axis, without folding."""
    position = (LOG2_HZ - anchor - RELATIVE_OCT[0]) * BINS_PER_OCT
    position = np.clip(position, 0, len(RELATIVE_OCT) - 1)
    lower = np.floor(position).astype(int)
    fraction = position - lower
    result = np.bincount(lower, weights=mass * (1 - fraction), minlength=len(RELATIVE_OCT))
    result += np.bincount(np.minimum(lower + 1, len(RELATIVE_OCT) - 1),
                          weights=mass * fraction, minlength=len(RELATIVE_OCT))
    return result


class SpectralObserver:
    """Fixed-stride, trailing windows; bounded input memory and no EOF inference."""

    def __init__(self, fs):
        if not isinstance(fs, int) or fs < 16_000:
            raise ValueError("expected an integer sample rate of at least 16000 Hz")
        self.fs = fs
        self.stride = round(STRIDE_SEC * fs)
        self.window_size = round(2048 * fs / 24_000)
        self.taper = np.hanning(self.window_size)
        frequencies = np.fft.rfftfreq(self.window_size, 1 / fs)
        self.in_band = (frequencies >= 55) & (frequencies <= 7040)
        position = (np.log2(frequencies[self.in_band]) - LOG2_HZ[0]) * BINS_PER_OCT
        self.lower = np.floor(position).astype(int)
        self.fraction = position - self.lower
        self.window = deque(maxlen=self.window_size)
        self.next_sample = None
        self.pending_count = 0

    def process(self, start_sample, audio):
        audio = np.asarray(audio, dtype=float)
        if (not isinstance(start_sample, int) or start_sample < 0 or audio.ndim != 1
                or not np.all(np.isfinite(audio))):
            raise ValueError("expected a nonnegative sample index and finite mono audio")
        if self.next_sample is not None and start_sample < self.next_sample:
            raise ValueError("audio overlaps or arrives out of order")
        if not len(audio):
            return []
        records = []
        if self.next_sample is not None and start_sample != self.next_sample:
            records.append({"kind": "input_gap", "available_sec": start_sample / self.fs,
                            "missing_start_sec": self.next_sample / self.fs, "mass": None})
            self.window.clear()
            self.pending_count = 0
        cursor = 0
        while cursor < len(audio):
            count = min(self.stride - self.pending_count, len(audio) - cursor)
            self.window.extend(audio[cursor:cursor + count])
            self.pending_count += count
            cursor += count
            if self.pending_count < self.stride:
                continue
            self.pending_count = 0
            x = np.asarray(self.window)
            assert len(x) == self.window_size
            rms = float(np.sqrt(np.mean(x * x)))
            mass = None
            if rms >= .005:
                power = np.abs(np.fft.rfft((x - np.mean(x)) * self.taper)) ** 2
                values = power[self.in_band]
                in_band_fraction = float(np.sum(values) / max(float(np.sum(power)), 1e-30))
                if in_band_fraction >= .95:
                    mass = np.bincount(self.lower, weights=values * (1 - self.fraction),
                                       minlength=len(LOG2_HZ))
                    mass += np.bincount(np.minimum(self.lower + 1, len(LOG2_HZ) - 1),
                                        weights=values * self.fraction, minlength=len(LOG2_HZ))
                    mass /= np.sum(mass)
            else:
                in_band_fraction = None
            end = start_sample + cursor
            records.append({"kind": "spectrum", "available_sec": end / self.fs,
                            "window_start_sec": (end - self.window_size) / self.fs,
                            "rms": rms, "in_band_fraction": in_band_fraction, "mass": mass})
        self.next_sample = start_sample + len(audio)
        return records


class SpectralPredictor:
    """Compare ordered context with one-frame, marginal and persistence forecasts."""

    def __init__(self):
        self.recent = deque(maxlen=2)
        self.history = deque(maxlen=MEMORY)

    def forecast(self):
        if len(self.recent) < 2 or not self.history:
            return None
        anchor = float(self.recent[-1] @ LOG2_HZ)
        context = np.sqrt([relative_mass(p, anchor) for p in self.recent])
        past_context = np.array([c for c, _ in self.history])
        targets = np.array([target for _, target in self.history])
        marginal = np.mean(targets, axis=0)
        result = {"anchor_log2_hz": anchor, "context": context,
                  "history_events": len(self.history), "support": {}}
        result["marginal"] = marginal
        result["persistence"] = context[-1] ** 2
        for name, count in (("one_frame", 1), ("ordered", 2)):
            distance = np.sum((past_context[:, -count:] - context[-count:]) ** 2,
                              axis=(1, 2)) / (2 * count)
            kernels = np.exp(-distance / (2 * KERNEL_WIDTH ** 2))
            support = float(np.sum(kernels))
            strength = support / (support + 4)
            conditional = np.sum(kernels[:, None] * targets, axis=0) / max(support, 1e-30)
            result[name] = (1 - strength) * marginal + strength * conditional
            result["support"][name] = support
        for name in ("marginal", "persistence", "one_frame", "ordered"):
            result[name] = .98 * result[name] + .02 / len(RELATIVE_OCT)
        return result

    def step(self, mass):
        if mass is not None:
            mass = np.asarray(mass, dtype=float)
            if (mass.shape != LOG2_HZ.shape or not np.all(np.isfinite(mass))
                    or np.any(mass < 0) or not np.isclose(np.sum(mass), 1, rtol=1e-8)):
                raise ValueError("expected normalized finite Log2Space mass, or unavailable")
        prediction = self.forecast()
        if mass is None:
            self.recent.clear()
            return None
        score = None
        if len(self.recent) == 2:
            anchor = float(self.recent[-1] @ LOG2_HZ)
            target = relative_mass(mass, anchor)
            context = np.sqrt([relative_mass(p, anchor) for p in self.recent])
            if prediction is not None:
                losses = {name: float(-target @ np.log2(prediction[name]))
                          for name in ("ordered", "one_frame", "marginal", "persistence")}
                score = {"loss_bits": losses,
                         "gain_bits": {name: losses[name] - losses["ordered"]
                                       for name in ("one_frame", "marginal", "persistence")},
                         "support": prediction["support"],
                         "history_events": prediction["history_events"]}
            self.history.append((context, target))
        self.recent.append(mass.copy())
        return score


def evaluate_audio(audio, fs):
    observer, predictor = SpectralObserver(fs), SpectralPredictor()
    records = []
    for start in range(0, len(audio), 4093):
        for row in observer.process(start, audio[start:start + 4093]):
            mass = row.pop("mass")
            records.append({**row, "observed": mass is not None, "score": predictor.step(mass)})
    return records


def summarize(records, start_sec):
    selected = [r for r in records if r["available_sec"] >= start_sec]
    scored = [r["score"] for r in selected if r["score"] is not None]
    return {"windows": len(selected), "observed_windows": sum(r["observed"] for r in selected),
            "scored_windows": len(scored),
            "gain_bits": {name: float(np.mean([s["gain_bits"][name] for s in scored])) if scored else None
                          for name in ("one_frame", "marginal", "persistence")}}


def assess_manifest(manifest):
    """Report model usefulness separately from successful campaign execution."""
    expected = {(seed, duration, case) for seed in manifest["seeds"]
                for duration in (.22, .50) for case in CASES}
    actual = [(row["seed"], row["duration_sec"], row["case"]) for row in manifest["cases"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("incomplete or duplicate spectral comparison conditions")
    rows = []
    for row in manifest["samples"]:
        scores = row["evaluation"]
        gains = scores["gain_bits"]
        rows.append({"source": row["source"], "scored_windows": scores["scored_windows"],
                     "beats_all_baselines": scores["scored_windows"] > 0 and all(
                         gain is not None and gain > 0 for gain in gains.values()),
                     "gain_bits": gains})
    return {"scope": "Descriptive model comparison, not a significance test or musical acceptance",
            "normal_samples": rows,
            "normal_order_advantage_conditions": sum(row["beats_all_baselines"] for row in rows),
            "runtime_attachment": "research_only",
            "open": ["Source pitch and melody remain unobserved",
                     "Silence and out-of-band windows break this fixed-stride prediction target",
                     "Controlled repetition success does not prove utility in normal performances",
                     "No closure or author discrimination claim"]}


def run(output, seeds, sample_root=None):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    sample_paths = sorted(sample_root.glob("*/seed-*/audio.wav")) if sample_root is not None else []
    if sample_root is not None and not sample_paths:
        raise ValueError("sample root contains no expected audio files")
    output.mkdir(parents=True, exist_ok=False)
    plan = {"created_at": dt.datetime.now().astimezone().isoformat(), "seeds": seeds,
            "cases": CASES, "note_durations_sec": [.22, .50],
            "stride_sec": STRIDE_SEC, "memory_events": MEMORY, "kernel_width": KERNEL_WIDTH,
            "target": "Next relative log-frequency power mass, not source pitch or melodic contour",
            "controls": "Same audio features and history for ordered, one-frame, marginal and persistence",
            "normal_utility_check": "Positive mean ordered gain against each of all three baselines; report per condition, without significance claims",
            "scope": "Exploratory comparison; no runtime attachment or generation coupling"}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    sources = {}
    for source in (Path(__file__), Path(__file__).with_name("evaluate_phrase_expectation.py")):
        shutil.copy2(source, output / source.name)
        sources[source.name] = sha256(source)
    manifest = {"status": "running", "seeds": seeds, "plan_sha256": sha256(output / "plan.json"),
                "sources": sources, "numpy_version": np.__version__, "cases": [], "samples": []}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for seed in seeds:
        for duration in plan["note_durations_sec"]:
            for case in CASES:
                folder = output / f"seed-{seed}" / f"duration-{duration}" / case
                folder.mkdir(parents=True)
                spec = stimulus(case, seed)
                for event in spec["events"]:
                    event["duration_sec"] = duration
                write_wav(folder / "audio.wav", synthesize(spec))
                with wave.open(str(folder / "audio.wav")) as wav:
                    audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float) / 32768
                    fs = wav.getframerate()
                records = evaluate_audio(audio, fs)
                (folder / "truth.json").write_text(json.dumps(spec, indent=2) + "\n")
                (folder / "observations.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
                row = {"seed": seed, "case": case, "duration_sec": duration,
                       "directory": str(folder.relative_to(output)),
                       "evaluation": summarize(records, spec["evaluation_start_sec"]),
                       "sha256": {p.name: sha256(p) for p in folder.iterdir()}}
                manifest["cases"].append(row)
                print(json.dumps({k: v for k, v in row.items() if k not in ("sha256", "directory")}), flush=True)
    if sample_root is not None:
        for path in sample_paths:
            with wave.open(str(path)) as wav:
                if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                    raise ValueError("expected mono PCM16 sample")
                audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float) / 32768
                fs = wav.getframerate()
            records = evaluate_audio(audio, fs)
            folder = output / "samples" / path.parent.parent.name / path.parent.name
            folder.mkdir(parents=True)
            (folder / "observations.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
            row = {"source": str(path), "audio_sha256": sha256(path),
                   "directory": str(folder.relative_to(output)), "evaluation": summarize(records, 10),
                   "sha256": sha256(folder / "observations.json")}
            manifest["samples"].append(row)
            print(json.dumps(row), flush=True)
    assessment = assess_manifest(manifest)
    (output / "assessment.json").write_text(json.dumps(assessment, indent=2, allow_nan=False) + "\n")
    manifest["assessment_sha256"] = sha256(output / "assessment.json")
    manifest["status"] = "complete"
    manifest["status_scope"] = "Execution complete; not musical or predictive acceptance"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--sample-root", type=Path)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    run(args.output, args.seeds, args.sample_root)
