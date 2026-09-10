#!/usr/bin/env python3
"""Compare causal local envelope histories without assigning acoustic sources."""

import argparse
from collections import deque
import datetime as dt
import json
import math
from pathlib import Path
import shutil
import wave

import numpy as np

from evaluate_component_relations import ComponentRelations, CENTERS_LOG2, control_audio as relation_audio
from evaluate_frequency_events import FrequencyEventObserver
from evaluate_interval_order import interval_audio
from evaluate_phrase_expectation import FS, sha256, synthesize, write_wav


STEP_SEC = .1
LAGS = 20
HORIZON = 3
MEMORY = 128
MIN_TRAIN = 32
RIDGE = .05
LEARNED = ("local", "unordered", "one_frame")
MODELS = LEARNED + ("marginal", "persistence")
CASES = ("synchronous", "alternating", "independent", "two_pairs", "weaker_partner",
         "steady_pair", "silence", "noise", "common_gain", "interval_repeat",
         "interval_shuffled", "interval_quieter", "interval_timbre", "interval_transposed")


class LocalPredictor:
    """Local regressions with bounded completed and pending examples."""

    def __init__(self, lags=LAGS, mode="channel"):
        if lags not in (2, 20) or mode not in ("channel", "neighborhood") or mode == "neighborhood" and lags != 2:
            raise ValueError("expected two/twenty channel lags or two neighborhood lags")
        self.lags = lags
        self.mode = mode
        self.last_time = None
        self.after_gap = False
        self.recent = deque(maxlen=lags)
        self.pending = deque()
        self.training = deque()
        self.target_sum = np.zeros(len(CENTERS_LOG2))
        self.sums = {}
        width = 5 if mode == "neighborhood" else 1
        for name, dimensions in (("local", lags * width), ("unordered", lags * width), ("one_frame", width)):
            self.sums[name] = {"x": np.zeros((len(CENTERS_LOG2), dimensions)),
                               "xx": np.zeros((len(CENTERS_LOG2), dimensions, dimensions)),
                               "xy": np.zeros((len(CENTERS_LOG2), dimensions))}

    def update(self, features, target, sign):
        self.target_sum += sign * target
        for name, x in features.items():
            self.sums[name]["x"] += sign * x
            self.sums[name]["xx"] += sign * x[:, :, None] * x[:, None, :]
            self.sums[name]["xy"] += sign * x * target[:, None]

    def forecast(self, features, current):
        n = len(self.training)
        if n < MIN_TRAIN:
            return None
        mean_target = self.target_sum / n
        forecasts = {"marginal": np.maximum(0, mean_target), "persistence": current.copy()}
        for name, x in features.items():
            sums = self.sums[name]
            count = n
            target_mean = mean_target
            if self.mode == "neighborhood":
                sums = {key: np.sum(value, axis=0, keepdims=True) for key, value in sums.items()}
                count *= len(CENTERS_LOG2)
                target_mean = np.asarray([np.mean(mean_target)])
            mean = sums["x"] / count
            covariance = sums["xx"] / count - mean[:, :, None] * mean[:, None, :]
            covariance = .5 * (covariance + np.swapaxes(covariance, 1, 2))
            diagonal = np.arange(x.shape[1])
            penalty = RIDGE * np.maximum(0, np.trace(covariance, axis1=1, axis2=2)) / x.shape[1] + 1e-12
            covariance[:, diagonal, diagonal] += penalty[:, None]
            cross = sums["xy"] / count - mean * target_mean[:, None]
            coefficients = np.linalg.solve(covariance, cross[:, :, None])[:, :, 0]
            forecasts[name] = np.maximum(0, target_mean + np.sum((x - mean) * coefficients, axis=1))
        return forecasts

    def process(self, time, amplitude):
        if not math.isfinite(time) or time < 0 or self.last_time is not None and time <= self.last_time:
            raise ValueError("expected increasing finite observation times")
        if amplitude is not None:
            amplitude = np.asarray(amplitude, dtype=float)
            if (amplitude.shape != CENTERS_LOG2.shape or not np.all(np.isfinite(amplitude))
                    or np.any(amplitude < 0)):
                raise ValueError("expected a finite nonnegative log2-channel amplitude vector")
        gap = (amplitude is None or self.last_time is not None and not self.after_gap
               and not math.isclose(time - self.last_time, STEP_SEC, abs_tol=1e-7))
        output = []
        if gap:
            output.append({"kind": "input_gap", "available_sec": time,
                           "censored_forecasts": sum(p["prediction"] is not None for p in self.pending)})
            self.__init__(self.lags, self.mode)
        self.last_time = time
        self.after_gap = amplitude is None
        if amplitude is None:
            return output
        if self.pending and time >= self.pending[0]["deadline_sec"] - 1e-7:
            completed = self.pending.popleft()
            assert math.isclose(time, completed["deadline_sec"], abs_tol=1e-7)
            prediction = completed["prediction"]
            if prediction is not None:
                output.append({"kind": "score", "available_sec": time,
                               "origin_sec": completed["origin_sec"],
                               "actual": amplitude.tolist(),
                               "squared_error": {name: float(np.sum((value - amplitude) ** 2))
                                                 for name, value in prediction.items()},
                               "target_energy": float(np.sum(amplitude ** 2))})
            if len(self.training) == MEMORY:
                old_features, old_target = self.training.popleft()
                self.update(old_features, old_target, -1)
            target = amplitude.copy()
            self.training.append((completed["features"], target))
            self.update(completed["features"], target, 1)
        self.recent.append(amplitude.copy())
        if len(self.recent) == self.lags:
            context = np.stack(self.recent, axis=1)
            if self.mode == "neighborhood":
                padded = np.pad(context, ((2, 2), (0, 0)))
                context = np.stack([padded[offset:offset + len(CENTERS_LOG2)] for offset in range(5)], axis=2)
            features = {"local": context.reshape(len(CENTERS_LOG2), -1),
                        "unordered": np.sort(context, axis=1).reshape(len(CENTERS_LOG2), -1),
                        "one_frame": context[:, -1:].reshape(len(CENTERS_LOG2), -1).copy()}
            prediction = self.forecast(features, amplitude)
            pending = {"origin_sec": time, "deadline_sec": time + HORIZON * STEP_SEC,
                       "features": features, "prediction": prediction}
            self.pending.append(pending)
            if prediction is not None:
                output.append({"kind": "forecast", "available_sec": time,
                               "deadline_sec": pending["deadline_sec"], "training_examples": len(self.training),
                               "prediction": {name: value.tolist() for name, value in prediction.items()}})
        assert len(self.pending) <= HORIZON
        return output


def evaluate(audio, fs, lags=LAGS, mode="channel"):
    front, relations, model = FrequencyEventObserver(fs), ComponentRelations(), LocalPredictor(lags, mode)
    records = []
    for start in range(0, len(audio), 4093):
        for row in front.process(start, audio[start:start + 4093]):
            result = relations.process(row)
            if result is None:
                continue
            amplitude = None if result["kind"] == "input_gap" else relations.fast
            records.extend(model.process(result["available_sec"], amplitude))
    return {"records": records, "right_censored_forecasts": sum(p["prediction"] is not None for p in model.pending)}


def control_audio(case, seed):
    if case not in CASES:
        raise ValueError("unknown local-prediction control")
    if case.startswith("interval_"):
        source_case = case.removeprefix("interval_")
        audio, truth = interval_audio("repeat" if source_case == "transposed" else source_case, seed)
        if source_case == "transposed":
            truth = {**truth, "source_frequency_hz": truth["source_frequency_hz"] * 2,
                     "events": [{**event, "frequency_hz": event["frequency_hz"] * 2}
                                        for event in truth["events"]]}
            audio = synthesize(truth)
        return audio, {**truth, "case": case}
    audio, truth = relation_audio(case, seed)
    # Extend controlled processes, not a repeated PCM block with artificial seams.
    if case in ("silence", "noise"):
        audio = np.zeros(30 * FS) if case == "silence" else np.random.default_rng(seed).normal(0, .07, 30 * FS)
        truth = {**truth, "duration_sec": 30.0}
    else:
        events = truth["events"]
        if case in ("steady_pair", "common_gain"):
            events = [{**e, "duration_sec": 29.3 - e["onset_sec"]} for e in events]
        else:
            sources = {(e["frequency_hz"], e["amplitude"]): e for e in events}
            expanded = []
            for (frequency, amplitude), event in sources.items():
                starts = sorted(e["onset_sec"] for e in events if e["frequency_hz"] == frequency)
                period = starts[1] - starts[0]
                expanded.extend({**event, "onset_sec": float(time)} for time in np.arange(starts[0], 29.3, period))
            events = expanded
        truth = {**truth, "events": events, "duration_sec": 30.0}
        audio = synthesize(truth)
        if case == "common_gain":
            audio *= .55 + .35 * np.sin(math.tau * np.arange(len(audio)) / FS / truth["period_base_sec"])
    return audio, truth


def summarize(result, duration):
    scores = [r for r in result["records"] if r["kind"] == "score"
              and r["origin_sec"] >= 10 and r["available_sec"] <= duration - 2]
    errors = {name: sum(r["squared_error"][name] for r in scores) for name in MODELS}
    energy = sum(r["target_energy"] for r in scores)
    gain = {name: 1 - errors["local"] / errors[name] if errors[name] > 1e-12 else None
            for name in MODELS if name != "local"}
    return {"scored_frames": len(scores), "target_energy_sum": energy, "squared_error_sum": errors,
            "error_relative_to_signal_energy": {name: error / energy if energy > 1e-12 else None
                                                 for name, error in errors.items()},
            "relative_error_reduction": gain,
            "normal_screen_passed": len(scores) >= 100 and energy > 1e-6
            and errors["persistence"] / energy >= 1e-6
            and all(value is not None and value >= .05 for value in gain.values()),
            "right_censored_forecasts": result["right_censored_forecasts"]}


def run(output, seeds, sample_root=None, lags=LAGS, mode="channel"):
    LocalPredictor(lags, mode)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("expected nonempty unique seeds")
    output.mkdir(parents=True, exist_ok=False)
    plan = {"created_at": dt.datetime.now().astimezone().isoformat(), "seeds": seeds, "cases": CASES,
            "step_sec": STEP_SEC, "lags": lags, "mode": mode, "horizon_steps": HORIZON, "memory": MEMORY,
            "minimum_training_examples": MIN_TRAIN, "ridge": RIDGE, "models": MODELS,
            "target": "Future 40 ms filtered amplitude on 113 fixed log2-frequency channels",
            "scope": "Research only. No source identity, phrase, closure or generation. Sorted history retains values but removes order; fit dimension equals local ordered history. Neighborhood mode uses two frames and five adjacent bins, pooling completed training examples over all frequency bins. All learned temporal baselines receive the same neighborhood and pooling.",
            "normal_screen": "Origin >=10 sec, deadline <=duration-2 sec, >=100 frames, >=5 percent total squared-error reduction against each baseline, every seed of a sample. Persistence error divided by signal energy must be >=1e-6 to exclude near-constant numerical variation. Not a significance test."}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    sources = {}
    for name in (Path(__file__).name, "evaluate_component_relations.py", "evaluate_frequency_events.py",
                 "evaluate_interval_order.py", "evaluate_phrase_expectation.py"):
        source = Path(__file__).with_name(name)
        shutil.copy2(source, output / name)
        sources[name] = sha256(source)
    inputs = []
    for seed in seeds:
        for case in CASES:
            folder = output / f"seed-{seed}-{case}"
            folder.mkdir()
            audio, truth = control_audio(case, seed)
            write_wav(folder / "audio.wav", audio)
            (folder / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
            inputs.append((folder, case, seed, "controlled", None))
    if sample_root is not None:
        samples = sorted(sample_root.glob("*/seed-*/audio.wav"))
        if not samples:
            raise ValueError("sample root contains no WAV inputs")
        for source in samples:
            folder = output / f"normal-{source.parent.parent.name}-{source.parent.name}"
            folder.mkdir()
            shutil.copy2(source, folder / "audio.wav")
            inputs.append((folder, source.parent.parent.name, int(source.parent.name.removeprefix("seed-")), "normal", source))
    manifest = {"status": "running", "sources": sources, "plan_sha256": sha256(output / "plan.json"), "rows": []}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for folder, case, seed, group, source in inputs:
        with wave.open(str(folder / "audio.wav")) as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError("expected mono PCM16")
            fs = wav.getframerate()
            signal = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float) / 32768
        result = evaluate(signal, fs, lags, mode)
        (folder / "predictions.json").write_text(json.dumps(result, allow_nan=False) + "\n")
        summary = summarize(result, len(signal) / fs)
        manifest["rows"].append({"directory": folder.name, "case": case, "seed": seed, "group": group,
                                 "summary": summary, "original_audio": str(source) if source else None,
                                 "sha256": {p.name: sha256(p) for p in folder.iterdir()}})
        print(json.dumps({"directory": folder.name, "summary": summary}), flush=True)
    manifest["status"] = "measured"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--sample-root", type=Path)
    parser.add_argument("--lags", type=int, choices=(2, 20), default=LAGS)
    parser.add_argument("--mode", choices=("channel", "neighborhood"), default="channel")
    args = parser.parse_args()
    run(args.output, args.seeds, args.sample_root, args.lags, args.mode)
