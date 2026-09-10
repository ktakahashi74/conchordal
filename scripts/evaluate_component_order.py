#!/usr/bin/env python3
"""Compare causal timing/frequency predictions of unordered spectral-rise groups."""

import argparse
from collections import deque
import datetime as dt
import json
import math
from pathlib import Path
import shutil
import wave

import numpy as np

from evaluate_frequency_events import FrequencyEventObserver, STRIDE_SEC
from evaluate_interval_order import interval_audio
from evaluate_phrase_expectation import FS, sha256, synthesize, write_wav


GROUP_SEC = .03
MEMORY = 128
CONTEXT_WIDTH_TIME = .10
CONTEXT_WIDTH_FREQUENCY = .10
TARGET_WIDTH_TIME = .07
TARGET_WIDTH_FREQUENCY = .04
MODELS = ("ordered", "time_only", "one_group", "marginal", "persistence")
CASES = ("frequency_repeat", "frequency_reordered", "simultaneous_repeat",
         "simultaneous_reordered", "interval_repeat", "interval_reordered",
         "frequency_transposed", "simultaneous_transposed", "frequency_quieter",
         "frequency_timbre", "frequency_random")


def normal_density(x, mean, width):
    return np.exp(-.5 * ((np.asarray(x) - mean) / width) ** 2) / (width * math.sqrt(math.tau))


def densities(prediction, interval_log2, frequencies_log2):
    """Normalized joint density in log2 seconds and log2 Hz, plus its time marginal."""
    x = np.asarray(frequencies_log2)
    prior_time = float(normal_density(interval_log2, -1, 2))
    prior_frequency = normal_density(x, 9, 3)
    time_kernels = np.asarray([normal_density(interval_log2, t["interval_log2_sec"], TARGET_WIDTH_TIME)
                               for t in prediction["targets"]])
    mark_kernels = np.asarray([np.sum(normal_density(
        x[:, None], np.asarray(t["frequencies_log2"]), TARGET_WIDTH_FREQUENCY)
        * np.asarray(t["mass"]), axis=1) for t in prediction["targets"]])
    result = {}
    for name, weights in prediction["weights"].items():
        weighted = np.asarray(weights) * time_kernels
        result[name] = {"time": .05 * prior_time + .95 * float(np.sum(weighted)),
                        "joint": .05 * prior_time * prior_frequency + .95 * np.dot(weighted, mark_kernels)}
    last = prediction["last_group"]
    time = float(normal_density(interval_log2, last["interval_log2_sec"], TARGET_WIDTH_TIME))
    mark = np.sum(normal_density(x[:, None], np.asarray(last["frequencies_log2"]), TARGET_WIDTH_FREQUENCY)
                  * np.asarray(last["mass"]), axis=1)
    result["persistence"] = {"time": .05 * prior_time + .95 * time,
                             "joint": .05 * prior_time * prior_frequency + .95 * time * mark}
    return result


class ComponentOrderPredictor:
    """Keep simultaneous rises unordered; never interpret silence as closure."""

    def __init__(self, coordinates="absolute", context_groups=2):
        if coordinates not in ("absolute", "relative") or context_groups not in (2, 3):
            raise ValueError("expected absolute/relative coordinates and two/three context groups")
        self.coordinates = coordinates
        self.history = deque(maxlen=MEMORY)
        self.context = deque(maxlen=context_groups)
        self.last_frame = None
        self.after_input_gap = False
        self.last_start = None
        self.group = None
        self.pending = None

    def forecast(self, available_sec):
        if len(self.context) < self.context.maxlen or not self.history:
            return None
        time = np.asarray([g["interval_log2_sec"] for g in self.context])
        frequency = np.asarray([g["quantiles_log2"] for g in self.context])
        past_time = np.asarray([[g["interval_log2_sec"] for g in row[0]] for row in self.history])
        past_frequency = np.asarray([[g["quantiles_log2"] for g in row[0]] for row in self.history])
        targets = [row[1] for row in self.history]
        if self.coordinates == "relative":
            # The last observed distribution supplies a reference, not a source pitch.
            origin = float(np.mean(frequency[-1]))
            past_origins = np.mean(past_frequency[:, -1], axis=1)
            frequency -= origin
            past_frequency -= past_origins[:, None, None]
            targets = [{**target,
                        "frequencies_log2": (np.asarray(target["frequencies_log2"]) + origin - past_origin).tolist(),
                        "quantiles_log2": (np.asarray(target["quantiles_log2"]) + origin - past_origin).tolist()}
                       for target, past_origin in zip(targets, past_origins)]
        marginal = np.full(len(self.history), 1 / len(self.history))
        weights, support = {"marginal": marginal.tolist()}, {}
        for name, count in (("ordered", self.context.maxlen), ("time_only", self.context.maxlen), ("one_group", 1)):
            distance = np.mean(((past_time[:, -count:] - time[-count:]) / CONTEXT_WIDTH_TIME) ** 2, axis=1)
            if name != "time_only":
                distance += np.mean(((past_frequency[:, -count:] - frequency[-count:])
                                     / CONTEXT_WIDTH_FREQUENCY) ** 2, axis=(1, 2))
            kernels = np.exp(-.5 * distance)
            support[name] = float(np.sum(kernels))
            strength = support[name] / (support[name] + 2)
            weights[name] = ((1 - strength) * marginal
                             + strength * kernels / max(support[name], 1e-300)).tolist()
        return {"issued_sec": available_sec, "targets": targets,
                "weights": weights, "support": support, "last_group": self.context[-1]}

    def process(self, row):
        time = row["available_sec"]
        if (not math.isfinite(time) or time < 0 or self.last_frame is not None and time <= self.last_frame
                or row["kind"] not in ("components", "input_gap")):
            raise ValueError("expected ordered finite component frames or an input gap")
        components = row.get("components", [])
        for c in components:
            if (not isinstance(c["rise"], bool) or not math.isfinite(c["frequency_log2"])
                    or not math.isfinite(c["power"]) or c["power"] <= 0):
                raise ValueError("expected finite component frequencies and positive powers")
        resumed = self.after_input_gap and row.get("initial_observation", False)
        gap = (row["kind"] == "input_gap" or not resumed and self.last_frame is not None
               and not math.isclose(time - self.last_frame, STRIDE_SEC, abs_tol=1e-7))
        self.last_frame = time
        self.after_input_gap = row["kind"] == "input_gap"
        if gap:
            result = {"kind": "gap", "available_sec": time,
                      "censored_group": self.group,
                      "censored_forecast_issued_sec": self.pending["issued_sec"] if self.pending else None}
            self.group = self.pending = self.last_start = None
            self.context.clear()
            # The first frame after missing input cannot establish a rise.
            return result
        rises = [c for c in components if c["rise"]]
        if rises:
            if self.group is None:
                self.group = {"start_sec": time, "components": []}
            self.group["components"].extend(rises)
        if self.group is None or time < self.group["start_sec"] + GROUP_SEC - 1e-7:
            return None
        group = self.group
        self.group = None
        # Frequency order canonicalizes a set; it does not assert an onset order.
        components = sorted(group["components"], key=lambda c: (c["frequency_log2"], c["power"]))
        frequencies = np.asarray([c["frequency_log2"] for c in components])
        mass = np.asarray([c["power"] for c in components])
        mass /= np.sum(mass)
        quantiles = frequencies[np.searchsorted(np.cumsum(mass), (np.arange(8) + .5) / 8)]
        target = {"frequencies_log2": frequencies.tolist(), "mass": mass.tolist(),
                  "quantiles_log2": quantiles.tolist()}
        interval = None if self.last_start is None else group["start_sec"] - self.last_start
        score = None
        if interval is not None:
            target["interval_log2_sec"] = math.log2(interval)
            if self.pending is not None:
                assert self.pending["issued_sec"] < group["start_sec"]
                values = densities(self.pending, target["interval_log2_sec"], target["frequencies_log2"])
                losses = {}
                for name, value in values.items():
                    time_loss = -math.log2(max(value["time"], 1e-300))
                    joint_loss = -float(np.dot(target["mass"], np.log2(np.maximum(value["joint"], 1e-300))))
                    losses[name] = {"joint": joint_loss, "time": time_loss, "frequency_given_time": joint_loss - time_loss}
                score = {"forecast_issued_sec": self.pending["issued_sec"],
                         "training_examples": len(self.pending["targets"]), "support": self.pending["support"],
                         "loss_bits": losses,
                         "gain_bits": {name: {part: losses[name][part] - losses["ordered"][part]
                                              for part in losses[name]} for name in MODELS[1:]}}
            if len(self.context) == self.context.maxlen:
                self.history.append((tuple(self.context), target))
            self.context.append(target)
        self.last_start = group["start_sec"]
        self.pending = self.forecast(time)
        return {"kind": "group", "available_sec": time, "start_sec": group["start_sec"],
                "interval_sec": interval, "component_count": len(group["components"]),
                "distribution": target, "score": score,
                "next_forecast_issued_sec": self.pending["issued_sec"] if self.pending else None}


def evaluate(rows, coordinates="absolute", context_groups=2):
    model, records = ComponentOrderPredictor(coordinates, context_groups), []
    for row in rows:
        result = model.process(row)
        if result is not None:
            records.append(result)
    return {"records": records, "right_censored_group": model.group,
            "right_censored_forecast_issued_sec": model.pending["issued_sec"] if model.pending else None}


def control_audio(case, seed):
    if case not in CASES:
        raise ValueError("unknown component-order control")
    if case.startswith("interval_"):
        return interval_audio(case.removeprefix("interval_"), seed)
    rng = np.random.default_rng(seed)
    root = float(rng.uniform(270, 350))
    interval = round(rng.uniform(.27, .33) * FS) / FS
    events = []
    for i in range(65):
        pattern = (0, 1, 0, 1) if "reordered" in case and i >= 32 else (0, 0, 1, 1)
        degree = int(rng.integers(0, 2)) if case == "frequency_random" else pattern[i % 4]
        transpose = .14 * (i // 4) if case.endswith("_transposed") else 0
        frequencies = [root * 2 ** (.65 * degree + transpose)]
        if case.startswith("simultaneous_"):
            frequencies.append(root * 2.9 * 2 ** transpose)
        for frequency in frequencies:
            events.append({"onset_sec": .6 + i * interval, "duration_sec": .12,
                           "frequency_hz": frequency,
                           "amplitude": .25 / math.sqrt(len(frequencies)) * (.5 if case == "frequency_quieter" and i >= 32 else 1),
                           "timbre": "harmonic" if case == "frequency_timbre" and i >= 32 else "sine"})
    truth = {"case": case, "seed": seed, "events": events, "duration_sec": .6 + 64 * interval + .5,
             "split_sec": .6 + 32 * interval, "expected_groups": 65}
    return synthesize(truth), truth


def summarize(result, truth=None):
    groups = [r for r in result["records"] if r["kind"] == "group"]
    windows = {"after_10_sec": (10, math.inf)} if truth is None else {
        "trained_before_change": (truth["split_sec"] - 6, truth["split_sec"]),
        "first_three_seconds_after_change": (truth["split_sec"], truth["split_sec"] + 3),
        "later": (truth["split_sec"] + 3, truth["duration_sec"])}
    summary = {"groups": len(groups), "gaps": sum(r["kind"] == "gap" for r in result["records"]),
               "maximum_group_components": max((g["component_count"] for g in groups), default=0), "windows": {}}
    for name, (start, end) in windows.items():
        scores = [r["score"] for r in groups if start <= r["start_sec"] < end and r["score"] is not None]
        summary["windows"][name] = {"scored_groups": len(scores), "gain_bits": {
            model: {part: float(np.mean([s["gain_bits"][model][part] for s in scores])) if scores else None
                    for part in ("joint", "time", "frequency_given_time")} for model in MODELS[1:]}}
    if truth is not None:
        expected = sorted({e["onset_sec"] for e in truth["events"]})
        latencies = [g["start_sec"] - t for g, t in zip(groups, expected)]
        summary.update(expected_groups=len(expected), observation_passed=len(groups) == len(expected)
                       and all(0 <= latency <= .15 for latency in latencies))
    return summary


def run(output, seeds, sample_root=None, coordinates=("absolute",), context_groups=(2,)):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("expected unique nonempty seeds")
    if (not coordinates or len(set(coordinates)) != len(coordinates)
            or not context_groups or len(set(context_groups)) != len(context_groups)):
        raise ValueError("expected unique nonempty coordinates and context group counts")
    for coord in coordinates:
        for count in context_groups:
            ComponentOrderPredictor(coord, count)
    output.mkdir(parents=True, exist_ok=False)
    plan = {"created_at": dt.datetime.now().astimezone().isoformat(), "seeds": seeds, "cases": CASES,
            "group_sec": GROUP_SEC, "memory": MEMORY, "models": MODELS,
            "coordinates": coordinates, "context_groups": context_groups,
            "relative_reference": "Mean of eight log-frequency quantiles in the last observed group. Subtract that reference from context and align every historical target, including all learned baselines, to the current reference. Persistence and the common prior retain identical absolute coordinates.",
            "context_widths": [CONTEXT_WIDTH_TIME, CONTEXT_WIDTH_FREQUENCY],
            "target_widths": [TARGET_WIDTH_TIME, TARGET_WIDTH_FREQUENCY],
            "evaluation": "Check controlled groups within 150 ms. Compare all models on identical group observations; report time and conditional-frequency losses separately. No forecast after EOF, source IDs, phrase or closure labels.",
            "normal_candidate_gate": "For each sample, all three seeds must have at least 16 scored groups after 10 seconds and positive mean joint gain against all four controls. Engineering screen, not musical acceptance or a statistical significance test."}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    sources = {}
    for name in (Path(__file__).name, "evaluate_frequency_events.py", "evaluate_interval_order.py", "evaluate_phrase_expectation.py"):
        source = Path(__file__).with_name(name)
        shutil.copy2(source, output / name)
        sources[name] = sha256(source)
    inputs = []
    for seed in seeds:
        for case in CASES:
            folder = output / f"seed-{seed}-{case}"; folder.mkdir()
            signal, truth = control_audio(case, seed)
            write_wav(folder / "audio.wav", signal)
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
        observer = FrequencyEventObserver(fs)
        observed = observer.process(0, signal)
        variants = []
        for coord in coordinates:
            for count in context_groups:
                result = evaluate(observed, coord, count)
                result_file = f"predictions-{coord}-{count}.json"
                (folder / result_file).write_text(json.dumps(result, allow_nan=False) + "\n")
                variants.append({"coordinates": coord, "context_groups": count, "predictions": result_file,
                                 "summary": summarize(result, truth)})
        row = {"directory": folder.name, "case": case, "seed": seed, "group": "controlled" if truth else "normal",
               "variants": variants, "original_audio": str(source) if source else None,
               "sha256": {p.name: sha256(p) for p in folder.iterdir()}}
        manifest["rows"].append(row)
        print(json.dumps({"directory": folder.name, "variants": variants}), flush=True)
    manifest["status"] = "observed" if all(v["summary"].get("observation_passed", True)
        for row in manifest["rows"] for v in row["variants"]) else "failed_observations"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["status"] != "observed":
        raise ValueError("component-order controls failed; preserve this run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--sample-root", type=Path)
    parser.add_argument("--coordinates", choices=("absolute", "relative"), nargs="+", default=["absolute"])
    parser.add_argument("--context-groups", type=int, choices=(2, 3), nargs="+", default=[2])
    args = parser.parse_args()
    run(args.output, args.seeds, args.sample_root, args.coordinates, args.context_groups)
