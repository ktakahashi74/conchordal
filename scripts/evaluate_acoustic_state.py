#!/usr/bin/env python3
"""Compare sustained changes of acoustic distributions without naming phrases."""

import argparse
from collections import deque
import datetime as dt
from itertools import combinations
import json
import math
from pathlib import Path
import shutil
import wave

import numpy as np

from evaluate_phrase_expectation import FS, sha256, write_wav
from evaluate_spectral_change import control_audio
from evaluate_spectral_expectation import LOG2_HZ, STRIDE_SEC, SpectralObserver


STATE_FRAMES = 20
KERNEL_WIDTH = .25
CHANGE_DISTANCE = .25
RELEASE_DISTANCE = .125
CHANGE_CONFIRM_FRAMES = 3
STABILITY_FRAMES = 6
BLOCK_FRAMES = 4
ENTER_QUANTILE = .95
LEAVE_QUANTILE = .90
TRANSPORT_ENTER_OCT = .10
TRANSPORT_LEAVE_OCT = .05
CHANNELS = ("distribution", "mean_spectrum", "frequency_transport")
CASES = ("steady", "gain_only", "brief_excursion", "texture_change", "register_change",
         "return", "order_only", "noise", "silence", "moving_texture", "moving_excursion")


class AcousticStateObserver:
    """Compare adjacent empirical distributions; discard history across missing evidence."""

    def __init__(self):
        self.roots = deque(maxlen=2 * STATE_FRAMES)
        self.kernel = np.zeros((2 * STATE_FRAMES, 2 * STATE_FRAMES))
        self.gates = {name: {"high": 0, "low": 0, "state": "unestablished"}
                      for name in (*CHANNELS, "combined")}
        self.last_time = None
        block_count = 2 * STATE_FRAMES // BLOCK_FRAMES
        self.partitions = np.zeros((math.comb(block_count - 1, block_count // 2 - 1), block_count))
        for row, selection in enumerate(combinations(range(1, block_count), block_count // 2 - 1)):
            self.partitions[row, (0, *selection)] = 1

    def process(self, time_sec, mass):
        if not math.isfinite(time_sec) or time_sec < 0:
            raise ValueError("expected a nonnegative finite time")
        if self.last_time is not None and time_sec <= self.last_time:
            raise ValueError("observations must arrive in strict time order")
        if mass is not None:
            mass = np.asarray(mass, dtype=float)
            if (mass.shape != LOG2_HZ.shape or not np.all(np.isfinite(mass))
                    or np.any(mass < 0) or not np.isclose(np.sum(mass), 1, rtol=1e-8)):
                raise ValueError("expected normalized finite Log2Space mass, or unavailable")
        gap = (self.last_time is not None
               and not math.isclose(time_sec - self.last_time, STRIDE_SEC, abs_tol=1e-7))
        censored = []
        if gap or mass is None:
            censored = [name for name, gate in self.gates.items() if gate["state"] == "changing"]
            self.roots.clear()
            self.kernel.fill(0)
            for gate in self.gates.values():
                gate.update(high=0, low=0, state="unestablished")
        self.last_time = time_sec
        output = {"available_sec": time_sec, "input_gap": gap, "observed": mass is not None,
                  "ready": False, "distance": None, "candidates": [], "settled": [],
                  "active_difference": None, "state": None, "censored_difference": censored, "thresholds": None}
        if mass is None:
            return output
        if len(self.roots) == self.roots.maxlen:
            self.kernel[:-1, :-1] = self.kernel[1:, 1:]
        self.roots.append(np.sqrt(mass))
        roots = np.asarray(self.roots)
        count = len(roots)
        distances_squared = np.maximum(0, 1 - np.sum(roots * roots[-1], axis=1))
        values = np.exp(-distances_squared / (2 * KERNEL_WIDTH ** 2))
        self.kernel[count - 1, :count] = values
        self.kernel[:count, count - 1] = values
        if count != self.roots.maxlen:
            return output
        n = STATE_FRAMES
        # Empirical MMD includes diagonal terms; no iid significance claim is made.
        mmd_squared = (np.mean(self.kernel[:n, :n]) + np.mean(self.kernel[n:, n:])
                       - 2 * np.mean(self.kernel[:n, n:]))
        average_old, average_new = np.mean(roots[:n] ** 2, axis=0), np.mean(roots[n:] ** 2, axis=0)
        scores = {"distribution": math.sqrt(max(0, float(mmd_squared)) / 2),
                  "mean_spectrum": float(np.linalg.norm(np.sqrt(average_new) - np.sqrt(average_old))
                                         / math.sqrt(2)),
                  "frequency_transport": float(np.sum(np.abs(np.cumsum(average_new - average_old)))
                                               * (LOG2_HZ[1] - LOG2_HZ[0]))}
        # Preserve short within-block trajectories; compare every balanced allocation.
        block_count = count // BLOCK_FRAMES
        block_kernel = self.kernel.reshape(block_count, BLOCK_FRAMES, block_count, BLOCK_FRAMES).mean(axis=(1, 3))
        weights = (2 * self.partitions - 1) / (block_count // 2)
        permuted_mmd = np.einsum("pi,ij,pj->p", weights, block_kernel, weights)
        block_mass = (roots ** 2).reshape(block_count, BLOCK_FRAMES, -1).mean(axis=1)
        old = np.einsum("pi,ij->pj", self.partitions, block_mass) / (block_count // 2)
        new = np.einsum("pi,ij->pj", 1 - self.partitions, block_mass) / (block_count // 2)
        surrogates = {"distribution": np.sqrt(np.maximum(0, permuted_mmd) / 2),
                      "mean_spectrum": np.linalg.norm(np.sqrt(old) - np.sqrt(new), axis=1) / math.sqrt(2),
                      "frequency_transport": np.sum(np.abs(np.cumsum(new - old, axis=1)), axis=1)
                      * (LOG2_HZ[1] - LOG2_HZ[0])}
        thresholds = {name: {"enter": max(TRANSPORT_ENTER_OCT if name == "frequency_transport" else CHANGE_DISTANCE,
                                          float(np.quantile(values, ENTER_QUANTILE))),
                             "leave": max(TRANSPORT_LEAVE_OCT if name == "frequency_transport" else RELEASE_DISTANCE,
                                          float(np.quantile(values, LEAVE_QUANTILE)))}
                      for name, values in surrogates.items()}
        output.update(ready=True, distance=scores,
                      thresholds=thresholds,
                      evidence_start_sec=time_sec - (2 * STATE_FRAMES - 1) * STRIDE_SEC)
        high = {name: score > thresholds[name]["enter"] + 1e-10 for name, score in scores.items()}
        low = {name: score <= thresholds[name]["leave"] for name, score in scores.items()}
        high["combined"] = high["distribution"] or high["frequency_transport"]
        low["combined"] = low["distribution"] and low["frequency_transport"]
        for name in self.gates:
            gate = self.gates[name]
            if gate["state"] == "stable":
                gate["high"] = gate["high"] + 1 if high[name] else 0
                if gate["high"] == CHANGE_CONFIRM_FRAMES:
                    output["candidates"].append(name)
                    gate.update(state="changing", high=0, low=0)
            else:
                gate["low"] = gate["low"] + 1 if low[name] else 0
                if gate["low"] == STABILITY_FRAMES:
                    if gate["state"] == "changing":
                        output["settled"].append(name)
                    gate.update(state="stable", high=0, low=0)
        output["state"] = {name: gate["state"] for name, gate in self.gates.items()}
        output["active_difference"] = {name: (None if gate["state"] == "unestablished" else gate["state"] == "changing")
                                       for name, gate in self.gates.items()}
        return output


def state_audio(case, seed):
    if case not in CASES:
        raise ValueError("unknown acoustic-state control")
    rng = np.random.default_rng(seed)
    t = np.arange(30 * FS) / FS
    base = rng.uniform(210, 310)
    frequencies = base * 2 ** (np.array([0, .23, .59, .94]) + rng.uniform(-.015, .015, 4))
    phases = rng.uniform(0, math.tau, 4)
    weights = np.zeros((len(t), 4))
    cursor, previous = 0, 0
    while cursor < len(t):
        order = rng.permutation(4)
        if case == "order_only" and cursor >= 12 * FS:
            order = np.arange(4)
        for index in order:
            count = min(round(rng.uniform(.24, .36) * FS), len(t) - cursor)
            if not count:
                break
            weights[cursor:cursor + count, index] = 1
            if previous != index:
                fade = min(round(.025 * FS), count)
                angle = np.linspace(0, math.pi / 2, fade)
                weights[cursor:cursor + fade, previous] = np.cos(angle)
                weights[cursor:cursor + fade, index] = np.sin(angle)
            cursor += count
            previous = index
    transitions = []
    blend = np.zeros(len(t))
    if case in ("texture_change", "brief_excursion", "return", "moving_texture", "moving_excursion"):
        brief = case in ("brief_excursion", "moving_excursion")
        end = 12.48 if brief else 22 if case == "return" else 31
        blend = np.minimum(np.clip((t - 12) / .06, 0, 1), np.clip((end - t) / .06, 0, 1))
        destination = .5
        if case in ("moving_texture", "moving_excursion"):
            # Retain the original event trajectory while changing which tones coexist.
            destination = weights + np.roll(weights, 1, axis=1)
            destination /= np.linalg.norm(destination, axis=1)[:, None]
        weights = (1 - blend[:, None]) * weights + blend[:, None] * destination
        weights /= np.linalg.norm(weights, axis=1)[:, None]
        if not brief:
            transitions = [12, 22] if case == "return" else [12]
    transposition = .37 * np.clip((t - 12) / .06, 0, 1) if case == "register_change" else np.zeros(len(t))
    if case == "register_change":
        transitions = [12]
    tones = np.sin(math.tau * np.cumsum(2 ** transposition[:, None] * frequencies[None, :], axis=0) / FS
                   + phases[None, :])
    audio = .16 * np.sum(weights * tones, axis=1)
    if case == "gain_only":
        audio *= .6 + .25 * np.sin(math.tau * t / 2.3)
    elif case == "noise":
        audio = control_audio("band_noise", seed)
    elif case == "silence":
        audio.fill(0)
    audio *= np.clip(t / .02, 0, 1) * np.clip((30 - t) / .02, 0, 1)
    assert np.max(np.abs(audio)) < 1
    return audio, {"case": case, "seed": seed, "frequencies_hz": frequencies.tolist(),
                   "sustained_transition_sec": transitions,
                   "brief_excursion_sec": [12, 12.48] if case in ("brief_excursion", "moving_excursion") else None,
                   "scope": "Acoustic occupancy statistics; an ordering change is not labeled a new distribution"}


def evaluate_audio(audio, fs):
    spectral, state = SpectralObserver(fs), AcousticStateObserver()
    rows = []
    for start in range(0, len(audio), 4093):
        for observation in spectral.process(start, audio[start:start + 4093]):
            rows.append(state.process(observation["available_sec"], observation["mass"]))
    return rows


def assess(rows, expected=None):
    candidates = {name: [r["available_sec"] for r in rows if name in r["candidates"]]
                  for name in (*CHANNELS, "combined")}
    result = {"candidate_sec": candidates, "ready_windows": sum(r["ready"] for r in rows),
              "max_distance": {name: max((r["distance"][name] for r in rows if r["ready"]), default=None)
                               for name in CHANNELS},
              "settled_sec": {name: [r["available_sec"] for r in rows if name in r["settled"]]
                              for name in candidates},
              "active_at_last_ready_window": next((r["active_difference"] for r in reversed(rows)
                                                   if r["ready"]), None),
              "state_at_last_ready_window": next((r["state"] for r in reversed(rows) if r["ready"]), None)}
    if expected is not None:
        actual = candidates["combined"]
        result["passed"] = (len(actual) == len(expected) and all(
            truth < found <= truth + 2 * STATE_FRAMES * STRIDE_SEC
            for truth, found in zip(expected, actual)))
    return result


def run(output, seeds, sample_root=None):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    samples = sorted(sample_root.glob("*/seed-*/audio.wav")) if sample_root is not None else []
    if sample_root is not None and not samples:
        raise ValueError("sample root contains no expected audio files")
    output.mkdir(parents=True, exist_ok=False)
    plan = {"created_at": dt.datetime.now().astimezone().isoformat(), "seeds": seeds, "cases": CASES,
            "state_frames": STATE_FRAMES, "stride_sec": STRIDE_SEC, "kernel_width": KERNEL_WIDTH,
            "change_distance": CHANGE_DISTANCE, "release_distance": RELEASE_DISTANCE,
            "change_confirm_frames": CHANGE_CONFIRM_FRAMES, "stability_frames": STABILITY_FRAMES,
            "block_frames": BLOCK_FRAMES, "enter_quantile": ENTER_QUANTILE, "leave_quantile": LEAVE_QUANTILE,
            "frequency_transport": "Sum of absolute cumulative mean-mass differences times the log2 grid step; measured in octaves",
            "transport_enter_floor_oct": TRANSPORT_ENTER_OCT, "transport_leave_floor_oct": TRANSPORT_LEAVE_OCT,
            "combined": "Distribution or frequency-transport evidence must remain high for three comparisons; both must be low for six to establish or recover stability. The mean-spectrum channel remains a comparator",
            "surrogates": "All 126 balanced allocations of ten four-frame blocks, modulo complementary assignments; use the greater of the empirical quantile and distance floor, with a 1e-10 numerical comparison margin",
            "kernel": "exp(-Hellinger_squared/(2*width_squared))",
            "distribution_distance": "sqrt(biased_empirical_MMD_squared/2)",
            "reference": "https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf (5)",
            "control_check": "Exactly one combined candidate within 4.8 seconds after each sustained transition, no extra candidates; no candidate for stationary, gain-only, brief, order-only, noise or silence",
            "moving_controls": "Change from alternating single tones to alternating neighboring pairs on the same event trajectory and frequencies. Retain equal component energy and continuous crossfades. Compare sustained and 0.48-second excursions; do not treat a static destination as the only positive control",
            "settling": "A candidate starts a difference episode; only six observed comparisons at or below the leave threshold mark settling. Missing audio censors an open episode. An unclosed candidate is not a completed state boundary",
            "initial_evidence": "Begin unestablished; require observed low comparisons before announcing a new transition. An already-varying input does not establish a transition onset",
            "scope": "Descriptive distribution difference, no iid significance, prediction, phrase or closure claim",
            "boundary": "A human must judge whether these acoustic state changes correspond to a meaningful change of musical grouping"}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    sources = {}
    for name in (Path(__file__).name, "evaluate_spectral_change.py", "evaluate_spectral_expectation.py", "evaluate_phrase_expectation.py"):
        path = Path(__file__).with_name(name)
        shutil.copy2(path, output / name)
        sources[name] = sha256(path)
    manifest = {"status": "running", "seeds": seeds, "sources": sources,
                "numpy_version": np.__version__, "plan_sha256": sha256(output / "plan.json"), "rows": []}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    inputs = []
    for seed in seeds:
        for case in CASES:
            folder = output / f"seed-{seed}" / case
            folder.mkdir(parents=True)
            audio, truth = state_audio(case, seed)
            write_wav(folder / "audio.wav", audio)
            (folder / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
            inputs.append((folder / "audio.wav", folder, "controlled", case, seed, truth["sustained_transition_sec"]))
    for path in samples:
        folder = output / "normal" / path.parent.parent.name / path.parent.name
        folder.mkdir(parents=True)
        inputs.append((path, folder, "normal", path.parent.parent.name,
                       int(path.parent.name.removeprefix("seed-")), None))
    for path, folder, group, case, seed, expected in inputs:
        with wave.open(str(path)) as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError("expected mono PCM16")
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float) / 32768
            fs = wav.getframerate()
        rows = evaluate_audio(audio, fs)
        (folder / "observations.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        result = {"group": group, "case": case, "seed": seed, "source": str(path),
                  "audio_sha256": sha256(path), "directory": str(folder.relative_to(output)),
                  "summary": assess(rows, expected), "sha256": {p.name: sha256(p) for p in folder.iterdir()}}
        manifest["rows"].append(result)
        print(json.dumps({k: result[k] for k in ("group", "case", "seed", "summary")}), flush=True)
    controls = [r for r in manifest["rows"] if r["group"] == "controlled"]
    assert len(controls) == len(seeds) * len(CASES)
    manifest["status"] = "complete" if all(r["summary"]["passed"] for r in controls) else "failed_controls"
    manifest["status_scope"] = "Control agreement; musical grouping remains unjudged"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    if manifest["status"] != "complete":
        raise ValueError("state comparison controls failed; preserve results before revision")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--sample-root", type=Path)
    args = parser.parse_args()
    run(args.output, args.seeds, args.sample_root)
