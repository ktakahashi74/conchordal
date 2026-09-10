#!/usr/bin/env python3
"""Audit waveform periodicity and missing-input semantics before runtime attachment."""

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import shutil

import numpy as np

from evaluate_phrase_expectation import (
    FS, AcousticObserver, observe_audio, predict_notes, read_wav, sha256,
    stimulus, synthesize, write_wav,
)


CASES = ("single", "harmonic", "missing_fundamental", "harmonic_mixture",
         "close_pair", "unrelated_pair", "dominant", "overlap", "noise", "silence")
DEV_SEEDS = (1, 21, 42)
VALIDATION_SEEDS = (617, 811, 1013)


def mixture(case, seed):
    rng = np.random.default_rng(seed)
    base = float(rng.uniform(190, 310))
    phase = float(rng.uniform(0, math.tau))
    audio = np.zeros(6 * FS)
    onsets = [.3 + index * .4 for index in range(12)]
    frequencies, amplitudes = [base], [1.0]
    if case == "harmonic":
        frequencies, amplitudes = [base, 2 * base, 3 * base], [1, .55, .3]
    elif case in ("missing_fundamental", "harmonic_mixture"):
        frequencies, amplitudes = [2 * base, 3 * base], [1, .8]
    elif case == "close_pair":
        frequencies, amplitudes = [base, 1.06 * base], [1, 1]
    elif case in ("unrelated_pair", "dominant", "overlap"):
        frequencies, amplitudes = [base, math.sqrt(2) * base], [1, .1 if case == "dominant" else 1]
    for onset in onsets:
        t = np.arange(round(.24 * FS)) / FS
        envelope = np.minimum(1, t / .012) * np.minimum(1, (.24 - t) / .025)
        components = [a * np.sin(math.tau * f * t + (index + 1) * phase)
                      for index, (f, a) in enumerate(zip(frequencies, amplitudes))]
        if case == "noise":
            components = [rng.uniform(-1, 1, len(t))]
        if case == "silence":
            components = [np.zeros(len(t))]
        if case == "overlap":
            for index, component in enumerate(components):
                start = round((onset + index * .2) * FS)
                audio[start:start + len(t)] += .15 * envelope * component
        else:
            # The two source interpretations deliberately have identical samples.
            start = round(onset * FS)
            audio[start:start + len(t)] += .15 * envelope * sum(components)
    return audio, {"case": case, "seed": seed, "base_hz": base,
                   "component_hz": frequencies, "component_amplitudes": amplitudes,
                   "onsets_sec": onsets, "duration_sec": 6}


def run(root, validation_seeds=None):
    validation_seeds = VALIDATION_SEEDS if validation_seeds is None else tuple(validation_seeds)
    if len(set(DEV_SEEDS + validation_seeds)) != len(DEV_SEEDS + validation_seeds):
        raise ValueError("development and validation seeds must be unique and disjoint")
    root.mkdir(parents=True, exist_ok=False)
    plan = {"created_at": dt.datetime.now().astimezone().isoformat(),
            "development_seeds": DEV_SEEDS, "validation_seeds": validation_seeds,
            "cases": CASES,
            "accept_periodic": ["single", "harmonic", "missing_fundamental", "harmonic_mixture", "dominant"],
            "reject_periodic": ["close_pair", "unrelated_pair", "noise"],
            "identity": "missing fundamental and harmonic mixture must have identical PCM and observations",
            "overlap": "One connected sound episode cannot expose the two individual onset streams",
            "missing_input": "Do not count missing samples as silence, reconnecting sound as an attack, or score an interval across the gap",
            "scope": "An acoustic periodicity feature; neither source count nor Voice pitch attribution"}
    (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    sources = {}
    for name in (Path(__file__).name, "evaluate_phrase_expectation.py"):
        source = Path(__file__).with_name(name)
        shutil.copy2(source, root / name)
        sources[name] = sha256(source)
    manifest = {"status": "running", "plan_sha256": sha256(root / "plan.json"),
                "sources": sources, "numpy_version": np.__version__, "cases": [], "missing": []}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    checks = []
    for seed in DEV_SEEDS + validation_seeds:
        previous = None
        for case in CASES:
            folder = root / f"seed-{seed}" / case
            folder.mkdir(parents=True)
            audio, truth = mixture(case, seed)
            write_wav(folder / "audio.wav", audio)
            audio = read_wav(folder / "audio.wav")
            (folder / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
            observations = {}
            for rule, coherent in (("short_lag", False), ("coherent", True)):
                observations[rule] = observe_audio(audio, coherence_check=coherent)
                (folder / f"{rule}.json").write_text(json.dumps(observations[rule], indent=2) + "\n")
            notes = observations["coherent"]["notes"]
            periodic = [n for n in notes if n["frequency_hz"] is not None]
            if case in plan["accept_periodic"]:
                passed = len(periodic) == 12 and all(
                    abs(1200 * math.log2(n["frequency_hz"] / truth["base_hz"])) < 3 for n in periodic)
            elif case in plan["reject_periodic"]:
                passed = len(notes) == 12 and not periodic
            elif case == "silence":
                passed = not notes and not observations["coherent"]["gaps"]
            else:
                passed = len(notes) == 1
            if case == "harmonic_mixture":
                passed = passed and previous == (sha256(folder / "audio.wav"), observations)
            if case == "missing_fundamental":
                previous = (sha256(folder / "audio.wav"), observations)
            checks.append(bool(passed))
            manifest["cases"].append({"seed": seed, "case": case, "passed": bool(passed),
                                      "observed_episodes": len(notes), "periodic_episodes": len(periodic),
                                      "short_lag_periodic_episodes": sum(n["frequency_hz"] is not None for n in observations["short_lag"]["notes"]),
                                      "directory": str(folder.relative_to(root)),
                                      "sha256": {p.name: sha256(p) for p in folder.iterdir()}})
        folder = root / f"seed-{seed}" / "missing"
        folder.mkdir()
        spec = stimulus("repeat", seed)
        spec["events"], spec["duration_sec"] = spec["events"][:20], 8.2
        write_wav(folder / "audio.wav", synthesize(spec))
        audio = read_wav(folder / "audio.wav")
        missing_start, resumed = round(3.51 * FS), round(5.09 * FS)
        observer = AcousticObserver(coherence_check=True)
        events = observer.process(0, audio[:missing_start])
        events += observer.process(resumed, audio[resumed:])
        notes = [e for e in events if e["kind"] == "note"]
        gaps = [e for e in events if e["kind"] == "input_gap"]
        predictions = predict_notes(notes, input_gaps=gaps)
        post = next(p for p in predictions if p["available_sec"] > resumed / FS)
        silence = audio.copy()
        silence[missing_start:resumed] = 0
        write_wav(folder / "true_silence.wav", silence)
        silenced = observe_audio(read_wav(folder / "true_silence.wav"), coherence_check=True)
        passed = (len(gaps) == 1 and post["onset_sec"] == 5.28 and post["delta_log2"] is None
                  and not any(e["kind"] == "gap" and e["available_sec"] <= 5.3 for e in events)
                  and any(g["available_sec"] < 5.09 for g in silenced["gaps"]))
        for name, data in (("events", events), ("predictions", predictions), ("true_silence", silenced)):
            (folder / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
        checks.append(passed)
        manifest["missing"].append({"seed": seed, "passed": passed,
                                     "first_resumed_onset_sec": post["onset_sec"],
                                     "directory": str(folder.relative_to(root)),
                                     "sha256": {p.name: sha256(p) for p in folder.iterdir()}})
    manifest["status"] = "complete" if all(checks) else "failed_checks"
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": manifest["status"], "checks": len(checks), "passed": sum(checks)}))
    if not all(checks):
        raise ValueError("observability checks failed; inspect manifest before changing the contract")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=VALIDATION_SEEDS)
    args = parser.parse_args()
    run(args.output, args.validation_seeds)
