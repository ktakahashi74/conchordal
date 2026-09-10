#!/usr/bin/env python3
"""Audit separate within-channel regularity and cross-channel coherence inputs.

Backgrounds reconstruct the timing described by Sollini et al. (2022), with
explicit diagnostic controls. These are not original stimuli, calibrated SPL,
participant data, a detection-threshold model, or a stream/phrase classifier.
"""

import argparse
import datetime as dt
import gzip
import json
from pathlib import Path
import shutil

import numpy as np

from evaluate_component_relations import ComponentRelations
from evaluate_frequency_events import FrequencyEventObserver
from evaluate_phrase_expectation import read_wav, sha256, write_wav


FS = 24_000
DURATION_SEC = .8
PRECURSOR_SEC = .5
PERIOD_SEC = .05
PIP_DURATION_SEC = .0125
FREQUENCIES_HZ = np.array([250., 500., 1000., 2000., 4000.])
CASES = ("regular_all", "jitter_all", "jitter_signal", "jitter_flankers",
         "coherent_jitter", "regular_offset", "narrow_regular", "narrow_jitter",
         "no_precursor")
DIAGNOSTICS = ("coherent_jitter", "regular_offset")
# Align the 240-sample research and 512-sample native hops; this is not a memory duration.
PRELUDES_SEC = (0., 2.24)


def stimulus(case, seed, fs=FS):
    if case not in CASES or not isinstance(seed, int) or seed < 0:
        raise ValueError("expected a known condition and a nonnegative integer seed")
    if not isinstance(fs, int) or fs < 16_000:
        raise ValueError("expected an integer sample rate of at least 16000 Hz")
    rng = np.random.default_rng(seed)
    centers = .025 + np.arange(16) * PERIOD_SEC
    offsets = rng.uniform(-.0125, .0125, (5, 10))
    shifts = np.zeros((5, 10))
    if case == "jitter_all":
        shifts[:] = offsets
    elif case in ("jitter_signal", "narrow_jitter"):
        shifts[2] = offsets[2]
    elif case == "jitter_flankers":
        shifts[[0, 1, 3, 4]] = offsets[[0, 1, 3, 4]]
    elif case == "coherent_jitter":
        shifts[:] = offsets[2]
    elif case == "regular_offset":
        shifts[[0, 1, 3, 4]] = .0125
    active = [2] if case.startswith("narrow_") else list(range(5))
    times = np.arange(round(DURATION_SEC * fs)) / fs
    envelopes = np.zeros((len(times), 5))
    channel_centers = []
    for channel in range(5):
        positions = centers.copy()
        positions[:10] += shifts[channel]
        if case == "no_precursor":
            positions = positions[10:]
        if channel not in active:
            positions = np.array([])
        channel_centers.append(positions.tolist())
        for center in positions:
            distance = times - center
            inside = np.abs(distance) <= PIP_DURATION_SEC / 2
            envelopes[inside, channel] += np.cos(np.pi * distance[inside] / PIP_DURATION_SEC)
    phases = rng.uniform(0, 2 * np.pi, 5)
    carriers = np.sin(2 * np.pi * times[:, None] * FREQUENCIES_HZ + phases)
    audio = .08 * np.sum(envelopes * carriers, axis=1)
    truth = {"case": case, "seed": seed, "sample_rate": fs,
             "origin": "additional crossed diagnostic" if case in DIAGNOSTICS else "paper background timing reconstruction",
             "frequencies_hz": FREQUENCIES_HZ.tolist(), "active_channels": active,
             "pip_centers_sec": channel_centers,
             "scope": "No target tone, original source code, calibrated SPL, or participant response. The no-precursor control is padded with observed silence to align the common masker. The paper used approximately 48 kHz; this assay uses the declared sample rate."}
    return audio, envelopes, truth


def cue_reference(envelopes, truth):
    """Stimulus audit only; source-resolved envelopes never enter the observer."""
    fs = truth["sample_rate"]
    # Avoid the onset/offset margins when testing stationary precursor structure.
    inside = envelopes[round(.1 * fs):round(.4 * fs)]
    active = truth["active_channels"]
    intervals = []
    for channel in active:
        times = [time for time in truth["pip_centers_sec"][channel] if time < PRECURSOR_SEC]
        intervals.append(float(np.std(np.diff(times))) if len(times) > 2 else None)
    pairs = []
    for offset, first in enumerate(active):
        for second in active[offset + 1:]:
            a, b = inside[:, first], inside[:, second]
            denom = float(np.std(a) * np.std(b))
            value = float(np.mean((a - np.mean(a)) * (b - np.mean(b))) / denom) if denom > 1e-20 else None
            pairs.append({"a": first, "b": second, "zero_lag_correlation": value})
    return {"interval_std_sec": intervals, "pairs": pairs}


def observe(audio, fs, packet_size=4093):
    if not isinstance(packet_size, int) or packet_size <= 0:
        raise ValueError("expected a positive packet size")
    front, relations = FrequencyEventObserver(fs), ComponentRelations()
    components, reports = [], []
    for start in range(0, len(audio), packet_size):
        for row in front.process(start, audio[start:start + packet_size]):
            components.append(row)
            report = relations.process(row)
            if report is not None:
                reports.append(report)
    return {"components": components, "relations": reports}


def observed_cues(times, frequency_hz, envelopes, prelude_sec):
    """Known-stimulus diagnostics; these frequencies and lags do not enter a model."""
    times, frequency_hz, envelopes = map(np.asarray, (times, frequency_hz, envelopes))
    assert envelopes.shape == (len(times), len(frequency_hz))
    assert len(times) > 1 and np.all(np.diff(times) > 0)
    assert np.all(np.isfinite(envelopes)) and np.all(envelopes >= 0)
    local_time = times - prelude_sec
    selected = (local_time >= .25 - 1e-9) & (local_time < .5 - 1e-9)
    assert np.sum(selected) > 1 and times[selected][0] - PERIOD_SEC >= times[0]
    indices = [int(np.argmin(abs(frequency_hz - hz))) for hz in FREQUENCIES_HZ]
    current = envelopes[selected][:, indices]
    channels = []
    for column, index in enumerate(indices):
        now = current[:, column]
        past = np.interp(times[selected] - PERIOD_SEC, times, envelopes[:, index])
        std, mean = float(np.std(now)), float(np.mean(now))
        denominator = std * np.std(past)
        correlation = float(np.mean((now - mean) * (past - np.mean(past))) / denominator) if denominator > 0 else None
        channels.append({"stimulus_frequency_hz": float(FREQUENCIES_HZ[column]),
                         "observed_center_hz": float(frequency_hz[index]),
                         "mean_envelope": mean, "std_over_mean": std / mean if mean > 0 else None,
                         "lag_50ms_correlation": correlation})
    pairs = []
    for column in (0, 1, 3, 4):
        a, b = current[:, 2], current[:, column]
        denominator = np.std(a) * np.std(b)
        value = float(np.mean((a - np.mean(a)) * (b - np.mean(b))) / denominator) if denominator > 0 else None
        pairs.append({"flanker_hz": float(FREQUENCIES_HZ[column]), "zero_lag_correlation": value})
    return {"frames": int(np.sum(selected)), "channels": channels, "signal_flanker_pairs": pairs}


def compare_observers(root, auditory):
    plan = {"window_sec": [.25, .5], "known_stimulus_lag_sec": PERIOD_SEC,
            "source_sha256": sha256(Path(__file__)), "auditory_plan_sha256": sha256(auditory / "plan.json"),
            "scope": "Descriptive cue retention, not inferred periodicity, source separation, perceptual grouping, or fitted cognitive duration. Diagnostic channel selection and 50 ms lag come from the stimulus. Native lagged envelopes use interpolation at the declared observation times; low modulation magnitudes and filter delays remain visible."}
    (auditory / "cue-comparison-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    summary = json.loads((auditory / "summary.json").read_text())
    assert [item["case"] for item in summary] == json.loads((root / "comparison.json").read_text())
    frequency = np.exp2(json.loads((auditory / "plan.json").read_text())["centers_log2"])
    comparison = []
    for item in summary:
        case = item["case"]
        with gzip.open(item["output"], "rt") as stream:
            rows = [json.loads(line) for line in stream]
        row = {"seed": case["seed"], "case": case["case"], "prelude_sec": case["prelude_sec"],
               "auditory": observed_cues([r["available_sec"] for r in rows], frequency,
                                          [r["envelope_scan"] for r in rows], case["prelude_sec"])}
        native_path = Path(case["input"]).with_name("native.jsonl")
        if native_path.is_file():
            with native_path.open() as stream:
                native = [json.loads(line) for line in stream]
            meta, native = native[0], native[1:]
            assert len(native) * meta["hop_samples"] + meta["trailing_samples"] == meta["input_samples"]
            for index, frame in enumerate(native):
                assert frame["available_sec"] == (index + 1) * meta["hop_samples"] / meta["sample_rate"]
            row["native"] = observed_cues([r["available_sec"] for r in native], meta["frequency_hz"],
                                         np.sqrt([r["nsgt_power_scan"] for r in native]), case["prelude_sec"])
        comparison.append(row)
    (auditory / "cue-comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    shutil.copy2(Path(__file__), auditory / "evaluate_grouping_cues.py")
    print(json.dumps({"cue_comparisons": len(comparison), "native_comparisons": sum("native" in row for row in comparison)}))


def run(output, seeds):
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("expected unique nonnegative seeds")
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("evaluate_frequency_events.py"),
               Path(__file__).with_name("evaluate_component_relations.py")]
    plan = {"created_at": dt.datetime.now().astimezone().isoformat(), "seeds": seeds,
            "cases": CASES, "sample_rate": FS,
            "observed_silence_preludes_sec": PRELUDES_SEC,
            "input_contract": "Both observers read the saved PCM16. The 2.24 s prelude is observed silence, aligned to 240- and 512-sample hops, not an inferred prehistory or a fitted memory time.",
            "paper": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9411505/",
            "precursor_sec": PRECURSOR_SEC, "masker_sec": .3, "period_sec": PERIOD_SEC,
            "pip_duration_sec": PIP_DURATION_SEC, "sources": {str(p): sha256(p) for p in sources},
            "checks": ["common masker PCM is identical across all broadband cases at each seed",
                       "regular_offset has regular within-channel intervals without zero-lag coherence",
                       "coherent_jitter has zero-lag coherence without regular within-channel intervals",
                       "report observer availability without turning unavailable into no relation"],
            "scope": "An input/representation audit, not a reproduction of participant results, temporal grouping, perceptual thresholds, or cognitive parameter fitting."}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows = []
    for seed in seeds:
        masker = None
        for case in CASES:
            audio, envelopes, truth = stimulus(case, seed)
            truth["reference_cues"] = cue_reference(envelopes, truth)
            for prelude in PRELUDES_SEC:
                dest = output / str(seed) / case / ("fresh" if prelude == 0 else "primed")
                dest.mkdir(parents=True)
                write_wav(dest / "audio.wav", np.concatenate((np.zeros(round(prelude * FS)), audio)))
                pcm = read_wav(dest / "audio.wav")
                current_masker = pcm[round((prelude + PRECURSOR_SEC) * FS):]
                if not case.startswith("narrow_"):
                    if masker is None:
                        masker = current_masker.copy()
                    assert np.array_equal(masker, current_masker), (case, prelude)
                case_truth = dict(truth, stimulus_start_sec=prelude,
                                  pip_time_origin="relative to stimulus_start_sec")
                (dest / "stimulus.json").write_text(json.dumps(case_truth, indent=2) + "\n")
                observation = observe(pcm, FS)
                with gzip.open(dest / "observations.json.gz", "wt") as target:
                    json.dump(observation, target, allow_nan=False)
                reports = [row for row in observation["relations"] if row["available_sec"] > prelude]
                components = [row for row in observation["components"] if row["available_sec"] > prelude]
                statuses = sorted({row.get("status", row["kind"]) for row in reports})
                rows.append({"seed": seed, "case": case, "prelude_sec": prelude,
                             "input": str((dest / "audio.wav").resolve()),
                             "wav_sha256": sha256(dest / "audio.wav"),
                             "component_frames": len(components),
                             "frames_with_components": sum(bool(row.get("components")) for row in components),
                             "relation_reports": len(reports), "relation_statuses": statuses,
                             "available_relations": sum(row.get("status") == "available" for row in reports),
                             "reference_cues": truth["reference_cues"]})
    for path in sources:
        shutil.copy2(path, output / path.name)
    (output / "comparison.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cases": len(rows), "available_relation_reports": sum(row["available_relations"] for row in rows),
                      "frames_with_components": sum(row["frames_with_components"] for row in rows),
                      "statuses": sorted({status for row in rows for status in row["relation_statuses"]})}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--compare-auditory", type=Path)
    args = parser.parse_args()
    if args.compare_auditory:
        compare_observers(args.output, args.compare_auditory)
    else:
        if not args.seeds:
            parser.error("--seeds is required for stimulus generation")
        run(args.output, args.seeds)


if __name__ == "__main__":
    main()
