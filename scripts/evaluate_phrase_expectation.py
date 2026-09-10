#!/usr/bin/env python3
"""Compare causal contour prediction on controlled audio, without driving the instrument."""

import argparse
from collections import deque
import datetime as dt
import hashlib
import html
import json
import math
from pathlib import Path
import random
import shutil
import sys
import wave

import numpy as np


FS = 24_000
HOP = 240
WINDOW = 2048
NOTE_SEC = .22
IOI_SEC = .36
TRAIN_CYCLES = 6
TEST_CYCLES = 4
CONTOUR_SIZE = 8
CASES = ("repeat", "altered", "shuffled", "transposed", "timbre", "quieter",
         "pause", "renewal", "noise", "silence")
DEV_SEEDS = (1, 21, 42)
VALIDATION_SEEDS = (7, 57, 113)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stimulus(case, seed):
    if case not in CASES:
        raise ValueError(f"unknown case: {case}")
    rng = random.Random(seed)
    # Continuous offsets avoid a mandatory note-name or scale representation.
    contour = rng.sample(range(16), CONTOUR_SIZE)
    contour = [(x - min(contour)) * .051 + rng.uniform(-.008, .008) for x in contour]
    events = []
    cursor = .6
    split = .6 + TRAIN_CYCLES * CONTOUR_SIZE * IOI_SEC
    for cycle in range(TRAIN_CYCLES + TEST_CYCLES):
        offsets = contour.copy()
        if cycle >= TRAIN_CYCLES and case == "altered":
            offsets[3], offsets[4] = offsets[4], offsets[3]
        if (cycle >= TRAIN_CYCLES and case == "shuffled") or case == "renewal":
            rng.shuffle(offsets)
        if case == "pause" and cycle == TRAIN_CYCLES + 1:
            cursor += 1.2
        for offset in offsets:
            evaluation = cycle >= TRAIN_CYCLES
            if case == "transposed" and evaluation:
                offset += .37
            events.append({
                "onset_sec": cursor, "duration_sec": NOTE_SEC,
                "frequency_hz": 220 * 2 ** offset,
                "amplitude": .12 if evaluation and case == "quieter" else .25,
                "timbre": ("noise" if case == "noise" else
                           "harmonic" if evaluation and case == "timbre" else "sine"),
                "evaluation": evaluation, "cycle": cycle,
            })
            cursor += rng.uniform(.29, .55) if case == "renewal" else IOI_SEC
    if case == "silence":
        events = []
    return {"case": case, "seed": seed, "events": events,
            "evaluation_start_sec": (events[TRAIN_CYCLES * CONTOUR_SIZE]["onset_sec"]
                                     if events else split),
            "duration_sec": cursor + 1.6, "contour_log2": contour}


def synthesize(spec):
    audio = np.zeros(round(spec["duration_sec"] * FS))
    rng = np.random.default_rng(spec["seed"])
    for event in spec["events"]:
        n = round(event["duration_sec"] * FS)
        t = np.arange(n) / FS
        envelope = np.minimum(1, t / .012) * np.minimum(1, (n / FS - t) / .025)
        phase = math.tau * event["frequency_hz"] * t
        if event["timbre"] == "noise":
            tone = rng.uniform(-1, 1, n)
            tone *= math.sqrt(1.5)
        elif event["timbre"] == "harmonic":
            tone = (np.sin(phase) + .55 * np.sin(2 * phase) + .3 * np.sin(3 * phase))
            tone /= math.sqrt(1 + .55 ** 2 + .3 ** 2)
        else:
            tone = np.sin(phase)
        start = round(event["onset_sec"] * FS)
        audio[start:start + n] += event["amplitude"] * envelope * tone
    if np.max(np.abs(audio), initial=0) >= 1:
        raise ValueError("stimulus clips")
    return audio


def write_wav(path, audio):
    pcm = np.rint(audio * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setparams((1, 2, FS, 0, "NONE", "not compressed"))
        out.writeframes(pcm.tobytes())


def read_wav(path):
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, FS):
            raise ValueError("expected mono PCM16 at 24000 Hz")
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(float) / 32768


def periodic_pitch(frame, *, coherence_check=False):
    """Estimate waveform periodicity, without identifying a source or Voice."""
    x = np.asarray(frame, dtype=float)
    x = x - np.mean(x)
    if np.mean(x * x) < 1e-6:
        return None
    spectrum = np.fft.rfft(x, 2 * len(x))
    correlation = np.fft.irfft(spectrum * spectrum.conjugate())[:len(x)]
    energy = np.concatenate(([0.0], np.cumsum(x * x)))
    lags = np.arange(FS // 1200, min(FS // 100, len(x) // 2))
    denominator = energy[len(x) - lags] + energy[-1] - energy[lags]
    similarity = 2 * correlation[lags] / np.maximum(denominator, 1e-15)
    peaks = [i for i in range(1, len(lags) - 1)
             if similarity[i] >= similarity[i - 1] and similarity[i] > similarity[i + 1]]
    if not peaks:
        return None
    best = max(similarity[i] for i in peaks)
    if best < .9:
        return None
    index = next(i for i in peaks if similarity[i] >= max(.9, .95 * best))
    left, center, right = similarity[index - 1:index + 2]
    shift = .5 * (left - right) / (left - 2 * center + right)
    period = lags[index] + shift
    periodicity = float(center)
    if coherence_check:
        # A strong short-lag match can arise from beating between nearby tones.
        long_lags = np.arange(1, len(x) // 2)
        denominator = energy[len(x) - long_lags] + energy[-1] - energy[long_lags]
        similarity = 2 * correlation[long_lags] / np.maximum(denominator, 1e-15)
        multiples = period * np.arange(1, int(long_lags[-1] / period) + 1)
        if np.min(np.interp(multiples, long_lags, similarity)) < .9:
            return None
        # Multiple returns reduce short-lag frequency bias from a weak interferer.
        periods = []
        for multiple, lag in enumerate(multiples, 1):
            index = round(lag) - 1
            if index < 2 or index + 2 >= len(similarity):
                continue
            index += int(np.argmax(similarity[index - 1:index + 2])) - 1
            left, center, right = similarity[index - 1:index + 2]
            curvature = left - 2 * center + right
            if curvature >= 0:
                return None
            shift = .5 * (left - right) / curvature
            if abs(shift) > 1:
                return None
            periods.append((long_lags[index] + shift) / multiple)
        period = float(np.median(periods))
    return {"frequency_hz": float(FS / period),
            "periodicity": periodicity}


class AcousticObserver:
    """Consume contiguous samples on a fixed grid; a missing span is not silence."""

    def __init__(self, *, coherence_check=False):
        self.coherence_check = coherence_check
        self.next_sample = None
        self.pending = np.empty(HOP)
        self.pending_count = 0
        self.window = deque(maxlen=WINDOW)
        self.intervals = deque(maxlen=8)
        self.stable = deque(maxlen=3)
        self.active = False
        self.quiet = 0
        self.current = None
        self.last_onset = None
        self.gap_sent = False
        self.await_silence = False

    def process(self, start_sample, audio):
        """Return causal episode/gap records; retain only bounded working history."""
        audio = np.asarray(audio, dtype=float)
        if (not isinstance(start_sample, int) or start_sample < 0 or audio.ndim != 1
                or not np.all(np.isfinite(audio))):
            raise ValueError("expected a nonnegative sample index and finite mono audio")
        if self.next_sample is not None and start_sample < self.next_sample:
            raise ValueError("audio overlaps or arrives out of order")
        if not len(audio):
            return []
        output = []
        if self.next_sample is not None and start_sample != self.next_sample:
            output.append({"kind": "input_gap", "available_sec": start_sample / FS,
                           "missing_start_sec": self.next_sample / FS})
            self.pending_count = 0
            self.window.clear()
            self.intervals.clear()
            self.stable.clear()
            self.current, self.last_onset = None, None
            self.active, self.gap_sent, self.await_silence = False, False, True
            self.quiet = 0
        cursor = 0
        while cursor < len(audio):
            count = min(HOP - self.pending_count, len(audio) - cursor)
            self.pending[self.pending_count:self.pending_count + count] = audio[cursor:cursor + count]
            self.pending_count += count
            cursor += count
            if self.pending_count < HOP:
                continue
            end = start_sample + cursor
            self.pending_count = 0
            self.window.extend(self.pending)
            rms = math.sqrt(float(np.mean(self.pending ** 2)))
            now = end / FS
            self.quiet = self.quiet + 1 if rms < .003 else 0
            if self.await_silence:
                # A sound already present on reconnection has no observed attack.
                if self.quiet >= 2:
                    self.await_silence = False
                continue
            if not self.active and rms >= .005:
                onset = (end - HOP) / FS
                if self.last_onset is not None:
                    self.intervals.append(onset - self.last_onset)
                self.last_onset = onset
                self.gap_sent = False
                self.active = True
                self.current = {"onset_sec": onset, "available_sec": None,
                                "frequency_hz": None, "periodicity": None}
                self.stable.clear()
            if self.active:
                if self.quiet >= 2:
                    if self.current["available_sec"] is None:
                        self.current["available_sec"] = now
                        output.append({"kind": "note", **self.current})
                    self.active = False
                elif (self.current["available_sec"] is None and rms >= .003
                      and end - round(self.current["onset_sec"] * FS) >= WINDOW):
                    estimate = periodic_pitch(self.window, coherence_check=self.coherence_check)
                    if estimate is None:
                        self.stable.clear()
                    else:
                        self.stable.append(estimate)
                        if len(self.stable) == 3:
                            pitches = [math.log2(p["frequency_hz"]) for p in self.stable]
                            if max(pitches) - min(pitches) < .025:
                                self.current.update({"available_sec": now,
                                                     "frequency_hz": 2 ** float(np.median(pitches)),
                                                     "periodicity": min(p["periodicity"] for p in self.stable)})
                                output.append({"kind": "note", **self.current})
            if len(self.intervals) >= 4 and not self.active and not self.gap_sent:
                threshold = 2.5 * float(np.median(self.intervals))
                if now - self.last_onset >= threshold:
                    output.append({"kind": "gap", "available_sec": now,
                                   "last_onset_sec": self.last_onset, "threshold_sec": threshold})
                    self.gap_sent = True
        self.next_sample = start_sample + len(audio)
        return output


def observe_audio(audio, *, coherence_check=False):
    """Only samples enter; end-of-file and an incomplete hop emit no ending."""
    events = AcousticObserver(coherence_check=coherence_check).process(0, audio)
    return {key: [{k: v for k, v in e.items() if k != "kind"}
                  for e in events if e["kind"] == kind]
            for key, kind in (("notes", "note"), ("gaps", "gap"))}


class ContourPredictor:
    """Bounded continuous interval density, with and without two-interval context."""

    def __init__(self, change_rule="supported"):
        if change_rule not in ("pooled", "supported"):
            raise ValueError("unknown change calibration rule")
        self.change_rule = change_rule
        self.history = deque(maxlen=128)
        self.context = deque(maxlen=2)
        self.previous_pitch = None
        self.losses = deque(maxlen=128)

    def forecast(self):
        means = [row[1] for row in self.history]
        kernels = [math.exp(-sum((a - b) ** 2 for a, b in zip(ctx, self.context))
                            / (2 * .04 ** 2))
                   if len(ctx) == len(self.context) == 2 else 0.0
                   for ctx, _ in self.history]
        support = math.fsum(kernels)
        strength = support / (support + 4)
        count = len(means)
        null_weights = [.95 / count] * count if count else []
        weights = [((1 - strength) * base + strength * .95 * k / support)
                   if support else base for k, base in zip(kernels, null_weights)]
        eligible = self.change_rule == "pooled" or support >= 2
        threshold = (float(np.quantile(self.losses, .95)) + 2
                     if eligible and len(self.losses) >= 16 else None)
        return {"means_log2": means, "weights": weights, "null_weights": null_weights,
                "prior_weight": .05 if count else 1.0, "support": support,
                "error_threshold_bits": threshold, "calibration_eligible": eligible,
                "calibration_events": len(self.losses)}

    def step(self, frequency_hz):
        prediction = self.forecast()
        if frequency_hz is None:
            self.previous_pitch = None
            self.context.clear()
            return {"prediction": prediction, "delta_log2": None, "gain_bits": None,
                    "loss_bits": None, "error_candidate": False}
        if not math.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError("pitch must be positive and finite, or unavailable")
        pitch = math.log2(frequency_hz)
        delta = pitch - self.previous_pitch if self.previous_pitch is not None else None
        gain, loss = None, None
        candidate = False
        if delta is not None:
            prior = math.exp(-.5 * (delta / .8) ** 2) / (.8 * math.sqrt(math.tau))
            kernels = [math.exp(-.5 * ((delta - mean) / .025) ** 2)
                       / (.025 * math.sqrt(math.tau)) for mean in prediction["means_log2"]]
            conditional = prediction["prior_weight"] * prior + math.fsum(
                w * k for w, k in zip(prediction["weights"], kernels))
            marginal = prediction["prior_weight"] * prior + math.fsum(
                w * k for w, k in zip(prediction["null_weights"], kernels))
            conditional, marginal = max(conditional, 1e-300), max(marginal, 1e-300)
            gain = math.log2(conditional / marginal)
            loss = -math.log2(conditional)
            threshold = prediction["error_threshold_bits"]
            candidate = threshold is not None and loss > threshold
            self.history.append((tuple(self.context), delta))
            self.context.append(delta)
            # Compare established expectations without treating initial ignorance as normal error.
            if prediction["calibration_eligible"]:
                self.losses.append(loss)
        self.previous_pitch = pitch
        return {"prediction": prediction, "delta_log2": delta, "gain_bits": gain,
                "loss_bits": loss, "error_candidate": candidate}


def predict_notes(notes, change_rule="supported", *, input_gaps=()):
    model = ContourPredictor(change_rule)
    gaps = iter(input_gaps)
    gap = next(gaps, None)
    output = []
    for note in notes:
        while gap is not None and gap["available_sec"] <= note["available_sec"]:
            model.step(None)
            gap = next(gaps, None)
        result = model.step(note["frequency_hz"])
        prediction = result.pop("prediction")
        output.append({**note, **result, "context_support": prediction["support"],
                       "error_threshold_bits": prediction["error_threshold_bits"],
                       "calibration_events": prediction["calibration_events"],
                       "calibration_eligible": prediction["calibration_eligible"]})
    return output


def assess(spec, observations, predictions):
    truth = spec["events"]
    notes = observations["notes"]
    matches, used, errors, latencies = [], set(), [], []
    for event in truth:
        candidates = [(abs(note["onset_sec"] - event["onset_sec"]), i)
                      for i, note in enumerate(notes) if i not in used]
        distance, index = min(candidates, default=(math.inf, -1))
        if distance > .025:
            continue
        used.add(index)
        note = notes[index]
        matches.append(index)
        if note["frequency_hz"] is not None and event["timbre"] != "noise":
            errors.append(abs(1200 * math.log2(note["frequency_hz"] / event["frequency_hz"])))
            latencies.append(note["available_sec"] - event["onset_sec"])
    evaluated = [p for p in predictions if p["onset_sec"] >= spec["evaluation_start_sec"] - .015
                 and p["gain_bits"] is not None]
    return {
        "expected_events": len(truth), "detected_events": len(notes), "matched_events": len(matches),
        "unmatched_detections": len(notes) - len(used),
        "unavailable_pitch_events": sum(n["frequency_hz"] is None for n in notes),
        "max_pitch_error_cents": max(errors, default=None),
        "max_pitch_latency_sec": max(latencies, default=None),
        "evaluation_events": len(evaluated),
        "mean_gain_bits": (math.fsum(p["gain_bits"] for p in evaluated) / len(evaluated)
                           if evaluated else None),
        "error_candidates_sec": [p["available_sec"] for p in evaluated if p["error_candidate"]],
        "gap_candidates": observations["gaps"],
    }


def validate_campaign(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] not in ("rendered", "complete"):
        raise ValueError("campaign has not completed")
    if sha256(root / Path(__file__).name) != manifest["source_sha256"]:
        raise ValueError("source snapshot changed")
    expected = {(seed, case) for seed in manifest["requested_seeds"]
                for case in manifest["requested_cases"]}
    cases = {(c["seed"], c["case"]): c for c in manifest["cases"]}
    if len(cases) != len(manifest["cases"]) or set(cases) != expected:
        raise ValueError("missing or duplicate conditions")
    for case in cases.values():
        folder = root / case["directory"]
        for name, digest in case["sha256"].items():
            if sha256(folder / name) != digest:
                raise ValueError(f"changed artifact: {folder / name}")
        if case["expected_events"] != case["matched_events"] or case["unmatched_detections"]:
            raise ValueError(f"unmatched acoustic events: {folder}")
        if case["case"] == "noise":
            if case["unavailable_pitch_events"] != case["detected_events"]:
                raise ValueError("noise was assigned a periodic pitch")
        elif case["case"] != "silence":
            if case["unavailable_pitch_events"] or case["max_pitch_error_cents"] > 3:
                raise ValueError(f"unreliable contour observation: {folder}")
    pairs = []
    for seed in manifest["requested_seeds"]:
        if (seed, "repeat") not in cases:
            continue
        base_path = root / cases[seed, "repeat"]["directory"]
        base = read_wav(base_path / "audio.wav")
        spec = json.loads((base_path / "truth.json").read_text())
        cutoff = round(spec["evaluation_start_sec"] * FS)
        for case in ("altered", "shuffled", "transposed", "timbre", "quieter", "pause"):
            if (seed, case) not in cases:
                continue
            path = root / cases[seed, case]["directory"]
            other = read_wav(path / "audio.wav")
            if not np.array_equal(base[:cutoff], other[:cutoff]):
                raise ValueError(f"pre-intervention audio differs: {path}")
            pairs.append({"seed": seed, "case": case, "identical_before_sec": cutoff / FS})
    return {"artifact_and_observation_checks": "pass", "verified_cases": len(cases),
            "identical_audio_prefixes": pairs,
            "scope": "Controlled monophonic observation and causal comparisons; not phrase closure or production acceptance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEV_SEEDS + VALIDATION_SEEDS)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--change-rule", choices=("pooled", "supported"), default="supported")
    parser.add_argument("--coherence-check", action="store_true",
                        help="Require periodic agreement across the observation window")
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=VALIDATION_SEEDS)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.cases)) != len(args.cases):
        parser.error("seeds and cases must be unique")
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    shutil.copy2(source, args.output / source.name)
    manifest = {"status": "running", "created_at": dt.datetime.now().astimezone().isoformat(),
                "source_sha256": sha256(source), "numpy_version": np.__version__,
                "python_version": sys.version, "sample_rate": FS, "hop": HOP,
                "observer_scope": "Silence-separated sound episodes, 100–1200 Hz waveform periodicity; no source identity or polyphonic contour separation.",
                "coherence_check": args.coherence_check,
                "coherence_refinement": "median_local_periods" if args.coherence_check else None,
                "predictor_scope": "Two continuous log2-intervals of context, 128 past transitions; causal online scoring.",
                "evaluation_scope": "First six cycles warm up; last four score future observations before update. No labels enter observation or prediction.",
                "parameters": {"context_sigma_oct": .04, "target_sigma_oct": .025,
                               "prior_sigma_oct": .8, "prior_mass": .05, "support_regularizer": 4},
                "change_calibration": {"rule": args.change_rule, "support_min": 2,
                                       "quantile": .95, "margin_bits": 2, "minimum_events": 16},
                "development_seeds": [s for s in args.seeds if s not in args.validation_seeds],
                "validation_seeds": list(args.validation_seeds),
                "requested_seeds": args.seeds, "requested_cases": list(args.cases),
                "production_connected": False, "cases": []}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for seed in args.seeds:
        for case in args.cases:
            folder = args.output / f"seed-{seed}" / case
            folder.mkdir(parents=True)
            spec = stimulus(case, seed)
            audio = synthesize(spec)
            write_wav(folder / "audio.wav", audio)
            (folder / "truth.json").write_text(json.dumps(spec, indent=2) + "\n")
            # Re-read the actual PCM. The observer receives no generator metadata.
            observed = observe_audio(read_wav(folder / "audio.wav"), coherence_check=args.coherence_check)
            predicted = predict_notes(observed["notes"], args.change_rule)
            for name, content in (("observations", observed), ("predictions", predicted)):
                (folder / f"{name}.json").write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
            result = assess(spec, observed, predicted)
            result.update({"seed": seed, "case": case, "directory": str(folder.relative_to(args.output)),
                           "partition": "validation" if seed in args.validation_seeds else "development",
                           "sha256": {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}})
            manifest["cases"].append(result)
            print(f"seed {seed} {case}: {result['matched_events']}/{result['expected_events']} events, gain={result['mean_gain_bits']}", flush=True)
    labels = {"repeat": "A：同じ輪郭の反復", "altered": "B：途中から一部を入れ替え", "shuffled": "C：途中から順序を変更", "pause": "途中の休止と、その後の再開"}
    seed_order = sorted(args.seeds, key=lambda seed: (seed not in args.validation_seeds, seed))
    selected = seed_order[0]
    options = ''.join(f'<option value="{seed}">{seed}</option>' for seed in seed_order)
    cards = []
    for case, label in labels.items():
        if case in args.cases:
            start = 19 if case == "pause" else 15
            cards.append(f'<section><h2>{html.escape(label)}</h2>'
                         f'<audio controls preload="metadata" data-case="{case}" src="seed-{selected}/{case}/audio.wav"></audio>'
                         f'<button data-start="{start}">比較箇所の前から</button> '
                         '<button data-start="0">最初から</button></section>')
    page = ('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>反復・変形・区切りの試聴</title><style>body{max-width:760px;margin:32px auto;padding:0 18px;font-family:sans-serif;line-height:1.7}audio{width:100%}section{margin:2em 0}button,select{font-size:1rem;padding:.5em}</style>'
            '<h1>反復・変形・区切り</h1><p>音が重ならない単旋律の実験音源。A・B・Cは発音時刻・音量・音色を揃え、音の順序を比べる。</p>'
            '<p>A・B・Cは最初の約18秒が共通。B・Cで流れが変わったと感じる箇所を聞く。休止の音源では、途中の休止と最後の休止を聞き比べる。</p>'
            f'<label>音の並び（seed） <select id="seed">{options}</select></label><p id="status" role="status"></p>'
            + ''.join(cards) + '''<script>
const audios=Array.from(document.querySelectorAll('audio'));
document.addEventListener('play',e=>{if(e.target.tagName==='AUDIO')audios.forEach(a=>{if(a!==e.target)a.pause()})},true);
document.querySelector('#seed').addEventListener('change',e=>{audios.forEach(a=>{a.pause();a.src=`seed-${e.target.value}/${a.dataset.case}/audio.wav`;a.load()});document.querySelector('#status').textContent=''});
document.querySelectorAll('button').forEach(b=>b.addEventListener('click',async()=>{
  const a=b.parentElement.querySelector('audio');
  try{a.currentTime=Number(b.dataset.start);await a.play();document.querySelector('#status').textContent=''}
  catch(e){document.querySelector('#status').textContent='再生できない。音声コントロールから再試行してほしい。'}
}));
</script></html>''')
    manifest["status"] = "rendered"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    try:
        validation = validate_campaign(args.output)
    except ValueError:
        manifest["status"] = "failed_validation"
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        raise
    (args.output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    manifest["status"] = "complete"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (args.output / "audition.html").write_text(page)


if __name__ == "__main__":
    main()
