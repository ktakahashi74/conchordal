#!/usr/bin/env python3
"""Audio-only modal continuation candidates, for offline research.

The truncated matrix pencil follows Hua and Sarkar (1990), eq. 2.17:
https://intra.ece.ucr.edu/~yhua/MPM.pdf . Ranks and windows are numerical
choices, not source counts, cognitive memory parameters, or phrase lengths.
"""

import argparse
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import wave

import numpy as np


@dataclass(frozen=True)
class ModalFit:
    start_sample: int
    poles: np.ndarray
    amplitudes: np.ndarray
    fit_mse: float
    family: str = "estimated"

    def predict(self, start_sample, count):
        if start_sample < self.start_sample or count < 0:
            raise ValueError("prediction must follow the fitted time origin")
        indices = np.arange(start_sample - self.start_sample,
                            start_sample - self.start_sample + count)
        with np.errstate(over="ignore", invalid="ignore"):
            values = (self.poles[None, :] ** indices[:, None]) @ self.amplitudes
        if not np.all(np.isfinite(values)):
            raise ArithmeticError("modal extrapolation is not finite")
        if np.max(np.abs(values.imag), initial=0.) > 1e-7 * max(1., np.max(np.abs(values.real), initial=0.)):
            raise ArithmeticError("modal conjugate symmetry was lost")
        return values.real


def fit_amplitudes(audio, poles, start_sample, family="estimated"):
    poles = np.array(poles, dtype=complex, copy=True)
    basis = poles[None, :] ** np.arange(len(audio))[:, None]
    amplitudes = np.linalg.lstsq(basis, audio, rcond=1e-10)[0]
    fitted = basis @ amplitudes
    error = float(np.mean(np.abs(audio - fitted) ** 2))
    poles.setflags(write=False)
    amplitudes.setflags(write=False)
    return ModalFit(start_sample, poles, amplitudes, error, family)


def fit_candidates(audio, start_sample, *, pencil_size, ranks):
    """Propose pole sets from PCM only; real signals usually need pole pairs."""
    audio = np.asarray(audio, dtype=float)
    if (audio.ndim != 1 or len(audio) <= 2 * pencil_size or start_sample < 0
            or pencil_size < 2 or not np.all(np.isfinite(audio))
            or not ranks or any(not isinstance(r, int) or r < 0 or r > pencil_size for r in ranks)):
        raise ValueError("expected finite mono PCM, valid ranks, and enough samples")
    zero = fit_amplitudes(audio, [], start_sample, "silence")
    scale = float(np.max(np.abs(audio)))
    if scale == 0:
        return [zero]
    windows = np.lib.stride_tricks.sliding_window_view(audio / scale, pencil_size + 1)
    y0, y1 = windows[:, :-1], windows[:, 1:]
    u, singular, vh = np.linalg.svd(y0, full_matrices=False)
    # Relative cutoff addresses floating-point rank, not auditory audibility.
    numerical_rank = int(np.count_nonzero(singular > singular[0] * 1e-10))
    result = []
    for rank in sorted({min(r, numerical_rank) for r in ranks}):
        if rank == 0:
            result.append(zero)
            continue
        reduced = (u[:, :rank].T @ y1 @ vh[:rank].T) / singular[:rank, None]
        poles = np.linalg.eigvals(reduced).astype(complex)
        poles = poles[np.lexsort((poles.real, poles.imag))]
        # Growing poles remain visible; they are not silently made passive.
        result.append(fit_amplitudes(audio, poles, start_sample))
        # A separate constant-amplitude hypothesis handles sustained carriers.
        unit_poles = np.divide(poles, np.abs(poles), out=np.zeros_like(poles), where=np.abs(poles) > 0)
        result.append(fit_amplitudes(audio, unit_poles, start_sample, "undamped"))
    return result


def select_candidate(audio, start_sample, *, validation_samples, pencil_size,
                     ranks, base=None):
    """Select a frozen fit on observed holdout PCM before issuing a forecast.

With base, retain its pole set and fit an amplitude increment plus residual
spectral content. This does not identify an extra perceptual source.
"""
    train, validation = audio[:-validation_samples], audio[-validation_samples:]
    residual = train
    if base is not None:
        base_train = fit_amplitudes(train, base.poles, start_sample, base.family)
        residual = train - base_train.predict(start_sample, len(train))
    candidates = fit_candidates(residual, start_sample, pencil_size=pencil_size, ranks=ranks)
    scores = []
    for candidate in candidates:
        try:
            prediction = candidate.predict(start_sample + len(train), validation_samples)
            if base is not None:
                prediction = prediction + base_train.predict(start_sample + len(train), validation_samples)
            score = float(np.mean((prediction - validation) ** 2))
        except ArithmeticError:
            score = float("inf")
        scores.append(score)
    # An unforced passive body cannot grow. Retain such fits as diagnostics,
    # rather than projecting their poles and pretending the estimate was stable.
    passive = [bool(np.all(np.abs(c.poles) <= 1. + 1e-10)) for c in candidates]
    best = int(np.argmin([s if valid else float("inf") for s, valid in zip(scores, passive)]))
    chosen_rank = len(candidates[best].poles)
    # Refitting poles after validation would create an unvalidated model.
    return candidates[best], base_train if base is not None else None, {
                                 "ranks": [len(c.poles) for c in candidates],
                                 "families": [c.family for c in candidates],
                                 "passive_eligible": passive,
                                 "validation_mse": [s if np.isfinite(s) else None for s in scores],
                                 "candidate_growth_per_sample": [float(np.max(np.log(np.maximum(np.abs(c.poles), np.finfo(float).tiny)), initial=0.)) for c in candidates],
                                 "selected_rank": chosen_rank,
                                 "selected_family": candidates[best].family}


class ModalObserver:
    """Issue frozen forecasts from contiguous past PCM; gaps cancel evidence."""

    variants = ("continue", "same_poles_updated_state", "same_plus_residual", "fresh")

    def __init__(self, fs, *, fit_sec=.04, horizon_sec=.01, step_sec=.05,
                 pencil_size=96, ranks=(0, 2, 4, 8, 12, 16)):
        if not isinstance(fs, int) or fs < 16000:
            raise ValueError("expected an integer sample rate >=16000")
        if not all(np.isfinite(v) and v > 0 for v in (fit_sec, horizon_sec, step_sec)):
            raise ValueError("expected positive finite numerical intervals")
        self.fs = fs
        self.fit_samples = round(fit_sec * fs)
        self.horizon = round(horizon_sec * fs)
        self.step = round(step_sec * fs)
        if (self.fit_samples <= 2 * pencil_size or self.horizon < 1
                or self.step < self.horizon or self.horizon > self.fit_samples
                or not ranks or 0 not in ranks
                or any(not isinstance(r, int) or r < 0 or r > pencil_size for r in ranks)):
            raise ValueError("invalid fit, holdout, step, pencil, or candidate ranks")
        self.pencil_size = pencil_size
        self.ranks = tuple(ranks)
        self.history_size = self.fit_samples + self.horizon
        self.history = np.empty(0)
        self.next_sample = None
        self.next_issue = None
        self.body = None

    def process(self, start_sample, audio):
        audio = np.asarray(audio, dtype=float)
        if (not isinstance(start_sample, int) or start_sample < 0 or audio.ndim != 1
                or not np.all(np.isfinite(audio))
                or self.next_sample is not None and start_sample < self.next_sample):
            raise ValueError("expected ordered finite mono PCM")
        if not len(audio):
            return []
        rows = []
        if self.next_sample is None or self.next_sample != start_sample:
            if self.next_sample is not None:
                rows.append({"kind": "input_gap", "missing_start_sample": self.next_sample,
                             "available_sample": start_sample})
            self.history = np.empty(0)
            self.body = None
            self.next_issue = start_sample + self.history_size
        cursor = start_sample
        end = start_sample + len(audio)
        while cursor < end:
            stop = min(end, self.next_issue)
            self.history = np.concatenate((self.history, audio[cursor-start_sample:stop-start_sample]))[-self.history_size:]
            cursor = stop
            if cursor != self.next_issue:
                continue
            evidence_start = cursor - len(self.history)
            fresh, _, selection = select_candidate(
                self.history, evidence_start, validation_samples=self.horizon,
                pencil_size=self.pencil_size, ranks=self.ranks)
            predictions = np.full((len(self.variants), self.horizon), np.nan)
            available = np.zeros(len(self.variants), dtype=bool)
            models = {"fresh": fresh}
            added_selection = None
            if self.body is not None:
                added, updated, added_selection = select_candidate(
                    self.history, evidence_start, validation_samples=self.horizon,
                    pencil_size=self.pencil_size, ranks=self.ranks, base=self.body)
                models["continue"] = self.body
                models["same_poles_updated_state"] = updated
                models["same_plus_residual"] = added
            for index, name in enumerate(self.variants):
                if name not in models:
                    continue
                try:
                    values = models[name].predict(cursor, self.horizon)
                    if name == "same_plus_residual":
                        values = values + models["same_poles_updated_state"].predict(cursor, self.horizon)
                    if np.all(np.isfinite(values)):
                        predictions[index] = values
                        available[index] = True
                except ArithmeticError:
                    pass
            predictions.setflags(write=False)
            available.setflags(write=False)
            rows.append({"kind": "forecast", "issued_sample": cursor,
                         "evidence_start_sample": evidence_start,
                         "target_end_sample": cursor + self.horizon,
                         "predictions": predictions, "available": available,
                         "selection": selection, "added_selection": added_selection,
                         "poles": fresh.poles,
                         "forecast_sha256": hashlib.sha256(predictions.tobytes()).hexdigest()})
            self.body = fresh
            self.next_issue += self.step
        self.next_sample = end
        return rows


def run(manifest, output):
    cases = json.loads(manifest.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("expected a nonempty list of name/wav/sha256 records")
    output.mkdir(parents=True, exist_ok=False)
    plan = {"created_at": dt.datetime.now().astimezone().isoformat(), "cases": cases,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fit_sec": .04, "validation_and_horizon_sec": .01, "step_sec": .05,
            "pencil_size": 96, "ranks": [0, 2, 4, 8, 12, 16],
            "variants": ModalObserver.variants, "zero_prediction_is_baseline": True,
            "families": ["silence", "estimated_passive", "undamped"],
            "positive_growth_fit": "retained in diagnostics, ineligible as an unforced body",
            "status": "Audio-only acoustic candidates; no source/flow/cognitive labels"}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    summaries = []
    for index, case in enumerate(cases):
        path = Path(case["wav"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
            raise ValueError(f"input digest mismatch: {path}")
        with wave.open(str(path)) as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
                raise ValueError("expected uncompressed mono PCM16 WAV")
            fs = wav.getframerate()
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float) / 32768.
        observer = ModalObserver(fs)
        forecasts, metadata, observed = [], [], []
        started = time.monotonic()
        # The observer never receives samples after an issue time before issuing.
        for start in range(0, len(audio), round(.037 * fs)):
            for row in observer.process(start, audio[start:start + round(.037 * fs)]):
                if row["kind"] != "forecast":
                    continue
                predictions = row.pop("predictions")
                available = row.pop("available")
                poles = row.pop("poles")
                row["available"] = available.tolist()
                row["poles_real"] = poles.real.tolist()
                row["poles_imag"] = poles.imag.tolist()
                row["growth_per_sec"] = (np.log(np.maximum(np.abs(poles), np.finfo(float).tiny)) * fs).tolist()
                row["outcome"] = "observed" if row["target_end_sample"] <= len(audio) else "unresolved_eof"
                truth = np.full(observer.horizon, np.nan)
                available_truth = audio[row["issued_sample"]:row["target_end_sample"]]
                truth[:len(available_truth)] = available_truth
                metadata.append(row)
                forecasts.append(predictions)
                observed.append(truth)
        predictions = np.array(forecasts)
        actual = np.array(observed)
        valid_target = np.all(np.isfinite(actual), axis=1)
        error = np.mean((predictions - actual[:, None, :]) ** 2, axis=2)
        zero_error = np.mean(actual ** 2, axis=1)
        common = valid_target & np.all(np.isfinite(error), axis=1)
        summary = {"name": case["name"], "fs": fs, "audio_sec": len(audio) / fs,
                   "issued": len(metadata), "common_scored": int(np.count_nonzero(common)),
                   "unresolved_eof": int(np.count_nonzero(~valid_target)),
                   "error_energy_over_zero_common": (np.sum(error[common], axis=0) / np.sum(zero_error[common])).tolist()
                   if np.sum(zero_error[common]) > 0 else [None] * len(observer.variants),
                   "elapsed_sec": time.monotonic() - started}
        np.savez_compressed(output / f"{index:03d}.npz", predictions=predictions,
                            actual=actual, mse=error, zero_mse=zero_error)
        (output / f"{index:03d}.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        summaries.append(summary)
        (output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.manifest, args.output)
