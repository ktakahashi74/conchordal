#!/usr/bin/env python3
"""Research: delayed multirate envelope coherence with explicit finite support.

The gamma-sine/Hilbert pair follows Elhilali et al. (2009). Finite support,
DC removal, FFT padding, reporting cadence, and delay are numerical choices.
Power and signed normalized inner products are cues, not grouping probabilities.
"""

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

from evaluate_phrase_expectation import sha256
from evaluate_auditory_envelope import AuditoryEnvelopeObserver


class TemporalCoherenceObserver:
    def __init__(self, centers_log2, pairs, rates_hz, step_sec, report_steps,
                 positive_tail_cycles, lookahead_cycles, hilbert_padding,
                 peripheral_sample_rate=None):
        centers = np.array(centers_log2, dtype=float, copy=True)
        pairs = np.array(pairs, copy=True)
        rates = np.array(rates_hz, dtype=float, copy=True)
        if (centers.ndim != 1 or len(centers) < 2 or not np.all(np.isfinite(centers))
                or np.any(np.diff(centers) <= 0)
                or not np.allclose(np.diff(centers), np.diff(centers)[0])
                or pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs)
                or pairs.dtype.kind not in "iu" or np.any(pairs < 0) or np.any(pairs >= len(centers))
                or np.any(pairs[:, 0] >= pairs[:, 1]) or len(set(map(tuple, pairs))) != len(pairs)
                or rates.ndim != 1 or not len(rates) or not np.all(np.isfinite(rates))
                or np.any(rates <= 0) or np.any(np.diff(rates) <= 0)
                or not math.isfinite(step_sec) or step_sec <= 0 or np.any(rates >= .5 / step_sec)
                or not isinstance(report_steps, int) or isinstance(report_steps, bool) or report_steps < 1
                or np.any(rates >= .5 / (step_sec * report_steps))
                or not math.isfinite(positive_tail_cycles) or positive_tail_cycles <= 0
                or not math.isfinite(lookahead_cycles) or lookahead_cycles <= 0
                or not isinstance(hilbert_padding, int) or isinstance(hilbert_padding, bool)
                or hilbert_padding < 2):
            raise ValueError("expected a uniform frequency grid, distinct pairs/rates, adequate cadence and explicit support")
        self.centers_log2, self.pairs, self.rates_hz = centers, pairs, rates
        self.step_sec, self.report_steps = step_sec, report_steps
        self.peripheral_sample_rate = peripheral_sample_rate
        radius = None
        if peripheral_sample_rate is not None:
            if isinstance(peripheral_sample_rate, bool):
                raise ValueError("expected an explicit peripheral sample rate")
            peripheral = AuditoryEnvelopeObserver(peripheral_sample_rate, centers, stride_samples=1)
            radius = np.abs(peripheral.pole)
        ahead = np.ceil(lookahead_cycles / (rates * step_sec)).astype(int)
        past = np.ceil(positive_tail_cycles / (rates * step_sec)).astype(int)
        self.delay_steps = int(ahead.max())
        self.delay_sec = self.delay_steps * step_sec
        self.support_steps = self.delay_steps + past + 1
        dimensions = (len(rates), int(self.support_steps.max()) - 1)
        self.weights = np.zeros(dimensions if radius is None else (*dimensions, len(centers)), dtype=complex)
        self.kernels = []
        self.fft_lengths = []
        for index, rate in enumerate(rates):
            count = int(past[index]) + 1
            length = 1 << math.ceil(math.log2(hilbert_padding * (count + int(ahead[index]))))
            u = np.arange(count) * step_sec * rate
            real = u ** 2 * np.exp(-3.5 * u) * np.sin(2 * np.pi * u) * rate * step_sec
            real -= real.mean()
            spectrum = np.fft.fft(real, n=length)
            multiplier = np.zeros(length)
            multiplier[0] = multiplier[length // 2] = 1.
            multiplier[1:length // 2] = 2.
            offsets = np.arange(-ahead[index], past[index] + 1)
            if radius is None:
                analytic = np.fft.ifft(spectrum * multiplier)
                selected = analytic[offsets % length].copy()
                selected -= selected.mean()
                kernel = np.zeros(int(self.support_steps[index]), dtype=complex)
                kernel[self.delay_steps + offsets] = selected
            else:
                frequency = np.fft.fftfreq(length, d=step_sec)
                kernel = np.zeros((int(self.support_steps[index]), len(centers)), dtype=complex)
                for band, pole_radius in enumerate(radius):
                    # Undo the centered-carrier modulation phase, without inverse gain.
                    denominator = 1 - pole_radius * np.exp(-2j * np.pi * frequency / peripheral_sample_rate)
                    compensation = (denominator / np.abs(denominator)) ** 4
                    analytic = np.fft.ifft(spectrum * multiplier * compensation)
                    selected = analytic[offsets % length].copy()
                    selected -= selected.mean()
                    kernel[self.delay_steps + offsets, band] = selected
            # Differences annihilate constant inputs exactly, without an activity threshold.
            cumulative = np.cumsum(kernel, axis=0)[:-1]
            effective = np.concatenate([cumulative[:1], np.diff(cumulative, axis=0), -cumulative[-1:]], axis=0)
            self.kernels.append(effective)
            self.weights[index, -len(cumulative):] = cumulative[::-1]
            self.fft_lengths.append(length)
        width = self.weights.shape[1]
        self.differences = np.zeros((2 * width, len(centers)))
        self.cursor, self.samples = 0, 0
        self.previous = None
        self.time_sec = 0.
        self.segment_start_sec = None

    def observe(self, end_sec, envelope):
        if not math.isfinite(end_sec) or end_sec <= self.time_sec:
            raise ValueError("expected a strictly increasing finite observation endpoint")
        if envelope is None:
            self.differences.fill(0.)
            self.cursor, self.samples = 0, 0
            self.previous, self.segment_start_sec = None, None
            self.time_sec = end_sec
            return {"kind": "input_gap", "available_sec": end_sec}
        value = np.array(envelope, dtype=float, copy=True)
        if (value.shape != self.centers_log2.shape or not np.all(np.isfinite(value))
                or np.any(value < 0)
                or not math.isclose(end_sec - self.time_sec, self.step_sec, rel_tol=0, abs_tol=1e-8)):
            raise ValueError("expected contiguous nonnegative envelopes; declare gaps explicitly")
        width = self.weights.shape[1]
        if self.previous is None:
            self.segment_start_sec = self.time_sec
        else:
            difference = value - self.previous
            self.differences[self.cursor] = difference
            self.differences[self.cursor + width] = difference
            self.cursor = (self.cursor + 1) % width
        self.previous = value
        self.samples += 1
        self.time_sec = end_sec
        if self.samples % self.report_steps:
            return None
        available = self.samples >= self.support_steps
        history = self.differences[self.cursor:self.cursor + width]
        response = (self.weights @ history if self.peripheral_sample_rate is None
                    else np.einsum("rkb,kb->rb", self.weights, history))
        response[~available] = np.nan
        # Summing the six phase products equals three times the quadrature inner product.
        power = 3 * np.abs(response) ** 2
        cross = 3 * np.real(response[:, self.pairs[:, 0]] * np.conj(response[:, self.pairs[:, 1]]))
        denominator = np.sqrt(power[:, self.pairs[:, 0]] * power[:, self.pairs[:, 1]])
        normalized = np.full_like(cross, np.nan)
        np.divide(cross, denominator, out=normalized, where=denominator > 0)
        np.clip(normalized, -1., 1., out=normalized)
        total_power = power.sum(axis=0)
        total_cross = cross.sum(axis=0)
        total_denominator = np.sqrt(total_power[self.pairs[:, 0]] * total_power[self.pairs[:, 1]])
        total_normalized = np.full_like(total_cross, np.nan)
        np.divide(total_cross, total_denominator, out=total_normalized, where=total_denominator > 0)
        np.clip(total_normalized, -1., 1., out=total_normalized)
        return {"kind": "temporal_coherence", "available_sec": end_sec,
                "observation_sec": end_sec - self.delay_sec,
                "segment_start_sec": self.segment_start_sec,
                "support_start_sec": end_sec - (self.support_steps - 1) * self.step_sec,
                "rate_available": available.copy(), "response": response,
                "modulation_power": power, "cross_power": cross,
                "normalized_inner_product": normalized,
                "total_modulation_power": total_power, "total_cross_power": total_cross,
                "total_normalized_inner_product": total_normalized}


def run(inputs_path, parameters_path, output):
    cases = json.loads(inputs_path.read_text())
    parameters = json.loads(parameters_path.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("expected a nonempty input manifest")
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps({"inputs": cases, "parameters": parameters,
        "source_sha256": sha256(Path(__file__)),
        "scope": "Finite delayed coherence cues from fine envelopes, not source/stream labels or cognitive fits. Unknown prehistory and gaps are unavailable; EOF is not padded."}, indent=2) + "\n")
    summary = []
    for case in cases:
        path = Path(case["input"])
        if sha256(path) != case["sha256"]:
            raise ValueError(f"input changed: {path}")
        model = TemporalCoherenceObserver(**parameters)
        opener = gzip.open if path.suffix == ".gz" else open
        records, gaps = [], []
        with opener(path, "rt") as source:
            for line in source:
                row = json.loads(line)
                if row["kind"] not in ("auditory_envelope", "input_gap"):
                    raise ValueError("expected fine auditory envelopes or explicit gaps")
                result = model.observe(row["available_sec"], None if row["kind"] == "input_gap" else row["envelope_scan"])
                if result is not None:
                    (gaps if result["kind"] == "input_gap" else records).append(result)
        target = output / f"{case['name']}.npz"
        keys = [key for key in records[0] if key != "kind"] if records else []
        np.savez_compressed(target, **{key: np.array([r[key] for r in records]) for key in keys})
        available = np.array([r["rate_available"] for r in records])
        summary.append({"case": case["name"], "output": str(target.resolve()), "sha256": sha256(target),
                        "reported_targets": len(records), "gap_endpoints": [r["available_sec"] for r in gaps],
                        "available_targets_per_rate": available.sum(axis=0).tolist() if records else [],
                        "delay_sec": model.delay_sec, "support_steps": model.support_steps.tolist(),
                        "fft_lengths": model.fft_lengths})
        print(json.dumps(summary[-1]), flush=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("parameters", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.inputs, args.parameters, args.output)
