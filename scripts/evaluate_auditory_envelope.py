#!/usr/bin/env python3
"""Causal auditory-band envelopes for research, without source or stream labels.

The four repeated poles and one-ERB bandwidth follow Hohmann (2002), equations
13-14 and the normalization in section 2.2. This independently written filter
is a linear auditory approximation, not a fitted listener or auditory nerve.
The Log2 grid and RMS reporting interval are explicit numerical choices.
"""

import argparse
import datetime as dt
import gzip
import json
from pathlib import Path
import shutil

import numpy as np

from evaluate_component_relations import CENTERS_LOG2
from evaluate_phrase_expectation import FS, read_wav, sha256


class AuditoryEnvelopeObserver:
    def __init__(self, fs, centers_log2, *, stride_samples):
        centers = np.array(centers_log2, dtype=float, copy=True)
        if (not isinstance(fs, int) or fs < 16_000 or not isinstance(stride_samples, int)
                or stride_samples < 1 or centers.ndim != 1 or not len(centers)
                or not np.all(np.isfinite(centers)) or np.any(centers < 0)
                or np.any(centers >= np.log2(fs / 2))
                or len(centers) > 1 and (np.any(np.diff(centers) <= 0)
                                       or not np.allclose(np.diff(centers), np.diff(centers)[0]))):
            raise ValueError("expected a sample rate >=16000, a uniform Log2 grid below Nyquist, and a positive integer stride")
        self.fs = fs
        self.centers_log2 = centers
        self.stride = stride_samples
        frequency = np.exp2(centers)
        self.erb_hz = 24.7 + frequency / 9.265
        # The fourth-order power integral is 5*pi/16 times its pole bandwidth.
        bandwidth = self.erb_hz * 16 / (5 * np.pi)
        radius = np.exp(-2 * np.pi * bandwidth / fs)
        self.pole = radius * np.exp(2j * np.pi * frequency / fs)
        self.gain = 2 * (1 - radius) ** 4
        self.state = np.zeros((4, len(centers)), dtype=complex)
        self.power_sum = np.zeros(len(centers))
        self.pending = 0
        self.next_sample = None
        self.segment_start = None

    def process(self, start_sample, audio):
        audio = np.asarray(audio, dtype=float)
        if (not isinstance(start_sample, int) or start_sample < 0 or audio.ndim != 1
                or not np.all(np.isfinite(audio))
                or self.next_sample is not None and start_sample < self.next_sample):
            raise ValueError("expected ordered sample indices and finite mono audio")
        if not len(audio):
            return []
        result = []
        if self.next_sample is None:
            self.segment_start = start_sample
        elif start_sample != self.next_sample:
            result.append({"kind": "input_gap", "available_sec": start_sample / self.fs,
                           "missing_start_sec": self.next_sample / self.fs})
            self.state.fill(0.)
            self.power_sum.fill(0.)
            self.pending = 0
            self.segment_start = start_sample
        for offset, sample in enumerate(audio):
            self.state *= self.pole
            self.state[0] += sample * self.gain
            self.state[1] += self.state[0]
            self.state[2] += self.state[1]
            self.state[3] += self.state[2]
            self.power_sum += self.state[3].real ** 2 + self.state[3].imag ** 2
            self.pending += 1
            if self.pending == self.stride:
                end = start_sample + offset + 1
                envelope = np.sqrt(self.power_sum / self.stride)
                assert envelope.shape == self.centers_log2.shape
                result.append({"kind": "auditory_envelope", "available_sec": end / self.fs,
                               "evidence_start_sec": self.segment_start / self.fs,
                               "observation_age_sec": (end - self.segment_start) / self.fs,
                               "envelope_scan": envelope.tolist()})
                self.power_sum.fill(0.)
                self.pending = 0
        self.next_sample = start_sample + len(audio)
        return result


def run(manifest, output):
    cases = json.loads(manifest.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("expected a nonempty input manifest")
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("evaluate_component_relations.py"),
               Path(__file__).with_name("evaluate_phrase_expectation.py")]
    plan = {"created_at": dt.datetime.now().astimezone().isoformat(),
            "input_manifest": str(manifest.resolve()), "input_manifest_sha256": sha256(manifest),
            "cases": cases, "sample_rate": FS, "centers_log2": CENTERS_LOG2.tolist(),
            "stride_samples": 24, "order": 4, "erb_l_hz": 24.7, "erb_q": 9.265,
            "sources": {str(path): sha256(path) for path in sources},
            "reference": "https://doi.org/10.5281/zenodo.20745033",
            "scope": "Observed PCM only. Full Log2 envelopes, without ridge gating, onset requirements, source labels, or a grouping decision. Each gap resets the filter approximation; unobserved prehistory is not silence. The 1 ms RMS reporting interval and grid are numerical choices, not cognitive integration or memory durations."}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    summary = []
    for index, case in enumerate(cases):
        path = Path(case["input"])
        if sha256(path) != case["wav_sha256"]:
            raise ValueError(f"input changed: {path}")
        observer = AuditoryEnvelopeObserver(FS, CENTERS_LOG2, stride_samples=24)
        audio = read_wav(path)
        target = output / f"{index:03}.jsonl.gz"
        count = nonzero = 0
        with gzip.open(target, "wt") as stream:
            for start in range(0, len(audio), 4093):
                for row in observer.process(start, audio[start:start + 4093]):
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    count += 1
                    nonzero += any(value > 0 for value in row["envelope_scan"])
        summary.append({"case": case, "output": str(target.resolve()),
                        "output_sha256": sha256(target), "frames": count,
                        "nonzero_frames": nonzero, "pending_samples": observer.pending})
        print(json.dumps({"completed": index + 1, "total": len(cases), "frames": count}), flush=True)
    for path in sources:
        shutil.copy2(path, output / path.name)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.manifest, args.output)
