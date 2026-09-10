#!/usr/bin/env python3
"""Audit information discarded by three-band temporal energy observations.

Constructed equal-energy signals test an acoustic representation, not human
stream identity, cognitive retention, or a participation policy.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import wave

import numpy as np

from evaluate_auditory_envelope import AuditoryEnvelopeObserver
from evaluate_component_relations import CENTERS_LOG2
from evaluate_frequency_events import FrequencyEventObserver


GROUPS_HZ = np.array([[100, 200, 300], [500, 600, 700], [900, 1000, 1100],
                     [1300, 1400, 1500], [1800, 2100, 2400], [3200, 4000, 4800]])
PATTERNS = dict(steady_a=(0,), steady_b=(1,), alternating=(0, 1), paired=(0, 0, 1, 1))


def period_pair(sample_rate, seed):
    """Equal ideal band energies, disjoint spectra and shared zero filter states."""
    rng = np.random.default_rng(seed)
    size = round(sample_rate / 100)
    alpha = 1 - np.exp(-2 * np.pi * np.array([200., 3000.]) / sample_rate)
    bases, powers, coefficients, boundaries = [], [], [], []
    for frequencies in GROUPS_HZ:
        omega = 2 * np.pi * frequencies / sample_rate
        h = alpha[:, None] / (1 - (1 - alpha[:, None]) * np.exp(-1j * omega))
        previous = h * np.exp(-1j * omega)
        constraints = np.vstack((np.c_[previous.real, previous.imag],
                                 np.r_[np.ones(3), np.zeros(3)],
                                 np.r_[np.cos(omega), -np.sin(omega)]))
        _, singular, vh = np.linalg.svd(constraints, full_matrices=True)
        rank = np.count_nonzero(singular > singular[0] * 1e-11)
        null = vh[rank:].T
        vector = null @ rng.normal(size=null.shape[1])
        vector /= np.sqrt(np.sum(vector * vector) / 2)
        phase = np.arange(size)[:, None] * omega
        basis = np.cos(phase) @ vector[:3] + np.sin(phase) @ vector[3:]
        component_power = (vector[:3] ** 2 + vector[3:] ** 2) / 2
        bands = np.array([h[0], h[1] - h[0], 1 - h[1]])
        powers.append(np.r_[np.abs(bands) ** 2 @ component_power, component_power.sum()])
        bases.append(basis)
        coefficients.append(vector)
        boundaries.append(constraints @ vector)
    bases, matrix = np.array(bases), np.array(powers).T
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    rank = np.count_nonzero(singular > singular[0] * 1e-11)
    null = vh[rank:].T
    signed = null @ rng.normal(size=null.shape[1])
    weights = np.array([np.maximum(signed, 0), np.maximum(-signed, 0)])
    if np.any(weights.sum(axis=1) <= 0):
        raise ValueError("construction did not produce two nonempty power mixtures")
    weights /= np.mean(weights.sum(axis=1))
    periods = np.sqrt(weights) @ bases
    gain = .2 / np.max(np.abs(periods))
    periods *= gain
    expected = (matrix @ weights.T).T * gain ** 2
    return dict(sample_rate=sample_rate, seed=seed, periods=periods, bases=bases,
                coefficients=np.array(coefficients), group_power=weights,
                energy_matrix=matrix, matrix_rank=int(rank), gain=gain,
                expected_energy=expected, boundary_constraints=np.array(boundaries),
                frequencies_hz=GROUPS_HZ.copy())


def sequence(pair, pattern, duration_sec=4.8):
    choices = PATTERNS[pattern]
    blocks = round(duration_sec * 100)
    # Forty analysis periods per segment; this is a construction schedule.
    selected = np.array([choices[(i // 40) % len(choices)] for i in range(blocks)])
    return pair["periods"][selected].reshape(-1), selected


def write_pcm(path, audio, sample_rate):
    if np.max(np.abs(audio), initial=0) >= 1:
        raise ValueError("constructed audio would clip")
    values = np.rint(audio * 32768).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(values.tobytes())


def read_pcm(path):
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getcomptype() != "NONE":
            raise ValueError("expected uncompressed mono PCM16")
        fs = source.getframerate()
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(float) / 32768
    return fs, audio


def generate(root):
    root = Path(root)
    if (root / "observation-plan.json").exists():
        raise FileExistsError("controls already generated")
    cases, native = [], []
    for fs in (24000, 48000):
        for seed in (4301, 4313, 4327):
            pair = period_pair(fs, seed)
            prefix = f"pair-{fs}-{seed}"
            np.savez_compressed(root / "controls" / (prefix + ".npz"), **pair)
            for pattern in PATTERNS:
                name = prefix + "-" + pattern
                audio, selected = sequence(pair, pattern)
                path = root / "controls" / (name + ".wav")
                write_pcm(path, audio, fs)
                cases.append(dict(case=name, role="constructed", input=str(path.resolve()),
                                  sample_rate=fs, seed=seed, pattern=pattern,
                                  pair=str((root / "controls" / (prefix + ".npz")).resolve())))
    for case in json.loads((root / "ordinary-plan.json").read_text()):
        if hashlib.sha256(Path(case["input"]).read_bytes()).hexdigest() != case["sha256"]:
            raise ValueError("ordinary input changed")
        cases.append(dict(case=case["case"], role="reused_ordinary", input=case["input"]))
    for case in cases:
        native.append(dict(input=case["input"], output=str((root / "native" / (case["case"] + ".jsonl")).resolve())))
        case["input_sha256"] = hashlib.sha256(Path(case["input"]).read_bytes()).hexdigest()
    (root / "observation-plan.json").write_text(json.dumps(cases, indent=2) + "\n")
    (root / "native-plan.json").write_text(json.dumps(native, indent=2) + "\n")
    print("Generated", len(cases) - 3, "controls; retained", 3, "ordinary inputs", flush=True)


def observe_pcm(audio, sample_rate, packet_size=4093):
    """Both existing observers receive only the same decoded causal PCM."""
    envelope = AuditoryEnvelopeObserver(sample_rate, CENTERS_LOG2,
                                       stride_samples=round(sample_rate / 100))
    components = FrequencyEventObserver(sample_rate)
    envelope_rows, component_rows = [], []
    for start in range(0, len(audio), packet_size):
        packet = audio[start:start + packet_size]
        envelope_rows.extend(envelope.process(start, packet))
        component_rows.extend(components.process(start, packet))
    return envelope_rows, component_rows


def observe(root):
    root = Path(root)
    output = root / "observed"
    output.mkdir(exist_ok=False)
    summaries = []
    for case in json.loads((root / "observation-plan.json").read_text()):
        source = Path(case["input"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != case["input_sha256"]:
            raise ValueError("PCM input changed")
        fs, audio = read_pcm(source)
        envelopes, components = observe_pcm(audio, fs)
        if any(r["kind"] != "auditory_envelope" for r in envelopes):
            raise ValueError("unexpected gap in a contiguous file")
        np.savez_compressed(output / (case["case"] + ".npz"),
                            sample_rate=fs, centers_log2=CENTERS_LOG2,
                            available_frame=np.rint(np.array([r["available_sec"] for r in envelopes]) * fs).astype(np.int64),
                            envelope=np.array([r["envelope_scan"] for r in envelopes]))
        with gzip.open(output / (case["case"] + "-components.jsonl.gz"), "wt") as stream:
            for row in components:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        summary = dict(case=case["case"], role=case["role"], sample_rate=fs,
                       samples=len(audio), envelope_frames=len(envelopes),
                       component_frames=len(components),
                       component_truncations=sum(r.get("truncated_components", 0) for r in components),
                       maximum_observed_components=max((len(r.get("components", [])) for r in components), default=0),
                       frames_with_multiple_components=sum(len(r.get("components", [])) > 1 for r in components),
                       pcm_sha256=case["input_sha256"])
        summaries.append(summary)
        (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print(case["case"], len(envelopes), "envelopes", summary["maximum_observed_components"], "max components", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("generate", "observe"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    (generate if args.operation == "generate" else observe)(args.root)
