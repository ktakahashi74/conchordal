"""Acquire and independently check offline counterfactual feature targets.

The Rust acquisition reuses the normal private PCM lane. NumPy recomputes every
raw coordinate from the saved assigned spectrum and the original PCM. This does
not independently validate NSGT, body-window numerics, or a prediction model.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_reference(energy, scan, coordinates, shape):
    """Batch formulas, independent of Rust's streaming accumulators."""
    expected = np.full((len(energy), 10), np.nan)
    expected[:, 2] = np.log2(np.maximum(np.sqrt(energy), 1e-6))
    expected[:, 6] = (energy > 0).astype(float)
    change = np.diff(expected[:, 2])
    expected[1:, 3] = np.maximum(change, 0)
    expected[1:, 4] = np.maximum(-change, 0)
    mass = scan.sum(axis=1)
    supported = shape & (mass > 0)
    weights = scan[supported] / mass[supported, None]
    center = np.sum(weights * coordinates, axis=1)
    delta = coordinates - center[:, None]
    expected[supported, 0] = center
    expected[supported, 1] = np.sqrt(np.sum(weights * delta**2, axis=1))
    expected[supported, 7] = np.sum(weights * (delta < -.5), axis=1)
    expected[supported, 8] = np.sum(weights * ((delta >= -.5) & (delta <= .5)), axis=1)
    expected[supported, 9] = np.sum(weights * (delta > .5), axis=1)
    flux_ok = shape[1:] & shape[:-1] & ~((energy[1:] > 0) & (mass[1:] == 0)) & ~((energy[:-1] > 0) & (mass[:-1] == 0))
    flux = np.maximum(np.diff(.5 * np.log2(np.maximum(scan, 1e-12)), axis=0), 0).mean(axis=1)
    expected[1:, 5] = np.where(flux_ok, flux, np.nan)
    return expected


def verify(inputs, output):
    manifest = json.loads((output / "manifest.json").read_text())
    source = json.loads((inputs / "manifest.json").read_text())
    medoids = source["schema"] == "i10-medoid-action-targets-v1"
    onset_branches = source["schema"] == "i10-onset-branches-v1"
    schema = ("i10-onset-branch-feature-targets-v1" if onset_branches else
              "i10-medoid-action-feature-targets-v1" if medoids else "i10-actual-action-feature-targets-v1")
    if manifest["schema"] != schema or manifest["source_manifest"] != source:
        raise ValueError("wrong source or feature schema")
    if (manifest["sample_rate"], manifest["nfft"], manifest["hop_samples"], manifest["kernel_align"], manifest["power_mode"]) != (48000, 2048, 512, "right", "coherent"):
        raise ValueError("unregistered feature acquisition")
    if manifest["body_config"] != dict(means=[0.]*6, deviations=[1.]*6, accent_means=[0.]*2, accent_deviations=[1.]*2):
        raise ValueError("changed body scales")
    if manifest["space"] != dict(fmin=55., fmax=8000., bins_per_octave=96) or manifest["rt_config"] != dict(tau_min=.005, tau_max=.020, f_ref=200.):
        raise ValueError("changed NSGT grid or time constants")
    coordinates = np.asarray(manifest["centers_log2"])
    grid = (np.float32(np.log2(55.)) + np.arange(690, dtype=np.float32) * np.float32(1/96)).astype(float)
    if not np.array_equal(coordinates, grid):
        raise ValueError("wrong Log2Space coordinates")
    hop, future_hops = 512, 375
    expected_prefix_rows = 0
    case_issues = []
    counts = dict(cases=0, branches=0, prefix_rows=0, future_rows=0, raw_coordinates=0,
                  body_publications=0, equivalent_pairs=0, delayed_prefix_pairs=0, routed_zero_streams=0)
    max_error = 0.
    hashes = {"manifest.json": digest(output / "manifest.json")}
    case_names = [c['id'] for c in source['cases'] if c['status'] == 'acquired'] if onset_branches else source['cases']
    for name in case_names:
        profiles = json.loads((inputs / name / ("branches.json" if onset_branches else "profiles.json")).read_text())
        issue = profiles['render_start'] if onset_branches else profiles["issue_sample"] if medoids else source["issue_sample"]
        decision = profiles['decision_sample'] if onset_branches else issue
        future_hops = (profiles['render_end']-issue)//hop if onset_branches else 375
        prefix_hops = issue // hop
        if issue % hop:
            raise ValueError("issue not on the physical hop grid")
        expected_prefix_rows += prefix_hops * 2
        case_issues.append(dict(case=name, issue_sample=issue, decision_sample=decision,
                               future_samples=future_hops*hop, first_fully_future_hop=(decision+hop-1)//hop*hop)
                           if onset_branches else dict(case=name, issue_sample=issue))
        saved = {}
        prefix_rows, prefix_scans = [], []
        for bus in range(2):
            row_path = output / name / f"prefix-bus{bus}.jsonl"
            scan_path = row_path.with_suffix(".f64le")
            prefix_rows.append([json.loads(line) for line in row_path.read_text().splitlines()])
            prefix_scans.append(np.fromfile(scan_path, dtype="<f8").reshape(prefix_hops, len(coordinates)))
            for path in (row_path, scan_path):
                hashes[str(path.relative_to(output))] = digest(path)
            counts["prefix_rows"] += len(prefix_rows[-1])
        prefix_pcm = [np.fromfile(inputs / name / (f"prefix-bus{bus}.f32le" if medoids or onset_branches else "prefix.f32le"), dtype="<f4").astype(float) for bus in range(2)]
        for branch in profiles["branches"]:
            cls = branch['input']['class'] if onset_branches else branch["action"]["class"] if medoids else branch["class"]
            offset = branch['input']['at']-decision if onset_branches else branch["offset"] if medoids else branch["candidate_sample"] - issue
            stem = f"{cls}-{offset}"
            if not medoids and not onset_branches and not branch["eligible"]:
                if any((output / name).glob(f"{stem}-bus*")):
                    raise ValueError("fabricated ineligible target")
                continue
            for bus in range(2):
                wave_path = f'{stem}-bus{bus}.f32le' if onset_branches else branch["pcm"][bus] if medoids else f"{stem}.f32le"
                wave = np.concatenate((prefix_pcm[bus], np.fromfile(inputs / name / wave_path, dtype="<f4").astype(float)))
                if len(wave) != (prefix_hops + future_hops) * hop or not np.isfinite(wave).all():
                    raise ValueError("incomplete or nonfinite input PCM")
                source_energy = np.mean(wave.reshape(-1, hop)**2, axis=1)
                row_path = output / name / f"{stem}-bus{bus}.jsonl"
                scan_path = row_path.with_suffix(".f64le")
                future = [json.loads(line) for line in row_path.read_text().splitlines()]
                if len(future) != future_hops or len(prefix_rows[bus]) != prefix_hops:
                    raise ValueError("incomplete feature timeline")
                rows = prefix_rows[bus] + future
                scan = np.concatenate((prefix_scans[bus], np.fromfile(scan_path, dtype="<f8").reshape(future_hops, len(coordinates))))
                if not np.isfinite(scan).all() or (scan < 0).any():
                    raise ValueError("invalid assigned spectrum")
                routed = profiles['routing']['to_habitat' if bus == 0 else 'to_presentation'] if onset_branches else profiles["routing"][bus] if medoids else branch["pcm"][bus] is not None
                energy = source_energy if routed else np.zeros_like(source_energy)
                actual_energy = np.array([r["energy"] for r in rows])
                if not np.isfinite(actual_energy).all() or not np.allclose(actual_energy, energy, rtol=2e-14, atol=1e-20):
                    raise ValueError("feature energy differs from private PCM")
                shape = np.arange(len(rows)) >= 3
                if [r["shape_supported"] for r in rows] != shape.tolist():
                    raise ValueError("NSGT warmup restarted or missing")
                mass = scan.sum(axis=1)
                if not np.allclose(mass[mass > 0], energy[mass > 0], rtol=2e-14, atol=1e-20):
                    raise ValueError("spectrum lost assigned source energy")
                if not routed:
                    if scan.any():
                        raise ValueError("unrouted bus acquired spectral energy")
                    counts["routed_zero_streams"] += 1
                expected = raw_reference(energy, scan, coordinates, shape)
                actual = np.array([r["raw"]["values"] for r in rows], dtype=float)
                if not np.array_equal(np.isnan(actual), np.isnan(expected)):
                    raise ValueError(f"raw feature support mismatch: {name}/{stem}/{bus}")
                if not np.allclose(actual, expected, equal_nan=True, rtol=2e-11, atol=2e-12):
                    raise ValueError(f"raw feature mismatch: {name}/{stem}/{bus}")
                max_error = max(max_error, float(np.nanmax(np.abs(actual-expected))))
                next_publish = 4800
                for index, row in enumerate(rows):
                    start, end = index*hop, (index+1)*hop
                    raw = row["raw"]
                    if (raw["group"] != dict(bus=bus, epoch=0, generation=1)
                            or [raw[k] for k in ("start", "end", "known_samples", "source_start", "source_end", "available_end")]
                            != [start, end, hop, max(0, end-2560), end, end]):
                        raise ValueError("raw source/time support changed")
                    if row["role"] != ("issue_prefix" if index < prefix_hops else "counterfactual_actual_audio_target"):
                        raise ValueError("issue input and actual-audio target confused")
                    if onset_branches and row['decision_position'] != (
                            'before_decision' if end <= decision else
                            'straddles_decision' if start < decision else 'after_decision'):
                        raise ValueError('decision support position changed')
                    record = row["body_descriptor"]
                    if (record is not None) != (end >= next_publish):
                        raise ValueError("body publication schedule restarted")
                    if record is not None:
                        if (record["source_id"] != (1 if medoids or onset_branches else source["source_id"]) or record["source_generation"] != 0
                                or record["body_generation"] != 1 or record["bus"] != bus or not record["active"]
                                or record["start"] != max(0, end-96000) or record["end"] != end or record["available"] != end):
                            raise ValueError("body publication identity or support changed")
                        next_publish = end + 4800
                        if index >= prefix_hops:
                            counts["body_publications"] += 1
                for path in (row_path, scan_path):
                    hashes[str(path.relative_to(output))] = digest(path)
                saved[(cls, offset, bus)] = (actual[prefix_hops:], scan[prefix_hops:], energy[prefix_hops:])
                counts["future_rows"] += len(future)
                counts["raw_coordinates"] += len(future)*10
            counts["branches"] += 1
        for bus in range(2):
            if onset_branches:
                offsets = source['registration']['offsets_samples'][1:]
                baseline = saved[('skip',0,bus)]
                pairs = [(baseline, values) for (cls,offset,b),values in saved.items()
                         if b == bus and cls in ['wait','continue']]
                pairs.extend((values,saved[('gap',offset,bus)]) for (cls,offset,b),values in saved.items()
                             if b == bus and cls == 'release' and ('gap',offset,bus) in saved)
            elif medoids:
                offsets = profiles["offset_samples"][1:]
                baseline = saved[("continue", 0, bus)]
                pairs = [(baseline, saved[("wait", offset, bus)]) for offset in offsets]
                pairs.append((baseline, saved[("skip", 0, bus)]))
                pairs.extend((saved[("release", offset, bus)], saved[("gap", offset, bus)]) for offset in profiles["offset_samples"])
            else:
                offsets = (6000, 24000)
                baseline = saved[("wait", 6000, bus)]
                pairs = [(baseline, saved[("wait", 24000, bus)]), (baseline, saved[("skip", 0, bus)])]
                if profiles["active_tones"]:
                    pairs.append((baseline, saved[("continue", 0, bus)]))
                    pairs.extend((saved[("release", offset, bus)], saved[("gap", offset, bus)]) for offset in (0, 6000, 24000))
            for left, right in pairs:
                if not all(np.array_equal(a, b, equal_nan=True) for a, b in zip(left, right)):
                    raise ValueError("equivalent actions acquired different features")
                counts["equivalent_pairs"] += 1
            for cls in ("delayed_onset", "release", "gap"):
                for offset in offsets:
                    if (cls, offset, bus) not in saved:
                        continue
                    before = (decision-issue+offset) // hop
                    if not all(np.array_equal(a[:before], b[:before], equal_nan=True)
                               for a, b in zip(baseline, saved[(cls, offset, bus)])):
                        raise ValueError("future action leaked into its prefix")
                    counts["delayed_prefix_pairs"] += 1
        both = (dict(to_habitat=True,to_presentation=True) if onset_branches else [True, True] if medoids else "both")
        if profiles["routing"] == both:
            for cls, offset, bus in saved:
                if bus == 0 and not all(np.array_equal(a, b, equal_nan=True) for a, b in zip(saved[(cls, offset, 0)], saved[(cls, offset, 1)])):
                    raise ValueError("identical bus copies diverged")
        counts["cases"] += 1
        print(f"verified {name}: {counts['branches']} branches", flush=True)
    expected = (4,71,544,53392) if onset_branches else (8, 1032, 1640, 774000) if medoids else (36, 396, 2088, 297000)
    if (counts["cases"], counts["branches"], counts["prefix_rows"], counts["future_rows"]) != expected or expected_prefix_rows != expected[2]:
        raise ValueError("wrong acquisition denominator")
    if (manifest["branches"], manifest["prefix_rows"], manifest["future_rows"], manifest["prefix_comparisons"]) != (*expected[1:], 134 if onset_branches else 2048 if medoids else 720):
        raise ValueError("incorrect acquisition summary")
    if (medoids or onset_branches) and manifest["case_issues"] != case_issues:
        raise ValueError("case issue times differ")
    return dict(schema="i10-onset-branch-feature-verification-v1" if onset_branches else "i10-medoid-action-feature-verification-v1" if medoids else "i10-actual-action-feature-verification-v1", counts=counts,
                max_raw_absolute_error=max_error, files_sha256=hashes,
                claim="all raw formulas, PCM energy, support and class comparisons; NSGT and six body-window values are not independently reimplemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--inputs", type=Path, help="verified medoid or onset-receipt targets; omit for the registered 36-case corpus")
    args = parser.parse_args()
    if args.inputs:
        inputs = args.inputs
        checked = json.loads((inputs / "verification.json").read_text())
        onset_branches = checked['schema'] == 'i10-onset-branches-verification-v1'
        if not onset_branches and (checked["schema"] != "i10-medoid-action-verification-v1" or checked["counts"]["branches"] != 1032):
            raise ValueError("unverified medoid action targets")
        hash_key = 'hashes' if onset_branches else 'files_sha256'
        registration = dict(manifest_sha256=digest(inputs / "manifest.json"), verification_sha256=digest(inputs / "verification.json"), verified_files=len(checked[hash_key]))
    else:
        hash_key = 'files_sha256'
        registration = json.loads(Path("docs/roadmap/temporal-dcc/i10-action-profiles-physical-grid.json").read_text())["runs"]["512"]
        inputs = Path(registration["directory"])
    for phase in ("before", "after"):
        if digest(inputs / "manifest.json") != registration["manifest_sha256"] or digest(inputs / "verification.json") != registration["verification_sha256"]:
            raise ValueError("changed input manifest or verification")
        hashes = json.loads((inputs / "verification.json").read_text())[hash_key]
        if len(hashes) != registration["verified_files"] or any(digest(inputs / p) != h for p, h in hashes.items()):
            raise ValueError("input artifact changed")
        if phase == "before":
            if args.collect:
                subprocess.run(["cargo", "test", "--lib", "temporal_cognition::body::action_targets::acquire_action_feature_targets", "--", "--ignored", "--exact", "--nocapture"],
                               env=dict(os.environ, CONCHORDAL_I10_PROFILE_DIR=str(inputs.resolve()),
                                        CONCHORDAL_I10_FEATURE_DIR=str(args.output.resolve())), check=True)
            summary = verify(inputs, args.output)
    summary["input_manifest_sha256"] = registration["manifest_sha256"]
    summary["input_verification_sha256"] = registration["verification_sha256"]
    (args.output / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "files_sha256"}, indent=2))
