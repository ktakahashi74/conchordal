"""Offline quadrature audit against frozen counterfactual PCM energy windows.

Future rendered energy is a fidelity target, never a live predictor or teacher.
Requires NumPy; reads the verified action-profile corpus without changing it.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform

import numpy as np


EPSILON = 1e-12
LEGENDRE = {}


class Curve:
    def __init__(self, start, width, values):
        self.start, self.width = start, width
        self.values = np.asarray(values, dtype=float)
        if width <= 0 or self.values.ndim != 1 or not len(self.values):
            raise ValueError("invalid energy grid")
        self.valid = np.isfinite(self.values) & (self.values >= 0)
        self.end = start + width * len(self.values)
        self.centers = start + width * (np.arange(len(self.values)) + 0.5)

    def at(self, times):
        times = np.asarray(times, dtype=float)
        indices = np.floor((times - self.start) / self.width).astype(int)
        clipped = np.clip(indices, 0, len(self.values) - 1)
        supported = (times >= self.start) & (times < self.end) & self.valid[clipped]
        center = self.centers[clipped]
        neighbors = clipped + np.where(times < center, -1, 1)
        adjacent = np.clip(neighbors, 0, len(self.values) - 1)
        connected = ((neighbors >= 0) & (neighbors < len(self.values))
                     & self.valid[adjacent])
        fraction = np.abs(times - center) / self.width
        result = self.values[clipped].copy()
        result = np.where(connected,
                          result * (1 - fraction) + self.values[adjacent] * fraction,
                          result)
        return np.where(supported, result, np.nan)

    def covers(self, begin, end):
        if end <= begin or begin < self.start or end > self.end:
            return False
        first = int(np.floor((begin - self.start) / self.width))
        last = int(np.ceil((end - self.start) / self.width))
        return bool(self.valid[first:last].all())

    def silent(self, begin, end):
        if not self.covers(begin, end):
            return False
        # Neighbor centers can influence the requested interval.
        first = max(0, int(np.floor((begin - self.start) / self.width)) - 1)
        last = min(len(self.values), int(np.ceil((end - self.start) / self.width)) + 1)
        return bool((self.values[first:last] == 0).all())

    def knots(self, begin, end):
        edges = self.start + np.arange(len(self.values) + 1) * self.width
        nodes = np.concatenate((edges, self.centers))
        return nodes[(nodes > begin) & (nodes < end)]


def ratios(own, external, weights, known_silent):
    own = np.asarray(own, dtype=float)
    external = np.asarray(external, dtype=float)
    weights = np.broadcast_to(weights, own.shape)
    if np.any(~np.isfinite(own)) or np.any(own < 0):
        return {"status": "unsupported", "overlap": None, "audibility": None,
                "own_integral": None}
    integral = float(np.sum(own * weights))
    if integral == 0:
        return {"status": "known_silent" if known_silent else "missed_own_energy",
                "overlap": 0.0 if known_silent else None, "audibility": None,
                "own_integral": 0.0}
    if np.any(~np.isfinite(external)) or np.any(external < 0):
        return {"status": "unsupported", "overlap": None, "audibility": None,
                "own_integral": integral}
    denominator = own + external + EPSILON
    return {"status": "supported", "own_integral": integral,
            "overlap": float(np.sum(weights * own * (external / denominator)) / integral),
            "audibility": float(np.sum(weights * own * (own / denominator)) / integral)}


def integrate(own, external, begin, end, count, dense=False):
    if end <= begin:
        return {"status": "empty_intersection", "overlap": None, "audibility": None,
                "own_integral": None}
    if dense:
        boundaries = np.unique(np.concatenate(([begin, end], own.knots(begin, end),
                                               external.knots(begin, end))))
        if count not in LEGENDRE:
            LEGENDRE[count] = np.polynomial.legendre.leggauss(count)
        nodes, factors = LEGENDRE[count]
        lengths = np.diff(boundaries)
        times = (boundaries[:-1, None] + lengths[:, None] * (nodes + 1) / 2).ravel()
        weights = (lengths[:, None] * factors / 2).ravel()
    else:
        times = begin + (np.arange(count) + 0.5) * (end - begin) / count
        weights = (end - begin) / count
    result = ratios(own.at(times), external.at(times), weights, own.silent(begin, end))
    # Point sampling cannot establish continuous support across a missing window.
    if not own.covers(begin, end) or (not external.covers(begin, end)
                                    and result["status"] != "known_silent"):
        result.update(status="unsupported", overlap=None, audibility=None)
    return result


def support(manifest, row, active):
    issue = manifest["issue_sample"]
    release = round(manifest["adsr"][3] * manifest["sample_rate"])
    spans = []
    if active:
        for tone in manifest["prefix_tones"]:
            hold = tone["onset"] + tone["hold"]
            if row["class"] in ("release", "gap"):
                hold = min(hold, row["candidate_sample"])
            spans.append((issue, hold + release))
    if row["class"] in ("onset_now", "delayed_onset"):
        start = row["candidate_sample"]
        spans.append((start, start + manifest["new_onset"]["hold"] + release))
    if not spans:
        return issue, issue + manifest["horizon_samples"]
    return min(a for a, _ in spans), max(b for _, b in spans)


def external_conditions(manifest):
    start, width = manifest["issue_sample"], manifest["hop_samples"]
    size = manifest["horizon_samples"] // width
    def curve(values):
        return Curve(start, width, values)
    burst = np.zeros(size)
    burst[12:14] = 0.1
    missing = np.full(size, 0.001)
    missing[12] = np.nan
    return {
        "silence": curve(np.zeros(size)),
        "quiet": curve(np.full(size, 1e-5)),
        "matched_scale": curve(np.full(size, 0.001)),
        "loud": curve(np.full(size, 0.1)),
        "ramp": curve(np.linspace(0, 0.02, size)),
        "burst_120_140ms": curve(burst),
        "missing_120_130ms": curve(missing),
        "half_second_support": curve(np.full(50, 0.001)),
    }


def evaluate(root, output):
    registered = json.loads((root / "verification.json").read_text())
    for name, expected in registered["files_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"verified input changed: {name}")
    registry_path = Path(__file__).parents[1] / "docs/roadmap/temporal-dcc/i10-action-profiles.json"
    frozen = json.loads(registry_path.read_text())
    if registered != frozen["verification"]:
        raise ValueError("corpus verification differs from the frozen I10 registry")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] != "i10-actual-action-profiles-v1":
        raise ValueError("unregistered acquisition")
    conditions = external_conditions(manifest)
    rows, comparisons, ineligible = [], [], 0
    max_dense_error = 0.0
    strata = {}
    errors = {str(n): {key: [] for key in ("overlap", "audibility")} for n in (8, 16, 32)}
    for case in manifest["cases"]:
        profile = json.loads((root / case / "profiles.json").read_text())
        groups = {}
        for branch in profile["branches"]:
            if not branch["eligible"]:
                ineligible += 1
                continue
            requested = support(manifest, branch, profile["active_tones"] > 0)
            for bus, values in enumerate(branch["energy_10ms"]):
                own = Curve(manifest["issue_sample"], manifest["hop_samples"], values)
                if not own.covers(*requested):
                    raise ValueError("actual profile does not cover declared envelope")
                full = integrate(own, own, *requested, 8, dense=True)["own_integral"]
                for name, external in conditions.items():
                    begin, end = max(requested[0], external.start), min(requested[1], external.end)
                    reference = integrate(own, external, begin, end, 16, dense=True)
                    check = integrate(own, external, begin, end, 8, dense=True)
                    order = 16
                    while True:
                        change = max((abs(reference[column] - check[column])
                                      for column in ("overlap", "audibility")
                                      if reference[column] is not None and check[column] is not None), default=0.)
                        if change <= 1e-10 or order == 512:
                            break
                        check = reference
                        order *= 2
                        reference = integrate(own, external, begin, end, order, dense=True)
                    max_dense_error = max(max_dense_error, change)
                    approximations = {str(n): integrate(own, external, begin, end, n)
                                      for n in (8, 16, 32)}
                    clipped_own = 0.0 if end <= begin else reference["own_integral"]
                    row = {"case": case, "body": profile["body"], "modulator": profile["modulator"], "bus": bus,
                           "class": branch["class"], "candidate_sample": branch["candidate_sample"],
                           "external": name, "requested": requested,
                           "intersection": [begin, end] if end > begin else None,
                           "known_silent": own.silent(*requested), "reference": reference,
                           "reference_order": order, "reference_change": change,
                           "reference_converged": change <= 1e-10,
                           "approximation": approximations,
                           "excluded_own_integral_sample_units": None if clipped_own is None
                               else max(0.0, full - clipped_own),
                           "excluded_own_mass_fraction": None if not full or clipped_own is None
                               else max(0.0, 1 - clipped_own / full)}
                    rows.append(row)
                    groups.setdefault((bus, name), []).append(row)
                    for n, estimate in approximations.items():
                        for column in ("overlap", "audibility"):
                            if reference[column] is not None and estimate[column] is not None:
                                errors[n][column].append((abs(reference[column] - estimate[column]),
                                                         len(rows) - 1))
                                key = (profile["body"], profile["modulator"], branch["class"], n, column)
                                strata.setdefault(key, []).append(errors[n][column][-1])
        for (bus, external), candidates in groups.items():
            for n, column in itertools.product((8, 16, 32), ("overlap", "audibility")):
                counts = dict(pairs=0, inversions=0, dense_ties=0, midpoint_ties=0, unsupported=0,
                              inversions_above_1e_6=0, inversions_above_1e_3=0)
                for a, b in itertools.combinations(candidates, 2):
                    values = [a["reference"][column], b["reference"][column],
                              a["approximation"][str(n)][column], b["approximation"][str(n)][column]]
                    counts["pairs"] += 1
                    if any(v is None for v in values):
                        counts["unsupported"] += 1
                        continue
                    dense, sampled = values[0] - values[1], values[2] - values[3]
                    counts["dense_ties"] += int(dense == 0)
                    counts["midpoint_ties"] += int(sampled == 0)
                    counts["inversions"] += int(dense * sampled < 0)
                    counts["inversions_above_1e_6"] += int(dense * sampled < 0 and abs(dense) > 1e-6)
                    counts["inversions_above_1e_3"] += int(dense * sampled < 0 and abs(dense) > 1e-3)
                comparisons.append(dict(case=case, bus=bus, external=external,
                                        points=n, column=column, **counts))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "rows.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    (output / "rank-comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    summary = {
        "schema": "i10-energy-quadrature-audit-v1",
        "corpus_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "corpus_verification_sha256": hashlib.sha256((root / "verification.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "epsilon_full_scale_squared": EPSILON, "cases": len(manifest["cases"]),
        "ineligible_branches": ineligible, "conditions": len(conditions), "rows": len(rows),
        "reference": "piecewise window-center/edge Gauss-Legendre 8/16, doubled to 512 until ratio change <= 1e-10",
        "max_reference_ratio_difference": max_dense_error,
        "unconverged_reference_rows": sum(not row["reference_converged"] for row in rows),
        "largest_reference_order": max(row["reference_order"] for row in rows),
        "strata_errors": [dict(body=key[0], modulator=key[1], action_class=key[2],
                               points=int(key[3]), column=key[4], compared=len(values),
                               max_absolute_error=max(values)[0], worst_row=max(values)[1])
                          for key, values in sorted(strata.items())],
        "errors": {n: {column: {"compared": len(values),
                    "max_absolute_error": max(values, default=(0., None))[0],
                    "worst_row": max(values, default=(0., None))[1],
                    "p95_absolute_error": float(np.quantile([v[0] for v in values], 0.95)) if values else None}
                    for column, values in columns.items()} for n, columns in errors.items()},
        "rank": {str(n): {column: {key: sum(r[key] for r in comparisons
                    if r["points"] == n and r["column"] == column)
                    for key in ("pairs", "inversions", "dense_ties", "midpoint_ties", "unsupported",
                                "inversions_above_1e_6", "inversions_above_1e_3")}
                    for column in ("overlap", "audibility")} for n in (8, 16, 32)},
        "artifacts_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                             for name in ("rows.jsonl", "rank-comparisons.json")},
        "claim": "offline ideal energy quadrature only; not live body prediction, calibration, promotion or full I10 acceptance",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps({k: v for k, v in evaluate(args.corpus, args.output).items()
                      if k != "strata_errors"}, indent=2))
