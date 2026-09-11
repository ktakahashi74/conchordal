"""Select registered private-trace probes against the unsaturated alternative."""

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import minimize

import evaluate_temporal_trace_recovery as recovery
import temporal_cognition_reference as reference


def profile_distance(parameters, truth_probability, counts, design, condition, rows, geometry):
    log_parameters = np.r_[parameters, math.log(2 * max(design["prefix_counts"]))]
    probability, gradient = recovery.forward(log_parameters, design, condition, rows, geometry)
    affinity = np.sqrt(truth_probability * probability).sum(axis=1)
    if np.any(affinity <= 0):
        return math.inf, np.zeros(2)
    ratio = np.sqrt(np.divide(truth_probability, probability,
                             out=np.zeros_like(probability), where=probability > 0))
    derivative = -0.5 * np.einsum("cr,crp,c->p", ratio, gradient[:, :, :2], counts / affinity)
    return float(-counts @ np.log(affinity)), derivative


def profile_point(design, condition, truth_id, rows, geometry, counts, planning, started, cpu_started):
    """Use the same registered multistart and unresolved-better-fit rule in both planners."""
    bounds = np.log(design["fit"]["log_bounds"][:2])
    truth = design["truth_grid"][truth_id]
    truth_probability = recovery.forward(np.log(truth), design, condition, rows, geometry)[0]
    starts = planning["profile_starts"] + ([truth[:2]] if planning["include_truth_tau_kappa_start"] else [])
    attempts = []
    for start in starts:
        if (time.monotonic() - started > planning["budget"]["wall_sec"]
                or time.process_time() - cpu_started > planning["budget"]["cpu_sec"]):
            raise TimeoutError("registered separation planning budget exhausted")
        fit = minimize(profile_distance, np.log(start),
                       args=(truth_probability, counts, design, condition, rows, geometry),
                       jac=True, bounds=bounds, method=planning["solver"]["method"],
                       options={k: planning["solver"][k] for k in ("maxiter", "maxls", "ftol", "gtol")})
        projected = fit.jac.copy()
        projected[(fit.x <= bounds[:, 0] + 1e-8) & (projected > 0)] = 0
        projected[(fit.x >= bounds[:, 1] - 1e-8) & (projected < 0)] = 0
        attempts.append({"start": start, "tau_kappa": np.exp(fit.x).tolist(), "distance": float(fit.fun),
                         "valid": bool(fit.success and np.isfinite(fit.fun) and np.max(np.abs(projected)) < 1e-5),
                         "gradient_max": float(np.max(np.abs(projected)))})
    valid = [item for item in attempts if item["valid"]]
    best = min(valid, key=lambda item: item["distance"]) if valid else None
    unresolved = best is None or any(item["distance"] < best["distance"] - 1e-8 for item in attempts)
    return {"condition": condition["name"], "truth_id": truth_id, "truth": truth, "attempts": attempts,
            "minimum_profiled_distance": best["distance"] if best else None,
            "valid": not unresolved,
            "passed": bool(not unresolved and best["distance"] >= planning["minimum_total_distance"])}


def plan(planning_path, output):
    planning = json.loads(planning_path.read_text())
    base = Path(planning["base_design"]["path"])
    if hashlib.sha256(base.read_bytes()).hexdigest() != planning["base_design"]["sha256"]:
        raise ValueError("base design hash mismatch")
    design = json.loads(base.read_text())
    output.mkdir(exist_ok=False, parents=True)
    for source, name in [(planning_path, "planning.json"), (base, "base-design.json"),
                         (Path(__file__), "planner.py"), (Path(recovery.__file__), "evaluate_temporal_trace_recovery.py"),
                         (Path(reference.__file__), "temporal_cognition_reference.py")]:
        shutil.copyfile(source, output / name)
    points = planning["truth_points"] if "truth_points" in planning else [[planning["condition"], tid] for tid in planning["truth_ids"]]
    rows = recovery.cells(design, design["conditions"][0])
    prepared = {}
    for other in design["conditions"]:
        if not np.array_equal(rows, recovery.cells(design, other)):
            raise ValueError("shared allocation requires identical exposure cells")
        if other["name"] in {point[0] for point in points}:
            prepared[other["name"]] = (other, recovery.emission_geometry(design, other))
    if "count_factors" in planning:
        indices = np.flatnonzero(rows[:, 4] == 1)
        choices = planning["count_factors"]
        allocation_key = "fully_supported_prefix_count_factor"
    else:
        indices = [np.flatnonzero(np.all(rows == cell, axis=1)).item() for cell in planning["calibration_cells"]]
        choices = planning["count_candidates"]
        allocation_key = "count_per_calibration_cell"
    baseline = np.asarray(design["probes_per_cell"])
    started, cpu_started = time.monotonic(), time.process_time()
    candidates, selected = [], None
    for count in choices:
        counts = baseline.copy()
        counts[indices] = counts[indices] * count if "count_factors" in planning else np.maximum(counts[indices], count)
        profiles = []
        for condition_name, truth_id in points:
            condition, geometry = prepared[condition_name]
            profiles.append(profile_point(design, condition, truth_id, rows, geometry, counts,
                                          planning, started, cpu_started))
        passed = all(row["passed"] for row in profiles)
        candidates.append({allocation_key: count, "profiles": profiles, "passed": passed})
        print(json.dumps({allocation_key: count, "passed": passed,
                          "distances": [row["minimum_profiled_distance"] for row in profiles]}), flush=True)
        if passed:
            selected = counts
            break
    result = {"scope": "Finite multistart separation heuristic; not a global certificate, recovery pass or empirical allocation.",
              "candidates": candidates, "selected_counts": selected.tolist() if selected is not None else None,
              "wall_sec": time.monotonic() - started, "cpu_sec": time.process_time() - cpu_started}
    (output / "separation.json").write_text(json.dumps(result, indent=2) + "\n")
    if selected is None:
        return
    design["schema"] = f"temporal-dcc-private-recovery-design-v{planning.get('study_version', 6)}"
    design["selected_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    design["seed"] = planning["full_recovery"]["seed"]
    design["probes_per_cell"] = selected.tolist()
    design["high_cap_separation"] = {"path": str(output / "separation.json"),
        "sha256": hashlib.sha256((output / "separation.json").read_bytes()).hexdigest()}
    cost = recovery.replay_cost_seconds(design, rows)
    for item in design["allocation_cost"]:
        item.update(physical_probes=int(selected.sum()), probes_per_head=int(selected.sum() // 2),
                    independent_full_prefix_replay_hours_lower_bound=float(cost @ selected / 3600),
                    probe_windows_only_hours=float(selected.sum() * 4 / 3600))
    (output / "selected-design.json").write_text(json.dumps(design, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan(args.planning, args.output)
