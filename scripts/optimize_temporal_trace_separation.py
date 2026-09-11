"""Cost-minimal additions against profiled unsaturated private-trace alternatives."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import linprog

import evaluate_temporal_trace_recovery as recovery
import plan_temporal_trace_separation as separation
import temporal_cognition_reference as reference


def allocate_counts(cost, baseline, indices, distances, target):
    """Retain baseline counts and solve the finite alternative-cut relaxation."""
    fit = linprog(cost[indices] / cost[indices].mean(), A_ub=-distances[:, indices],
                  b_ub=distances @ baseline - target, bounds=(0, None), method="highs")
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError(f"separation allocation failed: {fit.message}")
    counts = baseline.copy()
    counts[indices] = 2 * np.ceil((baseline[indices] + np.maximum(0, fit.x)) / 2).astype(int)
    if np.min(distances @ counts) < target - 1e-8:
        raise ValueError("rounded allocation violates its separation cuts")
    return counts


def plan(planning_path, output):
    planning = json.loads(planning_path.read_text())
    base = Path(planning["base_design"]["path"])
    if hashlib.sha256(base.read_bytes()).hexdigest() != planning["base_design"]["sha256"]:
        raise ValueError("base design hash mismatch")
    design = json.loads(base.read_text())
    output.mkdir(exist_ok=False, parents=True)
    for source, name in [(planning_path, "planning.json"), (base, "base-design.json"),
                         (Path(__file__), "optimizer.py"), (Path(separation.__file__), "plan_temporal_trace_separation.py"),
                         (Path(recovery.__file__), "evaluate_temporal_trace_recovery.py"),
                         (Path(reference.__file__), "temporal_cognition_reference.py")]:
        shutil.copyfile(source, output / name)
    started, cpu_started = time.monotonic(), time.process_time()
    rows = recovery.cells(design, design["conditions"][0])
    baseline = np.asarray(design["probes_per_cell"], dtype=int)
    if rows.shape[1] != 5 or baseline.shape != (len(rows),) or np.any(baseline < 2) or np.any(baseline % 2):
        raise ValueError("registered full-support cells and even baseline counts required")
    indices = np.flatnonzero(rows[:, 4] == 1)
    prepared = {}
    for condition in design["conditions"]:
        if not np.array_equal(rows, recovery.cells(design, condition)):
            raise ValueError("shared allocation requires identical condition cells")
        if condition["name"] in {point[0] for point in planning["truth_points"]}:
            prepared[condition["name"]] = (condition, recovery.emission_geometry(design, condition))
    costs = recovery.replay_cost_seconds(design, rows)
    counts, cuts, rounds, selected = baseline.copy(), [], [], None
    status = "round_budget_exhausted"
    for iteration in range(planning["maximum_profile_rounds"]):
        profiles = []
        try:
            for condition_name, truth_id in planning["truth_points"]:
                condition, geometry = prepared[condition_name]
                profiles.append(separation.profile_point(design, condition, truth_id, rows, geometry,
                                                         counts, planning, started, cpu_started))
        except TimeoutError:
            status = "compute_budget_exhausted"
            rounds.append({"iteration": iteration, "counts": counts.tolist(), "profiles": profiles,
                           "passed": False, "incomplete_profiles": True})
            break
        passed = all(profile["passed"] for profile in profiles)
        hours = float(costs @ counts / 3600)
        rounds.append({"iteration": iteration, "counts": counts.tolist(), "profiles": profiles,
                       "full_prefix_replay_hours_per_condition": hours, "passed": passed})
        print(json.dumps({"iteration": iteration, "probes": int(counts.sum()), "replay_hours": hours,
                          "distances": [p["minimum_profiled_distance"] for p in profiles], "passed": passed}), flush=True)
        if passed:
            selected, status = counts, "selected_by_finite_multistart_gate"
            break
        if not all(profile["valid"] for profile in profiles):
            status = "unresolved_profile_fit"
            break
        if iteration + 1 == planning["maximum_profile_rounds"]:
            break
        for profile in profiles:
            condition, geometry = prepared[profile["condition"]]
            truth = recovery.forward(np.log(profile["truth"]), design, condition, rows, geometry)[0]
            best = min((a for a in profile["attempts"] if a["valid"]), key=lambda a: a["distance"])
            alternative = [*best["tau_kappa"], 2 * max(design["prefix_counts"])]
            probability = recovery.forward(np.log(alternative), design, condition, rows, geometry)[0]
            affinity = np.sqrt(truth * probability).sum(axis=1)
            distance = -np.log(np.minimum(1.0, affinity))
            if not np.all(np.isfinite(distance)) or abs(distance @ counts - best["distance"]) > 1e-8:
                raise ValueError("alternative cut disagrees with its profiled distance")
            cuts.append(distance)
        try:
            counts = allocate_counts(costs, baseline, indices, np.asarray(cuts), planning["linear_cut_target"])
        except ValueError as error:
            status = str(error)
            break
        if float(costs @ counts / 3600) > planning["maximum_replay_hours_per_condition"]:
            status = "replay_cost_limit_exceeded"
            break
    result = {"scope": planning["scope"], "status": status, "rounds": rounds,
              "selected_counts": selected.tolist() if selected is not None else None,
              "wall_sec": time.monotonic() - started, "cpu_sec": time.process_time() - cpu_started,
              "timing_scope": "kernel preparation, profiles and LP; excludes archive/output IO"}
    np.savez_compressed(output / "alternative-cuts.npz", distances=np.asarray(cuts), costs=costs,
                        baseline=baseline, selected_indices=indices)
    (output / "separation.json").write_text(json.dumps(result, indent=2) + "\n")
    if selected is None:
        return
    design["schema"] = f"temporal-dcc-private-recovery-design-v{planning['study_version']}"
    design["selected_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    design["seed"] = planning["full_recovery"]["seed"]
    design["probes_per_cell"] = selected.tolist()
    design["high_cap_separation"] = {"path": str(output / "separation.json"),
        "sha256": hashlib.sha256((output / "separation.json").read_bytes()).hexdigest()}
    for item in design["allocation_cost"]:
        item.update(physical_probes=int(selected.sum()), probes_per_head=int(selected.sum() // 2),
                    independent_full_prefix_replay_hours_lower_bound=float(costs @ selected / 3600),
                    probe_windows_only_hours=float(selected.sum() * 4 / 3600))
    (output / "selected-design.json").write_text(json.dumps(design, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan(args.planning, args.output)
