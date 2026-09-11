"""Execute the registered multi-condition Fisher allocation, without responses."""

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


def covariance_constraints(counts, information):
    inverse = np.linalg.inv(np.einsum("i,tipq->tpq", counts, information))
    diagonal = np.diagonal(inverse, axis1=1, axis2=2)
    jacobian = -np.einsum("tap,tipq,tqa->tai", inverse, information, inverse)
    return diagonal, jacobian


def optimize(planning_path, fit_path, output):
    planning = json.loads(planning_path.read_text())
    fit_revision = json.loads(fit_path.read_text()) if fit_path else None
    base_path = Path(planning["base_design"]["path"])
    if hashlib.sha256(base_path.read_bytes()).hexdigest() != planning["base_design"]["sha256"]:
        raise ValueError("base design hash mismatch")
    design = json.loads(base_path.read_text())
    output.mkdir(exist_ok=False, parents=True)
    for path, name in [(planning_path, "planning.json"),
                       (base_path, "base-design.json"), (Path(__file__), "optimizer.py"),
                       (Path(recovery.__file__), "evaluate_temporal_trace_recovery.py"),
                       (Path(reference.__file__), "temporal_cognition_reference.py")]:
        shutil.copyfile(path, output / name)
    if fit_path:
        shutil.copyfile(fit_path, output / "fit-revision.json")
    started, cpu_start = time.monotonic(), time.process_time()
    information, metadata, baseline_rows = [], [], None
    for condition in design["conditions"]:
        rows = recovery.cells(design, condition)
        if baseline_rows is None:
            baseline_rows = rows
        elif not np.array_equal(rows, baseline_rows):
            raise ValueError("registered shared allocation needs identical exposure cells")
        geometry = recovery.emission_geometry(design, condition)
        for truth_id, truth in enumerate(design["truth_grid"]):
            p, gradient = recovery.forward(np.log(truth), design, condition, rows, geometry)
            inverse = np.divide(1, p, out=np.zeros_like(p), where=p > 0)
            information.append(np.einsum("crp,crq,cr->cpq", gradient, gradient, inverse))
            metadata.append([condition["name"], truth_id])
    information = np.asarray(information)
    limits = np.square(planning.get("maximum_log_parameter_se", [0.2, 0.2, 0.2]))
    minimum = planning.get("minimum_probes_per_cell", 2)
    diagonal, _ = covariance_constraints(np.ones(len(baseline_rows)), information)
    start_count = max(minimum, 2 * math.ceil(float(np.max(diagonal / limits)) / 2))
    costs = recovery.replay_cost_seconds(design, baseline_rows)
    cost_weights = costs / costs.sum()
    cache = {}

    def constraint(x):
        if time.monotonic() - started > planning["budget"]["wall_sec"] or time.process_time() - cpu_start > planning["budget"]["cpu_sec"]:
            raise TimeoutError("registered allocation budget exhausted")
        if "x" not in cache or not np.array_equal(cache["x"], x):
            diagonal, derivative = covariance_constraints(start_count * x, information)
            cache.update(x=x.copy(), value=(1 - diagonal / limits).ravel(),
                         jac=(-start_count * derivative / limits[None, :, None]).reshape(-1, len(x)))
        return cache

    try:
        result = minimize(lambda x: float(cost_weights @ x), np.ones(len(baseline_rows)),
                          jac=lambda x: cost_weights, method="SLSQP", bounds=[(minimum / start_count, None)] * len(baseline_rows),
                          constraints={"type": "ineq", "fun": lambda x: constraint(x)["value"],
                                       "jac": lambda x: constraint(x)["jac"]},
                          options={key: planning["solver"][key] for key in ("maxiter", "ftol")})
        raw = start_count * result.x
        raw_diagonal, _ = covariance_constraints(raw, information)
        valid = (np.all(np.isfinite(raw)) and np.all(raw >= minimum - 1e-8)
                 and np.max(raw_diagonal / limits) <= 1 + 1e-7
                 and costs @ raw <= costs.sum() * start_count)
        status = {"success": bool(result.success), "message": str(result.message), "iterations": int(result.nit),
                  "raw_feasible_lower_cost": bool(valid), "raw_max_variance": float(np.max(raw_diagonal))}
    except TimeoutError as error:
        raw, valid = np.full(len(baseline_rows), start_count), False
        status = {"success": False, "message": str(error), "raw_feasible_lower_cost": False}
    counts = (2 * np.ceil(raw / 2)).astype(int) if valid else np.full(len(baseline_rows), start_count, int)
    for _ in range(3):
        final_diagonal, _ = covariance_constraints(counts, information)
        ratio = float(np.max(final_diagonal / limits))
        if ratio <= 1:
            break
        counts = (2 * np.ceil(counts * ratio * (1 + 1e-9) / 2)).astype(int)
    if ratio > 1 or costs @ counts > costs.sum() * start_count:
        counts = np.full(len(baseline_rows), start_count, int)
        final_diagonal, _ = covariance_constraints(counts, information)
        status["rounded_fallback"] = True
    np.savez_compressed(output / "information.npz", information=information, cells=baseline_rows, costs=costs)
    result = {"kind": "expected_fisher_cost_allocation_not_recovery", "registered_plan": str(planning_path),
              "condition_truth_order": metadata, "start_uniform_count": start_count, "counts": counts.tolist(),
              "total_probes_per_condition": int(counts.sum()), "minimum_per_cell": int(counts.min()),
              "maximum_log_parameter_se": np.sqrt(final_diagonal).max(axis=0).tolist(),
              "full_prefix_replay_hours_per_condition": float(costs @ counts / 3600),
              "base_hours_per_condition": design.get("allocation_cost", [{}])[0].get("independent_full_prefix_replay_hours_lower_bound"),
              "feasible_uniform_start_hours_per_condition": float(costs.sum() * start_count / 3600),
              "optimizer": status, "wall_sec": time.monotonic() - started, "cpu_sec": time.process_time() - cpu_start}
    (output / "optimization.json").write_text(json.dumps(result, indent=2) + "\n")
    design["schema"] = f"temporal-dcc-private-recovery-design-v{planning.get('study_version', 2)}"
    design["selected_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    design["probes_per_cell"] = counts.tolist()
    design["allocation_optimization"] = {"path": str(output / "optimization.json"),
        "sha256": hashlib.sha256((output / "optimization.json").read_bytes()).hexdigest()}
    design["seed"] = planning.get("seed", 2026091102)
    if fit_revision:
        design["fit"]["nonsmooth_fallback"] = fit_revision
    if "allocation_cost" not in design:
        design["allocation_cost"] = [{"condition": row["name"], "cells": len(baseline_rows)} for row in design["conditions"]]
    for row in design["allocation_cost"]:
        row.update(physical_probes=int(counts.sum()), probes_per_head=int(counts.sum() // 2),
                   independent_full_prefix_replay_hours_lower_bound=result["full_prefix_replay_hours_per_condition"],
                   probe_windows_only_hours=float(counts.sum() * 4 / 3600))
    (output / "selected-design.json").write_text(json.dumps(design, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["total_probes_per_condition", "minimum_per_cell", "maximum_log_parameter_se",
          "full_prefix_replay_hours_per_condition", "base_hours_per_condition", "optimizer", "wall_sec"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning", type=Path, required=True)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    optimize(args.planning, args.fit, args.output)
