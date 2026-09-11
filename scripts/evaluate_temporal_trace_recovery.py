"""Preregistered physical-record recovery for the private temporal filter.

The fixed prefixes are experimental interventions, not samples falsely scored
as spontaneous outcomes. Each independent probe samples exactly one physical
record (or missing detection) from both frozen reference predictions together.
"""

import argparse
import concurrent.futures
import datetime
import hashlib
import itertools
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit

import temporal_cognition_reference as reference


def registration():
    spec = Path("docs/design-notes/dcc-neurocognitive-hierarchy.md")
    return {
        "schema": "temporal-dcc-private-recovery-planning-v1",
        "registered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "specification": {"path": str(spec), "sha256": hashlib.sha256(spec.read_bytes()).hexdigest()},
        "model": "retained-bin-filter-with-joint-physical-record-likelihood-v1",
        "state_parameters": ["tau_sec", "kappa", "strength_max"],
        "truth_grid": list(itertools.product([2, 20, 200, 1200], [1, 4, 16], [1.5, 3, 6])),
        "replicates": 100,
        "seed": 2026091101,
        "prefix_counts": [1, 4, 16, 64, 128],
        "delays_sec": [0.125, 1, 4, 16, 64, 256, 1024, 2048],
        "interference_units": [0, 1, 4, 16],
        "minimum_interference_spacing_sec": 0.0625,
        "interference_allocation": "Even onset/release pairs, each credit <= condition prefix credit; last pair may use fractional support. Require (event_count+1)*spacing <= delay.",
        "probe_window_sec": [0, 4],
        "sample_rate": 48000,
        "hop_samples": 512,
        "prefix_timestamp_width_sec": 512 / 48000,
        "prefix_geometry": {
            "nonperiodic": {"period_sec": 0.25, "repeat_sec": 0.0625, "first_bin": 4, "second_bin": 32},
            "periodic": {"period_sec": 4, "repeat_sec": 4, "first_bin": 4, "second_bin": 12}},
        "prefix_protocol": [
            "Begin with empty trace heads. Condition on N prescribed own outcomes assigned to one reference, with known uniform observed time support wholly inside the declared bin(s).",
            "Age through the independently chosen delay and total actually assigned competing-reference credit. Add one confirmed own outcome at a distinct relative timing bin (overflow for the nonperiodic reference).",
            "Train the nonperiodic and periodic references in separate successive prefixes. Later other-reference outcomes multiply the completed earlier trace uniformly, preserving its conditional bin distribution.",
            "Each targeted onset has a same-reference release counterpart, or a targeted release has a same-reference priming onset. Counterparts do not add to the targeted head or its competing-reference credit.",
            "Score one subsequently issued physical probe only. Its frozen inventory weights are independent of its timing; its eventual credit still uses observed support, never mixture responsibilities.",
            "Split the independent probe repetitions equally across onset/release target heads. They are separate fixed-policy runs, not two independent observations of one physical event."
        ],
        "conditions": [
            {"name": "complete", "observed_fraction": 1, "anchor_coverage": 1, "modes": 1, "skip_every": 0, "detection": 1, "gap_every": 0},
            {"name": "fractional", "observed_fraction": 0.5, "anchor_coverage": 1, "modes": 1, "skip_every": 0, "detection": 0.9, "gap_every": 0},
            {"name": "ambiguous_missing", "observed_fraction": 0.8, "anchor_coverage": 0.75, "modes": 2, "skip_every": 5, "detection": 0.85, "gap_every": 7}
        ],
        "probe_weights": {"nonperiodic": 0.45, "periodic": 0.35, "unassigned": 0.2},
        "probe_anchors": {"nonperiodic": [-0.13, -0.12], "periodic": [-0.005, 0.005]},
        "alternative_anchor_shift_sec": {"nonperiodic": 0.25, "periodic": 2},
        "planning": {"probes_per_cell_candidates": [16, 64, 256, 1024], "maximum_log_parameter_se": [0.2, 0.2, 0.2],
                     "selection": "Smallest allocation meeting every condition/truth expected Fisher SE, before random responses. No Monte Carlo if none qualifies."},
        "fit": {"log_bounds": [[0.5, 4800], [0.25, 64], [0.5, 24]],
                "starts": [[2, 1, 1.5], [80, 4, 3], [1200, 16, 6]],
                "method": "L-BFGS-B", "maxiter": 500, "ftol": 1e-12, "gtol": 1e-8,
                "maxls": 40, "regularization": "none", "objective": "mean categorical physical-record negative log likelihood, including missing detection",
                "failed": "no converged finite interior optimum; failed fits remain infinite errors"},
        "acceptance": {"median_abs_log_ratio_max": math.log(1.25), "p90_abs_log_ratio_max": math.log(2),
                       "scope": "Every positive parameter, truth and nuisance condition; no pooled substitute."},
        "compute_budget": {"wall_sec": 7200, "cpu_sec": 28800, "workers_max": 8, "blas_threads": 1},
        "scope_limits": ["Conditional simulator and engineered reference inputs; no waveform-derived matcher validation.",
                         "Actual body/opportunity feasibility, full rendering/delivery and real execution fits remain required.",
                         "The known timestamp intervals need not share a global hop origin in this engineering assay; actual acquisition-aligned replay remains required before empirical fitting.",
                         "Prefix interventions, detectors and baseline are fixed engineering choices, not fitted human cognition or a production action policy."],
        "human_collection_authorized": False,
    }


def cells(design, condition):
    credit = condition["observed_fraction"] * condition["anchor_coverage"]
    rows = []
    for count, delay, interference in itertools.product(design["prefix_counts"], design["delays_sec"], design["interference_units"]):
        events = 2 * math.ceil(interference / (2 * credit))
        if (events + 1) * design["minimum_interference_spacing_sec"] <= delay:
            if "second_prefix_counts" in design:
                rows.extend((count, delay, interference, second) for second in design["second_prefix_counts"])
            else:
                rows.append((count, delay, interference))
    if "fully_supported_first_prefix" in design:
        specification = design["fully_supported_first_prefix"]
        added = [(*row, 1) for row in rows if row[0] in specification["first_counts"]
                 and row[1] in specification["delays_sec"] and row[3] in specification["second_counts"]]
        rows = [(*row, 0) for row in rows] + added
    return np.asarray(rows, dtype=float)


def replay_cost_seconds(design, rows):
    extra = rows[:, 3] - 1 if rows.shape[1] >= 4 else 0
    return ((rows[:, 0] + extra) * sum(cfg["repeat_sec"] for cfg in design["prefix_geometry"].values())
            + 2 * rows[:, 1] + 4 + 0.05)


def emission_geometry(design, condition):
    window = tuple(design["probe_window_sec"])
    hop = design["hop_samples"] / design["sample_rate"]
    records = []
    for index in range(math.ceil((window[1] - window[0]) / hop)):
        if condition["gap_every"] and (index + 1) % condition["gap_every"] == 0:
            continue
        records.append({"interval": (window[0] + index * hop, min(window[1], window[0] + (index + 1) * hop)),
                        "detection": condition["detection"]})
    geometry = {}
    for family in ("nonperiodic", "periodic"):
        anchors = []
        for mode in range(condition["modes"]):
            shift = mode * design["alternative_anchor_shift_sec"][family]
            anchors.append({"weight": condition["anchor_coverage"] / condition["modes"],
                            "interval": tuple(v - shift for v in design["probe_anchors"][family]),
                            "period_sec": design["prefix_geometry"][family]["period_sec"]})
        kernel = reference.timing_observation_kernel(window, records, anchors, family == "periodic")
        matrix = np.asarray(kernel["matrix"])
        config = design["prefix_geometry"][family]
        first = [config["first_bin"]]
        second = [config["second_bin"]]
        if condition["modes"] == 2:
            first.append(first[0] + (8 if family == "nonperiodic" else 16))
            if family == "periodic":
                second.append(second[0] + 16)
        geometry[family] = {"first": matrix[:, first].mean(axis=1), "second": matrix[:, second].mean(axis=1),
                            "kernel": kernel, "first_bins": first, "second_bins": second}
    baseline = np.asarray(geometry["periodic"]["kernel"]["baseline"])
    fixed = design["probe_weights"]["unassigned"] * baseline
    differences = []
    for family in ("nonperiodic", "periodic"):
        weight = design["probe_weights"][family]
        fixed = fixed + weight * geometry[family]["second"]
        differences.append(weight * (geometry[family]["first"] - geometry[family]["second"]))
    return {"fixed": fixed, "differences": np.asarray(differences), "full": geometry}


def forward(log_parameters, design, condition, rows, geometry):
    tau, kappa, cap = np.exp(log_parameters)
    width = design["prefix_timestamp_width_sec"]
    g = -math.expm1(-width / tau) * tau / width
    dg = g - math.exp(-width / tau)
    credit = condition["observed_fraction"] * condition["anchor_coverage"]
    fractions, derivatives = [], []
    for family in ("nonperiodic", "periodic"):
        cfg = design["prefix_geometry"][family]
        n_first = len(geometry["full"][family]["first_bins"])
        n_second = len(geometry["full"][family]["second_bins"])
        a = math.exp(-cfg["repeat_sec"] / tau)
        prefix_masses, prefix_gradients = [], []
        second_counts = rows[:, 3] if rows.shape[1] >= 4 else np.ones(len(rows))
        first_credits = np.where(rows[:, 4] == 1, 1.0, credit) if rows.shape[1] == 5 else np.full(len(rows), credit)
        for n_bins, counts, credits in [(n_first, rows[:, 0], first_credits),
                                        (n_second, second_counts, np.full(len(rows), credit))]:
            masses, gradients = np.zeros(len(rows)), np.zeros((len(rows), 3))
            for assigned_credit in np.unique(credits):
                selected = credits == assigned_credit
                m, gradient = 0.0, np.zeros(3)
                saved = {}
                for step in range(1, int(max(counts[selected])) + 1):
                    old = m
                    m *= a
                    gradient = a * gradient
                    gradient[0] += a * old * cfg["repeat_sec"] / tau
                    if not condition["skip_every"] or step % condition["skip_every"]:
                        m += assigned_credit * g / n_bins
                        gradient[0] += assigned_credit * dg / n_bins
                        if m >= cap:
                            m, gradient = cap, np.asarray([0.0, 0.0, cap])
                    saved[step] = (m, gradient.copy())
                masses[selected] = [saved[int(count)][0] for count in counts[selected]]
                gradients[selected] = [saved[int(count)][1] for count in counts[selected]]
            prefix_masses.append(masses)
            prefix_gradients.append(gradients)
        initial, second = prefix_masses
        initial_gradient, second_gradient = prefix_gradients
        elapsed = rows[:, 1] + (second_counts - 1) * cfg["repeat_sec"]
        logits = np.log(initial * n_first) - elapsed / tau - rows[:, 2] / kappa - np.log(second * n_second)
        derivative = initial_gradient / initial[:, None] - second_gradient / second[:, None]
        derivative[:, 0] += elapsed / tau
        derivative[:, 1] += rows[:, 2] / kappa
        fraction = expit(logits)
        fractions.append(fraction)
        derivatives.append(fraction[:, None] * (1 - fraction[:, None]) * derivative)
    probabilities = geometry["fixed"][None, :] + np.asarray(fractions).T @ geometry["differences"]
    gradient = np.einsum("fcp,fr->crp", np.asarray(derivatives), geometry["differences"])
    if np.min(probabilities) < -1e-12 or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-9, rtol=0):
        raise ArithmeticError("forward observation normalization failed")
    return np.maximum(probabilities, 0), gradient


def objective(parameters, counts, design, condition, rows, geometry):
    probabilities, gradient = forward(parameters, design, condition, rows, geometry)
    if np.any((counts > 0) & (probabilities <= 0)):
        return math.inf, np.zeros(3)
    safe = np.where(probabilities > 0, probabilities, 1.0)
    total = counts.sum()
    if total == 0:
        return math.inf, np.zeros(3)
    loss = -np.sum(counts * np.log(safe)) / total
    derivative = -np.einsum("cr,crp->p", counts / safe, gradient) / total
    return float(loss), derivative


def fit(counts, design, condition, rows, geometry):
    options = design["fit"]
    bounds = np.log(options["log_bounds"])
    candidates, attempts = [], []
    for start in options["starts"]:
        result = minimize(objective, np.log(start), args=(counts, design, condition, rows, geometry), jac=True,
                          method=options["method"], bounds=bounds,
                          options={key: options[key] for key in ("maxiter", "ftol", "gtol", "maxls")})
        result["method_used"] = "L-BFGS-B"
        attempts.append(result)
        interior = np.all(result.x > bounds[:, 0] + 1e-5) and np.all(result.x < bounds[:, 1] - 1e-5)
        if result.success and interior and np.isfinite(result.fun) and np.max(np.abs(result.jac)) < 1e-5:
            candidates.append(result)
    unresolved_loss = math.inf
    if "nonsmooth_fallback" in options:
        available = sorted((item for item in attempts if np.isfinite(item.fun)), key=lambda item: item.fun)
        for best in available:
            accepted_loss = min((item.fun for item in candidates), default=math.inf)
            if candidates and (not options.get("refine_lower_loss_attempts", False)
                               or best.fun >= accepted_loss - 1e-12):
                continue
            spec = options["nonsmooth_fallback"]["fallback"]
            result = minimize(lambda x: objective(x, counts, design, condition, rows, geometry)[0],
                              best.x, method=spec["method"], bounds=bounds,
                              options={key: spec[key] for key in ("xatol", "fatol", "maxiter", "maxfev")})
            interior = np.all(result.x > bounds[:, 0] + 1e-5) and np.all(result.x < bounds[:, 1] - 1e-5)
            if result.success and interior and np.isfinite(result.fun):
                changes = []
                for step in (1e-4, 1e-5):
                    for direction in itertools.product([-1, 0, 1], repeat=3):
                        if direction == (0, 0, 0):
                            continue
                        point = result.x + step * np.asarray(direction)
                        if np.any(point < bounds[:, 0]) or np.any(point > bounds[:, 1]):
                            continue
                        changes.append(objective(point, counts, design, condition, rows, geometry)[0] - result.fun)
                if min(changes, default=0) >= -1e-10:
                    result["method_used"] = spec["method"]
                    result["local_probe_min_loss_change"] = min(changes, default=0)
                    candidates.append(result)
                    continue
            unresolved_loss = min(unresolved_loss, best.fun)
            if not options.get("refine_lower_loss_attempts", False):
                break
    if not candidates:
        return {"success": False, "estimate": None, "loss": None}
    result = min(candidates, key=lambda candidate: candidate.fun)
    if options.get("refine_lower_loss_attempts", False) and unresolved_loss < result.fun - 1e-10:
        return {"success": False, "estimate": None, "loss": None,
                "failure": "unresolved_lower_loss_fit"}
    if options.get("require_full_rank_information", False):
        probability, derivative = forward(result.x, design, condition, rows, geometry)
        inverse = np.divide(1, probability, out=np.zeros_like(probability), where=probability > 0)
        information = np.einsum("crp,crq,cr,c->pq", derivative, derivative, inverse, counts.sum(axis=1))
        if np.linalg.matrix_rank(information) != 3:
            return {"success": False, "estimate": None, "loss": None,
                    "failure": "rank_deficient_fitted_timing_model"}
    return {"success": True, "estimate": np.exp(result.x).tolist(), "loss": float(result.fun),
            "solver": result["method_used"], "local_probe_min_loss_change": result.get("local_probe_min_loss_change")}


def plan(design, output):
    output.mkdir(exist_ok=False, parents=True)
    results, geometries = [], []
    for condition in design["conditions"]:
        rows = cells(design, condition)
        geometry = emission_geometry(design, condition)
        geometries.append(geometry)
        for truth_id, truth in enumerate(design["truth_grid"]):
            probabilities, gradient = forward(np.log(truth), design, condition, rows, geometry)
            inverse = np.divide(1, probabilities, out=np.zeros_like(probabilities), where=probabilities > 0)
            fisher = np.einsum("crp,crq,cr->pq", gradient, gradient, inverse)
            covariance = np.linalg.inv(fisher) if np.linalg.matrix_rank(fisher) == 3 else np.full((3, 3), math.inf)
            results.append({"condition": condition["name"], "truth_id": truth_id,
                            "unit_covariance": covariance.tolist(), "cells": len(rows)})
    allocations = []
    for count in design["planning"]["probes_per_cell_candidates"]:
        failed, largest = [], np.zeros(3)
        for row in results:
            se = np.sqrt(np.diag(row["unit_covariance"]) / count)
            largest = np.maximum(largest, se)
            if np.any(se > design["planning"]["maximum_log_parameter_se"]):
                failed.append([row["condition"], row["truth_id"]])
        allocations.append({"probes_per_cell": count, "max_log_parameter_se": largest.tolist(), "failed": failed})
    chosen = next((row["probes_per_cell"] for row in allocations if not row["failed"]), None)
    result = {"kind": "expected_fisher_allocation_not_recovery", "allocations": allocations,
              "selected_probes_per_cell": chosen, "conditions": results}
    (output / "planning.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "design.json").write_text(json.dumps(design, indent=2) + "\n")
    shutil.copyfile(__file__, output / "planner.py")
    shutil.copyfile(reference.__file__, output / "temporal_cognition_reference.py")
    print(json.dumps({"selected_probes_per_cell": chosen,
                      "allocation_failures": [len(row["failed"]) for row in allocations]}, indent=2))


def select(planning, output):
    design = json.loads((planning / "design.json").read_text())
    result = json.loads((planning / "planning.json").read_text())
    chosen = next((row["probes_per_cell"] for row in result["allocations"] if not row["failed"]), None)
    if chosen is None or chosen != result["selected_probes_per_cell"]:
        raise ValueError("no eligible prospectively selected allocation")
    design["schema"] = "temporal-dcc-private-recovery-design-v1"
    design["selected_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    design["probes_per_cell"] = chosen
    design["planning_result"] = {"path": str(planning / "planning.json"),
                                  "sha256": hashlib.sha256((planning / "planning.json").read_bytes()).hexdigest()}
    design["acceptance"]["quantile_method"] = "Linear interpolation of ordered errors; a positively weighted infinite endpoint yields infinity."
    design["allocation_cost"] = []
    for condition in design["conditions"]:
        rows = cells(design, condition)
        one_pass_sec = np.sum(replay_cost_seconds(design, rows))
        design["allocation_cost"].append({"condition": condition["name"], "cells": len(rows),
            "physical_probes": int(len(rows) * chosen), "probes_per_head": int(len(rows) * chosen // 2),
            "independent_full_prefix_replay_hours_lower_bound": float(one_pass_sec * chosen / 3600),
            "probe_windows_only_hours": float(len(rows) * chosen * 4 / 3600)})
    design["allocation_cost_status"] = "Numerical candidate only. Large replay duration is not accepted as an empirical collection plan; prefix reuse/checkpointing or a more efficient registered allocation needs its own feasibility evidence."
    with output.open("x") as stream:
        json.dump(design, stream, indent=2)
        stream.write("\n")
    print(json.dumps(design["allocation_cost"], indent=2))


_RUN = None


def initialize_worker(design, prepared, deadline):
    global _RUN
    _RUN = (design, prepared, deadline)


def run_point(task):
    condition_id, truth_id = task
    design, prepared, deadline = _RUN
    condition = design["conditions"][condition_id]
    rows, geometry = prepared[condition_id]
    truth = np.asarray(design["truth_grid"][truth_id])
    probabilities = forward(np.log(truth), design, condition, rows, geometry)[0]
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    started = time.process_time()
    records = []
    for replicate in range(design["replicates"]):
        if time.monotonic() >= deadline:
            break
        rng = np.random.default_rng(np.random.SeedSequence([design["seed"], condition_id, truth_id, replicate]))
        counts = rng.multinomial(design["probes_per_cell"], probabilities)
        fitted = fit(counts, design, condition, rows, geometry)
        errors = np.abs(np.log(np.asarray(fitted["estimate"]) / truth)).tolist() if fitted["success"] else None
        records.append({"condition": condition_id, "truth_id": truth_id, "replicate": replicate,
                        "truth": truth.tolist(), **fitted, "absolute_log_errors": errors})
    return {"condition": condition_id, "truth_id": truth_id, "records": records,
            "worker_cpu_sec": time.process_time() - started}


def summarize(records, design):
    seen = set()
    for record in records:
        key = (record["condition"], record["truth_id"], record["replicate"])
        if (key in seen or not 0 <= key[0] < len(design["conditions"])
                or not 0 <= key[1] < len(design["truth_grid"]) or not 0 <= key[2] < design["replicates"]
                or record["truth"] != design["truth_grid"][key[1]]):
            raise ValueError("duplicate, out-of-scope, or mismatched recovery record")
        seen.add(key)
        if record["success"]:
            estimate = np.asarray(record["estimate"])
            if estimate.shape != (3,) or not np.all(np.isfinite(estimate)) or np.any(estimate <= 0):
                raise ValueError("invalid successful parameter estimate")
            errors = np.abs(np.log(estimate / record["truth"]))
            if not np.allclose(errors, record["absolute_log_errors"], atol=1e-12, rtol=0):
                raise ValueError("stored recovery error disagrees with estimate")
    summaries = []
    for condition_id, condition in enumerate(design["conditions"]):
        for truth_id, truth in enumerate(design["truth_grid"]):
            found = [row for row in records if row["condition"] == condition_id and row["truth_id"] == truth_id]
            errors = np.asarray([row["absolute_log_errors"] if row["success"] else [math.inf] * 3 for row in found])
            metrics = []
            for parameter in range(3):
                ordered = sorted(errors[:, parameter]) if found else []
                quantiles = []
                for fraction in (0.5, 0.9):
                    if not ordered:
                        value = math.inf
                    else:
                        position = fraction * (len(ordered) - 1)
                        lo, hi = math.floor(position), math.ceil(position)
                        if position == lo:
                            value = ordered[lo]
                        elif math.isinf(ordered[hi]):
                            value = math.inf
                        else:
                            value = ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])
                    quantiles.append(value)
                metrics.append({"median": quantiles[0] if math.isfinite(quantiles[0]) else None,
                                "p90": quantiles[1] if math.isfinite(quantiles[1]) else None,
                                "passed": bool(quantiles[0] <= design["acceptance"]["median_abs_log_ratio_max"]
                                and quantiles[1] <= design["acceptance"]["p90_abs_log_ratio_max"])})
            complete = len(found) == design["replicates"]
            summaries.append({"condition": condition["name"], "truth_id": truth_id, "truth": truth,
                              "replicates": len(found), "failed_fits": sum(not row["success"] for row in found),
                              "parameters": metrics, "passed": complete and all(row["passed"] for row in metrics)})
    return summaries


def run(design_path, output, workers):
    design = json.loads(design_path.read_text())
    if "probes_per_cell" not in design or not 1 <= workers <= design["compute_budget"]["workers_max"]:
        raise ValueError("a selected allocation and registered worker count are required")
    output.mkdir(exist_ok=False, parents=True)
    shutil.copyfile(design_path, output / "design.json")
    shutil.copyfile(__file__, output / "runner.py")
    shutil.copyfile(reference.__file__, output / "temporal_cognition_reference.py")
    started, parent_cpu = time.monotonic(), time.process_time()
    budget = design["compute_budget"]
    deadline = started + min(budget["wall_sec"], 0.95 * budget["cpu_sec"] / workers)
    prepared = [(cells(design, condition), emission_geometry(design, condition)) for condition in design["conditions"]]
    records, cpu = [], 0.0
    tasks = list(itertools.product(range(len(design["conditions"])), range(len(design["truth_grid"]))))
    context = multiprocessing.get_context("spawn")
    with (output / "fits.jsonl").open("x") as stream:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context,
                initializer=initialize_worker, initargs=(design, prepared, deadline)) as executor:
            futures = [executor.submit(run_point, task) for task in tasks]
            for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                cpu += result["worker_cpu_sec"]
                records.extend(result["records"])
                for row in result["records"]:
                    stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(f"completed_points={number}/{len(tasks)} fits={len(records)} wall_sec={time.monotonic()-started:.1f}", flush=True)
    wall, cpu = time.monotonic() - started, cpu + time.process_time() - parent_cpu
    (output / "timing.json").write_text(json.dumps({"wall_sec": wall, "measured_cpu_sec": cpu,
        "cpu_excludes_worker_startup": True, "workers": workers}) + "\n")
    summaries = summarize(records, design)
    complete = len(records) == len(tasks) * design["replicates"]
    result = {"schema": "temporal-dcc-private-recovery-result-v1", "complete": complete,
              "passed": complete and all(row["passed"] for row in summaries) and wall <= budget["wall_sec"] and cpu <= budget["cpu_sec"],
              "numerical_passed": complete and all(row["passed"] for row in summaries),
              "wall_sec": wall, "measured_cpu_sec": cpu, "cpu_excludes_worker_startup": True,
              "workers": workers, "fits": len(records), "passed_points": sum(row["passed"] for row in summaries),
              "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
              "summaries": summaries, "artifacts": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
              for name in ["design.json", "runner.py", "temporal_cognition_reference.py", "fits.jsonl", "timing.json"]},
              "scope": "Conditional simulator only; no empirical allocation feasibility, waveform validation, human equivalence or production adoption."}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["complete", "passed", "fits", "passed_points", "wall_sec", "measured_cpu_sec"]}, indent=2))


def report(directory, ledger):
    result = json.loads((directory / "result.json").read_text())
    for name, digest in result["artifacts"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"artifact hash mismatch: {name}")
    design = json.loads((directory / "design.json").read_text())
    records = [json.loads(line) for line in (directory / "fits.jsonl").read_text().splitlines()]
    summaries = summarize(records, design)
    if summaries != result["summaries"] or len(records) != result["fits"]:
        raise ValueError("recomputed recovery summaries disagree")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, len(design["conditions"]), figsize=(12, 4.2), layout="constrained")
    for condition_id, (axis, condition) in enumerate(zip(axes, design["conditions"])):
        selected = [row for row in records if row["condition"] == condition_id]
        errors = np.asarray([np.log(np.asarray(row["estimate"]) / row["truth"]) for row in selected if row["success"]])
        failed = sum(not row["success"] for row in selected)
        if len(errors):
            scatter = axis.scatter(errors[:, 0], errors[:, 1], c=errors[:, 2], cmap="coolwarm", s=7, alpha=0.6,
                                   vmin=-math.log(2), vmax=math.log(2))
            figure.colorbar(scatter, ax=axis, label="log(cap estimate / truth)", shrink=0.8)
        for bound in [-math.log(2), math.log(2)]:
            axis.axvline(bound, color="gray", linestyle="--", linewidth=0.7)
            axis.axhline(bound, color="gray", linestyle="--", linewidth=0.7)
        axis.set(xlabel="log(tau estimate / truth)", ylabel="log(kappa estimate / truth)",
                 title=f"{condition['name']}\n{len(selected)} fits; {failed} failed")
    version = design["schema"].rsplit("-", 1)[-1]
    figure.suptitle(f"Private physical-record recovery {version}: {result['passed_points']}/108 condition points passed\n"
                   "Conditional simulator; empirical allocation not adopted")
    for suffix in ("png", "pdf"):
        figure.savefig(directory / f"joint-errors.{suffix}", dpi=160)
    plt.close(figure)
    if ledger.exists():
        record = json.loads(ledger.read_text())
    else:
        record = {"schema": "temporal-dcc-parameter-recovery-v1", "collection_gate_passed": False}
    previous = record.get("private_trace", {})
    numeric = result.get("numerical_passed", result["passed"])
    previous.update({"status": "conditional_numerical_recovery_passed; empirical_allocation_not_adopted" if numeric else "conditional_recovery_failed",
                     "reference": "private-trace-reference.json", "human_collection_gate_passed": False,
                     "current_run": str(directory), "result_sha256": hashlib.sha256((directory / "result.json").read_bytes()).hexdigest(),
                     "passed_points": result["passed_points"], "total_points": len(summaries), "fits": len(records),
                     "wall_sec": result["wall_sec"], "measured_cpu_sec": result["measured_cpu_sec"],
                     "allocation_cost": design["allocation_cost"],
                     "remaining": ["Reduce/validate empirical fixed-policy allocation cost without deleting the truth envelope.",
                                   "Validate actual waveform-derived reference inputs, body/opportunity feasibility and acquisition-aligned timing.",
                                   "Complete all other M0 registrations and numerical preflight; no collection or production promotion follows this result."]})
    previous.setdefault("runs", []).append({"directory": str(directory), "passed": result["passed"],
        "result_sha256": previous["result_sha256"]})
    record["private_trace"] = previous
    ledger.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"verified_fits": len(records), "verified_summaries": len(summaries), "passed": result["passed"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register")
    register.add_argument("--output", type=Path, required=True)
    planning = sub.add_parser("plan")
    planning.add_argument("--design", type=Path, required=True)
    planning.add_argument("--output", type=Path, required=True)
    selection = sub.add_parser("select")
    selection.add_argument("--planning", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)
    running = sub.add_parser("run")
    running.add_argument("--design", type=Path, required=True)
    running.add_argument("--output", type=Path, required=True)
    running.add_argument("--workers", type=int, default=8)
    reporting = sub.add_parser("report")
    reporting.add_argument("--run", type=Path, required=True)
    reporting.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "register":
        with args.output.open("x") as stream:
            json.dump(registration(), stream, indent=2)
            stream.write("\n")
    elif args.command == "plan":
        plan(json.loads(args.design.read_text()), args.output)
    elif args.command == "select":
        select(args.planning, args.output)
    elif args.command == "run":
        run(args.design, args.output, args.workers)
    else:
        report(args.run, args.ledger)


if __name__ == "__main__":
    main()
