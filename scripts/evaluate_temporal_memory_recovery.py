#!/usr/bin/env python3
"""Register and test an episode-recognition allocation before human collection.

This simulator conditions on known acoustic scores and observed episode support.
It is not an acoustic extractor, a human experiment, or the private trace gate.
"""

import argparse
import csv
import datetime
import hashlib
import itertools
import json
import math
import multiprocessing as mp
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ["tau_sec", "kappa", "strength_max", "bias"]
_CPU_PROGRESS = None


def registration():
    return {
        "schema": "temporal-dcc-episode-recovery-design-v1",
        "version": "episode-allocation-1",
        "registered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "candidate_allocation_for_simulation; no_human_recruitment_authorized",
        "specification_sha256": hashlib.sha256(
            (ROOT / "docs/design-notes/dcc-neurocognitive-hierarchy.md").read_bytes()).hexdigest(),
        "parameters": PARAMETERS,
        "truth_grid": [[2, 20, 200, 1200], [1, 4, 16], [1.5, 3, 6], [-2, 0, 2]],
        "datasets_per_point_condition": 100,
        "seed": 2026091011,
        "seed_derivation": "numpy SeedSequence([seed, condition_index, grid_index, replicate])",
        "factorial": {
            "delay_to_target_start_sec": [4, 16, 64, 256, 1024, 2048],
            "interference_units": [0, 1, 4, 16],
            "prior_occurrences": [1, 2, 4, 8, 16],
            "support_per_occurrence": [0.5, 1.0],
            "cue_kind": ["exact", "transformed", "novel_foil"],
        },
        "allocation": {
            "cells": 720, "ratings_per_cell": 4, "assigned_targets": 2880,
            "candidate_participants": 960, "targets_per_participant": 3,
            "source_families": 720,
            "assignment_seed": 2026091012,
            "rule": "One cell per source family. Independently permute the 960 ratings in each delay block [4,16], [64,256], [1024,2048] onto participants 0..959; each participant receives one per block, hence three different families. Randomize block order per participant.",
            "participant_effect": "zero in this first conditional simulator; not an empirical human assumption",
        },
        "query": {
            "duration_sec": 8, "minimum_prefix_sec": 16,
            "elapsed_at_scoring": "delay_to_target_start_sec + 8",
            "censoring": "prefix ends at the end of the target window; no subsequent audio; only the observed query score is supplied",
            "bank_copy": "only episodes committed before target start; no query reinforcement, new episode or competing-content update",
            "occurrence_duration_sec_for_delivery_estimate": 2,
            "prefix_duration_estimate": "max(16, 2*prior_occurrences + delay_to_target_start_sec + 8)",
            "delivery_status": "No waveform has yet established this candidate allocation. Interference values are conditional observed weighted-span totals, not authored labels or proof of stimulus feasibility.",
        },
        "conditions": [
            {"id": "clear_complete", "cue_scores": [4, 2, -4],
             "missing_base": 0, "missing_ambiguity_gain": 0, "distractors": []},
            {"id": "clear_mcar10", "cue_scores": [4, 2, -4],
             "missing_base": 0.1, "missing_ambiguity_gain": 0, "distractors": []},
            {"id": "weak_competition_ambiguous_missing", "cue_scores": [2, 0, -6],
             "missing_base": 0.1, "missing_ambiguity_gain": 0.2,
             "distractors": [
                 {"extra_age_sec": 10, "extra_interference": 1, "score": -6},
                 {"extra_age_sec": 40, "extra_interference": 2, "score": -7},
                 {"extra_age_sec": 160, "extra_interference": 4, "score": -8}]},
        ],
        "forward": {
            "primary_strength": "min(strength_max, prior_occurrences*support_per_occurrence)",
            "primary_log_availability": "log(strength) - (delay+8)/tau_sec - interference/kappa",
            "distractor_strength": 1,
            "distractor_elapsed": "delay+8+2*(prior_occurrences+1)+extra_age_sec",
            "distractor_interference": "interference+extra_interference",
            "epsilon_avail": 1e-300,
            "recognition": "sigmoid(logsumexp(logaddexp(log_availability_i, log(epsilon_avail))+score_i)-bias)",
            "missing_probability": "missing_base + missing_ambiguity_gain * 4*p_yes*(1-p_yes), applied independently before sampling answers",
            "sampling": "n_usable~Binomial(4,1-missing_probability); yes~Binomial(n_usable,p_yes). Missing contributes no binary target; retain all assigned counts.",
            "acquisition_gaps": "none; this does not qualify the bounded-unknown-interference branch",
            "nuisance_origin": "fixed simulation-only engineering scores; repeat after separately fitted acoustic/correspondence scales",
        },
        "fit": {
            "solver": "L-BFGS-B; analytic gradient of summed binary log loss",
            "coordinates": ["log(tau_sec)", "log(kappa)", "log(strength_max)", "bias"],
            "physical_bounds": [[0.05, 100000], [0.05, 10000], [1, 32], [-12, 12]],
            "starts": [[20, 4, 3, 0], [2, 1, 1.25, -2], [1200, 16, 8, 2]],
            "selection": "minimum penalized loss among converged starts; retain start failures",
            "ridge_lambda": 1e-6,
            "ridge": "0.5*lambda*sum(transformed_parameter**2); no true-parameter centering",
            "options": {"maxiter": 400, "ftol": 1e-12, "gtol": 1e-6, "maxls": 40},
            "saturation_kink": "use unsaturated derivative exactly at equal accumulated support and cap",
            "information": "unpenalized expected Fisher J.T @ diag(n*p*(1-p)) @ J",
            "failure": "no converged start, nonfinite result, boundary distance <=1e-7 in fitted coordinates, smallest Fisher eigenvalue <=1e-9, or Fisher condition number >1e12",
        },
        "acceptance": {
            "positive_absolute_log_ratio": {"median_max": math.log(1.25), "p90_max": math.log(2)},
            "bias_absolute_error": {"median_max": 0.25, "p90_max": 0.75},
            "quantiles": "empirical inverse CDF: sorted_error[ceil(q*n)-1]",
            "failed_fit_error": "positive infinity for all four parameters; never exclude a dataset",
            "unit": "every parameter in every truth-grid point and condition; no pooling",
            "on_failure": "retain all failures; revise allocation/task/model in a new registration; never relax tolerance or delete a grid point",
        },
        "compute_envelope": {
            "datasets": 32400, "optimizer_starts_max": 97200,
            "workers_max": 8, "blas_threads_per_worker": 1,
            "wall_hours_max": 2, "cpu_core_hours_max": 8,
            "exhaustion": "stop and preserve incomplete result; no recovery pass",
            "separate_from_stage3_budget": True,
        },
        "limitations": [
            "This is a conditional episode-law gate, not the complete O11 gate.",
            "No participant, passage random effect or unheard acquisition-gap distribution is inferred.",
            "No human recruitment, expense, consent, waveform feasibility or cognitive validity is established.",
            "O12 event-head and O13 private-trace recovery remain separate.",
        ],
    }


def short_query_registration():
    design = registration()
    design.update(version="episode-allocation-2-planning", seed=2026091013,
                  status="registered_design_search; auxiliary_query_protocol_not_adopted")
    design["supersedes"] = {"version": "episode-allocation-1",
                            "reason": "Eight-second endpoint scoring gave insufficient short-retention information in all 81 tau=2 stress cells."}
    design["query"].update(
        auxiliary_duration_sec=[1, 2, 4],
        elapsed_at_scoring="delay_to_target_start_sec + per_trial_query_duration_sec",
        prefix_duration_estimate="max(16, 2*prior_occurrences + delay_to_target_start_sec + query_duration_sec)",
        protocol_revision="Retain every original eight-second cell; add separately identified 1/2/4-second query trials. The same supported ordered-return criterion and copy/no-update forward law apply. This is a proposed assay extension, not an adopted human protocol or a runtime change.",
        scope_test="Before human use, separately fit/validate correspondence support for each query duration and test whether one retention law transfers to held-out eight-second queries. A duration-dependent discrepancy requires a model/task revision, not relabeling short-query fits as eight-second validity.",
    )
    rows = []
    for delay in [1.25, 2.25, 4, 6, 8, 16, 64, 256, 1024, 2048]:
        levels = [0, 1] if delay == 1.25 else [0, 1, 4] if delay == 2.25 else [0, 1, 4, 16]
        durations = [1, 2, 4, 8] if delay <= 8 else [8]
        for interference, count, support, cue, duration in itertools.product(
                levels, [1, 2, 4, 8, 16], [0.5, 1.0], ["exact", "transformed", "novel_foil"], durations):
            rows.append([delay, interference, count, support, cue, duration])
    design["trial_cell_columns"] = ["delay_to_target_start_sec", "interference_units", "prior_occurrences",
                                    "support_per_occurrence", "cue_kind", "query_duration_sec"]
    design["trial_cells"] = rows
    design["factorial"].update(delay_to_target_start_sec=[1.25, 2.25, 4, 6, 8, 16, 64, 256, 1024, 2048],
                               query_duration_sec=[1, 2, 4, 8])
    design["physical_support_constraints"] = {
        "commit_lag_sec": 0.5, "minimum_delay_sec": 1.25,
        "interference_limit": "At 1.25 seconds allow at most one observed weighted span; at 2.25 at most four; from four seconds at most sixteen. Each contributing span must end and commit strictly before target start.",
        "factorial_support": "Time/content/exposure cross fully over the common 0/1-content subset; the complete 0/1/4/16-content factorial is retained at delays >=4 seconds. Shorter-delay exclusions are registered physical support constraints, not deleted recovery stress points.",
        "verification": "These caps and two-second occurrence durations are construction bounds; rendered/acoustically measured support and short-query heard correspondence remain unverified.",
    }
    design["allocation"].update(cells=len(rows), ratings_per_cell=None, assigned_targets=None,
                                candidate_participants=None, targets_per_participant=11,
                                source_families=len(rows), assignment_seed=2026091014,
                                rule="Preserve the original three delay-block assignments among eight-second cells. Round-robin the randomized auxiliary families across participants, assigning every family's ratings to different participants. Randomize all eleven trial positions per participant.")
    design["allocation_search"] = {
        "candidates_ratings_per_cell": [4, 8, 16, 32],
        "method": "At every registered truth/condition, invert the unpenalized expected Fisher matrix under that condition's usable-response mechanism. Choose the smallest uniform rating count whose four coordinate standard errors satisfy all planning limits.",
        "coordinate_standard_error_max": [0.25, 0.25, 0.25, 0.3],
        "nonpositive_information": "candidate fails; no pseudoinverse or regularization rescue",
        "selection_uses_responses": False,
        "status": "not_run",
        "limits_role": "Planning margins only. Actual 100-dataset recovery thresholds remain unchanged. No new allocation is tried after seeing its Monte Carlo outcomes without registering a new version.",
        "no_candidate": "register allocation infeasibility and revise task/design; do not generate human or Monte Carlo responses",
        "cpu_core_hours_max": 0.1,
    }
    design["forward"]["primary_log_availability"] = "log(strength) - (delay+query_duration_sec)/tau_sec - interference/kappa"
    design["forward"]["distractor_elapsed"] = "delay+query_duration_sec+2*(prior_occurrences+1)+extra_age_sec"
    design["forward"]["sampling"] = "n_usable~Binomial(selected_ratings_per_cell,1-missing_probability); yes~Binomial(n_usable,p_yes). Missing remains in all assigned denominators."
    return design


def plan_allocation(design_path, output):
    raw = design_path.read_bytes()
    design = json.loads(raw)
    if design["allocation_search"]["status"] != "not_run":
        raise ValueError("a registered planning design is required")
    output.mkdir(parents=True, exist_ok=False)
    (output / "planning-design.json").write_bytes(raw)
    (output / "planner.py").write_bytes(Path(__file__).read_bytes())
    started = time.process_time()
    information = []
    for condition in design["conditions"]:
        inputs = prepared_inputs(design, condition)
        for index, truth in enumerate(itertools.product(*design["truth_grid"])):
            theta = np.asarray([*np.log(truth[:3]), truth[3]])
            eta, jac = forward(theta, inputs, design["forward"]["epsilon_avail"])
            p = expit(eta)
            observed = 1 - condition["missing_base"] - condition["missing_ambiguity_gain"] * 4 * p * (1 - p)
            matrix = jac.T @ ((observed * p * (1 - p))[:, None] * jac)
            eigen = np.linalg.eigvalsh(matrix)
            if eigen[0] <= 0:
                variances = np.full(4, np.inf)
            else:
                variances = np.diag(np.linalg.inv(matrix))
            information.append({"condition": condition["id"], "grid_index": index, "truth": list(truth),
                                "variance_one_rating": [float(x) if math.isfinite(x) else "infinite" for x in variances]})
    candidates = []
    limits = np.asarray(design["allocation_search"]["coordinate_standard_error_max"])
    selected = None
    for n in design["allocation_search"]["candidates_ratings_per_cell"]:
        failures = []
        maxima = np.zeros(4)
        for point in information:
            variance = np.asarray([np.inf if x == "infinite" else x for x in point["variance_one_rating"]])
            standard_error = np.sqrt(variance / n)
            maxima = np.maximum(maxima, standard_error)
            if np.any(standard_error > limits):
                failures.append({"condition": point["condition"], "grid_index": point["grid_index"]})
        candidates.append({"ratings_per_cell": n, "assigned_targets": len(cells(design)) * n,
                           "candidate_participants": 240 * n, "failed_points": failures,
                           "maximum_standard_errors": [float(x) if math.isfinite(x) else "infinite" for x in maxima]})
        if selected is None and not failures:
            selected = n
    result = {"schema": "temporal-dcc-episode-allocation-search-v1", "design_sha256": hashlib.sha256(raw).hexdigest(),
              "planner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cpu_sec": time.process_time() - started, "information": information, "candidates": candidates,
              "selected_ratings_per_cell": selected, "status": "selected_for_recovery" if selected else "no_feasible_candidate",
              "human_collection_authorized": False}
    (output / "allocation-search.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if selected:
        design.update(version="episode-allocation-2", status="frozen_before_recovery; proposed_assay_extension_not_adopted",
                      registered_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        design["allocation_search"].update(status="selected_before_recovery",
                                          planning_design_sha256=hashlib.sha256(raw).hexdigest(),
                                          search_result_sha256=hashlib.sha256((output / "allocation-search.json").read_bytes()).hexdigest())
        design["allocation"].update(ratings_per_cell=selected, assigned_targets=len(cells(design)) * selected,
                                    candidate_participants=240 * selected)
        assignments = assignment(design)
        durations = np.zeros(design["allocation"]["candidate_participants"])
        for item in assignments:
            durations[item["synthetic_participant"]] += item["prefix_sec_estimate"]
        design["allocation"]["audio_burden_sec_estimate"] = {
            "mean": float(durations.mean()), "min": float(durations.min()), "max": float(durations.max()),
            "excludes": "instructions, responses, breaks and source-specific duration changes"}
        (output / "selected-design.json").write_text(json.dumps(design, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "selected_ratings_per_cell": selected,
                      "candidates": [{k: v for k, v in c.items() if k != "failed_points"}
                                     | {"failed_points": len(c["failed_points"])} for c in candidates]}))


def cells(design):
    if "trial_cells" in design:
        return np.asarray(design["trial_cells"], dtype=object)
    names = ["delay_to_target_start_sec", "interference_units", "prior_occurrences",
             "support_per_occurrence", "cue_kind"]
    return np.asarray(list(itertools.product(*(design["factorial"][n] for n in names))), dtype=object)


def assignment(design):
    rows = cells(design)
    allocation = design["allocation"]
    rng = np.random.default_rng(allocation["assignment_seed"])
    participants = [[] for _ in range(allocation["candidate_participants"])]
    query_durations = rows[:, 5].astype(float) if rows.shape[1] == 6 else np.full(len(rows), design["query"]["duration_sec"])
    assigned = set()
    for block in ([4, 16], [64, 256], [1024, 2048]):
        repeated = [i for i, row in enumerate(rows) if row[0] in block and query_durations[i] == 8
                    for _ in range(allocation["ratings_per_cell"])]
        if len(repeated) != len(participants):
            raise ValueError("delay block does not balance the registered participants")
        for participant, cell in zip(participants, rng.permutation(repeated)):
            participant.append(int(cell))
        assigned.update(repeated)
    extra = rng.permutation([i for i in range(len(rows)) if i not in assigned])
    order = rng.permutation(len(participants)) if len(extra) else []
    counter = 0
    for index in extra:
        for _ in range(allocation["ratings_per_cell"]):
            participants[order[counter % len(participants)]].append(int(index))
            counter += 1
    assignments = []
    for person, indices in enumerate(participants):
        for order, index in enumerate(rng.permutation(indices)):
            row = rows[index]
            assignments.append({"synthetic_participant": person, "order": order,
                                "source_family": int(index), "cell": int(index),
                                "query_duration_sec": float(query_durations[index]),
                                "prefix_sec_estimate": max(16, 2 * int(row[2]) + float(row[0]) + query_durations[index])})
    return assignments


def prepared_inputs(design, condition):
    rows = cells(design)
    numeric = rows[:, :4].astype(float)
    cue_index = np.asarray([design["factorial"]["cue_kind"].index(x) for x in rows[:, 4]])
    delay, interference, count, support = numeric.T
    durations = rows[:, 5].astype(float) if rows.shape[1] == 6 else design["query"]["duration_sec"]
    ages = [delay + durations]
    interference_columns = [interference]
    scores = [np.asarray(condition["cue_scores"])[cue_index]]
    for distractor in condition["distractors"]:
        ages.append(ages[0] + 2 * (count + 1) + distractor["extra_age_sec"])
        interference_columns.append(interference + distractor["extra_interference"])
        scores.append(np.full(len(rows), distractor["score"]))
    return (count * support, np.asarray(ages).T,
            np.asarray(interference_columns).T, np.asarray(scores).T)


def forward(theta, inputs, epsilon=1e-300):
    total_support, ages, interference, scores = inputs
    tau, kappa, cap = np.exp(theta[:3])
    strength = np.ones_like(ages)
    strength[:, 0] = np.minimum(cap, total_support)
    log_avail = np.log(strength) - ages / tau - interference / kappa
    protected = np.logaddexp(log_avail, math.log(epsilon))
    weighted = protected + scores
    total = logsumexp(weighted, axis=1)
    weights = np.exp(weighted - total[:, None]) * np.exp(log_avail - protected)
    jac = np.empty((len(ages), 4))
    jac[:, 0] = np.sum(weights * ages / tau, axis=1)
    jac[:, 1] = np.sum(weights * interference / kappa, axis=1)
    jac[:, 2] = weights[:, 0] * (total_support > cap)
    jac[:, 3] = -1
    return total - theta[3], jac


def objective(theta, inputs, usable, yes, ridge, epsilon):
    logit, jac = forward(theta, inputs, epsilon)
    loss = np.sum(yes * np.logaddexp(0, -logit) + (usable - yes) * np.logaddexp(0, logit))
    gradient = jac.T @ (usable * expit(logit) - yes)
    return loss + 0.5 * ridge * (theta @ theta), gradient + ridge * theta


def fit_counts(design, inputs, usable, yes):
    fit = design["fit"]
    bounds = [(math.log(lo), math.log(hi)) if i < 3 else (lo, hi)
              for i, (lo, hi) in enumerate(fit["physical_bounds"])]
    fits = []
    for initial in fit["starts"]:
        theta = np.asarray([*np.log(initial[:3]), initial[3]])
        fits.append(minimize(objective, theta,
                             args=(inputs, usable, yes, fit["ridge_lambda"],
                                   design["forward"]["epsilon_avail"]),
                             method="L-BFGS-B", jac=True, bounds=bounds, options=fit["options"]))
    converged = [x for x in fits if x.success and np.isfinite(x.fun) and np.all(np.isfinite(x.x))]
    record = {"start_status": [bool(x.success) for x in fits],
              "start_messages": [str(x.message) for x in fits], "status": "failed", "estimate": None}
    if not converged:
        record["reason"] = "no_converged_start"
        return record
    best = min(converged, key=lambda x: x.fun)
    theta = best.x
    logit, jac = forward(theta, inputs, design["forward"]["epsilon_avail"])
    p = expit(logit)
    information = jac.T @ ((usable * p * (1 - p))[:, None] * jac)
    eigen = np.linalg.eigvalsh(information)
    boundary = any(min(value - lo, hi - value) <= 1e-7 for value, (lo, hi) in zip(theta, bounds))
    record.update(estimate=[*np.exp(theta[:3]).tolist(), float(theta[3])],
                  objective=float(best.fun), fisher_eigenvalues=eigen.tolist())
    if boundary:
        record["reason"] = "at_parameter_bound"
    elif eigen[0] <= 1e-9 or eigen[-1] / eigen[0] > 1e12:
        record["reason"] = "insufficient_unpenalized_information"
    else:
        record["status"] = "fitted"
    return record


def simulate_cell(task):
    cell_index, design, condition_index, grid_index, truth = task
    condition = design["conditions"][condition_index]
    inputs = prepared_inputs(design, condition)
    theta = np.asarray([*np.log(truth[:3]), truth[3]])
    logit, _ = forward(theta, inputs, design["forward"]["epsilon_avail"])
    probabilities = expit(logit)
    missing = condition["missing_base"] + condition["missing_ambiguity_gain"] * 4 * probabilities * (1 - probabilities)
    records = []
    cpu_start = time.process_time()
    for replicate in range(design["datasets_per_point_condition"]):
        rng = np.random.default_rng(np.random.SeedSequence(
            [design["seed"], condition_index, grid_index, replicate]))
        usable = rng.binomial(design["allocation"]["ratings_per_cell"], 1 - missing)
        yes = rng.binomial(usable, probabilities)
        record = fit_counts(design, inputs, usable, yes)
        errors = ["infinite"] * 4
        if record["status"] == "fitted":
            errors = [*np.abs(np.log(np.asarray(record["estimate"][:3]) / truth[:3])).tolist(),
                      abs(record["estimate"][3] - truth[3])]
        record.update(condition=condition["id"], grid_index=grid_index, replicate=replicate,
                      truth=list(truth), absolute_errors=errors,
                      assigned=int(design["allocation"]["assigned_targets"]),
                      usable=int(usable.sum()), yes=int(yes.sum()))
        records.append(record)
        if _CPU_PROGRESS is not None:
            _CPU_PROGRESS[cell_index] = time.process_time() - cpu_start
    return records, time.process_time() - cpu_start


def init_worker(progress):
    global _CPU_PROGRESS
    _CPU_PROGRESS = progress


def recovery_summary(records, design):
    errors = np.asarray([[math.inf if x == "infinite" else x for x in r["absolute_errors"]]
                         for r in records])
    by_parameter = {}
    for i, name in enumerate(PARAMETERS):
        ordered = np.sort(errors[:, i])
        limits = design["acceptance"]["bias_absolute_error" if i == 3 else "positive_absolute_log_ratio"]
        median = float(ordered[math.ceil(0.5 * len(ordered)) - 1])
        p90 = float(ordered[math.ceil(0.9 * len(ordered)) - 1])
        by_parameter[name] = {"median": median if math.isfinite(median) else "infinite",
                              "p90": p90 if math.isfinite(p90) else "infinite",
                              "pass": median <= limits["median_max"] and p90 <= limits["p90_max"]}
    return {"condition": records[0]["condition"], "grid_index": records[0]["grid_index"],
            "truth": records[0]["truth"], "datasets": len(records),
            "failed_fits": sum(r["status"] != "fitted" for r in records),
            "parameters": by_parameter, "pass": all(x["pass"] for x in by_parameter.values())}


def run(design_path, output, workers):
    raw = design_path.read_bytes()
    design = json.loads(raw)
    if not 1 <= workers <= design["compute_envelope"]["workers_max"]:
        raise ValueError("worker count outside registered envelope")
    output.mkdir(parents=True, exist_ok=False)
    (output / "design.json").write_bytes(raw)
    (output / "runner.py").write_bytes(Path(__file__).read_bytes())
    grid = list(itertools.product(*design["truth_grid"]))
    tasks = [(design, c, i, truth) for c in range(len(design["conditions"]))
             for i, truth in enumerate(grid)]
    tasks = [(index, *task) for index, task in enumerate(tasks)]
    assignments = assignment(design)
    with (output / "assignment.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(assignments[0]))
        writer.writeheader()
        writer.writerows(assignments)
    manifest = {"schema": "temporal-dcc-episode-recovery-result-v1", "status": "running",
                "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "design_sha256": hashlib.sha256(raw).hexdigest(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
                "workers": workers, "cells": [], "cpu_sec": 0, "wall_sec": 0,
                "human_collection_authorized": False}
    result_path = output / "result.json"
    result_path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    parent_cpu_start = time.process_time()
    context = mp.get_context("spawn")
    progress = context.Array("d", len(tasks), lock=False)
    exhausted = False
    try:
        pool = context.Pool(workers, initializer=init_worker, initargs=(progress,))
    except OSError as error:
        manifest.update(status="execution_failed", execution_error=str(error))
        result_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        raise
    with (output / "fits.jsonl").open("x") as stream, pool:
        iterator = pool.imap_unordered(simulate_cell, tasks)
        while len(manifest["cells"]) < len(tasks):
            wall_sec = time.monotonic() - started
            cpu_sec = sum(progress) + time.process_time() - parent_cpu_start
            if (wall_sec >= design["compute_envelope"]["wall_hours_max"] * 3600
                    or cpu_sec >= design["compute_envelope"]["cpu_core_hours_max"] * 3600):
                exhausted = True
                pool.terminate()
                break
            try:
                records, _ = iterator.next(timeout=1)
            except mp.TimeoutError:
                continue
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            manifest["cells"].append(recovery_summary(records, design))
            manifest["cpu_sec"] = sum(progress) + time.process_time() - parent_cpu_start
            manifest["wall_sec"] = time.monotonic() - started
            result_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"completed_cells": len(manifest["cells"]), "total_cells": len(tasks),
                              "passed_cells": sum(x["pass"] for x in manifest["cells"]),
                              "cpu_sec": round(manifest["cpu_sec"], 2),
                              "wall_sec": round(manifest["wall_sec"], 2)}), flush=True)
    manifest["cells"].sort(key=lambda x: (x["condition"], x["grid_index"]))
    manifest["wall_sec"] = time.monotonic() - started
    manifest["cpu_sec"] = sum(progress) + time.process_time() - parent_cpu_start
    manifest["status"] = ("incomplete_budget_exhausted" if exhausted else
                          "conditional_simulator_pass" if all(x["pass"] for x in manifest["cells"])
                          else "recovery_failed")
    manifest["fits_sha256"] = hashlib.sha256((output / "fits.jsonl").read_bytes()).hexdigest()
    result_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def report(output, ledger_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = json.loads((output / "result.json").read_text())
    design_raw = (output / "design.json").read_bytes()
    design = json.loads(design_raw)
    if result["status"] not in {"conditional_simulator_pass", "recovery_failed"}:
        raise ValueError("only a complete simulation can receive a recovery report")
    for path, expected in [(output / "design.json", result["design_sha256"]),
                           (output / "runner.py", result["source_sha256"]),
                           (output / "fits.jsonl", result["fits_sha256"])]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"recorded content changed: {path}")
    records = [json.loads(line) for line in (output / "fits.jsonl").read_text().splitlines()]
    grid = list(itertools.product(*design["truth_grid"]))
    expected_keys = {(c["id"], i, j) for c in design["conditions"] for i in range(len(grid))
                     for j in range(design["datasets_per_point_condition"])}
    keys = [(r["condition"], r["grid_index"], r["replicate"]) for r in records]
    if len(keys) != len(expected_keys) or set(keys) != expected_keys:
        raise ValueError("missing, duplicate or extra simulation rows")
    regenerated = []
    for condition in design["conditions"]:
        for index, truth in enumerate(grid):
            group = [r for r in records if r["condition"] == condition["id"] and r["grid_index"] == index]
            if any(r["truth"] != list(truth) for r in group):
                raise ValueError("truth does not match registration")
            regenerated.append(recovery_summary(group, design))
    regenerated.sort(key=lambda x: (x["condition"], x["grid_index"]))
    if regenerated != result["cells"]:
        raise ValueError("stored summary does not match every retained fit")

    fig, axes = plt.subplots(1, len(design["conditions"]), figsize=(14, 4.5), layout="constrained")
    largest = max(r["absolute_errors"][2] for r in records if r["status"] == "fitted")
    condition_summaries = []
    for ax, condition in zip(np.atleast_1d(axes), design["conditions"]):
        group = [r for r in records if r["condition"] == condition["id"]]
        fitted = [r for r in group if r["status"] == "fitted"]
        signed = np.asarray([np.log(np.asarray(r["estimate"][:3]) / r["truth"][:3]) for r in fitted])
        points = ax.scatter(signed[:, 0], signed[:, 1], c=np.abs(signed[:, 2]),
                            s=4, alpha=0.3, cmap="viridis", vmin=0, vmax=largest, rasterized=True)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("log(estimated tau / true tau)")
        ax.set_ylabel("log(estimated kappa / true kappa)")
        label = condition["id"].replace("_", " ")
        ax.set_title(label.replace(" ambiguous ", "\nambiguous "), fontsize=10)
        failed = len(group) - len(fitted)
        ax.text(0.02, 0.98, f"{len(fitted)} finite fits\n{failed} failed fits: infinite errors",
                va="top", transform=ax.transAxes, fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        cells_for_condition = [c for c in regenerated if c["condition"] == condition["id"]]
        condition_summaries.append({"id": condition["id"], "datasets": len(group), "failed_fits": failed,
                                    "passed_grid_points": sum(c["pass"] for c in cells_for_condition),
                                    "total_grid_points": len(cells_for_condition)})
    fig.colorbar(points, ax=axes, label="absolute log strength-cap error", shrink=0.8)
    fig.suptitle(f"{design['version']}: joint parameter errors\nFailed fits remain in acceptance statistics; no finite plotting location", fontsize=12)
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"joint-errors.{suffix}", dpi=180)
    plt.close(fig)

    diagnostics = []
    truth = [2, 4, 3, 0]
    alternative = [3, 4, 3, 2]
    for condition in design["conditions"]:
        inputs = prepared_inputs(design, condition)
        eta, _ = forward(np.asarray([*np.log(truth[:3]), truth[3]]), inputs)
        alternative_eta, _ = forward(np.asarray([*np.log(alternative[:3]), alternative[3]]), inputs)
        probability = expit(eta)
        usable = design["allocation"]["ratings_per_cell"] * (
            1 - condition["missing_base"] - condition["missing_ambiguity_gain"] * 4 * probability * (1 - probability))
        kl = usable * (probability * (np.logaddexp(0, -alternative_eta) - np.logaddexp(0, -eta))
                       + (1 - probability) * (np.logaddexp(0, alternative_eta) - np.logaddexp(0, eta)))
        diagnostics.append({"condition": condition["id"], "truth": truth, "alternative": alternative,
                            "expected_yes": float(np.sum(usable * probability)),
                            "expected_conditional_response_log_likelihood_ratio_nats": float(kl.sum()),
                            "role": "post-run explanatory diagnostic, not a new acceptance threshold; conditions on usable responses"})

    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "schema": "temporal-dcc-parameter-recovery-ledger-v1", "collection_gate_passed": False,
        "episode": {"runs": []}, "private_trace": {"status": "not_run"},
        "event_heads": {"status": "not_run"}}
    run_record = {"design_version": design["version"], "design_sha256": result["design_sha256"],
                  "run_directory": str(output.resolve().relative_to(ROOT)), "status": result["status"],
                  "runner_sha256": result["source_sha256"],
                  "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "report_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                      "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
                  "total_datasets": len(records), "total_grid_condition_points": len(regenerated),
                  "passed_points": sum(c["pass"] for c in regenerated),
                  "wall_sec": result["wall_sec"], "cpu_sec": result["cpu_sec"],
                  "cpu_measurement_scope": "worker simulation loops plus parent; process startup is not included",
                  "conditions": condition_summaries, "post_run_diagnostics": diagnostics,
                  "artifacts": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                for name in ["result.json", "fits.jsonl", "assignment.csv", "joint-errors.png", "joint-errors.pdf"]},
                  "required_next": (["revise candidate allocation or task/model in a new registered version without dropping failed stress points"]
                                    if result["status"] == "recovery_failed" else
                                    ["explicitly reconcile the auxiliary-query protocol with the specification; verify its burden and waveform support"])
                                   + [
                                    "establish waveform feasibility and all human collection prerequisites",
                                    "repeat with actual fitted acoustic/match scales", "complete independent O12/O13 gates"]}
    ledger["episode"]["runs"] = [r for r in ledger["episode"]["runs"] if r["run_directory"] != run_record["run_directory"]]
    ledger["episode"]["runs"].append(run_record)
    ledger_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "datasets": len(records),
                      "passed_points": run_record["passed_points"], "points": len(regenerated)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    register_parser = sub.add_parser("register")
    register_parser.add_argument("--output", type=Path, required=True)
    register_parser.add_argument("--version", type=int, choices=[1, 2], default=1)
    plan_parser = sub.add_parser("plan-allocation")
    plan_parser.add_argument("--design", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--design", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--workers", type=int, default=1)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--run", type=Path, required=True)
    report_parser.add_argument("--output-ledger", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "register":
        with args.output.open("x") as stream:
            design = registration() if args.version == 1 else short_query_registration()
            stream.write(json.dumps(design, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    elif args.action == "plan-allocation":
        plan_allocation(args.design, args.output)
    elif args.action == "run":
        run(args.design, args.output, args.workers)
    else:
        report(args.run, args.output_ledger)


if __name__ == "__main__":
    main()
