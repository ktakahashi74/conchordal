"""Finite stage-2 ordinal fitting on frozen path-local features.

This statistical helper fits the whole-mixture likelihood. It does not extract
audio, create OOF features, select folds, calibrate, or authorize data collection.
"""

import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


PLAN = json.loads((Path(__file__).resolve().parents[1] /
                   "docs/roadmap/temporal-dcc/ordinal-fit-plan.json").read_text())


def crossed_partitions(source_folds, participant_folds):
    source = np.asarray(source_folds)
    participant = np.asarray(participant_folds)
    source_count, participant_count = PLAN["development_splits"]["crossed_folds"]
    if (source.ndim != 1 or source.shape != participant.shape or not len(source)
            or not np.isfinite(source).all() or not np.isfinite(participant).all()
            or np.any(source != np.floor(source)) or np.any(participant != np.floor(participant))
            or np.any((source < 0) | (source >= source_count)) or np.any((participant < 0) | (participant >= participant_count))):
        raise ValueError("frozen source-family and participant fold IDs in0..4 required")
    partitions = []
    for s in range(source_count):
        for p in range(participant_count):
            train = np.flatnonzero((source != s) & (participant != p))
            heldout = np.flatnonzero((source == s) & (participant == p))
            if not len(train) or not len(heldout):
                raise ValueError("every registered crossed fold needs training and scored targets")
            partitions.append((train, heldout))
    return partitions


def prepare(features, response_indices, path_weights, observed_coverage, targets, fixed_columns=(0,), training_prior=None):
    x = np.asarray(features, dtype=np.float64)
    index = np.asarray(response_indices)
    weights = np.asarray(path_weights, dtype=np.float64)
    coverage = np.asarray(observed_coverage, dtype=np.float64)
    y = np.asarray(targets)
    limits = PLAN["numerical_input_envelope"]
    if (x.ndim != 2 or x.shape[1] < 1 or x.shape[1] > limits["max_input_count"]
            or x.shape[0] > limits["max_total_supported_rows"] or x.nbytes > limits["max_feature_bytes"]
            or y.ndim != 1 or not 0 < len(y) <= limits["max_response_count"]
            or coverage.shape != y.shape or index.shape != (len(x),) or weights.shape != index.shape
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.any(y != np.floor(y)) or np.any((y < 0) | (y > 4))
            or not np.isfinite(index).all() or np.any(index != np.floor(index))
            or np.any((index < 0) | (index >= len(y)))
            or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1))
            or not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1))
            or not np.all(x[:, 0] == 1)):
        raise ValueError("finite aligned features, normalized factors, coverage and five-level targets required")
    index = index.astype(np.int64)
    y = y.astype(np.int64)
    fixed = tuple(fixed_columns)
    if (0 not in fixed or len(set(fixed)) != len(fixed)
            or any(not isinstance(i, int) or i < 0 or i >= x.shape[1] for i in fixed)
            or any(len(x) and not np.all(x[:, i] == x[0, i]) for i in fixed)):
        raise ValueError("only declared constant columns can have fixed-zero coefficients")
    support = np.bincount(index, weights=weights, minlength=len(y))
    if (np.any(support > 1 + 1e-12)
            or np.any(np.bincount(index, minlength=len(y)) > limits["max_supported_rows_per_response"])):
        raise ValueError("path mass or capacity exceeded")
    weights = weights.copy()
    oversized_roundoff = support[index] > 1.0
    weights[oversized_roundoff] /= support[index[oversized_roundoff]]
    if training_prior is None:
        prior_counts = np.bincount(y, minlength=5).astype(np.float64) + 0.5
        prior = prior_counts / prior_counts.sum()
        prior_source = "training_responses"
    else:
        prior = np.asarray(training_prior, dtype=np.float64).copy()
        if prior.shape != (5,) or not np.isfinite(prior).all() or np.any(prior <= 0) or abs(prior.sum()-1) > 1e-12:
            raise ValueError("evaluation requires the frozen positive normalized training prior")
        prior_source = "frozen_training_prior"
    # Own immutable copies: a caller cannot edit trial data during optimization.
    arrays = [x.copy(), index.copy(), weights.copy(), coverage.copy(), y.copy(), prior,
              np.minimum(support, 1.0), np.array([i for i in range(x.shape[1]) if i not in fixed], dtype=np.int64)]
    for array in arrays:
        array.flags.writeable = False
    return dict(zip(("x", "index", "weights", "coverage", "y", "prior", "support", "free"), arrays), prior_source=prior_source)


def objective(parameters, data, penalty):
    free = data["free"]
    n_beta = len(free)
    parameters = np.asarray(parameters, dtype=np.float64)
    if (parameters.shape != (n_beta + 4,) or not np.isfinite(parameters).all()
            or not np.isfinite(penalty) or penalty < 0):
        raise ValueError("finite aligned ordinal parameters and nonnegative penalty required")
    beta = parameters[:n_beta]
    steps = parameters[n_beta + 1:]
    cuts = parameters[n_beta] + np.r_[0.0, np.cumsum(np.logaddexp(0.0, steps) + 1e-8)]
    full_beta = np.zeros(data["x"].shape[1])
    full_beta[free] = beta
    eta = data["x"] @ full_beta
    logits = cuts[None, :] - eta[:, None]
    if not np.isfinite(logits).all() or np.any(np.diff(cuts) <= 0):
        raise FloatingPointError("ordinal arithmetic range or cutpoint ordering lost")
    labels = data["y"][data["index"]]
    log_p = np.empty(len(eta))
    d_eta = np.empty(len(eta))
    d_cuts = np.zeros((len(eta), 4))
    for category in range(5):
        chosen = labels == category
        if category == 0:
            a = logits[chosen, 0]
            log_p[chosen] = -np.logaddexp(0.0, -a)
            d_eta[chosen] = -expit(-a)
            d_cuts[chosen, 0] = expit(-a)
        elif category == 4:
            a = logits[chosen, 3]
            log_p[chosen] = -np.logaddexp(0.0, a)
            d_eta[chosen] = expit(a)
            d_cuts[chosen, 3] = -expit(a)
        else:
            a, b = logits[chosen, category - 1], logits[chosen, category]
            width = cuts[category] - cuts[category - 1]
            log_p[chosen] = -np.logaddexp(0.0, -b) - np.logaddexp(0.0, a) + np.log(-np.expm1(-width))
            d_eta[chosen] = expit(a) + expit(b) - 1
            with np.errstate(over="ignore"):
                inverse_gap = 1 / np.expm1(width)
            d_cuts[chosen, category - 1] = -expit(a) - inverse_gap
            d_cuts[chosen, category] = expit(-b) + inverse_gap
    reported_support = data["coverage"] * data["support"]
    with np.errstate(divide="ignore"):
        response_log = np.log1p(-reported_support) + np.log(data["prior"][data["y"]])
        row_log = np.log(data["coverage"][data["index"]]) + np.log(data["weights"]) + log_p
    np.logaddexp.at(response_log, data["index"], row_log)
    responsibilities = np.exp(row_log - response_log[data["index"]])
    loss = -np.mean(response_log) + 0.5 * penalty * (beta @ beta)
    cut_gradient = -(responsibilities[:, None] * d_cuts).sum(axis=0) / len(data["y"])
    gradient = np.r_[
        -(data["x"].T @ (responsibilities * d_eta))[free] / len(data["y"]) + penalty * beta,
        cut_gradient.sum(),
        expit(steps) * np.cumsum(cut_gradient[::-1])[::-1][1:],
    ]
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise FloatingPointError("nonfinite full-mixture objective or gradient")
    return float(loss), gradient


def fit(data, penalty, max_evaluations=None):
    solver = PLAN["solver"]
    if data["prior_source"] != "training_responses":
        raise ValueError("evaluation data with a frozen prior cannot enter a training fit")
    if penalty not in PLAN["lambda_grid"]:
        raise ValueError("penalty must be a preregistered candidate")
    limit = solver["max_evaluations"] if max_evaluations is None else max_evaluations
    if not isinstance(limit, int) or not 1 <= limit <= solver["max_evaluations"]:
        raise ValueError("evaluation limit cannot exceed the registered envelope")
    if not np.any(data["coverage"] * data["support"] > 0):
        return {"status": "unsupported", "fit_complete": False, "evaluations": 0}
    cumulative = data["prior"].cumsum()[:4]
    initial_cuts = np.log(cumulative) - np.log1p(-cumulative)
    differences = np.diff(initial_cuts) - 1e-8
    if np.any(differences <= 0):
        return {"status": "initial_cutpoint_guard", "fit_complete": False, "evaluations": 0}
    initial = np.r_[np.zeros(len(data["free"])), initial_cuts[0], np.log(np.expm1(differences))]
    lower, upper = solver["delta_bounds"]
    if np.any((initial[-3:] <= lower) | (initial[-3:] >= upper)):
        return {"status": "initial_cutpoint_guard", "fit_complete": False, "evaluations": 0}
    count = 0
    best = None
    started = time.perf_counter()

    def evaluate(parameters):
        nonlocal count, best
        if count >= limit:
            raise RuntimeError("evaluation_budget")
        if time.perf_counter() - started >= solver["max_elapsed_sec_per_job"]:
            raise RuntimeError("elapsed_budget")
        count += 1
        loss, gradient = objective(parameters, data, penalty)
        if best is None or loss < best[0]:
            best = (loss, parameters.copy(), float(np.max(np.abs(gradient))))
        if time.perf_counter() - started >= solver["max_elapsed_sec_per_job"]:
            raise RuntimeError("elapsed_budget")
        return loss, gradient

    try:
        result = minimize(evaluate, initial, jac=True, method="L-BFGS-B",
                          bounds=[(None, None)] * (len(initial) - 3) + [(lower, upper)] * 3,
                          options={"maxiter": solver["max_iterations"], "maxfun": limit,
                                   "maxls": solver["max_line_search_steps"], "ftol": solver["ftol"],
                                   "gtol": solver["gtol"]})
        status = "converged" if result.success else "solver_incomplete"
        if np.any((result.x[-3:] <= lower + 1e-8) | (result.x[-3:] >= upper - 1e-8)):
            status = "cutpoint_guard"
        if result.success and np.max(np.abs(result.jac)) > 10 * solver["gtol"]:
            status = "stationarity_incomplete"
        if status == "converged":
            best = (float(result.fun), result.x.copy(), float(np.max(np.abs(result.jac))))
    except (RuntimeError, FloatingPointError) as error:
        status = str(error)
    elapsed = time.perf_counter() - started
    if elapsed >= solver["max_elapsed_sec_per_job"]:
        status = "elapsed_budget"
    fitted_beta = None
    fitted_cutpoints = None
    if best is not None:
        vector = best[1]
        fitted_beta = np.zeros(data["x"].shape[1])
        fitted_beta[data["free"]] = vector[:len(data["free"])]
        fitted_cutpoints = vector[len(data["free"])] + np.r_[0.0, np.cumsum(np.logaddexp(0.0, vector[-3:]) + 1e-8)]
    return {
        "status": status, "fit_complete": status == "converged", "calibrated": False, "evaluations": count,
        "elapsed_sec": elapsed,
        "objective": None if best is None else best[0],
        "parameters": None if best is None else best[1].tolist(),
        "beta": None if fitted_beta is None else fitted_beta.tolist(),
        "cutpoints": None if fitted_cutpoints is None else fitted_cutpoints.tolist(),
        "gradient_max_abs": None if best is None else best[2],
        "training_prior": data["prior"].tolist(), "free_columns": data["free"].tolist(),
        "penalty": penalty,
    }
