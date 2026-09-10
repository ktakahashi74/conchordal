#!/usr/bin/env python3
"""Past-only residual forecasts for frozen local energy predictions.

This compares estimators on current-policy audio. It does not estimate the
causal effect, musical value, or cognitive lifetime of a participation action.
"""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


MODELS = ("mean_loss", "mean_residual", "geometry", "context",
          "loss_geometry", "loss_context")
RESIDUAL_MODELS = ("mean_residual", "geometry", "context")
LEADS = (0, 10, 20, 50, 100, 200, 400)


class CenteredRidge:
    """Unpenalized intercept and fixed-penalty centered slopes, no forgetting."""

    def __init__(self, dimensions, ridge=1.0, outputs=3):
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        self.n = 0
        self.mean_x = np.zeros(dimensions)
        self.mean_y = np.zeros(outputs)
        self.inverse = np.eye(dimensions) / ridge
        self.cross = np.zeros((dimensions, outputs))

    def update(self, x, y):
        dx, dy = x - self.mean_x, y - self.mean_y
        self.n += 1
        self.mean_x += dx / self.n
        self.mean_y += dy / self.n
        factor = (self.n - 1) / self.n
        projected = self.inverse @ dx
        self.inverse -= (factor / (1.0 + factor * (dx @ projected))) * np.outer(
            projected, projected
        )
        self.cross += factor * np.outer(dx, dy)

    def predict(self, x):
        if not self.n:
            return None
        return self.mean_y + ((x - self.mean_x) @ self.inverse) @ self.cross


def known_query(row):
    """Read only issuance fields; the observed outcome is deliberately absent."""
    bands = {
        key: np.asarray(row[key], dtype=np.float32).astype(float)
        for key in ("recurrence", "history", "mixed", "history_weight")
    }
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in bands.values()):
        raise ValueError("invalid issued three-band prediction")
    geometry = np.r_[bands["recurrence"], bands["history"]]
    norm = np.linalg.norm(geometry)
    if norm:
        geometry /= norm
    geometry = np.r_[geometry, bands["history_weight"]]
    context = row["issued_features"]
    if context is not None:
        context = np.asarray(context, dtype=float)
        if context.shape != (57,) or not np.isfinite(context).all() or context[0] != 1:
            raise ValueError("invalid frozen auditory context")
        rms = context[1:28].copy()
        norm = np.linalg.norm(rms)
        if norm:
            rms /= norm
        context = np.r_[geometry, rms, context[28:36] / np.sqrt(8),
                        context[36:] / np.sqrt(21)]
    return dict(a=bands["recurrence"], m=bands["mixed"],
                geometry=geometry, context=context)


class ResidualComparison:
    def __init__(self, ridge=1.0):
        self.geometry = CenteredRidge(9, ridge)
        self.context = CenteredRidge(65, ridge)
        self.loss_geometry = CenteredRidge(9, ridge, outputs=1)
        self.loss_context = CenteredRidge(65, ridge, outputs=1)
        self.mean_loss = 0.0

    def update(self, query, observed):
        # All models use the same available-context training set.
        if query["context"] is None:
            return
        a, m = query["a"], query["m"]
        residual = observed - a
        loss = float(np.sum((m - observed) ** 2) - np.sum(residual ** 2))
        self.geometry.update(query["geometry"], residual)
        self.context.update(query["context"], residual)
        self.loss_geometry.update(query["geometry"], np.array([loss]))
        self.loss_context.update(query["context"], np.array([loss]))
        self.mean_loss += (loss - self.mean_loss) / self.geometry.n

    def predict(self, *, a, m, geometry, context):
        if not self.geometry.n or context is None:
            return None
        residuals = dict(mean_residual=self.geometry.mean_y.copy(),
                         geometry=self.geometry.predict(geometry),
                         context=self.context.predict(context))
        delta = m - a
        losses = {name: float(delta @ delta - 2 * delta @ residual)
                  for name, residual in residuals.items()}
        losses["mean_loss"] = self.mean_loss
        # Identical issued candidates have equal loss for every possible outcome.
        losses["loss_geometry"] = (float(self.loss_geometry.predict(geometry)[0])
                                   if np.any(delta != 0) else 0.0)
        losses["loss_context"] = (float(self.loss_context.predict(context)[0])
                                  if np.any(delta != 0) else 0.0)
        return dict(loss_change=losses,
                    residual={name: r.tolist() for name, r in residuals.items()})


def evaluate_group(rows, ridge=1.0):
    """Deliver completions before each query, never at the target's issuance."""
    rows = sorted(rows, key=lambda r: r["target_end_frame"])
    requests = [r["requested_frame"] for r in rows]
    ends = [r["target_end_frame"] for r in rows]
    if requests != sorted(requests) or len(set(ends)) != len(ends):
        raise ValueError("expected unique ordered targets for one voice and horizon")
    queries = [known_query(row) for row in rows]
    observed = [np.asarray(r["observed"], dtype=np.float32).astype(float) for r in rows]
    if any(y.shape != (3,) or not np.isfinite(y).all() for y in observed):
        raise ValueError("invalid observed three-band energy")
    learner = ResidualComparison(ridge)
    completed = 0
    for i, row in enumerate(rows):
        request = row["requested_frame"]
        if not row["forecast_observed_frame"] <= request < row["target_end_frame"]:
            raise ValueError("request outside its observed-to-target boundary")
        while completed < len(rows) and ends[completed] <= request:
            learner.update(queries[completed], observed[completed])
            completed += 1
        if completed != row["completed_before_issue"] or completed > i:
            raise ValueError("prior completion count disagrees with actual clocks")
        query = queries[i]
        predicted = learner.predict(**query)
        delta, residual = query["m"] - query["a"], observed[i] - query["a"]
        recurrence_sse = float(residual @ residual)
        mixed_sse = float(np.sum((query["m"] - observed[i]) ** 2))
        actual_change = mixed_sse - recurrence_sse
        identity = float(delta @ delta - 2 * delta @ residual)
        if not np.isclose(actual_change, identity, rtol=2e-12,
                          atol=1e-14 * max(recurrence_sse + mixed_sse, 1e-30)):
            raise ValueError("loss identity failed")
        choices = {}
        for model in MODELS:
            use_mixed = predicted is None or predicted["loss_change"][model] < 0
            choices[model] = dict(forecast="mixed" if use_mixed else "recurrence",
                                  energy_sse=mixed_sse if use_mixed else recurrence_sse)
        yield dict(
            voice_id=row["voice_id"], issued_step=row["issued_step"],
            horizon_steps=row["target_step"] - row["issued_step"],
            requested_frame=request, request_sec=request / row["sample_rate"],
            target_end_frame=row["target_end_frame"], completed=completed,
            training_count=learner.geometry.n,
            last_completed_frame=ends[completed - 1] if completed else None,
            context_available=query["context"] is not None,
            nonzero_delta=bool(np.any(delta != 0)),
            predicted=predicted, actual_residual=residual.tolist(),
            actual_loss_change=actual_change, recurrence_sse=recurrence_sse,
            mixed_sse=mixed_sse, choices=choices,
        )


def empty_scores():
    return dict(n=0, eligible=0, unavailable=0, recurrence_sse=0.0, mixed_sse=0.0,
                eligible_recurrence_sse=0.0, eligible_mixed_sse=0.0,
                unavailable_mixed_sse=0.0,
                models={name: dict(loss_change_sse=0.0, residual_sse=0.0
                                   if name in RESIDUAL_MODELS else None,
                                   selected_sse=0.0, eligible_selected_sse=0.0,
                                   selected_mixed=0) for name in MODELS})


def add_score(scores, row):
    scores["n"] += 1
    scores["recurrence_sse"] += row["recurrence_sse"]
    scores["mixed_sse"] += row["mixed_sse"]
    predicted = row["predicted"]
    scores["eligible" if predicted is not None else "unavailable"] += 1
    if predicted is None:
        scores["unavailable_mixed_sse"] += row["mixed_sse"]
    else:
        scores["eligible_recurrence_sse"] += row["recurrence_sse"]
        scores["eligible_mixed_sse"] += row["mixed_sse"]
    for name, result in scores["models"].items():
        result["selected_sse"] += row["choices"][name]["energy_sse"]
        result["selected_mixed"] += row["choices"][name]["forecast"] == "mixed"
        if predicted is not None:
            result["eligible_selected_sse"] += row["choices"][name]["energy_sse"]
            result["loss_change_sse"] += (
                predicted["loss_change"][name] - row["actual_loss_change"]
            ) ** 2
            if name in RESIDUAL_MODELS:
                result["residual_sse"] += float(np.sum((
                    np.asarray(predicted["residual"][name]) - row["actual_residual"]
                ) ** 2))


def run(plan_path, output):
    plan_path, output = Path(plan_path), Path(output)
    plan = json.loads(plan_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    pooled = defaultdict(empty_scores)
    cases, inputs = [], []
    for case in plan["cases"]:
        source = Path(case["input"])
        if not source.is_absolute():
            source = plan_path.parent / source
        groups = defaultdict(list)
        with source.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["type"] == "local_prediction_match":
                    lead = row["target_step"] - row["issued_step"]
                    if lead not in LEADS:
                        raise ValueError("unregistered horizon")
                    groups[(row["voice_id"], lead)].append(row)
        if not groups:
            raise ValueError("input contains no completed local predictions")
        scores = {(cutoff, lead, stratum): empty_scores()
                  for cutoff in (0, 8) for lead in LEADS
                  for stratum in ("all", "nonzero_delta")}
        with gzip.open(output / (case["case"] + ".jsonl.gz"), "wt") as trace:
            for group in groups.values():
                for row in evaluate_group(group, plan["ridge"]):
                    trace.write(json.dumps(row, allow_nan=False) + "\n")
                    for cutoff in (0, 8):
                        if row["request_sec"] < cutoff:
                            continue
                        for stratum in ("all", "nonzero_delta"):
                            if stratum == "nonzero_delta" and not row["nonzero_delta"]:
                                continue
                            add_score(scores[(cutoff, row["horizon_steps"], stratum)], row)
                            add_score(pooled[(case["role"], cutoff, stratum)], row)
        comparisons = [dict(request_cutoff_sec=k[0], horizon_steps=k[1], stratum=k[2], **v)
                       for k, v in scores.items()]
        cases.append(dict(case=case["case"], role=case["role"], comparisons=comparisons))
        inputs.append(dict(case=case["case"], path=str(source.resolve()),
                           sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        print(case["case"], sum(len(g) for g in groups.values()), "matches", flush=True)
    result = dict(ridge=plan["ridge"], geometry_features=9, context_features=65,
                  separate_intercept=True, cases=cases,
                  pooled=[dict(role=k[0], request_cutoff_sec=k[1], stratum=k[2], **v)
                          for k, v in pooled.items()])
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps(dict(
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy_version=np.__version__), indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.output)
