#!/usr/bin/env python3
"""
Compute E4 RootFit / CeilingFit / DeltaBind from e4_tail_agents.csv.

Input CSV columns:
  mirror_weight,seed,step,agent_id,freq_hz

Output CSVs:
  e4_bind_metrics.csv:
    mirror_weight,seed,step,n_agents,root_fit,ceiling_fit,delta_bind
  e4_bind_summary.csv:
    mirror_weight,mean_root_fit,root_ci_lo,root_ci_hi,mean_ceiling_fit,ceiling_ci_lo,ceiling_ci_hi,mean_delta_bind,delta_bind_ci_lo,delta_bind_ci_hi,n_seeds
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path


def cents_distance_abs(a_hz: float, b_hz: float) -> float:
    if not math.isfinite(a_hz) or not math.isfinite(b_hz) or a_hz <= 0.0 or b_hz <= 0.0:
        return float("inf")
    return abs(1200.0 * math.log2(a_hz / b_hz))


def harmonic_fit_score(distance_cents: float, harmonic_index: int, sigma_cents: float, rho: float) -> float:
    sigma = max(sigma_cents, 1e-6)
    n = max(harmonic_index, 1)
    decay = n ** (-rho)
    return math.exp(-0.5 * (distance_cents / sigma) ** 2) * decay


def unique_candidates(raw: list[float]) -> list[float]:
    values = [x for x in raw if math.isfinite(x) and x > 0.0]
    values.sort()
    out: list[float] = []
    for value in values:
        if not out or abs(value - out[-1]) >= 1e-4:
            out.append(value)
    return out


def root_candidate_score(
    freqs: list[float], candidate_root_hz: float, sigma_cents: float, rho: float, max_harmonic: int
) -> float:
    if not math.isfinite(candidate_root_hz) or candidate_root_hz <= 0.0:
        return 0.0
    total = 0.0
    for freq in freqs:
        if not math.isfinite(freq) or freq <= 0.0:
            continue
        best = 0.0
        for n in range(1, max_harmonic + 1):
            target = candidate_root_hz * n
            if not math.isfinite(target) or target <= 0.0:
                continue
            score = harmonic_fit_score(
                cents_distance_abs(freq, target), n, sigma_cents, rho
            )
            if score > best:
                best = score
        total += best
    return total


def ceiling_candidate_score(
    freqs: list[float], candidate_ceiling_hz: float, sigma_cents: float, rho: float, max_harmonic: int
) -> float:
    if not math.isfinite(candidate_ceiling_hz) or candidate_ceiling_hz <= 0.0:
        return 0.0
    total = 0.0
    for freq in freqs:
        if not math.isfinite(freq) or freq <= 0.0:
            continue
        best = 0.0
        for n in range(1, max_harmonic + 1):
            target = candidate_ceiling_hz / n
            if not math.isfinite(target) or target <= 0.0:
                continue
            score = harmonic_fit_score(
                cents_distance_abs(freq, target), n, sigma_cents, rho
            )
            if score > best:
                best = score
        total += best
    return total


def root_fit(freqs: list[float], sigma_cents: float, rho: float, max_harmonic: int, top_k: int) -> float:
    candidates: list[float] = []
    for freq in freqs:
        if not math.isfinite(freq) or freq <= 0.0:
            continue
        for n in range(1, max_harmonic + 1):
            candidates.append(freq / n)
    candidates = unique_candidates(candidates)
    if not candidates:
        return 0.0
    scored = [
        (candidate, root_candidate_score(freqs, candidate, sigma_cents, rho, max_harmonic))
        for candidate in candidates
    ]
    scored.sort(key=lambda x: (-x[1], x[0]))
    scored = scored[: max(1, min(top_k, len(scored)))]
    return scored[0][1] if scored else 0.0


def ceiling_fit(
    freqs: list[float], sigma_cents: float, rho: float, max_harmonic: int, top_k: int
) -> float:
    candidates: list[float] = []
    for freq in freqs:
        if not math.isfinite(freq) or freq <= 0.0:
            continue
        for n in range(1, max_harmonic + 1):
            candidates.append(freq * n)
    candidates = unique_candidates(candidates)
    if not candidates:
        return 0.0
    scored = [
        (candidate, ceiling_candidate_score(freqs, candidate, sigma_cents, rho, max_harmonic))
        for candidate in candidates
    ]
    scored.sort(key=lambda x: (-x[1], x[0]))
    scored = scored[: max(1, min(top_k, len(scored)))]
    return scored[0][1] if scored else 0.0


def compute_bind_metrics(
    freqs: list[float], sigma_cents: float, rho: float, max_harmonic: int, top_k: int
) -> tuple[float, float, float]:
    clean = [f for f in freqs if math.isfinite(f) and f > 0.0]
    if not clean:
        return 0.0, 0.0, 0.0
    root = root_fit(clean, sigma_cents, rho, max_harmonic, top_k)
    ceiling = ceiling_fit(clean, sigma_cents, rho, max_harmonic, top_k)
    delta = (root - ceiling) / (root + ceiling + 1e-6)
    if delta > 1.0:
        delta = 1.0
    elif delta < -1.0:
        delta = -1.0
    return root, ceiling, delta


def bootstrap_mean_ci95(values: list[float], iters: int, seed: int) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2 or iters <= 0:
        return mean, mean, mean
    rng = random.Random(seed)
    n = len(values)
    samples: list[float] = []
    for _ in range(iters):
        acc = 0.0
        for _ in range(n):
            acc += values[rng.randrange(n)]
        samples.append(acc / n)
    samples.sort()
    lo_idx = min(int(math.floor(iters * 0.025)), iters - 1)
    hi_idx = min(int(math.floor(iters * 0.975)), iters - 1)
    return mean, samples[lo_idx], samples[hi_idx]


def load_tail_agents(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_metrics_rows(
    tail_rows: list[dict[str, str]],
    sigma_cents: float,
    rho: float,
    max_harmonic: int,
    top_k: int,
) -> list[dict[str, float | int]]:
    latest_step_by_run: dict[tuple[float, int], int] = {}
    for row in tail_rows:
        w = round(float(row["mirror_weight"]), 6)
        seed = int(row["seed"])
        step = int(row["step"])
        key = (w, seed)
        latest_step_by_run[key] = max(latest_step_by_run.get(key, step), step)

    freqs_by_run: dict[tuple[float, int], list[tuple[int, float]]] = defaultdict(list)
    for row in tail_rows:
        w = round(float(row["mirror_weight"]), 6)
        seed = int(row["seed"])
        step = int(row["step"])
        key = (w, seed)
        if latest_step_by_run.get(key) != step:
            continue
        freqs_by_run[key].append((int(row["agent_id"]), float(row["freq_hz"])))

    out: list[dict[str, float | int]] = []
    for key in sorted(freqs_by_run.keys()):
        w, seed = key
        agents = sorted(freqs_by_run[key], key=lambda t: t[0])
        freqs = [freq for _, freq in agents]
        root, ceiling, delta = compute_bind_metrics(freqs, sigma_cents, rho, max_harmonic, top_k)
        out.append(
            {
                "mirror_weight": w,
                "seed": seed,
                "step": latest_step_by_run[key],
                "n_agents": len(freqs),
                "root_fit": root,
                "ceiling_fit": ceiling,
                "delta_bind": delta,
            }
        )
    return out


def build_summary_rows(
    metric_rows: list[dict[str, float | int]], bootstrap_iters: int, bootstrap_seed: int
) -> list[dict[str, float | int]]:
    by_weight: dict[float, list[dict[str, float | int]]] = defaultdict(list)
    for row in metric_rows:
        by_weight[float(row["mirror_weight"])].append(row)

    out: list[dict[str, float | int]] = []
    for w in sorted(by_weight.keys()):
        rows = by_weight[w]
        roots = [float(r["root_fit"]) for r in rows]
        ceilings = [float(r["ceiling_fit"]) for r in rows]
        deltas = [float(r["delta_bind"]) for r in rows]
        weight_key = int(round(w * 1000.0))
        seed = bootstrap_seed ^ 0xE4B1D ^ (weight_key * 0x9E3779B9)
        mean_root, root_lo, root_hi = bootstrap_mean_ci95(roots, bootstrap_iters, seed ^ 0x11)
        mean_ceiling, ceil_lo, ceil_hi = bootstrap_mean_ci95(
            ceilings, bootstrap_iters, seed ^ 0x22
        )
        mean_delta, delta_lo, delta_hi = bootstrap_mean_ci95(
            deltas, bootstrap_iters, seed ^ 0x33
        )
        out.append(
            {
                "mirror_weight": w,
                "mean_root_fit": mean_root,
                "root_ci_lo": root_lo,
                "root_ci_hi": root_hi,
                "mean_ceiling_fit": mean_ceiling,
                "ceiling_ci_lo": ceil_lo,
                "ceiling_ci_hi": ceil_hi,
                "mean_delta_bind": mean_delta,
                "delta_bind_ci_lo": delta_lo,
                "delta_bind_ci_hi": delta_hi,
                "n_seeds": len(rows),
            }
        )
    return out


def write_metrics_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    fields = [
        "mirror_weight",
        "seed",
        "step",
        "n_agents",
        "root_fit",
        "ceiling_fit",
        "delta_bind",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    fields = [
        "mirror_weight",
        "mean_root_fit",
        "root_ci_lo",
        "root_ci_hi",
        "mean_ceiling_fit",
        "ceiling_ci_lo",
        "ceiling_ci_hi",
        "mean_delta_bind",
        "delta_bind_ci_lo",
        "delta_bind_ci_hi",
        "n_seeds",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute E4 Root/Ceiling bind metrics.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("examples/paper/plots/e4/e4_tail_agents.csv"),
        help="Input e4_tail_agents.csv",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Output e4_bind_metrics.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Output e4_bind_summary.csv",
    )
    parser.add_argument("--sigma-cents", type=float, default=15.0)
    parser.add_argument("--rho", type=float, default=0.4)
    parser.add_argument("--max-harmonic", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=0xE4600D)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path: Path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    metrics_out = args.metrics_out or input_path.with_name("e4_bind_metrics.csv")
    summary_out = args.summary_out or input_path.with_name("e4_bind_summary.csv")
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    tail_rows = load_tail_agents(input_path)
    metric_rows = build_metrics_rows(
        tail_rows,
        sigma_cents=args.sigma_cents,
        rho=args.rho,
        max_harmonic=max(1, args.max_harmonic),
        top_k=max(1, args.top_k),
    )
    summary_rows = build_summary_rows(
        metric_rows,
        bootstrap_iters=max(0, args.bootstrap_iters),
        bootstrap_seed=args.bootstrap_seed,
    )
    write_metrics_csv(metrics_out, metric_rows)
    write_summary_csv(summary_out, summary_rows)

    print(f"write {metrics_out}")
    print(f"write {summary_out}")
    print(f"runs={len(metric_rows)} weights={len(summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
