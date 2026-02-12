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

Optional plots (requires matplotlib):
  paper_e4_bind_vs_weight_py.svg
  paper_e4_root_ceiling_fit_vs_weight_py.svg
  paper_e4_interval_fingerprint_heatmap_py.svg
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


def freq_to_cents_class(anchor_hz: float, freq_hz: float) -> float | None:
    if (
        not math.isfinite(anchor_hz)
        or not math.isfinite(freq_hz)
        or anchor_hz <= 0.0
        or freq_hz <= 0.0
    ):
        return None
    cents = 1200.0 * math.log2(freq_hz / anchor_hz)
    cents = cents % 1200.0
    if cents >= 1200.0 - 1e-6:
        cents = 0.0
    return max(0.0, min(1200.0, cents))


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


def build_interval_heatmap_rows(
    tail_rows: list[dict[str, str]], anchor_hz: float, bin_width_cents: float
) -> tuple[list[float], list[float], list[list[float]]]:
    bin_width = max(1.0, float(bin_width_cents))
    n_bins = max(1, int(math.ceil(1200.0 / bin_width)))

    by_weight_counts: dict[float, list[int]] = {}
    for row in tail_rows:
        w = round(float(row["mirror_weight"]), 6)
        freq_hz = float(row["freq_hz"])
        cents = freq_to_cents_class(anchor_hz, freq_hz)
        if cents is None:
            continue
        counts = by_weight_counts.setdefault(w, [0] * n_bins)
        idx = int(cents // bin_width)
        if idx >= n_bins:
            idx = n_bins - 1
        counts[idx] += 1

    weights = sorted(by_weight_counts.keys())
    if not weights:
        centers = [(i + 0.5) * bin_width for i in range(n_bins)]
        return [], centers, []

    centers = [(i + 0.5) * bin_width for i in range(n_bins)]
    matrix: list[list[float]] = []
    for w in weights:
        counts = by_weight_counts[w]
        total = float(sum(counts))
        if total <= 0.0:
            matrix.append([0.0] * n_bins)
        else:
            matrix.append([c / total for c in counts])
    return weights, centers, matrix


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None
    return plt


def render_bind_vs_weight_plot(path: Path, summary_rows: list[dict[str, float | int]]) -> bool:
    plt = try_import_matplotlib()
    if plt is None or not summary_rows:
        return False
    rows = sorted(summary_rows, key=lambda r: float(r["mirror_weight"]))
    x = [float(r["mirror_weight"]) for r in rows]
    y = [float(r["mean_delta_bind"]) for r in rows]
    lo = [float(r["delta_bind_ci_lo"]) for r in rows]
    hi = [float(r["delta_bind_ci_hi"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.axhline(0.0, color="black", linewidth=2.2, alpha=0.85)
    ax.fill_between(x, lo, hi, color="#4c78a8", alpha=0.2, linewidth=0.0)
    ax.plot(x, y, color="#2f5597", linewidth=2.0, marker="o", markersize=3.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("mirror_weight")
    ax.set_ylabel("DeltaBind")
    ax.set_title("E4 DeltaBind vs mirror_weight (mean ± 95% CI)")
    ax.grid(alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def render_root_ceiling_plot(path: Path, summary_rows: list[dict[str, float | int]]) -> bool:
    plt = try_import_matplotlib()
    if plt is None or not summary_rows:
        return False
    rows = sorted(summary_rows, key=lambda r: float(r["mirror_weight"]))
    x = [float(r["mirror_weight"]) for r in rows]
    root_m = [float(r["mean_root_fit"]) for r in rows]
    root_lo = [float(r["root_ci_lo"]) for r in rows]
    root_hi = [float(r["root_ci_hi"]) for r in rows]
    ceil_m = [float(r["mean_ceiling_fit"]) for r in rows]
    ceil_lo = [float(r["ceiling_ci_lo"]) for r in rows]
    ceil_hi = [float(r["ceiling_ci_hi"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.fill_between(x, root_lo, root_hi, color="#4c78a8", alpha=0.17, linewidth=0.0)
    ax.plot(x, root_m, color="#2f5597", linewidth=2.0, marker="o", markersize=3.0, label="RootFit")
    ax.fill_between(x, ceil_lo, ceil_hi, color="#e45756", alpha=0.15, linewidth=0.0)
    ax.plot(
        x,
        ceil_m,
        color="#b22222",
        linewidth=2.0,
        marker="^",
        markersize=3.2,
        label="CeilingFit",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("mirror_weight")
    ax.set_ylabel("fit")
    ax.set_title("E4 RootFit / CeilingFit vs mirror_weight")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def render_interval_heatmap_plot(
    path: Path,
    weights: list[float],
    centers: list[float],
    matrix: list[list[float]],
) -> bool:
    plt = try_import_matplotlib()
    if plt is None or not weights or not centers or not matrix:
        return False
    n_weights = len(weights)
    n_bins = len(centers)
    matrix_t: list[list[float]] = [[matrix[x][y] for x in range(n_weights)] for y in range(n_bins)]
    bin_width = centers[1] - centers[0] if len(centers) > 1 else 25.0

    if n_weights == 1:
        x_min = max(0.0, weights[0] - 0.05)
        x_max = min(1.0, weights[0] + 0.05)
    else:
        x_min = max(0.0, weights[0] - 0.5 * (weights[1] - weights[0]))
        x_max = min(1.0, weights[-1] + 0.5 * (weights[-1] - weights[-2]))
    y_min = max(0.0, centers[0] - 0.5 * bin_width)
    y_max = min(1200.0, centers[-1] + 0.5 * bin_width)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    image = ax.imshow(
        matrix_t,
        aspect="auto",
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="magma",
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("density")
    for cents, label in [
        (112.0, "b2"),
        (316.0, "m3"),
        (386.0, "M3"),
        (498.0, "P4"),
        (702.0, "P5"),
    ]:
        ax.axhline(cents, color="white", linewidth=0.8, alpha=0.45)
        ax.text(
            x_min + 0.012 * (x_max - x_min + 1e-6),
            cents + 8.0,
            label,
            color="white",
            alpha=0.85,
            fontsize=8,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1200.0)
    ax.set_xlabel("mirror_weight")
    ax.set_ylabel("interval cents (pitch class)")
    ax.set_title("E4 Interval Fingerprint Heatmap")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


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
    parser.add_argument("--anchor-hz", type=float, default=196.0)
    parser.add_argument("--heatmap-bin-cents", type=float, default=25.0)
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        help="Render DeltaBind/fit/heatmap figures (requires matplotlib).",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip figure rendering and only write CSV.",
    )
    parser.set_defaults(plot=True)
    parser.add_argument(
        "--bind-plot-out",
        type=Path,
        default=None,
        help="Output DeltaBind-vs-weight plot path (svg/pdf).",
    )
    parser.add_argument(
        "--fit-plot-out",
        type=Path,
        default=None,
        help="Output RootFit/CeilingFit plot path (svg/pdf).",
    )
    parser.add_argument(
        "--heatmap-plot-out",
        type=Path,
        default=None,
        help="Output interval heatmap path (svg/pdf).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path: Path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    metrics_out = args.metrics_out or input_path.with_name("e4_bind_metrics.csv")
    summary_out = args.summary_out or input_path.with_name("e4_bind_summary.csv")
    bind_plot_out = args.bind_plot_out or input_path.with_name("paper_e4_bind_vs_weight_py.svg")
    fit_plot_out = args.fit_plot_out or input_path.with_name(
        "paper_e4_root_ceiling_fit_vs_weight_py.svg"
    )
    heatmap_plot_out = args.heatmap_plot_out or input_path.with_name(
        "paper_e4_interval_fingerprint_heatmap_py.svg"
    )
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

    plotted = False
    if args.plot:
        bind_plot_out.parent.mkdir(parents=True, exist_ok=True)
        fit_plot_out.parent.mkdir(parents=True, exist_ok=True)
        heatmap_plot_out.parent.mkdir(parents=True, exist_ok=True)
        wrote_bind = render_bind_vs_weight_plot(bind_plot_out, summary_rows)
        wrote_fit = render_root_ceiling_plot(fit_plot_out, summary_rows)
        weights, centers, matrix = build_interval_heatmap_rows(
            tail_rows,
            anchor_hz=float(args.anchor_hz),
            bin_width_cents=float(args.heatmap_bin_cents),
        )
        wrote_heat = render_interval_heatmap_plot(heatmap_plot_out, weights, centers, matrix)
        plotted = wrote_bind or wrote_fit or wrote_heat
        if wrote_bind:
            print(f"write {bind_plot_out}")
        if wrote_fit:
            print(f"write {fit_plot_out}")
        if wrote_heat:
            print(f"write {heatmap_plot_out}")
        if not plotted:
            print("note: plotting skipped (matplotlib unavailable or no data)")

    print(f"write {metrics_out}")
    print(f"write {summary_out}")
    print(f"runs={len(metric_rows)} weights={len(summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
