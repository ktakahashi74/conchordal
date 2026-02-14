use std::collections::HashMap;
use std::error::Error;
use std::fs::{create_dir_all, read_to_string, write};
use std::path::{Path, PathBuf};

use plotters::prelude::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

const DEFAULT_INPUT: &str = "examples/paper/plots/e4/paper_e4_abcd_trace.csv";
const DEFAULT_OUTDIR: &str = "examples/paper/plots/e4";
const DEFAULT_MAX_LAG: i32 = 10;
const BOOTSTRAP_ITERS: usize = 4000;
const BOOTSTRAP_SEED: u64 = 0xE4AB_CD00;
type AQuantilePoint = (u32, f32, f32, f32);
type ASeriesKey = (String, i32);

#[derive(Clone, Debug)]
struct Cli {
    input: PathBuf,
    outdir: PathBuf,
    max_lag: i32,
}

#[derive(Clone, Debug)]
struct TraceRow {
    run_id: String,
    seed: u64,
    wr: f32,
    mirror_weight: f32,
    timing_mode: String,
    step: u32,
    a: f32,
    agent_idx: Option<i32>,
    oracle_idx: Option<i32>,
    agent_log2: Option<f32>,
    oracle_log2: Option<f32>,
}

#[derive(Clone, Debug)]
struct RunSeries {
    run_id: String,
    seed: u64,
    wr: f32,
    mirror_weight: f32,
    timing_mode: String,
    rows: Vec<TraceRow>,
}

#[derive(Clone, Debug)]
struct RunLagResult {
    mirror_weight: f32,
    timing_mode: String,
    best_lag: i32,
    delta_a: f32,
    lag_scores: Vec<(i32, f32)>,
}

#[derive(Clone, Debug)]
struct SummaryRow {
    mirror_weight: f32,
    timing_mode: String,
    n_runs: usize,
    best_lag_mode: i32,
    best_lag_mean: f32,
    best_lag_ci_lo: f32,
    best_lag_ci_hi: f32,
    delta_a_mean: f32,
    delta_a_ci_lo: f32,
    delta_a_ci_hi: f32,
    nonzero_best_lag_frac: f32,
    dominant_lag_sign: i32,
}

fn usage() -> String {
    [
        "Usage: e4_abcd_analyze [--input PATH] [--outdir DIR] [--max-lag N]",
        &format!("Defaults: --input {DEFAULT_INPUT} --outdir {DEFAULT_OUTDIR} --max-lag {DEFAULT_MAX_LAG}"),
    ]
    .join("\n")
}

fn parse_args(args: &[String]) -> Result<Cli, String> {
    let mut input = PathBuf::from(DEFAULT_INPUT);
    let mut outdir = PathBuf::from(DEFAULT_OUTDIR);
    let mut max_lag = DEFAULT_MAX_LAG;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("missing value after --input\n{}", usage()));
                }
                input = PathBuf::from(&args[i]);
            }
            "--outdir" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("missing value after --outdir\n{}", usage()));
                }
                outdir = PathBuf::from(&args[i]);
            }
            "--max-lag" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("missing value after --max-lag\n{}", usage()));
                }
                max_lag = args[i]
                    .parse::<i32>()
                    .map_err(|e| format!("invalid --max-lag: {e}"))?;
            }
            "--help" | "-h" => return Err(usage()),
            other => {
                return Err(format!("unknown argument: {other}\n{}", usage()));
            }
        }
        i += 1;
    }
    Ok(Cli {
        input,
        outdir,
        max_lag: max_lag.max(1),
    })
}

fn parse_opt_i32(v: Option<&str>) -> Option<i32> {
    let s = v?.trim();
    if s.is_empty() {
        return None;
    }
    s.parse::<i32>().ok()
}

fn parse_opt_f32(v: Option<&str>) -> Option<f32> {
    let s = v?.trim();
    if s.is_empty() {
        return None;
    }
    let x = s.parse::<f32>().ok()?;
    if x.is_finite() { Some(x) } else { None }
}

fn parse_f32_required(cols: &[&str], idx: usize, name: &str) -> Result<f32, String> {
    cols.get(idx)
        .ok_or_else(|| format!("missing required column: {name}"))?
        .parse::<f32>()
        .map_err(|e| format!("invalid {name}: {e}"))
}

fn parse_u64_required(cols: &[&str], idx: usize, name: &str) -> Result<u64, String> {
    cols.get(idx)
        .ok_or_else(|| format!("missing required column: {name}"))?
        .parse::<u64>()
        .map_err(|e| format!("invalid {name}: {e}"))
}

fn parse_u32_required(cols: &[&str], idx: usize, name: &str) -> Result<u32, String> {
    cols.get(idx)
        .ok_or_else(|| format!("missing required column: {name}"))?
        .parse::<u32>()
        .map_err(|e| format!("invalid {name}: {e}"))
}

fn parse_trace_csv(text: &str) -> Result<Vec<TraceRow>, String> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().ok_or_else(|| "empty CSV".to_string())?;
    let header_cols: Vec<&str> = header.split(',').collect();
    let mut col_idx = HashMap::new();
    for (i, c) in header_cols.iter().enumerate() {
        col_idx.insert(c.trim().to_string(), i);
    }
    let required = [
        "run_id",
        "seed",
        "mirror_weight",
        "step",
        "A",
        "B",
        "C",
        "D",
    ];
    for name in required {
        if !col_idx.contains_key(name) {
            return Err(format!("missing required column `{name}` in input CSV"));
        }
    }
    let run_id_i = *col_idx.get("run_id").unwrap_or(&0);
    let seed_i = *col_idx.get("seed").unwrap_or(&0);
    let wr_i = col_idx.get("wr").copied();
    let mirror_i = *col_idx.get("mirror_weight").unwrap_or(&0);
    let timing_i = col_idx.get("timing_mode").copied();
    let step_i = *col_idx.get("step").unwrap_or(&0);
    let a_i = *col_idx.get("A").unwrap_or(&0);
    let b_i = *col_idx.get("B").unwrap_or(&0);
    let c_i = *col_idx.get("C").unwrap_or(&0);
    let d_i = *col_idx.get("D").unwrap_or(&0);
    let agent_idx_i = col_idx.get("agent_idx").copied();
    let oracle_idx_i = col_idx.get("oracle_idx").copied();
    let agent_log2_i = col_idx.get("agent_log2").copied();
    let oracle_log2_i = col_idx.get("oracle_log2").copied();

    let mut out = Vec::new();
    for (line_no, line) in lines.enumerate() {
        let cols: Vec<&str> = line.split(',').collect();
        let run_id = cols
            .get(run_id_i)
            .ok_or_else(|| format!("line {} missing run_id", line_no + 2))?
            .trim()
            .to_string();
        if run_id.is_empty() {
            return Err(format!("line {} has empty run_id", line_no + 2));
        }
        let seed = parse_u64_required(&cols, seed_i, "seed")?;
        let wr = if let Some(i) = wr_i {
            parse_f32_required(&cols, i, "wr")?
        } else {
            1.0
        };
        let mirror_weight = parse_f32_required(&cols, mirror_i, "mirror_weight")?;
        let timing_mode = if let Some(i) = timing_i {
            cols.get(i)
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .unwrap_or("baseline_seq")
                .to_string()
        } else {
            "baseline_seq".to_string()
        };
        let step = parse_u32_required(&cols, step_i, "step")?;
        let a = parse_f32_required(&cols, a_i, "A")?;
        let b = parse_f32_required(&cols, b_i, "B")?;
        let c = parse_f32_required(&cols, c_i, "C")?;
        let d = parse_f32_required(&cols, d_i, "D")?;
        let agent_idx = parse_opt_i32(agent_idx_i.and_then(|i| cols.get(i).copied()));
        let oracle_idx = parse_opt_i32(oracle_idx_i.and_then(|i| cols.get(i).copied()));
        let agent_log2 = parse_opt_f32(agent_log2_i.and_then(|i| cols.get(i).copied()));
        let oracle_log2 = parse_opt_f32(oracle_log2_i.and_then(|i| cols.get(i).copied()));

        if !(a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite()) {
            return Err(format!("line {} contains non-finite A/B/C/D", line_no + 2));
        }
        out.push(TraceRow {
            run_id,
            seed,
            wr,
            mirror_weight,
            timing_mode,
            step,
            a,
            agent_idx,
            oracle_idx,
            agent_log2,
            oracle_log2,
        });
    }
    Ok(out)
}

fn float_key(v: f32) -> i32 {
    (v * 1000.0).round() as i32
}

fn float_from_key(k: i32) -> f32 {
    k as f32 / 1000.0
}

fn group_runs(rows: &[TraceRow]) -> Vec<RunSeries> {
    let mut map: HashMap<(String, u64, i32, i32, String), Vec<TraceRow>> = HashMap::new();
    for row in rows {
        map.entry((
            row.run_id.clone(),
            row.seed,
            float_key(row.wr),
            float_key(row.mirror_weight),
            row.timing_mode.clone(),
        ))
        .or_default()
        .push(row.clone());
    }
    let mut out = Vec::new();
    for ((run_id, seed, wr_key, mirror_key, timing_mode), mut run_rows) in map {
        run_rows.sort_by(|a, b| a.step.cmp(&b.step));
        out.push(RunSeries {
            run_id,
            seed,
            wr: float_from_key(wr_key),
            mirror_weight: float_from_key(mirror_key),
            timing_mode,
            rows: run_rows,
        });
    }
    out.sort_by(|a, b| {
        a.wr.partial_cmp(&b.wr)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.mirror_weight
                    .partial_cmp(&b.mirror_weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.seed.cmp(&b.seed))
            .then_with(|| a.run_id.cmp(&b.run_id))
    });
    out
}

fn alignment_score(agent_row: &TraceRow, oracle_row: &TraceRow) -> Option<f32> {
    if let (Some(agent_idx), Some(oracle_idx)) = (agent_row.agent_idx, oracle_row.oracle_idx) {
        return Some(if agent_idx == oracle_idx { 1.0 } else { 0.0 });
    }
    if let (Some(agent_log2), Some(oracle_log2)) = (agent_row.agent_log2, oracle_row.oracle_log2) {
        let dist = (agent_log2 - oracle_log2).abs();
        return Some((1.0 - dist.min(1.0)).max(0.0));
    }
    if agent_row.a.is_finite() && oracle_row.a.is_finite() {
        return Some((1.0 - (agent_row.a - oracle_row.a).abs().min(1.0)).max(0.0));
    }
    None
}

fn lag_alignment(rows: &[TraceRow], lag: i32, burn_in_step: u32) -> (f32, usize) {
    if rows.is_empty() {
        return (0.0, 0);
    }
    let by_step: HashMap<u32, &TraceRow> = rows.iter().map(|row| (row.step, row)).collect();
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for row in rows {
        if row.step < burn_in_step {
            continue;
        }
        let target_step = row.step as i64 + lag as i64;
        if target_step < 0 {
            continue;
        }
        let Some(other) = by_step.get(&(target_step as u32)) else {
            continue;
        };
        if let Some(score) = alignment_score(row, other)
            && score.is_finite()
        {
            sum += score;
            count += 1;
        }
    }
    if count == 0 {
        (0.0, 0)
    } else {
        (sum / count as f32, count)
    }
}

fn analyze_run(run: &RunSeries, max_lag: i32) -> Option<RunLagResult> {
    if run.rows.is_empty() {
        return None;
    }
    let max_step = run.rows.iter().map(|r| r.step).max().unwrap_or(0);
    let burn_in_step = ((max_step as f32) * 0.25).floor() as u32;

    let mut lag_scores = Vec::new();
    let mut best_lag = 0i32;
    let mut best_score = f32::NEG_INFINITY;
    for lag in (-max_lag)..=max_lag {
        let (score, n_pairs) = lag_alignment(&run.rows, lag, burn_in_step);
        if n_pairs == 0 {
            continue;
        }
        lag_scores.push((lag, score));
        let better = score > best_score + 1e-9;
        let tie = (score - best_score).abs() <= 1e-9;
        let closer_to_zero = lag.abs() < best_lag.abs();
        if better
            || (tie && closer_to_zero)
            || (tie && lag.abs() == best_lag.abs() && lag < best_lag)
        {
            best_score = score;
            best_lag = lag;
        }
    }
    if lag_scores.is_empty() {
        return None;
    }
    let lag0_score = lag_scores
        .iter()
        .find_map(|(lag, score)| if *lag == 0 { Some(*score) } else { None })
        .unwrap_or(0.0);
    Some(RunLagResult {
        mirror_weight: run.mirror_weight,
        timing_mode: run.timing_mode.clone(),
        best_lag,
        delta_a: best_score - lag0_score,
        lag_scores,
    })
}

fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    if values.len() < 2 {
        return (mean, 0.0);
    }
    let var = values
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / (values.len() as f32 - 1.0);
    (mean, var.max(0.0).sqrt())
}

fn bootstrap_mean_ci95(values: &[f32], iters: usize, seed: u64) -> (f32, f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let (mean, _) = mean_std(values);
    if values.len() == 1 || iters == 0 {
        return (mean, mean, mean);
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut means = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut sum = 0.0f32;
        for _ in 0..values.len() {
            let idx = rng.random_range(0..values.len());
            sum += values[idx];
        }
        means.push(sum / values.len() as f32);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo_i = ((iters as f32) * 0.025).floor() as usize;
    let hi_i = ((iters as f32) * 0.975).floor() as usize;
    let lo = means[lo_i.min(iters - 1)];
    let hi = means[hi_i.min(iters - 1)];
    (mean, lo, hi)
}

fn quantile(values: &mut [f32], q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = q.clamp(0.0, 1.0);
    let pos = q * (values.len().saturating_sub(1) as f32);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let t = pos - lo as f32;
        values[lo] * (1.0 - t) + values[hi] * t
    }
}

fn mode_i32(values: &[i32]) -> i32 {
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for &v in values {
        *counts.entry(v).or_default() += 1;
    }
    let mut items: Vec<(i32, usize)> = counts.into_iter().collect();
    items.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| a.0.abs().cmp(&b.0.abs()))
            .then_with(|| a.0.cmp(&b.0))
    });
    items.first().map(|x| x.0).unwrap_or(0)
}

fn summarize_by_mirror(run_results: &[RunLagResult]) -> Vec<SummaryRow> {
    let mut grouped: HashMap<(String, i32), Vec<&RunLagResult>> = HashMap::new();
    for row in run_results {
        grouped
            .entry((row.timing_mode.clone(), float_key(row.mirror_weight)))
            .or_default()
            .push(row);
    }
    let mut out = Vec::new();
    for ((timing_mode, mirror_key), group) in grouped {
        let lag_vals_i32: Vec<i32> = group.iter().map(|r| r.best_lag).collect();
        let lag_vals_f32: Vec<f32> = lag_vals_i32.iter().map(|v| *v as f32).collect();
        let delta_vals: Vec<f32> = group.iter().map(|r| r.delta_a).collect();
        let mode = mode_i32(&lag_vals_i32);
        let nonzero_frac =
            lag_vals_i32.iter().filter(|v| **v != 0).count() as f32 / lag_vals_i32.len() as f32;
        let dominant_lag_sign = mode.signum();
        let seed = BOOTSTRAP_SEED
            ^ (mirror_key as i64 as u64).wrapping_mul(0x9E37_79B9)
            ^ (timing_mode
                .bytes()
                .fold(0u64, |acc, b| acc.wrapping_mul(131).wrapping_add(b as u64)));
        let (best_lag_mean, best_lag_ci_lo, best_lag_ci_hi) =
            bootstrap_mean_ci95(&lag_vals_f32, BOOTSTRAP_ITERS, seed ^ 0x11);
        let (delta_a_mean, delta_a_ci_lo, delta_a_ci_hi) =
            bootstrap_mean_ci95(&delta_vals, BOOTSTRAP_ITERS, seed ^ 0x22);
        out.push(SummaryRow {
            mirror_weight: float_from_key(mirror_key),
            timing_mode,
            n_runs: group.len(),
            best_lag_mode: mode,
            best_lag_mean,
            best_lag_ci_lo,
            best_lag_ci_hi,
            delta_a_mean,
            delta_a_ci_lo,
            delta_a_ci_hi,
            nonzero_best_lag_frac: nonzero_frac,
            dominant_lag_sign,
        });
    }
    out.sort_by(|a, b| {
        a.timing_mode.cmp(&b.timing_mode).then_with(|| {
            a.mirror_weight
                .partial_cmp(&b.mirror_weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
    out
}

fn summary_by_mirror_csv(rows: &[SummaryRow]) -> String {
    let mut out = String::from(
        "timing_mode,mirror_weight,n_runs,best_lag_mode,best_lag_mean,best_lag_ci_lo,best_lag_ci_hi,delta_a_mean,delta_a_ci_lo,delta_a_ci_hi,nonzero_best_lag_frac,dominant_lag_sign\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{},{:.3},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
            row.timing_mode,
            row.mirror_weight,
            row.n_runs,
            row.best_lag_mode,
            row.best_lag_mean,
            row.best_lag_ci_lo,
            row.best_lag_ci_hi,
            row.delta_a_mean,
            row.delta_a_ci_lo,
            row.delta_a_ci_hi,
            row.nonzero_best_lag_frac,
            row.dominant_lag_sign
        ));
    }
    out
}

fn render_a_over_time(out_path: &Path, rows: &[TraceRow]) -> Result<(), Box<dyn Error>> {
    if rows.is_empty() {
        return Ok(());
    }
    let mut grouped: HashMap<(String, i32, u32), Vec<f32>> = HashMap::new();
    let mut max_step = 0u32;
    for row in rows {
        grouped
            .entry((
                row.timing_mode.clone(),
                float_key(row.mirror_weight),
                row.step,
            ))
            .or_default()
            .push(row.a.clamp(0.0, 1.0));
        max_step = max_step.max(row.step);
    }
    let mut series: HashMap<ASeriesKey, Vec<AQuantilePoint>> = HashMap::new();
    for ((timing, mirror_key, step), mut values) in grouped {
        let q25 = quantile(&mut values.clone(), 0.25);
        let q50 = quantile(&mut values.clone(), 0.50);
        let q75 = quantile(&mut values, 0.75);
        series
            .entry((timing, mirror_key))
            .or_default()
            .push((step, q25, q50, q75));
    }
    let root = BitMapBackend::new(out_path, (1400, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("E4 A over time (median + IQR)", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(42)
        .y_label_area_size(56)
        .build_cartesian_2d(0f32..max_step as f32, 0f32..1f32)?;
    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc("A (alignment)")
        .draw()?;

    let mut keys: Vec<(String, i32)> = series.keys().cloned().collect();
    keys.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    for (i, key) in keys.iter().enumerate() {
        let mut pts = series.get(key).cloned().unwrap_or_default();
        pts.sort_by_key(|x| x.0);
        if pts.is_empty() {
            continue;
        }
        let color = Palette99::pick(i);
        let upper: Vec<(f32, f32)> = pts.iter().map(|(s, _, _, q75)| (*s as f32, *q75)).collect();
        let lower: Vec<(f32, f32)> = pts.iter().map(|(s, q25, _, _)| (*s as f32, *q25)).collect();
        let mut poly = upper.clone();
        for p in lower.iter().rev() {
            poly.push(*p);
        }
        chart.draw_series(std::iter::once(Polygon::new(
            poly,
            color.mix(0.20).filled(),
        )))?;
        chart
            .draw_series(LineSeries::new(
                pts.iter().map(|(s, _, q50, _)| (*s as f32, *q50)),
                color.stroke_width(2),
            ))?
            .label(format!("{} mw={:.2}", key.0, float_from_key(key.1)))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 18, y)], color.stroke_width(2))
            });
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK.mix(0.4))
        .draw()?;
    root.present()?;
    Ok(())
}

fn render_best_lag_hist(
    out_path: &Path,
    run_results: &[RunLagResult],
    max_lag: i32,
) -> Result<(), Box<dyn Error>> {
    if run_results.is_empty() {
        return Ok(());
    }
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for row in run_results {
        *counts.entry(row.best_lag).or_default() += 1;
    }
    let max_count = counts.values().copied().max().unwrap_or(1) as i32;
    let root = BitMapBackend::new(out_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("E4 best lag histogram", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(42)
        .y_label_area_size(52)
        .build_cartesian_2d(
            (-max_lag - 1) as f32..(max_lag + 1) as f32,
            0i32..(max_count + 1),
        )?;
    chart
        .configure_mesh()
        .x_desc("best_lag")
        .y_desc("run_count")
        .draw()?;
    chart.draw_series(((-max_lag)..=max_lag).map(|lag| {
        let c = counts.get(&lag).copied().unwrap_or(0) as i32;
        Rectangle::new(
            [(lag as f32 - 0.45, 0), (lag as f32 + 0.45, c)],
            BLUE.mix(0.65).filled(),
        )
    }))?;
    root.present()?;
    Ok(())
}

fn render_delta_a_by_lag_heatmap(
    out_path: &Path,
    run_results: &[RunLagResult],
    max_lag: i32,
) -> Result<(), Box<dyn Error>> {
    if run_results.is_empty() {
        return Ok(());
    }
    let mut mirror_keys: Vec<i32> = run_results
        .iter()
        .map(|r| float_key(r.mirror_weight))
        .collect();
    mirror_keys.sort();
    mirror_keys.dedup();
    let lag_values: Vec<i32> = ((-max_lag)..=max_lag).collect();

    let mut grouped: HashMap<(i32, i32), Vec<f32>> = HashMap::new();
    for run in run_results {
        let lag0 = run
            .lag_scores
            .iter()
            .find_map(|(lag, score)| if *lag == 0 { Some(*score) } else { None })
            .unwrap_or(0.0);
        let mirror_key = float_key(run.mirror_weight);
        for (lag, score) in &run.lag_scores {
            grouped
                .entry((mirror_key, *lag))
                .or_default()
                .push(*score - lag0);
        }
    }
    let mut value_map: HashMap<(i32, i32), f32> = HashMap::new();
    let mut max_abs = 1e-6f32;
    for mk in &mirror_keys {
        for lag in &lag_values {
            let vals = grouped.get(&(*mk, *lag)).cloned().unwrap_or_default();
            let mean = if vals.is_empty() {
                0.0
            } else {
                vals.iter().sum::<f32>() / vals.len() as f32
            };
            max_abs = max_abs.max(mean.abs());
            value_map.insert((*mk, *lag), mean);
        }
    }
    let root = BitMapBackend::new(out_path, (1400, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("E4 deltaA by lag heatmap", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(42)
        .y_label_area_size(70)
        .build_cartesian_2d((-max_lag)..(max_lag + 1), 0i32..(mirror_keys.len() as i32))?;
    chart
        .configure_mesh()
        .x_desc("lag")
        .y_desc("mirror_weight")
        .y_label_formatter(&|y| {
            let idx = (*y).clamp(0, mirror_keys.len().saturating_sub(1) as i32) as usize;
            format!("{:.2}", float_from_key(mirror_keys[idx]))
        })
        .draw()?;
    for (y, mk) in mirror_keys.iter().enumerate() {
        for lag in &lag_values {
            let v = value_map.get(&(*mk, *lag)).copied().unwrap_or(0.0);
            let t = if max_abs <= 1e-9 {
                0.5
            } else {
                (0.5 + 0.5 * (v / max_abs)).clamp(0.0, 1.0)
            };
            let color = HSLColor((240.0 - 240.0 * t as f64) / 360.0, 0.90, 0.45);
            chart.draw_series(std::iter::once(Rectangle::new(
                [(*lag, y as i32), (*lag + 1, y as i32 + 1)],
                color.filled(),
            )))?;
        }
    }
    root.present()?;
    Ok(())
}

fn build_overall_markdown(summary_rows: &[SummaryRow], run_results: &[RunLagResult]) -> String {
    if summary_rows.is_empty() {
        return "# E4 ABCD timing diagnosis\n\nNo data.\n".to_string();
    }
    let nonzero_majority = summary_rows
        .iter()
        .filter(|r| r.nonzero_best_lag_frac > 0.5)
        .count();
    let mut signs: Vec<i32> = summary_rows
        .iter()
        .map(|r| r.dominant_lag_sign)
        .filter(|s| *s != 0)
        .collect();
    signs.sort();
    signs.dedup();
    let sign_consistent = signs.len() <= 1;
    let mean_delta_a =
        summary_rows.iter().map(|r| r.delta_a_mean).sum::<f32>() / summary_rows.len() as f32;
    let promising =
        nonzero_majority * 2 > summary_rows.len() && sign_consistent && mean_delta_a > 0.05;

    let mut out = String::new();
    out.push_str("# E4 ABCD timing diagnosis\n\n");
    out.push_str("## Data scope\n");
    out.push_str(&format!("- runs: {}\n", run_results.len()));
    out.push_str(&format!("- mirror conditions: {}\n", summary_rows.len()));
    out.push_str("- burn-in: first 25% steps per run\n");
    out.push_str("- lag search range: ±10 steps (default)\n\n");

    out.push_str("## Summary by mirror\n");
    out.push_str("| timing_mode | mirror_weight | n_runs | best_lag_mode | nonzero_best_lag_frac | deltaA_mean | deltaA_CI95 |\n");
    out.push_str("| --- | ---: | ---: | ---: | ---: | ---: | --- |\n");
    for row in summary_rows {
        out.push_str(&format!(
            "| {} | {:.2} | {} | {} | {:.3} | {:.4} | [{:.4}, {:.4}] |\n",
            row.timing_mode,
            row.mirror_weight,
            row.n_runs,
            row.best_lag_mode,
            row.nonzero_best_lag_frac,
            row.delta_a_mean,
            row.delta_a_ci_lo,
            row.delta_a_ci_hi
        ));
    }
    out.push('\n');

    if promising {
        out.push_str("## Conclusion\n");
        out.push_str("判定: **タイミング修正が効く見込みが高い**。\n\n");
        let sign = summary_rows
            .iter()
            .map(|r| r.dominant_lag_sign)
            .find(|s| *s != 0)
            .unwrap_or(0);
        if sign > 0 {
            out.push_str("- best_lag が正方向優勢のため、agent が oracle より先行。oracle参照を1tick進める修正が候補。\n");
            out.push_str("- 実装案: `agent(step)` と比較する oracle を `step+1` 側に合わせる。\n");
        } else if sign < 0 {
            out.push_str("- best_lag が負方向優勢のため、oracle が先行。agent更新前oracleを参照する修正が候補。\n");
            out.push_str(
                "- 実装案: dispatch/update順を見直し、同一フレームの整合対象を統一する。\n",
            );
        } else {
            out.push_str("- 非ゼロlag優勢だが符号は混在。参照位置の一貫化を優先。\n");
        }
    } else {
        out.push_str("## Conclusion\n");
        out.push_str("判定: **タイミング修正のみで改善する見込みは低い**。\n\n");
        out.push_str("優先して疑う項目（高→低）:\n");
        out.push_str(
            "1. oracle と agent の参照状態不一致（env_scan/density、候補集合、clamp/step幅）。\n",
        );
        out.push_str(
            "2. oracle が到達不可能な最適を提示（1-step reachable oracle への変更が必要）。\n",
        );
        out.push_str("3. peak_sampler の候補母集団が oracle/agent で不一致。\n");
    }
    out
}

fn run(cli: &Cli) -> Result<(), Box<dyn Error>> {
    let input_text = read_to_string(&cli.input)?;
    let rows = parse_trace_csv(&input_text).map_err(std::io::Error::other)?;
    let runs = group_runs(&rows);
    let mut run_results = Vec::new();
    for run in &runs {
        if let Some(res) = analyze_run(run, cli.max_lag) {
            run_results.push(res);
        }
    }
    let summary_rows = summarize_by_mirror(&run_results);

    create_dir_all(&cli.outdir)?;
    write(
        cli.outdir.join("e4_abcd_summary_by_mirror.csv"),
        summary_by_mirror_csv(&summary_rows),
    )?;
    write(
        cli.outdir.join("e4_abcd_summary_overall.md"),
        build_overall_markdown(&summary_rows, &run_results),
    )?;

    render_a_over_time(&cli.outdir.join("e4_A_over_time.png"), &rows)?;
    render_best_lag_hist(
        &cli.outdir.join("e4_best_lag_hist.png"),
        &run_results,
        cli.max_lag,
    )?;
    render_delta_a_by_lag_heatmap(
        &cli.outdir.join("e4_deltaA_by_lag_heatmap.png"),
        &run_results,
        cli.max_lag,
    )?;
    Ok(())
}

pub fn run_from_args(args: &[String]) -> Result<(), Box<dyn Error>> {
    let cli = parse_args(args).map_err(std::io::Error::other)?;
    run(&cli)
}
