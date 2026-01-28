use std::error::Error;
use std::f32::consts::PI;
use std::fs::{create_dir_all, write};
use std::path::Path;

use plotters::prelude::*;

use conchordal::core::erb::hz_to_erb;
use conchordal::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
use conchordal::core::landscape::{LandscapeParams, RoughnessScalarMode};
use conchordal::core::log2space::Log2Space;
use conchordal::core::psycho_state;
use conchordal::core::roughness_kernel::{KernelParams, RoughnessKernel};
use conchordal::paper::sim::{E4_ANCHOR_HZ, E4_WINDOW_CENTS, interval_metrics, run_e4_condition};
use rand::{Rng, SeedableRng, rngs::StdRng};

const SPACE_BINS_PER_OCT: u32 = 400;

const E2_STEPS: usize = 400;
const E2_BURN_IN: usize = 100;
const E2_ANCHOR_SHIFT_STEP: usize = 200;
const E2_ANCHOR_SHIFT_RATIO: f32 = 0.5;
const E2_STEP_SEMITONES: f32 = 0.5;
const E2_ANCHOR_BIN_ST: f32 = 0.5;
const E2_PAIRWISE_BIN_ST: f32 = 0.25;

const E4_INTERVAL_BIN_ST: f32 = 0.5;

const E3_STEP_SEMITONES: f32 = 0.5;

const E5_GROUP_A_OMEGA: f32 = 2.0 * PI * 2.0;
const E5_GROUP_B_OMEGA: f32 = 2.0 * PI * 1.8;
const E5_JITTER: f32 = 0.02;
const E5_K_IN: f32 = 1.2;
const E5_K_CROSS: f32 = 1.6;
const E5_GROUP_N: usize = 16;
const E5_DT: f32 = 0.02;
const E5_STEPS: usize = 2000;
const E5_BURN_IN_STEPS: usize = 500;
const E5_SAMPLE_WINDOW_STEPS: usize = 250;
const E5_TIME_PLV_WINDOW_STEPS: usize = 200;
const E5_SEED: u64 = 0xC0FFEE_u64 + 2;

fn main() -> Result<(), Box<dyn Error>> {
    let out_dir = Path::new("target/plots/paper");
    create_dir_all(out_dir)?;

    let anchor_hz = E4_ANCHOR_HZ;
    let space = Log2Space::new(20.0, 8000.0, SPACE_BINS_PER_OCT);

    plot_e1_landscape_scan(out_dir, &space, anchor_hz)?;
    plot_e2_emergent_harmony(out_dir, &space, anchor_hz)?;
    plot_e3_metabolic_selection(out_dir, &space, anchor_hz)?;
    plot_e4_mirror_sweep(out_dir, anchor_hz)?;
    plot_e5_rhythmic_entrainment(out_dir)?;

    println!("Saved paper plots to {}", out_dir.display());
    Ok(())
}

fn plot_e1_landscape_scan(
    out_dir: &Path,
    space: &Log2Space,
    anchor_hz: f32,
) -> Result<(), Box<dyn Error>> {
    let anchor_idx = nearest_bin(space, anchor_hz);

    let (_erb_scan, du_scan) = erb_grid_for_space(space);
    let mut anchor_density_scan = vec![0.0f32; space.n_bins()];
    let denom = du_scan[anchor_idx].max(1e-12);
    anchor_density_scan[anchor_idx] = 1.0 / denom;

    let mut anchor_env_scan = vec![0.0f32; space.n_bins()];
    anchor_env_scan[anchor_idx] = 1.0;

    space.assert_scan_len_named(&anchor_density_scan, "anchor_density_scan");
    space.assert_scan_len_named(&anchor_env_scan, "anchor_env_scan");

    let roughness_kernel = RoughnessKernel::new(KernelParams::default(), 0.005);
    let harmonicity_kernel = HarmonicityKernel::new(space, HarmonicityParams::default());

    let (perc_h_pot_scan, _) =
        harmonicity_kernel.potential_h_from_log2_spectrum(&anchor_env_scan, space);
    let (perc_r_pot_scan, _) =
        roughness_kernel.potential_r_from_log2_spectrum_density(&anchor_density_scan, space);

    space.assert_scan_len_named(&perc_h_pot_scan, "perc_h_pot_scan");
    space.assert_scan_len_named(&perc_r_pot_scan, "perc_r_pot_scan");

    let params = LandscapeParams {
        fs: 48_000.0,
        max_hist_cols: 1,
        alpha: 0.0,
        roughness_kernel: roughness_kernel.clone(),
        harmonicity_kernel: harmonicity_kernel.clone(),
        roughness_scalar_mode: RoughnessScalarMode::Total,
        roughness_half: 0.1,
        consonance_harmonicity_deficit_weight: 1.0,
        consonance_roughness_weight_floor: 0.35,
        consonance_roughness_weight: 0.5,
        loudness_exp: 1.0,
        ref_power: 1.0,
        tau_ms: 1.0,
        roughness_k: 1.0,
        roughness_ref_f0_hz: 1000.0,
        roughness_ref_sep_erb: 0.25,
        roughness_ref_mass_split: 0.5,
        roughness_ref_eps: 1e-12,
    };

    let r_ref = psycho_state::compute_roughness_reference(&params, space);
    let mut perc_r_state01_scan = vec![0.0f32; space.n_bins()];
    psycho_state::r_pot_scan_to_r_state01_scan(
        &perc_r_pot_scan,
        r_ref.peak,
        params.roughness_k,
        &mut perc_r_state01_scan,
    );

    let h_ref_max = perc_h_pot_scan
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .max(1e-12);
    let mut perc_h_state01_scan = vec![0.0f32; space.n_bins()];
    psycho_state::h_pot_scan_to_h_state01_scan(
        &perc_h_pot_scan,
        h_ref_max,
        &mut perc_h_state01_scan,
    );

    let mut perc_c_raw_scan = vec![0.0f32; space.n_bins()];
    for i in 0..space.n_bins() {
        let h01 = perc_h_state01_scan[i];
        let r01 = perc_r_state01_scan[i];
        let dh = 1.0 - h01;
        let w = params.consonance_roughness_weight_floor + params.consonance_roughness_weight * dh;
        let d = params.consonance_harmonicity_deficit_weight * dh + w * r01;
        let c_raw = 1.0 / (1.0 + d);
        perc_c_raw_scan[i] = if c_raw.is_finite() { c_raw } else { 0.0 };
    }

    let anchor_log2 = anchor_hz.log2();
    let log2_ratio_scan: Vec<f32> = space
        .centers_log2
        .iter()
        .map(|&l| l - anchor_log2)
        .collect();

    space.assert_scan_len_named(&perc_r_state01_scan, "perc_r_state01_scan");
    space.assert_scan_len_named(&perc_h_state01_scan, "perc_h_state01_scan");
    space.assert_scan_len_named(&perc_c_raw_scan, "perc_c_raw_scan");
    space.assert_scan_len_named(&log2_ratio_scan, "log2_ratio_scan");

    let out_path = out_dir.join("paper_e1_landscape_scan_anchor220.png");
    render_e1_plot(
        &out_path,
        anchor_hz,
        &log2_ratio_scan,
        &perc_h_pot_scan,
        &perc_r_state01_scan,
        &perc_c_raw_scan,
    )?;

    Ok(())
}

fn plot_e2_emergent_harmony(
    out_dir: &Path,
    space: &Log2Space,
    anchor_hz: f32,
) -> Result<(), Box<dyn Error>> {
    let mut rng = seeded_rng(0xC0FFEE);
    let mut anchor_hz_current = anchor_hz;
    let mut anchor_idx = nearest_bin(space, anchor_hz_current);
    let mut log2_ratio_scan = build_log2_ratio_scan(space, anchor_hz_current);
    let (mut min_idx, mut max_idx) = log2_ratio_bounds(&log2_ratio_scan, -1.0, 1.0);
    let allowed_indices: Vec<usize> = (min_idx..=max_idx).collect();
    let n_agents = 24;

    let mut agent_indices: Vec<usize> = (0..n_agents)
        .map(|_| {
            let pick = rng.random_range(0..allowed_indices.len());
            allowed_indices[pick]
        })
        .collect();

    let (_erb_scan, du_scan) = erb_grid_for_space(space);
    let workspace = build_c01_workspace(space);

    let steps = E2_STEPS;
    let burn_in = E2_BURN_IN;
    let mut semitone_samples: Vec<f32> = Vec::new();
    let mut mean_c01_series: Vec<(f32, f32)> = Vec::with_capacity(steps);
    let mut mean_score_series: Vec<(f32, f32)> = Vec::with_capacity(steps);
    let mut mean_repulsion_series: Vec<(f32, f32)> = Vec::with_capacity(steps);
    let mut trajectory_csv = String::from("step,agent_id,semitones,c01\n");
    let mut trajectories: Vec<Vec<f32>> = vec![Vec::with_capacity(steps); n_agents];
    let mut anchor_shift_csv = String::from("step,count_min_idx,count_max_idx,respawned\n");
    let k_bins = k_from_semitones(E2_STEP_SEMITONES);

    for step in 0..steps {
        if step == E2_ANCHOR_SHIFT_STEP {
            anchor_hz_current = anchor_hz_current * E2_ANCHOR_SHIFT_RATIO;
            anchor_idx = nearest_bin(space, anchor_hz_current);
            log2_ratio_scan = build_log2_ratio_scan(space, anchor_hz_current);
            let bounds = log2_ratio_bounds(&log2_ratio_scan, -1.0, 1.0);
            min_idx = bounds.0;
            max_idx = bounds.1;
            let (count_min, count_max, respawned) = shift_indices_by_ratio(
                space,
                &mut agent_indices,
                E2_ANCHOR_SHIFT_RATIO,
                min_idx,
                max_idx,
                &mut rng,
            );
            anchor_shift_csv.push_str(&format!("{step},{count_min},{count_max},{respawned}\n"));
            semitone_samples.clear();
        }

        let (env_scan, density_scan) = build_env_scans(space, anchor_idx, &agent_indices, &du_scan);
        let c01_scan = compute_c01_scan(space, &workspace, &env_scan, &density_scan);
        let mean_c01 = mean_at_indices(&c01_scan, &agent_indices);
        mean_c01_series.push((step as f32, mean_c01));

        for (agent_id, &idx) in agent_indices.iter().enumerate() {
            let semitone = 12.0 * log2_ratio_scan[idx];
            let c01 = c01_scan[idx];
            trajectories[agent_id].push(semitone);
            trajectory_csv.push_str(&format!("{step},{agent_id},{semitone:.6},{c01:.6}\n"));
        }

        let stats = update_agent_indices_scored_stats(
            &mut agent_indices,
            &c01_scan,
            &log2_ratio_scan,
            min_idx,
            max_idx,
            k_bins,
            0.15,
            0.06,
        );
        mean_score_series.push((step as f32, stats.mean_score));
        mean_repulsion_series.push((step as f32, stats.mean_repulsion));

        if step >= burn_in {
            semitone_samples.extend(agent_indices.iter().map(|&idx| 12.0 * log2_ratio_scan[idx]));
        }
    }

    let mut csv_ts = String::from("t,mean_c01\n");
    for (t, mean) in &mean_c01_series {
        csv_ts.push_str(&format!("{t:.0},{mean:.6}\n"));
    }
    write(out_dir.join("paper_e2_timeseries.csv"), csv_ts)?;
    write(
        out_dir.join("paper_e2_agent_trajectories.csv"),
        trajectory_csv,
    )?;
    write(
        out_dir.join("paper_e2_anchor_shift_stats.csv"),
        anchor_shift_csv,
    )?;

    let mut csv_agents = String::from("agent_id,freq_hz,log2_ratio,semitones\n");
    for (id, &idx) in agent_indices.iter().enumerate() {
        let freq_hz = space.centers_hz[idx];
        let log2_ratio = log2_ratio_scan[idx];
        let semitones = 12.0 * log2_ratio;
        csv_agents.push_str(&format!(
            "{id},{freq_hz:.6},{log2_ratio:.6},{semitones:.3}\n"
        ));
    }
    write(out_dir.join("paper_e2_final_agents.csv"), csv_agents)?;

    let mean_plot_path = out_dir.join("paper_e2_mean_consonance_over_time.png");
    render_series_plot(
        &mean_plot_path,
        "E2 Mean Consonance Over Time",
        "mean C01",
        &mean_c01_series,
    )?;

    let mut csv_score = String::from("t,mean_score\n");
    for (t, mean) in &mean_score_series {
        csv_score.push_str(&format!("{t:.0},{mean:.6}\n"));
    }
    write(out_dir.join("paper_e2_score_timeseries.csv"), csv_score)?;

    let mean_score_path = out_dir.join("paper_e2_mean_score_over_time.png");
    render_series_plot(
        &mean_score_path,
        "E2 Mean Score Over Time",
        "mean score (c01 - λ·repulsion)",
        &mean_score_series,
    )?;

    let mut csv_repulsion = String::from("t,mean_repulsion\n");
    for (t, mean) in &mean_repulsion_series {
        csv_repulsion.push_str(&format!("{t:.0},{mean:.6}\n"));
    }
    write(
        out_dir.join("paper_e2_repulsion_timeseries.csv"),
        csv_repulsion,
    )?;

    let mean_repulsion_path = out_dir.join("paper_e2_mean_repulsion_over_time.png");
    render_series_plot(
        &mean_repulsion_path,
        "E2 Mean Repulsion Over Time",
        "mean repulsion",
        &mean_repulsion_series,
    )?;

    let trajectory_path = out_dir.join("paper_e2_agent_trajectories.png");
    render_agent_trajectories_plot(&trajectory_path, &trajectories)?;

    let hist_path = out_dir.join("paper_e2_interval_histogram.png");
    render_interval_histogram(
        &hist_path,
        "E2 Interval Histogram (Semitones)",
        &semitone_samples,
        -12.0,
        12.0,
        E2_ANCHOR_BIN_ST,
    )?;

    let final_semitones: Vec<f32> = agent_indices
        .iter()
        .map(|&idx| 12.0 * log2_ratio_scan[idx])
        .collect();
    let pairwise_intervals = pairwise_interval_samples(&final_semitones);
    let mut csv_pairwise = String::from("interval_semitones\n");
    for interval in &pairwise_intervals {
        csv_pairwise.push_str(&format!("{interval:.6}\n"));
    }
    write(
        out_dir.join("paper_e2_pairwise_intervals.csv"),
        csv_pairwise,
    )?;

    let pairwise_hist_path = out_dir.join("paper_e2_pairwise_interval_histogram.png");
    render_interval_histogram(
        &pairwise_hist_path,
        "E2 Pairwise Interval Histogram (Semitones, 12=octave)",
        &pairwise_intervals,
        0.0,
        12.0,
        E2_PAIRWISE_BIN_ST,
    )?;

    Ok(())
}

fn plot_e3_metabolic_selection(
    out_dir: &Path,
    space: &Log2Space,
    anchor_hz: f32,
) -> Result<(), Box<dyn Error>> {
    let mut rng = seeded_rng(0xC0FFEE_u64 + 1);
    let anchor_idx = nearest_bin(space, anchor_hz);
    let log2_ratio_scan = build_log2_ratio_scan(space, anchor_hz);
    let (min_idx, max_idx) = log2_ratio_bounds(&log2_ratio_scan, -1.0, 1.0);
    let allowed_indices: Vec<usize> = (min_idx..=max_idx).collect();

    let n_agents = 32;
    let mut agents: Vec<MetabolicAgent> = (0..n_agents)
        .map(|_| {
            let pick = rng.random_range(0..allowed_indices.len());
            MetabolicAgent {
                idx: allowed_indices[pick],
                energy: 1.0,
                age_steps: 0,
                sum_c01: 0.0,
            }
        })
        .collect();

    let (_erb_scan, du_scan) = erb_grid_for_space(space);
    let workspace = build_c01_workspace(space);

    let gain = 0.06f32;
    let decay = 0.04f32;
    let mut deaths: Vec<(usize, u32, f32)> = Vec::new();
    let mut death_id: usize = 0;
    let k_bins = k_from_semitones(E3_STEP_SEMITONES);

    let min_deaths = 200usize;
    let base_steps = 2000usize;
    let hard_cap = 6000usize;
    let mut step = 0usize;

    while step < hard_cap && (step < base_steps || deaths.len() < min_deaths) {
        let agent_indices: Vec<usize> = agents.iter().map(|a| a.idx).collect();
        let (env_scan, density_scan) = build_env_scans(space, anchor_idx, &agent_indices, &du_scan);
        let c01_scan = compute_c01_scan(space, &workspace, &env_scan, &density_scan);

        for agent in agents.iter_mut() {
            let c = c01_scan[agent.idx];
            agent.energy += gain * c - decay;
            agent.age_steps += 1;
            agent.sum_c01 += c;

            if agent.energy <= 0.0 {
                let avg_c01 = if agent.age_steps > 0 {
                    agent.sum_c01 / agent.age_steps as f32
                } else {
                    0.0
                };
                deaths.push((death_id, agent.age_steps, avg_c01));
                death_id += 1;

                let pick = rng.random_range(0..allowed_indices.len());
                agent.idx = allowed_indices[pick];
                agent.energy = 1.0;
                agent.age_steps = 0;
                agent.sum_c01 = 0.0;
            }
        }

        let mut updated_indices: Vec<usize> = agents.iter().map(|a| a.idx).collect();
        update_agent_indices(
            &mut updated_indices,
            &c01_scan,
            &log2_ratio_scan,
            min_idx,
            max_idx,
            k_bins,
            0.15,
            0.06,
        );
        for (agent, idx) in agents.iter_mut().zip(updated_indices.into_iter()) {
            agent.idx = idx;
        }

        step += 1;
    }

    let mut csv = String::from("death_id,lifetime_steps,avg_c01\n");
    for (id, lifetime, avg_c01) in &deaths {
        csv.push_str(&format!("{id},{lifetime},{avg_c01:.6}\n"));
    }
    write(out_dir.join("paper_e3_lifetimes.csv"), csv)?;

    let scatter_path = out_dir.join("paper_e3_consonance_vs_lifetime.png");
    render_consonance_lifetime_scatter(&scatter_path, &deaths)?;

    let survival_path = out_dir.join("paper_e3_survival_curve.png");
    render_survival_curve(&survival_path, &deaths)?;

    let survival_by_c01_path = out_dir.join("paper_e3_survival_by_c01.png");
    render_survival_by_c01(&survival_by_c01_path, &deaths)?;

    Ok(())
}

fn plot_e5_rhythmic_entrainment(out_dir: &Path) -> Result<(), Box<dyn Error>> {
    let sim = simulate_e5(E5_SEED, E5_STEPS);

    let mut csv = String::from("t,r_a,r_b,delta_phi,plv_ab_paired\n");
    let mut delta_phi_min = f32::INFINITY;
    let mut delta_phi_max = f32::NEG_INFINITY;
    for (t, r_a, r_b, delta_phi, plv_ab) in &sim.series {
        delta_phi_min = delta_phi_min.min(*delta_phi);
        delta_phi_max = delta_phi_max.max(*delta_phi);
        csv.push_str(&format!(
            "{t:.4},{r_a:.6},{r_b:.6},{delta_phi:.6},{plv_ab:.6}\n"
        ));
    }
    if !delta_phi_min.is_finite() || !delta_phi_max.is_finite() {
        delta_phi_min = 0.0;
        delta_phi_max = 0.0;
    }
    write(out_dir.join("paper_e5_order_offset.csv"), csv)?;

    let order_path = out_dir.join("paper_e5_order_over_time.png");
    render_order_offset_plot(&order_path, &sim.series)?;

    let delta_path = out_dir.join("paper_e5_delta_phi_over_time.png");
    render_delta_phi_plot(&delta_path, &sim.series)?;

    let plv_path = out_dir.join("paper_e5_plv_ab_over_time.png");
    render_plv_ab_plot(&plv_path, &sim.series)?;

    let bins = phase_hist_bins(sim.phase_hist_samples.len());
    let phase_path = out_dir.join("paper_e5_phase_diff_histogram.png");
    render_phase_histogram(&phase_path, &sim.phase_hist_samples, bins)?;

    let mut summary = String::from("delta_phi_min,delta_phi_max,plv_time\n");
    summary.push_str(&format!(
        "{delta_phi_min:.6},{delta_phi_max:.6},{:.6}\n",
        sim.plv_time
    ));
    write(out_dir.join("paper_e5_order_offset_summary.csv"), summary)?;

    Ok(())
}

#[allow(dead_code)]
struct E5SimResult {
    series: Vec<(f32, f32, f32, f32, f32)>,
    phase_hist_samples: Vec<f32>,
    thetas_last_a: Vec<f32>,
    thetas_last_b: Vec<f32>,
    final_t: f32,
    sample_window_steps: usize,
    n_a: usize,
    plv_time: f32,
}

fn simulate_e5(seed: u64, steps: usize) -> E5SimResult {
    let mut rng = seeded_rng(seed);
    let mut thetas_a: Vec<f32> = (0..E5_GROUP_N)
        .map(|_| rng.random_range(0.0f32..(2.0 * PI)))
        .collect();
    let mut thetas_b: Vec<f32> = (0..E5_GROUP_N)
        .map(|_| rng.random_range(0.0f32..(2.0 * PI)))
        .collect();
    let omegas_a: Vec<f32> = (0..E5_GROUP_N)
        .map(|_| {
            let jitter_scale = rng.random_range(-E5_JITTER..E5_JITTER);
            E5_GROUP_A_OMEGA * (1.0 + jitter_scale)
        })
        .collect();
    let omegas_b: Vec<f32> = (0..E5_GROUP_N)
        .map(|_| {
            let jitter_scale = rng.random_range(-E5_JITTER..E5_JITTER);
            E5_GROUP_B_OMEGA * (1.0 + jitter_scale)
        })
        .collect();

    let mut series: Vec<(f32, f32, f32, f32, f32)> = Vec::with_capacity(steps);
    let mut thetas_last_a: Vec<f32> = Vec::new();
    let mut thetas_last_b: Vec<f32> = Vec::new();
    let mut phase_hist_samples: Vec<f32> = Vec::new();
    let sample_start = steps
        .saturating_sub(E5_SAMPLE_WINDOW_STEPS)
        .max(E5_BURN_IN_STEPS);

    for step in 0..steps {
        let t = step as f32 * E5_DT;
        let (mut mean_cos_a, mut mean_sin_a) = (0.0f32, 0.0f32);
        let (mut mean_cos_b, mut mean_sin_b) = (0.0f32, 0.0f32);
        for &theta in &thetas_a {
            mean_cos_a += theta.cos();
            mean_sin_a += theta.sin();
        }
        for &theta in &thetas_b {
            mean_cos_b += theta.cos();
            mean_sin_b += theta.sin();
        }
        let inv_a = 1.0 / E5_GROUP_N as f32;
        let inv_b = 1.0 / E5_GROUP_N as f32;
        mean_cos_a *= inv_a;
        mean_sin_a *= inv_a;
        mean_cos_b *= inv_b;
        mean_sin_b *= inv_b;

        let r_a = (mean_cos_a * mean_cos_a + mean_sin_a * mean_sin_a).sqrt();
        let r_b = (mean_cos_b * mean_cos_b + mean_sin_b * mean_sin_b).sqrt();
        let phi_a = mean_sin_a.atan2(mean_cos_a);
        let phi_b = mean_sin_b.atan2(mean_cos_b);
        let delta_phi = wrap_to_pi(phi_a - phi_b);

        let pairs = E5_GROUP_N.min(E5_GROUP_N);
        let mut pair_cos = 0.0f32;
        let mut pair_sin = 0.0f32;
        for i in 0..pairs {
            let d = wrap_to_pi(thetas_a[i] - thetas_b[i]);
            pair_cos += d.cos();
            pair_sin += d.sin();
        }
        let inv_pairs = 1.0 / pairs as f32;
        pair_cos *= inv_pairs;
        pair_sin *= inv_pairs;
        let plv_ab = (pair_cos * pair_cos + pair_sin * pair_sin).sqrt();

        series.push((t, r_a, r_b, delta_phi, plv_ab));

        if step + 1 == steps {
            thetas_last_a = thetas_a.clone();
            thetas_last_b = thetas_b.clone();
        }

        if step >= sample_start {
            for i in 0..E5_GROUP_N {
                phase_hist_samples.push(wrap_to_pi(thetas_a[i] - thetas_b[i]));
            }
        }

        let mut next_a = vec![0.0f32; E5_GROUP_N];
        let mut next_b = vec![0.0f32; E5_GROUP_N];
        for i in 0..E5_GROUP_N {
            let theta_i = thetas_a[i];
            let mut coupling_in = 0.0f32;
            let mut coupling_cross = 0.0f32;
            for &theta_j in &thetas_a {
                coupling_in += (theta_j - theta_i).sin();
            }
            for &theta_j in &thetas_b {
                coupling_cross += (theta_j - theta_i).sin();
            }
            let dtheta = omegas_a[i]
                + (E5_K_IN * inv_a) * coupling_in
                + (E5_K_CROSS * inv_b) * coupling_cross;
            next_a[i] = theta_i + dtheta * E5_DT;
        }
        for i in 0..E5_GROUP_N {
            let theta_i = thetas_b[i];
            let mut coupling_in = 0.0f32;
            let mut coupling_cross = 0.0f32;
            for &theta_j in &thetas_b {
                coupling_in += (theta_j - theta_i).sin();
            }
            for &theta_j in &thetas_a {
                coupling_cross += (theta_j - theta_i).sin();
            }
            let dtheta = omegas_b[i]
                + (E5_K_IN * inv_b) * coupling_in
                + (E5_K_CROSS * inv_a) * coupling_cross;
            next_b[i] = theta_i + dtheta * E5_DT;
        }
        thetas_a = next_a;
        thetas_b = next_b;
    }

    let final_t = series.last().map(|(t, _, _, _, _)| *t).unwrap_or(0.0);
    let plv_time = plv_time_from_series(&series, E5_TIME_PLV_WINDOW_STEPS);
    E5SimResult {
        series,
        phase_hist_samples,
        thetas_last_a,
        thetas_last_b,
        final_t,
        sample_window_steps: steps.saturating_sub(sample_start),
        n_a: E5_GROUP_N,
        plv_time,
    }
}

fn render_e1_plot(
    out_path: &Path,
    anchor_hz: f32,
    log2_ratio_scan: &[f32],
    perc_h_pot_scan: &[f32],
    perc_r_state01_scan: &[f32],
    perc_c_raw_scan: &[f32],
) -> Result<(), Box<dyn Error>> {
    let x_min = -1.0f32;
    let x_max = 1.0f32;

    let mut h_points = Vec::new();
    let mut r_points = Vec::new();
    let mut c_points = Vec::new();
    for i in 0..log2_ratio_scan.len() {
        let x = log2_ratio_scan[i];
        if x < x_min || x > x_max {
            continue;
        }
        h_points.push((x, perc_h_pot_scan[i]));
        r_points.push((x, perc_r_state01_scan[i]));
        c_points.push((x, perc_c_raw_scan[i]));
    }

    let h_max = h_points
        .iter()
        .map(|(_, y)| *y)
        .fold(0.0f32, f32::max)
        .max(1e-6)
        * 1.1;

    let root = BitMapBackend::new(out_path, (1600, 1200)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((3, 1));

    let ratio_guides = [0.5f32, 6.0 / 5.0, 1.25, 4.0 / 3.0, 1.5, 5.0 / 3.0, 2.0];
    let ratio_guides_log2: Vec<f32> = ratio_guides.iter().map(|r| r.log2()).collect();

    let mut chart_h = ChartBuilder::on(&panels[0])
        .caption(
            format!("E1 Harmonicity Potential H(f) | anchor {} Hz", anchor_hz),
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, 0.0f32..h_max)?;

    chart_h
        .configure_mesh()
        .x_desc("log2(f / f_anchor)")
        .y_desc("H potential")
        .draw()?;

    for &x in &ratio_guides_log2 {
        if x >= x_min && x <= x_max {
            chart_h.draw_series(std::iter::once(PathElement::new(
                vec![(x, 0.0), (x, h_max)],
                BLACK.mix(0.15),
            )))?;
        }
    }

    chart_h.draw_series(LineSeries::new(h_points, &BLUE))?;

    let mut chart_r = ChartBuilder::on(&panels[1])
        .caption("E1 Roughness State R01(f)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, 0.0f32..1.05f32)?;

    chart_r
        .configure_mesh()
        .x_desc("log2(f / f_anchor)")
        .y_desc("R01")
        .draw()?;

    for &x in &ratio_guides_log2 {
        if x >= x_min && x <= x_max {
            chart_r.draw_series(std::iter::once(PathElement::new(
                vec![(x, 0.0), (x, 1.05)],
                BLACK.mix(0.15),
            )))?;
        }
    }

    chart_r.draw_series(LineSeries::new(r_points, &RED))?;

    let mut c_min = f32::INFINITY;
    let mut c_max = f32::NEG_INFINITY;
    for &(_, y) in &c_points {
        if y.is_finite() {
            c_min = c_min.min(y);
            c_max = c_max.max(y);
        }
    }
    if !c_min.is_finite() || !c_max.is_finite() || (c_max - c_min).abs() < 1e-6 {
        c_min = -1.0;
        c_max = 1.0;
    }
    let pad = 0.05f32;

    let mut chart_c = ChartBuilder::on(&panels[2])
        .caption("E1 Consonance Raw C(f)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, (c_min - pad)..(c_max + pad))?;

    chart_c
        .configure_mesh()
        .x_desc("log2(f / f_anchor)")
        .y_desc("C_raw")
        .draw()?;

    for &x in &ratio_guides_log2 {
        if x >= x_min && x <= x_max {
            chart_c.draw_series(std::iter::once(PathElement::new(
                vec![(x, c_min - pad), (x, c_max + pad)],
                BLACK.mix(0.15),
            )))?;
        }
    }

    chart_c.draw_series(LineSeries::new(c_points, &BLACK))?;

    root.present()?;
    Ok(())
}

fn plot_e4_mirror_sweep(out_dir: &Path, anchor_hz: f32) -> Result<(), Box<dyn Error>> {
    let seed = 7;
    let mut rows: Vec<(f32, f32, f32, f32, usize)> = Vec::new();
    for i in 0..=10 {
        let mirror_weight = i as f32 / 10.0;
        let voice_freqs = run_e4_condition(mirror_weight, seed);
        let metrics = interval_metrics(anchor_hz, &voice_freqs, E4_WINDOW_CENTS);
        rows.push((
            mirror_weight,
            metrics.mass_maj3,
            metrics.mass_min3,
            metrics.mass_p5,
            metrics.n_voices,
        ));
    }

    let mut csv = String::from("mirror_weight,mass_M3,mass_m3,mass_P5,n_voices,window_cents\n");
    for (w, mass_maj3, mass_min3, mass_p5, n_voices) in &rows {
        csv.push_str(&format!(
            "{w:.3},{mass_maj3:.3},{mass_min3:.3},{mass_p5:.3},{n_voices},{}\n",
            E4_WINDOW_CENTS
        ));
    }

    let csv_path = out_dir.join("paper_e4_mirror_sweep_metrics.csv");
    write(&csv_path, csv)?;

    let mut y_max = rows
        .iter()
        .map(|(_, m3, m3min, _p5, _)| (*m3).max(*m3min))
        .fold(0.0f32, f32::max);
    if y_max <= 0.0 {
        y_max = 1.0;
    }
    let y_hi = y_max * 1.1;

    let plot_path = out_dir.join("paper_e4_mirror_sweep.png");
    let root = BitMapBackend::new(&plot_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("E4 Mirror Weight Sweep (Interval Mass)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, 0.0f32..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("mirror_weight")
        .y_desc("interval mass (count)")
        .draw()?;

    let maj_points: Vec<(f32, f32)> = rows.iter().map(|(w, m3, _, _, _)| (*w, *m3)).collect();
    let min_points: Vec<(f32, f32)> = rows.iter().map(|(w, _, m3, _, _)| (*w, *m3)).collect();

    chart
        .draw_series(LineSeries::new(maj_points, &BLUE))?
        .label("mass_M3")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart
        .draw_series(LineSeries::new(min_points, &RED))?
        .label("mass_m3")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;

    let freqs_m0 = run_e4_condition(0.0, seed);
    let freqs_m1 = run_e4_condition(1.0, seed);
    let semitones_m0: Vec<f32> = freqs_m0
        .iter()
        .filter(|f| f.is_finite() && **f > 0.0)
        .map(|f| 12.0 * (f / anchor_hz).log2())
        .collect();
    let semitones_m1: Vec<f32> = freqs_m1
        .iter()
        .filter(|f| f.is_finite() && **f > 0.0)
        .map(|f| 12.0 * (f / anchor_hz).log2())
        .collect();

    let hist_m0_path = out_dir.join("paper_e4_interval_histogram_m0.png");
    render_interval_histogram(
        &hist_m0_path,
        "E4 Interval Histogram (mirror_weight=0.0)",
        &semitones_m0,
        0.0,
        12.0,
        E4_INTERVAL_BIN_ST,
    )?;
    let hist_m1_path = out_dir.join("paper_e4_interval_histogram_m1.png");
    render_interval_histogram(
        &hist_m1_path,
        "E4 Interval Histogram (mirror_weight=1.0)",
        &semitones_m1,
        0.0,
        12.0,
        E4_INTERVAL_BIN_ST,
    )?;
    Ok(())
}

struct C01Workspace {
    params: LandscapeParams,
    r_ref_peak: f32,
}

#[derive(Clone, Copy)]
struct MetabolicAgent {
    idx: usize,
    energy: f32,
    age_steps: u32,
    sum_c01: f32,
}

fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn build_log2_ratio_scan(space: &Log2Space, anchor_hz: f32) -> Vec<f32> {
    let anchor_log2 = anchor_hz.log2();
    space
        .centers_log2
        .iter()
        .map(|&l| l - anchor_log2)
        .collect()
}

fn log2_ratio_bounds(log2_ratio_scan: &[f32], min: f32, max: f32) -> (usize, usize) {
    let mut min_idx: Option<usize> = None;
    let mut max_idx: Option<usize> = None;
    for (i, &v) in log2_ratio_scan.iter().enumerate() {
        if v >= min && v <= max {
            if min_idx.is_none() {
                min_idx = Some(i);
            }
            max_idx = Some(i);
        }
    }
    let fallback_min = 0;
    let fallback_max = log2_ratio_scan.len().saturating_sub(1);
    (
        min_idx.unwrap_or(fallback_min),
        max_idx.unwrap_or(fallback_max),
    )
}

fn build_env_scans(
    space: &Log2Space,
    anchor_idx: usize,
    agent_indices: &[usize],
    du_scan: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut env_scan = vec![0.0f32; space.n_bins()];
    let mut density_scan = vec![0.0f32; space.n_bins()];

    let mut add_source = |idx: usize| {
        env_scan[idx] += 1.0;
        let denom = du_scan[idx].max(1e-12);
        density_scan[idx] += 1.0 / denom;
    };

    add_source(anchor_idx);
    for &idx in agent_indices {
        add_source(idx);
    }

    (env_scan, density_scan)
}

fn build_c01_workspace(space: &Log2Space) -> C01Workspace {
    let roughness_kernel = RoughnessKernel::new(KernelParams::default(), 0.005);
    let harmonicity_kernel = HarmonicityKernel::new(space, HarmonicityParams::default());
    let params = LandscapeParams {
        fs: 48_000.0,
        max_hist_cols: 1,
        alpha: 0.0,
        roughness_kernel,
        harmonicity_kernel,
        roughness_scalar_mode: RoughnessScalarMode::Total,
        roughness_half: 0.1,
        consonance_harmonicity_deficit_weight: 1.0,
        consonance_roughness_weight_floor: 0.35,
        consonance_roughness_weight: 0.5,
        loudness_exp: 1.0,
        ref_power: 1.0,
        tau_ms: 1.0,
        roughness_k: 1.0,
        roughness_ref_f0_hz: 1000.0,
        roughness_ref_sep_erb: 0.25,
        roughness_ref_mass_split: 0.5,
        roughness_ref_eps: 1e-12,
    };
    let r_ref = psycho_state::compute_roughness_reference(&params, space);
    C01Workspace {
        params,
        r_ref_peak: r_ref.peak,
    }
}

fn compute_c01_scan(
    space: &Log2Space,
    workspace: &C01Workspace,
    env_scan: &[f32],
    density_scan: &[f32],
) -> Vec<f32> {
    let (perc_h_pot_scan, _) = workspace
        .params
        .harmonicity_kernel
        .potential_h_from_log2_spectrum(env_scan, space);
    let (perc_r_pot_scan, _) = workspace
        .params
        .roughness_kernel
        .potential_r_from_log2_spectrum_density(density_scan, space);

    let mut perc_r_state01_scan = vec![0.0f32; space.n_bins()];
    psycho_state::r_pot_scan_to_r_state01_scan(
        &perc_r_pot_scan,
        workspace.r_ref_peak,
        workspace.params.roughness_k,
        &mut perc_r_state01_scan,
    );

    let h_ref_max = perc_h_pot_scan
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .max(1e-12);
    let mut perc_h_state01_scan = vec![0.0f32; space.n_bins()];
    psycho_state::h_pot_scan_to_h_state01_scan(
        &perc_h_pot_scan,
        h_ref_max,
        &mut perc_h_state01_scan,
    );

    let mut c01_scan = vec![0.0f32; space.n_bins()];
    for i in 0..space.n_bins() {
        let (_c_signed, c01) = psycho_state::compose_c_statepm1(
            perc_h_state01_scan[i],
            perc_r_state01_scan[i],
            workspace.params.consonance_harmonicity_deficit_weight,
            workspace.params.consonance_roughness_weight_floor,
            workspace.params.consonance_roughness_weight,
        );
        c01_scan[i] = c01.clamp(0.0, 1.0);
    }
    c01_scan
}

fn update_agent_indices(
    indices: &mut [usize],
    c01_scan: &[f32],
    log2_ratio_scan: &[f32],
    min_idx: usize,
    max_idx: usize,
    k: i32,
    lambda: f32,
    sigma: f32,
) {
    let _ = update_agent_indices_scored(
        indices,
        c01_scan,
        log2_ratio_scan,
        min_idx,
        max_idx,
        k,
        lambda,
        sigma,
    );
}

struct UpdateStats {
    mean_score: f32,
    mean_repulsion: f32,
}

fn k_from_semitones(step_semitones: f32) -> i32 {
    let bins_per_semitone = SPACE_BINS_PER_OCT as f32 / 12.0;
    let k = (bins_per_semitone * step_semitones).round() as i32;
    k.max(1)
}

fn shift_indices_by_ratio(
    space: &Log2Space,
    indices: &mut [usize],
    ratio: f32,
    min_idx: usize,
    max_idx: usize,
    rng: &mut StdRng,
) -> (usize, usize, usize) {
    let mut count_min = 0usize;
    let mut count_max = 0usize;
    let mut respawned = 0usize;
    for idx in indices.iter_mut() {
        let target_hz = space.centers_hz[*idx] * ratio;
        let mut new_idx = nearest_bin(space, target_hz);
        if new_idx < min_idx || new_idx > max_idx {
            let pick = rng.random_range(min_idx..(max_idx + 1));
            new_idx = pick;
            respawned += 1;
        }
        if new_idx == min_idx {
            count_min += 1;
        }
        if new_idx == max_idx {
            count_max += 1;
        }
        *idx = new_idx;
    }
    (count_min, count_max, respawned)
}

fn update_agent_indices_scored(
    indices: &mut [usize],
    c01_scan: &[f32],
    log2_ratio_scan: &[f32],
    min_idx: usize,
    max_idx: usize,
    k: i32,
    lambda: f32,
    sigma: f32,
) -> f32 {
    update_agent_indices_scored_stats(
        indices,
        c01_scan,
        log2_ratio_scan,
        min_idx,
        max_idx,
        k,
        lambda,
        sigma,
    )
    .mean_score
}

fn update_agent_indices_scored_stats(
    indices: &mut [usize],
    c01_scan: &[f32],
    log2_ratio_scan: &[f32],
    min_idx: usize,
    max_idx: usize,
    k: i32,
    lambda: f32,
    sigma: f32,
) -> UpdateStats {
    let order: Vec<usize> = (0..indices.len()).collect();
    update_agent_indices_scored_stats_with_order(
        indices,
        c01_scan,
        log2_ratio_scan,
        min_idx,
        max_idx,
        k,
        lambda,
        sigma,
        &order,
    )
}

fn update_agent_indices_scored_stats_with_order(
    indices: &mut [usize],
    c01_scan: &[f32],
    log2_ratio_scan: &[f32],
    min_idx: usize,
    max_idx: usize,
    k: i32,
    lambda: f32,
    sigma: f32,
    order: &[usize],
) -> UpdateStats {
    if indices.is_empty() {
        return UpdateStats {
            mean_score: 0.0,
            mean_repulsion: 0.0,
        };
    }
    let sigma = sigma.max(1e-6);
    let prev_indices = indices.to_vec();
    let prev_log2: Vec<f32> = prev_indices
        .iter()
        .map(|&idx| log2_ratio_scan[idx])
        .collect();
    let mut next_indices = prev_indices.clone();
    let mut score_sum = 0.0f32;
    let mut repulsion_sum = 0.0f32;
    let mut count = 0usize;

    for &agent_i in order {
        if agent_i >= prev_indices.len() {
            continue;
        }
        let current_idx = prev_indices[agent_i];
        let start = (current_idx as isize - k as isize).max(min_idx as isize) as usize;
        let end = (current_idx as isize + k as isize).min(max_idx as isize) as usize;
        let mut best_idx = current_idx;
        let mut best_score = f32::NEG_INFINITY;
        let mut best_repulsion = 0.0f32;
        for cand in start..=end {
            let cand_log2 = log2_ratio_scan[cand];
            let mut repulsion = 0.0f32;
            for (j, &other_log2) in prev_log2.iter().enumerate() {
                if j == agent_i {
                    continue;
                }
                let dist = (cand_log2 - other_log2).abs();
                repulsion += (-dist / sigma).exp();
            }
            let c01 = c01_scan[cand];
            let score = c01 - lambda * repulsion;
            if score > best_score {
                best_score = score;
                best_idx = cand;
                best_repulsion = repulsion;
            }
        }
        next_indices[agent_i] = best_idx;
        if best_score.is_finite() {
            score_sum += best_score;
        }
        if best_repulsion.is_finite() {
            repulsion_sum += best_repulsion;
        }
        count += 1;
    }

    indices.copy_from_slice(&next_indices);
    if count == 0 {
        return UpdateStats {
            mean_score: 0.0,
            mean_repulsion: 0.0,
        };
    }
    let inv = 1.0 / count as f32;
    UpdateStats {
        mean_score: score_sum * inv,
        mean_repulsion: repulsion_sum * inv,
    }
}

fn mean_at_indices(values: &[f32], indices: &[usize]) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }
    let sum: f32 = indices.iter().map(|&idx| values[idx]).sum();
    sum / indices.len() as f32
}

fn render_series_plot(
    out_path: &Path,
    caption: &str,
    y_desc: &str,
    series: &[(f32, f32)],
) -> Result<(), Box<dyn Error>> {
    let x_max = series.last().map(|(x, _)| *x).unwrap_or(0.0);
    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for &(_, y) in series {
        if y.is_finite() {
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = 0.0;
        y_max = 1.0;
    }
    let range = (y_max - y_min).abs();
    let pad = if range > 1e-6 {
        0.1 * range
    } else {
        0.1 * y_max.abs().max(1.0)
    };
    let y_lo = y_min - pad;
    let y_hi = y_max + pad;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..x_max.max(1.0), y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc(y_desc)
        .draw()?;

    chart.draw_series(LineSeries::new(series.iter().copied(), &BLUE))?;
    root.present()?;
    Ok(())
}

fn render_agent_trajectories_plot(
    out_path: &Path,
    trajectories: &[Vec<f32>],
) -> Result<(), Box<dyn Error>> {
    if trajectories.is_empty() {
        return Ok(());
    }
    let steps = trajectories
        .iter()
        .map(|trace| trace.len())
        .max()
        .unwrap_or(0);
    if steps == 0 {
        return Ok(());
    }

    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for trace in trajectories {
        for &val in trace {
            if val.is_finite() {
                y_min = y_min.min(val);
                y_max = y_max.max(val);
            }
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = -12.0;
        y_max = 12.0;
    }
    let range = (y_max - y_min).abs();
    let pad = if range > 1e-6 { 0.1 * range } else { 1.0 };
    let y_lo = y_min - pad;
    let y_hi = y_max + pad;

    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("E2 Agent Trajectories (Semitones)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0.0f32..(steps.saturating_sub(1) as f32).max(1.0),
            y_lo..y_hi,
        )?;

    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc("semitones")
        .draw()?;

    for (agent_id, trace) in trajectories.iter().enumerate() {
        if trace.is_empty() {
            continue;
        }
        let series = trace
            .iter()
            .enumerate()
            .map(|(step, &val)| (step as f32, val));
        let color = Palette99::pick(agent_id).mix(0.5);
        chart.draw_series(LineSeries::new(series, &color))?;
    }

    root.present()?;
    Ok(())
}

fn render_interval_histogram(
    out_path: &Path,
    caption: &str,
    values: &[f32],
    min: f32,
    max: f32,
    bin_width: f32,
) -> Result<(), Box<dyn Error>> {
    let counts = histogram_counts(values, min, max, bin_width);
    let y_max = counts
        .iter()
        .map(|(_, count)| *count as f32)
        .fold(0.0f32, f32::max)
        .max(1.0);

    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min..max, 0.0f32..(y_max * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("semitones")
        .y_desc("count")
        .x_labels(25)
        .draw()?;

    for (bin_start, count) in counts {
        let x0 = bin_start;
        let x1 = bin_start + bin_width;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, count as f32)],
            BLUE.mix(0.6).filled(),
        )))?;
    }

    root.present()?;
    Ok(())
}

fn render_consonance_lifetime_scatter(
    out_path: &Path,
    deaths: &[(usize, u32, f32)],
) -> Result<(), Box<dyn Error>> {
    let y_max = deaths
        .iter()
        .map(|(_, lifetime, _)| *lifetime as f32)
        .fold(0.0f32, f32::max)
        .max(1.0);

    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("E3 Consonance vs Lifetime", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, 0.0f32..(y_max * 1.05))?;

    chart
        .configure_mesh()
        .x_desc("avg C01")
        .y_desc("lifetime (steps)")
        .draw()?;

    let points = deaths
        .iter()
        .map(|(_, lifetime, avg_c01)| (*avg_c01, *lifetime as f32));
    chart.draw_series(points.map(|(x, y)| Circle::new((x, y), 3, RED.filled())))?;

    root.present()?;
    Ok(())
}

fn render_survival_curve(
    out_path: &Path,
    deaths: &[(usize, u32, f32)],
) -> Result<(), Box<dyn Error>> {
    if deaths.is_empty() {
        return Ok(());
    }
    let lifetimes: Vec<u32> = deaths.iter().map(|(_, lifetime, _)| *lifetime).collect();
    let max_t = lifetimes.iter().copied().max().unwrap_or(0) as usize;
    let series = build_survival_series(&lifetimes, max_t);

    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("E3 Survival Curve", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..(max_t as f32), 0.0f32..1.05f32)?;

    chart
        .configure_mesh()
        .x_desc("time (steps)")
        .y_desc("survival S(t)")
        .draw()?;

    chart.draw_series(LineSeries::new(series, &BLACK))?;
    root.present()?;
    Ok(())
}

fn render_survival_by_c01(
    out_path: &Path,
    deaths: &[(usize, u32, f32)],
) -> Result<(), Box<dyn Error>> {
    if deaths.is_empty() {
        return Ok(());
    }
    let mut c01_values: Vec<f32> = deaths.iter().map(|(_, _, c01)| *c01).collect();
    c01_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if c01_values.len() % 2 == 0 {
        let hi = c01_values.len() / 2;
        let lo = hi.saturating_sub(1);
        0.5 * (c01_values[lo] + c01_values[hi])
    } else {
        c01_values[c01_values.len() / 2]
    };

    let mut high: Vec<u32> = Vec::new();
    let mut low: Vec<u32> = Vec::new();
    for (_, lifetime, c01) in deaths {
        if *c01 >= median {
            high.push(*lifetime);
        } else {
            low.push(*lifetime);
        }
    }

    let max_t = deaths
        .iter()
        .map(|(_, lifetime, _)| *lifetime)
        .max()
        .unwrap_or(0) as usize;

    let high_series = build_survival_series(&high, max_t);
    let low_series = build_survival_series(&low, max_t);

    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("E3 Survival by C01 (Median Split)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..(max_t as f32), 0.0f32..1.05f32)?;

    chart
        .configure_mesh()
        .x_desc("time (steps)")
        .y_desc("survival S(t)")
        .draw()?;

    if !high_series.is_empty() {
        chart
            .draw_series(LineSeries::new(high_series, &BLUE))?
            .label("avg C01 >= median")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
    }
    if !low_series.is_empty() {
        chart
            .draw_series(LineSeries::new(low_series, &RED))?
            .label("avg C01 < median")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn render_order_offset_plot(
    out_path: &Path,
    series: &[(f32, f32, f32, f32, f32)],
) -> Result<(), Box<dyn Error>> {
    let x_max = series.last().map(|(x, _, _, _, _)| *x).unwrap_or(0.0);
    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("E5 Order Parameters", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..x_max.max(1.0), 0.0f32..1.05f32)?;

    chart
        .configure_mesh()
        .x_desc("time (s)")
        .y_desc("order r")
        .draw()?;

    let r_a_points = series.iter().map(|(t, r_a, _, _, _)| (*t, *r_a));
    let r_b_points = series.iter().map(|(t, _, r_b, _, _)| (*t, *r_b));

    chart
        .draw_series(LineSeries::new(r_a_points, &BLUE))?
        .label("r_a(t)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart
        .draw_series(LineSeries::new(r_b_points, &GREEN))?
        .label("r_b(t)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn render_delta_phi_plot(
    out_path: &Path,
    series: &[(f32, f32, f32, f32, f32)],
) -> Result<(), Box<dyn Error>> {
    let x_max = series.last().map(|(x, _, _, _, _)| *x).unwrap_or(0.0);
    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("E5 Mean Phase Offset Δφ", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..x_max.max(1.0), -PI..PI)?;

    chart
        .configure_mesh()
        .x_desc("time (s)")
        .y_desc("Δφ (rad)")
        .draw()?;

    let offset_points = series
        .iter()
        .map(|(t, _, _, delta_phi, _)| (*t, *delta_phi));
    chart.draw_series(LineSeries::new(offset_points, &RED))?;

    root.present()?;
    Ok(())
}

fn render_plv_ab_plot(
    out_path: &Path,
    series: &[(f32, f32, f32, f32, f32)],
) -> Result<(), Box<dyn Error>> {
    let x_max = series.last().map(|(x, _, _, _, _)| *x).unwrap_or(0.0);
    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("E5 Paired PLV_ab Over Time", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..x_max.max(1.0), 0.0f32..1.05f32)?;

    chart
        .configure_mesh()
        .x_desc("time (s)")
        .y_desc("value")
        .draw()?;

    let plv_points = series.iter().map(|(t, _, _, _, plv)| (*t, *plv));
    let r_a_points = series.iter().map(|(t, r_a, _, _, _)| (*t, *r_a));
    let r_b_points = series.iter().map(|(t, _, r_b, _, _)| (*t, *r_b));

    chart
        .draw_series(LineSeries::new(plv_points, &RED))?
        .label("plv_ab_paired(t)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .draw_series(LineSeries::new(r_a_points, &BLUE.mix(0.6)))?
        .label("r_a(t)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.mix(0.6)));

    chart
        .draw_series(LineSeries::new(r_b_points, &GREEN.mix(0.6)))?
        .label("r_b(t)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN.mix(0.6)));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn render_phase_histogram(
    out_path: &Path,
    phase_diffs: &[f32],
    bins: usize,
) -> Result<(), Box<dyn Error>> {
    if bins == 0 {
        return Ok(());
    }
    let min = -PI;
    let max = PI;
    let bin_width = (max - min) / bins as f32;
    let counts = histogram_counts(phase_diffs, min, max, bin_width);
    let y_max = counts
        .iter()
        .map(|(_, count)| *count as f32)
        .fold(0.0f32, f32::max)
        .max(1.0);

    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("E5 Phase Difference Histogram", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min..max, 0.0f32..(y_max * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("Δθ (rad)")
        .y_desc("count")
        .x_labels(9)
        .draw()?;

    for (bin_start, count) in counts {
        let x0 = bin_start;
        let x1 = bin_start + bin_width;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, count as f32)],
            GREEN.mix(0.6).filled(),
        )))?;
    }

    root.present()?;
    Ok(())
}

fn histogram_counts(values: &[f32], min: f32, max: f32, bin_width: f32) -> Vec<(f32, usize)> {
    if bin_width <= 0.0 {
        return Vec::new();
    }
    let bins = ((max - min) / bin_width).ceil().max(1.0) as usize;
    let mut counts = vec![0usize; bins];
    for &value in values {
        if value < min || value > max {
            continue;
        }
        let mut idx = ((value - min) / bin_width).floor() as isize;
        if idx as usize >= bins {
            idx = (bins - 1) as isize;
        }
        if idx >= 0 {
            counts[idx as usize] += 1;
        }
    }
    (0..bins)
        .map(|i| (min + i as f32 * bin_width, counts[i]))
        .collect()
}

fn phase_hist_bins(sample_count: usize) -> usize {
    if sample_count == 0 {
        return 12;
    }
    let bins = ((sample_count as f32).sqrt() * 2.0).round() as usize;
    bins.clamp(12, 96)
}

fn plv_time_from_series(series: &[(f32, f32, f32, f32, f32)], window: usize) -> f32 {
    if series.is_empty() {
        return 0.0;
    }
    let window = window.clamp(1, series.len());
    let start = series.len().saturating_sub(window);
    let mut mean_cos = 0.0f32;
    let mut mean_sin = 0.0f32;
    for (_, _, _, delta_phi, _) in series.iter().skip(start) {
        mean_cos += delta_phi.cos();
        mean_sin += delta_phi.sin();
    }
    let inv = 1.0 / window as f32;
    mean_cos *= inv;
    mean_sin *= inv;
    (mean_cos * mean_cos + mean_sin * mean_sin).sqrt()
}

fn wrap_to_pi(theta: f32) -> f32 {
    let two_pi = 2.0 * PI;
    let mut x = theta.rem_euclid(two_pi);
    if x > PI {
        x -= two_pi;
    }
    x
}

fn pairwise_interval_samples(semitones: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    let eps = 1e-6f32;
    for i in 0..semitones.len() {
        for j in (i + 1)..semitones.len() {
            let diff = (semitones[i] - semitones[j]).abs();
            let mut folded = diff.rem_euclid(12.0);
            if folded < eps && diff > eps {
                folded = 12.0;
            }
            out.push(folded);
        }
    }
    out
}

fn build_survival_series(lifetimes: &[u32], max_t: usize) -> Vec<(f32, f32)> {
    if lifetimes.is_empty() {
        return Vec::new();
    }
    let mut sorted = lifetimes.to_vec();
    sorted.sort_unstable();
    let total = sorted.len() as f32;
    let mut idx = 0usize;
    let mut series = Vec::with_capacity(max_t + 1);
    for t in 0..=max_t {
        let t_u32 = t as u32;
        while idx < sorted.len() && sorted[idx] < t_u32 {
            idx += 1;
        }
        let survivors = (sorted.len() - idx) as f32;
        series.push((t as f32, survivors / total));
    }
    series
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_to_pi_clamps_range() {
        let samples = [
            -10.0 * PI,
            -3.5 * PI,
            -PI,
            -0.5 * PI,
            0.0,
            0.5 * PI,
            PI,
            3.0 * PI,
            9.0 * PI,
        ];
        for &theta in &samples {
            let wrapped = wrap_to_pi(theta);
            assert!(
                wrapped >= -PI - 1e-6 && wrapped <= PI + 1e-6,
                "wrapped={wrapped} out of range for theta={theta}"
            );
        }
    }

    #[test]
    fn histogram_counts_include_endpoints() {
        let values = [0.0f32, 1.0f32];
        let counts = histogram_counts(&values, 0.0, 1.0, 0.5);
        assert_eq!(counts.len(), 2);
        assert_eq!(counts[0].1, 1);
        assert_eq!(counts[1].1, 1);
    }

    #[test]
    fn histogram_counts_sum_matches_in_range() {
        let values = [0.0f32, 0.25, 0.5, 1.0, 1.5, -0.2];
        let counts = histogram_counts(&values, 0.0, 1.0, 0.5);
        let sum: usize = counts.iter().map(|(_, c)| *c).sum();
        assert_eq!(sum, 4);
    }

    #[test]
    fn pairwise_intervals_have_expected_count_and_range() {
        let semitones = [0.0f32, 3.0, 4.0, 7.0];
        let pairs = pairwise_interval_samples(&semitones);
        assert_eq!(pairs.len(), 6);
        for &v in &pairs {
            assert!(v >= 0.0 && v <= 12.0 + 1e-6, "value out of range: {v}");
        }
    }

    #[test]
    fn pairwise_interval_fold_maps_octave_to_12() {
        let semitones = [0.0f32, 12.0];
        let pairs = pairwise_interval_samples(&semitones);
        assert_eq!(pairs.len(), 1);
        assert!(
            (pairs[0] - 12.0).abs() < 1e-6,
            "expected 12, got {}",
            pairs[0]
        );
    }

    #[test]
    fn update_agent_indices_stays_in_bounds() {
        let mut indices = vec![1usize, 2, 3];
        let c01_scan = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let log2_ratio_scan = vec![0.0f32, 0.1, 0.2, 0.3, 0.4];
        update_agent_indices(
            &mut indices,
            &c01_scan,
            &log2_ratio_scan,
            1,
            3,
            1,
            0.05,
            0.1,
        );
        assert!(indices.iter().all(|&idx| idx >= 1 && idx <= 3));
    }

    #[test]
    fn update_agent_indices_is_order_independent() {
        let c01_scan = vec![0.1f32, 0.4, 0.3, 0.8, 0.6, 0.2];
        let log2_ratio_scan = vec![0.0f32, 0.1, 0.2, 0.3, 0.4, 0.5];
        let mut indices_fwd = vec![1usize, 3, 4];
        let mut indices_rev = indices_fwd.clone();
        let order_fwd: Vec<usize> = (0..indices_fwd.len()).collect();
        let order_rev: Vec<usize> = (0..indices_fwd.len()).rev().collect();
        let stats_fwd = update_agent_indices_scored_stats_with_order(
            &mut indices_fwd,
            &c01_scan,
            &log2_ratio_scan,
            0,
            5,
            1,
            0.2,
            0.1,
            &order_fwd,
        );
        let stats_rev = update_agent_indices_scored_stats_with_order(
            &mut indices_rev,
            &c01_scan,
            &log2_ratio_scan,
            0,
            5,
            1,
            0.2,
            0.1,
            &order_rev,
        );
        assert_eq!(indices_fwd, indices_rev);
        assert!((stats_fwd.mean_score - stats_rev.mean_score).abs() < 1e-6);
    }

    #[test]
    fn e5_sim_outputs_have_expected_shapes() {
        let sim = simulate_e5(E5_SEED, 120);
        assert_eq!(sim.series.len(), 120);
        assert_eq!(sim.thetas_last_a.len(), E5_GROUP_N);
        assert_eq!(sim.thetas_last_b.len(), E5_GROUP_N);
        assert!(
            (sim.final_t - sim.series.last().unwrap().0).abs() < 1e-6,
            "final_t mismatch"
        );
    }

    #[test]
    fn e5_histogram_sample_count_matches_window() {
        let steps = E5_BURN_IN_STEPS + E5_SAMPLE_WINDOW_STEPS + 10;
        let sim = simulate_e5(E5_SEED, steps);
        let expected = sim.n_a * sim.sample_window_steps;
        assert_eq!(sim.phase_hist_samples.len(), expected);
    }

    #[test]
    fn e5_plv_ab_within_unit_range() {
        let sim = simulate_e5(E5_SEED, 200);
        for (_, _, _, _, plv_ab) in &sim.series {
            assert!(
                *plv_ab >= -1e-6 && *plv_ab <= 1.0 + 1e-6,
                "plv_ab out of range: {plv_ab}"
            );
        }
    }

    #[test]
    fn e5_plv_ab_improves_after_burn_in() {
        let sim = simulate_e5(E5_SEED, 600);
        let early = sim
            .series
            .iter()
            .take(100)
            .map(|(_, _, _, _, plv)| *plv)
            .sum::<f32>()
            / 100.0;
        let late = sim
            .series
            .iter()
            .rev()
            .take(100)
            .map(|(_, _, _, _, plv)| *plv)
            .sum::<f32>()
            / 100.0;
        assert!(
            late > early + 0.2,
            "expected PLV_ab to improve (early={early:.3}, late={late:.3})"
        );
    }

    #[test]
    fn e5_plv_time_is_high_in_late_window() {
        let sim = simulate_e5(E5_SEED, 600);
        assert!(
            sim.plv_time > 0.8,
            "expected PLV_time > 0.8, got {:.3}",
            sim.plv_time
        );
    }

    #[test]
    fn shift_indices_by_ratio_respawns_and_clamps() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let n = space.n_bins();
        assert!(n > 6);
        let mut rng = seeded_rng(42);
        let min_idx = 2usize;
        let max_idx = 5usize;
        let mut indices = vec![0usize, 1, n - 2, n - 1];
        let (_count_min, _count_max, respawned) =
            shift_indices_by_ratio(&space, &mut indices, 10.0, min_idx, max_idx, &mut rng);
        assert!(respawned > 0);
        assert!(indices.iter().all(|&idx| idx >= min_idx && idx <= max_idx));
    }
}

fn erb_grid_for_space(space: &Log2Space) -> (Vec<f32>, Vec<f32>) {
    let erb_scan: Vec<f32> = space.centers_hz.iter().map(|&f| hz_to_erb(f)).collect();
    let du_scan = local_du_from_grid(&erb_scan);
    (erb_scan, du_scan)
}

fn local_du_from_grid(grid: &[f32]) -> Vec<f32> {
    if grid.is_empty() {
        return Vec::new();
    }
    if grid.len() == 1 {
        return vec![1.0];
    }

    let n = grid.len();
    let mut du = vec![0.0f32; n];
    du[0] = (grid[1] - grid[0]).max(0.0);
    du[n - 1] = (grid[n - 1] - grid[n - 2]).max(0.0);
    for i in 1..n - 1 {
        du[i] = (0.5 * (grid[i + 1] - grid[i - 1])).max(0.0);
    }
    du
}

fn nearest_bin(space: &Log2Space, hz: f32) -> usize {
    let mut best_idx = 0;
    let mut best_diff = f32::MAX;
    for (i, &f) in space.centers_hz.iter().enumerate() {
        let diff = (f - hz).abs();
        if diff < best_diff {
            best_diff = diff;
            best_idx = i;
        }
    }
    best_idx
}
