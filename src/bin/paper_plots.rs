use std::error::Error;
use std::f32::consts::PI;
use std::fs::{create_dir_all, write};
use std::path::Path;

use plotters::coord::types::RangedCoordf32;
use plotters::prelude::*;

use conchordal::core::erb::hz_to_erb;
use conchordal::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
use conchordal::core::landscape::{LandscapeParams, RoughnessScalarMode};
use conchordal::core::log2space::Log2Space;
use conchordal::core::psycho_state;
use conchordal::core::roughness_kernel::{KernelParams, RoughnessKernel};
use conchordal::paper::sim::{E4_ANCHOR_HZ, E4TailSamples, run_e4_condition_tail_samples};
use rand::{Rng, SeedableRng, rngs::StdRng};

const SPACE_BINS_PER_OCT: u32 = 400;

const E2_STEPS: usize = 400;
const E2_BURN_IN: usize = 100;
const E2_ANCHOR_SHIFT_STEP: usize = 200;
const E2_ANCHOR_SHIFT_RATIO: f32 = 0.5;
const E2_STEP_SEMITONES: f32 = 0.5;
const E2_ANCHOR_BIN_ST: f32 = 0.5;
const E2_PAIRWISE_BIN_ST: f32 = 0.25;
const E2_N_AGENTS: usize = 24;
const E2_LAMBDA: f32 = 0.15;
const E2_SIGMA: f32 = 0.06;
const E2_SEEDS: [u64; 5] = [
    0xC0FFEE_u64,
    0xC0FFEE_u64 + 1,
    0xC0FFEE_u64 + 2,
    0xC0FFEE_u64 + 3,
    0xC0FFEE_u64 + 4,
];

const E4_TAIL_WINDOW_STEPS: u32 = 200;
const E4_DELTA_TAU: f32 = 0.02;
const E4_WEIGHT_COARSE_STEP: f32 = 0.1;
const E4_WEIGHT_FINE_STEP: f32 = 0.05;
const E4_BIN_WIDTHS: [f32; 2] = [0.25, 0.5];
const E4_SEEDS: [u64; 5] = [
    0xC0FFEE_u64 + 10,
    0xC0FFEE_u64 + 11,
    0xC0FFEE_u64 + 12,
    0xC0FFEE_u64 + 13,
    0xC0FFEE_u64 + 14,
];
const E4_REP_WEIGHTS: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

const E3_STEP_SEMITONES: f32 = 0.5;

const E5_KICK_OMEGA: f32 = 2.0 * PI * 2.0;
const E5_AGENT_OMEGA_MEAN: f32 = 2.0 * PI * 1.8;
const E5_AGENT_JITTER: f32 = 0.02;
const E5_K_KICK: f32 = 1.6;
const E5_N_AGENTS: usize = 32;
const E5_DT: f32 = 0.02;
const E5_STEPS: usize = 2000;
const E5_BURN_IN_STEPS: usize = 500;
const E5_SAMPLE_WINDOW_STEPS: usize = 250;
const E5_TIME_PLV_WINDOW_STEPS: usize = 200;
const E5_KICK_ON_STEP: Option<usize> = Some(800);
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
    let baseline_seed = E2_SEEDS[0];
    let baseline_run = run_e2_once(space, anchor_hz, baseline_seed, E2Condition::Baseline);
    let marker_steps = e2_marker_steps();

    write(
        out_dir.join("paper_e2_meta.txt"),
        e2_meta_text(baseline_run.n_agents, baseline_run.k_bins),
    )?;

    write(
        out_dir.join("paper_e2_timeseries.csv"),
        series_csv("t,mean_c01", &baseline_run.mean_c01_series),
    )?;
    write(
        out_dir.join("paper_e2_score_timeseries.csv"),
        series_csv("t,mean_score", &baseline_run.mean_score_series),
    )?;
    write(
        out_dir.join("paper_e2_repulsion_timeseries.csv"),
        series_csv("t,mean_repulsion", &baseline_run.mean_repulsion_series),
    )?;
    write(
        out_dir.join("paper_e2_moved_frac_timeseries.csv"),
        series_csv("t,moved_frac", &baseline_run.moved_frac_series),
    )?;

    write(
        out_dir.join("paper_e2_agent_trajectories.csv"),
        trajectories_csv(&baseline_run),
    )?;
    write(
        out_dir.join("paper_e2_anchor_shift_stats.csv"),
        anchor_shift_csv(&baseline_run),
    )?;
    write(
        out_dir.join("paper_e2_final_agents.csv"),
        final_agents_csv(&baseline_run),
    )?;

    let mean_plot_path = out_dir.join("paper_e2_mean_consonance_over_time.png");
    render_series_plot_with_markers(
        &mean_plot_path,
        "E2 Mean Consonance Over Time (burn-in=100, shift@200)",
        "mean C01",
        &series_pairs(&baseline_run.mean_c01_series),
        &marker_steps,
    )?;

    let mean_score_path = out_dir.join("paper_e2_mean_score_over_time.png");
    render_series_plot_with_markers(
        &mean_score_path,
        "E2 Mean Score Over Time (burn-in=100, shift@200)",
        "mean score (c01 - λ·repulsion)",
        &series_pairs(&baseline_run.mean_score_series),
        &marker_steps,
    )?;

    let mean_repulsion_path = out_dir.join("paper_e2_mean_repulsion_over_time.png");
    render_series_plot_with_markers(
        &mean_repulsion_path,
        "E2 Mean Repulsion Over Time (burn-in=100, shift@200)",
        "mean repulsion",
        &series_pairs(&baseline_run.mean_repulsion_series),
        &marker_steps,
    )?;

    let moved_frac_path = out_dir.join("paper_e2_moved_frac_over_time.png");
    render_series_plot_with_markers(
        &moved_frac_path,
        "E2 Moved Fraction Over Time (burn-in=100, shift@200)",
        "moved fraction",
        &series_pairs(&baseline_run.moved_frac_series),
        &marker_steps,
    )?;

    let trajectory_path = out_dir.join("paper_e2_agent_trajectories.png");
    render_agent_trajectories_plot(&trajectory_path, &baseline_run.trajectory_semitones)?;

    let pairwise_intervals = pairwise_interval_samples(&baseline_run.final_semitones);
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

    let hist_path = out_dir.join("paper_e2_interval_histogram.png");
    render_interval_histogram(
        &hist_path,
        "E2 Interval Histogram (post, bin=0.50st)",
        &baseline_run.semitone_samples_post,
        -12.0,
        12.0,
        E2_ANCHOR_BIN_ST,
    )?;

    render_e2_histogram_sweep(out_dir, &baseline_run)?;

    let (baseline_runs, baseline_stats) = e2_seed_sweep(space, anchor_hz, E2Condition::Baseline);
    let (nohill_runs, nohill_stats) = e2_seed_sweep(space, anchor_hz, E2Condition::NoHillClimb);
    let (norep_runs, norep_stats) = e2_seed_sweep(space, anchor_hz, E2Condition::NoRepulsion);

    write(
        out_dir.join("paper_e2_seed_sweep_mean_c01.csv"),
        sweep_csv(
            "step,mean,std,n",
            &baseline_stats.mean_c01,
            &baseline_stats.std_c01,
            baseline_stats.n,
        ),
    )?;
    write(
        out_dir.join("paper_e2_seed_sweep_mean_score.csv"),
        sweep_csv(
            "step,mean,std,n",
            &baseline_stats.mean_score,
            &baseline_stats.std_score,
            baseline_stats.n,
        ),
    )?;
    write(
        out_dir.join("paper_e2_seed_sweep_mean_repulsion.csv"),
        sweep_csv(
            "step,mean,std,n",
            &baseline_stats.mean_repulsion,
            &baseline_stats.std_repulsion,
            baseline_stats.n,
        ),
    )?;

    let sweep_mean_path = out_dir.join("paper_e2_mean_consonance_over_time_seeds.png");
    render_series_plot_with_band(
        &sweep_mean_path,
        "E2 Mean Consonance (seed sweep)",
        "mean C01",
        &baseline_stats.mean_c01,
        &baseline_stats.std_c01,
        &marker_steps,
    )?;

    let sweep_score_path = out_dir.join("paper_e2_mean_score_over_time_seeds.png");
    render_series_plot_with_band(
        &sweep_score_path,
        "E2 Mean Score (seed sweep)",
        "mean score",
        &baseline_stats.mean_score,
        &baseline_stats.std_score,
        &marker_steps,
    )?;

    let sweep_rep_path = out_dir.join("paper_e2_mean_repulsion_over_time_seeds.png");
    render_series_plot_with_band(
        &sweep_rep_path,
        "E2 Mean Repulsion (seed sweep)",
        "mean repulsion",
        &baseline_stats.mean_repulsion,
        &baseline_stats.std_repulsion,
        &marker_steps,
    )?;

    write(
        out_dir.join("paper_e2_control_mean_c01.csv"),
        e2_controls_csv(&baseline_stats, &nohill_stats, &norep_stats),
    )?;

    let control_plot_path = out_dir.join("paper_e2_mean_consonance_over_time_controls.png");
    render_series_plot_multi(
        &control_plot_path,
        "E2 Mean Consonance (controls)",
        "mean C01",
        &[
            ("baseline", &baseline_stats.mean_c01, BLUE),
            ("no hill-climb", &nohill_stats.mean_c01, RED),
            ("no repulsion", &norep_stats.mean_c01, GREEN),
        ],
        &marker_steps,
    )?;

    render_e2_control_histograms(out_dir, &baseline_runs[0], &nohill_runs[0], &norep_runs[0])?;

    Ok(())
}

fn run_e2_once(space: &Log2Space, anchor_hz: f32, seed: u64, condition: E2Condition) -> E2Run {
    let mut rng = seeded_rng(seed);
    let mut anchor_hz_current = anchor_hz;
    let mut log2_ratio_scan = build_log2_ratio_scan(space, anchor_hz_current);
    let (mut min_idx, mut max_idx) = log2_ratio_bounds(&log2_ratio_scan, -1.0, 1.0);

    let mut agent_indices: Vec<usize> = (0..E2_N_AGENTS)
        .map(|_| rng.random_range(min_idx..=max_idx))
        .collect();

    let (_erb_scan, du_scan) = erb_grid_for_space(space);
    let workspace = build_c01_workspace(space);
    let k_bins = k_from_semitones(E2_STEP_SEMITONES);

    let mut mean_c01_series = Vec::with_capacity(E2_STEPS);
    let mut mean_score_series = Vec::with_capacity(E2_STEPS);
    let mut mean_repulsion_series = Vec::with_capacity(E2_STEPS);
    let mut moved_frac_series = Vec::with_capacity(E2_STEPS);
    let mut semitone_samples_pre = Vec::new();
    let mut semitone_samples_post = Vec::new();

    let mut trajectory_semitones = vec![Vec::with_capacity(E2_STEPS); E2_N_AGENTS];
    let mut trajectory_c01 = vec![Vec::with_capacity(E2_STEPS); E2_N_AGENTS];

    let mut anchor_shift = E2AnchorShiftStats {
        step: E2_ANCHOR_SHIFT_STEP,
        anchor_hz_before: anchor_hz_current,
        anchor_hz_after: anchor_hz_current * E2_ANCHOR_SHIFT_RATIO,
        count_min: 0,
        count_max: 0,
        respawned: 0,
    };

    for step in 0..E2_STEPS {
        if step == E2_ANCHOR_SHIFT_STEP {
            let before = anchor_hz_current;
            anchor_hz_current *= E2_ANCHOR_SHIFT_RATIO;
            log2_ratio_scan = build_log2_ratio_scan(space, anchor_hz_current);
            let (new_min, new_max) = log2_ratio_bounds(&log2_ratio_scan, -1.0, 1.0);
            let (count_min, count_max, respawned) = shift_indices_by_ratio(
                space,
                &mut agent_indices,
                E2_ANCHOR_SHIFT_RATIO,
                new_min,
                new_max,
                &mut rng,
            );
            anchor_shift = E2AnchorShiftStats {
                step,
                anchor_hz_before: before,
                anchor_hz_after: anchor_hz_current,
                count_min,
                count_max,
                respawned,
            };
            min_idx = new_min;
            max_idx = new_max;
        }

        let anchor_idx = nearest_bin(space, anchor_hz_current);
        let (env_scan, density_scan) = build_env_scans(space, anchor_idx, &agent_indices, &du_scan);
        let c01_scan = compute_c01_scan(space, &workspace, &env_scan, &density_scan);

        let mean_c01 = mean_at_indices(&c01_scan, &agent_indices);
        mean_c01_series.push(mean_c01);

        for (agent_id, &idx) in agent_indices.iter().enumerate() {
            let semitone = 12.0 * log2_ratio_scan[idx];
            trajectory_semitones[agent_id].push(semitone);
            trajectory_c01[agent_id].push(c01_scan[idx]);
        }

        if step >= E2_BURN_IN {
            let target = if step < E2_ANCHOR_SHIFT_STEP {
                &mut semitone_samples_pre
            } else {
                &mut semitone_samples_post
            };
            target.extend(agent_indices.iter().map(|&idx| 12.0 * log2_ratio_scan[idx]));
        }

        let stats = match condition {
            E2Condition::Baseline => update_agent_indices_scored_stats(
                &mut agent_indices,
                &c01_scan,
                &log2_ratio_scan,
                min_idx,
                max_idx,
                k_bins,
                E2_LAMBDA,
                E2_SIGMA,
            ),
            E2Condition::NoRepulsion => update_agent_indices_scored_stats(
                &mut agent_indices,
                &c01_scan,
                &log2_ratio_scan,
                min_idx,
                max_idx,
                k_bins,
                0.0,
                E2_SIGMA,
            ),
            E2Condition::NoHillClimb => score_stats_at_indices(
                &agent_indices,
                &c01_scan,
                &log2_ratio_scan,
                E2_LAMBDA,
                E2_SIGMA,
            ),
        };

        mean_score_series.push(stats.mean_score);
        mean_repulsion_series.push(stats.mean_repulsion);
        moved_frac_series.push(stats.moved_frac);
    }

    let mut final_semitones = Vec::with_capacity(E2_N_AGENTS);
    let mut final_log2_ratios = Vec::with_capacity(E2_N_AGENTS);
    let mut final_freqs_hz = Vec::with_capacity(E2_N_AGENTS);
    for &idx in &agent_indices {
        final_semitones.push(12.0 * log2_ratio_scan[idx]);
        final_log2_ratios.push(log2_ratio_scan[idx]);
        final_freqs_hz.push(space.centers_hz[idx]);
    }

    E2Run {
        mean_c01_series,
        mean_score_series,
        mean_repulsion_series,
        moved_frac_series,
        semitone_samples_pre,
        semitone_samples_post,
        final_semitones,
        final_freqs_hz,
        final_log2_ratios,
        trajectory_semitones,
        trajectory_c01,
        anchor_shift,
        n_agents: E2_N_AGENTS,
        k_bins,
    }
}

fn e2_seed_sweep(
    space: &Log2Space,
    anchor_hz: f32,
    condition: E2Condition,
) -> (Vec<E2Run>, E2SweepStats) {
    let mut runs: Vec<E2Run> = Vec::new();
    for &seed in &E2_SEEDS {
        runs.push(run_e2_once(space, anchor_hz, seed, condition));
    }

    let n = runs.len();
    let mean_c01 = mean_std_series(runs.iter().map(|r| &r.mean_c01_series).collect::<Vec<_>>());
    let mean_score = mean_std_series(
        runs.iter()
            .map(|r| &r.mean_score_series)
            .collect::<Vec<_>>(),
    );
    let mean_repulsion = mean_std_series(
        runs.iter()
            .map(|r| &r.mean_repulsion_series)
            .collect::<Vec<_>>(),
    );

    (
        runs,
        E2SweepStats {
            mean_c01: mean_c01.0,
            std_c01: mean_c01.1,
            mean_score: mean_score.0,
            std_score: mean_score.1,
            mean_repulsion: mean_repulsion.0,
            std_repulsion: mean_repulsion.1,
            n,
        },
    )
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
    let sim_main = simulate_e5_kick(E5_SEED, E5_STEPS, E5_K_KICK, E5_KICK_ON_STEP);
    let sim_ctrl = simulate_e5_kick(E5_SEED, E5_STEPS, 0.0, E5_KICK_ON_STEP);

    let csv_path = out_dir.join("paper_e5_kick_entrainment.csv");
    write(&csv_path, e5_kick_csv(&sim_main, &sim_ctrl))?;

    let summary_path = out_dir.join("paper_e5_kick_summary.csv");
    write(&summary_path, e5_kick_summary_csv(&sim_main, &sim_ctrl))?;

    let order_path = out_dir.join("paper_e5_order_over_time.png");
    render_e5_order_plot(&order_path, &sim_main.series, &sim_ctrl.series)?;

    let delta_path = out_dir.join("paper_e5_delta_phi_over_time.png");
    render_e5_delta_phi_plot(&delta_path, &sim_main.series, &sim_ctrl.series)?;

    let plv_path = out_dir.join("paper_e5_plv_over_time.png");
    render_e5_plv_plot(&plv_path, &sim_main.series, &sim_ctrl.series)?;

    let bins = phase_hist_bins(
        sim_main
            .phase_hist_samples
            .len()
            .max(sim_ctrl.phase_hist_samples.len()),
    );
    let phase_path = out_dir.join("paper_e5_phase_diff_histogram.png");
    render_phase_histogram_compare(
        &phase_path,
        &sim_main.phase_hist_samples,
        &sim_ctrl.phase_hist_samples,
        bins,
    )?;

    Ok(())
}

#[allow(dead_code)]
struct E5KickSimResult {
    series: Vec<(f32, f32, f32, f32, f32, f32)>,
    phase_hist_samples: Vec<f32>,
    plv_time: f32,
    sample_window_steps: usize,
    n_agents: usize,
}

fn simulate_e5_kick(
    seed: u64,
    steps: usize,
    k_kick: f32,
    kick_on_step: Option<usize>,
) -> E5KickSimResult {
    let mut rng = seeded_rng(seed);
    let mut theta_kick = 0.0f32;
    let mut thetas: Vec<f32> = (0..E5_N_AGENTS)
        .map(|_| rng.random_range(0.0f32..(2.0 * PI)))
        .collect();
    let omegas: Vec<f32> = (0..E5_N_AGENTS)
        .map(|_| {
            let jitter_scale = rng.random_range(-E5_AGENT_JITTER..E5_AGENT_JITTER);
            E5_AGENT_OMEGA_MEAN * (1.0 + jitter_scale)
        })
        .collect();

    let mut plv_buffers: Vec<SlidingPlv> = (0..E5_N_AGENTS)
        .map(|_| SlidingPlv::new(E5_TIME_PLV_WINDOW_STEPS))
        .collect();
    let mut group_plv = SlidingPlv::new(E5_TIME_PLV_WINDOW_STEPS);

    let mut series: Vec<(f32, f32, f32, f32, f32, f32)> = Vec::with_capacity(steps);
    let mut phase_hist_samples: Vec<f32> = Vec::new();
    let sample_start = steps
        .saturating_sub(E5_SAMPLE_WINDOW_STEPS)
        .max(E5_BURN_IN_STEPS);

    for step in 0..steps {
        let t = step as f32 * E5_DT;
        let k_eff = if let Some(on_step) = kick_on_step {
            if step < on_step { 0.0 } else { k_kick }
        } else {
            k_kick
        };

        theta_kick += E5_KICK_OMEGA * E5_DT;
        for i in 0..E5_N_AGENTS {
            let theta_i = thetas[i];
            let dtheta = omegas[i] + k_eff * (theta_kick - theta_i).sin();
            thetas[i] = theta_i + dtheta * E5_DT;
        }

        let mut mean_cos = 0.0f32;
        let mut mean_sin = 0.0f32;
        for &theta in &thetas {
            mean_cos += theta.cos();
            mean_sin += theta.sin();
        }
        let inv = 1.0 / E5_N_AGENTS as f32;
        mean_cos *= inv;
        mean_sin *= inv;
        let r = (mean_cos * mean_cos + mean_sin * mean_sin).sqrt();
        let psi = mean_sin.atan2(mean_cos);
        let delta_phi = wrap_to_pi(psi - theta_kick);

        let mut plv_sum = 0.0f32;
        for i in 0..E5_N_AGENTS {
            let d_i = wrap_to_pi(thetas[i] - theta_kick);
            plv_buffers[i].push(d_i);
            plv_sum += plv_buffers[i].plv();
            if step >= sample_start {
                phase_hist_samples.push(d_i);
            }
        }
        let plv_agent_kick = plv_sum / E5_N_AGENTS as f32;
        group_plv.push(delta_phi);
        let plv_group_delta_phi = group_plv.plv();

        series.push((t, r, delta_phi, plv_agent_kick, plv_group_delta_phi, k_eff));
    }

    let plv_time = plv_time_from_series(&series, E5_TIME_PLV_WINDOW_STEPS);
    E5KickSimResult {
        series,
        phase_hist_samples,
        plv_time,
        sample_window_steps: steps.saturating_sub(sample_start),
        n_agents: E5_N_AGENTS,
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
    let coarse_weights = build_weight_grid(E4_WEIGHT_COARSE_STEP);
    let primary_bin = E4_BIN_WIDTHS[0];
    let (mut run_records, mut hist_records) =
        run_e4_sweep_for_weights(out_dir, anchor_hz, &coarse_weights, &E4_SEEDS, primary_bin)?;

    let mut weights = coarse_weights.clone();
    let fine_weights = refine_weights_from_sign_change(&run_records, primary_bin);
    for w in fine_weights {
        if !weights.iter().any(|&x| (x - w).abs() < 1e-6) {
            weights.push(w);
        }
    }
    weights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let fine_only: Vec<f32> = weights
        .iter()
        .copied()
        .filter(|w| !coarse_weights.iter().any(|c| (c - w).abs() < 1e-6))
        .collect();

    if !fine_only.is_empty() {
        let (more_runs, more_hists) =
            run_e4_sweep_for_weights(out_dir, anchor_hz, &fine_only, &E4_SEEDS, primary_bin)?;
        run_records.extend(more_runs);
        hist_records.extend(more_hists);
    }

    for &bin_width in E4_BIN_WIDTHS.iter().skip(1) {
        let (more_runs, more_hists) =
            run_e4_sweep_for_weights(out_dir, anchor_hz, &weights, &E4_SEEDS, bin_width)?;
        run_records.extend(more_runs);
        hist_records.extend(more_hists);
    }

    let runs_csv_path = out_dir.join("e4_mirror_sweep_runs.csv");
    write(&runs_csv_path, e4_runs_csv(&run_records))?;

    let summaries = summarize_e4_runs(&run_records);
    let summary_csv_path = out_dir.join("e4_mirror_sweep_summary.csv");
    write(&summary_csv_path, e4_summary_csv(&summaries))?;

    let overlay_path = out_dir.join(format!(
        "paper_e4_hist_overlay_bw{}.png",
        format_float_token(primary_bin)
    ));
    render_e4_hist_overlay(&overlay_path, &hist_records, primary_bin, &E4_REP_WEIGHTS)?;

    for &bin_width in &E4_BIN_WIDTHS {
        let delta_path = out_dir.join(format!(
            "paper_e4_delta_vs_weight_bw{}.png",
            format_float_token(bin_width)
        ));
        render_e4_delta_plot(&delta_path, &summaries, bin_width)?;

        let major_minor_path = out_dir.join(format!(
            "paper_e4_major_minor_vs_weight_bw{}.png",
            format_float_token(bin_width)
        ));
        render_e4_major_minor_plot(&major_minor_path, &summaries, bin_width)?;
    }

    let legacy_path = out_dir.join("paper_e4_mirror_sweep.png");
    render_e4_major_minor_plot(&legacy_path, &summaries, primary_bin)?;

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

#[derive(Clone, Copy)]
enum E2Condition {
    Baseline,
    NoHillClimb,
    NoRepulsion,
}

struct E2AnchorShiftStats {
    step: usize,
    anchor_hz_before: f32,
    anchor_hz_after: f32,
    count_min: usize,
    count_max: usize,
    respawned: usize,
}

struct E2Run {
    mean_c01_series: Vec<f32>,
    mean_score_series: Vec<f32>,
    mean_repulsion_series: Vec<f32>,
    moved_frac_series: Vec<f32>,
    semitone_samples_pre: Vec<f32>,
    semitone_samples_post: Vec<f32>,
    final_semitones: Vec<f32>,
    final_freqs_hz: Vec<f32>,
    final_log2_ratios: Vec<f32>,
    trajectory_semitones: Vec<Vec<f32>>,
    trajectory_c01: Vec<Vec<f32>>,
    anchor_shift: E2AnchorShiftStats,
    n_agents: usize,
    k_bins: i32,
}

struct E2SweepStats {
    mean_c01: Vec<f32>,
    std_c01: Vec<f32>,
    mean_score: Vec<f32>,
    std_score: Vec<f32>,
    mean_repulsion: Vec<f32>,
    std_repulsion: Vec<f32>,
    n: usize,
}

struct E4RunRecord {
    mirror_weight: f32,
    seed: u64,
    bin_width: f32,
    eps: f32,
    major_score: f32,
    minor_score: f32,
    delta: f32,
    major_frac: f32,
    steps_total: u32,
    burn_in: u32,
    tail_window: u32,
    histogram_source: &'static str,
}

struct E4SummaryRecord {
    mirror_weight: f32,
    bin_width: f32,
    eps: f32,
    mean_major: f32,
    std_major: f32,
    mean_minor: f32,
    std_minor: f32,
    mean_delta: f32,
    std_delta: f32,
    major_rate: f32,
    minor_rate: f32,
    ambiguous_rate: f32,
    n_runs: usize,
}

struct Histogram {
    bin_centers: Vec<f32>,
    masses: Vec<f32>,
}

struct E4HistRecord {
    mirror_weight: f32,
    bin_width: f32,
    histogram: Histogram,
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

fn build_weight_grid(step: f32) -> Vec<f32> {
    if step <= 0.0 {
        return vec![0.0, 1.0];
    }
    let mut weights = Vec::new();
    let mut w = 0.0f32;
    while w < 1.0 + 1e-6 {
        let rounded = (w * 1000.0).round() / 1000.0;
        weights.push(rounded.clamp(0.0, 1.0));
        w += step;
    }
    if (weights.last().copied().unwrap_or(0.0) - 1.0).abs() > 1e-6 {
        weights.push(1.0);
    }
    weights
}

fn refine_weights_from_sign_change(run_records: &[E4RunRecord], bin_width: f32) -> Vec<f32> {
    let mut weight_means: Vec<(f32, f32)> = Vec::new();
    let mut map: std::collections::HashMap<i32, Vec<f32>> = std::collections::HashMap::new();
    for record in run_records {
        if (record.bin_width - bin_width).abs() > 1e-6 {
            continue;
        }
        let key = float_key(record.mirror_weight);
        map.entry(key).or_default().push(record.delta);
    }
    for (key, deltas) in map {
        if deltas.is_empty() {
            continue;
        }
        let mean = deltas.iter().copied().sum::<f32>() / deltas.len() as f32;
        weight_means.push((key as f32 / 1000.0, mean));
    }
    weight_means.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut refined = Vec::new();
    for pair in weight_means.windows(2) {
        let (w0, d0) = pair[0];
        let (w1, d1) = pair[1];
        if d0 == 0.0 || d1 == 0.0 || d0.signum() != d1.signum() {
            let mut w = w0;
            while w < w1 - 1e-6 {
                let rounded = (w * 1000.0).round() / 1000.0;
                refined.push(rounded);
                w += E4_WEIGHT_FINE_STEP;
            }
            refined.push(w1);
        }
    }
    refined
}

fn float_key(value: f32) -> i32 {
    (value * 1000.0).round() as i32
}

fn format_float_token(value: f32) -> String {
    let s = format!("{value:.2}");
    s.replace('.', "p")
}

fn semitone_fold_to_octave(semitones: f32) -> f32 {
    let eps = 1e-6f32;
    let mut folded = semitones.rem_euclid(12.0);
    if folded < eps && semitones.abs() > eps {
        folded = 12.0;
    }
    folded
}

fn collect_e4_semitone_samples(samples: &E4TailSamples, anchor_hz: f32) -> Vec<f32> {
    let mut out = Vec::new();
    if !anchor_hz.is_finite() || anchor_hz <= 0.0 {
        return out;
    }
    for freqs in &samples.freqs_by_step {
        for &freq in freqs {
            if !freq.is_finite() || freq <= 0.0 {
                continue;
            }
            let ratio = freq / anchor_hz;
            if !ratio.is_finite() || ratio <= 0.0 {
                continue;
            }
            let semitones = 12.0 * ratio.log2();
            out.push(semitone_fold_to_octave(semitones));
        }
    }
    out
}

fn histogram_from_samples(samples: &[f32], min: f32, max: f32, bin_width: f32) -> Histogram {
    let counts = histogram_counts(samples, min, max, bin_width);
    let total: usize = counts.iter().map(|(_, count)| *count).sum();
    let total = total.max(1) as f32;
    let mut centers = Vec::with_capacity(counts.len());
    let mut masses = Vec::with_capacity(counts.len());
    for (bin_start, count) in counts {
        centers.push(bin_start + 0.5 * bin_width);
        masses.push(count as f32 / total);
    }
    Histogram {
        bin_centers: centers,
        masses,
    }
}

fn mass_around(hist: &Histogram, center: f32, eps: f32) -> f32 {
    let mut sum = 0.0f32;
    for (bin_center, mass) in hist.bin_centers.iter().zip(hist.masses.iter()) {
        if (*bin_center - center).abs() <= eps {
            sum += *mass;
        }
    }
    sum
}

fn run_e4_sweep_for_weights(
    out_dir: &Path,
    anchor_hz: f32,
    weights: &[f32],
    seeds: &[u64],
    bin_width: f32,
) -> Result<(Vec<E4RunRecord>, Vec<E4HistRecord>), Box<dyn Error>> {
    let mut runs = Vec::new();
    let mut hists = Vec::new();
    let eps = bin_width.max(1e-6);
    for &weight in weights {
        for &seed in seeds {
            let samples = run_e4_condition_tail_samples(weight, seed, E4_TAIL_WINDOW_STEPS);
            let semitone_samples = collect_e4_semitone_samples(&samples, anchor_hz);
            let histogram = histogram_from_samples(&semitone_samples, 0.0, 12.0, bin_width);

            let major_score = mass_around(&histogram, 4.0, eps) + mass_around(&histogram, 7.0, eps);
            let minor_score = mass_around(&histogram, 3.0, eps) + mass_around(&histogram, 7.0, eps);
            let delta = major_score - minor_score;
            let major_frac = major_score / (major_score + minor_score + 1e-12);

            let burn_in = samples.steps_total.saturating_sub(samples.tail_window);
            let record = E4RunRecord {
                mirror_weight: weight,
                seed,
                bin_width,
                eps,
                major_score,
                minor_score,
                delta,
                major_frac,
                steps_total: samples.steps_total,
                burn_in,
                tail_window: samples.tail_window,
                histogram_source: "tail_mean",
            };
            runs.push(record);
            hists.push(E4HistRecord {
                mirror_weight: weight,
                bin_width,
                histogram,
            });

            let w_token = format_float_token(weight);
            let bw_token = format_float_token(bin_width);
            let hist_csv_path =
                out_dir.join(format!("e4_hist_w{w_token}_seed{seed}_bw{bw_token}.csv"));
            write(
                &hist_csv_path,
                e4_hist_csv(
                    weight,
                    seed,
                    bin_width,
                    samples.steps_total,
                    burn_in,
                    samples.tail_window,
                    "tail_mean",
                    &hists.last().unwrap().histogram,
                ),
            )?;

            let hist_png_path =
                out_dir.join(format!("e4_hist_w{w_token}_seed{seed}_bw{bw_token}.png"));
            let caption =
                format!("E4 Interval Histogram (w={weight:.2}, seed={seed}, bw={bin_width:.2})");
            render_interval_histogram(
                &hist_png_path,
                &caption,
                &semitone_samples,
                0.0,
                12.0,
                bin_width,
            )?;
        }
    }
    Ok((runs, hists))
}

fn e4_runs_csv(records: &[E4RunRecord]) -> String {
    let mut out = String::from(
        "mirror_weight,seed,bin_width,eps,major_score,minor_score,delta,major_frac,steps_total,burn_in,tail_window,histogram_source\n",
    );
    for record in records {
        out.push_str(&format!(
            "{:.3},{},{:.3},{:.3},{:.6},{:.6},{:.6},{:.6},{},{},{},{}\n",
            record.mirror_weight,
            record.seed,
            record.bin_width,
            record.eps,
            record.major_score,
            record.minor_score,
            record.delta,
            record.major_frac,
            record.steps_total,
            record.burn_in,
            record.tail_window,
            record.histogram_source
        ));
    }
    out
}

fn e4_summary_csv(records: &[E4SummaryRecord]) -> String {
    let mut out = String::from(
        "mirror_weight,bin_width,eps,mean_major,std_major,mean_minor,std_minor,mean_delta,std_delta,major_rate,minor_rate,ambiguous_rate,n_runs\n",
    );
    for record in records {
        out.push_str(&format!(
            "{:.3},{:.3},{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3},{:.3},{:.3},{}\n",
            record.mirror_weight,
            record.bin_width,
            record.eps,
            record.mean_major,
            record.std_major,
            record.mean_minor,
            record.std_minor,
            record.mean_delta,
            record.std_delta,
            record.major_rate,
            record.minor_rate,
            record.ambiguous_rate,
            record.n_runs
        ));
    }
    out
}

fn e4_hist_csv(
    weight: f32,
    seed: u64,
    bin_width: f32,
    steps_total: u32,
    burn_in: u32,
    tail_window: u32,
    histogram_source: &str,
    hist: &Histogram,
) -> String {
    let mut out = String::from(
        "mirror_weight,seed,bin_width,steps_total,burn_in,tail_window,histogram_source,bin_center,mass\n",
    );
    for (center, mass) in hist.bin_centers.iter().zip(hist.masses.iter()) {
        out.push_str(&format!(
            "{:.3},{},{:.3},{},{},{},{},{:.3},{:.6}\n",
            weight,
            seed,
            bin_width,
            steps_total,
            burn_in,
            tail_window,
            histogram_source,
            center,
            mass
        ));
    }
    out
}

fn summarize_e4_runs(records: &[E4RunRecord]) -> Vec<E4SummaryRecord> {
    let mut map: std::collections::HashMap<(i32, i32), Vec<&E4RunRecord>> =
        std::collections::HashMap::new();
    for record in records {
        let key = (float_key(record.mirror_weight), float_key(record.bin_width));
        map.entry(key).or_default().push(record);
    }
    let mut summaries = Vec::new();
    for ((_w_key, _bw_key), runs) in map {
        if runs.is_empty() {
            continue;
        }
        let mirror_weight = runs[0].mirror_weight;
        let bin_width = runs[0].bin_width;
        let eps = runs[0].eps;
        let major_values: Vec<f32> = runs.iter().map(|r| r.major_score).collect();
        let minor_values: Vec<f32> = runs.iter().map(|r| r.minor_score).collect();
        let delta_values: Vec<f32> = runs.iter().map(|r| r.delta).collect();
        let (mean_major, std_major) = mean_std(&major_values);
        let (mean_minor, std_minor) = mean_std(&minor_values);
        let (mean_delta, std_delta) = mean_std(&delta_values);
        let mut major_count = 0usize;
        let mut minor_count = 0usize;
        let mut ambiguous_count = 0usize;
        for delta in &delta_values {
            if *delta > E4_DELTA_TAU {
                major_count += 1;
            } else if *delta < -E4_DELTA_TAU {
                minor_count += 1;
            } else {
                ambiguous_count += 1;
            }
        }
        let n_runs = runs.len();
        let inv = 1.0 / n_runs as f32;
        summaries.push(E4SummaryRecord {
            mirror_weight,
            bin_width,
            eps,
            mean_major,
            std_major,
            mean_minor,
            std_minor,
            mean_delta,
            std_delta,
            major_rate: major_count as f32 * inv,
            minor_rate: minor_count as f32 * inv,
            ambiguous_rate: ambiguous_count as f32 * inv,
            n_runs,
        });
    }
    summaries.sort_by(|a, b| {
        a.bin_width
            .partial_cmp(&b.bin_width)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.mirror_weight
                    .partial_cmp(&b.mirror_weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    summaries
}

fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
    let var = values.iter().map(|v| (*v - mean).powi(2)).sum::<f32>() / values.len() as f32;
    (mean, var.sqrt())
}

fn mean_histogram_for_weight(
    hist_records: &[E4HistRecord],
    weight: f32,
    bin_width: f32,
) -> Option<Histogram> {
    let mut selected: Vec<&Histogram> = Vec::new();
    for record in hist_records {
        if (record.bin_width - bin_width).abs() > 1e-6 {
            continue;
        }
        if (record.mirror_weight - weight).abs() > 1e-6 {
            continue;
        }
        selected.push(&record.histogram);
    }
    if selected.is_empty() {
        return None;
    }
    let n_bins = selected[0].masses.len();
    let mut sums = vec![0.0f32; n_bins];
    for hist in &selected {
        for (i, mass) in hist.masses.iter().enumerate() {
            sums[i] += *mass;
        }
    }
    let inv = 1.0 / selected.len() as f32;
    let masses: Vec<f32> = sums.iter().map(|v| v * inv).collect();
    Some(Histogram {
        bin_centers: selected[0].bin_centers.clone(),
        masses,
    })
}

fn render_e4_hist_overlay(
    out_path: &Path,
    hist_records: &[E4HistRecord],
    bin_width: f32,
    weights: &[f32],
) -> Result<(), Box<dyn Error>> {
    let mut series: Vec<(f32, Histogram)> = Vec::new();
    for &weight in weights {
        if let Some(hist) = mean_histogram_for_weight(hist_records, weight, bin_width) {
            series.push((weight, hist));
        }
    }
    if series.is_empty() {
        return Ok(());
    }
    let mut y_max = 0.0f32;
    for (_, hist) in &series {
        let local_max = hist.masses.iter().copied().fold(0.0f32, f32::max);
        y_max = y_max.max(local_max);
    }
    if y_max <= 0.0 {
        y_max = 1.0;
    }
    let root = BitMapBackend::new(out_path, (1400, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("E4 Interval Histograms (Overlay)", ("sans-serif", 22))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..12.0f32, 0.0f32..(y_max * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("semitones (mod 12)")
        .y_desc("mass")
        .x_labels(13)
        .draw()?;

    for &x in &[3.0f32, 4.0, 7.0] {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, 0.0), (x, y_max * 1.1)],
            BLACK.mix(0.15),
        )))?;
    }

    for (i, (weight, hist)) in series.iter().enumerate() {
        let color = Palette99::pick(i).mix(0.9);
        let points = hist
            .bin_centers
            .iter()
            .copied()
            .zip(hist.masses.iter().copied())
            .collect::<Vec<(f32, f32)>>();
        chart
            .draw_series(LineSeries::new(points, &color))?
            .label(format!("w={weight:.2}"))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn render_e4_delta_plot(
    out_path: &Path,
    summaries: &[E4SummaryRecord],
    bin_width: f32,
) -> Result<(), Box<dyn Error>> {
    let mut series: Vec<(f32, f32, f32)> = summaries
        .iter()
        .filter(|s| (s.bin_width - bin_width).abs() < 1e-6)
        .map(|s| (s.mirror_weight, s.mean_delta, s.std_delta))
        .collect();
    if series.is_empty() {
        return Ok(());
    }
    series.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for (_, mean, std) in &series {
        y_min = y_min.min(mean - std);
        y_max = y_max.max(mean + std);
    }
    if !y_min.is_finite() || !y_max.is_finite() || (y_max - y_min).abs() < 1e-6 {
        y_min = -0.1;
        y_max = 0.1;
    }
    let pad = 0.1 * (y_max - y_min);
    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("E4 Δ (Major-Minor) vs Mirror Weight (bw={bin_width:.2})"),
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, (y_min - pad)..(y_max + pad))?;

    chart
        .configure_mesh()
        .x_desc("mirror weight")
        .y_desc("Δ (major - minor)")
        .draw()?;

    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0.0, 0.0), (1.0, 0.0)],
        BLACK.mix(0.3),
    )))?;

    let cap = 0.01f32;
    for (w, mean, std) in &series {
        let y0 = mean - std;
        let y1 = mean + std;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w, y0), (*w, y1)],
            BLUE.mix(0.8),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w - cap, y0), (*w + cap, y0)],
            BLUE.mix(0.8),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w - cap, y1), (*w + cap, y1)],
            BLUE.mix(0.8),
        )))?;
        chart.draw_series(std::iter::once(Circle::new((*w, *mean), 3, BLUE.filled())))?;
    }

    root.present()?;
    Ok(())
}

fn render_e4_major_minor_plot(
    out_path: &Path,
    summaries: &[E4SummaryRecord],
    bin_width: f32,
) -> Result<(), Box<dyn Error>> {
    let mut series: Vec<(f32, f32, f32, f32, f32)> = summaries
        .iter()
        .filter(|s| (s.bin_width - bin_width).abs() < 1e-6)
        .map(|s| {
            (
                s.mirror_weight,
                s.mean_major,
                s.std_major,
                s.mean_minor,
                s.std_minor,
            )
        })
        .collect();
    if series.is_empty() {
        return Ok(());
    }
    series.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut y_max = series
        .iter()
        .map(|(_, m_major, s_major, m_minor, s_minor)| (m_major + s_major).max(m_minor + s_minor))
        .fold(0.0f32, f32::max);
    if y_max <= 0.0 {
        y_max = 1.0;
    }
    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("E4 Major/Minor Scores vs Mirror Weight (bw={bin_width:.2})"),
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, 0.0f32..(y_max * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("mirror weight")
        .y_desc("score")
        .draw()?;

    let cap = 0.01f32;
    for (w, mean_major, std_major, mean_minor, std_minor) in &series {
        let y0 = mean_major - std_major;
        let y1 = mean_major + std_major;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w, y0), (*w, y1)],
            BLUE.mix(0.8),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w - cap, y0), (*w + cap, y0)],
            BLUE.mix(0.8),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w - cap, y1), (*w + cap, y1)],
            BLUE.mix(0.8),
        )))?;
        chart.draw_series(std::iter::once(Circle::new(
            (*w, *mean_major),
            3,
            BLUE.filled(),
        )))?;

        let y0 = mean_minor - std_minor;
        let y1 = mean_minor + std_minor;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w, y0), (*w, y1)],
            RED.mix(0.8),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w - cap, y0), (*w + cap, y0)],
            RED.mix(0.8),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w - cap, y1), (*w + cap, y1)],
            RED.mix(0.8),
        )))?;
        chart.draw_series(std::iter::once(Circle::new(
            (*w, *mean_minor),
            3,
            RED.filled(),
        )))?;
    }

    root.present()?;
    Ok(())
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
    moved_frac: f32,
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
            moved_frac: 0.0,
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
    let mut moved_count = 0usize;
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
        if best_idx != current_idx {
            moved_count += 1;
        }
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
            moved_frac: 0.0,
        };
    }
    let inv = 1.0 / count as f32;
    UpdateStats {
        mean_score: score_sum * inv,
        mean_repulsion: repulsion_sum * inv,
        moved_frac: moved_count as f32 * inv,
    }
}

fn score_stats_at_indices(
    indices: &[usize],
    c01_scan: &[f32],
    log2_ratio_scan: &[f32],
    lambda: f32,
    sigma: f32,
) -> UpdateStats {
    if indices.is_empty() {
        return UpdateStats {
            mean_score: 0.0,
            mean_repulsion: 0.0,
            moved_frac: 0.0,
        };
    }
    let sigma = sigma.max(1e-6);
    let log2_vals: Vec<f32> = indices.iter().map(|&idx| log2_ratio_scan[idx]).collect();
    let mut score_sum = 0.0f32;
    let mut repulsion_sum = 0.0f32;
    for (i, &idx) in indices.iter().enumerate() {
        let cand_log2 = log2_ratio_scan[idx];
        let mut repulsion = 0.0f32;
        for (j, &other_log2) in log2_vals.iter().enumerate() {
            if i == j {
                continue;
            }
            let dist = (cand_log2 - other_log2).abs();
            repulsion += (-dist / sigma).exp();
        }
        let score = c01_scan[idx] - lambda * repulsion;
        if score.is_finite() {
            score_sum += score;
        }
        if repulsion.is_finite() {
            repulsion_sum += repulsion;
        }
    }
    let inv = 1.0 / indices.len() as f32;
    UpdateStats {
        mean_score: score_sum * inv,
        mean_repulsion: repulsion_sum * inv,
        moved_frac: 0.0,
    }
}

fn mean_at_indices(values: &[f32], indices: &[usize]) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }
    let sum: f32 = indices.iter().map(|&idx| values[idx]).sum();
    sum / indices.len() as f32
}

fn mean_std_series(series_list: Vec<&Vec<f32>>) -> (Vec<f32>, Vec<f32>) {
    if series_list.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let len = series_list[0].len();
    let mut sum = vec![0.0f32; len];
    let mut sum_sq = vec![0.0f32; len];
    for series in &series_list {
        debug_assert_eq!(series.len(), len, "series length mismatch");
        for (i, &val) in series.iter().enumerate() {
            sum[i] += val;
            sum_sq[i] += val * val;
        }
    }
    let n = series_list.len() as f32;
    let mut mean = vec![0.0f32; len];
    let mut std = vec![0.0f32; len];
    for i in 0..len {
        mean[i] = sum[i] / n;
        let var = (sum_sq[i] / n) - mean[i] * mean[i];
        std[i] = var.max(0.0).sqrt();
    }
    (mean, std)
}

fn series_pairs(series: &[f32]) -> Vec<(f32, f32)> {
    series
        .iter()
        .enumerate()
        .map(|(i, &y)| (i as f32, y))
        .collect()
}

fn series_csv(header: &str, series: &[f32]) -> String {
    let mut out = String::from(header);
    out.push('\n');
    for (i, value) in series.iter().enumerate() {
        out.push_str(&format!("{i},{value:.6}\n"));
    }
    out
}

fn sweep_csv(header: &str, mean: &[f32], std: &[f32], n: usize) -> String {
    let mut out = String::from(header);
    out.push('\n');
    let len = mean.len().min(std.len());
    for i in 0..len {
        out.push_str(&format!("{i},{:.6},{:.6},{}\n", mean[i], std[i], n));
    }
    out
}

fn e2_controls_csv(baseline: &E2SweepStats, nohill: &E2SweepStats, norep: &E2SweepStats) -> String {
    let mut out = String::from(
        "step,baseline_mean,baseline_std,nohill_mean,nohill_std,norep_mean,norep_std\n",
    );
    let len = baseline
        .mean_c01
        .len()
        .min(nohill.mean_c01.len())
        .min(norep.mean_c01.len());
    for i in 0..len {
        out.push_str(&format!(
            "{i},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            baseline.mean_c01[i],
            baseline.std_c01[i],
            nohill.mean_c01[i],
            nohill.std_c01[i],
            norep.mean_c01[i],
            norep.std_c01[i]
        ));
    }
    out
}

fn trajectories_csv(run: &E2Run) -> String {
    let mut out = String::from("step,agent_id,semitones,c01\n");
    for (agent_id, semis) in run.trajectory_semitones.iter().enumerate() {
        let c01 = &run.trajectory_c01[agent_id];
        let len = semis.len().min(c01.len());
        for step in 0..len {
            out.push_str(&format!(
                "{step},{agent_id},{:.6},{:.6}\n",
                semis[step], c01[step]
            ));
        }
    }
    out
}

fn anchor_shift_csv(run: &E2Run) -> String {
    let s = &run.anchor_shift;
    let mut out =
        String::from("step,anchor_hz_before,anchor_hz_after,count_min,count_max,respawned\n");
    out.push_str(&format!(
        "{},{:.3},{:.3},{},{},{}\n",
        s.step, s.anchor_hz_before, s.anchor_hz_after, s.count_min, s.count_max, s.respawned
    ));
    out
}

fn final_agents_csv(run: &E2Run) -> String {
    let mut out = String::from("agent_id,freq_hz,log2_ratio,semitones\n");
    let len = run
        .final_freqs_hz
        .len()
        .min(run.final_log2_ratios.len())
        .min(run.final_semitones.len());
    for i in 0..len {
        out.push_str(&format!(
            "{},{:.4},{:.6},{:.6}\n",
            i, run.final_freqs_hz[i], run.final_log2_ratios[i], run.final_semitones[i]
        ));
    }
    out
}

fn e2_meta_text(n_agents: usize, k_bins: i32) -> String {
    let mut out = String::new();
    out.push_str(&format!("SPACE_BINS_PER_OCT={}\n", SPACE_BINS_PER_OCT));
    out.push_str(&format!("E2_STEPS={}\n", E2_STEPS));
    out.push_str(&format!("E2_BURN_IN={}\n", E2_BURN_IN));
    out.push_str(&format!("E2_ANCHOR_SHIFT_STEP={}\n", E2_ANCHOR_SHIFT_STEP));
    out.push_str(&format!(
        "E2_ANCHOR_SHIFT_RATIO={:.3}\n",
        E2_ANCHOR_SHIFT_RATIO
    ));
    out.push_str(&format!("E2_STEP_SEMITONES={:.3}\n", E2_STEP_SEMITONES));
    out.push_str(&format!("E2_N_AGENTS={}\n", n_agents));
    out.push_str(&format!("E2_K_BINS={}\n", k_bins));
    out.push_str(&format!("E2_LAMBDA={:.3}\n", E2_LAMBDA));
    out.push_str(&format!("E2_SIGMA={:.3}\n", E2_SIGMA));
    out.push_str(&format!("E2_ANCHOR_BIN_ST={:.3}\n", E2_ANCHOR_BIN_ST));
    out.push_str(&format!("E2_PAIRWISE_BIN_ST={:.3}\n", E2_PAIRWISE_BIN_ST));
    out.push_str(&format!("E2_SEEDS={:?}\n", E2_SEEDS));
    out
}

fn e2_marker_steps() -> Vec<f32> {
    let mut steps = vec![E2_BURN_IN as f32, E2_ANCHOR_SHIFT_STEP as f32];
    steps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    steps.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    steps
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

fn render_series_plot_with_markers(
    out_path: &Path,
    caption: &str,
    y_desc: &str,
    series: &[(f32, f32)],
    markers: &[f32],
) -> Result<(), Box<dyn Error>> {
    if series.is_empty() {
        return Ok(());
    }
    let x_max = series.last().map(|(x, _)| *x).unwrap_or(0.0).max(1.0);
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

    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..x_max, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc(y_desc)
        .draw()?;

    draw_vertical_guides(&mut chart, markers, y_lo, y_hi)?;
    chart.draw_series(LineSeries::new(series.iter().copied(), &BLUE))?;
    root.present()?;
    Ok(())
}

fn render_series_plot_with_band(
    out_path: &Path,
    caption: &str,
    y_desc: &str,
    mean: &[f32],
    std: &[f32],
    markers: &[f32],
) -> Result<(), Box<dyn Error>> {
    if mean.is_empty() {
        return Ok(());
    }
    let len = mean.len().min(std.len());
    let x_max = (len.saturating_sub(1) as f32).max(1.0);

    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for i in 0..len {
        let lo = mean[i] - std[i];
        let hi = mean[i] + std[i];
        if lo.is_finite() && hi.is_finite() {
            y_min = y_min.min(lo);
            y_max = y_max.max(hi);
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

    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..x_max, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc(y_desc)
        .draw()?;

    let mut band_points: Vec<(f32, f32)> = Vec::with_capacity(len * 2);
    for i in 0..len {
        let x = i as f32;
        band_points.push((x, mean[i] + std[i]));
    }
    for i in (0..len).rev() {
        let x = i as f32;
        band_points.push((x, mean[i] - std[i]));
    }
    chart.draw_series(std::iter::once(Polygon::new(
        band_points,
        BLUE.mix(0.2).filled(),
    )))?;

    draw_vertical_guides(&mut chart, markers, y_lo, y_hi)?;
    let line = mean.iter().enumerate().map(|(i, &y)| (i as f32, y));
    chart.draw_series(LineSeries::new(line, &BLUE))?;
    root.present()?;
    Ok(())
}

fn render_series_plot_multi(
    out_path: &Path,
    caption: &str,
    y_desc: &str,
    series_list: &[(&str, &[f32], RGBColor)],
    markers: &[f32],
) -> Result<(), Box<dyn Error>> {
    if series_list.is_empty() {
        return Ok(());
    }
    let mut x_max = 1.0f32;
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for (_, series, _) in series_list {
        if series.is_empty() {
            continue;
        }
        x_max = x_max.max(series.len().saturating_sub(1) as f32);
        for &val in *series {
            if val.is_finite() {
                y_min = y_min.min(val);
                y_max = y_max.max(val);
            }
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

    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
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

    draw_vertical_guides(&mut chart, markers, y_lo, y_hi)?;

    for &(label, series, color) in series_list {
        if series.is_empty() {
            continue;
        }
        let line = series.iter().enumerate().map(|(i, &y)| (i as f32, y));
        chart
            .draw_series(LineSeries::new(line, color))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

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

fn render_e2_histogram_sweep(out_dir: &Path, run: &E2Run) -> Result<(), Box<dyn Error>> {
    let pre_start = E2_BURN_IN;
    let pre_end = E2_ANCHOR_SHIFT_STEP.saturating_sub(1);
    let post_start = E2_ANCHOR_SHIFT_STEP;
    let post_end = E2_STEPS.saturating_sub(1);
    let ranges = [
        ("pre", pre_start, pre_end, &run.semitone_samples_pre),
        ("post", post_start, post_end, &run.semitone_samples_post),
    ];
    let bins = [0.5f32, 0.25f32];
    for (label, start, end, values) in ranges {
        for &bin_width in &bins {
            let fname = format!(
                "paper_e2_interval_histogram_{}_bw{}.png",
                label,
                format_float_token(bin_width)
            );
            let caption = format!(
                "E2 Interval Histogram ({label}, steps {start}-{end}, bin={bin_width:.2}st)"
            );
            let out_path = out_dir.join(fname);
            render_interval_histogram(&out_path, &caption, values, -12.0, 12.0, bin_width)?;
        }
    }
    Ok(())
}

fn render_e2_control_histograms(
    out_dir: &Path,
    baseline: &E2Run,
    nohill: &E2Run,
    norep: &E2Run,
) -> Result<(), Box<dyn Error>> {
    let bin_width = E2_ANCHOR_BIN_ST;
    let min = -12.0f32;
    let max = 12.0f32;
    let counts_base = histogram_counts(&baseline.semitone_samples_post, min, max, bin_width);
    let counts_nohill = histogram_counts(&nohill.semitone_samples_post, min, max, bin_width);
    let counts_norep = histogram_counts(&norep.semitone_samples_post, min, max, bin_width);

    let mut y_max = 0.0f32;
    for counts in [&counts_base, &counts_nohill, &counts_norep] {
        for &(_, count) in counts.iter() {
            y_max = y_max.max(count as f32);
        }
    }
    y_max = y_max.max(1.0);

    let out_path = out_dir.join("paper_e2_interval_histogram_post_controls_bw0p50.png");
    let root = BitMapBackend::new(&out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "E2 Interval Histogram (post, controls, bin=0.50st)",
            ("sans-serif", 20),
        )
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

    let counts = [&counts_base, &counts_nohill, &counts_norep];
    let colors = [BLUE.mix(0.6), RED.mix(0.6), GREEN.mix(0.6)];
    let sub_width = bin_width / 3.0;
    for bin_idx in 0..counts_base.len() {
        let bin_start = counts_base[bin_idx].0;
        for (j, counts_set) in counts.iter().enumerate() {
            let count = counts_set[bin_idx].1 as f32;
            let x0 = bin_start + sub_width * j as f32;
            let x1 = x0 + sub_width;
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x0, 0.0), (x1, count)],
                colors[j].filled(),
            )))?;
        }
    }

    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(min, y_max * 1.05), (min + 0.3, y_max * 1.05)],
            BLUE,
        )))?
        .label("baseline")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(min, y_max * 1.02), (min + 0.3, y_max * 1.02)],
            RED,
        )))?
        .label("no hill-climb")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(min, y_max * 0.99), (min + 0.3, y_max * 0.99)],
            GREEN,
        )))?
        .label("no repulsion")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

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

fn render_e5_order_plot(
    out_path: &Path,
    main_series: &[(f32, f32, f32, f32, f32, f32)],
    ctrl_series: &[(f32, f32, f32, f32, f32, f32)],
) -> Result<(), Box<dyn Error>> {
    let x_max = main_series
        .last()
        .map(|(x, _, _, _, _, _)| *x)
        .unwrap_or(0.0);
    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("E5 Order Parameter r(t)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..x_max.max(1.0), 0.0f32..1.05f32)?;

    chart
        .configure_mesh()
        .x_desc("time (s)")
        .y_desc("r")
        .draw()?;

    let r_main = main_series.iter().map(|(t, r, _, _, _, _)| (*t, *r));
    let r_ctrl = ctrl_series.iter().map(|(t, r, _, _, _, _)| (*t, *r));

    chart
        .draw_series(LineSeries::new(r_main, &BLUE))?
        .label("main r(t)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
    chart
        .draw_series(LineSeries::new(r_ctrl, &RED))?
        .label("control r(t)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    draw_vertical_guides(&mut chart, &e5_marker_times(), 0.0, 1.05)?;

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn render_e5_delta_phi_plot(
    out_path: &Path,
    main_series: &[(f32, f32, f32, f32, f32, f32)],
    ctrl_series: &[(f32, f32, f32, f32, f32, f32)],
) -> Result<(), Box<dyn Error>> {
    let x_max = main_series
        .last()
        .map(|(x, _, _, _, _, _)| *x)
        .unwrap_or(0.0);
    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("E5 Group Phase Offset Δφ(t)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..x_max.max(1.0), -PI..PI)?;

    chart
        .configure_mesh()
        .x_desc("time (s)")
        .y_desc("Δφ (rad)")
        .draw()?;

    let main_points = main_series
        .iter()
        .map(|(t, _, delta_phi, _, _, _)| (*t, *delta_phi));
    let ctrl_points = ctrl_series
        .iter()
        .map(|(t, _, delta_phi, _, _, _)| (*t, *delta_phi));

    chart
        .draw_series(LineSeries::new(main_points, &BLUE))?
        .label("main Δφ(t)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
    chart
        .draw_series(LineSeries::new(ctrl_points, &RED))?
        .label("control Δφ(t)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    draw_vertical_guides(&mut chart, &e5_marker_times(), -PI, PI)?;

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn render_e5_plv_plot(
    out_path: &Path,
    main_series: &[(f32, f32, f32, f32, f32, f32)],
    ctrl_series: &[(f32, f32, f32, f32, f32, f32)],
) -> Result<(), Box<dyn Error>> {
    let x_max = main_series
        .last()
        .map(|(x, _, _, _, _, _)| *x)
        .unwrap_or(0.0);
    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("E5 PLV (Agent-Kick + Group Δφ)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..x_max.max(1.0), 0.0f32..1.05f32)?;

    chart
        .configure_mesh()
        .x_desc("time (s)")
        .y_desc("PLV")
        .draw()?;

    let plv_main = main_series.iter().map(|(t, _, _, plv, _, _)| (*t, *plv));
    let plv_ctrl = ctrl_series.iter().map(|(t, _, _, plv, _, _)| (*t, *plv));
    let group_main = main_series.iter().map(|(t, _, _, _, plv, _)| (*t, *plv));
    let group_ctrl = ctrl_series.iter().map(|(t, _, _, _, plv, _)| (*t, *plv));

    chart
        .draw_series(LineSeries::new(plv_main, &BLUE))?
        .label("main PLV_agent_kick")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
    chart
        .draw_series(LineSeries::new(plv_ctrl, &RED))?
        .label("control PLV_agent_kick")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .draw_series(LineSeries::new(group_main, &BLUE.mix(0.4)))?
        .label("main PLV_group_Δφ")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.mix(0.4)));
    chart
        .draw_series(LineSeries::new(group_ctrl, &RED.mix(0.4)))?
        .label("control PLV_group_Δφ")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.mix(0.4)));

    draw_vertical_guides(&mut chart, &e5_marker_times(), 0.0, 1.05)?;

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn render_phase_histogram_compare(
    out_path: &Path,
    main_samples: &[f32],
    ctrl_samples: &[f32],
    bins: usize,
) -> Result<(), Box<dyn Error>> {
    if bins == 0 {
        return Ok(());
    }
    let min = -PI;
    let max = PI;
    let bin_width = (max - min) / bins as f32;
    let counts_main = histogram_counts(main_samples, min, max, bin_width);
    let counts_ctrl = histogram_counts(ctrl_samples, min, max, bin_width);
    let y_max = counts_main
        .iter()
        .zip(counts_ctrl.iter())
        .map(|((_, c1), (_, c2))| (*c1).max(*c2) as f32)
        .fold(0.0f32, f32::max)
        .max(1.0);

    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "E5 Phase Difference Histogram (Main vs Control)",
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min..max, 0.0f32..(y_max * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("θ_i - θ_kick (rad)")
        .y_desc("count")
        .x_labels(9)
        .draw()?;

    let half = bin_width * 0.45;
    for ((bin_start, count_main), (_, count_ctrl)) in counts_main.iter().zip(counts_ctrl.iter()) {
        let x0 = *bin_start;
        let x1 = x0 + bin_width;
        let mid = 0.5 * (x0 + x1);
        chart.draw_series(std::iter::once(Rectangle::new(
            [(mid - half, 0.0), (mid, *count_main as f32)],
            BLUE.mix(0.6).filled(),
        )))?;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(mid, 0.0), (mid + half, *count_ctrl as f32)],
            RED.mix(0.6).filled(),
        )))?;
    }

    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(0.0, y_max * 1.05), (0.1, y_max * 1.05)],
            BLUE,
        )))?
        .label("main")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(0.0, y_max * 1.02), (0.1, y_max * 1.02)],
            RED,
        )))?
        .label("control")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn e5_marker_times() -> Vec<f32> {
    let mut times = Vec::new();
    let burn_t = E5_BURN_IN_STEPS as f32 * E5_DT;
    let sample_start = E5_STEPS
        .saturating_sub(E5_SAMPLE_WINDOW_STEPS)
        .max(E5_BURN_IN_STEPS);
    let sample_t = sample_start as f32 * E5_DT;
    times.push(burn_t);
    if (sample_t - burn_t).abs() > 1e-6 {
        times.push(sample_t);
    }
    if let Some(kick_on) = E5_KICK_ON_STEP {
        times.push(kick_on as f32 * E5_DT);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    times
}

fn draw_vertical_guides<DB: DrawingBackend>(
    chart: &mut ChartContext<DB, Cartesian2d<RangedCoordf32, RangedCoordf32>>,
    xs: &[f32],
    y_min: f32,
    y_max: f32,
) -> Result<(), Box<dyn Error>>
where
    DB::ErrorType: 'static,
{
    for &x in xs {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, y_min), (x, y_max)],
            BLACK.mix(0.2),
        )))?;
    }
    Ok(())
}

fn e5_kick_csv(main: &E5KickSimResult, ctrl: &E5KickSimResult) -> String {
    let mut out =
        String::from("condition,t,r,delta_phi,plv_agent_kick,plv_group_delta_phi,k_eff\n");
    for (label, sim) in [("main", main), ("control", ctrl)] {
        for (t, r, delta_phi, plv_agent, plv_group, k_eff) in &sim.series {
            out.push_str(&format!(
                "{label},{t:.4},{r:.6},{delta_phi:.6},{plv_agent:.6},{plv_group:.6},{k_eff:.4}\n"
            ));
        }
    }
    out
}

fn e5_post_stats(series: &[(f32, f32, f32, f32, f32, f32)], window: usize) -> (f32, f32, f32) {
    if series.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let window = window.clamp(1, series.len());
    let start = series.len().saturating_sub(window);
    let mut mean_plv = 0.0f32;
    let mut mean_group = 0.0f32;
    for (_, _, _, plv_agent, plv_group, _) in series.iter().skip(start) {
        mean_plv += *plv_agent;
        mean_group += *plv_group;
    }
    let inv = 1.0 / window as f32;
    mean_plv *= inv;
    mean_group *= inv;
    let mut var = 0.0f32;
    for (_, _, _, plv_agent, _, _) in series.iter().skip(start) {
        var += (*plv_agent - mean_plv).powi(2);
    }
    let std = (var * inv).sqrt();
    (mean_plv, std, mean_group)
}

fn e5_kick_summary_csv(main: &E5KickSimResult, ctrl: &E5KickSimResult) -> String {
    let mut out =
        String::from("condition,plv_post_mean,plv_post_std,delta_phi_post_plv,plv_time\n");
    let (mean_main, std_main, group_main) = e5_post_stats(&main.series, E5_SAMPLE_WINDOW_STEPS);
    let (mean_ctrl, std_ctrl, group_ctrl) = e5_post_stats(&ctrl.series, E5_SAMPLE_WINDOW_STEPS);
    out.push_str(&format!(
        "main,{:.6},{:.6},{:.6},{:.6}\n",
        mean_main, std_main, group_main, main.plv_time
    ));
    out.push_str(&format!(
        "control,{:.6},{:.6},{:.6},{:.6}\n",
        mean_ctrl, std_ctrl, group_ctrl, ctrl.plv_time
    ));
    out
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

fn plv_time_from_series(series: &[(f32, f32, f32, f32, f32, f32)], window: usize) -> f32 {
    if series.is_empty() {
        return 0.0;
    }
    let window = window.clamp(1, series.len());
    let start = series.len().saturating_sub(window);
    let mut mean_cos = 0.0f32;
    let mut mean_sin = 0.0f32;
    for (_, _, delta_phi, _, _, _) in series.iter().skip(start) {
        mean_cos += delta_phi.cos();
        mean_sin += delta_phi.sin();
    }
    let inv = 1.0 / window as f32;
    mean_cos *= inv;
    mean_sin *= inv;
    (mean_cos * mean_cos + mean_sin * mean_sin).sqrt()
}

struct SlidingPlv {
    window: usize,
    buf_cos: Vec<f32>,
    buf_sin: Vec<f32>,
    sum_cos: f32,
    sum_sin: f32,
    idx: usize,
    len: usize,
}

impl SlidingPlv {
    fn new(window: usize) -> Self {
        let window = window.max(1);
        Self {
            window,
            buf_cos: vec![0.0; window],
            buf_sin: vec![0.0; window],
            sum_cos: 0.0,
            sum_sin: 0.0,
            idx: 0,
            len: 0,
        }
    }

    fn push(&mut self, angle: f32) {
        let c = angle.cos();
        let s = angle.sin();
        if self.len < self.window {
            self.buf_cos[self.idx] = c;
            self.buf_sin[self.idx] = s;
            self.sum_cos += c;
            self.sum_sin += s;
            self.len += 1;
            self.idx = (self.idx + 1) % self.window;
            return;
        }

        let old_c = self.buf_cos[self.idx];
        let old_s = self.buf_sin[self.idx];
        self.sum_cos -= old_c;
        self.sum_sin -= old_s;
        self.buf_cos[self.idx] = c;
        self.buf_sin[self.idx] = s;
        self.sum_cos += c;
        self.sum_sin += s;
        self.idx = (self.idx + 1) % self.window;
    }

    fn plv(&self) -> f32 {
        if self.len == 0 {
            return 0.0;
        }
        let inv = 1.0 / self.len as f32;
        let mean_cos = self.sum_cos * inv;
        let mean_sin = self.sum_sin * inv;
        (mean_cos * mean_cos + mean_sin * mean_sin).sqrt()
    }
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
        let sim = simulate_e5_kick(E5_SEED, 120, E5_K_KICK, E5_KICK_ON_STEP);
        assert_eq!(sim.series.len(), 120);
        assert_eq!(sim.n_agents, E5_N_AGENTS);
        let last_t = sim.series.last().unwrap().0;
        assert!((last_t - (119.0 * E5_DT)).abs() < 1e-6, "last_t mismatch");
    }

    #[test]
    fn e5_histogram_sample_count_matches_window() {
        let steps = E5_BURN_IN_STEPS + E5_SAMPLE_WINDOW_STEPS + 10;
        let sim = simulate_e5_kick(E5_SEED, steps, E5_K_KICK, E5_KICK_ON_STEP);
        let expected = sim.n_agents * sim.sample_window_steps;
        assert_eq!(sim.phase_hist_samples.len(), expected);
    }

    #[test]
    fn e5_plv_agent_kick_within_unit_range() {
        let sim = simulate_e5_kick(E5_SEED, 200, E5_K_KICK, E5_KICK_ON_STEP);
        for (_, _, _, plv_agent, plv_group, _) in &sim.series {
            assert!(
                *plv_agent >= -1e-6 && *plv_agent <= 1.0 + 1e-6,
                "plv_agent_kick out of range: {plv_agent}"
            );
            assert!(
                *plv_group >= -1e-6 && *plv_group <= 1.0 + 1e-6,
                "plv_group_delta_phi out of range: {plv_group}"
            );
        }
    }

    #[test]
    fn e5_main_plv_exceeds_control() {
        let steps = if let Some(on_step) = E5_KICK_ON_STEP {
            on_step + E5_SAMPLE_WINDOW_STEPS + 50
        } else {
            E5_SAMPLE_WINDOW_STEPS + 200
        };
        let sim_main = simulate_e5_kick(E5_SEED, steps, E5_K_KICK, E5_KICK_ON_STEP);
        let sim_ctrl = simulate_e5_kick(E5_SEED, steps, 0.0, E5_KICK_ON_STEP);
        let (main_mean, _, _) = e5_post_stats(&sim_main.series, E5_SAMPLE_WINDOW_STEPS);
        let (ctrl_mean, _, _) = e5_post_stats(&sim_ctrl.series, E5_SAMPLE_WINDOW_STEPS);
        assert!(
            main_mean > ctrl_mean + 0.2,
            "expected main PLV to exceed control (main={main_mean:.3}, ctrl={ctrl_mean:.3})"
        );
    }

    #[test]
    fn e5_kick_on_step_changes_k_eff() {
        let sim = simulate_e5_kick(E5_SEED, 120, E5_K_KICK, E5_KICK_ON_STEP);
        if let Some(on_step) = E5_KICK_ON_STEP {
            if on_step < sim.series.len() {
                let k_before = sim.series[0].5;
                let k_after = sim.series[on_step].5;
                assert!(k_before.abs() < 1e-6);
                assert!((k_after - E5_K_KICK).abs() < 1e-6);
            }
        }
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
