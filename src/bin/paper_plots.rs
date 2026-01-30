use std::env;
use std::error::Error;
use std::f32::consts::PI;
use std::fs::{create_dir_all, write};
use std::io;
use std::path::Path;

use plotters::coord::types::RangedCoordf32;
use plotters::coord::{CoordTranslate, Shift};
use plotters::prelude::*;

use conchordal::core::erb::hz_to_erb;
use conchordal::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
use conchordal::core::landscape::{LandscapeParams, RoughnessScalarMode};
use conchordal::core::log2space::Log2Space;
use conchordal::core::psycho_state;
use conchordal::core::roughness_kernel::{KernelParams, RoughnessKernel};
use conchordal::paper::sim::{
    E3Condition, E3DeathRecord, E3RunConfig, E4_ANCHOR_HZ, E4TailSamples, e3_policy_params,
    run_e3_collect_deaths, run_e4_condition_tail_samples,
};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng, rngs::StdRng};

const SPACE_BINS_PER_OCT: u32 = 400;

const E2_STEPS: usize = 50;
const E2_BURN_IN: usize = 10;
const E2_ANCHOR_SHIFT_STEP: usize = usize::MAX;
const E2_ANCHOR_SHIFT_RATIO: f32 = 0.5;
const E2_STEP_SEMITONES: f32 = 0.5;
const E2_ANCHOR_BIN_ST: f32 = 0.5;
const E2_PAIRWISE_BIN_ST: f32 = 0.25;
const E2_N_AGENTS: usize = 24;
const E2_LAMBDA: f32 = 0.15;
const E2_SIGMA: f32 = 0.06;
const E2_INIT_CONSONANT_EXCLUSION_ST: f32 = 0.35;
const E2_INIT_MAX_TRIES: usize = 5000;
const E2_SEEDS: [u64; 5] = [
    0xC0FFEE_u64,
    0xC0FFEE_u64 + 1,
    0xC0FFEE_u64 + 2,
    0xC0FFEE_u64 + 3,
    0xC0FFEE_u64 + 4,
];

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
enum E2InitMode {
    Uniform,
    RejectConsonant,
}

impl E2InitMode {
    fn label(self) -> &'static str {
        match self {
            E2InitMode::Uniform => "uniform",
            E2InitMode::RejectConsonant => "reject_consonant",
        }
    }
}

const E2_INIT_MODE: E2InitMode = E2InitMode::RejectConsonant;
const E2_CONSONANT_STEPS: [f32; 8] = [0.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 12.0];

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

const E3_FIRST_K: usize = 20;
const E3_POP_SIZE: usize = 32;
const E3_MIN_DEATHS: usize = 200;
const E3_STEPS_CAP: usize = 6000;
const E3_SEEDS: [u64; 5] = [
    0xC0FFEE_u64 + 30,
    0xC0FFEE_u64 + 31,
    0xC0FFEE_u64 + 32,
    0xC0FFEE_u64 + 33,
    0xC0FFEE_u64 + 34,
];

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
const E5_MIN_R_FOR_GROUP_PHASE: f32 = 0.2;
const E5_KICK_ON_STEP: Option<usize> = Some(800);
const E5_SEED: u64 = 0xC0FFEE_u64 + 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Experiment {
    E1,
    E2,
    E3,
    E4,
    E5,
}

impl Experiment {
    fn label(self) -> &'static str {
        match self {
            Experiment::E1 => "E1",
            Experiment::E2 => "E2",
            Experiment::E3 => "E3",
            Experiment::E4 => "E4",
            Experiment::E5 => "E5",
        }
    }

    fn all() -> Vec<Experiment> {
        vec![
            Experiment::E1,
            Experiment::E2,
            Experiment::E3,
            Experiment::E4,
            Experiment::E5,
        ]
    }
}

fn usage() -> String {
    [
        "Usage: paper_plots [--exp E1,E2,...]",
        "Examples:",
        "  paper_plots --exp 2",
        "  paper_plots 1 3 5",
        "  paper_plots --exp e2,e4",
        "If no experiment is specified, all (E1-E5) run.",
    ]
    .join("\n")
}

fn parse_experiments(args: &[String]) -> Result<Vec<Experiment>, String> {
    let mut values: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "-e" || arg == "--exp" || arg == "--experiment" {
            if i + 1 >= args.len() {
                return Err(format!("Missing value after {arg}\n{}", usage()));
            }
            values.push(args[i + 1].clone());
            i += 2;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--exp=") {
            values.push(rest.to_string());
            i += 1;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--experiment=") {
            values.push(rest.to_string());
            i += 1;
            continue;
        }
        values.push(arg.to_string());
        i += 1;
    }

    if values.is_empty() {
        return Ok(Vec::new());
    }

    let mut experiments: Vec<Experiment> = Vec::new();
    let mut saw_all = false;
    for value in values {
        for token in value.split(',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            if token.eq_ignore_ascii_case("all") {
                saw_all = true;
                continue;
            }
            let exp = match token {
                "1" | "e1" | "E1" => Experiment::E1,
                "2" | "e2" | "E2" => Experiment::E2,
                "3" | "e3" | "E3" => Experiment::E3,
                "4" | "e4" | "E4" => Experiment::E4,
                "5" | "e5" | "E5" => Experiment::E5,
                _ => {
                    return Err(format!("Unknown experiment '{token}'.\n{}", usage()));
                }
            };
            if !experiments.contains(&exp) {
                experiments.push(exp);
            }
        }
    }

    if saw_all {
        return Ok(Experiment::all());
    }

    Ok(experiments)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("{}", usage());
        return Ok(());
    }
    let experiments = parse_experiments(&args).map_err(io::Error::other)?;
    let experiments = if experiments.is_empty() {
        Experiment::all()
    } else {
        experiments
    };

    let out_dir = Path::new("target/plots/paper");
    create_dir_all(out_dir)?;

    let anchor_hz = E4_ANCHOR_HZ;
    let space = Log2Space::new(20.0, 8000.0, SPACE_BINS_PER_OCT);

    std::thread::scope(|s| -> Result<(), Box<dyn Error>> {
        let mut handles = Vec::new();
        for exp in experiments {
            match exp {
                Experiment::E1 => {
                    let h = s.spawn(|| {
                        plot_e1_landscape_scan(out_dir, &space, anchor_hz)
                            .map_err(|err| io::Error::other(err.to_string()))
                    });
                    handles.push((exp.label(), h));
                }
                Experiment::E2 => {
                    let h = s.spawn(|| {
                        plot_e2_emergent_harmony(out_dir, &space, anchor_hz)
                            .map_err(|err| io::Error::other(err.to_string()))
                    });
                    handles.push((exp.label(), h));
                }
                Experiment::E3 => {
                    let h = s.spawn(|| {
                        plot_e3_metabolic_selection(out_dir, &space, anchor_hz)
                            .map_err(|err| io::Error::other(err.to_string()))
                    });
                    handles.push((exp.label(), h));
                }
                Experiment::E4 => {
                    let h = s.spawn(|| {
                        plot_e4_mirror_sweep(out_dir, anchor_hz)
                            .map_err(|err| io::Error::other(err.to_string()))
                    });
                    handles.push((exp.label(), h));
                }
                Experiment::E5 => {
                    let h = s.spawn(|| {
                        plot_e5_rhythmic_entrainment(out_dir)
                            .map_err(|err| io::Error::other(err.to_string()))
                    });
                    handles.push((exp.label(), h));
                }
            }
        }

        let mut first_err: Option<io::Error> = None;
        for (label, handle) in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    if first_err.is_none() {
                        first_err = Some(err);
                    }
                }
                Err(_) => {
                    if first_err.is_none() {
                        first_err = Some(io::Error::other(format!("{label} thread panicked")));
                    }
                }
            }
        }
        if let Some(err) = first_err {
            return Err(Box::new(err));
        }
        Ok(())
    })?;

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
    let (baseline_runs, baseline_stats) = e2_seed_sweep(space, anchor_hz, E2Condition::Baseline);
    let rep_index = pick_representative_run_index(&baseline_runs);
    let baseline_run = &baseline_runs[rep_index];
    let marker_steps = e2_marker_steps();
    let caption_suffix = e2_caption_suffix();
    let post_label = e2_post_label();
    let post_label_title = e2_post_label_title();

    write(
        out_dir.join("paper_e2_representative_seed.txt"),
        representative_seed_text(&baseline_runs, rep_index),
    )?;

    write(
        out_dir.join("paper_e2_meta.txt"),
        e2_meta_text(baseline_run.n_agents, baseline_run.k_bins),
    )?;

    write(
        out_dir.join("paper_e2_timeseries.csv"),
        series_csv("step,mean_c01", &baseline_run.mean_c01_series),
    )?;
    write(
        out_dir.join("paper_e2_score_timeseries.csv"),
        series_csv("step,mean_score", &baseline_run.mean_score_series),
    )?;
    write(
        out_dir.join("paper_e2_repulsion_timeseries.csv"),
        series_csv("step,mean_repulsion", &baseline_run.mean_repulsion_series),
    )?;
    write(
        out_dir.join("paper_e2_moved_frac_timeseries.csv"),
        series_csv("step,moved_frac", &baseline_run.moved_frac_series),
    )?;

    write(
        out_dir.join("paper_e2_agent_trajectories.csv"),
        trajectories_csv(baseline_run),
    )?;
    if e2_anchor_shift_enabled() {
        write(
            out_dir.join("paper_e2_anchor_shift_stats.csv"),
            anchor_shift_csv(baseline_run),
        )?;
    }
    write(
        out_dir.join("paper_e2_final_agents.csv"),
        final_agents_csv(baseline_run),
    )?;

    let mean_plot_path = out_dir.join("paper_e2_mean_consonance_over_time.png");
    render_series_plot_fixed_y(
        &mean_plot_path,
        &format!("E2 Mean Consonance Over Time ({caption_suffix})"),
        "mean C01",
        &series_pairs(&baseline_run.mean_c01_series),
        &marker_steps,
        0.0,
        1.0,
    )?;

    let mean_score_path = out_dir.join("paper_e2_mean_score_over_time.png");
    render_series_plot_with_markers(
        &mean_score_path,
        &format!("E2 Mean Score Over Time ({caption_suffix})"),
        "mean score (c01 - λ·repulsion)",
        &series_pairs(&baseline_run.mean_score_series),
        &marker_steps,
    )?;

    let mean_repulsion_path = out_dir.join("paper_e2_mean_repulsion_over_time.png");
    render_series_plot_with_markers(
        &mean_repulsion_path,
        &format!("E2 Mean Repulsion Over Time ({caption_suffix})"),
        "mean repulsion",
        &series_pairs(&baseline_run.mean_repulsion_series),
        &marker_steps,
    )?;

    let moved_frac_path = out_dir.join("paper_e2_moved_frac_over_time.png");
    render_series_plot_with_markers(
        &moved_frac_path,
        &format!("E2 Moved Fraction Over Time ({caption_suffix})"),
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
    let hist_caption = format!("E2 Interval Histogram ({post_label}, bin=0.50st)");
    render_interval_histogram(
        &hist_path,
        &hist_caption,
        &baseline_run.semitone_samples_post,
        -12.0,
        12.0,
        E2_ANCHOR_BIN_ST,
    )?;

    write(
        out_dir.join("paper_e2_summary.csv"),
        e2_summary_csv(&baseline_runs),
    )?;

    render_e2_histogram_sweep(out_dir, baseline_run)?;

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

    let nohill_rep = &nohill_runs[pick_representative_run_index(&nohill_runs)];
    let norep_rep = &norep_runs[pick_representative_run_index(&norep_runs)];
    render_e2_control_histograms(out_dir, baseline_run, nohill_rep, norep_rep)?;

    let hist_min = -12.0f32;
    let hist_max = 12.0f32;
    let hist_stats_05 = e2_hist_seed_sweep(&baseline_runs, 0.5, hist_min, hist_max);
    write(
        out_dir.join("paper_e2_interval_hist_post_seed_sweep_bw0p50.csv"),
        e2_hist_seed_sweep_csv(&hist_stats_05),
    )?;
    let hist_plot_05 = out_dir.join("paper_e2_interval_hist_post_seed_sweep_bw0p50.png");
    render_hist_mean_std(
        &hist_plot_05,
        &format!("E2 {post_label_title} Interval Histogram (seed sweep, bin=0.50st)"),
        &hist_stats_05.centers,
        &hist_stats_05.mean_count,
        &hist_stats_05.std_count,
        0.5,
    )?;

    let hist_stats_025 = e2_hist_seed_sweep(&baseline_runs, 0.25, hist_min, hist_max);
    write(
        out_dir.join("paper_e2_interval_hist_post_seed_sweep_bw0p25.csv"),
        e2_hist_seed_sweep_csv(&hist_stats_025),
    )?;
    let hist_plot_025 = out_dir.join("paper_e2_interval_hist_post_seed_sweep_bw0p25.png");
    render_hist_mean_std(
        &hist_plot_025,
        &format!("E2 {post_label_title} Interval Histogram (seed sweep, bin=0.25st)"),
        &hist_stats_025.centers,
        &hist_stats_025.mean_count,
        &hist_stats_025.std_count,
        0.25,
    )?;

    let nohill_hist_05 = e2_hist_seed_sweep(&nohill_runs, 0.5, hist_min, hist_max);
    let norep_hist_05 = e2_hist_seed_sweep(&norep_runs, 0.5, hist_min, hist_max);
    let mut controls_csv = String::from(
        "bin_center,baseline_mean,baseline_std,nohill_mean,nohill_std,norep_mean,norep_std\n",
    );
    let len = hist_stats_05
        .centers
        .len()
        .min(nohill_hist_05.centers.len())
        .min(norep_hist_05.centers.len());
    for i in 0..len {
        controls_csv.push_str(&format!(
            "{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            hist_stats_05.centers[i],
            hist_stats_05.mean_frac[i],
            hist_stats_05.std_frac[i],
            nohill_hist_05.mean_frac[i],
            nohill_hist_05.std_frac[i],
            norep_hist_05.mean_frac[i],
            norep_hist_05.std_frac[i]
        ));
    }
    write(
        out_dir.join("paper_e2_interval_hist_post_controls_seed_sweep_bw0p50.csv"),
        controls_csv,
    )?;

    let control_hist_plot =
        out_dir.join("paper_e2_interval_hist_post_controls_seed_sweep_bw0p50.png");
    render_hist_controls_fraction(
        &control_hist_plot,
        &format!("E2 {post_label_title} Interval Histogram (controls, mean frac, bin=0.50st)"),
        &hist_stats_05.centers,
        &[
            ("baseline", &hist_stats_05.mean_frac, BLUE),
            ("no hill-climb", &nohill_hist_05.mean_frac, RED),
            ("no repulsion", &norep_hist_05.mean_frac, GREEN),
        ],
    )?;

    let mut delta_csv = String::from("seed,cond,c01_init,c01_pre,c01_post,delta_pre,delta_post\n");
    let mut delta_summary = String::from(
        "cond,mean_init,std_init,mean_pre,std_pre,mean_post,std_post,mean_delta_pre,std_delta_pre,mean_delta_post,std_delta_post\n",
    );
    for (label, runs) in [
        ("baseline", &baseline_runs),
        ("nohill", &nohill_runs),
        ("norep", &norep_runs),
    ] {
        let mut init_vals = Vec::new();
        let mut pre_vals = Vec::new();
        let mut post_vals = Vec::new();
        let mut delta_pre_vals = Vec::new();
        let mut delta_post_vals = Vec::new();
        for run in runs.iter() {
            let (init, pre, post) = e2_c01_snapshot(run);
            let delta_pre = pre - init;
            let delta_post = post - init;
            init_vals.push(init);
            pre_vals.push(pre);
            post_vals.push(post);
            delta_pre_vals.push(delta_pre);
            delta_post_vals.push(delta_post);
            delta_csv.push_str(&format!(
                "{},{label},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                run.seed, init, pre, post, delta_pre, delta_post
            ));
        }
        let (mean_init, std_init) = mean_std_scalar(&init_vals);
        let (mean_pre, std_pre) = mean_std_scalar(&pre_vals);
        let (mean_post, std_post) = mean_std_scalar(&post_vals);
        let (mean_dpre, std_dpre) = mean_std_scalar(&delta_pre_vals);
        let (mean_dpost, std_dpost) = mean_std_scalar(&delta_post_vals);
        delta_summary.push_str(&format!(
            "{label},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            mean_init,
            std_init,
            mean_pre,
            std_pre,
            mean_post,
            std_post,
            mean_dpre,
            std_dpre,
            mean_dpost,
            std_dpost
        ));
    }
    write(out_dir.join("paper_e2_delta_c01_by_seed.csv"), delta_csv)?;
    write(
        out_dir.join("paper_e2_delta_c01_summary.csv"),
        delta_summary,
    )?;

    Ok(())
}

fn e2_anchor_shift_enabled() -> bool {
    E2_ANCHOR_SHIFT_STEP != usize::MAX
}

fn e2_caption_suffix() -> String {
    if e2_anchor_shift_enabled() {
        format!("burn-in={E2_BURN_IN}, shift@{E2_ANCHOR_SHIFT_STEP}")
    } else {
        format!("burn-in={E2_BURN_IN}, shift=off")
    }
}

fn e2_post_label() -> &'static str {
    if e2_anchor_shift_enabled() {
        "post"
    } else {
        "post-burn-in"
    }
}

fn e2_post_label_title() -> &'static str {
    if e2_anchor_shift_enabled() {
        "Post"
    } else {
        "Post-burn-in"
    }
}

fn is_consonant_near(semitone_abs: f32) -> bool {
    for target in E2_CONSONANT_STEPS {
        if (semitone_abs - target).abs() <= E2_INIT_CONSONANT_EXCLUSION_ST {
            return true;
        }
    }
    false
}

fn init_e2_agent_indices_uniform<R: Rng + ?Sized>(
    rng: &mut R,
    min_idx: usize,
    max_idx: usize,
) -> Vec<usize> {
    (0..E2_N_AGENTS)
        .map(|_| rng.random_range(min_idx..=max_idx))
        .collect()
}

fn init_e2_agent_indices_reject_consonant<R: Rng + ?Sized>(
    rng: &mut R,
    min_idx: usize,
    max_idx: usize,
    log2_ratio_scan: &[f32],
) -> Vec<usize> {
    let mut indices = Vec::with_capacity(E2_N_AGENTS);
    for _ in 0..E2_N_AGENTS {
        let mut last = min_idx;
        let mut chosen = None;
        for _ in 0..E2_INIT_MAX_TRIES {
            let idx = rng.random_range(min_idx..=max_idx);
            last = idx;
            let semitone_abs = (12.0 * log2_ratio_scan[idx]).abs();
            if !is_consonant_near(semitone_abs) {
                chosen = Some(idx);
                break;
            }
        }
        indices.push(chosen.unwrap_or(last));
    }
    indices
}

fn run_e2_once(space: &Log2Space, anchor_hz: f32, seed: u64, condition: E2Condition) -> E2Run {
    let mut rng = seeded_rng(seed);
    let mut anchor_hz_current = anchor_hz;
    let mut log2_ratio_scan = build_log2_ratio_scan(space, anchor_hz_current);
    let (mut min_idx, mut max_idx) = log2_ratio_bounds(&log2_ratio_scan, -1.0, 1.0);

    let mut agent_indices = match E2_INIT_MODE {
        E2InitMode::Uniform => init_e2_agent_indices_uniform(&mut rng, min_idx, max_idx),
        E2InitMode::RejectConsonant => {
            init_e2_agent_indices_reject_consonant(&mut rng, min_idx, max_idx, &log2_ratio_scan)
        }
    };

    let (_erb_scan, du_scan) = erb_grid_for_space(space);
    let workspace = build_c01_workspace(space);
    let k_bins = k_from_semitones(E2_STEP_SEMITONES);

    let mut mean_c01_series = Vec::with_capacity(E2_STEPS);
    let mut mean_score_series = Vec::with_capacity(E2_STEPS);
    let mut mean_repulsion_series = Vec::with_capacity(E2_STEPS);
    let mut moved_frac_series = Vec::with_capacity(E2_STEPS);
    let mut semitone_samples_pre = Vec::new();
    let mut semitone_samples_post = Vec::new();

    let mut trajectory_semitones = (0..E2_N_AGENTS)
        .map(|_| Vec::with_capacity(E2_STEPS))
        .collect::<Vec<_>>();
    let mut trajectory_c01 = (0..E2_N_AGENTS)
        .map(|_| Vec::with_capacity(E2_STEPS))
        .collect::<Vec<_>>();

    let mut anchor_shift = E2AnchorShiftStats {
        step: E2_ANCHOR_SHIFT_STEP,
        anchor_hz_before: anchor_hz_current,
        anchor_hz_after: anchor_hz_current * E2_ANCHOR_SHIFT_RATIO,
        count_min: 0,
        count_max: 0,
        respawned: 0,
    };

    let anchor_shift_enabled = e2_anchor_shift_enabled();
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
            let target = if anchor_shift_enabled && step < E2_ANCHOR_SHIFT_STEP {
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
            E2Condition::NoHillClimb => {
                let mut moved = 0usize;
                for idx in agent_indices.iter_mut() {
                    let step = rng.random_range(-k_bins..=k_bins);
                    let next = (*idx as i32 + step).clamp(min_idx as i32, max_idx as i32);
                    if next as usize != *idx {
                        moved += 1;
                    }
                    *idx = next as usize;
                }
                let mut stats = score_stats_at_indices(
                    &agent_indices,
                    &c01_scan,
                    &log2_ratio_scan,
                    E2_LAMBDA,
                    E2_SIGMA,
                );
                if !agent_indices.is_empty() {
                    stats.moved_frac = moved as f32 / agent_indices.len() as f32;
                }
                stats
            }
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
        seed,
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
    _space: &Log2Space,
    _anchor_hz: f32,
) -> Result<(), Box<dyn Error>> {
    let conditions = [E3Condition::Baseline, E3Condition::NoRecharge];

    let mut long_csv = String::from(
        "condition,seed,life_id,agent_id,birth_step,death_step,lifetime_steps,c01_birth,c01_firstk,avg_c01_tick,avg_c01_attack,attack_tick_count\n",
    );
    let mut summary_csv = String::from(
        "condition,seed,n_deaths,pearson_r_firstk,pearson_p_firstk,spearman_rho_firstk,spearman_p_firstk,logrank_p_firstk,logrank_p_firstk_q25q75,median_high_firstk,median_low_firstk,pearson_r_birth,pearson_p_birth,spearman_rho_birth,spearman_p_birth,pearson_r_attack,pearson_p_attack,spearman_rho_attack,spearman_p_attack,n_attack_lives\n",
    );
    let mut policy_csv = String::from(
        "condition,dt_sec,basal_cost_per_sec,action_cost_per_attack,recharge_per_attack,recharge_threshold\n",
    );
    for condition in conditions {
        let params = e3_policy_params(condition);
        policy_csv.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            params.condition,
            params.dt_sec,
            params.basal_cost_per_sec,
            params.action_cost_per_attack,
            params.recharge_per_attack,
            params.recharge_threshold
        ));
    }
    write(out_dir.join("paper_e3_policy_params.csv"), policy_csv)?;

    let mut seed_outputs: Vec<E3SeedOutput> = Vec::new();

    for condition in conditions {
        let cond_label = condition.label();
        for &seed in &E3_SEEDS {
            let cfg = E3RunConfig {
                seed,
                steps_cap: E3_STEPS_CAP,
                min_deaths: E3_MIN_DEATHS,
                pop_size: E3_POP_SIZE,
                first_k: E3_FIRST_K,
                condition,
            };
            let deaths = run_e3_collect_deaths(&cfg);

            for rec in &deaths {
                long_csv.push_str(&format!(
                    "{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{}\n",
                    rec.condition,
                    rec.seed,
                    rec.life_id,
                    rec.agent_id,
                    rec.birth_step,
                    rec.death_step,
                    rec.lifetime_steps,
                    rec.c01_birth,
                    rec.c01_firstk,
                    rec.avg_c01_tick,
                    rec.avg_c01_attack,
                    rec.attack_tick_count
                ));
            }

            let lifetimes_path = out_dir.join(format!(
                "paper_e3_lifetimes_seed{}_{}.csv",
                seed, cond_label
            ));
            write(lifetimes_path, e3_lifetimes_csv(&deaths))?;

            let arrays = e3_extract_arrays(&deaths);

            let scatter_firstk_path = out_dir.join(format!(
                "paper_e3_firstk_vs_lifetime_seed{}_{}.png",
                seed, cond_label
            ));
            let corr_stats_firstk = render_e3_scatter_with_stats(
                &scatter_firstk_path,
                "E3 C01_firstK vs Lifetime",
                "C01_firstK",
                &arrays.c01_firstk,
                &arrays.lifetimes,
                seed ^ 0xE301_u64,
            )?;

            let scatter_birth_path = out_dir.join(format!(
                "paper_e3_birth_vs_lifetime_seed{}_{}.png",
                seed, cond_label
            ));
            let corr_stats_birth = render_e3_scatter_with_stats(
                &scatter_birth_path,
                "E3 C01_birth vs Lifetime",
                "C01_birth",
                &arrays.c01_birth,
                &arrays.lifetimes,
                seed ^ 0xE302_u64,
            )?;

            let survival_path = out_dir.join(format!(
                "paper_e3_survival_by_firstk_seed{}_{}.png",
                seed, cond_label
            ));
            let surv_firstk_stats = render_survival_split_plot(
                &survival_path,
                "E3 Survival by C01_firstK (median split)",
                &arrays.lifetimes,
                &arrays.c01_firstk,
                SplitKind::Median,
                seed ^ 0xE310_u64,
            )?;

            let survival_q_path = out_dir.join(format!(
                "paper_e3_survival_by_firstk_q25q75_seed{}_{}.png",
                seed, cond_label
            ));
            let surv_firstk_q_stats = render_survival_split_plot(
                &survival_q_path,
                "E3 Survival by C01_firstK (q25 vs q75)",
                &arrays.lifetimes,
                &arrays.c01_firstk,
                SplitKind::Quartiles,
                seed ^ 0xE311_u64,
            )?;

            let (attack_lifetimes, attack_vals) = e3_attack_subset(&arrays);
            let attack_lives = attack_vals.len();
            let mut corr_stats_attack = None;
            if attack_lives >= 10 {
                let attack_scatter_path = out_dir.join(format!(
                    "paper_e3_attack_vs_lifetime_seed{}_{}.png",
                    seed, cond_label
                ));
                corr_stats_attack = Some(render_e3_scatter_with_stats(
                    &attack_scatter_path,
                    "E3 C01_attack vs Lifetime",
                    "C01_attack",
                    &attack_vals,
                    &attack_lifetimes,
                    seed ^ 0xE303_u64,
                )?);

                let attack_survival_path = out_dir.join(format!(
                    "paper_e3_survival_by_attack_seed{}_{}.png",
                    seed, cond_label
                ));
                let _ = render_survival_split_plot(
                    &attack_survival_path,
                    "E3 Survival by C01_attack (median split)",
                    &attack_lifetimes,
                    &attack_vals,
                    SplitKind::Median,
                    seed ^ 0xE313_u64,
                )?;
            }

            if cond_label == "baseline" && seed == E3_SEEDS[0] {
                let mut legacy_csv = String::from("life_id,lifetime_steps,c01_firstk\n");
                let mut legacy_deaths = Vec::with_capacity(deaths.len());
                for d in &deaths {
                    legacy_csv.push_str(&format!(
                        "{},{},{:.6}\n",
                        d.life_id, d.lifetime_steps, d.c01_firstk
                    ));
                    legacy_deaths.push((d.life_id as usize, d.lifetime_steps, d.c01_firstk));
                }
                write(out_dir.join("paper_e3_lifetimes.csv"), legacy_csv)?;
                let legacy_scatter = out_dir.join("paper_e3_firstk_vs_lifetime.png");
                render_consonance_lifetime_scatter(&legacy_scatter, &legacy_deaths)?;
                let legacy_scatter_alias = out_dir.join("paper_e3_consonance_vs_lifetime.png");
                std::fs::copy(&legacy_scatter, &legacy_scatter_alias)?;
                let legacy_survival = out_dir.join("paper_e3_survival_curve.png");
                render_survival_curve(&legacy_survival, &legacy_deaths)?;
                let legacy_survival_c01 = out_dir.join("paper_e3_survival_by_c01.png");
                render_survival_by_c01(&legacy_survival_c01, &legacy_deaths)?;
            }

            summary_csv.push_str(&format!(
                "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3},{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
                cond_label,
                seed,
                arrays.lifetimes.len(),
                corr_stats_firstk.pearson_r,
                corr_stats_firstk.pearson_p,
                corr_stats_firstk.spearman_rho,
                corr_stats_firstk.spearman_p,
                surv_firstk_stats.logrank_p,
                surv_firstk_q_stats.logrank_p,
                surv_firstk_stats.median_high,
                surv_firstk_stats.median_low,
                corr_stats_birth.pearson_r,
                corr_stats_birth.pearson_p,
                corr_stats_birth.spearman_rho,
                corr_stats_birth.spearman_p,
                corr_stats_attack
                    .as_ref()
                    .map(|s| s.pearson_r)
                    .unwrap_or(f32::NAN),
                corr_stats_attack
                    .as_ref()
                    .map(|s| s.pearson_p)
                    .unwrap_or(f32::NAN),
                corr_stats_attack
                    .as_ref()
                    .map(|s| s.spearman_rho)
                    .unwrap_or(f32::NAN),
                corr_stats_attack
                    .as_ref()
                    .map(|s| s.spearman_p)
                    .unwrap_or(f32::NAN),
                attack_lives
            ));

            seed_outputs.push(E3SeedOutput {
                condition,
                seed,
                arrays,
                corr_firstk: corr_stats_firstk,
            });
        }
    }

    if let Some(rep_seed) = pick_e3_representative_seed(&seed_outputs) {
        let mut rep_note =
            String::from("Representative seed selection (baseline firstK Pearson r):\n");
        let mut baseline_stats: Vec<(u64, f32)> = seed_outputs
            .iter()
            .filter(|o| o.condition == E3Condition::Baseline)
            .map(|o| (o.seed, o.corr_firstk.pearson_r))
            .collect();
        baseline_stats.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for (seed, r) in &baseline_stats {
            rep_note.push_str(&format!("{seed},{r:.6}\n"));
        }
        rep_note.push_str(&format!("chosen_seed={rep_seed}\n"));
        rep_note.push_str("note=paper_e3_firstk_vs_lifetime.png is canonical; paper_e3_consonance_vs_lifetime.png is a legacy alias\n");
        write(out_dir.join("paper_e3_representative_seed.txt"), rep_note)?;

        let base = seed_outputs
            .iter()
            .find(|o| o.condition == E3Condition::Baseline && o.seed == rep_seed);
        let norecharge = seed_outputs
            .iter()
            .find(|o| o.condition == E3Condition::NoRecharge && o.seed == rep_seed);
        if let (Some(base), Some(norecharge)) = (base, norecharge) {
            let base_scatter = build_scatter_data(
                &base.arrays.c01_firstk,
                &base.arrays.lifetimes,
                rep_seed ^ 0xE301_u64,
            );
            let norecharge_scatter = build_scatter_data(
                &norecharge.arrays.c01_firstk,
                &norecharge.arrays.lifetimes,
                rep_seed ^ 0xE301_u64,
            );
            let compare_scatter = out_dir.join("paper_e3_firstk_scatter_compare.png");
            render_scatter_compare(
                &compare_scatter,
                "E3 C01_firstK vs Lifetime",
                "C01_firstK",
                "Baseline",
                &base_scatter,
                "NoRecharge",
                &norecharge_scatter,
            )?;

            let base_surv = build_survival_data(
                &base.arrays.lifetimes,
                &base.arrays.c01_firstk,
                SplitKind::Median,
                rep_seed ^ 0xE310_u64,
            );
            let norecharge_surv = build_survival_data(
                &norecharge.arrays.lifetimes,
                &norecharge.arrays.c01_firstk,
                SplitKind::Median,
                rep_seed ^ 0xE310_u64,
            );
            let compare_surv = out_dir.join("paper_e3_firstk_survival_compare.png");
            render_survival_compare(
                &compare_surv,
                "E3 Survival by C01_firstK (median split)",
                "Baseline",
                &base_surv,
                "NoRecharge",
                &norecharge_surv,
            )?;

            let base_surv_q = build_survival_data(
                &base.arrays.lifetimes,
                &base.arrays.c01_firstk,
                SplitKind::Quartiles,
                rep_seed ^ 0xE311_u64,
            );
            let norecharge_surv_q = build_survival_data(
                &norecharge.arrays.lifetimes,
                &norecharge.arrays.c01_firstk,
                SplitKind::Quartiles,
                rep_seed ^ 0xE311_u64,
            );
            let compare_surv_q = out_dir.join("paper_e3_firstk_survival_compare_q25q75.png");
            render_survival_compare(
                &compare_surv_q,
                "E3 Survival by C01_firstK (q25 vs q75)",
                "Baseline",
                &base_surv_q,
                "NoRecharge",
                &norecharge_surv_q,
            )?;
        }
    }

    write(out_dir.join("paper_e3_lifetimes_long.csv"), long_csv)?;
    write(out_dir.join("paper_e3_summary_by_seed.csv"), summary_csv)?;

    Ok(())
}

fn e3_extract_arrays(deaths: &[E3DeathRecord]) -> E3Arrays {
    let mut lifetimes = Vec::with_capacity(deaths.len());
    let mut c01_birth = Vec::with_capacity(deaths.len());
    let mut c01_firstk = Vec::with_capacity(deaths.len());
    let mut avg_attack = Vec::with_capacity(deaths.len());
    let mut attack_tick_count = Vec::with_capacity(deaths.len());
    for d in deaths {
        lifetimes.push(d.lifetime_steps);
        c01_birth.push(d.c01_birth);
        c01_firstk.push(d.c01_firstk);
        avg_attack.push(d.avg_c01_attack);
        attack_tick_count.push(d.attack_tick_count);
    }
    E3Arrays {
        lifetimes,
        c01_birth,
        c01_firstk,
        avg_attack,
        attack_tick_count,
    }
}

fn e3_lifetimes_csv(deaths: &[E3DeathRecord]) -> String {
    let mut out = String::from(
        "life_id,agent_id,birth_step,death_step,lifetime_steps,c01_birth,c01_firstk,avg_c01_tick,avg_c01_attack,attack_tick_count\n",
    );
    for d in deaths {
        out.push_str(&format!(
            "{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{}\n",
            d.life_id,
            d.agent_id,
            d.birth_step,
            d.death_step,
            d.lifetime_steps,
            d.c01_birth,
            d.c01_firstk,
            d.avg_c01_tick,
            d.avg_c01_attack,
            d.attack_tick_count
        ));
    }
    out
}

fn e3_attack_subset(arrays: &E3Arrays) -> (Vec<u32>, Vec<f32>) {
    let mut lifetimes = Vec::new();
    let mut avg_attack = Vec::new();
    for ((&lt, &avg), &count) in arrays
        .lifetimes
        .iter()
        .zip(arrays.avg_attack.iter())
        .zip(arrays.attack_tick_count.iter())
    {
        if count > 0 && avg.is_finite() {
            lifetimes.push(lt);
            avg_attack.push(avg);
        }
    }
    (lifetimes, avg_attack)
}

fn pick_e3_representative_seed(outputs: &[E3SeedOutput]) -> Option<u64> {
    let mut baseline: Vec<(u64, f32)> = outputs
        .iter()
        .filter(|o| o.condition == E3Condition::Baseline)
        .map(|o| (o.seed, o.corr_firstk.pearson_r))
        .collect();
    if baseline.is_empty() {
        return None;
    }
    baseline.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    Some(baseline[baseline.len() / 2].0)
}

fn plot_e5_rhythmic_entrainment(out_dir: &Path) -> Result<(), Box<dyn Error>> {
    let sim_main = simulate_e5_kick(E5_SEED, E5_STEPS, E5_K_KICK, E5_KICK_ON_STEP);
    let sim_ctrl = simulate_e5_kick(E5_SEED, E5_STEPS, 0.0, E5_KICK_ON_STEP);

    let csv_path = out_dir.join("paper_e5_kick_entrainment.csv");
    write(&csv_path, e5_kick_csv(&sim_main, &sim_ctrl))?;

    let summary_path = out_dir.join("paper_e5_kick_summary.csv");
    write(&summary_path, e5_kick_summary_csv(&sim_main, &sim_ctrl))?;

    let meta_path = out_dir.join("paper_e5_meta.txt");
    write(&meta_path, e5_meta_text(E5_STEPS))?;

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

struct E5KickSimResult {
    series: Vec<(f32, f32, f32, f32, f32, f32)>,
    phase_hist_samples: Vec<f32>,
    plv_time: f32,
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
    let sample_start = e5_sample_start_step(steps);

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
        let mut plv_count = 0usize;
        for i in 0..E5_N_AGENTS {
            let d_i = wrap_to_pi(thetas[i] - theta_kick);
            plv_buffers[i].push(d_i);
            let plv_i = if plv_buffers[i].is_full() {
                plv_buffers[i].plv()
            } else {
                f32::NAN
            };
            if plv_i.is_finite() {
                plv_sum += plv_i;
                plv_count += 1;
            }
            if step >= sample_start {
                phase_hist_samples.push(d_i);
            }
        }
        let plv_agent_kick = if plv_count > 0 {
            plv_sum / plv_count as f32
        } else {
            f32::NAN
        };
        group_plv.push(delta_phi);
        let plv_group_delta_phi = if r < E5_MIN_R_FOR_GROUP_PHASE {
            f32::NAN
        } else if group_plv.is_full() {
            group_plv.plv()
        } else {
            f32::NAN
        };

        series.push((t, r, delta_phi, plv_agent_kick, plv_group_delta_phi, k_eff));
    }

    let plv_time = plv_time_from_series(&series, E5_TIME_PLV_WINDOW_STEPS);
    E5KickSimResult {
        series,
        phase_hist_samples,
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
    seed: u64,
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

struct E3Arrays {
    lifetimes: Vec<u32>,
    c01_birth: Vec<f32>,
    c01_firstk: Vec<f32>,
    avg_attack: Vec<f32>,
    attack_tick_count: Vec<u32>,
}

struct E3SeedOutput {
    condition: E3Condition,
    seed: u64,
    arrays: E3Arrays,
    corr_firstk: CorrStats,
}

struct HistSweepStats {
    centers: Vec<f32>,
    mean_count: Vec<f32>,
    std_count: Vec<f32>,
    mean_frac: Vec<f32>,
    std_frac: Vec<f32>,
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

#[allow(clippy::too_many_arguments)]
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

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
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

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
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

#[allow(clippy::too_many_arguments)]
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

#[allow(clippy::too_many_arguments)]
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

fn e2_summary_csv(runs: &[E2Run]) -> String {
    let mut out =
        String::from("seed,init_mode,steps,burn_in,mean_c01_step0,mean_c01_step_end,delta_c01\n");
    for run in runs {
        let start = run.mean_c01_series.first().copied().unwrap_or(0.0);
        let end = run.mean_c01_series.last().copied().unwrap_or(start);
        let delta = end - start;
        out.push_str(&format!(
            "{},{},{},{},{:.6},{:.6},{:.6}\n",
            run.seed,
            E2_INIT_MODE.label(),
            E2_STEPS,
            E2_BURN_IN,
            start,
            end,
            delta
        ));
    }
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
        "E2_ANCHOR_SHIFT_ENABLED={}\n",
        e2_anchor_shift_enabled()
    ));
    out.push_str(&format!(
        "E2_ANCHOR_SHIFT_RATIO={:.3}\n",
        E2_ANCHOR_SHIFT_RATIO
    ));
    out.push_str(&format!("E2_STEP_SEMITONES={:.3}\n", E2_STEP_SEMITONES));
    out.push_str(&format!("E2_N_AGENTS={}\n", n_agents));
    out.push_str(&format!("E2_K_BINS={}\n", k_bins));
    out.push_str(&format!("E2_LAMBDA={:.3}\n", E2_LAMBDA));
    out.push_str(&format!("E2_SIGMA={:.3}\n", E2_SIGMA));
    out.push_str(&format!("E2_INIT_MODE={}\n", E2_INIT_MODE.label()));
    out.push_str(&format!(
        "E2_INIT_CONSONANT_EXCLUSION_ST={:.3}\n",
        E2_INIT_CONSONANT_EXCLUSION_ST
    ));
    out.push_str(&format!("E2_INIT_MAX_TRIES={}\n", E2_INIT_MAX_TRIES));
    out.push_str(&format!("E2_ANCHOR_BIN_ST={:.3}\n", E2_ANCHOR_BIN_ST));
    out.push_str(&format!("E2_PAIRWISE_BIN_ST={:.3}\n", E2_PAIRWISE_BIN_ST));
    out.push_str(&format!("E2_SEEDS={:?}\n", E2_SEEDS));
    out
}

fn e2_marker_steps() -> Vec<f32> {
    let mut steps = vec![E2_BURN_IN as f32];
    if e2_anchor_shift_enabled() {
        steps.push(E2_ANCHOR_SHIFT_STEP as f32);
    }
    steps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    steps.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    steps
}

fn pick_representative_run_index(runs: &[E2Run]) -> usize {
    if runs.is_empty() {
        return 0;
    }
    let mut scored: Vec<(usize, f32)> = runs
        .iter()
        .enumerate()
        .map(|(i, r)| (i, r.mean_c01_series.last().copied().unwrap_or(0.0)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored[scored.len() / 2].0
}

fn representative_seed_text(runs: &[E2Run], rep_index: usize) -> String {
    let mut scored: Vec<(usize, f32)> = runs
        .iter()
        .enumerate()
        .map(|(i, r)| (i, r.mean_c01_series.last().copied().unwrap_or(0.0)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut out = String::from("metric=post_mean_c01\n");
    out.push_str("rank,seed,metric\n");
    for (rank, (idx, metric)) in scored.iter().enumerate() {
        out.push_str(&format!("{rank},{},{}\n", runs[*idx].seed, metric));
    }
    let rep_metric = runs
        .get(rep_index)
        .and_then(|r| r.mean_c01_series.last().copied())
        .unwrap_or(0.0);
    let rep_pre = runs
        .get(rep_index)
        .and_then(|r| {
            r.mean_c01_series
                .get(E2_ANCHOR_SHIFT_STEP.saturating_sub(1))
                .copied()
        })
        .unwrap_or(0.0);
    let rep_seed = runs.get(rep_index).map(|r| r.seed).unwrap_or(0);
    let rep_rank = scored
        .iter()
        .position(|(idx, _)| *idx == rep_index)
        .unwrap_or(0);
    out.push_str(&format!(
        "representative_seed={rep_seed}\nrepresentative_rank={rep_rank}\nrepresentative_metric={rep_metric}\nrepresentative_pre={rep_pre}\n"
    ));
    out
}

fn histogram_counts_fixed(values: &[f32], min: f32, max: f32, bin_width: f32) -> Vec<(f32, f32)> {
    if bin_width <= 0.0 {
        return Vec::new();
    }
    let bins = ((max - min) / bin_width).ceil().max(1.0) as usize;
    let mut counts = vec![0.0f32; bins];
    for &value in values {
        if !value.is_finite() {
            continue;
        }
        if value < min || value > max {
            continue;
        }
        let mut idx = ((value - min) / bin_width).floor() as isize;
        if idx as usize >= bins {
            idx = (bins - 1) as isize;
        }
        if idx >= 0 {
            counts[idx as usize] += 1.0;
        }
    }
    (0..bins)
        .map(|i| (min + (i as f32 + 0.5) * bin_width, counts[i]))
        .collect()
}

fn mean_std_values(values: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    if values.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let len = values[0].len();
    let mut sum = vec![0.0f32; len];
    let mut sum_sq = vec![0.0f32; len];
    for row in values {
        debug_assert_eq!(row.len(), len, "hist length mismatch");
        for (i, &val) in row.iter().enumerate() {
            sum[i] += val;
            sum_sq[i] += val * val;
        }
    }
    let n = values.len() as f32;
    let mut mean = vec![0.0f32; len];
    let mut std = vec![0.0f32; len];
    for i in 0..len {
        mean[i] = sum[i] / n;
        let var = (sum_sq[i] / n) - mean[i] * mean[i];
        std[i] = var.max(0.0).sqrt();
    }
    (mean, std)
}

fn mean_std_histograms(hists: &[Vec<(f32, f32)>]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    if hists.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let len = hists[0].len();
    let centers: Vec<f32> = hists[0].iter().map(|(c, _)| *c).collect();
    let mut values: Vec<Vec<f32>> = Vec::with_capacity(hists.len());
    for hist in hists {
        debug_assert_eq!(hist.len(), len, "hist length mismatch");
        values.push(hist.iter().map(|(_, v)| *v).collect());
    }
    let (mean, std) = mean_std_values(&values);
    (centers, mean, std)
}

fn mean_std_histogram_fractions(hists: &[Vec<(f32, f32)>]) -> (Vec<f32>, Vec<f32>) {
    if hists.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let len = hists[0].len();
    let mut values: Vec<Vec<f32>> = Vec::with_capacity(hists.len());
    for hist in hists {
        debug_assert_eq!(hist.len(), len, "hist length mismatch");
        let total: f32 = hist.iter().map(|(_, v)| *v).sum();
        let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
        values.push(hist.iter().map(|(_, v)| v * inv).collect());
    }
    mean_std_values(&values)
}

fn e2_hist_seed_sweep(runs: &[E2Run], bin_width: f32, min: f32, max: f32) -> HistSweepStats {
    let mut hists = Vec::with_capacity(runs.len());
    for run in runs {
        hists.push(histogram_counts_fixed(
            &run.semitone_samples_post,
            min,
            max,
            bin_width,
        ));
    }
    let (centers, mean_count, std_count) = mean_std_histograms(&hists);
    let (mean_frac, std_frac) = mean_std_histogram_fractions(&hists);
    HistSweepStats {
        centers,
        mean_count,
        std_count,
        mean_frac,
        std_frac,
        n: hists.len(),
    }
}

fn e2_hist_seed_sweep_csv(stats: &HistSweepStats) -> String {
    let mut out = String::from("bin_center,mean_count,std_count,n_seeds,mean_frac,std_frac\n");
    let len = stats
        .centers
        .len()
        .min(stats.mean_count.len())
        .min(stats.std_count.len())
        .min(stats.mean_frac.len())
        .min(stats.std_frac.len());
    for i in 0..len {
        out.push_str(&format!(
            "{:.4},{:.6},{:.6},{},{:.6},{:.6}\n",
            stats.centers[i],
            stats.mean_count[i],
            stats.std_count[i],
            stats.n,
            stats.mean_frac[i],
            stats.std_frac[i]
        ));
    }
    out
}

fn render_hist_mean_std(
    out_path: &Path,
    caption: &str,
    centers: &[f32],
    mean: &[f32],
    std: &[f32],
    bin_width: f32,
) -> Result<(), Box<dyn Error>> {
    if centers.is_empty() {
        return Ok(());
    }
    let min = centers.first().copied().unwrap_or(0.0) - 0.5 * bin_width;
    let max = centers.last().copied().unwrap_or(0.0) + 0.5 * bin_width;
    let mut y_max = 0.0f32;
    for i in 0..mean.len().min(std.len()) {
        y_max = y_max.max(mean[i] + std[i]);
    }
    y_max = y_max.max(1.0);

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
        .y_desc("mean count")
        .x_labels(25)
        .draw()?;

    let half = bin_width * 0.45;
    for i in 0..centers.len().min(mean.len()).min(std.len()) {
        let center = centers[i];
        let mean_val = mean[i];
        let std_val = std[i];
        chart.draw_series(std::iter::once(Rectangle::new(
            [(center - half, 0.0), (center + half, mean_val)],
            BLUE.mix(0.6).filled(),
        )))?;
        let y0 = (mean_val - std_val).max(0.0);
        let y1 = mean_val + std_val;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(center, y0), (center, y1)],
            BLACK.mix(0.6),
        )))?;
    }

    root.present()?;
    Ok(())
}

fn render_hist_controls_fraction(
    out_path: &Path,
    caption: &str,
    centers: &[f32],
    series: &[(&str, &[f32], RGBColor)],
) -> Result<(), Box<dyn Error>> {
    if centers.is_empty() || series.is_empty() {
        return Ok(());
    }
    let bin_width = if centers.len() > 1 {
        (centers[1] - centers[0]).abs()
    } else {
        0.5
    };
    let min = centers.first().copied().unwrap_or(0.0) - 0.5 * bin_width;
    let max = centers.last().copied().unwrap_or(0.0) + 0.5 * bin_width;
    let mut y_max = 0.0f32;
    for (_, values, _) in series {
        for &v in *values {
            y_max = y_max.max(v);
        }
    }
    y_max = y_max.max(1e-3);

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
        .y_desc("mean fraction")
        .x_labels(25)
        .draw()?;

    for &(label, values, color) in series {
        let line = centers.iter().copied().zip(values.iter().copied());
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

fn e2_c01_snapshot(run: &E2Run) -> (f32, f32, f32) {
    let init = run.mean_c01_series.first().copied().unwrap_or(0.0);
    let pre = run
        .mean_c01_series
        .get(E2_ANCHOR_SHIFT_STEP.saturating_sub(1))
        .copied()
        .unwrap_or(init);
    let post = run.mean_c01_series.last().copied().unwrap_or(pre);
    (init, pre, post)
}

fn mean_std_scalar(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f32;
    let mean = values.iter().copied().sum::<f32>() / n;
    let var = values
        .iter()
        .map(|v| (*v - mean) * (*v - mean))
        .sum::<f32>()
        / n;
    (mean, var.max(0.0).sqrt())
}

fn render_series_plot_fixed_y(
    out_path: &Path,
    caption: &str,
    y_desc: &str,
    series: &[(f32, f32)],
    markers: &[f32],
    mut y_lo: f32,
    mut y_hi: f32,
) -> Result<(), Box<dyn Error>> {
    if series.is_empty() {
        return Ok(());
    }
    if !matches!(y_lo.partial_cmp(&y_hi), Some(std::cmp::Ordering::Less)) {
        y_lo = 0.0;
        y_hi = 1.0;
    }
    let x_max = series.last().map(|(x, _)| *x).unwrap_or(0.0).max(1.0);

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
    let post_start = E2_BURN_IN;
    let post_end = E2_STEPS.saturating_sub(1);
    let ranges: Vec<(&str, usize, usize, &Vec<f32>)> = if e2_anchor_shift_enabled() {
        let pre_start = E2_BURN_IN;
        let pre_end = E2_ANCHOR_SHIFT_STEP.saturating_sub(1);
        let post_start = E2_ANCHOR_SHIFT_STEP;
        let post_end = E2_STEPS.saturating_sub(1);
        vec![
            ("pre", pre_start, pre_end, &run.semitone_samples_pre),
            ("post", post_start, post_end, &run.semitone_samples_post),
        ]
    } else {
        vec![("post", post_start, post_end, &run.semitone_samples_post)]
    };
    let bins = [0.5f32, 0.25f32];
    for (label, start, end, values) in ranges {
        for &bin_width in &bins {
            let fname = format!(
                "paper_e2_interval_histogram_{}_bw{}.png",
                label,
                format_float_token(bin_width)
            );
            let phase_label = if label == "post" {
                e2_post_label()
            } else {
                label
            };
            let caption = format!(
                "E2 Interval Histogram ({phase_label}, steps {start}-{end}, bin={bin_width:.2}st)"
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
    let post_label = e2_post_label();
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("E2 Interval Histogram ({post_label}, controls, bin=0.50st)"),
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

struct CorrStats {
    n: usize,
    pearson_r: f32,
    pearson_p: f32,
    spearman_rho: f32,
    spearman_p: f32,
}

fn pearson_r(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let n = x.len() as f32;
    let mean_x = x.iter().copied().sum::<f32>() / n;
    let mean_y = y.iter().copied().sum::<f32>() / n;
    let mut num = 0.0f32;
    let mut den_x = 0.0f32;
    let mut den_y = 0.0f32;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    let den = (den_x * den_y).sqrt();
    if den > 0.0 { num / den } else { 0.0 }
}

fn ranks(values: &[f32]) -> Vec<f32> {
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0f32; values.len()];
    let mut i = 0usize;
    while i < indexed.len() {
        let start = i;
        let val = indexed[i].1;
        let mut end = i + 1;
        while end < indexed.len() && (indexed[end].1 - val).abs() < 1e-6 {
            end += 1;
        }
        let rank = (start + end - 1) as f32 * 0.5 + 1.0;
        for j in start..end {
            ranks[indexed[j].0] = rank;
        }
        i = end;
    }
    ranks
}

fn spearman_rho(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let rx = ranks(x);
    let ry = ranks(y);
    pearson_r(&rx, &ry)
}

fn perm_pvalue(
    x: &[f32],
    y: &[f32],
    n_perm: usize,
    seed: u64,
    corr_fn: fn(&[f32], &[f32]) -> f32,
) -> f32 {
    if x.len() != y.len() || x.len() < 2 {
        return 1.0;
    }
    let obs = corr_fn(x, y).abs();
    let mut rng = seeded_rng(seed);
    let mut y_perm = y.to_vec();
    let mut count = 0usize;
    for _ in 0..n_perm {
        y_perm.shuffle(&mut rng);
        let r = corr_fn(x, &y_perm).abs();
        if r >= obs {
            count += 1;
        }
    }
    (count as f32 + 1.0) / (n_perm as f32 + 1.0)
}

fn corr_stats(x_raw: &[f32], y_raw: &[u32], seed: u64) -> CorrStats {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (x, y) in x_raw.iter().zip(y_raw.iter()) {
        if x.is_finite() {
            xs.push(*x);
            ys.push(*y as f32);
        }
    }
    let n = xs.len();
    let pearson = pearson_r(&xs, &ys);
    let spearman = spearman_rho(&xs, &ys);
    let pearson_p = perm_pvalue(&xs, &ys, 1000, seed ^ 0xA11CE_u64, pearson_r);
    let spearman_p = perm_pvalue(&xs, &ys, 1000, seed ^ 0xBEEF0_u64, spearman_rho);
    CorrStats {
        n,
        pearson_r: pearson,
        pearson_p,
        spearman_rho: spearman,
        spearman_p,
    }
}

struct ScatterData {
    points: Vec<(f32, f32)>,
    x_min: f32,
    x_max: f32,
    y_max: f32,
    stats: CorrStats,
}

fn build_scatter_data(x_values: &[f32], lifetimes: &[u32], seed: u64) -> ScatterData {
    let stats = corr_stats(x_values, lifetimes, seed);
    let mut points = Vec::new();
    for (x, y) in x_values.iter().zip(lifetimes.iter()) {
        if x.is_finite() {
            points.push((*x, *y as f32));
        }
    }
    let mut x_min = 0.0f32;
    let mut x_max = 1.0f32;
    let mut y_max = 1.0f32;
    if !points.is_empty() {
        x_min = f32::INFINITY;
        x_max = f32::NEG_INFINITY;
        y_max = f32::NEG_INFINITY;
        for &(x, y) in &points {
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_max = y_max.max(y);
        }
        if !x_min.is_finite() || !x_max.is_finite() {
            x_min = 0.0;
            x_max = 1.0;
        }
        if (x_max - x_min).abs() < 1e-6 {
            x_min -= 0.05;
            x_max += 0.05;
        }
        y_max = y_max.max(1.0);
    }
    ScatterData {
        points,
        x_min,
        x_max,
        y_max,
        stats,
    }
}

fn scatter_with_ranges(data: &ScatterData, x_min: f32, x_max: f32, y_max: f32) -> ScatterData {
    ScatterData {
        points: data.points.clone(),
        x_min,
        x_max,
        y_max,
        stats: CorrStats {
            n: data.stats.n,
            pearson_r: data.stats.pearson_r,
            pearson_p: data.stats.pearson_p,
            spearman_rho: data.stats.spearman_rho,
            spearman_p: data.stats.spearman_p,
        },
    }
}

fn draw_note_lines<DB: DrawingBackend, CT: CoordTranslate>(
    area: &DrawingArea<DB, CT>,
    lines: &[String],
    x_frac: f32,
    y_frac: f32,
    line_height_px: i32,
) -> Result<(), Box<dyn Error>>
where
    DB::ErrorType: 'static,
{
    let screen = area.strip_coord_spec();
    let (w, h) = screen.dim_in_pixel();
    let x = (w as f32 * x_frac).round() as i32;
    let mut y = (h as f32 * y_frac).round() as i32;
    for line in lines {
        screen.draw(&Text::new(
            line.clone(),
            (x, y),
            ("sans-serif", 14).into_font(),
        ))?;
        y += line_height_px;
    }
    Ok(())
}

fn render_scatter_on_area(
    area: &DrawingArea<BitMapBackend, Shift>,
    caption: &str,
    x_desc: &str,
    data: &ScatterData,
) -> Result<(), Box<dyn Error>> {
    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 18))
        .margin(10)
        .x_label_area_size(35)
        .y_label_area_size(50)
        .build_cartesian_2d(data.x_min..data.x_max, 0.0f32..(data.y_max * 1.05))?;

    chart
        .configure_mesh()
        .x_desc(x_desc)
        .y_desc("lifetime (steps)")
        .draw()?;

    if !data.points.is_empty() {
        chart.draw_series(
            data.points
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 3, BLUE.mix(0.5).filled())),
        )?;
    }

    let lines = vec![
        format!("N={}", data.stats.n),
        format!(
            "Pearson r={:.3} (p={:.3})",
            data.stats.pearson_r, data.stats.pearson_p
        ),
        format!(
            "Spearman ρ={:.3} (p={:.3})",
            data.stats.spearman_rho, data.stats.spearman_p
        ),
    ];
    draw_note_lines(chart.plotting_area(), &lines, 0.02, 0.05, 16)?;
    Ok(())
}

fn render_e3_scatter_with_stats(
    out_path: &Path,
    caption: &str,
    x_desc: &str,
    x_values: &[f32],
    lifetimes: &[u32],
    seed: u64,
) -> Result<CorrStats, Box<dyn Error>> {
    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let data = build_scatter_data(x_values, lifetimes, seed);
    render_scatter_on_area(&root, caption, x_desc, &data)?;
    root.present()?;
    Ok(data.stats)
}

enum SplitKind {
    Median,
    Quartiles,
}

struct SurvivalStats {
    median_high: f32,
    median_low: f32,
    logrank_p: f32,
}

struct SurvivalData {
    series_high: Vec<(f32, f32)>,
    series_low: Vec<(f32, f32)>,
    x_max: f32,
    stats: SurvivalStats,
    n_high: usize,
    n_low: usize,
}

fn survival_with_x_max(data: &SurvivalData, x_max: f32) -> SurvivalData {
    SurvivalData {
        series_high: data.series_high.clone(),
        series_low: data.series_low.clone(),
        x_max,
        stats: SurvivalStats {
            median_high: data.stats.median_high,
            median_low: data.stats.median_low,
            logrank_p: data.stats.logrank_p,
        },
        n_high: data.n_high,
        n_low: data.n_low,
    }
}

fn median_u32(values: &[u32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] as f32 + sorted[mid] as f32) * 0.5
    } else {
        sorted[mid] as f32
    }
}

fn split_by_median(values: &[f32]) -> (Vec<usize>, Vec<usize>) {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.is_empty() {
        0.0
    } else {
        sorted[sorted.len() / 2]
    };
    let mut high = Vec::new();
    let mut low = Vec::new();
    for (i, &v) in values.iter().enumerate() {
        if v >= median {
            high.push(i);
        } else {
            low.push(i);
        }
    }
    (high, low)
}

fn split_by_quartiles(values: &[f32]) -> (Vec<usize>, Vec<usize>) {
    if values.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q25 = sorted[(sorted.len() - 1) / 4];
    let q75 = sorted[(sorted.len() - 1) * 3 / 4];
    let mut high = Vec::new();
    let mut low = Vec::new();
    for (i, &v) in values.iter().enumerate() {
        if v >= q75 {
            high.push(i);
        } else if v <= q25 {
            low.push(i);
        }
    }
    (high, low)
}

fn logrank_statistic(high: &[u32], low: &[u32]) -> f32 {
    if high.is_empty() || low.is_empty() {
        return 0.0;
    }
    let mut times: Vec<u32> = high.iter().chain(low.iter()).copied().collect();
    times.sort_unstable();
    times.dedup();
    let mut o_minus_e = 0.0f32;
    let mut var = 0.0f32;
    for &t in &times {
        let n1 = high.iter().filter(|&&v| v >= t).count();
        let n2 = low.iter().filter(|&&v| v >= t).count();
        let d1 = high.iter().filter(|&&v| v == t).count();
        let d2 = low.iter().filter(|&&v| v == t).count();
        let n = n1 + n2;
        let d = d1 + d2;
        if n <= 1 || d == 0 {
            continue;
        }
        let n1f = n1 as f32;
        let nf = n as f32;
        let df = d as f32;
        let expected = df * n1f / nf;
        let var_t = df * (n1f / nf) * (1.0 - n1f / nf) * ((nf - df) / (nf - 1.0));
        o_minus_e += d1 as f32 - expected;
        var += var_t;
    }
    if var > 0.0 {
        o_minus_e / var.sqrt()
    } else {
        0.0
    }
}

fn logrank_pvalue(high: &[u32], low: &[u32], n_perm: usize, seed: u64) -> f32 {
    if high.is_empty() || low.is_empty() {
        return 1.0;
    }
    let obs = logrank_statistic(high, low).abs();
    let mut rng = seeded_rng(seed);
    let mut combined: Vec<u32> = high.iter().chain(low.iter()).copied().collect();
    let n_high = high.len();
    let mut count = 0usize;
    for _ in 0..n_perm {
        combined.shuffle(&mut rng);
        let (a, b) = combined.split_at(n_high);
        let stat = logrank_statistic(a, b).abs();
        if stat >= obs {
            count += 1;
        }
    }
    (count as f32 + 1.0) / (n_perm as f32 + 1.0)
}

fn build_survival_data(
    lifetimes: &[u32],
    values: &[f32],
    split: SplitKind,
    seed: u64,
) -> SurvivalData {
    let mut filtered_lifetimes = Vec::new();
    let mut filtered_values = Vec::new();
    for (&lt, &val) in lifetimes.iter().zip(values.iter()) {
        if val.is_finite() {
            filtered_lifetimes.push(lt);
            filtered_values.push(val);
        }
    }

    let (high_idx, low_idx) = match split {
        SplitKind::Median => split_by_median(&filtered_values),
        SplitKind::Quartiles => split_by_quartiles(&filtered_values),
    };
    let mut high = Vec::new();
    let mut low = Vec::new();
    for &i in &high_idx {
        if let Some(&lt) = filtered_lifetimes.get(i) {
            high.push(lt);
        }
    }
    for &i in &low_idx {
        if let Some(&lt) = filtered_lifetimes.get(i) {
            low.push(lt);
        }
    }

    let max_t = high.iter().chain(low.iter()).copied().max().unwrap_or(0) as usize;
    let series_high = build_survival_series(&high, max_t);
    let series_low = build_survival_series(&low, max_t);
    let x_max = max_t.max(1) as f32;

    let median_high = median_u32(&high);
    let median_low = median_u32(&low);
    let logrank_p = logrank_pvalue(&high, &low, 1000, seed ^ 0xE3AA_u64);
    SurvivalData {
        series_high,
        series_low,
        x_max,
        stats: SurvivalStats {
            median_high,
            median_low,
            logrank_p,
        },
        n_high: high.len(),
        n_low: low.len(),
    }
}

fn render_survival_on_area(
    area: &DrawingArea<BitMapBackend, Shift>,
    caption: &str,
    data: &SurvivalData,
) -> Result<(), Box<dyn Error>> {
    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 18))
        .margin(10)
        .x_label_area_size(35)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0f32..data.x_max, 0.0f32..1.05f32)?;

    chart
        .configure_mesh()
        .x_desc("time (steps)")
        .y_desc("survival")
        .draw()?;

    chart
        .draw_series(LineSeries::new(data.series_high.clone(), &BLUE))?
        .label("high")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
    chart
        .draw_series(LineSeries::new(data.series_low.clone(), &RED))?
        .label("low")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    let lines = vec![
        format!("n_high={}, n_low={}", data.n_high, data.n_low),
        format!(
            "median_high={:.1}, median_low={:.1}",
            data.stats.median_high, data.stats.median_low
        ),
        format!("logrank p={:.3}", data.stats.logrank_p),
    ];
    draw_note_lines(chart.plotting_area(), &lines, 0.02, 0.05, 16)?;

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    Ok(())
}

fn render_survival_split_plot(
    out_path: &Path,
    caption: &str,
    lifetimes: &[u32],
    values: &[f32],
    split: SplitKind,
    seed: u64,
) -> Result<SurvivalStats, Box<dyn Error>> {
    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let data = build_survival_data(lifetimes, values, split, seed);
    render_survival_on_area(&root, caption, &data)?;
    root.present()?;
    Ok(data.stats)
}

fn render_scatter_compare(
    out_path: &Path,
    caption: &str,
    x_desc: &str,
    left_label: &str,
    left_data: &ScatterData,
    right_label: &str,
    right_data: &ScatterData,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(out_path, (1400, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let areas = root.split_evenly((1, 2));
    let x_min = left_data.x_min.min(right_data.x_min);
    let x_max = left_data.x_max.max(right_data.x_max);
    let y_max = left_data.y_max.max(right_data.y_max);
    let left_common = scatter_with_ranges(left_data, x_min, x_max, y_max);
    let right_common = scatter_with_ranges(right_data, x_min, x_max, y_max);
    render_scatter_on_area(
        &areas[0],
        &format!("{caption} — {left_label}"),
        x_desc,
        &left_common,
    )?;
    render_scatter_on_area(
        &areas[1],
        &format!("{caption} — {right_label}"),
        x_desc,
        &right_common,
    )?;
    root.present()?;
    Ok(())
}

fn render_survival_compare(
    out_path: &Path,
    caption: &str,
    left_label: &str,
    left_data: &SurvivalData,
    right_label: &str,
    right_data: &SurvivalData,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(out_path, (1400, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let areas = root.split_evenly((1, 2));
    let x_max = left_data.x_max.max(right_data.x_max);
    let left_common = survival_with_x_max(left_data, x_max);
    let right_common = survival_with_x_max(right_data, x_max);
    render_survival_on_area(
        &areas[0],
        &format!("{caption} — {left_label}"),
        &left_common,
    )?;
    render_survival_on_area(
        &areas[1],
        &format!("{caption} — {right_label}"),
        &right_common,
    )?;
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
    let median = if c01_values.len().is_multiple_of(2) {
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

    let markers = e5_marker_specs(E5_STEPS);
    draw_vertical_guides_labeled(&mut chart, &markers, 0.0, 1.05)?;

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

    let markers = e5_marker_specs(E5_STEPS);
    draw_vertical_guides_labeled(&mut chart, &markers, -PI, PI)?;

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

    let plv_main: Vec<(f32, f32)> = main_series
        .iter()
        .filter_map(|(t, _, _, plv, _, _)| plv.is_finite().then_some((*t, *plv)))
        .collect();
    let plv_ctrl: Vec<(f32, f32)> = ctrl_series
        .iter()
        .filter_map(|(t, _, _, plv, _, _)| plv.is_finite().then_some((*t, *plv)))
        .collect();
    let group_main: Vec<(f32, f32)> = main_series
        .iter()
        .filter_map(|(t, _, _, _, plv, _)| plv.is_finite().then_some((*t, *plv)))
        .collect();
    let group_ctrl: Vec<(f32, f32)> = ctrl_series
        .iter()
        .filter_map(|(t, _, _, _, plv, _)| plv.is_finite().then_some((*t, *plv)))
        .collect();

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

    let markers = e5_marker_specs(E5_STEPS);
    draw_vertical_guides_labeled(&mut chart, &markers, 0.0, 1.05)?;

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

fn draw_vertical_guides_labeled<DB: DrawingBackend>(
    chart: &mut ChartContext<DB, Cartesian2d<RangedCoordf32, RangedCoordf32>>,
    markers: &[(f32, &str)],
    y_min: f32,
    y_max: f32,
) -> Result<(), Box<dyn Error>>
where
    DB::ErrorType: 'static,
{
    let style = ShapeStyle::from(&BLACK.mix(0.45)).stroke_width(2);
    let y_span = (y_max - y_min).abs();
    let text_y = y_max - 0.05 * y_span;
    for &(x, label) in markers {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, y_min), (x, y_max)],
            style,
        )))?;
        chart.draw_series(std::iter::once(Text::new(
            label.to_string(),
            (x, text_y),
            ("sans-serif", 14).into_font().color(&BLACK),
        )))?;
    }
    Ok(())
}

fn e5_sample_start_step(steps: usize) -> usize {
    steps
        .saturating_sub(E5_SAMPLE_WINDOW_STEPS)
        .max(E5_BURN_IN_STEPS)
}

fn e5_marker_specs(steps: usize) -> Vec<(f32, &'static str)> {
    let mut markers = Vec::new();
    let burn_t = E5_BURN_IN_STEPS as f32 * E5_DT;
    markers.push((burn_t, "burn-in end"));
    if let Some(kick_on) = E5_KICK_ON_STEP {
        markers.push((kick_on as f32 * E5_DT, "kick ON"));
    }
    let sample_start = e5_sample_start_step(steps);
    markers.push((sample_start as f32 * E5_DT, "eval start"));
    markers.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    markers.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6);
    markers
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

fn e5_mean_plv_range(
    series: &[(f32, f32, f32, f32, f32, f32)],
    t_min: f32,
    t_max: f32,
) -> (f32, f32, usize) {
    let mut values = Vec::new();
    for (t, _, _, plv_agent, _, _) in series {
        if *t >= t_min && *t < t_max && plv_agent.is_finite() {
            values.push(*plv_agent);
        }
    }
    if values.is_empty() {
        return (f32::NAN, f32::NAN, 0);
    }
    let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
    let var = values
        .iter()
        .map(|v| (*v - mean) * (*v - mean))
        .sum::<f32>()
        / values.len() as f32;
    (mean, var.sqrt(), values.len())
}

fn e5_mean_group_range(
    series: &[(f32, f32, f32, f32, f32, f32)],
    t_min: f32,
    t_max: f32,
) -> (f32, f32, usize) {
    let mut values = Vec::new();
    for (t, _, _, _, plv_group, _) in series {
        if *t >= t_min && *t < t_max && plv_group.is_finite() {
            values.push(*plv_group);
        }
    }
    if values.is_empty() {
        return (f32::NAN, f32::NAN, 0);
    }
    let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
    let var = values
        .iter()
        .map(|v| (*v - mean) * (*v - mean))
        .sum::<f32>()
        / values.len() as f32;
    (mean, var.sqrt(), values.len())
}

fn e5_kick_summary_csv(main: &E5KickSimResult, ctrl: &E5KickSimResult) -> String {
    let mut out = String::from(
        "condition,plv_pre_mean,plv_pre_std,plv_post_mean,plv_post_std,delta_phi_post_plv,plv_time\n",
    );
    let burn_end_t = E5_BURN_IN_STEPS as f32 * E5_DT;
    let kick_on_t = E5_KICK_ON_STEP
        .map(|s| s as f32 * E5_DT)
        .unwrap_or(burn_end_t);
    let window_t = E5_TIME_PLV_WINDOW_STEPS as f32 * E5_DT;
    let sample_start_t = e5_sample_start_step(E5_STEPS) as f32 * E5_DT;
    let pre_start = burn_end_t;
    let pre_end = kick_on_t;
    let post_start = (kick_on_t + window_t).max(burn_end_t);
    let post_end = sample_start_t;

    let (pre_main, pre_main_std, _) = e5_mean_plv_range(&main.series, pre_start, pre_end);
    let (post_main, post_main_std, _) = e5_mean_plv_range(&main.series, post_start, post_end);
    let (post_group_main, _, _) = e5_mean_group_range(&main.series, post_start, post_end);

    let (pre_ctrl, pre_ctrl_std, _) = e5_mean_plv_range(&ctrl.series, pre_start, pre_end);
    let (post_ctrl, post_ctrl_std, _) = e5_mean_plv_range(&ctrl.series, post_start, post_end);
    let (post_group_ctrl, _, _) = e5_mean_group_range(&ctrl.series, post_start, post_end);

    out.push_str(&format!(
        "main,{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
        pre_main, pre_main_std, post_main, post_main_std, post_group_main, main.plv_time
    ));
    out.push_str(&format!(
        "control,{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
        pre_ctrl, pre_ctrl_std, post_ctrl, post_ctrl_std, post_group_ctrl, ctrl.plv_time
    ));
    out
}

fn e5_meta_text(steps: usize) -> String {
    let sample_start = e5_sample_start_step(steps);
    let kick_on_time = E5_KICK_ON_STEP.map(|s| s as f32 * E5_DT);
    let eval_start_time = sample_start as f32 * E5_DT;
    let mut out = String::new();
    out.push_str(&format!("dt={}\n", E5_DT));
    out.push_str(&format!("steps={}\n", steps));
    out.push_str(&format!("burn_in_steps={}\n", E5_BURN_IN_STEPS));
    out.push_str(&format!("sample_window_steps={}\n", E5_SAMPLE_WINDOW_STEPS));
    out.push_str(&format!(
        "time_plv_window_steps={}\n",
        E5_TIME_PLV_WINDOW_STEPS
    ));
    match E5_KICK_ON_STEP {
        Some(step) => {
            out.push_str(&format!("kick_on_step={step}\n"));
            if let Some(t) = kick_on_time {
                out.push_str(&format!("kick_on_time_s={t}\n"));
            }
        }
        None => {
            out.push_str("kick_on_step=none\n");
            out.push_str("kick_on_time_s=none\n");
        }
    }
    out.push_str(&format!("eval_start_step={sample_start}\n"));
    out.push_str(&format!("eval_start_time_s={eval_start_time}\n"));
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

    fn is_full(&self) -> bool {
        self.len >= self.window
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

    fn mean_plv_over_time(series: &[(f32, f32, f32, f32, f32, f32)], t_min: f32) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for (t, _, _, plv_agent, _, _) in series {
            if *t >= t_min && plv_agent.is_finite() {
                sum += *plv_agent;
                count += 1;
            }
        }
        if count == 0 {
            return f32::NAN;
        }
        sum / count as f32
    }

    fn mean_plv_tail(series: &[(f32, f32, f32, f32, f32, f32)], window: usize) -> f32 {
        if series.is_empty() {
            return f32::NAN;
        }
        let window = window.min(series.len());
        let start = series.len() - window;
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for (_, _, _, plv_agent, _, _) in series.iter().skip(start) {
            if plv_agent.is_finite() {
                sum += *plv_agent;
                count += 1;
            }
        }
        if count == 0 {
            return f32::NAN;
        }
        sum / count as f32
    }

    fn test_e2_run(metric: f32, seed: u64) -> E2Run {
        E2Run {
            seed,
            mean_c01_series: vec![metric],
            mean_score_series: vec![0.0],
            mean_repulsion_series: vec![0.0],
            moved_frac_series: vec![0.0],
            semitone_samples_pre: Vec::new(),
            semitone_samples_post: Vec::new(),
            final_semitones: Vec::new(),
            final_freqs_hz: Vec::new(),
            final_log2_ratios: Vec::new(),
            trajectory_semitones: Vec::new(),
            trajectory_c01: Vec::new(),
            anchor_shift: E2AnchorShiftStats {
                step: 0,
                anchor_hz_before: 0.0,
                anchor_hz_after: 0.0,
                count_min: 0,
                count_max: 0,
                respawned: 0,
            },
            n_agents: 0,
            k_bins: 0,
        }
    }

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
    fn consonant_exclusion_flags_targets() {
        for &st in &[0.0, 3.0, 4.0, 7.0, 12.0] {
            assert!(is_consonant_near(st), "expected consonant near {st}");
        }
    }

    #[test]
    fn consonant_exclusion_rejects_clear_outside() {
        for &st in &[1.0, 2.0, 6.0, 11.0] {
            assert!(!is_consonant_near(st), "expected non-consonant near {st}");
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
        let last_t = sim.series.last().unwrap().0;
        assert!((last_t - (119.0 * E5_DT)).abs() < 1e-6, "last_t mismatch");
    }

    #[test]
    fn e5_phase_hist_sample_count_matches() {
        let steps = E5_BURN_IN_STEPS + E5_SAMPLE_WINDOW_STEPS + 10;
        let sim = simulate_e5_kick(E5_SEED, steps, E5_K_KICK, E5_KICK_ON_STEP);
        let sample_start = e5_sample_start_step(steps);
        let expected = (steps.saturating_sub(sample_start)) * E5_N_AGENTS;
        assert_eq!(sim.phase_hist_samples.len(), expected);
    }

    #[test]
    fn e5_plv_agent_kick_within_unit_range() {
        let sim = simulate_e5_kick(E5_SEED, 200, E5_K_KICK, E5_KICK_ON_STEP);
        for (_, _, _, plv_agent, plv_group, _) in &sim.series {
            if plv_agent.is_finite() {
                assert!(
                    *plv_agent >= -1e-6 && *plv_agent <= 1.0 + 1e-6,
                    "plv_agent_kick out of range: {plv_agent}"
                );
            } else {
                assert!(plv_agent.is_nan());
            }
            if plv_group.is_finite() {
                assert!(
                    *plv_group >= -1e-6 && *plv_group <= 1.0 + 1e-6,
                    "plv_group_delta_phi out of range: {plv_group}"
                );
            } else {
                assert!(plv_group.is_nan());
            }
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
        let main_mean = mean_plv_tail(&sim_main.series, E5_SAMPLE_WINDOW_STEPS);
        let ctrl_mean = mean_plv_tail(&sim_ctrl.series, E5_SAMPLE_WINDOW_STEPS);
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
    fn e5_plv_nan_until_window_full() {
        let steps = 300;
        let sim = simulate_e5_kick(E5_SEED, steps, E5_K_KICK, None);
        let window_full_step = E5_TIME_PLV_WINDOW_STEPS.saturating_sub(1);
        for (step, (_, r, _, plv_agent, plv_group, _)) in sim.series.iter().enumerate() {
            if step < window_full_step {
                assert!(plv_agent.is_nan());
                assert!(plv_group.is_nan());
            } else {
                assert!(plv_agent.is_finite());
                if *r >= E5_MIN_R_FOR_GROUP_PHASE {
                    assert!(plv_group.is_finite());
                } else {
                    assert!(plv_group.is_nan());
                }
            }
        }
    }

    #[test]
    fn e5_main_entrains_control_does_not() {
        let sim_main = simulate_e5_kick(E5_SEED, E5_STEPS, E5_K_KICK, E5_KICK_ON_STEP);
        let sim_ctrl = simulate_e5_kick(E5_SEED, E5_STEPS, 0.0, E5_KICK_ON_STEP);
        let kick_on_t = E5_KICK_ON_STEP.map(|s| s as f32 * E5_DT).unwrap_or(0.0);
        let t_min = kick_on_t + E5_TIME_PLV_WINDOW_STEPS as f32 * E5_DT;

        let mean_main = mean_plv_over_time(&sim_main.series, t_min);
        let mean_ctrl = mean_plv_over_time(&sim_ctrl.series, t_min);
        assert!(mean_main > 0.9, "mean main PLV too low: {mean_main}");
        assert!(mean_ctrl < 0.5, "mean ctrl PLV too high: {mean_ctrl}");

        if let Some(on_step) = E5_KICK_ON_STEP {
            if on_step < sim_main.series.len() {
                let k_before = sim_main.series[0].5;
                let k_after = sim_main.series[on_step].5;
                assert!(k_before.abs() < 1e-6);
                assert!((k_after - E5_K_KICK).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn histogram_counts_fixed_edges_inclusive() {
        let values = vec![0.0f32, 1.0f32];
        let hist = histogram_counts_fixed(&values, 0.0, 1.0, 0.5);
        assert_eq!(hist.len(), 2);
        assert_eq!(hist[0].1, 1.0);
        assert_eq!(hist[1].1, 1.0);
    }

    #[test]
    fn pick_representative_run_index_is_median() {
        let runs = vec![
            test_e2_run(0.1, 1),
            test_e2_run(0.2, 2),
            test_e2_run(0.3, 3),
            test_e2_run(0.4, 4),
            test_e2_run(0.5, 5),
        ];
        let idx = pick_representative_run_index(&runs);
        assert_eq!(runs[idx].seed, 3);
    }

    #[test]
    fn mean_std_values_sanity() {
        let values = vec![vec![1.0f32, 3.0], vec![3.0, 5.0]];
        let (mean, std) = mean_std_values(&values);
        assert_eq!(mean, vec![2.0, 4.0]);
        assert!((std[0] - 1.0).abs() < 1e-6);
        assert!((std[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ranks_ties_average_rank() {
        let values = vec![1.0f32, 1.0, 2.0];
        let ranked = ranks(&values);
        assert!((ranked[0] - 1.5).abs() < 1e-6);
        assert!((ranked[1] - 1.5).abs() < 1e-6);
        assert!((ranked[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn pearson_r_perfect() {
        let x = vec![1.0f32, 2.0, 3.0];
        let y = vec![1.0f32, 2.0, 3.0];
        let y_rev = vec![3.0f32, 2.0, 1.0];
        let y_flat = vec![1.0f32, 1.0, 1.0];
        assert!((pearson_r(&x, &y) - 1.0).abs() < 1e-6);
        assert!((pearson_r(&x, &y_rev) + 1.0).abs() < 1e-6);
        assert_eq!(pearson_r(&x, &y_flat), 0.0);
    }

    #[test]
    fn spearman_rho_perfect() {
        let x = vec![1.0f32, 2.0, 3.0];
        let y = vec![1.0f32, 2.0, 3.0];
        let y_rev = vec![3.0f32, 2.0, 1.0];
        assert!((spearman_rho(&x, &y) - 1.0).abs() < 1e-6);
        assert!((spearman_rho(&x, &y_rev) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn split_by_median_and_quartiles_sizes() {
        let values = vec![0.1f32, 0.2, 0.3, 0.4];
        let (high, low) = split_by_median(&values);
        assert_eq!(high.len(), 2);
        assert_eq!(low.len(), 2);
        let (high_q, low_q) = split_by_quartiles(&values);
        assert_eq!(high_q.len(), 2);
        assert_eq!(low_q.len(), 1);
    }

    #[test]
    fn logrank_statistic_zero_when_equal() {
        let high = vec![1u32, 2, 3];
        let low = vec![1u32, 2, 3];
        let stat = logrank_statistic(&high, &low);
        assert!(stat.abs() < 1e-6);
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
