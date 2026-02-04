use std::env;
use std::error::Error;
use std::f32::consts::PI;
use std::fs::{create_dir_all, remove_dir_all, write};
use std::io;
use std::path::{Path, PathBuf};

use plotters::coord::types::RangedCoordf32;
use plotters::coord::{CoordTranslate, Shift};
use plotters::prelude::*;

use crate::sim::{
    E3Condition, E3DeathRecord, E3RunConfig, E4_ANCHOR_HZ, E4TailSamples, e3_policy_params,
    run_e3_collect_deaths, run_e4_condition_tail_samples,
};
use conchordal::core::erb::hz_to_erb;
use conchordal::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
use conchordal::core::landscape::{LandscapeParams, RoughnessScalarMode};
use conchordal::core::log2space::Log2Space;
use conchordal::core::psycho_state;
use conchordal::core::roughness_kernel::{KernelParams, RoughnessKernel};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng, rngs::StdRng};

const SPACE_BINS_PER_OCT: u32 = 400;

const E2_STEPS: usize = 100;
const E2_BURN_IN: usize = 10;
const E2_ANCHOR_SHIFT_STEP: usize = usize::MAX;
const E2_ANCHOR_SHIFT_RATIO: f32 = 0.5;
const E2_STEP_SEMITONES: f32 = 0.25;
const E2_ANCHOR_BIN_ST: f32 = 0.5;
const E2_PAIRWISE_BIN_ST: f32 = 0.25;
const E2_N_AGENTS: usize = 24;
const E2_LAMBDA: f32 = 0.15;
const E2_SIGMA: f32 = 0.06;
const E2_INIT_CONSONANT_EXCLUSION_ST: f32 = 0.35;
const E2_INIT_MAX_TRIES: usize = 5000;
const E2_C_STATE_BETA: f32 = 2.0;
const E2_C_STATE_THETA: f32 = 0.0;
const E2_ACCEPT_ENABLED: bool = true;
const E2_ACCEPT_T0: f32 = 0.05;
const E2_ACCEPT_TAU_STEPS: f32 = 30.0;
const E2_ACCEPT_RESET_ON_PHASE: bool = true;
const E2_SCORE_IMPROVE_EPS: f32 = 1e-4;
const E2_ANTI_BACKTRACK_ENABLED: bool = true;
const E2_ANTI_BACKTRACK_PRE_SWITCH_ONLY: bool = false;
const E2_BACKTRACK_ALLOW_EPS: f32 = 1e-4;
const E2_PHASE_SWITCH_STEP: usize = E2_STEPS / 2;
const E2_DIVERSITY_BIN_ST: f32 = 0.25;
const E2_STEP_SEMITONES_SWEEP: [f32; 4] = [0.125, 0.25, 0.5, 1.0];
const E2_LAZY_MOVE_PROB: f32 = 0.65;
const E2_SEMITONE_EPS: f32 = 1e-6;
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

#[derive(Clone, Copy, Debug)]
enum E2UpdateSchedule {
    Checkerboard,
    Lazy,
}

const E2_UPDATE_SCHEDULE: E2UpdateSchedule = E2UpdateSchedule::Checkerboard;

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
const E5_SEEDS: [u64; 5] = [
    0xC0FFEE_u64 + 0,
    0xC0FFEE_u64 + 1,
    0xC0FFEE_u64 + 2,
    0xC0FFEE_u64 + 3,
    0xC0FFEE_u64 + 4,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Experiment {
    E1,
    E2,
    E3,
    E4,
    E5,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum E2PhaseMode {
    Normal,
    DissonanceThenConsonance,
}

impl E2PhaseMode {
    fn label(self) -> &'static str {
        match self {
            E2PhaseMode::Normal => "normal",
            E2PhaseMode::DissonanceThenConsonance => "dissonance_then_consonance",
        }
    }

    fn score_sign(self, step: usize) -> f32 {
        match self {
            E2PhaseMode::Normal => 1.0,
            E2PhaseMode::DissonanceThenConsonance => {
                if step < E2_PHASE_SWITCH_STEP {
                    -1.0
                } else {
                    1.0
                }
            }
        }
    }

    fn switch_step(self) -> Option<usize> {
        match self {
            E2PhaseMode::Normal => None,
            E2PhaseMode::DissonanceThenConsonance => Some(E2_PHASE_SWITCH_STEP),
        }
    }
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

    fn dir_name(self) -> &'static str {
        match self {
            Experiment::E1 => "e1",
            Experiment::E2 => "e2",
            Experiment::E3 => "e3",
            Experiment::E4 => "e4",
            Experiment::E5 => "e5",
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
        "Usage: paper [--exp E1,E2,...] [--e4-hist on|off] [--e2-phase mode]",
        "Examples:",
        "  paper --exp 2",
        "  paper --exp all",
        "  paper 1 3 5",
        "  paper --exp e2,e4",
        "  paper --exp e4 --e4-hist on",
        "  paper --exp e2 --e2-phase dissonance_then_consonance",
        "If no experiment is specified, all (E1-E5) run.",
        "E4 histogram dumps default to off (use --e4-hist on to enable).",
        "E2 phase modes: normal | dissonance_then_consonance (default)",
        "Outputs are written to target/plots/paper/<exp>/ (e.g. target/plots/paper/e2).",
        "target/plots/paper is cleared on each run.",
    ]
    .join("\n")
}

fn parse_experiments(args: &[String]) -> Result<Vec<Experiment>, String> {
    let mut values: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--e4-hist" {
            if i + 1 >= args.len() {
                return Err(format!("Missing value after {arg}\n{}", usage()));
            }
            i += 2;
            continue;
        }
        if arg.starts_with("--e4-hist=") {
            i += 1;
            continue;
        }
        if arg == "--e2-phase" {
            if i + 1 >= args.len() {
                return Err(format!("Missing value after {arg}\n{}", usage()));
            }
            i += 2;
            continue;
        }
        if arg.starts_with("--e2-phase=") {
            i += 1;
            continue;
        }
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

fn parse_e4_hist(args: &[String]) -> Result<bool, String> {
    let mut value: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--e4-hist" {
            if i + 1 >= args.len() {
                return Err(format!("Missing value after {arg}\n{}", usage()));
            }
            value = Some(args[i + 1].clone());
            i += 2;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--e4-hist=") {
            value = Some(rest.to_string());
            i += 1;
            continue;
        }
        i += 1;
    }

    let Some(value) = value else {
        return Ok(false);
    };
    let normalized = value.to_ascii_lowercase();
    match normalized.as_str() {
        "on" | "true" | "1" | "yes" => Ok(true),
        "off" | "false" | "0" | "no" => Ok(false),
        _ => Err(format!(
            "Invalid --e4-hist value '{value}'. Use on/off.\n{}",
            usage()
        )),
    }
}

fn parse_e2_phase(args: &[String]) -> Result<E2PhaseMode, String> {
    let mut value: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--e2-phase" {
            if i + 1 >= args.len() {
                return Err(format!("Missing value after {arg}\n{}", usage()));
            }
            value = Some(args[i + 1].clone());
            i += 2;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--e2-phase=") {
            value = Some(rest.to_string());
            i += 1;
            continue;
        }
        i += 1;
    }

    let Some(value) = value else {
        return Ok(E2PhaseMode::DissonanceThenConsonance);
    };
    let normalized = value.to_ascii_lowercase();
    match normalized.as_str() {
        "normal" => Ok(E2PhaseMode::Normal),
        "dissonance_then_consonance" | "dtc" => Ok(E2PhaseMode::DissonanceThenConsonance),
        _ => Err(format!(
            "Invalid --e2-phase value '{value}'. Use normal or dissonance_then_consonance.\n{}",
            usage()
        )),
    }
}

fn prepare_paper_output_dirs(
    base_dir: &Path,
    experiments: &[Experiment],
) -> io::Result<Vec<(Experiment, PathBuf)>> {
    if experiments.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(experiments.len());
    for &exp in experiments {
        let dir = base_dir.join(exp.dir_name());
        create_dir_all(&dir)?;
        out.push((exp, dir));
    }
    Ok(out)
}

pub(crate) fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("{}", usage());
        return Ok(());
    }
    let e4_hist_enabled = parse_e4_hist(&args).map_err(io::Error::other)?;
    let e2_phase_mode = parse_e2_phase(&args).map_err(io::Error::other)?;
    let experiments = parse_experiments(&args).map_err(io::Error::other)?;
    let experiments = if experiments.is_empty() {
        Experiment::all()
    } else {
        experiments
    };

    let base_dir = Path::new("target/plots/paper");
    debug_assert!(
        base_dir.ends_with(Path::new("target/plots/paper")),
        "refusing to clear unexpected path: {}",
        base_dir.display()
    );
    if base_dir.exists() {
        remove_dir_all(base_dir)?;
    }
    create_dir_all(base_dir)?;
    let experiment_dirs = prepare_paper_output_dirs(base_dir, &experiments)?;

    let anchor_hz = E4_ANCHOR_HZ;
    let space = Log2Space::new(20.0, 8000.0, SPACE_BINS_PER_OCT);

    let space_ref = &space;
    std::thread::scope(|s| -> Result<(), Box<dyn Error>> {
        let mut handles = Vec::new();
        for (exp, out_dir) in &experiment_dirs {
            let exp = *exp;
            let out_dir = out_dir.as_path();
            match exp {
                Experiment::E1 => {
                    let h = s.spawn(|| {
                        plot_e1_landscape_scan(out_dir, space_ref, anchor_hz)
                            .map_err(|err| io::Error::other(err.to_string()))
                    });
                    handles.push((exp.label(), h));
                }
                Experiment::E2 => {
                    let h = s.spawn(|| {
                        plot_e2_emergent_harmony(out_dir, space_ref, anchor_hz, e2_phase_mode)
                            .map_err(|err| io::Error::other(err.to_string()))
                    });
                    handles.push((exp.label(), h));
                }
                Experiment::E3 => {
                    let h = s.spawn(|| {
                        plot_e3_metabolic_selection(out_dir, space_ref, anchor_hz)
                            .map_err(|err| io::Error::other(err.to_string()))
                    });
                    handles.push((exp.label(), h));
                }
                Experiment::E4 => {
                    let h = s.spawn(|| {
                        plot_e4_mirror_sweep(out_dir, anchor_hz, e4_hist_enabled)
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

    println!("Saved paper plots to {}", base_dir.display());
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
        consonance_harmonicity_weight: 1.0,
        consonance_roughness_weight_floor: 0.35,
        consonance_roughness_weight: 0.5,
        c_state_beta: E2_C_STATE_BETA,
        c_state_theta: E2_C_STATE_THETA,
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

    let alpha_h = params.consonance_harmonicity_weight;
    let w0 = params.consonance_roughness_weight_floor;
    let w1 = params.consonance_roughness_weight;
    let mut perc_c_score_scan = vec![0.0f32; space.n_bins()];
    for i in 0..space.n_bins() {
        let h01 = perc_h_state01_scan[i];
        let r01 = perc_r_state01_scan[i];
        let c_score = psycho_state::compose_c_score(alpha_h, w0, w1, h01, r01);
        perc_c_score_scan[i] = if c_score.is_finite() { c_score } else { 0.0 };
    }

    let anchor_log2 = anchor_hz.log2();
    let log2_ratio_scan: Vec<f32> = space
        .centers_log2
        .iter()
        .map(|&l| l - anchor_log2)
        .collect();

    space.assert_scan_len_named(&perc_r_state01_scan, "perc_r_state01_scan");
    space.assert_scan_len_named(&perc_h_state01_scan, "perc_h_state01_scan");
    space.assert_scan_len_named(&perc_c_score_scan, "perc_c_score_scan");
    space.assert_scan_len_named(&log2_ratio_scan, "log2_ratio_scan");

    let out_path = out_dir.join("paper_e1_landscape_scan_anchor220.png");
    render_e1_plot(
        &out_path,
        anchor_hz,
        &log2_ratio_scan,
        &perc_h_pot_scan,
        &perc_r_state01_scan,
        &perc_c_score_scan,
    )?;

    Ok(())
}

fn plot_e2_emergent_harmony(
    out_dir: &Path,
    space: &Log2Space,
    anchor_hz: f32,
    phase_mode: E2PhaseMode,
) -> Result<(), Box<dyn Error>> {
    let (baseline_runs, baseline_stats) = e2_seed_sweep(
        space,
        anchor_hz,
        E2Condition::Baseline,
        E2_STEP_SEMITONES,
        phase_mode,
    );
    let rep_index = pick_representative_run_index(&baseline_runs);
    let baseline_run = &baseline_runs[rep_index];
    let marker_steps = e2_marker_steps(phase_mode);
    let caption_suffix = e2_caption_suffix(phase_mode);
    let post_label = e2_post_label();
    let post_label_title = e2_post_label_title();

    write(
        out_dir.join("paper_e2_representative_seed.txt"),
        representative_seed_text(&baseline_runs, rep_index, phase_mode),
    )?;

    write(
        out_dir.join("paper_e2_meta.txt"),
        e2_meta_text(
            baseline_run.n_agents,
            baseline_run.k_bins,
            baseline_run.density_mass_mean,
            baseline_run.density_mass_min,
            baseline_run.density_mass_max,
            baseline_run.r_ref_peak,
            baseline_run.roughness_k,
            baseline_run.roughness_ref_eps,
            baseline_run.r_state01_min,
            baseline_run.r_state01_mean,
            baseline_run.r_state01_max,
            phase_mode,
        ),
    )?;

    let c_score_csv = series_csv("step,mean_c_score", &baseline_run.mean_c_series);
    write(out_dir.join("paper_e2_c_score_timeseries.csv"), c_score_csv)?;
    write(
        out_dir.join("paper_e2_c_state_timeseries.csv"),
        series_csv("step,mean_c_state", &baseline_run.mean_c_state_series),
    )?;
    write(
        out_dir.join("paper_e2_mean_c_score_loo_over_time.csv"),
        series_csv(
            "step,mean_c_score_loo",
            &baseline_run.mean_c_score_loo_series,
        ),
    )?;
    write(
        out_dir.join("paper_e2_mean_c_score_chosen_loo_over_time.csv"),
        series_csv(
            "step,mean_c_score_chosen_loo",
            &baseline_run.mean_c_score_chosen_loo_series,
        ),
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
        out_dir.join("paper_e2_accepted_worse_frac_timeseries.csv"),
        series_csv(
            "step,accepted_worse_frac",
            &baseline_run.accepted_worse_frac_series,
        ),
    )?;
    write(
        out_dir.join("paper_e2_attempted_update_frac_timeseries.csv"),
        series_csv(
            "step,attempted_update_frac",
            &baseline_run.attempted_update_frac_series,
        ),
    )?;
    write(
        out_dir.join("paper_e2_moved_given_attempt_frac_timeseries.csv"),
        series_csv(
            "step,moved_given_attempt_frac",
            &baseline_run.moved_given_attempt_frac_series,
        ),
    )?;
    write(
        out_dir.join("paper_e2_mean_abs_delta_semitones_over_time.csv"),
        series_csv(
            "step,mean_abs_delta_semitones",
            &baseline_run.mean_abs_delta_semitones_series,
        ),
    )?;
    write(
        out_dir.join("paper_e2_mean_abs_delta_semitones_moved_over_time.csv"),
        series_csv(
            "step,mean_abs_delta_semitones_moved",
            &baseline_run.mean_abs_delta_semitones_moved_series,
        ),
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

    let mean_plot_path = out_dir.join("paper_e2_mean_c_state_over_time.png");
    render_series_plot_fixed_y(
        &mean_plot_path,
        &format!("E2 Mean C_state Over Time ({caption_suffix})"),
        "mean C_state",
        &series_pairs(&baseline_run.mean_c_state_series),
        &marker_steps,
        0.0,
        1.0,
    )?;

    let mean_c_score_path = out_dir.join("paper_e2_mean_c_score_over_time.png");
    render_series_plot_with_markers(
        &mean_c_score_path,
        &format!("E2 Mean C Score Over Time ({caption_suffix})"),
        "mean C score",
        &series_pairs(&baseline_run.mean_c_series),
        &marker_steps,
    )?;

    let mean_c_score_loo_path = out_dir.join("paper_e2_mean_c_score_loo_over_time.png");
    render_series_plot_with_markers(
        &mean_c_score_loo_path,
        &format!("E2 Mean C Score (LOO Current) Over Time ({caption_suffix})"),
        "mean C score (LOO current)",
        &series_pairs(&baseline_run.mean_c_score_loo_series),
        &marker_steps,
    )?;

    let mean_c_score_chosen_loo_path =
        out_dir.join("paper_e2_mean_c_score_chosen_loo_over_time.png");
    render_series_plot_with_markers(
        &mean_c_score_chosen_loo_path,
        &format!("E2 Mean C Score (LOO Chosen) Over Time ({caption_suffix})"),
        "mean C score (LOO chosen)",
        &series_pairs(&baseline_run.mean_c_score_chosen_loo_series),
        &marker_steps,
    )?;

    let accept_worse_path = out_dir.join("paper_e2_accepted_worse_frac_over_time.png");
    render_series_plot_fixed_y(
        &accept_worse_path,
        &format!("E2 Accepted Worse Fraction ({caption_suffix})"),
        "accepted worse frac",
        &series_pairs(&baseline_run.accepted_worse_frac_series),
        &marker_steps,
        0.0,
        1.0,
    )?;

    let mean_score_path = out_dir.join("paper_e2_mean_score_over_time.png");
    render_series_plot_with_markers(
        &mean_score_path,
        &format!("E2 Mean Score Over Time ({caption_suffix})"),
        "mean score (C - λ·repulsion)",
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

    let attempted_update_path = out_dir.join("paper_e2_attempted_update_frac_over_time.png");
    render_series_plot_fixed_y(
        &attempted_update_path,
        &format!("E2 Attempted Update Fraction ({caption_suffix})"),
        "attempted update frac",
        &series_pairs(&baseline_run.attempted_update_frac_series),
        &marker_steps,
        0.0,
        1.0,
    )?;

    let moved_given_attempt_path = out_dir.join("paper_e2_moved_given_attempt_frac_over_time.png");
    render_series_plot_fixed_y(
        &moved_given_attempt_path,
        &format!("E2 Moved Given Attempt ({caption_suffix})"),
        "moved given attempt frac",
        &series_pairs(&baseline_run.moved_given_attempt_frac_series),
        &marker_steps,
        0.0,
        1.0,
    )?;

    let abs_delta_path = out_dir.join("paper_e2_mean_abs_delta_semitones_over_time.png");
    render_series_plot_with_markers(
        &abs_delta_path,
        &format!("E2 Mean |Δ| Semitones Over Time ({caption_suffix})"),
        "mean |Δ| semitones",
        &series_pairs(&baseline_run.mean_abs_delta_semitones_series),
        &marker_steps,
    )?;

    let abs_delta_moved_path =
        out_dir.join("paper_e2_mean_abs_delta_semitones_moved_over_time.png");
    render_series_plot_with_markers(
        &abs_delta_moved_path,
        &format!("E2 Mean |Δ| Semitones (Moved) Over Time ({caption_suffix})"),
        "mean |Δ| semitones (moved only)",
        &series_pairs(&baseline_run.mean_abs_delta_semitones_moved_series),
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

    let flutter_segments = e2_flutter_segments(phase_mode);
    let mut flutter_csv = String::from(
        "segment,start_step,end_step,pingpong_rate_moves,reversal_rate_moves,move_rate_stepwise,mean_abs_delta_moved,step_count,moved_step_count,move_count,pingpong_count_moves,reversal_count_moves\n",
    );
    for (label, start, end) in &flutter_segments {
        let metrics =
            flutter_metrics_for_trajectories(&baseline_run.trajectory_semitones, *start, *end);
        flutter_csv.push_str(&format!(
            "{label},{start},{end},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{}\n",
            metrics.pingpong_rate_moves,
            metrics.reversal_rate_moves,
            metrics.move_rate_stepwise,
            metrics.mean_abs_delta_moved,
            metrics.step_count,
            metrics.moved_step_count,
            metrics.move_count,
            metrics.pingpong_count_moves,
            metrics.reversal_count_moves
        ));
    }
    write(out_dir.join("paper_e2_flutter_metrics.csv"), flutter_csv)?;

    render_e2_histogram_sweep(out_dir, baseline_run)?;

    let (nohill_runs, nohill_stats) = e2_seed_sweep(
        space,
        anchor_hz,
        E2Condition::NoHillClimb,
        E2_STEP_SEMITONES,
        phase_mode,
    );
    let (norep_runs, norep_stats) = e2_seed_sweep(
        space,
        anchor_hz,
        E2Condition::NoRepulsion,
        E2_STEP_SEMITONES,
        phase_mode,
    );

    let mut flutter_rows = Vec::new();
    for (cond, runs) in [
        ("baseline", &baseline_runs),
        ("nohill", &nohill_runs),
        ("norep", &norep_runs),
    ] {
        for run in runs.iter() {
            for (segment, start, end) in &flutter_segments {
                let metrics =
                    flutter_metrics_for_trajectories(&run.trajectory_semitones, *start, *end);
                flutter_rows.push(FlutterRow {
                    condition: cond,
                    seed: run.seed,
                    segment,
                    metrics,
                });
            }
        }
    }
    write(
        out_dir.join("paper_e2_flutter_by_seed.csv"),
        flutter_by_seed_csv(&flutter_rows),
    )?;
    write(
        out_dir.join("paper_e2_flutter_summary.csv"),
        flutter_summary_csv(&flutter_rows, &flutter_segments),
    )?;

    write(
        out_dir.join("paper_e2_seed_sweep_mean_c.csv"),
        sweep_csv(
            "step,mean,std,n",
            &baseline_stats.mean_c,
            &baseline_stats.std_c,
            baseline_stats.n,
        ),
    )?;
    write(
        out_dir.join("paper_e2_seed_sweep_mean_c_state.csv"),
        sweep_csv(
            "step,mean,std,n",
            &baseline_stats.mean_c_state,
            &baseline_stats.std_c_state,
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
    write(
        out_dir.join("paper_e2_seed_sweep_mean_c_score_loo.csv"),
        sweep_csv(
            "step,mean,std,n",
            &baseline_stats.mean_c_score_loo,
            &baseline_stats.std_c_score_loo,
            baseline_stats.n,
        ),
    )?;
    write(
        out_dir.join("paper_e2_kbins_sweep_summary.csv"),
        e2_kbins_sweep_csv(space, anchor_hz, phase_mode),
    )?;

    let sweep_mean_path = out_dir.join("paper_e2_mean_c_state_over_time_seeds.png");
    render_series_plot_with_band(
        &sweep_mean_path,
        "E2 Mean C_state (seed sweep)",
        "mean C_state",
        &baseline_stats.mean_c_state,
        &baseline_stats.std_c_state,
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

    let sweep_c_score_loo_path = out_dir.join("paper_e2_mean_c_score_loo_over_time_seeds.png");
    render_series_plot_with_band(
        &sweep_c_score_loo_path,
        "E2 Mean C Score (LOO current, seed sweep)",
        "mean C score (LOO current)",
        &baseline_stats.mean_c_score_loo,
        &baseline_stats.std_c_score_loo,
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
        out_dir.join("paper_e2_control_mean_c.csv"),
        e2_controls_csv_c(&baseline_stats, &nohill_stats, &norep_stats),
    )?;
    write(
        out_dir.join("paper_e2_control_mean_c_state.csv"),
        e2_controls_csv_c_state(&baseline_stats, &nohill_stats, &norep_stats),
    )?;

    let control_plot_path = out_dir.join("paper_e2_mean_c_state_over_time_controls.png");
    render_series_plot_multi(
        &control_plot_path,
        "E2 Mean C_state (controls)",
        "mean C_state",
        &[
            ("baseline", &baseline_stats.mean_c_state, BLUE),
            ("no hill-climb", &nohill_stats.mean_c_state, RED),
            ("no repulsion", &norep_stats.mean_c_state, GREEN),
        ],
        &marker_steps,
    )?;

    let control_c_path = out_dir.join("paper_e2_mean_c_over_time_controls_seeds.png");
    render_series_plot_multi_with_band(
        &control_c_path,
        "E2 Mean C score (controls, seed sweep)",
        "mean C score",
        &[
            (
                "baseline",
                &baseline_stats.mean_c,
                &baseline_stats.std_c,
                BLUE,
            ),
            (
                "no hill-climb",
                &nohill_stats.mean_c,
                &nohill_stats.std_c,
                RED,
            ),
            (
                "no repulsion",
                &norep_stats.mean_c,
                &norep_stats.std_c,
                GREEN,
            ),
        ],
        &marker_steps,
    )?;

    let control_c_score_loo_path =
        out_dir.join("paper_e2_mean_c_score_loo_over_time_controls_seeds.png");
    render_series_plot_multi_with_band(
        &control_c_score_loo_path,
        "E2 Mean C score (LOO current, controls, seed sweep)",
        "mean C score (LOO current)",
        &[
            (
                "baseline",
                &baseline_stats.mean_c_score_loo,
                &baseline_stats.std_c_score_loo,
                BLUE,
            ),
            (
                "no hill-climb",
                &nohill_stats.mean_c_score_loo,
                &nohill_stats.std_c_score_loo,
                RED,
            ),
            (
                "no repulsion",
                &norep_stats.mean_c_score_loo,
                &norep_stats.std_c_score_loo,
                GREEN,
            ),
        ],
        &marker_steps,
    )?;

    let mut diversity_rows_vec = Vec::new();
    diversity_rows_vec.extend(diversity_rows("baseline", &baseline_runs));
    diversity_rows_vec.extend(diversity_rows("nohill", &nohill_runs));
    diversity_rows_vec.extend(diversity_rows("norep", &norep_runs));
    write(
        out_dir.join("paper_e2_diversity_by_seed.csv"),
        diversity_by_seed_csv(&diversity_rows_vec),
    )?;
    write(
        out_dir.join("paper_e2_diversity_summary.csv"),
        diversity_summary_csv(&diversity_rows_vec),
    )?;
    let diversity_plot_path = out_dir.join("paper_e2_diversity_summary.png");
    render_diversity_summary_plot(&diversity_plot_path, &diversity_rows_vec)?;

    let mut hist_rows = Vec::new();
    hist_rows.extend(hist_structure_rows("baseline", &baseline_runs));
    hist_rows.extend(hist_structure_rows("nohill", &nohill_runs));
    hist_rows.extend(hist_structure_rows("norep", &norep_runs));
    write(
        out_dir.join("paper_e2_hist_structure_by_seed.csv"),
        hist_structure_by_seed_csv(&hist_rows),
    )?;
    write(
        out_dir.join("paper_e2_hist_structure_summary.csv"),
        hist_structure_summary_csv(&hist_rows),
    )?;
    let hist_plot_path = out_dir.join("paper_e2_hist_structure_summary.png");
    render_hist_structure_summary_plot(&hist_plot_path, &hist_rows)?;

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
        &format!("E2 {post_label_title} Interval Histogram (seed sweep, mean frac, bin=0.50st)"),
        &hist_stats_05.centers,
        &hist_stats_05.mean_frac,
        &hist_stats_05.std_frac,
        0.5,
        "mean fraction",
    )?;

    let hist_stats_025 = e2_hist_seed_sweep(&baseline_runs, 0.25, hist_min, hist_max);
    write(
        out_dir.join("paper_e2_interval_hist_post_seed_sweep_bw0p25.csv"),
        e2_hist_seed_sweep_csv(&hist_stats_025),
    )?;
    let hist_plot_025 = out_dir.join("paper_e2_interval_hist_post_seed_sweep_bw0p25.png");
    render_hist_mean_std(
        &hist_plot_025,
        &format!("E2 {post_label_title} Interval Histogram (seed sweep, mean frac, bin=0.25st)"),
        &hist_stats_025.centers,
        &hist_stats_025.mean_frac,
        &hist_stats_025.std_frac,
        0.25,
        "mean fraction",
    )?;

    let (pairwise_hist_stats, pairwise_n_pairs) =
        e2_pairwise_hist_seed_sweep(&baseline_runs, E2_PAIRWISE_BIN_ST, 0.0, 12.0);
    write(
        out_dir.join("paper_e2_pairwise_interval_histogram_seeds.csv"),
        e2_pairwise_hist_seed_sweep_csv(&pairwise_hist_stats, pairwise_n_pairs),
    )?;
    let pairwise_hist_plot = out_dir.join("paper_e2_pairwise_interval_histogram_seeds.png");
    render_hist_mean_std(
        &pairwise_hist_plot,
        &format!(
            "E2 Pairwise Interval Histogram (final snapshot, seed sweep, mean frac, bin={:.2}st)",
            E2_PAIRWISE_BIN_ST
        ),
        &pairwise_hist_stats.centers,
        &pairwise_hist_stats.mean_frac,
        &pairwise_hist_stats.std_frac,
        E2_PAIRWISE_BIN_ST,
        "mean fraction",
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

    let pre_label = e2_pre_label();
    let post_label = e2_post_label();
    let pre_step = e2_pre_step();
    let pre_meta = format!(
        "# pre_step={pre_step} pre_label={pre_label} shift_enabled={}\n",
        e2_anchor_shift_enabled()
    );
    let mut delta_csv = pre_meta.clone();
    delta_csv.push_str(&format!(
        "seed,cond,c_init,c_{pre_label},c_{post_label},delta_{pre_label},delta_{post_label}\n"
    ));
    let mut delta_summary = pre_meta;
    delta_summary.push_str(&format!(
        "cond,mean_init,std_init,mean_{pre_label},std_{pre_label},mean_{post_label},std_{post_label},mean_delta_{pre_label},std_delta_{pre_label},mean_delta_{post_label},std_delta_{post_label}\n"
    ));
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
            let (init, pre, post) = e2_c_snapshot(run);
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
    write(out_dir.join("paper_e2_delta_c_by_seed.csv"), delta_csv)?;
    write(out_dir.join("paper_e2_delta_c_summary.csv"), delta_summary)?;

    Ok(())
}

fn e2_anchor_shift_enabled() -> bool {
    E2_ANCHOR_SHIFT_STEP != usize::MAX
}

fn e2_pre_step_for(anchor_shift_enabled: bool, anchor_shift_step: usize, burn_in: usize) -> usize {
    if anchor_shift_enabled {
        anchor_shift_step.saturating_sub(1)
    } else {
        burn_in.saturating_sub(1)
    }
}

fn e2_pre_step() -> usize {
    e2_pre_step_for(e2_anchor_shift_enabled(), E2_ANCHOR_SHIFT_STEP, E2_BURN_IN)
}

fn e2_post_step_for(steps: usize) -> usize {
    steps.saturating_sub(1)
}

fn e2_caption_suffix(phase_mode: E2PhaseMode) -> String {
    let base = if e2_anchor_shift_enabled() {
        format!("burn-in={E2_BURN_IN}, shift@{E2_ANCHOR_SHIFT_STEP}")
    } else {
        format!("burn-in={E2_BURN_IN}, shift=off")
    };
    if let Some(step) = phase_mode.switch_step() {
        format!("{base}, phase_switch@{step}")
    } else {
        base
    }
}

fn e2_pre_label() -> &'static str {
    if e2_anchor_shift_enabled() {
        "pre"
    } else {
        "burnin_end"
    }
}

fn e2_post_label() -> &'static str {
    "post"
}

fn e2_post_label_title() -> &'static str {
    "Post"
}

fn e2_post_window_start_step() -> usize {
    if e2_anchor_shift_enabled() {
        E2_ANCHOR_SHIFT_STEP
    } else {
        E2_BURN_IN
    }
}

fn e2_post_window_end_step() -> usize {
    E2_STEPS.saturating_sub(1)
}

fn e2_accept_temperature(step: usize, phase_mode: E2PhaseMode) -> f32 {
    if !E2_ACCEPT_ENABLED {
        return 0.0;
    }
    if E2_ACCEPT_TAU_STEPS <= 0.0 {
        return E2_ACCEPT_T0.max(0.0);
    }
    let mut phase_step = step;
    if E2_ACCEPT_RESET_ON_PHASE {
        if let Some(switch_step) = phase_mode.switch_step() {
            if step >= switch_step {
                phase_step = step - switch_step;
            }
        }
    }
    E2_ACCEPT_T0.max(0.0) * (-(phase_step as f32) / E2_ACCEPT_TAU_STEPS).exp()
}

fn e2_should_attempt_update(agent_id: usize, step: usize, u_move: f32) -> bool {
    match E2_UPDATE_SCHEDULE {
        E2UpdateSchedule::Checkerboard => (agent_id + step) % 2 == 0,
        E2UpdateSchedule::Lazy => u_move < E2_LAZY_MOVE_PROB.clamp(0.0, 1.0),
    }
}

fn e2_should_block_backtrack(phase_mode: E2PhaseMode, step: usize) -> bool {
    if !E2_ANTI_BACKTRACK_ENABLED {
        return false;
    }
    if E2_ANTI_BACKTRACK_PRE_SWITCH_ONLY {
        if let Some(switch_step) = phase_mode.switch_step() {
            return step < switch_step;
        }
    }
    true
}

fn e2_update_backtrack_targets(targets: &mut [usize], before: &[usize], after: &[usize]) {
    debug_assert_eq!(
        targets.len(),
        before.len(),
        "backtrack_targets len mismatch"
    );
    debug_assert_eq!(before.len(), after.len(), "backtrack_targets len mismatch");
    for i in 0..targets.len() {
        if after[i] != before[i] {
            targets[i] = before[i];
        }
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

fn run_e2_once(
    space: &Log2Space,
    anchor_hz: f32,
    seed: u64,
    condition: E2Condition,
    step_semitones: f32,
    phase_mode: E2PhaseMode,
) -> E2Run {
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
    let workspace = build_consonance_workspace(space);
    let k_bins = k_from_semitones(step_semitones);

    let mut mean_c_series = Vec::with_capacity(E2_STEPS);
    let mut mean_c_state_series = Vec::with_capacity(E2_STEPS);
    let mut mean_c_score_loo_series = Vec::with_capacity(E2_STEPS);
    let mut mean_c_score_chosen_loo_series = Vec::with_capacity(E2_STEPS);
    let mut mean_score_series = Vec::with_capacity(E2_STEPS);
    let mut mean_repulsion_series = Vec::with_capacity(E2_STEPS);
    let mut moved_frac_series = Vec::with_capacity(E2_STEPS);
    let mut accepted_worse_frac_series = Vec::with_capacity(E2_STEPS);
    let mut attempted_update_frac_series = Vec::with_capacity(E2_STEPS);
    let mut moved_given_attempt_frac_series = Vec::with_capacity(E2_STEPS);
    let mut mean_abs_delta_semitones_series = Vec::with_capacity(E2_STEPS);
    let mut mean_abs_delta_semitones_moved_series = Vec::with_capacity(E2_STEPS);
    let mut semitone_samples_pre = Vec::new();
    let mut semitone_samples_post = Vec::new();
    let mut density_mass_sum = 0.0f32;
    let mut density_mass_min = f32::INFINITY;
    let mut density_mass_max = 0.0f32;
    let mut density_mass_count = 0u32;
    let mut r_state01_min = f32::INFINITY;
    let mut r_state01_max = f32::NEG_INFINITY;
    let mut r_state01_mean_sum = 0.0f32;
    let mut r_state01_mean_count = 0u32;

    let mut trajectory_semitones = (0..E2_N_AGENTS)
        .map(|_| Vec::with_capacity(E2_STEPS))
        .collect::<Vec<_>>();
    let mut trajectory_c_state = (0..E2_N_AGENTS)
        .map(|_| Vec::with_capacity(E2_STEPS))
        .collect::<Vec<_>>();
    let mut backtrack_targets = agent_indices.clone();

    let mut anchor_shift = E2AnchorShiftStats {
        step: E2_ANCHOR_SHIFT_STEP,
        anchor_hz_before: anchor_hz_current,
        anchor_hz_after: anchor_hz_current * E2_ANCHOR_SHIFT_RATIO,
        count_min: 0,
        count_max: 0,
        respawned: 0,
    };

    let anchor_shift_enabled = e2_anchor_shift_enabled();
    let phase_switch_step = phase_mode.switch_step();
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
        if let Some(switch_step) = phase_switch_step {
            if step == switch_step {
                backtrack_targets.clone_from_slice(&agent_indices);
            }
        }

        let anchor_idx = nearest_bin(space, anchor_hz_current);
        let (env_scan, density_scan) = build_env_scans(space, anchor_idx, &agent_indices, &du_scan);
        let (c_score_scan, c_state_scan, density_mass, r_state_stats) =
            compute_c_score_state_scans(space, &workspace, &env_scan, &density_scan, &du_scan);
        if density_mass.is_finite() {
            density_mass_sum += density_mass;
            density_mass_min = density_mass_min.min(density_mass);
            density_mass_max = density_mass_max.max(density_mass);
            density_mass_count += 1;
        }
        if r_state_stats.mean.is_finite() {
            r_state01_min = r_state01_min.min(r_state_stats.min);
            r_state01_max = r_state01_max.max(r_state_stats.max);
            r_state01_mean_sum += r_state_stats.mean;
            r_state01_mean_count += 1;
        }

        let mean_c = mean_at_indices(&c_score_scan, &agent_indices);
        let mean_c_state = mean_at_indices(&c_state_scan, &agent_indices);
        let mean_c_score_loo_current = if matches!(condition, E2Condition::NoHillClimb) {
            mean_c_score_loo_at_indices(
                space,
                &workspace,
                &env_scan,
                &density_scan,
                &du_scan,
                &agent_indices,
                &log2_ratio_scan,
            )
        } else {
            f32::NAN
        };
        let mut mean_c_score_loo_chosen = f32::NAN;
        mean_c_series.push(mean_c);
        mean_c_state_series.push(mean_c_state);

        for (agent_id, &idx) in agent_indices.iter().enumerate() {
            let semitone = 12.0 * log2_ratio_scan[idx];
            trajectory_semitones[agent_id].push(semitone);
            trajectory_c_state[agent_id].push(c_state_scan[idx]);
        }

        if step >= E2_BURN_IN {
            let target = if anchor_shift_enabled && step < E2_ANCHOR_SHIFT_STEP {
                &mut semitone_samples_pre
            } else {
                &mut semitone_samples_post
            };
            target.extend(agent_indices.iter().map(|&idx| 12.0 * log2_ratio_scan[idx]));
        }

        let temperature = e2_accept_temperature(step, phase_mode);
        let score_sign = phase_mode.score_sign(step);
        let block_backtrack = e2_should_block_backtrack(phase_mode, step);
        let positions_before_update = agent_indices.clone();
        let mut stats = match condition {
            E2Condition::Baseline => update_agent_indices_scored_stats_loo(
                &mut agent_indices,
                space,
                &workspace,
                &env_scan,
                &density_scan,
                &du_scan,
                &log2_ratio_scan,
                min_idx,
                max_idx,
                k_bins,
                score_sign,
                E2_LAMBDA,
                E2_SIGMA,
                temperature,
                step,
                block_backtrack,
                if block_backtrack {
                    Some(backtrack_targets.as_slice())
                } else {
                    None
                },
                &mut rng,
            ),
            E2Condition::NoRepulsion => update_agent_indices_scored_stats_loo(
                &mut agent_indices,
                space,
                &workspace,
                &env_scan,
                &density_scan,
                &du_scan,
                &log2_ratio_scan,
                min_idx,
                max_idx,
                k_bins,
                score_sign,
                0.0,
                E2_SIGMA,
                temperature,
                step,
                block_backtrack,
                if block_backtrack {
                    Some(backtrack_targets.as_slice())
                } else {
                    None
                },
                &mut rng,
            ),
            E2Condition::NoHillClimb => {
                let prev_indices = agent_indices.clone();
                let mut moved = 0usize;
                let mut abs_delta_sum = 0.0f32;
                let mut abs_delta_moved_sum = 0.0f32;
                for (i, idx) in agent_indices.iter_mut().enumerate() {
                    let step = rng.random_range(-k_bins..=k_bins);
                    let next = (*idx as i32 + step).clamp(min_idx as i32, max_idx as i32);
                    if next as usize != *idx {
                        moved += 1;
                    }
                    let delta_semitones =
                        12.0 * (log2_ratio_scan[next as usize] - log2_ratio_scan[*idx]);
                    let abs_delta = delta_semitones.abs();
                    if abs_delta.is_finite() {
                        abs_delta_sum += abs_delta;
                        if next as usize != prev_indices[i] {
                            abs_delta_moved_sum += abs_delta;
                        }
                    }
                    *idx = next as usize;
                }
                let mut stats = score_stats_at_indices(
                    &agent_indices,
                    &c_score_scan,
                    &log2_ratio_scan,
                    score_sign,
                    E2_LAMBDA,
                    E2_SIGMA,
                );
                if !agent_indices.is_empty() {
                    stats.moved_frac = moved as f32 / agent_indices.len() as f32;
                    stats.mean_abs_delta_semitones = abs_delta_sum / agent_indices.len() as f32;
                    stats.attempted_update_frac = 1.0;
                    stats.moved_given_attempt_frac = stats.moved_frac;
                }
                if moved > 0 {
                    stats.mean_abs_delta_semitones_moved = abs_delta_moved_sum / moved as f32;
                }
                mean_c_score_loo_chosen = mean_c_score_loo_at_indices_with_prev(
                    space,
                    &workspace,
                    &env_scan,
                    &density_scan,
                    &du_scan,
                    &prev_indices,
                    &agent_indices,
                );
                stats
            }
        };
        e2_update_backtrack_targets(
            &mut backtrack_targets,
            &positions_before_update,
            &agent_indices,
        );

        if matches!(condition, E2Condition::NoHillClimb) {
            stats.mean_c_score_current_loo = mean_c_score_loo_current;
            stats.mean_c_score_chosen_loo = mean_c_score_loo_chosen;
        }
        let condition_label = match condition {
            E2Condition::Baseline => "baseline",
            E2Condition::NoRepulsion => "norep",
            E2Condition::NoHillClimb => "nohill",
        };
        debug_assert!(
            stats.mean_c_score_current_loo.is_finite(),
            "mean_c_score_current_loo not finite (cond={condition_label}, step={step}, value={})",
            stats.mean_c_score_current_loo
        );
        debug_assert!(
            stats.mean_c_score_chosen_loo.is_finite(),
            "mean_c_score_chosen_loo not finite (cond={condition_label}, step={step}, value={})",
            stats.mean_c_score_chosen_loo
        );
        mean_c_score_loo_series.push(stats.mean_c_score_current_loo);
        mean_c_score_chosen_loo_series.push(stats.mean_c_score_chosen_loo);
        mean_score_series.push(stats.mean_score);
        mean_repulsion_series.push(stats.mean_repulsion);
        moved_frac_series.push(stats.moved_frac);
        accepted_worse_frac_series.push(stats.accepted_worse_frac);
        attempted_update_frac_series.push(stats.attempted_update_frac);
        moved_given_attempt_frac_series.push(stats.moved_given_attempt_frac);
        mean_abs_delta_semitones_series.push(stats.mean_abs_delta_semitones);
        mean_abs_delta_semitones_moved_series.push(stats.mean_abs_delta_semitones_moved);
    }

    let mut final_semitones = Vec::with_capacity(E2_N_AGENTS);
    let mut final_log2_ratios = Vec::with_capacity(E2_N_AGENTS);
    let mut final_freqs_hz = Vec::with_capacity(E2_N_AGENTS);
    for &idx in &agent_indices {
        final_semitones.push(12.0 * log2_ratio_scan[idx]);
        final_log2_ratios.push(log2_ratio_scan[idx]);
        final_freqs_hz.push(space.centers_hz[idx]);
    }

    let density_mass_mean = if density_mass_count > 0 {
        density_mass_sum / density_mass_count as f32
    } else {
        0.0
    };
    let density_mass_min = if density_mass_min.is_finite() {
        density_mass_min
    } else {
        0.0
    };
    let density_mass_max = if density_mass_max.is_finite() {
        density_mass_max
    } else {
        0.0
    };
    let r_state01_mean = if r_state01_mean_count > 0 {
        r_state01_mean_sum / r_state01_mean_count as f32
    } else {
        0.0
    };
    let r_state01_min = if r_state01_min.is_finite() {
        r_state01_min
    } else {
        0.0
    };
    let r_state01_max = if r_state01_max.is_finite() {
        r_state01_max
    } else {
        0.0
    };

    E2Run {
        seed,
        mean_c_series,
        mean_c_state_series,
        mean_c_score_loo_series,
        mean_c_score_chosen_loo_series,
        mean_score_series,
        mean_repulsion_series,
        moved_frac_series,
        accepted_worse_frac_series,
        attempted_update_frac_series,
        moved_given_attempt_frac_series,
        mean_abs_delta_semitones_series,
        mean_abs_delta_semitones_moved_series,
        semitone_samples_pre,
        semitone_samples_post,
        final_semitones,
        final_freqs_hz,
        final_log2_ratios,
        trajectory_semitones,
        trajectory_c_state,
        anchor_shift,
        density_mass_mean,
        density_mass_min,
        density_mass_max,
        r_state01_min,
        r_state01_mean,
        r_state01_max,
        r_ref_peak: workspace.r_ref_peak,
        roughness_k: workspace.params.roughness_k,
        roughness_ref_eps: workspace.params.roughness_ref_eps,
        n_agents: E2_N_AGENTS,
        k_bins,
    }
}

fn e2_seed_sweep(
    space: &Log2Space,
    anchor_hz: f32,
    condition: E2Condition,
    step_semitones: f32,
    phase_mode: E2PhaseMode,
) -> (Vec<E2Run>, E2SweepStats) {
    let mut runs: Vec<E2Run> = Vec::new();
    for &seed in &E2_SEEDS {
        runs.push(run_e2_once(
            space,
            anchor_hz,
            seed,
            condition,
            step_semitones,
            phase_mode,
        ));
    }

    let n = runs.len();
    let mean_c = mean_std_series(runs.iter().map(|r| &r.mean_c_series).collect::<Vec<_>>());
    let mean_c_state = mean_std_series(
        runs.iter()
            .map(|r| &r.mean_c_state_series)
            .collect::<Vec<_>>(),
    );
    let mean_c_score_loo = mean_std_series(
        runs.iter()
            .map(|r| &r.mean_c_score_loo_series)
            .collect::<Vec<_>>(),
    );
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
            mean_c: mean_c.0,
            std_c: mean_c.1,
            mean_c_state: mean_c_state.0,
            std_c_state: mean_c_state.1,
            mean_c_score_loo: mean_c_score_loo.0,
            std_c_score_loo: mean_c_score_loo.1,
            mean_score: mean_score.0,
            std_score: mean_score.1,
            mean_repulsion: mean_repulsion.0,
            std_repulsion: mean_repulsion.1,
            n,
        },
    )
}

fn e2_kbins_sweep_csv(space: &Log2Space, anchor_hz: f32, phase_mode: E2PhaseMode) -> String {
    let mut out = String::from(
        "step_semitones,k_bins,mean_delta_c,mean_delta_c_state,mean_delta_c_score_loo\n",
    );
    for &step_semitones in &E2_STEP_SEMITONES_SWEEP {
        let (runs, _) = e2_seed_sweep(
            space,
            anchor_hz,
            E2Condition::Baseline,
            step_semitones,
            phase_mode,
        );
        let mut delta_c = Vec::with_capacity(runs.len());
        let mut delta_c_state = Vec::with_capacity(runs.len());
        let mut delta_c_score_loo = Vec::with_capacity(runs.len());
        for run in &runs {
            let start = run.mean_c_series.first().copied().unwrap_or(0.0);
            let end = run.mean_c_series.last().copied().unwrap_or(start);
            delta_c.push(end - start);
            let start_state = run.mean_c_state_series.first().copied().unwrap_or(0.0);
            let end_state = run
                .mean_c_state_series
                .last()
                .copied()
                .unwrap_or(start_state);
            delta_c_state.push(end_state - start_state);
            let start_loo = run.mean_c_score_loo_series.first().copied().unwrap_or(0.0);
            let end_loo = run
                .mean_c_score_loo_series
                .last()
                .copied()
                .unwrap_or(start_loo);
            delta_c_score_loo.push(end_loo - start_loo);
        }
        let (mean_delta_c, _) = mean_std_scalar(&delta_c);
        let (mean_delta_c_state, _) = mean_std_scalar(&delta_c_state);
        let (mean_delta_c_score_loo, _) = mean_std_scalar(&delta_c_score_loo);
        out.push_str(&format!(
            "{:.3},{},{:.6},{:.6},{:.6}\n",
            step_semitones,
            k_from_semitones(step_semitones),
            mean_delta_c,
            mean_delta_c_state,
            mean_delta_c_score_loo
        ));
    }
    out
}

fn plot_e3_metabolic_selection(
    out_dir: &Path,
    _space: &Log2Space,
    _anchor_hz: f32,
) -> Result<(), Box<dyn Error>> {
    let conditions = [E3Condition::Baseline, E3Condition::NoRecharge];

    let mut long_csv = String::from(
        "condition,seed,life_id,agent_id,birth_step,death_step,lifetime_steps,c_state01_birth,c_state01_firstk,avg_c_state01_tick,c_state01_std_over_life,avg_c_state01_attack,attack_tick_count\n",
    );
    let mut summary_csv = String::from(
        "condition,seed,n_deaths,pearson_r_firstk,pearson_p_firstk,spearman_rho_firstk,spearman_p_firstk,logrank_p_firstk,logrank_p_firstk_q25q75,median_high_firstk,median_low_firstk,pearson_r_birth,pearson_p_birth,spearman_rho_birth,spearman_p_birth,pearson_r_attack,pearson_p_attack,spearman_rho_attack,spearman_p_attack,n_attack_lives\n",
    );
    let mut policy_csv = String::from(
        "condition,dt_sec,basal_cost_per_sec,action_cost_per_attack,recharge_per_attack\n",
    );
    for condition in conditions {
        let params = e3_policy_params(condition);
        policy_csv.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6}\n",
            params.condition,
            params.dt_sec,
            params.basal_cost_per_sec,
            params.action_cost_per_attack,
            params.recharge_per_attack
        ));
    }
    write(out_dir.join("paper_e3_policy_params.csv"), policy_csv)?;
    write(
        out_dir.join("paper_e3_metric_definition.txt"),
        e3_metric_definition_text(),
    )?;

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
                    "{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
                    rec.condition,
                    rec.seed,
                    rec.life_id,
                    rec.agent_id,
                    rec.birth_step,
                    rec.death_step,
                    rec.lifetime_steps,
                    rec.c_state_birth,
                    rec.c_state_firstk,
                    rec.avg_c_state_tick,
                    rec.c_state_std_over_life,
                    rec.avg_c_state_attack,
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
                "E3 C_state01_firstK vs Lifetime",
                "C_state01_firstK",
                &arrays.c_state_firstk,
                &arrays.lifetimes,
                seed ^ 0xE301_u64,
            )?;

            let scatter_birth_path = out_dir.join(format!(
                "paper_e3_birth_vs_lifetime_seed{}_{}.png",
                seed, cond_label
            ));
            let corr_stats_birth = render_e3_scatter_with_stats(
                &scatter_birth_path,
                "E3 C_state01_birth vs Lifetime",
                "C_state01_birth",
                &arrays.c_state_birth,
                &arrays.lifetimes,
                seed ^ 0xE302_u64,
            )?;

            let survival_path = out_dir.join(format!(
                "paper_e3_survival_by_firstk_seed{}_{}.png",
                seed, cond_label
            ));
            let surv_firstk_stats = render_survival_split_plot(
                &survival_path,
                "E3 Survival by C_state01_firstK (median split)",
                &arrays.lifetimes,
                &arrays.c_state_firstk,
                SplitKind::Median,
                seed ^ 0xE310_u64,
            )?;

            let survival_q_path = out_dir.join(format!(
                "paper_e3_survival_by_firstk_q25q75_seed{}_{}.png",
                seed, cond_label
            ));
            let surv_firstk_q_stats = render_survival_split_plot(
                &survival_q_path,
                "E3 Survival by C_state01_firstK (q25 vs q75)",
                &arrays.lifetimes,
                &arrays.c_state_firstk,
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
                    "E3 C_state01_attack vs Lifetime",
                    "C_state01_attack",
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
                    "E3 Survival by C_state01_attack (median split)",
                    &attack_lifetimes,
                    &attack_vals,
                    SplitKind::Median,
                    seed ^ 0xE313_u64,
                )?;
            }

            if cond_label == "baseline" && seed == E3_SEEDS[0] {
                let mut legacy_csv = String::from("life_id,lifetime_steps,c_state01_firstk\n");
                let mut legacy_deaths = Vec::with_capacity(deaths.len());
                for d in &deaths {
                    legacy_csv.push_str(&format!(
                        "{},{},{:.6}\n",
                        d.life_id, d.lifetime_steps, d.c_state_firstk
                    ));
                    legacy_deaths.push((d.life_id as usize, d.lifetime_steps, d.c_state_firstk));
                }
                write(out_dir.join("paper_e3_lifetimes.csv"), legacy_csv)?;
                let legacy_scatter = out_dir.join("paper_e3_firstk_vs_lifetime.png");
                render_consonance_lifetime_scatter(&legacy_scatter, &legacy_deaths)?;
                let legacy_scatter_alias = out_dir.join("paper_e3_consonance_vs_lifetime.png");
                std::fs::copy(&legacy_scatter, &legacy_scatter_alias)?;
                let legacy_survival = out_dir.join("paper_e3_survival_curve.png");
                render_survival_curve(&legacy_survival, &legacy_deaths)?;
                let legacy_survival_c_state = out_dir.join("paper_e3_survival_by_c_state.png");
                render_survival_by_c_state(&legacy_survival_c_state, &legacy_deaths)?;
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
                &base.arrays.c_state_firstk,
                &base.arrays.lifetimes,
                rep_seed ^ 0xE301_u64,
            );
            let norecharge_scatter = build_scatter_data(
                &norecharge.arrays.c_state_firstk,
                &norecharge.arrays.lifetimes,
                rep_seed ^ 0xE301_u64,
            );
            let compare_scatter = out_dir.join("paper_e3_firstk_scatter_compare.png");
            render_scatter_compare(
                &compare_scatter,
                "E3 C_state01_firstK vs Lifetime",
                "C_state01_firstK",
                "Baseline",
                &base_scatter,
                "NoRecharge",
                &norecharge_scatter,
            )?;

            let base_surv = build_survival_data(
                &base.arrays.lifetimes,
                &base.arrays.c_state_firstk,
                SplitKind::Median,
                rep_seed ^ 0xE310_u64,
            );
            let norecharge_surv = build_survival_data(
                &norecharge.arrays.lifetimes,
                &norecharge.arrays.c_state_firstk,
                SplitKind::Median,
                rep_seed ^ 0xE310_u64,
            );
            let compare_surv = out_dir.join("paper_e3_firstk_survival_compare.png");
            render_survival_compare(
                &compare_surv,
                "E3 Survival by C_state01_firstK (median split)",
                "Baseline",
                &base_surv,
                "NoRecharge",
                &norecharge_surv,
            )?;

            let base_surv_q = build_survival_data(
                &base.arrays.lifetimes,
                &base.arrays.c_state_firstk,
                SplitKind::Quartiles,
                rep_seed ^ 0xE311_u64,
            );
            let norecharge_surv_q = build_survival_data(
                &norecharge.arrays.lifetimes,
                &norecharge.arrays.c_state_firstk,
                SplitKind::Quartiles,
                rep_seed ^ 0xE311_u64,
            );
            let compare_surv_q = out_dir.join("paper_e3_firstk_survival_compare_q25q75.png");
            render_survival_compare(
                &compare_surv_q,
                "E3 Survival by C_state01_firstK (q25 vs q75)",
                "Baseline",
                &base_surv_q,
                "NoRecharge",
                &norecharge_surv_q,
            )?;
        }
    }

    let pooled_baseline = e3_pooled_arrays(&seed_outputs, E3Condition::Baseline);
    let pooled_norecharge = e3_pooled_arrays(&seed_outputs, E3Condition::NoRecharge);
    let pooled_scatter_path = out_dir.join("paper_e3_firstk_scatter_compare_pooled.png");
    let pooled_base_scatter = build_scatter_data(
        &pooled_baseline.c_state_firstk,
        &pooled_baseline.lifetimes,
        0xE3B0_u64,
    );
    let pooled_nore_scatter = build_scatter_data(
        &pooled_norecharge.c_state_firstk,
        &pooled_norecharge.lifetimes,
        0xE3B1_u64,
    );
    render_scatter_compare(
        &pooled_scatter_path,
        "E3 C_state01_firstK vs Lifetime (pooled)",
        "C_state01_firstK",
        "Baseline",
        &pooled_base_scatter,
        "NoRecharge",
        &pooled_nore_scatter,
    )?;

    let pooled_surv_base = build_survival_data(
        &pooled_baseline.lifetimes,
        &pooled_baseline.c_state_firstk,
        SplitKind::Median,
        0xE3B2_u64,
    );
    let pooled_surv_nore = build_survival_data(
        &pooled_norecharge.lifetimes,
        &pooled_norecharge.c_state_firstk,
        SplitKind::Median,
        0xE3B3_u64,
    );
    let pooled_surv_path = out_dir.join("paper_e3_firstk_survival_compare_pooled.png");
    render_survival_compare(
        &pooled_surv_path,
        "E3 Survival by C_state01_firstK (median split, pooled)",
        "Baseline",
        &pooled_surv_base,
        "NoRecharge",
        &pooled_surv_nore,
    )?;

    let pooled_surv_base_q = build_survival_data(
        &pooled_baseline.lifetimes,
        &pooled_baseline.c_state_firstk,
        SplitKind::Quartiles,
        0xE3B4_u64,
    );
    let pooled_surv_nore_q = build_survival_data(
        &pooled_norecharge.lifetimes,
        &pooled_norecharge.c_state_firstk,
        SplitKind::Quartiles,
        0xE3B5_u64,
    );
    let pooled_surv_q_path = out_dir.join("paper_e3_firstk_survival_compare_pooled_q25q75.png");
    render_survival_compare(
        &pooled_surv_q_path,
        "E3 Survival by C_state01_firstK (q25 vs q75, pooled)",
        "Baseline",
        &pooled_surv_base_q,
        "NoRecharge",
        &pooled_surv_nore_q,
    )?;

    let mut pooled_summary_csv = String::from(
        "condition,n,pearson_r_firstk,pearson_p_firstk,spearman_rho_firstk,spearman_p_firstk,logrank_p_firstk,median_high_firstk,median_low_firstk,median_diff_firstk\n",
    );
    for (condition, scatter, survival) in [
        (
            E3Condition::Baseline,
            &pooled_base_scatter,
            &pooled_surv_base,
        ),
        (
            E3Condition::NoRecharge,
            &pooled_nore_scatter,
            &pooled_surv_nore,
        ),
    ] {
        let median_diff = survival.stats.median_high - survival.stats.median_low;
        pooled_summary_csv.push_str(&format!(
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3},{:.3},{:.3}\n",
            condition.label(),
            scatter.stats.n,
            scatter.stats.pearson_r,
            scatter.stats.pearson_p,
            scatter.stats.spearman_rho,
            scatter.stats.spearman_p,
            survival.stats.logrank_p,
            survival.stats.median_high,
            survival.stats.median_low,
            median_diff
        ));
    }
    write(
        out_dir.join("paper_e3_summary_pooled.csv"),
        pooled_summary_csv,
    )?;

    let pooled_hist_path = out_dir.join("paper_e3_firstk_hist.png");
    render_e3_firstk_histogram(
        &pooled_hist_path,
        &pooled_baseline.c_state_firstk,
        &pooled_norecharge.c_state_firstk,
        0.02,
        0.5,
    )?;

    write(out_dir.join("paper_e3_lifetimes_long.csv"), long_csv)?;
    write(out_dir.join("paper_e3_summary_by_seed.csv"), summary_csv)?;

    Ok(())
}

fn e3_extract_arrays(deaths: &[E3DeathRecord]) -> E3Arrays {
    let mut lifetimes = Vec::with_capacity(deaths.len());
    let mut c_state_birth = Vec::with_capacity(deaths.len());
    let mut c_state_firstk = Vec::with_capacity(deaths.len());
    let mut avg_attack = Vec::with_capacity(deaths.len());
    let mut attack_tick_count = Vec::with_capacity(deaths.len());
    for d in deaths {
        lifetimes.push(d.lifetime_steps);
        c_state_birth.push(d.c_state_birth);
        c_state_firstk.push(d.c_state_firstk);
        avg_attack.push(d.avg_c_state_attack);
        attack_tick_count.push(d.attack_tick_count);
    }
    E3Arrays {
        lifetimes,
        c_state_birth,
        c_state_firstk,
        avg_attack,
        attack_tick_count,
    }
}

fn e3_pooled_arrays(outputs: &[E3SeedOutput], condition: E3Condition) -> E3Arrays {
    let mut lifetimes = Vec::new();
    let mut c_state_birth = Vec::new();
    let mut c_state_firstk = Vec::new();
    let mut avg_attack = Vec::new();
    let mut attack_tick_count = Vec::new();
    for output in outputs.iter().filter(|o| o.condition == condition) {
        lifetimes.extend(output.arrays.lifetimes.iter().copied());
        c_state_birth.extend(output.arrays.c_state_birth.iter().copied());
        c_state_firstk.extend(output.arrays.c_state_firstk.iter().copied());
        avg_attack.extend(output.arrays.avg_attack.iter().copied());
        attack_tick_count.extend(output.arrays.attack_tick_count.iter().copied());
    }
    E3Arrays {
        lifetimes,
        c_state_birth,
        c_state_firstk,
        avg_attack,
        attack_tick_count,
    }
}

fn fractions_from_counts(counts: &[(f32, f32)]) -> Vec<f32> {
    let total: f32 = counts.iter().map(|(_, v)| *v).sum();
    let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
    counts.iter().map(|(_, v)| v * inv).collect()
}

fn render_e3_firstk_histogram(
    out_path: &Path,
    baseline: &[f32],
    norecharge: &[f32],
    bin_width: f32,
    threshold: f32,
) -> Result<(), Box<dyn Error>> {
    let min = 0.0f32;
    let max = 1.0f32;
    let counts_base = histogram_counts_fixed(baseline, min, max, bin_width);
    let counts_nore = histogram_counts_fixed(norecharge, min, max, bin_width);
    let len = counts_base.len().min(counts_nore.len());
    if len == 0 {
        return Ok(());
    }
    let centers: Vec<f32> = counts_base.iter().take(len).map(|(c, _)| *c).collect();
    let base_frac = fractions_from_counts(&counts_base[..len]);
    let nore_frac = fractions_from_counts(&counts_nore[..len]);

    let mut y_max = 0.0f32;
    for &v in base_frac.iter().chain(nore_frac.iter()) {
        y_max = y_max.max(v);
    }
    y_max = y_max.max(1e-3);

    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("E3 C_state01_firstK Histogram (pooled)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min..max, 0.0f32..(y_max * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("C_state01_firstK")
        .y_desc("fraction")
        .x_labels(10)
        .draw()?;

    let base_line = centers.iter().copied().zip(base_frac.iter().copied());
    chart
        .draw_series(LineSeries::new(base_line, BLUE))?
        .label("baseline")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
    let nore_line = centers.iter().copied().zip(nore_frac.iter().copied());
    chart
        .draw_series(LineSeries::new(nore_line, RED))?
        .label("no recharge")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    let thresh = threshold.clamp(min, max);
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(thresh, 0.0), (thresh, y_max * 1.05)],
        BLACK.mix(0.6),
    )))?;

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn e3_lifetimes_csv(deaths: &[E3DeathRecord]) -> String {
    let mut out = String::from(
        "life_id,agent_id,birth_step,death_step,lifetime_steps,c_state01_birth,c_state01_firstk,avg_c_state01_tick,c_state01_std_over_life,avg_c_state01_attack,attack_tick_count\n",
    );
    for d in deaths {
        out.push_str(&format!(
            "{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
            d.life_id,
            d.agent_id,
            d.birth_step,
            d.death_step,
            d.lifetime_steps,
            d.c_state_birth,
            d.c_state_firstk,
            d.avg_c_state_tick,
            d.c_state_std_over_life,
            d.avg_c_state_attack,
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

    let seed_rows = e5_seed_sweep_rows(E5_STEPS);
    let seed_csv_path = out_dir.join("paper_e5_seed_sweep.csv");
    write(&seed_csv_path, e5_seed_sweep_csv(&seed_rows))?;
    let seed_plot_path = out_dir.join("paper_e5_seed_sweep.png");
    render_e5_seed_sweep_plot(&seed_plot_path, &seed_rows)?;

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
    perc_c_score_scan: &[f32],
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
        c_points.push((x, perc_c_score_scan[i]));
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
        .caption("E1 Consonance Score C(f)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, (c_min - pad)..(c_max + pad))?;

    chart_c
        .configure_mesh()
        .x_desc("log2(f / f_anchor)")
        .y_desc("C_score")
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

fn plot_e4_mirror_sweep(
    out_dir: &Path,
    anchor_hz: f32,
    emit_hist_files: bool,
) -> Result<(), Box<dyn Error>> {
    let coarse_weights = build_weight_grid(E4_WEIGHT_COARSE_STEP);
    let primary_bin = E4_BIN_WIDTHS[0];
    let (mut run_records, mut hist_records, mut tail_rows) = run_e4_sweep_for_weights(
        out_dir,
        anchor_hz,
        &coarse_weights,
        &E4_SEEDS,
        primary_bin,
        emit_hist_files,
    )?;

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
        let (more_runs, more_hists, more_tail) = run_e4_sweep_for_weights(
            out_dir,
            anchor_hz,
            &fine_only,
            &E4_SEEDS,
            primary_bin,
            emit_hist_files,
        )?;
        run_records.extend(more_runs);
        hist_records.extend(more_hists);
        tail_rows.extend(more_tail);
    }

    for &bin_width in E4_BIN_WIDTHS.iter().skip(1) {
        let (more_runs, more_hists, more_tail) = run_e4_sweep_for_weights(
            out_dir,
            anchor_hz,
            &weights,
            &E4_SEEDS,
            bin_width,
            emit_hist_files,
        )?;
        run_records.extend(more_runs);
        hist_records.extend(more_hists);
        tail_rows.extend(more_tail);
    }

    let runs_csv_path = out_dir.join("e4_mirror_sweep_runs.csv");
    write(&runs_csv_path, e4_runs_csv(&run_records))?;
    let tail_csv_path = out_dir.join("paper_e4_tail_interval_timeseries.csv");
    write(&tail_csv_path, e4_tail_interval_csv(&tail_rows))?;

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

        let third_mass_path = out_dir.join(format!(
            "paper_e4_third_mass_vs_weight_bw{}.png",
            format_float_token(bin_width)
        ));
        render_e4_third_mass_plot(&third_mass_path, &summaries, bin_width)?;

        let rate_path = out_dir.join(format!(
            "paper_e4_major_minor_rate_vs_weight_bw{}.png",
            format_float_token(bin_width)
        ));
        render_e4_major_minor_rate_plot(&rate_path, &summaries, bin_width)?;
    }

    let legacy_path = out_dir.join("paper_e4_mirror_sweep.png");
    render_e4_major_minor_plot(&legacy_path, &summaries, primary_bin)?;

    Ok(())
}

struct ConsonanceWorkspace {
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
    mean_c_series: Vec<f32>,
    mean_c_state_series: Vec<f32>,
    mean_c_score_loo_series: Vec<f32>,
    mean_c_score_chosen_loo_series: Vec<f32>,
    mean_score_series: Vec<f32>,
    mean_repulsion_series: Vec<f32>,
    moved_frac_series: Vec<f32>,
    accepted_worse_frac_series: Vec<f32>,
    attempted_update_frac_series: Vec<f32>,
    moved_given_attempt_frac_series: Vec<f32>,
    mean_abs_delta_semitones_series: Vec<f32>,
    mean_abs_delta_semitones_moved_series: Vec<f32>,
    semitone_samples_pre: Vec<f32>,
    semitone_samples_post: Vec<f32>,
    final_semitones: Vec<f32>,
    final_freqs_hz: Vec<f32>,
    final_log2_ratios: Vec<f32>,
    trajectory_semitones: Vec<Vec<f32>>,
    trajectory_c_state: Vec<Vec<f32>>,
    anchor_shift: E2AnchorShiftStats,
    density_mass_mean: f32,
    density_mass_min: f32,
    density_mass_max: f32,
    r_state01_min: f32,
    r_state01_mean: f32,
    r_state01_max: f32,
    r_ref_peak: f32,
    roughness_k: f32,
    roughness_ref_eps: f32,
    n_agents: usize,
    k_bins: i32,
}

struct E2SweepStats {
    mean_c: Vec<f32>,
    std_c: Vec<f32>,
    mean_c_state: Vec<f32>,
    std_c_state: Vec<f32>,
    mean_c_score_loo: Vec<f32>,
    std_c_score_loo: Vec<f32>,
    mean_score: Vec<f32>,
    std_score: Vec<f32>,
    mean_repulsion: Vec<f32>,
    std_repulsion: Vec<f32>,
    n: usize,
}

struct E3Arrays {
    lifetimes: Vec<u32>,
    c_state_birth: Vec<f32>,
    c_state_firstk: Vec<f32>,
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

#[derive(Clone, Copy)]
struct HistStructureMetrics {
    entropy: f32,
    gini: f32,
    peakiness: f32,
    kl_uniform: f32,
}

struct HistStructureRow {
    condition: &'static str,
    seed: u64,
    metrics: HistStructureMetrics,
}

#[derive(Clone, Copy)]
struct DiversityMetrics {
    unique_bins: usize,
    nn_mean: f32,
    nn_std: f32,
    semitone_var: f32,
    semitone_mad: f32,
}

struct DiversityRow {
    condition: &'static str,
    seed: u64,
    metrics: DiversityMetrics,
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
    mass_min3: f32,
    mass_maj3: f32,
    mass_p4: f32,
    mass_p5: f32,
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
    mean_min3: f32,
    std_min3: f32,
    mean_maj3: f32,
    std_maj3: f32,
    mean_p4: f32,
    std_p4: f32,
    mean_p5: f32,
    std_p5: f32,
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

struct E4TailIntervalRow {
    mirror_weight: f32,
    seed: u64,
    bin_width: f32,
    eps: f32,
    step: u32,
    mass_min3: f32,
    mass_maj3: f32,
    mass_p4: f32,
    mass_p5: f32,
    delta: f32,
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

#[derive(Clone, Copy)]
struct IntervalMasses {
    min3: f32,
    maj3: f32,
    p4: f32,
    p5: f32,
}

fn interval_masses_from_freqs(anchor_hz: f32, freqs: &[f32], eps_st: f32) -> IntervalMasses {
    if !anchor_hz.is_finite() || anchor_hz <= 0.0 || freqs.is_empty() {
        return IntervalMasses {
            min3: 0.0,
            maj3: 0.0,
            p4: 0.0,
            p5: 0.0,
        };
    }
    let eps = eps_st.max(1e-6);
    let mut count_min3 = 0u32;
    let mut count_maj3 = 0u32;
    let mut count_p4 = 0u32;
    let mut count_p5 = 0u32;
    let mut total = 0u32;
    for &freq in freqs {
        if !freq.is_finite() || freq <= 0.0 {
            continue;
        }
        let ratio = freq / anchor_hz;
        if !ratio.is_finite() || ratio <= 0.0 {
            continue;
        }
        let semitones = 12.0 * ratio.log2();
        let folded = semitone_fold_to_octave(semitones);
        total += 1;
        if (folded - 3.0).abs() <= eps {
            count_min3 += 1;
        }
        if (folded - 4.0).abs() <= eps {
            count_maj3 += 1;
        }
        if (folded - 5.0).abs() <= eps {
            count_p4 += 1;
        }
        if (folded - 7.0).abs() <= eps {
            count_p5 += 1;
        }
    }
    if total == 0 {
        return IntervalMasses {
            min3: 0.0,
            maj3: 0.0,
            p4: 0.0,
            p5: 0.0,
        };
    }
    let inv = 1.0 / total as f32;
    IntervalMasses {
        min3: count_min3 as f32 * inv,
        maj3: count_maj3 as f32 * inv,
        p4: count_p4 as f32 * inv,
        p5: count_p5 as f32 * inv,
    }
}

fn run_e4_sweep_for_weights(
    out_dir: &Path,
    anchor_hz: f32,
    weights: &[f32],
    seeds: &[u64],
    bin_width: f32,
    emit_hist_files: bool,
) -> Result<(Vec<E4RunRecord>, Vec<E4HistRecord>, Vec<E4TailIntervalRow>), Box<dyn Error>> {
    let mut runs = Vec::new();
    let mut hists = Vec::new();
    let mut tail_rows = Vec::new();
    let eps = bin_width.max(1e-6);
    for &weight in weights {
        for &seed in seeds {
            let samples = run_e4_condition_tail_samples(weight, seed, E4_TAIL_WINDOW_STEPS);
            let semitone_samples = collect_e4_semitone_samples(&samples, anchor_hz);
            let histogram = histogram_from_samples(&semitone_samples, 0.0, 12.0, bin_width);

            let mass_min3 = mass_around(&histogram, 3.0, eps);
            let mass_maj3 = mass_around(&histogram, 4.0, eps);
            let mass_p4 = mass_around(&histogram, 5.0, eps);
            let mass_p5 = mass_around(&histogram, 7.0, eps);
            let major_score = mass_maj3 + mass_p5;
            let minor_score = mass_min3 + mass_p5;
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
                mass_min3,
                mass_maj3,
                mass_p4,
                mass_p5,
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

            for (i, freqs) in samples.freqs_by_step.iter().enumerate() {
                let masses = interval_masses_from_freqs(anchor_hz, freqs, eps);
                let step = samples.steps_total.saturating_sub(samples.tail_window) + i as u32;
                tail_rows.push(E4TailIntervalRow {
                    mirror_weight: weight,
                    seed,
                    bin_width,
                    eps,
                    step,
                    mass_min3: masses.min3,
                    mass_maj3: masses.maj3,
                    mass_p4: masses.p4,
                    mass_p5: masses.p5,
                    delta: masses.maj3 - masses.min3,
                });
            }

            if emit_hist_files {
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
                let caption = format!(
                    "E4 Interval Histogram (w={weight:.2}, seed={seed}, bw={bin_width:.2})"
                );
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
    }
    Ok((runs, hists, tail_rows))
}

fn e4_runs_csv(records: &[E4RunRecord]) -> String {
    let mut out = String::from(
        "mirror_weight,seed,bin_width,eps,major_score,minor_score,delta,major_frac,mass_m3,mass_M3,mass_P4,mass_P5,steps_total,burn_in,tail_window,histogram_source\n",
    );
    for record in records {
        out.push_str(&format!(
            "{:.3},{},{:.3},{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{}\n",
            record.mirror_weight,
            record.seed,
            record.bin_width,
            record.eps,
            record.major_score,
            record.minor_score,
            record.delta,
            record.major_frac,
            record.mass_min3,
            record.mass_maj3,
            record.mass_p4,
            record.mass_p5,
            record.steps_total,
            record.burn_in,
            record.tail_window,
            record.histogram_source
        ));
    }
    out
}

fn e4_tail_interval_csv(rows: &[E4TailIntervalRow]) -> String {
    let mut out = String::from(
        "mirror_weight,seed,bin_width,eps,step,mass_m3,mass_M3,mass_P4,mass_P5,delta\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{:.3},{},{:.3},{:.3},{},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            row.mirror_weight,
            row.seed,
            row.bin_width,
            row.eps,
            row.step,
            row.mass_min3,
            row.mass_maj3,
            row.mass_p4,
            row.mass_p5,
            row.delta
        ));
    }
    out
}

fn e4_summary_csv(records: &[E4SummaryRecord]) -> String {
    let mut out = String::from(
        "mirror_weight,bin_width,eps,mean_major,std_major,mean_minor,std_minor,mean_delta,std_delta,mean_m3,std_m3,mean_M3,std_M3,mean_P4,std_P4,mean_P5,std_P5,major_rate,minor_rate,ambiguous_rate,n_runs\n",
    );
    for record in records {
        out.push_str(&format!(
            "{:.3},{:.3},{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3},{:.3},{:.3},{}\n",
            record.mirror_weight,
            record.bin_width,
            record.eps,
            record.mean_major,
            record.std_major,
            record.mean_minor,
            record.std_minor,
            record.mean_delta,
            record.std_delta,
            record.mean_min3,
            record.std_min3,
            record.mean_maj3,
            record.std_maj3,
            record.mean_p4,
            record.std_p4,
            record.mean_p5,
            record.std_p5,
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
        let min3_values: Vec<f32> = runs.iter().map(|r| r.mass_min3).collect();
        let maj3_values: Vec<f32> = runs.iter().map(|r| r.mass_maj3).collect();
        let p4_values: Vec<f32> = runs.iter().map(|r| r.mass_p4).collect();
        let p5_values: Vec<f32> = runs.iter().map(|r| r.mass_p5).collect();
        let (mean_major, std_major) = mean_std(&major_values);
        let (mean_minor, std_minor) = mean_std(&minor_values);
        let (mean_delta, std_delta) = mean_std(&delta_values);
        let (mean_min3, std_min3) = mean_std(&min3_values);
        let (mean_maj3, std_maj3) = mean_std(&maj3_values);
        let (mean_p4, std_p4) = mean_std(&p4_values);
        let (mean_p5, std_p5) = mean_std(&p5_values);
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
            mean_min3,
            std_min3,
            mean_maj3,
            std_maj3,
            mean_p4,
            std_p4,
            mean_p5,
            std_p5,
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

fn render_e4_third_mass_plot(
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
                s.mean_maj3,
                s.std_maj3,
                s.mean_min3,
                s.std_min3,
            )
        })
        .collect();
    if series.is_empty() {
        return Ok(());
    }
    series.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut y_max = series
        .iter()
        .map(|(_, m_maj, s_maj, m_min, s_min)| (m_maj + s_maj).max(m_min + s_min))
        .fold(0.0f32, f32::max);
    if y_max <= 0.0 {
        y_max = 1.0;
    }
    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("E4 Third Mass vs Mirror Weight (bw={bin_width:.2})"),
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, 0.0f32..(y_max * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("mirror weight")
        .y_desc("mass")
        .draw()?;

    let cap = 0.01f32;
    for (w, mean_maj, std_maj, mean_min, std_min) in &series {
        let y0 = mean_maj - std_maj;
        let y1 = mean_maj + std_maj;
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
            (*w, *mean_maj),
            3,
            BLUE.filled(),
        )))?;

        let y0 = mean_min - std_min;
        let y1 = mean_min + std_min;
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
            (*w, *mean_min),
            3,
            RED.filled(),
        )))?;
    }

    root.present()?;
    Ok(())
}

fn render_e4_major_minor_rate_plot(
    out_path: &Path,
    summaries: &[E4SummaryRecord],
    bin_width: f32,
) -> Result<(), Box<dyn Error>> {
    let mut series: Vec<(f32, f32, f32)> = summaries
        .iter()
        .filter(|s| (s.bin_width - bin_width).abs() < 1e-6)
        .map(|s| (s.mirror_weight, s.major_rate, s.minor_rate))
        .collect();
    if series.is_empty() {
        return Ok(());
    }
    series.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("E4 Major/Minor Rate vs Mirror Weight (bw={bin_width:.2})"),
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, 0.0f32..1.0f32)?;

    chart
        .configure_mesh()
        .x_desc("mirror weight")
        .y_desc("rate (Δ thresholded)")
        .draw()?;

    for (w, major_rate, minor_rate) in &series {
        chart.draw_series(std::iter::once(Circle::new(
            (*w, *major_rate),
            3,
            BLUE.filled(),
        )))?;
        chart.draw_series(std::iter::once(Circle::new(
            (*w, *minor_rate),
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

fn build_consonance_workspace(space: &Log2Space) -> ConsonanceWorkspace {
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
        consonance_harmonicity_weight: 1.0,
        consonance_roughness_weight_floor: 0.35,
        consonance_roughness_weight: 0.5,
        c_state_beta: E2_C_STATE_BETA,
        c_state_theta: E2_C_STATE_THETA,
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
    ConsonanceWorkspace {
        params,
        r_ref_peak: r_ref.peak,
    }
}

fn r_state01_stats(scan: &[f32]) -> RState01Stats {
    if scan.is_empty() {
        return RState01Stats {
            min: 0.0,
            mean: 0.0,
            max: 0.0,
        };
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for &value in scan {
        if !value.is_finite() {
            continue;
        }
        min = min.min(value);
        max = max.max(value);
        sum += value;
        count += 1;
    }
    if count == 0 {
        return RState01Stats {
            min: 0.0,
            mean: 0.0,
            max: 0.0,
        };
    }
    let mean = sum / count as f32;
    RState01Stats {
        min: min.clamp(0.0, 1.0),
        mean: mean.clamp(0.0, 1.0),
        max: max.clamp(0.0, 1.0),
    }
}

fn compute_c_score_state_scans(
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    env_scan: &[f32],
    density_scan: &[f32],
    du_scan: &[f32],
) -> (Vec<f32>, Vec<f32>, f32, RState01Stats) {
    space.assert_scan_len_named(du_scan, "du_scan");
    // Use the same epsilon as the roughness reference normalization so density scaling stays aligned.
    let (density_norm, density_mass) =
        psycho_state::normalize_density(density_scan, du_scan, workspace.params.roughness_ref_eps);
    let (perc_h_pot_scan, _) = workspace
        .params
        .harmonicity_kernel
        .potential_h_from_log2_spectrum(env_scan, space);
    let (perc_r_pot_scan, _) = workspace
        .params
        .roughness_kernel
        .potential_r_from_log2_spectrum_density(&density_norm, space);

    let mut perc_r_state01_scan = vec![0.0f32; space.n_bins()];
    psycho_state::r_pot_scan_to_r_state01_scan(
        &perc_r_pot_scan,
        workspace.r_ref_peak,
        workspace.params.roughness_k,
        &mut perc_r_state01_scan,
    );
    let r_state_stats = r_state01_stats(&perc_r_state01_scan);

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

    let mut c_score_scan = vec![0.0f32; space.n_bins()];
    let mut c_state_scan = vec![0.0f32; space.n_bins()];
    let alpha_h = workspace.params.consonance_harmonicity_weight;
    let w0 = workspace.params.consonance_roughness_weight_floor;
    let w1 = workspace.params.consonance_roughness_weight;
    for i in 0..space.n_bins() {
        let c_score = psycho_state::compose_c_score(
            alpha_h,
            w0,
            w1,
            perc_h_state01_scan[i],
            perc_r_state01_scan[i],
        );
        let c_state = psycho_state::compose_c_state(
            workspace.params.c_state_beta,
            workspace.params.c_state_theta,
            c_score,
        );
        c_score_scan[i] = c_score;
        c_state_scan[i] = c_state.clamp(0.0, 1.0);
    }
    (c_score_scan, c_state_scan, density_mass, r_state_stats)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn update_agent_indices(
    indices: &mut [usize],
    c_score_scan: &[f32],
    log2_ratio_scan: &[f32],
    min_idx: usize,
    max_idx: usize,
    k: i32,
    lambda: f32,
    sigma: f32,
) {
    let _ = update_agent_indices_scored(
        indices,
        c_score_scan,
        log2_ratio_scan,
        min_idx,
        max_idx,
        k,
        lambda,
        sigma,
    );
}

struct UpdateStats {
    mean_c_score_current_loo: f32,
    mean_c_score_chosen_loo: f32,
    mean_score: f32,
    mean_repulsion: f32,
    moved_frac: f32,
    accepted_worse_frac: f32,
    attempted_update_frac: f32,
    moved_given_attempt_frac: f32,
    mean_abs_delta_semitones: f32,
    mean_abs_delta_semitones_moved: f32,
}

#[derive(Clone, Copy, Debug)]
struct RState01Stats {
    min: f32,
    mean: f32,
    max: f32,
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
    c_score_scan: &[f32],
    log2_ratio_scan: &[f32],
    min_idx: usize,
    max_idx: usize,
    k: i32,
    lambda: f32,
    sigma: f32,
) -> f32 {
    update_agent_indices_scored_stats(
        indices,
        c_score_scan,
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
    c_score_scan: &[f32],
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
        c_score_scan,
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
    c_score_scan: &[f32],
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
            mean_c_score_current_loo: 0.0,
            mean_c_score_chosen_loo: 0.0,
            mean_score: 0.0,
            mean_repulsion: 0.0,
            moved_frac: 0.0,
            accepted_worse_frac: 0.0,
            attempted_update_frac: 0.0,
            moved_given_attempt_frac: 0.0,
            mean_abs_delta_semitones: 0.0,
            mean_abs_delta_semitones_moved: 0.0,
        };
    }
    let sigma = sigma.max(1e-6);
    let skip_repulsion = lambda <= 0.0;
    let prev_indices = indices.to_vec();
    let prev_log2: Vec<f32> = prev_indices
        .iter()
        .map(|&idx| log2_ratio_scan[idx])
        .collect();
    let mut next_indices = prev_indices.clone();
    let mut score_sum = 0.0f32;
    let mut score_count = 0u32;
    let mut repulsion_sum = 0.0f32;
    let mut repulsion_count = 0u32;
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
            if !skip_repulsion {
                for (j, &other_log2) in prev_log2.iter().enumerate() {
                    if j == agent_i {
                        continue;
                    }
                    let dist = (cand_log2 - other_log2).abs();
                    repulsion += (-dist / sigma).exp();
                }
            }
            let c_score = c_score_scan[cand];
            let score = c_score - lambda * repulsion;
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
            score_count += 1;
        }
        if best_repulsion.is_finite() {
            repulsion_sum += best_repulsion;
            repulsion_count += 1;
        }
        count += 1;
    }

    indices.copy_from_slice(&next_indices);
    if count == 0 {
        return UpdateStats {
            mean_c_score_current_loo: 0.0,
            mean_c_score_chosen_loo: 0.0,
            mean_score: 0.0,
            mean_repulsion: 0.0,
            moved_frac: 0.0,
            accepted_worse_frac: 0.0,
            attempted_update_frac: 0.0,
            moved_given_attempt_frac: 0.0,
            mean_abs_delta_semitones: 0.0,
            mean_abs_delta_semitones_moved: 0.0,
        };
    }
    let mean_score = if score_count > 0 {
        score_sum / score_count as f32
    } else {
        0.0
    };
    let mean_repulsion = if repulsion_count > 0 {
        repulsion_sum / repulsion_count as f32
    } else {
        0.0
    };
    let inv = 1.0 / count as f32;
    UpdateStats {
        mean_c_score_current_loo: f32::NAN,
        mean_c_score_chosen_loo: f32::NAN,
        mean_score,
        mean_repulsion,
        moved_frac: moved_count as f32 * inv,
        accepted_worse_frac: 0.0,
        attempted_update_frac: 1.0,
        moved_given_attempt_frac: moved_count as f32 * inv,
        mean_abs_delta_semitones: 0.0,
        mean_abs_delta_semitones_moved: 0.0,
    }
}

#[allow(clippy::too_many_arguments)]
fn update_agent_indices_scored_stats_loo(
    indices: &mut [usize],
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    env_total: &[f32],
    density_total: &[f32],
    du_scan: &[f32],
    log2_ratio_scan: &[f32],
    min_idx: usize,
    max_idx: usize,
    k: i32,
    score_sign: f32,
    lambda: f32,
    sigma: f32,
    temperature: f32,
    step: usize,
    block_backtrack: bool,
    prev_positions: Option<&[usize]>,
    rng: &mut StdRng,
) -> UpdateStats {
    let order: Vec<usize> = (0..indices.len()).collect();
    update_agent_indices_scored_stats_with_order_loo(
        indices,
        space,
        workspace,
        env_total,
        density_total,
        du_scan,
        log2_ratio_scan,
        min_idx,
        max_idx,
        k,
        score_sign,
        lambda,
        sigma,
        temperature,
        step,
        block_backtrack,
        prev_positions,
        rng,
        &order,
    )
}

fn metropolis_accept(delta: f32, temperature: f32, u01: f32) -> (bool, bool) {
    if !delta.is_finite() {
        return (false, false);
    }
    if delta >= 0.0 {
        return (true, false);
    }
    if temperature <= 0.0 {
        return (false, false);
    }
    let prob = (delta / temperature).exp();
    if u01 < prob {
        (true, true)
    } else {
        (false, false)
    }
}

#[allow(clippy::too_many_arguments)]
fn update_agent_indices_scored_stats_with_order_loo(
    indices: &mut [usize],
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    env_total: &[f32],
    density_total: &[f32],
    du_scan: &[f32],
    log2_ratio_scan: &[f32],
    min_idx: usize,
    max_idx: usize,
    k: i32,
    score_sign: f32,
    lambda: f32,
    sigma: f32,
    temperature: f32,
    step: usize,
    block_backtrack: bool,
    prev_positions: Option<&[usize]>,
    rng: &mut StdRng,
    order: &[usize],
) -> UpdateStats {
    if indices.is_empty() {
        return UpdateStats {
            mean_c_score_current_loo: 0.0,
            mean_c_score_chosen_loo: 0.0,
            mean_score: 0.0,
            mean_repulsion: 0.0,
            moved_frac: 0.0,
            accepted_worse_frac: 0.0,
            attempted_update_frac: 0.0,
            moved_given_attempt_frac: 0.0,
            mean_abs_delta_semitones: 0.0,
            mean_abs_delta_semitones_moved: 0.0,
        };
    }
    space.assert_scan_len_named(env_total, "env_total");
    space.assert_scan_len_named(density_total, "density_total");
    space.assert_scan_len_named(du_scan, "du_scan");
    let sigma = sigma.max(1e-6);
    let skip_repulsion = lambda <= 0.0;
    let prev_indices = indices.to_vec();
    let prev_log2: Vec<f32> = prev_indices
        .iter()
        .map(|&idx| log2_ratio_scan[idx])
        .collect();
    let u01_by_agent: Vec<f32> = (0..prev_indices.len())
        .map(|_| rng.random::<f32>())
        .collect();
    let u_move_by_agent: Vec<f32> = if matches!(E2_UPDATE_SCHEDULE, E2UpdateSchedule::Lazy) {
        (0..prev_indices.len())
            .map(|_| rng.random::<f32>())
            .collect()
    } else {
        vec![0.0; prev_indices.len()]
    };
    let mut env_loo = vec![0.0f32; env_total.len()];
    let mut density_loo = vec![0.0f32; density_total.len()];
    let mut next_indices = prev_indices.clone();
    let mut c_score_current_sum = 0.0f32;
    let mut c_score_current_count = 0u32;
    let mut c_score_chosen_sum = 0.0f32;
    let mut c_score_chosen_count = 0u32;
    let mut score_sum = 0.0f32;
    let mut score_count = 0u32;
    let mut repulsion_sum = 0.0f32;
    let mut repulsion_count = 0u32;
    let mut abs_delta_sum = 0.0f32;
    let mut abs_delta_moved_sum = 0.0f32;
    let mut attempt_count = 0usize;
    let mut moved_count = 0usize;
    let mut accepted_worse_count = 0usize;
    let mut count = 0usize;

    for &agent_i in order {
        if agent_i >= prev_indices.len() {
            continue;
        }
        let agent_idx = prev_indices[agent_i];
        env_loo.copy_from_slice(env_total);
        density_loo.copy_from_slice(density_total);
        env_loo[agent_idx] = (env_loo[agent_idx] - 1.0).max(0.0);
        let denom = du_scan[agent_idx].max(1e-12);
        density_loo[agent_idx] = (density_loo[agent_idx] - 1.0 / denom).max(0.0);
        let (c_score_scan, _, _, _) =
            compute_c_score_state_scans(space, workspace, &env_loo, &density_loo, du_scan);

        let current_idx = prev_indices[agent_i];
        let current_log2 = log2_ratio_scan[current_idx];
        let mut current_repulsion = 0.0f32;
        if !skip_repulsion {
            for (j, &other_log2) in prev_log2.iter().enumerate() {
                if j == agent_i {
                    continue;
                }
                let dist = (current_log2 - other_log2).abs();
                current_repulsion += (-dist / sigma).exp();
            }
        }
        let c_score_current = c_score_scan[current_idx];
        if c_score_current.is_finite() {
            c_score_current_sum += c_score_current;
            c_score_current_count += 1;
        }
        let current_score = score_sign * c_score_current - lambda * current_repulsion;
        let update_allowed = e2_should_attempt_update(agent_i, step, u_move_by_agent[agent_i]);
        if update_allowed {
            attempt_count += 1;
        }
        let backtrack_target = if block_backtrack {
            prev_positions.and_then(|prev| prev.get(agent_i).copied())
        } else {
            None
        };
        let (chosen_idx, chosen_score, chosen_repulsion, accepted_worse) = if update_allowed {
            let start = (current_idx as isize - k as isize).max(min_idx as isize) as usize;
            let end = (current_idx as isize + k as isize).min(max_idx as isize) as usize;
            let mut best_idx = current_idx;
            let mut best_score = f32::NEG_INFINITY;
            let mut best_repulsion = 0.0f32;
            let mut found_candidate = false;
            for cand in start..=end {
                if cand == current_idx {
                    continue;
                }
                let cand_log2 = log2_ratio_scan[cand];
                let mut repulsion = 0.0f32;
                if !skip_repulsion {
                    for (j, &other_log2) in prev_log2.iter().enumerate() {
                        if j == agent_i {
                            continue;
                        }
                        let dist = (cand_log2 - other_log2).abs();
                        repulsion += (-dist / sigma).exp();
                    }
                }
                let c_score = c_score_scan[cand];
                let score = score_sign * c_score - lambda * repulsion;
                if let Some(prev_idx) = backtrack_target {
                    if cand == prev_idx && (score - current_score) <= E2_BACKTRACK_ALLOW_EPS {
                        continue;
                    }
                }
                if score > best_score {
                    best_score = score;
                    best_idx = cand;
                    best_repulsion = repulsion;
                    found_candidate = true;
                }
            }
            if found_candidate {
                let delta = best_score - current_score;
                if delta > E2_SCORE_IMPROVE_EPS {
                    (best_idx, best_score, best_repulsion, false)
                } else if delta < 0.0 {
                    let u01 = u01_by_agent[agent_i];
                    let (accept, accepted_worse) = metropolis_accept(delta, temperature, u01);
                    if accept {
                        (best_idx, best_score, best_repulsion, accepted_worse)
                    } else {
                        (current_idx, current_score, current_repulsion, false)
                    }
                } else {
                    (current_idx, current_score, current_repulsion, false)
                }
            } else {
                (current_idx, current_score, current_repulsion, false)
            }
        } else {
            (current_idx, current_score, current_repulsion, false)
        };

        next_indices[agent_i] = chosen_idx;
        if chosen_idx != current_idx {
            moved_count += 1;
        }
        let delta_semitones = 12.0 * (log2_ratio_scan[chosen_idx] - log2_ratio_scan[current_idx]);
        let abs_delta = delta_semitones.abs();
        if abs_delta.is_finite() {
            abs_delta_sum += abs_delta;
            if chosen_idx != current_idx {
                abs_delta_moved_sum += abs_delta;
            }
        }
        if chosen_score.is_finite() {
            score_sum += chosen_score;
            score_count += 1;
        }
        if chosen_repulsion.is_finite() {
            repulsion_sum += chosen_repulsion;
            repulsion_count += 1;
        }
        let c_score_chosen = c_score_scan[chosen_idx];
        if c_score_chosen.is_finite() {
            c_score_chosen_sum += c_score_chosen;
            c_score_chosen_count += 1;
        }
        if accepted_worse {
            accepted_worse_count += 1;
        }
        count += 1;
    }

    indices.copy_from_slice(&next_indices);
    if count == 0 {
        return UpdateStats {
            mean_c_score_current_loo: 0.0,
            mean_c_score_chosen_loo: 0.0,
            mean_score: 0.0,
            mean_repulsion: 0.0,
            moved_frac: 0.0,
            accepted_worse_frac: 0.0,
            attempted_update_frac: 0.0,
            moved_given_attempt_frac: 0.0,
            mean_abs_delta_semitones: 0.0,
            mean_abs_delta_semitones_moved: 0.0,
        };
    }
    let mean_c_score_current_loo = if c_score_current_count > 0 {
        c_score_current_sum / c_score_current_count as f32
    } else {
        0.0
    };
    let mean_c_score_chosen_loo = if c_score_chosen_count > 0 {
        c_score_chosen_sum / c_score_chosen_count as f32
    } else {
        0.0
    };
    let mean_score = if score_count > 0 {
        score_sum / score_count as f32
    } else {
        0.0
    };
    let mean_repulsion = if repulsion_count > 0 {
        repulsion_sum / repulsion_count as f32
    } else {
        0.0
    };
    let inv = 1.0 / count as f32;
    let mean_abs_delta_semitones = abs_delta_sum * inv;
    let mean_abs_delta_semitones_moved = if moved_count > 0 {
        abs_delta_moved_sum / moved_count as f32
    } else {
        0.0
    };
    UpdateStats {
        mean_c_score_current_loo,
        mean_c_score_chosen_loo,
        mean_score,
        mean_repulsion,
        moved_frac: moved_count as f32 * inv,
        accepted_worse_frac: accepted_worse_count as f32 * inv,
        attempted_update_frac: attempt_count as f32 * inv,
        moved_given_attempt_frac: if attempt_count > 0 {
            moved_count as f32 / attempt_count as f32
        } else {
            0.0
        },
        mean_abs_delta_semitones,
        mean_abs_delta_semitones_moved,
    }
}

fn score_stats_at_indices(
    indices: &[usize],
    c_score_scan: &[f32],
    log2_ratio_scan: &[f32],
    score_sign: f32,
    lambda: f32,
    sigma: f32,
) -> UpdateStats {
    if indices.is_empty() {
        return UpdateStats {
            mean_c_score_current_loo: 0.0,
            mean_c_score_chosen_loo: 0.0,
            mean_score: 0.0,
            mean_repulsion: 0.0,
            moved_frac: 0.0,
            accepted_worse_frac: 0.0,
            attempted_update_frac: 0.0,
            moved_given_attempt_frac: 0.0,
            mean_abs_delta_semitones: 0.0,
            mean_abs_delta_semitones_moved: 0.0,
        };
    }
    let sigma = sigma.max(1e-6);
    let log2_vals: Vec<f32> = indices.iter().map(|&idx| log2_ratio_scan[idx]).collect();
    let mut score_sum = 0.0f32;
    let mut score_count = 0u32;
    let mut repulsion_sum = 0.0f32;
    let mut repulsion_count = 0u32;
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
        let score = score_sign * c_score_scan[idx] - lambda * repulsion;
        if score.is_finite() {
            score_sum += score;
            score_count += 1;
        }
        if repulsion.is_finite() {
            repulsion_sum += repulsion;
            repulsion_count += 1;
        }
    }
    let mean_score = if score_count > 0 {
        score_sum / score_count as f32
    } else {
        0.0
    };
    let mean_repulsion = if repulsion_count > 0 {
        repulsion_sum / repulsion_count as f32
    } else {
        0.0
    };
    UpdateStats {
        mean_c_score_current_loo: f32::NAN,
        mean_c_score_chosen_loo: f32::NAN,
        mean_score,
        mean_repulsion,
        moved_frac: 0.0,
        accepted_worse_frac: 0.0,
        attempted_update_frac: 0.0,
        moved_given_attempt_frac: 0.0,
        mean_abs_delta_semitones: 0.0,
        mean_abs_delta_semitones_moved: 0.0,
    }
}

fn mean_at_indices(values: &[f32], indices: &[usize]) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }
    let sum: f32 = indices.iter().map(|&idx| values[idx]).sum();
    sum / indices.len() as f32
}

fn mean_c_score_loo_at_indices_with_prev(
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    env_total: &[f32],
    density_total: &[f32],
    du_scan: &[f32],
    prev_indices: &[usize],
    eval_indices: &[usize],
) -> f32 {
    if prev_indices.is_empty() || eval_indices.is_empty() {
        return 0.0;
    }
    debug_assert_eq!(
        prev_indices.len(),
        eval_indices.len(),
        "prev_indices/eval_indices length mismatch"
    );
    space.assert_scan_len_named(env_total, "env_total");
    space.assert_scan_len_named(density_total, "density_total");
    space.assert_scan_len_named(du_scan, "du_scan");

    let mut env_loo = vec![0.0f32; env_total.len()];
    let mut density_loo = vec![0.0f32; density_total.len()];
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for (&prev_idx, &eval_idx) in prev_indices.iter().zip(eval_indices.iter()) {
        env_loo.copy_from_slice(env_total);
        density_loo.copy_from_slice(density_total);
        env_loo[prev_idx] = (env_loo[prev_idx] - 1.0).max(0.0);
        let denom = du_scan[prev_idx].max(1e-12);
        density_loo[prev_idx] = (density_loo[prev_idx] - 1.0 / denom).max(0.0);
        let (c_score_scan, _, _, _) =
            compute_c_score_state_scans(space, workspace, &env_loo, &density_loo, du_scan);
        let value = c_score_scan[eval_idx];
        if value.is_finite() {
            sum += value;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn mean_c_score_loo_at_indices(
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    env_total: &[f32],
    density_total: &[f32],
    du_scan: &[f32],
    indices: &[usize],
    _log2_ratio_scan: &[f32],
) -> f32 {
    mean_c_score_loo_at_indices_with_prev(
        space,
        workspace,
        env_total,
        density_total,
        du_scan,
        indices,
        indices,
    )
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

fn e2_controls_csv_c_state(
    baseline: &E2SweepStats,
    nohill: &E2SweepStats,
    norep: &E2SweepStats,
) -> String {
    let mut out = String::from(
        "step,baseline_mean,baseline_std,nohill_mean,nohill_std,norep_mean,norep_std\n",
    );
    let len = baseline
        .mean_c_state
        .len()
        .min(nohill.mean_c_state.len())
        .min(norep.mean_c_state.len());
    for i in 0..len {
        out.push_str(&format!(
            "{i},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            baseline.mean_c_state[i],
            baseline.std_c_state[i],
            nohill.mean_c_state[i],
            nohill.std_c_state[i],
            norep.mean_c_state[i],
            norep.std_c_state[i]
        ));
    }
    out
}

fn e2_controls_csv_c(
    baseline: &E2SweepStats,
    nohill: &E2SweepStats,
    norep: &E2SweepStats,
) -> String {
    let mut out = String::from(
        "step,baseline_mean,baseline_std,nohill_mean,nohill_std,norep_mean,norep_std\n",
    );
    let len = baseline
        .mean_c
        .len()
        .min(nohill.mean_c.len())
        .min(norep.mean_c.len());
    for i in 0..len {
        out.push_str(&format!(
            "{i},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            baseline.mean_c[i],
            baseline.std_c[i],
            nohill.mean_c[i],
            nohill.std_c[i],
            norep.mean_c[i],
            norep.std_c[i]
        ));
    }
    out
}

fn trajectories_csv(run: &E2Run) -> String {
    let mut out = String::from("step,agent_id,semitones,c_state\n");
    for (agent_id, semis) in run.trajectory_semitones.iter().enumerate() {
        let c_state = &run.trajectory_c_state[agent_id];
        let len = semis.len().min(c_state.len());
        for step in 0..len {
            out.push_str(&format!(
                "{step},{agent_id},{:.6},{:.6}\n",
                semis[step], c_state[step]
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
    let mut out = String::from(
        "seed,init_mode,steps,burn_in,mean_c_step0,mean_c_step_end,delta_c,mean_c_state_step0,mean_c_state_step_end,delta_c_state,mean_c_score_loo_step0,mean_c_score_loo_step_end,delta_c_score_loo\n",
    );
    for run in runs {
        let start = run.mean_c_series.first().copied().unwrap_or(0.0);
        let end = run.mean_c_series.last().copied().unwrap_or(start);
        let delta = end - start;
        let start_state = run.mean_c_state_series.first().copied().unwrap_or(0.0);
        let end_state = run
            .mean_c_state_series
            .last()
            .copied()
            .unwrap_or(start_state);
        let delta_state = end_state - start_state;
        let start_loo = run.mean_c_score_loo_series.first().copied().unwrap_or(0.0);
        let end_loo = run
            .mean_c_score_loo_series
            .last()
            .copied()
            .unwrap_or(start_loo);
        let delta_loo = end_loo - start_loo;
        out.push_str(&format!(
            "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            run.seed,
            E2_INIT_MODE.label(),
            E2_STEPS,
            E2_BURN_IN,
            start,
            end,
            delta,
            start_state,
            end_state,
            delta_state,
            start_loo,
            end_loo,
            delta_loo
        ));
    }
    out
}

#[derive(Clone, Copy, Debug, Default)]
struct FlutterMetrics {
    pingpong_rate_moves: f32,
    reversal_rate_moves: f32,
    move_rate_stepwise: f32,
    mean_abs_delta_moved: f32,
    step_count: usize,
    moved_step_count: usize,
    move_count: usize,
    pingpong_count_moves: usize,
    reversal_count_moves: usize,
}

fn flutter_metrics_for_trajectories(
    trajectories: &[Vec<f32>],
    start_step: usize,
    end_step: usize,
) -> FlutterMetrics {
    if trajectories.is_empty() || start_step >= end_step {
        return FlutterMetrics::default();
    }
    let mut step_count = 0usize;
    let mut moved_step_count = 0usize;
    let mut move_count = 0usize;
    let mut pingpong_count_moves = 0usize;
    let mut reversal_count_moves = 0usize;
    let mut pingpong_den_moves = 0usize;
    let mut reversal_den_moves = 0usize;
    let mut abs_delta_sum = 0.0f32;

    for traj in trajectories {
        if traj.len() <= start_step + 1 {
            continue;
        }
        let end = end_step.min(traj.len().saturating_sub(1));
        for t in (start_step + 1)..=end {
            let delta = traj[t] - traj[t - 1];
            let moved = delta.abs() > E2_SEMITONE_EPS;
            step_count += 1;
            if moved {
                moved_step_count += 1;
            }
        }

        let mut compressed: Vec<f32> = Vec::new();
        for t in start_step..=end {
            let v = traj[t];
            if compressed
                .last()
                .is_some_and(|last| (v - last).abs() <= E2_SEMITONE_EPS)
            {
                continue;
            }
            compressed.push(v);
        }
        let comp_len = compressed.len();
        if comp_len >= 2 {
            move_count += comp_len - 1;
            for i in 1..comp_len {
                abs_delta_sum += (compressed[i] - compressed[i - 1]).abs();
            }
        }
        if comp_len >= 3 {
            pingpong_den_moves += comp_len - 2;
            reversal_den_moves += comp_len - 2;
            for i in 2..comp_len {
                if (compressed[i] - compressed[i - 2]).abs() <= E2_SEMITONE_EPS {
                    pingpong_count_moves += 1;
                }
                let delta = compressed[i] - compressed[i - 1];
                let prev_delta = compressed[i - 1] - compressed[i - 2];
                if delta * prev_delta < 0.0 {
                    reversal_count_moves += 1;
                }
            }
        }
    }

    let move_rate_stepwise = if step_count > 0 {
        moved_step_count as f32 / step_count as f32
    } else {
        0.0
    };
    let pingpong_rate_moves = if pingpong_den_moves > 0 {
        pingpong_count_moves as f32 / pingpong_den_moves as f32
    } else {
        0.0
    };
    let reversal_rate_moves = if reversal_den_moves > 0 {
        reversal_count_moves as f32 / reversal_den_moves as f32
    } else {
        0.0
    };
    let mean_abs_delta_moved = if move_count > 0 {
        abs_delta_sum / move_count as f32
    } else {
        0.0
    };

    FlutterMetrics {
        pingpong_rate_moves,
        reversal_rate_moves,
        move_rate_stepwise,
        mean_abs_delta_moved,
        step_count,
        moved_step_count,
        move_count,
        pingpong_count_moves,
        reversal_count_moves,
    }
}

fn e2_flutter_segments(phase_mode: E2PhaseMode) -> Vec<(&'static str, usize, usize)> {
    if let Some(switch_step) = phase_mode.switch_step() {
        let pre_end = switch_step.saturating_sub(1);
        let post_start = switch_step;
        let mut segments = Vec::new();
        if pre_end >= E2_BURN_IN {
            segments.push(("pre", E2_BURN_IN, pre_end));
        }
        if post_start < E2_STEPS {
            segments.push(("post", post_start, E2_STEPS.saturating_sub(1)));
        }
        segments
    } else {
        vec![("all", E2_BURN_IN, E2_STEPS.saturating_sub(1))]
    }
}

#[derive(Clone)]
struct FlutterRow {
    condition: &'static str,
    seed: u64,
    segment: &'static str,
    metrics: FlutterMetrics,
}

fn flutter_by_seed_csv(rows: &[FlutterRow]) -> String {
    let mut out = String::from(
        "cond,seed,segment,pingpong_rate_moves,reversal_rate_moves,move_rate_stepwise,mean_abs_delta_moved,step_count,moved_step_count,move_count,pingpong_count_moves,reversal_count_moves\n",
    );
    for row in rows {
        let m = row.metrics;
        out.push_str(&format!(
            "{},{},{},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{}\n",
            row.condition,
            row.seed,
            row.segment,
            m.pingpong_rate_moves,
            m.reversal_rate_moves,
            m.move_rate_stepwise,
            m.mean_abs_delta_moved,
            m.step_count,
            m.moved_step_count,
            m.move_count,
            m.pingpong_count_moves,
            m.reversal_count_moves
        ));
    }
    out
}

fn flutter_summary_csv(rows: &[FlutterRow], segments: &[(&'static str, usize, usize)]) -> String {
    let mut out = String::from(
        "cond,segment,mean_pingpong_rate_moves,std_pingpong_rate_moves,mean_reversal_rate_moves,std_reversal_rate_moves,mean_move_rate_stepwise,std_move_rate_stepwise,mean_abs_delta_moved,std_abs_delta_moved,n\n",
    );
    for &cond in ["baseline", "nohill", "norep"].iter() {
        for (segment, _, _) in segments {
            let mut pingpong_vals = Vec::new();
            let mut reversal_vals = Vec::new();
            let mut move_vals = Vec::new();
            let mut abs_delta_vals = Vec::new();
            for row in rows
                .iter()
                .filter(|r| r.condition == cond && r.segment == *segment)
            {
                pingpong_vals.push(row.metrics.pingpong_rate_moves);
                reversal_vals.push(row.metrics.reversal_rate_moves);
                move_vals.push(row.metrics.move_rate_stepwise);
                abs_delta_vals.push(row.metrics.mean_abs_delta_moved);
            }
            let n = pingpong_vals.len();
            let (mean_ping, std_ping) = mean_std_scalar(&pingpong_vals);
            let (mean_rev, std_rev) = mean_std_scalar(&reversal_vals);
            let (mean_move, std_move) = mean_std_scalar(&move_vals);
            let (mean_abs, std_abs) = mean_std_scalar(&abs_delta_vals);
            out.push_str(&format!(
                "{cond},{segment},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
                mean_ping, std_ping, mean_rev, std_rev, mean_move, std_move, mean_abs, std_abs, n
            ));
        }
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

fn e2_meta_text(
    n_agents: usize,
    k_bins: i32,
    density_mass_mean: f32,
    density_mass_min: f32,
    density_mass_max: f32,
    r_ref_peak: f32,
    roughness_k: f32,
    roughness_ref_eps: f32,
    r_state01_min: f32,
    r_state01_mean: f32,
    r_state01_max: f32,
    phase_mode: E2PhaseMode,
) -> String {
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
    out.push_str(&format!("E2_ACCEPT_ENABLED={}\n", E2_ACCEPT_ENABLED));
    out.push_str(&format!("E2_ACCEPT_T0={:.3}\n", E2_ACCEPT_T0));
    out.push_str(&format!("E2_ACCEPT_TAU_STEPS={:.3}\n", E2_ACCEPT_TAU_STEPS));
    out.push_str(&format!(
        "E2_ACCEPT_RESET_ON_PHASE={}\n",
        E2_ACCEPT_RESET_ON_PHASE
    ));
    out.push_str(&format!(
        "E2_SCORE_IMPROVE_EPS={:.6}\n",
        E2_SCORE_IMPROVE_EPS
    ));
    let update_schedule = match E2_UPDATE_SCHEDULE {
        E2UpdateSchedule::Checkerboard => "checkerboard",
        E2UpdateSchedule::Lazy => "lazy",
    };
    out.push_str(&format!("E2_UPDATE_SCHEDULE={}\n", update_schedule));
    out.push_str(&format!("E2_LAZY_MOVE_PROB={:.3}\n", E2_LAZY_MOVE_PROB));
    out.push_str(&format!(
        "E2_ANTI_BACKTRACK_ENABLED={}\n",
        E2_ANTI_BACKTRACK_ENABLED
    ));
    out.push_str(&format!(
        "E2_ANTI_BACKTRACK_PRE_SWITCH_ONLY={}\n",
        E2_ANTI_BACKTRACK_PRE_SWITCH_ONLY
    ));
    out.push_str(&format!(
        "E2_BACKTRACK_ALLOW_EPS={:.6}\n",
        E2_BACKTRACK_ALLOW_EPS
    ));
    out.push_str(&format!("E2_SEMITONE_EPS={:.6}\n", E2_SEMITONE_EPS));
    out.push_str(&format!("C_STATE_BETA={:.3}\n", E2_C_STATE_BETA));
    out.push_str(&format!("C_STATE_THETA={:.3}\n", E2_C_STATE_THETA));
    out.push_str(&format!("ROUGHNESS_REF_EPS={:.3e}\n", roughness_ref_eps));
    out.push_str(&format!("ROUGHNESS_K={:.3}\n", roughness_k));
    out.push_str(&format!("R_REF_PEAK={:.6}\n", r_ref_peak));
    out.push_str(&format!("R_STATE01_MIN={:.6}\n", r_state01_min));
    out.push_str(&format!("R_STATE01_MEAN={:.6}\n", r_state01_mean));
    out.push_str(&format!("R_STATE01_MAX={:.6}\n", r_state01_max));
    out.push_str(&format!("E2_DENSITY_MASS_MEAN={:.6}\n", density_mass_mean));
    out.push_str(&format!("E2_DENSITY_MASS_MIN={:.6}\n", density_mass_min));
    out.push_str(&format!("E2_DENSITY_MASS_MAX={:.6}\n", density_mass_max));
    out.push_str(&format!("E2_INIT_MODE={}\n", E2_INIT_MODE.label()));
    out.push_str(&format!(
        "E2_INIT_CONSONANT_EXCLUSION_ST={:.3}\n",
        E2_INIT_CONSONANT_EXCLUSION_ST
    ));
    out.push_str(&format!("E2_INIT_MAX_TRIES={}\n", E2_INIT_MAX_TRIES));
    out.push_str(&format!("E2_ANCHOR_BIN_ST={:.3}\n", E2_ANCHOR_BIN_ST));
    out.push_str(&format!("E2_PAIRWISE_BIN_ST={:.3}\n", E2_PAIRWISE_BIN_ST));
    out.push_str(&format!("E2_DIVERSITY_BIN_ST={:.3}\n", E2_DIVERSITY_BIN_ST));
    out.push_str(&format!("E2_SEEDS={:?}\n", E2_SEEDS));
    out.push_str(&format!("E2_PHASE_MODE={}\n", phase_mode.label()));
    if let Some(step) = phase_mode.switch_step() {
        out.push_str(&format!("E2_PHASE_SWITCH_STEP={step}\n"));
    }
    out.push_str("E2_DISTRIBUTION_MODE=window_aggregated\n");
    out.push_str(&format!(
        "E2_POST_WINDOW_START_STEP={}\n",
        e2_post_window_start_step()
    ));
    out.push_str(&format!(
        "E2_POST_WINDOW_END_STEP={}\n",
        e2_post_window_end_step()
    ));
    let pairwise_n_pairs = if n_agents < 2 {
        0usize
    } else {
        n_agents * (n_agents - 1) / 2
    };
    out.push_str("E2_PAIRWISE_INTERVAL_SOURCE=final_snapshot\n");
    out.push_str(&format!(
        "E2_PAIRWISE_INTERVAL_N_PAIRS={}\n",
        pairwise_n_pairs
    ));
    out.push_str(
        "E2_MEAN_C_SCORE_CURRENT_LOO_DESC=mean C score at current positions using LOO env (env_total-1 at current bin, density_total-1/du)\n",
    );
    out.push_str(
        "E2_MEAN_C_SCORE_CHOSEN_LOO_DESC=mean C score at chosen positions using LOO env (removal at current bin)\n",
    );
    out.push_str("E2_MEAN_ABS_DELTA_SEMITONES_DESC=mean |Δ| over all agents\n");
    out.push_str("E2_MEAN_ABS_DELTA_SEMITONES_MOVED_DESC=mean |Δ| over moved agents\n");
    out.push_str("E2_ATTEMPTED_UPDATE_FRAC_DESC=attempted update / agents\n");
    out.push_str("E2_MOVED_GIVEN_ATTEMPT_FRAC_DESC=moved / attempted update\n");
    out.push_str(
        "E2_PINGPONG_RATE_DESC=move-compressed pingpong rate (count / max(0, compressed_len-2))\n",
    );
    out.push_str(
        "E2_REVERSAL_RATE_DESC=move-compressed reversal rate (count / max(0, move_count-1))\n",
    );
    out.push_str("E2_MOVE_RATE_STEPWISE_DESC=stepwise moved / steps\n");
    out
}

fn e2_marker_steps(phase_mode: E2PhaseMode) -> Vec<f32> {
    let mut steps = vec![E2_BURN_IN as f32];
    if e2_anchor_shift_enabled() {
        steps.push(E2_ANCHOR_SHIFT_STEP as f32);
    }
    if let Some(step) = phase_mode.switch_step() {
        steps.push(step as f32);
    }
    steps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    steps.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    steps
}

fn e3_metric_definition_text() -> String {
    let mut out = String::new();
    out.push_str("E3 metric definitions\n");
    out.push_str(
        "C is the 0-1 score passed into ArticulationCore::process as `consonance` (agent.last_consonance_state01()).\n",
    );
    out.push_str("C_firstK definition: mean over first K=20 ticks after birth (0..1).\n");
    out.push_str("Metabolism update (conceptual):\n");
    out.push_str("  E <- E - basal_cost_per_sec * dt\n");
    out.push_str("  Attack tick: E <- E - action_cost + recharge_per_attack * C\n");
    out.push_str(
        "C is clamped to [0,1] in the metabolism step (defensive), so recharge is continuous.\n",
    );
    out.push_str(
        "NoRecharge sets recharge_per_attack=0, so the C-dependent recharge term is removed.\n",
    );
    out.push_str(
        "Representative seed is chosen by the median Pearson r of baseline C_firstK vs lifetime; pooled plots concatenate all seeds.\n",
    );
    out.push_str(
        "c_state01_birth=first tick value; c_state01_firstk=mean of first K ticks; avg_c_state01_tick=mean over life; c_state01_std_over_life=std over life; avg_c_state01_attack=mean over attack ticks.\n",
    );
    out
}

fn pick_representative_run_index(runs: &[E2Run]) -> usize {
    if runs.is_empty() {
        return 0;
    }
    let mut scored: Vec<(usize, f32)> = runs
        .iter()
        .enumerate()
        .map(|(i, r)| (i, r.mean_c_state_series.last().copied().unwrap_or(0.0)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored[scored.len() / 2].0
}

fn representative_seed_text(runs: &[E2Run], rep_index: usize, phase_mode: E2PhaseMode) -> String {
    let mut scored: Vec<(usize, f32)> = runs
        .iter()
        .enumerate()
        .map(|(i, r)| (i, r.mean_c_state_series.last().copied().unwrap_or(0.0)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let metric_label = e2_post_label();
    let pre_label = e2_pre_label();
    let pre_step = e2_pre_step();
    let mut out = format!("metric={metric_label}_mean_c_state\n");
    out.push_str(&format!("phase_mode={}\n", phase_mode.label()));
    if let Some(step) = phase_mode.switch_step() {
        out.push_str(&format!("phase_switch_step={step}\n"));
    }
    out.push_str("rank,seed,metric\n");
    for (rank, (idx, metric)) in scored.iter().enumerate() {
        out.push_str(&format!("{rank},{},{}\n", runs[*idx].seed, metric));
    }
    let rep_metric = runs
        .get(rep_index)
        .and_then(|r| r.mean_c_state_series.last().copied())
        .unwrap_or(0.0);
    let rep_pre = runs
        .get(rep_index)
        .and_then(|r| r.mean_c_state_series.get(pre_step).copied())
        .unwrap_or(0.0);
    let rep_seed = runs.get(rep_index).map(|r| r.seed).unwrap_or(0);
    let rep_rank = scored
        .iter()
        .position(|(idx, _)| *idx == rep_index)
        .unwrap_or(0);
    out.push_str(&format!(
        "representative_seed={rep_seed}\nrepresentative_rank={rep_rank}\nrepresentative_metric={rep_metric}\nrepresentative_{pre_label}={rep_pre}\n"
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

fn histogram_probabilities_fixed(values: &[f32], min: f32, max: f32, bin_width: f32) -> Vec<f32> {
    let counts = histogram_counts_fixed(values, min, max, bin_width);
    let total: f32 = counts.iter().map(|(_, c)| *c).sum();
    if total <= 0.0 {
        return vec![0.0; counts.len()];
    }
    counts.into_iter().map(|(_, c)| c / total).collect()
}

fn hist_structure_metrics_from_probs(probs: &[f32]) -> HistStructureMetrics {
    if probs.is_empty() {
        return HistStructureMetrics {
            entropy: 0.0,
            gini: 0.0,
            peakiness: 0.0,
            kl_uniform: 0.0,
        };
    }
    let sum: f32 = probs.iter().copied().sum();
    if sum <= 0.0 {
        return HistStructureMetrics {
            entropy: 0.0,
            gini: 0.0,
            peakiness: 0.0,
            kl_uniform: 0.0,
        };
    }
    let inv_sum = 1.0 / sum;
    let mut norm: Vec<f32> = probs.iter().map(|p| (p * inv_sum).max(0.0)).collect();

    let mut entropy = 0.0f32;
    let mut kl_uniform = 0.0f32;
    let n = norm.len() as f32;
    let uniform = 1.0 / n;
    for p in &norm {
        if *p > 0.0 {
            entropy -= *p * p.ln();
            kl_uniform += *p * (p / uniform).ln();
        }
    }

    norm.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut weighted = 0.0f32;
    for (i, p) in norm.iter().enumerate() {
        weighted += (i as f32 + 1.0) * p;
    }
    let gini = ((2.0 * weighted) / (n * 1.0) - (n + 1.0) / n).clamp(0.0, 1.0);
    let peakiness = norm.iter().copied().fold(0.0f32, f32::max);

    HistStructureMetrics {
        entropy,
        gini,
        peakiness,
        kl_uniform,
    }
}

fn hist_structure_metrics_for_run(run: &E2Run) -> HistStructureMetrics {
    let samples = pairwise_interval_samples(&run.final_semitones);
    let probs = histogram_probabilities_fixed(&samples, 0.0, 12.0, E2_PAIRWISE_BIN_ST);
    hist_structure_metrics_from_probs(&probs)
}

fn hist_structure_rows(condition: &'static str, runs: &[E2Run]) -> Vec<HistStructureRow> {
    runs.iter()
        .map(|run| HistStructureRow {
            condition,
            seed: run.seed,
            metrics: hist_structure_metrics_for_run(run),
        })
        .collect()
}

fn hist_structure_by_seed_csv(rows: &[HistStructureRow]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# source=pairwise_intervals bin_width={:.3}\n",
        E2_PAIRWISE_BIN_ST
    ));
    out.push_str("seed,cond,entropy,gini,peakiness,kl_uniform\n");
    for row in rows {
        out.push_str(&format!(
            "{},{},{:.6},{:.6},{:.6},{:.6}\n",
            row.seed,
            row.condition,
            row.metrics.entropy,
            row.metrics.gini,
            row.metrics.peakiness,
            row.metrics.kl_uniform
        ));
    }
    out
}

fn hist_structure_summary_csv(rows: &[HistStructureRow]) -> String {
    let mut by_cond: std::collections::HashMap<&'static str, Vec<&HistStructureRow>> =
        std::collections::HashMap::new();
    for row in rows {
        by_cond.entry(row.condition).or_default().push(row);
    }
    let mut out = String::new();
    out.push_str(&format!(
        "# source=pairwise_intervals bin_width={:.3}\n",
        E2_PAIRWISE_BIN_ST
    ));
    out.push_str(
        "cond,mean_entropy,std_entropy,mean_gini,std_gini,mean_peakiness,std_peakiness,mean_kl_uniform,std_kl_uniform,n\n",
    );
    for cond in ["baseline", "nohill", "norep"] {
        let rows = by_cond.get(cond).cloned().unwrap_or_default();
        let entropy: Vec<f32> = rows.iter().map(|r| r.metrics.entropy).collect();
        let gini: Vec<f32> = rows.iter().map(|r| r.metrics.gini).collect();
        let peakiness: Vec<f32> = rows.iter().map(|r| r.metrics.peakiness).collect();
        let kl: Vec<f32> = rows.iter().map(|r| r.metrics.kl_uniform).collect();
        let (mean_entropy, std_entropy) = mean_std_scalar(&entropy);
        let (mean_gini, std_gini) = mean_std_scalar(&gini);
        let (mean_peak, std_peak) = mean_std_scalar(&peakiness);
        let (mean_kl, std_kl) = mean_std_scalar(&kl);
        out.push_str(&format!(
            "{cond},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
            mean_entropy,
            std_entropy,
            mean_gini,
            std_gini,
            mean_peak,
            std_peak,
            mean_kl,
            std_kl,
            rows.len()
        ));
    }
    out
}

fn render_hist_structure_summary_plot(
    out_path: &Path,
    rows: &[HistStructureRow],
) -> Result<(), Box<dyn Error>> {
    let mut by_cond: std::collections::HashMap<&'static str, Vec<&HistStructureRow>> =
        std::collections::HashMap::new();
    for row in rows {
        by_cond.entry(row.condition).or_default().push(row);
    }
    let conds = ["baseline", "nohill", "norep"];
    let colors = [&BLUE, &RED, &GREEN];

    let mut means_entropy = [0.0f32; 3];
    let mut means_gini = [0.0f32; 3];
    let mut means_peak = [0.0f32; 3];
    let mut means_kl = [0.0f32; 3];
    for (i, cond) in conds.iter().enumerate() {
        let rows = by_cond.get(cond).cloned().unwrap_or_default();
        let entropy: Vec<f32> = rows.iter().map(|r| r.metrics.entropy).collect();
        let gini: Vec<f32> = rows.iter().map(|r| r.metrics.gini).collect();
        let peakiness: Vec<f32> = rows.iter().map(|r| r.metrics.peakiness).collect();
        let kl: Vec<f32> = rows.iter().map(|r| r.metrics.kl_uniform).collect();
        means_entropy[i] = mean_std_scalar(&entropy).0;
        means_gini[i] = mean_std_scalar(&gini).0;
        means_peak[i] = mean_std_scalar(&peakiness).0;
        means_kl[i] = mean_std_scalar(&kl).0;
    }

    let root = BitMapBackend::new(out_path, (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((2, 2));

    draw_hist_structure_panel(
        &panels[0],
        "E2 Histogram Structure: Entropy",
        "entropy",
        &means_entropy,
        &conds,
        &colors,
    )?;
    draw_hist_structure_panel(
        &panels[1],
        "E2 Histogram Structure: Gini",
        "gini",
        &means_gini,
        &conds,
        &colors,
    )?;
    draw_hist_structure_panel(
        &panels[2],
        "E2 Histogram Structure: Peakiness",
        "peakiness",
        &means_peak,
        &conds,
        &colors,
    )?;
    draw_hist_structure_panel(
        &panels[3],
        "E2 Histogram Structure: KL vs Uniform",
        "KL",
        &means_kl,
        &conds,
        &colors,
    )?;

    root.present()?;
    Ok(())
}

fn draw_hist_structure_panel(
    area: &DrawingArea<BitMapBackend, Shift>,
    caption: &str,
    y_desc: &str,
    values: &[f32; 3],
    labels: &[&str; 3],
    colors: &[&RGBColor; 3],
) -> Result<(), Box<dyn Error>> {
    let mut y_max = values.iter().copied().fold(0.0f32, f32::max).max(1e-6);
    y_max *= 1.1;

    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 18))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(-0.5f32..2.5f32, 0f32..y_max)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("condition")
        .y_desc(y_desc)
        .x_labels(3)
        .x_label_formatter(&|x| {
            let idx = x.round() as isize;
            if (0..=2).contains(&idx) {
                labels[idx as usize].to_string()
            } else {
                String::new()
            }
        })
        .draw()?;

    for (i, value) in values.iter().enumerate() {
        let center = i as f32;
        let x0 = center - 0.3;
        let x1 = center + 0.3;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, *value)],
            colors[i].filled(),
        )))?;
    }
    Ok(())
}

fn diversity_metrics_for_run(run: &E2Run) -> DiversityMetrics {
    let mut values: Vec<f32> = run
        .final_semitones
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if values.is_empty() {
        return DiversityMetrics {
            unique_bins: 0,
            nn_mean: 0.0,
            nn_std: 0.0,
            semitone_var: 0.0,
            semitone_mad: 0.0,
        };
    }

    let mut unique_bins = std::collections::HashSet::new();
    for &v in &values {
        let bin = (v / E2_DIVERSITY_BIN_ST).round() as i32;
        unique_bins.insert(bin);
    }

    let mut nn_dists = Vec::with_capacity(values.len());
    for (i, &v) in values.iter().enumerate() {
        let mut best = f32::INFINITY;
        for (j, &u) in values.iter().enumerate() {
            if i == j {
                continue;
            }
            let dist = (v - u).abs();
            if dist < best {
                best = dist;
            }
        }
        if best.is_finite() {
            nn_dists.push(best);
        }
    }

    let (nn_mean, nn_std) = if nn_dists.is_empty() {
        (0.0, 0.0)
    } else {
        let (mean, std) = mean_std_scalar(&nn_dists);
        (mean, std)
    };

    let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
    let var = values
        .iter()
        .map(|v| (*v - mean) * (*v - mean))
        .sum::<f32>()
        / values.len() as f32;

    let median = median_of_values(&mut values);
    let mut abs_dev: Vec<f32> = values.iter().map(|v| (v - median).abs()).collect();
    let mad = median_of_values(&mut abs_dev);

    DiversityMetrics {
        unique_bins: unique_bins.len(),
        nn_mean,
        nn_std,
        semitone_var: var,
        semitone_mad: mad,
    }
}

fn diversity_rows(condition: &'static str, runs: &[E2Run]) -> Vec<DiversityRow> {
    runs.iter()
        .map(|run| DiversityRow {
            condition,
            seed: run.seed,
            metrics: diversity_metrics_for_run(run),
        })
        .collect()
}

fn diversity_by_seed_csv(rows: &[DiversityRow]) -> String {
    let mut out = String::new();
    out.push_str(&format!("# bin_width={:.3}\n", E2_DIVERSITY_BIN_ST));
    out.push_str("seed,cond,unique_bins,nn_mean,nn_std,semitone_var,semitone_mad\n");
    for row in rows {
        out.push_str(&format!(
            "{},{},{},{:.6},{:.6},{:.6},{:.6}\n",
            row.seed,
            row.condition,
            row.metrics.unique_bins,
            row.metrics.nn_mean,
            row.metrics.nn_std,
            row.metrics.semitone_var,
            row.metrics.semitone_mad
        ));
    }
    out
}

fn diversity_summary_csv(rows: &[DiversityRow]) -> String {
    let mut by_cond: std::collections::HashMap<&'static str, Vec<&DiversityRow>> =
        std::collections::HashMap::new();
    for row in rows {
        by_cond.entry(row.condition).or_default().push(row);
    }
    let mut out = String::new();
    out.push_str(&format!("# bin_width={:.3}\n", E2_DIVERSITY_BIN_ST));
    out.push_str(
        "cond,mean_unique_bins,std_unique_bins,mean_nn,stdev_nn,mean_semitone_var,std_semitone_var,mean_semitone_mad,std_semitone_mad,n\n",
    );
    for cond in ["baseline", "nohill", "norep"] {
        let rows = by_cond.get(cond).cloned().unwrap_or_default();
        let unique_bins: Vec<f32> = rows.iter().map(|r| r.metrics.unique_bins as f32).collect();
        let nn_mean: Vec<f32> = rows.iter().map(|r| r.metrics.nn_mean).collect();
        let semitone_var: Vec<f32> = rows.iter().map(|r| r.metrics.semitone_var).collect();
        let semitone_mad: Vec<f32> = rows.iter().map(|r| r.metrics.semitone_mad).collect();
        let (mean_bins, std_bins) = mean_std_scalar(&unique_bins);
        let (mean_nn, std_nn) = mean_std_scalar(&nn_mean);
        let (mean_var, std_var) = mean_std_scalar(&semitone_var);
        let (mean_mad, std_mad) = mean_std_scalar(&semitone_mad);
        out.push_str(&format!(
            "{cond},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
            mean_bins,
            std_bins,
            mean_nn,
            std_nn,
            mean_var,
            std_var,
            mean_mad,
            std_mad,
            rows.len()
        ));
    }
    out
}

fn render_diversity_summary_plot(
    out_path: &Path,
    rows: &[DiversityRow],
) -> Result<(), Box<dyn Error>> {
    let mut by_cond: std::collections::HashMap<&'static str, Vec<&DiversityRow>> =
        std::collections::HashMap::new();
    for row in rows {
        by_cond.entry(row.condition).or_default().push(row);
    }
    let conds = ["baseline", "nohill", "norep"];
    let colors = [&BLUE, &RED, &GREEN];

    let mut mean_bins = [0.0f32; 3];
    let mut mean_nn = [0.0f32; 3];
    let mut mean_var = [0.0f32; 3];
    let mut mean_mad = [0.0f32; 3];
    for (i, cond) in conds.iter().enumerate() {
        let rows = by_cond.get(cond).cloned().unwrap_or_default();
        let unique_bins: Vec<f32> = rows.iter().map(|r| r.metrics.unique_bins as f32).collect();
        let nn_mean: Vec<f32> = rows.iter().map(|r| r.metrics.nn_mean).collect();
        let semitone_var: Vec<f32> = rows.iter().map(|r| r.metrics.semitone_var).collect();
        let semitone_mad: Vec<f32> = rows.iter().map(|r| r.metrics.semitone_mad).collect();
        mean_bins[i] = mean_std_scalar(&unique_bins).0;
        mean_nn[i] = mean_std_scalar(&nn_mean).0;
        mean_var[i] = mean_std_scalar(&semitone_var).0;
        mean_mad[i] = mean_std_scalar(&semitone_mad).0;
    }

    let root = BitMapBackend::new(out_path, (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((2, 2));

    draw_diversity_panel(
        &panels[0],
        "E2 Diversity: Unique Bins",
        "unique bins",
        &mean_bins,
        &conds,
        &colors,
    )?;
    draw_diversity_panel(
        &panels[1],
        "E2 Diversity: NN Distance",
        "nn mean (st)",
        &mean_nn,
        &conds,
        &colors,
    )?;
    draw_diversity_panel(
        &panels[2],
        "E2 Diversity: Variance",
        "var (st^2)",
        &mean_var,
        &conds,
        &colors,
    )?;
    draw_diversity_panel(
        &panels[3],
        "E2 Diversity: MAD",
        "MAD (st)",
        &mean_mad,
        &conds,
        &colors,
    )?;

    root.present()?;
    Ok(())
}

fn draw_diversity_panel(
    area: &DrawingArea<BitMapBackend, Shift>,
    caption: &str,
    y_desc: &str,
    values: &[f32; 3],
    labels: &[&str; 3],
    colors: &[&RGBColor; 3],
) -> Result<(), Box<dyn Error>> {
    let mut y_max = values.iter().copied().fold(0.0f32, f32::max).max(1e-6);
    y_max *= 1.1;

    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 18))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(-0.5f32..2.5f32, 0f32..y_max)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("condition")
        .y_desc(y_desc)
        .x_labels(3)
        .x_label_formatter(&|x| {
            let idx = x.round() as isize;
            if (0..=2).contains(&idx) {
                labels[idx as usize].to_string()
            } else {
                String::new()
            }
        })
        .draw()?;

    for (i, value) in values.iter().enumerate() {
        let center = i as f32;
        let x0 = center - 0.3;
        let x1 = center + 0.3;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, *value)],
            colors[i].filled(),
        )))?;
    }
    Ok(())
}

fn median_of_values(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
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
    if !mean_frac.is_empty() {
        let sum: f32 = mean_frac.iter().copied().sum();
        debug_assert!((sum - 1.0).abs() < 1e-3, "mean_frac sum not ~1 (sum={sum})");
    }
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
    let mut out = String::from("bin_center,mean_frac,std_frac,n_seeds,mean_count,std_count\n");
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
            stats.mean_frac[i],
            stats.std_frac[i],
            stats.n,
            stats.mean_count[i],
            stats.std_count[i]
        ));
    }
    out
}

fn e2_pairwise_hist_seed_sweep(
    runs: &[E2Run],
    bin_width: f32,
    min: f32,
    max: f32,
) -> (HistSweepStats, usize) {
    let mut hists = Vec::with_capacity(runs.len());
    let mut n_pairs = 0usize;
    for run in runs {
        let samples = pairwise_interval_samples(&run.final_semitones);
        let expected = if run.final_semitones.len() < 2 {
            0usize
        } else {
            run.final_semitones.len() * (run.final_semitones.len() - 1) / 2
        };
        debug_assert_eq!(samples.len(), expected, "pairwise interval count mismatch");
        for &value in &samples {
            debug_assert!(
                value >= min - 1e-6 && value <= max + 1e-6,
                "pairwise interval out of range: {value}"
            );
        }
        n_pairs = expected;
        hists.push(histogram_counts_fixed(&samples, min, max, bin_width));
    }
    let (centers, mean_count, std_count) = mean_std_histograms(&hists);
    let (mean_frac, std_frac) = mean_std_histogram_fractions(&hists);
    if !mean_frac.is_empty() {
        let sum: f32 = mean_frac.iter().copied().sum();
        debug_assert!(
            (sum - 1.0).abs() < 1e-3,
            "pairwise mean_frac sum not ~1 (sum={sum})"
        );
    }
    (
        HistSweepStats {
            centers,
            mean_count,
            std_count,
            mean_frac,
            std_frac,
            n: hists.len(),
        },
        n_pairs,
    )
}

fn e2_pairwise_hist_seed_sweep_csv(stats: &HistSweepStats, n_pairs: usize) -> String {
    let mut out =
        String::from("bin_center,mean_frac,std_frac,n_seeds,n_pairs,mean_count,std_count\n");
    let len = stats
        .centers
        .len()
        .min(stats.mean_count.len())
        .min(stats.std_count.len())
        .min(stats.mean_frac.len())
        .min(stats.std_frac.len());
    for i in 0..len {
        out.push_str(&format!(
            "{:.4},{:.6},{:.6},{},{},{:.6},{:.6}\n",
            stats.centers[i],
            stats.mean_frac[i],
            stats.std_frac[i],
            stats.n,
            n_pairs,
            stats.mean_count[i],
            stats.std_count[i]
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
    y_desc: &str,
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
        .y_desc(y_desc)
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

fn e2_c_snapshot(run: &E2Run) -> (f32, f32, f32) {
    let pre_idx = e2_pre_step();
    let post_idx = e2_post_step_for(E2_STEPS);
    e2_c_snapshot_at(&run.mean_c_series, pre_idx, post_idx)
}

fn e2_c_snapshot_at(series: &[f32], pre_idx: usize, post_idx: usize) -> (f32, f32, f32) {
    let init = series.first().copied().unwrap_or(0.0);
    let pre = series.get(pre_idx).copied().unwrap_or(init);
    let post = series
        .get(post_idx.min(series.len().saturating_sub(1)))
        .copied()
        .unwrap_or(pre);
    (init, pre, post)
}

fn e2_c_snapshot_series(
    series: &[f32],
    anchor_shift_enabled: bool,
    anchor_shift_step: usize,
    burn_in: usize,
    steps: usize,
) -> (f32, f32, f32) {
    let pre_idx = e2_pre_step_for(anchor_shift_enabled, anchor_shift_step, burn_in);
    let post_idx = e2_post_step_for(steps);
    e2_c_snapshot_at(series, pre_idx, post_idx)
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

fn render_series_plot_multi_with_band(
    out_path: &Path,
    caption: &str,
    y_desc: &str,
    series_list: &[(&str, &[f32], &[f32], RGBColor)],
    markers: &[f32],
) -> Result<(), Box<dyn Error>> {
    if series_list.is_empty() {
        return Ok(());
    }
    let mut x_max = 1.0f32;
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for (_, mean, std, _) in series_list {
        let len = mean.len().min(std.len());
        if len == 0 {
            continue;
        }
        x_max = x_max.max(len.saturating_sub(1) as f32);
        for i in 0..len {
            let lo = mean[i] - std[i];
            let hi = mean[i] + std[i];
            if lo.is_finite() && hi.is_finite() {
                y_min = y_min.min(lo);
                y_max = y_max.max(hi);
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

    for &(label, mean, std, color) in series_list {
        let len = mean.len().min(std.len());
        if len == 0 {
            continue;
        }
        let mut band_points: Vec<(f32, f32)> = Vec::with_capacity(len * 2);
        for i in 0..len {
            band_points.push((i as f32, mean[i] + std[i]));
        }
        for i in (0..len).rev() {
            band_points.push((i as f32, mean[i] - std[i]));
        }
        chart.draw_series(std::iter::once(Polygon::new(
            band_points,
            color.mix(0.2).filled(),
        )))?;
        let line = mean.iter().enumerate().map(|(i, &y)| (i as f32, y));
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

fn format_p_value(p: f32) -> String {
    if !p.is_finite() {
        return "p=nan".to_string();
    }
    if p < 0.001 {
        "p<0.001".to_string()
    } else {
        format!("p={:.3}", p)
    }
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

fn force_scatter_x_range(data: &mut ScatterData, x_min: f32, x_max: f32) {
    data.x_min = x_min;
    data.x_max = x_max;
    if (data.x_max - data.x_min).abs() < 1e-6 {
        data.x_max = data.x_min + 1.0;
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
    if data.x_min <= 0.5 && data.x_max >= 0.5 {
        let y_top = data.y_max * 1.05;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(0.5, 0.0), (0.5, y_top)],
            BLACK.mix(0.3),
        )))?;
    }

    let pearson_p = format_p_value(data.stats.pearson_p);
    let spearman_p = format_p_value(data.stats.spearman_p);
    let lines = vec![
        format!("N={}", data.stats.n),
        format!("Pearson r={:.3} ({})", data.stats.pearson_r, pearson_p),
        format!("Spearman ρ={:.3} ({})", data.stats.spearman_rho, spearman_p),
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
    let mut data = build_scatter_data(x_values, lifetimes, seed);
    force_scatter_x_range(&mut data, 0.0, 1.0);
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

    let logrank_p = format_p_value(data.stats.logrank_p);
    let lines = vec![
        format!("n_high={}, n_low={}", data.n_high, data.n_low),
        format!(
            "median_high={:.1}, median_low={:.1}",
            data.stats.median_high, data.stats.median_low
        ),
        format!("logrank {}", logrank_p),
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
    let x_min = 0.0;
    let x_max = 1.0;
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
        .caption("E3 C_state01 vs Lifetime", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, 0.0f32..(y_max * 1.05))?;

    chart
        .configure_mesh()
        .x_desc("avg C_state01")
        .y_desc("lifetime (steps)")
        .draw()?;

    let points = deaths
        .iter()
        .map(|(_, lifetime, avg_c_state)| (*avg_c_state, *lifetime as f32));
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

fn render_survival_by_c_state(
    out_path: &Path,
    deaths: &[(usize, u32, f32)],
) -> Result<(), Box<dyn Error>> {
    if deaths.is_empty() {
        return Ok(());
    }
    let mut c_state_values: Vec<f32> = deaths.iter().map(|(_, _, c_state)| *c_state).collect();
    c_state_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if c_state_values.len().is_multiple_of(2) {
        let hi = c_state_values.len() / 2;
        let lo = hi.saturating_sub(1);
        0.5 * (c_state_values[lo] + c_state_values[hi])
    } else {
        c_state_values[c_state_values.len() / 2]
    };

    let mut high: Vec<u32> = Vec::new();
    let mut low: Vec<u32> = Vec::new();
    for (_, lifetime, c_state) in deaths {
        if *c_state >= median {
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
        .caption(
            "E3 Survival by C_state01 (Median Split)",
            ("sans-serif", 20),
        )
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
            .label("avg C_state01 >= median")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
    }
    if !low_series.is_empty() {
        chart
            .draw_series(LineSeries::new(low_series, &RED))?
            .label("avg C_state01 < median")
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
    let eval_start = e5_sample_start_step(main_series.len()) as f32 * E5_DT;
    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "E5 Order Parameter r(t) — pre-kick k_eff=0 so main/control overlap",
            ("sans-serif", 20),
        )
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

    chart.draw_series(std::iter::once(Rectangle::new(
        [(eval_start, 0.0f32), (x_max.max(1.0), 1.05f32)],
        RGBColor(160, 160, 160).mix(0.15).filled(),
    )))?;

    chart
        .draw_series(LineSeries::new(
            r_ctrl,
            ShapeStyle::from(&RED.mix(0.4)).stroke_width(2),
        ))?
        .label("control r(t)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.mix(0.4)));
    chart
        .draw_series(LineSeries::new(
            r_main,
            ShapeStyle::from(&BLUE).stroke_width(3),
        ))?
        .label("main r(t)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

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
    let eval_start = e5_sample_start_step(main_series.len()) as f32 * E5_DT;
    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "E5 Group Phase Offset Δφ(t) — pre-kick k_eff=0 so main/control overlap",
            ("sans-serif", 20),
        )
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

    chart.draw_series(std::iter::once(Rectangle::new(
        [(eval_start, -PI), (x_max.max(1.0), PI)],
        RGBColor(160, 160, 160).mix(0.15).filled(),
    )))?;

    chart
        .draw_series(LineSeries::new(
            ctrl_points,
            ShapeStyle::from(&RED.mix(0.4)).stroke_width(2),
        ))?
        .label("control Δφ(t)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.mix(0.4)));
    chart
        .draw_series(LineSeries::new(
            main_points,
            ShapeStyle::from(&BLUE).stroke_width(3),
        ))?
        .label("main Δφ(t)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

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
    let eval_start = e5_sample_start_step(main_series.len()) as f32 * E5_DT;
    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "E5 PLV_agent_kick — pre-kick k_eff=0 so main/control overlap",
            ("sans-serif", 20),
        )
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
    chart.draw_series(std::iter::once(Rectangle::new(
        [(eval_start, 0.0f32), (x_max.max(1.0), 1.05f32)],
        RGBColor(160, 160, 160).mix(0.15).filled(),
    )))?;

    chart
        .draw_series(LineSeries::new(
            plv_ctrl,
            ShapeStyle::from(&RED.mix(0.4)).stroke_width(2),
        ))?
        .label("control PLV_agent_kick")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.mix(0.4)));
    chart
        .draw_series(LineSeries::new(
            plv_main,
            ShapeStyle::from(&BLUE).stroke_width(3),
        ))?
        .label("main PLV_agent_kick")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

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

fn e5_plv_ranges(steps: usize) -> (f32, f32, f32, f32) {
    let burn_end_t = E5_BURN_IN_STEPS as f32 * E5_DT;
    let kick_on_t = E5_KICK_ON_STEP
        .map(|s| s as f32 * E5_DT)
        .unwrap_or(burn_end_t);
    let window_t = E5_TIME_PLV_WINDOW_STEPS as f32 * E5_DT;
    let sample_start_t = e5_sample_start_step(steps) as f32 * E5_DT;
    let pre_start = burn_end_t;
    let pre_end = kick_on_t;
    let post_start = (kick_on_t + window_t).max(burn_end_t);
    let post_end = sample_start_t;
    (pre_start, pre_end, post_start, post_end)
}

fn e5_kick_summary_csv(main: &E5KickSimResult, ctrl: &E5KickSimResult) -> String {
    let mut out = String::from(
        "condition,plv_pre_mean,plv_pre_std,plv_post_mean,plv_post_std,delta_phi_post_plv,plv_time\n",
    );
    let (pre_start, pre_end, post_start, post_end) = e5_plv_ranges(E5_STEPS);

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

struct E5SeedRow {
    condition: &'static str,
    seed: u64,
    plv_post_mean: f32,
}

fn e5_seed_sweep_rows(steps: usize) -> Vec<E5SeedRow> {
    let mut rows = Vec::new();
    let (_, _, post_start, post_end) = e5_plv_ranges(steps);
    for &seed in &E5_SEEDS {
        let sim_main = simulate_e5_kick(seed, steps, E5_K_KICK, E5_KICK_ON_STEP);
        let sim_ctrl = simulate_e5_kick(seed, steps, 0.0, E5_KICK_ON_STEP);
        let (post_main, _, _) = e5_mean_plv_range(&sim_main.series, post_start, post_end);
        let (post_ctrl, _, _) = e5_mean_plv_range(&sim_ctrl.series, post_start, post_end);
        rows.push(E5SeedRow {
            condition: "main",
            seed,
            plv_post_mean: post_main,
        });
        rows.push(E5SeedRow {
            condition: "control",
            seed,
            plv_post_mean: post_ctrl,
        });
    }
    rows
}

fn e5_seed_sweep_csv(rows: &[E5SeedRow]) -> String {
    let mut out = String::from("condition,seed,plv_post_mean\n");
    for row in rows {
        out.push_str(&format!(
            "{},{},{:.6}\n",
            row.condition, row.seed, row.plv_post_mean
        ));
    }
    out
}

fn render_e5_seed_sweep_plot(out_path: &Path, rows: &[E5SeedRow]) -> Result<(), Box<dyn Error>> {
    let mut main_vals = Vec::new();
    let mut ctrl_vals = Vec::new();
    for row in rows {
        if !row.plv_post_mean.is_finite() {
            continue;
        }
        match row.condition {
            "main" => main_vals.push(row.plv_post_mean),
            "control" => ctrl_vals.push(row.plv_post_mean),
            _ => {}
        }
    }
    let (main_mean, main_std) = mean_std_scalar(&main_vals);
    let (ctrl_mean, ctrl_std) = mean_std_scalar(&ctrl_vals);

    let mut y_max = (main_mean + main_std).max(ctrl_mean + ctrl_std);
    if !y_max.is_finite() || y_max <= 0.0 {
        y_max = 1.0;
    }
    let root = BitMapBackend::new(out_path, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "E5 Seed Sweep: PLV_post_mean (Agent-Kick)",
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(-0.5f32..1.5f32, 0.0f32..(y_max * 1.1))?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("condition")
        .y_desc("PLV_post_mean")
        .x_labels(2)
        .x_label_formatter(&|x| {
            let idx = x.round() as isize;
            match idx {
                0 => "main".to_string(),
                1 => "control".to_string(),
                _ => String::new(),
            }
        })
        .draw()?;

    let values = [(main_mean, main_std, BLUE), (ctrl_mean, ctrl_std, RED)];
    let cap = 0.05f32;
    for (i, (mean, std, color)) in values.iter().enumerate() {
        let center = i as f32;
        let x0 = center - 0.25;
        let x1 = center + 0.25;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, *mean)],
            color.filled(),
        )))?;
        let y0 = (mean - std).max(0.0);
        let y1 = mean + std;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(center, y0), (center, y1)],
            color.mix(0.8),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(center - cap, y0), (center + cap, y0)],
            color.mix(0.8),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(center - cap, y1), (center + cap, y1)],
            color.mix(0.8),
        )))?;
    }

    root.present()?;
    Ok(())
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
            mean_c_series: vec![metric],
            mean_c_state_series: vec![metric],
            mean_c_score_loo_series: vec![metric],
            mean_c_score_chosen_loo_series: vec![metric],
            mean_score_series: vec![0.0],
            mean_repulsion_series: vec![0.0],
            moved_frac_series: vec![0.0],
            accepted_worse_frac_series: vec![0.0],
            attempted_update_frac_series: vec![0.0],
            moved_given_attempt_frac_series: vec![0.0],
            mean_abs_delta_semitones_series: vec![0.0],
            mean_abs_delta_semitones_moved_series: vec![0.0],
            semitone_samples_pre: Vec::new(),
            semitone_samples_post: Vec::new(),
            final_semitones: Vec::new(),
            final_freqs_hz: Vec::new(),
            final_log2_ratios: Vec::new(),
            trajectory_semitones: Vec::new(),
            trajectory_c_state: Vec::new(),
            anchor_shift: E2AnchorShiftStats {
                step: 0,
                anchor_hz_before: 0.0,
                anchor_hz_after: 0.0,
                count_min: 0,
                count_max: 0,
                respawned: 0,
            },
            density_mass_mean: 0.0,
            density_mass_min: 0.0,
            density_mass_max: 0.0,
            r_state01_min: 0.0,
            r_state01_mean: 0.0,
            r_state01_max: 0.0,
            r_ref_peak: 0.0,
            roughness_k: 0.0,
            roughness_ref_eps: 0.0,
            n_agents: 0,
            k_bins: 0,
        }
    }

    fn assert_mean_c_score_loo_series_finite(
        condition: E2Condition,
        phase_mode: E2PhaseMode,
        step_semitones: f32,
        seed: u64,
    ) {
        let space = Log2Space::new(200.0, 400.0, 12);
        let anchor_idx = space.n_bins() / 2;
        let anchor_hz = space.centers_hz[anchor_idx];
        let run = run_e2_once(
            &space,
            anchor_hz,
            seed,
            condition,
            step_semitones,
            phase_mode,
        );
        assert_eq!(run.mean_c_score_loo_series.len(), E2_STEPS);
        let label = match condition {
            E2Condition::Baseline => "baseline",
            E2Condition::NoHillClimb => "nohill",
            E2Condition::NoRepulsion => "norep",
        };
        assert!(
            run.mean_c_score_loo_series.iter().all(|v| v.is_finite()),
            "mean_c_score_loo_series contains non-finite values (cond={label}, phase={phase_mode:?}, step={step_semitones})"
        );
    }

    #[test]
    fn e2_c_snapshot_uses_anchor_shift_pre_when_enabled() {
        let series: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let (init, pre, post) = e2_c_snapshot_series(&series, true, 6, 3, series.len());
        assert!((init - 0.0).abs() < 1e-6);
        assert!((pre - 5.0).abs() < 1e-6);
        assert!((post - 9.0).abs() < 1e-6);
    }

    #[test]
    fn e2_c_snapshot_uses_burnin_end_when_shift_disabled() {
        let series: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let (init, pre, post) = e2_c_snapshot_series(&series, false, 6, 3, series.len());
        assert!((init - 0.0).abs() < 1e-6);
        assert!((pre - 2.0).abs() < 1e-6);
        assert!((post - 9.0).abs() < 1e-6);
    }

    #[test]
    fn e2_density_normalization_invariant_to_scale() {
        let space = Log2Space::new(200.0, 400.0, 12);
        let workspace = build_consonance_workspace(&space);
        let (_erb_scan, du_scan) = erb_grid_for_space(&space);
        let anchor_idx = space.n_bins() / 2;
        let (env_scan, mut density_scan) = build_env_scans(&space, anchor_idx, &[], &du_scan);

        let (_, c_state_a, _, _) =
            compute_c_score_state_scans(&space, &workspace, &env_scan, &density_scan, &du_scan);

        for v in density_scan.iter_mut() {
            *v *= 4.0;
        }
        let (_, c_state_b, _, _) =
            compute_c_score_state_scans(&space, &workspace, &env_scan, &density_scan, &du_scan);

        assert_eq!(c_state_a.len(), c_state_b.len());
        for i in 0..c_state_a.len() {
            assert!(
                (c_state_a[i] - c_state_b[i]).abs() < 1e-5,
                "i={i} a={} b={}",
                c_state_a[i],
                c_state_b[i]
            );
        }
    }

    #[test]
    fn r_state01_stats_clamp_range() {
        let scan = [-0.5f32, 0.2, 1.2];
        let stats = r_state01_stats(&scan);
        assert!(stats.min >= 0.0 && stats.min <= 1.0);
        assert!(stats.mean >= 0.0 && stats.mean <= 1.0);
        assert!(stats.max >= 0.0 && stats.max <= 1.0);
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
        let c_score_scan = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let log2_ratio_scan = vec![0.0f32, 0.1, 0.2, 0.3, 0.4];
        update_agent_indices(
            &mut indices,
            &c_score_scan,
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
        let space = Log2Space::new(200.0, 400.0, 12);
        let workspace = build_consonance_workspace(&space);
        let (_erb_scan, du_scan) = erb_grid_for_space(&space);
        let anchor_idx = space.n_bins() / 2;
        let anchor_hz = space.centers_hz[anchor_idx];
        let log2_ratio_scan = build_log2_ratio_scan(&space, anchor_hz);
        let mut indices_fwd = vec![1usize, 3, 4];
        let mut indices_rev = indices_fwd.clone();
        let (env_scan, density_scan) = build_env_scans(&space, anchor_idx, &indices_fwd, &du_scan);
        let order_fwd: Vec<usize> = (0..indices_fwd.len()).collect();
        let order_rev: Vec<usize> = (0..indices_fwd.len()).rev().collect();
        let mut rng_fwd = StdRng::seed_from_u64(0);
        let mut rng_rev = StdRng::seed_from_u64(0);
        let stats_fwd = update_agent_indices_scored_stats_with_order_loo(
            &mut indices_fwd,
            &space,
            &workspace,
            &env_scan,
            &density_scan,
            &du_scan,
            &log2_ratio_scan,
            0,
            space.n_bins() - 1,
            1,
            1.0,
            0.2,
            0.1,
            0.05,
            3,
            false,
            None,
            &mut rng_fwd,
            &order_fwd,
        );
        let stats_rev = update_agent_indices_scored_stats_with_order_loo(
            &mut indices_rev,
            &space,
            &workspace,
            &env_scan,
            &density_scan,
            &du_scan,
            &log2_ratio_scan,
            0,
            space.n_bins() - 1,
            1,
            1.0,
            0.2,
            0.1,
            0.05,
            3,
            false,
            None,
            &mut rng_rev,
            &order_rev,
        );
        assert_eq!(indices_fwd, indices_rev);
        assert!((stats_fwd.mean_score - stats_rev.mean_score).abs() < 1e-6);
    }

    #[test]
    fn e2_update_schedule_checkerboard_updates_half_each_step() {
        if !matches!(E2_UPDATE_SCHEDULE, E2UpdateSchedule::Checkerboard) {
            return;
        }
        let n = 7usize;
        let step0 = (0..n)
            .filter(|&i| e2_should_attempt_update(i, 0, 0.0))
            .count();
        let step1 = (0..n)
            .filter(|&i| e2_should_attempt_update(i, 1, 0.0))
            .count();
        let diff = (step0 as isize - step1 as isize).abs();
        assert_eq!(step0 + step1, n);
        assert!(
            diff <= 1,
            "expected near-half updates, step0={step0}, step1={step1}"
        );
    }

    #[test]
    fn metropolis_accept_behaves_as_expected() {
        let (accept, worse) = metropolis_accept(0.1, 0.0, 0.9);
        assert!(accept);
        assert!(!worse);

        let (accept, worse) = metropolis_accept(-0.1, 0.0, 0.1);
        assert!(!accept);
        assert!(!worse);

        let (accept, worse) = metropolis_accept(-0.1, 0.1, 0.1);
        assert!(accept);
        assert!(worse);

        let (accept, worse) = metropolis_accept(-0.1, 0.1, 0.99);
        assert!(!accept);
        assert!(!worse);
    }

    #[test]
    fn histogram_probabilities_sum_to_one() {
        let values = [0.0f32, 0.25, 0.5, 0.75, 1.0];
        let probs = histogram_probabilities_fixed(&values, 0.0, 1.0, 0.5);
        let sum: f32 = probs.iter().copied().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
    }

    #[test]
    fn mean_std_histogram_fractions_sum_to_one() {
        let h1 = vec![(0.25, 1.0f32), (0.75, 1.0)];
        let h2 = vec![(0.25, 2.0f32), (0.75, 0.0)];
        let (mean, _std) = mean_std_histogram_fractions(&[h1, h2]);
        let sum: f32 = mean.iter().copied().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
    }

    #[test]
    fn hist_structure_metrics_uniform_and_point_mass() {
        let n = 8usize;
        let uniform = vec![1.0f32; n];
        let metrics = hist_structure_metrics_from_probs(&uniform);
        let expected_entropy = (n as f32).ln();
        assert!((metrics.entropy - expected_entropy).abs() < 1e-4);
        assert!(metrics.gini.abs() < 1e-6);
        assert!(metrics.kl_uniform.abs() < 1e-6);
        assert!((metrics.peakiness - 1.0 / n as f32).abs() < 1e-6);

        let mut point = vec![0.0f32; n];
        point[0] = 1.0;
        let metrics = hist_structure_metrics_from_probs(&point);
        assert!(metrics.peakiness > 0.9);
        assert!(metrics.gini > 1.0 - 2.0 / n as f32);
        assert!(metrics.kl_uniform > 0.1);
    }

    #[test]
    fn e2_accept_temperature_monotone() {
        let phase = E2PhaseMode::DissonanceThenConsonance;
        let t0 = e2_accept_temperature(0, phase);
        let t1 = e2_accept_temperature(1, phase);
        let t5 = e2_accept_temperature(5, phase);
        assert!((t0 - E2_ACCEPT_T0).abs() < 1e-6);
        assert!(t1 <= t0 + 1e-6);
        assert!(t5 <= t1 + 1e-6);
        if E2_ACCEPT_RESET_ON_PHASE {
            if let Some(switch_step) = phase.switch_step() {
                let t_switch = e2_accept_temperature(switch_step, phase);
                let t_after = e2_accept_temperature(switch_step + 1, phase);
                assert!((t_switch - E2_ACCEPT_T0).abs() < 1e-6);
                assert!(t_after <= t_switch + 1e-6);
            }
        }
    }

    #[test]
    fn update_agent_indices_accepts_worse_when_excluding_current() {
        let space = Log2Space::new(200.0, 400.0, 12);
        assert!(space.n_bins() >= 3);
        let mut workspace = build_consonance_workspace(&space);
        workspace.params.consonance_roughness_weight_floor = 0.0;
        workspace.params.consonance_roughness_weight = 0.0;

        let (_erb_scan, du_scan) = erb_grid_for_space(&space);
        let anchor_idx = space.n_bins() / 2;
        let anchor_hz = space.centers_hz[anchor_idx];
        let log2_ratio_scan = build_log2_ratio_scan(&space, anchor_hz);
        let mut indices = vec![anchor_idx];
        let (env_scan, density_scan) = build_env_scans(&space, anchor_idx, &indices, &du_scan);
        let order = vec![0usize];
        let mut rng = StdRng::seed_from_u64(0);
        let stats = update_agent_indices_scored_stats_with_order_loo(
            &mut indices,
            &space,
            &workspace,
            &env_scan,
            &density_scan,
            &du_scan,
            &log2_ratio_scan,
            0,
            space.n_bins() - 1,
            1,
            1.0,
            0.0,
            0.1,
            f32::INFINITY,
            0,
            false,
            None,
            &mut rng,
            &order,
        );
        assert!(
            stats.accepted_worse_frac > 0.0,
            "expected accepted_worse_frac > 0, got {}",
            stats.accepted_worse_frac
        );
    }

    #[test]
    fn anti_backtrack_blocks_backtrack_candidate() {
        let space = Log2Space::new(200.0, 400.0, 12);
        let workspace = build_consonance_workspace(&space);
        let (_erb_scan, du_scan) = erb_grid_for_space(&space);
        let anchor_idx = space.n_bins() / 2;
        let anchor_hz = space.centers_hz[anchor_idx];
        let log2_ratio_scan = build_log2_ratio_scan(&space, anchor_hz);

        let (env_anchor, density_anchor) = build_env_scans(&space, anchor_idx, &[], &du_scan);
        let (c_score_scan, _, _, _) =
            compute_c_score_state_scans(&space, &workspace, &env_anchor, &density_anchor, &du_scan);

        let mut max_idx = 0usize;
        let mut max_val = f32::NEG_INFINITY;
        let mut min_idx = 0usize;
        let mut min_val = f32::INFINITY;
        for (i, &value) in c_score_scan.iter().enumerate() {
            if !value.is_finite() {
                continue;
            }
            if value > max_val {
                max_val = value;
                max_idx = i;
            }
            if value < min_val {
                min_val = value;
                min_idx = i;
            }
        }
        assert_ne!(max_idx, min_idx, "expected distinct max/min indices");
        assert!(
            max_val > min_val + 1e-6,
            "expected score spread (max={max_val}, min={min_val})"
        );

        let current_idx = max_idx;
        let backtrack_idx = min_idx;
        let k = (backtrack_idx as i32 - current_idx as i32).abs();
        let order = vec![0usize];
        let (env_total, density_total) =
            build_env_scans(&space, anchor_idx, &[current_idx], &du_scan);

        let mut indices_no_block = vec![current_idx];
        let mut rng_no_block = StdRng::seed_from_u64(0);
        let _stats_no_block = update_agent_indices_scored_stats_with_order_loo(
            &mut indices_no_block,
            &space,
            &workspace,
            &env_total,
            &density_total,
            &du_scan,
            &log2_ratio_scan,
            backtrack_idx,
            backtrack_idx,
            k,
            1.0,
            0.0,
            0.1,
            f32::INFINITY,
            0,
            false,
            None,
            &mut rng_no_block,
            &order,
        );
        assert_eq!(
            indices_no_block[0], backtrack_idx,
            "expected backtrack candidate to be chosen when unblocked"
        );

        let mut indices_block = vec![current_idx];
        let mut rng_block = StdRng::seed_from_u64(0);
        let prev_positions = vec![backtrack_idx];
        let _stats_block = update_agent_indices_scored_stats_with_order_loo(
            &mut indices_block,
            &space,
            &workspace,
            &env_total,
            &density_total,
            &du_scan,
            &log2_ratio_scan,
            backtrack_idx,
            backtrack_idx,
            k,
            1.0,
            0.0,
            0.1,
            f32::INFINITY,
            0,
            true,
            Some(&prev_positions),
            &mut rng_block,
            &order,
        );
        assert_eq!(
            indices_block[0], current_idx,
            "expected backtrack candidate to be blocked"
        );
    }

    #[test]
    fn e2_flutter_regression_moved_frac_not_always_one() {
        let space = Log2Space::new(200.0, 400.0, 12);
        let anchor_idx = space.n_bins() / 2;
        let anchor_hz = space.centers_hz[anchor_idx];
        let run = run_e2_once(
            &space,
            anchor_hz,
            0xC0FFEE_u64 + 11,
            E2Condition::Baseline,
            E2_STEP_SEMITONES,
            E2PhaseMode::DissonanceThenConsonance,
        );
        let after_burn: Vec<f32> = run
            .moved_frac_series
            .iter()
            .skip(E2_BURN_IN)
            .copied()
            .collect();
        let all_one = after_burn.iter().all(|v| (*v - 1.0).abs() < 1e-6);
        assert!(
            !all_one,
            "moved_frac stuck at 1.0 after burn-in (len={})",
            run.moved_frac_series.len()
        );
    }

    #[test]
    fn update_stats_identity_relations_hold() {
        let space = Log2Space::new(200.0, 400.0, 12);
        let workspace = build_consonance_workspace(&space);
        let (_erb_scan, du_scan) = erb_grid_for_space(&space);
        let anchor_idx = space.n_bins() / 2;
        let anchor_hz = space.centers_hz[anchor_idx];
        let log2_ratio_scan = build_log2_ratio_scan(&space, anchor_hz);
        let mut indices = vec![1usize, 3, 4];
        let (env_scan, density_scan) = build_env_scans(&space, anchor_idx, &indices, &du_scan);
        let order: Vec<usize> = (0..indices.len()).collect();
        let mut rng = StdRng::seed_from_u64(0);
        let stats = update_agent_indices_scored_stats_with_order_loo(
            &mut indices,
            &space,
            &workspace,
            &env_scan,
            &density_scan,
            &du_scan,
            &log2_ratio_scan,
            0,
            space.n_bins() - 1,
            1,
            1.0,
            0.2,
            0.1,
            0.05,
            0,
            false,
            None,
            &mut rng,
            &order,
        );
        let expected_moved = stats.attempted_update_frac * stats.moved_given_attempt_frac;
        assert!(
            (stats.moved_frac - expected_moved).abs() < 1e-6,
            "moved_frac identity mismatch (moved_frac={}, attempted*given={})",
            stats.moved_frac,
            expected_moved
        );
        let expected_abs = stats.moved_frac * stats.mean_abs_delta_semitones_moved;
        assert!(
            (stats.mean_abs_delta_semitones - expected_abs).abs() < 1e-6,
            "mean_abs_delta identity mismatch (mean_abs_delta={}, moved_frac*mean_abs_delta_moved={})",
            stats.mean_abs_delta_semitones,
            expected_abs
        );
    }

    #[test]
    fn nohill_mean_c_score_loo_series_is_finite() {
        assert_mean_c_score_loo_series_finite(
            E2Condition::NoHillClimb,
            E2PhaseMode::Normal,
            E2_STEP_SEMITONES,
            0xC0FFEE_u64,
        );
    }

    #[test]
    fn parse_e2_phase_defaults_to_dtc() {
        let args: Vec<String> = Vec::new();
        let phase = parse_e2_phase(&args).expect("parse_e2_phase failed");
        assert_eq!(phase, E2PhaseMode::DissonanceThenConsonance);
    }

    #[test]
    fn parse_e2_phase_accepts_expected_values() {
        let cases: &[(&[&str], E2PhaseMode)] = &[
            (&["--e2-phase", "normal"], E2PhaseMode::Normal),
            (&["--e2-phase=normal"], E2PhaseMode::Normal),
            (
                &["--e2-phase", "dtc"],
                E2PhaseMode::DissonanceThenConsonance,
            ),
            (
                &["--e2-phase=dissonance_then_consonance"],
                E2PhaseMode::DissonanceThenConsonance,
            ),
        ];
        for (args, expected) in cases {
            let args = args.iter().map(|s| s.to_string()).collect::<Vec<_>>();
            let phase = parse_e2_phase(&args).expect("parse_e2_phase failed");
            assert_eq!(phase, *expected);
        }
    }

    #[test]
    fn parse_e2_phase_rejects_invalid_values() {
        let args = vec!["--e2-phase".to_string(), "foo".to_string()];
        let err = parse_e2_phase(&args).expect_err("expected parse_e2_phase to fail");
        assert!(
            err.contains("Usage: paper"),
            "expected usage in error, got: {err}"
        );
    }

    #[test]
    fn parse_e4_hist_defaults_off() {
        let args: Vec<String> = Vec::new();
        let enabled = parse_e4_hist(&args).expect("parse_e4_hist failed");
        assert!(!enabled);
    }

    #[test]
    fn parse_e4_hist_accepts_expected_values() {
        let cases: &[(&[&str], bool)] = &[
            (&["--e4-hist", "on"], true),
            (&["--e4-hist=on"], true),
            (&["--e4-hist", "off"], false),
            (&["--e4-hist=off"], false),
            (&["--e4-hist", "true"], true),
            (&["--e4-hist=false"], false),
            (&["--e4-hist", "1"], true),
            (&["--e4-hist=0"], false),
        ];
        for (args, expected) in cases {
            let args = args.iter().map(|s| s.to_string()).collect::<Vec<_>>();
            let enabled = parse_e4_hist(&args).expect("parse_e4_hist failed");
            assert_eq!(enabled, *expected);
        }
    }

    #[test]
    fn parse_e4_hist_rejects_invalid_values() {
        let args = vec!["--e4-hist".to_string(), "maybe".to_string()];
        let err = parse_e4_hist(&args).expect_err("expected parse_e4_hist to fail");
        assert!(
            err.contains("Usage: paper"),
            "expected usage in error, got: {err}"
        );
    }

    #[test]
    fn parse_experiments_ignores_other_flags() {
        let args = vec![
            "--e4-hist".to_string(),
            "on".to_string(),
            "--e2-phase".to_string(),
            "normal".to_string(),
            "--exp".to_string(),
            "e2,e4".to_string(),
        ];
        let experiments = parse_experiments(&args).expect("parse_experiments failed");
        assert_eq!(experiments, vec![Experiment::E2, Experiment::E4]);
    }

    #[test]
    fn e2_marker_steps_includes_phase_switch_when_dtc() {
        let steps = e2_marker_steps(E2PhaseMode::DissonanceThenConsonance);
        assert!(
            steps
                .iter()
                .any(|&s| (s - E2_PHASE_SWITCH_STEP as f32).abs() < 1e-6),
            "phase switch step not found in marker steps"
        );
    }

    #[test]
    fn baseline_mean_c_score_loo_series_is_finite() {
        assert_mean_c_score_loo_series_finite(
            E2Condition::Baseline,
            E2PhaseMode::Normal,
            E2_STEP_SEMITONES,
            0xC0FFEE_u64 + 1,
        );
    }

    #[test]
    fn baseline_mean_c_score_chosen_loo_series_is_finite() {
        let space = Log2Space::new(200.0, 400.0, 12);
        let anchor_idx = space.n_bins() / 2;
        let anchor_hz = space.centers_hz[anchor_idx];
        let run = run_e2_once(
            &space,
            anchor_hz,
            0xC0FFEE_u64 + 12,
            E2Condition::Baseline,
            E2_STEP_SEMITONES,
            E2PhaseMode::Normal,
        );
        assert_eq!(run.mean_c_score_chosen_loo_series.len(), E2_STEPS);
        assert!(
            run.mean_c_score_chosen_loo_series
                .iter()
                .all(|v| v.is_finite()),
            "baseline mean_c_score_chosen_loo_series contains non-finite values"
        );
    }

    #[test]
    fn baseline_mean_abs_delta_series_is_finite_and_nonnegative() {
        let space = Log2Space::new(200.0, 400.0, 12);
        let anchor_idx = space.n_bins() / 2;
        let anchor_hz = space.centers_hz[anchor_idx];
        let run = run_e2_once(
            &space,
            anchor_hz,
            0xC0FFEE_u64 + 13,
            E2Condition::Baseline,
            E2_STEP_SEMITONES,
            E2PhaseMode::Normal,
        );
        assert_eq!(run.mean_abs_delta_semitones_series.len(), E2_STEPS);
        assert!(
            run.mean_abs_delta_semitones_series
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0),
            "mean_abs_delta_semitones_series has non-finite or negative values"
        );
        assert!(
            run.mean_abs_delta_semitones_moved_series
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0),
            "mean_abs_delta_semitones_moved_series has non-finite or negative values"
        );
    }

    #[test]
    fn norep_mean_c_score_loo_series_is_finite() {
        assert_mean_c_score_loo_series_finite(
            E2Condition::NoRepulsion,
            E2PhaseMode::Normal,
            E2_STEP_SEMITONES,
            0xC0FFEE_u64 + 2,
        );
    }

    #[test]
    fn dtc_mean_c_score_loo_series_is_finite_across_conditions() {
        for (idx, condition) in [
            E2Condition::Baseline,
            E2Condition::NoHillClimb,
            E2Condition::NoRepulsion,
        ]
        .iter()
        .enumerate()
        {
            assert_mean_c_score_loo_series_finite(
                *condition,
                E2PhaseMode::DissonanceThenConsonance,
                E2_STEP_SEMITONES,
                0xC0FFEE_u64 + 10 + idx as u64,
            );
        }
    }

    #[test]
    fn mean_c_score_loo_series_is_finite_for_kbins_sweep() {
        for (idx, step) in E2_STEP_SEMITONES_SWEEP.iter().enumerate() {
            if (*step - E2_STEP_SEMITONES).abs() < 1e-6 {
                continue;
            }
            assert_mean_c_score_loo_series_finite(
                E2Condition::Baseline,
                E2PhaseMode::DissonanceThenConsonance,
                *step,
                0xC0FFEE_u64 + 20 + idx as u64,
            );
        }
    }

    #[test]
    fn k_from_semitones_rounds_as_expected() {
        let k_half = k_from_semitones(0.5);
        let k_quarter = k_from_semitones(0.25);
        assert!((k_half - 17).abs() <= 1);
        assert!((k_quarter - 8).abs() <= 1);
    }

    #[test]
    fn flutter_metrics_move_compressed_detects_pingpong_and_reversal() {
        let traj = vec![0.0f32, 1.0, 1.0, 0.0];
        let metrics = flutter_metrics_for_trajectories(&[traj], 0, 3);
        assert!((metrics.pingpong_rate_moves - 1.0).abs() < 1e-6);
        assert!((metrics.reversal_rate_moves - 1.0).abs() < 1e-6);
        assert!((metrics.move_rate_stepwise - (2.0 / 3.0)).abs() < 1e-6);
        assert!((metrics.mean_abs_delta_moved - 1.0).abs() < 1e-6);
    }

    #[test]
    fn flutter_metrics_move_compressed_no_pingpong_no_reversal() {
        let traj = vec![0.0f32, 1.0, 2.0, 3.0];
        let metrics = flutter_metrics_for_trajectories(&[traj], 0, 3);
        assert!(metrics.pingpong_rate_moves.abs() < 1e-6);
        assert!(metrics.reversal_rate_moves.abs() < 1e-6);
        assert!((metrics.move_rate_stepwise - 1.0).abs() < 1e-6);
        assert!((metrics.mean_abs_delta_moved - 1.0).abs() < 1e-6);
    }

    #[test]
    fn e2_backtrack_targets_update_retains_last_distinct() {
        let mut targets = vec![0usize];
        e2_update_backtrack_targets(&mut targets, &[0], &[1]);
        assert_eq!(targets[0], 0);
        e2_update_backtrack_targets(&mut targets, &[1], &[1]);
        assert_eq!(targets[0], 0);
        e2_update_backtrack_targets(&mut targets, &[1], &[2]);
        assert_eq!(targets[0], 1);
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
