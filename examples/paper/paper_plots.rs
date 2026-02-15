use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::f32::consts::PI;
use std::fs::{create_dir, create_dir_all, remove_dir, remove_dir_all};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use plotters::coord::types::RangedCoordf32;
use plotters::coord::{CoordTranslate, Shift};
use plotters::prelude::*;

use crate::sim::{
    E3Condition, E3DeathRecord, E3RunConfig, E4_ANCHOR_HZ, E4TailSamples, e3_policy_params,
    e4_paper_meta, run_e3_collect_deaths, run_e4_condition_tail_samples,
    run_e4_condition_tail_samples_with_wr, run_e4_mirror_schedule_samples,
};
use conchordal::core::erb::hz_to_erb;
use conchordal::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
use conchordal::core::landscape::{LandscapeParams, RoughnessScalarMode};
use conchordal::core::log2space::{Log2Space, sample_scan_linear_log2};
use conchordal::core::psycho_state;
use conchordal::core::roughness_kernel::{KernelParams, RoughnessKernel};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng, rngs::StdRng};

const SPACE_BINS_PER_OCT: u32 = 400;

const E2_SWEEPS: usize = 100;
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
const E2_PHASE_SWITCH_STEP: usize = E2_SWEEPS / 2;
const E2_DIVERSITY_BIN_ST: f32 = 0.25;
const E2_STEP_SEMITONES_SWEEP: [f32; 4] = [0.125, 0.25, 0.5, 1.0];
const E2_LAZY_MOVE_PROB: f32 = 0.65;
const E2_SEMITONE_EPS: f32 = 1e-6;
const E2_SEEDS: [u64; 20] = [
    0xC0FFEE_u64,
    0xA5A5A5A5_u64,
    0x1BADB002_u64,
    0xDEADBEEF_u64,
    0xFACEFEED_u64,
    0x1234ABCD_u64,
    0x31415926_u64,
    0x27182818_u64,
    0xCAFEBABE_u64,
    0x9E3779B9_u64,
    0x0F0F0F0F_u64,
    0x55AA55AA_u64,
    0x87654321_u64,
    0xABCDEF01_u64,
    0x0C0FFEE0_u64,
    0x13579BDF_u64,
    0x2468ACE0_u64,
    0xBADC0FFE_u64,
    0x10203040_u64,
    0x55667788_u64,
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
const E2_CONSONANT_TARGETS_CORE: [f32; 3] = [3.0, 4.0, 7.0];
const E2_CONSONANT_TARGETS_EXTENDED: [f32; 7] = [0.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0];
const E2_CONSONANT_WINDOW_ST: f32 = 0.25;
const E2_PERM_MAX_EXACT_COMBOS: u64 = 500_000;
const E2_PERM_MC_ITERS: usize = 50_000;
const E2_PERM_MC_SEED: u64 = 0xC0FFEE_u64 + 0xE2;

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
enum E2UpdateSchedule {
    Checkerboard,
    Lazy,
    RandomSingle,
}

const E2_UPDATE_SCHEDULE: E2UpdateSchedule = E2UpdateSchedule::Checkerboard;

const E4_TAIL_WINDOW_STEPS: u32 = 400;
const E4_DELTA_TAU: f32 = 0.05;
const E4_WEIGHT_COARSE_STEP: f32 = 0.1;
const E4_WEIGHT_FINE_STEP: f32 = 0.02;
const E4_BIN_WIDTHS: [f32; 2] = [0.25, 0.5];
const E4_EPS_CENTS: [f32; 3] = [25.0, 50.0, 12.5];
const E4_SOFT_SIGMA_SCALE: f32 = 0.5;
const E4_BOOTSTRAP_ITERS: usize = 4000;
const E4_BOOTSTRAP_SEED: u64 = 0xE4_600D_u64;
const E4_PAPER_HIST_BIN_CENTS: f32 = 25.0;
const E4_CENTS_MIN3: f32 = 316.0;
const E4_CENTS_MAJ3: f32 = 386.0;
const E4_CENTS_P4: f32 = 498.0;
const E4_CENTS_P5: f32 = 702.0;
const E4_CENTS_B2: f32 = 112.0;
const E4_BIND_SIGMA_CENTS: f32 = 15.0;
const E4_BIND_RHO: f32 = 0.4;
const E4_BIND_MAX_HARMONIC: u32 = 32;
const E4_BIND_TOP_CANDIDATES: usize = 512;
const E4_FINGERPRINT_TOL_CENTS: f32 = 25.0;
const E4_STEP_BURN_IN_STEPS: u32 = 600;
const E4_STEP_POST_STEPS: u32 = 600;
const E4_HYSTERESIS_SETTLE_STEPS: u32 = 180;
const E4_HYSTERESIS_EVAL_WINDOW: u32 = 80;
const E4_PROTOCOL_SEED: u64 = 0xC0FFEE_u64 + 10;
const FLOAT_KEY_SCALE: f32 = 1000.0;
const E4_SEEDS: [u64; 20] = [
    0xC0FFEE_u64 + 10,
    0xC0FFEE_u64 + 11,
    0xC0FFEE_u64 + 12,
    0xC0FFEE_u64 + 13,
    0xC0FFEE_u64 + 14,
    0xC0FFEE_u64 + 15,
    0xC0FFEE_u64 + 16,
    0xC0FFEE_u64 + 17,
    0xC0FFEE_u64 + 18,
    0xC0FFEE_u64 + 19,
    0xC0FFEE_u64 + 20,
    0xC0FFEE_u64 + 21,
    0xC0FFEE_u64 + 22,
    0xC0FFEE_u64 + 23,
    0xC0FFEE_u64 + 24,
    0xC0FFEE_u64 + 25,
    0xC0FFEE_u64 + 26,
    0xC0FFEE_u64 + 27,
    0xC0FFEE_u64 + 28,
    0xC0FFEE_u64 + 29,
];
const E4_REP_WEIGHTS: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
const E4_WR_GRID: [f32; 3] = [1.0, 0.5, 0.0];
const E4_WR_MIRROR_WEIGHTS: [f32; 3] = [0.0, 0.5, 1.0];
const E4_WR_FINGERPRINT_FOCUS: [f32; 3] = [1.0, 0.5, 0.0];
const E4_WR_REPS: usize = 20;
const E4_WR_BASE_SEED: u64 = 0xE4_7000_u64;
const E4_WR_PEAKLIST_TOP_N: usize = 64;
const E4_ORACLE_TOP_N: usize = 64;
const E4_ORACLE_SAMPLE_TRIALS: usize = 256;
const E4_DYNAMICS_PROBE_STEPS: u32 = 80;
const E4_DYNAMICS_BASE_LAMBDA: f32 = E2_LAMBDA;
const E4_DYNAMICS_REPULSION_SIGMA: f32 = E2_SIGMA;
const E4_DYNAMICS_STEP_SEMITONES: f32 = E2_STEP_SEMITONES;
const E4_DYNAMICS_PEAK_TOP_N: usize = 64;
const E4_ABCD_TRACE_STEPS: u32 = E4_DYNAMICS_PROBE_STEPS;
const E4_DIAG_PEAK_TOP_K: usize = 8;
const E4_EMIT_LEGACY_OUTPUTS: bool = false;

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
    0xC0FFEE_u64,
    0xC0FFEE_u64 + 1,
    0xC0FFEE_u64 + 2,
    0xC0FFEE_u64 + 3,
    0xC0FFEE_u64 + 4,
];
const PAPER_PLOTS_LOCK_DIR: &str = "examples/paper/.paper_plots.lock";
const PAPER_PLOTS_BASE_DIR: &str = "examples/paper/plots";

fn log_output_path(path: &Path) {
    println!("write {}", path.display());
}

fn write_with_log<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> io::Result<()> {
    let path = path.as_ref();
    log_output_path(path);
    std::fs::write(path, contents)
}

fn bitmap_root<'a>(out_path: &'a Path, size: (u32, u32)) -> SVGBackend<'a> {
    log_output_path(out_path);
    SVGBackend::new(out_path, size)
}

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

struct PaperRunLock {
    path: PathBuf,
}

impl PaperRunLock {
    fn acquire(path: &Path) -> io::Result<Self> {
        match create_dir(path) {
            Ok(()) => Ok(Self {
                path: path.to_path_buf(),
            }),
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {
                Err(io::Error::other(format!(
                    "paper plots already running (lock exists: {}). \
remove the lock directory if this is stale",
                    path.display()
                )))
            }
            Err(err) => Err(err),
        }
    }
}

impl Drop for PaperRunLock {
    fn drop(&mut self) {
        let _ = remove_dir(&self.path);
    }
}

fn usage() -> String {
    [
        "Usage: paper [--exp E1,E2,...] [--clean[=on|off]] [--e4-hist on|off] [--e4-kernel-gate on|off] [--e4-wr on|off] [--e2-phase mode]",
        "Examples:",
        "  paper --exp 2",
        "  paper --exp all",
        "  paper 1 3 5",
        "  paper --exp e2,e4",
        "  paper --clean --exp e4",
        "  paper --exp e4 --e4-hist on",
        "  paper --exp e4 --e4-kernel-gate on",
        "  paper --exp e4 --e4-wr on",
        "  paper --exp e2 --e2-phase dissonance_then_consonance",
        "If no experiment is specified, all (E1-E5) run.",
        "E4 histogram dumps default to off (use --e4-hist on to enable).",
        "E4 kernel gate plot default to off (use --e4-kernel-gate on to enable).",
        "E4 wr probe default to off (use --e4-wr on to enable).",
        "E2 phase modes: normal | dissonance_then_consonance (default)",
        "Outputs are written to examples/paper/plots/<exp>/ (e.g. examples/paper/plots/e2).",
        "By default only selected experiment dirs are overwritten.",
        "Use --clean to clear examples/paper/plots before running.",
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
        if arg == "--e4-kernel-gate" {
            if i + 1 >= args.len() {
                return Err(format!("Missing value after {arg}\n{}", usage()));
            }
            i += 2;
            continue;
        }
        if arg.starts_with("--e4-kernel-gate=") {
            i += 1;
            continue;
        }
        if arg == "--e4-wr" {
            if i + 1 >= args.len() {
                return Err(format!("Missing value after {arg}\n{}", usage()));
            }
            i += 2;
            continue;
        }
        if arg.starts_with("--e4-wr=") {
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
        if arg == "--clean" {
            if i + 1 < args.len() {
                let next = args[i + 1].as_str();
                if !next.starts_with('-') {
                    i += 2;
                    continue;
                }
            }
            i += 1;
            continue;
        }
        if arg.starts_with("--clean=") {
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

fn parse_e4_kernel_gate(args: &[String]) -> Result<bool, String> {
    let mut value: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--e4-kernel-gate" {
            if i + 1 >= args.len() {
                return Err(format!("Missing value after {arg}\n{}", usage()));
            }
            value = Some(args[i + 1].clone());
            i += 2;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--e4-kernel-gate=") {
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
            "Invalid --e4-kernel-gate value '{value}'. Use on/off.\n{}",
            usage()
        )),
    }
}

fn parse_e4_wr(args: &[String]) -> Result<bool, String> {
    let mut value: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--e4-wr" {
            if i + 1 >= args.len() {
                return Err(format!("Missing value after {arg}\n{}", usage()));
            }
            value = Some(args[i + 1].clone());
            i += 2;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--e4-wr=") {
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
            "Invalid --e4-wr value '{value}'. Use on/off.\n{}",
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

fn parse_clean(args: &[String]) -> Result<bool, String> {
    let mut clean_flag = false;
    let mut clean_value: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--clean" {
            clean_flag = true;
            if i + 1 < args.len() {
                let next = args[i + 1].as_str();
                if !next.starts_with('-') {
                    clean_value = Some(next.to_string());
                    i += 2;
                    continue;
                }
            }
            i += 1;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--clean=") {
            clean_value = Some(rest.to_string());
            i += 1;
            continue;
        }
        i += 1;
    }

    let Some(value) = clean_value else {
        return Ok(clean_flag);
    };
    let normalized = value.to_ascii_lowercase();
    match normalized.as_str() {
        "on" | "true" | "1" | "yes" => Ok(true),
        "off" | "false" | "0" | "no" => Ok(false),
        _ => Err(format!(
            "Invalid --clean value '{value}'. Use on/off.\n{}",
            usage()
        )),
    }
}

fn prepare_paper_output_dirs(
    base_dir: &Path,
    experiments: &[Experiment],
    clear_existing_selected: bool,
) -> io::Result<Vec<(Experiment, PathBuf)>> {
    if experiments.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(experiments.len());
    for &exp in experiments {
        let dir = base_dir.join(exp.dir_name());
        if clear_existing_selected && dir.exists() {
            remove_dir_all(&dir)?;
        }
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
    let e4_kernel_gate_enabled = parse_e4_kernel_gate(&args).map_err(io::Error::other)?;
    let e4_wr_enabled = parse_e4_wr(&args).map_err(io::Error::other)?;
    let e2_phase_mode = parse_e2_phase(&args).map_err(io::Error::other)?;
    let clean_all = parse_clean(&args).map_err(io::Error::other)?;
    let experiments = parse_experiments(&args).map_err(io::Error::other)?;
    let experiments = if experiments.is_empty() {
        Experiment::all()
    } else {
        experiments
    };

    let lock_dir = Path::new(PAPER_PLOTS_LOCK_DIR);
    if let Some(parent) = lock_dir.parent() {
        create_dir_all(parent)?;
    }
    let _run_lock = PaperRunLock::acquire(lock_dir)?;

    let base_dir = Path::new(PAPER_PLOTS_BASE_DIR);
    debug_assert!(
        base_dir.ends_with(Path::new("examples/paper/plots")),
        "refusing to clear unexpected path: {}",
        base_dir.display()
    );
    if clean_all && base_dir.exists() {
        remove_dir_all(base_dir)?;
    }
    create_dir_all(base_dir)?;
    let experiment_dirs = prepare_paper_output_dirs(base_dir, &experiments, !clean_all)?;

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
                        plot_e4_mirror_sweep(
                            out_dir,
                            anchor_hz,
                            e4_hist_enabled,
                            e4_kernel_gate_enabled,
                            e4_wr_enabled,
                        )
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
    let mut h_params_base = HarmonicityParams::default();
    h_params_base.rho_common_overtone = h_params_base.rho_common_root;
    h_params_base.gamma_root = 1.0;
    h_params_base.gamma_overtone = 1.0;
    let harmonicity_kernel = HarmonicityKernel::new(space, h_params_base);
    let mut h_params_m0 = h_params_base;
    h_params_m0.mirror_weight = 0.0;
    let mut h_params_m05 = h_params_base;
    h_params_m05.mirror_weight = 0.5;
    let mut h_params_m1 = h_params_base;
    h_params_m1.mirror_weight = 1.0;
    let harmonicity_kernel_m0 = HarmonicityKernel::new(space, h_params_m0);
    let harmonicity_kernel_m05 = HarmonicityKernel::new(space, h_params_m05);
    let harmonicity_kernel_m1 = HarmonicityKernel::new(space, h_params_m1);

    let (perc_h_pot_scan, _) =
        harmonicity_kernel.potential_h_from_log2_spectrum(&anchor_env_scan, space);
    let (perc_h_pot_scan_m0, _) =
        harmonicity_kernel_m0.potential_h_from_log2_spectrum(&anchor_env_scan, space);
    let (perc_h_pot_scan_m05, _) =
        harmonicity_kernel_m05.potential_h_from_log2_spectrum(&anchor_env_scan, space);
    let (perc_h_pot_scan_m1, _) =
        harmonicity_kernel_m1.potential_h_from_log2_spectrum(&anchor_env_scan, space);
    let (perc_r_pot_scan, _) =
        roughness_kernel.potential_r_from_log2_spectrum_density(&anchor_density_scan, space);

    space.assert_scan_len_named(&perc_h_pot_scan, "perc_h_pot_scan");
    space.assert_scan_len_named(&perc_h_pot_scan_m0, "perc_h_pot_scan_m0");
    space.assert_scan_len_named(&perc_h_pot_scan_m05, "perc_h_pot_scan_m05");
    space.assert_scan_len_named(&perc_h_pot_scan_m1, "perc_h_pot_scan_m1");
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

    let out_path = out_dir.join("paper_e1_landscape_scan_anchor220.svg");
    render_e1_plot(
        &out_path,
        anchor_hz,
        &log2_ratio_scan,
        &perc_h_pot_scan,
        &perc_r_state01_scan,
        &perc_c_score_scan,
    )?;
    let h_triplet_path = out_dir.join("paper_e1_h_mirror_m0_m05_m1.svg");
    render_e1_h_mirror_triplet_plot(
        &h_triplet_path,
        anchor_hz,
        &log2_ratio_scan,
        &perc_h_pot_scan_m0,
        &perc_h_pot_scan_m05,
        &perc_h_pot_scan_m1,
    )?;
    let h_diff_path = out_dir.join("paper_e1_h_mirror_m0_minus_m1.svg");
    render_e1_h_mirror_diff_plot(
        &h_diff_path,
        anchor_hz,
        &log2_ratio_scan,
        &perc_h_pot_scan_m0,
        &perc_h_pot_scan_m1,
    )?;

    Ok(())
}

fn plot_e2_emergent_harmony(
    out_dir: &Path,
    space: &Log2Space,
    anchor_hz: f32,
    phase_mode: E2PhaseMode,
) -> Result<(), Box<dyn Error>> {
    let (baseline_runs, baseline_stats, nohill_runs, nohill_stats, norep_runs, norep_stats) =
        std::thread::scope(|scope| {
            let baseline_handle = scope.spawn(|| {
                e2_seed_sweep(
                    space,
                    anchor_hz,
                    E2Condition::Baseline,
                    E2_STEP_SEMITONES,
                    phase_mode,
                )
            });
            let nohill_handle = scope.spawn(|| {
                e2_seed_sweep(
                    space,
                    anchor_hz,
                    E2Condition::NoHillClimb,
                    E2_STEP_SEMITONES,
                    phase_mode,
                )
            });
            let norep_handle = scope.spawn(|| {
                e2_seed_sweep(
                    space,
                    anchor_hz,
                    E2Condition::NoRepulsion,
                    E2_STEP_SEMITONES,
                    phase_mode,
                )
            });
            let (baseline_runs, baseline_stats) = baseline_handle
                .join()
                .expect("baseline seed sweep thread panicked");
            let (nohill_runs, nohill_stats) = nohill_handle
                .join()
                .expect("nohill seed sweep thread panicked");
            let (norep_runs, norep_stats) = norep_handle
                .join()
                .expect("norep seed sweep thread panicked");
            (
                baseline_runs,
                baseline_stats,
                nohill_runs,
                nohill_stats,
                norep_runs,
                norep_stats,
            )
        });
    let rep_index = pick_representative_run_index(&baseline_runs);
    let baseline_run = &baseline_runs[rep_index];
    let marker_steps = e2_marker_steps(phase_mode);
    let caption_suffix = e2_caption_suffix(phase_mode);
    let post_label = e2_post_label();
    let post_label_title = e2_post_label_title();
    let baseline_ci95_c_state = std_series_to_ci95(&baseline_stats.std_c_state, baseline_stats.n);
    let nohill_ci95_c_state = std_series_to_ci95(&nohill_stats.std_c_state, nohill_stats.n);
    let norep_ci95_c_state = std_series_to_ci95(&norep_stats.std_c_state, norep_stats.n);

    write_with_log(
        out_dir.join("paper_e2_representative_seed.txt"),
        representative_seed_text(&baseline_runs, rep_index, phase_mode),
    )?;

    write_with_log(
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
    write_with_log(out_dir.join("paper_e2_c_score_timeseries.csv"), c_score_csv)?;
    write_with_log(
        out_dir.join("paper_e2_c_state_timeseries.csv"),
        series_csv("step,mean_c_state", &baseline_run.mean_c_state_series),
    )?;
    write_with_log(
        out_dir.join("paper_e2_mean_c_score_loo_over_time.csv"),
        series_csv(
            "step,mean_c_score_loo",
            &baseline_run.mean_c_score_loo_series,
        ),
    )?;
    write_with_log(
        out_dir.join("paper_e2_mean_c_score_chosen_loo_over_time.csv"),
        series_csv(
            "step,mean_c_score_chosen_loo",
            &baseline_run.mean_c_score_chosen_loo_series,
        ),
    )?;
    write_with_log(
        out_dir.join("paper_e2_score_timeseries.csv"),
        series_csv("step,mean_score", &baseline_run.mean_score_series),
    )?;
    write_with_log(
        out_dir.join("paper_e2_repulsion_timeseries.csv"),
        series_csv("step,mean_repulsion", &baseline_run.mean_repulsion_series),
    )?;
    write_with_log(
        out_dir.join("paper_e2_moved_frac_timeseries.csv"),
        series_csv("step,moved_frac", &baseline_run.moved_frac_series),
    )?;
    write_with_log(
        out_dir.join("paper_e2_accepted_worse_frac_timeseries.csv"),
        series_csv(
            "step,accepted_worse_frac",
            &baseline_run.accepted_worse_frac_series,
        ),
    )?;
    write_with_log(
        out_dir.join("paper_e2_attempted_update_frac_timeseries.csv"),
        series_csv(
            "step,attempted_update_frac",
            &baseline_run.attempted_update_frac_series,
        ),
    )?;
    write_with_log(
        out_dir.join("paper_e2_moved_given_attempt_frac_timeseries.csv"),
        series_csv(
            "step,moved_given_attempt_frac",
            &baseline_run.moved_given_attempt_frac_series,
        ),
    )?;
    write_with_log(
        out_dir.join("paper_e2_mean_abs_delta_semitones_over_time.csv"),
        series_csv(
            "step,mean_abs_delta_semitones",
            &baseline_run.mean_abs_delta_semitones_series,
        ),
    )?;
    write_with_log(
        out_dir.join("paper_e2_mean_abs_delta_semitones_moved_over_time.csv"),
        series_csv(
            "step,mean_abs_delta_semitones_moved",
            &baseline_run.mean_abs_delta_semitones_moved_series,
        ),
    )?;

    write_with_log(
        out_dir.join("paper_e2_agent_trajectories.csv"),
        trajectories_csv(baseline_run),
    )?;
    if e2_anchor_shift_enabled() {
        write_with_log(
            out_dir.join("paper_e2_anchor_shift_stats.csv"),
            anchor_shift_csv(baseline_run),
        )?;
    }
    write_with_log(
        out_dir.join("paper_e2_final_agents.csv"),
        final_agents_csv(baseline_run),
    )?;

    let mean_plot_path = out_dir.join("paper_e2_mean_c_state_over_time.svg");
    render_series_plot_fixed_y(
        &mean_plot_path,
        &format!("E2 Mean C_state Over Time ({caption_suffix})"),
        "mean C_state",
        &series_pairs(&baseline_run.mean_c_state_series),
        &marker_steps,
        0.0,
        1.0,
    )?;

    let mean_c_score_path = out_dir.join("paper_e2_mean_c_score_over_time.svg");
    render_series_plot_with_markers(
        &mean_c_score_path,
        &format!("E2 Mean C Score Over Time ({caption_suffix})"),
        "mean C score",
        &series_pairs(&baseline_run.mean_c_series),
        &marker_steps,
    )?;

    let mean_c_score_loo_path = out_dir.join("paper_e2_mean_c_score_loo_over_time.svg");
    render_series_plot_with_markers(
        &mean_c_score_loo_path,
        &format!("E2 Mean C Score (LOO Current) Over Time ({caption_suffix})"),
        "mean C score (LOO current)",
        &series_pairs(&baseline_run.mean_c_score_loo_series),
        &marker_steps,
    )?;

    let mean_c_score_chosen_loo_path =
        out_dir.join("paper_e2_mean_c_score_chosen_loo_over_time.svg");
    render_series_plot_with_markers(
        &mean_c_score_chosen_loo_path,
        &format!("E2 Mean C Score (LOO Chosen) Over Time ({caption_suffix})"),
        "mean C score (LOO chosen)",
        &series_pairs(&baseline_run.mean_c_score_chosen_loo_series),
        &marker_steps,
    )?;

    let accept_worse_path = out_dir.join("paper_e2_accepted_worse_frac_over_time.svg");
    render_series_plot_fixed_y(
        &accept_worse_path,
        &format!("E2 Accepted Worse Fraction ({caption_suffix})"),
        "accepted worse frac",
        &series_pairs(&baseline_run.accepted_worse_frac_series),
        &marker_steps,
        0.0,
        1.0,
    )?;

    let mean_score_path = out_dir.join("paper_e2_mean_score_over_time.svg");
    render_series_plot_with_markers(
        &mean_score_path,
        &format!("E2 Mean Score Over Time ({caption_suffix})"),
        "mean score (C - λ·repulsion)",
        &series_pairs(&baseline_run.mean_score_series),
        &marker_steps,
    )?;

    let mean_repulsion_path = out_dir.join("paper_e2_mean_repulsion_over_time.svg");
    render_series_plot_with_markers(
        &mean_repulsion_path,
        &format!("E2 Mean Repulsion Over Time ({caption_suffix})"),
        "mean repulsion",
        &series_pairs(&baseline_run.mean_repulsion_series),
        &marker_steps,
    )?;

    let moved_frac_path = out_dir.join("paper_e2_moved_frac_over_time.svg");
    render_series_plot_with_markers(
        &moved_frac_path,
        &format!("E2 Moved Fraction Over Time ({caption_suffix})"),
        "moved fraction",
        &series_pairs(&baseline_run.moved_frac_series),
        &marker_steps,
    )?;

    let attempted_update_path = out_dir.join("paper_e2_attempted_update_frac_over_time.svg");
    render_series_plot_fixed_y(
        &attempted_update_path,
        &format!("E2 Attempted Update Fraction ({caption_suffix})"),
        "attempted update frac",
        &series_pairs(&baseline_run.attempted_update_frac_series),
        &marker_steps,
        0.0,
        1.0,
    )?;

    let moved_given_attempt_path = out_dir.join("paper_e2_moved_given_attempt_frac_over_time.svg");
    render_series_plot_fixed_y(
        &moved_given_attempt_path,
        &format!("E2 Moved Given Attempt ({caption_suffix})"),
        "moved given attempt frac",
        &series_pairs(&baseline_run.moved_given_attempt_frac_series),
        &marker_steps,
        0.0,
        1.0,
    )?;

    let abs_delta_path = out_dir.join("paper_e2_mean_abs_delta_semitones_over_time.svg");
    render_series_plot_with_markers(
        &abs_delta_path,
        &format!("E2 Mean |Δ| Semitones Over Time ({caption_suffix})"),
        "mean |Δ| semitones",
        &series_pairs(&baseline_run.mean_abs_delta_semitones_series),
        &marker_steps,
    )?;

    let abs_delta_moved_path =
        out_dir.join("paper_e2_mean_abs_delta_semitones_moved_over_time.svg");
    render_series_plot_with_markers(
        &abs_delta_moved_path,
        &format!("E2 Mean |Δ| Semitones (Moved) Over Time ({caption_suffix})"),
        "mean |Δ| semitones (moved only)",
        &series_pairs(&baseline_run.mean_abs_delta_semitones_moved_series),
        &marker_steps,
    )?;

    let trajectory_path = out_dir.join("paper_e2_agent_trajectories.svg");
    render_agent_trajectories_plot(&trajectory_path, &baseline_run.trajectory_semitones)?;

    let pairwise_intervals = pairwise_interval_samples(&baseline_run.final_semitones);
    write_with_log(
        out_dir.join("paper_e2_pairwise_intervals.csv"),
        pairwise_intervals_csv(&pairwise_intervals),
    )?;
    emit_pairwise_interval_dumps_for_condition(out_dir, "baseline", &baseline_runs)?;
    emit_pairwise_interval_dumps_for_condition(out_dir, "nohill", &nohill_runs)?;
    emit_pairwise_interval_dumps_for_condition(out_dir, "norep", &norep_runs)?;
    let pairwise_hist_path = out_dir.join("paper_e2_pairwise_interval_histogram.svg");
    render_interval_histogram(
        &pairwise_hist_path,
        "E2 Pairwise Interval Histogram (Semitones, 12=octave)",
        &pairwise_intervals,
        0.0,
        12.0,
        E2_PAIRWISE_BIN_ST,
        "semitones",
    )?;

    let hist_path = out_dir.join("paper_e2_interval_histogram.svg");
    let hist_caption = format!("E2 Interval Histogram ({post_label}, bin=0.50st)");
    render_interval_histogram(
        &hist_path,
        &hist_caption,
        &baseline_run.semitone_samples_post,
        -12.0,
        12.0,
        E2_ANCHOR_BIN_ST,
        "semitones",
    )?;

    write_with_log(
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
    write_with_log(out_dir.join("paper_e2_flutter_metrics.csv"), flutter_csv)?;

    render_e2_histogram_sweep(out_dir, baseline_run)?;

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
    write_with_log(
        out_dir.join("paper_e2_flutter_by_seed.csv"),
        flutter_by_seed_csv(&flutter_rows),
    )?;
    write_with_log(
        out_dir.join("paper_e2_flutter_summary.csv"),
        flutter_summary_csv(&flutter_rows, &flutter_segments),
    )?;

    write_with_log(
        out_dir.join("paper_e2_seed_sweep_mean_c.csv"),
        sweep_csv(
            "step,mean,std,n",
            &baseline_stats.mean_c,
            &baseline_stats.std_c,
            baseline_stats.n,
        ),
    )?;
    write_with_log(
        out_dir.join("paper_e2_seed_sweep_mean_c_ci95.csv"),
        sweep_csv_with_ci95(
            "step,mean,std,ci95,n\n",
            &baseline_stats.mean_c,
            &baseline_stats.std_c,
            baseline_stats.n,
        ),
    )?;
    write_with_log(
        out_dir.join("paper_e2_seed_sweep_mean_c_state.csv"),
        sweep_csv(
            "step,mean,std,n",
            &baseline_stats.mean_c_state,
            &baseline_stats.std_c_state,
            baseline_stats.n,
        ),
    )?;
    write_with_log(
        out_dir.join("paper_e2_seed_sweep_mean_c_state_ci95.csv"),
        sweep_csv_with_ci95(
            "step,mean,std,ci95,n\n",
            &baseline_stats.mean_c_state,
            &baseline_stats.std_c_state,
            baseline_stats.n,
        ),
    )?;
    write_with_log(
        out_dir.join("paper_e2_seed_sweep_mean_score.csv"),
        sweep_csv(
            "step,mean,std,n",
            &baseline_stats.mean_score,
            &baseline_stats.std_score,
            baseline_stats.n,
        ),
    )?;
    write_with_log(
        out_dir.join("paper_e2_seed_sweep_mean_repulsion.csv"),
        sweep_csv(
            "step,mean,std,n",
            &baseline_stats.mean_repulsion,
            &baseline_stats.std_repulsion,
            baseline_stats.n,
        ),
    )?;
    write_with_log(
        out_dir.join("paper_e2_seed_sweep_mean_c_score_loo.csv"),
        sweep_csv(
            "step,mean,std,n",
            &baseline_stats.mean_c_score_loo,
            &baseline_stats.std_c_score_loo,
            baseline_stats.n,
        ),
    )?;
    write_with_log(
        out_dir.join("paper_e2_kbins_sweep_summary.csv"),
        e2_kbins_sweep_csv(space, anchor_hz, phase_mode),
    )?;

    let sweep_mean_path = out_dir.join("paper_e2_mean_c_state_over_time_seeds.svg");
    render_series_plot_with_band(
        &sweep_mean_path,
        "E2 Mean C_state (seed sweep)",
        "mean C_state",
        &baseline_stats.mean_c_state,
        &baseline_stats.std_c_state,
        &marker_steps,
    )?;
    let sweep_mean_ci_path = out_dir.join("paper_e2_mean_c_state_over_time_seeds_ci95.svg");
    render_series_plot_with_band(
        &sweep_mean_ci_path,
        "E2 Mean C_state (seed sweep, 95% CI)",
        "mean C_state",
        &baseline_stats.mean_c_state,
        &baseline_ci95_c_state,
        &marker_steps,
    )?;

    let sweep_score_path = out_dir.join("paper_e2_mean_score_over_time_seeds.svg");
    render_series_plot_with_band(
        &sweep_score_path,
        "E2 Mean Score (seed sweep)",
        "mean score",
        &baseline_stats.mean_score,
        &baseline_stats.std_score,
        &marker_steps,
    )?;

    let sweep_c_score_loo_path = out_dir.join("paper_e2_mean_c_score_loo_over_time_seeds.svg");
    render_series_plot_with_band(
        &sweep_c_score_loo_path,
        "E2 Mean C Score (LOO current, seed sweep)",
        "mean C score (LOO current)",
        &baseline_stats.mean_c_score_loo,
        &baseline_stats.std_c_score_loo,
        &marker_steps,
    )?;

    let sweep_rep_path = out_dir.join("paper_e2_mean_repulsion_over_time_seeds.svg");
    render_series_plot_with_band(
        &sweep_rep_path,
        "E2 Mean Repulsion (seed sweep)",
        "mean repulsion",
        &baseline_stats.mean_repulsion,
        &baseline_stats.std_repulsion,
        &marker_steps,
    )?;

    write_with_log(
        out_dir.join("paper_e2_control_mean_c.csv"),
        e2_controls_csv_c(&baseline_stats, &nohill_stats, &norep_stats),
    )?;
    write_with_log(
        out_dir.join("paper_e2_control_mean_c_state.csv"),
        e2_controls_csv_c_state(&baseline_stats, &nohill_stats, &norep_stats),
    )?;

    let control_plot_path = out_dir.join("paper_e2_mean_c_state_over_time_controls.svg");
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

    let control_c_path = out_dir.join("paper_e2_mean_c_over_time_controls_seeds.svg");
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
        out_dir.join("paper_e2_mean_c_score_loo_over_time_controls_seeds.svg");
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

    let annotated_mean_path = out_dir.join("paper_e2_mean_c_state_over_time_seeds_annotated.svg");
    render_e2_mean_c_state_annotated(
        &annotated_mean_path,
        &baseline_stats.mean_c_state,
        &baseline_ci95_c_state,
        &nohill_stats.mean_c_state,
        &nohill_ci95_c_state,
        &norep_stats.mean_c_state,
        &norep_ci95_c_state,
        E2_BURN_IN,
        phase_mode.switch_step(),
    )?;

    let mut diversity_rows_vec = Vec::new();
    diversity_rows_vec.extend(diversity_rows("baseline", &baseline_runs));
    diversity_rows_vec.extend(diversity_rows("nohill", &nohill_runs));
    diversity_rows_vec.extend(diversity_rows("norep", &norep_runs));
    write_with_log(
        out_dir.join("paper_e2_diversity_by_seed.csv"),
        diversity_by_seed_csv(&diversity_rows_vec),
    )?;
    write_with_log(
        out_dir.join("paper_e2_diversity_summary.csv"),
        diversity_summary_csv(&diversity_rows_vec),
    )?;
    write_with_log(
        out_dir.join("paper_e2_diversity_summary_ci95.csv"),
        diversity_summary_ci95_csv(&diversity_rows_vec),
    )?;
    let diversity_plot_path = out_dir.join("paper_e2_diversity_summary.svg");
    render_diversity_summary_plot(&diversity_plot_path, &diversity_rows_vec)?;
    let diversity_ci95_plot_path = out_dir.join("paper_e2_diversity_summary_ci95.svg");
    render_diversity_summary_ci95_plot(&diversity_ci95_plot_path, &diversity_rows_vec)?;
    let figure1_path = out_dir.join("paper_e2_figure_e2_1.svg");
    render_e2_figure1(
        &figure1_path,
        &baseline_stats,
        &nohill_stats,
        &norep_stats,
        &baseline_ci95_c_state,
        &nohill_ci95_c_state,
        &norep_ci95_c_state,
        &diversity_rows_vec,
        &baseline_run.trajectory_semitones,
        phase_mode,
    )?;

    let mut hist_rows = Vec::new();
    hist_rows.extend(hist_structure_rows("baseline", &baseline_runs));
    hist_rows.extend(hist_structure_rows("nohill", &nohill_runs));
    hist_rows.extend(hist_structure_rows("norep", &norep_runs));
    write_with_log(
        out_dir.join("paper_e2_hist_structure_by_seed.csv"),
        hist_structure_by_seed_csv(&hist_rows),
    )?;
    write_with_log(
        out_dir.join("paper_e2_hist_structure_summary.csv"),
        hist_structure_summary_csv(&hist_rows),
    )?;
    let hist_plot_path = out_dir.join("paper_e2_hist_structure_summary.svg");
    render_hist_structure_summary_plot(&hist_plot_path, &hist_rows)?;

    let nohill_rep = &nohill_runs[pick_representative_run_index(&nohill_runs)];
    let norep_rep = &norep_runs[pick_representative_run_index(&norep_runs)];
    render_e2_control_histograms(out_dir, baseline_run, nohill_rep, norep_rep)?;

    let hist_min = -12.0f32;
    let hist_max = 12.0f32;
    let hist_stats_05 = e2_hist_seed_sweep(&baseline_runs, 0.5, hist_min, hist_max);
    write_with_log(
        out_dir.join("paper_e2_interval_hist_post_seed_sweep_bw0p50.csv"),
        e2_hist_seed_sweep_csv(&hist_stats_05),
    )?;
    let hist_plot_05 = out_dir.join("paper_e2_interval_hist_post_seed_sweep_bw0p50.svg");
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
    write_with_log(
        out_dir.join("paper_e2_interval_hist_post_seed_sweep_bw0p25.csv"),
        e2_hist_seed_sweep_csv(&hist_stats_025),
    )?;
    let hist_plot_025 = out_dir.join("paper_e2_interval_hist_post_seed_sweep_bw0p25.svg");
    render_hist_mean_std(
        &hist_plot_025,
        &format!("E2 {post_label_title} Interval Histogram (seed sweep, mean frac, bin=0.25st)"),
        &hist_stats_025.centers,
        &hist_stats_025.mean_frac,
        &hist_stats_025.std_frac,
        0.25,
        "mean fraction",
    )?;
    let hist_plot_025_paper =
        out_dir.join("paper_e2_interval_hist_post_seed_sweep_bw0p25_paper.svg");
    render_hist_mean_std_fraction_auto_y(
        &hist_plot_025_paper,
        &format!("E2 {post_label_title} Interval Histogram (paper, bin=0.25st)"),
        &hist_stats_025.centers,
        &hist_stats_025.mean_frac,
        &hist_stats_025.std_frac,
        0.25,
        "semitones",
        &[-7.0, -4.0, -3.0, 3.0, 4.0, 7.0],
    )?;
    let (folded_centers, folded_mean, folded_std) = fold_hist_abs_semitones(
        &hist_stats_025.centers,
        &hist_stats_025.mean_frac,
        &hist_stats_025.std_frac,
        0.25,
    );
    write_with_log(
        out_dir.join("paper_e2_anchor_interval_hist_post_folded.csv"),
        folded_hist_csv(&folded_centers, &folded_mean, &folded_std, hist_stats_025.n),
    )?;
    let folded_hist_plot = out_dir.join("paper_e2_anchor_interval_hist_post_folded.svg");
    render_anchor_hist_post_folded(
        &folded_hist_plot,
        &hist_stats_025.centers,
        &hist_stats_025.mean_frac,
        &hist_stats_025.std_frac,
        &folded_centers,
        &folded_mean,
        &folded_std,
        0.25,
    )?;

    let (pairwise_hist_stats, pairwise_n_pairs) =
        e2_pairwise_hist_seed_sweep(&baseline_runs, E2_PAIRWISE_BIN_ST, 0.0, 12.0);
    let (pairwise_hist_nohill, pairwise_n_pairs_nohill) =
        e2_pairwise_hist_seed_sweep(&nohill_runs, E2_PAIRWISE_BIN_ST, 0.0, 12.0);
    let (pairwise_hist_norep, pairwise_n_pairs_norep) =
        e2_pairwise_hist_seed_sweep(&norep_runs, E2_PAIRWISE_BIN_ST, 0.0, 12.0);
    let pairwise_n_pairs_controls = pairwise_n_pairs
        .min(pairwise_n_pairs_nohill)
        .min(pairwise_n_pairs_norep);
    write_with_log(
        out_dir.join("paper_e2_pairwise_interval_histogram_seeds.csv"),
        e2_pairwise_hist_seed_sweep_csv(&pairwise_hist_stats, pairwise_n_pairs),
    )?;
    write_with_log(
        out_dir.join("paper_e2_pairwise_interval_histogram_seeds_ci95.csv"),
        e2_pairwise_hist_seed_sweep_ci95_csv(&pairwise_hist_stats, pairwise_n_pairs),
    )?;
    let pairwise_hist_plot = out_dir.join("paper_e2_pairwise_interval_histogram_seeds.svg");
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
    let pairwise_ci95_frac =
        std_series_to_ci95(&pairwise_hist_stats.std_frac, pairwise_hist_stats.n);
    let pairwise_hist_plot_paper =
        out_dir.join("paper_e2_pairwise_interval_histogram_seeds_paper.svg");
    render_pairwise_histogram_paper(
        &pairwise_hist_plot_paper,
        "E2 Pairwise Interval Histogram (paper style, 95% CI)",
        &pairwise_hist_stats.centers,
        &pairwise_hist_stats.mean_frac,
        &pairwise_ci95_frac,
        E2_PAIRWISE_BIN_ST,
    )?;
    write_with_log(
        out_dir.join("paper_e2_pairwise_interval_histogram_controls_seeds.csv"),
        e2_pairwise_hist_controls_seed_sweep_csv(
            &pairwise_hist_stats,
            &pairwise_hist_nohill,
            &pairwise_hist_norep,
            pairwise_n_pairs_controls,
        ),
    )?;
    write_with_log(
        out_dir.join("paper_e2_pairwise_interval_histogram_controls_seeds_ci95.csv"),
        e2_pairwise_hist_controls_seed_sweep_ci95_csv(
            &pairwise_hist_stats,
            &pairwise_hist_nohill,
            &pairwise_hist_norep,
            pairwise_n_pairs_controls,
        ),
    )?;
    let pairwise_controls_plot =
        out_dir.join("paper_e2_pairwise_interval_histogram_controls_seeds.svg");
    render_pairwise_histogram_controls_overlay(
        &pairwise_controls_plot,
        "E2 Pairwise Interval Histogram (controls overlay)",
        &pairwise_hist_stats.centers,
        &pairwise_hist_stats.mean_frac,
        &pairwise_hist_nohill.mean_frac,
        &pairwise_hist_norep.mean_frac,
    )?;

    let mut consonant_rows = Vec::new();
    consonant_rows.extend(consonant_mass_rows_for_condition(
        "baseline",
        &baseline_runs,
    ));
    consonant_rows.extend(consonant_mass_rows_for_condition("nohill", &nohill_runs));
    consonant_rows.extend(consonant_mass_rows_for_condition("norep", &norep_runs));
    write_with_log(
        out_dir.join("paper_e2_consonant_mass_by_seed.csv"),
        consonant_mass_by_seed_csv(&consonant_rows),
    )?;
    write_with_log(
        out_dir.join("paper_e2_consonant_mass_summary.csv"),
        consonant_mass_summary_csv(&consonant_rows),
    )?;
    write_with_log(
        out_dir.join("paper_e2_consonant_mass_stats.csv"),
        consonant_mass_stats_csv(&consonant_rows),
    )?;
    let consonant_mass_plot = out_dir.join("paper_e2_consonant_mass_summary.svg");
    render_consonant_mass_summary_plot(&consonant_mass_plot, &consonant_rows)?;

    let figure2_path = out_dir.join("paper_e2_figure_e2_2.svg");
    render_e2_figure2(
        &figure2_path,
        &pairwise_hist_stats.centers,
        &pairwise_hist_stats.mean_frac,
        &pairwise_ci95_frac,
        &pairwise_hist_nohill.mean_frac,
        &pairwise_hist_norep.mean_frac,
        &consonant_rows,
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
    write_with_log(
        out_dir.join("paper_e2_interval_hist_post_controls_seed_sweep_bw0p50.csv"),
        controls_csv,
    )?;

    let control_hist_plot =
        out_dir.join("paper_e2_interval_hist_post_controls_seed_sweep_bw0p50.svg");
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
    write_with_log(out_dir.join("paper_e2_delta_c_by_seed.csv"), delta_csv)?;
    write_with_log(out_dir.join("paper_e2_delta_c_summary.csv"), delta_summary)?;

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
    E2_SWEEPS.saturating_sub(1)
}

fn e2_accept_temperature(step: usize, phase_mode: E2PhaseMode) -> f32 {
    if !E2_ACCEPT_ENABLED {
        return 0.0;
    }
    if E2_ACCEPT_TAU_STEPS <= 0.0 {
        return E2_ACCEPT_T0.max(0.0);
    }
    let mut phase_step = step;
    if E2_ACCEPT_RESET_ON_PHASE
        && let Some(switch_step) = phase_mode.switch_step()
        && step >= switch_step
    {
        phase_step = step - switch_step;
    }
    E2_ACCEPT_T0.max(0.0) * (-(phase_step as f32) / E2_ACCEPT_TAU_STEPS).exp()
}

#[cfg(test)]
fn e2_should_attempt_update(agent_id: usize, step: usize, u_move: f32) -> bool {
    match E2_UPDATE_SCHEDULE {
        E2UpdateSchedule::Checkerboard => (agent_id + step).is_multiple_of(2),
        E2UpdateSchedule::Lazy => u_move < E2_LAZY_MOVE_PROB.clamp(0.0, 1.0),
        E2UpdateSchedule::RandomSingle => true,
    }
}

fn e2_should_block_backtrack(phase_mode: E2PhaseMode, step: usize) -> bool {
    if !E2_ANTI_BACKTRACK_ENABLED {
        return false;
    }
    if E2_ANTI_BACKTRACK_PRE_SWITCH_ONLY && let Some(switch_step) = phase_mode.switch_step() {
        return step < switch_step;
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

    let mut mean_c_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_c_state_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_c_score_loo_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_c_score_chosen_loo_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_score_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_repulsion_series = Vec::with_capacity(E2_SWEEPS);
    let mut moved_frac_series = Vec::with_capacity(E2_SWEEPS);
    let mut accepted_worse_frac_series = Vec::with_capacity(E2_SWEEPS);
    let mut attempted_update_frac_series = Vec::with_capacity(E2_SWEEPS);
    let mut moved_given_attempt_frac_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_abs_delta_semitones_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_abs_delta_semitones_moved_series = Vec::with_capacity(E2_SWEEPS);
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
        .map(|_| Vec::with_capacity(E2_SWEEPS))
        .collect::<Vec<_>>();
    let mut trajectory_c_state = (0..E2_N_AGENTS)
        .map(|_| Vec::with_capacity(E2_SWEEPS))
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
    for sweep in 0..E2_SWEEPS {
        if sweep == E2_ANCHOR_SHIFT_STEP {
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
                step: sweep,
                anchor_hz_before: before,
                anchor_hz_after: anchor_hz_current,
                count_min,
                count_max,
                respawned,
            };
            min_idx = new_min;
            max_idx = new_max;
        }
        if let Some(switch_step) = phase_switch_step
            && sweep == switch_step
        {
            backtrack_targets.clone_from_slice(&agent_indices);
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
        let use_nohill = matches!(condition, E2Condition::NoHillClimb);
        let mut env_loo = if use_nohill {
            vec![0.0f32; env_scan.len()]
        } else {
            Vec::new()
        };
        let mut density_loo = if use_nohill {
            vec![0.0f32; density_scan.len()]
        } else {
            Vec::new()
        };
        let mean_c_score_loo_current = if use_nohill {
            mean_c_score_loo_at_indices_with_prev_reused(
                space,
                &workspace,
                &env_scan,
                &density_scan,
                &du_scan,
                &agent_indices,
                &agent_indices,
                &mut env_loo,
                &mut density_loo,
            )
        } else {
            f32::NAN
        };
        mean_c_series.push(mean_c);
        mean_c_state_series.push(mean_c_state);

        for (agent_id, &idx) in agent_indices.iter().enumerate() {
            let semitone = 12.0 * log2_ratio_scan[idx];
            trajectory_semitones[agent_id].push(semitone);
            trajectory_c_state[agent_id].push(c_state_scan[idx]);
        }

        if sweep >= E2_BURN_IN {
            let target = if anchor_shift_enabled && sweep < E2_ANCHOR_SHIFT_STEP {
                &mut semitone_samples_pre
            } else {
                &mut semitone_samples_post
            };
            target.extend(agent_indices.iter().map(|&idx| 12.0 * log2_ratio_scan[idx]));
        }

        let temperature = e2_accept_temperature(sweep, phase_mode);
        let score_sign = phase_mode.score_sign(sweep);
        let block_backtrack = e2_should_block_backtrack(phase_mode, sweep);
        let positions_before_update = agent_indices.clone();
        let mut stats = match condition {
            E2Condition::Baseline => update_e2_sweep_scored_loo(
                E2_UPDATE_SCHEDULE,
                &mut agent_indices,
                &positions_before_update,
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
                sweep,
                block_backtrack,
                if block_backtrack {
                    Some(backtrack_targets.as_slice())
                } else {
                    None
                },
                &mut rng,
            ),
            E2Condition::NoRepulsion => update_e2_sweep_scored_loo(
                E2_UPDATE_SCHEDULE,
                &mut agent_indices,
                &positions_before_update,
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
                sweep,
                block_backtrack,
                if block_backtrack {
                    Some(backtrack_targets.as_slice())
                } else {
                    None
                },
                &mut rng,
            ),
            E2Condition::NoHillClimb => {
                let (moved, attempts, abs_delta_sum, abs_delta_moved_sum) = update_e2_sweep_nohill(
                    E2_UPDATE_SCHEDULE,
                    &mut agent_indices,
                    &positions_before_update,
                    &log2_ratio_scan,
                    min_idx,
                    max_idx,
                    k_bins,
                    sweep,
                    &mut rng,
                );
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
                    stats.attempted_update_frac = attempts as f32 / agent_indices.len() as f32;
                    stats.moved_given_attempt_frac = if attempts > 0 {
                        moved as f32 / attempts as f32
                    } else {
                        0.0
                    };
                }
                if moved > 0 {
                    stats.mean_abs_delta_semitones_moved = abs_delta_moved_sum / moved as f32;
                }
                stats
            }
        };
        e2_update_backtrack_targets(
            &mut backtrack_targets,
            &positions_before_update,
            &agent_indices,
        );

        let mean_c_score_loo_chosen = if use_nohill {
            mean_c_score_loo_at_indices_with_prev_reused(
                space,
                &workspace,
                &env_scan,
                &density_scan,
                &du_scan,
                &positions_before_update,
                &agent_indices,
                &mut env_loo,
                &mut density_loo,
            )
        } else {
            stats.mean_c_score_chosen_loo
        };
        stats.mean_c_score_current_loo = if use_nohill {
            mean_c_score_loo_current
        } else {
            stats.mean_c_score_current_loo
        };
        stats.mean_c_score_chosen_loo = mean_c_score_loo_chosen;
        let condition_label = match condition {
            E2Condition::Baseline => "baseline",
            E2Condition::NoRepulsion => "norep",
            E2Condition::NoHillClimb => "nohill",
        };
        debug_assert!(
            stats.mean_c_score_current_loo.is_finite(),
            "mean_c_score_current_loo not finite (cond={condition_label}, sweep={sweep}, value={})",
            stats.mean_c_score_current_loo
        );
        debug_assert!(
            stats.mean_c_score_chosen_loo.is_finite(),
            "mean_c_score_chosen_loo not finite (cond={condition_label}, sweep={sweep}, value={})",
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
    let seeds = &E2_SEEDS;
    let max_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let per_condition_max_threads = (max_threads / 3).max(1);
    let worker_count = per_condition_max_threads.min(seeds.len()).max(1);
    let runs = if worker_count <= 1 || seeds.len() <= 1 {
        let mut runs = Vec::with_capacity(seeds.len());
        for &seed in seeds {
            runs.push(run_e2_once(
                space,
                anchor_hz,
                seed,
                condition,
                step_semitones,
                phase_mode,
            ));
        }
        runs
    } else {
        let next = AtomicUsize::new(0);
        let runs = Mutex::new({
            let mut runs = Vec::with_capacity(seeds.len());
            runs.resize_with(seeds.len(), || None);
            runs
        });
        std::thread::scope(|scope| {
            for _ in 0..worker_count {
                scope.spawn(|| {
                    loop {
                        let idx = next.fetch_add(1, Ordering::Relaxed);
                        if idx >= seeds.len() {
                            break;
                        }
                        let seed = seeds[idx];
                        let run = run_e2_once(
                            space,
                            anchor_hz,
                            seed,
                            condition,
                            step_semitones,
                            phase_mode,
                        );
                        let mut guard = runs.lock().expect("runs lock poisoned");
                        guard[idx] = Some(run);
                    }
                });
            }
        });
        runs.into_inner()
            .expect("runs lock poisoned")
            .into_iter()
            .map(|run| run.expect("missing run"))
            .collect()
    };

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
    write_with_log(out_dir.join("paper_e3_policy_params.csv"), policy_csv)?;
    write_with_log(
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
            write_with_log(lifetimes_path, e3_lifetimes_csv(&deaths))?;

            let arrays = e3_extract_arrays(&deaths);

            let scatter_firstk_path = out_dir.join(format!(
                "paper_e3_firstk_vs_lifetime_seed{}_{}.svg",
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
                "paper_e3_birth_vs_lifetime_seed{}_{}.svg",
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
                "paper_e3_survival_by_firstk_seed{}_{}.svg",
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
                "paper_e3_survival_by_firstk_q25q75_seed{}_{}.svg",
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
                    "paper_e3_attack_vs_lifetime_seed{}_{}.svg",
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
                    "paper_e3_survival_by_attack_seed{}_{}.svg",
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
                write_with_log(out_dir.join("paper_e3_lifetimes.csv"), legacy_csv)?;
                let legacy_scatter = out_dir.join("paper_e3_firstk_vs_lifetime.svg");
                render_consonance_lifetime_scatter(&legacy_scatter, &legacy_deaths)?;
                let legacy_survival = out_dir.join("paper_e3_survival_curve.svg");
                render_survival_curve(&legacy_survival, &legacy_deaths)?;
                let legacy_survival_c_state = out_dir.join("paper_e3_survival_by_c_state.svg");
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
        write_with_log(out_dir.join("paper_e3_representative_seed.txt"), rep_note)?;

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
            let compare_scatter = out_dir.join("paper_e3_firstk_scatter_compare.svg");
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
            let compare_surv = out_dir.join("paper_e3_firstk_survival_compare.svg");
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
            let compare_surv_q = out_dir.join("paper_e3_firstk_survival_compare_q25q75.svg");
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
    let pooled_scatter_path = out_dir.join("paper_e3_firstk_scatter_compare_pooled.svg");
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
    let pooled_surv_path = out_dir.join("paper_e3_firstk_survival_compare_pooled.svg");
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
    let pooled_surv_q_path = out_dir.join("paper_e3_firstk_survival_compare_pooled_q25q75.svg");
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
    write_with_log(
        out_dir.join("paper_e3_summary_pooled.csv"),
        pooled_summary_csv,
    )?;

    let pooled_hist_path = out_dir.join("paper_e3_firstk_hist.svg");
    render_e3_firstk_histogram(
        &pooled_hist_path,
        &pooled_baseline.c_state_firstk,
        &pooled_norecharge.c_state_firstk,
        0.02,
        0.5,
    )?;

    write_with_log(out_dir.join("paper_e3_lifetimes_long.csv"), long_csv)?;
    write_with_log(out_dir.join("paper_e3_summary_by_seed.csv"), summary_csv)?;

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

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
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
    write_with_log(&csv_path, e5_kick_csv(&sim_main, &sim_ctrl))?;

    let summary_path = out_dir.join("paper_e5_kick_summary.csv");
    write_with_log(&summary_path, e5_kick_summary_csv(&sim_main, &sim_ctrl))?;

    let meta_path = out_dir.join("paper_e5_meta.txt");
    write_with_log(&meta_path, e5_meta_text(E5_STEPS))?;

    let order_path = out_dir.join("paper_e5_order_over_time.svg");
    render_e5_order_plot(&order_path, &sim_main.series, &sim_ctrl.series)?;

    let delta_path = out_dir.join("paper_e5_delta_phi_over_time.svg");
    render_e5_delta_phi_plot(&delta_path, &sim_main.series, &sim_ctrl.series)?;

    let plv_path = out_dir.join("paper_e5_plv_over_time.svg");
    render_e5_plv_plot(&plv_path, &sim_main.series, &sim_ctrl.series)?;

    let seed_rows = e5_seed_sweep_rows(E5_STEPS);
    let seed_csv_path = out_dir.join("paper_e5_seed_sweep.csv");
    write_with_log(&seed_csv_path, e5_seed_sweep_csv(&seed_rows))?;
    let seed_plot_path = out_dir.join("paper_e5_seed_sweep.svg");
    render_e5_seed_sweep_plot(&seed_plot_path, &seed_rows)?;

    let bins = phase_hist_bins(
        sim_main
            .phase_hist_samples
            .len()
            .max(sim_ctrl.phase_hist_samples.len()),
    );
    let phase_path = out_dir.join("paper_e5_phase_diff_histogram.svg");
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

    let root = bitmap_root(out_path, (1600, 1200)).into_drawing_area();
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

fn render_e1_h_mirror_triplet_plot(
    out_path: &Path,
    anchor_hz: f32,
    log2_ratio_scan: &[f32],
    h_m0: &[f32],
    h_m05: &[f32],
    h_m1: &[f32],
) -> Result<(), Box<dyn Error>> {
    let x_min = -3.0f32;
    let x_max = 3.0f32;
    let n = log2_ratio_scan
        .len()
        .min(h_m0.len())
        .min(h_m05.len())
        .min(h_m1.len());

    let mut p_m0 = Vec::new();
    let mut p_m05 = Vec::new();
    let mut p_m1 = Vec::new();
    for i in 0..n {
        let x = log2_ratio_scan[i];
        if x < x_min || x > x_max {
            continue;
        }
        p_m0.push((x, h_m0[i]));
        p_m05.push((x, h_m05[i]));
        p_m1.push((x, h_m1[i]));
    }

    let y_max = h_m0
        .iter()
        .chain(h_m05.iter())
        .chain(h_m1.iter())
        .copied()
        .fold(0.0f32, f32::max)
        .max(1e-6)
        * 1.1;

    let root = bitmap_root(out_path, (1600, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let ratio_guides = [0.5f32, 6.0 / 5.0, 1.25, 4.0 / 3.0, 1.5, 5.0 / 3.0, 2.0];
    let ratio_guides_log2: Vec<f32> = ratio_guides.iter().map(|r| r.log2()).collect();

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "E1 Harmonicity H(f) by mirror_weight | anchor {} Hz",
                anchor_hz
            ),
            ("sans-serif", 24),
        )
        .margin(12)
        .x_label_area_size(42)
        .y_label_area_size(64)
        .build_cartesian_2d(x_min..x_max, 0.0f32..y_max)?;

    chart
        .configure_mesh()
        .x_desc("log2(f / f_anchor)")
        .y_desc("H potential")
        .draw()?;

    for &x in &ratio_guides_log2 {
        if x >= x_min && x <= x_max {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x, 0.0), (x, y_max)],
                BLACK.mix(0.15),
            )))?;
        }
    }

    chart
        .draw_series(LineSeries::new(p_m0, &BLUE))?
        .label("m0 (mirror_weight=0.0)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], BLUE));
    chart
        .draw_series(LineSeries::new(p_m05, &GREEN))?
        .label("m05 (mirror_weight=0.5)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], GREEN));
    chart
        .draw_series(LineSeries::new(p_m1, &RED))?
        .label("m1 (mirror_weight=1.0)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], RED));
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK.mix(0.25))
        .draw()?;

    root.present()?;
    Ok(())
}

fn render_e1_h_mirror_diff_plot(
    out_path: &Path,
    anchor_hz: f32,
    log2_ratio_scan: &[f32],
    h_m0: &[f32],
    h_m1: &[f32],
) -> Result<(), Box<dyn Error>> {
    let x_min = -3.0f32;
    let x_max = 3.0f32;
    let n = log2_ratio_scan.len().min(h_m0.len()).min(h_m1.len());
    let mut diff_points = Vec::new();
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for i in 0..n {
        let x = log2_ratio_scan[i];
        if x < x_min || x > x_max {
            continue;
        }
        let d = h_m0[i] - h_m1[i];
        diff_points.push((x, d));
        if d.is_finite() {
            y_min = y_min.min(d);
            y_max = y_max.max(d);
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = -1.0;
        y_max = 1.0;
    }
    let pad = ((y_max - y_min).abs() * 0.1).max(1e-4);
    let y_lo = (y_min - pad).min(0.0);
    let y_hi = (y_max + pad).max(0.0);

    let root = bitmap_root(out_path, (1600, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let ratio_guides = [0.5f32, 6.0 / 5.0, 1.25, 4.0 / 3.0, 1.5, 5.0 / 3.0, 2.0];
    let ratio_guides_log2: Vec<f32> = ratio_guides.iter().map(|r| r.log2()).collect();

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "E1 Harmonicity Difference H_m0(f) - H_m1(f) | anchor {} Hz",
                anchor_hz
            ),
            ("sans-serif", 24),
        )
        .margin(12)
        .x_label_area_size(42)
        .y_label_area_size(70)
        .build_cartesian_2d(x_min..x_max, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("log2(f / f_anchor)")
        .y_desc("H(m0) - H(m1)")
        .draw()?;

    for &x in &ratio_guides_log2 {
        if x >= x_min && x <= x_max {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x, y_lo), (x, y_hi)],
                BLACK.mix(0.12),
            )))?;
        }
    }
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(x_min, 0.0), (x_max, 0.0)],
        BLACK.mix(0.35),
    )))?;
    chart.draw_series(LineSeries::new(diff_points, &BLACK))?;

    root.present()?;
    Ok(())
}

fn plot_e4_mirror_sweep(
    out_dir: &Path,
    anchor_hz: f32,
    emit_hist_files: bool,
    emit_kernel_gate: bool,
    emit_wr_probe: bool,
) -> Result<(), Box<dyn Error>> {
    if !E4_EMIT_LEGACY_OUTPUTS {
        let weights = build_weight_grid(E4_WEIGHT_COARSE_STEP);
        let (_run_records, _hist_records, _tail_rows, tail_agent_rows) = run_e4_sweep_for_weights(
            out_dir,
            anchor_hz,
            &weights,
            &E4_SEEDS,
            E4_BIN_WIDTHS[0],
            &E4_EPS_CENTS,
            false,
            true,
        )?;

        let tail_agents_csv_path = out_dir.join("e4_tail_agents.csv");
        write_with_log(&tail_agents_csv_path, e4_tail_agents_csv(&tail_agent_rows))?;

        let bind_metrics = e4_bind_metrics_from_tail_agents(&tail_agent_rows);
        let bind_summary = e4_bind_summary_rows(&bind_metrics);
        let binding_metrics_raw_path = out_dir.join("paper_e4_binding_metrics_raw.csv");
        write_with_log(
            &binding_metrics_raw_path,
            e4_binding_metrics_raw_csv(&bind_metrics),
        )?;
        let binding_metrics_summary_path = out_dir.join("paper_e4_binding_metrics_summary.csv");
        write_with_log(
            &binding_metrics_summary_path,
            e4_binding_metrics_summary_csv(&bind_summary),
        )?;

        let fingerprint_rows = e4_fingerprint_rows_from_tail_agents(&tail_agent_rows);
        let fingerprint_summary = e4_fingerprint_summary_rows(&fingerprint_rows);
        let fingerprint_raw_path = out_dir.join("paper_e4_fingerprint_raw.csv");
        write_with_log(
            &fingerprint_raw_path,
            e4_fingerprint_raw_csv(&fingerprint_rows),
        )?;
        let fingerprint_summary_path = out_dir.join("paper_e4_fingerprint_summary.csv");
        write_with_log(
            &fingerprint_summary_path,
            e4_fingerprint_summary_csv(&fingerprint_summary),
        )?;

        let binding_phase_png_path = out_dir.join("paper_e4_binding_phase_diagram.png");
        render_e4_binding_phase_diagram_png(&binding_phase_png_path, &bind_summary)?;
        let delta_bind_png_path = out_dir.join("paper_e4_delta_bind.png");
        render_e4_delta_bind_png(&delta_bind_png_path, &bind_summary)?;
        let fingerprint_heatmap_png_path = out_dir.join("paper_e4_fingerprint_heatmap.png");
        render_e4_fingerprint_heatmap_png(&fingerprint_heatmap_png_path, &fingerprint_summary)?;

        let meta = e4_paper_meta();
        let space = Log2Space::new(meta.fmin, meta.fmax, meta.bins_per_oct);
        let (_erb_scan, du_scan) = erb_grid_for_space(&space);
        let final_freqs = e4_final_freqs_by_mw_seed(&tail_agent_rows);
        let k = k_from_semitones(E4_DYNAMICS_STEP_SEMITONES.max(1e-3));
        let (diag_rows, diag_peak_rows) = e4_diag_rows_from_final_freqs(
            &final_freqs,
            &space,
            anchor_hz,
            &du_scan,
            E4_ABCD_TRACE_STEPS,
            E4_DYNAMICS_BASE_LAMBDA.max(0.0),
            E4_DYNAMICS_REPULSION_SIGMA.max(1e-6),
            k,
        );
        write_with_log(
            out_dir.join("e4_oracle_step_trace.csv"),
            e4_diag_step_rows_csv(&diag_rows),
        )?;
        write_with_log(
            out_dir.join("e4_peaks_by_mw.csv"),
            e4_peaks_by_mw_csv(&diag_peak_rows),
        )?;
        let landscape_delta_rows = e4_landscape_delta_rows(&weights, &space, anchor_hz, &du_scan);
        write_with_log(
            out_dir.join("e4_landscape_delta_by_mw.csv"),
            e4_landscape_delta_by_mw_csv(&landscape_delta_rows),
        )?;
        render_e4_gap_over_time(
            &out_dir.join("e4_gap_global_over_time.svg"),
            &diag_rows,
            true,
        )?;
        render_e4_gap_over_time(
            &out_dir.join("e4_gap_reach_over_time.svg"),
            &diag_rows,
            false,
        )?;
        render_e4_gap_global_by_mw(&out_dir.join("e4_gap_global_by_mw.svg"), &diag_rows)?;
        render_e4_peak_positions_vs_mw(
            &out_dir.join("e4_peak_positions_vs_mw.svg"),
            &diag_peak_rows,
            E4_DIAG_PEAK_TOP_K,
        )?;
        if emit_wr_probe {
            plot_e4_mirror_sweep_wr_cut(out_dir, anchor_hz)?;
        }

        if emit_kernel_gate {
            let kernel_gate_path = out_dir.join("paper_e4_kernel_gate.svg");
            render_e4_kernel_gate(&kernel_gate_path, anchor_hz)?;
        }

        let _ = emit_hist_files;
        return Ok(());
    }

    let coarse_weights = build_weight_grid(E4_WEIGHT_COARSE_STEP);
    let primary_bin = E4_BIN_WIDTHS[0];
    let primary_eps = E4_EPS_CENTS[0];
    let (mut run_records, mut hist_records, mut tail_rows, mut tail_agent_rows) =
        run_e4_sweep_for_weights(
            out_dir,
            anchor_hz,
            &coarse_weights,
            &E4_SEEDS,
            primary_bin,
            &E4_EPS_CENTS,
            emit_hist_files,
            true,
        )?;

    let mut weights = coarse_weights.clone();
    let fine_weights = refine_weights_from_sign_change(&run_records, primary_bin, primary_eps);
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
        let (more_runs, more_hists, more_tail, more_tail_agents) = run_e4_sweep_for_weights(
            out_dir,
            anchor_hz,
            &fine_only,
            &E4_SEEDS,
            primary_bin,
            &E4_EPS_CENTS,
            emit_hist_files,
            true,
        )?;
        run_records.extend(more_runs);
        hist_records.extend(more_hists);
        tail_rows.extend(more_tail);
        tail_agent_rows.extend(more_tail_agents);
    }

    for &bin_width in E4_BIN_WIDTHS.iter().skip(1) {
        let (more_runs, more_hists, more_tail, _more_tail_agents) = run_e4_sweep_for_weights(
            out_dir,
            anchor_hz,
            &weights,
            &E4_SEEDS,
            bin_width,
            &E4_EPS_CENTS,
            emit_hist_files,
            false,
        )?;
        run_records.extend(more_runs);
        hist_records.extend(more_hists);
        tail_rows.extend(more_tail);
    }

    let runs_csv_path = out_dir.join("e4_mirror_sweep_runs.csv");
    write_with_log(&runs_csv_path, e4_runs_csv(&run_records))?;
    let tail_csv_path = out_dir.join("paper_e4_tail_interval_timeseries.csv");
    write_with_log(&tail_csv_path, e4_tail_interval_csv(&tail_rows))?;
    let tail_agents_csv_path = out_dir.join("e4_tail_agents.csv");
    write_with_log(&tail_agents_csv_path, e4_tail_agents_csv(&tail_agent_rows))?;

    let summaries = summarize_e4_runs(&run_records);
    let summary_csv_path = out_dir.join("e4_mirror_sweep_summary.csv");
    write_with_log(&summary_csv_path, e4_summary_csv(&summaries))?;
    let meta_diff_path = out_dir.join("paper_e4_protocol_meta_diff.csv");
    write_with_log(
        &meta_diff_path,
        e4_protocol_meta_diff_csv(anchor_hz, primary_bin, primary_eps),
    )?;
    let fixed_check_path = out_dir.join("paper_e4_fixed_except_mirror_check.csv");
    write_with_log(
        &fixed_check_path,
        e4_fixed_except_mirror_check_csv(&run_records, &tail_agent_rows),
    )?;

    let bind_metrics = e4_bind_metrics_from_tail_agents(&tail_agent_rows);
    let bind_metrics_path = out_dir.join("e4_bind_metrics.csv");
    write_with_log(&bind_metrics_path, e4_bind_metrics_csv(&bind_metrics))?;
    let bind_summary = e4_bind_summary_rows(&bind_metrics);
    let bind_summary_path = out_dir.join("e4_bind_summary.csv");
    write_with_log(&bind_summary_path, e4_bind_summary_csv(&bind_summary))?;
    let binding_metrics_raw_path = out_dir.join("paper_e4_binding_metrics_raw.csv");
    write_with_log(
        &binding_metrics_raw_path,
        e4_binding_metrics_raw_csv(&bind_metrics),
    )?;
    let binding_metrics_summary_path = out_dir.join("paper_e4_binding_metrics_summary.csv");
    write_with_log(
        &binding_metrics_summary_path,
        e4_binding_metrics_summary_csv(&bind_summary),
    )?;
    let fingerprint_rows = e4_fingerprint_rows_from_tail_agents(&tail_agent_rows);
    let fingerprint_raw_path = out_dir.join("paper_e4_fingerprint_raw.csv");
    write_with_log(
        &fingerprint_raw_path,
        e4_fingerprint_raw_csv(&fingerprint_rows),
    )?;
    let fingerprint_summary = e4_fingerprint_summary_rows(&fingerprint_rows);
    let fingerprint_summary_path = out_dir.join("paper_e4_fingerprint_summary.csv");
    write_with_log(
        &fingerprint_summary_path,
        e4_fingerprint_summary_csv(&fingerprint_summary),
    )?;

    let delta_effects = e4_delta_effects_from_summary(&summaries);
    let delta_effects_path = out_dir.join("e4_delta_effects.csv");
    write_with_log(&delta_effects_path, e4_delta_effects_csv(&delta_effects))?;

    let regression_rows = e4_regression_rows(&summaries);
    let regression_path = out_dir.join("e4_regression.csv");
    write_with_log(&regression_path, e4_regression_csv(&regression_rows))?;

    let endpoint_rows = e4_endpoint_effect_rows(&run_records);
    let endpoint_path = out_dir.join("e4_endpoint_effect.csv");
    write_with_log(&endpoint_path, e4_endpoint_effect_csv(&endpoint_rows))?;

    let seed_slopes = e4_seed_slopes_rows(&run_records);
    let seed_slopes_path = out_dir.join("e4_seed_slopes.csv");
    write_with_log(&seed_slopes_path, e4_seed_slopes_csv(&seed_slopes))?;

    let run_level_rows = e4_run_level_regression_rows(&run_records);
    let run_level_path = out_dir.join("e4_run_level_regression.csv");
    write_with_log(
        &run_level_path,
        e4_run_level_regression_csv(&run_level_rows),
    )?;

    let seed_slope_meta = e4_seed_slope_meta_rows(&seed_slopes);
    let seed_slope_meta_path = out_dir.join("e4_seed_slope_meta.csv");
    write_with_log(
        &seed_slope_meta_path,
        e4_seed_slope_meta_csv(&seed_slope_meta),
    )?;

    let third_mass_rows = e4_total_third_mass_rows(&run_records);
    let third_mass_path = out_dir.join("e4_total_third_mass.csv");
    write_with_log(&third_mass_path, e4_total_third_mass_csv(&third_mass_rows))?;

    let overlay_path = out_dir.join(format!(
        "paper_e4_hist_overlay_bw{}.svg",
        format_float_token(primary_bin)
    ));
    render_e4_hist_overlay(&overlay_path, &hist_records, primary_bin, &E4_REP_WEIGHTS)?;
    let fig1_soft_path = out_dir.join("paper_e4_figure1_mirror_vs_delta_t_soft.svg");
    render_e4_figure1_mirror_vs_delta_t(
        &fig1_soft_path,
        &run_records,
        primary_bin,
        primary_eps,
        "soft",
        "main",
    )?;
    let fig1_hard_path = out_dir.join("paper_e4_figure1_mirror_vs_delta_t_hard_appendix.svg");
    render_e4_figure1_mirror_vs_delta_t(
        &fig1_hard_path,
        &run_records,
        primary_bin,
        primary_eps,
        "hard",
        "appendix",
    )?;
    let fig2_path = out_dir.join("paper_e4_figure2_interval_hist_cents_triptych.svg");
    render_e4_figure2_interval_hist_triptych(&fig2_path, &hist_records, primary_bin)?;
    let fig3_path = out_dir.join("paper_e4_figure3_interval_cents_heatmap.svg");
    render_e4_interval_heatmap(&fig3_path, &hist_records, primary_bin)?;
    let bind_main_path = out_dir.join("paper_e4_bind_vs_weight.svg");
    render_e4_bind_vs_weight(&bind_main_path, &bind_summary)?;
    let bind_fit_path = out_dir.join("paper_e4_root_ceiling_fit_vs_weight.svg");
    render_e4_root_ceiling_fit_vs_weight(&bind_fit_path, &bind_summary)?;
    let fingerprint_path = out_dir.join("paper_e4_interval_fingerprint_heatmap.svg");
    render_e4_interval_heatmap(&fingerprint_path, &hist_records, primary_bin)?;
    let binding_phase_png_path = out_dir.join("paper_e4_binding_phase_diagram.png");
    render_e4_binding_phase_diagram_png(&binding_phase_png_path, &bind_summary)?;
    let delta_bind_png_path = out_dir.join("paper_e4_delta_bind.png");
    render_e4_delta_bind_png(&delta_bind_png_path, &bind_summary)?;
    let fingerprint_heatmap_png_path = out_dir.join("paper_e4_fingerprint_heatmap.png");
    render_e4_fingerprint_heatmap_png(&fingerprint_heatmap_png_path, &fingerprint_summary)?;

    for &bin_width in &E4_BIN_WIDTHS {
        let bw_token = format_float_token(bin_width);
        for &eps_cents in &E4_EPS_CENTS {
            let eps_token = fmt_eps_token(eps_cents);
            let delta_path = out_dir.join(format!(
                "paper_e4_delta_vs_weight_bw{bw_token}_eps{eps_token}.svg"
            ));
            render_e4_delta_plot(&delta_path, &summaries, bin_width, eps_cents)?;

            let spaghetti_path = out_dir.join(format!(
                "paper_e4_delta_spaghetti_bw{bw_token}_eps{eps_token}.svg"
            ));
            render_e4_delta_spaghetti(
                &spaghetti_path,
                &run_records,
                &summaries,
                bin_width,
                eps_cents,
            )?;

            let major_minor_path = out_dir.join(format!(
                "paper_e4_major_minor_vs_weight_bw{bw_token}_eps{eps_token}.svg"
            ));
            render_e4_major_minor_plot(&major_minor_path, &summaries, bin_width, eps_cents)?;

            let third_mass_path = out_dir.join(format!(
                "paper_e4_third_mass_vs_weight_bw{bw_token}_eps{eps_token}.svg"
            ));
            render_e4_third_mass_plot(&third_mass_path, &summaries, bin_width, eps_cents)?;

            let rate_path = out_dir.join(format!(
                "paper_e4_major_minor_rate_vs_weight_bw{bw_token}_eps{eps_token}.svg"
            ));
            render_e4_major_minor_rate_plot(&rate_path, &summaries, bin_width, eps_cents)?;
        }

        let heatmap_path = out_dir.join(format!("paper_e4_interval_heatmap_bw{bw_token}.svg"));
        render_e4_interval_heatmap(&heatmap_path, &hist_records, bin_width)?;
    }

    let legacy_path = out_dir.join("paper_e4_mirror_sweep.svg");
    render_e4_major_minor_plot(&legacy_path, &summaries, primary_bin, primary_eps)?;
    run_e4_step_and_hysteresis_protocol(out_dir, anchor_hz, primary_eps)?;

    if emit_kernel_gate {
        let kernel_gate_path = out_dir.join("paper_e4_kernel_gate.svg");
        render_e4_kernel_gate(&kernel_gate_path, anchor_hz)?;
    }

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

#[derive(Clone, Copy)]
struct ConsonantMassRow {
    condition: &'static str,
    seed: u64,
    mass_core: f32,
    mass_extended: f32,
}

struct E4RunRecord {
    count_mode: &'static str,
    mirror_weight: f32,
    seed: u64,
    bin_width: f32,
    eps_cents: f32,
    major_score: f32,
    minor_score: f32,
    delta: f32,
    triad_major: f32,
    triad_minor: f32,
    delta_t: f32,
    major_frac: f32,
    mass_min3: f32,
    mass_maj3: f32,
    mass_p4: f32,
    mass_p5: f32,
    mass_p5_class: f32,
    steps_total: u32,
    burn_in: u32,
    tail_window: u32,
    histogram_source: &'static str,
}

struct E4SummaryRecord {
    count_mode: &'static str,
    mirror_weight: f32,
    bin_width: f32,
    eps_cents: f32,
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
    mean_p5_class: f32,
    std_p5_class: f32,
    major_rate: f32,
    minor_rate: f32,
    ambiguous_rate: f32,
    n_runs: usize,
}

struct E4DeltaEffectRow {
    count_mode: &'static str,
    mirror_weight: f32,
    bin_width: f32,
    eps_cents: f32,
    mean_delta: f32,
    sd_delta: f32,
    se_delta: f32,
    ci_lo: f32,
    ci_hi: f32,
    n_seeds: usize,
    mean_mass_min3: f32,
    sd_mass_min3: f32,
    mean_mass_maj3: f32,
    sd_mass_maj3: f32,
    mean_mass_p5: f32,
    sd_mass_p5: f32,
    mean_mass_p5_class: f32,
    sd_mass_p5_class: f32,
}

struct E4RegressionRow {
    count_mode: &'static str,
    bin_width: f32,
    eps_cents: f32,
    slope: f32,
    slope_ci_lo: f32,
    slope_ci_hi: f32,
    r2: f32,
    spearman_rho: f32,
    spearman_p: f32,
    n_weights: usize,
}

struct E4EndpointEffectRow {
    count_mode: &'static str,
    bin_width: f32,
    eps_cents: f32,
    delta_end: f32,
    ci_lo: f32,
    ci_hi: f32,
    cohen_d: f32,
    n0: usize,
    n1: usize,
}

struct E4SeedSlopeRow {
    count_mode: &'static str,
    seed: u64,
    bin_width: f32,
    eps_cents: f32,
    slope_seed: f32,
    r2_seed: f32,
    n_weights: usize,
}

struct E4RunLevelRegressionRow {
    count_mode: &'static str,
    bin_width: f32,
    eps_cents: f32,
    slope: f32,
    slope_ci_lo: f32,
    slope_ci_hi: f32,
    r2: f32,
    n_runs: usize,
}

struct E4SeedSlopeMetaRow {
    count_mode: &'static str,
    bin_width: f32,
    eps_cents: f32,
    mean_slope: f32,
    ci_lo: f32,
    ci_hi: f32,
    sign_p: f32,
    var_slope_across_seeds: f32,
    mean_r2: f32,
    n_seeds: usize,
}

struct E4ThirdMassRow {
    count_mode: &'static str,
    mirror_weight: f32,
    bin_width: f32,
    eps_cents: f32,
    mean_third_mass: f32,
    std_third_mass: f32,
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
    count_mode: &'static str,
    mirror_weight: f32,
    seed: u64,
    bin_width: f32,
    eps_cents: f32,
    step: u32,
    mass_min3: f32,
    mass_maj3: f32,
    mass_p4: f32,
    mass_p5: f32,
    mass_p5_class: f32,
    delta_t: f32,
}

#[derive(Clone, Copy)]
struct E4TailAgentRow {
    mirror_weight: f32,
    seed: u64,
    step: u32,
    agent_id: u64,
    freq_hz: f32,
}

#[derive(Clone, Copy)]
struct E4BindMetricRow {
    mirror_weight: f32,
    seed: u64,
    root_fit: f32,
    ceiling_fit: f32,
    delta_bind: f32,
    n_agents: usize,
    step: u32,
}

#[derive(Clone, Copy)]
struct E4BindSummaryRow {
    mirror_weight: f32,
    mean_root_fit: f32,
    root_ci_lo: f32,
    root_ci_hi: f32,
    mean_ceiling_fit: f32,
    ceiling_ci_lo: f32,
    ceiling_ci_hi: f32,
    mean_delta_bind: f32,
    delta_bind_ci_lo: f32,
    delta_bind_ci_hi: f32,
    n_seeds: usize,
}

#[derive(Clone, Copy)]
struct E4FingerprintRow {
    mirror_weight: f32,
    seed: u64,
    category: &'static str,
    prob: f32,
}

#[derive(Clone, Copy)]
struct E4FingerprintSummaryRow {
    mirror_weight: f32,
    category: &'static str,
    mean_prob: f32,
    prob_ci_lo: f32,
    prob_ci_hi: f32,
    n_seeds: usize,
}

#[derive(Clone, Copy)]
struct E4WrTailAgentRow {
    wr: f32,
    mirror_weight: f32,
    seed: u64,
    step: u32,
    agent_id: u64,
    freq_hz: f32,
}

#[derive(Clone, Copy)]
struct E4WrBindRunRow {
    wr: f32,
    mirror_weight: f32,
    seed: u64,
    root_fit: f32,
    ceiling_fit: f32,
    delta_bind: f32,
    root_fit_anchor: f32,
    ceiling_fit_anchor: f32,
    delta_bind_anchor: f32,
}

#[derive(Clone, Copy)]
struct E4WrBindSummaryRow {
    wr: f32,
    mirror_weight: f32,
    root_fit_mean: f32,
    root_fit_ci_lo: f32,
    root_fit_ci_hi: f32,
    ceiling_fit_mean: f32,
    ceiling_fit_ci_lo: f32,
    ceiling_fit_ci_hi: f32,
    delta_bind_mean: f32,
    delta_bind_ci_lo: f32,
    delta_bind_ci_hi: f32,
    root_fit_anchor_mean: f32,
    root_fit_anchor_ci_lo: f32,
    root_fit_anchor_ci_hi: f32,
    ceiling_fit_anchor_mean: f32,
    ceiling_fit_anchor_ci_lo: f32,
    ceiling_fit_anchor_ci_hi: f32,
    delta_bind_anchor_mean: f32,
    delta_bind_anchor_ci_lo: f32,
    delta_bind_anchor_ci_hi: f32,
    n_seeds: usize,
}

#[derive(Clone, Copy)]
struct E4WrFingerprintRunRow {
    wr: f32,
    mirror_weight: f32,
    seed: u64,
    category: &'static str,
    prob: f32,
}

#[derive(Clone, Copy)]
struct E4WrFingerprintSummaryRow {
    wr: f32,
    mirror_weight: f32,
    category: &'static str,
    mean_prob: f32,
    prob_ci_lo: f32,
    prob_ci_hi: f32,
    n_seeds: usize,
}

#[derive(Clone)]
struct E4LandscapeScans {
    wr: f32,
    mirror_weight: f32,
    log2_ratio: Vec<f32>,
    semitones: Vec<f32>,
    h_lower: Vec<f32>,
    h_upper: Vec<f32>,
    h: Vec<f32>,
    r: Vec<f32>,
    c: Vec<f32>,
}

#[derive(Clone, Copy)]
struct E4PeakRow {
    rank: usize,
    bin_idx: usize,
    log2_ratio: f32,
    semitones: f32,
    freq_hz: f32,
    c_value: f32,
    prominence: f32,
    width_st: f32,
}

#[derive(Clone, Copy)]
struct E4WrOracleRow {
    wr: f32,
    mirror_weight: f32,
    n_seeds: usize,
    n_peaks: usize,
    agent_delta_bind: f32,
    oracle1_delta_bind: f32,
    oracle1_delta_bind_ci_lo: f32,
    oracle1_delta_bind_ci_hi: f32,
    oracle2_delta_bind_mean: f32,
    oracle2_delta_bind_ci_lo: f32,
    oracle2_delta_bind_ci_hi: f32,
}

#[derive(Clone, Copy)]
struct E4WrOracleRunRow {
    wr: f32,
    mirror_weight: f32,
    seed: u64,
    n_peaks: usize,
    agent_delta_bind: f32,
    oracle1_delta_bind: f32,
    oracle2_delta_bind: f32,
}

#[derive(Clone, Copy)]
struct E4WrDynamicsProbeRow {
    wr: f32,
    mirror_weight: f32,
    seed: u64,
    mode: &'static str,
    step: u32,
    n_agents: usize,
    mean_c01: f32,
    mean_h01: f32,
    mean_r01: f32,
    mean_repulsion: f32,
    root_fit: f32,
    ceiling_fit: f32,
    delta_bind: f32,
    pitch_diversity_st: f32,
}

#[derive(Clone, Copy)]
struct E4WrDynamicsProbeSummaryRow {
    wr: f32,
    mirror_weight: f32,
    mode: &'static str,
    step: u32,
    n_seeds: usize,
    mean_c01: f32,
    mean_h01: f32,
    mean_r01: f32,
    mean_repulsion: f32,
    root_fit_mean: f32,
    ceiling_fit_mean: f32,
    delta_bind_mean: f32,
    delta_bind_ci_lo: f32,
    delta_bind_ci_hi: f32,
    pitch_diversity_st_mean: f32,
}

#[derive(Clone, Copy)]
struct E4DiagStepRow {
    step: u32,
    mirror_weight: f32,
    seed: u64,
    agent_id: usize,
    agent_idx: usize,
    oracle_global_idx: usize,
    oracle_reachable_idx: usize,
    agent_score: f32,
    oracle_global_score: f32,
    oracle_reachable_score: f32,
    gap_global: f32,
    gap_reach: f32,
    idx_err_global: f32,
    idx_err_reach: f32,
    idx_err_global_st: f32,
    idx_err_reach_st: f32,
}

#[derive(Clone, Copy)]
struct E4DiagPeakRow {
    mirror_weight: f32,
    seed: u64,
    peak_rank: usize,
    peak_idx: usize,
    peak_log2: f32,
    peak_semitones: f32,
    peak_value: f32,
}

#[derive(Clone, Copy)]
struct E4LandscapeDeltaRow {
    mirror_weight: f32,
    cosine_similarity: f32,
    l1_distance: f32,
    topk_peak_shift_st: f32,
}

#[derive(Clone)]
struct E4AbcdTraceRow {
    run_id: String,
    wr: f32,
    mirror_weight: f32,
    seed: u64,
    timing_mode: &'static str,
    step: u32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    agent_idx: usize,
    oracle_idx: usize,
    agent_c01: f32,
    oracle_c01: f32,
    agent_log2: f32,
    oracle_log2: f32,
}

fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct E4IntervalCentsCacheKey {
    mirror_weight_bits: u32,
    seed: u64,
    tail_window: u32,
    anchor_hz_bits: u32,
}

#[derive(Clone, Copy)]
struct E4TailIntervalMasses {
    min3: f32,
    maj3: f32,
    p4: f32,
    p5: f32,
    p5_class: f32,
    delta_t: f32,
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct E4TailMassCacheKey {
    mirror_weight_bits: u32,
    seed: u64,
    tail_window: u32,
    anchor_hz_bits: u32,
    eps_cents_bits: u32,
    count_mode_soft: bool,
}

type E4IntervalCentsCache = HashMap<E4IntervalCentsCacheKey, Arc<Vec<f32>>>;
type E4TailMassCache = HashMap<E4TailMassCacheKey, Arc<Vec<E4TailIntervalMasses>>>;

static E4_INTERVAL_CENTS_CACHE: OnceLock<Mutex<E4IntervalCentsCache>> = OnceLock::new();
static E4_TAIL_MASS_CACHE: OnceLock<Mutex<E4TailMassCache>> = OnceLock::new();

fn e4_interval_cents_cache() -> &'static Mutex<E4IntervalCentsCache> {
    E4_INTERVAL_CENTS_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn e4_tail_mass_cache() -> &'static Mutex<E4TailMassCache> {
    E4_TAIL_MASS_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn e4_interval_cents_cache_key(
    mirror_weight: f32,
    seed: u64,
    anchor_hz: f32,
    tail_window: u32,
) -> E4IntervalCentsCacheKey {
    E4IntervalCentsCacheKey {
        mirror_weight_bits: mirror_weight.to_bits(),
        seed,
        tail_window,
        anchor_hz_bits: anchor_hz.to_bits(),
    }
}

fn e4_tail_mass_cache_key(
    mirror_weight: f32,
    seed: u64,
    anchor_hz: f32,
    tail_window: u32,
    eps_cents: f32,
    count_mode: E4CountMode,
) -> E4TailMassCacheKey {
    E4TailMassCacheKey {
        mirror_weight_bits: mirror_weight.to_bits(),
        seed,
        tail_window,
        anchor_hz_bits: anchor_hz.to_bits(),
        eps_cents_bits: eps_cents.to_bits(),
        count_mode_soft: matches!(count_mode, E4CountMode::Soft),
    }
}

fn collect_e4_interval_cents_samples_cached(
    samples: &E4TailSamples,
    anchor_hz: f32,
    mirror_weight: f32,
    seed: u64,
) -> Arc<Vec<f32>> {
    if !anchor_hz.is_finite() || anchor_hz <= 0.0 {
        return Arc::new(Vec::new());
    }

    let key = e4_interval_cents_cache_key(mirror_weight, seed, anchor_hz, samples.tail_window);
    {
        let cache = e4_interval_cents_cache()
            .lock()
            .expect("interval cents cache poisoned");
        if let Some(cents) = cache.get(&key) {
            return Arc::clone(cents);
        }
    }

    let mut cents = Vec::new();
    for freqs in &samples.freqs_by_step {
        for &freq in freqs {
            if let Some(cents_class) = freq_to_cents_class(anchor_hz, freq) {
                cents.push(cents_class);
            }
        }
    }

    let arc = Arc::new(cents);
    let mut cache = e4_interval_cents_cache()
        .lock()
        .expect("interval cents cache poisoned");
    if let Some(cached) = cache.get(&key) {
        return Arc::clone(cached);
    }
    cache.insert(key, Arc::clone(&arc));
    arc
}

fn collect_e4_interval_mass_series_cached(
    samples: &E4TailSamples,
    anchor_hz: f32,
    mirror_weight: f32,
    seed: u64,
    eps_cents: f32,
    count_mode: E4CountMode,
) -> Arc<Vec<E4TailIntervalMasses>> {
    let key = e4_tail_mass_cache_key(
        mirror_weight,
        seed,
        anchor_hz,
        samples.tail_window,
        eps_cents,
        count_mode,
    );
    {
        let cache = e4_tail_mass_cache()
            .lock()
            .expect("tail mass cache poisoned");
        if let Some(masses) = cache.get(&key) {
            return Arc::clone(masses);
        }
    }

    let mut out = Vec::with_capacity(samples.freqs_by_step.len());
    for freqs in &samples.freqs_by_step {
        let masses = interval_masses_from_freqs(anchor_hz, freqs, eps_cents, count_mode);
        let (_, _, delta_t, _) = triad_scores(masses);
        out.push(E4TailIntervalMasses {
            min3: masses.min3,
            maj3: masses.maj3,
            p4: masses.p4,
            p5: masses.p5,
            p5_class: masses.p5_class,
            delta_t,
        });
    }

    let arc = Arc::new(out);
    let mut cache = e4_tail_mass_cache()
        .lock()
        .expect("tail mass cache poisoned");
    if let Some(cached) = cache.get(&key) {
        return Arc::clone(cached);
    }
    cache.insert(key, Arc::clone(&arc));
    arc
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
        let rounded = (w * FLOAT_KEY_SCALE).round() / FLOAT_KEY_SCALE;
        weights.push(rounded.clamp(0.0, 1.0));
        w += step;
    }
    if (weights.last().copied().unwrap_or(0.0) - 1.0).abs() > 1e-6 {
        weights.push(1.0);
    }
    weights
}

fn refine_weights_from_sign_change(
    run_records: &[E4RunRecord],
    bin_width: f32,
    eps_cents: f32,
) -> Vec<f32> {
    let mut weight_means: Vec<(f32, f32)> = Vec::new();
    let mut map: std::collections::HashMap<i32, Vec<f32>> = std::collections::HashMap::new();
    for record in run_records {
        if record.count_mode != "soft" {
            continue;
        }
        if (record.bin_width - bin_width).abs() > 1e-6 {
            continue;
        }
        if (record.eps_cents - eps_cents).abs() > 1e-6 {
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
        weight_means.push((float_from_key(key), mean));
    }
    weight_means.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut refined = Vec::new();
    for pair in weight_means.windows(2) {
        let (w0, d0) = pair[0];
        let (w1, d1) = pair[1];
        if d0 == 0.0 || d1 == 0.0 || d0.signum() != d1.signum() {
            let mut w = w0;
            while w < w1 - 1e-6 {
                let rounded = (w * FLOAT_KEY_SCALE).round() / FLOAT_KEY_SCALE;
                refined.push(rounded);
                w += E4_WEIGHT_FINE_STEP;
            }
            refined.push(w1);
        }
    }
    refined
}

fn float_key(value: f32) -> i32 {
    (value * FLOAT_KEY_SCALE).round() as i32
}

fn float_from_key(key: i32) -> f32 {
    key as f32 / FLOAT_KEY_SCALE
}

fn format_float_token(value: f32) -> String {
    let s = format!("{value:.2}");
    s.replace('.', "p")
}

fn fmt_eps(eps_cents: f32) -> String {
    let rounded = eps_cents.round();
    if (eps_cents - rounded).abs() < 1e-3 {
        format!("{rounded:.0}c")
    } else {
        format!("{eps_cents:.1}c")
    }
}

fn fmt_eps_token(eps_cents: f32) -> String {
    fmt_eps(eps_cents).trim_end_matches('c').replace('.', "p")
}

#[derive(Clone, Copy)]
enum E4CountMode {
    Soft,
    Hard,
}

impl E4CountMode {
    fn label(self) -> &'static str {
        match self {
            E4CountMode::Soft => "soft",
            E4CountMode::Hard => "hard",
        }
    }
}

#[derive(Clone, Copy)]
enum E4DynamicsProbeMode {
    BaselineSequential,
    LambdaZeroSequential,
    PeakRestrictedSequential,
    PeakRestrictedSynchronous,
}

impl E4DynamicsProbeMode {
    const ALL: [Self; 4] = [
        Self::BaselineSequential,
        Self::LambdaZeroSequential,
        Self::PeakRestrictedSequential,
        Self::PeakRestrictedSynchronous,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::BaselineSequential => "baseline_seq",
            Self::LambdaZeroSequential => "lambda0_seq",
            Self::PeakRestrictedSequential => "peak_seq",
            Self::PeakRestrictedSynchronous => "peak_sync",
        }
    }

    fn lambda(self) -> f32 {
        match self {
            Self::LambdaZeroSequential => 0.0,
            _ => E4_DYNAMICS_BASE_LAMBDA,
        }
    }

    fn peak_restricted(self) -> bool {
        matches!(
            self,
            Self::PeakRestrictedSequential | Self::PeakRestrictedSynchronous
        )
    }

    fn synchronous(self) -> bool {
        matches!(self, Self::PeakRestrictedSynchronous)
    }
}

fn cents_class_from_ratio(ratio: f32) -> Option<f32> {
    if !ratio.is_finite() || ratio <= 0.0 {
        return None;
    }
    let mut cents = 1200.0 * ratio.log2();
    if !cents.is_finite() {
        return None;
    }
    cents = cents.rem_euclid(1200.0);
    if cents >= 1200.0 - 1e-6 {
        cents = 0.0;
    }
    Some(cents.clamp(0.0, 1200.0))
}

fn freq_to_cents_class(anchor_hz: f32, freq_hz: f32) -> Option<f32> {
    if !anchor_hz.is_finite() || anchor_hz <= 0.0 || !freq_hz.is_finite() || freq_hz <= 0.0 {
        return None;
    }
    cents_class_from_ratio(freq_hz / anchor_hz)
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

fn circular_cents_distance(value_cents: f32, target_cents: f32) -> f32 {
    let raw = (value_cents - target_cents).abs().rem_euclid(1200.0);
    raw.min(1200.0 - raw)
}

fn p5_class_distance_cents(value_cents: f32) -> f32 {
    let d_p5 = circular_cents_distance(value_cents, E4_CENTS_P5);
    let d_p4 = circular_cents_distance(value_cents, E4_CENTS_P4);
    d_p5.min(d_p4)
}

fn mass_weight(
    distance_cents: f32,
    eps_cents: f32,
    sigma_cents: f32,
    count_mode: E4CountMode,
) -> f32 {
    match count_mode {
        E4CountMode::Hard => {
            if distance_cents <= eps_cents {
                1.0
            } else {
                0.0
            }
        }
        E4CountMode::Soft => {
            let sigma = sigma_cents.max(1e-6);
            (-0.5 * (distance_cents / sigma).powi(2)).exp()
        }
    }
}

fn mass_around(
    hist: &Histogram,
    center_cents: f32,
    eps_cents: f32,
    sigma_cents: f32,
    count_mode: E4CountMode,
) -> f32 {
    let mut sum = 0.0f32;
    for (&bin_center, &mass) in hist.bin_centers.iter().zip(hist.masses.iter()) {
        let d = circular_cents_distance(bin_center, center_cents);
        sum += mass * mass_weight(d, eps_cents, sigma_cents, count_mode);
    }
    sum
}

fn mass_p5_class(
    hist: &Histogram,
    eps_cents: f32,
    sigma_cents: f32,
    count_mode: E4CountMode,
) -> f32 {
    let mut sum = 0.0f32;
    for (&bin_center, &mass) in hist.bin_centers.iter().zip(hist.masses.iter()) {
        let d = p5_class_distance_cents(bin_center);
        sum += mass * mass_weight(d, eps_cents, sigma_cents, count_mode);
    }
    sum
}

#[derive(Clone, Copy)]
struct IntervalMasses {
    min3: f32,
    maj3: f32,
    p4: f32,
    p5: f32,
    p5_class: f32,
}

fn interval_masses_from_histogram(
    hist: &Histogram,
    eps_cents: f32,
    count_mode: E4CountMode,
) -> IntervalMasses {
    let sigma_cents = (eps_cents * E4_SOFT_SIGMA_SCALE).max(1e-6);
    IntervalMasses {
        min3: mass_around(hist, E4_CENTS_MIN3, eps_cents, sigma_cents, count_mode),
        maj3: mass_around(hist, E4_CENTS_MAJ3, eps_cents, sigma_cents, count_mode),
        p4: mass_around(hist, E4_CENTS_P4, eps_cents, sigma_cents, count_mode),
        p5: mass_around(hist, E4_CENTS_P5, eps_cents, sigma_cents, count_mode),
        p5_class: mass_p5_class(hist, eps_cents, sigma_cents, count_mode),
    }
}

fn interval_masses_from_freqs(
    anchor_hz: f32,
    freqs: &[f32],
    eps_cents: f32,
    count_mode: E4CountMode,
) -> IntervalMasses {
    if !anchor_hz.is_finite() || anchor_hz <= 0.0 || freqs.is_empty() {
        return IntervalMasses {
            min3: 0.0,
            maj3: 0.0,
            p4: 0.0,
            p5: 0.0,
            p5_class: 0.0,
        };
    }
    let eps = eps_cents.max(1e-6);
    let sigma = (eps * E4_SOFT_SIGMA_SCALE).max(1e-6);
    let mut sum_min3 = 0.0f32;
    let mut sum_maj3 = 0.0f32;
    let mut sum_p4 = 0.0f32;
    let mut sum_p5 = 0.0f32;
    let mut sum_p5_class = 0.0f32;
    let mut total = 0u32;
    for &freq in freqs {
        let Some(cents_class) = freq_to_cents_class(anchor_hz, freq) else {
            continue;
        };
        total += 1;
        sum_min3 += mass_weight(
            circular_cents_distance(cents_class, E4_CENTS_MIN3),
            eps,
            sigma,
            count_mode,
        );
        sum_maj3 += mass_weight(
            circular_cents_distance(cents_class, E4_CENTS_MAJ3),
            eps,
            sigma,
            count_mode,
        );
        sum_p4 += mass_weight(
            circular_cents_distance(cents_class, E4_CENTS_P4),
            eps,
            sigma,
            count_mode,
        );
        sum_p5 += mass_weight(
            circular_cents_distance(cents_class, E4_CENTS_P5),
            eps,
            sigma,
            count_mode,
        );
        sum_p5_class += mass_weight(p5_class_distance_cents(cents_class), eps, sigma, count_mode);
    }
    if total == 0 {
        return IntervalMasses {
            min3: 0.0,
            maj3: 0.0,
            p4: 0.0,
            p5: 0.0,
            p5_class: 0.0,
        };
    }
    let inv = 1.0 / total as f32;
    IntervalMasses {
        min3: sum_min3 * inv,
        maj3: sum_maj3 * inv,
        p4: sum_p4 * inv,
        p5: sum_p5 * inv,
        p5_class: sum_p5_class * inv,
    }
}

fn triad_scores(masses: IntervalMasses) -> (f32, f32, f32, f32) {
    let triad_major = masses.maj3 * masses.p5_class;
    let triad_minor = masses.min3 * masses.p5_class;
    let denom = triad_major + triad_minor + 1e-6;
    let delta_t = (triad_major - triad_minor) / denom;
    let major_frac = triad_major / (triad_major + triad_minor + 1e-12);
    (triad_major, triad_minor, delta_t, major_frac)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn run_e4_sweep_for_weights(
    out_dir: &Path,
    anchor_hz: f32,
    weights: &[f32],
    seeds: &[u64],
    bin_width: f32,
    eps_cents_list: &[f32],
    emit_hist_files: bool,
    emit_tail_agents: bool,
) -> Result<
    (
        Vec<E4RunRecord>,
        Vec<E4HistRecord>,
        Vec<E4TailIntervalRow>,
        Vec<E4TailAgentRow>,
    ),
    Box<dyn Error>,
> {
    let mut runs = Vec::new();
    let mut hists = Vec::new();
    let mut tail_rows = Vec::new();
    let mut tail_agent_rows = Vec::new();
    for &weight in weights {
        for &seed in seeds {
            let samples = run_e4_condition_tail_samples(weight, seed, E4_TAIL_WINDOW_STEPS);
            let interval_cents_samples =
                collect_e4_interval_cents_samples_cached(&samples, anchor_hz, weight, seed);
            let bin_width_cents = (bin_width * 100.0).max(1.0);
            let histogram = histogram_from_samples(
                interval_cents_samples.as_ref(),
                0.0,
                1200.0,
                bin_width_cents,
            );

            let burn_in = samples.steps_total.saturating_sub(samples.tail_window);
            for &eps_cents in eps_cents_list {
                for count_mode in [E4CountMode::Soft, E4CountMode::Hard] {
                    let mode_label = count_mode.label();
                    let masses = interval_masses_from_histogram(&histogram, eps_cents, count_mode);
                    let (triad_major, triad_minor, delta_t, major_frac) = triad_scores(masses);

                    runs.push(E4RunRecord {
                        count_mode: mode_label,
                        mirror_weight: weight,
                        seed,
                        bin_width,
                        eps_cents,
                        major_score: triad_major,
                        minor_score: triad_minor,
                        delta: delta_t,
                        triad_major,
                        triad_minor,
                        delta_t,
                        major_frac,
                        mass_min3: masses.min3,
                        mass_maj3: masses.maj3,
                        mass_p4: masses.p4,
                        mass_p5: masses.p5,
                        mass_p5_class: masses.p5_class,
                        steps_total: samples.steps_total,
                        burn_in,
                        tail_window: samples.tail_window,
                        histogram_source: "tail_mean",
                    });

                    let tail_mass_series = collect_e4_interval_mass_series_cached(
                        &samples, anchor_hz, weight, seed, eps_cents, count_mode,
                    );
                    for (i, masses) in tail_mass_series.iter().enumerate() {
                        let step = burn_in + i as u32;
                        tail_rows.push(E4TailIntervalRow {
                            count_mode: mode_label,
                            mirror_weight: weight,
                            seed,
                            bin_width,
                            eps_cents,
                            step,
                            mass_min3: masses.min3,
                            mass_maj3: masses.maj3,
                            mass_p4: masses.p4,
                            mass_p5: masses.p5,
                            mass_p5_class: masses.p5_class,
                            delta_t: masses.delta_t,
                        });
                    }
                }
            }
            if emit_tail_agents {
                for (i, agent_rows) in samples.agent_freqs_by_step.iter().enumerate() {
                    let keep = (i % 10) == 0 || i + 1 == samples.agent_freqs_by_step.len();
                    if !keep {
                        continue;
                    }
                    let step = burn_in + i as u32;
                    for row in agent_rows {
                        if !row.freq_hz.is_finite() || row.freq_hz <= 0.0 {
                            continue;
                        }
                        tail_agent_rows.push(E4TailAgentRow {
                            mirror_weight: weight,
                            seed,
                            step,
                            agent_id: row.agent_id,
                            freq_hz: row.freq_hz,
                        });
                    }
                }
            }

            hists.push(E4HistRecord {
                mirror_weight: weight,
                bin_width,
                histogram,
            });

            if emit_hist_files {
                let w_token = format_float_token(weight);
                let bw_token = format_float_token(bin_width);
                let hist_csv_path =
                    out_dir.join(format!("e4_hist_w{w_token}_seed{seed}_bw{bw_token}.csv"));
                write_with_log(
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
                    out_dir.join(format!("e4_hist_w{w_token}_seed{seed}_bw{bw_token}.svg"));
                let caption = format!(
                    "E4 Interval Histogram (cents PC, w={weight:.2}, seed={seed}, bw={bin_width_cents:.1}c)"
                );
                render_interval_histogram(
                    &hist_png_path,
                    &caption,
                    interval_cents_samples.as_ref(),
                    0.0,
                    1200.0,
                    bin_width_cents,
                    "cents",
                )?;
            }
        }
    }
    Ok((runs, hists, tail_rows, tail_agent_rows))
}

fn e4_runs_csv(records: &[E4RunRecord]) -> String {
    let mut out = String::from(
        "count_mode,mirror_weight,seed,bin_width,eps_cents,triad_major,triad_minor,delta_t,major_frac,mass_m3,mass_M3,mass_P4,mass_P5,mass_P5class,steps_total,burn_in,tail_window,histogram_source\n",
    );
    for record in records {
        out.push_str(&format!(
            "{},{:.3},{},{:.3},{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{}\n",
            record.count_mode,
            record.mirror_weight,
            record.seed,
            record.bin_width,
            record.eps_cents,
            record.triad_major,
            record.triad_minor,
            record.delta_t,
            record.major_frac,
            record.mass_min3,
            record.mass_maj3,
            record.mass_p4,
            record.mass_p5,
            record.mass_p5_class,
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
        "count_mode,mirror_weight,seed,bin_width,eps_cents,step,mass_m3,mass_M3,mass_P4,mass_P5,mass_P5class,delta_t\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{},{:.3},{},{:.3},{:.3},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            row.count_mode,
            row.mirror_weight,
            row.seed,
            row.bin_width,
            row.eps_cents,
            row.step,
            row.mass_min3,
            row.mass_maj3,
            row.mass_p4,
            row.mass_p5,
            row.mass_p5_class,
            row.delta_t
        ));
    }
    out
}

fn e4_tail_agents_csv(rows: &[E4TailAgentRow]) -> String {
    let mut out = String::from("mirror_weight,seed,step,agent_id,freq_hz\n");
    for row in rows {
        out.push_str(&format!(
            "{:.3},{},{},{},{:.6}\n",
            row.mirror_weight, row.seed, row.step, row.agent_id, row.freq_hz
        ));
    }
    out
}

fn normalize_freq_ratios(freqs: &[f32]) -> Vec<f32> {
    let mut clean: Vec<f32> = freqs
        .iter()
        .copied()
        .filter(|f| f.is_finite() && *f > 0.0)
        .collect();
    if clean.is_empty() {
        return Vec::new();
    }
    clean.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = clean.len() / 2;
    let median = if clean.len().is_multiple_of(2) {
        0.5 * (clean[mid - 1] + clean[mid])
    } else {
        clean[mid]
    };
    if !median.is_finite() || median <= 0.0 {
        return Vec::new();
    }
    clean
        .into_iter()
        .map(|f| f / median)
        .filter(|r| r.is_finite() && *r > 0.0)
        .collect()
}

fn harmonic_index_penalty(n: u32) -> f32 {
    let n = n.max(1) as f32;
    let beta = E4_BIND_RHO.max(0.0);
    (-(beta * (n - 1.0))).exp()
}

fn root_candidate_score(ratios: &[f32], candidate_root: f32) -> f32 {
    if !candidate_root.is_finite() || candidate_root <= 0.0 {
        return 0.0;
    }
    let sigma = E4_BIND_SIGMA_CENTS.max(1e-6);
    let n_max = E4_BIND_MAX_HARMONIC.max(1) as i32;
    let mut total = 0.0f32;
    let mut count = 0usize;
    for &ratio in ratios {
        if !ratio.is_finite() || ratio <= 0.0 {
            continue;
        }
        let mut n = (ratio / candidate_root).round() as i32;
        n = n.clamp(1, n_max);
        let target = candidate_root * n as f32;
        if !target.is_finite() || target <= 0.0 {
            continue;
        }
        let err_cents = 1200.0 * (ratio / target).log2();
        let score = (-0.5 * (err_cents / sigma).powi(2)).exp() * harmonic_index_penalty(n as u32);
        total += score;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

fn ceiling_candidate_score(ratios: &[f32], candidate_ceiling: f32) -> f32 {
    if !candidate_ceiling.is_finite() || candidate_ceiling <= 0.0 {
        return 0.0;
    }
    let sigma = E4_BIND_SIGMA_CENTS.max(1e-6);
    let n_max = E4_BIND_MAX_HARMONIC.max(1) as i32;
    let mut total = 0.0f32;
    let mut count = 0usize;
    for &ratio in ratios {
        if !ratio.is_finite() || ratio <= 0.0 {
            continue;
        }
        let mut n = (candidate_ceiling / ratio).round() as i32;
        n = n.clamp(1, n_max);
        let target = candidate_ceiling / n as f32;
        if !target.is_finite() || target <= 0.0 {
            continue;
        }
        let err_cents = 1200.0 * (ratio / target).log2();
        let score = (-0.5 * (err_cents / sigma).powi(2)).exp() * harmonic_index_penalty(n as u32);
        total += score;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

fn unique_candidates(mut raw: Vec<f32>) -> Vec<f32> {
    raw.retain(|f| f.is_finite() && *f > 0.0);
    for f in &mut raw {
        *f = (*f * 10_000.0).round() / 10_000.0;
    }
    raw.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    raw.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    raw
}

fn root_fit_from_freqs(freqs: &[f32]) -> f32 {
    let ratios = normalize_freq_ratios(freqs);
    if ratios.is_empty() {
        return 0.0;
    }
    let mut candidates = Vec::with_capacity(ratios.len() * E4_BIND_MAX_HARMONIC as usize);
    for &ratio in &ratios {
        for n in 1..=E4_BIND_MAX_HARMONIC {
            candidates.push(ratio / n as f32);
        }
    }
    let mut candidates = unique_candidates(candidates);
    candidates.retain(|c| *c <= 1.0 + 1e-4);
    if candidates.is_empty() {
        return 0.0;
    }
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let top_k = E4_BIND_TOP_CANDIDATES.max(1).min(candidates.len());
    let mut best = 0.0f32;
    for &candidate in candidates.iter().take(top_k) {
        best = best.max(root_candidate_score(&ratios, candidate));
    }
    best
}

fn ceiling_fit_from_freqs(freqs: &[f32]) -> f32 {
    let ratios = normalize_freq_ratios(freqs);
    if ratios.is_empty() {
        return 0.0;
    }
    let mut candidates = Vec::with_capacity(ratios.len() * E4_BIND_MAX_HARMONIC as usize);
    for &ratio in &ratios {
        for n in 1..=E4_BIND_MAX_HARMONIC {
            candidates.push(ratio * n as f32);
        }
    }
    let mut candidates = unique_candidates(candidates);
    candidates.retain(|c| *c >= 1.0 - 1e-4);
    if candidates.is_empty() {
        return 0.0;
    }
    candidates.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let top_k = E4_BIND_TOP_CANDIDATES.max(1).min(candidates.len());
    let mut best = 0.0f32;
    for &candidate in candidates.iter().take(top_k) {
        best = best.max(ceiling_candidate_score(&ratios, candidate));
    }
    best
}

#[derive(Clone, Copy, Debug)]
struct BindEval {
    root_fit: f32,
    ceiling_fit: f32,
    delta_bind: f32,
}

fn bind_eval_from_freqs(freqs: &[f32]) -> BindEval {
    let clean_freqs: Vec<f32> = freqs
        .iter()
        .copied()
        .filter(|f| f.is_finite() && *f > 0.0)
        .collect();
    if clean_freqs.is_empty() {
        return BindEval {
            root_fit: 0.0,
            ceiling_fit: 0.0,
            delta_bind: 0.0,
        };
    }
    // Voices only (anchor excluded): this keeps the regime metric tied to population structure.
    let root_fit = root_fit_from_freqs(&clean_freqs);
    let ceiling_fit = ceiling_fit_from_freqs(&clean_freqs);
    let denom = root_fit + ceiling_fit + 1e-6;
    let delta_bind = ((root_fit - ceiling_fit) / denom).clamp(-1.0, 1.0);
    BindEval {
        root_fit,
        ceiling_fit,
        delta_bind,
    }
}

fn bind_scores_from_freqs(freqs: &[f32]) -> (f32, f32, f32) {
    let eval = bind_eval_from_freqs(freqs);
    (eval.root_fit, eval.ceiling_fit, eval.delta_bind)
}

fn anchored_fit_from_ratios(ratios: &[f32]) -> f32 {
    let sigma = E4_BIND_SIGMA_CENTS.max(1e-6);
    let n_max = E4_BIND_MAX_HARMONIC.max(1) as i32;
    let mut total = 0.0f32;
    let mut count = 0usize;
    for &ratio in ratios {
        if !ratio.is_finite() || ratio <= 0.0 {
            continue;
        }
        let mut n = ratio.round() as i32;
        n = n.clamp(1, n_max);
        let target = n as f32;
        let err_cents = 1200.0 * (ratio / target).log2();
        let score = (-0.5 * (err_cents / sigma).powi(2)).exp() * harmonic_index_penalty(n as u32);
        total += score;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

fn bind_scores_anchor_from_freqs(anchor_hz: f32, freqs: &[f32]) -> (f32, f32, f32) {
    if !anchor_hz.is_finite() || anchor_hz <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let ratios: Vec<f32> = freqs
        .iter()
        .copied()
        .filter(|f| f.is_finite() && *f > 0.0)
        .map(|f| f / anchor_hz)
        .filter(|r| r.is_finite() && *r > 0.0)
        .collect();
    if ratios.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let root_fit_anchor = anchored_fit_from_ratios(&ratios);
    let inv_ratios: Vec<f32> = ratios
        .iter()
        .copied()
        .filter(|r| *r > 0.0)
        .map(|r| 1.0 / r)
        .collect();
    let ceiling_fit_anchor = anchored_fit_from_ratios(&inv_ratios);
    let denom = root_fit_anchor + ceiling_fit_anchor + 1e-6;
    let delta_bind_anchor = (root_fit_anchor - ceiling_fit_anchor) / denom;
    (
        root_fit_anchor,
        ceiling_fit_anchor,
        delta_bind_anchor.clamp(-1.0, 1.0),
    )
}

fn sample_weighted_index(weights: &[f32], rng: &mut StdRng) -> Option<usize> {
    if weights.is_empty() {
        return None;
    }
    let mut total = 0.0f32;
    for &w in weights {
        if w.is_finite() && w > 0.0 {
            total += w;
        }
    }
    if total <= 0.0 {
        return Some(rng.random_range(0..weights.len()));
    }
    let mut x = rng.random_range(0.0..total);
    for (i, &w) in weights.iter().enumerate() {
        let ww = if w.is_finite() && w > 0.0 { w } else { 0.0 };
        if x <= ww {
            return Some(i);
        }
        x -= ww;
    }
    Some(weights.len().saturating_sub(1))
}

fn oracle1_delta_bind_from_peaks(peaks: &[E4PeakRow], k: usize) -> f32 {
    let k = k.max(1);
    let freqs: Vec<f32> = peaks.iter().take(k).map(|p| p.freq_hz).collect();
    bind_eval_from_freqs(&freqs).delta_bind
}

fn oracle2_delta_bind_from_peaks(
    peaks: &[E4PeakRow],
    top_n: usize,
    k: usize,
    trials: usize,
    seed: u64,
) -> (f32, f32, f32) {
    if peaks.is_empty() || trials == 0 {
        return (0.0, 0.0, 0.0);
    }
    let top_n = top_n.max(1).min(peaks.len());
    let k = k.max(1);
    let pool = &peaks[..top_n];
    let weights: Vec<f32> = pool.iter().map(|p| p.c_value.max(0.0)).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut deltas = Vec::with_capacity(trials);
    for _ in 0..trials {
        let mut freqs = Vec::with_capacity(k);
        for _ in 0..k {
            let idx = sample_weighted_index(&weights, &mut rng).unwrap_or(0);
            freqs.push(pool[idx].freq_hz);
        }
        deltas.push(bind_eval_from_freqs(&freqs).delta_bind);
    }
    bootstrap_mean_ci95(&deltas, E4_BOOTSTRAP_ITERS, seed ^ 0xB007_5EED)
}

fn e4_wr_oracle_run_rows(
    bind_runs: &[E4WrBindRunRow],
    grouped_freqs: &std::collections::HashMap<(i32, i32, u64), Vec<f32>>,
    space: &Log2Space,
    anchor_hz: f32,
    du_scan: &[f32],
    population_size: usize,
) -> Vec<E4WrOracleRunRow> {
    let mut out = Vec::new();
    for run in bind_runs {
        let key = (float_key(run.wr), float_key(run.mirror_weight), run.seed);
        let Some(freqs) = grouped_freqs.get(&key) else {
            continue;
        };
        let scan =
            compute_e4_landscape_scans(space, anchor_hz, run.wr, run.mirror_weight, freqs, du_scan);
        let peaks = extract_peak_rows_from_c_scan(space, anchor_hz, &scan, E4_ORACLE_TOP_N.max(1));
        let oracle1 = oracle1_delta_bind_from_peaks(&peaks, population_size);
        let oracle_seed = E4_WR_BASE_SEED
            ^ (key.0 as i64 as u64).wrapping_mul(0x9E37_79B9)
            ^ (key.1 as i64 as u64).wrapping_mul(0x85EB_CA6B)
            ^ key.2.wrapping_mul(0xC2B2_AE35);
        let (oracle2_mean, _oracle2_lo, _oracle2_hi) = oracle2_delta_bind_from_peaks(
            &peaks,
            E4_ORACLE_TOP_N,
            population_size,
            E4_ORACLE_SAMPLE_TRIALS,
            oracle_seed,
        );
        out.push(E4WrOracleRunRow {
            wr: run.wr,
            mirror_weight: run.mirror_weight,
            seed: run.seed,
            n_peaks: peaks.len(),
            agent_delta_bind: run.delta_bind,
            oracle1_delta_bind: oracle1,
            oracle2_delta_bind: oracle2_mean,
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
    });
    out
}

fn e4_wr_oracle_runs_csv(rows: &[E4WrOracleRunRow]) -> String {
    let mut out = String::from(
        "wr,mirror_weight,seed,n_peaks,delta_bind_agent,delta_bind_oracle1,delta_bind_oracle2\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{:.3},{:.3},{},{},{:.6},{:.6},{:.6}\n",
            row.wr,
            row.mirror_weight,
            row.seed,
            row.n_peaks,
            row.agent_delta_bind,
            row.oracle1_delta_bind,
            row.oracle2_delta_bind
        ));
    }
    out
}

fn e4_wr_oracle_rows_from_runs(rows: &[E4WrOracleRunRow]) -> Vec<E4WrOracleRow> {
    let mut grouped: std::collections::HashMap<(i32, i32), Vec<&E4WrOracleRunRow>> =
        std::collections::HashMap::new();
    for row in rows {
        grouped
            .entry((float_key(row.wr), float_key(row.mirror_weight)))
            .or_default()
            .push(row);
    }
    let mut out = Vec::with_capacity(grouped.len());
    for ((wr_key, mirror_key), group) in grouped {
        let agent_vals: Vec<f32> = group.iter().map(|r| r.agent_delta_bind).collect();
        let oracle1_vals: Vec<f32> = group.iter().map(|r| r.oracle1_delta_bind).collect();
        let oracle2_vals: Vec<f32> = group.iter().map(|r| r.oracle2_delta_bind).collect();
        let mean_peaks = group.iter().map(|r| r.n_peaks as f32).sum::<f32>() / group.len() as f32;
        let seed = E4_BOOTSTRAP_SEED
            ^ 0x0A_C1E_u64
            ^ (wr_key as i64 as u64).wrapping_mul(0x9E37_79B9)
            ^ (mirror_key as i64 as u64).wrapping_mul(0x85EB_CA6B);
        let (agent_mean, _agent_lo, _agent_hi) =
            bootstrap_mean_ci95(&agent_vals, E4_BOOTSTRAP_ITERS, seed ^ 0x11);
        let (oracle1_mean, oracle1_lo, oracle1_hi) =
            bootstrap_mean_ci95(&oracle1_vals, E4_BOOTSTRAP_ITERS, seed ^ 0x22);
        let (oracle2_mean, oracle2_lo, oracle2_hi) =
            bootstrap_mean_ci95(&oracle2_vals, E4_BOOTSTRAP_ITERS, seed ^ 0x33);
        out.push(E4WrOracleRow {
            wr: float_from_key(wr_key),
            mirror_weight: float_from_key(mirror_key),
            n_seeds: group.len(),
            n_peaks: mean_peaks.round().max(0.0) as usize,
            agent_delta_bind: agent_mean.clamp(-1.0, 1.0),
            oracle1_delta_bind: oracle1_mean.clamp(-1.0, 1.0),
            oracle1_delta_bind_ci_lo: oracle1_lo.clamp(-1.0, 1.0),
            oracle1_delta_bind_ci_hi: oracle1_hi.clamp(-1.0, 1.0),
            oracle2_delta_bind_mean: oracle2_mean.clamp(-1.0, 1.0),
            oracle2_delta_bind_ci_lo: oracle2_lo.clamp(-1.0, 1.0),
            oracle2_delta_bind_ci_hi: oracle2_hi.clamp(-1.0, 1.0),
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
    });
    out
}

fn e4_wr_oracle_csv(rows: &[E4WrOracleRow]) -> String {
    let mut out = String::from(
        "wr,mirror_weight,n_seeds,n_peaks,delta_bind_agent,delta_bind_oracle1,delta_bind_oracle1_ci_lo,delta_bind_oracle1_ci_hi,delta_bind_oracle2_mean,delta_bind_oracle2_ci_lo,delta_bind_oracle2_ci_hi\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{:.3},{:.3},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            row.wr,
            row.mirror_weight,
            row.n_seeds,
            row.n_peaks,
            row.agent_delta_bind,
            row.oracle1_delta_bind,
            row.oracle1_delta_bind_ci_lo,
            row.oracle1_delta_bind_ci_hi,
            row.oracle2_delta_bind_mean,
            row.oracle2_delta_bind_ci_lo,
            row.oracle2_delta_bind_ci_hi
        ));
    }
    out
}

fn e4_bind_metrics_from_tail_agents(rows: &[E4TailAgentRow]) -> Vec<E4BindMetricRow> {
    let mut latest_step: std::collections::HashMap<(i32, u64), u32> =
        std::collections::HashMap::new();
    for row in rows {
        let key = (float_key(row.mirror_weight), row.seed);
        latest_step
            .entry(key)
            .and_modify(|step| *step = (*step).max(row.step))
            .or_insert(row.step);
    }

    let mut final_freqs: std::collections::HashMap<(i32, u64), Vec<(u64, f32)>> =
        std::collections::HashMap::new();
    for row in rows {
        let key = (float_key(row.mirror_weight), row.seed);
        if latest_step.get(&key).copied() != Some(row.step) {
            continue;
        }
        final_freqs
            .entry(key)
            .or_default()
            .push((row.agent_id, row.freq_hz));
    }

    let mut metrics = Vec::new();
    for ((weight_key, seed), mut agent_rows) in final_freqs {
        agent_rows.sort_by_key(|(agent_id, _)| *agent_id);
        let freqs: Vec<f32> = agent_rows.iter().map(|(_, freq)| *freq).collect();
        let eval = bind_eval_from_freqs(&freqs);
        let step = latest_step.get(&(weight_key, seed)).copied().unwrap_or(0);
        metrics.push(E4BindMetricRow {
            mirror_weight: float_from_key(weight_key),
            seed,
            root_fit: eval.root_fit,
            ceiling_fit: eval.ceiling_fit,
            delta_bind: eval.delta_bind,
            n_agents: freqs.len(),
            step,
        });
    }
    metrics.sort_by(|a, b| {
        a.mirror_weight
            .partial_cmp(&b.mirror_weight)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.seed.cmp(&b.seed))
    });
    metrics
}

fn e4_bind_metrics_csv(rows: &[E4BindMetricRow]) -> String {
    let mut out =
        String::from("mirror_weight,seed,step,n_agents,root_fit,ceiling_fit,delta_bind\n");
    for row in rows {
        out.push_str(&format!(
            "{:.3},{},{},{},{:.6},{:.6},{:.6}\n",
            row.mirror_weight,
            row.seed,
            row.step,
            row.n_agents,
            row.root_fit,
            row.ceiling_fit,
            row.delta_bind
        ));
    }
    out
}

fn e4_bind_summary_rows(rows: &[E4BindMetricRow]) -> Vec<E4BindSummaryRow> {
    let mut by_weight: std::collections::HashMap<i32, Vec<&E4BindMetricRow>> =
        std::collections::HashMap::new();
    for row in rows {
        by_weight
            .entry(float_key(row.mirror_weight))
            .or_default()
            .push(row);
    }
    let mut out = Vec::new();
    for (weight_key, group) in by_weight {
        let root_vals: Vec<f32> = group.iter().map(|r| r.root_fit).collect();
        let ceiling_vals: Vec<f32> = group.iter().map(|r| r.ceiling_fit).collect();
        let delta_vals: Vec<f32> = group.iter().map(|r| r.delta_bind).collect();
        let seed =
            E4_BOOTSTRAP_SEED ^ 0xE4B1D_u64 ^ (weight_key as i64 as u64).wrapping_mul(0x9E37_79B9);
        let (mean_root_fit, root_ci_lo, root_ci_hi) =
            bootstrap_mean_ci95(&root_vals, E4_BOOTSTRAP_ITERS, seed ^ 0x11);
        let (mean_ceiling_fit, ceiling_ci_lo, ceiling_ci_hi) =
            bootstrap_mean_ci95(&ceiling_vals, E4_BOOTSTRAP_ITERS, seed ^ 0x22);
        let (mean_delta_bind, delta_bind_ci_lo, delta_bind_ci_hi) =
            bootstrap_mean_ci95(&delta_vals, E4_BOOTSTRAP_ITERS, seed ^ 0x33);
        out.push(E4BindSummaryRow {
            mirror_weight: float_from_key(weight_key),
            mean_root_fit,
            root_ci_lo,
            root_ci_hi,
            mean_ceiling_fit,
            ceiling_ci_lo,
            ceiling_ci_hi,
            mean_delta_bind,
            delta_bind_ci_lo,
            delta_bind_ci_hi,
            n_seeds: group.len(),
        });
    }
    out.sort_by(|a, b| {
        a.mirror_weight
            .partial_cmp(&b.mirror_weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

fn e4_bind_summary_csv(rows: &[E4BindSummaryRow]) -> String {
    let mut out = String::from(
        "mirror_weight,mean_root_fit,root_ci_lo,root_ci_hi,mean_ceiling_fit,ceiling_ci_lo,ceiling_ci_hi,mean_delta_bind,delta_bind_ci_lo,delta_bind_ci_hi,n_seeds\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
            row.mirror_weight,
            row.mean_root_fit,
            row.root_ci_lo,
            row.root_ci_hi,
            row.mean_ceiling_fit,
            row.ceiling_ci_lo,
            row.ceiling_ci_hi,
            row.mean_delta_bind,
            row.delta_bind_ci_lo,
            row.delta_bind_ci_hi,
            row.n_seeds
        ));
    }
    out
}

fn e4_binding_metrics_raw_csv(rows: &[E4BindMetricRow]) -> String {
    let mut out = String::from("mirror_weight,seed,root_fit,ceiling_fit,delta_bind\n");
    for row in rows {
        out.push_str(&format!(
            "{:.3},{},{:.6},{:.6},{:.6}\n",
            row.mirror_weight, row.seed, row.root_fit, row.ceiling_fit, row.delta_bind
        ));
    }
    out
}

fn e4_binding_metrics_summary_csv(rows: &[E4BindSummaryRow]) -> String {
    let mut out = String::from(
        "mirror_weight,root_fit_mean,root_fit_ci_lo,root_fit_ci_hi,ceiling_fit_mean,ceiling_fit_ci_lo,ceiling_fit_ci_hi,delta_bind_mean,delta_bind_ci_lo,delta_bind_ci_hi,n_seeds\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
            row.mirror_weight,
            row.mean_root_fit,
            row.root_ci_lo,
            row.root_ci_hi,
            row.mean_ceiling_fit,
            row.ceiling_ci_lo,
            row.ceiling_ci_hi,
            row.mean_delta_bind,
            row.delta_bind_ci_lo,
            row.delta_bind_ci_hi,
            row.n_seeds
        ));
    }
    out
}

const E4_FINGERPRINT_LABELS: [&str; 13] = [
    "b2", "M2", "m3", "M3", "P4", "TT", "P5", "m6", "M6", "m7", "M7", "8ve", "other",
];

fn e4_fingerprint_label_index(label: &str) -> usize {
    E4_FINGERPRINT_LABELS
        .iter()
        .position(|&v| v == label)
        .unwrap_or(E4_FINGERPRINT_LABELS.len().saturating_sub(1))
}

fn ji_cents(num: u32, den: u32) -> f32 {
    1200.0 * ((num as f32) / (den as f32)).log2()
}

fn e4_fingerprint_category_distance(cents_mod: f32, category_idx: usize) -> f32 {
    match category_idx {
        0 => circular_cents_distance(cents_mod, ji_cents(16, 15)), // 16/15
        1 => circular_cents_distance(cents_mod, ji_cents(9, 8)),   // 9/8
        2 => circular_cents_distance(cents_mod, ji_cents(6, 5)),   // 6/5
        3 => circular_cents_distance(cents_mod, ji_cents(5, 4)),   // 5/4
        4 => circular_cents_distance(cents_mod, ji_cents(4, 3)),   // 4/3
        5 => {
            let d0 = circular_cents_distance(cents_mod, ji_cents(7, 5)); // 7/5
            let d1 = circular_cents_distance(cents_mod, ji_cents(45, 32)); // 45/32
            d0.min(d1)
        }
        6 => circular_cents_distance(cents_mod, ji_cents(3, 2)), // 3/2
        7 => circular_cents_distance(cents_mod, ji_cents(8, 5)), // 8/5
        8 => circular_cents_distance(cents_mod, ji_cents(5, 3)), // 5/3
        9 => circular_cents_distance(cents_mod, ji_cents(9, 5)), // 9/5
        10 => circular_cents_distance(cents_mod, ji_cents(15, 8)), // 15/8
        11 => circular_cents_distance(cents_mod, 0.0),           // octave class (0/1200)
        _ => f32::INFINITY,
    }
}

fn e4_interval_fingerprint_probs(freqs: &[f32], tol_cents: f32) -> [f32; 13] {
    let mut probs = [0.0f32; 13];
    let clean: Vec<f32> = freqs
        .iter()
        .copied()
        .filter(|f| f.is_finite() && *f > 0.0)
        .collect();
    if clean.len() < 2 {
        probs[12] = 1.0;
        return probs;
    }

    let tol = tol_cents.max(0.0);
    let mut total = 0usize;
    for i in 0..clean.len() {
        for j in (i + 1)..clean.len() {
            let cents = (1200.0 * (clean[i] / clean[j]).log2())
                .abs()
                .rem_euclid(1200.0);
            if !cents.is_finite() {
                continue;
            }
            let mut best_idx = 12usize;
            let mut best_dist = f32::INFINITY;
            for idx in 0..12usize {
                let d = e4_fingerprint_category_distance(cents, idx);
                if d < best_dist {
                    best_dist = d;
                    best_idx = idx;
                }
            }
            if best_dist <= tol {
                probs[best_idx] += 1.0;
            } else {
                probs[12] += 1.0;
            }
            total += 1;
        }
    }

    if total == 0 {
        probs[12] = 1.0;
        return probs;
    }
    let inv = 1.0 / total as f32;
    for p in &mut probs {
        *p *= inv;
    }
    probs
}

fn e4_fingerprint_rows_from_tail_agents(rows: &[E4TailAgentRow]) -> Vec<E4FingerprintRow> {
    let mut latest_step: std::collections::HashMap<(i32, u64), u32> =
        std::collections::HashMap::new();
    for row in rows {
        let key = (float_key(row.mirror_weight), row.seed);
        latest_step
            .entry(key)
            .and_modify(|step| *step = (*step).max(row.step))
            .or_insert(row.step);
    }
    let mut final_freqs: std::collections::HashMap<(i32, u64), Vec<(u64, f32)>> =
        std::collections::HashMap::new();
    for row in rows {
        let key = (float_key(row.mirror_weight), row.seed);
        if latest_step.get(&key).copied() != Some(row.step) {
            continue;
        }
        final_freqs
            .entry(key)
            .or_default()
            .push((row.agent_id, row.freq_hz));
    }

    let mut out = Vec::new();
    for ((weight_key, seed), mut agent_rows) in final_freqs {
        agent_rows.sort_by_key(|(agent_id, _)| *agent_id);
        let freqs: Vec<f32> = agent_rows.iter().map(|(_, freq)| *freq).collect();
        let probs = e4_interval_fingerprint_probs(&freqs, E4_FINGERPRINT_TOL_CENTS);
        let mirror_weight = float_from_key(weight_key);
        for (idx, category) in E4_FINGERPRINT_LABELS.iter().enumerate() {
            out.push(E4FingerprintRow {
                mirror_weight,
                seed,
                category,
                prob: probs[idx],
            });
        }
    }
    out.sort_by(|a, b| {
        a.mirror_weight
            .partial_cmp(&b.mirror_weight)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.seed.cmp(&b.seed))
            .then_with(|| {
                e4_fingerprint_label_index(a.category).cmp(&e4_fingerprint_label_index(b.category))
            })
    });
    out
}

fn e4_fingerprint_raw_csv(rows: &[E4FingerprintRow]) -> String {
    let mut out = String::from("mirror_weight,seed,category,prob\n");
    for row in rows {
        out.push_str(&format!(
            "{:.3},{},{},{}\n",
            row.mirror_weight, row.seed, row.category, row.prob
        ));
    }
    out
}

fn e4_fingerprint_summary_rows(rows: &[E4FingerprintRow]) -> Vec<E4FingerprintSummaryRow> {
    let mut grouped: std::collections::HashMap<(i32, &'static str), Vec<f32>> =
        std::collections::HashMap::new();
    for row in rows {
        grouped
            .entry((float_key(row.mirror_weight), row.category))
            .or_default()
            .push(row.prob);
    }

    let mut out = Vec::new();
    for ((weight_key, category), values) in grouped {
        let seed = E4_BOOTSTRAP_SEED
            ^ (weight_key as i64 as u64).wrapping_mul(0x9E37_79B9)
            ^ (e4_fingerprint_label_index(category) as u64).wrapping_mul(0x85EB_CA6B);
        let (mean_prob, prob_ci_lo, prob_ci_hi) =
            bootstrap_mean_ci95(&values, E4_BOOTSTRAP_ITERS, seed);
        out.push(E4FingerprintSummaryRow {
            mirror_weight: float_from_key(weight_key),
            category,
            mean_prob: mean_prob.clamp(0.0, 1.0),
            prob_ci_lo: prob_ci_lo.clamp(0.0, 1.0),
            prob_ci_hi: prob_ci_hi.clamp(0.0, 1.0),
            n_seeds: values.len(),
        });
    }
    out.sort_by(|a, b| {
        a.mirror_weight
            .partial_cmp(&b.mirror_weight)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                e4_fingerprint_label_index(a.category).cmp(&e4_fingerprint_label_index(b.category))
            })
    });
    out
}

fn e4_fingerprint_summary_csv(rows: &[E4FingerprintSummaryRow]) -> String {
    let mut out = String::from("mirror_weight,category,prob_mean,prob_ci_lo,prob_ci_hi,n_seeds\n");
    for row in rows {
        out.push_str(&format!(
            "{:.3},{},{:.6},{:.6},{:.6},{}\n",
            row.mirror_weight,
            row.category,
            row.mean_prob,
            row.prob_ci_lo,
            row.prob_ci_hi,
            row.n_seeds
        ));
    }
    out
}

fn normalize_wr(wr: f32) -> f32 {
    if wr.is_finite() {
        wr.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn build_consonance_workspace_with_wr(space: &Log2Space, wr: f32) -> ConsonanceWorkspace {
    let wr = normalize_wr(wr);
    let mut ws = build_consonance_workspace(space);
    ws.params.consonance_roughness_weight_floor *= wr;
    ws.params.consonance_roughness_weight *= wr;
    ws
}

fn scan_to_state01(scan: &[f32]) -> Vec<f32> {
    let max_val = scan.iter().copied().fold(0.0f32, f32::max).max(1e-12);
    scan.iter()
        .map(|v| (*v / max_val).clamp(0.0, 1.0))
        .collect()
}

fn build_env_density_from_freqs(
    space: &Log2Space,
    freqs_hz: &[f32],
    du_scan: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut env_scan = vec![0.0f32; space.n_bins()];
    let mut density_scan = vec![0.0f32; space.n_bins()];
    for &freq in freqs_hz {
        if !freq.is_finite() || freq <= 0.0 {
            continue;
        }
        if let Some(idx) = space.index_of_freq(freq) {
            env_scan[idx] += 1.0;
            density_scan[idx] += 1.0 / du_scan[idx].max(1e-12);
        }
    }
    (env_scan, density_scan)
}

fn compute_e4_landscape_scans(
    space: &Log2Space,
    anchor_hz: f32,
    wr: f32,
    mirror_weight: f32,
    freqs_hz: &[f32],
    du_scan: &[f32],
) -> E4LandscapeScans {
    let mut workspace = build_consonance_workspace_with_wr(space, wr);
    workspace.params.harmonicity_kernel.params.mirror_weight = mirror_weight.clamp(0.0, 1.0);
    let (env_scan, density_scan) = build_env_density_from_freqs(space, freqs_hz, du_scan);
    let (density_norm, _) =
        psycho_state::normalize_density(&density_scan, du_scan, workspace.params.roughness_ref_eps);

    let mut h_params_lower = workspace.params.harmonicity_kernel.params;
    h_params_lower.mirror_weight = 0.0;
    let hk_lower = HarmonicityKernel::new(space, h_params_lower);
    let (h_lower_pot, _) = hk_lower.potential_h_from_log2_spectrum(&env_scan, space);
    let h_lower = scan_to_state01(&h_lower_pot);

    let mut h_params_upper = workspace.params.harmonicity_kernel.params;
    h_params_upper.mirror_weight = 1.0;
    let hk_upper = HarmonicityKernel::new(space, h_params_upper);
    let (h_upper_pot, _) = hk_upper.potential_h_from_log2_spectrum(&env_scan, space);
    let h_upper = scan_to_state01(&h_upper_pot);

    let (h_pot, _) = workspace
        .params
        .harmonicity_kernel
        .potential_h_from_log2_spectrum(&env_scan, space);
    let h = scan_to_state01(&h_pot);

    let (r_pot, _) = workspace
        .params
        .roughness_kernel
        .potential_r_from_log2_spectrum_density(&density_norm, space);
    let mut r = vec![0.0f32; space.n_bins()];
    psycho_state::r_pot_scan_to_r_state01_scan(
        &r_pot,
        workspace.r_ref_peak,
        workspace.params.roughness_k,
        &mut r,
    );

    let alpha_h = workspace.params.consonance_harmonicity_weight;
    let w0 = workspace.params.consonance_roughness_weight_floor;
    let w1 = workspace.params.consonance_roughness_weight;
    let mut c = vec![0.0f32; space.n_bins()];
    for i in 0..space.n_bins() {
        let score = psycho_state::compose_c_score(alpha_h, w0, w1, h[i], r[i]);
        c[i] = psycho_state::compose_c_state(
            workspace.params.c_state_beta,
            workspace.params.c_state_theta,
            score,
        );
    }

    let anchor_log2 = anchor_hz.max(1.0).log2();
    let log2_ratio: Vec<f32> = space
        .centers_log2
        .iter()
        .map(|l| *l - anchor_log2)
        .collect();
    let semitones: Vec<f32> = log2_ratio.iter().map(|v| *v * 12.0).collect();

    E4LandscapeScans {
        wr: normalize_wr(wr),
        mirror_weight: mirror_weight.clamp(0.0, 1.0),
        log2_ratio,
        semitones,
        h_lower,
        h_upper,
        h,
        r,
        c,
    }
}

fn e4_landscape_components_csv(scan: &E4LandscapeScans) -> String {
    let mut out =
        String::from("wr,mirror_weight,interval_unit,log2_ratio,semitones,H_lower,H_upper,H,R,C\n");
    for i in 0..scan.log2_ratio.len() {
        out.push_str(&format!(
            "{:.3},{:.3},cents,{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            scan.wr,
            scan.mirror_weight,
            scan.log2_ratio[i],
            scan.semitones[i],
            scan.h_lower[i],
            scan.h_upper[i],
            scan.h[i],
            scan.r[i],
            scan.c[i]
        ));
    }
    out
}

fn extract_peak_rows_from_c_scan(
    space: &Log2Space,
    anchor_hz: f32,
    scan: &E4LandscapeScans,
    top_n: usize,
) -> Vec<E4PeakRow> {
    let n = scan.c.len();
    if n < 3 {
        return Vec::new();
    }
    let mut peaks = Vec::new();
    for i in 1..(n - 1) {
        let c0 = scan.c[i];
        if !c0.is_finite() || c0 <= 0.0 {
            continue;
        }
        if !(scan.c[i - 1] < c0 && c0 >= scan.c[i + 1]) {
            continue;
        }

        let mut left_min = c0;
        let mut j = i;
        while j > 0 {
            j -= 1;
            let v = scan.c[j];
            if v > c0 {
                break;
            }
            left_min = left_min.min(v);
            if j == 0 {
                break;
            }
        }
        let mut right_min = c0;
        let mut j = i;
        while j + 1 < n {
            j += 1;
            let v = scan.c[j];
            if v > c0 {
                break;
            }
            right_min = right_min.min(v);
            if j + 1 >= n {
                break;
            }
        }
        let saddle = left_min.max(right_min);
        let prominence = (c0 - saddle).max(0.0);

        let width_level = c0 - 0.5 * prominence;
        let mut left = i;
        while left > 0 && scan.c[left] >= width_level {
            left -= 1;
        }
        let mut right = i;
        while right + 1 < n && scan.c[right] >= width_level {
            right += 1;
        }
        let width_st = (scan.semitones[right] - scan.semitones[left])
            .abs()
            .max(0.0);
        let freq_hz = anchor_hz * 2.0f32.powf(scan.log2_ratio[i]);
        peaks.push(E4PeakRow {
            rank: 0,
            bin_idx: i,
            log2_ratio: scan.log2_ratio[i],
            semitones: scan.semitones[i],
            freq_hz,
            c_value: c0,
            prominence,
            width_st,
        });
    }

    peaks.sort_by(|a, b| {
        b.c_value
            .partial_cmp(&a.c_value)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.prominence
                    .partial_cmp(&a.prominence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    peaks.truncate(top_n.max(1));
    for (rank, row) in peaks.iter_mut().enumerate() {
        row.rank = rank + 1;
    }
    let _ = space;
    peaks
}

fn e4_peaklist_csv(wr: f32, mirror_weight: f32, peaks: &[E4PeakRow]) -> String {
    let mut out = String::from(
        "wr,mirror_weight,rank,bin_idx,log2_ratio,semitones,freq_hz,c_value,peak_prominence,near_width_st\n",
    );
    for row in peaks {
        out.push_str(&format!(
            "{:.3},{:.3},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            wr,
            mirror_weight,
            row.rank,
            row.bin_idx,
            row.log2_ratio,
            row.semitones,
            row.freq_hz,
            row.c_value,
            row.prominence,
            row.width_st
        ));
    }
    out
}

#[cfg(test)]
fn t_critical_975(df: usize) -> f32 {
    match df {
        0 => 0.0,
        1 => 12.706,
        2 => 4.303,
        3 => 3.182,
        4 => 2.776,
        5 => 2.571,
        6 => 2.447,
        7 => 2.365,
        8 => 2.306,
        9 => 2.262,
        10 => 2.228,
        11 => 2.201,
        12 => 2.179,
        13 => 2.160,
        14 => 2.145,
        15 => 2.131,
        16 => 2.120,
        17 => 2.110,
        18 => 2.101,
        19 => 2.093,
        20 => 2.086,
        21 => 2.080,
        22 => 2.074,
        23 => 2.069,
        24 => 2.064,
        25 => 2.060,
        26 => 2.056,
        27 => 2.052,
        28 => 2.048,
        29 => 2.045,
        30 => 2.042,
        _ => 1.960,
    }
}

#[cfg(test)]
fn mean_se_t_ci95(values: &[f32]) -> (f32, f32, f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = values.len();
    let mean = values.iter().copied().sum::<f32>() / n as f32;
    if n < 2 {
        return (mean, 0.0, mean, mean);
    }
    let var = values
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / (n as f32 - 1.0);
    let se = var.max(0.0).sqrt() / (n as f32).sqrt();
    let half = t_critical_975(n - 1) * se;
    (mean, se, mean - half, mean + half)
}

fn e4_collect_wr_tail_agent_rows(weights: &[f32]) -> Vec<E4WrTailAgentRow> {
    let mut rows = Vec::new();
    for (wr_i, wr_raw) in E4_WR_GRID.iter().copied().enumerate() {
        let wr = normalize_wr(wr_raw);
        for (mw_i, mirror_raw) in weights.iter().copied().enumerate() {
            let mirror_weight = mirror_raw.clamp(0.0, 1.0);
            for rep_i in 0..E4_WR_REPS {
                let seed =
                    E4_WR_BASE_SEED + (wr_i as u64) * 10_000 + (mw_i as u64) * 1_000 + rep_i as u64;
                let samples = run_e4_condition_tail_samples_with_wr(
                    mirror_weight,
                    seed,
                    E4_TAIL_WINDOW_STEPS,
                    wr,
                );
                let step = samples.steps_total.saturating_sub(1);
                let Some(final_rows) = samples.agent_freqs_by_step.last() else {
                    continue;
                };
                for row in final_rows {
                    if !row.freq_hz.is_finite() || row.freq_hz <= 0.0 {
                        continue;
                    }
                    rows.push(E4WrTailAgentRow {
                        wr,
                        mirror_weight,
                        seed,
                        step,
                        agent_id: row.agent_id,
                        freq_hz: row.freq_hz,
                    });
                }
            }
        }
    }
    rows.sort_by(|a, b| {
        a.wr.partial_cmp(&b.wr)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.mirror_weight
                    .partial_cmp(&b.mirror_weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.seed.cmp(&b.seed))
            .then_with(|| a.agent_id.cmp(&b.agent_id))
    });
    rows
}

fn e4_wr_tail_agents_csv(rows: &[E4WrTailAgentRow]) -> String {
    let mut out = String::from("wr,mirror_weight,seed,step,agent_id,freq_hz\n");
    for row in rows {
        out.push_str(&format!(
            "{:.3},{:.3},{},{},{},{:.6}\n",
            row.wr, row.mirror_weight, row.seed, row.step, row.agent_id, row.freq_hz
        ));
    }
    out
}

fn e4_wr_bind_runs_from_tail_agents(
    rows: &[E4WrTailAgentRow],
    anchor_hz: f32,
) -> Vec<E4WrBindRunRow> {
    let mut grouped: std::collections::HashMap<(i32, i32, u64), Vec<(u64, f32)>> =
        std::collections::HashMap::new();
    for row in rows {
        grouped
            .entry((float_key(row.wr), float_key(row.mirror_weight), row.seed))
            .or_default()
            .push((row.agent_id, row.freq_hz));
    }
    let mut out = Vec::with_capacity(grouped.len());
    for ((wr_key, mirror_key, seed), mut agent_rows) in grouped {
        agent_rows.sort_by_key(|(agent_id, _)| *agent_id);
        let freqs: Vec<f32> = agent_rows.iter().map(|(_, freq)| *freq).collect();
        let eval = bind_eval_from_freqs(&freqs);
        let (root_fit_anchor, ceiling_fit_anchor, delta_bind_anchor) =
            bind_scores_anchor_from_freqs(anchor_hz, &freqs);
        out.push(E4WrBindRunRow {
            wr: float_from_key(wr_key),
            mirror_weight: float_from_key(mirror_key),
            seed,
            root_fit: eval.root_fit,
            ceiling_fit: eval.ceiling_fit,
            delta_bind: eval.delta_bind,
            root_fit_anchor,
            ceiling_fit_anchor,
            delta_bind_anchor,
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
    });
    out
}

fn e4_wr_bind_runs_csv(rows: &[E4WrBindRunRow]) -> String {
    let mut out = String::from(
        "wr,mirror_weight,seed,root_fit,ceiling_fit,delta_bind,root_fit_anchor,ceiling_fit_anchor,delta_bind_anchor\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{:.3},{:.3},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            row.wr,
            row.mirror_weight,
            row.seed,
            row.root_fit,
            row.ceiling_fit,
            row.delta_bind,
            row.root_fit_anchor,
            row.ceiling_fit_anchor,
            row.delta_bind_anchor
        ));
    }
    out
}

fn e4_wr_bind_summary_rows(rows: &[E4WrBindRunRow]) -> Vec<E4WrBindSummaryRow> {
    let mut grouped: std::collections::HashMap<(i32, i32), Vec<&E4WrBindRunRow>> =
        std::collections::HashMap::new();
    for row in rows {
        grouped
            .entry((float_key(row.wr), float_key(row.mirror_weight)))
            .or_default()
            .push(row);
    }
    let mut out = Vec::new();
    for ((wr_key, mirror_key), group) in grouped {
        let root_vals: Vec<f32> = group.iter().map(|row| row.root_fit).collect();
        let ceiling_vals: Vec<f32> = group.iter().map(|row| row.ceiling_fit).collect();
        let delta_vals: Vec<f32> = group.iter().map(|row| row.delta_bind).collect();
        let root_anchor_vals: Vec<f32> = group.iter().map(|row| row.root_fit_anchor).collect();
        let ceiling_anchor_vals: Vec<f32> =
            group.iter().map(|row| row.ceiling_fit_anchor).collect();
        let delta_anchor_vals: Vec<f32> = group.iter().map(|row| row.delta_bind_anchor).collect();
        let seed = E4_BOOTSTRAP_SEED
            ^ 0xE4B1_DA7Au64
            ^ (wr_key as i64 as u64).wrapping_mul(0x9E37_79B9)
            ^ (mirror_key as i64 as u64).wrapping_mul(0x85EB_CA6B);
        let (root_fit_mean, root_fit_ci_lo, root_fit_ci_hi) =
            bootstrap_mean_ci95(&root_vals, E4_BOOTSTRAP_ITERS, seed ^ 0x11);
        let (ceiling_fit_mean, ceiling_fit_ci_lo, ceiling_fit_ci_hi) =
            bootstrap_mean_ci95(&ceiling_vals, E4_BOOTSTRAP_ITERS, seed ^ 0x22);
        let (delta_bind_mean, delta_bind_ci_lo, delta_bind_ci_hi) =
            bootstrap_mean_ci95(&delta_vals, E4_BOOTSTRAP_ITERS, seed ^ 0x33);
        let (root_fit_anchor_mean, root_fit_anchor_ci_lo, root_fit_anchor_ci_hi) =
            bootstrap_mean_ci95(&root_anchor_vals, E4_BOOTSTRAP_ITERS, seed ^ 0x44);
        let (ceiling_fit_anchor_mean, ceiling_fit_anchor_ci_lo, ceiling_fit_anchor_ci_hi) =
            bootstrap_mean_ci95(&ceiling_anchor_vals, E4_BOOTSTRAP_ITERS, seed ^ 0x55);
        let (delta_bind_anchor_mean, delta_bind_anchor_ci_lo, delta_bind_anchor_ci_hi) =
            bootstrap_mean_ci95(&delta_anchor_vals, E4_BOOTSTRAP_ITERS, seed ^ 0x66);
        out.push(E4WrBindSummaryRow {
            wr: float_from_key(wr_key),
            mirror_weight: float_from_key(mirror_key),
            root_fit_mean,
            root_fit_ci_lo,
            root_fit_ci_hi,
            ceiling_fit_mean,
            ceiling_fit_ci_lo,
            ceiling_fit_ci_hi,
            delta_bind_mean,
            delta_bind_ci_lo,
            delta_bind_ci_hi,
            root_fit_anchor_mean,
            root_fit_anchor_ci_lo,
            root_fit_anchor_ci_hi,
            ceiling_fit_anchor_mean,
            ceiling_fit_anchor_ci_lo,
            ceiling_fit_anchor_ci_hi,
            delta_bind_anchor_mean,
            delta_bind_anchor_ci_lo,
            delta_bind_anchor_ci_hi,
            n_seeds: group.len(),
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
    });
    out
}

fn e4_wr_fingerprint_runs_from_tail_agents(
    rows: &[E4WrTailAgentRow],
) -> Vec<E4WrFingerprintRunRow> {
    let mut grouped: std::collections::HashMap<(i32, i32, u64), Vec<(u64, f32)>> =
        std::collections::HashMap::new();
    for row in rows {
        grouped
            .entry((float_key(row.wr), float_key(row.mirror_weight), row.seed))
            .or_default()
            .push((row.agent_id, row.freq_hz));
    }
    let mut out = Vec::new();
    for ((wr_key, mirror_key, seed), mut agent_rows) in grouped {
        agent_rows.sort_by_key(|(agent_id, _)| *agent_id);
        let freqs: Vec<f32> = agent_rows.iter().map(|(_, freq)| *freq).collect();
        let probs = e4_interval_fingerprint_probs(&freqs, E4_FINGERPRINT_TOL_CENTS);
        let wr = float_from_key(wr_key);
        let mirror_weight = float_from_key(mirror_key);
        for (idx, category) in E4_FINGERPRINT_LABELS.iter().enumerate() {
            out.push(E4WrFingerprintRunRow {
                wr,
                mirror_weight,
                seed,
                category,
                prob: probs[idx],
            });
        }
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
            .then_with(|| {
                e4_fingerprint_label_index(a.category).cmp(&e4_fingerprint_label_index(b.category))
            })
    });
    out
}

fn e4_wr_fingerprint_runs_csv(rows: &[E4WrFingerprintRunRow]) -> String {
    let mut out =
        String::from("wr,mirror_weight,seed,category,prob,interval_unit,tol_cents,folding\n");
    for row in rows {
        out.push_str(&format!(
            "{:.3},{:.3},{},{},{:.6},cents,{:.3},abs_mod_1200\n",
            row.wr, row.mirror_weight, row.seed, row.category, row.prob, E4_FINGERPRINT_TOL_CENTS
        ));
    }
    out
}

fn e4_wr_fingerprint_runs_wide_csv(rows: &[E4WrFingerprintRunRow]) -> String {
    let mut grouped: std::collections::HashMap<(i32, i32, u64), [f32; 13]> =
        std::collections::HashMap::new();
    for row in rows {
        let idx = e4_fingerprint_label_index(row.category).min(E4_FINGERPRINT_LABELS.len() - 1);
        grouped
            .entry((float_key(row.wr), float_key(row.mirror_weight), row.seed))
            .or_insert([0.0; 13])[idx] = row.prob;
    }
    let mut keys: Vec<(i32, i32, u64)> = grouped.keys().copied().collect();
    keys.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });

    let mut out = String::from("wr,mirror_weight,seed,interval_unit,tol_cents,folding");
    for category in &E4_FINGERPRINT_LABELS {
        out.push_str(&format!(",p_{category}"));
    }
    out.push('\n');

    for (wr_key, mirror_key, seed) in keys {
        let probs = grouped
            .get(&(wr_key, mirror_key, seed))
            .copied()
            .unwrap_or([0.0; 13]);
        out.push_str(&format!(
            "{:.3},{:.3},{},cents,{:.3},abs_mod_1200",
            float_from_key(wr_key),
            float_from_key(mirror_key),
            seed,
            E4_FINGERPRINT_TOL_CENTS
        ));
        for p in probs {
            out.push_str(&format!(",{p:.6}"));
        }
        out.push('\n');
    }
    out
}

fn e4_wr_fingerprint_summary_rows(
    rows: &[E4WrFingerprintRunRow],
) -> Vec<E4WrFingerprintSummaryRow> {
    let mut grouped: std::collections::HashMap<(i32, i32, &'static str), Vec<f32>> =
        std::collections::HashMap::new();
    for row in rows {
        grouped
            .entry((
                float_key(row.wr),
                float_key(row.mirror_weight),
                row.category,
            ))
            .or_default()
            .push(row.prob);
    }
    let mut out = Vec::new();
    for ((wr_key, mirror_key, category), values) in grouped {
        let seed = E4_BOOTSTRAP_SEED
            ^ 0xF1A6_11A5_u64
            ^ (wr_key as i64 as u64).wrapping_mul(0x9E37_79B9)
            ^ (mirror_key as i64 as u64).wrapping_mul(0x85EB_CA6B)
            ^ (e4_fingerprint_label_index(category) as u64).wrapping_mul(0xC2B2_AE35);
        let (mean_prob, prob_ci_lo, prob_ci_hi) =
            bootstrap_mean_ci95(&values, E4_BOOTSTRAP_ITERS, seed);
        out.push(E4WrFingerprintSummaryRow {
            wr: float_from_key(wr_key),
            mirror_weight: float_from_key(mirror_key),
            category,
            mean_prob: mean_prob.clamp(0.0, 1.0),
            prob_ci_lo: prob_ci_lo.clamp(0.0, 1.0),
            prob_ci_hi: prob_ci_hi.clamp(0.0, 1.0),
            n_seeds: values.len(),
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
            .then_with(|| {
                e4_fingerprint_label_index(a.category).cmp(&e4_fingerprint_label_index(b.category))
            })
    });
    out
}

fn e4_wr_fingerprint_summary_csv(rows: &[E4WrFingerprintSummaryRow]) -> String {
    let mut out = String::from(
        "wr,mirror_weight,category,prob_mean,prob_ci_lo,prob_ci_hi,n_seeds,error_kind,interval_unit,tol_cents,folding\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{:.3},{:.3},{},{:.6},{:.6},{:.6},{},bootstrap_pctl95,cents,{:.3},abs_mod_1200\n",
            row.wr,
            row.mirror_weight,
            row.category,
            row.mean_prob,
            row.prob_ci_lo,
            row.prob_ci_hi,
            row.n_seeds,
            E4_FINGERPRINT_TOL_CENTS
        ));
    }
    out
}

fn e4_wr_sweep_summary_csv(
    bind_rows: &[E4WrBindSummaryRow],
    fp_rows: &[E4WrFingerprintSummaryRow],
) -> String {
    let mut fp_mean_map: std::collections::HashMap<(i32, i32, &'static str), f32> =
        std::collections::HashMap::new();
    for row in fp_rows {
        fp_mean_map.insert(
            (
                float_key(row.wr),
                float_key(row.mirror_weight),
                row.category,
            ),
            row.mean_prob,
        );
    }

    let mut out = String::from(
        "wr,mirror_weight,root_fit_mean,root_fit_ci95,ceiling_fit_mean,ceiling_fit_ci95,delta_bind_mean,delta_bind_ci95,n_seeds,error_kind,root_fit_anchor_mean,root_fit_anchor_ci95,ceiling_fit_anchor_mean,ceiling_fit_anchor_ci95,delta_bind_anchor_mean,delta_bind_anchor_ci95",
    );
    for category in &E4_FINGERPRINT_LABELS {
        out.push_str(&format!(",p_{category}"));
    }
    out.push('\n');

    for row in bind_rows {
        let wr_key = float_key(row.wr);
        let mirror_key = float_key(row.mirror_weight);
        let root_ci95 = 0.5 * (row.root_fit_ci_hi - row.root_fit_ci_lo).max(0.0);
        let ceiling_ci95 = 0.5 * (row.ceiling_fit_ci_hi - row.ceiling_fit_ci_lo).max(0.0);
        let delta_ci95 = 0.5 * (row.delta_bind_ci_hi - row.delta_bind_ci_lo).max(0.0);
        let root_anchor_ci95 =
            0.5 * (row.root_fit_anchor_ci_hi - row.root_fit_anchor_ci_lo).max(0.0);
        let ceiling_anchor_ci95 =
            0.5 * (row.ceiling_fit_anchor_ci_hi - row.ceiling_fit_anchor_ci_lo).max(0.0);
        let delta_anchor_ci95 =
            0.5 * (row.delta_bind_anchor_ci_hi - row.delta_bind_anchor_ci_lo).max(0.0);
        out.push_str(&format!(
            "{:.3},{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},bootstrap_pctl95,{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            row.wr,
            row.mirror_weight,
            row.root_fit_mean,
            root_ci95,
            row.ceiling_fit_mean,
            ceiling_ci95,
            row.delta_bind_mean,
            delta_ci95,
            row.n_seeds,
            row.root_fit_anchor_mean,
            root_anchor_ci95,
            row.ceiling_fit_anchor_mean,
            ceiling_anchor_ci95,
            row.delta_bind_anchor_mean,
            delta_anchor_ci95
        ));
        for category in &E4_FINGERPRINT_LABELS {
            let prob = fp_mean_map
                .get(&(wr_key, mirror_key, *category))
                .copied()
                .unwrap_or(0.0);
            out.push_str(&format!(",{prob:.6}"));
        }
        out.push('\n');
    }
    out
}

fn render_e4_wr_delta_bind_vs_mirror(
    out_path: &Path,
    rows: &[E4WrBindSummaryRow],
) -> Result<(), Box<dyn Error>> {
    if rows.is_empty() {
        return Ok(());
    }
    let mut y_min = 0.0f32;
    let mut y_max = 0.0f32;
    for row in rows {
        y_min = y_min.min(row.delta_bind_ci_lo).min(row.delta_bind_mean);
        y_max = y_max.max(row.delta_bind_ci_hi).max(row.delta_bind_mean);
    }
    if (y_max - y_min).abs() < 1e-6 {
        y_min -= 0.1;
        y_max += 0.1;
    }
    let pad = 0.15 * (y_max - y_min);

    let mut wr_keys: Vec<i32> = rows.iter().map(|row| float_key(row.wr)).collect();
    wr_keys.sort();
    wr_keys.dedup();

    let root = bitmap_root(out_path, (1400, 860)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "E4 w_r sweep: DeltaBind vs mirror_weight (mean ± bootstrap CI95)",
            ("sans-serif", 24),
        )
        .margin(12)
        .x_label_area_size(46)
        .y_label_area_size(68)
        .build_cartesian_2d(0.0f32..1.0f32, (y_min - pad)..(y_max + pad))?;

    chart
        .configure_mesh()
        .x_desc("mirror_weight")
        .y_desc("DeltaBind")
        .draw()?;
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0.0, 0.0), (1.0, 0.0)],
        BLACK.mix(0.8).stroke_width(2),
    )))?;

    for (i, wr_key) in wr_keys.iter().enumerate() {
        let mut group: Vec<&E4WrBindSummaryRow> = rows
            .iter()
            .filter(|row| float_key(row.wr) == *wr_key)
            .collect();
        if group.is_empty() {
            continue;
        }
        group.sort_by(|a, b| {
            a.mirror_weight
                .partial_cmp(&b.mirror_weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let color = Palette99::pick(i).mix(0.90);
        chart
            .draw_series(LineSeries::new(
                group
                    .iter()
                    .map(|row| (row.mirror_weight, row.delta_bind_mean)),
                color.stroke_width(2),
            ))?
            .label(format!("w_r={:.2}", float_from_key(*wr_key)))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 22, y)], color.stroke_width(2))
            });

        chart.draw_series(group.iter().map(|row| {
            PathElement::new(
                vec![
                    (row.mirror_weight, row.delta_bind_ci_lo),
                    (row.mirror_weight, row.delta_bind_ci_hi),
                ],
                color.mix(0.65).stroke_width(1),
            )
        }))?;
        chart.draw_series(
            group.iter().map(|row| {
                Circle::new((row.mirror_weight, row.delta_bind_mean), 4, color.filled())
            }),
        )?;
    }

    chart
        .configure_series_labels()
        .border_style(BLACK.mix(0.4))
        .background_style(WHITE.mix(0.8))
        .draw()?;
    root.present()?;
    Ok(())
}

fn render_e4_wr_root_ceiling_vs_mirror(
    out_path: &Path,
    rows: &[E4WrBindSummaryRow],
    wr_grid: &[f32],
) -> Result<(), Box<dyn Error>> {
    if rows.is_empty() || wr_grid.is_empty() {
        return Ok(());
    }
    let mut y_max = 0.0f32;
    for row in rows {
        y_max = y_max.max(row.root_fit_ci_hi).max(row.ceiling_fit_ci_hi);
    }
    y_max = (y_max * 1.15).max(1e-4);

    let n = wr_grid.len();
    let cols = 3usize.min(n.max(1));
    let rows_n = n.div_ceil(cols);
    let root = bitmap_root(out_path, (1700, 420 * rows_n as u32)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((rows_n, cols));

    for (panel_i, wr_raw) in wr_grid.iter().copied().enumerate() {
        if panel_i >= panels.len() {
            break;
        }
        let wr = normalize_wr(wr_raw);
        let area = &panels[panel_i];
        let wr_key = float_key(wr);
        let mut group: Vec<&E4WrBindSummaryRow> = rows
            .iter()
            .filter(|row| float_key(row.wr) == wr_key)
            .collect();
        group.sort_by(|a, b| {
            a.mirror_weight
                .partial_cmp(&b.mirror_weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if group.is_empty() {
            continue;
        }
        let mut chart = ChartBuilder::on(area)
            .caption(format!("w_r={wr:.2}"), ("sans-serif", 18))
            .margin(8)
            .x_label_area_size(36)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0f32..1.0f32, 0.0f32..y_max)?;
        chart
            .configure_mesh()
            .x_desc("mirror_weight")
            .y_desc("fit")
            .draw()?;

        chart.draw_series(LineSeries::new(
            group
                .iter()
                .map(|row| (row.mirror_weight, row.root_fit_mean)),
            BLUE.mix(0.95).stroke_width(2),
        ))?;
        chart.draw_series(group.iter().map(|row| {
            Circle::new(
                (row.mirror_weight, row.root_fit_mean),
                3,
                BLUE.mix(0.95).filled(),
            )
        }))?;
        chart.draw_series(LineSeries::new(
            group
                .iter()
                .map(|row| (row.mirror_weight, row.ceiling_fit_mean)),
            RED.mix(0.90).stroke_width(2),
        ))?;
        chart.draw_series(group.iter().map(|row| {
            TriangleMarker::new(
                (row.mirror_weight, row.ceiling_fit_mean),
                5,
                RED.mix(0.90).filled(),
            )
        }))?;
    }

    root.present()?;
    Ok(())
}

fn render_e4_wr_fingerprint_heatmap(
    out_path: &Path,
    rows: &[E4WrFingerprintSummaryRow],
    wr_focus: &[f32],
    mirror_weights: &[f32],
) -> Result<(), Box<dyn Error>> {
    if rows.is_empty() || wr_focus.is_empty() || mirror_weights.is_empty() {
        return Ok(());
    }
    let labels = E4_FINGERPRINT_LABELS;
    let mut prob_map: std::collections::HashMap<(i32, i32, &'static str), f32> =
        std::collections::HashMap::new();
    let mut max_prob = 0.0f32;
    for row in rows {
        prob_map.insert(
            (
                float_key(row.wr),
                float_key(row.mirror_weight),
                row.category,
            ),
            row.mean_prob,
        );
        max_prob = max_prob.max(row.mean_prob);
    }
    max_prob = max_prob.max(1e-6);

    let root = bitmap_root(out_path, (1700, 360 * wr_focus.len() as u32)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut panels = root.split_evenly((wr_focus.len(), 1));
    for (panel_i, wr_raw) in wr_focus.iter().copied().enumerate() {
        if panel_i >= panels.len() {
            break;
        }
        let wr = normalize_wr(wr_raw);
        let wr_key = float_key(wr);
        let area = &mut panels[panel_i];
        let mut chart = ChartBuilder::on(area)
            .caption(
                format!("w_r={wr:.2} interval fingerprint"),
                ("sans-serif", 18),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(72)
            .build_cartesian_2d(0i32..mirror_weights.len() as i32, 0i32..labels.len() as i32)?;
        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("mirror_weight")
            .y_desc("category")
            .x_labels(mirror_weights.len())
            .y_labels(labels.len())
            .x_label_formatter(&|x| {
                let idx = (*x).clamp(0, mirror_weights.len().saturating_sub(1) as i32) as usize;
                format!("{:.2}", mirror_weights[idx])
            })
            .y_label_formatter(&|y| {
                let idx = (*y).clamp(0, labels.len().saturating_sub(1) as i32) as usize;
                labels[idx].to_string()
            })
            .draw()?;

        for (x, mirror_weight) in mirror_weights.iter().copied().enumerate() {
            let mirror_key = float_key(mirror_weight);
            for (y, category) in labels.iter().enumerate() {
                let prob = prob_map
                    .get(&(wr_key, mirror_key, *category))
                    .copied()
                    .unwrap_or(0.0);
                let t = (prob / max_prob).clamp(0.0, 1.0) as f64;
                let color = HSLColor((240.0 - 240.0 * t) / 360.0, 0.85, 0.22 + 0.50 * t);
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(x as i32, y as i32), (x as i32 + 1, y as i32 + 1)],
                    color.filled(),
                )))?;
            }
        }
    }
    root.present()?;
    Ok(())
}

fn grouped_freqs_by_wr_mw(
    rows: &[E4WrTailAgentRow],
) -> std::collections::HashMap<(i32, i32), Vec<f32>> {
    let mut map: std::collections::HashMap<(i32, i32), Vec<f32>> = std::collections::HashMap::new();
    for row in rows {
        map.entry((float_key(row.wr), float_key(row.mirror_weight)))
            .or_default()
            .push(row.freq_hz);
    }
    map
}

fn grouped_freqs_by_wr_mw_seed(
    rows: &[E4WrTailAgentRow],
) -> std::collections::HashMap<(i32, i32, u64), Vec<f32>> {
    let mut map: std::collections::HashMap<(i32, i32, u64), Vec<f32>> =
        std::collections::HashMap::new();
    for row in rows {
        map.entry((float_key(row.wr), float_key(row.mirror_weight), row.seed))
            .or_default()
            .push(row.freq_hz);
    }
    map
}

fn mean_scan_at_indices(scan: &[f32], indices: &[usize]) -> f32 {
    if indices.is_empty() || scan.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for &idx in indices {
        if let Some(&value) = scan.get(idx)
            && value.is_finite()
        {
            sum += value;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn mean_repulsion_from_indices(indices: &[usize], log2_ratio_scan: &[f32], sigma: f32) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }
    let sigma = sigma.max(1e-6);
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for (i, &idx) in indices.iter().enumerate() {
        let Some(&x) = log2_ratio_scan.get(idx) else {
            continue;
        };
        let mut rep = 0.0f32;
        for (j, &other_idx) in indices.iter().enumerate() {
            if i == j {
                continue;
            }
            let Some(&y) = log2_ratio_scan.get(other_idx) else {
                continue;
            };
            rep += (-(x - y).abs() / sigma).exp();
        }
        if rep.is_finite() {
            sum += rep;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn pitch_diversity_st_from_indices(semitones_scan: &[f32], indices: &[usize]) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }
    let mut vals = Vec::with_capacity(indices.len());
    for &idx in indices {
        if let Some(&v) = semitones_scan.get(idx)
            && v.is_finite()
        {
            vals.push(v);
        }
    }
    mean_std_scalar(&vals).1
}

fn bind_eval_from_indices(anchor_hz: f32, log2_ratio_scan: &[f32], indices: &[usize]) -> BindEval {
    let mut freqs = Vec::with_capacity(indices.len());
    for &idx in indices {
        let Some(&l2) = log2_ratio_scan.get(idx) else {
            continue;
        };
        let freq = anchor_hz * 2.0f32.powf(l2);
        if freq.is_finite() && freq > 0.0 {
            freqs.push(freq);
        }
    }
    bind_eval_from_freqs(&freqs)
}

#[allow(clippy::too_many_arguments)]
fn e4_wr_probe_best_index(
    agent_i: usize,
    current_indices: &[usize],
    c_scan: &[f32],
    log2_ratio_scan: &[f32],
    min_idx: usize,
    max_idx: usize,
    k: i32,
    lambda: f32,
    sigma: f32,
    peak_bins: Option<&[usize]>,
) -> usize {
    let current_idx = current_indices[agent_i].clamp(min_idx, max_idx);
    let sigma = sigma.max(1e-6);
    let skip_repulsion = lambda <= 0.0;
    let mut best_idx = current_idx;
    let mut best_score = f32::NEG_INFINITY;

    let evaluate = |cand_idx: usize, best_score: &mut f32, best_idx: &mut usize| {
        if cand_idx < min_idx || cand_idx > max_idx || cand_idx >= c_scan.len() {
            return;
        }
        let Some(&cand_log2) = log2_ratio_scan.get(cand_idx) else {
            return;
        };
        let mut repulsion = 0.0f32;
        if !skip_repulsion {
            for (j, &other_idx) in current_indices.iter().enumerate() {
                if j == agent_i {
                    continue;
                }
                let Some(&other_log2) = log2_ratio_scan.get(other_idx) else {
                    continue;
                };
                repulsion += (-(cand_log2 - other_log2).abs() / sigma).exp();
            }
        }
        let score = c_scan[cand_idx] - lambda * repulsion;
        if score > *best_score {
            *best_score = score;
            *best_idx = cand_idx;
        }
    };

    if let Some(pool) = peak_bins {
        for &cand_idx in pool {
            evaluate(cand_idx, &mut best_score, &mut best_idx);
        }
        evaluate(current_idx, &mut best_score, &mut best_idx);
    } else {
        let start = (current_idx as isize - k as isize).max(min_idx as isize) as usize;
        let end = (current_idx as isize + k as isize).min(max_idx as isize) as usize;
        for cand_idx in start..=end {
            evaluate(cand_idx, &mut best_score, &mut best_idx);
        }
    }
    best_idx
}

#[allow(clippy::too_many_arguments)]
fn e4_wr_probe_update_indices(
    indices: &mut [usize],
    c_scan: &[f32],
    log2_ratio_scan: &[f32],
    min_idx: usize,
    max_idx: usize,
    k: i32,
    lambda: f32,
    sigma: f32,
    peak_bins: Option<&[usize]>,
    synchronous: bool,
) {
    if indices.is_empty() {
        return;
    }
    if synchronous {
        let prev = indices.to_vec();
        let mut next = prev.clone();
        for (agent_i, slot) in next.iter_mut().enumerate() {
            *slot = e4_wr_probe_best_index(
                agent_i,
                &prev,
                c_scan,
                log2_ratio_scan,
                min_idx,
                max_idx,
                k,
                lambda,
                sigma,
                peak_bins,
            );
        }
        indices.copy_from_slice(&next);
    } else {
        for agent_i in 0..indices.len() {
            let best = e4_wr_probe_best_index(
                agent_i,
                indices,
                c_scan,
                log2_ratio_scan,
                min_idx,
                max_idx,
                k,
                lambda,
                sigma,
                peak_bins,
            );
            indices[agent_i] = best;
        }
    }
}

fn e4_wr_dynamics_probe_rows(
    grouped_freqs: &std::collections::HashMap<(i32, i32, u64), Vec<f32>>,
    space: &Log2Space,
    anchor_hz: f32,
    du_scan: &[f32],
) -> Vec<E4WrDynamicsProbeRow> {
    let meta = e4_paper_meta();
    let center_log2 = meta.center_cents / 1200.0;
    let half_range = 0.5 * meta.range_oct.max(0.0);
    let k = k_from_semitones(E4_DYNAMICS_STEP_SEMITONES.max(1e-3));
    let sigma = E4_DYNAMICS_REPULSION_SIGMA.max(1e-6);

    let mut rows = Vec::new();
    let mut keys: Vec<(i32, i32, u64)> = grouped_freqs.keys().copied().collect();
    keys.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });

    for (wr_key, mirror_key, seed) in keys {
        let Some(freqs) = grouped_freqs.get(&(wr_key, mirror_key, seed)) else {
            continue;
        };
        let wr = float_from_key(wr_key);
        let mirror_weight = float_from_key(mirror_key);
        let scan = compute_e4_landscape_scans(space, anchor_hz, wr, mirror_weight, freqs, du_scan);
        let (min_idx, max_idx) = log2_ratio_bounds(
            &scan.log2_ratio,
            center_log2 - half_range,
            center_log2 + half_range,
        );

        let mut initial_indices = Vec::new();
        for &freq in freqs {
            let idx = space
                .index_of_freq(freq)
                .unwrap_or_else(|| nearest_bin(space, freq))
                .clamp(min_idx, max_idx);
            initial_indices.push(idx);
        }
        if initial_indices.is_empty() {
            continue;
        }

        let mut peak_bins: Vec<usize> =
            extract_peak_rows_from_c_scan(space, anchor_hz, &scan, E4_DYNAMICS_PEAK_TOP_N.max(1))
                .into_iter()
                .map(|p| p.bin_idx)
                .filter(|idx| *idx >= min_idx && *idx <= max_idx)
                .collect();
        peak_bins.sort_unstable();
        peak_bins.dedup();
        if peak_bins.is_empty() {
            peak_bins = initial_indices.clone();
            peak_bins.sort_unstable();
            peak_bins.dedup();
        }

        for mode in E4DynamicsProbeMode::ALL {
            let mut indices = initial_indices.clone();
            for step in 0..=E4_DYNAMICS_PROBE_STEPS {
                let eval = bind_eval_from_indices(anchor_hz, &scan.log2_ratio, &indices);
                rows.push(E4WrDynamicsProbeRow {
                    wr,
                    mirror_weight,
                    seed,
                    mode: mode.label(),
                    step,
                    n_agents: indices.len(),
                    mean_c01: mean_scan_at_indices(&scan.c, &indices),
                    mean_h01: mean_scan_at_indices(&scan.h, &indices),
                    mean_r01: mean_scan_at_indices(&scan.r, &indices),
                    mean_repulsion: mean_repulsion_from_indices(&indices, &scan.log2_ratio, sigma),
                    root_fit: eval.root_fit,
                    ceiling_fit: eval.ceiling_fit,
                    delta_bind: eval.delta_bind,
                    pitch_diversity_st: pitch_diversity_st_from_indices(&scan.semitones, &indices),
                });

                if step == E4_DYNAMICS_PROBE_STEPS {
                    break;
                }
                let peak_pool = if mode.peak_restricted() {
                    Some(peak_bins.as_slice())
                } else {
                    None
                };
                e4_wr_probe_update_indices(
                    &mut indices,
                    &scan.c,
                    &scan.log2_ratio,
                    min_idx,
                    max_idx,
                    k,
                    mode.lambda(),
                    sigma,
                    peak_pool,
                    mode.synchronous(),
                );
            }
        }
    }

    rows.sort_by(|a, b| {
        a.wr.partial_cmp(&b.wr)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.mirror_weight
                    .partial_cmp(&b.mirror_weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.seed.cmp(&b.seed))
            .then_with(|| a.mode.cmp(b.mode))
            .then_with(|| a.step.cmp(&b.step))
    });
    rows
}

fn e4_wr_dynamics_probe_timeseries_csv(rows: &[E4WrDynamicsProbeRow]) -> String {
    let mut out = String::from(
        "wr,mirror_weight,seed,mode,step,n_agents,mean_c01,mean_h01,mean_r01,mean_repulsion,root_fit,ceiling_fit,delta_bind,pitch_diversity_st\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{:.3},{:.3},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            row.wr,
            row.mirror_weight,
            row.seed,
            row.mode,
            row.step,
            row.n_agents,
            row.mean_c01,
            row.mean_h01,
            row.mean_r01,
            row.mean_repulsion,
            row.root_fit,
            row.ceiling_fit,
            row.delta_bind,
            row.pitch_diversity_st
        ));
    }
    out
}

fn e4_wr_dynamics_probe_summary_rows(
    rows: &[E4WrDynamicsProbeRow],
) -> Vec<E4WrDynamicsProbeSummaryRow> {
    let mut grouped: std::collections::HashMap<
        (i32, i32, &'static str, u32),
        Vec<&E4WrDynamicsProbeRow>,
    > = std::collections::HashMap::new();
    for row in rows {
        grouped
            .entry((
                float_key(row.wr),
                float_key(row.mirror_weight),
                row.mode,
                row.step,
            ))
            .or_default()
            .push(row);
    }
    let mut out = Vec::new();
    for ((wr_key, mirror_key, mode, step), group) in grouped {
        let c_vals: Vec<f32> = group.iter().map(|r| r.mean_c01).collect();
        let h_vals: Vec<f32> = group.iter().map(|r| r.mean_h01).collect();
        let r_vals: Vec<f32> = group.iter().map(|r| r.mean_r01).collect();
        let rep_vals: Vec<f32> = group.iter().map(|r| r.mean_repulsion).collect();
        let root_vals: Vec<f32> = group.iter().map(|r| r.root_fit).collect();
        let ceiling_vals: Vec<f32> = group.iter().map(|r| r.ceiling_fit).collect();
        let delta_vals: Vec<f32> = group.iter().map(|r| r.delta_bind).collect();
        let diversity_vals: Vec<f32> = group.iter().map(|r| r.pitch_diversity_st).collect();
        let seed = E4_BOOTSTRAP_SEED
            ^ 0xD1A6_00A3_u64
            ^ (wr_key as i64 as u64).wrapping_mul(0x9E37_79B9)
            ^ (mirror_key as i64 as u64).wrapping_mul(0x85EB_CA6B)
            ^ (step as u64).wrapping_mul(0xC2B2_AE35);
        let (delta_bind_mean, delta_bind_ci_lo, delta_bind_ci_hi) =
            bootstrap_mean_ci95(&delta_vals, E4_BOOTSTRAP_ITERS, seed ^ 0x77);
        out.push(E4WrDynamicsProbeSummaryRow {
            wr: float_from_key(wr_key),
            mirror_weight: float_from_key(mirror_key),
            mode,
            step,
            n_seeds: group.len(),
            mean_c01: mean_std_scalar(&c_vals).0,
            mean_h01: mean_std_scalar(&h_vals).0,
            mean_r01: mean_std_scalar(&r_vals).0,
            mean_repulsion: mean_std_scalar(&rep_vals).0,
            root_fit_mean: mean_std_scalar(&root_vals).0,
            ceiling_fit_mean: mean_std_scalar(&ceiling_vals).0,
            delta_bind_mean,
            delta_bind_ci_lo,
            delta_bind_ci_hi,
            pitch_diversity_st_mean: mean_std_scalar(&diversity_vals).0,
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
            .then_with(|| a.mode.cmp(b.mode))
            .then_with(|| a.step.cmp(&b.step))
    });
    out
}

fn e4_wr_dynamics_probe_summary_csv(rows: &[E4WrDynamicsProbeSummaryRow]) -> String {
    let mut out = String::from(
        "wr,mirror_weight,mode,step,n_seeds,mean_c01,mean_h01,mean_r01,mean_repulsion,root_fit_mean,ceiling_fit_mean,delta_bind_mean,delta_bind_ci_lo,delta_bind_ci_hi,pitch_diversity_st_mean,error_kind\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{:.3},{:.3},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},bootstrap_pctl95\n",
            row.wr,
            row.mirror_weight,
            row.mode,
            row.step,
            row.n_seeds,
            row.mean_c01,
            row.mean_h01,
            row.mean_r01,
            row.mean_repulsion,
            row.root_fit_mean,
            row.ceiling_fit_mean,
            row.delta_bind_mean,
            row.delta_bind_ci_lo,
            row.delta_bind_ci_hi,
            row.pitch_diversity_st_mean
        ));
    }
    out
}

fn e4_diag_candidate_score(
    agent_i: usize,
    current_indices: &[usize],
    cand_idx: usize,
    c_scan: &[f32],
    log2_ratio_scan: &[f32],
    lambda: f32,
    sigma: f32,
) -> Option<f32> {
    if cand_idx >= c_scan.len() {
        return None;
    }
    let cand_log2 = *log2_ratio_scan.get(cand_idx)?;
    let mut repulsion = 0.0f32;
    if lambda > 0.0 {
        let sigma = sigma.max(1e-6);
        for (j, &other_idx) in current_indices.iter().enumerate() {
            if j == agent_i {
                continue;
            }
            let Some(&other_log2) = log2_ratio_scan.get(other_idx) else {
                continue;
            };
            repulsion += (-(cand_log2 - other_log2).abs() / sigma).exp();
        }
    }
    Some(c_scan[cand_idx] - lambda * repulsion)
}

#[allow(clippy::too_many_arguments)]
fn e4_diag_best_index_and_score(
    agent_i: usize,
    current_indices: &[usize],
    c_scan: &[f32],
    log2_ratio_scan: &[f32],
    min_idx: usize,
    max_idx: usize,
    k: i32,
    lambda: f32,
    sigma: f32,
    global: bool,
) -> (usize, f32) {
    let current_idx = current_indices[agent_i].clamp(min_idx, max_idx);
    let (start, end) = if global {
        (min_idx, max_idx)
    } else {
        (
            (current_idx as isize - k as isize).max(min_idx as isize) as usize,
            (current_idx as isize + k as isize).min(max_idx as isize) as usize,
        )
    };
    let mut best_idx = current_idx;
    let mut best_score = f32::NEG_INFINITY;
    for cand_idx in start..=end {
        let Some(score) = e4_diag_candidate_score(
            agent_i,
            current_indices,
            cand_idx,
            c_scan,
            log2_ratio_scan,
            lambda,
            sigma,
        ) else {
            continue;
        };
        if score > best_score {
            best_score = score;
            best_idx = cand_idx;
        }
    }
    if !best_score.is_finite() {
        best_score = 0.0;
    }
    (best_idx, best_score)
}

fn e4_freqs_from_indices(anchor_hz: f32, log2_ratio_scan: &[f32], indices: &[usize]) -> Vec<f32> {
    let mut freqs = Vec::with_capacity(indices.len());
    for &idx in indices {
        let Some(&l2) = log2_ratio_scan.get(idx) else {
            continue;
        };
        let freq = anchor_hz * 2.0f32.powf(l2);
        if freq.is_finite() && freq > 0.0 {
            freqs.push(freq);
        }
    }
    freqs
}

fn mean_alignment_ratio(indices: &[usize], oracle_indices: &[usize]) -> f32 {
    let n = indices.len().min(oracle_indices.len());
    if n == 0 {
        return 0.0;
    }
    let matches = indices
        .iter()
        .zip(oracle_indices.iter())
        .take(n)
        .filter(|(a, b)| a == b)
        .count();
    matches as f32 / n as f32
}

fn mean_abs_st_distance_at_indices(
    semitones_scan: &[f32],
    indices: &[usize],
    oracle_indices: &[usize],
) -> f32 {
    let n = indices.len().min(oracle_indices.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for (&idx, &oracle_idx) in indices.iter().zip(oracle_indices.iter()).take(n) {
        let Some(&a) = semitones_scan.get(idx) else {
            continue;
        };
        let Some(&b) = semitones_scan.get(oracle_idx) else {
            continue;
        };
        if a.is_finite() && b.is_finite() {
            sum += (a - b).abs();
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn mean_c_gain_at_indices(c_scan: &[f32], indices: &[usize], oracle_indices: &[usize]) -> f32 {
    let n = indices.len().min(oracle_indices.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for (&idx, &oracle_idx) in indices.iter().zip(oracle_indices.iter()).take(n) {
        let Some(&a) = c_scan.get(idx) else {
            continue;
        };
        let Some(&b) = c_scan.get(oracle_idx) else {
            continue;
        };
        if a.is_finite() && b.is_finite() {
            sum += b - a;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn e4_abcd_trace_rows(
    grouped_freqs: &std::collections::HashMap<(i32, i32, u64), Vec<f32>>,
    space: &Log2Space,
    anchor_hz: f32,
    du_scan: &[f32],
    steps: u32,
) -> Vec<E4AbcdTraceRow> {
    let meta = e4_paper_meta();
    let center_log2 = meta.center_cents / 1200.0;
    let half_range = 0.5 * meta.range_oct.max(0.0);
    let k = k_from_semitones(E4_DYNAMICS_STEP_SEMITONES.max(1e-3));
    let sigma = E4_DYNAMICS_REPULSION_SIGMA.max(1e-6);
    let lambda = E4_DYNAMICS_BASE_LAMBDA.max(0.0);

    let mut rows = Vec::new();
    let mut keys: Vec<(i32, i32, u64)> = grouped_freqs.keys().copied().collect();
    keys.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });

    for (wr_key, mirror_key, seed) in keys {
        let Some(freqs) = grouped_freqs.get(&(wr_key, mirror_key, seed)) else {
            continue;
        };
        let wr = float_from_key(wr_key);
        let mirror_weight = float_from_key(mirror_key);
        let run_id = format!(
            "wr{}_mw{}_seed{}",
            format_float_token(wr),
            format_float_token(mirror_weight),
            seed
        );

        let init_scan =
            compute_e4_landscape_scans(space, anchor_hz, wr, mirror_weight, freqs, du_scan);
        let (min_idx, max_idx) = log2_ratio_bounds(
            &init_scan.log2_ratio,
            center_log2 - half_range,
            center_log2 + half_range,
        );

        let mut indices = Vec::new();
        for &freq in freqs {
            let idx = space
                .index_of_freq(freq)
                .unwrap_or_else(|| nearest_bin(space, freq))
                .clamp(min_idx, max_idx);
            indices.push(idx);
        }
        if indices.is_empty() {
            continue;
        }

        for step in 0..=steps {
            let freqs_step = e4_freqs_from_indices(anchor_hz, &init_scan.log2_ratio, &indices);
            let scan = compute_e4_landscape_scans(
                space,
                anchor_hz,
                wr,
                mirror_weight,
                &freqs_step,
                du_scan,
            );
            let mut oracle_indices = indices.clone();
            for agent_i in 0..oracle_indices.len() {
                oracle_indices[agent_i] = e4_wr_probe_best_index(
                    agent_i,
                    &indices,
                    &scan.c,
                    &scan.log2_ratio,
                    min_idx,
                    max_idx,
                    k,
                    lambda,
                    sigma,
                    None,
                );
            }

            let a = mean_alignment_ratio(&indices, &oracle_indices);
            let b = mean_abs_st_distance_at_indices(&scan.semitones, &indices, &oracle_indices);
            let c = mean_c_gain_at_indices(&scan.c, &indices, &oracle_indices);
            let eval_agent = bind_eval_from_indices(anchor_hz, &scan.log2_ratio, &indices);
            let eval_oracle = bind_eval_from_indices(anchor_hz, &scan.log2_ratio, &oracle_indices);
            let d = (eval_oracle.delta_bind - eval_agent.delta_bind).clamp(-2.0, 2.0);

            let agent_idx = indices[0];
            let oracle_idx = oracle_indices[0];
            let agent_c01 = scan.c.get(agent_idx).copied().unwrap_or(0.0);
            let oracle_c01 = scan.c.get(oracle_idx).copied().unwrap_or(0.0);
            let agent_log2 = scan.log2_ratio.get(agent_idx).copied().unwrap_or(0.0);
            let oracle_log2 = scan.log2_ratio.get(oracle_idx).copied().unwrap_or(0.0);
            rows.push(E4AbcdTraceRow {
                run_id: run_id.clone(),
                wr,
                mirror_weight,
                seed,
                timing_mode: "baseline_seq",
                step,
                a,
                b,
                c,
                d,
                agent_idx,
                oracle_idx,
                agent_c01,
                oracle_c01,
                agent_log2,
                oracle_log2,
            });

            if step == steps {
                break;
            }
            e4_wr_probe_update_indices(
                &mut indices,
                &scan.c,
                &scan.log2_ratio,
                min_idx,
                max_idx,
                k,
                lambda,
                sigma,
                None,
                false,
            );
        }
    }

    rows.sort_by(|a, b| {
        a.wr.partial_cmp(&b.wr)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.mirror_weight
                    .partial_cmp(&b.mirror_weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.seed.cmp(&b.seed))
            .then_with(|| a.step.cmp(&b.step))
    });
    rows
}

fn e4_abcd_trace_csv(rows: &[E4AbcdTraceRow]) -> String {
    let mut out = String::from(
        "run_id,seed,wr,mirror_weight,timing_mode,step,A,B,C,D,agent_idx,oracle_idx,agent_c01,oracle_c01,agent_log2,oracle_log2\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{},{},{:.3},{:.3},{},{},{:.6},{:.6},{:.6},{:.6},{},{},{:.6},{:.6},{:.6},{:.6}\n",
            row.run_id,
            row.seed,
            row.wr,
            row.mirror_weight,
            row.timing_mode,
            row.step,
            row.a,
            row.b,
            row.c,
            row.d,
            row.agent_idx,
            row.oracle_idx,
            row.agent_c01,
            row.oracle_c01,
            row.agent_log2,
            row.oracle_log2
        ));
    }
    out
}

fn e4_final_freqs_by_mw_seed(
    rows: &[E4TailAgentRow],
) -> std::collections::HashMap<(i32, u64), Vec<f32>> {
    let mut latest_step: std::collections::HashMap<(i32, u64), u32> =
        std::collections::HashMap::new();
    for row in rows {
        let key = (float_key(row.mirror_weight), row.seed);
        latest_step
            .entry(key)
            .and_modify(|step| *step = (*step).max(row.step))
            .or_insert(row.step);
    }
    let mut grouped: std::collections::HashMap<(i32, u64), Vec<(u64, f32)>> =
        std::collections::HashMap::new();
    for row in rows {
        let key = (float_key(row.mirror_weight), row.seed);
        if latest_step.get(&key).copied() != Some(row.step) {
            continue;
        }
        grouped
            .entry(key)
            .or_default()
            .push((row.agent_id, row.freq_hz));
    }
    let mut out = std::collections::HashMap::new();
    for (key, mut vals) in grouped {
        vals.sort_by_key(|(agent_id, _)| *agent_id);
        let freqs: Vec<f32> = vals
            .into_iter()
            .map(|(_, freq)| freq)
            .filter(|freq| freq.is_finite() && *freq > 0.0)
            .collect();
        if !freqs.is_empty() {
            out.insert(key, freqs);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn e4_diag_rows_from_final_freqs(
    final_freqs: &std::collections::HashMap<(i32, u64), Vec<f32>>,
    space: &Log2Space,
    anchor_hz: f32,
    du_scan: &[f32],
    steps: u32,
    lambda: f32,
    sigma: f32,
    k: i32,
) -> (Vec<E4DiagStepRow>, Vec<E4DiagPeakRow>) {
    let meta = e4_paper_meta();
    let center_log2 = meta.center_cents / 1200.0;
    let half_range = 0.5 * meta.range_oct.max(0.0);
    let mut step_rows = Vec::new();
    let mut peak_rows = Vec::new();
    let mut keys: Vec<(i32, u64)> = final_freqs.keys().copied().collect();
    keys.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    for (mw_key, seed) in keys {
        let Some(freqs) = final_freqs.get(&(mw_key, seed)) else {
            continue;
        };
        let mirror_weight = float_from_key(mw_key);
        let init_scan =
            compute_e4_landscape_scans(space, anchor_hz, 1.0, mirror_weight, freqs, du_scan);
        let peaks = extract_peak_rows_from_c_scan(
            space,
            anchor_hz,
            &init_scan,
            E4_DYNAMICS_PEAK_TOP_N.max(1),
        );
        for peak in peaks {
            peak_rows.push(E4DiagPeakRow {
                mirror_weight,
                seed,
                peak_rank: peak.rank,
                peak_idx: peak.bin_idx,
                peak_log2: peak.log2_ratio,
                peak_semitones: peak.semitones,
                peak_value: peak.c_value,
            });
        }
        let (min_idx, max_idx) = log2_ratio_bounds(
            &init_scan.log2_ratio,
            center_log2 - half_range,
            center_log2 + half_range,
        );
        let mut indices = Vec::new();
        for &freq in freqs {
            let idx = space
                .index_of_freq(freq)
                .unwrap_or_else(|| nearest_bin(space, freq))
                .clamp(min_idx, max_idx);
            indices.push(idx);
        }
        if indices.is_empty() {
            continue;
        }

        for step in 0..=steps {
            let freqs_step = e4_freqs_from_indices(anchor_hz, &init_scan.log2_ratio, &indices);
            let scan = compute_e4_landscape_scans(
                space,
                anchor_hz,
                1.0,
                mirror_weight,
                &freqs_step,
                du_scan,
            );
            for agent_i in 0..indices.len() {
                let agent_idx = indices[agent_i].clamp(min_idx, max_idx);
                let agent_score = e4_diag_candidate_score(
                    agent_i,
                    &indices,
                    agent_idx,
                    &scan.c,
                    &scan.log2_ratio,
                    lambda,
                    sigma,
                )
                .unwrap_or(0.0);
                let (oracle_global_idx, oracle_global_score) = e4_diag_best_index_and_score(
                    agent_i,
                    &indices,
                    &scan.c,
                    &scan.log2_ratio,
                    min_idx,
                    max_idx,
                    k,
                    lambda,
                    sigma,
                    true,
                );
                let (oracle_reachable_idx, oracle_reachable_score) = e4_diag_best_index_and_score(
                    agent_i,
                    &indices,
                    &scan.c,
                    &scan.log2_ratio,
                    min_idx,
                    max_idx,
                    k,
                    lambda,
                    sigma,
                    false,
                );
                let agent_st = scan.semitones.get(agent_idx).copied().unwrap_or(0.0);
                let oracle_global_st = scan
                    .semitones
                    .get(oracle_global_idx)
                    .copied()
                    .unwrap_or(agent_st);
                let oracle_reach_st = scan
                    .semitones
                    .get(oracle_reachable_idx)
                    .copied()
                    .unwrap_or(agent_st);
                step_rows.push(E4DiagStepRow {
                    step,
                    mirror_weight,
                    seed,
                    agent_id: agent_i,
                    agent_idx,
                    oracle_global_idx,
                    oracle_reachable_idx,
                    agent_score,
                    oracle_global_score,
                    oracle_reachable_score,
                    gap_global: oracle_global_score - agent_score,
                    gap_reach: oracle_reachable_score - agent_score,
                    idx_err_global: oracle_global_idx.abs_diff(agent_idx) as f32,
                    idx_err_reach: oracle_reachable_idx.abs_diff(agent_idx) as f32,
                    idx_err_global_st: (oracle_global_st - agent_st).abs(),
                    idx_err_reach_st: (oracle_reach_st - agent_st).abs(),
                });
            }
            if step == steps {
                break;
            }
            e4_wr_probe_update_indices(
                &mut indices,
                &scan.c,
                &scan.log2_ratio,
                min_idx,
                max_idx,
                k,
                lambda,
                sigma,
                None,
                false,
            );
        }
    }

    step_rows.sort_by(|a, b| {
        a.mirror_weight
            .partial_cmp(&b.mirror_weight)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.seed.cmp(&b.seed))
            .then_with(|| a.step.cmp(&b.step))
            .then_with(|| a.agent_id.cmp(&b.agent_id))
    });
    peak_rows.sort_by(|a, b| {
        a.mirror_weight
            .partial_cmp(&b.mirror_weight)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.seed.cmp(&b.seed))
            .then_with(|| a.peak_rank.cmp(&b.peak_rank))
    });
    (step_rows, peak_rows)
}

fn e4_diag_step_rows_csv(rows: &[E4DiagStepRow]) -> String {
    let mut out = String::from(
        "step,mw,seed,agent_id,agent_idx,oracle_global_idx,oracle_reachable_idx,agent_score,oracle_global_score,oracle_reachable_score,gap_global,gap_reach,idx_err_global,idx_err_reach,idx_err_global_st,idx_err_reach_st\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{},{:.3},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            row.step,
            row.mirror_weight,
            row.seed,
            row.agent_id,
            row.agent_idx,
            row.oracle_global_idx,
            row.oracle_reachable_idx,
            row.agent_score,
            row.oracle_global_score,
            row.oracle_reachable_score,
            row.gap_global,
            row.gap_reach,
            row.idx_err_global,
            row.idx_err_reach,
            row.idx_err_global_st,
            row.idx_err_reach_st
        ));
    }
    out
}

fn e4_peaks_by_mw_csv(rows: &[E4DiagPeakRow]) -> String {
    let mut out = String::from("mw,seed,peak_rank,peak_idx,peak_log2,peak_semitones,peak_value\n");
    for row in rows {
        out.push_str(&format!(
            "{:.3},{},{},{},{:.6},{:.6},{:.6}\n",
            row.mirror_weight,
            row.seed,
            row.peak_rank,
            row.peak_idx,
            row.peak_log2,
            row.peak_semitones,
            row.peak_value
        ));
    }
    out
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..n {
        let x = a[i];
        let y = b[i];
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 1e-12 || nb <= 1e-12 {
        0.0
    } else {
        (dot / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0)
    }
}

fn l1_mean_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for i in 0..n {
        if a[i].is_finite() && b[i].is_finite() {
            sum += (a[i] - b[i]).abs();
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn e4_landscape_delta_rows(
    weights: &[f32],
    space: &Log2Space,
    anchor_hz: f32,
    du_scan: &[f32],
) -> Vec<E4LandscapeDeltaRow> {
    let fixed_freqs = [anchor_hz];
    let scan0 = compute_e4_landscape_scans(space, anchor_hz, 1.0, 0.0, &fixed_freqs, du_scan);
    let peaks0 = extract_peak_rows_from_c_scan(space, anchor_hz, &scan0, E4_DIAG_PEAK_TOP_K.max(1));
    let mut out = Vec::new();
    for &mw in weights {
        let mirror_weight = mw.clamp(0.0, 1.0);
        let scan =
            compute_e4_landscape_scans(space, anchor_hz, 1.0, mirror_weight, &fixed_freqs, du_scan);
        let peaks =
            extract_peak_rows_from_c_scan(space, anchor_hz, &scan, E4_DIAG_PEAK_TOP_K.max(1));
        let n = peaks0.len().min(peaks.len()).min(E4_DIAG_PEAK_TOP_K);
        let mut shift_sum = 0.0f32;
        for i in 0..n {
            shift_sum += (peaks0[i].semitones - peaks[i].semitones).abs();
        }
        let shift = if n == 0 { 0.0 } else { shift_sum / n as f32 };
        out.push(E4LandscapeDeltaRow {
            mirror_weight,
            cosine_similarity: cosine_similarity(&scan0.c, &scan.c),
            l1_distance: l1_mean_distance(&scan0.c, &scan.c),
            topk_peak_shift_st: shift,
        });
    }
    out.sort_by(|a, b| {
        a.mirror_weight
            .partial_cmp(&b.mirror_weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

fn e4_landscape_delta_by_mw_csv(rows: &[E4LandscapeDeltaRow]) -> String {
    let mut out = String::from("mw,cosine_similarity,l1_distance,topk_peak_shift_st,environment\n");
    for row in rows {
        out.push_str(&format!(
            "{:.3},{:.6},{:.6},{:.6},anchor_only\n",
            row.mirror_weight, row.cosine_similarity, row.l1_distance, row.topk_peak_shift_st
        ));
    }
    out
}

fn e4_quantile_points_by_step(
    rows: &[E4DiagStepRow],
    use_global: bool,
) -> std::collections::HashMap<(i32, u32), (f32, f32, f32)> {
    let mut grouped: std::collections::HashMap<(i32, u32), Vec<f32>> =
        std::collections::HashMap::new();
    for row in rows {
        let v = if use_global {
            row.gap_global
        } else {
            row.gap_reach
        };
        grouped
            .entry((float_key(row.mirror_weight), row.step))
            .or_default()
            .push(v);
    }
    let mut out = std::collections::HashMap::new();
    for (key, mut vals) in grouped {
        if vals.is_empty() {
            continue;
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = vals.len();
        let q = |p: f32| -> f32 {
            let idx = (p * (n.saturating_sub(1) as f32)).round() as usize;
            vals[idx.min(n - 1)]
        };
        out.insert(key, (q(0.25), q(0.50), q(0.75)));
    }
    out
}

fn render_e4_gap_over_time(
    out_path: &Path,
    rows: &[E4DiagStepRow],
    use_global: bool,
) -> Result<(), Box<dyn Error>> {
    if rows.is_empty() {
        return Ok(());
    }
    let quant = e4_quantile_points_by_step(rows, use_global);
    if quant.is_empty() {
        return Ok(());
    }
    let mut mw_keys: Vec<i32> = rows.iter().map(|r| float_key(r.mirror_weight)).collect();
    mw_keys.sort();
    mw_keys.dedup();
    let max_step = rows.iter().map(|r| r.step).max().unwrap_or(0);
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for &(q25, _q50, q75) in quant.values() {
        y_min = y_min.min(q25);
        y_max = y_max.max(q75);
    }
    if !y_min.is_finite() || !y_max.is_finite() || (y_max - y_min).abs() < 1e-9 {
        y_min = -0.1;
        y_max = 0.1;
    }
    let pad = 0.12 * (y_max - y_min).max(1e-3);
    let root = bitmap_root(out_path, (1500, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            if use_global {
                "E4 gap_global over time (median + IQR)"
            } else {
                "E4 gap_reach over time (median + IQR)"
            },
            ("sans-serif", 24),
        )
        .margin(16)
        .x_label_area_size(42)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..max_step as f32, (y_min - pad)..(y_max + pad))?;
    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc(if use_global {
            "gap_global"
        } else {
            "gap_reach"
        })
        .draw()?;
    for (i, mw_key) in mw_keys.iter().enumerate() {
        let mut pts: Vec<(u32, f32, f32, f32)> = quant
            .iter()
            .filter(|((k, _), _)| *k == *mw_key)
            .map(|((_, step), (q25, q50, q75))| (*step, *q25, *q50, *q75))
            .collect();
        pts.sort_by_key(|(step, _, _, _)| *step);
        if pts.is_empty() {
            continue;
        }
        let color = Palette99::pick(i);
        let upper: Vec<(f32, f32)> = pts.iter().map(|(s, _, _, q75)| (*s as f32, *q75)).collect();
        let lower: Vec<(f32, f32)> = pts.iter().map(|(s, q25, _, _)| (*s as f32, *q25)).collect();
        let mut band = upper.clone();
        for p in lower.iter().rev() {
            band.push(*p);
        }
        chart.draw_series(std::iter::once(Polygon::new(
            band,
            color.mix(0.12).filled(),
        )))?;
        chart
            .draw_series(LineSeries::new(
                pts.iter().map(|(s, _, q50, _)| (*s as f32, *q50)),
                color.stroke_width(2),
            ))?
            .label(format!("mw={:.2}", float_from_key(*mw_key)))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
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

fn render_e4_gap_global_by_mw(
    out_path: &Path,
    rows: &[E4DiagStepRow],
) -> Result<(), Box<dyn Error>> {
    if rows.is_empty() {
        return Ok(());
    }
    let max_step = rows.iter().map(|r| r.step).max().unwrap_or(0);
    let tail_start = max_step.saturating_sub((max_step / 5).max(5));
    let mut run_vals: std::collections::HashMap<(i32, u64), Vec<f32>> =
        std::collections::HashMap::new();
    for row in rows {
        if row.step < tail_start {
            continue;
        }
        run_vals
            .entry((float_key(row.mirror_weight), row.seed))
            .or_default()
            .push(row.gap_global);
    }
    let mut by_mw: std::collections::HashMap<i32, Vec<f32>> = std::collections::HashMap::new();
    for ((mw_key, _seed), vals) in run_vals {
        if vals.is_empty() {
            continue;
        }
        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
        by_mw.entry(mw_key).or_default().push(mean);
    }
    if by_mw.is_empty() {
        return Ok(());
    }
    let mut points: Vec<(f32, f32, f32, f32)> = Vec::new();
    for (mw_key, vals) in by_mw {
        let seed = E4_BOOTSTRAP_SEED ^ (mw_key as i64 as u64).wrapping_mul(0x9E37_79B9);
        let (mean, lo, hi) = bootstrap_mean_ci95(&vals, E4_BOOTSTRAP_ITERS, seed);
        points.push((float_from_key(mw_key), mean, lo, hi));
    }
    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut y_min = points.iter().map(|p| p.2).fold(f32::INFINITY, f32::min);
    let mut y_max = points.iter().map(|p| p.3).fold(f32::NEG_INFINITY, f32::max);
    if !y_min.is_finite() || !y_max.is_finite() || (y_max - y_min).abs() < 1e-9 {
        y_min = -0.1;
        y_max = 0.1;
    }
    let pad = 0.15 * (y_max - y_min).max(1e-3);
    let root = bitmap_root(out_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "E4 gap_global by mirror_weight (tail mean +/- CI)",
            ("sans-serif", 24),
        )
        .margin(16)
        .x_label_area_size(42)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, (y_min - pad)..(y_max + pad))?;
    chart
        .configure_mesh()
        .x_desc("mirror_weight")
        .y_desc("gap_global")
        .draw()?;
    chart.draw_series(LineSeries::new(
        points.iter().map(|(mw, mean, _lo, _hi)| (*mw, *mean)),
        BLUE.stroke_width(2),
    ))?;
    chart.draw_series(points.iter().map(|(mw, _mean, lo, hi)| {
        PathElement::new(vec![(*mw, *lo), (*mw, *hi)], BLUE.mix(0.6).stroke_width(1))
    }))?;
    chart.draw_series(
        points
            .iter()
            .map(|(mw, mean, _lo, _hi)| Circle::new((*mw, *mean), 3, BLUE.filled())),
    )?;
    root.present()?;
    Ok(())
}

fn render_e4_peak_positions_vs_mw(
    out_path: &Path,
    rows: &[E4DiagPeakRow],
    top_k: usize,
) -> Result<(), Box<dyn Error>> {
    if rows.is_empty() || top_k == 0 {
        return Ok(());
    }
    let mut grouped: std::collections::HashMap<(i32, usize), Vec<f32>> =
        std::collections::HashMap::new();
    for row in rows {
        if row.peak_rank == 0 || row.peak_rank > top_k {
            continue;
        }
        grouped
            .entry((float_key(row.mirror_weight), row.peak_rank))
            .or_default()
            .push(row.peak_semitones);
    }
    if grouped.is_empty() {
        return Ok(());
    }
    let mut mw_keys: Vec<i32> = grouped.keys().map(|(mw_key, _)| *mw_key).collect();
    mw_keys.sort();
    mw_keys.dedup();
    let mut rank_series: std::collections::HashMap<usize, Vec<(f32, f32)>> =
        std::collections::HashMap::new();
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for rank in 1..=top_k {
        let mut pts = Vec::new();
        for mw_key in &mw_keys {
            let Some(vals) = grouped.get(&(*mw_key, rank)) else {
                continue;
            };
            let mut vals = vals.clone();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med = vals[vals.len() / 2];
            pts.push((float_from_key(*mw_key), med));
            y_min = y_min.min(med);
            y_max = y_max.max(med);
        }
        if !pts.is_empty() {
            pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            rank_series.insert(rank, pts);
        }
    }
    if rank_series.is_empty() {
        return Ok(());
    }
    if !y_min.is_finite() || !y_max.is_finite() || (y_max - y_min).abs() < 1e-6 {
        y_min = -12.0;
        y_max = 12.0;
    }
    let pad = 0.10 * (y_max - y_min).max(1e-3);
    let root = bitmap_root(out_path, (1400, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "E4 peak positions vs mirror_weight (median across seeds)",
            ("sans-serif", 24),
        )
        .margin(16)
        .x_label_area_size(42)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, (y_min - pad)..(y_max + pad))?;
    chart
        .configure_mesh()
        .x_desc("mirror_weight")
        .y_desc("peak position (semitones)")
        .draw()?;
    for rank in 1..=top_k {
        let Some(pts) = rank_series.get(&rank) else {
            continue;
        };
        let color = Palette99::pick(rank - 1);
        chart
            .draw_series(LineSeries::new(pts.iter().copied(), color.stroke_width(2)))?
            .label(format!("rank {rank}"))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
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

fn render_e4_landscape_components_overlay(
    out_path: &Path,
    scans: &[E4LandscapeScans],
    wr: f32,
) -> Result<(), Box<dyn Error>> {
    let wr_key = float_key(wr);
    let mut rows: Vec<&E4LandscapeScans> =
        scans.iter().filter(|s| float_key(s.wr) == wr_key).collect();
    rows.sort_by(|a, b| {
        a.mirror_weight
            .partial_cmp(&b.mirror_weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if rows.is_empty() {
        return Ok(());
    }
    let root = bitmap_root(out_path, (1500, 1000)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((3, 1));
    let x_min = rows[0].semitones.first().copied().unwrap_or(-12.0);
    let x_max = rows[0].semitones.last().copied().unwrap_or(12.0);

    let mut y_c_min = f32::INFINITY;
    let mut y_c_max = f32::NEG_INFINITY;
    let mut y_h_min = f32::INFINITY;
    let mut y_h_max = f32::NEG_INFINITY;
    let mut y_r_min = f32::INFINITY;
    let mut y_r_max = f32::NEG_INFINITY;
    for row in &rows {
        for &v in &row.c {
            y_c_min = y_c_min.min(v);
            y_c_max = y_c_max.max(v);
        }
        for &v in &row.h {
            y_h_min = y_h_min.min(v);
            y_h_max = y_h_max.max(v);
        }
        for &v in &row.r {
            y_r_min = y_r_min.min(v);
            y_r_max = y_r_max.max(v);
        }
    }
    if !y_c_min.is_finite() || !y_c_max.is_finite() || (y_c_max - y_c_min).abs() < 1e-6 {
        y_c_min = 0.0;
        y_c_max = 1.0;
    }
    if !y_h_min.is_finite() || !y_h_max.is_finite() || (y_h_max - y_h_min).abs() < 1e-6 {
        y_h_min = 0.0;
        y_h_max = 1.0;
    }
    if !y_r_min.is_finite() || !y_r_max.is_finite() || (y_r_max - y_r_min).abs() < 1e-6 {
        y_r_min = 0.0;
        y_r_max = 1.0;
    }

    let mut chart_h = ChartBuilder::on(&panels[0])
        .caption(
            format!("E4 landscape components (w_r={wr:.2}): H"),
            ("sans-serif", 18),
        )
        .margin(8)
        .x_label_area_size(34)
        .y_label_area_size(52)
        .build_cartesian_2d(x_min..x_max, y_h_min..y_h_max)?;
    chart_h
        .configure_mesh()
        .x_desc("semitones")
        .y_desc("H (state01)")
        .draw()?;
    for (i, row) in rows.iter().enumerate() {
        let color = Palette99::pick(i).mix(0.90);
        chart_h
            .draw_series(LineSeries::new(
                row.semitones.iter().copied().zip(row.h.iter().copied()),
                color.stroke_width(2),
            ))?
            .label(format!("mw={:.2} H", row.mirror_weight))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 22, y)], color.stroke_width(2))
            });
    }
    if let Some(row_mid) = rows.iter().find(|s| (s.mirror_weight - 0.5).abs() < 1e-6) {
        chart_h
            .draw_series(LineSeries::new(
                row_mid
                    .semitones
                    .iter()
                    .copied()
                    .zip(row_mid.h_lower.iter().copied()),
                BLACK.mix(0.55).stroke_width(1),
            ))?
            .label("H_lower (mw=0.5)")
            .legend(|(x, y)| {
                PathElement::new(vec![(x, y), (x + 22, y)], BLACK.mix(0.55).stroke_width(1))
            });
        chart_h
            .draw_series(LineSeries::new(
                row_mid
                    .semitones
                    .iter()
                    .copied()
                    .zip(row_mid.h_upper.iter().copied()),
                BLACK.mix(0.25).stroke_width(1),
            ))?
            .label("H_upper (mw=0.5)")
            .legend(|(x, y)| {
                PathElement::new(vec![(x, y), (x + 22, y)], BLACK.mix(0.25).stroke_width(1))
            });
    }
    chart_h
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK.mix(0.4))
        .draw()?;

    let mut chart_r = ChartBuilder::on(&panels[1])
        .caption("R", ("sans-serif", 18))
        .margin(8)
        .x_label_area_size(34)
        .y_label_area_size(52)
        .build_cartesian_2d(x_min..x_max, y_r_min..y_r_max)?;
    chart_r
        .configure_mesh()
        .x_desc("semitones")
        .y_desc("R (state01)")
        .draw()?;
    for (i, row) in rows.iter().enumerate() {
        let color = Palette99::pick(i).mix(0.90);
        chart_r
            .draw_series(LineSeries::new(
                row.semitones.iter().copied().zip(row.r.iter().copied()),
                color.stroke_width(2),
            ))?
            .label(format!("mw={:.2}", row.mirror_weight))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 22, y)], color.stroke_width(2))
            });
    }
    chart_r
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK.mix(0.4))
        .draw()?;

    let mut chart_c = ChartBuilder::on(&panels[2])
        .caption("C", ("sans-serif", 18))
        .margin(8)
        .x_label_area_size(34)
        .y_label_area_size(52)
        .build_cartesian_2d(x_min..x_max, y_c_min..y_c_max)?;
    chart_c
        .configure_mesh()
        .x_desc("semitones")
        .y_desc("C (state01)")
        .draw()?;
    for (i, row) in rows.iter().enumerate() {
        let color = Palette99::pick(i).mix(0.90);
        chart_c
            .draw_series(LineSeries::new(
                row.semitones.iter().copied().zip(row.c.iter().copied()),
                color.stroke_width(2),
            ))?
            .label(format!("mw={:.2}", row.mirror_weight))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 22, y)], color.stroke_width(2))
            });
    }
    chart_c
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK.mix(0.4))
        .draw()?;

    root.present()?;
    Ok(())
}

fn render_e4_wr_agent_vs_oracle_plot(
    out_path: &Path,
    rows: &[E4WrOracleRow],
) -> Result<(), Box<dyn Error>> {
    if rows.is_empty() {
        return Ok(());
    }
    let mut wr_keys: Vec<i32> = rows.iter().map(|r| float_key(r.wr)).collect();
    wr_keys.sort();
    wr_keys.dedup();
    let cols = 3usize.min(wr_keys.len().max(1));
    let rows_n = wr_keys.len().div_ceil(cols);
    let root = bitmap_root(out_path, (1700, 420 * rows_n as u32)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((rows_n, cols));
    for (i, wr_key) in wr_keys.iter().enumerate() {
        if i >= panels.len() {
            break;
        }
        let wr = float_from_key(*wr_key);
        let mut group: Vec<&E4WrOracleRow> =
            rows.iter().filter(|r| float_key(r.wr) == *wr_key).collect();
        group.sort_by(|a, b| {
            a.mirror_weight
                .partial_cmp(&b.mirror_weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if group.is_empty() {
            continue;
        }
        let mut y_min = f32::INFINITY;
        let mut y_max = f32::NEG_INFINITY;
        for row in &group {
            y_min = y_min
                .min(row.agent_delta_bind)
                .min(row.oracle1_delta_bind_ci_lo)
                .min(row.oracle2_delta_bind_ci_lo);
            y_max = y_max
                .max(row.agent_delta_bind)
                .max(row.oracle1_delta_bind_ci_hi)
                .max(row.oracle2_delta_bind_ci_hi);
        }
        if !y_min.is_finite() || !y_max.is_finite() || (y_max - y_min).abs() < 1e-6 {
            y_min = -1.0;
            y_max = 1.0;
        }
        let pad = 0.12 * (y_max - y_min);
        let mut chart = ChartBuilder::on(&panels[i])
            .caption(format!("w_r={wr:.2}"), ("sans-serif", 18))
            .margin(8)
            .x_label_area_size(36)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0f32..1.0f32, (y_min - pad)..(y_max + pad))?;
        chart
            .configure_mesh()
            .x_desc("mirror_weight")
            .y_desc("DeltaBind")
            .draw()?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(0.0, 0.0), (1.0, 0.0)],
            BLACK.mix(0.5).stroke_width(1),
        )))?;
        chart
            .draw_series(LineSeries::new(
                group.iter().map(|r| (r.mirror_weight, r.agent_delta_bind)),
                BLUE.mix(0.95).stroke_width(2),
            ))?
            .label("agent")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], BLUE.stroke_width(2)));
        chart
            .draw_series(LineSeries::new(
                group
                    .iter()
                    .map(|r| (r.mirror_weight, r.oracle1_delta_bind)),
                RED.mix(0.90).stroke_width(2),
            ))?
            .label("oracle1")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], RED.stroke_width(2)));
        chart.draw_series(group.iter().map(|r| {
            PathElement::new(
                vec![
                    (r.mirror_weight, r.oracle1_delta_bind_ci_lo),
                    (r.mirror_weight, r.oracle1_delta_bind_ci_hi),
                ],
                RED.mix(0.45).stroke_width(1),
            )
        }))?;
        chart
            .draw_series(LineSeries::new(
                group
                    .iter()
                    .map(|r| (r.mirror_weight, r.oracle2_delta_bind_mean)),
                GREEN.mix(0.85).stroke_width(2),
            ))?
            .label("oracle2")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], GREEN.stroke_width(2)));
        chart.draw_series(group.iter().map(|r| {
            PathElement::new(
                vec![
                    (r.mirror_weight, r.oracle2_delta_bind_ci_lo),
                    (r.mirror_weight, r.oracle2_delta_bind_ci_hi),
                ],
                GREEN.mix(0.5).stroke_width(1),
            )
        }))?;
        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK.mix(0.4))
            .draw()?;
    }
    root.present()?;
    Ok(())
}

fn render_e4_wr_representative_histograms(
    out_dir: &Path,
    anchor_hz: f32,
    rows: &[E4WrTailAgentRow],
) -> Result<(), Box<dyn Error>> {
    if rows.is_empty() {
        return Ok(());
    }
    let bin_width_cents = E4_PAPER_HIST_BIN_CENTS.max(1.0);
    for wr in E4_WR_GRID {
        let wr_key = float_key(normalize_wr(wr));
        for mirror_weight in E4_WR_MIRROR_WEIGHTS {
            let mw = mirror_weight.clamp(0.0, 1.0);
            let mw_key = float_key(mw);
            let mut cents_samples = Vec::new();
            for row in rows {
                if float_key(row.wr) != wr_key || float_key(row.mirror_weight) != mw_key {
                    continue;
                }
                if let Some(cents) = freq_to_cents_class(anchor_hz, row.freq_hz) {
                    cents_samples.push(cents);
                }
            }
            if cents_samples.is_empty() {
                continue;
            }
            let wr_token = format_float_token(wr);
            let mw_token = format_float_token(mw);
            let out_path = out_dir.join(format!(
                "paper_e4_wr{wr_token}_mw{mw_token}_interval_histogram.svg"
            ));
            let caption = format!(
                "E4 interval histogram (w_r={wr:.2}, mirror_weight={mw:.2}, final step pooled seeds)"
            );
            render_interval_histogram(
                &out_path,
                &caption,
                &cents_samples,
                0.0,
                1200.0,
                bin_width_cents,
                "cents",
            )?;
        }
    }
    Ok(())
}

fn e4_wr_units_meta_text() -> String {
    let mut out = String::new();
    out.push_str("E4 wr-probe units / definitions\n");
    out.push_str("- interval histogram domain: cents pitch-class in [0, 1200)\n");
    out.push_str("- histogram x-axis unit: cents\n");
    out.push_str("- fingerprint source interval: pairwise |1200*log2(fi/fj)| mod 1200 (cents)\n");
    out.push_str(&format!(
        "- fingerprint tolerance: {:.3} cents\n",
        E4_FINGERPRINT_TOL_CENTS
    ));
    out.push_str("- fingerprint folding: abs_mod_1200\n");
    out.push_str("- wr summary error_kind: bootstrap_pctl95 (non-parametric percentile CI)\n");
    out.push_str("- landscape components: H_lower/H_upper/H/R/C are state01 on Log2 grid\n");
    out.push_str("- oracle1: greedy top-C peaks (K=population)\n");
    out.push_str("- oracle2: weighted sampling from top-C peaks\n");
    out
}

fn e4_validation_markdown() -> String {
    [
        "# E4 validation checklist",
        "",
        "## 1) Landscape changed?",
        "- Compare `paper_e4_landscape_components_mw*_wr*.csv`.",
        "- If peak positions/ranks of `C` and `H` shift with `mirror_weight`, landscape is changing.",
        "- Check `H_upper` magnitude at high `mirror_weight` for scale collapse.",
        "",
        "## 2) Oracle switches but agent does not?",
        "- Compare `paper_e4_wr_oracle_vs_agent.csv` or `paper_e4_wr_delta_bind_agent_vs_oracle.svg`.",
        "- If oracle curves switch sign/level but agent curve stays flat, dynamics/search is bottleneck.",
        "- If oracle also does not switch, model-side landscape definition is likely bottleneck.",
        "",
        "## 2b) Dynamics ablation (lambda / candidate vocabulary / update rule)",
        "- Inspect `paper_e4_wr_dynamics_probe_timeseries.csv` and `paper_e4_wr_dynamics_probe_summary.csv`.",
        "- Modes: `baseline_seq`, `lambda0_seq`, `peak_seq`, `peak_sync`.",
        "- If `lambda0_seq` tracks oracle better, repulsion is likely suppressing binding bias.",
        "- If `peak_seq`/`peak_sync` improve while baseline does not, candidate vocabulary or update schedule is the bottleneck.",
        "",
        "## 3) Anchor-fixed bias in metric?",
        "- In `paper_e4_wr_sweep_runs.csv` and `paper_e4_wr_sweep_summary.csv`, compare:",
        "  - set-estimated: `root_fit, ceiling_fit, delta_bind`",
        "  - anchor-fixed: `root_fit_anchor, ceiling_fit_anchor, delta_bind_anchor`",
        "- If only anchor-fixed is weak/flat, metric bias is likely.",
        "",
        "## Next actions",
        "- Model-side issue: inspect H_lower/H_upper blending and C = H - w_r R scaling.",
        "- Search-side issue: inspect peaklist diversity and peak sampler candidate policy.",
        "- Metric-side issue: prioritize set-estimated binding metric for regime claims.",
        "",
        "## Metric semantics",
        "- A/B/C/D are alignment diagnostics (agent-vs-oracle consistency), not harmonic regime indicators.",
        "- E4 regime claims should rely on RootFit/CeilingFit/DeltaBind and interval fingerprint summaries with CI.",
        "",
        "## Timing diagnosis (ABCD trace)",
        "- Generate trace: `cargo run --example paper -- --exp e4 --e4-wr on`",
        "- Analyze lag: `cargo run --example paper -- --e4-abcd-analyze --input examples/paper/plots/e4/paper_e4_abcd_trace.csv --outdir examples/paper/plots/e4`",
        "- Inspect: `e4_abcd_summary_by_mirror.csv`, `e4_abcd_summary_overall.md` and 3 PNG files.",
    ]
    .join("\n")
}

fn plot_e4_mirror_sweep_wr_cut(out_dir: &Path, anchor_hz: f32) -> Result<(), Box<dyn Error>> {
    let mirror_weights: Vec<f32> = E4_WR_MIRROR_WEIGHTS.to_vec();
    let tail_rows = e4_collect_wr_tail_agent_rows(&mirror_weights);
    write_with_log(
        out_dir.join("paper_e4_wr_units_meta.txt"),
        e4_wr_units_meta_text(),
    )?;
    write_with_log(
        out_dir.join("paper_e4_wr_tail_agents.csv"),
        e4_wr_tail_agents_csv(&tail_rows),
    )?;
    write_with_log(out_dir.join("VALIDATION.md"), e4_validation_markdown())?;

    let bind_runs = e4_wr_bind_runs_from_tail_agents(&tail_rows, anchor_hz);
    write_with_log(
        out_dir.join("paper_e4_wr_sweep_runs.csv"),
        e4_wr_bind_runs_csv(&bind_runs),
    )?;
    let bind_summary = e4_wr_bind_summary_rows(&bind_runs);

    let fp_runs = e4_wr_fingerprint_runs_from_tail_agents(&tail_rows);
    write_with_log(
        out_dir.join("paper_e4_wr_fingerprint_raw.csv"),
        e4_wr_fingerprint_runs_csv(&fp_runs),
    )?;
    write_with_log(
        out_dir.join("paper_e4_wr_fingerprint_runs_wide.csv"),
        e4_wr_fingerprint_runs_wide_csv(&fp_runs),
    )?;
    let fp_summary = e4_wr_fingerprint_summary_rows(&fp_runs);
    write_with_log(
        out_dir.join("paper_e4_wr_fingerprint_summary.csv"),
        e4_wr_fingerprint_summary_csv(&fp_summary),
    )?;

    write_with_log(
        out_dir.join("paper_e4_wr_sweep_summary.csv"),
        e4_wr_sweep_summary_csv(&bind_summary, &fp_summary),
    )?;

    let meta = e4_paper_meta();
    let space = Log2Space::new(meta.fmin, meta.fmax, meta.bins_per_oct);
    let (_erb_scan, du_scan) = erb_grid_for_space(&space);
    let grouped_freqs = grouped_freqs_by_wr_mw(&tail_rows);
    let mut landscape_scans = Vec::new();
    for &wr in &E4_WR_GRID {
        for &mw in &mirror_weights {
            let key = (float_key(normalize_wr(wr)), float_key(mw));
            let Some(freqs) = grouped_freqs.get(&key) else {
                continue;
            };
            let scan = compute_e4_landscape_scans(&space, anchor_hz, wr, mw, freqs, &du_scan);
            let wr_token = format_float_token(wr);
            let mw_token = format_float_token(mw);
            write_with_log(
                out_dir.join(format!(
                    "paper_e4_landscape_components_mw{mw_token}_wr{wr_token}.csv"
                )),
                e4_landscape_components_csv(&scan),
            )?;
            let peaks =
                extract_peak_rows_from_c_scan(&space, anchor_hz, &scan, E4_WR_PEAKLIST_TOP_N);
            write_with_log(
                out_dir.join(format!("paper_e4_peaklist_mw{mw_token}_wr{wr_token}.csv")),
                e4_peaklist_csv(scan.wr, scan.mirror_weight, &peaks),
            )?;
            if (wr - 1.0).abs() < 1e-6 {
                write_with_log(
                    out_dir.join(format!("paper_e4_peaklist_mw{mw_token}.csv")),
                    e4_peaklist_csv(scan.wr, scan.mirror_weight, &peaks),
                )?;
            }
            landscape_scans.push(scan);
        }
    }

    for &wr in &E4_WR_GRID {
        let wr_token = format_float_token(wr);
        let out_path = out_dir.join(format!(
            "paper_e4_landscape_components_overlay_wr{wr_token}.svg"
        ));
        render_e4_landscape_components_overlay(&out_path, &landscape_scans, wr)?;
    }
    render_e4_landscape_components_overlay(
        &out_dir.join("paper_e4_landscape_components_overlay.svg"),
        &landscape_scans,
        1.0,
    )?;

    let grouped_freqs_seed = grouped_freqs_by_wr_mw_seed(&tail_rows);
    let abcd_trace_rows = e4_abcd_trace_rows(
        &grouped_freqs_seed,
        &space,
        anchor_hz,
        &du_scan,
        E4_ABCD_TRACE_STEPS,
    );
    let abcd_trace_csv = e4_abcd_trace_csv(&abcd_trace_rows);
    write_with_log(out_dir.join("paper_e4_abcd_trace.csv"), &abcd_trace_csv)?;

    let oracle_run_rows = e4_wr_oracle_run_rows(
        &bind_runs,
        &grouped_freqs_seed,
        &space,
        anchor_hz,
        &du_scan,
        meta.voice_count,
    );
    write_with_log(
        out_dir.join("paper_e4_wr_oracle_vs_agent_runs.csv"),
        e4_wr_oracle_runs_csv(&oracle_run_rows),
    )?;
    let oracle_rows = e4_wr_oracle_rows_from_runs(&oracle_run_rows);
    write_with_log(
        out_dir.join("paper_e4_wr_oracle_vs_agent.csv"),
        e4_wr_oracle_csv(&oracle_rows),
    )?;
    render_e4_wr_agent_vs_oracle_plot(
        &out_dir.join("paper_e4_wr_delta_bind_agent_vs_oracle.svg"),
        &oracle_rows,
    )?;

    let dynamics_probe_rows =
        e4_wr_dynamics_probe_rows(&grouped_freqs_seed, &space, anchor_hz, &du_scan);
    write_with_log(
        out_dir.join("paper_e4_wr_dynamics_probe_timeseries.csv"),
        e4_wr_dynamics_probe_timeseries_csv(&dynamics_probe_rows),
    )?;
    let dynamics_probe_summary = e4_wr_dynamics_probe_summary_rows(&dynamics_probe_rows);
    write_with_log(
        out_dir.join("paper_e4_wr_dynamics_probe_summary.csv"),
        e4_wr_dynamics_probe_summary_csv(&dynamics_probe_summary),
    )?;

    render_e4_wr_delta_bind_vs_mirror(
        &out_dir.join("paper_e4_wr_delta_bind_vs_mirror.svg"),
        &bind_summary,
    )?;
    render_e4_wr_root_ceiling_vs_mirror(
        &out_dir.join("paper_e4_wr_root_ceiling_vs_mirror.svg"),
        &bind_summary,
        &E4_WR_GRID,
    )?;
    render_e4_wr_fingerprint_heatmap(
        &out_dir.join("paper_e4_wr_fingerprint_heatmap.svg"),
        &fp_summary,
        &E4_WR_FINGERPRINT_FOCUS,
        &mirror_weights,
    )?;
    render_e4_wr_representative_histograms(out_dir, anchor_hz, &tail_rows)?;
    Ok(())
}

fn e4_summary_csv(records: &[E4SummaryRecord]) -> String {
    let mut out = String::from(
        "count_mode,mirror_weight,bin_width,eps_cents,mean_major,std_major,mean_minor,std_minor,mean_delta_t,std_delta_t,mean_m3,std_m3,mean_M3,std_M3,mean_P4,std_P4,mean_P5,std_P5,mean_P5class,std_P5class,major_rate,minor_rate,ambiguous_rate,n_runs\n",
    );
    for record in records {
        out.push_str(&format!(
            "{},{:.3},{:.3},{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3},{:.3},{:.3},{}\n",
            record.count_mode,
            record.mirror_weight,
            record.bin_width,
            record.eps_cents,
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
            record.mean_p5_class,
            record.std_p5_class,
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
    let bin_width_cents = (bin_width * 100.0).max(1.0);
    let mut out = String::from(
        "mirror_weight,seed,bin_width_st,bin_width_cents,steps_total,burn_in,tail_window,histogram_source,interval_unit,bin_center_cents,mass\n",
    );
    for (center, mass) in hist.bin_centers.iter().zip(hist.masses.iter()) {
        out.push_str(&format!(
            "{:.3},{},{:.3},{:.3},{},{},{},{},cents,{:.3},{:.6}\n",
            weight,
            seed,
            bin_width,
            bin_width_cents,
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
    let mut map: std::collections::HashMap<(&'static str, i32, i32, i32), Vec<&E4RunRecord>> =
        std::collections::HashMap::new();
    for record in records {
        let key = (
            record.count_mode,
            float_key(record.mirror_weight),
            float_key(record.bin_width),
            float_key(record.eps_cents),
        );
        map.entry(key).or_default().push(record);
    }
    let mut summaries = Vec::new();
    for ((_mode_key, _w_key, _bw_key, _eps_key), runs) in map {
        if runs.is_empty() {
            continue;
        }
        let count_mode = runs[0].count_mode;
        let mirror_weight = runs[0].mirror_weight;
        let bin_width = runs[0].bin_width;
        let eps_cents = runs[0].eps_cents;
        let major_values: Vec<f32> = runs.iter().map(|r| r.major_score).collect();
        let minor_values: Vec<f32> = runs.iter().map(|r| r.minor_score).collect();
        let delta_values: Vec<f32> = runs.iter().map(|r| r.delta).collect();
        let min3_values: Vec<f32> = runs.iter().map(|r| r.mass_min3).collect();
        let maj3_values: Vec<f32> = runs.iter().map(|r| r.mass_maj3).collect();
        let p4_values: Vec<f32> = runs.iter().map(|r| r.mass_p4).collect();
        let p5_values: Vec<f32> = runs.iter().map(|r| r.mass_p5).collect();
        let p5_class_values: Vec<f32> = runs.iter().map(|r| r.mass_p5_class).collect();
        let (mean_major, std_major) = mean_std(&major_values);
        let (mean_minor, std_minor) = mean_std(&minor_values);
        let (mean_delta, std_delta) = mean_std(&delta_values);
        let (mean_min3, std_min3) = mean_std(&min3_values);
        let (mean_maj3, std_maj3) = mean_std(&maj3_values);
        let (mean_p4, std_p4) = mean_std(&p4_values);
        let (mean_p5, std_p5) = mean_std(&p5_values);
        let (mean_p5_class, std_p5_class) = mean_std(&p5_class_values);
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
            count_mode,
            mirror_weight,
            bin_width,
            eps_cents,
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
            mean_p5_class,
            std_p5_class,
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
                a.eps_cents
                    .partial_cmp(&b.eps_cents)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                a.mirror_weight
                    .partial_cmp(&b.mirror_weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.count_mode.cmp(b.count_mode))
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

fn mean_std_sample(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
    if values.len() < 2 {
        return (mean, 0.0);
    }
    let var = values.iter().map(|v| (*v - mean).powi(2)).sum::<f32>() / (values.len() - 1) as f32;
    (mean, var.sqrt())
}

fn std_to_se(std: f32, n: usize) -> f32 {
    if n == 0 { 0.0 } else { std / (n as f32).sqrt() }
}

struct LinearFit {
    slope: f32,
    intercept: f32,
    r2: f32,
    se_slope: f32,
}

fn linear_regression(x: &[f32], y: &[f32]) -> Option<LinearFit> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let n = x.len() as f32;
    let mean_x = x.iter().copied().sum::<f32>() / n;
    let mean_y = y.iter().copied().sum::<f32>() / n;
    let mut sxx = 0.0f32;
    let mut sxy = 0.0f32;
    let mut sst = 0.0f32;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        sxx += dx * dx;
        sxy += dx * dy;
        sst += dy * dy;
    }
    if sxx <= 0.0 {
        return None;
    }
    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    let mut sse = 0.0f32;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let resid = yi - (intercept + slope * xi);
        sse += resid * resid;
    }
    let r2 = if sst > 0.0 { 1.0 - sse / sst } else { 0.0 };
    let se_slope = if x.len() > 2 {
        let mse = sse / (x.len() as f32 - 2.0);
        (mse / sxx).sqrt()
    } else {
        0.0
    };
    Some(LinearFit {
        slope,
        intercept,
        r2,
        se_slope,
    })
}

fn t_crit_975(df: usize) -> f32 {
    match df {
        1 => 12.706,
        2 => 4.303,
        3 => 3.182,
        4 => 2.776,
        5 => 2.571,
        6 => 2.447,
        7 => 2.365,
        8 => 2.306,
        9 => 2.262,
        10 => 2.228,
        11 => 2.201,
        12 => 2.179,
        13 => 2.160,
        14 => 2.145,
        15 => 2.131,
        16 => 2.120,
        17 => 2.110,
        18 => 2.101,
        19 => 2.093,
        20 => 2.086,
        21 => 2.080,
        22 => 2.074,
        23 => 2.069,
        24 => 2.064,
        25 => 2.060,
        26 => 2.056,
        27 => 2.052,
        28 => 2.048,
        29 => 2.045,
        30 => 2.042,
        // df > 30: normal approximation
        _ => 1.96,
    }
}

fn binomial_coeff(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut c = 1.0f64;
    for i in 1..=k {
        let num = (n - k + i) as f64;
        c *= num / i as f64;
    }
    c
}

fn binomial_two_sided_p(k: usize, n: usize) -> f32 {
    if n == 0 {
        return 1.0;
    }
    let k = k.min(n - k);
    let mut sum = 0.0f64;
    let denom = 0.5f64.powi(n as i32);
    for i in 0..=k {
        sum += binomial_coeff(n, i) * denom;
    }
    let p = (2.0 * sum).min(1.0);
    p as f32
}

fn e4_delta_effects_from_summary(summaries: &[E4SummaryRecord]) -> Vec<E4DeltaEffectRow> {
    let mut rows = Vec::new();
    for summary in summaries {
        let se_delta = std_to_se(summary.std_delta, summary.n_runs);
        let ci_half = if summary.n_runs < 2 {
            0.0
        } else {
            let t_crit = t_crit_975(summary.n_runs.saturating_sub(1));
            t_crit * se_delta
        };
        rows.push(E4DeltaEffectRow {
            count_mode: summary.count_mode,
            mirror_weight: summary.mirror_weight,
            bin_width: summary.bin_width,
            eps_cents: summary.eps_cents,
            mean_delta: summary.mean_delta,
            sd_delta: summary.std_delta,
            se_delta,
            ci_lo: summary.mean_delta - ci_half,
            ci_hi: summary.mean_delta + ci_half,
            n_seeds: summary.n_runs,
            mean_mass_min3: summary.mean_min3,
            sd_mass_min3: summary.std_min3,
            mean_mass_maj3: summary.mean_maj3,
            sd_mass_maj3: summary.std_maj3,
            mean_mass_p5: summary.mean_p5,
            sd_mass_p5: summary.std_p5,
            mean_mass_p5_class: summary.mean_p5_class,
            sd_mass_p5_class: summary.std_p5_class,
        });
    }
    rows.sort_by(|a, b| {
        a.bin_width
            .partial_cmp(&b.bin_width)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.eps_cents
                    .partial_cmp(&b.eps_cents)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                a.mirror_weight
                    .partial_cmp(&b.mirror_weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.count_mode.cmp(b.count_mode))
    });
    rows
}

fn e4_delta_effects_csv(rows: &[E4DeltaEffectRow]) -> String {
    let mut out = String::from(
        "count_mode,mirror_weight,bin_width,eps_cents,mean_delta,sd_delta,se_delta,ci_lo,ci_hi,n_seeds,mean_mass_m3,sd_mass_m3,mean_mass_M3,sd_mass_M3,mean_mass_P5,sd_mass_P5,mean_mass_P5class,sd_mass_P5class\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{},{:.3},{:.3},{:.1},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            row.count_mode,
            row.mirror_weight,
            row.bin_width,
            row.eps_cents,
            row.mean_delta,
            row.sd_delta,
            row.se_delta,
            row.ci_lo,
            row.ci_hi,
            row.n_seeds,
            row.mean_mass_min3,
            row.sd_mass_min3,
            row.mean_mass_maj3,
            row.sd_mass_maj3,
            row.mean_mass_p5,
            row.sd_mass_p5,
            row.mean_mass_p5_class,
            row.sd_mass_p5_class
        ));
    }
    out
}

fn e4_regression_rows(summaries: &[E4SummaryRecord]) -> Vec<E4RegressionRow> {
    let mut map: std::collections::HashMap<(&'static str, i32, i32), Vec<&E4SummaryRecord>> =
        std::collections::HashMap::new();
    for summary in summaries {
        let key = (
            summary.count_mode,
            float_key(summary.bin_width),
            float_key(summary.eps_cents),
        );
        map.entry(key).or_default().push(summary);
    }
    let mut rows = Vec::new();
    for ((_mode_key, _bw_key, _eps_key), group) in map {
        let count_mode = group[0].count_mode;
        let bin_width = group[0].bin_width;
        let eps_cents = group[0].eps_cents;
        let mut points: Vec<(f32, f32)> = group
            .iter()
            .map(|s| (s.mirror_weight, s.mean_delta))
            .collect();
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let xs: Vec<f32> = points.iter().map(|(x, _)| *x).collect();
        let ys: Vec<f32> = points.iter().map(|(_, y)| *y).collect();
        if let Some(fit) = linear_regression(&xs, &ys) {
            let df = xs.len().saturating_sub(2);
            let ci_half = if df < 1 {
                0.0
            } else {
                let t_crit = t_crit_975(df);
                t_crit * fit.se_slope
            };
            let seed =
                0xE4E4_0000_u64 ^ (float_key(bin_width) as u64) ^ (float_key(eps_cents) as u64);
            let spearman = spearman_rho(&xs, &ys);
            let spearman_p = perm_pvalue(&xs, &ys, 1000, seed, spearman_rho);
            rows.push(E4RegressionRow {
                count_mode,
                bin_width,
                eps_cents,
                slope: fit.slope,
                slope_ci_lo: fit.slope - ci_half,
                slope_ci_hi: fit.slope + ci_half,
                r2: fit.r2,
                spearman_rho: spearman,
                spearman_p,
                n_weights: xs.len(),
            });
        }
    }
    rows.sort_by(|a, b| {
        a.bin_width
            .partial_cmp(&b.bin_width)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.eps_cents
                    .partial_cmp(&b.eps_cents)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.count_mode.cmp(b.count_mode))
    });
    rows
}

fn e4_regression_csv(rows: &[E4RegressionRow]) -> String {
    let mut out = String::from(
        "count_mode,bin_width,eps_cents,slope,slope_ci_lo,slope_ci_hi,r2,spearman_rho,spearman_p,n_weights\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{},{:.3},{:.1},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
            row.count_mode,
            row.bin_width,
            row.eps_cents,
            row.slope,
            row.slope_ci_lo,
            row.slope_ci_hi,
            row.r2,
            row.spearman_rho,
            row.spearman_p,
            row.n_weights
        ));
    }
    out
}

fn e4_endpoint_effect_rows(records: &[E4RunRecord]) -> Vec<E4EndpointEffectRow> {
    #[derive(Default)]
    struct EndpointAccum {
        count_mode: &'static str,
        bin_width: f32,
        eps_cents: f32,
        delta0: Vec<f32>,
        delta1: Vec<f32>,
    }

    let mut map: std::collections::HashMap<(&'static str, i32, i32), EndpointAccum> =
        std::collections::HashMap::new();
    for record in records {
        if (record.mirror_weight - 0.0).abs() > 1e-6 && (record.mirror_weight - 1.0).abs() > 1e-6 {
            continue;
        }
        let key = (
            record.count_mode,
            float_key(record.bin_width),
            float_key(record.eps_cents),
        );
        let entry = map.entry(key).or_insert_with(|| EndpointAccum {
            count_mode: record.count_mode,
            bin_width: record.bin_width,
            eps_cents: record.eps_cents,
            ..EndpointAccum::default()
        });
        if (record.mirror_weight - 0.0).abs() < 1e-6 {
            entry.delta0.push(record.delta);
        } else {
            entry.delta1.push(record.delta);
        }
    }

    let mut rows = Vec::new();
    for (_, entry) in map {
        let n0 = entry.delta0.len();
        let n1 = entry.delta1.len();
        if n0 == 0 || n1 == 0 {
            continue;
        }
        let (mean0, sd0) = mean_std_sample(&entry.delta0);
        let (mean1, sd1) = mean_std_sample(&entry.delta1);
        let delta_end = mean1 - mean0;
        let se = (sd0.powi(2) / n0 as f32 + sd1.powi(2) / n1 as f32).sqrt();
        let ci_half = 1.96 * se;
        let df = (n0 + n1).saturating_sub(2) as f32;
        let pooled_sd = if df > 0.0 {
            (((n0.saturating_sub(1)) as f32 * sd0.powi(2)
                + (n1.saturating_sub(1)) as f32 * sd1.powi(2))
                / df)
                .sqrt()
        } else {
            0.0
        };
        let cohen_d = if pooled_sd > 0.0 {
            delta_end / pooled_sd
        } else {
            0.0
        };
        rows.push(E4EndpointEffectRow {
            count_mode: entry.count_mode,
            bin_width: entry.bin_width,
            eps_cents: entry.eps_cents,
            delta_end,
            ci_lo: delta_end - ci_half,
            ci_hi: delta_end + ci_half,
            cohen_d,
            n0,
            n1,
        });
    }
    rows.sort_by(|a, b| {
        a.bin_width
            .partial_cmp(&b.bin_width)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.eps_cents
                    .partial_cmp(&b.eps_cents)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.count_mode.cmp(b.count_mode))
    });
    rows
}

fn e4_endpoint_effect_csv(rows: &[E4EndpointEffectRow]) -> String {
    let mut out =
        String::from("count_mode,bin_width,eps_cents,delta_end,ci_lo,ci_hi,cohen_d,n0,n1\n");
    for row in rows {
        out.push_str(&format!(
            "{},{:.3},{:.1},{:.6},{:.6},{:.6},{:.6},{},{}\n",
            row.count_mode,
            row.bin_width,
            row.eps_cents,
            row.delta_end,
            row.ci_lo,
            row.ci_hi,
            row.cohen_d,
            row.n0,
            row.n1
        ));
    }
    out
}

fn e4_seed_slopes_rows(records: &[E4RunRecord]) -> Vec<E4SeedSlopeRow> {
    type SeedSlopeKey = (&'static str, u64, i32, i32);
    type SeedSlopePoint = (f32, f32);
    let mut map: std::collections::HashMap<SeedSlopeKey, Vec<SeedSlopePoint>> =
        std::collections::HashMap::new();
    for record in records {
        let key = (
            record.count_mode,
            record.seed,
            float_key(record.bin_width),
            float_key(record.eps_cents),
        );
        map.entry(key)
            .or_default()
            .push((record.mirror_weight, record.delta));
    }
    let mut rows = Vec::new();
    for ((count_mode, seed, bw_key, eps_key), mut points) in map {
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let xs: Vec<f32> = points.iter().map(|(x, _)| *x).collect();
        let ys: Vec<f32> = points.iter().map(|(_, y)| *y).collect();
        if let Some(fit) = linear_regression(&xs, &ys) {
            rows.push(E4SeedSlopeRow {
                count_mode,
                seed,
                bin_width: float_from_key(bw_key),
                eps_cents: float_from_key(eps_key),
                slope_seed: fit.slope,
                r2_seed: fit.r2,
                n_weights: xs.len(),
            });
        }
    }
    rows.sort_by(|a, b| {
        a.bin_width
            .partial_cmp(&b.bin_width)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.eps_cents
                    .partial_cmp(&b.eps_cents)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.count_mode.cmp(b.count_mode))
            .then_with(|| a.seed.cmp(&b.seed))
    });
    rows
}

fn e4_seed_slopes_csv(rows: &[E4SeedSlopeRow]) -> String {
    let mut out =
        String::from("count_mode,seed,bin_width,eps_cents,slope_seed,r2_seed,n_weights\n");
    for row in rows {
        out.push_str(&format!(
            "{},{},{:.3},{:.1},{:.6},{:.6},{}\n",
            row.count_mode,
            row.seed,
            row.bin_width,
            row.eps_cents,
            row.slope_seed,
            row.r2_seed,
            row.n_weights
        ));
    }
    out
}

fn e4_run_level_regression_rows(records: &[E4RunRecord]) -> Vec<E4RunLevelRegressionRow> {
    type RunLevelKey = (&'static str, i32, i32);
    type RunLevelPoint = (f32, f32);
    let mut map: std::collections::HashMap<RunLevelKey, Vec<RunLevelPoint>> =
        std::collections::HashMap::new();
    for record in records {
        let key = (
            record.count_mode,
            float_key(record.bin_width),
            float_key(record.eps_cents),
        );
        map.entry(key)
            .or_default()
            .push((record.mirror_weight, record.delta));
    }
    let mut rows = Vec::new();
    for ((count_mode, bw_key, eps_key), mut points) in map {
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let xs: Vec<f32> = points.iter().map(|(x, _)| *x).collect();
        let ys: Vec<f32> = points.iter().map(|(_, y)| *y).collect();
        if let Some(fit) = linear_regression(&xs, &ys) {
            let df = xs.len().saturating_sub(2);
            let ci_half = if df < 1 {
                0.0
            } else {
                let t_crit = t_crit_975(df);
                t_crit * fit.se_slope
            };
            rows.push(E4RunLevelRegressionRow {
                count_mode,
                bin_width: float_from_key(bw_key),
                eps_cents: float_from_key(eps_key),
                slope: fit.slope,
                slope_ci_lo: fit.slope - ci_half,
                slope_ci_hi: fit.slope + ci_half,
                r2: fit.r2,
                n_runs: xs.len(),
            });
        }
    }
    rows.sort_by(|a, b| {
        a.bin_width
            .partial_cmp(&b.bin_width)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.eps_cents
                    .partial_cmp(&b.eps_cents)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.count_mode.cmp(b.count_mode))
    });
    rows
}

fn e4_run_level_regression_csv(rows: &[E4RunLevelRegressionRow]) -> String {
    let mut out =
        String::from("count_mode,bin_width,eps_cents,slope,slope_ci_lo,slope_ci_hi,r2,n_runs\n");
    for row in rows {
        out.push_str(&format!(
            "{},{:.3},{:.1},{:.6},{:.6},{:.6},{:.6},{}\n",
            row.count_mode,
            row.bin_width,
            row.eps_cents,
            row.slope,
            row.slope_ci_lo,
            row.slope_ci_hi,
            row.r2,
            row.n_runs
        ));
    }
    out
}

fn e4_seed_slope_meta_rows(seed_rows: &[E4SeedSlopeRow]) -> Vec<E4SeedSlopeMetaRow> {
    let mut map: std::collections::HashMap<(&'static str, i32, i32), Vec<&E4SeedSlopeRow>> =
        std::collections::HashMap::new();
    for row in seed_rows {
        let key = (
            row.count_mode,
            float_key(row.bin_width),
            float_key(row.eps_cents),
        );
        map.entry(key).or_default().push(row);
    }
    let mut rows = Vec::new();
    for ((count_mode, bw_key, eps_key), group) in map {
        let mut slopes = Vec::new();
        let mut r2s = Vec::new();
        let mut n_pos = 0usize;
        let mut n_nonzero = 0usize;
        for row in &group {
            slopes.push(row.slope_seed);
            r2s.push(row.r2_seed);
            if row.slope_seed.abs() > 1e-9 {
                n_nonzero += 1;
                if row.slope_seed > 0.0 {
                    n_pos += 1;
                }
            }
        }
        let (mean_slope, std_slope) = mean_std_sample(&slopes);
        let ci_half = if slopes.len() < 2 {
            0.0
        } else {
            let se = std_to_se(std_slope, slopes.len());
            let t_crit = t_crit_975(slopes.len().saturating_sub(1));
            t_crit * se
        };
        let sign_p = if n_nonzero > 0 {
            binomial_two_sided_p(n_pos.min(n_nonzero), n_nonzero)
        } else {
            1.0
        };
        let (mean_r2, _) = mean_std(&r2s);
        rows.push(E4SeedSlopeMetaRow {
            count_mode,
            bin_width: float_from_key(bw_key),
            eps_cents: float_from_key(eps_key),
            mean_slope,
            ci_lo: mean_slope - ci_half,
            ci_hi: mean_slope + ci_half,
            sign_p,
            var_slope_across_seeds: std_slope.powi(2),
            mean_r2,
            n_seeds: slopes.len(),
        });
    }
    rows.sort_by(|a, b| {
        a.bin_width
            .partial_cmp(&b.bin_width)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.eps_cents
                    .partial_cmp(&b.eps_cents)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.count_mode.cmp(b.count_mode))
    });
    rows
}

fn e4_seed_slope_meta_csv(rows: &[E4SeedSlopeMetaRow]) -> String {
    let mut out = String::from(
        "count_mode,bin_width,eps_cents,mean_slope,ci_lo,ci_hi,sign_p,var_slope_across_seeds,mean_r2,n_seeds\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{},{:.3},{:.1},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
            row.count_mode,
            row.bin_width,
            row.eps_cents,
            row.mean_slope,
            row.ci_lo,
            row.ci_hi,
            row.sign_p,
            row.var_slope_across_seeds,
            row.mean_r2,
            row.n_seeds
        ));
    }
    out
}

fn e4_total_third_mass_rows(records: &[E4RunRecord]) -> Vec<E4ThirdMassRow> {
    let mut map: std::collections::HashMap<(&'static str, i32, i32, i32), Vec<f32>> =
        std::collections::HashMap::new();
    for record in records {
        let key = (
            record.count_mode,
            float_key(record.mirror_weight),
            float_key(record.bin_width),
            float_key(record.eps_cents),
        );
        let third_mass = record.mass_min3 + record.mass_maj3;
        map.entry(key).or_default().push(third_mass);
    }
    let mut rows = Vec::new();
    for ((count_mode, w_key, bw_key, eps_key), values) in map {
        let (mean_third_mass, std_third_mass) = mean_std(&values);
        rows.push(E4ThirdMassRow {
            count_mode,
            mirror_weight: float_from_key(w_key),
            bin_width: float_from_key(bw_key),
            eps_cents: float_from_key(eps_key),
            mean_third_mass,
            std_third_mass,
            n_runs: values.len(),
        });
    }
    rows.sort_by(|a, b| {
        a.bin_width
            .partial_cmp(&b.bin_width)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.eps_cents
                    .partial_cmp(&b.eps_cents)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                a.mirror_weight
                    .partial_cmp(&b.mirror_weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.count_mode.cmp(b.count_mode))
    });
    rows
}

fn e4_total_third_mass_csv(rows: &[E4ThirdMassRow]) -> String {
    let mut out = String::from(
        "count_mode,mirror_weight,bin_width,eps_cents,mean_third_mass,std_third_mass,n_runs\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{},{:.3},{:.3},{:.1},{:.6},{:.6},{}\n",
            row.count_mode,
            row.mirror_weight,
            row.bin_width,
            row.eps_cents,
            row.mean_third_mass,
            row.std_third_mass,
            row.n_runs
        ));
    }
    out
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

fn bootstrap_mean_ci95(values: &[f32], iters: usize, seed: u64) -> (f32, f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
    if values.len() < 2 || iters == 0 {
        return (mean, mean, mean);
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut acc = 0.0f32;
        for _ in 0..values.len() {
            let idx = rng.random_range(0..values.len());
            acc += values[idx];
        }
        samples.push(acc / values.len() as f32);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo_idx = ((iters as f32 * 0.025).floor() as usize).min(iters - 1);
    let hi_idx = ((iters as f32 * 0.975).floor() as usize).min(iters - 1);
    (mean, samples[lo_idx], samples[hi_idx])
}

fn estimate_piecewise_changepoint(points: &[(f32, f32)]) -> Option<f32> {
    if points.len() < 4 {
        return None;
    }
    let mut best_sse = f32::INFINITY;
    let mut best_w = 0.5f32;
    for split in 1..(points.len() - 1) {
        let left = &points[..=split];
        let right = &points[split..];
        if left.len() < 2 || right.len() < 2 {
            continue;
        }
        let (lx, ly): (Vec<f32>, Vec<f32>) = left.iter().copied().unzip();
        let (rx, ry): (Vec<f32>, Vec<f32>) = right.iter().copied().unzip();
        let Some(lfit) = linear_regression(&lx, &ly) else {
            continue;
        };
        let Some(rfit) = linear_regression(&rx, &ry) else {
            continue;
        };
        let sse_left: f32 = left
            .iter()
            .map(|(x, y)| {
                let pred = lfit.intercept + lfit.slope * *x;
                (y - pred).powi(2)
            })
            .sum();
        let sse_right: f32 = right
            .iter()
            .map(|(x, y)| {
                let pred = rfit.intercept + rfit.slope * *x;
                (y - pred).powi(2)
            })
            .sum();
        let sse = sse_left + sse_right;
        if sse < best_sse {
            best_sse = sse;
            best_w = points[split].0;
        }
    }
    if best_sse.is_finite() {
        Some(best_w)
    } else {
        None
    }
}

fn render_e4_figure1_mirror_vs_delta_t(
    out_path: &Path,
    run_records: &[E4RunRecord],
    bin_width: f32,
    eps_cents: f32,
    count_mode: &'static str,
    panel_label: &str,
) -> Result<(), Box<dyn Error>> {
    let mut grouped: std::collections::HashMap<i32, Vec<f32>> = std::collections::HashMap::new();
    for record in run_records {
        if record.count_mode != count_mode {
            continue;
        }
        if (record.bin_width - bin_width).abs() > 1e-6
            || (record.eps_cents - eps_cents).abs() > 1e-6
        {
            continue;
        }
        grouped
            .entry(float_key(record.mirror_weight))
            .or_default()
            .push(record.delta_t);
    }
    if grouped.is_empty() {
        return Ok(());
    }
    let mut rows: Vec<(f32, f32, f32, f32, usize)> = Vec::new();
    for (w_key, deltas) in grouped {
        let w = float_from_key(w_key);
        let seed = E4_BOOTSTRAP_SEED ^ (w_key as u64) ^ (float_key(eps_cents) as u64);
        let (mean, lo, hi) = bootstrap_mean_ci95(&deltas, E4_BOOTSTRAP_ITERS, seed);
        rows.push((w, mean, lo, hi, deltas.len()));
    }
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if rows.is_empty() {
        return Ok(());
    }

    let mut y_min = 0.0f32;
    let mut y_max = 0.0f32;
    for (_, mean, lo, hi, _) in &rows {
        y_min = y_min.min(*lo).min(*mean);
        y_max = y_max.max(*hi).max(*mean);
    }
    if (y_max - y_min).abs() < 1e-6 {
        y_min -= 0.1;
        y_max += 0.1;
    }
    let pad = 0.15 * (y_max - y_min);

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "E4 Figure 1 ({panel_label}): Mirror Weight vs ΔT ({count_mode}, bw={:.1}c, eps={})",
                bin_width * 100.0,
                fmt_eps(eps_cents)
            ),
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, (y_min - pad)..(y_max + pad))?;

    chart
        .configure_mesh()
        .x_desc("mirror weight")
        .y_desc("ΔT")
        .draw()?;

    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0.0, 0.0), (1.0, 0.0)],
        BLACK.mix(0.85).stroke_width(3),
    )))?;

    if rows.len() >= 2 {
        let mut band: Vec<(f32, f32)> = rows.iter().map(|(w, _, _, hi, _)| (*w, *hi)).collect();
        band.extend(rows.iter().rev().map(|(w, _, lo, _, _)| (*w, *lo)));
        chart.draw_series(std::iter::once(Polygon::new(band, BLUE.mix(0.18).filled())))?;
    }

    chart.draw_series(LineSeries::new(
        rows.iter().map(|(w, mean, _, _, _)| (*w, *mean)),
        BLUE.mix(0.85).stroke_width(2),
    ))?;
    chart.draw_series(
        rows.iter()
            .map(|(w, mean, _, _, _)| Circle::new((*w, *mean), 4, BLUE.filled())),
    )?;

    let points: Vec<(f32, f32)> = rows.iter().map(|(w, mean, _, _, _)| (*w, *mean)).collect();
    if let Some(w_c) = estimate_piecewise_changepoint(&points) {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(w_c, y_min - pad), (w_c, y_max + pad)],
            BLACK.mix(0.5).stroke_width(2),
        )))?;
        chart.draw_series(std::iter::once(Text::new(
            format!("w_c≈{w_c:.2}"),
            (
                (w_c + 0.015).clamp(0.03, 0.92),
                y_max + 0.08 * (y_max - y_min + 1e-6),
            ),
            ("sans-serif", 14).into_font().color(&BLACK.mix(0.7)),
        )))?;
    }

    root.present()?;
    Ok(())
}

fn render_e4_bind_vs_weight(
    out_path: &Path,
    summary_rows: &[E4BindSummaryRow],
) -> Result<(), Box<dyn Error>> {
    if summary_rows.is_empty() {
        return Ok(());
    }
    let mut rows = summary_rows.to_vec();
    rows.sort_by(|a, b| {
        a.mirror_weight
            .partial_cmp(&b.mirror_weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut y_min = 0.0f32;
    let mut y_max = 0.0f32;
    for row in &rows {
        y_min = y_min.min(row.delta_bind_ci_lo).min(row.mean_delta_bind);
        y_max = y_max.max(row.delta_bind_ci_hi).max(row.mean_delta_bind);
    }
    if (y_max - y_min).abs() < 1e-6 {
        y_min -= 0.1;
        y_max += 0.1;
    }
    let pad = 0.15 * (y_max - y_min);

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "E4 Root/Ceiling Regime Shift: Δbind vs mirror weight",
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(42)
        .y_label_area_size(62)
        .build_cartesian_2d(0.0f32..1.0f32, (y_min - pad)..(y_max + pad))?;

    chart
        .configure_mesh()
        .x_desc("mirror weight")
        .y_desc("Δbind = (RootFit - CeilingFit) / (RootFit + CeilingFit)")
        .draw()?;

    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0.0, 0.0), (1.0, 0.0)],
        BLACK.mix(0.85).stroke_width(3),
    )))?;

    if rows.len() >= 2 {
        let mut band: Vec<(f32, f32)> = rows
            .iter()
            .map(|row| (row.mirror_weight, row.delta_bind_ci_hi))
            .collect();
        band.extend(
            rows.iter()
                .rev()
                .map(|row| (row.mirror_weight, row.delta_bind_ci_lo)),
        );
        chart.draw_series(std::iter::once(Polygon::new(band, BLUE.mix(0.20).filled())))?;
    }

    chart.draw_series(LineSeries::new(
        rows.iter()
            .map(|row| (row.mirror_weight, row.mean_delta_bind)),
        BLUE.mix(0.90).stroke_width(2),
    ))?;
    chart.draw_series(rows.iter().map(|row| {
        Circle::new(
            (row.mirror_weight, row.mean_delta_bind),
            4,
            BLUE.mix(0.95).filled(),
        )
    }))?;

    root.present()?;
    Ok(())
}

fn render_e4_root_ceiling_fit_vs_weight(
    out_path: &Path,
    summary_rows: &[E4BindSummaryRow],
) -> Result<(), Box<dyn Error>> {
    if summary_rows.is_empty() {
        return Ok(());
    }
    let mut rows = summary_rows.to_vec();
    rows.sort_by(|a, b| {
        a.mirror_weight
            .partial_cmp(&b.mirror_weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut y_max = 0.0f32;
    for row in &rows {
        y_max = y_max.max(row.root_ci_hi).max(row.ceiling_ci_hi);
    }
    y_max = (y_max * 1.15).max(1e-4);

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "E4 RootFit / CeilingFit vs mirror weight (mean ± 95% CI)",
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(42)
        .y_label_area_size(62)
        .build_cartesian_2d(0.0f32..1.0f32, 0.0f32..y_max)?;

    chart
        .configure_mesh()
        .x_desc("mirror weight")
        .y_desc("fit score")
        .draw()?;

    if rows.len() >= 2 {
        let mut root_band: Vec<(f32, f32)> = rows
            .iter()
            .map(|row| (row.mirror_weight, row.root_ci_hi))
            .collect();
        root_band.extend(
            rows.iter()
                .rev()
                .map(|row| (row.mirror_weight, row.root_ci_lo)),
        );
        chart.draw_series(std::iter::once(Polygon::new(
            root_band,
            BLUE.mix(0.15).filled(),
        )))?;

        let mut ceiling_band: Vec<(f32, f32)> = rows
            .iter()
            .map(|row| (row.mirror_weight, row.ceiling_ci_hi))
            .collect();
        ceiling_band.extend(
            rows.iter()
                .rev()
                .map(|row| (row.mirror_weight, row.ceiling_ci_lo)),
        );
        chart.draw_series(std::iter::once(Polygon::new(
            ceiling_band,
            RED.mix(0.12).filled(),
        )))?;
    }

    chart
        .draw_series(LineSeries::new(
            rows.iter()
                .map(|row| (row.mirror_weight, row.mean_root_fit)),
            BLUE.mix(0.95).stroke_width(2),
        ))?
        .label("RootFit")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], BLUE.stroke_width(2)));
    chart.draw_series(rows.iter().map(|row| {
        Circle::new(
            (row.mirror_weight, row.mean_root_fit),
            4,
            BLUE.mix(0.95).filled(),
        )
    }))?;

    chart
        .draw_series(LineSeries::new(
            rows.iter()
                .map(|row| (row.mirror_weight, row.mean_ceiling_fit)),
            RED.mix(0.90).stroke_width(2),
        ))?
        .label("CeilingFit")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], RED.stroke_width(2)));
    chart.draw_series(rows.iter().map(|row| {
        TriangleMarker::new(
            (row.mirror_weight, row.mean_ceiling_fit),
            6,
            RED.mix(0.90).filled(),
        )
    }))?;

    chart
        .configure_series_labels()
        .border_style(BLACK.mix(0.4))
        .background_style(WHITE.mix(0.8))
        .draw()?;

    root.present()?;
    Ok(())
}

fn render_e4_delta_bind_png(
    out_path: &Path,
    summary_rows: &[E4BindSummaryRow],
) -> Result<(), Box<dyn Error>> {
    if summary_rows.is_empty() {
        return Ok(());
    }
    let mut rows = summary_rows.to_vec();
    rows.sort_by(|a, b| {
        a.mirror_weight
            .partial_cmp(&b.mirror_weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut y_min = 0.0f32;
    let mut y_max = 0.0f32;
    for row in &rows {
        y_min = y_min.min(row.delta_bind_ci_lo).min(row.mean_delta_bind);
        y_max = y_max.max(row.delta_bind_ci_hi).max(row.mean_delta_bind);
    }
    if (y_max - y_min).abs() < 1e-6 {
        y_min -= 0.1;
        y_max += 0.1;
    }
    let pad = 0.15 * (y_max - y_min);

    log_output_path(out_path);
    let root = BitMapBackend::new(out_path, (1400, 850)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "E4 DeltaBind vs mirror_weight (mean ± 95% CI)",
            ("sans-serif", 24),
        )
        .margin(12)
        .x_label_area_size(46)
        .y_label_area_size(68)
        .build_cartesian_2d(0.0f32..1.0f32, (y_min - pad)..(y_max + pad))?;

    chart
        .configure_mesh()
        .x_desc("mirror_weight")
        .y_desc("DeltaBind")
        .draw()?;

    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0.0, 0.0), (1.0, 0.0)],
        BLACK.mix(0.85).stroke_width(2),
    )))?;

    if rows.len() >= 2 {
        let mut band: Vec<(f32, f32)> = rows
            .iter()
            .map(|row| (row.mirror_weight, row.delta_bind_ci_hi))
            .collect();
        band.extend(
            rows.iter()
                .rev()
                .map(|row| (row.mirror_weight, row.delta_bind_ci_lo)),
        );
        chart.draw_series(std::iter::once(Polygon::new(band, BLUE.mix(0.20).filled())))?;
    }

    chart.draw_series(LineSeries::new(
        rows.iter()
            .map(|row| (row.mirror_weight, row.mean_delta_bind)),
        BLUE.mix(0.95).stroke_width(3),
    ))?;
    chart.draw_series(rows.iter().map(|row| {
        Circle::new(
            (row.mirror_weight, row.mean_delta_bind),
            5,
            BLUE.mix(0.95).filled(),
        )
    }))?;

    root.present()?;
    Ok(())
}

fn render_e4_binding_phase_diagram_png(
    out_path: &Path,
    summary_rows: &[E4BindSummaryRow],
) -> Result<(), Box<dyn Error>> {
    if summary_rows.is_empty() {
        return Ok(());
    }
    let mut rows = summary_rows.to_vec();
    rows.sort_by(|a, b| {
        a.mirror_weight
            .partial_cmp(&b.mirror_weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut y_max = 0.0f32;
    for row in &rows {
        y_max = y_max.max(row.root_ci_hi).max(row.ceiling_ci_hi);
    }
    y_max = (y_max * 1.15).max(1e-4);

    log_output_path(out_path);
    let root = BitMapBackend::new(out_path, (1400, 850)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "E4 Binding Phase Diagram: RootFit / CeilingFit (mean ± 95% CI)",
            ("sans-serif", 24),
        )
        .margin(12)
        .x_label_area_size(46)
        .y_label_area_size(68)
        .build_cartesian_2d(0.0f32..1.0f32, 0.0f32..y_max)?;

    chart
        .configure_mesh()
        .x_desc("mirror_weight")
        .y_desc("fit")
        .draw()?;

    if rows.len() >= 2 {
        let mut root_band: Vec<(f32, f32)> = rows
            .iter()
            .map(|row| (row.mirror_weight, row.root_ci_hi))
            .collect();
        root_band.extend(
            rows.iter()
                .rev()
                .map(|row| (row.mirror_weight, row.root_ci_lo)),
        );
        chart.draw_series(std::iter::once(Polygon::new(
            root_band,
            BLUE.mix(0.14).filled(),
        )))?;

        let mut ceiling_band: Vec<(f32, f32)> = rows
            .iter()
            .map(|row| (row.mirror_weight, row.ceiling_ci_hi))
            .collect();
        ceiling_band.extend(
            rows.iter()
                .rev()
                .map(|row| (row.mirror_weight, row.ceiling_ci_lo)),
        );
        chart.draw_series(std::iter::once(Polygon::new(
            ceiling_band,
            RED.mix(0.12).filled(),
        )))?;
    }

    chart
        .draw_series(LineSeries::new(
            rows.iter()
                .map(|row| (row.mirror_weight, row.mean_root_fit)),
            BLUE.mix(0.95).stroke_width(3),
        ))?
        .label("RootFit")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], BLUE.stroke_width(3)));
    chart.draw_series(rows.iter().map(|row| {
        Circle::new(
            (row.mirror_weight, row.mean_root_fit),
            5,
            BLUE.mix(0.95).filled(),
        )
    }))?;

    chart
        .draw_series(LineSeries::new(
            rows.iter()
                .map(|row| (row.mirror_weight, row.mean_ceiling_fit)),
            RED.mix(0.92).stroke_width(3),
        ))?
        .label("CeilingFit")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], RED.stroke_width(3)));
    chart.draw_series(rows.iter().map(|row| {
        TriangleMarker::new(
            (row.mirror_weight, row.mean_ceiling_fit),
            7,
            RED.mix(0.92).filled(),
        )
    }))?;

    chart
        .configure_series_labels()
        .border_style(BLACK.mix(0.4))
        .background_style(WHITE.mix(0.8))
        .draw()?;

    root.present()?;
    Ok(())
}

fn render_e4_fingerprint_heatmap_png(
    out_path: &Path,
    summary_rows: &[E4FingerprintSummaryRow],
) -> Result<(), Box<dyn Error>> {
    if summary_rows.is_empty() {
        return Ok(());
    }
    let mut weight_keys: Vec<i32> = summary_rows
        .iter()
        .map(|row| float_key(row.mirror_weight))
        .collect();
    weight_keys.sort();
    weight_keys.dedup();
    if weight_keys.is_empty() {
        return Ok(());
    }
    let weights: Vec<f32> = weight_keys.iter().map(|key| float_from_key(*key)).collect();

    let mut mean_map: std::collections::HashMap<(i32, &'static str), f32> =
        std::collections::HashMap::new();
    let mut max_prob = 0.0f32;
    for row in summary_rows {
        mean_map.insert((float_key(row.mirror_weight), row.category), row.mean_prob);
        max_prob = max_prob.max(row.mean_prob);
    }
    let labels = E4_FINGERPRINT_LABELS;

    log_output_path(out_path);
    let root = BitMapBackend::new(out_path, (1500, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "E4 Interval Fingerprint Heatmap (JI categories)",
            ("sans-serif", 24),
        )
        .margin(14)
        .x_label_area_size(46)
        .y_label_area_size(90)
        .build_cartesian_2d(0i32..weights.len() as i32, 0i32..labels.len() as i32)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("mirror_weight")
        .y_desc("interval category")
        .x_labels(weights.len())
        .y_labels(labels.len())
        .x_label_formatter(&|x| {
            let idx = (*x).clamp(0, weights.len().saturating_sub(1) as i32) as usize;
            format!("{:.2}", weights[idx])
        })
        .y_label_formatter(&|y| {
            let idx = (*y).clamp(0, labels.len().saturating_sub(1) as i32) as usize;
            labels[idx].to_string()
        })
        .draw()?;

    for (x_idx, weight_key) in weight_keys.iter().enumerate() {
        for (y_idx, category) in labels.iter().enumerate() {
            let prob = mean_map
                .get(&(*weight_key, *category))
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let t = if max_prob > 1e-9 {
                (prob / max_prob).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let color = RGBColor(
                (255.0 * t) as u8,
                (30.0 + 170.0 * (1.0 - t)) as u8,
                (255.0 * (1.0 - t)) as u8,
            );
            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (x_idx as i32, y_idx as i32),
                    ((x_idx + 1) as i32, (y_idx + 1) as i32),
                ],
                color.filled(),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}

fn render_e4_figure2_interval_hist_triptych(
    out_path: &Path,
    hist_records: &[E4HistRecord],
    bin_width: f32,
) -> Result<(), Box<dyn Error>> {
    let targets = [0.0f32, 0.5f32, 1.0f32];
    let mut panels: Vec<(f32, Histogram)> = Vec::new();
    for &w in &targets {
        if let Some(hist) = mean_histogram_for_weight(hist_records, w, bin_width) {
            panels.push((w, hist));
        }
    }
    if panels.is_empty() {
        return Ok(());
    }
    let bin_width_cents = (bin_width * 100.0).max(1.0);
    let mut y_max = 0.0f32;
    for (_, hist) in &panels {
        y_max = y_max.max(hist.masses.iter().copied().fold(0.0f32, f32::max));
    }
    y_max = (y_max * 1.2).max(1e-4);

    let root = bitmap_root(out_path, (1500, 520)).into_drawing_area();
    root.fill(&WHITE)?;
    let areas = root.split_evenly((1, panels.len()));
    for (i, area) in areas.iter().enumerate() {
        let (weight, hist) = &panels[i];
        let mut chart = ChartBuilder::on(area)
            .caption(
                format!("E4 Figure 2: w={weight:.2} (bin={bin_width_cents:.1}c)"),
                ("sans-serif", 16),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(52)
            .build_cartesian_2d(0.0f32..1200.0f32, 0.0f32..y_max)?;
        chart
            .configure_mesh()
            .x_desc("interval cents (pitch class)")
            .y_desc("probability density (sum=1)")
            .x_labels(7)
            .draw()?;
        for &x in &[E4_CENTS_MIN3, E4_CENTS_MAJ3, E4_CENTS_P4, E4_CENTS_P5] {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x, 0.0), (x, y_max)],
                BLACK.mix(0.2),
            )))?;
        }
        chart.draw_series(
            hist.bin_centers
                .iter()
                .zip(hist.masses.iter())
                .map(|(x, y)| {
                    let x0 = (*x - 0.5 * bin_width_cents).max(0.0);
                    let x1 = (*x + 0.5 * bin_width_cents).min(1200.0);
                    Rectangle::new([(x0, 0.0), (x1, *y)], BLUE.mix(0.55).filled())
                }),
        )?;
    }
    root.present()?;
    Ok(())
}

fn delta_t_from_freqs(anchor_hz: f32, freqs: &[f32], eps_cents: f32, mode: E4CountMode) -> f32 {
    let masses = interval_masses_from_freqs(anchor_hz, freqs, eps_cents, mode);
    let (_, _, delta_t, _) = triad_scores(masses);
    delta_t
}

fn delta_bind_from_freqs(freqs: &[f32]) -> f32 {
    let (_, _, delta_bind) = bind_scores_from_freqs(freqs);
    delta_bind
}

fn render_e4_step_response_delta_plot(
    out_path: &Path,
    step_rows: &[(u32, f32, f32, f32)],
    switch_step: u32,
) -> Result<(), Box<dyn Error>> {
    if step_rows.is_empty() {
        return Ok(());
    }
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for (_, _, d_soft, d_hard) in step_rows {
        y_min = y_min.min(*d_soft).min(*d_hard).min(0.0);
        y_max = y_max.max(*d_soft).max(*d_hard).max(0.0);
    }
    if !y_min.is_finite() || !y_max.is_finite() || (y_max - y_min).abs() < 1e-6 {
        y_min = -0.1;
        y_max = 0.1;
    }
    let pad = 0.12 * (y_max - y_min);
    let x_end = step_rows
        .last()
        .map(|(s, _, _, _)| *s as f32)
        .unwrap_or(1.0f32);

    let root = bitmap_root(out_path, (1300, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("E4 Step Response: ΔT(t)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(42)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..x_end.max(1.0), (y_min - pad)..(y_max + pad))?;
    chart.configure_mesh().x_desc("step").y_desc("ΔT").draw()?;
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0.0, 0.0), (x_end.max(1.0), 0.0)],
        BLACK.mix(0.8).stroke_width(3),
    )))?;
    chart.draw_series(std::iter::once(PathElement::new(
        vec![
            (switch_step as f32, y_min - pad),
            (switch_step as f32, y_max + pad),
        ],
        BLACK.mix(0.45).stroke_width(2),
    )))?;
    chart.draw_series(std::iter::once(Text::new(
        "phase switch",
        (
            (switch_step as f32 + 5.0).min(x_end * 0.95),
            y_max + 0.03 * (y_max - y_min + 1e-6),
        ),
        ("sans-serif", 14).into_font().color(&BLACK.mix(0.7)),
    )))?;
    chart.draw_series(LineSeries::new(
        step_rows
            .iter()
            .map(|(step, _, delta_soft, _)| (*step as f32, *delta_soft)),
        BLUE.mix(0.9).stroke_width(2),
    ))?;
    chart.draw_series(LineSeries::new(
        step_rows
            .iter()
            .map(|(step, _, _, delta_hard)| (*step as f32, *delta_hard)),
        RED.mix(0.75).stroke_width(2),
    ))?;
    root.present()?;
    Ok(())
}

fn render_e4_step_response_delta_bind_plot(
    out_path: &Path,
    step_rows: &[(u32, f32, f32)],
    switch_step: u32,
) -> Result<(), Box<dyn Error>> {
    if step_rows.is_empty() {
        return Ok(());
    }
    let mut y_min = 0.0f32;
    let mut y_max = 0.0f32;
    for (_, _, delta_bind) in step_rows {
        y_min = y_min.min(*delta_bind).min(0.0);
        y_max = y_max.max(*delta_bind).max(0.0);
    }
    if !y_min.is_finite() || !y_max.is_finite() || (y_max - y_min).abs() < 1e-6 {
        y_min = -0.1;
        y_max = 0.1;
    }
    let pad = 0.12 * (y_max - y_min);
    let x_end = step_rows
        .last()
        .map(|(s, _, _)| *s as f32)
        .unwrap_or(1.0f32);

    let root = bitmap_root(out_path, (1300, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("E4 Step Response: Δbind(t)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(42)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..x_end.max(1.0), (y_min - pad)..(y_max + pad))?;
    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc("Δbind")
        .draw()?;
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0.0, 0.0), (x_end.max(1.0), 0.0)],
        BLACK.mix(0.85).stroke_width(3),
    )))?;
    chart.draw_series(std::iter::once(PathElement::new(
        vec![
            (switch_step as f32, y_min - pad),
            (switch_step as f32, y_max + pad),
        ],
        BLACK.mix(0.45).stroke_width(2),
    )))?;
    chart.draw_series(std::iter::once(Text::new(
        "phase switch",
        (
            (switch_step as f32 + 5.0).min(x_end * 0.95),
            y_max + 0.03 * (y_max - y_min + 1e-6),
        ),
        ("sans-serif", 14).into_font().color(&BLACK.mix(0.7)),
    )))?;
    chart.draw_series(LineSeries::new(
        step_rows
            .iter()
            .map(|(step, _, delta_bind)| (*step as f32, *delta_bind)),
        BLUE.mix(0.90).stroke_width(2),
    ))?;
    chart.draw_series(step_rows.iter().map(|(step, _, delta_bind)| {
        Circle::new((*step as f32, *delta_bind), 3, BLUE.mix(0.95).filled())
    }))?;
    root.present()?;
    Ok(())
}

fn stage_means_from_trace(
    weights: &[f32],
    delta_series: &[f32],
    settle_steps: u32,
    eval_window: u32,
) -> Vec<(f32, f32)> {
    if settle_steps == 0 {
        return Vec::new();
    }
    let window = eval_window.max(1).min(settle_steps) as usize;
    let mut out = Vec::new();
    for (idx, &weight) in weights.iter().enumerate() {
        let start = idx * settle_steps as usize;
        let end = ((idx + 1) * settle_steps as usize).min(delta_series.len());
        if end <= start {
            continue;
        }
        let begin = end.saturating_sub(window).max(start);
        let slice = &delta_series[begin..end];
        if slice.is_empty() {
            continue;
        }
        let mean = slice.iter().copied().sum::<f32>() / slice.len() as f32;
        out.push((weight, mean));
    }
    out
}

fn render_e4_hysteresis_plot(
    out_path: &Path,
    up_curve: &[(f32, f32)],
    down_curve: &[(f32, f32)],
) -> Result<(), Box<dyn Error>> {
    if up_curve.is_empty() || down_curve.is_empty() {
        return Ok(());
    }
    let mut y_min = 0.0f32;
    let mut y_max = 0.0f32;
    for (_, y) in up_curve.iter().chain(down_curve.iter()) {
        y_min = y_min.min(*y);
        y_max = y_max.max(*y);
    }
    if (y_max - y_min).abs() < 1e-6 {
        y_min -= 0.1;
        y_max += 0.1;
    }
    let pad = 0.12 * (y_max - y_min);
    let root = bitmap_root(out_path, (1100, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("E4 Hysteresis: ΔT vs Mirror Weight", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, (y_min - pad)..(y_max + pad))?;
    chart
        .configure_mesh()
        .x_desc("mirror weight")
        .y_desc("ΔT (soft)")
        .draw()?;
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0.0, 0.0), (1.0, 0.0)],
        BLACK.mix(0.8).stroke_width(2),
    )))?;
    chart.draw_series(LineSeries::new(
        up_curve.iter().copied(),
        BLUE.mix(0.9).stroke_width(2),
    ))?;
    chart.draw_series(LineSeries::new(
        down_curve.iter().copied(),
        RED.mix(0.85).stroke_width(2),
    ))?;
    chart.draw_series(
        up_curve
            .iter()
            .map(|(x, y)| Circle::new((*x, *y), 4, BLUE.filled())),
    )?;
    chart.draw_series(
        down_curve
            .iter()
            .map(|(x, y)| TriangleMarker::new((*x, *y), 6, RED.filled())),
    )?;
    root.present()?;
    Ok(())
}

fn render_e4_hysteresis_bind_plot(
    out_path: &Path,
    up_curve: &[(f32, f32)],
    down_curve: &[(f32, f32)],
) -> Result<(), Box<dyn Error>> {
    if up_curve.is_empty() || down_curve.is_empty() {
        return Ok(());
    }
    let mut y_min = 0.0f32;
    let mut y_max = 0.0f32;
    for (_, y) in up_curve.iter().chain(down_curve.iter()) {
        y_min = y_min.min(*y);
        y_max = y_max.max(*y);
    }
    if (y_max - y_min).abs() < 1e-6 {
        y_min -= 0.1;
        y_max += 0.1;
    }
    let pad = 0.12 * (y_max - y_min);
    let root = bitmap_root(out_path, (1100, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("E4 Hysteresis: Δbind vs Mirror Weight", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, (y_min - pad)..(y_max + pad))?;
    chart
        .configure_mesh()
        .x_desc("mirror weight")
        .y_desc("Δbind")
        .draw()?;
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0.0, 0.0), (1.0, 0.0)],
        BLACK.mix(0.8).stroke_width(2),
    )))?;
    chart.draw_series(LineSeries::new(
        up_curve.iter().copied(),
        BLUE.mix(0.9).stroke_width(2),
    ))?;
    chart.draw_series(LineSeries::new(
        down_curve.iter().copied(),
        RED.mix(0.85).stroke_width(2),
    ))?;
    chart.draw_series(
        up_curve
            .iter()
            .map(|(x, y)| Circle::new((*x, *y), 4, BLUE.filled())),
    )?;
    chart.draw_series(
        down_curve
            .iter()
            .map(|(x, y)| TriangleMarker::new((*x, *y), 6, RED.filled())),
    )?;
    root.present()?;
    Ok(())
}

fn run_e4_step_and_hysteresis_protocol(
    out_dir: &Path,
    anchor_hz: f32,
    eps_cents: f32,
) -> Result<(), Box<dyn Error>> {
    let step_schedule = vec![(0u32, 0.0f32), (E4_STEP_BURN_IN_STEPS, 1.0f32)];
    let step_total = E4_STEP_BURN_IN_STEPS + E4_STEP_POST_STEPS;
    let step_samples = run_e4_mirror_schedule_samples(E4_PROTOCOL_SEED, step_total, &step_schedule);
    let mut step_rows: Vec<(u32, f32, f32, f32)> =
        Vec::with_capacity(step_samples.freqs_by_step.len());
    let mut step_bind_rows: Vec<(u32, f32, f32)> =
        Vec::with_capacity(step_samples.freqs_by_step.len());
    let mut step_csv = String::from("seed,step,mirror_weight,delta_t_soft,delta_t_hard\n");
    let mut step_bind_csv = String::from("seed,step,mirror_weight,delta_bind\n");
    let mut step_hist_csv =
        String::from("seed,step,mirror_weight,bin_width_cents,bin_center_cents,mass\n");
    for (step, freqs) in step_samples.freqs_by_step.iter().enumerate() {
        let mirror_weight = step_samples
            .mirror_weight_by_step
            .get(step)
            .copied()
            .unwrap_or(0.0);
        let delta_soft = delta_t_from_freqs(anchor_hz, freqs, eps_cents, E4CountMode::Soft);
        let delta_hard = delta_t_from_freqs(anchor_hz, freqs, eps_cents, E4CountMode::Hard);
        let delta_bind = delta_bind_from_freqs(freqs);
        step_rows.push((step as u32, mirror_weight, delta_soft, delta_hard));
        step_bind_rows.push((step as u32, mirror_weight, delta_bind));
        step_csv.push_str(&format!(
            "{},{},{:.3},{:.6},{:.6}\n",
            E4_PROTOCOL_SEED, step, mirror_weight, delta_soft, delta_hard
        ));
        step_bind_csv.push_str(&format!(
            "{},{},{:.3},{:.6}\n",
            E4_PROTOCOL_SEED, step, mirror_weight, delta_bind
        ));
        let cents_values: Vec<f32> = freqs
            .iter()
            .filter_map(|&f| freq_to_cents_class(anchor_hz, f))
            .collect();
        let hist = histogram_from_samples(&cents_values, 0.0, 1200.0, E4_PAPER_HIST_BIN_CENTS);
        for (center, mass) in hist.bin_centers.iter().zip(hist.masses.iter()) {
            step_hist_csv.push_str(&format!(
                "{},{},{:.3},{:.1},{:.3},{:.6}\n",
                E4_PROTOCOL_SEED, step, mirror_weight, E4_PAPER_HIST_BIN_CENTS, center, mass
            ));
        }
    }
    write_with_log(out_dir.join("paper_e4_step_response_delta_t.csv"), step_csv)?;
    write_with_log(
        out_dir.join("paper_e4_step_response_delta_bind.csv"),
        step_bind_csv,
    )?;
    write_with_log(
        out_dir.join("paper_e4_step_response_interval_hist_timeseries.csv"),
        step_hist_csv,
    )?;
    render_e4_step_response_delta_plot(
        &out_dir.join("paper_e4_step_response_delta_t.svg"),
        &step_rows,
        E4_STEP_BURN_IN_STEPS,
    )?;
    render_e4_step_response_delta_bind_plot(
        &out_dir.join("paper_e4_step_response_delta_bind.svg"),
        &step_bind_rows,
        E4_STEP_BURN_IN_STEPS,
    )?;

    let weights_up = build_weight_grid(E4_WEIGHT_COARSE_STEP);
    if weights_up.is_empty() {
        return Ok(());
    }
    let weights_down: Vec<f32> = weights_up.iter().copied().rev().collect();
    let up_schedule: Vec<(u32, f32)> = weights_up
        .iter()
        .enumerate()
        .map(|(i, &w)| (i as u32 * E4_HYSTERESIS_SETTLE_STEPS, w))
        .collect();
    let down_schedule: Vec<(u32, f32)> = weights_down
        .iter()
        .enumerate()
        .map(|(i, &w)| (i as u32 * E4_HYSTERESIS_SETTLE_STEPS, w))
        .collect();
    let h_steps = E4_HYSTERESIS_SETTLE_STEPS * weights_up.len() as u32;
    let up_samples = run_e4_mirror_schedule_samples(E4_PROTOCOL_SEED, h_steps, &up_schedule);
    let down_samples = run_e4_mirror_schedule_samples(E4_PROTOCOL_SEED, h_steps, &down_schedule);

    let up_delta: Vec<f32> = up_samples
        .freqs_by_step
        .iter()
        .map(|freqs| delta_t_from_freqs(anchor_hz, freqs, eps_cents, E4CountMode::Soft))
        .collect();
    let up_delta_bind: Vec<f32> = up_samples
        .freqs_by_step
        .iter()
        .map(|freqs| delta_bind_from_freqs(freqs))
        .collect();
    let down_delta: Vec<f32> = down_samples
        .freqs_by_step
        .iter()
        .map(|freqs| delta_t_from_freqs(anchor_hz, freqs, eps_cents, E4CountMode::Soft))
        .collect();
    let down_delta_bind: Vec<f32> = down_samples
        .freqs_by_step
        .iter()
        .map(|freqs| delta_bind_from_freqs(freqs))
        .collect();

    let up_curve = stage_means_from_trace(
        &weights_up,
        &up_delta,
        E4_HYSTERESIS_SETTLE_STEPS,
        E4_HYSTERESIS_EVAL_WINDOW,
    );
    let down_curve_desc = stage_means_from_trace(
        &weights_down,
        &down_delta,
        E4_HYSTERESIS_SETTLE_STEPS,
        E4_HYSTERESIS_EVAL_WINDOW,
    );
    let up_curve_bind = stage_means_from_trace(
        &weights_up,
        &up_delta_bind,
        E4_HYSTERESIS_SETTLE_STEPS,
        E4_HYSTERESIS_EVAL_WINDOW,
    );
    let down_curve_bind_desc = stage_means_from_trace(
        &weights_down,
        &down_delta_bind,
        E4_HYSTERESIS_SETTLE_STEPS,
        E4_HYSTERESIS_EVAL_WINDOW,
    );
    let mut down_map = std::collections::HashMap::new();
    for (w, d) in &down_curve_desc {
        down_map.insert(float_key(*w), *d);
    }
    let mut down_bind_map = std::collections::HashMap::new();
    for (w, d) in &down_curve_bind_desc {
        down_bind_map.insert(float_key(*w), *d);
    }
    let mut down_curve = Vec::new();
    for &w in &weights_up {
        if let Some(v) = down_map.get(&float_key(w)).copied() {
            down_curve.push((w, v));
        }
    }
    let mut down_curve_bind = Vec::new();
    for &w in &weights_up {
        if let Some(v) = down_bind_map.get(&float_key(w)).copied() {
            down_curve_bind.push((w, v));
        }
    }

    let mut hysteresis_curve_csv = String::from("direction,seed,weight,delta_t_soft\n");
    for (w, d) in &up_curve {
        hysteresis_curve_csv.push_str(&format!("up,{},{:.3},{:.6}\n", E4_PROTOCOL_SEED, w, d));
    }
    for (w, d) in &down_curve {
        hysteresis_curve_csv.push_str(&format!("down,{},{:.3},{:.6}\n", E4_PROTOCOL_SEED, w, d));
    }
    write_with_log(
        out_dir.join("paper_e4_hysteresis_curve.csv"),
        hysteresis_curve_csv,
    )?;

    let mut hysteresis_diff_csv =
        String::from("weight,delta_t_up,delta_t_down,diff_up_minus_down\n");
    for (w, up) in &up_curve {
        let down = down_map.get(&float_key(*w)).copied().unwrap_or(0.0);
        hysteresis_diff_csv.push_str(&format!(
            "{:.3},{:.6},{:.6},{:.6}\n",
            w,
            up,
            down,
            up - down
        ));
    }
    write_with_log(
        out_dir.join("paper_e4_hysteresis_diff.csv"),
        hysteresis_diff_csv,
    )?;
    let mut hysteresis_bind_curve_csv = String::from("direction,seed,weight,delta_bind\n");
    for (w, d) in &up_curve_bind {
        hysteresis_bind_curve_csv.push_str(&format!("up,{},{:.3},{:.6}\n", E4_PROTOCOL_SEED, w, d));
    }
    for (w, d) in &down_curve_bind {
        hysteresis_bind_curve_csv
            .push_str(&format!("down,{},{:.3},{:.6}\n", E4_PROTOCOL_SEED, w, d));
    }
    write_with_log(
        out_dir.join("paper_e4_hysteresis_bind_curve.csv"),
        hysteresis_bind_curve_csv,
    )?;
    let mut hysteresis_bind_diff_csv =
        String::from("weight,delta_bind_up,delta_bind_down,diff_up_minus_down\n");
    for (w, up) in &up_curve_bind {
        let down = down_bind_map.get(&float_key(*w)).copied().unwrap_or(0.0);
        hysteresis_bind_diff_csv.push_str(&format!(
            "{:.3},{:.6},{:.6},{:.6}\n",
            w,
            up,
            down,
            up - down
        ));
    }
    write_with_log(
        out_dir.join("paper_e4_hysteresis_bind_diff.csv"),
        hysteresis_bind_diff_csv,
    )?;
    render_e4_hysteresis_plot(
        &out_dir.join("paper_e4_hysteresis_curve.svg"),
        &up_curve,
        &down_curve,
    )?;
    render_e4_hysteresis_bind_plot(
        &out_dir.join("paper_e4_hysteresis_bind_curve.svg"),
        &up_curve_bind,
        &down_curve_bind,
    )?;
    Ok(())
}

fn e4_protocol_meta_diff_csv(
    anchor_hz: f32,
    primary_bin_st: f32,
    primary_eps_cents: f32,
) -> String {
    let e4 = e4_paper_meta();
    let mut out = String::from("field,e2_ref,e4_value,match\n");
    let e2_obs_window = E2_SWEEPS.saturating_sub(E2_BURN_IN);
    let rows = vec![
        (
            "anchor_hz",
            format!("{anchor_hz:.3}"),
            format!("{:.3}", e4.anchor_hz),
            (anchor_hz - e4.anchor_hz).abs() < 1e-6,
        ),
        (
            "n_agents",
            E2_N_AGENTS.to_string(),
            e4.voice_count.to_string(),
            E2_N_AGENTS == e4.voice_count,
        ),
        (
            "observation_window_steps",
            e2_obs_window.to_string(),
            E4_TAIL_WINDOW_STEPS.to_string(),
            e2_obs_window == E4_TAIL_WINDOW_STEPS as usize,
        ),
        (
            "primary_hist_bin_cents",
            format!("{:.1}", primary_bin_st * 100.0),
            format!("{:.1}", E4_PAPER_HIST_BIN_CENTS),
            (primary_bin_st * 100.0 - E4_PAPER_HIST_BIN_CENTS).abs() < 1e-6,
        ),
        (
            "primary_eps_cents",
            format!("{primary_eps_cents:.1}"),
            format!("{primary_eps_cents:.1}"),
            true,
        ),
        (
            "init_mode_label",
            E2_INIT_MODE.label().to_string(),
            "weighted_landscape_sample".to_string(),
            false,
        ),
        (
            "mirror_only_protocol_seed",
            E4_PROTOCOL_SEED.to_string(),
            E4_PROTOCOL_SEED.to_string(),
            true,
        ),
    ];
    for (field, e2_ref, e4_value, matched) in rows {
        out.push_str(&format!("{field},{e2_ref},{e4_value},{}\n", matched as u8));
    }
    out
}

fn e4_fixed_except_mirror_check_csv(
    records: &[E4RunRecord],
    tail_agents: &[E4TailAgentRow],
) -> String {
    let mut agent_counts_by_run: std::collections::HashMap<
        (i32, u64),
        std::collections::HashMap<u32, usize>,
    > = std::collections::HashMap::new();
    for row in tail_agents {
        let run_key = (float_key(row.mirror_weight), row.seed);
        let by_step = agent_counts_by_run.entry(run_key).or_default();
        *by_step.entry(row.step).or_default() += 1;
    }

    let mut run_agent_span: std::collections::HashMap<(i32, u64), (usize, usize)> =
        std::collections::HashMap::new();
    for (run_key, counts_by_step) in &agent_counts_by_run {
        if counts_by_step.is_empty() {
            continue;
        }
        let mut min_count = usize::MAX;
        let mut max_count = 0usize;
        for count in counts_by_step.values() {
            min_count = min_count.min(*count);
            max_count = max_count.max(*count);
        }
        run_agent_span.insert(*run_key, (min_count, max_count));
    }

    #[derive(Default)]
    struct Accum {
        mirrors: std::collections::BTreeSet<i32>,
        settings: std::collections::BTreeSet<(u32, u32, u32, &'static str)>,
        agent_min: Option<usize>,
        agent_max: Option<usize>,
    }
    let mut map: std::collections::HashMap<(&'static str, u64, i32, i32), Accum> =
        std::collections::HashMap::new();
    for record in records {
        let key = (
            record.count_mode,
            record.seed,
            float_key(record.bin_width),
            float_key(record.eps_cents),
        );
        let entry = map.entry(key).or_default();
        entry.mirrors.insert(float_key(record.mirror_weight));
        entry.settings.insert((
            record.steps_total,
            record.burn_in,
            record.tail_window,
            record.histogram_source,
        ));
        let run_key = (float_key(record.mirror_weight), record.seed);
        if let Some((n_min, n_max)) = run_agent_span.get(&run_key).copied() {
            entry.agent_min = Some(entry.agent_min.map_or(n_min, |v| v.min(n_min)));
            entry.agent_max = Some(entry.agent_max.map_or(n_max, |v| v.max(n_max)));
        }
    }
    let mut keys: Vec<_> = map.keys().copied().collect();
    keys.sort_by(|a, b| {
        a.0.cmp(b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.3.cmp(&b.3))
    });
    let mut out = String::from(
        "count_mode,seed,bin_width,eps_cents,n_mirror_values,n_setting_variants,agent_count_min,agent_count_max,pass_population_constant,pass_fixed_except_mirror\n",
    );
    for key in keys {
        let Some(acc) = map.get(&key) else {
            continue;
        };
        let agent_min = acc.agent_min.unwrap_or(0);
        let agent_max = acc.agent_max.unwrap_or(0);
        let pass_population_constant = agent_min > 0 && agent_min == agent_max;
        let pass_fixed_except_mirror = acc.settings.len() == 1 && pass_population_constant;
        out.push_str(&format!(
            "{},{},{:.3},{:.3},{},{},{},{},{},{}\n",
            key.0,
            key.1,
            float_from_key(key.2),
            float_from_key(key.3),
            acc.mirrors.len(),
            acc.settings.len(),
            agent_min,
            agent_max,
            pass_population_constant as u8,
            pass_fixed_except_mirror as u8
        ));
    }
    out
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
    let root = bitmap_root(out_path, (1400, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "E4 Interval Histograms (Overlay, cents PC)",
            ("sans-serif", 22),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1200.0f32, 0.0f32..(y_max * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("interval cents (pitch class)")
        .y_desc("probability mass")
        .x_labels(7)
        .draw()?;

    for &x in &[E4_CENTS_MIN3, E4_CENTS_MAJ3, E4_CENTS_P4, E4_CENTS_P5] {
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
    eps_cents: f32,
) -> Result<(), Box<dyn Error>> {
    let mut series: Vec<(f32, f32, f32, usize)> = summaries
        .iter()
        .filter(|s| {
            s.count_mode == "soft"
                && (s.bin_width - bin_width).abs() < 1e-6
                && (s.eps_cents - eps_cents).abs() < 1e-6
        })
        .map(|s| (s.mirror_weight, s.mean_delta, s.std_delta, s.n_runs))
        .collect();
    if series.is_empty() {
        return Ok(());
    }
    series.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for (_, mean, std, _) in &series {
        y_min = y_min.min(mean - std);
        y_max = y_max.max(mean + std);
    }
    if !y_min.is_finite() || !y_max.is_finite() || (y_max - y_min).abs() < 1e-6 {
        y_min = -0.1;
        y_max = 0.1;
    }
    let pad = 0.1 * (y_max - y_min);
    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let eps_label = fmt_eps(eps_cents);
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("E4 ΔT vs Mirror Weight (bw={bin_width:.2}, eps={eps_label})"),
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, (y_min - pad)..(y_max + pad))?;

    chart
        .configure_mesh()
        .x_desc("mirror weight")
        .y_desc("ΔT")
        .draw()?;

    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0.0, 0.0), (1.0, 0.0)],
        BLACK.mix(0.3),
    )))?;

    let cap = 0.01f32;
    for (w, mean, std, n_runs) in &series {
        let sd0 = mean - std;
        let sd1 = mean + std;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w, sd0), (*w, sd1)],
            BLACK.mix(0.2),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w - cap, sd0), (*w + cap, sd0)],
            BLACK.mix(0.2),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w - cap, sd1), (*w + cap, sd1)],
            BLACK.mix(0.2),
        )))?;

        let se = if *n_runs > 1 {
            std / (*n_runs as f32).sqrt()
        } else {
            0.0
        };
        let t_crit = t_crit_975(n_runs.saturating_sub(1));
        let ci = t_crit * se;
        let y0 = mean - ci;
        let y1 = mean + ci;
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

    let xs: Vec<f32> = series.iter().map(|(w, _, _, _)| *w).collect();
    let ys: Vec<f32> = series.iter().map(|(_, mean, _, _)| *mean).collect();
    if let Some(fit) = linear_regression(&xs, &ys) {
        let y0 = fit.intercept + fit.slope * 0.0;
        let y1 = fit.intercept + fit.slope * 1.0;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(0.0, y0), (1.0, y1)],
            BLACK.mix(0.6),
        )))?;
    }

    root.present()?;
    Ok(())
}

fn render_e4_delta_spaghetti(
    out_path: &Path,
    run_records: &[E4RunRecord],
    summaries: &[E4SummaryRecord],
    bin_width: f32,
    eps_cents: f32,
) -> Result<(), Box<dyn Error>> {
    let mut by_seed: std::collections::HashMap<u64, Vec<(f32, f32)>> =
        std::collections::HashMap::new();
    for record in run_records {
        if record.count_mode != "soft" {
            continue;
        }
        if (record.bin_width - bin_width).abs() > 1e-6 {
            continue;
        }
        if (record.eps_cents - eps_cents).abs() > 1e-6 {
            continue;
        }
        by_seed
            .entry(record.seed)
            .or_default()
            .push((record.mirror_weight, record.delta));
    }
    if by_seed.is_empty() {
        return Ok(());
    }

    let mut mean_series: Vec<(f32, f32)> = summaries
        .iter()
        .filter(|s| {
            s.count_mode == "soft"
                && (s.bin_width - bin_width).abs() < 1e-6
                && (s.eps_cents - eps_cents).abs() < 1e-6
        })
        .map(|s| (s.mirror_weight, s.mean_delta))
        .collect();
    mean_series.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for points in by_seed.values() {
        for (_, y) in points {
            y_min = y_min.min(*y);
            y_max = y_max.max(*y);
        }
    }
    for (_, y) in &mean_series {
        y_min = y_min.min(*y);
        y_max = y_max.max(*y);
    }
    if !y_min.is_finite() || !y_max.is_finite() || (y_max - y_min).abs() < 1e-6 {
        y_min = -0.1;
        y_max = 0.1;
    }
    let pad = 0.1 * (y_max - y_min);

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let eps_label = fmt_eps(eps_cents);
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("E4 ΔT Spaghetti (bw={bin_width:.2}, eps={eps_label})"),
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, (y_min - pad)..(y_max + pad))?;

    chart
        .configure_mesh()
        .x_desc("mirror weight")
        .y_desc("ΔT")
        .draw()?;

    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0.0, 0.0), (1.0, 0.0)],
        BLACK.mix(0.3),
    )))?;

    let mut items: Vec<(u64, Vec<(f32, f32)>)> = by_seed.into_iter().collect();
    items.sort_by_key(|(seed, _)| *seed);
    for (i, (_seed, mut points)) in items.into_iter().enumerate() {
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let color = Palette99::pick(i).mix(0.35);
        chart.draw_series(LineSeries::new(points, &color))?;
    }

    if !mean_series.is_empty() {
        chart.draw_series(LineSeries::new(mean_series, &BLACK))?;
    }

    root.present()?;
    Ok(())
}

fn render_e4_major_minor_plot(
    out_path: &Path,
    summaries: &[E4SummaryRecord],
    bin_width: f32,
    eps_cents: f32,
) -> Result<(), Box<dyn Error>> {
    let mut series: Vec<(f32, f32, f32, f32, f32, usize)> = summaries
        .iter()
        .filter(|s| {
            s.count_mode == "soft"
                && (s.bin_width - bin_width).abs() < 1e-6
                && (s.eps_cents - eps_cents).abs() < 1e-6
        })
        .map(|s| {
            (
                s.mirror_weight,
                s.mean_major,
                s.std_major,
                s.mean_minor,
                s.std_minor,
                s.n_runs,
            )
        })
        .collect();
    if series.is_empty() {
        return Ok(());
    }
    series.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut y_max = series
        .iter()
        .map(|(_, m_major, s_major, m_minor, s_minor, _)| {
            (m_major + s_major).max(m_minor + s_minor)
        })
        .fold(0.0f32, f32::max);
    if y_max <= 0.0 {
        y_max = 1.0;
    }
    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let eps_label = fmt_eps(eps_cents);
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("E4 Triad Scores vs Mirror Weight (bw={bin_width:.2}, eps={eps_label})"),
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, 0.0f32..(y_max * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("mirror weight")
        .y_desc("triad score")
        .draw()?;

    let cap = 0.01f32;
    for (w, mean_major, std_major, mean_minor, std_minor, n_runs) in &series {
        let sd0 = mean_major - std_major;
        let sd1 = mean_major + std_major;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w, sd0), (*w, sd1)],
            BLUE.mix(0.2),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w - cap, sd0), (*w + cap, sd0)],
            BLUE.mix(0.2),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w - cap, sd1), (*w + cap, sd1)],
            BLUE.mix(0.2),
        )))?;

        let se = if *n_runs > 1 {
            std_major / (*n_runs as f32).sqrt()
        } else {
            0.0
        };
        let t_crit = t_crit_975(n_runs.saturating_sub(1));
        let ci = t_crit * se;
        let y0 = mean_major - ci;
        let y1 = mean_major + ci;
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

        let sd0 = mean_minor - std_minor;
        let sd1 = mean_minor + std_minor;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w, sd0), (*w, sd1)],
            RED.mix(0.2),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w - cap, sd0), (*w + cap, sd0)],
            RED.mix(0.2),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w - cap, sd1), (*w + cap, sd1)],
            RED.mix(0.2),
        )))?;

        let se = if *n_runs > 1 {
            std_minor / (*n_runs as f32).sqrt()
        } else {
            0.0
        };
        let t_crit = t_crit_975(n_runs.saturating_sub(1));
        let ci = t_crit * se;
        let y0 = mean_minor - ci;
        let y1 = mean_minor + ci;
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
    eps_cents: f32,
) -> Result<(), Box<dyn Error>> {
    let mut series: Vec<(f32, f32, f32, f32, f32, usize)> = summaries
        .iter()
        .filter(|s| {
            s.count_mode == "soft"
                && (s.bin_width - bin_width).abs() < 1e-6
                && (s.eps_cents - eps_cents).abs() < 1e-6
        })
        .map(|s| {
            (
                s.mirror_weight,
                s.mean_maj3,
                s.std_maj3,
                s.mean_min3,
                s.std_min3,
                s.n_runs,
            )
        })
        .collect();
    if series.is_empty() {
        return Ok(());
    }
    series.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut y_max = series
        .iter()
        .map(|(_, m_maj, s_maj, m_min, s_min, _)| (m_maj + s_maj).max(m_min + s_min))
        .fold(0.0f32, f32::max);
    if y_max <= 0.0 {
        y_max = 1.0;
    }
    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let eps_label = fmt_eps(eps_cents);
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("E4 Third Mass vs Mirror Weight (bw={bin_width:.2}, eps={eps_label})"),
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
    for (w, mean_maj, std_maj, mean_min, std_min, n_runs) in &series {
        let sd0 = mean_maj - std_maj;
        let sd1 = mean_maj + std_maj;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w, sd0), (*w, sd1)],
            BLUE.mix(0.2),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w - cap, sd0), (*w + cap, sd0)],
            BLUE.mix(0.2),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w - cap, sd1), (*w + cap, sd1)],
            BLUE.mix(0.2),
        )))?;

        let se = if *n_runs > 1 {
            std_maj / (*n_runs as f32).sqrt()
        } else {
            0.0
        };
        let t_crit = t_crit_975(n_runs.saturating_sub(1));
        let ci = t_crit * se;
        let y0 = mean_maj - ci;
        let y1 = mean_maj + ci;
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

        let sd0 = mean_min - std_min;
        let sd1 = mean_min + std_min;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w, sd0), (*w, sd1)],
            RED.mix(0.2),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w - cap, sd0), (*w + cap, sd0)],
            RED.mix(0.2),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(*w - cap, sd1), (*w + cap, sd1)],
            RED.mix(0.2),
        )))?;

        let se = if *n_runs > 1 {
            std_min / (*n_runs as f32).sqrt()
        } else {
            0.0
        };
        let t_crit = t_crit_975(n_runs.saturating_sub(1));
        let ci = t_crit * se;
        let y0 = mean_min - ci;
        let y1 = mean_min + ci;
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
    eps_cents: f32,
) -> Result<(), Box<dyn Error>> {
    let mut series: Vec<(f32, f32, f32)> = summaries
        .iter()
        .filter(|s| {
            s.count_mode == "soft"
                && (s.bin_width - bin_width).abs() < 1e-6
                && (s.eps_cents - eps_cents).abs() < 1e-6
        })
        .map(|s| (s.mirror_weight, s.major_rate, s.minor_rate))
        .collect();
    if series.is_empty() {
        return Ok(());
    }
    series.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let eps_label = fmt_eps(eps_cents);
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("E4 Major/Minor Rate vs Mirror Weight (bw={bin_width:.2}, eps={eps_label}, tau={E4_DELTA_TAU:.2})"),
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, 0.0f32..1.0f32)?;

    chart
        .configure_mesh()
        .x_desc("mirror weight")
        .y_desc("rate (ΔT thresholded)")
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

fn render_e4_interval_heatmap(
    out_path: &Path,
    hist_records: &[E4HistRecord],
    bin_width: f32,
) -> Result<(), Box<dyn Error>> {
    let bin_width_cents = (bin_width * 100.0).max(1.0);
    let mut weights: Vec<f32> = hist_records
        .iter()
        .filter(|r| (r.bin_width - bin_width).abs() < 1e-6)
        .map(|r| r.mirror_weight)
        .collect();
    weights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    weights.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    if weights.is_empty() {
        return Ok(());
    }

    let mut series: Vec<(f32, Histogram)> = Vec::new();
    for &weight in &weights {
        if let Some(hist) = mean_histogram_for_weight(hist_records, weight, bin_width) {
            series.push((weight, hist));
        }
    }
    if series.is_empty() {
        return Ok(());
    }
    let series_weights: Vec<f32> = series.iter().map(|(w, _)| *w).collect();

    let mut max_mass = 0.0f32;
    for (_, hist) in &series {
        for &mass in &hist.masses {
            max_mass = max_mass.max(mass);
        }
    }
    max_mass = max_mass.max(1e-6);

    let root = bitmap_root(out_path, (1500, 820)).into_drawing_area();
    root.fill(&WHITE)?;
    let (heat_area, colorbar_area) = root.split_horizontally(1260);
    let mut chart = ChartBuilder::on(&heat_area)
        .caption(
            format!("E4 Interval Heatmap (bw={bin_width_cents:.1}c)"),
            ("sans-serif", 22),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, 0.0f32..1200.0f32)?;

    chart
        .configure_mesh()
        .x_desc("mirror weight")
        .y_desc("interval cents (pitch class)")
        .x_labels(6)
        .y_labels(7)
        .draw()?;

    let default_half = E4_WEIGHT_COARSE_STEP.max(0.05) * 0.5;
    for (idx, (weight, hist)) in series.iter().enumerate() {
        let left = if idx == 0 {
            if series_weights.len() > 1 {
                weight - 0.5 * (series_weights[1] - *weight)
            } else {
                weight - default_half
            }
        } else {
            0.5 * (series_weights[idx - 1] + *weight)
        };
        let right = if idx + 1 < series_weights.len() {
            0.5 * (*weight + series_weights[idx + 1])
        } else if series_weights.len() > 1 {
            *weight + 0.5 * (*weight - series_weights[idx - 1])
        } else {
            *weight + default_half
        };
        let x0 = left.clamp(0.0, 1.0);
        let x1 = right.clamp(0.0, 1.0);
        for (center, mass) in hist.bin_centers.iter().zip(hist.masses.iter()) {
            let y0 = center - 0.5 * bin_width_cents;
            let y1 = center + 0.5 * bin_width_cents;
            let t = (mass / max_mass).clamp(0.0, 1.0);
            let inv = 1.0 - t;
            let color = RGBColor(255, (255.0 * inv) as u8, (255.0 * inv) as u8);
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x0, y0), (x1, y1)],
                color.filled(),
            )))?;
        }
    }

    for (cents, label) in [
        (E4_CENTS_B2, "b2"),
        (E4_CENTS_MIN3, "m3"),
        (E4_CENTS_MAJ3, "M3"),
        (E4_CENTS_P4, "P4"),
        (E4_CENTS_P5, "P5"),
    ] {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(0.0, cents), (1.0, cents)],
            BLACK.mix(0.2),
        )))?;
        chart.draw_series(std::iter::once(Text::new(
            label,
            (0.01, cents + 8.0),
            ("sans-serif", 12).into_font().color(&BLACK.mix(0.7)),
        )))?;
    }

    let mut bar_chart = ChartBuilder::on(&colorbar_area)
        .margin_top(60)
        .margin_bottom(60)
        .margin_left(20)
        .margin_right(20)
        .x_label_area_size(0)
        .y_label_area_size(56)
        .build_cartesian_2d(0.0f32..1.0f32, 0.0f32..max_mass)?;
    bar_chart
        .configure_mesh()
        .x_labels(0)
        .disable_x_mesh()
        .disable_y_mesh()
        .y_desc("mass")
        .draw()?;
    let n_grad = 240usize;
    for i in 0..n_grad {
        let y0 = max_mass * i as f32 / n_grad as f32;
        let y1 = max_mass * (i + 1) as f32 / n_grad as f32;
        let t = (y1 / max_mass).clamp(0.0, 1.0);
        let inv = 1.0 - t;
        let color = RGBColor(255, (255.0 * inv) as u8, (255.0 * inv) as u8);
        bar_chart.draw_series(std::iter::once(Rectangle::new(
            [(0.0, y0), (1.0, y1)],
            color.filled(),
        )))?;
    }

    root.present()?;
    Ok(())
}

fn render_e4_kernel_gate(out_path: &Path, anchor_hz: f32) -> Result<(), Box<dyn Error>> {
    let space = Log2Space::new(80.0, 2000.0, 96);
    let root_hz = anchor_hz;
    let fifth_hz = root_hz * 1.5;
    let mut env_scan = vec![0.0f32; space.n_bins()];
    env_scan[nearest_bin(&space, root_hz)] += 1.0;
    env_scan[nearest_bin(&space, fifth_hz)] += 1.0;

    let mut points: Vec<(f32, f32)> = Vec::new();
    for weight in build_weight_grid(E4_WEIGHT_FINE_STEP) {
        let params = HarmonicityParams {
            mirror_weight: weight,
            ..HarmonicityParams::default()
        };
        let kernel = HarmonicityKernel::new(&space, params);
        let (h_scan, _) = kernel.potential_h_from_log2_spectrum(&env_scan, &space);
        let freq_m3 = root_hz * 2.0f32.powf(3.0 / 12.0);
        let freq_maj3 = root_hz * 2.0f32.powf(4.0 / 12.0);
        let h3 = sample_scan_linear_log2(&space, &h_scan, freq_m3);
        let h4 = sample_scan_linear_log2(&space, &h_scan, freq_maj3);
        points.push((weight, h4 - h3));
    }
    if points.is_empty() {
        return Ok(());
    }
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for (_, y) in &points {
        y_min = y_min.min(*y);
        y_max = y_max.max(*y);
    }
    if !y_min.is_finite() || !y_max.is_finite() || (y_max - y_min).abs() < 1e-6 {
        y_min = -0.1;
        y_max = 0.1;
    }
    let pad = 0.1 * (y_max - y_min);
    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "E4 Kernel Gate: H(+4)-H(+3) vs Mirror Weight",
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, (y_min - pad)..(y_max + pad))?;

    chart
        .configure_mesh()
        .x_desc("mirror weight")
        .y_desc("H(+4) - H(+3)")
        .draw()?;

    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0.0, 0.0), (1.0, 0.0)],
        BLACK.mix(0.3),
    )))?;

    chart.draw_series(LineSeries::new(points, &BLUE))?;
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

struct OneUpdateStats {
    moved: bool,
    accepted_worse: bool,
    abs_delta_semitones: f32,
    abs_delta_semitones_moved: f32,
    c_score_current: f32,
    c_score_chosen: f32,
    chosen_score: f32,
    chosen_repulsion: f32,
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

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
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
#[cfg(test)]
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
fn update_one_agent_scored_loo(
    agent_i: usize,
    indices: &mut [usize],
    prev_indices: &[usize],
    prev_log2: &[f32],
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
    update_allowed: bool,
    block_backtrack: bool,
    backtrack_targets: Option<&[usize]>,
    u01: f32,
    env_loo: &mut [f32],
    density_loo: &mut [f32],
) -> OneUpdateStats {
    let agent_idx = prev_indices[agent_i];
    env_loo.copy_from_slice(env_total);
    density_loo.copy_from_slice(density_total);
    env_loo[agent_idx] = (env_loo[agent_idx] - 1.0).max(0.0);
    let denom = du_scan[agent_idx].max(1e-12);
    density_loo[agent_idx] = (density_loo[agent_idx] - 1.0 / denom).max(0.0);
    let (c_score_scan, _, _, _) =
        compute_c_score_state_scans(space, workspace, env_loo, density_loo, du_scan);

    let current_log2 = log2_ratio_scan[agent_idx];
    let skip_repulsion = lambda <= 0.0;
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
    let c_score_current = c_score_scan[agent_idx];
    let current_score = score_sign * c_score_current - lambda * current_repulsion;

    let backtrack_target = if block_backtrack {
        backtrack_targets.and_then(|prev| prev.get(agent_i).copied())
    } else {
        None
    };

    let (chosen_idx, chosen_score, chosen_repulsion, accepted_worse) = if update_allowed {
        let start = (agent_idx as isize - k as isize).max(min_idx as isize) as usize;
        let end = (agent_idx as isize + k as isize).min(max_idx as isize) as usize;
        let mut best_idx = agent_idx;
        let mut best_score = f32::NEG_INFINITY;
        let mut best_repulsion = 0.0f32;
        let mut found_candidate = false;
        for cand in start..=end {
            if cand == agent_idx {
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
            if let Some(prev_idx) = backtrack_target
                && cand == prev_idx
                && (score - current_score) <= E2_BACKTRACK_ALLOW_EPS
            {
                continue;
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
                let (accept, accepted_worse) = metropolis_accept(delta, temperature, u01);
                if accept {
                    (best_idx, best_score, best_repulsion, accepted_worse)
                } else {
                    (agent_idx, current_score, current_repulsion, false)
                }
            } else {
                (agent_idx, current_score, current_repulsion, false)
            }
        } else {
            (agent_idx, current_score, current_repulsion, false)
        }
    } else {
        (agent_idx, current_score, current_repulsion, false)
    };

    indices[agent_i] = chosen_idx;
    let moved = chosen_idx != agent_idx;
    let delta_semitones = 12.0 * (log2_ratio_scan[chosen_idx] - log2_ratio_scan[agent_idx]);
    let abs_delta = delta_semitones.abs();
    let abs_delta_moved = if moved { abs_delta } else { 0.0 };
    let c_score_chosen = c_score_scan[chosen_idx];

    OneUpdateStats {
        moved,
        accepted_worse,
        abs_delta_semitones: abs_delta,
        abs_delta_semitones_moved: abs_delta_moved,
        c_score_current,
        c_score_chosen,
        chosen_score,
        chosen_repulsion,
    }
}

#[allow(clippy::too_many_arguments)]
fn update_e2_sweep_scored_loo(
    schedule: E2UpdateSchedule,
    indices: &mut [usize],
    prev_indices: &[usize],
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
    sweep: usize,
    block_backtrack: bool,
    backtrack_targets: Option<&[usize]>,
    rng: &mut StdRng,
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
    let mut order: Vec<usize> = (0..indices.len()).collect();
    if matches!(schedule, E2UpdateSchedule::RandomSingle) {
        order.shuffle(rng);
    }
    let u01_by_agent: Vec<f32> = (0..indices.len()).map(|_| rng.random::<f32>()).collect();
    let u_move_by_agent: Vec<f32> = if matches!(schedule, E2UpdateSchedule::Lazy) {
        (0..indices.len()).map(|_| rng.random::<f32>()).collect()
    } else {
        vec![0.0; indices.len()]
    };
    let prev_log2: Vec<f32> = prev_indices
        .iter()
        .map(|&idx| log2_ratio_scan[idx])
        .collect();
    let mut env_loo = vec![0.0f32; env_total.len()];
    let mut density_loo = vec![0.0f32; density_total.len()];
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

    for &agent_i in &order {
        let update_allowed = match schedule {
            E2UpdateSchedule::Checkerboard => (agent_i + sweep).is_multiple_of(2),
            E2UpdateSchedule::Lazy => u_move_by_agent[agent_i] < E2_LAZY_MOVE_PROB.clamp(0.0, 1.0),
            E2UpdateSchedule::RandomSingle => true,
        };
        if update_allowed {
            attempt_count += 1;
        }
        let stats = update_one_agent_scored_loo(
            agent_i,
            indices,
            prev_indices,
            &prev_log2,
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
            update_allowed,
            block_backtrack,
            backtrack_targets,
            u01_by_agent[agent_i],
            &mut env_loo,
            &mut density_loo,
        );
        if stats.moved {
            moved_count += 1;
        }
        if stats.accepted_worse {
            accepted_worse_count += 1;
        }
        if stats.abs_delta_semitones.is_finite() {
            abs_delta_sum += stats.abs_delta_semitones;
            abs_delta_moved_sum += stats.abs_delta_semitones_moved;
        }
        if stats.c_score_current.is_finite() {
            c_score_current_sum += stats.c_score_current;
            c_score_current_count += 1;
        }
        if stats.c_score_chosen.is_finite() {
            c_score_chosen_sum += stats.c_score_chosen;
            c_score_chosen_count += 1;
        }
        if stats.chosen_score.is_finite() {
            score_sum += stats.chosen_score;
            score_count += 1;
        }
        if stats.chosen_repulsion.is_finite() {
            repulsion_sum += stats.chosen_repulsion;
            repulsion_count += 1;
        }
    }

    let n = indices.len() as f32;
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
    let mean_abs_delta_semitones = abs_delta_sum / n;
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
        moved_frac: moved_count as f32 / n,
        accepted_worse_frac: accepted_worse_count as f32 / n,
        attempted_update_frac: attempt_count as f32 / n,
        moved_given_attempt_frac: if attempt_count > 0 {
            moved_count as f32 / attempt_count as f32
        } else {
            0.0
        },
        mean_abs_delta_semitones,
        mean_abs_delta_semitones_moved,
    }
}

#[allow(clippy::too_many_arguments)]
fn update_e2_sweep_nohill(
    schedule: E2UpdateSchedule,
    indices: &mut [usize],
    prev_indices: &[usize],
    log2_ratio_scan: &[f32],
    min_idx: usize,
    max_idx: usize,
    k: i32,
    sweep: usize,
    rng: &mut StdRng,
) -> (usize, usize, f32, f32) {
    if indices.is_empty() {
        return (0, 0, 0.0, 0.0);
    }
    let mut order: Vec<usize> = (0..indices.len()).collect();
    if matches!(schedule, E2UpdateSchedule::RandomSingle) {
        order.shuffle(rng);
    }
    let u_move_by_agent: Vec<f32> = if matches!(schedule, E2UpdateSchedule::Lazy) {
        (0..indices.len()).map(|_| rng.random::<f32>()).collect()
    } else {
        vec![0.0; indices.len()]
    };
    let mut attempt_count = 0usize;
    let mut moved_count = 0usize;
    let mut abs_delta_sum = 0.0f32;
    let mut abs_delta_moved_sum = 0.0f32;
    for &agent_i in &order {
        let update_allowed = match schedule {
            E2UpdateSchedule::Checkerboard => (agent_i + sweep).is_multiple_of(2),
            E2UpdateSchedule::Lazy => u_move_by_agent[agent_i] < E2_LAZY_MOVE_PROB.clamp(0.0, 1.0),
            E2UpdateSchedule::RandomSingle => true,
        };
        let current_idx = prev_indices[agent_i];
        let next_idx = if update_allowed {
            attempt_count += 1;
            let step = rng.random_range(-k..=k);
            (current_idx as i32 + step).clamp(min_idx as i32, max_idx as i32) as usize
        } else {
            current_idx
        };
        indices[agent_i] = next_idx;
        if next_idx != current_idx {
            moved_count += 1;
        }
        let delta_semitones = 12.0 * (log2_ratio_scan[next_idx] - log2_ratio_scan[current_idx]);
        let abs_delta = delta_semitones.abs();
        if abs_delta.is_finite() {
            abs_delta_sum += abs_delta;
            if next_idx != current_idx {
                abs_delta_moved_sum += abs_delta;
            }
        }
    }
    (
        moved_count,
        attempt_count,
        abs_delta_sum,
        abs_delta_moved_sum,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
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

#[allow(dead_code)]
fn mean_c_score_loo_at_indices_with_prev(
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    env_total: &[f32],
    density_total: &[f32],
    du_scan: &[f32],
    prev_indices: &[usize],
    eval_indices: &[usize],
) -> f32 {
    let mut env_loo = Vec::new();
    let mut density_loo = Vec::new();
    mean_c_score_loo_at_indices_with_prev_reused(
        space,
        workspace,
        env_total,
        density_total,
        du_scan,
        prev_indices,
        eval_indices,
        &mut env_loo,
        &mut density_loo,
    )
}

#[allow(clippy::too_many_arguments)]
fn mean_c_score_loo_at_indices_with_prev_reused(
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    env_total: &[f32],
    density_total: &[f32],
    du_scan: &[f32],
    prev_indices: &[usize],
    eval_indices: &[usize],
    env_loo: &mut Vec<f32>,
    density_loo: &mut Vec<f32>,
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

    env_loo.resize(env_total.len(), 0.0);
    density_loo.resize(density_total.len(), 0.0);
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for (&prev_idx, &eval_idx) in prev_indices.iter().zip(eval_indices.iter()) {
        env_loo.copy_from_slice(env_total);
        density_loo.copy_from_slice(density_total);
        env_loo[prev_idx] = (env_loo[prev_idx] - 1.0).max(0.0);
        let denom = du_scan[prev_idx].max(1e-12);
        density_loo[prev_idx] = (density_loo[prev_idx] - 1.0 / denom).max(0.0);
        let (c_score_scan, _, _, _) =
            compute_c_score_state_scans(space, workspace, env_loo, density_loo, du_scan);
        let value = c_score_scan[eval_idx];
        if value.is_finite() {
            sum += value;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

#[cfg(test)]
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
            E2_SWEEPS,
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
        let trimmed = &traj[(start_step + 1)..=end];
        for pair in trimmed.windows(2) {
            let delta = pair[1] - pair[0];
            let moved = delta.abs() > E2_SEMITONE_EPS;
            step_count += 1;
            if moved {
                moved_step_count += 1;
            }
        }

        let mut compressed: Vec<f32> = Vec::new();
        for &v in &traj[start_step..=end] {
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
        if post_start < E2_SWEEPS {
            segments.push(("post", post_start, E2_SWEEPS.saturating_sub(1)));
        }
        segments
    } else {
        vec![("all", E2_BURN_IN, E2_SWEEPS.saturating_sub(1))]
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

#[allow(clippy::too_many_arguments)]
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
    out.push_str(&format!("E2_SWEEPS={}\n", E2_SWEEPS));
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
        E2UpdateSchedule::RandomSingle => "random_single",
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

    let root = bitmap_root(out_path, (1200, 900)).into_drawing_area();
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
    area: &DrawingArea<SVGBackend, Shift>,
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

    let root = bitmap_root(out_path, (1200, 900)).into_drawing_area();
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

fn render_diversity_summary_ci95_plot(
    out_path: &Path,
    rows: &[DiversityRow],
) -> Result<(), Box<dyn Error>> {
    let root = bitmap_root(out_path, (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((2, 2));
    draw_diversity_metric_panel(
        &panels[0],
        "E2 Diversity (95% CI): Unique Bins",
        "unique bins",
        rows,
        |metrics| metrics.unique_bins as f32,
    )?;
    draw_diversity_metric_panel(
        &panels[1],
        "E2 Diversity (95% CI): NN Distance",
        "nn mean (st)",
        rows,
        |metrics| metrics.nn_mean,
    )?;
    draw_diversity_metric_panel(
        &panels[2],
        "E2 Diversity (95% CI): Variance",
        "var (st^2)",
        rows,
        |metrics| metrics.semitone_var,
    )?;
    draw_diversity_metric_panel(
        &panels[3],
        "E2 Diversity (95% CI): MAD",
        "MAD (st)",
        rows,
        |metrics| metrics.semitone_mad,
    )?;
    root.present()?;
    Ok(())
}

fn draw_diversity_panel(
    area: &DrawingArea<SVGBackend, Shift>,
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
    if values.len().is_multiple_of(2) {
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

fn pairwise_intervals_csv(intervals: &[f32]) -> String {
    let mut out = String::from("interval_semitones\n");
    for &interval in intervals {
        out.push_str(&format!("{interval:.6}\n"));
    }
    out
}

fn pairwise_interval_histogram_csv(
    hist: &[(f32, f32)],
    intervals_total: usize,
    bin_width: f32,
) -> String {
    let total = intervals_total as f32;
    let inv_total = if total > 0.0 { 1.0 / total } else { 0.0 };
    let mut out =
        format!("# source=pairwise_intervals bin_width={bin_width:.3} n_pairs={intervals_total}\n");
    out.push_str("bin_center,count,frac\n");
    for &(center, count) in hist {
        let frac = count * inv_total;
        out.push_str(&format!("{center:.4},{count:.6},{frac:.6}\n"));
    }
    out
}

fn emit_pairwise_interval_dumps_for_condition(
    out_dir: &Path,
    condition: &str,
    runs: &[E2Run],
) -> Result<(), Box<dyn Error>> {
    for run in runs {
        let intervals = pairwise_interval_samples(&run.final_semitones);
        let raw_path = out_dir.join(format!(
            "pairwise_intervals_{condition}_seed{}.csv",
            run.seed
        ));
        write_with_log(raw_path, pairwise_intervals_csv(&intervals))?;

        let hist = histogram_counts_fixed(&intervals, 0.0, 12.0, E2_PAIRWISE_BIN_ST);
        let hist_path = out_dir.join(format!(
            "pairwise_interval_hist_{condition}_seed{}.csv",
            run.seed
        ));
        write_with_log(
            hist_path,
            pairwise_interval_histogram_csv(&hist, intervals.len(), E2_PAIRWISE_BIN_ST),
        )?;
    }
    Ok(())
}

fn e2_pairwise_hist_controls_seed_sweep_csv(
    baseline: &HistSweepStats,
    nohill: &HistSweepStats,
    norep: &HistSweepStats,
    n_pairs: usize,
) -> String {
    let mut out = String::from(
        "bin_center,baseline_mean,baseline_std,nohill_mean,nohill_std,norep_mean,norep_std,n_seeds,n_pairs\n",
    );
    let len = baseline
        .centers
        .len()
        .min(baseline.mean_frac.len())
        .min(baseline.std_frac.len())
        .min(nohill.mean_frac.len())
        .min(nohill.std_frac.len())
        .min(norep.mean_frac.len())
        .min(norep.std_frac.len());
    for i in 0..len {
        out.push_str(&format!(
            "{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}\n",
            baseline.centers[i],
            baseline.mean_frac[i],
            baseline.std_frac[i],
            nohill.mean_frac[i],
            nohill.std_frac[i],
            norep.mean_frac[i],
            norep.std_frac[i],
            baseline.n,
            n_pairs
        ));
    }
    out
}

fn e2_pairwise_hist_controls_seed_sweep_ci95_csv(
    baseline: &HistSweepStats,
    nohill: &HistSweepStats,
    norep: &HistSweepStats,
    n_pairs: usize,
) -> String {
    let mut out = String::from(
        "bin_center,baseline_mean,baseline_ci95,nohill_mean,nohill_ci95,norep_mean,norep_ci95,n_seeds,n_pairs\n",
    );
    let len = baseline
        .centers
        .len()
        .min(baseline.mean_frac.len())
        .min(baseline.std_frac.len())
        .min(nohill.mean_frac.len())
        .min(nohill.std_frac.len())
        .min(norep.mean_frac.len())
        .min(norep.std_frac.len());
    for i in 0..len {
        out.push_str(&format!(
            "{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}\n",
            baseline.centers[i],
            baseline.mean_frac[i],
            ci95_from_std(baseline.std_frac[i], baseline.n),
            nohill.mean_frac[i],
            ci95_from_std(nohill.std_frac[i], nohill.n),
            norep.mean_frac[i],
            ci95_from_std(norep.std_frac[i], norep.n),
            baseline.n,
            n_pairs
        ));
    }
    out
}

fn ci95_half_width(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let std = mean_std_scalar(values).1;
    1.96 * std / (values.len() as f32).sqrt()
}

fn ci95_from_std(std: f32, n: usize) -> f32 {
    if n == 0 {
        return 0.0;
    }
    1.96 * std / (n as f32).sqrt()
}

fn std_series_to_ci95(std: &[f32], n: usize) -> Vec<f32> {
    std.iter().map(|&s| ci95_from_std(s, n)).collect()
}

fn sweep_csv_with_ci95(header: &str, mean: &[f32], std: &[f32], n: usize) -> String {
    let mut out = String::from(header);
    let len = mean.len().min(std.len());
    for i in 0..len {
        out.push_str(&format!(
            "{i},{:.6},{:.6},{:.6},{}\n",
            mean[i],
            std[i],
            ci95_from_std(std[i], n),
            n
        ));
    }
    out
}

fn e2_pairwise_hist_seed_sweep_ci95_csv(stats: &HistSweepStats, n_pairs: usize) -> String {
    let mut out = String::from(
        "bin_center,mean_frac,std_frac,ci95_frac,n_seeds,n_pairs,mean_count,std_count\n",
    );
    let len = stats
        .centers
        .len()
        .min(stats.mean_count.len())
        .min(stats.std_count.len())
        .min(stats.mean_frac.len())
        .min(stats.std_frac.len());
    for i in 0..len {
        out.push_str(&format!(
            "{:.4},{:.6},{:.6},{:.6},{},{},{:.6},{:.6}\n",
            stats.centers[i],
            stats.mean_frac[i],
            stats.std_frac[i],
            ci95_from_std(stats.std_frac[i], stats.n),
            stats.n,
            n_pairs,
            stats.mean_count[i],
            stats.std_count[i]
        ));
    }
    out
}

fn diversity_summary_ci95_csv(rows: &[DiversityRow]) -> String {
    let mut by_cond: std::collections::HashMap<&'static str, Vec<&DiversityRow>> =
        std::collections::HashMap::new();
    for row in rows {
        by_cond.entry(row.condition).or_default().push(row);
    }
    let mut out = String::from("cond,metric,mean,std,ci95,n\n");
    for cond in ["baseline", "nohill", "norep"] {
        let rows = by_cond.get(cond).cloned().unwrap_or_default();
        let unique_bins: Vec<f32> = rows.iter().map(|r| r.metrics.unique_bins as f32).collect();
        let nn_mean: Vec<f32> = rows.iter().map(|r| r.metrics.nn_mean).collect();
        let semitone_var: Vec<f32> = rows.iter().map(|r| r.metrics.semitone_var).collect();
        let semitone_mad: Vec<f32> = rows.iter().map(|r| r.metrics.semitone_mad).collect();
        for (metric, values) in [
            ("unique_bins", &unique_bins),
            ("nn_mean", &nn_mean),
            ("semitone_var", &semitone_var),
            ("semitone_mad", &semitone_mad),
        ] {
            let (mean, std) = mean_std_scalar(values);
            out.push_str(&format!(
                "{cond},{metric},{mean:.6},{std:.6},{:.6},{}\n",
                ci95_half_width(values),
                values.len()
            ));
        }
    }
    out
}

fn interval_distance_mod_12(value: f32, target: f32) -> f32 {
    (value - target)
        .abs()
        .min((value - (target + 12.0)).abs())
        .min((value - (target - 12.0)).abs())
}

fn consonant_mass_for_intervals(intervals: &[f32], targets: &[f32], window_st: f32) -> f32 {
    if intervals.is_empty() {
        return 0.0;
    }
    let hits = intervals
        .iter()
        .filter(|&&interval| {
            targets
                .iter()
                .any(|&target| interval_distance_mod_12(interval, target) <= window_st + 1e-6)
        })
        .count();
    hits as f32 / intervals.len() as f32
}

fn consonant_mass_rows_for_condition(
    condition: &'static str,
    runs: &[E2Run],
) -> Vec<ConsonantMassRow> {
    runs.iter()
        .map(|run| {
            let intervals = pairwise_interval_samples(&run.final_semitones);
            ConsonantMassRow {
                condition,
                seed: run.seed,
                mass_core: consonant_mass_for_intervals(
                    &intervals,
                    &E2_CONSONANT_TARGETS_CORE,
                    E2_CONSONANT_WINDOW_ST,
                ),
                mass_extended: consonant_mass_for_intervals(
                    &intervals,
                    &E2_CONSONANT_TARGETS_EXTENDED,
                    E2_CONSONANT_WINDOW_ST,
                ),
            }
        })
        .collect()
}

fn consonant_mass_values(
    rows: &[ConsonantMassRow],
    condition: &str,
    select: fn(&ConsonantMassRow) -> f32,
) -> Vec<f32> {
    rows.iter()
        .filter(|row| row.condition == condition)
        .map(select)
        .collect()
}

fn consonant_mass_core(row: &ConsonantMassRow) -> f32 {
    row.mass_core
}

fn consonant_mass_extended(row: &ConsonantMassRow) -> f32 {
    row.mass_extended
}

fn n_choose_k_capped(n: usize, k: usize, cap: u64) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    if k == 0 {
        return 1;
    }
    let cap_u128 = cap as u128;
    let capped = cap.saturating_add(1);
    let mut value = 1u128;
    for i in 1..=k {
        value = value * (n - k + i) as u128 / i as u128;
        if value > cap_u128 {
            return capped;
        }
    }
    value as u64
}

fn exact_permutation_pvalue_mean_diff(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 1.0;
    }
    let mean_a = a.iter().copied().sum::<f32>() / a.len() as f32;
    let mean_b = b.iter().copied().sum::<f32>() / b.len() as f32;
    let obs_abs = (mean_a - mean_b).abs();
    let mut pooled = Vec::with_capacity(a.len() + b.len());
    pooled.extend_from_slice(a);
    pooled.extend_from_slice(b);
    let n_total = pooled.len();
    let n_a = a.len();
    if n_a == 0 || n_a >= n_total {
        return 1.0;
    }
    let pooled_sum = pooled.iter().copied().sum::<f32>();
    #[allow(clippy::too_many_arguments)]
    fn recurse(
        values: &[f32],
        n_a: usize,
        start: usize,
        picked: usize,
        sum_a: f32,
        pooled_sum: f32,
        obs_abs: f32,
        extreme: &mut usize,
        total: &mut usize,
    ) {
        if picked == n_a {
            let n_b = values.len() - n_a;
            if n_b == 0 {
                return;
            }
            let mean_a = sum_a / n_a as f32;
            let mean_b = (pooled_sum - sum_a) / n_b as f32;
            if (mean_a - mean_b).abs() + 1e-8 >= obs_abs {
                *extreme += 1;
            }
            *total += 1;
            return;
        }
        let remaining_needed = n_a - picked;
        let upper = values.len().saturating_sub(remaining_needed);
        for i in start..=upper {
            recurse(
                values,
                n_a,
                i + 1,
                picked + 1,
                sum_a + values[i],
                pooled_sum,
                obs_abs,
                extreme,
                total,
            );
        }
    }
    let mut extreme = 0usize;
    let mut total = 0usize;
    recurse(
        &pooled,
        n_a,
        0,
        0,
        0.0,
        pooled_sum,
        obs_abs,
        &mut extreme,
        &mut total,
    );
    if total == 0 {
        1.0
    } else {
        (extreme as f32 + 1.0) / (total as f32 + 1.0)
    }
}

fn permutation_pvalue_mean_diff(
    a: &[f32],
    b: &[f32],
    max_exact: u64,
    mc_iters: usize,
    rng_seed: u64,
) -> (f32, &'static str, u64) {
    if a.is_empty() || b.is_empty() {
        return (1.0, "exact", 0);
    }
    let n_total = a.len() + b.len();
    let n_a = a.len();
    let n_b = b.len();
    if n_a == 0 || n_b == 0 {
        return (1.0, "exact", 0);
    }

    let n_combos = n_choose_k_capped(n_total, n_a.min(n_b), max_exact);
    if n_combos <= max_exact {
        return (exact_permutation_pvalue_mean_diff(a, b), "exact", n_combos);
    }

    let mean_a = a.iter().copied().sum::<f32>() / n_a as f32;
    let mean_b = b.iter().copied().sum::<f32>() / n_b as f32;
    let obs_abs = (mean_a - mean_b).abs();
    let mut pooled = Vec::with_capacity(n_total);
    pooled.extend_from_slice(a);
    pooled.extend_from_slice(b);
    let pooled_sum = pooled.iter().copied().sum::<f32>();

    let iters = mc_iters.max(1) as u64;
    let mut rng = StdRng::seed_from_u64(rng_seed);
    let mut picks: Vec<usize> = (0..n_total).collect();
    let mut extreme = 0u64;
    for _ in 0..iters {
        picks.shuffle(&mut rng);
        let sum_a: f32 = picks[..n_a].iter().map(|&idx| pooled[idx]).sum();
        let perm_mean_a = sum_a / n_a as f32;
        let perm_mean_b = (pooled_sum - sum_a) / n_b as f32;
        if (perm_mean_a - perm_mean_b).abs() + 1e-8 >= obs_abs {
            extreme += 1;
        }
    }
    let p = (extreme as f32 + 1.0) / (iters as f32 + 1.0);
    (p, "mc", iters)
}

fn cliffs_delta(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut greater = 0usize;
    let mut less = 0usize;
    for &x in a {
        for &y in b {
            if x > y + 1e-8 {
                greater += 1;
            } else if x + 1e-8 < y {
                less += 1;
            }
        }
    }
    (greater as f32 - less as f32) / (a.len() * b.len()) as f32
}

fn consonant_mass_by_seed_csv(rows: &[ConsonantMassRow]) -> String {
    let mut out = format!(
        "# window_st={:.3} targets_core=3|4|7 targets_extended=0|3|4|5|7|8|9\n",
        E2_CONSONANT_WINDOW_ST
    );
    out.push_str("seed,cond,mass_core_347,mass_extended_0345789\n");
    for row in rows {
        out.push_str(&format!(
            "{},{},{:.6},{:.6}\n",
            row.seed, row.condition, row.mass_core, row.mass_extended
        ));
    }
    out
}

fn consonant_mass_summary_csv(rows: &[ConsonantMassRow]) -> String {
    let mut out = String::from("metric,cond,mean,std,ci95,n\n");
    for (metric, select) in [
        (
            "core_347",
            consonant_mass_core as fn(&ConsonantMassRow) -> f32,
        ),
        (
            "extended_0345789",
            consonant_mass_extended as fn(&ConsonantMassRow) -> f32,
        ),
    ] {
        for cond in ["baseline", "nohill", "norep"] {
            let values = consonant_mass_values(rows, cond, select);
            let (mean, std) = mean_std_scalar(&values);
            out.push_str(&format!(
                "{metric},{cond},{mean:.6},{std:.6},{:.6},{}\n",
                ci95_half_width(&values),
                values.len()
            ));
        }
    }
    out
}

fn consonant_mass_stats_csv(rows: &[ConsonantMassRow]) -> String {
    let mut out = String::from(
        "metric,comparison,mean_diff,p_perm,method,n_perm,cliffs_delta,n_baseline,n_control\n",
    );
    for (metric_idx, (metric, select)) in [
        (
            "core_347",
            consonant_mass_core as fn(&ConsonantMassRow) -> f32,
        ),
        (
            "extended_0345789",
            consonant_mass_extended as fn(&ConsonantMassRow) -> f32,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let baseline = consonant_mass_values(rows, "baseline", select);
        for (comp_idx, (comp_label, comp_cond)) in [
            ("baseline_vs_nohill", "nohill"),
            ("baseline_vs_norep", "norep"),
        ]
        .into_iter()
        .enumerate()
        {
            let control = consonant_mass_values(rows, comp_cond, select);
            let mean_diff = mean_std_scalar(&baseline).0 - mean_std_scalar(&control).0;
            let rng_seed = E2_PERM_MC_SEED ^ ((metric_idx as u64) << 32) ^ comp_idx as u64;
            let (p_perm, method, n_perm) = permutation_pvalue_mean_diff(
                &baseline,
                &control,
                E2_PERM_MAX_EXACT_COMBOS,
                E2_PERM_MC_ITERS,
                rng_seed,
            );
            let delta = cliffs_delta(&baseline, &control);
            out.push_str(&format!(
                "{metric},{comp_label},{mean_diff:.6},{p_perm:.6},{method},{n_perm},{delta:.6},{},{}\n",
                baseline.len(),
                control.len()
            ));
        }
    }
    out
}

fn fold_hist_abs_semitones(
    centers: &[f32],
    mean_frac: &[f32],
    std_frac: &[f32],
    bin_width: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = centers.len().min(mean_frac.len()).min(std_frac.len());
    let mut by_abs: std::collections::BTreeMap<i32, (f32, f32)> = std::collections::BTreeMap::new();
    for i in 0..len {
        let key = (centers[i].abs() / bin_width).round() as i32;
        let entry = by_abs.entry(key).or_insert((0.0, 0.0));
        entry.0 += mean_frac[i];
        entry.1 += std_frac[i] * std_frac[i];
    }
    let mut out_centers = Vec::with_capacity(by_abs.len());
    let mut out_mean = Vec::with_capacity(by_abs.len());
    let mut out_std = Vec::with_capacity(by_abs.len());
    for (key, (mean_sum, std_sq_sum)) in by_abs {
        out_centers.push(key as f32 * bin_width);
        out_mean.push(mean_sum);
        out_std.push(std_sq_sum.max(0.0).sqrt());
    }
    (out_centers, out_mean, out_std)
}

fn folded_hist_csv(centers: &[f32], mean: &[f32], std: &[f32], n_seeds: usize) -> String {
    let mut out = String::from("abs_semitones,mean_frac,std_frac,n_seeds\n");
    let len = centers.len().min(mean.len()).min(std.len());
    for i in 0..len {
        out.push_str(&format!(
            "{:.4},{:.6},{:.6},{}\n",
            centers[i], mean[i], std[i], n_seeds
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

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
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

fn e2_condition_display(condition: &str) -> &'static str {
    match condition {
        "baseline" => "baseline",
        "nohill" => "no hill-climb",
        "norep" => "no repulsion",
        _ => "unknown",
    }
}

fn e2_condition_color(condition: &str) -> RGBColor {
    match condition {
        "baseline" => BLUE,
        "nohill" => RED,
        "norep" => GREEN,
        _ => BLACK,
    }
}

fn draw_e2_interval_guides_with_windows<DB: DrawingBackend>(
    chart: &mut ChartContext<DB, Cartesian2d<RangedCoordf32, RangedCoordf32>>,
    y_max: f32,
) -> Result<(), Box<dyn Error>>
where
    DB::ErrorType: 'static,
{
    for &target in &E2_CONSONANT_TARGETS_CORE {
        chart.draw_series(std::iter::once(Rectangle::new(
            [
                (target - E2_CONSONANT_WINDOW_ST, 0.0),
                (target + E2_CONSONANT_WINDOW_ST, y_max),
            ],
            RGBColor(240, 170, 60).mix(0.12).filled(),
        )))?;
    }
    for &x in &E2_CONSONANT_STEPS {
        let is_core = E2_CONSONANT_TARGETS_CORE
            .iter()
            .any(|&core| (core - x).abs() < 1e-6);
        let style = if is_core {
            ShapeStyle::from(&BLACK.mix(0.6)).stroke_width(2)
        } else {
            ShapeStyle::from(&BLACK.mix(0.3)).stroke_width(1)
        };
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, 0.0), (x, y_max)],
            style,
        )))?;
    }
    Ok(())
}

fn y_max_from_mean_err(mean: &[f32], err: &[f32]) -> f32 {
    let len = mean.len().min(err.len());
    let mut y_peak = 0.0f32;
    for i in 0..len {
        y_peak = y_peak.max(mean[i] + err[i].max(0.0));
    }
    (1.15 * y_peak.max(1e-4)).max(1e-4)
}

fn render_pairwise_histogram_paper(
    out_path: &Path,
    caption: &str,
    centers: &[f32],
    mean_frac: &[f32],
    std_frac: &[f32],
    bin_width: f32,
) -> Result<(), Box<dyn Error>> {
    if centers.is_empty() {
        return Ok(());
    }
    let len = centers.len().min(mean_frac.len()).min(std_frac.len());
    if len == 0 {
        return Ok(());
    }
    let x_min = centers.first().copied().unwrap_or(0.0) - 0.5 * bin_width;
    let x_max = centers.get(len - 1).copied().unwrap_or(12.0) + 0.5 * bin_width;
    let y_max = y_max_from_mean_err(&mean_frac[..len], &std_frac[..len]);

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, 0.0f32..y_max)?;
    chart
        .configure_mesh()
        .x_desc("semitones")
        .y_desc("mean fraction")
        .x_labels(25)
        .draw()?;

    draw_e2_interval_guides_with_windows(&mut chart, y_max)?;

    let half = bin_width * 0.45;
    for i in 0..len {
        let x = centers[i];
        let m = mean_frac[i];
        let s = std_frac[i];
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x - half, 0.0), (x + half, m)],
            BLUE.mix(0.65).filled(),
        )))?;
        let y0 = (m - s).max(0.0);
        let y1 = (m + s).min(y_max);
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, y0), (x, y1)],
            BLACK.mix(0.6),
        )))?;
    }

    root.present()?;
    Ok(())
}

fn render_pairwise_histogram_controls_overlay(
    out_path: &Path,
    caption: &str,
    centers: &[f32],
    baseline: &[f32],
    nohill: &[f32],
    norep: &[f32],
) -> Result<(), Box<dyn Error>> {
    if centers.is_empty() {
        return Ok(());
    }
    let len = centers
        .len()
        .min(baseline.len())
        .min(nohill.len())
        .min(norep.len());
    if len == 0 {
        return Ok(());
    }
    let bin_width = if len > 1 {
        (centers[1] - centers[0]).abs().max(1e-6)
    } else {
        E2_PAIRWISE_BIN_ST
    };
    let x_min = centers[0] - 0.5 * bin_width;
    let x_max = centers[len - 1] + 0.5 * bin_width;
    let mut y_peak = 0.0f32;
    for i in 0..len {
        y_peak = y_peak.max(baseline[i]).max(nohill[i]).max(norep[i]);
    }
    let y_max = (1.15 * y_peak.max(1e-4)).max(1e-4);

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, 0.0f32..y_max)?;
    chart
        .configure_mesh()
        .x_desc("semitones")
        .y_desc("mean fraction")
        .x_labels(25)
        .draw()?;

    draw_e2_interval_guides_with_windows(&mut chart, y_max)?;

    for (label, values, color) in [
        ("baseline", baseline, BLUE),
        ("no hill-climb", nohill, RED),
        ("no repulsion", norep, GREEN),
    ] {
        let line = centers
            .iter()
            .take(len)
            .copied()
            .zip(values.iter().take(len).copied());
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

#[allow(clippy::too_many_arguments)]
fn render_hist_mean_std_fraction_auto_y(
    out_path: &Path,
    caption: &str,
    centers: &[f32],
    mean: &[f32],
    std: &[f32],
    bin_width: f32,
    x_desc: &str,
    guide_lines: &[f32],
) -> Result<(), Box<dyn Error>> {
    if centers.is_empty() {
        return Ok(());
    }
    let len = centers.len().min(mean.len()).min(std.len());
    if len == 0 {
        return Ok(());
    }
    let x_min = centers[0] - 0.5 * bin_width;
    let x_max = centers[len - 1] + 0.5 * bin_width;
    let y_max = y_max_from_mean_err(&mean[..len], &std[..len]);

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, 0.0f32..y_max)?;
    chart
        .configure_mesh()
        .x_desc(x_desc)
        .y_desc("mean fraction")
        .x_labels(25)
        .draw()?;

    for &x in guide_lines {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, 0.0), (x, y_max)],
            BLACK.mix(0.25),
        )))?;
    }

    let half = bin_width * 0.45;
    for i in 0..len {
        let x = centers[i];
        let m = mean[i];
        let s = std[i];
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x - half, 0.0), (x + half, m)],
            BLUE.mix(0.65).filled(),
        )))?;
        let y0 = (m - s).max(0.0);
        let y1 = (m + s).min(y_max);
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, y0), (x, y1)],
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

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
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

fn diversity_values_for_condition(
    rows: &[DiversityRow],
    condition: &str,
    select: fn(&DiversityMetrics) -> f32,
) -> Vec<f32> {
    rows.iter()
        .filter(|row| row.condition == condition)
        .map(|row| select(&row.metrics))
        .collect()
}

fn draw_diversity_metric_panel(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    y_desc: &str,
    rows: &[DiversityRow],
    select: fn(&DiversityMetrics) -> f32,
) -> Result<(), Box<dyn Error>> {
    let conditions = ["baseline", "nohill", "norep"];
    let mut means = [0.0f32; 3];
    let mut ci95 = [0.0f32; 3];
    for (i, cond) in conditions.iter().enumerate() {
        let values = diversity_values_for_condition(rows, cond, select);
        means[i] = mean_std_scalar(&values).0;
        ci95[i] = ci95_half_width(&values);
    }
    let mut y_max = 0.0f32;
    for i in 0..3 {
        y_max = y_max.max(means[i] + ci95[i]);
    }
    y_max = (1.15 * y_max.max(1e-4)).max(1e-4);

    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 18))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(-0.5f32..2.5f32, 0.0f32..y_max)?;
    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("condition")
        .y_desc(y_desc)
        .x_labels(3)
        .x_label_formatter(&|x| {
            let idx = x.round() as isize;
            if (0..=2).contains(&idx) {
                e2_condition_display(conditions[idx as usize]).to_string()
            } else {
                String::new()
            }
        })
        .draw()?;

    for (i, cond) in conditions.iter().enumerate() {
        let center = i as f32;
        let x0 = center - 0.3;
        let x1 = center + 0.3;
        let color = e2_condition_color(cond);
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, means[i])],
            color.mix(0.7).filled(),
        )))?;
        let y0 = (means[i] - ci95[i]).max(0.0);
        let y1 = (means[i] + ci95[i]).min(y_max);
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(center, y0), (center, y1)],
            BLACK.mix(0.7),
        )))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn draw_e2_timeseries_controls_panel(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    baseline_mean: &[f32],
    baseline_std: &[f32],
    nohill_mean: &[f32],
    nohill_std: &[f32],
    norep_mean: &[f32],
    norep_std: &[f32],
    burn_in: usize,
    phase_switch_step: Option<usize>,
    x_min: usize,
    x_max: usize,
    draw_legend: bool,
) -> Result<(), Box<dyn Error>> {
    let x_hi = x_max.max(x_min + 1);
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for (mean, std) in [
        (baseline_mean, baseline_std),
        (nohill_mean, nohill_std),
        (norep_mean, norep_std),
    ] {
        let len = mean.len().min(std.len());
        for i in x_min..len.min(x_hi + 1) {
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
    let pad = ((y_max - y_min).abs() * 0.1).max(1e-3);
    y_min = (y_min - pad).max(0.0);
    y_max += pad;

    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 18))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(55)
        .build_cartesian_2d(x_min as f32..x_hi as f32, y_min..y_max)?;
    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc("mean C_state")
        .draw()?;

    if burn_in > x_min {
        let burn_x1 = burn_in.min(x_hi) as f32;
        if burn_x1 > x_min as f32 {
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x_min as f32, y_min), (burn_x1, y_max)],
                RGBColor(180, 180, 180).mix(0.15).filled(),
            )))?;
        }
    }
    if let Some(step) = phase_switch_step
        && step >= x_min
        && step <= x_hi
    {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(step as f32, y_min), (step as f32, y_max)],
            ShapeStyle::from(&BLACK.mix(0.55)).stroke_width(2),
        )))?;
        let y_text = y_max - 0.05 * (y_max - y_min);
        chart.draw_series(std::iter::once(Text::new(
            "phase switch".to_string(),
            (step as f32, y_text),
            ("sans-serif", 13).into_font().color(&BLACK),
        )))?;
    }

    for (label, mean, std, color) in [
        (
            "baseline",
            baseline_mean,
            baseline_std,
            e2_condition_color("baseline"),
        ),
        (
            "no hill-climb",
            nohill_mean,
            nohill_std,
            e2_condition_color("nohill"),
        ),
        (
            "no repulsion",
            norep_mean,
            norep_std,
            e2_condition_color("norep"),
        ),
    ] {
        let len = mean.len().min(std.len());
        if len <= x_min {
            continue;
        }
        let end = len.min(x_hi + 1);
        let mut band: Vec<(f32, f32)> = Vec::with_capacity((end - x_min) * 2);
        for i in x_min..end {
            band.push((i as f32, mean[i] + std[i]));
        }
        for i in (x_min..end).rev() {
            band.push((i as f32, mean[i] - std[i]));
        }
        chart.draw_series(std::iter::once(Polygon::new(
            band,
            color.mix(0.15).filled(),
        )))?;
        let line = (x_min..end).map(|i| (i as f32, mean[i]));
        chart
            .draw_series(LineSeries::new(line, color))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], color));
    }
    if draw_legend {
        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }
    Ok(())
}

fn draw_trajectory_panel(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    trajectories: &[Vec<f32>],
) -> Result<(), Box<dyn Error>> {
    if trajectories.is_empty() {
        return Ok(());
    }
    let steps = trajectories.iter().map(|tr| tr.len()).max().unwrap_or(0);
    if steps == 0 {
        return Ok(());
    }
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for trace in trajectories {
        for &v in trace {
            if v.is_finite() {
                y_min = y_min.min(v);
                y_max = y_max.max(v);
            }
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = -12.0;
        y_max = 12.0;
    }
    let pad = ((y_max - y_min).abs() * 0.1).max(1e-3);
    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 18))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(55)
        .build_cartesian_2d(
            0.0f32..(steps.saturating_sub(1) as f32).max(1.0),
            (y_min - pad)..(y_max + pad),
        )?;
    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc("semitones")
        .draw()?;
    for (i, trace) in trajectories.iter().enumerate() {
        if trace.is_empty() {
            continue;
        }
        let color = Palette99::pick(i).mix(0.5);
        let line = trace.iter().enumerate().map(|(step, &v)| (step as f32, v));
        chart.draw_series(LineSeries::new(line, color))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render_e2_mean_c_state_annotated(
    out_path: &Path,
    baseline_mean: &[f32],
    baseline_std: &[f32],
    nohill_mean: &[f32],
    nohill_std: &[f32],
    norep_mean: &[f32],
    norep_std: &[f32],
    burn_in: usize,
    phase_switch_step: Option<usize>,
) -> Result<(), Box<dyn Error>> {
    let len = baseline_mean
        .len()
        .min(baseline_std.len())
        .min(nohill_mean.len())
        .min(nohill_std.len())
        .min(norep_mean.len())
        .min(norep_std.len());
    if len == 0 {
        return Ok(());
    }
    let root = bitmap_root(out_path, (1200, 1000)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((2, 1));
    draw_e2_timeseries_controls_panel(
        &panels[0],
        "E2 Mean C_state (annotated controls, 95% CI)",
        baseline_mean,
        baseline_std,
        nohill_mean,
        nohill_std,
        norep_mean,
        norep_std,
        burn_in,
        phase_switch_step,
        0,
        len.saturating_sub(1),
        true,
    )?;
    let zoom_start = phase_switch_step
        .unwrap_or(burn_in)
        .min(len.saturating_sub(1));
    draw_e2_timeseries_controls_panel(
        &panels[1],
        "Post-switch zoom (95% CI)",
        baseline_mean,
        baseline_std,
        nohill_mean,
        nohill_std,
        norep_mean,
        norep_std,
        burn_in,
        phase_switch_step,
        zoom_start,
        len.saturating_sub(1),
        false,
    )?;
    root.present()?;
    Ok(())
}

fn draw_consonant_mass_panel(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    rows: &[ConsonantMassRow],
    select: fn(&ConsonantMassRow) -> f32,
) -> Result<(), Box<dyn Error>> {
    let conditions = ["baseline", "nohill", "norep"];
    let mut means = [0.0f32; 3];
    let mut ci95 = [0.0f32; 3];
    for (i, cond) in conditions.iter().enumerate() {
        let values = consonant_mass_values(rows, cond, select);
        means[i] = mean_std_scalar(&values).0;
        ci95[i] = ci95_half_width(&values);
    }
    let mut y_max = 0.0f32;
    for i in 0..3 {
        y_max = y_max.max(means[i] + ci95[i]);
    }
    y_max = (1.15 * y_max.max(1e-4)).max(1e-4);

    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 18))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(-0.5f32..2.5f32, 0.0f32..y_max)?;
    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("condition")
        .y_desc("consonant mass")
        .x_labels(3)
        .x_label_formatter(&|x| {
            let idx = x.round() as isize;
            if (0..=2).contains(&idx) {
                e2_condition_display(conditions[idx as usize]).to_string()
            } else {
                String::new()
            }
        })
        .draw()?;

    for (i, cond) in conditions.iter().enumerate() {
        let center = i as f32;
        let x0 = center - 0.3;
        let x1 = center + 0.3;
        let color = e2_condition_color(cond);
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, means[i])],
            color.mix(0.7).filled(),
        )))?;
        let y0 = (means[i] - ci95[i]).max(0.0);
        let y1 = (means[i] + ci95[i]).min(y_max);
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(center, y0), (center, y1)],
            BLACK.mix(0.7),
        )))?;
    }
    Ok(())
}

fn render_consonant_mass_summary_plot(
    out_path: &Path,
    rows: &[ConsonantMassRow],
) -> Result<(), Box<dyn Error>> {
    if rows.is_empty() {
        return Ok(());
    }
    let root = bitmap_root(out_path, (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((2, 1));
    draw_consonant_mass_panel(
        &panels[0],
        "Consonant interval mass T={3,4,7} (95% CI)",
        rows,
        |row| row.mass_core,
    )?;
    draw_consonant_mass_panel(
        &panels[1],
        "Consonant interval mass T={0,3,4,5,7,8,9} (95% CI)",
        rows,
        |row| row.mass_extended,
    )?;
    root.present()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render_anchor_hist_post_folded(
    out_path: &Path,
    centers: &[f32],
    mean: &[f32],
    std: &[f32],
    folded_centers: &[f32],
    folded_mean: &[f32],
    folded_std: &[f32],
    bin_width: f32,
) -> Result<(), Box<dyn Error>> {
    if centers.is_empty() || folded_centers.is_empty() {
        return Ok(());
    }
    let root = bitmap_root(out_path, (1400, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, 2));

    {
        let len = centers.len().min(mean.len()).min(std.len());
        let x_min = centers[0] - 0.5 * bin_width;
        let x_max = centers[len - 1] + 0.5 * bin_width;
        let mut y_peak = 0.0f32;
        for &v in &mean[..len] {
            y_peak = y_peak.max(v);
        }
        let y_max = (1.15 * y_peak.max(1e-4)).max(1e-4);

        let mut chart = ChartBuilder::on(&panels[0])
            .caption("Anchor intervals (post, mean frac)", ("sans-serif", 16))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(x_min..x_max, 0.0f32..y_max)?;
        chart
            .configure_mesh()
            .x_desc("semitones")
            .y_desc("mean fraction")
            .draw()?;

        for &x in &[-7.0f32, -4.0, -3.0, 3.0, 4.0, 7.0] {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x, 0.0), (x, y_max)],
                BLACK.mix(0.25),
            )))?;
        }
        let half = bin_width * 0.45;
        for i in 0..len {
            chart.draw_series(std::iter::once(Rectangle::new(
                [(centers[i] - half, 0.0), (centers[i] + half, mean[i])],
                BLUE.mix(0.65).filled(),
            )))?;
            let y0 = (mean[i] - std[i]).max(0.0);
            let y1 = (mean[i] + std[i]).min(y_max);
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(centers[i], y0), (centers[i], y1)],
                BLACK.mix(0.6),
            )))?;
        }
    }

    {
        let len = folded_centers
            .len()
            .min(folded_mean.len())
            .min(folded_std.len());
        let x_min = folded_centers[0] - 0.5 * bin_width;
        let x_max = folded_centers[len - 1] + 0.5 * bin_width;
        let mut y_peak = 0.0f32;
        for &v in &folded_mean[..len] {
            y_peak = y_peak.max(v);
        }
        let y_max = (1.15 * y_peak.max(1e-4)).max(1e-4);

        let mut chart = ChartBuilder::on(&panels[1])
            .caption("|Anchor intervals| folded to [0,12]", ("sans-serif", 16))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(x_min..x_max, 0.0f32..y_max)?;
        chart
            .configure_mesh()
            .x_desc("|semitones|")
            .y_desc("mean fraction")
            .draw()?;
        for &x in &[3.0f32, 4.0, 7.0] {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x, 0.0), (x, y_max)],
                BLACK.mix(0.25),
            )))?;
        }
        let half = bin_width * 0.45;
        for i in 0..len {
            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (folded_centers[i] - half, 0.0),
                    (folded_centers[i] + half, folded_mean[i]),
                ],
                BLUE.mix(0.65).filled(),
            )))?;
            let y0 = (folded_mean[i] - folded_std[i]).max(0.0);
            let y1 = (folded_mean[i] + folded_std[i]).min(y_max);
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(folded_centers[i], y0), (folded_centers[i], y1)],
                BLACK.mix(0.6),
            )))?;
        }
    }
    root.present()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render_e2_figure1(
    out_path: &Path,
    baseline_stats: &E2SweepStats,
    nohill_stats: &E2SweepStats,
    norep_stats: &E2SweepStats,
    baseline_ci95_c_state: &[f32],
    nohill_ci95_c_state: &[f32],
    norep_ci95_c_state: &[f32],
    diversity_rows: &[DiversityRow],
    trajectories: &[Vec<f32>],
    phase_mode: E2PhaseMode,
) -> Result<(), Box<dyn Error>> {
    let root = bitmap_root(out_path, (1400, 1000)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((2, 2));

    let len = baseline_stats
        .mean_c_state
        .len()
        .min(baseline_stats.std_c_state.len())
        .min(nohill_stats.mean_c_state.len())
        .min(nohill_stats.std_c_state.len())
        .min(norep_stats.mean_c_state.len())
        .min(norep_stats.std_c_state.len());
    draw_e2_timeseries_controls_panel(
        &panels[0],
        "E2-1(a) Mean consonance over time (95% CI)",
        &baseline_stats.mean_c_state,
        baseline_ci95_c_state,
        &nohill_stats.mean_c_state,
        nohill_ci95_c_state,
        &norep_stats.mean_c_state,
        norep_ci95_c_state,
        E2_BURN_IN,
        phase_mode.switch_step(),
        0,
        len.saturating_sub(1),
        true,
    )?;
    draw_diversity_metric_panel(
        &panels[1],
        "E2-1(b) Non-collapse: unique bins (95% CI)",
        "unique bins",
        diversity_rows,
        |metrics| metrics.unique_bins as f32,
    )?;
    draw_trajectory_panel(
        &panels[2],
        "E2-1(c) Representative seed trajectories",
        trajectories,
    )?;
    draw_diversity_metric_panel(
        &panels[3],
        "E2-1(b-alt) Non-collapse: NN distance (95% CI)",
        "NN distance (st)",
        diversity_rows,
        |metrics| metrics.nn_mean,
    )?;
    root.present()?;
    Ok(())
}

fn render_e2_figure2(
    out_path: &Path,
    pairwise_centers: &[f32],
    pairwise_baseline_mean: &[f32],
    pairwise_baseline_std: &[f32],
    pairwise_nohill_mean: &[f32],
    pairwise_norep_mean: &[f32],
    consonant_rows: &[ConsonantMassRow],
) -> Result<(), Box<dyn Error>> {
    if pairwise_centers.is_empty() {
        return Ok(());
    }
    let len = pairwise_centers
        .len()
        .min(pairwise_baseline_mean.len())
        .min(pairwise_baseline_std.len())
        .min(pairwise_nohill_mean.len())
        .min(pairwise_norep_mean.len());
    if len == 0 {
        return Ok(());
    }
    let root = bitmap_root(out_path, (1400, 1000)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((2, 2));

    {
        let x_min = pairwise_centers[0] - 0.5 * E2_PAIRWISE_BIN_ST;
        let x_max = pairwise_centers[len - 1] + 0.5 * E2_PAIRWISE_BIN_ST;
        let y_max = y_max_from_mean_err(
            &pairwise_baseline_mean[..len],
            &pairwise_baseline_std[..len],
        );
        let mut chart = ChartBuilder::on(&panels[0])
            .caption(
                "E2-2(a) Pairwise interval histogram (baseline, 95% CI)",
                ("sans-serif", 18),
            )
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(50)
            .build_cartesian_2d(x_min..x_max, 0.0f32..y_max)?;
        chart
            .configure_mesh()
            .x_desc("semitones")
            .y_desc("mean fraction")
            .draw()?;
        draw_e2_interval_guides_with_windows(&mut chart, y_max)?;
        let half = E2_PAIRWISE_BIN_ST * 0.45;
        for i in 0..len {
            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (pairwise_centers[i] - half, 0.0),
                    (pairwise_centers[i] + half, pairwise_baseline_mean[i]),
                ],
                BLUE.mix(0.65).filled(),
            )))?;
            let y0 = (pairwise_baseline_mean[i] - pairwise_baseline_std[i]).max(0.0);
            let y1 = (pairwise_baseline_mean[i] + pairwise_baseline_std[i]).min(y_max);
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(pairwise_centers[i], y0), (pairwise_centers[i], y1)],
                BLACK.mix(0.6),
            )))?;
        }
    }

    {
        let x_min = pairwise_centers[0] - 0.5 * E2_PAIRWISE_BIN_ST;
        let x_max = pairwise_centers[len - 1] + 0.5 * E2_PAIRWISE_BIN_ST;
        let mut y_peak = 0.0f32;
        for i in 0..len {
            y_peak = y_peak
                .max(pairwise_baseline_mean[i])
                .max(pairwise_nohill_mean[i])
                .max(pairwise_norep_mean[i]);
        }
        let y_max = (1.15 * y_peak.max(1e-4)).max(1e-4);
        let mut chart = ChartBuilder::on(&panels[1])
            .caption("E2-2(b) Pairwise controls overlay", ("sans-serif", 18))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(50)
            .build_cartesian_2d(x_min..x_max, 0.0f32..y_max)?;
        chart
            .configure_mesh()
            .x_desc("semitones")
            .y_desc("mean fraction")
            .draw()?;
        draw_e2_interval_guides_with_windows(&mut chart, y_max)?;
        for (label, values, color) in [
            ("baseline", pairwise_baseline_mean, BLUE),
            ("no hill-climb", pairwise_nohill_mean, RED),
            ("no repulsion", pairwise_norep_mean, GREEN),
        ] {
            let line = pairwise_centers
                .iter()
                .take(len)
                .copied()
                .zip(values.iter().take(len).copied());
            chart
                .draw_series(LineSeries::new(line, color))?
                .label(label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], color));
        }
        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }

    draw_consonant_mass_panel(
        &panels[2],
        "E2-2(c) Consonant mass T={3,4,7} (95% CI)",
        consonant_rows,
        |row| row.mass_core,
    )?;
    draw_consonant_mass_panel(
        &panels[3],
        "E2-2(c-alt) Consonant mass T={0,3,4,5,7,8,9} (95% CI)",
        consonant_rows,
        |row| row.mass_extended,
    )?;

    root.present()?;
    Ok(())
}

fn e2_c_snapshot(run: &E2Run) -> (f32, f32, f32) {
    let pre_idx = e2_pre_step();
    let post_idx = e2_post_step_for(E2_SWEEPS);
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

#[cfg(test)]
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

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
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

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
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

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
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

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
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

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
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

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
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
    x_desc: &str,
) -> Result<(), Box<dyn Error>> {
    let counts = histogram_counts(values, min, max, bin_width);
    let y_max = counts
        .iter()
        .map(|(_, count)| *count as f32)
        .fold(0.0f32, f32::max)
        .max(1.0);

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min..max, 0.0f32..(y_max * 1.1))?;

    chart
        .configure_mesh()
        .x_desc(x_desc)
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
    let post_end = E2_SWEEPS.saturating_sub(1);
    let ranges: Vec<(&str, usize, usize, &Vec<f32>)> = if e2_anchor_shift_enabled() {
        let pre_start = E2_BURN_IN;
        let pre_end = E2_ANCHOR_SHIFT_STEP.saturating_sub(1);
        let post_start = E2_ANCHOR_SHIFT_STEP;
        let post_end = E2_SWEEPS.saturating_sub(1);
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
                "paper_e2_interval_histogram_{}_bw{}.svg",
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
            render_interval_histogram(
                &out_path,
                &caption,
                values,
                -12.0,
                12.0,
                bin_width,
                "semitones",
            )?;
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

    let out_path = out_dir.join("paper_e2_interval_histogram_post_controls_bw0p50.svg");
    let root = bitmap_root(&out_path, (1200, 700)).into_drawing_area();
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
    area: &DrawingArea<SVGBackend, Shift>,
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
    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
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
    area: &DrawingArea<SVGBackend, Shift>,
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
    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
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
    let root = bitmap_root(out_path, (1400, 700)).into_drawing_area();
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
    let root = bitmap_root(out_path, (1400, 700)).into_drawing_area();
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

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
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

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
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

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
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
    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
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
    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
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
    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
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

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
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
    let root = bitmap_root(out_path, (900, 600)).into_drawing_area();
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
            if folded >= 12.0 - eps {
                folded = 0.0;
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
        assert_eq!(run.mean_c_score_loo_series.len(), E2_SWEEPS);
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
    fn pairwise_interval_fold_maps_octave_to_zero() {
        let semitones = [0.0f32, 12.0];
        let pairs = pairwise_interval_samples(&semitones);
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].abs() < 1e-6, "expected 0, got {}", pairs[0]);
    }

    #[test]
    fn pairwise_interval_octave_hits_zero_hist_bin() {
        let semitones = [0.0f32, 12.0];
        let pairs = pairwise_interval_samples(&semitones);
        let hist = histogram_counts_fixed(&pairs, 0.0, 12.0, E2_PAIRWISE_BIN_ST);
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].abs() < 1e-6);
        assert!(
            !hist.is_empty() && hist[0].1 > 0.5,
            "expected first bin to receive octave sample"
        );
        let tail = hist.last().map(|(_, c)| *c).unwrap_or(0.0);
        assert!(tail < 0.5, "expected last bin not to receive octave sample");
    }

    #[test]
    fn interval_distance_mod_12_maps_octave_equivalence() {
        let d = interval_distance_mod_12(12.0, 0.0);
        assert!(
            d.abs() < 1e-6,
            "expected octave equivalence distance 0, got {d}"
        );
        let d = interval_distance_mod_12(11.9, 0.0);
        assert!(
            (d - 0.1).abs() < 1e-4,
            "expected wrap-around distance 0.1, got {d}"
        );
    }

    #[test]
    fn consonant_mass_for_intervals_counts_target_windows() {
        let intervals = [3.05f32, 4.1, 6.0, 7.2, 11.9, 0.1, 12.0];
        let core_mass = consonant_mass_for_intervals(&intervals, &E2_CONSONANT_TARGETS_CORE, 0.25);
        assert!(
            (core_mass - 3.0 / 7.0).abs() < 1e-6,
            "core_mass={core_mass}"
        );

        let octave_mass = consonant_mass_for_intervals(&intervals, &[0.0], 0.25);
        assert!(
            (octave_mass - 3.0 / 7.0).abs() < 1e-6,
            "octave_mass={octave_mass}"
        );
    }

    #[test]
    fn fold_hist_abs_semitones_merges_sign_pairs() {
        let centers = [-1.0f32, 0.0, 1.0];
        let mean = [0.2f32, 0.1, 0.3];
        let std = [0.01f32, 0.02, 0.03];
        let (fold_c, fold_m, fold_s) = fold_hist_abs_semitones(&centers, &mean, &std, 1.0);
        assert_eq!(fold_c, vec![0.0, 1.0]);
        assert!((fold_m[0] - 0.1).abs() < 1e-6);
        assert!((fold_m[1] - 0.5).abs() < 1e-6);
        let expected_std = (0.01f32 * 0.01 + 0.03 * 0.03).sqrt();
        assert!((fold_s[0] - 0.02).abs() < 1e-6);
        assert!((fold_s[1] - expected_std).abs() < 1e-6);
    }

    #[test]
    fn exact_permutation_pvalue_mean_diff_small_sample() {
        let a = [1.0f32, 1.0, 1.0];
        let b = [0.0f32, 0.0, 0.0];
        let p = exact_permutation_pvalue_mean_diff(&a, &b);
        assert!((p - (3.0 / 21.0)).abs() < 1e-6, "unexpected exact p={p}");
    }

    #[test]
    fn permutation_pvalue_mean_diff_small_sample_uses_exact() {
        let a = [1.0f32, 1.0, 1.0];
        let b = [0.0f32, 0.0, 0.0];
        let p_exact = exact_permutation_pvalue_mean_diff(&a, &b);
        let (p, method, n_perm) = permutation_pvalue_mean_diff(&a, &b, 1_000, 2_000, 1234);
        assert_eq!(method, "exact");
        assert_eq!(n_perm, 20);
        assert!(
            (p - p_exact).abs() < 1e-6,
            "auto exact p={p} differed from exact p={p_exact}"
        );
    }

    #[test]
    fn permutation_pvalue_mean_diff_falls_back_to_mc_and_is_deterministic() {
        let a = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let b = [0.2f32, 0.3, 0.4, 0.5, 0.6, 0.7];
        let (p1, method1, n_perm1) = permutation_pvalue_mean_diff(&a, &b, 10, 1_234, 0xBEEF);
        let (p2, method2, n_perm2) = permutation_pvalue_mean_diff(&a, &b, 10, 1_234, 0xBEEF);
        assert_eq!(method1, "mc");
        assert_eq!(method2, "mc");
        assert_eq!(n_perm1, 1_234);
        assert_eq!(n_perm2, 1_234);
        assert!((0.0..=1.0).contains(&p1), "p1 out of range: {p1}");
        assert!((0.0..=1.0).contains(&p2), "p2 out of range: {p2}");
        assert_eq!(
            p1.to_bits(),
            p2.to_bits(),
            "mc p-value must be deterministic"
        );
    }

    #[test]
    fn n_choose_k_capped_returns_cap_plus_one_on_overflow_path() {
        assert_eq!(n_choose_k_capped(6, 3, 100), 20);
        assert_eq!(n_choose_k_capped(40, 20, 1_000), 1_001);
    }

    #[test]
    fn y_max_from_mean_err_uses_error_upper_bound() {
        let mean = [0.1f32];
        let err = [0.2f32];
        let y_max = y_max_from_mean_err(&mean, &err);
        assert!(y_max >= 0.3, "y_max={y_max}");
    }

    #[test]
    fn p5_class_distance_treats_p4_and_p5_as_same_class() {
        assert!(p5_class_distance_cents(E4_CENTS_P5) < 1e-6);
        assert!(p5_class_distance_cents(E4_CENTS_P4) < 1e-6);
        let d = p5_class_distance_cents(E4_CENTS_P5 + 10.0);
        assert!((d - 10.0).abs() < 1e-4, "unexpected distance: {d}");
    }

    #[test]
    fn triad_scores_use_product_and_normalized_delta_t() {
        let masses = IntervalMasses {
            min3: 0.2,
            maj3: 0.4,
            p4: 0.1,
            p5: 0.1,
            p5_class: 0.5,
        };
        let (t_maj, t_min, delta_t, major_frac) = triad_scores(masses);
        assert!((t_maj - 0.2).abs() < 1e-6);
        assert!((t_min - 0.1).abs() < 1e-6);
        let expected = (0.2 - 0.1) / (0.2 + 0.1 + 1e-6);
        assert!((delta_t - expected).abs() < 1e-6, "delta_t={delta_t}");
        assert!(
            (major_frac - (2.0 / 3.0)).abs() < 1e-6,
            "major_frac={major_frac}"
        );
    }

    #[test]
    fn soft_counting_gives_nonzero_weight_outside_hard_window() {
        let hist = Histogram {
            bin_centers: vec![E4_CENTS_MAJ3 + 20.0],
            masses: vec![1.0],
        };
        let hard = interval_masses_from_histogram(&hist, 12.5, E4CountMode::Hard);
        let soft = interval_masses_from_histogram(&hist, 12.5, E4CountMode::Soft);
        assert!(hard.maj3 <= 1e-9, "hard.maj3={}", hard.maj3);
        assert!(soft.maj3 > 0.0, "soft.maj3 should be > 0");
    }

    #[test]
    fn fixed_except_mirror_check_passes_for_constant_settings() {
        let r1 = E4RunRecord {
            count_mode: "soft",
            mirror_weight: 0.0,
            seed: 1,
            bin_width: 0.25,
            eps_cents: 25.0,
            major_score: 0.1,
            minor_score: 0.05,
            delta: 0.1,
            triad_major: 0.1,
            triad_minor: 0.05,
            delta_t: 0.1,
            major_frac: 0.6666667,
            mass_min3: 0.2,
            mass_maj3: 0.3,
            mass_p4: 0.1,
            mass_p5: 0.2,
            mass_p5_class: 0.25,
            steps_total: 1200,
            burn_in: 800,
            tail_window: 400,
            histogram_source: "tail_mean",
        };
        let r2 = E4RunRecord {
            mirror_weight: 1.0,
            ..r1
        };
        let tail_agents = vec![
            E4TailAgentRow {
                mirror_weight: 0.0,
                seed: 1,
                step: 1000,
                agent_id: 1,
                freq_hz: 220.0,
            },
            E4TailAgentRow {
                mirror_weight: 0.0,
                seed: 1,
                step: 1000,
                agent_id: 2,
                freq_hz: 330.0,
            },
            E4TailAgentRow {
                mirror_weight: 0.0,
                seed: 1,
                step: 1000,
                agent_id: 3,
                freq_hz: 440.0,
            },
            E4TailAgentRow {
                mirror_weight: 1.0,
                seed: 1,
                step: 1000,
                agent_id: 1,
                freq_hz: 220.0,
            },
            E4TailAgentRow {
                mirror_weight: 1.0,
                seed: 1,
                step: 1000,
                agent_id: 2,
                freq_hz: 330.0,
            },
            E4TailAgentRow {
                mirror_weight: 1.0,
                seed: 1,
                step: 1000,
                agent_id: 3,
                freq_hz: 440.0,
            },
        ];
        let csv = e4_fixed_except_mirror_check_csv(&[r1, r2], &tail_agents);
        let line = csv
            .lines()
            .nth(1)
            .expect("expected one data row in fixed-check csv");
        assert!(
            line.ends_with(",1"),
            "expected pass_fixed_except_mirror=1, got: {line}"
        );
    }

    #[test]
    fn bind_metrics_use_latest_step_and_are_deterministic() {
        let rows = vec![
            E4TailAgentRow {
                mirror_weight: 0.5,
                seed: 7,
                step: 10,
                agent_id: 1,
                freq_hz: 220.0,
            },
            E4TailAgentRow {
                mirror_weight: 0.5,
                seed: 7,
                step: 10,
                agent_id: 2,
                freq_hz: 330.0,
            },
            E4TailAgentRow {
                mirror_weight: 0.5,
                seed: 7,
                step: 20,
                agent_id: 1,
                freq_hz: 220.0,
            },
            E4TailAgentRow {
                mirror_weight: 0.5,
                seed: 7,
                step: 20,
                agent_id: 2,
                freq_hz: 330.0,
            },
            E4TailAgentRow {
                mirror_weight: 0.5,
                seed: 7,
                step: 20,
                agent_id: 3,
                freq_hz: 440.0,
            },
        ];
        let m1 = e4_bind_metrics_from_tail_agents(&rows);
        let m2 = e4_bind_metrics_from_tail_agents(&rows);
        assert_eq!(m1.len(), 1);
        assert_eq!(m2.len(), 1);
        assert_eq!(m1[0].step, 20);
        assert_eq!(m1[0].n_agents, 3);
        assert!((0.0..=3.0).contains(&m1[0].root_fit));
        assert!((0.0..=3.0).contains(&m1[0].ceiling_fit));
        assert!((-1.0..=1.0).contains(&m1[0].delta_bind));
        assert_eq!(m1[0].root_fit.to_bits(), m2[0].root_fit.to_bits());
        assert_eq!(m1[0].ceiling_fit.to_bits(), m2[0].ceiling_fit.to_bits());
        assert_eq!(m1[0].delta_bind.to_bits(), m2[0].delta_bind.to_bits());
    }

    #[test]
    fn bind_scores_are_scale_invariant() {
        let freqs = [220.0f32, 330.0, 440.0, 550.0];
        let scaled = [374.0f32, 561.0, 748.0, 935.0];
        let (r0, c0, d0) = bind_scores_from_freqs(&freqs);
        let (r1, c1, d1) = bind_scores_from_freqs(&scaled);
        assert!((r0 - r1).abs() < 1e-5, "root fit changed by scaling");
        assert!((c0 - c1).abs() < 1e-5, "ceiling fit changed by scaling");
        assert!((d0 - d1).abs() < 1e-5, "delta bind changed by scaling");
    }

    #[test]
    fn bind_scores_swap_sign_between_root_and_ceiling_regimes() {
        let harmonic_series = [1.0f32, 2.0, 3.0, 4.0];
        let ceiling_series = [1.0f32, 0.5, 1.0 / 3.0, 0.25];
        let (r_h, c_h, d_h) = bind_scores_from_freqs(&harmonic_series);
        let (r_c, c_c, d_c) = bind_scores_from_freqs(&ceiling_series);
        assert!(r_h > c_h, "harmonic series should favor RootFit");
        assert!(c_c > r_c, "reciprocal series should favor CeilingFit");
        assert!(d_h > 0.0, "delta bind should be positive for root regime");
        assert!(
            d_c < 0.0,
            "delta bind should be negative for ceiling regime"
        );
    }

    #[test]
    fn mirror_weight_prefers_expected_binding_family() {
        let meta = e4_paper_meta();
        let space = Log2Space::new(meta.fmin, meta.fmax, meta.bins_per_oct);
        let (_erb_scan, du_scan) = erb_grid_for_space(&space);

        let root_family = [
            meta.anchor_hz,
            meta.anchor_hz * (5.0 / 4.0),
            meta.anchor_hz * (3.0 / 2.0),
            meta.anchor_hz * 2.0,
        ];
        let ceiling_family = [
            meta.anchor_hz,
            meta.anchor_hz * (4.0 / 5.0),
            meta.anchor_hz * (2.0 / 3.0),
            meta.anchor_hz * 0.5,
        ];

        let scan_m0 =
            compute_e4_landscape_scans(&space, meta.anchor_hz, 1.0, 0.0, &root_family, &du_scan);
        let scan_m1 =
            compute_e4_landscape_scans(&space, meta.anchor_hz, 1.0, 1.0, &ceiling_family, &du_scan);
        let p_m0 = extract_peak_rows_from_c_scan(&space, meta.anchor_hz, &scan_m0, 12);
        let p_m1 = extract_peak_rows_from_c_scan(&space, meta.anchor_hz, &scan_m1, 12);
        assert!(
            !p_m0.is_empty() && !p_m1.is_empty(),
            "expected non-empty peak sets for synthetic families"
        );
        let f_m0: Vec<f32> = p_m0.iter().take(8).map(|p| p.freq_hz).collect();
        let f_m1: Vec<f32> = p_m1.iter().take(8).map(|p| p.freq_hz).collect();
        let eval_m0 = bind_eval_from_freqs(&f_m0);
        let eval_m1 = bind_eval_from_freqs(&f_m1);

        println!(
            "mw=0 root_fit={:.6} ceiling_fit={:.6} delta={:.6}",
            eval_m0.root_fit, eval_m0.ceiling_fit, eval_m0.delta_bind
        );
        println!(
            "mw=1 root_fit={:.6} ceiling_fit={:.6} delta={:.6}",
            eval_m1.root_fit, eval_m1.ceiling_fit, eval_m1.delta_bind
        );
        assert!(
            eval_m0.root_fit > eval_m0.ceiling_fit,
            "mw=0 should favor root-binding"
        );
        assert!(
            eval_m1.ceiling_fit > eval_m1.root_fit,
            "mw=1 should favor ceiling-binding"
        );
    }

    #[test]
    fn oracle_and_agent_share_same_bind_evaluator() {
        let peaks = vec![
            E4PeakRow {
                rank: 1,
                bin_idx: 0,
                log2_ratio: 0.0,
                semitones: 0.0,
                freq_hz: 220.0,
                c_value: 1.0,
                prominence: 0.8,
                width_st: 0.3,
            },
            E4PeakRow {
                rank: 2,
                bin_idx: 1,
                log2_ratio: 0.3219281,
                semitones: 3.8631372,
                freq_hz: 275.0,
                c_value: 0.9,
                prominence: 0.7,
                width_st: 0.3,
            },
            E4PeakRow {
                rank: 3,
                bin_idx: 2,
                log2_ratio: 0.5849625,
                semitones: 7.01955,
                freq_hz: 330.0,
                c_value: 0.85,
                prominence: 0.7,
                width_st: 0.3,
            },
            E4PeakRow {
                rank: 4,
                bin_idx: 3,
                log2_ratio: 1.0,
                semitones: 12.0,
                freq_hz: 440.0,
                c_value: 0.8,
                prominence: 0.6,
                width_st: 0.3,
            },
        ];
        let freqs: Vec<f32> = peaks.iter().take(4).map(|p| p.freq_hz).collect();
        let oracle_delta = oracle1_delta_bind_from_peaks(&peaks, 4);
        let eval_delta = bind_eval_from_freqs(&freqs).delta_bind;
        assert!(
            (oracle_delta - eval_delta).abs() < 1e-6,
            "oracle_delta={oracle_delta} eval_delta={eval_delta}"
        );
    }

    #[test]
    fn oracle_chord_delta_is_stable_under_noop_update() {
        let meta = e4_paper_meta();
        let space = Log2Space::new(meta.fmin, meta.fmax, meta.bins_per_oct);
        let (_erb_scan, du_scan) = erb_grid_for_space(&space);
        let freqs = [
            meta.anchor_hz,
            meta.anchor_hz * (5.0 / 4.0),
            meta.anchor_hz * (3.0 / 2.0),
            meta.anchor_hz * 2.0,
        ];
        let scan = compute_e4_landscape_scans(&space, meta.anchor_hz, 1.0, 0.0, &freqs, &du_scan);
        let mut indices: Vec<usize> = freqs
            .iter()
            .map(|f| {
                space
                    .index_of_freq(*f)
                    .unwrap_or_else(|| nearest_bin(&space, *f))
            })
            .collect();
        let before = bind_eval_from_indices(meta.anchor_hz, &scan.log2_ratio, &indices).delta_bind;
        e4_wr_probe_update_indices(
            &mut indices,
            &scan.c,
            &scan.log2_ratio,
            0,
            scan.c.len().saturating_sub(1),
            0,
            0.0,
            E4_DYNAMICS_REPULSION_SIGMA,
            None,
            true,
        );
        let after = bind_eval_from_indices(meta.anchor_hz, &scan.log2_ratio, &indices).delta_bind;
        assert!(
            (before - after).abs() < 1e-6,
            "noop update should keep delta_bind (before={before}, after={after})"
        );
    }

    #[test]
    fn fingerprint_maps_octave_to_8ve_bin() {
        let probs = e4_interval_fingerprint_probs(&[220.0, 440.0], E4_FINGERPRINT_TOL_CENTS);
        let idx_oct = e4_fingerprint_label_index("8ve");
        let idx_other = e4_fingerprint_label_index("other");
        assert!(probs[idx_oct] > 0.99, "octave bin was not selected");
        assert!(probs[idx_other] < 1e-6, "octave should not map to other");
    }

    #[test]
    fn ci95_from_std_matches_closed_form() {
        let ci = ci95_from_std(0.5, 20);
        let expected = 1.96 * 0.5 / (20.0f32).sqrt();
        assert!((ci - expected).abs() < 1e-7, "ci={ci} expected={expected}");
    }

    #[test]
    fn wr_t_ci95_is_not_plain_se_for_n2() {
        let values = [1.0f32, 3.0];
        let (_mean, se, lo, hi) = mean_se_t_ci95(&values);
        let ci95 = 0.5 * (hi - lo);
        let expected = 12.706 * se;
        assert!(
            (ci95 - expected).abs() < 1e-3,
            "expected t-based CI95 half-width; got ci95={ci95}, expected={expected}, se={se}"
        );
    }

    #[test]
    fn wr_summary_csv_matches_run_aggregation() {
        let runs = vec![
            E4WrBindRunRow {
                wr: 0.5,
                mirror_weight: 0.5,
                seed: 1,
                root_fit: 1.0,
                ceiling_fit: 2.0,
                delta_bind: -1.0,
                root_fit_anchor: 0.5,
                ceiling_fit_anchor: 1.0,
                delta_bind_anchor: -0.33333334,
            },
            E4WrBindRunRow {
                wr: 0.5,
                mirror_weight: 0.5,
                seed: 2,
                root_fit: 3.0,
                ceiling_fit: 4.0,
                delta_bind: 1.0,
                root_fit_anchor: 1.5,
                ceiling_fit_anchor: 2.0,
                delta_bind_anchor: -0.14285715,
            },
        ];
        let summary_rows = e4_wr_bind_summary_rows(&runs);
        assert_eq!(summary_rows.len(), 1);
        let summary = summary_rows[0];
        let csv = e4_wr_sweep_summary_csv(&summary_rows, &[]);
        let line = csv
            .lines()
            .nth(1)
            .expect("expected single data row in wr summary csv");
        let cols: Vec<&str> = line.split(',').collect();
        assert!(cols.len() >= 10, "unexpected column count: {}", cols.len());

        let root_mean: f32 = cols[2].parse().expect("root mean parse");
        let root_ci95: f32 = cols[3].parse().expect("root ci95 parse");
        let ceiling_mean: f32 = cols[4].parse().expect("ceiling mean parse");
        let ceiling_ci95: f32 = cols[5].parse().expect("ceiling ci95 parse");
        let delta_mean: f32 = cols[6].parse().expect("delta mean parse");
        let delta_ci95: f32 = cols[7].parse().expect("delta ci95 parse");
        let n_seeds: usize = cols[8].parse().expect("n parse");
        let error_kind = cols[9];
        let exp_root_ci95 = 0.5 * (summary.root_fit_ci_hi - summary.root_fit_ci_lo);
        let exp_ceiling_ci95 = 0.5 * (summary.ceiling_fit_ci_hi - summary.ceiling_fit_ci_lo);
        let exp_delta_ci95 = 0.5 * (summary.delta_bind_ci_hi - summary.delta_bind_ci_lo);

        assert!((root_mean - 2.0).abs() < 1e-6, "root_mean={root_mean}");
        assert!(
            (root_ci95 - exp_root_ci95).abs() < 1e-3,
            "root_ci95={root_ci95} expected={exp_root_ci95}"
        );
        assert!(
            (ceiling_mean - 3.0).abs() < 1e-6,
            "ceiling_mean={ceiling_mean}"
        );
        assert!(
            (ceiling_ci95 - exp_ceiling_ci95).abs() < 1e-3,
            "ceiling_ci95={ceiling_ci95} expected={exp_ceiling_ci95}"
        );
        assert!((delta_mean - 0.0).abs() < 1e-6, "delta_mean={delta_mean}");
        assert!(
            (delta_ci95 - exp_delta_ci95).abs() < 1e-3,
            "delta_ci95={delta_ci95} expected={exp_delta_ci95}"
        );
        assert_eq!(n_seeds, 2);
        assert_eq!(error_kind, "bootstrap_pctl95");
    }

    #[test]
    fn e4_abcd_trace_has_required_columns_and_finite_metrics() {
        let meta = e4_paper_meta();
        let space = Log2Space::new(meta.fmin, meta.fmax, meta.bins_per_oct);
        let (_erb_scan, du_scan) = erb_grid_for_space(&space);
        let mut grouped = HashMap::new();
        grouped.insert(
            (float_key(1.0), float_key(0.5), 42_u64),
            vec![
                meta.anchor_hz,
                meta.anchor_hz * (5.0 / 4.0),
                meta.anchor_hz * (3.0 / 2.0),
            ],
        );

        let rows = e4_abcd_trace_rows(&grouped, &space, meta.anchor_hz, &du_scan, 20);
        assert!(!rows.is_empty(), "trace rows must not be empty");
        assert!(rows.iter().all(|r| {
            r.a.is_finite()
                && r.b.is_finite()
                && r.c.is_finite()
                && r.d.is_finite()
                && r.agent_c01.is_finite()
                && r.oracle_c01.is_finite()
                && r.agent_log2.is_finite()
                && r.oracle_log2.is_finite()
        }));
        for pair in rows.windows(2) {
            assert!(pair[1].step > pair[0].step, "step must be monotonic");
        }

        let csv = e4_abcd_trace_csv(&rows);
        let header = csv.lines().next().unwrap_or_default();
        for required in [
            "run_id",
            "seed",
            "mirror_weight",
            "step",
            "A",
            "B",
            "C",
            "D",
            "agent_idx",
            "oracle_idx",
        ] {
            assert!(
                header.split(',').any(|col| col == required),
                "missing required column: {required}"
            );
        }
    }

    #[test]
    fn e4_abcd_lag_estimator_recovers_positive_three_in_examples_paper() {
        fn best_lag_from_indices(
            agent: &[i32],
            oracle: &[i32],
            burn_in: usize,
            max_lag: i32,
        ) -> i32 {
            let n = agent.len().min(oracle.len());
            let mut best_lag = 0i32;
            let mut best_score = f32::NEG_INFINITY;
            for lag in -max_lag..=max_lag {
                let mut sum = 0.0f32;
                let mut count = 0usize;
                for t in burn_in..n {
                    let t2 = t as i32 + lag;
                    if t2 < 0 || t2 as usize >= n {
                        continue;
                    }
                    let score = if agent[t] == oracle[t2 as usize] {
                        1.0
                    } else {
                        0.0
                    };
                    sum += score;
                    count += 1;
                }
                if count == 0 {
                    continue;
                }
                let mean = sum / count as f32;
                let better = mean > best_score + 1e-9;
                let tie = (mean - best_score).abs() <= 1e-9;
                if better
                    || (tie && lag.abs() < best_lag.abs())
                    || (tie && lag.abs() == best_lag.abs() && lag < best_lag)
                {
                    best_score = mean;
                    best_lag = lag;
                }
            }
            best_lag
        }

        let shift = 3i32;
        let n = 40usize;
        let burn_in = (n as f32 * 0.25).floor() as usize;
        let agent: Vec<i32> = (0..n).map(|t| (t % 7) as i32).collect();
        let oracle: Vec<i32> = (0..n)
            .map(|t| {
                let src = t as i32 - shift;
                if src >= 0 {
                    (src as usize % 7) as i32
                } else {
                    -1
                }
            })
            .collect();
        let best = best_lag_from_indices(&agent, &oracle, burn_in, 8);
        assert_eq!(best, 3);
    }

    #[test]
    fn e4_diag_landscape_changes_with_mirror_weight() {
        let meta = e4_paper_meta();
        let space = Log2Space::new(meta.fmin, meta.fmax, meta.bins_per_oct);
        let (_erb_scan, du_scan) = erb_grid_for_space(&space);
        let freqs = [
            meta.anchor_hz,
            meta.anchor_hz * (5.0 / 4.0),
            meta.anchor_hz * (3.0 / 2.0),
        ];
        let scan0 = compute_e4_landscape_scans(&space, meta.anchor_hz, 1.0, 0.0, &freqs, &du_scan);
        let scan1 = compute_e4_landscape_scans(&space, meta.anchor_hz, 1.0, 1.0, &freqs, &du_scan);
        let delta = l1_mean_distance(&scan0.c, &scan1.c);
        assert!(delta > 1e-5, "landscape delta too small: {delta}");
    }

    #[test]
    fn oracle_global_score_is_not_lower_than_reachable() {
        let c_scan = vec![0.1f32, 0.3, 0.9, 0.2, 0.4];
        let log2_ratio_scan = vec![0.0f32, 0.1, 0.2, 0.3, 0.4];
        let indices = vec![1usize, 3usize];
        let (global_idx, global_score) = e4_diag_best_index_and_score(
            0,
            &indices,
            &c_scan,
            &log2_ratio_scan,
            0,
            c_scan.len() - 1,
            1,
            0.2,
            0.1,
            true,
        );
        let (reachable_idx, reachable_score) = e4_diag_best_index_and_score(
            0,
            &indices,
            &c_scan,
            &log2_ratio_scan,
            0,
            c_scan.len() - 1,
            1,
            0.2,
            0.1,
            false,
        );
        assert!(global_score + 1e-6 >= reachable_score);
        assert!(global_idx < c_scan.len());
        assert!(reachable_idx < c_scan.len());
    }

    #[test]
    fn gap_reach_is_zero_when_agent_matches_reachable_oracle() {
        let c_scan = vec![0.1f32, 0.2, 0.95, 0.4, 0.1];
        let log2_ratio_scan = vec![0.0f32, 0.1, 0.2, 0.3, 0.4];
        let indices = vec![2usize, 4usize];
        let (oracle_reach_idx, oracle_reach_score) = e4_diag_best_index_and_score(
            0,
            &indices,
            &c_scan,
            &log2_ratio_scan,
            0,
            c_scan.len() - 1,
            1,
            0.0,
            0.1,
            false,
        );
        let agent_score =
            e4_diag_candidate_score(0, &indices, indices[0], &c_scan, &log2_ratio_scan, 0.0, 0.1)
                .unwrap_or(0.0);
        if oracle_reach_idx == indices[0] {
            assert!((oracle_reach_score - agent_score).abs() < 1e-6);
        }
    }

    #[test]
    fn peaklist_changes_between_mw_zero_and_one() {
        let meta = e4_paper_meta();
        let space = Log2Space::new(meta.fmin, meta.fmax, meta.bins_per_oct);
        let (_erb_scan, du_scan) = erb_grid_for_space(&space);
        let freqs = [
            meta.anchor_hz,
            meta.anchor_hz * (3.0 / 2.0),
            meta.anchor_hz * (5.0 / 4.0),
            meta.anchor_hz * 2.0,
        ];
        let scan_m0 =
            compute_e4_landscape_scans(&space, meta.anchor_hz, 1.0, 0.0, &freqs, &du_scan);
        let scan_m1 =
            compute_e4_landscape_scans(&space, meta.anchor_hz, 1.0, 1.0, &freqs, &du_scan);
        let p0 = extract_peak_rows_from_c_scan(&space, meta.anchor_hz, &scan_m0, 16);
        let p1 = extract_peak_rows_from_c_scan(&space, meta.anchor_hz, &scan_m1, 16);
        assert!(
            !p0.is_empty() && !p1.is_empty(),
            "peak lists must be non-empty"
        );
        let n = p0.len().min(p1.len()).min(8);
        let mut same = true;
        for i in 0..n {
            if p0[i].bin_idx != p1[i].bin_idx {
                same = false;
                break;
            }
        }
        assert!(
            !same,
            "mw=0 and mw=1 peak lists unexpectedly identical in top-{n}"
        );
    }

    #[test]
    fn oracle_greedy_delta_bind_is_finite() {
        let peaks = vec![
            E4PeakRow {
                rank: 1,
                bin_idx: 0,
                log2_ratio: 0.0,
                semitones: 0.0,
                freq_hz: 220.0,
                c_value: 1.0,
                prominence: 0.9,
                width_st: 0.3,
            },
            E4PeakRow {
                rank: 2,
                bin_idx: 1,
                log2_ratio: 7.0 / 12.0,
                semitones: 7.0,
                freq_hz: 330.0,
                c_value: 0.95,
                prominence: 0.8,
                width_st: 0.3,
            },
            E4PeakRow {
                rank: 3,
                bin_idx: 2,
                log2_ratio: 1.0,
                semitones: 12.0,
                freq_hz: 440.0,
                c_value: 0.9,
                prominence: 0.7,
                width_st: 0.3,
            },
            E4PeakRow {
                rank: 4,
                bin_idx: 3,
                log2_ratio: 1.3219281,
                semitones: 15.863137,
                freq_hz: 550.0,
                c_value: 0.88,
                prominence: 0.7,
                width_st: 0.3,
            },
            E4PeakRow {
                rank: 5,
                bin_idx: 4,
                log2_ratio: 0.09,
                semitones: 1.08,
                freq_hz: 233.0,
                c_value: 0.12,
                prominence: 0.1,
                width_st: 0.2,
            },
            E4PeakRow {
                rank: 6,
                bin_idx: 5,
                log2_ratio: 0.30,
                semitones: 3.6,
                freq_hz: 271.0,
                c_value: 0.11,
                prominence: 0.1,
                width_st: 0.2,
            },
        ];
        let oracle_abs = oracle1_delta_bind_from_peaks(&peaks, 4).abs();
        assert!(
            oracle_abs.is_finite() && oracle_abs <= 1.0 + 1e-6,
            "oracle_abs={oracle_abs}"
        );
    }

    #[test]
    fn e2_seed_count_is_20() {
        assert_eq!(E2_SEEDS.len(), 20);
    }

    #[test]
    fn e4_seed_count_is_20() {
        assert_eq!(E4_SEEDS.len(), 20);
    }

    #[test]
    fn e4_wr_reps_is_at_least_20() {
        assert!(E4_WR_REPS >= 20);
    }

    #[test]
    fn update_agent_indices_stays_in_bounds() {
        let mut indices = vec![1usize, 2, 3];
        let c_score_scan = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let log2_ratio_scan = vec![0.0f32, 0.1, 0.2, 0.3, 0.4];
        let _ = update_agent_indices_scored_stats(
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
        let prev_indices = indices.clone();
        let mut rng = StdRng::seed_from_u64(0);
        let stats = update_e2_sweep_scored_loo(
            E2UpdateSchedule::RandomSingle,
            &mut indices,
            &prev_indices,
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
    fn e2_update_schedule_random_single_attempts_all_agents() {
        let space = Log2Space::new(200.0, 400.0, 12);
        let workspace = build_consonance_workspace(&space);
        let (_erb_scan, du_scan) = erb_grid_for_space(&space);
        let anchor_idx = space.n_bins() / 2;
        let anchor_hz = space.centers_hz[anchor_idx];
        let log2_ratio_scan = build_log2_ratio_scan(&space, anchor_hz);
        let mut indices = vec![1usize, 3, 4];
        let prev_indices = indices.clone();
        let (env_scan, density_scan) = build_env_scans(&space, anchor_idx, &indices, &du_scan);
        let mut rng = StdRng::seed_from_u64(1);
        let stats = update_e2_sweep_scored_loo(
            E2UpdateSchedule::RandomSingle,
            &mut indices,
            &prev_indices,
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
        );
        assert!(
            (stats.attempted_update_frac - 1.0).abs() < 1e-6,
            "expected attempted_update_frac=1.0, got {}",
            stats.attempted_update_frac
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
    fn parse_e4_kernel_gate_defaults_off() {
        let args: Vec<String> = Vec::new();
        let enabled = parse_e4_kernel_gate(&args).expect("parse_e4_kernel_gate failed");
        assert!(!enabled);
    }

    #[test]
    fn parse_e4_kernel_gate_accepts_expected_values() {
        let cases: &[(&[&str], bool)] = &[
            (&["--e4-kernel-gate", "on"], true),
            (&["--e4-kernel-gate=on"], true),
            (&["--e4-kernel-gate", "off"], false),
            (&["--e4-kernel-gate=off"], false),
            (&["--e4-kernel-gate", "true"], true),
            (&["--e4-kernel-gate=false"], false),
            (&["--e4-kernel-gate", "1"], true),
            (&["--e4-kernel-gate=0"], false),
        ];
        for (args, expected) in cases {
            let args = args.iter().map(|s| s.to_string()).collect::<Vec<_>>();
            let enabled = parse_e4_kernel_gate(&args).expect("parse_e4_kernel_gate failed");
            assert_eq!(enabled, *expected);
        }
    }

    #[test]
    fn parse_e4_kernel_gate_rejects_invalid_values() {
        let args = vec!["--e4-kernel-gate".to_string(), "maybe".to_string()];
        let err = parse_e4_kernel_gate(&args).expect_err("expected parse_e4_kernel_gate to fail");
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
            "--e4-kernel-gate".to_string(),
            "off".to_string(),
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
        assert_eq!(run.mean_c_score_chosen_loo_series.len(), E2_SWEEPS);
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
        assert_eq!(run.mean_abs_delta_semitones_series.len(), E2_SWEEPS);
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
        assert!((metrics.move_rate_stepwise - 0.5).abs() < 1e-6);
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
    fn binomial_two_sided_p_has_expected_limits() {
        let p_mid = binomial_two_sided_p(5, 10);
        assert!((p_mid - 1.0).abs() < 1e-6);
        let p_edge = binomial_two_sided_p(0, 10);
        let expected = 2.0 * 0.5f32.powi(10);
        assert!((p_edge - expected).abs() < 1e-6);
    }

    #[test]
    fn fmt_eps_formats_fractional_values() {
        assert_eq!(fmt_eps(25.0), "25c");
        assert_eq!(fmt_eps(12.5), "12.5c");
        assert_eq!(fmt_eps(12.0), "12c");
        assert_eq!(fmt_eps(12.49), "12.5c");
    }

    #[test]
    fn fmt_eps_token_formats_fractional_values() {
        assert_eq!(fmt_eps_token(12.5), "12p5");
    }

    #[test]
    fn t_crit_975_matches_reference_values() {
        assert!((t_crit_975(9) - 2.262).abs() < 0.01);
        assert!((t_crit_975(100) - 1.96).abs() < 1e-6);
    }

    #[test]
    fn float_key_roundtrip_is_stable() {
        let values = [0.02f32, 0.25, 0.98, 1.0];
        for value in values {
            let key = float_key(value);
            let expected = (value * FLOAT_KEY_SCALE).round() as i32;
            assert_eq!(key, expected);
            let roundtrip = float_from_key(key);
            assert!((roundtrip - value).abs() < 1e-6);
        }
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
