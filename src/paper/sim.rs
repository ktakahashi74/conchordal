use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
use crate::core::landscape::{Landscape, LandscapeParams, LandscapeUpdate, RoughnessScalarMode};
use crate::core::log2space::Log2Space;
use crate::core::modulation::{NeuralRhythms, RhythmBand};
use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};
use crate::core::timebase::Timebase;
use crate::life::control::{AgentControl, PhonationType, PitchMode};
use crate::life::individual::SoundBody;
use crate::life::population::Population;
use crate::life::scenario::{Action, ArticulationCoreConfig, SpawnSpec, SpawnStrategy};

pub const E4_ANCHOR_HZ: f32 = 220.0;
pub const E4_WINDOW_CENTS: f32 = 50.0;

const E4_GROUP_ANCHOR: u64 = 0;
const E4_GROUP_VOICES: u64 = 1;

#[derive(Clone, Copy, Debug)]
struct E4SimConfig {
    anchor_hz: f32,
    center_cents: f32,
    range_oct: f32,
    voice_count: usize,
    fs: f32,
    hop: usize,
    steps: u32,
    bins_per_oct: u32,
    fmin: f32,
    fmax: f32,
    min_dist_erb: f32,
    exploration: f32,
    persistence: f32,
    theta_freq_hz: f32,
}

impl E4SimConfig {
    fn paper_defaults() -> Self {
        Self {
            anchor_hz: E4_ANCHOR_HZ,
            center_cents: 350.0,
            range_oct: 0.5,
            voice_count: 24,
            fs: 48_000.0,
            hop: 512,
            steps: 1200,
            bins_per_oct: 96,
            fmin: 80.0,
            fmax: 2000.0,
            min_dist_erb: 0.0,
            exploration: 0.8,
            persistence: 0.2,
            theta_freq_hz: 6.0,
        }
    }

    #[cfg(test)]
    fn test_defaults() -> Self {
        Self {
            voice_count: 6,
            steps: 420,
            ..Self::paper_defaults()
        }
    }

    fn center_hz(&self) -> f32 {
        let ratio = 2.0f32.powf(self.center_cents / 1200.0);
        self.anchor_hz * ratio
    }

    fn range_bounds_hz(&self) -> (f32, f32) {
        let half = self.range_oct * 0.5;
        let center = self.center_hz();
        let lo = center * 2.0f32.powf(-half);
        let hi = center * 2.0f32.powf(half);
        (lo.min(hi), lo.max(hi))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct IntervalMetrics {
    pub mass_maj3: f32,
    pub mass_min3: f32,
    pub mass_p5: f32,
    pub n_voices: usize,
}

pub fn interval_metrics(anchor_hz: f32, freqs_hz: &[f32], window_cents: f32) -> IntervalMetrics {
    let mut metrics = IntervalMetrics {
        n_voices: freqs_hz.len(),
        ..IntervalMetrics::default()
    };
    if !anchor_hz.is_finite() || anchor_hz <= 0.0 {
        return metrics;
    }
    let w = window_cents.max(0.0);
    for &freq in freqs_hz {
        if !freq.is_finite() || freq <= 0.0 {
            continue;
        }
        let ratio = freq / anchor_hz;
        if !ratio.is_finite() || ratio <= 0.0 {
            continue;
        }
        let cents = 1200.0 * ratio.log2();
        let cents_mod = cents.rem_euclid(1200.0);
        if (cents_mod - 400.0).abs() <= w {
            metrics.mass_maj3 += 1.0;
        }
        if (cents_mod - 300.0).abs() <= w {
            metrics.mass_min3 += 1.0;
        }
        if (cents_mod - 700.0).abs() <= w {
            metrics.mass_p5 += 1.0;
        }
    }
    metrics
}

pub fn run_e4_condition(mirror_weight: f32, seed: u64) -> Vec<f32> {
    run_e4_condition_with_config(mirror_weight, seed, &E4SimConfig::paper_defaults())
}

fn run_e4_condition_with_config(mirror_weight: f32, seed: u64, cfg: &E4SimConfig) -> Vec<f32> {
    let space = Log2Space::new(cfg.fmin, cfg.fmax, cfg.bins_per_oct);
    let mut params = make_landscape_params(&space, cfg.fs);
    let mut landscape = Landscape::new(space.clone());

    let mut pop = Population::new(Timebase {
        fs: cfg.fs,
        hop: cfg.hop,
    });
    pop.set_seed(seed);

    let update = LandscapeUpdate {
        mirror: Some(mirror_weight),
        ..LandscapeUpdate::default()
    };
    pop.apply_action(Action::SetHarmonicityParams { update }, &landscape, None);
    if let Some(update) = pop.take_pending_update() {
        apply_params_update(&mut params, &update);
    }

    landscape = build_anchor_landscape(&space, &params, cfg.anchor_hz);
    landscape.rhythm = init_rhythms(cfg.theta_freq_hz);

    let anchor_spec = SpawnSpec {
        control: anchor_control(cfg.anchor_hz),
        articulation: ArticulationCoreConfig::Drone {
            sway: None,
            breath_gain_init: Some(1.0),
        },
    };
    pop.apply_action(
        Action::Spawn {
            group_id: E4_GROUP_ANCHOR,
            ids: vec![0],
            spec: anchor_spec,
            strategy: None,
        },
        &landscape,
        None,
    );

    let (min_freq, max_freq) = cfg.range_bounds_hz();
    let voice_spec = SpawnSpec {
        control: voice_control(cfg),
        articulation: ArticulationCoreConfig::Drone {
            sway: None,
            breath_gain_init: Some(1.0),
        },
    };
    let ids: Vec<u64> = (1..=cfg.voice_count as u64).collect();
    let strategy = SpawnStrategy::ConsonanceDensity {
        min_freq,
        max_freq,
        min_dist_erb: cfg.min_dist_erb,
    };
    pop.apply_action(
        Action::Spawn {
            group_id: E4_GROUP_VOICES,
            ids,
            spec: voice_spec,
            strategy: Some(strategy),
        },
        &landscape,
        None,
    );

    let dt = cfg.hop as f32 / cfg.fs;
    for step in 0..cfg.steps {
        pop.advance(cfg.hop, cfg.fs, step as u64, dt, &landscape);
        pop.cleanup_dead(step as u64, dt, false);
        landscape.rhythm.advance_in_place(dt);
    }

    let mut freqs = Vec::with_capacity(cfg.voice_count);
    for agent in &pop.individuals {
        if agent.metadata.group_id != E4_GROUP_VOICES {
            continue;
        }
        let freq = agent.body.base_freq_hz();
        if freq.is_finite() && freq > 0.0 {
            freqs.push(freq);
        }
    }
    freqs
}

fn make_landscape_params(space: &Log2Space, fs: f32) -> LandscapeParams {
    LandscapeParams {
        fs,
        max_hist_cols: 1,
        alpha: 0.0,
        roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005),
        harmonicity_kernel: HarmonicityKernel::new(space, HarmonicityParams::default()),
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
    }
}

fn apply_params_update(params: &mut LandscapeParams, upd: &LandscapeUpdate) {
    if let Some(m) = upd.mirror {
        params.harmonicity_kernel.params.mirror_weight = m;
    }
    if let Some(l) = upd.limit {
        params.harmonicity_kernel.params.param_limit = l;
    }
    if let Some(k) = upd.roughness_k {
        params.roughness_k = k.max(1e-6);
    }
}

fn build_anchor_landscape(
    space: &Log2Space,
    params: &LandscapeParams,
    anchor_hz: f32,
) -> Landscape {
    let mut landscape = Landscape::new(space.clone());
    let mut anchor_env_scan = vec![0.0f32; space.n_bins()];
    let anchor_idx = space.index_of_freq(anchor_hz).unwrap_or(space.n_bins() / 2);
    anchor_env_scan[anchor_idx] = 1.0;
    space.assert_scan_len_named(&anchor_env_scan, "anchor_env_scan");

    let (h_pot_scan, _) = params
        .harmonicity_kernel
        .potential_h_from_log2_spectrum(&anchor_env_scan, space);
    space.assert_scan_len_named(&h_pot_scan, "perc_h_pot_scan");

    landscape.subjective_intensity = anchor_env_scan.clone();
    landscape.nsgt_power = anchor_env_scan;
    landscape.harmonicity = h_pot_scan;
    landscape.roughness.fill(0.0);
    landscape.roughness01.fill(0.0);
    landscape.recompute_consonance(params);
    landscape
}

fn init_rhythms(theta_freq_hz: f32) -> NeuralRhythms {
    NeuralRhythms {
        theta: RhythmBand {
            phase: 0.0,
            freq_hz: theta_freq_hz.max(0.1),
            mag: 1.0,
            alpha: 1.0,
            beta: 0.0,
        },
        delta: RhythmBand {
            phase: 0.0,
            freq_hz: 1.0,
            mag: 0.0,
            alpha: 0.0,
            beta: 0.0,
        },
        env_open: 1.0,
        env_level: 1.0,
    }
}

fn anchor_control(anchor_hz: f32) -> AgentControl {
    let mut control = AgentControl::default();
    control.pitch.mode = PitchMode::Lock;
    control.pitch.freq = anchor_hz.max(1.0);
    control.phonation.r#type = PhonationType::Hold;
    control
}

fn voice_control(cfg: &E4SimConfig) -> AgentControl {
    let mut control = AgentControl::default();
    control.pitch.mode = PitchMode::Free;
    control.pitch.freq = cfg.center_hz().max(1.0);
    control.pitch.range_oct = cfg.range_oct;
    control.pitch.gravity = 0.0;
    control.pitch.exploration = cfg.exploration;
    control.pitch.persistence = cfg.persistence;
    control.phonation.r#type = PhonationType::Hold;
    control
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4_mirror_weight_flips_major_minor_bias() {
        let cfg = E4SimConfig::test_defaults();
        let seed = 7;

        let freqs_m0 = run_e4_condition_with_config(0.0, seed, &cfg);
        let freqs_m1 = run_e4_condition_with_config(1.0, seed, &cfg);

        let metrics_m0 = interval_metrics(cfg.anchor_hz, &freqs_m0, E4_WINDOW_CENTS);
        let metrics_m1 = interval_metrics(cfg.anchor_hz, &freqs_m1, E4_WINDOW_CENTS);

        let diff_m0 = metrics_m0.mass_maj3 - metrics_m0.mass_min3;
        let diff_m1 = metrics_m1.mass_maj3 - metrics_m1.mass_min3;

        assert!(
            diff_m0.is_finite() && diff_m1.is_finite(),
            "expected finite diffs, got diff0={diff_m0:.3}, diff1={diff_m1:.3}"
        );
        assert!(
            diff_m0 > diff_m1 + 0.5,
            "expected major-minor gap to shrink with mirror weight: diff0={diff_m0:.3}, diff1={diff_m1:.3}"
        );
    }
}
