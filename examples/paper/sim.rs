use conchordal::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
use conchordal::core::landscape::{Landscape, LandscapeParams, LandscapeUpdate, RoughnessScalarMode};
use conchordal::core::log2space::Log2Space;
use conchordal::core::modulation::{NeuralRhythms, RhythmBand};
use conchordal::core::roughness_kernel::{KernelParams, RoughnessKernel};
use conchordal::core::timebase::Timebase;
use conchordal::life::control::{AgentControl, PhonationType, PitchMode};
use conchordal::life::individual::SoundBody;
use conchordal::life::lifecycle::LifecycleConfig;
use conchordal::life::metabolism_policy::MetabolismPolicy;
use conchordal::life::population::Population;
use conchordal::life::scenario::{
    Action, ArticulationCoreConfig, EnvelopeConfig, SpawnSpec, SpawnStrategy,
};

pub const E4_ANCHOR_HZ: f32 = 220.0;
pub const E4_WINDOW_CENTS: f32 = 50.0;

const E4_GROUP_ANCHOR: u64 = 0;
const E4_GROUP_VOICES: u64 = 1;

const E3_GROUP_AGENTS: u64 = 2;
const E3_FS: f32 = 48_000.0;
const E3_HOP: usize = 512;
const E3_BINS_PER_OCT: u32 = 96;
const E3_FMIN: f32 = 80.0;
const E3_FMAX: f32 = 2000.0;
const E3_RANGE_OCT: f32 = 2.0; // +/- 1 octave around anchor
const E3_THETA_FREQ_HZ: f32 = 1.0;
const E3_METABOLISM_RATE: f32 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum E3Condition {
    Baseline,
    NoRecharge,
}

impl E3Condition {
    pub fn label(self) -> &'static str {
        match self {
            E3Condition::Baseline => "baseline",
            E3Condition::NoRecharge => "norecharge",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct E3RunConfig {
    pub seed: u64,
    pub steps_cap: usize,
    pub min_deaths: usize,
    pub pop_size: usize,
    pub first_k: usize,
    pub condition: E3Condition,
}

#[derive(Clone, Debug)]
pub struct E3DeathRecord {
    pub condition: String,
    pub seed: u64,
    pub life_id: u64,
    pub agent_id: usize,
    pub birth_step: u32,
    pub death_step: u32,
    pub lifetime_steps: u32,
    pub c_state_birth: f32,
    pub c_state_firstk: f32,
    pub avg_c_state_tick: f32,
    pub avg_c_state_attack: f32,
    pub attack_tick_count: u32,
}

#[derive(Clone, Debug)]
pub struct E3PolicyParams {
    pub condition: String,
    pub dt_sec: f32,
    pub basal_cost_per_sec: f32,
    pub action_cost_per_attack: f32,
    pub recharge_per_attack: f32,
    pub recharge_threshold: f32,
}

#[derive(Clone, Copy, Debug)]
struct E3LifeState {
    life_id: u64,
    birth_step: u32,
    ticks: u32,
    sum_c_state_tick: f32,
    sum_c_state_firstk: f32,
    firstk_count: u32,
    c_state_birth: f32,
    pending_birth: bool,
    was_alive: bool,
    sum_c_state_attack: f32,
    attack_tick_count: u32,
}

impl E3LifeState {
    fn new(life_id: u64) -> Self {
        Self {
            life_id,
            birth_step: 0,
            ticks: 0,
            sum_c_state_tick: 0.0,
            sum_c_state_firstk: 0.0,
            firstk_count: 0,
            c_state_birth: 0.0,
            pending_birth: true,
            was_alive: true,
            sum_c_state_attack: 0.0,
            attack_tick_count: 0,
        }
    }

    fn reset_for_new_life(&mut self, life_id: u64) {
        self.life_id = life_id;
        self.birth_step = 0;
        self.ticks = 0;
        self.sum_c_state_tick = 0.0;
        self.sum_c_state_firstk = 0.0;
        self.firstk_count = 0;
        self.c_state_birth = 0.0;
        self.pending_birth = true;
        self.was_alive = false;
        self.sum_c_state_attack = 0.0;
        self.attack_tick_count = 0;
    }
}

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

pub fn run_e3_collect_deaths(cfg: &E3RunConfig) -> Vec<E3DeathRecord> {
    let mut deaths = Vec::new();
    if cfg.pop_size == 0 || cfg.steps_cap == 0 {
        return deaths;
    }

    let anchor_hz = E4_ANCHOR_HZ;
    let space = Log2Space::new(E3_FMIN, E3_FMAX, E3_BINS_PER_OCT);
    let params = make_landscape_params(&space, E3_FS);
    let mut landscape = build_anchor_landscape(&space, &params, anchor_hz);
    landscape.rhythm = init_rhythms(E3_THETA_FREQ_HZ);

    let mut pop = Population::new(Timebase {
        fs: E3_FS,
        hop: E3_HOP,
    });
    pop.set_seed(cfg.seed);
    pop.set_current_frame(0);

    let spec = e3_spawn_spec(cfg.condition, anchor_hz);
    let strategy = e3_spawn_strategy(anchor_hz, &space);
    let ids: Vec<u64> = (0..cfg.pop_size as u64).collect();
    pop.apply_action(
        Action::Spawn {
            group_id: E3_GROUP_AGENTS,
            ids,
            spec: spec.clone(),
            strategy: Some(strategy.clone()),
        },
        &landscape,
        None,
    );

    let mut next_life_id = 0u64;
    let mut states: Vec<E3LifeState> = (0..cfg.pop_size)
        .map(|_| {
            let life_id = next_life_id;
            next_life_id += 1;
            E3LifeState::new(life_id)
        })
        .collect();

    let dt = E3_HOP as f32 / E3_FS;
    let first_k = cfg.first_k as u32;

    for step in 0..cfg.steps_cap {
        pop.advance(E3_HOP, E3_FS, step as u64, dt, &landscape);

        let mut respawn_ids: Vec<u64> = Vec::new();
        for agent in pop.individuals.iter_mut() {
            let id = agent.id();
            let idx = id as usize;
            if idx >= states.len() {
                continue;
            }
            let state = &mut states[idx];
            let (attack_tick_count, attack_sum) = agent.take_attack_telemetry();
            if attack_tick_count > 0 {
                state.attack_tick_count = state.attack_tick_count.saturating_add(attack_tick_count);
                state.sum_c_state_attack += attack_sum;
            }
            let alive = agent.is_alive();

            if alive {
                let c_state = agent.last_consonance_state01();
                if state.pending_birth {
                    state.birth_step = step as u32;
                    state.c_state_birth = c_state;
                    state.pending_birth = false;
                }
                state.ticks = state.ticks.saturating_add(1);
                state.sum_c_state_tick += c_state;
                if state.firstk_count < first_k {
                    state.sum_c_state_firstk += c_state;
                    state.firstk_count += 1;
                }
            }

            if state.was_alive && !alive {
                let ticks = state.ticks.max(1);
                let avg_c_state_tick = if state.ticks > 0 {
                    state.sum_c_state_tick / state.ticks as f32
                } else {
                    0.0
                };
                let c_state_firstk = if state.firstk_count > 0 {
                    state.sum_c_state_firstk / state.firstk_count as f32
                } else {
                    0.0
                };
                let avg_c_state_attack = if state.attack_tick_count > 0 {
                    state.sum_c_state_attack / state.attack_tick_count as f32
                } else {
                    f32::NAN
                };

                deaths.push(E3DeathRecord {
                    condition: cfg.condition.label().to_string(),
                    seed: cfg.seed,
                    life_id: state.life_id,
                    agent_id: idx,
                    birth_step: state.birth_step,
                    death_step: step as u32,
                    lifetime_steps: ticks,
                    c_state_birth: state.c_state_birth,
                    c_state_firstk,
                    avg_c_state_tick,
                    avg_c_state_attack,
                    attack_tick_count: state.attack_tick_count,
                });

                respawn_ids.push(id);
                state.reset_for_new_life(next_life_id);
                next_life_id += 1;
            } else {
                state.was_alive = alive;
            }
        }

        for id in respawn_ids {
            pop.remove_agent(id);
            pop.apply_action(
                Action::Spawn {
                    group_id: E3_GROUP_AGENTS,
                    ids: vec![id],
                    spec: spec.clone(),
                    strategy: Some(strategy.clone()),
                },
                &landscape,
                None,
            );
        }

        landscape.rhythm.advance_in_place(dt);

        if deaths.len() >= cfg.min_deaths {
            break;
        }
    }

    deaths
}

pub fn run_e4_condition(mirror_weight: f32, seed: u64) -> Vec<f32> {
    run_e4_condition_with_config(mirror_weight, seed, &E4SimConfig::paper_defaults())
}

#[derive(Clone, Debug)]
pub struct E4TailSamples {
    pub steps_total: u32,
    pub tail_window: u32,
    pub freqs_by_step: Vec<Vec<f32>>,
}

pub fn run_e4_condition_tail_samples(
    mirror_weight: f32,
    seed: u64,
    tail_window: u32,
) -> E4TailSamples {
    run_e4_condition_tail_samples_with_config(
        mirror_weight,
        seed,
        &E4SimConfig::paper_defaults(),
        tail_window,
    )
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

fn run_e4_condition_tail_samples_with_config(
    mirror_weight: f32,
    seed: u64,
    cfg: &E4SimConfig,
    tail_window: u32,
) -> E4TailSamples {
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

    let tail_window = tail_window.min(cfg.steps).max(1);
    let start_step = cfg.steps.saturating_sub(tail_window);
    let dt = cfg.hop as f32 / cfg.fs;
    let mut freqs_by_step: Vec<Vec<f32>> = Vec::with_capacity(tail_window as usize);

    for step in 0..cfg.steps {
        pop.advance(cfg.hop, cfg.fs, step as u64, dt, &landscape);
        pop.cleanup_dead(step as u64, dt, false);
        landscape.rhythm.advance_in_place(dt);

        if step >= start_step {
            freqs_by_step.push(collect_voice_freqs(&pop));
        }
    }

    E4TailSamples {
        steps_total: cfg.steps,
        tail_window,
        freqs_by_step,
    }
}

fn collect_voice_freqs(pop: &Population) -> Vec<f32> {
    let mut freqs = Vec::new();
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
        consonance_harmonicity_weight: 1.0,
        consonance_roughness_weight_floor: 0.35,
        consonance_roughness_weight: 0.5,
        c_state_beta: 2.0,
        c_state_theta: 0.0,
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

fn e3_spawn_spec(condition: E3Condition, anchor_hz: f32) -> SpawnSpec {
    let mut control = AgentControl::default();
    control.pitch.mode = PitchMode::Lock;
    control.pitch.freq = anchor_hz.max(1.0);
    control.phonation.r#type = PhonationType::Hold;
    let lifecycle = e3_lifecycle(condition);
    let articulation = ArticulationCoreConfig::Entrain {
        lifecycle,
        rhythm_freq: Some(E3_THETA_FREQ_HZ),
        rhythm_sensitivity: None,
        breath_gain_init: None,
    };

    SpawnSpec {
        control,
        articulation,
    }
}

fn e3_spawn_strategy(anchor_hz: f32, space: &Log2Space) -> SpawnStrategy {
    let half = 0.5 * E3_RANGE_OCT;
    let min_freq = anchor_hz * 2.0f32.powf(-half);
    let max_freq = anchor_hz * 2.0f32.powf(half);
    let min_freq = min_freq.clamp(space.fmin, space.fmax);
    let max_freq = max_freq.clamp(space.fmin, space.fmax);
    SpawnStrategy::RandomLog {
        min_freq: min_freq.min(max_freq),
        max_freq: max_freq.max(min_freq),
    }
}

fn e3_lifecycle(condition: E3Condition) -> LifecycleConfig {
    let recharge_rate = match condition {
        E3Condition::Baseline => None,
        E3Condition::NoRecharge => Some(0.0),
    };
    LifecycleConfig::Sustain {
        initial_energy: 1.0,
        metabolism_rate: E3_METABOLISM_RATE,
        recharge_rate,
        action_cost: None,
        envelope: EnvelopeConfig::default(),
    }
}

pub fn e3_policy_params(condition: E3Condition) -> E3PolicyParams {
    let lifecycle = e3_lifecycle(condition);
    let policy = MetabolismPolicy::from_lifecycle(&lifecycle);
    E3PolicyParams {
        condition: condition.label().to_string(),
        dt_sec: E3_HOP as f32 / E3_FS,
        basal_cost_per_sec: policy.basal_cost_per_sec,
        action_cost_per_attack: policy.action_cost_per_attack,
        recharge_per_attack: policy.recharge_per_attack,
        recharge_threshold: policy.recharge_threshold,
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
