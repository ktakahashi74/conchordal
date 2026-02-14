use conchordal::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
use conchordal::core::landscape::{
    Landscape, LandscapeParams, LandscapeUpdate, RoughnessScalarMode,
};
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
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::sync::{Arc, Mutex, OnceLock};

pub const E4_ANCHOR_HZ: f32 = 220.0;
#[allow(dead_code)]
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
    pub c_state_std_over_life: f32,
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
}

#[derive(Clone, Copy, Debug)]
struct E3LifeState {
    life_id: u64,
    birth_step: u32,
    ticks: u32,
    sum_c_state_tick: f32,
    sum_c_state_tick_sq: f32,
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
            sum_c_state_tick_sq: 0.0,
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
        self.sum_c_state_tick_sq = 0.0;
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
    neighbor_step_cents: f32,
}

impl E4SimConfig {
    fn paper_defaults() -> Self {
        Self {
            anchor_hz: E4_ANCHOR_HZ,
            center_cents: 350.0,
            range_oct: 0.5,
            voice_count: 32,
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
            neighbor_step_cents: 50.0,
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
#[allow(dead_code)]
pub struct IntervalMetrics {
    pub mass_maj3: f32,
    pub mass_min3: f32,
    pub mass_p5: f32,
    pub n_voices: usize,
}

#[allow(dead_code)]
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
                state.sum_c_state_tick_sq += c_state * c_state;
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
                let c_state_std_over_life = if state.ticks > 0 {
                    let mean_sq = state.sum_c_state_tick_sq / state.ticks as f32;
                    let var = (mean_sq - avg_c_state_tick * avg_c_state_tick).max(0.0);
                    var.sqrt()
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
                    c_state_std_over_life,
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

#[allow(dead_code)]
pub fn run_e4_condition(mirror_weight: f32, seed: u64) -> Vec<f32> {
    run_e4_condition_with_config(mirror_weight, seed, &E4SimConfig::paper_defaults())
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct E4TailSamples {
    pub steps_total: u32,
    pub tail_window: u32,
    pub freqs_by_step: Vec<Vec<f32>>,
    pub agent_freqs_by_step: Vec<Vec<E4AgentFreq>>,
}

#[derive(Hash, Eq, PartialEq)]
struct E4TailSamplesCacheKey {
    mirror_weight_bits: u32,
    seed: u64,
    tail_window: u32,
    steps: u32,
    fmin_bits: u32,
    fmax_bits: u32,
    anchor_hz_bits: u32,
    center_cents_bits: u32,
    range_oct_bits: u32,
    fs_bits: u32,
    hop: u32,
    bins_per_oct: u32,
    voice_count: u32,
    min_dist_erb_bits: u32,
    exploration_bits: u32,
    persistence_bits: u32,
    theta_freq_hz_bits: u32,
    neighbor_step_cents_bits: u32,
}

type E4TailSamplesCache = std::collections::HashMap<E4TailSamplesCacheKey, Arc<E4TailSamples>>;
static E4_TAIL_SAMPLES_CACHE: OnceLock<Mutex<E4TailSamplesCache>> = OnceLock::new();

fn e4_tail_samples_cache() -> &'static Mutex<E4TailSamplesCache> {
    E4_TAIL_SAMPLES_CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn e4_tail_samples_cache_key(
    cfg: &E4SimConfig,
    mirror_weight: f32,
    seed: u64,
    tail_window: u32,
) -> E4TailSamplesCacheKey {
    E4TailSamplesCacheKey {
        mirror_weight_bits: mirror_weight.to_bits(),
        seed,
        tail_window,
        steps: cfg.steps,
        fmin_bits: cfg.fmin.to_bits(),
        fmax_bits: cfg.fmax.to_bits(),
        anchor_hz_bits: cfg.anchor_hz.to_bits(),
        center_cents_bits: cfg.center_cents.to_bits(),
        range_oct_bits: cfg.range_oct.to_bits(),
        fs_bits: cfg.fs.to_bits(),
        hop: cfg.hop as u32,
        bins_per_oct: cfg.bins_per_oct,
        voice_count: cfg.voice_count as u32,
        min_dist_erb_bits: cfg.min_dist_erb.to_bits(),
        exploration_bits: cfg.exploration.to_bits(),
        persistence_bits: cfg.persistence.to_bits(),
        theta_freq_hz_bits: cfg.theta_freq_hz.to_bits(),
        neighbor_step_cents_bits: cfg.neighbor_step_cents.to_bits(),
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct E4MirrorScheduleSamples {
    pub freqs_by_step: Vec<Vec<f32>>,
    pub mirror_weight_by_step: Vec<f32>,
    pub agent_freqs_by_step: Vec<Vec<E4AgentFreq>>,
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct E4PaperMeta {
    pub anchor_hz: f32,
    pub center_cents: f32,
    pub range_oct: f32,
    pub voice_count: usize,
    pub fs: f32,
    pub hop: usize,
    pub steps: u32,
    pub bins_per_oct: u32,
    pub fmin: f32,
    pub fmax: f32,
    pub min_dist_erb: f32,
    pub exploration: f32,
    pub persistence: f32,
    pub theta_freq_hz: f32,
    pub neighbor_step_cents: f32,
}

#[allow(dead_code)]
pub fn e4_paper_meta() -> E4PaperMeta {
    let cfg = E4SimConfig::paper_defaults();
    E4PaperMeta {
        anchor_hz: cfg.anchor_hz,
        center_cents: cfg.center_cents,
        range_oct: cfg.range_oct,
        voice_count: cfg.voice_count,
        fs: cfg.fs,
        hop: cfg.hop,
        steps: cfg.steps,
        bins_per_oct: cfg.bins_per_oct,
        fmin: cfg.fmin,
        fmax: cfg.fmax,
        min_dist_erb: cfg.min_dist_erb,
        exploration: cfg.exploration,
        persistence: cfg.persistence,
        theta_freq_hz: cfg.theta_freq_hz,
        neighbor_step_cents: cfg.neighbor_step_cents,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct E4AgentFreq {
    pub agent_id: u64,
    pub freq_hz: f32,
}

pub fn run_e4_condition_tail_samples(
    mirror_weight: f32,
    seed: u64,
    tail_window: u32,
) -> Arc<E4TailSamples> {
    run_e4_condition_tail_samples_with_config(
        mirror_weight,
        seed,
        &E4SimConfig::paper_defaults(),
        tail_window,
    )
}

pub fn run_e4_mirror_schedule_samples(
    seed: u64,
    steps_total: u32,
    schedule: &[(u32, f32)],
) -> E4MirrorScheduleSamples {
    run_e4_mirror_schedule_samples_with_config(
        seed,
        steps_total,
        schedule,
        &E4SimConfig::paper_defaults(),
    )
}

#[allow(dead_code)]
fn run_e4_condition_with_config(mirror_weight: f32, seed: u64, cfg: &E4SimConfig) -> Vec<f32> {
    let space = Log2Space::new(cfg.fmin, cfg.fmax, cfg.bins_per_oct);
    let mut params = make_landscape_params(&space, cfg.fs);
    let mut landscape = Landscape::new(space.clone());
    let mut rhythms = init_e4_rhythms(cfg);

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
    pop.apply_action(
        Action::Spawn {
            group_id: E4_GROUP_VOICES,
            ids,
            spec: voice_spec,
            strategy: None,
        },
        &landscape,
        None,
    );
    apply_e4_initial_pitches(&mut pop, &space, &landscape, cfg, seed, min_freq, max_freq);

    let dt = cfg.hop as f32 / cfg.fs;
    for step in 0..cfg.steps {
        update_e4_landscape_from_population(&space, &params, &pop, &mut landscape);
        landscape.rhythm = rhythms;
        pop.advance(cfg.hop, cfg.fs, step as u64, dt, &landscape);
        pop.cleanup_dead(step as u64, dt, false);
        rhythms.advance_in_place(dt);
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

#[allow(dead_code)]
fn run_e4_condition_tail_samples_with_config(
    mirror_weight: f32,
    seed: u64,
    cfg: &E4SimConfig,
    tail_window: u32,
) -> Arc<E4TailSamples> {
    let key = e4_tail_samples_cache_key(cfg, mirror_weight, seed, tail_window);
    {
        let cache = e4_tail_samples_cache()
            .lock()
            .expect("tail samples cache poisoned");
        if let Some(samples) = cache.get(&key) {
            return Arc::clone(samples);
        }
    }

    let samples =
        run_e4_condition_tail_samples_with_config_uncached(mirror_weight, seed, cfg, tail_window);
    let mut cache = e4_tail_samples_cache()
        .lock()
        .expect("tail samples cache poisoned");
    if let Some(samples) = cache.get(&key) {
        return Arc::clone(samples);
    }
    let arc = Arc::new(samples);
    cache.insert(key, Arc::clone(&arc));
    arc
}

#[allow(dead_code)]
fn run_e4_condition_tail_samples_with_config_uncached(
    mirror_weight: f32,
    seed: u64,
    cfg: &E4SimConfig,
    tail_window: u32,
) -> E4TailSamples {
    let space = Log2Space::new(cfg.fmin, cfg.fmax, cfg.bins_per_oct);
    let mut params = make_landscape_params(&space, cfg.fs);
    let mut landscape = Landscape::new(space.clone());
    let mut rhythms = init_e4_rhythms(cfg);

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
    pop.apply_action(
        Action::Spawn {
            group_id: E4_GROUP_VOICES,
            ids,
            spec: voice_spec,
            strategy: None,
        },
        &landscape,
        None,
    );
    apply_e4_initial_pitches(&mut pop, &space, &landscape, cfg, seed, min_freq, max_freq);

    let tail_window = tail_window.min(cfg.steps).max(1);
    let start_step = cfg.steps.saturating_sub(tail_window);
    let dt = cfg.hop as f32 / cfg.fs;
    let mut freqs_by_step: Vec<Vec<f32>> = Vec::with_capacity(tail_window as usize);
    let mut agent_freqs_by_step: Vec<Vec<E4AgentFreq>> = Vec::with_capacity(tail_window as usize);

    for step in 0..cfg.steps {
        update_e4_landscape_from_population(&space, &params, &pop, &mut landscape);
        landscape.rhythm = rhythms;
        pop.advance(cfg.hop, cfg.fs, step as u64, dt, &landscape);
        pop.cleanup_dead(step as u64, dt, false);
        rhythms.advance_in_place(dt);

        if step >= start_step {
            let agent_freqs = collect_voice_freqs_with_ids(&pop);
            if agent_freqs.len() != cfg.voice_count {
                panic!(
                    "E4 protocol violation: seed={seed} step={step} voices={} expected={}",
                    agent_freqs.len(),
                    cfg.voice_count
                );
            }
            freqs_by_step.push(agent_freqs.iter().map(|row| row.freq_hz).collect());
            agent_freqs_by_step.push(agent_freqs);
        }
    }

    E4TailSamples {
        steps_total: cfg.steps,
        tail_window,
        freqs_by_step,
        agent_freqs_by_step,
    }
}

fn run_e4_mirror_schedule_samples_with_config(
    seed: u64,
    steps_total: u32,
    schedule: &[(u32, f32)],
    cfg: &E4SimConfig,
) -> E4MirrorScheduleSamples {
    if steps_total == 0 {
        return E4MirrorScheduleSamples {
            freqs_by_step: Vec::new(),
            mirror_weight_by_step: Vec::new(),
            agent_freqs_by_step: Vec::new(),
        };
    }
    let schedule = normalize_mirror_schedule(schedule, 0.0);
    let mut sched_idx = 0usize;
    let mut current_weight = schedule[0].1;

    let space = Log2Space::new(cfg.fmin, cfg.fmax, cfg.bins_per_oct);
    let mut params = make_landscape_params(&space, cfg.fs);
    let mut landscape = Landscape::new(space.clone());
    let mut rhythms = init_e4_rhythms(cfg);

    let mut pop = Population::new(Timebase {
        fs: cfg.fs,
        hop: cfg.hop,
    });
    pop.set_seed(seed);
    apply_mirror_weight(&mut pop, &landscape, &mut params, current_weight);

    landscape = build_anchor_landscape(&space, &params, cfg.anchor_hz);

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
    pop.apply_action(
        Action::Spawn {
            group_id: E4_GROUP_VOICES,
            ids,
            spec: voice_spec,
            strategy: None,
        },
        &landscape,
        None,
    );
    apply_e4_initial_pitches(&mut pop, &space, &landscape, cfg, seed, min_freq, max_freq);

    let dt = cfg.hop as f32 / cfg.fs;
    let mut freqs_by_step = Vec::with_capacity(steps_total as usize);
    let mut mirror_weight_by_step = Vec::with_capacity(steps_total as usize);
    let mut agent_freqs_by_step = Vec::with_capacity(steps_total as usize);

    for step in 0..steps_total {
        while sched_idx + 1 < schedule.len() && step >= schedule[sched_idx + 1].0 {
            sched_idx += 1;
            current_weight = schedule[sched_idx].1;
            apply_mirror_weight(&mut pop, &landscape, &mut params, current_weight);
        }

        update_e4_landscape_from_population(&space, &params, &pop, &mut landscape);
        landscape.rhythm = rhythms;
        pop.advance(cfg.hop, cfg.fs, step as u64, dt, &landscape);
        pop.cleanup_dead(step as u64, dt, false);
        rhythms.advance_in_place(dt);

        let agent_freqs = collect_voice_freqs_with_ids(&pop);
        if agent_freqs.len() != cfg.voice_count {
            panic!(
                "E4 schedule protocol violation: seed={seed} step={step} voices={} expected={}",
                agent_freqs.len(),
                cfg.voice_count
            );
        }
        mirror_weight_by_step.push(current_weight);
        freqs_by_step.push(agent_freqs.iter().map(|row| row.freq_hz).collect());
        agent_freqs_by_step.push(agent_freqs);
    }

    E4MirrorScheduleSamples {
        freqs_by_step,
        mirror_weight_by_step,
        agent_freqs_by_step,
    }
}

fn normalize_mirror_schedule(schedule: &[(u32, f32)], default_weight: f32) -> Vec<(u32, f32)> {
    let mut points: Vec<(u32, f32)> = schedule
        .iter()
        .map(|(step, w)| (*step, w.clamp(0.0, 1.0)))
        .collect();
    points.sort_by_key(|(step, _)| *step);
    points.dedup_by(|a, b| a.0 == b.0);
    if points.is_empty() {
        points.push((0, default_weight.clamp(0.0, 1.0)));
    } else if points[0].0 != 0 {
        points.insert(0, (0, points[0].1));
    }
    points
}

fn apply_mirror_weight(
    pop: &mut Population,
    landscape: &Landscape,
    params: &mut LandscapeParams,
    mirror_weight: f32,
) {
    let update = LandscapeUpdate {
        mirror: Some(mirror_weight.clamp(0.0, 1.0)),
        ..LandscapeUpdate::default()
    };
    pop.apply_action(Action::SetHarmonicityParams { update }, landscape, None);
    if let Some(update) = pop.take_pending_update() {
        apply_params_update(params, &update);
    }
}

#[allow(dead_code)]
fn collect_voice_freqs(pop: &Population) -> Vec<f32> {
    collect_voice_freqs_with_ids(pop)
        .into_iter()
        .map(|row| row.freq_hz)
        .collect()
}

fn collect_voice_freqs_with_ids(pop: &Population) -> Vec<E4AgentFreq> {
    let mut freqs = Vec::new();
    for agent in &pop.individuals {
        if agent.metadata.group_id != E4_GROUP_VOICES {
            continue;
        }
        let freq = agent.body.base_freq_hz();
        if freq.is_finite() && freq > 0.0 {
            freqs.push(E4AgentFreq {
                agent_id: agent.id,
                freq_hz: freq,
            });
        }
    }
    freqs.sort_by_key(|row| row.agent_id);
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

fn init_e4_rhythms(cfg: &E4SimConfig) -> NeuralRhythms {
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.freq_hz = cfg.theta_freq_hz.max(0.1);
    rhythms.theta.mag = 1.0;
    rhythms.theta.alpha = 1.0;
    rhythms.delta.freq_hz = 1.0;
    rhythms.delta.mag = 1.0;
    rhythms.delta.alpha = 1.0;
    rhythms.env_open = 1.0;
    rhythms.env_level = 1.0;
    rhythms
}

fn apply_e4_initial_pitches(
    pop: &mut Population,
    space: &Log2Space,
    landscape: &Landscape,
    cfg: &E4SimConfig,
    seed: u64,
    min_freq: f32,
    max_freq: f32,
) {
    let mut rng = SmallRng::seed_from_u64(seed ^ 0xE4E4_5EED);
    let min_freq = min_freq.clamp(space.fmin, space.fmax);
    let max_freq = max_freq.clamp(space.fmin, space.fmax);
    for agent in pop.individuals.iter_mut() {
        if agent.metadata.group_id != E4_GROUP_VOICES {
            continue;
        }
        let log2_freq =
            sample_e4_initial_pitch_log2(&mut rng, space, landscape, min_freq, max_freq);
        agent.force_set_pitch_log2(log2_freq);
        agent.set_neighbor_step_cents(cfg.neighbor_step_cents);
    }
}

fn sample_e4_initial_pitch_log2(
    rng: &mut impl Rng,
    space: &Log2Space,
    landscape: &Landscape,
    min_freq: f32,
    max_freq: f32,
) -> f32 {
    let min_idx = space.index_of_freq(min_freq).unwrap_or(0);
    let max_idx = space
        .index_of_freq(max_freq)
        .unwrap_or(space.n_bins().saturating_sub(1));
    let (min_idx, max_idx) = if min_idx <= max_idx {
        (min_idx, max_idx)
    } else {
        (max_idx, min_idx)
    };
    let mut weights = Vec::with_capacity(max_idx - min_idx + 1);
    for i in min_idx..=max_idx {
        let w = landscape
            .consonance_state01
            .get(i)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        weights.push(w);
    }
    if let Ok(dist) = WeightedIndex::new(&weights) {
        let idx = min_idx + dist.sample(rng);
        return space.freq_of_index(idx).log2();
    }
    let min_l = min_freq.log2();
    let max_l = max_freq.log2();
    if min_l.is_finite() && max_l.is_finite() && min_l < max_l {
        let r = rng.random_range(min_l..max_l);
        return r;
    }
    min_freq.max(1.0).log2()
}

fn update_e4_landscape_from_population(
    space: &Log2Space,
    params: &LandscapeParams,
    pop: &Population,
    landscape: &mut Landscape,
) {
    let mut env_scan = vec![0.0f32; space.n_bins()];
    for agent in &pop.individuals {
        let freq = agent.body.base_freq_hz();
        if !freq.is_finite() || freq <= 0.0 {
            continue;
        }
        if let Some(idx) = space.index_of_freq(freq) {
            env_scan[idx] += 1.0;
        }
    }
    space.assert_scan_len_named(&env_scan, "e4_env_scan");
    let (h_pot_scan, _) = params
        .harmonicity_kernel
        .potential_h_from_log2_spectrum(&env_scan, space);
    landscape.subjective_intensity = env_scan.clone();
    landscape.nsgt_power = env_scan;
    landscape.harmonicity = h_pot_scan;
    landscape.roughness.fill(0.0);
    landscape.roughness_shape_raw.fill(0.0);
    landscape.roughness01.fill(0.0);
    landscape.recompute_consonance(params);
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
    fn e4_mirror_weight_changes_consonance_landscape() {
        let cfg = E4SimConfig::test_defaults();
        let space = Log2Space::new(cfg.fmin, cfg.fmax, cfg.bins_per_oct);
        let mut params_m0 = make_landscape_params(&space, cfg.fs);
        let mut params_m1 = make_landscape_params(&space, cfg.fs);
        params_m0.harmonicity_kernel.params.mirror_weight = 0.0;
        params_m1.harmonicity_kernel.params.mirror_weight = 1.0;

        let landscape_m0 = build_anchor_landscape(&space, &params_m0, cfg.anchor_hz);
        let landscape_m1 = build_anchor_landscape(&space, &params_m1, cfg.anchor_hz);

        let diff_sum: f32 = landscape_m0
            .consonance_state01
            .iter()
            .zip(landscape_m1.consonance_state01.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff_sum > 1e-4,
            "expected mirror weight to change consonance landscape, diff_sum={diff_sum:.6}"
        );
    }
}
