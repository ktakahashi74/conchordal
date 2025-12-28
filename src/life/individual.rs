use crate::core::landscape::Landscape;
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::core::utils::pink_noise_tick;
use crate::life::lifecycle::LifecycleConfig;
use crate::life::scenario::{
    FieldCoreConfig, HarmonicMode, ModulationCoreConfig, SoundBodyConfig, TemporalCoreConfig,
    TimbreGenotype,
};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::f32::consts::PI;

pub trait AudioAgent {
    fn id(&self) -> u64;
    fn metadata(&self) -> &AgentMetadata;
    fn render_wave(
        &mut self,
        buffer: &mut [f32],
        fs: f32,
        current_frame: u64,
        dt_sec: f32,
        landscape: &Landscape,
        global_coupling: f32,
    );
    fn render_spectrum(&mut self, amps: &mut [f32], space: &Log2Space);
    fn is_alive(&self) -> bool;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ArticulationSignal {
    pub amplitude: f32,
    pub is_active: bool,
    pub relaxation: f32,
    pub tension: f32,
}

pub trait TemporalCore {
    fn process(
        &mut self,
        consonance: f32,
        rhythms: &NeuralRhythms,
        dt: f32,
        global_coupling: f32,
    ) -> ArticulationSignal;
    fn is_alive(&self) -> bool;
}

pub trait SoundBody {
    fn base_freq_hz(&self) -> f32;
    fn set_freq(&mut self, freq: f32);
    fn set_pitch_log2(&mut self, log_freq: f32);
    fn set_amp(&mut self, amp: f32);
    fn articulate_wave(&mut self, sample: &mut f32, fs: f32, dt: f32, signal: &ArticulationSignal);
    fn project_spectral_body(
        &mut self,
        amps: &mut [f32],
        space: &Log2Space,
        signal: &ArticulationSignal,
    );
}

#[derive(Debug, Clone, Default)]
pub struct AgentMetadata {
    pub id: u64,
    pub tag: Option<String>,
    pub group_idx: usize,
    pub member_idx: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Sensitivity {
    pub delta: f32,
    pub theta: f32,
    pub alpha: f32,
    pub beta: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct ModulationState {
    pub exploration: f32,
    pub persistence: f32,
    pub habituation_sensitivity: f32,
}

pub trait ModulationCore {
    fn state(&self) -> ModulationState;
    fn set_exploration(&mut self, value: f32);
    fn set_persistence(&mut self, value: f32);
    fn set_habituation_sensitivity(&mut self, value: f32);
}

#[derive(Debug, Clone)]
pub struct StaticModulationCore {
    state: ModulationState,
}

impl StaticModulationCore {
    pub fn new(state: ModulationState) -> Self {
        Self { state }
    }
}

impl ModulationCore for StaticModulationCore {
    fn state(&self) -> ModulationState {
        self.state
    }

    fn set_exploration(&mut self, value: f32) {
        self.state.exploration = value.clamp(0.0, 1.0);
    }

    fn set_persistence(&mut self, value: f32) {
        self.state.persistence = value.clamp(0.0, 1.0);
    }

    fn set_habituation_sensitivity(&mut self, value: f32) {
        self.state.habituation_sensitivity = value.max(0.0);
    }
}

#[derive(Debug, Clone)]
pub enum AnyModulationCore {
    Static(StaticModulationCore),
}

impl ModulationCore for AnyModulationCore {
    fn state(&self) -> ModulationState {
        match self {
            AnyModulationCore::Static(core) => core.state(),
        }
    }

    fn set_exploration(&mut self, value: f32) {
        match self {
            AnyModulationCore::Static(core) => core.set_exploration(value),
        }
    }

    fn set_persistence(&mut self, value: f32) {
        match self {
            AnyModulationCore::Static(core) => core.set_persistence(value),
        }
    }

    fn set_habituation_sensitivity(&mut self, value: f32) {
        match self {
            AnyModulationCore::Static(core) => core.set_habituation_sensitivity(value),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AudioSample {
    pub sample: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArticulationState {
    Idle,
    Attack,
    Decay,
}

#[derive(Debug, Clone)]
pub struct PinkNoise {
    rng: SmallRng,
    b0: f32,
    b1: f32,
    b2: f32,
    gain: f32,
}

impl PinkNoise {
    pub fn new(seed: u64, gain: f32) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            gain,
        }
    }

    pub fn sample(&mut self) -> f32 {
        let pink = pink_noise_tick(&mut self.rng, &mut self.b0, &mut self.b1, &mut self.b2);
        pink * self.gain
    }
}

fn add_log2_energy(amps: &mut [f32], space: &Log2Space, freq_hz: f32, energy: f32) {
    if !freq_hz.is_finite() || energy == 0.0 {
        return;
    }
    if freq_hz < space.fmin || freq_hz > space.fmax {
        return;
    }
    let log_f = freq_hz.log2();
    let base = space.centers_log2[0];
    let step = space.step();
    let pos = (log_f - base) / step;
    let idx_base = pos.floor();
    let idx = idx_base as isize;
    if idx < 0 {
        return;
    }
    let idx = idx as usize;
    let frac = pos - idx_base;
    if idx + 1 < amps.len() {
        amps[idx] += energy * (1.0 - frac);
        amps[idx + 1] += energy * frac;
    } else if idx < amps.len() {
        amps[idx] += energy;
    }
}

#[derive(Debug, Clone)]
pub struct KuramotoCore {
    pub energy: f32,
    pub basal_cost: f32,
    pub action_cost: f32,
    pub recharge_rate: f32,
    pub sensitivity: Sensitivity,
    pub rhythm_phase: f32,
    pub rhythm_freq: f32,
    pub env_level: f32,
    pub state: ArticulationState,
    pub attack_step: f32,
    pub decay_factor: f32,
    pub retrigger: bool,
    pub noise_1f: PinkNoise,
    pub confidence: f32,
    pub gate_threshold: f32,
}

impl TemporalCore for KuramotoCore {
    fn process(
        &mut self,
        consonance: f32,
        rhythms: &NeuralRhythms,
        dt: f32,
        global_coupling: f32,
    ) -> ArticulationSignal {
        self.energy -= self.basal_cost * dt;
        if self.energy <= 0.0 {
            self.state = ArticulationState::Idle;
            self.env_level = 0.0;
            return ArticulationSignal::default();
        }

        let delta_phase = rhythms.delta.phase;
        let delta_mag = rhythms.delta.mag * self.sensitivity.delta;
        let theta_signal = rhythms.theta.mag * rhythms.theta.phase.cos() * self.sensitivity.theta;
        let trigger_force = theta_signal * consonance;

        let beta_err = rhythms.beta.mag * self.sensitivity.beta;
        self.confidence = 1.0 - beta_err;
        let coupling = (self.sensitivity.beta * (1.0 / (self.confidence + 1e-3))).min(10.0);
        let noise = self.noise_1f.sample();
        let rhythm_omega = 2.0 * PI * self.rhythm_freq;
        let d_phi = rhythm_omega
            + noise
            + coupling * global_coupling * (delta_phase - self.rhythm_phase).sin();
        self.rhythm_phase += d_phi * dt;

        if self.rhythm_phase >= 2.0 * PI {
            self.rhythm_phase -= 2.0 * PI;
            let gate_bias = 0.2;
            let gate = delta_mag + trigger_force + gate_bias;
            if gate > self.gate_threshold && self.state == ArticulationState::Idle && self.retrigger
            {
                self.env_level = 0.0;
                self.state = ArticulationState::Attack;
                self.energy -= self.action_cost;
                if consonance > 0.5 {
                    self.energy += self.recharge_rate * consonance;
                }
            }
        }

        match self.state {
            ArticulationState::Attack => {
                self.env_level += self.attack_step;
                if self.env_level >= 1.0 {
                    self.env_level = 1.0;
                    self.state = ArticulationState::Decay;
                }
            }
            ArticulationState::Decay => {
                self.env_level *= self.decay_factor;
                if self.env_level < 0.001 {
                    self.env_level = 0.0;
                    self.state = ArticulationState::Idle;
                }
            }
            ArticulationState::Idle => {}
        }

        let relaxation = rhythms.alpha.mag * self.sensitivity.alpha;
        let tension = rhythms.beta.mag * self.sensitivity.beta;
        let is_active = self.env_level > 0.0 && self.state != ArticulationState::Idle;
        ArticulationSignal {
            amplitude: self.env_level,
            is_active,
            relaxation,
            tension,
        }
    }

    fn is_alive(&self) -> bool {
        if self.energy <= 0.0 {
            return false;
        }
        if !self.retrigger && self.state == ArticulationState::Idle {
            return false;
        }
        true
    }
}

#[derive(Debug, Clone)]
pub struct SequencedCore {
    pub timer: f32,
    pub duration: f32,
    pub env_level: f32,
}

impl TemporalCore for SequencedCore {
    fn process(
        &mut self,
        _consonance: f32,
        rhythms: &NeuralRhythms,
        dt: f32,
        _global_coupling: f32,
    ) -> ArticulationSignal {
        self.timer += dt.max(0.0);
        let active = self.timer < self.duration;
        self.env_level = if active { 1.0 } else { 0.0 };
        ArticulationSignal {
            amplitude: self.env_level,
            is_active: active,
            relaxation: rhythms.alpha.mag,
            tension: 0.0,
        }
    }

    fn is_alive(&self) -> bool {
        self.timer < self.duration
    }
}

#[derive(Debug, Clone)]
pub struct DroneCore {
    pub phase: f32,
    pub sway_rate: f32,
}

impl TemporalCore for DroneCore {
    fn process(
        &mut self,
        _consonance: f32,
        rhythms: &NeuralRhythms,
        dt: f32,
        _global_coupling: f32,
    ) -> ArticulationSignal {
        let omega = 2.0 * PI * self.sway_rate.max(0.01);
        self.phase = (self.phase + omega * dt).rem_euclid(2.0 * PI);
        let lfo = 0.5 * (self.phase.sin() + 1.0);
        let relax_boost = 1.0 + rhythms.alpha.mag * 0.5;
        let amplitude = (0.3 + 0.7 * lfo) * relax_boost;
        ArticulationSignal {
            amplitude: amplitude.clamp(0.0, 1.0),
            is_active: true,
            relaxation: rhythms.alpha.mag,
            tension: rhythms.beta.mag * 0.25,
        }
    }

    fn is_alive(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone)]
pub enum AnyTemporalCore {
    Entrain(KuramotoCore),
    Seq(SequencedCore),
    Drone(DroneCore),
}

impl TemporalCore for AnyTemporalCore {
    fn process(
        &mut self,
        consonance: f32,
        rhythms: &NeuralRhythms,
        dt: f32,
        global_coupling: f32,
    ) -> ArticulationSignal {
        match self {
            AnyTemporalCore::Entrain(c) => c.process(consonance, rhythms, dt, global_coupling),
            AnyTemporalCore::Seq(c) => c.process(consonance, rhythms, dt, global_coupling),
            AnyTemporalCore::Drone(c) => c.process(consonance, rhythms, dt, global_coupling),
        }
    }

    fn is_alive(&self) -> bool {
        match self {
            AnyTemporalCore::Entrain(c) => c.is_alive(),
            AnyTemporalCore::Seq(c) => c.is_alive(),
            AnyTemporalCore::Drone(c) => c.is_alive(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TargetProposal {
    pub target_pitch_log2: f32,
    pub salience: f32,
}

pub trait FieldCore {
    fn propose_target<R: Rng + ?Sized>(
        &mut self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        current_freq_hz: f32,
        integration_window: f32,
        landscape: &Landscape,
        modulation: ModulationState,
        rng: &mut R,
    ) -> TargetProposal;
}

#[derive(Debug, Clone)]
pub struct PitchHillClimbFieldCore {
    neighbor_step_log2: f32,
    tessitura_center: f32,
    tessitura_gravity: f32,
    satiety_weight: f32,
    improvement_threshold: f32,
}

impl PitchHillClimbFieldCore {
    pub fn new(
        neighbor_step_cents: f32,
        tessitura_center: f32,
        tessitura_gravity: f32,
        satiety_weight: f32,
        improvement_threshold: f32,
    ) -> Self {
        Self {
            neighbor_step_log2: neighbor_step_cents / 1200.0,
            tessitura_center,
            tessitura_gravity,
            satiety_weight,
            improvement_threshold,
        }
    }
}

impl FieldCore for PitchHillClimbFieldCore {
    fn propose_target<R: Rng + ?Sized>(
        &mut self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        _current_freq_hz: f32,
        integration_window: f32,
        landscape: &Landscape,
        modulation: ModulationState,
        rng: &mut R,
    ) -> TargetProposal {
        let (fmin, fmax) = landscape.freq_bounds_log2();
        let current_target_log2 = current_target_log2.clamp(fmin, fmax);
        let perfect_fifth = 1.5f32.log2();
        let imperfect_fifth = 0.66f32.log2();
        let mut candidates = vec![
            current_target_log2,
            current_target_log2 + self.neighbor_step_log2,
            current_target_log2 - self.neighbor_step_log2,
            current_target_log2 + perfect_fifth,
            current_target_log2 + imperfect_fifth,
        ];
        candidates.retain(|f| f.is_finite());

        let adjusted_score = |pitch_log2: f32| -> f32 {
            let clamped = pitch_log2.clamp(fmin, fmax);
            let score = landscape.evaluate_pitch01_log2(clamped);
            let distance_oct = (clamped - current_pitch_log2).abs();
            let penalty = distance_oct * integration_window * 0.5;
            let dist = clamped - self.tessitura_center;
            let gravity_penalty = dist * dist * self.tessitura_gravity;
            let satiety = landscape.get_spectral_satiety(2.0f32.powf(clamped));
            let mut adjusted = score - penalty - gravity_penalty;
            if satiety > 1.0 {
                adjusted -= (satiety - 1.0) * self.satiety_weight;
            }
            adjusted
        };

        let mut best_pitch = current_target_log2;
        let mut best_score = f32::MIN;
        for p in candidates {
            let clamped = p.clamp(fmin, fmax);
            let adjusted = adjusted_score(clamped);
            if adjusted > best_score {
                best_score = adjusted;
                best_pitch = clamped;
            }
        }

        let current_adjusted = adjusted_score(current_target_log2);
        let improvement = best_score - current_adjusted;
        let mut target_pitch_log2 = current_target_log2;

        if improvement > self.improvement_threshold {
            target_pitch_log2 = best_pitch;
        } else {
            let satisfaction = ((current_adjusted + 1.0) * 0.5).clamp(0.0, 1.0);
            let habituation_penalty =
                (1.0 - satisfaction) * modulation.habituation_sensitivity.max(0.0);
            let mut stay_prob =
                (modulation.persistence.clamp(0.0, 1.0) * satisfaction) - habituation_penalty;
            stay_prob = stay_prob.clamp(0.0, 1.0);
            let exploration = modulation.exploration.clamp(0.0, 1.0);
            stay_prob = (stay_prob * (1.0 - exploration)).clamp(0.0, 1.0);
            if rng.random_range(0.0..1.0) > stay_prob {
                target_pitch_log2 = best_pitch;
            }
        }

        TargetProposal {
            target_pitch_log2,
            salience: (improvement / 0.2).clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AnyFieldCore {
    PitchHillClimb(PitchHillClimbFieldCore),
}

impl FieldCore for AnyFieldCore {
    fn propose_target<R: Rng + ?Sized>(
        &mut self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        current_freq_hz: f32,
        integration_window: f32,
        landscape: &Landscape,
        modulation: ModulationState,
        rng: &mut R,
    ) -> TargetProposal {
        match self {
            AnyFieldCore::PitchHillClimb(core) => core.propose_target(
                current_pitch_log2,
                current_target_log2,
                current_freq_hz,
                integration_window,
                landscape,
                modulation,
                rng,
            ),
        }
    }
}

impl AnyTemporalCore {
    pub fn from_config<R: Rng + ?Sized>(
        config: &TemporalCoreConfig,
        fs: f32,
        noise_seed: u64,
        rng: &mut R,
    ) -> Self {
        match config {
            TemporalCoreConfig::Entrain {
                lifecycle,
                rhythm_freq,
                rhythm_sensitivity,
            } => {
                let (
                    energy,
                    basal_cost,
                    recharge_rate,
                    attack_step,
                    decay_factor,
                    state,
                    sensitivity,
                    retrigger,
                    action_cost,
                ) = envelope_from_lifecycle(lifecycle, fs);
                AnyTemporalCore::Entrain(KuramotoCore {
                    energy,
                    basal_cost,
                    action_cost,
                    recharge_rate,
                    sensitivity: Sensitivity {
                        beta: rhythm_sensitivity.unwrap_or(sensitivity.beta),
                        ..sensitivity
                    },
                    rhythm_phase: 0.0,
                    rhythm_freq: rhythm_freq.unwrap_or_else(|| rng.random_range(0.5..3.0)),
                    env_level: 0.0,
                    state,
                    attack_step,
                    decay_factor,
                    retrigger,
                    noise_1f: PinkNoise::new(noise_seed, 0.001),
                    confidence: 1.0,
                    gate_threshold: 0.02,
                })
            }
            TemporalCoreConfig::Seq { duration } => AnyTemporalCore::Seq(SequencedCore {
                timer: 0.0,
                duration: duration.max(0.0),
                env_level: 0.0,
            }),
            TemporalCoreConfig::Drone { sway } => {
                let sway_rate = sway.unwrap_or(0.05);
                let sway_rate = if sway_rate <= 0.0 { 0.05 } else { sway_rate };
                AnyTemporalCore::Drone(DroneCore {
                    phase: rng.random_range(0.0..std::f32::consts::TAU),
                    sway_rate,
                })
            }
        }
    }
}

impl AnyFieldCore {
    pub fn from_config<R: Rng + ?Sized>(
        config: &FieldCoreConfig,
        initial_pitch_log2: f32,
        _rng: &mut R,
    ) -> Self {
        match config {
            FieldCoreConfig::PitchHillClimb {
                neighbor_step_cents,
                tessitura_gravity,
                satiety_weight,
                improvement_threshold,
            } => {
                let neighbor_step_cents = neighbor_step_cents.unwrap_or(200.0);
                let tessitura_gravity = tessitura_gravity.unwrap_or(0.1);
                let satiety_weight = satiety_weight.unwrap_or(2.0);
                let improvement_threshold = improvement_threshold.unwrap_or(0.1);
                AnyFieldCore::PitchHillClimb(PitchHillClimbFieldCore::new(
                    neighbor_step_cents,
                    initial_pitch_log2,
                    tessitura_gravity,
                    satiety_weight,
                    improvement_threshold,
                ))
            }
        }
    }
}

impl AnyModulationCore {
    pub fn from_config(config: &ModulationCoreConfig) -> Self {
        match config {
            ModulationCoreConfig::Static {
                exploration,
                persistence,
                habituation_sensitivity,
            } => {
                let state = ModulationState {
                    exploration: exploration.unwrap_or(0.0).clamp(0.0, 1.0),
                    persistence: persistence.unwrap_or(0.5).clamp(0.0, 1.0),
                    habituation_sensitivity: habituation_sensitivity.unwrap_or(1.0).max(0.0),
                };
                AnyModulationCore::Static(StaticModulationCore::new(state))
            }
        }
    }
}

fn envelope_from_lifecycle(
    lifecycle: &LifecycleConfig,
    fs: f32,
) -> (
    f32,
    f32,
    f32,
    f32,
    f32,
    ArticulationState,
    Sensitivity,
    bool,
    f32,
) {
    match lifecycle {
        LifecycleConfig::Decay {
            initial_energy,
            half_life_sec,
            attack_sec,
        } => {
            let atk = attack_sec.max(0.0005);
            let attack_step = 1.0 / (fs * atk);
            let decay_sec = half_life_sec.max(0.01);
            let decay_factor = (-1.0f32 / (fs * decay_sec)).exp();
            let basal = 0.0;
            (
                *initial_energy,
                basal,
                0.0,
                attack_step,
                decay_factor,
                ArticulationState::Attack,
                Sensitivity::default(),
                false,
                0.02,
            )
        }
        LifecycleConfig::Sustain {
            initial_energy,
            metabolism_rate,
            recharge_rate,
            action_cost,
            envelope,
        } => {
            let atk = envelope.attack_sec.max(0.0005);
            let attack_step = 1.0 / (fs * atk);
            let decay_sec = envelope.decay_sec.max(0.01);
            let decay_factor = (-1.0f32 / (fs * decay_sec)).exp();
            (
                *initial_energy,
                *metabolism_rate,
                recharge_rate.unwrap_or(0.5),
                attack_step,
                decay_factor,
                ArticulationState::Idle,
                Sensitivity {
                    delta: 1.0,
                    theta: 1.0,
                    alpha: 0.5,
                    beta: 0.5,
                },
                true,
                action_cost.unwrap_or(0.02),
            )
        }
    }
}

#[derive(Debug, Clone)]
pub struct SineBody {
    pub freq_hz: f32,
    pub amp: f32,
    pub audio_phase: f32,
}

impl SoundBody for SineBody {
    fn base_freq_hz(&self) -> f32 {
        self.freq_hz
    }

    fn set_freq(&mut self, freq: f32) {
        self.freq_hz = freq;
    }

    fn set_pitch_log2(&mut self, log_freq: f32) {
        self.freq_hz = 2.0f32.powf(log_freq);
    }

    fn set_amp(&mut self, amp: f32) {
        self.amp = amp;
    }

    fn articulate_wave(
        &mut self,
        sample: &mut f32,
        _fs: f32,
        dt: f32,
        signal: &ArticulationSignal,
    ) {
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        self.audio_phase = (self.audio_phase + 2.0 * PI * self.freq_hz * dt).rem_euclid(2.0 * PI);
        *sample += self.amp * signal.amplitude * self.audio_phase.sin();
    }

    fn project_spectral_body(
        &mut self,
        amps: &mut [f32],
        space: &Log2Space,
        signal: &ArticulationSignal,
    ) {
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        let energy = self.amp.max(0.0) * signal.amplitude;
        add_log2_energy(amps, space, self.freq_hz, energy);
    }
}

#[derive(Debug, Clone)]
pub struct HarmonicBody {
    pub base_freq_hz: f32,
    pub amp: f32,
    pub genotype: TimbreGenotype,
    pub lfo_phase: f32,
    pub phases: Vec<f32>,
    pub detune_phases: Vec<f32>,
    pub jitter_gen: PinkNoise,
}

impl HarmonicBody {
    fn partial_ratio(&self, idx: usize) -> f32 {
        let k = (idx + 1) as f32;
        let base = match self.genotype.mode {
            HarmonicMode::Harmonic => k,
            HarmonicMode::Metallic => k.powf(1.4),
        };
        let stretch = 1.0 + self.genotype.stiffness * k * k;
        (base * stretch).max(0.1)
    }

    fn compute_partial_amp(&self, idx: usize, current_energy: f32) -> f32 {
        let k = (idx + 1) as f32;
        let slope = self.genotype.brightness.max(0.0);
        let mut amp = 1.0 / k.powf(slope.max(1e-6));
        if (idx + 1).is_multiple_of(2) {
            amp *= 1.0 - self.genotype.comb.clamp(0.0, 1.0);
        }
        let damping = self.genotype.damping.max(0.0);
        if damping > 0.0 {
            let energy = current_energy.clamp(0.0, 1.0);
            amp *= energy.powf(damping * k);
        }
        amp
    }

    fn partial_count(&self) -> usize {
        self.phases.len().min(self.detune_phases.len())
    }
}

impl SoundBody for HarmonicBody {
    fn base_freq_hz(&self) -> f32 {
        self.base_freq_hz
    }

    fn set_freq(&mut self, freq: f32) {
        self.base_freq_hz = freq;
    }

    fn set_pitch_log2(&mut self, log_freq: f32) {
        self.base_freq_hz = 2.0f32.powf(log_freq);
    }

    fn set_amp(&mut self, amp: f32) {
        self.amp = amp;
    }

    fn articulate_wave(
        &mut self,
        sample: &mut f32,
        _fs: f32,
        dt: f32,
        signal: &ArticulationSignal,
    ) {
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        let partials = self.partial_count();
        if partials == 0 {
            return;
        }
        self.lfo_phase =
            (self.lfo_phase + 2.0 * PI * self.genotype.vibrato_rate * dt).rem_euclid(2.0 * PI);
        let vibrato =
            self.genotype.vibrato_depth * (1.0 + signal.relaxation * 0.5) * self.lfo_phase.sin();
        let jitter_scale = (1.0 + signal.tension * 0.5) * (signal.amplitude + 0.1);
        let jitter = self.jitter_gen.sample() * self.genotype.jitter * jitter_scale;
        let base_freq = (self.base_freq_hz * (1.0 + vibrato + jitter)).max(1.0);
        let unison = (self.genotype.unison * (1.0 + signal.relaxation * 0.5)).max(0.0);

        let mut acc = 0.0;
        for idx in 0..partials {
            let ratio = self.partial_ratio(idx);
            let freq = base_freq * ratio;
            if !freq.is_finite() || freq <= 0.0 {
                continue;
            }
            let part_amp = self.compute_partial_amp(idx, signal.amplitude);
            if part_amp <= 0.0 {
                continue;
            }
            let phase = &mut self.phases[idx];
            *phase = (*phase + 2.0 * PI * freq * dt).rem_euclid(2.0 * PI);
            let mut part_sample = phase.sin();
            if unison > 0.0 {
                let detune_ratio = 1.0 + unison * 0.02;
                let detune_phase = &mut self.detune_phases[idx];
                *detune_phase =
                    (*detune_phase + 2.0 * PI * freq * detune_ratio * dt).rem_euclid(2.0 * PI);
                part_sample = 0.5 * (part_sample + detune_phase.sin());
            }
            acc += part_amp * part_sample;
        }
        *sample += self.amp * signal.amplitude * acc;
    }

    fn project_spectral_body(
        &mut self,
        amps: &mut [f32],
        space: &Log2Space,
        signal: &ArticulationSignal,
    ) {
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        let partials = self.partial_count();
        if partials == 0 {
            return;
        }
        let amp_scale = self.amp.max(0.0) * signal.amplitude;
        for idx in 0..partials {
            let ratio = self.partial_ratio(idx);
            let freq = self.base_freq_hz * ratio;
            let part_amp = self.compute_partial_amp(idx, signal.amplitude);
            add_log2_energy(amps, space, freq, amp_scale * part_amp);
        }
    }
}

#[derive(Debug, Clone)]
pub enum AnySoundBody {
    Sine(SineBody),
    Harmonic(HarmonicBody),
}

impl AnySoundBody {
    pub fn from_config<R: Rng + ?Sized>(
        config: &SoundBodyConfig,
        freq_hz: f32,
        amp: f32,
        rng: &mut R,
    ) -> Self {
        match config {
            SoundBodyConfig::Sine { phase } => AnySoundBody::Sine(SineBody {
                freq_hz,
                amp,
                audio_phase: phase.unwrap_or_else(|| rng.random_range(0.0..std::f32::consts::TAU)),
            }),
            SoundBodyConfig::Harmonic { genotype, partials } => {
                let partials = partials.unwrap_or(16).max(1);
                let mut phases = Vec::with_capacity(partials);
                let mut detune_phases = Vec::with_capacity(partials);
                for _ in 0..partials {
                    phases.push(rng.random_range(0.0..std::f32::consts::TAU));
                    detune_phases.push(rng.random_range(0.0..std::f32::consts::TAU));
                }
                AnySoundBody::Harmonic(HarmonicBody {
                    base_freq_hz: freq_hz,
                    amp,
                    genotype: genotype.clone(),
                    lfo_phase: 0.0,
                    phases,
                    detune_phases,
                    jitter_gen: PinkNoise::new(rng.next_u64(), 0.001),
                })
            }
        }
    }
}

impl SoundBody for AnySoundBody {
    fn base_freq_hz(&self) -> f32 {
        match self {
            AnySoundBody::Sine(body) => body.base_freq_hz(),
            AnySoundBody::Harmonic(body) => body.base_freq_hz(),
        }
    }

    fn set_freq(&mut self, freq: f32) {
        match self {
            AnySoundBody::Sine(body) => body.set_freq(freq),
            AnySoundBody::Harmonic(body) => body.set_freq(freq),
        }
    }

    fn set_pitch_log2(&mut self, log_freq: f32) {
        match self {
            AnySoundBody::Sine(body) => body.set_pitch_log2(log_freq),
            AnySoundBody::Harmonic(body) => body.set_pitch_log2(log_freq),
        }
    }

    fn set_amp(&mut self, amp: f32) {
        match self {
            AnySoundBody::Sine(body) => body.set_amp(amp),
            AnySoundBody::Harmonic(body) => body.set_amp(amp),
        }
    }

    fn articulate_wave(&mut self, sample: &mut f32, fs: f32, dt: f32, signal: &ArticulationSignal) {
        match self {
            AnySoundBody::Sine(body) => body.articulate_wave(sample, fs, dt, signal),
            AnySoundBody::Harmonic(body) => body.articulate_wave(sample, fs, dt, signal),
        }
    }

    fn project_spectral_body(
        &mut self,
        amps: &mut [f32],
        space: &Log2Space,
        signal: &ArticulationSignal,
    ) {
        match self {
            AnySoundBody::Sine(body) => body.project_spectral_body(amps, space, signal),
            AnySoundBody::Harmonic(body) => body.project_spectral_body(amps, space, signal),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Individual {
    pub id: u64,
    pub metadata: AgentMetadata,
    pub temporal: AnyTemporalCore,
    pub field: AnyFieldCore,
    pub modulation: AnyModulationCore,
    pub body: AnySoundBody,
    pub last_signal: ArticulationSignal,
    pub release_gain: f32,
    pub release_sec: f32,
    pub release_pending: bool,
    pub target_pitch_log2: f32,
    pub integration_window: f32,
    pub accumulated_time: f32,
    pub breath_gain: f32,
    pub last_theta_sample: f32,
    pub last_target_salience: f32,
    pub rng: SmallRng,
}

impl Individual {
    pub fn metadata(&self) -> &AgentMetadata {
        &self.metadata
    }

    pub fn force_set_pitch_log2(&mut self, log_freq: f32) {
        let log_freq = log_freq.max(0.0);
        self.body.set_pitch_log2(log_freq);
        self.target_pitch_log2 = log_freq;
        self.breath_gain = 1.0;
        self.accumulated_time = 0.0;
        self.last_theta_sample = 0.0;
        self.last_target_salience = 0.0;
    }

    pub fn update_field_target(&mut self, rhythms: &NeuralRhythms, dt: f32, landscape: &Landscape) {
        let dt = dt.max(0.0);
        let current_freq = self.body.base_freq_hz().max(1.0);
        let current_pitch_log2 = current_freq.log2();
        if self.target_pitch_log2 <= 0.0 {
            self.target_pitch_log2 = current_pitch_log2;
        }
        self.integration_window = 2.0 + 10.0 / current_freq.max(1.0);
        self.accumulated_time += dt;

        let theta_signal = rhythms.theta.mag * rhythms.theta.phase.sin();
        let theta_cross = self.last_theta_sample <= 0.0 && theta_signal > 0.0;
        self.last_theta_sample = theta_signal;

        if theta_cross && self.accumulated_time >= self.integration_window {
            self.accumulated_time = 0.0;
            let modulation = self.modulation.state();
            let proposal = self.field.propose_target(
                current_pitch_log2,
                self.target_pitch_log2,
                current_freq,
                self.integration_window,
                landscape,
                modulation,
                &mut self.rng,
            );
            self.target_pitch_log2 = proposal.target_pitch_log2;
            self.last_target_salience = proposal.salience;
        }

        let (fmin, fmax) = landscape.freq_bounds_log2();
        self.target_pitch_log2 = self.target_pitch_log2.clamp(fmin, fmax);
        let distance_cents = 1200.0 * (self.target_pitch_log2 - current_pitch_log2).abs();
        let move_threshold = 10.0;
        if distance_cents > move_threshold {
            self.breath_gain = (self.breath_gain - dt * 1.5).max(0.0);
            if self.breath_gain < 0.1 {
                self.body.set_pitch_log2(self.target_pitch_log2);
            }
        } else {
            self.body.set_pitch_log2(self.target_pitch_log2);
            let attack_rate = 1.0 + rhythms.beta.mag;
            self.breath_gain = (self.breath_gain + dt * attack_rate).clamp(0.0, 1.0);
        }
    }

    pub fn start_release(&mut self, release_sec: f32) {
        if self.release_pending {
            return;
        }
        self.release_pending = true;
        self.release_sec = release_sec.max(1e-4);
        self.release_gain = self.release_gain.clamp(0.0, 1.0);
    }
}

impl AudioAgent for Individual {
    fn id(&self) -> u64 {
        self.id
    }

    fn metadata(&self) -> &AgentMetadata {
        self.metadata()
    }

    fn render_wave(
        &mut self,
        buffer: &mut [f32],
        fs: f32,
        _current_frame: u64,
        dt_sec: f32,
        landscape: &Landscape,
        global_coupling: f32,
    ) {
        if buffer.is_empty() {
            return;
        }
        let dt_per_sample = dt_sec / buffer.len() as f32;
        let rhythms = landscape.rhythm;
        for sample in buffer.iter_mut() {
            self.update_field_target(&rhythms, dt_per_sample, landscape);
            let consonance = landscape.evaluate_pitch01(self.body.base_freq_hz());
            let mut signal =
                self.temporal
                    .process(consonance, &rhythms, dt_per_sample, global_coupling);
            signal.amplitude *= self.breath_gain;
            if self.release_pending {
                let step = dt_per_sample / self.release_sec.max(1e-6);
                self.release_gain = (self.release_gain - step).max(0.0);
            }
            signal.amplitude *= self.release_gain;
            signal.is_active = signal.is_active && signal.amplitude > 0.0;
            self.last_signal = signal;
            if !signal.is_active {
                continue;
            }
            self.body
                .articulate_wave(sample, fs, dt_per_sample, &signal);
        }
    }

    fn render_spectrum(&mut self, amps: &mut [f32], space: &Log2Space) {
        let signal = self.last_signal;
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        self.body.project_spectral_body(amps, space, &signal);
    }

    fn is_alive(&self) -> bool {
        self.temporal.is_alive() && self.release_gain > 0.0
    }
}
