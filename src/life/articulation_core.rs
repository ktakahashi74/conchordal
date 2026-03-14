use crate::core::float::clamp01_finite;
use crate::core::modulation::NeuralRhythms;
use crate::core::phase::{SlidingPlv, angle_diff_pm_pi, wrap_0_tau};
use crate::core::utils::pink_noise_tick;
use crate::life::articulation_envelope::step_attack_decay_envelope;
use crate::life::constants::{MAX_COUPLING_MULT, MAX_RECHARGE_MULT};
use crate::life::lifecycle::LifecycleConfig;
use crate::life::metabolism_policy::MetabolismPolicy;
use crate::life::scenario::{
    ArticulationCoreConfig, MetabolismRhythmReward, PhonationMode, RhythmCouplingMode,
    RhythmRewardMetric,
};
use crate::life::sound::{AutonomousPulseSpec, RenderModulatorSpec, RenderModulatorStateKind};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::f32::consts::{PI, TAU};

#[derive(Debug, Clone, Copy, Default)]
pub struct ArticulationSignal {
    pub amplitude: f32,
    pub is_active: bool,
    pub relaxation: f32,
    pub tension: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PlannedPitch {
    pub target_pitch_log2: f32,
    pub jump_cents_abs: f32,
    pub salience: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PlannedGate {
    pub gate: f32,
}

impl PlannedGate {
    pub fn update(&mut self, planned: &PlannedPitch, rhythms: &NeuralRhythms, dt: f32) -> bool {
        let dt = dt.max(0.0);
        let move_threshold = 10.0;
        if planned.jump_cents_abs > move_threshold {
            self.gate = (self.gate - dt * 1.5).max(0.0);
            self.gate < 0.1
        } else {
            let attack_rate = 1.0 + rhythms.theta.beta;
            self.gate = (self.gate + dt * attack_rate).clamp(0.0, 1.0);
            true
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArticulationWrapper {
    pub core: AnyArticulationCore,
    pub planned_gate: PlannedGate,
}

impl ArticulationWrapper {
    pub fn new(mut core: AnyArticulationCore, gate: f32, autonomous_attack: bool) -> Self {
        if let AnyArticulationCore::Entrain(ref mut k) = core {
            k.autonomous_attack = autonomous_attack;
        }
        Self {
            core,
            planned_gate: PlannedGate { gate },
        }
    }

    pub fn process(
        &mut self,
        consonance: f32,
        rhythms: &NeuralRhythms,
        dt: f32,
        global_coupling: f32,
    ) -> ArticulationSignal {
        self.core.process(consonance, rhythms, dt, global_coupling)
    }

    pub fn is_alive(&self) -> bool {
        self.core.is_alive()
    }

    pub fn update_gate(
        &mut self,
        planned: &PlannedPitch,
        rhythms: &NeuralRhythms,
        dt: f32,
    ) -> bool {
        self.planned_gate.update(planned, rhythms, dt)
    }

    pub fn gate(&self) -> f32 {
        self.planned_gate.gate
    }

    pub fn set_autonomous_attack_enabled(&mut self, enabled: bool) {
        if let AnyArticulationCore::Entrain(core) = &mut self.core {
            core.autonomous_attack = enabled;
        }
    }

    pub fn set_gate(&mut self, gate: f32) {
        self.planned_gate.gate = gate.clamp(0.0, 1.0);
    }

    pub fn apply_phonation_onset(&mut self, consonance: f32, strength: f32) {
        self.core.apply_phonation_onset(consonance, strength);
    }

    pub fn vitality_scalar(&self) -> f32 {
        match &self.core {
            AnyArticulationCore::Entrain(core) => clamp01_finite(core.vitality_level),
            AnyArticulationCore::Seq(_) | AnyArticulationCore::Drone(_) => 1.0,
        }
    }

    pub fn render_modulator_spec(&self, phonation_mode: PhonationMode) -> RenderModulatorSpec {
        self.core.render_modulator_spec(phonation_mode)
    }
}

pub trait ArticulationCore {
    fn process(
        &mut self,
        consonance: f32,
        rhythms: &NeuralRhythms,
        dt: f32,
        global_coupling: f32,
    ) -> ArticulationSignal;
    fn is_alive(&self) -> bool;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Sensitivity {
    pub delta: f32,
    pub theta: f32,
    pub alpha: f32,
    pub beta: f32,
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

#[derive(Debug, Clone, Default)]
struct KuramotoMetrics {
    last_k_eff: f32,
    total_attacks: u32,
}

#[derive(Debug, Clone, Default)]
struct KuramotoTelemetry {
    sliding_plv: Option<SlidingPlv>,
}

#[derive(Debug, Clone)]
pub struct KuramotoCore {
    pub energy: f32,
    pub energy_cap: f32,
    pub vitality_exponent: f32,
    pub vitality_level: f32,
    pub basal_cost: f32,
    pub action_cost: f32,
    pub recharge_rate: f32,
    pub sensitivity: Sensitivity,
    pub rhythm_coupling: RhythmCouplingMode,
    pub rhythm_reward: Option<MetabolismRhythmReward>,
    pub rhythm_phase: f32,
    pub rhythm_freq: f32,
    pub omega_rad: f32,
    pub phase_offset: f32,
    pub env_level: f32,
    pub state: ArticulationState,
    pub attack_step: f32,
    pub decay_rate: f32,
    pub retrigger: bool,
    pub noise_1f: PinkNoise,
    pub base_sigma: f32,
    pub beta_gain: f32,
    pub k_omega: f32,
    pub bootstrap_timer: f32,
    pub env_open_threshold: f32,
    pub env_level_min: f32,
    pub mag_threshold: f32,
    pub alpha_threshold: f32,
    pub beta_threshold: f32,
    pub autonomous_attack: bool,
    metrics: KuramotoMetrics,
    telemetry: KuramotoTelemetry,
}

#[derive(Clone, Copy, Debug)]
struct GateStatus {
    env_open_ok: bool,
    env_level_ok: bool,
    mag_ok: bool,
    alpha_ok: bool,
    beta_ok: bool,
    env_level_ok_boot: bool,
    mag_ok_boot: bool,
    alpha_ok_boot: bool,
}

impl GateStatus {
    #[inline]
    fn normal_ok(&self) -> bool {
        self.env_open_ok && self.env_level_ok && self.mag_ok && self.alpha_ok && self.beta_ok
    }

    #[inline]
    fn bootstrap_ok(&self) -> bool {
        self.env_level_ok_boot && (self.mag_ok_boot || self.alpha_ok_boot)
    }
}

#[derive(Clone, Copy, Debug)]
struct AttackInfo {
    phase_err_at_attack: f32,
}

#[derive(Clone, Copy, Debug)]
struct ThetaView {
    phase: f32,
    freq_hz: f32,
    mag: f32,
    alpha: f32,
    beta: f32,
}

fn normalized_vitality(energy: f32, energy_cap: f32, vitality_exponent: f32) -> f32 {
    if !energy.is_finite() || !energy_cap.is_finite() || energy_cap <= 0.0 {
        return 0.0;
    }
    let energy_clamped = energy.clamp(0.0, energy_cap);
    let mut vitality = (energy_clamped / energy_cap).clamp(0.0, 1.0);
    let exponent = if vitality_exponent.is_finite() && vitality_exponent > 0.0 {
        vitality_exponent
    } else {
        1.0
    };
    vitality = vitality.powf(exponent);
    if vitality.is_finite() { vitality } else { 0.0 }
}

#[inline]
fn coupling_multiplier_from_mode(mode: RhythmCouplingMode, vitality: f32) -> f32 {
    match mode {
        RhythmCouplingMode::TemporalOnly => 1.0,
        RhythmCouplingMode::TemporalTimesVitality { lambda_v, v_floor } => {
            debug_assert!(lambda_v.is_finite() && lambda_v >= 0.0);
            debug_assert!(v_floor.is_finite() && (0.0..1.0).contains(&v_floor));
            let vitality = clamp01_finite(vitality);
            let denom = (1.0 - v_floor).max(1e-6);
            let g = ((vitality - v_floor) / denom).clamp(0.0, 1.0);
            (lambda_v * g).clamp(0.0, MAX_COUPLING_MULT)
        }
    }
}

#[inline]
fn attack_metric_value(metric: RhythmRewardMetric, phase_err_at_attack: f32) -> f32 {
    match metric {
        RhythmRewardMetric::AttackPhaseMatch => {
            clamp01_finite(0.5 + 0.5 * phase_err_at_attack.cos())
        }
    }
}

#[inline]
fn recharge_multiplier_from_reward(
    reward: Option<MetabolismRhythmReward>,
    attack_metric: Option<f32>,
) -> f32 {
    let Some(reward) = reward else {
        return 1.0;
    };
    debug_assert!(reward.rho_t.is_finite() && reward.rho_t >= 0.0);
    let t = clamp01_finite(attack_metric.unwrap_or(0.0));
    (1.0 + reward.rho_t * t).clamp(0.0, MAX_RECHARGE_MULT)
}

/// Compute effective Kuramoto coupling strength.
///
/// This is the same calculation used inside `KuramotoCore::update_phase`.
/// Extracted as a pure function so external simulation harnesses can
/// replicate the coupling model without instantiating a full KuramotoCore.
#[inline]
pub fn kuramoto_k_eff(
    omega_target: f32,
    global_coupling: f32,
    sensitivity_theta: f32,
    theta_mag: f32,
    theta_alpha: f32,
    env_gate: f32,
    env_amp: f32,
) -> f32 {
    let base_k = omega_target.max(20.0);
    base_k * global_coupling * sensitivity_theta * theta_mag * theta_alpha * env_gate * env_amp
}

/// Perform one Kuramoto phase integration step.
///
/// Returns the new (unwrapped) phase. Caller is responsible for wrapping
/// and any side-effects (energy, envelope, etc.).
///
/// This is the same calculation used inside `KuramotoCore::update_phase`.
#[inline]
pub fn kuramoto_phase_step(
    phase: f32,
    omega: f32,
    target_phase: f32,
    k_eff: f32,
    noise: f32,
    dt: f32,
) -> f32 {
    let diff = angle_diff_pm_pi(target_phase, wrap_0_tau(phase));
    let d_phi = omega + noise + k_eff * diff.sin();
    phase + d_phi * dt
}

impl KuramotoCore {
    #[inline]
    fn begin_attack(&mut self) {
        self.env_level = 0.0;
        self.state = ArticulationState::Attack;
        self.metrics.total_attacks = self.metrics.total_attacks.saturating_add(1);
    }

    pub fn plv(&self) -> Option<f32> {
        self.telemetry.sliding_plv.as_ref().map(|p| p.plv())
    }

    pub fn plv_is_full(&self) -> bool {
        self.telemetry
            .sliding_plv
            .as_ref()
            .is_some_and(|p| p.is_full())
    }

    pub fn enable_plv(&mut self, window: usize) {
        self.telemetry.sliding_plv = Some(SlidingPlv::new(window));
    }

    #[inline]
    fn apply_energy_delta(&mut self, delta: f32) {
        self.energy += delta;
        self.handle_energy_depletion();
    }

    #[inline]
    fn metabolism_policy(&self) -> MetabolismPolicy {
        MetabolismPolicy {
            basal_cost_per_sec: self.basal_cost,
            action_cost_per_attack: self.action_cost,
            recharge_per_attack: self.recharge_rate,
        }
    }

    #[inline]
    fn compute_gate_status(&self, rhythms: &NeuralRhythms, theta: &ThetaView) -> GateStatus {
        GateStatus {
            env_open_ok: rhythms.env_open > self.env_open_threshold,
            env_level_ok: rhythms.env_level > self.env_level_min,
            mag_ok: theta.mag > self.mag_threshold,
            alpha_ok: theta.alpha > self.alpha_threshold,
            beta_ok: theta.beta < self.beta_threshold,
            env_level_ok_boot: rhythms.env_level > (self.env_level_min * 0.5),
            mag_ok_boot: theta.mag > (self.mag_threshold * 0.5),
            alpha_ok_boot: theta.alpha > (self.alpha_threshold * 0.5),
        }
    }

    #[inline]
    fn update_phase(
        &mut self,
        theta: &ThetaView,
        rhythms: &NeuralRhythms,
        dt: f32,
        global_coupling: f32,
        bootstrap_active: bool,
    ) -> f32 {
        let omega_target = TAU * theta.freq_hz;
        let env_gate = rhythms.env_open;
        let env_amp = rhythms.env_level.sqrt();
        let k_time = kuramoto_k_eff(
            omega_target,
            global_coupling,
            self.sensitivity.theta,
            theta.mag,
            theta.alpha,
            env_gate,
            env_amp,
        );
        let k_time = if k_time.is_finite() {
            k_time.max(0.0)
        } else {
            0.0
        };
        let vitality = clamp01_finite(self.vitality_level);
        let coupling_mult = coupling_multiplier_from_mode(self.rhythm_coupling, vitality);
        let k_eff = k_time * coupling_mult;
        let k_eff = if k_eff.is_finite() {
            k_eff.max(0.0)
        } else {
            0.0
        };
        self.metrics.last_k_eff = k_eff;

        let mut pull = self.k_omega * theta.alpha * env_gate;
        if pull > 0.0 {
            if bootstrap_active {
                pull *= 4.0;
            }
            let blend = (pull * dt).min(1.0);
            self.omega_rad = self.omega_rad + (omega_target - self.omega_rad) * blend;
        }
        let min_omega = TAU * 3.0;
        let max_omega = TAU * 12.0;
        self.omega_rad = self.omega_rad.clamp(min_omega, max_omega);
        self.rhythm_freq = self.omega_rad / TAU;

        let sigma = self.base_sigma * (1.0 + self.beta_gain * theta.beta);
        let noise = self.noise_1f.sample() * sigma;

        let target = wrap_0_tau(theta.phase + self.phase_offset);
        self.rhythm_phase =
            kuramoto_phase_step(self.rhythm_phase, self.omega_rad, target, k_eff, noise, dt);

        k_eff
    }

    #[inline]
    fn maybe_trigger_attack(
        &mut self,
        gate: &GateStatus,
        theta: &ThetaView,
        bootstrap_active: bool,
    ) -> Option<AttackInfo> {
        let target_phase = wrap_0_tau(theta.phase + self.phase_offset);
        let agent_phase = wrap_0_tau(self.rhythm_phase);
        let phase_err_at_attack = angle_diff_pm_pi(target_phase, agent_phase);

        if !self.autonomous_attack {
            return None;
        }
        let mut attack = false;
        if self.state == ArticulationState::Idle && self.retrigger {
            if bootstrap_active {
                if gate.bootstrap_ok() {
                    attack = true;
                }
            } else if gate.normal_ok() {
                attack = true;
            }
        }

        if attack {
            self.begin_attack();
            if let Some(ref mut plv) = self.telemetry.sliding_plv {
                plv.push(phase_err_at_attack);
            }
        }

        if attack {
            Some(AttackInfo {
                phase_err_at_attack,
            })
        } else {
            None
        }
    }

    #[inline]
    fn update_envelope(&mut self, dt: f32) {
        step_attack_decay_envelope(
            &mut self.state,
            &mut self.env_level,
            self.attack_step,
            self.decay_rate,
            dt,
        );
    }

    fn apply_phonation_onset(&mut self, consonance: f32, strength: f32) {
        let strength = strength.clamp(0.0, 1.0);
        if strength <= 0.0 {
            return;
        }
        self.begin_attack();
        let delta = self
            .metabolism_policy()
            .attack_delta_with_recharge_multiplier(consonance, 1.0);
        self.apply_energy_delta(delta * strength);
    }

    #[inline]
    fn handle_energy_depletion(&mut self) {
        if !self.energy.is_finite() {
            self.energy = 0.0;
        }
        if self.energy <= 0.0 {
            // Energy depletion disables retrigger and lets the envelope decay to idle before death.
            self.energy = 0.0;
            self.retrigger = false;
            if self.state != ArticulationState::Idle {
                self.state = ArticulationState::Decay;
            }
        }
    }

    #[inline]
    fn update_vitality(&mut self) -> f32 {
        let computed = normalized_vitality(self.energy, self.energy_cap, self.vitality_exponent);
        if !self.energy.is_finite() {
            self.vitality_level = 0.0;
            return self.vitality_level;
        }
        if self.state == ArticulationState::Idle || self.energy > 0.0 {
            self.vitality_level = computed;
        }
        self.vitality_level
    }
}

impl ArticulationCore for KuramotoCore {
    fn process(
        &mut self,
        consonance: f32,
        rhythms: &NeuralRhythms,
        dt: f32,
        global_coupling: f32,
    ) -> ArticulationSignal {
        let policy = self.metabolism_policy();
        self.apply_energy_delta(policy.basal_delta(dt));

        let theta = ThetaView {
            phase: rhythms.theta.phase,
            freq_hz: rhythms.theta.freq_hz,
            mag: rhythms.theta.mag,
            alpha: rhythms.theta.alpha,
            beta: rhythms.theta.beta,
        };

        let gate = self.compute_gate_status(rhythms, &theta);

        self.bootstrap_timer = (self.bootstrap_timer - dt).max(0.0);
        let bootstrap_active = self.bootstrap_timer > 0.0;
        self.update_phase(&theta, rhythms, dt, global_coupling, bootstrap_active);
        let mut attacked_this_sample = false;
        let mut attack_metric_sum = 0.0f32;
        let mut attack_metric_count = 0u32;

        while self.rhythm_phase >= 2.0 * PI {
            self.rhythm_phase -= 2.0 * PI;
            let attack_info = self.maybe_trigger_attack(&gate, &theta, bootstrap_active);
            if let Some(info) = attack_info {
                attacked_this_sample = true;
                if let Some(reward) = self.rhythm_reward {
                    let metric_t = attack_metric_value(reward.metric, info.phase_err_at_attack);
                    attack_metric_sum += metric_t;
                    attack_metric_count += 1;
                }
            }
        }
        if attacked_this_sample {
            let attack_metric = if attack_metric_count > 0 {
                Some(attack_metric_sum / attack_metric_count as f32)
            } else {
                None
            };
            let recharge_multiplier =
                recharge_multiplier_from_reward(self.rhythm_reward, attack_metric);
            let delta =
                policy.attack_delta_with_recharge_multiplier(consonance, recharge_multiplier);
            self.apply_energy_delta(delta);
        }
        self.update_envelope(dt);

        let relaxation = rhythms.theta.alpha * self.sensitivity.alpha;
        let tension = rhythms.theta.beta * self.sensitivity.beta;
        let vitality = self.update_vitality();
        let is_active = self.env_level * vitality > 1e-6 && self.state != ArticulationState::Idle;
        ArticulationSignal {
            amplitude: self.env_level * vitality,
            is_active,
            relaxation,
            tension,
        }
    }

    fn is_alive(&self) -> bool {
        if self.energy <= 0.0 {
            return self.state != ArticulationState::Idle;
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

impl ArticulationCore for SequencedCore {
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
            relaxation: rhythms.theta.alpha,
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

impl ArticulationCore for DroneCore {
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
        let relax_boost = 1.0 + rhythms.theta.alpha * 0.5;
        let amplitude = (0.3 + 0.7 * lfo) * relax_boost;
        ArticulationSignal {
            amplitude: amplitude.clamp(0.0, 1.0),
            is_active: true,
            relaxation: rhythms.theta.alpha,
            tension: rhythms.theta.beta * 0.25,
        }
    }

    fn is_alive(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum AnyArticulationCore {
    Entrain(KuramotoCore),
    Seq(SequencedCore),
    Drone(DroneCore),
}

impl ArticulationCore for AnyArticulationCore {
    fn process(
        &mut self,
        consonance: f32,
        rhythms: &NeuralRhythms,
        dt: f32,
        global_coupling: f32,
    ) -> ArticulationSignal {
        match self {
            AnyArticulationCore::Entrain(c) => c.process(consonance, rhythms, dt, global_coupling),
            AnyArticulationCore::Seq(c) => c.process(consonance, rhythms, dt, global_coupling),
            AnyArticulationCore::Drone(c) => c.process(consonance, rhythms, dt, global_coupling),
        }
    }

    fn is_alive(&self) -> bool {
        match self {
            AnyArticulationCore::Entrain(c) => c.is_alive(),
            AnyArticulationCore::Seq(c) => c.is_alive(),
            AnyArticulationCore::Drone(c) => c.is_alive(),
        }
    }
}

impl AnyArticulationCore {
    pub fn from_config<R: Rng + ?Sized>(
        config: &ArticulationCoreConfig,
        _fs: f32,
        noise_seed: u64,
        rng: &mut R,
    ) -> Self {
        match config {
            ArticulationCoreConfig::Entrain {
                lifecycle,
                rhythm_freq,
                rhythm_sensitivity,
                rhythm_coupling,
                rhythm_reward,
                ..
            } => {
                let derived = envelope_from_lifecycle(lifecycle);
                let energy = derived.initial_energy;
                let basal_cost = derived.policy.basal_cost_per_sec;
                let recharge_rate = derived.policy.recharge_per_attack;
                let action_cost = derived.policy.action_cost_per_attack;
                let attack_step = derived.attack_step;
                let decay_rate = derived.decay_rate;
                let state = derived.state;
                let sensitivity = derived.sensitivity;
                let retrigger = derived.retrigger;
                let energy_cap = energy.max(0.0);
                let vitality_exponent = 0.5;
                let vitality_level = normalized_vitality(energy, energy_cap, vitality_exponent);
                let rhythm_coupling = rhythm_coupling.sanitized();
                let rhythm_reward = rhythm_reward.map(MetabolismRhythmReward::sanitized);
                let env_level = if matches!(state, ArticulationState::Attack) {
                    attack_step
                } else {
                    0.0
                };
                let init_freq = rhythm_freq.unwrap_or_else(|| rng.random_range(5.0..7.0));
                // Phase/offset seed diversity; theta lock uses base_k ~ omega_target in process.
                AnyArticulationCore::Entrain(KuramotoCore {
                    energy,
                    energy_cap,
                    vitality_exponent,
                    vitality_level,
                    basal_cost,
                    action_cost,
                    recharge_rate,
                    sensitivity: Sensitivity {
                        // rhythm_sensitivity targets theta coupling strength.
                        theta: rhythm_sensitivity.unwrap_or(sensitivity.theta),
                        ..sensitivity
                    },
                    rhythm_coupling,
                    rhythm_reward,
                    rhythm_phase: rng.random_range(0.0..std::f32::consts::TAU),
                    rhythm_freq: init_freq,
                    omega_rad: TAU * init_freq,
                    phase_offset: rng.random_range(-std::f32::consts::PI..std::f32::consts::PI),
                    env_level,
                    state,
                    attack_step,
                    decay_rate,
                    retrigger,
                    noise_1f: PinkNoise::new(noise_seed, 0.001),
                    base_sigma: 0.3, // rad/s noise floor
                    beta_gain: 1.0,  // beta -> noise gain
                    k_omega: 3.0,    // omega pull toward theta
                    bootstrap_timer: 1.5,
                    env_open_threshold: 0.55,
                    env_level_min: 0.02,
                    mag_threshold: 0.04,
                    alpha_threshold: 0.2,
                    beta_threshold: 0.9,
                    autonomous_attack: true,
                    metrics: KuramotoMetrics::default(),
                    telemetry: KuramotoTelemetry::default(),
                })
            }
            ArticulationCoreConfig::Seq { duration, .. } => {
                AnyArticulationCore::Seq(SequencedCore {
                    timer: 0.0,
                    duration: duration.max(0.0),
                    env_level: 0.0,
                })
            }
            ArticulationCoreConfig::Drone { sway, .. } => {
                let sway_rate = sway.unwrap_or(0.05);
                let sway_rate = if sway_rate <= 0.0 { 0.05 } else { sway_rate };
                AnyArticulationCore::Drone(DroneCore {
                    phase: rng.random_range(0.0..std::f32::consts::TAU),
                    sway_rate,
                })
            }
        }
    }

    pub fn apply_phonation_onset(&mut self, consonance: f32, strength: f32) {
        let strength = strength.clamp(0.0, 1.0);
        if strength <= 0.0 {
            return;
        }
        match self {
            AnyArticulationCore::Entrain(core) => core.apply_phonation_onset(consonance, strength),
            AnyArticulationCore::Seq(core) => core.apply_phonation_onset(strength),
            AnyArticulationCore::Drone(core) => core.apply_phonation_onset(),
        }
    }

    pub fn render_modulator_spec(&self, phonation_mode: PhonationMode) -> RenderModulatorSpec {
        match self {
            AnyArticulationCore::Entrain(core) => {
                let autonomous_pulse =
                    matches!(phonation_mode, PhonationMode::Hold).then_some(AutonomousPulseSpec {
                        rate_hz: core.rhythm_freq.max(0.01),
                        phase_0_1: (core.rhythm_phase / TAU).rem_euclid(1.0),
                        retrigger: core.retrigger,
                        env_open_threshold: core.env_open_threshold,
                        mag_threshold: core.mag_threshold,
                        alpha_threshold: core.alpha_threshold,
                    });
                RenderModulatorSpec::EntrainPulse {
                    attack_step: core.attack_step,
                    decay_rate: core.decay_rate,
                    sustain_level: 0.0,
                    initial_state: RenderModulatorStateKind::from(core.state),
                    initial_env_level: core.env_level.clamp(0.0, 1.0),
                    alpha_gain: core.sensitivity.alpha,
                    beta_gain: core.sensitivity.beta,
                    autonomous_pulse,
                }
            }
            AnyArticulationCore::Seq(core) => RenderModulatorSpec::SeqGate {
                duration_sec: core.duration.max(0.0),
            },
            AnyArticulationCore::Drone(core) => RenderModulatorSpec::DroneSway {
                phase: core.phase,
                sway_rate: core.sway_rate.max(0.01),
            },
        }
    }
}

impl SequencedCore {
    fn apply_phonation_onset(&mut self, _strength: f32) {
        self.timer = 0.0;
    }
}

impl DroneCore {
    fn apply_phonation_onset(&mut self) {}
}

struct LifecycleDerived {
    initial_energy: f32,
    attack_step: f32,
    decay_rate: f32,
    state: ArticulationState,
    sensitivity: Sensitivity,
    retrigger: bool,
    policy: MetabolismPolicy,
}

fn envelope_from_lifecycle(lifecycle: &LifecycleConfig) -> LifecycleDerived {
    let policy = MetabolismPolicy::from_lifecycle(lifecycle);
    match lifecycle {
        LifecycleConfig::Decay {
            initial_energy,
            half_life_sec,
            attack_sec,
        } => {
            let atk = attack_sec.max(0.0005);
            let attack_step = 1.0 / atk;
            let decay_sec = half_life_sec.max(0.01);
            let decay_rate = std::f32::consts::LN_2 / decay_sec;
            LifecycleDerived {
                initial_energy: *initial_energy,
                attack_step,
                decay_rate,
                state: ArticulationState::Attack,
                sensitivity: Sensitivity::default(),
                retrigger: false,
                policy,
            }
        }
        LifecycleConfig::Sustain {
            initial_energy,
            envelope,
            ..
        } => {
            let atk = envelope.attack_sec.max(0.0005);
            let attack_step = 1.0 / atk;
            let decay_sec = envelope.decay_sec.max(0.01);
            let decay_rate = 1.0 / decay_sec;
            LifecycleDerived {
                initial_energy: *initial_energy,
                attack_step,
                decay_rate,
                state: ArticulationState::Idle,
                sensitivity: Sensitivity {
                    delta: 1.0,
                    theta: 1.0,
                    alpha: 0.5,
                    beta: 0.5,
                },
                retrigger: true,
                policy,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::modulation::NeuralRhythms;
    use std::f32::consts::{PI, TAU};

    fn test_core(energy: f32) -> KuramotoCore {
        let energy_cap = 1.0;
        let vitality_exponent = 0.5;
        let vitality_level = normalized_vitality(energy, energy_cap, vitality_exponent);
        KuramotoCore {
            energy,
            energy_cap,
            vitality_exponent,
            vitality_level,
            basal_cost: 0.0,
            action_cost: 0.0,
            recharge_rate: 0.0,
            sensitivity: Sensitivity::default(),
            rhythm_coupling: RhythmCouplingMode::TemporalOnly,
            rhythm_reward: None,
            rhythm_phase: 0.0,
            rhythm_freq: 5.0,
            omega_rad: TAU * 5.0,
            phase_offset: 0.0,
            env_level: 1.0,
            state: ArticulationState::Decay,
            attack_step: 0.1,
            decay_rate: 1.0,
            retrigger: true,
            noise_1f: PinkNoise::new(1, 0.0),
            base_sigma: 0.0,
            beta_gain: 0.0,
            k_omega: 0.0,
            bootstrap_timer: 0.0,
            env_open_threshold: 0.0,
            env_level_min: 0.0,
            mag_threshold: 0.0,
            alpha_threshold: 0.0,
            beta_threshold: 1.0,
            autonomous_attack: true,
            metrics: KuramotoMetrics::default(),
            telemetry: KuramotoTelemetry::default(),
        }
    }

    #[test]
    fn test_kuramoto_k_eff_zero_coupling() {
        let k = kuramoto_k_eff(2.0 * PI * 6.0, 0.0, 0.8, 0.9, 0.7, 1.0, 0.6);
        assert!(k.abs() <= 1e-6);
    }

    #[test]
    fn test_kuramoto_k_eff_scales_linearly() {
        let base = kuramoto_k_eff(2.0 * PI * 6.0, 0.2, 0.5, 0.9, 0.7, 1.0, 0.6);
        let double_coupling = kuramoto_k_eff(2.0 * PI * 6.0, 0.4, 0.5, 0.9, 0.7, 1.0, 0.6);
        let double_sensitivity = kuramoto_k_eff(2.0 * PI * 6.0, 0.2, 1.0, 0.9, 0.7, 1.0, 0.6);
        assert!((double_coupling - 2.0 * base).abs() <= 1e-6);
        assert!((double_sensitivity - 2.0 * base).abs() <= 1e-6);
    }

    #[test]
    fn test_kuramoto_phase_step_no_coupling() {
        let phase = 1.0;
        let omega = 3.0;
        let dt = 0.1;
        let next = kuramoto_phase_step(phase, omega, 2.0, 0.0, 0.0, dt);
        assert!((next - (phase + omega * dt)).abs() <= 1e-6);
    }

    #[test]
    fn test_kuramoto_phase_step_attracts() {
        let phase = 0.0;
        let target = PI * 0.5;
        let next = kuramoto_phase_step(phase, 0.0, target, 2.0, 0.0, 0.05);
        assert!(next > phase, "phase should move toward target when coupled");
    }

    #[test]
    fn test_refactored_update_phase_unchanged() {
        let mut core = test_core(1.0);
        let mut manual = test_core(1.0);

        core.sensitivity.theta = 0.7;
        manual.sensitivity.theta = 0.7;
        core.k_omega = 0.8;
        manual.k_omega = 0.8;
        core.base_sigma = 0.2;
        manual.base_sigma = 0.2;
        core.beta_gain = 0.5;
        manual.beta_gain = 0.5;
        core.phase_offset = 0.33;
        manual.phase_offset = 0.33;
        core.rhythm_phase = 1.0;
        manual.rhythm_phase = 1.0;
        core.omega_rad = TAU * 5.2;
        manual.omega_rad = TAU * 5.2;
        core.rhythm_freq = 5.2;
        manual.rhythm_freq = 5.2;
        core.noise_1f = PinkNoise::new(42, 1.0);
        manual.noise_1f = PinkNoise::new(42, 1.0);
        core.state = ArticulationState::Decay;
        manual.state = ArticulationState::Decay;
        core.decay_rate = 0.0;
        manual.decay_rate = 0.0;
        core.retrigger = false;
        manual.retrigger = false;
        core.bootstrap_timer = 0.0;
        manual.bootstrap_timer = 0.0;

        let dt = 0.01;
        let global_coupling = 0.6;
        let mut rhythms = NeuralRhythms::default();
        rhythms.theta.phase = 0.2;
        rhythms.theta.freq_hz = 6.8;
        rhythms.theta.mag = 0.9;
        rhythms.theta.alpha = 0.8;
        rhythms.theta.beta = 0.2;
        rhythms.env_open = 0.95;
        rhythms.env_level = 0.81;

        for _ in 0..100 {
            core.process(0.0, &rhythms, dt, global_coupling);

            let omega_target = TAU * rhythms.theta.freq_hz;
            let k_eff = kuramoto_k_eff(
                omega_target,
                global_coupling,
                manual.sensitivity.theta,
                rhythms.theta.mag,
                rhythms.theta.alpha,
                rhythms.env_open,
                rhythms.env_level.sqrt(),
            );
            let pull = manual.k_omega * rhythms.theta.alpha * rhythms.env_open;
            if pull > 0.0 {
                let blend = (pull * dt).min(1.0);
                manual.omega_rad = manual.omega_rad + (omega_target - manual.omega_rad) * blend;
            }
            manual.omega_rad = manual.omega_rad.clamp(TAU * 3.0, TAU * 12.0);
            manual.rhythm_freq = manual.omega_rad / TAU;

            let sigma = manual.base_sigma * (1.0 + manual.beta_gain * rhythms.theta.beta);
            let noise = manual.noise_1f.sample() * sigma;
            let target = wrap_0_tau(rhythms.theta.phase + manual.phase_offset);
            manual.rhythm_phase = kuramoto_phase_step(
                manual.rhythm_phase,
                manual.omega_rad,
                target,
                k_eff,
                noise,
                dt,
            );
            while manual.rhythm_phase >= 2.0 * PI {
                manual.rhythm_phase -= 2.0 * PI;
            }

            assert!((core.metrics.last_k_eff - k_eff).abs() <= 1e-6);
            assert!((core.omega_rad - manual.omega_rad).abs() <= 1e-6);
            assert!((core.rhythm_freq - manual.rhythm_freq).abs() <= 1e-6);
            assert!((core.rhythm_phase - manual.rhythm_phase).abs() <= 1e-6);
        }
    }

    #[test]
    fn vitality_coupling_increases_k_eff_when_opted_in() {
        let mut low = test_core(0.16);
        let mut high = test_core(1.0);

        low.sensitivity.theta = 0.8;
        high.sensitivity.theta = 0.8;
        low.rhythm_coupling = RhythmCouplingMode::TemporalTimesVitality {
            lambda_v: 1.5,
            v_floor: 0.2,
        };
        high.rhythm_coupling = low.rhythm_coupling;

        let theta = ThetaView {
            phase: 0.2,
            freq_hz: 6.0,
            mag: 0.9,
            alpha: 0.8,
            beta: 0.2,
        };
        let rhythms = NeuralRhythms {
            theta: crate::core::modulation::RhythmBand {
                phase: theta.phase,
                freq_hz: theta.freq_hz,
                mag: theta.mag,
                alpha: theta.alpha,
                beta: theta.beta,
            },
            delta: crate::core::modulation::RhythmBand::default(),
            env_level: 0.81,
            env_open: 0.95,
        };

        let dt = 0.01;
        let global_coupling = 0.7;
        low.update_phase(&theta, &rhythms, dt, global_coupling, false);
        high.update_phase(&theta, &rhythms, dt, global_coupling, false);

        assert!(
            high.metrics.last_k_eff > low.metrics.last_k_eff,
            "higher vitality should increase effective coupling in TemporalTimesVitality mode"
        );
        assert!(low.metrics.last_k_eff > 0.0);
    }

    #[test]
    fn vitality_coupling_multiplier_is_capped() {
        let mut core = test_core(1.0);
        core.sensitivity.theta = 0.8;
        core.rhythm_coupling = RhythmCouplingMode::TemporalTimesVitality {
            lambda_v: 1_000.0,
            v_floor: 0.0,
        };

        let theta = ThetaView {
            phase: 0.2,
            freq_hz: 6.0,
            mag: 0.9,
            alpha: 0.8,
            beta: 0.2,
        };
        let rhythms = NeuralRhythms {
            theta: crate::core::modulation::RhythmBand {
                phase: theta.phase,
                freq_hz: theta.freq_hz,
                mag: theta.mag,
                alpha: theta.alpha,
                beta: theta.beta,
            },
            delta: crate::core::modulation::RhythmBand::default(),
            env_level: 0.81,
            env_open: 0.95,
        };

        let dt = 0.01;
        let global_coupling = 0.7;
        core.update_phase(&theta, &rhythms, dt, global_coupling, false);
        assert!(core.metrics.last_k_eff.is_finite());

        let k_time = kuramoto_k_eff(
            TAU * theta.freq_hz,
            global_coupling,
            core.sensitivity.theta,
            theta.mag,
            theta.alpha,
            rhythms.env_open,
            rhythms.env_level.sqrt(),
        );
        assert!(
            core.metrics.last_k_eff <= MAX_COUPLING_MULT * k_time + 1e-6,
            "effective coupling should be capped at {}x temporal coupling",
            MAX_COUPLING_MULT
        );
    }

    #[test]
    fn vitality_mode_cannot_create_coupling_when_env_gate_closed() {
        let mut core = test_core(1.0);
        core.sensitivity.theta = 1.0;
        core.rhythm_coupling = RhythmCouplingMode::TemporalTimesVitality {
            lambda_v: 2.0,
            v_floor: 0.0,
        };
        core.vitality_level = 1.0;

        let theta = ThetaView {
            phase: 0.1,
            freq_hz: 6.0,
            mag: 1.0,
            alpha: 1.0,
            beta: 0.2,
        };
        let rhythms = NeuralRhythms {
            theta: crate::core::modulation::RhythmBand {
                phase: theta.phase,
                freq_hz: theta.freq_hz,
                mag: theta.mag,
                alpha: theta.alpha,
                beta: theta.beta,
            },
            delta: crate::core::modulation::RhythmBand::default(),
            env_level: 1.0,
            env_open: 0.0,
        };

        core.update_phase(&theta, &rhythms, 0.01, 0.7, false);
        assert_eq!(core.metrics.last_k_eff, 0.0);
    }

    #[test]
    fn update_phase_keff_is_finite_under_nonfinite_inputs() {
        let mut core = test_core(1.0);
        core.rhythm_coupling = RhythmCouplingMode::TemporalTimesVitality {
            lambda_v: 2.0,
            v_floor: 0.0,
        };

        let theta = ThetaView {
            phase: 0.1,
            freq_hz: 6.0,
            mag: 1.0,
            alpha: 1.0,
            beta: 0.2,
        };
        let rhythms = NeuralRhythms {
            theta: crate::core::modulation::RhythmBand {
                phase: theta.phase,
                freq_hz: theta.freq_hz,
                mag: theta.mag,
                alpha: theta.alpha,
                beta: theta.beta,
            },
            delta: crate::core::modulation::RhythmBand::default(),
            env_level: -1.0,
            env_open: 1.0,
        };

        core.update_phase(&theta, &rhythms, 0.01, 0.7, false);
        assert!(core.metrics.last_k_eff.is_finite());
        assert_eq!(core.metrics.last_k_eff, 0.0);
    }

    #[test]
    fn attack_metric_nan_is_sanitized() {
        let t = attack_metric_value(RhythmRewardMetric::AttackPhaseMatch, f32::NAN);
        assert_eq!(t, 0.0);
    }

    #[test]
    fn rhythm_reward_increases_energy_delta_when_attack_occurs() {
        let mut base = test_core(1.0);
        let mut rewarded = test_core(1.0);

        for core in [&mut base, &mut rewarded] {
            core.state = ArticulationState::Idle;
            core.retrigger = true;
            core.rhythm_phase = 2.0 * PI + 0.1;
            core.phase_offset = 0.0;
            core.basal_cost = 0.0;
            core.action_cost = 0.2;
            core.recharge_rate = 0.5;
            core.rhythm_coupling = RhythmCouplingMode::TemporalOnly;
            core.bootstrap_timer = 0.0;
            core.env_open_threshold = 0.0;
            core.env_level_min = 0.0;
            core.mag_threshold = 0.0;
            core.alpha_threshold = 0.0;
            core.beta_threshold = 1.0;
        }
        rewarded.rhythm_reward = Some(MetabolismRhythmReward {
            rho_t: 1.0,
            metric: RhythmRewardMetric::AttackPhaseMatch,
        });

        let mut rhythms = NeuralRhythms::default();
        rhythms.theta.phase = 0.1;
        rhythms.theta.freq_hz = 6.0;
        rhythms.theta.mag = 1.0;
        rhythms.theta.alpha = 1.0;
        rhythms.theta.beta = 0.2;
        rhythms.env_open = 1.0;
        rhythms.env_level = 1.0;

        let consonance = 1.0;
        base.process(consonance, &rhythms, 0.0, 1.0);
        rewarded.process(consonance, &rhythms, 0.0, 1.0);

        assert_eq!(base.metrics.total_attacks, 1);
        assert_eq!(rewarded.metrics.total_attacks, 1);
        assert!(rewarded.energy > base.energy);
        let expected_extra = 0.5;
        assert!(
            ((rewarded.energy - base.energy) - expected_extra).abs() < 1e-5,
            "rhythm reward should increase only recharge contribution"
        );
    }

    #[test]
    fn apply_phonation_onset_entrain_updates_attack_state_and_metrics() {
        let mut core = test_core(0.6);
        core.state = ArticulationState::Idle;
        core.env_level = 0.4;
        core.metrics.total_attacks = 0;
        core.action_cost = 0.2;
        core.recharge_rate = 0.3;

        core.apply_phonation_onset(0.8, 1.0);

        assert_eq!(core.state, ArticulationState::Attack);
        assert_eq!(core.env_level, 0.0);
        assert_eq!(core.metrics.total_attacks, 1);
        assert!(core.energy > 0.6 - 0.2);
    }

    #[test]
    fn render_modulator_spec_entrain_hold_has_autonomous_pulse() {
        let mut inner = test_core(0.6);
        inner.env_open_threshold = 0.3;
        inner.mag_threshold = 0.4;
        inner.alpha_threshold = 0.5;
        let core = AnyArticulationCore::Entrain(inner);
        let spec = core.render_modulator_spec(PhonationMode::Hold);
        let RenderModulatorSpec::EntrainPulse {
            autonomous_pulse, ..
        } = spec
        else {
            panic!("expected entrain pulse");
        };
        let pulse = autonomous_pulse.expect("expected hold autonomous pulse");
        assert_eq!(pulse.env_open_threshold, 0.3);
        assert_eq!(pulse.mag_threshold, 0.4);
        assert_eq!(pulse.alpha_threshold, 0.5);
    }

    #[test]
    fn energy_depletion_enters_decay_without_instant_silence() {
        let mut core = test_core(0.01);
        core.basal_cost = 2.0;
        core.retrigger = true;
        core.env_level = 1.0;
        core.state = ArticulationState::Decay;

        let rhythms = NeuralRhythms::default();
        let signal = core.process(0.0, &rhythms, 0.01, 0.0);

        assert!(core.energy <= 0.0);
        assert_eq!(core.state, ArticulationState::Decay);
        assert!(!core.retrigger);
        assert!(
            signal.amplitude > 0.0,
            "amplitude should release, not drop to zero immediately"
        );
    }

    #[test]
    fn attack_depletion_enters_decay_without_instant_silence() {
        let mut core = test_core(0.01);
        core.basal_cost = 2.0;
        core.retrigger = true;
        core.env_level = 0.4;
        core.state = ArticulationState::Attack;

        let rhythms = NeuralRhythms::default();
        let signal = core.process(0.0, &rhythms, 0.01, 0.0);

        assert!(core.energy <= 0.0);
        assert_eq!(core.state, ArticulationState::Decay);
        assert!(!core.retrigger);
        assert!(
            signal.amplitude > 0.0,
            "attack depletion should release, not drop to zero immediately"
        );
    }

    #[test]
    fn energy_zero_releases_to_idle() {
        let mut core = test_core(0.0);
        core.energy = 0.0;
        core.env_level = 1.0;
        core.state = ArticulationState::Decay;
        core.decay_rate = 5.0;

        let rhythms = NeuralRhythms::default();
        for _ in 0..240 {
            core.process(0.0, &rhythms, 0.01, 0.0);
        }

        assert_eq!(core.state, ArticulationState::Idle);
        assert_eq!(core.env_level, 0.0);
        assert!(!core.is_alive());
    }

    #[test]
    fn vitality_resets_after_idle() {
        let mut core = test_core(0.0);
        core.energy = 0.0;
        core.env_level = 1.0;
        core.state = ArticulationState::Decay;
        core.decay_rate = 6.0;
        core.vitality_level = 0.5;

        let rhythms = NeuralRhythms::default();
        for _ in 0..300 {
            core.process(0.0, &rhythms, 0.01, 0.0);
        }
        assert_eq!(core.state, ArticulationState::Idle);

        let signal = core.process(0.0, &rhythms, 0.01, 0.0);
        assert!(core.vitality_level <= 1e-6);
        assert!(!signal.is_active);
    }

    #[test]
    fn is_active_tracks_amplitude() {
        let mut core = test_core(0.0);
        core.energy = 0.0;
        core.vitality_level = 0.0;
        core.env_level = 0.5;
        core.state = ArticulationState::Decay;
        core.decay_rate = 0.0;

        let rhythms = NeuralRhythms::default();
        let signal = core.process(0.0, &rhythms, 0.01, 0.0);

        assert!(signal.amplitude <= 1e-6);
        assert!(!signal.is_active);
    }

    #[test]
    fn vitality_scales_amplitude() {
        let rhythms = NeuralRhythms::default();
        let mut high = test_core(1.0);
        high.decay_rate = 0.0;

        let mut low = test_core(0.25);
        low.decay_rate = 0.0;

        let high_signal = high.process(0.0, &rhythms, 0.01, 0.0);
        let low_signal = low.process(0.0, &rhythms, 0.01, 0.0);

        assert!(
            low_signal.amplitude < high_signal.amplitude,
            "lower energy should yield lower amplitude via vitality"
        );
    }
}
