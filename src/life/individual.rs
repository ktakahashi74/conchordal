use crate::core::landscape::Landscape;
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::core::phase::{angle_diff_pm_pi, wrap_0_tau};
use crate::core::utils::pink_noise_tick;
use crate::life::lifecycle::LifecycleConfig;
use crate::life::perceptual::{FeaturesNow, PerceptualContext};
use crate::life::scenario::{
    ArticulationCoreConfig, HarmonicMode, PitchCoreConfig, SoundBodyConfig, TimbreGenotype,
};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::f32::consts::{PI, TAU};
use tracing::debug;

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
    fn update(
        &mut self,
        planned: &PlannedPitch,
        rhythms: &NeuralRhythms,
        dt: f32,
    ) -> bool {
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
    pub fn new(core: AnyArticulationCore, gate: f32) -> Self {
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
        self.core
            .process(consonance, rhythms, dt, global_coupling)
    }

    pub fn is_alive(&self) -> bool {
        self.core.is_alive()
    }

    pub fn update_gate(&mut self, planned: &PlannedPitch, rhythms: &NeuralRhythms, dt: f32) -> bool {
        self.planned_gate.update(planned, rhythms, dt)
    }

    pub fn gate(&self) -> f32 {
        self.planned_gate.gate
    }

    pub fn set_gate(&mut self, gate: f32) {
        self.planned_gate.gate = gate.clamp(0.0, 1.0);
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
    pub omega_rad: f32,
    pub phase_offset: f32,
    pub debug_id: u64,
    pub env_level: f32,
    pub state: ArticulationState,
    pub attack_step: f32,
    pub decay_factor: f32,
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
    pub dbg_accum_time: f32,
    pub dbg_wraps: u32,
    pub dbg_attacks: u32,
    pub dbg_boot_attacks: u32,
    pub dbg_attack_logs_left: u32,
    pub dbg_attack_count_normal: u32,
    pub dbg_attack_sum_abs_diff: f32,
    pub dbg_attack_sum_cos: f32,
    pub dbg_attack_sum_sin: f32,
    pub dbg_fail_env: u32,
    pub dbg_fail_env_level: u32,
    pub dbg_fail_mag: u32,
    pub dbg_fail_alpha: u32,
    pub dbg_fail_beta: u32,
    pub dbg_last_env_open: f32,
    pub dbg_last_env_level: f32,
    pub dbg_last_theta_mag: f32,
    pub dbg_last_theta_alpha: f32,
    pub dbg_last_theta_beta: f32,
    pub dbg_last_k_eff: f32,
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
struct PhaseStep {
    k_eff: f32,
}

#[derive(Clone, Copy, Debug)]
struct ThetaView {
    phase: f32,
    freq_hz: f32,
    mag: f32,
    alpha: f32,
    beta: f32,
}

impl KuramotoCore {
    #[inline]
    fn compute_gate_status(&mut self, rhythms: &NeuralRhythms, theta: &ThetaView) -> GateStatus {
        self.dbg_last_env_open = rhythms.env_open;
        self.dbg_last_env_level = rhythms.env_level;
        self.dbg_last_theta_mag = theta.mag;
        self.dbg_last_theta_alpha = theta.alpha;
        self.dbg_last_theta_beta = theta.beta;

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
    fn record_gate_failures(&mut self, gate: &GateStatus) {
        if !gate.env_open_ok {
            self.dbg_fail_env += 1;
        }
        if !gate.env_level_ok {
            self.dbg_fail_env_level += 1;
        }
        if !gate.mag_ok {
            self.dbg_fail_mag += 1;
        }
        if !gate.alpha_ok {
            self.dbg_fail_alpha += 1;
        }
        if !gate.beta_ok {
            self.dbg_fail_beta += 1;
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
    ) -> PhaseStep {
        let omega_target = TAU * theta.freq_hz;
        // base_k scales with target omega so coupling stays in rad/s units.
        let base_k = omega_target.max(20.0);
        let env_gate = rhythms.env_open;
        let env_amp = rhythms.env_level.sqrt();
        let k_eff = base_k
            * global_coupling
            * self.sensitivity.theta
            * theta.mag
            * theta.alpha
            * env_gate
            * env_amp;
        self.dbg_last_k_eff = k_eff;

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
        let diff = angle_diff_pm_pi(target, wrap_0_tau(self.rhythm_phase));
        let d_phi = self.omega_rad + noise + k_eff * diff.sin();
        self.rhythm_phase += d_phi * dt;

        PhaseStep { k_eff }
    }

    #[inline]
    fn maybe_trigger_attack(
        &mut self,
        gate: &GateStatus,
        rhythms: &NeuralRhythms,
        theta: &ThetaView,
        consonance: f32,
        bootstrap_active: bool,
        k_eff: f32,
    ) -> bool {
        self.record_gate_failures(gate);

        let target_phase = wrap_0_tau(theta.phase + self.phase_offset);
        let agent_phase = wrap_0_tau(self.rhythm_phase);
        let phase_err_at_attack = angle_diff_pm_pi(target_phase, agent_phase);

        let mut attack = false;
        let mut boot_attack = false;
        if self.state == ArticulationState::Idle && self.retrigger {
            if bootstrap_active {
                if gate.bootstrap_ok() {
                    attack = true;
                    boot_attack = true;
                }
            } else if gate.normal_ok() {
                attack = true;
            }
        }

        if attack {
            self.env_level = 0.0;
            self.state = ArticulationState::Attack;
            self.energy -= self.action_cost;
            if consonance > 0.5 {
                self.energy += self.recharge_rate * consonance;
            }
            self.dbg_attacks += 1;
            if boot_attack {
                self.dbg_boot_attacks += 1;
            }
            if !boot_attack {
                self.dbg_attack_count_normal += 1;
                self.dbg_attack_sum_abs_diff += phase_err_at_attack.abs();
                self.dbg_attack_sum_cos += phase_err_at_attack.cos();
                self.dbg_attack_sum_sin += phase_err_at_attack.sin();
            }
            if self.dbg_attack_logs_left > 0 {
                debug!(
                    target: "rhythm::attack",
                    id = self.debug_id,
                    mode = if boot_attack { "bootstrap" } else { "normal" },
                    env_open = rhythms.env_open,
                    env_level = rhythms.env_level,
                    env_open_threshold = self.env_open_threshold,
                    env_level_min = self.env_level_min,
                    mag_threshold = self.mag_threshold,
                    alpha_threshold = self.alpha_threshold,
                    beta_threshold = self.beta_threshold,
                    theta_mag = theta.mag,
                    theta_alpha = theta.alpha,
                    theta_beta = theta.beta,
                    k_eff,
                    target_phase,
                    agent_phase,
                    phase_err_at_attack,
                    phase_offset = self.phase_offset
                );
                self.dbg_attack_logs_left -= 1;
            }
        }

        attack
    }

    #[inline]
    fn update_envelope(&mut self) {
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
        self.energy -= self.basal_cost * dt;
        if self.energy <= 0.0 {
            self.state = ArticulationState::Idle;
            self.env_level = 0.0;
            return ArticulationSignal::default();
        }

        let theta = ThetaView {
            phase: rhythms.theta.phase,
            freq_hz: rhythms.theta.freq_hz,
            mag: rhythms.theta.mag,
            alpha: rhythms.theta.alpha,
            beta: rhythms.theta.beta,
        };

        let gate = self.compute_gate_status(rhythms, &theta);
        self.dbg_accum_time += dt;

        self.bootstrap_timer = (self.bootstrap_timer - dt).max(0.0);
        let bootstrap_active = self.bootstrap_timer > 0.0;
        let step = self.update_phase(&theta, rhythms, dt, global_coupling, bootstrap_active);
        let mut attacked_this_sample = false;

        while self.rhythm_phase >= 2.0 * PI {
            self.rhythm_phase -= 2.0 * PI;
            self.dbg_wraps += 1;
            let attacked = self.maybe_trigger_attack(
                &gate,
                rhythms,
                &theta,
                consonance,
                bootstrap_active,
                step.k_eff,
            );
            attacked_this_sample = attacked_this_sample || attacked;
        }
        self.update_envelope();

        if self.dbg_accum_time >= 1.0 {
            let agent_freq_hz = self.omega_rad / TAU;
            let freq_err_hz = agent_freq_hz - rhythms.theta.freq_hz;
            let (attack_plv, attack_mean_abs_diff) = if self.dbg_attack_count_normal > 0 {
                let count = self.dbg_attack_count_normal as f32;
                let plv = (self.dbg_attack_sum_cos * self.dbg_attack_sum_cos
                    + self.dbg_attack_sum_sin * self.dbg_attack_sum_sin)
                    .sqrt()
                    / count;
                let mean_abs_diff = self.dbg_attack_sum_abs_diff / count;
                (plv, mean_abs_diff)
            } else {
                (0.0, 0.0)
            };
            debug!(
                target: "rhythm::metrics",
                id = self.debug_id,
                attack_count_normal = self.dbg_attack_count_normal,
                attack_mean_abs_diff = attack_mean_abs_diff,
                attack_plv_target = attack_plv,
                last_env_open = self.dbg_last_env_open,
                last_env_level = self.dbg_last_env_level,
                last_theta_mag = self.dbg_last_theta_mag,
                last_theta_alpha = self.dbg_last_theta_alpha,
                last_theta_beta = self.dbg_last_theta_beta,
                last_k_eff = self.dbg_last_k_eff,
                fail_env_open = self.dbg_fail_env,
                fail_env_level = self.dbg_fail_env_level,
                fail_mag = self.dbg_fail_mag,
                fail_alpha = self.dbg_fail_alpha,
                fail_beta = self.dbg_fail_beta
            );
            debug!(
                target: "rhythm::agent",
                id = self.debug_id,
                agent_freq_hz,
                freq_err_hz,
                wraps = self.dbg_wraps,
                attacks = self.dbg_attacks,
                boot_attacks = self.dbg_boot_attacks,
                fail_env = self.dbg_fail_env,
                fail_mag = self.dbg_fail_mag,
                fail_alpha = self.dbg_fail_alpha,
                fail_beta = self.dbg_fail_beta,
                last_env_open = self.dbg_last_env_open,
                last_env_level = self.dbg_last_env_level,
                last_theta_mag = self.dbg_last_theta_mag,
                last_theta_alpha = self.dbg_last_theta_alpha,
                last_theta_beta = self.dbg_last_theta_beta,
                last_k_eff = self.dbg_last_k_eff,
                omega_rad = self.omega_rad,
                theta_freq_hz = rhythms.theta.freq_hz
            );
            self.dbg_accum_time = 0.0;
            self.dbg_wraps = 0;
            self.dbg_attacks = 0;
            self.dbg_boot_attacks = 0;
            self.dbg_attack_count_normal = 0;
            self.dbg_attack_sum_abs_diff = 0.0;
            self.dbg_attack_sum_cos = 0.0;
            self.dbg_attack_sum_sin = 0.0;
            self.dbg_fail_env = 0;
            self.dbg_fail_env_level = 0;
            self.dbg_fail_mag = 0;
            self.dbg_fail_alpha = 0;
            self.dbg_fail_beta = 0;
        }

        let relaxation = rhythms.theta.alpha * self.sensitivity.alpha;
        let tension = rhythms.theta.beta * self.sensitivity.beta;
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

#[derive(Debug, Clone, Copy)]
pub struct TargetProposal {
    pub target_pitch_log2: f32,
    pub salience: f32,
}

#[allow(clippy::too_many_arguments)]
pub trait PitchCore {
    fn propose_target<R: Rng + ?Sized>(
        &mut self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        current_freq_hz: f32,
        integration_window: f32,
        landscape: &Landscape,
        perceptual: &PerceptualContext,
        _features: &FeaturesNow,
        rng: &mut R,
    ) -> TargetProposal;
}

#[derive(Debug, Clone)]
pub struct PitchHillClimbPitchCore {
    neighbor_step_log2: f32,
    tessitura_center: f32,
    tessitura_gravity: f32,
    improvement_threshold: f32,
    exploration: f32,
    persistence: f32,
}

impl PitchHillClimbPitchCore {
    pub fn new(
        neighbor_step_cents: f32,
        tessitura_center: f32,
        tessitura_gravity: f32,
        improvement_threshold: f32,
        exploration: f32,
        persistence: f32,
    ) -> Self {
        Self {
            neighbor_step_log2: neighbor_step_cents / 1200.0,
            tessitura_center,
            tessitura_gravity,
            improvement_threshold,
            exploration: exploration.clamp(0.0, 1.0),
            persistence: persistence.clamp(0.0, 1.0),
        }
    }

    pub fn set_exploration(&mut self, value: f32) {
        self.exploration = value.clamp(0.0, 1.0);
    }

    pub fn set_persistence(&mut self, value: f32) {
        self.persistence = value.clamp(0.0, 1.0);
    }
}

impl PitchCore for PitchHillClimbPitchCore {
    fn propose_target<R: Rng + ?Sized>(
        &mut self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        _current_freq_hz: f32,
        integration_window: f32,
        landscape: &Landscape,
        perceptual: &PerceptualContext,
        _features: &FeaturesNow,
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
            let base = score - penalty - gravity_penalty;
            let idx = landscape.space.index_of_log2(clamped).unwrap_or(0);
            base + perceptual.score_adjustment(idx)
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
            let mut stay_prob = self.persistence.clamp(0.0, 1.0) * satisfaction;
            stay_prob = stay_prob.clamp(0.0, 1.0);
            let exploration = self.exploration.clamp(0.0, 1.0);
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
pub enum AnyPitchCore {
    PitchHillClimb(PitchHillClimbPitchCore),
}

impl PitchCore for AnyPitchCore {
    fn propose_target<R: Rng + ?Sized>(
        &mut self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        current_freq_hz: f32,
        integration_window: f32,
        landscape: &Landscape,
        perceptual: &PerceptualContext,
        features: &FeaturesNow,
        rng: &mut R,
    ) -> TargetProposal {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.propose_target(
                current_pitch_log2,
                current_target_log2,
                current_freq_hz,
                integration_window,
                landscape,
                perceptual,
                features,
                rng,
            ),
        }
    }
}

impl AnyPitchCore {
    pub fn set_exploration(&mut self, value: f32) {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.set_exploration(value),
        }
    }

    pub fn set_persistence(&mut self, value: f32) {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.set_persistence(value),
        }
    }
}

impl AnyArticulationCore {
    pub fn from_config<R: Rng + ?Sized>(
        config: &ArticulationCoreConfig,
        fs: f32,
        noise_seed: u64,
        rng: &mut R,
    ) -> Self {
        match config {
            ArticulationCoreConfig::Entrain {
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
                let env_level = if matches!(state, ArticulationState::Attack) {
                    attack_step
                } else {
                    0.0
                };
                let init_freq = rhythm_freq.unwrap_or_else(|| rng.random_range(5.0..7.0));
                // Phase/offset seed diversity; theta lock uses base_k ~ omega_target in process.
                AnyArticulationCore::Entrain(KuramotoCore {
                    energy,
                    basal_cost,
                    action_cost,
                    recharge_rate,
                    sensitivity: Sensitivity {
                        // rhythm_sensitivity targets theta coupling strength.
                        theta: rhythm_sensitivity.unwrap_or(sensitivity.theta),
                        ..sensitivity
                    },
                    rhythm_phase: rng.random_range(0.0..std::f32::consts::TAU),
                    rhythm_freq: init_freq,
                    omega_rad: TAU * init_freq,
                    phase_offset: rng.random_range(-std::f32::consts::PI..std::f32::consts::PI),
                    debug_id: noise_seed,
                    env_level,
                    state,
                    attack_step,
                    decay_factor,
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
                    dbg_accum_time: 0.0,
                    dbg_wraps: 0,
                    dbg_attacks: 0,
                    dbg_boot_attacks: 0,
                    dbg_attack_logs_left: 5,
                    dbg_attack_count_normal: 0,
                    dbg_attack_sum_abs_diff: 0.0,
                    dbg_attack_sum_cos: 0.0,
                    dbg_attack_sum_sin: 0.0,
                    dbg_fail_env: 0,
                    dbg_fail_env_level: 0,
                    dbg_fail_mag: 0,
                    dbg_fail_alpha: 0,
                    dbg_fail_beta: 0,
                    dbg_last_env_open: 0.0,
                    dbg_last_env_level: 0.0,
                    dbg_last_theta_mag: 0.0,
                    dbg_last_theta_alpha: 0.0,
                    dbg_last_theta_beta: 0.0,
                    dbg_last_k_eff: 0.0,
                })
            }
            ArticulationCoreConfig::Seq { duration } => {
                AnyArticulationCore::Seq(SequencedCore {
                timer: 0.0,
                duration: duration.max(0.0),
                env_level: 0.0,
                })
            }
            ArticulationCoreConfig::Drone { sway } => {
                let sway_rate = sway.unwrap_or(0.05);
                let sway_rate = if sway_rate <= 0.0 { 0.05 } else { sway_rate };
                AnyArticulationCore::Drone(DroneCore {
                    phase: rng.random_range(0.0..std::f32::consts::TAU),
                    sway_rate,
                })
            }
        }
    }
}

impl AnyPitchCore {
    pub fn from_config<R: Rng + ?Sized>(
        config: &PitchCoreConfig,
        initial_pitch_log2: f32,
        _rng: &mut R,
    ) -> Self {
        match config {
            PitchCoreConfig::PitchHillClimb {
                neighbor_step_cents,
                tessitura_gravity,
                improvement_threshold,
                exploration,
                persistence,
            } => {
                let neighbor_step_cents = neighbor_step_cents.unwrap_or(200.0);
                let tessitura_gravity = tessitura_gravity.unwrap_or(0.1);
                let improvement_threshold = improvement_threshold.unwrap_or(0.1);
                let exploration = exploration.unwrap_or(0.0);
                let persistence = persistence.unwrap_or(0.5);
                AnyPitchCore::PitchHillClimb(PitchHillClimbPitchCore::new(
                    neighbor_step_cents,
                    initial_pitch_log2,
                    tessitura_gravity,
                    improvement_threshold,
                    exploration,
                    persistence,
                ))
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
    pub articulation: ArticulationWrapper,
    pub pitch: AnyPitchCore,
    pub perceptual: PerceptualContext,
    pub body: AnySoundBody,
    pub last_signal: ArticulationSignal,
    pub release_gain: f32,
    pub release_sec: f32,
    pub release_pending: bool,
    pub target_pitch_log2: f32,
    pub integration_window: f32,
    pub accumulated_time: f32,
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
        self.articulation.set_gate(1.0);
        self.accumulated_time = 0.0;
        self.last_theta_sample = 0.0;
        self.last_target_salience = 0.0;
    }

    pub fn update_pitch_target(&mut self, rhythms: &NeuralRhythms, dt: f32, landscape: &Landscape) {
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
            let elapsed = self.accumulated_time;
            self.accumulated_time = 0.0;
            let features = FeaturesNow::from_subjective_intensity(&landscape.subjective_intensity);
            debug_assert_eq!(features.distribution.len(), landscape.space.n_bins());
            self.perceptual.ensure_len(features.distribution.len());
            let proposal = self.pitch.propose_target(
                current_pitch_log2,
                self.target_pitch_log2,
                current_freq,
                self.integration_window,
                landscape,
                &self.perceptual,
                &features,
                &mut self.rng,
            );
            self.target_pitch_log2 = proposal.target_pitch_log2;
            self.last_target_salience = proposal.salience;
            if let Some(idx) = landscape.space.index_of_log2(self.target_pitch_log2) {
                self.perceptual.update(idx, &features, elapsed);
            }
        }

        let (fmin, fmax) = landscape.freq_bounds_log2();
        self.target_pitch_log2 = self.target_pitch_log2.clamp(fmin, fmax);
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
        let mut rhythms = landscape.rhythm;
        for sample in buffer.iter_mut() {
            self.update_pitch_target(&rhythms, dt_per_sample, landscape);
            let current_freq = self.body.base_freq_hz().max(1.0);
            let current_pitch_log2 = current_freq.log2();
            let planned = PlannedPitch {
                target_pitch_log2: self.target_pitch_log2,
                jump_cents_abs: 1200.0 * (self.target_pitch_log2 - current_pitch_log2).abs(),
                salience: self.last_target_salience,
            };
            let apply_planned_pitch =
                self.articulation
                    .update_gate(&planned, &rhythms, dt_per_sample);
            if apply_planned_pitch {
                self.body.set_pitch_log2(planned.target_pitch_log2);
            }
            let consonance = landscape.evaluate_pitch01(self.body.base_freq_hz());
            let mut signal =
                self.articulation
                    .process(consonance, &rhythms, dt_per_sample, global_coupling);
            signal.amplitude *= self.articulation.gate();
            if self.release_pending {
                let step = dt_per_sample / self.release_sec.max(1e-6);
                self.release_gain = (self.release_gain - step).max(0.0);
            }
            signal.amplitude *= self.release_gain;
            signal.is_active = signal.is_active && signal.amplitude > 0.0;
            self.last_signal = signal;
            if signal.is_active {
                self.body
                    .articulate_wave(sample, fs, dt_per_sample, &signal);
            }
            rhythms.advance_in_place(dt_per_sample);
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
        self.articulation.is_alive() && self.release_gain > 0.0
    }
}
