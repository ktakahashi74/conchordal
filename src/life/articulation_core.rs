use crate::core::modulation::NeuralRhythms;
use crate::core::phase::{angle_diff_pm_pi, wrap_0_tau};
use crate::core::utils::pink_noise_tick;
use crate::life::lifecycle::LifecycleConfig;
use crate::life::scenario::ArticulationCoreConfig;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::f32::consts::{PI, TAU};
use tracing::debug;

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
            ArticulationCoreConfig::Seq { duration } => AnyArticulationCore::Seq(SequencedCore {
                timer: 0.0,
                duration: duration.max(0.0),
                env_level: 0.0,
            }),
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
