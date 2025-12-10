use crate::core::landscape::Landscape;
use crate::core::modulation::NeuralRhythms;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::f32::consts::PI;

/// Hybrid synthesis agents render both time-domain audio and a spectral "body".
pub trait AudioAgent: Send + Sync + 'static {
    fn id(&self) -> u64;
    fn metadata(&self) -> &AgentMetadata;
    fn render_wave(
        &mut self,
        buffer: &mut [f32],
        fs: f32,
        current_frame: u64,
        dt_sec: f32,
        landscape: &Landscape,
    );
    fn render_spectrum(
        &mut self,
        amps: &mut [f32],
        fs: f32,
        nfft: usize,
        current_frame: u64,
        dt_sec: f32,
    );
    fn is_alive(&self) -> bool;
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
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
    state: f32,
    leak: f32,
    gain: f32,
}

impl PinkNoise {
    pub fn new(seed: u64, gain: f32, leak: f32) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
            state: 0.0,
            leak: leak.clamp(0.0, 0.9999),
            gain,
        }
    }

    pub fn next(&mut self) -> f32 {
        let white: f32 = self.rng.gen_range(-1.0..=1.0);
        self.state = self.state * self.leak + white * (1.0 - self.leak);
        self.state * self.gain
    }
}

/// Bio-Neuro-Rhythmic individual.
#[derive(Debug, Clone)]
pub struct Individual {
    pub id: u64,
    pub metadata: AgentMetadata,
    pub freq_hz: f32,
    pub amp: f32,
    pub energy: f32,
    pub basal_cost: f32,
    pub action_cost: f32,
    pub recharge_rate: f32,
    pub sensitivity: Sensitivity,
    // Rhythm domain
    pub rhythm_phase: f32,
    pub rhythm_freq: f32,
    // Audio domain
    pub audio_phase: f32,
    pub env_level: f32,
    pub state: ArticulationState,
    pub attack_step: f32,
    pub decay_factor: f32,
    pub omega: f32,
    pub noise_1f: PinkNoise,
    pub confidence: f32,
    pub gate_threshold: f32,
}

impl Individual {
    pub fn update(&mut self, consonance: f32, rhythms: &NeuralRhythms, dt: f32) {
        let dt = dt.max(1e-4);

        // 1. Metabolism
        self.energy -= self.basal_cost * dt;
        if self.energy <= 0.0 {
            self.state = ArticulationState::Idle;
            self.env_level = 0.0;
            return;
        }

        // 2. Rhythm sensing and phase update (Kuramoto on slow clock)
        let delta_phase = rhythms.delta.phase;
        let delta_mag = rhythms.delta.mag * self.sensitivity.delta;
        let theta_signal = rhythms.theta.mag * rhythms.theta.phase.cos() * self.sensitivity.theta;
        let trigger_force = theta_signal * consonance;

        // 3. Rhythm phase dynamics (confidence-weighted coupling)
        let beta_err = rhythms.beta.mag * self.sensitivity.beta;
        self.confidence = 1.0 - beta_err;
        let coupling = (self.sensitivity.beta * (1.0 / (self.confidence + 1e-3))).min(10.0);
        let noise = self.noise_1f.next();
        let rhythm_omega = 2.0 * PI * self.rhythm_freq;
        let d_phi = rhythm_omega + noise + coupling * (delta_phase - self.rhythm_phase).sin();
        self.rhythm_phase += d_phi * dt;

        // 4. Trigger on rhythm wrap + gate
        if self.rhythm_phase >= 2.0 * PI {
            self.rhythm_phase -= 2.0 * PI;
            let gate_bias = 0.2;
            let gate = delta_mag + trigger_force + gate_bias;
            if gate > self.gate_threshold && self.state == ArticulationState::Idle {
                self.audio_phase = 0.0;
                self.env_level = 0.0;
                self.state = ArticulationState::Attack;
                self.energy -= self.action_cost;
                if consonance > 0.5 {
                    self.energy += self.recharge_rate * consonance;
                }
            }
        }
    }
}

impl AudioAgent for Individual {
    fn id(&self) -> u64 {
        self.id
    }

    fn metadata(&self) -> &AgentMetadata {
        &self.metadata
    }

    fn render_wave(
        &mut self,
        buffer: &mut [f32],
        fs: f32,
        _current_frame: u64,
        dt_sec: f32,
        landscape: &Landscape,
    ) {
        let dt_per_sample = dt_sec / buffer.len() as f32;
        let rhythms = landscape.rhythm;
        let consonance = landscape.consonance_at(self.freq_hz);
        // Angular frequency in rad/s for audio oscillator
        let omega = 2.0 * PI * self.freq_hz;
        for s in buffer.iter_mut() {
            self.omega = omega;
            self.update(consonance, &rhythms, dt_per_sample);
            match self.state {
                ArticulationState::Idle => {
                    // silence
                }
                ArticulationState::Attack => {
                    self.env_level += self.attack_step;
                    if self.env_level >= 1.0 {
                        self.env_level = 1.0;
                        self.state = ArticulationState::Decay;
                    }
                    self.audio_phase =
                        (self.audio_phase + self.omega * dt_per_sample).rem_euclid(2.0 * PI);
                    *s += self.amp * self.env_level * self.audio_phase.sin();
                }
                ArticulationState::Decay => {
                    self.env_level *= self.decay_factor;
                    if self.env_level < 0.001 {
                        self.env_level = 0.0;
                        self.state = ArticulationState::Idle;
                    }
                    self.audio_phase =
                        (self.audio_phase + self.omega * dt_per_sample).rem_euclid(2.0 * PI);
                    *s += self.amp * self.env_level * self.audio_phase.sin();
                }
            }
        }
    }

    fn render_spectrum(
        &mut self,
        amps: &mut [f32],
        fs: f32,
        nfft: usize,
        _current_frame: u64,
        _dt_sec: f32,
    ) {
        // Use current envelope/state so spectral body matches audible body.
        if self.state == ArticulationState::Idle || self.env_level <= 0.0 {
            return;
        }
        let bin_f = self.freq_hz * nfft as f32 / fs;
        let k = bin_f.round() as isize;
        if k >= 0 && (k as usize) < amps.len() {
            amps[k as usize] += self.amp.max(0.0) * self.env_level;
        }
    }

    fn is_alive(&self) -> bool {
        self.energy > 0.0
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
