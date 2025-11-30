use serde::{Deserialize, Serialize};

use super::scenario::EnvelopeConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LifecycleConfig {
    Decay {
        initial_energy: f32,
        half_life_sec: f32,
    },
    Sustain {
        initial_energy: f32,
        metabolism_rate: f32,
        envelope: EnvelopeConfig,
    },
}

impl LifecycleConfig {
    pub fn create_lifecycle(self) -> Box<dyn Lifecycle> {
        match self {
            LifecycleConfig::Decay {
                initial_energy,
                half_life_sec,
            } => Box::new(DecayLifecycle::new(initial_energy, half_life_sec)),
            LifecycleConfig::Sustain {
                initial_energy,
                metabolism_rate,
                envelope,
            } => Box::new(SustainLifecycle::new(
                initial_energy,
                metabolism_rate,
                envelope,
            )),
        }
    }
}

pub trait Lifecycle: Send + Sync + std::fmt::Debug {
    /// Advance lifecycle and compute gain for this frame.
    fn process(&mut self, dt: f32, age: f32) -> f32;
    fn is_alive(&self) -> bool;
}

#[derive(Debug)]
pub struct DecayLifecycle {
    energy: f32,
    lambda: f32,
}

impl DecayLifecycle {
    pub fn new(initial_energy: f32, half_life_sec: f32) -> Self {
        let half_life_sec = half_life_sec.max(1e-6);
        let lambda = (0.5f32).ln() / half_life_sec;
        Self {
            energy: initial_energy.max(0.0),
            lambda,
        }
    }
}

impl Lifecycle for DecayLifecycle {
    fn process(&mut self, dt: f32, _age: f32) -> f32 {
        self.energy *= (self.lambda * dt).exp();
        self.energy
    }

    fn is_alive(&self) -> bool {
        self.energy > 1e-4
    }
}

#[derive(Debug)]
pub struct SustainLifecycle {
    energy: f32,
    metabolism_rate: f32,
    envelope: EnvelopeConfig,
    fade_out_factor: f32,
    alive: bool,
}

impl SustainLifecycle {
    pub fn new(initial_energy: f32, metabolism_rate: f32, envelope: EnvelopeConfig) -> Self {
        Self {
            energy: initial_energy.max(0.0),
            metabolism_rate,
            envelope,
            fade_out_factor: 1.0,
            alive: true,
        }
    }
}

impl Lifecycle for SustainLifecycle {
    fn process(&mut self, dt: f32, age: f32) -> f32 {
        if self.energy > 0.0 {
            self.energy -= self.metabolism_rate * dt;
        } else {
            self.energy = 0.0;
        }

        let env = &self.envelope;
        let attack = env.attack_sec.max(1e-6);
        let decay = env.decay_sec.max(1e-6);
        let sustain = env.sustain_level.clamp(0.0, 1.0);

        let gain_env = if age < attack {
            (age / attack).min(1.0)
        } else {
            let t_decay = age - attack;
            sustain + (1.0 - sustain) * (-t_decay / decay).exp()
        };

        // Energy directly scales the envelope; no normalization here so callers can choose absolute amplitude.
        let energy_gain = self.energy.max(0.0);

        if self.energy <= 0.0 {
            self.fade_out_factor *= 0.9;
        }

        let gain = gain_env * self.fade_out_factor * energy_gain;
        self.alive = self.energy > 0.0 || gain > 1e-4;
        gain
    }

    fn is_alive(&self) -> bool {
        self.alive
    }
}
