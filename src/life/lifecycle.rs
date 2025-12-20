use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::scenario::EnvelopeConfig;
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LifecycleConfig {
    Decay {
        initial_energy: f32,
        half_life_sec: f32,
        #[serde(default = "default_decay_attack")]
        attack_sec: f32,
    },
    Sustain {
        initial_energy: f32,
        metabolism_rate: f32,
        #[serde(default)]
        recharge_rate: Option<f32>,
        #[serde(default)]
        action_cost: Option<f32>,
        envelope: EnvelopeConfig,
    },
}

impl LifecycleConfig {
    pub fn create_lifecycle(self) -> Box<dyn Lifecycle> {
        match self {
            LifecycleConfig::Decay {
                initial_energy,
                half_life_sec,
                attack_sec,
            } => Box::new(DecayLifecycle::new(
                initial_energy,
                half_life_sec,
                attack_sec,
            )),
            LifecycleConfig::Sustain {
                initial_energy,
                metabolism_rate,
                envelope,
                ..
            } => Box::new(SustainLifecycle::new(
                initial_energy,
                metabolism_rate,
                envelope,
            )),
        }
    }
}

impl fmt::Display for LifecycleConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LifecycleConfig::Decay {
                initial_energy,
                half_life_sec,
                attack_sec,
            } => write!(
                f,
                "lifecycle=decay(init={:.2}, half={:.3}s, attack={:.3}s)",
                initial_energy, half_life_sec, attack_sec
            ),
            LifecycleConfig::Sustain {
                initial_energy,
                metabolism_rate,
                recharge_rate,
                action_cost,
                envelope,
            } => write!(
                f,
                "lifecycle=sustain(init={:.2}, metab={:.3}/s, recharge={:.3}, action_cost={:.3}, env=[atk={:.3}s, dec={:.3}s, sus={:.2}])",
                initial_energy,
                metabolism_rate,
                recharge_rate.unwrap_or(0.5),
                action_cost.unwrap_or(0.02),
                envelope.attack_sec,
                envelope.decay_sec,
                envelope.sustain_level
            ),
        }
    }
}

pub trait Lifecycle: Send + Sync + std::fmt::Debug {
    /// Advance lifecycle and compute gain for this frame.
    fn process(&mut self, dt: f32, age: f32) -> f32;
    fn is_alive(&self) -> bool;
}

pub fn default_decay_attack() -> f32 {
    0.01
}

#[derive(Debug)]
pub struct DecayLifecycle {
    energy: f32,
    lambda: f32,
    attack_sec: f32,
}

impl DecayLifecycle {
    pub fn new(initial_energy: f32, half_life_sec: f32, attack_sec: f32) -> Self {
        let half_life_sec = half_life_sec.max(1e-6);
        let lambda = (0.5f32).ln() / half_life_sec;
        Self {
            energy: initial_energy.max(0.0),
            lambda,
            attack_sec: attack_sec.max(1e-6),
        }
    }
}

impl Lifecycle for DecayLifecycle {
    fn process(&mut self, dt: f32, age: f32) -> f32 {
        self.energy *= (self.lambda * dt).exp();
        if self.energy.abs() < 1e-8 {
            self.energy = 0.0;
        }
        let attack_gain = (age / self.attack_sec).min(1.0);
        let gain = self.energy * attack_gain;
        if gain < 1e-6 { 0.0 } else { gain }
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
    alive: bool,
}

impl SustainLifecycle {
    pub fn new(initial_energy: f32, metabolism_rate: f32, envelope: EnvelopeConfig) -> Self {
        Self {
            energy: initial_energy.max(0.0),
            metabolism_rate,
            envelope,
            alive: true,
        }
    }
}

impl Lifecycle for SustainLifecycle {
    fn process(&mut self, dt: f32, age: f32) -> f32 {
        if self.energy > 0.0 {
            self.energy -= self.metabolism_rate * dt;
            if self.energy < 0.0 {
                self.energy = 0.0;
            }
        }
        if self.energy <= 0.0 {
            self.alive = false;
            return 0.0;
        }

        let env = &self.envelope;
        let attack = env.attack_sec.max(1e-6);
        let decay = env.decay_sec.max(1e-6);
        let sustain = env.sustain_level.clamp(0.0, 1.0);

        let gain_env = if age < attack {
            (age / attack).min(1.0)
        } else if age < attack + decay {
            let t = (age - attack) / decay;
            1.0 + (sustain - 1.0) * t.clamp(0.0, 1.0)
        } else {
            sustain
        };

        let gain = gain_env * self.energy;
        if age < attack + decay && self.energy > 0.0 {
            self.alive = true;
            return gain;
        }

        if self.energy <= 0.0 || (gain <= 1e-6 && age > attack) {
            self.alive = false;
            0.0
        } else {
            self.alive = true;
            gain
        }
    }

    fn is_alive(&self) -> bool {
        self.alive
    }
}
