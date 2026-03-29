use super::scenario::EnvelopeConfig;
use crate::life::metabolism_policy::{DEFAULT_ACTION_COST_PER_ATTACK, DEFAULT_RECHARGE_PER_ATTACK};
use std::fmt;

#[derive(Debug, Clone)]
pub enum LifecycleConfig {
    Decay {
        initial_energy: f32,
        half_life_sec: f32,
        attack_sec: f32,
    },
    Sustain {
        initial_energy: f32,
        metabolism_rate: f32,
        recharge_rate: Option<f32>,
        action_cost: Option<f32>,
        continuous_recharge_rate: Option<f32>,
        dissonance_cost: Option<f32>,
        envelope: EnvelopeConfig,
    },
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        LifecycleConfig::Decay {
            initial_energy: 1.0,
            half_life_sec: 1.0,
            attack_sec: default_decay_attack(),
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
                dissonance_cost,
                envelope,
                ..
            } => {
                write!(
                    f,
                    "lifecycle=sustain(init={:.2}, metab={:.3}/s, recharge={:.3}, action_cost={:.3}, env=[atk={:.3}s, dec={:.3}s, sus={:.2}]",
                    initial_energy,
                    metabolism_rate,
                    recharge_rate.unwrap_or(DEFAULT_RECHARGE_PER_ATTACK),
                    action_cost.unwrap_or(DEFAULT_ACTION_COST_PER_ATTACK),
                    envelope.attack_sec,
                    envelope.decay_sec,
                    envelope.sustain_level
                )?;
                if let Some(dc) = dissonance_cost {
                    write!(f, ", diss_cost={dc:.3}")?;
                }
                write!(f, ")")
            }
        }
    }
}

pub fn default_decay_attack() -> f32 {
    0.01
}
