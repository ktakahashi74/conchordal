use crate::life::metabolism_policy::{
    DEFAULT_ATTACK_COST_FRACTION, DEFAULT_ATTACK_RECHARGE_FRACTION,
};
use crate::scenario::EnvelopeConfig;
use std::fmt;

#[derive(Debug, Clone)]
pub enum LifecycleConfig {
    Decay {
        half_life_sec: f32,
        attack_sec: f32,
    },
    Sustain {
        endurance_sec: Option<f32>,
        recovery_sec: Option<f32>,
        attack_cost_fraction: Option<f32>,
        attack_recharge_fraction: Option<f32>,
        continuous_recharge_score_low: Option<f32>,
        continuous_recharge_score_high: Option<f32>,
        selection_approx_loo: bool,
        dissonance_penalty: f32,
        envelope: EnvelopeConfig,
    },
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        LifecycleConfig::Decay {
            half_life_sec: 1.0,
            attack_sec: default_decay_attack(),
        }
    }
}

impl LifecycleConfig {
    pub fn endurance_sec(&self) -> Option<f32> {
        match self {
            LifecycleConfig::Sustain { endurance_sec, .. } => *endurance_sec,
            LifecycleConfig::Decay { .. } => None,
        }
    }

    pub fn recovery_sec(&self) -> Option<f32> {
        match self {
            LifecycleConfig::Sustain { recovery_sec, .. } => *recovery_sec,
            LifecycleConfig::Decay { .. } => None,
        }
    }
}

impl fmt::Display for LifecycleConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LifecycleConfig::Decay {
                half_life_sec,
                attack_sec,
            } => write!(
                f,
                "lifecycle=decay(half={:.3}s, attack={:.3}s)",
                half_life_sec, attack_sec
            ),
            LifecycleConfig::Sustain {
                endurance_sec,
                recovery_sec,
                attack_cost_fraction,
                attack_recharge_fraction,
                continuous_recharge_score_low,
                continuous_recharge_score_high,
                selection_approx_loo,
                dissonance_penalty,
                envelope,
                ..
            } => {
                let endurance =
                    endurance_sec.map_or_else(|| "off".to_string(), |sec| format!("{sec:.3}s"));
                let recovery =
                    recovery_sec.map_or_else(|| "off".to_string(), |sec| format!("{sec:.3}s"));
                write!(
                    f,
                    "lifecycle=sustain(endurance={}, recovery={}, attack_recharge={:.3}, attack_cost={:.3}, diss_penalty={:.3}, env=[atk={:.3}s, dec={:.3}s, sus={:.2}]",
                    endurance,
                    recovery,
                    attack_recharge_fraction.unwrap_or(DEFAULT_ATTACK_RECHARGE_FRACTION),
                    attack_cost_fraction.unwrap_or(DEFAULT_ATTACK_COST_FRACTION),
                    dissonance_penalty,
                    envelope.attack_sec,
                    envelope.decay_sec,
                    envelope.sustain_level
                )?;
                if let (Some(low), Some(high)) = (
                    continuous_recharge_score_low,
                    continuous_recharge_score_high,
                ) {
                    write!(f, ", consonance_viability=[{low:.3}, {high:.3}]")?;
                }
                if *selection_approx_loo {
                    write!(f, ", environment_relative")?;
                }
                write!(f, ")")
            }
        }
    }
}

pub fn default_decay_attack() -> f32 {
    0.01
}
