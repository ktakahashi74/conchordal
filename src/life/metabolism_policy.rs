use crate::core::float::clamp01_finite;
use crate::life::constants::MAX_RECHARGE_MULT;
use crate::life::lifecycle::LifecycleConfig;

pub const DEFAULT_ATTACK_COST_FRACTION: f32 = 0.02;
pub const DEFAULT_ATTACK_RECHARGE_FRACTION: f32 = 0.5;

#[derive(Clone, Copy, Debug)]
pub struct MetabolismPolicy {
    pub basal_cost_per_sec: f32,
    pub action_cost_per_attack: f32,
    pub recharge_per_attack: f32,
    pub continuous_recharge_per_sec: f32,
    pub continuous_recharge_score_low: Option<f32>,
    pub continuous_recharge_score_high: Option<f32>,
    pub dissonance_penalty: f32,
}

impl MetabolismPolicy {
    pub fn from_lifecycle(lifecycle: &LifecycleConfig) -> Self {
        match lifecycle {
            LifecycleConfig::Decay { .. } => MetabolismPolicy {
                basal_cost_per_sec: 0.0,
                action_cost_per_attack: DEFAULT_ATTACK_COST_FRACTION,
                recharge_per_attack: 0.0,
                continuous_recharge_per_sec: 0.0,
                continuous_recharge_score_low: None,
                continuous_recharge_score_high: None,
                dissonance_penalty: 0.0,
            },
            LifecycleConfig::Sustain {
                endurance_sec,
                recovery_sec,
                attack_cost_fraction,
                attack_recharge_fraction,
                continuous_recharge_score_low,
                continuous_recharge_score_high,
                dissonance_penalty,
                ..
            } => {
                let penalty = if dissonance_penalty.is_finite() {
                    dissonance_penalty.max(0.0)
                } else {
                    0.0
                };
                let basal_cost_per_sec = endurance_sec
                    .filter(|sec| sec.is_finite() && *sec > 0.0)
                    .map_or(0.0, |sec| 1.0 / (sec * (1.0 + penalty)));
                let continuous_recharge_per_sec = recovery_sec
                    .filter(|sec| sec.is_finite() && *sec > 0.0)
                    .map_or(0.0, |sec| 1.0 / sec);
                MetabolismPolicy {
                    basal_cost_per_sec,
                    action_cost_per_attack: attack_cost_fraction
                        .unwrap_or(DEFAULT_ATTACK_COST_FRACTION)
                        .max(0.0),
                    recharge_per_attack: attack_recharge_fraction
                        .unwrap_or(DEFAULT_ATTACK_RECHARGE_FRACTION)
                        .max(0.0),
                    continuous_recharge_per_sec,
                    continuous_recharge_score_low: *continuous_recharge_score_low,
                    continuous_recharge_score_high: *continuous_recharge_score_high,
                    dissonance_penalty: penalty,
                }
            }
        }
    }

    pub fn basal_delta(&self, dt: f32, consonance: f32) -> f32 {
        let c = clamp01_finite(consonance);
        let factor = 1.0 + self.dissonance_penalty * (1.0 - c);
        -self.basal_cost_per_sec * factor * dt.max(0.0)
    }

    fn continuous_recharge_signal(&self, consonance_level: f32, selection_score: f32) -> f32 {
        match (
            self.continuous_recharge_score_low,
            self.continuous_recharge_score_high,
        ) {
            (Some(low), Some(high)) => {
                if !selection_score.is_finite() {
                    return 0.0;
                }
                let hi = if high > low { high } else { low + 1e-6 };
                ((selection_score - low) / (hi - low)).clamp(0.0, 1.0)
            }
            _ => clamp01_finite(consonance_level),
        }
    }

    pub fn continuous_recharge_delta(
        &self,
        dt: f32,
        consonance_level: f32,
        selection_score: f32,
    ) -> f32 {
        if self.continuous_recharge_per_sec == 0.0 {
            return 0.0;
        }
        self.continuous_recharge_per_sec
            * self.continuous_recharge_signal(consonance_level, selection_score)
            * dt.max(0.0)
    }

    pub fn attack_delta_with_recharge_multiplier(
        &self,
        consonance: f32,
        recharge_mult: f32,
    ) -> f32 {
        let mult = if recharge_mult.is_finite() {
            recharge_mult.clamp(0.0, MAX_RECHARGE_MULT)
        } else {
            1.0
        };
        -self.action_cost_per_attack + self.recharge_per_attack * clamp01_finite(consonance) * mult
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::EnvelopeConfig;

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
    }

    #[test]
    fn attack_delta_with_recharge_multiplier_scales_recharge_only() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 0.0,
            action_cost_per_attack: 0.3,
            recharge_per_attack: 0.5,
            continuous_recharge_per_sec: 0.0,
            continuous_recharge_score_low: None,
            continuous_recharge_score_high: None,
            dissonance_penalty: 0.0,
        };
        let c = 0.6;
        let base = policy.attack_delta_with_recharge_multiplier(c, 1.0);
        let boosted = policy.attack_delta_with_recharge_multiplier(c, 2.0);
        approx_eq(base, -0.3 + 0.5 * c);
        approx_eq(boosted, -0.3 + 0.5 * c * 2.0);
        approx_eq(boosted - base, 0.5 * c);
    }

    #[test]
    fn attack_delta_with_recharge_multiplier_clamps_and_handles_zero_consonance() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 0.0,
            action_cost_per_attack: 0.25,
            recharge_per_attack: 0.5,
            continuous_recharge_per_sec: 0.0,
            continuous_recharge_score_low: None,
            continuous_recharge_score_high: None,
            dissonance_penalty: 0.0,
        };
        let low = policy.attack_delta_with_recharge_multiplier(0.8, -10.0);
        let high = policy.attack_delta_with_recharge_multiplier(0.8, 10.0);
        let nan = policy.attack_delta_with_recharge_multiplier(0.8, f32::NAN);
        approx_eq(low, -0.25);
        approx_eq(high, -0.25 + 0.5 * 0.8 * 2.0);
        approx_eq(nan, -0.25 + 0.5 * 0.8);

        let zero_c_low = policy.attack_delta_with_recharge_multiplier(0.0, 0.0);
        let zero_c_high = policy.attack_delta_with_recharge_multiplier(0.0, 2.0);
        approx_eq(zero_c_low, -0.25);
        approx_eq(zero_c_high, -0.25);
    }

    #[test]
    fn from_lifecycle_sustain_defaults_attack_recharge_fraction() {
        let lifecycle = LifecycleConfig::Sustain {
            endurance_sec: Some(10.0),
            recovery_sec: None,
            attack_cost_fraction: None,
            attack_recharge_fraction: None,
            continuous_recharge_score_low: None,
            continuous_recharge_score_high: None,
            selection_approx_loo: false,
            dissonance_penalty: 0.0,
            envelope: EnvelopeConfig::default(),
        };
        let policy = MetabolismPolicy::from_lifecycle(&lifecycle);
        approx_eq(policy.recharge_per_attack, DEFAULT_ATTACK_RECHARGE_FRACTION);
    }

    #[test]
    fn from_lifecycle_sustain_zero_attack_recharge_fraction() {
        let lifecycle = LifecycleConfig::Sustain {
            endurance_sec: Some(10.0),
            recovery_sec: None,
            attack_cost_fraction: None,
            attack_recharge_fraction: Some(0.0),
            continuous_recharge_score_low: None,
            continuous_recharge_score_high: None,
            selection_approx_loo: false,
            dissonance_penalty: 0.0,
            envelope: EnvelopeConfig::default(),
        };
        let policy = MetabolismPolicy::from_lifecycle(&lifecycle);
        approx_eq(policy.recharge_per_attack, 0.0);
    }

    #[test]
    fn continuous_recharge_delta_scales_with_consonance_and_dt() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 0.0,
            action_cost_per_attack: 0.0,
            recharge_per_attack: 0.0,
            continuous_recharge_per_sec: 0.5,
            continuous_recharge_score_low: None,
            continuous_recharge_score_high: None,
            dissonance_penalty: 0.0,
        };
        approx_eq(policy.continuous_recharge_delta(1.0, 1.0, 0.0), 0.5);
        approx_eq(policy.continuous_recharge_delta(0.5, 1.0, 0.0), 0.25);
        approx_eq(policy.continuous_recharge_delta(1.0, 0.6, 0.0), 0.3);
        approx_eq(policy.continuous_recharge_delta(1.0, 0.0, 0.0), 0.0);
        // negative dt clamped to 0
        approx_eq(policy.continuous_recharge_delta(-1.0, 1.0, 0.0), 0.0);
    }

    #[test]
    fn continuous_recharge_delta_zero_rate_returns_zero() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 0.0,
            action_cost_per_attack: 0.0,
            recharge_per_attack: 0.0,
            continuous_recharge_per_sec: 0.0,
            continuous_recharge_score_low: None,
            continuous_recharge_score_high: None,
            dissonance_penalty: 0.0,
        };
        approx_eq(policy.continuous_recharge_delta(1.0, 1.0, 0.0), 0.0);
        approx_eq(policy.continuous_recharge_delta(1.0, 0.5, 0.0), 0.0);
    }

    #[test]
    fn continuous_recharge_delta_uses_consonance_viability_window_when_configured() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 0.0,
            action_cost_per_attack: 0.0,
            recharge_per_attack: 0.0,
            continuous_recharge_per_sec: 0.5,
            continuous_recharge_score_low: Some(0.3),
            continuous_recharge_score_high: Some(0.8),
            dissonance_penalty: 0.0,
        };
        approx_eq(policy.continuous_recharge_delta(1.0, 0.9, 0.2), 0.0);
        approx_eq(policy.continuous_recharge_delta(1.0, 0.1, 0.55), 0.25);
        approx_eq(policy.continuous_recharge_delta(1.0, 0.0, 0.8), 0.5);
    }

    #[test]
    fn from_lifecycle_derives_recovery_rate() {
        let lifecycle = LifecycleConfig::Sustain {
            endurance_sec: Some(10.0),
            recovery_sec: Some(4.0),
            attack_cost_fraction: None,
            attack_recharge_fraction: None,
            continuous_recharge_score_low: Some(0.3),
            continuous_recharge_score_high: Some(0.8),
            selection_approx_loo: false,
            dissonance_penalty: 0.0,
            envelope: EnvelopeConfig::default(),
        };
        let policy = MetabolismPolicy::from_lifecycle(&lifecycle);
        approx_eq(policy.continuous_recharge_per_sec, 0.25);
        assert_eq!(policy.continuous_recharge_score_low, Some(0.3));
        assert_eq!(policy.continuous_recharge_score_high, Some(0.8));

        let lifecycle_none = LifecycleConfig::Sustain {
            endurance_sec: Some(10.0),
            recovery_sec: None,
            attack_cost_fraction: None,
            attack_recharge_fraction: None,
            continuous_recharge_score_low: None,
            continuous_recharge_score_high: None,
            selection_approx_loo: false,
            dissonance_penalty: 0.0,
            envelope: EnvelopeConfig::default(),
        };
        let policy_none = MetabolismPolicy::from_lifecycle(&lifecycle_none);
        approx_eq(policy_none.continuous_recharge_per_sec, 0.0);
    }

    #[test]
    fn endurance_contract_is_exact_at_zero_fit() {
        let lifecycle = LifecycleConfig::Sustain {
            endurance_sec: Some(8.0),
            recovery_sec: None,
            attack_cost_fraction: Some(0.0),
            attack_recharge_fraction: Some(0.0),
            continuous_recharge_score_low: None,
            continuous_recharge_score_high: None,
            selection_approx_loo: false,
            dissonance_penalty: 3.0,
            envelope: EnvelopeConfig::default(),
        };
        let policy = MetabolismPolicy::from_lifecycle(&lifecycle);
        approx_eq(policy.basal_cost_per_sec, 1.0 / 32.0);
        approx_eq(policy.basal_delta(8.0, 0.0), -1.0);
    }

    #[test]
    fn endurance_penalty_only_shapes_the_fit_response() {
        let lifecycle = LifecycleConfig::Sustain {
            endurance_sec: Some(8.0),
            recovery_sec: None,
            attack_cost_fraction: Some(0.0),
            attack_recharge_fraction: Some(0.0),
            continuous_recharge_score_low: None,
            continuous_recharge_score_high: None,
            selection_approx_loo: false,
            dissonance_penalty: 3.0,
            envelope: EnvelopeConfig::default(),
        };
        let policy = MetabolismPolicy::from_lifecycle(&lifecycle);
        approx_eq(policy.basal_delta(8.0, 0.0), -1.0);
        approx_eq(policy.basal_delta(8.0, 1.0), -0.25);
    }

    #[test]
    fn zero_dissonance_penalty_is_fit_independent() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 1.0,
            action_cost_per_attack: 0.0,
            recharge_per_attack: 0.0,
            continuous_recharge_per_sec: 0.0,
            continuous_recharge_score_low: None,
            continuous_recharge_score_high: None,
            dissonance_penalty: 0.0,
        };
        let d1 = policy.basal_delta(1.0, 0.0);
        let d2 = policy.basal_delta(1.0, 1.0);
        assert!((d1 - d2).abs() < 1e-6);
    }

    #[test]
    fn dissonance_penalty_amplifies_drain_at_low_consonance() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 1.0,
            action_cost_per_attack: 0.0,
            recharge_per_attack: 0.0,
            continuous_recharge_per_sec: 0.0,
            continuous_recharge_score_low: None,
            continuous_recharge_score_high: None,
            dissonance_penalty: 2.0,
        };
        let at_zero = policy.basal_delta(1.0, 0.0);
        let at_half = policy.basal_delta(1.0, 0.5);
        let at_one = policy.basal_delta(1.0, 1.0);
        approx_eq(at_zero, -3.0);
        approx_eq(at_half, -2.0);
        approx_eq(at_one, -1.0);
    }
}
