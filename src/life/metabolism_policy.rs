use crate::life::lifecycle::LifecycleConfig;

pub const DEFAULT_ACTION_COST_PER_ATTACK: f32 = 0.02;
pub const DEFAULT_RECHARGE_PER_ATTACK: f32 = 0.5;
pub const DEFAULT_RECHARGE_THRESHOLD: f32 = 0.5;

#[derive(Clone, Copy, Debug)]
pub struct MetabolismPolicy {
    pub basal_cost_per_sec: f32,
    pub action_cost_per_attack: f32,
    pub recharge_per_attack: f32,
    pub recharge_threshold: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EnergyTelemetry {
    pub energy_before: f32,
    pub basal_delta: f32,
    pub action_delta: f32,
    pub recharge_delta: f32,
    pub energy_after: f32,
    pub did_attack: bool,
    pub consonance_used: f32,
}

impl MetabolismPolicy {
    pub fn from_lifecycle(lifecycle: &LifecycleConfig) -> Self {
        match lifecycle {
            LifecycleConfig::Decay { .. } => MetabolismPolicy {
                basal_cost_per_sec: 0.0,
                action_cost_per_attack: DEFAULT_ACTION_COST_PER_ATTACK,
                recharge_per_attack: 0.0,
                recharge_threshold: DEFAULT_RECHARGE_THRESHOLD,
            },
            LifecycleConfig::Sustain {
                metabolism_rate,
                recharge_rate,
                action_cost,
                ..
            } => MetabolismPolicy {
                basal_cost_per_sec: *metabolism_rate,
                action_cost_per_attack: action_cost.unwrap_or(DEFAULT_ACTION_COST_PER_ATTACK),
                recharge_per_attack: recharge_rate.unwrap_or(DEFAULT_RECHARGE_PER_ATTACK),
                recharge_threshold: DEFAULT_RECHARGE_THRESHOLD,
            },
        }
    }

    /// Energy update rule used by articulation cores.
    ///
    /// E' = E - basal_cost_per_sec * dt
    ///      - action_cost_per_attack * I_attack
    ///      + recharge_per_attack * consonance * I_attack * I(consonance > threshold)
    pub fn step(
        &self,
        energy: f32,
        dt: f32,
        did_attack: bool,
        consonance: f32,
    ) -> (f32, EnergyTelemetry) {
        let mut telemetry = EnergyTelemetry {
            energy_before: energy,
            did_attack,
            consonance_used: consonance,
            ..EnergyTelemetry::default()
        };

        let mut next = energy;
        let basal_delta = -self.basal_cost_per_sec * dt;
        next += basal_delta;
        telemetry.basal_delta = basal_delta;

        if did_attack {
            let action_delta = -self.action_cost_per_attack;
            let recharge_delta = if consonance > self.recharge_threshold {
                self.recharge_per_attack * consonance
            } else {
                0.0
            };
            next += action_delta + recharge_delta;
            telemetry.action_delta = action_delta;
            telemetry.recharge_delta = recharge_delta;
        }

        telemetry.energy_after = next;
        (next, telemetry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
    }

    #[test]
    fn step_no_attack_applies_basal_only() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 2.0,
            action_cost_per_attack: 1.0,
            recharge_per_attack: 0.5,
            recharge_threshold: 0.5,
        };
        let (next, tel) = policy.step(1.0, 0.25, false, 0.9);
        approx_eq(next, 0.5);
        approx_eq(tel.basal_delta, -0.5);
        approx_eq(tel.action_delta, 0.0);
        approx_eq(tel.recharge_delta, 0.0);
    }

    #[test]
    fn step_attack_without_recharge_below_threshold() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 0.0,
            action_cost_per_attack: 0.2,
            recharge_per_attack: 1.0,
            recharge_threshold: 0.5,
        };
        let (next, tel) = policy.step(1.0, 0.0, true, 0.5);
        approx_eq(next, 0.8);
        approx_eq(tel.action_delta, -0.2);
        approx_eq(tel.recharge_delta, 0.0);
    }

    #[test]
    fn step_attack_with_recharge_above_threshold() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 0.0,
            action_cost_per_attack: 0.2,
            recharge_per_attack: 0.5,
            recharge_threshold: 0.5,
        };
        let (next, tel) = policy.step(1.0, 0.0, true, 0.8);
        approx_eq(next, 1.0 - 0.2 + 0.4);
        approx_eq(tel.action_delta, -0.2);
        approx_eq(tel.recharge_delta, 0.4);
    }

    #[test]
    fn step_no_recharge_policy_is_consonance_independent() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 0.0,
            action_cost_per_attack: 0.2,
            recharge_per_attack: 0.0,
            recharge_threshold: 0.5,
        };
        let (a, _) = policy.step(1.0, 0.0, true, 0.2);
        let (b, _) = policy.step(1.0, 0.0, true, 0.9);
        approx_eq(a, b);
        approx_eq(a, 0.8);
    }
}
