use crate::life::lifecycle::LifecycleConfig;

pub const DEFAULT_ACTION_COST_PER_ATTACK: f32 = 0.02;
pub const DEFAULT_RECHARGE_PER_ATTACK: f32 = 0.5;

#[derive(Clone, Copy, Debug)]
pub struct MetabolismPolicy {
    pub basal_cost_per_sec: f32,
    pub action_cost_per_attack: f32,
    pub recharge_per_attack: f32,
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

fn clamp01_finite(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

impl MetabolismPolicy {
    pub fn from_lifecycle(lifecycle: &LifecycleConfig) -> Self {
        match lifecycle {
            LifecycleConfig::Decay { .. } => MetabolismPolicy {
                basal_cost_per_sec: 0.0,
                action_cost_per_attack: DEFAULT_ACTION_COST_PER_ATTACK,
                recharge_per_attack: 0.0,
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
            },
        }
    }

    pub fn basal_delta(&self, dt: f32) -> f32 {
        -self.basal_cost_per_sec * dt.max(0.0)
    }

    pub fn attack_delta_with_recharge(&self, consonance: f32) -> f32 {
        -self.action_cost_per_attack + self.recharge_per_attack * clamp01_finite(consonance)
    }

    pub fn attack_delta_cost_only(&self) -> f32 {
        -self.action_cost_per_attack
    }

    pub fn apply_basal(&self, energy: f32, dt_sec: f32) -> (f32, EnergyTelemetry) {
        let basal_delta = self.basal_delta(dt_sec);
        let next = energy + basal_delta;
        let telemetry = EnergyTelemetry {
            energy_before: energy,
            basal_delta,
            action_delta: 0.0,
            recharge_delta: 0.0,
            energy_after: next,
            did_attack: false,
            consonance_used: 0.0,
        };
        (next, telemetry)
    }

    pub fn apply_attack_with_recharge(
        &self,
        energy: f32,
        consonance: f32,
    ) -> (f32, EnergyTelemetry) {
        let c = clamp01_finite(consonance);
        let action_delta = self.attack_delta_cost_only();
        let recharge_delta = self.recharge_per_attack * c;
        let next = energy + action_delta + recharge_delta;
        let telemetry = EnergyTelemetry {
            energy_before: energy,
            basal_delta: 0.0,
            action_delta,
            recharge_delta,
            energy_after: next,
            did_attack: true,
            consonance_used: c,
        };
        (next, telemetry)
    }

    pub fn apply_attack_cost_only(&self, energy: f32) -> (f32, EnergyTelemetry) {
        let action_delta = self.attack_delta_cost_only();
        let next = energy + action_delta;
        let telemetry = EnergyTelemetry {
            energy_before: energy,
            basal_delta: 0.0,
            action_delta,
            recharge_delta: 0.0,
            energy_after: next,
            did_attack: true,
            consonance_used: 0.0,
        };
        (next, telemetry)
    }

    /// Energy update rule used by articulation cores.
    ///
    /// E' = E - basal_cost_per_sec * dt
    ///      - action_cost_per_attack * I_attack
    ///      + recharge_per_attack * consonance_clamped * I_attack
    pub fn step(
        &self,
        energy: f32,
        dt: f32,
        did_attack: bool,
        consonance: f32,
    ) -> (f32, EnergyTelemetry) {
        let basal_delta = self.basal_delta(dt);
        let after_basal = energy + basal_delta;
        if did_attack {
            let attack_delta = self.attack_delta_with_recharge(consonance);
            let c = clamp01_finite(consonance);
            let after_attack = after_basal + attack_delta;
            let telemetry = EnergyTelemetry {
                energy_before: energy,
                basal_delta,
                action_delta: -self.action_cost_per_attack,
                recharge_delta: self.recharge_per_attack * c,
                energy_after: after_attack,
                did_attack: true,
                consonance_used: c,
            };
            (after_attack, telemetry)
        } else {
            let telemetry = EnergyTelemetry {
                energy_before: energy,
                basal_delta,
                action_delta: 0.0,
                recharge_delta: 0.0,
                energy_after: after_basal,
                did_attack: false,
                consonance_used: 0.0,
            };
            (after_basal, telemetry)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::scenario::EnvelopeConfig;

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
    }

    #[test]
    fn step_no_attack_applies_basal_only() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 2.0,
            action_cost_per_attack: 1.0,
            recharge_per_attack: 0.5,
        };
        let (next, tel) = policy.step(1.0, 0.25, false, 0.9);
        approx_eq(next, 0.5);
        approx_eq(tel.basal_delta, -0.5);
        approx_eq(tel.action_delta, 0.0);
        approx_eq(tel.recharge_delta, 0.0);
    }

    #[test]
    fn step_attack_recharge_is_linear_and_no_threshold() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 0.0,
            action_cost_per_attack: 0.2,
            recharge_per_attack: 0.5,
        };
        let (next_low, tel_low) = policy.step(1.0, 0.0, true, 0.2);
        approx_eq(next_low, 1.0 - 0.2 + 0.1);
        approx_eq(tel_low.action_delta, -0.2);
        approx_eq(tel_low.recharge_delta, 0.1);

        let (next_high, tel_high) = policy.step(1.0, 0.0, true, 0.8);
        approx_eq(next_high, 1.0 - 0.2 + 0.4);
        approx_eq(tel_high.action_delta, -0.2);
        approx_eq(tel_high.recharge_delta, 0.4);
    }

    #[test]
    fn step_attack_recharge_is_continuous_near_midpoint() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 0.0,
            action_cost_per_attack: 0.2,
            recharge_per_attack: 1.0,
        };
        let (_, tel_low) = policy.step(1.0, 0.0, true, 0.49);
        let (_, tel_high) = policy.step(1.0, 0.0, true, 0.51);
        assert!(tel_low.recharge_delta > 0.0);
        assert!(tel_high.recharge_delta > 0.0);
        let diff = (tel_high.recharge_delta - tel_low.recharge_delta).abs();
        assert!(diff < 0.05, "unexpected jump near midpoint: {diff}");
    }

    #[test]
    fn step_no_recharge_policy_is_consonance_independent() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 0.0,
            action_cost_per_attack: 0.2,
            recharge_per_attack: 0.0,
        };
        let (a, _) = policy.step(1.0, 0.0, true, 0.2);
        let (b, _) = policy.step(1.0, 0.0, true, 0.9);
        approx_eq(a, b);
        approx_eq(a, 0.8);
    }

    #[test]
    fn apply_basal_matches_step_when_no_attack() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 1.5,
            action_cost_per_attack: 0.3,
            recharge_per_attack: 0.2,
        };
        let (step_energy, step_tel) = policy.step(2.0, 0.5, false, 0.9);
        let (basal_energy, basal_tel) = policy.apply_basal(2.0, 0.5);
        approx_eq(step_energy, basal_energy);
        approx_eq(step_tel.basal_delta, basal_tel.basal_delta);
        approx_eq(step_tel.action_delta, 0.0);
        approx_eq(step_tel.recharge_delta, 0.0);
    }

    #[test]
    fn apply_attack_with_recharge_matches_step_dt0_attack() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 0.0,
            action_cost_per_attack: 0.4,
            recharge_per_attack: 0.25,
        };
        let (step_energy, step_tel) = policy.step(1.0, 0.0, true, 0.6);
        let (attack_energy, attack_tel) = policy.apply_attack_with_recharge(1.0, 0.6);
        approx_eq(step_energy, attack_energy);
        approx_eq(step_tel.action_delta, attack_tel.action_delta);
        approx_eq(step_tel.recharge_delta, attack_tel.recharge_delta);
    }

    #[test]
    fn apply_attack_cost_only_is_consonance_independent() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 0.0,
            action_cost_per_attack: 0.4,
            recharge_per_attack: 0.0,
        };
        let (cost_energy, _) = policy.apply_attack_cost_only(1.0);
        let (attack_low, _) = policy.apply_attack_with_recharge(1.0, 0.2);
        let (attack_high, _) = policy.apply_attack_with_recharge(1.0, 0.9);
        approx_eq(cost_energy, attack_low);
        approx_eq(cost_energy, attack_high);
    }

    #[test]
    fn attack_with_recharge_clamps_out_of_range() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 0.0,
            action_cost_per_attack: 0.2,
            recharge_per_attack: 0.5,
        };
        let (_, tel_low) = policy.apply_attack_with_recharge(1.0, -1.0);
        let (_, tel_high) = policy.apply_attack_with_recharge(1.0, 2.0);
        approx_eq(tel_low.recharge_delta, 0.0);
        approx_eq(tel_high.recharge_delta, 0.5);
    }

    #[test]
    fn attack_with_recharge_handles_nan() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 0.0,
            action_cost_per_attack: 0.2,
            recharge_per_attack: 0.5,
        };
        let (next, tel) = policy.apply_attack_with_recharge(1.0, f32::NAN);
        assert!(next.is_finite());
        approx_eq(tel.recharge_delta, 0.0);
    }

    #[test]
    fn step_recharge_yields_longer_lifetime_for_higher_consonance() {
        let policy = MetabolismPolicy {
            basal_cost_per_sec: 0.4,
            action_cost_per_attack: 0.2,
            recharge_per_attack: 0.5,
        };
        let low = simulate_lifetime(policy, 0.2);
        let high = simulate_lifetime(policy, 0.8);
        assert!(high > low, "expected higher consonance to last longer");

        let no_recharge = MetabolismPolicy {
            recharge_per_attack: 0.0,
            ..policy
        };
        let low_nr = simulate_lifetime(no_recharge, 0.2);
        let high_nr = simulate_lifetime(no_recharge, 0.8);
        assert_eq!(low_nr, high_nr);
    }

    #[test]
    fn from_lifecycle_sustain_defaults_recharge_rate() {
        let lifecycle = LifecycleConfig::Sustain {
            initial_energy: 1.0,
            metabolism_rate: 0.1,
            recharge_rate: None,
            action_cost: None,
            envelope: EnvelopeConfig::default(),
        };
        let policy = MetabolismPolicy::from_lifecycle(&lifecycle);
        approx_eq(policy.recharge_per_attack, DEFAULT_RECHARGE_PER_ATTACK);
    }

    #[test]
    fn from_lifecycle_sustain_zero_recharge_rate() {
        let lifecycle = LifecycleConfig::Sustain {
            initial_energy: 1.0,
            metabolism_rate: 0.1,
            recharge_rate: Some(0.0),
            action_cost: None,
            envelope: EnvelopeConfig::default(),
        };
        let policy = MetabolismPolicy::from_lifecycle(&lifecycle);
        approx_eq(policy.recharge_per_attack, 0.0);
    }

    fn simulate_lifetime(policy: MetabolismPolicy, consonance: f32) -> usize {
        let mut energy = 1.0;
        let mut steps = 0usize;
        while energy > 0.0 && steps < 10_000 {
            let (next, _) = policy.step(energy, 1.0, true, consonance);
            energy = next;
            steps += 1;
        }
        steps
    }
}
