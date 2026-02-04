use conchordal::life::metabolism_policy::MetabolismPolicy;

fn approx_eq(a: f32, b: f32) {
    assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
}

#[test]
fn basal_delta_matches_step_when_no_attack() {
    let policy = MetabolismPolicy {
        basal_cost_per_sec: 1.5,
        action_cost_per_attack: 0.3,
        recharge_per_attack: 0.2,
    };
    let (step_energy, _) = policy.step(2.0, 0.5, false, 0.9);
    let energy = 2.0 + policy.basal_delta(0.5);
    approx_eq(step_energy, energy);
}

#[test]
fn attack_delta_with_recharge_is_linear() {
    let policy = MetabolismPolicy {
        basal_cost_per_sec: 0.0,
        action_cost_per_attack: 0.2,
        recharge_per_attack: 0.5,
    };
    let delta_low = policy.attack_delta_with_recharge(0.2);
    let delta_high = policy.attack_delta_with_recharge(0.8);
    approx_eq(delta_low, -0.2 + 0.1);
    approx_eq(delta_high, -0.2 + 0.4);
}

#[test]
fn attack_delta_continuous_near_midpoint() {
    let policy = MetabolismPolicy {
        basal_cost_per_sec: 0.0,
        action_cost_per_attack: 0.2,
        recharge_per_attack: 1.0,
    };
    let delta_low = policy.attack_delta_with_recharge(0.49);
    let delta_high = policy.attack_delta_with_recharge(0.51);
    let diff = (delta_high - delta_low).abs();
    assert!(diff < 0.05, "unexpected jump near midpoint: {diff}");
}

#[test]
fn attack_delta_cost_only_is_consonance_independent() {
    let policy = MetabolismPolicy {
        basal_cost_per_sec: 0.0,
        action_cost_per_attack: 0.2,
        recharge_per_attack: 0.0,
    };
    let delta_low = policy.attack_delta_with_recharge(0.2);
    let delta_high = policy.attack_delta_with_recharge(0.9);
    approx_eq(delta_low, delta_high);
    approx_eq(delta_low, policy.attack_delta_cost_only());
}

#[test]
fn attack_delta_clamps_out_of_range_and_nan() {
    let policy = MetabolismPolicy {
        basal_cost_per_sec: 0.0,
        action_cost_per_attack: 0.2,
        recharge_per_attack: 0.5,
    };
    let delta_low = policy.attack_delta_with_recharge(-1.0);
    let delta_high = policy.attack_delta_with_recharge(2.0);
    let delta_nan = policy.attack_delta_with_recharge(f32::NAN);
    approx_eq(delta_low, -0.2);
    approx_eq(delta_high, -0.2 + 0.5);
    approx_eq(delta_nan, -0.2);
}
