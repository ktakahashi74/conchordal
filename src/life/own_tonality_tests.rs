use super::*;

#[test]
fn tonality_rest_preserves_shape_and_new_input_replaces_tiny_memory() {
    let mut model = TonalityModel {
        active: Some((0, 0.0)),
        ..Default::default()
    };
    model.advance(24000, 48000.0);
    let mass = model.memory[0] * model.log_scale.exp();
    let expected = ((-0.5f64 / 3.0).exp() - (-1.0f64).exp()) / (5.0 / 3.0);
    assert!((mass - expected).abs() < 1e-15);
    model.active = None;
    let before = model.estimate().correlations.unwrap();
    model.advance(480000000, 48000.0);
    assert_eq!(before, model.estimate().correlations.unwrap());
    model.active = Some((7, 0.0));
    model.advance(480024000, 48000.0);
    assert_eq!(model.memory[7], 1.0);
    assert_eq!(model.memory[0], 0.0);
}

#[test]
fn tonality_symmetry_is_defined_but_selection_unresolved() {
    let model = TonalityModel {
        memory: [1.0; 12],
        ..Default::default()
    };
    assert!(model.estimate().correlations.is_none());
    let mut model = TonalityModel {
        memory: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ..Default::default()
    };
    assert!(model.estimate().correlations.is_some());
    assert_eq!(
        model.estimate().reason,
        Some("numerical_selection_unresolved")
    );
    model.memory[0] += 1e-14;
    assert!(model.estimate().key.is_none());
    model.memory = PROFILES[0];
    assert_eq!(model.estimate().key, Some(0));
}

fn event(kind: ExecutionKind, tone: u64, sample: u64, sequence: u64) -> ExecutionRecord {
    ExecutionRecord {
        voice_id: 2,
        generation: 0,
        tone_id: tone,
        kind,
        sample,
        available_at: sample,
        execution_sequence: sequence,
        frequency_hz: Some(440.0),
        strength: None,
    }
}

#[test]
fn tonality_support_and_identity_recovery_are_local() {
    let mut t = OwnTonality::new(2, 0, 48000.0, 1.0);
    t.observe(&event(ExecutionKind::On, 1, 0, 0));
    t.observe(&event(ExecutionKind::On, 1, 0, 0));
    assert_eq!(t.epoch, 0);
    t.observe(&event(ExecutionKind::On, 2, 100, 1));
    assert!(t.snapshot(200, (0.0, 14.0)).is_none());
    t.observe(&event(ExecutionKind::Off, 1, 200, 2));
    t.observe(&event(ExecutionKind::Off, 2, 200, 3));
    t.observe(&event(ExecutionKind::On, 3, 300, 4));
    assert!(t.snapshot(400, (0.0, 14.0)).unwrap().estimate.key.is_some());
    t.reconfigure(false);
    t.reconfigure(true);
    assert!(t.snapshot(500, (0.0, 14.0)).is_none());
    t.observe(&event(ExecutionKind::Off, 3, 600, 5));
    t.observe(&event(ExecutionKind::On, 4, 700, 6));
    assert!(t.snapshot(800, (0.0, 14.0)).is_some());
    t.observe(&event(ExecutionKind::Missing, 4, 900, 7));
    t.observe(&event(ExecutionKind::Off, 4, 1000, 8));
    t.observe(&event(ExecutionKind::On, 5, 1100, 9));
    assert!(t.stopped && t.snapshot(1200, (0.0, 14.0)).is_none());
}

#[test]
fn tonality_encoding_uses_absolute_pitch_and_euclidean_modulo() {
    assert_eq!(pitch_class(440.0), Some(9));
    assert_eq!(pitch_class(880.0), Some(9));
    assert_eq!(
        pitch_class(440.0 * 2.0f64.powf((-1.0 - 69.0) / 12.0)),
        Some(11)
    );
    assert_eq!(pitch_class(f64::NAN), None);
    assert_eq!(pitch_class(0.0), None);
}

#[test]
fn tonality_matches_independent_decimal_reference_at_every_query() {
    let data: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/own_tonality_reference.json"
    ))
    .unwrap();
    assert_eq!(data["cases"].as_array().unwrap().len(), 12);
    let tolerance = data["absolute_tolerance"].as_f64().unwrap();
    let mut queries = 0;
    let mut worst = 0.0f64;
    for case in data["cases"].as_array().unwrap() {
        assert_eq!(case["queries"].as_array().unwrap().len(), 22);
        for query in case["queries"].as_array().unwrap() {
            let mut model = TonalityModel::default();
            let at = query["at"].as_u64().unwrap();
            let after = query["after"].as_bool().unwrap();
            for e in case["events"].as_array().unwrap() {
                let time = e["at"].as_u64().unwrap();
                if time > at || (time == at && !after) {
                    continue;
                }
                model.advance(time, 48000.0);
                model.active =
                    (e["kind"] == "on").then(|| (e["pc"].as_u64().unwrap() as usize, 0.0));
            }
            model.advance(at, 48000.0);
            for c in 0..12 {
                let actual_m = model.memory[c] * model.log_scale.exp();
                let actual_p = model.active.filter(|p| p.0 == c).map_or(0.0, |p| p.1.exp());
                for (actual, expected) in [
                    (actual_m, query["m"][c].as_f64().unwrap()),
                    (actual_p, query["p"][c].as_f64().unwrap()),
                ] {
                    worst = worst.max((actual - expected).abs());
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "shift={} at={at} after={after} c={c} actual={actual} expected={expected}",
                        case["shift"]
                    );
                }
            }
            let estimate = model.estimate();
            assert_eq!(
                estimate.correlations.is_none(),
                query["correlations"].is_null()
            );
            if let Some(correlations) = estimate.correlations {
                for (i, actual) in correlations.into_iter().enumerate() {
                    let error = (actual - query["correlations"][i].as_f64().unwrap()).abs();
                    worst = worst.max(error);
                    assert!(error <= tolerance && error <= estimate.error_bound);
                }
                assert_eq!(
                    estimate.key.unwrap(),
                    query["key"].as_u64().unwrap() as usize
                );
            }
            queries += 1;
        }
    }
    for vector in data["vectors"].as_array().unwrap() {
        let mut model = TonalityModel::default();
        for (i, m) in model.memory.iter_mut().enumerate() {
            *m = vector["memory"][i].as_f64().unwrap();
        }
        let estimate = model.estimate();
        assert_eq!(
            estimate.correlations.is_none(),
            vector["correlations"].is_null()
        );
        if let Some(r) = estimate.correlations {
            for (i, actual) in r.into_iter().enumerate() {
                assert!((actual - vector["correlations"][i].as_f64().unwrap()).abs() < tolerance);
            }
        }
    }
    println!(
        "TK Decimal oracle: queries={queries}, vectors=7, max_abs_error={worst:.17e}, tolerance={tolerance}"
    );
}

#[test]
fn tonality_capacity_and_unknown_identity_do_not_falsely_recover() {
    let mut t = OwnTonality::new(2, 0, 48000.0, 1.0);
    for i in 0..17 {
        t.observe(&event(ExecutionKind::On, i, i, i));
    }
    assert!(t.stopped);
    assert_eq!(t.reason, Some("active_identity_capacity"));
    let mut t = OwnTonality::new(2, 0, 48000.0, 1.0);
    let mut e = event(ExecutionKind::On, 1, 0, 0);
    e.generation = 1;
    t.observe(&e);
    assert!(t.stopped);
    assert_eq!(t.reason, Some("identity_mismatch"));
}
