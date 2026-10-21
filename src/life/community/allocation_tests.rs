use super::tests::{runtime_landscape, spawn_spec_with_freq, test_pop};
use super::*;
use crate::scenario::{Action, ControlUpdateMode};

#[test]
fn community_steady_hops_reuse_respawn_social_and_sequential_storage() {
    let landscape = runtime_landscape();
    let mut community = test_pop();
    community.control_update_mode = ControlUpdateMode::SequentialRotating;
    community.apply_action(
        Action::Spawn {
            population_id: 7,
            ids: vec![70, 71, 72],
            spec: spawn_spec_with_freq(330.0),
            strategy: None,
        },
        &landscape,
    );
    for voice in &mut community.voices {
        voice.social_coupling = 0.5;
    }
    let mut model = GeneratorModel::new(community.time, landscape.space.clone());
    let mut batches = Vec::new();
    let dt = community.time.hop as f32 / community.time.fs;
    for frame in 0..4 {
        community.advance(64, frame, dt, &landscape);
        community.collect_phonation_batches_into(
            &mut model,
            &landscape,
            frame * 64,
            None,
            None,
            None,
            &mut batches,
        );
        community.cleanup_dead(frame, dt, false, &landscape);
    }
    crate::runtime_profile::begin_allocations();
    for frame in 4..104 {
        community.advance(64, frame, dt, &landscape);
        community.collect_phonation_batches_into(
            &mut model,
            &landscape,
            frame * 64,
            None,
            None,
            None,
            &mut batches,
        );
        community.cleanup_dead(frame, dt, false, &landscape);
    }
    let counts = crate::runtime_profile::finish_allocations().unwrap();
    assert_eq!((counts.count, counts.bytes), (0, 0));
    assert_eq!(community.voices.len(), 3);
    assert!(community.social_trace.is_some());
}

#[test]
fn loo_ownership_steady_state_allocates_nothing_and_keeps_quiet_other() {
    let landscape = runtime_landscape();
    let mut community = test_pop();
    community.apply_action(
        Action::Spawn {
            population_id: 7,
            ids: vec![70, 71, 72],
            spec: spawn_spec_with_freq(330.0),
            strategy: None,
        },
        &landscape,
    );
    for (i, voice) in community.voices.iter_mut().enumerate() {
        voice.last_signal = crate::life::voice::ArticulationSignal {
            amplitude: if i == 1 { 1. } else { 1e-15 },
            is_active: true,
            ..Default::default()
        };
    }
    community.update_loo_ownership(&landscape);
    let inputs: Vec<_> = community
        .voices
        .iter_mut()
        .map(|v| v.pitch_ctl.loo_input_mut().clone())
        .collect();
    for i in 0..inputs[0].own_power_scan.len() {
        assert_eq!(
            inputs[1].other_power_scan[i],
            inputs[0].own_power_scan[i] as f64 + inputs[2].own_power_scan[i] as f64
        );
    }
    assert!(inputs[1].other_power_scan.iter().any(|v| *v > 0.));
    for tone in [
        crate::scenario::FloraTone::Sine,
        crate::scenario::FloraTone::Resonant,
    ] {
        let mut flora = crate::life::flora::Flora::new(
            8,
            crate::scenario::FloraSpec {
                amp: 0.35,
                rate: 4.,
                renewal_sec: 10.,
                fade_sec: 0.,
                tone,
                residual: true,
                routing: Default::default(),
            },
            (220., 440.),
            crate::scenario::FloraSeeding::Consonance,
            42,
            &landscape,
            community.time.fs,
            community.time.hop,
        );
        flora.advance(&landscape);
        flora.render_add(
            &mut vec![0.; community.time.hop],
            &mut vec![0.; community.time.hop],
        );
        community.flora.push(flora);
    }
    community.update_loo_ownership(&landscape);
    crate::runtime_profile::begin_allocations();
    let start = std::time::Instant::now();
    for _ in 0..300 {
        community.update_loo_ownership(std::hint::black_box(&landscape));
    }
    let elapsed = start.elapsed();
    let counts = crate::runtime_profile::finish_allocations().unwrap();
    println!(
        "T36_OWNERSHIP_ALLOC counts={counts:?} iterations=300 voices=3 flora=2 elapsed={elapsed:?}"
    );
    assert_eq!((counts.count, counts.bytes), (0, 0));
}
