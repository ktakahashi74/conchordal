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
