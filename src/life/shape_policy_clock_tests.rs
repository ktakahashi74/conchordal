use super::*;
use crate::life::shape_policy::ShapePolicy;

#[test]
fn articulation_persistence_intent_resolve_does_not_update_memory_last_onset_or_skip_cost() {
    let mut clock = TemporalParticipation::new(48000, 2.0, 0.8, 0, 21, 512);
    clock.register_initial_hold(0);
    let forecast = TemporalForecast::energy_fixture(48000, 0, |_| [0.1, 0.2, 0.3]);
    let mut shape = ShapePolicy::new(0.875, 21, 48000.0);
    shape.reserve(16, 512);
    let tick = clock
        .candidate_with_shape(
            0,
            48000,
            true,
            Some(&forecast),
            None,
            None,
            Some(&mut shape),
        )
        .unwrap();
    assert!(clock.planned.unwrap().1.is_some());
    let due = clock.planned.unwrap().0 + clock.period_frames;
    clock.resolve(tick, false);
    assert!(clock.pending_contexts.is_empty());
    assert!(clock.memory.is_none());
    assert_eq!(clock.last_onset, Some(0));
    assert_eq!(clock.skipped_cycles, 0);
    assert_eq!(clock.due_frame.to_bits(), due.to_bits());
    assert!(clock.planned.is_none());
}

#[test]
fn articulation_persistence_future_plan_is_reused_without_new_shape_decisions() {
    let mut clock = TemporalParticipation::new(48000, 2.0, 0.8, 0, 21, 512);
    clock.due_frame = 24000.0;
    let mut shape = ShapePolicy::new(0.75, 21, 48000.0);
    shape.reserve(16, 512);
    assert!(
        clock
            .candidate_with_shape(22000, 22500, true, None, None, None, Some(&mut shape))
            .is_none()
    );
    assert!(clock.planned.is_some());
    let state = shape.state;
    let pending = clock.planned;
    for now in 22500..23999 {
        assert!(
            clock
                .candidate_with_shape(now, now + 1, true, None, None, None, Some(&mut shape))
                .is_none()
        );
        assert_eq!(shape.state, state);
        assert_eq!(clock.planned, pending);
    }
    assert_eq!(
        clock.candidate_with_shape(24000, 24001, true, None, None, None, Some(&mut shape)),
        Some(24000)
    );
    clock.resolve(24000, true);
    assert_eq!(clock.last_onset, Some(24000));
}
