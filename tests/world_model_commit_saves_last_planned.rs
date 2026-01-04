use std::collections::HashMap;

use conchordal::core::log2space::Log2Space;
use conchordal::core::timebase::{Tick, Timebase};
use conchordal::life::plan::{GateTarget, PhaseRef, PlannedIntent};
use conchordal::life::world_model::WorldModel;

#[test]
fn commit_saves_last_planned_next() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = WorldModel::new(tb, space);
    let planned = PlannedIntent {
        source_id: 1,
        plan_id: 7,
        phase: PhaseRef {
            gate: GateTarget::Next,
            target_phase: 0.0,
        },
        duration: 64,
        freq_hz: 440.0,
        amp: 0.5,
        tag: Some("test".to_string()),
        confidence: 1.0,
        body: None,
    };
    world.plan_board.publish_replace(planned);

    let now: Tick = 0;
    let frame_end: Tick = tb.hop as Tick;
    world.next_gate_tick_est = Some(now);
    world.commit_plans_if_due(now, frame_end);

    assert!(world.plan_board.snapshot_next().is_empty());
    assert!(!world.last_committed_plans_next.is_empty());
    assert_eq!(world.last_committed_gate_tick, Some(now));

    let last_plans = world.last_committed_plans_next.clone();
    world.commit_plans_if_due(now, frame_end);
    assert_eq!(world.last_committed_gate_tick, Some(now));

    let to_map = |plans: Vec<PlannedIntent>| {
        let mut map = HashMap::new();
        for plan in plans {
            map.insert((plan.source_id, plan.plan_id), plan);
        }
        map
    };
    let before_map = to_map(last_plans);
    let after_map = to_map(world.last_committed_plans_next.clone());
    assert_eq!(after_map.len(), before_map.len());
    for (key, before) in before_map {
        let after = after_map.get(&key).expect("missing planned intent");
        assert_eq!(after.duration, before.duration);
        assert_eq!(after.tag, before.tag);
        assert!((after.freq_hz - before.freq_hz).abs() < 1e-6);
        assert!((after.amp - before.amp).abs() < 1e-6);
        assert!((after.confidence - before.confidence).abs() < 1e-6);
    }
}
