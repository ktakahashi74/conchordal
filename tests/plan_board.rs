use conchordal::life::plan::{GateTarget, PhaseRef, PlanBoard, PlannedIntent};

fn make_plan(source_id: u64, plan_id: u64, freq_hz: f32) -> PlannedIntent {
    PlannedIntent {
        source_id,
        plan_id,
        phase: PhaseRef {
            gate: GateTarget::Next,
            target_phase: 0.0,
        },
        duration: 120,
        freq_hz,
        amp: 0.5,
        tag: None,
        confidence: 0.7,
        body: None,
    }
}

#[test]
fn publish_replace_keeps_one_and_last_wins() {
    let mut board = PlanBoard::new();
    let p1 = make_plan(1, 10, 1.0);
    let p2 = make_plan(1, 11, 2.0);
    board.publish_replace(p1);
    board.publish_replace(p2.clone());

    let snapshot = board.snapshot_next();
    assert_eq!(snapshot.len(), 1);
    assert_eq!(snapshot[0].source_id, 1);
    assert_eq!(snapshot[0].plan_id, p2.plan_id);
    assert_eq!(snapshot[0].freq_hz, p2.freq_hz);
}

#[test]
fn snapshot_next_is_stable_order() {
    let mut board = PlanBoard::new();
    board.publish_replace(make_plan(2, 20, 2.0));
    board.publish_replace(make_plan(1, 10, 1.0));

    let snapshot = board.snapshot_next();
    assert_eq!(snapshot.len(), 2);
    assert_eq!(snapshot[0].source_id, 1);
    assert_eq!(snapshot[1].source_id, 2);
}

#[test]
fn remove_and_clear_work() {
    let mut board = PlanBoard::new();
    board.publish_replace(make_plan(1, 10, 1.0));
    board.publish_replace(make_plan(2, 20, 2.0));

    board.remove_source(1);
    let snapshot = board.snapshot_next();
    assert_eq!(snapshot.len(), 1);
    assert_eq!(snapshot[0].source_id, 2);

    board.clear_next();
    assert!(board.snapshot_next().is_empty());
}
