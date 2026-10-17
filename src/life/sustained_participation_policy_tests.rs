use super::*;
fn setup() -> TemporalParticipation {
    let mut p = TemporalParticipation::new(1000, 2.0, 0.8, 10_000, 21, 1000);
    p.due_frame = 10_000.0;
    p.set_decision_trace(true);
    p.set_onset_comparison(Some(crate::config::TemporalOnsetComparisonConfig::default()));
    p
}
#[test]
fn sustained_participation_existing_grid_terms_and_tie_are_preserved() {
    let mut old = setup();
    let mut added = setup();
    let a = old.candidate(10_000, 10_001, true, None, None);
    let b = added.candidate_with_adoption(10_000, 10_001, true, None, None, Some(10_000));
    assert_eq!(a, b);
    let a = old.drain_decisions().next().unwrap();
    let b = added.drain_decisions().next().unwrap();
    assert_eq!(
        serde_json::to_value(a.candidates).unwrap(),
        serde_json::to_value(b.candidates).unwrap()
    );
    assert_eq!(b.selected_offset, 0);
    assert!(!b.skipped);
}
#[test]
fn sustained_participation_committed_plan_cannot_be_replaced_and_illegal_extra_is_neutral() {
    let mut old = setup();
    let mut added = setup();
    assert_eq!(
        old.candidate(10_000, 10_001, true, None, None),
        added.candidate_with_adoption(10_000, 10_001, true, None, None, Some(9_999))
    );
    let plan = added.planned;
    added.candidate_with_adoption(10_000, 10_100, true, None, None, Some(10_001));
    assert_eq!(added.planned, plan);
}
#[test]
fn sustained_participation_offer_uses_the_same_overlap_evaluator() {
    let mut won = false;
    for delay in [1, 65, 129] {
        let mut old = setup();
        let mut added = setup();
        old.due_frame = 10_050.0;
        added.due_frame = 10_050.0;
        old.set_overlap(0.8, [1.0, 0.0, 0.0]);
        added.set_overlap(0.8, [1.0, 0.0, 0.0]);
        let forecast = TemporalForecast::energy_fixture(1000, 10_000, |sec| {
            if sec < delay as f64 / 1000.0 {
                [0.0; 3]
            } else {
                [10.0, 0.0, 0.0]
            }
        });
        old.candidate(10_000, 10_001, true, Some(&forecast), None);
        added.candidate_with_adoption(10_000, 10_001, true, Some(&forecast), None, Some(10_000));
        let a = old.drain_decisions().next().unwrap();
        let b = added.drain_decisions().next().unwrap();
        assert_eq!(
            serde_json::to_value(a.candidates).unwrap(),
            serde_json::to_value(b.candidates).unwrap()
        );
        won |= b.selected_offset == 21 && !b.skipped;
    }
    assert!(
        won,
        "an earlier adoption must avoid the rising acoustic overlap"
    );
}
