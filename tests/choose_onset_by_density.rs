use conchordal::core::timebase::Tick;
use conchordal::life::intent::Intent;
use conchordal::life::intent_planner::choose_onset_by_density;

fn make_intent(onset: Tick) -> Intent {
    Intent {
        source_id: 0,
        intent_id: onset,
        onset,
        duration: 10,
        freq_hz: 440.0,
        amp: 0.2,
        tag: None,
        confidence: 1.0,
        body: None,
    }
}

#[test]
fn picks_highest_density_candidate() {
    let candidates = vec![100, 200, 300];
    let intents = vec![make_intent(198), make_intent(205), make_intent(95)];
    let chosen = choose_onset_by_density(&candidates, &intents, 10).unwrap();
    assert_eq!(chosen, 200);
}

#[test]
fn ties_choose_earliest_candidate() {
    let candidates = vec![100, 200, 300];
    let intents = vec![make_intent(95), make_intent(205)];
    let chosen = choose_onset_by_density(&candidates, &intents, 10).unwrap();
    assert_eq!(chosen, 100);
}
