use conchordal::core::log2space::Log2Space;
use conchordal::core::timebase::Timebase;
use conchordal::life::scenario::Action;
use conchordal::life::world_model::WorldModel;

#[test]
fn post_intent_applies_to_world_model() {
    let timebase = Timebase {
        fs: 48_000.0,
        hop: 512,
    };
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = WorldModel::new(timebase, space);
    let action = Action::PostIntent {
        source_id: 7,
        onset_sec: 1.0,
        duration_sec: 0.5,
        freq_hz: 440.0,
        amp: 0.5,
        tag: Some("test".to_string()),
        confidence: 0.8,
    };
    world.apply_action(&action);

    let view = world.ui_view();
    assert_eq!(view.intents.len(), 1);
    assert_eq!(view.intents[0].onset_tick, timebase.sec_to_tick(1.0));
    assert_eq!(view.intents[0].dur_tick, timebase.sec_to_tick(0.5));
}
