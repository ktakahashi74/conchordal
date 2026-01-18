use conchordal::core::landscape::LandscapeFrame;
use conchordal::core::log2space::Log2Space;
use conchordal::core::timebase::Timebase;
use conchordal::life::conductor::{Conductor, QueuedEvent};
use conchordal::life::population::Population;
use conchordal::life::scenario::Action;
use conchordal::life::world_model::WorldModel;

#[test]
fn conductor_routes_post_note_to_world() {
    let timebase = Timebase {
        fs: 48_000.0,
        hop: 512,
    };
    let space = Log2Space::new(55.0, 8000.0, 96);
    let landscape = LandscapeFrame::default();
    let mut pop = Population::new(timebase);
    let mut world = WorldModel::new(timebase, space);

    let event_time = 1.0_f32;
    let event = QueuedEvent {
        time: event_time,
        order: 0,
        actions: vec![Action::PostNote {
            source_id: 42,
            onset_sec: event_time,
            duration_sec: 0.25,
            freq_hz: 220.0,
            amp: 0.75,
            tag: None,
            confidence: 1.0,
        }],
    };
    let mut conductor = Conductor::from_events(vec![event]);

    let eps = 1.0e-3_f32;
    conductor.dispatch_until(
        event_time - eps,
        0,
        &landscape,
        None::<&mut conchordal::core::stream::roughness::RoughnessStream>,
        &mut pop,
        &mut world,
    );
    assert!(world.ui_view().notes.is_empty());

    conductor.dispatch_until(
        event_time,
        0,
        &landscape,
        None::<&mut conchordal::core::stream::roughness::RoughnessStream>,
        &mut pop,
        &mut world,
    );

    let view = world.ui_view();
    assert_eq!(view.notes.len(), 1);
    assert_eq!(view.notes[0].onset_tick, timebase.sec_to_tick(event_time));
    assert_eq!(view.notes[0].dur_tick, timebase.sec_to_tick(0.25));
}
