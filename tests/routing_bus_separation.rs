use conchordal::core::landscape::Landscape;
use conchordal::core::log2space::Log2Space;
use conchordal::core::timebase::{Tick, Timebase};
use conchordal::life::control::{Routing, VoiceControl};
use conchordal::life::population::Population;
use conchordal::life::scenario::{ArticulationCoreConfig, VoiceConfig};
use conchordal::life::schedule_renderer::ScheduleRenderer;
use conchordal::life::voice::VoiceMetadata;
use conchordal::life::world_model::WorldModel;

fn test_timebase() -> Timebase {
    Timebase {
        fs: 48_000.0,
        hop: 256,
    }
}

fn spawn_voice_with_routing(routing: Routing) -> VoiceConfig {
    let mut control = VoiceControl::default();
    control.pitch.freq = 440.0;
    control.body.amp = 0.4;
    control.body.routing = routing;
    VoiceConfig {
        control,
        articulation: ArticulationCoreConfig::default(),
    }
}

fn render_one_frame(routing: Routing) -> (bool, bool) {
    let tb = test_timebase();
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = WorldModel::new(tb, space.clone());
    let mut pop = Population::new(tb);

    let cfg = spawn_voice_with_routing(routing);
    let meta = VoiceMetadata {
        group_id: 0,
        member_idx: 0,
        generation: 0,
        parent_id: None,
    };
    pop.add_voice(cfg.spawn(1, 0, meta, tb.fs, 0));

    let landscape = Landscape::new(space);
    let now: Tick = 0;
    let batches = pop.collect_phonation_batches(&mut world, &landscape, now);

    let mut renderer = ScheduleRenderer::new(tb);
    let frame = renderer.render(&batches, now, &landscape.rhythm);
    let presentation_active = frame.presentation.iter().any(|s| s.abs() > 1e-6);
    let field_active = frame.field.iter().any(|s| s.abs() > 1e-6);
    (presentation_active, field_active)
}

#[test]
fn default_routing_fills_both_buses() {
    let (presentation, field) = render_one_frame(Routing::default());
    assert!(presentation, "default routing must feed presentation bus");
    assert!(field, "default routing must feed field bus");
}

#[test]
fn field_only_suppresses_only_presentation_bus() {
    let routing = Routing {
        to_presentation: false,
        to_field: true,
    };
    let (presentation, field) = render_one_frame(routing);
    assert!(
        !presentation,
        "field-only voice must not leak into presentation bus"
    );
    assert!(field, "field-only voice must still contribute to field bus");
}

#[test]
fn presentation_only_suppresses_only_field_bus() {
    let routing = Routing {
        to_presentation: true,
        to_field: false,
    };
    let (presentation, field) = render_one_frame(routing);
    assert!(
        presentation,
        "presentation-only voice must still reach presentation bus"
    );
    assert!(
        !field,
        "presentation-only voice must not leak into field bus"
    );
}

#[test]
fn both_flags_off_produces_silence_everywhere() {
    let routing = Routing {
        to_presentation: false,
        to_field: false,
    };
    let (presentation, field) = render_one_frame(routing);
    assert!(!presentation);
    assert!(!field);
}
