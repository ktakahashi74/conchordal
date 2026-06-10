use conchordal::core::landscape::Landscape;
use conchordal::core::log2space::Log2Space;
use conchordal::core::timebase::{Tick, Timebase};
use conchordal::life::control::{Routing, VoiceControl};
use conchordal::life::generator_model::GeneratorModel;
use conchordal::life::population::Population;
use conchordal::life::schedule_renderer::ScheduleRenderer;
use conchordal::life::voice::VoiceMetadata;
use conchordal::scenario::{ArticulationCoreConfig, VoiceSpec};

fn test_timebase() -> Timebase {
    Timebase {
        fs: 48_000.0,
        hop: 256,
    }
}

fn spawn_voice_with_routing(routing: Routing) -> VoiceSpec {
    let mut control = VoiceControl::default();
    control.pitch.freq = 440.0;
    control.body.amp = 0.4;
    control.body.routing = routing;
    VoiceSpec {
        control,
        articulation: ArticulationCoreConfig::default(),
    }
}

fn render_one_frame(routing: Routing) -> (bool, bool) {
    let tb = test_timebase();
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = GeneratorModel::new(tb, space.clone());
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
    let habitat_active = frame.habitat.iter().any(|s| s.abs() > 1e-6);
    (presentation_active, habitat_active)
}

#[test]
fn default_routing_fills_both_buses() {
    let (presentation, habitat) = render_one_frame(Routing::default());
    assert!(presentation, "default routing must feed presentation bus");
    assert!(habitat, "default routing must feed habitat bus");
}

#[test]
fn habitat_only_suppresses_only_presentation_bus() {
    let routing = Routing {
        to_presentation: false,
        to_habitat: true,
    };
    let (presentation, habitat) = render_one_frame(routing);
    assert!(
        !presentation,
        "habitat-only voice must not leak into presentation bus"
    );
    assert!(
        habitat,
        "habitat-only voice must still contribute to habitat bus"
    );
}

#[test]
fn presentation_only_suppresses_only_habitat_bus() {
    let routing = Routing {
        to_presentation: true,
        to_habitat: false,
    };
    let (presentation, habitat) = render_one_frame(routing);
    assert!(
        presentation,
        "presentation-only voice must still reach presentation bus"
    );
    assert!(
        !habitat,
        "presentation-only voice must not leak into habitat bus"
    );
}

#[test]
fn both_flags_off_produces_silence_everywhere() {
    let routing = Routing {
        to_presentation: false,
        to_habitat: false,
    };
    let (presentation, habitat) = render_one_frame(routing);
    assert!(!presentation);
    assert!(!habitat);
}
