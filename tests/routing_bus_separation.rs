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
    let listener_active = frame.listener.iter().any(|s| s.abs() > 1e-6);
    let perceptual_active = frame.perceptual.iter().any(|s| s.abs() > 1e-6);
    (listener_active, perceptual_active)
}

#[test]
fn default_routing_fills_both_buses() {
    let (listener, perceptual) = render_one_frame(Routing::default());
    assert!(listener, "default routing must feed listener bus");
    assert!(perceptual, "default routing must feed perceptual bus");
}

#[test]
fn mute_suppresses_only_listener_bus() {
    let routing = Routing {
        to_listener: false,
        to_voices: true,
    };
    let (listener, perceptual) = render_one_frame(routing);
    assert!(!listener, "muted voice must not leak into listener bus");
    assert!(
        perceptual,
        "muted voice must still contribute to perceptual bus"
    );
}

#[test]
fn unperceived_suppresses_only_perceptual_bus() {
    let routing = Routing {
        to_listener: true,
        to_voices: false,
    };
    let (listener, perceptual) = render_one_frame(routing);
    assert!(listener, "unperceived voice must still reach listener bus");
    assert!(
        !perceptual,
        "unperceived voice must not leak into perceptual bus"
    );
}

#[test]
fn both_flags_off_produces_silence_everywhere() {
    let routing = Routing {
        to_listener: false,
        to_voices: false,
    };
    let (listener, perceptual) = render_one_frame(routing);
    assert!(!listener);
    assert!(!perceptual);
}
