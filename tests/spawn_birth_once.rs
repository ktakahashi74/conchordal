use conchordal::core::landscape::Landscape;
use conchordal::core::log2space::Log2Space;
use conchordal::core::timebase::{Tick, Timebase};
use conchordal::life::audio::{AudioCommand, AudioEvent, VoiceTarget};
use conchordal::life::control::AgentControl;
use conchordal::life::individual::AgentMetadata;
use conchordal::life::population::Population;
use conchordal::life::scenario::IndividualConfig;
use conchordal::life::schedule_renderer::ScheduleRenderer;
use conchordal::life::world_model::WorldModel;

fn test_timebase() -> Timebase {
    Timebase {
        fs: 48_000.0,
        hop: 64,
    }
}

fn spawn_agent(freq: f32, amp: f32) -> IndividualConfig {
    let mut control = AgentControl::default();
    control.pitch.center_hz = freq;
    control.body.amp = amp;
    IndividualConfig { control, tag: None }
}

#[test]
fn spawn_does_not_publish_birth_intent() {
    let tb = test_timebase();
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = WorldModel::new(tb, space.clone());
    let mut pop = Population::new(tb);

    let cfg = spawn_agent(440.0, 0.4);
    let meta = AgentMetadata {
        id: 1,
        tag: None,
        group_idx: 0,
        member_idx: 0,
    };
    pop.add_individual(cfg.spawn(1, 0, meta, tb.fs, 0));

    let landscape = Landscape::new(space);
    let now: Tick = 0;
    let batches = pop.publish_intents(&mut world, &landscape, now);
    assert!(batches.is_empty());
    assert!(world.board.is_empty());
}

#[test]
fn spawn_sounds_only_with_audio_trigger() {
    let tb = test_timebase();
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = WorldModel::new(tb, space.clone());
    let mut pop = Population::new(tb);

    let cfg = spawn_agent(440.0, 0.4);
    let meta = AgentMetadata {
        id: 1,
        tag: None,
        group_idx: 0,
        member_idx: 0,
    };
    pop.add_individual(cfg.spawn(1, 0, meta, tb.fs, 0));

    let landscape = Landscape::new(space);
    let now: Tick = 0;
    let _ = pop.publish_intents(&mut world, &landscape, now);
    let mut voice_targets: Vec<VoiceTarget> = Vec::new();
    pop.fill_voice_targets(&mut voice_targets);

    let mut renderer = ScheduleRenderer::new(tb);
    let silent = renderer.render(
        &world.board,
        &[],
        now,
        &landscape.rhythm,
        &voice_targets,
        &[],
    );
    assert!(silent.iter().all(|s| s.abs() <= 1e-6));

    let cmd = AudioCommand::Trigger {
        id: 1,
        ev: AudioEvent::Impulse { energy: 1.0 },
        body: None,
    };
    let voiced = renderer.render(
        &world.board,
        &[],
        now,
        &landscape.rhythm,
        &voice_targets,
        &[cmd],
    );
    assert!(voiced.iter().any(|s| s.abs() > 1e-6));
}
