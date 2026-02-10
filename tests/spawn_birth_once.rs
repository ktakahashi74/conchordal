use conchordal::core::landscape::Landscape;
use conchordal::core::log2space::Log2Space;
use conchordal::core::timebase::{Tick, Timebase};
use conchordal::life::control::AgentControl;
use conchordal::life::individual::AgentMetadata;
use conchordal::life::population::Population;
use conchordal::life::scenario::{ArticulationCoreConfig, IndividualConfig};
use conchordal::life::schedule_renderer::ScheduleRenderer;
use conchordal::life::sound::{AudioCommand, BodyKind, VoiceTarget};
use conchordal::life::world_model::WorldModel;

fn test_timebase() -> Timebase {
    Timebase {
        fs: 48_000.0,
        hop: 64,
    }
}

fn spawn_agent(freq: f32, amp: f32) -> IndividualConfig {
    let mut control = AgentControl::default();
    control.pitch.freq = freq;
    control.body.amp = amp;
    IndividualConfig {
        control,
        articulation: ArticulationCoreConfig::default(),
    }
}

#[test]
fn spawn_does_not_publish_birth_note() {
    let tb = test_timebase();
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = WorldModel::new(tb, space.clone());
    let mut pop = Population::new(tb);

    let cfg = spawn_agent(440.0, 0.4);
    let assigned_id = 1;
    let meta = AgentMetadata {
        group_id: 0,
        member_idx: 0,
    };
    pop.add_individual(cfg.spawn(assigned_id, 0, meta, tb.fs, 0));

    let landscape = Landscape::new(space);
    let now: Tick = 0;
    let batches = pop.collect_phonation_batches(&mut world, &landscape, now);
    assert!(batches.is_empty());
}

#[test]
fn spawn_sounds_only_with_audio_trigger() {
    let tb = test_timebase();
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = WorldModel::new(tb, space.clone());
    let mut pop = Population::new(tb);

    let cfg = spawn_agent(440.0, 0.4);
    let assigned_id = 1;
    let meta = AgentMetadata {
        group_id: 0,
        member_idx: 0,
    };
    pop.add_individual(cfg.spawn(assigned_id, 0, meta, tb.fs, 0));

    let landscape = Landscape::new(space);
    let now: Tick = 0;
    let _ = pop.collect_phonation_batches(&mut world, &landscape, now);
    let mut voice_targets: Vec<VoiceTarget> = Vec::new();
    pop.fill_voice_targets(&mut voice_targets);

    let mut renderer = ScheduleRenderer::new(tb);
    let silent = renderer.render(&[], now, &landscape.rhythm, &voice_targets, &[]);
    assert!(silent.iter().all(|s| s.abs() <= 1e-6));

    let body = conchordal::life::sound::BodySnapshot {
        kind: BodyKind::Sine,
        amp_scale: 1.0,
        brightness: 0.0,
        noise_mix: 0.0,
    };
    let ensure = AudioCommand::EnsureVoice {
        id: assigned_id,
        body,
        pitch_hz: 440.0,
        amp: 0.4,
    };
    let ensured = renderer.render(&[], now, &landscape.rhythm, &voice_targets, &[ensure]);
    assert!(ensured.iter().all(|s| s.abs() <= 1e-6));

    let impulse = AudioCommand::Impulse {
        id: assigned_id,
        energy: 1.0,
    };
    let voiced = renderer.render(&[], now, &landscape.rhythm, &voice_targets, &[impulse]);
    assert!(voiced.iter().any(|s| s.abs() > 1e-6));
}
