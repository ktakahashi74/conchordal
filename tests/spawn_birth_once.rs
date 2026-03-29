use conchordal::core::landscape::Landscape;
use conchordal::core::log2space::Log2Space;
use conchordal::core::timebase::{Tick, Timebase};
use conchordal::life::control::VoiceControl;
use conchordal::life::population::Population;
use conchordal::life::scenario::{ArticulationCoreConfig, VoiceConfig};
use conchordal::life::schedule_renderer::ScheduleRenderer;
use conchordal::life::voice::VoiceMetadata;
use conchordal::life::world_model::WorldModel;

fn test_timebase() -> Timebase {
    Timebase {
        fs: 48_000.0,
        hop: 64,
    }
}

fn spawn_agent(freq: f32, amp: f32) -> VoiceConfig {
    let mut control = VoiceControl::default();
    control.pitch.freq = freq;
    control.body.amp = amp;
    VoiceConfig {
        control,
        articulation: ArticulationCoreConfig::default(),
    }
}

#[test]
fn spawn_sustain_publishes_note_on_first_tick() {
    let tb = test_timebase();
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = WorldModel::new(tb, space.clone());
    let mut pop = Population::new(tb);

    let cfg = spawn_agent(440.0, 0.4);
    let assigned_id = 1;
    let meta = VoiceMetadata {
        group_id: 0,
        member_idx: 0,
    };
    pop.add_voice(cfg.spawn(assigned_id, 0, meta, tb.fs, 0));

    let landscape = Landscape::new(space);
    let now: Tick = 0;
    let batches = pop.collect_phonation_batches(&mut world, &landscape, now);
    assert!(!batches.is_empty());
}

#[test]
fn spawn_emits_phonation_note_that_renders_audio() {
    let tb = test_timebase();
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = WorldModel::new(tb, space.clone());
    let mut pop = Population::new(tb);

    let cfg = spawn_agent(440.0, 0.4);
    let assigned_id = 1;
    let meta = VoiceMetadata {
        group_id: 0,
        member_idx: 0,
    };
    pop.add_voice(cfg.spawn(assigned_id, 0, meta, tb.fs, 0));

    let landscape = Landscape::new(space);
    let now: Tick = 0;
    let batches = pop.collect_phonation_batches(&mut world, &landscape, now);

    let mut renderer = ScheduleRenderer::new(tb);
    let out = renderer.render(&batches, now, &landscape.rhythm);
    // Phonation note should produce audio (Hold mode fires NoteOn on first tick)
    assert!(out.iter().any(|s| s.abs() > 1e-6));
}
