use conchordal::core::landscape::Landscape;
use conchordal::core::log2space::Log2Space;
use conchordal::core::timebase::{Tick, Timebase};
use conchordal::life::control::VoiceControl;
use conchordal::life::population::Population;
use conchordal::life::scenario::{
    ArticulationCoreConfig, DurationSpec, PhonationSpec, VoiceConfig, WhenSpec,
};
use conchordal::life::schedule_renderer::ScheduleRenderer;
use conchordal::life::voice::VoiceMetadata;
use conchordal::life::world_model::WorldModel;

#[test]
fn agents_publish_notes_and_render_audio() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = WorldModel::new(tb, space.clone());
    let mut pop = Population::new(tb);
    let mut control = VoiceControl::default();
    control.pitch.freq = 440.0;
    control.body.amp = 0.4;
    control.phonation.spec = PhonationSpec {
        when: WhenSpec::Pulse {
            rate: 3.3,
            sync: 0.0,
            social: 0.0,
        },
        duration: DurationSpec::Gates(5),
    };
    let agent_cfg = VoiceConfig {
        control,
        articulation: ArticulationCoreConfig::default(),
    };
    let assigned_id = 1;
    let metadata = VoiceMetadata {
        group_id: 0,
        member_idx: 0,
        generation: 0,
        parent_id: None,
    };
    let agent = agent_cfg.spawn(assigned_id, 0, metadata.clone(), tb.fs, 0);
    pop.add_voice(agent);

    let mut landscape = Landscape::new(space.clone());
    landscape.rhythm.theta.freq_hz = 6.0;
    landscape.rhythm.theta.phase = -0.01;
    landscape.rhythm.env_open = 1.0;
    landscape.rhythm.env_level = 1.0;

    let hop = tb.hop as Tick;
    let mut phonation_batches = Vec::new();
    let mut render_now: Tick = 0;
    let mut now: Tick = 0;
    for _ in 0..300 {
        let batches = pop.collect_phonation_batches(&mut world, &landscape, now);
        if !batches.is_empty() {
            phonation_batches = batches;
            render_now = now;
            break;
        }
        now = now.saturating_add(hop);
    }
    assert!(!phonation_batches.is_empty());
    let mut renderer = ScheduleRenderer::new(tb);
    let rhythms = landscape.rhythm;
    let out = renderer.render(&phonation_batches, render_now, &rhythms);
    assert!(out.iter().any(|s| s.abs() > 1e-6));
}

#[test]
fn publish_notes_runs_when_gate_in_hop_window() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = WorldModel::new(tb, space.clone());
    let mut pop = Population::new(tb);
    let mut control = VoiceControl::default();
    control.pitch.freq = 440.0;
    control.body.amp = 0.4;
    control.phonation.spec = PhonationSpec {
        when: WhenSpec::Pulse {
            rate: 3.3,
            sync: 0.0,
            social: 0.0,
        },
        duration: DurationSpec::Gates(5),
    };
    let agent_cfg = VoiceConfig {
        control,
        articulation: ArticulationCoreConfig::default(),
    };
    let assigned_id = 1;
    let metadata = VoiceMetadata {
        group_id: 0,
        member_idx: 0,
        generation: 0,
        parent_id: None,
    };
    let agent = agent_cfg.spawn(assigned_id, 0, metadata.clone(), tb.fs, 0);
    pop.add_voice(agent);

    let space = Log2Space::new(20.0, 20_000.0, 24);
    let mut landscape = Landscape::new(space.clone());
    landscape.rhythm.theta.freq_hz = 6.0;
    landscape.rhythm.theta.phase = -0.01;
    landscape.rhythm.env_open = 1.0;
    landscape.rhythm.env_level = 1.0;

    let hop = tb.hop as Tick;
    let mut now: Tick = 0;
    let mut batches = Vec::new();
    for _ in 0..300 {
        let next = pop.collect_phonation_batches(&mut world, &landscape, now);
        if !next.is_empty() {
            batches = next;
            break;
        }
        now = now.saturating_add(hop);
    }
    assert!(!batches.is_empty());

    let mut landscape_off = Landscape::new(space);
    landscape_off.rhythm.theta.freq_hz = 1.0;
    landscape_off.rhythm.theta.phase = 0.0;
    landscape_off.rhythm.env_open = 1.0;
    landscape_off.rhythm.env_level = 1.0;
    let mut world_off = WorldModel::new(tb, Log2Space::new(20.0, 20_000.0, 24));
    let mut pop_off = Population::new(tb);
    let agent = agent_cfg.spawn(assigned_id, 0, metadata, tb.fs, 0);
    pop_off.add_voice(agent);
    let batches_off = pop_off.collect_phonation_batches(&mut world_off, &landscape_off, now);
    assert!(batches_off.is_empty());
}
