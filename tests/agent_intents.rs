use conchordal::core::landscape::Landscape;
use conchordal::core::log2space::Log2Space;
use conchordal::core::timebase::{Tick, Timebase};
use conchordal::life::individual::AgentMetadata;
use conchordal::life::population::Population;
use conchordal::life::scenario::{IndividualConfig, LifeConfig, PhonationIntervalConfig};
use conchordal::life::schedule_renderer::ScheduleRenderer;
use conchordal::life::world_model::WorldModel;

#[test]
fn agents_publish_intents_and_render_audio() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = WorldModel::new(tb, space.clone());
    let mut pop = Population::new(tb);
    let mut life = LifeConfig::default();
    life.phonation.interval = PhonationIntervalConfig::Accumulator {
        rate: 1.0,
        refractory: 0,
    };
    let agent_cfg = IndividualConfig {
        freq: 440.0,
        amp: 0.4,
        life,
        tag: None,
    };
    let metadata = AgentMetadata {
        id: 1,
        tag: None,
        group_idx: 0,
        member_idx: 0,
    };
    let agent = agent_cfg.spawn(1, 0, metadata.clone(), tb.fs, 0);
    pop.add_individual(agent);

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
        let batches = pop.publish_intents(&mut world, &landscape, now);
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
    let out = renderer.render(&world.board, &phonation_batches, render_now, &rhythms);
    assert!(out.iter().any(|s| s.abs() > 1e-6));
}

#[test]
fn publish_intents_runs_when_gate_in_hop_window() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = WorldModel::new(tb, space.clone());
    let mut pop = Population::new(tb);
    let mut life = LifeConfig::default();
    life.phonation.interval = PhonationIntervalConfig::Accumulator {
        rate: 1.0,
        refractory: 0,
    };
    let agent_cfg = IndividualConfig {
        freq: 440.0,
        amp: 0.4,
        life,
        tag: None,
    };
    let metadata = AgentMetadata {
        id: 1,
        tag: None,
        group_idx: 0,
        member_idx: 0,
    };
    let agent = agent_cfg.spawn(1, 0, metadata.clone(), tb.fs, 0);
    pop.add_individual(agent);

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
        let next = pop.publish_intents(&mut world, &landscape, now);
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
    let agent = agent_cfg.spawn(1, 0, metadata, tb.fs, 0);
    pop_off.add_individual(agent);
    let batches_off = pop_off.publish_intents(&mut world_off, &landscape_off, now);
    assert!(batches_off.is_empty());
}
