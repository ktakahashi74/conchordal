use conchordal::core::landscape::Landscape;
use conchordal::core::log2space::Log2Space;
use conchordal::core::modulation::NeuralRhythms;
use conchordal::core::timebase::{Tick, Timebase};
use conchordal::life::individual::AgentMetadata;
use conchordal::life::population::Population;
use conchordal::life::scenario::{IndividualConfig, LifeConfig};
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
    life.planning.plan_rate = 1.0;
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
    let agent = agent_cfg.spawn(1, 0, metadata, tb.fs, 0);
    pop.add_individual(agent);

    let landscape = Landscape::new(space.clone());

    world.advance_to(0);
    world.next_gate_tick_est = Some(0);
    pop.publish_intents(&mut world, &landscape, 0, 0);
    assert!(!world.plan_board.snapshot_next().is_empty());
    world.next_gate_tick_est = Some(1);
    let frame_end: Tick = tb.hop as Tick;
    world.commit_plans_if_due(0, frame_end);
    assert!(world.board.len() > 0);

    let first_intent = world
        .board
        .query_range(0..u64::MAX)
        .next()
        .expect("expected intent");
    let render_start = first_intent.onset.saturating_sub(1);

    let mut renderer = ScheduleRenderer::new(tb);
    let rhythms = NeuralRhythms::default();
    let end = world
        .board
        .query_range(0..u64::MAX)
        .map(|i| i.onset.saturating_add(i.duration))
        .max()
        .unwrap_or(render_start);
    let mut out = Vec::new();
    let mut tick = render_start;
    while tick <= end {
        out.extend_from_slice(renderer.render(&world.board, tick, &rhythms));
        tick = tick.saturating_add(tb.hop as u64);
    }
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
    life.planning.plan_rate = 1.0;
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
    let agent = agent_cfg.spawn(1, 0, metadata, tb.fs, 0);
    pop.add_individual(agent);

    let space = Log2Space::new(20.0, 20_000.0, 24);
    let landscape = Landscape::new(space);

    let now: Tick = 0;
    world.advance_to(now);
    world.next_gate_tick_est = Some(now + (tb.hop as Tick / 2));
    pop.publish_intents(&mut world, &landscape, now, now);
    assert!(!world.plan_board.snapshot_next().is_empty());

    world.plan_board.clear_next();
    world.last_committed_gate_tick = None;
    world.next_gate_tick_est = Some(now + tb.hop as Tick);
    pop.publish_intents(&mut world, &landscape, now, now);
    assert!(world.plan_board.snapshot_next().is_empty());
}
