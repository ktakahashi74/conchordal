use conchordal::core::landscape::Landscape;
use conchordal::core::log2space::Log2Space;
use conchordal::core::timebase::{Tick, Timebase};
use conchordal::life::individual::AgentMetadata;
use conchordal::life::population::Population;
use conchordal::life::scenario::{IndividualConfig, LifeConfig};
use conchordal::life::schedule_renderer::ScheduleRenderer;
use conchordal::life::sound_voice::default_release_ticks;
use conchordal::life::world_model::WorldModel;

fn test_timebase() -> Timebase {
    Timebase {
        fs: 48_000.0,
        hop: 64,
    }
}

fn spawn_agent(freq: f32, amp: f32) -> IndividualConfig {
    IndividualConfig {
        freq,
        amp,
        life: LifeConfig::default(),
        tag: None,
    }
}

#[test]
fn spawn_birth_once_intent_is_single_and_immediate() {
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

    let intents = world
        .board
        .snapshot(now, tb.sec_to_tick(1.0), tb.sec_to_tick(1.0));
    assert_eq!(intents.len(), 1);
    let intent = &intents[0];
    assert_eq!(intent.onset, now);
    assert_eq!(
        intent.duration,
        tb.sec_to_tick(Population::BIRTH_ONCE_DURATION_SEC)
    );

    let next = now.saturating_add(tb.hop as Tick);
    let batches_next = pop.publish_intents(&mut world, &landscape, next);
    assert!(batches_next.is_empty());
    assert_eq!(world.board.len(), 1);
}

#[test]
fn birth_once_releases_to_silence_and_stops() {
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
    let intents = world
        .board
        .snapshot(now, tb.sec_to_tick(1.0), tb.sec_to_tick(1.0));
    let intent = intents.first().expect("birth intent exists");

    let mut renderer = ScheduleRenderer::new(tb);
    let rendered = renderer.render(&world.board, &[], now, &landscape.rhythm);
    assert!(rendered.iter().any(|s| s.abs() > 1e-6));

    let release_end = intent
        .onset
        .saturating_add(intent.duration)
        .saturating_add(default_release_ticks(tb));
    let late_now = release_end.saturating_add(tb.hop as Tick);
    let rendered_late = renderer.render(&world.board, &[], late_now, &landscape.rhythm);
    assert!(rendered_late.iter().all(|s| s.abs() <= 1e-6));
    assert!(renderer.is_idle());
}

#[test]
fn spawn_defaults_do_not_retrigger_over_time() {
    let tb = test_timebase();
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = WorldModel::new(tb, space.clone());
    let mut pop = Population::new(tb);

    for id in 0..3 {
        let cfg = spawn_agent(220.0 + id as f32 * 110.0, 0.3);
        let meta = AgentMetadata {
            id: id + 1,
            tag: None,
            group_idx: 0,
            member_idx: id as usize,
        };
        pop.add_individual(cfg.spawn(id + 1, 0, meta, tb.fs, 0));
    }

    let landscape = Landscape::new(space);
    let hop = tb.hop as Tick;
    let mut now: Tick = 0;
    let mut intent_counts = Vec::new();
    let steps = (tb.sec_to_tick(2.0) / hop).max(1);
    for _ in 0..steps {
        let batches = pop.publish_intents(&mut world, &landscape, now);
        assert!(batches.is_empty());
        intent_counts.push(world.board.len());
        now = now.saturating_add(hop);
    }

    assert_eq!(intent_counts[0], 3);
    assert!(intent_counts.iter().all(|&count| count == 3));
}
