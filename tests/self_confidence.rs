use std::collections::VecDeque;

use conchordal::core::landscape::Landscape;
use conchordal::core::log2space::Log2Space;
use conchordal::core::timebase::Timebase;
use conchordal::life::individual::{AgentMetadata, PredIntentRecord};
use conchordal::life::scenario::{IndividualConfig, LifeConfig};
use conchordal::life::world_model::WorldModel;

#[test]
fn self_confidence_updates_once_and_removes_record() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = WorldModel::new(tb, space.clone());
    let agent_cfg = IndividualConfig {
        freq: 440.0,
        amp: 0.4,
        life: LifeConfig::default(),
        tag: None,
    };
    let metadata = AgentMetadata {
        id: 1,
        tag: None,
        group_idx: 0,
        member_idx: 0,
    };
    let mut agent = agent_cfg.spawn(1, 0, metadata, tb.fs, 0);
    agent.self_confidence = 1.0;

    let space = Log2Space::new(110.0, 880.0, 12);
    let mut landscape = Landscape::new(space.clone());
    let idx = space.index_of_freq(440.0).expect("bin");
    landscape.consonance.fill(0.0);
    landscape.consonance[idx] = 1.0;
    if idx + 1 < landscape.consonance.len() {
        landscape.consonance[idx + 1] = 1.0;
    }

    let record = PredIntentRecord {
        intent_id: 1,
        onset: 0,
        end: 20,
        freq_hz: space.freq_of_index(idx),
        pred_c_statepm1: 0.0,
        created_at: 0,
        eval_tick: 10,
    };
    agent.pred_intent_records = VecDeque::from([record]);

    agent.update_self_confidence_from_perc(&space, &landscape, 10);
    assert!(agent.pred_intent_records.is_empty());
    let expected = 0.975_f32;
    assert!((agent.self_confidence - expected).abs() < 1e-6);

    world.advance_to(0);
}

#[test]
fn record_is_dropped_after_end() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let agent_cfg = IndividualConfig {
        freq: 440.0,
        amp: 0.4,
        life: LifeConfig::default(),
        tag: None,
    };
    let metadata = AgentMetadata {
        id: 2,
        tag: None,
        group_idx: 0,
        member_idx: 0,
    };
    let mut agent = agent_cfg.spawn(2, 0, metadata, tb.fs, 0);
    agent.self_confidence = 0.5;

    let space = Log2Space::new(110.0, 880.0, 12);
    let landscape = Landscape::new(space.clone());

    let record = PredIntentRecord {
        intent_id: 2,
        onset: 0,
        end: 5,
        freq_hz: space.freq_of_index(0),
        pred_c_statepm1: 0.0,
        created_at: 0,
        eval_tick: 2,
    };
    agent.pred_intent_records = VecDeque::from([record]);

    agent.update_self_confidence_from_perc(&space, &landscape, 10);
    assert!(agent.pred_intent_records.is_empty());
}
