use std::sync::Arc;

use conchordal::core::landscape::Landscape;
use conchordal::core::log2space::Log2Space;
use conchordal::core::timebase::{Tick, Timebase};
use conchordal::life::individual::AgentMetadata;
use conchordal::life::intent::Intent;
use conchordal::life::scenario::{IndividualConfig, LifeConfig, PlanPitchMode};

#[test]
fn pred_c_eval_tick_uses_gate_tick() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let space = Log2Space::new(55.0, 8000.0, 96);
    let landscape = Landscape::new(space.clone());
    let mut life = LifeConfig::default();
    life.planning.pitch_mode = PlanPitchMode::PredC;
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
    let mut agent = agent_cfg.spawn(1, 0, metadata, tb.fs, 0);

    let now: Tick = 0;
    let gate_tick: Tick = 32;
    let perc_tick: Tick = 0;
    let pred_eval_tick = Some(gate_tick);
    let hop = tb.hop as usize;
    let intents: Vec<Intent> = Vec::new();

    let mut called = false;
    let mut pred_c_scan_at = |tick: Tick| {
        assert_eq!(tick, gate_tick);
        called = true;
        let mut scan = vec![0.0; space.n_bins()];
        if !scan.is_empty() {
            let idx = scan.len() / 2;
            scan[idx] = 1.0;
        }
        Some(Arc::from(scan))
    };

    let out = agent.plan_intents(
        &tb,
        now,
        gate_tick,
        perc_tick,
        pred_eval_tick,
        hop,
        &landscape,
        &intents,
        &mut pred_c_scan_at,
    );

    assert!(called);
    assert!(!out.is_empty());
    assert_eq!(out[0].onset, gate_tick);
}
