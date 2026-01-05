use conchordal::core::landscape::Landscape;
use conchordal::core::log2space::Log2Space;
use conchordal::core::timebase::{Tick, Timebase};
use conchordal::life::individual::AgentMetadata;
use conchordal::life::intent::Intent;
use conchordal::life::scenario::{IndividualConfig, LifeConfig};

const AMP_EPS: f32 = 1e-6;

fn setup_agent(release_gain: f32) -> (conchordal::life::individual::Individual, Timebase) {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let life = LifeConfig::default();
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
    agent.release_gain = release_gain;
    (agent, tb)
}

fn run_plan(mut agent: conchordal::life::individual::Individual, tb: &Timebase) -> Vec<Intent> {
    let space = Log2Space::new(55.0, 8000.0, 96);
    let landscape = Landscape::new(space);
    let now: Tick = 0;
    let gate_tick: Tick = 0;
    let perc_tick: Tick = 0;
    let pred_eval_tick = None;
    let hop = tb.hop as usize;
    let intents: Vec<Intent> = Vec::new();
    let mut pred_c_scan_at = |_tick: Tick| None;
    agent.plan_intents(
        tb,
        now,
        gate_tick,
        perc_tick,
        pred_eval_tick,
        hop,
        &landscape,
        &intents,
        &mut pred_c_scan_at,
    )
}

#[test]
fn plan_intents_uses_release_gain_for_amp() {
    let (agent, tb) = setup_agent(0.25);
    let intents = run_plan(agent, &tb);
    assert!(!intents.is_empty());
    for intent in intents {
        assert!((intent.amp - 0.25).abs() < AMP_EPS);
    }
}

#[test]
fn plan_intents_skips_when_release_gain_zero() {
    let (agent, tb) = setup_agent(0.0);
    let intents = run_plan(agent, &tb);
    assert!(intents.is_empty());
}
