use crate::core::landscape::Landscape;
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::life::individual::AgentMetadata;
use crate::life::individual::{IndividualWrapper, SoundBody};
use crate::life::lifecycle::LifecycleConfig;
use crate::life::scenario::{BrainConfig, IndividualConfig};

fn make_landscape() -> Landscape {
    let space = Log2Space::new(55.0, 4000.0, 48);
    Landscape::new(space)
}

fn spawn_agent(freq: f32, id: u64) -> crate::life::individual::PureTone {
    let cfg = IndividualConfig::PureTone {
        freq,
        amp: 0.5,
        phase: None,
        rhythm_freq: None,
        rhythm_sensitivity: None,
        commitment: None,
        habituation_sensitivity: None,
        brain: BrainConfig::Entrain {
            lifecycle: LifecycleConfig::Decay {
                initial_energy: 1.0,
                half_life_sec: 0.5,
                attack_sec: crate::life::lifecycle::default_decay_attack(),
            },
        },
        tag: None,
    };
    let meta = AgentMetadata {
        id,
        tag: None,
        group_idx: 0,
        member_idx: 0,
    };
    match cfg.spawn(id, 0, meta) {
        IndividualWrapper::PureTone(ind) => ind,
        _ => unreachable!(),
    }
}

#[test]
fn test_inertia_calculation() {
    let landscape = make_landscape();
    let mut low = spawn_agent(60.0, 1);
    let mut high = spawn_agent(1000.0, 2);
    let rhythms = NeuralRhythms::default();

    low.update_organic_movement(&rhythms, 0.01, &landscape);
    high.update_organic_movement(&rhythms, 0.01, &landscape);

    assert!(
        low.integration_window > high.integration_window,
        "expected heavier (low) pitch to integrate longer than high pitch"
    );
}

#[test]
fn test_scan_logic() {
    let mut landscape = make_landscape();
    let mut agent = spawn_agent(220.0, 3);
    let n = landscape.consonance.len();
    landscape.subjective_intensity = vec![1.0; n];
    landscape.consonance.fill(-5.0);
    let idx_cur = landscape
        .space
        .index_of_freq(agent.body.base_freq_hz())
        .unwrap_or(0);
    landscape.consonance[idx_cur] = -5.0;
    let target_alt = agent.body.base_freq_hz() * 1.5;
    if let Some(idx_alt) = landscape.space.index_of_freq(target_alt) {
        if let Some(c) = landscape.consonance.get_mut(idx_alt) {
            *c = 5.0;
        }
    }

    agent.accumulated_time = 5.0;
    agent.last_theta_sample = -0.1;
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.mag = 1.0;
    rhythms.theta.phase = 0.25;

    let before = agent.target_freq;
    agent.update_organic_movement(&rhythms, 0.01, &landscape);
    assert!(
        agent.target_freq > before,
        "agent should move toward higher-scoring neighbor"
    );
}

#[test]
fn test_breath_gating() {
    let landscape = make_landscape();
    let mut agent = spawn_agent(330.0, 4);
    let original = agent.body.base_freq_hz();
    agent.target_freq = original * 1.5;
    agent.breath_gain = 1.0;

    let rhythms = NeuralRhythms::default();
    agent.update_organic_movement(&rhythms, 0.05, &landscape);

    assert!(
        agent.breath_gain < 1.0,
        "breath gain should fall before snapping to target"
    );
    let after_freq = agent.body.base_freq_hz();
    assert!(
        (after_freq - original).abs() < 1e-3,
        "frequency should remain until breath collapses ({} vs {})",
        after_freq,
        original
    );
}
