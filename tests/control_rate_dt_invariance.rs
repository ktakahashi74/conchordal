use conchordal::core::landscape::Landscape;
use conchordal::core::log2space::Log2Space;
use conchordal::core::modulation::{NeuralRhythms, RhythmBand};
use conchordal::life::individual::{
    AgentMetadata, AnyArticulationCore, ArticulationState, Individual,
};
use conchordal::life::scenario::{IndividualConfig, LifeConfig};

fn build_agent() -> Individual {
    let life = LifeConfig::default();
    let cfg = IndividualConfig {
        freq: 440.0,
        amp: 0.3,
        life,
        tag: None,
    };
    let metadata = AgentMetadata {
        id: 1,
        tag: None,
        group_idx: 0,
        member_idx: 0,
    };
    cfg.spawn(1, 0, metadata, 48_000.0, 0)
}

fn prepare_agent(mut agent: Individual) -> Individual {
    agent.articulation.set_gate(0.2);
    if let AnyArticulationCore::Entrain(core) = &mut agent.articulation.core {
        core.base_sigma = 0.0;
        core.beta_gain = 0.0;
        core.retrigger = false;
        core.state = ArticulationState::Decay;
        core.env_level = 0.7;
        core.energy = 0.9;
        core.bootstrap_timer = 0.0;
        core.rhythm_phase = 0.2;
    } else {
        panic!("expected entrain articulation core");
    }
    agent
}

#[test]
fn control_rate_dt_invariance() {
    let space = Log2Space::new(55.0, 8000.0, 96);
    let landscape = Landscape::new(space);
    let rhythms = NeuralRhythms {
        theta: RhythmBand {
            phase: 0.1,
            freq_hz: 6.0,
            mag: 1.0,
            alpha: 1.0,
            beta: 0.0,
        },
        delta: RhythmBand {
            phase: 0.0,
            freq_hz: 1.0,
            mag: 1.0,
            alpha: 1.0,
            beta: 0.0,
        },
        env_open: 1.0,
        env_level: 1.0,
    };

    let mut fine = prepare_agent(build_agent());
    let mut coarse = prepare_agent(build_agent());

    let total_sec = 0.05;
    let steps = 10;
    let dt = total_sec / steps as f32;
    for _ in 0..steps {
        fine.update_articulation(dt, &rhythms, &landscape, 1.0);
    }
    coarse.update_articulation(total_sec, &rhythms, &landscape, 1.0);

    let tol = 1e-2;
    let gate_fine = fine.articulation.gate();
    let gate_coarse = coarse.articulation.gate();
    assert!(
        (gate_fine - gate_coarse).abs() <= tol,
        "gate drift fine={gate_fine:.6} coarse={gate_coarse:.6}"
    );

    let (energy_fine, env_fine) = match &fine.articulation.core {
        AnyArticulationCore::Entrain(core) => (core.energy, core.env_level),
        _ => panic!("expected entrain articulation core"),
    };
    let (energy_coarse, env_coarse) = match &coarse.articulation.core {
        AnyArticulationCore::Entrain(core) => (core.energy, core.env_level),
        _ => panic!("expected entrain articulation core"),
    };

    assert!(
        (energy_fine - energy_coarse).abs() <= tol,
        "energy drift fine={energy_fine:.6} coarse={energy_coarse:.6}"
    );
    assert!(
        (env_fine - env_coarse).abs() <= tol,
        "env_level drift fine={env_fine:.6} coarse={env_coarse:.6}"
    );
}
