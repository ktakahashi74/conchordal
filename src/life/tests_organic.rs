use crate::core::landscape::Landscape;
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::life::individual::{AgentMetadata, Individual, SoundBody};
use crate::life::lifecycle::LifecycleConfig;
use crate::life::perceptual::PerceptualConfig;
use crate::life::population::Population;
use crate::life::scenario::Action;
use crate::life::scenario::{
    FieldCoreConfig, IndividualConfig, LifeConfig, ModulationCoreConfig, SoundBodyConfig,
    TemporalCoreConfig,
};

fn make_landscape() -> Landscape {
    let space = Log2Space::new(55.0, 4000.0, 48);
    Landscape::new(space)
}

fn life_with_lifecycle(lifecycle: LifecycleConfig) -> LifeConfig {
    LifeConfig {
        body: SoundBodyConfig::Sine { phase: None },
        temporal: TemporalCoreConfig::Entrain {
            lifecycle,
            rhythm_freq: None,
            rhythm_sensitivity: None,
        },
        field: FieldCoreConfig::PitchHillClimb {
            neighbor_step_cents: None,
            tessitura_gravity: None,
            improvement_threshold: None,
        },
        modulation: ModulationCoreConfig::Static {
            exploration: None,
            persistence: None,
        },
        perceptual: PerceptualConfig {
            tau_fast: None,
            tau_slow: None,
            w_boredom: None,
            w_familiarity: None,
            rho_self: None,
            boredom_gamma: None,
            self_smoothing_radius: None,
            silence_mass_epsilon: None,
        },
        breath_gain_init: None,
    }
}

fn spawn_agent(freq: f32, id: u64) -> Individual {
    let cfg = IndividualConfig {
        freq,
        amp: 0.5,
        life: life_with_lifecycle(LifecycleConfig::Decay {
            initial_energy: 1.0,
            half_life_sec: 0.5,
            attack_sec: crate::life::lifecycle::default_decay_attack(),
        }),
        tag: None,
    };
    let meta = AgentMetadata {
        id,
        tag: None,
        group_idx: 0,
        member_idx: 0,
    };
    cfg.spawn(id, 0, meta, 48_000.0)
}

#[test]
fn test_inertia_calculation() {
    let landscape = make_landscape();
    let mut low = spawn_agent(60.0, 1);
    let mut high = spawn_agent(1000.0, 2);
    let rhythms = NeuralRhythms::default();

    low.update_field_target(&rhythms, 0.01, &landscape);
    high.update_field_target(&rhythms, 0.01, &landscape);

    assert!(
        low.integration_window > high.integration_window,
        "expected heavier (low) pitch to integrate longer than high pitch"
    );
}

#[test]
fn test_scan_logic() {
    let mut landscape = make_landscape();
    let mut agent = spawn_agent(220.0, 3);
    let n = landscape.consonance01.len();
    landscape.subjective_intensity = vec![1.0; n];
    landscape.consonance01.fill(0.0);
    let idx_cur = landscape
        .space
        .index_of_freq(agent.body.base_freq_hz())
        .unwrap_or(0);
    landscape.consonance01[idx_cur] = 0.0;
    let target_alt = agent.body.base_freq_hz() * 1.5;
    if let Some(idx_alt) = landscape.space.index_of_freq(target_alt) {
        if let Some(c) = landscape.consonance01.get_mut(idx_alt) {
            *c = 1.0;
        }
    }

    agent.accumulated_time = 5.0;
    agent.last_theta_sample = -0.1;
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.mag = 1.0;
    rhythms.theta.phase = 0.25;

    let before = agent.target_pitch_log2;
    agent.update_field_target(&rhythms, 0.01, &landscape);
    assert!(
        agent.target_pitch_log2 > before,
        "agent should move toward higher-scoring neighbor"
    );
}

#[test]
fn test_breath_gating() {
    let landscape = make_landscape();
    let mut agent = spawn_agent(330.0, 4);
    let original = agent.body.base_freq_hz();
    agent.target_pitch_log2 = (original * 1.5).log2();
    agent.breath_gain = 1.0;

    let rhythms = NeuralRhythms::default();
    agent.update_field_target(&rhythms, 0.05, &landscape);

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

#[test]
fn setfreq_sync_prevents_snapback() {
    let landscape = make_landscape();
    let mut pop = Population::new(48_000.0);
    let agent_cfg = IndividualConfig {
        freq: 220.0,
        amp: 0.1,
        life: life_with_lifecycle(LifecycleConfig::Decay {
            initial_energy: 1.0,
            half_life_sec: 0.5,
            attack_sec: crate::life::lifecycle::default_decay_attack(),
        }),
        tag: Some("setfreq_test".to_string()),
    };
    pop.apply_action(Action::AddAgent { agent: agent_cfg }, &landscape, None);

    let (old_target, old_freq) = {
        let agent = pop.individuals.first().expect("agent exists");
        (agent.target_pitch_log2, agent.body.base_freq_hz())
    };

    let new_freq: f32 = 440.0;
    let new_log = new_freq.log2();
    pop.apply_action(
        Action::SetFreq {
            target: "setfreq_test".to_string(),
            freq_hz: new_freq,
        },
        &landscape,
        None,
    );

    let agent = pop.individuals.first_mut().expect("agent exists");
    assert!(
        (agent.target_pitch_log2 - new_log).abs() < 1e-6,
        "target should sync to new log2 pitch"
    );
    assert!(
        (agent.body.base_freq_hz() - new_freq).abs() < 1e-3,
        "body should snap to new frequency"
    );
    assert!(
        (agent.breath_gain - 1.0).abs() < 1e-6,
        "breath gain should reset to 1.0 on SetFreq"
    );

    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.mag = 0.0;
    rhythms.theta.phase = 0.0;
    let dt_sec = 0.02;
    let steps = 500;
    let agent = pop.individuals.first_mut().expect("agent exists");
    for _ in 0..steps {
        agent.update_field_target(&rhythms, dt_sec, &landscape);
    }
    assert!(
        (agent.target_pitch_log2 - new_log).abs() < 1e-6,
        "target should remain at SetFreq pitch"
    );
    assert!(
        (agent.body.base_freq_hz() - new_freq).abs() < 1e-3,
        "body should remain at SetFreq frequency"
    );
    assert!(
        (agent.target_pitch_log2 - old_target).abs() > 0.5,
        "target should not drift back toward old target"
    );
    assert!(
        (agent.body.base_freq_hz() - old_freq).abs() > 1.0,
        "body should not drift back toward old frequency"
    );
}
