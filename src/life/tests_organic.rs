use crate::core::landscape::Landscape;
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::life::individual::AgentMetadata;
use crate::life::individual::{IndividualWrapper, SoundBody};
use crate::life::lifecycle::LifecycleConfig;
use crate::life::population::{Population, PopulationParams};
use crate::life::scenario::Action;
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
    agent.update_organic_movement(&rhythms, 0.01, &landscape);
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

#[test]
fn movement_compares_adjusted_scores() {
    let space = Log2Space::new(32.0, 512.0, 12);
    let landscape = Landscape::new(space);
    let mut agent = spawn_agent(64.0, 5);
    let current_pitch = 64.0f32.log2();
    agent.body.set_pitch_log2(current_pitch);
    agent.target_pitch_log2 = current_pitch;
    agent.tessitura_center = current_pitch - 1.0;
    agent.tessitura_gravity = 1.0;
    agent.last_theta_sample = -1.0;
    agent.accumulated_time = 10.0;

    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.mag = 1.0;
    rhythms.theta.phase = std::f32::consts::FRAC_PI_2;

    let before = agent.target_pitch_log2;
    agent.update_organic_movement(&rhythms, 0.01, &landscape);
    assert!(
        agent.target_pitch_log2 < before,
        "expected adjusted-score improvement to move toward tessitura center"
    );
}

#[test]
fn setfreq_sync_prevents_snapback() {
    let landscape = make_landscape();
    let mut pop = Population::new(PopulationParams {
        initial_tones_hz: vec![220.0],
        amplitude: 0.1,
    });

    let agent = pop.individuals.first_mut().expect("agent exists");
    match agent {
        IndividualWrapper::PureTone(ind) => {
            ind.metadata.tag = Some("setfreq_test".to_string());
        }
        IndividualWrapper::Harmonic(ind) => {
            ind.metadata.tag = Some("setfreq_test".to_string());
        }
    }

    let (old_target, old_freq) = {
        let agent = pop.individuals.first().expect("agent exists");
        match agent {
            IndividualWrapper::PureTone(ind) => (ind.target_pitch_log2, ind.body.base_freq_hz()),
            IndividualWrapper::Harmonic(ind) => (ind.target_pitch_log2, ind.body.base_freq_hz()),
        }
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
    match agent {
        IndividualWrapper::PureTone(ind) => {
            assert!(
                (ind.target_pitch_log2 - new_log).abs() < 1e-6,
                "target should sync to new log2 pitch"
            );
            assert!(
                (ind.body.base_freq_hz() - new_freq).abs() < 1e-3,
                "body should snap to new frequency"
            );
            assert!(
                (ind.breath_gain - 1.0).abs() < 1e-6,
                "breath gain should reset to 1.0 on SetFreq"
            );
        }
        IndividualWrapper::Harmonic(ind) => {
            assert!(
                (ind.target_pitch_log2 - new_log).abs() < 1e-6,
                "target should sync to new log2 pitch"
            );
            assert!(
                (ind.body.base_freq_hz() - new_freq).abs() < 1e-3,
                "body should snap to new frequency"
            );
            assert!(
                (ind.breath_gain - 1.0).abs() < 1e-6,
                "breath gain should reset to 1.0 on SetFreq"
            );
        }
    }

    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.mag = 0.0;
    rhythms.theta.phase = 0.0;
    let dt_sec = 0.02;
    let steps = 500;
    let agent = pop.individuals.first_mut().expect("agent exists");
    match agent {
        IndividualWrapper::PureTone(ind) => {
            for _ in 0..steps {
                ind.update_organic_movement(&rhythms, dt_sec, &landscape);
            }
            assert!(
                (ind.target_pitch_log2 - new_log).abs() < 1e-6,
                "target should remain at SetFreq pitch"
            );
            assert!(
                (ind.body.base_freq_hz() - new_freq).abs() < 1e-3,
                "body should remain at SetFreq frequency"
            );
            assert!(
                (ind.target_pitch_log2 - old_target).abs() > 0.5,
                "target should not drift back toward old target"
            );
            assert!(
                (ind.body.base_freq_hz() - old_freq).abs() > 1.0,
                "body should not drift back toward old frequency"
            );
        }
        IndividualWrapper::Harmonic(ind) => {
            for _ in 0..steps {
                ind.update_organic_movement(&rhythms, dt_sec, &landscape);
            }
            assert!(
                (ind.target_pitch_log2 - new_log).abs() < 1e-6,
                "target should remain at SetFreq pitch"
            );
            assert!(
                (ind.body.base_freq_hz() - new_freq).abs() < 1e-3,
                "body should remain at SetFreq frequency"
            );
            assert!(
                (ind.target_pitch_log2 - old_target).abs() > 0.5,
                "target should not drift back toward old target"
            );
            assert!(
                (ind.body.base_freq_hz() - old_freq).abs() > 1.0,
                "body should not drift back toward old frequency"
            );
        }
    }
}
