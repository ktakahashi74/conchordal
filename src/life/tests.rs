use super::conductor::Conductor;
use super::individual::{
    AgentMetadata, ArticulationSignal, AudioAgent, IndividualWrapper, NeuralCore, SequencedCore,
    SoundBody,
};
use super::population::{Population, PopulationParams};
use super::scenario::{
    Action, BrainConfig, Event, HarmonicMode, IndividualConfig, Scenario, Scene, TimbreGenotype,
};
use crate::core::landscape::Landscape;
use crate::core::landscape::LandscapeFrame;
use crate::core::log2space::Log2Space;
use crate::life::lifecycle::LifecycleConfig;
use serde_json;

#[test]
fn test_population_add_remove_agent() {
    // 1. Setup
    let params = PopulationParams {
        initial_tones_hz: vec![],
        amplitude: 0.1,
    };
    let mut pop = Population::new(params, 48_000.0);
    let landscape = LandscapeFrame::default();

    assert_eq!(pop.individuals.len(), 0, "Population should start empty");

    // 2. Add Agent
    let life = LifecycleConfig::Decay {
        initial_energy: 1.0,
        half_life_sec: 1.0,
        attack_sec: 0.01,
    };
    let agent_cfg = IndividualConfig::PureTone {
        freq: 440.0,
        amp: 0.5,
        phase: None,
        rhythm_freq: None,
        rhythm_sensitivity: None,
        commitment: None,
        habituation_sensitivity: None,
        brain: BrainConfig::Entrain { lifecycle: life },
        tag: Some("test_agent".to_string()),
    };
    let action_add = Action::AddAgent { agent: agent_cfg };

    pop.apply_action(action_add, &landscape, None);
    assert_eq!(pop.individuals.len(), 1, "Agent should be added");

    // 3. Remove Agent
    let action_remove = Action::RemoveAgent {
        target: "test_agent".to_string(),
    };
    pop.apply_action(action_remove, &landscape, None);
    assert_eq!(pop.individuals.len(), 0, "Agent should be removed");
}

#[test]
fn wildcard_target_removes_matching_agents() {
    let params = PopulationParams {
        initial_tones_hz: vec![],
        amplitude: 0.1,
    };
    let mut pop = Population::new(params, 48_000.0);
    let landscape = LandscapeFrame::default();

    let life = LifecycleConfig::Decay {
        initial_energy: 1.0,
        half_life_sec: 1.0,
        attack_sec: 0.01,
    };
    let agent_cfg1 = IndividualConfig::PureTone {
        freq: 440.0,
        amp: 0.5,
        phase: None,
        rhythm_freq: None,
        rhythm_sensitivity: None,
        commitment: None,
        habituation_sensitivity: None,
        brain: BrainConfig::Entrain {
            lifecycle: life.clone(),
        },
        tag: Some("test_a".to_string()),
    };
    let agent_cfg2 = IndividualConfig::PureTone {
        freq: 330.0,
        amp: 0.4,
        phase: None,
        rhythm_freq: None,
        rhythm_sensitivity: None,
        commitment: None,
        habituation_sensitivity: None,
        brain: BrainConfig::Entrain { lifecycle: life },
        tag: Some("test_b".to_string()),
    };
    pop.apply_action(Action::AddAgent { agent: agent_cfg1 }, &landscape, None);
    pop.apply_action(Action::AddAgent { agent: agent_cfg2 }, &landscape, None);
    assert_eq!(pop.individuals.len(), 2);

    pop.apply_action(
        Action::RemoveAgent {
            target: "test_*".to_string(),
        },
        &landscape,
        None,
    );
    assert_eq!(
        pop.individuals.len(),
        0,
        "Wildcard should remove both agents"
    );
}

#[test]
fn test_conductor_timing() {
    // 1. Create a Scenario with an event at T=1.0s
    let action = Action::Finish; // Simple marker action
    let event = Event {
        time: 1.0,
        repeat: None,
        actions: vec![action],
    };
    let episode = Scene {
        name: Some("test".into()),
        start_time: 0.0,
        events: vec![event],
    };
    let scenario = Scenario {
        scenes: vec![episode],
    };

    let mut conductor = Conductor::from_scenario(scenario);
    let mut pop = Population::new(
        PopulationParams {
            initial_tones_hz: vec![],
            amplitude: 0.0,
        },
        48_000.0,
    );
    let landscape = LandscapeFrame::default();

    // 2. Dispatch at T=0.5 (Should NOT fire)
    conductor.dispatch_until(
        0.5,
        0,
        &landscape,
        None::<&mut crate::core::stream::roughness::RoughnessStream>,
        &mut pop,
    );
    assert!(
        !pop.abort_requested,
        "Finish action should not fire yet at T=0.5"
    );

    // 3. Dispatch at T=1.1 (Should fire)
    conductor.dispatch_until(
        1.1,
        100,
        &landscape,
        None::<&mut crate::core::stream::roughness::RoughnessStream>,
        &mut pop,
    );
    assert!(pop.abort_requested, "Finish action should fire at T=1.1");
}

fn make_test_landscape(_fs: f32) -> Landscape {
    let space = Log2Space::new(55.0, 4000.0, 64);
    Landscape::new(space)
}

#[test]
fn test_agent_lifecycle_decay_death() {
    // 1. Setup Population with 1 agent that has a very short half-life
    let params = PopulationParams {
        initial_tones_hz: vec![],
        amplitude: 0.1,
    };
    let mut pop = Population::new(params, 48_000.0);
    let landscape = LandscapeFrame::default(); // Dummy landscape

    // Half-life = 0.05s (very fast decay)
    let life = LifecycleConfig::Decay {
        initial_energy: 1.0,
        half_life_sec: 0.05,
        attack_sec: 0.001,
    };
    let agent_cfg = IndividualConfig::PureTone {
        freq: 440.0,
        amp: 0.5,
        phase: None,
        rhythm_freq: None,
        rhythm_sensitivity: None,
        commitment: None,
        habituation_sensitivity: None,
        brain: BrainConfig::Entrain { lifecycle: life },
        tag: None,
    };
    pop.apply_action(Action::AddAgent { agent: agent_cfg }, &landscape, None);

    assert_eq!(pop.individuals.len(), 1, "Agent added");

    // 2. Simulate time passing via process_frame
    // We need to run enough frames for energy to drop below threshold (1e-4)
    // Energy starts at 1.0.
    // After 0.05s -> 0.5
    // After 0.10s -> 0.25
    // ...
    // After 1.0s -> ~0.0
    let dt = 0.01; // 10ms steps
    let fs = 48_000.0;
    let samples_per_hop = (fs * dt) as usize;
    let landscape_rt = make_test_landscape(fs);
    let mut time = 0.0;

    // Run for 1.0 second (should be plenty for 0.05s half-life to die)
    for i in 0..100 {
        pop.process_audio(samples_per_hop, fs, i, dt, &landscape_rt);
        pop.process_frame(i, &landscape_rt.space, dt, false);
        time += dt;
    }

    // 3. Verify agent is cleaned up
    assert_eq!(
        pop.individuals.len(),
        0,
        "Agent should have died due to energy decay after {:.2}s",
        time
    );
}

#[test]
fn harmonic_render_spectrum_hits_expected_bins() {
    let genotype = TimbreGenotype {
        mode: HarmonicMode::Harmonic,
        stiffness: 0.0,
        brightness: 1.0,
        comb: 0.5,
        damping: 0.2,
        vibrato_rate: 5.0,
        vibrato_depth: 0.01,
        jitter: 0.5,
        unison: 0.1,
    };
    let cfg = IndividualConfig::Harmonic {
        freq: 55.0,
        amp: 0.8,
        genotype,
        brain: BrainConfig::Entrain {
            lifecycle: LifecycleConfig::Decay {
                initial_energy: 1.0,
                half_life_sec: 1.0,
                attack_sec: 0.01,
            },
        },
        tag: None,
        rhythm_freq: None,
        rhythm_sensitivity: None,
        commitment: None,
        habituation_sensitivity: None,
    };
    let metadata = AgentMetadata {
        id: 99,
        tag: None,
        group_idx: 0,
        member_idx: 0,
    };
    let mut agent = cfg.spawn(metadata.id, 0, metadata, 48_000.0);
    let space = Log2Space::new(55.0, 1760.0, 12);
    let mut amps = vec![0.0f32; space.n_bins()];

    match &mut agent {
        IndividualWrapper::Harmonic(ind) => {
            ind.last_signal = ArticulationSignal {
                amplitude: 1.0,
                is_active: true,
                relaxation: 0.0,
                tension: 0.0,
            };
            ind.render_spectrum(&mut amps, &space);
        }
        _ => panic!("expected harmonic agent"),
    }

    let base_bin = space.index_of_freq(55.0).expect("base bin");
    let even_bin = space.index_of_freq(110.0).expect("even bin");
    assert!(
        amps[base_bin] > 0.0,
        "fundamental bin should receive energy"
    );
    assert!(
        amps[even_bin] > 0.0,
        "second harmonic bin should receive energy"
    );
    assert!(
        amps[base_bin] > amps[even_bin],
        "comb and brightness should attenuate even harmonic"
    );
}

#[test]
fn set_freq_syncs_target_pitch_log2() {
    let space = Log2Space::new(55.0, 880.0, 12);
    let landscape = LandscapeFrame::new(space);
    let mut pop = Population::new(
        PopulationParams {
            initial_tones_hz: vec![220.0],
            amplitude: 0.1,
        },
        48_000.0,
    );

    let agent = pop.individuals.first_mut().expect("agent exists");
    match agent {
        IndividualWrapper::PureTone(ind) => {
            ind.metadata.tag = Some("test_agent".to_string());
        }
        IndividualWrapper::Harmonic(ind) => {
            ind.metadata.tag = Some("test_agent".to_string());
        }
    }

    pop.apply_action(
        Action::SetFreq {
            target: "test_agent".to_string(),
            freq_hz: 440.0,
        },
        &landscape,
        None,
    );

    let log_target = 440.0f32.log2();
    let agent = pop.individuals.first_mut().expect("agent exists");
    match agent {
        IndividualWrapper::PureTone(ind) => {
            assert!((ind.target_pitch_log2 - log_target).abs() < 1e-6);
            assert!((ind.body.base_freq_hz() - 440.0).abs() < 1e-3);
            ind.tessitura_center = ind.target_pitch_log2;
            ind.tessitura_gravity = 0.0;
            let rhythms = crate::core::modulation::NeuralRhythms::default();
            for _ in 0..16 {
                ind.update_organic_movement(&rhythms, 0.1, &landscape);
            }
            assert!((ind.target_pitch_log2 - log_target).abs() < 1e-6);
            assert!((ind.body.base_freq_hz() - 440.0).abs() < 1e-3);
        }
        IndividualWrapper::Harmonic(ind) => {
            assert!((ind.target_pitch_log2 - log_target).abs() < 1e-6);
            assert!((ind.body.base_freq_hz() - 440.0).abs() < 1e-3);
            ind.tessitura_center = ind.target_pitch_log2;
            ind.tessitura_gravity = 0.0;
            let rhythms = crate::core::modulation::NeuralRhythms::default();
            for _ in 0..16 {
                ind.update_organic_movement(&rhythms, 0.1, &landscape);
            }
            assert!((ind.target_pitch_log2 - log_target).abs() < 1e-6);
            assert!((ind.body.base_freq_hz() - 440.0).abs() < 1e-3);
        }
    }
}

#[test]
fn population_spectrum_uses_log2_space() {
    let space = Log2Space::new(55.0, 880.0, 12);
    let params = PopulationParams {
        initial_tones_hz: vec![55.0],
        amplitude: 0.5,
    };
    let mut pop = Population::new(params, 48_000.0);
    if let Some(IndividualWrapper::PureTone(ind)) = pop.individuals.first_mut() {
        ind.last_signal = ArticulationSignal {
            amplitude: 1.0,
            is_active: true,
            relaxation: 0.0,
            tension: 0.0,
        };
    } else {
        panic!("expected pure tone agent");
    }

    let amps = pop.process_frame(0, &space, 0.01, false);
    assert_eq!(amps.len(), space.n_bins());
    let idx = space.index_of_freq(55.0).expect("base bin");
    assert!(amps[idx] > 0.0);
}

#[test]
fn legacy_lifecycle_deserializes_as_brain_entrain() {
    let json = r#"{
        "type": "pure_tone",
        "freq": 330.0,
        "amp": 0.4,
        "lifecycle": { "type": "decay", "initial_energy": 1.0, "half_life_sec": 0.25 }
    }"#;
    let cfg: IndividualConfig = serde_json::from_str(json).expect("legacy config parses");
    match cfg {
        IndividualConfig::PureTone { brain, .. } => match brain {
            BrainConfig::Entrain { lifecycle } => {
                assert!(matches!(lifecycle, LifecycleConfig::Decay { .. }))
            }
            other => panic!("expected entrain brain, got {:?}", other),
        },
        other => panic!("unexpected variant: {:?}", other),
    }
}

#[test]
fn sequenced_core_stops_after_duration() {
    let mut core = SequencedCore {
        timer: 0.0,
        duration: 0.1,
        env_level: 0.0,
    };
    let rhythms = crate::core::modulation::NeuralRhythms::default();
    let active = core.process(0.0, &rhythms, 0.05, 1.0);
    assert!(active.is_active, "core should be active before duration");
    let finished = core.process(0.0, &rhythms, 0.1, 1.0);
    assert!(
        !finished.is_active && !core.is_alive(),
        "core should stop after duration"
    );
}
