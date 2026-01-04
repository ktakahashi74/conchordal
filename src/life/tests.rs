use super::conductor::Conductor;
use super::individual::{
    AgentMetadata, ArticulationCore, ArticulationSignal, ArticulationState, AudioAgent, Individual,
    KuramotoCore, PinkNoise, PitchCore, Sensitivity, SequencedCore, SoundBody,
};
use super::population::Population;
use super::scenario::{
    Action, ArticulationCoreConfig, EnvelopeConfig, HarmonicMode, IndividualConfig, LifeConfig,
    PitchCoreConfig, Scenario, SceneMarker, SoundBodyConfig, TargetRef, TimbreGenotype, TimedEvent,
};
use crate::core::landscape::Landscape;
use crate::core::landscape::LandscapeFrame;
use crate::core::log2space::Log2Space;
use crate::core::modulation::{NeuralRhythms, RhythmBand};
use crate::life::lifecycle::LifecycleConfig;
use crate::life::perceptual::{FeaturesNow, PerceptualConfig, PerceptualContext};
use rand::SeedableRng;
use serde_json;

fn mix_signature(mut acc: u64, value: u32) -> u64 {
    acc ^= value as u64;
    acc = acc.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    acc
}

fn life_with_lifecycle(lifecycle: LifecycleConfig) -> LifeConfig {
    LifeConfig {
        body: SoundBodyConfig::Sine { phase: None },
        articulation: ArticulationCoreConfig::Entrain {
            lifecycle,
            rhythm_freq: None,
            rhythm_sensitivity: None,
        },
        pitch: PitchCoreConfig::PitchHillClimb {
            neighbor_step_cents: None,
            tessitura_gravity: None,
            improvement_threshold: None,
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
        ..Default::default()
    }
}

#[test]
fn test_population_add_remove_agent() {
    // 1. Setup
    let mut pop = Population::new(48_000.0);
    let landscape = LandscapeFrame::default();

    assert_eq!(pop.individuals.len(), 0, "Population should start empty");

    // 2. Add Agent
    let life = LifecycleConfig::Decay {
        initial_energy: 1.0,
        half_life_sec: 1.0,
        attack_sec: 0.01,
    };
    let agent_cfg = IndividualConfig {
        freq: 440.0,
        amp: 0.5,
        life: life_with_lifecycle(life),
        tag: Some("test_agent".to_string()),
    };
    let action_add = Action::AddAgent {
        id: 1,
        agent: agent_cfg,
    };

    pop.apply_action(action_add, &landscape, None);
    assert_eq!(pop.individuals.len(), 1, "Agent should be added");

    // 3. Remove Agent
    let action_remove = Action::RemoveAgent {
        target: TargetRef::Tag {
            tag: "test_agent".to_string(),
        },
    };
    pop.apply_action(action_remove, &landscape, None);
    assert_eq!(pop.individuals.len(), 0, "Agent should be removed");
}

#[test]
fn tag_selector_removes_matching_agents() {
    let mut pop = Population::new(48_000.0);
    let landscape = LandscapeFrame::default();

    let life = LifecycleConfig::Decay {
        initial_energy: 1.0,
        half_life_sec: 1.0,
        attack_sec: 0.01,
    };
    let agent_cfg1 = IndividualConfig {
        freq: 440.0,
        amp: 0.5,
        life: life_with_lifecycle(life.clone()),
        tag: Some("test_group".to_string()),
    };
    let agent_cfg2 = IndividualConfig {
        freq: 330.0,
        amp: 0.4,
        life: life_with_lifecycle(life),
        tag: Some("test_group".to_string()),
    };
    pop.apply_action(
        Action::AddAgent {
            id: 1,
            agent: agent_cfg1,
        },
        &landscape,
        None,
    );
    pop.apply_action(
        Action::AddAgent {
            id: 2,
            agent: agent_cfg2,
        },
        &landscape,
        None,
    );
    assert_eq!(pop.individuals.len(), 2);

    pop.apply_action(
        Action::RemoveAgent {
            target: TargetRef::Tag {
                tag: "test_group".to_string(),
            },
        },
        &landscape,
        None,
    );
    assert_eq!(
        pop.individuals.len(),
        0,
        "Tag selector should remove both agents"
    );
}

#[test]
fn test_conductor_timing() {
    // 1. Create a Scenario with an event at T=1.0s
    let action = Action::Finish; // Simple marker action
    let event = TimedEvent {
        time: 1.0,
        order: 1,
        actions: vec![action],
    };
    let scenario = Scenario {
        seed: 0,
        scene_markers: vec![SceneMarker {
            name: "test".into(),
            time: 0.0,
            order: 0,
        }],
        events: vec![event],
        duration_sec: 2.0,
    };

    let mut conductor = Conductor::from_scenario(scenario);
    let mut pop = Population::new(48_000.0);
    let landscape = LandscapeFrame::default();
    let space = Log2Space::new(1.0, 2.0, 1);
    let mut world = crate::life::world_model::WorldModel::new(
        crate::core::timebase::Timebase {
            fs: 48_000.0,
            hop: 512,
        },
        space,
    );

    // 2. Dispatch at T=0.5 (Should NOT fire)
    conductor.dispatch_until(
        0.5,
        0,
        &landscape,
        None::<&mut crate::core::stream::roughness::RoughnessStream>,
        &mut pop,
        &mut world,
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
        &mut world,
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
    let mut pop = Population::new(48_000.0);
    let landscape = LandscapeFrame::default(); // Dummy landscape

    // Half-life = 0.05s (very fast decay)
    let life = LifecycleConfig::Decay {
        initial_energy: 1.0,
        half_life_sec: 0.05,
        attack_sec: 0.001,
    };
    let agent_cfg = IndividualConfig {
        freq: 440.0,
        amp: 0.5,
        life: life_with_lifecycle(life),
        tag: None,
    };
    pop.apply_action(
        Action::AddAgent {
            id: 1,
            agent: agent_cfg,
        },
        &landscape,
        None,
    );

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
        pop.advance(samples_per_hop, fs, i, dt, &landscape_rt);
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
    let cfg = IndividualConfig {
        freq: 55.0,
        amp: 0.8,
        life: LifeConfig {
            body: SoundBodyConfig::Harmonic {
                genotype: genotype.clone(),
                partials: Some(16),
            },
            articulation: ArticulationCoreConfig::Entrain {
                lifecycle: LifecycleConfig::Decay {
                    initial_energy: 1.0,
                    half_life_sec: 1.0,
                    attack_sec: 0.01,
                },
                rhythm_freq: None,
                rhythm_sensitivity: None,
            },
            pitch: PitchCoreConfig::PitchHillClimb {
                neighbor_step_cents: None,
                tessitura_gravity: None,
                improvement_threshold: None,
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
            ..Default::default()
        },
        tag: None,
    };
    let metadata = AgentMetadata {
        id: 99,
        tag: None,
        group_idx: 0,
        member_idx: 0,
    };
    let mut agent = cfg.spawn(metadata.id, 0, metadata, 48_000.0, 0);
    let space = Log2Space::new(55.0, 1760.0, 12);
    let mut amps = vec![0.0f32; space.n_bins()];

    agent.last_signal = ArticulationSignal {
        amplitude: 1.0,
        is_active: true,
        relaxation: 0.0,
        tension: 0.0,
    };
    agent.render_spectrum(&mut amps, &space);

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
    let mut pop = Population::new(48_000.0);
    let agent_cfg = IndividualConfig {
        freq: 220.0,
        amp: 0.1,
        life: life_with_lifecycle(LifecycleConfig::Decay {
            initial_energy: 1.0,
            half_life_sec: 1.0,
            attack_sec: 0.01,
        }),
        tag: Some("test_agent".to_string()),
    };
    pop.apply_action(
        Action::AddAgent {
            id: 1,
            agent: agent_cfg,
        },
        &landscape,
        None,
    );

    pop.apply_action(
        Action::SetFreq {
            target: TargetRef::Tag {
                tag: "test_agent".to_string(),
            },
            freq_hz: 440.0,
        },
        &landscape,
        None,
    );

    let log_target = 440.0f32.log2();
    let agent = pop.individuals.first_mut().expect("agent exists");
    assert!((agent.target_pitch_log2 - log_target).abs() < 1e-6);
    assert!((agent.body.base_freq_hz() - 440.0).abs() < 1e-3);
}

#[test]
fn population_spectrum_uses_log2_space() {
    let space = Log2Space::new(55.0, 880.0, 12);
    let mut pop = Population::new(48_000.0);
    let agent_cfg = IndividualConfig {
        freq: 55.0,
        amp: 0.5,
        life: life_with_lifecycle(LifecycleConfig::Decay {
            initial_energy: 1.0,
            half_life_sec: 1.0,
            attack_sec: 0.01,
        }),
        tag: None,
    };
    pop.apply_action(
        Action::AddAgent {
            id: 1,
            agent: agent_cfg,
        },
        &LandscapeFrame::new(space.clone()),
        None,
    );
    let agent = pop.individuals.first_mut().expect("agent exists");
    agent.last_signal = ArticulationSignal {
        amplitude: 1.0,
        is_active: true,
        relaxation: 0.0,
        tension: 0.0,
    };

    let amps = pop.process_frame(0, &space, 0.01, false);
    assert_eq!(amps.len(), space.n_bins());
    let idx = space.index_of_freq(55.0).expect("base bin");
    assert!(amps[idx] > 0.0);
}

#[test]
fn life_config_deserializes_and_rejects_unknown_fields() {
    let json = serde_json::json!({
        "body": { "core": "sine" },
        "articulation": {
            "core": "entrain",
            "type": "decay",
            "initial_energy": 1.0,
            "half_life_sec": 0.25
        },
        "pitch": { "core": "pitch_hill_climb" },
        "perceptual": {
            "tau_fast": 0.5,
            "tau_slow": 4.0,
            "w_boredom": 0.8,
            "w_familiarity": 0.2,
            "rho_self": 0.15,
            "boredom_gamma": 0.5,
            "self_smoothing_radius": 1
        }
    });
    let cfg: LifeConfig = serde_json::from_value(json).expect("life config parses");
    assert!(matches!(
        cfg.articulation,
        ArticulationCoreConfig::Entrain { .. }
    ));

    let bad = serde_json::json!({
        "body": { "core": "sine", "unknown": 1.0 },
        "articulation": {
            "core": "entrain",
            "type": "decay",
            "initial_energy": 1.0,
            "half_life_sec": 0.25,
            "unknown": 1.0
        },
        "pitch": { "core": "pitch_hill_climb" },
        "perceptual": {
            "tau_fast": 0.5,
            "tau_slow": 4.0,
            "w_boredom": 0.8,
            "w_familiarity": 0.2,
            "rho_self": 0.15,
            "boredom_gamma": 0.5,
            "self_smoothing_radius": 1
        }
    });
    assert!(serde_json::from_value::<LifeConfig>(bad).is_err());

    let missing = serde_json::json!({
        "articulation": {
            "core": "entrain",
            "type": "decay",
            "initial_energy": 1.0,
            "half_life_sec": 0.25
        },
        "pitch": { "core": "pitch_hill_climb" },
        "perceptual": {
            "tau_fast": 0.5,
            "tau_slow": 4.0,
            "w_boredom": 0.8,
            "w_familiarity": 0.2,
            "rho_self": 0.15,
            "boredom_gamma": 0.5,
            "self_smoothing_radius": 1
        }
    });
    let cfg_missing: LifeConfig =
        serde_json::from_value(missing).expect("missing body should default");
    assert!(matches!(cfg_missing.body, SoundBodyConfig::Sine { .. }));
}

#[test]
fn pitch_core_proposes_target_within_bounds() {
    let space = Log2Space::new(55.0, 880.0, 12);
    let landscape = Landscape::new(space.clone());
    let mut rng = rand::rngs::StdRng::seed_from_u64(4);
    let mut pitch = super::individual::AnyPitchCore::from_config(
        &PitchCoreConfig::PitchHillClimb {
            neighbor_step_cents: None,
            tessitura_gravity: None,
            improvement_threshold: None,
            exploration: None,
            persistence: None,
        },
        220.0f32.log2(),
        &mut rng,
    );
    let proposal = pitch.propose_target(
        220.0f32.log2(),
        220.0f32.log2(),
        220.0,
        2.0,
        &landscape,
        &crate::life::perceptual::PerceptualContext::from_config(
            &PerceptualConfig {
                tau_fast: Some(0.5),
                tau_slow: Some(4.0),
                w_boredom: Some(0.8),
                w_familiarity: Some(0.2),
                rho_self: Some(0.0),
                boredom_gamma: Some(0.5),
                self_smoothing_radius: Some(0),
                silence_mass_epsilon: Some(1e-6),
            },
            space.n_bins(),
        ),
        &crate::life::perceptual::FeaturesNow::from_subjective_intensity(&vec![
            0.0;
            space.n_bins()
        ]),
        &mut rng,
    );
    let (fmin, fmax) = landscape.freq_bounds_log2();
    assert!(proposal.target_pitch_log2 >= fmin && proposal.target_pitch_log2 <= fmax);
    assert!((0.0..=1.0).contains(&proposal.salience));
}

#[test]
fn deterministic_rng_produces_same_targets() {
    let life = LifeConfig {
        body: SoundBodyConfig::Sine { phase: None },
        articulation: ArticulationCoreConfig::Entrain {
            lifecycle: LifecycleConfig::Decay {
                initial_energy: 1.0,
                half_life_sec: 0.5,
                attack_sec: 0.01,
            },
            rhythm_freq: Some(1.0),
            rhythm_sensitivity: None,
        },
        pitch: PitchCoreConfig::PitchHillClimb {
            neighbor_step_cents: None,
            tessitura_gravity: None,
            improvement_threshold: None,
            exploration: Some(0.2),
            persistence: Some(0.5),
        },
        perceptual: PerceptualConfig {
            tau_fast: Some(0.5),
            tau_slow: Some(4.0),
            w_boredom: Some(0.8),
            w_familiarity: Some(0.2),
            rho_self: Some(0.15),
            boredom_gamma: Some(0.5),
            self_smoothing_radius: Some(1),
            silence_mass_epsilon: Some(1e-6),
        },
        breath_gain_init: None,
        ..Default::default()
    };
    let cfg = IndividualConfig {
        freq: 220.0,
        amp: 0.2,
        life,
        tag: None,
    };
    let meta = AgentMetadata {
        id: 10,
        tag: None,
        group_idx: 0,
        member_idx: 0,
    };
    let mut a = cfg.spawn(10, 4, meta.clone(), 48_000.0, 0);
    let mut b = cfg.spawn(10, 4, meta, 48_000.0, 0);
    let landscape = make_test_landscape(48_000.0);
    let mut rhythms = crate::core::modulation::NeuralRhythms::default();
    let dt = 0.5;
    let mut seq_a = Vec::new();
    let mut seq_b = Vec::new();
    for i in 0..16 {
        rhythms.theta.mag = 1.0;
        rhythms.theta.phase = if i % 2 == 0 {
            -std::f32::consts::FRAC_PI_2
        } else {
            std::f32::consts::FRAC_PI_2
        };
        a.update_pitch_target(&rhythms, dt, &landscape);
        b.update_pitch_target(&rhythms, dt, &landscape);
        seq_a.push(a.target_pitch_log2);
        seq_b.push(b.target_pitch_log2);
    }
    assert_eq!(seq_a, seq_b);
}

#[test]
fn kuramoto_locks_to_theta_phase() {
    let mut core = KuramotoCore {
        energy: 1.0,
        basal_cost: 0.0,
        action_cost: 0.0,
        recharge_rate: 0.0,
        sensitivity: Sensitivity {
            delta: 1.0,
            theta: 1.0,
            alpha: 1.0,
            beta: 1.0,
        },
        rhythm_phase: 1.7,
        rhythm_freq: 5.0,
        omega_rad: std::f32::consts::TAU * 5.0,
        phase_offset: 0.4,
        debug_id: 0,
        env_level: 0.05,
        state: ArticulationState::Idle,
        attack_step: 0.1,
        decay_factor: 0.9,
        retrigger: true,
        noise_1f: PinkNoise::new(9, 1.0),
        base_sigma: 0.05,
        beta_gain: 0.5,
        k_omega: 3.0,
        bootstrap_timer: 0.0,
        env_open_threshold: 0.55,
        env_level_min: 0.02,
        mag_threshold: 0.2,
        alpha_threshold: 0.2,
        beta_threshold: 0.8,
        dbg_accum_time: 0.0,
        dbg_wraps: 0,
        dbg_attacks: 0,
        dbg_boot_attacks: 0,
        dbg_attack_logs_left: 0,
        dbg_attack_count_normal: 0,
        dbg_attack_sum_abs_diff: 0.0,
        dbg_attack_sum_cos: 0.0,
        dbg_attack_sum_sin: 0.0,
        dbg_fail_env: 0,
        dbg_fail_env_level: 0,
        dbg_fail_mag: 0,
        dbg_fail_alpha: 0,
        dbg_fail_beta: 0,
        dbg_last_env_open: 0.0,
        dbg_last_env_level: 0.0,
        dbg_last_theta_mag: 0.0,
        dbg_last_theta_alpha: 0.0,
        dbg_last_theta_beta: 0.0,
        dbg_last_k_eff: 0.0,
    };
    let mut theta_phase = 0.0;
    let dt = 0.01;
    for _ in 0..3000 {
        theta_phase =
            (theta_phase + std::f32::consts::TAU * 6.0 * dt).rem_euclid(std::f32::consts::TAU);
        let rhythms = NeuralRhythms {
            theta: RhythmBand {
                phase: theta_phase,
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
        core.process(0.5, &rhythms, dt, 1.0);
    }

    let target = (theta_phase + core.phase_offset).rem_euclid(std::f32::consts::TAU);
    let mut diff = target - core.rhythm_phase.rem_euclid(std::f32::consts::TAU);
    while diff > std::f32::consts::PI {
        diff -= std::f32::consts::TAU;
    }
    while diff < -std::f32::consts::PI {
        diff += std::f32::consts::TAU;
    }
    assert!(
        diff.abs() < 0.5,
        "phase should lock near theta target, got diff={diff}"
    );
}

#[test]
fn kuramoto_bootstrap_triggers_attack() {
    let mut core = KuramotoCore {
        energy: 1.0,
        basal_cost: 0.0,
        action_cost: 0.0,
        recharge_rate: 0.0,
        sensitivity: Sensitivity {
            delta: 1.0,
            theta: 1.0,
            alpha: 1.0,
            beta: 1.0,
        },
        rhythm_phase: std::f32::consts::TAU + 0.1,
        rhythm_freq: 6.0,
        omega_rad: 0.0,
        phase_offset: 0.0,
        debug_id: 1,
        env_level: 0.05,
        state: ArticulationState::Idle,
        attack_step: 0.1,
        decay_factor: 0.9,
        retrigger: true,
        noise_1f: PinkNoise::new(7, 0.0),
        base_sigma: 0.0,
        beta_gain: 0.0,
        k_omega: 0.0,
        bootstrap_timer: 1.0,
        env_open_threshold: 0.55,
        env_level_min: 0.02,
        mag_threshold: 0.2,
        alpha_threshold: 0.2,
        beta_threshold: 0.9,
        dbg_accum_time: 0.0,
        dbg_wraps: 0,
        dbg_attacks: 0,
        dbg_boot_attacks: 0,
        dbg_attack_logs_left: 0,
        dbg_attack_count_normal: 0,
        dbg_attack_sum_abs_diff: 0.0,
        dbg_attack_sum_cos: 0.0,
        dbg_attack_sum_sin: 0.0,
        dbg_fail_env: 0,
        dbg_fail_env_level: 0,
        dbg_fail_mag: 0,
        dbg_fail_alpha: 0,
        dbg_fail_beta: 0,
        dbg_last_env_open: 0.0,
        dbg_last_env_level: 0.0,
        dbg_last_theta_mag: 0.0,
        dbg_last_theta_alpha: 0.0,
        dbg_last_theta_beta: 0.0,
        dbg_last_k_eff: 0.0,
    };

    let rhythms = NeuralRhythms {
        theta: RhythmBand {
            phase: 0.0,
            freq_hz: 6.0,
            mag: 1.0,
            alpha: 1.0,
            beta: 0.0,
        },
        delta: RhythmBand {
            phase: 0.0,
            freq_hz: 1.0,
            mag: 0.0,
            alpha: 0.0,
            beta: 0.0,
        },
        env_open: 1.0,
        env_level: 0.05,
    };

    core.process(0.0, &rhythms, 0.01, 1.0);
    assert_eq!(
        core.state,
        ArticulationState::Attack,
        "bootstrap should allow an attack during early onset"
    );
}

#[test]
fn kuramoto_normal_attacks_fire_and_lock() {
    let mut core = KuramotoCore {
        energy: 1.0,
        basal_cost: 0.0,
        action_cost: 0.0,
        recharge_rate: 0.0,
        sensitivity: Sensitivity {
            delta: 1.0,
            theta: 1.0,
            alpha: 1.0,
            beta: 1.0,
        },
        rhythm_phase: 0.1,
        rhythm_freq: 6.0,
        omega_rad: std::f32::consts::TAU * 6.0,
        phase_offset: 0.0,
        debug_id: 2,
        env_level: 0.0,
        state: ArticulationState::Idle,
        attack_step: 0.1,
        decay_factor: 0.9,
        retrigger: true,
        noise_1f: PinkNoise::new(11, 0.0),
        base_sigma: 0.0,
        beta_gain: 0.0,
        k_omega: 0.0,
        bootstrap_timer: 0.0,
        env_open_threshold: 0.55,
        env_level_min: 0.02,
        mag_threshold: 0.03,
        alpha_threshold: 0.2,
        beta_threshold: 0.9,
        dbg_accum_time: -100.0,
        dbg_wraps: 0,
        dbg_attacks: 0,
        dbg_boot_attacks: 0,
        dbg_attack_logs_left: 0,
        dbg_attack_count_normal: 0,
        dbg_attack_sum_abs_diff: 0.0,
        dbg_attack_sum_cos: 0.0,
        dbg_attack_sum_sin: 0.0,
        dbg_fail_env: 0,
        dbg_fail_env_level: 0,
        dbg_fail_mag: 0,
        dbg_fail_alpha: 0,
        dbg_fail_beta: 0,
        dbg_last_env_open: 0.0,
        dbg_last_env_level: 0.0,
        dbg_last_theta_mag: 0.0,
        dbg_last_theta_alpha: 0.0,
        dbg_last_theta_beta: 0.0,
        dbg_last_k_eff: 0.0,
    };

    let dt = 0.01;
    let mut t = 0.0;
    for _ in 0..1500 {
        let theta_phase = (t * 6.0 * std::f32::consts::TAU).rem_euclid(std::f32::consts::TAU);
        let rhythms = NeuralRhythms {
            theta: RhythmBand {
                phase: theta_phase,
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
        core.process(0.5, &rhythms, dt, 1.0);
        t += dt;
    }

    assert!(
        core.dbg_attack_count_normal >= 5,
        "expected normal attacks to fire"
    );
    let count = core.dbg_attack_count_normal as f32;
    let mean_abs_diff = core.dbg_attack_sum_abs_diff / count;
    let plv = (core.dbg_attack_sum_cos * core.dbg_attack_sum_cos
        + core.dbg_attack_sum_sin * core.dbg_attack_sum_sin)
        .sqrt()
        / count;
    assert!(
        mean_abs_diff < 0.5,
        "mean abs diff too large: {mean_abs_diff}"
    );
    assert!(plv > 0.9, "attack PLV too low: {plv}");
}

#[test]
fn kuramoto_attack_count_invariant_to_chunking() {
    fn run_with_chunk(chunk: usize) -> u32 {
        let mut core = KuramotoCore {
            energy: 1.0,
            basal_cost: 0.0,
            action_cost: 0.0,
            recharge_rate: 0.0,
            sensitivity: Sensitivity {
                delta: 1.0,
                theta: 1.0,
                alpha: 1.0,
                beta: 1.0,
            },
            rhythm_phase: std::f32::consts::TAU - 0.01,
            rhythm_freq: 2.0,
            omega_rad: std::f32::consts::TAU * 2.0,
            phase_offset: 0.0,
            debug_id: 3,
            env_level: 1.0,
            state: ArticulationState::Idle,
            attack_step: 1.0,
            decay_factor: 0.2,
            retrigger: true,
            noise_1f: PinkNoise::new(11, 0.0),
            base_sigma: 0.0,
            beta_gain: 0.0,
            k_omega: 0.0,
            bootstrap_timer: 0.0,
            env_open_threshold: 0.55,
            env_level_min: 0.02,
            mag_threshold: 0.01,
            alpha_threshold: 0.01,
            beta_threshold: 0.9,
            dbg_accum_time: -1000.0,
            dbg_wraps: 0,
            dbg_attacks: 0,
            dbg_boot_attacks: 0,
            dbg_attack_logs_left: 0,
            dbg_attack_count_normal: 0,
            dbg_attack_sum_abs_diff: 0.0,
            dbg_attack_sum_cos: 0.0,
            dbg_attack_sum_sin: 0.0,
            dbg_fail_env: 0,
            dbg_fail_env_level: 0,
            dbg_fail_mag: 0,
            dbg_fail_alpha: 0,
            dbg_fail_beta: 0,
            dbg_last_env_open: 0.0,
            dbg_last_env_level: 0.0,
            dbg_last_theta_mag: 0.0,
            dbg_last_theta_alpha: 0.0,
            dbg_last_theta_beta: 0.0,
            dbg_last_k_eff: 0.0,
        };

        let mut rhythms = NeuralRhythms {
            theta: RhythmBand {
                phase: 0.0,
                freq_hz: 2.0,
                mag: 1.0,
                alpha: 1.0,
                beta: 0.0,
            },
            delta: RhythmBand {
                phase: 0.0,
                freq_hz: 0.0,
                mag: 0.0,
                alpha: 0.0,
                beta: 0.0,
            },
            env_open: 1.0,
            env_level: 1.0,
        };

        let fs = 1000.0;
        let dt = 1.0 / fs;
        let total_samples = (fs * 2.0) as usize;
        let mut idx = 0;
        while idx < total_samples {
            let end = (idx + chunk).min(total_samples);
            for _ in idx..end {
                core.process(0.0, &rhythms, dt, 1.0);
                rhythms.advance_in_place(dt);
            }
            idx = end;
        }
        core.dbg_attack_count_normal
    }

    let a = run_with_chunk(1);
    let b = run_with_chunk(64);
    assert_eq!(a, b, "attack count should be chunk-size invariant");
    assert!(a > 0, "expected at least one attack");
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

#[test]
fn perceptual_boredom_increases_with_repetition() {
    let config = PerceptualConfig {
        tau_fast: Some(0.5),
        tau_slow: Some(20.0),
        w_boredom: Some(1.0),
        w_familiarity: Some(0.2),
        rho_self: Some(0.2),
        boredom_gamma: Some(0.5),
        self_smoothing_radius: Some(0),
        silence_mass_epsilon: Some(1e-6),
    };
    let mut ctx = PerceptualContext::from_config(&config, 8);
    let features = FeaturesNow::from_subjective_intensity(&vec![0.0f32; 8]);
    let idx = 3;
    let adj0 = ctx.score_adjustment(idx);
    ctx.update(idx, &features, 0.2);
    let adj1 = ctx.score_adjustment(idx);
    ctx.update(idx, &features, 0.2);
    let adj2 = ctx.score_adjustment(idx);
    assert!(adj1 < adj0, "expected adjustment to drop after repetition");
    assert!(adj2 < adj1, "expected adjustment to keep dropping");
}

#[test]
fn perceptual_fast_trace_recovers_over_time() {
    let config = PerceptualConfig {
        tau_fast: Some(0.5),
        tau_slow: Some(20.0),
        w_boredom: Some(1.0),
        w_familiarity: Some(0.2),
        rho_self: Some(0.2),
        boredom_gamma: Some(0.5),
        self_smoothing_radius: Some(0),
        silence_mass_epsilon: Some(1e-6),
    };
    let mut ctx = PerceptualContext::from_config(&config, 8);
    let mut raw = vec![0.0f32; 8];
    raw[2] = 1.0;
    let features = FeaturesNow::from_subjective_intensity(&raw);
    let idx = 2;
    ctx.update(idx, &features, 0.2);
    let adj1 = ctx.score_adjustment(idx);
    let silence = FeaturesNow::from_subjective_intensity(&vec![0.0f32; 8]);
    ctx.update(99, &silence, 2.0);
    let adj2 = ctx.score_adjustment(idx);
    assert!(adj2 > adj1, "expected boredom penalty to relax over time");
}

#[test]
fn perceptual_rho_self_zero_ignores_self_history() {
    let config = PerceptualConfig {
        tau_fast: Some(0.5),
        tau_slow: Some(20.0),
        w_boredom: Some(1.0),
        w_familiarity: Some(0.2),
        rho_self: Some(0.0),
        boredom_gamma: Some(0.5),
        self_smoothing_radius: Some(0),
        silence_mass_epsilon: Some(1e-6),
    };
    let mut ctx = PerceptualContext::from_config(&config, 8);
    let raw = vec![1.0f32; 8];
    let features = FeaturesNow::from_subjective_intensity(&raw);
    ctx.update(1, &features, 0.2);
    let adj_a = ctx.score_adjustment(1);
    let adj_b = ctx.score_adjustment(5);
    assert!(
        (adj_a - adj_b).abs() < 1e-6,
        "expected uniform adjustment when rho_self is zero"
    );
}

#[test]
fn perceptual_silence_still_updates_self_history() {
    let config = PerceptualConfig {
        tau_fast: Some(0.5),
        tau_slow: Some(20.0),
        w_boredom: Some(1.0),
        w_familiarity: Some(0.2),
        rho_self: Some(0.0),
        boredom_gamma: Some(0.5),
        self_smoothing_radius: Some(0),
        silence_mass_epsilon: Some(1e-6),
    };
    let mut ctx = PerceptualContext::from_config(&config, 8);
    let silence = FeaturesNow::from_subjective_intensity(&vec![0.0f32; 8]);
    let idx = 4;
    let adj0 = ctx.score_adjustment(idx);
    ctx.update(idx, &silence, 0.2);
    let adj1 = ctx.score_adjustment(idx);
    assert!(adj1 < adj0, "expected boredom to increase in silence");
}

#[test]
fn perceptual_silence_threshold_ignores_tiny_mass() {
    let config = PerceptualConfig {
        tau_fast: Some(0.5),
        tau_slow: Some(20.0),
        w_boredom: Some(1.0),
        w_familiarity: Some(0.2),
        rho_self: Some(0.15),
        boredom_gamma: Some(0.5),
        self_smoothing_radius: Some(0),
        silence_mass_epsilon: Some(1e-3),
    };
    let mut ctx = PerceptualContext::from_config(&config, 4);
    let features = FeaturesNow {
        distribution: vec![0.25, 0.25, 0.25, 0.25],
        mass: 1e-4,
    };
    let idx = 1;
    let adj0 = ctx.score_adjustment(idx);
    ctx.update(idx, &features, 0.2);
    let adj1 = ctx.score_adjustment(idx);
    assert!(
        adj1 < adj0,
        "expected self-update when mass is below epsilon"
    );
}

#[test]
fn articulation_snapshot_kuramoto_signature() {
    let fs = 48_000.0;
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);
    let core = ArticulationCoreConfig::Entrain {
        lifecycle: LifecycleConfig::Sustain {
            initial_energy: 1.0,
            metabolism_rate: 0.0,
            recharge_rate: Some(0.5),
            action_cost: Some(0.02),
            envelope: EnvelopeConfig {
                attack_sec: 0.01,
                decay_sec: 0.1,
                sustain_level: 0.8,
            },
        },
        rhythm_freq: Some(6.0),
        rhythm_sensitivity: None,
    };
    let mut articulation =
        super::individual::AnyArticulationCore::from_config(&core, fs, 7, &mut rng);
    let mut rhythms = NeuralRhythms {
        theta: RhythmBand {
            phase: 0.0,
            freq_hz: 6.0,
            mag: 1.0,
            alpha: 1.0,
            beta: 0.2,
        },
        delta: RhythmBand {
            phase: 0.0,
            freq_hz: 0.5,
            mag: 1.0,
            alpha: 1.0,
            beta: 0.0,
        },
        env_level: 1.0,
        env_open: 1.0,
    };
    let dt = 1.0 / fs;
    let consonance = 0.7;
    let mut signature = 0u64;

    for _ in 0..20_000 {
        let signal = articulation.process(consonance, &rhythms, dt, 1.0);
        signature = mix_signature(signature, signal.is_active as u32);
        signature = mix_signature(signature, signal.amplitude.to_bits());
        signature = mix_signature(signature, signal.relaxation.to_bits());
        signature = mix_signature(signature, signal.tension.to_bits());
        rhythms.advance_in_place(dt);
    }

    println!("articulation signature: {signature:016x}");
    assert_eq!(signature, 0x798d_a9fd_3822_a1ca);
}

#[test]
fn render_wave_snapshot_signature() {
    let fs = 48_000.0;
    let space = Log2Space::new(110.0, 880.0, 48);
    let mut landscape = Landscape::new(space.clone());
    landscape.consonance01.fill(0.1);
    let current_freq = 220.0;
    let target_freq: f32 = 330.0;
    let target_log2 = target_freq.log2();
    if let Some(idx) = landscape.space.index_of_log2(target_log2) {
        landscape.consonance01[idx] = 0.9;
    }
    landscape.rhythm = NeuralRhythms {
        theta: RhythmBand {
            phase: 0.0,
            freq_hz: 6.0,
            mag: 1.0,
            alpha: 1.0,
            beta: 0.2,
        },
        delta: RhythmBand {
            phase: 0.0,
            freq_hz: 0.5,
            mag: 1.0,
            alpha: 1.0,
            beta: 0.0,
        },
        env_level: 1.0,
        env_open: 1.0,
    };

    let cfg = IndividualConfig {
        freq: current_freq,
        amp: 0.3,
        life: LifeConfig {
            body: SoundBodyConfig::Sine { phase: Some(0.0) },
            articulation: ArticulationCoreConfig::Entrain {
                lifecycle: LifecycleConfig::Sustain {
                    initial_energy: 1.0,
                    metabolism_rate: 0.0,
                    recharge_rate: Some(0.5),
                    action_cost: Some(0.02),
                    envelope: EnvelopeConfig {
                        attack_sec: 0.01,
                        decay_sec: 0.1,
                        sustain_level: 0.8,
                    },
                },
                rhythm_freq: Some(6.0),
                rhythm_sensitivity: None,
            },
            pitch: PitchCoreConfig::PitchHillClimb {
                neighbor_step_cents: None,
                tessitura_gravity: None,
                improvement_threshold: None,
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
            breath_gain_init: Some(0.05),
            ..Default::default()
        },
        tag: None,
    };
    let metadata = AgentMetadata {
        id: 1,
        tag: None,
        group_idx: 0,
        member_idx: 0,
    };
    let mut agent = cfg.spawn(1, 0, metadata, fs, 0);
    agent.target_pitch_log2 = target_log2;

    let mut buffer = vec![0.0f32; 1024];
    let dt_sec = buffer.len() as f32 / fs;
    agent.render_wave(&mut buffer, fs, 0, dt_sec, &landscape, 1.0);

    let mut signature = 0u64;
    for sample in buffer {
        signature = mix_signature(signature, sample.to_bits());
    }

    println!("render signature: {signature:016x}");
    assert_eq!(signature, 0x3f6d_8c8d_dc30_d84c);
}

#[test]
fn render_wave_uses_dt_per_sample_for_seq_core() {
    let fs = 48_000.0;
    let mut agent = Individual {
        id: 1,
        metadata: AgentMetadata {
            id: 1,
            tag: None,
            group_idx: 0,
            member_idx: 0,
        },
        articulation: super::individual::ArticulationWrapper::new(
            super::individual::AnyArticulationCore::Seq(SequencedCore {
                timer: 0.0,
                duration: 0.1,
                env_level: 0.0,
            }),
            1.0,
        ),
        pitch: super::individual::AnyPitchCore::PitchHillClimb(
            super::individual::PitchHillClimbPitchCore::new(
                200.0,
                220.0f32.log2(),
                0.1,
                0.1,
                0.0,
                0.5,
            ),
        ),
        perceptual: PerceptualContext::from_config(
            &PerceptualConfig {
                tau_fast: None,
                tau_slow: None,
                w_boredom: None,
                w_familiarity: None,
                rho_self: None,
                boredom_gamma: None,
                self_smoothing_radius: None,
                silence_mass_epsilon: None,
            },
            0,
        ),
        planning: crate::life::scenario::PlanningConfig::default(),
        body: super::individual::AnySoundBody::Sine(super::individual::SineBody {
            freq_hz: 220.0,
            amp: 0.1,
            audio_phase: 0.0,
        }),
        last_signal: Default::default(),
        release_gain: 1.0,
        release_sec: 0.03,
        release_pending: false,
        target_pitch_log2: 220.0f32.log2(),
        integration_window: 2.0,
        accumulated_time: 0.0,
        last_theta_sample: 0.0,
        last_target_salience: 0.0,
        last_error_state: Default::default(),
        last_error_cents: 0.0,
        error_initialized: false,
        last_chosen_freq_hz: 220.0,
        next_intent_tick: 0,
        intent_seq: 0,
        self_confidence: 0.5,
        pred_intent_records: std::collections::VecDeque::new(),
        pred_intent_records_cap: 256,
        rng: rand::rngs::SmallRng::seed_from_u64(9),
    };
    let mut buffer = vec![0.0f32; 4800];
    let dt_sec = 0.1;
    let landscape = make_test_landscape(fs);
    agent.render_wave(&mut buffer, fs, 0, dt_sec, &landscape, 1.0);

    match &agent.articulation.core {
        super::individual::AnyArticulationCore::Seq(core) => {
            assert!((core.timer - 0.1).abs() < 1e-4);
        }
        _ => panic!("expected seq core"),
    }
}
