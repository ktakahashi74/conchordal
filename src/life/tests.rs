use super::conductor::Conductor;
use super::individual::{
    AgentMetadata, ArticulationCore, ArticulationSignal, ArticulationState, KuramotoCore,
    PinkNoise, PitchCore, Sensitivity, SequencedCore, SoundBody,
};
use super::population::Population;
use super::scenario::{
    Action, ArticulationCoreConfig, EnvelopeConfig, IndividualConfig, PhonationConnectConfig,
    PitchCoreConfig, Scenario, SceneMarker, TimedEvent,
};
use crate::core::landscape::Landscape;
use crate::core::landscape::LandscapeFrame;
use crate::core::log2space::Log2Space;
use crate::core::modulation::{NeuralRhythms, RhythmBand};
use crate::core::timebase::Timebase;
use crate::life::control::{AgentControl, BodyMethod};
use crate::life::lifecycle::LifecycleConfig;
use crate::life::perceptual::{FeaturesNow, PerceptualConfig, PerceptualContext};
use rand::SeedableRng;
use serde_json;

fn mix_signature(mut acc: u64, value: u32) -> u64 {
    acc ^= value as u64;
    acc = acc.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    acc
}

fn control_with_pitch(freq_hz: f32) -> AgentControl {
    let mut control = AgentControl::default();
    control.pitch.center_hz = freq_hz.max(1.0);
    control
}

fn test_timebase() -> Timebase {
    Timebase {
        fs: 48_000.0,
        hop: 64,
    }
}

#[test]
fn test_population_add_remove_agent() {
    // 1. Setup
    let mut pop = Population::new(test_timebase());
    let landscape = LandscapeFrame::default();

    assert_eq!(pop.individuals.len(), 0, "Population should start empty");

    // 2. Add Agent
    pop.apply_action(
        Action::Spawn {
            tag: "test_agent".to_string(),
            count: 1,
            patch: serde_json::json!({
                "pitch": { "center_hz": 440.0 }
            }),
        },
        &landscape,
        None,
    );
    assert_eq!(pop.individuals.len(), 1, "Agent should be added");

    // 3. Remove Agent
    pop.apply_action(
        Action::Remove {
            target: "test_agent".to_string(),
        },
        &landscape,
        None,
    );
    let fs = test_timebase().fs;
    let dt = 0.1;
    let samples_per_hop = (fs * dt) as usize;
    let landscape_rt = make_test_landscape(fs);
    pop.advance(samples_per_hop, fs, 0, dt, &landscape_rt);
    pop.process_frame(0, &landscape_rt.space, dt, false);
    assert_eq!(pop.individuals.len(), 0, "Agent should be removed");
}

#[test]
fn tag_selector_removes_matching_agents() {
    let mut pop = Population::new(test_timebase());
    let landscape = LandscapeFrame::default();

    pop.apply_action(
        Action::Spawn {
            tag: "test_group".to_string(),
            count: 2,
            patch: serde_json::json!({
                "pitch": { "center_hz": 330.0 }
            }),
        },
        &landscape,
        None,
    );
    assert_eq!(pop.individuals.len(), 2);

    pop.apply_action(
        Action::Remove {
            target: "test_group".to_string(),
        },
        &landscape,
        None,
    );
    let fs = test_timebase().fs;
    let dt = 0.1;
    let samples_per_hop = (fs * dt) as usize;
    let landscape_rt = make_test_landscape(fs);
    pop.advance(samples_per_hop, fs, 0, dt, &landscape_rt);
    pop.process_frame(0, &landscape_rt.space, dt, false);
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
    let mut pop = Population::new(test_timebase());
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
    let mut pop = Population::new(test_timebase());
    let landscape = LandscapeFrame::default(); // Dummy landscape

    let fs = 48_000.0;
    pop.apply_action(
        Action::Spawn {
            tag: "decay".to_string(),
            count: 1,
            patch: serde_json::json!({
                "pitch": { "center_hz": 440.0 }
            }),
        },
        &landscape,
        None,
    );
    let fs = 48_000.0;
    {
        let agent = pop.individuals.first_mut().expect("agent exists");
        let core_cfg = ArticulationCoreConfig::Entrain {
            lifecycle: LifecycleConfig::Decay {
                initial_energy: 1.0,
                half_life_sec: 0.05,
                attack_sec: 0.001,
            },
            rhythm_freq: None,
            rhythm_sensitivity: None,
            breath_gain_init: None,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let core = super::individual::AnyArticulationCore::from_config(&core_cfg, fs, 1, &mut rng);
        agent.articulation = super::individual::ArticulationWrapper::new(core, 1.0);
    }

    assert_eq!(pop.individuals.len(), 1, "Agent added");

    // 2. Simulate time passing via process_frame
    // We need to run enough frames for energy to drop below threshold (1e-4)
    // Energy starts at 1.0.
    // After 0.05s -> 0.5
    // After 0.10s -> 0.25
    // ...
    // After 1.0s -> ~0.0
    let dt = 0.01; // 10ms steps
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
fn motion_disabled_still_advances_release_gain() {
    let fs = 48_000.0;
    let mut pop = Population::new(test_timebase());
    pop.apply_action(
        Action::Spawn {
            tag: "silent".to_string(),
            count: 1,
            patch: serde_json::json!({
                "pitch": { "center_hz": 440.0 },
                "perceptual": { "enabled": false }
            }),
        },
        &LandscapeFrame::default(),
        None,
    );
    let agent = pop.individuals.first_mut().expect("agent exists");
    agent.start_remove_fade(0.05);
    let gain_before = agent.release_gain();

    let dt = 0.01;
    let samples_per_hop = (fs * dt) as usize;
    let landscape = make_test_landscape(fs);
    pop.advance(samples_per_hop, fs, 0, dt, &landscape);

    let gain_after = pop.individuals[0].release_gain();
    assert!(gain_after < gain_before);
}

#[test]
fn motion_disabled_does_not_change_pitch_or_target() {
    let fs = 48_000.0;
    let mut pop = Population::new(test_timebase());
    pop.apply_action(
        Action::Spawn {
            tag: "silent".to_string(),
            count: 1,
            patch: serde_json::json!({
                "pitch": { "center_hz": 440.0 },
                "perceptual": { "enabled": false }
            }),
        },
        &LandscapeFrame::default(),
        None,
    );

    let (freq_before, target_before) = {
        let agent = pop.individuals.first().expect("agent exists");
        (agent.body.base_freq_hz(), agent.target_pitch_log2())
    };

    let dt = 0.01;
    let samples_per_hop = (fs * dt) as usize;
    let landscape = make_test_landscape(fs);
    for i in 0..50 {
        pop.advance(samples_per_hop, fs, i, dt, &landscape);
    }

    let agent = pop.individuals.first().expect("agent exists");
    assert!(
        (agent.body.base_freq_hz() - freq_before).abs() <= 1e-6,
        "motion disabled should keep base frequency stable"
    );
    assert!(
        (agent.target_pitch_log2() - target_before).abs() <= 1e-6,
        "motion disabled should keep target pitch stable"
    );
}

#[test]
fn harmonic_render_spectrum_hits_expected_bins() {
    let mut control = control_with_pitch(55.0);
    control.body.method = BodyMethod::Harmonic;
    control.body.amp = 0.8;
    control.body.timbre.brightness = 1.0;
    control.body.timbre.inharmonic = 0.0;
    control.body.timbre.width = 0.1;
    control.body.timbre.motion = 0.5;
    let cfg = IndividualConfig { control, tag: None };
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
        "brightness should attenuate even harmonic"
    );
}

#[test]
fn lock_constraint_sets_target_pitch_log2() {
    let space = Log2Space::new(55.0, 880.0, 12);
    let landscape = LandscapeFrame::new(space.clone());
    let mut pop = Population::new(test_timebase());
    pop.apply_action(
        Action::Spawn {
            tag: "test_agent".to_string(),
            count: 1,
            patch: serde_json::json!({
                "pitch": { "center_hz": 220.0 }
            }),
        },
        &landscape,
        None,
    );

    pop.apply_action(
        Action::Set {
            target: "test_agent".to_string(),
            patch: serde_json::json!({
                "pitch": {
                    "constraint": { "mode": "lock", "freq_hz": 440.0 }
                }
            }),
        },
        &landscape,
        None,
    );

    let log_target = 440.0f32.log2();
    let agent = pop.individuals.first_mut().expect("agent exists");
    let rhythms = NeuralRhythms::default();
    agent.update_pitch_target(&rhythms, 0.01, &Landscape::new(space));
    assert!((agent.target_pitch_log2() - log_target).abs() < 1e-6);
}

#[test]
fn population_spectrum_uses_log2_space() {
    let space = Log2Space::new(55.0, 880.0, 12);
    let mut pop = Population::new(test_timebase());
    pop.apply_action(
        Action::Spawn {
            tag: "spec".to_string(),
            count: 1,
            patch: serde_json::json!({
                "pitch": { "center_hz": 55.0 }
            }),
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
fn agent_patch_rejects_unknown_fields() {
    let good = serde_json::json!({
        "body": { "amp": 0.25 },
        "pitch": { "center_hz": 220.0 },
        "perceptual": { "enabled": true }
    });
    let patch: crate::life::control::AgentPatch =
        serde_json::from_value(good).expect("agent patch parses");
    assert_eq!(patch.body.unwrap().amp, Some(0.25));

    let bad = serde_json::json!({
        "body": { "amp": 0.25, "unknown": 1.0 }
    });
    let err = serde_json::from_value::<crate::life::control::AgentPatch>(bad)
        .expect_err("unknown key should be rejected");
    assert!(err.to_string().contains("unknown"));
}

#[test]
fn phonation_connect_rejects_unknown_keys() {
    let err = serde_json::from_value::<PhonationConnectConfig>(serde_json::json!({
        "type": "fixed_gate",
        "mystery": 1
    }))
    .expect_err("unknown key should be rejected");
    assert!(err.to_string().contains("mystery"));
}

#[test]
fn agent_patch_clamps_and_coerces_numbers() {
    let patch: crate::life::control::AgentPatch = serde_json::from_value(serde_json::json!({
        "body": { "amp": 2 },
        "pitch": { "range_oct": 99.0 },
        "phonation": { "density": -1 }
    }))
    .expect("patch parses");
    let body = patch.body.expect("body patch");
    assert_eq!(body.amp, Some(1.0));
    let pitch = patch.pitch.expect("pitch patch");
    assert_eq!(pitch.range_oct, Some(6.0));
    let phonation = patch.phonation.expect("phonation patch");
    assert_eq!(phonation.density, Some(0.0));
}

#[test]
fn unset_restores_base_control() {
    let mut control = AgentControl::default();
    control.body.amp = 0.3;
    let cfg = IndividualConfig { control, tag: None };
    let meta = AgentMetadata {
        id: 1,
        tag: None,
        group_idx: 0,
        member_idx: 0,
    };
    let mut agent = cfg.spawn(meta.id, 0, meta, 48_000.0, 0);
    agent
        .apply_control_patch(serde_json::json!({ "body": { "amp": 0.8 } }))
        .expect("patch applies");
    assert!((agent.effective_control.body.amp - 0.8).abs() < 1e-6);
    agent.apply_unset_path("body.amp").expect("unset applies");
    assert!((agent.effective_control.body.amp - 0.3).abs() < 1e-6);
}

#[test]
fn constraint_mode_requires_freq() {
    let cfg = IndividualConfig {
        control: control_with_pitch(220.0),
        tag: None,
    };
    let meta = AgentMetadata {
        id: 2,
        tag: None,
        group_idx: 0,
        member_idx: 0,
    };
    let mut agent = cfg.spawn(meta.id, 0, meta, 48_000.0, 0);
    let err = agent.apply_control_patch(serde_json::json!({
        "pitch": { "constraint": { "mode": "lock" } }
    }));
    assert!(err.is_err());
}

#[test]
fn patch_rejects_type_switches() {
    let cfg = IndividualConfig {
        control: control_with_pitch(220.0),
        tag: None,
    };
    let meta = AgentMetadata {
        id: 3,
        tag: None,
        group_idx: 0,
        member_idx: 0,
    };
    let mut agent = cfg.spawn(meta.id, 0, meta, 48_000.0, 0);
    let err_body = agent.apply_control_patch(serde_json::json!({
        "body": { "method": "harmonic" }
    }));
    assert!(err_body.is_err());
    let err_phonation = agent.apply_control_patch(serde_json::json!({
        "phonation": { "type": "drone" }
    }));
    assert!(err_phonation.is_err());
}

#[test]
fn spawn_lock_constraint_sets_center_hz() {
    let mut pop = Population::new(test_timebase());
    let landscape = LandscapeFrame::default();
    pop.apply_action(
        Action::Spawn {
            tag: "lock".to_string(),
            count: 1,
            patch: serde_json::json!({
                "pitch": { "constraint": { "mode": "lock", "freq_hz": 440.0 } }
            }),
        },
        &landscape,
        None,
    );
    let agent = pop.individuals.first().expect("agent exists");
    assert!((agent.effective_control.pitch.center_hz - 440.0).abs() < 1e-6);
}

#[test]
fn remove_fade_reduces_gain_and_culls() {
    let fs = test_timebase().fs;
    let mut pop = Population::new(test_timebase());
    let landscape = LandscapeFrame::default();
    pop.apply_action(
        Action::Spawn {
            tag: "fade".to_string(),
            count: 1,
            patch: serde_json::json!({ "pitch": { "center_hz": 220.0 } }),
        },
        &landscape,
        None,
    );
    pop.apply_action(
        Action::Remove {
            target: "fade".to_string(),
        },
        &landscape,
        None,
    );
    let gain_before = pop.individuals[0].release_gain();
    let dt = 0.01;
    let samples_per_hop = (fs * dt) as usize;
    let landscape_rt = make_test_landscape(fs);
    pop.advance(samples_per_hop, fs, 0, dt, &landscape_rt);
    let gain_after = pop.individuals[0].release_gain();
    assert!(gain_after < gain_before);

    let dt_finish = 0.1;
    let samples_per_hop_finish = (fs * dt_finish) as usize;
    pop.advance(samples_per_hop_finish, fs, 1, dt_finish, &landscape_rt);
    pop.process_frame(1, &landscape_rt.space, dt_finish, false);
    assert!(pop.individuals.is_empty());
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
    let mut control = control_with_pitch(220.0);
    control.pitch.exploration = 0.2;
    control.pitch.persistence = 0.5;
    control.perceptual.adaptation = 0.5;
    control.perceptual.novelty_bias = 0.8;
    control.perceptual.self_focus = 0.15;
    let cfg = IndividualConfig { control, tag: None };
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
        seq_a.push(a.target_pitch_log2());
        seq_b.push(b.target_pitch_log2());
    }
    assert_eq!(seq_a, seq_b);
}

#[test]
fn theta_wrap_triggers_pitch_update_with_large_dt() {
    let mut pop = Population::new(test_timebase());
    let landscape = make_test_landscape(48_000.0);
    pop.apply_action(
        Action::Spawn {
            tag: "theta".to_string(),
            count: 1,
            patch: serde_json::json!({
                "pitch": { "center_hz": 440.0 }
            }),
        },
        &LandscapeFrame::default(),
        None,
    );
    let agent = pop.individuals.first_mut().expect("agent exists");
    let integration_window = agent.integration_window();
    agent.set_accumulated_time_for_test(integration_window + 1.0);
    agent.set_theta_phase_state_for_test(6.0, true);

    let rhythms = NeuralRhythms {
        theta: RhythmBand {
            phase: 0.1,
            freq_hz: 12.0,
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
        env_level: 1.0,
    };
    agent.update_pitch_target(&rhythms, 0.5, &landscape);
    assert_eq!(
        agent.accumulated_time_for_test(),
        0.0,
        "theta wrap should trigger a pitch update even for large dt"
    );
}

#[test]
fn kuramoto_locks_to_theta_phase() {
    let mut core = KuramotoCore {
        energy: 1.0,
        energy_cap: 1.0,
        vitality_exponent: 0.5,
        vitality_level: 1.0,
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
        decay_rate: 0.9,
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
        energy_cap: 1.0,
        vitality_exponent: 0.5,
        vitality_level: 1.0,
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
        decay_rate: 0.9,
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
    let dt = 0.01;
    // Convert per-step envelope tuning to per-second rates for k-rate updates.
    let attack_step = 0.1 / dt;
    let decay_rate = -0.9f32.ln() / dt;
    let mut core = KuramotoCore {
        energy: 1.0,
        energy_cap: 1.0,
        vitality_exponent: 0.5,
        vitality_level: 1.0,
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
        attack_step,
        decay_rate,
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
            energy_cap: 1.0,
            vitality_exponent: 0.5,
            vitality_level: 1.0,
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
            decay_rate: 0.2,
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
fn kuramoto_process_dt_invariance() {
    let base = KuramotoCore {
        energy: 1.0,
        energy_cap: 1.0,
        vitality_exponent: 0.5,
        vitality_level: 1.0,
        basal_cost: 0.1,
        action_cost: 0.0,
        recharge_rate: 0.0,
        sensitivity: Sensitivity {
            delta: 0.0,
            theta: 0.0,
            alpha: 0.0,
            beta: 0.0,
        },
        rhythm_phase: 1.0,
        rhythm_freq: 4.0,
        omega_rad: std::f32::consts::TAU * 4.0,
        phase_offset: 0.0,
        debug_id: 42,
        env_level: 1.0,
        state: ArticulationState::Decay,
        attack_step: 0.0,
        decay_rate: 1.0,
        retrigger: false,
        noise_1f: PinkNoise::new(5, 0.0),
        base_sigma: 0.0,
        beta_gain: 0.0,
        k_omega: 0.0,
        bootstrap_timer: 0.0,
        env_open_threshold: 1.0,
        env_level_min: 0.0,
        mag_threshold: 1.0,
        alpha_threshold: 1.0,
        beta_threshold: -1.0,
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
            freq_hz: 4.0,
            mag: 0.0,
            alpha: 0.0,
            beta: 0.0,
        },
        delta: RhythmBand {
            phase: 0.0,
            freq_hz: 1.0,
            mag: 0.0,
            alpha: 0.0,
            beta: 0.0,
        },
        env_open: 0.0,
        env_level: 0.0,
    };

    let total_sec = 0.2;
    let steps = 20;
    let dt = total_sec / steps as f32;

    let mut fine = base.clone();
    for _ in 0..steps {
        fine.process(0.0, &rhythms, dt, 1.0);
    }

    let mut coarse = base;
    coarse.process(0.0, &rhythms, total_sec, 1.0);

    let tol = 1e-3;
    assert_eq!(fine.state, coarse.state);
    assert!((fine.env_level - coarse.env_level).abs() < tol);
    assert!((fine.energy - coarse.energy).abs() < tol);
    assert!((fine.rhythm_phase - coarse.rhythm_phase).abs() < tol);
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
        breath_gain_init: None,
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
    assert_eq!(signature, 0x377c_d318_ee77_c649);
}
