use crate::core::landscape::Landscape;
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::life::individual::{AgentMetadata, AnyArticulationCore, ArticulationCore, AudioAgent, Individual, SoundBody};
use crate::life::lifecycle::LifecycleConfig;
use crate::life::perceptual::PerceptualConfig;
use crate::life::population::Population;
use crate::life::scenario::{Action, TargetRef};
use crate::life::scenario::{
    PitchCoreConfig, IndividualConfig, LifeConfig, SoundBodyConfig, ArticulationCoreConfig,
};
use rand::SeedableRng;

fn mix_signature(mut acc: u64, value: u32) -> u64 {
    acc ^= value as u64;
    acc = acc.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    acc
}

fn make_landscape() -> Landscape {
    let space = Log2Space::new(55.0, 4000.0, 48);
    Landscape::new(space)
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

    low.update_pitch_target(&rhythms, 0.01, &landscape);
    high.update_pitch_target(&rhythms, 0.01, &landscape);

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
    agent.update_pitch_target(&rhythms, 0.01, &landscape);
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
    agent.articulation.set_gate(1.0);

    let mut buffer = [0.0f32; 1];
    let dt_sec = 0.05;
    agent.render_wave(&mut buffer, 48_000.0, 0, dt_sec, &landscape, 1.0);

    assert!(
        agent.articulation.gate() < 1.0,
        "gate should fall before snapping to target"
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
    pop.apply_action(
        Action::AddAgent {
            id: 1,
            agent: agent_cfg,
        },
        &landscape,
        None,
    );

    let (old_target, old_freq) = {
        let agent = pop.individuals.first().expect("agent exists");
        (agent.target_pitch_log2, agent.body.base_freq_hz())
    };

    let new_freq: f32 = 440.0;
    let new_log = new_freq.log2();
    pop.apply_action(
        Action::SetFreq {
            target: TargetRef::Tag {
                tag: "setfreq_test".to_string(),
            },
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
        (agent.articulation.gate() - 1.0).abs() < 1e-6,
        "gate should reset to 1.0 on SetFreq"
    );

    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.mag = 0.0;
    rhythms.theta.phase = 0.0;
    let dt_sec = 0.02;
    let steps = 500;
    let agent = pop.individuals.first_mut().expect("agent exists");
    for _ in 0..steps {
        agent.update_pitch_target(&rhythms, dt_sec, &landscape);
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

#[test]
fn articulation_snapshot_kuramoto_decay_signature() {
    let fs = 48_000.0;
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);
    let core = ArticulationCoreConfig::Entrain {
        lifecycle: LifecycleConfig::Decay {
            initial_energy: 1.0,
            half_life_sec: 0.2,
            attack_sec: 0.1,
        },
        rhythm_freq: Some(6.0),
        rhythm_sensitivity: None,
    };
    let mut articulation = AnyArticulationCore::from_config(&core, fs, 7, &mut rng);
    let mut rhythms = NeuralRhythms {
        theta: crate::core::modulation::RhythmBand {
            phase: 0.0,
            freq_hz: 6.0,
            mag: 1.0,
            alpha: 1.0,
            beta: 0.2,
        },
        delta: crate::core::modulation::RhythmBand {
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
    let steps = 48_000;
    let mut signature = 0u64;
    let mut early_active = false;

    for i in 0..steps {
        let signal = articulation.process(consonance, &rhythms, dt, 1.0);
        if i < 10 && (signal.is_active || signal.amplitude > 0.0) {
            early_active = true;
        }
        signature = mix_signature(signature, signal.is_active as u32);
        signature = mix_signature(signature, signal.amplitude.to_bits());
        signature = mix_signature(signature, signal.relaxation.to_bits());
        signature = mix_signature(signature, signal.tension.to_bits());
        rhythms.advance_in_place(dt);
    }

    assert!(early_active, "expected early attack during decay lifecycle");
    println!("articulation decay signature: {signature:016x}");
    assert_eq!(signature, 0x59f2_e6d9_ebab_47f9);
}

#[test]
fn render_wave_snapshot_signature_with_forced_snap() {
    let fs = 48_000.0;
    let space = Log2Space::new(110.0, 880.0, 48);
    let mut landscape = Landscape::new(space.clone());
    landscape.consonance01.fill(0.1);
    let current_freq = 220.0;
    let target_freq: f32 = 440.0;
    let target_log2 = target_freq.log2();
    if let Some(idx) = landscape.space.index_of_log2(target_log2) {
        landscape.consonance01[idx] = 0.9;
    }

    let cfg = IndividualConfig {
        freq: current_freq,
        amp: 0.3,
        life: LifeConfig {
            body: SoundBodyConfig::Sine { phase: Some(0.25) },
            articulation: ArticulationCoreConfig::Entrain {
                lifecycle: LifecycleConfig::Decay {
                    initial_energy: 1.0,
                    half_life_sec: 0.2,
                    attack_sec: 0.1,
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
            breath_gain_init: Some(0.09),
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
    let mut agent = cfg.spawn(1, 0, metadata, fs);
    agent.target_pitch_log2 = target_log2;
    let before = agent.body.base_freq_hz();

    let mut buffer = [0.0f32; 1];
    let dt_sec = 0.02;
    agent.render_wave(&mut buffer, fs, 0, dt_sec, &landscape, 1.0);

    let after = agent.body.base_freq_hz();
    assert!(
        (after - target_freq).abs() < 1e-3,
        "expected forced snap to target pitch"
    );
    assert!(
        (after - before).abs() > 1.0,
        "expected pitch to move on forced snap"
    );

    let mut signature = 0u64;
    signature = mix_signature(signature, buffer[0].to_bits());
    println!("render forced snap signature: {signature:016x}");
    assert_eq!(signature, 0xdb56_414d_adec_e240);
}
