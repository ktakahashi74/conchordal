use crate::core::landscape::{Landscape, LandscapeFrame};
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::Timebase;
use crate::life::control::AgentControl;
use crate::life::individual::{
    AgentMetadata, AnyArticulationCore, ArticulationCore, Individual, SoundBody,
};
use crate::life::lifecycle::LifecycleConfig;
use crate::life::population::Population;
use crate::life::scenario::{Action, ArticulationCoreConfig, IndividualConfig};
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

fn control_with_pitch(freq: f32) -> AgentControl {
    let mut control = AgentControl::default();
    control.pitch.freq = freq.max(1.0);
    control
}

fn spawn_agent(freq: f32, assigned_id: u64) -> Individual {
    let control = control_with_pitch(freq);
    let cfg = IndividualConfig { control, tag: None };
    let meta = AgentMetadata {
        tag: None,
        group_idx: 0,
        member_idx: 0,
    };
    cfg.spawn(assigned_id, 0, meta, 48_000.0, 0)
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
        low.integration_window() > high.integration_window(),
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

    agent.set_accumulated_time_for_test(5.0);
    agent.set_theta_phase_state_for_test(6.0, true);
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.mag = 1.0;
    rhythms.theta.phase = 0.25;

    let before = agent.target_pitch_log2();
    agent.update_pitch_target(&rhythms, 0.01, &landscape);
    assert!(
        agent.target_pitch_log2() > before,
        "agent should move toward higher-scoring neighbor"
    );
}

#[test]
fn lock_mode_prevents_snapback() {
    let landscape = make_landscape();
    let mut pop = Population::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    pop.apply_action(
        Action::Spawn {
            tag: "setfreq_test".to_string(),
            count: 1,
            opts: None,
            patch: serde_json::json!({
                "pitch": { "freq": 220.0 }
            }),
        },
        &LandscapeFrame::default(),
        None,
    );

    let old_target = pop
        .individuals
        .first()
        .expect("agent exists")
        .target_pitch_log2();

    let new_freq: f32 = 440.0;
    let new_log = new_freq.log2();
    pop.apply_action(
        Action::Set {
            target: "setfreq_test".to_string(),
            patch: serde_json::json!({
                "pitch": { "mode": "lock", "freq": new_freq }
            }),
        },
        &LandscapeFrame::default(),
        None,
    );

    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.mag = 0.0;
    rhythms.theta.phase = 0.0;
    let dt_sec = 0.02;
    let steps = 50;
    let agent = pop.individuals.first_mut().expect("agent exists");
    for _ in 0..steps {
        agent.update_pitch_target(&rhythms, dt_sec, &landscape);
    }
    assert!(
        (agent.target_pitch_log2() - new_log).abs() < 1e-6,
        "target should remain locked to mode"
    );
    assert!(
        (agent.target_pitch_log2() - old_target).abs() > 0.5,
        "target should move away from old target"
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
        breath_gain_init: None,
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
    assert_eq!(signature, 0xc1be_e9d9_3e85_ee96);
}
