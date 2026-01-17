use super::conductor::Conductor;
use super::individual::{AgentMetadata, PhonationBatch, SoundBody};
use super::population::Population;
use super::scenario::{
    Action, ArticulationCoreConfig, IndividualConfig, Scenario, SpawnSpec, TimedEvent,
};
use crate::core::landscape::{Landscape, LandscapeFrame};
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::control::{AgentControl, PhonationType, PitchMode};
use crate::life::lifecycle::LifecycleConfig;
use crate::life::phonation_engine::{OnsetEvent, PhonationCmd};
use rand::SeedableRng;

fn test_timebase() -> Timebase {
    Timebase {
        fs: 48_000.0,
        hop: 64,
    }
}

fn make_test_landscape(_fs: f32) -> Landscape {
    let space = Log2Space::new(55.0, 4000.0, 64);
    Landscape::new(space)
}

fn spawn_spec_with_control(control: AgentControl) -> SpawnSpec {
    SpawnSpec {
        control,
        articulation: ArticulationCoreConfig::default(),
    }
}

#[test]
fn population_spawn_and_release_removes_agent() {
    let mut pop = Population::new(test_timebase());
    let landscape = LandscapeFrame::default();
    let mut control = AgentControl::default();
    control.pitch.freq = 440.0;

    pop.apply_action(
        Action::Spawn {
            group_id: 1,
            ids: vec![1],
            spec: spawn_spec_with_control(control),
            strategy: None,
        },
        &landscape,
        None,
    );
    assert_eq!(pop.individuals.len(), 1);

    pop.apply_action(
        Action::Release {
            group_id: 1,
            ids: vec![1],
            fade_sec: 0.05,
        },
        &landscape,
        None,
    );

    let fs = test_timebase().fs;
    let dt = 0.1;
    let samples_per_hop = (fs * dt) as usize;
    let landscape_rt = make_test_landscape(fs);
    pop.advance(samples_per_hop, fs, 0, dt, &landscape_rt);
    pop.cleanup_dead(0, dt, false);
    assert!(pop.individuals.is_empty());
}

#[test]
fn conductor_dispatches_finish_on_time() {
    let event = TimedEvent {
        time: 1.0,
        order: 1,
        actions: vec![Action::Finish],
    };
    let scenario = Scenario {
        seed: 0,
        scene_markers: Vec::new(),
        events: vec![event],
        duration_sec: 2.0,
    };

    let mut conductor = Conductor::from_scenario(scenario);
    let mut pop = Population::new(test_timebase());
    let landscape = LandscapeFrame::default();
    let space = Log2Space::new(1.0, 2.0, 1);
    let mut world = crate::life::world_model::WorldModel::new(
        Timebase {
            fs: 48_000.0,
            hop: 512,
        },
        space,
    );

    conductor.dispatch_until(
        0.5,
        0,
        &landscape,
        None::<&mut crate::core::stream::roughness::RoughnessStream>,
        &mut pop,
        &mut world,
    );
    assert!(!pop.abort_requested);

    conductor.dispatch_until(
        1.1,
        100,
        &landscape,
        None::<&mut crate::core::stream::roughness::RoughnessStream>,
        &mut pop,
        &mut world,
    );
    assert!(pop.abort_requested);
}

#[test]
fn lock_mode_advances_release_gain() {
    let fs = 48_000.0;
    let mut pop = Population::new(test_timebase());
    let mut control = AgentControl::default();
    control.pitch.freq = 440.0;
    control.pitch.mode = PitchMode::Lock;
    pop.apply_action(
        Action::Spawn {
            group_id: 2,
            ids: vec![2],
            spec: spawn_spec_with_control(control),
            strategy: None,
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
fn lock_mode_keeps_pitch_and_target() {
    let fs = 48_000.0;
    let mut pop = Population::new(test_timebase());
    let mut control = AgentControl::default();
    control.pitch.freq = 440.0;
    control.pitch.mode = PitchMode::Lock;
    pop.apply_action(
        Action::Spawn {
            group_id: 3,
            ids: vec![3],
            spec: spawn_spec_with_control(control),
            strategy: None,
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
    assert!((agent.body.base_freq_hz() - freq_before).abs() <= 1e-6);
    assert!((agent.target_pitch_log2() - target_before).abs() <= 1e-6);
}

#[test]
fn perceptual_disabled_does_not_propose_pitch() {
    let mut control = AgentControl::default();
    control.pitch.freq = 220.0;
    control.perceptual.enabled = false;
    let cfg = IndividualConfig {
        control,
        articulation: ArticulationCoreConfig::default(),
    };
    let meta = AgentMetadata {
        group_id: 0,
        member_idx: 0,
    };
    let mut agent = cfg.spawn(6, 0, meta, 48_000.0, 0);

    let integration_window = agent.integration_window();
    agent.set_accumulated_time_for_test(integration_window);
    agent.set_theta_phase_state_for_test(0.9, true);
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.phase = 0.1;
    rhythms.theta.mag = 1.0;

    let landscape = make_test_landscape(48_000.0);
    let before_target = agent.target_pitch_log2();
    let before_accum = agent.accumulated_time_for_test();
    agent.update_pitch_target(&rhythms, 0.01, &landscape);

    assert!((agent.target_pitch_log2() - before_target).abs() <= 1e-6);
    assert!(agent.accumulated_time_for_test() > before_accum);
}

#[test]
fn free_mode_uses_freq_center_when_range_zero() {
    let mut control = AgentControl::default();
    control.pitch.mode = PitchMode::Free;
    control.pitch.freq = 220.0;
    control.pitch.range_oct = 0.0;
    let cfg = IndividualConfig {
        control,
        articulation: ArticulationCoreConfig::default(),
    };
    let meta = AgentMetadata {
        group_id: 0,
        member_idx: 0,
    };
    let mut agent = cfg.spawn(4, 0, meta, 48_000.0, 0);
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.phase = 0.1;
    rhythms.theta.mag = 1.0;
    let integration_window = agent.integration_window();
    agent.set_theta_phase_state_for_test(0.9, true);
    agent.set_accumulated_time_for_test(integration_window);
    let landscape = make_test_landscape(48_000.0);
    agent.update_pitch_target(&rhythms, 0.01, &landscape);
    let expected = 220.0_f32.log2();
    assert!((agent.target_pitch_log2() - expected).abs() <= 1e-6);
}

#[test]
fn remove_pending_still_emits_note_offs() {
    let mut control = AgentControl::default();
    control.pitch.freq = 220.0;
    control.phonation.r#type = PhonationType::Interval;
    control.phonation.density = 1.0;
    control.phonation.legato = 0.0;
    let cfg = IndividualConfig {
        control,
        articulation: ArticulationCoreConfig::default(),
    };
    let meta = AgentMetadata {
        group_id: 0,
        member_idx: 0,
    };
    let mut agent = cfg.spawn(1, 0, meta, 48_000.0, 0);
    let tb = Timebase {
        fs: 48_000.0,
        hop: 12_001,
    };
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.freq_hz = 4.0;
    rhythms.theta.phase = 0.0;
    rhythms.env_open = 1.0;
    rhythms.env_level = 1.0;
    let mut batch = PhonationBatch::default();
    let mut now: Tick = 0;
    let mut saw_note_on = false;
    for _ in 0..20 {
        agent.tick_phonation_into(&tb, now, &rhythms, None, 0.0, 1.0, &mut batch);
        if batch
            .cmds
            .iter()
            .any(|cmd| matches!(cmd, PhonationCmd::NoteOn { .. }))
        {
            saw_note_on = true;
            break;
        }
        now = now.saturating_add(tb.hop as Tick);
        rhythms.advance_in_place(tb.hop as f32 / tb.fs);
    }
    assert!(saw_note_on, "expected at least one note-on before remove");

    agent.start_remove_fade(0.05);
    let mut saw_note_off = false;
    for _ in 0..40 {
        agent.tick_phonation_into(&tb, now, &rhythms, None, 0.0, 1.0, &mut batch);
        if batch
            .cmds
            .iter()
            .any(|cmd| matches!(cmd, PhonationCmd::NoteOff { .. }))
        {
            saw_note_off = true;
            break;
        }
        now = now.saturating_add(tb.hop as Tick);
        rhythms.advance_in_place(tb.hop as f32 / tb.fs);
    }
    assert!(saw_note_off, "expected note-off during remove fade");
}

#[test]
fn agent_lifecycle_decay_death() {
    let mut pop = Population::new(test_timebase());
    let landscape = LandscapeFrame::default();
    let mut control = AgentControl::default();
    control.pitch.freq = 440.0;
    pop.apply_action(
        Action::Spawn {
            group_id: 9,
            ids: vec![9],
            spec: spawn_spec_with_control(control),
            strategy: None,
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

    let dt = 0.01;
    let samples_per_hop = (fs * dt) as usize;
    let landscape_rt = make_test_landscape(fs);

    for i in 0..100 {
        pop.advance(samples_per_hop, fs, i, dt, &landscape_rt);
        pop.cleanup_dead(i, dt, false);
    }

    assert!(pop.individuals.is_empty());
}
