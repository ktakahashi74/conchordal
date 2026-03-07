use crate::core::landscape::{Landscape, LandscapeFrame};
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::conductor::Conductor;
use crate::life::control::{AgentControl, ControlUpdate, PitchApplyMode, PitchMode};
use crate::life::individual::{
    AgentMetadata, AnyArticulationCore, ArticulationCore, ArticulationWrapper, Individual,
    PhonationBatch, SoundBody,
};
use crate::life::lifecycle::LifecycleConfig;
use crate::life::phonation_engine::{
    CandidatePoint, NoteCmd, OnsetKick, OnsetRule, PhonationClock,
};
use crate::life::population::Population;
use crate::life::scenario::{
    Action, ArticulationCoreConfig, DurationSpec, EnvelopeConfig, IndividualConfig, PhonationSpec,
    Scenario, SpawnSpec, TimedEvent, WhenSpec,
};
use crate::life::sound::RenderModulator;
use rand::SeedableRng;

fn test_timebase() -> Timebase {
    Timebase {
        fs: 48_000.0,
        hop: 64,
    }
}

fn make_landscape() -> Landscape {
    let space = Log2Space::new(55.0, 4000.0, 64);
    Landscape::new(space)
}

fn spawn_spec_with_control(control: AgentControl) -> SpawnSpec {
    SpawnSpec {
        control,
        articulation: ArticulationCoreConfig::default(),
    }
}

fn control_with_pitch(freq: f32) -> AgentControl {
    let mut control = AgentControl::default();
    control.pitch.freq = freq.max(1.0);
    control
}

fn spawn_agent(freq: f32, assigned_id: u64) -> Individual {
    let cfg = IndividualConfig {
        control: control_with_pitch(freq),
        articulation: ArticulationCoreConfig::default(),
    };
    let meta = AgentMetadata {
        group_id: 0,
        member_idx: 0,
    };
    cfg.spawn(assigned_id, 0, meta, 48_000.0, 0)
}

fn sustain_entrain_articulation() -> ArticulationCoreConfig {
    ArticulationCoreConfig::Entrain {
        lifecycle: LifecycleConfig::Sustain {
            initial_energy: 0.2,
            metabolism_rate: 0.0,
            recharge_rate: Some(0.5),
            action_cost: Some(0.1),
            envelope: EnvelopeConfig {
                attack_sec: 0.01,
                decay_sec: 0.05,
                sustain_level: 0.0,
            },
        },
        rhythm_freq: Some(4.0),
        rhythm_sensitivity: None,
        rhythm_coupling: crate::life::scenario::RhythmCouplingMode::TemporalOnly,
        rhythm_reward: None,
        breath_gain_init: None,
    }
}

// ──── Population / Conductor ────

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
        Action::ReleaseGroup {
            group_id: 1,
            fade_sec: 0.05,
        },
        &landscape,
        None,
    );

    let fs = test_timebase().fs;
    let dt = 0.1;
    let samples_per_hop = (fs * dt) as usize;
    let landscape_rt = make_landscape();
    pop.advance(samples_per_hop, fs, 0, dt, &landscape_rt);
    pop.cleanup_dead(0, dt, false, &landscape);
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
    conductor.dispatch_until(
        0.5,
        0,
        &landscape,
        None::<&mut crate::core::stream::analysis::AnalysisStream>,
        &mut pop,
    );
    assert!(!pop.abort_requested);

    conductor.dispatch_until(
        1.1,
        100,
        &landscape,
        None::<&mut crate::core::stream::analysis::AnalysisStream>,
        &mut pop,
    );
    assert!(pop.abort_requested);
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
            rhythm_coupling: crate::life::scenario::RhythmCouplingMode::TemporalOnly,
            rhythm_reward: None,
            breath_gain_init: None,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let core = AnyArticulationCore::from_config(&core_cfg, fs, 1, &mut rng);
        agent.articulation = ArticulationWrapper::new(core, 1.0, false);
    }

    let dt = 0.01;
    let samples_per_hop = (fs * dt) as usize;
    let landscape_rt = make_landscape();

    for i in 0..100 {
        pop.advance(samples_per_hop, fs, i, dt, &landscape_rt);
        pop.cleanup_dead(i, dt, false, &landscape);
    }

    assert!(pop.individuals.is_empty());
}

// ──── Pitch: Lock / Free / Glide ────

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
    let landscape = make_landscape();
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
    let landscape = make_landscape();
    for i in 0..50 {
        pop.advance(samples_per_hop, fs, i, dt, &landscape);
    }

    let agent = pop.individuals.first().expect("agent exists");
    assert!((agent.body.base_freq_hz() - freq_before).abs() <= 1e-6);
    assert!((agent.target_pitch_log2() - target_before).abs() <= 1e-6);
}

#[test]
fn lock_mode_prevents_snapback() {
    let landscape = Landscape::new(Log2Space::new(55.0, 4000.0, 48));
    let mut pop = Population::new(test_timebase());
    let mut control = AgentControl::default();
    control.pitch.freq = 220.0;
    pop.apply_action(
        Action::Spawn {
            group_id: 1,
            ids: vec![1],
            spec: spawn_spec_with_control(control),
            strategy: None,
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
        Action::UpdateGroup {
            group_id: 1,
            patch: ControlUpdate {
                freq: Some(new_freq),
                ..ControlUpdate::default()
            },
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
        agent.update_pitch_target(&rhythms, dt_sec, &landscape, &[], &[]);
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
fn free_mode_uses_freq_center_when_range_zero() {
    let mut control = AgentControl::default();
    control.pitch.mode = PitchMode::Free;
    control.pitch.freq = 220.0;
    control.pitch.range_oct = 0.0;
    let mut agent = spawn_agent(220.0, 4);
    agent.effective_control = control;
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.phase = 0.1;
    rhythms.theta.mag = 1.0;
    let integration_window = agent.integration_window();
    agent.set_theta_phase_state_for_test(0.9, true);
    agent.set_accumulated_time_for_test(integration_window);
    let landscape = make_landscape();
    agent.update_pitch_target(&rhythms, 0.01, &landscape, &[], &[]);
    let expected = 220.0_f32.log2();
    assert!((agent.target_pitch_log2() - expected).abs() <= 1e-6);
}

#[test]
fn glide_mode_applies_pitch_without_gate_fade_delay() {
    let mut agent = spawn_agent(220.0, 62);
    let rhythms = NeuralRhythms::default();

    let current_log2 = agent.body.base_freq_hz().log2();
    let target_log2 = current_log2 + 1.0;
    agent.pitch_ctl.force_set_target_pitch_log2(target_log2);

    agent.effective_control.pitch.pitch_apply_mode = PitchApplyMode::GateSnap;
    let before_freq = agent.body.base_freq_hz();
    agent.update_articulation_autonomous(0.1, &rhythms);
    let snap_gate = agent.articulation.gate();
    let after_snap_freq = agent.body.base_freq_hz();
    assert!(
        snap_gate < 1.0,
        "gate-snap mode should fade down on large jump"
    );
    assert!((after_snap_freq - before_freq).abs() <= 1e-6);

    agent.effective_control.pitch.pitch_apply_mode = PitchApplyMode::Glide;
    agent.effective_control.pitch.pitch_glide_tau_sec = 0.05;
    agent.update_articulation_autonomous(0.1, &rhythms);
    let glide_gate = agent.articulation.gate();
    let after_glide_freq = agent.body.base_freq_hz();
    assert!(glide_gate >= 0.99, "glide mode should avoid gate fade-down");
    assert!(
        after_glide_freq > after_snap_freq,
        "glide mode should move pitch immediately"
    );
}

// ──── Pitch: proposal / perceptual / inertia ────

#[test]
fn perceptual_disabled_still_runs_pitch_proposal() {
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
    agent.set_accumulated_time_for_test(0.0);
    let mut rhythms = NeuralRhythms::default();
    let landscape = make_landscape();
    let mut proposal_path_ran = false;
    let dt = integration_window.max(0.05);
    for _ in 0..8 {
        agent.set_theta_phase_state_for_test(0.9, true);
        rhythms.theta.phase = 0.1;
        rhythms.theta.mag = 1.0;
        let before_accum = agent.accumulated_time_for_test();
        agent.update_pitch_target(&rhythms, dt, &landscape, &[], &[]);
        if before_accum + dt >= integration_window && agent.accumulated_time_for_test() <= 1e-6 {
            proposal_path_ran = true;
            break;
        }
    }

    assert!(
        proposal_path_ran,
        "expected pitch proposal path to run even when perceptual is disabled"
    );
}

#[test]
fn proposal_interval_decouples_from_integration_window() {
    let mut control = AgentControl::default();
    control.pitch.freq = 220.0;
    control.pitch.proposal_interval_sec = Some(0.2);
    let cfg = IndividualConfig {
        control,
        articulation: ArticulationCoreConfig::default(),
    };
    let meta = AgentMetadata {
        group_id: 0,
        member_idx: 0,
    };
    let mut agent = cfg.spawn(61, 0, meta, 48_000.0, 0);
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.mag = 0.0;
    let landscape = make_landscape();
    agent.set_accumulated_time_for_test(0.0);
    for _ in 0..6 {
        agent.update_pitch_target(&rhythms, 0.05, &landscape, &[], &[]);
    }
    assert!(
        agent.accumulated_time_for_test() < 0.2,
        "proposal interval should trigger before integration-window horizon"
    );
}

#[test]
fn inertia_depends_on_frequency() {
    let landscape = make_landscape();
    let mut low = spawn_agent(60.0, 1);
    let mut high = spawn_agent(1000.0, 2);
    let rhythms = NeuralRhythms::default();

    low.update_pitch_target(&rhythms, 0.01, &landscape, &[], &[]);
    high.update_pitch_target(&rhythms, 0.01, &landscape, &[], &[]);

    assert!(
        low.integration_window() > high.integration_window(),
        "expected heavier (low) pitch to integrate longer than high pitch"
    );
}

#[test]
fn scan_moves_toward_higher_scoring_neighbor() {
    let mut landscape = make_landscape();
    let mut agent = spawn_agent(220.0, 3);
    let n = landscape.consonance_field_score.len();
    landscape.subjective_intensity = vec![1.0; n];
    landscape.consonance_field_score.fill(0.0);
    landscape.consonance_field_level.fill(0.0);
    let idx_cur = landscape
        .space
        .index_of_freq(agent.body.base_freq_hz())
        .unwrap_or(0);
    landscape.consonance_field_score[idx_cur] = 0.0;
    landscape.consonance_field_level[idx_cur] = 0.0;
    let target_alt = agent.body.base_freq_hz() * 1.5;
    if let Some(idx_alt) = landscape.space.index_of_freq(target_alt) {
        let lo = idx_alt.saturating_sub(1);
        let hi = (idx_alt + 1).min(n.saturating_sub(1));
        for idx in lo..=hi {
            if let Some(c) = landscape.consonance_field_score.get_mut(idx) {
                *c = 1.0;
            }
            if let Some(c) = landscape.consonance_field_level.get_mut(idx) {
                *c = 1.0;
            }
        }
    }

    agent.set_accumulated_time_for_test(5.0);
    agent.set_theta_phase_state_for_test(6.0, true);
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.mag = 1.0;
    rhythms.theta.phase = 0.25;

    let before = agent.target_pitch_log2();
    agent.update_pitch_target(&rhythms, 0.01, &landscape, &[], &[]);
    assert!(
        agent.target_pitch_log2() > before,
        "agent should move toward higher-scoring neighbor"
    );
}

// ──── Phonation ────

#[test]
fn remove_pending_still_emits_note_offs() {
    let mut control = AgentControl::default();
    control.pitch.freq = 220.0;
    control.phonation.spec = PhonationSpec {
        when: WhenSpec::Pulse {
            rate: 4.0,
            sync: 0.0,
            social: 0.0,
        },
        duration: DurationSpec::Gates(1),
    };
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
        agent.tick_phonation_into(&tb, now, &rhythms, None, 0.0, 1.0, 1.0, &mut batch);
        if batch
            .cmds
            .iter()
            .any(|cmd| matches!(cmd, NoteCmd::NoteOn { .. }))
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
        agent.tick_phonation_into(&tb, now, &rhythms, None, 0.0, 1.0, 1.0, &mut batch);
        if batch
            .cmds
            .iter()
            .any(|cmd| matches!(cmd, NoteCmd::NoteOff { .. }))
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
fn tick_phonation_into_gated_bridges_each_onset_to_body_articulation() {
    let mut control = AgentControl::default();
    control.pitch.freq = 220.0;
    control.phonation.spec = PhonationSpec {
        when: WhenSpec::Once,
        duration: DurationSpec::Gates(1),
    };
    let cfg = IndividualConfig {
        control,
        articulation: sustain_entrain_articulation(),
    };
    let meta = AgentMetadata {
        group_id: 0,
        member_idx: 0,
    };
    let mut agent = cfg.spawn(77, 0, meta, 48_000.0, 0);
    agent.phonation_engine.clock = PhonationClock::Custom(Box::new(|_, out| {
        out.extend([
            CandidatePoint {
                tick: 0,
                gate: 0,
                theta_pos: 0.0,
                phase_in_gate: 0.0,
            },
            CandidatePoint {
                tick: 10,
                gate: 1,
                theta_pos: 1.0,
                phase_in_gate: 0.0,
            },
            CandidatePoint {
                tick: 20,
                gate: 2,
                theta_pos: 2.0,
                phase_in_gate: 0.0,
            },
        ]);
    }));
    agent.phonation_engine.onset_rule =
        OnsetRule::Custom(Box::new(|_, _| Some(OnsetKick { strength: 1.0 })));
    let tb = Timebase {
        fs: 48_000.0,
        hop: 48_001,
    };
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.freq_hz = 4.0;
    rhythms.theta.phase = 0.0;
    rhythms.theta.mag = 1.0;
    rhythms.theta.alpha = 1.0;
    rhythms.theta.beta = 0.2;
    rhythms.env_open = 1.0;
    rhythms.env_level = 1.0;

    let energy_before = match &agent.articulation.core {
        AnyArticulationCore::Entrain(core) => core.energy,
        _ => panic!("expected entrain articulation"),
    };

    let mut batch = PhonationBatch::default();
    agent.tick_phonation_into(&tb, 0, &rhythms, None, 0.0, 1.0, 1.0, &mut batch);

    assert_eq!(batch.onsets.len(), 3);
    let energy_after = match &agent.articulation.core {
        AnyArticulationCore::Entrain(core) => {
            assert_eq!(
                core.state,
                crate::life::individual::ArticulationState::Attack
            );
            core.energy
        }
        _ => panic!("expected entrain articulation"),
    };
    let expected = energy_before + batch.onsets.len() as f32 * 0.4;
    assert!(
        (energy_after - expected).abs() < 1e-4,
        "expected bridged energy {expected}, got {energy_after}"
    );
}

#[test]
fn hold_mode_renderer_still_pulses_after_render_clone_removal() {
    let mut control = AgentControl::default();
    control.pitch.freq = 220.0;
    control.phonation.spec = PhonationSpec {
        when: WhenSpec::Once,
        duration: DurationSpec::WhileAlive,
    };
    let cfg = IndividualConfig {
        control,
        articulation: sustain_entrain_articulation(),
    };
    let meta = AgentMetadata {
        group_id: 0,
        member_idx: 0,
    };
    let mut agent = cfg.spawn(78, 0, meta, 48_000.0, 0);
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.freq_hz = 4.0;
    rhythms.theta.phase = 0.0;
    rhythms.theta.mag = 1.0;
    rhythms.theta.alpha = 1.0;
    rhythms.theta.beta = 0.2;
    rhythms.env_open = 1.0;
    rhythms.env_level = 1.0;

    let mut batch = PhonationBatch::default();
    agent.tick_phonation_into(&tb, 0, &rhythms, None, 0.0, 1.0, 1.0, &mut batch);
    let note = batch.notes.first().cloned().expect("expected hold note");
    let mut render_modulator = RenderModulator::from_spec(note.render_modulator);

    let dt = 0.001;
    let mut render_rhythms = rhythms;
    let mut rises = 0u32;
    let mut prev_active = false;
    for _ in 0..2_000 {
        let signal = render_modulator.process(&render_rhythms, dt);
        if signal.is_active && !prev_active {
            rises += 1;
        }
        prev_active = signal.is_active;
        render_rhythms.advance_in_place(dt);
    }

    assert!(rises >= 2, "expected repeated hold pulses, got {rises}");
}

#[test]
fn hold_note_emits_target_amp_update_when_authority_amp_changes() {
    let mut control = AgentControl::default();
    control.pitch.freq = 220.0;
    control.phonation.spec = PhonationSpec {
        when: WhenSpec::Once,
        duration: DurationSpec::WhileAlive,
    };
    let cfg = IndividualConfig {
        control,
        articulation: sustain_entrain_articulation(),
    };
    let meta = AgentMetadata {
        group_id: 0,
        member_idx: 0,
    };
    let mut agent = cfg.spawn(79, 0, meta, 48_000.0, 0);
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.freq_hz = 4.0;
    rhythms.theta.phase = 0.0;
    rhythms.theta.mag = 1.0;
    rhythms.theta.alpha = 1.0;
    rhythms.theta.beta = 0.2;
    rhythms.env_open = 1.0;
    rhythms.env_level = 1.0;

    let mut batch = PhonationBatch::default();
    agent.tick_phonation_into(&tb, 0, &rhythms, None, 0.0, 1.0, 1.0, &mut batch);
    let note_id = batch.notes.first().expect("expected hold note").note_id;

    match &mut agent.articulation.core {
        AnyArticulationCore::Entrain(core) => core.vitality_level = 0.25,
        _ => panic!("expected entrain articulation"),
    }
    let expected_amp = agent.compute_target_amp();
    let mut update_batch = PhonationBatch::default();
    agent.tick_phonation_into(
        &tb,
        tb.hop as Tick,
        &rhythms,
        None,
        0.0,
        1.0,
        1.0,
        &mut update_batch,
    );

    assert!(update_batch.notes.is_empty());
    let update = update_batch
        .cmds
        .iter()
        .find_map(|cmd| match cmd {
            NoteCmd::Update {
                note_id: update_note_id,
                update,
                ..
            } if *update_note_id == note_id => update.target_amp,
            _ => None,
        })
        .expect("expected target_amp update");
    assert!((update - expected_amp).abs() < 1e-6);
}

#[test]
fn hold_note_does_not_emit_update_below_amp_threshold() {
    let mut control = AgentControl::default();
    control.pitch.freq = 220.0;
    control.phonation.spec = PhonationSpec {
        when: WhenSpec::Once,
        duration: DurationSpec::WhileAlive,
    };
    let cfg = IndividualConfig {
        control,
        articulation: sustain_entrain_articulation(),
    };
    let meta = AgentMetadata {
        group_id: 0,
        member_idx: 0,
    };
    let mut agent = cfg.spawn(80, 0, meta, 48_000.0, 0);
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.freq_hz = 4.0;
    rhythms.theta.phase = 0.0;
    rhythms.theta.mag = 1.0;
    rhythms.theta.alpha = 1.0;
    rhythms.theta.beta = 0.2;
    rhythms.env_open = 1.0;
    rhythms.env_level = 1.0;

    let mut batch = PhonationBatch::default();
    agent.tick_phonation_into(&tb, 0, &rhythms, None, 0.0, 1.0, 1.0, &mut batch);
    let current_amp = agent.compute_target_amp();
    match &mut agent.articulation.core {
        AnyArticulationCore::Entrain(core) => {
            let base_amp = current_amp / core.vitality_level.max(1e-6);
            core.vitality_level = ((current_amp - 0.005) / base_amp).clamp(0.0, 1.0);
        }
        _ => panic!("expected entrain articulation"),
    }
    let mut update_batch = PhonationBatch::default();
    agent.tick_phonation_into(
        &tb,
        tb.hop as Tick,
        &rhythms,
        None,
        0.0,
        1.0,
        1.0,
        &mut update_batch,
    );

    assert!(
        !update_batch
            .cmds
            .iter()
            .any(|cmd| matches!(cmd, NoteCmd::Update { .. })),
        "expected no amp update below threshold"
    );
}

#[test]
fn tracked_note_is_removed_after_note_off_for_future_hops() {
    let mut control = AgentControl::default();
    control.pitch.freq = 220.0;
    control.phonation.spec = PhonationSpec {
        when: WhenSpec::Once,
        duration: DurationSpec::WhileAlive,
    };
    let cfg = IndividualConfig {
        control,
        articulation: sustain_entrain_articulation(),
    };
    let meta = AgentMetadata {
        group_id: 0,
        member_idx: 0,
    };
    let mut agent = cfg.spawn(81, 0, meta, 48_000.0, 0);
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.freq_hz = 4.0;
    rhythms.theta.phase = 0.0;
    rhythms.theta.mag = 1.0;
    rhythms.theta.alpha = 1.0;
    rhythms.theta.beta = 0.2;
    rhythms.env_open = 1.0;
    rhythms.env_level = 1.0;

    let mut batch = PhonationBatch::default();
    agent.tick_phonation_into(&tb, 0, &rhythms, None, 0.0, 1.0, 1.0, &mut batch);
    agent.remove_pending = true;
    let mut off_batch = PhonationBatch::default();
    agent.tick_phonation_into(
        &tb,
        tb.hop as Tick,
        &rhythms,
        None,
        0.0,
        1.0,
        1.0,
        &mut off_batch,
    );
    assert!(
        off_batch
            .cmds
            .iter()
            .any(|cmd| matches!(cmd, NoteCmd::NoteOff { .. })),
        "expected note-off when hold note is removed"
    );

    match &mut agent.articulation.core {
        AnyArticulationCore::Entrain(core) => core.vitality_level = 0.25,
        _ => panic!("expected entrain articulation"),
    }
    let mut later_batch = PhonationBatch::default();
    agent.tick_phonation_into(
        &tb,
        (tb.hop as Tick) * 2,
        &rhythms,
        None,
        0.0,
        1.0,
        1.0,
        &mut later_batch,
    );
    assert!(
        !later_batch
            .cmds
            .iter()
            .any(|cmd| matches!(cmd, NoteCmd::Update { .. })),
        "expected no updates after tracked note was pruned"
    );
}

#[test]
fn render_modulator_snapshot_is_finite() {
    let mut control = AgentControl::default();
    control.pitch.freq = 220.0;
    control.phonation.spec = PhonationSpec {
        when: WhenSpec::Pulse {
            rate: 4.0,
            sync: 0.0,
            social: 0.0,
        },
        duration: DurationSpec::Gates(1),
    };
    let cfg = IndividualConfig {
        control,
        articulation: ArticulationCoreConfig::default(),
    };
    let meta = AgentMetadata {
        group_id: 0,
        member_idx: 0,
    };
    let mut agent = cfg.spawn(41, 0, meta, 48_000.0, 0);
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
    let mut note = None;
    for _ in 0..20 {
        agent.tick_phonation_into(&tb, now, &rhythms, None, 0.0, 1.0, 1.0, &mut batch);
        if let Some(first) = batch.notes.first() {
            note = Some(first.clone());
            break;
        }
        now = now.saturating_add(tb.hop as Tick);
        rhythms.advance_in_place(tb.hop as f32 / tb.fs);
    }
    let note = note.expect("expected at least one rendered note spec");
    let mut render_modulator = RenderModulator::from_spec(note.render_modulator.clone());

    let mut render_rhythms = NeuralRhythms::default();
    render_rhythms.theta.freq_hz = 4.0;
    render_rhythms.theta.phase = 0.0;
    render_rhythms.env_open = 1.0;
    render_rhythms.env_level = 1.0;
    for _ in 0..64 {
        let signal = render_modulator.process(&render_rhythms, 1.0 / tb.fs);
        let sample = signal.amplitude * note.amp;
        assert!(sample.is_finite());
        render_rhythms.advance_in_place(1.0 / tb.fs);
    }
}

// ──── Articulation ────

fn mix_signature(mut acc: u64, value: u32) -> u64 {
    acc ^= value as u64;
    acc = acc.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    acc
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
        rhythm_coupling: crate::life::scenario::RhythmCouplingMode::TemporalOnly,
        rhythm_reward: None,
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
    assert_eq!(signature, 0xc1e9_43c6_d8f8_6b6d);
}
