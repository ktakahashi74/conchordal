use crate::core::landscape::{Landscape, LandscapeFrame};
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::conductor::Conductor;
use crate::life::control::{ControlUpdate, PitchApplyMode, PitchMode, VoiceControl};
use crate::life::lifecycle::LifecycleConfig;
use crate::life::phonation_engine::{
    CandidatePoint, OnsetKick, OnsetRule, PhonationClock, ToneCmd,
};
use crate::life::population::Population;
use crate::life::scenario::{
    Action, ArticulationCoreConfig, DurationSpec, EnvelopeConfig, PhonationSpec, Scenario,
    SpawnSpec, TimedEvent, VoiceConfig, WhenSpec,
};
use crate::life::sound::RenderModulator;
use crate::life::voice::{
    AnyArticulationCore, ArticulationCore, ArticulationWrapper, PhonationBatch, SoundBody, Voice,
    VoiceMetadata,
};
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

fn spawn_spec_with_control(control: VoiceControl) -> SpawnSpec {
    SpawnSpec {
        control,
        articulation: ArticulationCoreConfig::default(),
    }
}

fn control_with_pitch(freq: f32) -> VoiceControl {
    let mut control = VoiceControl::default();
    control.pitch.freq = freq.max(1.0);
    control
}

fn spawn_voice(freq: f32, assigned_id: u64) -> Voice {
    let cfg = VoiceConfig {
        control: control_with_pitch(freq),
        articulation: ArticulationCoreConfig::default(),
    };
    let meta = VoiceMetadata {
        group_id: 0,
        member_idx: 0,
        generation: 0,
        parent_id: None,
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
            continuous_recharge_rate: None,
            continuous_recharge_score_low: None,
            continuous_recharge_score_high: None,
            dissonance_cost: None,
            envelope: EnvelopeConfig {
                attack_sec: 0.01,
                decay_sec: 0.05,
                sustain_level: 0.0,
                release_sec: 0.03,
            },
        },
        rhythm_freq: Some(4.0),
        rhythm_sensitivity: None,
        rhythm_coupling: crate::life::scenario::RhythmCouplingMode::TemporalOnly,
        rhythm_reward: None,
        breath_gain_init: None,
        k_omega: None,
        base_sigma: None,
        gate_thresholds: None,
        energy_cap: None,
    }
}

// ──── Population / Conductor ────

#[test]
fn population_spawn_and_release_removes_voice() {
    let mut pop = Population::new(test_timebase());
    let landscape = LandscapeFrame::default();
    let mut control = VoiceControl::default();
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
    assert_eq!(pop.voices.len(), 1);

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
    assert!(pop.voices.is_empty());
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
        control_update_mode: crate::life::scenario::ControlUpdateMode::SnapshotPhased,
        scaffold: crate::life::scenario::ScaffoldConfig::Off,
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
fn voice_lifecycle_decay_death() {
    let mut pop = Population::new(test_timebase());
    let landscape = LandscapeFrame::default();
    let mut control = VoiceControl::default();
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
        let voice = pop.voices.first_mut().expect("agent exists");
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
            k_omega: None,
            base_sigma: None,
            gate_thresholds: None,
            energy_cap: None,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let core = AnyArticulationCore::from_config(&core_cfg, fs, 1, &mut rng);
        voice.articulation = ArticulationWrapper::new(core, 1.0, false);
    }

    let dt = 0.01;
    let samples_per_hop = (fs * dt) as usize;
    let landscape_rt = make_landscape();

    for i in 0..100 {
        pop.advance(samples_per_hop, fs, i, dt, &landscape_rt);
        pop.cleanup_dead(i, dt, false, &landscape);
    }

    assert!(pop.voices.is_empty());
}

// ──── Pitch: Lock / Free / Glide ────

#[test]
fn lock_mode_advances_release_gain() {
    let fs = 48_000.0;
    let mut pop = Population::new(test_timebase());
    let mut control = VoiceControl::default();
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
    let voice = pop.voices.first_mut().expect("agent exists");
    voice.start_remove_fade(0.05);
    let gain_before = voice.release_gain();

    let dt = 0.01;
    let samples_per_hop = (fs * dt) as usize;
    let landscape = make_landscape();
    pop.advance(samples_per_hop, fs, 0, dt, &landscape);

    let gain_after = pop.voices[0].release_gain();
    assert!(gain_after < gain_before);
}

#[test]
fn lock_mode_keeps_pitch_and_target() {
    let fs = 48_000.0;
    let mut pop = Population::new(test_timebase());
    let mut control = VoiceControl::default();
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
        let voice = pop.voices.first().expect("agent exists");
        (voice.body.base_freq_hz(), voice.target_pitch_log2())
    };

    let dt = 0.01;
    let samples_per_hop = (fs * dt) as usize;
    let landscape = make_landscape();
    for i in 0..50 {
        pop.advance(samples_per_hop, fs, i, dt, &landscape);
    }

    let voice = pop.voices.first().expect("agent exists");
    assert!((voice.body.base_freq_hz() - freq_before).abs() <= 1e-6);
    assert!((voice.target_pitch_log2() - target_before).abs() <= 1e-6);
}

#[test]
fn lock_mode_prevents_snapback() {
    let landscape = Landscape::new(Log2Space::new(55.0, 4000.0, 48));
    let mut pop = Population::new(test_timebase());
    let mut control = VoiceControl::default();
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
        .voices
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
    let voice = pop.voices.first_mut().expect("agent exists");
    for _ in 0..steps {
        voice.update_pitch_target(&rhythms, dt_sec, &landscape, &[], &[]);
    }
    assert!(
        (voice.target_pitch_log2() - new_log).abs() < 1e-6,
        "target should remain locked to mode"
    );
    assert!(
        (voice.target_pitch_log2() - old_target).abs() > 0.5,
        "target should move away from old target"
    );
}

#[test]
fn free_mode_uses_freq_center_when_range_zero() {
    let mut control = VoiceControl::default();
    control.pitch.mode = PitchMode::Free;
    control.pitch.freq = 220.0;
    control.pitch.range_oct = 0.0;
    let mut voice = spawn_voice(220.0, 4);
    voice.effective_control = control;
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.phase = 0.1;
    rhythms.theta.mag = 1.0;
    let integration_window = voice.integration_window();
    voice.set_theta_phase_state_for_test(0.9, true);
    voice.set_accumulated_time_for_test(integration_window);
    let landscape = make_landscape();
    voice.update_pitch_target(&rhythms, 0.01, &landscape, &[], &[]);
    let expected = 220.0_f32.log2();
    assert!((voice.target_pitch_log2() - expected).abs() <= 1e-6);
}

#[test]
fn glide_mode_applies_pitch_without_gate_fade_delay() {
    let mut voice = spawn_voice(220.0, 62);
    let rhythms = NeuralRhythms::default();

    let current_log2 = voice.body.base_freq_hz().log2();
    let target_log2 = current_log2 + 1.0;
    voice.pitch_ctl.force_set_target_pitch_log2(target_log2);

    voice.effective_control.pitch.pitch_apply_mode = PitchApplyMode::GateSnap;
    let before_freq = voice.body.base_freq_hz();
    voice.update_articulation_autonomous(0.1, &rhythms);
    let snap_gate = voice.articulation.gate();
    let after_snap_freq = voice.body.base_freq_hz();
    assert!(
        snap_gate < 1.0,
        "gate-snap mode should fade down on large jump"
    );
    assert!((after_snap_freq - before_freq).abs() <= 1e-6);

    voice.effective_control.pitch.pitch_apply_mode = PitchApplyMode::Glide;
    voice.effective_control.pitch.pitch_glide_tau_sec = 0.05;
    voice.update_articulation_autonomous(0.1, &rhythms);
    let glide_gate = voice.articulation.gate();
    let after_glide_freq = voice.body.base_freq_hz();
    assert!(glide_gate >= 0.99, "glide mode should avoid gate fade-down");
    assert!(
        after_glide_freq > after_snap_freq,
        "glide mode should move pitch immediately"
    );
}

// ──── Pitch: proposal / adaptation / inertia ────

#[test]
fn adaptation_disabled_still_runs_pitch_proposal() {
    let mut control = VoiceControl::default();
    control.pitch.freq = 220.0;
    control.adaptation.enabled = false;
    let cfg = VoiceConfig {
        control,
        articulation: ArticulationCoreConfig::default(),
    };
    let meta = VoiceMetadata {
        group_id: 0,
        member_idx: 0,
        generation: 0,
        parent_id: None,
    };
    let mut voice = cfg.spawn(6, 0, meta, 48_000.0, 0);

    let integration_window = voice.integration_window();
    voice.set_accumulated_time_for_test(0.0);
    let mut rhythms = NeuralRhythms::default();
    let landscape = make_landscape();
    let mut proposal_path_ran = false;
    let dt = integration_window.max(0.05);
    for _ in 0..8 {
        voice.set_theta_phase_state_for_test(0.9, true);
        rhythms.theta.phase = 0.1;
        rhythms.theta.mag = 1.0;
        let before_accum = voice.accumulated_time_for_test();
        voice.update_pitch_target(&rhythms, dt, &landscape, &[], &[]);
        if before_accum + dt >= integration_window && voice.accumulated_time_for_test() <= 1e-6 {
            proposal_path_ran = true;
            break;
        }
    }

    assert!(
        proposal_path_ran,
        "expected pitch proposal path to run even when adaptation is disabled"
    );
}

#[test]
fn proposal_interval_decouples_from_integration_window() {
    let mut control = VoiceControl::default();
    control.pitch.freq = 220.0;
    control.pitch.proposal_interval_sec = Some(0.2);
    let cfg = VoiceConfig {
        control,
        articulation: ArticulationCoreConfig::default(),
    };
    let meta = VoiceMetadata {
        group_id: 0,
        member_idx: 0,
        generation: 0,
        parent_id: None,
    };
    let mut voice = cfg.spawn(61, 0, meta, 48_000.0, 0);
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.mag = 0.0;
    let landscape = make_landscape();
    voice.set_accumulated_time_for_test(0.0);
    for _ in 0..6 {
        voice.update_pitch_target(&rhythms, 0.05, &landscape, &[], &[]);
    }
    assert!(
        voice.accumulated_time_for_test() < 0.2,
        "proposal interval should trigger before integration-window horizon"
    );
}

#[test]
fn inertia_depends_on_frequency() {
    let landscape = make_landscape();
    let mut low = spawn_voice(60.0, 1);
    let mut high = spawn_voice(1000.0, 2);
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
    let mut voice = spawn_voice(220.0, 3);
    let n = landscape.consonance_field_score.len();
    landscape.subjective_intensity = vec![1.0; n];
    landscape.consonance_field_score.fill(0.0);
    landscape.consonance_field_level.fill(0.0);
    let idx_cur = landscape
        .space
        .index_of_freq(voice.body.base_freq_hz())
        .unwrap_or(0);
    landscape.consonance_field_score[idx_cur] = 0.0;
    landscape.consonance_field_level[idx_cur] = 0.0;
    let target_alt = voice.body.base_freq_hz() * 1.5;
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

    voice.set_accumulated_time_for_test(5.0);
    voice.set_theta_phase_state_for_test(6.0, true);
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.mag = 1.0;
    rhythms.theta.phase = 0.25;

    let before = voice.target_pitch_log2();
    voice.update_pitch_target(&rhythms, 0.01, &landscape, &[], &[]);
    assert!(
        voice.target_pitch_log2() > before,
        "agent should move toward higher-scoring neighbor"
    );
}

// ──── Phonation ────

#[test]
fn remove_pending_still_emits_note_offs() {
    let mut control = VoiceControl::default();
    control.pitch.freq = 220.0;
    control.phonation.spec = PhonationSpec {
        when: WhenSpec::Pulse {
            rate: 4.0,
            sync: 0.0,
            social: 0.0,
        },
        duration: DurationSpec::Gates(1),
    };
    let cfg = VoiceConfig {
        control,
        articulation: ArticulationCoreConfig::default(),
    };
    let meta = VoiceMetadata {
        group_id: 0,
        member_idx: 0,
        generation: 0,
        parent_id: None,
    };
    let mut voice = cfg.spawn(1, 0, meta, 48_000.0, 0);
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
        voice.tick_phonation_into(&tb, now, &rhythms, None, 0.0, 1.0, 1.0, &mut batch);
        if batch
            .cmds
            .iter()
            .any(|cmd| matches!(cmd, ToneCmd::On { .. }))
        {
            saw_note_on = true;
            break;
        }
        now = now.saturating_add(tb.hop as Tick);
        rhythms.advance_in_place(tb.hop as f32 / tb.fs);
    }
    assert!(saw_note_on, "expected at least one note-on before remove");

    voice.start_remove_fade(0.05);
    let mut saw_note_off = false;
    for _ in 0..40 {
        voice.tick_phonation_into(&tb, now, &rhythms, None, 0.0, 1.0, 1.0, &mut batch);
        if batch
            .cmds
            .iter()
            .any(|cmd| matches!(cmd, ToneCmd::Off { .. }))
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
    let mut control = VoiceControl::default();
    control.pitch.freq = 220.0;
    control.phonation.spec = PhonationSpec {
        when: WhenSpec::Once,
        duration: DurationSpec::Gates(1),
    };
    let cfg = VoiceConfig {
        control,
        articulation: sustain_entrain_articulation(),
    };
    let meta = VoiceMetadata {
        group_id: 0,
        member_idx: 0,
        generation: 0,
        parent_id: None,
    };
    let mut voice = cfg.spawn(77, 0, meta, 48_000.0, 0);
    voice.phonation_engine.clock = PhonationClock::Custom(Box::new(|_, out| {
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
    voice.phonation_engine.onset_rule =
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

    let energy_before = match &voice.articulation.core {
        AnyArticulationCore::Entrain(core) => core.energy,
        _ => panic!("expected entrain articulation"),
    };

    let mut batch = PhonationBatch::default();
    voice.tick_phonation_into(&tb, 0, &rhythms, None, 0.0, 1.0, 1.0, &mut batch);

    assert_eq!(batch.onsets.len(), 3);
    let energy_after = match &voice.articulation.core {
        AnyArticulationCore::Entrain(core) => {
            assert_eq!(core.state, crate::life::voice::ArticulationState::Attack);
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
    let mut control = VoiceControl::default();
    control.pitch.freq = 220.0;
    control.phonation.spec = PhonationSpec {
        when: WhenSpec::Once,
        duration: DurationSpec::WhileAlive,
    };
    let cfg = VoiceConfig {
        control,
        articulation: sustain_entrain_articulation(),
    };
    let meta = VoiceMetadata {
        group_id: 0,
        member_idx: 0,
        generation: 0,
        parent_id: None,
    };
    let mut voice = cfg.spawn(78, 0, meta, 48_000.0, 0);
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
    voice.tick_phonation_into(&tb, 0, &rhythms, None, 0.0, 1.0, 1.0, &mut batch);
    let tone = batch.tones.first().cloned().expect("expected hold note");
    let mut render_modulator = RenderModulator::from_spec(tone.render_modulator);

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
    let mut control = VoiceControl::default();
    control.pitch.freq = 220.0;
    control.phonation.spec = PhonationSpec {
        when: WhenSpec::Once,
        duration: DurationSpec::WhileAlive,
    };
    let cfg = VoiceConfig {
        control,
        articulation: sustain_entrain_articulation(),
    };
    let meta = VoiceMetadata {
        group_id: 0,
        member_idx: 0,
        generation: 0,
        parent_id: None,
    };
    let mut voice = cfg.spawn(79, 0, meta, 48_000.0, 0);
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
    voice.tick_phonation_into(&tb, 0, &rhythms, None, 0.0, 1.0, 1.0, &mut batch);
    let tone_id = batch.tones.first().expect("expected hold note").tone_id;

    match &mut voice.articulation.core {
        AnyArticulationCore::Entrain(core) => core.vitality_level = 0.25,
        _ => panic!("expected entrain articulation"),
    }
    let expected_amp = voice.compute_target_amp();
    let mut update_batch = PhonationBatch::default();
    voice.tick_phonation_into(
        &tb,
        tb.hop as Tick,
        &rhythms,
        None,
        0.0,
        1.0,
        1.0,
        &mut update_batch,
    );

    assert!(update_batch.tones.is_empty());
    let update = update_batch
        .cmds
        .iter()
        .find_map(|cmd| match cmd {
            ToneCmd::Update {
                tone_id: update_tone_id,
                update,
                ..
            } if *update_tone_id == tone_id => update.target_amp,
            _ => None,
        })
        .expect("expected target_amp update");
    assert!((update - expected_amp).abs() < 1e-6);
}

#[test]
fn hold_note_does_not_emit_update_below_amp_threshold() {
    let mut control = VoiceControl::default();
    control.pitch.freq = 220.0;
    control.phonation.spec = PhonationSpec {
        when: WhenSpec::Once,
        duration: DurationSpec::WhileAlive,
    };
    let cfg = VoiceConfig {
        control,
        articulation: sustain_entrain_articulation(),
    };
    let meta = VoiceMetadata {
        group_id: 0,
        member_idx: 0,
        generation: 0,
        parent_id: None,
    };
    let mut voice = cfg.spawn(80, 0, meta, 48_000.0, 0);
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
    voice.tick_phonation_into(&tb, 0, &rhythms, None, 0.0, 1.0, 1.0, &mut batch);
    let current_amp = voice.compute_target_amp();
    match &mut voice.articulation.core {
        AnyArticulationCore::Entrain(core) => {
            let base_amp = current_amp / core.vitality_level.max(1e-6);
            core.vitality_level = ((current_amp - 0.005) / base_amp).clamp(0.0, 1.0);
        }
        _ => panic!("expected entrain articulation"),
    }
    let mut update_batch = PhonationBatch::default();
    voice.tick_phonation_into(
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
            .any(|cmd| matches!(cmd, ToneCmd::Update { .. })),
        "expected no amp update below threshold"
    );
}

#[test]
fn tracked_note_is_removed_after_note_off_for_future_hops() {
    let mut control = VoiceControl::default();
    control.pitch.freq = 220.0;
    control.phonation.spec = PhonationSpec {
        when: WhenSpec::Once,
        duration: DurationSpec::WhileAlive,
    };
    let cfg = VoiceConfig {
        control,
        articulation: sustain_entrain_articulation(),
    };
    let meta = VoiceMetadata {
        group_id: 0,
        member_idx: 0,
        generation: 0,
        parent_id: None,
    };
    let mut voice = cfg.spawn(81, 0, meta, 48_000.0, 0);
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
    voice.tick_phonation_into(&tb, 0, &rhythms, None, 0.0, 1.0, 1.0, &mut batch);
    voice.remove_pending = true;
    let mut off_batch = PhonationBatch::default();
    voice.tick_phonation_into(
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
            .any(|cmd| matches!(cmd, ToneCmd::Off { .. })),
        "expected note-off when hold note is removed"
    );

    match &mut voice.articulation.core {
        AnyArticulationCore::Entrain(core) => core.vitality_level = 0.25,
        _ => panic!("expected entrain articulation"),
    }
    let mut later_batch = PhonationBatch::default();
    voice.tick_phonation_into(
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
            .any(|cmd| matches!(cmd, ToneCmd::Update { .. })),
        "expected no updates after tracked note was pruned"
    );
}

#[test]
fn render_modulator_snapshot_is_finite() {
    let mut control = VoiceControl::default();
    control.pitch.freq = 220.0;
    control.phonation.spec = PhonationSpec {
        when: WhenSpec::Pulse {
            rate: 4.0,
            sync: 0.0,
            social: 0.0,
        },
        duration: DurationSpec::Gates(1),
    };
    let cfg = VoiceConfig {
        control,
        articulation: ArticulationCoreConfig::default(),
    };
    let meta = VoiceMetadata {
        group_id: 0,
        member_idx: 0,
        generation: 0,
        parent_id: None,
    };
    let mut voice = cfg.spawn(41, 0, meta, 48_000.0, 0);
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
    let mut tone = None;
    for _ in 0..20 {
        voice.tick_phonation_into(&tb, now, &rhythms, None, 0.0, 1.0, 1.0, &mut batch);
        if let Some(first) = batch.tones.first() {
            tone = Some(first.clone());
            break;
        }
        now = now.saturating_add(tb.hop as Tick);
        rhythms.advance_in_place(tb.hop as f32 / tb.fs);
    }
    let tone = tone.expect("expected at least one rendered tone spec");
    let mut render_modulator = RenderModulator::from_spec(tone.render_modulator.clone());

    let mut render_rhythms = NeuralRhythms::default();
    render_rhythms.theta.freq_hz = 4.0;
    render_rhythms.theta.phase = 0.0;
    render_rhythms.env_open = 1.0;
    render_rhythms.env_level = 1.0;
    for _ in 0..64 {
        let signal = render_modulator.process(&render_rhythms, 1.0 / tb.fs);
        let sample = signal.amplitude * tone.amp;
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
        k_omega: None,
        base_sigma: None,
        gate_thresholds: None,
        energy_cap: None,
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
        let signal = articulation.process(consonance, 0.0, &rhythms, dt, 1.0);
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
