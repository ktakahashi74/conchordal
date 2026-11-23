use super::*;
use crate::scenario::control::BodyMethod;
use crate::scenario::{
    CoupledTimingSpec, DurationSpec, PhonationSpec, PhonationTiming, RhythmRelation,
};

fn setup(
    body: BodyMethod,
    mode: PitchApplyMode,
    tie: f32,
    persistence: f32,
    rate: f32,
) -> (Voice, Timebase, NeuralRhythms, PhonationBatch, Log2Space) {
    let mut c = VoiceControl::default();
    c.body.method = body;
    c.pitch.mode = PitchMode::Free;
    c.pitch.freq = 330.;
    c.pitch.pitch_apply_mode = mode;
    c.pitch.rearticulate_above_cents = 50.;
    c.phonation.tie = tie;
    c.phonation.articulation_persistence = persistence;
    c.phonation.spec = PhonationSpec {
        timing: PhonationTiming::Coupled(CoupledTimingSpec {
            relation: RhythmRelation::Synchronized,
            coupling: 0.95,
            base_rate_hz: rate,
            ..Default::default()
        }),
        duration: DurationSpec::WhileAlive,
        sustain_participation: true,
    };
    let mut v = Voice::spawn_from_control(
        c,
        ArticulationCoreConfig::Drone {
            sway: None,
            breath_gain_init: None,
        },
        2,
        0,
        VoiceMetadata::default(),
        48000.,
        None,
        21,
    );
    v.configure_control_hop(512, 64, 48000.);
    let mut out = PhonationBatch::default();
    v.prepare_phonation_batch(&mut out);
    (
        v,
        Timebase {
            fs: 48000.,
            hop: 512,
        },
        NeuralRhythms::default(),
        out,
        Log2Space::new(55., 8000., 96),
    )
}

fn emit(
    v: &mut Voice,
    tb: &Timebase,
    r: &NeuralRhythms,
    at: Tick,
    out: &mut PhonationBatch,
    space: &Log2Space,
) {
    v.tick_phonation_into_with_arrival(tb, at, r, None, 0., 1., 0.8, None, Some(space), out);
}

fn adoption(v: &mut Voice, at: Tick, hz: f32) {
    v.participation_pitch.rearticulations.push(Rearticulation {
        at,
        pitch_log2: hz.log2(),
    });
    v.participation_pitch.steps.push(ParticipationPitchStep {
        at,
        before_log2: v.body.base_freq_hz().log2(),
        target_log2: hz.log2(),
        dt: 64. / 48000.,
    });
}

#[test]
fn synchronized_bodies_modes_tie_same_sample_lifecycle_and_finite_rest() {
    let mut cases = 0;
    for body in [BodyMethod::Sine, BodyMethod::Harmonic, BodyMethod::Modal] {
        for mode in [
            PitchApplyMode::Glide,
            PitchApplyMode::GateSnap,
            PitchApplyMode::Rearticulate,
        ] {
            for tie in [0., 1.] {
                if mode == PitchApplyMode::Glide && tie > 0. {
                    continue;
                }
                let (mut v, tb, r, mut out, space) = setup(body, mode, tie, 0., 2.);
                emit(&mut v, &tb, &r, 0, &mut out, &space);
                assert_eq!(out.onsets.len(), 1);
                let mut onsets = 1;
                for i in 1..1000 {
                    emit(&mut v, &tb, &r, i * 512, &mut out, &space);
                    onsets += out.onsets.len();
                }
                if tie == 1. {
                    assert_eq!(onsets, 1);
                } else {
                    assert!(onsets > 20);
                }
                // A same-sample adoption must precede the actual crossing.
                let mut now = 1000 * 512;
                loop {
                    let ctx = CoreTickCtx {
                        now_tick: now,
                        frame_end: now + 512,
                        fs: tb.fs,
                        rhythms: r,
                    };
                    let due = v.phonation_engine.clock.opportunity(&ctx).unwrap().at;
                    if due < now + 512 {
                        adoption(&mut v, due, 440.);
                        emit(&mut v, &tb, &r, now, &mut out, &space);
                        assert!(
                            out.tones
                                .iter()
                                .any(|t| t.onset == due && (t.freq_hz - 440.).abs() < 0.001)
                        );
                        assert!(v.participation_pitch.pending.is_none());
                        break;
                    }
                    emit(&mut v, &tb, &r, now, &mut out, &space);
                    now += 512;
                }
                v.release.pending = true;
                emit(&mut v, &tb, &r, now + 1000 * 512, &mut out, &space);
                assert!(out.onsets.is_empty());
                assert!(v.participation_pitch.pending.is_none());
                cases += 1;
            }
        }
    }
    // Two crossings in a hop: the first consumes pending pitch, tie suppresses the next.
    let (mut v, tb, r, mut out, space) =
        setup(BodyMethod::Sine, PitchApplyMode::GateSnap, 1., 0., 200.);
    emit(&mut v, &tb, &r, 0, &mut out, &space);
    adoption(&mut v, 512, 440.);
    emit(&mut v, &tb, &r, 512, &mut out, &space);
    assert_eq!(out.onsets.len(), 1);
    assert_eq!(
        out.tones[0].freq_hz.to_bits(),
        440.0f32.log2().exp2().to_bits()
    );
    cases += 1;
    // Fixed finite Off, observed tail completion, intrinsic-period Gap, no replay.
    let (mut v, tb, r, mut out, space) =
        setup(BodyMethod::Sine, PitchApplyMode::GateSnap, 1., 0., 2.);
    v.phonation_engine.enable_rest(
        &crate::scenario::DurationConfig::FixedGate { length_gates: 1 },
        1.,
    );
    emit(&mut v, &tb, &r, 0, &mut out, &space);
    let mut off = None;
    for i in 1..100 {
        emit(&mut v, &tb, &r, i * 512, &mut out, &space);
        assert!(out.onsets.is_empty());
        for c in &out.cmds {
            if let ToneCmd::Off { off_tick, .. } = c {
                off = Some(*off_tick);
            }
        }
    }
    let off = off.expect("finite Off absent");
    v.phonation_engine.rest_tail_finished(off + 1000, tb.fs);
    for i in 100..150 {
        emit(&mut v, &tb, &r, i * 512, &mut out, &space);
        if !out.onsets.is_empty() {
            assert!(out.onsets[0].onset_tick >= off + 1000 + 24000);
            break;
        }
    }
    assert!(!out.onsets.is_empty(), "Gap never resumed");
    cases += 1;
    // Persistence can enter Connect on a small adoption and returns via Base crossing proposals.
    let (mut v, tb, r, mut out, space) =
        setup(BodyMethod::Sine, PitchApplyMode::Rearticulate, 0., 0.5, 2.);
    emit(&mut v, &tb, &r, 0, &mut out, &space);
    adoption(&mut v, 512, 331.);
    emit(&mut v, &tb, &r, 512, &mut out, &space);
    assert!(v.participation_pitch.pending.is_none());
    for i in 2..200 {
        emit(&mut v, &tb, &r, i * 512, &mut out, &space);
    }
    assert!(
        v.phonation_engine
            .clock
            .shape_mut()
            .unwrap()
            .retention_active
    );
    assert_eq!(cases + 1, 18);
}

#[test]
fn synchronized_crossings_conserve_sound_hold_and_legality_without_replay() {
    for tied in [0., 1.] {
        let (mut v, tb, r, mut out, space) =
            setup(BodyMethod::Sine, PitchApplyMode::GateSnap, tied, 0., 2.);
        emit(&mut v, &tb, &r, 0, &mut out, &space);
        let mut sounded = 0;
        for i in 1..100 {
            emit(&mut v, &tb, &r, i * 512, &mut out, &space);
            sounded += out.onsets.len();
        }
        let c = v.phonation_engine.synchronized_counts;
        assert_eq!(c[0], c[1] + c[2] + c[3]);
        assert_eq!(c[1], sounded);
        assert_eq!(c[3], 0);
        if tied == 1. {
            assert_eq!(c[1], 0);
            assert!(c[2] > 0);
        } else {
            assert_eq!(c[2], 0);
        }
        v.base_control.body.amp = 0.;
        v.apply_body_runtime();
        v.effective_control.body.amp = 0.;
        v.apply_body_runtime();
        for i in 100..200 {
            emit(&mut v, &tb, &r, i * 512, &mut out, &space);
            assert!(out.onsets.is_empty());
        }
        let c = v.phonation_engine.synchronized_counts;
        assert!(c[3] > 0);
        assert_eq!(c[0], c[1] + c[2] + c[3]);
    }
}

#[test]
fn synchronized_unconfirmed_clock_keeps_existing_rate_phase_and_crossings() {
    let (mut v, tb, r, mut out, space) = setup(BodyMethod::Sine, PitchApplyMode::Glide, 0., 0., 2.);
    let mut reference =
        PhonationEngine::from_spec(&v.effective_control.phonation.spec, 21 ^ 2).clock;
    let mut candidates = Vec::new();
    assert_eq!(r.delta.alpha, 0.);
    for i in 0..100 {
        let ctx = CoreTickCtx {
            now_tick: i * 512,
            frame_end: (i + 1) * 512,
            fs: tb.fs,
            rhythms: r,
        };
        candidates.clear();
        reference.gather_candidates(&ctx, &mut candidates);
        emit(&mut v, &tb, &r, i * 512, &mut out, &space);
        let actual: Vec<_> = out
            .tones
            .iter()
            .filter(|t| t.participation_onset)
            .map(|t| t.onset)
            .collect();
        assert_eq!(
            actual,
            candidates.iter().map(|c| c.tick).collect::<Vec<_>>()
        );
    }
}

#[test]
fn synchronized_mode_and_boundary_changes_keep_clock_and_cancel_pending_pitch() {
    let (mut v, tb, r, mut out, space) =
        setup(BodyMethod::Sine, PitchApplyMode::Rearticulate, 1., 0.5, 2.);
    emit(&mut v, &tb, &r, 0, &mut out, &space);
    adoption(&mut v, 512, 440.);
    emit(&mut v, &tb, &r, 512, &mut out, &space);
    let ctx = CoreTickCtx {
        now_tick: 1024,
        frame_end: 1536,
        fs: tb.fs,
        rhythms: r,
    };
    let due = v.phonation_engine.clock.opportunity(&ctx).unwrap().at;
    v.apply_patch(&ControlUpdate {
        pitch_apply_mode: Some(PitchApplyMode::Glide),
        ..Default::default()
    })
    .unwrap();
    assert!(v.participation_pitch.pending.is_none());
    assert_eq!(v.phonation_engine.clock.opportunity(&ctx).unwrap().at, due);
    emit(&mut v, &tb, &r, 1024, &mut out, &space);
    assert!(out.onsets.is_empty());
    v.apply_patch(&ControlUpdate {
        pitch_apply_mode: Some(PitchApplyMode::Rearticulate),
        rearticulate_above_cents: Some(0.),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(
        v.phonation_engine.clock.shape_mut().unwrap().state,
        crate::life::shape_policy::Shape::Base
    );
    emit(&mut v, &tb, &r, 1536, &mut out, &space);
    assert!(out.onsets.is_empty());
}

#[test]
fn synchronized_onset_metabolism_and_attack_learning_happen_once_per_actual_on() {
    let (mut v, tb, r, mut out, space) =
        setup(BodyMethod::Sine, PitchApplyMode::GateSnap, 1., 0., 2.);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(21);
    v.articulation.core =
        AnyArticulationCore::from_config(&ArticulationCoreConfig::default(), tb.fs, 2, &mut rng);
    v.articulation.set_autonomous_attack_enabled(false);
    let AnyArticulationCore::Entrain(core) = &mut v.articulation.core else {
        panic!()
    };
    core.energy = 0.8;
    core.metabolism.action_cost_per_attack = 0.1;
    core.metabolism.recharge_per_attack = 0.;
    let mut expected = 0.8f32;
    let mut total = 0;
    for i in 0..100 {
        if i == 50 {
            adoption(&mut v, i * 512, 440.);
        }
        emit(&mut v, &tb, &r, i * 512, &mut out, &space);
        for _ in &out.onsets {
            expected -= 0.1;
            total += 1;
        }
        let AnyArticulationCore::Entrain(core) = &v.articulation.core else {
            panic!()
        };
        assert_eq!(core.energy.to_bits(), expected.to_bits());
    }
    assert_eq!(total, 2);
    assert!(!v.phonation_engine.uses_participation());
}

#[test]
#[cfg(feature = "profile-alloc")]
fn synchronized_storage_is_prepared_for_the_first_hop_and_reused() {
    for (tie, persistence) in [(0., 0.), (1., 0.), (0.5, 0.5)] {
        let (mut v, tb, r, mut out, space) = setup(
            BodyMethod::Sine,
            PitchApplyMode::Rearticulate,
            tie,
            persistence,
            2.,
        );
        crate::runtime_profile::begin_allocations();
        for i in 0..1000 {
            if i % 40 == 10 {
                adoption(&mut v, i * 512, 440.);
            }
            emit(&mut v, &tb, &r, i * 512, &mut out, &space);
        }
        let counts = crate::runtime_profile::finish_allocations().unwrap();
        assert_eq!(counts.count, 0, "{counts:?}");
        assert_eq!(counts.bytes, 0, "{counts:?}");
    }
}

#[test]
fn span_gap_executes_on_real_synchronized_and_participation_paths() {
    use crate::life::span_gap::{Phase, SpanGap};
    for participating in [false, true] {
        for persistence in [0.0, 0.5] {
            for finite in [false, true] {
                let (mut v, tb, r, mut out, space) = setup(
                    BodyMethod::Sine,
                    PitchApplyMode::Rearticulate,
                    0.0,
                    persistence,
                    20.0,
                );
                if participating {
                    let shape = v.phonation_engine.clock.shape_slot().unwrap().take();
                    v.phonation_engine.clock = PhonationClock::from_config(
                        &crate::scenario::PhonationClockConfig::Participation {
                            coupling: 0.0,
                            base_rate_hz: 20.0,
                            flow_depth: 0.0,
                            steady: true,
                        },
                        21,
                    );
                    *v.phonation_engine.clock.shape_slot().unwrap() = shape;
                }
                if finite {
                    v.phonation_engine.enable_rest(
                        &crate::scenario::DurationConfig::FixedGate { length_gates: 20 },
                        1.0,
                    );
                }
                let mut evidence = Landscape::new(space.clone());
                evidence.onset_habituation_known = true;
                evidence.onset_habituation_state_scan = vec![1.0; space.n_bins()].into();
                let mut gap = SpanGap::new(1.0, 21, 2, 0, tb.fs, 330.0);
                gap.observe(&evidence, 0);
                gap.trace_enabled = true;
                v.phonation_engine.span_gap = Some(gap);
                emit(&mut v, &tb, &r, 0, &mut out, &space);
                assert_eq!(out.tones.len(), 1, "initial Hold");
                let original = out.tones[0].tone_id;
                v.phonation_engine.span_gap.as_mut().unwrap().target = 0.0;
                let mut released = None;
                for index in 1..20 {
                    emit(&mut v, &tb, &r, index * 512, &mut out, &space);
                    if let Some(at) = out.cmds.iter().find_map(|c| match c {
                        ToneCmd::Off { tone_id, off_tick } if *tone_id == original => {
                            Some(*off_tick)
                        }
                        _ => None,
                    }) {
                        released = Some((at, index));
                        break;
                    }
                }
                let (off, index) = released.expect("exit at an existing opportunity");
                assert!(out.tones.is_empty());
                assert_eq!(
                    v.phonation_engine.span_gap.as_ref().unwrap().phase,
                    Phase::GapTail
                );
                v.phonation_engine.rest_tail_finished(off - 1, tb.fs);
                assert_eq!(
                    v.phonation_engine.span_gap.as_ref().unwrap().phase,
                    Phase::GapTail
                );
                let quiet = (index + 1) * 512;
                for step in index + 1..index + 4 {
                    emit(&mut v, &tb, &r, step * 512, &mut out, &space);
                    assert!(out.tones.is_empty());
                    assert!(
                        !out.cmds
                            .iter()
                            .any(|c| matches!(c,ToneCmd::Off{tone_id,..} if *tone_id==original))
                    );
                }
                v.phonation_engine.rest_tail_finished(quiet, tb.fs);
                let gap = v.phonation_engine.span_gap.as_mut().unwrap();
                assert_eq!(gap.phase, Phase::GapActive);
                if finite {
                    assert_eq!(gap.return_integral, 0.0);
                }
                gap.target = 0.0;
                let mut returned = None;
                for step in index + 4..index + 80 {
                    let ctx = CoreTickCtx {
                        now_tick: step * 512,
                        frame_end: (step + 1) * 512,
                        fs: tb.fs,
                        rhythms: r,
                    };
                    if let Some(opportunity) = v.phonation_engine.clock.opportunity(&ctx)
                        && opportunity.at < ctx.frame_end
                    {
                        adoption(&mut v, opportunity.at, 440.0);
                    }
                    emit(&mut v, &tb, &r, step * 512, &mut out, &space);
                    if let Some(tone) = out.tones.first() {
                        assert!(
                            (tone.freq_hz - 440.0).abs() < 0.001,
                            "same-sample pending adoption"
                        );
                        assert!(v.participation_pitch.pending.is_none());
                        returned = Some(tone.onset);
                        break;
                    }
                }
                let returned = returned.expect("return on a legal opportunity");
                assert!(returned >= quiet);
                if finite {
                    assert!(returned >= quiet + 2400);
                }
                assert_eq!(
                    v.phonation_engine.span_gap.as_ref().unwrap().phase,
                    Phase::Span
                );
                v.release.pending = true;
                emit(&mut v, &tb, &r, returned + 512, &mut out, &space);
                assert!(out.tones.is_empty());
                assert_eq!(
                    v.phonation_engine.span_gap.as_ref().unwrap().phase,
                    Phase::Finished
                );
            }
        }
    }
}

#[test]
fn acoustic_history_changes_a_legal_exit_and_clamping_heard_removes_the_difference() {
    use crate::core::auditory_frontend::AuditoryFrontEnd;
    use crate::core::onset_habituation::OnsetHabituation;
    use crate::life::span_gap::{Phase, SpanGap};
    let (_, params) = crate::runtime::body_metabolism_test_core();
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut histories = Vec::new();
    for hz in [330.0, 2000.0] {
        let mut frontend = AuditoryFrontEnd::selected(space.clone(), 48000.0, 512, 16384);
        let mut observer = OnsetHabituation::new(
            &space,
            frontend.band_energy().0,
            crate::core::habituation::HabituationParams::default(),
            0.5,
            48000.0,
        );
        for hop in 0..500 {
            let audio: Vec<_> = (0..512)
                .map(|n| {
                    let t = (hop * 512 + n) as f64 / 48000.0;
                    (0.1 * (-t.rem_euclid(0.5) / 0.025).exp()
                        * (std::f64::consts::TAU * hz * t).sin()) as f32
                })
                .collect();
            frontend.process(&audio, &params);
            observer.process(frontend.band_energy().1, 512, (hop + 1) * 512);
        }
        let mut frame = Landscape::new(space.clone());
        frame.onset_habituation_state_scan = observer.state.into();
        frame.onset_habituation_end = 500 * 512;
        frame.onset_habituation_known = true;
        histories.push(frame);
    }
    for clamp in [false, true] {
        let mut outcomes = Vec::new();
        for history in &histories {
            let (mut voice, tb, r, mut out, space) =
                setup(BodyMethod::Sine, PitchApplyMode::GateSnap, 0.0, 0.0, 20.0);
            let mut evidence = history.clone();
            if clamp {
                std::sync::Arc::make_mut(&mut evidence.onset_habituation_state_scan).fill(0.0);
            }
            let start = 500 * 512;
            let mut gap = SpanGap::new(1.0, 21, 2, 0, 48000.0, 330.0);
            gap.observe(&evidence, start);
            voice.phonation_engine.span_gap = Some(gap);
            emit(&mut voice, &tb, &r, start, &mut out, &space);
            // This target lies strictly between the two held-history integrals.
            voice.phonation_engine.span_gap.as_mut().unwrap().target = 0.001;
            for hop in 1..12 {
                emit(&mut voice, &tb, &r, start + hop * 512, &mut out, &space);
            }
            outcomes.push(voice.phonation_engine.span_gap.as_ref().unwrap().phase);
        }
        assert_eq!(outcomes[1], Phase::Span);
        assert_eq!(
            outcomes[0],
            if clamp { Phase::Span } else { Phase::GapTail }
        );
    }
}

#[test]
fn span_gap_private_pitch_cancels_with_a_live_application_mode_change() {
    let (mut v, tb, r, mut out, space) = setup(
        BodyMethod::Sine,
        PitchApplyMode::Rearticulate,
        0.0,
        0.0,
        2.0,
    );
    v.phonation_engine.span_gap = Some(crate::life::span_gap::SpanGap::new(
        0.25, 21, 2, 0, tb.fs, 330.0,
    ));
    emit(&mut v, &tb, &r, 0, &mut out, &space);
    adoption(&mut v, 512, 440.0);
    emit(&mut v, &tb, &r, 512, &mut out, &space);
    assert!(v.participation_pitch.pending.is_some());
    assert!(
        v.phonation_engine
            .span_gap
            .as_ref()
            .unwrap()
            .pending_pitch
            .is_some()
    );
    v.apply_patch(&ControlUpdate {
        pitch_apply_mode: Some(PitchApplyMode::Glide),
        ..Default::default()
    })
    .unwrap();
    let gap = v.phonation_engine.span_gap.as_ref().unwrap();
    assert!(gap.pending_pitch.is_none());
    assert!(gap.glide_target.is_none());
    assert!(v.participation_pitch.pending.is_none());
}

#[test]
fn dwell_proposals_reach_distinct_legal_onsets_without_forcing_attacks() {
    let mut first_onsets = Vec::new();
    for weight in [0.0, 1.0] {
        for target in [330.0_f32, 331.0, 440.0] {
            let (mut v, tb, r, mut out, space) = setup(
                BodyMethod::Sine,
                PitchApplyMode::Rearticulate,
                1.0,
                0.0,
                2.0,
            );
            v.effective_control.pitch.dwell = weight;
            v.effective_control.pitch.proposal_interval_sec = Some(0.6);
            v.effective_control.pitch.neighbor_step_cents = Some(1200.0);
            v.effective_control.pitch.global_peak_count = 1;
            v.effective_control.pitch.temperature = Some(0.0);
            v.effective_control.pitch.tessitura_gravity = Some(0.0);
            v.effective_control.pitch.move_cost_coeff = 0.0;
            v.pitch_ctl.apply_control(&v.effective_control.pitch);
            v.pitch_ctl.set_adaptation_enabled(false);
            v.pitch_ctl.dwell_trace_enabled = true;
            let mut evidence = Landscape::new(space.clone());
            evidence.consonance_field_score_eff.fill(0.55);
            let index = space.nearest_index(target);
            evidence.consonance_field_score_eff[index] = 1.5;
            emit(&mut v, &tb, &r, 0, &mut out, &space);
            let mut first = None;
            for step in 1..300 {
                let at = step * 512;
                v.control_tick = Some(at);
                v.update_pitch_target_with_listener_pressure(
                    &r,
                    512.0 / tb.fs,
                    &evidence,
                    &[],
                    &[],
                    ListenerPressure::default(),
                );
                v.update_articulation_autonomous(512.0 / tb.fs, &r);
                emit(&mut v, &tb, &r, at, &mut out, &space);
                if let Some(on) = out.tones.first() {
                    first = Some(on.onset);
                    break;
                }
            }
            if target < 332.0 {
                assert!(
                    first.is_none(),
                    "no adoption or below-threshold adoption must not force On: {target}, {weight}"
                );
            } else {
                first_onsets.push(first.expect("valid adopted target must reach On"));
            }
        }
    }
    assert!(
        first_onsets[1] > first_onsets[0],
        "weighted clock must defer the real On: {first_onsets:?}"
    );
    assert_eq!(first_onsets[0] % 24000, first_onsets[1] % 24000);
}

#[test]
fn dwell_gap_clock_keeps_running_and_private_recovery_uses_real_time() {
    use crate::life::span_gap::{Phase, SpanGap};
    let mut recovery = Vec::new();
    for weight in [0.0, 1.0] {
        let (mut v, tb, r, mut out, space) = setup(
            BodyMethod::Sine,
            PitchApplyMode::Rearticulate,
            1.0,
            0.0,
            2.0,
        );
        v.effective_control.pitch.dwell = weight;
        v.effective_control.pitch.proposal_interval_sec = Some(0.2);
        v.pitch_ctl.dwell_trace_enabled = true;
        let mut gap = SpanGap::new(1.0, 21, 2, 0, tb.fs, 330.0);
        let mut evidence = Landscape::new(space.clone());
        evidence.consonance_field_score_eff.fill(-0.5);
        evidence.onset_habituation_known = true;
        evidence.onset_habituation_state_scan = vec![0.0; space.n_bins()].into();
        gap.observe(&evidence, 0);
        gap.phase = Phase::GapActive;
        gap.own = 0.8;
        gap.target = 1000.0;
        v.phonation_engine.span_gap = Some(gap);
        for step in 0..100 {
            let at = step * 512;
            v.control_tick = Some(at);
            v.update_pitch_target_with_listener_pressure(
                &r,
                512.0 / tb.fs,
                &evidence,
                &[],
                &[],
                ListenerPressure::default(),
            );
            v.update_articulation_autonomous(512.0 / tb.fs, &r);
            emit(&mut v, &tb, &r, at, &mut out, &space);
            assert!(out.tones.is_empty());
        }
        let gap = v.phonation_engine.span_gap.as_ref().unwrap();
        assert_eq!(gap.phase, Phase::GapActive);
        assert!(gap.own < 0.8);
        recovery.push(gap.own);
        if weight > 0.0 {
            assert!(v.pitch_ctl.dwell_trace.len() > 5);
            let before = v.pitch_ctl.dwell_state.unwrap().remainder;
            v.phonation_engine.span_gap.as_mut().unwrap().phase = Phase::Ready;
            emit(&mut v, &tb, &r, 100 * 512, &mut out, &space);
            assert_eq!(
                v.pitch_ctl.dwell_state.unwrap().remainder,
                before,
                "return cannot reset proposal phase"
            );
        }
    }
    assert_eq!(
        recovery[0], recovery[1],
        "private attack recovery must use real time"
    );
}

#[test]
fn dwell_adoption_at_exit_waits_for_tail_and_ready_veto_then_consumes_once() {
    use crate::life::span_gap::{Phase, SpanGap};
    for weight in [0.0, 1.0] {
        let (mut v, tb, r, mut out, space) = setup(
            BodyMethod::Sine,
            PitchApplyMode::Rearticulate,
            1.0,
            0.0,
            2.0,
        );
        v.effective_control.pitch.dwell = weight;
        v.effective_control.pitch.proposal_interval_sec = Some(0.1);
        v.effective_control.pitch.global_peak_count = 1;
        v.effective_control.pitch.temperature = Some(0.0);
        v.effective_control.pitch.tessitura_gravity = Some(0.0);
        v.effective_control.pitch.move_cost_coeff = 0.0;
        v.pitch_ctl.apply_control(&v.effective_control.pitch);
        v.pitch_ctl.set_adaptation_enabled(false);
        let mut evidence = Landscape::new(space.clone());
        evidence.consonance_field_score_eff.fill(-0.5);
        evidence.consonance_field_score_eff[space.nearest_index(440.0)] = 2.0;
        evidence.onset_habituation_known = true;
        evidence.onset_habituation_state_scan = vec![1.0; space.n_bins()].into();
        let mut gap = SpanGap::new(1.0, 21, 2, 0, tb.fs, 330.0);
        gap.observe(&evidence, 0);
        v.phonation_engine.span_gap = Some(gap);
        emit(&mut v, &tb, &r, 0, &mut out, &space);
        let first = out.tones[0].tone_id;
        v.phonation_engine.span_gap.as_mut().unwrap().target = 1000.0;
        let mut exit = None;
        for step in 1..100 {
            let ctx = CoreTickCtx {
                now_tick: step * 512,
                frame_end: (step + 1) * 512,
                fs: tb.fs,
                rhythms: r,
            };
            if let Some(opportunity) = v.phonation_engine.clock.opportunity(&ctx)
                && opportunity.at < ctx.frame_end
            {
                v.control_tick = Some(opportunity.at);
                v.update_pitch_target(&r, 0.5, &evidence, &[], &[]);
                assert!(v.participation_pitch.adopted);
                v.update_articulation_autonomous(0.5, &r);
                v.phonation_engine.span_gap.as_mut().unwrap().target = 0.0;
                emit(&mut v, &tb, &r, ctx.now_tick, &mut out, &space);
                assert!(
                    out.tones.is_empty(),
                    "same-sample exit must win over adoption"
                );
                assert!(out.cmds.iter().any(|cmd|matches!(cmd,ToneCmd::Off{tone_id,off_tick} if *tone_id==first && *off_tick==opportunity.at)));
                exit = Some(step);
                break;
            }
            emit(&mut v, &tb, &r, ctx.now_tick, &mut out, &space);
        }
        let step = exit.expect("legal exit");
        assert_eq!(
            v.phonation_engine.span_gap.as_ref().unwrap().phase,
            Phase::GapTail
        );
        emit(&mut v, &tb, &r, (step + 1) * 512, &mut out, &space);
        assert!(out.tones.is_empty());
        v.phonation_engine
            .rest_tail_finished((step + 1) * 512, tb.fs);
        assert_eq!(
            v.phonation_engine.span_gap.as_ref().unwrap().phase,
            Phase::GapActive
        );
        v.phonation_engine.span_gap.as_mut().unwrap().phase = Phase::Ready;
        v.apply_patch(&ControlUpdate {
            amp: Some(0.0),
            ..Default::default()
        })
        .unwrap();
        for i in step + 2..step + 110 {
            emit(&mut v, &tb, &r, i * 512, &mut out, &space);
            assert!(out.tones.is_empty());
            assert_eq!(
                v.phonation_engine.span_gap.as_ref().unwrap().phase,
                Phase::Ready
            );
        }
        assert!(v.participation_pitch.pending.is_some());
        v.apply_patch(&ControlUpdate {
            amp: Some(0.1),
            ..Default::default()
        })
        .unwrap();
        let mut resumed = 0;
        for i in step + 110..step + 210 {
            emit(&mut v, &tb, &r, i * 512, &mut out, &space);
            resumed += out.tones.len();
        }
        assert_eq!(resumed, 1, "consume the held target once, without replay");
        assert!(v.participation_pitch.pending.is_none());
    }
}

#[test]
fn dwell_multiple_adoptions_replace_pending_without_extra_onsets() {
    let (mut v, tb, r, mut out, space) = setup(
        BodyMethod::Sine,
        PitchApplyMode::Rearticulate,
        1.0,
        0.0,
        2.0,
    );
    v.effective_control.pitch.dwell = 1.0;
    v.effective_control.pitch.proposal_interval_sec = Some(0.2);
    v.effective_control.pitch.global_peak_count = 1;
    v.effective_control.pitch.temperature = Some(0.0);
    v.effective_control.pitch.tessitura_gravity = Some(0.0);
    v.effective_control.pitch.move_cost_coeff = 0.0;
    v.pitch_ctl.apply_control(&v.effective_control.pitch);
    v.pitch_ctl.set_adaptation_enabled(false);
    let mut evidence = Landscape::new(space.clone());
    emit(&mut v, &tb, &r, 0, &mut out, &space);
    let mut adopted = Vec::new();
    let mut sounded = Vec::new();
    for step in 1..90 {
        evidence.consonance_field_score_eff.fill(-100.0);
        evidence.consonance_field_score_eff
            [space.nearest_index(if step < 15 { 440.0 } else { 550.0 })] = 2.0;
        v.control_tick = Some(step * 512);
        v.update_pitch_target(&r, 512.0 / tb.fs, &evidence, &[], &[]);
        if v.participation_pitch.adopted {
            adopted.push((step * 512, v.target_pitch_log2().exp2()));
        }
        v.update_articulation_autonomous(512.0 / tb.fs, &r);
        emit(&mut v, &tb, &r, step * 512, &mut out, &space);
        sounded.extend(out.tones.iter().map(|t| (t.onset, t.freq_hz)));
    }
    assert!(adopted.iter().any(|&(_, hz)| hz > 400.0 && hz < 480.0));
    assert!(adopted.iter().any(|&(_, hz)| hz > 500.0));
    assert_eq!(sounded.len(), 1, "sounded={sounded:?}; adopted={adopted:?}");
    assert!(adopted.iter().all(|&(at, _)| at < sounded[0].0));
    assert!(sounded[0].1 > 500.0, "the last pending adoption must win");
}
