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
