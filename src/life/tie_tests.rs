use super::*;

fn setup(
    p: f32,
    mode: PitchApplyMode,
) -> (Voice, Timebase, NeuralRhythms, PhonationBatch, Log2Space) {
    let mut control = VoiceControl::default();
    control.pitch.mode = PitchMode::Free;
    control.pitch.freq = 330.0;
    control.pitch.pitch_apply_mode = mode;
    control.pitch.rearticulate_above_cents = 50.0;
    control.pitch.pitch_glide_tau_sec = 0.03;
    control.phonation.tie = p;
    control.phonation.spec = crate::scenario::PhonationSpec {
        timing: crate::scenario::PhonationTiming::Coupled(crate::scenario::CoupledTimingSpec {
            relation: crate::scenario::RhythmRelation::Participating,
            coupling: 0.0,
            ..Default::default()
        }),
        duration: crate::scenario::DurationSpec::WhileAlive,
        sustain_participation: true,
    };
    let mut v = Voice::spawn_from_control(
        control,
        ArticulationCoreConfig::Drone {
            sway: None,
            breath_gain_init: None,
        },
        2,
        0,
        VoiceMetadata::default(),
        48_000.0,
        None,
        21,
    );
    v.articulation.set_autonomous_attack_enabled(false);
    v.configure_control_hop(512, 64);
    let mut out = PhonationBatch::default();
    v.prepare_phonation_batch(&mut out);
    (
        v,
        Timebase {
            fs: 48_000.0,
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
    v.tick_phonation_into_with_arrival(tb, at, r, None, 0.0, 1.0, 0.8, None, Some(space), out);
}

fn adoption(v: &mut Voice, at: Tick, hz: f32) {
    v.rearticulations.push(Rearticulation {
        at,
        pitch_log2: hz.log2(),
    });
    v.participation_pitch_steps.push(ParticipationPitchStep {
        at,
        before_log2: v.body.base_freq_hz().log2(),
        target_log2: hz.log2(),
        dt: 64.0 / 48_000.0,
    });
}

fn due(v: &Voice) -> Tick {
    let PhonationClock::Participation(c) = &v.phonation_engine.clock else {
        panic!()
    };
    c.policy.as_ref().unwrap().opportunity(512).unwrap().at
}

#[test]
fn tie_birth_hold_pending_on_and_no_unpending_restrikes() {
    for mode in [PitchApplyMode::GateSnap, PitchApplyMode::Rearticulate] {
        let (mut v, tb, r, mut out, space) = setup(1.0, mode);
        emit(&mut v, &tb, &r, 0, &mut out, &space);
        assert_eq!(out.tones.len(), 1);
        let initial = v.last_onset_pitch_log2;
        for i in 1..100 {
            emit(&mut v, &tb, &r, i * 512, &mut out, &space);
            assert!(out.tones.is_empty());
        }
        assert_eq!(v.last_onset_pitch_log2, initial);
        assert_eq!(v.active_render_notes.len(), 1);
        adoption(&mut v, 51200, 440.0);
        let mut moved = false;
        for i in 100..200 {
            emit(&mut v, &tb, &r, i * 512, &mut out, &space);
            if let Some(tone) = out.tones.first() {
                assert_eq!(tone.freq_hz.to_bits(), 440.0f32.log2().exp2().to_bits());
                moved = true;
                break;
            }
        }
        assert!(moved, "pending movement never sounded");
    }
}

#[test]
fn tie_same_sample_adoption_precedes_on_and_small_adoption_cancels() {
    for small in [false, true] {
        let (mut v, tb, r, mut out, space) = setup(1.0, PitchApplyMode::Rearticulate);
        emit(&mut v, &tb, &r, 0, &mut out, &space);
        let due = due(&v);
        let hop = due / 512 * 512;
        for at in (512..hop).step_by(512) {
            emit(&mut v, &tb, &r, at, &mut out, &space);
        }
        v.pending_participation_pitch = Some(440.0f32.log2());
        adoption(
            &mut v,
            due,
            if small {
                330.0 * 2.0f32.powf(25.0 / 1200.0)
            } else {
                440.0
            },
        );
        emit(&mut v, &tb, &r, hop, &mut out, &space);
        if small {
            assert!(out.tones.is_empty());
            assert!(v.pending_participation_pitch.is_none());
            assert!(v.participation_glide_target.is_some());
            assert!(v.body.base_freq_hz() > 330.0);
        } else {
            assert!(
                out.tones
                    .iter()
                    .any(|t| t.onset == due
                        && t.freq_hz.to_bits() == 440.0f32.log2().exp2().to_bits())
            );
        }
    }
}

#[test]
fn tie_finite_off_tail_gap_then_legal_on() {
    let (mut v, tb, r, mut out, space) = setup(1.0, PitchApplyMode::Rearticulate);
    v.phonation_engine.enable_rest(
        &crate::scenario::DurationConfig::FixedGate { length_gates: 1 },
        1.0,
    );
    emit(&mut v, &tb, &r, 0, &mut out, &space);
    assert_eq!(out.tones.len(), 1);
    let mut off = None;
    for i in 1..150 {
        emit(&mut v, &tb, &r, i * 512, &mut out, &space);
        if let Some(at) = out.cmds.iter().find_map(|c| {
            if let ToneCmd::Off { off_tick, .. } = c {
                Some(*off_tick)
            } else {
                None
            }
        }) {
            off = Some(at);
            break;
        }
    }
    assert_eq!(off, Some(24000));
    for i in 100..110 {
        emit(&mut v, &tb, &r, i * 512, &mut out, &space);
        assert!(out.tones.is_empty());
    }
    let completed = 110 * 512;
    v.phonation_engine.rest_tail_finished(completed, 48_000.0);
    let until = completed + 24000;
    let mut resumed = false;
    for i in 110..250 {
        emit(&mut v, &tb, &r, i * 512, &mut out, &space);
        for t in &out.tones {
            assert!(t.onset >= until);
            resumed = true;
        }
        if resumed {
            break;
        }
    }
    assert!(resumed, "tie lost a legal onset after rest");
}

#[test]
fn tie_release_remove_death_priority() {
    for ending in 0..3 {
        let (mut v, tb, r, mut out, space) = setup(1.0, PitchApplyMode::Rearticulate);
        emit(&mut v, &tb, &r, 0, &mut out, &space);
        v.pending_participation_pitch = Some(440.0f32.log2());
        match ending {
            0 => v.release_pending = true,
            1 => v.start_remove_fade(0.12),
            _ => v.release_gain = 0.0,
        };
        emit(&mut v, &tb, &r, 512, &mut out, &space);
        assert!(out.tones.is_empty());
        assert!(v.pending_participation_pitch.is_none());
    }
}

#[test]
fn tie_mode_suspension_and_hop_partition_do_not_add_trials() {
    let run = |hop: usize| {
        let (mut v, mut tb, r, mut out, space) = setup(0.5, PitchApplyMode::Rearticulate);
        tb.hop = hop;
        v.configure_control_hop(hop, 64);
        let mut onsets = Vec::new();
        for at in (0..65536).step_by(hop) {
            emit(&mut v, &tb, &r, at, &mut out, &space);
            onsets.extend(out.tones.iter().map(|t| t.onset));
        }
        let PhonationClock::Participation(c) = &v.phonation_engine.clock else {
            panic!()
        };
        (
            onsets,
            c.shape.as_ref().unwrap().tie.as_ref().unwrap().trials,
        )
    };
    assert_eq!(run(512), run(256));
    let (mut v, tb, r, mut out, space) = setup(1., PitchApplyMode::Rearticulate);
    emit(&mut v, &tb, &r, 0, &mut out, &space);
    v.effective_control.pitch.pitch_apply_mode = PitchApplyMode::Glide;
    v.apply_pitch_control();
    let mut count = 0;
    for at in (512..51200).step_by(512) {
        emit(&mut v, &tb, &r, at, &mut out, &space);
        count += out.tones.len();
    }
    assert!(count > 0);
    v.effective_control.pitch.pitch_apply_mode = PitchApplyMode::GateSnap;
    v.apply_pitch_control();
    for at in (51200..102400).step_by(512) {
        emit(&mut v, &tb, &r, at, &mut out, &space);
        assert!(out.tones.is_empty());
    }
}

#[test]
#[cfg(feature = "profile-alloc")]
fn tie_additional_path_has_no_steady_state_allocations() {
    for p in [0.5, 1.] {
        let (mut v, tb, r, mut out, space) = setup(p, PitchApplyMode::Rearticulate);
        for i in 0..100 {
            emit(&mut v, &tb, &r, i * 512, &mut out, &space);
        }
        crate::runtime_profile::begin_allocations();
        for i in 100..1100 {
            if i % 29 == 0 {
                adoption(&mut v, i * 512, 440.0);
            }
            if i % 31 == 0 {
                adoption(&mut v, i * 512, 330.0);
            }
            emit(&mut v, &tb, &r, i * 512, &mut out, &space);
        }
        let counts = crate::runtime_profile::finish_allocations().unwrap();
        assert_eq!((counts.count, counts.bytes), (0, 0));
    }
}
