use super::*;

fn test_voice() -> Voice {
    Voice::spawn_from_control(
        VoiceControl::default(),
        ArticulationCoreConfig::Drone {
            sway: None,
            breath_gain_init: None,
        },
        1,
        0,
        VoiceMetadata::default(),
        48_000.0,
        None,
        0,
    )
}

#[test]
fn support_partial_cache_reuses_storage_and_rebuilds_after_body_change() {
    use crate::core::log2space::Log2Space;
    use crate::scenario::control::BodyMethod;
    let landscape = Landscape::new(Log2Space::new(55., 7040., 48));
    for method in [BodyMethod::Sine, BodyMethod::Harmonic, BodyMethod::Modal] {
        let mut control = VoiceControl::default();
        control.body.method = method;
        control.body.amp = 0.2;
        control.pitch.freq = 220.;
        let mut voice = Voice::spawn_from_control(
            control,
            ArticulationCoreConfig::Drone {
                sway: None,
                breath_gain_init: None,
            },
            1,
            0,
            VoiceMetadata::default(),
            48_000.,
            Some(&landscape),
            735,
        );
        voice.refresh_support_template(&landscape);
        let first = voice.support.template.clone().expect("body self-report");
        assert!(first.iter().all(|&(_, q)| q.is_finite() && q > 0.));
        #[cfg(feature = "profile-alloc")]
        crate::runtime_profile::begin_allocations();
        for _ in 0..8 {
            voice.refresh_support_template(&landscape);
            assert!(std::sync::Arc::ptr_eq(
                &first,
                voice.support.template.as_ref().unwrap()
            ));
        }
        #[cfg(feature = "profile-alloc")]
        {
            let counts = crate::runtime_profile::finish_allocations().unwrap();
            assert_eq!((counts.count, counts.bytes), (0, 0));
        }
        voice.effective_control.body.amp = 0.4;
        voice.apply_body_runtime();
        voice.refresh_support_template(&landscape);
        let changed = voice.support.template.as_ref().unwrap();
        assert!(!std::sync::Arc::ptr_eq(&first, changed));
        assert!(
            changed.iter().map(|&(_, q)| q).sum::<f32>()
                > first.iter().map(|&(_, q)| q).sum::<f32>()
        );
        voice.effective_control.body.amp = 0.;
        voice.apply_body_runtime();
        voice.refresh_support_template(&landscape);
        assert!(
            voice
                .support
                .template
                .as_ref()
                .expect("silent self-report remains known")
                .is_empty()
        );
    }
}

#[test]
fn note_ending_in_its_onset_hop_does_not_remain_tracked_for_release() {
    use crate::life::phonation_engine::{CandidatePoint, PhonationClock};
    use crate::scenario::{DurationConfig, OnsetConfig, PhonationClockConfig, PhonationConfig};

    let mut voice = test_voice();
    voice.phonation_engine = PhonationEngine::from_config(
        &PhonationConfig {
            measure_accent: 0.0,
            mode: PhonationMode::Gated,
            onset: OnsetConfig::Always { strength: 1.0 },
            duration: DurationConfig::FixedGate { length_gates: 0 },
            clock: PhonationClockConfig::ThetaGate,
        },
        0,
    );
    voice.phonation_engine.clock = PhonationClock::Custom(Box::new(|ctx, out| {
        out.push(CandidatePoint {
            tick: ctx.now_tick,
            gate: 0,
        });
    }));
    let mut rhythms = NeuralRhythms::default();
    rhythms.env_open = 1.0;
    let batch = voice.tick_phonation(
        &Timebase {
            fs: 48_000.0,
            hop: 512,
        },
        0,
        &rhythms,
        None,
        0.0,
        1.0,
        1.0,
    );
    assert_eq!(batch.tones.len(), 1);
    assert!(
        batch
            .cmds
            .iter()
            .any(|cmd| matches!(cmd, ToneCmd::Off { .. }))
    );
    assert!(voice.active_render_notes.is_empty());
    let mut release = Vec::new();
    voice.emit_release_note_offs(512, &mut release);
    assert!(release.is_empty());
}

#[test]
fn release_note_offs_cover_tracked_render_notes_once() {
    let mut voice = test_voice();
    voice.active_render_notes = vec![
        TrackedRenderNote {
            tone_id: 10,
            last_target_amp: 0.5,
            last_target_freq_hz: 220.0,
            last_continuous_drive: 0.0,
        },
        TrackedRenderNote {
            tone_id: 11,
            last_target_amp: 0.5,
            last_target_freq_hz: 330.0,
            last_continuous_drive: 0.0,
        },
    ];
    let mut cmds = vec![ToneCmd::Off {
        tone_id: 10,
        off_tick: 7,
    }];

    voice.emit_release_note_offs(9, &mut cmds);
    voice.prune_tracked_render_notes(&cmds);

    let off_10 = cmds
        .iter()
        .filter(|cmd| matches!(cmd, ToneCmd::Off { tone_id: 10, .. }))
        .count();
    assert_eq!(off_10, 1);
    assert!(cmds.iter().any(|cmd| matches!(
        cmd,
        ToneCmd::Off {
            tone_id: 11,
            off_tick: 9
        }
    )));
    assert!(voice.active_render_notes.is_empty());
}

#[test]
fn remove_pending_voice_waits_to_flush_tracked_notes() {
    let mut voice = test_voice();
    voice.release.remove_pending = true;
    voice.release.gain = 0.0;
    voice.active_render_notes = vec![TrackedRenderNote {
        tone_id: 10,
        last_target_amp: 0.5,
        last_target_freq_hz: 220.0,
        last_continuous_drive: 0.0,
    }];

    assert!(voice.should_retain());

    voice.active_render_notes.clear();
    assert!(!voice.should_retain());
}

fn drone_voice_with_control(control: VoiceControl) -> Voice {
    Voice::spawn_from_control(
        control,
        ArticulationCoreConfig::Drone {
            sway: None,
            breath_gain_init: None,
        },
        1,
        0,
        VoiceMetadata::default(),
        48_000.0,
        None,
        0,
    )
}

#[test]
fn support_action_held_uses_proposal_clock_and_common_next_window() {
    use crate::life::support_action::{Body, tests::fixture};
    let fixture = fixture();
    let mut landscape = Landscape::new(fixture.snapshot.geometry.space.clone());
    landscape.harmonic.support_weights = fixture.weights;
    let mut control = VoiceControl::default();
    control.pitch.freq = fixture.pitch.exp2();
    control.pitch.range_oct = 0.;
    control.pitch.proposal_interval_sec = Some(0.02);
    let mut voice = drone_voice_with_control(control);
    let tb = Timebase {
        fs: 48000.,
        hop: 64,
    };
    voice.configure_control_hop(tb.hop, tb.hop, tb.fs);
    let rhythms = NeuralRhythms::default();
    let mut batch = PhonationBatch::default();
    let mut opportunities = Vec::new();
    let mut comparisons = Vec::new();
    let mut transitions = Vec::new();
    for at in (512..8512).step_by(tb.hop) {
        voice.set_control_tick(at);
        voice.decide_pitch_target(tb.hop as f32 / tb.fs, &rhythms, &landscape, &[], &[]);
        let opportunity = voice.support.held_opportunities.first().copied();
        if let Some((tick, _)) = opportunity {
            opportunities.push(tick);
        }
        let mut context = fixture.clone();
        let sounding = voice.phonation_engine.has_active_notes();
        context.body = if sounding {
            Body::One {
                pitch: context.pitch,
                energy: context.onset.unwrap(),
            }
        } else {
            Body::Silent
        };
        context.weights.0[0] = if sounding { 1. } else { -1. };
        voice.support.body = context.body;
        voice.phonation_engine.support_action = Some(context);
        voice.phonation_engine.held_comparison = None;
        voice.tick_phonation_into(&tb, at, &rhythms, None, 0., 1., 0.5, &mut batch);
        if let Some(window) = voice.phonation_engine.held_comparison {
            if at != 512 {
                assert_eq!(Some(window), opportunity.map(|(t, end)| (t, end.unwrap())));
            }
            comparisons.push(window);
        }
        for cmd in &batch.cmds {
            if matches!(cmd, ToneCmd::Off { .. } | ToneCmd::On { .. }) {
                assert!(
                    at == 512 || opportunity.is_some(),
                    "Off/On outside a proposal: {at}"
                );
                transitions.push(at);
            }
        }
    }
    assert!(
        transitions.len() >= 4,
        "exercise release and subsequent onset"
    );
    for &(at, end) in comparisons.iter().take(comparisons.len() - 1) {
        assert_eq!(
            opportunities.iter().copied().find(|&tick| tick > at),
            Some(end)
        );
        assert!(end - at > tb.hop as u64);
    }
}

#[test]
fn support_action_hold_keeps_issued_pitch_when_rearticulation_is_declined() {
    use crate::life::support_action::{Body, tests::fixture};
    let mut context = fixture();
    let previous = context.pitch;
    context.body = Body::One {
        pitch: previous,
        energy: context.onset.unwrap(),
    };
    context.allow_release = false;
    let mut control = VoiceControl::default();
    control.pitch.freq = previous.exp2();
    control.pitch.pitch_apply_mode = PitchApplyMode::Rearticulate;
    let mut voice = drone_voice_with_control(control);
    let timebase = Timebase {
        fs: 48_000.,
        hop: 512,
    };
    let rhythms = NeuralRhythms::default();
    let mut batch = PhonationBatch::default();
    voice.tick_phonation_into(&timebase, 0, &rhythms, None, 0., 1., 0.5, &mut batch);
    assert_eq!(batch.tones.len(), 1);
    voice.support.body = context.body;
    voice.phonation_engine.support_action = Some(context);
    let proposed = previous + 1. / 1200.;
    voice.body.set_pitch_log2(proposed);
    voice
        .participation_pitch
        .rearticulations
        .push(Rearticulation {
            at: 1024,
            pitch_log2: proposed,
        });
    voice.support.held_opportunities.push((1024, Some(1536)));
    voice.tick_phonation_into(&timebase, 1024, &rhythms, None, 0., 1., 0.5, &mut batch);
    assert!(batch.onsets.is_empty());
    assert!(batch.cmds.iter().all(
        |cmd| matches!(cmd, ToneCmd::Update { update, .. } if update.target_freq_hz.is_none())
    ));
    assert_eq!(
        voice.body.base_freq_hz().to_bits(),
        previous.exp2().to_bits()
    );
}

#[test]
fn live_amp_controls_update_quiet_held_notes_including_mute_and_resume() {
    let mut control = VoiceControl::default();
    control.body.amp = 0.008;
    let mut voice = drone_voice_with_control(control);
    let timebase = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let rhythms = NeuralRhythms::default();
    let mut batch = PhonationBatch::default();
    voice.tick_phonation_into(&timebase, 0, &rhythms, None, 0.0, 1.0, 0.5, &mut batch);
    let tone_id = batch.tones.first().expect("held tone").tone_id;

    for (index, amp) in [0.006, 0.0, 0.004].into_iter().enumerate() {
        voice
            .apply_patch(&ControlUpdate {
                amp: Some(amp),
                ..Default::default()
            })
            .unwrap();
        let expected = voice.compute_target_amp();
        voice.tick_phonation_into(
            &timebase,
            (index as Tick * 2 + 1) * 64,
            &rhythms,
            None,
            0.0,
            1.0,
            0.5,
            &mut batch,
        );
        assert!(
            batch.tones.is_empty(),
            "control changes must update the existing tone"
        );
        assert!(
            batch.cmds.iter().any(|cmd| matches!(cmd,
                ToneCmd::Update { tone_id: id, update, .. }
                    if *id == tone_id && update.target_amp == Some(expected)
            )),
            "the amplitude command must reach the held tone even below the drift threshold"
        );

        voice
            .apply_patch(&ControlUpdate {
                amp: Some(amp),
                ..Default::default()
            })
            .unwrap();
        voice.tick_phonation_into(
            &timebase,
            (index as Tick * 2 + 2) * 64,
            &rhythms,
            None,
            0.0,
            1.0,
            0.5,
            &mut batch,
        );
        assert!(
            !batch
                .cmds
                .iter()
                .any(|cmd| matches!(cmd, ToneCmd::Update { .. })),
            "an unchanged amplitude must not keep emitting updates"
        );
    }
}

/// Entrain-brained voice with a consonance-viability window, so
/// `ArticulationWrapper::consonance_viability_low()` returns `low` instead
/// of the always-open (0.0) default.
fn entrain_voice_with_viability_low(control: VoiceControl, low: f32) -> Voice {
    let lifecycle = crate::scenario::lifecycle::LifecycleConfig::Sustain {
        endurance_sec: Some(10.0),
        recovery_sec: None,
        attack_cost_fraction: None,
        attack_recharge_fraction: None,
        continuous_recharge_score_low: Some(low),
        continuous_recharge_score_high: Some(0.95),
        selection_environment: false,
        dissonance_penalty: 0.0,
        envelope: crate::scenario::EnvelopeConfig::default(),
    };
    Voice::spawn_from_control(
        control,
        ArticulationCoreConfig::Entrain {
            lifecycle,
            rhythm_freq: None,
            rhythm_coupling: crate::scenario::RhythmCouplingMode::TemporalOnly,
            rhythm_reward: None,
            breath_gain_init: Some(1.0),
        },
        1,
        0,
        VoiceMetadata::default(),
        48_000.0,
        None,
        0,
    )
}

// --- Item A: Hold onset must not be consumed while amp is ~0 ---

#[test]
fn hold_voice_with_zero_amp_defers_onset_until_amp_rises() {
    // VoiceControl::default() phonation spec is once()+while_alive() (Hold).
    let mut control = VoiceControl::default();
    control.body.amp = 0.0;
    let mut voice = drone_voice_with_control(control);
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let rhythms = NeuralRhythms::default();
    let mut batch = PhonationBatch::default();

    voice.tick_phonation_into(&tb, 0, &rhythms, None, 0.0, 1.0, 0.5, &mut batch);
    assert!(
        batch.tones.is_empty(),
        "no tone spec while target amp is ~0"
    );
    assert!(!batch.cmds.iter().any(|c| matches!(c, ToneCmd::On { .. })));

    voice
        .apply_patch(&ControlUpdate {
            amp: Some(0.5),
            ..Default::default()
        })
        .expect("amp patch applies");
    voice.tick_phonation_into(&tb, 64, &rhythms, None, 0.0, 1.0, 0.5, &mut batch);
    assert!(
        !batch.tones.is_empty(),
        "the withheld onset should fire once amp rises above AMP_EPS"
    );
    assert!(batch.cmds.iter().any(|c| matches!(c, ToneCmd::On { .. })));
}

// --- Item B: phonate_when_viable() gate ---

#[test]
fn when_viable_gate_blocks_hold_onset_until_consonance_reaches_low_bound() {
    let mut control = VoiceControl::default();
    control.body.amp = 0.5;
    control.phonation.gate = PhonationGate::WhenViable;
    let mut voice = entrain_voice_with_viability_low(control, 0.5);
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let rhythms = NeuralRhythms::default();
    let mut batch = PhonationBatch::default();

    for hop in 0..5u64 {
        voice.tick_phonation_into(&tb, hop * 64, &rhythms, None, 0.0, 1.0, 0.2, &mut batch);
        assert!(
            batch.tones.is_empty(),
            "gated: no tone spec while consonance stays below the low bound"
        );
        assert!(!batch.cmds.iter().any(|c| matches!(c, ToneCmd::On { .. })));
    }

    // Consonance reaches the low bound: the withheld onset fires.
    voice.tick_phonation_into(&tb, 5 * 64, &rhythms, None, 0.0, 1.0, 0.6, &mut batch);
    assert!(
        !batch.tones.is_empty(),
        "onset should fire once consonance reaches the viability window"
    );
    assert!(batch.cmds.iter().any(|c| matches!(c, ToneCmd::On { .. })));
    assert!(voice.phonation_gate_open);

    // The latch does not re-close: a later consonance drop must not turn
    // the already-phonating note off (metabolism/viability handles
    // survival, not the gate).
    voice.tick_phonation_into(&tb, 6 * 64, &rhythms, None, 0.0, 1.0, 0.0, &mut batch);
    assert!(!batch.cmds.iter().any(|c| matches!(c, ToneCmd::Off { .. })));
    assert!(
        voice.phonation_gate_open,
        "the gate latch must stay open after consonance drops"
    );
}

#[test]
fn when_viable_gate_on_coupled_voice_suppresses_then_resumes_and_stays_open() {
    // Non-Hold (candidate-processing) path: a coupled voice fires onsets
    // repeatedly via `OnsetConfig::Always` on the coupling clock's own
    // timing. base_rate_hz=20 at fs=1000 gives a 50-tick period, matching
    // the 50-tick hop below so each open hop yields an onset.
    let mut control = VoiceControl::default();
    control.body.amp = 0.5;
    control.phonation.gate = PhonationGate::WhenViable;
    control.phonation.spec = crate::scenario::PhonationSpec {
        sustain_participation: false,
        timing: crate::scenario::PhonationTiming::Coupled(crate::scenario::CoupledTimingSpec {
            relation: crate::scenario::RhythmRelation::Synchronized,
            coupling: 0.0,
            base_rate_hz: 20.0,
            flow_depth: 0.0,
            steady: false,
            microtiming: 0.0,
            measure_accent: 0.0,
            role: crate::scenario::RhythmRole::Beat,
            social: 0.0,
            vitality_lambda: 0.0,
            vitality_floor: 0.0,
            reward: 0.0,
        }),
        duration: crate::scenario::DurationSpec::Gates(1),
    };
    let mut voice = entrain_voice_with_viability_low(control, 0.5);
    let tb = Timebase {
        fs: 1000.0,
        hop: 50,
    };
    let rhythms = NeuralRhythms::default();
    let mut batch = PhonationBatch::default();

    let mut gated_onsets = 0usize;
    for hop in 0..6u64 {
        voice.tick_phonation_into(&tb, hop * 50, &rhythms, None, 0.0, 1.0, 0.1, &mut batch);
        gated_onsets += batch.onsets.len();
    }
    assert_eq!(gated_onsets, 0, "no onsets while gated");

    let mut resumed_onsets = 0usize;
    for hop in 6..12u64 {
        voice.tick_phonation_into(&tb, hop * 50, &rhythms, None, 0.0, 1.0, 0.6, &mut batch);
        resumed_onsets += batch.onsets.len();
    }
    assert!(
        resumed_onsets > 0,
        "onsets should resume promptly once the gate opens"
    );

    // Consonance drops again; the latch must stay open (onsets keep firing).
    let mut post_drop_onsets = 0usize;
    for hop in 12..18u64 {
        voice.tick_phonation_into(&tb, hop * 50, &rhythms, None, 0.0, 1.0, 0.1, &mut batch);
        post_drop_onsets += batch.onsets.len();
    }
    assert!(
        post_drop_onsets > 0,
        "the gate latch must not re-close when consonance drops again"
    );
}

mod expression_release_tests {
    use super::*;

    #[test]
    fn expression_stops_late_once_onset_during_population_release_tail() {
        let mut voice = Voice::spawn_from_control(
            VoiceControl::default(),
            ArticulationCoreConfig::Drone {
                sway: None,
                breath_gain_init: None,
            },
            1,
            0,
            VoiceMetadata::default(),
            48000.0,
            None,
            0,
        );
        voice.phonation_engine.expression_active = true;
        voice.start_remove_fade(1.0);
        let tb = Timebase {
            fs: 48000.0,
            hop: 512,
        };
        let mut batch = PhonationBatch::default();
        voice.tick_phonation_into(
            &tb,
            0,
            &NeuralRhythms::default(),
            None,
            0.0,
            1.0,
            0.5,
            &mut batch,
        );
        assert!(batch.tones.is_empty());
        assert!(
            batch
                .cmds
                .iter()
                .all(|cmd| !matches!(cmd, ToneCmd::On { .. }))
        );
    }
}
