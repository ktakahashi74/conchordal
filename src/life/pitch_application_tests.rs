use super::*;

fn setup(hop: usize, entrain: bool) -> (Voice, Timebase, NeuralRhythms, PhonationBatch) {
    let mut control = VoiceControl::default();
    control.pitch.mode = PitchMode::Free;
    control.pitch.freq = 330.0;
    control.pitch.pitch_apply_mode = PitchApplyMode::Rearticulate;
    let core = if entrain {
        ArticulationCoreConfig::default()
    } else {
        ArticulationCoreConfig::Drone {
            sway: None,
            breath_gain_init: None,
        }
    };
    let mut voice = Voice::spawn_from_control(
        control,
        core,
        2,
        0,
        VoiceMetadata::default(),
        48_000.0,
        None,
        21,
    );
    voice.configure_control_hop(hop, 64, 48_000.0);
    let mut batch = PhonationBatch::default();
    voice.prepare_phonation_batch(&mut batch);
    (
        voice,
        Timebase { fs: 48_000.0, hop },
        NeuralRhythms::default(),
        batch,
    )
}

fn emit(v: &mut Voice, tb: &Timebase, rhythms: &NeuralRhythms, at: Tick, out: &mut PhonationBatch) {
    v.tick_phonation_into(tb, at, rhythms, None, 0.0, 1.0, 0.8, out);
}

#[test]
fn rearticulate_one_on_per_request_preserves_sample_pitch_and_old_tail() {
    for hop in [32, 65, 257, 512, 1024] {
        let (mut v, tb, rhythms, mut batch) = setup(hop, false);
        emit(&mut v, &tb, &rhythms, 0, &mut batch);
        assert_eq!(batch.onsets.len(), 1);
        let first = batch.tones[0].tone_id;
        let steps = (hop / 64).max(1);
        for i in 0..steps {
            v.rearticulations.push(Rearticulation {
                at: (hop + i * hop / steps) as Tick,
                pitch_log2: (350.0 + i as f32 * 10.0).log2(),
            });
        }
        let expected = v.rearticulations.clone();
        v.body.set_pitch_log2(expected.last().unwrap().pitch_log2);
        emit(&mut v, &tb, &rhythms, hop as Tick, &mut batch);
        assert_eq!(batch.onsets.len(), steps);
        assert_eq!(batch.tones.len(), steps);
        for (tone, request) in batch.tones.iter().zip(expected) {
            assert_eq!(tone.onset, request.at);
            assert_eq!(tone.freq_hz.to_bits(), request.pitch_log2.exp2().to_bits());
            assert!(batch.cmds.iter().any(|c| matches!(c,
                ToneCmd::On { tone_id, .. } if *tone_id == tone.tone_id)));
        }
        assert!(batch.cmds.iter().any(|c| matches!(c,
            ToneCmd::Off { tone_id, off_tick } if *tone_id == first && *off_tick == hop as Tick)));
        assert!(!batch.cmds.iter().any(|c| matches!(c,
            ToneCmd::Update { tone_id, .. } if *tone_id == first)));
        assert_eq!(v.active_render_notes.len(), 1);
        assert!(v.rearticulations.is_empty());
        emit(&mut v, &tb, &rhythms, 2 * hop as Tick, &mut batch);
        assert!(batch.onsets.is_empty());
    }
}

#[test]
fn rearticulate_controller_uses_formal_community_sample_clock_for_any_hop() {
    use crate::life::community::Community;
    for hop in [32, 65, 257, 512, 1024] {
        let (mut v, tb, rhythms, mut batch) = setup(hop, false);
        let mut landscape = Landscape::new(Log2Space::new(55.0, 4000.0, 64));
        for (i, value) in landscape.consonance_field_score_eff.iter_mut().enumerate() {
            *value = i as f32 * 10.0;
        }
        v.effective_control.pitch.temperature = Some(0.0);
        v.effective_control.pitch.range_oct = 6.0;
        v.effective_control.pitch.proposal_interval_sec = Some(0.000_001);
        v.pitch_ctl.set_adaptation_enabled(false);
        emit(&mut v, &tb, &rhythms, 0, &mut batch);
        let mut community = Community::new(tb);
        community.add_voice(v);
        community.advance(hop, tb.fs, 0, hop as f32 / tb.fs, &landscape);
        let v = &mut community.voices[0];
        let steps = (hop / 64).max(1);
        let count = v.rearticulations.len();
        assert!(count > 0 && count <= steps);
        let grid: Vec<_> = (0..steps)
            .map(|i| (hop + i * hop / steps) as Tick)
            .collect();
        for request in &v.rearticulations {
            assert!(grid.contains(&request.at));
        }
        assert!(v.rearticulations.windows(2).all(|w| w[0].at < w[1].at));
        emit(v, &tb, &rhythms, hop as Tick, &mut batch);
        assert_eq!(batch.tones.len(), count);
    }
}

#[test]
fn rearticulate_no_proposal_rejected_same_target_and_lock_produce_no_on() {
    let (mut v, tb, rhythms, mut batch) = setup(512, false);
    let landscape = Landscape::new(Log2Space::new(55.0, 4000.0, 64));
    v.effective_control.pitch.temperature = Some(0.0);
    v.effective_control.pitch.range_oct = 6.0;
    v.effective_control.pitch.proposal_interval_sec = Some(0.5);
    v.pitch_ctl.set_adaptation_enabled(false);
    emit(&mut v, &tb, &rhythms, 0, &mut batch);
    v.set_control_tick(512);
    v.update_pitch_target(&rhythms, 0.01, &landscape, &[], &[]);
    assert!(!v.pitch_adopted);
    v.set_accumulated_time_for_test(0.5);
    v.update_pitch_target(&rhythms, 0.001, &landscape, &[], &[]);
    assert!(!v.pitch_adopted);
    v.effective_control.pitch.range_oct = 0.0;
    v.set_accumulated_time_for_test(0.5);
    v.update_pitch_target(&rhythms, 0.001, &landscape, &[], &[]);
    assert!(v.rearticulations.is_empty());
    v.apply_patch(&ControlUpdate {
        freq: Some(440.0),
        ..Default::default()
    })
    .unwrap();
    v.set_accumulated_time_for_test(0.5);
    v.update_pitch_target(&rhythms, 0.001, &landscape, &[], &[]);
    assert!(v.rearticulations.is_empty());
    emit(&mut v, &tb, &rhythms, 512, &mut batch);
    assert!(batch.onsets.is_empty());
}

#[test]
fn rearticulate_release_remove_and_death_cancel_requests() {
    for state in 0..3 {
        let (mut v, tb, rhythms, mut batch) = setup(512, true);
        emit(&mut v, &tb, &rhythms, 0, &mut batch);
        v.rearticulations.push(Rearticulation {
            at: 576,
            pitch_log2: 440.0_f32.log2(),
        });
        match state {
            0 => v.release_pending = true,
            1 => v.remove_pending = true,
            _ => {
                let AnyArticulationCore::Entrain(core) = &mut v.articulation.core else {
                    unreachable!()
                };
                core.energy = 0.0;
                core.state = ArticulationState::Idle;
            }
        }
        emit(&mut v, &tb, &rhythms, 512, &mut batch);
        assert!(batch.onsets.is_empty());
        assert!(batch.tones.is_empty());
        assert!(batch.cmds.iter().any(|c| matches!(c, ToneCmd::Off { .. })));
        assert!(v.rearticulations.is_empty());
    }
}

#[test]
fn rearticulate_entrain_cost_once_per_on_and_no_autonomous_pulse() {
    let (mut v, tb, rhythms, mut batch) = setup(512, true);
    let AnyArticulationCore::Entrain(core) = &mut v.articulation.core else {
        unreachable!()
    };
    core.energy = 0.8;
    core.attack_cost_fraction = 0.1;
    core.attack_recharge_fraction = 0.0;
    emit(&mut v, &tb, &rhythms, 0, &mut batch);
    let AnyArticulationCore::Entrain(core) = &v.articulation.core else {
        unreachable!()
    };
    assert!((core.energy - 0.7).abs() < 1e-6);
    assert!(!core.autonomous_attack);
    assert!(matches!(
        batch.tones[0].render_modulator,
        RenderModulatorSpec::EntrainPulse {
            autonomous_pulse: None,
            ..
        }
    ));
    v.rearticulations.push(Rearticulation {
        at: 576,
        pitch_log2: 440.0_f32.log2(),
    });
    emit(&mut v, &tb, &rhythms, 512, &mut batch);
    let AnyArticulationCore::Entrain(core) = &v.articulation.core else {
        unreachable!()
    };
    assert!((core.energy - 0.6).abs() < 1e-6);
}

#[test]
fn rearticulate_delayed_viable_initial_on_uses_latest_pitch_once() {
    let (mut v, tb, rhythms, mut batch) = setup(512, true);
    v.phonation_gate_open = false;
    v.effective_control.phonation.gate = PhonationGate::WhenViable;
    let AnyArticulationCore::Entrain(core) = &mut v.articulation.core else {
        unreachable!()
    };
    core.continuous_recharge_score_low = Some(0.6);
    v.tick_phonation_into(&tb, 0, &rhythms, None, 0.0, 1.0, 0.1, &mut batch);
    assert!(batch.onsets.is_empty());
    assert!(!v.phonation_engine.has_active_notes());
    v.body.set_pitch_log2(440.0_f32.log2());
    emit(&mut v, &tb, &rhythms, 512, &mut batch);
    assert_eq!(batch.onsets.len(), 1);
    assert_eq!(batch.tones[0].onset, 512);
    assert!((batch.tones[0].freq_hz - 440.0).abs() < 0.001);
    emit(&mut v, &tb, &rhythms, 1024, &mut batch);
    assert!(batch.onsets.is_empty());
}

#[test]
fn rearticulate_live_round_trip_preserves_tone_and_cancels_unemitted_requests() {
    let (mut v, tb, rhythms, mut batch) = setup(512, true);
    v.apply_patch(&ControlUpdate {
        pitch_apply_mode: Some(PitchApplyMode::Glide),
        ..Default::default()
    })
    .unwrap();
    emit(&mut v, &tb, &rhythms, 0, &mut batch);
    let tone_id = batch.tones[0].tone_id;
    let freq_bits = v.body.base_freq_hz().to_bits();
    for (i, mode) in [
        PitchApplyMode::Rearticulate,
        PitchApplyMode::Glide,
        PitchApplyMode::Rearticulate,
    ]
    .into_iter()
    .enumerate()
    {
        v.rearticulations.push(Rearticulation {
            at: (512 + i * 512) as Tick,
            pitch_log2: 440.0_f32.log2(),
        });
        v.apply_patch(&ControlUpdate {
            pitch_apply_mode: Some(mode),
            ..Default::default()
        })
        .unwrap();
        assert!(v.rearticulations.is_empty());
        assert_eq!(v.body.base_freq_hz().to_bits(), freq_bits);
        emit(&mut v, &tb, &rhythms, (512 + i * 512) as Tick, &mut batch);
        assert!(batch.onsets.is_empty());
        assert!(
            !batch
                .cmds
                .iter()
                .any(|c| matches!(c, ToneCmd::On { .. } | ToneCmd::Off { .. }))
        );
        let expected_on = mode == PitchApplyMode::Glide;
        assert!(batch.cmds.iter().any(|c| matches!(c, ToneCmd::Update { tone_id: id, update, .. }
            if *id == tone_id && update.autonomous_pulse.is_some_and(|p| p.is_some() == expected_on))));
        assert_eq!(v.active_render_notes.len(), 1);
    }
}

#[test]
fn rearticulate_executor_accepts_same_pitch_without_an_adoption_condition() {
    let (mut v, tb, rhythms, mut batch) = setup(512, false);
    emit(&mut v, &tb, &rhythms, 0, &mut batch);
    let first = batch.tones[0].tone_id;
    // This tests the executor seam; no public policy schedules a same-pitch request.
    v.rearticulations.push(Rearticulation {
        at: 576,
        pitch_log2: v.body.base_freq_hz().log2(),
    });
    emit(&mut v, &tb, &rhythms, 512, &mut batch);
    assert_eq!(batch.onsets.len(), 1);
    assert_ne!(batch.tones[0].tone_id, first);
    assert!((batch.tones[0].freq_hz - 330.0).abs() < 0.001);
    assert!(!v.pitch_adopted);
}

#[test]
#[cfg(feature = "profile-alloc")]
fn rearticulate_emission_and_live_pulse_update_allocate_zero_after_birth() {
    for hop in [257, 1024] {
        let (mut v, tb, rhythms, mut batch) = setup(hop, true);
        emit(&mut v, &tb, &rhythms, 0, &mut batch);
        let steps = (hop / 64).max(1);
        for i in 0..steps {
            v.rearticulations.push(Rearticulation {
                at: (hop + i * hop / steps) as Tick,
                pitch_log2: (350.0 + i as f32).log2(),
            });
        }
        crate::runtime_profile::begin_allocations();
        emit(&mut v, &tb, &rhythms, hop as Tick, &mut batch);
        let counts = crate::runtime_profile::finish_allocations().unwrap();
        assert_eq!(counts.count, 0);
        assert_eq!(counts.bytes, 0);
        assert_eq!(batch.onsets.len(), steps);
        v.apply_patch(&ControlUpdate {
            pitch_apply_mode: Some(PitchApplyMode::Glide),
            ..Default::default()
        })
        .unwrap();
        crate::runtime_profile::begin_allocations();
        emit(&mut v, &tb, &rhythms, 2 * hop as Tick, &mut batch);
        let counts = crate::runtime_profile::finish_allocations().unwrap();
        assert_eq!(counts.count, 0);
        assert_eq!(counts.bytes, 0);
    }
}
