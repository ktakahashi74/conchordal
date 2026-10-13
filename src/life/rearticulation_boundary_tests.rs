use super::*;

fn fixture() -> (Voice, Landscape, Timebase, NeuralRhythms, PhonationBatch) {
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut control = VoiceControl::default();
    control.pitch.mode = PitchMode::Free;
    control.pitch.freq = space.centers_log2[100].exp2();
    control.pitch.pitch_apply_mode = PitchApplyMode::Rearticulate;
    control.pitch.pitch_glide_tau_sec = 0.03;
    let mut v = Voice::spawn_from_control(
        control,
        ArticulationCoreConfig::Drone {
            sway: None,
            breath_gain_init: None,
        },
        2,
        0,
        VoiceMetadata::default(),
        48000.0,
        None,
        21,
    );
    v.effective_control.pitch.rearticulate_above_cents = 50.0;
    v.configure_control_hop(512, 64, 48_000.0);
    let mut out = PhonationBatch::default();
    v.prepare_phonation_batch(&mut out);
    (
        v,
        Landscape::new(space),
        Timebase {
            fs: 48000.0,
            hop: 512,
        },
        NeuralRhythms::default(),
        out,
    )
}

fn emit(v: &mut Voice, tb: &Timebase, r: &NeuralRhythms, at: Tick, out: &mut PhonationBatch) {
    v.tick_phonation_into(tb, at, r, None, 0.0, 1.0, 0.8, out);
}

fn adopt(v: &mut Voice, land: &Landscape, r: &NeuralRhythms, index: usize, at: Tick) {
    v.effective_control.pitch.freq = land.space.centers_log2[index].exp2();
    v.effective_control.pitch.range_oct = 0.0;
    v.effective_control.pitch.proposal_interval_sec = Some(0.5);
    v.set_accumulated_time_for_test(0.5);
    v.set_control_tick(at);
    v.update_pitch_target(r, 0.001, land, &[], &[]);
    assert!(v.pitch_adopted);
    v.update_articulation_autonomous(0.001, r);
}

#[test]
fn boundary_is_inclusive_for_exact_bins_without_an_epsilon() {
    let space = Log2Space::new(55.0, 8000.0, 96);
    let pitch = crate::scenario::control::PitchControl {
        mode: PitchMode::Free,
        pitch_apply_mode: PitchApplyMode::Rearticulate,
        ..Default::default()
    };
    for (steps, threshold) in [(1, 12.5), (4, 50.0), (8, 100.0)] {
        let mut pitch = pitch.clone();
        pitch.rearticulate_above_cents = threshold;
        for i in 20..space.n_bins() - 20 {
            for sign in [-1isize, 1] {
                for (offset, expected) in [
                    (steps, PitchApplyMode::Glide),
                    (steps + 1, PitchApplyMode::Rearticulate),
                ] {
                    let j = (i as isize + sign * offset as isize) as usize;
                    assert_eq!(
                        select_pitch_application(
                            &pitch,
                            Some((&space, space.centers_log2[i], space.centers_log2[j]))
                        ),
                        expected
                    );
                }
            }
        }
    }
    let off_grid = space.centers_log2[100] + 0.003;
    assert_eq!(
        onset_distance_cents(&space, off_grid, space.centers_log2[101]),
        1200.0 * (space.centers_log2[101] as f64 - off_grid as f64).abs()
    );
}

#[test]
fn small_steps_glide_but_net_movement_from_actual_on_restarts() {
    let (mut v, land, tb, r, mut out) = fixture();
    emit(&mut v, &tb, &r, 0, &mut out);
    let mut attacks = 0;
    for step in 1..=5 {
        adopt(&mut v, &land, &r, 100 + step, step as Tick * 512 + 64);
        let choice = v.adoption_application;
        emit(&mut v, &tb, &r, step as Tick * 512, &mut out);
        attacks += out.onsets.len();
        if step != 5 {
            assert_eq!(choice, PitchApplyMode::Glide);
            assert!(out.onsets.is_empty());
            assert!(v.body.base_freq_hz().log2() < v.target_pitch_log2());
        }
    }
    assert_eq!(attacks, 1);
}

#[test]
fn earlier_queued_on_is_the_reference_within_one_output_hop() {
    let (mut v, land, tb, r, mut out) = fixture();
    emit(&mut v, &tb, &r, 0, &mut out);
    adopt(&mut v, &land, &r, 105, 576);
    adopt(&mut v, &land, &r, 106, 640);
    assert_eq!(v.rearticulations.len(), 1);
    assert_eq!(v.adoption_application, PitchApplyMode::Glide);
    emit(&mut v, &tb, &r, 512, &mut out);
    assert_eq!(out.onsets.len(), 1);
    assert_eq!(out.tones[0].onset, 576);
    assert_eq!(
        v.last_onset_pitch_log2.unwrap().to_bits(),
        out.tones[0].freq_hz.log2().to_bits()
    );
}

#[test]
fn lifecycle_cancel_does_not_replace_the_last_actual_on() {
    let (mut v, land, tb, r, mut out) = fixture();
    emit(&mut v, &tb, &r, 0, &mut out);
    let actual = v.last_onset_pitch_log2;
    adopt(&mut v, &land, &r, 105, 576);
    v.release_pending = true;
    emit(&mut v, &tb, &r, 512, &mut out);
    assert!(out.onsets.is_empty());
    assert!(v.rearticulations.is_empty());
    assert_eq!(v.last_onset_pitch_log2, actual);
}

#[test]
fn disabled_policy_and_lock_keep_the_configured_selection() {
    let mut p = crate::scenario::control::PitchControl {
        mode: PitchMode::Free,
        pitch_apply_mode: PitchApplyMode::Rearticulate,
        ..Default::default()
    };
    assert_eq!(
        select_pitch_application(&p, None),
        PitchApplyMode::Rearticulate
    );
    p.mode = PitchMode::Lock;
    let space = Log2Space::new(55.0, 8000.0, 96);
    p.rearticulate_above_cents = 50.0;
    assert_eq!(
        select_pitch_application(&p, Some((&space, 8.0, 9.0))),
        PitchApplyMode::GateSnap
    );
}

#[test]
#[cfg(feature = "profile-alloc")]
fn glide_selection_and_actual_on_reuse_birth_storage() {
    let (mut v, land, tb, r, mut out) = fixture();
    emit(&mut v, &tb, &r, 0, &mut out);
    v.set_control_tick(576);
    v.pitch_ctl
        .force_set_target_pitch_log2(land.space.centers_log2[101]);
    crate::runtime_profile::begin_allocations();
    v.select_adoption_application(&land.space, land.space.centers_log2[100]);
    let small = v.adoption_application;
    v.update_articulation_autonomous(0.001, &r);
    v.pitch_ctl
        .force_set_target_pitch_log2(land.space.centers_log2[105]);
    v.set_control_tick(640);
    v.select_adoption_application(&land.space, land.space.centers_log2[101]);
    let large = v.adoption_application;
    v.rearticulations.push(Rearticulation {
        at: 640,
        pitch_log2: v.target_pitch_log2(),
    });
    emit(&mut v, &tb, &r, 512, &mut out);
    let a = crate::runtime_profile::finish_allocations().unwrap();
    assert_eq!(small, PitchApplyMode::Glide);
    assert_eq!(large, PitchApplyMode::Rearticulate);
    assert_eq!(out.onsets.len(), 1);
    assert_eq!((a.count, a.bytes), (0, 0));
}

#[test]
fn live_boundary_keeps_choice_until_next_adoption_and_cancels_no_tone() {
    let (mut v, land, tb, r, mut out) = fixture();
    emit(&mut v, &tb, &r, 0, &mut out);
    adopt(&mut v, &land, &r, 101, 576);
    v.apply_patch(&ControlUpdate {
        rearticulate_above_cents: Some(0.0),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(v.adoption_application, PitchApplyMode::Glide);
    emit(&mut v, &tb, &r, 512, &mut out);
    assert!(out.onsets.is_empty());
    adopt(&mut v, &land, &r, 102, 1088);
    emit(&mut v, &tb, &r, 1024, &mut out);
    assert_eq!(out.onsets.len(), 1);
}

#[test]
fn rearticulation_reference_is_net_displacement_and_birth_state_is_empty() {
    let (mut v, land, tb, r, mut out) = fixture();
    assert!(v.last_onset_pitch_log2.is_none());
    emit(&mut v, &tb, &r, 0, &mut out);
    adopt(&mut v, &land, &r, 104, 576);
    emit(&mut v, &tb, &r, 512, &mut out);
    adopt(&mut v, &land, &r, 100, 1088);
    emit(&mut v, &tb, &r, 1024, &mut out);
    assert!(out.onsets.is_empty());
}
