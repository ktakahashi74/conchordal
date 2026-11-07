use super::*;
use rand::{SeedableRng, rngs::SmallRng};

fn fixture() -> Landscape {
    let mut l = Landscape::new(Log2Space::new(110.0, 880.0, 96));
    l.subjective_intensity[l.space.nearest_index(220.0)] = 1.0;
    l.subjective_intensity[l.space.nearest_index(330.0)] = 0.7;
    l.loudness_exp = 0.23;
    let density = l.subjective_intensity.clone();
    let score = recompute_loo_score_scan(&l, &density, &mut None);
    l.consonance_field_score.clone_from(&score);
    l.consonance_field_score_eff = score;
    l
}

#[test]
fn zero_self_preserves_observed_field() {
    let l = fixture();
    let mut input = LooInput {
        own_power_scan: vec![0.0; l.space.n_bins()],
        other_power_scan: vec![1.0; l.space.n_bins()],
        ..Default::default()
    };
    assert!(approx_loo_score_scan(&l, &mut input, &mut None).is_none());
    for &p in &l.space.centers_log2 {
        assert_eq!(
            sample_consonance_score_with_loo(
                p,
                p,
                &l,
                true,
                ReferenceLooMode::ApproxHarmonics,
                1,
                None,
                None
            ),
            l.evaluate_pitch_score_log2(p)
        );
    }
}

#[test]
fn shared_unison_leaves_other_voice_power() {
    let l = fixture();
    let n = l.space.n_bins();
    let idx = l.space.nearest_index(220.0);
    let mut own = vec![0.0; n];
    own[idx] = 1.0;
    let mut total = vec![0.0; n];
    total[idx] = 3.0;
    let mut input = LooInput {
        own_power_scan: own,
        other_power_scan: total,
        ..Default::default()
    };
    let actual = approx_loo_score_scan(&l, &mut input, &mut None).unwrap();
    let mut residual = l.subjective_intensity.clone();
    residual[idx] *= 0.75f32.powf(l.loudness_exp);
    assert!(residual[idx] > 0.0);
    let expected = recompute_loo_score_scan(&l, &residual, &mut None);
    assert_eq!(&*actual, expected.as_slice());
    let exact = exact_loo_consonance_score_scan(
        &l,
        220.0f32.log2(),
        true,
        ReferenceLooMode::ExactScan,
        &mut None,
    )
    .unwrap();
    assert!(actual.iter().zip(exact).any(|(a, b)| (a - b).abs() > 1e-6));
}

#[test]
fn isolated_fundamental_matches_exact_ranks_and_adoption() {
    let mut l = fixture();
    l.perc_habituation_state_scan.fill(0.35);
    let n = l.space.n_bins();
    let idx = l.space.nearest_index(220.0);
    let current = l.space.centers_log2[idx];
    let mut own = vec![0.0; n];
    own[idx] = 1.0;
    let mut input = LooInput {
        own_power_scan: own.clone(),
        other_power_scan: vec![0.; n],
        ..Default::default()
    };
    let approx = approx_loo_score_scan(&l, &mut input, &mut None).unwrap();
    let exact =
        exact_loo_consonance_score_scan(&l, current, true, ReferenceLooMode::ExactScan, &mut None)
            .unwrap();
    // Both residuals are the same; no score-error budget is needed.
    assert_eq!(&*approx, exact.as_slice());
    let mut a = PitchHillClimbPitchCore::new(200.0, current, 0.0);
    a.set_leave_self_out(true);
    a.set_temperature(0.12);
    input.exact_score = Some(approx.clone());
    a.loo_input = input;
    let mut b = a.clone();
    b.set_reference_loo_mode(ReferenceLooMode::ExactScan);
    let mut perceptual = AdaptationContext::from_config(&Default::default(), n);
    perceptual.set_enabled(false);
    let features = FeaturesNow::from_occupancy_scan(&l.space, &l.subjective_intensity);
    for seed in 0..64 {
        let mut ra = SmallRng::seed_from_u64(seed);
        let mut rb = ra.clone();
        let pa = a.propose_target(
            current,
            current,
            220.0,
            2.0,
            &l,
            &perceptual,
            &features,
            &[],
            &mut ra,
        );
        let pb = b.propose_target(
            current,
            current,
            220.0,
            2.0,
            &l,
            &perceptual,
            &features,
            &[],
            &mut rb,
        );
        assert_eq!(pa.target_pitch_log2, pb.target_pitch_log2);
        assert_eq!(pa.salience, pb.salience);
    }
}

#[test]
#[should_panic(expected = "loo_own_power_scan")]
fn ownership_scan_boundary_is_checked() {
    let l = fixture();
    let mut input = LooInput {
        own_power_scan: vec![1.0],
        other_power_scan: vec![1.0],
        ..Default::default()
    };
    approx_loo_score_scan(&l, &mut input, &mut None);
}

#[test]
fn exact_missing_fundamental_retains_previous_approximate_fallback() {
    let mut l = Landscape::new(Log2Space::new(110.0, 880.0, 96));
    l.consonance_field_score_eff.fill(1.0);
    let current = 220.0f32.log2();
    assert!(
        exact_loo_consonance_score_scan(&l, current, true, ReferenceLooMode::ExactScan, &mut None)
            .is_none()
    );
    let actual = sample_consonance_score_with_loo(
        current,
        current,
        &l,
        true,
        ReferenceLooMode::ExactScan,
        1,
        None,
        None,
    );
    let previous = legacy_sample_consonance_score_with_loo(
        current,
        current,
        &l,
        true,
        ReferenceLooMode::ExactScan,
        1,
        None,
        None,
    );
    assert_eq!(actual, previous);
    assert_eq!(actual, 0.0);
}

#[test]
#[should_panic(expected = "loo_other_power_scan")]
fn other_ownership_scan_boundary_is_checked() {
    let l = fixture();
    let mut input = LooInput {
        own_power_scan: vec![0.0; l.space.n_bins()],
        ..Default::default()
    };
    approx_loo_score_scan(&l, &mut input, &mut None);
}

#[test]
fn merged_residual_returns_to_external_projection_and_conserves_mass() {
    let l = fixture();
    let n = l.space.n_bins();
    let idx = l.space.nearest_index(220.0);
    let mut input = LooInput {
        own_power_scan: vec![0.; n],
        other_power_scan: vec![0.; n],
        ..Default::default()
    };
    input.own_power_scan[idx] = 1.;
    input.other_power_scan[idx + 1] = 1.;
    input.other_power_scan[idx + 2] = 3.;
    let mut scratch = None;
    let cache = prepare_loo_scratch(&l, &mut scratch);
    let mut density = l.subjective_intensity.clone();
    assert!(redistribute_loo_density(&l, &input, cache, &mut density));
    assert_eq!(density[idx], 0.);
    let mass = |i| density[i] as f64 * cache.potentials.du[i] as f64;
    let expected = l.subjective_intensity[idx] as f64
        * cache.potentials.du[idx] as f64
        * 0.8_f64.powf(l.loudness_exp as f64);
    assert!((mass(idx + 1) + mass(idx + 2) - expected).abs() < expected * f32::EPSILON as f64);
    assert!((mass(idx + 2) / mass(idx + 1) - 3.).abs() < 3. * f32::EPSILON as f64);
    let far = l.space.nearest_index(330.);
    assert_eq!(density[far], l.subjective_intensity[far]);
}

#[test]
fn quiet_external_power_survives_unison_and_neighbour_removal() {
    let l = fixture();
    let n = l.space.n_bins();
    let idx = l.space.nearest_index(220.0);
    let mut scratch = None;
    let cache = prepare_loo_scratch(&l, &mut scratch);
    for external in [idx, idx + 1] {
        let mut input = LooInput {
            own_power_scan: vec![0.; n],
            other_power_scan: vec![0.; n],
            ..Default::default()
        };
        input.own_power_scan[idx] = 1.;
        input.other_power_scan[external] = 1e-30;
        assert_eq!(1_f64 + input.other_power_scan[external], 1.);
        let mut density = l.subjective_intensity.clone();
        redistribute_loo_density(&l, &input, cache, &mut density);
        assert!(density[external] > 0.);
    }
}

#[test]
fn density_floor_silences_both_potentials_on_reference_and_loo_paths() {
    let mut l = fixture();
    l.subjective_intensity.fill(0.);
    let idx = l.space.nearest_index(220.);
    let mut scratch = None;
    let cache = prepare_loo_scratch(&l, &mut scratch);
    let eps = l.roughness_ref_eps.max(1e-12);
    l.subjective_intensity[idx] = eps / (2. * cache.potentials.du[idx]);
    let score = recompute_loo_score_scan(&l, &l.subjective_intensity, &mut scratch);
    assert!(
        score
            .iter()
            .all(|s| *s == l.consonance_kernel.score(0., 0.))
    );
    let cache = scratch.as_mut().unwrap();
    assert!(cache.h.iter().chain(&cache.r).all(|v| *v == 0.));
}

#[cfg(feature = "profile-alloc")]
#[test]
fn redistribution_steady_state_allocates_nothing() {
    let mut l = fixture();
    l.nsgt_power = l.subjective_intensity.clone();
    let n = l.space.n_bins();
    let idx = l.space.nearest_index(220.);
    let mut input = LooInput {
        own_power_scan: vec![0.; n],
        other_power_scan: vec![0.; n],
        ..Default::default()
    };
    input.own_power_scan[idx] = 1.;
    input.other_power_scan[idx + 2] = 0.3;
    let mut scratch = None;
    let cache = prepare_loo_scratch(&l, &mut scratch);
    let mut density = l.subjective_intensity.clone();
    crate::runtime_profile::begin_allocations();
    let start = crate::clock::Instant::now();
    for _ in 0..300 {
        density.copy_from_slice(&l.subjective_intensity);
        std::hint::black_box(redistribute_loo_density(&l, &input, cache, &mut density));
    }
    let elapsed = start.elapsed();
    let counts = crate::runtime_profile::finish_allocations().unwrap();
    println!(
        "T36_REDISTRIBUTION_ALLOC counts={counts:?} iterations=300 bins={n} elapsed={elapsed:?}"
    );
    assert_eq!((counts.count, counts.bytes), (0, 0));
}

#[test]
fn rearticulation_candidate_view_preserves_holds_and_glides() {
    let space = Log2Space::new(55., 8000., 96);
    let previous = space.centers_log2[100];
    let mut input = LooInput {
        rearticulation: Some(RearticulationView {
            weight: 1.0,
            id: 1,
            generation: 0,
            reference_log2: previous,
            threshold_cents: 50.,
            previous_target: previous,
            bounds: (space.fmin.log2(), space.fmax.log2()),
        }),
        rearticulation_score: vec![1.; space.n_bins()],
        ..Default::default()
    };
    let external = vec![0.; space.n_bins()];
    for (step, expected) in [(0, 0.), (1, 0.), (4, 0.), (5, 1.)] {
        let p = space.centers_log2[100 + step];
        assert_eq!(
            input.candidate_scan(&space, p, Some(&external)).unwrap()[0],
            expected
        );
    }
    input.rearticulation.as_mut().unwrap().threshold_cents = 0.;
    assert_eq!(
        input
            .candidate_scan(&space, previous, Some(&external))
            .unwrap()[0],
        0.
    );
    input.rearticulation.as_mut().unwrap().bounds = (previous, previous);
    assert_eq!(
        input
            .candidate_scan(&space, space.centers_log2[120], Some(&external))
            .unwrap()[0],
        0.
    );
    assert!(input.candidate_scan(&space, previous, None).is_none());
    input.rearticulation = None;
    assert_eq!(
        input
            .candidate_scan(&space, previous, Some(&external))
            .unwrap(),
        external
    );
}

#[test]
#[cfg(feature = "profile-alloc")]
fn rearticulation_view_reuses_proposal_storage() {
    let mut land = Landscape::new(Log2Space::new(55., 8000., 96));
    let n = land.space.n_bins();
    land.loo.push(crate::core::stream::loo::Observation {
        support: None,
        id: 1,
        generation: 0,
        h_scan: vec![0.3; n],
        r_scan: vec![0.1; n],
        score_scan: vec![0.2; n].into(),
    });
    let mut input = LooInput {
        exact_score: Some(land.loo[0].score_scan.clone()),
        rearticulation: Some(RearticulationView {
            weight: 1.0,
            id: 1,
            generation: 0,
            reference_log2: 8.,
            threshold_cents: 0.,
            previous_target: 8.,
            bounds: land.freq_bounds_log2(),
        }),
        ..Default::default()
    };
    loo_score_scan(&land, &mut input);
    crate::runtime_profile::begin_allocations();
    for _ in 0..128 {
        let scan = loo_score_scan(&land, &mut input);
        std::hint::black_box(input.candidate_scan(&land.space, 8.1, scan.as_deref()));
    }
    let counts = crate::runtime_profile::finish_allocations().unwrap();
    assert_eq!(counts.count, 0);
    assert_eq!(counts.bytes, 0);
    println!("VR1_ALLOCATION {counts:?}");
}

#[test]
fn harmonic_opportunity_uses_the_same_separate_pitch_view() {
    use crate::core::{modulation::NeuralRhythms, timebase::Timebase};
    use crate::life::voice::{PhonationBatch, Voice, VoiceMetadata};
    use crate::scenario::{
        ArticulationCoreConfig,
        control::{PitchApplyMode, PitchMode, VoiceControl},
    };
    for opportunity in [false, true] {
        let mut land = Landscape::new(Log2Space::new(55., 2000., 96));
        let n = land.space.n_bins();
        land.roughness01.fill(0.2);
        let external: Arc<[f32]> = vec![0.; n].into();
        land.loo.push(crate::core::stream::loo::Observation {
            support: None,
            id: 1,
            generation: 0,
            h_scan: vec![1.; n],
            r_scan: vec![0.; n],
            score_scan: external.clone(),
        });
        let mut c = VoiceControl::default();
        c.pitch.mode = PitchMode::Free;
        c.pitch.freq = 220.;
        c.pitch.range_oct = 4.;
        c.pitch.pitch_apply_mode = PitchApplyMode::Rearticulate;
        c.pitch.proposal_interval_sec = Some(0.001);
        c.pitch.leave_self_out = true;
        c.pitch.tail_roughness_aversion = 1.0;
        let mut voice = Voice::spawn_from_control(
            c,
            ArticulationCoreConfig::Drone {
                sway: None,
                breath_gain_init: None,
            },
            1,
            0,
            VoiceMetadata::default(),
            48000.,
            Some(&land),
            21,
        );
        voice.harmonic_member = true;
        voice.configure_control_hop(512, 64, 48000.);
        voice.pitch_ctl.loo_input_mut().exact_score = Some(external.clone());
        let mut batch = PhonationBatch::default();
        let tb = Timebase {
            fs: 48000.,
            hop: 512,
        };
        voice.tick_harmonic(&tb, 0, &land, &[], &mut batch);
        assert_eq!(batch.tones.len(), 1);
        voice.tick_harmonic(
            &tb,
            512,
            &land,
            if opportunity { &[600] } else { &[] },
            &mut batch,
        );
        let input = voice.pitch_ctl.loo_input_mut();
        assert_eq!(!input.rearticulation_score.is_empty(), opportunity);
        assert_eq!(input.exact_score.as_ref().unwrap(), &external);
        assert_eq!(land.loo[0].score_scan, external);
        assert_eq!(
            batch
                .cmds
                .iter()
                .any(|c| matches!(c, crate::life::phonation_engine::ToneCmd::Off { .. })),
            opportunity
        );
        assert_eq!(!batch.tones.is_empty(), opportunity);
        voice.start_remove_fade(1.);
        voice.tick_harmonic(&tb, 1024, &land, &[1100], &mut batch);
        assert!(batch.tones.is_empty());
        assert!(voice.pitch_ctl.loo_input_mut().rearticulation.is_none());
        // Ordinary updates do not run a second proposal for this member.
        voice.update_pitch_target(&NeuralRhythms::default(), 1., &land, &[], &[]);
    }
}

#[test]
fn tail_roughness_aversion_interpolates_r_without_changing_h_or_shared_observation() {
    use crate::core::consonance_kernel::ConsonanceKernel;
    for (external, mixed) in [(0.1, 0.9), (0.8, 0.2), (0.4, 0.4)] {
        let mut land = Landscape::new(Log2Space::new(55., 8000., 96));
        let n = land.space.n_bins();
        land.roughness01.fill(mixed);
        // Read R directly through the actual field path, then check the full kernel.
        for kernel in [
            ConsonanceKernel {
                a: 0.,
                b: 1.,
                c: 0.,
                d: 0.,
            },
            ConsonanceKernel::default(),
        ] {
            land.consonance_kernel = kernel;
            land.loo = vec![crate::core::stream::loo::Observation {
                support: None,
                id: 1,
                generation: 0,
                h_scan: vec![0.3; n],
                r_scan: vec![external; n],
                score_scan: vec![kernel.score(0.3, external); n].into(),
            }];
            let original = land.loo[0].score_scan.clone();
            let mut input = LooInput {
                exact_score: Some(original.clone()),
                ..Default::default()
            };
            let mut previous_r = external;
            for weight in [0., 0.125, 0.25, 0.5, 0.75, 1., 0.] {
                input.rearticulation = Some(RearticulationView {
                    weight,
                    id: 1,
                    generation: 0,
                    reference_log2: 8.,
                    threshold_cents: 0.,
                    previous_target: 8.,
                    bounds: land.freq_bounds_log2(),
                });
                let external_scan = loo_score_scan(&land, &mut input).unwrap();
                let actual = input
                    .candidate_scan(&land.space, 8.1, Some(&external_scan))
                    .unwrap()[0];
                let r = if weight == 0. {
                    external
                } else if weight == 1. {
                    mixed
                } else {
                    (1. - weight) * external + weight * mixed
                };
                assert_eq!(actual.to_bits(), kernel.score(0.3, r).to_bits());
                if kernel.a == 0. && weight > 0. {
                    assert!(actual >= external.min(mixed) && actual <= external.max(mixed));
                    assert!((actual - previous_r) * (mixed - external) >= 0.);
                    previous_r = actual;
                }
                if weight == 0. {
                    assert!(input.rearticulation_score.is_empty());
                }
                assert_eq!(&*land.loo[0].score_scan, &*original);
                assert!(land.loo[0].h_scan.iter().all(|&h| h == 0.3));
            }
        }
    }
}

#[test]
fn hr2_empty_target_bounds_project_rearticulation_to_voice_range() {
    let space = crate::core::log2space::production_space(48000).unwrap();
    let previous = 8000.0_f32.log2();
    let input = LooInput {
        rearticulation: Some(RearticulationView {
            weight: 1.0,
            id: 1,
            generation: 0,
            reference_log2: previous,
            threshold_cents: 0.0,
            previous_target: previous,
            bounds: (1.0, 0.0),
        }),
        rearticulation_score: vec![1.0; space.n_bins()],
        ..Default::default()
    };
    let external = vec![0.0; space.n_bins()];
    assert_eq!(
        input
            .candidate_scan(&space, 10000.0_f32.log2(), Some(&external))
            .unwrap(),
        external
    );
}
