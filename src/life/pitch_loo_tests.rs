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
    assert!(loo_score_scan(&l, &mut input, &mut None).is_none());
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
    let actual = loo_score_scan(&l, &mut input, &mut None).unwrap();
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
    let approx = loo_score_scan(&l, &mut input, &mut None).unwrap();
    let exact =
        exact_loo_consonance_score_scan(&l, current, true, ReferenceLooMode::ExactScan, &mut None)
            .unwrap();
    // Both residuals are the same; no score-error budget is needed.
    assert_eq!(&*approx, exact.as_slice());
    let mut a = PitchHillClimbPitchCore::new(200.0, current, 0.0);
    a.set_leave_self_out(true);
    a.set_temperature(0.12);
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
    loo_score_scan(&l, &mut input, &mut None);
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
    loo_score_scan(&l, &mut input, &mut None);
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
    let start = std::time::Instant::now();
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
