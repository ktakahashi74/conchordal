use super::candidates::{
    cents_to_log2, sample_softmax_candidate, sample_softmax_with_relations, top_global_peaks,
    top_local_peaks,
};
use super::scoring::{
    DEFAULT_LOO_HARMONICS, DEFAULT_MOVE_COST_COEFF, DEFAULT_MOVE_COST_EXP,
    EXACT_LOO_ROUGHNESS_ERB_STEP, ReferenceLooMode, adjusted_pitch_score,
    adjusted_pitch_score_with_loo_harmonics, exact_loo_consonance_score_scan,
    legacy_sample_consonance_score_with_loo, sample_consonance_score_with_loo,
};
use super::{PitchCore, PitchHillClimbPitchCore, PitchPeakSamplerCore};
use crate::core::harmonicity_kernel::HarmonicityKernel;
use crate::core::landscape::Landscape;
use crate::core::log2space::Log2Space;
use crate::core::psycho_state::roughness_ratio_to_state01;
use crate::core::roughness_kernel::RoughnessKernel;
use crate::life::adaptation::{AdaptationConfig, AdaptationContext, FeaturesNow};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::sync::Arc;

fn melodic_snapshot(xs: &[f64]) -> crate::life::melodic_continuity::Snapshot {
    let mut prefix = crate::life::melodic_continuity::Prefix::default();
    for &p in xs {
        prefix.push(p);
    }
    crate::life::melodic_continuity::Snapshot {
        distribution: prefix.distribution().unwrap(),
        bounds: (5.0, 62.0 / 12.0),
        weight: 1.0,
        epoch: 0,
        n: xs.len(),
    }
}

#[test]
fn harmonic_relation_selectors_preserve_zero_and_unknown_rng_states() {
    use crate::temporal_cognition::harmonic_relations::consumer::test_use;
    use rand::SeedableRng;
    for weight in [0., 1.] {
        let (temporal, pitches) = test_use(weight, false);
        for seed in 0..32 {
            let mut a = rand::rngs::SmallRng::seed_from_u64(seed);
            let mut b = a.clone();
            let scored = [(pitches[0], -0.3), (pitches[1], 0.8)];
            let old = sample_softmax_candidate(&scored, 0.5, &mut a);
            let new = sample_softmax_with_relations(&scored, 0.5, &mut b, Some(&temporal), None);
            assert_eq!(old.to_bits(), new.to_bits());
            assert_eq!(a.random::<u64>(), b.random::<u64>());
        }
    }
}

#[test]
fn harmonic_relation_softmax_keeps_native_class_and_conditional_quantile() {
    use crate::temporal_cognition::harmonic_relations::consumer::test_use;
    use rand::SeedableRng;
    for weight in [1e-12, 1., -1.] {
        let (temporal, pitches) = test_use(weight, true);
        let current = (pitches[0] + pitches[1]) * 0.5;
        let scored = [(current, 0.), (pitches[0], 0.), (pitches[1], 0.)];
        let mut changed = 0;
        for seed in 0..1024 {
            let mut a = rand::rngs::SmallRng::seed_from_u64(seed);
            let mut b = a.clone();
            let old = sample_softmax_candidate(&scored, 0.5, &mut a);
            let new =
                sample_softmax_with_relations(&scored, 0.5, &mut b, Some(&temporal), Some(current));
            assert_eq!(old == current, new == current);
            assert_eq!(a.random::<u64>(), b.random::<u64>());
            changed += usize::from(old != new);
        }
        if weight == 1e-12 {
            assert_eq!(
                changed, 0,
                "vanishing support must not resample destinations"
            );
        } else {
            assert!(
                changed > 0,
                "signed support must reach the sampled destination"
            );
        }
    }
}

#[test]
fn harmonic_relation_greedy_and_metropolis_use_the_existing_candidates() {
    use crate::temporal_cognition::harmonic_relations::consumer::test_use;
    use rand::SeedableRng;
    let (temporal, pitches) = test_use(1., true);
    let mut changed = PitchHillClimbPitchCore::new(100., 440., 0.);
    let old = changed.clone();
    changed.loo_input.relations = Some(temporal);
    let mut a = rand::rngs::SmallRng::seed_from_u64(14);
    let mut b = a.clone();
    // Relation support for the current pitch cannot turn a baseline move into hold.
    let scored = vec![(pitches[0], 0.), (pitches[1], 1.)];
    let before = old.select_scored(pitches[0], 0., &mut scored.clone(), &mut a, None);
    let after = changed.select_scored(pitches[0], 0., &mut scored.clone(), &mut b, None);
    assert_eq!(before.target_pitch_log2, after.target_pitch_log2);
    assert_eq!(a.random::<u64>(), b.random::<u64>());
    // It can distinguish two moving destinations without discarding native scores.
    let current = (pitches[0] + pitches[1]) * 0.5;
    let scored = vec![(pitches[0], 0.2), (pitches[1], 0.3)];
    let before = old.select_scored(current, 0., &mut scored.clone(), &mut a, None);
    let after = changed.select_scored(current, 0., &mut scored.clone(), &mut b, None);
    assert_eq!(before.target_pitch_log2, pitches[1]);
    assert_eq!(after.target_pitch_log2, pitches[0]);
    changed.set_temperature(1.);
    let mut old = old;
    old.set_temperature(1.);
    let mut difference = false;
    for seed in 0..64 {
        let mut a = rand::rngs::SmallRng::seed_from_u64(seed);
        let mut b = a.clone();
        let scored = vec![(pitches[0], -0.1), (pitches[1], -0.2)];
        let before = old.select_scored(current, 0., &mut scored.clone(), &mut a, None);
        let after = changed.select_scored(current, 0., &mut scored.clone(), &mut b, None);
        assert_eq!(
            before.target_pitch_log2 == current,
            after.target_pitch_log2 == current
        );
        assert!(after.target_pitch_log2 == current || pitches.contains(&after.target_pitch_log2));
        difference |= before.target_pitch_log2 != after.target_pitch_log2;
        assert_eq!(a.random::<u64>(), b.random::<u64>());
    }
    assert!(difference);
}

#[test]
fn support_movement_uses_one_loo_reference_and_ignores_mixed_without_loo() {
    use crate::core::harmonic_support::{Geometry, Snapshot};
    use crate::core::harmonicity_kernel::HarmonicityParams;
    let mut landscape = Landscape::new(Log2Space::new(55., 1760., 24));
    let geometry = Geometry::new(
        &landscape.space,
        &HarmonicityKernel::new(&landscape.space, HarmonicityParams::default()),
    );
    let mut density = vec![0.; landscape.space.n_bins()];
    density[72] = 1.;
    let mut snapshot = Snapshot::new(geometry);
    snapshot.observe(&density, 1e-12, 512);
    let snapshot = Arc::new(snapshot);
    let perceptual = test_adaptation(landscape.space.n_bins());
    let features =
        FeaturesNow::from_occupancy_scan(&landscape.space, &landscape.subjective_intensity);
    let current = 330.0f32.log2();
    let mut core = PitchHillClimbPitchCore::new(200., current, 0.);
    core.move_cost_coeff = 0.;
    core.crowding_strength = 0.;
    core.leave_self_out = true;
    core.loo_input.support = Some(snapshot.clone());
    core.loo_input.support_single_density = 1.;
    for template in [None, Some(Arc::from([(0, 0.4), (24, 0.2)]))] {
        core.loo_input.support_template = template.clone();
        for axis in 0..3 {
            landscape.harmonic.support_weights = Default::default();
            landscape.harmonic.support_weights.set(axis, 1.);
            let mut actual_rng = SmallRng::seed_from_u64(735);
            let mut expected_rng = actual_rng.clone();
            let expected = core.propose_with_scorer(
                current,
                current,
                &landscape,
                &[],
                &mut expected_rng,
                |pitch| {
                    template.as_ref().map_or_else(
                        || {
                            snapshot.single_partial_value(
                                pitch,
                                1.,
                                landscape.harmonic.support_weights,
                            )
                        },
                        |t| snapshot.template_value(pitch, t, landscape.harmonic.support_weights),
                    ) as f32
                },
            );
            let actual = core.propose_target(
                current,
                current,
                330.,
                1.,
                &landscape,
                &perceptual,
                &features,
                &[],
                &mut actual_rng,
            );
            assert_eq!(
                actual.target_pitch_log2.to_bits(),
                expected.target_pitch_log2.to_bits()
            );
            assert_eq!(actual_rng.random::<u64>(), expected_rng.random::<u64>());
        }
    }
    core.leave_self_out = false;
    let mut a = SmallRng::seed_from_u64(735);
    let mut b = a.clone();
    let ignored = core.propose_target(
        current,
        current,
        330.,
        1.,
        &landscape,
        &perceptual,
        &features,
        &[],
        &mut a,
    );
    landscape.harmonic.support_weights = Default::default();
    let neutral = core.propose_target(
        current,
        current,
        330.,
        1.,
        &landscape,
        &perceptual,
        &features,
        &[],
        &mut b,
    );
    assert_eq!(
        ignored.target_pitch_log2.to_bits(),
        neutral.target_pitch_log2.to_bits()
    );
    assert_eq!(a.random::<u64>(), b.random::<u64>());
}

#[test]
fn melodic_registered_prefix_contrast_and_f32_margin() {
    let mut core = PitchHillClimbPitchCore::new(200.0, 5.0, 0.0);
    core.set_temperature(0.0);
    for (xs, expected) in [
        ([60., 60., 62., 60.], 5.0),
        ([64., 60., 62., 60.], 62.0 / 12.0),
    ] {
        let snapshot = melodic_snapshot(&xs);
        let term = snapshot.contribution(62.0 / 12.0, 5.0).unwrap();
        let round_error = (term as f32 as f64 - term).abs();
        assert!((term - 0.1).abs() > round_error);
        let mut scored = vec![(5.0, 0.0), (62.0 / 12.0, term as f32)];
        let got = core.select_scored(
            5.0,
            0.0,
            &mut scored,
            &mut SmallRng::seed_from_u64(21),
            None,
        );
        assert_eq!(got.target_pitch_log2, expected);
    }
    let landscape = Landscape::new(Log2Space::new(20., 2000., 96));
    let snapshot = melodic_snapshot(&[60., 60., 62., 60.]);
    let choices = [0.0, 1000.0].map(|field_slope| {
        core.propose_with_melodic_scorer(
            5.0,
            5.0,
            &landscape,
            &[],
            &mut SmallRng::seed_from_u64(21),
            Some(snapshot),
            |pitch| field_slope * (pitch - 5.0),
        )
        .unwrap()
        .target_pitch_log2
    });
    assert!(choices[0] < snapshot.bounds.1);
    assert_eq!(choices[1], snapshot.bounds.1);
}

#[test]
fn melodic_legal_mapping_deduplicates_before_all_scoring() {
    let landscape = Landscape::new(Log2Space::new(20., 2000., 96));
    let mut core = PitchHillClimbPitchCore::new(200., 5., 0.);
    core.set_temperature(0.0);
    let snapshot = melodic_snapshot(&[64., 60., 62., 60.]);
    for sequence in [
        None,
        Some(crate::life::own_sequence::SequenceSnapshot {
            probabilities: [1.0 / 241.0; 241],
            alphabet: 241,
            offset: 120,
            previous_hz: 32.,
            weight: 1.,
            bounds_log2: snapshot.bounds,
            at: 10,
            version: 1,
            epoch: 0,
        }),
    ] {
        core.sequence = sequence;
        core.sequence_enabled = true;
        core.sequence_trace = true;
        let selected = core
            .propose_with_melodic_scorer(
                5.,
                5.,
                &landscape,
                &[],
                &mut SmallRng::seed_from_u64(3),
                Some(snapshot),
                |p| if p > 5.5 { 100. } else { 0. },
            )
            .unwrap();
        assert!(
            selected.target_pitch_log2 >= snapshot.bounds.0
                && selected.target_pitch_log2 <= snapshot.bounds.1
        );
        let scratch = core.scratch.borrow();
        let pitches: Vec<_> = scratch.scored.iter().map(|&(p, _)| p).collect();
        for (i, &p) in pitches.iter().enumerate() {
            assert!(!pitches[..i].contains(&p));
            assert!((snapshot.bounds.0..=snapshot.bounds.1).contains(&p));
        }
        assert!(scratch.scored.iter().all(|&(_, v)| v < 100.));
    }
}

#[test]
fn integrated_melodic_sequence_records_and_unknown_context_are_independent() {
    let landscape = test_landscape(&[(330., 1.), (440., 0.8)]);
    let perceptual = test_adaptation(landscape.space.n_bins());
    let features =
        FeaturesNow::from_occupancy_scan(&landscape.space, &landscape.subjective_intensity);
    let current = 330.0f32.log2();
    let mut core = PitchHillClimbPitchCore::new(200., current, 1000.);
    core.set_move_cost_coeff(1000.);
    core.set_temperature(0.0);
    let mut me = melodic_snapshot(&[96., 96., 98., 96.]);
    me.bounds = (220.0f32.log2(), 880.0f32.log2());
    core.melodic = Some(me);
    core.melodic_trace = true;
    core.sequence_enabled = true;
    core.sequence = Some(crate::life::own_sequence::SequenceSnapshot {
        probabilities: std::array::from_fn(|i| (i + 1) as f64 / 29161.0),
        alphabet: 241,
        offset: 120,
        previous_hz: 330.,
        weight: 0.05,
        bounds_log2: me.bounds,
        at: 10,
        version: 1,
        epoch: 0,
    });
    let run = |c: &mut PitchHillClimbPitchCore| {
        let mut rng = SmallRng::seed_from_u64(21);
        let result = c.propose_target(
            current,
            current,
            330.,
            1.,
            &landscape,
            &perceptual,
            &features,
            &[],
            &mut rng,
        );
        (result.target_pitch_log2, rng.random::<u64>())
    };
    let actual = run(&mut core);
    let mut temporal_zero = core.clone();
    temporal_zero.loo_input.relations =
        Some(crate::temporal_cognition::harmonic_relations::consumer::test_use(0., true).0);
    assert_eq!(run(&mut temporal_zero), actual);
    assert_eq!(
        core.sequence_decision
            .borrow()
            .as_ref()
            .unwrap()
            .selected_log2,
        actual.0
    );
    let mut no_me = core.clone();
    no_me.melodic = None;
    let without = run(&mut no_me);
    assert_eq!(
        core.melodic_decision
            .borrow()
            .as_ref()
            .unwrap()
            .without_melodic_log2,
        without.0
    );
    let mut quiet = core.clone();
    quiet.melodic_trace = false;
    assert_eq!(run(&mut quiet), actual);
    let mut unknown_tc = core.clone();
    let mut term = crate::core::tonal_context::Use::new();
    term.pin(None, 1., me.bounds);
    unknown_tc.tonal = Some(term);
    assert_eq!(run(&mut unknown_tc), actual);
    assert_eq!(
        unknown_tc.tonal.as_ref().unwrap().selection_known.get(),
        Some(false)
    );
    let mut failed_me = core.clone();
    failed_me.melodic.as_mut().unwrap().weight = f64::MAX;
    assert_eq!(run(&mut failed_me), without);
    assert!(
        core.melodic_decision
            .borrow()
            .as_ref()
            .unwrap()
            .old_terms_replaced
    );
}

#[test]
fn melodic_native_replacement_and_independent_failures() {
    for with_ps in [false, true] {
        let landscape = test_landscape(&[(330., 1.), (440., 0.8)]);
        let perceptual = test_adaptation(landscape.space.n_bins());
        let features =
            FeaturesNow::from_occupancy_scan(&landscape.space, &landscape.subjective_intensity);
        let current = 330.0f32.log2();
        let mut me = melodic_snapshot(&[96., 96., 98., 96.]);
        me.bounds = (220.0f32.log2(), 880.0f32.log2());
        let mut core = PitchHillClimbPitchCore::new(200., current, 1000.);
        core.set_move_cost_coeff(1000.);
        core.set_temperature(0.0);
        core.melodic = Some(me);
        if with_ps {
            core.pitch_set = Some(crate::core::tonal_context::pitch_set::Use {
                snapshot: Some(crate::core::tonal_context::pitch_set::heard_for_test(307.)),
                weight: 1.,
                bounds: me.bounds,
                trace: true,
                decision: Default::default(),
                added: Default::default(),
                tc_failed_added: Default::default(),
            });
        }
        core.melodic_trace = true;
        let run = |c: &mut PitchHillClimbPitchCore| {
            let mut rng = SmallRng::seed_from_u64(21);
            let result = c.propose_target(
                current,
                current,
                330.,
                1.,
                &landscape,
                &perceptual,
                &features,
                &[],
                &mut rng,
            );
            (result.target_pitch_log2, rng.random::<u64>())
        };
        let chosen = run(&mut core);
        let mut temporal_zero = core.clone();
        temporal_zero.loo_input.relations =
            Some(crate::temporal_cognition::harmonic_relations::consumer::test_use(0., true).0);
        assert_eq!(run(&mut temporal_zero), chosen);
        let mut zero_old = core.clone();
        zero_old.set_tessitura_gravity(0.);
        zero_old.set_move_cost_coeff(0.);
        assert_eq!(chosen, run(&mut zero_old));
        let mut failed_sequence = core.clone();
        failed_sequence.sequence_enabled = true;
        failed_sequence.sequence = Some(crate::life::own_sequence::SequenceSnapshot {
            probabilities: [0.; 241],
            alphabet: 241,
            offset: 120,
            previous_hz: 330.,
            weight: 1.,
            bounds_log2: me.bounds,
            at: 10,
            version: 1,
            epoch: 0,
        });
        assert_eq!(chosen, run(&mut failed_sequence));
        assert!(
            failed_sequence
                .melodic_decision
                .borrow()
                .as_ref()
                .unwrap()
                .old_terms_replaced
        );
        assert_eq!(
            failed_sequence
                .sequence_decision
                .borrow()
                .as_ref()
                .unwrap()
                .reason,
            Some("nonfinite_sequence_score")
        );
        core.sequence_enabled = true;
        core.sequence = Some(crate::life::own_sequence::SequenceSnapshot {
            probabilities: std::array::from_fn(|i| (i + 1) as f64 / 29161.0),
            alphabet: 241,
            offset: 120,
            previous_hz: 330.,
            weight: 0.1,
            bounds_log2: me.bounds,
            at: 10,
            version: 1,
            epoch: 0,
        });
        let mut legacy = core.clone();
        legacy.melodic = None;
        let expected = run(&mut legacy);
        core.melodic.as_mut().unwrap().weight = f64::MAX;
        assert_eq!(expected, run(&mut core));
        assert!(core.sequence_decision.borrow().as_ref().unwrap().active);
        assert_eq!(
            core.melodic_decision.borrow().as_ref().unwrap().reason,
            Some("nonfinite_melodic_score")
        );
        core.melodic.as_mut().unwrap().bounds = (20., 21.);
        assert_eq!(expected, run(&mut core));
    }
}

fn test_landscape(peaks_hz: &[(f32, f32)]) -> Landscape {
    let mut landscape = Landscape::new(Log2Space::new(110.0, 880.0, 96));
    let width_cents = 40.0f32;
    for (idx, &bin_log2) in landscape.space.centers_log2.iter().enumerate() {
        let mut score = 0.0f32;
        for &(freq_hz, gain) in peaks_hz {
            let center = freq_hz.max(1.0).log2();
            let d_cents = (bin_log2 - center).abs() * 1200.0;
            let g = (-(d_cents * d_cents) / (2.0 * width_cents * width_cents)).exp();
            score += gain * g;
        }
        landscape.consonance_field_score[idx] = score;
        landscape.consonance_field_level[idx] = score.clamp(0.0, 1.0);
        landscape.consonance_field_score_eff[idx] = score;
        landscape.consonance_field_level_eff[idx] = score.clamp(0.0, 1.0);
    }
    landscape
}

fn test_adaptation(n_bins: usize) -> AdaptationContext {
    AdaptationContext::from_config(&AdaptationConfig::default(), n_bins)
}

#[cfg(feature = "profile-alloc")]
#[test]
fn hill_climb_proposals_reuse_candidate_storage() {
    use rand::{SeedableRng, rngs::SmallRng};
    let landscape = Landscape::new(Log2Space::new(55.0, 1760.0, 96));
    let mut core = PitchHillClimbPitchCore::new(60.0, 330.0f32.log2(), 0.0);
    core.set_global_peaks(10, 20.0);
    core.set_ratio_candidates(true, 20);
    core.set_temperature(0.5);
    let mut rng = SmallRng::seed_from_u64(21);
    let pitch = 330.0f32.log2();
    core.propose_with_scorer(pitch, pitch, &landscape, &[pitch], &mut rng, |_| 0.0);
    crate::runtime_profile::begin_allocations();
    for _ in 0..100 {
        core.propose_with_scorer(pitch, pitch, &landscape, &[pitch], &mut rng, |_| 0.0);
    }
    let counts = crate::runtime_profile::finish_allocations().unwrap();
    assert_eq!((counts.count, counts.bytes), (0, 0));
}

#[test]
fn exact_loo_cache_matches_uncached_kernels_and_invalidates() {
    let mut landscape = Landscape::new(Log2Space::new(80., 8000., 24));
    let mut cache = None;
    for revision in 0..5 {
        if revision == 2 {
            landscape.harmonicity_params.gamma_root = 0.7;
        }
        if revision == 3 {
            landscape.roughness_kernel_params.sethares_b = 4.2;
        }
        if revision == 4 {
            landscape.resize_to_space(Log2Space::new(110., 6000., 36));
        }
        let n = landscape.space.n_bins();
        for (i, d) in landscape.subjective_intensity.iter_mut().enumerate() {
            *d = ((i * 17 + 3) % 29) as f32 * 0.03;
        }
        let index = n / 3;
        landscape.subjective_intensity[index] = 1.;
        let pitch = landscape.space.centers_log2[index];
        assert!(
            exact_loo_consonance_score_scan(
                &landscape,
                pitch,
                false,
                ReferenceLooMode::ExactScan,
                &mut cache
            )
            .is_none()
        );
        if revision == 0 {
            assert!(cache.is_none());
        }
        let previous_lut = cache.as_ref().map(|c| Arc::clone(&c.roughness.lut));
        let actual = exact_loo_consonance_score_scan(
            &landscape,
            pitch,
            true,
            ReferenceLooMode::ExactScan,
            &mut cache,
        )
        .unwrap();
        if revision == 1 {
            assert!(Arc::ptr_eq(
                previous_lut.as_ref().unwrap(),
                &cache.as_ref().unwrap().roughness.lut
            ));
        } else if let Some(previous) = previous_lut {
            assert!(!Arc::ptr_eq(
                &previous,
                &cache.as_ref().unwrap().roughness.lut
            ));
        }
        let mut density = landscape.subjective_intensity.clone();
        density[index] = 0.;
        let h = HarmonicityKernel::new(&landscape.space, landscape.harmonicity_params)
            .potential_h_from_log2_spectrum(&density, &landscape.space)
            .0;
        let (_, du) = crate::core::roughness_kernel::erb_grid(&landscape.space);
        let eps = landscape.roughness_ref_eps.max(1e-12);
        let (normalized, _) = crate::core::psycho_state::normalize_density(&density, &du, eps);
        let r = RoughnessKernel::new(
            landscape.roughness_kernel_params,
            EXACT_LOO_ROUGHNESS_ERB_STEP,
        )
        .potential_r_from_log2_spectrum_density(&normalized, &landscape.space)
        .0;
        for i in 0..n {
            let r01 = roughness_ratio_to_state01(
                r[i] / landscape.roughness_ref_peak.max(eps),
                landscape.roughness_k.max(1e-6),
            );
            let expected = landscape.consonance_kernel.score(h[i].clamp(0., 1.), r01);
            assert_eq!(
                actual[i].to_bits(),
                expected.to_bits(),
                "revision {revision}, bin {i}"
            );
        }
    }
}

#[test]
fn shared_habituation_alone_releases_a_greedy_voice_from_its_peak() {
    use crate::core::consonance_kernel::ConsonanceRepresentationParams;
    use rand::{SeedableRng, rngs::SmallRng};
    for leave_self_out in [false, true] {
        let mut landscape = Landscape::new(Log2Space::new(110.0, 880.0, 12));
        let n = landscape.space.n_bins();
        let here_idx = n / 3;
        let next_idx = here_idx + 2;
        let here = landscape.space.centers_log2[here_idx];
        let next = landscape.space.centers_log2[next_idx];
        landscape.consonance_field_score.fill(-1.0);
        landscape.consonance_field_score[here_idx] = 1.0;
        landscape.consonance_field_score[next_idx] = 0.6;
        let raw = landscape.consonance_field_score.clone();
        let mut core = PitchHillClimbPitchCore::new((next - here) * 1200.0, here, 0.0);
        core.set_temperature(0.0);
        core.set_move_cost_coeff(0.0);
        core.set_leave_self_out(leave_self_out);
        core.loo_input.exact_score = Some(raw.clone().into());
        let mut adaptation = test_adaptation(n);
        let features = FeaturesNow {
            density: vec![0.0; n],
            mass: 0.0,
        };
        let mut rng = SmallRng::seed_from_u64(21);
        for (erosion, expected) in [(0.0, here), (0.3, here), (0.8, next)] {
            let mut state = vec![0.0; n];
            state[here_idx] = erosion;
            landscape.apply_habituation(&state, 0.0, &ConsonanceRepresentationParams::default());
            for _ in 0..100 {
                let proposal = core.propose_target(
                    here,
                    here,
                    here.exp2(),
                    1.0,
                    &landscape,
                    &adaptation,
                    &features,
                    &[],
                    &mut rng,
                );
                assert_eq!(
                    proposal.target_pitch_log2, expected,
                    "LOO={leave_self_out}, h={erosion}"
                );
                adaptation.update(&features, 0.2);
                assert_eq!(adaptation.score_adjustment(here_idx), 0.0);
                assert_eq!(landscape.consonance_field_score, raw);
            }
        }
    }
}

#[test]
fn exact_loo_scan_is_eroded_by_habituation() {
    let mut landscape = Landscape::new(Log2Space::new(55.0, 1760.0, 12));
    let n = landscape.space.n_bins();
    landscape.perc_habituation_state_scan = vec![1.0; n];
    landscape.consonance_theta = 0.0;
    let scan = vec![0.8f32; n];
    let mid_log2 = landscape.space.centers_log2[n / 2];
    let s = sample_consonance_score_with_loo(
        mid_log2,
        mid_log2,
        &landscape,
        true,
        ReferenceLooMode::ExactScan,
        0,
        Some(&scan),
        None,
    );
    assert!(s.abs() < 1e-3, "expected eroded ~0, got {s}");
}

#[test]
fn exact_loo_scan_zero_state_is_identity() {
    let mut landscape = Landscape::new(Log2Space::new(55.0, 1760.0, 12));
    let n = landscape.space.n_bins();
    landscape.perc_habituation_state_scan = vec![0.0; n];
    landscape.consonance_theta = 0.0;
    let scan = vec![0.8f32; n];
    let mid_log2 = landscape.space.centers_log2[n / 2];
    let s = sample_consonance_score_with_loo(
        mid_log2,
        mid_log2,
        &landscape,
        true,
        ReferenceLooMode::ExactScan,
        0,
        Some(&scan),
        None,
    );
    assert!(
        (s - 0.8).abs() < 1e-6,
        "zero-state must be identity, got {s}"
    );
}

#[test]
fn approx_loo_pitch_score_reduces_positive_self_peak_score() {
    let landscape = test_landscape(&[(220.0, 1.0)]);
    let raw = landscape.evaluate_pitch_score(220.0);
    let loo = legacy_sample_consonance_score_with_loo(
        220.0f32.log2(),
        220.0f32.log2(),
        &landscape,
        true,
        ReferenceLooMode::ApproxHarmonics,
        1,
        None,
        None,
    );

    assert!(raw > 0.0);
    assert!(loo < raw);
}

#[test]
fn approx_loo_uses_body_ratios_not_generic_series() {
    // Body partials sit at 1.0 and 2.5; the generic 2-harmonic series sits at
    // 1.0 and 2.0. LOO must subtract at the body's positions, not the series'.
    let f0 = 220.0f32;
    let landscape = test_landscape(&[(f0, 1.0), (2.0 * f0, 1.0), (2.5 * f0, 1.0)]);
    let current = f0.log2();
    let ratios = [1.0f32, 2.5];

    // On the body's own 2.5 partial: body path subtracts it, generic path does not.
    let cand_25 = (2.5 * f0).log2();
    let body_25 = legacy_sample_consonance_score_with_loo(
        cand_25,
        current,
        &landscape,
        true,
        ReferenceLooMode::ApproxHarmonics,
        2,
        None,
        Some(&ratios),
    );
    let generic_25 = legacy_sample_consonance_score_with_loo(
        cand_25,
        current,
        &landscape,
        true,
        ReferenceLooMode::ApproxHarmonics,
        2,
        None,
        None,
    );
    assert!(
        body_25 < generic_25 - 1e-4,
        "body must subtract its own 2.5 partial (body={body_25}, generic={generic_25})"
    );

    // On the 2f position: generic (k=2) subtracts it, the body ([1.0, 2.5]) must not.
    let cand_20 = (2.0 * f0).log2();
    let body_20 = legacy_sample_consonance_score_with_loo(
        cand_20,
        current,
        &landscape,
        true,
        ReferenceLooMode::ApproxHarmonics,
        2,
        None,
        Some(&ratios),
    );
    let generic_20 = legacy_sample_consonance_score_with_loo(
        cand_20,
        current,
        &landscape,
        true,
        ReferenceLooMode::ApproxHarmonics,
        2,
        None,
        None,
    );
    assert!(
        generic_20 < body_20 - 1e-4,
        "generic series subtracts 2f, the body must not (body={body_20}, generic={generic_20})"
    );
}

#[test]
fn approx_loo_harmonic_ratios_match_generic_series() {
    // A plain harmonic ratio set must reproduce the generic integer series
    // exactly (same positions, same per-index 1/n weights).
    let f0 = 220.0f32;
    let landscape = test_landscape(&[(f0, 1.0), (2.0 * f0, 0.8), (3.0 * f0, 0.6)]);
    let current = f0.log2();
    let harmonic_ratios = [1.0f32, 2.0, 3.0];
    for cand_hz in [f0, 1.5 * f0, 2.0 * f0, 2.7 * f0, 3.0 * f0] {
        let cand = cand_hz.log2();
        let body = legacy_sample_consonance_score_with_loo(
            cand,
            current,
            &landscape,
            true,
            ReferenceLooMode::ApproxHarmonics,
            3,
            None,
            Some(&harmonic_ratios),
        );
        let generic = legacy_sample_consonance_score_with_loo(
            cand,
            current,
            &landscape,
            true,
            ReferenceLooMode::ApproxHarmonics,
            3,
            None,
            None,
        );
        assert!(
            (body - generic).abs() <= 1e-5,
            "harmonic ratios must match generic k=3 series at {cand_hz} Hz (body={body}, generic={generic})"
        );
    }
}

#[test]
fn adjusted_score_landscape_weight_zero_disables_landscape_term() {
    let mut landscape = Landscape::new(Log2Space::new(110.0, 880.0, 96));
    let idx = landscape.space.n_bins() / 2;
    let pitch_log2 = landscape.space.centers_log2[idx];
    landscape.consonance_field_score[idx] = 1.75;
    landscape.consonance_field_score_eff[idx] = 1.75;
    let perceptual = test_adaptation(landscape.space.n_bins());

    let weighted = adjusted_pitch_score(
        pitch_log2,
        pitch_log2,
        0.0,
        pitch_log2,
        0.0,
        1.0,
        0.0,
        1,
        &landscape,
        &perceptual,
        false,
        0.0,
        0.0,
        &[],
        &[],
    );
    let without_landscape = adjusted_pitch_score(
        pitch_log2,
        pitch_log2,
        0.0,
        pitch_log2,
        0.0,
        0.0,
        0.0,
        1,
        &landscape,
        &perceptual,
        false,
        0.0,
        0.0,
        &[],
        &[],
    );

    assert!((weighted - 1.75).abs() <= 1e-6);
    assert!(without_landscape.abs() <= 1e-6);
}

#[test]
fn adjusted_score_crowding_penalizes_close_neighbor() {
    let mut landscape = Landscape::new(Log2Space::new(110.0, 880.0, 96));
    let idx = landscape.space.n_bins() / 2;
    let pitch_log2 = landscape.space.centers_log2[idx];
    landscape.consonance_field_score[idx] = 1.0;
    landscape.consonance_field_score_eff[idx] = 1.0;
    let perceptual = test_adaptation(landscape.space.n_bins());

    let without_crowding = adjusted_pitch_score(
        pitch_log2,
        pitch_log2,
        0.0,
        pitch_log2,
        0.0,
        1.0,
        0.0,
        1,
        &landscape,
        &perceptual,
        false,
        1.0,
        20.0,
        &[],
        &[],
    );
    let with_close_neighbor = adjusted_pitch_score(
        pitch_log2,
        pitch_log2,
        0.0,
        pitch_log2,
        0.0,
        1.0,
        0.0,
        1,
        &landscape,
        &perceptual,
        false,
        1.0,
        20.0,
        &[pitch_log2],
        &[1.0],
    );

    assert!(
        with_close_neighbor < without_crowding,
        "crowding should lower score when neighbor is near"
    );
}

#[test]
fn crowding_pairwise_split_biases_agents_in_opposite_directions() {
    let mut landscape = Landscape::new(Log2Space::new(110.0, 880.0, 128));
    landscape.consonance_field_score.fill(1.0);
    landscape.consonance_field_score_eff.fill(1.0);
    let perceptual = test_adaptation(landscape.space.n_bins());
    let center = landscape.space.centers_log2[landscape.space.n_bins() / 2];
    let left = center - cents_to_log2(6.0);
    let right = center + cents_to_log2(6.0);
    let neighbor = [center];

    let left_pos = adjusted_pitch_score(
        left,
        center,
        0.0,
        center,
        0.0,
        1.0,
        0.0,
        1,
        &landscape,
        &perceptual,
        false,
        4.0,
        60.0,
        &neighbor,
        &[1.0],
    );
    let right_pos = adjusted_pitch_score(
        right,
        center,
        0.0,
        center,
        0.0,
        1.0,
        0.0,
        1,
        &landscape,
        &perceptual,
        false,
        4.0,
        60.0,
        &neighbor,
        &[1.0],
    );
    let left_neg = adjusted_pitch_score(
        left,
        center,
        0.0,
        center,
        0.0,
        1.0,
        0.0,
        1,
        &landscape,
        &perceptual,
        false,
        4.0,
        60.0,
        &neighbor,
        &[-1.0],
    );
    let right_neg = adjusted_pitch_score(
        right,
        center,
        0.0,
        center,
        0.0,
        1.0,
        0.0,
        1,
        &landscape,
        &perceptual,
        false,
        4.0,
        60.0,
        &neighbor,
        &[-1.0],
    );

    let dir_pos = right_pos - left_pos;
    let dir_neg = right_neg - left_neg;
    assert!(
        dir_pos * dir_neg < 0.0,
        "pairwise split sign should induce opposite directional preference (dir_pos={dir_pos}, dir_neg={dir_neg})"
    );
}

#[test]
fn crowding_samples_sigma_cents_gaussian_not_roughness() {
    let mut landscape = Landscape::new(Log2Space::new(110.0, 880.0, 96));
    let idx = landscape.space.n_bins() / 2;
    let pitch_log2 = landscape.space.centers_log2[idx];
    let neighbor_log2 = pitch_log2 + cents_to_log2(40.0);
    landscape.consonance_field_score[idx] = 1.0;
    landscape.consonance_field_score_eff[idx] = 1.0;
    let perceptual = test_adaptation(landscape.space.n_bins());

    landscape.roughness_suppress_sigma_erb = 0.04;
    landscape.roughness_kernel_params.suppress_sigma_erb = 0.04;
    let narrow = adjusted_pitch_score(
        pitch_log2,
        pitch_log2,
        0.0,
        pitch_log2,
        0.0,
        1.0,
        0.0,
        1,
        &landscape,
        &perceptual,
        false,
        1.0,
        20.0,
        &[neighbor_log2],
        &[1.0],
    );

    landscape.roughness_suppress_sigma_erb = 0.10;
    landscape.roughness_kernel_params.suppress_sigma_erb = 0.10;
    let wide = adjusted_pitch_score(
        pitch_log2,
        pitch_log2,
        0.0,
        pitch_log2,
        0.0,
        1.0,
        0.0,
        1,
        &landscape,
        &perceptual,
        false,
        1.0,
        20.0,
        &[neighbor_log2],
        &[1.0],
    );
    assert!(
        (wide - narrow).abs() <= 1e-6,
        "crowding samples a fixed sigma_cents Gaussian and ignores roughness sigma"
    );
}

#[test]
fn loo_off_preserves_legacy_score_formula() {
    let mut landscape = Landscape::new(Log2Space::new(110.0, 880.0, 96));
    let idx = landscape.space.n_bins() / 2;
    let pitch_log2 = landscape.space.centers_log2[idx];
    landscape.consonance_field_score[idx] = 1.23;
    landscape.consonance_field_score_eff[idx] = 1.23;
    let perceptual = test_adaptation(landscape.space.n_bins());

    let score = adjusted_pitch_score(
        pitch_log2,
        pitch_log2,
        0.0,
        pitch_log2,
        0.0,
        1.0,
        0.0,
        1,
        &landscape,
        &perceptual,
        false,
        0.0,
        0.0,
        &[],
        &[],
    );
    assert!((score - 1.23).abs() <= 1e-6);
}

#[test]
fn approximate_loo_weakens_self_locking() {
    let mut landscape = Landscape::new(Log2Space::new(110.0, 880.0, 96));
    let idx = landscape.space.n_bins() / 2;
    let current = landscape.space.centers_log2[idx];
    let nearby = landscape.space.centers_log2[idx + 1];
    landscape.consonance_field_score[idx] = 1.0;
    landscape.consonance_field_score[idx + 1] = 0.7;
    landscape.consonance_field_score_eff[idx] = 1.0;
    landscape.consonance_field_score_eff[idx + 1] = 0.7;
    let perceptual = test_adaptation(landscape.space.n_bins());

    let current_no_loo = adjusted_pitch_score(
        current,
        current,
        0.0,
        current,
        0.0,
        1.0,
        0.0,
        1,
        &landscape,
        &perceptual,
        false,
        0.0,
        0.0,
        &[],
        &[],
    );
    let nearby_no_loo = adjusted_pitch_score(
        nearby,
        current,
        0.0,
        current,
        0.0,
        1.0,
        0.0,
        1,
        &landscape,
        &perceptual,
        false,
        0.0,
        0.0,
        &[],
        &[],
    );
    assert!(current_no_loo > nearby_no_loo);

    let current_with_loo = legacy_sample_consonance_score_with_loo(
        current,
        current,
        &landscape,
        true,
        ReferenceLooMode::ApproxHarmonics,
        DEFAULT_LOO_HARMONICS,
        None,
        None,
    );
    let nearby_with_loo = legacy_sample_consonance_score_with_loo(
        nearby,
        current,
        &landscape,
        true,
        ReferenceLooMode::ApproxHarmonics,
        DEFAULT_LOO_HARMONICS,
        None,
        None,
    );
    assert!(nearby_with_loo > current_with_loo);
}

#[test]
fn crowding_strength_zero_keeps_behavior_with_neighbors() {
    let landscape = test_landscape(&[(330.0, 1.0)]);
    let perceptual = test_adaptation(landscape.space.n_bins());
    let features =
        FeaturesNow::from_occupancy_scan(&landscape.space, &landscape.subjective_intensity);
    let mut core = PitchHillClimbPitchCore::new(120.0, 330.0f32.log2(), 0.0);
    core.set_crowding(0.0, 20.0);

    let mut rng_a = SmallRng::seed_from_u64(77);
    let mut rng_b = SmallRng::seed_from_u64(77);
    let base = core.propose_target(
        330.0f32.log2(),
        330.0f32.log2(),
        330.0,
        1.0,
        &landscape,
        &perceptual,
        &features,
        &[],
        &mut rng_a,
    );
    let with_neighbors = core.propose_target(
        330.0f32.log2(),
        330.0f32.log2(),
        330.0,
        1.0,
        &landscape,
        &perceptual,
        &features,
        &[330.0f32.log2(), 440.0f32.log2()],
        &mut rng_b,
    );

    assert!((base.target_pitch_log2 - with_neighbors.target_pitch_log2).abs() <= 1e-6);
    assert!((base.salience - with_neighbors.salience).abs() <= 1e-6);
}

#[test]
fn peak_sampler_crowding_zero_ignores_neighbors() {
    let landscape = test_landscape(&[(330.0, 1.0)]);
    let perceptual = test_adaptation(landscape.space.n_bins());
    let features =
        FeaturesNow::from_occupancy_scan(&landscape.space, &landscape.subjective_intensity);
    let mut core = PitchPeakSamplerCore::new(120.0, 330.0f32.log2(), 0.0, 700.0, 16, 0.03, 10.0, 2);
    core.set_crowding(0.0, 20.0);

    let mut rng_a = SmallRng::seed_from_u64(71);
    let mut rng_b = SmallRng::seed_from_u64(71);
    let base = core.propose_target(
        330.0f32.log2(),
        330.0f32.log2(),
        330.0,
        1.0,
        &landscape,
        &perceptual,
        &features,
        &[],
        &mut rng_a,
    );
    let with_neighbors = core.propose_target(
        330.0f32.log2(),
        330.0f32.log2(),
        330.0,
        1.0,
        &landscape,
        &perceptual,
        &features,
        &[330.0f32.log2(), 440.0f32.log2()],
        &mut rng_b,
    );

    assert!((base.target_pitch_log2 - with_neighbors.target_pitch_log2).abs() <= 1e-6);
    assert!((base.salience - with_neighbors.salience).abs() <= 1e-6);
}

#[test]
fn peak_sampler_crowding_penalizes_close_neighbor_selection() {
    let landscape = test_landscape(&[(330.0, 1.0)]);
    let perceptual = test_adaptation(landscape.space.n_bins());
    let features =
        FeaturesNow::from_occupancy_scan(&landscape.space, &landscape.subjective_intensity);
    let mut no_crowding =
        PitchPeakSamplerCore::new(24.0, 330.0f32.log2(), 0.0, 700.0, 16, 0.001, 0.0, 0);
    let mut with_crowding = no_crowding.clone();
    no_crowding.set_crowding(0.0, 8.0);
    with_crowding.set_crowding(3.0, 8.0);
    let neighbor = 330.0f32.log2();

    let mut rng_a = SmallRng::seed_from_u64(99);
    let mut rng_b = SmallRng::seed_from_u64(99);
    let off = no_crowding.propose_target(
        neighbor,
        neighbor,
        330.0,
        1.0,
        &landscape,
        &perceptual,
        &features,
        &[neighbor],
        &mut rng_a,
    );
    let on = with_crowding.propose_target(
        neighbor,
        neighbor,
        330.0,
        1.0,
        &landscape,
        &perceptual,
        &features,
        &[neighbor],
        &mut rng_b,
    );

    let dist_off = (off.target_pitch_log2 - neighbor).abs();
    let dist_on = (on.target_pitch_log2 - neighbor).abs();
    assert!(
        dist_on > dist_off + 1e-6,
        "crowding should bias peak-sampler away from close neighbor"
    );
}

#[test]
fn hillclimb_temperature_zero_makes_no_downhill_move() {
    // On a plateau (no improvement above threshold) a zero-temperature
    // hill-climb must stay put: greedy, never downhill.
    let mut landscape = Landscape::new(Log2Space::new(220.0, 440.0, 256));
    let idx_center = landscape.space.n_bins() / 2;
    let current = landscape.space.centers_log2[idx_center];
    let width_cents = 40.0f32;
    for (idx, &bin_log2) in landscape.space.centers_log2.iter().enumerate() {
        let d_cents = (bin_log2 - current).abs() * 1200.0;
        let score = (-(d_cents * d_cents) / (2.0 * width_cents * width_cents)).exp();
        landscape.consonance_field_score[idx] = score;
        landscape.consonance_field_level[idx] = score.clamp(0.0, 1.0);
        landscape.consonance_field_score_eff[idx] = score;
        landscape.consonance_field_level_eff[idx] = score.clamp(0.0, 1.0);
    }
    let perceptual = test_adaptation(landscape.space.n_bins());
    let features =
        FeaturesNow::from_occupancy_scan(&landscape.space, &landscape.subjective_intensity);

    for seed in 0..256u64 {
        let mut core = PitchHillClimbPitchCore::new(30.0, current, 0.0);
        let mut rng = SmallRng::seed_from_u64(5_000 + seed);
        let p = core.propose_target(
            current,
            current,
            2.0f32.powf(current),
            1.0,
            &landscape,
            &perceptual,
            &features,
            &[],
            &mut rng,
        );
        assert!(
            (p.target_pitch_log2 - current).abs() <= 1e-6,
            "temperature=0 should keep greedy hill-climb behavior (no downhill move)"
        );
    }
}

#[test]
fn hillclimb_temperature_accepts_downhill_moves_probabilistically() {
    let mut landscape = Landscape::new(Log2Space::new(220.0, 440.0, 256));
    let idx_center = landscape.space.n_bins() / 2;
    let current = landscape.space.centers_log2[idx_center];
    let width_cents = 40.0f32;
    for (idx, &bin_log2) in landscape.space.centers_log2.iter().enumerate() {
        let d_cents = (bin_log2 - current).abs() * 1200.0;
        let score = (-(d_cents * d_cents) / (2.0 * width_cents * width_cents)).exp();
        landscape.consonance_field_score[idx] = score;
        landscape.consonance_field_level[idx] = score.clamp(0.0, 1.0);
        landscape.consonance_field_score_eff[idx] = score;
        landscape.consonance_field_level_eff[idx] = score.clamp(0.0, 1.0);
    }
    let perceptual = test_adaptation(landscape.space.n_bins());
    let features =
        FeaturesNow::from_occupancy_scan(&landscape.space, &landscape.subjective_intensity);
    let trials = 256u64;
    let mut moved_with_temp = 0usize;
    let mut moved_without_temp = 0usize;

    for seed in 0..trials {
        let mut cold = PitchHillClimbPitchCore::new(30.0, current, 0.0);
        cold.set_temperature(0.0);
        let mut hot = PitchHillClimbPitchCore::new(30.0, current, 0.0);
        hot.set_temperature(0.2);

        let mut rng_a = SmallRng::seed_from_u64(5_000 + seed);
        let mut rng_b = SmallRng::seed_from_u64(5_000 + seed);
        let p0 = cold.propose_target(
            current,
            current,
            2.0f32.powf(current),
            1.0,
            &landscape,
            &perceptual,
            &features,
            &[],
            &mut rng_a,
        );
        let p1 = hot.propose_target(
            current,
            current,
            2.0f32.powf(current),
            1.0,
            &landscape,
            &perceptual,
            &features,
            &[],
            &mut rng_b,
        );
        if (p0.target_pitch_log2 - current).abs() > 1e-6 {
            moved_without_temp += 1;
        }
        if (p1.target_pitch_log2 - current).abs() > 1e-6 {
            moved_with_temp += 1;
        }
    }

    assert_eq!(
        moved_without_temp, 0,
        "temperature=0 should keep greedy hill-climb behavior"
    );
    assert!(
        moved_with_temp > 0 && moved_with_temp < trials as usize,
        "temperature should probabilistically accept downhill moves"
    );
}

#[test]
fn hillclimb_crowding_moves_away_from_close_neighbor() {
    let mut landscape = Landscape::new(Log2Space::new(220.0, 440.0, 256));
    let center_log2 = 330.0f32.log2();
    let width_cents = 20.0f32;
    for (idx, &bin_log2) in landscape.space.centers_log2.iter().enumerate() {
        let d_cents = (bin_log2 - center_log2).abs() * 1200.0;
        let score = (-(d_cents * d_cents) / (2.0 * width_cents * width_cents)).exp();
        landscape.consonance_field_score[idx] = score;
        landscape.consonance_field_level[idx] = score.clamp(0.0, 1.0);
        landscape.consonance_field_score_eff[idx] = score;
        landscape.consonance_field_level_eff[idx] = score.clamp(0.0, 1.0);
    }
    let perceptual = test_adaptation(landscape.space.n_bins());
    let features =
        FeaturesNow::from_occupancy_scan(&landscape.space, &landscape.subjective_intensity);

    let mut without_crowding = PitchHillClimbPitchCore::new(24.0, center_log2, 0.0);
    let mut with_crowding = without_crowding.clone();
    without_crowding.set_crowding(0.0, 60.0);
    with_crowding.set_crowding(6.0, 60.0);

    let mut rng_a = SmallRng::seed_from_u64(6001);
    let mut rng_b = SmallRng::seed_from_u64(6001);
    let off = without_crowding.propose_target(
        center_log2,
        center_log2,
        330.0,
        1.0,
        &landscape,
        &perceptual,
        &features,
        &[center_log2],
        &mut rng_a,
    );
    let on = with_crowding.propose_target(
        center_log2,
        center_log2,
        330.0,
        1.0,
        &landscape,
        &perceptual,
        &features,
        &[center_log2],
        &mut rng_b,
    );
    let dist_off = (off.target_pitch_log2 - center_log2).abs();
    let dist_on = (on.target_pitch_log2 - center_log2).abs();
    assert!(
        dist_on > dist_off + 1e-6,
        "crowding should bias hill-climb away from a close neighbor"
    );
}

#[test]
fn hillclimb_global_peaks_enable_distant_peak_escape() {
    let current_hz = 220.0;
    let distant_hz = 330.0;
    let landscape = test_landscape(&[(current_hz, 0.4), (distant_hz, 1.0)]);
    let perceptual = test_adaptation(landscape.space.n_bins());
    let features =
        FeaturesNow::from_occupancy_scan(&landscape.space, &landscape.subjective_intensity);
    let current_log2 = current_hz.log2();

    let mut local_only = PitchHillClimbPitchCore::new(120.0, current_log2, 0.0);
    local_only.set_global_peaks(0, 0.0);
    let mut with_global = local_only.clone();
    with_global.set_global_peaks(8, 20.0);

    let mut rng_a = SmallRng::seed_from_u64(3001);
    let mut rng_b = SmallRng::seed_from_u64(3001);
    let local = local_only.propose_target(
        current_log2,
        current_log2,
        current_hz,
        1.0,
        &landscape,
        &perceptual,
        &features,
        &[],
        &mut rng_a,
    );
    let global = with_global.propose_target(
        current_log2,
        current_log2,
        current_hz,
        1.0,
        &landscape,
        &perceptual,
        &features,
        &[],
        &mut rng_b,
    );

    let local_dist_cents = (local.target_pitch_log2 - current_log2).abs() * 1200.0;
    let global_dist_cents = (global.target_pitch_log2 - current_log2).abs() * 1200.0;
    assert!(
        global_dist_cents > local_dist_cents + 80.0,
        "global peaks should enable distant move (local={local_dist_cents:.1}c, global={global_dist_cents:.1}c)"
    );
}

#[test]
fn peak_sampler_converges_to_single_peak() {
    let peak_hz = 330.0;
    let landscape = test_landscape(&[(peak_hz, 1.0)]);
    let perceptual = test_adaptation(landscape.space.n_bins());
    let features =
        FeaturesNow::from_occupancy_scan(&landscape.space, &landscape.subjective_intensity);
    let mut rng = SmallRng::seed_from_u64(17);

    let mut core = PitchPeakSamplerCore::new(120.0, 300.0f32.log2(), 0.0, 700.0, 16, 0.03, 10.0, 2);

    let mut target_log2 = 300.0f32.log2();
    for _ in 0..24 {
        let proposal = core.propose_target(
            target_log2,
            target_log2,
            2.0f32.powf(target_log2),
            1.0,
            &landscape,
            &perceptual,
            &features,
            &[],
            &mut rng,
        );
        target_log2 = proposal.target_pitch_log2;
    }

    let target_hz = 2.0f32.powf(target_log2);
    assert!(
        (target_hz - peak_hz).abs() < 25.0,
        "target {target_hz} Hz did not approach {peak_hz} Hz"
    );
}

fn right_peak_ratio(
    core: &mut PitchPeakSamplerCore,
    landscape: &Landscape,
    seed: u64,
    left_peak_hz: f32,
    right_peak_hz: f32,
) -> f32 {
    let perceptual = test_adaptation(landscape.space.n_bins());
    let features =
        FeaturesNow::from_occupancy_scan(&landscape.space, &landscape.subjective_intensity);
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut right_hits = 0u32;
    let trials = 512u32;
    let start_log2 = 330.0f32.log2();

    for _ in 0..trials {
        let proposal = core.propose_target(
            start_log2,
            start_log2,
            330.0,
            1.0,
            landscape,
            &perceptual,
            &features,
            &[],
            &mut rng,
        );
        let hz = 2.0f32.powf(proposal.target_pitch_log2);
        let d_left = (hz - left_peak_hz).abs();
        let d_right = (hz - right_peak_hz).abs();
        if d_right < d_left {
            right_hits = right_hits.saturating_add(1);
        }
    }

    right_hits as f32 / trials as f32
}

#[test]
fn peak_sampler_temperature_and_noise_change_bimodal_mix() {
    let left_peak_hz = 300.0;
    let right_peak_hz = 360.0;
    let landscape = test_landscape(&[(left_peak_hz, 1.0), (right_peak_hz, 0.95)]);

    let mut cold = PitchPeakSamplerCore::new(90.0, 330.0f32.log2(), 0.0, 900.0, 18, 0.01, 0.0, 0);
    let mut hot = PitchPeakSamplerCore::new(90.0, 330.0f32.log2(), 0.0, 900.0, 18, 0.25, 40.0, 3);

    let cold_right = right_peak_ratio(&mut cold, &landscape, 99, left_peak_hz, right_peak_hz);
    let hot_right = right_peak_ratio(&mut hot, &landscape, 99, left_peak_hz, right_peak_hz);

    assert!(
        hot_right > cold_right + 0.08,
        "expected higher right-peak ratio with hotter sampler (cold={cold_right:.3}, hot={hot_right:.3})"
    );
}

#[test]
fn source_has_no_fixed_twelve_tet_candidate_patterns() {
    let src = concat!(
        include_str!("../pitch_core.rs"),
        include_str!("candidates.rs"),
        include_str!("hill_climb.rs"),
        include_str!("hill_climb/selection.rs"),
        include_str!("peak_sampler.rs"),
        include_str!("scoring.rs"),
    );
    let compact: String = src.chars().filter(|c| !c.is_ascii_whitespace()).collect();

    let twelfth_patterns = [
        ["3", ".0", "/", "12", ".0"].concat(),
        ["4", ".0", "/", "12", ".0"].concat(),
        ["7", ".0", "/", "12", ".0"].concat(),
    ];
    for pattern in &twelfth_patterns {
        assert!(!compact.contains(pattern));
    }

    let semitone_tokens = [
        ["-", "12", ".0"].concat(),
        ["-", "9", ".0"].concat(),
        ["-", "7", ".0"].concat(),
    ];
    assert!(!contains_ordered_tokens_with_gap(
        &compact,
        &semitone_tokens,
        3
    ));
}

#[test]
fn non_tet_peak_is_reached_by_hillclimb_and_peaksampler() {
    let peak_hz = 347.0;
    let landscape = test_landscape(&[(peak_hz, 1.0)]);
    let perceptual = test_adaptation(landscape.space.n_bins());
    let features =
        FeaturesNow::from_occupancy_scan(&landscape.space, &landscape.subjective_intensity);

    let mut hill_rng = SmallRng::seed_from_u64(123);
    let mut hill = PitchHillClimbPitchCore::new(120.0, 330.0f32.log2(), 0.0);
    let mut hill_target = 330.0f32.log2();
    for _ in 0..32 {
        let proposal = hill.propose_target(
            hill_target,
            hill_target,
            2.0f32.powf(hill_target),
            1.0,
            &landscape,
            &perceptual,
            &features,
            &[],
            &mut hill_rng,
        );
        hill_target = proposal.target_pitch_log2;
    }
    let hill_hz = 2.0f32.powf(hill_target);
    assert!(
        (hill_hz - peak_hz).abs() < 20.0,
        "hill climb did not converge to non-12TET peak ({hill_hz} vs {peak_hz})"
    );

    let mut sampler_rng = SmallRng::seed_from_u64(456);
    let mut sampler =
        PitchPeakSamplerCore::new(120.0, 330.0f32.log2(), 0.0, 700.0, 16, 0.03, 10.0, 2);
    let mut sampler_target = 330.0f32.log2();
    for _ in 0..32 {
        let proposal = sampler.propose_target(
            sampler_target,
            sampler_target,
            2.0f32.powf(sampler_target),
            1.0,
            &landscape,
            &perceptual,
            &features,
            &[],
            &mut sampler_rng,
        );
        sampler_target = proposal.target_pitch_log2;
    }
    let sampler_hz = 2.0f32.powf(sampler_target);
    assert!(
        (sampler_hz - peak_hz).abs() < 20.0,
        "peak sampler did not converge to non-12TET peak ({sampler_hz} vs {peak_hz})"
    );
}

fn contains_ordered_tokens_with_gap(text: &str, tokens: &[String], max_gap: usize) -> bool {
    if tokens.is_empty() {
        return true;
    }
    let first = tokens[0].as_str();
    let mut start_at = 0usize;
    while let Some(pos0) = text[start_at..].find(first) {
        let mut prev_end = start_at + pos0 + first.len();
        let mut ok = true;
        for token in &tokens[1..] {
            let search_to = prev_end.saturating_add(max_gap);
            let Some(window) = text.get(prev_end..search_to.min(text.len())) else {
                ok = false;
                break;
            };
            if let Some(rel) = window.find(token.as_str()) {
                prev_end += rel + token.len();
            } else {
                ok = false;
                break;
            }
        }
        if ok {
            return true;
        }
        start_at += pos0 + 1;
    }
    false
}

#[test]
fn peak_sampler_scorer_matches_adjusted_pitch_score() {
    let landscape = test_landscape(&[(330.0, 1.0), (440.0, 0.6)]);
    let perceptual = test_adaptation(landscape.space.n_bins());
    let current = 330.0f32.log2();
    let neighbor = [440.0f32.log2()];
    let salience = [1.0f32];

    for &loo in &[false, true] {
        let via_wrapper = adjusted_pitch_score(
            current,
            current,
            1.0,
            current,
            0.1,
            1.0,
            DEFAULT_MOVE_COST_COEFF,
            DEFAULT_MOVE_COST_EXP,
            &landscape,
            &perceptual,
            loo,
            0.0,
            0.0,
            &neighbor,
            &salience,
        );
        let via_direct = adjusted_pitch_score_with_loo_harmonics(
            current,
            current,
            1.0,
            current,
            0.1,
            1.0,
            DEFAULT_MOVE_COST_COEFF,
            DEFAULT_MOVE_COST_EXP,
            &landscape,
            &perceptual,
            loo,
            DEFAULT_LOO_HARMONICS,
            0.0,
            0.0,
            &neighbor,
            &salience,
        );
        assert!(
            (via_wrapper - via_direct).abs() < 1e-9,
            "adjusted_pitch_score must equal adjusted_pitch_score_with_loo_harmonics(\
             DEFAULT_LOO_HARMONICS) when leave_self_out={loo}"
        );
    }
}

#[test]
fn peak_sampler_proposal_unchanged_after_refactor() {
    let landscape = test_landscape(&[(330.0, 1.0)]);
    let perceptual = test_adaptation(landscape.space.n_bins());
    let features =
        FeaturesNow::from_occupancy_scan(&landscape.space, &landscape.subjective_intensity);
    let mut core_a =
        PitchPeakSamplerCore::new(120.0, 330.0f32.log2(), 0.0, 700.0, 16, 0.03, 10.0, 2);
    core_a.set_leave_self_out(true);
    let mut core_b = core_a.clone();

    let mut rng_a = SmallRng::seed_from_u64(42);
    let mut rng_b = SmallRng::seed_from_u64(42);
    let pa = core_a.propose_target(
        330.0f32.log2(),
        330.0f32.log2(),
        330.0,
        1.0,
        &landscape,
        &perceptual,
        &features,
        &[],
        &mut rng_a,
    );
    let pb = core_b.propose_target(
        330.0f32.log2(),
        330.0f32.log2(),
        330.0,
        1.0,
        &landscape,
        &perceptual,
        &features,
        &[],
        &mut rng_b,
    );
    assert!(
        (pa.target_pitch_log2 - pb.target_pitch_log2).abs() < 1e-9,
        "same core, same seed must produce same proposal"
    );
}

#[test]
fn propose_with_scorer_default_matches_trait_method() {
    let landscape = test_landscape(&[(330.0, 1.0), (440.0, 0.6)]);
    let perceptual = test_adaptation(landscape.space.n_bins());
    let features =
        FeaturesNow::from_occupancy_scan(&landscape.space, &landscape.subjective_intensity);
    let core = PitchHillClimbPitchCore::new(120.0, 330.0f32.log2(), 0.0);
    let current = 330.0f32.log2();

    let mut rng_a = SmallRng::seed_from_u64(99);
    let via_scorer = core.propose_with_scorer(
        current,
        current,
        &landscape,
        &[],
        &mut rng_a,
        |pitch_log2| {
            adjusted_pitch_score_with_loo_harmonics(
                pitch_log2,
                current,
                1.0,
                current,
                0.0,
                1.0,
                DEFAULT_MOVE_COST_COEFF,
                DEFAULT_MOVE_COST_EXP,
                &landscape,
                &perceptual,
                false,
                DEFAULT_LOO_HARMONICS,
                0.0,
                0.0,
                &[],
                &[],
            )
        },
    );

    let mut rng_b = SmallRng::seed_from_u64(99);
    let mut core2 = core.clone();
    let via_trait = core2.propose_target(
        current,
        current,
        330.0,
        1.0,
        &landscape,
        &perceptual,
        &features,
        &[],
        &mut rng_b,
    );

    assert!(
        (via_scorer.target_pitch_log2 - via_trait.target_pitch_log2).abs() < 1e-9,
        "propose_with_scorer with default scorer must match propose_target"
    );
}

#[test]
fn propose_with_scorer_negated_prefers_different_peak() {
    let landscape = test_landscape(&[(330.0, 1.0), (550.0, 0.3)]);
    let _perceptual = test_adaptation(landscape.space.n_bins());
    let mut core = PitchHillClimbPitchCore::new(120.0, 440.0f32.log2(), 0.0);
    core.set_global_peaks(3, 0.0);
    let current = 440.0f32.log2();

    let mut rng_pos = SmallRng::seed_from_u64(200);
    let positive = core.propose_with_scorer(
        current,
        current,
        &landscape,
        &[],
        &mut rng_pos,
        |pitch_log2| landscape.evaluate_pitch_score_log2(pitch_log2),
    );

    let mut rng_neg = SmallRng::seed_from_u64(200);
    let negated = core.propose_with_scorer(
        current,
        current,
        &landscape,
        &[],
        &mut rng_neg,
        |pitch_log2| -landscape.evaluate_pitch_score_log2(pitch_log2),
    );

    assert!(
        (positive.target_pitch_log2 - negated.target_pitch_log2).abs() > 1e-3,
        "negated scorer should produce a different proposal than positive scorer \
         (pos={}, neg={})",
        positive.target_pitch_log2,
        negated.target_pitch_log2,
    );
}

#[test]
#[ignore = "TC1b-measure lock; builds the complete production probe table"]
fn tonal_prior_context_changes_choice() {
    use crate::core::tonal_context::{Context, Decoder, Use};
    let decoder = Arc::new(Decoder::new());
    let landscape = test_landscape(&[]);
    let current = 220_f32.log2();
    let mut selected = Vec::new();
    for hz in [220., 330.] {
        let mut heard = Context::new(decoder.clone(), 0);
        let pcm: Vec<_> = (0..96000)
            .map(|n| (0.1 * (std::f64::consts::TAU * hz * n as f64 / 48000.).sin()) as f32)
            .collect();
        heard.process(&pcm);
        let snapshot = heard.snapshot().unwrap();
        let q220 = snapshot.query(220.).unwrap();
        let q330 = snapshot.query(330.).unwrap();
        println!("context={hz} q220={q220} q330={q330}");
        assert_eq!(q220 > q330, hz == 220.);
        let mut core = PitchHillClimbPitchCore::new(701.955, current, 0.);
        let mut term = Use::new();
        term.pin(Some(snapshot), 2., (110_f32.log2(), 440_f32.log2()));
        core.tonal = Some(term);
        let mut rng = SmallRng::seed_from_u64(21);
        let proposal =
            core.propose_with_scorer(current, current, &landscape, &[], &mut rng, |_| 0.);
        assert_eq!(
            core.tonal.as_ref().unwrap().selection_known.get(),
            Some(true)
        );
        println!("context={hz} target={}", proposal.target_pitch_log2.exp2());
        selected.push(proposal.target_pitch_log2.to_bits());
    }
    assert_ne!(selected[0], selected[1]);
}

#[test]
fn pitch_set_history_changes_choice_and_unknown_tc_keeps_ps() {
    let landscape = test_landscape(&[]);
    let current = 250_f32.log2();
    let bounds = (100_f32.log2(), 400_f32.log2());
    let mut selected = Vec::new();
    for hz in [220., 307.] {
        let mut core = PitchHillClimbPitchCore::new(10., current, 0.);
        core.pitch_set = Some(crate::core::tonal_context::pitch_set::Use {
            snapshot: Some(crate::core::tonal_context::pitch_set::heard_for_test(hz)),
            weight: 4.,
            bounds,
            trace: true,
            decision: Default::default(),
            added: Default::default(),
            tc_failed_added: Default::default(),
        });
        let mut rng = SmallRng::seed_from_u64(21);
        let without = core.propose_with_scorer(current, current, &landscape, &[], &mut rng, |_| 0.);
        let mut tonal = crate::core::tonal_context::Use::new();
        tonal.pin(None, 1., bounds);
        core.tonal = Some(tonal);
        let mut rng2 = SmallRng::seed_from_u64(21);
        let with = core.propose_with_scorer(current, current, &landscape, &[], &mut rng2, |_| 0.);
        assert_eq!(without.target_pitch_log2, with.target_pitch_log2);
        assert_eq!(rng.random::<u64>(), rng2.random::<u64>());
        assert_eq!(
            core.tonal.as_ref().unwrap().selection_known.get(),
            Some(false)
        );
        assert_ne!(with.target_pitch_log2, current);
        selected.push(with.target_pitch_log2);
        // SQ failure restores its own term and leaves PS active.
        core.sequence_enabled = true;
        core.sequence = Some(crate::life::own_sequence::SequenceSnapshot {
            probabilities: [0.; 241],
            alphabet: 241,
            offset: 120,
            previous_hz: 250.,
            weight: 1.,
            bounds_log2: bounds,
            at: 0,
            version: 0,
            epoch: 0,
        });
        let mut rng3 = SmallRng::seed_from_u64(21);
        let failed_sq =
            core.propose_with_scorer(current, current, &landscape, &[], &mut rng3, |_| 0.);
        assert_eq!(failed_sq.target_pitch_log2, with.target_pitch_log2);
        assert!(!core.sequence_decision.borrow().as_ref().unwrap().active);
        core.sequence_trace = true;
        core.sequence
            .as_mut()
            .unwrap()
            .probabilities
            .fill(1. / 241.);
        core.propose_with_scorer(current, current, &landscape, &[], &mut rng3, |_| 0.);
        let decision = core.sequence_decision.borrow();
        let decision = decision.as_ref().unwrap();
        assert!(decision.active);
        assert!(!decision.candidates.is_empty());
        assert!(decision.precision.is_none());
    }
    assert_ne!(selected[0], selected[1]);
}

#[test]
fn pitch_set_added_unknown_candidate_disables_only_tc() {
    let landscape = test_landscape(&[]);
    let current = 250_f32.log2();
    let bounds = (100_f32.log2(), 400_f32.log2());
    let mut core = PitchHillClimbPitchCore::new(10., current, 0.);
    core.pitch_set = Some(crate::core::tonal_context::pitch_set::Use {
        snapshot: Some(crate::core::tonal_context::pitch_set::heard_for_test(220.)),
        weight: 4.,
        bounds,
        trace: true,
        decision: Default::default(),
        added: Default::default(),
        tc_failed_added: Default::default(),
    });
    let run = |core: &PitchHillClimbPitchCore| {
        let mut rng = SmallRng::seed_from_u64(21);
        core.propose_with_scorer(current, current, &landscape, &[], &mut rng, |_| 0.)
    };
    let expected = run(&core);
    let record = core
        .pitch_set
        .as_ref()
        .unwrap()
        .decision
        .borrow()
        .clone()
        .unwrap();
    assert!(!record.added_candidates.is_empty());
    let mut context = crate::core::tonal_context::Context::new(
        crate::core::tonal_context::Decoder::unqueried_for_test(),
        0,
    );
    context.process(&vec![0.1; 12000]);
    let mut tonal = crate::core::tonal_context::Use::new();
    tonal.pin(context.snapshot(), 1., bounds);
    tonal.seed_query_for_test(current.exp2(), Some(0.));
    for &(p, _, _) in &record.candidates {
        if !record.added_candidates.contains(&p) {
            tonal.seed_query_for_test(p.exp2(), Some(0.));
        }
    }
    core.tonal = Some(tonal);
    let actual = run(&core);
    assert_eq!(actual.target_pitch_log2, expected.target_pitch_log2);
    assert_eq!(
        core.tonal.as_ref().unwrap().selection_known.get(),
        Some(false)
    );
    assert!(
        core.pitch_set
            .as_ref()
            .unwrap()
            .decision
            .borrow()
            .as_ref()
            .unwrap()
            .tc_failed_added
    );
}

#[test]
#[ignore = "PS1_UNIT: independent continuous histories and candidate acceptance"]
fn pitch_set_independent_candidate_selection() {
    let root = std::path::PathBuf::from(std::env::var("PS1_UNIT").unwrap());
    let data: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("candidate-acceptance.json")).unwrap())
            .unwrap();
    let load = |hz: i64| -> Vec<f64> {
        let d: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join(format!("reference/history-{hz}.json"))).unwrap(),
        )
        .unwrap();
        d["coefficients"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    };
    let first = load(220);
    let mut definite = 0;
    for case in data["cases"].as_array().unwrap() {
        if case["extraction_uncertain"].as_bool().unwrap()
            || case["selection_uncertain"].as_bool().unwrap()
        {
            continue;
        }
        let second = load(case["history"][1].as_i64().unwrap());
        let mix = case["history"][2].as_f64().unwrap();
        let memory = crate::core::tonal_context::pitch_set::coefficients_for_test(
            first
                .iter()
                .zip(second)
                .map(|(a, b)| (1. - mix) * a + mix * b)
                .collect(),
        );
        let hz = case["current_hz"].as_f64().unwrap();
        let current = hz.log2() as f32;
        let bounds = (
            case["bounds"][0].as_f64().unwrap().log2() as f32,
            case["bounds"][1].as_f64().unwrap().log2() as f32,
        );
        let base = |p: f32| {
            let p = f64::from(p);
            let target = hz.log2();
            (-0.12 * (p - target).powi(2) + 0.03 * (3. * p).sin()
                - 0.02 * (p - (target + 0.1)).powi(2)
                + 0.01 * (5. * p).cos()) as f32
        };
        let mut scored: Vec<_> = [-0.04, 0., 0.04]
            .into_iter()
            .map(|d| ((hz.log2() + d) as f32).clamp(bounds.0, bounds.1))
            .map(|p| (p, base(p)))
            .collect();
        for p in memory.candidates(current, bounds).into_iter().flatten() {
            if !scored
                .iter()
                .any(|(x, _)| x.exp2().to_bits() == p.exp2().to_bits())
            {
                scored.push((p, base(p)));
            }
        }
        let mut core = PitchHillClimbPitchCore::new(0., current, 0.);
        core.pitch_set = Some(crate::core::tonal_context::pitch_set::Use {
            snapshot: Some(memory),
            weight: case["weight"].as_f64().unwrap(),
            bounds,
            trace: false,
            decision: Default::default(),
            added: Default::default(),
            tc_failed_added: Default::default(),
        });
        let actual = core.select_with_pitch_set(
            current,
            base(current),
            &mut scored,
            &mut SmallRng::seed_from_u64(21),
            None,
        );
        let expected = case["reference_hz"].as_f64().unwrap().log2();
        let allowance =
            case["position_error_cents"].as_f64().unwrap() / 1200. + 4. * f64::from(f32::EPSILON);
        assert!(
            (f64::from(actual.target_pitch_log2) - expected).abs() <= allowance,
            "{case}"
        );
        definite += 1;
    }
    println!(
        "independent candidate extraction and final Rust selection: {definite} definite decisions"
    );
    assert!(definite > 0);
}

#[test]
fn pitch_set_narrow_range_large_weight_and_shadow_rng() {
    let landscape = test_landscape(&[]);
    let current = 220_f32.log2();
    let mut core = PitchHillClimbPitchCore::new(500., current, 0.);
    let memory = crate::core::tonal_context::pitch_set::heard_for_test(220.);
    core.pitch_set = Some(crate::core::tonal_context::pitch_set::Use {
        snapshot: Some(memory),
        weight: 1e300,
        bounds: (219_f32.log2(), 221_f32.log2()),
        trace: false,
        decision: Default::default(),
        added: Default::default(),
        tc_failed_added: Default::default(),
    });
    core.temperature = 0.1;
    let mut rng = SmallRng::seed_from_u64(19);
    let mut shadow = rng.clone();
    let quiet = core.propose_with_scorer(current, current, &landscape, &[], &mut rng, |p| -p);
    core.pitch_set.as_mut().unwrap().trace = true;
    let traced = core.propose_with_scorer(current, current, &landscape, &[], &mut shadow, |p| -p);
    assert_eq!(quiet.target_pitch_log2, traced.target_pitch_log2);
    assert_eq!(rng.random::<u64>(), shadow.random::<u64>());
    assert!(traced.salience.is_finite());
    assert!((219_f32.log2()..=221_f32.log2()).contains(&traced.target_pitch_log2));
    for &(p, _, _) in &core
        .pitch_set
        .as_ref()
        .unwrap()
        .decision
        .borrow()
        .as_ref()
        .unwrap()
        .candidates
    {
        assert!((219_f32.log2()..=221_f32.log2()).contains(&p));
    }
}

#[test]
fn tonal_unknown_preserves_candidate_selection_and_rng() {
    let landscape = test_landscape(&[]);
    let mut off = PitchHillClimbPitchCore::new(100., 220_f32.log2(), 0.);
    off.temperature = 1.;
    let mut unknown = off.clone();
    let mut term = crate::core::tonal_context::Use::new();
    term.pin(None, 1e100, (110_f32.log2(), 440_f32.log2()));
    unknown.tonal = Some(term);
    let mut a = SmallRng::seed_from_u64(21);
    let mut b = a.clone();
    let current = 220_f32.log2();
    let pa = off.propose_with_scorer(current, current, &landscape, &[], &mut a, |p| p.sin());
    let pb = unknown.propose_with_scorer(current, current, &landscape, &[], &mut b, |p| p.sin());
    assert_eq!(
        pa.target_pitch_log2.to_bits(),
        pb.target_pitch_log2.to_bits()
    );
    assert_eq!(a.random::<u64>(), b.random::<u64>());
    assert_eq!(unknown.tonal.unwrap().selection_known.get(), Some(false));
}

#[test]
fn propose_with_scorer_non_exploratory_plateau_stays_put() {
    let landscape = test_landscape(&[(330.0, 1.0), (550.0, 0.3)]);
    let mut core = PitchHillClimbPitchCore::new(120.0, 440.0f32.log2(), 0.0);
    core.set_global_peaks(3, 0.0);
    let current = 440.0f32.log2();

    let mut rng = SmallRng::seed_from_u64(7);
    let proposal = core.propose_with_scorer(current, current, &landscape, &[], &mut rng, |_| 0.0);

    assert!(
        (proposal.target_pitch_log2 - current).abs() < 1e-9,
        "non-exploratory plateau should stay at current target (current={}, proposed={})",
        current,
        proposal.target_pitch_log2,
    );
}
#[test]
fn hr2_global_and_local_filter_before_scoring_and_top_k() {
    let landscape = Landscape::new(crate::core::log2space::production_space(48000).unwrap());
    let (lo, hi) = landscape.space.voice_pitch_bounds_log2();
    let inside = landscape.space.centers_log2[landscape.space.nearest_index(7000.0)];
    let mut scored = Vec::new();
    let mut out = Vec::new();
    let mut seen = Vec::new();
    let mut scorer = |pitch: f32| {
        seen.push(pitch);
        if pitch < lo || pitch > hi {
            1000.0
        } else {
            -(pitch - inside).abs()
        }
    };
    top_global_peaks(&landscape, 1, 0.0, &mut scorer, &mut scored, &mut out);
    assert_eq!(out, vec![inside]);
    assert!(seen.iter().all(|&p| p >= lo && p <= hi));
    for edge in [lo, hi] {
        seen.clear();
        out.clear();
        let expected = landscape
            .space
            .centers_log2
            .iter()
            .copied()
            .filter(|&p| p >= lo && p <= hi)
            .min_by(|a, b| (a - edge).abs().total_cmp(&(b - edge).abs()))
            .unwrap();
        top_local_peaks(
            &landscape,
            edge,
            48,
            1,
            &mut |pitch| {
                seen.push(pitch);
                if pitch < lo || pitch > hi {
                    1000.0
                } else {
                    -(pitch - edge).abs()
                }
            },
            &mut scored,
            &mut out,
        );
        assert_eq!(out, vec![expected]);
        assert!(seen.iter().all(|&p| p >= lo && p <= hi));
    }
}
