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
        total_power_scan: vec![1.0; l.space.n_bins()].into(),
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
    let mut total = own.clone();
    total[idx] += 3.0;
    let mut input = LooInput {
        own_power_scan: own,
        total_power_scan: total.into(),
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
        total_power_scan: own.into(),
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
        total_power_scan: vec![1.0].into(),
        ..Default::default()
    };
    loo_score_scan(&l, &mut input, &mut None);
}

#[test]
fn captured_work_proposals_preserve_scores_ranks_and_acceptance_probabilities() {
    let records: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("testdata/t31_loo_proposals.json")).unwrap();
    assert_eq!(records.len(), 6);
    let mut checked = 0;
    for row in &records {
        let floats = |key: &str| -> Vec<f32> {
            row[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        let meta = floats("space");
        let mut l = Landscape::new(Log2Space::new(meta[0], meta[1], meta[2] as u32));
        l.subjective_intensity = floats("density");
        l.perc_habituation_state_scan = floats("habituation");
        l.loudness_exp = row["loudness_exp"].as_f64().unwrap() as f32;
        l.consonance_theta = row["theta"].as_f64().unwrap() as f32;
        let r = floats("r_params");
        let rp = crate::core::roughness_kernel::KernelParams {
            tau_erb: r[0],
            mix_tail: r[1],
            sethares_gain: r[2],
            sethares_b: r[3],
            sethares_c: r[4],
            sethares_kappa_erb: r[5],
            half_width_erb: r[6],
            suppress_sigma_erb: r[7],
            suppress_pow: r[8],
            sigma_neural_erb: r[9],
            w_neural: r[10],
        };
        let h = row["h_params"].as_array().unwrap();
        let hf = |i: usize| h[i].as_f64().unwrap() as f32;
        let hp = crate::core::harmonicity_kernel::HarmonicityParams {
            num_subharmonics: h[0].as_u64().unwrap() as u32,
            num_harmonics: h[1].as_u64().unwrap() as u32,
            rho_common_root: hf(2),
            sigma_cents: hf(3),
            normalize_output: h[4].as_bool().unwrap(),
            gamma_root: hf(5),
            diag_weight: hf(6),
            freq_gate: h[7].as_bool().unwrap(),
            tfs_f_pl_hz: hf(8),
            tfs_eta: hf(9),
        };
        let k = floats("kernel");
        let kernel = crate::core::consonance_kernel::ConsonanceKernel {
            a: k[0],
            b: k[1],
            c: k[2],
            d: k[3],
        };
        l.set_analysis_metadata(
            rp,
            hp,
            kernel,
            row["r_k"].as_f64().unwrap() as f32,
            row["r_ref"].as_f64().unwrap() as f32,
            row["eps"].as_f64().unwrap() as f32,
        );
        let mut input = LooInput {
            own_power_scan: floats("own"),
            total_power_scan: floats("total").into(),
            ..Default::default()
        };
        let current = row["proposal"]["current"].as_f64().unwrap() as f32;
        let target = row["proposal"]["current_target"].as_f64().unwrap() as f32;
        let temperature = row["proposal"]["temperature"].as_f64().unwrap();
        let approx = loo_score_scan(&l, &mut input, &mut None);
        let exact = exact_loo_consonance_score_scan(
            &l,
            current,
            true,
            ReferenceLooMode::ExactScan,
            &mut None,
        );
        for (name, scan) in [("corrected", approx), ("exact", exact.map(Arc::from))] {
            // A missing residual is the observed-field fallback, whose full scan
            // is deliberately not copied into these small captured fixtures.
            let Some(scan) = scan else {
                continue;
            };
            let variant = row["proposal"]["variants"]
                .as_array()
                .unwrap()
                .iter()
                .find(|v| v["variant"] == name)
                .unwrap();
            let mut actual = Vec::new();
            let mut expected = Vec::new();
            for score in variant["scores"].as_array().unwrap() {
                let p = score[0].as_f64().unwrap() as f32;
                let field = sample_consonance_score_with_loo(
                    p,
                    current,
                    &l,
                    true,
                    ReferenceLooMode::ApproxHarmonics,
                    1,
                    Some(&scan),
                    None,
                );
                let old_field = score[1].as_f64().unwrap() as f32;
                // Each field error is bounded to one quarter of the existing
                // 1e-6 scorer tie width, leaving half for a two-score difference.
                assert!(
                    (field - old_field).abs() <= 2.5e-7,
                    "{} {name}: {p} {field} {old_field}",
                    row["work"]
                );
                let adjusted = score[2].as_f64().unwrap();
                actual.push((p, adjusted + f64::from(field - old_field)));
                expected.push((p, adjusted));
            }
            let current_score = actual.iter().find(|(p, _)| *p == target).unwrap().1;
            let old_current = expected.iter().find(|(p, _)| *p == target).unwrap().1;
            for i in 0..actual.len() {
                for j in i + 1..actual.len() {
                    let old_gap = expected[i].1 - expected[j].1;
                    if old_gap.abs() > 1e-6 {
                        assert_eq!(
                            (actual[i].1 - actual[j].1).is_sign_positive(),
                            old_gap.is_sign_positive()
                        );
                    }
                }
                let probability = |delta: f64| {
                    if delta >= 0.0 {
                        1.0
                    } else if temperature > 0.0 {
                        (delta / temperature).exp()
                    } else {
                        0.0
                    }
                };
                let p = probability(actual[i].1 - current_score);
                let old_p = probability(expected[i].1 - old_current);
                // Derivative of exp(delta/T) on delta<=0 is at most 1/T.
                let budget = if temperature > 0.0 {
                    5e-7 / temperature
                } else {
                    0.0
                };
                assert!((p - old_p).abs() <= budget + f64::EPSILON);
            }
            checked += 1;
        }
    }
    assert!(checked >= 6);
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
#[should_panic(expected = "loo_total_power_scan")]
fn total_ownership_scan_boundary_is_checked() {
    let l = fixture();
    let mut input = LooInput {
        own_power_scan: vec![0.0; l.space.n_bins()],
        ..Default::default()
    };
    loo_score_scan(&l, &mut input, &mut None);
}
