//! Unfitted rating feature assembly and the specified proportional-odds link.

const GROOVE_BLOCKS: [usize; 6] = [14, 14, 14, 8, 3, 1];

#[derive(Clone, Copy, Debug)]
pub(super) struct WeightedRating {
    pub weight: f64,
    pub log_probabilities: Option<[f64; 5]>,
}

pub(super) struct RatingProjection {
    pub log_probabilities: [f64; 5],
    pub observed_coverage: f64,
    pub supported_mass: f64,
    pub reported_support_mass: f64,
}

fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        b
    } else if b == f64::NEG_INFINITY {
        a
    } else {
        a.max(b) + (-(a - b).abs()).exp().ln_1p()
    }
}

/// Mix supported group/path mass before any temperature or prior backoff.
pub(super) fn mix_rating(predictions: &[WeightedRating]) -> Option<WeightedRating> {
    if predictions.len() > 1024 {
        return None;
    }
    // Missing and unlisted weight remains unknown, including inside one hypothesis.
    let mut total_mass = 0.0;
    let mut supported_mass = 0.0;
    let mut mixture = [f64::NEG_INFINITY; 5];
    for row in predictions {
        if !row.weight.is_finite() || !(0.0..=1.0).contains(&row.weight) {
            return None;
        }
        total_mass += row.weight;
        if let Some(log_p) = row.log_probabilities {
            if log_p.iter().any(|x| x.is_nan() || *x > 0.0)
                || (log_p.iter().map(|x| x.exp()).sum::<f64>() - 1.0).abs() > 1e-12
            {
                return None;
            }
            supported_mass += row.weight;
            for (target, value) in mixture.iter_mut().zip(log_p) {
                *target = log_add(*target, row.weight.ln() + value);
            }
        }
    }
    if total_mass > 1.0 + 1e-12 {
        return None;
    }
    // Tolerance covers summing at most 1024 normalized f64 factors, not missing mass.
    let normalization_mass = supported_mass;
    supported_mass = supported_mass.min(1.0);
    if normalization_mass > 0. {
        for value in &mut mixture {
            *value = (*value - normalization_mass.ln()).min(0.);
        }
    }
    Some(WeightedRating {
        weight: supported_mass,
        log_probabilities: (normalization_mass > 0.).then_some(mixture),
    })
}

pub(super) fn project_rating(
    predictions: &[WeightedRating],
    observed_coverage: f64,
    training_prior: [f64; 5],
    temperature: f64,
) -> Option<RatingProjection> {
    if !observed_coverage.is_finite()
        || !(0.0..=1.0).contains(&observed_coverage)
        || !temperature.is_finite()
        || temperature <= 0.0
        || training_prior.iter().any(|x| !x.is_finite() || *x <= 0.0)
        || (training_prior.iter().sum::<f64>() - 1.0).abs() > 1e-12
    {
        return None;
    }
    let mixed = mix_rating(predictions)?;
    let supported_mass = mixed.weight;
    let mut mixture = mixed.log_probabilities.unwrap_or([f64::NEG_INFINITY; 5]);
    let reported_support_mass = observed_coverage * supported_mass;
    let prior_log = training_prior.map(f64::ln);
    let mut reported = prior_log;
    if reported_support_mass > 0.0 {
        for value in &mut mixture {
            if *value != f64::NEG_INFINITY {
                *value /= temperature;
                if !value.is_finite() {
                    return None;
                }
            }
        }
        let log_normalizer = mixture.into_iter().fold(f64::NEG_INFINITY, log_add);
        if !log_normalizer.is_finite() {
            return None;
        }
        for index in 0..5 {
            reported[index] = log_add(
                reported_support_mass.ln() + mixture[index] - log_normalizer,
                (-reported_support_mass).ln_1p() + prior_log[index],
            );
        }
    }
    Some(RatingProjection {
        log_probabilities: reported,
        observed_coverage,
        supported_mass,
        reported_support_mass,
    })
}

pub(super) fn groove_features(
    raw: &[Option<f64>; 54],
    means: &[f64; 54],
    deviations: &[f64; 54],
) -> Option<[f64; 109]> {
    if means.iter().any(|x| !x.is_finite())
        || deviations.iter().any(|x| !x.is_finite() || *x < 0.0)
        || raw.iter().flatten().any(|x| !x.is_finite())
    {
        return None;
    }
    let mut out = [0.0; 109];
    out[0] = 1.0;
    let mut source = 0;
    let mut target = 1;
    for count in GROOVE_BLOCKS {
        for index in 0..count {
            if let Some(value) = raw[source + index] {
                let z = (value - means[source + index]) / deviations[source + index].max(1e-6);
                if !z.is_finite() {
                    return None;
                }
                out[target + index] = z;
            } else {
                out[target + count + index] = 1.0;
            }
        }
        source += count;
        target += count * 2;
    }
    Some(out)
}

pub(super) fn ordinal_log_probabilities(eta: f64, cutpoints: [f64; 4]) -> Option<[f64; 5]> {
    if !eta.is_finite()
        || cutpoints.iter().any(|x| !x.is_finite())
        || cutpoints.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return None;
    }
    let logits = cutpoints.map(|alpha| alpha - eta);
    if logits.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let log_sigmoid = |x: f64| x.min(0.0) - (-x.abs()).exp().ln_1p();
    let mut out = [0.0; 5];
    out[0] = log_sigmoid(logits[0]);
    out[4] = log_sigmoid(-logits[3]);
    for index in 1..4 {
        // Use original cutpoint separation: subtracting shifted logits loses narrow bins.
        let width = cutpoints[index] - cutpoints[index - 1];
        out[index] = log_sigmoid(logits[index])
            + log_sigmoid(-logits[index - 1])
            + (-(-width).exp_m1()).ln();
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rating_mixture_matches_independent_probability_space_oracle() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/temporal_cognition/ratings.json"
        ))
        .unwrap();
        for row in fixture["mixtures"].as_array().unwrap() {
            let predictions: Vec<_> = row["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| WeightedRating {
                    weight: r["weight"].as_f64().unwrap(),
                    log_probabilities: r["probabilities"]
                        .as_array()
                        .map(|p| std::array::from_fn(|i| p[i].as_f64().unwrap().ln())),
                })
                .collect();
            let coverage = row["observed_coverage"].as_f64().unwrap();
            let prior = std::array::from_fn(|i| row["training_prior"][i].as_f64().unwrap());
            let actual = project_rating(
                &predictions,
                coverage,
                prior,
                row["temperature"].as_f64().unwrap(),
            )
            .unwrap();
            assert_eq!(actual.observed_coverage, coverage);
            assert!(
                (actual.supported_mass - row["supported_mass"].as_f64().unwrap()).abs() < 1e-14
            );
            assert!(
                (actual.reported_support_mass - row["reported_support_mass"].as_f64().unwrap())
                    .abs()
                    < 1e-14
            );
            for (index, value) in actual.log_probabilities.into_iter().enumerate() {
                assert!((value - row["log_probabilities"][index].as_f64().unwrap()).abs() < 2e-13);
            }
        }
    }

    #[test]
    fn temperature_follows_marginalization_and_never_transforms_unknown_prior() {
        let a = [0.8_f64, 0.05, 0.05, 0.05, 0.05];
        let b = [0.05_f64, 0.05, 0.05, 0.05, 0.8];
        let rows = [
            WeightedRating {
                weight: 0.25,
                log_probabilities: Some(a.map(f64::ln)),
            },
            WeightedRating {
                weight: 0.25,
                log_probabilities: Some(b.map(f64::ln)),
            },
        ];
        let prior = [0.5, 0.2, 0.1, 0.1, 0.1];
        let actual = project_rating(&rows, 0.8, prior, 2.0).unwrap();
        assert_eq!(actual.supported_mass, 0.5);
        assert_eq!(actual.reported_support_mass, 0.4);
        let early_normalizer = a.into_iter().map(f64::sqrt).sum::<f64>();
        let incorrectly_tempered_paths =
            0.4 * (a[0].sqrt() + b[0].sqrt()) / (2.0 * early_normalizer) + 0.6 * prior[0];
        assert!((actual.log_probabilities[0].exp() - incorrectly_tempered_paths).abs() > 1e-3);
        for temperature in [0.1, 1.0, 10.0] {
            let unknown = project_rating(&[], 1.0, prior, temperature).unwrap();
            assert_eq!(unknown.log_probabilities, prior.map(f64::ln));
            assert_eq!(unknown.reported_support_mass, 0.0);
        }
        // Full support retains tiny log probabilities instead of clipping loss at zero CDF difference.
        let tail = ordinal_log_probabilities(1000.0, [-2.0, -0.5, 0.5, 2.0]).unwrap();
        let full = project_rating(
            &[WeightedRating {
                weight: 1.0,
                log_probabilities: Some(tail),
            }],
            1.0,
            prior,
            1.0,
        )
        .unwrap();
        assert_eq!(full.log_probabilities, tail);
    }

    #[test]
    fn malformed_mixture_weights_support_and_distributions_are_rejected() {
        let prior = [0.2; 5];
        let rows = [
            WeightedRating {
                weight: 0.6,
                log_probabilities: None,
            },
            WeightedRating {
                weight: 0.6,
                log_probabilities: None,
            },
        ];
        assert!(project_rating(&rows, 1.0, prior, 1.0).is_none());
        for logs in [[f64::NAN; 5], [f64::NEG_INFINITY; 5], [0.0; 5]] {
            assert!(
                project_rating(
                    &[WeightedRating {
                        weight: 1.0,
                        log_probabilities: Some(logs)
                    }],
                    1.0,
                    prior,
                    1.0
                )
                .is_none()
            );
        }
        assert!(project_rating(&[], 1.01, prior, 1.0).is_none());
        assert!(project_rating(&[], 1.0, [0.0; 5], 1.0).is_none());
        assert!(project_rating(&[], 1.0, prior, 0.0).is_none());
        let oversized: Vec<_> = (0..1025)
            .map(|_| WeightedRating {
                weight: 0.0,
                log_probabilities: None,
            })
            .collect();
        assert!(project_rating(&oversized, 1.0, prior, 1.0).is_none());
    }

    #[test]
    fn ordinal_link_matches_independent_decimal_cdf_differences() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/temporal_cognition/ratings.json"
        ))
        .unwrap();
        for row in fixture["ordinal"].as_array().unwrap() {
            let eta = row["eta"].as_f64().unwrap();
            let cuts = std::array::from_fn(|i| row["cutpoints"][i].as_f64().unwrap());
            let actual = ordinal_log_probabilities(eta, cuts).unwrap();
            for (index, value) in actual.into_iter().enumerate() {
                let expected = row["log_probabilities"][index].as_f64().unwrap();
                assert!(
                    (value - expected).abs() <= 2e-13 * expected.abs().max(1.0),
                    "eta={eta}, category={index}: {value} != {expected}"
                );
            }
            assert!((actual.into_iter().map(f64::exp).sum::<f64>() - 1.0).abs() < 2e-14);
        }
    }

    #[test]
    fn rating_layout_preserves_missing_known_zero_and_fixed_blocks() {
        let mut raw = std::array::from_fn(|i| Some(i as f64));
        raw[0] = None;
        raw[14] = Some(0.0);
        raw[28] = None;
        raw[53] = None;
        let values = groove_features(&raw, &[0.0; 54], &[1.0; 54]).unwrap();
        assert_eq!(values[0], 1.0);
        assert_eq!((values[1], values[15]), (0.0, 1.0));
        assert_eq!((values[29], values[43]), (0.0, 0.0));
        assert_eq!((values[57], values[71]), (0.0, 1.0));
        assert_eq!((values[85], values[92], values[93]), (42.0, 49.0, 0.0));
        assert_eq!((values[101], values[103], values[104]), (50.0, 52.0, 0.0));
        assert_eq!((values[107], values[108]), (0.0, 1.0));
        let scaled = groove_features(&[Some(0.0); 54], &[2.0; 54], &[0.0; 54]).unwrap();
        assert_eq!(scaled[1], -2_000_000.0);
        assert_eq!(scaled[15], 0.0);
        let registry: serde_json::Value = serde_json::from_str(include_str!(
            "../../docs/roadmap/temporal-dcc/feature-manifest.json"
        ))
        .unwrap();
        let layout = registry["ordered_layouts"]["groove_desire_109"]
            .as_array()
            .unwrap();
        assert_eq!(layout.len(), 109);
        for (index, id) in [
            (1, "timing.inter_1.mode_1.cos"),
            (29, "timing.inter_2.mode_1.cos"),
            (57, "timing.within.mode_1.cos"),
            (85, "timing.density_0.125s"),
            (101, "timing.word_entropy"),
            (107, "timing.grouping_support"),
        ] {
            assert_eq!(layout[index]["id"], id);
        }
    }

    #[test]
    fn malformed_ratings_and_unrepresentable_standardization_are_rejected() {
        for cuts in [[0.0; 4], [1.0, 0.0, 2.0, 3.0], [0.0, 1.0, 2.0, f64::NAN]] {
            assert!(ordinal_log_probabilities(0.0, cuts).is_none());
        }
        assert!(ordinal_log_probabilities(f64::INFINITY, [0.0, 1.0, 2.0, 3.0]).is_none());
        assert!(ordinal_log_probabilities(-f64::MAX, [1e308, 1.1e308, 1.2e308, 1.3e308]).is_none());
        assert!(groove_features(&[Some(f64::NAN); 54], &[0.0; 54], &[1.0; 54]).is_none());
        assert!(groove_features(&[None; 54], &[0.0; 54], &[-1.0; 54]).is_none());
        assert!(groove_features(&[Some(f64::MAX); 54], &[-f64::MAX; 54], &[1.0; 54]).is_none());
    }
}
