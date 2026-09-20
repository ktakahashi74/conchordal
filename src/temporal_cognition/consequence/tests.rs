use super::*;

fn full_head(log_probabilities: [f64; 5]) -> WeightedRating {
    WeightedRating {
        weight: 1.,
        log_probabilities: Some(log_probabilities),
    }
}

#[test]
fn partial_group_support_survives_candidate_reweighting_and_single_final_backoff() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/temporal_cognition/consequence_groups.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let rows: Vec<_> = case["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| Alternative {
                weight: row["weight"].as_f64().unwrap(),
                delta_log_score: row["delta"].as_f64(),
                heads: std::array::from_fn(|h| {
                    let groups: Vec<_> = row["head_groups"][h]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|g| WeightedRating {
                            weight: g["weight"].as_f64().unwrap(),
                            log_probabilities: g["probabilities"]
                                .as_array()
                                .map(|p| std::array::from_fn(|i| p[i].as_f64().unwrap().ln())),
                        })
                        .collect();
                    super::super::ratings::mix_rating(&groups)
                }),
            })
            .collect();
        let coverage = std::array::from_fn(|h| case["coverage"][h].as_f64().unwrap());
        let priors = std::array::from_fn(|h| {
            std::array::from_fn(|i| case["priors"][h][i].as_f64().unwrap())
        });
        let temperatures = std::array::from_fn(|h| case["temperatures"][h].as_f64().unwrap());
        let actual = project(&rows, coverage, priors, temperatures).unwrap();
        let close = |a: f64, b: f64| assert!((a - b).abs() < 3e-12, "{}: {a} != {b}", case["name"]);
        for (h, head) in actual.heads.iter().enumerate() {
            let expected = &case["expected"]["heads"][h];
            close(head.supported_mass, expected["supported"].as_f64().unwrap());
            close(
                head.reported_support_mass,
                expected["reported"].as_f64().unwrap(),
            );
            for i in 0..5 {
                close(
                    head.log_probabilities[i].exp(),
                    expected["probabilities"][i].as_f64().unwrap(),
                );
            }
            assert!(head.supported_mass <= actual.retained_mass);
        }
        close(actual.retained_mass, 0.6);
        close(actual.original_unknown_mass, 0.3);
        close(actual.unsupported_projection_mass, 0.1);
        let without_heads: Vec<_> = rows
            .iter()
            .map(|r| Alternative {
                heads: [None; 3],
                ..*r
            })
            .collect();
        let interpretation = project(&without_heads, coverage, priors, temperatures).unwrap();
        assert_eq!(
            actual.conditional_weights,
            interpretation.conditional_weights
        );
        assert_eq!(actual.entropy, interpretation.entropy);
        assert_eq!(actual.uncertainty, interpretation.uncertainty);
    }
}

#[test]
fn hypothetical_mixtures_match_independent_decimal_reference() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/temporal_cognition/consequence.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let rows: Vec<_> = case["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| Alternative {
                weight: row["weight"].as_f64().unwrap(),
                delta_log_score: row["delta"].as_f64(),
                heads: std::array::from_fn(|h| {
                    row["heads"][h]
                        .as_array()
                        .map(|p| full_head(std::array::from_fn(|i| p[i].as_f64().unwrap().ln())))
                }),
            })
            .collect();
        let coverage = case["coverage"].as_f64().unwrap();
        let priors = std::array::from_fn(|h| {
            std::array::from_fn(|i| case["priors"][h][i].as_f64().unwrap())
        });
        let temperatures = std::array::from_fn(|h| case["temperatures"][h].as_f64().unwrap());
        let result = project(&rows, [coverage; 3], priors, temperatures).unwrap();
        let expected = &case["expected"];
        let close = |actual: f64, expected: f64| {
            let tolerance = expected.abs().max(1e-280) * 3e-12;
            assert!(
                (actual - expected).abs() <= tolerance,
                "{}: {actual} != {expected}",
                case["name"]
            );
        };
        close(result.retained_mass, expected["retained"].as_f64().unwrap());
        close(result.unknown_mass, expected["unknown"].as_f64().unwrap());
        close(result.entropy, expected["entropy"].as_f64().unwrap());
        close(
            result.uncertainty,
            expected["uncertainty"].as_f64().unwrap(),
        );
        assert_eq!(
            result.resolved_count,
            expected["count"].as_u64().unwrap() as usize
        );
        for (actual, expected) in result
            .conditional_weights
            .iter()
            .zip(expected["weights"].as_array().unwrap())
        {
            close(*actual, expected.as_f64().unwrap());
        }
        assert!(
            result.conditional_weights[rows.len()..]
                .iter()
                .all(|p| *p == 0.)
        );
        let absolute = result.absolute(Some(2.));
        assert_eq!(absolute.relation, Some(2.));
        for (i, head) in result.heads.iter().enumerate() {
            let reference = &expected["heads"][i];
            assert_eq!(head.observed_coverage, coverage);
            close(
                head.supported_mass,
                reference["supported"].as_f64().unwrap(),
            );
            close(
                head.reported_support_mass,
                reference["reported"].as_f64().unwrap(),
            );
            for (category, log_p) in head.log_probabilities.iter().enumerate() {
                close(
                    log_p.exp(),
                    reference["probabilities"][category].as_f64().unwrap(),
                );
            }
            if head.reported_support_mass > 0. {
                close(
                    absolute.ratings[i].unwrap(),
                    reference["expected"].as_f64().unwrap(),
                );
            } else {
                assert_eq!(absolute.ratings[i], None);
            }
        }
        assert_eq!(absolute.uncertainty.is_some(), result.retained_mass > 0.);
    }
}

#[test]
fn descriptor_projection_uses_real_body_masks_scales_and_exact_ties() {
    use crate::{config::TemporalBodyConfig, temporal_cognition::body::Record};
    let config = TemporalBodyConfig {
        means: [10.; 6],
        deviations: [2., 0., 4., 1., 2., 2.],
        accent_means: [0.; 2],
        accent_deviations: [1.; 2],
    };
    let record = Record {
        raw_values: [10.25, 10., 20., 30., 40., 50.],
        mask: 0b000011,
        ..Record::default()
    };
    let values = record.standardized(config);
    assert_eq!(values, [Some(0.125), Some(0.), None, None, None, None]);
    let candidates = [
        Descriptor {
            key: (1, 1),
            values: [Some(0.125), None, None, None, None, None],
        },
        Descriptor {
            key: (9, 2),
            values,
        },
        Descriptor {
            key: (9, 1),
            values,
        },
    ];
    let result = nearest(&values, &candidates, 0.25).unwrap().unwrap();
    assert_eq!(
        result,
        Assignment {
            key: (9, 1),
            distance: 0.,
            common_coordinates: 2
        }
    );
    let mut reversed = candidates;
    reversed.reverse();
    assert_eq!(nearest(&values, &reversed, 0.25).unwrap(), Some(result));
    let no_common = [Descriptor {
        key: (0, 1),
        values: [None, None, Some(10.), None, None, None],
    }];
    assert_eq!(nearest(&values, &no_common, 0.25).unwrap(), None);
    assert_eq!(nearest(&[None; 6], &reversed, 0.25).unwrap(), None);
    let from_zero = [Some(0.), None, None, None, None, None];
    let gate = Descriptor {
        key: (5, 1),
        values: [Some(0.25), None, None, None, None, None],
    };
    assert_eq!(
        nearest(&from_zero, &[gate], 0.25)
            .unwrap()
            .unwrap()
            .distance,
        0.25
    );
    let above = f64::from_bits(0.25_f64.to_bits() + 1);
    let outside = Descriptor {
        key: (0, 1),
        values: [Some(above), None, None, None, None, None],
    };
    assert_eq!(nearest(&from_zero, &[outside], 0.25).unwrap(), None);
    let close = [
        Descriptor {
            key: (0, 1),
            values: [Some(above), None, None, None, None, None],
        },
        Descriptor {
            key: (8, 1),
            values: [Some(0.25), None, None, None, None, None],
        },
    ];
    assert_eq!(
        nearest(&from_zero, &close, 1.).unwrap().unwrap().key,
        (8, 1)
    );
}

#[test]
fn whole_cell_lookup_preserves_unknown_cells_and_absolute_origin() {
    let cells = [Some("first"), None, Some("last")];
    for (target, expected) in [
        (99, None),
        (100, Some(0)),
        (115, Some(0)),
        (116, Some(1)),
        (130, Some(1)),
        (145, Some(1)),
        (146, Some(2)),
        (160, Some(2)),
        (161, None),
    ] {
        let index = cell_index(100, 160, cells.len(), target);
        assert_eq!(index, expected);
        if target == 145 {
            assert_eq!(cells[index.unwrap()], None);
        }
    }
    assert_eq!(cell_index(100, 160, 1, 100), None);
    assert_eq!(cell_index(100, 100, 32, 100), None);
    // A long-running sample clock must retain integer midpoint precision above 2^53.
    let origin = u64::MAX - 192_000;
    for offset in 0..=192_000 {
        let expected = (0..32)
            .min_by_key(|i| (i * 192_000_i128 - i128::from(offset) * 31).abs())
            .unwrap() as usize;
        assert_eq!(
            cell_index(origin, u64::MAX, 32, origin + offset),
            Some(expected)
        );
    }
    // Odd denominators have no integer midpoint: the first point above half goes later.
    assert_eq!(cell_index(0, 3, 2, 1), Some(0));
    assert_eq!(cell_index(0, 3, 2, 2), Some(1));
}

#[test]
fn signed_columns_require_paired_support_and_keep_known_zero() {
    let a = Absolute {
        relation: Some(2.),
        ratings: [Some(0.8), None, Some(0.4)],
        uncertainty: Some(0.2),
    };
    let b = Absolute {
        relation: Some(1.),
        ratings: [Some(0.3), Some(0.5), None],
        uncertainty: Some(0.6),
    };
    let diff = differences(&a, &b, 2.).unwrap();
    assert_eq!(diff.relation, Some(0.5_f64.tanh()));
    assert_eq!(diff.ratings, [Some(0.5), None, None]);
    assert!((diff.uncertainty.unwrap() + 0.4).abs() < 1e-15);
    let reverse = differences(&b, &a, 2.).unwrap();
    assert_eq!(reverse.relation, diff.relation.map(|x| -x));
    assert_eq!(reverse.ratings, diff.ratings.map(|x| x.map(|v| -v)));
    let same = differences(&a, &a, 2.).unwrap();
    assert_eq!(same.relation, Some(0.));
    assert_eq!(same.ratings, [Some(0.), None, Some(0.)]);
    assert_eq!(same.uncertainty, Some(0.));
    let unknown = Absolute {
        relation: None,
        ratings: [None; 3],
        uncertainty: None,
    };
    for (left, right) in [(&a, &unknown), (&unknown, &a), (&unknown, &unknown)] {
        assert_eq!(
            differences(left, right, 2.).unwrap(),
            Differences {
                relation: None,
                ratings: [None; 3],
                uncertainty: None,
            }
        );
    }
    assert_eq!(differences(&a, &b, 0.).unwrap().relation, Some(1.));
}

#[test]
fn invalid_inputs_and_capacity_are_rejected_without_partial_projection() {
    let prior = [[0.2; 5]; 3];
    let mut rows = vec![Alternative {
        weight: 0.8,
        delta_log_score: Some(0.),
        heads: [None; 3],
    }];
    assert!(project(&rows, [1.; 3], prior, [1.; 3]).is_some());
    for invalid in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        assert!(project(&rows, [invalid; 3], prior, [1.; 3]).is_none());
        rows[0].weight = invalid;
        assert!(project(&rows, [1.; 3], prior, [1.; 3]).is_none());
    }
    rows[0].weight = 0.8;
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        rows[0].delta_log_score = Some(invalid);
        assert!(project(&rows, [1.; 3], prior, [1.; 3]).is_none());
    }
    rows[0].delta_log_score = Some(0.);
    for invalid in [f64::NAN, f64::INFINITY, 0., -1.] {
        assert!(project(&rows, [1.; 3], prior, [invalid; 3]).is_none());
    }
    assert!(project(&rows, [1.; 3], [[0.1; 5]; 3], [1.; 3]).is_none());
    rows[0].heads[0] = Some(full_head([0.; 5]));
    assert!(project(&rows, [1.; 3], prior, [1.; 3]).is_none());
    for invalid in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        rows[0].heads[0] = Some(WeightedRating {
            weight: invalid,
            log_probabilities: Some([0.2_f64.ln(); 5]),
        });
        assert!(project(&rows, [1.; 3], prior, [1.; 3]).is_none());
    }
    rows[0].heads[0] = None;
    rows.push(Alternative {
        weight: 0.3,
        delta_log_score: None,
        heads: [None; 3],
    });
    assert!(project(&rows, [1.; 3], prior, [1.; 3]).is_none());
    let rows: Vec<_> = (0..129)
        .map(|_| Alternative {
            weight: 1. / 256.,
            delta_log_score: Some(0.),
            heads: [None; 3],
        })
        .collect();
    assert!(project(&rows[..128], [1.; 3], prior, [1.; 3]).is_some());
    assert!(project(&rows, [1.; 3], prior, [1.; 3]).is_none());
    assert!(nearest(&[Some(f64::NAN); 6], &[], 0.25).is_err());
    assert!(nearest(&[None; 6], &[], f64::NAN).is_err());
    let descriptors: Vec<_> = (0..9)
        .map(|id| Descriptor {
            key: (id, 1),
            values: [Some(0.); 6],
        })
        .collect();
    assert!(nearest(&[Some(0.); 6], &descriptors, 0.25).is_err());
    let a = Absolute {
        relation: Some(0.),
        ratings: [Some(0.); 3],
        uncertainty: Some(0.),
    };
    for invalid in [-1., f64::NAN, f64::INFINITY] {
        assert!(differences(&a, &a, invalid).is_none());
    }
}

#[test]
fn candidate_score_changes_cannot_recover_original_unknown_mass() {
    let prior = [[0.2; 5]; 3];
    let mut rows = [
        Alternative {
            weight: 0.2,
            delta_log_score: Some(0.),
            heads: [Some(full_head([0.2_f64.ln(); 5])); 3],
        },
        Alternative {
            weight: 0.3,
            delta_log_score: Some(0.),
            heads: [None; 3],
        },
        Alternative {
            weight: 0.4,
            delta_log_score: None,
            heads: [Some(full_head([0.2_f64.ln(); 5])); 3],
        },
    ];
    let initial = project(&rows, [0.7; 3], prior, [1.; 3]).unwrap();
    for score in [-1e308, -1000., 0., 1000., 1e308] {
        rows[0].delta_log_score = Some(score);
        let result = project(&rows, [0.7; 3], prior, [1.; 3]).unwrap();
        assert_eq!(result.retained_mass, 0.5);
        assert_eq!(result.unknown_mass, 0.5);
        assert_eq!(result.resolved_count, 2);
        assert!(result.uncertainty >= 0.5);
        assert!(result.heads[0].reported_support_mass <= 0.35 + 1e-15);
    }
    rows[0].delta_log_score = Some(0.);
    let repeated = project(&rows, [0.7; 3], prior, [1.; 3]).unwrap();
    assert_eq!(initial.conditional_weights, repeated.conditional_weights);
    assert_eq!(
        initial.heads[0].log_probabilities,
        repeated.heads[0].log_probabilities
    );
    assert_eq!(rows[2].delta_log_score, None);
}
