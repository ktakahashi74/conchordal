use super::*;
use crate::temporal_cognition::joint::{Normalizer, Pair, Shared};

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 3e-13, "{a} != {b}");
}

#[test]
fn slots_and_leak_match_decimal_reference_including_missing_and_duplicate_support() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/temporal_cognition/shared_proposals.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let inputs: [Option<Known>; 3] = std::array::from_fn(|i| {
            let row = &case["inputs"][i];
            row["path"].as_u64().map(|path| Known {
                path,
                raw_score: row["raw_score"].as_f64().unwrap(),
            })
        });
        let output = build(
            case["parent"].as_u64(),
            inputs[0],
            inputs[1],
            inputs[2],
            case["dt"].as_f64().unwrap(),
            case["observed"].as_bool().unwrap(),
        )
        .unwrap();
        close(
            output
                .iter()
                .flatten()
                .map(|p| p.log_transition.exp())
                .sum(),
            1.,
        );
        for (actual, expected) in output.iter().zip(case["expected"].as_array().unwrap()) {
            assert_eq!(actual.is_none(), expected.is_null(), "{}", case["name"]);
            if let Some(actual) = actual {
                assert_eq!(actual.path, expected["path"].as_u64());
                close(
                    actual.log_transition.exp(),
                    expected["probability"].as_f64().unwrap(),
                );
            }
        }
    }
}

#[test]
fn malformed_clock_score_and_unknown_stay_are_rejected_before_masking() {
    let known = Known {
        path: 7,
        raw_score: 1.,
    };
    for dt in [f64::NAN, f64::INFINITY, -1.] {
        assert!(build(Some(7), Some(known), None, None, dt, true).is_err());
    }
    for raw_score in [-1., f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let invalid = Some(Known { raw_score, ..known });
        for observed in [true, false] {
            assert!(build(Some(7), invalid, None, None, 0.01, observed).is_err());
            assert!(build(None, None, invalid, None, 0.01, observed).is_err());
            assert!(build(None, None, None, invalid, 0.01, observed).is_err());
        }
    }
    assert!(build(None, Some(known), None, None, 0.01, true).is_err());
    assert!(build(Some(8), Some(known), None, None, 0.01, true).is_err());
}

#[test]
fn gap_transitions_preserve_elapsed_time_and_unknown_needs_new_support() {
    for duration in [2., 8., 32., 60.] {
        for steps in [1, 2, 1000] {
            let dt = duration / f64::from(steps);
            let mut mass = 1.;
            for _ in 0..steps {
                let proposals = build(
                    Some(7),
                    Some(Known {
                        path: 7,
                        raw_score: 1.,
                    }),
                    None,
                    None,
                    dt,
                    false,
                )
                .unwrap();
                mass *= proposals[0].unwrap().log_transition.exp();
                let unknown = build(
                    None,
                    None,
                    Some(Known {
                        path: 99,
                        raw_score: 1.,
                    }),
                    None,
                    dt,
                    false,
                )
                .unwrap();
                assert!(unknown[0].is_none() && unknown[2].is_none() && unknown[3].is_none());
                assert_eq!(unknown[1].unwrap().log_transition, 0.);
            }
            close(mass, (-duration / 120.).exp());
        }
    }
    let recovered = build(
        None,
        None,
        Some(Known {
            path: 101,
            raw_score: 1.,
        }),
        None,
        0.01,
        true,
    )
    .unwrap();
    assert_eq!(recovered[2].unwrap().path, Some(101));
    assert!(recovered[0].is_none());
}

#[test]
fn all_parent_slots_feed_joint_normalization_and_keep_identity_loss_separate() {
    let dt: f64 = 0.125;
    let keep = (-dt / 120.).exp();
    let inputs: [[Option<Choice>; 4]; 8] = std::array::from_fn(|p| {
        let parent = (p < 7).then_some(p as u64);
        build(
            parent,
            parent.map(|path| Known {
                path,
                raw_score: 1.,
            }),
            Some(Known {
                path: 100 + p as u64,
                raw_score: 2.,
            }),
            Some(Known {
                path: 200 + p as u64,
                raw_score: 3.,
            }),
            dt,
            true,
        )
        .unwrap()
    });
    let local = [Pair {
        parent: 0,
        extension: 0,
        resolved: false,
        log_prior: 0.,
        log_transition: 0.,
        log_potential: 0.,
    }];
    let groups: [&[Pair]; 8] = [&local; 8];
    let groups = &groups;
    let shared: Vec<_> = inputs
        .iter()
        .enumerate()
        .flat_map(|(p, rows)| {
            rows.iter().enumerate().filter_map(move |(slot, choice)| {
                choice.map(|choice| Shared {
                    pair: Pair {
                        parent: p as u8,
                        extension: slot as u8,
                        resolved: choice.path.is_some(),
                        log_prior: (1_f64 / 8.).ln(),
                        log_transition: choice.log_transition,
                        log_potential: choice
                            .path
                            .map_or(0., |path| if path >= 200 { 2_f64.ln() } else { 0. }),
                    },
                    groups,
                })
            })
        })
        .collect();
    assert_eq!(shared.len(), 31);
    let mut normalizer = Normalizer::new();
    for observed in [false, true] {
        let result = normalizer.normalize(&shared, observed).unwrap();
        let mut exhaustive: Vec<_> = shared
            .iter()
            .map(|s| {
                let path = inputs[usize::from(s.pair.parent)][usize::from(s.pair.extension)]
                    .unwrap()
                    .path;
                let raw = match s.pair.extension {
                    0 => 1.,
                    2 => 2.,
                    3 => 3.,
                    _ => 0.,
                };
                let transition = if path.is_none() {
                    1. - keep
                } else {
                    keep * raw / if s.pair.parent == 7 { 5. } else { 6. }
                };
                let likelihood = if observed && path.is_some_and(|p| p >= 200) {
                    2.
                } else {
                    1.
                };
                (
                    s.pair.parent,
                    s.pair.extension,
                    path,
                    transition * likelihood / 8.,
                )
            })
            .collect();
        let total: f64 = exhaustive.iter().map(|r| r.3).sum();
        for row in &mut exhaustive {
            row.3 /= total;
        }
        let unknown: f64 = exhaustive
            .iter()
            .filter(|r| r.2.is_none())
            .map(|r| r.3)
            .sum();
        exhaustive.retain(|r| r.2.is_some());
        exhaustive.sort_by(|a, b| b.3.total_cmp(&a.3).then((a.0, a.1).cmp(&(b.0, b.1))));
        close(result.log_evidence.unwrap(), total.ln());
        close(result.explicit_unknown, unknown);
        close(
            result.pruned_mass,
            exhaustive[7..].iter().map(|r| r.3).sum(),
        );
        assert_eq!(result.pruned_count, 16);
        for (actual, expected) in result.contexts.iter().flatten().zip(&exhaustive) {
            assert_eq!(
                (actual.weight.parent, actual.weight.extension),
                (expected.0, expected.1)
            );
            close(actual.weight.mass, expected.3);
            assert_eq!(
                inputs[usize::from(actual.weight.parent)][usize::from(actual.weight.extension)]
                    .unwrap()
                    .path,
                expected.2
            );
        }
        close(
            result
                .contexts
                .iter()
                .flatten()
                .map(|c| c.weight.mass)
                .sum::<f64>()
                + result.explicit_unknown
                + result.pruned_mass,
            1.,
        );
        // The aggregate parent has mass but cannot supply any discarded path as stay.
        assert!(
            build(
                None,
                Some(Known {
                    path: exhaustive[7].2.unwrap(),
                    raw_score: 1.
                }),
                None,
                None,
                dt,
                true
            )
            .is_err()
        );
    }
}
