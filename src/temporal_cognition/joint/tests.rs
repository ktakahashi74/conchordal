use super::*;

fn pair(
    parent: u8,
    extension: u8,
    prior: f64,
    transition: f64,
    potential: f64,
    resolved: bool,
) -> Pair {
    Pair {
        parent,
        extension,
        log_prior: prior.ln(),
        log_transition: transition.ln(),
        log_potential: potential,
        resolved,
    }
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 3e-13, "{a} != {b}");
}

fn decode(value: &serde_json::Value) -> Pair {
    pair(
        value["parent"].as_u64().unwrap() as u8,
        value["extension"].as_u64().unwrap() as u8,
        value["prior"].as_f64().unwrap(),
        value["transition"].as_f64().unwrap(),
        value["potential"].as_f64().unwrap(),
        value["resolved"].as_bool().unwrap(),
    )
}

#[test]
fn complete_joint_enumeration_matches_decimal_and_preserves_partitions_before_pruning() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/temporal_cognition/joint.json"
    ))
    .unwrap();
    let mut normalizer = Normalizer::new();
    for case in fixture["cases"].as_array().unwrap() {
        let inputs = case["shared"].as_array().unwrap();
        let groups: Vec<Vec<Vec<Pair>>> = inputs
            .iter()
            .map(|c| {
                c["groups"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|g| g.as_array().unwrap().iter().map(decode).collect())
                    .collect()
            })
            .collect();
        let refs: Vec<Vec<&[Pair]>> = groups
            .iter()
            .map(|c| c.iter().map(Vec::as_slice).collect())
            .collect();
        let shared: Vec<_> = inputs
            .iter()
            .zip(&refs)
            .map(|(c, g)| Shared {
                pair: decode(&c["pair"]),
                groups: g,
            })
            .collect();
        let result = normalizer
            .normalize(&shared, case["observed"].as_bool().unwrap())
            .unwrap();
        let expected = &case["expected"];
        close(
            result.log_evidence.unwrap(),
            expected["log_evidence"].as_f64().unwrap(),
        );
        close(
            result.explicit_unknown,
            expected["explicit_unknown"].as_f64().unwrap(),
        );
        close(
            result.pruned_mass,
            expected["pruned_mass"].as_f64().unwrap(),
        );
        assert_eq!(
            result.pruned_count,
            expected["pruned_count"].as_u64().unwrap() as usize
        );
        assert_eq!(
            result.contexts.iter().flatten().count(),
            expected["rows"].as_array().unwrap().len()
        );
        close(
            result
                .contexts
                .iter()
                .flatten()
                .map(|r| r.weight.mass)
                .sum::<f64>()
                + result.explicit_unknown
                + result.pruned_mass,
            1.,
        );
        for (actual, expected) in result
            .contexts
            .iter()
            .flatten()
            .zip(expected["rows"].as_array().unwrap())
        {
            assert_eq!(
                (actual.weight.parent, actual.weight.extension),
                (
                    expected["parent"].as_u64().unwrap() as u8,
                    expected["extension"].as_u64().unwrap() as u8
                )
            );
            close(actual.weight.mass, expected["mass"].as_f64().unwrap());
            for (local, wanted) in actual
                .groups
                .iter()
                .zip(expected["groups"].as_array().unwrap())
            {
                close(
                    local.explicit_unknown,
                    wanted["explicit_unknown"].as_f64().unwrap(),
                );
                close(local.pruned_mass, wanted["pruned_mass"].as_f64().unwrap());
                assert_eq!(
                    local.pruned_count,
                    wanted["pruned_count"].as_u64().unwrap() as usize
                );
                assert_eq!(
                    local.rows.iter().flatten().count(),
                    wanted["rows"].as_array().unwrap().len()
                );
                close(
                    local.rows.iter().flatten().map(|r| r.mass).sum::<f64>()
                        + local.explicit_unknown
                        + local.pruned_mass,
                    1.,
                );
                for (a, b) in local
                    .rows
                    .iter()
                    .flatten()
                    .zip(wanted["rows"].as_array().unwrap())
                {
                    assert_eq!(
                        (a.parent, a.extension),
                        (
                            b["parent"].as_u64().unwrap() as u8,
                            b["extension"].as_u64().unwrap() as u8
                        )
                    );
                    close(a.mass, b["mass"].as_f64().unwrap());
                }
            }
            assert!(
                actual.groups[result.group_count..]
                    .iter()
                    .all(|g| *g == Local::default())
            );
        }
    }
}

#[test]
fn empty_group_inventory_initializes_and_retires_without_fabricating_local_paths() {
    let initial = [Shared {
        pair: pair(0, 0, 1., 1., 0., false),
        groups: &[],
    }];
    let mut engine = Normalizer::new();
    let allocation = engine.groups.as_ptr();
    let unresolved = *engine.normalize(&initial, false).unwrap();
    assert_eq!(unresolved.group_count, 0);
    assert_eq!(unresolved.shared_enumerated, 1);
    assert_eq!(unresolved.local_enumerated, 0);
    close(unresolved.explicit_unknown, 1.);
    close(unresolved.log_evidence.unwrap(), 0.);
    assert!(unresolved.contexts.iter().all(Option::is_none));

    let local = [
        pair(0, 0, 1., 0.6, 0., true),
        pair(0, 1, 1., 0.4, 0., false),
    ];
    let groups = [&local[..]; 8];
    for count in [0, 8, 0, 1, 0] {
        let shared = [
            Shared {
                pair: pair(0, 0, 1., 0.75, 0., true),
                groups: &groups[..count],
            },
            Shared {
                pair: pair(0, 1, 1., 0.25, 0., false),
                groups: &groups[..count],
            },
        ];
        let result = engine.normalize(&shared, false).unwrap();
        assert_eq!(result.group_count, count);
        assert_eq!(result.local_enumerated, 4 * count);
        close(result.explicit_unknown, 0.25);
        close(result.log_evidence.unwrap(), 0.);
        let context = result.contexts[0].unwrap();
        close(context.weight.mass, 0.75);
        for group in &context.groups[..count] {
            close(group.rows[0].unwrap().mass, 0.6);
            close(group.explicit_unknown, 0.4);
        }
        assert!(
            context.groups[count..]
                .iter()
                .all(|g| *g == Local::default())
        );
        assert!(
            result
                .preview(count, &[], [1.; 3], [[0.2; 5]; 3], [1.; 3])
                .is_none()
        );
        assert_eq!(engine.groups.as_ptr(), allocation);
    }
    assert_eq!(*engine.normalize(&initial, false).unwrap(), unresolved);
}

#[test]
fn zero_groups_do_not_admit_empty_shared_or_mismatched_local_inventories() {
    let local = [pair(0, 0, 1., 1., 0., false)];
    let groups = [&local[..]; 9];
    let mut shared = [
        Shared {
            pair: pair(0, 0, 1., 0.8, 0., true),
            groups: &[],
        },
        Shared {
            pair: pair(0, 1, 1., 0.2, 0., false),
            groups: &[],
        },
    ];
    let mut engine = Normalizer::new();
    let before = *engine.normalize(&shared, true).unwrap();
    assert!(engine.normalize(&[], true).is_err());
    assert_eq!(engine.output, before);
    for invalid in [&groups[..1], &groups[..]] {
        shared[1].groups = invalid;
        assert!(engine.normalize(&shared, true).is_err());
        assert_eq!(engine.output, before);
    }
    shared[0].groups = &groups;
    assert!(engine.normalize(&shared, true).is_err());
    assert_eq!(engine.output, before);
    let empty_local: &[Pair] = &[];
    let empty_group = [empty_local];
    for candidate in &mut shared {
        candidate.groups = &empty_group;
    }
    assert!(engine.normalize(&shared, true).is_err());
    assert_eq!(engine.output, before);
    for candidate in &mut shared {
        candidate.groups = &[];
    }
    assert_eq!(*engine.normalize(&shared, true).unwrap(), before);
}

#[test]
fn full_caps_keep_enumerated_mass_and_stable_ties_without_growing_scratch() {
    let local: Vec<_> = (0..16)
        .flat_map(|p| (0..16).map(move |e| pair(p, e, 1. / 16., 1. / 16., 0., e != 15)))
        .collect();
    let groups = [local.as_slice(); 8];
    let groups = &groups;
    let shared: Vec<_> = (0..8)
        .flat_map(|p| {
            (0..4).map(move |e| Shared {
                pair: pair(p, e, 1. / 8., 1. / 4., 0., e != 3),
                groups,
            })
        })
        .collect();
    let mut engine = Normalizer::new();
    let allocation = engine.groups.as_ptr();
    let output = *engine.normalize(&shared, true).unwrap();
    assert_eq!(
        (output.shared_enumerated, output.local_enumerated),
        (32, 65_536)
    );
    assert_eq!(output.pruned_count, 17);
    close(output.explicit_unknown, 0.25);
    close(output.pruned_mass, 17. / 32.);
    for context in output.contexts.iter().flatten() {
        close(context.weight.mass, 1. / 32.);
        for group in &context.groups {
            close(group.explicit_unknown, 1. / 16.);
            close(group.pruned_mass, 225. / 256.);
            assert_eq!(group.pruned_count, 225);
            for (i, w) in group.rows.iter().flatten().enumerate() {
                assert_eq!((w.parent, w.extension), (0, i as u8));
                close(w.mass, 1. / 256.);
            }
        }
    }
    let again = *engine.normalize(&shared, false).unwrap();
    assert_eq!(output, again);
    assert_eq!(engine.groups.as_ptr(), allocation);
}

#[test]
fn invalid_parent_transition_inventory_is_atomic_and_missing_input_suppresses_potentials() {
    let local = [
        pair(0, 0, 1., 0.7, 3., true),
        pair(0, 1, 1., 0.3, -2., false),
    ];
    let groups = [&local[..]];
    let mut shared = [
        Shared {
            pair: pair(0, 0, 1., 0.8, 4., true),
            groups: &groups,
        },
        Shared {
            pair: pair(0, 1, 1., 0.2, -1., false),
            groups: &groups,
        },
    ];
    let mut engine = Normalizer::new();
    let missing = *engine.normalize(&shared, false).unwrap();
    close(missing.contexts[0].unwrap().weight.mass, 0.8);
    close(
        missing.contexts[0].unwrap().groups[0].rows[0].unwrap().mass,
        0.7,
    );
    close(missing.log_evidence.unwrap(), 0.);
    let observed = *engine.normalize(&shared, true).unwrap();
    assert!(observed.contexts[0].unwrap().weight.mass > 0.99);
    for fault in 0..5 {
        let saved = shared[1].pair;
        match fault {
            0 => shared[1].pair.extension = 0,
            1 => shared[1].pair.log_prior = 0.5_f64.ln(),
            2 => shared[1].pair.log_transition = 0.3_f64.ln(),
            3 => shared[1].pair.resolved = true,
            _ => shared[1].pair.log_potential = f64::NAN,
        }
        assert!(engine.normalize(&shared, true).is_err());
        assert_eq!(engine.output, observed);
        shared[1].pair = saved;
    }
}

#[test]
fn common_extreme_potentials_do_not_erase_priors_and_zero_mass_cannot_dominate() {
    let local = [
        pair(0, 0, 1., 0.7, 1e300, true),
        pair(0, 1, 1., 0.3, 1e300, false),
        pair(1, 0, 0., 1., f64::MAX, false),
    ];
    let groups = [&local[..]];
    let poison = [pair(0, 0, 1., 1., f64::MAX, false)];
    let poison_groups = [&poison[..]];
    let shared = [
        Shared {
            pair: pair(0, 0, 1., 0.8, 1e300, true),
            groups: &groups,
        },
        Shared {
            pair: pair(0, 1, 1., 0.2, 1e300, false),
            groups: &groups,
        },
        Shared {
            pair: pair(1, 0, 0., 1., f64::MAX, false),
            groups: &poison_groups,
        },
    ];
    let mut engine = Normalizer::new();
    let output = engine.normalize(&shared, true).unwrap();
    close(output.contexts[0].unwrap().weight.mass, 0.8);
    close(
        output.contexts[0].unwrap().groups[0].rows[0].unwrap().mass,
        0.7,
    );
    close(output.explicit_unknown, 0.2);
}

#[test]
fn all_unknown_local_and_shared_paths_keep_mass_without_exposing_resolved_identities() {
    let unknown = [pair(0, 0, 1., 1., 0., false)];
    let groups = [&unknown[..]];
    let mut input = [
        Shared {
            pair: pair(0, 0, 1., 0.9, 0., true),
            groups: &groups,
        },
        Shared {
            pair: pair(0, 1, 1., 0.1, 0., false),
            groups: &groups,
        },
    ];
    let mut engine = Normalizer::new();
    let output = engine.normalize(&input, true).unwrap();
    close(output.contexts[0].unwrap().groups[0].explicit_unknown, 1.);
    let preview = output
        .preview(0, &[], [1.; 3], [[0.2; 5]; 3], [1.; 3])
        .unwrap();
    close(preview.original_unknown_mass, 1.);
    close(preview.unsupported_projection_mass, 0.);
    close(preview.uncertainty, 1.);
    input[0].pair.log_transition = f64::NEG_INFINITY;
    input[1].pair.log_transition = 0.;
    let output = engine.normalize(&input, true).unwrap();
    assert!(output.contexts.iter().all(Option::is_none));
    close(output.explicit_unknown, 1.);
    close(output.pruned_mass, 0.);
    let before = *output;
    let bad_local = [
        pair(0, 0, 1., 0.9, 0., true),
        pair(0, 1, 1., 0.2, 0., false),
    ];
    let bad_groups = [&bad_local[..]];
    input[1].groups = &bad_groups;
    assert!(engine.normalize(&input, true).is_err());
    assert_eq!(engine.output, before);
}

#[test]
fn candidate_preview_uses_joint_mass_and_separates_original_from_projection_unknown() {
    let local = [
        pair(0, 0, 1., 0.25, 0., true),
        pair(0, 1, 1., 0.5, 0., true),
        pair(0, 2, 1., 0.25, 0., false),
    ];
    let groups = [&local[..]];
    let shared = [
        Shared {
            pair: pair(0, 0, 1., 0.6, 0., true),
            groups: &groups,
        },
        Shared {
            pair: pair(0, 1, 1., 0.3, 0., true),
            groups: &groups,
        },
        Shared {
            pair: pair(0, 2, 1., 0.1, 0., false),
            groups: &groups,
        },
    ];
    let mut engine = Normalizer::new();
    let posterior = *engine.normalize(&shared, true).unwrap();
    let a = [1_f64, 0., 0., 0., 0.].map(f64::ln);
    let b = [0_f64, 0., 0., 0., 1.].map(f64::ln);
    let mut candidates = [
        Candidate {
            shared_pair: (0, 0),
            local_pair: (0, 0),
            delta_log_score: Some(2_f64.ln()),
            heads: [Some(super::super::ratings::WeightedRating {
                weight: 1.,
                log_probabilities: Some(a),
            }); 3],
        },
        Candidate {
            shared_pair: (0, 1),
            local_pair: (0, 1),
            delta_log_score: Some(0.),
            heads: [Some(super::super::ratings::WeightedRating {
                weight: 1.,
                log_probabilities: Some(b),
            }); 3],
        },
    ];
    let projected = posterior
        .preview(0, &candidates, [0.8, 0.5, 0.], [[0.2; 5]; 3], [1.; 3])
        .unwrap();
    close(projected.original_unknown_mass, 0.325);
    close(projected.unsupported_projection_mass, 0.375);
    close(projected.unknown_mass, 0.7);
    close(projected.retained_mass, 0.3);
    assert_eq!(projected.resolved_count, 2);
    let entropy = -(2. / 3.) * (2_f64 / 3.).ln() - (1. / 3.) * (1_f64 / 3.).ln();
    close(projected.entropy, entropy);
    close(projected.uncertainty, 0.7 + 0.3 * entropy / 2_f64.ln());
    for (index, head) in projected.heads.into_iter().enumerate() {
        close(head.reported_support_mass, [0.24, 0.15, 0.][index]);
        let expected = [
            [0.312, 0.152, 0.152, 0.152, 0.232],
            [0.27, 0.17, 0.17, 0.17, 0.22],
            [0.2; 5],
        ][index];
        for (value, expected) in head
            .log_probabilities
            .map(f64::exp)
            .into_iter()
            .zip(expected)
        {
            close(value, expected);
        }
    }
    let unknown = posterior
        .preview(0, &[], [0.8; 3], [[0.2; 5]; 3], [2.; 3])
        .unwrap();
    close(unknown.original_unknown_mass, 0.325);
    close(unknown.unsupported_projection_mass, 0.675);
    close(unknown.unknown_mass, 1.);
    assert_eq!(posterior, engine.output);
    candidates[0].heads[0].as_mut().unwrap().weight = 0.4;
    candidates[1].heads[0].as_mut().unwrap().weight = 0.7;
    candidates[0].heads[1] = None;
    candidates[1].heads[1].as_mut().unwrap().weight = 0.25;
    candidates[0].heads[2] = None;
    candidates[1].heads[2] = None;
    let partial = posterior
        .preview(0, &candidates, [0.8, 0.5, 0.], [[0.2; 5]; 3], [1.; 3])
        .unwrap();
    close(partial.heads[0].supported_mass, 0.15);
    close(partial.heads[1].supported_mass, 0.025);
    close(partial.heads[2].supported_mass, 0.);
    close(partial.heads[0].reported_support_mass, 0.12);
    close(partial.heads[1].reported_support_mass, 0.0125);
    close(partial.unknown_mass, projected.unknown_mass);
    close(partial.entropy, projected.entropy);
    candidates[1].shared_pair = (0, 2);
    assert!(
        posterior
            .preview(0, &candidates, [0.8; 3], [[0.2; 5]; 3], [1.; 3])
            .is_none()
    );
    candidates[1].shared_pair = (0, 0);
    candidates[1].local_pair = (0, 0);
    assert!(
        posterior
            .preview(0, &candidates, [0.8; 3], [[0.2; 5]; 3], [1.; 3])
            .is_none()
    );
    assert!(
        posterior
            .preview(1, &[], [0.8; 3], [[0.2; 5]; 3], [1.; 3])
            .is_none()
    );
}

#[test]
fn pruned_identity_is_not_recovered_by_later_prediction_and_bias_is_explicit() {
    let diffuse: Vec<_> = (0..16)
        .flat_map(|p| {
            [
                pair(p, 0, 1. / 16., 0.98, 0., true),
                pair(p, 1, 1. / 16., 0.02, 0., false),
            ]
        })
        .collect();
    let focused = [
        pair(0, 0, 1., 0.98, 0., true),
        pair(0, 1, 1., 0.02, 0., false),
    ];
    let a = [&diffuse[..]];
    let b = [&focused[..]];
    let initial = [
        Shared {
            pair: pair(0, 0, 1., 0.45, 0., true),
            groups: &a,
        },
        Shared {
            pair: pair(0, 1, 1., 0.45, 0., true),
            groups: &b,
        },
        Shared {
            pair: pair(0, 2, 1., 0.1, 0., false),
            groups: &b,
        },
    ];
    let mut engine = Normalizer::new();
    let first = *engine.normalize(&initial, true).unwrap();
    let local = first.contexts[0].unwrap().groups[0];
    close(local.pruned_mass, 0.98 / 16.);
    assert!(local.rows.iter().flatten().all(|w| w.parent != 15));
    let mut retained: Vec<_> = local
        .rows
        .iter()
        .flatten()
        .flat_map(|w| {
            [
                pair(w.parent, 0, w.mass, 1., 0., true),
                pair(w.parent, 1, w.mass, 0., 0., false),
            ]
        })
        .collect();
    retained.push(pair(
        15,
        0,
        local.explicit_unknown + local.pruned_mass,
        1.,
        0.,
        false,
    ));
    let held = [&retained[..]];
    let next = [
        Shared {
            pair: pair(0, 0, 0.45, 1., 0., true),
            groups: &held,
        },
        Shared {
            pair: pair(0, 1, 0.45, 0., 0., false),
            groups: &held,
        },
        Shared {
            pair: pair(1, 0, 0.45, 1., 0., true),
            groups: &b,
        },
        Shared {
            pair: pair(1, 1, 0.45, 0., 0., false),
            groups: &b,
        },
        Shared {
            pair: pair(2, 0, 0.1, 1., 0., false),
            groups: &b,
        },
    ];
    let second = engine.normalize(&next, true).unwrap();
    close(second.contexts[0].unwrap().weight.mass, 0.45);
    let exhaustive_evidence = 1. + (0.98 / 16.) * 7.;
    let exhaustive_context = 0.45 * exhaustive_evidence / (0.45 * exhaustive_evidence + 0.55);
    assert!(exhaustive_context - second.contexts[0].unwrap().weight.mass > 0.08);
    assert!(
        second.contexts[0].unwrap().groups[0]
            .rows
            .iter()
            .flatten()
            .all(|w| w.parent != 15)
    );
}
