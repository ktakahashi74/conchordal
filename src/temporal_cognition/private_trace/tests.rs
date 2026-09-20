use super::*;
use serde_json::Value;

#[test]
fn paired_fit_matches_exact_hinge_integrals_without_redeeming_missing_mass() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/temporal_cognition/private_fit.json"
    ))
    .unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 64);
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let mut reference = Reference {
            key: Key {
                epoch: 1,
                episode: 1,
                generation: 1,
                family: if case["periodic"].as_bool().unwrap() {
                    Family::Periodic
                } else {
                    Family::Nonperiodic
                },
            },
            weight: case["weight"].as_f64().unwrap(),
            anchors: [Anchor::default(); 7],
        };
        for (out, input) in reference
            .anchors
            .iter_mut()
            .zip(case["anchors"].as_array().unwrap())
        {
            out.interval = Some([
                input["interval"][0].as_f64().unwrap(),
                input["interval"][1].as_f64().unwrap(),
            ]);
            out.period = input["period"].as_f64().unwrap();
            out.weight = input["weight"].as_f64().unwrap();
        }
        let masses = std::array::from_fn(|i| case["masses"][i].as_f64().unwrap());
        let c = case["candidate"].as_f64().unwrap();
        let d = case["default"].as_f64().unwrap();
        let actual = paired_lookup(&reference, &masses, c, d);
        for (key, value) in [
            ("candidate", actual.candidate),
            ("default", actual.body_default),
            ("support", actual.support),
        ] {
            let expected = case["expected"][key].as_f64().unwrap();
            assert!(
                (value - expected).abs() < 2e-11,
                "case {index} {key}: {value} != {expected}"
            );
        }
        let reverse = paired_lookup(&reference, &masses, d, c);
        assert_eq!(actual.support, reverse.support);
        assert_eq!(actual.candidate, reverse.body_default);
        assert_eq!(actual.body_default, reverse.candidate);
        assert!(actual.support <= reference.weight);
        assert!(actual.candidate <= actual.support && actual.body_default <= actual.support);
    }
}

fn number(value: &Value) -> f64 {
    f64::from_bits(value["f64"].as_u64().unwrap())
}

fn close(left: f64, right: f64) {
    assert!(
        left == right || (left - right).abs() <= 2e-11 * left.abs().max(right.abs()).max(1.),
        "{left:?} != {right:?}"
    );
}

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/temporal_cognition/private_trace.json"
    ))
    .unwrap()
}

fn key(value: &Value) -> Key {
    Key {
        epoch: value[0].as_u64().unwrap(),
        episode: value[1][0].as_u64().unwrap(),
        generation: value[1][1].as_u64().unwrap(),
        family: if value[2] == "periodic" {
            Family::Periodic
        } else {
            Family::Nonperiodic
        },
    }
}

fn reference(value: &Value) -> Reference {
    let mut result = Reference {
        key: key(&value["key"]),
        weight: number(&value["weight"]),
        anchors: [Anchor::default(); 7],
    };
    for (out, input) in result
        .anchors
        .iter_mut()
        .zip(value["anchors"].as_array().unwrap())
    {
        out.interval = input["interval"]
            .as_array()
            .map(|p| [number(&p[0]), number(&p[1])]);
        out.period = number(&input["period_sec"]);
        out.weight = number(&input["weight"]);
    }
    result
}

fn parameters() -> Parameters {
    Parameters {
        tau: 20.,
        kappa: 0.7,
        strength_max: 1.2,
        interference: true,
    }
}

#[test]
fn timing_support_and_retained_moments_match_independent_python_oracle() {
    let fixture = fixture();
    assert_eq!(fixture["timings"].as_array().unwrap().len(), 60);
    for case in fixture["timings"].as_array().unwrap() {
        let periodic = case["periodic"].as_bool().unwrap();
        let mut row = case.clone();
        row["key"] =
            serde_json::json!([1, [1, 1], if periodic { "periodic" } else { "nonperiodic" }]);
        row["weight"] = serde_json::json!({"f64": 1_f64.to_bits()});
        let reference = reference(&row);
        let interval = [number(&case["interval"][0]), number(&case["interval"][1])];
        let (support, retained, coverage) = timing(interval, &reference, number(&case["tau"]));
        close(1. - coverage, number(&case["support"]["unsupported"]));
        for (actual, expected) in support
            .iter()
            .zip(case["support"]["bins"].as_array().unwrap())
        {
            close(*actual, number(expected));
        }
        for (actual, expected) in retained
            .iter()
            .zip(case["retained"]["bins"].as_array().unwrap())
        {
            close(*actual, number(expected));
        }
        close(support.iter().sum(), coverage);
        assert!(retained.iter().sum::<f64>() <= coverage + 1e-12);
        if periodic {
            assert_eq!(support[BINS], 0.);
            assert_eq!(retained[BINS], 0.);
        }
    }
}

#[test]
fn stateful_filter_matches_all_native_oracle_transitions_without_growing_storage() {
    let fixture = fixture();
    let mut count = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let mut config = parameters();
        config.strength_max = number(&case["strength"]);
        let mut bank = Bank::new(1, config, case["capacity"].as_u64().unwrap() as usize).unwrap();
        let storage = bank.traces.as_ptr();
        for op in case["operations"].as_array().unwrap() {
            let head = if op["head"] == "onset" {
                Head::Onset
            } else {
                Head::Release
            };
            for before in op["before"].as_array().unwrap() {
                let actual = bank.probabilities(key(&before["key"]), head);
                if before["probabilities"].is_null() {
                    assert!(actual.is_none());
                } else {
                    for (actual, expected) in actual
                        .unwrap()
                        .iter()
                        .zip(before["probabilities"].as_array().unwrap())
                    {
                        close(*actual, number(expected));
                    }
                }
            }
            let references: Vec<_> = op["entries"]
                .as_array()
                .unwrap()
                .iter()
                .map(reference)
                .collect();
            let retained: Vec<_> = op["retained"].as_array().unwrap().iter().map(key).collect();
            let receipt = bank
                .observe(Outcome {
                    id: op["id"].as_u64().unwrap(),
                    head,
                    interval: Some([number(&op["interval"][0]), number(&op["interval"][1])]),
                    references: &references,
                    retained: &retained,
                    observed_fraction: number(&op["observed_fraction"]),
                    confirmed: op["confirmed"].as_bool().unwrap(),
                })
                .unwrap();
            assert_eq!(receipt.applied, op["applied"].as_bool().unwrap());
            for (actual, expected) in receipt
                .credits
                .iter()
                .zip(op["credits"].as_array().unwrap())
            {
                close(*actual, number(expected));
            }
            close(receipt.unassigned, number(&op["unassigned"]));
            for (actual, name) in [(receipt.removed, "removed"), (receipt.evicted, "evicted")] {
                let actual: Vec<_> = actual.into_iter().flatten().collect();
                let expected: Vec<_> = op[name].as_array().unwrap().iter().map(key).collect();
                assert_eq!(actual, expected);
            }
            assert_eq!(bank.traces.len(), op["traces"].as_array().unwrap().len());
            for expected in op["traces"].as_array().unwrap() {
                let trace = bank
                    .traces
                    .iter()
                    .find(|t| t.key == key(&expected["key"]))
                    .unwrap();
                close(trace.end, number(&expected["end"]));
                for (head, name) in [(0, "onset"), (1, "release")] {
                    for (actual, expected) in trace.mass[head]
                        .iter()
                        .zip(expected[name].as_array().unwrap())
                    {
                        close(*actual, number(expected));
                    }
                }
            }
            assert_eq!(bank.traces.as_ptr(), storage);
            count += 1;
        }
    }
    assert_eq!(count, 112);
}

#[test]
fn lookup_preserves_periodic_modes_and_linear_overflow_normalization() {
    for case in fixture()["lookups"].as_array().unwrap() {
        let key = Key {
            epoch: 1,
            episode: 1,
            generation: 1,
            family: if case["periodic"] == true {
                Family::Periodic
            } else {
                Family::Nonperiodic
            },
        };
        let mut bank = Bank::new(1, parameters(), 16).unwrap();
        let mut trace = Trace {
            key,
            mass: [[f64::NEG_INFINITY; COORDINATES]; 2],
            end: 0.,
        };
        for (out, value) in trace.mass[0]
            .iter_mut()
            .zip(case["masses"].as_array().unwrap())
        {
            *out = number(value).ln();
        }
        bank.traces.push(trace);
        let actual = bank.lookup(key, Head::Onset, number(&case["position"]));
        if case["expected"].is_null() {
            assert!(actual.is_none());
        } else {
            close(actual.unwrap(), number(&case["expected"]));
        }
        assert!(bank.lookup(key, Head::Release, 0.).is_none());
    }
}

#[test]
fn duplicate_reversed_and_invalid_outcomes_do_not_mutate_trace_state() {
    let key = Key {
        epoch: 1,
        episode: 1,
        generation: 1,
        family: Family::Periodic,
    };
    let mut r = Reference {
        key,
        weight: 1.,
        anchors: [Anchor::default(); 7],
    };
    r.anchors[0] = Anchor {
        interval: Some([0., 0.]),
        weight: 1.,
        ..Anchor::default()
    };
    let mut bank = Bank::new(1, parameters(), 16).unwrap();
    let apply = |bank: &mut Bank, id, head, interval, r: Reference| {
        bank.observe(Outcome {
            id,
            head,
            interval: Some(interval),
            references: &[r],
            observed_fraction: 1.,
            retained: &[key],
            confirmed: true,
        })
    };
    apply(&mut bank, 1, Head::Onset, [1., 1.], r).unwrap();
    let before = bank.traces[0].mass;
    assert!(apply(&mut bank, 1, Head::Onset, [1., 1.], r).is_err());
    assert!(apply(&mut bank, 2, Head::Onset, [0.8, 0.9], r).is_err());
    assert_eq!(bank.traces[0].mass, before);
    apply(&mut bank, 0, Head::Release, [1., 1.], r).unwrap();
    assert!(apply(&mut bank, 9, Head::Onset, [1., 1.], r).is_err());
    let before = bank.traces[0].mass;
    r.anchors[0].period = 0.;
    assert!(apply(&mut bank, 3, Head::Onset, [2., 2.], r).is_err());
    assert_eq!(bank.traces[0].mass, before);
    assert!(std::mem::size_of::<Reference>() <= 512);
}

#[test]
fn elapsed_only_control_keeps_identical_credit_and_omits_only_competing_interference() {
    let first = Key {
        epoch: 1,
        episode: 1,
        generation: 1,
        family: Family::Periodic,
    };
    let second = Key {
        episode: 2,
        ..first
    };
    let mut full = Bank::new(1, parameters(), 16).unwrap();
    let mut control = Bank::new(
        1,
        Parameters {
            interference: false,
            ..parameters()
        },
        16,
    )
    .unwrap();
    for (id, key) in [first, second, first].into_iter().enumerate() {
        let mut r = Reference {
            key,
            weight: 0.5,
            anchors: [Anchor::default(); 7],
        };
        r.anchors[0] = Anchor {
            interval: Some([0., 0.]),
            weight: 1.,
            ..Anchor::default()
        };
        let receipts: Vec<_> = [&mut full, &mut control]
            .into_iter()
            .map(|bank| {
                bank.observe(Outcome {
                    id: id as u64,
                    head: Head::Onset,
                    interval: Some([id as f64, id as f64]),
                    references: &[r],
                    observed_fraction: 0.5,
                    retained: &[first, second],
                    confirmed: true,
                })
                .unwrap()
            })
            .collect();
        assert_eq!(receipts[0].credits, receipts[1].credits);
        close(receipts[0].unassigned, 0.75);
    }
    let full_mass = full.traces.iter().find(|t| t.key == first).unwrap().mass[0][0].exp();
    let control_mass = control.traces.iter().find(|t| t.key == first).unwrap().mass[0][0].exp();
    close(full_mass, 0.25 * (-2_f64 / 20. - 0.25 / 0.7).exp() + 0.25);
    close(control_mass, 0.25 * (-2_f64 / 20.).exp() + 0.25);
    assert!(full_mass < control_mass);
}

#[test]
fn all_references_update_before_stable_capacity_eviction_and_epoch_loss() {
    let mut bank = Bank::new(1, parameters(), 16).unwrap();
    let storage = bank.traces.as_ptr();
    let keys: Vec<_> = (1..=32)
        .map(|episode| Key {
            epoch: 1,
            episode,
            generation: 1,
            family: Family::Periodic,
        })
        .collect();
    for (step, half) in keys.chunks(16).enumerate() {
        let references: Vec<_> = half
            .iter()
            .map(|key| {
                let mut r = Reference {
                    key: *key,
                    weight: 1. / 16.,
                    anchors: [Anchor::default(); 7],
                };
                r.anchors[0] = Anchor {
                    interval: Some([0., 0.]),
                    weight: 1.,
                    ..Anchor::default()
                };
                r
            })
            .collect();
        let receipt = bank
            .observe(Outcome {
                id: step as u64,
                head: Head::Onset,
                interval: Some([1., 1.]),
                references: &references,
                observed_fraction: 1.,
                retained: &keys,
                confirmed: true,
            })
            .unwrap();
        assert_eq!(receipt.credits, [1. / 16.; 16]);
        assert_eq!(bank.traces.as_ptr(), storage);
        assert_eq!(bank.traces.len(), 16);
        if step == 1 {
            assert_eq!(
                receipt.evicted.into_iter().flatten().collect::<Vec<_>>(),
                keys[..16]
            );
            for key in half {
                assert!(bank.probabilities(*key, Head::Onset).is_some());
            }
        }
    }
    let foreign = Key {
        epoch: 2,
        ..keys[16]
    };
    let reference = Reference {
        key: foreign,
        weight: 1.,
        anchors: [Anchor::default(); 7],
    };
    let receipt = bank
        .observe(Outcome {
            id: 3,
            head: Head::Onset,
            interval: Some([2., 2.]),
            references: &[reference],
            observed_fraction: 1.,
            retained: &[foreign],
            confirmed: true,
        })
        .unwrap();
    assert_eq!(receipt.unassigned, 1.);
    assert_eq!(receipt.removed.into_iter().flatten().count(), 16);
    assert!(bank.traces.is_empty());
}

#[test]
fn returned_episode_with_a_new_group_keeps_both_relative_timing_modes() {
    let key = Key {
        epoch: 1,
        episode: 9,
        generation: 4,
        family: Family::Periodic,
    };
    let mut bank = Bank::new(1, parameters(), 16).unwrap();
    for (id, time) in [1., 1.5].into_iter().enumerate() {
        let mut reference = Reference {
            key,
            weight: 1.,
            anchors: [Anchor::default(); 7],
        };
        reference.anchors[0] = Anchor {
            group: Handle {
                bus: 0,
                epoch: 1,
                generation: id as u64 + 10,
            },
            interval: Some([0., 0.]),
            period: 1.,
            weight: 1.,
        };
        bank.observe(Outcome {
            id: id as u64,
            head: Head::Onset,
            interval: Some([time, time]),
            references: &[reference],
            observed_fraction: 1.,
            retained: &[key],
            confirmed: true,
        })
        .unwrap();
    }
    assert_eq!(bank.traces.len(), 1);
    let probabilities = bank.probabilities(key, Head::Onset).unwrap();
    assert!(probabilities[0] > 0.49);
    assert!(probabilities[16] > 0.5);
    assert_eq!(bank.lookup(key, Head::Onset, 0.25), Some(0.));
    assert!(bank.probabilities(key, Head::Release).is_none());
}
