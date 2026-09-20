use super::*;
use crate::temporal_cognition::{
    gesture::State,
    joint::{Normalizer, Pair, Shared},
};

#[test]
fn observed_articulation_uses_owned_elapsed_time_and_original_acoustic_support() {
    use crate::temporal_cognition::{
        features::RawDescriptor,
        gesture::{
            Run,
            articulation::{Articulation, Input},
        },
        ridge::Handle,
    };
    let group = Handle {
        bus: 0,
        epoch: 2,
        generation: 3,
    };
    let mut config = crate::config::TemporalGestureConfig {
        rms_reference: 0.1,
        means: [0.2; 5],
        deviations: [2.; 5],
        coefficients: [[[0.; 11]; 4]; 4],
    };
    for (a, row) in config.coefficients.iter_mut().enumerate() {
        for (b, cell) in row.iter_mut().enumerate() {
            for (i, x) in cell.iter_mut().enumerate() {
                *x = (a + 2 * b + i) as f64 * 0.07 - 0.3;
            }
        }
    }
    let mut cases = 0;
    for raw_present in [true, false] {
        for quiet in [false, true] {
            for entered in [0, 900, 1000] {
                let raw = RawDescriptor {
                    group,
                    start: 1000,
                    end: 1100,
                    known_samples: 100,
                    source_start: 800,
                    source_end: 1100,
                    available_end: 1100,
                    values: [
                        None,
                        None,
                        Some(if quiet { -20. } else { -2. }),
                        Some(0.3),
                        None,
                        Some(0.7),
                        None,
                        None,
                        None,
                        None,
                    ],
                };
                let input = Input::new(
                    group,
                    raw_present.then_some(&raw),
                    Some(0.9),
                    [1000, 1100],
                    1000,
                    &config,
                )
                .unwrap();
                for state in [
                    None,
                    Some(State::Attack),
                    Some(State::Continuation),
                    Some(State::Release),
                    Some(State::Gap),
                ] {
                    let parent = state.map(|state| Articulation {
                        state,
                        entered,
                        run: Run::default(),
                    });
                    let (list, children) =
                        List::observed_articulation(parent, &input, &config).unwrap();
                    let mut scores = [0.; 4];
                    if let Some(state) = state {
                        let values = [
                            raw_present.then_some(0.3),
                            None,
                            raw_present.then_some(0.7),
                            Some(0.9),
                            Some(((1000 - entered) as f64 / 1000.).ln_1p()),
                        ];
                        let x: [f64; 11] = std::array::from_fn(|i| {
                            if i == 0 {
                                1.
                            } else if i < 6 {
                                values[i - 1].map_or(0., |v| (v - 0.2) / 2.)
                            } else {
                                f64::from(values[i - 6].is_none())
                            }
                        });
                        let rates: [f64; 4] = std::array::from_fn(|to| {
                            if to == state as usize {
                                0.
                            } else {
                                let z: f64 = config.coefficients[state as usize][to]
                                    .iter()
                                    .zip(x)
                                    .map(|(a, b)| a * b)
                                    .sum();
                                (1. + z.exp()).ln()
                            }
                        });
                        let total: f64 = rates.iter().sum();
                        let survival = (-0.1 * total).exp();
                        for i in 0..4 {
                            scores[i] = if i == state as usize {
                                survival
                            } else {
                                (1. - survival) * rates[i] / total
                            };
                        }
                        if state != State::Gap && !(raw_present && quiet) {
                            scores[3] = 0.;
                        }
                    } else if raw_present {
                        scores = [1., 1., 1., f64::from(quiet)];
                    }
                    let total: f64 = scores.iter().sum();
                    for entry in list.entries[..list.len].iter().flatten() {
                        let expected = entry.id.map_or(
                            if total == 0. {
                                1.
                            } else {
                                1. - (-0.01_f64).exp()
                            },
                            |id| (-0.01_f64).exp() * scores[id as usize] / total,
                        );
                        assert!((entry.log_weight.exp() - expected).abs() < 1e-13);
                        if let Some(id) = entry.id {
                            let child = children[id as usize].unwrap();
                            assert_eq!(child.state as u64, id);
                            assert_eq!(
                                child.entered,
                                if state == Some(child.state) {
                                    entered
                                } else {
                                    1100
                                }
                            );
                            if state != Some(child.state)
                                && child.state == State::Attack
                                && raw_present
                            {
                                let support = child.run.attack.unwrap();
                                assert_eq!(
                                    (
                                        support.start,
                                        support.end,
                                        support.source_start,
                                        support.source_end,
                                        support.available
                                    ),
                                    (1000, 1100, 800, 1100, 1100)
                                );
                            }
                            if !raw_present {
                                assert!(child.run.censored);
                            }
                        }
                    }
                    cases += 1;
                }
            }
        }
    }
    eprintln!("JOINT_ARTICULATION_RATE cases={cases}");
}

#[test]
fn active_phrase_is_conditional_on_one_articulation_and_matches_closed_form() {
    let mut cfg = crate::temporal_cognition::phrase::tests::config();
    cfg.means = [0.; 12];
    cfg.deviations = [1.; 12];
    cfg.means[4] = 0.4;
    cfg.deviations[4] = 2.;
    cfg.hazard = [0.; 26];
    cfg.hazard[0] = -1.3;
    cfg.hazard[2..6].copy_from_slice(&[2.1, -0.4, 1.1, -0.9]);
    cfg.hazard[6] = 0.7;
    cfg.hazard[14..18].copy_from_slice(&[0.15, -0.2, 0.3, -0.1]);
    cfg.hazard[18] = 0.25;
    cfg.exits = [[0.; 26]; 4];
    let bias = [0.7, -0.2, 0.1, -0.6];
    for (j, row) in cfg.exits.iter_mut().enumerate() {
        row[0] = bias[j];
        for i in 0..4 {
            row[2 + i] = (j as f64 - 1.5) * (i as f64 - 1.25) * 0.3;
            row[14 + i] = (j as f64 - 1.5) * 0.05;
        }
    }
    let mut stay_by_state = [0.; 5];
    let mut cases = 0;
    for (index, state) in [
        Some(State::Attack),
        Some(State::Continuation),
        Some(State::Release),
        Some(State::Gap),
        None,
    ]
    .into_iter()
    .enumerate()
    {
        for value in [None, Some(-2.), Some(2.)] {
            let mut values = [None; 12];
            // Caller-provided marginal indicators must never replace the selected state.
            values[..4].fill(Some(100.));
            values[4] = value;
            for elapsed in [[0_u64, 0], [0, 1000], [20000, 20125]] {
                for low in [false, true] {
                    let dt = (elapsed[1] - elapsed[0]) as f64 / 1000.;
                    let mut intercept: f64 = -1.3 + value.map_or(0.25, |v| 0.7 * (v - 0.4) / 2.);
                    intercept += if index < 4 {
                        [2.1, -0.4, 1.1, -0.9][index]
                    } else {
                        0.15 - 0.2 + 0.3 - 0.1
                    };
                    let hazard = intercept.max(0.) + (-intercept.abs()).exp().ln_1p();
                    let survival = (-hazard * dt).exp();
                    let logits: [f64; 4] = std::array::from_fn(|j| {
                        bias[j]
                            + if index < 4 {
                                (j as f64 - 1.5) * (index as f64 - 1.25) * 0.3
                            } else {
                                4. * (j as f64 - 1.5) * 0.05
                            }
                    });
                    let denominator: f64 = logits.iter().map(|l| l.exp()).sum();
                    let mut raw = [0.; 5];
                    raw[0] = survival;
                    for i in 0..4 {
                        if i != 3 || low {
                            raw[i + 1] = (1. - survival) * logits[i].exp() / denominator;
                        }
                    }
                    let total: f64 = raw.iter().sum();
                    let keep = (-dt / 30.).exp();
                    let ids = [10, 20, 30, 40, 50];
                    let list =
                        List::active_phrase(ids, values, state, elapsed, 1000, low, cfg).unwrap();
                    assert_eq!(list.kind, Kind::Phrase);
                    assert_eq!(list.truncated, 0);
                    for i in 0..5 {
                        let row = list.entries.iter().flatten().find(|r| r.id == Some(ids[i]));
                        assert_eq!(row.is_some(), raw[i] > 0.);
                        if let Some(row) = row {
                            close(row.log_weight.exp(), raw[i] / total * keep);
                            assert_eq!(row.stay, i == 0);
                            assert_eq!(
                                row.boundary,
                                Some(if i == 0 {
                                    Boundary::Stay
                                } else {
                                    Boundary::Exit
                                })
                            );
                        }
                    }
                    let unknown = list
                        .entries
                        .iter()
                        .flatten()
                        .find(|r| r.id.is_none())
                        .unwrap();
                    close(unknown.log_weight.exp(), -(-dt / 30.).exp_m1());
                    close(
                        list.entries
                            .iter()
                            .flatten()
                            .map(|r| r.log_weight.exp())
                            .sum(),
                        1.,
                    );
                    if value == Some(2.) && elapsed == [0, 1000] && low {
                        stay_by_state[index] = list
                            .entries
                            .iter()
                            .flatten()
                            .find(|r| r.stay)
                            .unwrap()
                            .log_weight
                            .exp();
                    }
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 90);
    assert!((stay_by_state[0] - stay_by_state[2]).abs() > 0.01);
    assert!((stay_by_state[4] - stay_by_state[1]).abs() > 0.01);
    let old = 1_u64 << 50;
    let aged = List::active_phrase(
        [1, 2, 3, 4, 5],
        [None; 12],
        None,
        [old, old + 512],
        48000,
        true,
        cfg,
    )
    .unwrap();
    let unknown = aged
        .entries
        .iter()
        .flatten()
        .find(|r| r.id.is_none())
        .unwrap();
    let expected = -(-(512_f64 / 48000.) / 30.).exp_m1();
    assert!((unknown.log_weight.exp() - expected).abs() < 1e-16);
    let rounded_dt = (old + 512) as f64 / 48000. - old as f64 / 48000.;
    assert!((expected - -(-rounded_dt / 30.).exp_m1()).abs() > 1e-10);
    cfg.hazard[1] = 1e100;
    let unsupported = List::active_phrase(
        [1, 2, 3, 4, 5],
        [None; 12],
        None,
        [1000, 2000],
        1000,
        true,
        cfg,
    )
    .unwrap();
    assert_eq!(unsupported.len, 1);
    assert_eq!(unsupported.entries[0].unwrap().id, None);
    assert_eq!(unsupported.entries[0].unwrap().log_weight, 0.);
    for (elapsed, rate) in [([2, 1], 1000), ([0, 1], 0)] {
        assert!(
            List::active_phrase([1, 2, 3, 4, 5], [None; 12], None, elapsed, rate, true, cfg)
                .is_err()
        );
    }
    assert!(
        List::active_phrase(
            [1, 1, 2, 3, 4],
            [None; 12],
            None,
            [1000, 2000],
            1000,
            true,
            cfg
        )
        .is_err()
    );
}

#[test]
fn conditional_active_phrase_enters_the_bounded_compatible_tuple_inventory() {
    let cfg = crate::temporal_cognition::phrase::tests::config();
    let known = |id, score, stay, boundary| Known {
        id,
        raw_score: score,
        stay,
        boundary,
    };
    let mut composer = Composer::new();
    for state in [State::Attack, State::Release] {
        let lists = [
            List::articulation(Some(state), [0.1, 0.2, 0.3, 0.4], 0.1, true, true).unwrap(),
            List::build(Kind::Grouping, &[known(7, 1., true, None)], false, 0.1, 19).unwrap(),
            List::active_phrase(
                [10, 20, 30, 40, 50],
                [None; 12],
                Some(state),
                [1000, 1100],
                1000,
                true,
                cfg,
            )
            .unwrap(),
            List::build(
                Kind::Section,
                &[
                    known(8, 0.8, true, Some(Boundary::Stay)),
                    known(9, 0.2, false, Some(Boundary::Exit)),
                ],
                false,
                0.1,
                5,
            )
            .unwrap(),
            List::build(
                Kind::Correspondence,
                &[known(11, 1., true, None)],
                false,
                0.1,
                19,
            )
            .unwrap(),
        ];
        let output = composer
            .compose(
                &lists,
                &crate::temporal_cognition::joint::proposals::Compatibility::default(),
            )
            .unwrap();
        assert!(output.reserved_stay);
        assert_eq!(output.reserved_boundary_pairs, 15);
        assert!(output.heap_pops <= 16 && output.priority_evaluations <= 86);
        close(
            output
                .tuples
                .iter()
                .flatten()
                .map(|t| t.log_transition.exp())
                .sum(),
            1.,
        );
        for t in output.tuples.iter().flatten() {
            let expected: f64 = lists
                .iter()
                .zip(t.indices)
                .map(|(l, i)| l.entries[i].unwrap().log_weight)
                .sum();
            close(t.log_product, expected);
        }
    }
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 3e-12, "{a} != {b}");
}

#[test]
fn active_section_matches_closed_form_with_supported_return_and_development() {
    let mut head = crate::temporal_cognition::section::Head {
        means: [0.; 82],
        deviations: [1.; 82],
        hazard: [0.; 83],
        exits: [[0.; 83]; 3],
    };
    head.means[0] = 0.5;
    head.deviations[0] = 2.;
    head.hazard[0] = -0.7;
    head.hazard[1] = 0.8;
    for (i, bias) in [-0.4, 0.7, -0.3].into_iter().enumerate() {
        head.exits[i][0] = bias;
        head.exits[i][1] = [0.2, -0.3, 0.1][i];
    }
    head.validate().unwrap();
    let ids = [10, 20, 30, 40];
    for value in [None, Some(-2.), Some(3.)] {
        let mut values = [None; 82];
        values[0] = value;
        values[78] = Some(123.);
        let x: f64 = value.map_or(0., |v| (v - 0.5) / 2.);
        let hazard = (1. + (-0.7 + 0.8 * x).exp()).ln();
        let logits = [-0.4 + 0.2 * x, 0.7 - 0.3 * x, -0.3 + 0.1 * x];
        let denominator: f64 = logits.iter().map(|v| v.exp()).sum();
        for elapsed in [[0_u64, 0], [0, 1000], [20000, 20125]] {
            let dt = (elapsed[1] - elapsed[0]) as f64 / 1000.;
            let survival = (-hazard * dt).exp();
            for returned in [
                None,
                Some((0., false)),
                Some((0.25, false)),
                Some((1., false)),
                Some((0.25, true)),
                Some((1., true)),
            ] {
                let mut raw = [survival, 0., 0., 0.];
                for i in 0..3 {
                    raw[i + 1] = (1. - survival) * logits[i].exp() / denominator;
                }
                raw[2] *= returned.map_or(0., |r| r.0);
                let total: f64 = raw.iter().sum();
                let keep = (-dt / 120.).exp();
                let list =
                    List::active_section(ids, &values, elapsed, 1000, returned, &head).unwrap();
                assert_eq!(list.kind, Kind::Section);
                assert_eq!(list.truncated, 0);
                assert!(list.len <= 5);
                for i in 0..4 {
                    let row = list.entries.iter().flatten().find(|r| r.id == Some(ids[i]));
                    assert_eq!(row.is_some(), raw[i] > 0.);
                    if let Some(row) = row {
                        close(row.log_weight.exp(), raw[i] / total * keep);
                        assert_eq!(row.stay, i == 0);
                        assert_eq!(
                            row.boundary,
                            Some(if i == 0 || (i == 2 && returned.is_some_and(|r| r.1)) {
                                Boundary::Stay
                            } else {
                                Boundary::Exit
                            })
                        );
                    }
                }
                let unknown = list
                    .entries
                    .iter()
                    .flatten()
                    .find(|r| r.id.is_none())
                    .unwrap();
                close(unknown.log_weight.exp(), 1. - keep);
                assert!(!unknown.stay);
                close(
                    list.entries
                        .iter()
                        .flatten()
                        .map(|r| r.log_weight.exp())
                        .sum(),
                    1.,
                );
            }
        }
    }
    let mut values = [None; 82];
    let unknown = List::active_section(ids, &values, [0, 100], 1000, None, &head).unwrap();
    assert_eq!(unknown.len, 1);
    assert_eq!(unknown.entries[0].unwrap().id, None);
    assert_eq!(unknown.entries[0].unwrap().log_weight, 0.);
    values[78] = Some(0.);
    for returned in [
        Some((-0.1, false)),
        Some((1.1, true)),
        Some((f64::NAN, false)),
    ] {
        assert!(List::active_section(ids, &values, [0, 100], 1000, returned, &head).is_err());
    }
    for (elapsed, rate) in [([2, 1], 1000), ([0, 1], 0)] {
        assert!(List::active_section(ids, &values, elapsed, rate, None, &head).is_err());
    }
    assert!(List::active_section([1, 1, 2, 3], &values, [0, 100], 1000, None, &head).is_err());
    values[0] = Some(f64::NAN);
    assert!(List::active_section(ids, &values, [0, 100], 1000, None, &head).is_err());
    values[0] = Some(f64::MAX);
    head.hazard[1] = f64::MAX;
    let unsupported = List::active_section(ids, &values, [0, 100], 1000, None, &head).unwrap();
    assert_eq!(unsupported.len, 1);
    assert_eq!(unsupported.entries[0].unwrap().id, None);
    assert_eq!(unsupported.entries[0].unwrap().log_weight, 0.);
    head.hazard[1] = 0.;
    values[0] = None;
    let old = 1_u64 << 50;
    let aged = List::active_section(ids, &values, [old, old + 512], 48000, None, &head).unwrap();
    let unknown = aged
        .entries
        .iter()
        .flatten()
        .find(|r| r.id.is_none())
        .unwrap();
    assert!((unknown.log_weight.exp() - -(-(512_f64 / 48000.) / 120.).exp_m1()).abs() < 1e-16);
}

#[test]
fn development_reserves_boundary_stay_separately_from_complete_identity_stay() {
    let known = |id, raw_score, stay, boundary| Known {
        id,
        raw_score,
        stay,
        boundary,
    };
    let mut composer = Composer::new();
    for development_logit in [1., 2.] {
        let mut head = crate::temporal_cognition::section::Head {
            means: [0.; 82],
            deviations: [1.; 82],
            hazard: [0.; 83],
            exits: [[0.; 83]; 3],
        };
        head.hazard[0] = 50.;
        head.exits[0][0] = -development_logit / 2.;
        head.exits[1][0] = development_logit;
        head.exits[2][0] = -development_logit / 2.;
        head.validate().unwrap();
        let mut values = [None; 82];
        values[78] = Some(0.);
        let lists = [
            List::build(
                Kind::Articulation,
                &[known(1, 1., true, None)],
                false,
                0.1,
                8,
            )
            .unwrap(),
            List::build(Kind::Grouping, &[known(2, 1., true, None)], false, 0.1, 19).unwrap(),
            List::build(
                Kind::Phrase,
                &[
                    known(3, 1., true, Some(Boundary::Stay)),
                    known(4, 2., false, Some(Boundary::Exit)),
                ],
                false,
                0.1,
                6,
            )
            .unwrap(),
            List::active_section(
                [5, 7, 6, 9],
                &values,
                [0, 100],
                1000,
                Some((1., true)),
                &head,
            )
            .unwrap(),
            List::build(
                Kind::Correspondence,
                &[known(8, 1., true, None)],
                false,
                0.1,
                19,
            )
            .unwrap(),
        ];
        let output = composer
            .compose(
                &lists,
                &crate::temporal_cognition::joint::proposals::Compatibility::default(),
            )
            .unwrap();
        assert!(output.reserved_stay);
        assert_eq!(output.reserved_boundary_pairs, 15);
        assert!(output.heap_pops <= 16 && output.priority_evaluations <= 86);
        let tuples: Vec<_> = output.tuples.iter().flatten().collect();
        assert!(tuples.iter().any(|t| t.ids == [1, 2, 3, 5, 8].map(Some)));
        for (phrase, section) in [(3, 6), (3, 7), (4, 6), (4, 7)] {
            assert!(
                tuples
                    .iter()
                    .any(|t| t.ids == [1, 2, phrase, section, 8].map(Some))
            );
        }
        close(tuples.iter().map(|t| t.log_transition.exp()).sum(), 1.);
        let pairs: Vec<_> = tuples
            .iter()
            .enumerate()
            .map(|(i, t)| Pair {
                parent: 0,
                extension: i as u8,
                resolved: !t.all_unknown,
                log_prior: 0.,
                log_transition: t.log_transition,
                log_potential: 0.,
            })
            .collect();
        let groups = [pairs.as_slice()];
        let shared = [
            Shared {
                pair: Pair {
                    parent: 0,
                    extension: 0,
                    resolved: true,
                    log_prior: 0.,
                    log_transition: 0.9_f64.ln(),
                    log_potential: 0.,
                },
                groups: &groups,
            },
            Shared {
                pair: Pair {
                    parent: 0,
                    extension: 1,
                    resolved: false,
                    log_prior: 0.,
                    log_transition: 0.1_f64.ln(),
                    log_potential: 0.,
                },
                groups: &groups,
            },
        ];
        let mut engine = Normalizer::new();
        let posterior = engine.normalize(&shared, true).unwrap();
        let context = posterior.contexts[0].unwrap();
        close(
            context.weight.mass + posterior.explicit_unknown + posterior.pruned_mass,
            1.,
        );
        let group = context.groups[0];
        close(
            group.rows.iter().flatten().map(|w| w.mass).sum::<f64>()
                + group.explicit_unknown
                + group.pruned_mass,
            1.,
        );
        close(
            group.explicit_unknown,
            tuples
                .iter()
                .find(|t| t.all_unknown)
                .unwrap()
                .log_transition
                .exp(),
        );
    }
    assert!(
        List::build(
            Kind::Section,
            &[known(1, 1., true, Some(Boundary::Exit))],
            false,
            0.1,
            5
        )
        .is_err()
    );
}

#[test]
fn bounded_heap_matches_independent_full_cartesian_ranking_and_joint_normalization() {
    let data: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/temporal_cognition/joint_proposals.json"
    ))
    .unwrap();
    let mut composer = Composer::new();
    let capacity = composer.heap.capacity();
    let index_address = composer.index.as_ptr();
    assert_eq!(composer.index.len(), 86_640);
    for case in data["cases"].as_array().unwrap() {
        let lists: [List; 5] = std::array::from_fn(|index| {
            let raw: Vec<Known> = case["components"][index]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| Known {
                    id: r["id"].as_u64().unwrap(),
                    raw_score: r["score"].as_f64().unwrap(),
                    stay: r["stay"].as_bool().unwrap(),
                    boundary: r["boundary"].as_str().map(|s| {
                        if s == "stay" {
                            Boundary::Stay
                        } else {
                            Boundary::Exit
                        }
                    }),
                })
                .collect();
            List::build(
                KINDS[index],
                &raw,
                case["unknown_parent"][index].as_bool().unwrap(),
                case["dt"].as_f64().unwrap(),
                case["caps"][index].as_u64().unwrap() as usize,
            )
            .unwrap()
        });
        for (g, list) in lists.iter().enumerate() {
            assert_eq!(
                list.truncated,
                case["expected"]["truncated"][g].as_u64().unwrap() as usize
            );
            close(
                list.entries[..list.len]
                    .iter()
                    .map(|e| e.unwrap().log_weight.exp())
                    .sum(),
                1.,
            );
            for (a, b) in list.entries[..list.len]
                .iter()
                .zip(case["expected"]["lists"][g].as_array().unwrap())
            {
                let a = a.unwrap();
                assert_eq!(a.id, b["id"].as_u64());
                close(a.log_weight.exp(), b["probability"].as_f64().unwrap());
            }
        }
        let mut legal = Compatibility::default();
        for pair in case["illegal_pairs"].as_array().unwrap() {
            let p = lists[2].entries[..lists[2].len]
                .iter()
                .position(|e| e.unwrap().id == pair[0].as_u64());
            let s = lists[3].entries[..lists[3].len]
                .iter()
                .position(|e| e.unwrap().id == pair[1].as_u64());
            if let (Some(p), Some(s)) = (p, s) {
                legal.phrase_section[p][s] = false;
            }
        }
        if let Some(pairs) = case["illegal_correspondences"].as_array() {
            for pair in pairs {
                let s = lists[3].entries[..lists[3].len]
                    .iter()
                    .position(|e| e.unwrap().id == pair[0].as_u64());
                let c = lists[4].entries[..lists[4].len]
                    .iter()
                    .position(|e| e.unwrap().id == pair[1].as_u64());
                if let (Some(s), Some(c)) = (s, c) {
                    legal.section_correspondence[s][c] = false;
                }
            }
        }
        let output = composer.compose(&lists, &legal).unwrap();
        let expected = &case["expected"];
        assert_eq!(
            output.tuples.iter().flatten().count(),
            expected["rows"].as_array().unwrap().len()
        );
        assert_eq!(
            output.heap_pops,
            expected["heap_pops"].as_u64().unwrap() as usize
        );
        assert_eq!(
            output.reserved_stay,
            expected["reserved_stay"].as_bool().unwrap()
        );
        assert_eq!(
            output.reserved_boundary_pairs,
            expected["boundary_mask"].as_u64().unwrap() as u8
        );
        assert!(
            output.priority_evaluations <= 86
                && output.heap_pops <= 16
                && output.compatibility_checks <= 48
                && output.correspondence_checks <= 95
        );
        assert_eq!(composer.heap.capacity(), capacity);
        assert_eq!(composer.index.as_ptr(), index_address);
        close(
            output.log_admitted_product_mass.exp(),
            expected["admitted_product_mass"].as_f64().unwrap(),
        );
        for (a, b) in output
            .tuples
            .iter()
            .flatten()
            .zip(expected["rows"].as_array().unwrap())
        {
            assert_eq!(
                a.ids,
                std::array::from_fn(|i| b["ids"][i].as_u64()),
                "{}",
                case["name"]
            );
            close(a.log_transition.exp(), b["probability"].as_f64().unwrap());
        }
        let pairs: Vec<_> = output
            .tuples
            .iter()
            .flatten()
            .enumerate()
            .map(|(i, t)| Pair {
                parent: 0,
                extension: i as u8,
                resolved: !t.all_unknown,
                log_prior: 0.,
                log_transition: t.log_transition,
                log_potential: 0.,
            })
            .collect();
        let groups = [pairs.as_slice()];
        let shared = [
            Shared {
                pair: Pair {
                    parent: 0,
                    extension: 0,
                    resolved: true,
                    log_prior: 0.,
                    log_transition: 0.9_f64.ln(),
                    log_potential: 0.,
                },
                groups: &groups,
            },
            Shared {
                pair: Pair {
                    parent: 0,
                    extension: 1,
                    resolved: false,
                    log_prior: 0.,
                    log_transition: 0.1_f64.ln(),
                    log_potential: 0.,
                },
                groups: &groups,
            },
        ];
        let mut engine = Normalizer::new();
        let posterior = engine.normalize(&shared, true).unwrap();
        close(posterior.contexts[0].unwrap().weight.mass, 0.9);
        close(
            posterior.contexts[0].unwrap().groups[0].explicit_unknown,
            output
                .tuples
                .iter()
                .flatten()
                .find(|t| t.all_unknown)
                .unwrap()
                .log_transition
                .exp(),
        );
    }
}

#[test]
fn articulation_inventory_has_one_unknown_and_observed_gap_admission_only() {
    for parent in [
        None,
        Some(State::Attack),
        Some(State::Continuation),
        Some(State::Release),
        Some(State::Gap),
    ] {
        for observed in [false, true] {
            for low in [false, true] {
                let list = List::articulation(parent, [1.; 4], 0.1, observed, low).unwrap();
                assert!(list.len <= 5);
                close(
                    list.entries[..list.len]
                        .iter()
                        .map(|e| e.unwrap().log_weight.exp())
                        .sum(),
                    1.,
                );
                assert_eq!(
                    list.entries[..list.len]
                        .iter()
                        .filter(|e| e.unwrap().id.is_none())
                        .count(),
                    1
                );
                let gap = list.entries[..list.len]
                    .iter()
                    .find(|e| e.unwrap().id == Some(3));
                assert_eq!(gap.is_some(), parent == Some(State::Gap) || observed && low);
                if parent.is_none() && !observed {
                    assert_eq!(list.len, 1);
                    assert_eq!(list.entries[0].unwrap().log_weight, 0.);
                } else {
                    let keep = (-0.01_f64).exp();
                    close(
                        list.entries[..list.len]
                            .iter()
                            .filter(|e| e.unwrap().id.is_some())
                            .map(|e| e.unwrap().log_weight.exp())
                            .sum(),
                        keep,
                    );
                    if parent.is_none() {
                        for entry in list.entries[..list.len]
                            .iter()
                            .flatten()
                            .filter(|e| e.id.is_some())
                        {
                            close(entry.log_weight.exp(), keep / if low { 4. } else { 3. });
                        }
                    }
                }
            }
        }
    }
    let quiet = List::articulation(Some(State::Gap), [1.; 4], 0.1, false, false).unwrap();
    close(
        quiet.entries[..quiet.len]
            .iter()
            .flatten()
            .find(|e| e.id == Some(3))
            .unwrap()
            .log_weight
            .exp(),
        (-0.31_f64).exp(),
    );
    assert!(List::articulation(Some(State::Attack), [f64::MAX; 4], 0.1, true, true).is_err());
}

#[test]
fn unsupported_lists_stay_unknown_and_invalid_aliases_or_boundaries_are_rejected() {
    let unknown: [List; 5] =
        std::array::from_fn(|i| List::build(KINDS[i], &[], true, 0.1, CAPS[i]).unwrap());
    let mut composer = Composer::new();
    let output = composer
        .compose(
            &unknown,
            &crate::temporal_cognition::joint::proposals::Compatibility::default(),
        )
        .unwrap();
    assert_eq!(output.tuples.iter().flatten().count(), 1);
    assert!(output.tuples[0].unwrap().all_unknown);
    assert_eq!(output.tuples[0].unwrap().log_transition, 0.);
    assert_eq!(output.priority_evaluations, 1);
    assert_eq!(output.heap_pops, 1);
    assert!(output.reserved_stay);
    let mut illegal = Compatibility::default();
    illegal.phrase_section[0][0] = false;
    assert!(composer.compose(&unknown, &illegal).is_err());
    let mut raw = [Known {
        id: 0,
        raw_score: 1.,
        stay: true,
        boundary: None,
    }; 2];
    assert!(List::build(Kind::Grouping, &raw, false, 0.1, 19).is_err());
    raw[1].id = 1;
    raw[1].stay = false;
    assert!(List::build(Kind::Grouping, &raw, false, 0.1, 1).is_err());
    assert!(List::build(Kind::Grouping, &raw, true, 0.1, 19).is_err());
    assert!(List::build(Kind::Phrase, &raw, false, 0.1, 6).is_err());
    raw[0].raw_score = f64::NAN;
    assert!(List::build(Kind::Grouping, &raw, false, 0.1, 19).is_err());
    assert!(List::build(Kind::Grouping, &[], true, -0.1, 19).is_err());
    assert!(List::build(Kind::Grouping, &[], true, 0.1, 20).is_err());
    let recovered = composer
        .compose(
            &unknown,
            &crate::temporal_cognition::joint::proposals::Compatibility::default(),
        )
        .unwrap();
    assert_eq!(recovered.priority_evaluations, 1);
    assert_eq!(recovered.tuples[0].unwrap().log_transition, 0.);
}

#[test]
fn equal_priority_tuples_keep_identity_order_across_reordered_inputs() {
    let mut composer = Composer::new();
    let mut expected = vec![
        [None; 5],
        [Some(1); 5],
        [Some(0), Some(0), Some(1), Some(1), Some(0)],
        [Some(0), Some(0), Some(1), Some(0), Some(0)],
        [Some(0), Some(0), Some(0), Some(1), Some(0)],
        [Some(0); 5],
    ];
    for bits in 0..32 {
        let ids = std::array::from_fn(|g| Some((bits >> (4 - g)) & 1));
        if expected.len() < 16 && !expected.contains(&ids) {
            expected.push(ids);
        }
    }
    for reverse in [false, true, false] {
        let lists = std::array::from_fn(|g| {
            let raw: [Known; 2] = std::array::from_fn(|i| {
                let id = if reverse { 1 - i } else { i } as u64;
                Known {
                    id,
                    raw_score: 1.,
                    stay: id == 1,
                    boundary: matches!(KINDS[g], Kind::Phrase | Kind::Section).then_some(
                        if id == 1 {
                            Boundary::Stay
                        } else {
                            Boundary::Exit
                        },
                    ),
                }
            });
            List::build(KINDS[g], &raw, false, 0., CAPS[g]).unwrap()
        });
        let output = composer
            .compose(
                &lists,
                &crate::temporal_cognition::joint::proposals::Compatibility::default(),
            )
            .unwrap();
        let actual: Vec<_> = output.tuples.iter().flatten().map(|t| t.ids).collect();
        assert_eq!(actual, expected);
        assert_eq!(output.reserved_boundary_pairs, 15);
        for tuple in output.tuples.iter().flatten() {
            close(
                tuple.log_transition.exp(),
                if tuple.all_unknown { 0. } else { 1. / 15. },
            );
        }
    }
}

#[test]
fn unknown_leak_uses_each_components_physical_time_scale_once() {
    for (g, tau) in [10., 10., 30., 120., 10.].into_iter().enumerate() {
        let known = [Known {
            id: 0,
            raw_score: 1.,
            stay: true,
            boundary: matches!(KINDS[g], Kind::Phrase | Kind::Section).then_some(Boundary::Stay),
        }];
        for dt in [0., 0.01, 2., 60., 240.] {
            let full = List::build(KINDS[g], &known, false, dt, CAPS[g]).unwrap();
            let half = List::build(KINDS[g], &known, false, dt / 2., CAPS[g]).unwrap();
            let known_log = |list: &List| {
                list.entries[..list.len]
                    .iter()
                    .flatten()
                    .find(|e| e.id.is_some())
                    .unwrap()
                    .log_weight
            };
            close(known_log(&full), -dt / tau);
            close(known_log(&full), 2. * known_log(&half));
            close(
                full.entries[..full.len]
                    .iter()
                    .flatten()
                    .find(|e| e.id.is_none())
                    .unwrap()
                    .log_weight
                    .exp(),
                1. - (-dt / tau).exp(),
            );
        }
    }
}
