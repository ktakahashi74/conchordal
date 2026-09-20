use super::*;
use serde_json::{Value, json};

fn group() -> Handle {
    Handle {
        bus: 0,
        epoch: 7,
        generation: 2,
    }
}

fn controls() -> Controls {
    Controls {
        tolerance: 0.1,
        integers_234_only: false,
        strict_integer: false,
        one_skip_words: false,
    }
}

fn accent(end: u64, prefix: u64, weight: f64) -> Accent {
    Accent {
        group: group(),
        event_start: end - 16,
        event_end: end,
        raw_intervals: [
            (end - 48, end - 32),
            (end - 32, end - 16),
            (end - 16, end),
            (end, end + 16),
        ],
        source_start: end - 48,
        source_end: end + 16,
        available_end: end + 32,
        weight,
        observed_prefix: prefix,
    }
}

fn blank() -> View {
    View {
        group: group(),
        refreshed_at: 0,
        cadence_slot: 0,
        superseded_slots: 0,
        period_source_end: None,
        period_available_end: None,
        retained_span: None,
        capacity_evicted_through: None,
        proposals: [None; 16],
        work: Work::default(),
        controls: controls(),
    }
}

fn scan(ends: &[u64], c: Controls) -> View {
    let bank = ends.iter().map(|&end| accent(end, end, 1.)).collect();
    let mut peaks = [None; 8];
    peaks[0] = Some(Peak {
        bin: 96,
        period_seconds: 0.5,
        support: 1.,
    });
    let mut view = blank();
    view.controls = c;
    admit(&bank, &peaks, 1000, c, &mut view);
    view
}

fn at_start(view: &View, shape: Shape, start: u64) -> Option<Proposal> {
    view.proposals
        .iter()
        .flatten()
        .find(|p| p.key.shape == shape && p.key.anchors[0].1 == start)
        .copied()
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-11,
        "actual={actual} expected={expected}"
    );
}

#[test]
fn exhaustive_independent_inventory_matches_all_controls_and_saturation() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../../tests/fixtures/temporal_cognition/grouping-inventory.json"
    ))
    .unwrap();
    let mut totals = [0usize; 3];
    for case in fixture["cases"].as_array().unwrap() {
        let bank: VecDeque<_> = case["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| {
                accent(
                    e[0].as_u64().unwrap(),
                    e[1].as_u64().unwrap(),
                    e[2].as_f64().unwrap(),
                )
            })
            .collect();
        let mut peaks = [None; 8];
        for (p, e) in peaks.iter_mut().zip(case["peaks"].as_array().unwrap()) {
            *p = Some(Peak {
                bin: e[0].as_u64().unwrap() as usize,
                period_seconds: e[1].as_f64().unwrap(),
                support: 1.,
            });
        }
        let c = &case["controls"];
        let c = Controls {
            tolerance: c["tolerance"].as_f64().unwrap(),
            integers_234_only: c["integers_234_only"].as_bool().unwrap(),
            strict_integer: c["strict_integer"].as_bool().unwrap(),
            one_skip_words: c["one_skip_words"].as_bool().unwrap(),
        };
        let mut view = blank();
        admit(
            &bank,
            &peaks,
            case["sample_rate"].as_u64().unwrap() as u32,
            c,
            &mut view,
        );
        let expected = &case["expected"];
        assert_eq!(
            view.work.admitted_cases,
            expected["admitted_cases"].as_u64().unwrap() as usize,
            "{}",
            case["name"]
        );
        let actual: Vec<_> = view.proposals.iter().flatten().collect();
        assert_eq!(
            actual.len(),
            expected["proposals"].as_array().unwrap().len(),
            "{}",
            case["name"]
        );
        for (a, b) in actual.iter().zip(expected["proposals"].as_array().unwrap()) {
            assert_eq!(
                u64::from(a.key.period_bin),
                b["period_bin"].as_u64().unwrap(),
                "{}",
                case["name"]
            );
            match a.key.shape {
                Shape::Integer(n) => {
                    assert_eq!(b["kind"], "integer");
                    assert_eq!(b["shape"], n);
                }
                Shape::Word { len, symbols } => {
                    assert_eq!(b["kind"], "word");
                    assert_eq!(json!(symbols[..len as usize]), b["shape"]);
                }
            }
            assert_eq!(
                json!(
                    a.key.anchors[..a.key.anchor_count as usize]
                        .iter()
                        .map(|(_, end)| end)
                        .collect::<Vec<_>>()
                ),
                b["anchors"],
                "{}",
                case["name"]
            );
            assert_eq!(
                json!(a.observed_duration_samples),
                b["observed_duration_samples"]
            );
            assert_eq!(json!(a.coverage), b["coverage"]);
            close(a.period_seconds, b["period_seconds"].as_f64().unwrap());
            close(
                a.nominal_duration_seconds,
                b["nominal_duration_seconds"].as_f64().unwrap(),
            );
            close(a.endpoint_weight, b["endpoint_weight"].as_f64().unwrap());
            close(
                a.mean_endpoint_weight,
                b["mean_endpoint_weight"].as_f64().unwrap(),
            );
            for (actual, expected) in a
                .timing_residuals
                .iter()
                .zip(b["timing_residuals"].as_array().unwrap())
            {
                close(*actual, expected.as_f64().unwrap());
            }
            assert!(
                a.timing_residuals[a.key.anchor_count as usize - 1..]
                    .iter()
                    .all(|&x| x == 0.)
            );
            assert_eq!(a.source_start, b["source_start"].as_u64().unwrap());
            assert_eq!(a.source_end, b["source_end"].as_u64().unwrap());
            assert_eq!(a.available_end, b["available_end"].as_u64().unwrap());
            assert_eq!(
                a.skipped_accent.map(|(_, end)| end),
                b["skipped_accent"].as_u64()
            );
        }
        totals[0] += 1;
        totals[1] += view.work.admitted_cases;
        totals[2] += actual.len();
        if case["name"] == "saturated_128" {
            assert_eq!(view.work.integer_cases, 15360);
            assert_eq!(view.work.word_cases, 7168);
            assert!(view.work.boundary_search_comparisons <= 15360 * 2 * 8);
            assert!(view.work.step_comparisons <= 7168 * 16 * 9);
        }
    }
    println!(
        "GROUPING_INVENTORY_ORACLE {}",
        json!({"cases":totals[0],"admitted_cases":totals[1],"retained_proposals":totals[2]})
    );
}

#[test]
fn integer_search_is_two_sided_fixed_to_origin_and_each_duration_is_checked() {
    let shape = Shape::Integer(4);
    let positive = scan(&[1000, 2900, 4900], controls());
    let p = at_start(&positive, shape, 1000).unwrap();
    assert_eq!(p.observed_duration_samples, [1900, 2000]);
    close(p.timing_residuals[0], -0.05);
    assert!(at_start(&scan(&[1000, 3000, 5350], controls()), shape, 1000).is_none());
    assert!(at_start(&scan(&[1000, 2800, 5200], controls()), shape, 1000).is_none());
    let tie = scan(&[1000, 2800, 3200, 5000], controls());
    assert_eq!(at_start(&tie, shape, 1000).unwrap().key.anchors[1].1, 2800);
    let extra = [1000, 1250, 3000, 3250, 5000];
    assert!(at_start(&scan(&extra, controls()), shape, 1000).is_some());
    assert!(
        at_start(
            &scan(
                &extra,
                Controls {
                    strict_integer: true,
                    ..controls()
                }
            ),
            shape,
            1000
        )
        .is_none()
    );
    let large = scan(&[1 << 60, (1 << 60) + 1900, (1 << 60) + 3900], controls());
    assert_eq!(
        at_start(&large, shape, 1 << 60)
            .unwrap()
            .observed_duration_samples,
        [1900, 2000]
    );
}

#[test]
fn words_require_order_and_consecutive_steps_with_explicit_skip_control() {
    let word = Shape::Word {
        len: 2,
        symbols: [4, 5, 0, 0, 0, 0, 0, 0],
    };
    let positive = scan(&[1000, 1500, 2250, 2750, 3500], controls());
    let p = at_start(&positive, word, 1000).unwrap();
    assert_eq!(p.observed_duration_samples, [1250, 1250]);
    assert_eq!(p.mean_endpoint_weight, 1.);
    assert!(
        at_start(
            &scan(&[1000, 1500, 2250, 3000, 3500], controls()),
            word,
            1000
        )
        .is_none()
    );
    let extra = [1000, 1250, 1500, 2250, 2750, 3500];
    assert!(at_start(&scan(&extra, controls()), word, 1000).is_none());
    let control = scan(
        &extra,
        Controls {
            one_skip_words: true,
            ..controls()
        },
    );
    assert_eq!(
        at_start(&control, word, 1000).unwrap().skipped_accent,
        Some((1234, 1250))
    );
    let rotations = scan(&[1000, 1500, 2250, 2750, 3500, 4000, 4750], controls());
    assert!(at_start(&rotations, word, 1000).is_some());
    assert!(
        at_start(
            &rotations,
            Shape::Word {
                len: 2,
                symbols: [5, 4, 0, 0, 0, 0, 0, 0]
            },
            1500
        )
        .is_some()
    );
}

#[test]
fn observation_support_is_per_repetition_and_not_repaired_by_later_good_coverage() {
    let peaks = [
        Some(Peak {
            bin: 96,
            period_seconds: 0.5,
            support: 1.,
        }),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ];
    for (first, expected) in [(1800, true), (1780, false)] {
        let bank = VecDeque::from([
            accent(1000, 1000, 1.),
            accent(3000, 1000 + first, 1.),
            accent(5000, 3000 + first, 1.),
        ]);
        let mut view = blank();
        admit(&bank, &peaks, 1000, controls(), &mut view);
        assert_eq!(at_start(&view, Shape::Integer(4), 1000).is_some(), expected);
    }
}

#[test]
fn shared_period_bank_drives_frozen_cadence_age_expiry_and_group_isolation() {
    let mut estimator = Estimator::new(group(), 128, 1000, 32000, 1024, 1. / 24.).unwrap();
    let mut inventory = Inventory::new(group(), 0, controls()).unwrap();
    for end in (1000..=5000).step_by(500) {
        estimator.deliver(accent(end, end, 1.), end + 32).unwrap();
    }
    assert!(inventory.refresh(&estimator).unwrap());
    let first = inventory.view.unwrap();
    assert_eq!(first.group, group());
    assert_eq!(first.refreshed_at, 5032);
    assert_eq!(first.cadence_slot, 50);
    assert_eq!(first.superseded_slots, 50);
    assert_eq!(first.period_source_end, Some(5016));
    assert_eq!(first.period_available_end, Some(5032));
    assert_eq!(first.retained_span, Some((1000, 5000)));
    // Summed endpoint support ranks these longer words ahead of three-anchor integers.
    assert_eq!(first.proposals.iter().flatten().count(), 16);
    assert!(
        first
            .proposals
            .iter()
            .flatten()
            .all(|p| matches!(p.key.shape, Shape::Word { .. }))
    );
    let mut integer_control = Inventory::new(
        group(),
        0,
        Controls {
            integers_234_only: true,
            ..controls()
        },
    )
    .unwrap();
    integer_control.refresh(&estimator).unwrap();
    assert!(
        integer_control
            .view
            .unwrap()
            .proposals
            .iter()
            .flatten()
            .any(|p| p.key.shape == Shape::Integer(2))
    );
    assert!(first.proposals.iter().flatten().all(|p| {
        estimator
            .view
            .peaks
            .iter()
            .flatten()
            .any(|peak| peak.bin == p.key.period_bin as usize)
    }));
    assert!(
        estimator
            .deliver(accent(5500, 5500, 1.), 5500)
            .unwrap()
            .is_none()
    );
    assert!(!inventory.refresh(&estimator).unwrap());
    assert_eq!(inventory.view.unwrap().refreshed_at, 5032);
    estimator.advance(5099).unwrap();
    assert!(!inventory.refresh(&estimator).unwrap());
    assert_eq!(inventory.view.unwrap().refreshed_at, 5032);
    estimator.advance(5100).unwrap();
    assert!(inventory.refresh(&estimator).unwrap());
    assert_eq!(inventory.view.unwrap().superseded_slots, 0);
    estimator.advance(40000).unwrap();
    assert!(inventory.refresh(&estimator).unwrap());
    let expired = inventory.view.unwrap();
    assert_eq!(expired.superseded_slots, 348);
    assert!(expired.proposals.iter().all(Option::is_none));
    assert_eq!(expired.retained_span, None);
    let foreign = Estimator::new(
        Handle {
            generation: 3,
            ..group()
        },
        128,
        1000,
        32000,
        1024,
        1. / 24.,
    )
    .unwrap();
    assert!(inventory.refresh(&foreign).is_err());
    assert_eq!(inventory.view.unwrap().refreshed_at, 40000);
    assert!(
        Inventory::new(
            Handle {
                generation: 1,
                ..group()
            },
            0,
            controls()
        )
        .is_err()
    );
    assert!(
        Inventory::new(
            group(),
            0,
            Controls {
                tolerance: f64::NAN,
                ..controls()
            }
        )
        .is_err()
    );
}

#[test]
fn cap_loss_and_raw_window_limits_are_visible_without_asserting_absent_meter() {
    let mut estimator = Estimator::new(group(), 4, 1000, 32000, 1024, 1. / 24.).unwrap();
    for end in (1000..=5000).step_by(500) {
        estimator.deliver(accent(end, end, 1.), end + 32).unwrap();
    }
    let mut inventory = Inventory::new(group(), 0, controls()).unwrap();
    inventory.refresh(&estimator).unwrap();
    let view = inventory.view.unwrap();
    assert_eq!(view.capacity_evicted_through, Some(3000));
    assert_eq!(view.retained_span, Some((3500, 5000)));
    assert!(view.work.integer_window_limited_cases > 0);
    assert!(view.work.word_insufficient_endpoints > 0);
    assert!(
        view.proposals
            .iter()
            .flatten()
            .all(|p| p.key.anchors[0].1 >= 3500)
    );
    assert!(!view.controls.one_skip_words);
}

#[test]
#[ignore = "release-only grouping scan cost; excludes full O04 workload"]
fn grouping_inventory_cost_probe() {
    use std::{hint::black_box, time::Instant};
    let bank: VecDeque<_> = (0..128)
        .map(|i| {
            accent(
                1000 + i * 64,
                1000 + i * 64,
                [0.25, 0.5, 0.75, 1.][i as usize % 4],
            )
        })
        .collect();
    let peaks = [0, 32, 48, 64, 80, 96, 112, 144].map(|bin| {
        Some(Peak {
            bin,
            period_seconds: 0.125 * 2f64.powf(bin as f64 / 48.),
            support: 1. / 8.,
        })
    });
    let mut durations = Vec::with_capacity(6000);
    let mut last = blank();
    for i in 0..6100 {
        let mut view = blank();
        let start = Instant::now();
        admit(
            black_box(&bank),
            black_box(&peaks),
            1000,
            controls(),
            &mut view,
        );
        black_box(&view);
        if i >= 100 {
            durations.push(start.elapsed().as_secs_f64() * 1e6);
        }
        last = view;
    }
    durations.sort_by(f64::total_cmp);
    println!(
        "grouping_inventory_cost {}",
        json!({"calls":durations.len(),"capacity":128,"periods":8,"median_us":durations[3000],"p99_us":durations[5940],"max_us":durations[5999],"integer_cases":last.work.integer_cases,"word_cases":last.work.word_cases,"admitted_cases":last.work.admitted_cases,"ranking_comparisons":last.work.ranking_comparisons,"boundary_search_comparisons":last.work.boundary_search_comparisons,"step_comparisons":last.work.step_comparisons,"inventory_bytes":std::mem::size_of::<Inventory>(),"view_bytes":std::mem::size_of::<View>(),"proposal_bytes":std::mem::size_of::<Proposal>(),"full_O04":false,"scope":"one capacity128 grouping refresh with eight periods and nonzero admitted proposals; DSP, pair-grid, beam, both workers,64 Voice and device excluded"})
    );
}

#[test]
fn repeated_keys_do_not_fill_cap_and_cadence_uses_fractional_sample_boundaries() {
    let mut view = scan(&(1000..=9000).step_by(500).collect::<Vec<_>>(), controls());
    let before = view.proposals;
    for proposal in before.into_iter().flatten() {
        retain(proposal, &mut view);
    }
    assert_eq!(view.work.duplicate_retained_keys, 16);
    assert_eq!(
        view.proposals.map(|p| p.map(|p| p.key)),
        before.map(|p| p.map(|p| p.key))
    );
    let mut estimator = Estimator::new(group(), 128, 44101, 44101 * 32, 1024, 1. / 24.).unwrap();
    let mut inventory = Inventory::new(group(), 0, controls()).unwrap();
    estimator.advance(4410).unwrap();
    inventory.refresh(&estimator).unwrap();
    assert_eq!(inventory.view.unwrap().cadence_slot, 0);
    estimator.advance(4411).unwrap();
    assert!(inventory.refresh(&estimator).unwrap());
    assert_eq!(inventory.view.unwrap().cadence_slot, 1);
    estimator.advance(44101).unwrap();
    inventory.refresh(&estimator).unwrap();
    assert_eq!(inventory.view.unwrap().cadence_slot, 10);
    assert_eq!(inventory.view.unwrap().superseded_slots, 8);
}

#[test]
fn late_group_activation_does_not_claim_refresh_slots_before_it_existed() {
    let mut estimator = Estimator::new(group(), 128, 1000, 32000, 1024, 1. / 24.).unwrap();
    let mut inventory = Inventory::new(group(), 0, controls()).unwrap();
    estimator.advance(20132).unwrap();
    inventory.refresh(&estimator).unwrap();
    assert_eq!(inventory.view.unwrap().superseded_slots, 201);
    let child = Handle {
        generation: 3,
        ..group()
    };
    estimator.reset(child, 20132).unwrap();
    inventory.reset(child, 20132).unwrap();
    assert!(inventory.snapshot().is_none());
    estimator.advance(20144).unwrap();
    inventory.refresh(&estimator).unwrap();
    assert_eq!(inventory.view.unwrap().group, child);
    assert_eq!(inventory.view.unwrap().superseded_slots, 0);
    estimator.advance(20401).unwrap();
    inventory.refresh(&estimator).unwrap();
    assert_eq!(inventory.view.unwrap().superseded_slots, 2);
    let before = inventory.view.unwrap().refreshed_at;
    assert!(
        inventory
            .reset(
                Handle {
                    generation: 1,
                    ..group()
                },
                20401
            )
            .is_err()
    );
    assert_eq!(inventory.view.unwrap().refreshed_at, before);
}

#[test]
fn arrival_word_feature_masks_stale_or_incomplete_searches() {
    let mut view = scan(&[1000, 1500, 2250, 2750, 3500], controls());
    view.retained_span = Some((1000, 3500));
    view.period_source_end = Some(3516);
    view.period_available_end = Some(3532);
    let support = Some([952, 3516, 3532]);
    assert_eq!(view.word_indicator(96, support), Some(true));
    assert_eq!(view.word_indicator(96, Some([952, 4016, 4032])), None);
    view.proposals = [None; 16];
    assert_eq!(view.word_indicator(96, support), None);
    view.work.word_insufficient_endpoints = 0;
    assert_eq!(view.word_indicator(96, support), Some(false));
    view.capacity_evicted_through = Some(900);
    assert_eq!(view.word_indicator(96, support), None);
}

#[test]
fn rating_words_coalesce_period_aliases_and_preserve_original_observation_gaps() {
    let ends = [1000, 1250, 1750, 2000, 2500];
    let mut estimator = Estimator::new(group(), 128, 1000, 32000, 1024, 1. / 24.).unwrap();
    for (i, end) in ends.into_iter().enumerate() {
        // Ten missing samples leave both repetitions above 90 percent coverage.
        estimator
            .deliver(accent(end, end - if i > 0 { 10 } else { 0 }, 0.8), end + 32)
            .unwrap();
    }
    let peak = Peak {
        bin: 96,
        period_seconds: 0.5,
        support: 1.,
    };
    let mut view = blank();
    let mut peaks = [None; 8];
    peaks[0] = Some(peak);
    admit(&estimator.ledger.bank, &peaks, 1000, controls(), &mut view);
    view.retained_span = Some((1000, 2500));
    view.period_source_end = Some(2516);
    view.period_available_end = Some(2532);
    let original = *view
        .proposals
        .iter()
        .flatten()
        .find(|p| matches!(p.key.shape, Shape::Word { len: 2, .. }))
        .unwrap();
    view.proposals = [None; 16];
    view.proposals[0] = Some(original);
    let mut alias = original;
    alias.key.period_bin += 1;
    alias.mean_endpoint_weight = 0.4;
    view.proposals[1] = Some(alias);
    let words = view.words(&estimator);
    assert_eq!(words.iter().flatten().count(), 1);
    let word = words[0].unwrap();
    assert_eq!(word.symbols[..2], [2, 4]);
    close(word.support, 0.8);
    assert_eq!(word.observed_pairs, 0b110);
    // A newer accent makes the cached word stale before the next refresh.
    estimator.deliver(accent(2750, 2740, 0.8), 2782).unwrap();
    assert!(view.words(&estimator).iter().all(Option::is_none));
}
