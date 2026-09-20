use super::*;
use serde_json::Value;

fn number(v: &Value) -> f64 {
    f64::from_bits(v["f64"].as_u64().expect("bit-preserving reference scalar"))
}

fn close(a: f64, b: f64) {
    assert!(
        a == b || (a - b).abs() <= 1e-11 * a.abs().max(b.abs()).max(1.),
        "{a:?} != {b:?}"
    );
}

fn assignments(v: &Value) -> Vec<(u64, f64)> {
    let mut rows: Vec<_> = v
        .as_object()
        .unwrap()
        .iter()
        .map(|(h, v)| (h.parse().unwrap(), number(v)))
        .collect();
    rows.sort_by_key(|(h, _)| *h);
    rows
}

#[test]
fn native_retention_matches_the_chronological_python_oracle() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/temporal_cognition/retention.json"
    ))
    .unwrap();
    let mut operations = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let mut clock = Clock::new(
            1,
            100,
            10,
            0,
            case["clock_capacity"].as_u64().unwrap() as usize,
        )
        .unwrap();
        for r in case["acquisitions"].as_array().unwrap() {
            let intervals: Vec<_> = r["known_sample_intervals"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| (r[0].as_u64().unwrap(), r[1].as_u64().unwrap()))
                .collect();
            clock
                .observe(super::super::clock::Acquisition {
                    epoch: 1,
                    start: r["sample_start"].as_u64().unwrap(),
                    end: r["sample_end"].as_u64().unwrap(),
                    available: r["sample_end"].as_u64().unwrap(),
                    cut: r["sample_end"].as_u64().unwrap(),
                    observed: r["observed"].as_bool().unwrap(),
                    intervals: &intervals,
                })
                .unwrap();
        }
        let mut memory = Retention::new(
            &clock,
            Parameters {
                tau: 20.,
                kappa: 4.,
                strength_max: 2.,
                r_max: number(&case["r_max"]),
            },
            4,
            case["rate_capacity"].as_u64().unwrap() as usize,
        )
        .unwrap();
        let buffers = [memory.records.as_ptr(), memory.scratch.as_ptr()];
        let receipts = [memory.last_write.as_ptr(), memory.write_scratch.as_ptr()];
        let rate_values = memory.window.values.as_ptr();
        let scores = assignments(&case["scores"]);
        let bias = number(&case["bias"]);
        let mut frozen = None;
        for op in case["operations"].as_array().unwrap() {
            match op["kind"].as_str().unwrap() {
                "write" => {
                    let w = &op["write"];
                    let s = &w["support"];
                    let assigned = assignments(&s["episodes"]);
                    let c = &w["coarse_snapshot"];
                    let mut entries = Vec::new();
                    if !c.is_null() {
                        entries = c["entries"]
                            .as_object()
                            .unwrap()
                            .iter()
                            .map(|(h, v)| (h.parse().unwrap(), (!v.is_null()).then(|| number(v))))
                            .collect();
                        entries.sort_by_key(|(h, _)| *h);
                    }
                    let new: Vec<_> = op["new_handles"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|h| h.as_u64().unwrap())
                        .collect();
                    let accepted = memory
                        .apply(
                            &clock,
                            &Write {
                                epoch: w["epoch"].as_u64().unwrap(),
                                sequence: w["sequence"].as_u64().unwrap(),
                                occurrence_id: w["occurrence_id"].as_u64().unwrap(),
                                support_id: w["support_id"].as_u64().unwrap(),
                                start: number(&w["start"]),
                                end: number(&w["support_end"]),
                                committed: number(&w["committed_at"]),
                                delivered: number(&w["delivered_at"]),
                                available: number(&s["available_support"]),
                                unknown: number(&s["unknown_episode_support"]),
                                unassigned: number(&s["unassigned_available_support"]),
                                assignments: &assigned,
                                coarse: (!c.is_null()).then(|| Coarse {
                                    epoch: c["epoch"].as_u64().unwrap(),
                                    generation: c["generation"].as_u64().unwrap(),
                                    query_id: c["query_id"].as_u64().unwrap(),
                                    occurrence_id: c["occurrence_id"].as_u64().unwrap(),
                                    support_id: c["support_id"].as_u64().unwrap(),
                                    support_end: number(&c["support_end"]),
                                    supporting_audio_end: Some(number(&c["supporting_audio_end"])),
                                    available_end: number(&c["available_end"]),
                                    entries: &entries,
                                }),
                            },
                            number(&op["cut"]),
                            &new,
                        )
                        .unwrap();
                    assert_eq!(accepted, op["accepted"].as_bool().unwrap());
                }
                "assay" => {
                    frozen = Some(
                        memory
                            .assay_snapshot(&clock, number(&op["target_start"]))
                            .unwrap(),
                    )
                }
                "evict" => assert_eq!(
                    memory.evict(op["slot"].as_u64().unwrap() as usize).unwrap(),
                    op["handle"].as_u64()
                ),
                other => panic!("unknown oracle operation {other}"),
            }
            for (slot, r) in memory.records.iter().enumerate() {
                let actual = serde_json::to_value(r).unwrap();
                for (key, expected) in op["records"][slot].as_object().unwrap() {
                    if expected.is_object() {
                        close(actual[key].as_f64().unwrap(), number(expected));
                    } else {
                        assert_eq!(&actual[key], expected, "metadata {key}");
                    }
                }
                let value = memory.availability(&clock, slot, 4.).unwrap();
                let expected = &op["availability"][slot];
                if let Some(value) = value {
                    let actual = serde_json::to_value(value).unwrap();
                    for (key, expected) in expected.as_object().unwrap() {
                        if expected.is_object() {
                            let scalar = match key.as_str() {
                                "log_lower" => value.log_lower,
                                "gap_interference_upper" => value.gap_interference_upper,
                                _ => actual[key].as_f64().unwrap(),
                            };
                            close(scalar, number(expected));
                        } else {
                            assert_eq!(&actual[key], expected, "availability {key}");
                        }
                    }
                } else {
                    assert!(expected.is_null());
                }
                close(memory.window.totals[slot], number(&op["totals"][slot]));
            }
            let rate = memory.window.snapshot();
            let actual = memory.recognition(&clock, &scores, bias, 4.).unwrap();
            close(actual.lower, number(&op["recognition"]["lower"]));
            close(actual.upper, number(&op["recognition"]["upper"]));
            assert_eq!(
                actual.supported_matches as u64,
                op["recognition"]["supported_matches"].as_u64().unwrap()
            );
            if let Some(frozen) = &frozen {
                let actual = frozen
                    .recognition(1, &scores, bias, number(&case["query_end"]))
                    .unwrap();
                close(actual.lower, number(&op["frozen_recognition"]["lower"]));
                close(actual.upper, number(&op["frozen_recognition"]["upper"]));
                assert_eq!(
                    actual.supported_matches as u64,
                    op["frozen_recognition"]["supported_matches"]
                        .as_u64()
                        .unwrap()
                );
            }
            assert_eq!(rate.envelope_invalid, op["rate_invalid"].as_bool().unwrap());
            assert_eq!(
                rate.envelope_unverified,
                op["rate_unverified"].as_bool().unwrap()
            );
            assert_eq!(
                rate.rate_history_unknown,
                op["rate_history_unknown"].as_bool().unwrap()
            );
            assert_eq!(
                memory.window.count as u64,
                op["rate_count"].as_u64().unwrap()
            );
            assert_eq!(rate.overflow_count, op["rate_losses"].as_u64().unwrap());
            assert_eq!(
                memory.eviction_candidate(&clock, 4.).unwrap(),
                op["eviction_candidate"].as_u64().map(|i| i as usize)
            );
            assert_eq!(memory.sequence, op["sequence"].as_u64().unwrap());
            assert_eq!(memory.replays, op["replays"].as_u64().unwrap());
            assert_eq!(memory.evictions, op["evictions"].as_u64().unwrap());
            assert!(
                buffers.contains(&memory.records.as_ptr())
                    && buffers.contains(&memory.scratch.as_ptr())
            );
            assert!(
                receipts.contains(&memory.last_write.as_ptr())
                    && receipts.contains(&memory.write_scratch.as_ptr())
            );
            assert_eq!(rate_values, memory.window.values.as_ptr());
            operations += 1;
        }
    }
    assert_eq!(operations, 56);
    assert_eq!(std::mem::size_of::<Record>(), 112);
    assert_eq!(std::mem::size_of::<Row>(), 16);
}

#[test]
fn invalid_writes_and_backdated_queries_cannot_change_retention() {
    let mut clock = Clock::new(1, 100, 10, 0, 128).unwrap();
    for i in 0..20 {
        clock
            .observe(super::super::clock::Acquisition {
                epoch: 1,
                start: i * 10,
                end: (i + 1) * 10,
                available: (i + 1) * 10,
                cut: (i + 1) * 10,
                observed: true,
                intervals: &[(i * 10, (i + 1) * 10)],
            })
            .unwrap();
    }
    let mut memory = Retention::new(
        &clock,
        Parameters {
            tau: 20.,
            kappa: 4.,
            strength_max: 2.,
            r_max: 10.,
        },
        1,
        4,
    )
    .unwrap();
    let mut write = Write {
        epoch: 1,
        sequence: 1,
        occurrence_id: 10,
        support_id: 100,
        start: 0.,
        end: 0.1,
        committed: 0.6,
        delivered: 0.6,
        available: 1.,
        unknown: 0.,
        unassigned: 0.,
        assignments: &[(10, 1.)],
        coarse: None,
    };
    memory.apply(&clock, &write, 2., &[10]).unwrap();
    let original = memory.records.to_vec();
    let receipt = memory.last_write.clone();
    let rate = serde_json::to_value(memory.window.snapshot()).unwrap();
    write.start = 0.01;
    assert!(memory.apply(&clock, &write, 2., &[]).is_err());
    write.start = 0.;
    write.sequence = 3;
    assert!(memory.apply(&clock, &write, 2., &[]).is_err());
    write.sequence = 2;
    write.epoch = 2;
    assert!(memory.apply(&clock, &write, 2., &[]).is_err());
    write.epoch = 1;
    write.available = 0.5;
    assert!(memory.apply(&clock, &write, 2., &[]).is_err());
    write.available = 1.;
    write.end = 0.05;
    assert!(memory.apply(&clock, &write, 2., &[]).is_err());
    write.end = 0.1;
    write.assignments = &[(20, 1.)];
    assert!(memory.apply(&clock, &write, 2., &[20]).is_err());
    write.assignments = &[(10, 1.)];
    assert!(memory.apply(&clock, &write, 1., &[]).is_err());
    assert_eq!(memory.records.as_ref(), original);
    assert_eq!(memory.last_write, receipt);
    assert_eq!(
        serde_json::to_value(memory.window.snapshot()).unwrap(),
        rate
    );
    assert_eq!(memory.sequence, 1);
    assert_eq!(memory.replays, 0);
    assert_eq!(memory.cut, 2.);
    assert!(memory.availability(&clock, 0, 1.).is_err());
    assert!(memory.recognition(&clock, &[], 0., 1.).is_err());
    assert!(memory.assay_snapshot(&clock, 1.).is_err());
    let frozen = memory.assay_snapshot(&clock, 2.).unwrap();
    let before = frozen.recognition(1, &[(10, 0.)], 0., 2.25).unwrap();
    clock
        .observe(super::super::clock::Acquisition {
            epoch: 1,
            start: 200,
            end: 210,
            available: 210,
            cut: 210,
            observed: false,
            intervals: &[],
        })
        .unwrap();
    assert!(memory.availability(&clock, 0, 2.).is_err());
    assert!(memory.assay_snapshot(&clock, 2.).is_err());
    let after = frozen.recognition(1, &[(10, 0.)], 0., 2.25).unwrap();
    assert_eq!(
        serde_json::to_value(before).unwrap(),
        serde_json::to_value(after).unwrap()
    );
    assert!(frozen.recognition(2, &[(10, 0.)], 0., 2.25).is_err());
    assert!(frozen.recognition(1, &[(10, 0.)], 0., 1.9).is_err());
    write.start = 2.;
    write.end = 2.1;
    write.committed = 2.1;
    write.delivered = 2.1;
    assert!(memory.apply(&clock, &write, 2.1, &[]).is_err());
    assert_eq!(memory.records.as_ref(), original);
}

#[test]
fn rate_capacities_match_independent_retained_event_sums_and_generation_births() {
    for capacity in [72, 144, 288, 512] {
        let mut window = RateWindow::new(3, capacity).unwrap();
        let mut births = [1, 1, 1];
        let mut events = Vec::new();
        let mut invalid = false;
        let before = window.values.as_ptr();
        for sequence in 1..=900 {
            let time = sequence as f64 * 0.002;
            if sequence == 233 {
                births[0] = sequence;
                window.totals[0] = 0.;
            }
            let increments = if sequence % 5 == 0 {
                [0.; 3]
            } else {
                [0.01 * (sequence % 3) as f64, 0.02, 0.01]
            };
            if increments.iter().any(|v| *v > 0.) {
                events.push((time, sequence, increments));
            }
            let actual = window
                .observe(time, sequence, &increments, &births, 3.125)
                .unwrap();
            let retained: Vec<_> = events
                .iter()
                .rev()
                .filter(|(t, _, _)| *t > time - 1.)
                .take(capacity)
                .collect();
            for (slot, &birth) in births.iter().enumerate() {
                let expected: f64 = retained
                    .iter()
                    .filter(|(_, s, _)| *s >= birth)
                    .map(|(_, _, v)| v[slot])
                    .sum();
                close(window.totals[slot], expected);
                invalid |= expected > 3.125;
            }
            assert_eq!(actual.envelope_invalid, invalid);
            assert_eq!(window.count, retained.len());
            assert_eq!(window.values.as_ptr(), before);
        }
        let lost = window.losses;
        let last = window.observe(20., 901, &[0.; 3], &births, 3.125).unwrap();
        assert_eq!(window.count, 0);
        assert!(!last.rate_history_unknown);
        assert_eq!(last.envelope_unverified, lost > 0);
        assert_eq!(last.overflow_count, lost);
        assert!(last.largest_retained_one_second_increment < 1e-11);
        let snapshot = serde_json::to_value(last).unwrap();
        assert!(window.observe(19., 902, &[0.; 3], &births, 3.125).is_err());
        assert!(
            window
                .observe(21., 902, &[f64::NAN, 0., 0.], &births, 3.125)
                .is_err()
        );
        assert_eq!(serde_json::to_value(window.snapshot()).unwrap(), snapshot);
    }
}

#[test]
fn recognition_floor_preserves_true_zero_and_remains_stable_at_extreme_scores() {
    assert_eq!(
        probability([(f64::NEG_INFINITY, 1e300)].into_iter(), 0.),
        0.
    );
    assert_eq!(probability([].into_iter(), -1e300), 0.);
    assert_eq!(probability([(0., 1e300)].into_iter(), -1e300), 1.);
    assert_eq!(probability([(0., -1e300)].into_iter(), 1e300), 0.);
    assert!(probability([(-10000., 700.)].into_iter(), 0.) > 0.99);
    close(probability([(0., 0.), (0., 0.)].into_iter(), 0.), 2. / 3.);
}
