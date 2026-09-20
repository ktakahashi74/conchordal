use super::{matcher::*, memory, transport::*};
use serde_json::Value;
use std::time::Instant;

#[test]
fn wide_transport_validates_every_relation_before_consuming_the_ticket() {
    let mut controller = Controller::new(0, 7, 1, 128);
    let mut relations = Vec::new();
    for id in 1..=80 {
        let identity = Identity { id, generation: 1 };
        controller.register(identity).unwrap();
        relations.push(Some(Relation {
            identity,
            supported: true,
            ambiguous: false,
            transformation: [Some(0.); 2],
        }));
    }
    let t = Ticket {
        model_version: 1,
        ..ticket()
    };
    controller.dispatch(t, t.issued_at).unwrap();
    let mut packet = Packet {
        ticket: t,
        completed_at: 125,
        relations: relations.into_boxed_slice(),
    };
    packet.relations[79].as_mut().unwrap().identity.generation = 2;
    assert!(matches!(
        controller.receive(&packet, 130),
        Err(Rejection::Retired)
    ));
    packet.relations[79].as_mut().unwrap().identity.generation = 1;
    assert_eq!(controller.receive(&packet, 130).unwrap().supported, 80);
    assert!(matches!(
        controller.receive(&packet, 130),
        Err(Rejection::Stale)
    ));
}

fn number(v: &Value) -> f64 {
    v.get("f64").map_or_else(
        || v.as_f64().unwrap(),
        |bits| f64::from_bits(bits.as_u64().unwrap()),
    )
}

fn descriptor(rows: &Value) -> memory::Descriptor {
    let mut knots = Vec::new();
    for row in rows.as_array().unwrap() {
        let mut k = Knot::default();
        for (d, value) in row["values"].as_array().unwrap().iter().enumerate() {
            if !value.is_null() {
                k.values[d] = number(value);
                k.mask |= 1 << d;
            }
        }
        k.time = number(&row["time"]);
        k.start = number(&row["start"]);
        k.end = number(&row["end"]);
        k.observed_sec = number(&row["observed_sec"]);
        k.raw_start = number(&row["raw_support_start"]);
        k.raw_end = number(&row["raw_support_end"]);
        k.available_end = number(&row["available_end"]);
        k.gap = u64::from(row["gap"].as_bool().unwrap());
        k.timing = u64::from(k.observed_sec > 0.0 && k.gap == 0 && k.mask != 0);
        knots.push(k);
    }
    let local_intervals = knots
        .iter()
        .enumerate()
        .map(|(i, k)| {
            (i > 0 && k.timing != 0 && knots[i - 1].timing != 0).then(|| k.time - knots[i - 1].time)
        })
        .collect();
    memory::Descriptor {
        knots,
        local_intervals,
    }
}

fn path(value: &Value) -> Vec<[i16; 3]> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|step| {
            [
                match step[0].as_str().unwrap() {
                    "pair" => 1,
                    "insert" => 2,
                    "delete" => 3,
                    _ => panic!(),
                },
                step[1].as_i64().unwrap_or(-1) as i16,
                step[2].as_i64().unwrap_or(-1) as i16,
            ]
        })
        .collect()
}

fn same(actual: f64, expected: &Value) {
    assert_eq!(
        actual.to_bits(),
        number(expected).to_bits(),
        "{actual} != {}",
        number(expected)
    );
}

fn fixtures() -> Value {
    serde_json::from_str(include_str!(
        "../../tests/fixtures/temporal_cognition/reference.json"
    ))
    .unwrap()
}

#[test]
fn frozen_python_dtw_and_coarse_results_are_bit_exact() {
    for (index, case) in fixtures()["dtw"].as_array().unwrap().iter().enumerate() {
        let cue = descriptor(&case["cue"]);
        let reference = descriptor(&case["reference"]);
        validate(&cue.knots, 1000.0).unwrap();
        validate(&reference.knots, 1000.0).unwrap();
        let scales = std::array::from_fn(|d| number(&case["scales"][d]));
        let config = Config {
            anchor: case["transform"]["anchor"].as_u64().unwrap() as u32,
            band: case["band"].as_i64().unwrap_or(-1) as i32,
            shift: number(&case["transform"]["applied"][0]),
            ratio: 2.0_f64.powf(number(&case["transform"]["applied"][1])),
            tempo_shift: number(&case["transform"]["applied"][1]),
            scales,
            insertion: number(&case["insertion"]),
            deletion: number(&case["deletion"]),
        };
        let mut output = Output::default();
        dtw(&cue.knots, &reference.knots, &config, &mut output).unwrap();
        let expected = &case["result"];
        assert_eq!(
            u64::from(output.cells),
            expected["dp_cells"].as_u64().unwrap(),
            "case {index}"
        );
        if expected["band_disconnected"] == true {
            assert!(!output.total.is_finite());
            continue;
        }
        same(output.total, &expected["total_cost"]);
        assert_eq!(
            output.path[..output.path_len as usize],
            path(&expected["path"]),
            "case {index}"
        );
        assert_eq!(
            u64::from(output.reference_start),
            expected["reference_start"].as_u64().unwrap()
        );
        assert_eq!(
            output.band_edge != 0,
            expected["band_edge_hit"].as_bool().unwrap()
        );
        for d in 0..10 {
            assert_eq!(
                u64::from(output.coordinate_count[d]),
                expected["coordinate_counts"][d].as_u64().unwrap()
            );
            if output.coordinate_count[d] > 0 {
                same(
                    output.coordinate_error[d] / f64::from(output.coordinate_count[d]),
                    &expected["coordinate_residuals"][d],
                );
            }
        }
        let mut anchors = [Anchor::default(); CAPACITY];
        let mut diagnostic = AnchorDiagnostic::default();
        coarse(
            &cue.knots,
            &reference.knots,
            &CoarseConfig {
                spacing: 4,
                limit: 32,
                bounds: [2.0; 2],
                grid: 1.0 / 16.0,
                scales,
            },
            &mut anchors,
            &mut diagnostic,
        )
        .unwrap();
        for (i, row) in case["coarse"].as_array().unwrap().iter().enumerate() {
            let actual = anchors[i];
            if row["score"]["cost"].is_null() {
                assert!(actual.cost.is_nan());
            } else {
                same(actual.cost, &row["score"]["cost"]);
            }
            assert_eq!(
                actual.bound_hit != 0,
                row["transform"]["bound_hit"].as_bool().unwrap()
            );
            for d in 0..2 {
                same(actual.applied[d], &row["transform"]["applied"][d]);
            }
        }
    }
}

#[test]
fn full_bounded_query_matches_frozen_python_ranking_masks_and_paths() {
    for case in fixtures()["queries"].as_array().unwrap() {
        let cue = descriptor(&case["query"]["knots"]);
        let episodes: Vec<_> = case["episodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| memory::Episode {
                identity: Identity {
                    id: e["episode_id"].as_u64().unwrap(),
                    generation: e["generation"].as_u64().unwrap(),
                },
                epoch: e["epoch"].as_u64().unwrap(),
                available_end: number(&e["available_end"]),
                first_observed_end: number(&e["first_observed_end"]),
                scales: [1.0; 10],
                descriptor: descriptor(&e["knots"]),
            })
            .collect();
        let query = memory::Query {
            descriptor: &cue,
            epoch: 1,
            end: number(&case["query"]["query_end"]),
            observed_end: 200.0,
            scales: [1.0; 10],
        };
        let report = memory::ordered_prefix(&query, &episodes).unwrap();
        let expected = &case["result"];
        assert_eq!(report.cutoff_tie, expected["cutoff_tie"].as_bool().unwrap());
        assert_eq!(report.dp_cells, expected["dp_cells"].as_u64().unwrap());
        assert_eq!(
            report.coarse.len(),
            expected["coarse_entries"].as_array().unwrap().len()
        );
        for (a, b) in report
            .coarse
            .iter()
            .zip(expected["coarse_entries"].as_array().unwrap())
        {
            assert_eq!(a.identity.id, b["episode_id"].as_u64().unwrap());
            if let Some(cost) = a.cost {
                same(cost, &b["cost"]);
            } else {
                assert!(b["cost"].is_null());
            }
        }
        assert_eq!(
            report.matches.len(),
            expected["matches"].as_array().unwrap().len()
        );
        for (a, b) in report
            .matches
            .iter()
            .zip(expected["matches"].as_array().unwrap())
        {
            assert_eq!(a.relation.identity.id, b["episode_id"].as_u64().unwrap());
            assert_eq!(
                a.relation.identity.generation,
                b["episode_generation"].as_u64().unwrap()
            );
            if let Some(cost) = a.cost {
                same(cost, &b["cost"]);
            } else {
                assert!(b["cost"].is_null());
            }
            assert_eq!(a.path, path(&b["path"]));
            let residuals = a.residuals.unwrap();
            let reference = &episodes
                .iter()
                .find(|e| e.identity == a.relation.identity)
                .unwrap()
                .descriptor;
            assert_eq!(
                residuals.observed as usize,
                cue.knots.iter().filter(|k| k.observed_sec > 0.).count()
            );
            let mut sums = [0.; 10];
            let mut counts = [0; 10];
            let mut edits = [0; 4];
            for step in &a.path {
                match step[0] {
                    1 => {
                        let q = &cue.knots[step[1] as usize];
                        let r = &reference.knots[step[2] as usize];
                        let mask = q.mask & r.mask;
                        edits[usize::from(mask == 0)] += 1;
                        for coordinate in 0..10 {
                            if mask & (1 << coordinate) != 0 {
                                let shift = if coordinate == 0 { a.applied[0] } else { 0. };
                                let delta = (q.values[coordinate] - r.values[coordinate] - shift)
                                    / query.scales[coordinate];
                                sums[coordinate] += delta * delta;
                                counts[coordinate] += 1;
                            }
                        }
                    }
                    2 => edits[2] += 1,
                    3 => edits[3] += 1,
                    _ => unreachable!(),
                }
            }
            assert_eq!(residuals.coordinate_count, counts);
            assert_eq!(
                [
                    residuals.matched,
                    residuals.missing,
                    residuals.inserted,
                    residuals.deleted
                ],
                edits
            );
            for (value, expected) in residuals.coordinate_squared_error.into_iter().zip(sums) {
                assert!((value - expected).abs() <= 1e-12 * (1. + expected.abs()));
            }
            assert_eq!(a.anchor.unwrap() as u64, b["anchor"].as_u64().unwrap());
            assert_eq!(a.relation.supported, b["supported"].as_bool().unwrap());
            assert_eq!(
                a.relation.ambiguous,
                b["ambiguous_cutoff"] == true || b["band_edge_hit"] == true
            );
        }
    }
}

fn sequence(values: &[Option<f64>], start: f64) -> memory::Descriptor {
    let knots: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, value)| {
            let start = start + i as f64;
            let end = start + 1.0;
            let mut values = [0.0; 10];
            values[0] = value.unwrap_or(0.0);
            Knot {
                values,
                mask: u64::from(value.is_some()),
                time: end,
                start,
                end,
                raw_start: start,
                raw_end: end,
                available_end: end,
                observed_sec: f64::from(value.is_some()),
                gap: u64::from(value.is_none()),
                timing: u64::from(value.is_some()),
                lineage_changed: 0,
            }
        })
        .collect();
    let local_intervals = vec![Some(1.0); knots.len()];
    memory::Descriptor {
        knots,
        local_intervals,
    }
}

fn episode(id: u64, values: &[Option<f64>]) -> memory::Episode {
    memory::Episode {
        identity: Identity { id, generation: 3 },
        epoch: 7,
        available_end: values.len() as f64,
        first_observed_end: values.len() as f64,
        scales: [1.0; 10],
        descriptor: sequence(values, 0.0),
    }
}

fn ticket() -> Ticket {
    Ticket {
        bus: 0,
        epoch: 7,
        model_version: 1,
        query: Identity {
            id: 1,
            generation: 9,
        },
        support_id: 11,
        support_start: 100,
        support_end: 110,
        supporting_audio_end: Some(112),
        available_at: 115,
        issued_at: 120,
        deadline: 160,
    }
}

type Model = fn(&memory::Query<'_>, &[memory::Episode]) -> Result<memory::Report, Error>;

#[test]
fn mr1_transport_keeps_prefix_and_optional_audio_endpoints_independent() {
    for audio in [None, Some(105), Some(110), Some(112)] {
        let t = Ticket {
            supporting_audio_end: audio,
            ..ticket()
        };
        let mut controller = Controller::new(t.bus, t.epoch, t.model_version, 256);
        controller.dispatch(t, t.issued_at).unwrap();
        let mut packet = Packet {
            ticket: t,
            completed_at: 125,
            relations: vec![None; RESULT_LIMIT].into_boxed_slice(),
        };
        packet.ticket.supporting_audio_end = Some(114);
        assert!(matches!(
            controller.receive(&packet, 130),
            Err(Rejection::Stale)
        ));
        packet.ticket = t;
        let receipt = controller.receive(&packet, 130).unwrap();
        assert_eq!(receipt.ticket.supporting_audio_end, audio);
        assert_eq!(receipt.ticket.support_end, 110);
        assert_eq!(receipt.received_at, 130);
        assert_eq!(receipt.unknown, 1);
    }
    for t in [
        Ticket {
            supporting_audio_end: Some(116),
            ..ticket()
        },
        Ticket {
            support_end: 116,
            supporting_audio_end: None,
            ..ticket()
        },
    ] {
        let mut controller = Controller::new(t.bus, t.epoch, t.model_version, 256);
        assert!(matches!(
            controller.dispatch(t, 120),
            Err(Rejection::Invalid)
        ));
        controller.dispatch(ticket(), 120).unwrap();
    }
}

#[test]
fn mr1_order_counterexample_and_shared_transport() {
    let cue = sequence(&[Some(0.0), Some(1.0), Some(0.0), Some(-1.0)], 100.0);
    let bank = [
        episode(1, &[Some(0.0), Some(1.0), Some(0.0), Some(-1.0)]),
        episode(2, &[Some(0.0), Some(-1.0), Some(0.0), Some(1.0)]),
    ];
    let query = memory::Query {
        descriptor: &cue,
        epoch: 7,
        end: 104.0,
        observed_end: 120.0,
        scales: [1.0; 10],
    };
    let mut costs = Vec::new();
    for (version, model) in [
        (1, memory::ordered_prefix as Model),
        (2, memory::orderless as Model),
    ] {
        let report = model(&query, &bank).unwrap();
        let mut controller = Controller::new(0, 7, version, 256);
        for e in &bank {
            controller.register(e.identity).unwrap();
        }
        let t = Ticket {
            model_version: version,
            support_end: 104,
            supporting_audio_end: Some(104),
            available_at: 104,
            ..ticket()
        };
        controller.dispatch(t, 120).unwrap();
        let packet = Packet {
            ticket: t,
            completed_at: 125,
            relations: report.matches.iter().map(|m| Some(m.relation)).collect(),
        };
        let receipt = controller.receive(&packet, 130).unwrap();
        assert_eq!(receipt.ticket, t);
        assert_eq!(receipt.received_at, 130);
        assert_eq!(receipt.supported, report.matches.len());
        assert!(matches!(
            controller.receive(&packet, 131),
            Err(Rejection::Stale)
        ));
        costs.push([
            report
                .matches
                .iter()
                .find(|m| m.relation.identity.id == 1)
                .unwrap()
                .cost
                .unwrap(),
            report
                .matches
                .iter()
                .find(|m| m.relation.identity.id == 2)
                .unwrap()
                .cost
                .unwrap(),
        ]);
        if version == 2 {
            assert!(
                report
                    .matches
                    .iter()
                    .all(|m| m.path.is_empty() && m.anchor.is_none())
            );
        }
    }
    assert_eq!(costs[0][0], 0.0);
    assert!(costs[0][1] > costs[0][0]);
    assert_eq!(costs[1], [0.0, 0.0]);
}

#[test]
fn bag_uses_whole_content_population_variance_and_stored_intervals() {
    let cue = sequence(&[Some(-1.0), Some(1.0)], 100.0);
    let bank = [episode(1, &[Some(0.0), Some(0.0)])];
    let query = memory::Query {
        descriptor: &cue,
        epoch: 7,
        end: 102.0,
        observed_end: 120.0,
        scales: [1.0; 10],
    };
    let report = memory::orderless(&query, &bank).unwrap();
    assert_eq!(report.matches[0].cost, Some(0.5));
    let mut long = sequence(&[Some(0.0); 12], 100.0);
    long.knots[11].values[0] = 1.0;
    long.local_intervals = vec![Some(0.5); 12];
    let bank = [episode(1, &[Some(0.0); 12])];
    let query = memory::Query {
        descriptor: &long,
        end: 112.0,
        ..query
    };
    let a = memory::orderless(&query, &bank).unwrap();
    assert!(a.matches[0].cost.unwrap() > 0.0);
    assert_eq!(a.matches[0].relation.transformation[1], Some(1.0));
    let original = a.matches[0].cost;
    long.knots.swap(0, 11);
    // Reassign acquisition metadata; the bag must not use it to restore order.
    for (i, k) in long.knots.iter_mut().enumerate() {
        k.start = 200.0 + i as f64;
        k.end = k.start + 1.0;
        k.time = k.end;
        k.raw_start = k.start;
        k.raw_end = k.end;
        k.available_end = k.end;
    }
    let b = memory::orderless(
        &memory::Query {
            descriptor: &long,
            end: 212.0,
            observed_end: 220.0,
            epoch: 7,
            scales: [1.0; 10],
        },
        &bank,
    )
    .unwrap();
    assert_eq!(b.matches[0].cost, original);
    assert_eq!(b.matches[0].relation.transformation[1], Some(1.0));
}

#[test]
fn both_models_keep_missing_unknown_and_known_zero_supported() {
    for model in [memory::ordered_prefix as Model, memory::orderless as Model] {
        for value in [None, Some(0.0)] {
            let cue = sequence(&[value; 4], 100.0);
            let bank = [episode(1, &[value; 4])];
            let result = model(
                &memory::Query {
                    descriptor: &cue,
                    epoch: 7,
                    end: 104.0,
                    observed_end: 120.0,
                    scales: [1.0; 10],
                },
                &bank,
            )
            .unwrap();
            assert_eq!(result.matches.is_empty(), value.is_none());
            if value.is_some() {
                assert!(result.matches[0].relation.supported);
            }
        }
    }
}

#[test]
fn receipt_checks_are_atomic_for_both_model_versions() {
    for version in [1, 2] {
        let t = Ticket {
            model_version: version,
            ..ticket()
        };
        let identity = Identity {
            id: u64::MAX - 1,
            generation: u64::MAX,
        };
        let relation = Relation {
            identity,
            supported: true,
            ambiguous: false,
            transformation: [None, None],
        };
        let mut controller = Controller::new(0, 7, version, 256);
        controller.register(identity).unwrap();
        controller.dispatch(t, 120).unwrap();
        let mut packet = Packet {
            ticket: t,
            completed_at: 125,
            relations: vec![None; RESULT_LIMIT].into_boxed_slice(),
        };
        packet.relations[0] = Some(relation);
        for mutated in [
            Ticket { bus: 1, ..t },
            Ticket { epoch: 8, ..t },
            Ticket {
                model_version: version + 1,
                ..t
            },
            Ticket {
                query: Identity {
                    generation: 10,
                    ..t.query
                },
                ..t
            },
            Ticket {
                support_id: 12,
                ..t
            },
            Ticket {
                supporting_audio_end: Some(113),
                ..t
            },
        ] {
            packet.ticket = mutated;
            assert!(matches!(
                controller.receive(&packet, 130),
                Err(Rejection::Stale)
            ));
        }
        packet.ticket = t;
        assert!(matches!(
            controller.receive(&packet, 124),
            Err(Rejection::Invalid)
        ));
        assert!(matches!(
            controller.receive(&packet, 161),
            Err(Rejection::Stale)
        ));
        packet.relations[1] = Some(Relation {
            identity: Identity {
                generation: 0,
                ..identity
            },
            ..relation
        });
        assert!(matches!(
            controller.receive(&packet, 130),
            Err(Rejection::Retired)
        ));
        packet.relations[1] = None;
        packet.relations[0] = Some(Relation {
            transformation: [Some(f64::NAN), None],
            ..relation
        });
        assert!(matches!(
            controller.receive(&packet, 130),
            Err(Rejection::Invalid)
        ));
        packet.relations[0] = Some(relation);
        let receipt = controller.receive(&packet, 130).unwrap();
        assert_eq!(receipt.supported, 1);
        assert_eq!(receipt.unknown, 0);
        assert_eq!(receipt.ambiguous, 0);
        controller.retire(identity);
        assert_eq!(controller.register(identity), Err(Rejection::Stale));
        let next = Ticket {
            query: Identity { id: 2, ..t.query },
            issued_at: 131,
            ..t
        };
        controller.dispatch(next, 131).unwrap();
        packet.ticket = next;
        packet.completed_at = 132;
        assert!(matches!(
            controller.receive(&packet, 133),
            Err(Rejection::Retired)
        ));
        controller.restart(8, version).unwrap();
        assert!(matches!(
            controller.receive(&packet, 134),
            Err(Rejection::Stale)
        ));
        assert_eq!(controller.restart(8, version), Err(Rejection::Stale));
    }
}

#[test]
fn bounded_controller_rejects_future_busy_overflow_and_duplicate_inputs() {
    let mut c = Controller::new(0, 7, 1, 256);
    for id in 1..=ACTIVE_LIMIT as u64 {
        c.register(Identity { id, generation: 0 }).unwrap();
    }
    assert_eq!(
        c.register(Identity {
            id: 300,
            generation: 0
        }),
        Err(Rejection::Capacity)
    );
    assert_eq!(
        c.register(Identity {
            id: 1,
            generation: 1
        }),
        Err(Rejection::Invalid)
    );
    assert_eq!(
        c.dispatch(
            Ticket {
                available_at: 121,
                ..ticket()
            },
            120
        ),
        Err(Rejection::Invalid)
    );
    c.dispatch(ticket(), 120).unwrap();
    let next = Ticket {
        query: Identity {
            id: 2,
            ..ticket().query
        },
        ..ticket()
    };
    assert_eq!(c.dispatch(next, 120), Err(Rejection::Busy));
    let next = Ticket {
        issued_at: 161,
        deadline: 180,
        ..next
    };
    c.dispatch(next, 161).unwrap();
    let p = Packet {
        ticket: next,
        completed_at: 162,
        relations: vec![None; RESULT_LIMIT].into_boxed_slice(),
    };
    assert_eq!(c.receive(&p, 163).unwrap().unknown, 1);
    assert_eq!(c.dispatch(next, 161), Err(Rejection::Stale));
}

#[test]
fn numerical_boundaries_and_extreme_arithmetic_reject_without_unsafe_access() {
    let mut cue = sequence(&[Some(0.0); 2], 100.0);
    assert_eq!(validate(&cue.knots, 101.0), Err(Error::InvalidInput));
    cue.knots[1].lineage_changed = 1;
    assert_eq!(validate(&cue.knots, 120.0), Err(Error::InvalidInput));
    cue.knots[1].lineage_changed = 0;
    cue.knots[0].mask = 1024;
    assert_eq!(validate(&cue.knots, 120.0), Err(Error::InvalidInput));
    assert_eq!(
        validate(&vec![Knot::default(); 129], 120.0),
        Err(Error::InvalidInput)
    );
    let a = sequence(&[Some(f64::MAX); 2], 100.0);
    let b = sequence(&[Some(-f64::MAX); 2], 0.0);
    let mut anchors = [Anchor::default(); CAPACITY];
    let mut diag = AnchorDiagnostic::default();
    assert_eq!(
        coarse(
            &a.knots,
            &b.knots,
            &CoarseConfig {
                spacing: 4,
                limit: 32,
                bounds: [2.0; 2],
                grid: 1.0 / 16.0,
                scales: [1.0; 10]
            },
            &mut anchors,
            &mut diag
        ),
        Err(Error::NumericalRange)
    );
}

#[test]
fn migrated_power_counterexample_and_rounding_boundaries() {
    let value = f64::from_bits(0xbffe2cb19a44a7ec);
    let a = sequence(&[Some(0.0), Some(value)], 100.0);
    let b = sequence(&[Some(0.0); 2], 0.0);
    let mut out = Output::default();
    let mut config = Config {
        anchor: 0,
        band: -1,
        shift: 0.0,
        ratio: 1.0,
        tempo_shift: 0.0,
        scales: [1.0; 10],
        insertion: 4.0,
        deletion: 4.0,
    };
    config.scales[0] = value.abs();
    dtw(&a.knots, &b.knots, &config, &mut out).unwrap();
    assert_eq!(out.motion_count, 1);
    assert_eq!(out.motion_error[0].to_bits(), 0x400c740b6d82ad27);
    assert_ne!((value * value).to_bits(), out.motion_error[0].to_bits());
    let mut anchors = [Anchor::default(); CAPACITY];
    let mut diagnostic = AnchorDiagnostic::default();
    for (pitch, rounded, bound) in [
        (-0.0, 0.0, false),
        (-0.03125, -0.0625, false),
        (0.03125, 0.0, false),
        (-1.96875, -2.0, true),
        (1.96875, 1.9375, false),
    ] {
        let a = sequence(&[Some(pitch); 2], 100.0);
        coarse(
            &a.knots,
            &b.knots,
            &CoarseConfig {
                spacing: 4,
                limit: 32,
                bounds: [2.0; 2],
                grid: 1.0 / 16.0,
                scales: [1.0; 10],
            },
            &mut anchors,
            &mut diagnostic,
        )
        .unwrap();
        assert_eq!(anchors[0].applied[0].to_bits(), f64::to_bits(rounded));
        assert_eq!(anchors[0].bound_hit != 0, bound);
    }
}

#[test]
fn whole_bag_agrees_with_independent_statistics_oracle_and_permutations() {
    for case in fixtures()["bags"].as_array().unwrap() {
        let mut cue = descriptor(&case["cue"]);
        let mut reference = descriptor(&case["reference"]);
        for (d, values) in [&mut cue, &mut reference]
            .into_iter()
            .zip(case["intervals"].as_array().unwrap())
        {
            d.local_intervals = values
                .as_array()
                .unwrap()
                .iter()
                .map(|v| (!v.is_null()).then(|| number(v)))
                .collect();
        }
        let scales = std::array::from_fn(|d| number(&case["scales"][d]));
        let end = reference.knots.last().unwrap().end;
        let bank = [memory::Episode {
            descriptor: reference,
            scales,
            first_observed_end: end,
            available_end: end,
            ..episode(1, &[])
        }];
        let check = |cue: &memory::Descriptor| {
            let query = memory::Query {
                descriptor: cue,
                epoch: 7,
                end: 120.0,
                observed_end: 120.0,
                scales,
            };
            let report = memory::orderless(&query, &bank).unwrap();
            assert_eq!(
                report.matches.len(),
                case["matches"].as_array().unwrap().len()
            );
            for (actual, expected) in report
                .matches
                .iter()
                .zip(case["matches"].as_array().unwrap())
            {
                let cost = number(&expected["cost"]);
                assert!((actual.cost.unwrap() - cost).abs() <= 8e-15 * cost.abs().max(1.0));
                for d in 0..2 {
                    assert_eq!(actual.applied[d], number(&expected["applied"][d]));
                    assert_eq!(
                        actual.relation.transformation[d].is_some(),
                        expected["identified"][d].as_bool().unwrap()
                    );
                }
                assert!(actual.path.is_empty() && actual.anchor.is_none());
            }
            report
                .matches
                .iter()
                .map(|m| m.cost.map(f64::to_bits))
                .collect::<Vec<_>>()
        };
        let original = check(&cue);
        let values: Vec<_> = cue.knots.iter().map(|k| (k.values, k.mask)).rev().collect();
        for (k, (values, mask)) in cue.knots.iter_mut().zip(values) {
            k.values = values;
            k.mask = mask;
        }
        cue.local_intervals.reverse();
        assert_eq!(check(&cue), original);
    }
}

#[test]
fn model_inputs_reject_changed_scales_future_support_and_aliases() {
    for model in [memory::ordered_prefix as Model, memory::orderless as Model] {
        let cue = sequence(&[Some(0.0); 2], 100.0);
        let query = memory::Query {
            descriptor: &cue,
            epoch: 7,
            end: 102.0,
            observed_end: 120.0,
            scales: [1.0; 10],
        };
        let mut bank = [episode(1, &[Some(0.0); 2]), episode(2, &[Some(0.0); 2])];
        bank[1].scales[0] = 2.0;
        assert!(matches!(model(&query, &bank), Err(Error::InvalidInput)));
        bank[1].scales[0] = 1.0;
        bank[1].identity = bank[0].identity;
        assert!(matches!(model(&query, &bank), Err(Error::InvalidInput)));
        bank[1].identity.id = 2;
        assert!(matches!(
            model(
                &memory::Query {
                    end: 101.0,
                    ..query
                },
                &bank
            ),
            Err(Error::InvalidInput)
        ));
        bank[1].descriptor.knots[1].available_end = 121.0;
        assert!(matches!(model(&query, &bank), Err(Error::InvalidInput)));
    }
}

#[test]
fn small_unbanded_dtw_agrees_with_exhaustive_path_costs() {
    fn enumerate(a: &[Knot], b: &[Knot], i: usize, j: usize, cost: f64, best: &mut f64) {
        if i == a.len() {
            if j > 0 {
                *best = best.min(cost);
            }
            return;
        }
        if cost > *best {
            return;
        }
        enumerate(a, b, i + 1, j, cost + 1.0, best);
        if j < b.len() {
            let residual = if a[i].mask & b[j].mask == 0 {
                0.0
            } else {
                (a[i].values[0] - b[j].values[0]).powi(2)
            };
            enumerate(a, b, i + 1, j + 1, cost + residual, best);
            enumerate(a, b, i, j + 1, cost + 1.0, best);
        }
    }
    for pattern in 0..81usize {
        let mut code = pattern;
        let values: Vec<_> = (0..4)
            .map(|_| {
                let x = [None, Some(0.0), Some(1.0)][code % 3];
                code /= 3;
                x
            })
            .collect();
        let a = sequence(&values[..2], 100.0);
        let b = sequence(&values[1..], 0.0);
        let mut best = f64::INFINITY;
        for j in 0..b.knots.len() {
            enumerate(&a.knots, &b.knots, 0, j, 0.0, &mut best);
        }
        let mut out = Output::default();
        dtw(
            &a.knots,
            &b.knots,
            &Config {
                anchor: 0,
                band: -1,
                shift: 0.0,
                ratio: 1.0,
                tempo_shift: 0.0,
                scales: [1.0; 10],
                insertion: 1.0,
                deletion: 1.0,
            },
            &mut out,
        )
        .unwrap();
        assert_eq!(out.total, best);
    }
}

#[test]
#[ignore = "release-only M0 resource assay; set CONCHORDAL_M0_REPORT to a new file"]
fn resource_assay() {
    if cfg!(debug_assertions) {
        panic!("run resource assay with --release");
    }
    let mut rows = Vec::new();
    for coordinates in [1, 10] {
        let mut cue = sequence(&[Some(0.0); CAPACITY], 1000.0);
        let mut bank: Vec<_> = (1..=memory::EPISODES)
            .map(|id| episode(id as u64, &[Some(0.0); CAPACITY]))
            .collect();
        for (e, descriptor) in std::iter::once(&mut cue)
            .chain(bank.iter_mut().map(|e| &mut e.descriptor))
            .enumerate()
        {
            for (i, k) in descriptor.knots.iter_mut().enumerate() {
                k.mask = (1 << coordinates) - 1;
                for d in 0..coordinates {
                    k.values[d] = ((i * 11 + e * 7 + d * 3) % 61) as f64 / 128.0;
                }
            }
        }
        let query = memory::Query {
            descriptor: &cue,
            epoch: 7,
            end: 1128.0,
            observed_end: 1200.0,
            scales: [1.0; 10],
        };
        for (name, model) in [
            ("ordered", memory::ordered_prefix as Model),
            ("memory_orderless", memory::orderless as Model),
        ] {
            let mut times = Vec::with_capacity(200);
            let mut match_count = 0;
            for i in 0..220 {
                let start = Instant::now();
                let result = std::hint::black_box(
                    model(std::hint::black_box(&query), std::hint::black_box(&bank)).unwrap(),
                );
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(result.coarse.len(), 256);
                assert!(result.matches.len() <= memory::MATCHES);
                match_count = result.matches.len();
                if i >= 20 {
                    times.push(ms);
                }
            }
            times.sort_by(f64::total_cmp);
            rows.push(serde_json::json!({"model":name,"coordinates":coordinates,"matches":match_count,
                "measurements":times,"median_ms":times[100],"p99_ms":times[197],"max_ms":times[199]}));
        }
    }
    let cue = sequence(&[Some(0.0), Some(1.0), Some(0.0), Some(-1.0)], 100.0);
    let bank = [
        episode(1, &[Some(0.0), Some(1.0), Some(0.0), Some(-1.0)]),
        episode(2, &[Some(0.0), Some(-1.0), Some(0.0), Some(1.0)]),
    ];
    let query = memory::Query {
        descriptor: &cue,
        epoch: 7,
        end: 104.0,
        observed_end: 120.0,
        scales: [1.0; 10],
    };
    let mut exchange = Vec::new();
    for (version, model) in [
        (1, memory::ordered_prefix as Model),
        (2, memory::orderless as Model),
    ] {
        let mut controller = Controller::new(0, 7, version, 256);
        for e in &bank {
            controller.register(e.identity).unwrap();
        }
        // Samples at 1 Hz in this small contract fixture; no real audio inferred.
        let t = Ticket {
            model_version: version,
            support_end: 104,
            supporting_audio_end: Some(104),
            available_at: 104,
            ..ticket()
        };
        let start = Instant::now();
        controller.dispatch(t, 120).unwrap();
        let result = model(&query, &bank).unwrap();
        let packet = Packet {
            ticket: t,
            completed_at: 125,
            relations: result.matches.iter().map(|m| Some(m.relation)).collect(),
        };
        let delivered = std::hint::black_box(&packet).clone();
        let receipt = controller
            .receive(std::hint::black_box(&delivered), 130)
            .unwrap();
        assert_eq!(std::hint::black_box(packet.ticket), delivered.ticket);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        exchange.push(serde_json::json!({"model_version":version,"wall_ms":elapsed,"packet_copy_bytes":std::mem::size_of::<Packet>() + packet.relations.len()*std::mem::size_of::<Option<Relation>>(),
            "input":{"cue":[0,1,0,-1],"episodes":[[0,1,0,-1],[0,-1,0,1]],"support":[100,104],"issued_at":120,"completed_at":125,"received_at":130},
            "receipt":{"supported":receipt.supported,"unknown":receipt.unknown,"ambiguous":receipt.ambiguous},
            "matches":result.matches.iter().map(|m|serde_json::json!({"episode":m.relation.identity.id,"cost":m.cost,"transformation":m.relation.transformation,"anchor":m.anchor,"path":m.path})).collect::<Vec<_>>() }));
    }
    let report = serde_json::json!({"schema":"m0-rust-resource-v1","scope":"single caller synthetic numerical query; bank creation excluded; not O04", "rows":rows,"mr1_exchange":exchange,
        "physical_layout_bytes":{"knot":std::mem::size_of::<Knot>(),"dp_output":std::mem::size_of::<Output>(),"anchors":CAPACITY*std::mem::size_of::<Anchor>(),"controller":std::mem::size_of::<Controller>(),"packet_header":std::mem::size_of::<Packet>(),"packet_baseline_64_relations":std::mem::size_of::<Packet>() + RESULT_LIMIT*std::mem::size_of::<Option<Relation>>()},
        "limitations":["model report allocates vectors and ordered paths","no allocator/RSS census","no device/two-worker/64-Voice load","no cognition/human evidence"]});
    let output = std::env::var_os("CONCHORDAL_M0_REPORT").expect("CONCHORDAL_M0_REPORT required");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    serde_json::to_writer_pretty(file, &report).unwrap();
}
