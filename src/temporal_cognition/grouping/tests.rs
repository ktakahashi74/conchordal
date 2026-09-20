use super::*;

fn handle(generation: u64) -> Handle {
    Handle {
        bus: 0,
        epoch: 2,
        generation,
    }
}

fn frame(a: f64, b: Option<f64>) -> [Option<Envelope>; 8] {
    let mut out = [None; 8];
    out[0] = Some(Envelope {
        handle: handle(1),
        log_envelope: a,
    });
    out[1] = b.map(|b| Envelope {
        handle: handle(2),
        log_envelope: b,
    });
    out
}

#[test]
fn epoch_clipping_and_minimum_pair_count_are_separate_from_coverage() {
    let mut w = Window::new(0, 2, 500, 10, 250, 8, 0.9).unwrap();
    for i in 1..=7 {
        w.push(500 + i * 10, frame(i as f64, Some(2. * i as f64 + 3.)))
            .unwrap();
    }
    let before = w.correlations(&[handle(1), handle(2)]).unwrap();
    assert_eq!((before.window_start, before.window_end), (500, 570));
    assert_eq!(before.paired_samples[0][1], 70);
    assert_eq!(before.paired_hops[0][1], 7);
    assert!(before.values[0][1].is_none());
    w.push(580, frame(8., Some(19.))).unwrap();
    let ready = w.correlations(&[handle(1), handle(2)]).unwrap();
    assert!((ready.values[0][1].unwrap() - 1.).abs() < 1e-14);
    assert_eq!(ready.paired_hops[0][1], 8);
}

#[test]
fn missing_samples_are_not_time_compressed_and_ninety_percent_is_inclusive() {
    for missing in [vec![2], vec![2, 3]] {
        for skip in [false, true] {
            let mut w = Window::new(0, 2, 0, 10, 100, 4, 0.9).unwrap();
            for i in 1..=10 {
                if missing.contains(&i) && skip {
                    continue;
                }
                w.push(
                    i * 10,
                    frame(i as f64, (!missing.contains(&i)).then_some(-(i as f64))),
                )
                .unwrap();
            }
            let result = w.correlations(&[handle(1), handle(2)]).unwrap();
            assert_eq!(result.paired_samples[0][1], 100 - missing.len() as u64 * 10);
            assert_eq!(result.paired_hops[0][1], 10 - missing.len());
            if missing.len() == 1 {
                assert!((result.values[0][1].unwrap() + 1.).abs() < 1e-14);
            } else {
                assert!(result.values[0][1].is_none());
            }
        }
    }
}

#[test]
fn a_partial_oldest_hop_contributes_clipped_support_but_one_unweighted_pair() {
    for missing in [1, 2] {
        let mut w = Window::new(0, 2, 0, 10, 95, 4, 0.9).unwrap();
        for i in 1..=10 {
            w.push(i * 10, frame(i as f64, (i != missing).then_some(i as f64)))
                .unwrap();
        }
        let result = w.correlations(&[handle(1), handle(2)]).unwrap();
        assert_eq!(result.window_start, 5);
        assert_eq!(result.paired_hops[0][1], 9);
        assert_eq!(
            result.paired_samples[0][1],
            if missing == 1 { 90 } else { 85 }
        );
        assert_eq!(result.values[0][1].is_some(), missing == 1);
    }
    let mut w = Window::new(0, 2, 0, 10, 95, 4, 0.9).unwrap();
    for i in 1..=10 {
        w.push(
            i * 10,
            frame(if i == 1 { 100. } else { i as f64 }, Some(i as f64)),
        )
        .unwrap();
    }
    let actual = w.correlations(&[handle(1), handle(2)]).unwrap();
    // A partly covered oldest interval still contributes one equally weighted pair.
    let x = [100., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
    let y = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
    let mx = x.iter().sum::<f64>() / 10.;
    let my = y.iter().sum::<f64>() / 10.;
    let xx: f64 = x.iter().map(|v| (v - mx).powi(2)).sum();
    let yy: f64 = y.iter().map(|v| (v - my).powi(2)).sum();
    let xy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    assert!((actual.values[0][1].unwrap() - xy / (xx * yy).sqrt()).abs() < 1e-14);
}

#[test]
fn generation_change_constant_series_and_invalid_frames_do_not_create_correlations() {
    let mut w = Window::new(0, 2, 0, 10, 100, 4, 0.9).unwrap();
    for i in 1..=10 {
        let mut values = frame(i as f64, Some(i as f64));
        if i > 5 {
            values[0].as_mut().unwrap().handle = handle(3);
        }
        w.push(i * 10, values).unwrap();
    }
    let changed = w.correlations(&[handle(3), handle(2)]).unwrap();
    assert_eq!(changed.paired_samples[0][1], 50);
    assert!(changed.values[0][1].is_none());
    assert!(w.correlations(&[handle(1), handle(2)]).is_err());
    let before = w.last_end;
    assert!(w.push(100, frame(1., Some(1.))).is_err());
    assert!(w.push(111, frame(1., Some(1.))).is_err());
    assert!(w.push(110, frame(f64::NAN, Some(1.))).is_err());
    assert_eq!(w.last_end, before);
    let mut w = Window::new(0, 2, 0, 10, 100, 4, 0.9).unwrap();
    for i in 1..=10 {
        w.push(i * 10, frame(2., Some(i as f64))).unwrap();
    }
    let constant = w.correlations(&[handle(1), handle(2)]).unwrap();
    assert_eq!(constant.paired_samples[0][1], 100);
    assert!(constant.values[0][1].is_none());
}

#[test]
fn rolling_windows_and_registered_variants_remain_bounded() {
    for (samples, min_pairs) in [(6000, 4), (12000, 8), (24000, 16)] {
        let mut w = Window::new(0, 2, 0, 512, samples, min_pairs, 0.9).unwrap();
        let bytes = w.frames.capacity() * std::mem::size_of::<Frame>();
        for i in 1..=300 {
            w.push(i * 512, frame(1e12 + i as f64, Some(-1e12 - 2. * i as f64)))
                .unwrap();
        }
        let out = w.correlations(&[handle(1), handle(2)]).unwrap();
        assert_eq!(out.paired_samples[0][1], samples);
        assert_eq!(out.paired_hops[0][1], samples.div_ceil(512) as usize);
        assert!((out.values[0][1].unwrap() + 1.).abs() < 1e-14);
        assert_eq!(w.frames.capacity() * std::mem::size_of::<Frame>(), bytes);
    }
    assert!(Window::new(0, 2, 0, 1, 129, 8, 0.9).is_err());
    assert!(Window::new(0, 2, 0, 10, 30, 8, 0.9).is_err());
}

fn matrix(ids: &[u64], coefficient: impl Fn(u64, u64) -> Option<f64>) -> Correlations {
    Correlations {
        handles: std::array::from_fn(|i| ids.get(i).copied().map(handle)),
        count: ids.len(),
        values: std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if i < ids.len() && j < ids.len() && i != j {
                    coefficient(ids[i], ids[j])
                } else {
                    None
                }
            })
        }),
        paired_samples: [[0; 8]; 8],
        paired_hops: [[0; 8]; 8],
        window_start: 0,
        window_end: 100,
    }
}

#[test]
fn complete_link_does_not_turn_nontransitive_or_missing_pairs_into_one_bundle() {
    let m = matrix(&[3, 1, 2], |a, b| match (a.min(b), a.max(b)) {
        (1, 2) => Some(0.9),
        (2, 3) => Some(0.8),
        _ => Some(0.5),
    });
    let out = complete_link(&m, 0.8).unwrap();
    assert_eq!(out.count, 2);
    assert_eq!(
        out.members[0],
        [
            Some(handle(1)),
            Some(handle(2)),
            None,
            None,
            None,
            None,
            None,
            None
        ]
    );
    assert_eq!(out.members[1][0], Some(handle(3)));
    let m = matrix(&[1, 2, 3], |a, b| {
        if (a.min(b), a.max(b)) == (1, 3) {
            None
        } else {
            Some(0.9)
        }
    });
    let out = complete_link(&m, 0.8).unwrap();
    assert_eq!(out.count, 2);
    assert_eq!(out.members[0][1], Some(handle(2)));
}

#[test]
fn tie_order_uses_sorted_handles_and_eight_members_stay_under_196_reads() {
    for shift in 0..8 {
        let ids: Vec<_> = (0..8).map(|i| ((i + shift) % 8 + 1) as u64).collect();
        let m = matrix(&ids, |_, _| Some(0.85));
        let out = complete_link(&m, 0.8).unwrap();
        assert_eq!(out.count, 1);
        assert_eq!(
            out.members[0],
            std::array::from_fn(|i| Some(handle(i as u64 + 1)))
        );
        assert!(out.correlation_reads <= 196);
        assert_eq!(out.correlation_reads, 140);
    }
}

#[test]
fn malformed_matrices_are_rejected_and_empty_input_is_well_defined() {
    let empty = complete_link(&matrix(&[], |_, _| None), 0.8).unwrap();
    assert_eq!(empty.count, 0);
    assert_eq!(empty.correlation_reads, 0);
    let mut invalid = matrix(&[1, 2], |_, _| Some(0.9));
    invalid.values[1][0] = None;
    assert!(complete_link(&invalid, 0.8).is_err());
    assert!(complete_link(&matrix(&[1, 1], |_, _| Some(0.9)), 0.8).is_err());
    assert!(complete_link(&matrix(&[1, 2], |_, _| Some(f64::NAN)), 0.8).is_err());
}

#[test]
fn independent_oracles_cover_pearson_windows_and_exhaustive_three_member_graphs() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/temporal_cognition/grouping.json"
    ))
    .unwrap();
    let mut json_input_rounding = 0;
    for (case_index, case) in fixture["windows"].as_array().unwrap().iter().enumerate() {
        let mut window = Window::new(
            0,
            2,
            case["epoch_start"].as_u64().unwrap(),
            case["hop"].as_u64().unwrap(),
            case["window_samples"].as_u64().unwrap(),
            case["min_pairs"].as_u64().unwrap() as usize,
            0.9,
        )
        .unwrap();
        for row in case["frames"].as_array().unwrap() {
            let decode = |name: &str| {
                row[name]
                    .as_str()
                    .map(|bits| f64::from_bits(u64::from_str_radix(bits, 16).unwrap()))
            };
            let x = decode("x_bits");
            let y = decode("y_bits");
            for (exact, mirror) in [(x, row["x"].as_f64()), (y, row["y"].as_f64())] {
                if exact.map(f64::to_bits) != mirror.map(f64::to_bits) {
                    json_input_rounding += 1;
                }
            }
            let mut values = [None; 8];
            values[0] = x.map(|x| Envelope {
                handle: handle(row["generation_x"].as_u64().unwrap()),
                log_envelope: x,
            });
            values[1] = y.map(|y| Envelope {
                handle: handle(2),
                log_envelope: y,
            });
            window
                .push(row["end_sample"].as_u64().unwrap(), values)
                .unwrap();
        }
        let out = window
            .correlations(&[handle(case["current_x"].as_u64().unwrap()), handle(2)])
            .unwrap();
        let expected = &case["expected"];
        assert_eq!(out.window_start, expected["start"].as_u64().unwrap());
        assert_eq!(out.window_end, expected["end"].as_u64().unwrap());
        assert_eq!(
            out.paired_samples[0][1],
            expected["paired_samples"].as_u64().unwrap()
        );
        assert_eq!(
            out.paired_hops[0][1],
            expected["paired_hops"].as_u64().unwrap() as usize
        );
        if let Some(value) = expected["coefficient"].as_f64() {
            assert!(
                (out.values[0][1].unwrap() - value).abs() < 1e-12,
                "case {case_index}: Rust {:?}, Decimal {value}",
                out.values[0][1]
            );
        } else {
            assert!(out.values[0][1].is_none());
        }
    }
    println!("grouping_fixture_decimal_parse_differences={json_input_rounding}");
    for case in fixture["graphs"].as_array().unwrap() {
        let ids: Vec<_> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        let mut m = matrix(&ids, |_, _| None);
        for i in 0..ids.len() {
            for j in 0..ids.len() {
                m.values[i][j] = case["matrix"][i][j].as_f64();
            }
        }
        let out = complete_link(&m, 0.8).unwrap();
        let expected = case["expected"].as_array().unwrap();
        assert_eq!(out.count, expected.len());
        assert!(out.correlation_reads <= 196);
        for (index, bundle) in expected.iter().enumerate() {
            let actual: Vec<_> = out.members[index]
                .iter()
                .flatten()
                .map(|h| h.generation)
                .collect();
            let expected: Vec<_> = bundle
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            assert_eq!(actual, expected);
        }
    }
}

#[test]
#[ignore = "release physical-window and bundle cost; excludes other lifecycle and full O04"]
fn grouping_window_cost_probe() {
    use std::hint::black_box;
    use std::time::Instant;
    let handles: Vec<_> = (1..=8).map(handle).collect();
    for samples in [6000, 12000, 24000] {
        let mut window = Window::new(0, 2, 0, 512, samples, 8, 0.9).unwrap();
        let mut times = Vec::with_capacity(6000);
        let mut reads = 0;
        for step in 1..=6100 {
            let values = std::array::from_fn(|i| {
                Some(Envelope {
                    handle: handles[i],
                    log_envelope: -3.
                        + (step % 17) as f64 * 0.02
                        + i as f64 * 0.05
                        + (step % 7) as f64 * i as f64 * 0.0003,
                })
            });
            let start = Instant::now();
            window.push(step * 512, black_box(values)).unwrap();
            let correlations = window.correlations(black_box(&handles)).unwrap();
            let result = black_box(complete_link(&correlations, 0.8).unwrap());
            let elapsed = start.elapsed().as_secs_f64() * 1e6;
            if step > 100 {
                times.push(elapsed);
                reads += result.correlation_reads;
                assert_eq!(result.count, 1);
            }
        }
        times.sort_by(f64::total_cmp);
        println!(
            "grouping_cost {}",
            serde_json::json!({
                "window_samples":samples,"hop":512,"calls":times.len(),"current_trajectories":8,
                "median_us":times[3000],"p99_us":times[5939],"max_us":times[5999],
                "bundle_correlation_reads":reads,"frame_capacity":window.frames.len(),
                "frame_buffer_bytes":window.frames.capacity()*std::mem::size_of::<Frame>(),
                "window_header_bytes":std::mem::size_of::<Window>(),"correlation_bytes":std::mem::size_of::<Correlations>(),
                "bundle_bytes":std::mem::size_of::<Bundles>(),"input":"varying nonzero envelopes; all28 pair correlations supported",
                "full_O04":false
            })
        );
    }
}
