use super::*;

fn tracker() -> Tracker {
    Tracker::new(
        0,
        3,
        1000,
        10,
        Config {
            means: [0.; 3],
            deviations: [1.; 3],
            distance_limit: 1.0,
            retirement_sec: 0.25,
        },
    )
    .unwrap()
}

fn point(frequency: f64) -> Point {
    Point {
        frequency_log2: Some(frequency),
        log_envelope: Some(0.0),
    }
}

#[test]
fn decimal_oracle_checks_global_scales_masks_secants_and_gap_extrapolation() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/temporal_cognition/ridges.json"
    ))
    .unwrap();
    for row in fixture["cases"].as_array().unwrap() {
        let t = Tracker::new(
            0,
            0,
            row["sample_rate"].as_u64().unwrap() as u32,
            row["hop"].as_u64().unwrap(),
            Config {
                means: std::array::from_fn(|i| row["means"][i].as_f64().unwrap()),
                deviations: std::array::from_fn(|i| row["deviations"][i].as_f64().unwrap()),
                distance_limit: 1.0,
                retirement_sec: 0.25,
            },
        )
        .unwrap();
        let old = Ridge {
            handle: Handle {
                bus: 0,
                epoch: 0,
                generation: 1,
            },
            point: Point {
                frequency_log2: row["old"][0].as_f64(),
                log_envelope: row["old"][1].as_f64(),
            },
            end_sample: row["old_end_sample"].as_u64().unwrap(),
            slopes: [row["slope"].as_f64(), None],
            links: [None; 2],
            no_continuation_samples: 0,
        };
        let current = Point {
            frequency_log2: row["new"][0].as_f64(),
            log_envelope: row["new"][1].as_f64(),
        };
        let actual = distance(&old, current, row["end_sample"].as_u64().unwrap(), 0, &t);
        if let Some(expected) = row["expected"]["distance"].as_f64() {
            let (d, slope) = actual.unwrap();
            assert!((d - expected).abs() < 1e-12 * expected.abs().max(1.0));
            if let Some(expected) = row["expected"]["secant"].as_f64() {
                assert!((slope.unwrap() - expected).abs() < 1e-12 * expected.abs().max(1.0));
            } else {
                assert!(slope.is_none());
            }
        } else {
            assert!(actual.is_none());
        }
    }
}

#[test]
fn crossing_uses_direction_and_two_links_preserve_unequal_continuity_weights() {
    let mut t = tracker();
    let out = t.advance(10, Some(&[point(8.), point(8.04)])).unwrap();
    t.slots[0].as_mut().unwrap().slopes = [Some(1.), None];
    t.slots[1].as_mut().unwrap().slopes = [Some(-1.), None];
    let next = t.advance(20, Some(&[point(8.01), point(8.03)])).unwrap();
    assert_eq!(
        next.current[0].unwrap().handle,
        out.current[0].unwrap().handle
    );
    assert_eq!(
        next.current[1].unwrap().handle,
        out.current[1].unwrap().handle
    );
    let crossing = t.advance(30, Some(&[point(8.02)])).unwrap().current[0].unwrap();
    assert_eq!(crossing.links.iter().flatten().count(), 2);
    let outgoing = t.advance(40, Some(&[point(8.01), point(8.03)])).unwrap();
    assert_eq!(
        outgoing.current[0].unwrap().links[0].unwrap().parent_slope,
        1
    );
    assert_eq!(
        outgoing.current[1].unwrap().links[0].unwrap().parent_slope,
        0
    );

    let mut t = tracker();
    t.advance(10, Some(&[point(8.), point(9.)])).unwrap();
    let mixture = t.advance(20, Some(&[point(8.)])).unwrap().current[0].unwrap();
    let a = mixture.links[0].unwrap();
    let b = mixture.links[1].unwrap();
    let expected = (-0.5_f64).exp();
    assert!((a.weight - 1.0 / (1.0 + expected)).abs() < 1e-14);
    assert!((b.weight - expected / (1.0 + expected)).abs() < 1e-14);
    assert!((a.weight + b.weight - 1.0).abs() < 1e-14);
}

#[test]
fn distance_threshold_variants_and_epoch_identity_remain_explicit() {
    for limit in [0.5, 1.0, 2.0] {
        let mut t = tracker();
        t.config.distance_limit = limit;
        // With pitch comparison masked, this is an exact one-coordinate boundary.
        let prior = t.advance(10, Some(&[point(8.)])).unwrap().current[0].unwrap();
        let prior = Ridge {
            point: Point {
                frequency_log2: None,
                log_envelope: Some(0.),
            },
            ..prior
        };
        t.slots[0] = Some(prior);
        let current = Point {
            frequency_log2: Some(8.),
            log_envelope: Some(limit),
        };
        let accepted = t.advance(20, Some(&[current])).unwrap().current[0].unwrap();
        assert_eq!(accepted.handle, prior.handle);
        let mut outside = tracker();
        outside.config.distance_limit = limit;
        outside.advance(10, Some(&[point(8.)])).unwrap();
        outside.slots[0] = Some(prior);
        let rejected = outside
            .advance(
                20,
                Some(&[Point {
                    log_envelope: Some(limit + 1e-6),
                    ..current
                }]),
            )
            .unwrap()
            .current[0]
            .unwrap();
        assert!(rejected.links.iter().all(Option::is_none));
    }
    let config = tracker().config;
    let mut a = Tracker::new(0, 0, 1000, 10, config).unwrap();
    let mut b = Tracker::new(1, 0, 1000, 10, config).unwrap();
    let mut c = Tracker::new(0, 1, 1000, 10, config).unwrap();
    let x = a.advance(10, Some(&[point(8.)])).unwrap().current[0]
        .unwrap()
        .handle;
    let y = b.advance(10, Some(&[point(8.)])).unwrap().current[0]
        .unwrap()
        .handle;
    let z = c.advance(10, Some(&[point(8.)])).unwrap().current[0]
        .unwrap()
        .handle;
    assert!(x != y && x != z && y != z);
}

#[test]
fn long_mixed_lifecycle_stays_bounded_without_duplicating_generation_or_link_weight() {
    let mut t = tracker();
    let mut created = std::collections::BTreeSet::new();
    let mut last_generation = 0;
    let mut evictions = 0;
    let mut supersessions = 0;
    for step in 1..=3000 {
        let count = [7, 1, 2, 7, 0, 3, 7][step % 7];
        let points: Vec<_> = (0..count)
            .map(|i| point(8. + ((step / 100) % 2) as f64 * 10. + i as f64 * 0.002))
            .collect();
        let missing = step % 13 == 0;
        let before = t.slots;
        let out = t
            .advance(step as u64 * 10, (!missing).then_some(points.as_slice()))
            .unwrap();
        assert!(out.distance_evaluations <= 98);
        assert!(t.slots.iter().flatten().count() <= 7);
        if missing {
            assert_eq!(t.slots, before);
            continue;
        }
        let mut current_handles = std::collections::BTreeSet::new();
        for (index, current) in out.current.iter().flatten().enumerate() {
            assert_eq!(current.point, points[index]);
            assert!(current_handles.insert(current.handle));
            assert_eq!(current.end_sample, step as u64 * 10);
            if !before
                .iter()
                .flatten()
                .any(|old| old.handle == current.handle)
            {
                assert!(current.handle.generation > last_generation);
                assert!(created.insert(current.handle));
                last_generation = current.handle.generation;
            }
            if current.links[0].is_some() {
                let weight: f64 = current.links.iter().flatten().map(|l| l.weight).sum();
                assert!((weight - 1.0).abs() < 1e-14);
                for link in current.links.iter().flatten() {
                    assert!(link.parent_end_sample < current.end_sample);
                    assert!(
                        before.iter().flatten().any(|old| old.handle == link.parent
                            && old.end_sample == link.parent_end_sample)
                    );
                }
            }
        }
        for removed in out
            .retired
            .iter()
            .chain(out.evicted.iter())
            .chain(out.superseded.iter())
            .flatten()
        {
            assert!(!t.slots.iter().flatten().any(|r| r.handle == *removed));
        }
        evictions += out.evicted.iter().flatten().count();
        supersessions += out.superseded.iter().flatten().count();
    }
    assert!(created.len() > 1000 && evictions > 0 && supersessions > 0);
}

#[test]
#[ignore = "release ridge update cost; no fitted acoustic scale or full O04 workload"]
fn ridge_update_cost_probe() {
    use std::hint::black_box;
    use std::time::Instant;

    for count in [1, 7] {
        let mut t = tracker();
        let mut points = vec![point(8.); count];
        let mut times = Vec::with_capacity(6000);
        let mut comparisons = 0;
        for step in 1..=6100 {
            for (index, point) in points.iter_mut().enumerate() {
                point.frequency_log2 =
                    Some(8.0 + index as f64 * 0.001 + (step % 7) as f64 * 0.0003);
                point.log_envelope = Some(index as f64 * 0.025 + (step % 5) as f64 * 0.0125);
            }
            let start = Instant::now();
            let output = black_box(
                t.advance(step * 10, Some(black_box(points.as_slice())))
                    .unwrap(),
            );
            let elapsed = start.elapsed().as_secs_f64() * 1e6;
            if step > 100 {
                times.push(elapsed);
                comparisons += output.distance_evaluations;
            }
        }
        times.sort_by(f64::total_cmp);
        assert_eq!(comparisons, 6000 * if count == 1 { 1 } else { 98 });
        println!(
            "ridge_cost {}",
            serde_json::json!({
                "peaks": count, "calls": times.len(), "distance_evaluations": comparisons,
                "median_us": times[3000], "p99_us": times[5939], "max_us": times[5999],
                "tracker_bytes": std::mem::size_of::<Tracker>(), "update_bytes": std::mem::size_of::<Update>(),
            "config": "constructed means0/sd1, 1-distance, .25s retirement; not development fitted",
            "input": "distinct peaks with varying frequency/envelope and nonzero residuals",
                "full_O04": false
            })
        );
    }
}

#[test]
fn birth_first_continuation_glide_and_changed_slope_keep_observation_semantics() {
    let mut t = tracker();
    let birth = t.advance(10, Some(&[point(8.0)])).unwrap().current[0].unwrap();
    assert_eq!(birth.slopes, [None; 2]);
    assert_eq!(birth.links, [None; 2]);
    let second = t.advance(20, Some(&[point(8.005)])).unwrap().current[0].unwrap();
    assert_eq!(second.handle, birth.handle);
    assert!((second.slopes[0].unwrap() - 0.5).abs() < 1e-12);
    assert!((second.links[0].unwrap().distance - 0.005 / 2.0_f64.sqrt()).abs() < 1e-12);
    assert_eq!(second.links[0].unwrap().parent_end_sample, 10);
    let third = t.advance(30, Some(&[point(8.01)])).unwrap().current[0].unwrap();
    assert_eq!(third.handle, birth.handle);
    assert!(third.links[0].unwrap().distance < 1e-12);
    let changed = t.advance(40, Some(&[point(8.02)])).unwrap().current[0].unwrap();
    assert_eq!(changed.handle, birth.handle);
    assert!((changed.slopes[0].unwrap() - 1.0).abs() < 1e-12);
    assert!(
        (changed.links[0].unwrap().distance - ((0.005_f64.powi(2) + 0.5_f64.powi(2)) / 3.0).sqrt())
            .abs()
            < 1e-12
    );
}

#[test]
fn two_predecessors_and_split_get_fresh_unique_handles_with_normalized_links() {
    let mut t = tracker();
    let first = t.advance(10, Some(&[point(8.0), point(8.02)])).unwrap();
    let a = first.current[0].unwrap().handle;
    let b = first.current[1].unwrap().handle;
    let merge = t.advance(20, Some(&[point(8.01)])).unwrap();
    let merged = merge.current[0].unwrap();
    assert!(merged.handle != a && merged.handle != b);
    assert_eq!(merged.links.map(|l| l.unwrap().parent), [a, b]);
    assert_eq!(merged.links.map(|l| l.unwrap().weight), [0.5, 0.5]);
    assert!((merged.slopes[0].unwrap() - 1.0).abs() < 1e-12);
    assert!((merged.slopes[1].unwrap() + 1.0).abs() < 1e-12);
    assert_eq!(merge.superseded.iter().flatten().count(), 2);
    let split = t.advance(30, Some(&[point(8.0), point(8.02)])).unwrap();
    let left = split.current[0].unwrap();
    let right = split.current[1].unwrap();
    assert!(
        left.handle != right.handle
            && left.handle != merged.handle
            && right.handle != merged.handle
    );
    assert_eq!(left.links[0].unwrap().parent, merged.handle);
    assert_eq!(right.links[0].unwrap().parent, merged.handle);
    assert_eq!(left.links[0].unwrap().parent_slope, 1);
    assert_eq!(right.links[0].unwrap().parent_slope, 0);
    assert_eq!(split.superseded.iter().flatten().count(), 1);
    assert_eq!(t.slots.iter().flatten().count(), 2);
}

#[test]
fn gap_masks_new_slope_and_does_not_advance_observed_retirement() {
    let mut t = tracker();
    let birth = t.advance(10, Some(&[point(8.0)])).unwrap().current[0].unwrap();
    t.advance(20, Some(&[point(8.005)])).unwrap();
    let absent = t.advance(1000, None).unwrap();
    assert!(!absent.observed);
    assert!(absent.current.iter().all(Option::is_none));
    assert_eq!(t.slots[0].unwrap().no_continuation_samples, 0);
    let recovery = t.advance(1010, Some(&[point(8.5)])).unwrap().current[0].unwrap();
    assert_eq!(recovery.handle, birth.handle);
    assert_eq!(recovery.slopes, [None; 2]);
    assert!(recovery.links[0].unwrap().distance < 1e-12);
    let next = t.advance(1020, Some(&[point(8.505)])).unwrap().current[0].unwrap();
    assert!((next.slopes[0].unwrap() - 0.5).abs() < 1e-12);
    for frame in 103..127 {
        t.advance(frame * 10, Some(&[])).unwrap();
    }
    assert!(t.slots.iter().flatten().any(|r| r.handle == birth.handle));
    t.advance(10000, None).unwrap();
    let retirement = t.advance(10010, Some(&[])).unwrap();
    assert_eq!(
        retirement
            .retired
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
        vec![birth.handle]
    );
    assert!(t.slots.iter().all(Option::is_none));
    let reborn = t.advance(10020, Some(&[point(8.5)])).unwrap().current[0].unwrap();
    assert_ne!(reborn.handle, birth.handle);
    assert_eq!(reborn.links, [None; 2]);
}

#[test]
fn observed_empty_hops_retire_on_each_registered_duration_boundary() {
    for duration in [0.125, 0.25, 0.5] {
        let mut t = tracker();
        t.config.retirement_sec = duration;
        t.retirement_samples = (duration * 1000.0).ceil() as u64;
        let identity = t.advance(10, Some(&[point(8.)])).unwrap().current[0]
            .unwrap()
            .handle;
        let steps = t.retirement_samples.div_ceil(10);
        for i in 1..steps {
            let out = t.advance((i + 1) * 10, Some(&[])).unwrap();
            assert!(out.retired.iter().all(Option::is_none));
        }
        let out = t.advance((steps + 1) * 10, Some(&[])).unwrap();
        assert_eq!(
            out.retired.iter().flatten().copied().collect::<Vec<_>>(),
            vec![identity]
        );
    }
}

#[test]
fn capacity_preserves_current_peaks_and_evicts_oldest_dormant_without_handle_reuse() {
    let mut t = tracker();
    let first: Vec<_> = (0..7).map(|i| point(i as f64 * 100.0)).collect();
    let old = t.advance(10, Some(&first)).unwrap();
    let points: Vec<_> = (0..6).map(|i| point(i as f64 * 100.0)).collect();
    t.advance(20, Some(&points)).unwrap();
    let input: Vec<_> = (0..6).map(|i| point(i as f64 * 100.0 + 2000.0)).collect();
    let out = t.advance(30, Some(&input)).unwrap();
    assert_eq!(out.evicted.iter().flatten().count(), 6);
    assert!(
        out.evicted
            .iter()
            .flatten()
            .any(|h| *h == old.current[6].unwrap().handle)
    );
    assert_eq!(t.slots.iter().flatten().count(), 7);
    for current in out.current.iter().flatten() {
        assert!(
            old.current
                .iter()
                .flatten()
                .all(|r| r.handle != current.handle)
        );
    }
}

#[test]
fn two_slopes_have_98_evaluation_cap_and_stable_distinct_parent_ties() {
    let mut t = tracker();
    t.advance(10, Some(&[point(8.0); 7])).unwrap();
    for ridge in t.slots.iter_mut().flatten() {
        ridge.slopes = [Some(0.), Some(0.)];
    }
    let out = t.advance(20, Some(&[point(8.0); 7])).unwrap();
    assert_eq!(out.distance_evaluations, 98);
    for current in out.current.iter().flatten() {
        let a = current.links[0].unwrap();
        let b = current.links[1].unwrap();
        assert_eq!(a.parent.generation, 1);
        assert_eq!(b.parent.generation, 2);
        assert_eq!(a.parent_slope, 0);
        assert_eq!(b.parent_slope, 0);
        assert_eq!(a.weight + b.weight, 1.0);
    }
    let mut ids: Vec<_> = out.current.iter().flatten().map(|r| r.handle).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 7);
}

#[test]
fn frozen_coordinate_masks_thresholds_and_invalid_input_do_not_silently_change_state() {
    let mut t = tracker();
    let birth = t.advance(10, Some(&[point(8.)])).unwrap().current[0].unwrap();
    let envelope_only = Ridge {
        point: Point {
            frequency_log2: None,
            log_envelope: Some(0.),
        },
        ..birth
    };
    let masked = Point {
        frequency_log2: None,
        log_envelope: None,
    };
    assert!(distance(&envelope_only, masked, 20, 0, &t).is_none());
    assert_eq!(
        distance(
            &envelope_only,
            Point {
                log_envelope: Some(1.),
                ..masked
            },
            20,
            0,
            &t
        ),
        Some((1., None))
    );
    let initial = t.slots;
    assert!(t.advance(10, Some(&[])).is_err());
    assert!(t.advance(21, Some(&[])).is_err());
    assert!(t.advance(20, Some(&[point(f64::NAN)])).is_err());
    assert!(t.advance(20, Some(&[point(8.); 8])).is_err());
    assert_eq!(t.slots, initial);
    assert_eq!(t.last_end, 10);
    t.next_generation = u64::MAX;
    assert!(t.advance(20, Some(&[point(100.)])).is_err());
    assert_eq!(t.slots, initial);
    assert_eq!(t.last_end, 10);
}
