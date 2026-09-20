use super::*;
use serde_json::Value;

fn group() -> Handle {
    Handle {
        bus: 0,
        epoch: 2,
        generation: 3,
    }
}

fn accent(end: u64, observed_prefix: u64, weight: f64) -> Accent {
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
        available_end: end + 16,
        weight,
        observed_prefix,
    }
}

fn direct_grid(e: &Estimator) -> [f64; BINS] {
    let mut sums = [0.; BINS];
    for (j, newer) in e.ledger.bank.iter().enumerate() {
        for older in e.ledger.bank.iter().take(j) {
            let samples = newer.event_end - older.event_end;
            if samples == 0
                || (newer.observed_prefix - older.observed_prefix) as f64 / (samples as f64) < 0.9
            {
                continue;
            }
            for (bin, sum) in sums.iter_mut().enumerate() {
                let period = 0.125 * 2f64.powf(bin as f64 / 48.);
                let seconds = samples as f64 / f64::from(e.sample_rate);
                *sum += older.weight
                    * newer.weight
                    * (1. - (seconds / period).log2().abs() * 24.).max(0.);
            }
        }
    }
    sums
}

#[test]
fn reset_clears_wrapped_pair_and_credit_state_without_reallocating_or_accepting_old_owner() {
    let mut e = Estimator::new(group(), 8, 1000, 32000, 1024, 1. / 24.).unwrap();
    let layout = e.storage_layout();
    for i in 1..=40 {
        e.deliver(accent(i * 256, i * 256, 1.), i * 256 + 16)
            .unwrap();
    }
    assert_eq!(e.ledger_summary().cumulative_count, 40);
    assert!(e.ledger_summary().capacity_evicted_through.is_some());
    assert!(e.view().pairs > 0 && e.view().peaks.iter().any(Option::is_some));
    assert!(e.reset(Handle { bus: 2, ..group() }, 11000).is_err());
    assert_eq!(e.ledger_summary().cumulative_count, 40);
    let next = Handle {
        generation: 4,
        ..group()
    };
    e.reset(next, 11000).unwrap();
    assert_eq!(e.storage_layout(), layout);
    let summary = e.ledger_summary();
    assert_eq!(summary.group, next);
    assert_eq!(summary.received_at, 11000);
    assert_eq!((summary.cumulative_count, summary.retained_accents), (0, 0));
    assert_eq!(summary.cumulative_weight, 0.);
    assert_eq!(summary.capacity_evicted_through, None);
    assert_eq!(e.view().pairs, 0);
    assert_eq!(e.view().supported_pairs, 0);
    assert_eq!(e.view().nonzero_pairs, 0);
    assert!(e.view().probabilities.iter().all(|&p| p == 0.));
    assert!(e.view().peaks.iter().all(Option::is_none));
    assert!(e.cache.iter().all(|p| p.flags == 0));
    assert!(e.points.iter().all(Option::is_none));
    assert!(e.order.is_empty());
    assert!(
        e.deliver(accent(11264, 11264, 1.), 11280)
            .unwrap()
            .is_none()
    );
    let fresh = Accent {
        group: next,
        observed_prefix: 64,
        ..accent(11264, 11264, 1.)
    };
    assert_eq!(e.deliver(fresh, 11280).unwrap().unwrap().sequence, 1);
    assert_eq!(e.view().pairs, 0);
    assert_eq!(e.storage_layout(), layout);
}

#[test]
fn direct_all_bin_python_kernels_match_incremental_cache_and_rebuild_controls() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/temporal_cognition/periods.json"
    ))
    .unwrap();
    let mut checks = 0;
    let mut max_drift: f64 = 0.;
    let mut max_rounding: f64 = 0.;
    let mut rounding_peak_changes = 0;
    for sequence in fixture["sequences"].as_array().unwrap() {
        for schedule in sequence["rebuild_controls"].as_array().unwrap() {
            let capacity = sequence["capacity"].as_u64().unwrap() as usize;
            let mut e = Estimator::new(
                group(),
                capacity,
                1000,
                32000,
                schedule.as_u64().unwrap(),
                1. / 24.,
            )
            .unwrap();
            let cache_pointer = e.cache.as_ptr();
            let bank_capacity = e.ledger.bank.capacity();
            let order_capacity = e.order.capacity();
            for (index, step) in sequence["steps"].as_array().unwrap().iter().enumerate() {
                let a = &step["accent"];
                let packet = accent(
                    a["end"].as_u64().unwrap(),
                    a["observed_prefix"].as_u64().unwrap(),
                    a["weight"].as_f64().unwrap(),
                );
                e.deliver(packet, step["received_at"].as_u64().unwrap())
                    .unwrap()
                    .unwrap();
                let view = e.view();
                assert_eq!(view.group, group());
                assert_eq!(view.received_at, packet.available_end);
                assert_eq!(
                    view.work.rebuilds,
                    usize::from((index as u64 + 1).is_multiple_of(schedule.as_u64().unwrap()))
                );
                assert!(view.work.inserted_pairs < capacity);
                assert!(view.work.peak_candidates <= 121 && view.work.separation_checks <= 121 * 8);
                assert_eq!(e.cache.as_ptr(), cache_pointer);
                assert_eq!(e.ledger.bank.capacity(), bank_capacity);
                assert_eq!(e.order.capacity(), order_capacity);
                let expected = &step["checkpoint"];
                if expected.is_null() {
                    continue;
                }
                let mut records = 0;
                for record in e.cache.iter().filter(|p| p.flags & 1 != 0) {
                    assert!(record.left < record.right && (record.right as usize) < capacity);
                    assert!(
                        e.points[record.left as usize].is_some()
                            && e.points[record.right as usize].is_some()
                    );
                    assert_eq!(record.padding, [0; 4]);
                    records += 1;
                }
                assert_eq!(records, view.pairs);
                for (bin, &actual) in e.sums.iter().enumerate() {
                    let cached = expected["masses"][bin].as_f64().unwrap();
                    let direct = expected["full_f64"][bin].as_f64().unwrap();
                    max_drift = max_drift.max((actual - cached).abs());
                    max_rounding = max_rounding.max((cached - direct).abs());
                    assert!(
                        (actual - cached).abs() < 1e-10,
                        "capacity={capacity} end={} bin={bin}: {actual} vs {cached}",
                        packet.event_end
                    );
                    assert!(
                        (view.probabilities[bin]
                            - expected["probabilities"][bin].as_f64().unwrap())
                        .abs()
                            < 1e-11
                    );
                }
                assert_eq!(view.pairs, expected["pairs"].as_u64().unwrap() as usize);
                assert_eq!(
                    view.supported_pairs,
                    expected["supported_pairs"].as_u64().unwrap() as usize
                );
                assert_eq!(
                    view.nonzero_pairs,
                    expected["nonzero_pairs"].as_u64().unwrap() as usize
                );
                let peaks: Vec<_> = view.peaks.iter().flatten().map(|p| p.bin).collect();
                let expected_peaks: Vec<_> = expected["peaks"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as usize)
                    .collect();
                assert_eq!(peaks, expected_peaks);
                rounding_peak_changes +=
                    usize::from(expected["peaks"] != expected["full_f64_peaks"]);
                checks += 1;
            }
        }
    }
    println!(
        "PERIOD_GRID_ORACLE {}",
        serde_json::json!({"checkpoints":checks,"max_cached_sum_drift":max_drift,"max_f32_rounding_delta":max_rounding,"rounding_peak_inventory_changes":rounding_peak_changes})
    );
}

#[test]
fn peak_plateaus_edges_ladders_and_separation_share_one_selection_rule() {
    let periods = std::array::from_fn(|i| 0.125 * 2f64.powf(i as f64 / 48.));
    for uniform in [0., 1. / 241.] {
        assert!(
            select_peaks(&[uniform; BINS], &periods, 1. / 24.)
                .0
                .iter()
                .all(Option::is_none)
        );
    }
    let mut values = [0.; BINS];
    values[0] = 1.;
    values[240] = 1.;
    values[10..15].fill(2.);
    values[16] = 2.;
    values[18] = 1.5;
    let (peaks, count, _) = select_peaks(&values, &periods, 1. / 24.);
    assert_eq!(count, 5);
    assert_eq!(
        peaks.iter().flatten().map(|p| p.bin).collect::<Vec<_>>(),
        [10, 16, 18, 0, 240]
    );
    let wider = select_peaks(&values, &periods, 1. / 12.).0;
    assert_eq!(
        wider.iter().flatten().map(|p| p.bin).collect::<Vec<_>>(),
        [10, 16, 0, 240]
    );
    let narrow = select_peaks(&values, &periods, 1. / 48.).0;
    assert_eq!(narrow, peaks);
    let alternating = std::array::from_fn(|i| if i % 2 == 0 { 1. } else { 0. });
    assert_eq!(select_peaks(&alternating, &periods, 1. / 24.).1, 121);
    let mut e = Estimator::new(group(), 128, 1000, 32000, 1024, 1. / 24.).unwrap();
    for i in 0..20 {
        e.deliver(accent(64 + i * 500, 64 + i * 500, 1.), 80 + i * 500)
            .unwrap();
    }
    let view = e.view();
    assert_eq!(view.peaks[0].unwrap().period_seconds, 0.5);
    assert!(view.peaks.iter().flatten().any(|p| p.period_seconds == 1.));
    assert!(view.peaks.iter().flatten().any(|p| p.period_seconds == 2.));
}

#[test]
fn cached_support_survives_later_gaps_and_slot_reuse_removes_incident_pairs_once() {
    let mut e = Estimator::new(group(), 3, 1000, 1000, 1024, 1. / 24.).unwrap();
    e.deliver(accent(64, 64, 1.), 80).unwrap();
    e.deliver(accent(264, 244, 1.), 280).unwrap();
    assert_eq!(e.view().supported_pairs, 1);
    let heard = e.sums;
    for (cached, exact) in heard.iter().zip(direct_grid(&e)) {
        assert!((cached - exact).abs() < 1e-6);
    }
    e.advance(400).unwrap();
    assert_eq!(e.sums, heard);
    e.deliver(accent(464, 422, 1.), 480).unwrap();
    assert_eq!(e.view().supported_pairs, 1);
    e.deliver(accent(664, 622, 1.), 680).unwrap();
    assert_eq!(e.view().work.removed_pairs, 2);
    assert_eq!(e.view().pairs, 3);
    assert_eq!(e.view().supported_pairs, 2);
    e.advance(1664).unwrap();
    assert_eq!(e.view().pairs, 0);
    assert_eq!(e.view().work.removed_pairs, 3);
    assert_eq!(e.sums, [0.; BINS]);
    assert!(e.view().peaks.iter().all(Option::is_none));
}

#[test]
fn packed_pair_layout_and_tiny_weight_approximation_are_explicit() {
    assert_eq!(std::mem::size_of::<Pair>(), 32);
    assert_eq!(std::mem::offset_of!(Pair, values), 8);
    assert_eq!(std::mem::offset_of!(Pair, padding), 28);
    let mut e = Estimator::new(group(), 128, 1000, 32000, 1024, 1. / 24.).unwrap();
    e.deliver(accent(64, 64, 1e-30), 80).unwrap();
    e.deliver(accent(564, 564, 1e-30), 580).unwrap();
    assert_eq!(e.view().supported_pairs, 1);
    assert_eq!(e.view().nonzero_pairs, 0);
    assert!(e.view().peaks.iter().all(Option::is_none));
    let direct = direct_grid(&e);
    assert!(direct[96] > 0.);
    assert_eq!(
        select_peaks(&direct, &e.periods, 1. / 24.).0[0]
            .unwrap()
            .bin,
        96
    );
    assert!(e.cache.iter().all(|p| p.padding == [0; 4]));
}

#[test]
fn f32_rounding_can_change_best_period_at_a_near_tie() {
    let mut e = Estimator::new(group(), 128, 1000, 32000, 1024, 1. / 24.).unwrap();
    for a in [
        accent(64, 64, 1.),
        accent(564, 564, 0.5),
        accent(10064, 628, 1.),
        accent(11064, 1628, 0.500000001),
    ] {
        e.deliver(a, a.available_end).unwrap();
    }
    assert_eq!(e.view().supported_pairs, 2);
    assert_eq!(e.sums[96], 0.5);
    assert_eq!(e.sums[144], 0.5);
    assert_eq!(e.view().peaks[0].unwrap().bin, 96);
    let direct = direct_grid(&e);
    assert!(direct[144] > direct[96]);
    assert_eq!(
        select_peaks(&direct, &e.periods, 1. / 24.).0[0]
            .unwrap()
            .bin,
        144
    );
    println!(
        "PERIOD_ROUNDING_COUNTEREXAMPLE rounded_best_seconds=0.5 all_f64_best_seconds=1.0 weights=0.5,0.500000001"
    );
}

#[test]
fn original_observation_prefix_connects_detector_ledger_and_cross_gap_periods() {
    use crate::core::log2space::Log2Space;
    use crate::temporal_cognition::features;
    let grid = Log2Space::new(64., 1024., 2);
    let mut stream = features::Stream::new(
        grid.n_bins(),
        features::Config {
            means: [0.; 2],
            deviations: [1.; 2],
            threshold: 1.,
            rms_floor: 1e-6,
        },
    )
    .unwrap();
    let mut estimator = Estimator::new(group(), 128, 512, 32000, 1024, 1. / 24.).unwrap();
    let mut end: u64 = 0;
    let mut actual = Vec::new();
    for index in 0..21 {
        let missing = index == 8;
        let association = index != 13;
        let start = end;
        end += if missing { 1000 } else { 64 };
        let position = if index < 8 { index } else { index - 9 };
        let energy = if position % 4 < 2 { 1. } else { 16. };
        let scan = [energy / 9.; 9];
        let update = stream
            .push(
                &grid,
                features::Input {
                    stamp: features::Stamp {
                        group: group(),
                        association: association.then_some(3),
                        grid_id: 1,
                        start,
                        end,
                        source_start: if index < 8 {
                            start.saturating_sub(192)
                        } else {
                            start
                        },
                        source_end: end,
                        available_end: end,
                        known_samples: if missing { 0 } else { 64 },
                        observed: !missing,
                    },
                    energy: (!missing).then_some(energy),
                    bus_energy: (!missing).then_some(energy),
                    energy_scan: (!missing).then_some(scan.as_slice()),
                },
                end,
            )
            .unwrap()
            .unwrap();
        if let Some(a) = update.detector.and_then(|d| d.accent) {
            assert!(estimator.deliver(a, a.event_end).unwrap().is_none());
            estimator.deliver(a, end + 5).unwrap().unwrap();
            actual.push((a.event_end, a.observed_prefix));
        } else {
            estimator.advance(end).unwrap();
        }
    }
    assert_eq!(actual, [(192, 192), (448, 448), (1704, 704), (2216, 1152)]);
    assert_eq!(estimator.view().pairs, 6);
    assert_eq!(estimator.view().supported_pairs, 1);
    assert_eq!(estimator.view().peaks[0].unwrap().period_seconds, 0.5);
}

#[test]
fn observation_prefix_replays_and_impossible_coverage_are_rejected_atomically() {
    let mut e = Estimator::new(group(), 128, 1000, 32000, 1024, 1. / 24.).unwrap();
    let first = accent(64, 64, 1.);
    e.deliver(first, 80).unwrap();
    let saved = e.view();
    for a in [
        accent(64, 63, 1.),
        accent(264, 63, 1.),
        accent(264, 265, 1.),
    ] {
        assert!(e.deliver(a, a.available_end).is_err());
        assert_eq!(e.view().received_at, saved.received_at);
        assert_eq!(e.view().pairs, saved.pairs);
        assert_eq!(e.view().probabilities, saved.probabilities);
    }
    assert!(e.deliver(first, 80).unwrap().is_none());
    assert_eq!(e.view().pairs, 0);
    let mut partial = Estimator::new(group(), 128, 1000, 32000, 1024, 1. / 24.).unwrap();
    partial.deliver(accent(64, 48, 1.), 80).unwrap();
    assert!(partial.deliver(accent(264, 264, 1.), 280).is_err());
    assert_eq!(partial.view().received_at, 80);
}

#[test]
#[ignore = "release saturated period cache, rebuild and expiry cost; not full O04"]
fn period_grid_cost_probe() {
    use std::{hint::black_box, time::Instant};
    for capacity in [64, 128, 256] {
        let mut e = Estimator::new(group(), capacity, 1000, 32000, 1024, 1. / 24.).unwrap();
        let mut times = Vec::with_capacity(6000);
        let mut rebuild_times = Vec::new();
        let mut inserted = 0;
        let mut removed = 0;
        let mut bin_updates = 0;
        let mut maximum_pairs = 0;
        for i in 0..capacity + 6000 {
            let end = 64 + i as u64 * 64;
            let a = accent(end, end, [0.25, 0.5, 0.75, 1.][i % 4]);
            let start = Instant::now();
            e.deliver(black_box(a), a.available_end).unwrap();
            let view = black_box(e.view());
            let elapsed = start.elapsed().as_secs_f64() * 1e6;
            if i >= capacity {
                times.push(elapsed);
                assert_eq!(view.pairs, capacity * (capacity - 1) / 2);
                maximum_pairs = maximum_pairs.max(view.pairs);
                inserted += view.work.inserted_pairs;
                removed += view.work.removed_pairs;
                bin_updates +=
                    view.work.added_bins + view.work.subtracted_bins + view.work.rebuilt_bins;
                if view.work.rebuilds != 0 {
                    rebuild_times.push(elapsed);
                }
            }
        }
        let end = e.view().received_at + 32001;
        let start = Instant::now();
        e.advance(black_box(end)).unwrap();
        let expiry_us = start.elapsed().as_secs_f64() * 1e6;
        assert_eq!(e.view().work.removed_pairs, maximum_pairs);
        assert_eq!(e.view().pairs, 0);
        assert_eq!(e.sums, [0.; BINS]);
        times.sort_by(f64::total_cmp);
        println!(
            "period_grid_cost {}",
            serde_json::json!({
                "capacity":capacity,"calls":6000,"maximum_pairs":maximum_pairs,"median_us":times[3000],"p99_us":times[5939],"max_us":times[5999],
                "rebuild_call_us":rebuild_times,"full_expiry_us":expiry_us,"inserted_pairs":inserted,"removed_pairs":removed,"bin_updates_including_rebuild":bin_updates,
                "estimator_header_bytes":std::mem::size_of::<Estimator>(),"pair_record_bytes":std::mem::size_of::<Pair>(),"pair_payload_bytes":e.cache.capacity()*std::mem::size_of::<Pair>(),
                "accent_payload_bytes":e.ledger.bank.capacity()*std::mem::size_of::<Accent>(),"slot_order_payload_bytes":e.order.capacity()*std::mem::size_of::<u16>(),
                "view_bytes":std::mem::size_of::<View>(),"full_O04":false,
                "scope":"one saturated resolved-group estimator including ledger, cache removal/addition, normalized peaks and view copy; periodic rebuild and one full-expiry burst measured; DSP/beam/two-bus workers/device excluded"
            })
        );
    }
}
