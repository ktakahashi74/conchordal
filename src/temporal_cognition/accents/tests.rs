use super::*;
use serde_json::Value;

fn group() -> Handle {
    Handle {
        bus: 0,
        epoch: 2,
        generation: 3,
    }
}

fn event(start: u64) -> Accent {
    Accent {
        group: group(),
        event_start: start + 32,
        event_end: start + 48,
        raw_intervals: std::array::from_fn(|i| {
            (start + i as u64 * 16, start + (i as u64 + 1) * 16)
        }),
        source_start: start,
        source_end: start + 64,
        available_end: start + 64,
        weight: 0.5,
        observed_prefix: start + 48,
    }
}

#[test]
fn bounded_bank_and_density_windows_match_python_delivery_reference() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/temporal_cognition/accent-ledger.json"
    ))
    .unwrap();
    let mut operations = 0;
    for sequence in fixture["sequences"].as_array().unwrap() {
        let capacity = sequence["capacity"].as_u64().unwrap() as usize;
        let mut ledger = Ledger::new(group(), capacity, 32000).unwrap();
        let allocated = ledger.bank.capacity();
        for step in sequence["steps"].as_array().unwrap() {
            let clock = step["received_at"].as_u64().unwrap();
            let outcome = if step["action"] == "deliver" {
                let a = &step["accent"];
                let first = a["raw_support_intervals"][0][0].as_u64().unwrap();
                let accent = Accent {
                    weight: a["weight"].as_f64().unwrap(),
                    source_end: a["raw_support_end"].as_u64().unwrap(),
                    available_end: a["available_end"].as_u64().unwrap(),
                    raw_intervals: std::array::from_fn(|i| {
                        (
                            a["raw_support_intervals"][i][0].as_u64().unwrap(),
                            a["raw_support_intervals"][i][1].as_u64().unwrap(),
                        )
                    }),
                    ..event(first)
                };
                ledger.deliver(accent, clock)
            } else {
                ledger.advance(clock).map(|_| None)
            };
            match step["result"].as_str().unwrap() {
                "error" => assert!(outcome.is_err()),
                "none" => assert_eq!(outcome.unwrap(), None),
                "delivered" => {
                    let delivery = outcome.unwrap().unwrap();
                    assert_eq!(delivery.sequence, step["sequence"].as_u64().unwrap());
                    assert_eq!(delivery.received_at, clock);
                }
                _ => unreachable!(),
            }
            for expected in step["snapshots"].as_array().unwrap() {
                let snapshot = ledger
                    .snapshot(expected["start"].as_u64().unwrap())
                    .unwrap();
                assert_eq!(snapshot.group, group());
                assert_eq!(snapshot.end, expected["end"].as_u64().unwrap());
                assert_eq!(
                    snapshot.capacity_valid,
                    expected["capacity_valid"].as_bool().unwrap()
                );
                assert_eq!(snapshot.weight, expected["weight"].as_f64());
                assert_eq!(
                    snapshot.cumulative_count,
                    expected["count"].as_u64().unwrap()
                );
                assert_eq!(
                    snapshot.cumulative_weight,
                    expected["cumulative_weight"].as_f64().unwrap()
                );
                assert_eq!(
                    snapshot.capacity_evicted_through,
                    expected["evicted"].as_u64()
                );
                let rows = expected["accents"].as_array().unwrap();
                assert_eq!(snapshot.accents.len(), rows.len());
                for (a, expected) in snapshot.accents.iter().zip(rows) {
                    assert_eq!(
                        (a.event_start, a.event_end),
                        (expected[0].as_u64().unwrap(), expected[1].as_u64().unwrap())
                    );
                    assert_eq!(a.weight, expected[2].as_f64().unwrap());
                }
            }
            assert_eq!(ledger.bank.capacity(), allocated);
            assert!(ledger.bank.len() <= capacity);
            operations += 1;
        }
    }
    println!("ACCENT_LEDGER_ORACLE operations={operations}");
    assert!(operations > 1200);
}

#[test]
fn original_support_and_late_delivery_preserve_once_only_credit_and_frozen_snapshots() {
    let mut ledger = Ledger::new(group(), 2, 100).unwrap();
    let a = Accent {
        source_start: 0,
        source_end: 140,
        available_end: 160,
        ..event(64)
    };
    assert_eq!(ledger.deliver(a, a.event_end).unwrap(), None);
    assert_eq!(ledger.deliver(a, 140).unwrap(), None);
    let delivery = ledger.deliver(a, 180).unwrap().unwrap();
    assert_eq!(
        delivery,
        Delivery {
            sequence: 1,
            received_at: 180,
            accent: a
        }
    );
    let frozen = ledger.snapshot(80).unwrap();
    assert_eq!(frozen.accents, [a]);
    assert_eq!(ledger.deliver(a, 200).unwrap(), None);
    assert_eq!(ledger.snapshot(80).unwrap(), frozen);
    let mut changed = a;
    changed.source_end = 150;
    assert!(ledger.deliver(changed, 200).is_err());
    assert_eq!(ledger.snapshot(80).unwrap(), frozen);
    ledger.advance(300).unwrap();
    assert!(ledger.snapshot(200).unwrap().accents.is_empty());
    assert_eq!(frozen.accents, [a]);
    let late = ledger.deliver(event(128), 300).unwrap().unwrap();
    assert_eq!(late.sequence, 2);
    assert!(ledger.bank.is_empty());
    assert_eq!(ledger.cumulative_weight, 1.);
    assert_eq!(ledger.deliver(event(128), 300).unwrap(), None);
}

#[test]
fn ties_expire_before_capacity_and_loss_masks_only_affected_windows() {
    let mut ledger = Ledger::new(group(), 2, 100).unwrap();
    let a = event(0);
    let mut b = a;
    b.event_start += 1;
    b.raw_intervals[1].1 += 1;
    b.raw_intervals[2].0 += 1;
    b.weight = 0.25;
    ledger.deliver(a, 64).unwrap();
    ledger.deliver(b, 64).unwrap();
    ledger.deliver(event(64), 128).unwrap();
    assert_eq!(ledger.bank.front(), Some(&b));
    assert_eq!(ledger.capacity_evicted_through, Some(48));
    ledger.advance(148).unwrap();
    assert_eq!(ledger.bank.front(), Some(&b));
    assert!(!ledger.snapshot(48).unwrap().capacity_valid);
    assert!(ledger.snapshot(49).unwrap().capacity_valid);
    ledger.advance(149).unwrap();
    assert!(!ledger.bank.iter().any(|a| a.event_end == 48));
    ledger.deliver(event(128), 192).unwrap();
    assert_eq!(ledger.capacity_evicted_through, Some(48));
    assert_eq!(ledger.bank.len(), 2);
    let old = ledger.snapshot(92).unwrap();
    let mut fabricated = a;
    fabricated.source_end = 250;
    fabricated.available_end = 250;
    assert!(ledger.deliver(fabricated, 250).is_err());
    assert_eq!(ledger.snapshot(92).unwrap(), old);
}

#[test]
fn malformed_foreign_clock_and_counter_failures_do_not_mutate_ledger() {
    assert!(Ledger::new(group(), 257, 100).is_err());
    assert!(Ledger::new(group(), 0, 100).is_err());
    assert!(Ledger::new(group(), 128, 0).is_err());
    let mut ledger = Ledger::new(group(), 128, 32000).unwrap();
    ledger.deliver(event(0), 64).unwrap();
    let before = ledger.snapshot(0).unwrap();
    for variant in 0..7 {
        let mut a = event(64);
        match variant {
            0 => a.weight = f64::NAN,
            1 => a.weight = 0.,
            2 => a.raw_intervals[0].1 -= 1,
            3 => a.event_start += 1,
            4 => a.source_start += 1,
            5 => a.source_end -= 1,
            _ => a.available_end -= 1,
        }
        assert!(ledger.deliver(a, 200).is_err());
        assert_eq!(ledger.snapshot(0).unwrap(), before);
    }
    let mut foreign = event(64);
    foreign.group.bus = 1;
    assert!(ledger.deliver(foreign, 200).unwrap().is_none());
    assert!(ledger.advance(63).is_err());
    assert!(ledger.snapshot(65).is_err());
    assert_eq!(ledger.snapshot(0).unwrap(), before);
    ledger.cumulative_count = u64::MAX;
    let before = ledger.snapshot(0).unwrap();
    assert!(ledger.deliver(event(64), 128).is_err());
    assert_eq!(ledger.snapshot(0).unwrap(), before);
}

#[test]
fn evicted_tied_id_cannot_return_with_extended_evidence() {
    let mut ledger = Ledger::new(group(), 1, 1000).unwrap();
    let first = event(0);
    let mut second = first;
    second.event_start += 1;
    second.raw_intervals[1].1 += 1;
    second.raw_intervals[2].0 += 1;
    ledger.deliver(first, 64).unwrap();
    ledger.deliver(second, 64).unwrap();
    let before = ledger.snapshot(0).unwrap();
    let mut fabricated = first;
    fabricated.source_end = 80;
    fabricated.available_end = 80;
    assert!(ledger.deliver(fabricated, 80).is_err());
    assert_eq!(ledger.snapshot(0).unwrap(), before);
}

#[test]
fn actual_four_hop_detector_feeds_delayed_once_only_ledger() {
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
    let mut ledger = Ledger::new(group(), 128, 32000).unwrap();
    let mut first = None;
    let mut count = 0;
    for step in 0..80 {
        let energy = if step % 4 < 2 { 1. } else { 16. };
        let start = step * 16;
        let scan = [energy / 9.; 9];
        let result = stream
            .push(
                &grid,
                features::Input {
                    stamp: features::Stamp {
                        group: group(),
                        association: Some(3),
                        grid_id: 1,
                        start,
                        end: start + 16,
                        source_start: start.saturating_sub(32),
                        source_end: start + 16,
                        available_end: start + 16,
                        known_samples: 16,
                        observed: true,
                    },
                    energy: Some(energy),
                    bus_energy: Some(energy),
                    energy_scan: Some(&scan),
                },
                start + 16,
            )
            .unwrap()
            .unwrap();
        if let Some(a) = result.detector.and_then(|d| d.accent) {
            assert!(ledger.deliver(a, a.event_end).unwrap().is_none());
            count += 1;
            let delivery = ledger.deliver(a, a.available_end + 7).unwrap().unwrap();
            assert_eq!(delivery.sequence, count);
            assert_eq!(delivery.accent, a);
            assert_eq!(delivery.received_at, a.available_end + 7);
            assert!(ledger.deliver(a, a.available_end + 7).unwrap().is_none());
            first.get_or_insert_with(|| ledger.snapshot(0).unwrap());
        }
    }
    assert_eq!(count, 20);
    assert_eq!(ledger.snapshot(0).unwrap().cumulative_count, count);
    assert_eq!(first.unwrap().cumulative_count, 1);
    ledger.advance(64000).unwrap();
    let expired = ledger.snapshot(32000).unwrap();
    assert!(expired.accents.is_empty());
    assert_eq!(expired.cumulative_count, 20);
    assert_eq!(expired.cumulative_weight, 20.);
    let next_generation = Ledger::new(
        Handle {
            generation: 4,
            ..group()
        },
        128,
        32000,
    )
    .unwrap();
    assert_eq!(next_generation.snapshot(0).unwrap().cumulative_count, 0);
}

#[test]
#[ignore = "release saturated accent bank and snapshot-copy costs; not full O04"]
fn accent_ledger_cost_probe() {
    use std::{hint::black_box, time::Instant};
    for capacity in [64, 128, 256] {
        let mut ledger = Ledger::new(group(), capacity, 32 * 48000).unwrap();
        let mut admission_times = Vec::with_capacity(6000);
        let mut snapshot_times = Vec::with_capacity(6000);
        let mut snapshot_payload_bytes = 0;
        for i in 0..capacity + 6000 {
            let a = Accent {
                weight: [0.25, 0.5, 0.75][i % 3],
                ..event(i as u64 * 64)
            };
            let start = Instant::now();
            let delivery = ledger
                .deliver(black_box(a), a.available_end)
                .unwrap()
                .unwrap();
            let admission = start.elapsed().as_secs_f64() * 1e6;
            let query_start = ledger.bank.front().unwrap().event_end;
            let start = Instant::now();
            let snapshot = black_box(ledger.snapshot(black_box(query_start)).unwrap());
            let query = start.elapsed().as_secs_f64() * 1e6;
            assert_eq!(snapshot.cumulative_count, delivery.sequence);
            assert!(snapshot.capacity_valid);
            if i >= capacity {
                assert_eq!(snapshot.accents.len(), capacity);
                snapshot_payload_bytes =
                    snapshot.accents.capacity() * std::mem::size_of::<Accent>();
                admission_times.push(admission);
                snapshot_times.push(query);
            }
        }
        admission_times.sort_by(f64::total_cmp);
        snapshot_times.sort_by(f64::total_cmp);
        println!(
            "accent_ledger_cost {}",
            serde_json::json!({
                "capacity":capacity,"calls":6000,"full_O04":false,
                "admission_p99_us":admission_times[5939],"admission_max_us":admission_times[5999],
                "snapshot_p99_us":snapshot_times[5939],"snapshot_max_us":snapshot_times[5999],
                "ledger_header_bytes":std::mem::size_of::<Ledger>(),
                "bank_payload_bytes":ledger.bank.capacity()*std::mem::size_of::<Accent>(),
            "snapshot_payload_bytes":snapshot_payload_bytes,
                "snapshot_header_bytes":std::mem::size_of::<Snapshot>(),
            "scope":"single saturated group; admission reuses bank; snapshot timing includes independent Vec copy and allocation, excludes destruction; source/period pairs/beam/workers/device excluded"
            })
        );
    }
}
