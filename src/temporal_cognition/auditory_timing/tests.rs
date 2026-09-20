use super::*;

fn handle(generation: u64) -> Handle {
    Handle {
        bus: 0,
        epoch: 1,
        generation,
    }
}

fn accent(group: Handle, end: u64) -> Accent {
    Accent {
        group,
        event_start: end - 1,
        event_end: end,
        raw_intervals: [
            (end - 3, end - 2),
            (end - 2, end - 1),
            (end - 1, end),
            (end, end + 1),
        ],
        source_start: end - 3,
        source_end: end + 1,
        available_end: end + 1,
        weight: 0.5,
        observed_prefix: end,
    }
}

fn peak(bin: usize, support: f64) -> Peak {
    Peak {
        bin,
        period_seconds: 0.1 * (bin + 1) as f64,
        support,
    }
}

fn run(
    stream: &mut Stream,
    cut: u64,
    events: [Option<u64>; 7],
    peaks: [Option<Peak>; 8],
    known: bool,
) {
    let owners = stream.groups.map(|g| g.owner);
    stream.advance(
        cut,
        std::array::from_fn(|i| {
            owners[i].map(|group| Input {
                group,
                known,
                accent: events[i].map(|end| accent(group, end)),
                peaks,
            })
        }),
        owners,
    );
}

fn stream(count: usize) -> Stream {
    let mut stream = Stream::new(100, 1, 0);
    let owners = std::array::from_fn(|i| (i < count).then_some(handle(i as u64 + 2)));
    stream.advance(1, [None; 7], owners);
    stream
}

#[test]
fn batch_freezes_periods_before_new_accents_and_within_uses_previous_freeze() {
    let mut stream = stream(2);
    for cut in 2..=41 {
        let event = [11, 21, 31, 41].contains(&cut).then_some(cut - 1);
        let mut events = [None; 7];
        events[..2].fill(event);
        let mut peaks = [None; 8];
        if cut >= 11 {
            peaks[0] = Some(peak(0, 1.));
        }
        run(&mut stream, cut, events, peaks, true);
        if cut == 11 {
            assert_eq!(stream.summary(0, 1, true).unwrap().records, 0);
            assert_eq!(stream.summary(0, 0, true).unwrap().records, 0);
        }
        if cut == 21 {
            assert_eq!(stream.summary(0, 1, true).unwrap().records, 1);
            assert_eq!(stream.summary(0, 0, true).unwrap().records, 0);
        }
    }
    let within = stream.summary(0, 0, true).unwrap();
    assert!(within.supported);
    assert_eq!(within.records, 2);
    assert_eq!(within.retained_weight, 1.);
    let inter = stream.summary(0, 1, true).unwrap();
    assert_eq!(inter.records, 3);
    assert_eq!(inter.retained_weight, 0.75);
    assert!(inter.bins[0] > 0. && inter.bins[31] > 0.);
    let nonperiodic = stream.summary(0, 1, false).unwrap();
    assert_eq!(nonperiodic.records, 2);
    // Simultaneous uncertain endpoints straddle zero; the linear family never wraps.
    assert!((nonperiodic.overflow - 0.5).abs() < 1e-12);
}

#[test]
fn challenger_requires_continuous_dwell_ratio_and_resets_on_gaps_or_changes() {
    let mut reference = Reference::default();
    let mut peaks = [None; 8];
    peaks[0] = Some(peak(0, 0.4));
    assert!(reference.update(peaks, true, 0, 100));
    peaks[1] = Some(peak(1, 0.45));
    assert!(!reference.update(peaks, true, 200, 100));
    assert_eq!(reference.challenger, None);
    peaks[1] = Some(peak(1, 0.6));
    assert!(!reference.update(peaks, true, 201, 100));
    assert!(!reference.update(peaks, true, 251, 100));
    assert!(!reference.update(peaks, false, 252, 100));
    assert!(!reference.supported);
    assert!(!reference.update(peaks, true, 300, 100));
    peaks[2] = Some(peak(2, 0.7));
    assert!(!reference.update(peaks, true, 350, 100));
    assert!(!reference.update(peaks, true, 400, 100));
    assert!(reference.update(peaks, true, 550, 100));
    assert_eq!(reference.peak.unwrap().bin, 2);
    assert_eq!(reference.version, 2);
    // Unsupported incumbent stays dormant; replacement still needs one second.
    peaks.fill(None);
    peaks[0] = Some(peak(0, 0.1));
    assert!(!reference.update(peaks, true, 551, 100));
    assert!(!reference.supported);
    assert!(!reference.update(peaks, true, 601, 100));
    assert!(reference.update(peaks, true, 651, 100));
    let mut tie = Reference::default();
    peaks[0] = Some(peak(9, 0.5));
    peaks[1] = Some(peak(2, 0.5));
    tie.update(peaks, true, 0, 100);
    assert_eq!(tie.peak.unwrap().bin, 2);
}

#[test]
fn median_uses_last_four_completed_intervals_and_gap_restarts_minimum_count() {
    let mut group = Group::default();
    for (end, expected) in [
        (10, None),
        (20, None),
        (40, Some(15.)),
        (70, Some(20.)),
        (110, Some(25.)),
        (160, Some(35.)),
    ] {
        group.observe(accent(handle(2), end));
        assert_eq!(group.scale(), expected);
    }
    let mut after_gap = accent(handle(2), 170);
    after_gap.observed_prefix -= 1;
    group.observe(after_gap);
    assert_eq!(group.scale(), None);
    for end in [190, 230] {
        let mut a = accent(handle(2), end);
        a.observed_prefix -= 1;
        group.observe(a);
    }
    assert_eq!(group.scale(), Some(30.));
}

#[test]
fn integrated_bins_match_independent_uniform_timestamp_grid_and_large_clocks() {
    for periodic in [true, false] {
        for (target, anchor, scale) in [(32, 11, 10.), (41, 40, 8.), (74, 11, 10.), (31, 31, 10.)] {
            let a = accent(handle(2), target);
            let b = accent(handle(3), anchor);
            let record = Record::new(a, b, scale, 0.3, 8);
            let masses = record.masses(periodic);
            let mut reference = [0.; 33];
            let n = 401;
            for i in 0..n {
                for j in 0..n {
                    let x = ((target - 1) as f64 + (i as f64 + 0.5) / n as f64
                        - (anchor - 1) as f64
                        - (j as f64 + 0.5) / n as f64)
                        / scale;
                    let bin = if periodic {
                        (x.rem_euclid(1.) * 32.).floor() as usize
                    } else if (0. ..4.).contains(&x) {
                        (x * 8.).floor() as usize
                    } else {
                        32
                    };
                    reference[bin] += 1. / f64::from(n * n);
                }
            }
            assert!((masses.iter().sum::<f64>() - 1.).abs() < 1e-12);
            for bin in 0..33 {
                assert!(
                    (masses[bin] - reference[bin]).abs() < 0.003,
                    "periodic={periodic} target={target} anchor={anchor} bin={bin}"
                );
            }
            let shift = 1 << 60;
            let mut a2 = a;
            let mut b2 = b;
            a2.event_start += shift;
            a2.event_end += shift;
            b2.event_start += shift;
            b2.event_end += shift;
            assert_eq!(Record::new(a2, b2, scale, 0.3, 8).masses(periodic), masses);
        }
    }
}

#[test]
fn retention_subtracts_expired_mass_and_reports_capacity_and_reference_loss() {
    let mut history = History::new();
    let bytes = history.records.capacity();
    for end in 10..150 {
        history.insert(
            Record::new(
                accent(handle(2), end),
                accent(handle(3), end - 3),
                10.,
                0.5,
                1,
            ),
            true,
        );
    }
    assert_eq!(history.records.len(), 128);
    assert_eq!(history.capacity_evicted, 12);
    assert_eq!(history.lost_through, 21);
    assert_eq!(history.records.capacity(), bytes);
    history.expire(140, true);
    assert_eq!(history.records.len(), 9);
    let expected = history.records[0].masses(true).map(|m| m * 4.5);
    for (a, b) in history.masses.iter().zip(expected) {
        assert!((a - b).abs() < 1e-12);
    }
    history.clear();
    assert_eq!(history.discarded, 9);
    assert_eq!(history.masses, [0.; 33]);
    assert_eq!(history.lost_through, 0);
    assert!(std::mem::size_of::<Record>() <= 64);
}

#[test]
fn gaps_mask_retained_evidence_and_break_within_intervals_without_zero_imputation() {
    let mut stream = stream(1);
    let mut peaks = [None; 8];
    peaks[0] = Some(peak(0, 1.));
    for cut in 2..=61 {
        let mut events = [None; 7];
        if [11, 21, 31, 51, 61].contains(&cut) {
            events[0] = Some(cut - 1);
        }
        run(&mut stream, cut, events, peaks, !(32..=50).contains(&cut));
        if cut == 31 {
            assert_eq!(stream.summary(0, 0, true).unwrap().records, 2);
        }
        if cut == 40 {
            let s = stream.summary(0, 0, true).unwrap();
            assert_eq!(s.records, 2);
            assert!(!s.supported);
            assert_eq!(s.reference_support, 0.);
        }
    }
    assert_eq!(stream.summary(0, 0, true).unwrap().records, 2);
    assert!(!stream.summary(0, 0, true).unwrap().supported);
    for cut in 62..=1000 {
        run(&mut stream, cut, [None; 7], peaks, true);
    }
    assert_eq!(stream.summary(0, 0, true).unwrap().records, 0);
    assert!(!stream.summary(0, 0, true).unwrap().supported);
}

#[test]
fn group_order_does_not_change_simultaneous_or_earlier_anchor_selection() {
    let mut a = stream(2);
    let mut b = stream(2);
    b.groups.swap(0, 1);
    // No observations exist yet, so swapping the two initial owner slots is safe.
    let mut peaks = [None; 8];
    peaks[0] = Some(peak(0, 1.));
    for cut in 2..=101 {
        let mut events = [None; 7];
        if cut % 10 == 1 {
            events[0] = Some(cut - 1);
            events[1] = Some(cut - 2);
        }
        run(&mut a, cut, events, peaks, true);
        events.swap(0, 1);
        run(&mut b, cut, events, peaks, true);
    }
    for (i, j) in [(0, 1), (1, 0)] {
        for periodic in [true, false] {
            let x = a.summary(i, j, periodic).unwrap();
            let y = b.summary(1 - i, 1 - j, periodic).unwrap();
            assert_eq!(
                serde_json::to_value(x).unwrap(),
                serde_json::to_value(y).unwrap()
            );
        }
    }
}

#[test]
fn retirement_clears_both_pair_directions_and_within_but_preserves_other_groups() {
    let mut stream = stream(3);
    let mut peaks = [None; 8];
    peaks[0] = Some(peak(0, 1.));
    for cut in 2..=101 {
        let mut events = [None; 7];
        if cut % 10 == 1 {
            events[..3].fill(Some(cut - 1));
        }
        run(&mut stream, cut, events, peaks, true);
    }
    let old = stream.summary(1, 2, true).unwrap().records;
    let owners = stream.groups.map(|g| g.owner);
    let mut next = owners;
    next[0] = Some(handle(99));
    stream.advance(
        102,
        std::array::from_fn(|i| {
            owners[i].map(|group| Input {
                group,
                known: true,
                accent: None,
                peaks,
            })
        }),
        next,
    );
    assert_eq!(stream.summary(0, 1, true).unwrap().records, 0);
    assert_eq!(stream.summary(1, 0, false).unwrap().records, 0);
    assert_eq!(stream.summary(0, 0, true).unwrap().records, 0);
    assert_eq!(stream.summary(1, 2, true).unwrap().records, old);
    assert!(stream.snapshot().discarded > 0);
}

#[test]
fn reference_replacement_discards_periodic_records_and_preserves_median_family() {
    let mut stream = stream(2);
    let mut discarded = 0;
    for cut in 2..=231 {
        let mut events = [None; 7];
        if cut % 10 == 1 {
            events[..2].fill(Some(cut - 1));
        }
        let mut peaks = [None; 8];
        peaks[0] = Some(peak(0, 0.4));
        if cut >= 101 {
            peaks[1] = Some(peak(1, 0.6));
        }
        run(&mut stream, cut, events, peaks, true);
        if cut == 201 {
            let inter = stream.summary(0, 1, true).unwrap();
            assert_eq!(inter.version, 2);
            assert_eq!(inter.records, 0);
            assert!(inter.discarded > 0);
            assert_eq!(stream.summary(0, 0, true).unwrap().records, 0);
            assert!(stream.summary(0, 1, false).unwrap().records > 0);
            discarded = inter.discarded;
        }
        if cut == 211 {
            assert_eq!(stream.summary(0, 0, true).unwrap().records, 0);
        }
        if cut == 221 {
            assert_eq!(stream.summary(0, 0, true).unwrap().records, 1);
        }
    }
    assert!(discarded > 0);
    assert_eq!(stream.summary(0, 1, false).unwrap().discarded, 0);
}

#[test]
fn capacity_loss_masks_public_coverage_and_owned_storage_stays_bounded() {
    let mut stream = stream(2);
    let bytes = stream.snapshot().owned_bytes;
    let mut peaks = [None; 8];
    peaks[0] = Some(peak(0, 1.));
    for cut in 2..=2000 {
        let mut events = [None; 7];
        if cut >= 11 {
            events[..2].fill(Some(cut - 1));
        }
        run(&mut stream, cut, events, peaks, true);
    }
    let inter = stream.summary(0, 1, true).unwrap();
    assert_eq!(inter.records, 128);
    assert!(inter.capacity_evicted > 0);
    assert!(inter.coverage < 0.9);
    assert!(!inter.supported);
    assert_eq!(stream.snapshot().owned_bytes, bytes);
    assert_eq!(stream.snapshot().groups[0].unwrap().supported_outgoing, 0);
    assert!(stream.coverage.len() <= stream.coverage_capacity);
}

#[test]
fn normal_snapshot_exposes_three_feature_blocks_and_normalizes_selected_history_weights() {
    let mut stream = stream(3);
    let mut peaks = [None; 8];
    peaks[0] = Some(peak(0, 1.));
    for cut in 2..=101 {
        let mut events = [None; 7];
        if cut % 10 == 1 {
            events[..3].fill(Some(cut - 1));
        }
        run(&mut stream, cut, events, peaks, true);
    }
    let snapshot = stream.snapshot();
    let first = snapshot.groups[0].unwrap();
    assert_eq!(first.outgoing[0].unwrap().reference, handle(3));
    assert_eq!(first.outgoing[1].unwrap().reference, handle(4));
    assert_eq!(first.supported_outgoing, 4);
    assert_eq!(first.timing_features[0][9], Some(0.5));
    assert_eq!(first.timing_features[1][9], Some(0.5));
    assert_eq!(first.timing_features[2][9], Some(1.));
    for values in first.timing_features {
        assert_eq!(values[8], Some(1.));
        assert!(values[10].unwrap().is_finite());
        assert_eq!(values[12], Some(0.));
        assert!(values[13].unwrap() >= 0.9);
    }
    assert!(stream.histories[index(0, 1, true)].shape.get().is_some());
    assert!(stream.histories[index(0, 1, false)].shape.get().is_none());
    assert!(snapshot.groups[3..].iter().all(Option::is_none));
    // Acoustic coverage masks both diagnostics and all42 feature values after a gap.
    run(&mut stream, 102, [None; 7], peaks, false);
    let first = stream.snapshot().groups[0].unwrap();
    assert!(first.outgoing.iter().all(Option::is_none));
    assert_eq!(first.timing_features, [[None; 14]; 3]);
}
