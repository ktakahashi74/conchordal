use super::*;

fn prepared(count: usize) -> Stream {
    let mut stream = Stream::new(100, 1, 0);
    let owners = std::array::from_fn(|i| {
        (i < count).then_some(Handle {
            bus: 0,
            epoch: 1,
            generation: i as u64 + 2,
        })
    });
    stream.advance(1, [None; 7], owners);
    for cut in 2..=801 {
        let input = owners.map(|owner| {
            owner.map(|group| Input {
                group,
                known: true,
                accent: (cut % 10 == 1).then(|| Accent {
                    group,
                    event_start: cut - 2,
                    event_end: cut - 1,
                    raw_intervals: [
                        (cut - 4, cut - 3),
                        (cut - 3, cut - 2),
                        (cut - 2, cut - 1),
                        (cut - 1, cut),
                    ],
                    source_start: cut - 4,
                    source_end: cut,
                    available_end: cut,
                    weight: 0.5,
                    observed_prefix: cut - 1,
                }),
                peaks: [
                    Some(Peak {
                        bin: 0,
                        period_seconds: 0.1,
                        support: 1.,
                    }),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
            })
        });
        stream.advance(cut, input, owners);
    }
    stream
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 2e-11, "{a} != {b}");
}

// Equal-width timestamp uniforms have a triangular difference distribution.
fn independent_bins(records: &[Record], periodic: bool) -> [f64; 33] {
    let mut bins = [0.; 33];
    let total: f64 = records.iter().map(|r| r.weight).sum();
    for r in records {
        let width = r.target[1] - r.target[0];
        close(width, r.anchor[1] - r.anchor[0]);
        let center = (r.target[0] + r.target[1] - r.anchor[0] - r.anchor[1]) / 2.;
        let cdf = |x: f64| {
            let z = (x - center) / width;
            if z <= -1. {
                0.
            } else if z >= 1. {
                1.
            } else if z < 0. {
                (1. + z).powi(2) / 2.
            } else {
                1. - (1. - z).powi(2) / 2.
            }
        };
        let cycles = if periodic {
            ((center - width).floor() as i64)..=((center + width).floor() as i64)
        } else {
            0..=0
        };
        let step = if periodic { 1. / 32. } else { 4. / 32. };
        let mut inside = 0.;
        for cycle in cycles {
            for (i, bin) in bins[..32].iter_mut().enumerate() {
                let left = cycle as f64 + i as f64 * step;
                let mass = cdf(left + step) - cdf(left);
                *bin += r.weight * mass / total;
                inside += mass;
            }
        }
        if !periodic {
            bins[32] += r.weight * (1. - inside) / total;
        }
    }
    bins
}

#[test]
fn issue_identity_and_candidate_periodic_median_within_match_independent_histograms() {
    let stream = prepared(2);
    let group = stream.groups[0].owner.unwrap();
    let before = serde_json::to_value(stream.snapshot()).unwrap();
    let mut scratch = None;
    let at_issue = stream
        .project(group, 801, 801, &mut scratch)
        .unwrap()
        .finish();
    assert_eq!(
        at_issue.values.map(|b| b.map(Feature::value)),
        stream.snapshot().groups[0].unwrap().timing_features
    );
    assert_eq!(at_issue.projected_samples, [0; 3]);
    let mut projection = stream.project(group, 801, 821, &mut scratch).unwrap();
    for end in 802..=821 {
        projection
            .receipt(
                [end - 1, end],
                true,
                [807, 817].contains(&end).then_some(0.8),
            )
            .unwrap();
    }
    let actual = projection.finish();
    assert_eq!(actual.projected_records, [2; 3]);
    assert_eq!(actual.projected_samples, [20; 3]);
    assert!(
        actual
            .values
            .iter()
            .flatten()
            .any(|v| matches!(v, Feature::Projected(_)))
    );
    for (slot, lane) in [0, 1, 12].into_iter().enumerate() {
        let periodic = lane != 1;
        let reference = usize::from(lane != 12);
        let mut expected: Vec<_> = stream.histories[index(0, reference, periodic)]
            .records
            .iter()
            .copied()
            .filter(|r| r.end > 21)
            .collect();
        for end in [807, 817] {
            let anchor = if lane == 12 && end == 817 { 807 } else { 800 };
            let weight = if lane != 12 {
                0.4
            } else if end == 807 {
                0.5
            } else {
                0.8
            };
            let origin = anchor as f64;
            expected.push(Record {
                end,
                target: [
                    (end as f64 - 1. - origin) / 10.,
                    (end as f64 - origin) / 10.,
                ],
                anchor: [-0.1, 0.],
                weight,
                version: u64::from(periodic),
            });
        }
        let summary = actual.histories[slot].unwrap();
        assert_eq!(
            summary.family,
            if periodic {
                Family::Periodic
            } else {
                Family::MedianInterval
            }
        );
        assert_eq!(summary.records, expected.len());
        let bins = independent_bins(&expected, periodic);
        for (a, b) in summary.bins.into_iter().chain([summary.overflow]).zip(bins) {
            close(a, b);
        }
        close(
            summary.retained_weight,
            expected.iter().map(|r| r.weight).sum(),
        );
        close(summary.coverage, 1.);
        assert_eq!(summary.version, u64::from(periodic));
    }
    assert_eq!(before, serde_json::to_value(stream.snapshot()).unwrap());
}

#[test]
fn hypothetical_gaps_break_within_chain_and_cannot_create_a_period_reference() {
    let mut stream = prepared(1);
    let group = stream.groups[0].owner.unwrap();
    let mut scratch = None;
    let mut projection = stream.project(group, 801, 821, &mut scratch).unwrap();
    for end in 802..=821 {
        projection
            .receipt(
                [end - 1, end],
                end != 802,
                [807, 817].contains(&end).then_some(0.8),
            )
            .unwrap();
    }
    let result = projection.finish();
    assert_eq!(result.projected_records[2], 1);
    assert_eq!(result.projected_samples[2], 19);
    assert!(result.histories[2].unwrap().supported);
    stream.groups[0].reference.supported = false;
    let mut projection = stream.project(group, 801, 821, &mut scratch).unwrap();
    for end in 802..=821 {
        projection.receipt([end - 1, end], true, Some(1.)).unwrap();
    }
    let missing = projection.finish();
    assert_eq!(missing.projected_records[2], 0);
    assert_eq!(missing.values, [[Feature::Unsupported; 14]; 3]);
    assert!(!stream.groups[0].reference.supported);
}

#[test]
fn scratch_cap_loss_stays_unknown_and_reuse_restores_original_reference_state() {
    let stream = prepared(2);
    let group = stream.groups[0].owner.unwrap();
    let before = serde_json::to_value(stream.snapshot()).unwrap();
    let mut scratch = None;
    let initial = stream
        .project(group, 801, 801, &mut scratch)
        .unwrap()
        .finish();
    let bytes = scratch.as_ref().unwrap().owned_bytes();
    let mut projection = stream.project(group, 801, 1201, &mut scratch).unwrap();
    for end in 802..=1201 {
        projection.receipt([end - 1, end], true, Some(0.5)).unwrap();
    }
    let result = projection.finish();
    assert_eq!(result.values, [[Feature::Unsupported; 14]; 3]);
    let within = result.histories[2].unwrap();
    assert_eq!(within.records, 128);
    assert!(within.capacity_evicted > 0 && within.coverage < 0.9);
    assert_eq!(result.observed_records[2], 0);
    assert_eq!(result.projected_records[2], 128);
    assert_eq!(scratch.as_ref().unwrap().owned_bytes(), bytes);
    assert!(
        scratch
            .as_ref()
            .unwrap()
            .histories
            .iter()
            .all(|h| h.records.len() <= 128)
    );
    let repeated = stream
        .project(group, 801, 801, &mut scratch)
        .unwrap()
        .finish();
    assert_eq!(
        serde_json::to_value(initial).unwrap(),
        serde_json::to_value(repeated).unwrap()
    );
    assert_eq!(before, serde_json::to_value(stream.snapshot()).unwrap());
    assert!(stream.project(group, 800, 801, &mut scratch).is_none());
    assert!(stream.project(group, 801, 1202, &mut scratch).is_none());
    let retired = Handle {
        generation: 99,
        ..group
    };
    assert!(stream.project(retired, 801, 801, &mut scratch).is_none());
}
