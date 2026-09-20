use super::*;
use crate::temporal_cognition::{features, group};

fn acoustic(cut: u64, handles: &[u64], weights: &[f64], accents: bool) -> frontend::Snapshot {
    let mut out = frontend::Snapshot {
        assignment: group::Assignment {
            end_sample: cut,
            group_handles: [None; 7],
            rows: [None; 8],
            distance_evaluations: 0,
        },
        retained_groups: [None; 7],
        correlation_window: [0, cut],
        group_handles: [None; 8],
        eligible: [false; 8],
        energy: Some([0.; 8]),
        spectral_shape_supported: true,
        features: [None; 8],
        feature_gaps: [None; 8],
        admissions: 0,
        admissions_by_kind: [0; 3],
        rejections: 0,
        retired: [None; 7],
        superseded: 0,
        pending_proposals: 0,
        proposal_conflicts: 0,
        candidate_count: 0,
        continued_candidates: 0,
        bundle_count: 0,
        correlation_reads: 0,
        cross_pair_reads: 0,
    };
    let mut row = group::Row {
        trajectory: Handle {
            bus: 0,
            epoch: 3,
            generation: 1000,
        },
        weights: [0.; 8],
        matched_members: [None; 7],
    };
    for (i, (&id, &weight)) in handles.iter().zip(weights).enumerate() {
        let handle = Handle {
            bus: 0,
            epoch: 3,
            generation: id,
        };
        row.weights[i] = weight;
        out.retained_groups[i] = Some(handle);
        out.group_handles[i] = Some(handle);
        out.assignment.group_handles[i] = Some(handle);
        out.eligible[i] = true;
        out.features[i] = Some(features::Update {
            raw: features::RawDescriptor {
                group: handle,
                start: cut - 100,
                end: cut,
                known_samples: 100,
                values: [None; 10],
                source_start: cut - 100,
                source_end: cut,
                available_end: cut,
            },
            detector: accents.then_some(features::Detection {
                acquisition_coverage: 1.,
                detector_coverage: true,
                saliences: Some([2.; 3]),
                status: features::Status::Admitted,
                accent: Some(Accent {
                    group: handle,
                    event_start: cut - 100,
                    event_end: cut,
                    raw_intervals: [(cut - 100, cut); 4],
                    source_start: cut - 100,
                    source_end: cut,
                    available_end: cut,
                    weight: 1.,
                    observed_prefix: cut,
                }),
            }),
        });
    }
    row.weights[7] = 1. - weights.iter().sum::<f64>();
    out.assignment.rows[0] = Some(row);
    out
}

fn memory(a: &frontend::Snapshot, weights: &[(u64, [f64; 2])]) -> recall::Snapshot {
    recall::Snapshot {
        retrieval: std::array::from_fn(|i| {
            a.retained_groups[i]
                .map(|h| recall::retrieval::fixture(h, a.assignment.end_sample, weights))
        }),
        ..Default::default()
    }
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} != {b}");
}

#[test]
fn physical_coverage_and_assignment_mix_preserve_two_anchor_modes() {
    let mut stream = Stream::new(0, 3, 1000, 1000, 100);
    let a = acoustic(1100, &[2, 3], &[0.25, 0.75], true);
    let mut m = memory(&a, &[(8, [0.8, 0.9])]);
    m.retrieval[1].as_mut().unwrap().entries[0]
        .as_mut()
        .unwrap()
        .weight = [0.4, 0.5];
    let first = stream.advance(&a, None, &m, 0., 1100).unwrap();
    assert_eq!(first.window, [1000, 1100]);
    assert!(first.epoch_clipped);
    close(first.assigned, 0.25 * 0.8 + 0.75 * 0.4);
    let r = first.references[0].unwrap();
    assert_eq!(r.key.family, Family::Nonperiodic);
    close(r.weight_upper, 0.25 * 0.9 + 0.75 * 0.5);
    close(r.anchors[0].unwrap().weight, 0.4);
    close(r.anchors[1].unwrap().weight, 0.6);
    assert!(r.anchors[0].unwrap().period_sec.is_none());
    let mut b = acoustic(1200, &[2, 3], &[0.25, 0.75], false);
    b.features[1].as_mut().unwrap().raw.known_samples = 0;
    let m = memory(&b, &[(8, [0.8, 0.9])]);
    let second = stream.advance(&b, None, &m, 0., 1200).unwrap();
    let ga = second.groups[0].unwrap();
    let gb = second.groups[1].unwrap();
    close(ga.normalized_assignment, 50. / 125.);
    close(gb.coverage, 0.5);
    assert!(gb.anchor.is_none());
    close(second.assigned, 50. / 125. * 0.8);
    close(second.anchor_unsupported[0], 75. / 125. * 0.5 * 0.8);
    close(second.assigned + second.unassigned, 1.);
    assert_eq!(
        second.references[0]
            .unwrap()
            .anchors
            .iter()
            .flatten()
            .count(),
        1
    );
    assert!(stream.advance(&b, None, &m, 0., 1200).is_err());
    let b = acoustic(1250, &[2, 3], &[0.25, 0.75], false);
    assert!(
        stream
            .advance(&b, None, &memory(&b, &[]), 0., 1250)
            .is_err()
    );
    assert_eq!(stream.end, 1200);
}

#[test]
fn retirement_does_not_donate_last_hop_or_history_to_new_generation() {
    let mut stream = Stream::new(0, 3, 0, 1000, 100);
    let a = acoustic(100, &[2, 3], &[0.5, 0.5], true);
    stream
        .advance(&a, None, &memory(&a, &[(8, [0.8; 2])]), 0., 100)
        .unwrap();
    let mut a = acoustic(200, &[2, 3], &[0.5, 0.5], false);
    a.retained_groups[1] = Some(Handle {
        bus: 0,
        epoch: 3,
        generation: 4,
    });
    let s = stream
        .advance(&a, None, &memory(&a, &[(8, [0.8; 2])]), 0., 200)
        .unwrap();
    close(s.assignment_samples, 200.);
    close(s.groups[0].unwrap().normalized_assignment, 0.5);
    close(s.groups[1].unwrap().coverage, 0.);
    close(s.assigned, 0.4);
    let a = acoustic(300, &[2, 4], &[0.5, 0.5], true);
    let s = stream
        .advance(&a, None, &memory(&a, &[(8, [0.8; 2])]), 0., 300)
        .unwrap();
    close(s.groups[1].unwrap().coverage, 1. / 3.);
    close(s.groups[1].unwrap().normalized_assignment, 1. / 6.);
    close(s.assigned, 0.8 * (0.5 + (1. / 6.) * (1. / 3.)));
    let r = s.references[0].unwrap();
    assert!(r.anchors.iter().flatten().all(|a| a.group.generation != 3));
    assert_eq!(r.anchors.iter().flatten().count(), 2);
}

#[test]
fn cutoff_retains_original_credit_and_separates_search_loss() {
    let mut stream = Stream::new(0, 3, 0, 1000, 100);
    let a = acoustic(100, &[2, 3], &[0.5, 0.5], true);
    let weights: Vec<_> = (1..=16).map(|i| (i, [0.04, 0.05])).collect();
    let mut m = memory(&a, &weights);
    let q = m.retrieval[1].as_mut().unwrap();
    q.discarded_weight = [0.1, 0.2];
    for e in q.entries.iter_mut().flatten() {
        e.episode += 16;
        e.generation += 16;
    }
    let s = stream.advance(&a, None, &m, 0., 100).unwrap();
    assert_eq!(s.examined_bindings, 32);
    assert_eq!(s.references.iter().flatten().count(), 16);
    assert_eq!(s.references[15].unwrap().key.episode, 16);
    close(s.assigned, 0.32);
    close(s.inventory_discarded[0], 0.32);
    close(s.inventory_discarded[1], 0.4);
    close(s.search_discarded[0], 0.05);
    close(s.search_discarded[1], 0.1);
    assert!(s.reference_bytes <= 8192);
    assert!(std::mem::size_of::<Reference>() <= 512);
}

#[test]
fn gaps_clipping_and_known_inactivity_use_physical_support() {
    let mut stream = Stream::new(0, 3, 0, 1000, 100);
    let a = acoustic(100, &[2], &[1.], true);
    stream
        .advance(&a, None, &memory(&a, &[(8, [0.8; 2])]), 0., 100)
        .unwrap();
    let mut a = acoustic(300, &[2], &[0.], false);
    a.assignment.rows.fill(None);
    let m = memory(&a, &[(8, [0.8; 2])]);
    let s = stream.advance(&a, None, &m, 0., 300).unwrap();
    close(s.groups[0].unwrap().coverage, 2. / 3.);
    assert!(s.groups[0].unwrap().anchor.is_none());
    close(s.assigned, 0.);
    let ring = stream.frames.capacity();
    let scratch = stream.scratch.capacity();
    for end in (400..=5000).step_by(100) {
        let a = acoustic(end, &[2], &[1.], true);
        let s = stream
            .advance(&a, None, &memory(&a, &[(8, [0.8; 2])]), 0., end)
            .unwrap();
        assert_eq!(s.denominator_samples, end.min(2000));
        if end >= 2300 {
            close(s.groups[0].unwrap().coverage, 1.);
        }
    }
    assert_eq!(stream.frames.capacity(), ring);
    assert_eq!(stream.scratch.capacity(), scratch);
    let a = acoustic(5100, &[], &[], false);
    let s = stream
        .advance(&a, None, &memory(&a, &[]), 0., 5100)
        .unwrap();
    assert_eq!(s.references.iter().flatten().count(), 0);
    close(s.unassigned, 1.);
    let a = acoustic(5200, &[2], &[1.], false);
    let mut m = memory(&a, &[(8, [0.8; 2])]);
    m.retrieval[0].as_mut().unwrap().no_memory_bias = 1.;
    assert!(stream.advance(&a, None, &m, 0., 5200).is_err());
    let mut wrong_epoch = Stream::new(0, 4, 5100, 1000, 100);
    assert!(wrong_epoch.advance(&a, None, &m, 0., 5200).is_err());
}

#[test]
fn supported_periods_keep_distinct_group_anchors_and_reject_future_evidence() {
    use crate::temporal_cognition::{
        accents::{Summary, periods::Peak},
        proposals::frontend::recurrence,
    };
    let mut stream = Stream::new(0, 3, 0, 1000, 100);
    let a = acoustic(100, &[2, 3], &[0.5, 0.5], true);
    let mut period = recurrence::Snapshot {
        timing: None,
        groove_heads: None,
        end_sample: 100,
        received_at: 100,
        residual: Summary {
            group: Handle {
                bus: 0,
                epoch: 3,
                generation: 1,
            },
            received_at: 100,
            retained_accents: 0,
            cumulative_count: 0,
            cumulative_weight: 0.,
            capacity_evicted_through: None,
        },
        groups: [None; 7],
    };
    for i in 0..2 {
        let mut ledger = period.residual;
        ledger.group = a.retained_groups[i].unwrap();
        ledger.retained_accents = 2;
        period.groups[i] = Some(recurrence::GroupSnapshot {
            groove: None,
            ledger,
            active: true,
            acoustic_eligible: true,
            association_known: true,
            peaks: [None; 8],
            period_source: Some([0, 100, 100]),
            grouping: None,
            forecast: None,
        });
        period.groups[i].as_mut().unwrap().peaks[0] = Some(Peak {
            bin: 96 + i * 48,
            period_seconds: 0.5 + i as f64 * 0.5,
            support: 0.8,
        });
    }
    let s = stream
        .advance(&a, Some(&period), &memory(&a, &[(8, [0.8; 2])]), 0., 100)
        .unwrap();
    let r = s.references[0].unwrap();
    assert_eq!(r.key.family, Family::Periodic);
    assert_eq!(r.anchors[0].unwrap().period_sec, Some(0.5));
    assert_eq!(r.anchors[1].unwrap().period_sec, Some(1.));
    close(r.weight, 0.8);
    let a = acoustic(200, &[2, 3], &[0.5, 0.5], false);
    period.end_sample = 200;
    period.received_at = 200;
    period.groups[0].as_mut().unwrap().period_source = Some([0, 100, 300]);
    let s = stream
        .advance(&a, Some(&period), &memory(&a, &[(8, [0.8; 2])]), 0., 200)
        .unwrap();
    assert_eq!(s.references.iter().flatten().count(), 2);
    assert_eq!(s.references[0].unwrap().key.family, Family::Nonperiodic);
    assert_eq!(s.references[1].unwrap().key.family, Family::Periodic);
    close(s.references[0].unwrap().weight, 0.4);
    close(s.references[1].unwrap().weight, 0.4);
}
