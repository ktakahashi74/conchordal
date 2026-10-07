use super::*;
use crate::temporal_cognition::{features, group, ridge::Handle};

pub(in crate::temporal_cognition) fn input(step: u64, value: f64) -> frontend::Snapshot {
    let handle = Handle {
        bus: 1,
        epoch: 0,
        generation: 2,
    };
    let mut out = frontend::Snapshot {
        assignment: group::Assignment {
            end_sample: step * 100,
            group_handles: [None; 7],
            rows: [None; 8],
            distance_evaluations: 0,
        },
        retained_groups: [None; 7],
        correlation_window: [0, step * 100],
        group_handles: [None; 8],
        eligible: [false; 8],
        energy: None,
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
    out.retained_groups[0] = Some(handle);
    out.group_handles[0] = Some(handle);
    let mut values = [None; 10];
    values[0] = Some(8.);
    values[1] = Some(value);
    out.features[0] = Some(features::Update {
        raw: features::RawDescriptor {
            group: handle,
            start: (step - 1) * 100,
            end: step * 100,
            known_samples: 100,
            values,
            source_start: (step * 100).saturating_sub(400),
            source_end: step * 100,
            available_end: step * 100,
        },
        detector: None,
    });
    out.eligible[0] = true;
    out.assignment.group_handles[0] = Some(handle);
    out.assignment.rows[0] = Some(group::Row {
        trajectory: Handle {
            generation: 4,
            ..handle
        },
        weights: [1., 0., 0., 0., 0., 0., 0., 0.],
        matched_members: [None; 7],
    });
    let raw = &mut out.features[0].as_mut().unwrap().raw;
    raw.values[2] = Some(-3.);
    raw.values[3] = Some(0.);
    raw.values[4] = Some(0.);
    raw.values[5] = Some(0.);
    raw.values[6] = Some(1.);
    out
}
