use super::*;
use crate::temporal_cognition::{features, group, ridge::Handle};

thread_local! {
    pub(in crate::temporal_cognition) static PROJECTION_NANOS: std::cell::Cell<Option<[u64; 11]>> = const { std::cell::Cell::new(None) };
}

pub(in crate::temporal_cognition) fn projection_clock() -> Option<std::time::Instant> {
    PROJECTION_NANOS.with(|cost| cost.get().map(|_| std::time::Instant::now()))
}

pub(in crate::temporal_cognition) fn record_projection_cost(
    clock: &mut Option<std::time::Instant>,
    index: usize,
) {
    if let Some(start) = clock {
        let now = std::time::Instant::now();
        PROJECTION_NANOS.with(|cost| {
            let mut values = cost.get().unwrap();
            values[index] += now.duration_since(*start).as_nanos() as u64;
            cost.set(Some(values));
        });
        *start = now;
    }
}

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

pub(in crate::temporal_cognition) fn ridges() -> super::super::ridge::Update {
    super::super::ridge::Update {
        observed: true,
        current: [None; 7],
        retired: [None; 7],
        superseded: [None; 7],
        evicted: [None; 7],
        distance_evaluations: 0,
    }
}

fn hop_input(step: u64) -> frontend::Snapshot {
    let mut acoustic = input(step, 0.5);
    let end = step * 512;
    acoustic.assignment.end_sample = end;
    acoustic.correlation_window = [end.saturating_sub(96000), end];
    acoustic.energy = Some([1., 0., 0., 0., 0., 0., 0., 0.]);
    let raw = &mut acoustic.features[0].as_mut().unwrap().raw;
    raw.start = end - 512;
    raw.end = end;
    raw.known_samples = 512;
    raw.source_start = end.saturating_sub(2048);
    raw.source_end = end;
    raw.available_end = end;
    acoustic
}

#[test]
fn observed_context_keeps_group_identity_and_bounded_history() {
    let mut context = Context::new(1, 0, 0, 48000, 512).unwrap();
    let mut model = gesture::Gesture::new(
        1,
        0,
        48000,
        512,
        crate::config::TemporalGestureConfig {
            rms_reference: 0.1,
            means: [0.; 5],
            deviations: [1.; 5],
            coefficients: [[[0.; 11]; 4]; 4],
        },
    )
    .unwrap();
    for step in 1..=300 {
        let acoustic = hop_input(step);
        model.advance(&acoustic, &ridges(), step * 512).unwrap();
        context.advance(&acoustic, None, step * 512).unwrap();
    }
    let snapshot = context.snapshot();
    assert_eq!(snapshot.end_sample, 300 * 512);
    assert!(!snapshot.censored);
    let group = snapshot.groups[0].unwrap();
    assert_eq!(group.group.bus, 1);
    // Two seconds of 512-sample hops, plus the partially covered edge.
    assert!(group.history_hops <= (2 * 48000_usize).div_ceil(512) + 1);
    assert!(group.acoustic_weight > 0.);
    assert!((snapshot.observed_coverage - 1.).abs() < 1e-9);

    let descriptor = context.body_descriptors()[0].unwrap();
    assert_eq!(descriptor.group, group.group);
    assert_eq!(descriptor.end, 300 * 512);
    assert!(descriptor.available <= descriptor.end);

    // Backward observation and EOF censoring.
    assert!(context.advance(&hop_input(1), None, 512).is_err());
    context.finish(300 * 512 + 1000).unwrap();
    assert!(context.snapshot().censored);
    assert_eq!(context.snapshot().end_sample, 300 * 512 + 1000);
    assert!(context.finish(0).is_err());
}

#[test]
fn candidate_windows_read_the_observed_prefix_without_changing_it() {
    use crate::life::action_candidates::Class;
    use crate::temporal_cognition::action_profiles;
    let profiles = action_profiles::tests::model();
    let mut context = Context::new(1, 0, 0, 48000, 512).unwrap();
    let mut model = gesture::Gesture::new(
        1,
        0,
        48000,
        512,
        crate::config::TemporalGestureConfig {
            rms_reference: 0.1,
            means: [0.; 5],
            deviations: [1.; 5],
            coefficients: [[[0.; 11]; 4]; 4],
        },
    )
    .unwrap();
    for step in 1..=8 {
        let acoustic = hop_input(step);
        model.advance(&acoustic, &ridges(), step * 512).unwrap();
        context.advance(&acoustic, None, step * 512).unwrap();
    }
    let issue = 8 * 512;
    let handle = context.snapshot().groups[0].unwrap().group;
    let before = serde_json::to_value(context.snapshot()).unwrap();
    let cell = context
        .project_action_window(
            &profiles,
            0,
            Class::Continue,
            0,
            issue + 4096,
            handle,
            1.,
            Some(&model),
            None,
            None,
        )
        .unwrap();
    assert_eq!(cell.issued_at, issue);
    assert_eq!(cell.action_at, issue);
    assert_eq!(cell.evaluation_at, issue + 4096);
    assert_eq!(cell.group, handle);
    assert!(cell.window_start <= issue);
    // Projected frames may only extend the window past the issue clock.
    assert!(
        cell.short_projected_fraction
            .iter()
            .all(|v| (0. ..=1.).contains(v))
    );
    assert!(
        cell.short_observed_fraction
            .iter()
            .zip(cell.short_projected_fraction)
            .all(|(o, p)| o + p <= 1. + 1e-12)
    );
    assert!(matches!(
        cell.accent_density.value,
        Feature::Unsupported | Feature::Observed(_) | Feature::Projected(_)
    ));
    assert_eq!(cell.context_features[2], cell.grouping);

    // A foreign or retired group, a rewound clock and an over-long horizon are refused.
    let mut retired = handle;
    retired.generation = u64::MAX;
    for (group, evaluation) in [
        (retired, issue),
        (handle, issue - 1),
        (handle, issue + 192_001),
    ] {
        assert!(
            context
                .project_action_window(
                    &profiles,
                    0,
                    Class::Continue,
                    0,
                    evaluation,
                    group,
                    1.,
                    Some(&model),
                    None,
                    None,
                )
                .is_none()
        );
    }
    assert_eq!(before, serde_json::to_value(context.snapshot()).unwrap());
}
