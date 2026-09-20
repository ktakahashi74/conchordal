use super::*;

#[test]
fn raw_head_mixture_freezes_other_groups_and_never_redistributes_missing_path_mass() {
    let profiles = crate::temporal_cognition::action_profiles::tests::model();
    let mut phrase = Phrase::new(1, 0, 0, 48000, 512, super::super::tests::config()).unwrap();
    let mut group = group();
    group.end = 512;
    phrase.end = 512;
    let handle = group.handle;
    phrase.groups[0] = Some(group);
    let mut cell = phrase
        .project_action_window(
            &profiles,
            0,
            crate::life::action_candidates::Class::OnsetNow,
            0,
            512,
            handle,
            1.,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    cell.continuation_paths.fill(None);
    let mut issue = phrase.issue_heads();
    issue.observed_coverage = 0.8;
    issue.priors = [[0.1, 0.2, 0.4, 0.2, 0.1]; 2];
    issue.groups[0] = Some(IssueGroupHead {
        group: handle,
        acoustic_weight: 0.4,
        known_mass: 0.5,
        categories: [None; 2],
    });
    issue.groups[1] = Some(IssueGroupHead {
        group: Handle {
            generation: handle.generation + 1,
            ..handle
        },
        acoustic_weight: 0.6,
        known_mass: 0.5,
        categories: [Some([0., 0., 1., 0., 0.]); 2],
    });
    cell.continuation_paths[0] = Some(ConditionalOrdinal {
        path_index: 0,
        issue_mass: 0.2,
        foreground_start: None,
        survival: Feature::Observed(1.),
        categories: Some([1., 0., 0., 0., 0.]),
    });
    cell.continuation_paths[1] = Some(ConditionalOrdinal {
        path_index: 1,
        issue_mass: 0.3,
        foreground_start: None,
        survival: Feature::Observed(1.),
        categories: Some([0., 0., 0., 0., 1.]),
    });
    cell.issue_phrase_unknown = 0.5;
    cell.closure = None;
    let result = issue.project(&cell).unwrap();
    assert!((result.continuation.supported_mass - 0.5).abs() < 1e-14);
    assert!((result.continuation.selected_supported_mass - 0.2).abs() < 1e-14);
    assert_eq!(result.continuation.held_supported_mass, 0.3);
    for (actual, expected) in result
        .continuation
        .categories
        .into_iter()
        .zip([0.124, 0.12, 0.48, 0.12, 0.156])
    {
        assert!((actual - expected).abs() < 1e-14);
    }
    assert_eq!(result.closure.selected_supported_mass, 0.);
    assert_eq!(result.closure.supported_mass, 0.3);
    cell.continuation_paths[1].as_mut().unwrap().categories = None;
    let partial = issue.project(&cell).unwrap();
    assert!((partial.continuation.supported_mass - 0.38).abs() < 1e-14);
    assert!((partial.continuation.reported_support_mass - 0.304).abs() < 1e-14);
    issue.groups[1].as_mut().unwrap().categories = [None; 2];
    cell.continuation_paths[0].as_mut().unwrap().categories = None;
    let unknown = issue.project(&cell).unwrap();
    assert_eq!(unknown.continuation.supported_mass, 0.);
    for (a, b) in unknown
        .continuation
        .categories
        .into_iter()
        .zip(issue.priors[1])
    {
        assert!((a - b).abs() < 1e-14);
    }
    cell.issued_at += 1;
    assert!(issue.project(&cell).is_none());
    cell.issued_at -= 1;
    cell.issue_phrase_unknown = 0.;
    assert!(issue.project(&cell).is_none());
}

#[test]
fn accent_density_requires_observed_support_and_keeps_known_silence() {
    let mut group = group();
    assert_eq!(group.accent_density(0, 100, 100, 1000), Some(0.));
    group
        .accents
        .push_back(super::super::super::features::Accent {
            group: group.handle,
            event_start: 20,
            event_end: 50,
            raw_intervals: [(0, 25), (25, 50), (50, 75), (75, 100)],
            source_start: 0,
            source_end: 100,
            available_end: 100,
            weight: 0.5,
            observed_prefix: 100,
        });
    assert_eq!(group.accent_density(0, 100, 100, 1000), Some(5.));
    assert_eq!(group.accent_density(0, 110, 100, 1000), Some(5.));
    assert_eq!(group.accent_density(0, 112, 100, 1000), None);
    group.accents[0].available_end = 101;
    assert_eq!(group.accent_density(0, 100, 100, 1000), Some(0.));
    group.history[0].raw.available_end = 101;
    assert_eq!(group.accent_density(0, 100, 100, 1000), None);
    group.history[0].raw.available_end = 100;
    group.evicted_accent = Some(50);
    assert_eq!(group.accent_density(0, 100, 100, 1000), None);
    assert_eq!(group.accent_density(51, 100, 100, 1000), Some(0.));
}

fn group() -> Group {
    let mut phrase = Phrase::new(
        1,
        0,
        0,
        1000,
        100,
        crate::temporal_cognition::phrase::tests::config(),
    )
    .unwrap();
    let mut gesture = crate::temporal_cognition::phrase::tests::gesture_model();
    let acoustic = crate::temporal_cognition::phrase::tests::input(1, 0.5);
    gesture
        .advance(
            &acoustic,
            &crate::temporal_cognition::phrase::tests::ridges(),
            100,
        )
        .unwrap();
    phrase
        .advance(&acoustic, &gesture, None, None, 100)
        .unwrap();
    let mut group = phrase.groups[0].take().unwrap();
    group.history[0].residual = Some(2.);
    group.history[0].residual_interval = (0, 100);
    group.cue = Some(recall::Prediction {
        episode_id: 7,
        query_id: 9,
        group: group.handle,
        issued_at: 50,
        source_start: 0,
        source_end: 50,
        available: 50,
        expected_start: 100,
        expected_end: 200,
        values: [
            Some(0.),
            Some(0.),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        scales: [2., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    });
    group
}

fn future() -> window::Frame {
    let mut raw = [Feature::Unsupported; 10];
    raw[0] = Feature::Projected(2.);
    raw[1] = Feature::Projected(2.);
    window::Frame {
        start: 100,
        end: 200,
        source_end: 200,
        available: 200,
        raw,
        energy: Feature::Projected(1.),
    }
}

fn accent_material() -> (Group, Vec<window::Frame>) {
    let mut g = group();
    let sample = g.history[0];
    g.history.clear();
    for i in 0..10 {
        let mut s = sample;
        s.raw.start = i * 100;
        s.raw.end = (i + 1) * 100;
        s.raw.source_end = s.raw.end;
        s.raw.available_end = s.raw.end;
        s.raw.values[3] = Some(0.);
        s.raw.values[5] = Some(0.);
        s.energy = Some(1.);
        g.history.push_back(s);
    }
    g.end = 1000;
    let frames = (0..10)
        .map(|i| {
            let mut f = future();
            f.start = 1000 + i * 100;
            f.end = f.start + 100;
            f.source_end = f.end;
            f.available = f.end;
            f.raw[3] = Feature::Projected(if i == 4 || i == 5 { 4. } else { 0. });
            f.raw[5] = Feature::Projected(0.);
            f
        })
        .collect();
    (g, frames)
}

#[test]
fn candidate_accent_density_keeps_plateau_ties_support_and_observed_ledger() {
    let (g, frames) = accent_material();
    let original = serde_json::to_string(&g.snapshot).unwrap();
    let at_issue = g
        .projected_accent_density(
            [0, 1000],
            1000,
            1000,
            ([0.; 2], [1.; 2]),
            None,
            None,
            None,
            frames.iter().copied(),
        )
        .unwrap();
    assert_eq!(at_issue.value, Feature::Observed(0.));
    assert_eq!(at_issue.projected_samples, 0);
    let future = g
        .projected_accent_density(
            [0, 2000],
            1000,
            1000,
            ([0.; 2], [1.; 2]),
            None,
            None,
            None,
            frames.iter().copied(),
        )
        .unwrap();
    assert_eq!(future.observed_samples, 1000);
    assert_eq!(future.projected_samples, 900);
    assert_eq!(future.projected_accents, 1);
    assert_eq!(future.projected_weight, 1.);
    assert_eq!(future.latest_projected_interval, Some([1400, 1500]));
    assert_eq!(future.value, Feature::Projected(1000. / 1900.));
    let clipped = g
        .projected_accent_density(
            [0, 1500],
            1000,
            1000,
            ([0.; 2], [1.; 2]),
            None,
            None,
            None,
            frames.iter().copied(),
        )
        .unwrap();
    assert_eq!(clipped.projected_samples, 400);
    assert_eq!(clipped.projected_accents, 0);
    assert_eq!(clipped.value, Feature::Projected(0.));
    assert_eq!(original, serde_json::to_string(&g.snapshot).unwrap());
    assert!(g.accents.is_empty());
}

#[test]
fn groove_windows_consume_the_same_supported_candidate_accents() {
    use crate::temporal_cognition::groove::DensityProjection;
    let (g, frames) = accent_material();
    let starts = [1875, 1750, 1500, 1000, 0, 0, 0, 0];
    let mut density = DensityProjection {
        issued_at: 1000,
        evaluated_at: 2000,
        starts,
        values: [Feature::Unsupported; 8],
        observed_samples: starts.map(|s| 1000_u64.saturating_sub(s)),
        projected_samples: [0; 8],
        observed_weight: [0.; 8],
        projected_weight: [0.; 8],
        retention_supported: [true; 8],
    };
    let continuation = g
        .projected_accent_density(
            [0, 2000],
            1000,
            1000,
            ([0.; 2], [1.; 2]),
            None,
            Some(&mut density),
            None,
            frames.iter().copied(),
        )
        .unwrap();
    density.finish(1000);
    assert_eq!(
        density.projected_samples,
        [25, 150, 400, 900, 900, 900, 900, 900]
    );
    assert_eq!(density.projected_weight, [0., 0., 1., 1., 1., 1., 1., 1.]);
    assert_eq!(density.values[..3], [Feature::Unsupported; 3]);
    assert_eq!(density.values[3], Feature::Projected(1000. / 900.));
    for value in &density.values[4..] {
        assert_eq!(*value, continuation.value);
    }
    assert!(g.accents.is_empty());
}

#[test]
fn candidate_accent_density_completes_issue_boundary_without_observing_future() {
    let (mut g, mut frames) = accent_material();
    g.history.back_mut().unwrap().raw.values[3] = Some(4.);
    for f in &mut frames {
        f.raw[3] = Feature::Projected(0.);
    }
    let result = g
        .projected_accent_density(
            [0, 1500],
            1000,
            1000,
            ([0.; 2], [1.; 2]),
            None,
            None,
            None,
            frames.iter().copied(),
        )
        .unwrap();
    assert_eq!(result.projected_accents, 1);
    assert_eq!(result.latest_projected_interval, Some([900, 1000]));
    assert_eq!(result.observed_weight, 0.);
    assert_eq!(result.value, Feature::Projected(1000. / 1400.));
    assert!(g.accents.is_empty());
    frames[0].raw[5] = Feature::Unsupported;
    let seam = g
        .projected_accent_density(
            [0, 1500],
            1000,
            1000,
            ([0.; 2], [1.; 2]),
            None,
            None,
            None,
            frames.iter().copied(),
        )
        .unwrap();
    assert_eq!(seam.projected_accents, 0);
    assert_eq!(seam.projected_samples, 200);
    assert_eq!(seam.value, Feature::Unsupported);
}

#[test]
fn candidate_accent_density_masks_missing_hops_and_rejects_noncausal_components() {
    let (g, frames) = accent_material();
    let mut missing = frames.clone();
    missing[4].energy = Feature::Unsupported;
    let result = g
        .projected_accent_density(
            [0, 2000],
            1000,
            1000,
            ([0.; 2], [1.; 2]),
            None,
            None,
            None,
            missing.into_iter(),
        )
        .unwrap();
    assert_eq!(result.projected_samples, 500);
    assert_eq!(result.projected_accents, 0);
    assert_eq!(result.value, Feature::Unsupported);
    for fault in 0..4 {
        let mut bad = frames.clone();
        match fault {
            0 => bad[0].raw[3] = Feature::Observed(0.),
            1 => bad[0].energy = Feature::Projected(f64::NAN),
            2 => bad[0].available = 2001,
            _ => bad.swap(0, 1),
        }
        assert!(
            g.projected_accent_density(
                [0, 2000],
                1000,
                1000,
                ([0.; 2], [1.; 2]),
                None,
                None,
                None,
                bad.into_iter()
            )
            .is_err()
        );
    }
}

#[test]
fn residual_uses_common_scaled_coordinates_and_distinguishes_known_zero() {
    let mut cue = group().cue.unwrap();
    assert_eq!(
        cue.normalized_residual(future().raw.map(Feature::value)),
        Some(2.5)
    );
    assert_eq!(cue.normalized_residual(cue.values), Some(0.));
    assert_eq!(cue.normalized_residual([None; 10]), None);
    cue.scales[0] = 0.;
    assert_eq!(
        cue.normalized_residual(future().raw.map(Feature::value)),
        None
    );
    cue.scales[0] = 1.;
    cue.values[0] = Some(f64::INFINITY);
    assert_eq!(
        cue.normalized_residual(future().raw.map(Feature::value)),
        None
    );
}

#[test]
fn projected_window_keeps_original_prefix_reference_and_physical_overlap() {
    let group = group();
    let before = serde_json::to_string(&group.snapshot).unwrap();
    let at_issue = group
        .projected_residual(0, 100, 100, std::iter::once(future()))
        .unwrap();
    assert_eq!(at_issue.value, Feature::Observed(2.));
    assert_eq!(
        (at_issue.observed_samples, at_issue.projected_samples),
        (100, 0)
    );
    assert!(at_issue.future_reference.is_none());
    let mixed = group
        .projected_residual(50, 175, 100, std::iter::once(future()))
        .unwrap();
    assert_eq!(mixed.value, Feature::Projected(2.3));
    assert_eq!(mixed.observed_mean, Some(2.));
    assert_eq!(mixed.projected_mean, Some(2.5));
    assert_eq!((mixed.observed_samples, mixed.projected_samples), (50, 75));
    assert_eq!(mixed.future_reference.unwrap().query_id, 9);
    let clipped = group
        .projected_residual(50, 300, 100, std::iter::once(future()))
        .unwrap();
    assert!((clipped.value.value().unwrap() - 7. / 3.).abs() < 1e-14);
    assert_eq!(
        (clipped.observed_samples, clipped.projected_samples),
        (50, 100)
    );
    let expired = group
        .projected_residual(200, 300, 100, std::iter::once(future()))
        .unwrap();
    assert_eq!(expired.value, Feature::Unsupported);
    assert!(expired.future_reference.is_none());
    assert_eq!(before, serde_json::to_string(&group.snapshot).unwrap());
    assert_eq!(group.history[0].residual, Some(2.));
}

#[test]
fn unavailable_or_other_group_reference_cannot_supply_future_evidence() {
    for fault in 0..6 {
        let mut group = group();
        let cue = group.cue.as_mut().unwrap();
        match fault {
            0 => cue.group.generation += 1,
            1 => cue.issued_at = 101,
            2 => cue.available = 51,
            3 => cue.source_end = 51,
            4 => cue.expected_end = cue.expected_start,
            _ => cue.source_start = 51,
        }
        let output = group
            .projected_residual(0, 200, 100, std::iter::once(future()))
            .unwrap();
        assert_eq!(output.value, Feature::Observed(2.), "fault {fault}");
        assert_eq!(output.projected_samples, 0);
        assert!(output.future_reference.is_none());
    }
    let mut group = group();
    group.history[0].raw.available_end = 101;
    assert!(
        group
            .projected_residual(0, 200, 100, std::iter::once(future()))
            .is_err()
    );
    let mut bad = future();
    bad.raw[0] = Feature::Observed(2.);
    assert!(
        self::group()
            .projected_residual(100, 200, 100, std::iter::once(bad))
            .is_err()
    );
    let mut bad = future();
    bad.start = 99;
    assert!(
        self::group()
            .projected_residual(100, 200, 100, std::iter::once(bad))
            .is_err()
    );
}

#[test]
fn projected_zero_is_a_supported_ordinal_input_but_unknown_is_not() {
    let group = group();
    let mut zero = future();
    zero.raw[0] = Feature::Projected(0.);
    zero.raw[1] = Feature::Projected(0.);
    let projected = group
        .projected_residual(100, 200, 100, std::iter::once(zero))
        .unwrap();
    assert_eq!(projected.value, Feature::Projected(0.));
    assert_eq!(projected.projected_samples, 100);
    let mut inputs = [None; 14];
    let cfg = crate::temporal_cognition::phrase::tests::config().closure;
    assert!(ordinal(&inputs, cfg).is_none());
    inputs[0] = projected.value.value();
    let probabilities = ordinal(&inputs, cfg).unwrap();
    let sigmoid = |v: f64| 1. / (1. + (-v).exp());
    let mut previous = 0.;
    for (index, cut) in cfg.cutpoints.into_iter().enumerate() {
        let cdf = sigmoid(cut);
        assert!((probabilities[index] - (cdf - previous)).abs() < 1e-14);
        previous = cdf;
    }
    assert!((probabilities[4] - (1. - previous)).abs() < 1e-14);
}
