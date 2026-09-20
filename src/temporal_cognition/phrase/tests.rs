use super::*;

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

#[test]
fn equal_duration_continuations_preserve_distinct_paths_and_missing_support() {
    use crate::life::action_candidates::Class;
    use crate::temporal_cognition::action_profiles;
    let profiles = action_profiles::tests::distinct_release_model();
    let mut phrase = Phrase::new(1, 0, 0, 48000, 512, config()).unwrap();
    let mut gesture = gesture::Gesture::new(
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
    let mut acoustic = input(1, 0.5);
    acoustic.assignment.end_sample = 512;
    acoustic.correlation_window[1] = 512;
    acoustic.energy = Some([1., 0., 0., 0., 0., 0., 0., 0.]);
    let raw = &mut acoustic.features[0].as_mut().unwrap().raw;
    raw.end = 512;
    raw.known_samples = 512;
    raw.source_end = 512;
    raw.available_end = 512;
    gesture.advance(&acoustic, &ridges(), 512).unwrap();
    phrase
        .advance(&acoustic, &gesture, None, None, 512)
        .unwrap();
    let group = phrase.groups[0].as_mut().unwrap();
    let base = *group
        .paths
        .iter()
        .find(|p| p.state.foreground.is_some())
        .unwrap();
    group.paths.clear();
    for index in 0..PATHS {
        let mut path = base;
        path.mass = 0.95 * (index + 1) as f64 / 120.;
        path.state.foreground = (index % 4 != 3).then(|| Foreground {
            id: 100 + index as u64,
            credit: 200 + index as u64,
            start: (index % 4) as u64 * 128,
            heard_end: 512,
        });
        group.paths.push(path);
    }
    group.unknown = 0.05;
    let original = group.paths.clone();
    let handle = group.handle;
    for slope in [0.8, 1e100] {
        phrase.config.hazard[1] = slope;
        for observed_articulation in [false, true] {
            for class in [Class::Continue, Class::Release] {
                for end in [512, 1536, 1537] {
                    let cell = phrase
                        .project_action_window(
                            &profiles,
                            0,
                            class,
                            0,
                            end,
                            handle,
                            1.,
                            observed_articulation.then_some(&gesture),
                            None,
                            None,
                            None,
                            None,
                        )
                        .unwrap();
                    assert_eq!(cell.continuation_paths.iter().flatten().count(), PATHS);
                    let mut values = [None; 12];
                    for (out, value) in values[4..].iter_mut().zip(cell.phrase_features) {
                        *out = value.value();
                    }
                    for (index, path) in original.iter().enumerate() {
                        let survival = path.state.foreground.map_or(Some(1.), |foreground| {
                            cell.articulation.and_then(|a| {
                                let elapsed = (end - foreground.start) as f64 / 48000.;
                                law::<true>(
                                    values,
                                    [
                                        a.states[0],
                                        a.states[1],
                                        a.states[2],
                                        a.states[3],
                                        a.unknown,
                                    ],
                                    elapsed,
                                    elapsed + 1.,
                                    false,
                                    phrase.config,
                                )
                                .1
                            })
                        });
                        let mut inputs = [None; 14];
                        inputs[0] = cell.residual.value.value();
                        inputs[10] = cell.continuation_context[0].value();
                        inputs[11] = survival;
                        inputs[12] = cell.continuation_context[1].value();
                        inputs[13] = cell.continuation_context[2].value();
                        let expected = ConditionalOrdinal {
                            path_index: index,
                            issue_mass: path.mass,
                            foreground_start: path.state.foreground.map(|f| f.start),
                            survival: survival.map_or(Feature::Unsupported, |v| {
                                if end == 512 {
                                    Feature::Observed(v)
                                } else {
                                    Feature::Projected(v)
                                }
                            }),
                            categories: ordinal(&inputs, phrase.config.continuation),
                        };
                        assert_eq!(
                            serde_json::to_vec(&cell.continuation_paths[index].unwrap()).unwrap(),
                            serde_json::to_vec(&expected).unwrap(),
                        );
                    }
                }
            }
        }
    }
    for (actual, expected) in phrase.groups[0]
        .as_ref()
        .unwrap()
        .paths
        .iter()
        .zip(original)
    {
        assert_eq!(actual.mass.to_bits(), expected.mass.to_bits());
        assert_eq!(
            actual.state.foreground.map(|f| (f.id, f.credit, f.start)),
            expected.state.foreground.map(|f| (f.id, f.credit, f.start))
        );
    }
}

#[test]
fn conditional_action_windows_are_read_only_and_retire_with_group_identity() {
    use crate::life::action_candidates::Class;
    use crate::temporal_cognition::{action_profiles, body_model};
    let profiles = action_profiles::tests::model();
    let mut phrase = Phrase::new(1, 0, 0, 48000, 512, config()).unwrap();
    let mut gesture = gesture::Gesture::new(
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
    let mut a = input(1, 0.5);
    a.assignment.end_sample = 512;
    a.correlation_window[1] = 512;
    a.energy = Some([1., 0., 0., 0., 0., 0., 0., 0.]);
    let raw = &mut a.features[0].as_mut().unwrap().raw;
    raw.end = 512;
    raw.known_samples = 512;
    raw.source_end = 512;
    raw.available_end = 512;
    raw.values[2] = Some(0.);
    raw.values[6] = Some(0.5);
    gesture.advance(&a, &ridges(), 512).unwrap();
    phrase.advance(&a, &gesture, None, None, 512).unwrap();
    let group = a.group_handles[0].unwrap();
    let before = serde_json::to_string(&phrase.snapshot()).unwrap();
    let at_action = phrase
        .project_action_window(
            &profiles,
            0,
            Class::OnsetNow,
            0,
            512,
            group,
            1.,
            Some(&gesture),
            None,
            None,
            None,
            None,
        )
        .unwrap();
    assert_eq!(at_action.features.values[0], Feature::Observed(0.));
    let actual = phrase.groups[0].as_ref().unwrap();
    let mut marginal = [0.; 5];
    let mut supported = 0.;
    for path in at_action.continuation_paths.iter().flatten() {
        assert_eq!(path.issue_mass, actual.paths[path.path_index].mass);
        assert!(matches!(path.survival, Feature::Observed(_)));
        if let Some(categories) = path.categories {
            for (total, value) in marginal.iter_mut().zip(categories) {
                *total += path.issue_mass * value;
            }
            supported += path.issue_mass;
        }
    }
    assert!(supported > 0.);
    for (projected, observed) in marginal
        .map(|v| v / supported)
        .into_iter()
        .zip(actual.snapshot.continuation.unwrap())
    {
        assert!((projected - observed).abs() < 1e-14);
    }
    assert_eq!(at_action.issue_phrase_unknown, actual.unknown);
    let issue_heads = phrase.issue_heads();
    let mixed = issue_heads.project(&at_action).unwrap();
    for (head, observed) in [mixed.closure, mixed.continuation]
        .into_iter()
        .zip([phrase.snapshot.closure, phrase.snapshot.continuation])
    {
        assert_eq!(head.observed_coverage, observed.observed_coverage);
        assert!((head.supported_mass - observed.supported_mass).abs() < 1e-14);
        for (a, b) in head.categories.into_iter().zip(observed.categories) {
            assert!((a - b).abs() < 1e-14);
        }
    }
    let after = phrase
        .project_action_window(
            &profiles,
            0,
            Class::OnsetNow,
            0,
            1536,
            group,
            1.,
            Some(&gesture),
            None,
            None,
            None,
            None,
        )
        .unwrap();
    assert_eq!((after.action_at, after.evaluation_at), (512, 1536));
    assert_eq!(after.features.values[0], Feature::Projected(1. / 3.));
    assert!(matches!(after.features.values[3], Feature::Projected(v) if (v - 0.7).abs() < 1e-14));
    assert_eq!(after.features.observed_fraction[0], 1. / 3.);
    assert_eq!(after.continuation_context[0], Feature::Unsupported);
    assert_eq!(after.continuation_context[1], Feature::Unsupported);
    assert_eq!(after.issue_phrase_unknown, at_action.issue_phrase_unknown);
    let future_heads = issue_heads.project(&after).unwrap();
    assert_eq!(future_heads.continuation.observed_coverage, 1.);
    assert_eq!(
        future_heads.continuation.supported_mass,
        mixed.continuation.supported_mass
    );
    for (projected, original) in after
        .continuation_paths
        .iter()
        .flatten()
        .zip(at_action.continuation_paths.iter().flatten())
    {
        assert_eq!(projected.issue_mass, original.issue_mass);
        assert!(matches!(projected.survival, Feature::Projected(_)));
    }
    assert!(
        phrase
            .project_action_window(
                &profiles,
                0,
                Class::OnsetNow,
                0,
                511,
                group,
                1.,
                Some(&gesture),
                None,
                None,
                None,
                None,
            )
            .is_none()
    );
    assert!(
        phrase
            .project_action_window(
                &profiles,
                0,
                Class::OnsetNow,
                0,
                192513,
                group,
                1.,
                Some(&gesture),
                None,
                None,
                None,
                None,
            )
            .is_none()
    );
    assert_eq!(before, serde_json::to_string(&phrase.snapshot()).unwrap());
    let raw = &actual.history.back().unwrap().raw;
    let previous = window::Frame {
        start: raw.start,
        end: raw.end,
        source_end: raw.source_end,
        available: raw.available_end,
        raw: raw
            .values
            .map(|v| v.map_or(Feature::Unsupported, Feature::Observed)),
        energy: Feature::Observed(1.),
    };
    for duration in [
        0, 1, 511, 512, 513, 11999, 12000, 12001, 95999, 96000, 96001, 191999, 192000,
    ] {
        let end = 512 + duration;
        let cell = phrase
            .project_action_window(
                &profiles,
                0,
                Class::OnsetNow,
                0,
                end,
                group,
                1.,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        for (window_samples, short) in [(96000, false), (12000, true)] {
            let full = window::summarize(
                actual.born.max(end.saturating_sub(window_samples)),
                end,
                512,
                48000,
                1.,
                std::iter::once(previous).chain(
                    profiles
                        .frames(0, Class::OnsetNow, 0, 512, Some(previous), Some(1.))
                        .unwrap(),
                ),
            )
            .unwrap();
            if short {
                assert_eq!(
                    cell.phrase_features[..4],
                    full.values[..4],
                    "short duration {duration}"
                );
            } else {
                assert_eq!(
                    serde_json::to_vec(&cell.features).unwrap(),
                    serde_json::to_vec(&full).unwrap(),
                    "long duration {duration}"
                );
            }
        }
    }
    let prefix_profiles = action_profiles::tests::shared_prefix_model();
    let mut shared = body_model::Shared {
        model_version: profiles.body_model_version,
        bus: 1,
        epoch: 0,
        end_sample: 512,
        descriptors: [None; 7],
        assignments: [None; 8],
    };
    shared.assignments[0] = Some(body_model::Assignment {
        key: (0, group.generation),
        distance: 0.,
        common_coordinates: 6,
    });
    let mut table = action_profiles::Table::new();
    let first = table
        .refresh(&profiles, &phrase, &shared, &gesture, None)
        .unwrap();
    assert_eq!(first.cells, 129);
    assert!(first.projected_coordinates > 0);
    assert_eq!(first.articulation_supported_cells, first.cells);
    for cell in first.latest.iter().flatten() {
        assert!(cell.raw_unreweighted_heads.is_some());
        let states = cell.articulation.unwrap();
        assert_eq!(states.issued_at, cell.issued_at);
        assert_eq!(states.evaluation_at, cell.evaluation_at);
        assert!((states.states.iter().sum::<f64>() + states.unknown - 1.).abs() < 1e-12);
        assert!(states.unknown >= states.original_unknown);
        assert!(
            cell.phrase_features[6..]
                .iter()
                .all(|v| *v == Feature::Unsupported)
        );
    }
    assert_eq!(
        first.issue_observed_coverage,
        phrase.snapshot().closure.observed_coverage
    );
    assert_eq!(
        serde_json::to_string(&first).unwrap(),
        serde_json::to_string(
            &table
                .refresh(&profiles, &phrase, &shared, &gesture, None)
                .unwrap()
        )
        .unwrap()
    );
    let mut prefix_table = action_profiles::Table::new();
    let prefix_snapshot = prefix_table
        .refresh(&prefix_profiles, &phrase, &shared, &gesture, None)
        .unwrap();
    action_profiles::tests::assert_table_matches_direct(
        &prefix_table,
        &prefix_profiles,
        &phrase,
        &gesture,
    );
    assert_eq!(
        prefix_snapshot.articulation_supported_cells,
        prefix_snapshot.cells
    );
    assert!(prefix_table.resources().articulation_frames < 2_000);
    println!(
        "I10_CROSS_TRAJECTORY_PREFIX advanced={} reused={}",
        prefix_table.resources().articulation_frames,
        prefix_table.resources().reused_articulation_frames
    );
    shared.assignments[0] = None;
    assert!(
        table
            .refresh(&profiles, &phrase, &shared, &gesture, None)
            .is_none()
    );
    assert_eq!(table.resources().rebuilds, 1);
    assert_eq!(table.resources().invalidations, 1);
    assert_eq!(table.resources().deferred_calls, 1);
    assert_eq!(before, serde_json::to_string(&phrase.snapshot()).unwrap());
    let sample = phrase.groups[0]
        .as_mut()
        .unwrap()
        .history
        .back_mut()
        .unwrap();
    sample.grouping = Some(0.5);
    sample.alpha = 0.3;
    let g = phrase.groups[0].as_ref().unwrap();
    assert_eq!(g.grouping_window(0, 512, 512), (Some(0.5), 1.));
    assert_eq!(g.grouping_window(0, 568, 512), (Some(0.5), 512. / 568.));
    assert_eq!(g.grouping_window(0, 569, 512), (None, 512. / 569.));
    assert_eq!(g.grouping_window(0, 512, 511), (None, 0.));
    let future = phrase
        .project_action_window(
            &profiles,
            0,
            Class::Continue,
            0,
            569,
            group,
            1.,
            Some(&gesture),
            None,
            None,
            None,
            None,
        )
        .unwrap();
    assert_eq!(future.phrase_features[4], Feature::Unsupported);
    assert_eq!(future.grouping_observed_fraction, 512. / 569.);
    phrase.groups[0] = None;
    assert!(
        phrase
            .project_action_window(
                &profiles,
                0,
                Class::Continue,
                0,
                1536,
                group,
                1.,
                Some(&gesture),
                None,
                None,
                None,
                None,
            )
            .is_none()
    );
}

#[test]
fn group_body_window_uses_physical_history_not_foreground_hypotheses() {
    let mut p = Phrase::new(1, 0, 0, 1000, 100, config()).unwrap();
    let mut articulation = gesture_model();
    for step in 1..=25 {
        let mut a = input(step, 0.5);
        let energy = if step % 2 == 0 { 4. } else { 1. };
        a.energy = Some([energy, 0., 0., 0., 0., 0., 0., 0.]);
        a.features[0].as_mut().unwrap().raw.values[0] = Some(step as f64);
        a.features[0].as_mut().unwrap().raw.values[3] = Some(step as f64 / 100.);
        a.features[0].as_mut().unwrap().raw.values[5] = Some(step as f64 / 200.);
        articulation.advance(&a, &ridges(), step * 100).unwrap();
        p.advance(&a, &articulation, None, None, step * 100)
            .unwrap();
        if step == 1 {
            let young = p.body_descriptors()[0].unwrap();
            assert_eq!((young.start, young.end), (0, 100));
            assert!(young.raw_values.iter().all(Option::is_some));
            assert_eq!(young.coverage, [1.; 6]);
        }
    }
    let measured = p.body_descriptors()[0].unwrap();
    assert_eq!((measured.start, measured.end), (500, 2500));
    let energy = |step: u64| if step % 2 == 0 { 4. } else { 1. };
    let mass: f64 = (6..=25).map(energy).sum();
    let centroid = (6..=25).map(|i| i as f64 * energy(i)).sum::<f64>() / mass;
    let variance = (6..=25)
        .map(|i| ((i as f64 - centroid).powi(2) + 0.25) * energy(i))
        .sum::<f64>()
        / mass;
    let expected = [
        centroid,
        variance.sqrt(),
        (mass / 20.).sqrt().log2(),
        0.155,
        0.0775,
        0.,
    ];
    for (actual, expected) in measured.raw_values.into_iter().zip(expected) {
        assert!((actual.unwrap() - expected).abs() < 1e-12);
    }
    for path in &mut p.groups[0].as_mut().unwrap().paths {
        path.state.foreground = None;
    }
    assert_eq!(p.body_descriptors()[0], Some(measured));
    let group = p.groups[0].as_mut().unwrap();
    for _ in 0..3 {
        group.history.pop_front();
    }
    let missing = p.body_descriptors()[0].unwrap();
    assert_eq!(missing.coverage, [0.85; 6]);
    assert_eq!(missing.raw_values, [None; 6]);
    assert_eq!(missing.end, measured.end);
}

#[test]
fn shared_prototype_cache_masks_loss_before_rebuild_without_rebinding() {
    use crate::config::{TemporalBodyConfig, TemporalBodyMedoid, TemporalBodyPrototypesConfig};
    use crate::temporal_cognition::body_model::{Prototypes, Shared};
    let scales = TemporalBodyConfig {
        means: [0.; 6],
        deviations: [1.; 6],
        accent_means: [0.; 2],
        accent_deviations: [1.; 2],
    };
    let model = Prototypes::new(
        &TemporalBodyPrototypesConfig {
            model_version: "ab".repeat(32),
            sample_rate: 2000,
            nfft: 400,
            hop_size: 100,
            means: scales.means,
            deviations: scales.deviations,
            accent_means: scales.accent_means,
            accent_deviations: scales.accent_deviations,
            medoids: vec![TemporalBodyMedoid {
                record_id: "observed-template".into(),
                raw_values: [8., 0.5, -3., 0., 0., 0.],
                mask: 63,
            }],
        },
        scales,
    );
    let mut p = Phrase::new(1, 0, 0, 2000, 100, config()).unwrap();
    let mut g = gesture_model();
    let mut cached = None;
    for step in 1..=3 {
        let mut a = input(step, 0.5);
        a.energy = Some([0.015625, 0., 0., 0., 0., 0., 0., 0.]);
        g.advance(&a, &ridges(), step * 100).unwrap();
        p.advance(&a, &g, None, None, step * 100).unwrap();
        let before = serde_json::to_string(&p.snapshot()).unwrap();
        Shared::refresh(&mut cached, &model, scales, &p, &a, (1, 0));
        assert_eq!(before, serde_json::to_string(&p.snapshot()).unwrap());
        let context = cached.unwrap();
        assert_eq!(context.end_sample, if step == 3 { 300 } else { 100 });
        assert_eq!(context.assignments[0].unwrap().key, (0, 2));
        assert_eq!(context.assignments[0].unwrap().common_coordinates, 6);
        assert!(context.assignments[1..].iter().all(Option::is_none));
        if step == 2 {
            let mut lost = a;
            lost.eligible[0] = false;
            Shared::refresh(&mut cached, &model, scales, &p, &lost, (1, 0));
            assert_eq!(cached.unwrap().end_sample, 100);
            assert!(cached.unwrap().descriptors.iter().all(Option::is_none));
            assert!(cached.unwrap().assignments.iter().all(Option::is_none));
            Shared::refresh(&mut cached, &model, scales, &p, &a, (1, 0));
            assert!(cached.unwrap().assignments.iter().all(Option::is_none));
        }
        if step == 3 {
            let handle = a.retained_groups[0].unwrap();
            a.retained_groups[0] = Some(Handle {
                generation: handle.generation + 1,
                ..handle
            });
            Shared::refresh(&mut cached, &model, scales, &p, &a, (1, 0));
            assert!(cached.unwrap().assignments.iter().all(Option::is_none));
            a.retained_groups[0] = Some(handle);
            Shared::refresh(&mut cached, &model, scales, &p, &a, (0, 0));
            assert!(cached.unwrap().descriptors.iter().all(Option::is_none));
            Shared::refresh(&mut cached, &model, scales, &p, &a, (1, 9));
            assert!(cached.unwrap().descriptors.iter().all(Option::is_none));
        }
    }
}

#[test]
fn gap_energy_control_uses_physical_support_and_original_endpoint_times() {
    let mut p = Phrase::new(1, 0, 0, 1000, 100, config()).unwrap();
    let mut articulation = gesture_model();
    for step in 1..=20 {
        let rms: f64 = if step <= 10 { 0.005 } else { 0.1 };
        let mut a = input(step, 0.);
        a.features[0].as_mut().unwrap().raw.values[2] = Some(rms.log2());
        let mut energy = [0.; 8];
        energy[0] = rms * rms;
        a.energy = Some(energy);
        articulation.advance(&a, &ridges(), step * 100).unwrap();
        p.advance(&a, &articulation, None, None, step * 100)
            .unwrap();
    }
    let g = p.groups[0].as_mut().unwrap();
    let r = g.recent_energy(0, 2000, 1000, 1.);
    assert_eq!(r.coverage, [1.; 3]);
    assert_eq!(r.values[0], Some(0.5));
    assert!((r.values[1].unwrap() - 0.0525).abs() < 1e-14);
    assert!((r.values[2].unwrap() - 20_f64.log2() / 1.9).abs() < 1e-14);
    let clipped = g.recent_energy(50, 2000, 1000, 1.);
    assert!((clipped.values[0].unwrap() - 950. / 1950.).abs() < 1e-14);
    assert!((clipped.values[2].unwrap() - 20_f64.log2() / 1.875).abs() < 1e-14);
    // A missing endpoint is not replaced by a later, more convenient observation.
    g.history[0].raw.values[2] = None;
    assert!(g.recent_energy(0, 2000, 1000, 1.).values[2].is_none());
    for i in 1..=2 {
        g.history[i].raw.values[2] = None;
    }
    let missing = g.recent_energy(0, 2000, 1000, 1.);
    assert!(missing.values[0].is_none());
    assert_eq!(missing.coverage[0], 0.85);
    assert!(missing.values[1].is_some());
    for s in &mut g.history {
        s.energy = Some(0.);
        s.raw.values[2] = Some(1e-6_f64.log2());
    }
    assert_eq!(
        g.recent_energy(0, 2000, 1000, 1.).values,
        [Some(1.), Some(0.), Some(0.)]
    );
}

#[test]
fn narrow_gap_hazard_cannot_pass_as_zero_from_missed_quadrature_nodes() {
    assert!(integrate(20., -40., 0., 1200.).is_none());
}

#[test]
fn retained_children_identify_their_exact_prior_path_and_original_closed_support() {
    let mut cfg = config();
    cfg.hazard = [0.; 26];
    cfg.hazard[0] = 2.;
    let mut p = Phrase::new(1, 0, 0, 1000, 100, cfg).unwrap();
    let mut g = gesture_model();
    let mut prior: Option<GroupSnapshot> = None;
    let mut exits = [0; 4];
    let mut inherited_nonzero_parent = false;
    for step in 1..=30 {
        let mut a = input(step, 0.);
        if step >= 15 {
            a.features[0].as_mut().unwrap().raw.values[2] = Some(-30.);
        }
        g.advance(&a, &ridges(), step * 100).unwrap();
        p.advance(&a, &g, None, None, step * 100).unwrap();
        let now = p.snapshot().groups[0].unwrap();
        assert_eq!(now.previous_end_sample, (step - 1) * 100);
        if let Some(prior) = prior {
            for child in now.candidates.iter().flatten() {
                let parent = prior.candidates[child.parent_index].unwrap();
                inherited_nonzero_parent |= child.parent_index > 0;
                if let Some((kind, cut)) = child.event.filter(|(_, cut)| *cut == step * 100) {
                    exits[kind as usize] += 1;
                    if kind == Exit::Reinterpret {
                        assert_eq!(child.completed_foreground, None);
                        assert_eq!(
                            child.foreground.unwrap().credit,
                            parent.foreground.unwrap().credit
                        );
                        assert_eq!(
                            child.foreground.unwrap().start,
                            parent.foreground.unwrap().start
                        );
                    } else {
                        assert_eq!(child.completed_foreground, parent.foreground);
                        assert!(
                            child.completed_foreground.unwrap().heard_end
                                <= now.previous_end_sample
                        );
                    }
                    assert_eq!(cut, step * 100);
                } else {
                    assert_eq!(child.completed_foreground, None);
                }
            }
        }
        prior = Some(now);
    }
    assert!(inherited_nonzero_parent);
    assert!(exits.iter().all(|n| *n > 0), "{exits:?}");
}

#[test]
fn missing_interval_receipts_never_complete_an_observed_span() {
    let mut p = Phrase::new(1, 0, 0, 1000, 100, config()).unwrap();
    let mut g = gesture_model();
    let a = input(1, 0.);
    g.advance(&a, &ridges(), 100).unwrap();
    p.advance(&a, &g, None, None, 100).unwrap();
    let prior = p.snapshot().groups[0].unwrap();
    p.finish(1000).unwrap();
    let after = p.snapshot().groups[0].unwrap();
    assert_eq!(after.previous_end_sample, 100);
    for child in after.candidates.iter().flatten() {
        assert_eq!(child.completed_foreground, None);
        assert_eq!(
            child.foreground,
            prior.candidates[child.parent_index].unwrap().foreground
        );
    }
}

#[test]
fn receipt_parent_composes_across_an_implicit_gap_before_the_next_observation() {
    let mut p = Phrase::new(1, 0, 0, 1000, 100, config()).unwrap();
    let mut g = gesture_model();
    for step in 1..=10 {
        let a = input(step, 0.);
        g.advance(&a, &ridges(), step * 100).unwrap();
        p.advance(&a, &g, None, None, step * 100).unwrap();
    }
    let prior = p.snapshot().groups[0].unwrap();
    let a = input(25, 0.);
    g.advance(&a, &ridges(), 2500).unwrap();
    p.advance(&a, &g, None, None, 2500).unwrap();
    let now = p.snapshot().groups[0].unwrap();
    assert_eq!(now.previous_end_sample, 1000);
    for child in now.candidates.iter().flatten() {
        let parent = prior.candidates[child.parent_index].unwrap();
        if let Some(closed) = child.completed_foreground {
            assert_eq!(Some(closed), parent.foreground);
        } else if let (Some(c), Some(p)) = (child.foreground, parent.foreground) {
            assert_eq!(c.credit, p.credit);
            assert_eq!(c.start, p.start);
        }
    }
}

#[test]
fn original_ending_moments_match_python_pooled_spectra_and_clipped_support() {
    let data: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/temporal_cognition/sections.json"
    ))
    .unwrap();
    let mut p = Phrase::new(1, 0, 0, 1000, 100, config()).unwrap();
    let mut articulation = gesture_model();
    let a = input(1, 0.);
    articulation.advance(&a, &ridges(), 100).unwrap();
    p.advance(&a, &articulation, None, None, 100).unwrap();
    let g = p.groups[0].as_mut().unwrap();
    for case in data["endings"].as_array().unwrap() {
        g.history.clear();
        g.accents.clear();
        g.born = (case["generation_start"].as_f64().unwrap() * 1000.).round() as u64;
        g.evicted_accent = case["evicted"].as_f64().map(|v| (v * 1000.).round() as u64);
        for h in case["hops"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|h| h["observed"] == true)
        {
            let start = (h["start"].as_f64().unwrap() * 1000.).round() as u64;
            let end = (h["end"].as_f64().unwrap() * 1000.).round() as u64;
            let energy = h["energy"].as_f64();
            let mut values = [None; 10];
            if let Some(spectrum) = h["spectrum"].as_array() {
                let left = spectrum[0].as_f64().unwrap();
                let right = spectrum[1].as_f64().unwrap();
                let mass = left + right;
                if mass > 0. {
                    let center = (left + 3. * right) / mass;
                    values[0] = Some(center);
                    values[1] = Some(
                        ((left * (1. - center).powi(2) + right * (3. - center).powi(2)) / mass)
                            .sqrt(),
                    );
                }
            }
            values[3] = h["rise"].as_f64();
            values[5] = h["flux"].as_f64();
            g.history.push_back(Sample {
                raw: RawDescriptor {
                    group: g.handle,
                    start,
                    end,
                    known_samples: end - start,
                    values,
                    source_start: start,
                    source_end: end,
                    available_end: end,
                },
                energy,
                spectral_shape_supported: !h["spectrum"].is_null(),
                alpha: 1.,
                grouping: None,
                residual: None,
                residual_interval: (0, 0),
            });
        }
        for a in case["accents"].as_array().unwrap() {
            let end = (a["time"].as_f64().unwrap() * 1000.).round() as u64;
            g.accents.push_back(super::super::features::Accent {
                group: g.handle,
                event_start: end - 250,
                event_end: end,
                raw_intervals: [
                    (end - 750, end - 500),
                    (end - 500, end - 250),
                    (end - 250, end),
                    (end, end + 250),
                ],
                source_start: end - 750,
                source_end: end + 250,
                available_end: end + 250,
                weight: a["weight"].as_f64().unwrap(),
                observed_prefix: end,
            });
        }
        let span = Foreground {
            id: 1,
            credit: 1,
            start: (case["span_start"].as_f64().unwrap() * 1000.).round() as u64,
            heard_end: (case["cut"].as_f64().unwrap() * 1000.).round() as u64,
        };
        let actual = g.ending(span, 1000);
        for i in 0..6 {
            match (actual.raw_values[i], case["expected"]["raw"][i].as_f64()) {
                (Some(a), Some(e)) => assert!((a - e).abs() < 1e-12, "coordinate {i}: {a} != {e}"),
                (None, None) => (),
                _ => panic!("ending validity differs at {i}: {case}"),
            }
            assert!(
                (actual.coverage[i] - case["expected"]["coverage"][i].as_f64().unwrap()).abs()
                    < 1e-12
            );
        }
    }
}

#[test]
fn completed_ending_keeps_original_cut_and_ignores_the_boundary_hops_new_audio() {
    let mut cfg = config();
    cfg.hazard = [0.; 26];
    cfg.hazard[0] = 1.;
    let mut p = Phrase::new(1, 0, 0, 1000, 100, cfg).unwrap();
    let mut g = gesture_model();
    let mut expected = Vec::new();
    let mut checked = 0;
    for step in 1..=12 {
        let mut a = input(step, 0.);
        a.energy = Some([0.01, 0., 0., 0., 0., 0., 0., 0.]);
        a.features[0].as_mut().unwrap().raw.values[0] = Some(step as f64);
        g.advance(&a, &ridges(), step * 100).unwrap();
        p.advance(&a, &g, None, None, step * 100).unwrap();
        for c in p.snapshot().groups[0].unwrap().candidates.iter().flatten() {
            if let Some(ending) = c.completed_ending {
                if step == 1 {
                    assert_eq!(ending.end, 100);
                    assert_eq!(ending.raw_values[0], Some(1.));
                    assert_eq!(ending.coverage, [1.; 6]);
                } else {
                    assert_eq!(
                        Some(&ending),
                        expected.get(c.parent_index).and_then(Option::as_ref)
                    );
                    assert!(ending.end < step * 100);
                }
                assert!(ending.available <= ending.end);
                checked += 1;
            }
        }
        expected = p.groups[0]
            .as_ref()
            .unwrap()
            .paths
            .iter()
            .map(|p| p.state.ending)
            .collect();
    }
    assert!(checked > 0);
}
pub(in crate::temporal_cognition) fn config() -> TemporalPhraseConfig {
    let ordinal = TemporalOrdinalConfig {
        means: [0.; 14],
        deviations: [1.; 14],
        coefficients: [0.; 29],
        cutpoints: [-1.5, -0.5, 0.5, 1.5],
        prior: [0.2; 5],
    };
    let mut cfg = TemporalPhraseConfig {
        means: [0.; 12],
        deviations: [1.; 12],
        hazard: [0.; 26],
        exits: [[0.; 26]; 4],
        closure: ordinal,
        continuation: ordinal,
    };
    cfg.hazard[0] = -3.;
    cfg.hazard[6] = 1.;
    cfg.hazard[11] = 1.;
    cfg.closure.coefficients[1] = -2.;
    cfg.continuation.coefficients[12] = 2.;
    cfg
}
#[test]
fn phrase_head_layout_and_missing_inputs_remain_separate_from_articulation_marginalization() {
    let cfg = config();
    let mut values = [None; 12];
    values[4] = Some(2.);
    values[9] = Some(0.7);
    let x = vector(values, 2, 3., cfg);
    assert_eq!(x.len(), 26);
    assert_eq!(&x[2..6], &[0., 0., 1., 0.]);
    assert_eq!(x[6], 2.);
    assert_eq!(x[11], 0.7);
    assert_eq!(x[1], 4_f64.ln());
    assert_eq!(&vector(values, 4, 3., cfg)[14..18], &[1.; 4]);
    let states = [0.2, 0.3, 0.1, 0.15, 0.25];
    let result = law::<true>(values, states, 2., 3., false, cfg).0;
    assert!((result.iter().sum::<f64>() - 1.).abs() < 1e-12);
    assert_eq!(result[4], 0.);
    assert!(result[5] > 0.);
}
#[test]
fn survival_only_preserves_state_mixture_and_unsupported_hazard() {
    let mut cfg = config();
    cfg.means = std::array::from_fn(|i| (i as f64 - 5.) * 0.03);
    cfg.deviations = std::array::from_fn(|i| if i == 7 { 1e-9 } else { 0.5 + i as f64 });
    cfg.hazard = std::array::from_fn(|i| (i as f64 - 13.) * 0.02);
    cfg.exits = std::array::from_fn(|i| std::array::from_fn(|j| ((i * 3 + j) as f64 - 12.) * 0.1));
    let mut mixtures = vec![[0.2, 0.3, 0.1, 0.15, 0.25], [0.; 5]];
    mixtures.extend((0..5).map(|state| std::array::from_fn(|i| f64::from(i == state))));
    for mask in [0_u16, 0xfff, 0x555, 0xaaa] {
        let values =
            std::array::from_fn(|i| (mask & (1 << i) == 0).then_some((i as f64 - 6.) * 0.1));
        for states in &mixtures {
            for (lo, hi) in [
                (0., 0.),
                (-0., 0.01),
                (0., 1.),
                (2., 3.),
                (1200., 1201.),
                (2., 1.),
                (-1., 0.),
                (0., f64::INFINITY),
            ] {
                for slope in [-2., 0., 0.7, 1000.] {
                    cfg.hazard[1] = slope;
                    let full = law::<true>(values, *states, lo, hi, false, cfg).1;
                    let survival = law::<false>(values, *states, lo, hi, false, cfg).1;
                    assert_eq!(full.map(f64::to_bits), survival.map(f64::to_bits));
                }
            }
        }
    }
    cfg.hazard = [0.; 26];
    cfg.exits = [[f64::MAX; 26]; 4];
    let states = [0.2, 0.3, 0.1, 0.15, 0.25];
    let full = law::<true>([None; 12], states, 0., 1., true, cfg);
    let survival = law::<false>([None; 12], states, 0., 1., false, cfg).1;
    assert!(full.0[5] > 0.);
    assert!(survival.is_some());
    assert_eq!(full.1.map(f64::to_bits), survival.map(f64::to_bits));
}

#[test]
fn boundary_and_closure_can_disagree_and_ordinal_backoff_is_not_survival() {
    let mut cfg = config();
    cfg.hazard = [0.; 26];
    cfg.hazard[0] = 4.;
    let boundary = law::<true>([None; 12], [1., 0., 0., 0., 0.], 1., 2., true, cfg).0;
    let mut values = [None; 14];
    values[0] = Some(5.);
    let closure = ordinal(&values, cfg.closure).unwrap();
    assert!(boundary[0] < 0.02);
    assert!(closure[0] > 0.99);
    cfg.hazard[0] = -8.;
    cfg.closure.coefficients[1] = 2.;
    let boundary = law::<true>([None; 12], [1., 0., 0., 0., 0.], 1., 2., true, cfg).0;
    let closure = ordinal(&values, cfg.closure).unwrap();
    assert!(boundary[0] > 0.999);
    assert!(closure[4] > 0.99);
    assert!(ordinal(&[None; 14], cfg.closure).is_none());
    let p = Phrase::new(0, 0, 0, 1000, 10, cfg).unwrap();
    assert_eq!(p.snapshot().closure.categories, cfg.closure.prior);
    assert_eq!(p.snapshot().closure.unknown, 1.);
}
#[test]
fn duration_quadrature_matches_closed_form_and_bounded_gap_integral() {
    let a = 2_f64.exp_m1().ln();
    assert!((integrate(a, 0., 0., 32.).unwrap() - 64.).abs() < 1e-10);
    for lo in [0., 0.1, 1., 20.] {
        let reference = (0..100000)
            .map(|i| {
                let z = 0.4 + 1.3 * (lo + (i as f64 + 0.5) / 100000.).ln_1p();
                z.max(0.) + (-z.abs()).exp().ln_1p()
            })
            .sum::<f64>()
            / 100000.;
        assert!((integrate(0.4, 1.3, lo, lo + 1.).unwrap() - reference).abs() < 1e-6);
    }
    assert!(integrate(0., 100., 0., 1e8).is_none());
}

use super::super::{features, group};
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
    out.assignment.rows[0] = Some(super::super::group::Row {
        trajectory: Handle {
            generation: 4,
            ..handle
        },
        weights: [1., 0., 0., 0., 0., 0., 0., 0.],
        matched_members: [None; 7],
    });
    out.features[0].as_mut().unwrap().raw.values[2] = Some(-3.);
    out.features[0].as_mut().unwrap().raw.values[3] = Some(0.);
    out.features[0].as_mut().unwrap().raw.values[4] = Some(0.);
    out.features[0].as_mut().unwrap().raw.values[5] = Some(0.);
    out.features[0].as_mut().unwrap().raw.values[6] = Some(1.);
    out
}

pub(in crate::temporal_cognition) fn gesture_model() -> gesture::Gesture {
    gesture::Gesture::new(
        1,
        0,
        1000,
        100,
        crate::config::TemporalGestureConfig {
            rms_reference: 0.1,
            means: [0.; 5],
            deviations: [1.; 5],
            coefficients: [[[0.; 11]; 4]; 4],
        },
    )
    .unwrap()
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

#[test]
fn identical_suffix_uses_issued_memory_predictions_and_eof_censors_without_closure() {
    let mut closure = Vec::new();
    for expected in [0., 4.] {
        let mut p = Phrase::new(1, 0, 0, 1000, 100, config()).unwrap();
        let mut g = gesture_model();
        let mut prior = None;
        for step in 1..=12 {
            let a = input(step, 0.);
            g.advance(&a, &ridges(), step * 100).unwrap();
            let memory = if step == 10 {
                let mut values = [None; 10];
                values[1] = Some(expected);
                let mut memory = recall::Snapshot {
                    latest: Some(recall::ResultSnapshot {
                        search_covered: true,
                        cutoff_tie: false,
                        pruned_candidates: 0,
                        pruned_ties: 0,
                        query_id: 1,
                        cue: None,
                        group: a.group_handles[0].unwrap(),
                        support_start_sample: 500,
                        support_end_sample: 900,
                        source_start_sample: 200,
                        supporting_audio_end: Some(900),
                        available_at: 900,
                        issued_at: 900,
                        completed_at: 900,
                        received_at: 1000,
                        deadline: 1200,
                        candidates: 1,
                        dp_cells: 0,
                        reconstruction_error: 0.,
                        best: None,
                        prediction: Some(recall::Prediction {
                            episode_id: 1,
                            query_id: 1,
                            group: a.group_handles[0].unwrap(),
                            issued_at: 900,
                            source_start: 0,
                            source_end: 900,
                            available: 900,
                            expected_start: 900,
                            expected_end: 1100,
                            values,
                            scales: [1.; 10],
                        }),
                    }),
                    ..Default::default()
                };
                memory.group_queries[0] = memory.latest;
                Some(memory)
            } else {
                None
            };
            p.advance(&a, &g, memory, None, step * 100).unwrap();
            if step == 9 {
                prior = p.snapshot().groups[0].unwrap().forecast;
            }
        }
        let before = p.snapshot();
        let group = before.groups[0].unwrap();
        assert!(group.prediction_comparisons >= 2);
        assert_eq!(group.closure_values[0], Some(expected * expected));
        assert!(before.closure.supported_mass > 0.);
        closure.push(before.closure.expected_rating);
        let f = group.forecast.unwrap();
        assert!((f.survival + f.exits.iter().sum::<f64>() + f.unknown - 1.).abs() < 1e-9);
        assert_eq!(prior.unwrap().issued_at, 900);
        p.finish(1200).unwrap();
        assert!(p.snapshot().censored);
        assert_eq!(p.snapshot().closure.categories, before.closure.categories);
        p.finish(2000).unwrap();
        assert_eq!(p.snapshot().closure.supported_mass, 0.);
        assert_eq!(p.snapshot().closure.categories, config().closure.prior);
        assert!(p.snapshot().groups[0].unwrap().forecast.is_none());
    }
    assert!(closure[0] > closure[1] + 0.1, "{closure:?}");
}

#[test]
fn gaps_mask_windows_without_turning_missing_samples_into_observed_time() {
    let mut p = Phrase::new(1, 0, 0, 1000, 100, config()).unwrap();
    let mut g = gesture_model();
    for step in 1..=10 {
        let a = input(step, 0.);
        g.advance(&a, &ridges(), step * 100).unwrap();
        p.advance(&a, &g, None, None, step * 100).unwrap();
    }
    let before = p.snapshot();
    assert_eq!(before.groups[0].unwrap().phrase_values[7], Some(1.));
    let a = input(15, 0.);
    g.advance(&a, &ridges(), 1500).unwrap();
    p.advance(&a, &g, None, None, 1500).unwrap();
    let after = p.snapshot();
    assert!(after.groups[0].unwrap().phrase_values[7].is_none());
    assert!(after.groups[0].unwrap().unknown > before.groups[0].unwrap().unknown);
    assert!(after.closure.observed_coverage < 0.9);
}

#[test]
fn all_exit_successors_preserve_clock_credit_and_unknown_gaps_do_not_label_exits() {
    for kind in EXITS {
        let mut p = Phrase::new(1, 0, 0, 1000, 100, config()).unwrap();
        let mut g = gesture_model();
        let a = input(1, 0.);
        g.advance(&a, &ridges(), 100).unwrap();
        p.advance(&a, &g, None, None, 100).unwrap();
        let group = p.groups[0].as_mut().unwrap();
        group.paths.truncate(1);
        group.paths[0].mass = 1.;
        group.unknown = 0.;
        let original = group.paths[0].state.foreground.unwrap();
        let mut cfg = config();
        cfg.hazard[0] = 100.;
        cfg.exits[kind as usize][0] = 20.;
        group.step(200, true, true, 1000, cfg, &mut p.next_span);
        let chosen = group
            .paths
            .iter()
            .find(|p| p.state.event == Some((kind, 200)))
            .unwrap();
        match kind {
            Exit::Reinterpret => {
                let f = chosen.state.foreground.unwrap();
                assert_eq!(f.start, original.start);
                assert_eq!(f.credit, original.credit);
                assert_ne!(f.id, original.id);
                assert!(chosen.state.links.iter().all(Option::is_none));
            }
            Exit::Inactive => assert!(chosen.state.foreground.is_none()),
            _ => {
                let f = chosen.state.foreground.unwrap();
                assert_eq!(f.start, 200);
                assert_ne!(f.credit, original.credit);
                let link = chosen.state.links.iter().flatten().next().unwrap();
                assert_eq!(link.span.start, original.start);
                assert_eq!(link.right_censored, kind == Exit::Overlap);
            }
        }
        let events: Vec<_> = group.paths.iter().map(|p| p.state.event).collect();
        group.step(300, false, false, 1000, cfg, &mut p.next_span);
        assert!(group.paths.iter().all(|p| events.contains(&p.state.event)));
        assert!(group.unknown > 0.);
    }
}

#[test]
fn ordered_recall_forecast_reaches_phrase_on_later_observed_support() {
    let mut recall = recall::Recall::new(
        1,
        0,
        1000,
        100,
        crate::config::TemporalMemoryConfig {
            retention: None,
            candidates: None,
            scales: [1.; 10],
            span_hops: 16,
            episodes: 16,
            query_cadence_ms: 100,
            deadline_ms: 200,
        },
        None,
    )
    .unwrap();
    let mut phrase = Phrase::new(1, 0, 0, 1000, 100, config()).unwrap();
    let mut g = gesture_model();
    let mut emitted = 0;
    let mut scored = 0;
    for step in 1..=64 {
        let a = input(step, (((step - 1) % 16) / 2) as f64 * 0.5);
        recall.advance(&a, step * 100, None).unwrap();
        g.advance(&a, &ridges(), step * 100).unwrap();
        let memory = recall.snapshot();
        if memory.latest.is_some_and(|q| q.prediction.is_some()) {
            emitted += 1;
        }
        phrase
            .advance(&a, &g, Some(memory), None, step * 100)
            .unwrap();
        scored = phrase.snapshot().groups[0].unwrap().prediction_comparisons;
    }
    println!("I7_RECALL_HANDOFF emitted={emitted} scored={scored}");
    assert!(emitted > 0 && scored > 0);
}

#[test]
fn inactive_reentry_and_link_capacity_preserve_unresolved_history() {
    let mut p = Phrase::new(1, 0, 0, 1000, 100, config()).unwrap();
    let mut g = gesture_model();
    let a = input(1, 0.);
    g.advance(&a, &ridges(), 100).unwrap();
    p.advance(&a, &g, None, None, 100).unwrap();
    let group = p.groups[0].as_mut().unwrap();
    group.paths.truncate(1);
    group.paths[0].mass = 1.;
    group.paths[0].state.foreground = None;
    group.unknown = 0.;
    group.states = [1., 0., 0., 0., 0.];
    group.step(200, false, false, 1000, config(), &mut p.next_span);
    assert!(group.paths[0].state.foreground.is_none());
    group.step(300, true, false, 1000, config(), &mut p.next_span);
    assert_eq!(group.paths[0].state.foreground.unwrap().start, 200);
    assert!(group.paths[0].state.event.is_none());
    let mut cfg = config();
    cfg.hazard = [0.; 26];
    cfg.hazard[0] = 100.;
    cfg.exits[0][0] = 100.;
    for step in 4..=30 {
        group.step(step * 100, true, true, 1000, cfg, &mut p.next_span);
    }
    assert_eq!(group.paths[0].state.links.iter().flatten().count(), 16);
    assert!(group.paths[0].state.lost_links > 0);
    assert!(std::mem::size_of::<interpretation::Link>() <= 128);
    let sum: f64 = group.paths.iter().map(|p| p.mass).sum();
    assert!((sum + group.unknown - 1.).abs() < 1e-8);
}

#[test]
fn invalid_scales_cutpoints_and_cross_head_coefficients_are_rejected() {
    for case in 0..7 {
        let mut c = config();
        match case {
            0 => c.hazard[0] = f64::NAN,
            1 => c.deviations[0] = -1.,
            2 => c.closure.cutpoints[1] = c.closure.cutpoints[0],
            3 => c.closure.prior[0] = 0.,
            4 => c.closure.coefficients[11] = 1.,
            5 => c.continuation.coefficients[28] = f64::INFINITY,
            _ => c.closure.means[0] = f64::INFINITY,
        }
        assert!(Phrase::validate(c).is_err());
    }
    let cfg = crate::config::AppConfig {
        temporal_phrase: Some(config()),
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
    let text = toml::to_string(&config()).unwrap();
    let parsed: TemporalPhraseConfig = toml::from_str(&text).unwrap();
    Phrase::validate(parsed).unwrap();
    assert!(
        toml::from_str::<TemporalPhraseConfig>(&text.replace("cutpoints", "cutpoint")).is_err()
    );
}

#[test]
fn stale_foreign_and_future_query_results_are_masked_at_the_phrase_consumer() {
    let mut r = recall::Recall::new(
        1,
        0,
        1000,
        100,
        crate::config::TemporalMemoryConfig {
            retention: None,
            candidates: None,
            scales: [1.; 10],
            span_hops: 16,
            episodes: 16,
            query_cadence_ms: 100,
            deadline_ms: 200,
        },
        None,
    )
    .unwrap();
    for step in 1..=24 {
        r.advance(
            &input(step, (((step - 1) % 16) / 2) as f64 * 0.5),
            step * 100,
            None,
        )
        .unwrap();
    }
    let q = r.snapshot().latest.unwrap();
    assert!(q.best.is_some());
    for case in 0..6 {
        let mut candidate = q;
        match case {
            0 => {}
            1 => candidate.group.generation += 1,
            2 => candidate.available_at = candidate.issued_at + 1,
            3 => candidate.supporting_audio_end = Some(2501),
            4 => candidate.received_at = 2501,
            _ => candidate.supporting_audio_end = Some(2000),
        }
        let a = input(25, 0.);
        let mut g = gesture_model();
        g.advance(&a, &ridges(), 2500).unwrap();
        let mut p = Phrase::new(1, 0, 0, 1000, 100, config()).unwrap();
        p.advance(
            &a,
            &g,
            Some(recall::Snapshot {
                latest: Some(candidate),
                group_queries: [Some(candidate), None, None, None, None, None, None],
                ..Default::default()
            }),
            None,
            2500,
        )
        .unwrap();
        assert_eq!(
            p.groups[0].as_ref().unwrap().query.is_some(),
            case == 0,
            "case {case}"
        );
    }
}

#[test]
fn ongoing_credit_does_not_identify_a_unique_closed_endpoint_within_commit_lag() {
    let mut cfg = config();
    cfg.hazard[0] = 2.;
    let mut p = Phrase::new(1, 0, 0, 1000, 100, cfg).unwrap();
    let mut g = gesture_model();
    let mut endpoints =
        std::collections::BTreeMap::<u64, std::collections::BTreeSet<(u64, u64)>>::new();
    for step in 1..=20 {
        let a = input(step, 0.);
        g.advance(&a, &ridges(), step * 100).unwrap();
        p.advance(&a, &g, None, None, step * 100).unwrap();
        for group in p.snapshot().groups.iter().flatten() {
            for candidate in group.candidates.iter().flatten() {
                if let (Some(f), Some(ending)) =
                    (candidate.completed_foreground, candidate.completed_ending)
                {
                    assert_eq!(ending.end, f.heard_end);
                    endpoints
                        .entry(f.credit)
                        .or_default()
                        .insert((f.start, f.heard_end));
                }
            }
        }
    }
    let (credit, alternatives) = endpoints
        .iter()
        .find(|(_, ends)| {
            let values: Vec<_> = ends.iter().collect();
            values
                .windows(2)
                .any(|w| w[0].0 == w[1].0 && w[1].1 > w[0].1 && w[1].1 - w[0].1 <= 500)
        })
        .expect("the producer must preserve alternative endpoints before commitment");
    println!(
        "I8_ENDPOINT_IDENTITY credit={credit} distinct_endpoints={} first={:?}",
        alternatives.len(),
        alternatives.iter().take(3).collect::<Vec<_>>()
    );
}

#[test]
fn projected_arrival_reaches_continuation_and_preserves_issue_evidence() {
    use crate::life::action_candidates::Class;
    use crate::temporal_cognition::{action_profiles, arrival, features::Accent};
    let profiles = action_profiles::tests::pulse_model();
    let mut cfg = config();
    cfg.continuation.coefficients[11] = 2.;
    let mut phrase = Phrase::new(1, 0, 0, 48000, 512, cfg).unwrap();
    let mut g = gesture::Gesture::new(
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
    // Build actual issue observations independently of the future profile.
    for step in 1..=6 {
        let mut a = input(step, 0.5);
        a.assignment.end_sample = step * 512;
        a.correlation_window[1] = step * 512;
        a.energy = Some([1., 0., 0., 0., 0., 0., 0., 0.]);
        let raw = &mut a.features[0].as_mut().unwrap().raw;
        raw.start = (step - 1) * 512;
        raw.end = step * 512;
        raw.known_samples = 512;
        raw.source_end = raw.end;
        raw.available_end = raw.end;
        raw.values[3] = Some(0.);
        raw.values[5] = Some(0.);
        g.advance(&a, &ridges(), step * 512).unwrap();
        phrase.advance(&a, &g, None, None, step * 512).unwrap();
    }
    let group = phrase.groups[0].as_ref().unwrap();
    let handle = group.handle;
    let issue = phrase.end;
    let mut e = arrival::Engine::new(crate::config::TemporalPeriodConfig {
        model: crate::config::ArrivalModel::Hazard,
        coefficients: [0.; 18],
        means: [0.; 8],
        deviations: [1.; 8],
        horizon_sec: 1.,
    })
    .unwrap();
    e.advance(
        Some(&group.history.back().unwrap().raw),
        Some(Accent {
            group: handle,
            event_start: 2048,
            event_end: 2560,
            raw_intervals: [(1024, 1536), (1536, 2048), (2048, 2560), (2560, 3072)],
            source_start: 1024,
            source_end: 3072,
            available_end: 3072,
            weight: 1.,
            observed_prefix: 3072,
        }),
        arrival::Context::default(),
        issue,
        48000,
    )
    .unwrap();
    let frozen = e.freeze(handle, issue, 48000).unwrap();
    let original = serde_json::to_string(&phrase.snapshot()).unwrap();
    let mut found = None;
    for future_hops in [5, 6, 7] {
        let evaluation = issue + future_hops * 512;
        let cell = phrase
            .project_action_window(
                &profiles,
                0,
                Class::Continue,
                0,
                evaluation,
                handle,
                1.,
                Some(&g),
                Some(&frozen),
                None,
                None,
                None,
            )
            .unwrap();
        let forecast = cell.arrival.unwrap();
        if future_hops == 5 {
            assert!(forecast.reset_unknown);
            assert_eq!(cell.continuation_context[0], Feature::Unsupported);
        } else {
            assert_eq!(forecast.candidate_last_accent, Some(issue + 5 * 512));
            assert!(!forecast.reset_unknown);
            assert!(
                matches!(cell.continuation_context[0], Feature::Projected(v) if (v - 0.5).abs() < 1e-14)
            );
            let without = phrase
                .project_action_window(
                    &profiles,
                    0,
                    Class::Continue,
                    0,
                    evaluation,
                    handle,
                    1.,
                    Some(&g),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();
            assert!(
                cell.continuation_paths
                    .iter()
                    .flatten()
                    .zip(without.continuation_paths.iter().flatten())
                    .any(|(a, b)| a.categories != b.categories)
            );
            found = Some(cell);
        }
    }
    let cell = found.unwrap();
    for fault in 0..3 {
        let mut bad = frozen;
        match fault {
            0 => bad.group.generation += 1,
            1 => bad.issued_at += 1,
            _ => bad.sample_rate = 44100,
        }
        let invalid = phrase
            .project_action_window(
                &profiles,
                0,
                Class::Continue,
                0,
                cell.evaluation_at,
                handle,
                1.,
                Some(&g),
                Some(&bad),
                None,
                None,
                None,
            )
            .unwrap();
        assert!(invalid.arrival.is_none());
        assert_eq!(invalid.continuation_context[0], Feature::Unsupported);
    }
    assert_eq!(original, serde_json::to_string(&phrase.snapshot()).unwrap());
    println!(
        "PROJECTED_ARRIVAL_CONSUMER {}",
        serde_json::to_string(&cell).unwrap()
    );
}

#[test]
fn shared_table_rebuild_clock_survives_invalidation_and_rejects_future_or_rewound_support() {
    use crate::temporal_cognition::{action_profiles, body_model};
    let profiles = action_profiles::tests::distinct_release_model();
    let mut phrase = Phrase::new(1, 0, 0, 48000, 512, config()).unwrap();
    let mut gesture = gesture::Gesture::new(
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
    let mut table = action_profiles::Table::new();
    let mut shared = body_model::Shared {
        model_version: profiles.body_model_version,
        bus: 1,
        epoch: 0,
        end_sample: 0,
        descriptors: [None; 7],
        assignments: [None; 8],
    };
    let mut previous_end = 0;
    for (index, end) in [
        512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632,
    ]
    .into_iter()
    .enumerate()
    {
        let mut a = input(1, 0.5);
        a.assignment.end_sample = end;
        a.correlation_window[1] = end;
        let raw = &mut a.features[0].as_mut().unwrap().raw;
        raw.start = previous_end;
        raw.end = end;
        raw.known_samples = end - previous_end;
        raw.source_start = previous_end.saturating_sub(2048);
        raw.source_end = end;
        raw.available_end = end;
        previous_end = end;
        gesture.advance(&a, &ridges(), end).unwrap();
        phrase.advance(&a, &gesture, None, None, end).unwrap();
        shared.end_sample = end;
        shared.assignments[0] = (index != 1).then_some(body_model::Assignment {
            key: (0, a.group_handles[0].unwrap().generation),
            distance: 0.,
            common_coordinates: 6,
        });
        let observed = serde_json::to_string(&phrase.snapshot()).unwrap();
        let projected = table.refresh(&profiles, &phrase, &shared, &gesture, None);
        assert_eq!(projected.is_some(), index == 0 || index == 10);
        assert_eq!(observed, serde_json::to_string(&phrase.snapshot()).unwrap());
        if let Some(snapshot) = projected {
            action_profiles::tests::assert_table_matches_direct(
                &table, &profiles, &phrase, &gesture,
            );
            assert!(snapshot.cells > 0);
            assert_eq!(snapshot.issued_at, end);
            assert_eq!(
                serde_json::to_string(&snapshot).unwrap(),
                serde_json::to_string(
                    &table
                        .refresh(&profiles, &phrase, &shared, &gesture, None)
                        .unwrap()
                )
                .unwrap()
            );
        }
    }
    let stats = table.resources();
    assert_eq!(stats.rebuilds, 2);
    assert_eq!(
        stats.builds_by_assigned_prototypes,
        [0, 2, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(stats.invalidations, 1);
    assert_eq!(stats.deferred_calls, 9);
    assert_eq!(stats.attempted_cells, 2 * 7 * 32);
    assert_eq!(stats.reused_cells, 2 * 65);
    assert_eq!(stats.projection_calls, 2 * 159);
    assert_eq!(stats.last_build_sample, Some(5632));
    assert!(stats.max_build_us >= stats.last_build_us);
    assert!(stats.total_build_us >= stats.max_build_us);
    shared.end_sample = 5633;
    assert!(
        table
            .refresh(&profiles, &phrase, &shared, &gesture, None)
            .is_none()
    );
    shared.end_sample = 5632;
    assert!(
        table
            .refresh(&profiles, &phrase, &shared, &gesture, None)
            .is_none()
    );
    phrase.snapshot.end_sample = 5631;
    shared.end_sample = 5631;
    assert!(
        table
            .refresh(&profiles, &phrase, &shared, &gesture, None)
            .is_none()
    );
    assert_eq!(table.resources().rejected_inputs, 2);
    assert_eq!(table.resources().rebuilds, 2);
    assert_eq!(
        table.resources().builds_by_assigned_prototypes,
        stats.builds_by_assigned_prototypes
    );
    assert_eq!(table.resources().attempted_cells, 2 * 7 * 32);
}
