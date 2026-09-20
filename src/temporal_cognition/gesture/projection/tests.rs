use super::*;

fn model(unknown: f64) -> (Gesture, ridge::Handle) {
    let handle = ridge::Handle {
        bus: 0,
        epoch: 0,
        generation: 1,
    };
    let mut model = Gesture::new(
        0,
        0,
        48000,
        512,
        TemporalGestureConfig {
            rms_reference: 0.1,
            means: [0.; 5],
            deviations: [1.; 5],
            coefficients: [[[0.; 11]; 4]; 4],
        },
    )
    .unwrap();
    model.last = Some(48000);
    model.groups[0] = Some(Group {
        handle,
        born: 0,
        end: 48000,
        paths: if unknown == 1. {
            Vec::new()
        } else {
            vec![Path {
                state: State::Attack,
                entered: 24000,
                run: Run::default(),
                mass: 1. - unknown,
            }]
        },
        scratch: Vec::new(),
        unknown,
        support: VecDeque::new(),
    });
    (model, handle)
}

fn frame(start: u64, quiet: bool) -> Frame {
    let mut raw = [Feature::Projected(0.); 10];
    raw[2] = Feature::Projected(if quiet { 1e-6_f64.log2() } else { 0. });
    Frame {
        start,
        end: start + 512,
        source_end: start + 512,
        available: start + 512,
        raw,
        energy: Feature::Projected(if quiet { 0. } else { 1. }),
    }
}

#[test]
fn immutable_trajectory_prefix_reuses_only_complete_hops_without_changing_results() {
    let (mut model, mut handle) = model(0.25);
    model.config.coefficients[0][2][5] = 0.7;
    model.config.coefficients[1][2][5] = -0.2;
    let mut cache = ProjectionCache::default();
    let mut comparisons = 0;
    for epoch in 0..2 {
        if epoch > 0 {
            handle.generation += 1;
            model.groups[0].as_mut().unwrap().handle = handle;
        }
        for (trajectory, quiet) in [(0, true), (1, false), (0, true)] {
            for duration in [1, 511, 512, 513, 1025, 4000, 4000, 2000, 6000, 8192] {
                let frames = || (0..16).map(|i| frame(48000 + i * 512, quiet));
                let direct = model.project_states(handle, 48000, 48000 + duration, frames(), None);
                let cached = model.project_states(
                    handle,
                    48000,
                    48000 + duration,
                    frames(),
                    Some((trajectory, &mut cache)),
                );
                assert!(direct.is_some());
                assert_eq!(
                    serde_json::to_vec(&direct).unwrap(),
                    serde_json::to_vec(&cached).unwrap()
                );
                comparisons += 1;
            }
        }
    }
    assert!(cache.reused_frames > 0);
    assert!(cache.prefix.unwrap().pruned_mass > 0.);
    assert!(std::mem::size_of::<ProjectionCache>() <= 1024);
    println!(
        "I10_ARTICULATION_PREFIX comparisons={comparisons} advanced={} reused={} scratch_bytes={}",
        cache.advanced_frames,
        cache.reused_frames,
        std::mem::size_of::<ProjectionCache>()
    );
}

#[test]
fn conditional_transition_matches_closed_form_without_evidence_admission_or_decay() {
    let (model, handle) = model(0.25);
    for duration in [1, 256, 512] {
        let result = model
            .project_states(
                handle,
                48000,
                48000 + duration,
                std::iter::once(frame(48000, true)),
                None,
            )
            .unwrap();
        let survival = (-3. * 2_f64.ln() * duration as f64 / 48000.).exp();
        assert!((result.states[0] - 0.75 * survival).abs() < 1e-14);
        for mass in &result.states[1..] {
            assert!((*mass - 0.25 * (1. - survival)).abs() < 1e-14);
        }
        assert_eq!(result.original_unknown, 0.25);
        assert_eq!(result.unknown, 0.25);
        assert_eq!(result.pruned_mass, 0.);
        assert!(result.projected);
    }
    let at_issue = model
        .project_states(handle, 48000, 48000, std::iter::empty(), None)
        .unwrap();
    assert_eq!(at_issue.states, [0.75, 0., 0., 0.]);
    assert!(!at_issue.projected);
    let loud = model
        .project_states(
            handle,
            48000,
            48512,
            std::iter::once(frame(48000, false)),
            None,
        )
        .unwrap();
    assert_eq!(loud.states[3], 0.);
    assert!((loud.states.iter().sum::<f64>() - 0.75).abs() < 1e-14);
    let (unknown, handle) = self::model(1.);
    let unknown = unknown
        .project_states(
            handle,
            48000,
            48512,
            std::iter::once(frame(48000, true)),
            None,
        )
        .unwrap();
    assert_eq!(unknown.states, [0.; 4]);
    assert_eq!(unknown.unknown, 1.);
    assert_eq!(unknown.retained_paths, 0);
}

#[test]
fn conditional_projection_requires_supported_identity_and_contiguous_future() {
    let (model, handle) = model(0.);
    assert!(
        model
            .project_states(
                handle,
                48001,
                48512,
                std::iter::once(frame(48000, true)),
                None
            )
            .is_none()
    );
    assert!(
        model
            .project_states(
                ridge::Handle {
                    generation: 2,
                    ..handle
                },
                48000,
                48512,
                std::iter::once(frame(48000, true)),
                None
            )
            .is_none()
    );
    assert!(
        model
            .project_states(
                handle,
                48000,
                48513,
                std::iter::once(frame(48000, true)),
                None
            )
            .is_none()
    );
    assert!(
        model
            .project_states(handle, 48000, 47999, std::iter::empty(), None)
            .is_none()
    );
    assert!(
        model
            .project_states(handle, 48000, 240001, std::iter::empty(), None)
            .is_none()
    );
    for fault in 0..5 {
        let mut bad = frame(48000, true);
        match fault {
            0 => bad.start += 1,
            1 => bad.end = bad.start,
            2 => bad.raw[2] = Feature::Unsupported,
            3 => bad.raw[3] = Feature::Observed(0.),
            _ => bad.raw[4] = Feature::Projected(f64::NAN),
        }
        assert!(
            model
                .project_states(handle, 48000, 48512, std::iter::once(bad), None)
                .is_none(),
            "fault {fault}"
        );
    }
}

#[test]
fn bounded_conditional_paths_preserve_pruned_mass_and_original_observation() {
    let (model, handle) = model(0.25);
    let before = format!("{:?}", model.groups[0].as_ref().unwrap().paths);
    let snapshot = serde_json::to_string(&model.snapshot).unwrap();
    let project = || {
        model
            .project_states(
                handle,
                48000,
                240000,
                (0..375).map(|i| frame(48000 + i * 512, i > 180)),
                None,
            )
            .unwrap()
    };
    let a = project();
    let b = project();
    assert!(a.pruned_mass > 0.);
    assert!(a.retained_paths <= 15);
    assert!((a.states.iter().sum::<f64>() + a.unknown - 1.).abs() < 1e-12);
    assert_eq!(a.unknown, a.original_unknown + a.pruned_mass);
    assert_eq!(
        serde_json::to_string(&a).unwrap(),
        serde_json::to_string(&b).unwrap()
    );
    assert_eq!(
        before,
        format!("{:?}", model.groups[0].as_ref().unwrap().paths)
    );
    assert_eq!(snapshot, serde_json::to_string(&model.snapshot).unwrap());
}

#[test]
fn shared_rate_formula_preserves_missing_motion_and_elapsed_scaling() {
    let (mut model, _) = model(0.);
    model.config.means[4] = 0.25;
    model.config.deviations[4] = 0.5;
    model.config.coefficients[0][2][5] = 2.;
    model.config.coefficients[0][2][9] = 3.;
    let missing = rates(
        State::Attack,
        &rate_features([Some(0.), Some(0.), Some(0.), None], &model.config),
        0.5,
        &model.config,
    )
    .unwrap();
    assert!((missing[2] - 4_f64.exp().ln_1p()).abs() < 1e-14);
    let known = rates(
        State::Attack,
        &rate_features([Some(0.); 4], &model.config),
        0.5,
        &model.config,
    )
    .unwrap();
    assert!((known[2] - 1_f64.exp().ln_1p()).abs() < 1e-14);
    assert_eq!(missing[0], 0.);
    assert!(
        rates(
            State::Attack,
            &rate_features([None; 4], &model.config),
            f64::INFINITY,
            &model.config
        )
        .is_err()
    );
}

#[test]
fn reused_frame_rate_features_match_full_design_for_all_missing_masks_and_states() {
    let (mut model, _) = model(0.);
    model.config.means = [0.2, -0.1, 0.7, 0.9, 0.25];
    model.config.deviations = [0., 2., 1e-7, 0.3, 0.5];
    model.config.coefficients = std::array::from_fn(|s| {
        std::array::from_fn(|t| {
            std::array::from_fn(|i| ((s * 7 + t * 3 + i) % 9) as f64 * 0.05 - 0.2)
        })
    });
    let cfg = &model.config;
    for mask in 0..16 {
        let values: [Option<f64>; 4] =
            std::array::from_fn(|i| (mask & (1 << i) != 0).then_some([0.17, -0., 0.0003, 0.8][i]));
        let common = rate_features(values, cfg);
        for log_elapsed in [0., -0., 1e-12, 0.5, 4_f64.ln_1p(), 100.] {
            let raw = [
                values[0],
                values[1],
                values[2],
                values[3],
                Some(log_elapsed),
            ];
            let design: [f64; 11] = std::array::from_fn(|i| match i {
                0 => 1.,
                1..=5 => raw[i - 1].map_or(0., |v| {
                    (v - cfg.means[i - 1]) / cfg.deviations[i - 1].max(1e-6)
                }),
                _ => f64::from(raw[i - 6].is_none()),
            });
            for state in STATES {
                let expected: [f64; 4] = std::array::from_fn(|target| {
                    if target == state as usize {
                        return 0.;
                    }
                    let z: f64 = cfg.coefficients[state as usize][target]
                        .iter()
                        .zip(design)
                        .map(|(a, b)| a * b)
                        .sum();
                    z.max(0.) + (-z.abs()).exp().ln_1p()
                });
                let actual = rates(state, &common, log_elapsed, cfg).unwrap();
                assert_eq!(
                    actual.map(f64::to_bits),
                    expected.map(f64::to_bits),
                    "mask {mask} state {state:?} elapsed {log_elapsed}"
                );
            }
        }
    }
}

#[test]
fn pre_action_checkpoint_reuses_common_prefix_after_diverging_suffixes() {
    let (mut model, mut handle) = model(0.25);
    model.config.coefficients[0][2][5] = 0.7;
    model.config.coefficients[1][2][5] = -0.2;
    let cuts = [2_u64, 4, 1, 8, 16];
    let trajectories: Vec<Vec<Frame>> = cuts
        .into_iter()
        .map(|cut| (0..24).map(|i| frame(48000 + i * 512, i >= cut)).collect())
        .collect();
    let mut cache = ProjectionCache::default();
    let mut first_generation_reused = 0;
    for generation in 1..=2 {
        handle.generation = generation;
        model.groups[0].as_mut().unwrap().handle = handle;
        let reused_before_group = cache.reused_frames;
        for (iteration, (trajectory, duration)) in [
            (0, 5 * 512),
            (1, 7 * 512),
            (2, 6 * 512),
            (3, 12 * 512 + 13),
            (4, 20 * 512),
            (0, 511),
            (1, 7 * 512),
            (1, 4 * 512),
        ]
        .into_iter()
        .enumerate()
        {
            let direct = model
                .project_states(
                    handle,
                    48000,
                    48000 + duration,
                    trajectories[trajectory].iter().copied(),
                    None,
                )
                .unwrap();
            cache.retag_matching_prefix(
                trajectory,
                48000 + cuts[trajectory] * 512 + 17,
                |old, count| {
                    trajectories[old][..count as usize]
                        .iter()
                        .zip(&trajectories[trajectory])
                        .all(|(a, b)| {
                            a.raw.map(|v| v.value().map(f64::to_bits))
                                == b.raw.map(|v| v.value().map(f64::to_bits))
                        })
                },
            );
            let reused_before = cache.reused_frames;
            let cached = model
                .project_states(
                    handle,
                    48000,
                    48000 + duration,
                    trajectories[trajectory].iter().copied(),
                    Some((trajectory, &mut cache)),
                )
                .unwrap();
            assert_eq!(
                serde_json::to_vec(&direct).unwrap(),
                serde_json::to_vec(&cached).unwrap()
            );
            if iteration == 0 {
                assert_eq!(cache.reused_frames, reused_before_group);
            }
            if iteration == 1 {
                assert_eq!(cache.reused_frames - reused_before, 2);
                assert_eq!(cache.branch_prefix.unwrap().frames, 4);
                assert_eq!(cache.prefix.unwrap().frames, 7);
            }
            if iteration == 2 {
                assert_eq!(cache.reused_frames, reused_before);
            }
            if let Some(prefix) = cache.branch_prefix {
                assert!(prefix.end <= 48000 + cuts[trajectory] * 512 + 17);
                assert_eq!((prefix.end - 48000) % 512, 0);
            }
        }
        if generation == 1 {
            first_generation_reused = cache.reused_frames;
        }
    }
    assert!(first_generation_reused > 0);
    let issue = 48512;
    model.last = Some(issue);
    model.groups[0].as_mut().unwrap().end = issue;
    cache.retag_matching_prefix(1, issue + 4 * 512, |_, _| true);
    let reused_before = cache.reused_frames;
    let frames = || (0..8).map(|i| frame(issue + i * 512, i >= 4));
    let direct = model
        .project_states(handle, issue, issue + 8 * 512, frames(), None)
        .unwrap();
    let cached = model
        .project_states(
            handle,
            issue,
            issue + 8 * 512,
            frames(),
            Some((1, &mut cache)),
        )
        .unwrap();
    assert_eq!(cache.reused_frames, reused_before);
    assert_eq!(
        serde_json::to_vec(&direct).unwrap(),
        serde_json::to_vec(&cached).unwrap()
    );
    assert!(std::mem::size_of::<ProjectionCache>() <= 1024);
    println!(
        "I10_BRANCH_PREFIX advanced={} reused={} scratch_bytes={}",
        cache.advanced_frames,
        cache.reused_frames,
        std::mem::size_of::<ProjectionCache>()
    );
}

#[test]
fn retagged_prefix_preserves_partial_hops_and_rejects_rewind_or_new_group() {
    let (mut model, mut handle) = model(0.25);
    model.config.coefficients[0][2][5] = 0.7;
    model.config.coefficients[1][2][5] = -0.2;
    let trajectories: Vec<Vec<Frame>> = [2, 4, 1, 8, 16]
        .into_iter()
        .map(|cut| (0..16).map(|i| frame(48000 + i * 512, i >= cut)).collect())
        .collect();
    let mut cache = ProjectionCache::default();
    for generation in 1..=2 {
        handle.generation = generation;
        model.groups[0].as_mut().unwrap().handle = handle;
        for (trajectory, duration) in [
            (0, 1025),
            (1, 2050),
            (2, 2500),
            (3, 3000),
            (4, 4001),
            (0, 500),
            (1, 8192),
        ] {
            let direct = model
                .project_states(
                    handle,
                    48000,
                    48000 + duration,
                    trajectories[trajectory].iter().copied(),
                    None,
                )
                .unwrap();
            cache.retag_matching_prefix(trajectory, 48000 + duration, |old, count| {
                trajectories[old][..count as usize]
                    .iter()
                    .zip(&trajectories[trajectory])
                    .all(|(a, b)| {
                        a.raw.map(|v| v.value().map(f64::to_bits))
                            == b.raw.map(|v| v.value().map(f64::to_bits))
                    })
            });
            let cached = model
                .project_states(
                    handle,
                    48000,
                    48000 + duration,
                    trajectories[trajectory].iter().copied(),
                    Some((trajectory, &mut cache)),
                )
                .unwrap();
            assert_eq!(
                serde_json::to_vec(&direct).unwrap(),
                serde_json::to_vec(&cached).unwrap(),
                "generation {generation} trajectory {trajectory} duration {duration}"
            );
        }
    }
    assert!(cache.reused_frames > 0);
    assert!(std::mem::size_of::<ProjectionCache>() <= 1024);
}
