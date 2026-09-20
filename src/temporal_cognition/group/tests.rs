use super::*;
use crate::core::log2space::Log2Space;

fn config() -> Config {
    Config {
        bus: 0,
        epoch: 1,
        sample_rate: 1000,
        hop: 10,
        means: [0.; 3],
        deviations: [1.; 3],
        distance_limit: 1.0,
        residual_raw: (-1.0_f64).exp(),
    }
}

fn handle(generation: u64) -> Handle {
    Handle {
        bus: 0,
        epoch: 1,
        generation,
    }
}

fn trajectory(generation: u64, frequency: Option<f64>, envelope: Option<f64>) -> Trajectory {
    Trajectory {
        handle: handle(generation),
        point: Point {
            frequency_log2: frequency,
            log_envelope: envelope,
        },
        slopes: [None; 2],
    }
}

#[test]
fn decimal_reference_checks_weighted_maxima_masks_and_member_specific_times() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/temporal_cognition/groups.json"
    ))
    .unwrap();
    let decode = |row: &serde_json::Value| Trajectory {
        handle: handle(row["generation"].as_u64().unwrap()),
        point: Point {
            frequency_log2: row["point"][0].as_f64(),
            log_envelope: row["point"][1].as_f64(),
        },
        slopes: std::array::from_fn(|i| row["slopes"][i].as_f64()),
    };
    let mut observed_rows = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let settings = Config {
            sample_rate: case["sample_rate"].as_u64().unwrap() as u32,
            hop: case["hop"].as_u64().unwrap(),
            means: std::array::from_fn(|i| case["means"][i].as_f64().unwrap()),
            deviations: std::array::from_fn(|i| case["deviations"][i].as_f64().unwrap()),
            residual_raw: case["residual_raw"].as_f64().unwrap(),
            ..config()
        };
        let current = std::array::from_fn(|i| {
            (!case["current"][i].is_null()).then(|| decode(&case["current"][i]))
        });
        let mut groups = std::array::from_fn(|i| {
            let g = &case["groups"][i];
            Some(Group {
                handle: handle(g["generation"].as_u64().unwrap()),
                eligible: g["eligible"].as_bool().unwrap(),
                members: std::array::from_fn(|j| {
                    let m = &g["members"][j];
                    Some(Member {
                        trajectory: decode(&m["trajectory"]),
                        end_sample: m["end_sample"].as_u64().unwrap(),
                        weight: m["weight"].as_f64().unwrap(),
                    })
                }),
            })
        });
        let output = assign_and_refresh(
            &settings,
            case["end_sample"].as_u64().unwrap(),
            &current,
            &mut groups,
        )
        .unwrap();
        for i in 0..8 {
            let expected = &case["expected"][i];
            if expected.is_null() {
                assert!(output.rows[i].is_none());
                continue;
            }
            observed_rows += 1;
            let row = output.rows[i].unwrap();
            for j in 0..8 {
                assert!((row.weights[j] - expected["weights"][j].as_f64().unwrap()).abs() < 1e-13);
            }
            for j in 0..7 {
                let m = &expected["matched_members"][j];
                if m.is_null() {
                    assert!(row.matched_members[j].is_none());
                    continue;
                }
                let actual = row.matched_members[j].unwrap();
                assert_eq!(actual.member.generation, m["generation"].as_u64().unwrap());
                assert_eq!(
                    actual.slope_index,
                    m["slope_index"].as_u64().unwrap() as usize
                );
                assert_eq!(actual.end_sample, m["end_sample"].as_u64().unwrap());
                assert!((actual.distance - m["distance"].as_f64().unwrap()).abs() < 1e-13);
            }
        }
    }
    assert!(observed_rows > 350);
}

#[test]
fn bundle_unassigned_fraction_uses_its_own_supported_members_only() {
    let old = trajectory(1, None, Some(0.));
    let mut input = [None; 8];
    input[0] = Some(old);
    input[1] = Some(trajectory(2, None, Some(1.)));
    input[2] = Some(trajectory(3, None, Some(2.)));
    let mut groups = [None; 7];
    groups[0] = Some(Group::seed(handle(100), 10, &[old]).unwrap());
    let assignment = assign_and_refresh(&config(), 20, &input, &mut groups).unwrap();
    assert!(assignment.unassigned_fraction(1).unwrap() < 0.5);
    assert_eq!(assignment.unassigned_fraction(2), Some(0.5));
    assert_eq!(assignment.unassigned_fraction(4), Some(1.0));
    assert_eq!(assignment.unassigned_fraction(0), None);
    assert_eq!(assignment.unassigned_fraction(128), None);
    assert_eq!(assignment.unassigned_fraction(6), Some(0.75));
    assert_eq!(assignment.unassigned_fraction(132), Some(1.0));
}

#[test]
fn full_member_inventory_respects_896_distance_cap_and_descriptor_storage_bound() {
    let members: Vec<_> = (0..8)
        .map(|i| Trajectory {
            slopes: [Some(0.1), Some(-0.1)],
            ..trajectory(i + 1, Some(8. + i as f64 * 0.001), Some(i as f64 * 0.025))
        })
        .collect();
    let mut groups =
        std::array::from_fn(|i| Some(Group::seed(handle(i as u64 + 100), 10, &members).unwrap()));
    let current = std::array::from_fn(|i| {
        Some(Trajectory {
            handle: handle(i as u64 + 1000),
            ..members[i]
        })
    });
    let result = assign_and_refresh(&config(), 20, &current, &mut groups).unwrap();
    assert_eq!(result.distance_evaluations, 896);
    assert!(std::mem::size_of::<Option<Member>>() <= 128);
    for row in result.rows.iter().flatten() {
        assert!((row.weights.iter().sum::<f64>() - 1.0).abs() < 1e-14);
    }
}

#[test]
#[ignore = "release saved-member assignment cost; excludes lifecycle, audio and full O04"]
fn group_assignment_cost_probe() {
    use std::hint::black_box;
    use std::time::Instant;
    let settings = config();
    for (group_count, member_count, current_count) in [(1, 1, 1), (7, 8, 8)] {
        let members: Vec<_> = (0..member_count)
            .map(|i| Trajectory {
                slopes: [Some(0.1), Some(-0.1)],
                ..trajectory(
                    i as u64 + 1,
                    Some(8. + i as f64 * 0.001),
                    Some(i as f64 * 0.025),
                )
            })
            .collect();
        let mut groups = std::array::from_fn(|i| {
            (i < group_count).then(|| {
                let mut group = Group::seed(handle(i as u64 + 100), 10, &members).unwrap();
                for member in group.members.iter_mut().flatten() {
                    member.weight = 0.125;
                }
                group
            })
        });
        let mut current = [None; 8];
        let mut timings = Vec::with_capacity(6000);
        let mut comparisons = 0;
        for step in 1..=6100 {
            for (i, slot) in current.iter_mut().enumerate().take(current_count) {
                *slot = Some(Trajectory {
                    slopes: [Some(0.1), Some(-0.1)],
                    ..trajectory(
                        i as u64 + 1000,
                        Some(8. + i as f64 * 0.001 + (step % 7) as f64 * 0.0003),
                        Some(i as f64 * 0.025 + (step % 5) as f64 * 0.0125),
                    )
                });
            }
            let started = Instant::now();
            let result = black_box(
                assign_and_refresh(&settings, (step + 1) * 10, black_box(&current), &mut groups)
                    .unwrap(),
            );
            let elapsed = started.elapsed().as_secs_f64() * 1e6;
            if step > 100 {
                comparisons += result.distance_evaluations;
                timings.push(elapsed);
            }
        }
        timings.sort_by(f64::total_cmp);
        assert_eq!(
            comparisons,
            6000 * group_count * member_count * current_count * 2
        );
        println!(
            "group_cost {}",
            serde_json::json!({
                "groups": group_count, "members": member_count, "current_trajectories": current_count,
                "calls": timings.len(), "distance_evaluations": comparisons,
                "median_us": timings[3000], "p99_us": timings[5939], "max_us": timings[5999],
                "member_slot_bytes": std::mem::size_of::<Option<Member>>(),
                "group_bytes": std::mem::size_of::<Group>(), "assignment_bytes": std::mem::size_of::<Assignment>(),
                "input": "varying frequency/envelope, nonzero slopes/residuals, initial member weights .125; constructed scales",
                "full_O04": false
            })
        );
    }
}

#[test]
#[ignore = "release group-energy scan cost; excludes NSGT, grouping, queues and full O04"]
fn group_energy_cost_probe() {
    use std::hint::black_box;
    use std::time::Instant;
    let assignment = Assignment {
        end_sample: 20,
        group_handles: std::array::from_fn(|i| Some(handle(i as u64 + 100))),
        rows: std::array::from_fn(|i| {
            let raw: [f64; 8] = std::array::from_fn(|j| (i + j + 1) as f64);
            let total: f64 = raw.iter().sum();
            Some(Row {
                trajectory: handle(i as u64 + 1),
                weights: raw.map(|w| w / total),
                matched_members: [None; 7],
            })
        }),
        distance_evaluations: 0,
    };
    for bins_per_oct in [128, 512, 2048] {
        let space = Log2Space::new(256., 512., bins_per_oct);
        let power: Vec<_> = (0..space.n_bins())
            .map(|i| ((i * 29 % 67) + 1) as f32 / 8.)
            .collect();
        let peaks = super::super::trajectory::partition(&space, &power, 0.25)
            .unwrap()
            .peak_bins;
        let mut scans = std::array::from_fn(|_| vec![0.; space.n_bins()]);
        let mut timings = Vec::with_capacity(6000);
        for step in 0..6100 {
            let mono = 0.25 + (step % 13) as f64 * 0.001;
            let start = Instant::now();
            let result = black_box(
                allocate_energy(
                    &space,
                    black_box(&power),
                    Some(black_box(mono)),
                    &peaks,
                    black_box(&assignment),
                    &mut scans,
                )
                .unwrap()
                .unwrap(),
            );
            let elapsed = start.elapsed().as_secs_f64() * 1e6;
            assert!((result.group_energy.iter().sum::<f64>() - mono).abs() < 1e-12);
            if step >= 100 {
                timings.push(elapsed);
            }
        }
        timings.sort_by(f64::total_cmp);
        println!(
            "group_energy_cost {}",
            serde_json::json!({
                "bins": space.n_bins(), "calls": timings.len(), "median_us": timings[3000],
                "p99_us": timings[5939], "max_us": timings[5999],
                "output_scan_bytes": 8*space.n_bins()*std::mem::size_of::<f64>(),
                "input": "positive spectrum, varying mono energy, unequal constructed fractional weights; all8 groups populated",
                "full_O04": false
            })
        );
    }
}

#[test]
fn no_eligible_reference_is_residual_but_missing_trajectory_is_unknown() {
    let mut groups = [None; 7];
    let current = [
        Some(trajectory(1, None, Some(0.))),
        None,
        Some(trajectory(2, None, None)),
        None,
        None,
        None,
        None,
        None,
    ];
    let result = assign_and_refresh(&config(), 10, &current, &mut groups).unwrap();
    assert_eq!(result.end_sample, 10);
    assert_eq!(result.group_handles, [None; 7]);
    assert_eq!(
        result.rows[0].unwrap().weights,
        [0., 0., 0., 0., 0., 0., 0., 1.]
    );
    assert_eq!(result.rows[0].unwrap().trajectory, handle(1));
    assert!(result.rows[1].is_none() && result.rows[2].is_none());
    assert_eq!(result.distance_evaluations, 0);
}

#[test]
fn equal_groups_remain_fractional_and_members_use_maximum_not_centroid_or_mean() {
    let a = trajectory(1, Some(8.), Some(0.));
    let b = trajectory(2, Some(12.), Some(0.));
    let mut groups = [None; 7];
    groups[0] = Some(Group::seed(handle(100), 10, &[a, b]).unwrap());
    groups[1] = Some(Group::seed(handle(101), 10, &[a]).unwrap());
    let mut current = [None; 8];
    current[0] = Some(a);
    let result = assign_and_refresh(&config(), 20, &current, &mut groups).unwrap();
    let row = result.rows[0].unwrap();
    let expected = 1.0 / (2.0 + (-1.0_f64).exp());
    assert!((row.weights[0] - expected).abs() < 1e-14);
    assert_eq!(row.weights[0], row.weights[1]);
    assert!((row.weights.iter().sum::<f64>() - 1.0).abs() < 1e-14);
    assert_eq!(row.matched_members[0].unwrap().member, a.handle);
    assert_eq!(row.matched_members[0].unwrap().end_sample, 10);
    assert_eq!(groups[0].unwrap().members[0].unwrap().end_sample, 20);
    assert_eq!(
        groups[0].unwrap().members[0].unwrap().weight,
        row.weights[0]
    );
    assert!(groups[0].unwrap().members[1].is_none());
}

#[test]
fn all_rows_use_saved_members_before_refresh_and_ties_use_generation_and_slope() {
    let mut member = trajectory(1, Some(8.), Some(0.));
    member.slopes = [Some(0.), Some(0.)];
    let later = Trajectory {
        handle: handle(2),
        ..member
    };
    let mut groups = [None; 7];
    groups[0] = Some(Group::seed(handle(100), 10, &[later, member]).unwrap());
    let mut current = [None; 8];
    current[0] = Some(member);
    current[1] = Some(later);
    let result = assign_and_refresh(&config(), 20, &current, &mut groups).unwrap();
    for row in result.rows.iter().flatten() {
        let matched = row.matched_members[0].unwrap();
        assert_eq!(matched.member, member.handle);
        assert_eq!(matched.slope_index, 0);
        assert_eq!(matched.end_sample, 10);
        assert_eq!(matched.distance, 0.);
    }
    assert_eq!(
        result.rows[0].unwrap().weights,
        result.rows[1].unwrap().weights
    );
    assert_eq!(result.distance_evaluations, 8);
}

#[test]
fn seed_changes_only_next_hop_and_superseded_or_unobserved_references_do_not_refresh() {
    let t = trajectory(1, Some(8.), Some(0.));
    let mut current = [None; 8];
    current[0] = Some(t);
    let mut groups = [None; 7];
    let first = assign_and_refresh(&config(), 10, &current, &mut groups).unwrap();
    groups[0] = Some(Group::seed(handle(100), 10, &[t]).unwrap());
    assert_eq!(first.rows[0].unwrap().weights[7], 1.);
    assert_eq!(first.group_handles[0], None);
    assert!(assign_and_refresh(&config(), 10, &current, &mut groups).is_err());
    let second = assign_and_refresh(&config(), 20, &current, &mut groups).unwrap();
    assert!(second.rows[0].unwrap().weights[0] > 0.5);
    groups[0].as_mut().unwrap().eligible = false;
    let before = groups;
    let third = assign_and_refresh(&config(), 30, &current, &mut groups).unwrap();
    assert_eq!(third.rows[0].unwrap().weights[7], 1.);
    assert_eq!(groups, before);
    groups[0].as_mut().unwrap().eligible = true;
    let before = groups;
    assign_and_refresh(&config(), 40, &[None; 8], &mut groups).unwrap();
    assert_eq!(groups, before);
}

#[test]
fn threshold_and_gap_keep_envelope_only_support_and_residual_sensitivity_explicit() {
    let old = trajectory(1, None, Some(0.));
    let current = trajectory(2, None, Some(1.));
    for multiplier in [0.5, 1.0, 2.0] {
        let mut config = config();
        config.residual_raw *= multiplier;
        let mut groups = [None; 7];
        groups[0] = Some(Group::seed(handle(100), 10, &[old]).unwrap());
        let mut input = [None; 8];
        input[0] = Some(current);
        let out = assign_and_refresh(&config, 1000, &input, &mut groups)
            .unwrap()
            .rows[0]
            .unwrap();
        assert!((out.weights[0] - 1.0 / (1.0 + multiplier)).abs() < 1e-14);
        assert_eq!(out.matched_members[0].unwrap().distance, 1.0);
    }
    let mut groups = [None; 7];
    groups[0] = Some(Group::seed(handle(100), 10, &[old]).unwrap());
    let mut input = [None; 8];
    input[0] = Some(Trajectory {
        point: Point {
            log_envelope: Some(1.000001),
            ..current.point
        },
        ..current
    });
    assert_eq!(
        assign_and_refresh(&config(), 20, &input, &mut groups)
            .unwrap()
            .rows[0]
            .unwrap()
            .weights[7],
        1.0
    );
}

#[test]
fn malformed_or_future_members_reject_before_any_reference_changes() {
    let t = trajectory(1, Some(8.), Some(0.));
    let mut groups = [None; 7];
    groups[0] = Some(Group::seed(handle(100), 10, &[t]).unwrap());
    let mut current = [None; 8];
    current[0] = Some(t);
    for case in 0..5 {
        let mut invalid = groups;
        let m = invalid[0].as_mut().unwrap().members[0].as_mut().unwrap();
        match case {
            0 => m.end_sample = 30,
            1 => m.weight = -1.,
            2 => m.trajectory.handle.epoch = 0,
            3 => m.trajectory.point.log_envelope = Some(f64::NAN),
            _ => m.end_sample = 15,
        }
        let before = format!("{invalid:?}");
        assert!(assign_and_refresh(&config(), 20, &current, &mut invalid).is_err());
        assert_eq!(format!("{invalid:?}"), before);
    }
    current[1] = Some(t);
    let before = groups;
    assert!(assign_and_refresh(&config(), 20, &current, &mut groups).is_err());
    assert_eq!(groups, before);
}

#[test]
fn spectral_group_assignment_conserves_energy_and_masks_unknown_allocation() {
    let space = Log2Space::new(1., 2., 8);
    let power = [4., 0., 0., 1., 1., 0., 0., 0., 2.];
    let partition = super::super::trajectory::partition(&space, &power, 8.).unwrap();
    let a = trajectory(1, Some(8.), Some(0.));
    let b = trajectory(2, Some(10.), Some(0.));
    let mut groups = [None; 7];
    groups[0] = Some(Group::seed(handle(100), 10, &[a]).unwrap());
    let mut current = [None; 8];
    current[0] = Some(a);
    current[1] = Some(b);
    current[2] = Some(trajectory(3, None, Some(0.)));
    let assignment = assign_and_refresh(&config(), 20, &current, &mut groups).unwrap();
    let mut scans = std::array::from_fn(|_| vec![99.; space.n_bins()]);
    let energy = allocate_energy(
        &space,
        &power,
        Some(8.),
        &partition.peak_bins,
        &assignment,
        &mut scans,
    )
    .unwrap()
    .unwrap();
    assert!(energy.spectral_shape_supported);
    assert!((energy.group_energy.iter().sum::<f64>() - 8.).abs() < 1e-13);
    for bin in 0..space.n_bins() {
        assert!((scans.iter().map(|s| s[bin]).sum::<f64>() - f64::from(power[bin])).abs() < 1e-13);
    }
    let expected: f64 = partition
        .energy
        .iter()
        .enumerate()
        .filter(|(_, e)| **e > 0.)
        .map(|(i, e)| e * assignment.rows[i].unwrap().weights[0])
        .sum();
    assert!((energy.group_energy[0] - expected).abs() < 1e-13);
    let mut unknown = assignment;
    unknown.rows[0] = None;
    let previous = scans.clone();
    assert!(
        allocate_energy(
            &space,
            &power,
            Some(8.),
            &partition.peak_bins,
            &unknown,
            &mut scans
        )
        .unwrap()
        .is_none()
    );
    assert_eq!(scans, previous);
    let silent = allocate_energy(
        &space,
        &power,
        Some(0.),
        &partition.peak_bins,
        &unknown,
        &mut scans,
    )
    .unwrap()
    .unwrap();
    assert_eq!(silent.group_energy, [0.; 8]);
    let unavailable = allocate_energy(
        &space,
        &[0.; 9],
        Some(2.),
        &[None; 7],
        &assignment,
        &mut scans,
    )
    .unwrap()
    .unwrap();
    assert!(!unavailable.spectral_shape_supported);
    assert_eq!(unavailable.group_energy, [0., 0., 0., 0., 0., 0., 0., 2.]);
}

#[test]
#[should_panic(expected = "scan length mismatch: group_power_scan")]
fn group_energy_input_obeys_log2_scan_boundary() {
    let space = Log2Space::new(1., 2., 8);
    let assignment = assign_and_refresh(&config(), 10, &[None; 8], &mut [None; 7]).unwrap();
    allocate_energy(
        &space,
        &[1.],
        None,
        &[None; 7],
        &assignment,
        &mut std::array::from_fn(|_| vec![0.; 9]),
    )
    .unwrap();
}

#[test]
#[should_panic(expected = "scan length mismatch: group_energy_scan")]
fn group_energy_output_obeys_log2_scan_boundary() {
    let space = Log2Space::new(1., 2., 8);
    let assignment = assign_and_refresh(&config(), 10, &[None; 8], &mut [None; 7]).unwrap();
    let mut scans = std::array::from_fn(|_| vec![0.; 9]);
    scans[7].pop();
    allocate_energy(&space, &[0.; 9], None, &[None; 7], &assignment, &mut scans).unwrap();
}
