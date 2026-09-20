use super::*;
use crate::temporal_cognition::ridge::Point;

fn h(generation: u64) -> Handle {
    Handle {
        bus: 0,
        epoch: 2,
        generation,
    }
}

fn config() -> Config {
    Config {
        bus: 0,
        epoch: 2,
        sample_rate: 1000,
        hop: 10,
        means: [0.; 3],
        deviations: [1.; 3],
        distance_limit: 1.,
        residual_raw: (-1.0_f64).exp(),
    }
}

fn current(ids: &[u64]) -> [Option<Trajectory>; 8] {
    std::array::from_fn(|i| {
        ids.get(i).map(|id| Trajectory {
            handle: h(*id),
            point: Point {
                frequency_log2: Some(10. * *id as f64),
                log_envelope: Some(-1.),
            },
            slopes: [Some(0.), Some(0.1)],
        })
    })
}

fn proposal(p: &Prepared, kind: Kind, parents: &[u64], left: &[u64], right: &[u64]) -> Candidate {
    let handles = |x: &[u64]| x.iter().copied().map(h).collect::<Vec<_>>();
    let key = Key::new(kind, &handles(parents), &handles(left), &handles(right)).unwrap();
    Candidate::measure(
        key,
        &p.assignment,
        Some(if kind == Kind::Split { 0.1 } else { 0.9 }),
    )
    .unwrap()
    .unwrap()
}

fn populate(n: usize) -> Lifecycle {
    let mut state = Lifecycle::new(0, 2, 0, 10, 2000, 1e-6).unwrap();
    let ids: Vec<_> = (1..=n as u64).collect();
    let p = state.prepare(&config(), 10, &current(&ids)).unwrap();
    let proposals: Vec<_> = ids
        .iter()
        .map(|id| proposal(&p, Kind::Birth, &[], &[*id], &[]))
        .collect();
    let out = state.commit(p, [Some(0.); 7], &proposals).unwrap();
    assert_eq!(
        out.admissions
            .iter()
            .flatten()
            .filter(|a| a.rejection.is_none())
            .count(),
        n.min(7)
    );
    state
}

#[test]
fn birth_seeds_next_hop_without_reassigning_current_hop_or_residual_identity() {
    let mut state = Lifecycle::new(0, 2, 0, 10, 2000, 1e-6).unwrap();
    let now = current(&[1, 2]);
    let p = state.prepare(&config(), 10, &now).unwrap();
    assert!(state.slots.iter().all(Option::is_none));
    let c = proposal(&p, Kind::Birth, &[], &[1, 2], &[]);
    let out = state.commit(p, [Some(0.); 7], &[c]).unwrap();
    assert_eq!(state.residual, h(1));
    assert_eq!(out.admissions[0].unwrap().children, [Some(h(2)), None]);
    assert!(out.assignment.group_handles.iter().all(Option::is_none));
    for row in out.assignment.rows.iter().flatten() {
        assert_eq!(row.weights[7], 1.);
    }
    let group = state.slots[0].unwrap().group;
    assert_eq!(group.members.iter().flatten().count(), 2);
    assert!(
        group
            .members
            .iter()
            .flatten()
            .all(|m| m.weight == 1. && m.end_sample == 10)
    );
    let next = state.prepare(&config(), 20, &now).unwrap();
    for row in next.assignment.rows.iter().flatten() {
        assert!(row.weights[0] > 0. && row.weights[7] < 1.);
        assert!((row.weights.iter().sum::<f64>() - 1.).abs() < 1e-14);
    }
    assert_eq!(state.slots[0].unwrap().group, group);
}

#[test]
fn split_and_merge_use_fresh_handles_and_keep_superseded_pre_hop_references() {
    let mut state = Lifecycle::new(0, 2, 0, 10, 2000, 1e-6).unwrap();
    let mut now = current(&[1, 2]);
    now[1].as_mut().unwrap().point.frequency_log2 = Some(10.002);
    let p = state.prepare(&config(), 10, &now).unwrap();
    let birth = proposal(&p, Kind::Birth, &[], &[1, 2], &[]);
    state.commit(p, [Some(0.); 7], &[birth]).unwrap();
    let original = state.slots[0].unwrap().group;
    for t in now.iter_mut().flatten() {
        *t.point.frequency_log2.as_mut().unwrap() += 0.001;
    }
    let p = state.prepare(&config(), 20, &now).unwrap();
    assert!(
        p.refreshed[0]
            .unwrap()
            .members
            .iter()
            .flatten()
            .all(|m| m.end_sample == 20)
    );
    let split = proposal(&p, Kind::Split, &[2], &[1], &[2]);
    let out = state
        .commit(p, [Some(0.2), None, None, None, None, None, None], &[split])
        .unwrap();
    assert_eq!(
        out.admissions[0].unwrap().children,
        [Some(h(3)), Some(h(4))]
    );
    assert_eq!(out.superseded[0].unwrap(), original);
    assert!(!state.slots[0].unwrap().group.eligible);
    assert_eq!(state.slots[0].unwrap().group.members, original.members);
    assert_eq!(
        out.assignment.group_handles,
        [Some(h(2)), None, None, None, None, None, None]
    );
    assert!(
        out.assignment
            .rows
            .iter()
            .flatten()
            .all(|r| r.weights[0] > 0.)
    );
    let p = state.prepare(&config(), 30, &now).unwrap();
    assert!(
        p.assignment
            .rows
            .iter()
            .flatten()
            .all(|r| r.weights[0] == 0. && r.weights[1] > 0. && r.weights[2] > 0.)
    );
    let merge = proposal(&p, Kind::Merge, &[3, 4], &[1], &[2]);
    let out = state
        .commit(
            p,
            [Some(0.), Some(0.1), Some(0.1), None, None, None, None],
            &[merge],
        )
        .unwrap();
    assert_eq!(out.admissions[0].unwrap().children, [Some(h(5)), None]);
    assert_eq!(
        out.superseded
            .iter()
            .flatten()
            .map(|g| g.handle)
            .collect::<Vec<_>>(),
        [h(3), h(4)]
    );
    assert!(
        out.superseded.iter().flatten().all(|g| g
            .members
            .iter()
            .flatten()
            .all(|m| m.end_sample == 20))
    );
    assert!(
        state.slots[3]
            .unwrap()
            .group
            .members
            .iter()
            .flatten()
            .all(|m| m.weight == 1. && m.end_sample == 30)
    );
}

#[test]
fn capacity_rejects_eighth_birth_without_using_a_generation_and_evicts_oldest_dormant() {
    let mut state = populate(8);
    assert_eq!(state.next_generation, 9);
    let mut now = current(&[1, 2, 3, 4, 5, 6, 7, 8]);
    now[6] = None;
    let p = state.prepare(&config(), 20, &now).unwrap();
    let birth = proposal(&p, Kind::Birth, &[], &[8], &[]);
    let out = state
        .commit(
            p,
            [
                Some(0.1),
                Some(0.1),
                Some(0.1),
                Some(0.1),
                Some(0.1),
                Some(0.1),
                Some(0.),
            ],
            &[birth],
        )
        .unwrap();
    assert_eq!(out.admissions[0].unwrap().children, [Some(h(9)), None]);
    let retired = out.retired[0].unwrap();
    assert_eq!(retired.group.handle, h(8));
    assert_eq!(retired.reason, Retirement::Capacity);
    assert_eq!(retired.last_active_end, 10);
    assert_eq!(retired.inactive_samples, 10);
    assert_eq!(state.residual, h(1));
}

#[test]
fn split_reserves_both_children_before_superseding_parent_and_reports_capacity() {
    let mut state = populate(7);
    let mut now = current(&[1, 2, 3, 4, 5, 6, 7, 8]);
    now[7].as_mut().unwrap().point.frequency_log2 = Some(10.1);
    let p = state.prepare(&config(), 20, &now).unwrap();
    let split = proposal(&p, Kind::Split, &[2], &[1], &[8]);
    let out = state.commit(p, [Some(0.1); 7], &[split]).unwrap();
    assert_eq!(
        out.admissions[0].unwrap().rejection,
        Some(Rejection::Capacity)
    );
    assert_eq!(out.admissions[0].unwrap().key, split.key);
    assert!(out.superseded.iter().all(Option::is_none));
    assert!(out.retired.iter().all(Option::is_none));
    assert!(state.slots[0].unwrap().group.eligible);
    assert_eq!(state.next_generation, 9);
    now[6] = None;
    let p = state.prepare(&config(), 30, &now).unwrap();
    let split = proposal(&p, Kind::Split, &[2], &[1], &[8]);
    let out = state
        .commit(
            p,
            [
                Some(0.1),
                Some(0.1),
                Some(0.1),
                Some(0.1),
                Some(0.1),
                Some(0.1),
                Some(0.),
            ],
            &[split],
        )
        .unwrap();
    assert_eq!(
        out.admissions[0].unwrap().children,
        [Some(h(9)), Some(h(10))]
    );
    assert_eq!(
        out.retired
            .iter()
            .flatten()
            .map(|r| r.group.handle)
            .collect::<Vec<_>>(),
        [h(8), h(2)]
    );
    assert_eq!(out.superseded[0].unwrap().handle, h(2));
    assert_eq!(
        state
            .slots
            .iter()
            .flatten()
            .filter(|s| s.group.eligible)
            .count(),
        7
    );
}

#[test]
fn observed_retirement_variants_and_recovery_do_not_bridge_missing_time() {
    for duration in [1000, 2000, 4000] {
        let mut state = populate(1);
        state.retirement_samples = duration;
        let mut end = 10;
        for step in 1..=duration / 10 {
            if step == 2 {
                end += 10000;
                let p = state.prepare(&config(), end, &[None; 8]).unwrap();
                let out = state.commit(p, [None; 7], &[]).unwrap();
                assert!(out.retired.iter().all(Option::is_none));
                assert_eq!(state.slots[0].unwrap().inactive_samples, 10);
            }
            end += 10;
            let p = state.prepare(&config(), end, &[None; 8]).unwrap();
            let out = state.commit(p, [Some(0.); 7], &[]).unwrap();
            assert_eq!(out.retired[0].is_some(), step == duration / 10);
            if let Some(r) = out.retired[0] {
                assert_eq!(r.reason, Retirement::ObservedInactivity);
                assert_eq!(r.inactive_samples, duration);
            }
        }
        assert!(state.slots.iter().all(Option::is_none));
        let p = state.prepare(&config(), end + 10, &current(&[1])).unwrap();
        let recovered = proposal(&p, Kind::Birth, &[], &[1], &[]);
        let out = state.commit(p, [Some(0.); 7], &[recovered]).unwrap();
        assert_eq!(out.admissions[0].unwrap().children, [Some(h(3)), None]);
    }
}

#[test]
fn low_energy_boundary_activity_reset_and_superseded_retirement_are_distinct() {
    let mut state = populate(1);
    state.retirement_samples = 30;
    for (end, energy, count) in [(20, 1e-6, 10), (30, 1.00001e-6, 0), (40, 0., 10)] {
        let p = state.prepare(&config(), end, &current(&[1])).unwrap();
        state
            .commit(p, [Some(energy), None, None, None, None, None, None], &[])
            .unwrap();
        assert_eq!(state.slots[0].unwrap().inactive_samples, count);
    }
    let mut now = current(&[1, 2]);
    now[1].as_mut().unwrap().point.frequency_log2 = Some(10.1);
    let p = state.prepare(&config(), 50, &now).unwrap();
    let split = proposal(&p, Kind::Split, &[2], &[1], &[2]);
    state
        .commit(p, [Some(0.1), None, None, None, None, None, None], &[split])
        .unwrap();
    for end in [60, 70, 80] {
        let p = state.prepare(&config(), end, &now).unwrap();
        let out = state
            .commit(
                p,
                [Some(0.), Some(0.1), Some(0.1), None, None, None, None],
                &[],
            )
            .unwrap();
        assert_eq!(out.retired[0].is_some(), end == 80);
        if let Some(r) = out.retired[0] {
            assert!(!r.group.eligible);
        }
    }
}

#[test]
fn errors_leave_all_lifecycle_state_unchanged_and_retired_parent_rejects_admission() {
    let mut state = populate(1);
    let now = current(&[1]);
    let stale = state.prepare(&config(), 20, &now).unwrap();
    let p = state.prepare(&config(), 20, &now).unwrap();
    state
        .commit(p, [Some(0.1), None, None, None, None, None, None], &[])
        .unwrap();
    assert!(state.commit(stale, [None; 7], &[]).is_err());
    let before = state.slots[0].unwrap();
    let p = state.prepare(&config(), 30, &now).unwrap();
    assert!(
        state
            .commit(p, [Some(f64::NAN), None, None, None, None, None, None], &[])
            .is_err()
    );
    assert_eq!(state.last_end, 20);
    assert_eq!(state.slots[0].unwrap().group, before.group);
    let mut now = current(&[1, 2]);
    now[1].as_mut().unwrap().point.frequency_log2 = Some(10.1);
    state.next_generation = u64::MAX;
    let p = state.prepare(&config(), 30, &now).unwrap();
    let split = proposal(&p, Kind::Split, &[2], &[1], &[2]);
    assert!(
        state
            .commit(p, [Some(0.1), None, None, None, None, None, None], &[split])
            .is_err()
    );
    assert_eq!(state.next_generation, u64::MAX);
    assert_eq!(state.last_end, 20);
    assert_eq!(state.slots[0].unwrap().group, before.group);
    state.next_generation = 3;
    state.retirement_samples = 10;
    let p = state.prepare(&config(), 30, &now).unwrap();
    let split = proposal(&p, Kind::Split, &[2], &[1], &[2]);
    let out = state
        .commit(p, [Some(0.), None, None, None, None, None, None], &[split])
        .unwrap();
    assert_eq!(
        out.admissions[0].unwrap().rejection,
        Some(Rejection::ParentUnavailable)
    );
    assert_eq!(
        out.retired[0].unwrap().reason,
        Retirement::ObservedInactivity
    );
    assert_eq!(state.next_generation, 3);
}

#[test]
fn assignment_window_generator_and_lifecycle_share_one_immutable_hop() {
    use crate::temporal_cognition::grouping::{Envelope, Window};
    use crate::temporal_cognition::proposals::producer::Generator;
    let mut window = Window::new(0, 2, 0, 10, 80, 8, 0.9).unwrap();
    let mut state = Lifecycle::new(0, 2, 70, 10, 2000, 1e-6).unwrap();
    let mut generator = Generator::new(0, 2, 70, 10, 3).unwrap();
    let mut births = Vec::new();
    let mut splits = Vec::new();
    for step in 1..=20 {
        let mut now = current(&[1, 2]);
        now[1].as_mut().unwrap().point.frequency_log2 = Some(10.2);
        now[0].as_mut().unwrap().point.log_envelope = Some(-4. + step as f64 * 0.01);
        now[1].as_mut().unwrap().point.log_envelope =
            Some(-4. + step as f64 * if step <= 10 { 0.01 } else { -0.01 });
        window
            .push(
                step * 10,
                now.map(|t| {
                    t.map(|t| Envelope {
                        handle: t.handle,
                        log_envelope: t.point.log_envelope.unwrap(),
                    })
                }),
            )
            .unwrap();
        if step <= 7 {
            continue;
        }
        let p = state.prepare(&config(), step * 10, &now).unwrap();
        let active: Vec<_> = state
            .slots
            .iter()
            .flatten()
            .filter(|s| s.group.eligible)
            .map(|s| s.group.handle)
            .collect();
        let correlations = window.correlations(&[h(1), h(2)]).unwrap();
        let generated = generator
            .advance(true, &correlations, &p.assignment, &active)
            .unwrap();
        let accepted: Vec<_> = generated
            .proposals
            .accepted
            .iter()
            .flatten()
            .copied()
            .collect();
        let mut energy = [Some(0.); 7];
        for row in p.assignment.rows.iter().flatten() {
            let descriptor = now
                .iter()
                .flatten()
                .find(|t| t.handle == row.trajectory)
                .unwrap();
            let trajectory_energy = (2.0 * descriptor.point.log_envelope.unwrap()).exp2();
            for (i, e) in energy.iter_mut().enumerate() {
                *e.as_mut().unwrap() += row.weights[i] * trajectory_energy;
            }
        }
        let old_handles = p.assignment.group_handles;
        let out = state.commit(p, energy, &accepted).unwrap();
        assert_eq!(out.assignment.group_handles, old_handles);
        for admission in out.admissions.iter().flatten() {
            assert!(admission.rejection.is_none());
            match admission.key.kind {
                Kind::Birth => births.push(step * 10),
                Kind::Split => splits.push(step * 10),
                Kind::Merge => panic!("opposite envelopes must not merge"),
            }
            for child in admission.children.iter().flatten() {
                assert!(!old_handles.contains(&Some(*child)));
            }
        }
    }
    assert_eq!(births, [100]);
    assert_eq!(splits, [140]);
    assert_eq!(
        state
            .slots
            .iter()
            .flatten()
            .filter(|s| s.group.eligible)
            .map(|s| s.group.handle)
            .collect::<Vec<_>>(),
        [h(3), h(4)]
    );
}

#[test]
fn independent_slot_and_observed_clock_reference_matches_every_transition() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/temporal_cognition/lifecycle.json"
    ))
    .unwrap();
    let mut totals = [0usize; 4];
    for sequence in fixture["sequences"].as_array().unwrap() {
        let mut state = Lifecycle::new(
            0,
            2,
            0,
            10,
            sequence["retirement_samples"].as_u64().unwrap(),
            1e-6,
        )
        .unwrap();
        for step in sequence["steps"].as_array().unwrap() {
            let energy: [Option<f64>; 7] = std::array::from_fn(|i| step["energy"][i].as_f64());
            let mut now = [None; 8];
            for (i, slot) in state.slots.iter().enumerate() {
                if let Some(s) = slot
                    && energy[i].is_some_and(|e| e > 1e-6)
                {
                    now[i] = Some(s.group.members[0].unwrap().trajectory);
                }
            }
            let birth = step["birth"].as_u64();
            if let Some(id) = birth {
                now[7] = current(&[id])[0];
            }
            let p = state
                .prepare(&config(), step["end"].as_u64().unwrap(), &now)
                .unwrap();
            let candidate = birth.map(|id| proposal(&p, Kind::Birth, &[], &[id], &[]));
            let out = state
                .commit(p, energy, &candidate.into_iter().collect::<Vec<_>>())
                .unwrap();
            let expected = &step["expected"];
            assert_eq!(
                state.next_generation,
                expected["next_generation"].as_u64().unwrap()
            );
            for (i, slot) in state.slots.iter().enumerate() {
                let reference = &expected["slots"][i];
                assert_eq!(slot.is_some(), !reference.is_null());
                if let Some(s) = slot {
                    assert_eq!(
                        s.group.handle.generation,
                        reference["handle"].as_u64().unwrap()
                    );
                    assert_eq!(
                        s.group.members[0].unwrap().trajectory.handle.generation,
                        reference["trajectory"].as_u64().unwrap()
                    );
                    assert_eq!(s.dormant, reference["dormant"].as_bool().unwrap());
                    assert_eq!(
                        s.last_active_end,
                        reference["last_active"].as_u64().unwrap()
                    );
                    assert_eq!(s.inactive_samples, reference["inactive"].as_u64().unwrap());
                }
            }
            let admitted = out.admissions[0].and_then(|a| a.children[0]);
            assert_eq!(
                admitted.map(|h| h.generation),
                expected["admitted"].as_u64()
            );
            assert_eq!(
                out.admissions[0].and_then(|a| a.rejection).is_some(),
                expected["rejection"] == "capacity"
            );
            let retired: Vec<_> = out.retired.iter().flatten().collect();
            assert_eq!(retired.len(), expected["retired"].as_array().unwrap().len());
            for (actual, reference) in retired.iter().zip(expected["retired"].as_array().unwrap()) {
                assert_eq!(
                    actual.group.handle.generation,
                    reference["handle"].as_u64().unwrap()
                );
                assert_eq!(
                    actual.reason,
                    if reference["reason"] == "capacity" {
                        Retirement::Capacity
                    } else {
                        Retirement::ObservedInactivity
                    }
                );
                assert_eq!(
                    actual.last_active_end,
                    reference["last_active"].as_u64().unwrap()
                );
                assert_eq!(
                    actual.inactive_samples,
                    reference["inactive"].as_u64().unwrap()
                );
            }
            totals[0] += 1;
            totals[1] += usize::from(admitted.is_some());
            totals[2] += usize::from(out.admissions[0].and_then(|a| a.rejection).is_some());
            totals[3] += retired.len();
        }
    }
    assert_eq!(totals[0], 768);
    assert!(totals[1] > 100 && totals[2] > 0 && totals[3] > 100);
    println!(
        "lifecycle_reference endpoints={} births={} capacity_rejections={} retirements={}",
        totals[0], totals[1], totals[2], totals[3]
    );
}

#[test]
#[ignore = "release preparation and group commit cost; excludes proposal generation and all O04"]
fn group_lifecycle_cost_probe() {
    use std::{hint::black_box, time::Instant};
    let mut state = Lifecycle::new(0, 2, 0, 512, 96000, 1e-6).unwrap();
    let mut config = config();
    config.hop = 512;
    config.sample_rate = 48000;
    let mut now = current(&[1, 2, 3, 4, 5, 6, 7, 8]);
    for (i, t) in now.iter_mut().flatten().enumerate() {
        t.point.frequency_log2 = Some(10. + i as f64 * 0.01);
        t.point.log_envelope = Some(-4. + i as f64 * 0.005);
    }
    let p = state.prepare(&config, 512, &now).unwrap();
    let births: Vec<_> = (1..=7)
        .map(|i| proposal(&p, Kind::Birth, &[], &[i], &[]))
        .collect();
    state.commit(p, [Some(0.); 7], &births).unwrap();
    let mut times = Vec::with_capacity(6000);
    let mut evaluations = 0;
    for step in 2..=6101 {
        for (i, t) in now.iter_mut().flatten().enumerate() {
            t.point.frequency_log2 = Some(10. + i as f64 * 0.01 + (step % 11) as f64 * 0.0001);
            t.point.log_envelope = Some(-4. + i as f64 * 0.005 + (step % 7) as f64 * 0.001);
        }
        let start = Instant::now();
        let p = state
            .prepare(black_box(&config), step * 512, black_box(&now))
            .unwrap();
        let count = p.assignment.distance_evaluations;
        let out = black_box(state.commit(p, black_box([Some(0.001); 7]), &[]).unwrap());
        let elapsed = start.elapsed().as_secs_f64() * 1e6;
        if step > 101 {
            times.push(elapsed);
            evaluations += count;
            assert_eq!(count, 896);
            assert!(out.retired.iter().all(Option::is_none));
        }
    }
    times.sort_by(f64::total_cmp);
    println!(
        "lifecycle_cost {}",
        serde_json::json!({"calls":times.len(),"member_distance_evaluations":evaluations,"median_us":times[3000],"p99_us":times[5939],"max_us":times[5999],"lifecycle_bytes":std::mem::size_of::<Lifecycle>(),"prepared_bytes":std::mem::size_of::<Prepared>(),"update_bytes":std::mem::size_of::<Update>(),"full_O04":false,"scope":"full7x8x8x2 member assignment, reference preparation and no-topology commit; proposal generation, per-bin energy construction and context/beam handling excluded"})
    );

    let mut state = populate(7);
    let original = state.slots;
    let mut now = current(&[1, 2, 3, 4, 5, 6]);
    for i in 0..3 {
        now[i + 3].as_mut().unwrap().point.frequency_log2 = Some(10. * (i + 1) as f64 + 0.002);
    }
    let mut times = Vec::with_capacity(6000);
    for iteration in 0..6100 {
        state.slots = original;
        state.last_end = 10;
        state.next_generation = 9;
        let p = state.prepare(&self::config(), 20, &now).unwrap();
        let splits: Vec<_> = (0..3)
            .map(|i| proposal(&p, Kind::Split, &[2 + i], &[1 + i], &[4 + i]))
            .collect();
        let start = Instant::now();
        let out = black_box(
            state
                .commit(
                    p,
                    [
                        Some(0.1),
                        Some(0.1),
                        Some(0.1),
                        Some(0.),
                        Some(0.),
                        Some(0.),
                        Some(0.),
                    ],
                    black_box(&splits),
                )
                .unwrap(),
        );
        let elapsed = start.elapsed().as_secs_f64() * 1e6;
        assert_eq!(
            out.admissions
                .iter()
                .flatten()
                .filter(|a| a.rejection.is_none())
                .count(),
            3
        );
        assert_eq!(out.retired.iter().flatten().count(), 6);
        assert_eq!(out.superseded.iter().flatten().count(), 3);
        if iteration >= 100 {
            times.push(elapsed);
        }
    }
    times.sort_by(f64::total_cmp);
    println!(
        "lifecycle_topology_cost {}",
        serde_json::json!({"calls":times.len(),"median_us":times[3000],"p99_us":times[5939],"max_us":times[5999],"splits_per_call":3,"fresh_children_per_call":6,"capacity_retirements_per_call":6,"full_O04":false,"scope":"commit-only replay of the same three disjoint splits; fixture reset, preparation and candidate creation outside timer; no beam/context storage"})
    );
}
