use super::*;

fn handle(generation: u64) -> Handle {
    Handle {
        bus: 0,
        epoch: 2,
        generation,
    }
}

fn key(kind: Kind, parents: &[u64], left: &[u64], right: &[u64]) -> Key {
    Key::new(
        kind,
        &parents.iter().copied().map(handle).collect::<Vec<_>>(),
        &left.iter().copied().map(handle).collect::<Vec<_>>(),
        &right.iter().copied().map(handle).collect::<Vec<_>>(),
    )
    .unwrap()
}

fn assignment(end: u64, residual: f64) -> Assignment {
    Assignment {
        end_sample: end,
        group_handles: std::array::from_fn(|i| Some(handle(101 + i as u64))),
        rows: std::array::from_fn(|i| {
            Some(Row {
                trajectory: handle(1 + i as u64),
                weights: std::array::from_fn(|j| {
                    if j == 7 {
                        residual
                    } else if j == i % 7 {
                        1.0 - residual
                    } else {
                        0.0
                    }
                }),
                matched_members: [None; 7],
            })
        }),
        distance_evaluations: 0,
    }
}

fn candidate(key: Key, end: u64) -> Candidate {
    Candidate::measure(
        key,
        &assignment(end, 0.5),
        Some(if key.kind == Kind::Split { 0.2 } else { 0.8 }),
    )
    .unwrap()
    .unwrap()
}

#[test]
fn canonical_keys_preserve_attached_parent_sets_and_generation_identity() {
    let merge = key(Kind::Merge, &[102, 101], &[4, 3], &[2, 1]);
    assert_eq!(merge, key(Kind::Merge, &[101, 102], &[1, 2], &[3, 4]));
    assert_ne!(merge, key(Kind::Merge, &[101, 102], &[3, 4], &[1, 2]));
    assert_eq!(
        key(Kind::Split, &[101], &[3, 2], &[1]),
        key(Kind::Split, &[101], &[1], &[2, 3])
    );
    assert_ne!(
        key(Kind::Birth, &[], &[1], &[]),
        key(Kind::Birth, &[], &[9], &[])
    );
    assert!(
        Key::new(
            Kind::Merge,
            &[handle(101), handle(101)],
            &[handle(1)],
            &[handle(2)]
        )
        .is_err()
    );
    assert!(Key::new(Kind::Split, &[handle(101)], &[handle(1)], &[handle(1)]).is_err());
    assert!(Key::new(Kind::Birth, &[], &[], &[]).is_err());
    let mut foreign = handle(2);
    foreign.epoch += 1;
    assert!(Key::new(Kind::Birth, &[], &[handle(1), foreign], &[]).is_err());
}

#[test]
fn former_parent_requires_unique_resolved_maximum_including_residual() {
    let mut a = assignment(10, 0.25);
    let row = a.rows[0].as_mut().unwrap();
    assert_eq!(former_parent(row, &a.group_handles), Some(handle(101)));
    for weights in [
        [0.5, 0., 0., 0., 0., 0., 0., 0.5],
        [0.4, 0.4, 0., 0., 0., 0., 0., 0.2],
        [0.25, 0., 0., 0., 0., 0., 0., 0.75],
        [f64::NAN, 0., 0., 0., 0., 0., 0., 0.],
    ] {
        row.weights = weights;
        assert_eq!(former_parent(row, &a.group_handles), None);
    }
}

#[test]
fn birth_gate_is_separate_from_split_merge_and_support_uses_each_distinct_member() {
    let birth = key(Kind::Birth, &[], &[1, 2], &[]);
    for residual in [0.499, 0.5, 0.501] {
        let c = Candidate::measure(birth, &assignment(10, residual), None).unwrap();
        assert_eq!(c.is_some(), residual >= 0.5);
        if let Some(c) = c {
            assert!((c.score - residual).abs() < 1e-14);
            assert!((c.support - 2.0 * residual).abs() < 1e-14);
        }
    }
    let split = key(Kind::Split, &[101], &[1], &[2]);
    let merge = key(Kind::Merge, &[101, 102], &[1], &[2]);
    for (k, edge, pass) in [
        (split, 0.2, true),
        (split, 0.200001, false),
        (merge, 0.8, true),
        (merge, 0.799999, false),
    ] {
        let c = Candidate::measure(k, &assignment(10, 0.01), Some(edge)).unwrap();
        assert_eq!(c.is_some(), pass);
        if let Some(c) = c {
            assert!((c.support - 1.98).abs() < 1e-14);
        }
        assert!(
            Candidate::measure(k, &assignment(10, 0.01), None)
                .unwrap()
                .is_none()
        );
    }
    let mut a = assignment(10, 0.5);
    a.rows.rotate_left(3);
    assert_eq!(
        Candidate::measure(birth, &a, None)
            .unwrap()
            .unwrap()
            .support,
        1.0
    );
    a.rows
        .iter_mut()
        .find(|r| r.is_some_and(|r| r.trajectory == handle(1)))
        .unwrap()
        .take();
    assert!(Candidate::measure(birth, &a, None).unwrap().is_none());
    a.group_handles[0] = None;
    assert!(Candidate::measure(split, &a, Some(0.1)).unwrap().is_none());
}

#[test]
fn changing_current_weights_does_not_rewrite_an_immutable_former_parent_key() {
    let old = assignment(10, 0.1);
    let parent = former_parent(&old.rows[0].unwrap(), &old.group_handles).unwrap();
    let k = Key::new(Kind::Split, &[parent], &[handle(1)], &[handle(2)]).unwrap();
    let mut tracker = Tracker::new(0, 2, 10, 10, 3).unwrap();
    for end in [20, 30, 40] {
        let a = assignment(end, 0.9);
        assert!(former_parent(&a.rows[0].unwrap(), &a.group_handles).is_none());
        let c = Candidate::measure(k, &a, Some(0.1)).unwrap().unwrap();
        let out = tracker.advance(end, true, &[c]).unwrap();
        assert_eq!(out.accepted[0].is_some(), end == 40);
        if end == 40 {
            assert_eq!(out.accepted[0].unwrap().key.parents[0], Some(parent));
        }
    }
}

#[test]
fn missing_conditions_skipped_hops_and_identity_churn_reset_persistence() {
    let k = key(Kind::Birth, &[], &[1], &[]);
    for disturbance in 0..5 {
        let mut tracker = Tracker::new(0, 2, 0, 10, 3).unwrap();
        tracker.advance(10, true, &[candidate(k, 10)]).unwrap();
        tracker.advance(20, true, &[candidate(k, 20)]).unwrap();
        let (end, changed, observed) = match disturbance {
            0 => (30, None, true),
            1 => (30, Some(k), false),
            2 => (40, Some(k), true),
            3 => (30, Some(key(Kind::Birth, &[], &[2], &[])), true),
            _ => (30, Some(key(Kind::Birth, &[], &[1, 2], &[])), true),
        };
        let input: Vec<_> = changed.map(|k| candidate(k, end)).into_iter().collect();
        assert!(tracker.advance(end, observed, &input).unwrap().accepted[0].is_none());
        let out = tracker
            .advance(end + 10, true, &[candidate(k, end + 10)])
            .unwrap();
        assert!(out.accepted[0].is_none());
        assert_eq!(
            tracker.pending[0].unwrap().hops,
            if disturbance == 2 { 2 } else { 1 }
        );
    }
    let a = key(Kind::Split, &[101], &[1], &[2]);
    let b = key(Kind::Split, &[102], &[1], &[2]);
    let mut tracker = Tracker::new(0, 2, 0, 10, 3).unwrap();
    for (end, k) in [(10, a), (20, a), (30, b)] {
        assert!(
            tracker
                .advance(end, true, &[candidate(k, end)])
                .unwrap()
                .accepted[0]
                .is_none()
        );
    }
    assert_eq!(tracker.pending[0].unwrap().hops, 1);
}

#[test]
fn ranking_resolves_member_and_parent_conflicts_and_resets_rejected_keys() {
    let merge = key(Kind::Merge, &[101, 102], &[1], &[2]);
    let split = key(Kind::Split, &[101], &[3], &[4]);
    let birth = key(Kind::Birth, &[], &[1, 2], &[]);
    for order in [
        [merge, split, birth],
        [birth, split, merge],
        [split, merge, birth],
    ] {
        let mut tracker = Tracker::new(0, 2, 0, 10, 3).unwrap();
        for end in [10, 20, 30] {
            let input: Vec<_> = order.map(|k| candidate(k, end)).into();
            let out = tracker.advance(end, true, &input).unwrap();
            if end < 30 {
                assert_eq!(out.pending, 3);
            } else {
                assert_eq!(out.accepted[0].unwrap().key, merge);
                assert_eq!(out.conflicts, 2);
                assert_eq!(out.pending, 0);
            }
        }
        let out = tracker.advance(40, true, &[candidate(split, 40)]).unwrap();
        assert!(out.accepted[0].is_none());
        assert_eq!(tracker.pending[0].unwrap().hops, 1);
    }
    let mut a = assignment(10, 0.5);
    a.rows[2].as_mut().unwrap().weights = [0., 0., 0., 0., 0., 0., 0., 1.];
    a.rows[3].as_mut().unwrap().weights = [0., 0., 0., 0., 0., 0., 0., 1.];
    let candidates = [
        Candidate::measure(merge, &a, Some(0.9)).unwrap().unwrap(),
        Candidate::measure(split, &a, Some(0.1)).unwrap().unwrap(),
    ];
    let out = Tracker::new(0, 2, 0, 10, 1)
        .unwrap()
        .advance(10, true, &candidates)
        .unwrap();
    assert_eq!(out.accepted[0].unwrap().key, split);
    assert_eq!(out.conflicts, 1);
}

fn saturated_keys() -> Vec<Key> {
    let mut keys: Vec<_> = (1..=8).map(|i| key(Kind::Birth, &[], &[i], &[])).collect();
    for a in 1..=8 {
        for b in a + 1..=8 {
            keys.push(key(Kind::Split, &[101], &[a], &[b]));
        }
    }
    for a in 1..=7 {
        for b in a + 1..=7 {
            keys.push(key(Kind::Merge, &[100 + a, 100 + b], &[a], &[b]));
        }
    }
    keys
}

#[test]
fn capacities_and_invalid_updates_are_atomic_and_persistence_is_configurable() {
    assert_eq!(saturated_keys().len(), 57);
    for persistence in [1, 3, 6] {
        let mut tracker = Tracker::new(0, 2, 0, 10, persistence).unwrap();
        for step in 1..=persistence {
            let end = u64::from(step) * 10;
            let input: Vec<_> = saturated_keys()
                .into_iter()
                .map(|k| candidate(k, end))
                .collect();
            let out = tracker.advance(end, true, &input).unwrap();
            assert_eq!(out.accepted[0].is_some(), step == persistence);
            if step < persistence {
                assert_eq!(out.pending, 57);
            } else {
                assert_eq!(out.accepted.iter().flatten().count(), 5);
                assert_eq!(out.conflicts, 52);
            }
        }
    }
    let k = key(Kind::Birth, &[], &[1], &[]);
    let mut tracker = Tracker::new(0, 2, 0, 10, 3).unwrap();
    tracker.advance(10, true, &[candidate(k, 10)]).unwrap();
    assert!(tracker.advance(20, true, &[candidate(k, 10)]).is_err());
    assert!(
        tracker
            .advance(20, true, &[candidate(k, 20), candidate(k, 20)])
            .is_err()
    );
    assert_eq!(tracker.last_end, 10);
    assert_eq!(tracker.pending[0].unwrap().hops, 1);
    let too_many: Vec<_> = (1..=9)
        .map(|i| Candidate {
            key: key(Kind::Birth, &[], &[i], &[]),
            end_sample: 20,
            support: 1.,
            score: 1.,
        })
        .collect();
    assert!(tracker.advance(20, true, &too_many).is_err());
    let mut a = assignment(20, 0.5);
    a.rows[0].as_mut().unwrap().weights[7] = f64::NAN;
    assert!(Candidate::measure(k, &a, None).is_err());
    assert_eq!(tracker.last_end, 10);
}

#[test]
fn exact_rational_dictionary_reference_matches_all_supplied_key_sequences() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/temporal_cognition/proposals.json"
    ))
    .unwrap();
    let list = |v: &serde_json::Value| {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect::<Vec<_>>()
    };
    let keys: Vec<_> = fixture["keys"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| {
            key(
                match k["kind"].as_str().unwrap() {
                    "merge" => Kind::Merge,
                    "split" => Kind::Split,
                    "birth" => Kind::Birth,
                    _ => unreachable!(),
                },
                &list(&k["parents"]),
                &list(&k["left"]),
                &list(&k["right"]),
            )
        })
        .collect();
    let mut total_accepted = 0;
    let mut total_conflicts = 0;
    for sequence in fixture["sequences"].as_array().unwrap() {
        let mut tracker =
            Tracker::new(0, 2, 0, 10, sequence["persistence"].as_u64().unwrap() as u8).unwrap();
        for step in sequence["steps"].as_array().unwrap() {
            let end = step["end"].as_u64().unwrap();
            let mut a = assignment(end, 0.5);
            for (i, row) in a.rows.iter_mut().enumerate() {
                let r = step["residual_eighths"][i].as_f64().unwrap() / 8.0;
                row.as_mut().unwrap().weights = std::array::from_fn(|j| {
                    if j == 7 {
                        r
                    } else if j == i % 7 {
                        1.0 - r
                    } else {
                        0.0
                    }
                });
            }
            let mut candidates = Vec::new();
            for (offset, id) in list(&step["seeds"]).into_iter().enumerate() {
                let c = Candidate::measure(
                    keys[id as usize],
                    &a,
                    step["correlations"][offset].as_f64(),
                )
                .unwrap();
                let expected = &step["qualified"][id.to_string()];
                assert_eq!(c.is_some(), !expected.is_null());
                if let Some(c) = c {
                    assert_eq!(c.support, expected["support"].as_f64().unwrap());
                    assert!((c.score - expected["score"].as_f64().unwrap()).abs() < 1e-12);
                    candidates.push(c);
                }
            }
            let out = tracker
                .advance(end, step["observed"].as_bool().unwrap(), &candidates)
                .unwrap();
            let actual: Vec<_> = out
                .accepted
                .iter()
                .flatten()
                .map(|c| keys.iter().position(|k| *k == c.key).unwrap() as u64)
                .collect();
            assert_eq!(actual, list(&step["expected"]["accepted"]));
            assert_eq!(
                out.conflicts as u64,
                step["expected"]["conflicts"].as_u64().unwrap()
            );
            assert_eq!(
                out.pending,
                step["expected"]["pending"].as_object().unwrap().len()
            );
            for entry in tracker.pending.iter().flatten() {
                let id = keys.iter().position(|k| *k == entry.candidate.key).unwrap();
                assert_eq!(
                    entry.hops as u64,
                    step["expected"]["pending"][id.to_string()]
                        .as_u64()
                        .unwrap()
                );
            }
            total_accepted += actual.len();
            total_conflicts += out.conflicts;
        }
    }
    assert!(total_accepted > 100 && total_conflicts > 100);
    println!(
        "proposal_reference accepted={total_accepted} conflicts={total_conflicts} endpoints=576"
    );
}

#[test]
#[ignore = "release bounded proposal reducer cost, not candidate generation or full O04"]
fn proposal_persistence_cost_probe() {
    use std::{hint::black_box, time::Instant};
    let keys = saturated_keys();
    let mut input: Vec<_> = keys.iter().map(|&k| candidate(k, 512)).collect();
    let mut tracker = Tracker::new(0, 2, 0, 512, 3).unwrap();
    let mut times = Vec::with_capacity(6000);
    let mut accepted = 0;
    let mut conflicts = 0;
    for step in 1..=6100 {
        for c in &mut input {
            c.end_sample = step * 512;
        }
        input.rotate_left(1);
        let start = Instant::now();
        let out = black_box(
            tracker
                .advance(step * 512, true, black_box(&input))
                .unwrap(),
        );
        let elapsed = start.elapsed().as_secs_f64() * 1e6;
        if step > 100 {
            times.push(elapsed);
            accepted += out.accepted.iter().flatten().count();
            conflicts += out.conflicts;
        }
    }
    times.sort_by(f64::total_cmp);
    println!(
        "proposal_cost {}",
        serde_json::json!({"calls":6000,"seeds":57,"median_us":times[3000],"p99_us":times[5939],"max_us":times[5999],"accepted":accepted,"conflicts":conflicts,"tracker_bytes":std::mem::size_of::<Tracker>(),"update_bytes":std::mem::size_of::<Update>(),"candidate_bytes":std::mem::size_of::<Candidate>(),"full_O04":false,"scope":"capacity-bound supplied keys, excludes candidate generation and acoustic feasibility"})
    );
}
