use super::*;
use crate::temporal_cognition::group::Row;

fn h(generation: u64) -> Handle {
    Handle {
        bus: 0,
        epoch: 2,
        generation,
    }
}

fn frame(
    end: u64,
    ids: &[u64],
    parent_ids: &[Option<u64>],
    edges: impl Fn(usize, usize) -> Option<f64>,
) -> (Correlations, Assignment) {
    let n = ids.len();
    let correlations = Correlations {
        handles: std::array::from_fn(|i| ids.get(i).copied().map(h)),
        count: n,
        values: std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                (i < n && j < n && i != j)
                    .then(|| edges(i.min(j), i.max(j)))
                    .flatten()
            })
        }),
        paired_samples: [[250; 8]; 8],
        paired_hops: [[25; 8]; 8],
        window_start: end.saturating_sub(250),
        window_end: end,
    };
    let assignment = Assignment {
        end_sample: end,
        group_handles: std::array::from_fn(|i| Some(h(101 + i as u64))),
        rows: std::array::from_fn(|i| {
            ids.get(i).map(|id| Row {
                trajectory: h(*id),
                weights: std::array::from_fn(|j| {
                    if let Some(parent) = parent_ids[i] {
                        if j == 7 {
                            0.125
                        } else if 101 + j as u64 == parent {
                            0.875
                        } else {
                            0.0
                        }
                    } else if j == 7 {
                        1.0
                    } else {
                        0.0
                    }
                }),
                matched_members: [None; 7],
            })
        }),
        distance_evaluations: 0,
    };
    (correlations, assignment)
}

fn eligible() -> [Handle; 7] {
    std::array::from_fn(|i| h(101 + i as u64))
}

fn keys(out: &Output, kind: Kind) -> Vec<Key> {
    out.candidates
        .iter()
        .flatten()
        .filter(|c| c.key.kind == kind)
        .map(|c| c.key)
        .collect()
}

#[test]
fn new_parent_uses_previous_hop_and_current_weights_cannot_replace_a_live_key() {
    let mut g = Generator::new(0, 2, 0, 10, 3).unwrap();
    for step in 1..=4 {
        let parents = if step == 1 {
            [Some(101), Some(102)]
        } else {
            [Some(103), Some(104)]
        };
        let (c, a) = frame(step * 10, &[1, 2], &parents, |_, _| Some(0.9));
        let out = g.advance(true, &c, &a, &eligible()).unwrap();
        assert_eq!(out.bundles.count, 1);
        let merges = keys(&out, Kind::Merge);
        if step == 1 {
            assert!(merges.is_empty());
        } else {
            assert!(
                merges
                    .iter()
                    .any(|k| k.parents == [Some(h(101)), Some(h(102))])
            );
        }
        if step == 4 {
            let accepted: Vec<_> = out.proposals.accepted.iter().flatten().collect();
            assert_eq!(accepted.len(), 1);
            assert_eq!(accepted[0].key.parents, [Some(h(101)), Some(h(102))]);
        }
    }
}

#[test]
fn split_requires_two_exact_final_bundles_and_all_cross_pairs_known_and_low() {
    for bad in [None, Some(0.200001)] {
        let mut g = Generator::new(0, 2, 0, 10, 3).unwrap();
        let parent = [Some(101); 4];
        for step in 1..=5 {
            let (c, a) = frame(step * 10, &[1, 2, 3, 4], &parent, |i, j| {
                if (i < 2) == (j < 2) {
                    Some(0.9)
                } else if step == 3 && i == 1 && j == 3 {
                    bad
                } else {
                    Some(0.2)
                }
            });
            let out = g.advance(true, &c, &a, &eligible()).unwrap();
            assert_eq!(out.bundles.count, 2);
            assert_eq!(
                keys(&out, Kind::Split).len(),
                usize::from(step != 1 && step != 3)
            );
            assert!(
                out.proposals
                    .accepted
                    .iter()
                    .flatten()
                    .all(|c| c.key.kind != Kind::Split)
            );
            if step == 5 {
                assert_eq!(
                    g.tracker
                        .pending
                        .iter()
                        .flatten()
                        .find(|e| e.candidate.key.kind == Kind::Split)
                        .unwrap()
                        .hops,
                    2
                );
            }
        }
    }
}

#[test]
fn split_frozen_parent_survives_later_ties_but_not_parent_generation_loss() {
    for remove in [false, true] {
        let mut g = Generator::new(0, 2, 0, 10, 3).unwrap();
        for step in 1..=4 {
            let parents = if step == 1 { [Some(101); 2] } else { [None; 2] };
            let (c, a) = frame(step * 10, &[1, 2], &parents, |_, _| Some(0.1));
            let active = eligible();
            let active = if remove && step >= 3 {
                &active[1..]
            } else {
                &active[..]
            };
            let out = g.advance(true, &c, &a, active).unwrap();
            if step == 4 {
                assert_eq!(
                    out.proposals
                        .accepted
                        .iter()
                        .flatten()
                        .any(|c| c.key.kind == Kind::Split),
                    !remove
                );
                if remove {
                    assert!(keys(&out, Kind::Split).is_empty());
                }
            }
        }
    }
}

#[test]
fn merge_pair_selection_is_global_score_then_handle_order_and_live_key_first() {
    for tie in [false, true] {
        for reverse in [false, true] {
            let mut g = Generator::new(0, 2, 0, 10, 3).unwrap();
            for step in 1..=4 {
                let ids = if reverse { [4, 3, 2, 1] } else { [1, 2, 3, 4] };
                let parents = ids.map(|i| Some(if i % 2 == 1 { 101 } else { 102 }));
                let (c, mut a) = frame(step * 10, &ids, &parents, |i, j| {
                    let same = (ids[i] <= 2) == (ids[j] <= 2);
                    if !same {
                        Some(0.1)
                    } else if step >= 3 {
                        Some(if ids[i] <= 2 { 0.99 } else { 0.85 })
                    } else {
                        Some(if tie || ids[i] <= 2 { 0.85 } else { 0.9 })
                    }
                });
                a.rows.rotate_left(1);
                let out = g.advance(true, &c, &a, &eligible()).unwrap();
                let merges = keys(&out, Kind::Merge);
                if step > 1 {
                    assert_eq!(merges.len(), 1);
                    let expected = if tie {
                        [Some(h(1)), Some(h(2))]
                    } else {
                        [Some(h(3)), Some(h(4))]
                    };
                    assert_eq!(merges[0].members[..2], expected);
                }
            }
        }
    }
}

#[test]
fn unrelated_member_can_join_but_new_member_of_frozen_parent_restarts_the_pair_key() {
    for added_parent in [101, 103] {
        let mut g = Generator::new(0, 2, 0, 10, 3).unwrap();
        for step in 1..=4 {
            let ids: &[u64] = if step < 3 { &[1, 2] } else { &[1, 2, 3] };
            let p = [Some(101), Some(102), Some(added_parent)];
            // The new trajectory is already associated in the preceding hop, before joining the bundle.
            let (mut c, a) = frame(step * 10, &[1, 2, 3], &p, |i, j| {
                if j == 2 && step < 3 {
                    Some(0.1)
                } else {
                    let _ = i;
                    Some(0.9)
                }
            });
            if step == 1 {
                c.handles[2] = None;
                c.count = 2;
            }
            let out = g.advance(true, &c, &a, &eligible()).unwrap();
            let pair = keys(&out, Kind::Merge)
                .into_iter()
                .find(|k| k.parents == [Some(h(101)), Some(h(102))]);
            if step == 4 {
                let pair = pair.unwrap();
                assert_eq!(
                    pair.members.iter().flatten().count(),
                    if added_parent == 101 { ids.len() } else { 2 }
                );
                assert_eq!(
                    out.proposals
                        .accepted
                        .iter()
                        .flatten()
                        .any(|c| c.key == pair),
                    added_parent == 103
                );
                if added_parent == 101 {
                    assert_eq!(
                        g.tracker
                            .pending
                            .iter()
                            .flatten()
                            .find(|e| e.candidate.key == pair)
                            .unwrap()
                            .hops,
                        2
                    );
                }
            }
        }
    }
}

#[test]
fn observed_gaps_and_absent_frames_reset_parent_snapshot_and_persistence() {
    for skip in [false, true] {
        let mut g = Generator::new(0, 2, 0, 10, 3).unwrap();
        for end in [10, 20] {
            let (c, a) = frame(end, &[1, 2], &[Some(101), Some(102)], |_, _| Some(0.9));
            g.advance(true, &c, &a, &eligible()).unwrap();
        }
        if !skip {
            let (c, a) = frame(30, &[], &[], |_, _| None);
            let out = g.advance(false, &c, &a, &eligible()).unwrap();
            assert!(out.candidates.iter().all(Option::is_none));
        }
        let (c, a) = frame(40, &[1, 2], &[Some(101), Some(102)], |_, _| Some(0.9));
        assert!(keys(&g.advance(true, &c, &a, &eligible()).unwrap(), Kind::Merge).is_empty());
        let (c, a) = frame(50, &[1, 2], &[Some(101), Some(102)], |_, _| Some(0.9));
        let out = g.advance(true, &c, &a, &eligible()).unwrap();
        assert_eq!(keys(&out, Kind::Merge).len(), 1);
        assert!(out.proposals.accepted[0].is_none());
    }
}

#[test]
fn largest_structural_inventories_obey_bounds_and_input_errors_preserve_state() {
    for merge in [false, true] {
        let mut g = Generator::new(0, 2, 0, 10, 6).unwrap();
        for step in 1..=3 {
            let parents: [Option<u64>; 8] =
                std::array::from_fn(|i| Some(if merge { 101 + (i % 7) as u64 } else { 101 }));
            let (c, mut a) = frame(step * 10, &[1, 2, 3, 4, 5, 6, 7, 8], &parents, |_, _| {
                Some(if merge { 0.9 } else { 0.1 })
            });
            if step > 1 {
                for r in a.rows.iter_mut().flatten() {
                    r.weights = [0., 0., 0., 0., 0., 0., 0., 1.];
                }
            }
            let out = g.advance(true, &c, &a, &eligible()).unwrap();
            if step > 1 {
                assert_eq!(keys(&out, Kind::Birth).len(), if merge { 1 } else { 8 });
                assert_eq!(
                    keys(&out, if merge { Kind::Merge } else { Kind::Split }).len(),
                    if merge { 21 } else { 28 }
                );
            }
        }
        let (mut c, a) = frame(40, &[1, 2], &[Some(101), Some(102)], |_, _| Some(0.9));
        c.values[0][1] = None;
        assert!(g.advance(true, &c, &a, &eligible()).is_err());
        assert_eq!(g.tracker.last_end, 30);
        assert_eq!(g.previous.unwrap().end_sample, 30);
        assert_eq!(
            g.tracker
                .pending
                .iter()
                .flatten()
                .filter(|e| e.hops == 2)
                .count(),
            if merge { 22 } else { 36 }
        );
    }
}

#[test]
fn physical_envelope_window_drives_split_only_after_paired_support_and_three_hops() {
    use crate::temporal_cognition::grouping::{Envelope, Window};
    let mut w = Window::new(0, 2, 0, 10, 100, 8, 0.9).unwrap();
    let mut g = Generator::new(0, 2, 0, 10, 3).unwrap();
    for step in 1..=10 {
        let mut values = [None; 8];
        values[0] = Some(Envelope {
            handle: h(1),
            log_envelope: step as f64,
        });
        values[1] = Some(Envelope {
            handle: h(2),
            log_envelope: -(step as f64),
        });
        w.push(step * 10, values).unwrap();
        let c = w.correlations(&[h(1), h(2)]).unwrap();
        let (_, a) = frame(step * 10, &[1, 2], &[Some(101); 2], |_, _| None);
        let out = g.advance(true, &c, &a, &eligible()).unwrap();
        assert_eq!(keys(&out, Kind::Split).len(), usize::from(step >= 8));
        assert_eq!(out.proposals.accepted[0].is_some(), step == 10);
        assert_eq!(out.continued, usize::from(step >= 9));
        if step >= 8 {
            assert!(out.cross_pair_reads >= 1);
        }
    }
}

fn decode_key(value: &serde_json::Value) -> Key {
    let handles = |v: &serde_json::Value| {
        v.as_array()
            .unwrap()
            .iter()
            .map(|h| self::h(h.as_u64().unwrap()))
            .collect::<Vec<_>>()
    };
    Key::new(
        match value["kind"].as_str().unwrap() {
            "birth" => Kind::Birth,
            "split" => Kind::Split,
            "merge" => Kind::Merge,
            _ => unreachable!(),
        },
        &handles(&value["parents"]),
        &handles(&value["left"]),
        &handles(&value["right"]),
    )
    .unwrap()
}

#[test]
fn independent_set_and_dictionary_producer_matches_keys_bindings_and_reducer() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/temporal_cognition/producer.json"
    ))
    .unwrap();
    let mut totals = [0usize; 5];
    for (sequence_id, sequence) in fixture["sequences"].as_array().unwrap().iter().enumerate() {
        let mut g =
            Generator::new(0, 2, 0, 10, sequence["persistence"].as_u64().unwrap() as u8).unwrap();
        for (offset, input) in sequence["frames"].as_array().unwrap().iter().enumerate() {
            let ids: Vec<_> = input["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|i| i.as_u64().unwrap())
                .collect();
            let parents: Vec<_> = input["parents"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p.as_u64())
                .collect();
            let eligible: Vec<_> = input["eligible"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| h(p.as_u64().unwrap()))
                .collect();
            let (c, a) = frame(input["end"].as_u64().unwrap(), &ids, &parents, |i, j| {
                input["matrix"][i][j].as_f64()
            });
            let out = g
                .advance(input["observed"].as_bool().unwrap(), &c, &a, &eligible)
                .unwrap();
            let expected = &sequence["expected"][offset];
            assert_eq!(
                out.bundles.count,
                expected["bundles"].as_array().unwrap().len()
            );
            for (i, bundle) in expected["bundles"].as_array().unwrap().iter().enumerate() {
                assert_eq!(
                    out.bundles.members[i]
                        .iter()
                        .flatten()
                        .map(|h| h.generation)
                        .collect::<Vec<_>>(),
                    bundle
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|h| h.as_u64().unwrap())
                        .collect::<Vec<_>>()
                );
            }
            assert_eq!(
                out.continued as u64,
                expected["continued"].as_u64().unwrap(),
                "sequence {sequence_id} endpoint {offset}"
            );
            assert_eq!(
                out.candidates.iter().flatten().count(),
                expected["candidates"].as_array().unwrap().len()
            );
            for reference in expected["candidates"].as_array().unwrap() {
                let key = decode_key(&reference["key"]);
                let actual = out
                    .candidates
                    .iter()
                    .flatten()
                    .find(|c| c.key == key)
                    .unwrap_or_else(|| {
                        panic!("missing key sequence {sequence_id} endpoint {offset}: {key:?}")
                    });
                assert_eq!(actual.support, reference["support"].as_f64().unwrap());
                assert!((actual.score - reference["score"].as_f64().unwrap()).abs() < 1e-12);
            }
            assert_eq!(
                out.proposals
                    .accepted
                    .iter()
                    .flatten()
                    .map(|c| c.key)
                    .collect::<Vec<_>>(),
                expected["accepted"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(decode_key)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                out.proposals.conflicts as u64,
                expected["conflicts"].as_u64().unwrap()
            );
            assert_eq!(
                out.proposals.pending,
                expected["pending"].as_array().unwrap().len()
            );
            for reference in expected["pending"].as_array().unwrap() {
                let key = decode_key(&reference["key"]);
                let (slot, actual) = g
                    .tracker
                    .pending
                    .iter()
                    .enumerate()
                    .filter_map(|(i, p)| p.map(|p| (i, p)))
                    .find(|(_, p)| p.candidate.key == key)
                    .unwrap();
                assert_eq!(actual.hops as u64, reference["hops"].as_u64().unwrap());
                for i in 0..ids.len() {
                    assert_eq!(
                        g.parents[slot][i].map(|h| h.generation),
                        reference["bindings"][i].as_u64()
                    );
                }
            }
            totals[0] += 1;
            totals[1] += out.candidates.iter().flatten().count();
            totals[2] += out.continued;
            totals[3] += out.proposals.accepted.iter().flatten().count();
            totals[4] += out.proposals.conflicts;
        }
    }
    assert_eq!(totals[0], 864);
    assert!(totals[1] > 1000 && totals[2] > 100 && totals[3] > 100);
    println!(
        "producer_reference endpoints={} candidates={} continued={} accepted={} conflicts={}",
        totals[0], totals[1], totals[2], totals[3], totals[4]
    );
}

#[test]
#[ignore = "release bundle-derived proposal generation cost, excludes acoustic front end and group mutation"]
fn proposal_generation_cost_probe() {
    use std::{hint::black_box, time::Instant};
    for merge in [false, true] {
        let mut g = Generator::new(0, 2, 0, 512, 3).unwrap();
        let mut times = Vec::with_capacity(6000);
        let mut counts = [0usize; 58];
        let mut accepted = 0;
        let mut continued = 0;
        let mut reads = 0;
        for step in 1..=6100 {
            let parents: [Option<u64>; 8] = std::array::from_fn(|i| {
                (step % 4 == 1).then_some(if merge { 101 + (i % 7) as u64 } else { 101 })
            });
            let (c, a) = frame(step * 512, &[1, 2, 3, 4, 5, 6, 7, 8], &parents, |_, _| {
                Some(if merge { 0.9 } else { 0.1 })
            });
            let start = Instant::now();
            let out = black_box(
                g.advance(true, black_box(&c), black_box(&a), black_box(&eligible()))
                    .unwrap(),
            );
            let elapsed = start.elapsed().as_secs_f64() * 1e6;
            if step > 100 {
                times.push(elapsed);
                counts[out.candidates.iter().flatten().count()] += 1;
                accepted += out.proposals.accepted.iter().flatten().count();
                continued += out.continued;
                reads += out.cross_pair_reads + out.bundles.correlation_reads;
            }
        }
        times.sort_by(f64::total_cmp);
        println!(
            "producer_cost {}",
            serde_json::json!({"case":if merge {"merge_plus_birth"} else {"split_plus_birth"},"calls":times.len(),"median_us":times[3000],"p99_us":times[5939],"max_us":times[5999],"candidate_count_histogram":counts.iter().enumerate().filter(|(_,n)|**n>0).map(|(c,n)|(c,*n)).collect::<Vec<_>>(),"accepted":accepted,"continued":continued,"correlation_reads":reads,"generator_bytes":std::mem::size_of::<Generator>(),"output_bytes":std::mem::size_of::<Output>(),"full_O04":false,"scope":"generator incl complete-link, association validation, support measurement, frozen bindings, key selection and reducer; coefficient/association inputs prepared outside timer"})
        );
    }
}
