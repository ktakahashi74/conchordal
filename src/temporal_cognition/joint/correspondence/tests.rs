use super::*;
use crate::temporal_cognition::memory;

fn group() -> Handle {
    Handle {
        bus: 1,
        epoch: 0,
        generation: 2,
    }
}
fn query(id: u64, end: u64) -> recall::ResultSnapshot {
    recall::ResultSnapshot {
        query_id: id,
        cue: None,
        group: group(),
        support_start_sample: 1000,
        support_end_sample: end,
        source_start_sample: 800,
        supporting_audio_end: Some(end),
        available_at: end,
        issued_at: end,
        completed_at: end,
        received_at: end,
        deadline: end + 1000,
        candidates: 16,
        search_covered: true,
        cutoff_tie: false,
        pruned_candidates: 0,
        pruned_ties: 0,
        dp_cells: 100,
        reconstruction_error: 0.,
        best: None,
        prediction: None,
    }
}
fn matched(id: u64, shift: f64, cost: f64) -> recall::MatchSnapshot {
    recall::MatchSnapshot {
        episode_id: id,
        episode_generation: id,
        support_start_sample: 0,
        support_end_sample: 900,
        source_start_sample: 0,
        available_at: 900,
        cost,
        transformation: [Some(shift), Some(0.)],
        ambiguous: false,
        path_steps: 4,
        anchor: Some(0),
        residuals: Some(memory::Residuals {
            coordinate_squared_error: [cost * 4.; 10],
            coordinate_count: [4; 10],
            observed: 4,
            matched: 4,
            ..Default::default()
        }),
    }
}
fn new_cache() -> Cache {
    let mut model = section::tests::runtime_config();
    model.match_coefficients = [0.; 15];
    model.match_means = [0.; 14];
    model.match_deviations = [1.; 14];
    model.match_coefficients[1] = -1.;
    Cache::new(group(), 1000, model, Some(-0.4)).unwrap()
}

#[test]
fn retrieval_projection_precedes_candidate_cap_and_keeps_original_query_targets() {
    let mut cache = new_cache();
    let retained = [(1, 1), (2, 2)];
    let mut q = query(1, 1100);
    q.cue = Some(section::cue::Selection {
        occurrence_id: 7,
        start_sample: 1000,
        support_end_sample: 1100,
        selected_at: 1000,
        last_observed_sample: 1100,
        window_start_sample: 1000.,
        weighted_seconds: 0.1,
    });
    q.candidates = 20;
    let rows: Vec<_> = (0..20)
        .map(|row| {
            std::array::from_fn(|column| {
                let index = row * 4 + column;
                let mut m = matched(
                    if row == 19 { 2 } else { 1 },
                    index as f64 / 128.,
                    if row == 19 {
                        4. + column as f64
                    } else {
                        0.01 + index as f64 / 100.
                    },
                );
                m.anchor = Some(index);
                Some(m)
            })
        })
        .collect();
    cache.refresh(1100, Some((q, &rows)), &retained).unwrap();
    assert_eq!(cache.fresh_supported, 80);
    assert!(
        cache.states[..16]
            .iter()
            .flatten()
            .all(|s| s.matched.unwrap().episode_id == 1)
    );
    let projection = cache.retrieval(1100, true, &retained).unwrap();
    assert!((projection.values[0].unwrap() + 0.1).abs() < 1e-12);
    assert!((projection.values[1].unwrap() - 1.9).abs() < 1e-12);
    assert_eq!(
        projection.targets.map(|t| t.map(|t| (t.id, t.generation))),
        [Some((1, 1)), Some((2, 2))]
    );
    assert_eq!(
        cache.retrieval(1100, false, &[]).unwrap(),
        section::Retrieval::default()
    );
    assert!(cache.retrieval(1200, true, &retained).is_err());
    assert!(cache.retrieval(1100, true, &retained[..1]).is_err());
    let mut invalid = q;
    invalid.group.epoch += 1;
    assert!(
        cache
            .refresh(1200, Some((invalid, &rows)), &retained)
            .is_err()
    );
    assert_eq!(cache.retrieval(1100, true, &retained).unwrap(), projection);
    cache
        .refresh(1200, Some((q, &rows)), &retained[..1])
        .unwrap();
    assert_eq!(
        cache.retrieval(1200, true, &retained[..1]).unwrap().values,
        [Some(-0.1), None]
    );
    cache.refresh(1599, Some((q, &rows)), &retained).unwrap();
    assert_eq!(cache.retrieval(1599, true, &retained).unwrap(), projection);
    cache.refresh(1600, Some((q, &rows)), &retained).unwrap();
    assert_eq!(
        cache.retrieval(1600, true, &retained).unwrap(),
        section::Retrieval::default()
    );
    eprintln!(
        "JOINT_RETRIEVAL_PROJECTION full_candidates=80 retained_candidates=16 distinct_targets=2"
    );
}

#[test]
fn correspondence_reserves_cached_stay_full_fresh_inventory_no_memory_and_unknown() {
    let mut cache = new_cache();
    let retained: Vec<_> = (1..=20).map(|i| (i, i)).collect();
    let original = [[Some(matched(20, 0., 0.1)), None, None, None]];
    cache
        .refresh(1100, Some((query(1, 1100), &original)), &retained)
        .unwrap();
    let parent = cache.states[0].unwrap();
    let rows: Vec<_> = (1..=16)
        .map(|i| [Some(matched(i, 0., i as f64 / 20.)), None, None, None])
        .collect();
    cache
        .refresh(1200, Some((query(2, 1200), &rows)), &retained)
        .unwrap();
    let out = cache.proposals(Some(parent), true, 0.1, &retained).unwrap();
    assert_eq!(out.list.len, 19);
    assert_eq!(out.fresh_truncated, 0);
    assert!(!out.parent_retired);
    assert_eq!(
        out.states
            .iter()
            .flatten()
            .filter(|s| s.matched.is_some())
            .count(),
        17
    );
    assert_eq!(out.states[0].unwrap().id, parent.id);
    assert!(!out.feature_supported[0]);
    assert_eq!(out.states.iter().flatten().filter(|s| s.id == 0).count(), 1);
    let total: f64 = out.states.iter().flatten().map(|s| s.log_score.exp()).sum();
    for choice in out.list.entries[..out.list.len].iter().flatten() {
        let expected = choice.id.map_or_else(
            || 1. - (-0.1_f64 / 10.).exp(),
            |id| {
                out.states
                    .iter()
                    .flatten()
                    .find(|s| s.id == id)
                    .unwrap()
                    .log_score
                    .exp()
                    / total
                    * (-0.1_f64 / 10.).exp()
            },
        );
        assert!((choice.log_weight.exp() - expected).abs() < 1e-14);
    }
    let gap = cache.proposals(None, false, 0.1, &retained).unwrap();
    assert_eq!(gap.list.len, 1);
    assert_eq!(gap.list.entries[0].unwrap().id, None);
    let old = cache
        .proposals(Some(parent), false, 0.1, &retained)
        .unwrap();
    assert_eq!(old.list.len, 2);
    assert!(old.feature_supported.into_iter().all(|v| !v));
    let retired = cache
        .proposals(Some(parent), false, 0.1, &retained[..16])
        .unwrap();
    assert!(retired.parent_retired);
    assert_eq!(retired.list.len, 1);
}

#[test]
fn correspondence_preserves_transform_alternatives_and_deduplicates_current_assignment() {
    let mut cache = new_cache();
    let rows: Vec<_> = (1..=16)
        .map(|id| {
            std::array::from_fn(|i| Some(matched(id, i as f64 / 64., id as f64 + i as f64 / 10.)))
        })
        .collect();
    let retained: Vec<_> = (1..=16).map(|i| (i, i)).collect();
    cache
        .refresh(1100, Some((query(1, 1100), &rows)), &retained)
        .unwrap();
    let ids = cache.states.map(|s| s.map(|s| s.id));
    assert_eq!(cache.fresh_supported, 64);
    let parent = cache.states[0].unwrap();
    let output = cache.proposals(Some(parent), true, 0.1, &retained).unwrap();
    assert_eq!(output.fresh_truncated, 48);
    assert_eq!(output.list.len, 18);
    assert!(output.feature_supported[0]);
    assert_eq!(output.states[0].unwrap().id, parent.id);
    cache
        .refresh(1100, Some((query(1, 1100), &rows)), &retained)
        .unwrap();
    assert_eq!(cache.states.map(|s| s.map(|s| s.id)), ids);
    let mut reversed = rows.clone();
    reversed.reverse();
    cache
        .refresh(1100, Some((query(1, 1100), &reversed)), &retained)
        .unwrap();
    assert_eq!(cache.states.map(|s| s.map(|s| s.id)), ids);
    cache
        .refresh(1200, Some((query(2, 1200), &rows)), &retained)
        .unwrap();
    let output = cache.proposals(Some(parent), true, 0.1, &retained).unwrap();
    assert_eq!(output.list.len, 18);
    assert_eq!(output.states[0].unwrap().id, parent.id);
    assert_eq!(output.states[0].unwrap().support.query, 2);
    assert_eq!(parent.support.query, 1);
}

#[test]
fn correspondence_missing_stale_retired_or_ambiguous_input_cannot_supply_fresh_evidence() {
    let mut cache = new_cache();
    let retained = [(1, 1), (2, 2)];
    let mut rows = [
        [
            Some(matched(1, 0., 0.)),
            Some(matched(1, 0.5, 0.)),
            None,
            None,
        ],
        [Some(matched(2, 0., 0.)), None, None, None],
    ];
    rows[0][1].as_mut().unwrap().ambiguous = true;
    rows[1][0]
        .as_mut()
        .unwrap()
        .residuals
        .as_mut()
        .unwrap()
        .matched = 0;
    let mut q = query(1, 1100);
    q.cutoff_tie = true;
    q.search_covered = false;
    q.pruned_candidates = 7;
    cache.refresh(1100, Some((q, &rows)), &retained).unwrap();
    assert_eq!(
        (cache.excluded_ambiguous, cache.excluded_unsupported),
        (1, 1)
    );
    let parent = cache.states[0].unwrap();
    assert!(parent.support.cutoff_tie);
    assert_eq!(parent.support.search_pruned, 7);
    cache.refresh(1599, Some((q, &rows)), &retained).unwrap();
    assert!(
        cache
            .proposals(None, true, 0.1, &retained)
            .unwrap()
            .list
            .len
            > 1
    );
    cache.refresh(1600, Some((q, &rows)), &retained).unwrap();
    assert_eq!(
        cache
            .proposals(None, true, 0.1, &retained)
            .unwrap()
            .list
            .len,
        1
    );
    let stale = cache.proposals(Some(parent), true, 0.1, &retained).unwrap();
    assert_eq!(stale.list.len, 2);
    assert!(!stale.feature_supported[0]);
    q = query(2, 1700);
    q.supporting_audio_end = None;
    cache.refresh(1700, Some((q, &rows)), &retained).unwrap();
    assert_eq!(
        cache
            .proposals(None, true, 0.1, &retained)
            .unwrap()
            .list
            .len,
        1
    );
    q = query(3, 1800);
    cache
        .refresh(1800, Some((q, &rows)), &retained[1..])
        .unwrap();
    assert_eq!(cache.excluded_retired, 2);
    let no_memory = cache.proposals(None, true, 0.1, &retained).unwrap();
    assert_eq!(no_memory.list.len, 1);
    assert!(no_memory.states.iter().all(Option::is_none));
    let before = (cache.cut, cache.next_id, cache.support);
    let mut foreign = q;
    foreign.group.epoch += 1;
    assert!(
        cache
            .refresh(1800, Some((foreign, &rows)), &retained)
            .is_err()
    );
    assert_eq!((cache.cut, cache.next_id, cache.support), before);
    let mut exhausted = new_cache();
    exhausted.next_id = u64::MAX;
    assert!(
        exhausted
            .refresh(1800, Some((q, &rows)), &retained)
            .is_err()
    );
    assert_eq!(exhausted.next_id, u64::MAX);
    assert_eq!(exhausted.cut, 0);
}

#[test]
fn actual_correspondence_inventory_composes_and_normalizes_without_merging_no_memory_and_unknown() {
    use crate::temporal_cognition::joint::{Normalizer, Pair, Shared, proposals::Composer};
    let mut cache = new_cache();
    let rows: Vec<_> = (1..=16)
        .map(|id| [Some(matched(id, 0., id as f64 / 20.)), None, None, None])
        .collect();
    let retained: Vec<_> = (1..=16).map(|id| (id, id)).collect();
    cache
        .refresh(1100, Some((query(1, 1100), &rows)), &retained)
        .unwrap();
    let candidates = cache.proposals(None, true, 0.1, &retained).unwrap();
    for state in candidates.states.iter().flatten() {
        if let Some(m) = state.matched {
            assert_eq!(state.log_score, -m.cost.sqrt());
        }
    }
    let lists = [
        List::build(Kind::Articulation, &[], true, 0.1, 8).unwrap(),
        List::build(Kind::Grouping, &[], true, 0.1, 19).unwrap(),
        List::build(Kind::Phrase, &[], true, 0.1, 6).unwrap(),
        List::build(Kind::Section, &[], true, 0.1, 5).unwrap(),
        candidates.list,
    ];
    let mut composer = Composer::new();
    let tuples = composer
        .compose(
            &lists,
            &crate::temporal_cognition::joint::proposals::Compatibility::default(),
        )
        .unwrap();
    assert!(tuples.heap_pops <= 16 && tuples.priority_evaluations <= 86);
    let known_none = tuples
        .tuples
        .iter()
        .flatten()
        .find(|t| t.ids[4] == Some(0))
        .unwrap();
    assert!(!known_none.all_unknown);
    assert_eq!(
        tuples
            .tuples
            .iter()
            .flatten()
            .filter(|t| t.all_unknown)
            .count(),
        1
    );
    let pairs: Vec<_> = tuples
        .tuples
        .iter()
        .flatten()
        .enumerate()
        .map(|(i, t)| Pair {
            parent: 0,
            extension: i as u8,
            resolved: !t.all_unknown,
            log_prior: 0.,
            log_transition: t.log_transition,
            log_potential: 0.,
        })
        .collect();
    let groups: &[&[Pair]] = &[&pairs];
    let contexts = [
        Shared {
            pair: Pair {
                parent: 0,
                extension: 0,
                resolved: true,
                log_prior: 0.,
                log_transition: 0.9_f64.ln(),
                log_potential: 0.,
            },
            groups,
        },
        Shared {
            pair: Pair {
                parent: 0,
                extension: 1,
                resolved: false,
                log_prior: 0.,
                log_transition: 0.1_f64.ln(),
                log_potential: 0.,
            },
            groups,
        },
    ];
    let mut normalizer = Normalizer::new();
    let out = normalizer.normalize(&contexts, true).unwrap();
    let context = out.contexts[0].unwrap();
    assert!((context.weight.mass - 0.9).abs() < 1e-12);
    assert!((out.explicit_unknown - 0.1).abs() < 1e-12);
    let conditional = context.groups[0];
    for w in conditional.rows.iter().flatten() {
        let expected = pairs[w.extension as usize].log_transition.exp();
        assert!((w.mass - expected).abs() < 1e-12);
    }
    let unknown = pairs
        .iter()
        .find(|p| !p.resolved)
        .unwrap()
        .log_transition
        .exp();
    assert!((conditional.explicit_unknown - unknown).abs() < 1e-12);
    eprintln!(
        "JOINT_CORRESPONDENCE_INVENTORY fresh=16 known_no_memory=1 tuple_count={} cache_bytes={} state_bytes={} output_bytes={}",
        pairs.len(),
        std::mem::size_of::<Cache>(),
        std::mem::size_of::<State>(),
        std::mem::size_of::<Output>()
    );
}

#[test]
fn correspondence_origins_resolve_only_from_the_current_cache_or_exact_parent() {
    let mut cache = new_cache();
    let retained = [(1, 1)];
    let rows = [[Some(matched(1, 0., 0.1)), None, None, None]];
    cache
        .refresh(1100, Some((query(1, 1100), &rows)), &retained)
        .unwrap();
    let out = cache.proposals(None, true, 0.1, &retained).unwrap();
    for (i, s) in out
        .states
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.map(|s| (i, s)))
    {
        let choice = Extension {
            id: s.id,
            origin: out.origins[i].unwrap(),
        };
        let (actual, supported) = cache.resolve(choice, None, true, 1100, &retained).unwrap();
        assert_eq!(*actual, s);
        assert!(supported);
        assert_eq!(
            cache.resolve(choice, None, false, 1100, &retained).err(),
            Some("missing interval cannot refresh correspondence evidence")
        );
        assert_eq!(
            cache.resolve(choice, None, true, 1101, &retained).err(),
            Some("invalid correspondence source clock or parent owner")
        );
        assert_eq!(
            cache
                .resolve(
                    Extension {
                        id: s.id + 100,
                        ..choice
                    },
                    None,
                    true,
                    1100,
                    &retained
                )
                .err(),
            Some("correspondence identity differs from selected source")
        );
    }
    let old = cache.states[0].unwrap();
    let stay = Extension {
        id: old.id,
        origin: Origin::Retained,
    };
    assert_eq!(
        cache.resolve(stay, None, true, 1100, &retained).err(),
        Some("missing retained correspondence parent")
    );
    assert_eq!(
        cache.resolve(stay, Some(&old), false, 1100, &[]).err(),
        Some("retired correspondence target")
    );
    cache
        .refresh(1200, Some((query(2, 1200), &rows)), &retained)
        .unwrap();
    let out = cache.proposals(Some(old), true, 0.1, &retained).unwrap();
    let choice = Extension {
        id: out.states[0].unwrap().id,
        origin: out.origins[0].unwrap(),
    };
    let (new, supported) = cache
        .resolve(choice, Some(&old), true, 1200, &retained)
        .unwrap();
    assert!(supported);
    assert_eq!(choice.id, old.id);
    assert_eq!(new.support.query, 2);
    assert_eq!(old.support.query, 1);
    assert_eq!(
        cache
            .resolve(
                Extension {
                    id: 1,
                    origin: Origin::Fresh(255)
                },
                None,
                true,
                1200,
                &retained
            )
            .err(),
        Some("missing fresh correspondence slot")
    );
    let mut foreign = old;
    foreign.group.epoch += 1;
    assert_eq!(
        cache
            .resolve(stay, Some(&foreign), false, 1200, &retained)
            .err(),
        Some("invalid correspondence source clock or parent owner")
    );
}
