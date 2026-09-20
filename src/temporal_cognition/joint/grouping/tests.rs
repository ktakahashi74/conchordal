use super::*;
use crate::temporal_cognition::features::Accent;

pub(in crate::temporal_cognition::joint) fn group() -> Handle {
    Handle {
        bus: 0,
        epoch: 7,
        generation: 2,
    }
}

pub(in crate::temporal_cognition::joint) fn estimator() -> Estimator {
    Estimator::new(group(), 128, 1000, 32000, 1024, 1. / 24.).unwrap()
}

pub(in crate::temporal_cognition::joint) fn inventory() -> groupings::Inventory {
    groupings::Inventory::new(
        group(),
        0,
        groupings::Controls {
            tolerance: 0.1,
            integers_234_only: false,
            strict_integer: false,
            one_skip_words: false,
        },
    )
    .unwrap()
}

pub(in crate::temporal_cognition::joint) fn deliver(estimator: &mut Estimator, end: u64) {
    estimator
        .deliver(
            Accent {
                group: group(),
                event_start: end - 16,
                event_end: end,
                raw_intervals: [
                    (end - 48, end - 32),
                    (end - 32, end - 16),
                    (end - 16, end),
                    (end, end + 16),
                ],
                source_start: end - 48,
                source_end: end + 16,
                available_end: end + 32,
                weight: 1.,
                observed_prefix: end,
            },
            end + 32,
        )
        .unwrap();
}

#[test]
fn actual_grouping_inventory_preserves_acoustic_keys_support_and_transition_mass() {
    let mut estimator = estimator();
    let mut inventory = inventory();
    let mut cache = Cache::new(group(), Some(1.)).unwrap();
    let mut parent = None;
    let (mut candidates, mut inherited, mut exclusions, mut max_error) = (0, 0, 0, 0_f64);
    for step in 1..=90 {
        let end = step * 500 + 500;
        deliver(&mut estimator, end);
        assert!(inventory.refresh(&estimator).unwrap());
        let view = inventory.snapshot().unwrap();
        cache.refresh(end + 32, Some((&view, &estimator))).unwrap();
        let first = cache.states;
        let next_id = cache.next_id;
        cache.refresh(end + 32, Some((&view, &estimator))).unwrap();
        assert_eq!(cache.states, first);
        assert_eq!(cache.next_id, next_id);
        let observed = step % 11 != 0;
        let out = cache.proposals(parent, observed, 0.5).unwrap();
        assert_eq!(out.list.truncated, 0);
        if step == 1 {
            assert_eq!(
                out.list.len, 1,
                "one accent is not a supported period search"
            );
            assert!(out.states.iter().all(Option::is_none));
        }
        let total: f64 = out.states.iter().flatten().map(|s| s.raw_score).sum();
        for entry in out.list.entries[..out.list.len].iter().flatten() {
            let expected = if let Some(id) = entry.id {
                let (i, state) = out
                    .states
                    .iter()
                    .enumerate()
                    .find_map(|(i, s)| s.filter(|s| s.id == id).map(|s| (i, s)))
                    .unwrap();
                if out.feature_supported[i] {
                    assert!(observed);
                    if let Some(proposal) = state.proposal {
                        assert_eq!(
                            Some(&proposal),
                            view.proposals
                                .iter()
                                .flatten()
                                .find(|p| p.key == proposal.key)
                        );
                        assert_eq!(state.raw_score, proposal.endpoint_weight);
                        assert_eq!(state.support.source_end, view.period_source_end.unwrap());
                        assert_eq!(state.support.available, view.period_available_end.unwrap());
                        assert_eq!(state.support.refreshed_at, view.refreshed_at);
                        candidates += 1;
                    }
                } else {
                    assert_eq!(Some(state), parent);
                    inherited += 1;
                }
                if state.support.window_limited_cases > 0
                    || state.support.capacity_evicted_through.is_some()
                    || state.support.excluded_cases > 0
                {
                    assert!(state.proposal.is_some());
                    exclusions += 1;
                }
                (-0.5_f64 / 10.).exp() * state.raw_score / total
            } else if total > 0. {
                -(-0.5_f64 / 10.).exp_m1()
            } else {
                1.
            };
            max_error = max_error.max((entry.log_weight.exp() - expected).abs());
        }
        if !observed && parent.is_none() {
            assert_eq!(out.list.len, 1);
        }
        if parent.is_none() {
            parent = out.states.iter().flatten().next().copied();
        }
    }
    assert!(candidates > 500 && inherited > 5 && exclusions > 500);
    assert!(max_error < 1e-14);
    println!(
        "JOINT_GROUPING_REAL steps=90 candidates={candidates} inherited={inherited} incomplete_checks={exclusions} max_error={max_error:e} cache_bytes={} state_bytes={} output_bytes={}",
        std::mem::size_of::<Cache>(),
        std::mem::size_of::<State>(),
        std::mem::size_of::<Output>()
    );
}

#[test]
fn grouping_source_gaps_foreign_owners_and_rejected_refresh_do_not_invent_support() {
    let mut estimator = estimator();
    let mut inventory = inventory();
    let mut cache = Cache::new(group(), Some(1.)).unwrap();
    inventory.refresh(&estimator).unwrap();
    cache
        .refresh(0, Some((&inventory.snapshot().unwrap(), &estimator)))
        .unwrap();
    assert_eq!(cache.proposals(None, true, 0.1).unwrap().list.len, 1);
    for end in (1000..=6000).step_by(500) {
        deliver(&mut estimator, end);
    }
    inventory.refresh(&estimator).unwrap();
    let view = inventory.snapshot().unwrap();
    cache.refresh(6032, Some((&view, &estimator))).unwrap();
    let parent = cache.states[0].unwrap();
    let saved = cache.states;
    let saved_id = cache.next_id;
    let mut foreign = view;
    foreign.group.generation += 1;
    assert_eq!(
        cache.refresh(6032, Some((&foreign, &estimator))),
        Err("invalid grouping source owner or original clock")
    );
    assert_eq!(cache.states, saved);
    assert_eq!(cache.next_id, saved_id);
    assert!(cache.refresh(6031, Some((&view, &estimator))).is_err());
    deliver(&mut estimator, 6500);
    cache.refresh(6532, Some((&view, &estimator))).unwrap();
    assert!(cache.states.iter().all(Option::is_none));
    let stale = cache.proposals(Some(parent), true, 0.5).unwrap();
    assert_eq!(stale.states[0], Some(parent));
    assert!(!stale.feature_supported[0]);
    assert_eq!(stale.list.len, 2);
    inventory.refresh(&estimator).unwrap();
    let next_view = inventory.snapshot().unwrap();
    cache.next_id = u64::MAX;
    assert_eq!(
        cache.refresh(6532, Some((&next_view, &estimator))),
        Err("grouping identity exhausted")
    );
    assert!(cache.states.iter().all(Option::is_none));
    assert_eq!(cache.next_id, u64::MAX);
    cache.next_id = saved_id;
    cache.refresh(6532, Some((&next_view, &estimator))).unwrap();
    let missing = cache.proposals(Some(parent), false, 0.5).unwrap();
    assert_eq!(missing.states[0], Some(parent));
    assert_eq!(missing.list.len, 2);
    assert!(missing.feature_supported.iter().all(|s| !s));
    assert_eq!(cache.proposals(None, false, 0.5).unwrap().list.len, 1);
    cache.refresh(6600, None).unwrap();
    assert_eq!(
        cache.proposals(Some(parent), true, 0.1).unwrap().states[0],
        Some(parent)
    );
    let mut foreign_parent = parent;
    foreign_parent.group.bus = 1;
    assert!(cache.proposals(Some(foreign_parent), true, 0.1).is_err());
}

#[test]
fn full_grouping_inventory_reserves_stay_ungrouped_and_unknown_before_composition() {
    use super::super::proposals::Composer;
    let mut estimator = estimator();
    let mut inventory = inventory();
    let mut cache = Cache::new(group(), Some(1.)).unwrap();
    for end in (1000..=6000).step_by(500) {
        deliver(&mut estimator, end);
    }
    inventory.refresh(&estimator).unwrap();
    cache
        .refresh(6032, Some((&inventory.snapshot().unwrap(), &estimator)))
        .unwrap();
    let parent = cache.states[0].unwrap();
    for end in (6500..=40000).step_by(500) {
        deliver(&mut estimator, end);
    }
    inventory.refresh(&estimator).unwrap();
    let mut view = inventory.snapshot().unwrap();
    assert_eq!(view.proposals.iter().flatten().count(), 16);
    assert!(
        !view
            .proposals
            .iter()
            .flatten()
            .any(|p| Some(p.key) == parent.proposal.map(|p| p.key))
    );
    // A controlled coverage fixture exercises the full inventory; it is not measured absence evidence.
    view.work.integer_window_limited_cases = 0;
    view.work.word_insufficient_endpoints = 0;
    view.work.admitted_cases = 16;
    view.work.duplicate_retained_keys = 0;
    view.capacity_evicted_through = None;
    cache.refresh(40032, Some((&view, &estimator))).unwrap();
    let out = cache.proposals(Some(parent), true, 0.1).unwrap();
    assert_eq!(out.list.len, 19);
    assert_eq!(out.states[0], Some(parent));
    assert!(!out.feature_supported[0]);
    assert_eq!(
        out.states
            .iter()
            .flatten()
            .filter(|s| s.proposal.is_none())
            .count(),
        1
    );
    assert_eq!(
        out.list
            .entries
            .iter()
            .flatten()
            .filter(|s| s.id == Some(0))
            .count(),
        1
    );
    assert_eq!(
        out.list
            .entries
            .iter()
            .flatten()
            .filter(|s| s.id.is_none())
            .count(),
        1
    );
    let ungrouped = out
        .states
        .iter()
        .flatten()
        .find(|s| s.id == 0)
        .copied()
        .unwrap();
    for exclusion in 0..3 {
        let mut incomplete = view;
        match exclusion {
            0 => incomplete.work.integer_window_limited_cases = 1,
            1 => incomplete.capacity_evicted_through = Some(1000),
            _ => incomplete.work.admitted_cases += 1,
        }
        cache
            .refresh(40032, Some((&incomplete, &estimator)))
            .unwrap();
        assert!(
            cache
                .proposals(None, true, 0.1)
                .unwrap()
                .states
                .iter()
                .flatten()
                .all(|s| s.id != 0)
        );
        let retained = cache.proposals(Some(ungrouped), true, 0.1).unwrap();
        assert_eq!(retained.states[0], Some(ungrouped));
        assert!(!retained.feature_supported[0]);
    }
    let mut no_prior = Cache::new(group(), None).unwrap();
    no_prior.refresh(40032, Some((&view, &estimator))).unwrap();
    assert!(
        no_prior
            .proposals(None, true, 0.1)
            .unwrap()
            .states
            .iter()
            .flatten()
            .all(|s| s.id != 0)
    );
    let entries = [
        Kind::Articulation,
        Kind::Grouping,
        Kind::Phrase,
        Kind::Section,
        Kind::Correspondence,
    ]
    .map(|kind| {
        if kind == Kind::Grouping {
            out.list
        } else {
            List::build(kind, &[], true, 0.1, 1).unwrap()
        }
    });
    let mut composer = Composer::new();
    let tuples = composer
        .compose(
            &entries,
            &crate::temporal_cognition::joint::proposals::Compatibility::default(),
        )
        .unwrap();
    assert_eq!(tuples.tuples.iter().flatten().count(), 16);
    println!(
        "JOINT_GROUPING_CAPACITY list={} tuples={} excluded_cases={}",
        out.list.len,
        tuples.tuples.iter().flatten().count(),
        out.states[1].unwrap().support.excluded_cases
    );
}

#[test]
fn grouping_origins_resolve_only_from_the_current_cache_or_exact_parent() {
    let mut estimator = estimator();
    let mut inventory = inventory();
    let mut cache = Cache::new(group(), Some(1.)).unwrap();
    for end in (1000..=6000).step_by(500) {
        deliver(&mut estimator, end);
    }
    inventory.refresh(&estimator).unwrap();
    cache
        .refresh(6032, Some((&inventory.snapshot().unwrap(), &estimator)))
        .unwrap();
    let output = cache.proposals(None, true, 0.5).unwrap();
    let parent = output.states[0].unwrap();
    let choice = Extension {
        id: parent.id,
        origin: output.origins[0].unwrap(),
    };
    assert_eq!(cache.resolve(choice, None, true, 6032), Ok((&parent, true)));
    assert!(cache.resolve(choice, None, false, 6032).is_err());
    assert!(cache.resolve(choice, None, true, 6031).is_err());
    assert!(
        cache
            .resolve(
                Extension {
                    id: parent.id + 99,
                    ..choice
                },
                None,
                true,
                6032
            )
            .is_err()
    );
    assert!(
        cache
            .resolve(
                Extension {
                    origin: Origin::Fresh(17),
                    ..choice
                },
                None,
                true,
                6032
            )
            .is_err()
    );
    let mut foreign = parent;
    foreign.group.generation += 1;
    assert!(cache.resolve(choice, Some(&foreign), true, 6032).is_err());
    cache.refresh(6132, None).unwrap();
    let retained = Extension {
        id: parent.id,
        origin: Origin::Retained,
    };
    assert_eq!(
        cache.resolve(retained, Some(&parent), false, 6132),
        Ok((&parent, false))
    );
    assert!(cache.resolve(retained, None, false, 6132).is_err());
    assert!(cache.resolve(choice, Some(&parent), true, 6132).is_err());
    assert!(
        cache
            .resolve(
                Extension {
                    id: parent.id + 1,
                    ..retained
                },
                Some(&parent),
                false,
                6132
            )
            .is_err()
    );
}
