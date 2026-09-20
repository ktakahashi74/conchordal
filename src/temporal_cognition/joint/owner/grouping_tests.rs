use super::tests::unknown;
use super::*;
use crate::temporal_cognition::joint::proposals::{Composer, Kind, List};

#[test]
fn acoustic_grouping_payloads_survive_joint_pruning_missing_input_and_group_reordering() {
    let mut estimator = grouping::tests::estimator();
    let mut inventory = grouping::tests::inventory();
    let group = grouping::tests::group();
    let other = Handle {
        generation: 3,
        ..group
    };
    let mut cache = grouping::Cache::new(group, Some(1.)).unwrap();
    let mut owner = State::new(group.bus, group.epoch, 0, 500).unwrap();
    let arenas = [
        owner.current.groupings.as_ptr(),
        owner.next.groupings.as_ptr(),
    ];
    let mut composer = Composer::new();
    let unknown_components = [
        Kind::Articulation,
        Kind::Grouping,
        Kind::Phrase,
        Kind::Section,
        Kind::Correspondence,
    ]
    .map(|kind| List::build(kind, &[], true, 0.5, 1).unwrap());
    let other_local = [unknown()];
    let (mut retained, mut inherited, mut rejected, mut reorders) = (0, 0, 0, 0);
    for step in 1..=90 {
        let end = step * 500;
        let observed = ![27, 28, 48, 49].contains(&step);
        if observed {
            grouping::tests::deliver(&mut estimator, end - 32);
            inventory.refresh(&estimator).unwrap();
        }
        let view = inventory.snapshot();
        cache
            .refresh(end, view.as_ref().map(|v| (v, &estimator)))
            .unwrap();
        let g = (step % 2) as usize;
        let groups = if g == 0 {
            [group, other]
        } else {
            [other, group]
        };
        let previous = owner.snapshot().clone();
        let previous_g = previous.groups.iter().position(|h| *h == Some(group));
        reorders += usize::from(previous_g.is_some_and(|old| old != g));
        let mut contexts: Vec<_> = previous.paths.iter().flatten().copied().map(Some).collect();
        contexts.push(None);
        let mut rows = Vec::new();
        let mut metadata = Vec::new();
        for context in contexts {
            let mut parents: Vec<_> = context.zip(previous_g).map_or_else(Vec::new, |(c, g)| {
                c.groups[g].iter().flatten().copied().map(Some).collect()
            });
            parents.push(None);
            for known in [true, false] {
                if known && context.is_none() && !observed {
                    continue;
                }
                let mut local = Vec::new();
                for parent in &parents {
                    let old = parent
                        .and_then(|p| p.grouping_slot)
                        .and_then(|i| previous.groupings[i]);
                    let candidates = cache.proposals(old, observed, 0.5).unwrap();
                    let mut lists = unknown_components;
                    lists[1] = candidates.list;
                    let tuples = composer
                        .compose(
                            &lists,
                            &crate::temporal_cognition::joint::proposals::Compatibility::default(),
                        )
                        .unwrap();
                    for (extension, tuple) in tuples.tuples.iter().flatten().enumerate() {
                        let choice = tuple.ids[1].map(|id| {
                            let i = candidates
                                .states
                                .iter()
                                .position(|s| s.is_some_and(|s| s.id == id))
                                .unwrap();
                            grouping::Extension {
                                id,
                                origin: candidates.origins[i].unwrap(),
                            }
                        });
                        local.push(LocalExtension {
                            parent: parent.map(|p| p.id),
                            extension: extension as u8,
                            components: tuple.ids,
                            grouping: choice,
                            log_transition: tuple.log_transition,
                            ..unknown()
                        });
                    }
                }
                rows.push(local);
                metadata.push((context.map(|c| c.id), known));
            }
        }
        let refs: Vec<[&[LocalExtension]; 2]> = rows
            .iter()
            .map(|r| {
                if g == 0 {
                    [r.as_slice(), &other_local]
                } else {
                    [&other_local, r.as_slice()]
                }
            })
            .collect();
        let inputs: Vec<_> = metadata
            .iter()
            .enumerate()
            .map(|(i, &(parent, known))| ContextExtension {
                parent,
                extension: u8::from(!known),
                key: known.then_some(42),
                log_transition: if parent.is_none() && !observed {
                    0.
                } else if known {
                    0.9_f64.ln()
                } else {
                    0.1_f64.ln()
                },
                log_potential: 0.,
                groups: &refs[i],
            })
            .collect();
        let interval = [end - 500, end];
        if rows.iter().flatten().any(|r| r.grouping.is_some()) {
            let next_id = owner.next_id;
            assert_eq!(
                owner
                    .advance(interval, observed, &groups, &inputs, Sources::default())
                    .err(),
                Some("missing joint grouping source")
            );
            assert_eq!(owner.snapshot(), &previous);
            assert_eq!(owner.next_id, next_id);
            rejected += 1;
        }
        let sources = if g == 0 {
            [Some(&cache), None]
        } else {
            [None, Some(&cache)]
        };
        let output = owner
            .advance(
                interval,
                observed,
                &groups,
                &inputs,
                Sources {
                    groupings: &sources,
                    ..Sources::default()
                },
            )
            .unwrap();
        assert!(arenas.contains(&output.groupings.as_ptr()));
        assert_eq!(
            output.groupings.iter().flatten().count(),
            output
                .paths
                .iter()
                .flatten()
                .flat_map(|c| c.groups[g].iter().flatten())
                .count()
        );
        for context in output.paths.iter().flatten() {
            let summary = context.state.groups[g].unwrap();
            assert_eq!(summary.group, group);
            assert!(
                (summary.grouping.iter().sum::<f64>() - summary.represented_mass).abs() < 1e-12
            );
            assert!(
                (summary.represented_mass + summary.explicit_unknown + summary.pruned_mass - 1.)
                    .abs()
                    < 1e-12
            );
            assert!(
                summary.support.is_none(),
                "component evidence is not a raw acoustic interval"
            );

            assert!(context.groups[1 - g].iter().all(Option::is_none));
            for path in context.groups[g].iter().flatten() {
                let state = output.groupings[path.grouping_slot.unwrap()].unwrap();
                assert_eq!(path.components[1], Some(state.id));
                assert_eq!(state.group, group);
                assert!(state.support.available <= end && state.support.refreshed_at <= end);
                if path.grouping_supported {
                    assert!(observed);
                    let view = view.unwrap();
                    assert_eq!(state.support.refreshed_at, view.refreshed_at);
                    assert!(
                        view.proposals
                            .iter()
                            .flatten()
                            .any(|p| Some(*p) == state.proposal)
                    );
                } else {
                    let old = previous
                        .paths
                        .iter()
                        .flatten()
                        .flat_map(|c| c.groups[previous_g.unwrap()].iter().flatten())
                        .find(|p| Some(p.id) == path.parent)
                        .unwrap();
                    assert_eq!(
                        state,
                        previous.groupings[old.grouping_slot.unwrap()].unwrap()
                    );
                    inherited += 1;
                }
                assert!(observed || !path.grouping_supported);
                retained += 1;
            }
        }
    }
    assert!(retained > 1000 && inherited > 100 && rejected > 50 && reorders == 89);
    eprintln!(
        "JOINT_GROUPING_OWNER steps=90 retained={retained} inherited={inherited} rejected={rejected} reorders={reorders} arena_bytes={} snapshot_bytes={} instruction_bytes={}",
        std::mem::size_of_val(owner.current.groupings.as_ref()),
        std::mem::size_of::<Snapshot>(),
        std::mem::size_of::<grouping::Extension>()
    );
}
