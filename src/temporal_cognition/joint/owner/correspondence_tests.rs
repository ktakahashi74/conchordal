use super::tests::unknown;
use super::*;
use crate::temporal_cognition::joint::proposals::{Composer, Kind, List};
use crate::temporal_cognition::{recall, section};

#[test]
fn actual_matcher_payloads_survive_joint_pruning_and_condition_the_next_query() {
    let mut cfg = recall::tests::config();
    cfg.episodes = 2;
    let mut recall = recall::Recall::new(1, 0, 1000, 100, cfg, None).unwrap();
    let group = recall::tests::input(1, 0.).retained_groups[0].unwrap();
    let mut cache =
        correspondence::Cache::new(group, 1000, section::tests::runtime_config(), Some(-3.))
            .unwrap();
    let mut owner = State::new(1, 0, 0, 100).unwrap();
    let arenas = [
        owner.current.correspondences.as_ptr(),
        owner.next.correspondences.as_ptr(),
    ];
    let mut composer = Composer::new();
    let unknown_components = [
        List::build(Kind::Articulation, &[], true, 0.1, 8).unwrap(),
        List::build(Kind::Grouping, &[], true, 0.1, 19).unwrap(),
        List::build(Kind::Phrase, &[], true, 0.1, 6).unwrap(),
        List::build(Kind::Section, &[], true, 0.1, 5).unwrap(),
    ];
    let (mut matched, mut no_memory, mut inherited, mut retired, mut rejected) = (0, 0, 0, 0, 0);
    for step in 1..=96 {
        let observed = ![27, 28, 48, 49].contains(&step);
        if observed {
            recall
                .advance(
                    &recall::tests::input(step, ((step - 1) % 16) as f64 * 0.02),
                    step * 100,
                    None,
                )
                .unwrap();
        }
        let retained = recall.retained_ids();
        cache
            .refresh(step * 100, recall.matches_for(group), &retained)
            .unwrap();
        let previous = owner.snapshot().clone();
        let mut contexts: Vec<_> = previous.paths.iter().flatten().copied().map(Some).collect();
        contexts.push(None);
        let mut rows = Vec::new();
        let mut metadata = Vec::new();
        for context in contexts {
            let mut parents: Vec<_> = context.map_or_else(Vec::new, |c| {
                c.groups[0].iter().flatten().copied().map(Some).collect()
            });
            parents.push(None);
            for known in [true, false] {
                if known && context.is_none() && !observed {
                    continue;
                }
                let mut local = Vec::new();
                for parent in &parents {
                    let old = parent
                        .and_then(|p| p.correspondence_slot)
                        .and_then(|i| previous.correspondences[i]);
                    let candidates = cache.proposals(old, observed, 0.1, &retained).unwrap();
                    retired += usize::from(candidates.parent_retired);
                    let lists = [
                        unknown_components[0],
                        unknown_components[1],
                        unknown_components[2],
                        unknown_components[3],
                        candidates.list,
                    ];
                    let tuples = composer
                        .compose(
                            &lists,
                            &crate::temporal_cognition::joint::proposals::Compatibility::default(),
                        )
                        .unwrap();
                    for (extension, t) in tuples.tuples.iter().flatten().enumerate() {
                        let choice = t.ids[4].map(|id| {
                            let i = candidates
                                .states
                                .iter()
                                .position(|s| s.is_some_and(|s| s.id == id))
                                .unwrap();
                            correspondence::Extension {
                                id,
                                origin: candidates.origins[i].unwrap(),
                            }
                        });
                        local.push(LocalExtension {
                            parent: parent.map(|p| p.id),
                            extension: extension as u8,
                            components: t.ids,
                            correspondence: choice,
                            log_transition: t.log_transition,
                            ..unknown()
                        });
                    }
                }
                rows.push(local);
                metadata.push((context.map(|c| c.id), known));
            }
        }
        let refs: Vec<[&[LocalExtension]; 1]> = rows.iter().map(|r| [r.as_slice()]).collect();
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
        let interval = [(step - 1) * 100, step * 100];
        if rows.iter().flatten().any(|r| r.correspondence.is_some()) {
            let next_id = owner.next_id;
            assert_eq!(
                owner
                    .advance(
                        interval,
                        observed,
                        &[group],
                        &inputs,
                        Sources {
                            retained_episodes: &retained,
                            ..Sources::default()
                        }
                    )
                    .err(),
                Some("missing joint correspondence source")
            );
            assert_eq!(owner.snapshot(), &previous);
            assert_eq!(owner.next_id, next_id);
            rejected += 1;
        }
        let output = owner
            .advance(
                interval,
                observed,
                &[group],
                &inputs,
                Sources {
                    correspondences: &[Some(&cache)],
                    retained_episodes: &retained,
                    ..Sources::default()
                },
            )
            .unwrap();
        assert!(arenas.contains(&output.correspondences.as_ptr()));
        assert_eq!(
            output.correspondences.iter().flatten().count(),
            output
                .paths
                .iter()
                .flatten()
                .flat_map(|c| c.groups[0].iter().flatten())
                .count()
        );
        for context in output.paths.iter().flatten() {
            let summary = context.state.groups[0].unwrap();
            assert_eq!(summary.group, group);
            assert!(
                (summary.correspondence.iter().sum::<f64>() - summary.represented_mass).abs()
                    < 1e-12
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

            for path in context.groups[0].iter().flatten() {
                let state = output.correspondences[path.correspondence_slot.unwrap()].unwrap();
                assert_eq!(path.components[4], Some(state.id));
                assert_eq!(state.group, group);
                assert!(
                    state.support.received <= step * 100
                        && state.support.audio_end <= state.support.end
                );
                if let Some(m) = state.matched {
                    assert!(retained.contains(&(m.episode_id, m.episode_generation)));
                    matched += 1;
                } else {
                    assert_eq!(state.id, 0);
                    no_memory += 1;
                }
                if path.correspondence_supported {
                    assert!(observed);
                    let (query, matches) = recall.matches_for(group).unwrap();
                    assert_eq!(state.support.query, query.query_id);
                    if let Some(m) = state.matched {
                        assert!(matches.iter().flatten().flatten().any(|s| *s == m));
                    }
                } else {
                    let old = previous
                        .paths
                        .iter()
                        .flatten()
                        .flat_map(|c| c.groups[0].iter().flatten())
                        .find(|p| Some(p.id) == path.parent)
                        .unwrap();
                    assert_eq!(
                        state,
                        previous.correspondences[old.correspondence_slot.unwrap()].unwrap()
                    );
                    inherited += 1;
                }
                if !observed {
                    assert!(!path.correspondence_supported);
                }
            }
        }
    }
    assert!(
        matched > 100 && no_memory > 10 && inherited > 10 && retired > 0 && rejected > 10,
        "matched={matched} none={no_memory} inherited={inherited} retired={retired} rejected={rejected}"
    );
    eprintln!(
        "JOINT_CORRESPONDENCE_OWNER steps=96 matched={matched} no_memory={no_memory} inherited={inherited} retired={retired} rejected={rejected} arena_bytes={} snapshot_bytes={} instruction_bytes={}",
        std::mem::size_of_val(owner.current.correspondences.as_ref()),
        std::mem::size_of::<Snapshot>(),
        std::mem::size_of::<correspondence::Extension>()
    );
}
