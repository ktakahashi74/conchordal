use super::tests::unknown;
use super::*;
use crate::temporal_cognition::{
    phrase, recall,
    section::{
        self as marginal, Activity,
        form::Transition,
        interpretation::{Change, Input},
    },
};

#[test]
fn sealed_contexts_follow_actual_selected_matches_through_return_and_development() {
    for masses in [[0.6, 0.2], [0.4, 0.4], [0., 0.]] {
        let cfg = crate::config::TemporalMemoryConfig {
            retention: None,
            candidates: None,
            scales: [1.; 10],
            span_hops: 8,
            episodes: 16,
            query_cadence_ms: 100,
            deadline_ms: 200,
        };
        let mut memory = recall::Recall::new(1, 0, 1000, 100, cfg, None).unwrap();
        let mut cues = marginal::cue::Stream::new(1, 0, 1000, 100, 0, [1.; 10]).unwrap();
        let (mut phrase, base) = marginal::cue::tests::base();
        let group = phrase.groups[0].unwrap().group;
        let mut cache =
            correspondence::Cache::new(group, 1000, marginal::tests::runtime_config(), Some(-3.))
                .unwrap();
        let mut returns = section::Returns::new(group);
        let (mut supported, mut recovered, mut developed, mut rejected) = (0, 0, 0, 0);
        let (mut shared_returns, mut shared_rejected) = (0, 0);
        for step in 1..=100 {
            let end = step * 100;
            let acoustic = phrase::tests::input(step, 0.);
            phrase.end_sample = end;
            let g = phrase.groups[0].as_mut().unwrap();
            g.candidates.fill(None);
            let (credit, start) = if step <= 10 { (30, 0) } else { (40, 1000) };
            g.candidates[0] = Some(phrase::Candidate {
                mass: 1.,
                parent_index: 0,
                completed_foreground: (step == 11).then_some(phrase::Foreground {
                    id: 30,
                    credit: 30,
                    start: 0,
                    heard_end: 1000,
                }),
                foreground: Some(phrase::Foreground {
                    id: credit,
                    credit,
                    start,
                    heard_end: end,
                }),
                ..base
            });
            // These are controlled context assignments; cue sealing and matching are real.
            marginal::cue::tests::project_and_seal(
                &mut cues,
                &acoustic,
                &phrase,
                &[(30, 1000, 70, masses[0]), (30, 1000, 90, masses[1])],
            );
            memory.advance(&acoustic, end, Some(&cues)).unwrap();
            let retained = memory.retained_ids();
            cache
                .refresh(end, memory.matches_for(group), &retained)
                .unwrap();
            returns.refresh(&cache, &memory).unwrap();
            let choices = cache.proposals(None, true, 0.1, &retained).unwrap();
            for state in choices
                .states
                .iter()
                .flatten()
                .filter(|s| s.matched.is_none())
            {
                assert!(returns.for_correspondence(state).is_none());
            }
            let Some((slot, state, proof)) =
                choices.states.iter().enumerate().find_map(|(i, s)| {
                    let state = s.as_ref()?;
                    Some((i, state, returns.for_correspondence(state)?))
                })
            else {
                continue;
            };
            supported += 1;
            assert_eq!(proof.context().context_id, 70);
            assert!((proof.context().mass - masses[0]).abs() < 1e-12);
            let matched = state.matched.unwrap();
            let origin = proof.origin();
            assert_eq!(origin.ongoing_credit, 30);
            assert_eq!((origin.start, origin.end), (0, 1000));
            assert!(origin.sealed_at <= state.support.issued);
            assert!(
                memory
                    .occurrence(matched.episode_id, matched.episode_generation + 1)
                    .is_none()
            );
            assert_eq!(
                origin.occurrence_id,
                memory
                    .occurrence(matched.episode_id, matched.episode_generation)
                    .unwrap()
                    .occurrence_id
            );
            // Exercise ownership in the first six supported windows for each context assignment.
            if supported > 6 {
                continue;
            }
            crate::temporal_cognition::joint::producer::tests::matched_composition(
                &acoustic, &cache, &returns, &retained,
            );
            {
                use shared::producer::{Inputs, Kind, Producer};
                let interval = [end - 100, end];
                let observation = Observation::new(&acoustic, group, interval, 1000, None).unwrap();
                let head = marginal::Head {
                    means: [0.; 82],
                    deviations: [1.; 82],
                    hazard: [0.; 83],
                    exits: [[0.; 83]; 3],
                };
                let mut shared_owner = State::new(group.bus, group.epoch, end - 100, 100).unwrap();
                let mut producer = Producer::new(end - 100);
                let batch = producer
                    .produce(
                        shared_owner.snapshot(),
                        Inputs {
                            interval,
                            rate: 1000,
                            observed: true,
                            groups: &[group],
                            observations: &[Some(&observation)],
                            returns: &[Some(&returns)],
                            retained_episodes: &retained,
                            head: &head,
                            rms_reference: 0.1,
                        },
                    )
                    .unwrap();
                assert!((1..=16).contains(&batch.examined_returns));
                let admitted = batch
                    .entries()
                    .find(|p| p.origin.is_some_and(|o| o.kind == Kind::Retrieved))
                    .unwrap();
                assert!((admitted.raw_score - masses[0]).abs() < 1e-12);
                let witness = admitted.origin.unwrap();
                assert_eq!(
                    witness.episode,
                    Some((matched.episode_id, matched.episode_generation))
                );
                assert_eq!(witness.query, Some(state.support.query));
                assert_eq!(witness.section, Some(70));
                let local = [unknown()];
                let local_refs = [local.as_slice()];
                let inputs: Vec<_> = batch
                    .entries()
                    .map(|p| ContextExtension {
                        parent: p.parent,
                        extension: p.slot,
                        key: p.key,
                        log_transition: p.log_transition,
                        log_potential: 0.,
                        groups: &local_refs,
                    })
                    .collect();
                let model = marginal::tests::runtime_config();
                let next_path = shared_owner.next_id;
                assert_eq!(
                    shared_owner
                        .advance(
                            interval,
                            true,
                            &[group],
                            &inputs,
                            Sources {
                                shared: Some(&batch),
                                observations: &[Some(&observation)],
                                section_config: Some(&model),
                                ..Sources::default()
                            }
                        )
                        .err(),
                    Some("retired shared return target")
                );
                assert_eq!(shared_owner.snapshot().end, end - 100);
                assert_eq!(shared_owner.next_id, next_path);
                shared_rejected += 1;
                let output = shared_owner
                    .advance(
                        interval,
                        true,
                        &[group],
                        &inputs,
                        Sources {
                            shared: Some(&batch),
                            observations: &[Some(&observation)],
                            section_config: Some(&model),
                            retained_episodes: &retained,
                            ..Sources::default()
                        },
                    )
                    .unwrap();
                let (i, p) = output
                    .paths
                    .iter()
                    .enumerate()
                    .find_map(|(i, p)| {
                        p.filter(|p| p.state.admission.is_some_and(|o| o.kind == Kind::Retrieved))
                            .map(|p| (i, p))
                    })
                    .unwrap();
                assert_eq!(p.state.admission, Some(witness));
                let expected = masses[0] / (1. + masses[0]) * (-0.1_f64 / 120.).exp();
                assert!(
                    (output.posterior.contexts[i].unwrap().weight.mass - expected).abs() < 1e-12
                );
                shared_returns += 1;
            }
            let correspondence = correspondence::Extension {
                id: state.id,
                origin: choices.origins[slot].unwrap(),
            };
            let mut owner = State::new(1, 0, end - 200, 100).unwrap();
            let mut next_section = 1;
            for phase in 0..2 {
                let interval = if phase == 0 {
                    [end - 200, end - 100]
                } else {
                    [end - 100, end]
                };
                let window = interval.map(|t| t as f64 / 1000.);
                let dt = window[1] - window[0];
                let input = Input {
                    group,
                    interval,
                    rate: 1000,
                    observed: true,
                    delta: Activity {
                        window,
                        numerators: [0.; 9],
                        denominators: [dt; 9],
                        physical_valid_seconds: [dt; 9],
                        assignment_seconds: dt,
                        physical_window_seconds: dt,
                    },
                    deliveries: &[],
                    completed: None,
                    retrieval: if phase == 1 {
                        cache.retrieval(end, true, &retained).unwrap().values
                    } else {
                        [None; 2]
                    },
                    query: None,
                };
                let previous = owner.snapshot().clone();
                if phase == 1 {
                    let observation =
                        Observation::new(&acoustic, group, interval, 1000, None).unwrap();
                    super::local_section_tests::composed_returns(
                        &previous,
                        &observation,
                        &cache,
                        &returns,
                        &retained,
                        masses[0],
                    );
                }
                let mut contexts: Vec<_> =
                    previous.paths.iter().flatten().copied().map(Some).collect();
                contexts.push(None);
                let (mut rows, mut metadata, mut sources) = (Vec::new(), Vec::new(), Vec::new());
                for context in contexts {
                    let mut parents: Vec<_> = context.map_or_else(Vec::new, |c| {
                        c.groups[0].iter().flatten().copied().map(Some).collect()
                    });
                    parents.push(None);
                    for known in [true, false] {
                        let mut local = Vec::new();
                        for parent in &parents {
                            let old = parent
                                .and_then(|p| p.section_slot)
                                .map(|i| &previous.sections[i]);
                            let source = if phase == 0 {
                                section::Source {
                                    parent: parent.map(|p| p.id),
                                    input,
                                    change: Change {
                                        transition: Transition::NewContext,
                                        context: 70,
                                        focus: None,
                                    },
                                    returned: None,
                                }
                            } else {
                                section::Source::returning(parent.map(|p| p.id), input, old, proof)
                            };
                            let source_slot = sources.len();
                            sources.push(source);
                            local.push(LocalExtension {
                                parent: parent.map(|p| p.id),
                                extension: 0,
                                components: [
                                    None,
                                    None,
                                    None,
                                    Some(next_section),
                                    (phase == 1).then_some(state.id),
                                ],
                                section: Some(section::Extension {
                                    id: next_section,
                                    parent_slot: parent.and_then(|p| p.section_slot),
                                    source_slot,
                                }),
                                correspondence: (phase == 1).then_some(correspondence),
                                log_transition: 0.9_f64.ln(),
                                ..unknown()
                            });
                            next_section += 1;
                            local.push(LocalExtension {
                                parent: parent.map(|p| p.id),
                                extension: 1,
                                log_transition: 0.1_f64.ln(),
                                ..unknown()
                            });
                        }
                        rows.push(local);
                        metadata.push((context.map(|c| c.id), known));
                    }
                }
                let refs: Vec<[&[LocalExtension]; 1]> =
                    rows.iter().map(|r| [r.as_slice()]).collect();
                let inputs: Vec<_> = metadata
                    .iter()
                    .enumerate()
                    .map(|(i, &(parent, known))| ContextExtension {
                        parent,
                        extension: u8::from(!known),
                        key: known.then_some(42),
                        log_transition: if known { 0.9_f64.ln() } else { 0.1_f64.ln() },
                        log_potential: 0.,
                        groups: &refs[i],
                    })
                    .collect();
                if phase == 1 {
                    let source = sources[0];
                    assert!(source.validate_return(None).is_err());
                    assert!(source.validate_return(Some((state, false))).is_err());
                    let mut missing = source;
                    missing.input.observed = false;
                    assert!(missing.validate_return(Some((state, true))).is_err());
                    let mut stale = source;
                    stale.input.interval[1] += 100;
                    assert!(stale.validate_return(Some((state, true))).is_err());
                    let mut different = *state;
                    different.support.query += 1;
                    assert!(source.validate_return(Some((&different, true))).is_err());
                    let mut alias = *state;
                    alias.id += 999;
                    assert!(source.validate_return(Some((&alias, true))).is_ok());
                    for defect in 0..4 {
                        let mut invalid = sources.clone();
                        for source in &mut invalid {
                            match defect {
                                0 => source.returned = None,
                                1 => source.input.query.as_mut().unwrap().query_id += 1,
                                2 => source.change.context = 90,
                                _ => source.change.focus = None,
                            }
                        }
                        let next_id = owner.next_id;
                        assert!(
                            owner
                                .advance(
                                    interval,
                                    true,
                                    &[group],
                                    &inputs,
                                    Sources {
                                        sections: &invalid,
                                        correspondences: &[(phase == 1).then_some(&cache)],
                                        retained_episodes: &retained,
                                        ..Sources::default()
                                    }
                                )
                                .is_err()
                        );
                        assert_eq!(owner.snapshot(), &previous);
                        assert_eq!(owner.next_id, next_id);
                        rejected += 1;
                    }
                    assert_eq!(
                        owner
                            .advance(
                                interval,
                                true,
                                &[group],
                                &inputs,
                                Sources {
                                    sections: &sources,
                                    correspondences: &[(phase == 1).then_some(&cache)],
                                    ..Sources::default()
                                }
                            )
                            .err(),
                        Some("retired correspondence target")
                    );
                    assert_eq!(owner.snapshot(), &previous);
                    rejected += 1;
                }
                let result = owner
                    .advance(
                        interval,
                        true,
                        &[group],
                        &inputs,
                        Sources {
                            sections: &sources,
                            correspondences: &[(phase == 1).then_some(&cache)],
                            retained_episodes: &retained,
                            ..Sources::default()
                        },
                    )
                    .unwrap();
                if phase == 1 {
                    for p in result
                        .paths
                        .iter()
                        .flatten()
                        .flat_map(|c| c.groups[0].iter().flatten())
                    {
                        let section = &result.sections[p.section_slot.unwrap()];
                        assert_eq!(section.context, 70);
                        assert_eq!(section.end, end);
                        assert_eq!(section.focus, Some(matched));
                        assert_eq!(section.query, memory.matches_for(group).map(|(q, _)| q));
                        assert!(section.owners.is_empty());
                        if section.relation == marginal::form::Relation::Development {
                            assert_eq!(section.start, end - 200);
                            developed += 1;
                        } else {
                            assert_eq!(section.start, end - 100);
                            assert!(matches!(
                                section.relation,
                                marginal::form::Relation::Recurrence
                                    | marginal::form::Relation::TransformedRecurrence
                            ));
                            recovered += 1;
                        }
                    }
                }
            }
        }
        if masses[0] == 0. {
            assert_eq!((supported, recovered, developed, rejected), (0, 0, 0, 0));
            assert!(
                cache.fresh_supported > 0,
                "actual matches without context provenance"
            );
        } else {
            assert!(supported >= 6 && recovered > 0 && developed > 0);
        }
        println!(
            "JOINT_SHARED_RETURNS context_mass={} retained={shared_returns} rejected={shared_rejected}",
            masses[0]
        );
        println!(
            "JOINT_SECTION_RETURNS context_mass={} steps=100 supported={supported} recovered={recovered} developed={developed} rejected={rejected}",
            masses[0]
        );
    }
}
