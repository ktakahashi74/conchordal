use super::tests::unknown;
use super::*;
use crate::temporal_cognition::{
    joint::{
        proposals::{Composer, Kind, List},
        section::producer::{self, Inputs},
    },
    phrase,
    section::{self as marginal, form::Transition},
};

// The caller supplies real matcher/cache evidence and a controlled prior from the return assay.
pub(super) fn composed_returns(
    previous: &Snapshot,
    observation: &Observation,
    cache: &correspondence::Cache,
    returns: &section::Returns,
    retained: &[(u64, u64)],
    context_mass: f64,
) {
    let group = observation.group;
    let interval = observation.interval;
    let mut owner = State::new(group.bus, group.epoch, interval[0], 100).unwrap();
    owner.current.clone_from(&Box::new(previous.clone()));
    owner.next_id = previous
        .paths
        .iter()
        .flatten()
        .flat_map(|c| {
            std::iter::once(c.id.generation)
                .chain(c.groups.iter().flatten().flatten().map(|p| p.id.generation))
        })
        .max()
        .unwrap_or(0)
        + 1;
    owner.maximum_group_generation = group.generation;
    let mut composer = Composer::new();
    let head = marginal::Head {
        means: [0.; 82],
        deviations: [1.; 82],
        hazard: [0.; 83],
        exits: [[0.; 83]; 3],
    };
    let mut id = 10_000;
    let mut sources = Vec::new();
    let mut rows = Vec::new();
    let mut metadata = Vec::new();
    let mut generated = 0;
    let mut excluded = 0;
    let mut return_tuples = 0;
    for context in previous
        .paths
        .iter()
        .flatten()
        .map(Some)
        .chain(std::iter::once(None))
    {
        let parents: Vec<_> = context
            .into_iter()
            .flat_map(|c| c.groups[0].iter().flatten())
            .map(Some)
            .chain(std::iter::once(None))
            .collect();
        let mut local = Vec::new();
        for parent in parents {
            let old_correspondence = parent
                .and_then(|p| p.correspondence_slot)
                .and_then(|i| previous.correspondences[i]);
            let mut choices = cache
                .proposals(old_correspondence, true, 0.1, retained)
                .unwrap();
            let ids = [
                parent.and_then(|p| p.components[3]).unwrap_or(id),
                id + 1,
                id + 2,
                id + 3,
            ];
            let output = producer::build(Inputs {
                snapshot: previous,
                parent,
                observation,
                correspondences: &choices,
                cache,
                retained_episodes: retained,
                returns,
                head: &head,
                ids,
                new_contexts: [id + 4, id + 5],
                rms_reference: 0.1,
            })
            .unwrap();
            id += 10;
            assert!(output.examined_returns > 0 && output.examined_returns <= 16);
            let old = parent
                .and_then(|p| p.section_slot)
                .map(|i| &previous.sections[i]);
            let mut missing = *observation;
            missing.observed = false;
            let missing_output = producer::build(Inputs {
                snapshot: previous,
                parent,
                observation: &missing,
                correspondences: &choices,
                cache,
                retained_episodes: retained,
                returns,
                head: &head,
                ids,
                new_contexts: [id + 4, id + 5],
                rms_reference: 0.1,
            })
            .unwrap();
            assert_eq!(missing_output.examined_returns, 0);
            assert_eq!(missing_output.list.len, if old.is_some() { 2 } else { 1 });
            if old.is_some() {
                let stay = missing_output
                    .list
                    .entries
                    .iter()
                    .flatten()
                    .find(|c| c.stay)
                    .unwrap();
                assert_eq!(stay.id, Some(ids[0]));
                assert!(
                    (stay.log_weight.exp() - 2_f64.powf(-0.1) * (-0.1_f64 / 120.).exp()).abs()
                        < 1e-12
                );
                assert_eq!(
                    missing_output.sources[0].unwrap().1.change.transition,
                    Transition::Stay
                );
            }
            for (bad_ids, bad_contexts) in [(ids, [70, id + 5]), ([ids[0]; 4], [id + 4, id + 5])] {
                assert!(
                    producer::build(Inputs {
                        snapshot: previous,
                        parent,
                        observation,
                        correspondences: &choices,
                        cache,
                        retained_episodes: retained,
                        returns,
                        head: &head,
                        ids: bad_ids,
                        new_contexts: bad_contexts,
                        rms_reference: 0.1,
                    })
                    .is_err()
                );
            }
            let returned = output
                .sources
                .iter()
                .flatten()
                .find(|(_, s)| s.returned.is_some())
                .unwrap();
            assert_eq!(returned.1.change.context, 70);
            assert_eq!(
                returned.1.change.transition,
                if old.is_some_and(|p| p.context == 70) {
                    Transition::Development
                } else {
                    Transition::Return
                }
            );
            let p_return = output
                .list
                .entries
                .iter()
                .flatten()
                .find(|c| c.id == Some(returned.0))
                .unwrap();
            let expected = if old.is_some() {
                let stay = 2_f64.powf(-0.1);
                let exit = (1. - stay) / 3.;
                exit * context_mass / (stay + exit * (2. + context_mass))
            } else {
                context_mass / (1. + context_mass)
            } * (-0.1_f64 / 120.).exp();
            assert!((p_return.log_weight.exp() - expected).abs() < 1e-12);
            let lists = [
                List::build(Kind::Articulation, &[], true, 0.1, 8).unwrap(),
                List::build(Kind::Grouping, &[], true, 0.1, 19).unwrap(),
                List::build(Kind::Phrase, &[], true, 0.1, 6).unwrap(),
                output.list,
                choices.list,
            ];
            let legal = output.compatibility(&choices);
            let fresh = choices.feature_supported;
            choices.feature_supported.fill(false);
            let stale = output.compatibility(&choices);
            let return_slot = output
                .list
                .entries
                .iter()
                .position(|c| c.is_some_and(|c| c.id == Some(returned.0)))
                .unwrap();
            assert!(
                stale.section_correspondence[return_slot][..choices.list.len]
                    .iter()
                    .all(|v| !v)
            );
            choices.feature_supported = fresh;
            excluded += legal.section_correspondence[..output.list.len]
                .iter()
                .map(|r| r[..choices.list.len].iter().filter(|v| !**v).count())
                .sum::<usize>();
            let tuples = composer.compose(&lists, &legal).unwrap();
            assert!(
                tuples.priority_evaluations <= 86
                    && tuples.heap_pops <= 16
                    && tuples.correspondence_checks <= 95
            );
            assert!(
                (tuples
                    .tuples
                    .iter()
                    .flatten()
                    .map(|t| t.log_transition.exp())
                    .sum::<f64>()
                    - 1.)
                    .abs()
                    < 1e-12
            );
            for (extension, t) in tuples.tuples.iter().flatten().enumerate() {
                let source = output
                    .sources
                    .iter()
                    .flatten()
                    .find(|(sid, _)| Some(*sid) == t.ids[3])
                    .map(|(_, s)| *s);
                let selected = choices
                    .states
                    .iter()
                    .enumerate()
                    .find_map(|(i, c)| c.filter(|c| Some(c.id) == t.ids[4]).map(|c| (i, c)));
                if let Some(source) = source {
                    source
                        .validate_return(
                            selected
                                .as_ref()
                                .map(|(i, c)| (c, choices.feature_supported[*i])),
                        )
                        .unwrap();
                    return_tuples += usize::from(source.returned.is_some());
                }
                let section = source.map(|s| {
                    let slot = sources.len();
                    sources.push(s);
                    section::Extension {
                        id: t.ids[3].unwrap(),
                        parent_slot: parent.and_then(|p| p.section_slot),
                        source_slot: slot,
                    }
                });
                local.push(LocalExtension {
                    parent: parent.map(|p| p.id),
                    extension: extension as u8,
                    components: t.ids,
                    section,
                    correspondence: selected.map(|(i, c)| correspondence::Extension {
                        id: c.id,
                        origin: choices.origins[i].unwrap(),
                    }),
                    log_transition: t.log_transition,
                    ..unknown()
                });
                generated += 1;
            }
        }
        rows.push(local);
        metadata.push(context);
    }
    assert!(excluded > 0 && return_tuples > 0);
    let references: Vec<_> = rows.iter().map(|r| [r.as_slice()]).collect();
    let inputs: Vec<_> = metadata
        .iter()
        .enumerate()
        .flat_map(|(i, context)| {
            [true, false].map(|known| ContextExtension {
                parent: context.map(|c| c.id),
                extension: u8::from(!known),
                key: known.then_some(42),
                log_transition: if known { 0.9_f64.ln() } else { 0.1_f64.ln() },
                log_potential: 0.,
                groups: &references[i],
            })
        })
        .collect();
    let model = marginal::tests::runtime_config();
    let mut altered = sources.clone();
    for source in &mut altered {
        source.input.retrieval[0] = Some(100.);
    }
    let before_id = owner.next_id;
    assert_eq!(
        owner
            .advance(
                interval,
                true,
                &[group],
                &inputs,
                Sources {
                    sections: &altered,
                    correspondences: &[Some(cache)],
                    retained_episodes: retained,
                    observations: &[Some(observation)],
                    section_config: Some(&model),
                    ..Sources::default()
                }
            )
            .err(),
        Some("joint section retrieval differs from original query projection")
    );
    assert_eq!(owner.snapshot(), previous);
    assert_eq!(owner.next_id, before_id);
    let result = owner
        .advance(
            interval,
            true,
            &[group],
            &inputs,
            Sources {
                sections: &sources,
                correspondences: &[Some(cache)],
                retained_episodes: retained,
                observations: &[Some(observation)],
                section_config: Some(&model),
                ..Sources::default()
            },
        )
        .unwrap();
    let mut retained_returns = 0;
    for path in result
        .paths
        .iter()
        .flatten()
        .flat_map(|p| p.groups[0].iter().flatten())
    {
        let Some(i) = path.section_slot else {
            continue;
        };
        let s = &result.sections[i];
        assert_eq!(
            &s.values[80..],
            &cache.retrieval(interval[1], true, retained).unwrap().values
        );
        if matches!(
            s.relation,
            marginal::form::Relation::Recurrence
                | marginal::form::Relation::TransformedRecurrence
                | marginal::form::Relation::Development
        ) {
            assert_eq!(s.context, 70);
            let c = result.correspondences[path.correspondence_slot.unwrap()].unwrap();
            assert!(path.correspondence_supported);
            assert_eq!(s.focus, c.matched);
            assert_eq!(s.query.unwrap().query_id, c.support.query);
            retained_returns += 1;
        }
    }
    assert!(retained_returns > 0);
    eprintln!(
        "JOINT_LOCAL_SECTIONS generated={generated} incompatible={excluded} return_tuples={return_tuples} retained_returns={retained_returns}"
    );
}

#[test]
fn section_source_production_masks_missing_and_quiet_recovery_and_rejects_aliases() {
    let acoustic = phrase::tests::input(1, 0.);
    let group = acoustic.group_handles[0].unwrap();
    let mut o = Observation::new(&acoustic, group, [0, 100], 1000, None).unwrap();
    let owner = State::new(group.bus, group.epoch, 0, 100).unwrap();
    let mut cache =
        correspondence::Cache::new(group, 1000, marginal::tests::runtime_config(), None).unwrap();
    cache.refresh(100, None, &[]).unwrap();
    let choices = cache.proposals(None, true, 0.1, &[]).unwrap();
    let returns = section::Returns::new(group);
    let head = marginal::Head {
        means: [0.; 82],
        deviations: [1.; 82],
        hazard: [0.; 83],
        exits: [[0.; 83]; 3],
    };
    for variant in 0..6 {
        let mut input = o;
        match variant {
            1 => input.raw.as_mut().unwrap().raw.values[2] = Some(-30.),
            2 => input.observed = false,
            3 => input.delta.assignment_seconds = 0.,
            4 => input.raw = None,
            5 => input.delta.assignment_seconds = f64::NAN,
            _ => {}
        }
        let result = producer::build(Inputs {
            snapshot: owner.snapshot(),
            parent: None,
            observation: &input,
            correspondences: &choices,
            cache: &cache,
            retained_episodes: &[],
            returns: &returns,
            head: &head,
            ids: [1, 2, 3, 4],
            new_contexts: [5, 6],
            rms_reference: 0.1,
        });
        if variant >= 4 {
            assert!(result.is_err());
            continue;
        }
        let result = result.unwrap();
        assert_eq!(result.list.len, if variant == 0 { 2 } else { 1 });
        assert_eq!(result.examined_returns, 0);
        if variant != 0 {
            assert_eq!(result.list.entries[0].unwrap().id, None);
        }
    }
    o.raw.as_mut().unwrap().raw.start = 101;
    assert!(
        producer::build(Inputs {
            snapshot: owner.snapshot(),
            parent: None,
            observation: &o,
            correspondences: &choices,
            cache: &cache,
            retained_episodes: &[],
            returns: &returns,
            head: &head,
            ids: [1, 2, 3, 4],
            new_contexts: [5, 6],
            rms_reference: 0.1
        })
        .is_err()
    );
}
