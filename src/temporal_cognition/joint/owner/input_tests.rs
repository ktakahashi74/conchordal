use super::tests::unknown;
use super::*;
use crate::temporal_cognition::{phrase, section as marginal};

#[test]
fn acoustic_inputs_and_selected_phrases_survive_unknown_sections_and_local_missing_windows() {
    let group = grouping::tests::group();
    let mut estimator = grouping::tests::estimator();
    let mut inventory = grouping::tests::inventory();
    let mut cache = grouping::Cache::new(group, Some(1.)).unwrap();
    let mut owner = State::new(group.bus, group.epoch, 0, 500).unwrap();
    let config = marginal::tests::runtime_config();
    let mut acoustic_cache = acoustics::Cache::new(group, 0, 1000, 500).unwrap();
    let arenas = [
        owner.current.phrase_activity.as_ptr(),
        owner.next.phrase_activity.as_ptr(),
    ];
    let (mut next_phrase, mut next_section) = (1, 1);
    let (mut accumulated, mut completed, mut unknown_section, mut grouped, mut rejected) =
        (0, 0, 0, 0, 0);
    for step in 1..=12 {
        let interval = [(step - 1) * 500, step * 500];
        let mut acoustic = phrase::tests::input(step * 5, 0.);
        acoustic.group_handles[0] = Some(group);
        acoustic.retained_groups[0] = Some(group);
        acoustic.features[0].as_mut().unwrap().raw.group = group;
        acoustic.eligible[0] = ![4, 9].contains(&step);
        acoustic.assignment.rows[0].as_mut().unwrap().weights[0] = 0.4;
        let observation = acoustic_cache.advance(&acoustic, interval).unwrap();
        if observation.observed {
            grouping::tests::deliver(&mut estimator, interval[1] - 32);
            inventory.refresh(&estimator).unwrap();
        }
        let view = inventory.snapshot();
        cache
            .refresh(interval[1], view.as_ref().map(|v| (v, &estimator)))
            .unwrap();
        let previous = owner.snapshot().clone();
        let mut contexts: Vec<_> = previous.paths.iter().flatten().copied().map(Some).collect();
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
                    let old_phrase = parent
                        .and_then(|p| p.phrase_slot)
                        .and_then(|i| previous.phrases[i]);
                    let old_grouping = parent
                        .and_then(|p| p.grouping_slot)
                        .and_then(|i| previous.groupings[i]);
                    let choices = cache
                        .proposals(old_grouping, observation.observed, 0.5)
                        .unwrap();
                    let selected = choices
                        .states
                        .iter()
                        .enumerate()
                        .find_map(|(i, s)| s.map(|s| (i, s)));
                    let grouping = selected.map(|(i, s)| grouping::Extension {
                        id: s.id,
                        origin: choices.origins[i].unwrap(),
                    });
                    if observation.observed || old_phrase.is_some() {
                        let kind = if old_phrase.is_some() && observation.observed && step % 3 == 0
                        {
                            Some(phrase::Exit::New)
                        } else if old_phrase.is_some() && step == 5 {
                            Some(phrase::Exit::Reinterpret)
                        } else {
                            None
                        };
                        let id = if kind.is_none() && old_phrase.is_some() {
                            old_phrase.unwrap().id
                        } else {
                            let id = next_phrase;
                            next_phrase += 1;
                            id
                        };
                        for section_known in [true, false] {
                            let old_section = parent.and_then(|p| p.section_slot);
                            if section_known && !observation.observed && old_section.is_none() {
                                continue;
                            }
                            let section = section_known.then(|| {
                                let (id, context, transition) = if let Some(i) = old_section {
                                    (
                                        parent.unwrap().components[3].unwrap(),
                                        previous.sections[i].context,
                                        marginal::form::Transition::Stay,
                                    )
                                } else {
                                    let id = next_section;
                                    next_section += 1;
                                    (id, 1, marginal::form::Transition::NewContext)
                                };
                                let source_slot = sources.len();
                                sources.push(section::Source {
                                    parent: parent.map(|p| p.id),
                                    returned: None,
                                    input: marginal::interpretation::Input {
                                        group,
                                        interval,
                                        rate: 1000,
                                        observed: observation.observed,
                                        delta: observation.delta,
                                        deliveries: &[],
                                        completed: None,
                                        retrieval: [None; 2],
                                        query: None,
                                    },
                                    change: marginal::interpretation::Change {
                                        transition,
                                        context,
                                        focus: None,
                                    },
                                });
                                section::Extension {
                                    id,
                                    parent_slot: old_section,
                                    source_slot,
                                }
                            });
                            local.push(LocalExtension {
                                parent: parent.map(|p| p.id),
                                extension: u8::from(!section_known),
                                components: [
                                    None,
                                    grouping.map(|g| g.id),
                                    Some(id),
                                    section.map(|s| s.id),
                                    None,
                                ],
                                grouping,
                                section,
                                phrase: Some(PhraseExtension {
                                    parent_slot: parent.and_then(|p| p.phrase_slot),
                                    kind,
                                    id,
                                }),
                                log_transition: if !observation.observed && old_section.is_none() {
                                    0.9_f64.ln()
                                } else {
                                    0.45_f64.ln()
                                },
                                // A missing group's observation potential must be masked even while another group keeps the bus observed.
                                log_potential: if observation.observed { 0. } else { 123. },
                                ..unknown()
                            });
                        }
                        local.push(LocalExtension {
                            parent: parent.map(|p| p.id),
                            extension: 2,
                            log_transition: 0.1_f64.ln(),
                            ..unknown()
                        });
                    } else {
                        local.push(LocalExtension {
                            parent: parent.map(|p| p.id),
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
                log_transition: if known { 0.9_f64.ln() } else { 0.1_f64.ln() },
                log_potential: 0.,
                groups: &refs[i],
            })
            .collect();
        let mut wrong = observation;
        wrong.group.generation += 1;
        assert!(
            owner
                .advance(
                    interval,
                    true,
                    &[group],
                    &inputs,
                    Sources {
                        observations: &[Some(&wrong)],
                        section_config: Some(&config),
                        sections: &sources,
                        groupings: &[Some(&cache)],
                        ..Sources::default()
                    }
                )
                .is_err()
        );
        assert_eq!(owner.snapshot(), &previous);
        rejected += 1;
        if observation.observed {
            let mut broken = observation;
            broken.delta.assignment_seconds = -1.;
            let mut broken_sources = sources.clone();
            for source in &mut broken_sources {
                source.input.delta = broken.delta;
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
                            observations: &[Some(&broken)],
                            section_config: Some(&config),
                            sections: &broken_sources,
                            groupings: &[Some(&cache)],
                            ..Sources::default()
                        }
                    )
                    .is_err()
            );
            assert_eq!(owner.snapshot(), &previous);
            assert_eq!(owner.next_id, next_id);
            rejected += 1;
        }
        let mut altered = observation;
        altered.raw.as_mut().unwrap().raw.values[2] = Some(100.);
        assert_eq!(
            owner
                .advance(
                    interval,
                    true,
                    &[group],
                    &inputs,
                    Sources {
                        observations: &[Some(&altered)],
                        acoustics: &[Some(&acoustic_cache)],
                        section_config: Some(&config),
                        sections: &sources,
                        groupings: &[Some(&cache)],
                        ..Sources::default()
                    }
                )
                .err(),
            Some("joint acoustic cache differs from original observation")
        );
        assert_eq!(owner.snapshot(), &previous);
        rejected += 1;
        let out = owner
            .advance(
                interval,
                true,
                &[group],
                &inputs,
                Sources {
                    observations: &[Some(&observation)],
                    acoustics: &[Some(&acoustic_cache)],
                    section_config: Some(&config),
                    sections: &sources,
                    groupings: &[Some(&cache)],
                    ..Sources::default()
                },
            )
            .unwrap();
        assert!(arenas.contains(&out.phrase_activity.as_ptr()));
        for context in out.paths.iter().flatten() {
            let summary = context.state.groups[0].unwrap();
            assert_eq!(summary.group, group);
            assert_eq!(summary.phrase[0], 0.);
            assert!((summary.phrase[1] - summary.represented_mass).abs() < 1e-12);
            assert_eq!(summary.support.is_some(), observation.observed);
            assert!(summary.sections.iter().flatten().all(|s| s.context == 1));
            assert!(
                summary
                    .sections
                    .iter()
                    .flatten()
                    .map(|s| s.mass)
                    .sum::<f64>()
                    <= summary.represented_mass + 1e-12
            );
            assert!(
                (summary.represented_mass + summary.explicit_unknown + summary.pruned_mass - 1.)
                    .abs()
                    < 1e-12
            );
        }

        for path in out
            .paths
            .iter()
            .flatten()
            .flat_map(|c| c.groups[0].iter().flatten())
        {
            let slot = path.phrase_slot.unwrap();
            let phrase = out.phrases[slot].unwrap();
            let foreground = phrase.foreground.unwrap();
            if observation.observed && foreground.start < foreground.heard_end {
                assert_eq!(
                    phrase.ending,
                    Some(acoustic_cache.ending(foreground).unwrap())
                );
            }
            if let Some(e) = phrase.completed_ending {
                assert!(e.end <= interval[0]);
            }

            let mut audit = out.phrase_activity[slot];
            let completion = audit
                .advance(
                    &observation,
                    Phrase {
                        foreground: None,
                        completed_foreground: Some(foreground),
                        completed_ending: None,
                    },
                    false,
                    &config,
                )
                .unwrap()
                .unwrap();
            let valid = (1..=step)
                .filter(|t| {
                    ![4, 9].contains(t)
                        && t * 500 > foreground.start
                        && t * 500 <= foreground.heard_end
                })
                .count();
            assert!(
                (completion.1.assignment_seconds - valid as f64 * 0.04).abs() < 1e-12,
                "step={step}, foreground={foreground:?}"
            );
            accumulated += 1;
            completed += usize::from(out.phrase_activity[slot].sequence > 0);
            unknown_section +=
                usize::from(path.section_slot.is_none() && completion.1.assignment_seconds > 0.);
            if let Some(i) = path.section_slot {
                assert_eq!(out.sections[i].end, interval[1]);
                if path.grouping_supported
                    && out.groupings[path.grouping_slot.unwrap()]
                        .unwrap()
                        .proposal
                        .is_some()
                {
                    grouped += 1;
                }
            }
        }
    }
    assert!(accumulated > 20 && completed > 0 && unknown_section > 0 && grouped > 0);
    println!(
        "JOINT_SECTION_INPUTS steps=12 accumulated={accumulated} completed={completed} unknown_section={unknown_section} grouped={grouped} rejected={rejected} span_bytes={}",
        std::mem::size_of::<Span>()
    );
}
