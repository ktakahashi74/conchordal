use super::tests::unknown;
use super::*;
use crate::temporal_cognition::{
    joint::proposals::{Boundary, Kind, Known, List},
    section::{
        Activity, Commit, Head,
        form::Transition,
        interpretation::{Change, Input},
    },
};

#[test]
fn section_history_survives_joint_pruning_and_failed_materialization_is_atomic() {
    let group = Handle {
        bus: 0,
        epoch: 1,
        generation: 4,
    };
    let mut owner = State::new(0, 1, 0, 100).unwrap();
    let storage: Vec<_> = [&owner.current.sections, &owner.next.sections]
        .into_iter()
        .map(|s| {
            (
                s.as_ptr(),
                s.iter().map(Section::storage).collect::<Vec<_>>(),
            )
        })
        .collect();
    let head = Head {
        means: [0.; 82],
        deviations: [1.; 82],
        hazard: [0.; 83],
        exits: [[0.; 83]; 3],
    };
    let (mut next_id, mut next_context) = (1_u64, 99_u64);
    let (mut retained, mut reused, mut rejected, mut returns, mut developments) = (0, 0, 0, 0, 0);
    for step in 1..=8 {
        let interval = [(step - 1) * 100, step * 100];
        let observed = step != 4;
        let seconds = interval.map(|t| t as f64 / 1000.);
        let duration = seconds[1] - seconds[0];
        let delta = Activity {
            window: seconds,
            numerators: [0.; 9],
            denominators: [if observed { duration } else { 0. }; 9],
            physical_valid_seconds: [if observed { duration } else { 0. }; 9],
            assignment_seconds: if observed { duration } else { 0. },
            physical_window_seconds: duration,
        };
        let completed:Option<Commit>=observed.then(|| serde_json::from_value(serde_json::json!({
            "epoch":1,"occurrence_id":step,"start":seconds[0],"support_end":seconds[1],"assignment_seconds":duration,
            "membership":1.,"ordering_known":true,"ending_descriptor":[0.,0.,0.,0.,0.,0.],"ending_generation":4,
            "assignment":{"status":"unresolved","cost":null,"supported":false,"search_completed":false,
                "search_covered":false,"search_nonempty":false,"frequency_shift_log2":null,"tempo_shift_log2":null}
        })).unwrap());
        let input = Input {
            group,
            interval,
            rate: 1000,
            observed,
            delta,
            deliveries: &[],
            completed: completed.map(|c| (c, delta, step, 1.)),
            retrieval: [None; 2],
            query: None,
        };
        let previous = owner.snapshot().clone();
        let mut contexts: Vec<_> = previous.paths.iter().flatten().copied().map(Some).collect();
        contexts.push(None);
        let mut rows = Vec::new();
        let mut metadata = Vec::new();
        let mut sources = Vec::new();
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
                        .and_then(|p| p.section_slot)
                        .map(|i| &previous.sections[i]);
                    let mut ids = [0; 4];
                    for (i, id) in ids.iter_mut().enumerate() {
                        if i == 0 && old.is_some() {
                            *id = parent.unwrap().components[3].unwrap();
                        } else {
                            *id = next_id;
                            next_id += 1;
                        }
                    }
                    let returned = None;
                    let list = if let Some(old) = old {
                        if observed {
                            List::owned_active_section(ids, old, interval, 1000, returned, &head)
                                .unwrap()
                        } else {
                            List::build(
                                Kind::Section,
                                &[Known {
                                    id: ids[0],
                                    raw_score: 1.,
                                    stay: true,
                                    boundary: Some(Boundary::Stay),
                                }],
                                false,
                                0.1,
                                5,
                            )
                            .unwrap()
                        }
                    } else if observed {
                        List::build(
                            Kind::Section,
                            &[Known {
                                id: ids[0],
                                raw_score: 1.,
                                stay: false,
                                boundary: Some(Boundary::Exit),
                            }],
                            true,
                            0.1,
                            5,
                        )
                        .unwrap()
                    } else {
                        List::build(Kind::Section, &[], true, 0.1, 5).unwrap()
                    };
                    for (extension, choice) in list.entries.iter().flatten().enumerate() {
                        let section = choice.id.map(|id| {
                            let kind = ids.iter().position(|&i| i == id).unwrap();
                            let change = if let Some(old) = old {
                                match kind {
                                    0 => Change {
                                        transition: Transition::Stay,
                                        context: old.context,
                                        focus: old.focus,
                                    },
                                    2 => unreachable!("no return source in this fixture"),
                                    _ => {
                                        next_context += 1;
                                        Change {
                                            transition: if kind == 1 {
                                                Transition::NewContext
                                            } else {
                                                Transition::Contrast
                                            },
                                            context: next_context,
                                            focus: None,
                                        }
                                    }
                                }
                            } else {
                                next_context += 1;
                                Change {
                                    transition: Transition::NewContext,
                                    context: next_context,
                                    focus: None,
                                }
                            };
                            let source_slot = sources.len();
                            sources.push(section::Source {
                                parent: parent.map(|p| p.id),
                                input,
                                change,
                                returned: None,
                            });
                            section::Extension {
                                id,
                                parent_slot: parent.and_then(|p| p.section_slot),
                                source_slot,
                            }
                        });
                        local.push(LocalExtension {
                            parent: parent.map(|p| p.id),
                            extension: extension as u8,
                            components: [None, None, None, choice.id, None],
                            section,
                            log_transition: choice.log_weight,
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
        let next_path = owner.next_id;
        assert_eq!(
            owner
                .advance(interval, observed, &[group], &inputs, Sources::default())
                .err(),
            Some("missing joint section source")
        );
        assert_eq!(owner.snapshot(), &previous);
        assert_eq!(owner.next_id, next_path);
        let mut foreign = sources.clone();
        for source in &mut foreign {
            source.parent = Some(Handle {
                generation: u64::MAX,
                ..group
            });
        }
        assert_eq!(
            owner
                .advance(
                    interval,
                    observed,
                    &[group],
                    &inputs,
                    Sources {
                        sections: &foreign,
                        ..Sources::default()
                    }
                )
                .err(),
            Some("joint section source belongs to another local parent")
        );
        assert_eq!(owner.snapshot(), &previous);
        assert_eq!(owner.next_id, next_path);
        let mut invalid = sources.clone();
        for source in &mut invalid {
            source.input.delta.numerators[0] = f64::NAN;
        }
        assert_eq!(
            owner
                .advance(
                    interval,
                    observed,
                    &[group],
                    &inputs,
                    Sources {
                        sections: &invalid,
                        ..Sources::default()
                    }
                )
                .err(),
            Some("invalid or overflowing section activity statistic")
        );
        assert_eq!(owner.snapshot(), &previous);
        assert_eq!(owner.next_id, next_path);
        rejected += 1;
        let output = owner
            .advance(
                interval,
                observed,
                &[group],
                &inputs,
                Sources {
                    sections: &sources,
                    ..Sources::default()
                },
            )
            .unwrap();
        let (_, pointers) = storage
            .iter()
            .find(|(p, _)| *p == output.sections.as_ptr())
            .unwrap();
        for context in output.paths.iter().flatten() {
            for path in context.groups[0].iter().flatten() {
                let slot = path.section_slot.unwrap();
                let state = &output.sections[slot];
                assert_eq!(state.storage(), pointers[slot]);
                assert_eq!(state.group, group);
                assert_eq!(state.end, interval[1]);
                let source = rows
                    .iter()
                    .flatten()
                    .find(|r| {
                        r.parent == path.parent
                            && r.section.is_some_and(|s| Some(s.id) == path.components[3])
                    })
                    .unwrap()
                    .section
                    .unwrap();
                let selected = &sources[source.source_slot];
                if let Some(parent_slot) = source.parent_slot {
                    let old = &previous.sections[parent_slot];
                    match selected.change.transition {
                        Transition::Stay | Transition::Development => {
                            assert_eq!(state.start, old.start);
                            assert_eq!(state.context, old.context);
                            reused += 1;
                        }
                        _ => {
                            assert_eq!(state.start, interval[1]);
                        }
                    }
                    returns += usize::from(selected.change.transition == Transition::Return);
                    developments +=
                        usize::from(selected.change.transition == Transition::Development);
                } else {
                    assert_eq!(state.start, interval[0]);
                }
                assert!(state.owners.iter().all(|(_, t, _)| *t <= interval[1]));
                assert!(
                    observed
                        || state.owners == previous.sections[source.parent_slot.unwrap()].owners
                );
                retained += 1;
            }
        }
    }
    assert!(retained > 100 && reused > 50 && rejected == 8 && returns == 0 && developments == 0);
    eprintln!(
        "JOINT_SECTION_OWNER steps=8 retained={retained} stayed={reused} rejected={rejected} returns={returns} developments={developments} arena_bytes={} snapshot_bytes={} instruction_bytes={}",
        std::mem::size_of_val(owner.current.sections.as_ref()),
        std::mem::size_of::<Snapshot>(),
        std::mem::size_of::<section::Extension>()
    );
}
