use super::*;

fn handle(generation: u64) -> Handle {
    Handle {
        bus: 0,
        epoch: 3,
        generation,
    }
}

fn articulation(state: u64) -> Articulation {
    use crate::temporal_cognition::gesture::{Run, State};
    Articulation {
        state: [
            State::Attack,
            State::Continuation,
            State::Release,
            State::Gap,
        ][state as usize],
        entered: 0,
        run: Run::default(),
    }
}

#[test]
fn actual_articulation_proposals_survive_joint_ownership_into_the_next_acoustic_input() {
    use crate::temporal_cognition::{
        features::RawDescriptor, gesture::articulation::Input, joint::proposals::List,
    };
    let mut config = crate::config::TemporalGestureConfig {
        rms_reference: 0.1,
        means: [0.; 5],
        deviations: [1.; 5],
        coefficients: [[[0.; 11]; 4]; 4],
    };
    // Retain release-to-gap histories despite the deliberately small local beam.
    config.coefficients[0][2][0] = 20.;
    config.coefficients[1][2][0] = 20.;
    config.coefficients[2][3][0] = 20.;
    let mut owner = State::new(0, 3, 0, 64).unwrap();
    let group = handle(1);
    let mut retained = 0;
    let mut preserved = 0;
    let mut release_seen = false;
    let mut gap_seen = false;
    for step in 0..6 {
        let [start, end] = [step * 64, (step + 1) * 64];
        let observed = step != 3;
        let raw = RawDescriptor {
            group,
            start,
            end,
            known_samples: 64,
            source_start: start.saturating_sub(128),
            source_end: end,
            available_end: end,
            values: [
                None,
                None,
                Some(if step == 2 { -20. } else { -2. }),
                Some(0.1),
                Some(0.),
                Some(0.),
                None,
                None,
                None,
                None,
            ],
        };
        let input = Input::new(
            group,
            observed.then_some(&raw),
            None,
            [start, end],
            8000,
            &config,
        )
        .unwrap();
        let previous = owner.snapshot().clone();
        let mut contexts: Vec<_> = previous.paths.iter().flatten().copied().map(Some).collect();
        contexts.push(None);
        let mut metadata = Vec::new();
        let mut rows = Vec::new();
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
                    let articulation = parent.and_then(|p| p.articulation);
                    let (list, children) =
                        List::observed_articulation(articulation, &input, &config).unwrap();
                    for (extension, choice) in list.entries[..list.len].iter().flatten().enumerate()
                    {
                        local.push(LocalExtension {
                            parent: parent.map(|p| p.id),
                            extension: extension as u8,
                            components: [choice.id, None, None, None, None],
                            articulation: choice.id.and_then(|id| children[id as usize]),
                            phrase: None,
                            correspondence: None,
                            grouping: None,
                            section: None,
                            log_transition: choice.log_weight,
                            log_potential: 0.,
                        });
                    }
                }
                rows.push(local);
                metadata.push((context.map(|c| c.id), known));
            }
        }
        let group_rows: Vec<[&[LocalExtension]; 1]> = rows.iter().map(|r| [r.as_slice()]).collect();
        let extensions: Vec<_> = metadata
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
                groups: &group_rows[i],
            })
            .collect();
        let output = owner
            .advance(
                [start, end],
                observed,
                &[group],
                &extensions,
                Sources::default(),
            )
            .unwrap();
        for context in output.paths.iter().flatten() {
            for path in context.groups[0].iter().flatten() {
                let state = path.articulation.unwrap();
                assert_eq!(path.components[0], Some(state.state as u64));
                assert!(state.entered <= end && state.run.observed_end <= end);
                if let Some(parent) = path.parent {
                    let old = previous
                        .paths
                        .iter()
                        .flatten()
                        .flat_map(|c| c.groups[0].iter().flatten())
                        .find(|p| p.id == parent)
                        .unwrap()
                        .articulation
                        .unwrap();
                    if state.state == old.state {
                        assert_eq!(state.entered, old.entered);
                        if !observed {
                            assert_eq!(state.run.observed_end, old.run.observed_end);
                        }
                        preserved += 1;
                    }
                }
                if !observed {
                    assert!(state.run.censored);
                }
                for support in [state.run.attack, state.run.release, state.run.gap]
                    .into_iter()
                    .flatten()
                {
                    assert!(
                        support.source_start <= support.start
                            && support.end <= support.source_end
                            && support.source_end <= support.available
                            && support.available <= end
                    );
                    assert_ne!(
                        support.available, 256,
                        "missing frame cannot fabricate an endpoint"
                    );
                }
                release_seen |= state.run.release.is_some();
                gap_seen |= state.run.gap.is_some();
                retained += 1;
            }
        }
    }
    assert!(
        retained > 30 && preserved > 10 && release_seen && gap_seen,
        "retained={retained} preserved={preserved} release={release_seen} gap={gap_seen}"
    );
    eprintln!(
        "JOINT_ARTICULATION_OWNERSHIP steps=6 retained_checks={retained} stay_checks={preserved}"
    );
}

pub(super) fn unknown() -> LocalExtension {
    LocalExtension {
        parent: None,
        extension: 0,
        components: [None; 5],
        articulation: None,
        phrase: None,
        correspondence: None,
        grouping: None,
        section: None,
        log_transition: 0.,
        log_potential: 0.,
    }
}

#[test]
fn retained_priors_and_owned_paths_match_multistep_probability_enumeration() {
    let mut state = State::new(0, 3, 0, 64).unwrap();
    let mut maximum_error = 0_f64;
    let allocation = state.pairs.as_ptr();
    let allocations = [
        state.current.as_ref() as *const Snapshot,
        state.next.as_ref() as *const Snapshot,
    ];
    for step in 0..10 {
        let observed = step != 3;
        let groups = if step == 5 {
            vec![]
        } else if step > 5 {
            vec![handle(4)]
        } else if step % 2 == 0 {
            vec![handle(1), handle(2)]
        } else {
            vec![handle(2), handle(1)]
        };
        let previous = state.snapshot().clone();
        let mut parents: Vec<_> = previous
            .paths
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.map(|p| (Some(p.id), i)))
            .collect();
        parents.push((None, 7));
        let mut metadata = Vec::new();
        let mut inputs = Vec::new();
        let mut expected_local = Vec::new();
        let mut joint_weights = Vec::new();
        for (parent, parent_slot) in parents {
            let prior = if parent.is_some() {
                previous.posterior.contexts[parent_slot]
                    .unwrap()
                    .weight
                    .mass
            } else {
                previous.posterior.explicit_unknown + previous.posterior.pruned_mass
            };
            for extension in 0..2 {
                let known = extension == 0 && (observed || parent.is_some());
                if extension == 0 && !known {
                    continue;
                }
                let transition = if parent.is_none() && !observed {
                    1.
                } else if extension == 0 {
                    0.8
                } else {
                    0.2
                };
                let potential: f64 = if known {
                    0.1 * (parent_slot + 1) as f64
                } else {
                    -0.2
                };
                metadata.push((parent, parent_slot, extension, known, transition, potential));
                let mut input_groups = Vec::new();
                let mut reference_groups = Vec::new();
                let mut joint = prior * transition * if observed { potential.exp() } else { 1. };
                for (group_index, group) in groups.iter().enumerate() {
                    let old_group = previous.groups.iter().position(|g| *g == Some(*group));
                    let old = parent
                        .zip(old_group)
                        .map(|(_, g)| previous.paths[parent_slot].unwrap().groups[g]);
                    let posterior = parent
                        .zip(old_group)
                        .map(|(_, g)| previous.posterior.contexts[parent_slot].unwrap().groups[g]);
                    let mut local_parents: Vec<_> = old.map_or_else(Vec::new, |paths| {
                        paths
                            .iter()
                            .enumerate()
                            .filter_map(|(i, p)| p.map(|p| (Some(p.id), i)))
                            .collect()
                    });
                    local_parents.push((None, 15));
                    let mut rows = Vec::new();
                    let mut reference = Vec::new();
                    for (local_parent, slot) in local_parents {
                        let local_prior = posterior.map_or(1., |p| {
                            if local_parent.is_some() {
                                p.rows[slot].unwrap().mass
                            } else {
                                p.explicit_unknown + p.pruned_mass
                            }
                        });
                        for child in 0..3 {
                            let resolved = child < 2 && (observed || local_parent.is_some());
                            if child < 2 && !resolved {
                                continue;
                            }
                            let probability = if local_parent.is_none() && !observed {
                                1.
                            } else {
                                [0.45, 0.35, 0.2][child]
                            };
                            let score = ((slot + child + group_index) % 5) as f64 / 4.;
                            let components = if resolved {
                                [Some(child as u64), None, None, None, None]
                            } else {
                                [None; 5]
                            };
                            rows.push(LocalExtension {
                                parent: local_parent,
                                extension: child as u8,
                                components,
                                articulation: components[0].map(articulation),
                                phrase: None,
                                correspondence: None,
                                grouping: None,
                                section: None,
                                log_transition: f64::ln(probability),
                                log_potential: score,
                            });
                            reference.push((
                                slot as u8,
                                child as u8,
                                resolved,
                                local_prior * probability * if observed { score.exp() } else { 1. },
                            ));
                        }
                    }
                    let total: f64 = reference.iter().map(|r| r.3).sum();
                    joint *= total;
                    for r in &mut reference {
                        r.3 /= total;
                    }
                    input_groups.push(rows);
                    reference_groups.push(reference);
                }
                inputs.push(input_groups);
                expected_local.push(reference_groups);
                joint_weights.push(joint);
            }
        }
        let references: Vec<Vec<&[LocalExtension]>> = inputs
            .iter()
            .map(|g| g.iter().map(Vec::as_slice).collect())
            .collect();
        let extensions: Vec<_> = metadata
            .iter()
            .enumerate()
            .map(
                |(i, &(parent, _, extension, known, transition, potential))| ContextExtension {
                    parent,
                    extension,
                    key: known.then_some(42),
                    log_transition: f64::ln(transition),
                    log_potential: potential,
                    groups: &references[i],
                },
            )
            .collect();
        let total: f64 = joint_weights.iter().sum();
        for w in &mut joint_weights {
            *w /= total;
        }
        let output = state
            .advance(
                [step * 64, (step + 1) * 64],
                observed,
                &groups,
                &extensions,
                Sources::default(),
            )
            .unwrap();
        let mut retained = 0.;
        for (i, context) in output
            .posterior
            .contexts
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.map(|c| (i, c)))
        {
            let index = metadata
                .iter()
                .position(|m| m.1 as u8 == context.weight.parent && m.2 == context.weight.extension)
                .unwrap();
            maximum_error = maximum_error.max((context.weight.mass - joint_weights[index]).abs());
            retained += context.weight.mass;
            let path = output.paths[i].unwrap();
            assert_eq!(path.parent, metadata[index].0);
            assert_eq!(path.state.key, 42);
            assert!(!previous.paths.iter().flatten().any(|p| p.id == path.id));
            for (g, local) in context.groups.iter().enumerate().take(groups.len()) {
                let mut kept = 0.;
                for (j, row) in local
                    .rows
                    .iter()
                    .enumerate()
                    .filter_map(|(j, r)| r.map(|r| (j, r)))
                {
                    let k = expected_local[index][g]
                        .iter()
                        .position(|r| r.0 == row.parent && r.1 == row.extension)
                        .unwrap();
                    maximum_error =
                        maximum_error.max((row.mass - expected_local[index][g][k].3).abs());
                    let owned = path.groups[g][j].unwrap();
                    assert_eq!(owned.parent, inputs[index][g][k].parent);
                    assert_eq!(owned.components, inputs[index][g][k].components);
                    assert_eq!((owned.id.bus, owned.id.epoch), (0, 3));
                    kept += row.mass;
                }
                assert!((kept + local.explicit_unknown + local.pruned_mass - 1.).abs() < 1e-12);
            }
        }
        assert!(
            (retained + output.posterior.explicit_unknown + output.posterior.pruned_mass - 1.)
                .abs()
                < 1e-12
        );
        assert!(maximum_error < 1e-12);
        assert_eq!(state.pairs.as_ptr(), allocation);
        assert!(allocations.contains(&(state.current.as_ref() as *const Snapshot)));
        if step == 8 {
            assert!(state.snapshot().posterior.pruned_count > 0);
        }
    }
    eprintln!(
        "JOINT_OWNER steps=10 maximum_weight_error={maximum_error} pair_capacity={} snapshot_bytes={}",
        state.pairs.capacity(),
        std::mem::size_of::<Snapshot>()
    );
}

#[test]
fn failed_observation_is_atomic_and_cannot_reuse_a_retired_group_or_foreign_parent() {
    let mut state = State::new(0, 3, 0, 64).unwrap();
    let local = [
        LocalExtension {
            components: [Some(1), None, None, None, None],
            articulation: Some(articulation(1)),
            log_transition: 0.6_f64.ln(),
            ..unknown()
        },
        LocalExtension {
            extension: 1,
            log_transition: 0.4_f64.ln(),
            ..unknown()
        },
    ];
    let groups: &[&[LocalExtension]] = &[&local];
    let extensions = [
        ContextExtension {
            parent: None,
            extension: 0,
            key: Some(42),
            log_transition: 0.8_f64.ln(),
            log_potential: 0.,
            groups,
        },
        ContextExtension {
            parent: None,
            extension: 1,
            key: None,
            log_transition: 0.2_f64.ln(),
            log_potential: 0.,
            groups,
        },
    ];
    let first = state
        .advance([0, 64], true, &[handle(1)], &extensions, Sources::default())
        .unwrap()
        .clone();
    let next_id = state.next_id;
    for interval in [
        [0, 64],
        [32, 96],
        [128, 192],
        [64, 64],
        [64, 129],
        [64, 192],
    ] {
        assert!(
            state
                .advance(
                    interval,
                    true,
                    &[handle(1)],
                    &extensions,
                    Sources::default()
                )
                .is_err()
        );
        assert_eq!(state.snapshot(), &first);
        assert_eq!(state.next_id, next_id);
    }
    let parent = first.paths[0].unwrap();
    let mut keep = [
        ContextExtension {
            parent: Some(parent.id),
            extension: 0,
            key: None,
            log_transition: 0.,
            log_potential: 0.,
            groups: &[],
        },
        ContextExtension {
            parent: None,
            extension: 0,
            key: None,
            log_transition: 0.,
            log_potential: 0.,
            groups: &[],
        },
    ];
    for foreign in [
        Handle {
            bus: 1,
            ..parent.id
        },
        Handle {
            epoch: 4,
            ..parent.id
        },
        handle(u64::MAX),
    ] {
        keep[0].parent = Some(foreign);
        assert!(
            state
                .advance([64, 128], true, &[], &keep, Sources::default())
                .is_err()
        );
        assert_eq!(state.snapshot(), &first);
    }
    keep[0].parent = Some(parent.id);
    assert!(
        state
            .advance([64, 128], true, &[], &keep[..1], Sources::default())
            .is_err()
    );
    keep[0].log_transition = 0.5_f64.ln();
    assert!(
        state
            .advance([64, 128], true, &[], &keep, Sources::default())
            .is_err()
    );
    assert_eq!(state.snapshot(), &first);
    keep[0].log_transition = 0.;
    state
        .advance([64, 128], true, &[], &keep, Sources::default())
        .unwrap();
    assert!(state.snapshot().paths.iter().all(Option::is_none));
    assert!(
        state
            .advance(
                [128, 192],
                true,
                &[handle(1)],
                &extensions,
                Sources::default()
            )
            .is_err()
    );
    for group in [
        Handle {
            bus: 1,
            ..handle(2)
        },
        Handle {
            epoch: 4,
            ..handle(2)
        },
        handle(0),
    ] {
        assert!(
            state
                .advance([128, 192], true, &[group], &extensions, Sources::default())
                .is_err()
        );
    }
    state
        .advance(
            [128, 192],
            true,
            &[handle(2)],
            &extensions,
            Sources::default(),
        )
        .unwrap();
    assert_ne!(state.snapshot().paths[0].unwrap().id, parent.id);
}

#[test]
fn local_parent_cannot_cross_group_context_or_unknown_and_identity_overflow_is_atomic() {
    let mut state = State::new(0, 3, 0, 64).unwrap();
    let one = [
        LocalExtension {
            components: [Some(1), None, None, None, None],
            articulation: Some(articulation(1)),
            log_transition: 0.5_f64.ln(),
            ..unknown()
        },
        LocalExtension {
            extension: 1,
            log_transition: 0.5_f64.ln(),
            ..unknown()
        },
    ];
    let rows: &[&[LocalExtension]] = &[&one, &one];
    let initial = [
        ContextExtension {
            parent: None,
            extension: 0,
            key: Some(9),
            log_transition: 0.5_f64.ln(),
            log_potential: 0.,
            groups: rows,
        },
        ContextExtension {
            parent: None,
            extension: 1,
            key: None,
            log_transition: 0.5_f64.ln(),
            log_potential: 0.,
            groups: rows,
        },
    ];
    let first = state
        .advance(
            [0, 64],
            true,
            &[handle(1), handle(2)],
            &initial,
            Sources::default(),
        )
        .unwrap()
        .clone();
    let context = first.paths[0].unwrap();
    let wrong = [
        LocalExtension {
            parent: Some(context.groups[0][0].unwrap().id),
            ..unknown()
        },
        unknown(),
    ];
    let wrong_rows: &[&[LocalExtension]] = &[&wrong, &wrong];
    let invalid = [
        ContextExtension {
            parent: Some(context.id),
            extension: 0,
            key: None,
            log_transition: 0.,
            log_potential: 0.,
            groups: wrong_rows,
        },
        ContextExtension {
            parent: None,
            extension: 0,
            key: None,
            log_transition: 0.,
            log_potential: 0.,
            groups: wrong_rows,
        },
    ];
    assert!(
        state
            .advance(
                [64, 128],
                true,
                &[handle(1), handle(2)],
                &invalid,
                Sources::default()
            )
            .is_err()
    );
    assert_eq!(state.snapshot(), &first);
    let mut fresh = State::new(0, 3, 0, 64).unwrap();
    fresh.next_id = u64::MAX;
    let old = fresh.snapshot().clone();
    assert!(
        fresh
            .advance(
                [0, 64],
                true,
                &[handle(1), handle(2)],
                &initial,
                Sources::default()
            )
            .is_err()
    );
    assert_eq!(fresh.snapshot(), &old);
    assert_eq!(fresh.next_id, u64::MAX);
    fresh.next_id = 1;
    assert_eq!(
        fresh
            .advance(
                [0, 64],
                true,
                &[handle(1), handle(2)],
                &initial,
                Sources::default()
            )
            .unwrap(),
        &first
    );
    let mut missing = State::new(0, 3, 0, 64).unwrap();
    assert!(
        missing
            .advance(
                [0, 128],
                false,
                &[handle(1), handle(2)],
                &initial,
                Sources::default()
            )
            .is_err()
    );
    assert_eq!(missing.snapshot().end, 0);
}

#[test]
fn full_owned_inventory_keeps_unknown_and_pruned_mass_without_growing_scratch() {
    use crate::temporal_cognition::joint::{Context, Local, Weight};
    let mut state = State::new(0, 3, 0, 64).unwrap();
    let groups: Vec<_> = (1..=8).map(handle).collect();
    state.current.groups = std::array::from_fn(|g| Some(groups[g]));
    state.current.posterior.group_count = 8;
    state.current.posterior.explicit_unknown = 1. / 8.;
    for c in 0..7 {
        state.current.paths[c] = Some(ContextPath {
            id: handle(10 + c as u64),
            parent: None,
            state: shared::State {
                key: 42,
                admission: None,
                end: 0,
                groups: [None; 8],
            },
            groups: std::array::from_fn(|g| {
                std::array::from_fn(|l| {
                    Some(LocalPath {
                        id: handle(100 + (c * 120 + g * 15 + l) as u64),
                        parent: None,
                        components: [Some(l as u64 % 4), None, None, None, None],
                        articulation: Some(articulation(l as u64 % 4)),
                        phrase_slot: None,
                        section_slot: None,
                        correspondence_slot: None,
                        grouping_slot: None,
                        grouping_supported: false,
                        correspondence_supported: false,
                    })
                })
            }),
        });
        state.current.posterior.contexts[c] = Some(Context {
            weight: Weight {
                parent: 7,
                extension: c as u8,
                mass: 1. / 8.,
            },
            groups: [Local {
                rows: std::array::from_fn(|l| {
                    Some(Weight {
                        parent: 15,
                        extension: l as u8,
                        mass: 1. / 16.,
                    })
                }),
                explicit_unknown: 1. / 16.,
                ..Local::default()
            }; 8],
        });
    }
    state.maximum_group_generation = 8;
    state.next_id = 2000;
    let pointer = state.pairs.as_ptr();
    let mut inputs = Vec::new();
    for c in 0..8 {
        for _ in 0..4 {
            let mut local_groups = Vec::new();
            for g in 0..8 {
                let mut rows = Vec::new();
                for p in 0..if c == 7 { 1 } else { 16 } {
                    let parent = if c == 7 || p == 15 {
                        None
                    } else {
                        Some(state.current.paths[c].unwrap().groups[g][p].unwrap().id)
                    };
                    for e in 0..16 {
                        rows.push(LocalExtension {
                            parent,
                            extension: e,
                            components: if e == 15 {
                                [None; 5]
                            } else {
                                [Some(u64::from(e) % 4), None, None, None, None]
                            },
                            articulation: (e != 15).then(|| articulation(u64::from(e) % 4)),
                            phrase: None,
                            correspondence: None,
                            grouping: None,
                            section: None,
                            log_transition: (1_f64 / 16.).ln(),
                            log_potential: 0.,
                        });
                    }
                }
                local_groups.push(rows);
            }
            inputs.push(local_groups);
        }
    }
    let references: Vec<Vec<&[LocalExtension]>> = inputs
        .iter()
        .map(|g| g.iter().map(Vec::as_slice).collect())
        .collect();
    let extensions: Vec<_> = (0..32)
        .map(|i| ContextExtension {
            parent: if i / 4 == 7 {
                None
            } else {
                Some(state.current.paths[i / 4].unwrap().id)
            },
            extension: (i % 4) as u8,
            key: (i % 4 < 3).then_some(42),
            log_transition: (1_f64 / 4.).ln(),
            log_potential: 0.,
            groups: &references[i],
        })
        .collect();
    let output = state
        .advance([0, 64], true, &groups, &extensions, Sources::default())
        .unwrap();
    assert_eq!(output.posterior.shared_enumerated, 32);
    // The shared unknown has one unresolved local parent, not fifteen invented identities.
    assert_eq!(output.posterior.local_enumerated, 28 * 8 * 256 + 4 * 8 * 16);
    assert_eq!(output.posterior.pruned_count, 17);
    assert!((output.posterior.explicit_unknown - 0.25).abs() < 1e-12);
    assert!((output.posterior.pruned_mass - 17. / 32.).abs() < 1e-12);
    for context in output.posterior.contexts.iter().flatten() {
        for local in &context.groups {
            assert_eq!(local.pruned_count, 225);
            assert!((local.explicit_unknown - 1. / 16.).abs() < 1e-12);
            assert!((local.pruned_mass - 225. / 256.).abs() < 1e-12);
        }
    }
    assert_eq!(state.pairs.as_ptr(), pointer);
    assert_eq!(state.pairs.capacity(), 65536);
}

#[test]
fn phrase_proposals_keep_owned_payloads_and_feed_the_next_conditional_transition() {
    use crate::temporal_cognition::{gesture, joint::proposals::List, phrase};
    let mut cfg = phrase::tests::config();
    cfg.hazard = [0.; 26];
    cfg.hazard[0] = 8.;
    cfg.exits = [[0.; 26]; 4];
    let mut state = State::new(0, 3, 0, 64).unwrap();
    let arenas = [state.current.phrases.as_ptr(), state.next.phrases.as_ptr()];
    let mut next_phrase = 0;
    let mut checks = 0;
    let mut seen = [false; 4];
    for step in 0..6 {
        let interval = [step * 64, (step + 1) * 64];
        let observed = step != 3;
        let previous = state.snapshot().clone();
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
                    let payload = parent
                        .and_then(|p| p.phrase_slot)
                        .and_then(|i| previous.phrases[i].as_ref());
                    if let Some(payload) = payload.filter(|p| p.foreground.is_some()) {
                        let ids = std::array::from_fn(|_| {
                            next_phrase += 1;
                            next_phrase
                        });
                        let (list, choices) = List::owned_active_phrase(
                            (parent.unwrap().phrase_slot.unwrap(), payload),
                            ids,
                            [None; 12],
                            Some(gesture::State::Release),
                            interval,
                            1000,
                            observed,
                            cfg,
                        )
                        .unwrap();
                        for (extension, choice) in
                            list.entries[..list.len].iter().flatten().enumerate()
                        {
                            local.push(LocalExtension {
                                parent: parent.map(|p| p.id),
                                extension: extension as u8,
                                components: [None, None, choice.id, None, None],
                                articulation: None,
                                correspondence: None,
                                grouping: None,
                                section: None,
                                phrase: choice
                                    .id
                                    .map(|id| *choices.iter().find(|c| c.id == id).unwrap()),
                                log_transition: choice.log_weight,
                                log_potential: 0.,
                            });
                        }
                    } else if payload.is_some() || observed {
                        let id = payload.map_or_else(
                            || {
                                next_phrase += 1;
                                next_phrase
                            },
                            |p| p.id,
                        );
                        local.push(LocalExtension {
                            parent: parent.map(|p| p.id),
                            extension: 0,
                            components: [None, None, Some(id), None, None],
                            articulation: None,
                            correspondence: None,
                            grouping: None,
                            section: None,
                            phrase: Some(PhraseExtension {
                                parent_slot: parent.and_then(|p| p.phrase_slot),
                                kind: None,
                                id,
                            }),
                            log_transition: 0.9_f64.ln(),
                            log_potential: 0.,
                        });
                        local.push(LocalExtension {
                            parent: parent.map(|p| p.id),
                            extension: 1,
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
        let output = state
            .advance(
                interval,
                observed,
                &[handle(1)],
                &inputs,
                Sources::default(),
            )
            .unwrap();
        assert!(arenas.contains(&output.phrases.as_ptr()));
        assert_eq!(
            output.phrases.iter().flatten().count(),
            output
                .paths
                .iter()
                .flatten()
                .flat_map(|c| c.groups[0].iter().flatten())
                .count()
        );
        for context in output.paths.iter().flatten() {
            for path in context.groups[0].iter().flatten() {
                let child = output.phrases[path.phrase_slot.unwrap()].as_ref().unwrap();
                assert_eq!(path.components[2], Some(child.id));
                let old = path
                    .parent
                    .and_then(|id| {
                        previous
                            .paths
                            .iter()
                            .flatten()
                            .flat_map(|c| c.groups[0].iter().flatten())
                            .find(|p| p.id == id)
                    })
                    .and_then(|p| p.phrase_slot)
                    .and_then(|i| previous.phrases[i].as_ref());
                if let Some(old) = old {
                    if child.id == old.id {
                        assert_eq!(
                            child.foreground.map(|f| (f.start, f.credit)),
                            old.foreground.map(|f| (f.start, f.credit))
                        );
                        if !observed {
                            assert_eq!(child.foreground, old.foreground);
                        }
                    } else if let Some((kind, at)) = child.event {
                        seen[kind as usize] = true;
                        assert_eq!(at, interval[1]);
                        if kind == phrase::Exit::Reinterpret {
                            assert_eq!(
                                child.foreground.unwrap().credit,
                                old.foreground.unwrap().credit
                            );
                            assert_eq!(
                                child.foreground.unwrap().start,
                                old.foreground.unwrap().start
                            );
                            assert!(child.completed_foreground.is_none());
                        } else {
                            assert_eq!(child.completed_foreground, old.foreground);
                        }
                    }
                }
                if !observed {
                    assert!(
                        child
                            .foreground
                            .is_none_or(|f| f.heard_end <= interval[0] || f.heard_end == f.start)
                    );
                }
                checks += 1;
            }
        }
    }
    assert!(
        checks > 50 && seen.into_iter().all(|v| v),
        "checks={checks} exits={seen:?}"
    );
    eprintln!(
        "JOINT_PHRASE_OWNERSHIP steps=6 retained_checks={checks} payload_bytes={} arena_slots={} arena_bytes={} extension_bytes={}",
        std::mem::size_of::<Interpretation>(),
        state.current.phrases.len(),
        std::mem::size_of_val(state.current.phrases.as_ref()),
        std::mem::size_of::<PhraseExtension>()
    );
}

#[test]
fn phrase_parent_slot_cannot_cross_paths_and_rejection_is_atomic() {
    let mut state = State::new(0, 3, 0, 64).unwrap();
    let seed = [
        LocalExtension {
            components: [None, None, Some(1), None, None],
            phrase: Some(PhraseExtension {
                parent_slot: None,
                kind: None,
                id: 1,
            }),
            log_transition: 0.45_f64.ln(),
            ..unknown()
        },
        LocalExtension {
            extension: 1,
            components: [None, None, Some(2), None, None],
            phrase: Some(PhraseExtension {
                parent_slot: None,
                kind: None,
                id: 2,
            }),
            log_transition: 0.45_f64.ln(),
            ..unknown()
        },
        LocalExtension {
            extension: 2,
            log_transition: 0.1_f64.ln(),
            ..unknown()
        },
    ];
    let groups: &[&[LocalExtension]] = &[&seed];
    let initial = [
        ContextExtension {
            parent: None,
            extension: 0,
            key: Some(42),
            log_transition: 0.9_f64.ln(),
            log_potential: 0.,
            groups,
        },
        ContextExtension {
            parent: None,
            extension: 1,
            key: None,
            log_transition: 0.1_f64.ln(),
            log_potential: 0.,
            groups,
        },
    ];
    state
        .advance([0, 64], true, &[handle(1)], &initial, Sources::default())
        .unwrap();
    let before = state.snapshot().clone();
    let context = before.paths[0].unwrap();
    let path = context.groups[0][0].unwrap();
    let other = context.groups[0][1].unwrap();
    let parent = before.phrases[path.phrase_slot.unwrap()].unwrap();
    let next_id = state.next_id;
    for case in 0..6 {
        let slot = match case {
            0 => other.phrase_slot,
            1 => Some(usize::MAX),
            3 => None,
            _ => path.phrase_slot,
        };
        let id = if case == 2 { parent.id + 1 } else { parent.id };
        let proposal = PhraseExtension {
            parent_slot: slot,
            kind: None,
            id,
        };
        let rows = [
            LocalExtension {
                parent: Some(path.id),
                components: [
                    None,
                    None,
                    Some(if case == 4 { 99 } else { id }),
                    None,
                    None,
                ],
                phrase: Some(proposal),
                log_transition: 0.9_f64.ln(),
                ..unknown()
            },
            LocalExtension {
                parent: Some(path.id),
                extension: 1,
                log_transition: 0.1_f64.ln(),
                ..unknown()
            },
            LocalExtension {
                parent: Some(other.id),
                ..unknown()
            },
            unknown(),
        ];
        let groups: &[&[LocalExtension]] = &[&rows];
        let unknown_rows = [unknown()];
        let unresolved: &[&[LocalExtension]] = &[&unknown_rows];
        let input = [
            ContextExtension {
                parent: Some(context.id),
                extension: 0,
                key: Some(42),
                log_transition: 0.9_f64.ln(),
                log_potential: 0.,
                groups,
            },
            ContextExtension {
                parent: Some(context.id),
                extension: 1,
                key: None,
                log_transition: 0.1_f64.ln(),
                log_potential: 0.,
                groups,
            },
            ContextExtension {
                parent: None,
                extension: 0,
                key: None,
                log_transition: 0.,
                log_potential: 0.,
                groups: unresolved,
            },
        ];
        let error = state
            .advance([64, 128], true, &[handle(1)], &input, Sources::default())
            .err();
        if case == 5 {
            assert_eq!(error, None, "valid retry after rejected slots");
        } else {
            assert_eq!(
                error,
                Some(match case {
                    2 => "invalid joint phrase parent identity or evidence",
                    4 => "joint phrase differs from component identity",
                    _ => "joint phrase slot differs from selected local parent",
                }),
                "case={case}"
            );
            assert_eq!(state.snapshot(), &before);
            assert_eq!(state.next_id, next_id);
        }
    }
    let admission = PhraseExtension {
        parent_slot: None,
        kind: None,
        id: 100,
    };
    assert!(admission.validate(None, [128, 192], false).is_err());
    let stay = PhraseExtension {
        parent_slot: path.phrase_slot,
        kind: None,
        id: parent.id,
    };
    for kind in 0..3 {
        let mut future = parent;
        match kind {
            0 => future.foreground.as_mut().unwrap().heard_end = 129,
            1 => future.event = Some((crate::temporal_cognition::phrase::Exit::New, 129)),
            _ => future.foreground.as_mut().unwrap().start = 129,
        }
        assert!(stay.validate(Some(&future), [128, 192], true).is_err());
    }
}
