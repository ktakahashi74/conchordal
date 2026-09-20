use super::tests::unknown;
use super::*;
use crate::temporal_cognition::{phrase, section as marginal};
use shared::producer::{Inputs, Kind, Producer};

#[test]
fn shared_producer_uses_section_laws_actual_support_and_atomic_bound_sources() {
    let group = |generation| Handle {
        bus: 1,
        epoch: 0,
        generation,
    };
    let mut owner = State::new(1, 0, 0, 100).unwrap();
    let mut producer = Producer::new(0);
    let mut next_section = 1;
    let config = marginal::tests::runtime_config();
    let mut head = marginal::Head {
        means: [0.; 82],
        deviations: [1.; 82],
        hazard: [0.; 83],
        exits: [[0.; 83]; 3],
    };
    head.hazard[0] = 4_f64.ln();
    for (row, value) in head.exits.iter_mut().zip([-1., -1., 2.]) {
        row[0] = value;
    }
    let (mut generated, mut contrasts, mut rejected) = (0, 0, 0);
    for step in 1..=5 {
        let interval = [(step - 1) * 100, step * 100];
        let observed = step != 4;
        let groups = if step == 5 {
            [group(3), group(2)]
        } else {
            [group(2), group(3)]
        };
        let observations: [Observation; 2] = std::array::from_fn(|i| {
            let mut raw = phrase::tests::input(step, 0.);
            raw.group_handles[0] = Some(groups[i]);
            raw.features[0].as_mut().unwrap().raw.group = groups[i];
            if [1, 5].contains(&step) {
                raw.features[0].as_mut().unwrap().raw.values[2] = Some(-30.);
            }
            raw.assignment.rows[0].as_mut().unwrap().weights[0] = 0.4;
            raw.eligible[0] = observed;
            Observation::new(&raw, groups[i], interval, 1000, None).unwrap()
        });
        let previous = owner.snapshot().clone();
        let batch = producer
            .produce(
                &previous,
                Inputs {
                    interval,
                    rate: 1000,
                    observed,
                    groups: &groups,
                    observations: &[Some(&observations[0]), Some(&observations[1])],
                    returns: &[None, None],
                    retained_episodes: &[],
                    head: &head,
                    rms_reference: 0.1,
                },
            )
            .unwrap();
        assert_eq!(batch.examined_returns, 0);
        assert!(batch.entries().count() <= 31);
        for p in batch.entries() {
            generated += 1;
            if [1, 4, 5].contains(&step) {
                assert!(p.origin.is_none());
            }
            if let Some(origin) = p.origin {
                assert_eq!(
                    origin.group,
                    group(2),
                    "equal support uses stable group identity"
                );
                assert_eq!(origin.cut, interval[1]);
                assert!((origin.support.assignment_seconds - 0.04).abs() < 1e-12);
                if origin.kind == Kind::Contrast {
                    assert_eq!(step, 3);
                    let expected = 0.4 * 0.99 * (1. - 5_f64.powf(-0.1)) * 2_f64.exp()
                        / (2. * (-1_f64).exp() + 2_f64.exp());
                    assert!(
                        (p.raw_score - expected).abs() < 1e-12,
                        "actual={}, expected={expected}",
                        p.raw_score
                    );
                    assert!(origin.local_parent.is_some() && origin.section.is_some());
                    contrasts += 1;
                } else {
                    assert!((p.raw_score - 0.4).abs() < 1e-12);
                }
            }
        }
        if step == 1 {
            assert_eq!(batch.entries().count(), 1);
            assert_eq!(batch.entries().next().unwrap().log_transition, 0.);
        }
        let (mut rows, mut sources) = (Vec::new(), Vec::new());
        for p in batch.entries() {
            let parent = p
                .parent
                .and_then(|id| previous.paths.iter().flatten().find(|p| p.id == id));
            let pair: [Vec<LocalExtension>; 2] = std::array::from_fn(|g| {
                let previous_group = previous.groups.iter().position(|h| *h == Some(groups[g]));
                let mut locals: Vec<_> =
                    parent.zip(previous_group).map_or_else(Vec::new, |(p, g)| {
                        p.groups[g].iter().flatten().copied().map(Some).collect()
                    });
                locals.push(None);
                let mut rows = Vec::new();
                for local in locals {
                    let old = local.and_then(|p| p.section_slot);
                    if old.is_some() || observed {
                        let id = if let Some(p) = local {
                            p.components[3].unwrap()
                        } else {
                            let id = next_section;
                            next_section += 1;
                            id
                        };
                        let source_slot = sources.len();
                        sources.push(section::Source {
                            parent: local.map(|p| p.id),
                            returned: None,
                            input: marginal::interpretation::Input {
                                group: groups[g],
                                interval,
                                rate: 1000,
                                observed,
                                delta: observations[g].delta,
                                deliveries: &[],
                                completed: None,
                                retrieval: [None; 2],
                                query: None,
                            },
                            change: marginal::interpretation::Change {
                                transition: if old.is_some() {
                                    marginal::form::Transition::Stay
                                } else {
                                    marginal::form::Transition::NewContext
                                },
                                context: 1,
                                focus: None,
                            },
                        });
                        rows.push(LocalExtension {
                            parent: local.map(|p| p.id),
                            components: [None, None, None, Some(id), None],
                            section: Some(section::Extension {
                                id,
                                parent_slot: old,
                                source_slot,
                            }),
                            log_transition: 0.99_f64.ln(),
                            ..unknown()
                        });
                        rows.push(LocalExtension {
                            parent: local.map(|p| p.id),
                            extension: 1,
                            log_transition: 0.01_f64.ln(),
                            ..unknown()
                        });
                    } else {
                        rows.push(unknown());
                    }
                }
                rows
            });
            rows.push(pair);
        }
        let refs: Vec<[&[LocalExtension]; 2]> = rows
            .iter()
            .map(|r| [r[0].as_slice(), r[1].as_slice()])
            .collect();
        let mut inputs: Vec<_> = batch
            .entries()
            .enumerate()
            .map(|(i, p)| ContextExtension {
                parent: p.parent,
                extension: p.slot,
                key: p.key,
                log_transition: p.log_transition,
                log_potential: 0.,
                groups: &refs[i],
            })
            .collect();
        let old_key = inputs[0].key;
        inputs[0].key = Some(u64::MAX);
        assert!(
            owner
                .advance(
                    interval,
                    observed,
                    &groups,
                    &inputs,
                    Sources {
                        shared: Some(&batch),
                        observations: &[Some(&observations[0]), Some(&observations[1])],
                        section_config: Some(&config),
                        sections: &sources,
                        ..Sources::default()
                    }
                )
                .is_err()
        );
        assert_eq!(owner.snapshot(), &previous);
        inputs[0].key = old_key;
        rejected += 1;
        let next_path = owner.next_id;
        let mut changed = observations;
        changed[0].delta.assignment_seconds *= 0.5;
        if batch.entries().any(|p| p.origin.is_some()) {
            assert!(
                owner
                    .advance(
                        interval,
                        observed,
                        &groups,
                        &inputs,
                        Sources {
                            shared: Some(&batch),
                            observations: &[Some(&changed[0]), Some(&changed[1])],
                            section_config: Some(&config),
                            sections: &sources,
                            ..Sources::default()
                        }
                    )
                    .is_err()
            );
            assert_eq!(owner.snapshot(), &previous);
            assert_eq!(owner.next_id, next_path);
            rejected += 1;
        }
        let output = owner
            .advance(
                interval,
                observed,
                &groups,
                &inputs,
                Sources {
                    shared: Some(&batch),
                    observations: &[Some(&observations[0]), Some(&observations[1])],
                    section_config: Some(&config),
                    sections: &sources,
                    ..Sources::default()
                },
            )
            .unwrap();
        for p in output.paths.iter().flatten() {
            let origin = p.state.admission.unwrap();
            assert!(origin.cut <= interval[1]);
            assert_eq!(origin.group, group(2));
            if step >= 4 {
                assert!(
                    origin.cut < interval[1],
                    "missing/quiet cannot refresh the admission witness"
                );
            }
        }
    }
    assert!(contrasts > 0);
    println!(
        "JOINT_SHARED_PRODUCER updates=5 generated={generated} contrasts={contrasts} rejected={rejected}"
    );
}
