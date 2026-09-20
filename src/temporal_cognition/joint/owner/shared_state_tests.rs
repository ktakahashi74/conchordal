use super::tests::unknown;
use super::*;
use crate::temporal_cognition::{gesture, phrase, section as marginal};

#[test]
fn shared_summaries_preserve_conditional_mass_path_identity_and_original_group_clocks() {
    let group = |generation| Handle {
        bus: 1,
        epoch: 0,
        generation,
    };
    let mut owner = State::new(1, 0, 0, 100).unwrap();
    let config = marginal::tests::runtime_config();
    let mut checked = 0;
    for step in 1..=3 {
        let interval = [(step - 1) * 100, step * 100];
        let groups = match step {
            1 => [group(2), group(3)],
            2 => [group(3), group(2)],
            _ => [group(3), group(4)],
        };
        let observations: [Observation; 2] = std::array::from_fn(|i| {
            let mut raw = phrase::tests::input(step, 0.);
            raw.group_handles[0] = Some(groups[i]);
            raw.features[0].as_mut().unwrap().raw.group = groups[i];
            raw.assignment.rows[0].as_mut().unwrap().weights[0] =
                if groups[i] == group(2) { 0.4 } else { 0.6 };
            raw.eligible[0] =
                !(step == 2 && groups[i] == group(2) || step == 3 && groups[i] == group(4));
            Observation::new(&raw, groups[i], interval, 1000, None).unwrap()
        });
        let previous = owner.snapshot().clone();
        let mut parents: Vec<_> = previous.paths.iter().flatten().copied().map(Some).collect();
        parents.push(None);
        let (mut rows, mut metadata) = (Vec::new(), Vec::new());
        for parent in parents {
            let choices: &[Option<u8>] = if step == 1 {
                &[Some(0), Some(1), None]
            } else if parent.is_some() {
                &[Some(0), None]
            } else {
                &[None]
            };
            for (extension, &known) in choices.iter().enumerate() {
                let pair: [Vec<LocalExtension>; 2] = std::array::from_fn(|g| {
                    if step == 1 {
                        let probability = if groups[g] == group(2) {
                            if known == Some(1) {
                                [0.2_f64, 0.3, 0.5]
                            } else {
                                [0.6, 0.3, 0.1]
                            }
                        } else {
                            [0.7, 0.2, 0.1]
                        };
                        let states = if groups[g] == group(2) {
                            [gesture::State::Attack, gesture::State::Gap]
                        } else {
                            [gesture::State::Continuation, gesture::State::Release]
                        };
                        let mut rows: Vec<_> = states
                            .into_iter()
                            .enumerate()
                            .map(|(i, state)| LocalExtension {
                                extension: i as u8,
                                articulation: Some(Articulation {
                                    state,
                                    entered: 0,
                                    run: gesture::Run::default(),
                                }),
                                components: [Some(state as u64), None, None, None, None],
                                log_transition: probability[i].ln(),
                                ..unknown()
                            })
                            .collect();
                        rows.push(LocalExtension {
                            extension: 2,
                            log_transition: probability[2].ln(),
                            ..unknown()
                        });
                        return rows;
                    }
                    let old_group = previous.groups.iter().position(|h| *h == Some(groups[g]));
                    let local = parent.zip(old_group).map(|(p, g)| p.groups[g]);
                    let mut rows = Vec::new();
                    for p in local.iter().flatten().flatten() {
                        rows.push(LocalExtension {
                            parent: Some(p.id),
                            articulation: p.articulation,
                            components: p.components,
                            log_transition: 0.5_f64.ln(),
                            log_potential: if observations[g].observed { 0. } else { 123. },
                            ..unknown()
                        });
                        rows.push(LocalExtension {
                            parent: Some(p.id),
                            extension: 1,
                            log_transition: 0.5_f64.ln(),
                            ..unknown()
                        });
                    }
                    rows.push(unknown());
                    rows
                });
                rows.push(pair);
                let weight: f64 = if step == 1 {
                    [0.6, 0.3, 0.1][extension]
                } else if parent.is_none() {
                    1.
                } else if known.is_some() {
                    0.9
                } else {
                    0.1
                };
                metadata.push((
                    parent.map(|p| p.id),
                    extension as u8,
                    known.is_some(),
                    weight,
                ));
            }
        }
        let refs: Vec<[&[LocalExtension]; 2]> = rows
            .iter()
            .map(|r| [r[0].as_slice(), r[1].as_slice()])
            .collect();
        let inputs: Vec<_> = metadata
            .iter()
            .enumerate()
            .map(
                |(i, &(parent, extension, known, weight))| ContextExtension {
                    parent,
                    extension,
                    key: known.then_some(42),
                    log_transition: weight.ln(),
                    log_potential: 0.,
                    groups: &refs[i],
                },
            )
            .collect();
        let out = owner
            .advance(
                interval,
                true,
                &groups,
                &inputs,
                Sources {
                    observations: &[Some(&observations[0]), Some(&observations[1])],
                    section_config: Some(&config),
                    ..Sources::default()
                },
            )
            .unwrap();
        assert_eq!(out.paths.iter().flatten().count(), 2);
        assert_ne!(out.paths[0].unwrap().id, out.paths[1].unwrap().id);
        for (c, path) in out.paths.iter().flatten().enumerate() {
            assert_eq!(path.state.key, 42);
            assert_eq!(path.state.end, interval[1]);
            assert!(path.state.groups[2..].iter().all(Option::is_none));
            for (g, summary) in path.state.groups[..2].iter().flatten().enumerate() {
                assert_eq!(summary.group, groups[g]);
                let expected = if summary.group == group(2) {
                    let first = if c == 0 { 0.6 } else { 0.2 };
                    let factor = if step == 1 { 1. } else { 0.5 };
                    [first * factor, 0., 0., 0.3 * factor]
                } else if summary.group == group(3) {
                    let factor = 0.5_f64.powi(step as i32 - 1);
                    [0., 0.7 * factor, 0.2 * factor, 0.]
                } else {
                    [0.; 4]
                };
                for (actual, expected) in summary.articulation.into_iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-12);
                }
                let mass: f64 = expected.iter().sum();
                assert!((summary.represented_mass - mass).abs() < 1e-12);
                assert!((summary.explicit_unknown - (1. - mass)).abs() < 1e-12);
                assert_eq!(summary.pruned_mass, 0.);
                assert_eq!(summary.grouping, [0.; 2]);
                assert_eq!(summary.phrase, [0.; 2]);
                assert_eq!(summary.correspondence, [0.; 2]);
                assert!(summary.sections.iter().all(Option::is_none));
                if observations[g].observed {
                    let support = summary.support.unwrap();
                    assert_eq!(support.interval, interval);
                    assert_eq!(support.available, interval[1]);
                    assert_eq!(summary.last_observed_end, Some(interval[1]));
                    let expected = if summary.group == group(2) {
                        0.04
                    } else {
                        0.06
                    };
                    assert!((support.assignment_seconds - expected).abs() < 1e-12);
                } else {
                    assert!(summary.support.is_none());
                    assert_eq!(
                        summary.last_observed_end,
                        (summary.group == group(2)).then_some(100)
                    );
                }
                checked += 1;
            }
            if let Some(parent) = path.parent {
                let old = previous
                    .paths
                    .iter()
                    .flatten()
                    .find(|p| p.id == parent)
                    .unwrap();
                assert_eq!(
                    old.state.end, interval[0],
                    "the feature source is the previous cut"
                );
            }
        }
    }
    println!(
        "JOINT_SHARED_STATE updates=3 context_checks=6 group_checks={checked} state_bytes={} group_bytes={}",
        std::mem::size_of::<shared::State>(),
        std::mem::size_of::<shared::Group>()
    );
}
