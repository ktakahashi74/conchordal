use super::*;

#[test]
fn generated_local_batch_advances_observed_and_missing_physical_intervals() {
    let group = phrase::tests::input(1, 0.).group_handles[0].unwrap();
    let gesture = TemporalGestureConfig {
        rms_reference: 0.1,
        means: [0.; 5],
        deviations: [1.; 5],
        coefficients: [[[0.; 11]; 4]; 4],
    };
    let phrase = phrase::tests::config();
    let section_config = crate::temporal_cognition::section::tests::runtime_config();
    let head = Head {
        means: [0.; 82],
        deviations: [1.; 82],
        hazard: [0.; 83],
        exits: [[0.; 83]; 3],
    };
    let mut acoustic = acoustics::Cache::new(group, 0, 1000, 100).unwrap();
    let mut grouping = grouping::Cache::new(group, Some(1.)).unwrap();
    let mut correspondence =
        correspondence::Cache::new(group, 1000, section_config, Some(0.)).unwrap();
    let returns = section::Returns::new(group);
    let mut owner = owner::State::new(group.bus, group.epoch, 0, 100).unwrap();
    let mut shared = shared::producer::Producer::new(0);
    let mut producer = Producer::new(0);
    let capacities = (producer.rows.capacity(), producer.sources.capacity());
    let mut total = 0;
    for step in 1..=8 {
        let interval = [(step - 1) * 100, step * 100];
        let observed = step != 5;
        let mut frontend = phrase::tests::input(step, 0.2);
        frontend.eligible[0] = observed;
        let observation = acoustic.advance(&frontend, interval).unwrap();
        grouping.refresh(interval[1], None).unwrap();
        correspondence.refresh(interval[1], None, &[]).unwrap();
        let batch = shared
            .produce(
                owner.snapshot(),
                shared::producer::Inputs {
                    interval,
                    rate: 1000,
                    observed,
                    groups: &[group],
                    observations: &[Some(&observation)],
                    returns: &[Some(&returns)],
                    retained_episodes: &[],
                    head: &head,
                    rms_reference: gesture.rms_reference,
                },
            )
            .unwrap();
        let groups = [Group {
            observation: &observation,
            acoustics: &acoustic,
            grouping: &grouping,
            correspondence: &correspondence,
            returns: &returns,
            motion: None,
            phrase_context: [None; 12],
        }];
        let input = Input {
            interval,
            rate: 1000,
            observed,
            groups: &groups,
            retained: &[],
            gesture: &gesture,
            phrase,
            section: &head,
            section_config: &section_config,
        };
        if step == 1 {
            let ids = (producer.next_id, producer.next_context, producer.cut);
            assert!(
                producer
                    .produce(
                        owner.snapshot(),
                        &batch,
                        &input,
                        |_| Ok(0.),
                        |_, _, _| Err("injected score failure")
                    )
                    .is_err()
            );
            assert_eq!((producer.next_id, producer.next_context, producer.cut), ids);
            producer.next_id = u64::MAX - 7;
            assert!(
                producer
                    .produce(
                        owner.snapshot(),
                        &batch,
                        &input,
                        |_| Ok(0.),
                        |_, _, _| Ok(0.)
                    )
                    .is_err()
            );
            assert_eq!(producer.next_id, u64::MAX - 7);
            producer.next_id = ids.0;
            let original = producer
                .produce(
                    owner.snapshot(),
                    &batch,
                    &input,
                    |_| Ok(0.),
                    |_, _, _| Ok(0.),
                )
                .unwrap();
            let largest = original
                .producer
                .rows
                .iter()
                .flat_map(|r| r.components[2..4].iter().flatten())
                .copied()
                .max()
                .unwrap();
            assert!(original.producer.next_id > largest);
        }
        let prepared = producer
            .produce(
                owner.snapshot(),
                &batch,
                &input,
                |_| Ok(0.2),
                |_, _, row| Ok(row.components.iter().flatten().count() as f64 * 0.1),
            )
            .unwrap();
        assert!(
            prepared.producer.work.tuples > 0
                && prepared.producer.work.tuples <= prepared.producer.work.parents * 16
        );
        assert!(prepared.producer.work.priorities <= prepared.producer.work.parents * 86);
        assert!(prepared.producer.work.heap_pops <= prepared.producer.work.parents * 16);
        if !observed {
            assert!(prepared.producer.rows.iter().all(|r| r.log_potential == 0.));
            assert!(
                prepared.producer.shared_scores[..batch.entries().count()]
                    .iter()
                    .all(|v| *v == 0.)
            );
        }
        total += prepared.producer.work.tuples;
        // Direct ordinary-probability enumeration, including local partitions in shared evidence.
        let mut enumerated = Vec::new();
        for (c, proposal) in batch.entries().enumerate() {
            let parent = owner
                .snapshot()
                .paths
                .iter()
                .position(|p| p.is_some_and(|p| Some(p.id) == proposal.parent));
            let context = parent.and_then(|i| owner.snapshot().posterior.contexts[i]);
            let prior = context.map_or(
                owner.snapshot().posterior.explicit_unknown
                    + owner.snapshot().posterior.pruned_mass,
                |c| c.weight.mass,
            );
            let [a, b] = prepared.producer.ranges[c][0];
            let mut local = Vec::new();
            for row in &prepared.producer.rows[a..b] {
                let local_parent = parent
                    .and_then(|i| owner.snapshot().paths[i])
                    .and_then(|c| {
                        c.groups[0]
                            .iter()
                            .position(|p| p.is_some_and(|p| Some(p.id) == row.parent))
                    });
                let prior = context.map_or(1., |c| {
                    local_parent.and_then(|p| c.groups[0].rows[p]).map_or(
                        c.groups[0].explicit_unknown + c.groups[0].pruned_mass,
                        |p| p.mass,
                    )
                });
                local.push((
                    local_parent.unwrap_or(15) as u8,
                    row.extension,
                    prior * row.log_transition.exp() * row.log_potential.exp(),
                ));
            }
            let z = local.iter().map(|r| r.2).sum::<f64>();
            let score = prior
                * proposal.log_transition.exp()
                * prepared.producer.shared_scores[c].exp()
                * z;
            enumerated.push((parent.unwrap_or(7) as u8, proposal.slot, score, z, local));
        }
        let evidence = enumerated.iter().map(|e| e.2).sum::<f64>();
        let output = prepared.advance(&mut owner).unwrap();
        assert_eq!(output.end, interval[1]);
        assert!((output.posterior.log_evidence.unwrap() - evidence.ln()).abs() < 1e-12);
        for c in output.posterior.contexts.iter().flatten() {
            let expected = enumerated
                .iter()
                .find(|e| (e.0, e.1) == (c.weight.parent, c.weight.extension))
                .unwrap();
            assert!((c.weight.mass - expected.2 / evidence).abs() < 1e-12);
            for row in c.groups[0].rows.iter().flatten() {
                let expected_row = expected
                    .4
                    .iter()
                    .find(|r| (r.0, r.1) == (row.parent, row.extension))
                    .unwrap();
                assert!((row.mass - expected_row.2 / expected.3).abs() < 1e-12);
            }
        }
        assert_eq!(
            (producer.rows.capacity(), producer.sources.capacity()),
            capacities
        );
    }
    eprintln!(
        "JOINT_LOCAL_PRODUCER steps=8 tuples={total} row_capacity={} source_capacity={}",
        capacities.0, capacities.1
    );
    let batch = shared
        .produce(
            owner.snapshot(),
            shared::producer::Inputs {
                interval: [800, 900],
                rate: 1000,
                observed: true,
                groups: &[],
                observations: &[],
                returns: &[],
                retained_episodes: &[],
                head: &head,
                rms_reference: gesture.rms_reference,
            },
        )
        .unwrap();
    let input = Input {
        interval: [800, 900],
        rate: 1000,
        observed: true,
        groups: &[],
        retained: &[],
        gesture: &gesture,
        phrase,
        section: &head,
        section_config: &section_config,
    };
    let retired = producer
        .produce(
            owner.snapshot(),
            &batch,
            &input,
            |_| Ok(0.),
            |_, _, _| panic!("retired group scored"),
        )
        .unwrap();
    assert_eq!(retired.producer.work.tuples, 0);
    assert!(
        retired
            .advance(&mut owner)
            .unwrap()
            .groups
            .iter()
            .all(Option::is_none)
    );
    assert_eq!(
        (producer.rows.capacity(), producer.sources.capacity()),
        capacities
    );
}

pub(in crate::temporal_cognition::joint) fn matched_composition(
    frontend: &crate::temporal_cognition::proposals::frontend::Snapshot,
    correspondence: &correspondence::Cache,
    returns: &section::Returns,
    retained: &[(u64, u64)],
) {
    use crate::temporal_cognition::{
        accents::periods::{Estimator, groupings},
        features::Accent,
    };
    let group = correspondence.group;
    let end = correspondence.cut;
    let interval = [end - 100, end];
    let gesture = TemporalGestureConfig {
        rms_reference: 0.1,
        means: [0.; 5],
        deviations: [1.; 5],
        coefficients: [[[0.; 11]; 4]; 4],
    };
    let model = crate::temporal_cognition::section::tests::runtime_config();
    let head = Head {
        means: [0.; 82],
        deviations: [1.; 82],
        hazard: [0.; 83],
        exits: [[0.; 83]; 3],
    };
    let mut acoustics = acoustics::Cache::new(group, interval[0], 1000, 100).unwrap();
    let observation = acoustics.advance(frontend, interval).unwrap();
    let mut estimator = Estimator::new(group, 128, 1000, 32000, 1024, 1. / 24.).unwrap();
    let mut inventory = groupings::Inventory::new(
        group,
        0,
        groupings::Controls {
            tolerance: 0.1,
            integers_234_only: false,
            strict_integer: false,
            one_skip_words: false,
        },
    )
    .unwrap();
    for t in (200..end - 32).step_by(200) {
        estimator
            .deliver(
                Accent {
                    group,
                    event_start: t - 16,
                    event_end: t,
                    raw_intervals: [(t - 48, t - 32), (t - 32, t - 16), (t - 16, t), (t, t + 16)],
                    source_start: t - 48,
                    source_end: t + 16,
                    available_end: t + 32,
                    weight: 1.,
                    observed_prefix: t,
                },
                t + 32,
            )
            .unwrap();
    }
    inventory.refresh(&estimator).unwrap();
    let mut grouping = grouping::Cache::new(group, Some(1.)).unwrap();
    grouping
        .refresh(end, Some((&inventory.snapshot().unwrap(), &estimator)))
        .unwrap();
    let mut missing_grouping = grouping::Cache::new(group, Some(1.)).unwrap();
    missing_grouping.refresh(end, None).unwrap();
    for (has_grouping, grouping) in [(true, &grouping), (false, &missing_grouping)] {
        let mut owner = owner::State::new(group.bus, group.epoch, interval[0], 100).unwrap();
        let mut shared_producer = shared::producer::Producer::new(interval[0]);
        let batch = shared_producer
            .produce(
                owner.snapshot(),
                shared::producer::Inputs {
                    interval,
                    rate: 1000,
                    observed: true,
                    groups: &[group],
                    observations: &[Some(&observation)],
                    returns: &[Some(returns)],
                    retained_episodes: retained,
                    head: &head,
                    rms_reference: 0.1,
                },
            )
            .unwrap();
        let groups = [Group {
            observation: &observation,
            acoustics: &acoustics,
            grouping,
            correspondence,
            returns,
            motion: None,
            phrase_context: [None; 12],
        }];
        let input = Input {
            interval,
            rate: 1000,
            observed: true,
            groups: &groups,
            retained,
            gesture: &gesture,
            phrase: phrase::tests::config(),
            section: &head,
            section_config: &model,
        };
        let mut producer = Producer::new(interval[0]);
        let prepared = producer
            .produce(
                owner.snapshot(),
                &batch,
                &input,
                |_| Ok(0.2),
                |_, _, r| Ok(r.components.iter().flatten().count() as f64 * 0.1),
            )
            .unwrap();
        let tuples = prepared.producer.work.tuples;
        let all = prepared
            .producer
            .rows
            .iter()
            .filter(|r| r.components.iter().all(Option::is_some))
            .count();
        assert_eq!(all > 0, has_grouping);
        let returned = prepared
            .producer
            .rows
            .iter()
            .filter(|r| {
                r.section
                    .is_some_and(|s| prepared.producer.sources[s.source_slot].returned.is_some())
            })
            .count();
        assert!(
            prepared
                .producer
                .sources
                .iter()
                .any(|s| s.returned.is_some())
        );
        if !has_grouping {
            assert!(returned > 0);
        }
        let output = prepared.advance(&mut owner).unwrap();
        let mut kept = 0;
        for p in output
            .paths
            .iter()
            .flatten()
            .flat_map(|c| c.groups[0].iter().flatten())
        {
            if p.components.iter().all(Option::is_some) {
                kept += 1;
            }
            if let Some(s) = p.section_slot.filter(|i| output.sections[*i].context == 70) {
                let state = output.correspondences[p.correspondence_slot.unwrap()].unwrap();
                assert!(p.correspondence_supported && state.matched.is_some());
                assert_eq!(
                    output.sections[s].context,
                    returns
                        .for_correspondence(&state)
                        .unwrap()
                        .context()
                        .context_id
                );
            }
        }
        assert_eq!(kept > 0, has_grouping);
        eprintln!(
            "JOINT_ALL_COMPONENTS grouping={has_grouping} cut={end} tuples={tuples} five_known={all} returned={returned} kept_five={kept}"
        );
    }
}

#[test]
fn missing_boundaries_keep_only_duration_survival_and_send_exits_to_unknown() {
    use proposals::{Kind, List};
    for (kind, tau) in [(Kind::Phrase, 30.), (Kind::Section, 120.)] {
        for dt in [0., 0.1, 1., 1000.] {
            for survival in [0., 0.2, 0.9, 1.] {
                let list = List::missing_boundary(kind, 7, survival, dt).unwrap();
                let expected = survival * (-dt / tau).exp();
                let known = list
                    .entries
                    .iter()
                    .flatten()
                    .filter(|e| e.id.is_some())
                    .map(|e| e.log_weight.exp())
                    .sum::<f64>();
                let unknown = list
                    .entries
                    .iter()
                    .flatten()
                    .find(|e| e.id.is_none())
                    .unwrap()
                    .log_weight
                    .exp();
                assert!((known - expected).abs() < 1e-14);
                assert!((unknown - (1. - expected)).abs() < 1e-14);
            }
        }
    }
    assert!(List::missing_boundary(Kind::Grouping, 7, 0.5, 1.).is_err());
    assert!(List::missing_boundary(Kind::Phrase, 7, f64::NAN, 1.).is_err());
}
