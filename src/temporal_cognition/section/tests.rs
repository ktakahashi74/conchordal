use super::*;
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/temporal_cognition/sections.json"
    ))
    .unwrap()
}

fn assert_values(actual: &[Option<f64>], expected: &Value) {
    let expected = expected.as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        match (a, e.as_f64()) {
            (Some(a), Some(e)) => assert!(
                (a - e).abs() <= 1e-13 * (1. + e.abs()),
                "coordinate {i}: {a} != {e}"
            ),
            (None, None) => (),
            _ => panic!("coordinate {i}: validity differs: {a:?} vs {e}"),
        }
    }
}

fn assert_bytes(record: Record, expected: &Value) {
    let expected = expected.as_str().unwrap();
    let bytes = record.bytes();
    assert_eq!(expected.len(), bytes.len() * 2);
    for (i, byte) in bytes.into_iter().enumerate() {
        assert_eq!(
            byte,
            u8::from_str_radix(&expected[i * 2..i * 2 + 2], 16).unwrap(),
            "byte {i}"
        );
    }
}

#[test]
fn registered_python_records_are_byte_exact_at_every_prefix() {
    for case in fixture()["cases"].as_array().unwrap() {
        let mut history =
            History::new(1, 4, 19, 0., case["ring_size"].as_u64().unwrap() as usize).unwrap();
        for (i, step) in case["steps"].as_array().unwrap().iter().enumerate() {
            let activity = serde_json::from_value(step["activity"].clone()).unwrap();
            let changed = match step["kind"].as_str().unwrap() {
                "observe" => history.observe(activity, 1, 4, 1000, &[]),
                "commit" => history.commit(
                    serde_json::from_value(step["record"].clone()).unwrap(),
                    activity,
                    step["sequence"].as_u64().unwrap(),
                    step["adjacency"].as_f64().unwrap(),
                ),
                _ => unreachable!(),
            }
            .unwrap_or_else(|e| panic!("{} step {i}: {e}", case["name"]));
            assert_eq!(changed, step["replayed"] != true);
            let expected = &step["expected"];
            assert_bytes(history.cumulative, &expected["cumulative_hex"]);
            assert_eq!(
                history.ring.len(),
                expected["ring_hex"].as_array().unwrap().len()
            );
            for (record, bytes) in history
                .ring
                .iter()
                .zip(expected["ring_hex"].as_array().unwrap())
            {
                assert_bytes(*record, bytes);
            }
            assert_values(&history.cumulative.values(), &expected["cumulative"]);
            assert_values(&history.recent().values(), &expected["recent"]);
            assert_values(
                &history
                    .covariates(history.cumulative.time(5), [None; 2])
                    .unwrap(),
                &expected["covariates"],
            );
        }
    }
}

#[test]
fn correspondence_matches_registered_threshold_and_missingness_cases() {
    for case in fixture()["categories"].as_array().unwrap() {
        assert_eq!(
            category(
                serde_json::from_value(case["assignment"].clone()).unwrap(),
                serde_json::from_value(case["ending"].clone()).unwrap(),
                serde_json::from_value(case["predecessor"].clone()).unwrap()
            )
            .unwrap(),
            case["expected"].as_u64().unwrap() as usize,
            "{case}"
        );
    }
}

fn activity(start: f64, end: f64) -> Activity {
    Activity {
        window: [start, end],
        numerators: [0.; 9],
        denominators: [end - start; 9],
        physical_valid_seconds: [end - start; 9],
        assignment_seconds: end - start,
        physical_window_seconds: end - start,
    }
}

#[test]
fn record_layout_and_atomic_failure_keep_original_statistics() {
    assert_eq!(std::mem::size_of::<Record>(), 640);
    let mut record = Record::default();
    record.add_activity(activity(0., 1.), 1., 1.).unwrap();
    for index in 0..9 {
        let before = record.bytes();
        let mut invalid = activity(1., 2.);
        invalid.numerators[index] = f64::INFINITY;
        assert!(record.add_activity(invalid, 1., 1.).is_err());
        assert_eq!(before, record.bytes());
    }
    let before = record.bytes();
    let mut invalid = activity(1., 2.);
    invalid.denominators[3] = 0.5;
    assert!(record.add_activity(invalid, 1., 1.).is_err());
    assert_eq!(before, record.bytes());
    assert!(
        record
            .add_activity(activity(1., 2.), 1., f64::INFINITY)
            .is_err()
    );
    assert_eq!(before, record.bytes());
}

#[test]
fn rounded_shared_denominators_and_signed_zero_match_reference_storage() {
    let mut r = Record::default();
    r.statistics[40] = 2f64.powi(53);
    let mut a = activity(0., 1.);
    a.denominators[..4].copy_from_slice(&[0., 1., 0., 1.]);
    r.add_activity(a, 1., 1.).unwrap();
    assert_eq!(r.statistics[40], 2f64.powi(53));
    let mut r = Record::default();
    r.statistics[40] = -0.;
    let mut a = activity(0., 1.);
    a.denominators[..4].copy_from_slice(&[-0., 0., -0., 0.]);
    let before = r.bytes();
    assert!(r.add_activity(a, 1., 1.).is_err());
    assert_eq!(r.bytes(), before);
}

#[test]
fn gaps_fractional_assignment_replays_and_generation_inheritance() {
    let mut h = History::new(1, 4, 19, 0., 4).unwrap();
    let mut a = activity(0., 1.);
    a.assignment_seconds = 0.1;
    assert!(h.observe(a, 1, 4, 1000, &[]).unwrap());
    assert!(!h.observe(a, 1, 4, 1000, &[]).unwrap());
    assert_eq!(h.covariates(1., [None; 2]).unwrap()[79], Some(0.));
    assert!(h.observe(activity(2., 3.), 1, 4, 1000, &[]).unwrap());
    assert_eq!(h.covariates(3., [None; 2]).unwrap()[79], Some(1. - 2. / 3.));
    assert!(h.observe(activity(1., 2.), 1, 4, 1000, &[]).is_err());
    assert!(!h.observe(activity(3., 4.), 2, 4, 1000, &[]).unwrap());
    let mut child = h.inherit(4, 5).unwrap();
    assert!(child.observe(activity(3., 4.), 1, 5, 1000, &[]).unwrap());
    assert_eq!(child.covariates(4., [None; 2]).unwrap()[79], Some(0.25));
    assert_eq!(h.cumulative.time(5), 3.);
    assert!(h.inherit(3, 5).is_err());
    assert!(h.inherit(4, 4).is_err());
    assert!(History::new(1, 5, 20, 3., 4).unwrap().ring.is_empty());
}

#[test]
fn stable_sum_preserves_threshold_relevant_small_terms() {
    assert_eq!(sum(&[1e16, 1., 1.]), 1e16 + 2.);
    assert_eq!(sum(&[1e16, 1., -1e16]), 1.);
    assert_eq!(sum(&[1., 1e-16, 1e-16]), 1.0000000000000002);
}

fn delivery(sequence: u64, event_end: u64, received_at: u64) -> Delivery {
    use crate::temporal_cognition::{features::Accent, ridge::Handle};
    Delivery {
        sequence,
        received_at,
        accent: Accent {
            group: Handle {
                bus: 0,
                epoch: 1,
                generation: 4,
            },
            event_start: event_end - 100,
            event_end,
            raw_intervals: [
                (event_end - 300, event_end - 200),
                (event_end - 200, event_end - 100),
                (event_end - 100, event_end),
                (event_end, event_end + 100),
            ],
            source_start: event_end - 300,
            source_end: event_end + 100,
            available_end: event_end + 100,
            weight: 0.5,
            observed_prefix: event_end,
        },
    }
}

#[test]
fn late_accent_credits_original_section_once_without_extending_audio() {
    let mut h = History::new(1, 4, 19, 0., 4).unwrap();
    h.observe(activity(0., 1.), 1, 4, 1000, &[]).unwrap();
    let d = delivery(1, 900, 1500);
    h.observe(activity(1., 2.), 1, 4, 1000, &[d]).unwrap();
    assert_eq!(h.cumulative.statistics[35], 0.5);
    assert_eq!(h.cumulative.statistics[52], 2.);
    assert_eq!(h.cumulative.time(4), 0.);
    assert_eq!(h.cumulative.bookkeeping[15], 1);
    let repeat = Delivery {
        received_at: 2500,
        ..d
    };
    h.observe(activity(2., 3.), 1, 4, 1000, &[repeat]).unwrap();
    assert_eq!(h.cumulative.statistics[35], 0.5);
    let mut later = History::new(1, 4, 20, 1., 4).unwrap();
    later.observe(activity(1., 2.), 1, 4, 1000, &[d]).unwrap();
    assert_eq!(later.cumulative.statistics[35], 0.);
    assert_eq!(later.cumulative.bookkeeping[15], 1);
    let mut child = h.inherit(4, 5).unwrap();
    let mut next = delivery(1, 3300, 3500);
    next.accent.group.generation = 5;
    child
        .observe(activity(3., 4.), 1, 5, 1000, &[next])
        .unwrap();
    assert_eq!(child.cumulative.statistics[35], 1.);
    assert_eq!(h.cumulative.statistics[35], 0.5);
}

#[test]
fn invalid_delivery_batch_leaves_both_watermark_and_activity_unchanged() {
    let mut h = History::new(1, 4, 19, 0., 4).unwrap();
    let d = delivery(1, 500, 900);
    let before = h.cumulative.bytes();
    for input in [
        vec![d, d],
        vec![delivery(2, 600, 900), d],
        vec![delivery(1, 950, 990)],
        vec![Delivery {
            received_at: 1500,
            ..d
        }],
    ] {
        assert!(h.observe(activity(0., 1.), 1, 4, 1000, &input).is_err());
        assert_eq!(h.cumulative.bytes(), before);
    }
    let mut counted = activity(0., 1.);
    counted.numerators[4] = 0.5;
    assert!(h.observe(counted, 1, 4, 1000, &[d]).is_err());
    assert_eq!(h.cumulative.bytes(), before);
    let mut invalid = activity(0., 1.);
    invalid.physical_window_seconds = f64::NAN;
    assert!(h.observe(invalid, 1, 4, 1000, &[]).is_err());
    assert_eq!(h.cumulative.bytes(), before);
}

#[test]
fn old_evicted_commits_require_reconciliation_and_cannot_recredit() {
    let cases = fixture();
    let case = &cases["cases"][0];
    let steps = case["steps"].as_array().unwrap();
    let mut h = History::new(1, 4, 19, 0., 2).unwrap();
    for step in steps {
        let activity = serde_json::from_value(step["activity"].clone()).unwrap();
        if step["kind"] == "observe" {
            h.observe(activity, 1, 4, 1000, &[]).unwrap();
        } else {
            h.commit(
                serde_json::from_value(step["record"].clone()).unwrap(),
                activity,
                step["sequence"].as_u64().unwrap(),
                step["adjacency"].as_f64().unwrap(),
            )
            .unwrap();
        }
    }
    let step = &steps[1];
    let before = h.cumulative.bytes();
    assert!(
        h.commit(
            serde_json::from_value(step["record"].clone()).unwrap(),
            serde_json::from_value(step["activity"].clone()).unwrap(),
            1,
            1.
        )
        .is_err()
    );
    assert_eq!(h.cumulative.bytes(), before);
    assert_eq!(h.ring.len(), 2);
    let child = h.inherit(4, 5).unwrap();
    assert_eq!(child.ring, h.ring);
    assert!(child.ring.capacity() >= child.ring_size);
}

#[test]
fn predecessor_before_section_start_classifies_departure_without_adding_mass() {
    let data = fixture();
    let mut first: Commit =
        serde_json::from_value(data["cases"][0]["steps"][1]["record"].clone()).unwrap();
    first.start = 0.;
    first.support_end = 1.;
    first.assignment_seconds = 1.;
    first.membership = 1.;
    let mut h = History::new(1, 4, 19, 1., 4).unwrap();
    h.observe(activity(1., 3.), 1, 4, 1000, &[]).unwrap();
    h.commit(first, activity(0., 1.), 1, 1.).unwrap();
    assert!(h.ring.is_empty());
    assert_eq!(&h.cumulative.values()[..30], &[None; 30]);
    let second = Commit {
        occurrence_id: 2,
        start: 1.,
        support_end: 2.,
        assignment: Assignment {
            status: Status::NoMemory,
            cost: None,
            ..first.assignment
        },
        ending_descriptor: [Some(2.); 6],
        ..first
    };
    h.commit(second, activity(1., 2.), 2, 1.).unwrap();
    assert_eq!(h.cumulative.values()[2], Some(1.));
    assert_eq!(&h.cumulative.values()[5..30], &[None; 25]);
    assert_eq!(h.ring.len(), 1);
}

#[test]
fn covariance_missingness_is_per_coordinate_and_elapsed_uses_physical_time() {
    let mut h = History::new(1, 4, 19, 0., 4).unwrap();
    let empty = h.covariates(0., [None; 2]).unwrap();
    assert_eq!(empty, [None; 82]);
    h.observe(activity(0., 1.), 1, 4, 1000, &[]).unwrap();
    let values = h.covariates(1., [Some(3.), Some(0.25)]).unwrap();
    assert_eq!(values[30], Some(0.));
    assert_eq!(values[39 + 30], None);
    assert_eq!(values[78], Some(1f64.ln_1p()));
    assert_eq!(values[79], Some(0.));
    assert_eq!(&values[80..], &[Some(3.), Some(0.25)]);
    assert!(h.covariates(0.5, [None; 2]).is_err());
    assert!(h.covariates(1., [Some(f64::INFINITY), None]).is_err());
}

#[test]
fn section_head_matches_scaled_masked_82_input_reference_and_frozen_forecasts() {
    let data: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/temporal_cognition/section_hazards.json"
    ))
    .unwrap();
    for row in data["heads"].as_array().unwrap() {
        let values = |key: &str| {
            row[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect::<Vec<_>>()
        };
        let head = Head {
            means: values("means").try_into().unwrap(),
            deviations: values("deviations").try_into().unwrap(),
            hazard: values("hazard").try_into().unwrap(),
            exits: std::array::from_fn(|i| {
                row["exits"][i]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            }),
        };
        head.validate().unwrap();
        let raw: [Option<f64>; 82] = row["raw"]
            .as_array()
            .unwrap()
            .iter()
            .map(Value::as_f64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let before = raw;
        for step in row["steps"].as_array().unwrap() {
            let actual = head
                .law(
                    &raw,
                    step["lo"].as_f64().unwrap(),
                    step["hi"].as_f64().unwrap(),
                )
                .unwrap();
            if let Some(actual) = actual {
                assert_values(&actual.map(Some), &step["expected"]);
                assert!((actual.iter().sum::<f64>() - 1.).abs() < 1e-12);
            } else {
                assert!(step["expected"].is_null());
            }
        }
        assert_eq!(raw, before);
    }
}

#[test]
fn section_head_requires_centered_logits_and_known_start() {
    let mut head = Head {
        means: [0.; 82],
        deviations: [1.; 82],
        hazard: [0.; 83],
        exits: [[0.; 83]; 3],
    };
    assert!(head.law(&[None; 82], 0., 1.).unwrap().is_none());
    head.validate().unwrap();
    for coordinate in [0, 1, 79, 82] {
        head.exits[0][coordinate] = 1.;
        assert!(head.validate().is_err());
        head.exits[0][coordinate] = 0.;
    }
    head.deviations[81] = -1.;
    assert!(head.validate().is_err());
    head.deviations[81] = 1.;
    head.hazard[0] = 20.;
    head.hazard[79] = -40.;
    let mut raw = [None; 82];
    raw[78] = Some(0.);
    assert!(head.law(&raw, 0., 1200.).unwrap().is_none());
}

#[test]
fn retrieval_uses_distinct_episodes_and_original_audio_age() {
    use crate::temporal_cognition::transport::Identity;
    let a = Identity {
        id: 1,
        generation: 3,
    };
    let b = Identity {
        id: 2,
        generation: 3,
    };
    let candidates = [
        (a, Some(1.)),
        (a, Some(3.)),
        (a, Some(2.9)),
        (b, Some(2.)),
        (b, None),
    ];
    assert_eq!(
        retrieval_scores(candidates.iter().copied(), Some(1000), 1100, 1499, 1000)
            .unwrap()
            .values,
        [Some(3.), Some(1.)]
    );
    assert_eq!(
        retrieval_scores(candidates.iter().copied(), Some(1000), 1499, 1500, 1000)
            .unwrap()
            .values,
        [None; 2]
    );
    assert_eq!(
        retrieval_scores(
            candidates[..3].iter().copied(),
            Some(1000),
            1100,
            1100,
            1000
        )
        .unwrap()
        .values,
        [Some(3.), None]
    );
    assert_eq!(
        retrieval_scores([], Some(1000), 1000, 1000, 1000)
            .unwrap()
            .values,
        [None; 2]
    );
    assert_eq!(
        retrieval_scores(candidates.iter().copied(), None, 1000, 1000, 1000)
            .unwrap()
            .values,
        [None; 2]
    );
    assert!(retrieval_scores(candidates.iter().copied(), Some(1001), 1000, 1100, 1000).is_err());
    assert!(retrieval_scores(candidates.iter().copied(), Some(1000), 1101, 1100, 1000).is_err());
    let full: Vec<_> = (1..=17)
        .map(|id| (Identity { id, generation: 1 }, Some(id as f64)))
        .collect();
    assert_eq!(
        retrieval_scores(full.iter().copied(), Some(1000), 1000, 1000, 1000)
            .unwrap()
            .values,
        [Some(17.), Some(1.)]
    );
    let oversized = vec![(a, Some(1.)); 4 * super::super::memory::MAX_CANDIDATES + 1];
    assert!(retrieval_scores(oversized.iter().copied(), Some(1000), 1000, 1000, 1000).is_err());
    assert_eq!(
        retrieval_scores(full[..16].iter().copied(), Some(1000), 1000, 1000, 1000)
            .unwrap()
            .values,
        [Some(16.), Some(1.)]
    );
    let ties = [(a, Some(1.)), (b, Some(1.))];
    assert_eq!(
        retrieval_scores(ties.iter().copied(), Some(1000), 1000, 1000, 1000)
            .unwrap()
            .values,
        [Some(1.), Some(0.)]
    );
}

pub(in crate::temporal_cognition) fn runtime_config() -> crate::config::TemporalSectionConfig {
    crate::config::TemporalSectionConfig {
        means: [0.; 82],
        deviations: [1.; 82],
        hazard: [0.; 83],
        new_context: [0.; 83],
        recurrence: [0.; 83],
        contrast: [0.; 83],
        ending_means: [0.; 6],
        ending_deviations: [1.; 6],
        match_means: [0.; 14],
        match_deviations: [1.; 14],
        match_coefficients: [0.; 15],
    }
}

#[test]
fn live_phrase_receipts_preserve_conditional_histories_and_mask_absent_inputs() {
    use crate::temporal_cognition::phrase::{self, tests as fixture};
    let mut config = fixture::config();
    config.hazard[0] = 20.;
    let mut phrase = phrase::Phrase::new(1, 0, 0, 1000, 100, config).unwrap();
    let mut gesture = fixture::gesture_model();
    let mut section = Stream::new(
        1,
        0,
        1000,
        0,
        100,
        crate::config::TemporalMemoryConfig {
            retention: None,
            candidates: None,
            scales: [1.; 10],
            span_hops: 8,
            episodes: 16,
            query_cadence_ms: 100,
            deadline_ms: 200,
        },
        runtime_config(),
    )
    .unwrap();
    let mut closed = 0;
    let mut previous_paths = std::collections::BTreeMap::new();
    let mut seen_paths = std::collections::BTreeSet::new();
    let mut saw_fork = false;
    let mut saw_shared_context = false;
    for step in (1..=20).chain(35..=60) {
        let a = fixture::input(step, 0.);
        gesture.advance(&a, &fixture::ridges(), step * 100).unwrap();
        phrase
            .advance(&a, &gesture, None, None, step * 100)
            .unwrap();
        section.advance(&a, &phrase.snapshot(), None, None).unwrap();
        let state = section.snapshot();
        let g = state.groups[0].unwrap();
        let parents: std::collections::BTreeMap<_, _> = g
            .parents
            .iter()
            .flatten()
            .map(|p| (p.path_id, p.phrase_path_index))
            .collect();
        assert_eq!(parents.len(), g.parents.iter().flatten().count());
        if !previous_paths.is_empty() {
            assert_eq!(parents, previous_paths);
        } else {
            assert_eq!(parents.len(), 1);
        }
        let mut children = std::collections::BTreeMap::new();
        let mut contexts = std::collections::BTreeMap::new();
        previous_paths.clear();
        let phrase_snapshot = phrase.snapshot();
        let phrase_group = phrase_snapshot.groups[0].unwrap();
        assert_eq!(g.previous_end_sample, phrase_group.previous_end_sample);
        for candidate in g.candidates.iter().flatten() {
            assert!(seen_paths.insert(candidate.path_id));
            assert!(candidate.path_id > candidate.parent_path_id);
            let parent_phrase = parents[&candidate.parent_path_id];
            let phrase_child = phrase_group.candidates[candidate.phrase_path_index].unwrap();
            assert_eq!(phrase_child.parent_index, parent_phrase);
            previous_paths.insert(candidate.path_id, candidate.phrase_path_index);
            *children.entry(candidate.parent_path_id).or_insert(0) += 1;
            *contexts.entry(candidate.context_id).or_insert(0) += 1;
        }
        saw_fork |= children.values().any(|n| *n > 1);
        saw_shared_context |= contexts.values().any(|n| *n > 1);
        closed = closed.max(g.observed_spans);
        assert!(!state.initial_context_only);
        assert_eq!(g.covariates[80..], [None; 2]);
        assert_eq!(g.cumulative[30..34], [None; 4]);
        assert_eq!(g.cumulative[35..38], [None; 3]);
        assert!((g.survival + g.exits.iter().sum::<f64>() + g.unknown - 1.).abs() < 1e-12);
        assert_eq!(g.matched_prefixes, 0);
        if step <= 20 {
            assert!(g.covariates[79].unwrap().abs() < 1e-12);
        }
        if step == 35 {
            assert!(g.covariates[79].unwrap() > 0.3);
        }
        assert!(section.advance(&a, &phrase.snapshot(), None, None).is_err());
    }
    assert!(closed > 0);
    assert!(saw_fork && saw_shared_context);
    section.finish(7000).unwrap();
    let g = section.snapshot().groups[0].unwrap();
    assert!(section.snapshot().censored);
    assert_eq!(g.unknown, 1.);
    assert_eq!(g.exits, [0.; 3]);
}

#[test]
fn explicit_section_configuration_round_trips_and_rejects_wrong_shape_or_uncentered_logits() {
    let config = runtime_config();
    let text = toml::to_string(&config).unwrap();
    let read: crate::config::TemporalSectionConfig = toml::from_str(&text).unwrap();
    Stream::validate(read).unwrap();
    let mut invalid = config;
    invalid.contrast[82] = 1.;
    assert!(Stream::validate(invalid).is_err());
    let mut value = serde_json::to_value(config).unwrap();
    value["means"].as_array_mut().unwrap().pop();
    assert!(serde_json::from_value::<crate::config::TemporalSectionConfig>(value).is_err());
    let mut app = crate::config::AppConfig::default();
    app.temporal_section = Some(config);
    assert!(app.validate().is_err());
}

#[test]
fn pending_ring_accepts_a_boundary_accent_once_then_seals_original_statistics() {
    let mut h = History::new(1, 4, 19, 0., 4).unwrap();
    h.observe(activity(0., 2.), 1, 4, 1000, &[]).unwrap();
    let record = Commit {
        epoch: 1,
        occurrence_id: 7,
        start: 0.,
        support_end: 1.,
        assignment_seconds: 1.,
        membership: 1.,
        ordering_known: true,
        ending_descriptor: [Some(3.); 6],
        ending_generation: Some(4),
        assignment: Assignment {
            status: Status::Unresolved,
            cost: None,
            supported: false,
            search_completed: false,
            search_covered: false,
            search_nonempty: false,
            frequency_shift_log2: None,
            tempo_shift_log2: None,
            bound_hit: false,
            ambiguous_cutoff: false,
        },
    };
    h.commit(record, activity(0., 1.), 1, 1.).unwrap();
    h.commit(
        Commit {
            occurrence_id: 8,
            start: 1.,
            support_end: 2.,
            ..record
        },
        activity(1., 2.),
        2,
        1.,
    )
    .unwrap();
    let original_ending = h.ring[0].ending();
    let d = delivery(1, 1000, 1500);
    assert_eq!(h.admit_pending(d, 1000), 0.);
    assert_eq!(h.admit_pending(d, 1000), 0.);
    assert_eq!(h.ring[0].statistics[35], 0.5);
    assert_eq!(h.ring[1].statistics[35], 0.);
    assert_eq!(h.cumulative.statistics[35], 0.);
    assert_eq!(h.admit_pending(delivery(2, 900, 1501), 1000), 0.5);
    assert_eq!(h.admit_pending(delivery(2, 900, 1501), 1000), 0.);
    assert_eq!(h.ring[0].statistics[35], 0.5);
    assert_eq!(h.ring[0].ending(), original_ending);
}

#[test]
fn section_rejects_wrong_parent_receipts_before_mutating_contexts() {
    use crate::temporal_cognition::phrase::{self, tests as fixture};
    let create = || {
        Stream::new(
            1,
            0,
            1000,
            0,
            100,
            crate::config::TemporalMemoryConfig {
                retention: None,
                candidates: None,
                scales: [1.; 10],
                span_hops: 8,
                episodes: 16,
                query_cadence_ms: 100,
                deadline_ms: 200,
            },
            runtime_config(),
        )
        .unwrap()
    };
    let mut cfg = fixture::config();
    cfg.hazard[0] = 20.;
    let mut phrase = phrase::Phrase::new(1, 0, 0, 1000, 100, cfg).unwrap();
    let mut gesture = fixture::gesture_model();
    let mut section = create();
    let mut control = create();
    for step in 1..=4 {
        let a = fixture::input(step, 0.);
        gesture.advance(&a, &fixture::ridges(), step * 100).unwrap();
        phrase
            .advance(&a, &gesture, None, None, step * 100)
            .unwrap();
        let receipt = phrase.snapshot();
        let original = serde_json::to_string(section.snapshot()).unwrap();
        if step > 1 {
            for fault in 0..8 {
                let mut bad = receipt;
                let mut acoustic = a;
                match fault {
                    0 => bad.groups[0].as_mut().unwrap().previous_end_sample -= 1,
                    1 => {
                        bad.groups[0].as_mut().unwrap().candidates[0]
                            .as_mut()
                            .unwrap()
                            .parent_index = 15
                    }
                    2 => bad.groups[0].as_mut().unwrap().candidates[0] = None,
                    3 => bad.sample_rate = 48000,
                    4 => acoustic.assignment.end_sample -= 1,
                    5 => bad.censored = true,
                    6 => {
                        bad.groups[1] = bad.groups[0];
                    }
                    _ => {
                        // A valid first group must not advance before a later group fails.
                        bad.groups[1] = bad.groups[0];
                        bad.groups[1].as_mut().unwrap().group.epoch += 1;
                    }
                }
                assert!(
                    section.advance(&acoustic, &bad, None, None).is_err(),
                    "fault {fault}"
                );
                assert_eq!(original, serde_json::to_string(section.snapshot()).unwrap());
            }
        }
        section.advance(&a, &receipt, None, None).unwrap();
        control.advance(&a, &receipt, None, None).unwrap();
        assert_eq!(
            serde_json::to_string(section.snapshot()).unwrap(),
            serde_json::to_string(control.snapshot()).unwrap()
        );
        assert!(section.advance(&a, &receipt, None, None).is_err());
    }
}

#[test]
fn skipped_phrase_receipt_cannot_reuse_an_in_range_parent_slot() {
    use crate::temporal_cognition::phrase::{self, tests as fixture};
    let mut cfg = fixture::config();
    cfg.hazard[0] = -1000.;
    let mut phrase = phrase::Phrase::new(1, 0, 0, 1000, 100, cfg).unwrap();
    let mut gesture = fixture::gesture_model();
    let mut section = Stream::new(
        1,
        0,
        1000,
        0,
        100,
        crate::config::TemporalMemoryConfig {
            retention: None,
            candidates: None,
            scales: [1.; 10],
            span_hops: 8,
            episodes: 16,
            query_cadence_ms: 100,
            deadline_ms: 200,
        },
        runtime_config(),
    )
    .unwrap();
    let mut second = None;
    for step in 1..=3 {
        let a = fixture::input(step, 0.);
        gesture.advance(&a, &fixture::ridges(), step * 100).unwrap();
        phrase
            .advance(&a, &gesture, None, None, step * 100)
            .unwrap();
        let receipt = phrase.snapshot();
        assert_eq!(
            receipt.groups[0]
                .unwrap()
                .candidates
                .iter()
                .flatten()
                .count(),
            1
        );
        assert_eq!(
            receipt.groups[0].unwrap().candidates[0]
                .unwrap()
                .parent_index,
            0
        );
        if step == 1 {
            section.advance(&a, &receipt, None, None).unwrap();
        }
        if step == 2 {
            second = Some((a, receipt));
        }
        if step == 3 {
            let before = serde_json::to_string(section.snapshot()).unwrap();
            assert!(section.advance(&a, &receipt, None, None).is_err());
            assert_eq!(before, serde_json::to_string(section.snapshot()).unwrap());
            let (old_a, old_p) = second.unwrap();
            section.advance(&old_a, &old_p, None, None).unwrap();
            section.advance(&a, &receipt, None, None).unwrap();
            assert_eq!(section.snapshot().end_sample, 300);
        }
    }
}
