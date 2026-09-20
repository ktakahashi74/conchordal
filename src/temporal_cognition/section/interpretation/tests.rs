use super::*;
use crate::temporal_cognition::{
    features::Accent,
    section::{Assignment, Status},
};

fn group() -> Handle {
    Handle {
        bus: 0,
        epoch: 1,
        generation: 4,
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

fn filled() -> Interpretation {
    let mut state = Interpretation::new(group(), 0, 1000, 40).unwrap();
    for step in 1..=4 {
        let delta = activity((step - 1) as f64, step as f64);
        let record = Commit {
            epoch: 1,
            occurrence_id: step,
            start: delta.window[0],
            support_end: delta.window[1],
            assignment_seconds: 1.,
            membership: 1.,
            ordering_known: true,
            ending_descriptor: [Some(0.); 6],
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
            ending_generation: Some(4),
        };
        state
            .advance(
                Input {
                    group: group(),
                    interval: [(step - 1) * 1000, step * 1000],
                    rate: 1000,
                    observed: true,
                    delta,
                    deliveries: &[],
                    completed: Some((record, delta, step, 1.)),
                    retrieval: [None; 2],
                    query: None,
                },
                Change {
                    transition: Transition::Stay,
                    context: 40,
                    focus: None,
                },
            )
            .unwrap();
    }
    state
}

#[test]
fn section_interpretation_preserves_history_and_reuses_storage_across_all_transitions() {
    let parent = filled();
    assert_eq!(parent.history.ring.len(), 4);
    let before = parent.history.cumulative.bytes();
    let focus = recall::MatchSnapshot {
        episode_id: 9,
        episode_generation: 2,
        support_start_sample: 0,
        support_end_sample: 1000,
        source_start_sample: 0,
        available_at: 1100,
        cost: 0.1,
        transformation: [Some(0.1), Some(0.)],
        ambiguous: false,
        path_steps: 10,
        anchor: Some(0),
        residuals: None,
    };
    let mut child = Interpretation::new(group(), 0, 1000, 40).unwrap();
    let pointers = (child.history.ring.as_ptr(), child.owners.as_ptr());
    for transition in [
        Transition::Stay,
        Transition::NewContext,
        Transition::Return,
        Transition::Contrast,
        Transition::Development,
    ] {
        child.clone_from(&parent);
        assert_eq!(child.history.cumulative.bytes(), before);
        assert_eq!(child.history.ring, parent.history.ring);
        assert_eq!(child.values, parent.values);
        let stays = matches!(transition, Transition::Stay | Transition::Development);
        let context = if stays {
            40
        } else if transition == Transition::Return {
            10
        } else {
            41
        };
        child
            .advance(
                Input {
                    group: group(),
                    interval: [4000, 5000],
                    rate: 1000,
                    observed: true,
                    delta: activity(4., 5.),
                    deliveries: &[],
                    completed: None,
                    retrieval: [Some(-1.), Some(-2.)],
                    query: None,
                },
                Change {
                    transition,
                    context,
                    focus: matches!(transition, Transition::Return | Transition::Development)
                        .then_some(focus),
                },
            )
            .unwrap();
        assert_eq!(
            (child.history.ring.as_ptr(), child.owners.as_ptr()),
            pointers
        );
        assert_eq!(child.end, 5000);
        assert_eq!(child.context, context);
        assert_eq!(child.start, if stays { 0 } else { 5000 });
        assert_eq!(child.history.ring.len(), if stays { 4 } else { 0 });
        assert_eq!(child.owners, parent.owners);
        assert_eq!(child.values[78], stays.then_some(5_f64.ln_1p()));
        assert_eq!(child.values[80..], [Some(-1.), Some(-2.)]);
        match transition {
            Transition::Stay => assert!(matches!(child.relation, Relation::Initial)),
            Transition::NewContext => assert!(matches!(child.relation, Relation::NewContext)),
            Transition::Return => {
                assert!(matches!(child.relation, Relation::TransformedRecurrence))
            }
            Transition::Contrast => assert!(matches!(child.relation, Relation::Contrast)),
            Transition::Development => assert!(matches!(child.relation, Relation::Development)),
        }
    }
    assert_eq!(parent.history.cumulative.bytes(), before);
    println!(
        "JOINT_SECTION_STATE transitions=5 payload_bytes={} history_capacity={} owner_capacity={}",
        std::mem::size_of::<Interpretation>(),
        child.history.ring.capacity(),
        child.owners.capacity()
    );
}

#[test]
fn section_missing_input_and_delayed_accents_keep_original_support() {
    let mut state = filled();
    let input = Input {
        group: group(),
        interval: [4000, 5000],
        rate: 1000,
        observed: false,
        delta: Activity {
            window: [4., 5.],
            numerators: [0.; 9],
            denominators: [0.; 9],
            physical_valid_seconds: [0.; 9],
            assignment_seconds: 0.,
            physical_window_seconds: 1.,
        },
        deliveries: &[],
        completed: None,
        retrieval: [None; 2],
        query: None,
    };
    let before = state.history.cumulative.bytes();
    assert!(
        state
            .advance(
                Input {
                    delta: activity(3., 4.),
                    ..input
                },
                Change {
                    transition: Transition::Stay,
                    context: 40,
                    focus: None
                }
            )
            .is_err()
    );

    assert!(
        state
            .advance(
                input,
                Change {
                    transition: Transition::NewContext,
                    context: 41,
                    focus: None
                }
            )
            .is_err()
    );
    assert_eq!(state.history.cumulative.bytes(), before);
    state
        .advance(
            input,
            Change {
                transition: Transition::Stay,
                context: 40,
                focus: None,
            },
        )
        .unwrap();
    assert_eq!(state.history.cumulative.statistics[47], 4.);
    assert!((state.values[79].unwrap() - 0.2).abs() < 1e-14);
    assert_eq!(state.history.ring.last().unwrap().time(5), 4.);
    assert_eq!(state.end, 5000);
    let mut state = filled();
    let delivery = |sequence, end, received| Delivery {
        sequence,
        received_at: received,
        accent: Accent {
            group: group(),
            event_start: end - 16,
            event_end: end,
            raw_intervals: [
                (end - 48, end - 32),
                (end - 32, end - 16),
                (end - 16, end),
                (end, end + 16),
            ],
            source_start: end - 48,
            source_end: end + 16,
            available_end: end + 32,
            weight: 0.25,
            observed_prefix: end,
        },
    };
    let deliveries = [delivery(1, 3500, 4500), delivery(2, 3550, 4600)];
    state
        .advance(
            Input {
                observed: true,
                delta: activity(4., 5.),
                deliveries: &deliveries,
                ..input
            },
            Change {
                transition: Transition::Stay,
                context: 40,
                focus: None,
            },
        )
        .unwrap();
    assert_eq!(state.history.cumulative.statistics[35], 0.5);
    assert_eq!(state.history.ring.last().unwrap().statistics[35], 0.25);
    assert_eq!(state.late_accent_support, 0.25);
    assert_eq!(state.owners[0].1, 4000);
}
