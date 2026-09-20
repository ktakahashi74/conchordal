use super::*;
use crate::temporal_cognition::{phrase::tests as fixture, recall};

pub(in crate::temporal_cognition) fn base() -> (phrase::Snapshot, phrase::Candidate) {
    let a = fixture::input(1, 0.);
    let mut g = fixture::gesture_model();
    g.advance(&a, &fixture::ridges(), 100).unwrap();
    let mut p = phrase::Phrase::new(1, 0, 0, 1000, 100, fixture::config()).unwrap();
    p.advance(&a, &g, None, None, 100).unwrap();
    let p = p.snapshot();
    (p, p.groups[0].unwrap().candidates[0].unwrap())
}

#[test]
fn ongoing_cues_match_python_support_ties_aliases_and_missing_windows() {
    let data: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/temporal_cognition/section_cues.json"
    ))
    .unwrap();
    for case in data["cases"].as_array().unwrap() {
        let (mut phrase, base) = base();
        let mut cues = Stream::new(1, 0, 1000, 100, 0, [1.; 10]).unwrap();
        for row in case["steps"].as_array().unwrap() {
            let end = row["end"].as_u64().unwrap();
            let mut acoustic = fixture::input(end / 100, 0.);
            acoustic.assignment.rows[0].as_mut().unwrap().weights[0] =
                row["weight"].as_f64().unwrap();
            if row["missing"].as_bool().unwrap() {
                acoustic.features[0] = None;
                acoustic.eligible[0] = false;
            }
            phrase.end_sample = end;
            let group = phrase.groups[0].as_mut().unwrap();
            group.candidates.fill(None);
            let mut n = 0;
            for f in row["foregrounds"].as_array().unwrap() {
                let mut c = base;
                c.foreground = Some(phrase::Foreground {
                    id: f[0].as_u64().unwrap(),
                    credit: f[0].as_u64().unwrap(),
                    start: f[1].as_u64().unwrap(),
                    heard_end: f[2].as_u64().unwrap(),
                });
                c.mass = if n == 0 { 0.999 } else { 0.0001 };
                group.candidates[n] = Some(c);
                n += 1;
            }
            // A reinterpreted alias supplies no second descriptor or selection weight.
            let mut alias = group.candidates[n - 1].unwrap();
            alias.foreground.as_mut().unwrap().id += 10000;
            group.candidates[n] = Some(alias);
            let handle = group.group;
            cues.advance(&acoustic, &phrase).unwrap();
            let expected = &row["expected"];
            let selection = cues.selection(handle);
            if expected.is_null() {
                assert!(selection.is_none());
                continue;
            }
            let selection = selection.unwrap();
            assert_eq!(
                selection.occurrence_id,
                expected["occurrence_id"].as_u64().unwrap(),
                "end={end}"
            );
            assert!(
                (selection.weighted_seconds
                    - expected["selection_weighted_seconds"].as_f64().unwrap())
                .abs()
                    < 1e-12
            );
            assert_eq!(
                selection.last_observed_sample,
                (expected["last_observed_end"].as_f64().unwrap() * 1000.).round() as u64
            );
            let (frozen, _) = cues.prefix(handle, end).unwrap().unwrap();
            assert_eq!(
                (frozen.start * 1000.).round() as u64,
                selection.start_sample
            );
            assert_eq!(
                (frozen.end * 1000.).round() as u64,
                selection.support_end_sample
            );
            assert_eq!(cues.groups[0].as_ref().unwrap().entries.len(), n);
            assert!(cues.groups[0].as_ref().unwrap().history.len() <= 5);
            assert!(cues.prefix(handle, end + 1).is_err());
        }
    }
}

#[test]
fn reused_matcher_receives_the_selected_whole_prefix_and_stable_occurrence() {
    let cfg = crate::config::TemporalMemoryConfig {
        retention: None,
        candidates: None,
        scales: [1.; 10],
        span_hops: 8,
        episodes: 16,
        query_cadence_ms: 100,
        deadline_ms: 200,
    };
    let mut memory = recall::Recall::new(1, 0, 1000, 100, cfg).unwrap();
    let mut cues = Stream::new(1, 0, 1000, 100, 0, [1.; 10]).unwrap();
    let (mut phrase, base) = base();
    let mut long_prefix = false;
    let mut returning_match = false;
    for step in 1..=300 {
        let end = step * 100;
        let a = fixture::input(step, 0.);
        phrase.end_sample = end;
        let group = phrase.groups[0].as_mut().unwrap();
        group.candidates.fill(None);
        let (credit, start) = if step <= 10 { (30, 0) } else { (40, 1000) };
        group.candidates[0] = Some(phrase::Candidate {
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
        cues.advance(&a, &phrase).unwrap();
        memory.advance(&a, end, Some(&cues)).unwrap();
        if let Some(q) = memory.snapshot().latest {
            let selected = q.cue.unwrap();
            assert_eq!(q.support_start_sample, selected.start_sample);
            assert_eq!(q.support_end_sample, selected.support_end_sample);
            assert!(q.received_at > selected.selected_at);
            assert_eq!(
                selected.occurrence_id,
                if selected.selected_at <= 1000 { 30 } else { 40 }
            );
            long_prefix |= q.support_end_sample - q.support_start_sample > 500;
            returning_match |= selected.occurrence_id == 40 && q.best.is_some();
        }
    }
    assert!(
        long_prefix,
        "the half-second selector must not trim the cue descriptor"
    );
    assert!(
        returning_match,
        "a later occurrence must reach retained earlier material"
    );
    assert!(memory.snapshot().queries <= 300);
    let (frozen, _) = cues
        .prefix(phrase.groups[0].unwrap().group, 30000)
        .unwrap()
        .unwrap();
    assert_eq!(frozen.blocks.len(), 128);
    assert_eq!(frozen.start, 1.);
    assert_eq!(frozen.end, 30.);
    assert!(frozen.reconstruction_error.is_finite());
    assert!(
        memory
            .advance(&fixture::input(301, 0.), 30100, None)
            .is_err()
    );
}

#[test]
fn retired_generation_and_conflicting_aliases_cannot_reuse_a_cue() {
    let (mut p, base) = base();
    let a = fixture::input(1, 0.);
    let mut cues = Stream::new(1, 0, 1000, 100, 0, [1.; 10]).unwrap();
    cues.advance(&a, &p).unwrap();
    let old = p.groups[0].unwrap().group;
    assert!(cues.selection(old).is_some());
    let mut empty = p;
    empty.end_sample = 200;
    empty.groups.fill(None);
    cues.advance(&fixture::input(2, 0.), &empty).unwrap();
    assert!(cues.selection(old).is_none());
    assert!(cues.prefix(old, 200).unwrap().is_none());
    p.end_sample = 300;
    let g = p.groups[0].as_mut().unwrap();
    g.group.generation += 1;
    g.previous_end_sample = 200;
    g.candidates.fill(None);
    g.candidates[0] = Some(phrase::Candidate {
        foreground: Some(phrase::Foreground {
            id: 1,
            credit: 1,
            start: 200,
            heard_end: 300,
        }),
        ..base
    });
    let handle = g.group;
    let mut a = fixture::input(3, 0.);
    a.group_handles[0] = Some(handle);
    a.features[0].as_mut().unwrap().raw.group = handle;
    cues.advance(&a, &p).unwrap();
    let selected = cues.selection(handle).unwrap();
    assert_eq!(selected.start_sample, 200);
    assert!((selected.weighted_seconds - 0.1).abs() < 1e-12);
    assert!(cues.selection(old).is_none());
    p.end_sample = 400;
    let g = p.groups[0].as_mut().unwrap();
    g.candidates[0]
        .as_mut()
        .unwrap()
        .foreground
        .as_mut()
        .unwrap()
        .heard_end = 400;
    g.candidates[1] = g.candidates[0];
    g.candidates[1]
        .as_mut()
        .unwrap()
        .foreground
        .as_mut()
        .unwrap()
        .start = 250;
    let mut a = fixture::input(4, 0.);
    a.group_handles[0] = Some(handle);
    a.features[0].as_mut().unwrap().raw.group = handle;
    assert_eq!(
        cues.advance(&a, &p).unwrap_err(),
        "conflicting phrase cue aliases"
    );
}

#[test]
fn half_second_selection_keeps_fractional_sample_boundaries_at_odd_rates() {
    let (mut p, base) = base();
    let mut cues = Stream::new(1, 0, 1001, 100, 0, [1.; 10]).unwrap();
    for step in 1..=10 {
        p.end_sample = step * 100;
        let g = p.groups[0].as_mut().unwrap();
        g.candidates.fill(None);
        g.candidates[0] = Some(phrase::Candidate {
            foreground: Some(phrase::Foreground {
                id: 1,
                credit: 1,
                start: 0,
                heard_end: step * 100,
            }),
            ..base
        });
        cues.advance(&fixture::input(step, 0.), &p).unwrap();
    }
    let selected = cues.selection(p.groups[0].unwrap().group).unwrap();
    assert_eq!(selected.window_start_sample, 499.5);
    assert!((selected.weighted_seconds - 0.5).abs() < 1e-15);
}

#[test]
fn competing_endpoints_seal_disjoint_ancestral_mass_and_fallback_keeps_original_age() {
    let (mut phrase, base) = base();
    let mut cues = Stream::new(1, 0, 1000, 100, 0, [1.; 10]).unwrap();
    let handle = phrase.groups[0].unwrap().group;
    for step in 1..=10 {
        let end = step * 100;
        let mut a = fixture::input(step, 0.);
        a.assignment.rows[0].as_mut().unwrap().weights[0] = 0.5;
        phrase.end_sample = end;
        let g = phrase.groups[0].as_mut().unwrap();
        g.candidates.fill(None);
        let f = |heard_end| phrase::Foreground {
            id: 99,
            credit: 99,
            start: 0,
            heard_end,
        };
        if step == 1 {
            g.candidates[0] = Some(phrase::Candidate {
                parent_index: 0,
                mass: 1.,
                foreground: Some(f(end)),
                completed_foreground: None,
                ..base
            });
        } else {
            g.candidates[0] = Some(phrase::Candidate {
                parent_index: 0,
                mass: 0.25,
                foreground: None,
                completed_foreground: (step == 2).then_some(f(100)),
                ..base
            });
            g.candidates[1] = Some(phrase::Candidate {
                parent_index: if step == 2 { 0 } else { 1 },
                mass: 0.5,
                foreground: (step == 2).then_some(f(end)),
                completed_foreground: (step == 3).then_some(f(200)),
                ..base
            });
        }
        cues.advance(&a, &phrase).unwrap();
        if end < 600 {
            assert_eq!(cues.commitments.total, 0);
        }
    }
    assert_eq!(cues.commitments.total, 2);
    let records = &cues.commitments.sealed;
    assert_ne!(
        records[0].evidence.occurrence_id,
        records[1].evidence.occurrence_id
    );
    assert_eq!(
        records[0].evidence.ongoing_credit,
        records[1].evidence.ongoing_credit
    );
    assert!((records[0].evidence.support - 0.125).abs() < 1e-12);
    assert!((records[1].evidence.support - 0.25).abs() < 1e-12);
    assert_eq!(records[0].evidence.end, 100);
    assert_eq!(records[0].evidence.sealed_at, 600);
    let (frozen, selected) = cues.prefix(handle, 1000).unwrap().unwrap();
    assert_eq!(selected.support_end_sample, 200);
    assert_eq!(selected.selected_at, 1000);
    assert_eq!(frozen.supporting_audio_end, Some(200));
    assert_eq!(frozen.end, 0.2);
    assert_eq!(frozen.captured_at, 1000);
}

#[test]
fn commitment_crossing_deadline_cannot_use_later_branch_evidence_or_fill_a_gap() {
    let (mut phrase, base) = base();
    let mut cues = Stream::new(1, 0, 1000, 100, 0, [1.; 10]).unwrap();
    for step in [1, 2, 8] {
        let end = step * 100;
        let mut a = fixture::input(step, 0.);
        a.assignment.rows[0].as_mut().unwrap().weights[0] = 1.;
        phrase.end_sample = end;
        let g = phrase.groups[0].as_mut().unwrap();
        g.candidates.fill(None);
        let f = phrase::Foreground {
            id: 77,
            credit: 77,
            start: 0,
            heard_end: 100,
        };
        g.candidates[0] = Some(phrase::Candidate {
            parent_index: 0,
            mass: if step == 8 { 1. } else { 0.2 },
            foreground: (step == 1).then_some(f),
            completed_foreground: (step == 2).then_some(f),
            ..base
        });
        cues.advance(&a, &phrase).unwrap();
    }
    let p = &cues.commitments.sealed[0];
    assert!((p.evidence.support - 0.2).abs() < 1e-12);
    assert_eq!(p.evidence.deadline, 600);
    assert_eq!(p.evidence.sealed_at, 800);
    assert!(p.evidence.lag_missing_seconds >= 0.4);
    assert_eq!(p.descriptor.end, 0.1);
}

#[test]
fn thirty_second_phrase_enters_memory_once_with_its_whole_bounded_descriptor() {
    let (mut phrase, base) = base();
    let cfg = crate::config::TemporalMemoryConfig {
        retention: None,
        candidates: None,
        scales: [1.; 10],
        span_hops: 8,
        episodes: 16,
        query_cadence_ms: 100,
        deadline_ms: 200,
    };
    let mut cues = Stream::new(1, 0, 1000, 100, 0, [1.; 10]).unwrap();
    let mut memory = recall::Recall::new(1, 0, 1000, 100, cfg).unwrap();
    for step in 1..=320 {
        let end = step * 100;
        let a = fixture::input(step, 0.);
        phrase.end_sample = end;
        let g = phrase.groups[0].as_mut().unwrap();
        g.candidates.fill(None);
        g.candidates[0] = Some(phrase::Candidate {
            parent_index: 0,
            mass: 1.,
            foreground: Some(phrase::Foreground {
                id: if step <= 300 { 90 } else { 91 },
                credit: if step <= 300 { 90 } else { 91 },
                start: if step <= 300 { 0 } else { 30000 },
                heard_end: end,
            }),
            completed_foreground: (step == 301).then_some(phrase::Foreground {
                id: 90,
                credit: 90,
                start: 0,
                heard_end: 30000,
            }),
            ..base
        });
        cues.advance(&a, &phrase).unwrap();
        memory.advance(&a, end, Some(&cues)).unwrap();
        let group = phrase.groups[0].unwrap().group;
        assert_eq!(cues.retains_prefix(group, 90, 0), step < 305);
        assert!(!cues.retains_prefix(group, 90, 1));
        assert!(!cues.retains_prefix(
            Handle {
                generation: group.generation + 1,
                ..group
            },
            90,
            0
        ));
        if step < 305 {
            assert_eq!(memory.snapshot().stored_total, 0);
        }
    }
    let (e, frozen) = cues.committed().next().unwrap();
    assert_eq!((frozen.start, frozen.end), (0., 30.));
    assert_eq!(frozen.blocks.len(), 128);
    assert_eq!(e.end, 30000);
    assert_eq!(e.sealed_at, 30500);
    assert_eq!(memory.snapshot().stored_total, 1);
    assert_eq!(memory.snapshot().phrase_episodes, 1);
    let query = memory.snapshot().latest.unwrap();
    assert!(query.best.is_some());
    assert_eq!(query.best.unwrap().support_end_sample, 30000);
    assert_eq!(query.best.unwrap().available_at, 30500);
}

#[test]
fn odd_rate_commit_does_not_round_post_deadline_evidence_into_the_lag() {
    let (mut phrase, base) = base();
    let mut cues = Stream::new(1, 0, 1001, 100, 0, [1.; 10]).unwrap();
    for end in [100, 200, 300, 400, 500, 600, 601] {
        let mut a = fixture::input(end / 100, 0.);
        if end == 601 {
            a.features[0] = None;
            a.eligible[0] = false;
        }
        phrase.end_sample = end;
        let g = phrase.groups[0].as_mut().unwrap();
        g.candidates.fill(None);
        let f = phrase::Foreground {
            id: 55,
            credit: 55,
            start: 0,
            heard_end: 100,
        };
        g.candidates[0] = Some(phrase::Candidate {
            parent_index: 0,
            mass: if end == 601 { 1. } else { 0.2 },
            foreground: (end == 100).then_some(f),
            completed_foreground: (end == 200).then_some(f),
            ..base
        });
        cues.advance(&a, &phrase).unwrap();
        if end <= 600 {
            assert_eq!(cues.commitments.total, 0);
        }
    }
    let e = cues.commitments.sealed[0].evidence;
    assert_eq!(e.deadline, 601);
    assert!((e.retained_mass - 0.2).abs() < 1e-12);
    assert!((e.lag_missing_seconds - (0.5 - 500. / 1001.)).abs() < 1e-12);
}
