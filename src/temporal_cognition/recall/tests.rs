use super::*;
use crate::temporal_cognition::{features, group};

pub(in crate::temporal_cognition) fn config() -> TemporalMemoryConfig {
    TemporalMemoryConfig {
        retention: None,
        candidates: None,
        scales: [1.; 10],
        span_hops: 16,
        episodes: 16,
        query_cadence_ms: 100,
        deadline_ms: 200,
    }
}

#[test]
fn retained_live_queries_publish_causal_score_bounds_and_source_strength() {
    let mut c = config();
    c.span_hops = 8;
    c.episodes = 32;
    c.candidates = Some(32);
    c.retention = Some(crate::config::TemporalRetentionConfig {
        tau_sec: 20.,
        kappa: 4.,
        strength_max: 2.,
        r_max: 4.,
        no_memory_bias: 0.,
    });
    let model = crate::temporal_cognition::section::tests::runtime_config();
    assert!(Recall::new(1, 0, 1000, 100, c, None).is_err());
    for change in 0..5 {
        let mut bad = c;
        let p = bad.retention.as_mut().unwrap();
        match change {
            0 => p.tau_sec = 0.,
            1 => p.kappa = f64::NAN,
            2 => p.strength_max = 0.5,
            3 => p.r_max = f64::INFINITY,
            _ => p.no_memory_bias = f64::NAN,
        }
        assert!(Recall::new(1, 0, 1000, 100, bad, Some(model)).is_err());
    }
    let mut recall = Recall::new(1, 0, 1000, 100, c, Some(model)).unwrap();
    let mut scored = 0;
    let mut truncated = 0;
    for step in 1..=192 {
        recall
            .observe_acquisition((step - 1) * 100, step * 100, step * 100)
            .unwrap();
        recall
            .advance(
                &input(step, ((step - 1) % 8) as f64 * 0.03),
                step * 100,
                None,
            )
            .unwrap();
        let snapshot = recall.snapshot();
        let retained = snapshot.retention.unwrap();
        assert_eq!(retained.records, snapshot.stored_episodes);
        assert_eq!(retained.sequence, snapshot.stored_total);
        if let Some(record) = retained.latest_record {
            assert!((record.strength - 1.).abs() < 1e-12);
            assert_eq!(record.strength, record.membership_total);
            assert!(record.last_observed_end <= record.last_delivered_at);
        }
        for group in snapshot.retrieval.iter().flatten() {
            assert_eq!(group.group.bus, 1);
            assert!(group.support_end_sample <= group.available_at_sample);
            assert!(group.available_at_sample <= group.evaluated_at_sample);
            let mut lower = 0.;
            let mut upper = 0.;
            for entry in group.entries.iter().flatten() {
                scored += 1;
                assert_eq!(entry.acoustic_score, 0.);
                assert!(entry.score[0] <= entry.score[1]);
                assert!(
                    0. <= entry.weight[0]
                        && entry.weight[0] <= entry.weight[1]
                        && entry.weight[1] <= 1.
                );
                lower += entry.weight[0];
                upper += entry.weight[1];
                assert_eq!(entry.availability.handle, entry.episode);
                assert!(entry.availability.elapsed_sec >= 0.);
            }
            assert!(lower + group.discarded_weight[0] <= group.recognition.lower + 1e-12);
            assert!(upper + group.discarded_weight[1] + 1e-12 >= group.recognition.upper);
            if group.scored_episodes > 16 {
                truncated += 1;
                assert_eq!(group.entries.iter().flatten().count(), 16);
                assert!(group.discarded_weight[0] > 0.);
                assert!(group.discarded_weight[0] <= group.discarded_weight[1]);
            }
        }
    }
    assert!(scored > 0);
    assert!(truncated > 0);
    let before: Vec<_> = recall
        .retention
        .as_ref()
        .unwrap()
        .records
        .iter()
        .map(|r| r.membership_total)
        .collect();
    recall.refresh_retrieval(19200).unwrap();
    assert_eq!(
        before,
        recall
            .retention
            .as_ref()
            .unwrap()
            .records
            .iter()
            .map(|r| r.membership_total)
            .collect::<Vec<_>>()
    );
}

#[test]
fn live_retention_evicts_weak_support_and_invalidates_its_descriptor() {
    let mut c = config();
    c.episodes = 2;
    c.span_hops = 8;
    c.retention = Some(crate::config::TemporalRetentionConfig {
        tau_sec: 20.,
        kappa: 4.,
        strength_max: 2.,
        r_max: 4.,
        no_memory_bias: 0.,
    });
    let mut recall = Recall::new(
        1,
        0,
        1000,
        100,
        c,
        Some(crate::temporal_cognition::section::tests::runtime_config()),
    )
    .unwrap();
    for step in 1..=24 {
        recall
            .observe_acquisition((step - 1) * 100, step * 100, step * 100)
            .unwrap();
        let mut frame = input(step, ((step - 1) % 8) as f64 * 0.03);
        if (10..=16).contains(&step) {
            let raw = &mut frame.features[0].as_mut().unwrap().raw;
            raw.known_samples = 0;
            raw.values = [None; 10];
        }
        recall.advance(&frame, step * 100, None).unwrap();
        if step == 16 {
            let record = recall.snapshot().retention.unwrap().latest_record.unwrap();
            assert_eq!(record.handle, 2);
            assert!((record.strength - 0.125).abs() < 1e-12);
        }
    }
    assert_eq!(
        recall
            .episodes
            .iter()
            .map(|e| e.identity.id)
            .collect::<Vec<_>>(),
        [1, 3]
    );
    assert_eq!(recall.snapshot().retention.unwrap().evictions, 1);
    assert!(
        recall
            .retention
            .as_ref()
            .unwrap()
            .records
            .iter()
            .all(|r| r.handle != 2)
    );
    assert!(
        recall
            .groups
            .iter()
            .flatten()
            .flat_map(|g| g.matches.iter().flatten().flatten())
            .all(|m| m.episode_id != 2)
    );
}

pub(in crate::temporal_cognition) fn input(step: u64, value: f64) -> frontend::Snapshot {
    let handle = Handle {
        bus: 1,
        epoch: 0,
        generation: 2,
    };
    let mut out = frontend::Snapshot {
        assignment: group::Assignment {
            end_sample: step * 100,
            group_handles: [None; 7],
            rows: [None; 8],
            distance_evaluations: 0,
        },
        retained_groups: [None; 7],
        correlation_window: [0, step * 100],
        group_handles: [None; 8],
        eligible: [false; 8],
        energy: None,
        spectral_shape_supported: true,
        features: [None; 8],
        feature_gaps: [None; 8],
        admissions: 0,
        admissions_by_kind: [0; 3],
        rejections: 0,
        retired: [None; 7],
        superseded: 0,
        pending_proposals: 0,
        proposal_conflicts: 0,
        candidate_count: 0,
        continued_candidates: 0,
        bundle_count: 0,
        correlation_reads: 0,
        cross_pair_reads: 0,
    };
    out.retained_groups[0] = Some(handle);
    out.group_handles[0] = Some(handle);
    let mut values = [None; 10];
    values[0] = Some(8.);
    values[1] = Some(value);
    out.features[0] = Some(features::Update {
        raw: features::RawDescriptor {
            group: handle,
            start: (step - 1) * 100,
            end: step * 100,
            known_samples: 100,
            values,
            source_start: (step * 100).saturating_sub(400),
            source_end: step * 100,
            available_end: step * 100,
        },
        detector: None,
    });
    out
}

#[test]
fn configured_search_capacity_reaches_the_received_cache_without_truncation() {
    let mut c = config();
    c.episodes = 64;
    c.candidates = Some(32);
    c.span_hops = 8;
    let mut recall = Recall::new(1, 0, 1000, 100, c, None).unwrap();
    let mut peak = 0;
    for step in 1..=320 {
        let value = ((step - 1) / 8) as f64 * 0.001 + ((step - 1) % 8) as f64 * 0.03;
        recall
            .advance(&input(step, value), step * 100, None)
            .unwrap();
        if let Some((q, rows)) = recall.latest_matches() {
            assert_eq!(rows.len(), 32);
            let received = rows.iter().filter(|r| r[0].is_some()).count();
            peak = peak.max(received);
            assert_eq!(q.group.generation, 2);
        }
    }
    assert_eq!(peak, 32);
    for invalid in [0, memory::MAX_CANDIDATES + 1] {
        c.candidates = Some(invalid);
        assert!(Recall::new(1, 0, 1000, 100, c, None).is_err());
    }
}

#[test]
fn first_and_later_queries_preserve_support_and_detect_order_changes() {
    let mut costs = Vec::new();
    for reversed in [false, true] {
        let mut recall = Recall::new(1, 0, 1000, 100, config(), None).unwrap();
        for step in 1..=32 {
            let index = (step - 1) % 16;
            let index = if reversed && step > 16 {
                15 - index
            } else {
                index
            };
            recall
                .advance(&input(step, (index / 2) as f64 * 0.5), step * 100, None)
                .unwrap();
            if step == 16 {
                recall.finish(step * 100).unwrap();
                let first = recall.snapshot().latest.unwrap();
                assert_eq!(first.candidates, 0);
                assert!(first.best.is_none());
            }
        }
        recall.finish(3200).unwrap();
        let result = recall.snapshot().latest.unwrap();
        let best = result.best.unwrap();
        assert_eq!(result.support_start_sample, 1600);
        assert_eq!(result.support_end_sample, 3200);
        assert_eq!(result.source_start_sample, 1300);
        assert_eq!(result.supporting_audio_end, Some(3200));
        assert_eq!(best.support_start_sample, 0);
        assert_eq!(best.support_end_sample, 1600);
        assert_eq!(best.available_at, 1600);
        assert_eq!(best.episode_id, 1);
        assert!(result.dp_cells > 0);
        costs.push(best.cost);
    }
    assert!(costs[0] < 1e-12, "repeat cost: {costs:?}");
    assert!(costs[1] > costs[0] + 0.01, "order was ignored: {costs:?}");
}

#[test]
fn late_and_retired_queries_are_rejected_and_next_queries_recover() {
    for retirement in [false, true] {
        let mut recall = Recall::new(1, 0, 1000, 100, config(), None).unwrap();
        recall.advance(&input(1, 1.), 100, None).unwrap();
        assert!(recall.pending.is_some());
        if retirement {
            let mut next = input(2, 1.);
            next.retained_groups = [None; 7];
            next.features = [None; 8];
            recall.advance(&next, 200, None).unwrap();
            assert_eq!(recall.snapshot().rejected_retired, 1);
        } else {
            recall.finish(400).unwrap();
            assert_eq!(recall.snapshot().rejected_late, 1);
        }
        assert!(recall.snapshot().latest.is_none());
        let mut next = input(5, 1.);
        if retirement {
            next.retained_groups[0].as_mut().unwrap().generation = 3;
            next.features[0].as_mut().unwrap().raw.group.generation = 3;
        }
        recall.advance(&next, 500, None).unwrap();
        recall.finish(500).unwrap();
        assert!(recall.snapshot().latest.is_some());
    }
}

#[test]
fn capacity_retirement_invalidates_pending_matches() {
    let mut cfg = config();
    cfg.episodes = 1;
    let mut recall = Recall::new(1, 0, 1000, 100, cfg, None).unwrap();
    for step in 1..=32 {
        recall.advance(&input(step, 1.), step * 100, None).unwrap();
    }
    // The full second query references episode 1, evicted by the second commit.
    recall.finish(3200).unwrap();
    assert_eq!(recall.snapshot().evicted, 1);
    assert_eq!(recall.snapshot().rejected_retired, 1);
    assert!(recall.snapshot().latest.is_none());
}

#[test]
fn missing_acquisition_stays_a_gap_and_epoch_ownership_is_enforced() {
    let mut recall = Recall::new(1, 0, 1000, 100, config(), None).unwrap();
    for step in [1, 2, 5, 6] {
        recall.advance(&input(step, 1.), step * 100, None).unwrap();
    }
    let frozen = recall.groups[0]
        .as_ref()
        .unwrap()
        .span
        .as_ref()
        .unwrap()
        .prefix(600)
        .unwrap();
    let descriptor = frozen.matching().unwrap();
    assert!(
        descriptor
            .knots
            .iter()
            .any(|k| k.gap == 1 && k.observed_sec == 0.)
    );
    let mut foreign = input(7, 1.);
    foreign.features[0].as_mut().unwrap().raw.group.epoch = 1;
    foreign.retained_groups[0].as_mut().unwrap().epoch = 1;
    assert!(recall.advance(&foreign, 700, None).is_err());
}

#[test]
fn all_received_transformations_stay_bound_to_the_original_query() {
    let mut r = Recall::new(1, 0, 1000, 100, config(), None).unwrap();
    let mut compared = 0;
    let mut two_episodes = false;
    let mut previous_query = 0;
    let mut previous_matches = String::new();
    for step in 1..=72 {
        r.advance(
            &input(step, ((step - 1) % 16) as f64 * 0.02),
            step * 100,
            None,
        )
        .unwrap();
        if let Some((query, matches)) = r.latest_matches() {
            assert_eq!(query.query_id, r.snapshot().latest.unwrap().query_id);
            assert!(query.issued_at <= query.received_at && query.received_at <= step * 100);
            assert_eq!(
                r.snapshot().retained_matches,
                matches.iter().flatten().flatten().count()
            );
            let json = serde_json::to_string(matches).unwrap();
            if query.query_id == previous_query {
                assert_eq!(json, previous_matches);
            }
            previous_query = query.query_id;
            previous_matches = json;
            let mut identities = Vec::new();
            for row in matches {
                if let Some(first) = row[0] {
                    let id = (first.episode_id, first.episode_generation);
                    assert!(!identities.contains(&id));
                    identities.push(id);
                    for m in row.iter().flatten() {
                        compared += 1;
                        assert_eq!((m.episode_id, m.episode_generation), id);
                        assert!(m.support_end_sample <= query.support_start_sample);
                        assert!(m.available_at <= query.issued_at);
                        let residuals = m.residuals.unwrap();
                        assert!(residuals.observed > 0 && residuals.matched > 0);
                        assert_eq!(
                            residuals.matched + residuals.missing + residuals.inserted,
                            residuals.observed
                        );
                    }
                }
            }
            two_episodes |= identities.len() >= 2;
            if let Some(best) = query.best {
                assert!(
                    matches
                        .iter()
                        .flatten()
                        .flatten()
                        .any(|m| m.episode_id == best.episode_id && m.cost == best.cost)
                );
            }
        }
    }
    assert!(compared > 0 && two_episodes);
    r.finish(100_000).unwrap();
    assert!(r.latest_matches().is_none());
    assert_eq!(r.snapshot().retained_matches, 0);
    assert!(std::mem::size_of::<ResultSnapshot>() < 2048);
}

#[test]
fn interleaved_groups_keep_independent_received_matches_and_deadlines() {
    let mut r = Recall::new(1, 0, 1000, 100, config(), None).unwrap();
    let first = input(1, 0.).group_handles[0].unwrap();
    let second = Handle {
        generation: first.generation + 10,
        ..first
    };
    for step in 1..=32 {
        let mut a = input(step, ((step - 1) % 8) as f64 * 0.02);
        a.retained_groups[1] = Some(second);
        a.group_handles[1] = Some(second);
        a.features[1] = a.features[0];
        a.features[1].as_mut().unwrap().raw.group = second;
        a.eligible[1] = true;
        r.advance(&a, step * 100, None).unwrap();
    }
    r.finish(3200).unwrap();
    let (a, ma) = r
        .matches_for(first)
        .expect("first group's still-valid receipt was displaced");
    let (b, mb) = r
        .matches_for(second)
        .expect("second group's still-valid receipt was displaced");
    assert_ne!(a.query_id, b.query_id);
    assert!(ma.iter().flatten().flatten().count() > 0 && mb.iter().flatten().flatten().count() > 0);
    assert!(a.deadline >= 3200 && b.deadline >= 3200);
    let newer = if a.deadline > b.deadline {
        first
    } else {
        second
    };
    let older = if newer == first { second } else { first };
    let cut = a.deadline.min(b.deadline) + 1;
    r.finish(cut).unwrap();
    assert!(r.matches_for(older).is_none());
    assert!(r.matches_for(newer).is_some());
    r.finish(a.deadline.max(b.deadline) + 1).unwrap();
    assert!(r.matches_for(first).is_none() && r.matches_for(second).is_none());
}
