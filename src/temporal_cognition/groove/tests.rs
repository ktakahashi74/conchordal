use super::*;
use crate::temporal_cognition::{features::Accent, ridge::Handle};

fn owner() -> Handle {
    Handle {
        bus: 0,
        epoch: 3,
        generation: 2,
    }
}

fn estimator(rate: u32, capacity: usize) -> Estimator {
    Estimator::new(
        owner(),
        capacity,
        rate,
        u64::from(rate) * 32,
        1024,
        1. / 24.,
    )
    .unwrap()
}

fn accent(end: u64) -> Accent {
    Accent {
        group: owner(),
        event_start: end - 1,
        event_end: end,
        raw_intervals: [
            (end - 3, end - 2),
            (end - 2, end - 1),
            (end - 1, end),
            (end, end + 1),
        ],
        source_start: end - 3,
        source_end: end + 1,
        available_end: end + 2,
        weight: 0.5,
        observed_prefix: end,
    }
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-11,
        "actual={actual}, expected={expected}"
    );
}

fn frame(start: u64, end: u64, alpha: f64, symbol: usize, grouping: f64) -> Frame {
    let mut word = [0.; 9];
    word[symbol] = 1.;
    Frame {
        start,
        end,
        known: true,
        alpha,
        word: Some(word),
        grouping: Some(grouping),
    }
}

#[test]
fn projected_density_matches_independent_window_counts_and_keeps_observations_immutable() {
    use crate::temporal_cognition::feature_projection::Feature;
    let mut estimator = estimator(1000, 256);
    let ends: Vec<_> = (100..16000).step_by(100).collect();
    for &end in &ends {
        estimator.deliver(accent(end), end + 2).unwrap();
    }
    estimator.advance(16000).unwrap();
    let mut group = Group::new(1000, 100, 0);
    group.last_end = 16000;
    group.frames.push_back(frame(0, 16000, 1., 0, 0.));
    let before = group.summarize(&estimator, group.last_end);
    for offset in [0, 100, 500, 1000, 4000] {
        let end = 16000 + offset;
        let mut projection = group.project_density(&estimator, end).unwrap();
        let receipts: Vec<_> = (16000..end)
            .step_by(100)
            .filter(|t| *t != 16200)
            .map(|t| ([t, t + 100], (t % 300 == 0).then_some(0.75)))
            .collect();
        for &(interval, weight) in &receipts {
            projection.receipt(interval, weight);
        }
        projection.finish(1000);
        for (i, seconds) in WINDOWS.into_iter().enumerate() {
            let start = end.saturating_sub((seconds * 1000.) as u64);
            let observed = 16000_u64.saturating_sub(start);
            let projected: u64 = receipts
                .iter()
                .map(|(span, _)| span[1].saturating_sub(span[0].max(start)))
                .sum();
            let old_weight = ends.iter().filter(|&&t| t >= start).count() as f64 * 0.5;
            let new_weight: f64 = receipts
                .iter()
                .filter(|(span, _)| span[1] >= start)
                .map(|(_, w)| w.unwrap_or(0.))
                .sum();
            assert_eq!(projection.observed_samples[i], observed);
            assert_eq!(projection.projected_samples[i], projected);
            close(projection.observed_weight[i], old_weight);
            close(projection.projected_weight[i], new_weight);
            let supported = (observed + projected) as f64 >= 0.9 * (end - start) as f64;
            if supported {
                close(
                    projection.values[i].value().unwrap(),
                    (old_weight + new_weight) * 1000. / (observed + projected) as f64,
                );
                if offset == 0 {
                    assert_eq!(
                        projection.values[i],
                        Feature::Observed(before.raw[42 + i].unwrap())
                    );
                } else if projected > 0 {
                    assert!(matches!(projection.values[i], Feature::Projected(_)));
                }
            } else {
                assert_eq!(projection.values[i], Feature::Unsupported);
            }
        }
    }
    assert!(group.project_density(&estimator, 15999).is_none());
    assert_eq!(
        serde_json::to_value(before).unwrap(),
        serde_json::to_value(group.summarize(&estimator, group.last_end)).unwrap()
    );
}

#[test]
fn projected_density_preserves_cap_masks_epoch_clipping_and_missing_future() {
    use crate::temporal_cognition::feature_projection::Feature;
    let mut estimator = estimator(1000, 128);
    for end in (9400..=9912).step_by(4) {
        estimator.deliver(accent(end), end + 2).unwrap();
    }
    estimator.advance(10000).unwrap();
    let mut group = Group::new(1000, 100, 9000);
    group.last_end = 10000;
    group.frames.push_back(frame(9000, 10000, 1., 0, 0.));
    let mut projection = group.project_density(&estimator, 10500).unwrap();
    projection.finish(1000);
    assert_eq!(projection.starts[7], 9000);
    assert!(projection.values.iter().all(|v| *v == Feature::Unsupported));
    for start in (10000..10500).step_by(100) {
        projection.receipt([start, start + 100], None);
    }
    projection.finish(1000);
    assert_eq!(projection.values[0], Feature::Projected(0.));
    assert_eq!(projection.values[1], Feature::Projected(0.));
    for i in 4..8 {
        assert!(!projection.retention_supported[i]);
        assert_eq!(projection.values[i], Feature::Unsupported);
    }
}

#[test]
fn all_density_windows_mask_128_cap_loss_and_recover_only_after_eviction_exits() {
    let rate = 48000;
    let cut = 16 * u64::from(rate);
    let mut capped = estimator(rate, 128);
    let mut reference = estimator(rate, 256);
    let ends: Vec<_> = (0..129).map(|i| cut - 520 + i * 4).collect();
    for &end in &ends {
        capped.deliver(accent(end), end + 2).unwrap();
        reference.deliver(accent(end), end + 2).unwrap();
    }
    capped.advance(cut).unwrap();
    reference.advance(cut).unwrap();
    let mut group = Group::new(rate, 512, 0);
    group.last_end = cut;
    group.frames.push_back(frame(0, cut, 1., 0, 0.));
    let full = group.summarize(&reference, group.last_end);
    let lost = group.summarize(&capped, group.last_end);
    for (i, seconds) in WINDOWS.into_iter().enumerate() {
        close(full.raw[42 + i].unwrap(), 64.5 / seconds);
        assert_eq!(lost.raw[42 + i], None);
        assert!(!lost.density_retention_supported[i]);
        assert_eq!(lost.density_coverage[i], 1.);
        let end = ends[0] + (seconds * f64::from(rate)) as u64 + 1;
        capped.advance(end.max(cut)).unwrap();
        group.last_end = end.max(cut);
        group.frames.clear();
        group.frames.push_back(frame(0, group.last_end, 1., 0, 0.));
        let s = group.summarize(&capped, group.last_end);
        let count = ends
            .iter()
            .filter(|&&t| t >= end - (seconds * f64::from(rate)) as u64)
            .count();
        close(s.raw[42 + i].unwrap(), count as f64 * 0.5 / seconds);
    }
    // A delivery clock far ahead of physical audio can expire the requested history.
    capped
        .advance(group.last_end + 33 * u64::from(rate))
        .unwrap();
    assert!(
        group.summarize(&capped, group.last_end).raw[42..50]
            .iter()
            .all(Option::is_none)
    );
}

#[test]
fn physical_coverage_and_assignment_weights_control_entropy_grouping_and_known_zero() {
    let mut e = estimator(1000, 128);
    e.advance(8000).unwrap();
    let mut group = Group::new(1000, 100, 0);
    group.last_end = 8000;
    group.frames.extend([
        frame(0, 4000, 0.25, 0, 0.2),
        frame(4000, 8000, 0.75, 1, 0.8),
    ]);
    let s = group.summarize(&e, group.last_end);
    let h = -(0.25f64 * 0.25f64.ln() + 0.75f64 * 0.75f64.ln()) / 9f64.ln();
    close(s.raw[50].unwrap(), h);
    close(s.raw[51].unwrap(), h * h);
    close(s.raw[53].unwrap(), 0.65);
    assert_eq!(s.raw[52], None);
    assert_eq!(s.raw[42..50], [Some(0.); 8]);
    close(s.assignment_sample_weight, 4000.);
    group.frames.clear();
    group
        .frames
        .extend([frame(0, 7000, 1., 0, 0.), frame(7000, 8000, 1., 1, 1.)]);
    group.frames[1].word = None;
    let s = group.summarize(&e, group.last_end);
    close(s.word_coverage, 0.875);
    assert_eq!(s.raw[50..53], [None; 3]);
    assert_eq!(s.raw[53], Some(0.125));
    // Exactly 90 percent is supported; densities divide by observed seconds.
    group.frames[0].end = 7200;
    group.frames[1].start = 7200;
    group.frames[1].known = false;
    e.deliver(accent(4000), 8000).unwrap();
    let s = group.summarize(&e, group.last_end);
    assert_eq!(s.raw[50], Some(-0.));
    assert_eq!(s.raw[53], Some(0.));
    close(s.raw[49].unwrap(), 0.5 / 7.2);
    assert_eq!(s.raw[42], None);
    for f in &mut group.frames {
        f.alpha = 0.;
    }
    let s = group.summarize(&e, group.last_end);
    assert_eq!(s.raw[50..54], [None; 4]);
}

#[test]
fn shifted_context_reuses_observed_formulas_and_never_learns_future_words() {
    use crate::temporal_cognition::feature_projection::Feature;
    let mut estimator = estimator(1000, 128);
    estimator.advance(16000).unwrap();
    let mut group = Group::new(1000, 100, 0);
    group.last_end = 16000;
    group.frames.extend([
        frame(8000, 12000, 0.25, 0, 0.2),
        frame(12000, 16000, 0.75, 1, 0.8),
    ]);
    for (end, value) in [(8300, 1.), (9000, 3.), (15500, 5.)] {
        group.surprise.push_back(Surprise { end, value });
    }
    group.counts[2][4] = 3.;
    let before = serde_json::to_value(group.summarize(&estimator, 16000)).unwrap();
    for offset in [0, 600, 800, 801, 4000] {
        let context = group.project_context(&estimator, 16000 + offset).unwrap();
        close(context.word_coverage, (8000 - offset) as f64 / 8000.);
        close(context.grouping_coverage, context.word_coverage);
        if offset <= 800 {
            let a = (4000 - offset) as f64 * 0.25;
            let b = 4000. * 0.75;
            let p = a / (a + b);
            let h = -(p * p.ln() + (1. - p) * (1. - p).ln()) / 9f64.ln();
            let expected = [
                h,
                h * h,
                if offset == 0 { 3. } else { 4. },
                (0.2 * a + 0.8 * b) / (a + b),
            ];
            for (actual, expected) in context.values.into_iter().zip(expected) {
                assert!(matches!(actual, Feature::Observed(_)));
                close(actual.value().unwrap(), expected);
            }
            close(context.word_probabilities.unwrap()[0], p);
        } else {
            assert_eq!(context.values, [Feature::Unsupported; 4]);
            assert!(context.word_probabilities.is_none());
        }
        assert_eq!(
            context.retained_pairs,
            if offset == 0 {
                3
            } else if offset == 4000 {
                1
            } else {
                2
            }
        );
    }
    assert_eq!(group.counts[2][4], 3.);
    assert_eq!(group.surprise.len(), 3);
    assert_eq!(
        before,
        serde_json::to_value(group.summarize(&estimator, 16000)).unwrap()
    );
    let issue = group.project_context(&estimator, 16000).unwrap();
    assert_eq!(
        issue.values.map(Feature::value),
        group.summarize(&estimator, 16000).raw[50..54]
    );
}

#[test]
fn context_projection_keeps_cap_loss_missing_pairs_and_epoch_clipping_explicit() {
    use crate::temporal_cognition::feature_projection::Feature;
    let mut estimator = estimator(1000, 128);
    estimator.advance(16000).unwrap();
    let mut group = Group::new(1000, 100, 0);
    group.last_end = 16000;
    group.frames.push_back(frame(8000, 16000, 1., 0, 0.));
    group.surprise.push_back(Surprise {
        end: 9000,
        value: 3.,
    });
    group.lost_through = Some(8500);
    let lost = group.project_context(&estimator, 16500).unwrap();
    assert_eq!(lost.values[0], Feature::Observed(-0.));
    assert_eq!(lost.values[3], Feature::Observed(0.));
    assert_eq!(lost.values[2], Feature::Unsupported);
    assert!(!lost.surprise_capacity_supported);
    let recovered = group.project_context(&estimator, 16501).unwrap();
    assert_eq!(recovered.values[2], Feature::Observed(3.));
    assert!(recovered.surprise_capacity_supported);
    group.surprise.clear();
    assert_eq!(
        group.project_context(&estimator, 16501).unwrap().values[2],
        Feature::Unsupported
    );
    group = Group::new(1000, 100, 100000);
    group.last_end = 101000;
    group.frames.push_back(frame(100000, 101000, 1., 0, 0.));
    let at_issue = group.project_context(&estimator, 101000).unwrap();
    assert_eq!(at_issue.window_start, 100000);
    assert_eq!(at_issue.word_coverage, 1.);
    assert_eq!(at_issue.values[3], Feature::Observed(0.));
    assert_eq!(
        group.project_context(&estimator, 101112).unwrap().values,
        [Feature::Unsupported; 4]
    );
    assert!(group.project_context(&estimator, 100999).is_none());
    assert!(group.project_context(&estimator, 105001).is_none());
    group.reset();
    assert_eq!(
        group.project_context(&estimator, 100000).unwrap().values,
        [Feature::Unsupported; 4]
    );
}

fn word(end: u64, symbols: [u8; 2], support: f64) -> groupings::Word {
    let mut result = groupings::Word {
        len: 2,
        symbols: [0; 8],
        anchors: [(0, 0); 17],
        support,
        observed_pairs: 0b111,
    };
    result.symbols[..2].copy_from_slice(&symbols);
    for (i, a) in result.anchors[..5].iter_mut().enumerate() {
        let t = end - 400 + i as u64 * 100;
        *a = (t - 1, t);
    }
    result
}

#[test]
fn ambiguous_original_pairs_score_before_fractional_learning_and_never_relearn() {
    let mut group = Group::new(1000, 100, 0);
    group.processed_through = Some(400);
    let mut words = [None; 16];
    words[0] = Some(word(500, [2, 4], 0.25));
    words[1] = Some(word(500, [4, 2], 0.75));
    group.observe_words(&words, Some(500), 0);
    close(group.counts[2][4], 0.25);
    close(group.counts[4][2], 0.75);
    close(group.counts.iter().flatten().sum(), 1.);
    close(group.surprise[0].value, 9f64.ln());
    words[0] = Some(word(600, [2, 4], 0.25));
    words[1] = Some(word(600, [4, 2], 0.75));
    group.observe_words(&words, Some(600), 0);
    close(
        group.surprise[1].value,
        -0.25 * (0.75f64 / 4.75).ln() - 0.75 * (1.25f64 / 5.25).ln(),
    );
    let before = group.counts;
    group.observe_words(&words, Some(600), 0);
    assert_eq!(group.counts, before);
    assert_eq!(group.surprise.len(), 2);
    group.observe_words(&[None; 16], Some(700), 0);
    words[0] = Some(word(700, [2, 4], 1.));
    words[1] = None;
    group.observe_words(&words, Some(700), 0);
    assert_eq!(group.counts, before);
    // A gap in either original interval blocks the pair even if the word was admitted.
    let mut missing = word(800, [2, 4], 1.);
    missing.observed_pairs = 0;
    words[0] = Some(missing);
    group.observe_words(&words, Some(800), 0);
    assert_eq!(group.counts, before);
    group.reset();
    assert_eq!(group.counts, [[0.; 9]; 9]);
    assert!(group.surprise.is_empty());
    assert!(group.word.is_none());
}

#[test]
fn multiple_new_pairs_are_scored_in_original_order_and_surprise_capacity_is_explicit() {
    let mut group = Group::new(4000, 10, 0);
    group.processed_through = Some(200);
    let mut words = [None; 16];
    words[0] = Some(word(500, [2, 2], 1.));
    group.observe_words(&words, Some(500), 0);
    assert_eq!(group.surprise.len(), 3);
    for (i, s) in group.surprise.iter().enumerate() {
        close(s.value, -((i as f64 + 0.5) / (i as f64 + 4.5)).ln());
    }
    for end in (600..=14000).step_by(100) {
        words[0] = Some(word(end, [2, 2], 1.));
        group.observe_words(&words, Some(end), 0);
    }
    assert_eq!(group.surprise.len(), 128);
    assert!(group.capacity_evicted > 0);
    let mut e = estimator(4000, 128);
    e.advance(14000).unwrap();
    group.last_end = 14000;
    group.frames.push_back(frame(0, 14000, 1., 2, 1.));
    let s = group.summarize(&e, group.last_end);
    assert!(s.raw[50].is_some());
    assert!(!s.surprise_capacity_supported);
    assert_eq!(s.raw[52], None);
}

#[test]
fn epoch_clipping_and_retirement_keep_physical_support_and_storage_bounded() {
    let mut e = estimator(1000, 128);
    let mut group = Group::new(1000, 100, 100000);
    e.reset(owner(), 100000).unwrap();
    let mut bytes = None;
    for end in (100100..=120000).step_by(100) {
        e.advance(end).unwrap();
        let s = group.advance(
            &e,
            None,
            Input {
                start: end - 100,
                end,
                known: true,
                alpha: 1.,
                refreshed: true,
            },
        );
        assert_eq!(s.raw[42..50], [Some(0.); 8]);
        assert_eq!(s.word_coverage, 0.);
        assert_eq!(*bytes.get_or_insert(s.owned_bytes), s.owned_bytes);
        assert!(group.frames.len() <= group.frame_capacity);
    }
    group.reset();
    e.reset(
        Handle {
            generation: 3,
            ..owner()
        },
        120000,
    )
    .unwrap();
    e.advance(120100).unwrap();
    let s = group.advance(
        &e,
        None,
        Input {
            start: 120000,
            end: 120100,
            known: true,
            alpha: 1.,
            refreshed: true,
        },
    );
    assert_eq!(s.raw[42..50], [None; 8]);
    assert_eq!(s.learned_pair_mass, 0.);
    assert_eq!(s.owned_bytes, bytes.unwrap());
}
