use super::*;
use crate::temporal_cognition::{features, phrase};

#[test]
fn acoustic_windows_match_physical_moments_and_preserve_completed_endings() {
    let first = phrase::tests::input(1, 0.);
    let group = first.group_handles[0].unwrap();
    let mut cache = Cache::new(group, 0, 1000, 100).unwrap();
    let capacities = (cache.history.capacity(), cache.accents.capacity());
    let mut span = Interpretation::new(1, [0, 0]).unwrap();
    let mut samples = Vec::new();
    let mut checkpoints = 0;
    let mut previous_short = cache.short(0, 0.1).unwrap();
    for step in 1..=30 {
        let end = step * 100;
        let mut acoustic = phrase::tests::input(step, 0.2);
        let value = step as f64 / 10.;
        let u = acoustic.features[0].as_mut().unwrap();
        u.raw.values[0] = Some(value);
        u.raw.values[2] = Some(value.log2());
        u.raw.values[3] = Some(step as f64 / 100.);
        u.raw.values[5] = Some(step as f64 / 50.);
        acoustic.energy = Some([value * value; 8]);
        let observed = ![13, 14].contains(&step);
        acoustic.eligible[0] = observed;
        if observed && step % 3 == 0 {
            u.detector = Some(features::Detection {
                acquisition_coverage: 1.,
                detector_coverage: true,
                saliences: None,
                status: features::Status::Admitted,
                accent: Some(Accent {
                    group,
                    event_start: end - 50,
                    event_end: end - 20,
                    raw_intervals: [(end - 100, end); 4],
                    source_start: end - 100,
                    source_end: end,
                    available_end: end,
                    weight: 0.5,
                    observed_prefix: end,
                }),
            });
        }
        let previous_ending = span.ending;
        let o = cache.advance(&acoustic, [end - 100, end]).unwrap();
        assert_eq!(
            cache.short(end - 100, 0.1).unwrap().values,
            previous_short.values
        );
        assert!(cache.matches(&o));
        let mut wrong = o;
        wrong.raw.as_mut().unwrap().raw.values[0] = Some(-10.);
        assert!(!cache.matches(&wrong));
        span.completed_foreground = None;
        span.completed_ending = None;
        span.advance(
            (step == 20).then_some(phrase::Exit::New),
            [end - 100, end],
            observed,
            (step == 20).then_some(2),
        )
        .unwrap();
        cache.project(&mut span, observed).unwrap();
        if step == 20 {
            assert_eq!(span.completed_ending, previous_ending);
            assert_eq!(span.completed_ending.unwrap().end, 1900);
            assert!(span.ending.is_none());
        }
        if observed {
            samples.push((step, value));
        }
        let endpoint = span.foreground.unwrap().heard_end;
        if let Some(ending) = span.ending {
            let start = span
                .foreground
                .unwrap()
                .start
                .max(endpoint.saturating_sub(2000));
            let kept: Vec<_> = samples
                .iter()
                .copied()
                .filter(|(s, _)| *s * 100 > start && *s * 100 <= endpoint)
                .collect();
            let coverage = kept.len() as f64 * 100. / (endpoint - start) as f64;
            assert!((ending.coverage[0] - coverage).abs() < 1e-12);
            if coverage >= 0.9 {
                let energy: f64 = kept.iter().map(|(_, v)| v * v).sum();
                let center = kept.iter().map(|(_, v)| v * v * v).sum::<f64>() / energy;
                let spread = (kept
                    .iter()
                    .map(|(_, v)| v * v * (0.04 + (v - center).powi(2)))
                    .sum::<f64>()
                    / energy)
                    .sqrt();
                let density = kept.iter().filter(|(s, _)| s % 3 == 0).count() as f64 * 0.5 * 10.
                    / kept.len() as f64;
                let expected = [
                    center,
                    spread,
                    (energy / kept.len() as f64).sqrt().log2(),
                    kept.iter().map(|(s, _)| *s as f64 / 100.).sum::<f64>() / kept.len() as f64,
                    kept.iter().map(|(s, _)| *s as f64 / 50.).sum::<f64>() / kept.len() as f64,
                    density.ln_1p(),
                ];
                for (a, b) in ending.raw_values.into_iter().zip(expected) {
                    assert!((a.unwrap() - b).abs() < 1e-12);
                }
                checkpoints += 1;
            } else {
                assert!(ending.raw_values.iter().all(Option::is_none));
            }
        }
        if !observed {
            assert_eq!(span.ending, previous_ending);
        }
        let short = cache.short(end, 0.1).unwrap();
        previous_short = short;
        let start = end.saturating_sub(250);
        let mut total = 0.;
        let mut duration = 0;
        for (s, _) in &samples {
            let n = (s * 100)
                .min(end)
                .saturating_sub(((s - 1) * 100).max(start));
            total += *s as f64 / 100. * n as f64;
            duration += n;
        }
        let expected =
            (duration as f64 >= 0.9 * (end - start) as f64).then(|| total / duration as f64);
        assert_eq!(short.values[0].value().is_some(), expected.is_some());
        if let Some(value) = expected {
            assert!((short.values[0].value().unwrap() - value).abs() < 1e-12);
        }
        assert_eq!(
            (cache.history.capacity(), cache.accents.capacity()),
            capacities
        );
        assert!(cache.history.len() <= 21);
    }
    let before = cache.end;
    assert!(cache.advance(&first, [0, 100]).is_err());
    assert_eq!(cache.end, before);
    assert!(cache.short(before + 1, 0.1).is_err());
    for defect in 0..4 {
        let mut invalid = phrase::tests::input(31, 0.);
        match defect {
            0 => invalid.features[0].as_mut().unwrap().raw.source_end = 3000,
            1 => invalid.retained_groups[0] = None,
            2 => invalid.energy = Some([f64::NAN; 8]),
            _ => invalid.assignment.rows[0].as_mut().unwrap().weights[0] = -1.,
        }
        assert!(cache.advance(&invalid, [3000, 3100]).is_err());
        assert_eq!(cache.end, before);
    }
    eprintln!(
        "JOINT_ACOUSTIC_WINDOWS steps=30 moment_checkpoints={checkpoints} history_capacity={} accent_capacity={}",
        capacities.0, capacities.1
    );
}

#[test]
fn bounded_accent_loss_masks_density_without_growing_storage() {
    let group = phrase::tests::input(1, 0.).group_handles[0].unwrap();
    let mut cache = Cache::new(group, 0, 1000, 10).unwrap();
    let capacities = (cache.history.capacity(), cache.accents.capacity());
    for step in 1..=200 {
        let end = step * 10;
        let mut acoustic = phrase::tests::input(step, 0.);
        let u = acoustic.features[0].as_mut().unwrap();
        u.raw.start = end - 10;
        u.raw.end = end;
        u.raw.known_samples = 10;
        u.raw.source_start = 0;
        u.raw.source_end = end;
        u.raw.available_end = end;
        u.detector = Some(features::Detection {
            acquisition_coverage: 1.,
            detector_coverage: true,
            saliences: None,
            status: features::Status::Admitted,
            accent: Some(Accent {
                group,
                event_start: end - 1,
                event_end: end,
                raw_intervals: [(end - 10, end); 4],
                source_start: 0,
                source_end: end,
                available_end: end,
                weight: 1.,
                observed_prefix: end,
            }),
        });
        cache.advance(&acoustic, [end - 10, end]).unwrap();
    }
    let ending = cache
        .ending(Foreground {
            id: 1,
            credit: 1,
            start: 0,
            heard_end: 2000,
        })
        .unwrap();
    assert_eq!(ending.coverage[5], 1.);
    assert_eq!(ending.raw_values[5], None);
    assert_eq!(cache.accents.len(), 128);
    assert_eq!(cache.evicted_accent, Some(720));
    assert_eq!(
        (cache.history.capacity(), cache.accents.capacity()),
        capacities
    );
    eprintln!("JOINT_ACOUSTIC_CAPACITY samples=200 retained_accents=128 evicted_through=720");
}
