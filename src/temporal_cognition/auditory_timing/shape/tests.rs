use super::*;
use crate::temporal_cognition::ridge::Handle;

fn record(target: [f64; 2], anchor: [f64; 2], weight: f64, end: u64) -> Record {
    Record {
        end,
        target,
        anchor,
        weight,
        version: 1,
    }
}

fn summary(history: &History, periodic: bool) -> Summary {
    let weight: f64 = history.records.iter().map(|r| r.weight).sum();
    Summary {
        reference: Handle {
            bus: 0,
            epoch: 1,
            generation: 2,
        },
        family: if periodic {
            Family::Periodic
        } else {
            Family::MedianInterval
        },
        version: 1,
        period_seconds: periodic.then_some(0.5),
        reference_support: 1.,
        records: history.records.len(),
        retained_weight: weight,
        coverage: 1.,
        supported: weight > 0.,
        bins: std::array::from_fn(|i| {
            if weight > 0. {
                history.masses[i] / weight
            } else {
                0.
            }
        }),
        overflow: if weight > 0. {
            history.masses[BINS] / weight
        } else {
            0.
        },
        capacity_evicted: 0,
        discarded: 0,
        mode_bins: [None; 2],
        mode_count: None,
        residual_dispersion: None,
    }
}

#[test]
fn exact_plateaus_wrap_endpoints_uniform_and_small_contrasts_have_registered_modes() {
    for periodic in [false, true] {
        for mass in [0., 1. / 32.] {
            assert_eq!(Shape::modes(&[mass; BINS], periodic).count, 0);
        }
        let mut bins = [0.; BINS];
        bins[3..6].fill(0.1);
        bins[17..20].fill(0.1);
        let s = Shape::modes(&bins, periodic);
        assert_eq!(&s.bins[..s.count], &[3, 17]);
        bins[17..20].fill(0.15);
        let s = Shape::modes(&bins, periodic);
        assert_eq!(&s.bins[..s.count], &[17, 3]);
        let mut tiny = [1_f64 / 32.; BINS];
        tiny[5] = f64::from_bits(tiny[5].to_bits() + 1);
        let s = Shape::modes(&tiny, periodic);
        assert_eq!(&s.bins[..s.count], &[5]);
    }
    let mut bins = [0.; BINS];
    bins[30..].fill(0.25);
    bins[..2].fill(0.25);
    let circular = Shape::modes(&bins, true);
    assert_eq!(&circular.bins[..circular.count], &[30]);
    let linear = Shape::modes(&bins, false);
    assert_eq!(&linear.bins[..linear.count], &[0, 30]);
    let alternating = std::array::from_fn(|i| if i % 2 == 0 { 1. / 16. } else { 0. });
    for periodic in [true, false] {
        let s = Shape::modes(&alternating, periodic);
        assert_eq!(s.count, 16);
        assert_eq!(s.bins, std::array::from_fn(|i| i as u8 * 2));
    }
}

#[test]
fn continuous_residual_matches_closed_form_uniform_moments_and_all_modes() {
    let r = record([1., 2.], [0., 1.], 1., 10);
    let expected = ((1_f64 - 1.0625).powi(2) + 1. / 6.) / 16.;
    assert!((r.residual(&[8], false) - expected).abs() < 1e-15);
    let center = 31.5 / 32.;
    for shift in [-10., 0., 10.] {
        let r = record(
            [center + shift - 0.1, center + shift + 0.1],
            [-0.05, 0.05],
            1.,
            10,
        );
        assert!((r.residual(&[31], true) - 1. / 60.).abs() < 1e-14);
    }
    let uniform = record([0., 1.], [0., 0.], 1., 10);
    assert!((uniform.residual(&[0], true) - 1. / 3.).abs() < 1e-15);
    let modes: [u8; 16] = std::array::from_fn(|i| i as u8 * 2);
    assert!((uniform.residual(&modes, true) - 1. / 768.).abs() < 1e-15);
    let third = record([16.5 / 32.; 2], [0., 0.], 1., 10);
    assert_eq!(third.residual(&[0, 8, 16], true), 0.);
    assert!(third.residual(&[0, 8], true) > 0.2);
    let overflow = record([10., 11.], [0., 0.], 1., 10);
    let expected = ((10.5_f64 - 0.0625).powi(2) + 1. / 12.) / 16.;
    assert!(expected > 1.);
    assert!((overflow.residual(&[0], false) - expected).abs() < 1e-14);
}

#[test]
fn continuous_residual_matches_independent_two_dimensional_timestamp_grid() {
    let cases = [
        ([0.91, 1.08], [-0.09, 0.03]),
        ([1.1, 1.2], [-0.1, 0.1]),
        ([-1., 5.], [-0.3, 0.4]),
        ([4.2, 4.7], [0.1, 0.4]),
    ];
    let n = 601;
    let mut maximum = 0_f64;
    for periodic in [true, false] {
        for (target, anchor) in cases {
            for modes in [&[0_u8][..], &[0, 8, 17, 31][..]] {
                let mut reference = 0.;
                for i in 0..n {
                    for j in 0..n {
                        let t = target[0] + (target[1] - target[0]) * (i as f64 + 0.5) / n as f64;
                        let a = anchor[0] + (anchor[1] - anchor[0]) * (j as f64 + 0.5) / n as f64;
                        let distance = modes
                            .iter()
                            .map(|&bin| {
                                let position =
                                    (f64::from(bin) + 0.5) / 32. * if periodic { 1. } else { 4. };
                                let mut d = (t - a - position).abs();
                                if periodic {
                                    d = d.rem_euclid(1.).min(1. - d.rem_euclid(1.));
                                }
                                (d / if periodic { 0.5 } else { 4. }).powi(2)
                            })
                            .fold(f64::INFINITY, f64::min);
                        reference += distance / f64::from(n * n);
                    }
                }
                let expected = record(target, anchor, 1., 10).residual(modes, periodic);
                let error = (expected - reference).abs();
                maximum = maximum.max(error);
                assert!(
                    error < 0.00002,
                    "periodic={periodic} target={target:?} anchor={anchor:?} modes={modes:?} expected={expected} grid={reference}"
                );
            }
        }
    }
    println!("timing dispersion independent grid maximum_error={maximum}");
}

#[test]
fn raw_map_distinguishes_uniform_missing_and_family_specific_zeros() {
    let mut history = History::new();
    let mut empty = summary(&history, true);
    history.describe(&mut empty);
    assert_eq!(empty.features(1.), [None; 14]);
    history.insert(record([0., 1.], [0., 0.], 1., 10), true);
    let mut uniform = summary(&history, true);
    history.describe(&mut uniform);
    assert_eq!(uniform.mode_count, Some(0));
    assert_eq!(uniform.residual_dispersion, None);
    assert_eq!(&uniform.features(1.)[..8], &[None; 8]);
    assert_eq!(
        &uniform.features(1.)[8..],
        &[Some(1.), Some(1.), None, Some(0.), Some(0.), Some(1.)]
    );
    history.clear();
    history.insert(record([0.02, 0.025], [0., 0.], 0.5, 20), true);
    let mut periodic = summary(&history, true);
    history.describe(&mut periodic);
    assert_eq!(periodic.mode_bins, [Some(0), None]);
    let values = periodic.features(0.75);
    assert_eq!(&values[4..8], &[None; 4]);
    assert_eq!(values[2], Some(0.));
    assert_eq!(values[3], Some(1.));
    assert_eq!(values[9], Some(0.75));
    assert!((values[0].unwrap() - (std::f64::consts::TAU / 64.).cos()).abs() < 1e-15);
    history.clear();
    history.insert(record([1.01, 1.02], [0., 0.], 1., 30), false);
    let mut linear = summary(&history, false);
    history.describe(&mut linear);
    let values = linear.features(0.25);
    assert_eq!(&values[..3], &[Some(0.), Some(0.), Some(8.5 / 32.)]);
    assert_eq!(values[8], Some(0.));
    assert!(values[10].unwrap() > 0.);
    linear.supported = false;
    assert_eq!(linear.features(1.), [None; 14]);
    history.clear();
    history.insert(record([5., 6.], [0., 0.], 1., 40), false);
    let mut overflow = summary(&history, false);
    history.describe(&mut overflow);
    let values = overflow.features(1.);
    assert_eq!(values[10], None);
    assert_eq!(values[11], Some(0.));
    assert_eq!(values[12], Some(1.));
}

#[test]
fn feature_cache_expires_with_records_and_uses_all_modes_for_dispersion() {
    let mut history = History::new();
    for (bin, weight, end) in [(0, 1., 10), (8, 0.8, 20), (16, 0.6, 30)] {
        let center = (bin as f64 + 0.5) / 32.;
        history.insert(
            record([center - 0.001, center + 0.001], [0., 0.], weight, end),
            true,
        );
    }
    let mut s = summary(&history, true);
    history.describe(&mut s);
    assert_eq!(s.mode_bins, [Some(0), Some(8)]);
    assert_eq!(s.mode_count, Some(3));
    assert!((s.residual_dispersion.unwrap() - 4e-6 / 3.).abs() < 1e-15);
    let cached = history.shape.get().unwrap().1;
    history.describe(&mut s);
    assert_eq!(history.shape.get().unwrap().1.dispersion, cached.dispersion);
    history.expire(20, true);
    assert!(history.shape.get().is_none());
    let mut s = summary(&history, true);
    history.describe(&mut s);
    assert_eq!(s.mode_bins, [Some(16), None]);
    assert_eq!(s.mode_count, Some(1));
    history.clear();
    assert!(history.shape.get().is_none());
    assert_eq!(summary(&history, true).features(1.), [None; 14]);
}
