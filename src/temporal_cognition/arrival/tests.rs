use super::*;

pub(super) fn config(model: ArrivalModel) -> TemporalPeriodConfig {
    let mut coefficients = [0.; 18];
    coefficients[0] = 2_f64.exp_m1().ln();
    TemporalPeriodConfig {
        model,
        coefficients,
        means: [0.; 8],
        deviations: [1.; 8],
        horizon_sec: 0.1,
    }
}
pub(super) fn accent(end: u64) -> Accent {
    Accent {
        group: Handle {
            bus: 0,
            epoch: 2,
            generation: 3,
        },
        event_start: end - 10,
        event_end: end,
        raw_intervals: std::array::from_fn(|i| {
            (end - 30 + i as u64 * 10, end - 20 + i as u64 * 10)
        }),
        source_start: end - 30,
        source_end: end + 10,
        available_end: end + 10,
        weight: 0.5,
        observed_prefix: end,
    }
}
pub(super) fn raw(start: u64, end: u64) -> RawDescriptor {
    RawDescriptor {
        group: accent(100).group,
        start,
        end,
        known_samples: end - start,
        values: [None; 10],
        source_start: start,
        source_end: end,
        available_end: end,
    }
}
pub(super) fn context(end: u64) -> Context {
    Context {
        peak: Some(Peak {
            bin: 96,
            period_seconds: 0.5,
            support: 0.6,
        }),
        word: Some(false),
        source_start: 0,
        source_end: end,
        available: end,
    }
}
#[test]
fn observed_omissions_advance_without_reset_and_constant_hazard_matches_closed_form() {
    let mut engine = Engine::new(config(ArrivalModel::Hazard)).unwrap();
    assert!(
        engine
            .advance(Some(&raw(0, 100)), None, context(100), 100, 1000)
            .unwrap()
            .is_none()
    );
    let first = engine
        .advance(
            Some(&raw(100, 110)),
            Some(accent(100)),
            context(110),
            110,
            1000,
        )
        .unwrap()
        .unwrap();
    for end in (120..=810).step_by(10) {
        let f = engine
            .advance(Some(&raw(end - 10, end)), None, context(end), end, 1000)
            .unwrap()
            .unwrap();
        assert_eq!(f.last_accent, 100);
        assert!(!f.reset_unknown);
        assert_eq!(f.elapsed_seconds, [(end - 100) as f64 / 1000.; 2]);
        assert!((f.probability.unwrap()[0] - (1. - (-0.2_f64).exp())).abs() < 1e-10);
        assert!(
            (f.observed_survival.unwrap() - (-2. * (end - 100) as f64 / 1000.).exp()).abs() < 1e-10
        );
    }
    assert_eq!(first.issued_at, 110);
    let mut changed = config(ArrivalModel::Hazard);
    changed.coefficients[1] = 3.;
    let mut model = Engine::new(changed).unwrap();
    let a = model
        .advance(
            Some(&raw(0, 110)),
            Some(accent(100)),
            context(110),
            110,
            1000,
        )
        .unwrap()
        .unwrap();
    let b = model
        .advance(Some(&raw(110, 810)), None, context(810), 810, 1000)
        .unwrap()
        .unwrap();
    assert!(b.probability.unwrap()[0] > a.probability.unwrap()[0]);
}
#[test]
fn acquisition_gaps_keep_reset_bounds_and_exclude_cross_gap_ratios() {
    let mut e = Engine::new(config(ArrivalModel::Hazard)).unwrap();
    for end in [100, 600, 1100, 1600, 2100] {
        let start = e.cut.unwrap_or(0);
        e.advance(
            Some(&raw(start, end + 10)),
            Some(accent(end)),
            context(end + 10),
            end + 10,
            1000,
        )
        .unwrap();
    }
    assert_eq!(e.intervals, [Some(0.5); 4]);
    let gap = e
        .advance(None, None, context(2110), 2310, 1000)
        .unwrap()
        .unwrap();
    assert_eq!(gap.elapsed_seconds, [0., 0.21]);
    assert!(gap.reset_unknown);
    let after = e
        .advance(Some(&raw(2310, 2410)), None, context(2410), 2410, 1000)
        .unwrap()
        .unwrap();
    assert_eq!(after.elapsed_seconds, [0.1, 0.31]);
    assert!(after.observed_survival.is_none());
    let reset = e
        .advance(
            Some(&raw(2410, 2610)),
            Some(accent(2600)),
            context(2610),
            2610,
            1000,
        )
        .unwrap()
        .unwrap();
    assert!(!reset.reset_unknown);
    assert_eq!(e.intervals, [None; 4]);
    e.advance(
        Some(&raw(2610, 3110)),
        Some(accent(3100)),
        context(3110),
        3110,
        1000,
    )
    .unwrap();
    assert_eq!(e.intervals, [Some(0.5), None, None, None]);
}
#[test]
fn frozen_layout_masks_ratios_and_encodes_circular_phase() {
    let mut e = Engine::new(config(ArrivalModel::Hazard)).unwrap();
    e.intervals = [Some(0.5), Some(1.), None, Some(0.25)];
    let x = e.features(0.125, context(100));
    assert_eq!(&x[2..8], &[2_f64.ln(), 0., 0.5_f64.ln(), 0., 1., 0.]);
    assert_eq!(x[8], -1.);
    assert_eq!(x[9], 0.6);
    assert!(x[10].abs() < 1e-12);
    assert!((x[11] - 1.).abs() < 1e-12);
    assert_eq!(&x[12..18], &[0., 0., 0., 0., 0., 0.]);
    let absent = e.features(0.125, Context::default());
    assert_eq!(&absent[8..13], &[0.; 5]);
    assert_eq!(&absent[13..18], &[1.; 5]);
    let a = e.features(0.4999, context(100));
    let b = e.features(0.0001, context(100));
    assert!((a[10] - b[10]).abs() < 1e-5);
    assert!((a[11] - b[11]).abs() < 0.003);
}
#[test]
fn uncertain_phase_bounds_enclose_interior_predictions_and_work_is_bounded() {
    let mut cfg = config(ArrivalModel::Hazard);
    cfg.coefficients[1] = 0.7;
    cfg.coefficients[10] = 8.;
    cfg.coefficients[11] = -4.;
    let e = Engine::new(cfg).unwrap();
    let bounds = e.uncertain_probability([0., 0.7], context(100)).unwrap();
    for i in 0..=100 {
        let elapsed = i as f64 * 0.007;
        let (value, n) = e.integral(elapsed, elapsed + 0.1, context(100));
        assert!(n <= 62);
        let reference = (0..10000)
            .map(|j| {
                e.hazard(elapsed + (j as f64 + 0.5) * 0.1 / 10000., context(100))
                    .unwrap()
            })
            .sum::<f64>()
            * 0.1
            / 10000.;
        if let Some(value) = value {
            assert!((value - reference).abs() < 2e-6);
        }
        let p = -(-reference).exp_m1();
        assert!(p >= bounds[0] - 1e-12 && p <= bounds[1] + 1e-12);
    }
    let (value, n) = e.integral(0., 32., context(100));
    assert!(value.is_none());
    assert_eq!(n, 62);
}
#[test]
fn phase_independent_hazard_keeps_support_across_short_periods_and_long_horizons() {
    for period in [None, Some(0.001), Some(0.17174420593226114), Some(0.5)] {
        for bias in [-2_f64, 0., 2.] {
            for elapsed in [0., 0.23466666666666666, 10.] {
                for horizon in [0.1, 1., 32.] {
                    let mut cfg = config(ArrivalModel::Hazard);
                    cfg.coefficients = [0.; 18];
                    cfg.coefficients[0] = bias;
                    cfg.coefficients[8] = 0.1;
                    cfg.coefficients[9] = -0.2;
                    cfg.coefficients[16] = 0.25;
                    cfg.coefficients[17] = -0.5;
                    cfg.horizon_sec = horizon;
                    let e = Engine::new(cfg).unwrap();
                    let mut ctx = context(110);
                    ctx.peak = period.map(|period_seconds| Peak {
                        period_seconds,
                        ..ctx.peak.unwrap()
                    });
                    let z = bias + period.map_or(-0.25, |p| 0.1 * p.log2() - 0.2 * 0.6);
                    let hazard = (1. + z.exp()).ln();
                    let (integral, evaluations) = e.integral(elapsed, elapsed + horizon, ctx);
                    assert_eq!(evaluations, 6);
                    assert!((integral.unwrap() - hazard * horizon).abs() < 2e-13);
                    let (probability, evaluations) = e.probability([elapsed; 2], ctx, false);
                    assert_eq!(evaluations, 6);
                    for p in probability.unwrap() {
                        assert!((p - (1. - (-hazard * horizon).exp())).abs() < 2e-13);
                    }
                }
            }
        }
    }
    let mut cfg = config(ArrivalModel::Hazard);
    cfg.coefficients = [0.; 18];
    cfg.coefficients[1] = 1.;
    let e = Engine::new(cfg).unwrap();
    let mut ctx = context(110);
    ctx.peak.as_mut().unwrap().period_seconds = 0.001;
    for lo in [0_f64, 0.23466666666666666, 10.] {
        let primitive = |t: f64| (2. + t) * (2. + t).ln() - (2. + t);
        let expected = primitive(lo + 1.) - primitive(lo);
        let (actual, evaluations) = e.integral(lo, lo + 1., ctx);
        assert!(evaluations <= 62);
        assert!((actual.unwrap() - expected).abs() <= 1e-8 + 1e-6 * expected);
    }
}

#[test]
fn nonzero_phase_coefficients_keep_resolution_guard_and_match_independent_quadrature() {
    let mut ctx = context(110);
    ctx.peak.as_mut().unwrap().period_seconds = 0.17174420593226114;
    for (cosine, sine) in [(1e-15, 0.), (0., -1e-15), (0.4, -0.3)] {
        let mut cfg = config(ArrivalModel::Hazard);
        cfg.coefficients = [0.; 18];
        cfg.coefficients[10] = cosine;
        cfg.coefficients[11] = sine;
        cfg.horizon_sec = 1.;
        let e = Engine::new(cfg).unwrap();
        assert_eq!(e.integral(0., 1., ctx), (None, 62));
        let (integral, evaluations) = e.integral(0.017, 0.047, ctx);
        let reference = (0..10_000)
            .map(|j| {
                let t = 0.017 + (j as f64 + 0.5) * 0.03 / 10_000.;
                let phase = std::f64::consts::TAU * t / 0.17174420593226114;
                let z = cosine * phase.cos() + sine * phase.sin();
                (1. + z.exp()).ln()
            })
            .sum::<f64>()
            * 0.03
            / 10_000.;
        assert!(evaluations <= 62);
        assert!((integral.unwrap() - reference).abs() < 1e-8);
        ctx.peak = None;
        assert_eq!(e.integral(0., 1., ctx).1, 6);
        ctx = context(110);
        ctx.peak.as_mut().unwrap().period_seconds = 0.17174420593226114;
    }
}

#[test]
fn mr2_same_accent_trace_swaps_model_without_changing_prediction_contract() {
    let mut hazard = Engine::new(config(ArrivalModel::Hazard)).unwrap();
    let mut periodic = Engine::new(config(ArrivalModel::Periodic)).unwrap();
    let mut previous = 0;
    // Regular pulses, omission, tempo change, then nonmetrical intervals.
    for end in [100, 600, 1600, 2000, 2400, 2691, 3378, 3589] {
        let raw = raw(previous, end + 10);
        let ctx = context(end + 10);
        let a = accent(end);
        let h = hazard
            .advance(Some(&raw), Some(a), ctx, end + 10, 1000)
            .unwrap()
            .unwrap();
        let p = periodic
            .advance(Some(&raw), Some(a), ctx, end + 10, 1000)
            .unwrap()
            .unwrap();
        assert_eq!(
            (
                h.group,
                h.source_start,
                h.source_end,
                h.available,
                h.issued_at,
                h.horizon_end,
                h.last_accent,
                h.elapsed_seconds
            ),
            (
                p.group,
                p.source_start,
                p.source_end,
                p.available,
                p.issued_at,
                p.horizon_end,
                p.last_accent,
                p.elapsed_seconds
            )
        );
        assert!(h.valid_for(a.group, ArrivalModel::Hazard, end + 10));
        assert!(!h.valid_for(a.group, ArrivalModel::Periodic, end + 10));
        assert!(!h.valid_for(
            Handle {
                epoch: 3,
                ..a.group
            },
            ArrivalModel::Hazard,
            end + 10
        ));
        assert!(!h.valid_for(a.group, ArrivalModel::Hazard, h.horizon_end + 1));
        assert!(h.probability.unwrap()[0] > 0.);
        assert_eq!(p.probability, Some([0.; 2]));
        previous = end + 10;
    }
    assert_eq!(hazard.intervals, periodic.intervals);
    assert!(
        Engine::new(config(ArrivalModel::Periodic))
            .unwrap()
            .last
            .is_none()
    );
}

#[test]
fn delayed_delivery_and_eof_preserve_evidence_and_old_interval_support() {
    let mut invalid = raw(0, 10);
    invalid.start = 20;
    assert!(
        Engine::new(config(ArrivalModel::Hazard))
            .unwrap()
            .advance(Some(&invalid), None, context(10), 10, 1000)
            .is_err()
    );
    let mut e = Engine::new(config(ArrivalModel::Hazard)).unwrap();
    let f = e
        .advance(
            Some(&raw(0, 110)),
            Some(accent(100)),
            context(110),
            310,
            1000,
        )
        .unwrap()
        .unwrap();
    assert!(f.reset_unknown);
    assert_eq!(f.elapsed_seconds, [0., 0.21]);
    assert_eq!((f.source_end, f.available), (110, 110));
    let eof = e
        .advance(None, None, Context::default(), 410, 1000)
        .unwrap()
        .unwrap();
    assert_eq!(
        (eof.source_start, eof.source_end, eof.available),
        (70, 110, 110)
    );
    assert_eq!(eof.issued_at, 410);
    let mut e = Engine::new(config(ArrivalModel::Hazard)).unwrap();
    let mut prior = 0;
    for end in [100, 20100, 40100, 60100, 80100] {
        let ctx = Context {
            source_start: end - 30,
            ..context(end + 10)
        };
        let f = e
            .advance(
                Some(&raw(prior, end + 10)),
                Some(accent(end)),
                ctx,
                end + 10,
                1000,
            )
            .unwrap()
            .unwrap();
        assert_eq!(f.source_start, 70);
        prior = end + 10;
    }
    assert_eq!(e.intervals, [Some(20.); 4]);
}
