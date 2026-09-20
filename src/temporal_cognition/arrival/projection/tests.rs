use super::super::tests::{accent, config, context, raw};
use super::*;

fn engine() -> Engine {
    let mut e = Engine::new(config(ArrivalModel::Hazard)).unwrap();
    e.advance(
        Some(&raw(0, 110)),
        Some(accent(100)),
        context(110),
        110,
        1000,
    )
    .unwrap();
    e
}

#[test]
fn frozen_candidate_uses_same_hazard_without_mutating_observed_engine() {
    let e = engine();
    let original = serde_json::to_string(&e).unwrap();
    let frozen = e.freeze(accent(100).group, 110, 1000).unwrap();
    let issue = frozen.project(110).unwrap().finish().unwrap();
    assert_eq!(
        issue.probability,
        Feature::Observed(e.probability([0.01; 2], context(110), false).0.unwrap()[0])
    );
    let mut scratch = frozen.project(150).unwrap();
    scratch.receipt([100, 110], 120, true, None).unwrap();
    scratch.receipt([110, 120], 130, true, Some(0.5)).unwrap();
    scratch.receipt([120, 130], 140, true, None).unwrap();
    scratch.receipt([130, 140], 150, true, Some(1.)).unwrap();
    let result = scratch.finish().unwrap();
    assert_eq!(result.candidate_last_accent, Some(140));
    assert_eq!(result.candidate_accents, 2);
    assert_eq!(
        scratch.engine.intervals,
        [Some(0.02), Some(0.02), None, None]
    );
    assert_eq!(result.elapsed_seconds, 0.01);
    assert!((result.probability.value().unwrap() - (1. - (-0.2_f64).exp())).abs() < 1e-12);
    assert!(!result.reset_unknown);
    assert_eq!(original, serde_json::to_string(&e).unwrap());
}

#[test]
fn short_period_with_zero_phase_weights_supports_observed_and_candidate_forecasts() {
    let mut cfg = config(ArrivalModel::Hazard);
    cfg.coefficients = [0.; 18];
    cfg.horizon_sec = 1.;
    let mut ctx = context(110);
    ctx.peak.as_mut().unwrap().period_seconds = 0.17174420593226114;
    let mut e = Engine::new(cfg).unwrap();
    let observed = e
        .advance(Some(&raw(0, 110)), Some(accent(100)), ctx, 110, 1000)
        .unwrap()
        .unwrap();
    assert_eq!(observed.evaluations, 6);
    for p in observed.probability.unwrap() {
        assert!((p - 0.5).abs() < 1e-14);
    }
    let original = serde_json::to_string(&e).unwrap();
    let frozen = e.freeze(accent(100).group, 110, 1000).unwrap();
    let mut candidate = frozen.project(150).unwrap();
    candidate.receipt([100, 110], 120, false, None).unwrap();
    candidate.receipt([110, 120], 130, true, Some(0.5)).unwrap();
    candidate.receipt([120, 130], 140, true, None).unwrap();
    candidate.receipt([130, 140], 150, true, None).unwrap();
    let result = candidate.finish().unwrap();
    assert_eq!(result.evaluations, 6);
    assert!(!result.reset_unknown && !result.incomplete_tail);
    assert!(matches!(result.probability, Feature::Projected(p) if (p - 0.5).abs() < 1e-14));
    assert_eq!(result.candidate_last_accent, Some(120));
    assert_eq!(original, serde_json::to_string(&e).unwrap());
}

#[test]
fn candidate_can_resolve_new_gap_but_cannot_redeem_original_unknown() {
    for original_gap in [false, true] {
        let mut e = engine();
        if original_gap {
            e.uncertain_since = Some(105);
        }
        let f = e.freeze(accent(100).group, 110, 1000).unwrap();
        let mut s = f.project(150).unwrap();
        s.receipt([100, 110], 120, false, None).unwrap();
        s.receipt([110, 120], 130, true, None).unwrap();
        assert_eq!(s.finish().unwrap().probability, Feature::Unsupported);
        s.receipt([120, 130], 140, true, Some(1.)).unwrap();
        s.receipt([130, 140], 150, true, None).unwrap();
        assert_eq!(s.engine.intervals, [None; 4]);
        let p = s.finish().unwrap();
        assert_eq!(p.original_reset_unknown, original_gap);
        assert_eq!(p.reset_unknown, original_gap);
        assert_eq!(p.probability.value().is_some(), !original_gap);
    }
}

#[test]
fn missing_receipts_partial_tail_and_discontinuous_centers_do_not_mean_silence() {
    let f = engine().freeze(accent(100).group, 110, 1000).unwrap();
    assert_eq!(
        f.project(120).unwrap().finish().unwrap().probability,
        Feature::Unsupported
    );
    let mut s = f.project(151).unwrap();
    s.receipt([100, 110], 120, true, None).unwrap();
    s.receipt([120, 130], 140, true, None).unwrap();
    s.receipt([130, 140], 150, true, None).unwrap();
    let r = s.finish().unwrap();
    assert!(r.incomplete_tail && r.reset_unknown);
    assert_eq!(r.unsupported_receipts, 1);
    assert_eq!(r.probability, Feature::Unsupported);
    let mut s = f.project(130).unwrap();
    s.receipt([110, 120], 130, true, None).unwrap();
    assert!(s.finish().unwrap().reset_unknown);
}

#[test]
fn invalid_receipts_are_atomic_and_issue_identity_is_checked() {
    let e = engine();
    let group = accent(100).group;
    assert!(
        e.freeze(
            Handle {
                generation: 4,
                ..group
            },
            110,
            1000
        )
        .is_none()
    );
    assert!(e.freeze(group, 111, 1000).is_none());
    assert!(e.freeze(group, 110, 0).is_none());
    assert!(
        Engine::new(config(ArrivalModel::Hazard))
            .unwrap()
            .freeze(group, 110, 1000)
            .is_none()
    );
    let f = e.freeze(group, 110, 1000).unwrap();
    assert!(f.project(109).is_none());
    assert!(f.project(4111).is_none());
    let mut s = f.project(150).unwrap();
    for (interval, available, supported, weight) in [
        ([110, 110], 120, true, None),
        ([100, 110], 110, true, None),
        ([100, 110], 151, true, None),
        ([100, 110], 120, false, Some(1.)),
        ([100, 110], 120, true, Some(f64::NAN)),
    ] {
        let before = serde_json::to_string(&s.finish()).unwrap();
        assert!(s.receipt(interval, available, supported, weight).is_err());
        assert_eq!(before, serde_json::to_string(&s.finish()).unwrap());
    }
    s.receipt([100, 110], 120, true, Some(1.)).unwrap();
    let before = serde_json::to_string(&s.finish()).unwrap();
    assert!(s.receipt([100, 110], 130, true, Some(1.)).is_err());
    assert_eq!(before, serde_json::to_string(&s.finish()).unwrap());
}

#[test]
fn candidate_anchor_changes_duration_hazard_and_periodic_forecast() {
    for model in [ArrivalModel::Hazard, ArrivalModel::Periodic] {
        let mut e = engine();
        e.config.model = model;
        e.config.coefficients = [0.; 18];
        e.config.coefficients[1] = 1.;
        let f = e.freeze(accent(100).group, 110, 1000).unwrap();
        let mut keep = f.project(550).unwrap();
        let mut reset = f.project(550).unwrap();
        for end in (110..=540).step_by(10) {
            keep.receipt([end - 10, end], end + 10, true, None).unwrap();
            reset
                .receipt([end - 10, end], end + 10, true, (end == 510).then_some(1.))
                .unwrap();
        }
        let a = keep.finish().unwrap().probability.value().unwrap();
        let b = reset.finish().unwrap().probability.value().unwrap();
        assert!(a > b);
        if matches!(model, ArrivalModel::Hazard) {
            // With coefficient log(1+d), softplus gives log(2+d).
            let primitive = |x: f64| (2. + x) * ((2. + x).ln() - 1.);
            for (actual, age) in [(a, 0.45), (b, 0.04)] {
                let expected = -(-(primitive(age + 0.1) - primitive(age))).exp_m1();
                assert!((actual - expected).abs() < 1e-6);
            }
        } else {
            assert_eq!((a, b), (1., 0.));
        }
    }
}
