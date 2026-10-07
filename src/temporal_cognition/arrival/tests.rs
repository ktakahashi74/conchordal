use super::*;

pub(super) fn config(model: ArrivalModel) -> TemporalPeriodConfig {
    TemporalPeriodConfig {
        model,
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
        source_start: 0,
        source_end: end,
        available: end,
    }
}

#[test]
fn acquisition_gaps_keep_reset_bounds_and_exclude_cross_gap_ratios() {
    let mut e = Engine::new(config(ArrivalModel::Periodic)).unwrap();
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
    assert_eq!(e.interval_starts, [1570, 1070, 570, 70]);
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
    assert_eq!(e.interval_starts, [u64::MAX; 4]);
    e.advance(
        Some(&raw(2610, 3110)),
        Some(accent(3100)),
        context(3110),
        3110,
        1000,
    )
    .unwrap();
    assert_eq!(e.interval_starts, [2570, u64::MAX, u64::MAX, u64::MAX]);
}

#[test]
fn delayed_delivery_and_eof_preserve_evidence_and_old_interval_support() {
    let mut invalid = raw(0, 10);
    invalid.start = 20;
    assert!(
        Engine::new(config(ArrivalModel::Periodic))
            .unwrap()
            .advance(Some(&invalid), None, context(10), 10, 1000)
            .is_err()
    );
    let mut e = Engine::new(config(ArrivalModel::Periodic)).unwrap();
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
    let mut e = Engine::new(config(ArrivalModel::Periodic)).unwrap();
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
    assert_eq!(e.interval_starts, [60070, 40070, 20070, 70]);
}

#[test]
fn periodic_omissions_and_unknown_phase_preserve_forecast_contract() {
    let mut engine = Engine::new(config(ArrivalModel::Periodic)).unwrap();
    let a = accent(100);
    engine
        .advance(Some(&raw(0, 110)), Some(a), context(110), 110, 1000)
        .unwrap();
    for end in [200, 550, 610] {
        let start = engine.cut.unwrap();
        let f = engine
            .advance(Some(&raw(start, end)), None, context(end), end, 1000)
            .unwrap()
            .unwrap();
        assert_eq!(f.last_accent, 100);
        assert!(!f.reset_unknown);
        assert_eq!(f.probability, Some([f64::from(end == 550); 2]));
        assert!(f.valid_for(a.group, ArrivalModel::Periodic, end));
        assert!(!f.valid_for(
            Handle {
                epoch: 3,
                ..a.group
            },
            ArrivalModel::Periodic,
            end
        ));
        assert!(!f.valid_for(a.group, ArrivalModel::Periodic, f.horizon_end + 1));
    }
    let f = engine
        .advance(None, None, context(610), 710, 1000)
        .unwrap()
        .unwrap();
    assert!(f.reset_unknown);
    assert_eq!(f.probability, Some([0., 1.]));
    let f = engine
        .advance(None, None, Context::default(), 810, 1000)
        .unwrap()
        .unwrap();
    assert_eq!(f.probability, None);
    let mut full = Engine::new(TemporalPeriodConfig {
        horizon_sec: 0.5,
        ..config(ArrivalModel::Periodic)
    })
    .unwrap();
    let f = full
        .advance(None, Some(a), context(110), 110, 1000)
        .unwrap()
        .unwrap();
    assert_eq!(f.probability, Some([1.; 2]));
}
