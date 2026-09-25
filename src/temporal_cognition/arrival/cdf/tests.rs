use super::*;
use crate::temporal_cognition::arrival::tests::{accent, context, raw};

const GROUP: Handle = Handle {
    bus: 0,
    epoch: 2,
    generation: 3,
};
const CUT: u64 = 110;
const RATE: u32 = 1000;

fn model(kind: ArrivalModel, coefficients: [f64; 18], period: Option<f64>) -> (Engine, Forecast) {
    let mut cfg = crate::temporal_cognition::arrival::tests::config(kind);
    cfg.coefficients = coefficients;
    cfg.horizon_sec = 4.0;
    let mut engine = Engine::new(cfg).unwrap();
    let mut ctx = context(CUT);
    ctx.peak = period.map(|period_seconds| Peak {
        bin: 96,
        period_seconds,
        support: 0.6,
    });
    ctx.word = period.map(|_| false);
    let forecast = engine
        .advance(Some(&raw(0, CUT)), Some(accent(100)), ctx, CUT, RATE)
        .unwrap()
        .unwrap();
    (engine, forecast)
}

fn primary() -> (Engine, Forecast) {
    let mut coefficients = [0.; 18];
    coefficients[0] = -2.;
    coefficients[10] = 4.;
    model(ArrivalModel::Hazard, coefficients, Some(0.5))
}

#[test]
fn constant_hazard_cdf_matches_analytic_probability_on_all_grid_points() {
    let mut coefficients = [0.; 18];
    coefficients[0] = 2_f64.exp_m1().ln();
    let (engine, forecast) = model(ArrivalModel::Hazard, coefficients, Some(0.5));
    let result = engine.cdf(GROUP, CUT, RATE, Some(forecast));
    assert_eq!(result.unknown, None);
    assert!(result.periodic);
    assert!(result.evaluations <= 200 * 62);
    let cdf = result.cdf.unwrap();
    for (k, &value) in cdf.iter().enumerate() {
        let expected = -(-2. * k as f64 / 50.).exp_m1();
        assert!((value - expected).abs() < 1e-11, "k={k}");
    }
}

#[test]
fn phase_cdf_uses_bounded_small_intervals_independent_of_i6_whole_horizon() {
    let (engine, forecast) = primary();
    assert!(forecast.probability.is_none());
    let result = engine.cdf(GROUP, CUT, RATE, Some(forecast));
    assert_eq!(result.unknown, None);
    let cdf = result.cdf.unwrap();
    assert_eq!(cdf[0], 0.);
    assert!(cdf[200] > cdf[1] && cdf[200] < 1.);
    assert!(cdf.windows(2).all(|w| w[0] <= w[1]));
    assert!(result.evaluations <= 200 * 62);
}

#[test]
fn periodic_step_and_nonperiodic_hazard_stay_distinct() {
    let (periodic, forecast) = model(ArrivalModel::Periodic, [0.; 18], Some(0.5));
    let result = periodic.cdf(GROUP, CUT, RATE, Some(forecast));
    let cdf = result.cdf.unwrap();
    assert_eq!(result.evaluations, 0);
    assert_eq!(cdf[24], 0.);
    assert_eq!(cdf[25], 1.);
    assert_eq!(cdf[200], 1.);

    let (hazard, forecast) = model(ArrivalModel::Hazard, [0.; 18], None);
    let result = hazard.cdf(GROUP, CUT, RATE, Some(forecast));
    assert!(!result.periodic);
    assert!(result.cdf.is_some());
    let (periodic, forecast) = model(ArrivalModel::Periodic, [0.; 18], None);
    let result = periodic.cdf(GROUP, CUT, RATE, Some(forecast));
    assert_eq!(result.unknown, Some(CdfUnknown::Nonperiodic));
}

#[test]
fn cdf_rejects_uncertain_reset_bad_clocks_and_unconverged_intervals() {
    let (engine, forecast) = primary();
    assert_eq!(
        engine.cdf(GROUP, CUT, RATE, None).unknown,
        Some(CdfUnknown::NoForecast)
    );
    assert_eq!(
        engine
            .cdf(
                GROUP,
                CUT,
                RATE,
                Some(Forecast {
                    reset_unknown: true,
                    ..forecast
                })
            )
            .unknown,
        Some(CdfUnknown::ResetUnknown)
    );
    assert_eq!(
        engine
            .cdf(
                GROUP,
                CUT,
                RATE,
                Some(Forecast {
                    elapsed_seconds: [0., 0.01],
                    ..forecast
                })
            )
            .unknown,
        Some(CdfUnknown::UncertainElapsed)
    );
    assert_eq!(
        engine
            .cdf(
                GROUP,
                CUT,
                RATE,
                Some(Forecast {
                    elapsed_seconds: [0.1; 2],
                    ..forecast
                })
            )
            .unknown,
        Some(CdfUnknown::InvalidClock)
    );
    assert_eq!(
        engine.cdf(GROUP, CUT + 1, RATE, Some(forecast)).unknown,
        Some(CdfUnknown::InvalidClock)
    );
    let mut coefficients = [0.; 18];
    coefficients[10] = 4.;
    let (engine, forecast) = model(ArrivalModel::Hazard, coefficients, Some(0.001));
    let result = engine.cdf(GROUP, CUT, RATE, Some(forecast));
    assert_eq!(result.unknown, Some(CdfUnknown::IntegralFailed));
    assert_eq!(result.evaluations, 62);
    assert!(result.cdf.is_none());

    let mut cfg = crate::temporal_cognition::arrival::tests::config(ArrivalModel::Hazard);
    cfg.horizon_sec = 1.;
    let mut engine = Engine::new(cfg).unwrap();
    let forecast = engine
        .advance(
            Some(&raw(0, CUT)),
            Some(accent(100)),
            context(CUT),
            CUT,
            RATE,
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        engine.cdf(GROUP, CUT, RATE, Some(forecast)).unknown,
        Some(CdfUnknown::UnsupportedHorizon)
    );
}

#[test]
#[ignore = "write deterministic I11-2 CDF fixture for independent Python integration"]
fn export_cdf_fixture() {
    let output = std::env::var("I11_CDF_FIXTURE_OUT").expect("fixture output path");
    let mut constant = [0.; 18];
    constant[0] = 2_f64.exp_m1().ln();
    let (constant_engine, constant_forecast) = model(ArrivalModel::Hazard, constant, Some(0.5));
    let (phase_engine, phase_forecast) = primary();
    let (periodic_engine, periodic_forecast) = model(ArrivalModel::Periodic, [0.; 18], Some(0.5));
    let (nonperiodic_engine, nonperiodic_forecast) = model(ArrivalModel::Hazard, [0.; 18], None);
    let mut failing = [0.; 18];
    failing[10] = 4.;
    let (failing_engine, failing_forecast) = model(ArrivalModel::Hazard, failing, Some(0.001));
    let cases = [
        fixture_case("constant_hazard", &constant_engine, constant_forecast),
        fixture_case("phase_hazard", &phase_engine, phase_forecast),
        fixture_case("periodic", &periodic_engine, periodic_forecast),
        fixture_case(
            "nonperiodic_hazard",
            &nonperiodic_engine,
            nonperiodic_forecast,
        ),
        fixture_case("integral_failed", &failing_engine, failing_forecast),
    ];
    std::fs::write(
        output,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema": "i11-2-cdf-fixture-v2",
            "cases": cases,
        }))
        .unwrap(),
    )
    .unwrap();
}

fn fixture_case(name: &str, engine: &Engine, forecast: Forecast) -> serde_json::Value {
    let result = engine.cdf(GROUP, CUT, RATE, Some(forecast));
    let mut cumulative_hazard = [0.; CDF_LEN];
    let mut interval_evaluations = Vec::with_capacity(CDF_LEN - 1);
    let mut failed_interval = None;
    if engine.config.model == ArrivalModel::Hazard {
        let elapsed = forecast.elapsed_seconds[1];
        for k in 1..CDF_LEN {
            let (integral, evaluations) = engine.integral(
                elapsed + (k - 1) as f64 / 50.,
                elapsed + k as f64 / 50.,
                engine.context,
            );
            interval_evaluations.push(evaluations);
            let Some(integral) = integral else {
                failed_interval = Some(k);
                break;
            };
            cumulative_hazard[k] = cumulative_hazard[k - 1] + integral;
        }
    }
    let complete_hazard = engine.config.model == ArrivalModel::Hazard && failed_interval.is_none();
    let legacy_full_interval_integral = (engine.config.model == ArrivalModel::Hazard).then(|| {
        let elapsed = forecast.elapsed_seconds[1];
        (0..CDF_LEN)
            .map(|k| {
                if k == 0 {
                    Some(0.)
                } else {
                    engine
                        .integral(elapsed, elapsed + k as f64 / 50., engine.context)
                        .0
                }
            })
            .collect::<Vec<_>>()
    });
    serde_json::json!({
        "name": name,
        "config": engine.config,
        "group": GROUP,
        "sample_rate": RATE,
        "issued_at": CUT,
        "horizon_end": forecast.horizon_end,
        "elapsed_seconds": forecast.elapsed_seconds,
        "reset_unknown": forecast.reset_unknown,
        "context": engine.context,
        "intervals": engine.intervals,
        "forecast_probability": forecast.probability,
        "cdf": result.cdf.as_ref().map(|values| values.as_slice()),
        "unknown": result.unknown,
        "cumulative_hazard": complete_hazard.then_some(cumulative_hazard.as_slice()),
        "legacy_full_interval_integral": legacy_full_interval_integral,
        "interval_evaluations": (engine.config.model == ArrivalModel::Hazard).then_some(interval_evaluations),
        "failed_interval": failed_interval,
        "evaluations": result.evaluations,
        "forced_out_of_estimator_period_range": name == "integral_failed",
    })
}
