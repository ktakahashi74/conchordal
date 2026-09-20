//! Bounded duration-hazard integration with a fourth-derivative error certificate.

pub(super) fn integrate(intercept: f64, slope: f64, lo: f64, hi: f64) -> Option<f64> {
    if [intercept, slope, lo, hi].iter().any(|v| !v.is_finite()) || lo < 0. || hi < lo {
        return None;
    }
    if lo == hi {
        return Some(0.);
    }
    let mut sigmoids = [0.; 2];
    for (value, time) in sigmoids.iter_mut().zip([lo, hi]) {
        let z = intercept + slope * time.ln_1p();
        if !z.is_finite() {
            return None;
        }
        let odds = (-z.abs()).exp();
        *value = if z >= 0. {
            1. / (1. + odds)
        } else {
            odds / (1. + odds)
        };
    }
    let low = sigmoids[0].min(sigmoids[1]);
    let range = (sigmoids[0] - sigmoids[1]).abs();
    let b = slope;
    let coefficients = [
        0.,
        b.powi(4) - 6. * b.powi(3) + 11. * b * b - 6. * b,
        -7. * b.powi(4) + 18. * b.powi(3) - 11. * b * b,
        12. * b.powi(4) - 12. * b.powi(3),
        -6. * b.powi(4),
    ];
    if coefficients.iter().any(|v| !v.is_finite()) {
        return None;
    }
    const CHOOSE: [[f64; 5]; 5] = [
        [1., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0.],
        [1., 2., 1., 0., 0.],
        [1., 3., 3., 1., 0.],
        [1., 4., 6., 4., 1.],
    ];
    let power: [f64; 5] = std::array::from_fn(|j| {
        (j..5)
            .map(|k| {
                coefficients[k] * CHOOSE[k][j] * low.powi((k - j) as i32) * range.powi(j as i32)
            })
            .sum()
    });
    let bernstein: [f64; 5] = std::array::from_fn(|i| {
        (0..=i)
            .map(|j| power[j] * CHOOSE[i][j] / CHOOSE[4][j])
            .sum()
    });
    let rounding = 64. * f64::EPSILON * coefficients.iter().map(|v| v.abs()).sum::<f64>();
    let derivative_bound =
        (bernstein.iter().map(|v| v.abs()).fold(0., f64::max) + rounding) / (1. + lo).powi(4);
    if !derivative_bound.is_finite() {
        return None;
    }
    let mut previous = None;
    // Two endpoint evaluations plus 2*(1+2+4+8+16) quadrature evaluations.
    for panels in [1, 2, 4, 8, 16] {
        let width = (hi - lo) / panels as f64;
        let mut integral = 0.;
        for i in 0..panels {
            let midpoint = lo + (i as f64 + 0.5) * width;
            let offset = width / (2. * 3_f64.sqrt());
            for time in [midpoint - offset, midpoint + offset] {
                let z = intercept + slope * time.ln_1p();
                integral += 0.5 * width * (z.max(0.) + (-z.abs()).exp().ln_1p());
            }
        }
        if !integral.is_finite() {
            return None;
        }
        let error =
            (hi - lo) * width.powi(4) * derivative_bound / 4320. + 128. * f64::EPSILON * integral;
        let tolerance = 1e-9 + 1e-7 * integral;
        if previous.is_some_and(|old: f64| (old - integral).abs().max(error) <= tolerance) {
            return Some(integral);
        }
        previous = Some(integral);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn certified_integrals_match_independent_python_reference() {
        let data: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/temporal_cognition/section_hazards.json"
        ))
        .unwrap();
        for row in data["cases"].as_array().unwrap() {
            let args = row["input"].as_array().unwrap();
            let actual = integrate(
                args[0].as_f64().unwrap(),
                args[1].as_f64().unwrap(),
                args[2].as_f64().unwrap(),
                args[3].as_f64().unwrap(),
            );
            match (actual, row["expected"].as_f64()) {
                (Some(a), Some(e)) => assert!((a - e).abs() <= 1e-9 + 1e-7 * e.abs(), "{row}: {a}"),
                (None, None) => (),
                _ => panic!("integration support differs: {row}, actual {actual:?}"),
            }
        }
    }

    #[test]
    fn invalid_or_unresolvable_integrals_never_supply_survival_evidence() {
        for (a, b, lo, hi) in [
            (20., -40., 0., 1200.),
            (0., 1e100, 0., 1.),
            (f64::NAN, 0., 0., 1.),
            (0., 0., 2., 1.),
            (0., 0., -1., 1.),
            (0., 0., 0., f64::INFINITY),
        ] {
            assert_eq!(integrate(a, b, lo, hi), None);
        }
        assert_eq!(integrate(0., 1., 1., 1.), Some(0.));
        let h = integrate(0., 0., 0., 2.).unwrap();
        assert!((h - 2. * 2f64.ln()).abs() < 1e-14);
    }
}
