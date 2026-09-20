//! Scoring-only whole-piece completion from supported lower-head predictions.

use super::{phrase, recall, section};
use crate::config::{TemporalScoringConfig, TemporalWholeConfig};

pub(crate) mod controls;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Rating {
    pub supported_categories: Option<[f64; 5]>,
    pub categories: [f64; 5],
    pub support: f64,
    pub expected_rating: f64,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Snapshot {
    pub observed_end_sample: u64,
    pub graph: Option<recall::GraphSnapshot>,
    pub controls: Option<controls::Snapshot>,
    pub values: [Option<f64>; 4],
    pub input_support: [f64; 4],
    pub coordinates: [f64; 8],
    pub supported_categories: Option<[f64; 5]>,
    pub categories: [f64; 5],
    pub support: f64,
    pub expected_rating: f64,
}

fn head(c: TemporalWholeConfig) -> TemporalScoringConfig<4, 9> {
    TemporalScoringConfig {
        means: c.means,
        deviations: c.deviations,
        coefficients: c.coefficients,
        cutpoints: c.cutpoints,
        prior: c.prior,
        temperature: c.temperature,
    }
}

fn validate_head<const N: usize, const C: usize>(
    c: TemporalScoringConfig<N, C>,
) -> Result<(), &'static str> {
    if C != 2 * N + 1
        || c.means
            .iter()
            .chain(&c.deviations)
            .chain(&c.coefficients)
            .chain(&c.cutpoints)
            .chain(&c.prior)
            .any(|x| !x.is_finite())
        || c.deviations.iter().any(|x| *x < 0.)
        || c.prior.iter().any(|x| *x < 0.)
        || (c.prior.iter().sum::<f64>() - 1.).abs() > 1e-10
        || c.cutpoints.windows(2).any(|w| w[0] >= w[1])
        || !c.temperature.is_finite()
        || c.temperature <= 0.
    {
        return Err(
            "finite scoring coefficients, scalar/mask layout, ordered cutpoints, prior and positive temperature required",
        );
    }
    Ok(())
}

pub(crate) fn validate(c: TemporalWholeConfig) -> Result<(), &'static str> {
    validate_head(head(c))?;
    if let Some(c) = c.controls {
        validate_head(c.closure_only)?;
        validate_head(c.gap_energy_2s)?;
        validate_head(c.elapsed_only)?;
    }
    Ok(())
}

fn evaluate<const N: usize, const C: usize>(
    c: TemporalScoringConfig<N, C>,
    values: [Option<f64>; N],
) -> Result<([f64; C], Option<[f64; 5]>), &'static str> {
    validate_head(c)?;
    if values.iter().flatten().any(|v| !v.is_finite()) {
        return Err("scoring inputs must be finite or missing");
    }
    let coordinates = std::array::from_fn(|i| {
        if i == 0 {
            1.
        } else if i <= N {
            values[i - 1].map_or(0., |x| (x - c.means[i - 1]) / c.deviations[i - 1].max(1e-6))
        } else {
            f64::from(values[i - N - 1].is_none())
        }
    });
    let predictor: f64 = coordinates
        .iter()
        .zip(c.coefficients)
        .map(|(x, w)| x * w)
        .sum();
    if !predictor.is_finite() {
        return Err("scoring predictor overflow");
    }
    let mut p = [0.; 5];
    let mut last = 0.;
    for (i, cut) in c.cutpoints.iter().enumerate() {
        let x = cut - predictor;
        let cdf = if x >= 0. {
            1. / (1. + (-x).exp())
        } else {
            let z = x.exp();
            z / (1. + z)
        };
        p[i] = (cdf - last).max(0.);
        last = cdf;
    }
    p[4] = 1. - last;
    Ok((coordinates, values.iter().any(Option::is_some).then_some(p)))
}

fn rating(p: Option<[f64; 5]>, support: f64, prior: [f64; 5], temperature: f64) -> Rating {
    let supported_categories = p.filter(|_| support > 0.);
    let categories = if let Some(p) = supported_categories {
        let logs = p.map(|p| p.ln() / temperature);
        let max = logs.into_iter().fold(f64::NEG_INFINITY, f64::max);
        let weights = logs.map(|v| (v - max).exp());
        let total: f64 = weights.iter().sum();
        std::array::from_fn(|i| support * weights[i] / total + (1. - support) * prior[i])
    } else {
        prior
    };
    Rating {
        supported_categories,
        categories,
        support,
        expected_rating: categories
            .iter()
            .enumerate()
            .map(|(i, p)| i as f64 * p / 4.)
            .sum(),
    }
}

pub(crate) fn score(
    c: TemporalWholeConfig,
    end: u64,
    values: [Option<f64>; 4],
    input_support: [f64; 4],
) -> Result<Snapshot, &'static str> {
    validate(c)?;
    if input_support
        .iter()
        .any(|x| !x.is_finite() || !(0. ..=1.).contains(x))
        || values.iter().zip(input_support).any(|(x, s)| match x {
            Some(x) => !x.is_finite() || !(0. ..=1.).contains(x) || s == 0.,
            None => s != 0.,
        })
    {
        return Err("whole-piece values require matching fractional support");
    }
    let (coordinates, p) = evaluate(head(c), values)?;
    let r = rating(
        p,
        input_support.iter().sum::<f64>() / 4.,
        c.prior,
        c.temperature,
    );
    Ok(Snapshot {
        observed_end_sample: end,
        graph: None,
        controls: None,
        values,
        input_support,
        coordinates: std::array::from_fn(|i| coordinates[i + 1]),
        supported_categories: r.supported_categories,
        categories: r.categories,
        support: r.support,
        expected_rating: r.expected_rating,
    })
}

pub(crate) fn observe(
    c: TemporalWholeConfig,
    p: &phrase::Snapshot,
    s: &section::Snapshot,
    memory: Option<recall::Snapshot>,
) -> Result<Snapshot, &'static str> {
    if p.end_sample != s.end_sample || p.censored || s.censored {
        return Err("whole-piece scoring requires the common pre-EOF observation snapshot");
    }
    let mut values = [None; 4];
    let mut support = [0.; 4];
    for (i, head) in [p.closure, p.continuation].into_iter().enumerate() {
        if let Some(v) = head.supported_expected_rating.filter(|_| head.unknown < 1.) {
            values[i] = Some(v);
            support[i] = 1. - head.unknown;
        }
    }
    // Unmatched expected support needs its own causal producer, not elapsed cue time.
    let mut exits = 0.;
    let mut returns = 0.;
    for g in p.groups.iter().flatten() {
        if let Some(section) = s.groups.iter().flatten().find(|s| s.group == g.group) {
            let total: f64 = section.exits.iter().sum();
            if total > 0. {
                exits += g.acoustic_weight * total;
                returns += g.acoustic_weight * section.exits[1];
                support[3] += g.acoustic_weight * (1. - section.unknown);
            }
        }
    }
    support[3] = (support[3] * p.closure.observed_coverage).clamp(0., 1.);
    if exits > 0. && support[3] > 0. {
        values[3] = Some((returns / exits).clamp(0., 1.));
    } else {
        support[3] = 0.;
    }
    let mut snapshot = score(c, p.end_sample, values, support)?;
    snapshot.controls = c.controls.map(|c| controls::observe(c, p)).transpose()?;
    snapshot.graph = memory.and_then(|m| m.graph);
    if snapshot.graph.is_some_and(|g| g.end_sample > p.end_sample) {
        return Err("whole-piece graph cannot use a future observation state");
    }
    Ok(snapshot)
}

#[cfg(test)]
pub(in crate::temporal_cognition) mod tests {
    use super::*;

    pub(in crate::temporal_cognition) fn config() -> TemporalWholeConfig {
        TemporalWholeConfig {
            means: [0.5; 4],
            deviations: [1.; 4],
            coefficients: [0.; 9],
            cutpoints: [-2., -1., 1., 2.],
            prior: [0.1, 0.2, 0.4, 0.2, 0.1],
            temperature: 1.,
            controls: Some(controls::tests::config()),
        }
    }

    #[test]
    fn missing_input_support_is_arithmetic_mean_and_prior_is_not_tempered() {
        let mut c = config();
        c.temperature = 2.;
        let s = score(
            c,
            100,
            [Some(0.2), None, Some(0.7), None],
            [1., 0., 0.5, 0.],
        )
        .unwrap();
        assert_eq!(s.support, 0.375);
        assert_eq!(s.coordinates, [-0.3, 0., 0.7 - 0.5, 0., 0., 1., 0., 1.]);
        let raw = [
            1. / (1. + 2_f64.exp()),
            1. / (1. + 1_f64.exp()) - 1. / (1. + 2_f64.exp()),
            1. / (1. + (-1_f64).exp()) - 1. / (1. + 1_f64.exp()),
            1. / (1. + (-2_f64).exp()) - 1. / (1. + (-1_f64).exp()),
            1. - 1. / (1. + (-2_f64).exp()),
        ];
        let total: f64 = raw.iter().map(|v| v.sqrt()).sum();
        for i in 0..5 {
            assert!(
                (s.categories[i] - (0.375 * raw[i].sqrt() / total + 0.625 * c.prior[i])).abs()
                    < 1e-14
            );
        }
        let unknown = score(c, 200, [None; 4], [0.; 4]).unwrap();
        assert_eq!(unknown.categories, c.prior);
        assert!(unknown.supported_categories.is_none());
        assert_eq!(unknown.support, 0.);
        assert!(score(c, 300, [Some(0.); 4], [0.; 4]).is_err());
    }

    #[test]
    fn scalar_and_indicator_coefficients_have_distinct_coordinates() {
        let mut c = config();
        c.coefficients = [0.3, 1., 2., 3., 4., -1., -2., -3., -4.];
        let a = score(c, 1, [Some(0.5); 4], [1.; 4]).unwrap();
        let b = score(
            c,
            2,
            [Some(0.5), None, Some(0.5), Some(0.5)],
            [1., 0., 1., 1.],
        )
        .unwrap();
        assert!(a.supported_categories.unwrap()[4] > b.supported_categories.unwrap()[4]);
        assert_eq!(b.support, 0.75);
        let mut invalid = c;
        invalid.cutpoints[2] = invalid.cutpoints[1];
        assert!(validate(invalid).is_err());
        invalid = c;
        invalid.temperature = 0.;
        assert!(validate(invalid).is_err());
    }
}
