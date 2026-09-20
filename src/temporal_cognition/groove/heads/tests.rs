use super::*;
use crate::config::{AppConfig, TemporalGrooveHeadConfig};

pub(super) fn config() -> TemporalGrooveConfig {
    let mut groove = TemporalGrooveHeadConfig {
        coefficients: [0.; 109],
        cutpoints: [-1., -0.2, 0.8, 1.7],
        prior: [0.4, 0.2, 0.15, 0.15, 0.1],
        temperature: 2.,
    };
    for (i, v) in [
        (0, 0.2),
        (1, 0.4),
        (15, 0.2),
        (85, 0.5),
        (104, 0.1),
        (107, -0.2),
        (108, 0.3),
    ] {
        groove.coefficients[i] = v;
    }
    let mut desire = TemporalGrooveHeadConfig {
        coefficients: [0.; 109],
        cutpoints: [-1.5, -0.4, 0.4, 1.2],
        prior: [0.05, 0.1, 0.3, 0.35, 0.2],
        temperature: 0.5,
    };
    for (i, v) in [
        (0, -0.3),
        (1, 0.1),
        (15, -0.2),
        (85, -0.4),
        (104, 0.2),
        (107, 0.8),
        (108, -0.1),
    ] {
        desire.coefficients[i] = v;
    }
    TemporalGrooveConfig {
        means: [0.; 54],
        deviations: [1.; 54],
        groove,
        desire,
    }
}

pub(super) fn summary(end: u64, assignment: f64, density: Option<f64>) -> Summary {
    let mut raw = [None; 54];
    raw[42] = density;
    Summary {
        end_sample: end,
        raw,
        density_coverage: [0.; 8],
        density_retention_supported: [true; 8],
        word_coverage: 0.,
        word_probabilities: None,
        grouping_coverage: 0.,
        observed_coverage: 0.3,
        assignment_sample_weight: assignment,
        learned_pair_mass: 0.,
        retained_pairs: 0,
        surprise_capacity_evicted: 0,
        surprise_capacity_supported: true,
        owned_bytes: 0,
    }
}

pub(super) fn handle(generation: u64) -> Handle {
    Handle {
        bus: 0,
        epoch: 7,
        generation,
    }
}

pub(super) fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "actual={a} expected={b}");
}

pub(super) fn probabilities(eta: f64, cuts: [f64; 4]) -> [f64; 5] {
    let cdf = cuts.map(|a| 1. / (1. + (eta - a).exp()));
    [
        cdf[0],
        cdf[1] - cdf[0],
        cdf[2] - cdf[1],
        cdf[3] - cdf[2],
        1. - cdf[3],
    ]
}

#[test]
fn normal_heads_match_probability_space_mixture_with_independent_parameters_and_unknown_mass() {
    let config = config();
    let mut heads = Heads::new(config, 1000, 100, 0).unwrap();
    for end in (100..=1000).step_by(100) {
        heads.advance(end, end <= 800);
    }
    let mut a = summary(1000, 2., Some(2.));
    a.raw[0] = Some(0.3);
    a.raw[53] = Some(0.7);
    let b = summary(1000, 1., Some(0.5));
    let unknown = summary(1000, 1., None);
    let mut input = [None; 7];
    input[0] = Some((handle(2), &a));
    input[1] = Some((handle(3), &b));
    input[2] = Some((handle(4), &unknown));
    let actual = heads.score(input);
    close(actual.groups[0].unwrap().acoustic_weight, 0.5);
    close(actual.groups[1].unwrap().acoustic_weight, 0.25);
    assert_eq!(actual.groups[2].unwrap().local.log_probabilities, [None; 2]);
    for (i, (head, rating, eta_a, eta_b)) in [
        (&config.groove, actual.groove.unwrap(), 1.28, 1.05),
        (&config.desire, actual.desire.unwrap(), -0.31, -0.6),
    ]
    .into_iter()
    .enumerate()
    {
        close(
            actual.groups[0].unwrap().local.predictors[i].unwrap(),
            eta_a,
        );
        close(
            actual.groups[1].unwrap().local.predictors[i].unwrap(),
            eta_b,
        );
        let a = probabilities(eta_a, head.cutpoints);
        let b = probabilities(eta_b, head.cutpoints);
        let powers: [f64; 5] =
            std::array::from_fn(|j| ((2. * a[j] + b[j]) / 3.).powf(1. / head.temperature));
        let total: f64 = powers.iter().sum();
        close(rating.observed_coverage, 0.8);
        close(rating.supported_mass, 0.75);
        close(rating.reported_support_mass, 0.6);
        let expected: [f64; 5] =
            std::array::from_fn(|j| 0.6 * powers[j] / total + 0.4 * head.prior[j]);
        for (value, expected) in rating.categories.into_iter().zip(expected) {
            close(value, expected);
        }
        close(
            rating.expected_rating,
            expected
                .iter()
                .enumerate()
                .map(|(j, p)| j as f64 * p / 4.)
                .sum(),
        );
    }
    assert_ne!(
        actual.groove.unwrap().categories,
        actual.desire.unwrap().categories
    );
}

#[test]
fn physical_union_epoch_clipping_and_missing_heads_back_off_without_fabricating_support() {
    let config = config();
    let mut heads = Heads::new(config, 1000, 1000, 100000).unwrap();
    assert_eq!(heads.score([None; 7]).window, [100000; 2]);
    close(heads.score([None; 7]).groove.unwrap().observed_coverage, 0.);
    heads.advance(101000, true);
    let no_groups = heads.score([None; 7]);
    close(no_groups.groove.unwrap().observed_coverage, 1.);
    close(no_groups.groove.unwrap().reported_support_mass, 0.);
    assert_eq!(
        no_groups.groove.unwrap().log_probabilities,
        config.groove.prior.map(f64::ln)
    );
    heads.advance(104000, true);
    heads.advance(105000, false);
    let raw = summary(105000, 1., None);
    let mut input = [None; 7];
    input[0] = Some((handle(2), &raw));
    let s = heads.score(input);
    close(s.desire.unwrap().observed_coverage, 0.4);
    close(s.desire.unwrap().supported_mass, 0.);
    assert_eq!(
        s.desire.unwrap().log_probabilities,
        config.desire.prior.map(f64::ln)
    );
    let owned = s.owned_bytes;
    for end in (106000..=130000).step_by(1000) {
        heads.advance(end, true);
    }
    let s = heads.score([None; 7]);
    assert_eq!(s.window, [122000, 130000]);
    close(s.groove.unwrap().observed_coverage, 1.);
    assert_eq!(s.owned_bytes, owned);
    assert!(heads.history.len() <= heads.capacity);
}

#[test]
fn one_head_overflow_is_explicit_and_does_not_disable_the_other_head() {
    let mut c = config();
    c.groove.coefficients[85] = f64::MAX;
    let mut heads = Heads::new(c, 1000, 100, 0).unwrap();
    heads.advance(100, true);
    let raw = summary(100, 1., Some(2.));
    let mut input = [None; 7];
    input[0] = Some((handle(2), &raw));
    let s = heads.score(input);
    assert_eq!(
        s.groups[0].unwrap().local.errors[0],
        Some("ordinal predictor overflow")
    );
    close(s.groove.unwrap().reported_support_mass, 0.);
    close(s.desire.unwrap().reported_support_mass, 1.);
    let mut c = config();
    c.groove.temperature = 1e-310;
    let mut heads = Heads::new(c, 1000, 100, 0).unwrap();
    heads.advance(100, true);
    let s = heads.score(input);
    assert!(s.groove.is_none() && s.errors[0].is_some());
    assert!(s.desire.is_some() && s.errors[1].is_none());
    let mut c = config();
    c.means[42] = -f64::MAX;
    let mut heads = Heads::new(c, 1000, 100, 0).unwrap();
    heads.advance(100, true);
    let raw = summary(100, 1., Some(f64::MAX));
    input[0] = Some((handle(2), &raw));
    let s = heads.score(input);
    assert_eq!(
        s.groups[0].unwrap().local.errors,
        [Some("groove feature standardization overflow"); 2]
    );
    close(s.desire.unwrap().supported_mass, 0.);
}

#[test]
fn head_configuration_is_explicit_strict_and_roundtrips_all_coordinates() {
    let config = config();
    validate(&config).unwrap();
    let app = AppConfig {
        temporal_groove: Some(config),
        ..AppConfig::default()
    };
    assert!(
        app.validate()
            .unwrap_err()
            .to_string()
            .contains("temporal_groove requires temporal_period")
    );
    let encoded = toml::to_string(&app).unwrap();
    let restored: AppConfig = toml::from_str(&encoded).unwrap();
    assert_eq!(
        serde_json::to_value(restored.temporal_groove).unwrap(),
        serde_json::to_value(config).unwrap()
    );
    assert!(AppConfig::default().temporal_groove.is_none());
    assert!(
        !toml::to_string(&AppConfig::default())
            .unwrap()
            .contains("temporal_groove")
    );
    let value = serde_json::to_value(config).unwrap();
    let mut extra = value.clone();
    extra["groove"]["typo"] = 1.into();
    assert!(serde_json::from_value::<TemporalGrooveConfig>(extra).is_err());
    let mut short = value;
    short["desire"]["coefficients"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(serde_json::from_value::<TemporalGrooveConfig>(short).is_err());
    for i in 0..8 {
        let mut bad = config;
        match i {
            0 => bad.means[0] = f64::NAN,
            1 => bad.deviations[2] = -1.,
            2 => bad.groove.coefficients[108] = f64::INFINITY,
            3 => bad.desire.cutpoints[1] = bad.desire.cutpoints[0],
            4 => bad.groove.prior[0] = 0.,
            5 => bad.desire.prior[0] += 0.01,
            6 => bad.groove.temperature = 0.,
            _ => bad.desire.temperature = f64::NAN,
        }
        assert!(validate(&bad).is_err(), "invalid case {i}");
    }
}
