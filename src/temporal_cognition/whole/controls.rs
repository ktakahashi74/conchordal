//! Independent whole-mixture controls with frozen coefficients and causal inputs.

use super::*;
use crate::config::TemporalWholeControlsConfig;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Scalar {
    pub value: Option<f64>,
    pub coordinates: [f64; 2],
    pub rating: Rating,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct GroupInput {
    pub group: super::super::ridge::Handle,
    pub input: phrase::RecentEnergy,
    pub coordinates: [f64; 6],
    pub acoustic_weight: f64,
    pub supported_path_mass: f64,
    pub categories: Option<[f64; 5]>,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct GapEnergy {
    pub groups: [Option<GroupInput>; 7],
    pub observed_coverage: f64,
    pub supported_mass: f64,
    pub rating: Rating,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Snapshot {
    pub closure_only: Scalar,
    pub gap_energy_2s: GapEnergy,
    pub elapsed_only: Scalar,
}

pub(super) fn observe(
    c: TemporalWholeControlsConfig,
    p: &phrase::Snapshot,
) -> Result<Snapshot, &'static str> {
    if p.censored || p.sample_rate == 0 || p.epoch_start_sample > p.end_sample {
        return Err("completion controls require a causal pre-EOF audio clock");
    }
    let value = p
        .closure
        .supported_expected_rating
        .filter(|_| p.closure.unknown < 1.);
    let (x, probabilities) = evaluate(c.closure_only, [value])?;
    let closure_only = Scalar {
        value,
        coordinates: [x[1], x[2]],
        rating: rating(
            probabilities,
            if value.is_some() {
                1. - p.closure.unknown
            } else {
                0.
            },
            c.closure_only.prior,
            c.closure_only.temperature,
        ),
    };
    let value = (p.end_sample - p.epoch_start_sample) as f64 / f64::from(p.sample_rate);
    let (x, probabilities) = evaluate(c.elapsed_only, [Some(value)])?;
    let elapsed_only = Scalar {
        value: Some(value),
        coordinates: [x[1], x[2]],
        rating: rating(
            probabilities,
            1.,
            c.elapsed_only.prior,
            c.elapsed_only.temperature,
        ),
    };
    let mut groups = [None; 7];
    let mut mixture = [0.; 5];
    let mut mass = 0.;
    for (slot, group) in p
        .groups
        .iter()
        .enumerate()
        .filter_map(|(i, g)| g.map(|g| (i, g)))
    {
        let Some(input) = group.recent_energy else {
            continue;
        };
        if input.end_sample != p.end_sample || input.start_sample < p.epoch_start_sample {
            return Err("completion control cannot use stale group energy");
        }
        let (x, categories) = evaluate(c.gap_energy_2s, input.values)?;
        // Features contain no phrase state; only prediction mixing uses retained path mass.
        let weight = group.acoustic_weight * (1. - group.unknown);
        if let Some(categories) = categories {
            mass += weight;
            for i in 0..5 {
                mixture[i] += weight * categories[i];
            }
        }
        groups[slot] = Some(GroupInput {
            group: group.group,
            input,
            coordinates: std::array::from_fn(|i| x[i + 1]),
            acoustic_weight: group.acoustic_weight,
            supported_path_mass: 1. - group.unknown,
            categories,
        });
    }
    let distribution = (mass > 0.).then(|| mixture.map(|x| x / mass));
    let gap_energy_2s = GapEnergy {
        groups,
        observed_coverage: p.closure.observed_coverage,
        supported_mass: mass,
        rating: rating(
            distribution,
            (mass * p.closure.observed_coverage).clamp(0., 1.),
            c.gap_energy_2s.prior,
            c.gap_energy_2s.temperature,
        ),
    };
    Ok(Snapshot {
        closure_only,
        gap_energy_2s,
        elapsed_only,
    })
}

#[cfg(test)]
pub(in crate::temporal_cognition) mod tests {
    use super::*;

    pub(in crate::temporal_cognition) fn config() -> TemporalWholeControlsConfig {
        let scalar = TemporalScoringConfig {
            means: [0.],
            deviations: [1.],
            coefficients: [0., 1., -1.],
            cutpoints: [-2., -1., 1., 2.],
            prior: [0.1, 0.2, 0.4, 0.2, 0.1],
            temperature: 2.,
        };
        TemporalWholeControlsConfig {
            closure_only: scalar,
            elapsed_only: scalar,
            gap_energy_2s: TemporalScoringConfig {
                means: [0.; 3],
                deviations: [1.; 3],
                coefficients: [0., 0., 3., 0., 0., 0., 0.],
                cutpoints: scalar.cutpoints,
                prior: scalar.prior,
                temperature: 2.,
            },
        }
    }

    #[test]
    fn separate_controls_mix_predictions_before_temperature_and_preserve_missing_mass() {
        let c = config();
        let mut model = phrase::Phrase::new(1, 0, 0, 1000, 100, phrase::tests::config()).unwrap();
        let mut gesture = phrase::tests::gesture_model();
        let a = phrase::tests::input(1, 0.);
        gesture.advance(&a, &phrase::tests::ridges(), 100).unwrap();
        model.advance(&a, &gesture, None, None, 100).unwrap();
        let mut p = model.snapshot();
        p.epoch_start_sample = 1000;
        p.end_sample = 3000;
        p.closure.supported_expected_rating = Some(0.25);
        p.closure.expected_rating = 0.99;
        p.closure.unknown = 0.4;
        p.closure.observed_coverage = 0.8;
        let mut group = p.groups[0].unwrap();
        group.unknown = 0.;
        group.acoustic_weight = 0.25;
        group.recent_energy = Some(phrase::RecentEnergy {
            start_sample: 1000,
            end_sample: 3000,
            values: [Some(0.), Some(0.), None],
            coverage: [1., 1., 0.],
        });
        p.groups[0] = Some(group);
        group.group.generation += 1;
        group.acoustic_weight = 0.5;
        group.recent_energy.as_mut().unwrap().values[1] = Some(2.);
        p.groups[1] = Some(group);
        let s = observe(c, &p).unwrap();
        assert_eq!(s.closure_only.value, Some(0.25));
        assert_eq!(s.closure_only.rating.support, 0.6);
        assert_eq!(s.elapsed_only.value, Some(2.));
        assert_eq!(s.gap_energy_2s.supported_mass, 0.75);
        assert!((s.gap_energy_2s.rating.support - 0.6).abs() < 1e-14);
        let a = evaluate(c.gap_energy_2s, [Some(0.), Some(0.), None])
            .unwrap()
            .1
            .unwrap();
        let b = evaluate(c.gap_energy_2s, [Some(0.), Some(2.), None])
            .unwrap()
            .1
            .unwrap();
        let mixed: [f64; 5] = std::array::from_fn(|i| (a[i] + 2. * b[i]) / 3.);
        for (x, y) in s
            .gap_energy_2s
            .rating
            .supported_categories
            .unwrap()
            .into_iter()
            .zip(mixed)
        {
            assert!((x - y).abs() < 1e-14);
        }
        let mean = evaluate(c.gap_energy_2s, [Some(0.), Some(4. / 3.), None])
            .unwrap()
            .1
            .unwrap();
        assert!((mean[4] - mixed[4]).abs() > 0.05);
        let normalizer = mixed.iter().map(|x| x.sqrt()).sum::<f64>();
        for (i, x) in s.gap_energy_2s.rating.categories.iter().enumerate() {
            assert!(
                (x - (0.6 * mixed[i].sqrt() / normalizer + 0.4 * c.gap_energy_2s.prior[i])).abs()
                    < 1e-14
            );
        }
        p.groups.fill(None);
        p.closure.supported_expected_rating = None;
        p.closure.unknown = 1.;
        let absent = observe(c, &p).unwrap();
        assert_eq!(
            absent.gap_energy_2s.rating.categories,
            c.gap_energy_2s.prior
        );
        assert_eq!(absent.closure_only.rating.categories, c.closure_only.prior);
        assert_eq!(absent.elapsed_only.value, Some(2.));
        p.censored = true;
        assert!(observe(c, &p).is_err());
    }

    #[test]
    fn control_parameters_round_trip_and_reject_wrong_shapes() {
        let mut c = super::super::tests::config();
        c.controls = Some(config());
        let text = toml::to_string(&c).unwrap();
        let restored: TemporalWholeConfig = toml::from_str(&text).unwrap();
        validate(restored).unwrap();
        assert_eq!(
            restored.controls.unwrap().gap_energy_2s.coefficients,
            c.controls.unwrap().gap_energy_2s.coefficients
        );
        let mut value = toml::Value::try_from(c).unwrap();
        value["controls"]["elapsed_only"]["coefficients"] = toml::Value::Array(vec![]);
        assert!(value.try_into::<TemporalWholeConfig>().is_err());
        c.controls.as_mut().unwrap().gap_energy_2s.temperature = 0.;
        assert!(validate(c).is_err());
    }
}
