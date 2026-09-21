//! Read-only ratios of issued source energy and the local own-excluded forecast.

use super::EnergyForecast;
use crate::core::temporal_expectation::{ExternalEnergyFootprint, TemporalForecast};

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct EnergyRatioPreview {
    pub(crate) version: u8,
    pub(crate) bus: u8,
    pub(crate) status: &'static str,
    pub(crate) requested: [u64; 2],
    pub(crate) external_origin: Option<u64>,
    pub(crate) external_model: Option<&'static str>,
    pub(crate) external: Option<ExternalEnergyFootprint>,
    pub(crate) own: [Option<f64>; 16],
    pub(crate) overlap: Option<f64>,
    pub(crate) audibility: Option<f64>,
    pub(crate) own_integral_sample_units: Option<f64>,
    pub(crate) modeled_own_integral_sample_units: Option<f64>,
    pub(crate) excluded_modeled_integral_sample_units: Option<f64>,
    pub(crate) excluded_modeled_mass_fraction: Option<f64>,
}

struct EnergyCurve<'a> {
    pub windows: &'a [[u64; 2]; 16],
    pub values: &'a [Option<f64>; 16],
}

impl EnergyForecast {
    pub(crate) fn preview_ratio(
        &self,
        external: Option<&TemporalForecast>,
        sample_rate: u32,
    ) -> EnergyRatioPreview {
        EnergyCurve {
            windows: &self.windows,
            values: &self.predictions[1],
        }
        .preview_ratio(external, sample_rate)
    }
}

impl EnergyCurve<'_> {
    fn value_at(&self, sample: u64, fraction: f64) -> Option<f64> {
        if !fraction.is_finite() || !(0. ..1.).contains(&fraction) {
            return None;
        }
        let index = self.windows.partition_point(|w| w[1] <= sample);
        let [left, right] = *self.windows.get(index)?;
        if sample < left {
            return None;
        }
        let value = self.values[index].filter(|v| v.is_finite() && *v >= 0.)?;
        let within = (sample - left) as f64 + fraction;
        let half = (right - left) as f64 / 2.;
        let neighbor = if within < half {
            (0..index)
                .rev()
                .find(|&i| self.windows[i][0] < self.windows[i][1])
        } else {
            (index + 1..self.windows.len()).find(|&i| self.windows[i][0] < self.windows[i][1])
        };
        if let Some(i) = neighbor {
            let [a, b] = self.windows[i];
            if (b == left || a == right)
                && let Some(other) = self.values[i].filter(|v| v.is_finite() && *v >= 0.)
            {
                let distance = half + (b - a) as f64 / 2.;
                let mix = (within - half).abs() / distance;
                return Some(value * (1. - mix) + other * mix);
            }
        }
        Some(value)
    }

    fn integral(&self, interval: [u64; 2]) -> Option<f64> {
        let [start, end] = interval;
        if end < start || start < self.windows[0][0] || end > self.windows[15][1] {
            return None;
        }
        let mut sum = 0.;
        for &[a, b] in self.windows {
            let edges = [
                2 * u128::from(a),
                u128::from(a) + u128::from(b),
                2 * u128::from(b),
            ];
            for pair in edges.windows(2) {
                let left = pair[0].max(2 * u128::from(start));
                let right = pair[1].min(2 * u128::from(end));
                if right > left {
                    let midpoint = left + right;
                    let value = self.value_at((midpoint / 4) as u64, (midpoint % 4) as f64 / 4.)?;
                    sum += value * (right - left) as f64 / 2.;
                }
            }
        }
        Some(sum)
    }

    pub(crate) fn preview_ratio(
        &self,
        external: Option<&TemporalForecast>,
        sample_rate: u32,
    ) -> EnergyRatioPreview {
        let requested = [self.windows[0][0], self.windows[15][1]];
        let mut result = EnergyRatioPreview {
            version: 1,
            bus: 0,
            status: "external_unavailable",
            requested,
            external_origin: external.map(TemporalForecast::observed_frame),
            external_model: external.map(|f| f.energy_prediction_model),
            external: None,
            own: [None; 16],
            overlap: None,
            audibility: None,
            own_integral_sample_units: None,
            modeled_own_integral_sample_units: self.integral(requested),
            excluded_modeled_integral_sample_units: None,
            excluded_modeled_mass_fraction: None,
        };
        let Some(external) = external else {
            return result;
        };
        let footprint = external.external_footprint(
            requested,
            external
                .observed_frame()
                .saturating_add(4 * u64::from(sample_rate)),
        );
        result.external = Some(footprint);
        result.status = "empty_intersection";
        let Some(interval) = footprint.horizon_intersection else {
            result.excluded_modeled_integral_sample_units =
                result.modeled_own_integral_sample_units;
            result.excluded_modeled_mass_fraction = result
                .modeled_own_integral_sample_units
                .filter(|v| *v > 0.)
                .map(|_| 1.);
            return result;
        };
        let covered = self.integral(interval);
        if let Some((full, covered)) = result.modeled_own_integral_sample_units.zip(covered) {
            result.excluded_modeled_integral_sample_units = Some((full - covered).max(0.));
            result.excluded_modeled_mass_fraction =
                (full > 0.).then(|| (1. - covered / full).clamp(0., 1.));
        }
        result.own = footprint
            .points
            .map(|point| point.and_then(|p| self.value_at(p.sample, p.sample_fraction)));
        result.status = "unsupported";
        if footprint.continuous_support != Some(true) || covered.is_none() {
            return result;
        }
        let mut own_sum = 0.;
        let mut overlap = 0.;
        let mut audibility = 0.;
        for (own, external) in result.own.into_iter().zip(footprint.points) {
            let Some((own, external)) = own.zip(external.and_then(|p| p.band_energy_sum)) else {
                return result;
            };
            let denominator = own + external + 1e-12;
            own_sum += own;
            overlap += own * (external / denominator);
            audibility += own * (own / denominator);
        }
        result.own_integral_sample_units = Some(own_sum * (interval[1] - interval[0]) as f64 / 16.);
        if own_sum == 0. {
            result.status = if covered.unwrap() > 0. {
                "missed_own_energy"
            } else {
                "zero_prediction_not_silence_proof"
            };
            return result;
        }
        result.overlap = Some((overlap / own_sum).clamp(0., 1.));
        result.audibility = Some((audibility / own_sum).clamp(0., 1.));
        result.status = "supported";
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(source: &EnergyForecast) -> EnergyCurve<'_> {
        EnergyCurve {
            windows: &source.windows,
            values: &source.predictions[1],
        }
    }

    fn own(start: u64, width: u64, value: f64) -> EnergyForecast {
        let input = crate::life::self_prediction::Input {
            scheduled_release: None,
            control: Some(crate::life::sound::control_forecast::ControlForecast {
                issued_at: 0,
                valid_until: None,
                amplitude_smoothing: None,
                amplitude_updates: None,
                sample_dt: 1. / 8000.,
                starts_at: Some(0),
                kick_at: None,
                model: crate::life::sound::control_forecast::AmplitudeModel::Unmodulated {
                    gain: 1.,
                },
            }),
            retained_energy: [Default::default(); 2],
            coherent_energy: [[None; 16]; 2],
            sine: None,
            bank: None,
            body_generation: 1,
            descriptors: [[None; 6]; 2],
            descriptor_support: [None; 2],
            frequency_hz: 220.,
            amplitude: 0.2,
            envelope: crate::life::sound::envelope::Envelope {
                onset: start,
                hold_end: start + width,
                release_end: start + width,
                attack_ticks: 0,
                decay_ticks: 0,
                sustain_level: 1.,
                decay_lambda: 0.,
                release_ticks: 0,
            },
            descriptor_target_end: start + width,
            descriptor_slot: 0,
        };
        let mut own = EnergyForecast::issue(
            &input,
            [start, start + width],
            true,
            0,
            [0.; 18],
            &[[[0.; 19]; 16]; 2],
            0,
        );
        for (window, prediction) in own.windows.iter().zip(&mut own.predictions[1]) {
            *prediction = (window[0] < window[1]).then_some(value);
        }
        own
    }

    #[test]
    fn energy_ratio_constant_curves_clip_mass_and_preserve_large_absolute_clocks() {
        for start in [0, (1_u64 << 60) + 7] {
            let source = own(start, 1600, 2.);
            let external = TemporalForecast::energy_fixture(1000, start, |_| [1., 2., 3.]);
            let ratio = source.preview_ratio(Some(&external), 1000);
            assert_eq!(ratio.status, "supported");
            assert!((ratio.overlap.unwrap() - 6. / (8. + 1e-12)).abs() < 1e-14);
            assert!((ratio.audibility.unwrap() - 2. / (8. + 1e-12)).abs() < 1e-14);
            assert_eq!(ratio.own_integral_sample_units, Some(3200.));
            assert_eq!(ratio.modeled_own_integral_sample_units, Some(3200.));
            assert_eq!(ratio.excluded_modeled_mass_fraction, Some(0.));
            let clipped = TemporalForecast::energy_fixture(1000, start + 800, |_| [0.; 3]);
            let ratio = source.preview_ratio(Some(&clipped), 1000);
            assert_eq!(ratio.overlap, Some(0.));
            assert_eq!(ratio.own_integral_sample_units, Some(1600.));
            assert_eq!(ratio.excluded_modeled_integral_sample_units, Some(1600.));
            assert_eq!(ratio.excluded_modeled_mass_fraction, Some(0.5));
            let late = TemporalForecast::energy_fixture(1000, start + 1600, |_| [0.; 3]);
            let ratio = source.preview_ratio(Some(&late), 1000);
            assert_eq!(ratio.status, "empty_intersection");
            assert_eq!(ratio.excluded_modeled_mass_fraction, Some(1.));
            assert_eq!(ratio.overlap, None);
            let long = own(start, 8000, 2.).preview_ratio(Some(&external), 1000);
            assert_eq!(
                long.external.unwrap().horizon_intersection,
                Some([start, start + 4000])
            );
            assert_eq!(long.excluded_modeled_mass_fraction, Some(0.5));
        }
    }

    #[test]
    fn energy_ratio_piecewise_integral_and_centers_handle_odd_and_empty_windows() {
        let mut source = own(7, 48, 0.);
        source.predictions[1] = std::array::from_fn(|k| Some((3 * k) as f64 + 1.5));
        assert_eq!(curve(&source).value_at(7, 0.), Some(1.5));
        assert!((curve(&source).value_at(10, 0.25).unwrap() - 3.25).abs() < 1e-12);
        assert_eq!(curve(&source).value_at(54, 0.75), Some(46.5));
        assert!((curve(&source).integral([7, 55]).unwrap() - 1152.).abs() < 1e-12);
        assert!((curve(&source).integral([10, 20]).unwrap() - 80.).abs() < 1e-12);
        assert_eq!(curve(&source).value_at(55, 0.), None);
        assert_eq!(curve(&source).value_at(6, 0.99), None);
        for fraction in [-0.1, 1., f64::NAN] {
            assert_eq!(curve(&source).value_at(10, fraction), None);
        }
        let short = own(7, 3, 2.);
        assert_eq!(curve(&short).integral([7, 10]), Some(6.));
        for tick in 7..10 {
            assert_eq!(curve(&short).value_at(tick, 0.5), Some(2.));
        }
        assert_eq!(curve(&own(7, 0, 2.)).integral([7, 7]), Some(0.));
    }

    #[test]
    fn energy_ratio_checks_unsampled_gaps_and_never_certifies_predicted_silence() {
        let external = TemporalForecast::energy_fixture(1000, 0, |_| [0.; 3]);
        let zero = own(0, 4000, 0.).preview_ratio(Some(&external), 1000);
        assert_eq!(zero.status, "zero_prediction_not_silence_proof");
        assert_eq!(zero.overlap, None);
        assert_eq!(zero.audibility, None);
        let mut source = own(0, 4000, 1.);
        let broken = TemporalForecast::energy_fixture(1000, 0, |t| {
            if (t - 0.02).abs() < 1e-8 {
                [f32::NAN; 3]
            } else {
                [0.; 3]
            }
        });
        let ratio = source.preview_ratio(Some(&broken), 1000);
        assert!(
            ratio
                .external
                .unwrap()
                .points
                .iter()
                .flatten()
                .all(|p| p.band_energy_sum.is_some())
        );
        assert_eq!(ratio.status, "unsupported");
        assert_eq!(ratio.overlap, None);
        // A short unknown source window lies between all quadrature points.
        source.windows[0] = [0, 1];
        source.windows[1] = [1, 500];
        source.predictions[1][0] = None;
        let ratio = source.preview_ratio(Some(&external), 1000);
        assert!(ratio.own.iter().all(Option::is_some));
        assert_eq!(ratio.modeled_own_integral_sample_units, None);
        assert_eq!(ratio.status, "unsupported");
        assert_eq!(ratio.overlap, None);
        assert_eq!(
            source.preview_ratio(None, 1000).status,
            "external_unavailable"
        );
        source.windows[1] = [1, 2];
        source.windows[2] = [2, 750];
        source.predictions[1].fill(Some(0.));
        source.predictions[1][0] = Some(1.);
        let missed = source.preview_ratio(Some(&external), 1000);
        assert!(missed.modeled_own_integral_sample_units.unwrap() > 0.);
        assert_eq!(missed.own_integral_sample_units, Some(0.));
        assert_eq!(missed.status, "missed_own_energy");
        assert_eq!(missed.overlap, None);
    }
}
