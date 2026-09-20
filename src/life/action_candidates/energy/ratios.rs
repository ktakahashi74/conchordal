//! Direct frozen-body footprints; candidate evaluation never issues or learns a command.

use super::{Body, Record};
use crate::core::temporal_expectation::{ExternalEnergyFootprint, TemporalForecast};
use crate::life::self_prediction::ToneEnergy;

#[derive(Debug, serde::Serialize)]
pub(crate) struct Context {
    pub version: u8,
    pub bus: u8,
    pub own_model: &'static str,
    pub sample_mapping: &'static str,
    pub limitation: &'static str,
    pub common_horizon: [u64; 2],
    pub external_origin: Option<u64>,
    pub external_model: Option<&'static str>,
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct CandidateRatio {
    pub status: &'static str,
    pub body_support: Option<[u64; 2]>,
    pub external: Option<ExternalEnergyFootprint>,
    pub continuous_own_support: Option<bool>,
    pub own_points: [Option<f64>; 16],
    pub coherent_points: u8,
    pub overlap: Option<f64>,
    pub audibility: Option<f64>,
    pub default_overlap: Option<f64>,
    pub default_audibility: Option<f64>,
    pub overlap_difference: Option<f64>,
    pub audibility_difference: Option<f64>,
    pub own_integral_sample_units: Option<f64>,
    pub excluded_energy_mass: Option<f64>,
}

fn preview(
    body: &Body<'_>,
    horizon: [u64; 2],
    external: Option<&TemporalForecast>,
) -> CandidateRatio {
    let support = body.support_after(horizon[0], 0);
    let mut result = CandidateRatio {
        status: "external_unavailable",
        body_support: support,
        external: None,
        continuous_own_support: None,
        own_points: [None; 16],
        coherent_points: 0,
        overlap: None,
        audibility: None,
        default_overlap: None,
        default_audibility: None,
        overlap_difference: None,
        audibility_difference: None,
        own_integral_sample_units: None,
        excluded_energy_mass: None,
    };
    let Some(support) = support else {
        result.status = "known_silent_model";
        result.overlap = Some(0.);
        result.own_integral_sample_units = Some(0.);
        result.excluded_energy_mass = Some(0.);
        return result;
    };
    let Some(external) = external else {
        return result;
    };
    let footprint = external.external_footprint(support, horizon[1]);
    result.external = Some(footprint);
    result.status = "empty_intersection";
    let Some([start, end]) = footprint.horizon_intersection else {
        return result;
    };
    if [start, end] == support {
        result.excluded_energy_mass = Some(0.);
    }
    let known = body.known_on([start, end], 0);
    result.continuous_own_support = Some(known);
    let width = end - start;
    for (i, point) in footprint.points.iter().enumerate() {
        let point = point.expect("nonempty footprint has sixteen points");
        let [mut left, mut right] =
            [i, i + 1].map(|edge| start + (u128::from(width) * edge as u128).div_ceil(16) as u64);
        // The discrete renderer holds a sample within its fractional cell.
        if point.sample < left || point.sample >= right {
            left = point.sample;
            right = left + 1;
        }
        let (incoherent, coherent) = body.point(point.sample, left, right, 0);
        result.coherent_points += u8::from(coherent.is_some());
        result.own_points[i] = coherent
            .or(incoherent)
            .filter(|v| v.is_finite() && *v >= 0.);
    }
    result.status = "unsupported";
    if !known || footprint.continuous_support != Some(true) {
        return result;
    }
    let mut total = 0.;
    let mut overlap = 0.;
    let mut audibility = 0.;
    for (own, point) in result.own_points.into_iter().zip(footprint.points) {
        let Some((own, external)) = own.zip(point.and_then(|p| p.band_energy_sum)) else {
            return result;
        };
        let denominator = own + external + 1e-12;
        total += own;
        overlap += own * (external / denominator);
        audibility += own * (own / denominator);
    }
    result.own_integral_sample_units = Some(total * width as f64 / 16.);
    if total == 0. {
        result.status = "zero_prediction_not_silence_proof";
        return result;
    }
    result.status = "supported";
    result.overlap = Some((overlap / total).clamp(0., 1.));
    result.audibility = Some((audibility / total).clamp(0., 1.));
    result
}

pub(super) fn attach(
    record: &mut Record,
    retained: &[(u64, [bool; 2], ToneEnergy)],
    external: Option<&TemporalForecast>,
) {
    let width = 4 * u64::from(record.sample_rate);
    let limit = record
        .decision_at
        .saturating_add(width)
        .min(external.map_or(u64::MAX, |f| f.observed_frame().saturating_add(width)));
    let horizon = [record.decision_at, limit];
    record.energy_ratio_context = Some(Context {
        version: 2,
        bus: 0,
        own_model: "frozen_body_envelope_sine_bins_v3",
        sample_mapping: "16 uniform fractional midpoints; renderer envelope/control at the containing integer sample; coherent carrier mean on integer bin edges, containing-sample fallback for sub-sample bins",
        limitation: "Conditional fixed-body model, coherent sine bins with additive fallback. Excluded physical energy mass remains unknown when clipped. No acoustic calibration or generation effect. A sampled zero within possible support is not silence evidence.",
        common_horizon: horizon,
        external_origin: external.map(TemporalForecast::observed_frame),
        external_model: external.map(|f| f.energy_prediction_model),
    });
    let mut default = None;
    for candidate in &mut record.candidates {
        let body = Body {
            retained,
            added: candidate.added,
            at: candidate.input.at,
            intervention: (candidate.intervention, candidate.input.withhold_until),
        };
        let mut ratio = preview(&body, horizon, external);
        let (overlap, audibility) = *default.get_or_insert((ratio.overlap, ratio.audibility));
        ratio.default_overlap = overlap;
        ratio.default_audibility = audibility;
        ratio.overlap_difference = ratio.overlap.zip(overlap).map(|(a, b)| a - b);
        ratio.audibility_difference = ratio.audibility.zip(audibility).map(|(a, b)| a - b);
        candidate.energy_ratio = Some(ratio);
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::tone;
    use super::*;
    use crate::life::self_prediction::ScheduledRelease;
    use crate::life::sound::sine_forecast::SineForecast;

    #[test]
    fn delayed_short_body_uses_its_own_support_without_onset_interpolation() {
        let external = TemporalForecast::energy_fixture(8000, 0, |_| [0.375, 0., 0.]);
        for width in [1, 8, 17, 256] {
            let mut t = tone(37);
            t.envelope.hold_end = 37 + width;
            t.envelope.release_end = 37 + width;
            let body = Body {
                retained: &[],
                added: Some(([true, false], t)),
                at: 37,
                intervention: (None, None),
            };
            let r = preview(&body, [0, 32000], Some(&external));
            assert_eq!(r.status, "supported");
            assert_eq!(r.body_support, Some([37, 37 + width]));
            assert_eq!(r.own_points, [Some(0.125); 16]);
            assert_eq!(r.excluded_energy_mass, Some(0.));
            assert_eq!(r.own_integral_sample_units, Some(width as f64 * 0.125));
            assert!((r.overlap.unwrap() - 0.375 / (0.5 + 1e-12)).abs() < 1e-12);
            assert_eq!(body.point(36, 36, 37, 0).0, Some(0.));
            assert_eq!(
                body.point(37 + width, 37 + width, 38 + width, 0).0,
                Some(0.)
            );
        }
    }

    #[test]
    fn release_application_and_gap_keep_only_owned_support() {
        let mut t = tone(100);
        t.envelope.release_ticks = 20;
        t.scheduled_release = Some(ScheduledRelease {
            apply_at_sample: 250,
            off_sample: 180,
        });
        // A late command does not erase samples rendered before its application.
        assert_eq!(t.support_after(0, None), Some([100, 250]));
        assert_eq!(t.support_after(250, None), None);
        assert!(t.at(249, None).unwrap() > 0.);
        assert_eq!(t.at(250, None), Some(0.));
        let early = Some(ScheduledRelease {
            apply_at_sample: 150,
            off_sample: 150,
        });
        assert_eq!(t.support_after(0, early), Some([100, 170]));
        assert_eq!(t.support_after(170, early), None);
        t.scheduled_release = Some(ScheduledRelease {
            apply_at_sample: 50,
            off_sample: 50,
        });
        assert_eq!(t.support_after(0, None), None);

        let retained = [
            (1, [true, false], tone(100)),
            (2, [true, false], tone(200)),
            (3, [false, true], tone(0)),
        ];
        let mut body = Body {
            retained: &retained,
            added: None,
            at: 100,
            intervention: (None, Some(200)),
        };
        assert_eq!(body.support_after(0, 0), Some([200, 100000]));
        assert_eq!(body.point(199, 199, 200, 0).0, Some(0.));
        assert_eq!(body.point(200, 200, 201, 0).0, Some(0.125));
        body.intervention.1 = Some(201);
        let r = preview(&body, [0, 32000], None);
        assert_eq!(r.status, "known_silent_model");
        assert_eq!(r.overlap, Some(0.));
        assert_eq!(r.audibility, None);
        assert!(body.support_after(0, 1).is_some());
    }

    #[test]
    fn separated_and_unknown_tones_are_not_filled_between_sample_points() {
        let external = TemporalForecast::energy_fixture(8000, 0, |_| [0.; 3]);
        let mut a = tone(0);
        a.envelope.hold_end = 1;
        a.envelope.release_end = 1;
        let mut b = tone(999);
        b.envelope.hold_end = 1000;
        b.envelope.release_end = 1000;
        let retained = [(1, [true, false], a)];
        let mut body = Body {
            retained: &retained,
            added: Some(([true, false], b)),
            at: 999,
            intervention: (None, None),
        };
        let r = preview(&body, [0, 32000], Some(&external));
        assert_eq!(r.body_support, Some([0, 1000]));
        assert_eq!(r.own_points, [Some(0.); 16]);
        assert_eq!(r.continuous_own_support, Some(true));
        assert_eq!(r.status, "zero_prediction_not_silence_proof");
        assert_eq!(r.overlap, None);
        b.control = None;
        body.added = Some(([true, false], b));
        let r = preview(&body, [0, 32000], Some(&external));
        assert_eq!(r.own_points, [Some(0.); 16]);
        assert_eq!(r.continuous_own_support, Some(false));
        assert_eq!(r.status, "unsupported");
    }

    #[test]
    fn sine_bins_preserve_in_phase_and_cancelled_carriers() {
        let external = TemporalForecast::energy_fixture(8000, 0, |_| [0.375, 0., 0.]);
        let mut t = tone(0);
        t.envelope.hold_end = 256;
        t.envelope.release_end = 256;
        t.sine = Some(SineForecast {
            first_sample: 0,
            state: [1., 0.],
            rotation: [0., 1.],
            boost: 0.,
            boost_decay: 1.,
        });
        let retained = [(1, [true, false], t)];
        let mut body = Body {
            retained: &retained,
            added: Some(([true, false], t)),
            at: 0,
            intervention: (None, None),
        };
        let r = preview(&body, [0, 32000], Some(&external));
        assert_eq!(r.coherent_points, 16);
        for v in r.own_points {
            assert!((v.unwrap() - 0.5).abs() < 1e-12);
        }
        assert!((r.overlap.unwrap() - 0.375 / (0.875 + 1e-12)).abs() < 1e-12);
        t.sine.as_mut().unwrap().state = [-1., 0.];
        body.added = Some(([true, false], t));
        let r = preview(&body, [0, 32000], Some(&external));
        assert_eq!(r.coherent_points, 16);
        assert_eq!(r.status, "zero_prediction_not_silence_proof");
        assert_eq!(r.overlap, None);
    }

    #[test]
    fn release_quadrature_matches_closed_form_and_dense_sample_reference() {
        let external = TemporalForecast::energy_fixture(8000, 0, |_| [0.375, 0., 0.]);
        let mut t = tone(0);
        t.envelope.hold_end = 0;
        t.envelope.release_end = 512;
        t.envelope.release_ticks = 512;
        let body = Body {
            retained: &[],
            added: Some(([true, false], t)),
            at: 0,
            intervention: (None, None),
        };
        let dense = (0..512)
            .map(|tick| {
                let gain = (512 - tick) as f64 / 512.;
                let own = 0.125 * gain * gain;
                (own, own * 0.375 / (own + 0.375 + 1e-12))
            })
            .fold((0., 0.), |(a, b), (x, y)| (a + x, b + y));
        for count in [8_u64, 16, 32] {
            let mut sum = 0.;
            let mut overlap = 0.;
            for i in 0..count {
                let tick = 512 * (2 * i + 1) / (2 * count);
                let own = body.point(tick, tick, tick + 1, 0).0.unwrap();
                let expected = 0.125 * ((512 - tick) as f64 / 512.).powi(2);
                assert_eq!(own, expected);
                sum += own;
                overlap += own * 0.375 / (own + 0.375 + 1e-12);
            }
            let integral = sum * 512. / count as f64;
            // This smooth release case bounds quadrature only, not arbitrary rhythms.
            assert!((integral - dense.0).abs() / dense.0 < 0.01);
            assert!((overlap / sum - dense.1 / dense.0).abs() < 0.003);
            eprintln!(
                "direct release count={count}, integral={integral}, dense={}, overlap={}, dense_overlap={}",
                dense.0,
                overlap / sum,
                dense.1 / dense.0
            );
            if count == 16 {
                let r = preview(&body, [0, 32000], Some(&external));
                assert_eq!(r.own_integral_sample_units, Some(integral));
                assert!((r.overlap.unwrap() - overlap / sum).abs() < 1e-15);
            }
        }
    }
}
