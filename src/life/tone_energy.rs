//! Frozen tone energy and coherent integration for rendered bodies and onset footprints.

#[cfg(any(test, feature = "research"))]
#[path = "tone_energy/onset_window.rs"]
mod onset_window;
#[cfg(any(test, feature = "research"))]
pub(crate) use onset_window::CoherentWindow;

use crate::life::sound::{control_forecast::ControlForecast, envelope::Envelope};

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct ScheduledRelease {
    pub apply_at_sample: u64,
    pub off_sample: u64,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct ToneEnergy {
    pub amplitude: f32,
    pub envelope: Envelope,
    pub control: Option<ControlForecast>,
    pub scheduled_release: Option<ScheduledRelease>,
    pub sine: Option<crate::life::sound::sine_forecast::SineForecast>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bank: Option<crate::life::sound::bank_forecast::BankForecast>,
}

impl ToneEnergy {
    pub(crate) fn control_for(
        self,
        intervention: Option<ScheduledRelease>,
    ) -> Option<ControlForecast> {
        let mut control = self.control?;
        // Due updates run before the queue is cleared, even on a late release.
        let mut clear_at = self.envelope.hold_end.max(control.issued_at);
        for release in [self.scheduled_release, intervention].into_iter().flatten() {
            clear_at = clear_at.min(
                release
                    .apply_at_sample
                    .max(release.off_sample)
                    .max(control.issued_at),
            );
        }
        control.valid_until = control.valid_until.filter(|at| *at <= clear_at);
        if let Some(updates) = control.amplitude_updates.as_mut() {
            updates.len = updates.events[..updates.len]
                .iter()
                .take_while(|e| e.at_sample.max(control.issued_at) <= clear_at)
                .count();
        }
        Some(control)
    }

    fn envelope_at(self, tick: u64, intervention: Option<ScheduledRelease>) -> Envelope {
        let mut envelope = self.envelope;
        for release in [self.scheduled_release, intervention].into_iter().flatten() {
            if tick >= release.apply_at_sample {
                envelope = envelope.with_release(release.off_sample);
            }
        }
        envelope
    }

    #[cfg(test)]
    pub(crate) fn sine_point(
        self,
        tick: u64,
        intervention: Option<ScheduledRelease>,
    ) -> Option<([f64; 4], u64, u64)> {
        self.sine_point_from_energy(tick, intervention, self.at(tick, intervention))
    }

    pub(crate) fn at(self, tick: u64, intervention: Option<ScheduledRelease>) -> Option<f64> {
        let envelope = self.envelope_at(tick, intervention);
        let outer = f64::from(envelope.gain_at(tick));
        if outer == 0. {
            return Some(0.);
        }
        let control = self.control_for(intervention)?;
        let gain = control.gain_at(tick)?;
        let amplitude = control.amplitude_at(tick, self.amplitude)?;
        Some(amplitude.powi(2) * outer.powi(2) * gain.powi(2) / 2.)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::sound::control_forecast::AmplitudeModel;

    #[test]
    fn conditional_release_preserves_old_plan_and_causal_attack_clipping() {
        let tone = ToneEnergy {
            amplitude: 0.2,
            sine: None,
            bank: None,
            envelope: Envelope {
                onset: 100,
                hold_end: 1000,
                release_end: 1040,
                attack_ticks: 80,
                decay_ticks: 0,
                sustain_level: 1.,
                decay_lambda: 0.,
                release_ticks: 40,
                replacement_level: None,
            },
            control: Some(ControlForecast {
                issued_at: 96,
                valid_until: None,
                amplitude_smoothing: None,
                amplitude_updates: None,
                sample_dt: 1. / 8000.,
                starts_at: Some(100),
                kick_at: None,
                model: AmplitudeModel::Unmodulated { gain: 1. },
            }),
            scheduled_release: Some(ScheduledRelease {
                apply_at_sample: 128,
                off_sample: 144,
            }),
        };
        let added = ScheduledRelease {
            apply_at_sample: 192,
            off_sample: 200,
        };
        for tick in 96..256 {
            assert_eq!(tone.at(tick, Some(added)), tone.at(tick, None));
        }
        let early = ScheduledRelease {
            apply_at_sample: 112,
            off_sample: 120,
        };
        for tick in 100..112 {
            assert_eq!(tone.at(tick, Some(early)), tone.at(tick, None));
        }
        assert_ne!(tone.at(112, Some(early)), tone.at(112, None));
        assert!(tone.at(159, Some(early)).unwrap() > 0.);
        assert_eq!(tone.at(160, Some(early)), Some(0.));
        let swapped = ToneEnergy {
            scheduled_release: Some(early),
            ..tone
        };
        for tick in 96..256 {
            assert_eq!(
                tone.at(tick, Some(early)),
                swapped.at(tick, tone.scheduled_release)
            );
        }
        let unknown = ToneEnergy {
            control: None,
            ..tone
        };
        assert_eq!(unknown.at(100, None), None);
        assert_eq!(unknown.at(184, None), Some(0.));
    }
}

#[cfg(test)]
mod coherent_tests {
    use super::*;

    #[test]
    #[ignore = "explicit kernel cost probe, not complete hop-resource acceptance"]
    fn coherent_window_cost_probe() {
        for support in [
            "aligned",
            "staggered",
            "near_unison",
            "unison",
            "staggered_near_unison",
            "staggered_unison",
        ] {
            for count in [1, 4, 16, 64] {
                let mut nanos = Vec::new();
                for repetition in 0..257 {
                    let start = crate::clock::Instant::now();
                    for k in 0..32 {
                        let mut window = CoherentWindow::new();
                        for i in 0..count {
                            let angle = (i + k + repetition) as f64 * 0.11;
                            let omega = match support {
                                "near_unison" | "staggered_near_unison" => 0.02 + i as f64 * 1e-7,
                                "unison" | "staggered_unison" => 0.02,
                                _ => 0.02 + i as f64 * 0.003,
                            };
                            let (begin, end) = if support.starts_with("staggered") {
                                (1000 + i as u64 * 5, 2000 - i as u64 * 3)
                            } else {
                                (0, 100000)
                            };
                            window.add(Some((
                                [0.2 * angle.cos(), 0.2 * angle.sin(), omega, 0.],
                                begin,
                                end,
                            )));
                        }
                        std::hint::black_box(window.mean(1000, 2000, 1500));
                    }
                    nanos.push(start.elapsed().as_nanos());
                }
                nanos.sort_unstable();
                eprintln!(
                    "COHERENT_COST {{\"support\":\"{support}\",\"tones\":{count},\"windows\":32,\"samples\":257,\"median_ns\":{},\"p99_ns\":{},\"scratch_bytes\":{}}}",
                    nanos[128],
                    nanos[254],
                    32 * std::mem::size_of::<CoherentWindow>()
                        + 64 * 8 * std::mem::size_of::<f64>()
                );
            }
        }
    }

    #[test]
    fn coherent_edges_match_direct_samples_at_capacity_and_near_singular_frequencies() {
        let mut maximum = 0.0f64;
        let mut cases = 0;
        for width in [1, 7, 4096] {
            for count in [1, 4, 16, 64] {
                for mode in 0..7 {
                    let left = 1_000_000_000_000_000;
                    let right = left + width;
                    let mid = left + width / 2;
                    let mut window = CoherentWindow::new();
                    let mut direct = vec![0.; width as usize];
                    for i in 0..count {
                        let angle = i as f64 * 1.39;
                        let omega = match mode {
                            0 => 0.02 + i as f64 * 0.003,
                            1 => 0.2,
                            2 => 0.2 + i as f64 * 1e-12,
                            3 => i as f64 * 1e-12,
                            4 => std::f64::consts::PI - i as f64 * 1e-12,
                            6 => 0.2 + i as f64 * 1e-7,
                            _ => 0.2 + i as f64 * 0.000099,
                        };
                        let (s, c) = angle.sin_cos();
                        let (x, y) = (0.02 * c, 0.02 * s);
                        let start = left + i as u64 * (width / 128);
                        let end = right - (count - 1 - i) as u64 * (width / 128);
                        window.add(Some(([x, y, omega, 0.], start, end)));
                        for tick in start..end {
                            let (s, c) = (omega * (tick as i128 - mid as i128) as f64).sin_cos();
                            direct[(tick - left) as usize] += x * s + y * c;
                        }
                    }
                    let direct = direct.iter().map(|x| x * x).sum::<f64>() / width as f64;
                    let error = (window.mean(left, right, mid).unwrap() - direct).abs();
                    maximum = maximum.max(error);
                    cases += 1;
                    assert!(
                        error < 1e-12,
                        "width={width} count={count} mode={mode} error={error}"
                    );
                }
            }
        }
        eprintln!("COHERENT_EDGES cases={cases} max_error={maximum}");
    }

    #[test]
    fn coherent_window_matches_direct_wave_sums_with_phase_cancellation_and_partial_overlap() {
        for left in [0, 1_000_000_000_000_000] {
            for width in [1, 7, 60, 750, 12000] {
                let right = left + width;
                let mid = left + width / 2;
                for (frequencies, decays) in [
                    ([0.04, 0.04], [0., 0.]),
                    ([0.04, 0.04001], [0., 0.]),
                    ([0.04, 0.12], [0., 0.]),
                    ([0.04, 0.04], [-3e-4, -3e-4]),
                    ([0.04, 0.04000001], [-1e-9, 0.]),
                    ([0.04, 0.12], [-2e-3, -1e-7]),
                ] {
                    for shift in [0., 1., std::f64::consts::PI] {
                        let mut window = CoherentWindow::new();
                        let points = [
                            ([0.2, 0., frequencies[0], decays[0]], left, right),
                            (
                                [
                                    0.2 * shift.cos(),
                                    0.2 * shift.sin(),
                                    frequencies[1],
                                    decays[1],
                                ],
                                left + width / 3,
                                right,
                            ),
                        ];
                        for point in points {
                            window.add(Some(point));
                        }
                        let direct: f64 = (left..right)
                            .map(|tick| {
                                points
                                    .iter()
                                    .map(|&([x, y, omega, decay], start, end)| {
                                        if tick < start || tick >= end {
                                            return 0.;
                                        }
                                        let k = (tick as i128 - mid as i128) as f64;
                                        let (s, c) = (omega * k).sin_cos();
                                        (x * s + y * c) * (decay * k).exp()
                                    })
                                    .sum::<f64>()
                                    .powi(2)
                            })
                            .sum::<f64>()
                            / width as f64;
                        let error = (window.mean(left, right, mid).unwrap() - direct).abs();
                        assert!(
                            error < 1e-12 * direct.max(1.),
                            "width={width} decays={decays:?} error={error} direct={direct}"
                        );
                    }
                }
            }
        }
        let mut unsupported = CoherentWindow::new();
        unsupported.add(None);
        assert_eq!(unsupported.mean(0, 10, 5), None);
        let mut full = CoherentWindow::new();
        for _ in 0..64 {
            full.add(Some(([1., 0., 0.2, 0.], 0, 100)));
        }
        assert!(full.mean(0, 10, 5).is_some());
        full.add(Some(([1., 0., 0.2, 0.], 0, 100)));
        assert_eq!(full.mean(0, 10, 5), None);
    }
}
