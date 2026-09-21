//! Read-only stationary sine carrier state; amplitude envelopes stay separate.

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct SineForecast {
    pub first_sample: u64,
    pub state: [f32; 2],
    pub rotation: [f32; 2],
    pub boost: f32,
    pub boost_decay: f32,
}

impl SineForecast {
    pub(crate) fn for_new_onset(mut self, first_sample: u64, mut seed: u64) -> Self {
        self.first_sample = first_sample;
        // An out-of-band carrier stays silent when its onset moves.
        if self.state != [0.; 2] {
            self.state = super::oscillator_bank::seeded_phase_state(&mut seed);
        }
        self
    }

    pub(crate) fn at(self, tick: u64) -> Option<[f64; 3]> {
        let elapsed = tick.checked_sub(self.first_sample)?;
        let [x, y] = self.state.map(f64::from);
        let [c, s] = self.rotation.map(f64::from);
        let omega = s.atan2(c);
        let steps = elapsed as f64 + 1.;
        let (sin, cos) = (omega * steps).sin_cos();
        let gain = c.hypot(s).powf(steps)
            * (1. + f64::from(self.boost) * f64::from(self.boost_decay).powf(elapsed as f64));
        let result = [
            (x * cos - y * sin) * gain,
            (x * sin + y * cos) * gain,
            omega,
        ];
        result.iter().all(|x| x.is_finite()).then_some(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::{modulation::NeuralRhythms, timebase::Timebase};
    use crate::life::phonation_engine::{OnsetKick, ToneUpdate};
    use crate::life::self_prediction::ToneEnergy;
    use crate::life::sound::{RenderModulatorSpec, Tone, ToneAdsr};

    #[test]
    fn amplitude_smoothing_forecast_tracks_the_actual_tone_waveform() {
        let mut maximum = 0_f64;
        for fs in [8000., 48000.] {
            for tau in [0.001, 0.05, 1.] {
                for target in [0., 0.05, 0.4] {
                    let mut tone = Tone::from_parts(
                        Timebase { fs, hop: 64 },
                        0,
                        5 * fs as u64,
                        293.,
                        0.2,
                        None,
                        Some(RenderModulatorSpec::SeqGate { duration_sec: 10. }),
                        Some(ToneAdsr {
                            attack_sec: 0.,
                            decay_sec: 0.,
                            sustain_level: 1.,
                            release_sec: 0.1,
                        }),
                    )
                    .unwrap();
                    tone.seed_modal_phases(7731);
                    tone.schedule_planned_kick(OnsetKick { strength: 1. });
                    tone.arm_onset_trigger(1.);
                    let rhythms = NeuralRhythms::default();
                    for tick in 0..512 {
                        tone.kick_planned_if_due(tick);
                        tone.render_tick(tick, fs, 1. / fs, &rhythms);
                    }
                    tone.set_smoothing_tau_sec(tau);
                    tone.schedule_update(
                        512,
                        ToneUpdate {
                            target_freq_hz: None,
                            target_amp: Some(target),
                            continuous_drive: None,
                        },
                    );
                    tone.apply_updates_if_due(512);
                    let (_, amplitude, envelope) = tone.prediction_parameters(None);
                    let frozen = ToneEnergy {
                        amplitude,
                        envelope,
                        control: Some(tone.prediction_control(512, &rhythms)),
                        sine: tone.prediction_sine(512),
                        bank: tone.prediction_bank(512),
                        scheduled_release: None,
                    };
                    assert!(frozen.control.unwrap().amplitude_smoothing.is_some());
                    assert!(frozen.sine.is_some());
                    let mut untouched = tone.clone();
                    for tick in 512..512 + 4 * fs as u64 {
                        let actual = tone.render_tick(tick, fs, 1. / fs, &rhythms);
                        assert_eq!(actual, untouched.render_tick(tick, fs, 1. / fs, &rhythms));
                        if tick % 512 == 0 {
                            let predicted = frozen.sine_point(tick, None).unwrap().0[1];
                            let error = (predicted - f64::from(actual)).abs();
                            maximum = maximum.max(error);
                            assert!(
                                error < 0.002,
                                "fs={fs}, tau={tau}, target={target}, tick={tick}, error={error}"
                            );
                        }
                    }
                }
            }
        }
        eprintln!("smoothed actual sine maximum waveform error={maximum}");
    }

    #[test]
    fn sine_forecast_tracks_actual_seeded_render_state_and_boost_without_advancing_it() {
        let mut maximum = 0.0f64;
        let mut comparisons = 0;
        for fs in [8000., 48000.] {
            for prefix in [0, 512, 8192] {
                for frequency in [293., 911.] {
                    let time = Timebase { fs, hop: 64 };
                    let mut tone = Tone::from_parts(
                        time,
                        37,
                        200000,
                        frequency,
                        0.2,
                        None,
                        Some(RenderModulatorSpec::SeqGate { duration_sec: 10. }),
                        Some(ToneAdsr {
                            attack_sec: 0.001,
                            decay_sec: 0.,
                            sustain_level: 1.,
                            release_sec: 0.1,
                        }),
                    )
                    .unwrap();
                    tone.seed_modal_phases(7731);
                    tone.schedule_planned_kick(OnsetKick { strength: 0.85 });
                    tone.arm_onset_trigger(0.85);
                    let rhythms = NeuralRhythms::default();
                    for tick in 0..prefix {
                        tone.kick_planned_if_due(tick);
                        tone.render_tick(tick, fs, 1. / fs, &rhythms);
                    }
                    let saved = tone.prediction_sine(prefix).unwrap();
                    let (_, amplitude, envelope) = tone.prediction_parameters(None);
                    let model = ToneEnergy {
                        amplitude,
                        envelope,
                        control: Some(tone.prediction_control(prefix, &rhythms)),
                        sine: Some(saved),
                        bank: None,
                        scheduled_release: None,
                    };
                    for offset in 0..=48000 {
                        let tick = prefix + offset;
                        tone.kick_planned_if_due(tick);
                        let actual = tone.render_tick(tick, fs, 1. / fs, &rhythms);
                        if [0, 36, 37, 64, 1024, 8192, 48000].contains(&offset) {
                            let projected = model.sine_point(tick, None).unwrap().0[1];
                            maximum = maximum.max((projected - f64::from(actual)).abs());
                            assert!(
                                (projected - f64::from(actual)).abs() < 0.002,
                                "fs={fs} prefix={prefix} frequency={frequency} tick={tick} predicted={projected} actual={actual}"
                            );
                            comparisons += 1;
                        }
                    }
                    assert_eq!(model.sine, Some(saved));
                    tone.schedule_update(
                        100000,
                        ToneUpdate {
                            target_freq_hz: Some(440.),
                            target_amp: None,
                            continuous_drive: None,
                        },
                    );
                    assert!(tone.prediction_sine(prefix + 48001).is_none());
                }
            }
        }
        eprintln!("sine waveform comparisons={comparisons}, maximum absolute error={maximum}");
    }
}
