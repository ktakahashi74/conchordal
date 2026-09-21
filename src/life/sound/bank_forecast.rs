//! Read-only carrier state of the multi-lane backends; amplitude envelopes stay separate.

use super::oscillator_bank::seeded_phase_state;
use crate::synth::resonator::{input_phase_seed, seeded_input_coupling};

pub(crate) const LANES: usize = 16;

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct Lane {
    /// State before the first forecast sample is rendered.
    pub state: [f32; 2],
    /// Oscillator: cosine and sine of the phase step. Resonator: `r` and `e`.
    pub step: [f32; 2],
    pub gain: f32,
    /// Oscillator: gain held until the next refresh. Resonator: input coupling magnitude.
    pub held: f32,
    /// Oscillator: spectral damping exponent.
    pub damping: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub enum Response {
    /// Envelopes are the values rendered at `first_sample`.
    Oscillator {
        drive_env: f32,
        drive_decay: f32,
        spectral_env: f32,
        spectral_decay: f32,
        spectral_floor: f32,
        refresh_at: u64,
        refresh_period: u64,
    },
    Resonator {
        /// Onset impulse of a tone that has not rendered yet.
        impulse: Option<f32>,
        seeded: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct BankForecast {
    pub first_sample: u64,
    pub response: Response,
    pub len: usize,
    pub lanes: [Lane; LANES],
}

impl BankForecast {
    /// A delayed onset draws the phases the renderer would draw for that onset.
    pub(crate) fn for_new_onset(mut self, first_sample: u64, mut seed: u64) -> Self {
        self.first_sample = first_sample;
        match &mut self.response {
            Response::Oscillator { refresh_at, .. } => {
                *refresh_at = first_sample;
                for lane in &mut self.lanes[..self.len] {
                    lane.state = seeded_phase_state(&mut seed);
                }
            }
            Response::Resonator { impulse, seeded } => {
                if let Some(impulse) = *impulse
                    && *seeded
                {
                    let mut state = input_phase_seed(seed, self.len);
                    for lane in &mut self.lanes[..self.len] {
                        if let Some(coupling) = seeded_input_coupling(&mut state, lane.held) {
                            lane.state = struck_state(lane.step, coupling, impulse);
                        }
                    }
                }
            }
        }
        self
    }

    /// Slowly varying gain of one lane relative to `lane_at` at `reference`: the damped
    /// partial gain and the excitation gain of an oscillator. A resonator's decay stays in
    /// the carrier, so its ratio is one.
    pub(crate) fn gain_ratio(&self, lane: usize, tick: u64, reference: u64) -> Option<f64> {
        if matches!(self.response, Response::Resonator { .. }) {
            return Some(1.);
        }
        let magnitude = |at| {
            self.lane_at(lane, at)
                .map(|[x, y, ..]: [f64; 4]| x.hypot(y))
        };
        let base = magnitude(reference)?;
        (base > 0.).then_some(magnitude(tick)? / base)
    }

    /// `[x, y, omega, log_decay]` of one lane, where `y` is its unit-envelope output.
    pub(crate) fn lane_at(&self, lane: usize, tick: u64) -> Option<[f64; 4]> {
        // A span that contains the onset is referenced before it; gains stay at the onset.
        let steps = (i128::from(tick) - i128::from(self.first_sample)) as f64 + 1.;
        let tick = tick.max(self.first_sample);
        let lane = self.lanes[..self.len].get(lane)?;
        let [x, y] = lane.state.map(f64::from);
        let result = match self.response {
            Response::Oscillator {
                drive_env,
                drive_decay,
                spectral_env,
                spectral_decay,
                spectral_floor,
                refresh_at,
                refresh_period,
            } => {
                let [c, s] = lane.step.map(f64::from);
                let omega = s.atan2(c);
                let mask = if tick < refresh_at {
                    f64::from(lane.held)
                } else {
                    // The renderer holds each damped gain until its next pitch refresh.
                    let refreshed = tick - (tick - refresh_at) % refresh_period.max(1);
                    let env = f64::from(spectral_env)
                        * f64::from(spectral_decay).powf((refreshed - self.first_sample) as f64);
                    let floor = f64::from(spectral_floor);
                    let energy = (floor + (1. - floor) * env).clamp(floor, 1.);
                    f64::from(lane.gain) * energy.powf(f64::from(lane.damping))
                };
                let drive = f64::from(drive_env)
                    * f64::from(drive_decay).powf((tick - self.first_sample) as f64);
                let gain = c.hypot(s).powf(steps) * mask * (1. + 0.25 * drive);
                let (sin, cos) = (omega * steps).sin_cos();
                [
                    (x * cos - y * sin) * gain,
                    (x * sin + y * cos) * gain,
                    omega,
                    0.,
                ]
            }
            Response::Resonator { .. } => {
                let [r, e] = lane.step.map(f64::from);
                // The magic-circle update is `r` times a unit-determinant rotation.
                let cosine = 1. - r * e * e / 2.;
                let sine = (1. - cosine * cosine).sqrt();
                if r <= 0. || sine <= 0. {
                    return None;
                }
                let omega = sine.atan2(cosine);
                let quadrature = (r * e * x - r * e * e / 2. * y) / sine;
                let (sin, cos) = (omega * steps).sin_cos();
                let gain = r.powf(steps) * f64::from(lane.gain);
                [
                    (quadrature * cos - y * sin) * gain,
                    (y * cos + quadrature * sin) * gain,
                    omega,
                    r.ln(),
                ]
            }
        };
        result.iter().all(|v| v.is_finite()).then_some(result)
    }
}

/// State one step before an impulse strikes a resting resonator.
pub(crate) fn struck_state([r, e]: [f32; 2], [b1, b2]: [f32; 2], impulse: f32) -> [f32; 2] {
    let (r, e) = (f64::from(r), f64::from(e));
    let x1 = f64::from(b1) * f64::from(impulse);
    let y1 = r * e * x1 + f64::from(b2) * f64::from(impulse);
    // Inverse of the update `x1 = r(x0 - e y0)`, `y1 = r(e x1 + y0)`.
    let y0 = y1 / r - e * x1;
    let x0 = x1 / r + e * y0;
    [x0 as f32, y0 as f32]
}

#[cfg(test)]
mod tests {
    use crate::core::{modulation::NeuralRhythms, timebase::Timebase};
    use crate::life::phonation_engine::OnsetKick;
    use crate::life::self_prediction::ToneEnergy;
    use crate::life::sound::{BodyKind, BodySnapshot, RenderModulatorSpec, Tone, ToneAdsr};

    fn tone(kind: BodyKind, onset: u64, frequency: f32, seed: u64) -> Tone {
        let mut tone = Tone::from_parts(
            Timebase {
                fs: 48000.,
                hop: 64,
            },
            onset,
            200000,
            frequency,
            0.2,
            Some(BodySnapshot {
                kind,
                amp_scale: 1.,
                brightness: 0.6,
                inharmonic: 0.,
                spread: 0.,
                unison: 1,
                motion: 0.,
                ratios: matches!(kind, BodyKind::Modal)
                    .then(|| (1..=16).map(|k| k as f32).collect::<Vec<_>>().into()),
            }),
            Some(RenderModulatorSpec::SeqGate { duration_sec: 10. }),
            Some(ToneAdsr {
                attack_sec: 0.001,
                decay_sec: 0.,
                sustain_level: 1.,
                release_sec: 0.1,
            }),
        )
        .unwrap();
        tone.seed_modal_phases(seed);
        tone.schedule_planned_kick(OnsetKick { strength: 0.85 });
        tone.arm_onset_trigger(0.85);
        tone
    }

    fn model(tone: &Tone, now: u64, rhythms: &NeuralRhythms) -> ToneEnergy {
        let (_, amplitude, envelope) = tone.prediction_parameters(None);
        ToneEnergy {
            amplitude,
            envelope,
            control: Some(tone.prediction_control(now, rhythms)),
            sine: tone.prediction_sine(now),
            bank: tone.prediction_bank(now),
            scheduled_release: None,
        }
    }

    fn sample(model: ToneEnergy, tick: u64) -> f64 {
        let bank = model.bank.unwrap();
        let amplitude = (2. * model.at(tick, None).unwrap()).sqrt();
        (0..bank.len)
            .map(|lane| bank.lane_at(lane, tick).unwrap()[1] * amplitude)
            .sum()
    }

    #[test]
    fn bank_forecast_tracks_the_rendered_harmonic_and_modal_waveforms() {
        let rhythms = NeuralRhythms::default();
        for kind in [BodyKind::Harmonic, BodyKind::Modal] {
            let (mut maximum, mut peak) = (0_f64, 0_f64);
            for prefix in [0, 37, 300, 8192] {
                for frequency in [181., 523., 2900.] {
                    let mut tone = tone(kind, 37, frequency, 7731);
                    for tick in 0..prefix {
                        tone.kick_planned_if_due(tick);
                        tone.render_tick(tick, 48000., 1. / 48000., &rhythms);
                    }
                    let saved = model(&tone, prefix, &rhythms);
                    assert!(saved.sine.is_none() && saved.bank.is_some());
                    for tick in prefix..prefix + 48000 {
                        tone.kick_planned_if_due(tick);
                        let actual =
                            f64::from(tone.render_tick(tick, 48000., 1. / 48000., &rhythms));
                        peak = peak.max(actual.abs());
                        if tick >= saved.bank.unwrap().first_sample {
                            let error = (sample(saved, tick) - actual).abs();
                            maximum = maximum.max(error);
                            assert!(
                                error < 1e-4,
                                "{kind:?} prefix={prefix} frequency={frequency} tick={tick} error={error}"
                            );
                        }
                    }
                }
            }
            eprintln!("BANK_WAVEFORM {kind:?} maximum_error={maximum} peak={peak}");
        }
    }

    #[test]
    fn delayed_onset_draws_the_phases_of_a_tone_rendered_at_that_onset() {
        let rhythms = NeuralRhythms::default();
        for kind in [BodyKind::Harmonic, BodyKind::Modal] {
            let issued = model(&tone(kind, 37, 293., 11), 0, &rhythms);
            let mut delayed = issued;
            delayed.envelope.onset += 5000;
            delayed.envelope.hold_end += 5000;
            delayed.envelope.release_end += 5000;
            let control = delayed.control.as_mut().unwrap();
            control.starts_at = control.starts_at.map(|at| at + 5000);
            control.kick_at = control.kick_at.map(|at| at + 5000);
            delayed.bank = delayed.bank.map(|bank| bank.for_new_onset(5037, 99));
            let mut actual = tone(kind, 5037, 293., 99);
            for tick in 0..20000 {
                actual.kick_planned_if_due(tick);
                let rendered = f64::from(actual.render_tick(tick, 48000., 1. / 48000., &rhythms));
                if tick >= 5037 {
                    let error = (sample(delayed, tick) - rendered).abs();
                    assert!(error < 1e-4, "{kind:?} tick={tick} error={error}");
                }
            }
        }
    }

    /// Rendered mean square over `[now, now + width)` against the candidate quadrature.
    fn window_error(mut tone: Tone, now: u64, width: u64, rhythms: &NeuralRhythms) -> f64 {
        let frozen = model(&tone, now, rhythms);
        assert!(frozen.sine.is_some() || frozen.bank.is_some());
        let predicted = crate::life::action_candidates::energy::project_window(
            &[(1, [true, true], frozen)],
            None,
            now,
            (None, None),
            0,
            [now, now + width],
            true,
        )
        .unwrap();
        assert!(predicted.coherent_energies.iter().all(Option::is_some));
        let mut actual = 0.;
        for tick in now..now + width {
            tone.kick_planned_if_due(tick);
            tone.apply_updates_if_due(tick);
            actual += f64::from(tone.render_tick(tick, 48000., 1. / 48000., rhythms)).powi(2);
        }
        actual /= width as f64;
        (predicted.mean.unwrap() / actual - 1.).abs()
    }

    #[test]
    fn amplitude_steps_and_short_envelopes_match_the_rendered_window_energy() {
        let rhythms = NeuralRhythms::default();
        let mut maximum = 0_f64;
        for kind in [BodyKind::Sine, BodyKind::Harmonic, BodyKind::Modal] {
            // A smoothed amplitude step between the quadrature nodes of a long span.
            let mut stepped = tone(kind, 37, 293., 11);
            stepped.set_smoothing_tau_sec(0.002);
            for tick in 0..2048 {
                stepped.kick_planned_if_due(tick);
                stepped.render_tick(tick, 48000., 1. / 48000., &rhythms);
            }
            for (at, target) in [(2300, 1.), (2400, 0.2)] {
                stepped.schedule_update(
                    at,
                    crate::life::phonation_engine::ToneUpdate {
                        target_freq_hz: None,
                        target_amp: Some(target),
                        continuous_drive: None,
                    },
                );
            }
            // An attack and decay far shorter than the old 16-sample span floor.
            let mut short = Tone::from_parts(
                Timebase {
                    fs: 48000.,
                    hop: 64,
                },
                2100,
                200000,
                293.,
                0.2,
                Some(BodySnapshot {
                    kind,
                    amp_scale: 1.,
                    brightness: 0.6,
                    inharmonic: 0.,
                    spread: 0.,
                    unison: 1,
                    motion: 0.,
                    ratios: matches!(kind, BodyKind::Modal)
                        .then(|| (1..=16).map(|k| k as f32).collect::<Vec<_>>().into()),
                }),
                Some(RenderModulatorSpec::SeqGate { duration_sec: 10. }),
                Some(ToneAdsr {
                    attack_sec: 0.001,
                    decay_sec: 0.001,
                    sustain_level: 0.5,
                    release_sec: 0.1,
                }),
            )
            .unwrap();
            short.seed_modal_phases(5);
            short.schedule_planned_kick(OnsetKick { strength: 0.85 });
            short.arm_onset_trigger(0.85);
            for (label, tone) in [("step", stepped), ("short", short)] {
                let error = window_error(tone, 2048, 12000, &rhythms);
                eprintln!("BANK_WINDOW {kind:?} {label} relative_error={error}");
                maximum = maximum.max(error);
                assert!(error < 1e-3, "{kind:?} {label} relative_error={error}");
            }
        }
        eprintln!("BANK_WINDOW maximum_relative_error={maximum}");
    }

    fn custom(kind: BodyKind, modulator: RenderModulatorSpec, adsr: ToneAdsr) -> Tone {
        let mut tone = Tone::from_parts(
            Timebase {
                fs: 48000.,
                hop: 64,
            },
            2100,
            200000,
            293.,
            0.2,
            Some(BodySnapshot {
                kind,
                amp_scale: 1.,
                brightness: 0.6,
                inharmonic: 0.,
                spread: 0.,
                unison: 1,
                motion: 0.,
                ratios: matches!(kind, BodyKind::Modal)
                    .then(|| (1..=16).map(|k| k as f32).collect::<Vec<_>>().into()),
            }),
            Some(modulator),
            Some(adsr),
        )
        .unwrap();
        tone.seed_modal_phases(5);
        tone.schedule_planned_kick(OnsetKick { strength: 0.85 });
        tone.arm_onset_trigger(0.85);
        tone
    }

    #[test]
    fn a_closing_gate_and_two_sample_spans_match_the_renderer() {
        let rhythms = NeuralRhythms::default();
        let flat = ToneAdsr {
            attack_sec: 0.001,
            decay_sec: 0.,
            sustain_level: 1.,
            release_sec: 0.1,
        };
        for kind in [BodyKind::Sine, BodyKind::Harmonic, BodyKind::Modal] {
            // The gate breakpoint is the first sample the renderer silences, on and off a
            // whole number of samples. Bins around it stay unsupported by rounding doubt.
            for duration_sec in [3000. / 48000., 3000.5 / 48000.] {
                let mut gated = custom(kind, RenderModulatorSpec::SeqGate { duration_sec }, flat);
                let closing = model(&gated, 2048, &rhythms).control.unwrap().steps()[4].unwrap();
                let mut last_sound = 0;
                for tick in 2048..8000 {
                    gated.kick_planned_if_due(tick);
                    if gated.render_tick(tick, 48000., 1. / 48000., &rhythms) != 0. {
                        last_sound = tick;
                    }
                }
                assert_eq!(closing, last_sound + 1, "{kind:?} gate {duration_sec}");
            }
            // A 64-sample attack in 32 spans leaves two samples in each.
            let short = custom(
                kind,
                RenderModulatorSpec::SeqGate { duration_sec: 10. },
                ToneAdsr {
                    attack_sec: 64. / 48000.,
                    ..flat
                },
            );
            let error = window_error(short, 2048, 12000, &rhythms);
            eprintln!("BANK_WINDOW {kind:?} attack64 relative_error={error}");
            assert!(error < 1e-3, "{kind:?} attack64 error={error}");
        }
    }

    #[test]
    fn breakpoints_that_do_not_fit_leave_the_bin_without_a_coherent_value() {
        let rhythms = NeuralRhythms::default();
        let mut retained = Vec::new();
        for id in 0..40_u64 {
            let mut tone = tone(BodyKind::Sine, 37, 293., id);
            tone.set_smoothing_tau_sec(0.002);
            for tick in 0..2048 {
                tone.kick_planned_if_due(tick);
                tone.render_tick(tick, 48000., 1. / 48000., &rhythms);
            }
            for step in 0..4 {
                tone.schedule_update(
                    2100 + id * 8 + step * 400,
                    crate::life::phonation_engine::ToneUpdate {
                        target_freq_hz: None,
                        target_amp: Some(0.1 + step as f32 * 0.1),
                        continuous_drive: None,
                    },
                );
            }
            retained.push((id, [true, true], model(&tone, 2048, &rhythms)));
        }
        let window = crate::life::action_candidates::energy::project_window(
            &retained,
            None,
            2048,
            (None, None),
            0,
            [2048, 2048 + 192000],
            true,
        )
        .unwrap();
        // All 160 steps fall in the first 12000-sample bin; later bins still fit.
        assert_eq!(window.coherent_energies[0], None);
        assert!(window.coherent_energies[15].is_some());
        assert!(window.mean.is_some());
    }

    #[test]
    fn a_direct_impulse_on_a_ringing_bank_is_not_forecast() {
        let rhythms = NeuralRhythms::default();
        for kind in [BodyKind::Harmonic, BodyKind::Modal] {
            let mut tone = tone(kind, 37, 293., 11);
            for tick in 0..512 {
                tone.kick_planned_if_due(tick);
                tone.render_tick(tick, 48000., 1. / 48000., &rhythms);
            }
            assert!(tone.prediction_bank(512).is_some());
            tone.trigger_impulse(0.5);
            assert!(tone.prediction_bank(512).is_none());
        }
    }

    #[test]
    fn stochastic_or_moving_bodies_stay_unsupported() {
        let rhythms = NeuralRhythms::default();
        let mut driven = tone(BodyKind::Modal, 37, 293., 11);
        driven.set_continuous_drive(0.02);
        assert!(driven.prediction_bank(0).is_none());
        let mut moving = tone(BodyKind::Harmonic, 37, 293., 11);
        moving.schedule_update(
            100,
            crate::life::phonation_engine::ToneUpdate {
                target_freq_hz: Some(440.),
                target_amp: None,
                continuous_drive: None,
            },
        );
        assert!(moving.prediction_bank(0).is_none());
        assert!(
            model(&tone(BodyKind::Sine, 37, 293., 11), 0, &rhythms)
                .bank
                .is_none()
        );
    }
}
