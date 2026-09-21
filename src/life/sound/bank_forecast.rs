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
