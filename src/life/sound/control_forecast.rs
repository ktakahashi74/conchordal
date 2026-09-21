//! Bounded analytic control-amplitude forecasts from issued renderer state.

use super::render_modulator::RenderModulatorStateKind;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub enum AmplitudeModel {
    Unmodulated {
        gain: f32,
    },
    EntrainPulse {
        attack_step: f32,
        decay_rate: f32,
        sustain_level: f32,
        state: RenderModulatorStateKind,
        env_level: f32,
        autonomous_retrigger: bool,
    },
    SeqGate {
        timer: f32,
        duration_sec: f32,
    },
    DroneSway {
        phase: f32,
        sway_rate: f32,
        alpha: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct AmplitudeSmoothing {
    pub current: f32,
    pub target: f32,
    pub alpha: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct AmplitudeUpdate {
    pub at_sample: u64,
    pub target: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct AmplitudeUpdates {
    pub events: [AmplitudeUpdate; 4],
    pub len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct ControlForecast {
    pub issued_at: u64,
    /// Exclusive absolute sample of the first unsupported or overflowing update.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub amplitude_smoothing: Option<AmplitudeSmoothing>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub amplitude_updates: Option<AmplitudeUpdates>,
    pub sample_dt: f32,
    pub starts_at: Option<u64>,
    pub kick_at: Option<u64>,
    pub model: AmplitudeModel,
}

impl ControlForecast {
    /// Closed-form renderer smoothing, including the target sample's update.
    pub(crate) fn amplitude_at(self, tick: u64, constant: f32) -> Option<f64> {
        if tick < self.issued_at || self.valid_until.is_some_and(|end| tick >= end) {
            return None;
        }
        let Some(s) = self.amplitude_smoothing else {
            return (constant.is_finite() && constant >= 0.).then_some(f64::from(constant));
        };
        if !s.current.is_finite()
            || !s.target.is_finite()
            || s.current < 0.
            || s.target < 0.
            || !(0. ..=1.).contains(&s.alpha)
        {
            return None;
        }
        let mut current = f64::from(s.current);
        let mut target = f64::from(s.target);
        let mut from = self.issued_at;
        if let Some(updates) = self.amplitude_updates {
            for update in &updates.events[..updates.len] {
                let at = update.at_sample.max(self.issued_at);
                if at > tick {
                    break;
                }
                let decay = (1. - f64::from(s.alpha)).powf((at - from) as f64);
                current = target + (current - target) * decay;
                target = f64::from(update.target);
                from = at;
            }
        }
        let steps = (tick - from).saturating_add(1) as f64;
        let decay = (1. - f64::from(s.alpha)).powf(steps);
        Some(target + (current - target) * decay)
    }

    pub(crate) fn known_on(self, [start, end]: [u64; 2]) -> bool {
        let mut edges = [start, end, start, start];
        edges[2] = self.starts_at.unwrap_or(start).clamp(start, end);
        edges[3] = self.kick_at.unwrap_or(start).clamp(start, end);
        edges.sort_unstable();
        edges.windows(2).all(|pair| {
            if pair[0] == pair[1] {
                return true;
            }
            let a = self.gain_at(pair[0]);
            let b = self.gain_at(pair[1] - 1);
            // A gate crossing can hide rounding uncertainty between the endpoints.
            a.is_some()
                && b.is_some()
                && self.amplitude_at(pair[0], 0.).is_some()
                && self.amplitude_at(pair[1] - 1, 0.).is_some()
                && (!matches!(self.model, AmplitudeModel::SeqGate { .. }) || a == b)
        })
    }

    /// Includes the target sample's control step, without stepping a DSP backend.
    /// Ticks where the amplitude target steps or a sequence gate closes.
    pub(crate) fn steps(self) -> [Option<u64>; 5] {
        let mut out = [None; 5];
        if let Some(updates) = self.amplitude_updates {
            for (slot, update) in out.iter_mut().zip(&updates.events[..updates.len]) {
                *slot = Some(update.at_sample.max(self.issued_at));
            }
        }
        if let AmplitudeModel::SeqGate {
            timer,
            duration_sec,
        } = self.model
        {
            let dt = f64::from(self.sample_dt);
            let (origin, timer) = match self.kick_at {
                Some(at) => (at.max(self.issued_at), 0.),
                None => (self.issued_at, f64::from(timer)),
            };
            let left = (f64::from(duration_sec) - timer) / dt;
            if left.is_finite() && left > 0. {
                out[4] = Some(origin.saturating_add(left.ceil() as u64));
            }
        }
        out
    }

    /// Span bound while `[from, to)` overlaps an amplitude-smoothing transient: the same
    /// `2 lambda h <= 0.1` rule as the envelope decay, over seven time constants after the
    /// issue or a target step.
    pub(crate) fn smoothing_span(self, [from, to]: [u64; 2]) -> Option<u64> {
        let alpha = f64::from(self.amplitude_smoothing?.alpha);
        if !(alpha > 0. && alpha < 1.) {
            return None;
        }
        let lambda = -(1. - alpha).ln();
        let settle = (7. / lambda).ceil() as u64;
        std::iter::once(Some(self.issued_at))
            .chain(self.steps().into_iter().take(4))
            .flatten()
            .any(|at| from < at.saturating_add(settle) && to > at)
            .then(|| (0.1 / (2. * lambda)) as u64)
    }

    /// The modulator's linear attack `[origin, end)`; its gain changes slope at `end`.
    pub(crate) fn attack_span(self) -> Option<[u64; 2]> {
        let AmplitudeModel::EntrainPulse {
            attack_step,
            state,
            env_level,
            ..
        } = self.model
        else {
            return None;
        };
        let step = f64::from(attack_step * self.sample_dt);
        if !step.is_finite() || step <= 0. {
            return None;
        }
        let (origin, level) = match self.kick_at {
            Some(at) => (at.max(self.issued_at), 0.),
            None if state == RenderModulatorStateKind::Attack => {
                (self.issued_at, f64::from(env_level))
            }
            None => return None,
        };
        Some([
            origin,
            origin.saturating_add(((1. - level) / step).ceil().max(1.) as u64),
        ])
    }

    pub(crate) fn gain_at(self, tick: u64) -> Option<f64> {
        if tick < self.issued_at
            || self.valid_until.is_some_and(|end| tick >= end)
            || !self.sample_dt.is_finite()
            || self.sample_dt <= 0.
        {
            return None;
        }
        if self.starts_at.is_none_or(|start| tick < start) {
            return Some(0.);
        }
        let kick = self.kick_at.filter(|at| tick >= *at);
        let steps = (tick - self.issued_at).saturating_add(1) as f64;
        let since_kick = kick.map(|at| (tick - at.max(self.issued_at)).saturating_add(1) as f64);
        let dt = f64::from(self.sample_dt);
        let gain = match self.model {
            AmplitudeModel::Unmodulated { gain } => f64::from(gain),
            AmplitudeModel::SeqGate {
                timer,
                duration_sec,
            } => {
                if !timer.is_finite() || !duration_sec.is_finite() {
                    return None;
                }
                if duration_sec <= 0. || (since_kick.is_none() && timer >= duration_sec) {
                    return Some(0.);
                }
                let n = since_kick.unwrap_or(steps);
                let timer = since_kick.map_or(f64::from(timer) + steps * dt, |n| n * dt);
                // Repeated f32 additions can move the strict gate boundary.
                let rounding = n * f64::from(f32::EPSILON);
                if rounding >= 1.
                    || (timer - f64::from(duration_sec)).abs()
                        <= rounding / (1. - rounding) * timer.abs()
                {
                    return None;
                }
                f64::from(timer < f64::from(duration_sec))
            }
            AmplitudeModel::DroneSway {
                phase,
                sway_rate,
                alpha,
            } => {
                if !phase.is_finite() || !sway_rate.is_finite() || !alpha.is_finite() {
                    return None;
                }
                let step = f64::from(std::f32::consts::TAU * sway_rate.max(0.01) * self.sample_dt);
                let phase =
                    (f64::from(phase) + step * steps).rem_euclid(f64::from(std::f32::consts::TAU));
                ((0.3 + 0.7 * (0.5 * (phase.sin() + 1.))) * (1. + 0.5 * f64::from(alpha)))
                    .clamp(0., 1.)
            }
            AmplitudeModel::EntrainPulse {
                attack_step,
                decay_rate,
                sustain_level,
                state,
                env_level,
                autonomous_retrigger,
            } => {
                if [attack_step, decay_rate, sustain_level, env_level]
                    .iter()
                    .any(|x| !x.is_finite())
                    || attack_step < 0.
                    || decay_rate < 0.
                {
                    return None;
                }
                let floor = f64::from(sustain_level.clamp(0., 1.));
                let (state, mut level, mut remaining) = since_kick
                    .map_or((state, f64::from(env_level), steps), |n| {
                        (RenderModulatorStateKind::Attack, 0., n)
                    });
                // A positive sustain after an issued kick never returns to Idle.
                if autonomous_retrigger && (state == RenderModulatorStateKind::Idle || floor == 0.)
                {
                    return None;
                }
                if state == RenderModulatorStateKind::Idle {
                    return Some(0.);
                }
                if state == RenderModulatorStateKind::Attack {
                    let step = f64::from(attack_step * self.sample_dt);
                    if step <= 0. {
                        return Some(if level > 1e-6 { level } else { 0. });
                    }
                    let attack_steps = ((1. - level) / step).ceil().max(1.);
                    if remaining < attack_steps {
                        level += remaining * step;
                        return Some(if level > 1e-6 { level } else { 0. });
                    }
                    level = 1.;
                    remaining -= attack_steps;
                }
                if remaining > 0. {
                    // Match the renderer's rounded one-sample decay factor.
                    level *= f64::from((-decay_rate * self.sample_dt).exp()).powf(remaining);
                    if level <= floor + f64::from(0.001f32) {
                        level = floor;
                    }
                }
                if level > 1e-6 { level } else { 0. }
            }
        };
        (gain.is_finite() && gain >= 0.).then_some(gain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::modulation::NeuralRhythms;
    use crate::life::phonation_engine::OnsetKick;
    use crate::life::sound::{AutonomousPulseSpec, RenderModulator, RenderModulatorSpec};

    #[test]
    fn amplitude_smoothing_matches_sample_steps_and_preserves_absolute_clock() {
        let mut maximum_error = 0_f64;
        for rate in [8000., 48000.] {
            for tau in [0., 0.001, 0.05, 1.] {
                let alpha = if tau == 0. {
                    1.
                } else {
                    1. - (-1_f32 / (rate * tau)).exp()
                };
                for (current, target) in [
                    (0_f32, 0.2_f32),
                    (0.5, 0.),
                    (0.1, 0.4),
                    (0.4, 0.1),
                    (0.1, 0.1),
                ] {
                    let frozen = ControlForecast {
                        issued_at: 100,
                        valid_until: Some(100 + 4 * rate as u64),
                        amplitude_smoothing: Some(AmplitudeSmoothing {
                            current,
                            target,
                            alpha,
                        }),
                        amplitude_updates: None,
                        sample_dt: 1. / rate,
                        starts_at: Some(100),
                        kick_at: None,
                        model: AmplitudeModel::Unmodulated { gain: 1. },
                    };
                    let mut actual = current;
                    for step in 0..4 * rate as u64 {
                        actual += alpha * (target - actual);
                        if step < 2 || step % 512 == 0 {
                            let predicted = frozen.amplitude_at(100 + step, 99.).unwrap();
                            let error = (predicted - f64::from(actual)).abs();
                            maximum_error = maximum_error.max(error);
                            // f32 accumulation can stop short of the target; no exactness claim.
                            assert!(
                                error < 0.001,
                                "rate={rate}, tau={tau}, {current}->{target}, step={step}, error={error}"
                            );
                        }
                    }
                    assert!(frozen.known_on([100, 100 + 4 * rate as u64]));
                    assert_eq!(frozen.amplitude_at(99, current), None);
                    assert_eq!(frozen.amplitude_at(100 + 4 * rate as u64, current), None);
                    let mut invalid = frozen;
                    invalid.amplitude_smoothing.as_mut().unwrap().alpha = f32::NAN;
                    assert_eq!(invalid.amplitude_at(100, current), None);
                    assert!(!invalid.known_on([100, 101]));
                }
            }
        }
        eprintln!("amplitude smoothing max absolute error={maximum_error}");
    }

    #[test]
    fn pending_update_is_an_exclusive_absolute_support_boundary() {
        let mut control = ControlForecast {
            issued_at: 100,
            valid_until: Some(800),
            amplitude_smoothing: None,
            amplitude_updates: None,
            sample_dt: 1. / 8000.,
            starts_at: Some(200),
            kick_at: None,
            model: AmplitudeModel::Unmodulated { gain: 1. },
        };
        assert_eq!(control.gain_at(99), None);
        assert_eq!(control.gain_at(199), Some(0.));
        assert_eq!(control.gain_at(200), Some(1.));
        assert_eq!(control.gain_at(799), Some(1.));
        assert_eq!(control.gain_at(800), None);
        assert_eq!(control.gain_at(801), None);
        assert!(control.known_on([100, 800]));
        assert!(control.known_on([800, 800]));
        assert!(!control.known_on([100, 801]));
        assert!(!control.known_on([800, 801]));
        control.starts_at = None;
        assert_eq!(control.gain_at(799), Some(0.));
        assert_eq!(control.gain_at(800), None);
        control.valid_until = Some(100);
        assert_eq!(control.gain_at(100), None);
        assert!(!control.known_on([100, 101]));
    }

    #[test]
    fn continuous_support_rejects_hidden_gate_uncertainty_and_splits_kicks() {
        let mut control = ControlForecast {
            issued_at: 0,
            valid_until: None,
            amplitude_smoothing: None,
            amplitude_updates: None,
            sample_dt: 1. / 8000.,
            starts_at: Some(0),
            kick_at: None,
            model: AmplitudeModel::SeqGate {
                timer: 0.,
                duration_sec: 0.1,
            },
        };
        assert!(control.known_on([0, 700]));
        assert!(control.known_on([900, 1000]));
        assert!(control.gain_at(700).is_some() && control.gain_at(899).is_some());
        assert_eq!(control.gain_at(799), None);
        assert!(!control.known_on([700, 900]));
        control.kick_at = Some(1000);
        assert!(control.known_on([900, 1100]));
        assert!(!control.known_on([900, 1900]));
        control.starts_at = Some(1000);
        assert!(control.known_on([0, 1100]));
        control.model = AmplitudeModel::Unmodulated { gain: f32::NAN };
        assert!(control.known_on([0, 1000]));
        assert!(!control.known_on([0, 1001]));
    }

    #[test]
    fn issued_controls_match_sample_stepped_modulators_without_advancing_them() {
        let mut rhythms = NeuralRhythms::default();
        rhythms.theta.alpha = 0.7;
        let specs = [
            RenderModulatorSpec::EntrainPulse {
                attack_step: 58.823524,
                decay_rate: 14.084507,
                sustain_level: 0.61,
                initial_state: RenderModulatorStateKind::Idle,
                initial_env_level: 0.,
                alpha_gain: 0.5,
                beta_gain: 0.5,
                autonomous_pulse: None,
            },
            RenderModulatorSpec::SeqGate { duration_sec: 0.23 },
            RenderModulatorSpec::DroneSway {
                phase: 0.7,
                sway_rate: 0.41,
            },
        ];
        let mut maximum_error = 0.0f64;
        let mut comparisons = 0;
        for fs in [8000., 48000.] {
            let dt = 1. / fs;
            for spec in &specs {
                for prefix in [0, 400, 8000] {
                    let mut live = RenderModulator::from_spec(spec.clone());
                    live.kick_planned(OnsetKick { strength: 1. });
                    for _ in 0..prefix {
                        live.process(&rhythms, dt);
                    }
                    for kick_delay in [None, Some(0), Some(37)] {
                        let issued = ControlForecast {
                            issued_at: prefix,
                            valid_until: None,
                            amplitude_smoothing: None,
                            amplitude_updates: None,
                            sample_dt: dt,
                            starts_at: Some(prefix),
                            kick_at: kick_delay.map(|d| prefix + d),
                            model: live.amplitude_model(&rhythms),
                        };
                        let saved = live.amplitude_model(&rhythms);
                        let mut reference = live.clone();
                        for offset in 0..=16000 {
                            let tick = prefix + offset;
                            if issued.kick_at == Some(tick) {
                                reference.kick_planned(OnsetKick { strength: 1. });
                            }
                            let signal = reference.process(&rhythms, dt);
                            if [0, 36, 37, 63, 400, 800, 4000, 16000].contains(&offset) {
                                let expected = if signal.is_active {
                                    f64::from(signal.amplitude)
                                } else {
                                    0.
                                };
                                let predicted =
                                    issued.gain_at(tick).expect("away from ambiguous gate edge");
                                let error = (predicted - expected).abs();
                                maximum_error = maximum_error.max(error);
                                assert!(
                                    error < 0.002,
                                    "{spec:?}, fs={fs}, prefix={prefix}, kick={kick_delay:?}, offset={offset}, expected={expected}, predicted={predicted}"
                                );
                                comparisons += 1;
                            }
                        }
                        assert_eq!(saved, live.amplitude_model(&rhythms));
                    }
                }
            }
        }
        eprintln!("control comparisons={comparisons}, maximum absolute gain error={maximum_error}");
    }

    #[test]
    fn excitation_and_autonomous_uncertainty_are_not_unity_fallbacks() {
        let spec = RenderModulatorSpec::EntrainPulse {
            attack_step: 100.,
            decay_rate: 10.,
            sustain_level: 0.6,
            initial_state: RenderModulatorStateKind::Idle,
            initial_env_level: 0.,
            alpha_gain: 0.5,
            beta_gain: 0.5,
            autonomous_pulse: Some(AutonomousPulseSpec {
                rate_hz: 2.,
                phase_0_1: 0.9,
                retrigger: true,
                env_open_threshold: 0.1,
                mag_threshold: 0.1,
                alpha_threshold: 0.1,
            }),
        };
        let live = RenderModulator::from_spec(spec);
        let mut issued = ControlForecast {
            issued_at: 100,
            valid_until: None,
            amplitude_smoothing: None,
            amplitude_updates: None,
            sample_dt: 1. / 8000.,
            starts_at: Some(100),
            kick_at: None,
            model: live.amplitude_model(&NeuralRhythms::default()),
        };
        assert_eq!(issued.gain_at(99), None);
        assert_eq!(issued.gain_at(100), None);
        issued.starts_at = None;
        assert_eq!(issued.gain_at(10000), Some(0.));
        issued.starts_at = Some(200);
        issued.kick_at = Some(200);
        assert_eq!(issued.gain_at(199), Some(0.));
        assert!(issued.gain_at(200).unwrap() > 0.);
        assert!((issued.gain_at(10000).unwrap() - f64::from(0.6f32)).abs() < 1e-12);
        if let AmplitudeModel::EntrainPulse { sustain_level, .. } = &mut issued.model {
            *sustain_level = 0.;
        }
        assert_eq!(issued.gain_at(10000), None);
        issued.model = AmplitudeModel::SeqGate {
            timer: 0.,
            duration_sec: 0.1,
        };
        assert_eq!(issued.gain_at(999), None);
    }
}
