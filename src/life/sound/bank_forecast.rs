//! Read-only carrier state of the multi-lane backends; amplitude envelopes stay separate.

#[cfg(any(test, feature = "research"))]
#[path = "bank_forecast/onset_window.rs"]
mod onset_window;

#[cfg(test)]
use super::oscillator_bank::seeded_phase_state;
#[cfg(test)]
use crate::synth::resonator::{input_phase_seed, seeded_input_coupling};

pub(crate) const LANES: usize = 16;

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct Lane {
    /// State before the first forecast sample is rendered.
    pub state: [f32; 2],
    /// Static oscillator: cosine and sine. Moving oscillator: base Hz and lane ratio. Resonator: `r` and `e`.
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
        #[serde(skip_serializing_if = "Option::is_none")]
        motion: Option<(super::pitch_motion::MotionClock, f32, f32)>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub burst: Option<super::attack_noise::Burst>,
    pub first_sample: u64,
    pub response: Response,
    pub len: usize,
    pub lanes: [Lane; LANES],
}

impl BankForecast {
    /// A delayed onset draws the phases the renderer would draw for that onset.
    #[cfg(test)]
    pub(crate) fn for_new_onset(mut self, first_sample: u64, mut seed: u64) -> Self {
        self.first_sample = first_sample;
        if let Some(burst) = &mut self.burst {
            burst.seed = seed;
            burst.onset = first_sample;
        }
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
    use crate::life::sound::{BodyKind, BodySnapshot, RenderModulatorSpec, Tone, ToneAdsr};
    use crate::life::tone_energy::ToneEnergy;

    fn tone(kind: BodyKind, onset: u64, frequency: f32, seed: u64) -> Tone {
        tone_with_motion(kind, onset, frequency, seed, 0.)
    }

    fn tone_with_motion(
        kind: BodyKind,
        onset: u64,
        frequency: f32,
        seed: u64,
        motion: f32,
    ) -> Tone {
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
                attack_noise: 0.,
                amp_scale: 1.,
                brightness: 0.6,
                inharmonic: 0.,
                spread: 0.,
                unison: 1,
                motion,
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

    #[test]
    fn moving_attack_and_ringing_forecast_match_pcm() {
        let rhythms = NeuralRhythms::default();
        for pitch in [180., 400., 1100.] {
            let mut tone = Tone::from_parts(
                Timebase {
                    fs: 48000.,
                    hop: 64,
                },
                0,
                48000,
                pitch,
                0.4,
                Some(BodySnapshot {
                    kind: BodyKind::Harmonic,
                    amp_scale: 1.,
                    brightness: 0.5,
                    inharmonic: 0.,
                    spread: 0.3,
                    unison: 3,
                    motion: 0.5,
                    attack_noise: 1.,
                    ratios: Some(std::sync::Arc::from([1., 2., 3., 4.])),
                }),
                None,
                Some(ToneAdsr {
                    attack_sec: 0.02,
                    decay_sec: 0.5,
                    sustain_level: 0.8,
                    release_sec: 0.002,
                }),
            )
            .unwrap();
            tone.seed_modal_phases(212);
            tone.arm_onset_trigger(0.85);
            #[cfg(feature = "research")]
            let mut phase3 = {
                let mut phase3 = tone.clone();
                phase3.enable_phase3();
                phase3
            };
            let energy_error = window_error(tone.clone(), 0, 768, &rhythms);
            eprintln!("DR42 onset energy {pitch} Hz relative error {energy_error}");
            assert!(energy_error < 0.001);
            let mut saved = model(&tone, 0, &rhythms);
            let mut max_error = 0_f64;
            for tick in 0..13500 {
                if tick == 900 {
                    tone.note_off(tick);
                    #[cfg(feature = "research")]
                    phase3.note_off(tick);
                    let tail_error = window_error(tone.clone(), tick, 768, &rhythms);
                    eprintln!("DR42 release energy {pitch} Hz relative error {tail_error}");
                    assert!(tail_error < 0.001);
                }
                if [805, 900, 1200].contains(&tick) {
                    saved = model(&tone, tick, &rhythms);
                }
                let pcm = tone.render_tick(tick, 48000., 1. / 48000., &rhythms);
                let forecast = sample(saved, tick);
                max_error = max_error.max((forecast - f64::from(pcm)).abs());
                #[cfg(feature = "research")]
                {
                    let native = phase3.render_tick(tick, 48000., 1. / 48000., &rhythms);
                    max_error = max_error.max((forecast - f64::from(native)).abs());
                    if tick == 1200 {
                        assert!(phase3.residual_bound(tick).is_none());
                    }
                    if tick == 12960 {
                        assert_eq!(phase3.residual_bound(tick), Some(0.));
                        assert_eq!(native, 0.);
                    }
                }
                if tick == 1200 {
                    assert!(!tone.is_done(tick));
                }
                if tick == 12960 {
                    assert!(tone.is_done(tick));
                    assert_eq!(tone.end_tick(), tick);
                    assert_eq!(pcm, 0.);
                }
            }
            eprintln!("DR42 waveform {pitch} Hz max error {max_error}");
            assert!(max_error < 1e-4);
        }
    }

    #[test]
    fn seeded_attack_noise_has_the_requested_level_and_decay_on_restrike() {
        let mut rhythms = NeuralRhythms::default();
        rhythms.theta.beta = 0.;
        let render = |seed, level: Option<f32>, native: bool, attack_sec| {
            let mut tone = Tone::from_parts(
                Timebase {
                    fs: 48000.,
                    hop: 256,
                },
                0,
                12000,
                340.,
                0.4,
                Some(BodySnapshot {
                    kind: BodyKind::Harmonic,
                    amp_scale: 1.,
                    brightness: 0.6,
                    inharmonic: 0.,
                    spread: 0.,
                    unison: 1,
                    motion: 0.,
                    ratios: None,
                    attack_noise: level.unwrap_or_default(),
                }),
                None,
                Some(ToneAdsr {
                    attack_sec,
                    decay_sec: if attack_sec == 0. { 0. } else { 0.5 },
                    sustain_level: if attack_sec == 0. { 1. } else { 0.8 },
                    release_sec: 2.,
                }),
            )
            .unwrap();
            tone.seed_modal_phases(seed);
            #[cfg(feature = "research")]
            if native {
                tone.enable_phase3();
            }
            #[cfg(not(feature = "research"))]
            assert!(!native);
            (0..8192)
                .map(|tick| {
                    if tick == 0 || tick == 4096 {
                        tone.trigger_impulse(if tick == 0 { 1. } else { 0.5 });
                    }
                    tone.render_tick(tick, 48000., 1. / 48000., &rhythms)
                })
                .collect::<Vec<_>>()
        };
        let level = 0.25;
        for attack_sec in [0., 0.02] {
            for native in [
                false,
                #[cfg(feature = "research")]
                true,
            ] {
                let bits = |level| {
                    render(18, level, native, attack_sec)
                        .into_iter()
                        .map(f32::to_bits)
                };
                assert!(bits(Some(level)).eq(bits(Some(level))));
                assert!(bits(Some(0.)).eq(bits(None)));
                let mut energy = [[0f64; 12]; 2];
                for seed in 0..48 {
                    let wet = render(seed, Some(level), native, attack_sec);
                    let dry = render(seed, None, native, attack_sec);
                    for (hit, bins) in energy.iter_mut().enumerate() {
                        for (i, e) in bins.iter_mut().enumerate() {
                            for n in hit * 4096 + i * 240..hit * 4096 + (i + 1) * 240 {
                                *e += f64::from(wet[n] - dry[n]).powi(2);
                            }
                        }
                    }
                }
                for (hit, bins) in energy.iter().enumerate() {
                    let log_rms = bins.map(|e| 0.5 * (e / (48. * 240.)).ln());
                    let mean = log_rms.iter().sum::<f64>() / 12.;
                    let slope = log_rms
                        .iter()
                        .enumerate()
                        .map(|(i, y)| (i as f64 - 5.5) * (y - mean))
                        .sum::<f64>()
                        / 143.;
                    let tau = -5. / slope;
                    // Exact mean squared exponential over one 5 ms bin.
                    let q = (2. * slope / 240.).exp();
                    let bin_factor = ((1. - q.powi(240)) / (240. * (1. - q))).sqrt();
                    let initial = (mean - 5.5 * slope).exp() / bin_factor;
                    let expected = if hit == 0 { 0.1 } else { 0.05 };
                    eprintln!(
                        "DR42 burst attack_s={attack_sec} native={native} hit={hit} tau_ms={tau} initial_rms={initial}"
                    );
                    assert!((tau - 15.).abs() < 0.5 && (initial / expected - 1.).abs() < 0.04);
                }
            }
        }
    }

    #[cfg(feature = "profile-alloc")]
    #[test]

    fn unmodulated_harmonic_render_has_no_steady_allocations() {
        let rhythms = NeuralRhythms::default();
        let mut tones: Vec<_> = (0..8)
            .map(|i| {
                let mut tone = Tone::from_parts(
                    Timebase {
                        fs: 48000.,
                        hop: 256,
                    },
                    0,
                    1000000,
                    180. + i as f32 * 19.,
                    0.2,
                    Some(BodySnapshot {
                        kind: BodyKind::Harmonic,
                        amp_scale: 1.,
                        brightness: 0.6,
                        inharmonic: 0.,
                        spread: 0.25,
                        unison: 3,
                        motion: 0.,
                        ratios: None,
                        attack_noise: 0.,
                    }),
                    None,
                    Some(ToneAdsr {
                        attack_sec: 0.,
                        decay_sec: 0.,
                        sustain_level: 1.,
                        release_sec: 0.1,
                    }),
                )
                .unwrap();
                tone.seed_modal_phases(i);
                tone
            })
            .collect();
        let mut hop = |h: u64| {
            let mut mix = [0.; 256];
            for (i, tone) in tones.iter_mut().enumerate() {
                if h % 64 == 0 {
                    tone.trigger_impulse(1.);
                    tone.schedule_update(
                        h * 256,
                        crate::life::phonation_engine::ToneUpdate {
                            target_freq_hz: Some(
                                180. + i as f32 * 19. + ((h / 64) % 2) as f32 * 20.,
                            ),
                            ..Default::default()
                        },
                    );
                    tone.apply_updates_if_due(h * 256);
                }
                for (j, y) in mix.iter_mut().enumerate() {
                    *y += tone.render_tick(h * 256 + j as u64, 48000., 1. / 48000., &rhythms);
                }
            }
            std::hint::black_box(mix);
        };
        for h in 0..128 {
            hop(h);
        }
        crate::runtime_profile::begin_allocations();
        for h in 128..2176 {
            hop(h);
        }
        let alloc = crate::runtime_profile::finish_allocations().unwrap();
        eprintln!("steady harmonic allocations={alloc:?}");
        assert_eq!((alloc.count, alloc.bytes), (0, 0));
    }

    fn model(tone: &Tone, now: u64, rhythms: &NeuralRhythms) -> ToneEnergy {
        let (_, amplitude, envelope) = tone.prediction_parameters(None);
        ToneEnergy {
            amplitude,
            envelope,
            control: Some(tone.prediction_control(now, rhythms)),
            sine: tone.prediction_sine(now),
            bank: tone.prediction_bank(now, rhythms),
            scheduled_release: None,
        }
    }

    fn sample(model: ToneEnergy, tick: u64) -> f64 {
        let bank = model.bank.unwrap();
        let amplitude = (2. * model.at(tick, None).unwrap()).sqrt();
        let carrier: f64 = (0..bank.len)
            .map(|lane| bank.lane_at(lane, tick).unwrap()[1] * amplitude)
            .sum();
        carrier
            + bank
                .burst
                .filter(|b| (b.onset..b.end()).contains(&tick))
                .map_or(0., |b| {
                    (0..super::super::attack_noise::NOISE_LANES)
                        .map(|i| b.point(i, tick)[1])
                        .sum()
                })
    }

    #[test]
    fn bank_forecast_tracks_the_rendered_harmonic_and_modal_waveforms() {
        let rhythms = NeuralRhythms::default();
        for (kind, motion) in [
            (BodyKind::Harmonic, 0.),
            (BodyKind::Modal, 0.),
            (BodyKind::Harmonic, 0.9),
        ] {
            let (mut maximum, mut peak) = (0_f64, 0_f64);
            for prefix in [0, 37, 300, 8192] {
                for frequency in [181., 523., 2900.] {
                    let mut tone = tone_with_motion(kind, 37, frequency, 7731, motion);
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
        let predicted = crate::life::energy_window::project_window(
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
                        autonomous_pulse: None,
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
                    attack_noise: 0.,
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
                attack_noise: 0.,
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
                        autonomous_pulse: None,
                        target_freq_hz: None,
                        target_amp: Some(0.1 + step as f32 * 0.1),
                        continuous_drive: None,
                    },
                );
            }
            retained.push((id, [true, true], model(&tone, 2048, &rhythms)));
        }
        let window = crate::life::energy_window::project_window(
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
            assert!(tone.prediction_bank(512, &rhythms).is_some());
            tone.trigger_impulse(0.5);
            assert!(tone.prediction_bank(512, &rhythms).is_none());
        }
    }

    #[test]
    fn stochastic_or_moving_bodies_stay_unsupported() {
        let rhythms = NeuralRhythms::default();
        let mut driven = tone(BodyKind::Modal, 37, 293., 11);
        driven.set_continuous_drive(0.02);
        assert!(driven.prediction_bank(0, &rhythms).is_none());
        let mut moving = tone(BodyKind::Harmonic, 37, 293., 11);
        moving.schedule_update(
            100,
            crate::life::phonation_engine::ToneUpdate {
                autonomous_pulse: None,
                target_freq_hz: Some(440.),
                target_amp: None,
                continuous_drive: None,
            },
        );
        assert!(moving.prediction_bank(0, &rhythms).is_none());
        assert!(
            model(&tone(BodyKind::Sine, 37, 293., 11), 0, &rhythms)
                .bank
                .is_none()
        );
    }
}
