use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExcitedState {
    closed: bool,
    retired: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct PrototypeTone(Option<ExcitedState>);

impl PrototypeTone {
    #[inline]
    pub(super) fn enabled(&self) -> bool {
        self.0.is_some()
    }

    #[inline]
    pub(super) fn blocked(&self) -> bool {
        self.0
            .as_ref()
            .is_some_and(|state| state.closed || state.retired)
    }

    #[inline]
    pub(super) fn retired(&self) -> Option<bool> {
        self.0.as_ref().map(|state| state.retired)
    }

    pub(super) fn rejects_update(&self, at: Tick, release_end: Tick, update: &ToneUpdate) -> bool {
        self.enabled()
            && (at >= release_end
                || self.blocked()
                || update
                    .target_freq_hz
                    .is_some_and(|v| !v.is_finite() || v <= 0.0)
                || update.target_amp.is_some_and(|v| !v.is_finite() || v < 0.0)
                || update
                    .continuous_drive
                    .is_some_and(|v| !v.is_finite() || v < 0.0))
    }
}

impl Tone {
    #[inline]
    pub(super) fn render_prototype_sample(
        &mut self,
        tick: Tick,
        dt: f32,
        rhythms: &NeuralRhythms,
    ) -> Option<f32> {
        self.prototype
            .enabled()
            .then(|| self.render_excited_tick(tick, dt, rhythms))
    }

    pub(crate) fn enable_phase3(&mut self) {
        self.prototype.0 = Some(ExcitedState {
            closed: false,
            retired: false,
        });
    }

    pub(crate) fn residual_bound(&self, now: Tick) -> Option<f64> {
        let state = self.prototype.0.as_ref()?;
        if now < self.envelope.release_end {
            return None;
        }
        if !self.started {
            return Some(0.0);
        }
        if !state.closed || now < self.sound_end_tick() {
            return None;
        }
        Some(match &self.backend {
            AnyBackend::Oscillator(_) => 0.0,
            AnyBackend::Resonator(engine) => engine.phase3_residual_bound(),
        })
    }

    pub(crate) fn retire_phase3(&mut self) {
        if let Some(state) = &mut self.prototype.0 {
            state.retired = true;
        }
    }

    fn render_excited_tick(&mut self, tick: Tick, dt: f32, rhythms: &NeuralRhythms) -> f32 {
        if tick < self.envelope.onset || self.prototype.0.as_ref().unwrap().retired {
            return 0.0;
        }
        let open = tick < self.envelope.release_end;
        if !open {
            let state = self.prototype.0.as_mut().unwrap();
            state.closed = true;
            self.pending_updates.clear();
            self.pending_trigger = None;
            self.planned_kick_pending = None;
            self.pending_impulse_energy = 0.0;
            if let AnyBackend::Oscillator(bank) = &mut self.backend {
                return bank.attack_sample(self.current_pitch_hz, tick, 0., 0.);
            }
        } else {
            if let Some(trigger) = self.pending_trigger
                && tick >= trigger.at_tick
            {
                self.pending_trigger = None;
                self.trigger_impulse(trigger.energy);
            }
            self.advance_smoothing();
        }
        if self.pending_impulse_energy > 0.0 {
            self.started = true;
        }
        if !self.started {
            return 0.0;
        }
        let impulse = std::mem::take(&mut self.pending_impulse_energy);
        let mut z = 0.0;
        let mut strike_gain = 0.;
        let mut spectral_drive = 0.0;
        let mut active = false;
        if open {
            let signal = self.articulation_signal(rhythms, dt);
            strike_gain = signal.strike_gain;
            // Preserve the legacy harmonic burst independently of amplitude kick.
            spectral_drive =
                impulse + self.continuous_drive * (signal.amplitude * self.envelope.gain_at(tick));
            active = signal.is_active;
            if active {
                z = self.envelope.gain_at(tick) * signal.amplitude.max(0.0);
            }
        }
        let a = self.current_amp;
        let sample = match &mut self.backend {
            AnyBackend::Oscillator(bank) => {
                let gain = if bank.is_sine() {
                    if impulse > 0.0 {
                        self.sine_impulse_boost = (self.sine_impulse_boost
                            + impulse * SINE_IMPULSE_BOOST_GAIN)
                            .clamp(0.0, SINE_IMPULSE_BOOST_MAX);
                    }
                    let gain = 1.0 + self.sine_impulse_boost;
                    if active && z > 0.0 {
                        self.sine_impulse_boost *= impulse_boost_decay(self.sample_dt);
                    }
                    gain
                } else {
                    bank.excitation_gain(spectral_drive)
                };
                let excitation = if active && z > 0.0 { a * z * gain } else { 0.0 };
                #[cfg(test)]
                {
                    self.radiated_amplitude = excitation;
                }
                let spectral_drive = if bank.is_sine() { 0.0 } else { spectral_drive };
                let carrier =
                    excitation * bank.phase3_carrier(self.current_pitch_hz, spectral_drive, open);
                let noise =
                    bank.attack_sample(self.current_pitch_hz, tick, impulse, a * strike_gain);
                if noise == 0. {
                    carrier
                } else {
                    carrier + noise
                }
            }
            AnyBackend::Resonator(engine) => {
                let mut drive = 0.0;
                if open {
                    // Address noise by Tone identity and sample clock, never pool order.
                    let mut noise = self
                        .noise_state
                        .wrapping_add(tick.wrapping_mul(0x9E3779B97F4A7C15));
                    drive =
                        a * impulse + a * self.continuous_drive * z * z * fast_noise(&mut noise);
                }
                engine.phase3_sample(self.current_pitch_hz, drive, open)
            }
        };
        assert!(sample.is_finite(), "phase3 produced non-finite PCM");
        sample
    }
}
