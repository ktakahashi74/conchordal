use super::envelope::Envelope;
use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::phonation_engine::{OnsetKick, ToneUpdate};
use crate::life::sound::any_backend::{AnyBackend, DriveMode};
use crate::life::sound::control::{ControlRamp, ToneControlBlock};
use crate::life::sound::{BodyKind, BodySnapshot, RenderModulator, RenderModulatorSpec};
use crate::life::voice::ArticulationSignal;
use crate::scenario::lifecycle::default_decay_attack;
use std::collections::VecDeque;

const SINE_IMPULSE_BOOST_GAIN: f32 = 0.2;
const SINE_IMPULSE_BOOST_MAX: f32 = 1.0;
const SINE_IMPULSE_BOOST_DECAY_SEC: f32 = 0.08;

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
pub struct ToneAdsr {
    pub attack_sec: f32,
    pub decay_sec: f32,
    pub sustain_level: f32,
    pub release_sec: f32,
}

#[derive(Debug, Clone, Copy)]
struct PendingUpdate {
    at_tick: Tick,
    update: ToneUpdate,
}

#[derive(Debug, Clone, Copy)]
struct PendingTrigger {
    at_tick: Tick,
    energy: f32,
}

#[derive(Clone)]
struct ExcitedState {
    config: crate::config::RenderPrototypeConfig,
    amplitude: f32,
    decay: f32,
    rise_alpha: f32,
    closed: bool,
    retired: bool,
}

#[derive(Clone)]
pub struct Tone {
    excited: Option<ExcitedState>,
    backend: AnyBackend,
    render_modulator: Option<RenderModulator>,
    pending_impulse_energy: f32,
    envelope: Envelope,
    planned_kick_pending: Option<OnsetKick>,
    pending_updates: VecDeque<PendingUpdate>,
    pending_trigger: Option<PendingTrigger>,
    current_amp: f32,
    target_amp: f32,
    current_pitch_hz: f32,
    target_pitch_hz: f32,
    amp_tau_sec: f32,
    pitch_tau_sec: f32,
    amp_alpha: f32,
    pitch_alpha: f32,
    sample_dt: f32,
    continuous_drive: f32,
    noise_state: u64,
    started: bool,
    sine_impulse_boost: f32,
}

impl Tone {
    pub(crate) fn set_representative_modulator(&mut self, spec: RenderModulatorSpec) {
        assert!(!self.started);
        self.render_modulator = Some(RenderModulator::from_spec(spec));
    }
    /// Reset a probe of this same body without rebuilding its allocated backend.
    pub(crate) fn reset_representative(&mut self, template: &Self, pitch_hz: f32) {
        assert!(!template.started && template.pending_updates.is_empty());
        self.backend.reset_representative();
        self.render_modulator = template.render_modulator.clone();
        self.pending_impulse_energy = template.pending_impulse_energy;
        self.envelope = template.envelope;
        self.planned_kick_pending = template.planned_kick_pending;
        self.pending_updates.clear();
        self.pending_trigger = template.pending_trigger;
        self.current_amp = template.current_amp;
        self.target_amp = template.target_amp;
        self.current_pitch_hz = pitch_hz;
        self.target_pitch_hz = pitch_hz;
        self.amp_tau_sec = template.amp_tau_sec;
        self.pitch_tau_sec = template.pitch_tau_sec;
        self.amp_alpha = template.amp_alpha;
        self.pitch_alpha = template.pitch_alpha;
        self.sample_dt = template.sample_dt;
        self.continuous_drive = template.continuous_drive;
        self.noise_state = template.noise_state;
        self.started = false;
        self.sine_impulse_boost = template.sine_impulse_boost;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        time: Timebase,
        onset: Tick,
        duration: Tick,
        freq_hz: f32,
        amp: f32,
        body: Option<BodySnapshot>,
        render_modulator: Option<RenderModulatorSpec>,
        adsr: Option<ToneAdsr>,
    ) -> Option<Self> {
        if duration == 0 || freq_hz <= 0.0 {
            return None;
        }
        if !freq_hz.is_finite() || !amp.is_finite() {
            return None;
        }
        if amp == 0.0 {
            return None;
        }

        let snapshot = body.unwrap_or_else(default_body_snapshot);
        let amp_scale = snapshot.amp_scale;
        let amp = amp * amp_scale.clamp(0.0, 1.0);
        if !amp.is_finite() || amp <= 0.0 {
            return None;
        }

        let backend = AnyBackend::from_snapshot(time.fs, &snapshot).ok()?;

        let (attack_ticks, decay_ticks, sustain_level, release_ticks) = if let Some(adsr) = adsr {
            let attack = sec_to_tick_at_least_one(time, adsr.attack_sec.max(0.0));
            let s = adsr.sustain_level.clamp(0.0, 1.0);
            let decay = if s < 1.0 {
                sec_to_tick_at_least_one(time, adsr.decay_sec.max(0.0))
            } else {
                0
            };
            let release = sec_to_tick_at_least_one(time, adsr.release_sec.max(0.0));
            (attack, decay, s, release)
        } else {
            (
                default_attack_ticks(time),
                0,
                1.0f32,
                default_release_ticks(time),
            )
        };
        let decay_lambda = if decay_ticks > 0 {
            6.908 / decay_ticks as f32
        } else {
            0.0
        };

        let (hold_end, release_end) = if duration == Tick::MAX {
            (Tick::MAX, Tick::MAX)
        } else {
            let hold_end = onset.saturating_add(duration);
            let release_end = hold_end.saturating_add(release_ticks);
            (hold_end, release_end)
        };

        let sample_dt = if time.fs.is_finite() && time.fs > 0.0 {
            1.0 / time.fs
        } else {
            0.0
        };
        let current_amp = amp.max(0.0);
        let target_amp = current_amp;
        let current_pitch_hz = freq_hz;
        let target_pitch_hz = current_pitch_hz;
        let amp_tau_sec = 0.0;
        let pitch_tau_sec = 0.0;
        let amp_alpha = smoothing_alpha(sample_dt, amp_tau_sec);
        let pitch_alpha = smoothing_alpha(sample_dt, pitch_tau_sec);

        Some(Self {
            excited: None,
            backend,
            render_modulator: render_modulator.map(RenderModulator::from_spec),
            pending_impulse_energy: 0.0,
            envelope: Envelope {
                onset,
                hold_end,
                release_end,
                attack_ticks,
                decay_ticks,
                sustain_level,
                decay_lambda,
                release_ticks,
            },
            planned_kick_pending: None,
            // Pre-sized so the common worker_loop insert path does not reallocate.
            // Deeper queues still grow; the capacity is a budget, not a bound.
            pending_updates: VecDeque::with_capacity(16),
            pending_trigger: None,
            current_amp,
            target_amp,
            current_pitch_hz,
            target_pitch_hz,
            amp_tau_sec,
            pitch_tau_sec,
            amp_alpha,
            pitch_alpha,
            sample_dt,
            continuous_drive: 0.0,
            noise_state: 0x9E3779B97F4A7C15_u64.wrapping_add(onset),
            started: false,
            sine_impulse_boost: 0.0,
        })
    }

    pub fn seed_modal_phases(&mut self, seed: u64) {
        self.backend.seed_modal_phases(seed);
        if self.excited.is_some() {
            self.noise_state = seed;
        }
    }

    pub(crate) fn enable_phase3(&mut self, config: crate::config::RenderPrototypeConfig) {
        let tau = config.decay_t60_sec / (3.0 * std::f32::consts::LN_10);
        self.excited = Some(ExcitedState {
            config,
            amplitude: 0.0,
            decay: (-self.sample_dt / tau).exp(),
            rise_alpha: -(-self.sample_dt / config.rise_tau_sec.unwrap_or(tau)).exp_m1(),
            closed: false,
            retired: false,
        });
        if let AnyBackend::Oscillator(bank) = &mut self.backend {
            bank.phase3_motion_scale(config.motion_scale);
        }
    }

    pub(crate) fn supports_self_model(&self) -> bool {
        self.excited.is_none()
    }

    pub(crate) fn residual_bound(&self, now: Tick) -> Option<f64> {
        let state = self.excited.as_ref()?;
        if now < self.envelope.release_end {
            return None;
        }
        if !self.started {
            return Some(0.0);
        }
        if !state.closed {
            return None;
        }
        Some(match &self.backend {
            AnyBackend::Oscillator(bank) => {
                f64::from(state.amplitude) * bank.phase3_carrier_bound()
            }
            AnyBackend::Resonator(engine) => engine.phase3_residual_bound(),
        })
    }

    pub(crate) fn retire_phase3(&mut self) {
        if let Some(state) = &mut self.excited {
            state.retired = true;
        }
    }

    pub fn note_off(&mut self, tick: Tick) {
        self.envelope = self.envelope.with_release(tick);
    }

    pub fn note_on(&mut self, tick: Tick) {
        if tick > self.envelope.onset {
            self.envelope.onset = tick;
            if self.envelope.hold_end < self.envelope.onset {
                self.envelope.hold_end = self.envelope.onset;
                self.envelope.release_end = self
                    .envelope
                    .hold_end
                    .saturating_add(self.envelope.release_ticks);
            }
        }
    }

    pub fn arm_onset_trigger(&mut self, energy: f32) {
        if !energy.is_finite() || energy <= 0.0 {
            return;
        }
        self.pending_trigger = Some(PendingTrigger {
            at_tick: self.envelope.onset,
            energy,
        });
    }

    pub fn trigger_impulse(&mut self, energy: f32) {
        if self.excited.as_ref().is_some_and(|s| s.closed || s.retired) {
            return;
        }
        if !energy.is_finite() || energy <= 0.0 {
            return;
        }
        self.pending_impulse_energy += energy;
    }

    pub fn kick_planned(&mut self, kick: OnsetKick) -> bool {
        if let Some(render_modulator) = self.render_modulator.as_mut() {
            render_modulator.kick_planned(kick);
            return true;
        }
        false
    }

    pub fn schedule_planned_kick(&mut self, kick: OnsetKick) {
        self.planned_kick_pending = Some(kick);
    }

    pub fn schedule_update(&mut self, at_tick: Tick, update: ToneUpdate) -> bool {
        if self.excited.is_some()
            && (at_tick >= self.envelope.release_end
                || self.excited.as_ref().is_some_and(|s| s.closed || s.retired)
                || update
                    .target_freq_hz
                    .is_some_and(|v| !v.is_finite() || v <= 0.0)
                || update.target_amp.is_some_and(|v| !v.is_finite() || v < 0.0)
                || update
                    .continuous_drive
                    .is_some_and(|v| !v.is_finite() || v < 0.0))
        {
            return false;
        }
        if update.is_empty() {
            return true;
        }
        let insert_at = self
            .pending_updates
            .iter()
            .position(|pending| pending.at_tick > at_tick)
            .unwrap_or(self.pending_updates.len());
        self.pending_updates
            .insert(insert_at, PendingUpdate { at_tick, update });
        true
    }

    pub fn apply_updates_if_due(&mut self, tick: Tick) {
        if self.excited.is_some() && tick >= self.envelope.release_end {
            self.pending_updates.clear();
            return;
        }
        while let Some(pending) = self.pending_updates.pop_front() {
            if pending.at_tick > tick {
                self.pending_updates.push_front(pending);
                break;
            }
            self.apply_update(&pending.update);
        }
        if self.excited.is_none() && tick >= self.envelope.hold_end {
            self.pending_updates.clear();
        }
    }

    pub fn kick_planned_if_due(&mut self, tick: Tick) -> bool {
        if self.excited.is_some() && tick >= self.envelope.release_end {
            self.planned_kick_pending = None;
            return false;
        }
        let Some(kick) = self.planned_kick_pending else {
            return false;
        };
        if tick >= self.envelope.onset {
            self.planned_kick_pending = None;
            return self.kick_planned(kick);
        }
        false
    }

    pub fn render_tick(&mut self, tick: Tick, _fs: f32, dt: f32, rhythms: &NeuralRhythms) -> f32 {
        if self.excited.is_some() {
            return self.render_excited_tick(tick, dt, rhythms);
        }
        if let Some(trigger) = self.pending_trigger
            && tick >= trigger.at_tick
        {
            self.pending_trigger = None;
            self.trigger_impulse(trigger.energy);
        }
        self.advance_smoothing();
        let gain = self.gain_at(tick);

        let mut signal = if let Some(render_modulator) = self.render_modulator.as_mut() {
            render_modulator.process(rhythms, dt)
        } else {
            ArticulationSignal {
                amplitude: 1.0,
                is_active: true,
                relaxation: rhythms.theta.alpha,
                tension: rhythms.theta.beta,
            }
        };
        if self.render_modulator.is_none() {
            let tension = signal.tension.clamp(0.0, 1.0);
            signal.amplitude *= 1.0 + 0.05 * tension;
        }
        signal.amplitude *= gain;
        signal.is_active = signal.is_active && signal.amplitude > 0.0;

        let impulse = self.pending_impulse_energy;
        self.pending_impulse_energy = 0.0;
        if impulse > 0.0 {
            self.started = true;
            if self.backend.is_sine() {
                self.sine_impulse_boost = (self.sine_impulse_boost
                    + impulse * SINE_IMPULSE_BOOST_GAIN)
                    .clamp(0.0, SINE_IMPULSE_BOOST_MAX);
            }
        }
        if !self.started {
            return 0.0;
        }

        let drive = match self.backend.drive_mode() {
            DriveMode::None => 0.0,
            DriveMode::Deterministic => impulse + self.continuous_drive * signal.amplitude,
            DriveMode::Noisy => {
                let noise = fast_noise(&mut self.noise_state);
                impulse + self.continuous_drive * signal.amplitude * noise
            }
        };
        let ctrl = ToneControlBlock {
            pitch_hz: ControlRamp {
                start: self.current_pitch_hz.max(1.0),
                step: 0.0,
            },
            amp: ControlRamp {
                start: self.current_amp.max(0.0),
                step: 0.0,
            },
        };
        let mut out = [0.0f32; 1];
        self.backend
            .render_block(std::slice::from_ref(&drive), ctrl, &mut out);
        if !signal.is_active {
            return 0.0;
        }
        let mut sample = out[0] * signal.amplitude;
        if self.backend.is_sine() {
            sample *= 1.0 + self.sine_impulse_boost;
            self.sine_impulse_boost *= impulse_boost_decay(self.sample_dt);
        }
        sample
    }

    fn render_excited_tick(&mut self, tick: Tick, dt: f32, rhythms: &NeuralRhythms) -> f32 {
        if tick < self.envelope.onset || self.excited.as_ref().unwrap().retired {
            return 0.0;
        }
        let open = tick < self.envelope.release_end;
        if !open {
            let state = self.excited.as_mut().unwrap();
            state.closed = true;
            self.pending_updates.clear();
            self.pending_trigger = None;
            self.planned_kick_pending = None;
            self.pending_impulse_energy = 0.0;
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
        let mut spectral_drive = 0.0;
        if open {
            let signal = self.render_modulator.as_mut().map_or_else(
                || ArticulationSignal {
                    amplitude: 1.0 + 0.05 * rhythms.theta.beta.clamp(0.0, 1.0),
                    is_active: true,
                    relaxation: rhythms.theta.alpha,
                    tension: rhythms.theta.beta,
                },
                |m| m.process(rhythms, dt),
            );
            // Preserve the legacy harmonic burst independently of amplitude kick.
            spectral_drive =
                impulse + self.continuous_drive * (signal.amplitude * self.envelope.gain_at(tick));
            if signal.is_active {
                z = self.envelope.gain_at(tick) * signal.amplitude.max(0.0);
            }
        }
        let a = self.current_amp;
        let qz = self.continuous_drive * z;
        let state = self.excited.as_mut().unwrap();
        let sample = match &mut self.backend {
            AnyBackend::Oscillator(bank) => {
                let target = if open {
                    a * z
                        * if bank.is_sine() {
                            1.0
                        } else {
                            1.0 + 0.25 * qz.clamp(0.0, 4.0)
                        }
                } else {
                    0.0
                };
                if open {
                    let alpha = if target > state.amplitude {
                        state.rise_alpha
                    } else {
                        1.0 - state.decay
                    };
                    state.amplitude += alpha * (target - state.amplitude);
                    state.amplitude += state.config.kick_gain * a * z * impulse;
                } else {
                    state.amplitude *= state.decay;
                }
                let spectral_drive = if bank.is_sine() { qz } else { spectral_drive };
                state.amplitude
                    * bank.phase3_carrier(self.current_pitch_hz, a, spectral_drive, open)
            }
            AnyBackend::Resonator(engine) => {
                let mut drive = 0.0;
                if open {
                    // Address noise by Tone identity and sample clock, never pool order.
                    let mut noise = self
                        .noise_state
                        .wrapping_add(tick.wrapping_mul(0x9E3779B97F4A7C15));
                    drive = state.config.kick_gain * a * impulse
                        + a * self.continuous_drive * z * z * fast_noise(&mut noise);
                }
                engine.phase3_sample(self.current_pitch_hz, drive, open)
            }
        };
        assert!(sample.is_finite(), "phase3 produced non-finite PCM");
        sample
    }

    pub fn render_block(
        &mut self,
        start_tick: Tick,
        fs: f32,
        dt: f32,
        rhythms: &mut NeuralRhythms,
        out: &mut [f32],
    ) {
        let end = start_tick.saturating_add(out.len() as Tick);
        let mut tick = start_tick;
        for sample in out.iter_mut() {
            *sample = self.render_tick(tick, fs, dt, rhythms);
            rhythms.advance_in_place(dt);
            tick = tick.saturating_add(1);
            if tick >= end {
                break;
            }
        }
    }

    pub fn set_continuous_drive(&mut self, level: f32) {
        self.continuous_drive = if self.backend.supports_continuous_drive() {
            if level.is_finite() {
                level.max(0.0)
            } else {
                0.0
            }
        } else {
            0.0
        };
    }

    pub fn set_smoothing_tau_sec(&mut self, tau_sec: f32) {
        let tau = if tau_sec.is_finite() {
            tau_sec.max(0.0)
        } else {
            0.0
        };
        self.amp_tau_sec = tau;
        self.pitch_tau_sec = tau;
        self.amp_alpha = smoothing_alpha(self.sample_dt, self.amp_tau_sec);
        self.pitch_alpha = smoothing_alpha(self.sample_dt, self.pitch_tau_sec);
        if self.amp_alpha >= 1.0 {
            self.current_amp = self.target_amp;
        }
        if self.pitch_alpha >= 1.0 {
            self.current_pitch_hz = self.target_pitch_hz;
        }
    }

    #[cfg(test)]
    pub(crate) fn debug_target_amp(&self) -> f32 {
        self.target_amp
    }

    #[cfg(test)]
    pub(crate) fn debug_current_amp(&self) -> f32 {
        self.current_amp
    }

    #[cfg(test)]
    pub(crate) fn debug_target_freq_hz(&self) -> f32 {
        self.target_pitch_hz
    }

    #[cfg(test)]
    pub(crate) fn debug_current_freq_hz(&self) -> f32 {
        self.current_pitch_hz
    }

    pub(crate) fn prediction_sine(&self, now: Tick) -> Option<super::sine_forecast::SineForecast> {
        if !self.supports_self_model() {
            return None;
        }
        // Log-domain smoothing can settle one rounding step away from its target.
        if (self.current_pitch_hz != self.target_pitch_hz
            && self.next_pitch_hz() != self.current_pitch_hz)
            || self
                .pending_updates
                .iter()
                .take(5)
                .any(|u| u.update.target_freq_hz.is_some() || u.update.continuous_drive.is_some())
            || (self.started
                && (self.pending_trigger.is_some() || self.planned_kick_pending.is_some()))
            || (self.pending_impulse_energy > 0. && self.pending_trigger.is_some())
        {
            return None;
        }
        let AnyBackend::Oscillator(backend) = &self.backend else {
            return None;
        };
        let (state, rotation) = backend.sine_state(if self.started {
            self.current_pitch_hz
        } else {
            self.settled_pitch_hz()
        })?;
        let first_sample = if self.started || self.pending_impulse_energy > 0. {
            now
        } else {
            self.pending_trigger?.at_tick.max(now)
        };
        if first_sample < self.envelope.onset {
            return None;
        }
        let impulse = if self.pending_impulse_energy > 0. {
            self.pending_impulse_energy
        } else {
            self.pending_trigger.map_or(0., |t| t.energy)
        };
        Some(super::sine_forecast::SineForecast {
            first_sample,
            state,
            rotation,
            boost: (self.sine_impulse_boost + impulse * SINE_IMPULSE_BOOST_GAIN)
                .clamp(0., SINE_IMPULSE_BOOST_MAX),
            boost_decay: impulse_boost_decay(self.sample_dt),
        })
    }

    /// Carrier lanes of the harmonic and modal backends under the same static-tone
    /// conditions as `prediction_sine`. Continuous drive is not forecast.
    pub(crate) fn prediction_bank(&self, now: Tick) -> Option<super::bank_forecast::BankForecast> {
        if !self.supports_self_model() {
            return None;
        }
        if (self.current_pitch_hz != self.target_pitch_hz
            && self.next_pitch_hz() != self.current_pitch_hz)
            || self.continuous_drive != 0.
            || self
                .pending_updates
                .iter()
                .take(5)
                .any(|u| u.update.target_freq_hz.is_some() || u.update.continuous_drive.is_some())
            || (self.started
                && (self.pending_trigger.is_some() || self.planned_kick_pending.is_some()))
            || (self.pending_impulse_energy > 0. && self.pending_trigger.is_some())
        {
            return None;
        }
        let (first_sample, impulse) = if self.started {
            // A direct impulse would re-excite the ringing bank on the next sample.
            if self.pending_impulse_energy > 0. {
                return None;
            }
            (now, None)
        } else if self.pending_impulse_energy > 0. {
            (now, Some(self.pending_impulse_energy))
        } else {
            let trigger = self.pending_trigger?;
            (trigger.at_tick.max(now), Some(trigger.energy))
        };
        if first_sample < self.envelope.onset {
            return None;
        }
        let pitch = if self.started {
            self.current_pitch_hz
        } else {
            self.settled_pitch_hz()
        }
        .max(1.);
        match &self.backend {
            AnyBackend::Oscillator(bank) => bank.bank_forecast(pitch, first_sample, impulse),
            AnyBackend::Resonator(engine) => engine.bank_forecast(pitch, first_sample, impulse),
        }
    }

    pub(crate) fn prediction_control(
        &self,
        now: Tick,
        rhythms: &NeuralRhythms,
    ) -> super::control_forecast::ControlForecast {
        assert!(
            self.supports_self_model(),
            "phase3 self model unsupported: renderer"
        );
        use super::control_forecast::{
            AmplitudeModel, AmplitudeSmoothing, AmplitudeUpdate, AmplitudeUpdates, ControlForecast,
        };
        let mut updates = AmplitudeUpdates::default();
        let mut valid_until = None;
        for (scanned, pending) in self.pending_updates.iter().enumerate() {
            if scanned == updates.events.len()
                || pending.update.target_freq_hz.is_some()
                || pending.update.continuous_drive.is_some()
            {
                valid_until = Some(pending.at_tick.max(now));
                break;
            }
            if let Some(target) = pending.update.target_amp.filter(|a| a.is_finite()) {
                updates.events[updates.len] = AmplitudeUpdate {
                    at_sample: pending.at_tick.max(now),
                    target: target.max(0.),
                };
                updates.len += 1;
            }
        }
        ControlForecast {
            issued_at: now,
            valid_until,
            amplitude_smoothing: (self.current_amp != self.target_amp || updates.len > 0)
                .then_some(AmplitudeSmoothing {
                    current: self.current_amp,
                    target: self.target_amp,
                    alpha: self.amp_alpha,
                }),
            amplitude_updates: (updates.len > 0).then_some(updates),
            sample_dt: self.sample_dt,
            starts_at: if self.started || self.pending_impulse_energy > 0. {
                Some(now)
            } else {
                self.pending_trigger.map(|t| t.at_tick.max(now))
            },
            kick_at: self
                .planned_kick_pending
                .filter(|k| k.strength > 0.)
                .map(|_| self.envelope.onset.max(now)),
            model: self.render_modulator.as_ref().map_or(
                AmplitudeModel::Unmodulated {
                    gain: 1. + 0.05 * rhythms.theta.beta.clamp(0., 1.),
                },
                |m| m.amplitude_model(rhythms),
            ),
        }
    }

    pub(crate) fn prediction_parameters(&self, release: Option<Tick>) -> (f32, f32, Envelope) {
        assert!(
            self.supports_self_model(),
            "phase3 self model unsupported: renderer"
        );
        let envelope = release.map_or(self.envelope, |at| self.envelope.with_release(at));
        (self.current_pitch_hz, self.current_amp, envelope)
    }

    pub fn end_tick(&self) -> Tick {
        if self.excited.is_some() {
            return Tick::MAX;
        }
        self.envelope.release_end
    }

    pub fn onset(&self) -> Tick {
        self.envelope.onset
    }

    pub fn is_done(&self, now: Tick) -> bool {
        if let Some(state) = &self.excited {
            return state.retired;
        }
        now >= self.envelope.release_end
    }

    fn apply_update(&mut self, update: &ToneUpdate) {
        if let Some(freq_hz) = update.target_freq_hz
            && freq_hz.is_finite()
            && freq_hz > 0.0
        {
            self.target_pitch_hz = freq_hz;
            if self.pitch_alpha >= 1.0 {
                self.current_pitch_hz = freq_hz;
            }
        }
        if let Some(amp) = update.target_amp
            && amp.is_finite()
        {
            let amp = amp.max(0.0);
            self.target_amp = amp;
            if self.amp_alpha >= 1.0 {
                self.current_amp = amp;
            }
        }
        if let Some(drive) = update.continuous_drive {
            self.set_continuous_drive(drive);
        }
    }

    /// The pitch the next smoothing step would render.
    fn next_pitch_hz(&self) -> f32 {
        self.pitch_step_from(self.current_pitch_hz)
    }

    fn pitch_step_from(&self, current: f32) -> f32 {
        let next = if self.pitch_alpha >= 1.0 || current <= 0.0 || self.target_pitch_hz <= 0.0 {
            self.target_pitch_hz
        } else {
            smooth_step(current.ln(), self.target_pitch_hz.ln(), self.pitch_alpha).exp()
        };
        if next.is_finite() && next > 0.0 {
            next
        } else {
            self.target_pitch_hz
        }
    }

    /// The pitch an unstarted tone renders: log-domain smoothing rounds a static pitch onto
    /// its own fixed point within a few samples, and a carrier forecast must turn at it.
    fn settled_pitch_hz(&self) -> f32 {
        let mut pitch = self.current_pitch_hz;
        for _ in 0..8 {
            let next = self.pitch_step_from(pitch);
            if next == pitch {
                break;
            }
            pitch = next;
        }
        pitch
    }

    fn advance_smoothing(&mut self) {
        self.current_amp = smooth_step(self.current_amp, self.target_amp, self.amp_alpha);
        if !self.current_amp.is_finite() {
            self.current_amp = self.target_amp;
        }
        self.current_amp = self.current_amp.max(0.0);

        // Smooth pitch in log2 space so equal semitone steps take equal time.
        self.current_pitch_hz = self.next_pitch_hz();
    }

    fn gain_at(&self, tick: Tick) -> f32 {
        self.envelope.gain_at(tick)
    }
}

/// Fast per-sample noise via splitmix64, returns value in [-1, 1].
fn fast_noise(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    // Map to [-1, 1]
    (z as i64 as f64 / i64::MAX as f64) as f32
}

fn smooth_step(current: f32, target: f32, alpha: f32) -> f32 {
    if !current.is_finite() {
        return target;
    }
    if !target.is_finite() {
        return current;
    }
    current + alpha * (target - current)
}

fn smoothing_alpha(dt: f32, tau_sec: f32) -> f32 {
    if !dt.is_finite() || dt <= 0.0 {
        return 1.0;
    }
    if !tau_sec.is_finite() || tau_sec <= 0.0 {
        return 1.0;
    }
    let alpha = 1.0 - (-dt / tau_sec).exp();
    if alpha.is_finite() {
        alpha.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

fn impulse_boost_decay(dt: f32) -> f32 {
    if !dt.is_finite() || dt <= 0.0 {
        return 0.0;
    }
    (-dt / SINE_IMPULSE_BOOST_DECAY_SEC).exp().clamp(0.0, 1.0)
}

pub fn default_release_ticks(time: Timebase) -> Tick {
    let release_sec = default_decay_attack();
    sec_to_tick_at_least_one(time, release_sec)
}

fn default_attack_ticks(time: Timebase) -> Tick {
    let attack_sec = default_decay_attack();
    sec_to_tick_at_least_one(time, attack_sec)
}

fn sec_to_tick_at_least_one(time: Timebase, sec: f32) -> Tick {
    if !sec.is_finite() || sec <= 0.0 {
        return 1;
    }
    let ticks = time.sec_to_tick(sec);
    if ticks < 1 { 1 } else { ticks }
}

fn default_body_snapshot() -> BodySnapshot {
    BodySnapshot {
        kind: BodyKind::Sine,
        amp_scale: 1.0,
        brightness: 0.0,
        inharmonic: 0.0,
        spread: 0.0,
        unison: 1,
        motion: 0.0,
        ratios: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase3_harmonic_spectral_burst_matches_legacy_and_freezes_at_off() {
        let time = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let rhythms = NeuralRhythms::default();
        for attack_sec in [0.0, 0.005, 0.010, 0.024, 0.2] {
            for continuous_drive in [0.0, 1.25] {
                for kick_gain in [0.0, 1.0] {
                    let mut old = Tone::from_parts(
                        time,
                        0,
                        time.sec_to_tick(0.4),
                        220.0,
                        0.2,
                        Some(BodySnapshot {
                            kind: BodyKind::Harmonic,
                            brightness: 0.7,
                            ..default_body_snapshot()
                        }),
                        None,
                        Some(ToneAdsr {
                            attack_sec,
                            decay_sec: 0.1,
                            sustain_level: 0.55,
                            release_sec: 0.1,
                        }),
                    )
                    .unwrap();
                    old.set_continuous_drive(continuous_drive);
                    old.seed_modal_phases(73);
                    old.arm_onset_trigger(1.0);
                    let mut new = old.clone();
                    new.enable_phase3(crate::config::RenderPrototypeConfig {
                        kick_gain,
                        ..Default::default()
                    });
                    let off = new.envelope.release_end;
                    let mut frozen = None;
                    let mut checkpoints = Vec::new();
                    let mut frozen_gains = Vec::new();
                    let mut tail_nonzero = false;
                    for tick in 0..off + time.sec_to_tick(0.1) {
                        // The driven case also checks an independent later spectral burst.
                        if continuous_drive > 0.0 && tick == time.sec_to_tick(0.24) {
                            old.trigger_impulse(0.4);
                            new.trigger_impulse(0.4);
                        }
                        if tick == off + 1 {
                            new.trigger_impulse(1.0);
                        }
                        let old_pcm = old.render_tick(tick, time.fs, 1.0 / time.fs, &rhythms);
                        let new_pcm = new.render_tick(tick, time.fs, 1.0 / time.fs, &rhythms);
                        assert!(old_pcm.is_finite() && new_pcm.is_finite());
                        let AnyBackend::Oscillator(old_bank) = &old.backend else {
                            unreachable!()
                        };
                        let AnyBackend::Oscillator(new_bank) = &new.backend else {
                            unreachable!()
                        };
                        let (old_env, _) = old_bank.spectral_state_for_test();
                        let (new_env, new_gains) = new_bank.spectral_state_for_test();
                        if tick < off {
                            assert_eq!(old_env, new_env, "spectral state at tick {tick}");
                            if tick == 0 {
                                assert_eq!(new_env, 1.0, "ADSR suppressed the onset burst");
                            }
                            if tick == off - 1 {
                                frozen = Some(new_env);
                                frozen_gains.extend_from_slice(new_gains);
                            }
                        } else {
                            assert_eq!(Some(new_env), frozen, "Off changed spectral state");
                            assert_eq!(new_gains, frozen_gains, "Off changed partial gains");
                            tail_nonzero |= new_pcm != 0.0;
                        }
                        if [0, 3840, 9600, off - 1, off, off + 4799].contains(&tick) {
                            checkpoints.push(serde_json::json!({
                                "tick": tick, "old_env": old_env, "new_env": new_env,
                            }));
                        }
                    }
                    assert!(
                        tail_nonzero,
                        "the frozen check must exercise a sounding tail"
                    );
                    println!(
                        "M3R_SPECTRAL {}",
                        serde_json::json!({
                            "attack_sec": attack_sec, "continuous_drive": continuous_drive,
                            "kick_gain": kick_gain, "open_samples": off,
                            "closed_samples": 4800, "open_max_abs_difference": 0.0,
                            "closed_max_abs_state_change": 0.0,
                            "closed_max_abs_gain_change": 0.0, "checkpoints": checkpoints,
                        })
                    );
                }
            }
        }
    }

    #[test]
    fn phase3_default_rise_follows_composer_attack() {
        for fs in [44_100.0, 48_000.0, 96_000.0] {
            let time = Timebase { fs, hop: 64 };
            for attack_sec in [0.0, 0.001, 0.01, 0.8] {
                let adsr = ToneAdsr {
                    attack_sec,
                    decay_sec: 0.0,
                    sustain_level: 1.0,
                    release_sec: 0.1,
                };
                let mut tone = Tone::from_parts(
                    time,
                    0,
                    Tick::MAX,
                    220.0,
                    0.2,
                    Some(default_body_snapshot()),
                    None,
                    Some(adsr),
                )
                .unwrap();
                // Isolate attack from the still-unselected onset coupling.
                tone.enable_phase3(crate::config::RenderPrototypeConfig {
                    kick_gain: 0.0,
                    ..Default::default()
                });
                tone.arm_onset_trigger(1.0);
                for tick in 0..=time.sec_to_tick(attack_sec) {
                    let sample = tone.render_tick(tick, fs, 1.0 / fs, &NeuralRhythms::default());
                    let expected = 0.2 * tone.envelope.gain_at(tick);
                    let actual = tone.excited.as_ref().unwrap().amplitude;
                    assert!(sample.is_finite());
                    assert!((actual - expected).abs() <= f32::EPSILON * 0.2);
                }
            }
        }
    }

    #[test]
    fn phase3_preserves_explicit_motion_and_has_no_implicit_motion() {
        let time = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let adsr = ToneAdsr {
            attack_sec: 0.0,
            decay_sec: 0.0,
            sustain_level: 1.0,
            release_sec: 0.1,
        };
        let make = |motion, scale| {
            let mut tone = Tone::from_parts(
                time,
                0,
                Tick::MAX,
                220.0,
                0.2,
                Some(BodySnapshot {
                    kind: BodyKind::Harmonic,
                    motion,
                    ..default_body_snapshot()
                }),
                None,
                Some(adsr),
            )
            .unwrap();
            tone.enable_phase3(crate::config::RenderPrototypeConfig {
                motion_scale: scale,
                ..Default::default()
            });
            tone.seed_modal_phases(73);
            tone.arm_onset_trigger(1.0);
            tone
        };
        let scale = crate::config::RenderPrototypeConfig::default().motion_scale;
        let motion = crate::scenario::control::TimbreControl::default().motion;
        assert_eq!(motion, 0.0);
        let mut default = make(motion, scale);
        let mut zero = make(0.0, 0.0);
        let mut explicit = make(0.2, scale);
        let mut differs = false;
        for tick in 0..24_000 {
            let rhythms = NeuralRhythms::default();
            let sample = default.render_tick(tick, time.fs, 1.0 / time.fs, &rhythms);
            assert_eq!(
                sample,
                zero.render_tick(tick, time.fs, 1.0 / time.fs, &rhythms)
            );
            differs |= sample != explicit.render_tick(tick, time.fs, 1.0 / time.fs, &rhythms);
        }
        assert!(differs, "explicit composer motion was erased");
    }

    #[test]
    fn phase3_hold_release_pitch_and_future_residual_are_separate() {
        for fs in [44_100.0, 48_000.0, 96_000.0] {
            for kind in [BodyKind::Sine, BodyKind::Harmonic, BodyKind::Modal] {
                let time = Timebase { fs, hop: 64 };
                let body = BodySnapshot {
                    kind,
                    motion: 0.2,
                    ..default_body_snapshot()
                };
                let adsr = ToneAdsr {
                    attack_sec: 0.001,
                    decay_sec: 0.0,
                    sustain_level: 1.0,
                    release_sec: 0.005,
                };
                let mut tone =
                    Tone::from_parts(time, 0, Tick::MAX, 220.0, 0.2, Some(body), None, Some(adsr))
                        .unwrap();
                tone.enable_phase3(Default::default());
                tone.seed_modal_phases(73);
                tone.arm_onset_trigger(1.0);
                let close_request = time.sec_to_tick(0.2);
                let mut hold_peak = 0.0_f32;
                for tick in 0..close_request {
                    if tick == close_request / 2 {
                        assert!(tone.schedule_update(
                            tick,
                            ToneUpdate {
                                target_freq_hz: Some(330.0),
                                ..Default::default()
                            }
                        ));
                    }
                    tone.apply_updates_if_due(tick);
                    let y = tone.render_tick(tick, fs, 1.0 / fs, &NeuralRhythms::default());
                    if tick > close_request / 2 {
                        hold_peak = hold_peak.max(y.abs());
                    }
                }
                assert!(hold_peak > 1e-4, "{kind:?} sustained/struck voice vanished");
                assert_eq!(tone.debug_current_freq_hz(), 330.0);
                tone.note_off(close_request);
                let close = tone.envelope.release_end;
                // A release-ramp Update is still an open-handle operation.
                assert!(tone.schedule_update(
                    close - 1,
                    ToneUpdate {
                        target_freq_hz: Some(440.0),
                        ..Default::default()
                    }
                ));
                for tick in close_request..=close {
                    tone.apply_updates_if_due(tick);
                    tone.render_tick(tick, fs, 1.0 / fs, &NeuralRhythms::default());
                }
                assert_eq!(tone.debug_current_freq_hz(), 440.0);
                assert!(!tone.is_done(close));
                let bound = tone.residual_bound(close + 1).unwrap();
                assert!(bound.is_finite() && bound > f64::from(crate::life::voice::Voice::AMP_EPS));
                let mut control = tone.clone();
                assert!(!tone.schedule_update(
                    close + 1,
                    ToneUpdate {
                        target_freq_hz: Some(880.0),
                        target_amp: Some(0.0),
                        continuous_drive: Some(10.0)
                    }
                ));
                tone.trigger_impulse(10.0);
                let mut tail_peak = 0.0_f32;
                for tick in close + 1..close + time.sec_to_tick(1.3) {
                    tone.apply_updates_if_due(tick);
                    let y = tone.render_tick(tick, fs, 1.0 / fs, &NeuralRhythms::default());
                    let expected =
                        control.render_tick(tick, fs, 1.0 / fs, &NeuralRhythms::default());
                    assert_eq!(y.to_bits(), expected.to_bits());
                    assert!(
                        f64::from(y.abs()) <= bound,
                        "future PCM exceeded {kind:?} bound"
                    );
                    tail_peak = tail_peak.max(y.abs());
                }
                assert!(tail_peak > 1e-6, "old release_end cut off the free tail");
                assert_eq!(tone.debug_current_freq_hz(), 440.0);
                assert!(
                    tone.residual_bound(close + time.sec_to_tick(1.3)).unwrap()
                        < f64::from(crate::life::voice::Voice::AMP_EPS)
                );
                assert!(!tone.supports_self_model());
                assert!(tone.prediction_sine(close).is_none());
                assert!(tone.prediction_bank(close).is_none());
            }
        }
    }

    #[test]
    fn phase3_rejects_entire_invalid_update_and_prevents_old_energy_prior() {
        let time = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let mut tone = Tone::from_parts(time, 0, Tick::MAX, 220.0, 0.2, None, None, None).unwrap();
        tone.enable_phase3(Default::default());
        assert!(!tone.schedule_update(
            0,
            ToneUpdate {
                target_freq_hz: Some(440.0),
                target_amp: Some(f32::NAN),
                continuous_drive: None
            }
        ));
        tone.apply_updates_if_due(0);
        assert_eq!(tone.debug_target_freq_hz(), 220.0);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || tone.prediction_parameters(None)
            ))
            .is_err()
        );
    }

    #[test]
    fn issued_envelope_preserves_attack_tail_and_snapshot_identity() {
        let time = Timebase { fs: 1000., hop: 4 };
        let mut tone = Tone::from_parts(
            time,
            10,
            40,
            220.,
            0.2,
            None,
            None,
            Some(ToneAdsr {
                attack_sec: 0.01,
                decay_sec: 0.,
                sustain_level: 1.,
                release_sec: 1.,
            }),
        )
        .unwrap();
        let frozen = tone.prediction_parameters(None).2;
        let released = tone.prediction_parameters(Some(15)).2;
        assert_eq!((frozen.hold_end, frozen.release_end), (50, 1050));
        assert_eq!((released.hold_end, released.release_end), (15, 1015));
        // The renderer shortens its attack to the actual hold duration.
        for (tick, expected) in [
            (9, 0.),
            (10, 0.2),
            (14, 1.),
            (15, 1.),
            (515, 0.5),
            (1015, 0.),
        ] {
            assert_eq!(released.gain_at(tick), expected);
        }
        assert_eq!(frozen.gain_at(10), 0.1);
        tone.note_off(15);
        assert_eq!(tone.prediction_parameters(None).2, released);
        assert_eq!(tone.prediction_parameters(Some(100)).2, released);
        tone.note_on(20);
        assert_eq!(frozen.onset, 10);
        assert_eq!(released.onset, 10);
        assert_eq!(tone.onset(), 20);
    }

    #[test]
    fn indefinite_envelope_release_and_saturated_end_remain_bounded() {
        let time = Timebase { fs: 1000., hop: 4 };
        let mut tone = Tone::from_parts(
            time,
            10,
            Tick::MAX,
            220.,
            0.2,
            None,
            None,
            Some(ToneAdsr {
                attack_sec: 0.01,
                decay_sec: 0.,
                sustain_level: 1.,
                release_sec: 1.,
            }),
        )
        .unwrap();
        let held = tone.prediction_parameters(None).2;
        assert_eq!((held.hold_end, held.release_end), (Tick::MAX, Tick::MAX));
        assert_eq!(held.gain_at(Tick::MAX - 1), 1.);
        let saturated = tone.prediction_parameters(Some(Tick::MAX - 10)).2;
        assert_eq!(saturated.release_end, Tick::MAX);
        assert_eq!(saturated.gain_at(Tick::MAX - 1), 0.001);
        assert_eq!(saturated.gain_at(Tick::MAX), 0.);
        tone.note_off(30);
        assert_eq!(tone.end_tick(), 1030);
        assert_eq!(tone.gain_at(530), 0.5);
        assert_eq!(tone.gain_at(1030), 0.);
        assert_eq!(held.gain_at(1030), 1.);
    }

    #[test]
    fn spawn_does_not_sound_until_triggered() {
        let tb = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let mut tone =
            Tone::from_parts(tb, 0, Tick::MAX, 440.0, 0.5, None, None, None).expect("tone");

        let mut rhythms = NeuralRhythms::default();
        let mut out = vec![0.0f32; tb.hop];
        tone.render_block(0, tb.fs, 1.0 / tb.fs, &mut rhythms, &mut out);
        assert!(out.iter().all(|s| s.abs() <= 1e-6));

        tone.trigger_impulse(1.0);
        let mut rhythms = NeuralRhythms::default();
        let mut out = vec![0.0f32; tb.hop];
        tone.render_block(0, tb.fs, 1.0 / tb.fs, &mut rhythms, &mut out);
        assert!(out.iter().any(|s| s.abs() > 1e-6));
    }

    #[test]
    fn continuous_drive_does_not_start_sine_without_impulse() {
        let tb = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let mut tone =
            Tone::from_parts(tb, 0, Tick::MAX, 440.0, 0.5, None, None, None).expect("tone");
        tone.set_continuous_drive(0.01);

        let dt = 1.0 / tb.fs;
        let blocks = 64;
        let mut last_block = vec![0.0f32; tb.hop];
        for b in 0..blocks {
            let tick = (b * tb.hop) as Tick;
            let mut rhythms = NeuralRhythms::default();
            tone.render_block(tick, tb.fs, dt, &mut rhythms, &mut last_block);
        }
        assert!(last_block.iter().all(|s| s.abs() <= 1e-6));
    }

    #[test]
    fn harmonic_continuous_drive_sustains_after_first_impulse() {
        let tb = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let harmonic = BodySnapshot {
            kind: BodyKind::Harmonic,
            amp_scale: 1.0,
            brightness: 0.6,
            inharmonic: 0.0,
            spread: 0.0,
            unison: 1,
            motion: 0.0,
            ratios: None,
        };
        let mut tone = Tone::from_parts(tb, 0, Tick::MAX, 220.0, 0.5, Some(harmonic), None, None)
            .expect("tone");
        tone.set_continuous_drive(0.03);
        tone.trigger_impulse(1.0);
        let dt = 1.0 / tb.fs;
        let blocks = (2.0 * tb.fs as f64 / tb.hop as f64) as usize;
        let mut last_block = vec![0.0f32; tb.hop];
        for b in 0..blocks {
            let tick = (b * tb.hop) as Tick;
            let mut rhythms = NeuralRhythms::default();
            tone.render_block(tick, tb.fs, dt, &mut rhythms, &mut last_block);
        }
        let peak = last_block.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(
            peak > 1e-4,
            "harmonic sustain drive should keep energy; peak={peak}"
        );
    }

    #[test]
    fn harmonic_note_without_continuous_drive_stays_audible() {
        let tb = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let harmonic = BodySnapshot {
            kind: BodyKind::Harmonic,
            amp_scale: 1.0,
            brightness: 0.6,
            inharmonic: 0.0,
            spread: 0.0,
            unison: 1,
            motion: 0.0,
            ratios: None,
        };
        let mut tone = Tone::from_parts(tb, 0, Tick::MAX, 220.0, 0.5, Some(harmonic), None, None)
            .expect("tone");
        tone.trigger_impulse(1.0);
        let dt = 1.0 / tb.fs;
        let blocks = (2.0 * tb.fs as f64 / tb.hop as f64) as usize;
        let mut last_block = vec![0.0f32; tb.hop];
        for b in 0..blocks {
            let tick = (b * tb.hop) as Tick;
            let mut rhythms = NeuralRhythms::default();
            tone.render_block(tick, tb.fs, dt, &mut rhythms, &mut last_block);
        }
        let peak = last_block.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(
            peak > 1e-4,
            "harmonic note should remain audible; peak={peak}"
        );
    }

    #[test]
    fn sine_reimpulse_adds_short_boost() {
        let tb = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let mut tone =
            Tone::from_parts(tb, 0, Tick::MAX, 440.0, 0.5, None, None, None).expect("tone");
        let dt = 1.0 / tb.fs;
        let mut rhythms = NeuralRhythms::default();

        tone.trigger_impulse(1.0);
        let mut before = vec![0.0f32; tb.hop];
        tone.render_block(0, tb.fs, dt, &mut rhythms, &mut before);
        let before_peak = before.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

        tone.trigger_impulse(1.0);
        let mut after = vec![0.0f32; tb.hop];
        tone.render_block(tb.hop as Tick, tb.fs, dt, &mut rhythms, &mut after);
        let after_peak = after.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

        assert!(
            after_peak > before_peak,
            "re-impulse should add short sine boost: before={before_peak}, after={after_peak}"
        );
    }

    #[test]
    fn release_tick_update_applies_before_pending_updates_are_cleared() {
        let tb = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let mut tone = Tone::from_parts(tb, 0, 8, 440.0, 0.5, None, None, None).expect("tone");
        tone.set_smoothing_tau_sec(0.0);
        tone.note_off(4);
        tone.schedule_update(
            4,
            ToneUpdate {
                target_freq_hz: None,
                target_amp: Some(0.25),
                continuous_drive: None,
            },
        );

        tone.apply_updates_if_due(4);

        assert!((tone.debug_target_amp() - 0.25).abs() < 1e-6);
        assert!((tone.debug_current_amp() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn forecast_bounds_pending_updates_and_retains_amplitude_smoothing() {
        let tb = Timebase {
            fs: 8000.,
            hop: 400,
        };
        let mut tone = Tone::from_parts(tb, 0, 4000, 440., 0.5, None, None, None).unwrap();
        let rhythms = NeuralRhythms::default();
        tone.trigger_impulse(1.);
        let original = tone.prediction_control(0, &rhythms);
        assert_eq!(original.valid_until, None);
        tone.set_smoothing_tau_sec(0.05);
        tone.schedule_update(
            1600,
            ToneUpdate {
                target_freq_hz: Some(220.),
                target_amp: None,
                continuous_drive: None,
            },
        );
        tone.schedule_update(
            800,
            ToneUpdate {
                target_freq_hz: None,
                target_amp: Some(0.25),
                continuous_drive: None,
            },
        );
        let bounded = tone.prediction_control(0, &rhythms);
        assert_eq!(bounded.valid_until, Some(1600));
        assert_eq!(bounded.gain_at(799), original.gain_at(799));
        assert!(bounded.gain_at(799).unwrap() > 0.);
        assert_eq!(bounded.gain_at(800), original.gain_at(800));
        assert!(bounded.amplitude_at(800, 0.5).unwrap() < 0.5);
        assert!(bounded.known_on([0, 1600]));
        assert!(!bounded.known_on([0, 1601]));
        assert_eq!(tone.prediction_control(0, &rhythms), bounded);
        assert_eq!(
            tone.prediction_control(900, &rhythms).valid_until,
            Some(1600)
        );
        assert!(tone.prediction_sine(0).is_none());
        assert_eq!(tone.pending_updates.len(), 2);
        assert_eq!(tone.debug_target_amp(), 0.5);
        tone.apply_updates_if_due(800);
        assert_eq!(tone.debug_target_amp(), 0.25);
        let smoothing = tone.prediction_control(800, &rhythms);
        assert_eq!(smoothing.amplitude_smoothing.unwrap().current, 0.5);
        assert_eq!(smoothing.amplitude_smoothing.unwrap().target, 0.25);
        assert!(smoothing.amplitude_at(800, 0.5).unwrap() < 0.5);
        assert_eq!(smoothing.amplitude_at(1600, 0.5), None);
        tone.set_smoothing_tau_sec(0.);
        tone.schedule_update(
            800,
            ToneUpdate {
                target_freq_hz: None,
                target_amp: Some(0.25),
                continuous_drive: None,
            },
        );
        tone.apply_updates_if_due(800);
        assert_eq!(
            tone.prediction_control(800, &rhythms).valid_until,
            Some(1600)
        );
        assert_eq!(tone.pending_updates.len(), 1);
    }

    #[test]
    fn scheduled_amplitude_forecast_matches_renderer_order_and_release_clearing() {
        use crate::life::tone_energy::{ScheduledRelease, ToneEnergy};
        let mut amplitude_error = 0_f64;
        let mut waveform_error = 0_f64;
        let mut cases = 0;
        for fs in [8000., 48000.] {
            for tau in [0., 0.001, 0.05, 1., f32::MAX] {
                for onset in [0, 600] {
                    for releases in [
                        [None, None],
                        [Some((400, 700)), None],
                        [None, Some((700, 400))],
                        [Some((400, 500)), None],
                        [Some((600, 100)), Some((400, 900))],
                        [Some((400, 900)), Some((600, 100))],
                    ] {
                        let mut tone = Tone::from_parts(
                            Timebase { fs, hop: 64 },
                            onset,
                            1000 - onset,
                            293.,
                            0.2,
                            None,
                            None,
                            Some(ToneAdsr {
                                attack_sec: 0.,
                                decay_sec: 0.,
                                sustain_level: 1.,
                                release_sec: 1.,
                            }),
                        )
                        .unwrap();
                        tone.seed_modal_phases(7731);
                        tone.arm_onset_trigger(1.);
                        let rhythms = NeuralRhythms::default();
                        for tick in 0..256 {
                            tone.render_tick(tick, fs, 1. / fs, &rhythms);
                        }
                        tone.set_smoothing_tau_sec(tau);
                        // Out-of-order insertion, one overdue update, and two at the same sample.
                        for (at, target) in [(1200, 0.3), (100, 0.05), (500, 0.6), (500, -0.1)] {
                            tone.schedule_update(
                                at,
                                ToneUpdate {
                                    target_freq_hz: None,
                                    target_amp: Some(target),
                                    continuous_drive: None,
                                },
                            );
                        }
                        let releases = releases.map(|r| {
                            r.map(|(apply_at_sample, off_sample)| ScheduledRelease {
                                apply_at_sample,
                                off_sample,
                            })
                        });
                        let (_, amplitude, envelope) = tone.prediction_parameters(None);
                        let frozen = ToneEnergy {
                            amplitude,
                            envelope,
                            control: Some(tone.prediction_control(256, &rhythms)),
                            sine: tone.prediction_sine(256),
                            bank: tone.prediction_bank(256),
                            scheduled_release: releases[0],
                        };
                        assert!(frozen.sine.is_some());
                        assert_eq!(frozen.control.unwrap().amplitude_updates.unwrap().len, 4);
                        let saved = tone.prediction_control(256, &rhythms);
                        let control = frozen.control_for(releases[1]).unwrap();
                        assert!(control.known_on([256, 4000]));
                        let mut untouched = tone.clone();
                        for tick in 256..4000 {
                            for release in releases.into_iter().flatten() {
                                if tick == release.apply_at_sample {
                                    tone.note_off(release.off_sample);
                                    untouched.note_off(release.off_sample);
                                }
                            }
                            tone.apply_updates_if_due(tick);
                            untouched.apply_updates_if_due(tick);
                            let actual = tone.render_tick(tick, fs, 1. / fs, &rhythms);
                            assert_eq!(actual, untouched.render_tick(tick, fs, 1. / fs, &rhythms));
                            let error = (control.amplitude_at(tick, amplitude).unwrap()
                                - f64::from(tone.current_amp))
                            .abs();
                            amplitude_error = amplitude_error.max(error);
                            assert!(
                                error < 0.001,
                                "fs={fs}, tau={tau}, onset={onset}, tick={tick}, error={error}"
                            );
                            let error = (frozen.sine_point(tick, releases[1]).unwrap().0[1]
                                - f64::from(actual))
                            .abs();
                            waveform_error = waveform_error.max(error);
                            assert!(
                                error < 0.002,
                                "fs={fs}, tau={tau}, onset={onset}, tick={tick}, error={error}"
                            );
                        }
                        assert_eq!(frozen.control.unwrap(), saved);
                        cases += 1;
                    }
                }
            }
        }
        eprintln!(
            "SCHEDULED_AMPLITUDE cases={cases} amplitude_error={amplitude_error} waveform_error={waveform_error} control_bytes={} tone_energy_bytes={}",
            std::mem::size_of::<super::super::control_forecast::ControlForecast>(),
            std::mem::size_of::<ToneEnergy>()
        );
    }

    #[test]
    fn scheduled_amplitude_overflow_and_unknown_controls_bound_support() {
        use crate::life::tone_energy::{ScheduledRelease, ToneEnergy};
        let mut tone = Tone::from_parts(
            Timebase { fs: 8000., hop: 64 },
            0,
            4000,
            293.,
            0.2,
            None,
            None,
            None,
        )
        .unwrap();
        let rhythms = NeuralRhythms::default();
        for at in [300, 400, 500, 600, 700, 800] {
            tone.schedule_update(
                at,
                ToneUpdate {
                    target_freq_hz: None,
                    target_amp: Some(0.4),
                    continuous_drive: None,
                },
            );
        }
        let (_, amplitude, envelope) = tone.prediction_parameters(None);
        let frozen = ToneEnergy {
            amplitude,
            envelope,
            control: Some(tone.prediction_control(256, &rhythms)),
            sine: None,
            bank: None,
            scheduled_release: None,
        };
        let control = frozen.control_for(None).unwrap();
        assert_eq!(control.valid_until, Some(700));
        assert!(control.known_on([256, 700]));
        assert!(!control.known_on([256, 701]));
        for (off_sample, supported) in [(699, true), (700, false), (701, false)] {
            let released = frozen
                .control_for(Some(ScheduledRelease {
                    apply_at_sample: 640,
                    off_sample,
                }))
                .unwrap();
            assert_eq!(released.known_on([256, 1000]), supported);
        }
        for update in [
            ToneUpdate {
                target_freq_hz: Some(220.),
                target_amp: Some(0.1),
                continuous_drive: None,
            },
            ToneUpdate {
                target_freq_hz: None,
                target_amp: Some(0.1),
                continuous_drive: Some(0.1),
            },
        ] {
            let mut unsupported = tone.clone();
            unsupported.schedule_update(450, update);
            assert_eq!(
                unsupported.prediction_control(256, &rhythms).valid_until,
                Some(450)
            );
        }
        tone.pending_updates.clear();
        tone.schedule_update(
            300,
            ToneUpdate {
                target_freq_hz: None,
                target_amp: Some(f32::NAN),
                continuous_drive: None,
            },
        );
        assert_eq!(
            tone.prediction_control(256, &rhythms).amplitude_updates,
            None
        );
        assert_eq!(tone.prediction_control(256, &rhythms).valid_until, None);
    }

    #[test]
    fn pending_updates_past_the_tick_stay_queued_until_due() {
        let tb = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let mut tone = Tone::from_parts(tb, 0, 1_000, 440.0, 0.5, None, None, None).expect("tone");
        tone.set_smoothing_tau_sec(0.0);
        let amp_update = |amp: f32| ToneUpdate {
            target_freq_hz: None,
            target_amp: Some(amp),
            continuous_drive: None,
        };
        tone.schedule_update(4, amp_update(0.25));
        tone.schedule_update(12, amp_update(0.75));

        tone.apply_updates_if_due(4);
        assert!(
            (tone.debug_target_amp() - 0.25).abs() < 1e-6,
            "only the due update applies"
        );

        tone.apply_updates_if_due(12);
        assert!(
            (tone.debug_target_amp() - 0.75).abs() < 1e-6,
            "the deferred update must survive the earlier pass"
        );
    }

    #[test]
    fn adsr_gain_follows_attack_decay_sustain_release() {
        let tb = Timebase { fs: 1000.0, hop: 4 };
        let adsr = ToneAdsr {
            attack_sec: 0.01,
            decay_sec: 0.02,
            sustain_level: 0.5,
            release_sec: 0.01,
        };
        let mut tone =
            Tone::from_parts(tb, 0, 100, 440.0, 1.0, None, None, Some(adsr)).expect("tone");
        tone.trigger_impulse(1.0);

        // Attack phase: gain ramps 0 → 1 over ~10 ticks
        let g_mid_attack = tone.gain_at(5);
        assert!(
            g_mid_attack > 0.0 && g_mid_attack < 1.0,
            "mid-attack: {g_mid_attack}"
        );

        // After attack (tick 10): gain should be ~1.0
        let g_attack_end = tone.gain_at(10);
        assert!(g_attack_end > 0.95, "attack end: {g_attack_end}");

        // After decay (tick 30 = 10 attack + 20 decay): close to sustain 0.5
        let g_sustain = tone.gain_at(35);
        assert!((g_sustain - 0.5).abs() < 0.05, "sustain level: {g_sustain}");

        // Release: note_off at tick 50
        tone.note_off(50);
        // Right after release starts
        let g_release_start = tone.gain_at(51);
        assert!(
            g_release_start > 0.0 && g_release_start < 0.5,
            "release start: {g_release_start}"
        );

        // After release_end
        let g_done = tone.gain_at(61);
        assert!(g_done == 0.0, "after release end: {g_done}");
    }

    #[test]
    fn no_adsr_preserves_original_behavior() {
        let tb = Timebase { fs: 1000.0, hop: 4 };
        let tone = Tone::from_parts(tb, 0, 100, 440.0, 1.0, None, None, None).expect("tone");

        // In sustain phase: gain should be 1.0
        let g = tone.gain_at(50);
        assert!((g - 1.0).abs() < 1e-6, "sustain without adsr: {g}");
    }
}
