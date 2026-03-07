use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::individual::ArticulationSignal;
use crate::life::lifecycle::default_decay_attack;
use crate::life::phonation_engine::{NoteUpdate, OnsetKick};
use crate::life::sound::any_backend::AnyBackend;
use crate::life::sound::control::{ControlRamp, VoiceControlBlock};
use crate::life::sound::{BodyKind, BodySnapshot, RenderModulator, RenderModulatorSpec};
use std::collections::VecDeque;

const SINE_IMPULSE_BOOST_GAIN: f32 = 0.2;
const SINE_IMPULSE_BOOST_MAX: f32 = 1.0;
const SINE_IMPULSE_BOOST_DECAY_SEC: f32 = 0.08;

#[derive(Debug, Clone, Copy)]
struct PendingUpdate {
    at_tick: Tick,
    update: NoteUpdate,
}

#[derive(Debug, Clone, Copy)]
struct PendingTrigger {
    at_tick: Tick,
    energy: f32,
}

pub struct Voice {
    backend: AnyBackend,
    render_modulator: Option<RenderModulator>,
    pending_impulse_energy: f32,
    onset: Tick,
    hold_end: Tick,
    release_end: Tick,
    attack_ticks: Tick,
    release_ticks: Tick,
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

impl Voice {
    pub fn from_parts(
        time: Timebase,
        onset: Tick,
        duration: Tick,
        freq_hz: f32,
        amp: f32,
        body: Option<BodySnapshot>,
        render_modulator: Option<RenderModulatorSpec>,
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

        let attack_ticks = default_attack_ticks(time);
        let release_ticks = default_release_ticks(time);
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
            backend,
            render_modulator: render_modulator.map(RenderModulator::from_spec),
            pending_impulse_energy: 0.0,
            onset,
            hold_end,
            release_end,
            attack_ticks,
            release_ticks,
            planned_kick_pending: None,
            pending_updates: VecDeque::new(),
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
    }

    pub fn note_off(&mut self, tick: Tick) {
        if tick < self.hold_end {
            self.hold_end = tick;
            self.release_end = self.hold_end.saturating_add(self.release_ticks);
        }
    }

    pub fn note_on(&mut self, tick: Tick) {
        if tick > self.onset {
            self.onset = tick;
            if self.hold_end < self.onset {
                self.hold_end = self.onset;
                self.release_end = self.hold_end.saturating_add(self.release_ticks);
            }
        }
    }

    pub fn arm_onset_trigger(&mut self, energy: f32) {
        if !energy.is_finite() || energy <= 0.0 {
            return;
        }
        self.pending_trigger = Some(PendingTrigger {
            at_tick: self.onset,
            energy,
        });
    }

    pub fn trigger_impulse(&mut self, energy: f32) {
        if !energy.is_finite() || energy <= 0.0 {
            return;
        }
        self.pending_impulse_energy += energy;
    }

    pub fn set_target(&mut self, pitch_hz: f32, amp: f32, tau_sec: f32) {
        self.set_smoothing_tau_sec(tau_sec);
        if pitch_hz.is_finite() && pitch_hz > 0.0 {
            self.target_pitch_hz = pitch_hz;
            if self.pitch_alpha >= 1.0 {
                self.current_pitch_hz = pitch_hz;
            }
        }
        if amp.is_finite() {
            let amp = amp.max(0.0);
            self.target_amp = amp;
            if self.amp_alpha >= 1.0 {
                self.current_amp = amp;
            }
        }
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

    pub fn schedule_update(&mut self, at_tick: Tick, update: NoteUpdate) {
        if update.is_empty() {
            return;
        }
        let insert_at = self
            .pending_updates
            .iter()
            .position(|pending| pending.at_tick > at_tick)
            .unwrap_or(self.pending_updates.len());
        self.pending_updates
            .insert(insert_at, PendingUpdate { at_tick, update });
    }

    pub fn apply_updates_if_due(&mut self, tick: Tick) {
        while let Some(pending) = self.pending_updates.front().copied() {
            if pending.at_tick > tick {
                break;
            }
            let pending = self.pending_updates.pop_front().expect("pending update");
            self.apply_update(&pending.update);
        }
        if tick >= self.hold_end {
            self.pending_updates.clear();
        }
    }

    pub fn kick_planned_if_due(&mut self, tick: Tick) -> bool {
        let Some(kick) = self.planned_kick_pending else {
            return false;
        };
        if tick >= self.onset {
            self.planned_kick_pending = None;
            return self.kick_planned(kick);
        }
        false
    }

    pub fn render_tick(&mut self, tick: Tick, _fs: f32, dt: f32, rhythms: &NeuralRhythms) -> f32 {
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

        let drive = if self.backend.supports_continuous_drive() {
            let noise = fast_noise(&mut self.noise_state);
            impulse + self.continuous_drive * signal.amplitude * noise
        } else {
            0.0
        };
        let ctrl = VoiceControlBlock {
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
        // Sine backend is pure oscillator; sustain-drive is intentionally ignored.
        self.continuous_drive = if self.backend.is_sine() {
            0.0
        } else if level.is_finite() {
            level.max(0.0)
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

    pub fn end_tick(&self) -> Tick {
        self.release_end
    }

    pub fn onset(&self) -> Tick {
        self.onset
    }

    pub fn is_done(&self, now: Tick) -> bool {
        now >= self.release_end
    }

    fn apply_update(&mut self, update: &NoteUpdate) {
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
    }

    fn advance_smoothing(&mut self) {
        self.current_amp = smooth_step(self.current_amp, self.target_amp, self.amp_alpha);
        if !self.current_amp.is_finite() {
            self.current_amp = self.target_amp;
        }
        self.current_amp = self.current_amp.max(0.0);

        // Smooth pitch in log2 space so equal semitone steps take equal time.
        if self.pitch_alpha >= 1.0 || self.current_pitch_hz <= 0.0 || self.target_pitch_hz <= 0.0 {
            self.current_pitch_hz = self.target_pitch_hz;
        } else {
            let cur_log = self.current_pitch_hz.ln();
            let tgt_log = self.target_pitch_hz.ln();
            let next_log = smooth_step(cur_log, tgt_log, self.pitch_alpha);
            self.current_pitch_hz = next_log.exp();
        }
        if !self.current_pitch_hz.is_finite() || self.current_pitch_hz <= 0.0 {
            self.current_pitch_hz = self.target_pitch_hz;
        }
    }

    fn gain_at(&self, tick: Tick) -> f32 {
        if tick < self.onset || tick >= self.release_end {
            return 0.0;
        }

        let duration_ticks = self.hold_end.saturating_sub(self.onset).max(1);
        let pos = tick.saturating_sub(self.onset);
        let attack_len = self.attack_ticks.min(duration_ticks);
        let attack = if attack_len > 0 && pos < attack_len {
            (pos.saturating_add(1) as f32 / attack_len as f32).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let release = if tick >= self.hold_end {
            if self.release_ticks == 0 {
                0.0
            } else {
                let remain = self.release_end.saturating_sub(tick);
                (remain as f32 / self.release_ticks as f32).clamp(0.0, 1.0)
            }
        } else {
            1.0
        };

        (attack * release).clamp(0.0, 1.0)
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
        width: 0.0,
        noise_mix: 0.0,
        ratios: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_does_not_sound_until_triggered() {
        let tb = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let mut voice = Voice::from_parts(tb, 0, Tick::MAX, 440.0, 0.5, None, None).expect("voice");

        let mut rhythms = NeuralRhythms::default();
        let mut out = vec![0.0f32; tb.hop];
        voice.render_block(0, tb.fs, 1.0 / tb.fs, &mut rhythms, &mut out);
        assert!(out.iter().all(|s| s.abs() <= 1e-6));

        voice.trigger_impulse(1.0);
        let mut rhythms = NeuralRhythms::default();
        let mut out = vec![0.0f32; tb.hop];
        voice.render_block(0, tb.fs, 1.0 / tb.fs, &mut rhythms, &mut out);
        assert!(out.iter().any(|s| s.abs() > 1e-6));
    }

    #[test]
    fn continuous_drive_does_not_start_sine_without_impulse() {
        let tb = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let mut voice = Voice::from_parts(tb, 0, Tick::MAX, 440.0, 0.5, None, None).expect("voice");
        voice.set_continuous_drive(0.01);

        let dt = 1.0 / tb.fs;
        let blocks = 64;
        let mut last_block = vec![0.0f32; tb.hop];
        for b in 0..blocks {
            let tick = (b * tb.hop) as Tick;
            let mut rhythms = NeuralRhythms::default();
            voice.render_block(tick, tb.fs, dt, &mut rhythms, &mut last_block);
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
            width: 0.1,
            noise_mix: 0.0,
            ratios: None,
        };
        let mut voice =
            Voice::from_parts(tb, 0, Tick::MAX, 220.0, 0.5, Some(harmonic), None).expect("voice");
        voice.set_continuous_drive(0.03);
        voice.trigger_impulse(1.0);
        let dt = 1.0 / tb.fs;
        let blocks = (2.0 * tb.fs as f64 / tb.hop as f64) as usize;
        let mut last_block = vec![0.0f32; tb.hop];
        for b in 0..blocks {
            let tick = (b * tb.hop) as Tick;
            let mut rhythms = NeuralRhythms::default();
            voice.render_block(tick, tb.fs, dt, &mut rhythms, &mut last_block);
        }
        let peak = last_block.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(
            peak > 1e-4,
            "harmonic sustain drive should keep energy; peak={peak}"
        );
    }

    #[test]
    fn sine_reimpulse_adds_short_boost() {
        let tb = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let mut voice = Voice::from_parts(tb, 0, Tick::MAX, 440.0, 0.5, None, None).expect("voice");
        let dt = 1.0 / tb.fs;
        let mut rhythms = NeuralRhythms::default();

        voice.trigger_impulse(1.0);
        let mut before = vec![0.0f32; tb.hop];
        voice.render_block(0, tb.fs, dt, &mut rhythms, &mut before);
        let before_peak = before.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

        voice.trigger_impulse(1.0);
        let mut after = vec![0.0f32; tb.hop];
        voice.render_block(tb.hop as Tick, tb.fs, dt, &mut rhythms, &mut after);
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
        let mut voice =
            Voice::from_parts(tb, 0, 8, 440.0, 0.5, None, None).expect("voice");
        voice.set_smoothing_tau_sec(0.0);
        voice.note_off(4);
        voice.schedule_update(
            4,
            NoteUpdate {
                target_freq_hz: None,
                target_amp: Some(0.25),
            },
        );

        voice.apply_updates_if_due(4);

        assert!((voice.debug_target_amp() - 0.25).abs() < 1e-6);
        assert!((voice.debug_current_amp() - 0.25).abs() < 1e-6);
    }
}
