use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::audio::any_backend::AnyBackend;
use crate::life::audio::control::{ControlRamp, VoiceControlBlock};
use crate::life::audio::events::AudioEvent;
use crate::life::audio::exciter::{Exciter, ImpulseExciter};
use crate::life::audio::modal_engine::ModeShape;
use crate::life::individual::{ArticulationSignal, ArticulationWrapper};
use crate::life::intent::{BodySnapshot, Intent};
use crate::life::lifecycle::default_decay_attack;
use crate::life::phonation_engine::{PhonationKick, PhonationUpdate};
use crate::life::scenario::TimbreGenotype;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
struct PendingUpdate {
    at_tick: Tick,
    update: PhonationUpdate,
}

#[derive(Debug, Clone, Copy)]
struct PendingTrigger {
    at_tick: Tick,
    energy: f32,
}

pub struct Voice {
    backend: AnyBackend,
    exciter: ImpulseExciter,
    articulation: Option<ArticulationWrapper>,
    onset: Tick,
    hold_end: Tick,
    release_end: Tick,
    attack_ticks: Tick,
    release_ticks: Tick,
    planned_kick_pending: Option<PhonationKick>,
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
}

impl Voice {
    pub fn from_intent(time: Timebase, mut intent: Intent) -> Option<Self> {
        if intent.duration == 0 || intent.freq_hz <= 0.0 {
            return None;
        }
        if !intent.freq_hz.is_finite() || !intent.amp.is_finite() {
            return None;
        }
        if intent.amp == 0.0 {
            return None;
        }

        let (shape, amp_scale) = match intent.body.as_ref() {
            Some(snapshot) => (mode_shape_from_snapshot(snapshot), snapshot.amp_scale),
            None => (default_mode_shape(), 1.0),
        };
        let amp = intent.amp * amp_scale.clamp(0.0, 1.0);
        if !amp.is_finite() || amp <= 0.0 {
            return None;
        }

        let backend = AnyBackend::from_shape(time.fs, shape).ok()?;
        let articulation = intent.articulation.take();

        let attack_ticks = default_attack_ticks(time);
        let release_ticks = default_release_ticks(time);
        let (hold_end, release_end) = if intent.duration == Tick::MAX {
            (Tick::MAX, Tick::MAX)
        } else {
            let hold_end = intent.onset.saturating_add(intent.duration);
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
        let current_pitch_hz = intent.freq_hz;
        let target_pitch_hz = current_pitch_hz;
        let amp_tau_sec = 0.0;
        let pitch_tau_sec = 0.0;
        let amp_alpha = smoothing_alpha(sample_dt, amp_tau_sec);
        let pitch_alpha = smoothing_alpha(sample_dt, pitch_tau_sec);

        Some(Self {
            backend,
            exciter: ImpulseExciter::new(),
            articulation,
            onset: intent.onset,
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
        })
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

    pub fn trigger(&mut self, ev: AudioEvent) {
        self.exciter.trigger(ev);
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

    pub fn kick_planned(&mut self, kick: PhonationKick, rhythms: &NeuralRhythms, dt: f32) -> bool {
        if let Some(articulation) = self.articulation.as_mut() {
            articulation.kick_planned(kick, rhythms, dt);
            return true;
        }
        false
    }

    pub fn schedule_planned_kick(&mut self, kick: PhonationKick) {
        self.planned_kick_pending = Some(kick);
    }

    pub fn schedule_update(&mut self, at_tick: Tick, update: PhonationUpdate) {
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
        if tick >= self.hold_end {
            self.pending_updates.clear();
            return;
        }
        while let Some(pending) = self.pending_updates.front().copied() {
            if pending.at_tick > tick {
                break;
            }
            let pending = self.pending_updates.pop_front().expect("pending update");
            self.apply_update(&pending.update);
        }
    }

    pub fn kick_planned_if_due(&mut self, tick: Tick, rhythms: &NeuralRhythms, dt: f32) -> bool {
        let Some(kick) = self.planned_kick_pending else {
            return false;
        };
        if tick >= self.onset {
            self.planned_kick_pending = None;
            return self.kick_planned(kick, rhythms, dt);
        }
        false
    }

    pub fn render_tick(&mut self, tick: Tick, _fs: f32, dt: f32, rhythms: &NeuralRhythms) -> f32 {
        if let Some(trigger) = self.pending_trigger
            && tick >= trigger.at_tick
        {
            self.pending_trigger = None;
            self.trigger(AudioEvent::Impulse {
                energy: trigger.energy,
            });
        }
        self.advance_smoothing();
        let gain = self.gain_at(tick);

        let mut signal = if let Some(articulation) = self.articulation.as_mut() {
            let mut signal = articulation.process(1.0, rhythms, dt, 1.0);
            signal.amplitude *= articulation.gate();
            signal
        } else {
            ArticulationSignal {
                amplitude: 1.0,
                is_active: true,
                relaxation: rhythms.theta.alpha,
                tension: rhythms.theta.beta,
            }
        };
        if self.articulation.is_none() {
            let tension = signal.tension.clamp(0.0, 1.0);
            signal.amplitude *= 1.0 + 0.05 * tension;
        }
        signal.amplitude *= gain;
        signal.is_active = signal.is_active && signal.amplitude > 0.0;

        let drive = self.exciter.next_drive();
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
        out[0] * signal.amplitude
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

    fn apply_update(&mut self, update: &PhonationUpdate) {
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

        self.current_pitch_hz = smooth_step(
            self.current_pitch_hz,
            self.target_pitch_hz,
            self.pitch_alpha,
        );
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

fn default_mode_shape() -> ModeShape {
    ModeShape::Sine {
        t60_s: 0.8,
        out_gain: 1.0,
        in_gain: 1.0,
    }
}

fn mode_shape_from_snapshot(snapshot: &BodySnapshot) -> ModeShape {
    match snapshot.kind.as_str() {
        "harmonic" => ModeShape::Harmonic {
            partials: 16,
            base_t60_s: 0.8,
            in_gain: 1.0,
            genotype: TimbreGenotype {
                brightness: snapshot.brightness.clamp(0.0, 1.0),
                jitter: snapshot.noise_mix.clamp(0.0, 1.0),
                ..TimbreGenotype::default()
            },
        },
        _ => default_mode_shape(),
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
        let intent = Intent {
            source_id: 1,
            intent_id: 0,
            onset: 0,
            duration: Tick::MAX,
            freq_hz: 440.0,
            amp: 0.5,
            tag: None,
            confidence: 1.0,
            body: None,
            articulation: None,
        };
        let mut voice = Voice::from_intent(tb, intent).expect("voice");

        let mut rhythms = NeuralRhythms::default();
        let mut out = vec![0.0f32; tb.hop];
        voice.render_block(0, tb.fs, 1.0 / tb.fs, &mut rhythms, &mut out);
        assert!(out.iter().all(|s| s.abs() <= 1e-6));

        voice.trigger(AudioEvent::Impulse { energy: 1.0 });
        let mut rhythms = NeuralRhythms::default();
        let mut out = vec![0.0f32; tb.hop];
        voice.render_block(0, tb.fs, 1.0 / tb.fs, &mut rhythms, &mut out);
        assert!(out.iter().any(|s| s.abs() > 1e-6));
    }
}
