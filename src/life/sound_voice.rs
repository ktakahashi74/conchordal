use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::individual::{AnySoundBody, ArticulationSignal, ArticulationWrapper, SoundBody};
use crate::life::intent::{BodySnapshot, Intent};
use crate::life::lifecycle::default_decay_attack;
use crate::life::phonation_engine::{PhonationKick, PhonationUpdate};
use crate::life::scenario::{SoundBodyConfig, TimbreGenotype};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
struct PendingUpdate {
    at_tick: Tick,
    update: PhonationUpdate,
}

pub struct SoundVoice {
    body: AnySoundBody,
    articulation: Option<ArticulationWrapper>,
    onset: Tick,
    hold_end: Tick,
    release_end: Tick,
    attack_ticks: Tick,
    release_ticks: Tick,
    planned_kick_pending: Option<PhonationKick>,
    pending_updates: VecDeque<PendingUpdate>,
    current_amp: f32,
    target_amp: f32,
    current_freq_hz: f32,
    target_freq_hz: f32,
    amp_tau_sec: f32,
    freq_tau_sec: f32,
    amp_alpha: f32,
    freq_alpha: f32,
    sample_dt: f32,
}

impl SoundVoice {
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

        let (config, amp_scale) = match intent.body.as_ref() {
            Some(snapshot) => (
                sound_body_config_from_snapshot(snapshot),
                snapshot.amp_scale,
            ),
            None => (SoundBodyConfig::Sine { phase: None }, 1.0),
        };
        let amp = intent.amp * amp_scale.clamp(0.0, 1.0);
        if !amp.is_finite() {
            return None;
        }
        if amp <= 0.0 {
            return None;
        }

        let seed = intent
            .intent_id
            .wrapping_add(intent.source_id.rotate_left(17))
            .wrapping_add(intent.onset);
        let mut rng = SmallRng::seed_from_u64(seed);
        let body = AnySoundBody::from_config(&config, intent.freq_hz, amp, &mut rng);
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
        let current_freq_hz = intent.freq_hz;
        let target_freq_hz = current_freq_hz;
        let amp_tau_sec = 0.0;
        let freq_tau_sec = 0.0;
        let amp_alpha = smoothing_alpha(sample_dt, amp_tau_sec);
        let freq_alpha = smoothing_alpha(sample_dt, freq_tau_sec);

        Some(Self {
            body,
            articulation,
            onset: intent.onset,
            hold_end,
            release_end,
            attack_ticks,
            release_ticks,
            planned_kick_pending: None,
            pending_updates: VecDeque::new(),
            current_amp,
            target_amp,
            current_freq_hz,
            target_freq_hz,
            amp_tau_sec,
            freq_tau_sec,
            amp_alpha,
            freq_alpha,
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

    pub fn apply_update(&mut self, update: &PhonationUpdate) {
        if let Some(freq_hz) = update.target_freq_hz
            && freq_hz.is_finite()
            && freq_hz > 0.0
        {
            self.target_freq_hz = freq_hz;
            if self.freq_alpha >= 1.0 {
                self.current_freq_hz = freq_hz;
                self.body.set_freq(freq_hz);
            }
        }
        if let Some(amp) = update.target_amp
            && amp.is_finite()
        {
            let amp = amp.max(0.0);
            self.target_amp = amp;
            if self.amp_alpha >= 1.0 {
                self.current_amp = amp;
                self.body.set_amp(amp);
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
        self.freq_tau_sec = tau;
        self.amp_alpha = smoothing_alpha(self.sample_dt, self.amp_tau_sec);
        self.freq_alpha = smoothing_alpha(self.sample_dt, self.freq_tau_sec);
        if self.amp_alpha >= 1.0 {
            self.current_amp = self.target_amp;
            self.body.set_amp(self.current_amp);
        }
        if self.freq_alpha >= 1.0 {
            self.current_freq_hz = self.target_freq_hz;
            self.body.set_freq(self.current_freq_hz);
        }
    }

    #[cfg(test)]
    pub(crate) fn debug_body_freq_hz(&self) -> f32 {
        self.body.base_freq_hz()
    }

    #[cfg(test)]
    pub(crate) fn debug_body_amp(&self) -> f32 {
        self.body.amp()
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
        self.target_freq_hz
    }

    #[cfg(test)]
    pub(crate) fn debug_current_freq_hz(&self) -> f32 {
        self.current_freq_hz
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

    pub fn render_tick(&mut self, tick: Tick, fs: f32, dt: f32, rhythms: &NeuralRhythms) -> f32 {
        self.advance_smoothing();
        let gain = self.gain_at(tick);
        if gain <= 0.0 {
            return 0.0;
        }
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
        signal.amplitude *= gain;
        signal.is_active = signal.is_active && signal.amplitude > 0.0;
        let mut sample = 0.0f32;
        self.body.articulate_wave(&mut sample, fs, dt, &signal);
        sample
    }

    fn advance_smoothing(&mut self) {
        self.current_amp = smooth_step(self.current_amp, self.target_amp, self.amp_alpha);
        if !self.current_amp.is_finite() {
            self.current_amp = self.target_amp;
        }
        self.current_amp = self.current_amp.max(0.0);

        self.current_freq_hz =
            smooth_step(self.current_freq_hz, self.target_freq_hz, self.freq_alpha);
        if !self.current_freq_hz.is_finite() || self.current_freq_hz <= 0.0 {
            self.current_freq_hz = self.target_freq_hz;
        }

        if self.current_amp.is_finite() {
            self.body.set_amp(self.current_amp);
        }
        if self.current_freq_hz.is_finite() && self.current_freq_hz > 0.0 {
            self.body.set_freq(self.current_freq_hz);
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

fn sound_body_config_from_snapshot(snapshot: &BodySnapshot) -> SoundBodyConfig {
    match snapshot.kind.as_str() {
        "harmonic" => SoundBodyConfig::Harmonic {
            genotype: TimbreGenotype {
                brightness: snapshot.brightness.clamp(0.0, 1.0),
                jitter: snapshot.noise_mix.clamp(0.0, 1.0),
                ..TimbreGenotype::default()
            },
            partials: None,
        },
        _ => SoundBodyConfig::Sine { phase: None },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::individual::{AnyArticulationCore, ArticulationWrapper, SequencedCore};

    #[test]
    fn planned_kick_waits_for_onset() {
        let tb = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let articulation = ArticulationWrapper::new(
            AnyArticulationCore::Seq(SequencedCore {
                timer: 0.0,
                duration: 0.1,
                env_level: 0.0,
            }),
            1.0,
        );
        let intent = Intent {
            source_id: 1,
            intent_id: 0,
            onset: 10,
            duration: 20,
            freq_hz: 440.0,
            amp: 0.5,
            tag: None,
            confidence: 1.0,
            body: None,
            articulation: Some(articulation),
        };
        let mut voice = SoundVoice::from_intent(tb, intent).expect("voice");
        let rhythms = NeuralRhythms::default();
        let dt = 1.0 / tb.fs;
        voice.schedule_planned_kick(PhonationKick::Planned { strength: 1.0 });
        assert!(!voice.kick_planned_if_due(9, &rhythms, dt));
        assert!(voice.kick_planned_if_due(10, &rhythms, dt));
        assert!(!voice.kick_planned_if_due(11, &rhythms, dt));
    }

    #[test]
    fn max_duration_note_off_releases() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let intent = Intent {
            source_id: 1,
            intent_id: 1,
            onset: 0,
            duration: Tick::MAX,
            freq_hz: 220.0,
            amp: 0.5,
            tag: None,
            confidence: 1.0,
            body: None,
            articulation: None,
        };
        let mut voice = SoundVoice::from_intent(tb, intent).expect("voice");
        let off_tick = 4;
        voice.note_off(off_tick);
        let done_tick = off_tick
            .saturating_add(default_release_ticks(tb))
            .saturating_add(1);
        assert!(voice.is_done(done_tick));
    }

    #[test]
    fn update_applies_pitch_and_orders_updates() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let intent = Intent {
            source_id: 1,
            intent_id: 1,
            onset: 0,
            duration: 10,
            freq_hz: 220.0,
            amp: 0.5,
            tag: None,
            confidence: 1.0,
            body: None,
            articulation: None,
        };
        let mut voice = SoundVoice::from_intent(tb, intent).expect("voice");
        voice.schedule_update(
            0,
            PhonationUpdate {
                target_freq_hz: Some(330.0),
                target_amp: None,
            },
        );
        voice.schedule_update(
            0,
            PhonationUpdate {
                target_freq_hz: Some(440.0),
                target_amp: None,
            },
        );
        voice.apply_updates_if_due(0);
        assert!((voice.debug_target_freq_hz() - 440.0).abs() < 1e-6);
        assert!((voice.debug_current_freq_hz() - 440.0).abs() < 1e-6);
    }

    #[test]
    fn note_off_wins_over_same_tick_update() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let intent = Intent {
            source_id: 1,
            intent_id: 1,
            onset: 0,
            duration: 10,
            freq_hz: 220.0,
            amp: 0.5,
            tag: None,
            confidence: 1.0,
            body: None,
            articulation: None,
        };
        let mut voice = SoundVoice::from_intent(tb, intent).expect("voice");
        voice.note_off(0);
        voice.schedule_update(
            0,
            PhonationUpdate {
                target_freq_hz: Some(440.0),
                target_amp: None,
            },
        );
        voice.apply_updates_if_due(0);
        assert!((voice.body.base_freq_hz() - 220.0).abs() < 1e-6);
    }

    #[test]
    fn note_off_discards_past_update_after_release() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let intent = Intent {
            source_id: 1,
            intent_id: 1,
            onset: 0,
            duration: 10,
            freq_hz: 220.0,
            amp: 0.5,
            tag: None,
            confidence: 1.0,
            body: None,
            articulation: None,
        };
        let mut voice = SoundVoice::from_intent(tb, intent).expect("voice");
        voice.note_off(1);
        voice.schedule_update(
            0,
            PhonationUpdate {
                target_freq_hz: Some(440.0),
                target_amp: None,
            },
        );
        voice.apply_updates_if_due(2);
        assert!((voice.body.base_freq_hz() - 220.0).abs() < 1e-6);
    }

    #[test]
    fn smoothing_tau_zero_updates_immediately() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let intent = Intent {
            source_id: 1,
            intent_id: 1,
            onset: 0,
            duration: 10,
            freq_hz: 220.0,
            amp: 0.1,
            tag: None,
            confidence: 1.0,
            body: None,
            articulation: None,
        };
        let mut voice = SoundVoice::from_intent(tb, intent).expect("voice");
        voice.set_smoothing_tau_sec(0.0);
        voice.schedule_update(
            0,
            PhonationUpdate {
                target_freq_hz: Some(440.0),
                target_amp: Some(0.5),
            },
        );
        voice.apply_updates_if_due(0);
        assert!((voice.debug_current_freq_hz() - 440.0).abs() < 1e-6);
        assert!((voice.debug_current_amp() - 0.5).abs() < 1e-6);
        assert!((voice.debug_target_freq_hz() - 440.0).abs() < 1e-6);
        assert!((voice.debug_target_amp() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn smoothing_tau_positive_moves_toward_target() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let intent = Intent {
            source_id: 1,
            intent_id: 1,
            onset: 0,
            duration: 10,
            freq_hz: 220.0,
            amp: 0.1,
            tag: None,
            confidence: 1.0,
            body: None,
            articulation: None,
        };
        let mut voice = SoundVoice::from_intent(tb, intent).expect("voice");
        voice.set_smoothing_tau_sec(0.1);
        voice.schedule_update(
            0,
            PhonationUpdate {
                target_freq_hz: Some(440.0),
                target_amp: Some(1.0),
            },
        );
        voice.apply_updates_if_due(0);

        let amp_before = voice.debug_current_amp();
        let freq_before = voice.debug_current_freq_hz();
        let rhythms = NeuralRhythms::default();
        let dt = 1.0 / tb.fs;
        let _ = voice.render_tick(0, tb.fs, dt, &rhythms);

        let amp_after = voice.debug_current_amp();
        let freq_after = voice.debug_current_freq_hz();
        assert!(amp_after > amp_before && amp_after < voice.debug_target_amp());
        assert!(freq_after > freq_before && freq_after < voice.debug_target_freq_hz());
    }
}
