use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::individual::{AnySoundBody, ArticulationSignal, SoundBody};
use crate::life::intent::{BodySnapshot, Intent};
use crate::life::lifecycle::default_decay_attack;
use crate::life::scenario::{SoundBodyConfig, TimbreGenotype};
use rand::SeedableRng;
use rand::rngs::SmallRng;

pub struct SoundVoice {
    body: AnySoundBody,
    onset: Tick,
    hold_end: Tick,
    release_end: Tick,
    attack_ticks: Tick,
    release_ticks: Tick,
}

impl SoundVoice {
    pub fn from_intent(time: Timebase, intent: &Intent) -> Option<Self> {
        if intent.duration == 0 || intent.amp == 0.0 || intent.freq_hz <= 0.0 {
            return None;
        }
        if !intent.freq_hz.is_finite() || !intent.amp.is_finite() {
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
        if !amp.is_finite() || amp <= 0.0 {
            return None;
        }

        let seed = intent
            .intent_id
            .wrapping_add(intent.source_id.rotate_left(17))
            .wrapping_add(intent.onset);
        let mut rng = SmallRng::seed_from_u64(seed);
        let body = AnySoundBody::from_config(&config, intent.freq_hz, amp, &mut rng);

        let attack_ticks = default_attack_ticks(time);
        let release_ticks = default_release_ticks(time);
        let hold_end = intent.onset.saturating_add(intent.duration);
        let release_end = hold_end.saturating_add(release_ticks);

        Some(Self {
            body,
            onset: intent.onset,
            hold_end,
            release_end,
            attack_ticks,
            release_ticks,
        })
    }

    pub fn note_off(&mut self, tick: Tick) {
        if tick < self.hold_end {
            self.hold_end = tick;
            self.release_end = self.hold_end.saturating_add(self.release_ticks);
        }
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
        let gain = self.gain_at(tick);
        if gain <= 0.0 {
            return 0.0;
        }
        let signal = ArticulationSignal {
            amplitude: gain,
            is_active: gain > 0.0,
            relaxation: rhythms.theta.alpha,
            tension: rhythms.theta.beta,
        };
        let mut sample = 0.0f32;
        self.body.articulate_wave(&mut sample, fs, dt, &signal);
        sample
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
