use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::gate_envelope::GateEnvelope;
use crate::life::individual::{AnySoundBody, ArticulationSignal, SoundBody};
use crate::life::intent::{BodySnapshot, Intent, IntentBoard};
use crate::life::scenario::{SoundBodyConfig, TimbreGenotype};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct VoiceKey {
    source_id: u64,
    intent_id: u64,
}

struct Voice {
    intent: Intent,
    body: AnySoundBody,
    end_tick: Tick,
}

pub struct ScheduleRenderer {
    time: Timebase,
    buf: Vec<f32>,
    voices: HashMap<VoiceKey, Voice>,
    envelope: GateEnvelope,
    add_future_ticks: Tick,
    add_past_ticks: Tick,
    did_full_resync: bool,
}

impl ScheduleRenderer {
    pub fn new(time: Timebase) -> Self {
        let add_future_ticks = (time.hop as Tick).saturating_mul(4).max(1);
        let envelope = GateEnvelope::new(time);
        let add_past_ticks = schedule_add_past_ticks(time, &envelope);
        Self {
            time,
            buf: vec![0.0; time.hop],
            voices: HashMap::new(),
            envelope,
            add_future_ticks,
            add_past_ticks,
            did_full_resync: false,
        }
    }

    pub fn render(&mut self, board: &IntentBoard, now: Tick, rhythms: &NeuralRhythms) -> &[f32] {
        let hop = self.time.hop;
        if self.buf.len() != hop {
            self.buf.resize(hop, 0.0);
        }
        self.buf.fill(0.0);

        let fs = self.time.fs;
        if fs <= 0.0 {
            return &self.buf;
        }

        self.voices.retain(|_, voice| voice.end_tick > now);

        let past = if self.did_full_resync {
            board.retention_past.min(self.add_past_ticks)
        } else {
            self.did_full_resync = true;
            board.retention_past
        };
        let future = board.horizon_future.min(self.add_future_ticks);
        for intent in board.snapshot(now, past, future) {
            self.add_voice_if_needed(intent, now);
        }

        let end = now.saturating_add(hop as Tick);
        let dt = 1.0 / fs;
        let mut rhythms = *rhythms;
        for tick in now..end {
            let idx = (tick - now) as usize;
            let mut acc = 0.0f32;
            for voice in self.voices.values_mut() {
                if tick < voice.intent.onset || tick >= voice.end_tick {
                    continue;
                }
                let env = self.envelope.gain(&voice.intent, tick);
                if env <= 0.0 {
                    continue;
                }
                let signal = ArticulationSignal {
                    amplitude: env,
                    is_active: env > 0.0,
                    relaxation: rhythms.theta.alpha,
                    tension: rhythms.theta.beta,
                };
                let mut sample = 0.0f32;
                voice.body.articulate_wave(&mut sample, fs, dt, &signal);
                acc += sample;
            }
            self.buf[idx] = acc;
            rhythms.advance_in_place(dt);
        }

        self.apply_limiter();
        &self.buf
    }

    fn add_voice_if_needed(&mut self, intent: Intent, now: Tick) {
        if intent.duration == 0 || intent.amp == 0.0 || intent.freq_hz <= 0.0 {
            return;
        }
        let end_tick = intent
            .onset
            .saturating_add(intent.duration)
            .saturating_add(self.envelope.release_ticks());
        if end_tick <= now {
            return;
        }
        let key = VoiceKey {
            source_id: intent.source_id,
            intent_id: intent.intent_id,
        };
        if self.voices.contains_key(&key) {
            return;
        }

        let (config, amp_scale) = match intent.body.as_ref() {
            Some(snapshot) => (
                sound_body_config_from_snapshot(snapshot),
                snapshot.amp_scale,
            ),
            None => (SoundBodyConfig::Sine { phase: None }, 1.0),
        };
        let amp = intent.amp * amp_scale.clamp(0.0, 1.0);
        if !amp.is_finite() || amp == 0.0 {
            return;
        }

        let seed = intent
            .intent_id
            .wrapping_add(intent.source_id.rotate_left(17))
            .wrapping_add(intent.onset);
        let mut rng = SmallRng::seed_from_u64(seed);
        let body = AnySoundBody::from_config(&config, intent.freq_hz, amp, &mut rng);

        self.voices.insert(
            key,
            Voice {
                intent,
                body,
                end_tick,
            },
        );
    }

    fn apply_limiter(&mut self) {
        let mut peak = 0.0f32;
        for &s in &self.buf {
            if s.is_finite() {
                peak = peak.max(s.abs());
            }
        }
        let target = 0.98f32;
        if peak > target && peak > 0.0 {
            let g = target / peak;
            if g.is_finite() {
                for s in &mut self.buf {
                    *s *= g;
                }
            }
        }
    }
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

fn schedule_add_past_ticks(time: Timebase, envelope: &GateEnvelope) -> Tick {
    let hop_window = (time.hop as Tick).saturating_mul(2);
    envelope.release_ticks().max(hop_window).max(1)
}
