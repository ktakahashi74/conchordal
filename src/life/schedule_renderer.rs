use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::intent::{Intent, IntentBoard};
use crate::life::sound_voice::{SoundVoice, default_release_ticks};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct VoiceKey {
    source_id: u64,
    intent_id: u64,
}

pub struct ScheduleRenderer {
    time: Timebase,
    buf: Vec<f32>,
    voices: HashMap<VoiceKey, SoundVoice>,
    add_future_ticks: Tick,
    add_past_ticks: Tick,
    did_full_resync: bool,
    cutoff_tick: Option<Tick>,
}

impl ScheduleRenderer {
    pub fn new(time: Timebase) -> Self {
        let add_future_ticks = (time.hop as Tick).saturating_mul(4).max(1);
        let add_past_ticks = schedule_add_past_ticks(time);
        Self {
            time,
            buf: vec![0.0; time.hop],
            voices: HashMap::new(),
            add_future_ticks,
            add_past_ticks,
            did_full_resync: false,
            cutoff_tick: None,
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

        self.voices.retain(|_, voice| !voice.is_done(now));

        let past = if self.did_full_resync {
            board.retention_past.min(self.add_past_ticks)
        } else {
            self.did_full_resync = true;
            board.retention_past
        };
        let future = board.horizon_future.min(self.add_future_ticks);
        for intent in board.snapshot(now, past, future) {
            if let Some(cutoff) = self.cutoff_tick
                && intent.onset >= cutoff
            {
                continue;
            }
            self.add_voice_if_needed(intent, now);
        }

        let end = now.saturating_add(hop as Tick);
        let dt = 1.0 / fs;
        let mut rhythms = *rhythms;
        for tick in now..end {
            let idx = (tick - now) as usize;
            let mut acc = 0.0f32;
            for voice in self.voices.values_mut() {
                acc += voice.render_tick(tick, fs, dt, &rhythms);
            }
            self.buf[idx] = acc;
            rhythms.advance_in_place(dt);
        }

        self.apply_limiter();
        &self.buf
    }

    pub fn is_idle(&self) -> bool {
        self.voices.is_empty()
    }

    pub fn set_cutoff_tick(&mut self, cutoff: Option<Tick>) {
        self.cutoff_tick = cutoff;
    }

    pub fn shutdown_at(&mut self, tick: Tick) {
        self.cutoff_tick = Some(tick);
        self.voices.retain(|_, voice| voice.onset() <= tick);
        for voice in self.voices.values_mut() {
            voice.note_off(tick);
        }
    }

    fn add_voice_if_needed(&mut self, intent: Intent, now: Tick) {
        if intent.duration == 0 || intent.amp == 0.0 || intent.freq_hz <= 0.0 {
            return;
        }
        let end_tick = intent
            .onset
            .saturating_add(intent.duration)
            .saturating_add(default_release_ticks(self.time));
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
        if let Some(voice) = SoundVoice::from_intent(self.time, &intent) {
            self.voices.insert(key, voice);
        }
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

fn schedule_add_past_ticks(time: Timebase) -> Tick {
    let hop_window = (time.hop as Tick).saturating_mul(2);
    default_release_ticks(time).max(hop_window).max(1)
}
