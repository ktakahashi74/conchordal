use crate::core::timebase::{Tick, Timebase};
use crate::life::intent::IntentBoard;

pub struct IntentRenderer {
    time: Timebase,
    buf: Vec<f32>,
    release_ticks: Tick,
}

impl IntentRenderer {
    pub fn new(time: Timebase) -> Self {
        let mut release_ticks = time.sec_to_tick(0.005);
        if release_ticks == 0 {
            release_ticks = 1;
        }
        Self {
            time,
            buf: vec![0.0; time.hop],
            release_ticks,
        }
    }

    /// Render scheduled intents from the IntentBoard (plans are committed before render).
    pub fn render(&mut self, board: &IntentBoard, now: Tick) -> &[f32] {
        let hop = self.time.hop;
        if self.buf.len() != hop {
            self.buf.resize(hop, 0.0);
        }
        self.buf.fill(0.0);

        let end = now.saturating_add(hop as Tick);
        let fs = self.time.fs;
        if fs <= 0.0 {
            return &self.buf;
        }

        for intent in board.query_range(now..end) {
            if intent.duration == 0 || intent.amp == 0.0 || intent.freq_hz <= 0.0 {
                continue;
            }

            let intent_end = intent.onset.saturating_add(intent.duration);
            let start_tick = intent.onset.max(now);
            let stop_tick = intent_end.min(end);

            for tick in start_tick..stop_tick {
                let idx = (tick - now) as usize;
                let t_rel = (tick - intent.onset) as f32 / fs;
                let phase = std::f32::consts::TAU * intent.freq_hz * t_rel;
                let env = self.release_env(intent, tick);
                let mut amp = intent.amp;
                let sample = if let Some(body) = &intent.body {
                    let amp_scale = body.amp_scale.clamp(0.0, 1.0);
                    amp *= amp_scale;
                    let b = body.brightness.clamp(0.0, 1.0);
                    let s1 = phase.cos();
                    let s2 = (2.0 * phase).cos() * 0.5;
                    let s3 = (3.0 * phase).cos() * 0.25;
                    let harmonic = (1.0 - b) * s1 + b * (s1 + s2 + s3);
                    let nm = body.noise_mix.clamp(0.0, 1.0);
                    if nm > 0.0 {
                        let n = hash_noise(tick, intent.source_id, intent.intent_id);
                        (1.0 - nm) * harmonic + nm * n
                    } else {
                        harmonic
                    }
                } else {
                    phase.cos()
                };
                self.buf[idx] += amp * env * sample;
            }
        }

        self.apply_limiter();
        &self.buf
    }

    fn release_env(&self, intent: &crate::life::intent::Intent, tick: Tick) -> f32 {
        if self.release_ticks == 0 || intent.duration <= self.release_ticks {
            return 1.0;
        }
        let pos = tick.saturating_sub(intent.onset);
        let tail = intent.duration.saturating_sub(self.release_ticks);
        if pos >= tail {
            let intent_end = intent.onset.saturating_add(intent.duration);
            let remain = intent_end.saturating_sub(tick);
            (remain as f32 / self.release_ticks as f32).clamp(0.0, 1.0)
        } else {
            1.0
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

fn hash_noise(tick: Tick, source_id: u64, intent_id: u64) -> f32 {
    let mut v = tick ^ source_id.rotate_left(17) ^ intent_id.rotate_left(41);
    v ^= v >> 33;
    v = v.wrapping_mul(0xff51afd7ed558ccd);
    v ^= v >> 33;
    v = v.wrapping_mul(0xc4ceb9fe1a85ec53);
    v ^= v >> 33;
    let unit = (v as u32) as f32 / u32::MAX as f32;
    unit * 2.0 - 1.0
}
