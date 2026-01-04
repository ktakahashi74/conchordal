use crate::core::timebase::{Tick, Timebase};
use crate::life::intent::Intent;
use crate::life::lifecycle::default_decay_attack;

pub struct GateEnvelope {
    attack_ticks: Tick,
    release_ticks: Tick,
}

impl GateEnvelope {
    pub fn new(time: Timebase) -> Self {
        let attack_sec = default_decay_attack();
        let release_sec = default_decay_attack();
        let mut attack_ticks = time.sec_to_tick(attack_sec);
        if attack_ticks == 0 {
            attack_ticks = 1;
        }
        let mut release_ticks = time.sec_to_tick(release_sec);
        if release_ticks == 0 {
            release_ticks = 1;
        }
        Self {
            attack_ticks,
            release_ticks,
        }
    }

    #[allow(dead_code)]
    pub fn gain(&self, intent: &Intent, tick: Tick) -> f32 {
        self.gain_with_end(intent, tick, None)
    }

    pub fn gain_with_end(&self, intent: &Intent, tick: Tick, end_override: Option<Tick>) -> f32 {
        let intent_end =
            end_override.unwrap_or_else(|| intent.onset.saturating_add(intent.duration));
        let mut duration = intent_end.saturating_sub(intent.onset);
        if duration == 0 {
            if end_override.is_some() {
                duration = 1;
            } else {
                return 0.0;
            }
        }
        if tick < intent.onset {
            return 0.0;
        }
        let release_end = intent_end.saturating_add(self.release_ticks);
        if tick >= release_end {
            return 0.0;
        }

        let pos = tick.saturating_sub(intent.onset);
        let attack_len = self.attack_ticks.min(duration);
        if attack_len == 0 {
            return 0.0;
        }
        let attack = if pos < attack_len {
            (pos.saturating_add(1) as f32 / attack_len as f32).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let release = if tick >= intent_end {
            if self.release_ticks == 0 {
                0.0
            } else {
                let remain = release_end.saturating_sub(tick);
                (remain as f32 / self.release_ticks as f32).clamp(0.0, 1.0)
            }
        } else {
            1.0
        };

        (attack * release).clamp(0.0, 1.0)
    }

    pub fn release_ticks(&self) -> Tick {
        self.release_ticks
    }
}
