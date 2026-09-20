//! Frozen renderer gain, not acoustic energy or a forecast of future control messages.

use crate::core::timebase::Tick;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
#[cfg_attr(test, derive(serde::Deserialize))]
pub struct Envelope {
    pub(crate) onset: Tick,
    pub(crate) hold_end: Tick,
    pub(crate) release_end: Tick,
    pub(crate) attack_ticks: Tick,
    pub(crate) decay_ticks: Tick,
    pub(crate) sustain_level: f32,
    pub(crate) decay_lambda: f32,
    pub(crate) release_ticks: Tick,
}

impl Envelope {
    pub(crate) fn with_release(mut self, tick: Tick) -> Self {
        if tick < self.hold_end {
            self.hold_end = tick;
            self.release_end = tick.saturating_add(self.release_ticks);
        }
        self
    }

    pub(crate) fn gain_at(&self, tick: Tick) -> f32 {
        if tick < self.onset || tick >= self.release_end {
            return 0.0;
        }

        let duration_ticks = self.hold_end.saturating_sub(self.onset).max(1);
        let pos = tick.saturating_sub(self.onset);
        let attack_len = self.attack_ticks.min(duration_ticks);

        // Pre-release level: attack → decay → sustain
        let level = if attack_len > 0 && pos < attack_len {
            (pos.saturating_add(1) as f32 / attack_len as f32).clamp(0.0, 1.0)
        } else if self.decay_ticks > 0 && pos < attack_len.saturating_add(self.decay_ticks) {
            let decay_pos = pos.saturating_sub(attack_len);
            self.sustain_level
                + (1.0 - self.sustain_level) * (-self.decay_lambda * decay_pos as f32).exp()
        } else {
            self.sustain_level
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

        (level * release).clamp(0.0, 1.0)
    }
}
