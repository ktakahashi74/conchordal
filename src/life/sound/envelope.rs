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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) replacement_level: Option<f32>,
}

impl Envelope {
    pub(crate) fn with_replacement_release(mut self, tick: Tick) -> Self {
        if tick < self.hold_end {
            let level = self.gain_at(tick);
            // Keep the already sounding attack intact, including a future Off
            // delivered at the start of a renderer hop.
            self.attack_ticks = self.attack_len();
            self.replacement_level = Some(level);
            self.hold_end = tick;
            self.release_end = tick.saturating_add(self.release_ticks);
        }
        self
    }

    pub(crate) fn with_release(mut self, tick: Tick) -> Self {
        if self.replacement_level.is_some() {
            return self.with_replacement_release(tick);
        }
        if tick < self.hold_end {
            self.hold_end = tick;
            self.release_end = tick.saturating_add(self.release_ticks);
        }
        self
    }

    pub(crate) fn attack_len(&self) -> Tick {
        if self.replacement_level.is_some() {
            self.attack_ticks
        } else {
            self.attack_ticks
                .min(self.hold_end.saturating_sub(self.onset).max(1))
        }
    }

    pub(crate) fn gain_at(&self, tick: Tick) -> f32 {
        if tick < self.onset || tick >= self.release_end {
            return 0.0;
        }

        let pos = tick.saturating_sub(self.onset);
        let attack_len = self.attack_len();

        // Pre-release level: attack → decay → sustain
        let level = if tick >= self.hold_end
            && let Some(level) = self.replacement_level
        {
            level
        } else if attack_len > 0 && pos < attack_len {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_release_preserves_attack_and_earliest_off_without_restarting_tail() {
        let original = Envelope {
            onset: 10,
            hold_end: Tick::MAX,
            release_end: Tick::MAX,
            attack_ticks: 100,
            decay_ticks: 0,
            sustain_level: 1.,
            decay_lambda: 0.,
            release_ticks: 40,
            replacement_level: None,
        };
        let released = original.with_replacement_release(60);
        for tick in 0..60 {
            assert_eq!(released.gain_at(tick), original.gain_at(tick));
        }
        assert_eq!(released.gain_at(60), 0.51);
        assert_eq!(released.gain_at(80), 0.255);
        assert_eq!(released.gain_at(100), 0.);
        assert_eq!(released.with_release(70), released);
        assert_eq!(released.with_replacement_release(60), released);
        let earlier = released.with_release(40);
        assert_eq!(earlier.attack_len(), original.attack_len());
        assert_eq!(earlier.gain_at(40), original.gain_at(40));
        assert_eq!(earlier.gain_at(60), original.gain_at(40) * 0.5);
        // Ordinary finite and explicitly released envelopes retain their old law.
        assert_eq!(original.with_release(60).gain_at(10), 1. / 50.);
        assert_eq!(original.with_release(60).gain_at(60), 1.);
    }
}
