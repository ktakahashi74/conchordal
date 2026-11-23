use super::*;

#[derive(Debug)]
pub(super) enum RecurrenceInput {
    Events(Recurrence),
    Continuous(Box<continuous::State>),
}

impl RecurrenceInput {
    pub(super) fn new() -> Self {
        Self::Events(Recurrence::new())
    }

    pub(super) fn events(&self) -> Option<&Recurrence> {
        match self {
            Self::Events(s) => Some(s),
            Self::Continuous(_) => None,
        }
    }

    pub(super) fn issue(&mut self) {
        if let Self::Events(s) = self {
            s.issue();
        }
    }

    pub(super) fn confirm(
        &mut self,
        dt: f32,
        onset: Onset,
        start: u64,
        hop: usize,
        fs: f32,
    ) -> Option<f32> {
        match self {
            RecurrenceInput::Continuous(state) => {
                let onset_tick = state.fast_start as f64
                    + dt as f64 * onset.frac as f64 * state.sample_rate() as f64;
                state
                    .pending
                    .filter(|p| {
                        !state.suppressed
                            && onset.fired
                            && onset_tick > p.received as f64
                            && onset_tick > p.use_cut as f64
                    })
                    .map(|p| p.lag_sec as f32)
            }
            RecurrenceInput::Events(state) => {
                state.observe(start, hop, fs, onset.fired, onset.frac)
            }
        }
    }

    pub(super) fn accept(&mut self, period: f32, ratio_support: f32) {
        match self {
            Self::Events(s) => s.accept(period, ratio_support),
            Self::Continuous(s) => s.accept(ratio_support),
        }
    }

    pub(super) fn seed(&mut self) {
        match self {
            Self::Events(s) => s.reset_evidence(),
            Self::Continuous(s) => s.seed(),
        }
    }

    pub(super) fn confidence(&self, beat_plv: f32, presence: f32) -> f32 {
        match self {
            RecurrenceInput::Continuous(state) => state.confidence(beat_plv, presence),
            RecurrenceInput::Events(state) => {
                let support = if state.multi {
                    beat_plv.max(state.support())
                } else {
                    beat_plv
                };
                (support * presence).clamp(0.0, 1.0)
            }
        }
    }
}
