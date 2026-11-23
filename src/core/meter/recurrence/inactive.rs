use super::*;

#[derive(Debug)]
pub(super) struct RecurrenceInput(Recurrence);

impl RecurrenceInput {
    pub(super) fn new() -> Self {
        Self(Recurrence::new())
    }

    #[inline(always)]
    pub(super) fn events(&self) -> Option<&Recurrence> {
        Some(&self.0)
    }

    #[inline(always)]
    pub(super) fn issue(&mut self) {
        self.0.issue();
    }

    #[inline(always)]
    pub(super) fn confirm(
        &mut self,
        _: f32,
        onset: Onset,
        start: u64,
        hop: usize,
        fs: f32,
    ) -> Option<f32> {
        self.0.observe(start, hop, fs, onset.fired, onset.frac)
    }

    #[inline(always)]
    pub(super) fn accept(&mut self, period: f32, ratio_support: f32) {
        self.0.accept(period, ratio_support);
    }

    #[inline(always)]
    pub(super) fn seed(&mut self) {
        self.0.reset_evidence();
    }

    #[inline(always)]
    pub(super) fn confidence(&self, beat_plv: f32, presence: f32) -> f32 {
        let support = if self.0.multi {
            beat_plv.max(self.0.support())
        } else {
            beat_plv
        };
        (support * presence).clamp(0.0, 1.0)
    }
}
