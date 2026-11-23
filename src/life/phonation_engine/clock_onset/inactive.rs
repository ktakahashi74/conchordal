use super::TemporalParticipation;

#[derive(Default)]
pub(super) struct ClockOnset;

impl ClockOnset {
    #[inline(always)]
    pub(super) fn apply(&self, _: &mut TemporalParticipation) {}
}
