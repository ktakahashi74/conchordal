use super::{ScheduleRenderer, Tick, Tone};

impl ScheduleRenderer {
    #[inline(always)]
    pub(super) fn retire_prototype_sources(&mut self, _: Tick) {}

    #[inline(always)]
    pub(super) fn enable_prototype_tone(&self, _: &mut Tone) {}
}
