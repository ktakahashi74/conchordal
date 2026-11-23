#[cfg(feature = "native")]
use crate::core::timebase::Tick;
use crate::life::voice::Voice;
use crate::scenario::control::Routing;

#[derive(Default)]
pub(crate) struct BodyCapture;

#[derive(Default)]
pub(crate) struct BodyCaptureSlot;

impl BodyCapture {
    #[inline(always)]
    pub(crate) fn prepare(&mut self, _: &[Voice], _: crate::core::timebase::Tick) {}

    #[cfg(feature = "native")]
    #[inline(always)]
    pub(crate) fn begin(&mut self, _: Tick) {}

    #[inline(always)]
    pub(crate) fn slot(&self, _: u64, _: u32) -> BodyCaptureSlot {
        BodyCaptureSlot
    }

    #[inline(always)]
    pub(crate) fn record_block(&mut self, _: &BodyCaptureSlot, _: &[f32], _: Routing) {}

    #[inline(always)]
    pub(crate) fn record_sample(&mut self, _: &BodyCaptureSlot, _: usize, _: f32, _: Routing) {}

    #[cfg(feature = "native")]
    #[inline(always)]
    pub(crate) fn end(&mut self) {}
}
