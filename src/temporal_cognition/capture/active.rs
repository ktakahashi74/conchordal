use crate::core::timebase::Tick;
use crate::life::voice::Voice;
use crate::scenario::control::Routing;
use crate::temporal_cognition::body::Capture;

#[derive(Default)]
pub(crate) struct BodyCapture(Option<Capture>);

#[derive(Default)]
pub(crate) struct BodyCaptureSlot(Option<(usize, u32)>);

impl BodyCapture {
    #[cfg(feature = "native")]
    pub(crate) fn new(capture: Option<Capture>) -> Self {
        Self(capture)
    }

    pub(crate) fn as_ref(&self) -> Option<&Capture> {
        self.0.as_ref()
    }

    pub(crate) fn as_mut(&mut self) -> Option<&mut Capture> {
        self.0.as_mut()
    }

    pub(crate) fn prepare(&mut self, voices: &[Voice], now: Tick) {
        if let Some(capture) = self.as_mut() {
            capture.prepare(
                voices
                    .iter()
                    .map(|v| (v.id(), v.metadata.generation, v.body_snapshot())),
                now,
            );
        }
    }

    #[cfg(feature = "native")]
    pub(crate) fn begin(&mut self, now: Tick) {
        if let Some(capture) = self.as_mut() {
            capture.begin(now);
        }
    }

    pub(crate) fn slot(&self, id: u64, generation: u32) -> BodyCaptureSlot {
        BodyCaptureSlot(
            self.as_ref()
                .and_then(|capture| capture.token(id, generation)),
        )
    }

    pub(crate) fn record_block(&mut self, slot: &BodyCaptureSlot, pcm: &[f32], routing: Routing) {
        if let (Some(capture), Some(slot)) = (self.as_mut(), slot.0) {
            for (idx, &sample) in pcm.iter().enumerate() {
                capture.sample(slot, idx, sample, routing);
            }
        }
    }

    pub(crate) fn record_sample(
        &mut self,
        slot: &BodyCaptureSlot,
        idx: usize,
        sample: f32,
        routing: Routing,
    ) {
        if let (Some(capture), Some(slot)) = (self.as_mut(), slot.0) {
            capture.sample(slot, idx, sample, routing);
        }
    }

    #[cfg(feature = "native")]
    pub(crate) fn end(&mut self) {
        if let Some(capture) = self.as_mut() {
            capture.end();
        }
    }
}
