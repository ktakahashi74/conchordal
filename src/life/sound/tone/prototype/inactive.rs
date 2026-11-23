use super::{NeuralRhythms, Tick, Tone, ToneUpdate};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct PrototypeTone;

impl PrototypeTone {
    #[inline(always)]
    pub(super) fn enabled(&self) -> bool {
        false
    }

    #[inline(always)]
    pub(super) fn blocked(&self) -> bool {
        false
    }

    #[inline(always)]
    pub(super) fn retired(&self) -> Option<bool> {
        None
    }

    #[inline(always)]
    pub(super) fn rejects_update(&self, _: Tick, _: Tick, _: &ToneUpdate) -> bool {
        false
    }
}

impl Tone {
    #[inline(always)]
    pub(super) fn render_prototype_sample(
        &mut self,
        _: Tick,
        _: f32,
        _: &NeuralRhythms,
    ) -> Option<f32> {
        None
    }
}
