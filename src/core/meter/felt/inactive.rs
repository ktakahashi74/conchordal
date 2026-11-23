use super::{MeterNetwork, Onset};

#[derive(Debug, Default)]
pub(super) struct FeltMeter;

impl FeltMeter {
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn observe_onset(
        &mut self,
        _: f32,
        _: f32,
        _: Onset,
        _: bool,
        _: f32,
        _: f32,
        _: bool,
    ) {
    }

    #[inline(always)]
    pub(super) fn blocks_period_update(&self, _: f32, _: f32) -> bool {
        false
    }
}

impl MeterNetwork {
    #[inline(always)]
    pub(crate) fn needs_felt_bands(&self) -> bool {
        false
    }

    #[inline(always)]
    pub(crate) fn set_felt_bands(&mut self, _: u64, _: Option<[f32; 3]>) {}
}
