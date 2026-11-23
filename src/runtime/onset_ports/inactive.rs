use super::temporal_ports::TemporalObservation;
use super::*;
use crate::life::arrival_cost::ArrivalSources;
use crate::life::voice::Voice;
use crate::temporal_cognition::capture::BodyCapture;

#[derive(Default)]
pub(super) struct OnsetConfig;

impl OnsetConfig {
    #[inline(always)]
    pub(super) fn from_config(_: &AppConfig, _: bool) -> Self {
        Self
    }

    #[inline(always)]
    pub(super) fn prepare(&self) -> OnsetRuntime {
        OnsetRuntime
    }

    #[inline(always)]
    pub(super) fn configure(&self, _: &mut [Voice], _: bool) -> bool {
        false
    }

    #[inline(always)]
    pub(super) fn arrival<'a>(
        &self,
        _: &'a TemporalObservation,
        _: &'a BodyCapture,
    ) -> ArrivalSources<'a> {
        ArrivalSources::default()
    }
}

pub(super) struct OnsetRuntime;

impl OnsetRuntime {
    #[inline(always)]
    pub(super) fn request(&mut self, _: &mut [Voice], _: f32, _: Tick, _: bool) {}

    #[cfg(feature = "native")]
    #[inline(always)]
    pub(super) fn route(&mut self, _: &mut [Voice], _: &mut Option<JsonlReporter>, _: Tick) {}

    #[cfg(feature = "native")]
    #[inline(always)]
    pub(super) fn finish(&mut self, _: &mut [Voice], _: &mut Option<JsonlReporter>, _: Tick) {}

    #[inline(always)]
    pub(super) fn collect(&mut self, _: u64, _: &mut Voice) {}

    #[inline(always)]
    pub(super) fn emit(&mut self, _: &mut Option<JsonlReporter>, _: u32) {}
}
