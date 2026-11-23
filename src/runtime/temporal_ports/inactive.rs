use super::wiring::AnalysisRuntimeCore;
use super::*;
use crate::core::analysis_worker::AnalysisTemporalInput;
use crate::core::stream::loo::PcmPacket;
use crate::runtime_profile::{BackgroundTemporalProfile, TemporalMeterProfile};
use crate::temporal_cognition::{capture::BodyCapture, ui_observation::TemporalUi};

#[derive(Default)]
pub(super) struct TemporalConfig;

impl TemporalConfig {
    #[inline(always)]
    pub(super) fn from_config(_: &AppConfig) -> Self {
        Self
    }

    #[cfg(test)]
    #[inline(always)]
    pub(super) fn from_enabled(enabled: bool) -> Self {
        assert!(!enabled);
        Self
    }

    #[inline(always)]
    pub(super) fn t2_enabled(&self) -> bool {
        false
    }

    #[inline(always)]
    pub(super) fn make_meter(&self, _: u32, _: u64) -> MeterNetwork {
        MeterNetwork::new()
    }

    #[inline(always)]
    pub(super) fn observing(_: &Scenario) -> bool {
        false
    }

    #[inline(always)]
    pub(super) fn supports_observe() -> bool {
        false
    }

    #[inline(always)]
    pub(super) fn validate_step(_: &AppConfig, _: &Scenario) -> Result<(), &'static str> {
        Ok(())
    }
}

#[derive(Default)]
pub(super) struct TemporalChannels;

#[derive(Default)]
pub(super) struct TemporalObservation;

#[derive(Default)]
pub(super) struct TemporalWiring {
    pub(super) habitat: (bool, AnalysisTemporalInput),
    pub(super) presentation: (bool, AnalysisTemporalInput),
    pub(super) channels: TemporalChannels,
    pub(super) capture: BodyCapture,
}

impl TemporalWiring {
    #[inline(always)]
    pub(super) fn prepare(
        _: &AppConfig,
        _: &Scenario,
        _: &AnalysisRuntimeCore,
        _: u32,
        _: bool,
        _: Duration,
    ) -> Self {
        Self::default()
    }
}

impl TemporalObservation {
    #[inline(always)]
    pub(super) fn initialize_capture(&mut self, _: &BodyCapture) {}

    #[inline(always)]
    #[cfg(feature = "native")]
    pub(super) fn set_initial_ui(&self, _: &mut UiFrame) {}

    #[inline(always)]
    pub(super) fn ui_snapshot(&self) -> TemporalUi {
        TemporalUi
    }

    #[inline(always)]
    pub(super) fn meter_profile(&self) -> TemporalMeterProfile {
        TemporalMeterProfile {}
    }

    #[inline(always)]
    #[cfg(feature = "native")]
    pub(super) fn background_profile(&self, _: &TemporalChannels) -> BackgroundTemporalProfile {
        BackgroundTemporalProfile {
            shared: [None; 2],
            body: None,
        }
    }
}

#[inline(always)]
#[cfg(feature = "native")]
pub(super) fn read_initial(_: &mut WorkerState, _: &WorkerChannels) {}

#[inline(always)]
pub(super) fn read_live(_: &mut WorkerState, _: &WorkerChannels) {}

#[inline(always)]
pub(super) fn refresh_body(_: &mut WorkerState) {}

#[inline(always)]
pub(super) fn emit_observations(_: &mut WorkerState, _: f64) {}

#[inline(always)]
#[cfg(feature = "native")]
pub(super) fn finish_capture(_: &mut WorkerState) {}

#[inline(always)]
#[cfg(feature = "native")]
pub(super) fn finish_analysis(_: &mut WorkerState, _: &mut WorkerChannels, _: &WorkerConfig) {}

#[inline(always)]
pub(super) fn send_before_meter(
    _: &mut WorkerState,
    _: &WorkerChannels,
    _: &WorkerConfig,
    _: &Arc<[f32]>,
    _: &mut Option<PcmPacket>,
) {
}

#[inline(always)]
pub(super) fn update_meter(
    _: &mut WorkerState,
    _: &mut TemporalChannels,
    _: &WorkerConfig,
    _: bool,
    _: f32,
    _: f32,
) -> Option<crate::core::meter::MeterState> {
    None
}
