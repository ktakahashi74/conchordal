use crate::config::AppConfig;
use crate::core::meter::MeterNetwork;
#[cfg(feature = "native")]
use crate::scenario::Action;
use crate::scenario::{Scenario, control::PhonationControl};

#[derive(Default)]
pub(crate) struct FeltConfig;

impl FeltConfig {
    #[inline(always)]
    pub(crate) fn from_scene(_: &AppConfig, _: &Scenario) -> Self {
        Self
    }

    #[inline(always)]
    pub(crate) fn requested(&self) -> bool {
        false
    }

    #[inline(always)]
    pub(crate) fn weight(_: &PhonationControl) -> f32 {
        0.0
    }

    #[inline(always)]
    pub(crate) fn setup_meter(&self, _: &mut MeterNetwork, _: f32, _: usize) {}

    #[cfg(feature = "native")]
    #[inline(always)]
    pub(crate) fn before_action(_: &Action, _: Option<&mut MeterNetwork>) {}
}
