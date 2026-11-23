use crate::config::AppConfig;
use crate::core::landscape::{Landscape, LandscapeParams};
use crate::core::modulation::NeuralRhythms;
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::core::timebase::Tick;
use crate::life::schedule_renderer::{SiteBuses, SourcePcm};
use crate::life::sound::BodyKind;
use crate::life::voice::Voice;

#[derive(Default)]
pub(crate) struct BodyMetabolismPort;

impl BodyMetabolismPort {
    #[inline(always)]
    pub(crate) fn from_config(_: &AppConfig, _: &RtNsgtKernelLog2, _: &LandscapeParams) -> Self {
        Self
    }

    #[inline(always)]
    pub(crate) fn prepare_sites(&mut self, _: Option<&SiteBuses>) {}

    #[inline(always)]
    pub(crate) fn prepare(&mut self, _: &[Voice], _: Tick, _: &LandscapeParams) {}

    #[inline(always)]
    pub(crate) fn source_pcm(&mut self) -> &mut [SourcePcm] {
        &mut []
    }

    #[inline(always)]
    pub(crate) fn observe_at_sites(
        &mut self,
        _: Tick,
        _: &[f32],
        _: Option<&SiteBuses>,
        _: &LandscapeParams,
    ) {
    }

    #[inline(always)]
    pub(crate) fn try_commit(
        &mut self,
        _: &mut Voice,
        _: f32,
        _: &NeuralRhythms,
        _: &Landscape,
        _: f32,
    ) -> bool {
        false
    }

    #[inline(always)]
    #[expect(
        dead_code,
        reason = "Only the research body model queries factory conformance"
    )]
    pub(crate) fn model_kind(_: BodyKind, _: bool) -> Option<BodyKind> {
        None
    }
}
