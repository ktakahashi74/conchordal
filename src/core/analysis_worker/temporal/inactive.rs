use super::AnalysisStream;

#[derive(Default)]
pub(crate) struct AnalysisTemporalInput;

pub(super) struct AnalysisTemporal;

pub(super) struct AnalysisTemporalFrame;

impl AnalysisTemporalInput {
    #[inline(always)]
    pub(super) fn start(self, _: &AnalysisStream) -> AnalysisTemporal {
        AnalysisTemporal
    }
}

impl AnalysisTemporal {
    #[inline(always)]
    pub(super) fn observe_raw(&mut self, _: u64, _: &[f32]) -> AnalysisTemporalFrame {
        AnalysisTemporalFrame
    }

    #[inline(always)]
    pub(super) fn observe_spectrum(
        &mut self,
        _: u64,
        _: &[f32],
        _: &[f32],
        _: AnalysisTemporalFrame,
    ) {
    }

    #[inline(always)]
    pub(super) fn finish(&mut self) {}
}
