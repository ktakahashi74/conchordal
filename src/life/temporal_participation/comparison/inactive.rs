use super::*;

#[derive(Default)]
pub(super) struct OnsetComparison;

pub(super) struct FootprintTrace;

impl FootprintTrace {
    #[inline(always)]
    pub(super) fn new(_: f64) -> Self {
        Self
    }
}

impl OnsetComparison {
    #[inline(always)]
    pub(super) fn footprint_clocks(&self) -> (Option<u64>, Option<u64>) {
        (None, None)
    }

    #[inline(always)]
    pub(super) fn fill_footprint(
        &self,
        _: &TemporalParticipation,
        _: &mut [(f64, f32); 64],
        _: u64,
        _: f32,
        _: f32,
    ) -> Option<(u8, &'static str, bool, u64, f64)> {
        None
    }

    #[inline(always)]
    pub(super) fn record_decision(
        &mut self,
        _: DecisionComparison<'_>,
        _: bool,
        _: f32,
        _: (i8, f64, f32),
        _: (f32, f32),
    ) {
    }
}

#[derive(Default)]
pub(super) struct CandidateTrace;

impl CandidateTrace {
    #[inline(always)]
    pub(super) fn distance(&mut self, _: f32) {}

    #[inline(always)]
    pub(super) fn overlap(&mut self, _: f32) {}

    #[inline(always)]
    pub(super) fn energy(&mut self, _: usize, _: [f32; 3], _: bool) {}
}

pub(super) struct DecisionComparison<'a>(std::marker::PhantomData<&'a ()>);

impl<'a> DecisionComparison<'a> {
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        _: &TemporalParticipation,
        _: u64,
        _: f64,
        _: f64,
        _: bool,
        footprint: &DecisionFootprint,
        _: Option<&TemporalForecast>,
        _: Option<&ArrivalContext<'a>>,
        _: &[Option<f64>; 23],
    ) -> Self {
        let _ = &footprint.footprint_trace;
        Self(std::marker::PhantomData)
    }

    #[inline(always)]
    pub(super) fn tracing(&self) -> bool {
        false
    }

    #[inline(always)]
    pub(super) fn rejects_adoption(&self, _: usize, _: f64, _: f64, _: &[Option<f64>; 23]) -> bool {
        false
    }

    #[inline(always)]
    pub(super) fn arrival_cost(&self, _: f64, _: f64, _: f32) -> (Option<f64>, Option<f32>) {
        (None, None)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_candidate(
        &mut self,
        _: usize,
        _: i8,
        _: f64,
        _: f64,
        _: Option<ParticipationContextPrediction>,
        _: CandidateTrace,
        _: Option<f64>,
        _: Option<f32>,
        _: f32,
    ) {
    }
}
