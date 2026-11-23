use super::*;
use crate::core::meter::MeterNetwork;
use crate::life::{
    flora::TemporalObservation,
    voice::{PhonationBatch, Voice},
};
use crate::scenario::control::PhonationControl;

#[derive(Default)]
pub(crate) struct FeltVoice;

impl FeltVoice {
    #[inline(always)]
    pub(crate) fn new(_: &PhonationControl, _: u64, _: u64, _: u32) -> Self {
        Self
    }

    #[inline(always)]
    pub(crate) fn set_hop_input(
        _: &mut [Voice],
        _: Option<&MeterNetwork>,
        _: &TemporalObservation,
        _: Tick,
        _: bool,
    ) {
    }

    #[inline(always)]
    pub(crate) fn rhythms(&self, original: NeuralRhythms, _: Tick, _: f32) -> NeuralRhythms {
        original
    }

    #[inline(always)]
    pub(crate) fn executed(&mut self, _: &PhonationBatch) {}

    #[inline(always)]
    pub(super) fn hold_position(&self, pos: f64) -> f64 {
        pos
    }

    #[inline(always)]
    pub(super) fn take_points(&mut self) -> Option<Vec<CandidatePoint>> {
        None
    }

    #[inline(always)]
    pub(super) fn restore_points(&mut self, _: Option<Vec<CandidatePoint>>) {}

    #[inline(always)]
    pub(super) fn trace_len(&self) -> usize {
        0
    }

    #[inline(always)]
    pub(super) fn allow_candidate(
        &mut self,
        _: &CandidatePoint,
        _: OnsetKick,
        _: &CoreTickCtx,
    ) -> bool {
        true
    }

    #[inline(always)]
    pub(super) fn record_veto(&mut self, _: &CandidatePoint, _: &CoreTickCtx, _: usize) {}
}

impl PhonationEngine {
    #[inline(always)]
    pub(crate) fn reserve_felt_hop(&mut self, _: usize, _: f32) {}

    #[inline(always)]
    pub(super) fn try_gather_felt_candidates(&mut self, _: &CoreTickCtx) -> bool {
        false
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn deliver_felt_candidate(
        &mut self,
        _: &CoreTickCtx,
        _: &[CandidatePoint],
        _: &CandidatePoint,
        _: &TimingField,
        _: Option<f32>,
        _: &CoreState,
        _: Option<f64>,
        _: &mut Vec<ToneCmd>,
        _: &mut Vec<ToneOnEvent>,
        _: &mut Vec<OnsetEvent>,
    ) -> bool {
        false
    }
}
