use super::*;
use crate::core::meter::MeterNetwork;
use crate::life::{
    flora::TemporalObservation,
    voice::{PhonationBatch, Voice},
};
use crate::scenario::control::PhonationControl;

#[path = "../../felt_rhythm.rs"]
mod rhythm;
pub(crate) use rhythm::FeltRhythmRecord;
use rhythm::{FeltRhythm, FeltRhythmInput};

#[derive(Default)]
pub(crate) struct FeltVoice(Option<Box<FeltRhythm>>);

impl FeltVoice {
    pub(crate) fn new(control: &PhonationControl, seed: u64, id: u64, generation: u32) -> Self {
        Self((control.syncopate > 0.0).then(|| {
            Box::new(FeltRhythm::new(
                control.syncopate,
                seed ^ id.rotate_left(17)
                    ^ u64::from(generation).rotate_left(39)
                    ^ 0x4645_4c54_5f42_4541,
            ))
        }))
    }

    pub(crate) fn set_hop_input(
        voices: &mut [Voice],
        meter: Option<&MeterNetwork>,
        rhythm: &TemporalObservation,
        now: Tick,
        trace: bool,
    ) {
        let input = voices
            .iter()
            .any(|v| v.phonation_engine.felt.0.is_some())
            .then(|| {
                meter.and_then(|m| m.felt_beat()).and_then(|snapshot| {
                    let end = rhythm.observed_end?;
                    (rhythm.known && end == now).then_some(FeltRhythmInput {
                        snapshot,
                        observed_end: end,
                    })
                })
            })
            .flatten();
        for voice in voices {
            if let Some(felt) = &mut voice.phonation_engine.felt.0 {
                felt.set_input(input);
                felt.enable_trace(trace);
            }
        }
    }

    pub(crate) fn rhythms(&self, original: NeuralRhythms, now: Tick, fs: f32) -> NeuralRhythms {
        self.0
            .as_ref()
            .map_or(original, |felt| felt.rhythms(original, now, fs))
    }

    pub(crate) fn executed(&mut self, batch: &PhonationBatch) {
        if let Some(felt) = &mut self.0 {
            for tone in &batch.tones {
                felt.executed(tone.onset, tone.freq_hz);
            }
        }
    }

    pub(crate) fn drain_records(&mut self) -> Option<std::vec::Drain<'_, FeltRhythmRecord>> {
        self.0.as_mut().map(|felt| felt.trace.drain(..))
    }

    pub(super) fn hold_position(&self, pos: f64) -> f64 {
        pos + if self.0.as_ref().is_some_and(|felt| felt.delivering_half) {
            0.5
        } else {
            0.0
        }
    }

    pub(super) fn take_points(&mut self) -> Option<Vec<CandidatePoint>> {
        self.0.as_mut().map(|felt| std::mem::take(&mut felt.points))
    }

    pub(super) fn restore_points(&mut self, points: Option<Vec<CandidatePoint>>) {
        if let Some(points) = points {
            self.0.as_mut().unwrap().points = points;
        }
    }

    pub(super) fn trace_len(&self) -> usize {
        self.0.as_ref().map_or(0, |felt| felt.trace.len())
    }

    pub(super) fn allow_candidate(
        &mut self,
        c: &CandidatePoint,
        kick: OnsetKick,
        ctx: &CoreTickCtx,
    ) -> bool {
        self.0.as_mut().is_none_or(|felt| felt.decide(c, kick, ctx))
    }

    pub(super) fn record_veto(
        &mut self,
        c: &CandidatePoint,
        ctx: &CoreTickCtx,
        trace_before: usize,
    ) {
        if let Some(felt) = &mut self.0
            && felt.trace_enabled
            && felt.trace.len() == trace_before
        {
            felt.veto(c, ctx);
        }
    }
}

impl PhonationEngine {
    pub(crate) fn reserve_felt_hop(&mut self, hop: usize, fs: f32) {
        if self.felt.0.is_none() {
            return;
        }
        let count = self.reserve_synchronized_hop(hop, fs);
        let hold = self
            .duration_rule
            .estimate_hold(1.0, 0.0)
            .unwrap_or(0.0)
            .ceil()
            .clamp(0.0, MAX_EXTRA_GATES as f32) as usize;
        let felt = self.felt.0.as_mut().unwrap();
        felt.points.reserve(2 * count);
        felt.delivery_grid.boundaries.reserve(hold + 2);
    }

    pub(super) fn try_gather_felt_candidates(&mut self, ctx: &CoreTickCtx) -> bool {
        if let (Some(felt), PhonationClock::Coupling(clock)) = (&mut self.felt.0, &mut self.clock) {
            felt.points.clear();
            clock.gather_rhythm_subdivided(
                ctx.fs,
                ctx.now_tick..ctx.frame_end,
                &ctx.rhythms,
                &mut self.scratch_candidates,
                Some((2, &mut felt.points)),
            );
            true
        } else {
            false
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn deliver_felt_candidate(
        &mut self,
        ctx: &CoreTickCtx,
        candidates: &[CandidatePoint],
        c: &CandidatePoint,
        timing_field: &TimingField,
        prev_gate_exc: Option<f32>,
        state: &CoreState,
        fixed_period: Option<f64>,
        out_cmds: &mut Vec<ToneCmd>,
        out_events: &mut Vec<ToneOnEvent>,
        out_onsets: &mut Vec<OnsetEvent>,
    ) -> bool {
        let half = !candidates
            .iter()
            .any(|n| n.tick == c.tick && n.gate == c.gate);
        if let Some(kick) = self.felt.0.as_mut().and_then(|f| f.take_due(c, half))
            && state.is_alive
            && state.onset_allowed
        {
            self.drain_note_offs(c.tick, out_cmds);
            let step = self.step_for_candidate(c, timing_field, prev_gate_exc, ctx.fs);
            let felt = self.felt.0.as_mut().unwrap();
            felt.delivering_half = half;
            let mut grid = std::mem::take(&mut felt.delivery_grid);
            grid.rebuild_from_candidates(std::slice::from_ref(c));
            self.emit_candidate(
                ctx,
                c,
                kick,
                &step,
                None,
                &mut grid,
                fixed_period,
                out_cmds,
                out_events,
                out_onsets,
            );
            let felt = self.felt.0.as_mut().unwrap();
            felt.delivery_grid = grid;
            felt.delivering_half = false;
        }
        half
    }
}
