const MAX_EXTRA_GATES: u64 = 4096;

use crate::life::arrival_cost::ArrivalContext;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::f32::consts::TAU;
use std::fmt;

use crate::core::consonance_kernel::sigmoid01_stable;
use crate::core::float::sanitize01;
use crate::core::modulation::NeuralRhythms;
use crate::core::temporal_expectation::TemporalForecast;
use crate::core::timebase::Tick;
use crate::life::control_adapters::phonation_config_from_spec;
use crate::life::gate_clock::next_gate_tick;
use crate::life::social_density::SocialDensityTrace;
use crate::life::temporal_participation::TemporalParticipation;
use crate::scenario::{
    DurationConfig, OnsetConfig, PhonationClockConfig, PhonationConfig, PhonationMode,
    PhonationSpec,
};

#[path = "phonation_clock.rs"]
mod clock;
#[path = "phonation_rules.rs"]
mod rules;
#[path = "phonation_timing.rs"]
mod timing;

use clock::COUPLING_PULL_K;
pub use clock::{CouplingClock, ParticipationClock, PhonationClock, ThetaGateClock};
pub(crate) use clock::{MAX_ONSET_RATE_HZ, ParticipationPrediction};
pub use rules::{DurationPlan, DurationRule, OnsetContext, OnsetRule};
pub use timing::{GateBoundary, ThetaGrid, TimingField};

pub type ToneId = u64;

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
pub struct OnsetKick {
    pub strength: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
pub struct ToneUpdate {
    pub target_freq_hz: Option<f32>,
    /// Final output target amp (linear).
    pub target_amp: Option<f32>,
    pub continuous_drive: Option<f32>,
    /// None preserves the current pulse; Some(None) disables it without an onset.
    #[cfg_attr(test, serde(default, skip_serializing_if = "Option::is_none"))]
    pub autonomous_pulse: Option<Option<crate::life::sound::AutonomousPulseSpec>>,
}

impl ToneUpdate {
    pub fn is_empty(&self) -> bool {
        self.target_freq_hz.is_none()
            && self.target_amp.is_none()
            && self.continuous_drive.is_none()
            && self.autonomous_pulse.is_none()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
pub enum ToneCmd {
    On {
        tone_id: ToneId,
        kick: OnsetKick,
    },
    Off {
        tone_id: ToneId,
        off_tick: Tick,
    },
    Update {
        tone_id: ToneId,
        at_tick: Option<Tick>,
        update: ToneUpdate,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct CandidatePoint {
    pub tick: Tick,
    pub gate: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct IntervalInput {
    pub gate: u64,
    pub tick: Tick,
    pub dt_sec: f32,
    pub weight: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct CoreTickCtx {
    pub now_tick: Tick,
    pub frame_end: Tick,
    pub fs: f32,
    pub rhythms: NeuralRhythms,
}

#[derive(Debug, Clone, Copy)]
pub struct CoreState {
    pub is_alive: bool,
    /// Whether a new onset may be materialized this tick. `false` suppresses
    /// onset emission (Hold and candidate-processing paths alike) without
    /// touching clock/candidate generation, so entrainment keeps advancing
    /// while a voice is gated (e.g. `phonate_when_viable()`) and can join the
    /// beat the moment it opens.
    pub onset_allowed: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct ToneOnEvent {
    pub tone_id: ToneId,
    pub onset_tick: Tick,
    pub participation_onset: bool,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
pub struct OnsetEvent {
    pub gate: u64,
    pub onset_tick: Tick,
    pub strength: f32,
}

struct CandidateStep {
    dt_sec: f32,
    exc_gate: f32,
    exc_slope: f32,
    weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingOff {
    off_tick: Tick,
    tone_id: ToneId,
}

impl Ord for PendingOff {
    fn cmp(&self, other: &Self) -> Ordering {
        self.off_tick
            .cmp(&other.off_tick)
            .then_with(|| self.tone_id.cmp(&other.tone_id))
    }
}

impl PartialOrd for PendingOff {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Default)]
struct HoldCore {
    note_on_sent: bool,
    tone_id: Option<ToneId>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RestPhase {
    Ready,
    Continue(Tick),
    Waiting(Tick),
    Gap { origin: Tick, until: Option<Tick> },
    Finished,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct RestState {
    pub periods: f32,
    pub phase: RestPhase,
}

pub struct PhonationEngine {
    pub(crate) phrase_boundary: Option<crate::life::phrase_boundary::PhraseBoundary>,
    pub(crate) expression_active: bool,
    pub(crate) rest: Option<RestState>,
    pub(crate) span_gap: Option<crate::life::span_gap::SpanGap>,
    pub(crate) sustain_participation: bool,
    #[cfg(test)]
    pub(crate) synchronized_counts: [usize; 4],
    pub clock: PhonationClock,
    pub onset_rule: OnsetRule,
    pub duration_rule: DurationRule,
    pub mode: PhonationMode,
    measure_accent: f32,
    hold: HoldCore,
    pub next_tone_id: ToneId,
    last_tick: Option<Tick>,
    active_notes: u32,
    pending_off: BinaryHeap<Reverse<PendingOff>>,
    scratch_candidates: Vec<CandidatePoint>,
    scratch_merged: Vec<CandidatePoint>,
    scratch_grid: ThetaGrid,
    scratch_field: TimingField,
}

impl fmt::Debug for PhonationEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PhonationEngine")
            .field("next_tone_id", &self.next_tone_id)
            .field("active_notes", &self.active_notes)
            .field("mode", &self.mode)
            .finish()
    }
}

impl PhonationEngine {
    pub(crate) fn tracks_tails(&self) -> bool {
        self.rest.is_some() || self.span_gap.is_some()
    }

    pub(crate) fn advance_span(&mut self, at: Tick) {
        if let Some(gap) = &mut self.span_gap {
            if let Some(RestState {
                phase: RestPhase::Gap {
                    until: Some(until), ..
                },
                ..
            }) = self.rest
                && at >= until
            {
                gap.advance(until.max(gap.last_tick.unwrap_or(until)), true);
                if let Some(rest) = &mut self.rest {
                    rest.phase = RestPhase::Ready;
                }
            }
            let frozen = self.rest.is_some_and(|rest| {
                matches!(
                    rest.phase,
                    RestPhase::Waiting(_) | RestPhase::Gap { .. } | RestPhase::Finished
                )
            });
            gap.advance(at, frozen);
        }
    }

    pub(crate) fn set_span_pitch(&mut self, at: Tick, current: f32, next: f32) {
        self.advance_span(at);
        if let Some(gap) = &mut self.span_gap {
            gap.pitch = current;
            gap.next_pitch = next;
        }
    }

    fn span_opportunity(&mut self, at: Tick, fs: f32, out: &mut Vec<ToneCmd>) {
        let frozen = self.rest.is_some_and(|rest| {
            matches!(
                rest.phase,
                RestPhase::Waiting(_) | RestPhase::Gap { .. } | RestPhase::Finished
            )
        });
        let Some(gap) = &mut self.span_gap else {
            return;
        };
        gap.opportunity(at, frozen);
        if gap.phase == crate::life::span_gap::Phase::GapRequested && !frozen {
            if let Some(tone_id) = self.hold.tone_id.take() {
                self.pending_off.retain(|off| off.0.tone_id != tone_id);
                if !gap.acoustic_quiet {
                    out.push(ToneCmd::Off {
                        tone_id,
                        off_tick: at,
                    });
                }
                self.active_notes = self.active_notes.saturating_sub(1);
                if let Some(rest) = &mut self.rest {
                    rest.phase = RestPhase::Waiting(at);
                }
                gap.off(at);
                if gap.acoustic_quiet {
                    self.rest_tail_finished(at, fs);
                }
            } else {
                gap.off(at);
                gap.quiet(at, Some(at));
            }
        }
    }

    pub(crate) fn reserve_synchronized_hop(&mut self, hop: usize, fs: f32) -> usize {
        // Phase pull is bounded by the existing half-cycle error range.
        let count = (hop as f64 / f64::from(fs)
            * f64::from(MAX_ONSET_RATE_HZ)
            * f64::from(1.0 + COUPLING_PULL_K * 0.5))
        .ceil() as usize
            + 1;
        self.scratch_candidates.reserve(count);
        self.scratch_merged.reserve(count);
        let hold = if self.rest.is_some() {
            self.duration_rule
                .estimate_hold(1.0, 0.0)
                .unwrap_or(0.0)
                .ceil()
                .min(MAX_EXTRA_GATES as f32) as usize
        } else {
            0
        };
        self.scratch_grid.boundaries.reserve(count + hold + 2);
        self.scratch_field.e_gate.reserve(count);
        count
    }

    pub(crate) fn prepare_synchronized_hop(&mut self, ctx: &CoreTickCtx) {
        if self.sustain_participation && matches!(self.clock, PhonationClock::Coupling(_)) {
            self.scratch_candidates.clear();
            self.clock
                .gather_candidates(ctx, &mut self.scratch_candidates);
            Self::merge_candidates_into(&mut self.scratch_merged, &mut self.scratch_candidates);
        }
    }

    pub(crate) fn enable_rest(&mut self, duration: &DurationConfig, periods: f32) {
        assert!(self.sustain_participation);
        self.duration_rule = DurationRule::from_config(duration);
        self.rest = Some(RestState {
            periods,
            phase: RestPhase::Ready,
        });
        self.pending_off.reserve(1);
    }

    pub(crate) fn rest_tail_finished(&mut self, at: Tick, fs: f32) {
        let period = self.clock.intrinsic_period_sec();
        if let Some(rest) = self.rest.as_mut()
            && let RestPhase::Waiting(released) = rest.phase
            && at >= released
        {
            let until = period.map(|p| {
                let ticks = (p * f64::from(fs) * f64::from(rest.periods)).round() as Tick;
                at.saturating_add(ticks)
            });
            rest.phase = RestPhase::Gap { origin: at, until };
        }
        if let Some(gap) = &mut self.span_gap {
            let resume = match self.rest.map(|rest| rest.phase) {
                Some(RestPhase::Gap { until, .. }) => until.map(|until| until.max(at)),
                Some(RestPhase::Waiting(_) | RestPhase::Finished) => None,
                _ => Some(at),
            };
            gap.quiet(at, resume);
        }
    }

    pub(crate) fn rest_resume(&mut self, at: Tick) {
        if let Some(rest) = self.rest.as_mut()
            && let RestPhase::Gap {
                until: Some(until), ..
            } = rest.phase
            && at >= until
        {
            rest.phase = RestPhase::Ready;
        }
    }

    pub(crate) fn observe_participation_context(
        &mut self,
        own: &crate::core::temporal_expectation::OwnSoundHistory,
        emit: impl FnMut(crate::life::temporal_participation::ParticipationContextUpdate),
    ) {
        if let PhonationClock::Participation(clock) = &mut self.clock
            && let Some(policy) = clock.policy.as_mut()
        {
            policy.observe_context(own, emit);
        }
    }

    pub(crate) fn set_outcome_forecast(&mut self, forecast: Option<&TemporalForecast>) {
        if let PhonationClock::Participation(clock) = &mut self.clock {
            clock.outcome_forecast = forecast.copied();
        }
    }

    pub(crate) fn drain_participation_predictions(
        &mut self,
    ) -> impl Iterator<Item = ParticipationPrediction> + '_ {
        match &mut self.clock {
            PhonationClock::Participation(clock) => Some(&mut clock.issued_predictions),
            _ => None,
        }
        .into_iter()
        .flat_map(|predictions| predictions.drain(..))
    }

    pub(crate) fn uses_participation(&self) -> bool {
        matches!(self.clock, PhonationClock::Participation(_))
    }

    pub(crate) fn needs_temporal_context(&self, now: Tick, end: Tick) -> bool {
        matches!(&self.clock, PhonationClock::Participation(clock)
            if clock.coupling > 0.0 && clock.policy.as_ref().is_none_or(|p| p.needs_forecast(now, end)))
    }

    pub(crate) fn set_sound_adsr(&mut self, adsr: Option<crate::life::sound::ToneAdsr>) {
        if let PhonationClock::Participation(clock) = &mut self.clock {
            clock.adsr = adsr;
        }
    }

    pub(crate) fn set_onset_comparison(
        &mut self,
        config: Option<crate::config::TemporalOnsetComparisonConfig>,
    ) {
        if let PhonationClock::Participation(clock) = &mut self.clock {
            clock.onset_comparison = config;
        }
    }

    pub(crate) fn set_decision_trace(&mut self, enabled: bool) {
        if let PhonationClock::Participation(clock) = &mut self.clock {
            clock.decision_trace = enabled;
        }
    }

    pub(crate) fn drain_participation_decisions(
        &mut self,
    ) -> impl Iterator<Item = crate::life::temporal_participation::ParticipationDecision> + '_ {
        match &mut self.clock {
            PhonationClock::Participation(clock) => clock.policy.as_mut(),
            _ => None,
        }
        .into_iter()
        .flat_map(|policy| policy.drain_decisions())
    }

    pub(crate) fn footprint_hold(&self) -> Option<(f32, Option<crate::life::sound::ToneAdsr>)> {
        match &self.clock {
            PhonationClock::Participation(clock) => clock.footprint_hold(),
            _ => None,
        }
    }

    /// I11-1 §4.2 hop step: adopt the identity and send at most one request.
    /// Returns true when the supplier had already been asked for this identity.
    pub(crate) fn footprint_hop(
        &mut self,
        identity: crate::life::onset_footprint::Identity,
        now: Tick,
        recipe: &crate::life::onset_footprint::Recipe,
        send: impl FnOnce(crate::life::onset_footprint::Request) -> bool,
    ) -> bool {
        match &mut self.clock {
            PhonationClock::Participation(clock) => {
                clock.footprint.hop(identity, now, recipe, send)
            }
            _ => false,
        }
    }

    /// Returns true when this Voice was waiting for the dropped reply.
    pub(crate) fn footprint_release(
        &mut self,
        identity: crate::life::onset_footprint::Identity,
    ) -> bool {
        match &mut self.clock {
            PhonationClock::Participation(clock) => clock.footprint.release(identity),
            _ => false,
        }
    }

    /// Returns true when the record was discarded because its request was superseded.
    pub(crate) fn footprint_receive(
        &mut self,
        record: &crate::life::onset_footprint::Record,
    ) -> bool {
        match &mut self.clock {
            PhonationClock::Participation(clock) => clock.footprint.receive(record),
            _ => false,
        }
    }

    pub(crate) fn set_temporal_context(
        &mut self,
        forecast: Option<&TemporalForecast>,
        own_profile: [f32; 3],
    ) {
        if let PhonationClock::Participation(clock) = &mut self.clock {
            clock.forecast = forecast.copied();
            clock.own_profile = own_profile;
        }
    }

    pub fn from_spec(spec: &PhonationSpec, seed: u64) -> Self {
        let config = phonation_config_from_spec(spec);
        Self::from_config(&config, seed)
    }

    pub(crate) fn from_config(config: &PhonationConfig, seed: u64) -> Self {
        let onset_rule = OnsetRule::from_config(&config.onset, seed);
        let duration_rule = DurationRule::from_config(&config.duration);
        Self {
            phrase_boundary: None,
            expression_active: false,
            rest: None,
            span_gap: None,
            #[cfg(test)]
            synchronized_counts: [0; 4],
            sustain_participation: config.mode == PhonationMode::Hold
                && matches!(
                    config.clock,
                    PhonationClockConfig::Participation { .. }
                        | PhonationClockConfig::Coupling { .. }
                ),
            clock: PhonationClock::from_config(&config.clock, seed),
            onset_rule,
            duration_rule,
            mode: config.mode,
            measure_accent: sanitize01(config.measure_accent),
            hold: HoldCore::default(),
            next_tone_id: 0,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),

            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
            scratch_grid: ThetaGrid::default(),
            scratch_field: TimingField::default(),
        }
    }

    pub fn has_active_notes(&self) -> bool {
        self.active_notes > 0
    }

    fn tick_hold(
        &mut self,
        ctx: &CoreTickCtx,
        state: &CoreState,
        min_allowed_onset_tick: Option<Tick>,
        out_cmds: &mut Vec<ToneCmd>,
        out_events: &mut Vec<ToneOnEvent>,
        out_onsets: &mut Vec<OnsetEvent>,
    ) {
        self.advance_span(ctx.now_tick);
        let allow_onset = self.rest.is_none_or(|rest| rest.phase == RestPhase::Ready)
            && self.span_gap.as_ref().is_none_or(|gap| gap.allows_on())
            && min_allowed_onset_tick
                .map(|min_tick| ctx.now_tick >= min_tick)
                .unwrap_or(true);
        if allow_onset && state.onset_allowed && state.is_alive && !self.hold.note_on_sent {
            let tone_id = self.next_tone_id;
            self.next_tone_id = self.next_tone_id.wrapping_add(1);
            self.hold.note_on_sent = true;
            self.hold.tone_id = Some(tone_id);
            if let Some(gap) = &mut self.span_gap {
                gap.on(ctx.now_tick, false);
            }
            out_cmds.push(ToneCmd::On {
                tone_id,
                kick: OnsetKick { strength: 1.0 },
            });
            self.active_notes = self.active_notes.saturating_add(1);
            out_events.push(ToneOnEvent {
                tone_id,
                onset_tick: ctx.now_tick,
                participation_onset: false,
            });
            out_onsets.push(OnsetEvent {
                gate: 0,
                onset_tick: ctx.now_tick,
                strength: 1.0,
            });
        }
        if !state.is_alive
            && let Some(gap) = &mut self.span_gap
        {
            gap.finish(ctx.now_tick);
        }
        if !state.is_alive
            && let Some(tone_id) = self.hold.tone_id.take()
        {
            out_cmds.push(ToneCmd::Off {
                tone_id,
                off_tick: ctx.now_tick,
            });
            self.active_notes = self.active_notes.saturating_sub(1);
        }
    }

    /// Restart a held tone at the requested sample, independent of pitch selection.
    pub(crate) fn rearticulate_hold(
        &mut self,
        ctx: &CoreTickCtx,
        state: &CoreState,
        out_cmds: &mut Vec<ToneCmd>,
        out_events: &mut Vec<ToneOnEvent>,
        out_onsets: &mut Vec<OnsetEvent>,
    ) {
        debug_assert!(matches!(self.mode, PhonationMode::Hold));
        // Release ignores a held-tone restart on another phonation mode.
        if !matches!(self.mode, PhonationMode::Hold) {
            return;
        }
        if !state.is_alive || !state.onset_allowed {
            return;
        }
        if let Some(tone_id) = self.hold.tone_id.take() {
            out_cmds.push(ToneCmd::Off {
                tone_id,
                off_tick: ctx.now_tick,
            });
            self.active_notes = self.active_notes.saturating_sub(1);
        }
        self.hold.note_on_sent = false;
        self.tick_hold(ctx, state, None, out_cmds, out_events, out_onsets);
    }

    pub(crate) fn initial_participating_hold(
        &mut self,
        ctx: &CoreTickCtx,
        state: &CoreState,
        cmds: &mut Vec<ToneCmd>,
        events: &mut Vec<ToneOnEvent>,
        onsets: &mut Vec<OnsetEvent>,
    ) {
        let before = onsets.len();
        if !state.is_alive
            && let Some(rest) = self.rest.as_mut()
        {
            self.pending_off.clear();
            rest.phase = RestPhase::Finished;
        }
        self.tick_hold(ctx, state, None, cmds, events, onsets);
        if onsets.len() > before
            && let PhonationClock::Participation(c) = &mut self.clock
        {
            c.initial_hold_on = Some(ctx.now_tick);
        }
        if onsets.len() > before && self.rest.is_some() {
            debug_assert!(self.hold.tone_id.is_some());
            // Release skips duration scheduling when no held tone can receive its Off.
            let Some(tone_id) = self.hold.tone_id else {
                return;
            };
            let onset = OnsetContext {
                tone_id,
                tick: ctx.now_tick,
                gate: 0,
                exc_gate: 1.0,
                exc_slope: 0.0,
            };
            let mut grid = std::mem::take(&mut self.scratch_grid);
            grid.rebuild_from_candidates(&[CandidatePoint {
                tick: ctx.now_tick,
                gate: 0,
            }]);
            let period = self.clock.fixed_period_ticks(ctx.fs);
            self.schedule_duration(onset, ctx, &mut grid, period);
            self.scratch_grid = grid;
        }
    }

    pub(crate) fn set_adoption_candidate(&mut self, at: Option<Tick>) {
        if let PhonationClock::Participation(c) = &mut self.clock {
            c.adoption_candidate = at;
        }
    }

    fn schedule_note_off(&mut self, tone_id: ToneId, off_tick: Tick) {
        self.pending_off
            .push(Reverse(PendingOff { off_tick, tone_id }));
    }

    fn drain_note_offs(&mut self, up_to_tick: Tick, out_cmds: &mut Vec<ToneCmd>) {
        while let Some(Reverse(next)) = self.pending_off.peek().copied() {
            if next.off_tick > up_to_tick {
                break;
            }
            self.advance_span(next.off_tick);
            self.pending_off.pop();
            if let Some(gap) = &mut self.span_gap {
                gap.off(next.off_tick);
            }
            if let Some(rest) = self.rest.as_mut()
                && self.hold.tone_id == Some(next.tone_id)
            {
                self.hold.tone_id = None;
                rest.phase = RestPhase::Waiting(next.off_tick);
            }
            out_cmds.push(ToneCmd::Off {
                tone_id: next.tone_id,
                off_tick: next.off_tick,
            });
            self.active_notes = self.active_notes.saturating_sub(1);
        }
    }

    fn schedule_hold_theta(
        &mut self,
        onset: OnsetContext,
        hold_theta: f32,
        ctx: &CoreTickCtx,
        timing_grid: &mut ThetaGrid,
        fixed_period: Option<f64>,
    ) -> Tick {
        let hold_theta = if hold_theta.is_finite() {
            hold_theta.max(0.0)
        } else {
            0.0
        };
        let off_pos = onset.gate as f64 + hold_theta as f64;
        let mapped_off = match &self.clock {
            PhonationClock::Coupling(clock)
                if clock.expression.is_some_and(|e| {
                    onset.tick >= (e.start * f64::from(ctx.fs)).round() as Tick
                }) =>
            {
                clock.expression_gate_tick(off_pos, ctx.fs)
            }
            _ => None,
        };
        let off_tick_opt =
            mapped_off.or_else(|| Self::tick_for_gate_pos(off_pos, ctx, timing_grid, fixed_period));
        let fallback = timing_grid
            .boundaries
            .last()
            .map(|boundary| boundary.tick)
            .unwrap_or(onset.tick);
        let off_tick = off_tick_opt.unwrap_or(fallback).max(onset.tick);
        self.schedule_note_off(onset.tone_id, off_tick);
        off_tick
    }

    /// Map a fractional gate position to a tick on the timing grid.
    fn tick_for_gate_pos(
        gate_pos: f64,
        ctx: &CoreTickCtx,
        timing_grid: &mut ThetaGrid,
        fixed_period: Option<f64>,
    ) -> Option<Tick> {
        if !gate_pos.is_finite() || gate_pos < 0.0 {
            return None;
        }
        let off_gate_f = gate_pos.floor();
        if off_gate_f >= u64::MAX as f64 {
            return None;
        }
        let mut off_gate = off_gate_f as u64;
        let mut off_phase = (gate_pos - off_gate_f).clamp(0.0, 1.0);
        if off_phase >= 1.0 {
            off_phase = 0.0;
            off_gate = off_gate.saturating_add(1);
        }
        timing_grid.ensure_boundaries_until(ctx, off_gate.saturating_add(1), fixed_period);
        timing_grid.tick_at(off_gate, off_phase as f32)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tick(
        &mut self,
        ctx: &CoreTickCtx,
        state: &CoreState,
        social: Option<&SocialDensityTrace>,
        social_coupling: f32,
        extra_gate_gain: f32,
        min_allowed_onset_tick: Option<Tick>,
        out_cmds: &mut Vec<ToneCmd>,
        out_events: &mut Vec<ToneOnEvent>,
        out_onsets: &mut Vec<OnsetEvent>,
    ) {
        if let PhonationClock::Participation(clock) = &mut self.clock {
            clock.hop_frames = ctx.frame_end.saturating_sub(ctx.now_tick);
        }
        self.tick_with_arrival(
            ctx,
            state,
            social,
            social_coupling,
            extra_gate_gain,
            min_allowed_onset_tick,
            None,
            out_cmds,
            out_events,
            out_onsets,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tick_with_arrival(
        &mut self,
        ctx: &CoreTickCtx,
        state: &CoreState,
        social: Option<&SocialDensityTrace>,
        social_coupling: f32,
        extra_gate_gain: f32,
        min_allowed_onset_tick: Option<Tick>,
        arrival: Option<&ArrivalContext<'_>>,
        out_cmds: &mut Vec<ToneCmd>,
        out_events: &mut Vec<ToneOnEvent>,
        out_onsets: &mut Vec<OnsetEvent>,
    ) {
        if matches!(self.mode, PhonationMode::Hold) && !self.sustain_participation {
            self.tick_hold(
                ctx,
                state,
                min_allowed_onset_tick,
                out_cmds,
                out_events,
                out_onsets,
            );
            return;
        }
        if self.sustain_participation && matches!(self.clock, PhonationClock::Coupling(_)) {
            let merged = std::mem::take(&mut self.scratch_merged);
            let start = merged.partition_point(|c| c.tick < ctx.now_tick);
            let end = merged.partition_point(|c| c.tick < ctx.frame_end);
            let candidates = &merged[start..end];
            let mut grid = std::mem::take(&mut self.scratch_grid);
            let mut field = std::mem::take(&mut self.scratch_field);
            grid.rebuild_from_candidates(candidates);
            field.rebuild_flat(&grid);
            self.process_candidates(
                ctx,
                candidates,
                &mut grid,
                &field,
                state,
                min_allowed_onset_tick,
                out_cmds,
                out_events,
                out_onsets,
            );
            self.scratch_merged = merged;
            self.scratch_grid = grid;
            self.scratch_field = field;
            return;
        }
        if self.uses_participation() {
            if let PhonationClock::Participation(clock) = &mut self.clock {
                clock.hold_theta = self.duration_rule.estimate_hold(1.0, 0.0).unwrap_or(0.0);
            }
            let mut timing_grid = std::mem::take(&mut self.scratch_grid);
            let mut timing_field = std::mem::take(&mut self.scratch_field);
            let mut cursor = ctx.now_tick;
            while cursor < ctx.frame_end {
                let (boundary, segment_ctx, effective_state) = if self.rest.is_some() {
                    self.drain_note_offs(cursor, out_cmds);
                    self.advance_span(cursor);
                    self.rest_resume(cursor);
                    let boundary = self
                        .rest
                        .and_then(|rest| match rest.phase {
                            RestPhase::Continue(off) => Some(off),
                            RestPhase::Gap { until, .. } => until,
                            _ => None,
                        })
                        .filter(|b| *b > cursor)
                        .unwrap_or(ctx.frame_end)
                        .min(ctx.frame_end);
                    let mut state = *state;
                    state.onset_allowed &=
                        self.rest.is_some_and(|rest| rest.phase == RestPhase::Ready);
                    (
                        boundary,
                        CoreTickCtx {
                            frame_end: boundary,
                            ..*ctx
                        },
                        state,
                    )
                } else {
                    (ctx.frame_end, *ctx, *state)
                };
                let PhonationClock::Participation(clock) = &mut self.clock else {
                    unreachable!()
                };
                if let Some(shape) = clock.shape.as_mut() {
                    shape.has_tone = self.hold.tone_id.is_some();
                }
                let Some(candidate) = clock.candidate(
                    &segment_ctx,
                    cursor,
                    effective_state.is_alive && effective_state.onset_allowed,
                    arrival,
                ) else {
                    if boundary == ctx.frame_end {
                        break;
                    }
                    self.drain_note_offs(boundary.saturating_sub(1), out_cmds);
                    cursor = boundary;
                    continue;
                };
                timing_grid.rebuild_from_candidates(&[candidate]);
                timing_field.rebuild_flat(&timing_grid);
                let before = out_onsets.len();
                // Process in timestamp order so earlier outcomes inform later candidates.
                let segment = CoreTickCtx {
                    frame_end: candidate.tick.saturating_add(1),
                    ..*ctx
                };
                self.process_candidates(
                    &segment,
                    &[candidate],
                    &mut timing_grid,
                    &timing_field,
                    &effective_state,
                    min_allowed_onset_tick,
                    out_cmds,
                    out_events,
                    out_onsets,
                );
                let PhonationClock::Participation(clock) = &mut self.clock else {
                    unreachable!()
                };
                clock.resolve(candidate.tick, out_onsets.len() > before);
                if out_onsets.len() > before
                    && let Some(tie) = clock.shape.as_mut().and_then(|s| s.tie.as_mut())
                {
                    tie.pending_pitch = false;
                }
                cursor = candidate.tick.saturating_add(1);
            }
            self.drain_note_offs(ctx.frame_end.saturating_sub(1), out_cmds);
            self.scratch_grid = timing_grid;
            self.scratch_field = timing_field;
            return;
        }
        self.scratch_candidates.clear();
        self.clock
            .gather_candidates(ctx, &mut self.scratch_candidates);
        Self::merge_candidates_into(&mut self.scratch_merged, &mut self.scratch_candidates);
        let merged = std::mem::take(&mut self.scratch_merged);
        let mut timing_grid = std::mem::take(&mut self.scratch_grid);
        timing_grid.rebuild_from_candidates(&merged);
        // Fixed-rate and coupling clocks own their own timing (an external grid
        // or an internal entrained oscillator): their onsets must not be gated by
        // the adaptive `env_open`, or they would re-couple to theta (metric loses
        // its steady pulse; flow droplets lock to one period; the coupling clock's
        // entrainment is double-counted). Use a flat field.
        let mut timing_field = std::mem::take(&mut self.scratch_field);
        if self.clock.uses_flat_field() {
            timing_field.rebuild_flat(&timing_grid);
        } else {
            timing_field.rebuild_from(
                ctx,
                &timing_grid,
                social.map(|trace| (trace, social_coupling)),
                extra_gate_gain,
            );
        }
        self.process_candidates(
            ctx,
            &merged,
            &mut timing_grid,
            &timing_field,
            state,
            min_allowed_onset_tick,
            out_cmds,
            out_events,
            out_onsets,
        );
        self.scratch_merged = merged;
        self.scratch_grid = timing_grid;
        self.scratch_field = timing_field;
    }

    fn step_for_candidate(
        &self,
        c: &CandidatePoint,
        timing_field: &TimingField,
        prev_gate_exc: Option<f32>,
        fs: f32,
    ) -> CandidateStep {
        let dt_sec = if self.last_tick == Some(c.tick) {
            0.0
        } else {
            match self.last_tick {
                Some(prev_tick) => {
                    if c.tick < prev_tick {
                        debug_assert!(false, "candidate ticks must be non-decreasing");
                        0.0
                    } else {
                        let dt_ticks = c.tick - prev_tick;
                        let dt = dt_ticks as f32 / fs;
                        if dt.is_finite() { dt } else { 0.0 }
                    }
                }
                None => 0.0,
            }
        };
        let exc_gate = timing_field.e(c.gate);
        let exc_slope = match prev_gate_exc {
            Some(prev_exc) => (exc_gate - prev_exc).clamp(-1.0, 1.0),
            None => 0.0,
        };
        CandidateStep {
            dt_sec,
            exc_gate,
            exc_slope,
            weight: exc_gate,
        }
    }

    fn schedule_duration(
        &mut self,
        onset: OnsetContext,
        ctx: &CoreTickCtx,
        timing_grid: &mut ThetaGrid,
        fixed_period: Option<f64>,
    ) -> Option<Tick> {
        let plan = self.duration_rule.on_note_on(onset);
        match plan {
            DurationPlan::HoldTheta(hold_theta) => {
                let off =
                    self.schedule_hold_theta(onset, hold_theta, ctx, timing_grid, fixed_period);
                if let Some(a) = self.rest.as_mut() {
                    a.phase = RestPhase::Continue(off);
                }
                Some(off)
            }
            DurationPlan::None => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_candidates(
        &mut self,
        ctx: &CoreTickCtx,
        candidates: &[CandidatePoint],
        timing_grid: &mut ThetaGrid,
        timing_field: &TimingField,
        state: &CoreState,
        min_allowed_onset_tick: Option<Tick>,
        out_cmds: &mut Vec<ToneCmd>,
        out_events: &mut Vec<ToneOnEvent>,
        out_onsets: &mut Vec<OnsetEvent>,
    ) {
        let mut prev_gate_exc: Option<f32> = None;
        let fixed_period = self.clock.fixed_period_ticks(ctx.fs);
        self.drain_note_offs(ctx.now_tick, out_cmds);
        for c in candidates {
            let phrase = self
                .phrase_boundary
                .as_mut()
                .and_then(|p| p.opportunity(c.tick));
            self.drain_note_offs(c.tick, out_cmds);
            self.advance_span(c.tick);
            if state.is_alive {
                self.span_opportunity(c.tick, ctx.fs, out_cmds);
            }
            let span_allowed = self.span_gap.as_ref().is_none_or(|gap| gap.allows_on());
            let synchronized =
                self.sustain_participation && matches!(self.clock, PhonationClock::Coupling(_));
            if synchronized {
                self.rest_resume(c.tick);
            }
            let rest_allowed = span_allowed
                && (!synchronized || self.rest.is_none_or(|r| r.phase == RestPhase::Ready));
            let live_tone = self.hold.tone_id.is_some();
            if synchronized && let Some(shape) = self.clock.shape_mut() {
                shape.has_tone = live_tone;
                if shape.enabled {
                    shape.propose(
                        crate::life::shape_policy::Shape::Base,
                        c.tick,
                        !state.is_alive || !state.onset_allowed || !rest_allowed,
                    );
                    shape.veto_plan =
                        shape.state == crate::life::shape_policy::Shape::Connect && live_tone;
                }
            }
            let onset_before = out_onsets.len();
            #[cfg(test)]
            if synchronized {
                self.synchronized_counts[0] += 1;
            }
            let step = self.step_for_candidate(c, timing_field, prev_gate_exc, ctx.fs);
            let allow_onset = min_allowed_onset_tick
                .map(|min_tick| c.tick >= min_tick)
                .unwrap_or(true);
            let input = IntervalInput {
                gate: c.gate,
                tick: c.tick,
                dt_sec: step.dt_sec,
                weight: step.weight,
            };
            let intentional = allow_onset
                && rest_allowed
                && state.is_alive
                && state.onset_allowed
                && self.hold.tone_id.is_some()
                && match &mut self.clock {
                    PhonationClock::Participation(clock) => {
                        clock.shape.as_mut().is_some_and(|s| s.veto_onset(c.tick))
                    }
                    PhonationClock::Coupling(clock) if synchronized => {
                        clock.shape.as_mut().is_some_and(|s| s.veto_onset(c.tick))
                    }
                    _ => false,
                };
            if allow_onset
                && rest_allowed
                && !intentional
                && state.onset_allowed
                && let Some(mut kick) = self.onset_rule.on_candidate(&input, state)
            {
                let measure = ctx.rhythms.measure;
                if self.measure_accent > 0.0
                    && ctx.rhythms.measure_ratio > 0
                    && measure.freq_hz > 0.0
                    && measure.freq_hz.is_finite()
                    && measure.phase.is_finite()
                {
                    let dt = c.tick.saturating_sub(ctx.now_tick) as f32 / ctx.fs;
                    let phase = measure.phase + TAU * measure.freq_hz * dt;
                    // Weak, symmetric emphasis; no onset is scheduled by the measure.
                    kick.strength *=
                        1.0 + 0.35 * self.measure_accent * sanitize01(measure.alpha) * phase.cos();
                }
                if self.sustain_participation {
                    if out_events.iter().any(|e| e.onset_tick == c.tick) {
                        #[cfg(test)]
                        if synchronized {
                            self.synchronized_counts[1] += 1;
                        }
                        self.last_tick = Some(c.tick);
                        prev_gate_exc = Some(step.exc_gate);
                        continue;
                    }
                    if let Some(tone_id) = self.hold.tone_id.take() {
                        out_cmds.push(ToneCmd::Off {
                            tone_id,
                            off_tick: c.tick,
                        });
                        self.active_notes = self.active_notes.saturating_sub(1);
                    }
                }
                let tone_id = self.next_tone_id;
                if let Some(mut reservation) = phrase {
                    reservation.tone = tone_id;
                    self.phrase_boundary.as_mut().unwrap().selected = Some(reservation);
                }
                self.next_tone_id = self.next_tone_id.wrapping_add(1);
                if self.sustain_participation {
                    self.hold.tone_id = Some(tone_id);
                    self.hold.note_on_sent = true;
                }
                if let Some(gap) = &mut self.span_gap {
                    gap.on(c.tick, true);
                }
                out_cmds.push(ToneCmd::On { tone_id, kick });
                self.active_notes = self.active_notes.saturating_add(1);
                out_events.push(ToneOnEvent {
                    tone_id,
                    onset_tick: c.tick,
                    participation_onset: true,
                });
                out_onsets.push(OnsetEvent {
                    gate: c.gate,
                    onset_tick: c.tick,
                    strength: kick.strength,
                });
                let onset = OnsetContext {
                    tone_id,
                    tick: c.tick,
                    gate: c.gate,
                    exc_gate: step.exc_gate,
                    exc_slope: step.exc_slope,
                };
                if !self.sustain_participation || self.rest.is_some() {
                    self.schedule_duration(onset, ctx, timing_grid, fixed_period);
                }
            }
            if synchronized {
                let sounded = out_onsets.len() > onset_before;
                #[cfg(test)]
                {
                    self.synchronized_counts[if sounded {
                        1
                    } else if intentional {
                        2
                    } else {
                        3
                    }] += 1;
                }
                if sounded && let Some(tie) = self.clock.shape_mut().and_then(|s| s.tie.as_mut()) {
                    tie.pending_pitch = false;
                }
            }
            self.last_tick = Some(c.tick);
            prev_gate_exc = Some(step.exc_gate);
        }
        self.drain_note_offs(ctx.frame_end.saturating_sub(1), out_cmds);
    }

    /// Sort by (tick, gate) and drop duplicate (tick, gate) candidates.
    #[cfg(test)]
    fn merge_candidates(mut candidates: Vec<CandidatePoint>) -> Vec<CandidatePoint> {
        let mut merged = Vec::with_capacity(candidates.len());
        Self::merge_candidates_into(&mut merged, &mut candidates);
        merged
    }

    fn merge_candidates_into(out: &mut Vec<CandidatePoint>, candidates: &mut Vec<CandidatePoint>) {
        // Unstable sort: equal (tick, gate) keys are identical points, and it
        // avoids the stable sort's scratch allocation on the per-hop path.
        candidates.sort_unstable_by(|a, b| a.tick.cmp(&b.tick).then_with(|| a.gate.cmp(&b.gate)));
        out.clear();
        out.reserve(candidates.len());
        for candidate in candidates.drain(..) {
            if let Some(last) = out.last()
                && last.tick == candidate.tick
                && last.gate == candidate.gate
            {
                continue;
            }
            out.push(candidate);
        }
    }
}

#[cfg(test)]
#[path = "phonation_engine_tests.rs"]
mod tests;
