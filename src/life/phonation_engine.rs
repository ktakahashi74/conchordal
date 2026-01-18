use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::f32::consts::TAU;
use std::fmt;

use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::Tick;
use crate::life::gate_clock::next_gate_tick;
use crate::life::scenario::{
    PhonationClockConfig, PhonationConfig, PhonationConnectConfig, PhonationIntervalConfig,
    PhonationMode, SubThetaModConfig,
};
use crate::life::social_density::SocialDensityTrace;
use tracing::{debug, warn};

pub type NoteId = u64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhonationKick {
    Planned { strength: f32 },
}

impl PhonationKick {
    pub fn strength(self) -> f32 {
        match self {
            PhonationKick::Planned { strength } => strength,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PhonationUpdate {
    pub target_freq_hz: Option<f32>,
    /// Final output target amp (linear).
    pub target_amp: Option<f32>,
}

impl PhonationUpdate {
    pub fn is_empty(&self) -> bool {
        self.target_freq_hz.is_none() && self.target_amp.is_none()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhonationCmd {
    NoteOn {
        note_id: NoteId,
        kick: PhonationKick,
    },
    NoteOff {
        note_id: NoteId,
        off_tick: Tick,
    },
    Update {
        note_id: NoteId,
        at_tick: Option<Tick>,
        update: PhonationUpdate,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClockSource {
    GateBoundary,
    Subdivision { n: u32 },
    InternalPhase,
}

#[derive(Debug, Clone)]
pub struct CandidatePoint {
    pub tick: Tick,
    pub gate: u64,
    pub theta_pos: f64,
    pub phase_in_gate: f32,
    pub sources: Vec<ClockSource>,
}

#[derive(Debug, Clone, Copy)]
pub struct IntervalInput {
    pub gate: u64,
    pub tick: Tick,
    pub dt_theta: f32,
    pub dt_sec: f32,
    pub weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateBoundary {
    pub gate: u64,
    pub tick: Tick,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ThetaGrid {
    pub boundaries: Vec<GateBoundary>,
}

impl ThetaGrid {
    /// Input candidates must be sorted by tick ascending.
    pub fn from_candidates(candidates: &[CandidatePoint]) -> Self {
        debug_assert!(
            candidates.windows(2).all(|p| p[0].tick <= p[1].tick),
            "candidates must be sorted by tick"
        );
        let mut boundaries = Vec::with_capacity(candidates.len());
        let mut last_gate = None;
        let mut last_tick = None;
        for candidate in candidates {
            if candidate.phase_in_gate != 0.0 {
                continue;
            }
            if let Some(prev_gate) = last_gate {
                if candidate.gate < prev_gate {
                    panic!("candidate gates must be non-decreasing");
                }
                if candidate.gate == prev_gate {
                    continue;
                }
                debug_assert!(
                    candidate.gate == prev_gate.saturating_add(1),
                    "theta grid gates must be contiguous"
                );
                if candidate.gate != prev_gate.saturating_add(1) {
                    continue;
                }
                if let Some(prev_tick) = last_tick {
                    debug_assert!(
                        candidate.tick >= prev_tick,
                        "theta grid ticks must be non-decreasing"
                    );
                    if candidate.tick < prev_tick {
                        continue;
                    }
                }
            }
            boundaries.push(GateBoundary {
                gate: candidate.gate,
                tick: candidate.tick,
            });
            last_gate = Some(candidate.gate);
            last_tick = Some(candidate.tick);
        }
        Self { boundaries }
    }

    fn boundaries_for_gate(&self, gate: u64) -> Option<(GateBoundary, GateBoundary)> {
        if self.boundaries.is_empty() {
            return None;
        }
        let start_gate = self.boundaries[0].gate;
        let idx = gate.checked_sub(start_gate)? as usize;
        let next_idx = idx.checked_add(1)?;
        let b0 = *self.boundaries.get(idx)?;
        let b1 = *self.boundaries.get(next_idx)?;
        if b0.gate != gate || b1.gate != gate.saturating_add(1) {
            return None;
        }
        Some((b0, b1))
    }

    /// Returns a tick by rounding the interpolated position, clamping interior phases away from
    /// gate boundary ticks.
    pub fn tick_at(&self, gate: u64, phase_in_gate: f32) -> Option<Tick> {
        let (b0, b1) = self.boundaries_for_gate(gate)?;
        let phase = phase_in_gate.clamp(0.0, 1.0);
        if phase > 0.0 && phase < 1.0 && b1.tick <= b0.tick.saturating_add(1) {
            return None;
        }
        let dt = b1.tick.saturating_sub(b0.tick) as f64;
        let tick_f = b0.tick as f64 + dt * phase as f64;
        let tick = tick_f.round() as Tick;
        if phase > 0.0 && phase < 1.0 {
            let lo = b0.tick.saturating_add(1);
            let hi = b1.tick.saturating_sub(1);
            if lo > hi {
                return None;
            }
            return Some(tick.clamp(lo, hi));
        }
        Some(tick)
    }

    pub fn ensure_boundaries_until(&mut self, ctx: &CoreTickCtx, target_gate_plus_1: u64) {
        const MAX_EXTRA_GATES: u64 = 4096;
        let Some(mut last) = self.boundaries.last().copied() else {
            return;
        };
        if last.gate >= target_gate_plus_1 {
            return;
        }
        let mut cursor_tick = last.tick;
        let mut next_gate = last.gate.saturating_add(1);
        let mut added = 0u64;
        while last.gate < target_gate_plus_1 && added < MAX_EXTRA_GATES {
            let search_tick = cursor_tick.saturating_add(1);
            let Some(mut next_tick) = next_gate_tick(search_tick, ctx.fs, ctx.rhythms.theta, 0.0)
            else {
                break;
            };
            // Failsafe: if the clock still returns a non-advancing tick, force monotonic progress.
            if next_tick <= cursor_tick {
                next_tick = cursor_tick.saturating_add(1);
            }
            if next_tick <= cursor_tick {
                break;
            }
            let boundary = GateBoundary {
                gate: next_gate,
                tick: next_tick,
            };
            self.boundaries.push(boundary);
            last = boundary;
            cursor_tick = next_tick;
            next_gate = next_gate.saturating_add(1);
            added += 1;
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimingField {
    pub start_gate: u64,
    pub e_gate: Vec<f32>,
}

impl TimingField {
    pub fn from_values(start_gate: u64, e_gate: Vec<f32>) -> Self {
        Self { start_gate, e_gate }
    }

    pub fn build_from(
        ctx: &CoreTickCtx,
        grid: &ThetaGrid,
        social: Option<(&SocialDensityTrace, f32)>,
        extra_gate_gain: f32,
    ) -> Self {
        const MAX_GATES_PER_HOP: u64 = 4096;
        if grid.boundaries.is_empty() {
            return Self {
                start_gate: 0,
                e_gate: Vec::new(),
            };
        }
        debug_assert!(
            grid.boundaries
                .windows(2)
                .all(|pair| pair[0].tick <= pair[1].tick)
        );
        debug_assert!(
            grid.boundaries
                .windows(2)
                .all(|pair| pair[0].gate < pair[1].gate)
        );
        let mut rhythms = ctx.rhythms;
        let mut cursor_tick = ctx.now_tick;
        let mut e_gate = Vec::with_capacity(grid.boundaries.len());
        let mut expected_gate = grid.boundaries[0].gate;
        let mut last_weight = None;
        for boundary in &grid.boundaries {
            if boundary.tick > cursor_tick {
                let dt_sec = (boundary.tick - cursor_tick) as f32 / ctx.fs;
                rhythms.advance_in_place(dt_sec);
                cursor_tick = boundary.tick;
            }
            let mut weight = rhythms.env_open.clamp(0.0, 1.0);
            match social {
                Some((trace, coupling)) if coupling != 0.0 => {
                    let density = trace.density_at(boundary.tick);
                    if density.is_finite() {
                        let arg = (coupling * density).clamp(-10.0, 10.0);
                        weight = (weight * arg.exp()).max(0.0);
                        if !weight.is_finite() {
                            weight = 0.0;
                        }
                    }
                }
                _ => {}
            }
            let extra_gain = if extra_gate_gain.is_finite() {
                extra_gate_gain.max(0.0)
            } else {
                1.0
            };
            weight *= extra_gain;
            if !weight.is_finite() {
                weight = 0.0;
            }
            if boundary.gate > expected_gate {
                let fill_weight = last_weight.unwrap_or(1.0);
                let missing = boundary.gate - expected_gate;
                if missing > MAX_GATES_PER_HOP {
                    panic!("timing field gap too large: missing={missing}");
                }
                // Fill gaps to keep gate-indexed lookups contiguous.
                e_gate.extend(std::iter::repeat_n(fill_weight, missing as usize));
            }
            e_gate.push(weight);
            last_weight = Some(weight);
            expected_gate = boundary.gate.saturating_add(1);
        }
        Self {
            start_gate: grid.boundaries[0].gate,
            e_gate,
        }
    }

    pub fn e(&self, gate: u64) -> f32 {
        if gate < self.start_gate {
            return 1.0;
        }
        let idx = gate - self.start_gate;
        self.e_gate.get(idx as usize).copied().unwrap_or(1.0)
    }
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
}

pub trait PhonationClock: Send {
    fn gather_candidates(&mut self, ctx: &CoreTickCtx, out: &mut Vec<CandidatePoint>);
}

#[derive(Debug, Default)]
pub struct ThetaGateClock {
    last_gate_tick: Option<Tick>,
    gate_index: u64,
    has_gate: bool,
}

impl ThetaGateClock {
    pub fn gate_index_hint(&self) -> u64 {
        if self.has_gate {
            self.gate_index + 1
        } else {
            0
        }
    }

    fn gather_candidates_with<F>(
        &mut self,
        ctx: &CoreTickCtx,
        out: &mut Vec<CandidatePoint>,
        mut next_gate_tick: F,
    ) where
        F: FnMut(Tick, &CoreTickCtx) -> Option<Tick>,
    {
        let mut cursor = ctx.now_tick;
        while cursor < ctx.frame_end {
            let gate_tick = match next_gate_tick(cursor, ctx) {
                Some(tick) => tick,
                None => return,
            };
            if gate_tick < cursor {
                cursor = cursor.saturating_add(1);
                continue;
            }
            if gate_tick < ctx.now_tick || gate_tick >= ctx.frame_end {
                return;
            }
            if self.last_gate_tick == Some(gate_tick) {
                cursor = gate_tick.saturating_add(1);
                continue;
            }
            if self.has_gate {
                self.gate_index = self.gate_index.saturating_add(1);
            } else {
                self.gate_index = 0;
                self.has_gate = true;
            }
            self.last_gate_tick = Some(gate_tick);
            out.push(CandidatePoint {
                tick: gate_tick,
                gate: self.gate_index,
                theta_pos: self.gate_index as f64,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            });
            cursor = gate_tick.saturating_add(1);
        }
    }
}

impl PhonationClock for ThetaGateClock {
    fn gather_candidates(&mut self, ctx: &CoreTickCtx, out: &mut Vec<CandidatePoint>) {
        self.gather_candidates_with(ctx, out, |cursor, ctx| {
            next_gate_tick(cursor, ctx.fs, ctx.rhythms.theta, 0.0)
        });
    }
}

#[derive(Debug)]
pub struct SubdivisionClock {
    pub divisions: Vec<u32>,
}

impl SubdivisionClock {
    pub fn new(divisions: Vec<u32>) -> Self {
        Self { divisions }
    }

    fn gather_candidates(&self, grid: &ThetaGrid, out: &mut Vec<CandidatePoint>) {
        if grid.boundaries.len() < 2 {
            return;
        }
        for pair in grid.boundaries.windows(2) {
            let b0 = pair[0];
            let b1 = pair[1];
            if b1.gate != b0.gate.saturating_add(1) {
                continue;
            }
            if b1.tick <= b0.tick.saturating_add(1) {
                continue;
            }
            let gate = b0.gate;
            for &n in &self.divisions {
                if n < 2 {
                    continue;
                }
                for k in 1..n {
                    let phase = k as f32 / n as f32;
                    if phase <= 0.0 || phase >= 1.0 {
                        continue;
                    }
                    let tick = match grid.tick_at(gate, phase) {
                        Some(tick) => tick,
                        None => continue,
                    };
                    if tick == b0.tick || tick == b1.tick {
                        continue;
                    }
                    debug_assert!((gate as f64 + phase as f64).is_finite());
                    out.push(CandidatePoint {
                        tick,
                        gate,
                        theta_pos: gate as f64 + phase as f64,
                        phase_in_gate: phase,
                        sources: vec![ClockSource::Subdivision { n }],
                    });
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct InternalPhaseClock {
    pub ratio: f32,
    pub phase: f32,
}

impl InternalPhaseClock {
    pub fn new(ratio: f32, phase: f32) -> Self {
        Self {
            ratio,
            phase: phase.rem_euclid(1.0),
        }
    }

    fn gather_candidates(&mut self, grid: &ThetaGrid, out: &mut Vec<CandidatePoint>) {
        if !self.ratio.is_finite() || self.ratio <= 0.0 {
            return;
        }
        if grid.boundaries.len() < 2 {
            return;
        }
        for pair in grid.boundaries.windows(2) {
            let b0 = pair[0];
            let b1 = pair[1];
            if b1.gate != b0.gate.saturating_add(1) {
                continue;
            }
            if b1.tick <= b0.tick.saturating_add(1) {
                continue;
            }
            let gate = b0.gate;
            let start_phase = self.phase;
            let end_phase = start_phase + self.ratio;
            let wraps = end_phase.floor() as i32;
            for m in 1..=wraps {
                let phase_in_gate = (m as f32 - start_phase) / self.ratio;
                if phase_in_gate <= 0.0 || phase_in_gate >= 1.0 {
                    continue;
                }
                let tick = match grid.tick_at(gate, phase_in_gate) {
                    Some(tick) => tick,
                    None => continue,
                };
                if tick == b0.tick || tick == b1.tick {
                    continue;
                }
                debug_assert!((gate as f64 + phase_in_gate as f64).is_finite());
                out.push(CandidatePoint {
                    tick,
                    gate,
                    theta_pos: gate as f64 + phase_in_gate as f64,
                    phase_in_gate,
                    sources: vec![ClockSource::InternalPhase],
                });
            }
            self.phase = end_phase.rem_euclid(1.0);
        }
    }
}

#[derive(Debug)]
pub struct CompositeClock {
    gate_clock: ThetaGateClock,
    subdivision: Option<SubdivisionClock>,
    internal_phase: Option<InternalPhaseClock>,
}

impl CompositeClock {
    pub fn new(
        gate_clock: ThetaGateClock,
        subdivision: Option<SubdivisionClock>,
        internal_phase: Option<InternalPhaseClock>,
    ) -> Self {
        Self {
            gate_clock,
            subdivision,
            internal_phase,
        }
    }
}

impl PhonationClock for CompositeClock {
    fn gather_candidates(&mut self, ctx: &CoreTickCtx, out: &mut Vec<CandidatePoint>) {
        let mut gate_candidates = Vec::new();
        self.gate_clock.gather_candidates(ctx, &mut gate_candidates);
        out.extend(gate_candidates.iter().cloned());
        let grid = ThetaGrid::from_candidates(&gate_candidates);
        if let Some(clock) = self.subdivision.as_ref() {
            clock.gather_candidates(&grid, out);
        }
        if let Some(clock) = self.internal_phase.as_mut() {
            clock.gather_candidates(&grid, out);
        }
    }
}

pub trait PhonationInterval: Send {
    fn on_candidate(&mut self, c: &IntervalInput, state: &CoreState) -> Option<PhonationKick>;
    fn update_config(&mut self, _config: &PhonationIntervalConfig) -> bool {
        false
    }
}

pub trait SubThetaMod: Send + Sync {
    fn mod_at_phase(&self, phase_in_gate: f32) -> f32;
    fn update_config(&mut self, _config: &SubThetaModConfig) -> bool {
        false
    }
}

#[derive(Debug, Default)]
pub struct NoneMod;

impl SubThetaMod for NoneMod {
    fn mod_at_phase(&self, _phase_in_gate: f32) -> f32 {
        1.0
    }

    fn update_config(&mut self, config: &SubThetaModConfig) -> bool {
        matches!(config, SubThetaModConfig::None)
    }
}

#[derive(Debug)]
pub struct CosineHarmonicMod {
    pub n: u32,
    pub depth: f32,
    pub phase0: f32,
}

impl SubThetaMod for CosineHarmonicMod {
    fn mod_at_phase(&self, phase_in_gate: f32) -> f32 {
        let depth = self.depth.clamp(0.0, 1.0);
        let phase = phase_in_gate.rem_euclid(1.0);
        1.0 + depth * (TAU * self.n as f32 * phase + self.phase0).cos()
    }

    fn update_config(&mut self, config: &SubThetaModConfig) -> bool {
        match config {
            SubThetaModConfig::Cosine { n, depth, phase0 } => {
                self.n = *n;
                self.depth = *depth;
                self.phase0 = *phase0;
                true
            }
            _ => false,
        }
    }
}

#[derive(Debug, Default)]
pub struct NoneInterval;

impl PhonationInterval for NoneInterval {
    fn on_candidate(&mut self, _c: &IntervalInput, _state: &CoreState) -> Option<PhonationKick> {
        None
    }

    fn update_config(&mut self, config: &PhonationIntervalConfig) -> bool {
        matches!(config, PhonationIntervalConfig::None)
    }
}

#[derive(Debug)]
pub struct AccumulatorInterval {
    pub rate_hz: f32,
    pub refractory_gates: u32,
    pub acc: f32,
    pub next_allowed_gate: u64,
}

impl AccumulatorInterval {
    pub fn new(rate_hz: f32, refractory_gates: u32, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let acc = rng.random_range(0.7..1.0);
        Self {
            rate_hz,
            refractory_gates,
            acc,
            next_allowed_gate: 0,
        }
    }
}

impl PhonationInterval for AccumulatorInterval {
    fn on_candidate(&mut self, c: &IntervalInput, _state: &CoreState) -> Option<PhonationKick> {
        const ACC_MAX: f32 = 32.0;
        if !_state.is_alive {
            return None;
        }
        if !self.rate_hz.is_finite() || self.rate_hz <= 0.0 {
            return None;
        }
        let weight = c.weight.max(0.0);
        self.acc += self.rate_hz * weight * c.dt_sec;
        if !self.acc.is_finite() || self.acc > ACC_MAX {
            self.acc = ACC_MAX;
        }
        let refractory_ok = c.gate >= self.next_allowed_gate;
        if self.acc >= 1.0 && refractory_ok {
            self.acc -= 1.0;
            self.next_allowed_gate = c.gate + self.refractory_gates as u64 + 1;
            return Some(PhonationKick::Planned { strength: 1.0 });
        }
        None
    }

    fn update_config(&mut self, config: &PhonationIntervalConfig) -> bool {
        match config {
            PhonationIntervalConfig::Accumulator { rate, refractory } => {
                self.rate_hz = *rate;
                self.refractory_gates = *refractory;
                true
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConnectNow {
    pub tick: Tick,
    pub gate: u64,
    pub theta_pos: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct ConnectOnset {
    pub note_id: NoteId,
    pub tick: Tick,
    pub gate: u64,
    pub theta_pos: f64,
    pub exc_gate: f32,
    pub exc_slope: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConnectPlan {
    None,
    HoldTheta(f32),
}

pub trait PhonationConnect: Send {
    /// If on_note_on returns ConnectPlan::HoldTheta, the engine schedules the NoteOff via
    /// its pending-off queue, and implementations must not emit a NoteOff for that note_id in poll.
    fn on_note_on(&mut self, onset: ConnectOnset) -> ConnectPlan;
    fn poll(&mut self, now: ConnectNow, out: &mut Vec<PhonationCmd>);
    fn update_config(&mut self, _config: &PhonationConnectConfig) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct FixedGateConnect {
    pub length_gates: u32,
    pending: Vec<(NoteId, u64)>,
}

impl FixedGateConnect {
    pub fn new(length_gates: u32) -> Self {
        Self {
            length_gates,
            pending: Vec::new(),
        }
    }
}

impl PhonationConnect for FixedGateConnect {
    fn on_note_on(&mut self, onset: ConnectOnset) -> ConnectPlan {
        let off_gate = onset.gate + self.length_gates as u64;
        self.pending.push((onset.note_id, off_gate));
        ConnectPlan::None
    }

    fn poll(&mut self, now: ConnectNow, out: &mut Vec<PhonationCmd>) {
        let mut idx = 0;
        while idx < self.pending.len() {
            if self.pending[idx].1 <= now.gate {
                let note_id = self.pending[idx].0;
                self.pending.swap_remove(idx);
                out.push(PhonationCmd::NoteOff {
                    note_id,
                    off_tick: now.tick,
                });
            } else {
                idx += 1;
            }
        }
    }

    fn update_config(&mut self, config: &PhonationConnectConfig) -> bool {
        match config {
            PhonationConnectConfig::FixedGate { length_gates } => {
                self.length_gates = *length_gates;
                true
            }
            _ => false,
        }
    }
}

#[derive(Debug)]
pub struct FieldConnect {
    pub hold_min_theta: f32,
    pub hold_max_theta: f32,
    pub curve_k: f32,
    pub curve_x0: f32,
    pub drop_gain: f32,
}

impl FieldConnect {
    pub fn new(
        hold_min_theta: f32,
        hold_max_theta: f32,
        curve_k: f32,
        curve_x0: f32,
        drop_gain: f32,
    ) -> Self {
        let hold_min_theta = if hold_min_theta.is_finite() {
            hold_min_theta.max(0.0)
        } else {
            0.25
        };
        let mut hold_max_theta = if hold_max_theta.is_finite() {
            hold_max_theta.max(0.0)
        } else {
            1.0
        };
        if hold_max_theta < hold_min_theta {
            hold_max_theta = hold_min_theta;
        }
        let curve_k = if curve_k.is_finite() { curve_k } else { 4.0 };
        let curve_x0 = if curve_x0.is_finite() { curve_x0 } else { 0.5 };
        let drop_gain = if drop_gain.is_finite() {
            drop_gain
        } else {
            0.0
        };
        Self {
            hold_min_theta,
            hold_max_theta,
            curve_k,
            curve_x0,
            drop_gain,
        }
    }

    fn hold_theta(&self, exc_gate: f32, exc_slope: f32) -> f32 {
        let min = self.hold_min_theta.max(0.0);
        let max = self.hold_max_theta.max(min);
        let p = 1.0 / (1.0 + (-(self.curve_k * (exc_gate - self.curve_x0))).exp());
        let mut hold = min + (max - min) * p;
        if self.drop_gain > 0.0 && exc_slope.is_finite() {
            let drop = (-exc_slope).max(0.0).clamp(0.0, 1.0);
            hold *= 1.0 - self.drop_gain.clamp(0.0, 1.0) * drop;
        }
        hold.max(0.0)
    }
}

impl PhonationConnect for FieldConnect {
    fn on_note_on(&mut self, onset: ConnectOnset) -> ConnectPlan {
        let exc_gate = onset.exc_gate.clamp(0.0, 1.0);
        let hold = self.hold_theta(exc_gate, onset.exc_slope);
        ConnectPlan::HoldTheta(hold)
    }

    fn poll(&mut self, now: ConnectNow, out: &mut Vec<PhonationCmd>) {
        let _ = now;
        let _ = out;
    }

    fn update_config(&mut self, config: &PhonationConnectConfig) -> bool {
        match config {
            PhonationConnectConfig::Field {
                hold_min_theta,
                hold_max_theta,
                curve_k,
                curve_x0,
                drop_gain,
            } => {
                self.hold_min_theta = *hold_min_theta;
                self.hold_max_theta = *hold_max_theta;
                self.curve_k = *curve_k;
                self.curve_x0 = *curve_x0;
                self.drop_gain = *drop_gain;
                true
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PhonationNoteOn {
    pub note_id: NoteId,
    pub onset_tick: Tick,
}

#[derive(Debug, Clone, Copy)]
pub struct OnsetEvent {
    pub gate: u64,
    pub onset_tick: Tick,
    pub strength: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingOff {
    off_tick: Tick,
    note_id: NoteId,
}

impl Ord for PendingOff {
    fn cmp(&self, other: &Self) -> Ordering {
        self.off_tick
            .cmp(&other.off_tick)
            .then_with(|| self.note_id.cmp(&other.note_id))
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
    note_id: Option<NoteId>,
}

pub struct PhonationEngine {
    pub clock: Box<dyn PhonationClock + Send>,
    pub interval: Box<dyn PhonationInterval + Send>,
    pub connect: Box<dyn PhonationConnect + Send>,
    pub sub_theta_mod: Box<dyn SubThetaMod + Send + Sync>,
    pub mode: PhonationMode,
    hold: HoldCore,
    initial_seed: u64,
    pub next_note_id: NoteId,
    last_gate_index: Option<u64>,
    last_theta_pos: Option<f64>,
    last_tick: Option<Tick>,
    active_notes: u32,
    pending_off: BinaryHeap<Reverse<PendingOff>>,
    scratch_candidates: Vec<CandidatePoint>,
    scratch_merged: Vec<CandidatePoint>,
}

impl fmt::Debug for PhonationEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PhonationEngine")
            .field("next_note_id", &self.next_note_id)
            .field("last_gate_index", &self.last_gate_index)
            .field("active_notes", &self.active_notes)
            .field("mode", &self.mode)
            .finish()
    }
}

impl PhonationEngine {
    fn make_interval_from_config(
        config: &PhonationIntervalConfig,
        seed: u64,
    ) -> Box<dyn PhonationInterval + Send> {
        match config {
            PhonationIntervalConfig::None => Box::<NoneInterval>::default(),
            PhonationIntervalConfig::Accumulator { rate, refractory } => {
                Box::new(AccumulatorInterval::new(*rate, *refractory, seed))
            }
        }
    }

    fn make_connect_from_config(
        config: &PhonationConnectConfig,
    ) -> Box<dyn PhonationConnect + Send> {
        match config {
            PhonationConnectConfig::FixedGate { length_gates } => {
                Box::new(FixedGateConnect::new(*length_gates))
            }
            PhonationConnectConfig::Field {
                hold_min_theta,
                hold_max_theta,
                curve_k,
                curve_x0,
                drop_gain,
            } => Box::new(FieldConnect::new(
                *hold_min_theta,
                *hold_max_theta,
                *curve_k,
                *curve_x0,
                *drop_gain,
            )),
        }
    }

    pub fn from_config(config: &PhonationConfig, seed: u64) -> Self {
        let interval = Self::make_interval_from_config(&config.interval, seed);
        let sub_theta_mod: Box<dyn SubThetaMod + Send + Sync> = match &config.sub_theta_mod {
            SubThetaModConfig::None => Box::<NoneMod>::default(),
            SubThetaModConfig::Cosine { n, depth, phase0 } => Box::new(CosineHarmonicMod {
                n: *n,
                depth: *depth,
                phase0: *phase0,
            }),
        };
        let connect = Self::make_connect_from_config(&config.connect);
        let (subdivision, internal_phase) = match &config.clock {
            PhonationClockConfig::ThetaGate => (None, None),
            PhonationClockConfig::Composite {
                subdivision,
                internal_phase,
            } => {
                let subdivision = subdivision
                    .as_ref()
                    .map(|config| SubdivisionClock::new(config.divisions.clone()));
                let internal_phase = internal_phase
                    .as_ref()
                    .map(|config| InternalPhaseClock::new(config.ratio, config.phase0));
                (subdivision, internal_phase)
            }
        };
        Self {
            clock: match &config.clock {
                PhonationClockConfig::ThetaGate => Box::<ThetaGateClock>::default(),
                PhonationClockConfig::Composite { .. } => Box::new(CompositeClock::new(
                    ThetaGateClock::default(),
                    subdivision,
                    internal_phase,
                )),
            },
            interval,
            connect,
            sub_theta_mod,
            mode: config.mode,
            hold: HoldCore::default(),
            initial_seed: seed,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        }
    }

    pub fn update_from_config(&mut self, config: &PhonationConfig) {
        self.mode = config.mode;
        if !self.interval.update_config(&config.interval) {
            warn!("phonation interval config mismatch; rebuilding interval to match config");
            let seed = self.initial_seed ^ 0xA5A5_5A5A_5A5A_A5A5;
            self.interval = Self::make_interval_from_config(&config.interval, seed);
        }
        if !self.connect.update_config(&config.connect) {
            warn!("phonation connect config mismatch; rebuilding connect to match config");
            self.connect = Self::make_connect_from_config(&config.connect);
        }
        if !self.sub_theta_mod.update_config(&config.sub_theta_mod) {
            debug!("phonation sub-theta config mismatch; rebuilding modulation to match config");
            self.sub_theta_mod = match &config.sub_theta_mod {
                SubThetaModConfig::None => Box::<NoneMod>::default(),
                SubThetaModConfig::Cosine { n, depth, phase0 } => Box::new(CosineHarmonicMod {
                    n: *n,
                    depth: *depth,
                    phase0: *phase0,
                }),
            };
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
        out_cmds: &mut Vec<PhonationCmd>,
        out_events: &mut Vec<PhonationNoteOn>,
        out_onsets: &mut Vec<OnsetEvent>,
    ) {
        let allow_onset = min_allowed_onset_tick
            .map(|min_tick| ctx.now_tick >= min_tick)
            .unwrap_or(true);
        if allow_onset && state.is_alive && !self.hold.note_on_sent {
            let note_id = self.next_note_id;
            self.next_note_id = self.next_note_id.wrapping_add(1);
            self.hold.note_on_sent = true;
            self.hold.note_id = Some(note_id);
            out_cmds.push(PhonationCmd::NoteOn {
                note_id,
                kick: PhonationKick::Planned { strength: 1.0 },
            });
            self.active_notes = self.active_notes.saturating_add(1);
            out_events.push(PhonationNoteOn {
                note_id,
                onset_tick: ctx.now_tick,
            });
            out_onsets.push(OnsetEvent {
                gate: 0,
                onset_tick: ctx.now_tick,
                strength: 1.0,
            });
        }
        if !state.is_alive
            && let Some(note_id) = self.hold.note_id.take()
        {
            out_cmds.push(PhonationCmd::NoteOff {
                note_id,
                off_tick: ctx.now_tick,
            });
            self.active_notes = self.active_notes.saturating_sub(1);
        }
    }

    fn schedule_note_off(&mut self, note_id: NoteId, off_tick: Tick) {
        self.pending_off
            .push(Reverse(PendingOff { off_tick, note_id }));
    }

    fn drain_note_offs(&mut self, up_to_tick: Tick, out_cmds: &mut Vec<PhonationCmd>) {
        while let Some(Reverse(next)) = self.pending_off.peek().copied() {
            if next.off_tick > up_to_tick {
                break;
            }
            self.pending_off.pop();
            out_cmds.push(PhonationCmd::NoteOff {
                note_id: next.note_id,
                off_tick: next.off_tick,
            });
            self.active_notes = self.active_notes.saturating_sub(1);
        }
    }

    fn schedule_hold_theta(
        &mut self,
        onset: ConnectOnset,
        hold_theta: f32,
        ctx: &CoreTickCtx,
        timing_grid: &mut ThetaGrid,
    ) {
        let hold_theta = if hold_theta.is_finite() {
            hold_theta.max(0.0)
        } else {
            0.0
        };
        let off_theta = onset.theta_pos + hold_theta as f64;
        let off_tick_opt = Self::tick_for_theta(off_theta, ctx, timing_grid);
        let fallback = timing_grid
            .boundaries
            .last()
            .map(|boundary| boundary.tick)
            .unwrap_or(onset.tick);
        let off_tick = off_tick_opt.unwrap_or(fallback).max(onset.tick);
        self.schedule_note_off(onset.note_id, off_tick);
    }

    fn tick_for_theta(
        theta_pos: f64,
        ctx: &CoreTickCtx,
        timing_grid: &mut ThetaGrid,
    ) -> Option<Tick> {
        if !theta_pos.is_finite() || theta_pos < 0.0 {
            return None;
        }
        let off_gate_f = theta_pos.floor();
        if off_gate_f >= u64::MAX as f64 {
            return None;
        }
        let mut off_gate = off_gate_f as u64;
        let mut off_phase = (theta_pos - off_gate_f).clamp(0.0, 1.0);
        if off_phase >= 1.0 {
            off_phase = 0.0;
            off_gate = off_gate.saturating_add(1);
        }
        timing_grid.ensure_boundaries_until(ctx, off_gate.saturating_add(1));
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
        out_cmds: &mut Vec<PhonationCmd>,
        out_events: &mut Vec<PhonationNoteOn>,
        out_onsets: &mut Vec<OnsetEvent>,
    ) {
        if matches!(self.mode, PhonationMode::Hold) {
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
        self.scratch_candidates.clear();
        self.clock
            .gather_candidates(ctx, &mut self.scratch_candidates);
        Self::merge_candidates_into(&mut self.scratch_merged, &mut self.scratch_candidates);
        let merged = std::mem::take(&mut self.scratch_merged);
        let mut timing_grid = ThetaGrid::from_candidates(&merged);
        let timing_field = TimingField::build_from(
            ctx,
            &timing_grid,
            social.map(|trace| (trace, social_coupling)),
            extra_gate_gain,
        );
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
        out_cmds: &mut Vec<PhonationCmd>,
        out_events: &mut Vec<PhonationNoteOn>,
        out_onsets: &mut Vec<OnsetEvent>,
    ) {
        let mut prev_gate_exc: Option<f32> = None;
        self.drain_note_offs(ctx.now_tick, out_cmds);
        for c in candidates {
            let allow_onset = min_allowed_onset_tick
                .map(|min_tick| c.tick >= min_tick)
                .unwrap_or(true);
            debug_assert!(c.theta_pos.is_finite());
            self.drain_note_offs(c.tick, out_cmds);
            // dt_theta spec: same tick -> 0; negative/non-finite delta -> 0 (debug assert);
            // NaN/Inf after cast -> 0.
            let dt_theta = if self.last_tick == Some(c.tick) {
                0.0
            } else {
                match self.last_theta_pos {
                    Some(prev_pos) => {
                        let dt = c.theta_pos - prev_pos;
                        if !dt.is_finite() {
                            debug_assert!(false, "candidate theta_pos delta must be finite");
                            0.0
                        } else if dt < 0.0 {
                            debug_assert!(false, "candidate theta_pos must be non-decreasing");
                            0.0
                        } else {
                            let dt_f32 = dt as f32;
                            if dt_f32.is_finite() { dt_f32 } else { 0.0 }
                        }
                    }
                    None => 1.0,
                }
            };
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
                            let dt = dt_ticks as f32 / ctx.fs;
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
            let connect_now = ConnectNow {
                tick: c.tick,
                gate: c.gate,
                theta_pos: c.theta_pos,
            };
            let before = out_cmds.len();
            self.connect.poll(connect_now, out_cmds);
            for cmd in &out_cmds[before..] {
                if matches!(cmd, PhonationCmd::NoteOff { .. }) {
                    self.active_notes = self.active_notes.saturating_sub(1);
                }
            }
            let sub_theta_mod = if c.phase_in_gate == 0.0 {
                1.0
            } else {
                self.sub_theta_mod.mod_at_phase(c.phase_in_gate)
            };
            let input = IntervalInput {
                gate: c.gate,
                tick: c.tick,
                dt_theta,
                dt_sec,
                weight: exc_gate * sub_theta_mod,
            };
            if allow_onset && let Some(kick) = self.interval.on_candidate(&input, state) {
                let note_id = self.next_note_id;
                self.next_note_id = self.next_note_id.wrapping_add(1);
                out_cmds.push(PhonationCmd::NoteOn { note_id, kick });
                self.active_notes = self.active_notes.saturating_add(1);
                out_events.push(PhonationNoteOn {
                    note_id,
                    onset_tick: c.tick,
                });
                out_onsets.push(OnsetEvent {
                    gate: c.gate,
                    onset_tick: c.tick,
                    strength: kick.strength(),
                });
                let onset = ConnectOnset {
                    note_id,
                    tick: c.tick,
                    gate: c.gate,
                    theta_pos: c.theta_pos,
                    exc_gate,
                    exc_slope,
                };
                let plan = self.connect.on_note_on(onset);
                if let ConnectPlan::HoldTheta(hold_theta) = plan {
                    self.schedule_hold_theta(onset, hold_theta, ctx, timing_grid);
                }
            }
            self.last_gate_index = Some(c.gate);
            let before = out_cmds.len();
            self.connect.poll(connect_now, out_cmds);
            for cmd in &out_cmds[before..] {
                if matches!(cmd, PhonationCmd::NoteOff { .. }) {
                    self.active_notes = self.active_notes.saturating_sub(1);
                }
            }
            self.last_theta_pos = Some(c.theta_pos);
            self.last_tick = Some(c.tick);
            if c.phase_in_gate == 0.0 {
                prev_gate_exc = Some(exc_gate);
            }
        }
        self.drain_note_offs(ctx.frame_end.saturating_sub(1), out_cmds);
    }

    /// Merge rules:
    /// - sort by (tick, gate, phase_in_gate)
    /// - merge only when tick and gate match
    /// - prefer phase_in_gate == 0.0 (GateBoundary) as the representative
    /// - append sources (duplicates allowed)
    #[cfg(test)]
    fn merge_candidates(mut candidates: Vec<CandidatePoint>) -> Vec<CandidatePoint> {
        let mut merged = Vec::with_capacity(candidates.len());
        Self::merge_candidates_into(&mut merged, &mut candidates);
        merged
    }

    fn merge_candidates_into(out: &mut Vec<CandidatePoint>, candidates: &mut Vec<CandidatePoint>) {
        candidates.sort_by(|a, b| {
            a.tick
                .cmp(&b.tick)
                .then_with(|| a.gate.cmp(&b.gate))
                .then_with(|| a.phase_in_gate.total_cmp(&b.phase_in_gate))
        });
        out.clear();
        out.reserve(candidates.len());
        for candidate in candidates.drain(..) {
            if let Some(last) = out.last_mut()
                && last.tick == candidate.tick
                && last.gate == candidate.gate
            {
                if last.phase_in_gate != 0.0 && candidate.phase_in_gate == 0.0 {
                    let mut replacement = candidate;
                    replacement.sources.append(&mut last.sources);
                    *last = replacement;
                    continue;
                }
                last.sources.extend(candidate.sources);
                continue;
            }
            out.push(candidate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::scenario::SocialConfig;
    use std::collections::BinaryHeap;
    use std::sync::{Arc, Mutex};

    fn candidate_at_gate(gate: u64) -> CandidatePoint {
        CandidatePoint {
            tick: gate as u64,
            gate,
            theta_pos: gate as f64,
            phase_in_gate: 0.0,
            sources: vec![ClockSource::GateBoundary],
        }
    }

    #[test]
    fn accumulator_rate_zero_never_fires() {
        let mut interval = AccumulatorInterval::new(0.0, 0, 1);
        interval.acc = 0.0;
        let state = CoreState { is_alive: true };
        for gate in 0..10u64 {
            let c = candidate_at_gate(gate);
            let input = IntervalInput {
                gate: c.gate,
                tick: c.tick,
                dt_theta: 1.0,
                dt_sec: 1.0,
                weight: 1.0,
            };
            assert!(interval.on_candidate(&input, &state).is_none());
        }
    }

    #[test]
    fn accumulator_rate_half_fires_every_other_gate() {
        let mut interval = AccumulatorInterval::new(0.5, 0, 1);
        interval.acc = 0.0;
        let state = CoreState { is_alive: true };
        let mut fired = Vec::new();
        for gate in 0..10u64 {
            let c = candidate_at_gate(gate);
            let input = IntervalInput {
                gate: c.gate,
                tick: c.tick,
                dt_theta: 1.0,
                dt_sec: 1.0,
                weight: 1.0,
            };
            if interval.on_candidate(&input, &state).is_some() {
                fired.push(gate);
            }
        }
        assert_eq!(fired, vec![1u64, 3, 5, 7, 9]);
    }

    #[test]
    fn accumulator_refractory_blocks_adjacent_gates() {
        let mut interval = AccumulatorInterval::new(1.0, 1, 1);
        interval.acc = 0.0;
        let state = CoreState { is_alive: true };
        let mut fired = Vec::new();
        for gate in 0..6u64 {
            let c = candidate_at_gate(gate);
            let input = IntervalInput {
                gate: c.gate,
                tick: c.tick,
                dt_theta: 1.0,
                dt_sec: 1.0,
                weight: 1.0,
            };
            if interval.on_candidate(&input, &state).is_some() {
                fired.push(gate);
            }
        }
        assert_eq!(fired, vec![0u64, 2, 4]);
    }

    #[test]
    fn accumulator_weighted_steps_fire_after_sum() {
        let mut interval = AccumulatorInterval::new(1.0, 0, 1);
        interval.acc = 0.0;
        let state = CoreState { is_alive: true };
        let mut fired = Vec::new();
        let inputs = [
            IntervalInput {
                gate: 0,
                tick: 0,
                dt_theta: 1.0,
                dt_sec: 1.0,
                weight: 0.5,
            },
            IntervalInput {
                gate: 1,
                tick: 1,
                dt_theta: 1.0,
                dt_sec: 1.0,
                weight: 0.5,
            },
        ];
        for input in inputs {
            if interval.on_candidate(&input, &state).is_some() {
                fired.push(input.gate);
            }
        }
        assert_eq!(fired, vec![1u64]);
    }

    #[test]
    fn accumulator_ignores_zero_weight() {
        let mut interval = AccumulatorInterval::new(1.0, 0, 1);
        interval.acc = 0.0;
        let state = CoreState { is_alive: true };
        let input = IntervalInput {
            gate: 0,
            tick: 0,
            dt_theta: 1.0,
            dt_sec: 1.0,
            weight: 0.0,
        };
        assert!(interval.on_candidate(&input, &state).is_none());
    }

    #[test]
    fn accumulator_respects_dt_sec() {
        let mut interval = AccumulatorInterval::new(1.0, 0, 1);
        interval.acc = 0.0;
        let state = CoreState { is_alive: true };
        let input = IntervalInput {
            gate: 0,
            tick: 0,
            dt_theta: 0.5,
            dt_sec: 0.5,
            weight: 1.0,
        };
        assert!(interval.on_candidate(&input, &state).is_none());
        let input = IntervalInput {
            gate: 1,
            tick: 1,
            dt_theta: 0.5,
            dt_sec: 0.5,
            weight: 1.0,
        };
        assert!(interval.on_candidate(&input, &state).is_some());
    }

    #[test]
    fn accumulator_acc_is_clamped() {
        let mut interval = AccumulatorInterval::new(1.0, 0, 1);
        interval.acc = 0.0;
        let state = CoreState { is_alive: true };
        let input = IntervalInput {
            gate: 0,
            tick: 0,
            dt_theta: 1.0,
            dt_sec: 1.0,
            weight: 1.0e9,
        };
        let _ = interval.on_candidate(&input, &state);
        assert!(interval.acc.is_finite());
        assert!(interval.acc <= 32.0);
    }

    #[test]
    fn fixed_gate_connect_sets_off_tick() {
        let mut connect = FixedGateConnect::new(0);
        let mut out = Vec::new();
        let plan = connect.on_note_on(ConnectOnset {
            note_id: 1,
            tick: 123,
            gate: 10,
            theta_pos: 10.0,
            exc_gate: 1.0,
            exc_slope: 0.0,
        });
        assert_eq!(plan, ConnectPlan::None);
        connect.poll(
            ConnectNow {
                tick: 123,
                gate: 10,
                theta_pos: 10.0,
            },
            &mut out,
        );
        assert_eq!(
            out,
            vec![PhonationCmd::NoteOff {
                note_id: 1,
                off_tick: 123,
            }]
        );
    }

    #[test]
    fn theta_gate_clock_emits_multiple_gates_in_hop() {
        let mut clock = ThetaGateClock::default();
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 10,
            fs: 1000.0,
            rhythms: NeuralRhythms::default(),
        };
        let mut out = Vec::new();
        let mut calls = 0;
        clock.gather_candidates_with(&ctx, &mut out, |_cursor, _ctx| {
            let tick = match calls {
                0 => Some(0),
                1 => Some(4),
                2 => Some(8),
                _ => None,
            };
            calls += 1;
            tick
        });
        assert!(out.len() >= 2, "expected multiple gates in hop");
        assert!(out[0].tick < out[1].tick);
    }

    #[test]
    fn theta_gate_clock_emits_candidate_at_frame_start_gate() {
        let mut clock = ThetaGateClock::default();
        let rhythms = NeuralRhythms::default();
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 4,
            fs: 1000.0,
            rhythms,
        };
        let mut out = Vec::new();
        clock.gather_candidates_with(&ctx, &mut out, |cursor, _ctx| {
            if cursor == 0 { Some(cursor) } else { None }
        });
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].tick, 0);
    }

    #[test]
    fn phonation_engine_uses_timing_field_weight() {
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 8,
            fs: 1000.0,
            rhythms: NeuralRhythms::default(),
        };
        let candidates = vec![
            CandidatePoint {
                tick: 0,
                gate: 0,
                theta_pos: 0.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 4,
                gate: 1,
                theta_pos: 1.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
        ];
        let timing_field = TimingField::from_values(0, vec![0.0, 1.0]);
        let mut interval = AccumulatorInterval::new(250.0, 0, 1);
        interval.acc = 0.0;
        let mut engine = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::new(interval),
            connect: Box::new(FixedGateConnect::new(1)),
            sub_theta_mod: Box::<NoneMod>::default(),
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        };
        let state = CoreState { is_alive: true };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&candidates);
        let _ = engine.process_candidates(
            &ctx,
            &candidates,
            &mut timing_grid,
            &timing_field,
            &state,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].onset_tick, 4);
    }

    #[test]
    fn timing_field_indexing_from_values() {
        let timing_field = TimingField::from_values(10, vec![0.2, 0.8]);
        assert_eq!(timing_field.e(10), 0.2);
        assert_eq!(timing_field.e(11), 0.8);
        assert_eq!(timing_field.e(12), 1.0);
    }

    #[test]
    fn timing_field_build_from_uses_env_gate() {
        let mut rhythms = NeuralRhythms::default();
        rhythms.env_open = 0.0;
        rhythms.env_level = 1.0;
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 4,
            fs: 1000.0,
            rhythms,
        };
        let grid = ThetaGrid {
            boundaries: vec![GateBoundary { gate: 0, tick: 0 }],
        };
        let timing_field = TimingField::build_from(&ctx, &grid, None, 1.0);
        assert_eq!(timing_field.e(0), 0.0);
    }

    #[test]
    fn theta_grid_skips_duplicate_gates() {
        let candidates = vec![
            CandidatePoint {
                tick: 0,
                gate: 0,
                theta_pos: 0.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 1,
                gate: 0,
                theta_pos: 0.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 2,
                gate: 1,
                theta_pos: 1.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
        ];
        let grid = ThetaGrid::from_candidates(&candidates);
        assert_eq!(
            grid.boundaries,
            vec![
                GateBoundary { gate: 0, tick: 0 },
                GateBoundary { gate: 1, tick: 2 },
            ]
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn theta_grid_panics_on_reverse_gates() {
        let candidates = vec![
            CandidatePoint {
                tick: 0,
                gate: 1,
                theta_pos: 1.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 1,
                gate: 0,
                theta_pos: 0.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
        ];
        let _ = ThetaGrid::from_candidates(&candidates);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn theta_grid_panics_on_unsorted_ticks() {
        let candidates = vec![
            CandidatePoint {
                tick: 2,
                gate: 0,
                theta_pos: 0.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 1,
                gate: 1,
                theta_pos: 1.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
        ];
        let _ = ThetaGrid::from_candidates(&candidates);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn theta_grid_panics_on_noncontiguous_gates() {
        let candidates = vec![
            CandidatePoint {
                tick: 0,
                gate: 0,
                theta_pos: 0.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 10,
                gate: 2,
                theta_pos: 2.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
        ];
        let _ = ThetaGrid::from_candidates(&candidates);
    }

    #[test]
    fn timing_field_fill_missing_gates() {
        let mut rhythms = NeuralRhythms::default();
        rhythms.env_open = 0.5;
        rhythms.env_level = 1.0;
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 1,
            fs: 1000.0,
            rhythms,
        };
        let grid = ThetaGrid {
            boundaries: vec![
                GateBoundary { gate: 0, tick: 0 },
                GateBoundary { gate: 2, tick: 0 },
            ],
        };
        let timing_field = TimingField::build_from(&ctx, &grid, None, 1.0);
        assert_eq!(timing_field.e(0), 0.5);
        assert_eq!(timing_field.e(1), 0.5);
        assert_eq!(timing_field.e(2), 0.5);
    }

    #[test]
    fn theta_grid_tick_at_interpolates() {
        let grid = ThetaGrid {
            boundaries: vec![
                GateBoundary { gate: 0, tick: 0 },
                GateBoundary { gate: 1, tick: 100 },
            ],
        };
        assert_eq!(grid.tick_at(0, 0.5), Some(50));
    }

    #[test]
    fn theta_grid_tick_at_avoids_boundaries() {
        let grid = ThetaGrid {
            boundaries: vec![
                GateBoundary { gate: 0, tick: 0 },
                GateBoundary { gate: 1, tick: 10 },
            ],
        };
        assert_ne!(grid.tick_at(0, 0.95), Some(10));
    }

    #[test]
    fn theta_grid_tick_at_returns_none_when_interval_too_small() {
        let grid = ThetaGrid {
            boundaries: vec![
                GateBoundary { gate: 0, tick: 0 },
                GateBoundary { gate: 1, tick: 1 },
            ],
        };
        assert_eq!(grid.tick_at(0, 0.5), None);
    }

    #[test]
    fn tick_at_returns_none_on_too_short_interval_for_interior_phase() {
        let grid = ThetaGrid {
            boundaries: vec![
                GateBoundary { gate: 0, tick: 0 },
                GateBoundary { gate: 1, tick: 1 },
            ],
        };
        assert_eq!(grid.tick_at(0, 0.5), None);
    }

    #[test]
    fn tick_at_returns_boundary_ticks_for_phase_0_and_1() {
        let grid = ThetaGrid {
            boundaries: vec![
                GateBoundary { gate: 0, tick: 0 },
                GateBoundary { gate: 1, tick: 100 },
            ],
        };
        assert_eq!(grid.tick_at(0, 0.0), Some(0));
        assert_eq!(grid.tick_at(0, 1.0), Some(100));
    }

    #[test]
    fn theta_grid_ensure_boundaries_until_uses_next_gate_tick_after_boundary() {
        let mut rhythms = NeuralRhythms::default();
        rhythms.theta.freq_hz = 1.0;
        rhythms.theta.phase = 0.0;
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 1000,
            fs: 100.0,
            rhythms,
        };
        let t0 = 0;
        let expected =
            next_gate_tick(t0 + 1, ctx.fs, ctx.rhythms.theta, 0.0).expect("expected tick");
        let mut grid = ThetaGrid {
            boundaries: vec![GateBoundary { gate: 0, tick: t0 }],
        };
        grid.ensure_boundaries_until(&ctx, 1);
        assert_eq!(grid.boundaries.len(), 2);
        assert_eq!(grid.boundaries[1].tick, expected);
    }

    #[test]
    fn theta_grid_ensure_boundaries_until_is_monotonic() {
        let mut rhythms = NeuralRhythms::default();
        rhythms.theta.freq_hz = 1.0;
        rhythms.theta.phase = 0.0;
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 1000,
            fs: 100.0,
            rhythms,
        };
        let mut grid = ThetaGrid {
            boundaries: vec![GateBoundary { gate: 0, tick: 0 }],
        };
        grid.ensure_boundaries_until(&ctx, 4);
        assert!(grid.boundaries.len() >= 5);
        for pair in grid.boundaries.windows(2) {
            assert_eq!(pair[1].gate, pair[0].gate + 1);
            assert!(pair[1].tick > pair[0].tick);
        }
    }

    #[test]
    fn subdivision_clock_skips_short_interval() {
        let grid = ThetaGrid {
            boundaries: vec![
                GateBoundary { gate: 0, tick: 0 },
                GateBoundary { gate: 1, tick: 1 },
            ],
        };
        let mut candidates = Vec::new();
        let clock = SubdivisionClock::new(vec![2]);
        clock.gather_candidates(&grid, &mut candidates);
        assert!(candidates.is_empty());
    }

    #[test]
    fn subdivision_clock_merges_duplicate_ticks() {
        let grid = ThetaGrid {
            boundaries: vec![
                GateBoundary { gate: 0, tick: 0 },
                GateBoundary { gate: 1, tick: 100 },
            ],
        };
        let mut candidates = Vec::new();
        let clock = SubdivisionClock::new(vec![2, 4]);
        clock.gather_candidates(&grid, &mut candidates);
        let merged = PhonationEngine::merge_candidates(candidates);
        let tick_50 = merged.iter().find(|c| c.tick == 50).expect("tick 50");
        assert!(tick_50.sources.contains(&ClockSource::Subdivision { n: 2 }));
        assert!(tick_50.sources.contains(&ClockSource::Subdivision { n: 4 }));
    }

    #[test]
    fn dt_theta_is_continuous() {
        let log = Arc::new(Mutex::new(Vec::new()));
        struct RecordingInterval {
            log: Arc<Mutex<Vec<f32>>>,
        }

        impl PhonationInterval for RecordingInterval {
            fn on_candidate(
                &mut self,
                c: &IntervalInput,
                _state: &CoreState,
            ) -> Option<PhonationKick> {
                self.log.lock().expect("dt log").push(c.dt_theta);
                None
            }
        }

        let candidates = vec![
            CandidatePoint {
                tick: 0,
                gate: 0,
                theta_pos: 0.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 10,
                gate: 0,
                theta_pos: 0.5,
                phase_in_gate: 0.5,
                sources: vec![ClockSource::Subdivision { n: 2 }],
            },
            CandidatePoint {
                tick: 20,
                gate: 1,
                theta_pos: 1.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
        ];
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 30,
            fs: 1000.0,
            rhythms: NeuralRhythms::default(),
        };
        let timing_field = TimingField::from_values(0, vec![1.0, 1.0]);
        let mut engine = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::new(RecordingInterval { log: log.clone() }),
            connect: Box::new(FixedGateConnect::new(1)),
            sub_theta_mod: Box::<NoneMod>::default(),
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        };
        let state = CoreState { is_alive: true };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&candidates);
        let _ = engine.process_candidates(
            &ctx,
            &candidates,
            &mut timing_grid,
            &timing_field,
            &state,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        assert_eq!(*log.lock().expect("dt log"), vec![1.0, 0.5, 0.5]);
    }

    #[test]
    fn dt_theta_zero_when_same_tick() {
        let log = Arc::new(Mutex::new(Vec::new()));
        struct RecordingInterval {
            log: Arc<Mutex<Vec<f32>>>,
        }

        impl PhonationInterval for RecordingInterval {
            fn on_candidate(
                &mut self,
                c: &IntervalInput,
                _state: &CoreState,
            ) -> Option<PhonationKick> {
                self.log.lock().expect("dt log").push(c.dt_theta);
                None
            }
        }

        let candidates = vec![
            CandidatePoint {
                tick: 0,
                gate: 0,
                theta_pos: 0.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 0,
                gate: 1,
                theta_pos: 1.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
        ];
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 1,
            fs: 1000.0,
            rhythms: NeuralRhythms::default(),
        };
        let timing_field = TimingField::from_values(0, vec![1.0, 1.0]);
        let mut engine = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::new(RecordingInterval { log: log.clone() }),
            connect: Box::new(FixedGateConnect::new(1)),
            sub_theta_mod: Box::<NoneMod>::default(),
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        };
        let state = CoreState { is_alive: true };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&candidates);
        let _ = engine.process_candidates(
            &ctx,
            &candidates,
            &mut timing_grid,
            &timing_field,
            &state,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        assert_eq!(*log.lock().expect("dt log"), vec![1.0, 0.0]);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn dt_theta_panics_on_negative_theta() {
        let candidates = vec![CandidatePoint {
            tick: 1,
            gate: 0,
            theta_pos: 0.0,
            phase_in_gate: 0.0,
            sources: vec![ClockSource::GateBoundary],
        }];
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 2,
            fs: 1000.0,
            rhythms: NeuralRhythms::default(),
        };
        let timing_field = TimingField::from_values(0, vec![1.0]);
        let mut engine = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::<NoneInterval>::default(),
            connect: Box::new(FixedGateConnect::new(1)),
            sub_theta_mod: Box::<NoneMod>::default(),
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: Some(1.0),
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        };
        let state = CoreState { is_alive: true };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&candidates);
        let _ = engine.process_candidates(
            &ctx,
            &candidates,
            &mut timing_grid,
            &timing_field,
            &state,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
    }

    #[test]
    fn dt_theta_large_gate_precision() {
        let log = Arc::new(Mutex::new(Vec::new()));
        struct RecordingInterval {
            log: Arc<Mutex<Vec<f32>>>,
        }

        impl PhonationInterval for RecordingInterval {
            fn on_candidate(
                &mut self,
                c: &IntervalInput,
                _state: &CoreState,
            ) -> Option<PhonationKick> {
                self.log.lock().expect("dt log").push(c.dt_theta);
                None
            }
        }

        let candidates = vec![
            CandidatePoint {
                tick: 0,
                gate: 20_000_000,
                theta_pos: 20_000_000.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 1,
                gate: 20_000_001,
                theta_pos: 20_000_001.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
        ];
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 2,
            fs: 1000.0,
            rhythms: NeuralRhythms::default(),
        };
        let timing_field = TimingField::from_values(20_000_000, vec![1.0, 1.0]);
        let mut engine = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::new(RecordingInterval { log: log.clone() }),
            connect: Box::new(FixedGateConnect::new(1)),
            sub_theta_mod: Box::<NoneMod>::default(),
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        };
        let state = CoreState { is_alive: true };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&candidates);
        let _ = engine.process_candidates(
            &ctx,
            &candidates,
            &mut timing_grid,
            &timing_field,
            &state,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        assert_eq!(*log.lock().expect("dt log"), vec![1.0, 1.0]);
    }

    #[test]
    fn cosine_mod_has_unit_mean() {
        let modulator = CosineHarmonicMod {
            n: 3,
            depth: 0.8,
            phase0: 0.2,
        };
        let mut sum = 0.0;
        let samples = 1000;
        for i in 0..samples {
            let phase = i as f32 / samples as f32;
            sum += modulator.mod_at_phase(phase);
        }
        let avg = sum / samples as f32;
        assert!((avg - 1.0).abs() < 1e-3);
    }

    #[test]
    fn sub_theta_mod_biases_onset() {
        let candidates = vec![
            CandidatePoint {
                tick: 0,
                gate: 0,
                theta_pos: 0.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 10,
                gate: 0,
                theta_pos: 0.5,
                phase_in_gate: 0.5,
                sources: vec![ClockSource::Subdivision { n: 2 }],
            },
            CandidatePoint {
                tick: 20,
                gate: 1,
                theta_pos: 1.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
        ];
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 30,
            fs: 1000.0,
            rhythms: NeuralRhythms::default(),
        };
        let timing_field = TimingField::from_values(0, vec![1.0, 1.0]);
        let mut interval = AccumulatorInterval::new(50.0, 0, 1);
        interval.acc = 0.0;
        let mut engine = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::new(interval),
            connect: Box::new(FixedGateConnect::new(1)),
            sub_theta_mod: Box::new(CosineHarmonicMod {
                n: 1,
                depth: 1.0,
                phase0: -std::f32::consts::PI,
            }),
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: Some(0.0),
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        };
        let state = CoreState { is_alive: true };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&candidates);
        let _ = engine.process_candidates(
            &ctx,
            &candidates,
            &mut timing_grid,
            &timing_field,
            &state,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].onset_tick, 10);
    }

    #[test]
    fn active_notes_tracks_note_on_and_off() {
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 2,
            fs: 1000.0,
            rhythms: NeuralRhythms::default(),
        };
        let mut interval = AccumulatorInterval::new(1.0, 0, 1);
        interval.acc = 1.0;
        let mut engine = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::new(interval),
            connect: Box::new(FixedGateConnect::new(0)),
            sub_theta_mod: Box::<NoneMod>::default(),
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        };
        let state = CoreState { is_alive: true };
        let timing_field = TimingField::from_values(0, vec![1.0]);
        let candidates = vec![CandidatePoint {
            tick: 0,
            gate: 0,
            theta_pos: 0.0,
            phase_in_gate: 0.0,
            sources: vec![ClockSource::GateBoundary],
        }];
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&candidates);
        let _ = engine.process_candidates(
            &ctx,
            &candidates,
            &mut timing_grid,
            &timing_field,
            &state,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        assert_eq!(engine.active_notes, 0, "note on/off should settle to zero");
        assert!(
            cmds.iter()
                .any(|cmd| matches!(cmd, PhonationCmd::NoteOn { .. }))
        );
        assert!(
            cmds.iter()
                .any(|cmd| matches!(cmd, PhonationCmd::NoteOff { .. }))
        );
    }

    #[test]
    fn active_notes_increments_on_note_on() {
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 2,
            fs: 1000.0,
            rhythms: NeuralRhythms::default(),
        };
        let mut interval = AccumulatorInterval::new(1.0, 0, 1);
        interval.acc = 1.0;
        let mut engine = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::new(interval),
            connect: Box::new(FixedGateConnect::new(1)),
            sub_theta_mod: Box::<NoneMod>::default(),
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        };
        let state = CoreState { is_alive: true };
        let timing_field = TimingField::from_values(0, vec![1.0]);
        let candidates = vec![CandidatePoint {
            tick: 0,
            gate: 0,
            theta_pos: 0.0,
            phase_in_gate: 0.0,
            sources: vec![ClockSource::GateBoundary],
        }];
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&candidates);
        let _ = engine.process_candidates(
            &ctx,
            &candidates,
            &mut timing_grid,
            &timing_field,
            &state,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        assert_eq!(engine.active_notes, 1);
        assert!(
            cmds.iter()
                .any(|cmd| matches!(cmd, PhonationCmd::NoteOn { .. }))
        );
        assert!(
            !cmds
                .iter()
                .any(|cmd| matches!(cmd, PhonationCmd::NoteOff { .. }))
        );
    }

    #[test]
    fn pending_off_drains_without_candidates() {
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 10,
            fs: 1000.0,
            rhythms: NeuralRhythms::default(),
        };
        let mut engine = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::<NoneInterval>::default(),
            connect: Box::new(FixedGateConnect::new(1)),
            sub_theta_mod: Box::<NoneMod>::default(),
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        };
        engine.schedule_note_off(42, 5);
        let state = CoreState { is_alive: true };
        let timing_field = TimingField::from_values(0, Vec::new());
        let candidates: Vec<CandidatePoint> = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&candidates);
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let _ = engine.process_candidates(
            &ctx,
            &candidates,
            &mut timing_grid,
            &timing_field,
            &state,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        assert!(cmds.iter().any(|cmd| matches!(
            cmd,
            PhonationCmd::NoteOff {
                note_id: 42,
                off_tick: 5
            }
        )));
    }

    #[test]
    fn note_off_emits_before_note_on_when_due() {
        struct AlwaysInterval;

        impl PhonationInterval for AlwaysInterval {
            fn on_candidate(
                &mut self,
                _c: &IntervalInput,
                _state: &CoreState,
            ) -> Option<PhonationKick> {
                Some(PhonationKick::Planned { strength: 1.0 })
            }
        }

        struct NoopConnect;

        impl PhonationConnect for NoopConnect {
            fn on_note_on(&mut self, _onset: ConnectOnset) -> ConnectPlan {
                ConnectPlan::None
            }

            fn poll(&mut self, _now: ConnectNow, _out: &mut Vec<PhonationCmd>) {}
        }

        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 2,
            fs: 1000.0,
            rhythms: NeuralRhythms::default(),
        };
        let candidates = vec![CandidatePoint {
            tick: 0,
            gate: 0,
            theta_pos: 0.0,
            phase_in_gate: 0.0,
            sources: vec![ClockSource::GateBoundary],
        }];
        let timing_field = TimingField::from_values(0, vec![1.0]);
        let mut engine = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::new(AlwaysInterval),
            connect: Box::new(NoopConnect),
            sub_theta_mod: Box::<NoneMod>::default(),
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        };
        engine.schedule_note_off(99, 0);
        let state = CoreState { is_alive: true };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&candidates);
        let _ = engine.process_candidates(
            &ctx,
            &candidates,
            &mut timing_grid,
            &timing_field,
            &state,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        let off_pos = cmds
            .iter()
            .position(|cmd| matches!(cmd, PhonationCmd::NoteOff { note_id: 99, .. }));
        let on_pos = cmds
            .iter()
            .position(|cmd| matches!(cmd, PhonationCmd::NoteOn { .. }));
        assert!(off_pos.is_some());
        assert!(on_pos.is_some());
        assert!(off_pos < on_pos);
    }

    #[test]
    fn schedule_hold_theta_fallback_never_schedules_before_onset() {
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 1,
            fs: 100.0,
            rhythms: NeuralRhythms::default(),
        };
        let mut engine = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::<NoneInterval>::default(),
            connect: Box::new(FixedGateConnect::new(1)),
            sub_theta_mod: Box::<NoneMod>::default(),
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        };
        let onset = ConnectOnset {
            note_id: 7,
            tick: 10,
            gate: 0,
            theta_pos: 0.0,
            exc_gate: 0.0,
            exc_slope: 0.0,
        };
        let mut timing_grid = ThetaGrid {
            boundaries: Vec::new(),
        };
        engine.schedule_hold_theta(onset, 1.0, &ctx, &mut timing_grid);
        let off_tick = engine
            .pending_off
            .peek()
            .map(|entry| entry.0.off_tick)
            .expect("pending off");
        assert!(off_tick >= onset.tick);
    }

    #[test]
    fn schedule_hold_theta_does_not_panic_on_short_interval() {
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 2,
            fs: 100.0,
            rhythms: NeuralRhythms::default(),
        };
        let mut engine = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::<NoneInterval>::default(),
            connect: Box::new(FixedGateConnect::new(1)),
            sub_theta_mod: Box::<NoneMod>::default(),
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        };
        let onset = ConnectOnset {
            note_id: 11,
            tick: 0,
            gate: 0,
            theta_pos: 0.0,
            exc_gate: 0.0,
            exc_slope: 0.0,
        };
        let mut timing_grid = ThetaGrid {
            boundaries: vec![
                GateBoundary { gate: 0, tick: 0 },
                GateBoundary { gate: 1, tick: 1 },
            ],
        };
        engine.schedule_hold_theta(onset, 0.5, &ctx, &mut timing_grid);
        let off_tick = engine
            .pending_off
            .peek()
            .map(|entry| entry.0.off_tick)
            .expect("pending off");
        assert!(off_tick >= onset.tick);
    }

    #[test]
    fn tick_at_large_tick_precision() {
        let grid = ThetaGrid {
            boundaries: vec![
                GateBoundary {
                    gate: 0,
                    tick: 100_000_000,
                },
                GateBoundary {
                    gate: 1,
                    tick: 100_000_100,
                },
            ],
        };
        assert_eq!(grid.tick_at(0, 0.5), Some(100_000_050));
    }

    #[test]
    fn merge_candidates_keeps_distinct_gates() {
        let candidates = vec![
            CandidatePoint {
                tick: 10,
                gate: 0,
                theta_pos: 0.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 10,
                gate: 1,
                theta_pos: 1.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
        ];
        let merged = PhonationEngine::merge_candidates(candidates);
        assert_eq!(merged.len(), 2);
        assert_ne!(merged[0].gate, merged[1].gate);
    }

    #[test]
    fn update_from_config_falls_back_on_mismatch() {
        let base = PhonationConfig {
            mode: PhonationMode::Gated,
            interval: PhonationIntervalConfig::None,
            connect: PhonationConnectConfig::FixedGate { length_gates: 1 },
            clock: PhonationClockConfig::ThetaGate,
            sub_theta_mod: SubThetaModConfig::None,
            social: SocialConfig::default(),
        };
        let mut engine = PhonationEngine::from_config(&base, 7);
        let changed = PhonationConfig {
            interval: PhonationIntervalConfig::Accumulator {
                rate: 2.0,
                refractory: 2,
            },
            connect: PhonationConnectConfig::Field {
                hold_min_theta: 0.2,
                hold_max_theta: 0.5,
                curve_k: 2.0,
                curve_x0: 0.4,
                drop_gain: 0.1,
            },
            ..base
        };
        engine.update_from_config(&changed);
        let state = CoreState { is_alive: true };
        let kick = engine.interval.on_candidate(
            &IntervalInput {
                gate: 0,
                tick: 0,
                dt_theta: 1.0,
                dt_sec: 1.0,
                weight: 1.0,
            },
            &state,
        );
        assert!(kick.is_some(), "expected accumulator interval to be active");
        let plan = engine.connect.on_note_on(ConnectOnset {
            note_id: 1,
            tick: 0,
            gate: 0,
            theta_pos: 0.0,
            exc_gate: 0.2,
            exc_slope: 0.1,
        });
        assert!(
            matches!(plan, ConnectPlan::HoldTheta(_)),
            "expected field connect plan"
        );
    }

    #[test]
    fn merge_candidates_prioritizes_gate_boundary_on_collision() {
        let candidates = vec![
            CandidatePoint {
                tick: 10,
                gate: 1,
                theta_pos: 1.5,
                phase_in_gate: 0.5,
                sources: vec![ClockSource::Subdivision { n: 2 }],
            },
            CandidatePoint {
                tick: 10,
                gate: 1,
                theta_pos: 1.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
        ];
        let merged = PhonationEngine::merge_candidates(candidates);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].phase_in_gate, 0.0);
        assert!(merged[0].sources.contains(&ClockSource::GateBoundary));
        assert!(
            merged[0]
                .sources
                .contains(&ClockSource::Subdivision { n: 2 })
        );
    }

    #[test]
    fn fixed_gate_connect_releases_on_next_gate_tick() {
        let mut rhythms = NeuralRhythms::default();
        rhythms.env_open = 1.0;
        rhythms.env_level = 1.0;
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 10,
            fs: 1000.0,
            rhythms,
        };
        let mut interval = AccumulatorInterval::new(250.0, 0, 1);
        interval.acc = 0.0;
        struct StubClock {
            ticks: Vec<Tick>,
            index: usize,
        }

        impl PhonationClock for StubClock {
            fn gather_candidates(&mut self, ctx: &CoreTickCtx, out: &mut Vec<CandidatePoint>) {
                while self.index < self.ticks.len() {
                    let tick = self.ticks[self.index];
                    if tick < ctx.now_tick || tick >= ctx.frame_end {
                        return;
                    }
                    let gate = self.index as u64;
                    out.push(CandidatePoint {
                        tick,
                        gate,
                        theta_pos: gate as f64,
                        phase_in_gate: 0.0,
                        sources: vec![ClockSource::GateBoundary],
                    });
                    self.index += 1;
                }
            }
        }

        let mut engine = PhonationEngine {
            clock: Box::new(StubClock {
                ticks: vec![0, 4, 8],
                index: 0,
            }),
            interval: Box::new(interval),
            connect: Box::new(FixedGateConnect::new(1)),
            sub_theta_mod: Box::<NoneMod>::default(),
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        };
        let state = CoreState { is_alive: true };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        engine.tick(
            &ctx,
            &state,
            None,
            0.0,
            1.0,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        assert!(events.len() >= 2, "expected at least two note ons");
        let first = events[0];
        let second = events[1];
        let off_tick = cmds.iter().find_map(|cmd| match cmd {
            PhonationCmd::NoteOff { note_id, off_tick } if *note_id == first.note_id => {
                Some(*off_tick)
            }
            _ => None,
        });
        assert_eq!(off_tick, Some(second.onset_tick));
    }

    #[test]
    fn timing_field_applies_social_coupling() {
        let mut rhythms = NeuralRhythms::default();
        rhythms.env_open = 1.0;
        rhythms.env_level = 1.0;
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 20,
            fs: 1000.0,
            rhythms,
        };
        let grid = ThetaGrid {
            boundaries: vec![
                GateBoundary { gate: 0, tick: 0 },
                GateBoundary { gate: 1, tick: 10 },
            ],
        };
        let trace = SocialDensityTrace {
            start_tick: 0,
            bin_ticks: 10,
            bins: vec![1.0, 0.0],
        };
        let timing_field = TimingField::build_from(&ctx, &grid, Some((&trace, -1.0)), 1.0);
        assert!(timing_field.e(0) < 0.5);
        assert!((timing_field.e(1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn field_connect_holds_longer_with_high_excitation() {
        let mut connect = FieldConnect::new(0.25, 1.0, 10.0, 0.5, 0.0);
        let low_plan = connect.on_note_on(ConnectOnset {
            note_id: 1,
            tick: 0,
            gate: 0,
            theta_pos: 0.0,
            exc_gate: 0.0,
            exc_slope: 0.0,
        });
        let high_plan = connect.on_note_on(ConnectOnset {
            note_id: 2,
            tick: 0,
            gate: 0,
            theta_pos: 0.0,
            exc_gate: 1.0,
            exc_slope: 0.0,
        });
        let hold_low = match low_plan {
            ConnectPlan::HoldTheta(hold) => hold,
            _ => panic!("expected hold plan for low excitation"),
        };
        let hold_high = match high_plan {
            ConnectPlan::HoldTheta(hold) => hold,
            _ => panic!("expected hold plan for high excitation"),
        };
        assert!(hold_high > hold_low);
    }

    #[test]
    fn field_connect_schedules_off_tick_when_next_boundary_not_in_candidates() {
        struct TickInterval {
            tick: Tick,
        }

        impl PhonationInterval for TickInterval {
            fn on_candidate(
                &mut self,
                c: &IntervalInput,
                _state: &CoreState,
            ) -> Option<PhonationKick> {
                if c.tick == self.tick {
                    Some(PhonationKick::Planned { strength: 1.0 })
                } else {
                    None
                }
            }
        }

        let mut rhythms = NeuralRhythms::default();
        rhythms.theta.freq_hz = 1.0;
        rhythms.theta.phase = 0.0;
        let fs = 100.0;
        let expected_gate1_tick =
            next_gate_tick(1, fs, rhythms.theta, 0.0).expect("expected gate1 tick");
        let mut expected_off_tick = ((expected_gate1_tick as f64) * 0.5).round() as Tick;
        let lo = 1;
        let hi = expected_gate1_tick.saturating_sub(1);
        assert!(lo <= hi);
        expected_off_tick = expected_off_tick.clamp(lo, hi);
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: expected_off_tick.saturating_add(1),
            fs,
            rhythms,
        };
        let timing_field = TimingField::from_values(0, vec![1.0, 1.0]);
        let state = CoreState { is_alive: true };
        let candidates = vec![CandidatePoint {
            tick: 0,
            gate: 0,
            theta_pos: 0.0,
            phase_in_gate: 0.0,
            sources: vec![ClockSource::GateBoundary],
        }];
        let mut engine = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::new(TickInterval { tick: 0 }),
            connect: Box::new(FieldConnect::new(0.5, 0.5, 10.0, 0.5, 0.0)),
            sub_theta_mod: Box::<NoneMod>::default(),
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&candidates);
        let _ = engine.process_candidates(
            &ctx,
            &candidates,
            &mut timing_grid,
            &timing_field,
            &state,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        assert!(cmds.iter().any(|cmd| matches!(
            cmd,
            PhonationCmd::NoteOff {
                off_tick,
                ..
            } if *off_tick == expected_off_tick
        )));
    }

    #[test]
    fn field_connect_off_tick_independent_of_subdivision() {
        struct TickInterval {
            tick: Tick,
        }

        impl PhonationInterval for TickInterval {
            fn on_candidate(
                &mut self,
                c: &IntervalInput,
                _state: &CoreState,
            ) -> Option<PhonationKick> {
                if c.tick == self.tick {
                    Some(PhonationKick::Planned { strength: 1.0 })
                } else {
                    None
                }
            }
        }

        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 120,
            fs: 1000.0,
            rhythms: NeuralRhythms::default(),
        };
        let timing_field = TimingField::from_values(0, vec![1.0, 1.0]);
        let state = CoreState { is_alive: true };

        let candidates_base = vec![
            CandidatePoint {
                tick: 0,
                gate: 0,
                theta_pos: 0.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 100,
                gate: 1,
                theta_pos: 1.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
        ];

        let mut engine_base = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::new(TickInterval { tick: 0 }),
            connect: Box::new(FieldConnect::new(0.5, 0.5, 10.0, 0.5, 0.0)),
            sub_theta_mod: Box::<NoneMod>::default(),
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&candidates_base);
        let _ = engine_base.process_candidates(
            &ctx,
            &candidates_base,
            &mut timing_grid,
            &timing_field,
            &state,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        let note_on_id = cmds.iter().find_map(|cmd| match cmd {
            PhonationCmd::NoteOn { note_id, .. } => Some(*note_id),
            _ => None,
        });
        let off_tick_base = cmds.iter().find_map(|cmd| match cmd {
            PhonationCmd::NoteOff { note_id, off_tick } if Some(*note_id) == note_on_id => {
                Some(*off_tick)
            }
            _ => None,
        });

        let candidates_sub = vec![
            CandidatePoint {
                tick: 0,
                gate: 0,
                theta_pos: 0.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 50,
                gate: 0,
                theta_pos: 0.5,
                phase_in_gate: 0.5,
                sources: vec![ClockSource::Subdivision { n: 2 }],
            },
            CandidatePoint {
                tick: 100,
                gate: 1,
                theta_pos: 1.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
        ];

        let mut engine_sub = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::new(TickInterval { tick: 0 }),
            connect: Box::new(FieldConnect::new(0.5, 0.5, 10.0, 0.5, 0.0)),
            sub_theta_mod: Box::<NoneMod>::default(),
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        };
        let mut cmds_sub = Vec::new();
        let mut events_sub = Vec::new();
        let mut onsets_sub = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&candidates_sub);
        let _ = engine_sub.process_candidates(
            &ctx,
            &candidates_sub,
            &mut timing_grid,
            &timing_field,
            &state,
            None,
            &mut cmds_sub,
            &mut events_sub,
            &mut onsets_sub,
        );
        let note_on_id_sub = cmds_sub.iter().find_map(|cmd| match cmd {
            PhonationCmd::NoteOn { note_id, .. } => Some(*note_id),
            _ => None,
        });
        let off_tick_sub = cmds_sub.iter().find_map(|cmd| match cmd {
            PhonationCmd::NoteOff { note_id, off_tick } if Some(*note_id) == note_on_id_sub => {
                Some(*off_tick)
            }
            _ => None,
        });

        assert_eq!(off_tick_base, Some(50));
        assert_eq!(off_tick_sub, Some(50));
    }

    #[test]
    fn field_connect_note_off_survives_empty_hop() {
        struct TickInterval {
            tick: Tick,
        }

        impl PhonationInterval for TickInterval {
            fn on_candidate(
                &mut self,
                c: &IntervalInput,
                _state: &CoreState,
            ) -> Option<PhonationKick> {
                if c.tick == self.tick {
                    Some(PhonationKick::Planned { strength: 1.0 })
                } else {
                    None
                }
            }
        }

        let timing_field = TimingField::from_values(0, vec![1.0, 1.0]);
        let state = CoreState { is_alive: true };
        let mut engine = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::new(TickInterval { tick: 0 }),
            connect: Box::new(FieldConnect::new(1.0, 1.0, 10.0, 0.5, 0.0)),
            sub_theta_mod: Box::<NoneMod>::default(),
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        };

        let ctx1 = CoreTickCtx {
            now_tick: 0,
            frame_end: 50,
            fs: 1000.0,
            rhythms: NeuralRhythms::default(),
        };
        let candidates = vec![CandidatePoint {
            tick: 0,
            gate: 0,
            theta_pos: 0.0,
            phase_in_gate: 0.0,
            sources: vec![ClockSource::GateBoundary],
        }];
        let mut timing_grid = ThetaGrid {
            boundaries: vec![
                GateBoundary { gate: 0, tick: 0 },
                GateBoundary { gate: 1, tick: 100 },
                GateBoundary { gate: 2, tick: 200 },
            ],
        };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let _ = engine.process_candidates(
            &ctx1,
            &candidates,
            &mut timing_grid,
            &timing_field,
            &state,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        let note_on_id = cmds.iter().find_map(|cmd| match cmd {
            PhonationCmd::NoteOn { note_id, .. } => Some(*note_id),
            _ => None,
        });
        assert!(note_on_id.is_some());
        assert!(
            !cmds
                .iter()
                .any(|cmd| matches!(cmd, PhonationCmd::NoteOff { .. }))
        );

        let ctx2 = CoreTickCtx {
            now_tick: 50,
            frame_end: 120,
            fs: 1000.0,
            rhythms: NeuralRhythms::default(),
        };
        let empty: Vec<CandidatePoint> = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&empty);
        let mut cmds2 = Vec::new();
        let mut events2 = Vec::new();
        let mut onsets2 = Vec::new();
        let _ = engine.process_candidates(
            &ctx2,
            &empty,
            &mut timing_grid,
            &timing_field,
            &state,
            None,
            &mut cmds2,
            &mut events2,
            &mut onsets2,
        );
        let off_tick = cmds2.iter().find_map(|cmd| match cmd {
            PhonationCmd::NoteOff { note_id, off_tick } if Some(*note_id) == note_on_id => {
                Some(*off_tick)
            }
            _ => None,
        });
        assert_eq!(off_tick, Some(100));
    }

    #[test]
    fn exc_slope_is_stable_with_sub_candidates() {
        use std::collections::HashSet;
        use std::sync::{Arc, Mutex};

        struct TickFilteredInterval {
            ticks: HashSet<Tick>,
        }

        impl PhonationInterval for TickFilteredInterval {
            fn on_candidate(
                &mut self,
                c: &IntervalInput,
                _state: &CoreState,
            ) -> Option<PhonationKick> {
                if self.ticks.contains(&c.tick) {
                    Some(PhonationKick::Planned { strength: 1.0 })
                } else {
                    None
                }
            }
        }

        struct RecordingConnect {
            slopes: Arc<Mutex<Vec<f32>>>,
        }

        impl PhonationConnect for RecordingConnect {
            fn on_note_on(&mut self, onset: ConnectOnset) -> ConnectPlan {
                self.slopes.lock().expect("slopes").push(onset.exc_slope);
                ConnectPlan::None
            }

            fn poll(&mut self, _now: ConnectNow, _out: &mut Vec<PhonationCmd>) {}
        }

        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 30,
            fs: 1000.0,
            rhythms: NeuralRhythms::default(),
        };
        let timing_field = TimingField::from_values(0, vec![0.2, 0.8, 0.2]);
        let state = CoreState { is_alive: true };

        let slopes_base = Arc::new(Mutex::new(Vec::new()));
        let mut engine_base = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::new(TickFilteredInterval {
                ticks: [0, 10, 20].into_iter().collect(),
            }),
            connect: Box::new(RecordingConnect {
                slopes: slopes_base.clone(),
            }),
            sub_theta_mod: Box::<NoneMod>::default(),
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        };
        let candidates_base = vec![
            CandidatePoint {
                tick: 0,
                gate: 0,
                theta_pos: 0.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 10,
                gate: 1,
                theta_pos: 1.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 20,
                gate: 2,
                theta_pos: 2.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
        ];
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&candidates_base);
        let _ = engine_base.process_candidates(
            &ctx,
            &candidates_base,
            &mut timing_grid,
            &timing_field,
            &state,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        let base_slopes = slopes_base.lock().expect("slopes").clone();

        let slopes_sub = Arc::new(Mutex::new(Vec::new()));
        let mut engine_sub = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::new(TickFilteredInterval {
                ticks: [0, 10, 20].into_iter().collect(),
            }),
            connect: Box::new(RecordingConnect {
                slopes: slopes_sub.clone(),
            }),
            sub_theta_mod: Box::<NoneMod>::default(),
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        };
        let candidates_sub = vec![
            CandidatePoint {
                tick: 0,
                gate: 0,
                theta_pos: 0.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 5,
                gate: 0,
                theta_pos: 0.5,
                phase_in_gate: 0.5,
                sources: vec![ClockSource::Subdivision { n: 2 }],
            },
            CandidatePoint {
                tick: 10,
                gate: 1,
                theta_pos: 1.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 15,
                gate: 1,
                theta_pos: 1.5,
                phase_in_gate: 0.5,
                sources: vec![ClockSource::Subdivision { n: 2 }],
            },
            CandidatePoint {
                tick: 20,
                gate: 2,
                theta_pos: 2.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
        ];
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&candidates_sub);
        let _ = engine_sub.process_candidates(
            &ctx,
            &candidates_sub,
            &mut timing_grid,
            &timing_field,
            &state,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        let sub_slopes = slopes_sub.lock().expect("slopes").clone();
        assert_eq!(base_slopes, sub_slopes);
    }

    #[test]
    fn timing_field_social_coupling_clamps_exponent() {
        let mut rhythms = NeuralRhythms::default();
        rhythms.env_open = 1.0;
        rhythms.env_level = 1.0;
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 1,
            fs: 1000.0,
            rhythms,
        };
        let grid = ThetaGrid {
            boundaries: vec![GateBoundary { gate: 0, tick: 0 }],
        };
        let trace = SocialDensityTrace {
            start_tick: 0,
            bin_ticks: 1,
            bins: vec![1000.0],
        };
        let timing_field = TimingField::build_from(&ctx, &grid, Some((&trace, 1000.0)), 1.0);
        assert!(timing_field.e(0).is_finite());
    }
}
