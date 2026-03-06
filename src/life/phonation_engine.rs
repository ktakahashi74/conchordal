use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::f32::consts::TAU;
use std::fmt;

use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::Tick;
use crate::life::control_adapters::phonation_config_from_spec;
use crate::life::gate_clock::next_gate_tick;
use crate::life::scenario::{
    DurationConfig, OnsetConfig, PhonationClockConfig, PhonationConfig, PhonationMode,
    PhonationSpec, SubThetaModConfig,
};
use crate::life::social_density::SocialDensityTrace;

pub type NoteId = u64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OnsetKick {
    Planned { strength: f32 },
}

impl OnsetKick {
    pub fn strength(self) -> f32 {
        match self {
            OnsetKick::Planned { strength } => strength,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct NoteUpdate {
    pub target_freq_hz: Option<f32>,
    /// Final output target amp (linear).
    pub target_amp: Option<f32>,
}

impl NoteUpdate {
    pub fn is_empty(&self) -> bool {
        self.target_freq_hz.is_none() && self.target_amp.is_none()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoteCmd {
    NoteOn {
        note_id: NoteId,
        kick: OnsetKick,
    },
    NoteOff {
        note_id: NoteId,
        off_tick: Tick,
    },
    Update {
        note_id: NoteId,
        at_tick: Option<Tick>,
        update: NoteUpdate,
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

impl ThetaGateClock {
    fn gather_candidates_impl(&mut self, ctx: &CoreTickCtx, out: &mut Vec<CandidatePoint>) {
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

pub enum PhonationClock {
    ThetaGate(ThetaGateClock),
    Composite {
        gate_clock: ThetaGateClock,
        subdivision: Option<SubdivisionClock>,
        internal_phase: Option<InternalPhaseClock>,
    },
    #[cfg(test)]
    Custom(Box<dyn FnMut(&CoreTickCtx, &mut Vec<CandidatePoint>) + Send>),
}

impl PhonationClock {
    pub fn gather_candidates(&mut self, ctx: &CoreTickCtx, out: &mut Vec<CandidatePoint>) {
        match self {
            PhonationClock::ThetaGate(clock) => {
                clock.gather_candidates_impl(ctx, out);
            }
            PhonationClock::Composite {
                gate_clock,
                subdivision,
                internal_phase,
            } => {
                let mut gate_candidates = Vec::new();
                gate_clock.gather_candidates_impl(ctx, &mut gate_candidates);
                out.extend(gate_candidates.iter().cloned());
                let grid = ThetaGrid::from_candidates(&gate_candidates);
                if let Some(clock) = subdivision.as_ref() {
                    clock.gather_candidates(&grid, out);
                }
                if let Some(clock) = internal_phase.as_mut() {
                    clock.gather_candidates(&grid, out);
                }
            }
            #[cfg(test)]
            PhonationClock::Custom(f) => f(ctx, out),
        }
    }

    pub fn from_config(config: &PhonationClockConfig) -> Self {
        match config {
            PhonationClockConfig::ThetaGate => PhonationClock::ThetaGate(ThetaGateClock::default()),
            PhonationClockConfig::Composite {
                subdivision,
                internal_phase,
            } => {
                let subdivision = subdivision
                    .as_ref()
                    .map(|c| SubdivisionClock::new(c.divisions.clone()));
                let internal_phase = internal_phase
                    .as_ref()
                    .map(|c| InternalPhaseClock::new(c.ratio, c.phase0));
                PhonationClock::Composite {
                    gate_clock: ThetaGateClock::default(),
                    subdivision,
                    internal_phase,
                }
            }
        }
    }
}

pub enum SubThetaMod {
    None,
    CosineHarmonic { n: u32, depth: f32, phase0: f32 },
}

impl SubThetaMod {
    pub fn mod_at_phase(&self, phase_in_gate: f32) -> f32 {
        match self {
            SubThetaMod::None => 1.0,
            SubThetaMod::CosineHarmonic { n, depth, phase0 } => {
                let depth = depth.clamp(0.0, 1.0);
                let phase = phase_in_gate.rem_euclid(1.0);
                1.0 + depth * (TAU * *n as f32 * phase + phase0).cos()
            }
        }
    }

    fn from_config(config: &SubThetaModConfig) -> Self {
        match config {
            SubThetaModConfig::None => SubThetaMod::None,
            SubThetaModConfig::Cosine { n, depth, phase0 } => SubThetaMod::CosineHarmonic {
                n: *n,
                depth: *depth,
                phase0: *phase0,
            },
        }
    }

    fn update_config(&mut self, config: &SubThetaModConfig) {
        *self = Self::from_config(config);
    }
}

pub enum OnsetRule {
    None,
    Accumulator {
        rate_hz: f32,
        refractory_gates: u32,
        acc: f32,
        next_allowed_gate: u64,
    },
    #[cfg(test)]
    Custom(Box<dyn FnMut(&IntervalInput, &CoreState) -> Option<OnsetKick> + Send>),
}

impl OnsetRule {
    pub fn accumulator(rate_hz: f32, refractory_gates: u32, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let acc = rng.random_range(0.7..1.0);
        OnsetRule::Accumulator {
            rate_hz,
            refractory_gates,
            acc,
            next_allowed_gate: 0,
        }
    }

    pub fn on_candidate(&mut self, c: &IntervalInput, state: &CoreState) -> Option<OnsetKick> {
        match self {
            OnsetRule::None => None,
            OnsetRule::Accumulator {
                rate_hz,
                refractory_gates,
                acc,
                next_allowed_gate,
            } => {
                const ACC_MAX: f32 = 32.0;
                if !state.is_alive {
                    return None;
                }
                if !rate_hz.is_finite() || *rate_hz <= 0.0 {
                    return None;
                }
                let weight = c.weight.max(0.0);
                *acc += *rate_hz * weight * c.dt_sec;
                if !acc.is_finite() || *acc > ACC_MAX {
                    *acc = ACC_MAX;
                }
                let refractory_ok = c.gate >= *next_allowed_gate;
                if *acc >= 1.0 && refractory_ok {
                    *acc -= 1.0;
                    *next_allowed_gate = c.gate + *refractory_gates as u64 + 1;
                    return Some(OnsetKick::Planned { strength: 1.0 });
                }
                None
            }
            #[cfg(test)]
            OnsetRule::Custom(f) => f(c, state),
        }
    }

    fn from_config(config: &OnsetConfig, seed: u64) -> Self {
        match config {
            OnsetConfig::None => OnsetRule::None,
            OnsetConfig::Accumulator { rate, refractory } => {
                OnsetRule::accumulator(*rate, *refractory, seed)
            }
        }
    }

    fn update_config(&mut self, config: &OnsetConfig, seed: u64) {
        if let (
            OnsetConfig::Accumulator { rate, refractory },
            OnsetRule::Accumulator {
                rate_hz,
                refractory_gates,
                ..
            },
        ) = (config, &mut *self)
        {
            *rate_hz = *rate;
            *refractory_gates = *refractory;
        } else {
            *self = Self::from_config(config, seed);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OnsetContext {
    pub note_id: NoteId,
    pub tick: Tick,
    pub gate: u64,
    pub theta_pos: f64,
    pub exc_gate: f32,
    pub exc_slope: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DurationPlan {
    None,
    HoldTheta(f32),
    AtGate(u64),
}

pub enum DurationRule {
    FixedGate {
        length_gates: u32,
    },
    Field {
        hold_min_theta: f32,
        hold_max_theta: f32,
        curve_k: f32,
        curve_x0: f32,
        drop_gain: f32,
    },
    #[cfg(test)]
    Custom(Box<dyn FnMut(OnsetContext) -> DurationPlan + Send>),
}

impl DurationRule {
    pub fn fixed_gate(length_gates: u32) -> Self {
        DurationRule::FixedGate { length_gates }
    }

    pub fn field(
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
        DurationRule::Field {
            hold_min_theta,
            hold_max_theta,
            curve_k,
            curve_x0,
            drop_gain,
        }
    }

    pub fn on_note_on(&mut self, onset: OnsetContext) -> DurationPlan {
        match self {
            DurationRule::FixedGate { length_gates } => {
                let off_gate = onset.gate + *length_gates as u64;
                DurationPlan::AtGate(off_gate)
            }
            DurationRule::Field {
                hold_min_theta,
                hold_max_theta,
                curve_k,
                curve_x0,
                drop_gain,
            } => {
                let exc_gate = onset.exc_gate.clamp(0.0, 1.0);
                let min = hold_min_theta.max(0.0);
                let max = hold_max_theta.max(min);
                let p = 1.0 / (1.0 + (-(*curve_k * (exc_gate - *curve_x0))).exp());
                let mut hold = min + (max - min) * p;
                if *drop_gain > 0.0 && onset.exc_slope.is_finite() {
                    let drop_val = (-onset.exc_slope).max(0.0).clamp(0.0, 1.0);
                    hold *= 1.0 - drop_gain.clamp(0.0, 1.0) * drop_val;
                }
                DurationPlan::HoldTheta(hold.max(0.0))
            }
            #[cfg(test)]
            DurationRule::Custom(f) => f(onset),
        }
    }

    fn from_config(config: &DurationConfig) -> Self {
        match config {
            DurationConfig::FixedGate { length_gates } => DurationRule::fixed_gate(*length_gates),
            DurationConfig::Field {
                hold_min_theta,
                hold_max_theta,
                curve_k,
                curve_x0,
                drop_gain,
            } => DurationRule::field(
                *hold_min_theta,
                *hold_max_theta,
                *curve_k,
                *curve_x0,
                *drop_gain,
            ),
        }
    }

    fn update_config(&mut self, config: &DurationConfig) {
        *self = Self::from_config(config);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NoteOnEvent {
    pub note_id: NoteId,
    pub onset_tick: Tick,
}

#[derive(Debug, Clone, Copy)]
pub struct OnsetEvent {
    pub gate: u64,
    pub onset_tick: Tick,
    pub strength: f32,
}

struct CandidateStep {
    dt_theta: f32,
    dt_sec: f32,
    exc_gate: f32,
    exc_slope: f32,
    weight: f32,
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
    pub clock: PhonationClock,
    pub onset_rule: OnsetRule,
    pub duration_rule: DurationRule,
    pub sub_theta_mod: SubThetaMod,
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
    pub fn from_spec(spec: &PhonationSpec, seed: u64) -> Self {
        let config = phonation_config_from_spec(spec);
        Self::from_config(&config, seed)
    }

    pub fn update_from_spec(&mut self, spec: &PhonationSpec) {
        let config = phonation_config_from_spec(spec);
        self.update_from_config(&config);
    }

    pub fn from_config(config: &PhonationConfig, seed: u64) -> Self {
        let onset_rule = OnsetRule::from_config(&config.onset, seed);
        let sub_theta_mod = SubThetaMod::from_config(&config.sub_theta_mod);
        let duration_rule = DurationRule::from_config(&config.duration);
        Self {
            clock: PhonationClock::from_config(&config.clock),
            onset_rule,
            duration_rule,
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
        let seed = self.initial_seed ^ 0xA5A5_5A5A_5A5A_A5A5;
        self.onset_rule.update_config(&config.onset, seed);
        self.duration_rule.update_config(&config.duration);
        self.sub_theta_mod.update_config(&config.sub_theta_mod);
    }

    pub fn has_active_notes(&self) -> bool {
        self.active_notes > 0
    }

    fn tick_hold(
        &mut self,
        ctx: &CoreTickCtx,
        state: &CoreState,
        min_allowed_onset_tick: Option<Tick>,
        out_cmds: &mut Vec<NoteCmd>,
        out_events: &mut Vec<NoteOnEvent>,
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
            out_cmds.push(NoteCmd::NoteOn {
                note_id,
                kick: OnsetKick::Planned { strength: 1.0 },
            });
            self.active_notes = self.active_notes.saturating_add(1);
            out_events.push(NoteOnEvent {
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
            out_cmds.push(NoteCmd::NoteOff {
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

    fn drain_note_offs(&mut self, up_to_tick: Tick, out_cmds: &mut Vec<NoteCmd>) {
        while let Some(Reverse(next)) = self.pending_off.peek().copied() {
            if next.off_tick > up_to_tick {
                break;
            }
            self.pending_off.pop();
            out_cmds.push(NoteCmd::NoteOff {
                note_id: next.note_id,
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

    fn extrapolate_gate_tick(grid: &ThetaGrid, target_gate: u64) -> Option<Tick> {
        let n = grid.boundaries.len();
        if n < 2 {
            return None;
        }
        let last = grid.boundaries[n - 1];
        let prev = grid.boundaries[n - 2];
        let gate_span = last.gate.saturating_sub(prev.gate);
        if gate_span == 0 {
            return None;
        }
        let ticks_per_gate = last.tick.saturating_sub(prev.tick) / gate_span;
        if ticks_per_gate == 0 {
            return None;
        }
        let extra_gates = target_gate.saturating_sub(last.gate);
        Some(
            last.tick
                .saturating_add(extra_gates.saturating_mul(ticks_per_gate)),
        )
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
        out_cmds: &mut Vec<NoteCmd>,
        out_events: &mut Vec<NoteOnEvent>,
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

    fn step_for_candidate(
        &self,
        c: &CandidatePoint,
        timing_field: &TimingField,
        prev_gate_exc: Option<f32>,
        fs: f32,
    ) -> CandidateStep {
        // dt_theta spec: same tick -> 0; negative/non-finite delta -> 0; NaN/Inf after cast -> 0.
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
        let mod_val = if c.phase_in_gate == 0.0 {
            1.0
        } else {
            self.sub_theta_mod.mod_at_phase(c.phase_in_gate)
        };
        CandidateStep {
            dt_theta,
            dt_sec,
            exc_gate,
            exc_slope,
            weight: exc_gate * mod_val,
        }
    }

    fn schedule_duration(
        &mut self,
        onset: OnsetContext,
        ctx: &CoreTickCtx,
        timing_grid: &mut ThetaGrid,
    ) {
        let plan = self.duration_rule.on_note_on(onset);
        match plan {
            DurationPlan::HoldTheta(hold_theta) => {
                self.schedule_hold_theta(onset, hold_theta, ctx, timing_grid);
            }
            DurationPlan::AtGate(off_gate) => {
                timing_grid.ensure_boundaries_until(ctx, off_gate.saturating_add(1));
                let off_tick = timing_grid
                    .boundaries
                    .iter()
                    .find(|b| b.gate >= off_gate)
                    .map(|b| b.tick)
                    .unwrap_or_else(|| {
                        Self::extrapolate_gate_tick(timing_grid, off_gate).unwrap_or(ctx.frame_end)
                    })
                    .max(onset.tick);
                self.schedule_note_off(onset.note_id, off_tick);
            }
            DurationPlan::None => {}
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
        out_cmds: &mut Vec<NoteCmd>,
        out_events: &mut Vec<NoteOnEvent>,
        out_onsets: &mut Vec<OnsetEvent>,
    ) {
        let mut prev_gate_exc: Option<f32> = None;
        self.drain_note_offs(ctx.now_tick, out_cmds);
        for c in candidates {
            debug_assert!(c.theta_pos.is_finite());
            self.drain_note_offs(c.tick, out_cmds);
            let step = self.step_for_candidate(c, timing_field, prev_gate_exc, ctx.fs);
            let allow_onset = min_allowed_onset_tick
                .map(|min_tick| c.tick >= min_tick)
                .unwrap_or(true);
            let input = IntervalInput {
                gate: c.gate,
                tick: c.tick,
                dt_theta: step.dt_theta,
                dt_sec: step.dt_sec,
                weight: step.weight,
            };
            if allow_onset && let Some(kick) = self.onset_rule.on_candidate(&input, state) {
                let note_id = self.next_note_id;
                self.next_note_id = self.next_note_id.wrapping_add(1);
                out_cmds.push(NoteCmd::NoteOn { note_id, kick });
                self.active_notes = self.active_notes.saturating_add(1);
                out_events.push(NoteOnEvent {
                    note_id,
                    onset_tick: c.tick,
                });
                out_onsets.push(OnsetEvent {
                    gate: c.gate,
                    onset_tick: c.tick,
                    strength: kick.strength(),
                });
                let onset = OnsetContext {
                    note_id,
                    tick: c.tick,
                    gate: c.gate,
                    theta_pos: c.theta_pos,
                    exc_gate: step.exc_gate,
                    exc_slope: step.exc_slope,
                };
                self.schedule_duration(onset, ctx, timing_grid);
            }
            self.last_gate_index = Some(c.gate);
            self.last_theta_pos = Some(c.theta_pos);
            self.last_tick = Some(c.tick);
            if c.phase_in_gate == 0.0 {
                prev_gate_exc = Some(step.exc_gate);
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
        let mut interval = OnsetRule::Accumulator {
            rate_hz: 0.0,
            refractory_gates: 0,
            acc: 0.0,
            next_allowed_gate: 0,
        };
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
        let mut interval = OnsetRule::Accumulator {
            rate_hz: 0.5,
            refractory_gates: 0,
            acc: 0.0,
            next_allowed_gate: 0,
        };
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
        let mut interval = OnsetRule::Accumulator {
            rate_hz: 1.0,
            refractory_gates: 1,
            acc: 0.0,
            next_allowed_gate: 0,
        };
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
        let mut interval = OnsetRule::Accumulator {
            rate_hz: 1.0,
            refractory_gates: 0,
            acc: 0.0,
            next_allowed_gate: 0,
        };
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
        let mut interval = OnsetRule::Accumulator {
            rate_hz: 1.0,
            refractory_gates: 0,
            acc: 0.0,
            next_allowed_gate: 0,
        };
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
        let mut interval = OnsetRule::Accumulator {
            rate_hz: 1.0,
            refractory_gates: 0,
            acc: 0.0,
            next_allowed_gate: 0,
        };
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
        let mut interval = OnsetRule::Accumulator {
            rate_hz: 1.0,
            refractory_gates: 0,
            acc: 0.0,
            next_allowed_gate: 0,
        };
        let state = CoreState { is_alive: true };
        let input = IntervalInput {
            gate: 0,
            tick: 0,
            dt_theta: 1.0,
            dt_sec: 1.0,
            weight: 1.0e9,
        };
        let _ = interval.on_candidate(&input, &state);
        let OnsetRule::Accumulator { acc, .. } = &interval else {
            panic!("expected Accumulator");
        };
        assert!(acc.is_finite());
        assert!(*acc <= 32.0);
    }

    #[test]
    fn fixed_gate_connect_returns_at_gate() {
        let mut connect = DurationRule::fixed_gate(0);
        let plan = connect.on_note_on(OnsetContext {
            note_id: 1,
            tick: 123,
            gate: 10,
            theta_pos: 10.0,
            exc_gate: 1.0,
            exc_slope: 0.0,
        });
        assert_eq!(plan, DurationPlan::AtGate(10));
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
        let interval = OnsetRule::Accumulator {
            rate_hz: 250.0,
            refractory_gates: 0,
            acc: 0.0,
            next_allowed_gate: 0,
        };
        let mut engine = PhonationEngine {
            clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
            onset_rule: interval,
            duration_rule: DurationRule::fixed_gate(1),
            sub_theta_mod: SubThetaMod::None,
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
        let log_c = log.clone();

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
            clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
            onset_rule: OnsetRule::Custom(Box::new(move |c, _| {
                log_c.lock().expect("dt log").push(c.dt_theta);
                None
            })),
            duration_rule: DurationRule::fixed_gate(1),
            sub_theta_mod: SubThetaMod::None,
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
        let log_c = log.clone();

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
            clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
            onset_rule: OnsetRule::Custom(Box::new({
                let log = log_c.clone();
                move |c, _| {
                    log.lock().expect("dt log").push(c.dt_theta);
                    None
                }
            })),
            duration_rule: DurationRule::fixed_gate(1),
            sub_theta_mod: SubThetaMod::None,
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
            clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
            onset_rule: OnsetRule::None,
            duration_rule: DurationRule::fixed_gate(1),
            sub_theta_mod: SubThetaMod::None,
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
        let log_c = log.clone();

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
            clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
            onset_rule: OnsetRule::Custom(Box::new({
                let log = log_c.clone();
                move |c, _| {
                    log.lock().expect("dt log").push(c.dt_theta);
                    None
                }
            })),
            duration_rule: DurationRule::fixed_gate(1),
            sub_theta_mod: SubThetaMod::None,
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
        let modulator = SubThetaMod::CosineHarmonic {
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
        let interval = OnsetRule::Accumulator {
            rate_hz: 50.0,
            refractory_gates: 0,
            acc: 0.0,
            next_allowed_gate: 0,
        };
        let mut engine = PhonationEngine {
            clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
            onset_rule: interval,
            duration_rule: DurationRule::fixed_gate(1),
            sub_theta_mod: SubThetaMod::CosineHarmonic {
                n: 1,
                depth: 1.0,
                phase0: -std::f32::consts::PI,
            },
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
        let interval = OnsetRule::Accumulator {
            rate_hz: 1.0,
            refractory_gates: 0,
            acc: 1.0,
            next_allowed_gate: 0,
        };
        let mut engine = PhonationEngine {
            clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
            onset_rule: interval,
            duration_rule: DurationRule::fixed_gate(0),
            sub_theta_mod: SubThetaMod::None,
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
        assert!(cmds.iter().any(|cmd| matches!(cmd, NoteCmd::NoteOn { .. })));
        assert!(
            cmds.iter()
                .any(|cmd| matches!(cmd, NoteCmd::NoteOff { .. }))
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
        let interval = OnsetRule::Accumulator {
            rate_hz: 1.0,
            refractory_gates: 0,
            acc: 1.0,
            next_allowed_gate: 0,
        };
        let mut engine = PhonationEngine {
            clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
            onset_rule: interval,
            duration_rule: DurationRule::fixed_gate(1),
            sub_theta_mod: SubThetaMod::None,
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
        assert!(cmds.iter().any(|cmd| matches!(cmd, NoteCmd::NoteOn { .. })));
        assert!(
            !cmds
                .iter()
                .any(|cmd| matches!(cmd, NoteCmd::NoteOff { .. }))
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
            clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
            onset_rule: OnsetRule::None,
            duration_rule: DurationRule::fixed_gate(1),
            sub_theta_mod: SubThetaMod::None,
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
            NoteCmd::NoteOff {
                note_id: 42,
                off_tick: 5
            }
        )));
    }

    #[test]
    fn note_off_emits_before_note_on_when_due() {
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
            clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
            onset_rule: OnsetRule::Custom(Box::new(|_, _| {
                Some(OnsetKick::Planned { strength: 1.0 })
            })),
            duration_rule: DurationRule::Custom(Box::new(|_| DurationPlan::None)),
            sub_theta_mod: SubThetaMod::None,
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
            .position(|cmd| matches!(cmd, NoteCmd::NoteOff { note_id: 99, .. }));
        let on_pos = cmds
            .iter()
            .position(|cmd| matches!(cmd, NoteCmd::NoteOn { .. }));
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
            clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
            onset_rule: OnsetRule::None,
            duration_rule: DurationRule::fixed_gate(1),
            sub_theta_mod: SubThetaMod::None,
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
        let onset = OnsetContext {
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
            clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
            onset_rule: OnsetRule::None,
            duration_rule: DurationRule::fixed_gate(1),
            sub_theta_mod: SubThetaMod::None,
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
        let onset = OnsetContext {
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
            onset: OnsetConfig::None,
            duration: DurationConfig::FixedGate { length_gates: 1 },
            clock: PhonationClockConfig::ThetaGate,
            sub_theta_mod: SubThetaModConfig::None,
            social: SocialConfig::default(),
        };
        let mut engine = PhonationEngine::from_config(&base, 7);
        let changed = PhonationConfig {
            onset: OnsetConfig::Accumulator {
                rate: 2.0,
                refractory: 2,
            },
            duration: DurationConfig::Field {
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
        let kick = engine.onset_rule.on_candidate(
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
        let plan = engine.duration_rule.on_note_on(OnsetContext {
            note_id: 1,
            tick: 0,
            gate: 0,
            theta_pos: 0.0,
            exc_gate: 0.2,
            exc_slope: 0.1,
        });
        assert!(
            matches!(plan, DurationPlan::HoldTheta(_)),
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
        let interval = OnsetRule::Accumulator {
            rate_hz: 250.0,
            refractory_gates: 0,
            acc: 0.0,
            next_allowed_gate: 0,
        };
        let stub_ticks: Vec<Tick> = vec![0, 4, 8];
        let mut stub_index = 0usize;
        let mut engine = PhonationEngine {
            clock: PhonationClock::Custom(Box::new(
                move |ctx: &CoreTickCtx, out: &mut Vec<CandidatePoint>| {
                    while stub_index < stub_ticks.len() {
                        let tick = stub_ticks[stub_index];
                        if tick < ctx.now_tick || tick >= ctx.frame_end {
                            return;
                        }
                        let gate = stub_index as u64;
                        out.push(CandidatePoint {
                            tick,
                            gate,
                            theta_pos: gate as f64,
                            phase_in_gate: 0.0,
                            sources: vec![ClockSource::GateBoundary],
                        });
                        stub_index += 1;
                    }
                },
            )),
            onset_rule: interval,
            duration_rule: DurationRule::fixed_gate(1),
            sub_theta_mod: SubThetaMod::None,
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
            NoteCmd::NoteOff { note_id, off_tick } if *note_id == first.note_id => Some(*off_tick),
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
        let mut connect = DurationRule::field(0.25, 1.0, 10.0, 0.5, 0.0);
        let low_plan = connect.on_note_on(OnsetContext {
            note_id: 1,
            tick: 0,
            gate: 0,
            theta_pos: 0.0,
            exc_gate: 0.0,
            exc_slope: 0.0,
        });
        let high_plan = connect.on_note_on(OnsetContext {
            note_id: 2,
            tick: 0,
            gate: 0,
            theta_pos: 0.0,
            exc_gate: 1.0,
            exc_slope: 0.0,
        });
        let hold_low = match low_plan {
            DurationPlan::HoldTheta(hold) => hold,
            _ => panic!("expected hold plan for low excitation"),
        };
        let hold_high = match high_plan {
            DurationPlan::HoldTheta(hold) => hold,
            _ => panic!("expected hold plan for high excitation"),
        };
        assert!(hold_high > hold_low);
    }

    #[test]
    fn field_connect_schedules_off_tick_when_next_boundary_not_in_candidates() {
        let onset_tick: Tick = 0;

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
            clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
            onset_rule: OnsetRule::Custom(Box::new(move |c, _| {
                if c.tick == onset_tick {
                    Some(OnsetKick::Planned { strength: 1.0 })
                } else {
                    None
                }
            })),
            duration_rule: DurationRule::field(0.5, 0.5, 10.0, 0.5, 0.0),
            sub_theta_mod: SubThetaMod::None,
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
            NoteCmd::NoteOff {
                off_tick,
                ..
            } if *off_tick == expected_off_tick
        )));
    }

    #[test]
    fn field_connect_off_tick_independent_of_subdivision() {
        let onset_tick: Tick = 0;

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
            clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
            onset_rule: OnsetRule::Custom(Box::new(move |c, _| {
                if c.tick == onset_tick {
                    Some(OnsetKick::Planned { strength: 1.0 })
                } else {
                    None
                }
            })),
            duration_rule: DurationRule::field(0.5, 0.5, 10.0, 0.5, 0.0),
            sub_theta_mod: SubThetaMod::None,
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
            NoteCmd::NoteOn { note_id, .. } => Some(*note_id),
            _ => None,
        });
        let off_tick_base = cmds.iter().find_map(|cmd| match cmd {
            NoteCmd::NoteOff { note_id, off_tick } if Some(*note_id) == note_on_id => {
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
            clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
            onset_rule: OnsetRule::Custom(Box::new(move |c, _| {
                if c.tick == onset_tick {
                    Some(OnsetKick::Planned { strength: 1.0 })
                } else {
                    None
                }
            })),
            duration_rule: DurationRule::field(0.5, 0.5, 10.0, 0.5, 0.0),
            sub_theta_mod: SubThetaMod::None,
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
            NoteCmd::NoteOn { note_id, .. } => Some(*note_id),
            _ => None,
        });
        let off_tick_sub = cmds_sub.iter().find_map(|cmd| match cmd {
            NoteCmd::NoteOff { note_id, off_tick } if Some(*note_id) == note_on_id_sub => {
                Some(*off_tick)
            }
            _ => None,
        });

        assert_eq!(off_tick_base, Some(50));
        assert_eq!(off_tick_sub, Some(50));
    }

    #[test]
    fn field_connect_note_off_survives_empty_hop() {
        let onset_tick: Tick = 0;

        let timing_field = TimingField::from_values(0, vec![1.0, 1.0]);
        let state = CoreState { is_alive: true };
        let mut engine = PhonationEngine {
            clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
            onset_rule: OnsetRule::Custom(Box::new(move |c, _| {
                if c.tick == onset_tick {
                    Some(OnsetKick::Planned { strength: 1.0 })
                } else {
                    None
                }
            })),
            duration_rule: DurationRule::field(1.0, 1.0, 10.0, 0.5, 0.0),
            sub_theta_mod: SubThetaMod::None,
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
            NoteCmd::NoteOn { note_id, .. } => Some(*note_id),
            _ => None,
        });
        assert!(note_on_id.is_some());
        assert!(
            !cmds
                .iter()
                .any(|cmd| matches!(cmd, NoteCmd::NoteOff { .. }))
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
            NoteCmd::NoteOff { note_id, off_tick } if Some(*note_id) == note_on_id => {
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

        let fire_ticks: HashSet<Tick> = [0, 10, 20].into_iter().collect();

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
            clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
            onset_rule: OnsetRule::Custom(Box::new({
                let ticks = fire_ticks.clone();
                move |c, _| {
                    if ticks.contains(&c.tick) {
                        Some(OnsetKick::Planned { strength: 1.0 })
                    } else {
                        None
                    }
                }
            })),
            duration_rule: DurationRule::Custom(Box::new({
                let slopes = slopes_base.clone();
                move |onset| {
                    slopes.lock().expect("slopes").push(onset.exc_slope);
                    DurationPlan::None
                }
            })),
            sub_theta_mod: SubThetaMod::None,
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
            clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
            onset_rule: OnsetRule::Custom(Box::new({
                let ticks = fire_ticks.clone();
                move |c, _| {
                    if ticks.contains(&c.tick) {
                        Some(OnsetKick::Planned { strength: 1.0 })
                    } else {
                        None
                    }
                }
            })),
            duration_rule: DurationRule::Custom(Box::new({
                let slopes = slopes_sub.clone();
                move |onset| {
                    slopes.lock().expect("slopes").push(onset.exc_slope);
                    DurationPlan::None
                }
            })),
            sub_theta_mod: SubThetaMod::None,
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
