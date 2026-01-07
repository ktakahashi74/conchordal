use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::f32::consts::TAU;
use std::fmt;

use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::Tick;
use crate::life::gate_clock::next_gate_tick;
use crate::life::scenario::{
    PhonationClockConfig, PhonationConfig, PhonationConnectConfig, PhonationIntervalConfig,
    SubThetaModConfig,
};
use crate::life::social_density::SocialDensityTrace;

pub type NoteId = u64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhonationKick {
    Birth,
    Planned { strength: f32 },
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OnsetKind {
    Birth,
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
            }
            boundaries.push(GateBoundary {
                gate: candidate.gate,
                tick: candidate.tick,
            });
            last_gate = Some(candidate.gate);
        }
        Self { boundaries }
    }

    pub fn tick_at(&self, gate: u64, phase_in_gate: f32) -> Option<Tick> {
        if self.boundaries.is_empty() {
            return None;
        }
        let start_gate = self.boundaries[0].gate;
        let idx = gate.checked_sub(start_gate)? as usize;
        let next_idx = idx.checked_add(1)?;
        let b0 = *self.boundaries.get(idx)?;
        let b1 = *self.boundaries.get(next_idx)?;
        if b0.gate != gate || b1.gate != gate + 1 {
            return None;
        }
        let phase = phase_in_gate.clamp(0.0, 1.0);
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
            let mut weight = (rhythms.env_open * rhythms.env_level).clamp(0.0, 1.0);
            if let Some((trace, coupling)) = social {
                if coupling != 0.0 {
                    let density = trace.density_at(boundary.tick);
                    if density.is_finite() {
                        let arg = (coupling * density).clamp(-10.0, 10.0);
                        weight = (weight * arg.exp()).max(0.0);
                        if !weight.is_finite() {
                            weight = 0.0;
                        }
                    }
                }
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
        for boundary in &grid.boundaries[..grid.boundaries.len() - 1] {
            let gate = boundary.gate;
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
        for boundary in &grid.boundaries[..grid.boundaries.len() - 1] {
            let gate = boundary.gate;
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
    fn on_external_onset(&mut self, _kind: OnsetKind, _at_theta_gate: u64) {}
}

pub trait SubThetaMod: Send + Sync {
    fn mod_at_phase(&self, phase_in_gate: f32) -> f32;
}

#[derive(Debug, Default)]
pub struct NoneMod;

impl SubThetaMod for NoneMod {
    fn mod_at_phase(&self, _phase_in_gate: f32) -> f32 {
        1.0
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
}

#[derive(Debug, Default)]
pub struct NoneInterval;

impl PhonationInterval for NoneInterval {
    fn on_candidate(&mut self, _c: &IntervalInput, _state: &CoreState) -> Option<PhonationKick> {
        None
    }
}

#[derive(Debug)]
pub struct AccumulatorInterval {
    pub rate: f32,
    pub refractory_gates: u32,
    pub acc: f32,
    pub next_allowed_gate: u64,
}

impl AccumulatorInterval {
    pub fn new(rate: f32, refractory_gates: u32, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let acc = rng.random::<f32>().clamp(0.0, 1.0);
        Self {
            rate,
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
        if !self.rate.is_finite() || self.rate <= 0.0 {
            return None;
        }
        let max_rate = 1.0 / (self.refractory_gates as f32 + 1.0);
        let rate_eff = if max_rate.is_finite() && max_rate > 0.0 {
            self.rate.min(max_rate)
        } else {
            self.rate
        };
        let weight = c.weight.max(0.0);
        self.acc += rate_eff * weight * c.dt_theta;
        if !self.acc.is_finite() {
            self.acc = ACC_MAX;
        } else if self.acc > ACC_MAX {
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

    fn on_external_onset(&mut self, _kind: OnsetKind, at_theta_gate: u64) {
        self.next_allowed_gate = at_theta_gate + self.refractory_gates as u64 + 1;
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

pub trait PhonationConnect: Send {
    fn on_note_on(&mut self, onset: ConnectOnset);
    fn poll(&mut self, now: ConnectNow, out: &mut Vec<PhonationCmd>);
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
    fn on_note_on(&mut self, onset: ConnectOnset) {
        let off_gate = onset.gate + self.length_gates as u64;
        self.pending.push((onset.note_id, off_gate));
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
}

#[derive(Debug)]
pub struct FieldConnect {
    pub hold_min_theta: f32,
    pub hold_max_theta: f32,
    pub curve_k: f32,
    pub curve_x0: f32,
    pub drop_gain: f32,
    pending: Vec<(NoteId, f64)>,
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
            pending: Vec::new(),
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
    fn on_note_on(&mut self, onset: ConnectOnset) {
        let exc_gate = onset.exc_gate.clamp(0.0, 1.0);
        let hold = self.hold_theta(exc_gate, onset.exc_slope);
        let off_theta = onset.theta_pos + hold as f64;
        self.pending.push((onset.note_id, off_theta));
    }

    fn poll(&mut self, now: ConnectNow, out: &mut Vec<PhonationCmd>) {
        let mut idx = 0;
        while idx < self.pending.len() {
            if self.pending[idx].1 <= now.theta_pos {
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
}

#[derive(Debug, Clone, Copy)]
pub struct PhonationNoteEvent {
    pub note_id: NoteId,
    pub onset_tick: Tick,
}

#[derive(Debug, Clone, Copy)]
pub struct OnsetEvent {
    pub gate: u64,
    pub onset_tick: Tick,
    pub strength: f32,
}

pub struct PhonationEngine {
    pub clock: Box<dyn PhonationClock + Send>,
    pub interval: Box<dyn PhonationInterval + Send>,
    pub connect: Box<dyn PhonationConnect + Send>,
    pub sub_theta_mod: Box<dyn SubThetaMod + Send + Sync>,
    pub next_note_id: NoteId,
    last_gate_index: Option<u64>,
    last_theta_pos: Option<f64>,
    active_notes: u32,
}

impl fmt::Debug for PhonationEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PhonationEngine")
            .field("next_note_id", &self.next_note_id)
            .field("last_gate_index", &self.last_gate_index)
            .field("active_notes", &self.active_notes)
            .finish()
    }
}

impl PhonationEngine {
    pub fn from_config(config: &PhonationConfig, seed: u64) -> Self {
        let interval: Box<dyn PhonationInterval + Send> = match config.interval {
            PhonationIntervalConfig::None => Box::<NoneInterval>::default(),
            PhonationIntervalConfig::Accumulator { rate, refractory } => {
                Box::new(AccumulatorInterval::new(rate, refractory, seed))
            }
        };
        let sub_theta_mod: Box<dyn SubThetaMod + Send + Sync> = match &config.sub_theta_mod {
            SubThetaModConfig::None => Box::<NoneMod>::default(),
            SubThetaModConfig::Cosine { n, depth, phase0 } => Box::new(CosineHarmonicMod {
                n: *n,
                depth: *depth,
                phase0: *phase0,
            }),
        };
        let connect: Box<dyn PhonationConnect + Send> = match config.connect {
            PhonationConnectConfig::FixedGate { length_gates } => {
                Box::new(FixedGateConnect::new(length_gates))
            }
            PhonationConnectConfig::Field {
                hold_min_theta,
                hold_max_theta,
                curve_k,
                curve_x0,
                drop_gain,
            } => Box::new(FieldConnect::new(
                hold_min_theta,
                hold_max_theta,
                curve_k,
                curve_x0,
                drop_gain,
            )),
        };
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
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            active_notes: 0,
        }
    }

    pub fn next_gate_index_hint(&self) -> u64 {
        self.last_gate_index
            .map_or(0, |gate| gate.saturating_add(1))
    }

    pub fn notify_birth_onset(&mut self, at_theta_gate: u64) {
        self.interval
            .on_external_onset(OnsetKind::Birth, at_theta_gate);
    }

    pub fn has_active_notes(&self) -> bool {
        self.active_notes > 0
    }

    pub fn tick(
        &mut self,
        ctx: &CoreTickCtx,
        state: &CoreState,
        birth_onset_tick: Option<Tick>,
        social: Option<&SocialDensityTrace>,
        social_coupling: f32,
        out_cmds: &mut Vec<PhonationCmd>,
        out_events: &mut Vec<PhonationNoteEvent>,
        out_onsets: &mut Vec<OnsetEvent>,
    ) -> bool {
        let mut candidates = Vec::new();
        self.clock.gather_candidates(ctx, &mut candidates);
        let merged = Self::merge_candidates(candidates);
        let timing_grid = ThetaGrid::from_candidates(&merged);
        let timing_field = TimingField::build_from(
            ctx,
            &timing_grid,
            social.map(|trace| (trace, social_coupling)),
        );
        self.process_candidates(
            &merged,
            &timing_field,
            state,
            birth_onset_tick,
            out_cmds,
            out_events,
            out_onsets,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn process_candidates(
        &mut self,
        candidates: &[CandidatePoint],
        timing_field: &TimingField,
        state: &CoreState,
        birth_onset_tick: Option<Tick>,
        out_cmds: &mut Vec<PhonationCmd>,
        out_events: &mut Vec<PhonationNoteEvent>,
        out_onsets: &mut Vec<OnsetEvent>,
    ) -> bool {
        let mut birth_applied = false;
        let mut prev_gate_exc: Option<f32> = None;
        for c in candidates {
            debug_assert!(c.theta_pos.is_finite());
            if !birth_applied && birth_onset_tick.is_some_and(|tick| tick <= c.tick) {
                self.notify_birth_onset(c.gate);
                birth_applied = true;
            }
            let dt_theta = match self.last_theta_pos {
                Some(prev_pos) => {
                    let dt = c.theta_pos - prev_pos;
                    debug_assert!(
                        dt.is_finite() && dt >= 0.0,
                        "candidate theta_pos must be non-decreasing"
                    );
                    if dt < 0.0 { 0.0 } else { dt as f32 }
                }
                None => 1.0,
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
                weight: exc_gate * sub_theta_mod,
            };
            if let Some(kick) = self.interval.on_candidate(&input, state) {
                let note_id = self.next_note_id;
                self.next_note_id = self.next_note_id.wrapping_add(1);
                out_cmds.push(PhonationCmd::NoteOn { note_id, kick });
                self.active_notes = self.active_notes.saturating_add(1);
                out_events.push(PhonationNoteEvent {
                    note_id,
                    onset_tick: c.tick,
                });
                let strength = match kick {
                    PhonationKick::Birth => 1.0,
                    PhonationKick::Planned { strength } => strength,
                };
                out_onsets.push(OnsetEvent {
                    gate: c.gate,
                    onset_tick: c.tick,
                    strength,
                });
                self.connect.on_note_on(ConnectOnset {
                    note_id,
                    tick: c.tick,
                    gate: c.gate,
                    theta_pos: c.theta_pos,
                    exc_gate,
                    exc_slope,
                });
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
            if c.phase_in_gate == 0.0 {
                prev_gate_exc = Some(exc_gate);
            }
        }
        birth_applied
    }

    fn merge_candidates(mut candidates: Vec<CandidatePoint>) -> Vec<CandidatePoint> {
        candidates.sort_by(|a, b| {
            a.tick.cmp(&b.tick).then_with(|| {
                a.theta_pos
                    .partial_cmp(&b.theta_pos)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        let mut merged: Vec<CandidatePoint> = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if let Some(last) = merged.last_mut()
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
            merged.push(candidate);
        }
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
                weight: 1.0,
            };
            if interval.on_candidate(&input, &state).is_some() {
                fired.push(gate);
            }
        }
        assert_eq!(fired, vec![1u64, 3, 5]);
    }

    #[test]
    fn birth_onset_applies_refractory() {
        let mut interval = AccumulatorInterval::new(1.0, 1, 1);
        interval.acc = 1.0;
        interval.on_external_onset(OnsetKind::Birth, 0);
        let state = CoreState { is_alive: true };
        let c = candidate_at_gate(0);
        let input = IntervalInput {
            gate: c.gate,
            tick: c.tick,
            dt_theta: 1.0,
            weight: 1.0,
        };
        assert!(interval.on_candidate(&input, &state).is_none());
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
                weight: 0.5,
            },
            IntervalInput {
                gate: 1,
                tick: 1,
                dt_theta: 1.0,
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
            weight: 0.0,
        };
        assert!(interval.on_candidate(&input, &state).is_none());
    }

    #[test]
    fn accumulator_respects_dt_theta() {
        let mut interval = AccumulatorInterval::new(1.0, 0, 1);
        interval.acc = 0.0;
        let state = CoreState { is_alive: true };
        let input = IntervalInput {
            gate: 0,
            tick: 0,
            dt_theta: 0.5,
            weight: 1.0,
        };
        assert!(interval.on_candidate(&input, &state).is_none());
        let input = IntervalInput {
            gate: 1,
            tick: 1,
            dt_theta: 0.5,
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
        connect.on_note_on(ConnectOnset {
            note_id: 1,
            tick: 123,
            gate: 10,
            theta_pos: 10.0,
            exc_gate: 1.0,
            exc_slope: 0.0,
        });
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
        let mut interval = AccumulatorInterval::new(1.0, 0, 1);
        interval.acc = 0.0;
        let mut engine = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::new(interval),
            connect: Box::new(FixedGateConnect::new(1)),
            sub_theta_mod: Box::<NoneMod>::default(),
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            active_notes: 0,
        };
        let state = CoreState { is_alive: true };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        engine.process_candidates(
            &candidates,
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
        let timing_field = TimingField::build_from(&ctx, &grid, None);
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
        let timing_field = TimingField::build_from(&ctx, &grid, None);
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
        let timing_field = TimingField::from_values(0, vec![1.0, 1.0]);
        let mut engine = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::new(RecordingInterval { log: log.clone() }),
            connect: Box::new(FixedGateConnect::new(1)),
            sub_theta_mod: Box::<NoneMod>::default(),
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            active_notes: 0,
        };
        let state = CoreState { is_alive: true };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        engine.process_candidates(
            &candidates,
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
        let timing_field = TimingField::from_values(20_000_000, vec![1.0, 1.0]);
        let mut engine = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::new(RecordingInterval { log: log.clone() }),
            connect: Box::new(FixedGateConnect::new(1)),
            sub_theta_mod: Box::<NoneMod>::default(),
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            active_notes: 0,
        };
        let state = CoreState { is_alive: true };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        engine.process_candidates(
            &candidates,
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
        let timing_field = TimingField::from_values(0, vec![1.0, 1.0]);
        let mut interval = AccumulatorInterval::new(1.0, 0, 1);
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
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: Some(0.0),
            active_notes: 0,
        };
        let state = CoreState { is_alive: true };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        engine.process_candidates(
            &candidates,
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
        let mut engine = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::new(AccumulatorInterval::new(1.0, 0, 1)),
            connect: Box::new(FixedGateConnect::new(0)),
            sub_theta_mod: Box::<NoneMod>::default(),
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            active_notes: 0,
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
        engine.process_candidates(
            &candidates,
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
        let mut engine = PhonationEngine {
            clock: Box::<ThetaGateClock>::default(),
            interval: Box::new(AccumulatorInterval::new(1.0, 0, 1)),
            connect: Box::new(FixedGateConnect::new(1)),
            sub_theta_mod: Box::<NoneMod>::default(),
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            active_notes: 0,
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
        engine.process_candidates(
            &candidates,
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
        let mut interval = AccumulatorInterval::new(1.0, 0, 1);
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
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            active_notes: 0,
        };
        let state = CoreState { is_alive: true };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        engine.tick(
            &ctx,
            &state,
            None,
            None,
            0.0,
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
        let timing_field = TimingField::build_from(&ctx, &grid, Some((&trace, -1.0)));
        assert!(timing_field.e(0) < 0.5);
        assert!((timing_field.e(1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn field_connect_holds_longer_with_high_excitation() {
        let mut connect = FieldConnect::new(0.25, 1.0, 10.0, 0.5, 0.0);
        connect.on_note_on(ConnectOnset {
            note_id: 1,
            tick: 0,
            gate: 0,
            theta_pos: 0.0,
            exc_gate: 0.0,
            exc_slope: 0.0,
        });
        let mut out = Vec::new();
        connect.poll(
            ConnectNow {
                tick: 2,
                gate: 0,
                theta_pos: 0.2,
            },
            &mut out,
        );
        assert!(out.is_empty());
        connect.poll(
            ConnectNow {
                tick: 3,
                gate: 0,
                theta_pos: 0.3,
            },
            &mut out,
        );
        assert!(
            out.iter()
                .any(|cmd| matches!(cmd, PhonationCmd::NoteOff { note_id: 1, .. }))
        );
        let mut connect = FieldConnect::new(0.25, 1.0, 10.0, 0.5, 0.0);
        connect.on_note_on(ConnectOnset {
            note_id: 2,
            tick: 0,
            gate: 0,
            theta_pos: 0.0,
            exc_gate: 1.0,
            exc_slope: 0.0,
        });
        let mut out_high = Vec::new();
        connect.poll(
            ConnectNow {
                tick: 3,
                gate: 0,
                theta_pos: 0.3,
            },
            &mut out_high,
        );
        assert!(out_high.is_empty());
        connect.poll(
            ConnectNow {
                tick: 10,
                gate: 1,
                theta_pos: 1.0,
            },
            &mut out_high,
        );
        let off_tick = out_high.iter().find_map(|cmd| match cmd {
            PhonationCmd::NoteOff {
                note_id: 2,
                off_tick,
            } => Some(*off_tick),
            _ => None,
        });
        assert_eq!(off_tick, Some(10));
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
            fn on_note_on(&mut self, onset: ConnectOnset) {
                self.slopes.lock().expect("slopes").push(onset.exc_slope);
            }

            fn poll(&mut self, _now: ConnectNow, _out: &mut Vec<PhonationCmd>) {}
        }

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
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            active_notes: 0,
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
        engine_base.process_candidates(
            &candidates_base,
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
            next_note_id: 0,
            last_gate_index: None,
            last_theta_pos: None,
            active_notes: 0,
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
        engine_sub.process_candidates(
            &candidates_sub,
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
        let timing_field = TimingField::build_from(&ctx, &grid, Some((&trace, 1000.0)));
        assert!(timing_field.e(0).is_finite());
    }
}
