use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::f32::consts::TAU;
use std::fmt;

use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::Tick;
use crate::life::control_adapters::phonation_config_from_spec;
use crate::life::gate_clock::next_gate_tick;
use crate::life::social_density::SocialDensityTrace;
use crate::scenario::{
    DurationConfig, OnsetConfig, PhonationClockConfig, PhonationConfig, PhonationMode,
    PhonationSpec,
};

pub type ToneId = u64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnsetKick {
    pub strength: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ToneUpdate {
    pub target_freq_hz: Option<f32>,
    /// Final output target amp (linear).
    pub target_amp: Option<f32>,
    pub continuous_drive: Option<f32>,
}

impl ToneUpdate {
    pub fn is_empty(&self) -> bool {
        self.target_freq_hz.is_none()
            && self.target_amp.is_none()
            && self.continuous_drive.is_none()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
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
            if let Some(prev_gate) = last_gate {
                if candidate.gate < prev_gate {
                    panic!("candidate gates must be non-decreasing");
                }
                if candidate.gate == prev_gate || candidate.gate != prev_gate.saturating_add(1) {
                    continue;
                }
                if last_tick.is_some_and(|t| candidate.tick < t) {
                    continue;
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

    /// Extend the gate grid up to `target_gate_plus_1`. `fixed_period`, when set,
    /// extends at a constant tick spacing (the coupling clock's nominal onset
    /// period) so a note-off lands on the same grid that generated the onset.
    /// When `None`, the grid follows the adaptive theta (a `ThetaGate` clock).
    /// Using theta to extend a coupling-clock voice would make its note length
    /// drift with theta.
    pub fn ensure_boundaries_until(
        &mut self,
        ctx: &CoreTickCtx,
        target_gate_plus_1: u64,
        fixed_period: Option<f64>,
    ) {
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
            let mut next_tick = match fixed_period {
                Some(period) => {
                    let step = (period.round() as Tick).max(1);
                    cursor_tick.saturating_add(step)
                }
                None => {
                    let search_tick = cursor_tick.saturating_add(1);
                    match next_gate_tick(search_tick, ctx.fs, ctx.rhythms.theta, 0.0) {
                        Some(tick) => tick,
                        None => break,
                    }
                }
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

    /// Uniform field (weight 1.0 at every gate), ignoring `env_open`/social.
    /// Used by fixed-rate clocks so their onsets are not gated by the adaptive
    /// theta envelope.
    pub fn flat(grid: &ThetaGrid) -> Self {
        let Some(start_gate) = grid.boundaries.first().map(|b| b.gate) else {
            return Self {
                start_gate: 0,
                e_gate: Vec::new(),
            };
        };
        let last_gate = grid.boundaries.last().map(|b| b.gate).unwrap_or(start_gate);
        let len = (last_gate.saturating_sub(start_gate) as usize).saturating_add(1);
        Self {
            start_gate,
            e_gate: vec![1.0; len],
        }
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
    fn gather_candidates_core(
        &mut self,
        ctx: &CoreTickCtx,
        out: &mut Vec<CandidatePoint>,
        mut next_gate_tick_fn: impl FnMut(Tick, &CoreTickCtx) -> Option<Tick>,
    ) {
        let mut cursor = ctx.now_tick;
        while cursor < ctx.frame_end {
            let gate_tick = match next_gate_tick_fn(cursor, ctx) {
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
            });
            cursor = gate_tick.saturating_add(1);
        }
    }
}

impl ThetaGateClock {
    fn gather_candidates_impl(&mut self, ctx: &CoreTickCtx, out: &mut Vec<CandidatePoint>) {
        self.gather_candidates_core(ctx, out, |cursor, ctx| {
            next_gate_tick(cursor, ctx.fs, ctx.rhythms.theta, 0.0)
        });
    }
}

/// Upper bound on a clock's effective onset rate. Caps candidate density so a
/// pathological base/meter rate cannot collapse the period to ~1 tick and emit
/// a candidate at every sample.
const MAX_ONSET_RATE_HZ: f32 = 200.0;

/// Strength of the phase pull toward the shared meter beat. Kept at 1.0 so the
/// effective-frequency factor `1 + lock * PULL_K * err` stays in `[0.5, 1.5]`
/// (with `lock` in `[0,1]` and `err` in `[-0.5, 0.5]`) and the phase advances
/// monotonically. A backward phase jump would corrupt the in-hop integer-
/// crossing emission below.
const COUPLING_PULL_K: f32 = 1.0;

/// Per-voice phase oscillator that entrains its onset phase to the shared
/// production meter beat with a coupling strength `coupling`. This single
/// mechanism spans the rhythm family continuum: `coupling -> 0` is a free
/// renewal process (flow texture), medium coupling entrains loosely as the
/// meter gains confidence (entrained beat), and high coupling locks into the
/// shared beat as a deep attractor (metric beat). Onsets are emitted at integer
/// crossings of the oscillator phase; a phase pull toward the meter beat phase
/// drags those crossings into a common alignment, so independently placed
/// voices synchronize without any externally imposed grid.
pub struct CouplingClock {
    coupling: f32,
    base_rate_hz: f32,
    flow_depth: f32,
    microtiming: f32,
    /// Monotonic oscillator phase in cycles.
    phase: f64,
    /// Intrinsic onset rate for the current interval, redrawn from the renewal
    /// distribution at each onset (constant when `flow_depth == 0`).
    intrinsic_rate_hz: f32,
    /// Last effective rate, used to estimate a nominal period for duration grid
    /// extrapolation.
    last_f_eff_hz: f32,
    cluster_remaining: u8,
    rng: SmallRng,
    gate_index: u64,
    has_gate: bool,
}

impl CouplingClock {
    fn new(coupling: f32, base_rate_hz: f32, flow_depth: f32, microtiming: f32, seed: u64) -> Self {
        let base_rate_hz = if base_rate_hz.is_finite() {
            base_rate_hz.clamp(0.01, MAX_ONSET_RATE_HZ)
        } else {
            1.0
        };
        let flow_depth = flow_depth.clamp(0.0, 1.0);
        let mut rng = SmallRng::seed_from_u64(seed ^ 0xC0D1_4E55_7A1C_0911);
        let mut cluster_remaining = 0u8;
        let first_ioi =
            OnsetRule::flow_next_ioi(base_rate_hz, flow_depth, &mut cluster_remaining, &mut rng);
        // Random initial phase so free-running voices placed together do not all
        // fire on the same tick before the meter entrains them.
        let phase: f64 = rng.random_range(0.0..1.0);
        Self {
            coupling: coupling.clamp(0.0, 1.0),
            base_rate_hz,
            flow_depth,
            microtiming: if microtiming.is_finite() {
                microtiming
            } else {
                0.0
            },
            phase,
            intrinsic_rate_hz: (1.0 / first_ioi).clamp(0.01, MAX_ONSET_RATE_HZ),
            last_f_eff_hz: base_rate_hz,
            cluster_remaining,
            rng,
            gate_index: 0,
            has_gate: false,
        }
    }

    fn nominal_period_ticks(&self, fs: f32) -> Option<f64> {
        if fs.is_finite() && fs > 0.0 {
            Some((fs as f64 / self.last_f_eff_hz.max(0.01) as f64).max(1.0))
        } else {
            None
        }
    }

    fn gather_candidates_impl(&mut self, ctx: &CoreTickCtx, out: &mut Vec<CandidatePoint>) {
        let fs = ctx.fs;
        if !fs.is_finite() || fs <= 0.0 || ctx.frame_end <= ctx.now_tick {
            return;
        }
        let conf = ctx.rhythms.delta.alpha.clamp(0.0, 1.0);
        let lock = (self.coupling * conf).clamp(0.0, 1.0);
        let f_band = {
            let f = ctx.rhythms.delta.freq_hz;
            if f.is_finite() && f > 0.0 { f } else { 2.0 }
        };
        // Meter beat phase in cycles, [0,1); microtiming offsets the lock target.
        let target_frac = ((ctx.rhythms.delta.phase / TAU) + self.microtiming).rem_euclid(1.0);

        let frame_end = ctx.frame_end as f64;
        let mut cursor = ctx.now_tick as f64;
        // Defensive bound on in-hop crossings (degenerate rates only).
        let mut guard = 0u32;
        loop {
            guard += 1;
            if guard > 100_000 {
                break;
            }
            let f_intrinsic = self.intrinsic_rate_hz.max(0.01);
            let mut f_eff = (1.0 - lock) * f_intrinsic + lock * f_band;
            if !f_eff.is_finite() {
                f_eff = f_intrinsic;
            }
            f_eff = f_eff.clamp(0.01, MAX_ONSET_RATE_HZ);
            self.last_f_eff_hz = f_eff;

            let my_frac = self.phase.rem_euclid(1.0) as f32;
            let raw_err = target_frac - my_frac;
            let err = raw_err - raw_err.round(); // signed phase error in [-0.5, 0.5]
            let mut phase_dot = f_eff * (1.0 + lock * COUPLING_PULL_K * err);
            if !phase_dot.is_finite() || phase_dot <= 0.0 {
                phase_dot = f_eff.max(0.01);
            }

            // Cycles to the next integer crossing (a full cycle when on a beat).
            let frac_to_next = 1.0 - my_frac as f64;
            let dt_cross_sec = frac_to_next / phase_dot as f64;
            let cross_tick = cursor + dt_cross_sec * fs as f64;
            if !cross_tick.is_finite() || cross_tick >= frame_end {
                let remaining_sec = (frame_end - cursor) / fs as f64;
                if remaining_sec > 0.0 {
                    self.phase += phase_dot as f64 * remaining_sec;
                }
                break;
            }

            let adv_sec = (cross_tick - cursor) / fs as f64;
            self.phase += phase_dot as f64 * adv_sec;
            // Land exactly on the integer to stop floating drift accumulating.
            self.phase = self.phase.round();

            let tick = (cross_tick.round() as Tick).clamp(ctx.now_tick, ctx.frame_end - 1);
            if self.has_gate {
                self.gate_index = self.gate_index.saturating_add(1);
            } else {
                self.has_gate = true;
            }
            out.push(CandidatePoint {
                tick,
                gate: self.gate_index,
            });

            // Renewal: redraw the intrinsic rate for the next interval.
            let ioi = OnsetRule::flow_next_ioi(
                self.base_rate_hz,
                self.flow_depth,
                &mut self.cluster_remaining,
                &mut self.rng,
            );
            self.intrinsic_rate_hz = (1.0 / ioi).clamp(0.01, MAX_ONSET_RATE_HZ);
            cursor = cross_tick;
        }
    }
}

pub enum PhonationClock {
    ThetaGate(ThetaGateClock),
    Coupling(CouplingClock),
    #[cfg(test)]
    Custom(Box<dyn FnMut(&CoreTickCtx, &mut Vec<CandidatePoint>) + Send>),
}

impl PhonationClock {
    pub fn gather_candidates(&mut self, ctx: &CoreTickCtx, out: &mut Vec<CandidatePoint>) {
        match self {
            PhonationClock::ThetaGate(clock) => {
                clock.gather_candidates_impl(ctx, out);
            }
            PhonationClock::Coupling(clock) => {
                clock.gather_candidates_impl(ctx, out);
            }
            #[cfg(test)]
            PhonationClock::Custom(f) => f(ctx, out),
        }
    }

    /// Whether onsets should bypass the adaptive `env_open`/social gating and use
    /// a flat timing field. The coupling clock owns its own timing (an internal
    /// entrained oscillator); gating it by theta would double-count the
    /// entrainment and destroy the renewal / lock continuum it produces.
    fn uses_flat_field(&self) -> bool {
        matches!(self, PhonationClock::Coupling(_))
    }

    /// Nominal tick spacing used to extend the duration grid without falling back
    /// to the adaptive theta. The last effective period for `Coupling`. `None`
    /// for theta-driven clocks, whose spacing is not constant.
    fn fixed_period_ticks(&self, fs: f32) -> Option<f64> {
        match self {
            PhonationClock::Coupling(clock) => clock.nominal_period_ticks(fs),
            _ => None,
        }
    }

    pub(crate) fn from_config(config: &PhonationClockConfig, seed: u64) -> Self {
        match config {
            PhonationClockConfig::ThetaGate => PhonationClock::ThetaGate(ThetaGateClock::default()),
            PhonationClockConfig::Coupling {
                coupling,
                base_rate_hz,
                flow_depth,
                microtiming,
            } => PhonationClock::Coupling(CouplingClock::new(
                *coupling,
                *base_rate_hz,
                *flow_depth,
                *microtiming,
                seed,
            )),
        }
    }
}

pub enum OnsetRule {
    None,
    /// Fire on every live candidate, each onset carrying `strength` (1.0 = plain
    /// beat, > 1.0 = accent). The coupling clock emits candidates exactly at
    /// onset times, so no further timing decision is needed here.
    Always {
        strength: f32,
    },
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

    /// Renewal inter-onset interval used by the coupling clock to shape the
    /// free-running (low-coupling) limit: a regular period when `depth == 0` and
    /// a clustered, gappy non-metric texture as `depth` rises.
    fn flow_next_ioi(
        mean_rate_hz: f32,
        depth: f32,
        cluster_remaining: &mut u8,
        rng: &mut SmallRng,
    ) -> f32 {
        let mean_rate_hz = if mean_rate_hz.is_finite() {
            mean_rate_hz.max(0.01)
        } else {
            1.0
        };
        let base_ioi = 1.0 / mean_rate_hz;
        let depth = depth.clamp(0.0, 1.0);

        if depth <= f32::EPSILON {
            *cluster_remaining = 0;
            return base_ioi.clamp(0.035, 4.0);
        }

        let mult = if *cluster_remaining > 0 {
            *cluster_remaining = cluster_remaining.saturating_sub(1);
            rng.random_range(0.30..0.62)
        } else {
            let p_cluster = 0.05 + 0.17 * depth;
            let p_gap = 0.04 + 0.10 * depth;
            let draw = rng.random_range(0.0..1.0);
            if draw < p_cluster {
                *cluster_remaining = rng.random_range(1..=2);
                rng.random_range(0.34..0.72)
            } else if draw < p_cluster + p_gap {
                rng.random_range(1.18..1.72)
            } else {
                let local = rng.random_range(-1.0..1.0) + rng.random_range(-1.0..1.0);
                (0.22 * depth * local).exp().clamp(0.58, 1.58)
            }
        };
        (base_ioi * mult).clamp(0.035, 4.0)
    }

    pub fn on_candidate(&mut self, c: &IntervalInput, state: &CoreState) -> Option<OnsetKick> {
        match self {
            OnsetRule::None => None,
            OnsetRule::Always { strength } => {
                if !state.is_alive || c.weight <= 0.0 {
                    return None;
                }
                Some(OnsetKick {
                    strength: *strength,
                })
            }
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
                    return Some(OnsetKick { strength: 1.0 });
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
            OnsetConfig::Always { strength } => OnsetRule::Always {
                strength: *strength,
            },
            OnsetConfig::Accumulator { rate, refractory } => {
                OnsetRule::accumulator(*rate, *refractory, seed)
            }
        }
    }

    fn update_config(&mut self, config: &OnsetConfig, seed: u64) {
        match (config, &mut *self) {
            (
                OnsetConfig::Accumulator { rate, refractory },
                OnsetRule::Accumulator {
                    rate_hz,
                    refractory_gates,
                    ..
                },
            ) => {
                *rate_hz = *rate;
                *refractory_gates = *refractory;
            }
            _ => {
                *self = Self::from_config(config, seed);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OnsetContext {
    pub tone_id: ToneId,
    pub tick: Tick,
    pub gate: u64,
    pub exc_gate: f32,
    pub exc_slope: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DurationPlan {
    None,
    HoldTheta(f32),
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
                DurationPlan::HoldTheta(*length_gates as f32)
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
pub struct ToneOnEvent {
    pub tone_id: ToneId,
    pub onset_tick: Tick,
}

#[derive(Debug, Clone, Copy)]
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

pub struct PhonationEngine {
    pub clock: PhonationClock,
    pub onset_rule: OnsetRule,
    pub duration_rule: DurationRule,
    pub mode: PhonationMode,
    hold: HoldCore,
    initial_seed: u64,
    pub next_tone_id: ToneId,
    last_tick: Option<Tick>,
    active_notes: u32,
    pending_off: BinaryHeap<Reverse<PendingOff>>,
    scratch_candidates: Vec<CandidatePoint>,
    scratch_merged: Vec<CandidatePoint>,
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
    pub fn from_spec(spec: &PhonationSpec, seed: u64) -> Self {
        let config = phonation_config_from_spec(spec);
        Self::from_config(&config, seed)
    }

    pub fn update_from_spec(&mut self, spec: &PhonationSpec) {
        let config = phonation_config_from_spec(spec);
        self.update_from_config(&config);
    }

    pub(crate) fn from_config(config: &PhonationConfig, seed: u64) -> Self {
        let onset_rule = OnsetRule::from_config(&config.onset, seed);
        let duration_rule = DurationRule::from_config(&config.duration);
        Self {
            clock: PhonationClock::from_config(&config.clock, seed),
            onset_rule,
            duration_rule,
            mode: config.mode,
            hold: HoldCore::default(),
            initial_seed: seed,
            next_tone_id: 0,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),

            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
        }
    }

    pub(crate) fn update_from_config(&mut self, config: &PhonationConfig) {
        self.mode = config.mode;
        let seed = self.initial_seed ^ 0xA5A5_5A5A_5A5A_A5A5;
        self.onset_rule.update_config(&config.onset, seed);
        self.duration_rule.update_config(&config.duration);
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
        let allow_onset = min_allowed_onset_tick
            .map(|min_tick| ctx.now_tick >= min_tick)
            .unwrap_or(true);
        if allow_onset && state.is_alive && !self.hold.note_on_sent {
            let tone_id = self.next_tone_id;
            self.next_tone_id = self.next_tone_id.wrapping_add(1);
            self.hold.note_on_sent = true;
            self.hold.tone_id = Some(tone_id);
            out_cmds.push(ToneCmd::On {
                tone_id,
                kick: OnsetKick { strength: 1.0 },
            });
            self.active_notes = self.active_notes.saturating_add(1);
            out_events.push(ToneOnEvent {
                tone_id,
                onset_tick: ctx.now_tick,
            });
            out_onsets.push(OnsetEvent {
                gate: 0,
                onset_tick: ctx.now_tick,
                strength: 1.0,
            });
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

    fn schedule_note_off(&mut self, tone_id: ToneId, off_tick: Tick) {
        self.pending_off
            .push(Reverse(PendingOff { off_tick, tone_id }));
    }

    fn drain_note_offs(&mut self, up_to_tick: Tick, out_cmds: &mut Vec<ToneCmd>) {
        while let Some(Reverse(next)) = self.pending_off.peek().copied() {
            if next.off_tick > up_to_tick {
                break;
            }
            self.pending_off.pop();
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
    ) {
        let hold_theta = if hold_theta.is_finite() {
            hold_theta.max(0.0)
        } else {
            0.0
        };
        let off_pos = onset.gate as f64 + hold_theta as f64;
        let off_tick_opt = Self::tick_for_gate_pos(off_pos, ctx, timing_grid, fixed_period);
        let fallback = timing_grid
            .boundaries
            .last()
            .map(|boundary| boundary.tick)
            .unwrap_or(onset.tick);
        let off_tick = off_tick_opt.unwrap_or(fallback).max(onset.tick);
        self.schedule_note_off(onset.tone_id, off_tick);
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
        // Fixed-rate and coupling clocks own their own timing (an external grid
        // or an internal entrained oscillator): their onsets must not be gated by
        // the adaptive `env_open`, or they would re-couple to theta (metric loses
        // its steady pulse; flow droplets lock to one period; the coupling clock's
        // entrainment is double-counted). Use a flat field.
        let timing_field = if self.clock.uses_flat_field() {
            TimingField::flat(&timing_grid)
        } else {
            TimingField::build_from(
                ctx,
                &timing_grid,
                social.map(|trace| (trace, social_coupling)),
                extra_gate_gain,
            )
        };
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
    ) {
        let plan = self.duration_rule.on_note_on(onset);
        match plan {
            DurationPlan::HoldTheta(hold_theta) => {
                self.schedule_hold_theta(onset, hold_theta, ctx, timing_grid, fixed_period);
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
        out_cmds: &mut Vec<ToneCmd>,
        out_events: &mut Vec<ToneOnEvent>,
        out_onsets: &mut Vec<OnsetEvent>,
    ) {
        let mut prev_gate_exc: Option<f32> = None;
        let fixed_period = self.clock.fixed_period_ticks(ctx.fs);
        self.drain_note_offs(ctx.now_tick, out_cmds);
        for c in candidates {
            self.drain_note_offs(c.tick, out_cmds);
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
            if allow_onset && let Some(kick) = self.onset_rule.on_candidate(&input, state) {
                let tone_id = self.next_tone_id;
                self.next_tone_id = self.next_tone_id.wrapping_add(1);
                out_cmds.push(ToneCmd::On { tone_id, kick });
                self.active_notes = self.active_notes.saturating_add(1);
                out_events.push(ToneOnEvent {
                    tone_id,
                    onset_tick: c.tick,
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
                self.schedule_duration(onset, ctx, timing_grid, fixed_period);
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
        candidates.sort_by(|a, b| a.tick.cmp(&b.tick).then_with(|| a.gate.cmp(&b.gate)));
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
mod tests {
    use super::*;
    use std::collections::BinaryHeap;

    fn candidate_at_gate(gate: u64) -> CandidatePoint {
        CandidatePoint { tick: gate, gate }
    }

    fn test_engine(onset_rule: OnsetRule, duration_rule: DurationRule) -> PhonationEngine {
        PhonationEngine {
            clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
            onset_rule,
            duration_rule,
            mode: PhonationMode::Gated,
            hold: HoldCore::default(),
            initial_seed: 0,
            next_tone_id: 0,
            last_tick: None,
            active_notes: 0,
            pending_off: BinaryHeap::new(),
            scratch_candidates: Vec::new(),
            scratch_merged: Vec::new(),
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
                dt_sec: 1.0,
                weight: 0.5,
            },
            IntervalInput {
                gate: 1,
                tick: 1,
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
            dt_sec: 0.5,
            weight: 1.0,
        };
        assert!(interval.on_candidate(&input, &state).is_none());
        let input = IntervalInput {
            gate: 1,
            tick: 1,
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
    fn fixed_gate_connect_returns_hold_theta() {
        let mut connect = DurationRule::fixed_gate(0);
        let plan = connect.on_note_on(OnsetContext {
            tone_id: 1,
            tick: 123,
            gate: 10,
            exc_gate: 1.0,
            exc_slope: 0.0,
        });
        assert_eq!(plan, DurationPlan::HoldTheta(0.0));
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
        clock.gather_candidates_core(&ctx, &mut out, |_cursor, _ctx| {
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
        clock.gather_candidates_core(&ctx, &mut out, |cursor, _ctx| {
            if cursor == 0 { Some(cursor) } else { None }
        });
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].tick, 0);
    }

    #[test]
    fn coupling_clock_free_runs_at_base_rate_with_zero_confidence() {
        // No metrical confidence -> lock collapses to 0 and the clock free-runs
        // at its intrinsic base rate (renewal/flow limit). depth = 0 keeps the
        // period regular so the assay is deterministic.
        let mut clock = CouplingClock::new(0.95, 10.0, 0.0, 0.0, 1);
        let rhythms = NeuralRhythms::default(); // delta.alpha (confidence) = 0
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 350,
            fs: 1000.0,
            rhythms,
        };
        let mut out = Vec::new();
        clock.gather_candidates_impl(&ctx, &mut out);
        // base_rate 10 Hz @ fs 1000 -> 100-tick period; a few onsets per hop,
        // never one-per-tick.
        assert!(out.len() >= 2 && out.len() <= 5, "got {}", out.len());
        for pair in out.windows(2) {
            assert!(pair[1].tick - pair[0].tick >= 80);
        }
    }

    #[test]
    fn coupling_clock_clamps_extreme_base_rate() {
        // A pathological base rate must not collapse the period to ~1 tick.
        let mut clock = CouplingClock::new(0.0, 1.0e9, 0.0, 0.0, 1);
        let rhythms = NeuralRhythms::default();
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 1000,
            fs: 48_000.0,
            rhythms,
        };
        let mut out = Vec::new();
        clock.gather_candidates_impl(&ctx, &mut out);
        // At the MAX_ONSET_RATE_HZ cap, fs/200 = 240-tick period -> only a few
        // onsets in a 1000-tick hop, never one-per-tick.
        assert!(
            out.len() <= 1000 / 200 + 2,
            "clamped clock emitted too many onsets: {}",
            out.len()
        );
        for pair in out.windows(2) {
            assert!(pair[1].tick - pair[0].tick >= 200);
        }
    }

    #[test]
    fn coupling_clock_high_coupling_locks_to_beat_band() {
        // With strong confidence and high coupling, the effective rate is pulled
        // toward the shared beat (delta) band, not the intrinsic base rate.
        let mut rhythms = NeuralRhythms::default();
        rhythms.delta.alpha = 1.0; // full confidence
        rhythms.delta.freq_hz = 2.0; // shared beat at 2 Hz
        let mut clock = CouplingClock::new(1.0, 10.0, 0.0, 0.0, 7);
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 4000,
            fs: 1000.0,
            rhythms,
        };
        let mut out = Vec::new();
        clock.gather_candidates_impl(&ctx, &mut out);
        // Locked to 2 Hz (500-tick period), the free-run base of 10 Hz would have
        // produced ~40 onsets; locking yields far fewer.
        assert!(
            out.len() <= 12,
            "expected beat-locked onsets near 2 Hz, got {}",
            out.len()
        );
        for pair in out.windows(2) {
            assert!(pair[1].tick - pair[0].tick >= 300);
        }
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
            CandidatePoint { tick: 0, gate: 0 },
            CandidatePoint { tick: 4, gate: 1 },
        ];
        let timing_field = TimingField::from_values(0, vec![0.0, 1.0]);
        let interval = OnsetRule::Accumulator {
            rate_hz: 250.0,
            refractory_gates: 0,
            acc: 0.0,
            next_allowed_gate: 0,
        };
        let mut engine = test_engine(interval, DurationRule::fixed_gate(2));
        let state = CoreState { is_alive: true };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let mut timing_grid = ThetaGrid {
            boundaries: vec![
                GateBoundary { gate: 0, tick: 0 },
                GateBoundary { gate: 1, tick: 10 },
                GateBoundary { gate: 2, tick: 20 },
                GateBoundary { gate: 3, tick: 30 },
            ],
        };
        engine.process_candidates(
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
            CandidatePoint { tick: 0, gate: 0 },
            CandidatePoint { tick: 1, gate: 0 },
            CandidatePoint { tick: 2, gate: 1 },
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
            CandidatePoint { tick: 0, gate: 1 },
            CandidatePoint { tick: 1, gate: 0 },
        ];
        let _ = ThetaGrid::from_candidates(&candidates);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn theta_grid_panics_on_unsorted_ticks() {
        let candidates = vec![
            CandidatePoint { tick: 2, gate: 0 },
            CandidatePoint { tick: 1, gate: 1 },
        ];
        let _ = ThetaGrid::from_candidates(&candidates);
    }

    #[test]
    fn theta_grid_skips_noncontiguous_gates() {
        let candidates = vec![
            CandidatePoint { tick: 0, gate: 0 },
            CandidatePoint { tick: 10, gate: 2 },
        ];
        let grid = ThetaGrid::from_candidates(&candidates);
        assert_eq!(
            grid.boundaries.len(),
            1,
            "non-contiguous gate 2 should be skipped"
        );
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
        grid.ensure_boundaries_until(&ctx, 1, None);
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
        grid.ensure_boundaries_until(&ctx, 4, None);
        assert!(grid.boundaries.len() >= 5);
        for pair in grid.boundaries.windows(2) {
            assert_eq!(pair[1].gate, pair[0].gate + 1);
            assert!(pair[1].tick > pair[0].tick);
        }
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
        let mut engine = test_engine(interval, DurationRule::fixed_gate(0));
        let state = CoreState { is_alive: true };
        let timing_field = TimingField::from_values(0, vec![1.0]);
        let candidates = vec![CandidatePoint { tick: 0, gate: 0 }];
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&candidates);
        engine.process_candidates(
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
        assert!(cmds.iter().any(|cmd| matches!(cmd, ToneCmd::On { .. })));
        assert!(cmds.iter().any(|cmd| matches!(cmd, ToneCmd::Off { .. })));
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
        let mut engine = test_engine(interval, DurationRule::fixed_gate(2));
        let state = CoreState { is_alive: true };
        let timing_field = TimingField::from_values(0, vec![1.0]);
        let candidates = vec![CandidatePoint { tick: 0, gate: 0 }];
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let mut timing_grid = ThetaGrid {
            boundaries: vec![
                GateBoundary { gate: 0, tick: 0 },
                GateBoundary { gate: 1, tick: 10 },
                GateBoundary { gate: 2, tick: 20 },
                GateBoundary { gate: 3, tick: 30 },
            ],
        };
        engine.process_candidates(
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
        assert!(cmds.iter().any(|cmd| matches!(cmd, ToneCmd::On { .. })));
        assert!(!cmds.iter().any(|cmd| matches!(cmd, ToneCmd::Off { .. })));
    }

    #[test]
    fn pending_off_drains_without_candidates() {
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 10,
            fs: 1000.0,
            rhythms: NeuralRhythms::default(),
        };
        let mut engine = test_engine(OnsetRule::None, DurationRule::fixed_gate(1));
        engine.schedule_note_off(42, 5);
        let state = CoreState { is_alive: true };
        let timing_field = TimingField::from_values(0, Vec::new());
        let candidates: Vec<CandidatePoint> = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&candidates);
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        engine.process_candidates(
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
            ToneCmd::Off {
                tone_id: 42,
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
        let candidates = vec![CandidatePoint { tick: 0, gate: 0 }];
        let timing_field = TimingField::from_values(0, vec![1.0]);
        let mut engine = test_engine(
            OnsetRule::Custom(Box::new(|_, _| Some(OnsetKick { strength: 1.0 }))),
            DurationRule::Custom(Box::new(|_| DurationPlan::None)),
        );
        engine.schedule_note_off(99, 0);
        let state = CoreState { is_alive: true };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&candidates);
        engine.process_candidates(
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
            .position(|cmd| matches!(cmd, ToneCmd::Off { tone_id: 99, .. }));
        let on_pos = cmds
            .iter()
            .position(|cmd| matches!(cmd, ToneCmd::On { .. }));
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
        let mut engine = test_engine(OnsetRule::None, DurationRule::fixed_gate(1));
        let onset = OnsetContext {
            tone_id: 7,
            tick: 10,
            gate: 0,
            exc_gate: 0.0,
            exc_slope: 0.0,
        };
        let mut timing_grid = ThetaGrid {
            boundaries: Vec::new(),
        };
        engine.schedule_hold_theta(onset, 1.0, &ctx, &mut timing_grid, None);
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
        let mut engine = test_engine(OnsetRule::None, DurationRule::fixed_gate(1));
        let onset = OnsetContext {
            tone_id: 11,
            tick: 0,
            gate: 0,
            exc_gate: 0.0,
            exc_slope: 0.0,
        };
        let mut timing_grid = ThetaGrid {
            boundaries: vec![
                GateBoundary { gate: 0, tick: 0 },
                GateBoundary { gate: 1, tick: 1 },
            ],
        };
        engine.schedule_hold_theta(onset, 0.5, &ctx, &mut timing_grid, None);
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
            CandidatePoint { tick: 10, gate: 0 },
            CandidatePoint { tick: 10, gate: 1 },
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
                dt_sec: 1.0,
                weight: 1.0,
            },
            &state,
        );
        assert!(kick.is_some(), "expected accumulator interval to be active");
        let plan = engine.duration_rule.on_note_on(OnsetContext {
            tone_id: 1,
            tick: 0,
            gate: 0,
            exc_gate: 0.2,
            exc_slope: 0.1,
        });
        assert!(
            matches!(plan, DurationPlan::HoldTheta(_)),
            "expected field connect plan"
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
        let mut engine = test_engine(interval, DurationRule::fixed_gate(1));
        engine.clock = PhonationClock::Custom(Box::new(
            move |ctx: &CoreTickCtx, out: &mut Vec<CandidatePoint>| {
                while stub_index < stub_ticks.len() {
                    let tick = stub_ticks[stub_index];
                    if tick < ctx.now_tick || tick >= ctx.frame_end {
                        return;
                    }
                    let gate = stub_index as u64;
                    out.push(CandidatePoint { tick, gate });
                    stub_index += 1;
                }
            },
        ));
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
            ToneCmd::Off { tone_id, off_tick } if *tone_id == first.tone_id => Some(*off_tick),
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
            tone_id: 1,
            tick: 0,
            gate: 0,
            exc_gate: 0.0,
            exc_slope: 0.0,
        });
        let high_plan = connect.on_note_on(OnsetContext {
            tone_id: 2,
            tick: 0,
            gate: 0,
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
        let candidates = vec![CandidatePoint { tick: 0, gate: 0 }];
        let mut engine = test_engine(
            OnsetRule::Custom(Box::new(move |c, _| {
                if c.tick == onset_tick {
                    Some(OnsetKick { strength: 1.0 })
                } else {
                    None
                }
            })),
            DurationRule::field(0.5, 0.5, 10.0, 0.5, 0.0),
        );
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let mut timing_grid = ThetaGrid::from_candidates(&candidates);
        engine.process_candidates(
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
            ToneCmd::Off {
                off_tick,
                ..
            } if *off_tick == expected_off_tick
        )));
    }

    #[test]
    fn field_connect_note_off_survives_empty_hop() {
        let onset_tick: Tick = 0;

        let timing_field = TimingField::from_values(0, vec![1.0, 1.0]);
        let state = CoreState { is_alive: true };
        let mut engine = test_engine(
            OnsetRule::Custom(Box::new(move |c, _| {
                if c.tick == onset_tick {
                    Some(OnsetKick { strength: 1.0 })
                } else {
                    None
                }
            })),
            DurationRule::field(1.0, 1.0, 10.0, 0.5, 0.0),
        );

        let ctx1 = CoreTickCtx {
            now_tick: 0,
            frame_end: 50,
            fs: 1000.0,
            rhythms: NeuralRhythms::default(),
        };
        let candidates = vec![CandidatePoint { tick: 0, gate: 0 }];
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
        engine.process_candidates(
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
            ToneCmd::On { tone_id, .. } => Some(*tone_id),
            _ => None,
        });
        assert!(note_on_id.is_some());
        assert!(!cmds.iter().any(|cmd| matches!(cmd, ToneCmd::Off { .. })));

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
        engine.process_candidates(
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
            ToneCmd::Off { tone_id, off_tick } if Some(*tone_id) == note_on_id => Some(*off_tick),
            _ => None,
        });
        assert_eq!(off_tick, Some(100));
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

    #[test]
    fn flow_next_ioi_depth_zero_returns_base_interval() {
        let mut cluster_remaining = 2;
        let mut rng = SmallRng::seed_from_u64(7);

        let ioi = OnsetRule::flow_next_ioi(2.5, 0.0, &mut cluster_remaining, &mut rng);

        assert!((ioi - 0.4).abs() < f32::EPSILON);
        assert_eq!(cluster_remaining, 0);
    }

    #[test]
    fn flow_next_ioi_keeps_stationary_local_variation() {
        let mut cluster_remaining = 0;
        let mut rng = SmallRng::seed_from_u64(11);
        let mut iois = Vec::new();
        for _ in 0..96 {
            iois.push(OnsetRule::flow_next_ioi(
                2.5,
                0.9,
                &mut cluster_remaining,
                &mut rng,
            ));
        }

        let first = iois[..32].iter().sum::<f32>() / 32.0;
        let middle = iois[32..64].iter().sum::<f32>() / 32.0;
        let last = iois[64..].iter().sum::<f32>() / 32.0;
        let min = first.min(middle).min(last);
        let max = first.max(middle).max(last);
        assert!(max / min < 1.55, "block means drifted too far: {iois:?}");
        assert!(iois.iter().any(|ioi| *ioi < 0.28));
        assert!(iois.iter().any(|ioi| *ioi > 0.50));
    }
}
