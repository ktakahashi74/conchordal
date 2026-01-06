use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::fmt;

use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::Tick;
use crate::life::gate_clock::next_gate_tick;
use crate::life::scenario::{PhonationConfig, PhonationConnectConfig, PhonationIntervalConfig};

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

#[derive(Debug, Clone, Copy)]
pub struct CandidateTick {
    pub t: Tick,
    pub theta_gate: u64,
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
    pub fn from_candidates(candidates: &[CandidateTick]) -> Self {
        let mut boundaries = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            boundaries.push(GateBoundary {
                gate: candidate.theta_gate,
                tick: candidate.t,
            });
        }
        Self { boundaries }
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

    pub fn build_from(ctx: &CoreTickCtx, grid: &ThetaGrid) -> Self {
        if grid.boundaries.is_empty() {
            return Self {
                start_gate: 0,
                e_gate: Vec::new(),
            };
        }
        let mut rhythms = ctx.rhythms;
        let mut cursor_tick = ctx.now_tick;
        let mut e_gate = Vec::with_capacity(grid.boundaries.len());
        for boundary in &grid.boundaries {
            if boundary.tick > cursor_tick {
                let dt_sec = (boundary.tick - cursor_tick) as f32 / ctx.fs;
                rhythms.advance_in_place(dt_sec);
                cursor_tick = boundary.tick;
            }
            let weight = (rhythms.env_open * rhythms.env_level).clamp(0.0, 1.0);
            e_gate.push(weight);
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
    fn gather_candidates(&mut self, ctx: &CoreTickCtx, out: &mut Vec<CandidateTick>);
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
        out: &mut Vec<CandidateTick>,
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
            out.push(CandidateTick {
                t: gate_tick,
                theta_gate: self.gate_index,
            });
            cursor = gate_tick.saturating_add(1);
        }
    }
}

impl PhonationClock for ThetaGateClock {
    fn gather_candidates(&mut self, ctx: &CoreTickCtx, out: &mut Vec<CandidateTick>) {
        self.gather_candidates_with(ctx, out, |cursor, ctx| {
            next_gate_tick(cursor, ctx.fs, ctx.rhythms.theta, 0.0)
        });
    }
}

pub trait PhonationInterval: Send {
    fn on_candidate(&mut self, c: &IntervalInput, state: &CoreState) -> Option<PhonationKick>;
    fn on_external_onset(&mut self, _kind: OnsetKind, _at_theta_gate: u64) {}
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
        let weight = c.weight.clamp(0.0, 1.0);
        self.acc += rate_eff * weight * c.dt_theta;
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

pub trait PhonationConnect: Send {
    fn on_note_on(&mut self, note_id: NoteId, at_gate: u64);
    fn poll(&mut self, now_gate: u64, now_tick: Tick, out: &mut Vec<PhonationCmd>);
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
    fn on_note_on(&mut self, note_id: NoteId, at_gate: u64) {
        let off_gate = at_gate + self.length_gates as u64;
        self.pending.push((note_id, off_gate));
    }

    fn poll(&mut self, now_gate: u64, now_tick: Tick, out: &mut Vec<PhonationCmd>) {
        let mut idx = 0;
        while idx < self.pending.len() {
            if self.pending[idx].1 <= now_gate {
                let note_id = self.pending[idx].0;
                self.pending.swap_remove(idx);
                out.push(PhonationCmd::NoteOff {
                    note_id,
                    off_tick: now_tick,
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

pub struct PhonationEngine {
    pub clock: Box<dyn PhonationClock + Send>,
    pub interval: Box<dyn PhonationInterval + Send>,
    pub connect: Box<dyn PhonationConnect + Send>,
    pub next_note_id: NoteId,
    last_gate_index: Option<u64>,
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
        let connect: Box<dyn PhonationConnect + Send> = match config.connect {
            PhonationConnectConfig::FixedGate { length_gates } => {
                Box::new(FixedGateConnect::new(length_gates))
            }
        };
        Self {
            clock: Box::<ThetaGateClock>::default(),
            interval,
            connect,
            next_note_id: 0,
            last_gate_index: None,
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
        out_cmds: &mut Vec<PhonationCmd>,
        out_events: &mut Vec<PhonationNoteEvent>,
    ) -> bool {
        let mut candidates = Vec::new();
        self.clock.gather_candidates(ctx, &mut candidates);
        candidates.sort_by_key(|c| c.t);
        let timing_grid = ThetaGrid::from_candidates(&candidates);
        let timing_field = TimingField::build_from(ctx, &timing_grid);
        let mut birth_applied = false;
        let mut prev_gate = None;
        for c in candidates {
            if !birth_applied && birth_onset_tick.is_some_and(|tick| tick <= c.t) {
                self.notify_birth_onset(c.theta_gate);
                birth_applied = true;
            }
            let dt_theta = prev_gate
                .map(|gate| c.theta_gate.saturating_sub(gate) as f32)
                .filter(|dt| *dt > 0.0)
                .unwrap_or(1.0);
            let input = IntervalInput {
                gate: c.theta_gate,
                tick: c.t,
                dt_theta,
                weight: timing_field.e(c.theta_gate),
            };
            if let Some(kick) = self.interval.on_candidate(&input, state) {
                let note_id = self.next_note_id;
                self.next_note_id = self.next_note_id.wrapping_add(1);
                self.active_notes = self.active_notes.saturating_add(1);
                out_cmds.push(PhonationCmd::NoteOn { note_id, kick });
                out_events.push(PhonationNoteEvent {
                    note_id,
                    onset_tick: c.t,
                });
                self.connect.on_note_on(note_id, c.theta_gate);
            }
            self.last_gate_index = Some(c.theta_gate);
            let before = out_cmds.len();
            self.connect.poll(c.theta_gate, c.t, out_cmds);
            for cmd in &out_cmds[before..] {
                if matches!(cmd, PhonationCmd::NoteOff { .. }) {
                    self.active_notes = self.active_notes.saturating_sub(1);
                }
            }
            prev_gate = Some(c.theta_gate);
        }
        birth_applied
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate_at_gate(gate: u64) -> CandidateTick {
        CandidateTick {
            t: gate as u64,
            theta_gate: gate,
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
                gate: c.theta_gate,
                tick: c.t,
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
                gate: c.theta_gate,
                tick: c.t,
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
                gate: c.theta_gate,
                tick: c.t,
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
            gate: c.theta_gate,
            tick: c.t,
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
    fn fixed_gate_connect_sets_off_tick() {
        let mut connect = FixedGateConnect::new(0);
        let mut out = Vec::new();
        connect.on_note_on(1, 10);
        connect.poll(10, 123, &mut out);
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
        assert!(out[0].t < out[1].t);
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
        assert_eq!(out[0].t, 0);
    }

    #[test]
    fn phonation_engine_uses_timing_field_weight() {
        let mut rhythms = NeuralRhythms::default();
        rhythms.env_open = 0.0;
        rhythms.env_level = 1.0;
        rhythms.delta.phase = std::f32::consts::PI;
        rhythms.delta.freq_hz = 125.0;
        rhythms.delta.mag = 1.0;
        rhythms.delta.alpha = 1.0;
        let ctx = CoreTickCtx {
            now_tick: 0,
            frame_end: 8,
            fs: 1000.0,
            rhythms,
        };
        struct StubClock {
            ticks: Vec<Tick>,
            index: usize,
        }

        impl PhonationClock for StubClock {
            fn gather_candidates(&mut self, ctx: &CoreTickCtx, out: &mut Vec<CandidateTick>) {
                while self.index < self.ticks.len() {
                    let tick = self.ticks[self.index];
                    if tick < ctx.now_tick || tick >= ctx.frame_end {
                        return;
                    }
                    let gate = self.index as u64;
                    out.push(CandidateTick {
                        t: tick,
                        theta_gate: gate,
                    });
                    self.index += 1;
                }
            }
        }

        let mut interval = AccumulatorInterval::new(1.0, 0, 1);
        interval.acc = 0.0;
        let mut engine = PhonationEngine {
            clock: Box::new(StubClock {
                ticks: vec![0, 4],
                index: 0,
            }),
            interval: Box::new(interval),
            connect: Box::new(FixedGateConnect::new(1)),
            next_note_id: 0,
            last_gate_index: None,
            active_notes: 0,
        };
        let state = CoreState { is_alive: true };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        engine.tick(&ctx, &state, None, &mut cmds, &mut events);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].onset_tick, 4);
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
            fn gather_candidates(&mut self, ctx: &CoreTickCtx, out: &mut Vec<CandidateTick>) {
                while self.index < self.ticks.len() {
                    let tick = self.ticks[self.index];
                    if tick < ctx.now_tick || tick >= ctx.frame_end {
                        return;
                    }
                    let gate = self.index as u64;
                    out.push(CandidateTick {
                        t: tick,
                        theta_gate: gate,
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
            next_note_id: 0,
            last_gate_index: None,
            active_notes: 0,
        };
        let state = CoreState { is_alive: true };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        engine.tick(&ctx, &state, None, &mut cmds, &mut events);
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
}
