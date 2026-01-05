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
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OnsetKind {
    Birth,
}

#[derive(Debug, Clone, Copy)]
pub struct CandidateTick {
    pub t: Tick,
    pub theta_gate: i64,
    pub dt_theta: f32,
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
    gate_index: i64,
}

impl ThetaGateClock {
    pub fn gate_index_hint(&self) -> i64 {
        if self.gate_index < 0 {
            0
        } else {
            self.gate_index + 1
        }
    }
}

impl PhonationClock for ThetaGateClock {
    fn gather_candidates(&mut self, ctx: &CoreTickCtx, out: &mut Vec<CandidateTick>) {
        let gate_tick = next_gate_tick(ctx.now_tick, ctx.fs, ctx.rhythms.theta, 0.0);
        let gate_tick = match gate_tick {
            Some(tick) => tick,
            None => return,
        };
        if gate_tick < ctx.now_tick || gate_tick >= ctx.frame_end {
            return;
        }
        if self.last_gate_tick == Some(gate_tick) {
            return;
        }
        self.gate_index = self.gate_index.saturating_add(1);
        self.last_gate_tick = Some(gate_tick);
        out.push(CandidateTick {
            t: gate_tick,
            theta_gate: self.gate_index,
            dt_theta: 1.0,
        });
    }
}

pub trait PhonationInterval: Send {
    fn on_candidate(&mut self, c: &CandidateTick, state: &CoreState) -> Option<PhonationKick>;
    fn on_external_onset(&mut self, _kind: OnsetKind, _at_theta_gate: i64) {}
}

#[derive(Debug, Default)]
pub struct NoneInterval;

impl PhonationInterval for NoneInterval {
    fn on_candidate(&mut self, _c: &CandidateTick, _state: &CoreState) -> Option<PhonationKick> {
        None
    }
}

#[derive(Debug)]
pub struct AccumulatorInterval {
    pub rate: f32,
    pub refractory_gates: u32,
    pub acc: f32,
    pub next_allowed_gate: i64,
}

impl AccumulatorInterval {
    pub fn new(rate: f32, refractory_gates: u32, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let acc = rng.random::<f32>().clamp(0.0, 1.0);
        Self {
            rate,
            refractory_gates,
            acc,
            next_allowed_gate: i64::MIN,
        }
    }
}

impl PhonationInterval for AccumulatorInterval {
    fn on_candidate(&mut self, c: &CandidateTick, _state: &CoreState) -> Option<PhonationKick> {
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
        self.acc += rate_eff * c.dt_theta;
        let refractory_ok = c.theta_gate >= self.next_allowed_gate;
        if self.acc >= 1.0 && refractory_ok {
            self.acc -= 1.0;
            self.next_allowed_gate = c.theta_gate + self.refractory_gates as i64 + 1;
            return Some(PhonationKick::Planned { strength: 1.0 });
        }
        None
    }

    fn on_external_onset(&mut self, _kind: OnsetKind, at_theta_gate: i64) {
        self.next_allowed_gate = at_theta_gate + self.refractory_gates as i64 + 1;
    }
}

pub trait PhonationConnect: Send {
    fn on_note_on(&mut self, note_id: NoteId, at_gate: i64);
    fn poll(&mut self, now_gate: i64, out: &mut Vec<PhonationCmd>);
}

#[derive(Debug)]
pub struct FixedGateConnect {
    pub length_gates: u32,
    pending: Vec<(NoteId, i64)>,
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
    fn on_note_on(&mut self, note_id: NoteId, at_gate: i64) {
        let off_gate = at_gate + self.length_gates as i64;
        self.pending.push((note_id, off_gate));
    }

    fn poll(&mut self, now_gate: i64, out: &mut Vec<PhonationCmd>) {
        let mut idx = 0;
        while idx < self.pending.len() {
            if self.pending[idx].1 <= now_gate {
                let note_id = self.pending[idx].0;
                self.pending.swap_remove(idx);
                out.push(PhonationCmd::NoteOff { note_id });
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
    last_gate_index: i64,
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
            last_gate_index: -1,
            active_notes: 0,
        }
    }

    pub fn next_gate_index_hint(&self) -> i64 {
        if self.last_gate_index < 0 {
            0
        } else {
            self.last_gate_index + 1
        }
    }

    pub fn notify_birth_onset(&mut self, at_theta_gate: i64) {
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
        let mut birth_applied = false;
        for c in candidates {
            if !birth_applied && birth_onset_tick.is_some_and(|tick| tick <= c.t) {
                self.notify_birth_onset(c.theta_gate);
                birth_applied = true;
            }
            if let Some(kick) = self.interval.on_candidate(&c, state) {
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
            self.last_gate_index = c.theta_gate;
        }
        let before = out_cmds.len();
        self.connect.poll(self.last_gate_index, out_cmds);
        for cmd in &out_cmds[before..] {
            if matches!(cmd, PhonationCmd::NoteOff { .. }) {
                self.active_notes = self.active_notes.saturating_sub(1);
            }
        }
        birth_applied
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate_at_gate(gate: i64) -> CandidateTick {
        CandidateTick {
            t: gate as u64,
            theta_gate: gate,
            dt_theta: 1.0,
        }
    }

    #[test]
    fn accumulator_rate_zero_never_fires() {
        let mut interval = AccumulatorInterval::new(0.0, 0, 1);
        interval.acc = 0.0;
        let state = CoreState { is_alive: true };
        for gate in 0..10 {
            let c = candidate_at_gate(gate);
            assert!(interval.on_candidate(&c, &state).is_none());
        }
    }

    #[test]
    fn accumulator_rate_half_fires_every_other_gate() {
        let mut interval = AccumulatorInterval::new(0.5, 0, 1);
        interval.acc = 0.0;
        let state = CoreState { is_alive: true };
        let mut fired = Vec::new();
        for gate in 0..10 {
            let c = candidate_at_gate(gate);
            if interval.on_candidate(&c, &state).is_some() {
                fired.push(gate);
            }
        }
        assert_eq!(fired, vec![1, 3, 5, 7, 9]);
    }

    #[test]
    fn accumulator_refractory_blocks_adjacent_gates() {
        let mut interval = AccumulatorInterval::new(1.0, 1, 1);
        interval.acc = 0.0;
        let state = CoreState { is_alive: true };
        let mut fired = Vec::new();
        for gate in 0..6 {
            let c = candidate_at_gate(gate);
            if interval.on_candidate(&c, &state).is_some() {
                fired.push(gate);
            }
        }
        assert_eq!(fired, vec![1, 3, 5]);
    }

    #[test]
    fn birth_onset_applies_refractory() {
        let mut interval = AccumulatorInterval::new(1.0, 1, 1);
        interval.acc = 1.0;
        interval.on_external_onset(OnsetKind::Birth, 0);
        let state = CoreState { is_alive: true };
        let c = candidate_at_gate(0);
        assert!(interval.on_candidate(&c, &state).is_none());
    }
}
