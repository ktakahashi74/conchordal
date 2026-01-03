use std::collections::BTreeMap;

use crate::core::timebase::Tick;
use crate::life::intent::BodySnapshot;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GateTarget {
    Next,
    NextNext,
}

#[derive(Clone, Copy, Debug)]
pub struct PhaseRef {
    pub gate: GateTarget,
    pub target_phase: f32,
}

#[derive(Clone, Debug)]
pub struct PlannedIntent {
    pub source_id: u64,
    pub plan_id: u64,
    pub phase: PhaseRef,
    pub duration: Tick,
    pub freq_hz: f32,
    pub amp: f32,
    pub tag: Option<String>,
    pub confidence: f32,
    pub body: Option<BodySnapshot>,
}

pub struct PlanBoard {
    next: BTreeMap<u64, PlannedIntent>,
}

impl PlanBoard {
    pub fn new() -> Self {
        Self {
            next: BTreeMap::new(),
        }
    }

    /// v0 keeps only GateTarget::Next; NextNext is ignored.
    pub fn publish_replace(&mut self, p: PlannedIntent) {
        if p.phase.gate == GateTarget::Next {
            self.next.insert(p.source_id, p);
        }
    }

    pub fn remove_source(&mut self, source_id: u64) {
        self.next.remove(&source_id);
    }

    /// Returns GateTarget::Next in stable source_id order.
    pub fn snapshot_next(&self) -> Vec<PlannedIntent> {
        self.next.values().cloned().collect()
    }

    pub fn clear_next(&mut self) {
        self.next.clear();
    }
}

impl Default for PlanBoard {
    fn default() -> Self {
        Self::new()
    }
}
