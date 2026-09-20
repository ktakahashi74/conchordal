//! Bounded acoustic group slots, transactional admission, and observed retirement.

use super::*;
use crate::temporal_cognition::group::{Config, Group, Trajectory, assign_and_refresh};

#[derive(Clone, Copy, Debug)]
struct Slot {
    group: Group,
    dormant: bool,
    last_active_end: u64,
    inactive_samples: u64,
}

pub(super) struct Lifecycle {
    residual: Handle,
    hop: u64,
    retirement_samples: u64,
    inactive_energy_max: f64,
    last_end: u64,
    next_generation: u64,
    slots: [Option<Slot>; 7],
}

pub(super) struct Prepared {
    pub assignment: Assignment,
    source_end: u64,
    refreshed: [Option<Group>; 7],
    current: [Option<Trajectory>; 8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Rejection {
    Capacity,
    ParentUnavailable,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Admission {
    pub key: Key,
    pub children: [Option<Handle>; 2],
    pub rejection: Option<Rejection>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Retirement {
    ObservedInactivity,
    Capacity,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Retired {
    pub group: Group,
    pub reason: Retirement,
    pub last_active_end: u64,
    pub inactive_samples: u64,
}

pub(super) struct Update {
    pub assignment: Assignment,
    pub admissions: [Option<Admission>; 8],
    pub superseded: [Option<Group>; 7],
    pub retired: [Option<Retired>; 7],
}

impl Lifecycle {
    pub fn retained_handles(&self) -> [Option<Handle>; 7] {
        self.slots.map(|s| s.map(|s| s.group.handle))
    }

    pub fn eligible_handles(&self) -> [Option<Handle>; 7] {
        self.slots
            .map(|s| s.filter(|s| s.group.eligible).map(|s| s.group.handle))
    }

    pub fn new(
        bus: u8,
        epoch: u64,
        start: u64,
        hop: u64,
        retirement_samples: u64,
        inactive_energy_max: f64,
    ) -> Result<Self, &'static str> {
        if bus > 1
            || hop == 0
            || !start.is_multiple_of(hop)
            || retirement_samples == 0
            || !inactive_energy_max.is_finite()
            || inactive_energy_max < 0.0
        {
            return Err("invalid group lifecycle configuration");
        }
        Ok(Self {
            residual: Handle {
                bus,
                epoch,
                generation: 1,
            },
            hop,
            retirement_samples,
            inactive_energy_max,
            last_end: start,
            next_generation: 2,
            slots: [None; 7],
        })
    }

    pub fn prepare(
        &self,
        config: &Config,
        end: u64,
        current: &[Option<Trajectory>; 8],
    ) -> Result<Prepared, &'static str> {
        if config.bus != self.residual.bus
            || config.epoch != self.residual.epoch
            || config.hop != self.hop
            || end <= self.last_end
        {
            return Err("foreign or noncausal group preparation");
        }
        let mut refreshed = self.slots.map(|s| s.map(|s| s.group));
        let assignment = assign_and_refresh(config, end, current, &mut refreshed)?;
        Ok(Prepared {
            assignment,
            source_end: self.last_end,
            refreshed,
            current: *current,
        })
    }

    pub fn commit(
        &mut self,
        prepared: Prepared,
        group_energy: [Option<f64>; 7],
        accepted: &[Candidate],
    ) -> Result<Update, &'static str> {
        let end = prepared.assignment.end_sample;
        if prepared.source_end != self.last_end
            || end <= self.last_end
            || !end.is_multiple_of(self.hop)
            || prepared.assignment.group_handles != self.slots.map(|s| s.map(|s| s.group.handle))
            || accepted.len() > 8
        {
            return Err("stale group preparation or invalid admission capacity");
        }
        for (i, energy) in group_energy.iter().enumerate() {
            if energy.is_some_and(|e| {
                !e.is_finite()
                    || e < 0.0
                    || (e > 0.0 && self.slots[i].is_none_or(|s| !s.group.eligible))
            }) {
                return Err("invalid current group energy");
            }
        }
        for (i, candidate) in accepted.iter().enumerate() {
            if candidate.end_sample != end
                || accepted[..i].iter().any(|c| c.key.conflicts(candidate.key))
            {
                return Err("stale or conflicting accepted proposal");
            }
            let measured =
                Candidate::measure(candidate.key, &prepared.assignment, Some(candidate.score))?;
            if measured.is_none_or(|c| c.support != candidate.support || c.score != candidate.score)
            {
                return Err("proposal does not match immutable current assignment");
            }
        }
        let mut slots = self.slots;
        let mut next_generation = self.next_generation;
        let mut out = Update {
            assignment: prepared.assignment,
            admissions: [None; 8],
            superseded: [None; 7],
            retired: [None; 7],
        };
        let (mut retired, mut superseded) = (0, 0);
        for (i, slot) in slots.iter_mut().enumerate() {
            let Some(state) = slot else { continue };
            state.group = prepared.refreshed[i].unwrap();
            if let Some(energy) = group_energy[i] {
                if state.group.eligible && energy > self.inactive_energy_max {
                    state.dormant = false;
                    state.last_active_end = end;
                    state.inactive_samples = 0;
                } else {
                    state.dormant = true;
                    state.inactive_samples = state.inactive_samples.saturating_add(self.hop);
                }
            }
            if state.inactive_samples >= self.retirement_samples {
                out.retired[retired] = Some(Retired {
                    group: state.group,
                    reason: Retirement::ObservedInactivity,
                    last_active_end: state.last_active_end,
                    inactive_samples: state.inactive_samples,
                });
                retired += 1;
                *slot = None;
            }
        }
        for (index, candidate) in accepted.iter().enumerate() {
            let key = candidate.key;
            let mut admission = Admission {
                key,
                children: [None; 2],
                rejection: None,
            };
            if key.parents.iter().flatten().any(|p| {
                !slots
                    .iter()
                    .flatten()
                    .any(|s| s.group.handle == *p && s.group.eligible)
            }) {
                admission.rejection = Some(Rejection::ParentUnavailable);
                out.admissions[index] = Some(admission);
                continue;
            }
            let needed = if key.kind == Kind::Split { 2 } else { 1 };
            let available = slots
                .iter()
                .filter(|slot| {
                    slot.is_none_or(|s| s.dormant || key.parents.contains(&Some(s.group.handle)))
                })
                .count();
            if available < needed {
                admission.rejection = Some(Rejection::Capacity);
                out.admissions[index] = Some(admission);
                continue;
            }
            let next = next_generation
                .checked_add(needed as u64)
                .ok_or("group generation exhausted")?;
            // Reserve all children before superseding any parent; rejected splits leave parents live.
            for slot in slots.iter_mut().flatten() {
                if key.parents.contains(&Some(slot.group.handle)) {
                    let original = self
                        .slots
                        .iter()
                        .flatten()
                        .find(|s| s.group.handle == slot.group.handle)
                        .unwrap()
                        .group;
                    out.superseded[superseded] = Some(original);
                    superseded += 1;
                    slot.group = original;
                    slot.group.eligible = false;
                    slot.dormant = true;
                }
            }
            for child in 0..needed {
                let destination = if let Some(free) = slots.iter().position(Option::is_none) {
                    free
                } else {
                    let (i, old) = slots
                        .iter()
                        .enumerate()
                        .filter_map(|(i, s)| s.filter(|s| s.dormant).map(|s| (i, s)))
                        .min_by_key(|(_, s)| (s.last_active_end, s.group.handle))
                        .unwrap();
                    out.retired[retired] = Some(Retired {
                        group: old.group,
                        reason: Retirement::Capacity,
                        last_active_end: old.last_active_end,
                        inactive_samples: old.inactive_samples,
                    });
                    retired += 1;
                    i
                };
                let handles = if key.kind == Kind::Split {
                    &key.sets[child]
                } else {
                    &key.members
                };
                let mut members = [Trajectory {
                    handle: self.residual,
                    point: crate::temporal_cognition::ridge::Point {
                        frequency_log2: None,
                        log_envelope: None,
                    },
                    slopes: [None; 2],
                }; 8];
                let mut count = 0;
                for h in handles.iter().flatten() {
                    members[count] = prepared
                        .current
                        .iter()
                        .flatten()
                        .find(|t| t.handle == *h)
                        .copied()
                        .ok_or("missing admission descriptor")?;
                    count += 1;
                }
                let handle = Handle {
                    generation: next_generation + child as u64,
                    ..self.residual
                };
                let group = Group::seed(handle, end, &members[..count])?;
                slots[destination] = Some(Slot {
                    group,
                    dormant: false,
                    last_active_end: end,
                    inactive_samples: 0,
                });
                admission.children[child] = Some(handle);
            }
            next_generation = next;
            out.admissions[index] = Some(admission);
        }
        self.slots = slots;
        self.next_generation = next_generation;
        self.last_end = end;
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
