//! Ordered accent delivery and bounded retrospective retention on source clocks.

use std::collections::VecDeque;

use super::{features::Accent, ridge::Handle};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Delivery {
    pub sequence: u64,
    pub received_at: u64,
    pub accent: Accent,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Snapshot {
    pub group: Handle,
    pub start: u64,
    pub end: u64,
    pub accents: Vec<Accent>,
    pub capacity_valid: bool,
    pub weight: Option<f64>,
    pub cumulative_count: u64,
    pub cumulative_weight: f64,
    pub capacity_evicted_through: Option<u64>,
}

pub(super) struct Ledger {
    group: Handle,
    window_samples: u64,
    capacity: usize,
    bank: VecDeque<Accent>,
    last: Option<Accent>,
    received_at: u64,
    cumulative_count: u64,
    cumulative_weight: f64,
    capacity_evicted_through: Option<u64>,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Summary {
    pub group: Handle,
    pub received_at: u64,
    pub retained_accents: usize,
    pub cumulative_count: u64,
    pub cumulative_weight: f64,
    pub capacity_evicted_through: Option<u64>,
}

impl Ledger {
    pub fn reset(&mut self, group: Handle, received_at: u64) -> Result<(), &'static str> {
        if group.bus > 1 || group.generation == 0 {
            return Err("invalid accent ledger generation");
        }
        self.group = group;
        self.bank.clear();
        self.last = None;
        self.received_at = received_at;
        self.cumulative_count = 0;
        self.cumulative_weight = 0.;
        self.capacity_evicted_through = None;
        Ok(())
    }

    pub fn summary(&self) -> Summary {
        Summary {
            group: self.group,
            received_at: self.received_at,
            retained_accents: self.bank.len(),
            cumulative_count: self.cumulative_count,
            cumulative_weight: self.cumulative_weight,
            capacity_evicted_through: self.capacity_evicted_through,
        }
    }

    pub fn new(group: Handle, capacity: usize, window_samples: u64) -> Result<Self, &'static str> {
        if group.bus > 1
            || group.generation == 0
            || capacity == 0
            || capacity > 256
            || window_samples == 0
        {
            return Err("invalid bounded accent ledger configuration");
        }
        Ok(Self {
            group,
            window_samples,
            capacity,
            bank: VecDeque::with_capacity(capacity),
            last: None,
            received_at: 0,
            cumulative_count: 0,
            cumulative_weight: 0.,
            capacity_evicted_through: None,
        })
    }

    pub fn deliver(
        &mut self,
        accent: Accent,
        received_at: u64,
    ) -> Result<Option<Delivery>, &'static str> {
        if accent.group != self.group {
            return Ok(None);
        }
        if !accent.weight.is_finite()
            || accent.weight <= 0.
            || accent.weight > 1.
            || accent.event_start >= accent.event_end
            || accent.raw_intervals[2] != (accent.event_start, accent.event_end)
            || accent.raw_intervals.iter().any(|(a, b)| a >= b)
            || accent.raw_intervals.windows(2).any(|p| p[0].1 != p[1].0)
            || accent.source_start > accent.raw_intervals[0].0
            || accent.source_end < accent.raw_intervals[3].1
            || accent.available_end < accent.source_end
            || accent.observed_prefix > accent.event_end
            || accent.observed_prefix < accent.event_end.saturating_sub(accent.raw_intervals[0].0)
        {
            return Err("incomplete accent provenance or nonpositive weight");
        }
        if !accent.at_cut(received_at) {
            return Ok(None);
        }
        if received_at < self.received_at {
            return Err("accent receive clock moved backwards");
        }
        let identity = (accent.event_start, accent.event_end);
        if let Some(old) = self
            .last
            .iter()
            .chain(self.bank.iter())
            .find(|a| (a.event_start, a.event_end) == identity)
        {
            if *old != accent {
                return Err("conflicting repeat of an admitted accent");
            }
            return Ok(None);
        }
        if let Some(last) = self.last
            && ((accent.source_end, identity)
                <= (last.source_end, (last.event_start, last.event_end))
                || (accent.event_end, accent.event_start) <= (last.event_end, last.event_start))
        {
            return Err("stale accent requires upstream reconciliation");
        }
        if let Some(last) = self.last
            && (accent.observed_prefix < last.observed_prefix
                || accent.observed_prefix - last.observed_prefix
                    > accent.event_end - last.event_end)
        {
            return Err("accent observation prefix contradicts source interval");
        }
        let sequence = self
            .cumulative_count
            .checked_add(1)
            .ok_or("accent delivery sequence exhausted")?;
        self.advance(received_at)?;
        self.last = Some(accent);
        self.cumulative_count = sequence;
        self.cumulative_weight += accent.weight;
        if accent.event_end >= received_at.saturating_sub(self.window_samples) {
            if self.bank.len() == self.capacity {
                let evicted = self.bank.pop_front().unwrap();
                self.capacity_evicted_through = Some(
                    self.capacity_evicted_through
                        .map_or(evicted.event_end, |t| t.max(evicted.event_end)),
                );
            }
            self.bank.push_back(accent);
        }
        Ok(Some(Delivery {
            sequence,
            received_at,
            accent,
        }))
    }

    pub fn advance(&mut self, received_at: u64) -> Result<(), &'static str> {
        if received_at < self.received_at {
            return Err("accent receive clock moved backwards");
        }
        self.received_at = received_at;
        let start = received_at.saturating_sub(self.window_samples);
        while self.bank.front().is_some_and(|a| a.event_end < start) {
            self.bank.pop_front();
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn snapshot(&self, start: u64) -> Result<Snapshot, &'static str> {
        if start < self.received_at.saturating_sub(self.window_samples) || start > self.received_at
        {
            return Err("accent query lies outside the retained physical window");
        }
        let accents: Vec<_> = self
            .bank
            .iter()
            .copied()
            .filter(|a| a.event_end >= start)
            .collect();
        let capacity_valid = self.capacity_evicted_through.is_none_or(|t| t < start);
        let weight = capacity_valid.then(|| accents.iter().map(|a| a.weight).sum());
        Ok(Snapshot {
            group: self.group,
            start,
            end: self.received_at,
            accents,
            capacity_valid,
            weight,
            cumulative_count: self.cumulative_count,
            cumulative_weight: self.cumulative_weight,
            capacity_evicted_through: self.capacity_evicted_through,
        })
    }
}

pub(in crate::temporal_cognition) mod periods;
#[cfg(test)]
mod tests;
