//! Exact proposal identities and bounded persistence/arbitration, before group mutation.

use super::group::{Assignment, Row};
use super::ridge::Handle;

const CAPACITY: usize = 8 + 28 + 21;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Kind {
    Merge,
    Split,
    Birth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Key {
    kind: Kind,
    parents: [Option<Handle>; 2],
    sets: [[Option<Handle>; 8]; 2],
    members: [Option<Handle>; 8],
}

impl Key {
    pub fn new(
        kind: Kind,
        parents: &[Handle],
        left: &[Handle],
        right: &[Handle],
    ) -> Result<Self, &'static str> {
        let expected_parents = match kind {
            Kind::Birth => 0,
            Kind::Split => 1,
            Kind::Merge => 2,
        };
        if parents.len() != expected_parents
            || left.is_empty()
            || left.len() + right.len() > 8
            || (kind == Kind::Birth) != right.is_empty()
            || (parents.len() == 2 && parents[0] == parents[1])
        {
            return Err("invalid proposal shape");
        }
        let first = left[0];
        if parents
            .iter()
            .chain(left)
            .chain(right)
            .any(|h| h.bus > 1 || h.bus != first.bus || h.epoch != first.epoch || h.generation == 0)
        {
            return Err("invalid proposal generation");
        }
        let mut out = Self {
            kind,
            parents: std::array::from_fn(|i| parents.get(i).copied()),
            sets: [[None; 8]; 2],
            members: [None; 8],
        };
        for (destination, source) in out.sets.iter_mut().zip([left, right]) {
            for (slot, &h) in destination.iter_mut().zip(source) {
                *slot = Some(h);
            }
            destination[..source.len()].sort_unstable();
        }
        if (kind == Kind::Split && out.sets[0] > out.sets[1])
            || (kind == Kind::Merge && out.parents[0] > out.parents[1])
        {
            out.sets.swap(0, 1);
            if kind == Kind::Merge {
                out.parents.swap(0, 1);
            }
        }
        let n = left.len() + right.len();
        for (slot, &h) in out.members.iter_mut().zip(left.iter().chain(right)) {
            *slot = Some(h);
        }
        out.members[..n].sort_unstable();
        if out.members[..n].windows(2).any(|w| w[0] == w[1]) {
            return Err("repeated proposal member");
        }
        Ok(out)
    }

    fn conflicts(self, other: Self) -> bool {
        self.members
            .iter()
            .flatten()
            .any(|h| other.members.contains(&Some(*h)))
            || self
                .parents
                .iter()
                .flatten()
                .any(|h| other.parents.contains(&Some(*h)))
    }
}

pub(super) fn former_parent(row: &Row, groups: &[Option<Handle>; 7]) -> Option<Handle> {
    if row
        .weights
        .iter()
        .any(|w| !w.is_finite() || !(0.0..=1.0).contains(w))
        || (row.weights.iter().sum::<f64>() - 1.0).abs() > 1e-12
    {
        return None;
    }
    let maximum = row.weights.iter().copied().fold(0.0, f64::max);
    let mut winners = row
        .weights
        .iter()
        .enumerate()
        .filter(|(_, w)| **w == maximum);
    let (winner, _) = winners.next()?;
    if winners.next().is_some() || winner == 7 {
        None
    } else {
        groups[winner]
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Candidate {
    key: Key,
    end_sample: u64,
    support: f64,
    score: f64,
}

impl Candidate {
    // The caller supplies the already frozen key and the required cross-pair extremum.
    pub fn measure(
        key: Key,
        assignment: &Assignment,
        cross_correlation: Option<f64>,
    ) -> Result<Option<Self>, &'static str> {
        let first = key.members[0].unwrap();
        if assignment.group_handles.iter().enumerate().any(|(i, h)| {
            h.is_some_and(|h| {
                h.bus != first.bus
                    || h.epoch != first.epoch
                    || h.generation == 0
                    || assignment.group_handles[..i].contains(&Some(h))
            })
        }) {
            return Err("invalid proposal group inventory");
        }
        if cross_correlation.is_some_and(|x| !x.is_finite() || !(-1.0..=1.0).contains(&x)) {
            return Err("invalid proposal correlation");
        }
        if key
            .parents
            .iter()
            .flatten()
            .any(|h| !assignment.group_handles.contains(&Some(*h)))
        {
            return Ok(None);
        }
        let mut support = 0.0;
        let mut residual = 0.0;
        let mut total = 0.0;
        for h in key.members.iter().flatten() {
            let mut matching = assignment
                .rows
                .iter()
                .flatten()
                .filter(|r| r.trajectory == *h);
            let Some(row) = matching.next() else {
                return Ok(None);
            };
            let sum = row.weights.iter().sum::<f64>();
            if matching.next().is_some()
                || row
                    .weights
                    .iter()
                    .any(|w| !w.is_finite() || !(0.0..=1.0).contains(w))
                || (sum - 1.0).abs() > 1e-12
                || row.weights[..7]
                    .iter()
                    .enumerate()
                    .any(|(i, w)| *w > 0.0 && assignment.group_handles[i].is_none())
            {
                return Err("invalid normalized proposal assignment");
            }
            support += row.weights.iter().copied().fold(0.0, f64::max);
            residual += row.weights[7];
            total += sum;
        }
        let score = match key.kind {
            Kind::Birth => residual / total,
            _ => match cross_correlation {
                Some(x) => x,
                None => return Ok(None),
            },
        };
        let qualifies = match key.kind {
            Kind::Birth => score >= 0.5,
            Kind::Split => score <= 0.2,
            Kind::Merge => score >= 0.8,
        };
        Ok(qualifies.then_some(Self {
            key,
            end_sample: assignment.end_sample,
            support,
            score,
        }))
    }
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    candidate: Candidate,
    hops: u8,
}

pub(super) struct Tracker {
    bus: u8,
    epoch: u64,
    hop: u64,
    persistence: u8,
    last_end: u64,
    pending: [Option<Entry>; CAPACITY],
}

#[derive(Debug)]
pub(super) struct Update {
    pub accepted: [Option<Candidate>; 8],
    pub conflicts: usize,
    pub pending: usize,
}

impl Tracker {
    pub fn new(
        bus: u8,
        epoch: u64,
        start: u64,
        hop: u64,
        persistence: u8,
    ) -> Result<Self, &'static str> {
        if bus > 1 || hop == 0 || !start.is_multiple_of(hop) || persistence == 0 {
            return Err("invalid proposal clock or persistence");
        }
        Ok(Self {
            bus,
            epoch,
            hop,
            persistence,
            last_end: start,
            pending: [None; CAPACITY],
        })
    }

    pub fn advance(
        &mut self,
        end: u64,
        fully_observed: bool,
        candidates: &[Candidate],
    ) -> Result<Update, &'static str> {
        if end <= self.last_end || !end.is_multiple_of(self.hop) || candidates.len() > CAPACITY {
            return Err("invalid proposal endpoint or capacity");
        }
        let mut counts = [0; 3];
        let mut members = [None; 8];
        let mut parents = [None; 7];
        for (i, candidate) in candidates.iter().enumerate() {
            let h = candidate.key.members[0].unwrap();
            if h.bus != self.bus
                || h.epoch != self.epoch
                || candidate.end_sample != end
                || candidates[..i].iter().any(|c| c.key == candidate.key)
            {
                return Err("duplicate or foreign proposal key");
            }
            counts[candidate.key.kind as usize] += 1;
            for (inventory, identities) in [
                (members.as_mut_slice(), candidate.key.members.as_slice()),
                (parents.as_mut_slice(), candidate.key.parents.as_slice()),
            ] {
                for h in identities.iter().flatten() {
                    if !inventory.contains(&Some(*h)) {
                        let Some(slot) = inventory.iter_mut().find(|s| s.is_none()) else {
                            return Err("proposal identities exceed frame capacity");
                        };
                        *slot = Some(*h);
                    }
                }
            }
        }
        if counts[0] > 21 || counts[1] > 28 || counts[2] > 8 {
            return Err("proposal kind exceeds registered bound");
        }
        let mut current = [None; CAPACITY];
        let mut count = 0;
        if fully_observed {
            for &candidate in candidates {
                let previous = (end - self.last_end == self.hop)
                    .then(|| {
                        self.pending
                            .iter()
                            .flatten()
                            .find(|p| p.candidate.key == candidate.key)
                    })
                    .flatten();
                current[count] = Some(Entry {
                    candidate,
                    hops: previous.map_or(1, |p| p.hops.saturating_add(1)),
                });
                count += 1;
            }
        }
        current[..count].sort_unstable_by(|a, b| {
            let a = a.unwrap().candidate;
            let b = b.unwrap().candidate;
            b.support
                .total_cmp(&a.support)
                .then(a.key.kind.cmp(&b.key.kind))
                .then(a.key.parents.cmp(&b.key.parents))
                .then(a.key.members.cmp(&b.key.members))
                .then(a.key.sets.cmp(&b.key.sets))
        });
        let mut out = Update {
            accepted: [None; 8],
            conflicts: 0,
            pending: 0,
        };
        let mut accepted = 0;
        let mut pending = [None; CAPACITY];
        for entry in current[..count].iter().flatten() {
            if entry.hops < self.persistence {
                pending[out.pending] = Some(*entry);
                out.pending += 1;
            } else if out
                .accepted
                .iter()
                .flatten()
                .any(|c| c.key.conflicts(entry.candidate.key))
            {
                out.conflicts += 1;
            } else {
                out.accepted[accepted] = Some(entry.candidate);
                accepted += 1;
            }
        }
        self.pending = pending;
        self.last_end = end;
        Ok(out)
    }
}

pub(crate) mod frontend;
mod lifecycle;
mod producer;

#[cfg(test)]
mod tests;
