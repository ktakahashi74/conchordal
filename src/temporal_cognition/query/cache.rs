//! Original-query snapshots with fixed generation handles and separate validity bits.

use super::{Header, memory};
use crate::temporal_cognition::transport::Identity;

#[derive(Clone, Copy, Debug)]
pub(in crate::temporal_cognition) struct Binding {
    pub identity: Identity,
    pub slot: usize,
    pub handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Cell {
    handle: u64,
    cost: f64,
    similarity: f64,
}

#[repr(C)]
#[derive(Debug)]
pub(in crate::temporal_cognition) struct Snapshot {
    ids: [u64; 5],
    times: [f64; 4],
    valid: Vec<u64>,
    approximate: Vec<u64>,
    cells: Vec<Cell>,
}

impl Clone for Snapshot {
    fn clone(&self) -> Self {
        Self {
            ids: self.ids,
            times: self.times,
            valid: self.valid.clone(),
            approximate: self.approximate.clone(),
            cells: self.cells.clone(),
        }
    }

    fn clone_from(&mut self, other: &Self) {
        self.ids = other.ids;
        self.times = other.times;
        self.valid.clone_from(&other.valid);
        self.approximate.clone_from(&other.approximate);
        self.cells.clone_from(&other.cells);
    }
}

#[cfg(test)]
impl Default for Snapshot {
    fn default() -> Self {
        Self::new(256)
    }
}

impl Snapshot {
    pub(super) fn storage_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.valid.capacity() * std::mem::size_of::<u64>()
            + self.approximate.capacity() * std::mem::size_of::<u64>()
            + self.cells.capacity() * std::mem::size_of::<Cell>()
    }

    pub(super) fn new(capacity: usize) -> Self {
        assert!((1..=memory::MAX_EPISODES).contains(&capacity));
        Self {
            ids: [0; 5],
            times: [0.; 4],
            valid: vec![0; capacity.div_ceil(64)],
            approximate: vec![0; capacity.div_ceil(64)],
            cells: vec![Cell::default(); capacity],
        }
    }
}

impl Snapshot {
    pub(super) fn pack(
        &mut self,
        query: &Header,
        entries: &[memory::CoarseEntry],
        bindings: &[Binding],
        received: f64,
    ) -> Result<u64, &'static str> {
        if !([
            query.start,
            query.end,
            query.available,
            query.captured,
            received,
        ]
        .iter()
        .all(|v| v.is_finite())
            && 0. <= query.start
            && query.start <= query.end
            && query.end <= query.available
            && query.available <= query.captured
            && query.captured <= received)
            || (!query.audio_end.is_nan()
                && (!query.audio_end.is_finite()
                    || query.audio_end < 0.
                    || query.audio_end > query.available))
            || entries.len() > self.cells.len()
            || bindings.len() > self.cells.len()
        {
            return Err("causal completed query and bounded entries required");
        }
        for (i, b) in bindings.iter().enumerate() {
            if b.slot >= self.cells.len()
                || b.handle == 0
                || bindings[..i].iter().any(|old| {
                    old.slot == b.slot || old.handle == b.handle || old.identity == b.identity
                })
            {
                return Err("distinct generation handles and bank slots required");
            }
        }
        for (i, e) in entries.iter().enumerate() {
            if entries[..i]
                .iter()
                .any(|old| old.identity.id == e.identity.id)
                || e.cost.is_some() != e.similarity.is_some()
                || e.cost.is_some_and(|v| !v.is_finite() || v < 0.)
                || e.similarity
                    .is_some_and(|v| !v.is_finite() || !(0. ..=1.).contains(&v))
            {
                return Err("unique finite coarse entries with matching masks required");
            }
        }
        self.valid.fill(0);
        self.approximate.fill(0);
        self.cells.fill(Cell::default());
        self.ids = [
            query.epoch,
            query.generation,
            query.query_id,
            query.occurrence_id,
            query.support_id,
        ];
        self.times = [query.end, query.audio_end, received, query.start];
        let mut missing = 0;
        for e in entries {
            let Some(b) = bindings.iter().find(|b| b.identity == e.identity) else {
                missing += 1;
                continue;
            };
            self.cells[b.slot] = Cell {
                handle: b.handle,
                cost: e.cost.unwrap_or(0.),
                similarity: e.similarity.unwrap_or(0.),
            };
            let (word, mask) = (b.slot / 64, 1 << (b.slot % 64));
            if e.cost.is_some() {
                self.valid[word] |= mask;
            }
            if e.approximate {
                self.approximate[word] |= mask;
            }
        }
        Ok(missing)
    }

    fn same(&self, other: &Self) -> bool {
        self.ids == other.ids
            && self.times.map(f64::to_bits) == other.times.map(f64::to_bits)
            && self.valid == other.valid
            && self.approximate == other.approximate
            && self.cells == other.cells
    }
}

#[derive(Debug)]
pub(in crate::temporal_cognition) struct Entry {
    pub handle: u64,
    pub cost: Option<f64>,
    pub similarity: Option<f64>,
    pub approximate: bool,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct CoarseEvidence {
    pub epoch: u64,
    pub generation: u64,
    pub query_id: u64,
    pub occurrence_id: u64,
    pub support_id: u64,
    pub start: f64,
    pub end: f64,
    pub audio_end: Option<f64>,
    pub received_at: f64,
}

impl Snapshot {
    pub(in crate::temporal_cognition) fn evidence(&self) -> CoarseEvidence {
        CoarseEvidence {
            epoch: self.ids[0],
            generation: self.ids[1],
            query_id: self.ids[2],
            occurrence_id: self.ids[3],
            support_id: self.ids[4],
            start: self.times[3],
            end: self.times[0],
            audio_end: (!self.times[1].is_nan()).then_some(self.times[1]),
            received_at: self.times[2],
        }
    }

    pub(in crate::temporal_cognition) fn entry(&self, slot: usize, handle: u64) -> Option<Entry> {
        let c = self
            .cells
            .get(slot)
            .filter(|c| handle != 0 && c.handle == handle)?;
        let (word, mask) = (slot / 64, 1 << (slot % 64));
        let valid = self.valid[word] & mask != 0;
        Some(Entry {
            handle,
            cost: valid.then_some(c.cost),
            similarity: valid.then_some(c.similarity),
            approximate: self.approximate[word] & mask != 0,
        })
    }
}

#[cfg(test)]
#[derive(Debug)]
pub(in crate::temporal_cognition) struct View {
    pub ids: [u64; 5],
    pub end: f64,
    pub audio_end: Option<f64>,
    pub received_at: f64,
    pub entries: Vec<Entry>,
}

pub(in crate::temporal_cognition) struct Cache {
    epoch: u64,
    generation: u64,
    blocks: Box<[Snapshot]>,
    occupied: Box<[bool]>,
    highwater: Option<u64>,
    pub evictions: u64,
    pub stale: u64,
}

impl Cache {
    pub(super) fn storage_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.occupied.len() * std::mem::size_of::<bool>()
            + self
                .blocks
                .iter()
                .map(Snapshot::storage_bytes)
                .sum::<usize>()
    }

    pub(super) fn for_commitment(
        &self,
        occurrence: u64,
        span: [u64; 2],
        deadline: u64,
        rate: u32,
        cut: u64,
    ) -> Option<&Snapshot> {
        self.blocks
            .iter()
            .zip(&self.occupied)
            .filter_map(|(s, used)| {
                if !used || s.ids[3] != occurrence || s.times[1].is_nan() {
                    return None;
                }
                let samples = |value: f64| (value * f64::from(rate)).round() as u64;
                let end = samples(s.times[0]);
                let audio_end = samples(s.times[1]);
                (samples(s.times[3]) == span[0]
                    && end <= span[1]
                    && audio_end <= span[1]
                    && u128::from(span[1] - end) * 10 <= u128::from(rate)
                    && u128::from(span[1] - audio_end) * 10 <= u128::from(rate)
                    && samples(s.times[2]) <= deadline.min(cut))
                .then_some(s)
            })
            .max_by(|a, b| {
                a.times[0]
                    .total_cmp(&b.times[0])
                    .then(a.ids[2].cmp(&b.ids[2]))
            })
    }

    pub(super) fn new(epoch: u64, generation: u64, capacity: usize, episodes: usize) -> Self {
        Self {
            epoch,
            generation,
            blocks: vec![Snapshot::new(episodes); capacity].into_boxed_slice(),
            occupied: vec![false; capacity].into_boxed_slice(),
            highwater: None,
            evictions: 0,
            stale: 0,
        }
    }

    pub(super) fn clear(&mut self, epoch: u64, generation: u64) -> u64 {
        let count = self.occupied.iter().filter(|x| **x).count() as u64;
        self.epoch = epoch;
        self.generation = generation;
        self.highwater = None;
        self.occupied.fill(false);
        count
    }

    pub(super) fn insert(&mut self, snapshot: &Snapshot) -> Result<bool, &'static str> {
        if self
            .blocks
            .first()
            .is_none_or(|b| b.cells.len() != snapshot.cells.len())
        {
            return Err("snapshot and cache episode capacities must agree");
        }
        if snapshot.ids[..2] != [self.epoch, self.generation] {
            self.stale += 1;
            return Ok(false);
        }
        let (end, received) = (snapshot.times[0], snapshot.times[2]);
        if !end.is_finite() || !received.is_finite() || received < end {
            return Err("invalid snapshot endpoints");
        }
        let id = snapshot.ids[2];
        let mut free = None;
        let mut oldest: Option<(usize, (f64, u64))> = None;
        for (i, used) in self.occupied.iter().enumerate() {
            if !used {
                if free.is_none() {
                    free = Some(i);
                }
                continue;
            }
            let old = &self.blocks[i];
            if old.ids[2] == id {
                if !old.same(snapshot) {
                    return Err("conflicting retained snapshot replay");
                }
                return Ok(false);
            }
            let key = (old.times[0], old.ids[2]);
            if oldest.is_none_or(|(_, prior)| key < prior) {
                oldest = Some((i, key));
            }
        }
        if self.highwater.is_some_and(|old| id <= old) {
            self.stale += 1;
            return Ok(false);
        }
        self.highwater = Some(id);
        let target = if let Some(i) = free {
            i
        } else {
            self.evictions += 1;
            let (i, key) = oldest.unwrap();
            if (end, id) < key {
                return Ok(false);
            }
            i
        };
        self.blocks[target].clone_from(snapshot);
        self.occupied[target] = true;
        Ok(true)
    }

    #[cfg(test)]
    pub(in crate::temporal_cognition) fn snapshots(
        &self,
        current: &[(usize, u64)],
        cut: f64,
    ) -> Result<Vec<View>, &'static str> {
        let capacity = self.blocks.first().map_or(0, |b| b.cells.len());
        if !cut.is_finite() || current.len() > capacity {
            return Err("finite snapshot cut and bounded bindings required");
        }
        for (i, &(slot, handle)) in current.iter().enumerate() {
            if slot >= capacity
                || handle == 0
                || current[..i].iter().any(|&(s, h)| s == slot || h == handle)
            {
                return Err("distinct current bank slots and handles required");
            }
        }
        let mut views = Vec::new();
        for (snapshot, &used) in self.blocks.iter().zip(&self.occupied) {
            if !used || snapshot.times[2] > cut {
                continue;
            }
            let mut entries = Vec::new();
            for &(slot, handle) in current {
                let c = snapshot.cells[slot];
                if c.handle != handle {
                    continue;
                }
                let (word, mask) = (slot / 64, 1 << (slot % 64));
                let valid = snapshot.valid[word] & mask != 0;
                entries.push(Entry {
                    handle,
                    cost: valid.then_some(c.cost),
                    similarity: valid.then_some(c.similarity),
                    approximate: snapshot.approximate[word] & mask != 0,
                });
            }
            views.push(View {
                ids: snapshot.ids,
                end: snapshot.times[0],
                audio_end: (!snapshot.times[1].is_nan()).then_some(snapshot.times[1]),
                received_at: snapshot.times[2],
                entries,
            });
        }
        views.sort_by(|a, b| a.end.total_cmp(&b.end).then(a.ids[2].cmp(&b.ids[2])));
        Ok(views)
    }
}

#[cfg(test)]
mod tests;
