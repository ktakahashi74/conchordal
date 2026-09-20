//! Acquired-sample dispatch clock and bounded, generation-qualified query ownership.

use super::{
    descriptor::{Block, Frozen},
    features::RawDescriptor,
    memory,
    ridge::Handle,
};

mod cache;
pub(super) use cache::{Binding, Cache, Snapshot};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Header {
    pub epoch: u64,
    pub generation: u64,
    pub query_id: u64,
    pub occurrence_id: u64,
    pub support_id: u64,
    pub count: u64,
    pub start: f64,
    pub end: f64,
    pub audio_end: f64,
    pub available: f64,
    pub captured: f64,
    pub error: f64,
    pub scales: [f64; 10],
}

impl Header {
    fn same(&self, other: &Self) -> bool {
        [
            self.epoch,
            self.generation,
            self.query_id,
            self.occurrence_id,
            self.support_id,
            self.count,
        ] == [
            other.epoch,
            other.generation,
            other.query_id,
            other.occurrence_id,
            other.support_id,
            other.count,
        ] && [
            self.start,
            self.end,
            self.audio_end,
            self.available,
            self.captured,
            self.error,
        ]
        .map(f64::to_bits)
            == [
                other.start,
                other.end,
                other.audio_end,
                other.available,
                other.captured,
                other.error,
            ]
            .map(f64::to_bits)
            && self.scales.map(f64::to_bits) == other.scales.map(f64::to_bits)
    }
}

struct Slot {
    header: Header,
    blocks: Box<[Block]>,
}

impl Slot {
    fn new(capacity: usize) -> Self {
        Self {
            header: Header::default(),
            blocks: vec![Block::default(); capacity].into_boxed_slice(),
        }
    }

    fn capture(&mut self, frozen: &Frozen, ids: [u64; 3], rate: u32) -> Result<(), &'static str> {
        if frozen.blocks.is_empty() || frozen.blocks.len() > self.blocks.len() {
            return Err("query exceeds its bounded slot or has no span");
        }
        let rate = f64::from(rate);
        self.header = Header {
            epoch: frozen.group.epoch,
            generation: frozen.group.generation,
            query_id: ids[0],
            occurrence_id: ids[1],
            support_id: ids[2],
            count: frozen.blocks.len() as u64,
            start: frozen.start,
            end: frozen.end,
            audio_end: frozen
                .supporting_audio_end
                .map_or(f64::NAN, |s| s as f64 / rate),
            available: frozen.available_at as f64 / rate,
            captured: frozen.captured_at as f64 / rate,
            error: frozen.reconstruction_error,
            scales: frozen.scales,
        };
        self.blocks.fill(Block::default());
        self.blocks[..frozen.blocks.len()].copy_from_slice(&frozen.blocks);
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(super) struct Config {
    pub bus: u8,
    pub epoch: u64,
    pub sample_rate: u32,
    pub hop: u64,
    pub cadence_ms: u64,
    pub groups: usize,
    pub knots: usize,
    pub snapshots: usize,
    pub episodes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Counts {
    pub submitted: u64,
    pub superseded_pending: u64,
    pub older_query_loss: u64,
    pub stale_query: u64,
    pub dispatched: u64,
    pub completed: u64,
    pub incomplete: u64,
    pub stale_completion: u64,
    pub retired_pending: u64,
    pub retired_snapshots: u64,
    pub invalidated_active: u64,
    pub unbound_episode_entries: u64,
}

struct Group {
    generation: Option<u64>,
    bound: bool,
    samples: u64,
    next_due: u64,
    last: Option<[u64; 4]>,
    intervals: Box<[u64]>,
    query_highwater: Option<u64>,
    endpoint_highwater: f64,
    ready_order: u64,
    pending: Slot,
    pending_valid: bool,
    cache: Cache,
}

pub(super) struct Dispatch {
    pub ticket: u64,
    pub group_slot: usize,
    pub header: Header,
    pub dispatched_at: u64,
    pub descriptor: memory::Descriptor,
}

pub(super) struct Completion<'a> {
    pub header: Header,
    pub completed_at: u64,
    pub complete: bool,
    pub superseded: bool,
    pub coarse: &'a [memory::CoarseEntry],
}

pub(super) struct Scheduler {
    config: Config,
    period: u64,
    groups: Vec<Group>,
    interval_scratch: Box<[u64]>,
    active: Slot,
    scratch: Slot,
    coarse_scratch: Box<Snapshot>,
    active_slot: Option<usize>,
    active_obsolete: bool,
    dispatched_at: u64,
    cut: u64,
    order: u64,
    ticket: u64,
    pub counts: Counts,
}

impl Scheduler {
    pub fn new(config: Config) -> Result<Self, &'static str> {
        if config.bus > 1
            || config.sample_rate == 0
            || config.hop == 0
            || ![50, 100, 200].contains(&config.cadence_ms)
            || !(1..=8).contains(&config.groups)
            || !(3..=256).contains(&config.knots)
            || !(1..=16).contains(&config.snapshots)
            || !(1..=memory::MAX_EPISODES).contains(&config.episodes)
        {
            return Err("invalid query scheduler clock or capacities");
        }
        let period = (config.cadence_ms * u64::from(config.sample_rate)).div_ceil(1000);
        let words = usize::try_from(config.hop.div_ceil(64))
            .map_err(|_| "acquisition bitmap exceeds address space")?;
        let groups = (0..config.groups)
            .map(|_| Group {
                generation: None,
                bound: false,
                samples: 0,
                next_due: period,
                last: None,
                intervals: vec![0; words].into_boxed_slice(),
                query_highwater: None,
                endpoint_highwater: 0.,
                ready_order: 0,
                pending: Slot::new(config.knots),
                pending_valid: false,
                cache: Cache::new(config.epoch, 0, config.snapshots, config.episodes),
            })
            .collect();
        Ok(Self {
            config,
            period,
            groups,
            interval_scratch: vec![0; words].into_boxed_slice(),
            active: Slot::new(config.knots),
            scratch: Slot::new(config.knots),
            coarse_scratch: Box::new(Snapshot::new(config.episodes)),
            active_slot: None,
            active_obsolete: false,
            dispatched_at: 0,
            cut: 0,
            order: 0,
            ticket: 0,
            counts: Counts::default(),
        })
    }

    fn check(&self, cut: u64, slot: Option<usize>) -> Result<(), &'static str> {
        if cut < self.cut || slot.is_some_and(|s| self.groups.get(s).is_none_or(|g| !g.bound)) {
            Err("monotone bus cut and bound group required")
        } else {
            Ok(())
        }
    }

    pub fn coarse_storage_bytes(&self) -> usize {
        self.coarse_scratch.storage_bytes()
            + self
                .groups
                .iter()
                .map(|g| g.cache.storage_bytes())
                .sum::<usize>()
    }

    pub fn retire(&mut self, slot: usize, cut: u64) -> Result<(), &'static str> {
        self.check(cut, Some(slot))?;
        let g = &mut self.groups[slot];
        self.counts.retired_pending += u64::from(g.pending_valid);
        self.counts.retired_snapshots += g.cache.clear(self.config.epoch, g.generation.unwrap());
        if self.active_slot == Some(slot) && !self.active_obsolete {
            self.active_obsolete = true;
            self.counts.invalidated_active += 1;
        }
        g.pending_valid = false;
        g.bound = false;
        self.cut = cut;
        Ok(())
    }

    pub fn bind(&mut self, slot: usize, generation: u64, cut: u64) -> Result<(), &'static str> {
        self.check(cut, None)?;
        let g = self.groups.get(slot).ok_or("invalid group slot")?;
        if generation == 0 || g.generation.is_some_and(|old| generation <= old) {
            return Err("group generation must increase on slot reuse");
        }
        if g.bound {
            self.retire(slot, cut)?;
        }
        let g = &mut self.groups[slot];
        g.generation = Some(generation);
        g.bound = true;
        g.samples = 0;
        g.next_due = self.period;
        g.last = None;
        g.intervals.fill(0);
        g.query_highwater = None;
        g.endpoint_highwater = 0.;
        g.pending_valid = false;
        g.cache.clear(self.config.epoch, generation);
        self.cut = cut;
        Ok(())
    }

    #[cfg(test)]
    pub fn restart(
        &mut self,
        epoch: u64,
        rate: u32,
        hop: u64,
        cut: u64,
    ) -> Result<(), &'static str> {
        if epoch <= self.config.epoch || rate == 0 || hop == 0 {
            return Err("new epoch and valid clock required");
        }
        let words = usize::try_from(hop.div_ceil(64))
            .map_err(|_| "acquisition bitmap exceeds address space")?;
        for slot in 0..self.groups.len() {
            if self.groups[slot].bound {
                self.retire(slot, self.cut)?;
            }
            self.groups[slot].generation = None;
            self.groups[slot].intervals = vec![0; words].into_boxed_slice();
        }
        self.interval_scratch = vec![0; words].into_boxed_slice();
        self.config.epoch = epoch;
        self.config.sample_rate = rate;
        self.config.hop = hop;
        self.period = (self.config.cadence_ms * u64::from(rate)).div_ceil(1000);
        self.cut = cut;
        Ok(())
    }

    pub fn observe(
        &mut self,
        slot: usize,
        raw: &RawDescriptor,
        intervals: &[(u64, u64)],
        cut: u64,
    ) -> Result<bool, &'static str> {
        self.check(cut, Some(slot))?;
        let g = &mut self.groups[slot];
        if raw.group
            != (Handle {
                bus: self.config.bus,
                epoch: self.config.epoch,
                generation: g.generation.unwrap(),
            })
            || raw.end.checked_sub(raw.start) != Some(self.config.hop)
            || raw.source_end < raw.end
            || raw.available_end < raw.source_end
            || raw.available_end > cut
            || intervals.len() as u64 > self.config.hop
        {
            return Err("current canonical acquisition and causal source required");
        }
        self.interval_scratch.fill(0);
        for &(a, b) in intervals {
            if a < raw.start || a > b || b > raw.end {
                return Err("acquisition interval outside raw hop");
            }
            for i in a - raw.start..b - raw.start {
                self.interval_scratch[i as usize / 64] |= 1 << (i % 64);
            }
        }
        let acquired: u64 = self
            .interval_scratch
            .iter()
            .map(|v| u64::from(v.count_ones()))
            .sum();
        if acquired != raw.known_samples {
            return Err("acquisition union disagrees with known sample count");
        }
        let key = [raw.start, raw.end, raw.source_end, raw.available_end];
        if let Some(old) = g.last
            && raw.start < old[1]
        {
            if old == key && g.intervals == self.interval_scratch {
                self.cut = cut;
                return Ok(false);
            }
            return Err("conflicting or stale canonical acquisition");
        }
        let total = g
            .samples
            .checked_add(acquired)
            .ok_or("sample clock exhausted; restart epoch")?;
        g.samples = total;
        g.last = Some(key);
        g.intervals.copy_from_slice(&self.interval_scratch);
        self.cut = cut;
        Ok(true)
    }

    pub fn submit(
        &mut self,
        slot: usize,
        query: &Frozen,
        ids: [u64; 3],
        cut: u64,
    ) -> Result<bool, &'static str> {
        self.check(cut, Some(slot))?;
        let g = &mut self.groups[slot];
        if query.group
            != (Handle {
                bus: self.config.bus,
                epoch: self.config.epoch,
                generation: g.generation.unwrap(),
            })
            || query.sample_rate != self.config.sample_rate
            || query.captured_at > cut
            || query.available_at > query.captured_at
            || query.end > query.captured_at as f64 / f64::from(self.config.sample_rate)
        {
            return Err("current group and available frozen prefix required");
        }
        if g.query_highwater.is_some_and(|old| ids[0] <= old) {
            self.counts.stale_query += 1;
            self.cut = cut;
            return Ok(false);
        }
        self.scratch.capture(query, ids, self.config.sample_rate)?;
        let order = if g.pending_valid {
            self.order
        } else {
            self.order
                .checked_add(1)
                .ok_or("query admission order exhausted")?
        };
        g.query_highwater = Some(ids[0]);
        self.counts.submitted += 1;
        self.cut = cut;
        if query.end < g.endpoint_highwater {
            self.counts.older_query_loss += 1;
            return Ok(false);
        }
        g.endpoint_highwater = query.end;
        if g.pending_valid {
            self.counts.superseded_pending += 1;
        } else {
            self.order = order;
            g.ready_order = order;
        }
        g.pending.header = self.scratch.header;
        g.pending.blocks.copy_from_slice(&self.scratch.blocks);
        g.pending_valid = true;
        Ok(true)
    }

    pub fn take(&mut self, cut: u64) -> Result<Option<Dispatch>, &'static str> {
        self.check(cut, None)?;
        self.cut = cut;
        if self.active_slot.is_some() {
            return Ok(None);
        }
        let chosen = self
            .groups
            .iter()
            .enumerate()
            .filter(|(_, g)| g.bound && g.pending_valid && g.samples >= g.next_due)
            .min_by_key(|(i, g)| (g.ready_order, *i))
            .map(|(i, _)| i);
        let Some(slot) = chosen else {
            return Ok(None);
        };
        let g = &mut self.groups[slot];
        let next = (g.samples / self.period)
            .checked_add(1)
            .and_then(|n| n.checked_mul(self.period))
            .ok_or("query cadence exhausted")?;
        let ticket = self
            .ticket
            .checked_add(1)
            .ok_or("worker ticket exhausted")?;
        self.active.header = g.pending.header;
        self.active.blocks.copy_from_slice(&g.pending.blocks);
        g.pending_valid = false;
        g.next_due = next;
        self.active_slot = Some(slot);
        self.active_obsolete = false;
        self.dispatched_at = cut;
        self.ticket = ticket;
        self.counts.dispatched += 1;
        let knots: Vec<_> = self.active.blocks[..self.active.header.count as usize]
            .iter()
            .map(|b| b.matching())
            .collect();
        #[cfg(test)]
        let local_intervals = knots
            .iter()
            .enumerate()
            .map(|(i, k)| {
                (i > 0 && k.timing != 0 && knots[i - 1].timing != 0)
                    .then(|| k.time - knots[i - 1].time)
            })
            .collect();
        Ok(Some(Dispatch {
            ticket,
            group_slot: slot,
            header: self.active.header,
            dispatched_at: cut,
            descriptor: memory::Descriptor {
                knots,
                #[cfg(test)]
                local_intervals,
            },
        }))
    }

    pub fn finish(
        &mut self,
        ticket: u64,
        result: Option<Completion<'_>>,
        bindings: &[Binding],
        cut: u64,
    ) -> Result<bool, &'static str> {
        self.check(cut, None)?;
        let Some(slot) = self.active_slot.filter(|_| ticket == self.ticket) else {
            self.counts.stale_completion += 1;
            self.cut = cut;
            return Ok(false);
        };
        if self.active_obsolete {
            self.counts.stale_completion += 1;
            self.active_slot = None;
            self.cut = cut;
            return Ok(false);
        }
        if let Some(r) = &result
            && (!self.active.header.same(&r.header)
                || r.completed_at < self.dispatched_at
                || r.completed_at > cut)
        {
            return Err("completion must echo the active query and actual completion cut");
        }
        let accepted = if let Some(r) = result.filter(|r| r.complete && !r.superseded) {
            let missing = self.coarse_scratch.pack(
                &r.header,
                r.coarse,
                bindings,
                cut as f64 / f64::from(self.config.sample_rate),
            )?;
            let accepted = self.groups[slot].cache.insert(&self.coarse_scratch)?;
            self.counts.unbound_episode_entries += missing;
            self.counts.completed += 1;
            accepted
        } else {
            self.counts.incomplete += 1;
            false
        };
        self.active_slot = None;
        self.cut = cut;
        Ok(accepted)
    }
}

#[cfg(test)]
mod tests;
