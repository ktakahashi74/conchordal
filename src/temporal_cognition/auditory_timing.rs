//! Public acoustic timing evidence, independent of Voice actions and private traces.

use super::{accents::periods::Peak, features::Accent, ridge::Handle};
use std::cell::Cell;
use std::collections::VecDeque;

mod shape;

const GROUPS: usize = 7;
const BINS: usize = 32;
const CAPACITY: usize = 128;
const HISTORIES: usize = 91;

#[derive(Clone, Copy)]
pub(super) struct Input {
    pub group: Handle,
    pub known: bool,
    /// Newly delivered accent only, after the ledger's provenance/deduplication checks.
    pub accent: Option<Accent>,
    pub peaks: [Option<Peak>; 8],
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) enum Family {
    Periodic,
    MedianInterval,
}

#[derive(Clone, Copy)]
struct Record {
    end: u64,
    target: [f64; 2],
    anchor: [f64; 2],
    weight: f64,
    version: u64,
}

impl Record {
    fn new(target: Accent, anchor: Accent, scale: f64, weight: f64, version: u64) -> Self {
        Self::from_intervals(
            [target.event_start, target.event_end],
            [anchor.event_start, anchor.event_end],
            scale,
            weight,
            version,
        )
    }

    fn from_intervals(
        target: [u64; 2],
        anchor: [u64; 2],
        scale: f64,
        weight: f64,
        version: u64,
    ) -> Self {
        let origin = anchor[1];
        let relative = |tick: u64| (i128::from(tick) - i128::from(origin)) as f64 / scale;
        Self {
            end: target[1],
            target: target.map(relative),
            anchor: [relative(anchor[0]), 0.],
            weight,
            version,
        }
    }

    fn masses(self, periodic: bool) -> [f64; BINS + 1] {
        let mut masses = [0.; BINS + 1];
        let low = self.target[0] - self.anchor[1];
        let high = self.target[1] - self.anchor[0];
        let width = if periodic { 1. } else { 4. } / BINS as f64;
        let (first, last) = if periodic {
            (low.floor() as i64, high.floor() as i64)
        } else {
            (0, 0)
        };
        for cycle in first..=last {
            for (bin, mass) in masses[..BINS].iter_mut().enumerate() {
                let left = cycle as f64 + bin as f64 * width;
                let right = left + width;
                *mass += super::private_trace::difference_cdf(right, self.target, self.anchor)
                    - super::private_trace::difference_cdf(left, self.target, self.anchor);
            }
        }
        if !periodic {
            masses[BINS] = (1. - masses[..BINS].iter().sum::<f64>()).max(0.);
        }
        masses
    }
}

struct History {
    records: VecDeque<Record>,
    masses: [f64; BINS + 1],
    capacity_evicted: u64,
    lost_through: u64,
    discarded: u64,
    shape: Cell<Option<(bool, shape::Shape)>>,
}

impl History {
    fn new() -> Self {
        Self {
            records: VecDeque::with_capacity(CAPACITY),
            masses: [0.; BINS + 1],
            capacity_evicted: 0,
            lost_through: 0,
            discarded: 0,
            shape: Cell::new(None),
        }
    }

    fn clear(&mut self) {
        self.shape.set(None);
        self.discarded += self.records.len() as u64;
        self.records.clear();
        self.masses.fill(0.);
        self.lost_through = 0;
    }

    fn remove(&mut self, periodic: bool) -> Record {
        self.shape.set(None);
        let old = self.records.pop_front().unwrap();
        if self.records.is_empty() {
            self.masses.fill(0.);
        } else {
            for (mass, removed) in self.masses.iter_mut().zip(old.masses(periodic)) {
                *mass = (*mass - old.weight * removed).max(0.);
            }
        }
        old
    }

    fn expire(&mut self, start: u64, periodic: bool) {
        while self.records.front().is_some_and(|r| r.end <= start) {
            self.remove(periodic);
        }
    }

    fn insert(&mut self, record: Record, periodic: bool) {
        self.shape.set(None);
        if self.records.len() == CAPACITY {
            let old = self.remove(periodic);
            self.capacity_evicted += 1;
            self.lost_through = self.lost_through.max(old.end);
        }
        for (mass, added) in self.masses.iter_mut().zip(record.masses(periodic)) {
            *mass += record.weight * added;
        }
        self.records.push_back(record);
    }
}

#[derive(Clone, Copy, Default)]
struct Reference {
    peak: Option<Peak>,
    supported: bool,
    version: u64,
    challenger: Option<(usize, u64)>,
}

impl Reference {
    fn update(&mut self, peaks: [Option<Peak>; 8], known: bool, cut: u64, rate: u32) -> bool {
        let valid = |p: &&Peak| {
            known
                && p.support.is_finite()
                && p.support > 0.
                && p.period_seconds.is_finite()
                && p.period_seconds > 0.
        };
        let best = peaks
            .iter()
            .flatten()
            .filter(valid)
            .copied()
            .max_by(|a, b| a.support.total_cmp(&b.support).then(b.bin.cmp(&a.bin)));
        let Some(incumbent) = self.peak else {
            if let Some(best) = best {
                self.peak = Some(best);
                self.supported = true;
                self.version += 1;
                return true;
            }
            return false;
        };
        let current = peaks
            .iter()
            .flatten()
            .filter(valid)
            .find(|p| p.bin == incumbent.bin);
        self.supported = current.is_some();
        if let Some(current) = current {
            self.peak = Some(*current);
        }
        let challenger = best.filter(|p| {
            p.bin != incumbent.bin && p.support >= 1.25 * current.map_or(0., |p| p.support)
        });
        let Some(challenger) = challenger else {
            self.challenger = None;
            return false;
        };
        let start = match self.challenger {
            Some((bin, start)) if bin == challenger.bin => start,
            _ => cut,
        };
        self.challenger = Some((challenger.bin, start));
        if cut - start < u64::from(rate) {
            return false;
        }
        self.peak = Some(challenger);
        self.supported = true;
        self.version += 1;
        self.challenger = None;
        true
    }

    fn scale(self, rate: u32) -> Option<f64> {
        self.peak
            .filter(|_| self.supported)
            .map(|p| p.period_seconds * f64::from(rate))
    }
}

#[derive(Clone, Copy, Default)]
struct Group {
    owner: Option<Handle>,
    reference: Reference,
    latest: Option<Accent>,
    previous: Option<(Accent, Reference)>,
    intervals: [u64; 4],
    interval_count: usize,
}

impl Group {
    fn scale(self) -> Option<f64> {
        if self.interval_count < 2 {
            return None;
        }
        let mut sorted = self.intervals;
        sorted[..self.interval_count].sort_unstable();
        let n = self.interval_count;
        Some((sorted[(n - 1) / 2] as f64 + sorted[n / 2] as f64) / 2.)
    }

    fn observe(&mut self, accent: Accent) {
        if let Some(old) = self.latest {
            let elapsed = accent.event_end - old.event_end;
            if elapsed > 0
                && accent.observed_prefix.checked_sub(old.observed_prefix) == Some(elapsed)
            {
                self.intervals.rotate_right(1);
                self.intervals[0] = elapsed;
                self.interval_count = (self.interval_count + 1).min(4);
            } else {
                self.interval_count = 0;
            }
        }
        self.latest = Some(accent);
    }
}

#[derive(Clone, Copy)]
struct Coverage {
    start: u64,
    end: u64,
    generations: [u64; GROUPS],
    known: u8,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Summary {
    pub reference: Handle,
    pub family: Family,
    pub version: u64,
    pub period_seconds: Option<f64>,
    pub reference_support: f64,
    pub records: usize,
    pub retained_weight: f64,
    pub coverage: f64,
    pub supported: bool,
    /// Normalized retained evidence; consumers must honor `supported`.
    pub bins: [f64; BINS],
    pub overflow: f64,
    pub capacity_evicted: u64,
    pub discarded: u64,
    pub mode_bins: [Option<u8>; 2],
    pub mode_count: Option<usize>,
    pub residual_dispersion: Option<f64>,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct GroupSummary {
    pub group: Handle,
    pub outgoing: [Option<Summary>; 2],
    pub supported_outgoing: usize,
    pub within: Summary,
    /// Ordered raw blocks: inter_1, inter_2, within; 14 values each, missing as null.
    pub timing_features: [[Option<f64>; 14]; 3],
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Snapshot {
    pub window: [u64; 2],
    pub groups: [Option<GroupSummary>; GROUPS],
    pub record_bytes: usize,
    pub owned_bytes: usize,
    pub capacity_evicted: u64,
    pub discarded: u64,
}

pub(super) struct Stream {
    rate: u32,
    hop: u64,
    origin: u64,
    cut: u64,
    groups: [Group; GROUPS],
    histories: Box<[History; HISTORIES]>,
    coverage: VecDeque<Coverage>,
    coverage_capacity: usize,
}

fn index(target: usize, reference: usize, periodic: bool) -> usize {
    if target == reference {
        84 + target
    } else {
        (target * 6 + reference - usize::from(reference > target)) * 2 + usize::from(!periodic)
    }
}

impl Stream {
    pub(super) fn new(rate: u32, hop: u64, origin: u64) -> Self {
        assert!(rate > 0 && hop > 0);
        assert!(std::mem::size_of::<Record>() <= 64);
        let coverage_capacity = (8 * u64::from(rate)).div_ceil(hop) as usize + 1;
        Self {
            rate,
            hop,
            origin,
            cut: origin,
            groups: [Group::default(); GROUPS],
            histories: (0..HISTORIES)
                .map(|_| History::new())
                .collect::<Vec<_>>()
                .into_boxed_slice()
                .try_into()
                .unwrap_or_else(|_| unreachable!()),
            coverage: VecDeque::with_capacity(coverage_capacity),
            coverage_capacity,
        }
    }

    fn start(&self) -> u64 {
        self.cut
            .saturating_sub(8 * u64::from(self.rate))
            .max(self.origin)
    }

    fn covered(&self, target: usize, reference: usize, after: u64) -> f64 {
        let start = self.start();
        let width = self.cut - start;
        if width == 0 {
            return 0.;
        }
        let Some(a) = self.groups[target].owner else {
            return 0.;
        };
        let Some(b) = self.groups[reference].owner else {
            return 0.;
        };
        let mask = (1 << target) | (1 << reference);
        let samples: u64 = self
            .coverage
            .iter()
            .filter(|f| {
                f.known & mask == mask
                    && f.generations[target] == a.generation
                    && f.generations[reference] == b.generation
            })
            .map(|f| f.end.saturating_sub(f.start.max(start).max(after)))
            .sum();
        samples as f64 / width as f64
    }

    /// Inputs contain post-admission peaks; only the previously frozen reference scores this batch.
    pub(super) fn advance(
        &mut self,
        cut: u64,
        input: [Option<Input>; GROUPS],
        next: [Option<Handle>; GROUPS],
    ) {
        assert!(cut > self.cut);
        let contiguous = cut - self.cut == self.hop;
        self.cut = cut;
        let start = self.start();
        for (i, h) in self.histories.iter_mut().enumerate() {
            h.expire(start, i >= 84 || i % 2 == 0);
        }
        while self.coverage.front().is_some_and(|f| f.end <= start) {
            self.coverage.pop_front();
        }
        if self.coverage.len() == self.coverage_capacity {
            self.coverage.pop_front();
        }
        let frame = Coverage {
            start: cut.saturating_sub(self.hop).max(self.origin),
            end: cut,
            generations: std::array::from_fn(|i| input[i].map_or(0, |g| g.group.generation)),
            known: input.iter().enumerate().fold(0, |mask, (i, g)| {
                mask | (u8::from(g.is_some_and(|g| g.known)) << i)
            }),
        };
        self.coverage.push_back(frame);
        let mut frozen = self.groups.map(|g| g.reference);
        for i in 0..GROUPS {
            assert_eq!(input[i].map(|g| g.group), self.groups[i].owner);
            let known = input[i].is_some_and(|g| g.known);
            if !contiguous || !known {
                self.groups[i].latest = None;
                self.groups[i].previous = None;
                self.groups[i].interval_count = 0;
                self.groups[i].reference.challenger = None;
            }
            frozen[i].supported &= known && self.covered(i, i, 0) >= 0.9;
        }
        // All available anchors enter together, independent of group iteration order.
        let previous = self.groups;
        let mut anchors = previous;
        for i in 0..GROUPS {
            if let Some(accent) = input[i].filter(|g| g.known).and_then(|g| g.accent) {
                assert_eq!(Some(accent.group), self.groups[i].owner);
                anchors[i].observe(accent);
            }
        }
        for (target, row) in input.iter().enumerate() {
            let Some(accent) = row
                .filter(|g| g.known)
                .and_then(|g| g.accent)
                .filter(|a| a.event_end > start)
            else {
                continue;
            };
            for reference in 0..GROUPS {
                if target == reference {
                    continue;
                }
                let current = anchors[reference]
                    .latest
                    .filter(|a| a.event_end <= accent.event_end);
                let (anchor, context) = if current.is_some() {
                    (current, anchors[reference])
                } else {
                    (
                        previous[reference]
                            .latest
                            .filter(|a| a.event_end <= accent.event_end),
                        previous[reference],
                    )
                };
                let Some(anchor) = anchor else { continue };
                for periodic in [true, false] {
                    let scale = if periodic {
                        frozen[reference].scale(self.rate)
                    } else {
                        context.scale()
                    };
                    let Some(scale) = scale else { continue };
                    let version = if periodic {
                        frozen[reference].version
                    } else {
                        0
                    };
                    self.histories[index(target, reference, periodic)].insert(
                        Record::new(
                            accent,
                            anchor,
                            scale,
                            accent.weight * anchor.weight,
                            version,
                        ),
                        periodic,
                    );
                }
            }
            if let Some((previous, reference)) = previous[target].previous
                && previous.event_end < accent.event_end
                && reference.version == frozen[target].version
                && frozen[target].supported
                && accent.observed_prefix.checked_sub(previous.observed_prefix)
                    == Some(accent.event_end - previous.event_end)
                && let Some(scale) = reference.scale(self.rate)
            {
                self.histories[84 + target].insert(
                    Record::new(
                        accent,
                        previous,
                        scale,
                        accent.weight.min(previous.weight),
                        reference.version,
                    ),
                    true,
                );
            }
            self.groups[target].latest = anchors[target].latest;
            self.groups[target].intervals = anchors[target].intervals;
            self.groups[target].interval_count = anchors[target].interval_count;
            self.groups[target].previous = Some((accent, frozen[target]));
        }
        for (i, row) in input.iter().enumerate() {
            let Some(row) = row else { continue };
            let eligible = row.known && self.covered(i, i, 0) >= 0.9;
            if self.groups[i]
                .reference
                .update(row.peaks, eligible, cut, self.rate)
            {
                for target in 0..GROUPS {
                    self.histories[index(target, i, true)].clear();
                }
            }
        }
        // Retired acoustic generations cannot lend evidence to their replacements.
        for (i, owner) in next.into_iter().enumerate() {
            if self.groups[i].owner != owner {
                for j in 0..GROUPS {
                    for periodic in [true, false] {
                        self.histories[index(i, j, periodic)].clear();
                        self.histories[index(j, i, periodic)].clear();
                    }
                }
                self.groups[i] = Group {
                    owner,
                    ..Group::default()
                };
            }
        }
    }

    pub(super) fn snapshot(&self) -> Snapshot {
        Snapshot {
            window: [self.start(), self.cut],
            groups: std::array::from_fn(|i| {
                let group = self.groups[i].owner?;
                let mut outgoing = [None; 12];
                for (slot, value) in outgoing.iter_mut().enumerate() {
                    let reference = slot / 2 + usize::from(slot / 2 >= i);
                    *value = self
                        .summary(i, reference, slot % 2 == 0)
                        .filter(|s| s.supported);
                }
                outgoing.sort_by(|a, b| match (a, b) {
                    (Some(a), Some(b)) => (b.retained_weight * b.reference_support)
                        .total_cmp(&(a.retained_weight * a.reference_support))
                        .then(a.reference.cmp(&b.reference))
                        .then(
                            usize::from(a.family == Family::MedianInterval)
                                .cmp(&usize::from(b.family == Family::MedianInterval)),
                        ),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                });
                let supported_outgoing = outgoing.iter().flatten().count();
                let mut outgoing = [outgoing[0], outgoing[1]];
                let selected_weight = outgoing
                    .iter()
                    .flatten()
                    .map(|s| s.retained_weight * s.reference_support)
                    .sum::<f64>();
                let mut timing_features = [[None; 14]; 3];
                for (slot, summary) in outgoing.iter_mut().enumerate() {
                    if let Some(summary) = summary {
                        let reference = self
                            .groups
                            .iter()
                            .position(|g| g.owner == Some(summary.reference))
                            .unwrap();
                        let periodic = summary.family == Family::Periodic;
                        self.histories[index(i, reference, periodic)].describe(summary);
                        timing_features[slot] = summary.features(
                            summary.retained_weight * summary.reference_support / selected_weight,
                        );
                    }
                }
                let mut within = self.summary(i, i, true).unwrap();
                self.histories[84 + i].describe(&mut within);
                timing_features[2] = within.features(1.);
                Some(GroupSummary {
                    group,
                    outgoing,
                    supported_outgoing,
                    within,
                    timing_features,
                })
            }),
            record_bytes: std::mem::size_of::<Record>(),
            owned_bytes: std::mem::size_of::<Self>()
                + std::mem::size_of_val(self.histories.as_ref())
                + self.coverage.capacity() * std::mem::size_of::<Coverage>()
                + self
                    .histories
                    .iter()
                    .map(|h| h.records.capacity() * std::mem::size_of::<Record>())
                    .sum::<usize>(),
            capacity_evicted: self.histories.iter().map(|h| h.capacity_evicted).sum(),
            discarded: self.histories.iter().map(|h| h.discarded).sum(),
        }
    }

    fn summary(&self, target: usize, reference: usize, periodic: bool) -> Option<Summary> {
        let group = self.groups[reference];
        let owner = group.owner?;
        let history = &self.histories[index(target, reference, periodic)];
        let weight: f64 = history.records.iter().map(|r| r.weight).sum();
        let coverage = self.covered(target, reference, history.lost_through);
        let reference_support = if periodic {
            group
                .reference
                .peak
                .filter(|_| group.reference.supported)
                .map_or(0., |p| p.support)
        } else {
            f64::from(group.scale().is_some())
        };
        let version = if periodic { group.reference.version } else { 0 };
        debug_assert!(history.records.iter().all(|r| r.version == version));
        Some(Summary {
            reference: owner,
            family: if periodic {
                Family::Periodic
            } else {
                Family::MedianInterval
            },
            version,
            period_seconds: periodic
                .then_some(group.reference.peak)
                .flatten()
                .map(|p| p.period_seconds),
            reference_support,
            records: history.records.len(),
            retained_weight: weight,
            coverage,
            supported: weight > 0. && reference_support > 0. && coverage >= 0.9,
            bins: std::array::from_fn(|bin| {
                if weight > 0. {
                    history.masses[bin] / weight
                } else {
                    0.
                }
            }),
            overflow: if weight > 0. {
                history.masses[BINS] / weight
            } else {
                0.
            },
            capacity_evicted: history.capacity_evicted,
            discarded: history.discarded,
            mode_bins: [None; 2],
            mode_count: None,
            residual_dispersion: None,
        })
    }
}

#[cfg(test)]
mod tests;
