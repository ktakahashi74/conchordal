//! Cached pair-interval recurrence proposals, independent of the metrical bank.

use super::{Accent, Delivery, Handle, Ledger};
use std::collections::VecDeque;

const BINS: usize = 241;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Pair {
    left: u16,
    right: u16,
    base: u16,
    flags: u16,
    values: [f32; 5],
    padding: [u8; 4],
}

#[derive(Clone, Copy)]
struct Point {
    start: u64,
    end: u64,
    observed_prefix: u64,
    weight: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct Peak {
    pub bin: usize,
    pub period_seconds: f64,
    pub support: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::temporal_cognition) struct Work {
    pub inserted_pairs: usize,
    pub removed_pairs: usize,
    pub added_bins: usize,
    pub subtracted_bins: usize,
    pub rebuilt_bins: usize,
    pub cleared_bins: usize,
    pub normalization_bins: usize,
    pub peak_bins: usize,
    pub peak_candidates: usize,
    pub separation_checks: usize,
    pub rebuilds: usize,
}

#[derive(Clone, Copy)]
pub(in crate::temporal_cognition) struct View {
    pub group: Handle,
    pub received_at: u64,
    pub probabilities: [f64; BINS],
    pub peaks: [Option<Peak>; 8],
    pub pairs: usize,
    pub supported_pairs: usize,
    pub nonzero_pairs: usize,
    pub minimum_raw_sum: f64,
    pub work: Work,
}

pub(in crate::temporal_cognition) struct Estimator {
    ledger: Ledger,
    sample_rate: u32,
    rebuild_every: u64,
    separation: f64,
    periods: [f64; BINS],
    order: VecDeque<u16>,
    points: [Option<Point>; 256],
    cache: Vec<Pair>,
    sums: [f64; BINS],
    view: View,
}

impl Estimator {
    pub fn reset(&mut self, group: Handle, received_at: u64) -> Result<(), &'static str> {
        self.ledger.reset(group, received_at)?;
        self.order.clear();
        self.points.fill(None);
        self.cache.fill(Pair::default());
        self.sums.fill(0.);
        self.view.group = group;
        self.view.received_at = received_at;
        self.view.probabilities.fill(0.);
        self.view.peaks.fill(None);
        self.view.pairs = 0;
        self.view.supported_pairs = 0;
        self.view.nonzero_pairs = 0;
        self.view.minimum_raw_sum = 0.;
        self.view.work = Work::default();
        Ok(())
    }

    pub fn ledger_summary(&self) -> super::Summary {
        self.ledger.summary()
    }

    #[cfg(test)]
    pub(in crate::temporal_cognition) fn latest_accent_end(&self) -> Option<u64> {
        self.ledger.last.map(|a| a.event_end)
    }

    pub(in crate::temporal_cognition) fn source_support(&self) -> Option<[u64; 3]> {
        self.ledger.bank.front().map(|_| {
            [
                self.ledger
                    .bank
                    .iter()
                    .map(|a| a.source_start)
                    .min()
                    .unwrap(),
                self.ledger.bank.iter().map(|a| a.source_end).max().unwrap(),
                self.ledger
                    .bank
                    .iter()
                    .map(|a| a.available_end)
                    .max()
                    .unwrap(),
            ]
        })
    }

    #[cfg(test)]
    pub fn storage_layout(&self) -> [usize; 4] {
        [
            self.cache.as_ptr() as usize,
            self.cache.capacity(),
            self.ledger.bank.capacity(),
            self.order.capacity(),
        ]
    }

    pub fn new(
        group: Handle,
        capacity: usize,
        sample_rate: u32,
        window_samples: u64,
        rebuild_every: u64,
        separation: f64,
    ) -> Result<Self, &'static str> {
        if sample_rate == 0 || rebuild_every == 0 || !separation.is_finite() || separation <= 0. {
            return Err("invalid period estimator clock or frozen controls");
        }
        let ledger = Ledger::new(group, capacity, window_samples)?;
        Ok(Self {
            ledger,
            sample_rate,
            rebuild_every,
            separation,
            periods: std::array::from_fn(|i| 0.125 * 2f64.powf(i as f64 / 48.)),
            order: VecDeque::with_capacity(capacity),
            points: [None; 256],
            cache: vec![Pair::default(); capacity * (capacity - 1) / 2],
            sums: [0.; BINS],
            view: View {
                group,
                received_at: 0,
                probabilities: [0.; BINS],
                peaks: [None; 8],
                pairs: 0,
                supported_pairs: 0,
                nonzero_pairs: 0,
                minimum_raw_sum: 0.,
                work: Work::default(),
            },
        })
    }

    pub fn deliver(
        &mut self,
        accent: Accent,
        received_at: u64,
    ) -> Result<Option<Delivery>, &'static str> {
        let delivery = self.ledger.deliver(accent, received_at)?;
        if delivery.is_some() {
            self.refresh(
                Some(accent),
                self.ledger
                    .cumulative_count
                    .is_multiple_of(self.rebuild_every),
            );
        }
        Ok(delivery)
    }

    pub fn advance(&mut self, received_at: u64) -> Result<(), &'static str> {
        self.ledger.advance(received_at)?;
        self.refresh(None, false);
        Ok(())
    }

    pub fn view(&self) -> View {
        self.view
    }

    fn refresh(&mut self, incoming: Option<Accent>, rebuild: bool) {
        let mut work = Work::default();
        let had_contributions = self.view.nonzero_pairs > 0;
        let first = self
            .ledger
            .bank
            .front()
            .map(|a| (a.event_start, a.event_end));
        while let Some(&slot) = self.order.front() {
            let point = self.points[slot as usize].unwrap();
            if first == Some((point.start, point.end)) {
                break;
            }
            for other in 0..self.ledger.capacity {
                if other == slot as usize {
                    continue;
                }
                let (lo, hi) = if other < slot as usize {
                    (other, slot as usize)
                } else {
                    (slot as usize, other)
                };
                let record = &mut self.cache[hi * (hi - 1) / 2 + lo];
                if record.flags & 1 == 0 {
                    continue;
                }
                for (offset, &value) in record.values.iter().enumerate() {
                    if let Some(sum) = self.sums.get_mut(record.base as usize + offset) {
                        *sum -= f64::from(value);
                        work.subtracted_bins += 1;
                    }
                }
                self.view.pairs -= 1;
                self.view.supported_pairs -= usize::from(record.flags & 2 != 0);
                self.view.nonzero_pairs -= usize::from(record.flags & 4 != 0);
                *record = Pair::default();
                work.removed_pairs += 1;
            }
            self.points[slot as usize] = None;
            self.order.pop_front();
        }
        // No rounded contribution remains: cancellation residue cannot invent a period.
        if had_contributions && self.view.nonzero_pairs == 0 {
            self.sums.fill(0.);
            work.cleared_bins += BINS;
        }
        if let Some(a) = incoming.filter(|a| self.ledger.bank.back() == Some(a)) {
            let slot = self.points[..self.ledger.capacity]
                .iter()
                .position(Option::is_none)
                .unwrap();
            for &other in &self.order {
                let p = self.points[other as usize].unwrap();
                let dt = a.event_end - p.end;
                let supported = dt > 0
                    && u128::from(a.observed_prefix - p.observed_prefix) * 10 >= u128::from(dt) * 9;
                let (lo, hi) = if slot < other as usize {
                    (slot, other as usize)
                } else {
                    (other as usize, slot)
                };
                let mut record = Pair {
                    left: lo as u16,
                    right: hi as u16,
                    flags: 1 | (u16::from(supported) << 1),
                    ..Pair::default()
                };
                if supported {
                    let seconds = dt as f64 / f64::from(self.sample_rate);
                    let coordinate = (seconds / 0.125).log2() * 48.;
                    let base = (coordinate.floor() as i64 - 2).max(0);
                    if base < BINS as i64 {
                        record.base = base as u16;
                        for (offset, value) in record.values.iter_mut().enumerate() {
                            if let Some(&period) = self.periods.get(base as usize + offset) {
                                let kernel =
                                    (1. - (seconds / period).log2().abs() / (1. / 24.)).max(0.);
                                *value = (p.weight * a.weight * kernel) as f32;
                            }
                        }
                    }
                }
                if record.values.iter().any(|&x| x > 0.) {
                    record.flags |= 4;
                }
                for (offset, &value) in record.values.iter().enumerate() {
                    if let Some(sum) = self.sums.get_mut(record.base as usize + offset) {
                        *sum += f64::from(value);
                        work.added_bins += 1;
                    }
                }
                self.cache[hi * (hi - 1) / 2 + lo] = record;
                self.view.pairs += 1;
                self.view.supported_pairs += usize::from(supported);
                self.view.nonzero_pairs += usize::from(record.flags & 4 != 0);
                work.inserted_pairs += 1;
            }
            self.points[slot] = Some(Point {
                start: a.event_start,
                end: a.event_end,
                observed_prefix: a.observed_prefix,
                weight: a.weight,
            });
            self.order.push_back(slot as u16);
        }
        if rebuild {
            self.sums.fill(0.);
            work.cleared_bins += BINS;
            for record in self.cache.iter().filter(|p| p.flags & 1 != 0) {
                for (offset, &value) in record.values.iter().enumerate() {
                    if let Some(sum) = self.sums.get_mut(record.base as usize + offset) {
                        *sum += f64::from(value);
                        work.rebuilt_bins += 1;
                    }
                }
            }
            work.rebuilds = 1;
        }
        let mut total = 0.;
        self.view.minimum_raw_sum = 0.;
        for &sum in &self.sums {
            total += sum.max(0.);
            self.view.minimum_raw_sum = self.view.minimum_raw_sum.min(sum);
        }
        for (probability, &sum) in self.view.probabilities.iter_mut().zip(&self.sums) {
            *probability = if total > 0. { sum.max(0.) / total } else { 0. };
        }
        work.normalization_bins = BINS * 2;
        let (peaks, candidates, comparisons) =
            select_peaks(&self.view.probabilities, &self.periods, self.separation);
        self.view.peaks = peaks;
        work.peak_bins = BINS;
        work.peak_candidates = candidates;
        work.separation_checks = comparisons;
        self.view.work = work;
        self.view.received_at = self.ledger.received_at;
    }
}

fn select_peaks(
    probabilities: &[f64; BINS],
    periods: &[f64; BINS],
    separation: f64,
) -> ([Option<Peak>; 8], usize, usize) {
    let mut candidates = [Peak {
        bin: 0,
        period_seconds: 0.,
        support: 0.,
    }; 121];
    let mut count = 0;
    let mut bin = 0;
    while bin < BINS {
        let start = bin;
        while bin + 1 < BINS && probabilities[bin + 1] == probabilities[start] {
            bin += 1;
        }
        let positive = probabilities[start] > 0.;
        let left = start == 0 || probabilities[start] > probabilities[start - 1];
        let right = bin + 1 == BINS || probabilities[start] > probabilities[bin + 1];
        if positive && left && right && !(start == 0 && bin + 1 == BINS) {
            candidates[count] = Peak {
                bin: start,
                period_seconds: periods[start],
                support: probabilities[start],
            };
            count += 1;
        }
        bin += 1;
    }
    candidates[..count]
        .sort_unstable_by(|a, b| b.support.total_cmp(&a.support).then(a.bin.cmp(&b.bin)));
    let mut selected: [Option<Peak>; 8] = [None; 8];
    let mut kept = 0;
    let mut comparisons = 0;
    for candidate in &candidates[..count] {
        let spaced = selected[..kept].iter().flatten().all(|prior| {
            comparisons += 1;
            candidate.bin.abs_diff(prior.bin) as f64 / 48. >= separation
        });
        if spaced {
            selected[kept] = Some(*candidate);
            kept += 1;
            if kept == selected.len() {
                break;
            }
        }
    }
    (selected, count, comparisons)
}

pub(in crate::temporal_cognition) mod groupings;
#[cfg(test)]
mod tests;
