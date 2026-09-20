//! Bounded acoustic grouping admission on the shared recurrence inventory.

use super::{Estimator, Peak};
use crate::temporal_cognition::{features::Accent, ridge::Handle};
use std::collections::VecDeque;

const STEPS: [f64; 9] = [0.25, 1. / 3., 0.5, 2. / 3., 1., 1.5, 2., 3., 4.];

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(in crate::temporal_cognition) struct Controls {
    pub tolerance: f64,
    pub integers_234_only: bool,
    pub strict_integer: bool,
    pub one_skip_words: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
enum Shape {
    Integer(u8),
    Word { len: u8, symbols: [u8; 8] },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
pub(in crate::temporal_cognition) struct Key {
    period_bin: u16,
    shape: Shape,
    anchors: [(u64, u64); 17],
    anchor_count: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(in crate::temporal_cognition) struct Proposal {
    pub(in crate::temporal_cognition) key: Key,
    period_seconds: f64,
    nominal_duration_seconds: f64,
    observed_duration_samples: [u64; 2],
    coverage: [(u64, u64); 2],
    timing_residuals: [f64; 16],
    pub(in crate::temporal_cognition) endpoint_weight: f64,
    mean_endpoint_weight: f64,
    pub(in crate::temporal_cognition) source_start: u64,
    pub(in crate::temporal_cognition) source_end: u64,
    pub(in crate::temporal_cognition) available_end: u64,
    skipped_accent: Option<(u64, u64)>,
    observed_pairs: u16,
}

/// A currently matched acoustic word, retaining original observations for deduplication.
#[derive(Clone, Copy, Debug)]
pub(in crate::temporal_cognition) struct Word {
    pub len: u8,
    pub symbols: [u8; 8],
    pub anchors: [(u64, u64); 17],
    pub support: f64,
    pub observed_pairs: u16,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(in crate::temporal_cognition) struct Work {
    integer_cases: usize,
    word_cases: usize,
    skipped_word_cases: usize,
    boundary_search_comparisons: usize,
    step_comparisons: usize,
    duration_checks: usize,
    pub(in crate::temporal_cognition) admitted_cases: usize,
    pub(in crate::temporal_cognition) duplicate_retained_keys: usize,
    ranking_comparisons: usize,
    pub(in crate::temporal_cognition) integer_window_limited_cases: usize,
    pub(in crate::temporal_cognition) word_insufficient_endpoints: usize,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct View {
    pub group: Handle,
    pub refreshed_at: u64,
    cadence_slot: u64,
    superseded_slots: u64,
    pub(in crate::temporal_cognition) period_source_end: Option<u64>,
    pub(in crate::temporal_cognition) period_available_end: Option<u64>,
    pub(in crate::temporal_cognition) retained_span: Option<(u64, u64)>,
    pub(in crate::temporal_cognition) capacity_evicted_through: Option<u64>,
    pub(in crate::temporal_cognition) proposals: [Option<Proposal>; 16],
    pub(in crate::temporal_cognition) work: Work,
    controls: Controls,
}

pub(in crate::temporal_cognition) struct Inventory {
    group: Handle,
    epoch_start: u64,
    controls: Controls,
    view: Option<View>,
    last_clock: u64,
}

impl Inventory {
    pub fn new(group: Handle, epoch_start: u64, controls: Controls) -> Result<Self, &'static str> {
        if group.bus > 1 || group.generation <= 1 {
            return Err("grouping requires a resolved group generation");
        }
        if !controls.tolerance.is_finite() || controls.tolerance <= 0. || controls.tolerance >= 1. {
            return Err("invalid frozen grouping tolerance");
        }
        Ok(Self {
            group,
            epoch_start,
            controls,
            view: None,
            last_clock: epoch_start,
        })
    }

    pub fn reset(&mut self, group: Handle, activated_at: u64) -> Result<(), &'static str> {
        if activated_at < self.epoch_start {
            return Err("group activation precedes the acoustic epoch");
        }
        let mut fresh = Self::new(group, self.epoch_start, self.controls)?;
        fresh.last_clock = activated_at;
        *self = fresh;
        Ok(())
    }

    pub fn snapshot(&self) -> Option<View> {
        self.view
    }

    pub fn refresh(&mut self, estimator: &Estimator) -> Result<bool, &'static str> {
        if estimator.ledger.group != self.group {
            return Err("grouping and recurrence group generations differ");
        }
        let now = estimator.ledger.received_at;
        if now < self.last_clock {
            return Err("grouping source clock moved backwards");
        }
        let slot = u64::try_from(
            u128::from(now - self.epoch_start) * 10 / u128::from(estimator.sample_rate),
        )
        .map_err(|_| "grouping cadence exhausted")?;
        if self.view.is_some_and(|v| v.cadence_slot == slot) {
            self.last_clock = now;
            return Ok(false);
        }
        let initial_slot = u64::try_from(
            u128::from(self.last_clock - self.epoch_start) * 10 / u128::from(estimator.sample_rate),
        )
        .map_err(|_| "grouping activation cadence exhausted")?;
        let superseded = self
            .view
            .map_or(slot - initial_slot, |v| slot - v.cadence_slot - 1);
        let mut view = View {
            group: self.group,
            refreshed_at: now,
            cadence_slot: slot,
            superseded_slots: superseded,
            period_source_end: estimator.ledger.bank.iter().map(|a| a.source_end).max(),
            period_available_end: estimator.ledger.bank.iter().map(|a| a.available_end).max(),
            retained_span: estimator
                .ledger
                .bank
                .front()
                .zip(estimator.ledger.bank.back())
                .map(|(a, b)| (a.event_end, b.event_end)),
            capacity_evicted_through: estimator.ledger.capacity_evicted_through,
            proposals: [None; 16],
            work: Work::default(),
            controls: self.controls,
        };
        admit(
            &estimator.ledger.bank,
            &estimator.view.peaks,
            estimator.sample_rate,
            self.controls,
            &mut view,
        );
        self.view = Some(view);
        self.last_clock = now;
        Ok(true)
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct CandidateSnapshot {
    pub period_seconds: f64,
    pub integer_len: Option<u8>,
    pub word_len: u8,
    pub word_symbols: [u8; 8],
    pub nominal_duration_seconds: f64,
    pub observed_duration_samples: [u64; 2],
    pub source_start: u64,
    pub source_end: u64,
    pub available: u64,
}
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Snapshot {
    pub admission_support: Option<f64>,
    pub refreshed_at: u64,
    pub proposals: [Option<CandidateSnapshot>; 16],
    pub window_limited_cases: usize,
    pub capacity_evicted_through: Option<u64>,
}

impl View {
    pub(in crate::temporal_cognition) fn matches_source(&self, estimator: &Estimator) -> bool {
        self.group == estimator.ledger.group
            && estimator.source_support().is_some_and(|s| {
                self.period_source_end == Some(s[1]) && self.period_available_end == Some(s[2])
            })
            && self.retained_span.map(|s| s.1) == estimator.latest_accent_end()
    }

    pub(in crate::temporal_cognition) fn words(&self, estimator: &Estimator) -> [Option<Word>; 16] {
        let mut words: [Option<Word>; 16] = [None; 16];
        if !self.matches_source(estimator) {
            return words;
        }
        let mut count = 0;
        for p in self.proposals.iter().flatten() {
            let Shape::Word { len, symbols } = p.key.shape else {
                continue;
            };
            if p.skipped_accent.is_some()
                || Some(p.key.anchors[2 * usize::from(len)].1) != estimator.latest_accent_end()
                || p.available_end > estimator.ledger.received_at
                || p.mean_endpoint_weight <= 0.
            {
                continue;
            }
            if let Some(old) = words[..count]
                .iter_mut()
                .flatten()
                .find(|w| w.len == len && w.symbols == symbols && w.anchors == p.key.anchors)
            {
                old.support = old.support.max(p.mean_endpoint_weight);
                continue;
            }
            words[count] = Some(Word {
                len,
                symbols,
                anchors: p.key.anchors,
                support: p.mean_endpoint_weight,
                observed_pairs: p.observed_pairs,
            });
            count += 1;
        }
        words
    }

    pub(in crate::temporal_cognition) fn diagnostics(&self) -> Snapshot {
        let support = self
            .proposals
            .iter()
            .flatten()
            .map(|p| p.mean_endpoint_weight)
            .max_by(f64::total_cmp)
            .or_else(|| {
                (self.capacity_evicted_through.is_none()
                    && self.work.integer_window_limited_cases == 0
                    && self.work.word_insufficient_endpoints == 0)
                    .then_some(0.)
            });
        Snapshot {
            admission_support: support,
            refreshed_at: self.refreshed_at,
            window_limited_cases: self.work.integer_window_limited_cases
                + self.work.word_insufficient_endpoints,
            capacity_evicted_through: self.capacity_evicted_through,
            proposals: self.proposals.map(|p| {
                p.map(|p| {
                    let (integer_len, word_len, word_symbols) = match p.key.shape {
                        Shape::Integer(n) => (Some(n), 0, [0; 8]),
                        Shape::Word { len, symbols } => (None, len, symbols),
                    };
                    CandidateSnapshot {
                        period_seconds: p.period_seconds,
                        integer_len,
                        word_len,
                        word_symbols,
                        nominal_duration_seconds: p.nominal_duration_seconds,
                        observed_duration_samples: p.observed_duration_samples,
                        source_start: p.source_start,
                        source_end: p.source_end,
                        available: p.available_end,
                    }
                })
            }),
        }
    }

    pub(in crate::temporal_cognition) fn word_indicator(
        &self,
        period_bin: usize,
        support: Option<[u64; 3]>,
    ) -> Option<bool> {
        if support.is_none_or(|s| {
            self.period_source_end != Some(s[1]) || self.period_available_end != Some(s[2])
        }) {
            return None;
        }
        if self.proposals.iter().flatten().any(|p| {
            matches!(p.key.shape, Shape::Word { .. })
                && usize::from(p.key.period_bin) == period_bin
                && p.key.anchors[p.key.anchor_count as usize - 1].1
                    == self.retained_span.map_or(0, |s| s.1)
        }) {
            return Some(true);
        }
        if self.capacity_evicted_through.is_some() || self.work.word_insufficient_endpoints > 0 {
            None
        } else {
            Some(false)
        }
    }

    #[cfg(test)]
    pub fn proposal_count(&self) -> usize {
        self.proposals.iter().flatten().count()
    }
}

fn admit(
    bank: &VecDeque<Accent>,
    peaks: &[Option<Peak>; 8],
    sample_rate: u32,
    controls: Controls,
    view: &mut View,
) {
    let scan = Scan {
        bank,
        sample_rate,
        controls,
    };
    for peak in peaks.iter().flatten() {
        let period = peak.period_seconds * f64::from(sample_rate);
        for start in 0..bank.len() {
            for length in 2..=if controls.integers_234_only { 4 } else { 16 } {
                view.work.integer_cases += 1;
                let duration = f64::from(length) * period;
                // Offset arithmetic keeps large epoch sample indices out of f64.
                if ((bank.back().unwrap().event_end - bank[start].event_end) as f64)
                    < (2. - controls.tolerance) * duration
                {
                    view.work.integer_window_limited_cases += 1;
                }
                let Some(middle) = scan.nearest(
                    start,
                    start,
                    duration,
                    controls.tolerance * duration,
                    &mut view.work,
                ) else {
                    continue;
                };
                let Some(end) = scan.nearest(
                    start,
                    middle,
                    2. * duration,
                    controls.tolerance * duration,
                    &mut view.work,
                ) else {
                    continue;
                };
                if controls.strict_integer && (middle != start + 1 || end != middle + 1) {
                    continue;
                }
                let mut anchors = [0; 17];
                anchors[..3].copy_from_slice(&[start, middle, end]);
                if let Some(proposal) = scan.evaluate(
                    peak,
                    Shape::Integer(length),
                    &anchors,
                    duration,
                    None,
                    &mut view.work,
                ) {
                    retain(proposal, view);
                }
            }
            if controls.integers_234_only {
                continue;
            }
            for length in 2..=8 {
                view.work.word_cases += 1;
                if start + 2 * length >= bank.len() {
                    view.work.word_insufficient_endpoints += 1;
                    continue;
                }
                let mut anchors = [0; 17];
                for (i, a) in anchors[..2 * length + 1].iter_mut().enumerate() {
                    *a = start + i;
                }
                if let Some(p) = scan.word(peak, &anchors, length, None, &mut view.work) {
                    retain(p, view);
                }
                if controls.one_skip_words && start + 2 * length + 1 < bank.len() {
                    // The declared control may omit exactly one interior accent, never an endpoint.
                    for skip in 1..=2 * length {
                        view.work.skipped_word_cases += 1;
                        for (i, a) in anchors[..2 * length + 1].iter_mut().enumerate() {
                            *a = start + i + usize::from(i >= skip);
                        }
                        if let Some(p) =
                            scan.word(peak, &anchors, length, Some(start + skip), &mut view.work)
                        {
                            retain(p, view);
                        }
                    }
                }
            }
        }
    }
}

struct Scan<'a> {
    bank: &'a VecDeque<Accent>,
    sample_rate: u32,
    controls: Controls,
}

impl Scan<'_> {
    fn nearest(
        &self,
        start: usize,
        previous: usize,
        target: f64,
        tolerance: f64,
        work: &mut Work,
    ) -> Option<usize> {
        let bank = self.bank;
        let (mut lo, mut hi) = (previous + 1, bank.len());
        while lo < hi {
            work.boundary_search_comparisons += 1;
            let mid = lo + (hi - lo) / 2;
            if ((bank[mid].event_end - bank[start].event_end) as f64) < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        [lo.checked_sub(1), (lo < bank.len()).then_some(lo)]
            .into_iter()
            .flatten()
            .filter(|&i| i > previous)
            .map(|i| {
                (
                    i,
                    ((bank[i].event_end - bank[start].event_end) as f64 - target).abs(),
                )
            })
            .filter(|(_, error)| *error <= tolerance)
            .min_by(|(i, a), (j, b)| a.total_cmp(b).then(i.cmp(j)))
            .map(|(i, _)| i)
    }

    fn word(
        &self,
        peak: &Peak,
        anchors: &[usize; 17],
        length: usize,
        skipped: Option<usize>,
        work: &mut Work,
    ) -> Option<Proposal> {
        let bank = self.bank;
        let sample_rate = self.sample_rate;
        let tolerance = self.controls.tolerance;
        let period = peak.period_seconds * f64::from(sample_rate);
        let mut symbols = [0u8; 8];
        let mut beats = 0.;
        for step in 0..length {
            let interval =
                (bank[anchors[step + 1]].event_end - bank[anchors[step]].event_end) as f64;
            let mut symbol = 0;
            let mut error = f64::INFINITY;
            for (i, &relative) in STEPS.iter().enumerate() {
                work.step_comparisons += 1;
                let candidate = (interval - relative * period).abs();
                if candidate < error {
                    error = candidate;
                    symbol = i;
                }
            }
            if error > tolerance * STEPS[symbol] * period {
                return None;
            }
            symbols[step] = symbol as u8;
            beats += STEPS[symbol];
            let repeat = (bank[anchors[step + length + 1]].event_end
                - bank[anchors[step + length]].event_end) as f64;
            work.step_comparisons += 1;
            if (repeat - STEPS[symbol] * period).abs() > tolerance * STEPS[symbol] * period {
                return None;
            }
        }
        if beats > 16. {
            return None;
        }
        self.evaluate(
            peak,
            Shape::Word {
                len: length as u8,
                symbols,
            },
            anchors,
            beats * period,
            skipped,
            work,
        )
    }

    fn evaluate(
        &self,
        peak: &Peak,
        shape: Shape,
        indices: &[usize; 17],
        duration: f64,
        skipped: Option<usize>,
        work: &mut Work,
    ) -> Option<Proposal> {
        let bank = self.bank;
        let sample_rate = self.sample_rate;
        let tolerance = self.controls.tolerance;
        let count = match shape {
            Shape::Integer(_) => 3,
            Shape::Word { len, .. } => 2 * usize::from(len) + 1,
        };
        let middle = count / 2;
        let a = bank[indices[0]];
        let b = bank[indices[middle]];
        let c = bank[indices[count - 1]];
        let durations = [b.event_end - a.event_end, c.event_end - b.event_end];
        let supports = [
            b.observed_prefix - a.observed_prefix,
            c.observed_prefix - b.observed_prefix,
        ];
        for j in 0..2 {
            work.duration_checks += 1;
            if (durations[j] as f64 - duration).abs() > tolerance * duration
                || u128::from(supports[j]) * 10 < u128::from(durations[j]) * 9
            {
                return None;
            }
        }
        let mut result = Proposal {
            key: Key {
                period_bin: peak.bin as u16,
                shape,
                anchors: [(0, 0); 17],
                anchor_count: count as u8,
            },
            period_seconds: peak.period_seconds,
            nominal_duration_seconds: duration / f64::from(sample_rate),
            observed_duration_samples: durations,
            coverage: [(supports[0], durations[0]), (supports[1], durations[1])],
            timing_residuals: [0.; 16],
            endpoint_weight: 0.,
            mean_endpoint_weight: 0.,
            source_start: u64::MAX,
            source_end: 0,
            available_end: 0,
            skipped_accent: skipped.map(|i| (bank[i].event_start, bank[i].event_end)),
            observed_pairs: 0,
        };
        for (i, &index) in indices[..count].iter().enumerate() {
            let accent = bank[index];
            result.key.anchors[i] = (accent.event_start, accent.event_end);
            result.endpoint_weight += accent.weight;
            result.source_start = result.source_start.min(accent.source_start);
            result.source_end = result.source_end.max(accent.source_end);
            result.available_end = result.available_end.max(accent.available_end);
            if i > 0 {
                let nominal = match shape {
                    Shape::Integer(_) => duration,
                    Shape::Word { len, symbols } => {
                        STEPS[symbols[(i - 1) % len as usize] as usize]
                            * peak.period_seconds
                            * f64::from(sample_rate)
                    }
                };
                result.timing_residuals[i - 1] =
                    ((accent.event_end - bank[indices[i - 1]].event_end) as f64 - nominal)
                        / nominal;
            }
            if i >= 2 && skipped.is_none() {
                let first = bank[indices[i - 2]];
                if accent.observed_prefix.checked_sub(first.observed_prefix)
                    == Some(accent.event_end - first.event_end)
                {
                    result.observed_pairs |= 1 << (i - 2);
                }
            }
        }
        result.mean_endpoint_weight = result.endpoint_weight / count as f64;
        Some(result)
    }
}

fn retain(proposal: Proposal, view: &mut View) {
    view.work.admitted_cases += 1;
    if view
        .proposals
        .iter()
        .flatten()
        .any(|p| p.key == proposal.key)
    {
        view.work.duplicate_retained_keys += 1;
        return;
    }
    for slot in 0..16 {
        view.work.ranking_comparisons += 1;
        if view.proposals[slot].is_none_or(|p| {
            proposal.endpoint_weight > p.endpoint_weight
                || (proposal.endpoint_weight == p.endpoint_weight && proposal.key < p.key)
        }) {
            view.proposals.copy_within(slot..15, slot + 1);
            view.proposals[slot] = Some(proposal);
            return;
        }
    }
}

#[cfg(test)]
mod tests;
