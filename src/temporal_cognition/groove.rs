//! Public acoustic inputs shared by the groove and participation rating heads.

use super::accents::periods::{Estimator, groupings};
use std::collections::VecDeque;

mod heads;
mod projection;
pub(crate) use heads::CandidateHeads;
pub(crate) use heads::validate;
pub(super) use heads::{Heads, Issue, Snapshot};
pub(crate) use projection::{ContextProjection, DensityProjection};

const WINDOWS: [f64; 8] = [0.125, 0.25, 0.5, 1., 2., 4., 8., 16.];
const PAIRS: usize = 128;

#[derive(Clone, Copy)]
struct Frame {
    start: u64,
    end: u64,
    known: bool,
    alpha: f64,
    word: Option<[f64; 9]>,
    grouping: Option<f64>,
}

#[derive(Clone, Copy)]
struct Surprise {
    end: u64,
    value: f64,
}

pub(super) struct Group {
    rate: u32,
    origin: u64,
    last_end: u64,
    frames: VecDeque<Frame>,
    frame_capacity: usize,
    word: Option<[f64; 9]>,
    processed_through: Option<u64>,
    counts: [[f64; 9]; 9],
    surprise: VecDeque<Surprise>,
    lost_through: Option<u64>,
    capacity_evicted: u64,
}

pub(super) struct Input {
    pub start: u64,
    pub end: u64,
    pub known: bool,
    pub alpha: f64,
    pub refreshed: bool,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Summary {
    pub end_sample: u64,
    /// Raw blocks: inter_1, inter_2, within, density, H/H^2/surprise, grouping.
    #[serde(with = "crate::config::fixed_array")]
    pub raw: [Option<f64>; 54],
    pub density_coverage: [f64; 8],
    pub density_retention_supported: [bool; 8],
    pub word_coverage: f64,
    pub word_probabilities: Option<[f64; 9]>,
    pub grouping_coverage: f64,
    pub observed_coverage: f64,
    pub assignment_sample_weight: f64,
    pub learned_pair_mass: f64,
    pub retained_pairs: usize,
    pub surprise_capacity_evicted: u64,
    pub surprise_capacity_supported: bool,
    pub owned_bytes: usize,
}

impl Group {
    pub(super) fn new(rate: u32, hop: u64, origin: u64) -> Self {
        assert!(rate > 0 && hop > 0);
        let frame_capacity = (16 * u64::from(rate)).div_ceil(hop) as usize + 1;
        Self {
            rate,
            origin,
            last_end: origin,
            frames: VecDeque::with_capacity(frame_capacity),
            frame_capacity,
            word: None,
            processed_through: None,
            counts: [[0.; 9]; 9],
            surprise: VecDeque::with_capacity(PAIRS),
            lost_through: None,
            capacity_evicted: 0,
        }
    }

    pub(super) fn reset(&mut self) {
        self.last_end = self.origin;
        self.frames.clear();
        self.word = None;
        self.processed_through = None;
        self.counts = [[0.; 9]; 9];
        self.surprise.clear();
        self.lost_through = None;
        self.capacity_evicted = 0;
    }

    pub(super) fn advance(
        &mut self,
        estimator: &Estimator,
        view: Option<&groupings::View>,
        input: Input,
    ) -> Summary {
        assert!(input.end > input.start && input.start >= self.last_end);
        assert!(input.alpha.is_finite() && (0. ..=1.).contains(&input.alpha));
        self.last_end = input.end;
        let start = input
            .end
            .saturating_sub(16 * u64::from(self.rate))
            .max(self.origin);
        while self.frames.front().is_some_and(|f| f.end <= start) {
            self.frames.pop_front();
        }
        let word_start = input
            .end
            .saturating_sub(8 * u64::from(self.rate))
            .max(self.origin);
        while self.surprise.front().is_some_and(|s| s.end < word_start) {
            self.surprise.pop_front();
        }
        if input.refreshed {
            let words = view.map_or([None; 16], |v| v.words(estimator));
            self.observe_words(&words, estimator.latest_accent_end(), word_start);
        }
        let matched = view.is_some_and(|v| v.matches_source(estimator));
        if self.frames.len() == self.frame_capacity {
            self.frames.pop_front();
        }
        self.frames.push_back(Frame {
            start: input.start,
            end: input.end,
            known: input.known,
            alpha: if input.known { input.alpha } else { 0. },
            word: self.word.filter(|_| matched && input.known),
            grouping: view
                .and_then(|v| v.diagnostics().admission_support)
                .filter(|_| input.known),
        });
        self.summarize(estimator, self.last_end)
    }

    fn observe_words(
        &mut self,
        words: &[Option<groupings::Word>; 16],
        latest: Option<u64>,
        start: u64,
    ) {
        let weight: f64 = words.iter().flatten().map(|w| w.support).sum();
        self.word = (weight > 0.).then(|| {
            let mut probabilities = [0.; 9];
            for word in words.iter().flatten() {
                for &symbol in &word.symbols[..usize::from(word.len)] {
                    probabilities[usize::from(symbol)] +=
                        word.support / (weight * f64::from(word.len));
                }
            }
            probabilities
        });
        // Consecutive words ending at the same accent share the original pair identity.
        // Process all newly observed pairs in order, including multiple accents per refresh.
        let mut after = self.processed_through.unwrap_or(0);
        loop {
            let next = words
                .iter()
                .flatten()
                .flat_map(|w| {
                    (0..2 * usize::from(w.len) - 1).filter_map(move |i| {
                        let end = w.anchors[i + 2].1;
                        (w.observed_pairs & (1 << i) != 0 && end > after).then_some(end)
                    })
                })
                .min();
            let Some(end) = next else { break };
            let mut assignments = [[0.; 9]; 9];
            for word in words.iter().flatten() {
                let len = usize::from(word.len);
                for i in 0..2 * len - 1 {
                    if word.anchors[i + 2].1 == end && word.observed_pairs & (1 << i) != 0 {
                        assignments[usize::from(word.symbols[i % len])]
                            [usize::from(word.symbols[(i + 1) % len])] += word.support;
                    }
                }
            }
            let total: f64 = assignments.iter().flatten().sum();
            let mut value = 0.;
            for (a, row) in assignments.iter().enumerate() {
                let denominator = self.counts[a].iter().sum::<f64>() + 4.5;
                for (b, &mass) in row.iter().enumerate() {
                    value -= mass / total * ((self.counts[a][b] + 0.5) / denominator).ln();
                }
            }
            for (row, incoming) in self.counts.iter_mut().zip(assignments) {
                for (count, mass) in row.iter_mut().zip(incoming) {
                    *count += mass / total;
                }
            }
            if end >= start {
                if self.surprise.len() == PAIRS {
                    let old = self.surprise.pop_front().unwrap();
                    self.lost_through = Some(old.end);
                    self.capacity_evicted += 1;
                }
                self.surprise.push_back(Surprise { end, value });
            }
            after = end;
        }
        // A later reinterpretation must not learn pairs previously left unquantized.
        self.processed_through = latest.or(self.processed_through);
    }

    fn summarize(&self, estimator: &Estimator, end: u64) -> Summary {
        assert!(end >= self.last_end);
        let starts = WINDOWS.map(|s| {
            end.saturating_sub((s * f64::from(self.rate)).round() as u64)
                .max(self.origin)
        });
        let mut raw = [None; 54];
        let mut known = [0u64; 8];
        let mut word_known = 0u64;
        let mut grouping_known = 0u64;
        let mut word_weight = 0.;
        let mut grouping_weight = 0.;
        let mut grouping_sum = 0.;
        let mut symbols = [0.; 9];
        let mut assignment_weight = 0.;
        for frame in &self.frames {
            if !frame.known {
                continue;
            }
            for (i, &start) in starts.iter().enumerate() {
                known[i] += frame.end.min(end).saturating_sub(frame.start.max(start));
            }
            let n = frame
                .end
                .min(end)
                .saturating_sub(frame.start.max(starts[6]));
            let weight = n as f64 * frame.alpha;
            assignment_weight += weight;
            if let Some(word) = frame.word {
                word_known += n;
                word_weight += weight;
                for (sum, p) in symbols.iter_mut().zip(word) {
                    *sum += weight * p;
                }
            }
            if let Some(grouping) = frame.grouping {
                grouping_known += n;
                grouping_weight += weight;
                grouping_sum += weight * grouping;
            }
        }
        let density_coverage = std::array::from_fn(|i| {
            if end > starts[i] {
                known[i] as f64 / (end - starts[i]) as f64
            } else {
                0.
            }
        });
        let mut retention = [false; 8];
        for i in 0..8 {
            let weight = estimator.accent_weight(starts[i], end);
            retention[i] = weight.is_some();
            raw[42 + i] = weight
                .filter(|_| known[i] > 0 && density_coverage[i] >= 0.9)
                .map(|w| w * f64::from(self.rate) / known[i] as f64);
        }
        let duration = (end - starts[6]) as f64;
        let word_coverage = if duration > 0. {
            word_known as f64 / duration
        } else {
            0.
        };
        let grouping_coverage = if duration > 0. {
            grouping_known as f64 / duration
        } else {
            0.
        };
        let word_probabilities =
            (word_weight > 0. && word_coverage >= 0.9).then(|| symbols.map(|x| x / word_weight));
        if let Some(p) = word_probabilities {
            let h = -p
                .iter()
                .filter(|&&p| p > 0.)
                .map(|p| p * p.ln())
                .sum::<f64>()
                / 9f64.ln();
            raw[50] = Some(h);
            raw[51] = Some(h * h);
        }
        let capacity_supported = self.lost_through.is_none_or(|t| t < starts[6]);
        let pairs = self.surprise.iter().filter(|s| s.end >= starts[6]);
        let pair_count = pairs.clone().count();
        raw[52] = (word_probabilities.is_some() && pair_count > 0 && capacity_supported)
            .then(|| pairs.map(|s| s.value).sum::<f64>() / pair_count as f64);
        raw[53] = (grouping_weight > 0. && grouping_coverage >= 0.9)
            .then(|| grouping_sum / grouping_weight);
        Summary {
            end_sample: end,
            raw,
            density_coverage,
            density_retention_supported: retention,
            word_coverage,
            word_probabilities,
            grouping_coverage,
            observed_coverage: density_coverage[6],
            assignment_sample_weight: assignment_weight,
            learned_pair_mass: self.counts.iter().flatten().sum(),
            retained_pairs: pair_count,
            surprise_capacity_evicted: self.capacity_evicted,
            surprise_capacity_supported: capacity_supported,
            owned_bytes: std::mem::size_of::<Self>()
                + self.frames.capacity() * std::mem::size_of::<Frame>()
                + self.surprise.capacity() * std::mem::size_of::<Surprise>(),
        }
    }
}

#[cfg(test)]
mod tests;
