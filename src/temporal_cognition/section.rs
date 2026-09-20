//! Section sufficient statistics, preserving observed support and sealed ownership.

use super::accents::Delivery;
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    Match,
    NoMemory,
    #[serde(other)]
    Unresolved,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub(super) struct Assignment {
    status: Status,
    cost: Option<f64>,
    supported: bool,
    search_completed: bool,
    search_covered: bool,
    search_nonempty: bool,
    frequency_shift_log2: Option<f64>,
    tempo_shift_log2: Option<f64>,
    #[serde(default)]
    bound_hit: bool,
    #[serde(default)]
    ambiguous_cutoff: bool,
}

// The final half-even correction keeps coverage decisions aligned with math.fsum.
fn sum(values: &[f64]) -> f64 {
    assert!(values.len() <= 32);
    let mut partials = [0.; 32];
    let mut count = 0;
    for &value in values {
        let mut x = value;
        let mut used = 0;
        for index in 0..count {
            let mut y: f64 = partials[index];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0. {
                partials[used] = lo;
                used += 1;
            }
            x = hi;
        }
        count = used;
        if x != 0. {
            partials[count] = x;
            count += 1;
        }
    }
    let mut hi = 0.;
    let mut lo = 0.;
    if count > 0 {
        count -= 1;
        hi = partials[count];
        while count > 0 {
            let x = hi;
            count -= 1;
            let y = partials[count];
            hi = x + y;
            lo = y - (hi - x);
            if lo != 0. {
                break;
            }
        }
        if count > 0 && lo.signum() == partials[count - 1].signum() {
            let y = lo * 2.;
            let x = hi + y;
            if y == x - hi {
                hi = x;
            }
        }
    }
    hi
}

fn category(
    a: Assignment,
    ending: [Option<f64>; 6],
    predecessor: Option<[Option<f64>; 6]>,
) -> Result<usize, &'static str> {
    if matches!(a.status, Status::Unresolved)
        || a.bound_hit
        || !a.supported
        || a.ambiguous_cutoff
        || !a.search_nonempty
    {
        return Ok(4);
    }
    if matches!(a.status, Status::Match) {
        let Some(cost) = a.cost else { return Ok(4) };
        if !cost.is_finite() || cost < 0. {
            return Err("invalid cached section match cost");
        }
        if cost <= 1. {
            let shifts = [a.frequency_shift_log2, a.tempo_shift_log2];
            if shifts.iter().flatten().any(|x| !x.is_finite()) {
                return Err("invalid cached section transformation");
            }
            if cost > 0.25 || shifts.iter().flatten().any(|x| x.abs() > 1. / 48.) {
                return Ok(1);
            }
            return Ok(if shifts.iter().all(Option::is_some) {
                0
            } else {
                4
            });
        }
    }
    if !a.search_completed || !a.search_covered {
        return Ok(4);
    }
    let Some(previous) = predecessor else {
        return Ok(3);
    };
    let mut differences = [0.; 6];
    let mut n = 0;
    for (value, old) in ending.into_iter().zip(previous) {
        if let (Some(value), Some(old)) = (value, old) {
            let difference = (value - old).powi(2);
            if !value.is_finite() || !old.is_finite() || !difference.is_finite() {
                return Err("invalid standardized section ending difference");
            }
            differences[n] = difference;
            n += 1;
        }
    }
    Ok(if n == 0 {
        4
    } else if (sum(&differences[..n]) / n as f64).sqrt() > 1. {
        2
    } else {
        3
    })
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub(super) struct Activity {
    pub window: [f64; 2],
    pub numerators: [f64; 9],
    pub denominators: [f64; 9],
    pub physical_valid_seconds: [f64; 9],
    pub assignment_seconds: f64,
    pub physical_window_seconds: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Record {
    statistics: [f64; 60],
    padding: [u64; 4],
    bookkeeping: [u64; 16],
}

impl Default for Record {
    fn default() -> Self {
        Self {
            statistics: [0.; 60],
            padding: [0; 4],
            bookkeeping: [0; 16],
        }
    }
}

impl Record {
    fn time(&self, slot: usize) -> f64 {
        f64::from_bits(self.bookkeeping[slot])
    }

    fn set_time(&mut self, slot: usize, value: f64) {
        self.bookkeeping[slot] = value.to_bits();
    }

    fn ending(&self) -> [Option<f64>; 6] {
        std::array::from_fn(|i| {
            (self.bookkeeping[12] & (1 << i) != 0).then_some(self.statistics[54 + i])
        })
    }

    fn add_activity(
        &mut self,
        activity: Activity,
        scale: f64,
        physical: f64,
    ) -> Result<(), &'static str> {
        if !scale.is_finite() || scale < 0. {
            return Err("invalid section activity scale");
        }
        let mut updated = self.statistics;
        for (values, offset) in [
            (&activity.numerators, 31),
            (&activity.denominators, 40),
            (&activity.physical_valid_seconds, 46),
        ] {
            let mut shared = None;
            for (i, &value) in values.iter().enumerate() {
                let index = offset + if offset == 31 { i } else { i.saturating_sub(3) };
                let total = self.statistics[index] + scale * value;
                if !value.is_finite() || value < 0. || !total.is_finite() {
                    return Err("invalid or overflowing section activity statistic");
                }
                if offset != 31 && i < 4 {
                    if shared.is_some_and(|bits| bits != total.to_bits()) {
                        return Err("articulation coordinates must share rounded support");
                    }
                    shared = Some(total.to_bits());
                }
                updated[index] = total;
            }
        }
        for (i, value) in [(52, activity.assignment_seconds), (53, physical)] {
            updated[i] = self.statistics[i] + scale * value;
            if !value.is_finite() || value < 0. || !updated[i].is_finite() {
                return Err("invalid or overflowing section physical support");
            }
        }
        self.statistics = updated;
        Ok(())
    }

    fn values(&self) -> [Option<f64>; 39] {
        let mut out = [None; 39];
        let edge = sum(&self.statistics[..5]);
        let pair = sum(&self.statistics[5..30]);
        if edge > 0. {
            for (i, value) in out[..5].iter_mut().enumerate() {
                *value = Some(self.statistics[i] / edge);
            }
        }
        if pair > 0. && pair >= 0.9 * self.statistics[30] {
            for (i, value) in out[5..30].iter_mut().enumerate() {
                *value = Some(self.statistics[5 + i] / pair);
            }
        }
        if self.statistics[53] > 0. {
            for i in 0usize..9 {
                let shared = i.saturating_sub(3);
                let denominator = self.statistics[40 + shared];
                if denominator > 0. && self.statistics[46 + shared] / self.statistics[53] >= 0.9 {
                    out[30 + i] = Some(self.statistics[31 + i] / denominator);
                }
            }
        }
        out
    }

    #[cfg(test)]
    fn bytes(&self) -> [u8; 640] {
        let mut bytes = [0; 640];
        for (chunk, value) in bytes.chunks_exact_mut(8).zip(
            self.statistics
                .iter()
                .map(|x| x.to_bits())
                .chain(self.padding)
                .chain(self.bookkeeping),
        ) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub(super) struct Commit {
    epoch: u64,
    occurrence_id: u64,
    start: f64,
    support_end: f64,
    assignment_seconds: f64,
    membership: f64,
    ordering_known: bool,
    ending_descriptor: [Option<f64>; 6],
    assignment: Assignment,
    ending_generation: Option<u64>,
}

#[derive(Debug, PartialEq)]
pub(super) struct History {
    cumulative: Record,
    ring: Vec<Record>,
    ring_size: usize,
}

impl Clone for History {
    fn clone(&self) -> Self {
        Self {
            cumulative: self.cumulative,
            ring: self.ring.clone(),
            ring_size: self.ring_size,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.cumulative = source.cumulative;
        self.ring.clone_from(&source.ring);
        self.ring_size = source.ring_size;
    }
}

impl History {
    pub fn new(
        epoch: u64,
        generation: u64,
        section: u64,
        start: f64,
        ring_size: usize,
    ) -> Result<Self, &'static str> {
        if !start.is_finite() || ![2, 4, 8].contains(&ring_size) {
            return Err("invalid section start or recent window");
        }
        let mut history = Self {
            cumulative: Record::default(),
            ring: Vec::with_capacity(ring_size),
            ring_size,
        };
        history.reset(epoch, generation, section, start)?;
        Ok(history)
    }

    fn reset(
        &mut self,
        epoch: u64,
        generation: u64,
        section: u64,
        start: f64,
    ) -> Result<(), &'static str> {
        if !start.is_finite() {
            return Err("invalid section start or recent window");
        }
        let mut cumulative = Record::default();
        cumulative.bookkeeping[..3].copy_from_slice(&[epoch, generation, section]);
        for i in 4..8 {
            cumulative.set_time(i, start);
        }
        self.cumulative = cumulative;
        self.ring.clear();
        self.ring.reserve(self.ring_size);
        Ok(())
    }

    pub fn observe(
        &mut self,
        activity: Activity,
        epoch: u64,
        generation: u64,
        rate: u32,
        deliveries: &[Delivery],
    ) -> Result<bool, &'static str> {
        if [epoch, generation] != self.cumulative.bookkeeping[..2] {
            return Ok(false);
        }
        let [start, end] = activity.window;
        if start == self.cumulative.time(6) && end == self.cumulative.time(7) {
            return Ok(false);
        }
        if rate == 0
            || !start.is_finite()
            || !end.is_finite()
            || start < self.cumulative.time(7)
            || end <= start
            || !activity.physical_window_seconds.is_finite()
            || (activity.physical_window_seconds - (end - start)).abs()
                > 1e-12_f64.max(1e-12 * (end - start).abs())
        {
            return Err("canonical section delta must retain full physical extent");
        }
        let mut candidate = self.cumulative;
        if !deliveries.is_empty() && activity.numerators[4] != 0. {
            return Err("accent deliveries require an accent-free acoustic delta");
        }
        let rate = f64::from(rate);
        let mut last = 0;
        for delivery in deliveries {
            let accent = delivery.accent;
            let received = delivery.received_at as f64 / rate;
            if delivery.sequence <= last
                || accent.group.epoch != epoch
                || accent.group.generation != generation
                || received < start
                || received > end
                || accent.source_end as f64 / rate > end
                || accent.available_end as f64 / rate > end
                || accent.event_start >= accent.event_end
                || !accent.weight.is_finite()
                || !(0. ..=1.).contains(&accent.weight)
            {
                return Err("invalid generation-local accent delivery");
            }
            last = delivery.sequence;
            if delivery.sequence > candidate.bookkeeping[15] {
                candidate.bookkeeping[15] = delivery.sequence;
                if accent.event_end as f64 / rate >= candidate.time(4) {
                    candidate.statistics[35] += accent.weight;
                    if !candidate.statistics[35].is_finite() {
                        return Err("section accent weight overflow");
                    }
                }
            }
        }
        candidate.add_activity(activity, 1., end - candidate.time(7))?;
        candidate.set_time(5, end);
        candidate.set_time(6, start);
        candidate.set_time(7, end);
        self.cumulative = candidate;
        Ok(true)
    }

    pub fn commit(
        &mut self,
        record: Commit,
        activity: Activity,
        sequence: u64,
        adjacency: f64,
    ) -> Result<bool, &'static str> {
        if record.epoch != self.cumulative.bookkeeping[0] {
            return Ok(false);
        }
        if sequence == 0 || !adjacency.is_finite() || !(0. ..=1.).contains(&adjacency) {
            return Err("invalid section sequence or adjacency coverage");
        }
        let key = (record.support_end, record.start, record.occurrence_id);
        if sequence <= self.cumulative.bookkeeping[13] {
            if self.ring.iter().any(|r| {
                r.bookkeeping[13] == sequence && (r.time(5), r.time(4), r.bookkeeping[3]) == key
            }) {
                return Ok(false);
            }
            return Err("old or conflicting section replay requires reconciliation");
        }
        let previous = (
            self.cumulative.time(9),
            self.cumulative.time(8),
            self.cumulative.bookkeeping[10],
        );
        let seconds = record.assignment_seconds;
        let membership = record.membership;
        if !record.start.is_finite()
            || !record.support_end.is_finite()
            || (self.cumulative.bookkeeping[11] != 0 && key < previous)
            || record.support_end > self.cumulative.time(5)
            || record.start > record.support_end
            || !membership.is_finite()
            || !(0. ..=1.).contains(&membership)
            || !seconds.is_finite()
            || seconds < 0.
            || seconds > record.support_end - record.start + 1e-12
            || (seconds - activity.assignment_seconds).abs()
                > 1e-12_f64.max(1e-9 * seconds.abs().max(activity.assignment_seconds.abs()))
            || activity.window != [record.start, record.support_end]
            || record
                .ending_descriptor
                .iter()
                .flatten()
                .any(|x| !x.is_finite())
        {
            return Err("section commit must preserve original support and membership");
        }
        let weight = membership * seconds;
        if weight == 0. {
            return Ok(false);
        }
        let c = category(
            record.assignment,
            record.ending_descriptor,
            (self.cumulative.bookkeeping[11] != 0).then(|| self.cumulative.ending()),
        )?;
        let mut item = Record::default();
        item.statistics[c] = weight;
        item.add_activity(activity, membership, activity.physical_window_seconds)?;
        for (i, v) in record.ending_descriptor.into_iter().enumerate() {
            if let Some(v) = v {
                item.statistics[54 + i] = v;
                item.bookkeeping[12] |= 1 << i;
            }
        }
        item.bookkeeping[..4].copy_from_slice(&[
            record.epoch,
            record
                .ending_generation
                .unwrap_or(self.cumulative.bookkeeping[1]),
            self.cumulative.bookkeeping[2],
            record.occurrence_id,
        ]);
        item.bookkeeping[13] = sequence;
        item.bookkeeping[14] = u64::from(record.ordering_known);
        item.set_time(4, record.start);
        item.set_time(5, record.support_end);
        let mut cumulative = self.cumulative;
        let append = record.start >= cumulative.time(4);
        if append {
            if let Some(prior) = self.ring.last() {
                let pair = sum(&prior.statistics[..5]).min(weight);
                item.statistics[30] = pair;
                if record.ordering_known && prior.bookkeeping[14] != 0 && adjacency >= 0.9 {
                    let category = (0..5)
                        .max_by(|&a, &b| prior.statistics[a].total_cmp(&prior.statistics[b]))
                        .unwrap();
                    item.statistics[5 + category * 5 + c] = pair;
                }
            }
            for i in 0..31 {
                cumulative.statistics[i] += item.statistics[i];
                if !cumulative.statistics[i].is_finite() {
                    return Err("overflowing section correspondence statistic");
                }
            }
        }
        cumulative.statistics[54..].copy_from_slice(&item.statistics[54..]);
        cumulative.bookkeeping[12] = item.bookkeeping[12];
        cumulative.bookkeeping[3] = item.bookkeeping[1];
        cumulative.bookkeeping[11] = 1;
        cumulative.bookkeeping[10] = record.occurrence_id;
        cumulative.bookkeeping[13] = sequence;
        cumulative.set_time(8, record.start);
        cumulative.set_time(9, record.support_end);
        self.cumulative = cumulative;
        if append {
            if self.ring.len() == self.ring_size {
                self.ring.remove(0);
            }
            self.ring.push(item);
        }
        Ok(true)
    }

    #[cfg(test)]
    pub fn inherit(&self, old: u64, new: u64) -> Result<Self, &'static str> {
        if old != self.cumulative.bookkeeping[1] || old == new {
            return Err("section continuation must name predecessor generation");
        }
        let mut child = Self::new(
            self.cumulative.bookkeeping[0],
            new,
            self.cumulative.bookkeeping[2],
            self.cumulative.time(4),
            self.ring_size,
        )?;
        child.cumulative = self.cumulative;
        child.cumulative.bookkeeping[1] = new;
        child.cumulative.bookkeeping[15] = 0;
        child.ring.extend_from_slice(&self.ring);
        Ok(child)
    }

    pub fn recent(&self) -> Record {
        let mut recent = Record::default();
        for (n, record) in self.ring.iter().enumerate() {
            for i in 0..54 {
                if n != 0 || !(5..31).contains(&i) {
                    recent.statistics[i] += record.statistics[i];
                }
            }
        }
        recent
    }

    fn admit_pending(&mut self, delivery: Delivery, rate: u32) -> f64 {
        let event = delivery.accent.event_end as f64 / f64::from(rate);
        let received = delivery.received_at as f64 / f64::from(rate);
        let Some(record) = self
            .ring
            .iter_mut()
            .find(|r| r.time(4) < event && event <= r.time(5))
        else {
            return 0.;
        };
        if delivery.sequence <= record.bookkeeping[15] {
            return 0.;
        }
        record.bookkeeping[15] = delivery.sequence;
        if received > record.time(5) + 0.5 {
            return delivery.accent.weight;
        }
        // Cumulative events use their independent canonical delivery watermark.
        record.statistics[35] += delivery.accent.weight;
        0.
    }

    pub fn covariates(
        &self,
        observed: f64,
        retrieval: [Option<f64>; 2],
    ) -> Result<[Option<f64>; 82], &'static str> {
        if !observed.is_finite()
            || observed < self.cumulative.time(5)
            || retrieval.iter().flatten().any(|x| !x.is_finite())
        {
            return Err("section head cannot precede its observations");
        }
        let cumulative = self.cumulative.values();
        let recent = self.recent().values();
        let mut result = [None; 82];
        result[..39].copy_from_slice(&cumulative);
        for i in 0..39 {
            result[39 + i] = cumulative[i].zip(recent[i]).map(|(c, r)| r - c);
        }
        let elapsed = observed - self.cumulative.time(4);
        if elapsed > 0. {
            result[78] = Some(elapsed.ln_1p());
            result[79] = Some((1. - self.cumulative.statistics[47] / elapsed).clamp(0., 1.));
        }
        result[80..].copy_from_slice(&retrieval);
        Ok(result)
    }
}

// The receipt owner must validate the selected cue before supplying its acoustic scores.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Retrieval {
    pub values: [Option<f64>; 2],
    pub targets: [Option<super::transport::Identity>; 2],
}

pub(super) fn retrieval_scores(
    candidates: impl IntoIterator<Item = (super::transport::Identity, Option<f64>)>,
    support_end: Option<u64>,
    query_end: u64,
    observed_end: u64,
    rate: u32,
) -> Result<Retrieval, &'static str> {
    if rate == 0 || query_end > observed_end || support_end.is_some_and(|t| t > query_end) {
        return Err("invalid reused section query clock or transformation inventory");
    }
    let current = support_end
        .is_some_and(|support| ((observed_end - support) as f64) < f64::from(rate) * 0.5);
    // Only two distinct episode maxima are needed, regardless of search capacity.
    let mut best: [Option<(super::transport::Identity, f64)>; 2] = [None; 2];
    for (index, (identity, score)) in candidates.into_iter().enumerate() {
        if index >= 4 * super::memory::MAX_CANDIDATES {
            return Err("invalid reused section query clock or transformation inventory");
        }
        if !current {
            continue;
        }
        let Some(score) = score else { continue };
        if !score.is_finite() {
            return Err("nonfinite cached acoustic retrieval score");
        }
        if let Some(old) = best.iter_mut().flatten().find(|(id, _)| *id == identity) {
            old.1 = old.1.max(score);
        } else if best[1].is_none_or(|(_, second)| score > second) {
            best[1] = Some((identity, score));
        }
        if best[0].is_none_or(|(_, first)| best[1].is_some_and(|(_, second)| second > first)) {
            best.swap(0, 1);
        }
    }
    let gap = best[0]
        .zip(best[1])
        .map(|((_, first), (_, second))| first - second);
    if gap.is_some_and(|v| !v.is_finite()) {
        return Err("section retrieval ambiguity overflow");
    }
    Ok(Retrieval {
        values: [best[0].map(|(_, score)| score), gap],
        targets: best.map(|entry| entry.map(|(id, _)| id)),
    })
}

pub(super) struct Head {
    pub means: [f64; 82],
    pub deviations: [f64; 82],
    pub hazard: [f64; 83],
    pub exits: [[f64; 83]; 3],
}

impl Head {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self
            .means
            .iter()
            .chain(&self.deviations)
            .chain(&self.hazard)
            .chain(self.exits.iter().flatten())
            .any(|x| !x.is_finite())
            || self.deviations.iter().any(|x| *x < 0.)
            || (0..83).any(|i| {
                let column = self.exits.map(|row| row[i]);
                sum(&column).abs()
                    > 1e-10 * (1. + column.iter().map(|x| x.abs()).fold(0., f64::max))
            })
        {
            return Err("finite section scales and coefficient-wise centered exit logits required");
        }
        Ok(())
    }

    pub fn law(
        &self,
        raw: &[Option<f64>; 82],
        lo: f64,
        hi: f64,
    ) -> Result<Option<[f64; 4]>, &'static str> {
        if raw.iter().flatten().any(|x| !x.is_finite())
            || !lo.is_finite()
            || !hi.is_finite()
            || lo < 0.
            || hi < lo
        {
            return Err("invalid section head evidence or forecast duration");
        }
        if raw[78].is_none() {
            return Ok(None);
        }
        let mut x = [0.; 83];
        x[0] = 1.;
        for i in 0..82 {
            if let Some(value) = raw[i] {
                x[i + 1] = (value - self.means[i]) / self.deviations[i].max(1e-6);
            }
        }
        // Remove issue-time elapsed before advancing only that frozen covariate.
        x[79] = -self.means[78] / self.deviations[78].max(1e-6);
        let intercept: f64 = x.iter().zip(self.hazard).map(|(a, b)| a * b).sum();
        let slope = self.hazard[79] / self.deviations[78].max(1e-6);
        let Some(integral) = super::hazard::integrate(intercept, slope, lo, hi) else {
            return Ok(None);
        };
        x[79] = (hi.ln_1p() - self.means[78]) / self.deviations[78].max(1e-6);
        let logits = self
            .exits
            .map(|row| x.iter().zip(row).map(|(a, b)| a * b).sum::<f64>());
        if logits.iter().any(|x| !x.is_finite()) {
            return Ok(None);
        }
        let max = logits.into_iter().fold(f64::NEG_INFINITY, f64::max);
        let scores = logits.map(|x| (x - max).exp());
        let denominator = sum(&scores);
        let mut out = [0.; 4];
        out[0] = (-integral).exp();
        let exited = -(-integral).exp_m1();
        for i in 0..3 {
            out[i + 1] = exited * scores[i] / denominator;
        }
        Ok(Some(out))
    }
}

#[cfg(test)]
pub(in crate::temporal_cognition) mod tests;

pub(in crate::temporal_cognition) mod commitment;
pub(crate) mod cue;
pub(in crate::temporal_cognition) mod form;
pub(in crate::temporal_cognition) mod input;
pub(in crate::temporal_cognition) mod interpretation;
mod runtime;
pub(crate) use runtime::{Snapshot, Stream};
