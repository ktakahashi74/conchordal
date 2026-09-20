//! Original-time retention with a shared acquisition clock and bounded uncertainty.

use super::clock::Clock;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct RateSnapshot {
    pub envelope_invalid: bool,
    pub envelope_unverified: bool,
    pub rate_history_unknown: bool,
    pub largest_retained_one_second_increment: f64,
    pub overflow_count: u64,
}

#[derive(Clone, Copy, Default)]
struct Row {
    time: f64,
    sequence: u64,
}

pub(super) struct RateWindow {
    rows: Box<[Row]>,
    values: Box<[f64]>,
    totals: Box<[f64]>,
    head: usize,
    count: usize,
    losses: u64,
    lost_until: f64,
    last_time: f64,
    invalid: bool,
    unverified: bool,
}

impl RateWindow {
    pub fn new(slots: usize, capacity: usize) -> Result<Self, &'static str> {
        let length = slots
            .checked_mul(capacity)
            .filter(|n| *n > 0)
            .ok_or("positive bounded rate-window dimensions required")?;
        Ok(Self {
            rows: vec![Row::default(); capacity].into_boxed_slice(),
            values: vec![0.; length].into_boxed_slice(),
            totals: vec![0.; slots].into_boxed_slice(),
            head: 0,
            count: 0,
            losses: 0,
            lost_until: f64::NEG_INFINITY,
            last_time: f64::NEG_INFINITY,
            invalid: false,
            unverified: false,
        })
    }

    pub fn observe(
        &mut self,
        time: f64,
        sequence: u64,
        increments: &[f64],
        births: &[u64],
        r_max: f64,
    ) -> Result<RateSnapshot, &'static str> {
        if !time.is_finite()
            || time < self.last_time
            || sequence == 0
            || increments.len() != self.totals.len()
            || births.len() != self.totals.len()
            || increments.iter().any(|v| !v.is_finite() || *v < 0.)
            || !r_max.is_finite()
            || r_max <= 0.
        {
            return Err("ordered finite slot-aligned observed rate increments required");
        }
        let positive = increments.iter().any(|x| *x > 0.);
        while self.count > 0 {
            let at = (self.head + self.rows.len() - self.count) % self.rows.len();
            let row = self.rows[at];
            let expired = row.time <= time - 1.;
            if !(expired || positive && self.count == self.rows.len()) {
                break;
            }
            if !expired {
                self.lost_until = self.lost_until.max(row.time + 1.);
                self.losses += 1;
                self.unverified = true;
            }
            for (slot, (&birth, total)) in births.iter().zip(self.totals.iter_mut()).enumerate() {
                if birth > 0 && row.sequence >= birth {
                    *total = (*total - self.values[at * births.len() + slot]).max(0.);
                }
            }
            self.count -= 1;
        }
        if positive {
            self.rows[self.head] = Row { time, sequence };
            let start = self.head * increments.len();
            self.values[start..start + increments.len()].copy_from_slice(increments);
            for (total, increment) in self.totals.iter_mut().zip(increments) {
                *total += increment;
            }
            self.head = (self.head + 1) % self.rows.len();
            self.count += 1;
        }
        self.last_time = time;
        self.invalid |= self.totals.iter().any(|v| *v > r_max);
        Ok(self.snapshot())
    }

    pub fn snapshot(&self) -> RateSnapshot {
        RateSnapshot {
            envelope_invalid: self.invalid,
            envelope_unverified: self.unverified,
            rate_history_unknown: self.last_time < self.lost_until,
            largest_retained_one_second_increment: self.totals.iter().copied().fold(0., f64::max),
            overflow_count: self.losses,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub(crate) struct Record {
    pub handle: u64,
    pub birth_sequence: u64,
    pub strength: f64,
    pub first_observed_end: f64,
    pub last_observed_end: f64,
    pub interference_lower: f64,
    pub interference_upper: f64,
    pub gap_origin_lower: f64,
    pub gap_origin_upper: f64,
    pub membership_total: f64,
    pub first_committed_at: f64,
    pub last_delivered_at: f64,
    pub occurrence_id: u64,
    pub support_id: u64,
}

#[derive(Clone, Copy)]
pub(super) struct Parameters {
    pub tau: f64,
    pub kappa: f64,
    pub strength_max: f64,
    pub r_max: f64,
}

pub(super) struct Coarse<'a> {
    pub epoch: u64,
    pub generation: u64,
    pub query_id: u64,
    pub occurrence_id: u64,
    pub support_id: u64,
    pub support_end: f64,
    pub supporting_audio_end: Option<f64>,
    pub available_end: f64,
    pub entries: &'a [(u64, Option<f64>)],
}

pub(super) struct Write<'a> {
    pub epoch: u64,
    pub sequence: u64,
    pub occurrence_id: u64,
    pub support_id: u64,
    pub start: f64,
    pub end: f64,
    pub committed: f64,
    pub delivered: f64,
    pub available: f64,
    pub unknown: f64,
    pub unassigned: f64,
    pub assignments: &'a [(u64, f64)],
    pub coarse: Option<Coarse<'a>>,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Availability {
    pub handle: u64,
    pub log_lower: f64,
    pub log_upper: f64,
    pub elapsed_sec: f64,
    pub strength: f64,
    pub observed_interference: f64,
    pub coarse_unknown_interference: f64,
    pub missing_seconds_lower: f64,
    pub missing_seconds_upper: f64,
    pub gap_interference_upper: f64,
    pub envelope_invalid: bool,
    pub envelope_unverified: bool,
    pub rate_history_unknown: bool,
    pub clock_history_lost: bool,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Recognition {
    pub lower: f64,
    pub upper: f64,
    pub supported_matches: usize,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Snapshot {
    pub sequence: u64,
    pub original_end_sec: f64,
    pub evaluated_at_sec: f64,
    pub records: usize,
    pub evictions: u64,
    pub replays: u64,
    pub owned_bytes: usize,
    pub rate: RateSnapshot,
    pub latest_record: Option<Record>,
    pub latest_availability: Option<Availability>,
}

fn probability(weights: impl Iterator<Item = (f64, f64)> + Clone, bias: f64) -> f64 {
    let scores = weights
        .filter(|(availability, _)| *availability != f64::NEG_INFINITY)
        .map(|(availability, acoustic)| {
            let floor = 1e-300_f64.ln();
            let upper = availability.max(floor);
            upper + (availability.min(floor) - upper).exp().ln_1p() + acoustic
        });
    let maximum = scores.clone().fold(bias, f64::max);
    let evidence: f64 = scores.map(|s| (s - maximum).exp()).sum();
    evidence / (evidence + (bias - maximum).exp())
}

#[cfg(test)]
pub(super) struct Assay {
    epoch: u64,
    start: f64,
    tau: f64,
    rows: Box<[(u64, f64, f64)]>,
}

#[cfg(test)]
impl Assay {
    pub fn recognition(
        &self,
        epoch: u64,
        scores: &[(u64, f64)],
        bias: f64,
        query_end: f64,
    ) -> Result<Recognition, &'static str> {
        if epoch != self.epoch
            || !query_end.is_finite()
            || query_end < self.start
            || !bias.is_finite()
            || scores.iter().any(|(_, s)| !s.is_finite())
            || scores.windows(2).any(|p| p[0].0 >= p[1].0)
        {
            return Err(
                "same-epoch frozen assay, heard endpoint and sorted finite match scores required",
            );
        }
        let elapsed = (query_end - self.start) / self.tau;
        let matched = self.rows.iter().filter_map(|&(handle, lo, hi)| {
            scores
                .binary_search_by_key(&handle, |(h, _)| *h)
                .ok()
                .map(|i| (lo - elapsed, hi - elapsed, scores[i].1))
        });
        Ok(Recognition {
            lower: probability(matched.clone().map(|(lo, _, score)| (lo, score)), bias),
            upper: probability(matched.clone().map(|(_, hi, score)| (hi, score)), bias),
            supported_matches: matched.count(),
        })
    }
}

pub(super) struct Retention {
    parameters: Parameters,
    epoch: u64,
    pub(super) records: Box<[Record]>,
    scratch: Box<[Record]>,
    births: Box<[u64]>,
    increments: Box<[f64]>,
    window: RateWindow,
    sequence: u64,
    last_end: f64,
    cut: f64,
    last_write: Vec<u8>,
    write_scratch: Vec<u8>,
    score_scratch: Vec<(f64, f64, f64)>,
    evictions: u64,
    replays: u64,
}

impl Retention {
    pub fn snapshot(&self, clock: &Clock, time: f64) -> Result<Snapshot, &'static str> {
        let latest = self
            .records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.handle != 0)
            .max_by_key(|(_, r)| r.birth_sequence);
        Ok(Snapshot {
            sequence: self.sequence,
            original_end_sec: self.last_end,
            evaluated_at_sec: time,
            records: self.records.iter().filter(|r| r.handle != 0).count(),
            evictions: self.evictions,
            replays: self.replays,
            owned_bytes: std::mem::size_of::<Self>()
                + (self.records.len() + self.scratch.len()) * std::mem::size_of::<Record>()
                + (self.births.len()
                    + self.increments.len()
                    + self.window.totals.len()
                    + self.window.values.len())
                    * 8
                + self.window.rows.len() * std::mem::size_of::<Row>()
                + self.last_write.capacity()
                + self.write_scratch.capacity()
                + self.score_scratch.capacity() * std::mem::size_of::<(f64, f64, f64)>(),
            rate: self.window.snapshot(),
            latest_record: latest.map(|(_, r)| *r),
            latest_availability: latest
                .map(|(i, _)| self.availability(clock, i, time))
                .transpose()?
                .flatten(),
        })
    }
    pub fn new(
        clock: &Clock,
        parameters: Parameters,
        capacity: usize,
        rate_capacity: usize,
    ) -> Result<Self, &'static str> {
        let Parameters {
            tau,
            kappa,
            strength_max,
            r_max,
        } = parameters;
        if capacity == 0
            || capacity > super::memory::MAX_EPISODES
            || [tau, kappa, strength_max, r_max]
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.)
            || strength_max < 1.
        {
            return Err(
                "positive retention parameters, strength cap >= 1 and bounded capacity required",
            );
        }
        let (epoch, origin, _, _) = clock.context();
        let receipt_bytes = 256 + (capacity + 1) * 40;
        Ok(Self {
            parameters,
            epoch,
            records: vec![Record::default(); capacity].into_boxed_slice(),
            scratch: vec![Record::default(); capacity].into_boxed_slice(),
            births: vec![0; capacity].into_boxed_slice(),
            increments: vec![0.; capacity].into_boxed_slice(),
            window: RateWindow::new(capacity, rate_capacity)?,
            sequence: 0,
            last_end: origin,
            cut: f64::NEG_INFINITY,
            last_write: Vec::with_capacity(receipt_bytes),
            write_scratch: Vec::with_capacity(receipt_bytes),
            score_scratch: Vec::with_capacity(capacity),
            evictions: 0,
            replays: 0,
        })
    }

    pub fn apply(
        &mut self,
        clock: &Clock,
        w: &Write<'_>,
        cut: f64,
        new_handles: &[u64],
    ) -> Result<bool, &'static str> {
        let (epoch, origin, clock_end, clock_cut) = clock.context();
        if epoch != self.epoch
            || w.epoch != epoch
            || !cut.is_finite()
            || cut < self.cut.max(clock_cut)
            || [w.start, w.end, w.committed, w.delivered]
                .iter()
                .any(|v| !v.is_finite())
            || !(origin <= w.start
                && w.start < w.end
                && w.end <= w.committed
                && w.committed <= w.delivered
                && w.delivered <= cut
                && w.end <= clock_end)
        {
            return Err("current-epoch sealed original support and causal delivery cut required");
        }
        if w.assignments.len() > self.records.len() + 1
            || w.assignments
                .iter()
                .any(|(h, v)| *h == 0 || !v.is_finite() || *v <= 0.)
            || w.assignments.windows(2).any(|p| p[0].0 >= p[1].0)
            || [w.available, w.unknown, w.unassigned]
                .iter()
                .any(|v| !v.is_finite() || !(0. ..=1.).contains(v))
            || (w.assignments.iter().map(|(_, v)| v).sum::<f64>() + w.unknown + w.unassigned
                - w.available)
                .abs()
                > 1e-12
        {
            return Err("bounded canonical sorted assignments with conserved support required");
        }
        if let Some(c) = &w.coarse
            && (c.entries.len() > self.records.len() + 1
                || c.entries.windows(2).any(|p| p[0].0 >= p[1].0)
                || c.entries
                    .iter()
                    .any(|(h, v)| *h == 0 || v.is_some_and(|v| !v.is_finite() || v < 0.))
                || [c.support_end, c.available_end]
                    .iter()
                    .any(|v| !v.is_finite())
                || c.supporting_audio_end.is_some_and(|v| !v.is_finite()))
        {
            return Err("bounded canonical cached coarse receipt required");
        }
        self.write_scratch.clear();
        for value in [
            w.sequence,
            w.occurrence_id,
            w.support_id,
            w.start.to_bits(),
            w.end.to_bits(),
            w.committed.to_bits(),
            w.delivered.to_bits(),
            w.unknown.to_bits(),
            w.unassigned.to_bits(),
            w.assignments.len() as u64,
        ] {
            self.write_scratch.extend_from_slice(&value.to_le_bytes());
        }
        for &(handle, value) in w.assignments {
            self.write_scratch.extend_from_slice(&handle.to_le_bytes());
            self.write_scratch
                .extend_from_slice(&value.to_bits().to_le_bytes());
        }
        self.write_scratch.push(u8::from(w.coarse.is_some()));
        if let Some(c) = &w.coarse {
            for value in [
                c.epoch,
                c.generation,
                c.query_id,
                c.occurrence_id,
                c.support_id,
                c.support_end.to_bits(),
                c.available_end.to_bits(),
                u64::from(c.supporting_audio_end.is_some()),
                c.supporting_audio_end.unwrap_or(0.).to_bits(),
                c.entries.len() as u64,
            ] {
                self.write_scratch.extend_from_slice(&value.to_le_bytes());
            }
            for &(handle, cost) in c.entries {
                self.write_scratch.extend_from_slice(&handle.to_le_bytes());
                self.write_scratch.push(u8::from(cost.is_some()));
                self.write_scratch
                    .extend_from_slice(&cost.unwrap_or(0.).to_bits().to_le_bytes());
            }
        }
        if w.sequence <= self.sequence {
            if w.sequence == self.sequence && self.write_scratch != self.last_write {
                return Err("conflicting repeated sealed write");
            }
            self.replays += 1;
            return Ok(false);
        }
        if self.sequence.checked_add(1) != Some(w.sequence) || w.end < self.last_end {
            return Err("complete chronological sealed ledger sequence required");
        }
        if new_handles.windows(2).any(|p| p[0] >= p[1])
            || new_handles.iter().any(|h| {
                self.records.iter().any(|r| r.handle == *h)
                    || w.assignments.binary_search_by_key(h, |(h, _)| *h).is_err()
            })
            || self.records.iter().filter(|r| r.handle == 0).count() < new_handles.len()
        {
            return Err(
                "explicit fresh positive-support admission after complete bank eviction required",
            );
        }
        let prefix = clock.prefix(w.end)?;
        let start_prefix = clock.prefix(w.start)?;
        if w.available
            > 1. - (prefix.lower - start_prefix.upper).max(0.) / (w.end - w.start) + 1e-12
        {
            return Err("sealed support contradicts known acquisition loss");
        }
        self.scratch.copy_from_slice(&self.records);
        self.increments.fill(0.);
        for (slot, r) in self
            .scratch
            .iter_mut()
            .enumerate()
            .filter(|(_, r)| r.handle != 0)
        {
            let recurrence = w
                .assignments
                .binary_search_by_key(&r.handle, |(h, _)| *h)
                .map_or(0., |i| w.assignments[i].1);
            let competing: f64 = w
                .assignments
                .iter()
                .filter(|(h, _)| *h != r.handle)
                .map(|(_, v)| v)
                .sum();
            let uncertain = w.unknown + w.unassigned;
            let cost = w.coarse.as_ref().and_then(|c| {
                c.entries
                    .binary_search_by_key(&r.handle, |(h, _)| *h)
                    .ok()
                    .and_then(|i| c.entries[i].1)
            });
            let (lower, upper) = if r.first_observed_end >= w.end {
                (0., 0.)
            } else if let Some(cost) = cost {
                let similarity = (-cost).exp();
                (competing * similarity, (competing + uncertain) * similarity)
            } else {
                (0., competing + uncertain)
            };
            if recurrence > 0. {
                r.strength = self.parameters.strength_max.min(r.strength + recurrence);
                r.membership_total += recurrence;
                r.last_observed_end = w.end;
                r.interference_lower = 0.;
                r.interference_upper = 0.;
                r.gap_origin_lower = prefix.lower;
                r.gap_origin_upper = prefix.upper;
                r.occurrence_id = w.occurrence_id;
                r.support_id = w.support_id;
            }
            r.interference_lower += lower;
            r.interference_upper += upper;
            r.last_delivered_at = w.delivered;
            self.increments[slot] = lower;
        }
        for (r, handle) in self
            .scratch
            .iter_mut()
            .filter(|r| r.handle == 0)
            .zip(new_handles)
        {
            let weight = w.assignments[w
                .assignments
                .binary_search_by_key(handle, |(h, _)| *h)
                .unwrap()]
            .1;
            *r = Record {
                handle: *handle,
                birth_sequence: w.sequence,
                strength: weight,
                first_observed_end: w.end,
                last_observed_end: w.end,
                gap_origin_lower: prefix.lower,
                gap_origin_upper: prefix.upper,
                membership_total: weight,
                first_committed_at: w.committed,
                last_delivered_at: w.delivered,
                occurrence_id: w.occurrence_id,
                support_id: w.support_id,
                ..Record::default()
            };
        }
        // No live state changes before all write validation and metadata staging complete.
        for (slot, r) in self.scratch.iter().enumerate() {
            if r.birth_sequence == w.sequence {
                self.births[slot] = w.sequence;
                self.window.totals[slot] = 0.;
            }
        }
        self.window
            .observe(
                w.end,
                w.sequence,
                &self.increments,
                &self.births,
                self.parameters.r_max,
            )
            .expect("validated chronological finite increments");
        std::mem::swap(&mut self.records, &mut self.scratch);
        std::mem::swap(&mut self.last_write, &mut self.write_scratch);
        self.sequence = w.sequence;
        self.last_end = w.end;
        self.cut = cut;
        Ok(true)
    }

    pub fn availability(
        &self,
        clock: &Clock,
        slot: usize,
        time: f64,
    ) -> Result<Option<Availability>, &'static str> {
        let r = self
            .records
            .get(slot)
            .ok_or("retained metadata slot required")?;
        if r.handle == 0 {
            return Ok(None);
        }
        let (epoch, _, _, cut) = clock.context();
        if epoch != self.epoch
            || !time.is_finite()
            || time < self.cut.max(cut)
            || time < r.last_observed_end
        {
            return Err("live availability cannot precede the current clock or delivered update");
        }
        let now = clock.prefix(time)?;
        let elapsed = time - r.last_observed_end;
        let missing_max = elapsed.min((now.upper - r.gap_origin_lower).max(0.));
        let missing_min = (now.lower - r.gap_origin_upper).max(0.);
        let uncertain_rate = self.window.invalid || self.window.unverified;
        let base = r.strength.ln() - elapsed / self.parameters.tau;
        Ok(Some(Availability {
            handle: r.handle,
            log_lower: if missing_max > 0. && uncertain_rate {
                f64::NEG_INFINITY
            } else {
                base - (r.interference_upper + self.parameters.r_max * missing_max)
                    / self.parameters.kappa
            },
            log_upper: base - r.interference_lower / self.parameters.kappa,
            elapsed_sec: elapsed,
            strength: r.strength,
            observed_interference: r.interference_lower,
            coarse_unknown_interference: r.interference_upper - r.interference_lower,
            missing_seconds_lower: missing_min,
            missing_seconds_upper: missing_max,
            gap_interference_upper: if missing_max > 0. && uncertain_rate {
                f64::INFINITY
            } else {
                self.parameters.r_max * missing_max
            },
            envelope_invalid: self.window.invalid,
            envelope_unverified: self.window.unverified,
            rate_history_unknown: self.last_end < self.window.lost_until,
            clock_history_lost: r.gap_origin_lower != r.gap_origin_upper || now.history_lost,
        }))
    }

    pub fn recognition(
        &mut self,
        clock: &Clock,
        scores: &[(u64, f64)],
        bias: f64,
        time: f64,
    ) -> Result<Recognition, &'static str> {
        let (epoch, _, end, cut) = clock.context();
        if epoch != self.epoch
            || !time.is_finite()
            || time < self.cut.max(cut)
            || time > end
            || !bias.is_finite()
            || scores.iter().any(|(_, s)| !s.is_finite())
            || scores.windows(2).any(|p| p[0].0 >= p[1].0)
        {
            return Err("sorted finite match scores and no-memory bias required");
        }
        self.score_scratch.clear();
        for (slot, r) in self
            .records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.handle != 0)
        {
            if let Ok(i) = scores.binary_search_by_key(&r.handle, |(h, _)| *h) {
                let a = self.availability(clock, slot, time)?.unwrap();
                self.score_scratch
                    .push((a.log_lower, a.log_upper, scores[i].1));
            }
        }
        Ok(Recognition {
            lower: probability(
                self.score_scratch.iter().map(|&(lo, _, score)| (lo, score)),
                bias,
            ),
            upper: probability(
                self.score_scratch.iter().map(|&(_, hi, score)| (hi, score)),
                bias,
            ),
            supported_matches: self.score_scratch.len(),
        })
    }

    #[cfg(test)]
    pub fn assay_snapshot(&self, clock: &Clock, target_start: f64) -> Result<Assay, &'static str> {
        let (epoch, _, _, cut) = clock.context();
        if epoch != self.epoch || !target_start.is_finite() || target_start < self.cut.max(cut) {
            return Err("assay must be frozen before hearing the target at the live delivery cut");
        }
        let mut rows = Vec::with_capacity(self.records.len());
        for (slot, r) in self
            .records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.handle != 0 && r.last_observed_end < target_start)
        {
            let a = self.availability(clock, slot, target_start)?.unwrap();
            rows.push((r.handle, a.log_lower, a.log_upper));
        }
        Ok(Assay {
            epoch,
            start: target_start,
            tau: self.parameters.tau,
            rows: rows.into_boxed_slice(),
        })
    }

    pub fn eviction_candidate(
        &self,
        clock: &Clock,
        time: f64,
    ) -> Result<Option<usize>, &'static str> {
        let mut best = None;
        for (slot, r) in self
            .records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.handle != 0)
        {
            let score = self.availability(clock, slot, time)?.unwrap().log_upper;
            let key = (score, r.first_committed_at, r.handle);
            if best.is_none_or(|(_, previous)| key < previous) {
                best = Some((slot, key));
            }
        }
        Ok(best.map(|(slot, _)| slot))
    }

    pub fn evict(&mut self, slot: usize) -> Result<Option<u64>, &'static str> {
        let r = self
            .records
            .get_mut(slot)
            .ok_or("retained metadata slot required")?;
        if r.handle == 0 {
            return Ok(None);
        }
        let handle = r.handle;
        *r = Record::default();
        self.births[slot] = 0;
        self.window.totals[slot] = 0.;
        self.evictions += 1;
        Ok(Some(handle))
    }
}

#[cfg(test)]
mod tests;
