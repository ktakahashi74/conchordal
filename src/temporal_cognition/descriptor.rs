//! Fixed-payload moment compression with original acquisition support and span cuts.

use super::{features::RawDescriptor, matcher, memory, ridge::Handle};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Block {
    moments: [[f64; 3]; 10],
    time: f64,
    observed: f64,
    start: f64,
    end: f64,
    raw_start: f64,
    raw_end: f64,
    available: f64,
    missing: f64,
    epoch: u64,
    generation: u64,
}

impl Block {
    fn from_raw(
        raw: &RawDescriptor,
        rate: u32,
        intervals: &[(u64, u64)],
        start: f64,
        end: f64,
    ) -> Result<Self, &'static str> {
        let rate = f64::from(rate);
        if !(rate > 0.
            && raw.source_start <= raw.start
            && raw.start < raw.end
            && raw.end <= raw.source_end
            && raw.source_end <= raw.available_end)
            || !(raw.start as f64 / rate <= start && start < end && end <= raw.end as f64 / rate)
            || raw.values.iter().flatten().any(|v| !v.is_finite())
        {
            return Err("invalid clipped raw descriptor support or values");
        }
        let mut acquired = 0u64;
        let mut last = None;
        let mut observed = 0.;
        for &(a, b) in intervals {
            if !(raw.start <= a && a < b && b <= raw.end) || last.is_some_and(|old| a <= old) {
                return Err("acquired intervals must be ordered, disjoint and coalesced");
            }
            acquired += b - a;
            last = Some(b);
            observed += (end.min(b as f64 / rate) - start.max(a as f64 / rate)).max(0.);
        }
        if acquired != raw.known_samples {
            return Err("acquisition locations disagree with known sample count");
        }
        let mut out = Self {
            time: if observed > 0. {
                raw.end as f64 / rate
            } else {
                (start + end) / 2.
            },
            observed,
            start,
            end,
            raw_start: raw.source_start as f64 / rate,
            raw_end: raw.source_end as f64 / rate,
            available: raw.available_end as f64 / rate,
            missing: if acquired < raw.end - raw.start {
                (end - start - observed).max(0.)
            } else {
                0.
            },
            epoch: raw.group.epoch,
            generation: raw.group.generation,
            ..Self::default()
        };
        for (moment, value) in out.moments.iter_mut().zip(raw.values) {
            if let Some(value) = value.filter(|_| observed > 0.) {
                *moment = [observed, value, 0.];
            }
        }
        Ok(out)
    }

    fn merge(self, other: Self) -> Result<Self, &'static str> {
        if self.end != other.start
            || (self.epoch, self.generation) != (other.epoch, other.generation)
        {
            return Err("noncontiguous or foreign descriptor blocks");
        }
        let mut out = self;
        for (result, (a, b)) in out
            .moments
            .iter_mut()
            .zip(self.moments.into_iter().zip(other.moments))
        {
            let w = a[0] + b[0];
            *result = if a[0] > 0. && b[0] > 0. {
                let delta = b[1] - a[1];
                [
                    w,
                    a[1] + b[0] / w * delta,
                    a[2] + b[2] + (a[0] / w) * b[0] * delta * delta,
                ]
            } else if b[0] > 0. {
                b
            } else {
                a
            };
            if result.iter().any(|x| !x.is_finite()) {
                return Err("descriptor moment overflow");
            }
        }
        let (a, b) = (self.observed, other.observed);
        out.time = if a > 0. && b > 0. {
            self.time + b / (a + b) * (other.time - self.time)
        } else if a > 0. {
            self.time
        } else if b > 0. {
            other.time
        } else {
            (self.start + other.end) / 2.
        };
        out.observed = a + b;
        out.end = other.end;
        out.raw_start = self.raw_start.min(other.raw_start);
        out.raw_end = self.raw_end.max(other.raw_end);
        out.available = self.available.max(other.available);
        out.missing = self.missing + other.missing;
        Ok(out)
    }

    pub(super) fn matching(self) -> matcher::Knot {
        let mut knot = matcher::Knot {
            start: self.start,
            end: self.end,
            time: self.time,
            observed_sec: self.observed,
            raw_start: self.raw_start,
            raw_end: self.raw_end,
            available_end: self.available,
            gap: u64::from(self.missing > 0.),
            ..matcher::Knot::default()
        };
        for (j, m) in self.moments.iter().enumerate() {
            if m[0] / (self.end - self.start) >= 0.9 {
                knot.values[j] = m[1];
                knot.mask |= 1 << j;
            }
        }
        knot.timing = u64::from(knot.observed_sec > 0. && knot.gap == 0 && knot.mask != 0);
        knot
    }

    #[cfg(test)]
    fn encode(self, bytes: &mut Vec<u8>) {
        for value in self.moments.into_iter().flatten().chain([
            self.time,
            self.observed,
            self.start,
            self.end,
            self.raw_start,
            self.raw_end,
            self.available,
            self.missing,
        ]) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&self.epoch.to_le_bytes());
        bytes.extend_from_slice(&self.generation.to_le_bytes());
    }
}

#[derive(Clone)]
struct Bank {
    blocks: Box<[Block]>,
    count: usize,
    scratch: Block,
    scales: [f64; 10],
    reconstruction_error: f64,
    merges: u64,
    coordinate_evaluations: u64,
    frozen: bool,
}

impl Bank {
    fn new(scales: [f64; 10], capacity: usize) -> Result<Self, &'static str> {
        if !(3..=256).contains(&capacity) || scales.iter().any(|s| !s.is_finite() || *s < 0.) {
            return Err("invalid descriptor capacity or frozen scales");
        }
        Ok(Self {
            blocks: vec![Block::default(); capacity].into_boxed_slice(),
            count: 0,
            scratch: Block::default(),
            scales: scales.map(|s| s.max(1e-6)),
            reconstruction_error: 0.,
            merges: 0,
            coordinate_evaluations: 0,
            frozen: false,
        })
    }

    fn append(&mut self, block: Block) -> Result<(), &'static str> {
        if self.frozen {
            return Err("committed descriptor cannot consume observations");
        }
        if self.count > 0 {
            let old = self.blocks[self.count - 1];
            if old.end != block.start
                || (old.epoch, old.generation) != (block.epoch, block.generation)
            {
                return Err("noncontiguous or foreign descriptor insertion");
            }
        }
        let added_error: f64 = block
            .moments
            .iter()
            .zip(self.scales)
            .map(|(m, sd)| m[2] / (sd * sd))
            .sum();
        let mut selection = None;
        if self.count == self.blocks.len() {
            let mut best = 1;
            let mut cost = f64::INFINITY;
            for i in 1..self.count - 1 {
                let mut increment = 0.;
                for ((a, b), sd) in self.blocks[i]
                    .moments
                    .iter()
                    .zip(self.blocks[i + 1].moments)
                    .zip(self.scales)
                {
                    if a[0] > 0. && b[0] > 0. {
                        let delta = (a[1] - b[1]) / sd;
                        increment += (a[0] / (a[0] + b[0])) * b[0] * delta * delta;
                    }
                }
                if increment < cost {
                    best = i;
                    cost = increment;
                }
            }
            selection = Some((best, cost, self.blocks[best].merge(self.blocks[best + 1])?));
        }
        let error =
            self.reconstruction_error + added_error + selection.map_or(0., |(_, cost, _)| cost);
        if !error.is_finite() {
            return Err("descriptor reconstruction error overflow");
        }
        self.scratch = block;
        if let Some((best, _, merged)) = selection {
            self.blocks[best] = merged;
            self.blocks.copy_within(best + 2..self.count, best + 1);
            self.merges += 1;
            self.coordinate_evaluations += ((self.count - 2) * 10) as u64;
        } else {
            self.count += 1;
        }
        self.blocks[self.count - 1] = self.scratch;
        self.reconstruction_error = error;
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(super) struct Config {
    pub group: Handle,
    pub sample_rate: u32,
    pub first_hop_start: u64,
    pub hop: u64,
    pub cadence: u8,
    pub span_start: f64,
    pub span_end: Option<f64>,
    pub scales: [f64; 10],
    pub capacity: usize,
}

pub(super) struct Span {
    config: Config,
    bank: Bank,
    pending: Option<Block>,
    gap_pending: Option<Block>,
    cursor: f64,
    last: Option<RawDescriptor>,
    last_intervals: Vec<u64>,
    interval_scratch: Vec<u64>,
    last_cut: u64,
    required_cut: u64,
    pub insertions: u64,
    pub last_insertions: u64,
    pub max_insertions: u64,
    pub last_supported_audio_end: Option<u64>,
}

#[derive(Clone)]
pub(super) struct Frozen {
    pub group: Handle,
    pub sample_rate: u32,
    pub start: f64,
    pub end: f64,
    pub captured_at: u64,
    pub supporting_audio_end: Option<u64>,
    pub scales: [f64; 10],
    pub available_at: u64,
    pub blocks: Box<[Block]>,
    pub reconstruction_error: f64,
}

impl Span {
    pub fn new(config: Config) -> Result<Self, &'static str> {
        if config.group.bus > 1
            || config.group.generation == 0
            || config.sample_rate == 0
            || config.hop == 0
            || ![1, 2, 4].contains(&config.cadence)
            || !config.span_start.is_finite()
            || config.span_start < config.first_hop_start as f64 / f64::from(config.sample_rate)
            || config
                .span_end
                .is_some_and(|e| !e.is_finite() || e <= config.span_start)
        {
            return Err("invalid descriptor generation, span or sample cadence");
        }
        let words = usize::try_from(config.hop.div_ceil(64))
            .map_err(|_| "descriptor acquisition bitmap exceeds address range")?;
        Ok(Self {
            config,
            bank: Bank::new(config.scales, config.capacity)?,
            pending: None,
            gap_pending: None,
            cursor: config.span_start,
            last: None,
            last_intervals: vec![0; words],
            interval_scratch: vec![0; words],
            last_cut: 0,
            required_cut: 0,
            insertions: 0,
            last_insertions: 0,
            max_insertions: 0,
            last_supported_audio_end: None,
        })
    }

    fn flush(&mut self, gap: bool) -> Result<(), &'static str> {
        let block = if gap { self.gap_pending } else { self.pending };
        if let Some(block) = block {
            self.bank.append(block)?;
            if gap {
                self.gap_pending = None;
            } else {
                self.pending = None;
            }
            self.insertions += 1;
        }
        Ok(())
    }

    pub fn gap(&mut self, end: f64, available_at: u64, cut: u64) -> Result<(), &'static str> {
        let available = available_at as f64 / f64::from(self.config.sample_rate);
        if self.bank.frozen
            || !(self.cursor < end && end <= available && available_at <= cut)
            || cut < self.last_cut
            || self.config.span_end.is_some_and(|e| end > e)
        {
            return Err("invalid or unavailable descriptor gap");
        }
        let block = Block {
            start: self.cursor,
            end,
            time: (self.cursor + end) / 2.,
            raw_start: self.cursor,
            raw_end: end,
            available,
            missing: end - self.cursor,
            epoch: self.config.group.epoch,
            generation: self.config.group.generation,
            ..Block::default()
        };
        let merged = if let Some(old) = self.gap_pending {
            old.merge(block)?
        } else {
            block
        };
        self.flush(false)?;
        self.gap_pending = Some(merged);
        self.cursor = end;
        self.last_cut = cut;
        self.required_cut = self.required_cut.max(available_at);
        Ok(())
    }

    pub fn push(
        &mut self,
        raw: &RawDescriptor,
        intervals: &[(u64, u64)],
        cut: u64,
    ) -> Result<bool, &'static str> {
        let cfg = self.config;
        let rate = f64::from(cfg.sample_rate);
        if self.bank.frozen
            || raw.group != cfg.group
            || raw.end.checked_sub(raw.start) != Some(cfg.hop)
            || raw.start < cfg.first_hop_start
            || !(raw.start - cfg.first_hop_start).is_multiple_of(cfg.hop)
            || raw.available_end > cut
            || cut < self.last_cut
        {
            return Err("noncanonical, foreign or unavailable descriptor delivery");
        }
        let start = (raw.start as f64 / rate).max(cfg.span_start);
        let end = (raw.end as f64 / rate).min(cfg.span_end.unwrap_or(f64::INFINITY));
        // Validate original intervals before bitmap construction or any moment update.
        let block = Block::from_raw(raw, cfg.sample_rate, intervals, start, end)?;
        self.interval_scratch.fill(0);
        for &(a, b) in intervals {
            for i in a - raw.start..b - raw.start {
                self.interval_scratch[i as usize / 64] |= 1 << (i % 64);
            }
        }
        if let Some(old) = self.last {
            let equal = old.group == raw.group
                && old.start == raw.start
                && old.end == raw.end
                && old.known_samples == raw.known_samples
                && old.source_start == raw.source_start
                && old.source_end == raw.source_end
                && old.available_end == raw.available_end
                && old
                    .values
                    .iter()
                    .zip(raw.values)
                    .all(|(a, b)| a.map(f64::to_bits) == b.map(f64::to_bits));
            if equal && self.last_intervals == self.interval_scratch {
                self.last_cut = cut;
                self.last_insertions = 0;
                return Ok(false);
            }
            if raw.start < old.end {
                return Err("stale or conflicting descriptor delivery");
            }
        }
        if end <= self.cursor || start < self.cursor {
            return Err("descriptor has no new nonoverlapping span support");
        }
        let before = self.insertions;
        if start > self.cursor {
            self.gap(start, raw.available_end, cut)?;
        }
        if raw.known_samples == 0 {
            self.gap(end, raw.available_end, cut)?;
        } else {
            if raw.known_samples < cfg.hop {
                self.flush(false)?;
            }
            self.flush(true)?;
            self.pending = Some(if let Some(old) = self.pending {
                old.merge(block)?
            } else {
                block
            });
            self.cursor = end;
            let index = (raw.start - cfg.first_hop_start) / cfg.hop;
            if (index + 1).is_multiple_of(u64::from(cfg.cadence))
                || cfg.span_end == Some(end)
                || raw.known_samples < cfg.hop
            {
                self.flush(false)?;
            }
        }
        self.last = Some(*raw);
        self.last_intervals.copy_from_slice(&self.interval_scratch);
        self.last_cut = cut;
        self.required_cut = self.required_cut.max(raw.available_end);
        if raw.known_samples > 0 && raw.values.iter().any(Option::is_some) {
            self.last_supported_audio_end = Some(
                self.last_supported_audio_end
                    .map_or(raw.source_end, |t| t.max(raw.source_end)),
            );
        }
        self.last_insertions = self.insertions - before;
        self.max_insertions = self.max_insertions.max(self.last_insertions);
        assert!(self.last_insertions <= 3);
        Ok(true)
    }

    pub fn prefix(&self, cut: u64) -> Result<Frozen, &'static str> {
        if cut < self.last_cut
            || cut < self.required_cut
            || self.cursor <= self.config.span_start
            || (cut as f64) / f64::from(self.config.sample_rate) < self.cursor
        {
            return Err("nonempty current prefix required; older cuts need saved copies");
        }
        let mut bank = self.bank.clone();
        for block in [self.pending, self.gap_pending].into_iter().flatten() {
            bank.append(block)?;
        }
        Ok(Frozen {
            group: self.config.group,
            sample_rate: self.config.sample_rate,
            start: self.config.span_start,
            end: self.cursor,
            captured_at: cut,
            supporting_audio_end: self.last_supported_audio_end,
            scales: bank.scales,
            available_at: self.required_cut,
            blocks: bank.blocks[..bank.count].to_vec().into_boxed_slice(),
            reconstruction_error: bank.reconstruction_error,
        })
    }

    pub fn finish(&mut self, cut: u64) -> Result<Frozen, &'static str> {
        if cut < self.last_cut
            || cut < self.required_cut
            || self.config.span_end.is_some_and(|e| self.cursor != e)
        {
            return Err("unfinished span or unavailable original evidence at commit");
        }
        self.flush(false)?;
        self.flush(true)?;
        self.bank.frozen = true;
        Ok(Frozen {
            group: self.config.group,
            sample_rate: self.config.sample_rate,
            start: self.config.span_start,
            end: self.cursor,
            captured_at: cut,
            supporting_audio_end: self.last_supported_audio_end,
            scales: self.bank.scales,
            available_at: self.required_cut,
            blocks: self.bank.blocks[..self.bank.count]
                .to_vec()
                .into_boxed_slice(),
            reconstruction_error: self.bank.reconstruction_error,
        })
    }
}

impl Frozen {
    #[cfg(test)]
    pub fn packed(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.blocks.len() * 320);
        for block in &self.blocks {
            block.encode(&mut bytes);
        }
        bytes
    }

    pub fn matching(&self) -> Result<memory::Descriptor, matcher::Error> {
        if self.blocks.len() > matcher::CAPACITY {
            return Err(matcher::Error::InvalidInput);
        }
        let knots: Vec<_> = self.blocks.iter().map(|block| block.matching()).collect();
        #[cfg(test)]
        let local_intervals = knots
            .iter()
            .enumerate()
            .map(|(i, k)| {
                (i > 0 && k.timing != 0 && knots[i - 1].timing != 0)
                    .then(|| k.time - knots[i - 1].time)
            })
            .collect();
        matcher::validate(
            &knots,
            self.blocks.iter().map(|b| b.available).fold(0., f64::max),
        )?;
        Ok(memory::Descriptor {
            knots,
            #[cfg(test)]
            local_intervals,
        })
    }
}

#[cfg(test)]
mod tests;
