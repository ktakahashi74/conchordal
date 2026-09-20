//! Shared acquisition time, independent of spectral or group-feature support.

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Bounds {
    pub lower: f64,
    pub upper: f64,
    pub history_lost: bool,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Snapshot {
    pub origin_sample: u64,
    pub end_sample: u64,
    pub delivery_cut_sample: u64,
    pub missing_seconds: f64,
    pub retained_records: usize,
    pub evicted_records: u64,
    pub record_capacity: usize,
    pub record_bytes: usize,
    pub allocated_bytes: usize,
    pub prefix_at_seconds: Option<f64>,
    pub prefix: Option<Bounds>,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Record {
    start: u64,
    end: u64,
    missing_before: f64,
    available: u64,
}

pub(super) struct Clock {
    epoch: u64,
    rate: u32,
    hop: u64,
    origin: u64,
    end: u64,
    cut: u64,
    missing: f64,
    records: Box<[Record]>,
    masks: Box<[u8]>,
    scratch: Box<[u8]>,
    head: usize,
    count: usize,
    evictions: u64,
    last_observation: bool,
}

pub(super) struct Acquisition<'a> {
    pub epoch: u64,
    pub start: u64,
    pub end: u64,
    pub available: u64,
    pub cut: u64,
    pub observed: bool,
    pub intervals: &'a [(u64, u64)],
}

impl Clock {
    pub(super) fn context(&self) -> (u64, f64, f64, f64) {
        let rate = f64::from(self.rate);
        (
            self.epoch,
            self.origin as f64 / rate,
            self.end as f64 / rate,
            self.cut as f64 / rate,
        )
    }

    pub fn new(
        epoch: u64,
        rate: u32,
        hop: u64,
        origin: u64,
        capacity: usize,
    ) -> Result<Self, &'static str> {
        if rate == 0 || hop == 0 || capacity == 0 {
            return Err("positive acquisition clock dimensions required");
        }
        let bytes = usize::try_from(hop.div_ceil(8)).map_err(|_| "acquisition bitmap too large")?;
        let total = bytes
            .checked_mul(capacity)
            .ok_or("acquisition bitmap capacity overflow")?;
        Ok(Self {
            epoch,
            rate,
            hop,
            origin,
            end: origin,
            cut: origin,
            missing: 0.,
            records: vec![Record::default(); capacity].into_boxed_slice(),
            masks: vec![0; total].into_boxed_slice(),
            scratch: vec![0; bytes].into_boxed_slice(),
            head: 0,
            count: 0,
            evictions: 0,
            last_observation: false,
        })
    }

    pub fn observe(&mut self, frame: Acquisition<'_>) -> Result<bool, &'static str> {
        let Acquisition {
            epoch,
            start,
            end,
            available,
            cut,
            observed,
            intervals,
        } = frame;
        if epoch != self.epoch
            || start < self.origin
            || end.checked_sub(start) != Some(self.hop)
            || !(start - self.origin).is_multiple_of(self.hop)
            || end > available
            || available > cut
            || cut < self.cut
            || intervals.len() as u64 > self.hop
        {
            return Err("ordered causal canonical acquisition hop required");
        }
        self.scratch.fill(0);
        for &(a, b) in intervals {
            if !(start <= a && a <= b && b <= end) {
                return Err("acquired intervals must lie inside their hop");
            }
            for i in a - start..b - start {
                self.scratch[i as usize / 8] |= 1 << (i % 8);
            }
        }
        if observed != self.scratch.iter().any(|b| *b != 0) {
            return Err("acquisition flag disagrees with the sample union");
        }
        if start < self.end {
            let slot = (self.head + self.records.len() - 1) % self.records.len();
            let old = self.records[slot];
            let offset = slot * self.scratch.len();
            if self.last_observation
                && old.start == start
                && old.end == end
                && old.available == available
                && self.masks[offset..offset + self.scratch.len()] == *self.scratch
            {
                self.cut = cut;
                return Ok(false);
            }
            return Err("stale or conflicting acquisition cannot revise the clock");
        }
        self.append(start, end, available);
        self.cut = cut;
        self.last_observation = true;
        Ok(true)
    }

    #[cfg(test)]
    fn gap(&mut self, end: u64, cut: u64) -> Result<(), &'static str> {
        if end <= self.end || end > cut || cut < self.cut {
            return Err("forward missing-sample notification required");
        }
        self.scratch.fill(0);
        self.append(self.end, end, cut);
        self.cut = cut;
        self.last_observation = false;
        Ok(())
    }

    fn append(&mut self, start: u64, end: u64, available: u64) {
        let before = self.missing + (start - self.end) as f64 / f64::from(self.rate);
        let known: u64 = self.scratch.iter().map(|b| u64::from(b.count_ones())).sum();
        self.records[self.head] = Record {
            start,
            end,
            missing_before: before,
            available,
        };
        let offset = self.head * self.scratch.len();
        self.masks[offset..offset + self.scratch.len()].copy_from_slice(&self.scratch);
        self.head = (self.head + 1) % self.records.len();
        self.evictions += u64::from(self.count == self.records.len());
        self.count = (self.count + 1).min(self.records.len());
        self.end = end;
        self.missing = before + (end - start - known) as f64 / f64::from(self.rate);
    }

    pub fn prefix(&self, time: f64) -> Result<Bounds, &'static str> {
        let rate = f64::from(self.rate);
        let origin = self.origin as f64 / rate;
        if !time.is_finite() || time < origin || time > self.end as f64 / rate {
            return Err("gap prefix must be inside acquired or notified time");
        }
        if time == origin {
            return Ok(Bounds {
                lower: 0.,
                upper: 0.,
                history_lost: false,
            });
        }
        for offset in 0..self.count {
            let slot = (self.head + self.records.len() - self.count + offset) % self.records.len();
            let r = self.records[slot];
            let left = r.start as f64 / rate;
            let right = r.end as f64 / rate;
            if time < left {
                if offset == 0 && self.evictions > 0 {
                    return Ok(Bounds {
                        lower: (r.missing_before - (left - time)).max(0.),
                        upper: r.missing_before.min(time - origin),
                        history_lost: true,
                    });
                }
                let value = (r.missing_before - (left - time)).max(0.);
                return Ok(Bounds {
                    lower: value,
                    upper: value,
                    history_lost: false,
                });
            }
            if time <= right {
                let mask = &self.masks[slot * self.scratch.len()..(slot + 1) * self.scratch.len()];
                let position = ((time - left) * rate).clamp(0., (r.end - r.start) as f64);
                let full = position.floor() as u64;
                let mut known = (0..full.min(self.hop))
                    .filter(|i| mask[*i as usize / 8] & (1 << (i % 8)) != 0)
                    .count() as f64;
                if full < self.hop && mask[full as usize / 8] & (1 << (full % 8)) != 0 {
                    known += position - full as f64;
                }
                let value = (r.missing_before + (position - known) / rate).max(0.);
                return Ok(Bounds {
                    lower: value,
                    upper: value,
                    history_lost: false,
                });
            }
        }
        Ok(Bounds {
            lower: self.missing,
            upper: self.missing,
            history_lost: false,
        })
    }

    pub fn snapshot(&self, prefix_at: Option<f64>) -> Snapshot {
        let record_bytes = self.records.len() * std::mem::size_of::<Record>() + self.masks.len();
        Snapshot {
            origin_sample: self.origin,
            end_sample: self.end,
            delivery_cut_sample: self.cut,
            missing_seconds: self.missing,
            retained_records: self.count,
            evicted_records: self.evictions,
            record_capacity: self.records.len(),
            record_bytes,
            allocated_bytes: std::mem::size_of::<Self>() + record_bytes + self.scratch.len(),
            prefix_at_seconds: prefix_at,
            prefix: prefix_at.map(|t| {
                self.prefix(t)
                    .expect("retained episode endpoint inside its acquisition epoch")
            }),
        }
    }
}

#[cfg(test)]
mod tests;
