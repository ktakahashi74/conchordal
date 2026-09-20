//! Physical-support envelope correlations and deterministic complete-link bundles.

use super::ridge::Handle;

const MAX_HOPS: usize = 128;

#[derive(Clone, Copy, Debug)]
pub(super) struct Envelope {
    pub handle: Handle,
    pub log_envelope: f64,
}

#[derive(Clone, Copy)]
struct Frame {
    end_sample: u64,
    values: [Option<Envelope>; 8],
}

pub(super) struct Window {
    bus: u8,
    epoch: u64,
    epoch_start: u64,
    hop: u64,
    window_samples: u64,
    min_pairs: usize,
    min_coverage: f64,
    frames: Vec<Frame>,
    next: usize,
    len: usize,
    last_end: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Correlations {
    pub handles: [Option<Handle>; 8],
    pub count: usize,
    pub values: [[Option<f64>; 8]; 8],
    pub paired_samples: [[u64; 8]; 8],
    pub paired_hops: [[usize; 8]; 8],
    pub window_start: u64,
    pub window_end: u64,
}

impl Window {
    pub fn new(
        bus: u8,
        epoch: u64,
        epoch_start: u64,
        hop: u64,
        window_samples: u64,
        min_pairs: usize,
        min_coverage: f64,
    ) -> Result<Self, &'static str> {
        if bus > 1
            || hop == 0
            || !epoch_start.is_multiple_of(hop)
            || window_samples == 0
            || window_samples.div_ceil(hop) > MAX_HOPS as u64
            || !(2..=MAX_HOPS).contains(&min_pairs)
            || min_pairs as u64 > window_samples.div_ceil(hop)
            || !min_coverage.is_finite()
            || !(0.0..=1.0).contains(&min_coverage)
        {
            return Err("invalid or over-capacity envelope window");
        }
        Ok(Self {
            bus,
            epoch,
            epoch_start,
            hop,
            window_samples,
            min_pairs,
            min_coverage,
            frames: vec![
                Frame {
                    end_sample: 0,
                    values: [None; 8]
                };
                window_samples.div_ceil(hop) as usize
            ],
            next: 0,
            len: 0,
            last_end: epoch_start,
        })
    }

    pub fn push(
        &mut self,
        end_sample: u64,
        values: [Option<Envelope>; 8],
    ) -> Result<(), &'static str> {
        if end_sample <= self.last_end
            || !end_sample.is_multiple_of(self.hop)
            || values.iter().enumerate().any(|(i, value)| {
                value.is_some_and(|v| {
                    v.handle.bus != self.bus
                        || v.handle.epoch != self.epoch
                        || v.handle.generation == 0
                        || !v.log_envelope.is_finite()
                        || values[..i]
                            .iter()
                            .flatten()
                            .any(|previous| previous.handle == v.handle)
                })
            })
        {
            return Err("invalid or noncausal envelope frame");
        }
        self.frames[self.next] = Frame { end_sample, values };
        self.next = (self.next + 1) % self.frames.len();
        self.len = (self.len + 1).min(self.frames.len());
        self.last_end = end_sample;
        Ok(())
    }

    pub fn correlations(&self, current: &[Handle]) -> Result<Correlations, &'static str> {
        let latest = &self.frames[(self.next + self.frames.len() - 1) % self.frames.len()];
        if current.len() > 8
            || current.iter().enumerate().any(|(i, &h)| {
                current[..i].contains(&h) || !latest.values.iter().flatten().any(|v| v.handle == h)
            })
        {
            return Err("correlations require unique currently supported handles");
        }
        let start = self
            .last_end
            .saturating_sub(self.window_samples)
            .max(self.epoch_start);
        let mut out = Correlations {
            handles: std::array::from_fn(|i| current.get(i).copied()),
            count: current.len(),
            values: [[None; 8]; 8],
            paired_samples: [[0; 8]; 8],
            paired_hops: [[0; 8]; 8],
            window_start: start,
            window_end: self.last_end,
        };
        let first = (self.next + self.frames.len() - self.len) % self.frames.len();
        for a in 0..current.len() {
            for b in a + 1..current.len() {
                let pair = |frame: &Frame| {
                    let overlap = frame
                        .end_sample
                        .min(self.last_end)
                        .saturating_sub(frame.end_sample.saturating_sub(self.hop).max(start));
                    if overlap == 0 {
                        return None;
                    }
                    let x = frame
                        .values
                        .iter()
                        .flatten()
                        .find(|v| v.handle == current[a])?
                        .log_envelope;
                    let y = frame
                        .values
                        .iter()
                        .flatten()
                        .find(|v| v.handle == current[b])?
                        .log_envelope;
                    Some((x, y, overlap))
                };
                let mut count = 0;
                let mut support = 0;
                let mut anchor = None;
                let (mut sum_x, mut sum_y) = (0.0, 0.0);
                for i in 0..self.len {
                    if let Some((x, y, samples)) =
                        pair(&self.frames[(first + i) % self.frames.len()])
                    {
                        let (ax, ay) = *anchor.get_or_insert((x, y));
                        sum_x += x - ax;
                        sum_y += y - ay;
                        count += 1;
                        support += samples;
                    }
                }
                out.paired_samples[a][b] = support;
                out.paired_samples[b][a] = support;
                out.paired_hops[a][b] = count;
                out.paired_hops[b][a] = count;
                if count < self.min_pairs
                    || (support as f64) < self.min_coverage * (self.last_end - start) as f64
                {
                    continue;
                }
                let (ax, ay) = anchor.unwrap();
                let (mx, my) = (sum_x / count as f64, sum_y / count as f64);
                let (mut xx, mut yy, mut xy) = (0.0, 0.0, 0.0);
                for i in 0..self.len {
                    if let Some((x, y, _)) = pair(&self.frames[(first + i) % self.frames.len()]) {
                        let (dx, dy) = ((x - ax) - mx, (y - ay) - my);
                        xx += dx * dx;
                        yy += dy * dy;
                        xy += dx * dy;
                    }
                }
                if xx > 0.0 && yy > 0.0 && xx.is_finite() && yy.is_finite() && xy.is_finite() {
                    let coefficient = (xy / xx.sqrt()) / yy.sqrt();
                    if coefficient.is_finite() && coefficient.abs() <= 1.0 + 1e-12 {
                        let coefficient = coefficient.clamp(-1.0, 1.0);
                        out.values[a][b] = Some(coefficient);
                        out.values[b][a] = Some(coefficient);
                    }
                }
            }
        }
        Ok(out)
    }
}

#[derive(Debug)]
pub(super) struct Bundles {
    pub members: [[Option<Handle>; 8]; 8],
    pub count: usize,
    pub correlation_reads: usize,
}

pub(super) fn complete_link(
    correlations: &Correlations,
    threshold: f64,
) -> Result<Bundles, &'static str> {
    let n = correlations.count;
    if n > 8
        || !threshold.is_finite()
        || !(-1.0..=1.0).contains(&threshold)
        || correlations.handles[..n].iter().any(Option::is_none)
        || correlations.handles[n..].iter().any(Option::is_some)
    {
        return Err("invalid correlation inventory or threshold");
    }
    for i in 0..n {
        let h = correlations.handles[i].unwrap();
        let first = correlations.handles[0].unwrap();
        if h.bus > 1
            || h.bus != first.bus
            || h.epoch != first.epoch
            || h.generation == 0
            || correlations.handles[..i].contains(&Some(h))
        {
            return Err("invalid or repeated correlation identity");
        }
        for j in 0..n {
            let c = correlations.values[i][j];
            if c != correlations.values[j][i]
                || c.is_some_and(|v| !v.is_finite() || !(-1.0..=1.0).contains(&v))
            {
                return Err("correlations must be finite masked symmetric coefficients");
            }
        }
    }
    let mut order: [usize; 8] = std::array::from_fn(|i| i);
    order[..n].sort_unstable_by_key(|&i| correlations.handles[i]);
    let compare = |a: u8, b: u8| {
        order[..n]
            .iter()
            .filter(|&&i| a & (1 << i) != 0)
            .map(|&i| correlations.handles[i])
            .cmp(
                order[..n]
                    .iter()
                    .filter(|&&i| b & (1 << i) != 0)
                    .map(|&i| correlations.handles[i]),
            )
    };
    let mut masks: [u8; 8] = std::array::from_fn(|i| if i < n { 1 << order[i] } else { 0 });
    let mut count = n;
    let mut reads = 0;
    loop {
        let mut best: Option<(usize, usize, f64)> = None;
        for a in 0..count {
            for b in a + 1..count {
                let mut score = 1.0;
                let mut eligible = true;
                for i in 0..n {
                    if masks[a] & (1 << i) == 0 {
                        continue;
                    }
                    for j in 0..n {
                        if masks[b] & (1 << j) == 0 {
                            continue;
                        }
                        reads += 1;
                        if let Some(value) = correlations.values[i][j].filter(|v| *v >= threshold) {
                            score = f64::min(score, value);
                        } else {
                            eligible = false;
                        }
                    }
                }
                // Sorted bundles make the first equal-scoring pair the canonical tie winner.
                if eligible && best.is_none_or(|(_, _, previous)| score > previous) {
                    best = Some((a, b, score));
                }
            }
        }
        let Some((a, b, _)) = best else {
            break;
        };
        masks[a] |= masks[b];
        for i in b..count - 1 {
            masks[i] = masks[i + 1];
        }
        count -= 1;
        masks[count] = 0;
        masks[..count].sort_unstable_by(|a, b| compare(*a, *b));
    }
    let mut members = [[None; 8]; 8];
    for bundle in 0..count {
        let mut destination = 0;
        for &i in &order[..n] {
            if masks[bundle] & (1 << i) != 0 {
                members[bundle][destination] = correlations.handles[i];
                destination += 1;
            }
        }
    }
    Ok(Bundles {
        members,
        count,
        correlation_reads: reads,
    })
}

#[cfg(test)]
mod tests;
