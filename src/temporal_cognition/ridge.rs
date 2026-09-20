//! Bounded ridge continuity. Development scales must be supplied and frozen.

use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub(crate) struct Handle {
    pub bus: u8,
    pub epoch: u64,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(crate) struct Point {
    pub frequency_log2: Option<f64>,
    pub log_envelope: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(crate) struct Link {
    pub parent: Handle,
    pub parent_end_sample: u64,
    pub parent_slope: usize,
    pub distance: f64,
    pub weight: f64,
    pub secant: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(crate) struct Ridge {
    pub handle: Handle,
    pub point: Point,
    pub end_sample: u64,
    pub slopes: [Option<f64>; 2],
    pub links: [Option<Link>; 2],
    pub no_continuation_samples: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Config {
    pub means: [f64; 3],
    pub deviations: [f64; 3],
    pub distance_limit: f64,
    pub retirement_sec: f64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Update {
    pub observed: bool,
    pub current: [Option<Ridge>; 7],
    pub retired: [Option<Handle>; 7],
    pub superseded: [Option<Handle>; 7],
    pub evicted: [Option<Handle>; 7],
    pub distance_evaluations: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Tracker {
    bus: u8,
    epoch: u64,
    sample_rate: u32,
    hop: u64,
    config: Config,
    retirement_samples: u64,
    next_generation: u64,
    last_end: u64,
    slots: [Option<Ridge>; 7],
}

// Residual order is frequency prediction, secant/slope, then log envelope.
fn distance(
    old: &Ridge,
    point: Point,
    end: u64,
    slope_id: usize,
    tracker: &Tracker,
) -> Option<(f64, Option<f64>)> {
    let dt = (end - old.end_sample) as f64 / f64::from(tracker.sample_rate);
    let old_slope = old.slopes[slope_id];
    let contiguous = end - old.end_sample == tracker.hop;
    continuity_distance(
        old.point,
        old_slope,
        point,
        dt,
        contiguous,
        &tracker.config.means,
        &tracker.config.deviations,
    )
}

pub(super) fn continuity_distance(
    old: Point,
    old_slope: Option<f64>,
    point: Point,
    dt: f64,
    contiguous: bool,
    means: &[f64; 3],
    deviations: &[f64; 3],
) -> Option<(f64, Option<f64>)> {
    let secant = old
        .frequency_log2
        .zip(point.frequency_log2)
        .filter(|_| contiguous)
        .map(|(a, b)| (b - a) / dt);
    let residuals = [
        old.frequency_log2
            .zip(point.frequency_log2)
            .map(|(a, b)| b - (a + old_slope.unwrap_or(0.0) * dt)),
        secant.zip(old_slope).map(|(a, b)| a - b),
        point.log_envelope.zip(old.log_envelope).map(|(a, b)| a - b),
    ];
    let mut norm = 0.0_f64;
    let mut count = 0;
    for (index, residual) in residuals.into_iter().enumerate() {
        if let Some(value) = residual {
            let z = (value - means[index]) / deviations[index].max(1e-6);
            if !z.is_finite() {
                return None;
            }
            norm = norm.hypot(z);
            count += 1;
        }
    }
    if count == 0 || secant.is_some_and(|v| !v.is_finite()) {
        return None;
    }
    let rms = norm / (count as f64).sqrt();
    rms.is_finite().then_some((rms, secant))
}

impl Tracker {
    pub fn new(
        bus: u8,
        epoch: u64,
        sample_rate: u32,
        hop: u64,
        config: Config,
    ) -> Result<Self, &'static str> {
        let retirement = (config.retirement_sec * f64::from(sample_rate)).ceil();
        if bus > 1
            || sample_rate == 0
            || hop == 0
            || config.means.iter().any(|v| !v.is_finite())
            || config.deviations.iter().any(|v| !v.is_finite() || *v < 0.0)
            || !config.distance_limit.is_finite()
            || config.distance_limit <= 0.0
            || !retirement.is_finite()
            || retirement < 1.0
            || retirement >= u64::MAX as f64
        {
            return Err("invalid frozen ridge configuration");
        }
        Ok(Self {
            bus,
            epoch,
            sample_rate,
            hop,
            config,
            retirement_samples: retirement as u64,
            next_generation: 1,
            last_end: 0,
            slots: [None; 7],
        })
    }

    pub fn advance(
        &mut self,
        end_sample: u64,
        points: Option<&[Point]>,
    ) -> Result<Update, &'static str> {
        if end_sample <= self.last_end
            || !end_sample.is_multiple_of(self.hop)
            || points.is_some_and(|points| {
                points.len() > 7
                    || points.iter().any(|point| {
                        point.frequency_log2.is_none_or(|v| !v.is_finite())
                            || point.log_envelope.is_some_and(|v| !v.is_finite())
                    })
            })
        {
            return Err("invalid or noncausal ridge observation");
        }
        let mut output = Update {
            observed: points.is_some(),
            current: [None; 7],
            retired: [None; 7],
            superseded: [None; 7],
            evicted: [None; 7],
            distance_evaluations: 0,
        };
        let Some(points) = points else {
            self.last_end = end_sample;
            return Ok(output);
        };
        let mut choices: [[Option<(usize, Link)>; 2]; 7] = [[None; 2]; 7];
        let mut successors = [0; 7];
        for (index, &point) in points.iter().enumerate() {
            for (slot, old) in self
                .slots
                .iter()
                .enumerate()
                .filter_map(|(i, s)| s.as_ref().map(|s| (i, s)))
            {
                let mut best: Option<Link> = None;
                // A birth/gap carries one persistence prediction with a masked slope.
                let alternatives = if old.slopes[1].is_some() { 2 } else { 1 };
                for slope_id in 0..alternatives {
                    output.distance_evaluations += 1;
                    if let Some((d, secant)) = distance(old, point, end_sample, slope_id, self)
                        && d <= self.config.distance_limit
                        && best.is_none_or(|b| d < b.distance)
                    {
                        best = Some(Link {
                            parent: old.handle,
                            parent_end_sample: old.end_sample,
                            parent_slope: slope_id,
                            distance: d,
                            weight: 0.0,
                            secant,
                        });
                    }
                }
                if let Some(link) = best {
                    let earlier = |previous: Option<(usize, Link)>| {
                        previous.is_none_or(|(_, p)| {
                            (link.distance, link.parent) < (p.distance, p.parent)
                        })
                    };
                    if earlier(choices[index][0]) {
                        choices[index][1] = choices[index][0];
                        choices[index][0] = Some((slot, link));
                    } else if earlier(choices[index][1]) {
                        choices[index][1] = Some((slot, link));
                    }
                }
            }
            for (slot, _) in choices[index].iter().flatten() {
                successors[*slot] += 1;
            }
        }
        // Resolve every link before issuing handles; no current peaks share a generation.
        let mut next = *self;
        let mut reused = [false; 7];
        for (index, &point) in points.iter().enumerate() {
            let first = choices[index][0];
            let reuse =
                first.filter(|(slot, _)| choices[index][1].is_none() && successors[*slot] == 1);
            let handle = if let Some((slot, link)) = reuse {
                reused[slot] = true;
                link.parent
            } else {
                let generation = next.next_generation;
                next.next_generation = generation.checked_add(1).ok_or("ridge handle exhausted")?;
                Handle {
                    bus: self.bus,
                    epoch: self.epoch,
                    generation,
                }
            };
            let mut links = choices[index].map(|entry| entry.map(|(_, link)| link));
            let mut slopes = [None; 2];
            if let Some(first) = links[0] {
                let mut sum = 0.0;
                for link in links.iter_mut().flatten() {
                    // Subtract the nearest exponent before normalization.
                    link.weight = if link.distance == first.distance {
                        1.0
                    } else {
                        ((first.distance - link.distance) * (first.distance + link.distance)).exp()
                    };
                    sum += link.weight;
                }
                for (slot, link) in links
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(i, l)| l.as_mut().map(|l| (i, l)))
                {
                    link.weight /= sum;
                    slopes[slot] = link.secant;
                }
            }
            output.current[index] = Some(Ridge {
                handle,
                point,
                end_sample,
                slopes,
                links,
                no_continuation_samples: 0,
            });
        }
        let mut dormant = [None; 7];
        for (slot, old) in self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.map(|s| (i, s)))
        {
            if reused[slot] {
                continue;
            }
            if successors[slot] > 0 {
                output.superseded[slot] = Some(old.handle);
                continue;
            }
            let mut ridge = old;
            ridge.no_continuation_samples = ridge.no_continuation_samples.saturating_add(self.hop);
            if ridge.no_continuation_samples >= self.retirement_samples {
                output.retired[slot] = Some(ridge.handle);
            } else {
                dormant[slot] = Some(ridge);
            }
        }
        // Keep the most recently supported dormant references when current peaks fill slots.
        dormant.sort_unstable_by_key(|r| {
            r.map(|r| (std::cmp::Reverse(r.end_sample), std::cmp::Reverse(r.handle)))
        });
        next.slots = output.current;
        let mut destination = points.len();
        let mut evicted = 0;
        for ridge in dormant.into_iter().flatten() {
            if destination < 7 {
                next.slots[destination] = Some(ridge);
                destination += 1;
            } else {
                output.evicted[evicted] = Some(ridge.handle);
                evicted += 1;
            }
        }
        next.last_end = end_sample;
        *self = next;
        Ok(output)
    }
}

#[cfg(test)]
mod tests;
