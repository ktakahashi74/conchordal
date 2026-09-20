//! Candidate-local histories borrow frozen acoustic references and never advance them.

use super::*;
use crate::temporal_cognition::feature_projection::Feature;

const LANES: usize = 13;

pub(in crate::temporal_cognition) struct Scratch {
    histories: Box<[History; LANES]>,
    observed_records: [usize; LANES],
    future: Vec<[u64; 2]>,
}

impl Scratch {
    pub(in crate::temporal_cognition) fn owned_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + std::mem::size_of_val(self.histories.as_ref())
            + self.future.capacity() * std::mem::size_of::<[u64; 2]>()
            + self
                .histories
                .iter()
                .map(|h| h.records.capacity() * std::mem::size_of::<Record>())
                .sum::<usize>()
    }
}

#[derive(Clone, Copy)]
struct Anchor {
    interval: [u64; 2],
    weight: f64,
    reference: Reference,
}

pub(in crate::temporal_cognition) struct Projection<'a> {
    source: &'a Stream,
    scratch: &'a mut Scratch,
    target: usize,
    start: u64,
    end: u64,
    previous: Option<Anchor>,
    last_center_end: Option<u64>,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct ProjectedFeatures {
    pub issued_at: u64,
    pub evaluated_at: u64,
    pub window_start: u64,
    pub group: Handle,
    pub histories: [Option<Summary>; 3],
    pub(in crate::temporal_cognition) values: [[Feature; 14]; 3],
    pub observed_samples: [u64; 3],
    pub projected_samples: [u64; 3],
    pub observed_records: [usize; 3],
    pub projected_records: [usize; 3],
}

impl Stream {
    pub(in crate::temporal_cognition) fn project<'a>(
        &'a self,
        group: Handle,
        issue: u64,
        end: u64,
        scratch: &'a mut Option<Scratch>,
    ) -> Option<Projection<'a>> {
        if issue != self.cut || end < issue || end - issue > 4 * u64::from(self.rate) {
            return None;
        }
        let target = self.groups.iter().position(|g| g.owner == Some(group))?;
        let scratch = scratch.get_or_insert_with(|| Scratch {
            histories: (0..LANES)
                .map(|_| History::new())
                .collect::<Vec<_>>()
                .into_boxed_slice()
                .try_into()
                .unwrap_or_else(|_| unreachable!()),
            observed_records: [0; LANES],
            future: Vec::with_capacity((4 * u64::from(self.rate)).div_ceil(self.hop) as usize + 1),
        });
        scratch.future.clear();
        let start = end
            .saturating_sub(8 * u64::from(self.rate))
            .max(self.origin);
        for lane in 0..LANES {
            let reference = if lane == 12 {
                target
            } else {
                lane / 2 + usize::from(lane / 2 >= target)
            };
            let periodic = lane == 12 || lane % 2 == 0;
            let from = &self.histories[index(target, reference, periodic)];
            let to = &mut scratch.histories[lane];
            to.records.clear();
            to.records.extend(from.records.iter().copied());
            to.masses = from.masses;
            to.capacity_evicted = from.capacity_evicted;
            to.lost_through = from.lost_through;
            to.discarded = from.discarded;
            to.shape.set(from.shape.get());
            to.expire(start, periodic);
            scratch.observed_records[lane] = to.records.len();
        }
        let previous = self.groups[target].previous.map(|(a, reference)| Anchor {
            interval: [a.event_start, a.event_end],
            weight: a.weight,
            reference,
        });
        Some(Projection {
            source: self,
            scratch,
            target,
            start,
            end,
            previous,
            last_center_end: None,
        })
    }
}

impl Projection<'_> {
    pub(in crate::temporal_cognition) fn receipt(
        &mut self,
        interval: [u64; 2],
        known: bool,
        weight: Option<f64>,
    ) -> Result<(), &'static str> {
        if interval[0] >= interval[1]
            || interval[1] < self.source.cut
            || interval[1] > self.end
            || self.last_center_end.is_some_and(|t| interval[0] < t)
            || weight.is_some_and(|w| !w.is_finite() || !(0. ..=1.).contains(&w))
        {
            return Err("invalid candidate timing receipt");
        }
        if interval[0] > self.last_center_end.unwrap_or(self.source.cut) || !known {
            self.previous = None;
        }
        self.last_center_end = Some(interval[1]);
        if !known {
            return Ok(());
        }
        let future_start = interval[0].max(self.source.cut);
        if interval[1] > future_start {
            if self.scratch.future.len() == self.scratch.future.capacity() {
                return Err("candidate timing support capacity exceeded");
            }
            self.scratch.future.push([future_start, interval[1]]);
        }
        let Some(weight) = weight.filter(|w| *w > 0.) else {
            return Ok(());
        };
        let own_reference = self.source.groups[self.target].reference;
        for lane in 0..LANES {
            let reference = if lane == 12 {
                self.target
            } else {
                lane / 2 + usize::from(lane / 2 >= self.target)
            };
            let group = self.source.groups[reference];
            let periodic = lane == 12 || lane % 2 == 0;
            let context = if lane == 12 {
                self.previous
                    .filter(|a| {
                        a.interval[1] < interval[1]
                            && a.reference.version == own_reference.version
                            && own_reference.supported
                    })
                    .and_then(|a| {
                        a.reference.scale(self.source.rate).map(|scale| {
                            (a.interval, scale, weight.min(a.weight), a.reference.version)
                        })
                    })
            } else {
                let scale = if periodic {
                    group.reference.scale(self.source.rate)
                } else {
                    group.scale()
                };
                group
                    .latest
                    .filter(|a| a.event_end <= interval[1])
                    .zip(scale)
                    .map(|(a, scale)| {
                        (
                            [a.event_start, a.event_end],
                            scale,
                            weight * a.weight,
                            if periodic { group.reference.version } else { 0 },
                        )
                    })
            };
            if let Some((anchor, scale, weight, version)) = context
                && interval[1] > self.start
            {
                let history = &mut self.scratch.histories[lane];
                if history.records.len() == CAPACITY {
                    self.scratch.observed_records[lane] =
                        self.scratch.observed_records[lane].saturating_sub(1);
                }
                history.insert(
                    Record::from_intervals(interval, anchor, scale, weight, version),
                    periodic,
                );
            }
        }
        self.previous = Some(Anchor {
            interval,
            weight,
            reference: own_reference,
        });
        Ok(())
    }

    // Keep candidate summaries away from the observer's acoustic-analysis stack.
    #[inline(never)]
    pub(in crate::temporal_cognition) fn finish(self) -> ProjectedFeatures {
        let mut observed = [0; LANES];
        let mut projected = [0; LANES];
        let mut summaries = [None; LANES];
        for lane in 0..LANES {
            let reference = if lane == 12 {
                self.target
            } else {
                lane / 2 + usize::from(lane / 2 >= self.target)
            };
            let group = self.source.groups[reference];
            let Some(owner) = group.owner else { continue };
            let target = self.source.groups[self.target].owner.unwrap();
            let periodic = lane == 12 || lane % 2 == 0;
            let history = &self.scratch.histories[lane];
            let start = self.start.max(history.lost_through);
            let mask = (1 << self.target) | (1 << reference);
            observed[lane] = self
                .source
                .coverage
                .iter()
                .filter(|f| {
                    f.known & mask == mask
                        && f.generations[self.target] == target.generation
                        && f.generations[reference] == owner.generation
                })
                .map(|f| f.end.saturating_sub(f.start.max(start)))
                .sum();
            let reference_support = if periodic {
                group
                    .reference
                    .peak
                    .filter(|_| group.reference.supported)
                    .map_or(0., |p| p.support)
            } else {
                f64::from(group.scale().is_some())
            };
            if reference_support > 0. {
                projected[lane] = self
                    .scratch
                    .future
                    .iter()
                    .map(|s| s[1].saturating_sub(s[0].max(start)))
                    .sum();
            }
            let weight: f64 = history.records.iter().map(|r| r.weight).sum();
            let coverage = if self.end > self.start {
                (observed[lane] + projected[lane]) as f64 / (self.end - self.start) as f64
            } else {
                0.
            };
            let summary = Summary {
                reference: owner,
                family: if periodic {
                    Family::Periodic
                } else {
                    Family::MedianInterval
                },
                version: if periodic { group.reference.version } else { 0 },
                period_seconds: periodic
                    .then_some(group.reference.peak)
                    .flatten()
                    .map(|p| p.period_seconds),
                reference_support,
                records: history.records.len(),
                retained_weight: weight,
                coverage,
                supported: weight > 0. && reference_support > 0. && coverage >= 0.9,
                bins: std::array::from_fn(|i| {
                    if weight > 0. {
                        history.masses[i] / weight
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
            };
            summaries[lane] = Some(summary);
        }
        let mut order: [usize; 12] = std::array::from_fn(|i| i);
        order.sort_by(|&a, &b| {
            match (
                summaries[a].filter(|s| s.supported),
                summaries[b].filter(|s| s.supported),
            ) {
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
            }
        });
        let lanes = [order[0], order[1], 12];
        let mut histories =
            std::array::from_fn(|i| summaries[lanes[i]].filter(|s| i == 2 || s.supported));
        for (i, summary) in histories.iter_mut().enumerate() {
            if let Some(summary) = summary {
                self.scratch.histories[lanes[i]].describe(summary);
            }
        }
        let total: f64 = histories[..2]
            .iter()
            .flatten()
            .map(|s| s.retained_weight * s.reference_support)
            .sum();
        let values = std::array::from_fn(|i| {
            let Some(summary) = histories[i] else {
                return [Feature::Unsupported; 14];
            };
            let hypothetical = projected[lanes[i]] > 0
                || summary.records > self.scratch.observed_records[lanes[i]];
            summary
                .features(if i == 2 {
                    1.
                } else {
                    summary.retained_weight * summary.reference_support / total
                })
                .map(|v| match v {
                    None => Feature::Unsupported,
                    Some(x) if hypothetical => Feature::Projected(x),
                    Some(x) => Feature::Observed(x),
                })
        });
        ProjectedFeatures {
            issued_at: self.source.cut,
            evaluated_at: self.end,
            window_start: self.start,
            group: self.source.groups[self.target].owner.unwrap(),
            histories,
            values,
            observed_samples: std::array::from_fn(|i| {
                if histories[i].is_some() {
                    observed[lanes[i]]
                } else {
                    0
                }
            }),
            projected_samples: std::array::from_fn(|i| {
                if histories[i].is_some() {
                    projected[lanes[i]]
                } else {
                    0
                }
            }),
            observed_records: std::array::from_fn(|i| {
                if histories[i].is_some() {
                    self.scratch.observed_records[lanes[i]]
                } else {
                    0
                }
            }),
            projected_records: std::array::from_fn(|i| {
                histories[i].map_or(0, |s| s.records - self.scratch.observed_records[lanes[i]])
            }),
        }
    }
}

#[cfg(test)]
mod tests;
