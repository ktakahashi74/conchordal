//! Causal, bounded episode references from shared acoustic evidence.

use std::collections::VecDeque;
use std::sync::Arc;

pub(crate) use super::ridge::Handle;
use super::{features::Accent, proposals::frontend, recall};

pub(crate) const REFERENCES: usize = 16;

#[derive(Clone, Default)]
pub(crate) struct Context {
    pub inventory: Option<Snapshot>,
    pub retained: Arc<[(u64, u64)]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Family {
    Nonperiodic,
    Periodic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
pub(crate) struct Key {
    pub epoch: u64,
    pub episode: u64,
    pub generation: u64,
    pub family: Family,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Anchor {
    pub group: Handle,
    pub interval_sec: [f64; 2],
    // None means the Voice's intrinsic period must be frozen at action issue.
    pub period_sec: Option<f64>,
    pub weight: f64,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Reference {
    pub key: Key,
    pub weight: f64,
    pub weight_upper: f64,
    pub anchors: [Option<Anchor>; 7],
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct GroupSupport {
    pub group: Handle,
    pub observed_samples: u64,
    pub assignment_samples: f64,
    pub normalized_assignment: f64,
    pub coverage: f64,
    pub anchor: Option<Accent>,
    pub period_sec: Option<f64>,
    pub period_source: Option<[u64; 3]>,
    pub query_id: Option<u64>,
    pub query_support: Option<[u64; 2]>,
    pub query_available: Option<u64>,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Snapshot {
    pub bus: u8,
    pub epoch: u64,
    pub sample_rate: u32,
    pub window: [u64; 2],
    pub denominator_samples: u64,
    pub epoch_clipped: bool,
    pub available_at_sample: u64,
    pub credit_policy: &'static str,
    pub bias_parameter: &'static str,
    pub no_memory_bias: f64,
    pub assignment_samples: f64,
    pub groups: [Option<GroupSupport>; 7],
    pub references: [Option<Reference>; REFERENCES],
    pub examined_bindings: usize,
    pub assigned: f64,
    pub unassigned: f64,
    pub search_discarded: [f64; 2],
    pub inventory_discarded: [f64; 2],
    pub anchor_unsupported: [f64; 2],
    pub reference_bytes: usize,
    pub owned_bytes: usize,
}

#[derive(Clone, Copy)]
struct Contribution {
    group: Handle,
    alpha: f64,
    known: bool,
}

#[derive(Clone, Copy)]
struct Frame {
    start: u64,
    end: u64,
    groups: [Option<Contribution>; 7],
}

pub(crate) struct Stream {
    bus: u8,
    epoch: u64,
    origin: u64,
    rate: u32,
    hop: u64,
    end: u64,
    frames: VecDeque<Frame>,
    anchors: [Option<Accent>; 7],
    unknown_through: [u64; 7],
    owners: [Option<Handle>; 7],
    scratch: Vec<Reference>,
}

impl Stream {
    pub fn new(bus: u8, epoch: u64, origin: u64, rate: u32, hop: u64) -> Self {
        assert!(bus <= 1 && rate > 0 && hop > 0);
        assert!(std::mem::size_of::<Reference>() <= 512);
        Self {
            bus,
            epoch,
            origin,
            rate,
            hop,
            end: origin,
            frames: VecDeque::with_capacity((2 * u64::from(rate)).div_ceil(hop) as usize + 1),
            anchors: [None; 7],
            unknown_through: [origin; 7],
            owners: [None; 7],
            scratch: Vec::with_capacity(7 * REFERENCES),
        }
    }

    pub fn advance(
        &mut self,
        acoustic: &frontend::Snapshot,
        period: Option<&frontend::recurrence::Snapshot>,
        memory: &recall::Snapshot,
        bias: f64,
        cut: u64,
    ) -> Result<Snapshot, &'static str> {
        if cut <= self.end
            || !(cut - self.origin).is_multiple_of(self.hop)
            || acoustic.assignment.end_sample != cut
            || !bias.is_finite()
        {
            return Err("ordered inventory observation and shared finite bias required");
        }
        let start = cut.saturating_sub(self.hop).max(self.origin);
        let window_start = cut
            .saturating_sub(2 * u64::from(self.rate))
            .max(self.origin);
        while self.frames.front().is_some_and(|f| f.end <= window_start) {
            self.frames.pop_front();
        }
        let mut frame = Frame {
            start,
            end: cut,
            groups: [None; 7],
        };
        let rows = acoustic.assignment.rows.iter().flatten().count();
        for i in 0..7 {
            // This hop still belongs to the old assignment when lifecycle retires a group.
            let Some(group) = acoustic.group_handles[i] else {
                continue;
            };
            if group.bus != self.bus || group.epoch != self.epoch {
                return Err("inventory acoustic ownership differs from bus epoch");
            }
            let update = acoustic.features[i];
            let known = acoustic.eligible[i]
                && acoustic.spectral_shape_supported
                && update.is_some_and(|u| {
                    u.raw.group == group
                        && u.raw.start == start
                        && u.raw.end == cut
                        && u.raw.known_samples == cut - start
                        && u.raw.source_start <= start
                        && u.raw.source_end >= u.raw.end
                        && u.raw.source_end <= cut
                        && u.raw.available_end <= cut
                });
            let alpha = if known && rows > 0 {
                let j = acoustic
                    .assignment
                    .group_handles
                    .iter()
                    .position(|g| *g == Some(group));
                j.map_or(0., |j| {
                    acoustic
                        .assignment
                        .rows
                        .iter()
                        .flatten()
                        .map(|r| r.weights[j])
                        .sum::<f64>()
                        / rows as f64
                })
            } else {
                0.
            };
            if !alpha.is_finite() || !(0. ..=1.).contains(&alpha) {
                return Err("inventory requires normalized observed assignments");
            }
            frame.groups[i] = Some(Contribution {
                group,
                alpha,
                known,
            });
        }
        for i in 0..7 {
            let owner = acoustic.retained_groups[i];
            if self.owners[i] != owner {
                self.owners[i] = owner;
                self.anchors[i] = None;
                self.unknown_through[i] = self.end;
            }
            let Some(group) = owner else { continue };
            if group.bus != self.bus || group.epoch != self.epoch {
                return Err("inventory retained ownership differs from bus epoch");
            }
            let known = frame
                .groups
                .iter()
                .flatten()
                .any(|c| c.group == group && c.known);
            let update = acoustic.features[..7]
                .iter()
                .flatten()
                .find(|u| u.raw.group == group);
            if !known || start != self.end {
                self.anchors[i] = None;
                self.unknown_through[i] = if known { start } else { cut };
            }
            if known && let Some(accent) = update.and_then(|u| u.detector.and_then(|d| d.accent)) {
                if accent.group != group
                    || !accent.at_cut(cut)
                    || accent.event_start > accent.event_end
                    || accent.event_start < accent.source_start
                    || accent.event_end > accent.source_end
                    || !accent.weight.is_finite()
                    || accent.weight < 0.
                {
                    return Err("inventory anchor lacks causal original support");
                }
                if accent.weight > 0. && accent.source_start >= self.unknown_through[i] {
                    self.anchors[i] = Some(accent);
                }
            }
        }
        if self.frames.len() == self.frames.capacity() {
            return Err("inventory support exceeded its bounded hop ring");
        }
        self.frames.push_back(frame);
        self.end = cut;
        let denominator = cut - window_start;
        let mut groups = [None; 7];
        for (i, group) in self.owners.iter().enumerate() {
            if let Some(group) = group {
                groups[i] = Some(GroupSupport {
                    group: *group,
                    observed_samples: 0,
                    assignment_samples: 0.,
                    normalized_assignment: 0.,
                    coverage: 0.,
                    anchor: self.anchors[i],
                    period_sec: None,
                    period_source: None,
                    query_id: None,
                    query_support: None,
                    query_available: None,
                });
            }
        }
        let mut total = 0.;
        for f in &self.frames {
            let n = f.end - f.start.max(window_start);
            for c in f.groups.iter().flatten() {
                total += c.alpha * n as f64;
                if let Some(g) = groups.iter_mut().flatten().find(|g| g.group == c.group) {
                    g.observed_samples += if c.known { n } else { 0 };
                    g.assignment_samples += c.alpha * n as f64;
                }
            }
        }
        self.scratch.clear();
        let mut examined = 0;
        let mut search_discarded = [0.; 2];
        let mut anchor_unsupported = [0.; 2];
        for g in groups.iter_mut().flatten() {
            g.normalized_assignment = if total > 0. {
                g.assignment_samples / total
            } else {
                0.
            };
            g.coverage = if denominator > 0 {
                g.observed_samples as f64 / denominator as f64
            } else {
                0.
            };
            if let Some(pg) = period
                .filter(|p| p.end_sample == cut && p.received_at <= cut)
                .and_then(|p| {
                    p.groups
                        .iter()
                        .flatten()
                        .find(|p| p.active && p.association_known && p.ledger.group == g.group)
                })
                && let Some(source) = pg
                    .period_source
                    .filter(|s| s[0] <= s[1] && s[1] <= s[2] && s[2] <= cut)
                && let Some(peak) = pg
                    .peaks
                    .iter()
                    .flatten()
                    .filter(|p| {
                        p.support > 0.
                            && p.support.is_finite()
                            && p.period_seconds > 0.
                            && p.period_seconds.is_finite()
                    })
                    .max_by(|a, b| {
                        a.support
                            .total_cmp(&b.support)
                            .then(b.period_seconds.total_cmp(&a.period_seconds))
                    })
            {
                g.period_sec = Some(peak.period_seconds);
                g.period_source = Some(source);
            }
            let Some(q) = memory.retrieval.iter().flatten().find(|q| {
                q.group == g.group && q.evaluated_at_sample == cut && q.available_at_sample <= cut
            }) else {
                continue;
            };
            if q.no_memory_bias != bias {
                return Err("private inventory cannot substitute its own no-memory bias");
            }
            g.query_id = Some(q.query_id);
            g.query_support = Some([q.support_start_sample, q.support_end_sample]);
            g.query_available = Some(q.available_at_sample);
            let factor = g.normalized_assignment * g.coverage;
            for (discarded, weight) in search_discarded.iter_mut().zip(q.discarded_weight) {
                *discarded += factor * weight;
            }
            for e in q.entries.iter().flatten() {
                examined += 1;
                let weights = e.weight.map(|w| factor * w);
                let Some(anchor) = g.anchor else {
                    for (v, w) in anchor_unsupported.iter_mut().zip(weights) {
                        *v += w;
                    }
                    continue;
                };
                if weights[0] <= 0. {
                    continue;
                }
                let key = Key {
                    epoch: self.epoch,
                    episode: e.episode,
                    generation: e.generation,
                    family: if g.period_sec.is_some() {
                        Family::Periodic
                    } else {
                        Family::Nonperiodic
                    },
                };
                let index = if let Some(i) = self.scratch.iter().position(|r| r.key == key) {
                    i
                } else {
                    self.scratch.push(Reference {
                        key,
                        weight: 0.,
                        weight_upper: 0.,
                        anchors: [None; 7],
                    });
                    self.scratch.len() - 1
                };
                let r = &mut self.scratch[index];
                r.weight += weights[0];
                r.weight_upper += weights[1];
                let slot = r
                    .anchors
                    .iter_mut()
                    .find(|a| a.is_none())
                    .ok_or("episode reference exceeded seven group bindings")?;
                *slot = Some(Anchor {
                    group: g.group,
                    interval_sec: [anchor.event_start, anchor.event_end]
                        .map(|t| t as f64 / f64::from(self.rate)),
                    period_sec: g.period_sec,
                    weight: weights[0],
                });
            }
        }
        self.scratch
            .sort_unstable_by(|a, b| b.weight.total_cmp(&a.weight).then(a.key.cmp(&b.key)));
        let inventory_discarded = [
            self.scratch.iter().skip(REFERENCES).map(|r| r.weight).sum(),
            self.scratch
                .iter()
                .skip(REFERENCES)
                .map(|r| r.weight_upper)
                .sum(),
        ];
        let references = std::array::from_fn(|i| {
            self.scratch.get(i).copied().map(|mut r| {
                for a in r.anchors.iter_mut().flatten() {
                    a.weight /= r.weight;
                }
                r
            })
        });
        let assigned: f64 = references.iter().flatten().map(|r| r.weight).sum();
        if assigned > 1. + 1e-12 {
            return Err("inventory assigned more than unit credit");
        }
        Ok(Snapshot {
            bus: self.bus,
            epoch: self.epoch,
            sample_rate: self.rate,
            window: [window_start, cut],
            denominator_samples: denominator,
            epoch_clipped: cut - self.origin < 2 * u64::from(self.rate),
            available_at_sample: cut,
            credit_policy: "lower_bound",
            bias_parameter: "temporal_memory.retention.no_memory_bias",
            no_memory_bias: bias,
            assignment_samples: total,
            groups,
            references,
            examined_bindings: examined,
            assigned,
            unassigned: (1. - assigned).max(0.),
            search_discarded,
            inventory_discarded,
            anchor_unsupported,
            reference_bytes: std::mem::size_of_val(&references),
            owned_bytes: std::mem::size_of::<Self>()
                + self.frames.capacity() * std::mem::size_of::<Frame>()
                + self.scratch.capacity() * std::mem::size_of::<Reference>(),
        })
    }
}

#[cfg(test)]
mod tests;
