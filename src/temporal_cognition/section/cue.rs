//! Acoustic-support cue selection over shared ongoing phrase occurrences.

use std::collections::VecDeque;

use crate::temporal_cognition::{descriptor, phrase, proposals::frontend, ridge::Handle};

pub(in crate::temporal_cognition) const MAX_RETAINED_PREFIXES: usize =
    7 * 15 + super::commitment::PENDING;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct Selection {
    pub occurrence_id: u64,
    pub start_sample: u64,
    pub support_end_sample: u64,
    pub selected_at: u64,
    pub last_observed_sample: u64,
    pub window_start_sample: f64,
    pub weighted_seconds: f64,
}

struct Entry {
    foreground: phrase::Foreground,
    span: descriptor::Span,
    assignment_seconds: f64,
}

struct Group {
    handle: Handle,
    start: u64,
    entries: Vec<Entry>,
    history: VecDeque<(u64, u64, f64)>,
    selection: Option<Selection>,
}

pub(crate) struct Stream {
    bus: u8,
    epoch: u64,
    rate: u32,
    hop: u64,
    scales: [f64; 10],
    end: u64,
    start: u64,
    groups: [Option<Group>; 7],
    pub(super) commitments: super::commitment::Ledger,
}

impl Stream {
    #[cfg(test)]
    pub(in crate::temporal_cognition) fn trace_prefix(
        &self,
        group: Handle,
        credit: u64,
        start: u64,
    ) -> serde_json::Value {
        let owner = self.groups.iter().flatten().find(|g| g.handle == group);
        serde_json::json!({
            "group_retained": owner.is_some(),
            "ongoing": owner.into_iter().flat_map(|g| &g.entries)
                .filter(|e| e.foreground.credit == credit && e.foreground.start == start)
                .map(|e| serde_json::json!({"foreground":e.foreground,
                    "assignment_seconds":e.assignment_seconds})).collect::<Vec<_>>(),
            "endpoints": self.commitments.trace_prefix(group, credit, start)
        })
    }

    pub(in crate::temporal_cognition) fn retains_prefix(
        &self,
        group: Handle,
        credit: u64,
        start: u64,
    ) -> bool {
        self.groups.iter().flatten().any(|g| {
            g.handle == group
                && g.entries
                    .iter()
                    .any(|e| e.foreground.credit == credit && e.foreground.start == start)
        }) || self.commitments.retains_prefix(group, credit, start)
    }

    pub fn new(
        bus: u8,
        epoch: u64,
        rate: u32,
        hop: u64,
        start: u64,
        scales: [f64; 10],
    ) -> Result<Self, &'static str> {
        if bus > 1 || rate == 0 || hop == 0 || scales.iter().any(|v| !v.is_finite() || *v < 1e-6) {
            return Err("invalid phrase cue clock or descriptor scales");
        }
        Ok(Self {
            bus,
            epoch,
            rate,
            hop,
            scales,
            end: start,
            start,
            groups: std::array::from_fn(|_| None),
            commitments: super::commitment::Ledger::new(rate, 256),
        })
    }

    #[cfg(test)]
    pub fn advance(
        &mut self,
        acoustic: &frontend::Snapshot,
        phrase: &phrase::Snapshot,
    ) -> Result<(), &'static str> {
        self.prepare(acoustic, phrase)?;
        self.commitments.seal(phrase.end_sample, &phrase.groups);
        Ok(())
    }

    pub(super) fn prepare(
        &mut self,
        acoustic: &frontend::Snapshot,
        phrase: &phrase::Snapshot,
    ) -> Result<(), &'static str> {
        let end = phrase.end_sample;
        if end <= self.end {
            return Err("phrase cues require an ordered new receipt");
        }
        for slot in &mut self.groups {
            if slot
                .as_ref()
                .is_some_and(|g| !phrase.groups.iter().flatten().any(|p| p.group == g.handle))
            {
                *slot = None;
            }
        }
        let window = (end as f64 - f64::from(self.rate) / 2.).max(self.start as f64);
        for p in phrase.groups.iter().flatten() {
            if p.group.bus != self.bus || p.group.epoch != self.epoch {
                return Err("foreign phrase cue group");
            }
            let slot = if let Some(slot) = self
                .groups
                .iter()
                .position(|g| g.as_ref().is_some_and(|g| g.handle == p.group))
            {
                slot
            } else {
                let slot = self
                    .groups
                    .iter()
                    .position(Option::is_none)
                    .ok_or("phrase cue group capacity")?;
                self.groups[slot] = Some(Group {
                    handle: p.group,
                    start: p.previous_end_sample,
                    entries: Vec::with_capacity(15),
                    history: VecDeque::with_capacity(
                        u64::from(self.rate).div_ceil(2 * self.hop) as usize + 2,
                    ),
                    selection: None,
                });
                slot
            };
            let g = self.groups[slot].as_mut().unwrap();
            let index = acoustic
                .group_handles
                .iter()
                .position(|h| *h == Some(g.handle));
            let raw = index
                .and_then(|i| acoustic.features[i])
                .map(|u| u.raw)
                .filter(|r| {
                    r.group == g.handle
                        && r.start >= self.end
                        && r.start < end
                        && r.end == end
                        && r.available_end <= end
                        && r.source_end <= end
                        && r.known_samples == r.end - r.start
                        && acoustic.eligible[index.unwrap()]
                });
            let weight = if raw.is_some() {
                let i = index.unwrap();
                let count = acoustic.assignment.rows.iter().flatten().count();
                let weight = if count == 0 {
                    0.
                } else {
                    acoustic
                        .assignment
                        .rows
                        .iter()
                        .flatten()
                        .map(|r| r.weights[i])
                        .sum::<f64>()
                        / count as f64
                };
                if !weight.is_finite() || !(0. ..=1.).contains(&weight) {
                    return Err("invalid phrase cue acoustic weight");
                }
                weight
            } else {
                0.
            };
            self.commitments
                .advance(p, end, raw.map(|r| (r.start, r.end)))?;
            let mut closed_seen = [(0, 0); 15];
            let mut closed_count = 0;
            for f in p
                .candidates
                .iter()
                .flatten()
                .filter_map(|c| c.completed_foreground)
            {
                if closed_seen[..closed_count].contains(&(f.credit, f.heard_end)) {
                    continue;
                }
                closed_seen[closed_count] = (f.credit, f.heard_end);
                closed_count += 1;
                if let Some(e) = g.entries.iter().find(|e| {
                    e.foreground.credit == f.credit
                        && e.foreground.start == f.start
                        && e.foreground.heard_end == f.heard_end
                        && e.assignment_seconds > 0.
                        && f.heard_end > f.start
                }) {
                    self.commitments.stage(
                        p,
                        f,
                        e.span.prefix(end)?,
                        e.assignment_seconds,
                        end,
                        raw.map(|r| (r.start, r.end)),
                    )?;
                } else if let Some(r) = raw
                    .filter(|r| f.start >= r.start && f.start < f.heard_end && f.heard_end == r.end)
                {
                    let mut span = descriptor::Span::new(descriptor::Config {
                        group: g.handle,
                        sample_rate: self.rate,
                        first_hop_start: r.start,
                        hop: self.hop,
                        cadence: 2,
                        span_start: f.start as f64 / f64::from(self.rate),
                        span_end: None,
                        scales: self.scales,
                        capacity: 128,
                    })?;
                    span.push(&r, &[(r.start, r.end)], end)?;
                    self.commitments.stage(
                        p,
                        f,
                        span.prefix(end)?,
                        (f.heard_end - f.start) as f64 * weight / f64::from(self.rate),
                        end,
                        Some((r.start, r.end)),
                    )?;
                }
            }
            g.entries.retain(|e| {
                p.candidates.iter().flatten().any(|c| {
                    c.foreground
                        .is_some_and(|f| f.credit == e.foreground.credit)
                })
            });
            if let Some(raw) = raw {
                g.history.push_back((raw.start, raw.end, weight));
            }
            while g
                .history
                .front()
                .is_some_and(|(_, b, _)| *b as f64 <= window)
            {
                g.history.pop_front();
            }
            let mut visited = [0; 15];
            let mut count = 0;
            for f in p.candidates.iter().flatten().filter_map(|c| c.foreground) {
                if visited[..count].contains(&f.credit) {
                    let old = g
                        .entries
                        .iter()
                        .find(|e| e.foreground.credit == f.credit)
                        .unwrap()
                        .foreground;
                    if old.start != f.start || old.heard_end != f.heard_end {
                        return Err("conflicting phrase cue aliases");
                    }
                    continue;
                }
                if count == visited.len()
                    || f.credit == 0
                    || f.start < self.start.max(g.start)
                    || f.start > f.heard_end
                    || f.heard_end > end
                {
                    return Err("invalid phrase cue occurrence inventory");
                }
                visited[count] = f.credit;
                count += 1;
                let existing = g
                    .entries
                    .iter()
                    .position(|e| e.foreground.credit == f.credit);
                if let Some(i) = existing {
                    let prior = g.entries[i].foreground;
                    if f.start != prior.start || f.heard_end < prior.heard_end {
                        return Err("phrase cue changed original occurrence support");
                    }
                } else {
                    g.entries.push(Entry {
                        foreground: f,
                        assignment_seconds: 0.,
                        span: descriptor::Span::new(descriptor::Config {
                            group: g.handle,
                            sample_rate: self.rate,
                            first_hop_start: (f.start / self.hop) * self.hop,
                            hop: self.hop,
                            cadence: 2,
                            span_start: f.start as f64 / f64::from(self.rate),
                            span_end: None,
                            scales: self.scales,
                            capacity: 128,
                        })?,
                    });
                }
                let i = existing.unwrap_or(g.entries.len() - 1);
                let e = &mut g.entries[i];
                if let Some(raw) = raw.filter(|r| {
                    r.end <= f.heard_end && r.end > f.start && r.start.max(f.start) < r.end
                }) {
                    e.span.push(&raw, &[(raw.start, raw.end)], end)?;
                    if let Some((a, b, w)) = g.history.back() {
                        e.assignment_seconds +=
                            b.saturating_sub((*a).max(f.start)) as f64 * w / f64::from(self.rate);
                    }
                }
                e.foreground = f;
            }
            g.selection = None;
            for e in &g.entries {
                let f = e.foreground;
                let lo = window.max(f.start as f64);
                let weighted = g
                    .history
                    .iter()
                    .map(|(a, b, w)| {
                        ((*b).min(f.heard_end) as f64 - (*a as f64).max(lo)).max(0.) * w
                            / f64::from(self.rate)
                    })
                    .sum::<f64>();
                if weighted <= 0. {
                    continue;
                }
                let last_observed = g
                    .history
                    .iter()
                    .filter(|(a, b, w)| {
                        *w > 0. && (*b).min(f.heard_end) as f64 > (*a as f64).max(lo)
                    })
                    .map(|(_, b, _)| (*b).min(f.heard_end))
                    .max()
                    .unwrap();
                let now = Selection {
                    occurrence_id: f.credit,
                    start_sample: f.start,
                    support_end_sample: f.heard_end,
                    selected_at: end,
                    last_observed_sample: last_observed,
                    window_start_sample: lo,
                    weighted_seconds: weighted,
                };
                if g.selection.is_none_or(|old| {
                    now.weighted_seconds
                        .total_cmp(&old.weighted_seconds)
                        .then(now.last_observed_sample.cmp(&old.last_observed_sample))
                        .then(now.start_sample.cmp(&old.start_sample))
                        .then(old.occurrence_id.cmp(&now.occurrence_id))
                        .is_gt()
                }) {
                    g.selection = Some(now);
                }
            }
        }
        self.end = end;
        Ok(())
    }

    pub fn selection(&self, group: Handle) -> Option<Selection> {
        let g = self.groups.iter().flatten().find(|g| g.handle == group)?;
        g.selection.or_else(|| {
            self.commitments
                .sealed
                .iter()
                .filter(|s| s.evidence.group == group)
                .max_by_key(|s| {
                    (
                        s.evidence.end,
                        s.evidence.start,
                        std::cmp::Reverse(s.evidence.occurrence_id),
                    )
                })
                .map(|s| Selection {
                    occurrence_id: s.evidence.occurrence_id,
                    start_sample: s.evidence.start,
                    support_end_sample: s.evidence.end,
                    selected_at: self.end,
                    last_observed_sample: s
                        .descriptor
                        .supporting_audio_end
                        .unwrap_or(s.evidence.end),
                    window_start_sample: s.evidence.start as f64,
                    weighted_seconds: s.evidence.assignment_seconds,
                })
        })
    }

    pub(in crate::temporal_cognition) fn committed(
        &self,
    ) -> impl Iterator<Item = (&super::commitment::Evidence, &descriptor::Frozen)> {
        self.commitments
            .sealed
            .iter()
            .map(|s| (&s.evidence, s.descriptor.as_ref()))
    }

    pub(in crate::temporal_cognition) fn prefix(
        &self,
        group: Handle,
        cut: u64,
    ) -> Result<Option<(descriptor::Frozen, Selection)>, &'static str> {
        if cut != self.end {
            return Err("phrase cue must be selected at the query observation cut");
        }
        let Some(g) = self.groups.iter().flatten().find(|g| g.handle == group) else {
            return Ok(None);
        };
        let Some(selected) = self.selection(group) else {
            return Ok(None);
        };
        let frozen = if let Some(s) = self
            .commitments
            .sealed
            .iter()
            .find(|s| s.evidence.occurrence_id == selected.occurrence_id)
        {
            let mut frozen = s.descriptor.as_ref().clone();
            frozen.captured_at = cut;
            frozen
        } else {
            let e = g
                .entries
                .iter()
                .find(|e| e.foreground.credit == selected.occurrence_id)
                .unwrap();
            e.span.prefix(cut)?
        };
        if (frozen.start * f64::from(self.rate)).round() as u64 != selected.start_sample
            || (frozen.end * f64::from(self.rate)).round() as u64 != selected.support_end_sample
        {
            return Err("selected phrase cue differs from its original descriptor support");
        }
        Ok(Some((frozen, selected)))
    }
}

#[cfg(test)]
pub(in crate::temporal_cognition) mod tests;
