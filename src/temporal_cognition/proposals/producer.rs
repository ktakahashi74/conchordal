//! Causal bundle-derived keys, frozen parent bindings, and reducer integration.

use super::*;
use crate::temporal_cognition::grouping::{Bundles, Correlations, complete_link};

pub(super) struct Generator {
    tracker: Tracker,
    previous: Option<Assignment>,
    last_handles: [Option<Handle>; 8],
    parents: [[Option<Handle>; 8]; CAPACITY],
}

pub(super) struct Output {
    pub bundles: Bundles,
    pub proposals: Update,
    pub candidates: [Option<Candidate>; CAPACITY],
    pub continued: usize,
    pub cross_pair_reads: usize,
}

fn cross_extremum(key: Key, correlations: &Correlations, reads: &mut usize) -> Option<f64> {
    let mut score: f64 = if key.kind == Kind::Split { -1.0 } else { 1.0 };
    for a in key.sets[0].iter().flatten() {
        for b in key.sets[1].iter().flatten() {
            let i = correlations.handles.iter().position(|h| *h == Some(*a))?;
            let j = correlations.handles.iter().position(|h| *h == Some(*b))?;
            *reads += 1;
            let c = correlations.values[i][j]?;
            score = if key.kind == Kind::Split {
                score.max(c)
            } else {
                score.min(c)
            };
        }
    }
    Some(score)
}

impl Generator {
    pub fn new(
        bus: u8,
        epoch: u64,
        start: u64,
        hop: u64,
        persistence: u8,
    ) -> Result<Self, &'static str> {
        Ok(Self {
            tracker: Tracker::new(bus, epoch, start, hop, persistence)?,
            previous: None,
            last_handles: [None; 8],
            parents: [[None; 8]; CAPACITY],
        })
    }

    pub fn advance(
        &mut self,
        fully_observed: bool,
        correlations: &Correlations,
        assignment: &Assignment,
        eligible_parents: &[Handle],
    ) -> Result<Output, &'static str> {
        let end = assignment.end_sample;
        if end <= self.tracker.last_end
            || !end.is_multiple_of(self.tracker.hop)
            || correlations.window_end != end
            || eligible_parents.len() > 7
            || eligible_parents.iter().enumerate().any(|(i, h)| {
                h.bus != self.tracker.bus
                    || h.epoch != self.tracker.epoch
                    || h.generation == 0
                    || eligible_parents[..i].contains(h)
                    || !assignment.group_handles.contains(&Some(*h))
            })
        {
            return Err("invalid proposal source frame or eligible parents");
        }
        let bundles = complete_link(correlations, 0.8)?;
        for row in assignment.rows.iter().flatten() {
            if row.trajectory.bus != self.tracker.bus || row.trajectory.epoch != self.tracker.epoch
            {
                return Err("foreign proposal assignment row");
            }
            Candidate::measure(
                Key::new(Kind::Birth, &[], &[row.trajectory], &[])?,
                assignment,
                None,
            )?;
        }
        if correlations.handles.iter().flatten().any(|h| {
            h.bus != self.tracker.bus
                || h.epoch != self.tracker.epoch
                || !assignment.rows.iter().flatten().any(|r| r.trajectory == *h)
        }) {
            return Err("supported envelope handle has no current association");
        }
        let previous = self
            .previous
            .as_ref()
            .filter(|p| end - p.end_sample == self.tracker.hop);
        let fresh_parents: [Option<Handle>; 8] = std::array::from_fn(|i| {
            let h = correlations.handles[i]?;
            let p = previous?;
            let row = p.rows.iter().flatten().find(|r| r.trajectory == h)?;
            former_parent(row, &p.group_handles).filter(|p| eligible_parents.contains(p))
        });
        let mut candidates: [Option<Candidate>; CAPACITY] = [None; CAPACITY];
        let mut bindings = [[None; 8]; CAPACITY];
        let mut continuations = [None; CAPACITY];
        let mut count = 0;
        let mut continued = 0;
        let mut reads = 0;
        if fully_observed {
            if end - self.tracker.last_end == self.tracker.hop {
                for (slot, entry) in self
                    .tracker
                    .pending
                    .iter()
                    .enumerate()
                    .filter_map(|(i, e)| e.map(|e| (i, e)))
                {
                    let key = entry.candidate.key;
                    if key
                        .parents
                        .iter()
                        .flatten()
                        .any(|p| !eligible_parents.contains(p))
                    {
                        continue;
                    }
                    let bound: [Option<Handle>; 8] = std::array::from_fn(|i| {
                        let h = correlations.handles[i]?;
                        self.last_handles
                            .iter()
                            .position(|old| *old == Some(h))
                            .map_or(fresh_parents[i], |old| self.parents[slot][old])
                    });
                    let structure_matches = match key.kind {
                        Kind::Birth => bundles.members[..bundles.count].contains(&key.sets[0]),
                        Kind::Split => key
                            .sets
                            .iter()
                            .all(|s| bundles.members[..bundles.count].contains(s)),
                        Kind::Merge => bundles.members[..bundles.count].iter().any(|bundle| {
                            let mut sets = [[None; 8]; 2];
                            let mut lengths = [0; 2];
                            for h in bundle.iter().flatten() {
                                let i = correlations
                                    .handles
                                    .iter()
                                    .position(|c| *c == Some(*h))
                                    .unwrap();
                                for side in 0..2 {
                                    if bound[i] == key.parents[side] {
                                        sets[side][lengths[side]] = Some(*h);
                                        lengths[side] += 1;
                                    }
                                }
                            }
                            sets == key.sets
                        }),
                    };
                    if !structure_matches {
                        continue;
                    }
                    let score = (key.kind != Kind::Birth)
                        .then(|| cross_extremum(key, correlations, &mut reads))
                        .flatten();
                    if let Some(c) = Candidate::measure(key, assignment, score)? {
                        candidates[count] = Some(c);
                        bindings[count] = bound;
                        continuations[continued] = Some(key);
                        count += 1;
                        continued += 1;
                    }
                }
            }
            for bundle in &bundles.members[..bundles.count] {
                let n = bundle.iter().flatten().count();
                let mut members = [Handle {
                    bus: 0,
                    epoch: 0,
                    generation: 0,
                }; 8];
                for (dst, h) in members.iter_mut().zip(bundle.iter().flatten()) {
                    *dst = *h;
                }
                let key = Key::new(Kind::Birth, &[], &members[..n], &[])?;
                if !candidates[..count].iter().flatten().any(|c| c.key == key)
                    && let Some(c) = Candidate::measure(key, assignment, None)?
                {
                    candidates[count] = Some(c);
                    bindings[count] = fresh_parents;
                    count += 1;
                }
            }
            for a in 0..bundles.count {
                for b in a + 1..bundles.count {
                    let sets = [bundles.members[a], bundles.members[b]];
                    if candidates[..count]
                        .iter()
                        .flatten()
                        .any(|c| c.key.kind == Kind::Split && c.key.sets == sets)
                    {
                        continue;
                    }
                    let mut parent = None;
                    let mut homogeneous = true;
                    let mut members = [[Handle {
                        bus: 0,
                        epoch: 0,
                        generation: 0,
                    }; 8]; 2];
                    let mut lengths = [0; 2];
                    for side in 0..2 {
                        for h in sets[side].iter().flatten() {
                            let i = correlations
                                .handles
                                .iter()
                                .position(|c| *c == Some(*h))
                                .unwrap();
                            let p = fresh_parents[i];
                            if p.is_none() || parent.is_some_and(|previous| Some(previous) != p) {
                                homogeneous = false;
                            }
                            if parent.is_none() {
                                parent = p;
                            }
                            members[side][lengths[side]] = *h;
                            lengths[side] += 1;
                        }
                    }
                    if !homogeneous {
                        continue;
                    }
                    let key = Key::new(
                        Kind::Split,
                        &[parent.unwrap()],
                        &members[0][..lengths[0]],
                        &members[1][..lengths[1]],
                    )?;
                    let score = cross_extremum(key, correlations, &mut reads);
                    if let Some(c) = Candidate::measure(key, assignment, score)? {
                        candidates[count] = Some(c);
                        bindings[count] = fresh_parents;
                        count += 1;
                    }
                }
            }
            // One live key per unordered parent pair preserves the registered 21-key bound.
            let mut parents = [None; 7];
            for (dst, &h) in parents.iter_mut().zip(eligible_parents) {
                *dst = Some(h);
            }
            parents[..eligible_parents.len()].sort_unstable();
            for a in 0..eligible_parents.len() {
                for b in a + 1..eligible_parents.len() {
                    let pair = [parents[a], parents[b]];
                    if candidates[..count]
                        .iter()
                        .flatten()
                        .any(|c| c.key.kind == Kind::Merge && c.key.parents == pair)
                    {
                        continue;
                    }
                    let mut best: Option<Candidate> = None;
                    for bundle in &bundles.members[..bundles.count] {
                        let mut members = [[Handle {
                            bus: 0,
                            epoch: 0,
                            generation: 0,
                        }; 8]; 2];
                        let mut lengths = [0; 2];
                        for h in bundle.iter().flatten() {
                            let i = correlations
                                .handles
                                .iter()
                                .position(|c| *c == Some(*h))
                                .unwrap();
                            for side in 0..2 {
                                if fresh_parents[i] == pair[side] {
                                    members[side][lengths[side]] = *h;
                                    lengths[side] += 1;
                                }
                            }
                        }
                        if lengths.contains(&0) {
                            continue;
                        }
                        let key = Key::new(
                            Kind::Merge,
                            &[pair[0].unwrap(), pair[1].unwrap()],
                            &members[0][..lengths[0]],
                            &members[1][..lengths[1]],
                        )?;
                        let score = cross_extremum(key, correlations, &mut reads);
                        if let Some(c) = Candidate::measure(key, assignment, score)?
                            && best.is_none_or(|previous| {
                                c.score > previous.score
                                    || (c.score == previous.score
                                        && (c.key.members, c.key.sets)
                                            < (previous.key.members, previous.key.sets))
                            })
                        {
                            best = Some(c);
                        }
                    }
                    if let Some(c) = best {
                        candidates[count] = Some(c);
                        bindings[count] = fresh_parents;
                        count += 1;
                    }
                }
            }
        }
        let mut packed = [Candidate {
            key: Key {
                kind: Kind::Birth,
                parents: [None; 2],
                sets: [[None; 8]; 2],
                members: [None; 8],
            },
            end_sample: end,
            support: 0.0,
            score: 0.0,
        }; CAPACITY];
        for (dst, c) in packed.iter_mut().zip(candidates[..count].iter().flatten()) {
            *dst = *c;
        }
        let saved = self.tracker.pending;
        for entry in &mut self.tracker.pending {
            if entry.is_some_and(|e| !continuations[..continued].contains(&Some(e.candidate.key))) {
                *entry = None;
            }
        }
        let proposals = match self.tracker.advance(end, fully_observed, &packed[..count]) {
            Ok(out) => out,
            Err(error) => {
                self.tracker.pending = saved;
                return Err(error);
            }
        };
        for (slot, entry) in self.tracker.pending.iter().enumerate() {
            self.parents[slot] = if let Some(entry) = entry {
                let i = candidates[..count]
                    .iter()
                    .position(|c| c.is_some_and(|c| c.key == entry.candidate.key))
                    .unwrap();
                bindings[i]
            } else {
                [None; 8]
            };
        }
        self.last_handles = correlations.handles;
        self.previous = fully_observed.then_some(*assignment);
        Ok(Output {
            bundles,
            proposals,
            candidates,
            continued,
            cross_pair_reads: reads,
        })
    }
}

#[cfg(test)]
mod tests;
