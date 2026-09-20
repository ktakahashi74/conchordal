//! Bounded transition proposals; observation log-potentials are evaluated after composition.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Articulation,
    Grouping,
    Phrase,
    Section,
    Correspondence,
}
const KINDS: [Kind; 5] = [
    Kind::Articulation,
    Kind::Grouping,
    Kind::Phrase,
    Kind::Section,
    Kind::Correspondence,
];
const CAPS: [usize; 5] = [8, 19, 6, 5, 19];
const KNOWN: [usize; 5] = [4, 18, 5, 4, 18];
const TAU: [f64; 5] = [10., 10., 30., 120., 10.];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::temporal_cognition) enum Boundary {
    Stay,
    Exit,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Known {
    pub id: u64,
    pub raw_score: f64,
    // Exact component identity stay; an in-context revision can also keep the boundary.
    pub stay: bool,
    pub boundary: Option<Boundary>,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::temporal_cognition) struct Choice {
    pub(in crate::temporal_cognition) id: Option<u64>,
    pub(in crate::temporal_cognition) log_weight: f64,
    pub(in crate::temporal_cognition) stay: bool,
    pub(in crate::temporal_cognition) boundary: Option<Boundary>,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::temporal_cognition) struct List {
    pub(super) kind: Kind,
    pub(in crate::temporal_cognition) entries: [Option<Choice>; 19],
    pub len: usize,
    pub truncated: usize,
}

#[derive(Clone, Copy)]
pub(super) struct PhraseExtension {
    pub parent_slot: Option<usize>,
    pub kind: Option<super::super::phrase::Exit>,
    pub id: u64,
}

impl PhraseExtension {
    pub(super) fn validate(
        &self,
        parent: Option<&super::super::phrase::interpretation::Interpretation>,
        [start, end]: [u64; 2],
        observed: bool,
    ) -> Result<(), &'static str> {
        if start >= end || self.id == 0 || self.parent_slot.is_some() != parent.is_some() {
            return Err("invalid joint phrase clock or identity");
        }
        match parent {
            None if !observed || self.kind.is_some() => {
                return Err("unsupported joint phrase admission");
            }
            Some(p)
                if (self.kind.is_none() && self.id != p.id)
                    || (self.kind.is_some() && self.id <= p.id)
                    || p.foreground
                        .is_some_and(|f| f.start > start || f.heard_end > start)
                    || p.event.is_some_and(|(_, at)| at > start)
                    || [p.ending, p.completed_ending]
                        .iter()
                        .flatten()
                        .any(|d| d.end > start || d.available > start) =>
            {
                return Err("invalid joint phrase parent identity or evidence");
            }
            _ => {}
        }
        Ok(())
    }

    pub(super) fn materialize(
        &self,
        parent: Option<&super::super::phrase::interpretation::Interpretation>,
        interval: [u64; 2],
        observed: bool,
    ) -> Result<super::super::phrase::interpretation::Interpretation, &'static str> {
        self.validate(parent, interval, observed)?;
        let Some(parent) = parent else {
            return super::super::phrase::interpretation::Interpretation::new(self.id, interval);
        };
        let mut child = *parent;
        child.completed_foreground = None;
        child.completed_ending = None;
        child.advance(self.kind, interval, observed, self.kind.map(|_| self.id))?;
        Ok(child)
    }
}

impl List {
    pub(super) fn missing_boundary(
        kind: Kind,
        id: u64,
        survival: f64,
        dt: f64,
    ) -> Result<Self, &'static str> {
        if !matches!(kind, Kind::Phrase | Kind::Section)
            || id == 0
            || !survival.is_finite()
            || !(0. ..=1.).contains(&survival)
            || !dt.is_finite()
            || dt < 0.
        {
            return Err("invalid missing-boundary survival");
        }
        let log_stay = survival.ln() - dt / TAU[kind as usize];
        let mut entries = [None; 19];
        let mut len = 0;
        if log_stay.is_finite() {
            entries[0] = Some(Choice {
                id: Some(id),
                log_weight: log_stay,
                stay: true,
                boundary: Some(Boundary::Stay),
            });
            len += 1;
        }
        entries[len] = Some(Choice {
            id: None,
            log_weight: (-log_stay.exp_m1()).ln(),
            stay: false,
            boundary: None,
        });
        entries[..=len]
            .sort_unstable_by(|a, b| b.unwrap().log_weight.total_cmp(&a.unwrap().log_weight));
        Ok(Self {
            kind,
            entries,
            len: len + 1,
            truncated: 0,
        })
    }
    pub(super) fn observed_articulation(
        parent: Option<super::super::gesture::articulation::Articulation>,
        input: &super::super::gesture::articulation::Input,
        config: &crate::config::TemporalGestureConfig,
    ) -> Result<
        (
            Self,
            [Option<super::super::gesture::articulation::Articulation>; 4],
        ),
        &'static str,
    > {
        use super::super::gesture::State;
        let list = Self::articulation(
            parent.map(|p| p.state),
            input.rates(parent, config)?,
            input.dt,
            input.observed,
            input.low,
        )?;
        let mut children = [None; 4];
        for choice in list.entries[..list.len].iter().flatten() {
            if let Some(id) = choice.id {
                children[id as usize] = Some(input.child(
                    parent,
                    [
                        State::Attack,
                        State::Continuation,
                        State::Release,
                        State::Gap,
                    ][id as usize],
                ));
            }
        }
        Ok((list, children))
    }

    pub(super) fn build(
        kind: Kind,
        raw: &[Known],
        unknown_parent: bool,
        dt: f64,
        cap: usize,
    ) -> Result<Self, &'static str> {
        let index = kind as usize;
        if !dt.is_finite() || dt < 0. || cap == 0 || cap > CAPS[index] || raw.len() > KNOWN[index] {
            return Err("invalid component clock or capacity");
        }
        if raw.iter().any(|e| {
            !e.raw_score.is_finite()
                || e.raw_score < 0.
                || (kind == Kind::Articulation && e.id > 3)
                || (matches!(kind, Kind::Phrase | Kind::Section)
                    && (e.boundary.is_none() || (e.stay && e.boundary != Some(Boundary::Stay))))
                || (!matches!(kind, Kind::Phrase | Kind::Section) && e.boundary.is_some())
                || (unknown_parent && e.stay)
        }) || raw.iter().filter(|e| e.stay).count() > 1
        {
            return Err("invalid component score, identity or stay");
        }
        for (i, entry) in raw.iter().enumerate() {
            if raw[..i].iter().any(|e| e.id == entry.id) {
                return Err("duplicate component identity");
            }
        }
        let mut order = [0; 18];
        for (i, target) in order.iter_mut().enumerate().take(raw.len()) {
            *target = i;
        }
        order[..raw.len()].sort_unstable_by(|&a, &b| {
            raw[b]
                .raw_score
                .total_cmp(&raw[a].raw_score)
                .then(raw[a].id.cmp(&raw[b].id))
        });
        let mut admitted = [0; 18];
        let mut count = 0;
        if let Some((i, _)) = raw
            .iter()
            .enumerate()
            .find(|(_, e)| e.stay && e.raw_score > 0.)
        {
            if cap < 2 {
                return Err("component capacity cannot reserve stay and unknown");
            }
            admitted[count] = i;
            count += 1;
        }
        for &i in &order[..raw.len()] {
            if raw[i].raw_score > 0. && count < cap - 1 && !admitted[..count].contains(&i) {
                admitted[count] = i;
                count += 1;
            }
        }
        let maximum = admitted[..count]
            .iter()
            .map(|&i| raw[i].raw_score)
            .fold(0., f64::max);
        let total = admitted[..count]
            .iter()
            .map(|&i| raw[i].raw_score / maximum)
            .sum::<f64>();
        let mut entries = [None; 19];
        for (slot, &i) in admitted[..count].iter().enumerate() {
            let e = raw[i];
            entries[slot] = Some(Choice {
                id: Some(e.id),
                log_weight: e.raw_score.ln() - maximum.ln() - total.ln() - dt / TAU[index],
                stay: e.stay,
                boundary: e.boundary,
            });
        }
        entries[count] = Some(Choice {
            id: None,
            log_weight: if count == 0 {
                0.
            } else {
                (-(-dt / TAU[index]).exp_m1()).ln()
            },
            stay: unknown_parent,
            boundary: None,
        });
        entries[..=count].sort_unstable_by(|a, b| {
            let (a, b) = (a.unwrap(), b.unwrap());
            b.log_weight
                .total_cmp(&a.log_weight)
                .then((a.id.is_none(), a.id).cmp(&(b.id.is_none(), b.id)))
        });
        Ok(Self {
            kind,
            entries,
            len: count + 1,
            truncated: raw.iter().filter(|e| e.raw_score > 0.).count() - count,
        })
    }

    pub(super) fn articulation(
        parent: Option<super::super::gesture::State>,
        rates: [f64; 4],
        dt: f64,
        observed: bool,
        low: bool,
    ) -> Result<Self, &'static str> {
        let proposal =
            super::super::gesture::articulation_proposals(parent, rates, dt, observed, low)?;
        let maximum = proposal.scores.into_iter().fold(0., f64::max);
        let total: f64 = if maximum > 0. {
            proposal.scores.iter().map(|v| v / maximum).sum()
        } else {
            0.
        };
        let weights: [f64; 5] = std::array::from_fn(|i| {
            if i == 4 {
                proposal.unknown.ln()
            } else if proposal.scores[i] == 0. {
                f64::NEG_INFINITY
            } else {
                proposal.scores[i].ln() - maximum.ln() - total.ln() + proposal.log_keep
            }
        });
        let mut entries = [None; 19];
        let mut len = 0;
        for (i, log_weight) in weights.into_iter().enumerate() {
            if i < 4 && !log_weight.is_finite() {
                continue;
            }
            entries[len] = Some(Choice {
                id: (i < 4).then_some(i as u64),
                log_weight,
                stay: parent.map_or(i == 4, |s| i == s as usize),
                boundary: None,
            });
            len += 1;
        }
        entries[..len].sort_unstable_by(|a, b| {
            let (a, b) = (a.unwrap(), b.unwrap());
            b.log_weight
                .total_cmp(&a.log_weight)
                .then((a.id.is_none(), a.id).cmp(&(b.id.is_none(), b.id)))
        });
        Ok(Self {
            kind: Kind::Articulation,
            entries,
            len,
            truncated: 0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn owned_active_phrase(
        (parent_slot, parent): (usize, &super::super::phrase::interpretation::Interpretation),
        exit_ids: [u64; 4],
        values: [Option<f64>; 12],
        articulation: Option<super::super::gesture::State>,
        interval: [u64; 2],
        sample_rate: u32,
        observed_low: bool,
        config: crate::config::TemporalPhraseConfig,
    ) -> Result<(Self, [PhraseExtension; 5]), &'static str> {
        use super::super::phrase::Exit;
        let foreground = parent
            .foreground
            .ok_or("active phrase requires an active parent")?;
        if interval[0] < foreground.start || interval[1] <= interval[0] {
            return Err("invalid owned phrase interval");
        }
        let ids = [
            parent.id,
            exit_ids[0],
            exit_ids[1],
            exit_ids[2],
            exit_ids[3],
        ];
        let children = std::array::from_fn(|i| PhraseExtension {
            parent_slot: Some(parent_slot),
            kind: [
                None,
                Some(Exit::New),
                Some(Exit::Overlap),
                Some(Exit::Reinterpret),
                Some(Exit::Inactive),
            ][i],
            id: ids[i],
        });
        for child in &children {
            child.validate(Some(parent), interval, true)?;
        }
        let list = Self::active_phrase(
            ids,
            values,
            articulation,
            interval.map(|t| t - foreground.start),
            sample_rate,
            observed_low,
            config,
        )?;
        Ok((list, children))
    }

    pub(super) fn active_phrase(
        ids: [u64; 5],
        values: [Option<f64>; 12],
        articulation: Option<super::super::gesture::State>,
        elapsed_samples: [u64; 2],
        sample_rate: u32,
        observed_low: bool,
        config: crate::config::TemporalPhraseConfig,
    ) -> Result<Self, &'static str> {
        if sample_rate == 0 || elapsed_samples[1] < elapsed_samples[0] {
            return Err("invalid active phrase proposal duration");
        }
        let duration = elapsed_samples.map(|n| n as f64 / f64::from(sample_rate));
        let dt = (elapsed_samples[1] - elapsed_samples[0]) as f64 / f64::from(sample_rate);
        let law =
            super::super::phrase::conditional_law::<true>(values, articulation, duration, config);
        let raw: [Known; 5] = std::array::from_fn(|i| Known {
            id: ids[i],
            raw_score: law.as_ref().map_or(0., |law| {
                if i == 0 {
                    law.survival
                } else if i == 4 && !observed_low {
                    0.
                } else {
                    law.exits.map_or(0., |(scores, total)| {
                        (1. - law.survival) * scores[i - 1] / total
                    })
                }
            }),
            stay: i == 0,
            boundary: Some(if i == 0 {
                Boundary::Stay
            } else {
                Boundary::Exit
            }),
        });
        Self::build(Kind::Phrase, &raw, false, dt, 6)
    }

    pub(in crate::temporal_cognition) fn owned_active_section(
        ids: [u64; 4],
        parent: &super::super::section::interpretation::Interpretation,
        interval: [u64; 2],
        sample_rate: u32,
        returned: Option<super::super::section::commitment::ContextSupport>,
        head: &super::super::section::Head,
    ) -> Result<Self, &'static str> {
        if interval[0] != parent.end
            || parent.start > interval[0]
            || interval[1] <= interval[0]
            || returned.is_some_and(|r| r.context_id == 0)
        {
            return Err("invalid owned section interval or return context");
        }
        Self::active_section(
            ids,
            &parent.values,
            interval.map(|t| t - parent.start),
            sample_rate,
            returned.map(|r| (r.mass, r.context_id == parent.context)),
            head,
        )
    }

    // IDs name stay/new/return/contrast child paths; the return flag means current context.
    pub(in crate::temporal_cognition) fn active_section(
        ids: [u64; 4],
        values: &[Option<f64>; 82],
        elapsed_samples: [u64; 2],
        sample_rate: u32,
        returned: Option<(f64, bool)>,
        head: &super::super::section::Head,
    ) -> Result<Self, &'static str> {
        if sample_rate == 0
            || elapsed_samples[1] < elapsed_samples[0]
            || returned
                .is_some_and(|(support, _)| !support.is_finite() || !(0. ..=1.).contains(&support))
        {
            return Err("invalid active section duration or return support");
        }
        let duration = elapsed_samples.map(|n| n as f64 / f64::from(sample_rate));
        let dt = (elapsed_samples[1] - elapsed_samples[0]) as f64 / f64::from(sample_rate);
        let law = head.law(values, duration[0], duration[1])?;
        let raw = std::array::from_fn::<_, 4, _>(|i| Known {
            id: ids[i],
            raw_score: law.map_or(0., |p| p[i])
                * if i == 2 {
                    returned.map_or(0., |r| r.0)
                } else {
                    1.
                },
            stay: i == 0,
            boundary: Some(if i == 0 || (i == 2 && returned.is_some_and(|r| r.1)) {
                Boundary::Stay
            } else {
                Boundary::Exit
            }),
        });
        Self::build(Kind::Section, &raw, false, dt, 5)
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Tuple {
    pub indices: [usize; 5],
    pub ids: [Option<u64>; 5],
    pub log_product: f64,
    pub log_transition: f64,
    pub all_unknown: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Output {
    pub tuples: [Option<Tuple>; 16],
    pub priority_evaluations: usize,
    pub heap_pops: usize,
    pub compatibility_checks: usize,
    pub correspondence_checks: usize,
    pub reserved_boundary_pairs: u8,
    pub reserved_stay: bool,
    pub log_admitted_product_mass: f64,
}

pub(super) struct Compatibility {
    pub phrase_section: [[bool; 5]; 6],
    pub section_correspondence: [[bool; 19]; 5],
}

impl Default for Compatibility {
    fn default() -> Self {
        Self {
            phrase_section: [[true; 5]; 6],
            section_correspondence: [[true; 19]; 5],
        }
    }
}

impl Compatibility {
    fn allows(&self, indices: [usize; 5]) -> bool {
        self.phrase_section[indices[2]][indices[3]]
            && self.section_correspondence[indices[3]][indices[4]]
    }
}

#[derive(Clone, Copy)]
struct Pending {
    index: usize,
    score: f64,
    key: [(bool, u64); 5],
}
impl Ord for Pending {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then(other.key.cmp(&self.key))
    }
}
impl PartialOrd for Pending {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Pending {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Pending {}

pub(super) struct Composer {
    cache: [Option<Tuple>; 86],
    index: Box<[u8]>,
    keys: [usize; 86],
    count: usize,
    heap: BinaryHeap<Pending>,
    queued: [bool; 86],
}

impl Composer {
    pub(super) fn new() -> Self {
        Self {
            cache: [None; 86],
            index: vec![u8::MAX; CAPS.iter().product()].into_boxed_slice(),
            keys: [0; 86],
            count: 0,
            heap: BinaryHeap::with_capacity(86),
            queued: [false; 86],
        }
    }

    fn priority(&mut self, lists: &[List; 5], indices: [usize; 5]) -> usize {
        let key = indices
            .into_iter()
            .zip(CAPS)
            .fold(0, |key, (index, cap)| key * cap + index);
        if self.index[key] != u8::MAX {
            return usize::from(self.index[key]);
        }
        assert!(self.count < 86);
        self.index[key] = self.count as u8;
        self.keys[self.count] = key;
        let ids = std::array::from_fn(|g| lists[g].entries[indices[g]].unwrap().id);
        self.cache[self.count] = Some(Tuple {
            indices,
            ids,
            log_product: (0..5)
                .map(|g| lists[g].entries[indices[g]].unwrap().log_weight)
                .sum(),
            log_transition: f64::NEG_INFINITY,
            all_unknown: ids.iter().all(Option::is_none),
        });
        self.count += 1;
        self.count - 1
    }

    pub(super) fn compose(
        &mut self,
        lists: &[List; 5],
        legal: &Compatibility,
    ) -> Result<Output, &'static str> {
        if lists.iter().zip(KINDS).any(|(l, k)| l.kind != k) {
            return Err("wrong component order");
        }
        // Clear only the keys visited by the preceding composition.
        for &key in &self.keys[..self.count] {
            self.index[key] = u8::MAX;
        }
        self.count = 0;
        self.heap.clear();
        self.queued.fill(false);
        let mut chosen = [usize::MAX; 16];
        let mut count = 0;
        let mut compatibility_checks = 1;
        let unknown: [usize; 5] = std::array::from_fn(|g| {
            lists[g].entries[..lists[g].len]
                .iter()
                .position(|e| e.unwrap().id.is_none())
                .unwrap()
        });
        if !legal.allows(unknown) {
            return Err("all-unknown tuple must remain legal");
        }
        chosen[count] = self.priority(lists, unknown);
        count += 1;
        let stay: [Option<usize>; 5] = std::array::from_fn(|g| {
            lists[g].entries[..lists[g].len]
                .iter()
                .position(|e| e.unwrap().stay)
        });
        let mut reserved_stay = false;
        if stay.iter().all(Option::is_some) {
            let indices = stay.map(Option::unwrap);
            compatibility_checks += 1;
            if legal.allows(indices) {
                reserved_stay = true;
                let i = self.priority(lists, indices);
                if !chosen[..count].contains(&i) {
                    chosen[count] = i;
                    count += 1;
                }
            }
        }
        // Each section can require a different fresh correspondence witness.
        let mut correspondence_checks = 0;
        let best_correspondence: [Option<usize>; 5] = std::array::from_fn(|s| {
            if s >= lists[3].len {
                return None;
            }
            (0..lists[4].len).find(|&c| {
                correspondence_checks += 1;
                legal.section_correspondence[s][c]
            })
        });
        let mut best: [Option<(f64, [usize; 5])>; 4] = [None; 4];
        for (p, row) in legal.phrase_section.iter().enumerate().take(lists[2].len) {
            for (s, &allowed) in row.iter().enumerate().take(lists[3].len) {
                compatibility_checks += 1;
                let (a, b) = (lists[2].entries[p].unwrap(), lists[3].entries[s].unwrap());
                let (Some(pa), Some(sb)) = (a.boundary, b.boundary) else {
                    continue;
                };
                if !allowed {
                    continue;
                }
                let Some(c) = best_correspondence[s] else {
                    continue;
                };
                let category = 2 * pa as usize + sb as usize;
                let indices = [0, 0, p, s, c];
                let correspondence = lists[4].entries[c].unwrap();
                let score = a.log_weight + b.log_weight + correspondence.log_weight;
                if best[category].is_none_or(|(old, old_indices)| {
                    score > old
                        || (score == old
                            && (a.id, b.id, correspondence.id)
                                < (
                                    lists[2].entries[old_indices[2]].unwrap().id,
                                    lists[3].entries[old_indices[3]].unwrap().id,
                                    lists[4].entries[old_indices[4]].unwrap().id,
                                ))
                }) {
                    best[category] = Some((score, indices));
                }
            }
        }
        let mut reserved_boundary_pairs = 0;
        for (category, entry) in best.into_iter().enumerate() {
            if let Some((_, indices)) = entry {
                reserved_boundary_pairs |= 1 << category;
                let i = self.priority(lists, indices);
                if !chosen[..count].contains(&i) {
                    chosen[count] = i;
                    count += 1;
                }
            }
        }
        let root = self.priority(lists, [0; 5]);
        let t = self.cache[root].unwrap();
        self.heap.push(Pending {
            index: root,
            score: t.log_product,
            key: t.ids.map(|id| (id.is_none(), id.unwrap_or(0))),
        });
        self.queued[root] = true;
        let mut pops = 0;
        while pops < 16 && count < 16 {
            let Some(node) = self.heap.pop() else {
                break;
            };
            pops += 1;
            compatibility_checks += 1;
            let tuple = self.cache[node.index].unwrap();
            if legal.allows(tuple.indices) && !chosen[..count].contains(&node.index) {
                chosen[count] = node.index;
                count += 1;
            }
            if pops == 16 || count == 16 {
                break;
            }
            for g in 0..5 {
                let mut indices = tuple.indices;
                indices[g] += 1;
                if indices[g] >= lists[g].len {
                    continue;
                }
                let i = self.priority(lists, indices);
                if !self.queued[i] {
                    let t = self.cache[i].unwrap();
                    self.heap.push(Pending {
                        index: i,
                        score: t.log_product,
                        key: t.ids.map(|id| (id.is_none(), id.unwrap_or(0))),
                    });
                    self.queued[i] = true;
                }
            }
        }
        let maximum = chosen[..count]
            .iter()
            .map(|&i| self.cache[i].unwrap().log_product)
            .fold(f64::NEG_INFINITY, f64::max);
        if !maximum.is_finite() {
            return Err("no positive legal tuple transition");
        }
        let log_sum = chosen[..count]
            .iter()
            .map(|&i| (self.cache[i].unwrap().log_product - maximum).exp())
            .sum::<f64>()
            .ln();
        Ok(Output {
            tuples: std::array::from_fn(|i| {
                (i < count).then(|| {
                    let mut t = self.cache[chosen[i]].unwrap();
                    t.log_transition = (t.log_product - maximum) - log_sum;
                    t
                })
            }),
            priority_evaluations: self.count,
            heap_pops: pops,
            compatibility_checks,
            correspondence_checks,
            reserved_boundary_pairs,
            reserved_stay,
            log_admitted_product_mass: maximum + log_sum,
        })
    }
}

#[cfg(test)]
mod tests;
