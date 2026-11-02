use super::*;
use graph::{Chunk, Graph};
use matching::Matcher;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Context {
    pub id: u64,
    pub version: u64,
    pub mass: f64,
    pub responsibility: f64,
    pub last: f64,
    pub representative: u64,
}
#[derive(Clone, Copy)]
struct Membership {
    chunk: u64,
    anchors: usize,
    ids: [u64; KMAX],
    weights: [f64; KMAX],
}
pub(super) struct Contexts {
    pub active: [Option<Context>; KMAX],
    members: VecDeque<Membership>,
    next_id: u64,
    completed: u64,
    pub unknown: f64,
    pub evidence: [f64; KMAX],
    pub evidence_ids: [u64; KMAX],
}
impl Contexts {
    pub fn new(capacity: usize) -> Self {
        Self {
            active: [None; KMAX],
            members: VecDeque::with_capacity(capacity),
            next_id: 1,
            completed: 0,
            unknown: 1.,
            evidence: [0.; KMAX],
            evidence_ids: [0; KMAX],
        }
    }
    pub fn bytes(&self) -> usize {
        self.members.capacity() * size_of::<Membership>()
    }
    pub fn expire(&mut self, g: &Graph) {
        self.members.retain(|m| g.chunk(m.chunk).is_some());
        for slot in &mut self.active {
            let Some(c) = slot else {
                continue;
            };
            let old_mass = c.mass;
            c.mass = self
                .members
                .iter()
                .map(|m| {
                    let retained = g
                        .chunk(m.chunk)
                        .map_or(0, |x| (x.last - x.first + 1) as usize);
                    m.ids
                        .iter()
                        .zip(m.weights)
                        .filter(|(id, _)| **id == c.id)
                        .map(|(_, w)| w)
                        .sum::<f64>()
                        * retained as f64
                        / m.anchors as f64
                })
                .sum();
            if c.mass != old_mass {
                c.version += 1;
            }
            if g.chunk(c.representative).is_none()
                && let Some(member) = self.members.iter().find(|m| m.ids.contains(&c.id))
            {
                c.representative = member.chunk;
                c.version += 1;
            }
            if c.mass == 0. {
                *slot = None;
            }
        }
    }
    pub fn complete(&mut self, g: &Graph, matcher: &Matcher) {
        self.evidence.fill(0.);
        self.evidence_ids.fill(0);
        let start = g.chunks.partition_point(|c| c.id <= self.completed);
        for chunk in g.chunks.range(start..).filter(|c| c.closed && c.learned) {
            if chunk.id <= self.completed {
                continue;
            }
            let n = (chunk.last - chunk.first + 1) as usize;
            let time = g
                .observation(g.anchor(chunk.last).expect("retained chunk"))
                .time;
            self.score(g, matcher, time);
            let sum = self.evidence.iter().sum::<f64>();
            let best = self.evidence.iter().copied().fold(0., f64::max);
            let mut member = Membership {
                chunk: chunk.id,
                anchors: n,
                ids: [0; KMAX],
                weights: [0.; KMAX],
            };
            if self.unknown > best {
                let slot = self
                    .active
                    .iter()
                    .position(Option::is_none)
                    .unwrap_or_else(|| {
                        (0..KMAX)
                            .min_by(|a, b| {
                                let a = self.active[*a].unwrap();
                                let b = self.active[*b].unwrap();
                                a.mass
                                    .total_cmp(&b.mass)
                                    .then(a.last.total_cmp(&b.last))
                                    .then(a.id.cmp(&b.id))
                            })
                            .unwrap()
                    });
                if let Some(old) = self.active[slot] {
                    for m in &mut self.members {
                        for j in 0..KMAX {
                            if m.ids[j] == old.id {
                                m.weights[j] = 0.;
                                m.ids[j] = 0;
                            }
                        }
                    }
                }
                let id = self.next_id;
                self.next_id += 1;
                self.active[slot] = Some(Context {
                    id,
                    version: 1,
                    mass: 1.,
                    responsibility: 1.,
                    last: time,
                    representative: chunk.id,
                });
                member.ids[slot] = id;
                member.weights[slot] = 1.;
                for (j, c) in self.active.iter_mut().enumerate() {
                    if j != slot
                        && let Some(c) = c
                    {
                        c.responsibility = 0.;
                    }
                }
            } else if sum > 0. {
                for (i, c) in self
                    .active
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(i, c)| c.as_mut().map(|c| (i, c)))
                {
                    let w = self.evidence[i] / sum;
                    c.responsibility = w;
                    if w > 0. {
                        c.mass += w;
                        c.last = time;
                        c.version += 1;
                        member.ids[i] = c.id;
                        member.weights[i] = w;
                    }
                }
            }
            self.members.push_back(member);
            self.completed = chunk.id;
            self.merge_identical(g);
        }
        if let Some(chunk) = g.chunks.back().filter(|c| !c.closed && c.learned)
            && let Some(last) = g.anchor(chunk.last)
        {
            self.score(g, matcher, g.observation(last).time);
        }
    }
    fn score(&mut self, g: &Graph, matcher: &Matcher, observed_at: f64) {
        self.evidence.fill(0.);
        self.evidence_ids = std::array::from_fn(|i| self.active[i].map_or(0, |c| c.id));
        let total = self.active.iter().flatten().map(|c| c.mass).sum::<f64>();
        // The retained context prior cancels its mixture denominator. Missing,
        // pruned and unproposed endpoints keep their empirical mass unknown.
        if total > 0. {
            for mapping in &matcher.best[..matcher.len] {
                let Some(anchor) = g.anchor(mapping.endpoint) else {
                    continue;
                };
                let Some(node) = g.node(anchor.node) else {
                    continue;
                };
                let Ok(index) = self.members.binary_search_by_key(&node.chunk, |m| m.chunk) else {
                    continue;
                };
                let member = &self.members[index];
                for (i, context) in self.active.iter().enumerate() {
                    let Some(context) = context else { continue };
                    let membership = member
                        .ids
                        .iter()
                        .zip(member.weights)
                        .filter(|(id, _)| **id == context.id)
                        .map(|(_, w)| w)
                        .sum::<f64>();
                    self.evidence[i] +=
                        membership / member.anchors as f64 / total * mapping.weight();
                }
            }
        }
        let sum = self.evidence.iter().sum::<f64>();
        self.unknown = (1. - sum).clamp(0., 1.);
        for (i, c) in self
            .active
            .iter_mut()
            .enumerate()
            .filter_map(|(i, c)| c.as_mut().map(|c| (i, c)))
        {
            let responsibility = if sum > 0. { self.evidence[i] / sum } else { 0. };
            if c.responsibility != responsibility || (responsibility > 0. && c.last != observed_at)
            {
                c.version += 1;
            }
            if responsibility > 0. {
                c.last = observed_at;
            }
            c.responsibility = responsibility;
        }
    }

    /// Equality is checked on records, not a similarity threshold or a hash.
    pub fn merge_identical(&mut self, g: &Graph) {
        for i in 0..KMAX {
            for j in i + 1..KMAX {
                let (Some(a), Some(b)) = (self.active[i], self.active[j]) else {
                    continue;
                };
                if !self.same_summary(g, a, b) {
                    continue;
                }
                for m in &mut self.members {
                    let wa = m
                        .ids
                        .iter()
                        .zip(m.weights)
                        .filter(|(id, _)| **id == a.id)
                        .map(|(_, w)| w)
                        .sum::<f64>();
                    let wb = m
                        .ids
                        .iter()
                        .zip(m.weights)
                        .filter(|(id, _)| **id == b.id)
                        .map(|(_, w)| w)
                        .sum::<f64>();
                    for k in 0..KMAX {
                        if m.ids[k] == a.id || m.ids[k] == b.id {
                            m.ids[k] = 0;
                            m.weights[k] = 0.;
                        }
                    }
                    if wa + wb > 0. {
                        m.ids[i] = a.id;
                        m.weights[i] = (wa + wb).min(1.);
                    }
                }
                self.active[j] = None;
                let c = self.active[i].as_mut().unwrap();
                c.version = a.version.max(b.version) + 1;
                c.last = a.last.max(b.last);
                c.responsibility = (a.responsibility + b.responsibility).min(1.);
                c.mass = self.members.iter().map(|m| m.weights[i]).sum();
            }
        }
    }
    fn same_summary(&self, g: &Graph, a: Context, b: Context) -> bool {
        let (Some(ac), Some(bc)) = (g.chunk(a.representative), g.chunk(b.representative)) else {
            return false;
        };
        if !same_chunk(g, ac, bc) {
            return false;
        }
        // Compare normalized membership measures in both directions, including successors.
        for (left, right) in [(a, b), (b, a)] {
            for m in &self.members {
                if !m.ids.contains(&left.id) {
                    continue;
                }
                let Some(chunk) = g.chunk(m.chunk) else {
                    return false;
                };
                let mut mass_left = 0.;
                let mut mass_right = 0.;
                for other in &self.members {
                    let Some(other_chunk) = g.chunk(other.chunk) else {
                        continue;
                    };
                    if !same_chunk(g, chunk, other_chunk) {
                        continue;
                    }
                    for (id, w) in other.ids.iter().zip(other.weights) {
                        if *id == left.id {
                            mass_left += w / left.mass;
                        }
                        if *id == right.id {
                            mass_right += w / right.mass;
                        }
                    }
                }
                if mass_left != mass_right {
                    return false;
                }
            }
        }
        true
    }
}
fn same_chunk(g: &Graph, a: &Chunk, b: &Chunk) -> bool {
    if a.last - a.first != b.last - b.first {
        return false;
    }
    let (Some(aa), Some(bb)) = (g.anchor(a.first), g.anchor(b.first)) else {
        return false;
    };
    let at = g.observation(aa).time;
    let bt = g.observation(bb).time;
    for delta in 0..=a.last - a.first {
        let (Some(x), Some(y)) = (g.anchor(a.first + delta), g.anchor(b.first + delta)) else {
            return false;
        };
        let xo = g.observation(x);
        let yo = g.observation(y);
        if xo.time - at != yo.time - bt || !same_state(&xo, &yo) {
            return false;
        }
        match (g.successor(x), g.successor(y)) {
            (Some(xs), Some(ys)) => {
                let xs = g.observation(xs);
                let ys = g.observation(ys);
                if xs.time - xo.time != ys.time - yo.time || !same_state(&xs, &ys) {
                    return false;
                }
            }
            (None, None) => (),
            _ => return false,
        }
    }
    true
}
fn same_state(a: &Observation<'_>, b: &Observation<'_>) -> bool {
    a.acoustic.coordinate_version == b.acoustic.coordinate_version
        && a.acoustic.mass == b.acoustic.mass
        && a.acoustic.focus == b.acoustic.focus
        && a.acoustic.competition == b.acoustic.competition
        && a.support.bins == b.support.bins
        && a.support.residual == b.support.residual
        && a.support.valid == b.support.valid
        && a.support.error == b.support.error
        && a.support.bound == b.support.bound
        && a.timing_error == b.timing_error
        && a.events == b.events
        && a.trajectories == b.trajectories
}

#[cfg(test)]
mod tests {
    use super::super::tests::{Input, model, observe};
    use super::*;

    #[test]
    fn context_mixture_keeps_incompatible_members_as_unknown() {
        let mut m = model();
        let first = Input::new(0., 1).boundary();
        observe(&mut m, &first);
        let second = Input::new(1., 1).boundary();
        let mut o = second.observation();
        let incompatible = [Event {
            source: None,
            flags: OFFSET | BOUNDARY,
        }];
        o.events = &incompatible;
        observe_raw(&mut m, o);
        let third = Input::new(2., 1).boundary();
        let mut o = third.observation();
        o.events = &incompatible;
        observe_raw(&mut m, o);
        // A single context with three equally responsible observed members.
        m.contexts.active = [None; KMAX];
        m.contexts.active[0] = Some(Context {
            id: 100,
            version: 1,
            mass: 3.,
            responsibility: 1.,
            last: 2.,
            representative: 1,
        });
        for member in &mut m.contexts.members {
            member.ids = [0; KMAX];
            member.weights = [0.; KMAX];
            member.ids[0] = 100;
            member.weights[0] = 1.;
        }
        let query = Input::new(10., 1).boundary();
        let mut query_obs = query.observation();
        let only_opportunity = [Event {
            source: None,
            flags: OPPORTUNITY,
        }];
        query_obs.events = &only_opportunity;
        query_obs.trajectories = &[];
        let mut matcher = Matcher::new();
        matcher.prepare(8);
        let mut mean = 0.;
        for member in &m.contexts.members {
            let chunk = m
                .graph
                .chunks
                .iter()
                .find(|c| c.id == member.chunk)
                .unwrap();
            matcher.len = 0;
            mean +=
                matcher.endpoint(&m.graph, &[query_obs], m.graph.anchor(chunk.last).unwrap()) / 3.;
        }
        // Only the first endpoint has the requested relation type; max would give 1.
        assert!(mean > 0. && mean <= 1. / 3.);
        let query_input = Input::new(10., 1);
        let mut query = query_input.observation();
        let events = [Event {
            source: None,
            flags: OPPORTUNITY,
        }];
        query.events = &events;
        query.trajectories = &[];
        // Close this resource chunk explicitly, without adding a musical boundary flag.
        m.graph.insert(&query, true).unwrap();
        m.graph.chunks.back_mut().unwrap().closed = true;
        matcher.search(&m.graph, &[query], true);
        m.contexts.complete(&m.graph, &matcher);
        assert!((m.contexts.unknown - (1. - mean)).abs() < 1e-10);
    }
    fn observe_raw(m: &mut Model, o: Observation<'_>) {
        m.observe(o).unwrap();
    }

    #[test]
    fn identical_contexts_merge_observation_memberships_once() {
        let mut m = model();
        observe(&mut m, &Input::new(0., 1).boundary());
        let original = m.contexts.active[0].unwrap();
        let duplicate = Context { id: 50, ..original };
        m.contexts.active[1] = Some(duplicate);
        let member = m.contexts.members.front_mut().unwrap();
        member.ids[1] = 50;
        member.weights[0] = 0.5;
        member.weights[1] = 0.5;
        m.contexts.active[0].as_mut().unwrap().mass = 0.5;
        m.contexts.active[1].as_mut().unwrap().mass = 0.5;
        m.contexts.merge_identical(&m.graph);
        assert_eq!(m.contexts.active.iter().flatten().count(), 1);
        assert_eq!(m.contexts.members[0].weights.iter().sum::<f64>(), 1.);
        assert!(m.contexts.active[0].unwrap().version > original.version);
    }
    #[test]
    fn partial_expiration_subtracts_responsibility_and_keeps_live_representative() {
        let mut m = model();
        observe(&mut m, &Input::new(0., 1));
        observe(&mut m, &Input::new(1., 2).boundary());
        m.graph.expire(600.5);
        m.contexts.expire(&m.graph);
        assert_eq!(m.contexts.active[0].unwrap().mass, 0.5);
    }
}

#[cfg(test)]
mod capacity_tests {
    use super::super::tests::{Input, model};
    use super::*;
    #[test]
    fn replacement_uses_responsibility_age_and_id_without_erasing_history() {
        let mut m = model();
        let old = Input::new(0., 1).boundary();
        m.graph.insert(&old.observation(), true).unwrap();
        let mut member = Membership {
            chunk: 1,
            anchors: 1,
            ids: [0; KMAX],
            weights: [1. / KMAX as f64; KMAX],
        };
        for i in 0..KMAX {
            let id = 100 + i as u64;
            member.ids[i] = id;
            m.contexts.active[i] = Some(Context {
                id,
                version: 1,
                mass: 1. / KMAX as f64,
                responsibility: 1. / KMAX as f64,
                last: 0.,
                representative: 1,
            });
        }
        m.contexts.members.push_back(member);
        m.contexts.completed = 1;
        m.contexts.next_id = 1000;
        let fresh = Input::new(10., 7).boundary();
        let mut o = fresh.observation();
        o.acoustic.mass = Value::exact(1000.);
        o.acoustic.focus.valid = false;
        o.acoustic.competition.valid = false;
        o.support.valid = false;
        o.trajectories = &[];
        m.graph.insert(&o, true).unwrap();
        m.contexts.complete(&m.graph, &m.matcher);
        // The fifteen identical survivors merge; the newly born context stays distinct.
        assert_eq!(m.contexts.active.iter().flatten().count(), 2);
        assert_eq!(m.contexts.active[0].unwrap().id, 1000);
        assert!(m.graph.anchor(1).is_some());
        assert!(m.contexts.members.iter().all(|m| !m.ids.contains(&100)));
    }
}
