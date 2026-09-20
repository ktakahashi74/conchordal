//! Bounded shared admission from current acoustic support and sealed section returns.

use super::{Known, Support, build};
use crate::temporal_cognition::{
    joint::{owner, section::Returns},
    ridge::Handle,
    section::{Head, input::Observation},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::temporal_cognition::joint) enum Kind {
    New,
    Contrast,
    Retrieved,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::temporal_cognition::joint) struct Origin {
    pub kind: Kind,
    pub group: Handle,
    pub cut: u64,
    pub support: Support,
    pub query: Option<u64>,
    pub episode: Option<(u64, u64)>,
    pub section: Option<u64>,
    pub local_parent: Option<Handle>,
}

impl Origin {
    fn identity(&self) -> (u64, Kind, Option<(u64, u64)>, Option<u64>) {
        (self.group.generation, self.kind, self.episode, self.section)
    }
}

pub(in crate::temporal_cognition::joint) struct Inputs<'a> {
    pub interval: [u64; 2],
    pub rate: u32,
    pub observed: bool,
    pub groups: &'a [Handle],
    pub observations: &'a [Option<&'a Observation>],
    pub returns: &'a [Option<&'a Returns>],
    pub retained_episodes: &'a [(u64, u64)],
    pub head: &'a Head,
    pub rms_reference: f64,
}

#[derive(Clone, Copy)]
struct Candidate {
    origin: Origin,
    score: f64,
}

impl Candidate {
    fn strongest(target: &mut Option<Self>, candidate: Self) {
        if candidate.score > 0.
            && target.is_none_or(|old| {
                candidate.score > old.score
                    || (candidate.score == old.score
                        && candidate.origin.identity() < old.origin.identity())
            })
        {
            *target = Some(candidate);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(in crate::temporal_cognition::joint) struct Proposal {
    pub parent: Option<Handle>,
    pub slot: u8,
    pub key: Option<u64>,
    pub log_transition: f64,
    pub raw_score: f64,
    pub origin: Option<Origin>,
}

pub(in crate::temporal_cognition::joint) struct Batch {
    interval: [u64; 2],
    rate: u32,
    parents: [Option<Handle>; 7],
    groups: [Option<Handle>; 8],
    proposals: [Option<Proposal>; 32],
    pub examined_returns: usize,
    pub examined_sections: usize,
}

impl Batch {
    pub fn entries(&self) -> impl Iterator<Item = &Proposal> {
        self.proposals.iter().flatten()
    }

    pub fn validate(
        &self,
        snapshot: &owner::Snapshot,
        interval: [u64; 2],
        groups: &[Handle],
        extensions: &[owner::ContextExtension<'_>],
        sources: &owner::Sources<'_>,
    ) -> Result<(), &'static str> {
        if self.interval != interval
            || snapshot.end != interval[0]
            || self.parents != snapshot.paths.map(|p| p.map(|p| p.id))
            || groups.len() != self.groups.iter().flatten().count()
            || !groups
                .iter()
                .copied()
                .eq(self.groups.iter().flatten().copied())
            || extensions.len() != self.entries().count()
        {
            return Err("shared proposal batch belongs to another parent cut or inventory");
        }
        for extension in extensions {
            let proposal = self
                .entries()
                .find(|p| p.parent == extension.parent && p.slot == extension.extension)
                .ok_or("missing generated shared proposal")?;
            if extension.key != proposal.key || extension.log_transition != proposal.log_transition
            {
                return Err("shared proposal identity or transition differs from generated source");
            }
            if let Some(origin) = proposal.origin {
                let index = groups
                    .iter()
                    .position(|g| *g == origin.group)
                    .ok_or("retired shared admission group")?;
                let observation = sources
                    .observations
                    .get(index)
                    .copied()
                    .flatten()
                    .ok_or("missing shared admission observation")?;
                let raw = observation
                    .raw
                    .ok_or("missing shared admission raw evidence")?
                    .raw;
                if observation.rate != self.rate
                    || !observation.observed
                    || observation.interval != interval
                    || origin.cut != interval[1]
                    || origin.support
                        != (Support {
                            interval: [raw.start, raw.end],
                            source_start: raw.source_start,
                            available: raw.available_end,
                            assignment_seconds: observation.delta.assignment_seconds,
                        })
                {
                    return Err("shared admission differs from original acoustic evidence");
                }
                if origin
                    .episode
                    .is_some_and(|e| !sources.retained_episodes.contains(&e))
                {
                    return Err("retired shared return target");
                }
            }
        }
        Ok(())
    }

    pub fn origin(&self, parent: Option<Handle>, slot: u8) -> Option<Origin> {
        self.entries()
            .find(|p| p.parent == parent && p.slot == slot)
            .and_then(|p| p.origin)
    }
}

pub(in crate::temporal_cognition::joint) struct Producer {
    next_key: u64,
    cut: u64,
}

impl Producer {
    pub fn new(start: u64) -> Self {
        Self {
            next_key: 1,
            cut: start,
        }
    }

    pub fn produce(
        &mut self,
        snapshot: &owner::Snapshot,
        input: Inputs<'_>,
    ) -> Result<Batch, &'static str> {
        let [start, end] = input.interval;
        if snapshot.end != start
            || end <= start
            || input.rate == 0
            || input.groups.len() > 8
            || input.groups.len() != input.observations.len()
            || input.groups.len() != input.returns.len()
            || !input.rms_reference.is_finite()
            || input.rms_reference < 1e-6
            || end < self.cut
            || self.next_key == 0
            || snapshot
                .paths
                .iter()
                .flatten()
                .any(|p| p.state.key >= self.next_key)
        {
            return Err("invalid shared producer clock, inventory or next identity");
        }
        input.head.validate()?;
        for (i, group) in input.groups.iter().enumerate() {
            if group.generation == 0
                || group.bus > 1
                || input.groups[..i].contains(group)
                || input
                    .groups
                    .first()
                    .is_some_and(|g| (g.bus, g.epoch) != (group.bus, group.epoch))
                || input.returns[i].is_some_and(|r| r.group != *group)
                || input.observations[i].is_some_and(|o| {
                    o.group != *group
                        || o.interval != input.interval
                        || o.rate != input.rate
                        || (!input.observed && o.observed)
                        || !o.delta.assignment_seconds.is_finite()
                        || o.delta.assignment_seconds < 0.
                        || !o.delta.physical_window_seconds.is_finite()
                        || o.delta.physical_window_seconds <= 0.
                        || o.delta.assignment_seconds > o.delta.physical_window_seconds + 1e-12
                })
            {
                return Err("invalid shared admission acoustic owner or support");
            }
        }
        let mut batch = Batch {
            interval: input.interval,
            rate: input.rate,
            parents: snapshot.paths.map(|p| p.map(|p| p.id)),
            groups: std::array::from_fn(|i| input.groups.get(i).copied()),
            proposals: [None; 32],
            examined_returns: 0,
            examined_sections: 0,
        };
        let mut next = self.next_key;
        let mut used = 0;
        for parent_slot in 0..8 {
            let parent = snapshot.paths.get(parent_slot).copied().flatten();
            if parent_slot < 7 && parent.is_none() {
                continue;
            }
            let mut retrieved = None;
            let mut novel = None;
            for (g, group) in input.groups.iter().copied().enumerate() {
                let Some(observation) = input.observations[g].filter(|o| o.observed) else {
                    continue;
                };
                let Some(raw) = observation.raw.map(|r| r.raw) else {
                    continue;
                };
                if raw.group != group
                    || raw.start < start
                    || raw.start >= raw.end
                    || raw.end != end
                    || raw.source_start > raw.start
                    || raw.source_end < raw.end
                    || raw.available_end > end
                    || raw.source_end > raw.available_end
                    || raw.known_samples != raw.end - raw.start
                {
                    return Err("invalid shared admission raw interval");
                }
                let alpha = observation.delta.assignment_seconds
                    / observation.delta.physical_window_seconds;
                if alpha <= 0. {
                    continue;
                }
                let base = Origin {
                    kind: Kind::New,
                    group,
                    cut: end,
                    support: Support {
                        interval: [raw.start, raw.end],
                        source_start: raw.source_start,
                        available: raw.available_end,
                        assignment_seconds: observation.delta.assignment_seconds,
                    },
                    query: None,
                    episode: None,
                    section: None,
                    local_parent: None,
                };
                if let Some(returns) = input.returns[g] {
                    for proof in returns.entries() {
                        batch.examined_returns += 1;
                        let Some((correspondence, query)) = proof.current(end, input.rate) else {
                            continue;
                        };
                        let Some(matched) = correspondence.matched else {
                            continue;
                        };
                        if correspondence.group != group
                            || !input
                                .retained_episodes
                                .contains(&(matched.episode_id, matched.episode_generation))
                        {
                            continue;
                        }
                        Candidate::strongest(
                            &mut retrieved,
                            Candidate {
                                origin: Origin {
                                    kind: Kind::Retrieved,
                                    query: Some(query.query_id),
                                    episode: Some((matched.episode_id, matched.episode_generation)),
                                    section: Some(proof.context().context_id),
                                    ..base
                                },
                                score: alpha * proof.context().mass,
                            },
                        );
                    }
                }
                // The existing low-energy gate cannot initialize or force a new context in quiet.
                if !raw.values[2]
                    .is_some_and(|v| v.is_finite() && v > (0.01 * input.rms_reference).log2())
                {
                    continue;
                }
                let old_group = snapshot.groups.iter().position(|h| *h == Some(group));
                let old = parent.zip(old_group);
                let mut unknown_section = 1.;
                if let Some((p, i)) = old {
                    let local = &snapshot.posterior.contexts[parent_slot]
                        .as_ref()
                        .ok_or("missing shared conditional prior")?
                        .groups[i];
                    for (slot, path) in p.groups[i]
                        .iter()
                        .enumerate()
                        .filter_map(|(i, p)| p.map(|p| (i, p)))
                    {
                        let Some(section_slot) = path.section_slot else {
                            continue;
                        };
                        let mass = local.rows[slot]
                            .as_ref()
                            .ok_or("missing shared local prior")?
                            .mass;
                        unknown_section -= mass;
                        let section = &snapshot.sections[section_slot];
                        if section.group != group || section.end != start || section.start > start {
                            return Err("invalid shared section parent support");
                        }
                        batch.examined_sections += 1;
                        let times = [start - section.start, end - section.start]
                            .map(|n| n as f64 / f64::from(input.rate));
                        if let Some(law) = input.head.law(&section.values, times[0], times[1])? {
                            for (kind, score) in [(Kind::New, law[1]), (Kind::Contrast, law[3])] {
                                Candidate::strongest(
                                    &mut novel,
                                    Candidate {
                                        origin: Origin {
                                            kind,
                                            section: path.components[3],
                                            local_parent: Some(path.id),
                                            ..base
                                        },
                                        score: alpha * mass * score,
                                    },
                                );
                            }
                        }
                    }
                }
                Candidate::strongest(
                    &mut novel,
                    Candidate {
                        origin: base,
                        score: alpha * unknown_section.max(0.),
                    },
                );
            }
            let mut candidates = [None, None, retrieved, novel];
            let mut keys = [parent.map(|p| p.state.key), None, None, None];
            for i in 2..4 {
                if let Some(c) = candidates[i] {
                    keys[i] = if let Some(p) = parent.filter(|p| {
                        p.state
                            .admission
                            .is_some_and(|o| o.identity() == c.origin.identity())
                    }) {
                        Some(p.state.key)
                    } else {
                        let key = next;
                        next = next
                            .checked_add(1)
                            .ok_or("shared admission identity exhausted")?;
                        Some(key)
                    };
                }
            }
            let choices = build(
                parent.map(|p| p.state.key),
                parent.map(|p| Known {
                    path: p.state.key,
                    raw_score: 1.,
                }),
                keys[2].zip(retrieved).map(|(path, c)| Known {
                    path,
                    raw_score: c.score,
                }),
                keys[3].zip(novel).map(|(path, c)| Known {
                    path,
                    raw_score: c.score,
                }),
                (end - start) as f64 / f64::from(input.rate),
                input.observed,
            )?;
            for (slot, choice) in choices.into_iter().enumerate() {
                let Some(choice) = choice else { continue };
                let candidate = candidates[slot].take();
                batch.proposals[used] = Some(Proposal {
                    parent: parent.map(|p| p.id),
                    slot: slot as u8,
                    key: choice.path,
                    log_transition: choice.log_transition,
                    raw_score: candidate.map_or(f64::from(slot == 0), |c| c.score),
                    origin: candidate.map(|c| c.origin),
                });
                used += 1;
            }
        }
        self.next_key = next;
        self.cut = end;
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_cognition::phrase;

    #[test]
    fn proposal_identity_overflow_is_atomic_and_replanning_never_reuses_keys() {
        let group = Handle {
            bus: 1,
            epoch: 0,
            generation: 2,
        };
        let owner = owner::State::new(1, 0, 0, 100).unwrap();
        let observation =
            Observation::new(&phrase::tests::input(1, 0.), group, [0, 100], 1000, None).unwrap();
        let head = Head {
            means: [0.; 82],
            deviations: [1.; 82],
            hazard: [0.; 83],
            exits: [[0.; 83]; 3],
        };
        let groups = [group];
        let observations = [Some(&observation)];
        let returns = [None];
        let input = || Inputs {
            interval: [0, 100],
            rate: 1000,
            observed: true,
            groups: &groups,
            observations: &observations,
            returns: &returns,
            retained_episodes: &[],
            head: &head,
            rms_reference: 0.1,
        };
        let mut producer = Producer {
            next_key: u64::MAX,
            cut: 0,
        };
        assert_eq!(
            producer.produce(owner.snapshot(), input()).err(),
            Some("shared admission identity exhausted")
        );
        assert_eq!((producer.next_key, producer.cut), (u64::MAX, 0));
        let mut producer = Producer::new(0);
        let first = producer.produce(owner.snapshot(), input()).unwrap();
        let second = producer.produce(owner.snapshot(), input()).unwrap();
        let key = |batch: &Batch| batch.entries().find_map(|p| p.key).unwrap();
        assert!(
            key(&second) > key(&first),
            "rejected ownership can be replanned without recycling an identity"
        );
        assert_eq!(producer.cut, 100);
        let mut invalid = input();
        invalid.interval = [0, 99];
        assert!(producer.produce(owner.snapshot(), invalid).is_err());
        assert_eq!(producer.cut, 100);
    }
}
