//! Retained joint priors and path identity across ordered acoustic intervals.

use super::proposals::PhraseExtension;
use super::{Normalizer, Pair, Posterior, Shared};
use super::{acoustics, correspondence, grouping, section, shared};
use crate::temporal_cognition::gesture::articulation::Articulation;
use crate::temporal_cognition::phrase::interpretation::Interpretation;
use crate::temporal_cognition::ridge::Handle;
use crate::temporal_cognition::section::input::{Observation, Phrase, Span};
use crate::temporal_cognition::section::interpretation::Interpretation as Section;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LocalPath {
    pub id: Handle,
    pub parent: Option<Handle>,
    pub components: [Option<u64>; 5],
    pub articulation: Option<Articulation>,
    pub phrase_slot: Option<usize>,
    pub section_slot: Option<usize>,
    pub correspondence_slot: Option<usize>,
    pub grouping_slot: Option<usize>,
    pub grouping_supported: bool,
    pub correspondence_supported: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ContextPath {
    pub id: Handle,
    pub parent: Option<Handle>,
    pub state: shared::State,
    pub groups: [[Option<LocalPath>; 15]; 8],
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Snapshot {
    pub end: u64,
    pub groups: [Option<Handle>; 8],
    pub paths: [Option<ContextPath>; 7],
    pub posterior: Posterior,
    pub phrases: Box<[Option<Interpretation>]>,
    pub correspondences: Box<[Option<correspondence::State>]>,
    pub groupings: Box<[Option<grouping::State>]>,
    pub sections: Box<[Section]>,
    pub phrase_activity: Box<[Span]>,
}

#[derive(Clone, Copy)]
pub(super) struct LocalExtension {
    pub parent: Option<Handle>,
    pub extension: u8,
    pub components: [Option<u64>; 5],
    pub articulation: Option<Articulation>,
    pub phrase: Option<PhraseExtension>,
    pub correspondence: Option<correspondence::Extension>,
    pub grouping: Option<grouping::Extension>,
    pub section: Option<section::Extension>,
    pub log_transition: f64,
    pub log_potential: f64,
}

pub(super) struct ContextExtension<'a> {
    pub parent: Option<Handle>,
    pub extension: u8,
    pub key: Option<u64>,
    pub log_transition: f64,
    pub log_potential: f64,
    pub groups: &'a [&'a [LocalExtension]],
}

#[derive(Default)]
pub(super) struct Sources<'a> {
    pub correspondences: &'a [Option<&'a correspondence::Cache>],
    pub groupings: &'a [Option<&'a grouping::Cache>],
    pub retained_episodes: &'a [(u64, u64)],
    pub sections: &'a [section::Source<'a>],
    pub observations: &'a [Option<&'a Observation>],
    pub section_config: Option<&'a crate::config::TemporalSectionConfig>,
    pub shared: Option<&'a shared::producer::Batch>,
    pub acoustics: &'a [Option<&'a acoustics::Cache>],
}

pub(super) struct State {
    bus: u8,
    epoch: u64,
    hop: u64,
    next_id: u64,
    maximum_group_generation: u64,
    current: Box<Snapshot>,
    next: Box<Snapshot>,
    normalizer: Normalizer,
    pairs: Vec<Pair>,
}

impl State {
    pub(super) fn new(bus: u8, epoch: u64, start: u64, hop: u64) -> Result<Self, &'static str> {
        if bus > 1 || hop == 0 || !start.is_multiple_of(hop) {
            return Err("invalid joint owner clock");
        }
        let snapshot = || {
            Box::new(Snapshot {
                end: start,
                groups: [None; 8],
                paths: [None; 7],
                phrases: vec![None; 7 * 8 * 15].into_boxed_slice(),
                phrase_activity: vec![Span::default(); 7 * 8 * 15].into_boxed_slice(),
                correspondences: vec![None; 7 * 8 * 15].into_boxed_slice(),
                groupings: vec![None; 7 * 8 * 15].into_boxed_slice(),
                sections: (0..7 * 8 * 15)
                    .map(|_| {
                        Section::new(
                            Handle {
                                bus,
                                epoch,
                                generation: 0,
                            },
                            start,
                            1,
                            0,
                        )
                        .expect("valid reserved section arena")
                    })
                    .collect(),
                posterior: Posterior {
                    explicit_unknown: 1.,
                    ..Posterior::default()
                },
            })
        };
        Ok(Self {
            bus,
            epoch,
            hop,
            next_id: 1,
            maximum_group_generation: 0,
            next: snapshot(),
            current: snapshot(),
            normalizer: Normalizer::new(),
            pairs: Vec::with_capacity(32 * 8 * 256),
        })
    }

    pub(super) fn snapshot(&self) -> &Snapshot {
        &self.current
    }

    pub(super) fn advance(
        &mut self,
        interval: [u64; 2],
        observed: bool,
        groups: &[Handle],
        extensions: &[ContextExtension<'_>],
        sources: Sources<'_>,
    ) -> Result<&Snapshot, &'static str> {
        if interval[0] != self.current.end
            || interval[1] <= interval[0]
            || !interval[1].is_multiple_of(self.hop)
            || (observed && interval[1] - interval[0] != self.hop)
        {
            return Err("noncausal joint owner interval");
        }
        if (!sources.correspondences.is_empty() && sources.correspondences.len() != groups.len())
            || sources
                .correspondences
                .iter()
                .zip(groups)
                .any(|(c, g)| c.is_some_and(|c| c.group != *g))
            || (!sources.groupings.is_empty() && sources.groupings.len() != groups.len())
            || sources
                .groupings
                .iter()
                .zip(groups)
                .any(|(c, g)| c.is_some_and(|c| c.group != *g))
            || sources.sections.len() > 32 * 8 * 256
            || groups.len() > 8
            || groups.iter().enumerate().any(|(i, g)| {
                g.bus != self.bus
                    || g.epoch != self.epoch
                    || g.generation == 0
                    || groups[..i].contains(g)
                    || (!self.current.groups.contains(&Some(*g))
                        && g.generation <= self.maximum_group_generation)
            })
            || extensions.is_empty()
            || extensions.len() > 32
            || extensions.iter().any(|e| e.groups.len() != groups.len())
        {
            return Err("invalid joint owner inventory");
        }
        if !sources.observations.is_empty() {
            if sources.observations.len() != groups.len() || sources.section_config.is_none() {
                return Err("missing joint section observation inventory or model");
            }
            for (input, group) in sources.observations.iter().zip(groups) {
                if input.is_some_and(|o| {
                    o.group != *group
                        || o.interval != interval
                        || o.rate == 0
                        || (!observed && o.observed)
                }) {
                    return Err("invalid joint section observation owner or clock");
                }
            }
        }
        if let Some(batch) = sources.shared {
            batch.validate(&self.current, interval, groups, extensions, &sources)?;
        }
        if !sources.acoustics.is_empty() {
            if sources.acoustics.len() != groups.len()
                || sources.observations.len() != groups.len()
                || sources
                    .acoustics
                    .iter()
                    .zip(sources.observations)
                    .any(|(cache, o)| {
                        cache.is_none() || o.is_none_or(|o| !cache.unwrap().matches(o))
                    })
            {
                return Err("joint acoustic cache differs from original observation");
            }
        }
        let mut shared_pairs = [Pair {
            parent: 0,
            extension: 0,
            resolved: false,
            log_prior: f64::NEG_INFINITY,
            log_transition: f64::NEG_INFINITY,
            log_potential: 0.,
        }; 32];
        let mut ranges = [[[0; 2]; 8]; 32];
        let mut seen_shared = [false; 8];
        self.pairs.clear();
        for (index, extension) in extensions.iter().enumerate() {
            let parent = match extension.parent {
                Some(id) => self
                    .current
                    .paths
                    .iter()
                    .position(|p| p.is_some_and(|p| p.id == id))
                    .ok_or("stale joint context parent")?,
                None => 7,
            };
            seen_shared[parent] = true;
            let context = self
                .current
                .posterior
                .contexts
                .get(parent)
                .copied()
                .flatten();
            let prior = context.map_or(
                self.current.posterior.explicit_unknown + self.current.posterior.pruned_mass,
                |c| c.weight.mass,
            );
            if !observed && parent == 7 && extension.key.is_some() {
                return Err("missing input cannot recover a joint context");
            }
            shared_pairs[index] = Pair {
                parent: parent as u8,
                extension: extension.extension,
                resolved: extension.key.is_some(),
                log_prior: prior.ln(),
                log_transition: extension.log_transition,
                log_potential: extension.log_potential,
            };
            for (group, (handle, children)) in groups.iter().zip(extension.groups).enumerate() {
                let observation = sources.observations.get(group).copied().flatten();
                let local_observed = observation.map_or(observed, |o| o.observed);
                if !sources.observations.is_empty()
                    && observation.is_none()
                    && children
                        .iter()
                        .any(|c| c.phrase.is_some() || c.section.is_some())
                {
                    return Err("missing joint section observation");
                }

                if children.is_empty() || children.len() > 256 {
                    return Err("invalid joint local extension capacity");
                }
                let previous_group = self.current.groups.iter().position(|g| *g == Some(*handle));
                let previous = context.zip(previous_group).map(|(c, g)| c.groups[g]);
                let paths = self
                    .current
                    .paths
                    .get(parent)
                    .copied()
                    .flatten()
                    .zip(previous_group)
                    .map(|(p, g)| p.groups[g]);
                let mut seen_local = [false; 16];
                let start = self.pairs.len();
                for child in *children {
                    if child.components[2] != child.phrase.map(|p| p.id) {
                        return Err("joint phrase differs from component identity");
                    }
                    if child.components[0] != child.articulation.map(|p| p.state as u64)
                        || child.articulation.is_some_and(|p| {
                            p.entered > interval[1] || p.run.observed_end > interval[1]
                        })
                    {
                        return Err(
                            "joint articulation state differs from component identity or clock",
                        );
                    }
                    let local_parent = match child.parent {
                        Some(id) => paths
                            .and_then(|paths| {
                                paths.iter().position(|p| p.is_some_and(|p| p.id == id))
                            })
                            .ok_or("stale or foreign joint local parent")?,
                        None => 15,
                    };
                    if let Some(p) = child.phrase {
                        let retained_slot = paths
                            .and_then(|paths| paths.get(local_parent).copied().flatten())
                            .and_then(|p| p.phrase_slot);
                        if p.parent_slot != retained_slot {
                            return Err("joint phrase slot differs from selected local parent");
                        }
                        let retained = retained_slot.and_then(|i| self.current.phrases[i].as_ref());
                        p.validate(retained, interval, local_observed)?;
                    }
                    if child.components[1] != child.grouping.map(|c| c.id) {
                        return Err("joint grouping differs from component identity");
                    }
                    if let Some(choice) = child.grouping {
                        let cache = sources
                            .groupings
                            .get(group)
                            .copied()
                            .flatten()
                            .ok_or("missing joint grouping source")?;
                        let retained = paths
                            .and_then(|paths| paths.get(local_parent).copied().flatten())
                            .and_then(|p| p.grouping_slot)
                            .and_then(|i| self.current.groupings[i].as_ref());
                        cache.resolve(choice, retained, local_observed, interval[1])?;
                    }
                    if child.components[4] != child.correspondence.map(|c| c.id) {
                        return Err("joint correspondence differs from component identity");
                    }
                    let selected_correspondence = if let Some(choice) = child.correspondence {
                        let cache = sources
                            .correspondences
                            .get(group)
                            .copied()
                            .flatten()
                            .ok_or("missing joint correspondence source")?;
                        let retained = paths
                            .and_then(|paths| paths.get(local_parent).copied().flatten())
                            .and_then(|p| p.correspondence_slot)
                            .and_then(|i| self.current.correspondences[i].as_ref());
                        Some(cache.resolve(
                            choice,
                            retained,
                            local_observed,
                            interval[1],
                            sources.retained_episodes,
                        )?)
                    } else {
                        None
                    };
                    if child.components[3] != child.section.map(|s| s.id) {
                        return Err("joint section differs from component identity");
                    }
                    if let Some(choice) = child.section {
                        let old =
                            paths.and_then(|paths| paths.get(local_parent).copied().flatten());
                        if choice.parent_slot != old.and_then(|p| p.section_slot) {
                            return Err("joint section slot differs from selected local parent");
                        }
                        let parent = old
                            .and_then(|p| p.section_slot.zip(p.components[3]))
                            .map(|(i, id)| (&self.current.sections[i], id));
                        let source = sources
                            .sections
                            .get(choice.source_slot)
                            .ok_or("missing joint section source")?;
                        if source.parent != child.parent {
                            return Err("joint section source belongs to another local parent");
                        }
                        if let Some(cache) = sources.correspondences.get(group).copied().flatten() {
                            if cache.rate != source.input.rate
                                || source.input.retrieval
                                    != cache
                                        .retrieval(
                                            interval[1],
                                            local_observed,
                                            sources.retained_episodes,
                                        )?
                                        .values
                            {
                                return Err(
                                    "joint section retrieval differs from original query projection",
                                );
                            }
                        }
                        if let Some(observation) = observation {
                            if source.input.rate != observation.rate
                                || source.input.delta != observation.delta
                                || source.input.completed.is_some()
                                || !source.input.deliveries.is_empty()
                            {
                                return Err("joint section source differs from acoustic assembly");
                            }
                        } else if !sources.observations.is_empty() {
                            return Err("missing joint section observation");
                        }
                        source.validate_return(selected_correspondence)?;
                        choice.validate(parent, source, *handle, interval, local_observed)?;
                    }
                    seen_local[local_parent] = true;
                    let prior = previous.map_or(1., |p| {
                        p.rows
                            .get(local_parent)
                            .copied()
                            .flatten()
                            .map_or(p.explicit_unknown + p.pruned_mass, |w| w.mass)
                    });
                    let resolved = child.components.iter().any(Option::is_some);
                    if !local_observed && local_parent == 15 && resolved {
                        return Err("missing input cannot recover a joint local path");
                    }
                    self.pairs.push(Pair {
                        parent: local_parent as u8,
                        extension: child.extension,
                        resolved,
                        log_prior: prior.ln(),
                        log_transition: child.log_transition,
                        log_potential: if local_observed {
                            child.log_potential
                        } else {
                            0.
                        },
                    });
                }
                if !seen_local[15]
                    || previous.is_some_and(|p| {
                        p.rows
                            .iter()
                            .enumerate()
                            .any(|(i, row)| row.is_some() && !seen_local[i])
                    })
                {
                    return Err("missing joint local parent extensions");
                }
                ranges[index][group] = [start, self.pairs.len()];
            }
        }
        if !seen_shared[7]
            || self
                .current
                .paths
                .iter()
                .enumerate()
                .any(|(i, p)| p.is_some() && !seen_shared[i])
        {
            return Err("missing joint context parent extensions");
        }
        let references: [[&[Pair]; 8]; 32] = std::array::from_fn(|c| {
            std::array::from_fn(|g| {
                let [a, b] = ranges[c][g];
                &self.pairs[a..b]
            })
        });
        let shared: [Shared<'_>; 32] = std::array::from_fn(|i| Shared {
            pair: shared_pairs[i],
            groups: &references[i][..groups.len()],
        });
        let posterior = *self
            .normalizer
            .normalize(&shared[..extensions.len()], observed)?;
        let count = posterior
            .contexts
            .iter()
            .flatten()
            .map(|c| {
                1 + c.groups[..groups.len()]
                    .iter()
                    .map(|g| g.rows.iter().flatten().count())
                    .sum::<usize>()
            })
            .sum::<usize>();
        let after = self
            .next_id
            .checked_add(count as u64)
            .ok_or("joint path identity exhausted")?;
        self.next.end = interval[1];
        self.next.groups.fill(None);
        for (target, &handle) in self.next.groups.iter_mut().zip(groups) {
            *target = Some(handle);
        }
        self.next.paths.fill(None);
        self.next.phrases.fill(None);
        self.next.correspondences.fill(None);
        self.next.groupings.fill(None);
        self.next.posterior = posterior;
        let mut id = self.next_id;
        for (context_slot, context) in posterior
            .contexts
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.map(|c| (i, c)))
        {
            let index = shared_pairs[..extensions.len()]
                .iter()
                .position(|p| {
                    (p.parent, p.extension) == (context.weight.parent, context.weight.extension)
                })
                .unwrap();
            let input = &extensions[index];
            let mut path = ContextPath {
                id: Handle {
                    bus: self.bus,
                    epoch: self.epoch,
                    generation: id,
                },
                parent: input.parent,
                state: shared::State {
                    key: input.key.unwrap(),
                    admission: sources
                        .shared
                        .and_then(|b| b.origin(input.parent, input.extension))
                        .or_else(|| {
                            self.current
                                .paths
                                .get(usize::from(context.weight.parent))
                                .and_then(Option::as_ref)
                                .filter(|p| Some(p.state.key) == input.key)
                                .and_then(|p| p.state.admission)
                        }),
                    end: interval[1],
                    groups: [None; 8],
                },
                groups: [[None; 15]; 8],
            };
            id += 1;
            for (g, local) in context.groups.iter().enumerate().take(groups.len()) {
                let observation = sources.observations.get(g).copied().flatten();
                let local_observed = observation.map_or(observed, |o| o.observed);
                let [a, b] = ranges[index][g];
                for (slot, row) in local
                    .rows
                    .iter()
                    .enumerate()
                    .filter_map(|(i, r)| r.map(|r| (i, r)))
                {
                    let choice = self.pairs[a..b]
                        .iter()
                        .position(|p| (p.parent, p.extension) == (row.parent, row.extension))
                        .unwrap();
                    let child = &input.groups[g][choice];
                    let phrase_slot = child.phrase.map(|proposal| {
                        let index = (context_slot * 8 + g) * 15 + slot;
                        self.next.phrases[index] = Some(
                            proposal
                                .materialize(
                                    proposal
                                        .parent_slot
                                        .and_then(|i| self.current.phrases[i].as_ref()),
                                    interval,
                                    local_observed,
                                )
                                .expect("validated retained phrase extension"),
                        );
                        index
                    });
                    if let Some(index) = phrase_slot
                        && let Some(cache) = sources.acoustics.get(g).copied().flatten()
                    {
                        cache
                            .project(self.next.phrases[index].as_mut().unwrap(), local_observed)?;
                    }
                    let previous_group = self
                        .current
                        .groups
                        .iter()
                        .position(|h| *h == Some(groups[g]));
                    let old_path = self
                        .current
                        .paths
                        .get(usize::from(context.weight.parent))
                        .and_then(Option::as_ref)
                        .zip(previous_group)
                        .and_then(|(c, g)| {
                            c.groups[g].get(usize::from(row.parent)).copied().flatten()
                        });
                    let mut grouping_supported = false;
                    let grouping_slot = child.grouping.map(|choice| {
                        let index = (context_slot * 8 + g) * 15 + slot;
                        let old = old_path
                            .and_then(|p| p.grouping_slot)
                            .and_then(|i| self.current.groupings[i].as_ref());
                        let (payload, supported) = sources.groupings[g]
                            .unwrap()
                            .resolve(choice, old, local_observed, interval[1])
                            .expect("validated retained grouping extension");
                        self.next.groupings[index] = Some(grouping::State {
                            id: choice.id,
                            ..*payload
                        });
                        grouping_supported = supported;
                        index
                    });
                    let mut assembled = observation.copied();
                    if let Some(input) = assembled.as_mut() {
                        input.condition_grouping(
                            grouping_slot
                                .and_then(|i| self.next.groupings[i])
                                .filter(|_| grouping_supported)
                                .map(|s| s.proposal.is_some()),
                        );
                    }
                    let mut completion = None;
                    if let Some(index) = phrase_slot {
                        let old_slot = old_path.and_then(|p| p.phrase_slot);
                        self.next.phrase_activity[index] = old_slot
                            .map_or_else(Span::default, |i| self.current.phrase_activity[i]);
                        if let Some(input) = assembled.as_ref() {
                            let phrase = self.next.phrases[index].unwrap();
                            completion = self.next.phrase_activity[index].advance(
                                input,
                                Phrase {
                                    foreground: phrase.foreground,
                                    completed_foreground: phrase.completed_foreground,
                                    completed_ending: phrase.completed_ending,
                                },
                                old_slot.is_none(),
                                sources.section_config.unwrap(),
                            )?;
                        }
                    }
                    let section_slot = if let Some(choice) = child.section {
                        let index = (context_slot * 8 + g) * 15 + slot;
                        let parent = old_path
                            .and_then(|p| p.section_slot)
                            .map(|i| &self.current.sections[i]);
                        let mut source = sources.sections[choice.source_slot];
                        if let Some(input) = assembled.as_ref() {
                            source.input.delta = input.delta;
                            source.input.deliveries = input.delivery.as_slice();
                            source.input.completed = completion;
                        }
                        choice.materialize(parent, &source, &mut self.next.sections[index])?;
                        Some(index)
                    } else {
                        None
                    };
                    let mut correspondence_supported = false;
                    let correspondence_slot = child.correspondence.map(|choice| {
                        let index = (context_slot * 8 + g) * 15 + slot;
                        let old = old_path
                            .and_then(|p| p.correspondence_slot)
                            .and_then(|i| self.current.correspondences[i].as_ref());
                        let (payload, supported) = sources.correspondences[g]
                            .unwrap()
                            .resolve(
                                choice,
                                old,
                                local_observed,
                                interval[1],
                                sources.retained_episodes,
                            )
                            .expect("validated retained correspondence extension");
                        self.next.correspondences[index] = Some(correspondence::State {
                            id: choice.id,
                            ..*payload
                        });
                        correspondence_supported = supported;
                        index
                    });
                    path.groups[g][slot] = Some(LocalPath {
                        id: Handle {
                            bus: self.bus,
                            epoch: self.epoch,
                            generation: id,
                        },
                        parent: child.parent,
                        components: child.components,
                        articulation: child.articulation,
                        phrase_slot,
                        section_slot,
                        correspondence_slot,
                        correspondence_supported,
                        grouping_slot,
                        grouping_supported,
                    });
                    id += 1;
                }
            }
            // These summaries become inputs only after the final arena swap.
            for (g, group) in groups.iter().copied().enumerate() {
                let local = &context.groups[g];
                let observation = sources.observations.get(g).copied().flatten();
                let support = observation.filter(|o| o.observed).and_then(|o| {
                    o.raw.map(|r| shared::Support {
                        interval: [r.raw.start, r.raw.end],
                        source_start: r.raw.source_start,
                        available: r.raw.available_end,
                        assignment_seconds: o.delta.assignment_seconds,
                    })
                });
                if support
                    .is_some_and(|s| !s.assignment_seconds.is_finite() || s.assignment_seconds < 0.)
                {
                    return Err("invalid shared observation support");
                }
                let old = self
                    .current
                    .paths
                    .get(usize::from(context.weight.parent))
                    .and_then(Option::as_ref)
                    .and_then(|p| p.state.groups.iter().flatten().find(|s| s.group == group));
                let mut summary = shared::Group {
                    group,
                    support,
                    last_observed_end: support
                        .map(|s| s.interval[1])
                        .or_else(|| old.and_then(|s| s.last_observed_end)),
                    explicit_unknown: local.explicit_unknown,
                    pruned_mass: local.pruned_mass,
                    represented_mass: 0.,
                    articulation: [0.; 4],
                    grouping: [0.; 2],
                    phrase: [0.; 2],
                    correspondence: [0.; 2],
                    sections: [None; 15],
                };
                for (slot, p) in path.groups[g]
                    .iter()
                    .enumerate()
                    .filter_map(|(i, p)| p.map(|p| (i, p)))
                {
                    let mass = local.rows[slot].unwrap().mass;
                    summary.represented_mass += mass;
                    if let Some(a) = p.articulation {
                        summary.articulation[a.state as usize] += mass;
                    }
                    if let Some(i) = p.grouping_slot {
                        summary.grouping
                            [usize::from(self.next.groupings[i].unwrap().proposal.is_some())] +=
                            mass;
                    }
                    if let Some(i) = p.phrase_slot {
                        summary.phrase
                            [usize::from(self.next.phrases[i].unwrap().foreground.is_some())] +=
                            mass;
                    }
                    if let Some(i) = p.correspondence_slot {
                        summary.correspondence[usize::from(
                            self.next.correspondences[i].unwrap().matched.is_some(),
                        )] += mass;
                    }
                    if let Some(i) = p.section_slot {
                        let context = self.next.sections[i].context;
                        let slot = summary
                            .sections
                            .iter()
                            .position(|s| s.is_some_and(|s| s.context == context))
                            .or_else(|| summary.sections.iter().position(Option::is_none))
                            .unwrap();
                        let entry = summary.sections[slot]
                            .get_or_insert(shared::Section { context, mass: 0. });
                        entry.mass += mass;
                    }
                }
                summary.sections.sort_by(|a, b| match (a, b) {
                    (Some(a), Some(b)) => b.mass.total_cmp(&a.mass).then(a.context.cmp(&b.context)),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                });
                path.state.groups[g] = Some(summary);
            }
            self.next.paths[context_slot] = Some(path);
        }
        self.next_id = after;
        self.maximum_group_generation = groups
            .iter()
            .map(|g| g.generation)
            .fold(self.maximum_group_generation, u64::max);
        std::mem::swap(&mut self.current, &mut self.next);
        Ok(&self.current)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod correspondence_tests;

#[cfg(test)]
mod grouping_tests;

#[cfg(test)]
mod section_tests;

#[cfg(test)]
mod return_tests;

#[cfg(test)]
mod input_tests;

#[cfg(test)]
mod shared_state_tests;

#[cfg(test)]
mod shared_producer_tests;

#[cfg(test)]
mod local_section_tests;
