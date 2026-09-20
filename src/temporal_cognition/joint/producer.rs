//! Bounded local transition assembly; the caller supplies declared conditional potentials.

use super::{acoustics, correspondence, grouping, owner, proposals, section, shared};
use crate::{
    config::{TemporalGestureConfig, TemporalPhraseConfig, TemporalSectionConfig},
    temporal_cognition::{
        gesture, phrase,
        ridge::Handle,
        section::{Head, input::Observation},
    },
};

pub(super) struct Group<'a> {
    pub observation: &'a Observation,
    pub acoustics: &'a acoustics::Cache,
    pub grouping: &'a grouping::Cache,
    pub correspondence: &'a correspondence::Cache,
    pub returns: &'a section::Returns,
    pub motion: Option<f64>,
    // Frozen earlier-stage context at the previous completed cut, never a new joint posterior.
    pub phrase_context: [Option<f64>; 12],
}

pub(super) struct Input<'a> {
    pub interval: [u64; 2],
    pub rate: u32,
    pub observed: bool,
    pub groups: &'a [Group<'a>],
    pub retained: &'a [(u64, u64)],
    pub gesture: &'a TemporalGestureConfig,
    pub phrase: TemporalPhraseConfig,
    pub section: &'a Head,
    pub section_config: &'a TemporalSectionConfig,
}

#[derive(Clone, Copy, Default)]
pub(super) struct Work {
    pub parents: usize,
    pub tuples: usize,
    pub priorities: usize,
    pub heap_pops: usize,
    pub compatibility_checks: usize,
    pub component_truncated: usize,
}

pub(super) struct Producer {
    next_id: u64,
    next_context: u64,
    cut: u64,
    shared: [Option<shared::producer::Proposal>; 32],
    shared_scores: [f64; 32],
    ranges: [[[usize; 2]; 8]; 32],
    rows: Vec<owner::LocalExtension>,
    sources: Vec<section::Source<'static>>,
    composer: proposals::Composer,
    pub work: Work,
}

// Hold the generated rows and their original sources immutably until ownership advances.
pub(super) struct Batch<'p, 'i> {
    producer: &'p Producer,
    shared: &'i shared::producer::Batch,
    input: &'i Input<'i>,
}

impl Producer {
    pub fn new(start: u64) -> Self {
        Self {
            next_id: 1,
            next_context: 1,
            cut: start,
            shared: [None; 32],
            shared_scores: [0.; 32],
            ranges: [[[0; 2]; 8]; 32],
            rows: Vec::with_capacity(32 * 8 * 16 * 16),
            sources: Vec::with_capacity(32 * 8 * 16 * 4),
            composer: proposals::Composer::new(),
            work: Work::default(),
        }
    }

    pub fn produce<'p, 'i>(
        &'p mut self,
        previous: &owner::Snapshot,
        shared: &'i shared::producer::Batch,
        input: &'i Input<'i>,
        mut shared_score: impl FnMut(&shared::producer::Proposal) -> Result<f64, &'static str>,
        mut local_score: impl FnMut(
            &shared::producer::Proposal,
            usize,
            &owner::LocalExtension,
        ) -> Result<f64, &'static str>,
    ) -> Result<Batch<'p, 'i>, &'static str> {
        let [start, end] = input.interval;
        if previous.end != start
            || start >= end
            || end < self.cut
            || input.rate == 0
            || input.groups.len() > 8
        {
            return Err("invalid local producer clock or inventory");
        }
        phrase::Phrase::validate(input.phrase)?;
        input.section.validate()?;
        let mut next_id = self.next_id;
        let mut next_context = self.next_context;
        for p in previous
            .paths
            .iter()
            .flatten()
            .flat_map(|c| c.groups.iter().flatten().flatten())
        {
            if [p.components[2], p.components[3]]
                .into_iter()
                .flatten()
                .any(|id| id >= next_id)
            {
                return Err("local producer does not own retained component identity");
            }
            if let Some(i) = p.section_slot {
                next_context = next_context.max(
                    previous.sections[i]
                        .context
                        .checked_add(1)
                        .ok_or("local context identity exhausted")?,
                );
            }
        }
        for (i, g) in input.groups.iter().enumerate() {
            let o = g.observation;
            if o.interval != input.interval
                || o.rate != input.rate
                || (!input.observed && o.observed)
                || !g.acoustics.matches(o)
                || g.grouping.group != o.group
                || g.correspondence.group != o.group
                || g.returns.group != o.group
                || input.groups[..i]
                    .iter()
                    .any(|g| g.observation.group == o.group)
                || g.phrase_context.iter().flatten().any(|v| !v.is_finite())
            {
                return Err("invalid local producer group sources");
            }
            for proof in g.returns.entries() {
                next_context = next_context.max(
                    proof
                        .context()
                        .context_id
                        .checked_add(1)
                        .ok_or("local context identity exhausted")?,
                );
            }
        }
        self.rows.clear();
        self.sources.clear();
        self.shared.fill(None);
        self.ranges = [[[0; 2]; 8]; 32];
        self.work = Work::default();
        let dt = (end - start) as f64 / f64::from(input.rate);
        for (c, proposal) in shared.entries().enumerate() {
            self.shared[c] = Some(*proposal);
            let score = shared_score(proposal)?;
            if !score.is_finite() {
                return Err("nonfinite generated shared potential");
            }
            self.shared_scores[c] = if input.observed { score } else { 0. };
            let parent = proposal
                .parent
                .map(|id| {
                    previous
                        .paths
                        .iter()
                        .flatten()
                        .find(|p| p.id == id)
                        .ok_or("stale local producer shared parent")
                })
                .transpose()?;
            for (g, source) in input.groups.iter().enumerate() {
                let o = source.observation;
                let old_group = previous.groups.iter().position(|g| *g == Some(o.group));
                let paths = parent.zip(old_group).map(|(p, g)| &p.groups[g]);
                let raw = o.raw.as_ref().filter(|_| o.observed).map(|u| &u.raw);
                let articulation_input = gesture::articulation::Input::new(
                    o.group,
                    raw,
                    source.motion,
                    input.interval,
                    input.rate,
                    input.gesture,
                )?;
                let mut context = source.phrase_context;
                if o.observed {
                    let short = source.acoustics.short(start, input.gesture.rms_reference)?;
                    for (v, x) in context[4..8].iter_mut().zip(short.values) {
                        *v = x.value();
                    }
                }
                let begin = self.rows.len();
                for p in 0..16 {
                    let old = paths.and_then(|p0| p0.get(p)).and_then(Option::as_ref);
                    if p < 15 && old.is_none() {
                        continue;
                    }
                    self.work.parents += 1;
                    let after = next_id
                        .checked_add(8)
                        .ok_or("local component identity exhausted")?;
                    let after_context = next_context
                        .checked_add(2)
                        .ok_or("local context identity exhausted")?;
                    let (articulation, children) = proposals::List::observed_articulation(
                        old.and_then(|p| p.articulation),
                        &articulation_input,
                        input.gesture,
                    )?;
                    let old_grouping = old
                        .and_then(|p| p.grouping_slot)
                        .and_then(|i| previous.groupings[i]);
                    let grouping = source.grouping.proposals(old_grouping, o.observed, dt)?;
                    let old_correspondence = old
                        .and_then(|p| p.correspondence_slot)
                        .and_then(|i| previous.correspondences[i]);
                    let correspondence = source.correspondence.proposals(
                        old_correspondence,
                        o.observed,
                        dt,
                        input.retained,
                    )?;
                    let mut phrase_values = context;
                    phrase_values[9] = old_correspondence
                        .filter(|s| {
                            old.is_some_and(|p| p.correspondence_supported)
                                && s.support.received <= start
                                && start <= s.support.deadline
                                && s.support.audio_end <= start
                                && u128::from(start - s.support.audio_end) * 2
                                    < u128::from(input.rate)
                        })
                        .and_then(|s| s.matched)
                        .map(|m| m.cost.ln_1p());
                    let old_phrase = old
                        .and_then(|p| p.phrase_slot)
                        .and_then(|i| previous.phrases[i].as_ref());
                    let articulation_state = old.and_then(|p| p.articulation).map(|a| a.state);
                    let phrase_ids = [
                        old_phrase.map_or(next_id, |p| p.id),
                        next_id + 1,
                        next_id + 2,
                        next_id + 3,
                        next_id + 4,
                    ];
                    let phrase_children: [proposals::PhraseExtension; 5] =
                        std::array::from_fn(|i| proposals::PhraseExtension {
                            parent_slot: old.and_then(|p| p.phrase_slot),
                            id: phrase_ids[i],
                            kind: [
                                None,
                                Some(phrase::Exit::New),
                                Some(phrase::Exit::Overlap),
                                Some(phrase::Exit::Reinterpret),
                                Some(phrase::Exit::Inactive),
                            ][i],
                        });
                    let phrase_list = if let Some(prior) =
                        old_phrase.filter(|p| p.foreground.is_some())
                    {
                        let span = prior.foreground.unwrap();
                        if o.observed {
                            proposals::List::active_phrase(
                                phrase_ids,
                                phrase_values,
                                articulation_state,
                                [start - span.start, end - span.start],
                                input.rate,
                                articulation_input.low,
                                input.phrase,
                            )?
                        } else {
                            let elapsed = [start - span.start, end - span.start]
                                .map(|t| t as f64 / f64::from(input.rate));
                            let survival = phrase::conditional_law::<false>(
                                phrase_values,
                                articulation_state,
                                elapsed,
                                input.phrase,
                            )
                            .map_or(0., |l| l.survival);
                            proposals::List::missing_boundary(
                                proposals::Kind::Phrase,
                                prior.id,
                                survival,
                                dt,
                            )?
                        }
                    } else {
                        let admit = articulation_input.observed && !articulation_input.low;
                        let reentry = admit && articulation_state == Some(gesture::State::Attack);
                        let entries = [
                            proposals::Known {
                                id: phrase_ids[0],
                                raw_score: if old_phrase.is_some() {
                                    f64::from(!reentry)
                                } else {
                                    f64::from(admit)
                                },
                                stay: old_phrase.is_some(),
                                boundary: Some(proposals::Boundary::Stay),
                            },
                            proposals::Known {
                                id: phrase_ids[1],
                                raw_score: f64::from(old_phrase.is_some() && reentry),
                                stay: false,
                                boundary: Some(proposals::Boundary::Exit),
                            },
                        ];
                        proposals::List::build(
                            proposals::Kind::Phrase,
                            &entries,
                            old_phrase.is_none(),
                            dt,
                            6,
                        )?
                    };
                    let section_ids = [
                        old.and_then(|p| p.components[3]).unwrap_or(next_id + 5),
                        next_id + 5,
                        next_id + 6,
                        next_id + 7,
                    ];
                    // The unused stay slot of an unknown section must still have a distinct identity.
                    let section_ids = if old.and_then(|p| p.section_slot).is_none() {
                        [next_id, next_id + 5, next_id + 6, next_id + 7]
                    } else {
                        section_ids
                    };
                    let section_output = section::producer::build(section::producer::Inputs {
                        snapshot: previous,
                        parent: old,
                        observation: o,
                        correspondences: &correspondence,
                        cache: source.correspondence,
                        retained_episodes: input.retained,
                        returns: source.returns,
                        head: input.section,
                        ids: section_ids,
                        new_contexts: [next_context, next_context + 1],
                        rms_reference: input.gesture.rms_reference,
                    })?;
                    let mut section_slots = [None; 4];
                    for (i, s) in section_output.sources.iter().enumerate() {
                        if let Some((id, s)) = s {
                            section_slots[i] = Some((*id, self.sources.len()));
                            self.sources.push(*s);
                        }
                    }
                    let lists = [
                        articulation,
                        grouping.list,
                        phrase_list,
                        section_output.list,
                        correspondence.list,
                    ];
                    let tuples = self
                        .composer
                        .compose(&lists, &section_output.compatibility(&correspondence))?;
                    self.work.priorities += tuples.priority_evaluations;
                    self.work.heap_pops += tuples.heap_pops;
                    self.work.compatibility_checks +=
                        tuples.compatibility_checks + tuples.correspondence_checks;
                    self.work.component_truncated +=
                        lists.iter().map(|l| l.truncated).sum::<usize>();
                    for (extension, t) in tuples.tuples.iter().flatten().enumerate() {
                        let group_choice = grouping
                            .states
                            .iter()
                            .enumerate()
                            .find(|(_, s)| s.is_some_and(|s| Some(s.id) == t.ids[1]));
                        let corr_choice = correspondence
                            .states
                            .iter()
                            .enumerate()
                            .find(|(_, s)| s.is_some_and(|s| Some(s.id) == t.ids[4]));
                        let mut row = owner::LocalExtension {
                            parent: old.map(|p| p.id),
                            extension: extension as u8,
                            components: t.ids,
                            articulation: t.ids[0].and_then(|id| children[id as usize]),
                            phrase: phrase_children
                                .iter()
                                .find(|p| Some(p.id) == t.ids[2])
                                .copied(),
                            grouping: group_choice.map(|(i, s)| grouping::Extension {
                                id: s.unwrap().id,
                                origin: grouping.origins[i].unwrap(),
                            }),
                            correspondence: corr_choice.map(|(i, s)| correspondence::Extension {
                                id: s.unwrap().id,
                                origin: correspondence.origins[i].unwrap(),
                            }),
                            section: section_slots
                                .iter()
                                .flatten()
                                .find(|(id, _)| Some(*id) == t.ids[3])
                                .map(|(id, slot)| section::Extension {
                                    id: *id,
                                    parent_slot: old.and_then(|p| p.section_slot),
                                    source_slot: *slot,
                                }),
                            log_transition: t.log_transition,
                            log_potential: 0.,
                        };
                        let potential = local_score(proposal, g, &row)?;
                        if !potential.is_finite() {
                            return Err("nonfinite generated local potential");
                        }
                        row.log_potential = if o.observed { potential } else { 0. };
                        self.rows.push(row);
                    }
                    next_id = after;
                    next_context = after_context;
                }
                self.ranges[c][g] = [begin, self.rows.len()];
            }
        }
        self.work.tuples = self.rows.len();
        self.next_id = next_id;
        self.next_context = next_context;
        self.cut = end;
        Ok(Batch {
            producer: self,
            shared,
            input,
        })
    }
}

impl Batch<'_, '_> {
    pub fn advance<'o>(
        &self,
        owner: &'o mut owner::State,
    ) -> Result<&'o owner::Snapshot, &'static str> {
        let Self {
            producer,
            shared,
            input,
        } = *self;
        let count = producer.shared.iter().flatten().count();
        let refs: [[&[owner::LocalExtension]; 8]; 32] = std::array::from_fn(|c| {
            std::array::from_fn(|g| {
                let [a, b] = producer.ranges[c][g];
                &producer.rows[a..b]
            })
        });
        let contexts: [owner::ContextExtension<'_>; 32] = std::array::from_fn(|c| {
            let p = producer.shared[c];
            owner::ContextExtension {
                parent: p.and_then(|p| p.parent),
                extension: p.map_or(0, |p| p.slot),
                key: p.and_then(|p| p.key),
                log_transition: p.map_or(0., |p| p.log_transition),
                log_potential: producer.shared_scores[c],
                groups: &refs[c][..input.groups.len()],
            }
        });
        let groups: [Handle; 8] = std::array::from_fn(|g| {
            input
                .groups
                .get(g)
                .map(|g| g.observation.group)
                .unwrap_or(Handle {
                    bus: 0,
                    epoch: 0,
                    generation: 0,
                })
        });
        let observations: [Option<&Observation>; 8] =
            std::array::from_fn(|g| input.groups.get(g).map(|g| g.observation));
        let acoustics: [Option<&acoustics::Cache>; 8] =
            std::array::from_fn(|g| input.groups.get(g).map(|g| g.acoustics));
        let groupings: [Option<&grouping::Cache>; 8] =
            std::array::from_fn(|g| input.groups.get(g).map(|g| g.grouping));
        let correspondences: [Option<&correspondence::Cache>; 8] =
            std::array::from_fn(|g| input.groups.get(g).map(|g| g.correspondence));
        owner.advance(
            input.interval,
            input.observed,
            &groups[..input.groups.len()],
            &contexts[..count],
            owner::Sources {
                shared: Some(shared),
                observations: &observations[..input.groups.len()],
                acoustics: &acoustics[..input.groups.len()],
                groupings: &groupings[..input.groups.len()],
                correspondences: &correspondences[..input.groups.len()],
                sections: &producer.sources,
                retained_episodes: input.retained,
                section_config: Some(input.section_config),
            },
        )
    }
}

#[cfg(test)]
pub(super) mod tests;
