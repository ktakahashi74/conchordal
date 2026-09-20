//! Path-local section input consumption; section-boundary branching remains separate.

use super::{Head, retrieval_scores};
use crate::config::TemporalSectionConfig;
use crate::temporal_cognition::{
    accents, phrase, proposals::frontend, recall, ridge::Handle, transport::Identity,
};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct GroupSnapshot {
    pub group: Handle,
    pub previous_end_sample: u64,
    pub parents: [Option<super::form::Parent>; 16],
    pub representative_path: usize,
    pub candidate_start_sample: u64,
    pub observed_end_sample: u64,
    pub cue: Option<super::cue::Selection>,
    pub observed_spans: u64,
    pub candidates: [Option<super::form::Candidate>; 16],
    pub pruned_candidate_mass: f64,
    pub late_accent_support: f64,
    #[serde(with = "crate::config::fixed_array")]
    pub cumulative: [Option<f64>; 39],
    #[serde(with = "crate::config::fixed_array")]
    pub recent: [Option<f64>; 39],
    #[serde(with = "crate::config::fixed_array")]
    pub covariates: [Option<f64>; 82],
    pub survival: f64,
    pub exits: [f64; 3],
    pub unknown: f64,
    pub matched_prefixes: usize,
    pub prefix_support_queries: u64,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct Snapshot {
    pub end_sample: u64,
    pub groups: [Option<GroupSnapshot>; 7],
    pub initial_context_only: bool,
    pub censored: bool,
    pub committed_phrases: u64,
    pub commitment_losses: u64,
    pub commitment_revisions: u64,
}

struct Path {
    activity: super::input::Span,
    forms: Vec<super::form::Path>,
}

struct Group {
    handle: Handle,
    start: u64,
    end: u64,
    paths: Vec<Path>,
    scratch: Vec<Path>,
    count: usize,
    ledger: accents::Ledger,
    prefix_support_queries: u64,
    last_query: u64,
}

pub(crate) struct Stream {
    config: TemporalSectionConfig,
    head: Head,
    bus: u8,
    epoch: u64,
    rate: u32,
    end: u64,
    groups: [Option<Group>; 7],
    snapshot: Snapshot,
    cues: super::cue::Stream,
    next_context: u64,
    next_path: u64,
    score_scratch: Vec<(Identity, Option<f64>)>,
    match_scratch: Vec<(recall::MatchSnapshot, f64)>,
    identity_scratch: Vec<u64>,
}

impl Stream {
    pub fn validate(config: TemporalSectionConfig) -> Result<(), &'static str> {
        Head {
            means: config.means,
            deviations: config.deviations,
            hazard: config.hazard,
            exits: [config.new_context, config.recurrence, config.contrast],
        }
        .validate()?;
        if config
            .ending_means
            .iter()
            .chain(&config.ending_deviations)
            .chain(&config.match_means)
            .chain(&config.match_deviations)
            .chain(&config.match_coefficients)
            .any(|v| !v.is_finite())
            || config
                .ending_deviations
                .iter()
                .chain(&config.match_deviations)
                .any(|v| *v < 0.)
        {
            return Err("invalid frozen section ending or acoustic-match scales");
        }
        Ok(())
    }

    pub fn new(
        bus: u8,
        epoch: u64,
        rate: u32,
        start: u64,
        hop: u64,
        memory: crate::config::TemporalMemoryConfig,
        config: TemporalSectionConfig,
    ) -> Result<Box<Self>, &'static str> {
        Self::validate(config)?;
        if bus > 1 || rate == 0 {
            return Err("invalid section stream clock");
        }
        Ok(Box::new(Self {
            config,
            next_context: 0,
            next_path: 0,
            score_scratch: Vec::with_capacity(
                4 * memory
                    .candidates
                    .unwrap_or(super::super::memory::CANDIDATES),
            ),
            match_scratch: Vec::with_capacity(
                4 * memory
                    .candidates
                    .unwrap_or(super::super::memory::CANDIDATES),
            ),
            identity_scratch: Vec::with_capacity(
                memory
                    .candidates
                    .unwrap_or(super::super::memory::CANDIDATES),
            ),
            head: Head {
                means: config.means,
                deviations: config.deviations,
                hazard: config.hazard,
                exits: [config.new_context, config.recurrence, config.contrast],
            },
            bus,
            epoch,
            rate,
            end: start,
            groups: std::array::from_fn(|_| None),
            cues: super::cue::Stream::new(bus, epoch, rate, hop, start, memory.scales)?,
            snapshot: Snapshot {
                end_sample: start,
                initial_context_only: false,
                ..Default::default()
            },
        }))
    }

    pub fn cues(&self) -> &super::cue::Stream {
        &self.cues
    }

    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    pub fn advance(
        &mut self,
        acoustic: &frontend::Snapshot,
        phrase: &phrase::Snapshot,
        period: Option<frontend::recurrence::Snapshot>,
        memory: Option<&recall::Recall>,
    ) -> Result<(), &'static str> {
        let end = phrase.end_sample;
        if end <= self.end {
            return Err("section requires a new ordered phrase receipt");
        }
        if phrase.sample_rate != self.rate
            || acoustic.assignment.end_sample != end
            || phrase.censored
        {
            return Err("section receipt clock or acquisition differs from phrase");
        }
        // Parent indices belong to one exact preceding beam, not merely an array slot.
        // Validate every group before cue consumption, retirement or context allocation.
        for (index, incoming) in phrase
            .groups
            .iter()
            .enumerate()
            .filter_map(|(i, g)| g.map(|g| (i, g)))
        {
            if incoming.group.bus != self.bus
                || incoming.group.epoch != self.epoch
                || phrase.groups[..index]
                    .iter()
                    .flatten()
                    .any(|g| g.group == incoming.group)
            {
                return Err("foreign or duplicate section receipt group");
            }
            let previous = self
                .groups
                .iter()
                .flatten()
                .find(|g| g.handle == incoming.group);
            if incoming.previous_end_sample >= end
                || previous.map_or(incoming.previous_end_sample < self.end, |g| {
                    incoming.previous_end_sample != g.end
                })
            {
                return Err("section parent beam clock differs from phrase receipt");
            }
            let parents = previous.map_or(1, |g| g.count);
            for (count, (child_index, child)) in incoming
                .candidates
                .iter()
                .enumerate()
                .filter_map(|(i, c)| c.map(|c| (i, c)))
                .enumerate()
            {
                if child_index != count || child.parent_index >= parents {
                    return Err("noncontiguous section children or invalid parent");
                }
            }
        }
        self.cues.prepare(acoustic, phrase)?;
        let rate = f64::from(self.rate);
        self.snapshot.groups.fill(None);
        for slot in &mut self.groups {
            if slot
                .as_ref()
                .is_some_and(|g| !phrase.groups.iter().flatten().any(|p| p.group == g.handle))
            {
                *slot = None;
            }
        }
        for incoming in phrase.groups.iter().flatten() {
            if incoming.group.bus != self.bus || incoming.group.epoch != self.epoch {
                return Err("foreign section group");
            }
            let index = if let Some(i) = self
                .groups
                .iter()
                .position(|g| g.as_ref().is_some_and(|g| g.handle == incoming.group))
            {
                i
            } else {
                let i = self
                    .groups
                    .iter()
                    .position(Option::is_none)
                    .ok_or("section group capacity")?;
                self.next_context = self
                    .next_context
                    .checked_add(1)
                    .ok_or("section context identity exhausted")?;
                let initial_context = self.next_context;
                self.next_path = self
                    .next_path
                    .checked_add(1)
                    .ok_or("section path identity exhausted")?;
                let initial_path = self.next_path;
                let make_paths = || {
                    (0..15)
                        .map(|_| {
                            Ok(Path {
                                activity: super::input::Span::default(),
                                forms: vec![super::form::Path::initial(
                                    incoming.group,
                                    incoming.previous_end_sample,
                                    self.rate,
                                    initial_context,
                                    initial_path,
                                )?],
                            })
                        })
                        .collect::<Result<Vec<_>, &'static str>>()
                };
                self.groups[i] = Some(Group {
                    handle: incoming.group,
                    start: incoming.previous_end_sample,
                    end: incoming.previous_end_sample,
                    paths: make_paths()?,
                    scratch: make_paths()?,
                    count: 1,
                    ledger: accents::Ledger::new(incoming.group, 128, 2 * u64::from(self.rate))?,
                    prefix_support_queries: 0,
                    last_query: 0,
                });
                i
            };
            let g = self.groups[index].as_mut().unwrap();
            if incoming.previous_end_sample != g.end {
                return Err("section receipt skipped its parent generation or endpoint");
            }
            let mut parents = [None; 16];
            for (slot, (phrase_path_index, form)) in g.paths[..g.count]
                .iter()
                .enumerate()
                .flat_map(|(i, p)| p.forms.iter().map(move |f| (i, f)))
                .enumerate()
            {
                let target = parents
                    .get_mut(slot)
                    .ok_or("section parent path capacity")?;
                *target = Some(super::form::Parent {
                    path_id: form.id,
                    phrase_path_index,
                });
            }
            let grouped = period
                .and_then(|p| {
                    p.groups
                        .into_iter()
                        .flatten()
                        .find(|p| p.ledger.group == g.handle && p.active)
                })
                .and_then(|p| p.grouping)
                .and_then(|p| p.admission_support)
                .map(|support| support > 0.);
            let mut observation = super::input::Observation::new(
                acoustic,
                g.handle,
                [g.end, end],
                self.rate,
                grouped,
            )?;
            let raw = observation.raw;
            let observed = raw.filter(|_| observation.observed);
            let delta = observation.delta;
            let delivery = raw
                .and_then(|u| u.detector)
                .and_then(|d| d.accent)
                .map(|a| g.ledger.deliver(a, end))
                .transpose()?
                .flatten();
            observation.delivery = delivery;
            let deliveries = delivery.as_slice();
            let query = memory
                .and_then(|m| m.matches_for(g.handle))
                .filter(|(q, _)| q.group == g.handle && q.received_at <= end && q.deadline >= end);
            let mut count = 0;
            let mut representative = 0;
            let mut representative_mass = -1.;
            let mut survival = 0.;
            let mut exits = [0.; 3];
            let mut supported = 0.;
            let mut matched_prefixes = 0;
            let mut representative_values = [None; 82];
            let mut used_query = false;
            self.score_scratch.clear();
            self.match_scratch.clear();
            let scores = &mut self.score_scratch;
            let scored_matches = &mut self.match_scratch;
            let mut retrieval = [None; 2];
            if let Some((q, matches)) = query.filter(|(q, _)| {
                q.cue.is_some_and(|s| {
                    s.occurrence_id > 0
                        && s.start_sample == q.support_start_sample
                        && s.support_end_sample == q.support_end_sample
                        && s.selected_at <= q.issued_at
                        && s.weighted_seconds > 0.
                })
            }) {
                used_query = true;
                for m in matches.iter().flatten().flatten().filter(|m| !m.ambiguous) {
                    let Some(score) = m.acoustic_score(&self.config) else {
                        continue;
                    };
                    scores.push((
                        Identity {
                            id: m.episode_id,
                            generation: m.episode_generation,
                        },
                        Some(score),
                    ));
                    scored_matches.push((*m, score));
                }
                retrieval = retrieval_scores(
                    scores.iter().copied(),
                    q.supporting_audio_end,
                    q.support_end_sample,
                    end,
                    self.rate,
                )?
                .values;
                matched_prefixes += usize::from(retrieval[0].is_some());
            }
            let mut returns = Vec::with_capacity(256);
            if retrieval[0].is_some()
                && let (Some(memory), Some((_, _))) = (memory, query.filter(|(q, _)| !q.cutoff_tie))
            {
                let eligible = scored_matches;
                eligible.retain(|(m, _)| {
                    !m.ambiguous
                        && m.cost <= 1.
                        && m.transformation
                            .iter()
                            .all(|v| v.is_some_and(|v| v.abs() < 2.))
                });
                eligible.sort_by(|(a, x), (b, y)| {
                    y.total_cmp(x)
                        .then(a.episode_id.cmp(&b.episode_id))
                        .then(a.cost.total_cmp(&b.cost))
                });
                self.identity_scratch.clear();
                let seen = &mut self.identity_scratch;
                eligible.retain(|(m, _)| {
                    if seen.contains(&m.episode_id) {
                        false
                    } else {
                        seen.push(m.episode_id);
                        true
                    }
                });
                let max = eligible.first().map_or(0., |(_, s)| *s);
                let total: f64 = eligible.iter().map(|(_, s)| (s - max).exp()).sum();
                for &(m, score) in eligible.iter() {
                    if let Some(origin) = memory.occurrence(m.episode_id, m.episode_generation) {
                        for mut context in origin.contexts.into_iter().flatten() {
                            context.mass *= (score - max).exp() / total;
                            returns.push((m, context));
                        }
                    }
                }
            }
            let mut form_candidates = Vec::with_capacity(240);
            let mut projections = Vec::with_capacity(240 * 16);
            let mut representative_form = 0;
            for (child_index, child) in incoming
                .candidates
                .iter()
                .enumerate()
                .filter_map(|(i, c)| c.map(|c| (i, c)))
            {
                if child_index != count || child.parent_index >= g.count {
                    return Err("noncontiguous section children or invalid parent");
                }
                count += 1;
                let parent = &g.paths[child.parent_index];
                let target = &mut g.scratch[child_index];
                target.activity = parent.activity;
                let completion = target.activity.advance(
                    &observation,
                    super::input::Phrase {
                        foreground: child.foreground,
                        completed_foreground: child.completed_foreground,
                        completed_ending: child.completed_ending,
                    },
                    g.end == g.start,
                    &self.config,
                )?;
                super::form::advance(
                    &parent.forms,
                    &mut target.forms,
                    super::form::Step {
                        group: g.handle,
                        previous_end: g.end,
                        end,
                        rate: self.rate,
                        observed: observed.is_some(),
                        delta,
                        deliveries,
                        completed: completion,
                        retrieval,
                        query: query.map(|(q, _)| q),
                        returns: &returns,
                    },
                    &self.head,
                    &mut self.next_context,
                    &mut self.next_path,
                )?;
            }
            let mut order: Vec<_> = incoming
                .candidates
                .iter()
                .enumerate()
                .filter_map(|(i, c)| c.map(|c| (i, c)))
                .flat_map(|(i, c)| {
                    g.scratch[i]
                        .forms
                        .iter()
                        .enumerate()
                        .map(move |(j, f)| (i, j, c.mass * f.weight))
                })
                .collect();
            order.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
            for i in 0..count {
                let mut j = 0;
                g.scratch[i].forms.retain(|_| {
                    let keep = order.iter().take(16).any(|&(pi, pj, _)| pi == i && pj == j);
                    j += 1;
                    keep
                });
            }
            for (child_index, child) in incoming
                .candidates
                .iter()
                .enumerate()
                .filter_map(|(i, c)| c.map(|c| (i, c)))
            {
                let target = &g.scratch[child_index];
                for (form_index, form) in target.forms.iter().enumerate() {
                    let mass = child.mass * form.weight;
                    let elapsed = (end - form.state.start) as f64 / rate;
                    if let Some(law) = self.head.law(&form.state.values, elapsed, elapsed + 1.)? {
                        survival += mass * law[0];
                        for j in 0..3 {
                            exits[j] += mass * law[j + 1];
                        }
                        supported += mass;
                    }
                    form_candidates.push(super::form::Candidate {
                        path_id: form.id,
                        parent_path_id: form.parent,
                        phrase_path_index: child_index,
                        transition: form.transition,
                        transition_score: form.transition_score,
                        context_id: form.state.context,
                        start_sample: form.state.start,
                        mass,
                        relation: form.state.relation,
                        focus: form.state.focus,
                        query_id: form.state.query.map(|q| q.query_id),
                    });
                    for &(credit, endpoint, context) in &form.state.owners {
                        projections.push((credit, endpoint, context, mass));
                    }
                    if mass > representative_mass {
                        representative_mass = mass;
                        representative = child_index;
                        representative_form = form_index;
                        representative_values = form.state.values;
                    }
                }
            }
            self.cues.commitments.project(g.handle, end, &projections)?;
            form_candidates.sort_by(|a, b| {
                b.mass
                    .total_cmp(&a.mass)
                    .then(a.context_id.cmp(&b.context_id))
            });
            if used_query
                && let Some((q, _)) = query
                && q.query_id > g.last_query
            {
                g.last_query = q.query_id;
                g.prefix_support_queries += 1;
            }
            std::mem::swap(&mut g.paths, &mut g.scratch);
            g.count = count;
            g.end = end;
            if representative_mass >= 0. {
                let path = &g.paths[representative];
                let form = &path.forms[representative_form];
                self.snapshot.groups[index] = Some(GroupSnapshot {
                    group: g.handle,
                    previous_end_sample: incoming.previous_end_sample,
                    parents,
                    representative_path: representative,
                    candidate_start_sample: form.state.start,
                    observed_end_sample: end,
                    cue: self.cues.selection(g.handle),
                    observed_spans: path.activity.sequence,
                    candidates: std::array::from_fn(|i| form_candidates.get(i).copied()),
                    pruned_candidate_mass: order.iter().skip(16).map(|c| c.2).sum(),
                    late_accent_support: form.state.late_accent_support,
                    cumulative: form.state.history.cumulative.values(),
                    recent: form.state.history.recent().values(),
                    covariates: representative_values,
                    survival,
                    exits,
                    unknown: (1. - supported).clamp(0., 1.),
                    matched_prefixes,
                    prefix_support_queries: g.prefix_support_queries,
                });
            }
        }
        self.cues.commitments.seal(end, &phrase.groups);
        self.snapshot.committed_phrases = self.cues.commitments.total;
        self.snapshot.commitment_losses = self.cues.commitments.lost;
        self.snapshot.commitment_revisions = self.cues.commitments.revisions;
        self.end = end;
        self.snapshot.end_sample = end;
        self.snapshot.censored = phrase.censored;
        Ok(())
    }

    pub fn finish(&mut self, end: u64) -> Result<(), &'static str> {
        if end < self.end {
            return Err("backward section EOF");
        }
        self.end = end;
        self.snapshot.end_sample = end;
        self.snapshot.censored = true;
        for g in self.snapshot.groups.iter_mut().flatten() {
            g.survival = 0.;
            g.exits = [0.; 3];
            g.unknown = 1.;
        }
        Ok(())
    }
}
