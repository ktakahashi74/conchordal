//! Received memory evidence and bounded assay or selected-phrase prefix queries.

use super::{descriptor, memory, proposals::frontend, query, ridge::Handle, transport};
use crate::config::TemporalMemoryConfig;

pub(in crate::temporal_cognition) use graph::Snapshot as GraphSnapshot;

const MODEL_VERSION: u64 = 5;
const KNOTS: usize = 128;
const COARSE_SNAPSHOTS: usize = 8;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct SealedCoarse {
    pub occurrence_id: u64,
    pub support_end_sample: u64,
    pub query: Option<query::CoarseEvidence>,
    pub known_costs: usize,
    pub unknown_costs: usize,
    pub approximate_costs: usize,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(crate) struct Snapshot {
    pub retention: Option<retention::Snapshot>,
    pub retrieval: [Option<retrieval::Group>; 7],
    pub coarse_snapshots_per_group: usize,
    pub coarse_cache_owned_bytes: usize,
    pub latest_sealed_coarse: Option<SealedCoarse>,
    pub sealed_with_coarse_query: u64,
    pub sealed_without_coarse_query: u64,
    pub acquisition: Option<clock::Snapshot>,
    pub graph: Option<graph::Snapshot>,
    pub stored_episodes: usize,
    pub phrase_episodes: usize,
    pub committed_support: f64,
    pub stored_total: u64,
    pub evicted: u64,
    pub queries: u64,
    pub completed: u64,
    pub pruned_candidates: u64,
    pub pruned_ties: u64,
    pub rejected_late: u64,
    pub rejected_retired: u64,
    pub latest: Option<ResultSnapshot>,
    pub retained_matches: usize,
    pub group_queries: [Option<ResultSnapshot>; 7],
}

impl Snapshot {
    pub fn latest_for(&self, group: Handle) -> Option<ResultSnapshot> {
        self.group_queries
            .iter()
            .flatten()
            .find(|q| q.group == group)
            .copied()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct ResultSnapshot {
    pub query_id: u64,
    pub cue: Option<super::section::cue::Selection>,
    pub group: Handle,
    pub support_start_sample: u64,
    pub support_end_sample: u64,
    pub source_start_sample: u64,
    pub supporting_audio_end: Option<u64>,
    pub available_at: u64,
    pub issued_at: u64,
    pub completed_at: u64,
    pub received_at: u64,
    pub deadline: u64,
    pub candidates: usize,
    pub search_covered: bool,
    pub cutoff_tie: bool,
    pub pruned_candidates: usize,
    pub pruned_ties: usize,
    pub dp_cells: u64,
    pub reconstruction_error: f64,
    pub best: Option<MatchSnapshot>,
    pub prediction: Option<Prediction>,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct MatchSnapshot {
    pub episode_id: u64,
    pub episode_generation: u64,
    pub support_start_sample: u64,
    pub support_end_sample: u64,
    pub source_start_sample: u64,
    pub available_at: u64,
    pub cost: f64,
    pub transformation: [Option<f64>; 2],
    pub ambiguous: bool,
    pub path_steps: usize,
    pub anchor: Option<usize>,
    pub residuals: Option<memory::Residuals>,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct Prediction {
    pub episode_id: u64,
    pub query_id: u64,
    pub group: Handle,
    pub issued_at: u64,
    pub source_start: u64,
    pub source_end: u64,
    pub available: u64,
    pub expected_start: u64,
    pub expected_end: u64,
    pub values: [Option<f64>; 10],
    pub scales: [f64; 10],
}

impl Prediction {
    pub(super) fn normalized_residual(&self, values: [Option<f64>; 10]) -> Option<f64> {
        let mut sum = 0.;
        let mut count = 0;
        for (i, actual) in values.into_iter().enumerate() {
            if let (Some(actual), Some(expected)) = (actual, self.values[i]) {
                if !self.scales[i].is_finite() || self.scales[i] <= 0. {
                    return None;
                }
                sum += ((actual - expected) / self.scales[i]).powi(2);
                count += 1;
            }
        }
        (count > 0)
            .then(|| sum / count as f64)
            .filter(|v| v.is_finite())
    }
}

struct Group {
    handle: Handle,
    span: Option<descriptor::Span>,
    latest: Option<ResultSnapshot>,
    matches: Box<[[Option<MatchSnapshot>; 4]]>,
    pending_cue: Option<(u64, super::section::cue::Selection)>,
    start: u64,
    end: u64,
}

struct Pending {
    dispatch: query::Dispatch,
    packet: transport::Packet,
    report: memory::Report,
    snapshot: ResultSnapshot,
}

pub(crate) struct Recall {
    retained_ids: (u64, std::sync::Arc<[(u64, u64)]>),
    acquisition: Option<clock::Clock>,
    retention: Option<retention::Retention>,
    retention_costs: Vec<(u64, Option<f64>)>,
    retrieval_scratch: Vec<retrieval::Entry>,
    recognition_scratch: Vec<(u64, f64)>,
    graph: Option<graph::Graph>,
    config: TemporalMemoryConfig,
    rate: u32,
    hop: u64,
    epoch: u64,
    bus: u8,
    groups: [Option<Group>; 7],
    scheduler: query::Scheduler,
    controller: transport::Controller,
    episodes: Vec<memory::Episode>,
    pending: Option<Pending>,
    pending_matches: Box<[[Option<MatchSnapshot>; 4]]>,
    query_id: u64,
    cue_mode: Option<bool>,
    episode_id: u64,
    last_commit: u64,
    phrase_provenance: Vec<(transport::Identity, super::section::commitment::Evidence)>,
    snapshot: Snapshot,
}

impl Recall {
    pub(crate) fn retained_ids(&mut self) -> std::sync::Arc<[(u64, u64)]> {
        if self.retained_ids.0 != self.snapshot.stored_total {
            self.retained_ids = (
                self.snapshot.stored_total,
                self.episodes
                    .iter()
                    .map(|e| (e.identity.id, e.identity.generation))
                    .collect(),
            );
        }
        std::sync::Arc::clone(&self.retained_ids.1)
    }
    pub(crate) fn new(
        bus: u8,
        epoch: u64,
        rate: u32,
        hop: u64,
        config: TemporalMemoryConfig,
    ) -> Result<Self, &'static str> {
        if let Some(p) = config.retention
            && ([
                p.tau_sec,
                p.kappa,
                p.strength_max,
                p.r_max,
                p.match_temperature,
                p.motion_scale,
                p.interval_scale,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.)
                || p.strength_max < 1.
                || !p.no_memory_bias.is_finite()
                || !p.edit_penalty.is_finite()
                || p.edit_penalty < 0.)
        {
            return Err("invalid explicit memory retention parameters or no-memory bias");
        }
        if config.span_hops == 0
            || config.span_hops.checked_mul(hop).is_none()
            || !(1..=memory::MAX_EPISODES).contains(&config.episodes)
            || !(1..=memory::MAX_CANDIDATES)
                .contains(&config.candidates.unwrap_or(memory::CANDIDATES))
            || config.deadline_ms == 0
            || config.deadline_ms.checked_mul(u64::from(rate)).is_none()
            || config.scales.iter().any(|s| !s.is_finite() || *s < 1e-6)
        {
            return Err("invalid memory diagnostic windows, capacities or scales");
        }
        let scheduler = query::Scheduler::new(query::Config {
            bus,
            epoch,
            sample_rate: rate,
            hop,
            cadence_ms: config.query_cadence_ms,
            groups: 7,
            knots: KNOTS,
            snapshots: COARSE_SNAPSHOTS,
            episodes: config.episodes,
        })?;
        let coarse_cache_owned_bytes = scheduler.coarse_storage_bytes();
        Ok(Self {
            acquisition: None,
            retention: None,
            retention_costs: Vec::with_capacity(if config.retention.is_some() {
                config.episodes
            } else {
                0
            }),
            retrieval_scratch: Vec::with_capacity(if config.retention.is_some() {
                config.candidates.unwrap_or(memory::CANDIDATES)
            } else {
                0
            }),
            recognition_scratch: Vec::with_capacity(if config.retention.is_some() {
                config.candidates.unwrap_or(memory::CANDIDATES)
            } else {
                0
            }),
            graph: None,
            config,
            rate,
            hop,
            epoch,
            bus,
            groups: std::array::from_fn(|_| None),
            scheduler,
            controller: transport::Controller::new(bus, epoch, MODEL_VERSION, config.episodes),
            episodes: Vec::with_capacity(config.episodes),
            pending: None,
            pending_matches: vec![[None; 4]; config.candidates.unwrap_or(memory::CANDIDATES)]
                .into_boxed_slice(),
            query_id: 0,
            cue_mode: None,
            episode_id: 0,
            last_commit: 0,
            phrase_provenance: Vec::with_capacity(config.episodes),
            retained_ids: (0, std::sync::Arc::from([])),
            snapshot: Snapshot {
                coarse_snapshots_per_group: COARSE_SNAPSHOTS,
                coarse_cache_owned_bytes,
                ..Snapshot::default()
            },
        })
    }

    pub(crate) fn snapshot(&self) -> Snapshot {
        Snapshot {
            acquisition: self
                .acquisition
                .as_ref()
                .map(|c| c.snapshot(self.episodes.first().map(|e| e.first_observed_end))),
            graph: self.graph.as_ref().map(|g| g.snapshot()),
            phrase_episodes: self.phrase_provenance.len(),
            committed_support: self.phrase_provenance.iter().map(|(_, p)| p.support).sum(),
            latest: self
                .groups
                .iter()
                .flatten()
                .filter_map(|g| g.latest)
                .max_by_key(|q| q.query_id),
            group_queries: std::array::from_fn(|i| self.groups[i].as_ref().and_then(|g| g.latest)),
            retained_matches: self
                .groups
                .iter()
                .flatten()
                .filter(|g| g.latest.is_some())
                .map(|g| g.matches.iter().flatten().flatten().count())
                .sum(),
            ..self.snapshot
        }
    }

    pub(crate) fn observe_acquisition(
        &mut self,
        start: u64,
        end: u64,
        cut: u64,
    ) -> Result<(), &'static str> {
        if self.acquisition.is_none() {
            self.acquisition = Some(clock::Clock::new(
                self.epoch, self.rate, self.hop, start, 128,
            )?);
            if let Some(p) = self.config.retention {
                self.retention = Some(retention::Retention::new(
                    self.acquisition.as_ref().unwrap(),
                    retention::Parameters {
                        tau: p.tau_sec,
                        kappa: p.kappa,
                        strength_max: p.strength_max,
                        r_max: p.r_max,
                    },
                    self.config.episodes,
                    144,
                )?);
            }
        }
        self.acquisition
            .as_mut()
            .unwrap()
            .observe(clock::Acquisition {
                epoch: self.epoch,
                start,
                end,
                available: cut,
                cut,
                observed: true,
                intervals: &[(start, end)],
            })?;
        Ok(())
    }

    #[cfg(test)]
    pub(in crate::temporal_cognition) fn latest_matches(
        &self,
    ) -> Option<(ResultSnapshot, &[[Option<MatchSnapshot>; 4]])> {
        self.snapshot()
            .latest
            .and_then(|q| self.matches_for(q.group))
    }

    pub(in crate::temporal_cognition) fn matches_for(
        &self,
        group: Handle,
    ) -> Option<(ResultSnapshot, &[[Option<MatchSnapshot>; 4]])> {
        let g = self.groups.iter().flatten().find(|g| g.handle == group)?;
        g.latest.map(|q| (q, g.matches.as_ref()))
    }

    pub(crate) fn receive(
        &mut self,
        acoustic: &frontend::Snapshot,
        cut: u64,
    ) -> Result<(), &'static str> {
        // Retire query ownership before accepting a completion. Saved episodes remain shared history.
        for slot in 0..self.groups.len() {
            if self.groups[slot]
                .as_ref()
                .is_some_and(|g| !acoustic.retained_groups.contains(&Some(g.handle)))
            {
                self.groups[slot] = None;
                self.scheduler.retire(slot, cut)?;
            }
        }
        self.finish(cut)
    }

    pub(crate) fn advance(
        &mut self,
        acoustic: &frontend::Snapshot,
        cut: u64,
        cues: Option<&super::section::cue::Stream>,
    ) -> Result<(), &'static str> {
        if self.cue_mode.is_some_and(|mode| mode != cues.is_some()) {
            return Err("memory query ownership cannot change within an epoch");
        }
        if cues.is_some() && self.config.query_cadence_ms != 100 {
            return Err("phrase cue queries require the registered 100 ms cadence");
        }
        self.cue_mode = Some(cues.is_some());
        self.receive(acoustic, cut)?;
        for update in acoustic.features[..7].iter().flatten() {
            let raw = &update.raw;
            if !acoustic.retained_groups.contains(&Some(raw.group)) {
                continue;
            }
            let slot = if let Some(slot) = self
                .groups
                .iter()
                .position(|g| g.as_ref().is_some_and(|g| g.handle == raw.group))
            {
                slot
            } else {
                let slot = self
                    .groups
                    .iter()
                    .position(Option::is_none)
                    .ok_or("no free memory group slot")?;
                self.scheduler.bind(slot, raw.group.generation, cut)?;
                self.groups[slot] = Some(Group {
                    handle: raw.group,
                    span: None,
                    latest: None,
                    matches: vec![[None; 4]; self.pending_matches.len()].into_boxed_slice(),
                    pending_cue: None,
                    start: raw.start,
                    end: raw.start,
                });
                slot
            };
            let group = self.groups[slot].as_mut().unwrap();
            // A noncanonical missing span carries no acquisition; retain its gap, never fabricate hops.
            if raw.known_samples == 0 {
                if let Some(span) = &mut group.span {
                    span.gap(
                        raw.end as f64 / f64::from(self.rate),
                        raw.available_end,
                        cut,
                    )?;
                    group.end = raw.end;
                }
                continue;
            }
            if raw.known_samples != self.hop || raw.end - raw.start != self.hop {
                return Err(
                    "memory frontend requires canonical full acquisition or an explicit gap",
                );
            }
            let intervals = [(raw.start, raw.end)];
            if group.span.is_none() {
                group.start = raw.start;
                group.span = Some(descriptor::Span::new(descriptor::Config {
                    group: raw.group,
                    sample_rate: self.rate,
                    first_hop_start: raw.start,
                    hop: self.hop,
                    cadence: 2,
                    span_start: raw.start as f64 / f64::from(self.rate),
                    span_end: None,
                    scales: self.config.scales,
                    capacity: KNOTS,
                })?);
            }
            let span = group.span.as_mut().unwrap();
            span.push(raw, &intervals, cut)?;
            group.end = raw.end;
            self.scheduler.observe(slot, raw, &intervals, cut)?;
            // Emit bounded prefixes at the scheduler cadence, without redefining the fixed assay span.
            let period = (self.config.query_cadence_ms * u64::from(self.rate)).div_ceil(1000);
            if (raw.end - group.start) / period > (raw.start.saturating_sub(group.start)) / period
                || raw.end - group.start >= self.config.span_hops * self.hop
            {
                self.query_id = self
                    .query_id
                    .checked_add(1)
                    .ok_or("memory query IDs exhausted")?;
                let (frozen, selected) = if let Some(cues) = cues {
                    let Some((frozen, selected)) = cues.prefix(group.handle, cut)? else {
                        continue;
                    };
                    (frozen, Some(selected))
                } else {
                    (span.prefix(cut)?, None)
                };
                let occurrence = selected.map_or(self.query_id, |s| s.occurrence_id);
                if self.scheduler.submit(
                    slot,
                    &frozen,
                    [self.query_id, occurrence, self.query_id],
                    cut,
                )? {
                    group.pending_cue = selected.map(|s| (self.query_id, s));
                }
            }
        }
        if let Some(dispatch) = self.scheduler.take(cut)? {
            debug_assert_eq!(dispatch.dispatched_at, cut);
            let group = self.groups[dispatch.group_slot].as_ref().unwrap();
            let rate = f64::from(self.rate);
            let start = (dispatch.header.start * rate).round() as u64;
            let end = (dispatch.header.end * rate).round() as u64;
            // Only earlier nonoverlapping assay supports are eligible, including across sound groups.
            let count = self
                .episodes
                .partition_point(|e| e.first_observed_end <= dispatch.header.start);
            let report = memory::ordered_index(
                &memory::Query {
                    descriptor: &dispatch.descriptor,
                    epoch: self.epoch,
                    end: dispatch.header.end,
                    observed_end: cut as f64 / rate,
                    scales: self.config.scales,
                },
                &self.episodes[..count],
                true,
                self.pending_matches.len(),
            )
            .map_err(|_| "memory matching failed")?;

            let ticket = transport::Ticket {
                bus: self.bus,
                epoch: self.epoch,
                model_version: MODEL_VERSION,
                query: transport::Identity {
                    id: dispatch.ticket,
                    generation: group.handle.generation,
                },
                support_id: dispatch.header.support_id,
                support_start: start,
                support_end: end,
                supporting_audio_end: (!dispatch.header.audio_end.is_nan())
                    .then(|| (dispatch.header.audio_end * rate).round() as u64),
                available_at: (dispatch.header.available * rate).round() as u64,
                issued_at: cut,
                deadline: cut
                    .checked_add((self.config.deadline_ms * u64::from(self.rate)).div_ceil(1000))
                    .ok_or("memory deadline overflow")?,
            };
            self.controller
                .dispatch(ticket, cut)
                .map_err(|_| "memory dispatch rejected")?;
            let make_snapshot = |m: &memory::Match| {
                let e = self
                    .episodes
                    .iter()
                    .find(|e| e.identity == m.relation.identity)
                    .unwrap();
                MatchSnapshot {
                    episode_id: e.identity.id,
                    episode_generation: e.identity.generation,
                    support_start_sample: (e.descriptor.knots[0].start * rate).round() as u64,
                    support_end_sample: (e.first_observed_end * rate).round() as u64,
                    source_start_sample: (e
                        .descriptor
                        .knots
                        .iter()
                        .map(|k| k.raw_start)
                        .fold(f64::INFINITY, f64::min)
                        * rate)
                        .round() as u64,
                    available_at: (e.available_end * rate).round() as u64,
                    cost: m.cost.unwrap(),
                    transformation: m.relation.transformation,
                    ambiguous: m.relation.ambiguous,
                    path_steps: m.path.len(),
                    anchor: m.anchor,
                    residuals: m.residuals,
                }
            };
            let best = report
                .matches
                .iter()
                .filter(|m| m.relation.supported && m.cost.is_some())
                .min_by(|a, b| a.cost.unwrap().total_cmp(&b.cost.unwrap()))
                .map(make_snapshot);
            self.pending_matches.fill([None; 4]);
            let matches = self.pending_matches.as_mut();
            for m in report
                .matches
                .iter()
                .filter(|m| m.relation.supported && m.cost.is_some())
            {
                let identity = m.relation.identity;
                let slot = matches
                    .iter()
                    .position(|rows| {
                        rows[0].is_some_and(|old| {
                            old.episode_id == identity.id
                                && old.episode_generation == identity.generation
                        })
                    })
                    .or_else(|| matches.iter().position(|rows| rows[0].is_none()))
                    .ok_or("memory result exceeds configured episode inventory")?;
                let entry = matches[slot]
                    .iter_mut()
                    .find(|entry| entry.is_none())
                    .ok_or("memory result exceeds four transformations per episode")?;
                *entry = Some(make_snapshot(m));
            }
            let prediction = best.filter(|b| !b.ambiguous).and_then(|best| {
                let matched = report.matches.iter().find(|m| {
                    m.relation.identity.id == best.episode_id && m.cost == Some(best.cost)
                })?;
                let (_, tempo) = (
                    matched.relation.transformation[0]?,
                    matched.relation.transformation[1]?,
                );
                let step = matched.path.iter().rev().find(|s| s[0] == 1)?;
                let episode = self
                    .episodes
                    .iter()
                    .find(|e| e.identity.id == best.episode_id)?;
                let q = dispatch.descriptor.knots.get(step[1] as usize)?;
                let r = episode.descriptor.knots.get(step[2] as usize)?;
                let next = episode.descriptor.knots.get(step[2] as usize + 1)?;
                if next.gap != 0 || next.observed_sec < 0.9 * (next.end - next.start) {
                    return None;
                }
                let ratio = 2_f64.powf(tempo);
                let start = ((q.time + (next.start - r.time) / ratio) * rate).round();
                let end = ((q.time + (next.end - r.time) / ratio) * rate).round();
                if !start.is_finite()
                    || !end.is_finite()
                    || start < cut as f64
                    || end <= start
                    || end >= u64::MAX as f64
                {
                    return None;
                }
                Some(Prediction {
                    episode_id: best.episode_id,
                    query_id: ticket.query.id,
                    group: group.handle,
                    issued_at: cut,
                    source_start: best.source_start_sample.min(
                        (dispatch
                            .descriptor
                            .knots
                            .iter()
                            .map(|k| k.raw_start)
                            .fold(f64::INFINITY, f64::min)
                            * rate)
                            .round() as u64,
                    ),
                    source_end: best.support_end_sample.max(ticket.supporting_audio_end?),
                    available: best.available_at.max(ticket.available_at),
                    expected_start: start as u64,
                    expected_end: end as u64,
                    values: std::array::from_fn(|i| {
                        (next.mask & (1 << i) != 0).then_some(
                            next.values[i] + if i == 0 { matched.applied[0] } else { 0. },
                        )
                    }),
                    scales: self.config.scales,
                })
            });
            let cue = group
                .pending_cue
                .filter(|(id, _)| *id == dispatch.header.query_id)
                .map(|(_, s)| s);
            if cues.is_some()
                && cue.is_none_or(|s| {
                    s.occurrence_id != dispatch.header.occurrence_id
                        || s.start_sample != start
                        || s.support_end_sample != end
                        || s.selected_at != (dispatch.header.captured * rate).round() as u64
                        || s.last_observed_sample > end
                        || !s.weighted_seconds.is_finite()
                        || s.weighted_seconds <= 0.
                })
            {
                return Err("dispatched phrase cue differs from its frozen selection");
            }
            let snapshot = ResultSnapshot {
                search_covered: report.search_covered,
                cutoff_tie: report.cutoff_tie,
                pruned_candidates: report.pruned_candidates,
                pruned_ties: report.pruned_ties,
                query_id: ticket.query.id,
                cue,
                group: group.handle,
                support_start_sample: start,
                support_end_sample: end,
                source_start_sample: (dispatch
                    .descriptor
                    .knots
                    .iter()
                    .map(|k| k.raw_start)
                    .fold(f64::INFINITY, f64::min)
                    * rate)
                    .round() as u64,
                supporting_audio_end: ticket.supporting_audio_end,
                available_at: ticket.available_at,
                issued_at: cut,
                completed_at: cut,
                received_at: cut,
                deadline: ticket.deadline,
                candidates: report.coarse.len(),
                dp_cells: report.dp_cells,
                reconstruction_error: dispatch.header.error,
                best,
                prediction,
            };
            let packet = transport::Packet {
                ticket,
                completed_at: cut,
                relations: report.matches.iter().map(|m| Some(m.relation)).collect(),
            };
            self.snapshot.pruned_candidates += report.pruned_candidates as u64;
            self.snapshot.pruned_ties += report.pruned_ties as u64;
            self.pending = Some(Pending {
                dispatch,
                report,
                packet,
                snapshot,
            });
            self.snapshot.queries += 1;
        }
        // Commit after querying, so the current span can never recall itself.
        for slot in 0..self.groups.len() {
            let Some(group) = self.groups[slot].as_mut() else {
                continue;
            };
            if group.span.is_none() || group.end - group.start < self.config.span_hops * self.hop {
                continue;
            }
            let frozen = group.span.take().unwrap().finish(cut)?;
            if cues.is_none() {
                self.store(&frozen, None, cut)?;
            }
        }
        if let Some(cues) = cues {
            for (evidence, frozen) in cues.committed() {
                if evidence.sequence <= self.last_commit {
                    continue;
                }
                self.store(frozen, Some(*evidence), cut)?;
                self.last_commit = evidence.sequence;
            }
            if let Some(graph) = self.graph.as_mut() {
                graph.retain_sources(cut, |group, credit, start| {
                    cues.retains_prefix(group, credit, start)
                })?;
            }
        }
        self.snapshot.stored_episodes = self.episodes.len();
        self.refresh_retrieval(cut)?;
        Ok(())
    }

    fn store(
        &mut self,
        frozen: &descriptor::Frozen,
        provenance: Option<super::section::commitment::Evidence>,
        cut: u64,
    ) -> Result<(), &'static str> {
        if frozen.supporting_audio_end.is_none() {
            return Ok(());
        }
        let descriptor = frozen
            .matching()
            .map_err(|_| "memory descriptor export failed")?;
        let support = provenance.map_or_else(
            || {
                (descriptor.knots.iter().map(|k| k.observed_sec).sum::<f64>()
                    / (frozen.end - frozen.start))
                    .clamp(0., 1.)
            },
            |p| p.support,
        );
        if self.config.retention.is_some() && (!support.is_finite() || support <= 0.) {
            return Err("retained episode admission requires positive observed assignment");
        }
        self.retention_costs.clear();
        let sealed_coarse = if let Some(p) = provenance {
            let cached = self.scheduler.coarse_for_commitment(
                p.group,
                p.ongoing_credit,
                [p.start, p.end],
                p.deadline.min(p.end + u64::from(self.rate) / 2),
                cut,
            )?;
            let mut summary = SealedCoarse {
                occurrence_id: p.occurrence_id,
                support_end_sample: p.end,
                query: cached.map(|c| c.evidence()),
                known_costs: 0,
                unknown_costs: 0,
                approximate_costs: 0,
            };
            for (slot, episode) in self
                .episodes
                .iter()
                .enumerate()
                .filter(|(_, e)| e.first_observed_end < p.end as f64 / f64::from(self.rate))
            {
                let entry = cached.and_then(|c| c.entry(slot, episode.identity.id));
                if let Some(entry) = &entry {
                    debug_assert_eq!(entry.handle, episode.identity.id);
                    debug_assert_eq!(entry.cost.is_some(), entry.similarity.is_some());
                }
                if self.retention.is_some() {
                    self.retention_costs
                        .push((episode.identity.id, entry.as_ref().and_then(|e| e.cost)));
                }
                if entry.as_ref().is_some_and(|e| e.cost.is_some()) {
                    summary.known_costs += 1;
                } else {
                    summary.unknown_costs += 1;
                }
                summary.approximate_costs += usize::from(entry.is_some_and(|e| e.approximate));
            }
            Some(summary)
        } else {
            if self.retention.is_some() {
                self.retention_costs
                    .extend(self.episodes.iter().map(|e| (e.identity.id, None)));
            }
            None
        };
        self.retention_costs.sort_unstable_by_key(|(id, _)| *id);
        if self.episodes.len() == self.config.episodes {
            let victim = if let Some(retention) = self.retention.as_mut() {
                let slot = retention
                    .eviction_candidate(
                        self.acquisition.as_ref().unwrap(),
                        cut as f64 / f64::from(self.rate),
                    )?
                    .ok_or("full descriptor bank lacks retained metadata")?;
                let handle = retention.evict(slot)?.unwrap();
                self.episodes
                    .iter()
                    .position(|e| e.identity.id == handle)
                    .ok_or("retention victim lacks descriptor ownership")?
            } else {
                0
            };
            let old = self.episodes.remove(victim);
            if let Some(graph) = self.graph.as_mut() {
                graph.retire(old.identity);
            }
            self.controller.retire(old.identity);
            self.phrase_provenance.retain(|(id, _)| *id != old.identity);
            for g in self.groups.iter_mut().flatten() {
                for row in g.matches.iter_mut() {
                    if row[0].is_some_and(|m| {
                        m.episode_id == old.identity.id
                            && m.episode_generation == old.identity.generation
                    }) {
                        row.fill(None);
                    }
                }
                if g.latest.is_some_and(|q| {
                    q.best.is_some_and(|m| {
                        m.episode_id == old.identity.id
                            && m.episode_generation == old.identity.generation
                    })
                }) {
                    g.latest = None;
                }
            }
            self.snapshot.evicted += 1;
        }
        self.episode_id = self
            .episode_id
            .checked_add(1)
            .ok_or("memory episode IDs exhausted")?;
        let identity = transport::Identity {
            id: self.episode_id,
            generation: self.episode_id,
        };
        self.controller
            .register(identity)
            .map_err(|_| "memory registration failed")?;
        if let Some(retention) = self.retention.as_mut() {
            let rate = f64::from(self.rate);
            let committed = provenance.map_or(frozen.available_at, |p| p.sealed_at) as f64 / rate;
            let coarse = sealed_coarse
                .and_then(|s| s.query)
                .map(|c| retention::Coarse {
                    epoch: c.epoch,
                    generation: c.generation,
                    query_id: c.query_id,
                    occurrence_id: c.occurrence_id,
                    support_id: c.support_id,
                    support_end: c.end,
                    supporting_audio_end: c.audio_end,
                    available_end: c.received_at,
                    entries: &self.retention_costs,
                });
            retention.apply(
                self.acquisition.as_ref().unwrap(),
                &retention::Write {
                    epoch: self.epoch,
                    sequence: provenance.map_or(self.snapshot.stored_total + 1, |p| p.sequence),
                    occurrence_id: provenance.map_or(identity.id, |p| p.occurrence_id),
                    support_id: provenance.map_or(identity.id, |p| p.ongoing_credit),
                    start: frozen.start,
                    end: frozen.end,
                    committed,
                    delivered: cut as f64 / rate,
                    available: support,
                    unknown: 0.,
                    unassigned: 0.,
                    assignments: &[(identity.id, support)],
                    coarse,
                },
                cut as f64 / rate,
                &[identity.id],
            )?;
        }
        let episode = memory::Episode {
            identity,
            epoch: self.epoch,
            available_end: provenance.map_or(frozen.available_at, |p| p.sealed_at) as f64
                / f64::from(self.rate),
            first_observed_end: frozen.end,
            scales: self.config.scales,
            descriptor,
        };
        let slot = self
            .episodes
            .partition_point(|e| e.first_observed_end <= episode.first_observed_end);
        self.episodes.insert(slot, episode);
        if let Some(provenance) = provenance {
            self.graph
                .get_or_insert_with(|| {
                    graph::Graph::new(
                        self.bus,
                        self.epoch,
                        self.rate,
                        self.config.episodes,
                        self.pending_matches.len(),
                    )
                })
                .register(identity, provenance, cut)?;
            self.phrase_provenance.push((identity, provenance));
        }
        self.snapshot.stored_total += 1;
        if let Some(summary) = sealed_coarse {
            if summary.query.is_some() {
                self.snapshot.sealed_with_coarse_query += 1;
            } else {
                self.snapshot.sealed_without_coarse_query += 1;
            }
            self.snapshot.latest_sealed_coarse = Some(summary);
        }
        Ok(())
    }

    pub(in crate::temporal_cognition) fn occurrence(
        &self,
        episode: u64,
        generation: u64,
    ) -> Option<super::section::commitment::Evidence> {
        self.phrase_provenance
            .iter()
            .find(|(id, _)| id.id == episode && id.generation == generation)
            .map(|(_, e)| *e)
    }

    pub(crate) fn finish(&mut self, cut: u64) -> Result<(), &'static str> {
        for g in self.groups.iter_mut().flatten() {
            if g.latest.is_some_and(|q| q.deadline < cut) {
                g.latest = None;
            }
        }
        let Some(mut pending) = self.pending.take() else {
            return Ok(());
        };
        let current = self.groups[pending.dispatch.group_slot]
            .as_ref()
            .is_some_and(|g| g.handle == pending.snapshot.group);
        let received = if current {
            self.controller.receive(&pending.packet, cut)
        } else {
            Err(transport::Rejection::Retired)
        };
        if let Ok(receipt) = &received {
            debug_assert_eq!(receipt.ticket, pending.packet.ticket);
            debug_assert_eq!(receipt.received_at, cut);
        }
        if received.is_err() {
            self.controller.cancel(pending.packet.ticket);
            if matches!(received, Err(transport::Rejection::Retired)) {
                self.snapshot.rejected_retired += 1;
            } else {
                self.snapshot.rejected_late += 1;
            }
        }
        let bindings: Vec<_> = self
            .episodes
            .iter()
            .enumerate()
            .map(|(slot, e)| query::Binding {
                identity: e.identity,
                slot,
                handle: e.identity.id,
            })
            .collect();
        let completion = received.is_ok().then_some(query::Completion {
            header: pending.dispatch.header,
            completed_at: pending.packet.completed_at,
            complete: true,
            superseded: false,
            coarse: &pending.report.coarse,
        });
        if self
            .scheduler
            .finish(pending.dispatch.ticket, completion, &bindings, cut)?
        {
            pending.snapshot.received_at = cut;
            if let Some(graph) = self.graph.as_mut() {
                graph.receive(pending.snapshot, &self.pending_matches)?;
            }
            let group = self.groups[pending.dispatch.group_slot].as_mut().unwrap();
            std::mem::swap(&mut group.matches, &mut self.pending_matches);
            group.latest = Some(pending.snapshot);
            self.snapshot.completed += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
pub(in crate::temporal_cognition) mod tests;

mod clock;
mod graph;
mod retention;
pub(super) mod retrieval;
