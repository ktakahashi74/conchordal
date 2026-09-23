//! Received memory evidence and bounded fixed-span assay queries.

use super::{descriptor, memory, proposals::frontend, query, ridge::Handle, transport};
use crate::config::TemporalMemoryConfig;

const MODEL_VERSION: u64 = 5;
const KNOTS: usize = 128;
const COARSE_SNAPSHOTS: usize = 8;

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(crate) struct Snapshot {
    pub retention: Option<retention::Snapshot>,
    pub retrieval: [Option<retrieval::Group>; 7],
    pub coarse_snapshots_per_group: usize,
    pub coarse_cache_owned_bytes: usize,
    pub acquisition: Option<clock::Snapshot>,
    pub stored_episodes: usize,
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

impl Snapshot {}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct ResultSnapshot {
    pub query_id: u64,
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

impl Prediction {}

struct Group {
    handle: Handle,
    span: Option<descriptor::Span>,
    latest: Option<ResultSnapshot>,
    matches: Box<[[Option<MatchSnapshot>; 4]]>,
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
    episode_id: u64,
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
            episode_id: 0,
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
            let acquisition = self.acquisition.insert(clock::Clock::new(
                self.epoch, self.rate, self.hop, start, 128,
            )?);
            if let Some(p) = self.config.retention {
                self.retention = Some(retention::Retention::new(
                    acquisition,
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
    pub(in crate::temporal_cognition) fn matches_for(
        &self,
        group: Handle,
    ) -> Option<(ResultSnapshot, &[[Option<MatchSnapshot>; 4]])> {
        let g = self.groups.iter().flatten().find(|g| g.handle == group)?;
        g.latest.map(|q| (q, g.matches.as_ref()))
    }

    #[cfg(test)]
    pub(in crate::temporal_cognition) fn latest_matches(
        &self,
    ) -> Option<(ResultSnapshot, &[[Option<MatchSnapshot>; 4]])> {
        self.snapshot()
            .latest
            .and_then(|q| self.matches_for(q.group))
    }

    fn retire_unheld_groups(
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
    ) -> Result<(), &'static str> {
        self.retire_unheld_groups(acoustic, cut)?;
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
                let frozen = span.prefix(cut)?;
                self.scheduler.submit(
                    slot,
                    &frozen,
                    [self.query_id, self.query_id, self.query_id],
                    cut,
                )?;
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
            let snapshot = ResultSnapshot {
                search_covered: report.search_covered,
                cutoff_tie: report.cutoff_tie,
                pruned_candidates: report.pruned_candidates,
                pruned_ties: report.pruned_ties,
                query_id: ticket.query.id,
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
            self.store(&frozen, cut)?;
        }
        self.snapshot.stored_episodes = self.episodes.len();
        self.refresh_retrieval(cut)?;
        Ok(())
    }

    fn store(&mut self, frozen: &descriptor::Frozen, cut: u64) -> Result<(), &'static str> {
        if frozen.supporting_audio_end.is_none() {
            return Ok(());
        }
        let descriptor = frozen
            .matching()
            .map_err(|_| "memory descriptor export failed")?;
        let support = (descriptor.knots.iter().map(|k| k.observed_sec).sum::<f64>()
            / (frozen.end - frozen.start))
            .clamp(0., 1.);
        if self.config.retention.is_some() && (!support.is_finite() || support <= 0.) {
            return Err("retained episode admission requires positive observed assignment");
        }
        self.retention_costs.clear();
        if self.retention.is_some() {
            self.retention_costs
                .extend(self.episodes.iter().map(|e| (e.identity.id, None)));
        }
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
            self.controller.retire(old.identity);
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
            let committed = frozen.available_at as f64 / rate;
            let coarse = None;
            retention.apply(
                self.acquisition.as_ref().unwrap(),
                &retention::Write {
                    epoch: self.epoch,
                    sequence: self.snapshot.stored_total + 1,
                    occurrence_id: identity.id,
                    support_id: identity.id,
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
            available_end: frozen.available_at as f64 / f64::from(self.rate),
            first_observed_end: frozen.end,
            scales: self.config.scales,
            descriptor,
        };
        let slot = self
            .episodes
            .partition_point(|e| e.first_observed_end <= episode.first_observed_end);
        self.episodes.insert(slot, episode);
        self.snapshot.stored_total += 1;
        Ok(())
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
mod retention;
pub(super) mod retrieval;
