//! Bounded correspondence proposals from already received, immutable matcher results.

use super::proposals::{Kind, Known, List};
use crate::config::TemporalSectionConfig;
use crate::temporal_cognition::{recall, ridge::Handle, section};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::temporal_cognition) struct Support {
    pub query: u64,
    pub occurrence: Option<u64>,
    pub start: u64,
    pub source_start: u64,
    pub end: u64,
    pub audio_end: u64,
    pub available: u64,
    pub issued: u64,
    pub completed: u64,
    pub received: u64,
    pub search_covered: bool,
    pub cutoff_tie: bool,
    pub search_candidates: usize,
    pub search_pruned: usize,
    pub reconstruction_error: f64,
    pub pruned_ties: usize,
    pub deadline: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::temporal_cognition) struct State {
    // Zero is known no-memory; epistemic unknown is None outside this type.
    pub id: u64,
    pub group: Handle,
    pub support: Support,
    pub matched: Option<recall::MatchSnapshot>,
    pub log_score: f64,
}

impl State {
    fn same_assignment(&self, other: &Self) -> bool {
        self.group == other.group
            && match (self.matched, other.matched) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    self.support.start == other.support.start
                        && self.support.occurrence == other.support.occurrence
                        && (a.episode_id, a.episode_generation, a.anchor)
                            == (b.episode_id, b.episode_generation, b.anchor)
                        && a.transformation == b.transformation
                }
                _ => false,
            }
    }
}

pub(in crate::temporal_cognition) struct Cache {
    pub(super) group: Handle,
    pub(super) rate: u32,
    model: TemporalSectionConfig,
    no_memory_bias: Option<f64>,
    next_id: u64,
    pub(super) cut: u64,
    pub(super) support: Option<Support>,
    retrieval: section::Retrieval,
    pub(super) states: [Option<State>; 17],
    pub fresh_supported: usize,
    pub excluded_ambiguous: usize,
    pub excluded_unsupported: usize,
    pub excluded_retired: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Origin {
    Retained,
    Fresh(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Extension {
    pub id: u64,
    pub origin: Origin,
}

pub(in crate::temporal_cognition) struct Output {
    pub list: List,
    pub states: [Option<State>; 18],
    pub feature_supported: [bool; 18],
    pub(super) origins: [Option<Origin>; 18],
    pub fresh_truncated: usize,
    pub parent_retired: bool,
}

impl Cache {
    pub fn new(
        group: Handle,
        rate: u32,
        model: TemporalSectionConfig,
        no_memory_bias: Option<f64>,
    ) -> Result<Self, &'static str> {
        if group.bus > 1
            || group.generation == 0
            || rate == 0
            || no_memory_bias.is_some_and(|v| !v.is_finite())
        {
            return Err("invalid correspondence owner, rate or no-memory score");
        }
        section::Stream::validate(model)?;
        Ok(Self {
            group,
            rate,
            model,
            no_memory_bias,
            next_id: 1,
            cut: 0,
            support: None,
            retrieval: section::Retrieval::default(),
            states: [None; 17],
            fresh_supported: 0,
            excluded_ambiguous: 0,
            excluded_unsupported: 0,
            excluded_retired: 0,
        })
    }

    pub fn refresh(
        &mut self,
        cut: u64,
        query: Option<(
            recall::ResultSnapshot,
            &[[Option<recall::MatchSnapshot>; 4]],
        )>,
        retained: &[(u64, u64)],
    ) -> Result<(), &'static str> {
        if cut < self.cut {
            return Err("backward correspondence observation");
        }
        let mut next = [None; 17];
        let mut support = None;
        let mut retrieval = section::Retrieval::default();
        let (mut supported, mut ambiguous, mut unsupported, mut retired) = (0, 0, 0, 0);
        if let Some((q, rows)) = query {
            if q.group != self.group
                || q.query_id == 0
                || (q.search_covered && (q.cutoff_tie || q.pruned_candidates > 0))
                || rows.len() > 1024
                || !q.reconstruction_error.is_finite()
                || q.reconstruction_error < 0.
                || !(q.source_start_sample <= q.support_start_sample
                    && q.support_start_sample < q.support_end_sample
                    && q.support_end_sample <= q.available_at
                    && q.available_at <= q.issued_at
                    && q.issued_at <= q.completed_at
                    && q.completed_at <= q.received_at
                    && q.received_at <= cut
                    && q.received_at <= q.deadline)
                || q.supporting_audio_end
                    .is_some_and(|t| t > q.support_end_sample)
            {
                return Err("invalid correspondence query owner or original clock");
            }
            if let Some(audio_end) = q
                .supporting_audio_end
                .filter(|&t| cut <= q.deadline && u128::from(cut - t) * 2 < u128::from(self.rate))
            {
                support = Some(Support {
                    query: q.query_id,
                    occurrence: q.cue.map(|c| c.occurrence_id),
                    start: q.support_start_sample,
                    source_start: q.source_start_sample,
                    end: q.support_end_sample,
                    audio_end,
                    available: q.available_at,
                    issued: q.issued_at,
                    completed: q.completed_at,
                    received: q.received_at,
                    search_covered: q.search_covered,
                    cutoff_tie: q.cutoff_tie,
                    search_candidates: q.candidates,
                    search_pruned: q.pruned_candidates,
                    reconstruction_error: q.reconstruction_error,
                    pruned_ties: q.pruned_ties,
                    deadline: q.deadline,
                });
                for m in rows.iter().flatten().flatten() {
                    if !m.cost.is_finite()
                        || m.cost < 0.
                        || m.episode_id == 0
                        || m.episode_generation == 0
                        || m.source_start_sample > m.support_start_sample
                        || m.support_start_sample >= m.support_end_sample
                        || m.support_end_sample > q.support_start_sample
                        || m.available_at > q.issued_at
                        || m.transformation.iter().flatten().any(|v| !v.is_finite())
                    {
                        return Err("invalid correspondence target identity or support");
                    }
                    if !retained.contains(&(m.episode_id, m.episode_generation)) {
                        retired += 1;
                        continue;
                    }
                    if m.ambiguous {
                        ambiguous += 1;
                        continue;
                    }
                    let Some(score) = m
                        .residuals
                        .filter(|r| r.matched > 0)
                        .and_then(|_| m.acoustic_score(&self.model))
                    else {
                        unsupported += 1;
                        continue;
                    };
                    let candidate = State {
                        id: 0,
                        group: self.group,
                        support: support.unwrap(),
                        matched: Some(*m),
                        log_score: score,
                    };
                    if next
                        .iter()
                        .flatten()
                        .any(|p: &State| p.same_assignment(&candidate))
                    {
                        return Err("duplicate correspondence assignment in immutable query");
                    }
                    supported += 1;
                    let position = next[..16].iter().position(|p| {
                        p.as_ref().is_none_or(|p| {
                            let old = p.matched.unwrap();
                            score
                                .total_cmp(&p.log_score)
                                .reverse()
                                .then((m.episode_id, m.episode_generation, m.anchor).cmp(&(
                                    old.episode_id,
                                    old.episode_generation,
                                    old.anchor,
                                )))
                                .then_with(|| {
                                    m.transformation
                                        .iter()
                                        .zip(old.transformation)
                                        .map(|(a, b)| match (*a, b) {
                                            (Some(a), Some(b)) => a.total_cmp(&b),
                                            (None, Some(_)) => std::cmp::Ordering::Greater,
                                            (Some(_), None) => std::cmp::Ordering::Less,
                                            _ => std::cmp::Ordering::Equal,
                                        })
                                        .find(|v| !v.is_eq())
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .is_lt()
                        })
                    });
                    if let Some(i) = position {
                        next.copy_within(i..15, i + 1);
                        next[i] = Some(candidate);
                    }
                }
            }
        }
        if let Some((q, rows)) = query.filter(|(q, _)| {
            support.is_some()
                && q.cue.is_some_and(|c| {
                    c.occurrence_id > 0
                        && c.start_sample == q.support_start_sample
                        && c.support_end_sample == q.support_end_sample
                        && c.selected_at <= q.issued_at
                        && c.weighted_seconds > 0.
                })
        }) {
            retrieval = section::retrieval_scores(
                rows.iter()
                    .flatten()
                    .flatten()
                    .filter(|m| {
                        !m.ambiguous
                            && m.residuals.is_some_and(|r| r.matched > 0)
                            && retained.contains(&(m.episode_id, m.episode_generation))
                    })
                    .map(|m| {
                        (
                            crate::temporal_cognition::transport::Identity {
                                id: m.episode_id,
                                generation: m.episode_generation,
                            },
                            m.acoustic_score(&self.model),
                        )
                    }),
                q.supporting_audio_end,
                q.support_end_sample,
                cut,
                self.rate,
            )?;
        }
        let mut next_id = self.next_id;
        for candidate in next[..16].iter_mut().flatten() {
            candidate.id = if let Some(previous) = self
                .states
                .iter()
                .flatten()
                .find(|p| p.support == candidate.support && p.same_assignment(candidate))
            {
                previous.id
            } else {
                let id = next_id;
                next_id = next_id
                    .checked_add(1)
                    .ok_or("correspondence identity exhausted")?;
                id
            };
        }
        if let (Some(support), Some(log_score)) = (
            support.filter(|s| s.search_covered && unsupported == 0 && retired == 0),
            self.no_memory_bias,
        ) {
            next[16] = Some(State {
                id: 0,
                group: self.group,
                support,
                matched: None,
                log_score,
            });
        }
        self.cut = cut;
        self.support = support;
        self.retrieval = retrieval;
        self.states = next;
        self.next_id = next_id;
        self.fresh_supported = supported;
        self.excluded_ambiguous = ambiguous;
        self.excluded_unsupported = unsupported;
        self.excluded_retired = retired;
        Ok(())
    }

    pub(super) fn retrieval(
        &self,
        cut: u64,
        observed: bool,
        retained: &[(u64, u64)],
    ) -> Result<section::Retrieval, &'static str> {
        if cut != self.cut {
            return Err("retrieval projection belongs to another cut");
        }
        if !observed {
            return Ok(section::Retrieval::default());
        }
        if self
            .retrieval
            .targets
            .iter()
            .flatten()
            .any(|t| !retained.contains(&(t.id, t.generation)))
        {
            return Err("retired retrieval projection target");
        }
        Ok(self.retrieval)
    }

    pub(super) fn resolve<'a>(
        &'a self,
        choice: Extension,
        parent: Option<&'a State>,
        observed: bool,
        cut: u64,
        retained: &[(u64, u64)],
    ) -> Result<(&'a State, bool), &'static str> {
        if cut != self.cut
            || parent.is_some_and(|p| {
                p.group != self.group || p.support.received > cut || p.support.end > cut
            })
        {
            return Err("invalid correspondence source clock or parent owner");
        }
        let (state, supported) = match choice.origin {
            Origin::Retained => (
                parent.ok_or("missing retained correspondence parent")?,
                false,
            ),
            Origin::Fresh(i) => {
                if !observed {
                    return Err("missing interval cannot refresh correspondence evidence");
                }
                (
                    self.states
                        .get(usize::from(i))
                        .and_then(Option::as_ref)
                        .ok_or("missing fresh correspondence slot")?,
                    true,
                )
            }
        };
        let expected_id = if supported {
            parent
                .filter(|p| p.same_assignment(state))
                .map_or(state.id, |p| p.id)
        } else {
            state.id
        };
        if choice.id != expected_id {
            return Err("correspondence identity differs from selected source");
        }
        if state
            .matched
            .is_some_and(|m| !retained.contains(&(m.episode_id, m.episode_generation)))
        {
            return Err("retired correspondence target");
        }
        Ok((state, supported))
    }

    pub fn proposals(
        &self,
        parent: Option<State>,
        observed: bool,
        dt: f64,
        retained: &[(u64, u64)],
    ) -> Result<Output, &'static str> {
        if parent.is_some_and(|p| {
            p.group != self.group
                || !p.log_score.is_finite()
                || (p.id == 0) != p.matched.is_none()
                || (p.id != 0 && p.id >= self.next_id)
                || p.support.received > self.cut
                || p.support.end > self.cut
        }) {
            return Err("invalid correspondence parent owner, clock or identity");
        }
        let parent_retired = parent
            .and_then(|p| p.matched)
            .is_some_and(|m| !retained.contains(&(m.episode_id, m.episode_generation)));
        let mut states = [None; 18];
        let mut feature_supported = [false; 18];
        let mut origins = [None; 18];
        let mut count = 0;
        if let Some(p) = parent.filter(|_| !parent_retired) {
            let fresh = observed
                .then(|| {
                    self.states
                        .iter()
                        .enumerate()
                        .filter_map(|(i, s)| s.map(|s| (i, s)))
                        .find(|(_, s)| s.same_assignment(&p))
                })
                .flatten();
            states[0] = Some(fresh.map_or(p, |(_, s)| State { id: p.id, ..s }));
            feature_supported[0] = fresh.is_some();
            origins[0] = Some(fresh.map_or(Origin::Retained, |(i, _)| Origin::Fresh(i as u8)));
            count += 1;
        }
        if observed {
            for (i, candidate) in self
                .states
                .iter()
                .enumerate()
                .filter_map(|(i, s)| s.map(|s| (i, s)))
            {
                if candidate
                    .matched
                    .is_some_and(|m| !retained.contains(&(m.episode_id, m.episode_generation)))
                {
                    continue;
                }
                if states[..count]
                    .iter()
                    .flatten()
                    .any(|p| p.same_assignment(&candidate))
                {
                    continue;
                }
                states[count] = Some(candidate);
                feature_supported[count] = true;
                origins[count] = Some(Origin::Fresh(i as u8));
                count += 1;
            }
        }
        let max = states[..count]
            .iter()
            .flatten()
            .map(|p| p.log_score)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut raw = [Known {
            id: 0,
            raw_score: 0.,
            stay: false,
            boundary: None,
        }; 18];
        for (i, p) in states[..count].iter().flatten().enumerate() {
            let score = (p.log_score - max).exp();
            if score == 0. {
                return Err("unrepresentable correspondence transition ratio");
            }
            raw[i] = Known {
                id: p.id,
                raw_score: score,
                stay: parent.is_some_and(|old| old.id == p.id),
                boundary: None,
            };
        }
        Ok(Output {
            list: List::build(
                Kind::Correspondence,
                &raw[..count],
                parent.is_none(),
                dt,
                19,
            )?,
            states,
            feature_supported,
            origins,
            fresh_truncated: self.fresh_supported.saturating_sub(16),
            parent_retired,
        })
    }
}

#[cfg(test)]
mod tests;
