//! Joint proposals from the existing acoustic grouping inventory, without rescanning accents.

use super::proposals::{Kind, Known, List};
use crate::temporal_cognition::{
    accents::periods::{Estimator, groupings},
    ridge::Handle,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Support {
    pub refreshed_at: u64,
    pub source_end: u64,
    pub available: u64,
    pub retained_span: (u64, u64),
    pub capacity_evicted_through: Option<u64>,
    pub window_limited_cases: usize,
    pub supported_period_pairs: usize,
    pub period_peaks: usize,
    // Admission cases excluded by the bounded inventory, not unique interpretations.
    pub excluded_cases: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct State {
    // Zero is known ungrouped; epistemic unknown is represented outside this payload.
    pub id: u64,
    pub group: Handle,
    pub support: Support,
    pub proposal: Option<groupings::Proposal>,
    pub raw_score: f64,
}

impl State {
    fn same_assignment(&self, other: &Self) -> bool {
        self.group == other.group && self.proposal.map(|p| p.key) == other.proposal.map(|p| p.key)
    }
}

pub(super) struct Cache {
    pub(super) group: Handle,
    ungrouped_score: Option<f64>,
    pub(super) cut: u64,
    pub(super) admission: Option<f64>,
    next_id: u64,
    states: [Option<State>; 17],
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

pub(super) struct Output {
    pub list: List,
    pub states: [Option<State>; 18],
    pub feature_supported: [bool; 18],
    pub origins: [Option<Origin>; 18],
}

impl Cache {
    pub fn new(group: Handle, ungrouped_score: Option<f64>) -> Result<Self, &'static str> {
        if group.bus > 1
            || group.generation <= 1
            || ungrouped_score.is_some_and(|v| !v.is_finite() || v <= 0.)
        {
            return Err("invalid grouping owner or ungrouped score");
        }
        Ok(Self {
            group,
            ungrouped_score,
            cut: 0,
            admission: None,
            next_id: 1,
            states: [None; 17],
        })
    }

    pub fn refresh(
        &mut self,
        cut: u64,
        source: Option<(&groupings::View, &Estimator)>,
    ) -> Result<(), &'static str> {
        if cut < self.cut {
            return Err("backward grouping observation");
        }
        let mut states = [None; 17];
        let mut admission = None;
        let mut next_id = self.next_id;
        if let Some((view, estimator)) = source {
            let ledger = estimator.ledger_summary();
            if view.group != self.group
                || ledger.group != self.group
                || view.refreshed_at > cut
                || ledger.received_at > cut
                || view.period_source_end > view.period_available_end
                || view
                    .period_available_end
                    .is_some_and(|t| t > view.refreshed_at)
            {
                return Err("invalid grouping source owner or original clock");
            }
            if let (true, Some(source_end), Some(available), Some(retained_span)) = (
                view.matches_source(estimator),
                view.period_source_end,
                view.period_available_end,
                view.retained_span,
            ) {
                let count = view.proposals.iter().flatten().count();
                admission = view.diagnostics().admission_support;
                let period = estimator.view();
                let support = Support {
                    refreshed_at: view.refreshed_at,
                    supported_period_pairs: period.supported_pairs,
                    period_peaks: period.peaks.iter().flatten().count(),
                    source_end,
                    available,
                    retained_span,
                    capacity_evicted_through: view.capacity_evicted_through,
                    window_limited_cases: view.work.integer_window_limited_cases
                        + view.work.word_insufficient_endpoints,
                    excluded_cases: view.work.admitted_cases
                        - view.work.duplicate_retained_keys
                        - count,
                };
                for (i, proposal) in view.proposals.iter().enumerate() {
                    let Some(proposal) = proposal else { continue };
                    if !proposal.endpoint_weight.is_finite()
                        || proposal.endpoint_weight <= 0.
                        || proposal.source_start > proposal.source_end
                        || proposal.source_end > proposal.available_end
                        || proposal.available_end > available
                    {
                        return Err("invalid grouping proposal score or original clock");
                    }
                    let mut state = State {
                        id: 0,
                        group: self.group,
                        support,
                        proposal: Some(*proposal),
                        raw_score: proposal.endpoint_weight,
                    };
                    state.id = if let Some(old) = self
                        .states
                        .iter()
                        .flatten()
                        .find(|old| old.same_assignment(&state))
                    {
                        old.id
                    } else {
                        let id = next_id;
                        next_id = next_id
                            .checked_add(1)
                            .ok_or("grouping identity exhausted")?;
                        id
                    };
                    states[i] = Some(state);
                }
                // Incomplete search cannot support the absence of a grouping interpretation.
                if let Some(raw_score) = self.ungrouped_score.filter(|_| {
                    support.supported_period_pairs > 0
                        && support.period_peaks > 0
                        && support.capacity_evicted_through.is_none()
                        && support.window_limited_cases == 0
                        && support.excluded_cases == 0
                }) {
                    states[16] = Some(State {
                        id: 0,
                        group: self.group,
                        support,
                        proposal: None,
                        raw_score,
                    });
                }
            }
        }
        self.cut = cut;
        self.admission = admission;
        self.states = states;
        self.next_id = next_id;
        Ok(())
    }

    pub(super) fn resolve<'a>(
        &'a self,
        choice: Extension,
        parent: Option<&'a State>,
        observed: bool,
        cut: u64,
    ) -> Result<(&'a State, bool), &'static str> {
        if cut != self.cut
            || parent.is_some_and(|p| {
                p.group != self.group || p.support.refreshed_at > cut || p.support.available > cut
            })
        {
            return Err("invalid grouping source clock or parent owner");
        }
        let (state, supported) = match choice.origin {
            Origin::Retained => (parent.ok_or("missing retained grouping parent")?, false),
            Origin::Fresh(i) => {
                if !observed {
                    return Err("missing interval cannot refresh grouping evidence");
                }
                (
                    self.states
                        .get(usize::from(i))
                        .and_then(Option::as_ref)
                        .ok_or("missing fresh grouping slot")?,
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
            return Err("grouping identity differs from selected source");
        }
        Ok((state, supported))
    }

    pub fn proposals(
        &self,
        parent: Option<State>,
        observed: bool,
        dt: f64,
    ) -> Result<Output, &'static str> {
        if parent.is_some_and(|p| {
            p.group != self.group
                || !p.raw_score.is_finite()
                || p.raw_score <= 0.
                || (p.id == 0) != p.proposal.is_none()
                || p.id >= self.next_id
                || p.support.refreshed_at > self.cut
                || p.support.available > self.cut
        }) {
            return Err("invalid grouping parent owner, score, clock or identity");
        }
        let mut states = [None; 18];
        let mut feature_supported = [false; 18];
        let mut origins = [None; 18];
        let mut count = 0;
        if let Some(parent) = parent {
            let fresh = self
                .states
                .iter()
                .enumerate()
                .filter_map(|(i, s)| s.as_ref().map(|s| (i, s)))
                .find(|(_, s)| observed && s.same_assignment(&parent));
            states[0] = Some(fresh.map_or(parent, |(_, s)| State {
                id: parent.id,
                ..*s
            }));
            feature_supported[0] = fresh.is_some();
            origins[0] = Some(fresh.map_or(Origin::Retained, |(i, _)| Origin::Fresh(i as u8)));
            count += 1;
        }
        if observed {
            for (i, state) in self
                .states
                .iter()
                .enumerate()
                .filter_map(|(i, s)| s.as_ref().map(|s| (i, s)))
            {
                if states[..count]
                    .iter()
                    .flatten()
                    .any(|s| s.same_assignment(state))
                {
                    continue;
                }
                states[count] = Some(*state);
                feature_supported[count] = true;
                origins[count] = Some(Origin::Fresh(i as u8));
                count += 1;
            }
        }
        // Scaling preserves the existing endpoint-weight ranking and avoids sum overflow.
        let max = states[..count]
            .iter()
            .flatten()
            .map(|s| s.raw_score)
            .fold(0., f64::max);
        let mut raw = [Known {
            id: 0,
            raw_score: 0.,
            stay: false,
            boundary: None,
        }; 18];
        for (i, state) in states[..count].iter().flatten().enumerate() {
            raw[i] = Known {
                id: state.id,
                raw_score: state.raw_score / max,
                stay: parent.is_some_and(|p| p.id == state.id),
                boundary: None,
            };
            if raw[i].raw_score == 0. {
                return Err("unrepresentable grouping score ratio");
            }
        }
        Ok(Output {
            list: List::build(Kind::Grouping, &raw[..count], parent.is_none(), dt, 19)?,
            states,
            feature_supported,
            origins,
        })
    }
}

#[cfg(test)]
pub(super) mod tests;
