//! Section interpretation shared by marginal and joint paths; no hypothesis weight.

use super::{
    Activity, Commit, History,
    form::{Relation, Transition},
};
use crate::temporal_cognition::{accents::Delivery, recall, ridge::Handle};

#[derive(Debug, PartialEq)]
pub(in crate::temporal_cognition) struct Interpretation {
    pub group: Handle,
    pub end: u64,
    pub history: History,
    pub start: u64,
    pub context: u64,
    pub relation: Relation,
    pub focus: Option<recall::MatchSnapshot>,
    pub query: Option<recall::ResultSnapshot>,
    pub values: [Option<f64>; 82],
    pub owners: Vec<(u64, u64, u64)>,
    pub late_accent_support: f64,
}

impl Clone for Interpretation {
    fn clone(&self) -> Self {
        Self {
            history: self.history.clone(),
            owners: self.owners.clone(),
            ..*self
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.history.clone_from(&source.history);
        self.owners.clone_from(&source.owners);
        self.group = source.group;
        self.end = source.end;
        self.start = source.start;
        self.context = source.context;
        self.relation = source.relation;
        self.focus = source.focus;
        self.query = source.query;
        self.values = source.values;
        self.late_accent_support = source.late_accent_support;
    }
}

#[derive(Clone, Copy)]
pub(in crate::temporal_cognition) struct Input<'a> {
    pub group: Handle,
    pub interval: [u64; 2],
    pub rate: u32,
    pub observed: bool,
    pub delta: Activity,
    pub deliveries: &'a [Delivery],
    pub completed: Option<(Commit, Activity, u64, f64)>,
    pub retrieval: [Option<f64>; 2],
    pub query: Option<recall::ResultSnapshot>,
}

#[derive(Clone, Copy)]
pub(in crate::temporal_cognition) struct Change {
    pub transition: Transition,
    pub context: u64,
    pub focus: Option<recall::MatchSnapshot>,
}

impl Interpretation {
    pub fn new(group: Handle, start: u64, rate: u32, context: u64) -> Result<Self, &'static str> {
        if rate == 0 {
            return Err("invalid section sample rate");
        }
        Ok(Self {
            group,
            end: start,
            start,
            context,
            history: History::new(
                group.epoch,
                group.generation,
                context,
                start as f64 / f64::from(rate),
                4,
            )?,
            relation: Relation::Initial,
            focus: None,
            query: None,
            values: [None; 82],
            owners: Vec::with_capacity(512),
            late_accent_support: 0.,
        })
    }

    #[cfg(test)]
    pub(in crate::temporal_cognition) fn reset(
        &mut self,
        group: Handle,
        start: u64,
        rate: u32,
        context: u64,
    ) -> Result<(), &'static str> {
        if rate == 0 {
            return Err("invalid section sample rate");
        }
        self.history.reset(
            group.epoch,
            group.generation,
            context,
            start as f64 / f64::from(rate),
        )?;
        self.group = group;
        self.end = start;
        self.start = start;
        self.context = context;
        self.relation = Relation::Initial;
        self.focus = None;
        self.query = None;
        self.values.fill(None);
        self.owners.clear();
        self.late_accent_support = 0.;
        Ok(())
    }

    #[cfg(test)]
    pub(in crate::temporal_cognition) fn storage(&self) -> [(usize, usize); 2] {
        [
            (
                self.history.ring.as_ptr() as usize,
                self.history.ring.capacity(),
            ),
            (self.owners.as_ptr() as usize, self.owners.capacity()),
        ]
    }

    pub fn advance(&mut self, input: Input<'_>, change: Change) -> Result<(), &'static str> {
        if input.group != self.group
            || input.rate == 0
            || input.interval[0] != self.end
            || input.interval[1] <= input.interval[0]
            || input.delta.window != input.interval.map(|t| t as f64 / f64::from(input.rate))
            || change.context == 0
            || (matches!(
                change.transition,
                Transition::Return | Transition::Development
            ) && change.focus.is_none())
            || (!input.observed && change.transition != Transition::Stay)
            || (matches!(
                change.transition,
                Transition::Stay | Transition::Development
            ) && change.context != self.context)
        {
            return Err("invalid section interpretation owner, interval or transition");
        }
        let kind = match change.transition {
            Transition::Stay => 0,
            Transition::NewContext => 1,
            Transition::Return => 2,
            Transition::Contrast => 3,
            Transition::Development => 4,
        };
        let rate = f64::from(input.rate);
        self.history.observe(
            input.delta,
            input.group.epoch,
            input.group.generation,
            input.rate,
            input.deliveries,
        )?;
        if let Some((mut record, activity, sequence, adjacency)) = input.completed {
            if let (Some(focus), Some(query)) = (self.focus, self.query)
                && query.support_start_sample == (record.start * rate).round() as u64
                && query.support_end_sample == (record.support_end * rate).round() as u64
                && query.received_at <= input.interval[1]
                && (input.interval[1].saturating_sub(query.supporting_audio_end.unwrap_or(0))
                    as f64)
                    < rate * 0.5
            {
                record.assignment = super::Assignment {
                    status: super::Status::Match,
                    cost: Some(focus.cost),
                    supported: true,
                    search_completed: true,
                    search_covered: query.search_covered,
                    search_nonempty: true,
                    frequency_shift_log2: focus.transformation[0],
                    tempo_shift_log2: focus.transformation[1],
                    bound_hit: focus.transformation.iter().flatten().any(|v| v.abs() >= 2.),
                    ambiguous_cutoff: focus.ambiguous || query.cutoff_tie,
                };
            }
            if record.start >= self.start as f64 / rate {
                self.history.commit(record, activity, sequence, adjacency)?;
            }
            self.owners.retain(|(_, end, _)| {
                u128::from(*end) * 2 + u128::from(input.rate) >= u128::from(input.interval[1]) * 2
            });
            if self.owners.len() < 512 {
                self.owners.push((
                    record.occurrence_id,
                    (record.support_end * rate).round() as u64,
                    self.context,
                ));
            }
        }
        for delivery in input.deliveries {
            self.late_accent_support += self.history.admit_pending(*delivery, input.rate);
        }
        if (1..=3).contains(&kind) {
            self.context = change.context;
            self.start = input.interval[1];
            self.history.reset(
                input.group.epoch,
                input.group.generation,
                self.context,
                input.interval[1] as f64 / rate,
            )?;
            self.focus = change.focus;
            self.query = change.focus.and(input.query);
            self.relation = match kind {
                1 => Relation::NewContext,
                2 => {
                    if change.focus.is_some_and(|m| {
                        m.cost > 0.25
                            || m.transformation
                                .iter()
                                .flatten()
                                .any(|v| v.abs() > 1. / 48.)
                    }) {
                        Relation::TransformedRecurrence
                    } else {
                        Relation::Recurrence
                    }
                }
                _ => Relation::Contrast,
            };
        }
        if kind == 4 {
            self.focus = change.focus;
            self.query = input.query;
            self.relation = Relation::Development;
        }
        self.values = self
            .history
            .covariates(input.interval[1] as f64 / rate, input.retrieval)?;
        self.end = input.interval[1];
        Ok(())
    }
}

#[cfg(test)]
mod tests;
