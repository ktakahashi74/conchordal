//! Conditional section source references; retained payloads are updated only after pruning.

use crate::temporal_cognition::{
    joint::correspondence,
    recall,
    ridge::Handle,
    section::{
        commitment::{ContextSupport, Evidence},
        form::{Relation, Transition},
        interpretation::{Change, Input, Interpretation},
    },
};

pub(super) mod producer;
mod returns;
pub(in crate::temporal_cognition) use returns::Returns;

#[derive(Clone, Copy)]
pub(in crate::temporal_cognition) struct Return {
    cut: u64,
    rate: u32,
    correspondence: correspondence::State,
    query: recall::ResultSnapshot,
    origin: Evidence,
    context: ContextSupport,
}

impl Return {
    pub(in crate::temporal_cognition::joint) fn current(
        &self,
        cut: u64,
        rate: u32,
    ) -> Option<(&correspondence::State, recall::ResultSnapshot)> {
        (self.cut == cut && self.rate == rate).then_some((&self.correspondence, self.query))
    }
    pub fn context(&self) -> ContextSupport {
        self.context
    }
    pub fn origin(&self) -> Evidence {
        self.origin
    }
}

#[derive(Clone, Copy)]
pub(super) struct Source<'a> {
    pub parent: Option<Handle>,
    pub input: Input<'a>,
    pub change: Change,
    pub returned: Option<Return>,
}

impl<'a> Source<'a> {
    pub(super) fn returning(
        parent: Option<Handle>,
        mut input: Input<'a>,
        previous: Option<&Interpretation>,
        returned: &Return,
    ) -> Self {
        input.query = Some(returned.query);
        Self {
            parent,
            input,
            returned: Some(*returned),
            change: Change {
                transition: if previous.is_some_and(|p| p.context == returned.context.context_id) {
                    Transition::Development
                } else {
                    Transition::Return
                },
                context: returned.context.context_id,
                focus: returned.correspondence.matched,
            },
        }
    }

    pub(super) fn validate_return(
        &self,
        selected: Option<(&correspondence::State, bool)>,
    ) -> Result<(), &'static str> {
        if !matches!(
            self.change.transition,
            Transition::Return | Transition::Development
        ) {
            return if self.returned.is_none() {
                Ok(())
            } else {
                Err("unused section return source")
            };
        }
        let proof = self
            .returned
            .ok_or("missing sealed section return source")?;
        let (state, true) = selected.ok_or("section return requires selected correspondence")?
        else {
            return Err("section return requires fresh correspondence support");
        };
        if !self.input.observed
            || proof.cut != self.input.interval[1]
            || proof.rate != self.input.rate
            || proof.correspondence.group != state.group
            || proof.correspondence.support != state.support
            || proof.correspondence.matched != state.matched
            || self.input.query != Some(proof.query)
            || self.change.context != proof.context.context_id
            || self.change.focus != state.matched
        {
            return Err("section return differs from selected correspondence or original query");
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(super) struct Extension {
    pub id: u64,
    pub parent_slot: Option<usize>,
    pub source_slot: usize,
}

impl Extension {
    pub(super) fn validate(
        &self,
        parent: Option<(&Interpretation, u64)>,
        source: &Source<'_>,
        group: Handle,
        interval: [u64; 2],
        observed: bool,
    ) -> Result<(), &'static str> {
        if self.id == 0
            || self.parent_slot.is_some() != parent.is_some()
            || source.input.group != group
            || source.input.interval != interval
            || source.input.observed != observed
            || source.input.rate == 0
            || source.change.context == 0
            || source
                .input
                .query
                .is_some_and(|q| q.group != group || q.received_at > interval[1])
            || source
                .change
                .focus
                .is_some_and(|m| m.available_at > interval[1])
        {
            return Err("invalid joint section source, identity or clock");
        }
        if let Some((parent, id)) = parent {
            if parent.group != group
                || parent.end != interval[0]
                || (source.change.transition == Transition::Stay && self.id != id)
                || (source.change.transition != Transition::Stay && self.id <= id)
            {
                return Err("invalid joint section parent identity or clock");
            }
        } else if !observed
            || !matches!(
                source.change.transition,
                Transition::NewContext | Transition::Return
            )
            || (source.change.transition == Transition::NewContext && source.change.focus.is_some())
            || (source.change.transition == Transition::Return && source.returned.is_none())
        {
            return Err("unsupported joint section admission");
        }
        Ok(())
    }

    pub(super) fn materialize(
        &self,
        parent: Option<&Interpretation>,
        source: &Source<'_>,
        destination: &mut Interpretation,
    ) -> Result<(), &'static str> {
        let mut change = source.change;
        if let Some(parent) = parent {
            destination.clone_from(parent);
        } else {
            destination.reset(
                source.input.group,
                source.input.interval[0],
                source.input.rate,
                change.context,
            )?;
            if change.transition == Transition::Return {
                destination.focus = change.focus;
                destination.query = source.input.query;
                destination.relation = if change.focus.is_some_and(|m| {
                    m.cost > 0.25
                        || m.transformation
                            .iter()
                            .flatten()
                            .any(|v| v.abs() > 1. / 48.)
                }) {
                    Relation::TransformedRecurrence
                } else {
                    Relation::Recurrence
                };
            }
            // Admission observes its first interval; it is not an exit at the right endpoint.
            change.transition = Transition::Stay;
        }
        destination.advance(source.input, change)
    }
}
