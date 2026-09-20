//! Conditional local section alternatives with retained earlier-context provenance.

use super::{
    Activity, Commit, Head,
    commitment::ContextSupport,
    interpretation::{Change, Input, Interpretation},
};
use crate::temporal_cognition::{accents::Delivery, recall, ridge::Handle};

pub(super) const LIMIT: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Relation {
    Initial,
    NewContext,
    Recurrence,
    TransformedRecurrence,
    Contrast,
    Development,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
// The current update's role, unlike a Relation carried through subsequent stays.
pub(crate) enum Transition {
    Stay,
    NewContext,
    Return,
    Contrast,
    Development,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Candidate {
    pub path_id: u64,
    pub parent_path_id: u64,
    pub phrase_path_index: usize,
    pub transition: Transition,
    pub transition_score: f64,
    pub context_id: u64,
    pub start_sample: u64,
    pub mass: f64,
    pub relation: Relation,
    pub focus: Option<recall::MatchSnapshot>,
    pub query_id: Option<u64>,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Parent {
    pub path_id: u64,
    pub phrase_path_index: usize,
}

#[derive(Clone)]
pub(super) struct Path {
    pub id: u64,
    pub parent: u64,
    pub transition: Transition,
    pub transition_score: f64,
    pub weight: f64,
    pub state: Interpretation,
}

#[derive(Clone, Copy)]
struct Proposal {
    parent: usize,
    kind: usize,
    weight: f64,
    raw_score: f64,
    context: u64,
    focus: Option<recall::MatchSnapshot>,
}

pub(super) struct Step<'a> {
    pub group: Handle,
    pub previous_end: u64,
    pub end: u64,
    pub rate: u32,
    pub observed: bool,
    pub delta: Activity,
    pub deliveries: &'a [Delivery],
    pub completed: Option<(Commit, Activity, u64, f64)>,
    pub retrieval: [Option<f64>; 2],
    pub query: Option<recall::ResultSnapshot>,
    pub returns: &'a [(recall::MatchSnapshot, ContextSupport)],
}

impl Path {
    pub fn initial(
        group: Handle,
        start: u64,
        rate: u32,
        context: u64,
        id: u64,
    ) -> Result<Self, &'static str> {
        Ok(Self {
            id,
            parent: id,
            transition: Transition::Stay,
            transition_score: 1.,
            weight: 1.,
            state: Interpretation::new(group, start, rate, context)?,
        })
    }
}

pub(super) fn advance(
    parents: &[Path],
    target: &mut Vec<Path>,
    step: Step<'_>,
    head: &Head,
    next: &mut u64,
    next_path: &mut u64,
) -> Result<(), &'static str> {
    let mut proposals = Vec::with_capacity(parents.len() * (3 + step.returns.len()));
    let rate = f64::from(step.rate);
    let return_sum: f64 = step.returns.iter().map(|(_, c)| c.mass).sum();
    for (i, p) in parents.iter().enumerate() {
        let lo = step.previous_end.saturating_sub(p.state.start) as f64 / rate;
        let hi = step.end.saturating_sub(p.state.start) as f64 / rate;
        let law = if p.state.values[78].is_none() {
            Some([1., 0., 0., 0.])
        } else {
            head.law(&p.state.values, lo, hi)?
        };
        let Some(law) = law else { continue };
        proposals.push(Proposal {
            parent: i,
            kind: 0,
            weight: p.weight * law[0],
            raw_score: law[0],
            context: p.state.context,
            focus: p.state.focus,
        });
        if !step.observed {
            continue;
        }
        for kind in [1, 3] {
            proposals.push(Proposal {
                parent: i,
                kind,
                weight: p.weight * law[kind],
                raw_score: law[kind],
                context: 0,
                focus: None,
            });
        }
        if return_sum > 0. {
            for &(focus, context) in step.returns {
                proposals.push(Proposal {
                    parent: i,
                    kind: if context.context_id == p.state.context {
                        4
                    } else {
                        2
                    },
                    weight: p.weight * law[2] * context.mass,
                    raw_score: law[2] * context.mass,
                    context: context.context_id,
                    focus: Some(focus),
                });
            }
        }
    }
    proposals.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then(a.parent.cmp(&b.parent))
            .then(a.kind.cmp(&b.kind))
            .then(a.context.cmp(&b.context))
    });
    target.clear();
    for proposal in proposals.iter().filter(|p| p.weight > 0.).take(LIMIT) {
        let parent = &parents[proposal.parent];
        let mut path = parent.clone();
        *next_path = next_path
            .checked_add(1)
            .ok_or("section path identity exhausted")?;
        path.id = *next_path;
        path.parent = parent.id;
        path.transition = [
            Transition::Stay,
            Transition::NewContext,
            Transition::Return,
            Transition::Contrast,
            Transition::Development,
        ][proposal.kind];
        path.transition_score = proposal.raw_score;
        path.weight = proposal.weight;
        let new_context = (1..=3).contains(&proposal.kind) && proposal.context == 0;
        let context = if new_context {
            next.checked_add(1)
                .ok_or("section context identity exhausted")?
        } else {
            proposal.context
        };
        path.state.advance(
            Input {
                group: step.group,
                interval: [step.previous_end, step.end],
                rate: step.rate,
                observed: step.observed,
                delta: step.delta,
                deliveries: step.deliveries,
                completed: step.completed,
                retrieval: step.retrieval,
                query: step.query,
            },
            Change {
                transition: path.transition,
                context,
                focus: proposal.focus,
            },
        )?;
        if new_context {
            *next = context;
        }
        target.push(path);
    }
    Ok(())
}
