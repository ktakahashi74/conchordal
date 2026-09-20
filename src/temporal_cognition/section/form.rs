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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn return_requires_an_earlier_context_and_preserves_transformation_and_unknown_mass() {
        let a = crate::temporal_cognition::phrase::tests::input(1, 0.);
        let group = a.group_handles[0].unwrap();
        let mut original = Path::initial(group, 0, 1000, 40, 1).unwrap();
        let activity = Activity {
            window: [0., 0.1],
            numerators: [0.; 9],
            denominators: [0.; 9],
            physical_valid_seconds: [0.; 9],
            assignment_seconds: 0.,
            physical_window_seconds: 0.1,
        };
        original
            .state
            .history
            .observe(activity, group.epoch, group.generation, 1000, &[])
            .unwrap();
        original.state.end = 100;
        original.state.values = original.state.history.covariates(0.1, [None; 2]).unwrap();
        let mut head = Head {
            means: [0.; 82],
            deviations: [1.; 82],
            hazard: [0.; 83],
            exits: [[0.; 83]; 3],
        };
        head.hazard[0] = 100.;
        head.exits[0][0] = 20.;
        head.exits[1][0] = -10.;
        head.exits[2][0] = -10.;
        let delta = Activity {
            window: [0.1, 0.2],
            ..activity
        };
        let mut paths = Vec::new();
        let mut next = 40;
        let mut next_path = 1;
        advance(
            &[original],
            &mut paths,
            Step {
                group,
                previous_end: 100,
                end: 200,
                rate: 1000,
                observed: true,
                delta,
                deliveries: &[],
                completed: None,
                retrieval: [None; 2],
                query: None,
                returns: &[],
            },
            &head,
            &mut next,
            &mut next_path,
        )
        .unwrap();
        let mut changed = paths.remove(0);
        assert!(matches!(changed.state.relation, Relation::NewContext));
        assert_eq!(changed.transition, Transition::NewContext);
        assert_eq!(changed.parent, 1);
        assert!(changed.id > changed.parent);
        assert_ne!(changed.state.context, 40);
        let delta = Activity {
            window: [0.2, 0.3],
            ..activity
        };
        changed
            .state
            .history
            .observe(delta, group.epoch, group.generation, 1000, &[])
            .unwrap();
        changed.state.end = 300;
        changed.state.values = changed
            .state
            .history
            .covariates(0.3, [Some(0.), None])
            .unwrap();
        let focus = recall::MatchSnapshot {
            episode_id: 9,
            episode_generation: 9,
            support_start_sample: 0,
            support_end_sample: 100,
            source_start_sample: 0,
            available_at: 150,
            cost: 0.1,
            transformation: [Some(0.1), Some(0.)],
            ambiguous: false,
            path_steps: 10,
            anchor: Some(0),
            residuals: None,
        };
        let returns = [(
            focus,
            ContextSupport {
                context_id: 40,
                mass: 0.25,
            },
        )];
        head.exits[0][0] = -10.;
        head.exits[1][0] = 20.;
        let delta = Activity {
            window: [0.3, 0.4],
            ..activity
        };
        advance(
            &[changed.clone()],
            &mut paths,
            Step {
                group,
                previous_end: 300,
                end: 400,
                rate: 1000,
                observed: true,
                delta,
                deliveries: &[],
                completed: None,
                retrieval: [Some(0.), None],
                query: None,
                returns: &returns,
            },
            &head,
            &mut next,
            &mut next_path,
        )
        .unwrap();
        assert!(matches!(
            paths[0].state.relation,
            Relation::TransformedRecurrence
        ));
        assert_eq!(paths[0].transition, Transition::Return);
        let law = head.law(&changed.state.values, 0.1, 0.2).unwrap().unwrap();
        for path in &paths {
            assert_eq!(path.parent, changed.id);
            assert!(path.id > changed.id);
            let expected = match path.transition {
                Transition::Stay => law[0],
                Transition::NewContext => law[1],
                Transition::Return => law[2] * 0.25,
                Transition::Contrast => law[3],
                Transition::Development => panic!("returning context differs from current context"),
            };
            assert!((path.transition_score - expected).abs() < 1e-14);
            assert!((path.weight - changed.weight * expected).abs() < 1e-14);
        }
        assert_eq!(paths[0].state.context, 40);
        assert_eq!(paths[0].state.focus.unwrap().episode_id, 9);
        assert!(paths.iter().map(|p| p.weight).sum::<f64>() < 0.251);
        for development in [false, true] {
            let selected_returns = [(
                focus,
                ContextSupport {
                    context_id: if development {
                        changed.state.context
                    } else {
                        40
                    },
                    mass: 0.25,
                },
            )];
            advance(
                &[changed.clone()],
                &mut paths,
                Step {
                    group,
                    previous_end: 300,
                    end: 400,
                    rate: 1000,
                    observed: true,
                    delta,
                    deliveries: &[],
                    completed: None,
                    retrieval: [Some(0.), None],
                    query: None,
                    returns: &selected_returns,
                },
                &head,
                &mut next,
                &mut next_path,
            )
            .unwrap();
            let transitions = [
                Transition::Stay,
                Transition::NewContext,
                if development {
                    Transition::Development
                } else {
                    Transition::Return
                },
                Transition::Contrast,
            ];
            let ids = transitions.map(|t| paths.iter().find(|p| p.transition == t).unwrap().id);
            let list = crate::temporal_cognition::joint::proposals::List::active_section(
                ids,
                &changed.state.values,
                [300 - changed.state.start, 400 - changed.state.start],
                1000,
                Some((0.25, development)),
                &head,
            )
            .unwrap();
            let total: f64 = paths.iter().map(|p| p.transition_score).sum();
            for path in &paths {
                let row = list
                    .entries
                    .iter()
                    .flatten()
                    .find(|r| r.id == Some(path.id))
                    .unwrap();
                let expected = path.transition_score / total * (-0.1_f64 / 120.).exp();
                assert!((row.log_weight.exp() - expected).abs() < 1e-14);
                assert_eq!(row.stay, path.transition == Transition::Stay);
                if path.transition == Transition::Development {
                    assert!(!row.stay);
                    assert_eq!(
                        row.boundary,
                        Some(crate::temporal_cognition::joint::proposals::Boundary::Stay)
                    );
                    assert_eq!(path.state.context, changed.state.context);
                    assert_eq!(path.state.start, changed.state.start);
                    assert_ne!(path.id, changed.id);
                    assert_eq!(path.parent, changed.id);
                    assert!(matches!(path.state.relation, Relation::Development));
                    assert_eq!(path.state.focus.unwrap().episode_id, focus.episode_id);
                }
            }
        }
        advance(
            &[changed.clone()],
            &mut paths,
            Step {
                group,
                previous_end: 300,
                end: 400,
                rate: 1000,
                observed: true,
                delta,
                deliveries: &[],
                completed: None,
                retrieval: [None; 2],
                query: None,
                returns: &[],
            },
            &head,
            &mut next,
            &mut next_path,
        )
        .unwrap();
        assert!(paths.iter().all(|p| p.state.focus.is_none()));
        advance(
            &[changed.clone()],
            &mut paths,
            Step {
                group,
                previous_end: 300,
                end: 400,
                rate: 1000,
                observed: false,
                delta,
                deliveries: &[],
                completed: None,
                retrieval: [Some(0.), None],
                query: None,
                returns: &returns,
            },
            &head,
            &mut next,
            &mut next_path,
        )
        .unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].state.context, changed.state.context);
        assert!(paths[0].state.focus.is_none());
        assert_eq!(paths[0].transition, Transition::Stay);
        assert!(matches!(paths[0].state.relation, Relation::NewContext));
        assert_eq!(paths[0].parent, changed.id);
        assert_eq!(paths[0].transition_score, law[0]);
        let stay = paths.remove(0);
        next_path = u64::MAX;
        let error = advance(
            &[stay],
            &mut paths,
            Step {
                group,
                previous_end: 400,
                end: 500,
                rate: 1000,
                observed: false,
                delta: Activity {
                    window: [0.4, 0.5],
                    ..delta
                },
                deliveries: &[],
                completed: None,
                retrieval: [None; 2],
                query: None,
                returns: &[],
            },
            &head,
            &mut next,
            &mut next_path,
        );
        assert_eq!(error, Err("section path identity exhausted"));
        assert_eq!(next_path, u64::MAX);
        assert!(paths.is_empty());
    }
}
