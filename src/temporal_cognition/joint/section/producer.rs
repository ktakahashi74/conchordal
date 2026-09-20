//! Section component proposals from retained state and immutable acoustic sources.

use super::{Return, Returns, Source};
use crate::temporal_cognition::{
    joint::{
        correspondence,
        owner::{LocalPath, Snapshot},
        proposals::{Boundary, Compatibility, Kind, Known, List},
    },
    section::{
        Head,
        form::Transition,
        input::Observation,
        interpretation::{Change, Input},
    },
};

pub(in crate::temporal_cognition::joint) struct Inputs<'a, 'r> {
    pub snapshot: &'a Snapshot,
    pub parent: Option<&'a LocalPath>,
    pub observation: &'a Observation,
    pub correspondences: &'a correspondence::Output,
    pub cache: &'a correspondence::Cache,
    pub retained_episodes: &'a [(u64, u64)],
    pub returns: &'r Returns,
    pub head: &'a Head,
    // Stay, new, return/development, contrast; allocation belongs to the local producer.
    pub ids: [u64; 4],
    pub new_contexts: [u64; 2],
    pub rms_reference: f64,
}

pub(in crate::temporal_cognition::joint) struct Output {
    pub list: List,
    pub sources: [Option<(u64, Source<'static>)>; 4],
    pub examined_returns: usize,
}

impl Output {
    pub fn compatibility(&self, correspondences: &correspondence::Output) -> Compatibility {
        let mut legal = Compatibility::default();
        for (s, choice) in self.list.entries[..self.list.len]
            .iter()
            .flatten()
            .enumerate()
        {
            let Some((_, source)) = self
                .sources
                .iter()
                .flatten()
                .find(|(id, _)| Some(*id) == choice.id)
            else {
                continue;
            };
            if source.returned.is_none() {
                continue;
            }
            for (c, candidate) in correspondences.list.entries[..correspondences.list.len]
                .iter()
                .flatten()
                .enumerate()
            {
                let selected = correspondences
                    .states
                    .iter()
                    .enumerate()
                    .find_map(|(i, state)| {
                        state
                            .as_ref()
                            .filter(|state| Some(state.id) == candidate.id)
                            .map(|state| (state, correspondences.feature_supported[i]))
                    });
                legal.section_correspondence[s][c] = source.validate_return(selected).is_ok();
            }
        }
        legal
    }
}

pub(in crate::temporal_cognition::joint) fn build(
    input: Inputs<'_, '_>,
) -> Result<Output, &'static str> {
    let Inputs {
        snapshot,
        parent,
        observation: o,
        correspondences,
        cache,
        retained_episodes,
        returns,
        head,
        ids,
        new_contexts,
        rms_reference,
    } = input;
    let old = parent
        .and_then(|p| p.section_slot)
        .map(|i| {
            snapshot
                .sections
                .get(i)
                .ok_or("invalid section producer parent slot")
        })
        .transpose()?;
    if snapshot.end != o.interval[0]
        || o.interval[1] <= o.interval[0]
        || o.rate == 0
        || returns.group != o.group
        || cache.group != o.group
        || cache.rate != o.rate
        || !rms_reference.is_finite()
        || rms_reference < 1e-6
        || parent.is_some_and(|p| p.components[3].is_some() != p.section_slot.is_some())
        || old.is_some_and(|p| {
            p.group != o.group || p.end != o.interval[0] || p.start > o.interval[0]
        })
        || correspondences
            .states
            .iter()
            .flatten()
            .any(|s| s.group != o.group || s.support.received > o.interval[1])
        || ids
            .iter()
            .enumerate()
            .any(|(i, id)| *id == 0 || ids[..i].contains(id))
        || parent
            .and_then(|p| p.components[3])
            .is_some_and(|id| ids[0] != id || ids[1..].iter().any(|i| *i <= id))
        || new_contexts[0] == 0
        || new_contexts[1] == 0
        || new_contexts[0] == new_contexts[1]
        || snapshot
            .paths
            .iter()
            .flatten()
            .flat_map(|p| p.groups.iter().flatten().flatten())
            .filter_map(|p| p.section_slot)
            .any(|i| new_contexts.contains(&snapshot.sections[i].context))
        || returns
            .entries()
            .any(|r| new_contexts.contains(&r.context().context_id))
    {
        return Err("invalid section producer clock, owner or identity");
    }
    head.validate()?;
    let raw = o.raw.filter(|_| o.observed);
    if o.observed
        && raw.is_none_or(|u| {
            u.raw.group != o.group
                || u.raw.start < o.interval[0]
                || u.raw.start >= u.raw.end
                || u.raw.end != o.interval[1]
                || u.raw.source_start > u.raw.start
                || u.raw.source_end < u.raw.end
                || u.raw.available_end > o.interval[1]
                || u.raw.source_end > u.raw.available_end
                || u.raw.known_samples != u.raw.end - u.raw.start
                || u.raw.values.iter().flatten().any(|v| !v.is_finite())
        })
        || !o.delta.assignment_seconds.is_finite()
        || o.delta.window != o.interval.map(|t| t as f64 / f64::from(o.rate))
        || o.delta.physical_window_seconds
            != (o.interval[1] - o.interval[0]) as f64 / f64::from(o.rate)
        || o.delta.assignment_seconds < 0.
        || o.delta.assignment_seconds > o.delta.physical_window_seconds
    {
        return Err("invalid section producer acoustic support");
    }
    let novel = raw
        .and_then(|u| u.raw.values[2])
        .is_some_and(|rms| rms > (0.01 * rms_reference).log2())
        && o.delta.assignment_seconds > 0.;
    let mut returned: Option<(&Return, &correspondence::State)> = None;
    let mut examined_returns = 0;
    if o.observed {
        for (i, state) in correspondences
            .states
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|s| (i, s)))
        {
            if !correspondences.feature_supported[i] {
                continue;
            }
            let Some(proof) = returns
                .for_correspondence(state)
                .filter(|p| p.current(o.interval[1], o.rate).is_some())
            else {
                continue;
            };
            examined_returns += 1;
            let matched = state.matched.unwrap();
            if returned.is_none_or(|(previous, previous_state)| {
                let old_match = previous_state.matched.unwrap();
                proof
                    .context()
                    .mass
                    .total_cmp(&previous.context().mass)
                    .reverse()
                    .then(
                        (
                            proof.context().context_id,
                            matched.episode_id,
                            matched.episode_generation,
                            matched.anchor,
                            state.id,
                        )
                            .cmp(&(
                                previous.context().context_id,
                                old_match.episode_id,
                                old_match.episode_generation,
                                old_match.anchor,
                                previous_state.id,
                            )),
                    )
                    .is_lt()
            }) {
                returned = Some((proof, state));
            }
        }
    }
    let dt = (o.interval[1] - o.interval[0]) as f64 / f64::from(o.rate);
    let mut scores = if let Some(old) = old {
        if o.observed {
            let elapsed = o
                .interval
                .map(|t| (t - old.start) as f64 / f64::from(o.rate));
            head.law(&old.values, elapsed[0], elapsed[1])?
                .unwrap_or([0.; 4])
        } else {
            let elapsed = o
                .interval
                .map(|t| (t - old.start) as f64 / f64::from(o.rate));
            [
                head.law(&old.values, elapsed[0], elapsed[1])?
                    .map_or(0., |law| law[0]),
                0.,
                0.,
                0.,
            ]
        }
    } else {
        [0., f64::from(novel), 1., 0.]
    };
    scores[2] *= returned.map_or(0., |(r, _)| r.context().mass);
    let base = Input {
        group: o.group,
        interval: o.interval,
        rate: o.rate,
        observed: o.observed,
        delta: o.delta,
        completed: None,
        deliveries: &[],
        retrieval: cache
            .retrieval(o.interval[1], o.observed, retained_episodes)?
            .values,
        query: None,
    };
    let mut sources = [None; 4];
    let mut entries = [Known {
        id: 0,
        raw_score: 0.,
        stay: false,
        boundary: Some(Boundary::Exit),
    }; 4];
    for i in 0..4 {
        let source = if i == 2 {
            returned.map(|(r, _)| Source::returning(parent.map(|p| p.id), base, old, r))
        } else {
            let context = match i {
                0 => old.map(|p| p.context),
                1 => Some(new_contexts[0]),
                _ => Some(new_contexts[1]),
            };
            context.map(|context| Source {
                parent: parent.map(|p| p.id),
                input: base,
                returned: None,
                change: Change {
                    transition: [
                        Transition::Stay,
                        Transition::NewContext,
                        Transition::Return,
                        Transition::Contrast,
                    ][i],
                    context,
                    focus: None,
                },
            })
        };
        let boundary =
            if i == 0 || source.is_some_and(|s| s.change.transition == Transition::Development) {
                Boundary::Stay
            } else {
                Boundary::Exit
            };
        entries[i] = Known {
            id: ids[i],
            raw_score: scores[i],
            stay: i == 0 && old.is_some(),
            boundary: Some(boundary),
        };
        if scores[i] > 0. {
            sources[i] = Some((ids[i], source.ok_or("missing admitted section source")?));
        }
    }
    Ok(Output {
        list: if !o.observed && old.is_some() {
            List::missing_boundary(Kind::Section, ids[0], scores[0], dt)?
        } else {
            List::build(Kind::Section, &entries, old.is_none(), dt, 5)?
        },
        sources,
        examined_returns,
    })
}
