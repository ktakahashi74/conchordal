//! Conditional base heads; acoustic issue weights and coverage never move into the future.

use super::{Heads, LocalPrediction, Rating, Snapshot, ratings};
use crate::temporal_cognition::{
    action_profiles::Cell, feature_projection::Feature, ridge::Handle,
};

pub(in crate::temporal_cognition) struct Issue<'a> {
    pub snapshot: Snapshot,
    heads: &'a Heads,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct CandidateHeads {
    pub scope: &'static str,
    pub issue_window: [u64; 2],
    pub evaluated_at: u64,
    pub selected_group: Handle,
    pub selected_group_weight: f64,
    pub observed_inputs: usize,
    pub projected_inputs: usize,
    pub unsupported_inputs: usize,
    pub selected_prediction: LocalPrediction,
    pub groove: Option<Rating>,
    pub desire: Option<Rating>,
    pub errors: [Option<&'static str>; 2],
}

impl Heads {
    pub(in crate::temporal_cognition) fn freeze(&self, snapshot: Snapshot) -> Option<Issue<'_>> {
        (snapshot.window[1] == self.cut).then_some(Issue {
            snapshot,
            heads: self,
        })
    }
}

impl Issue<'_> {
    #[inline(never)]
    pub(in crate::temporal_cognition) fn project(&self, cell: &Cell) -> Option<CandidateHeads> {
        let issue = self.snapshot.window[1];
        if cell.issued_at != issue
            || cell.evaluation_at < issue
            || cell.evaluation_at - issue > 4 * u64::from(self.heads.rate)
        {
            return None;
        }
        let density = cell.groove_density.as_ref();
        let timing = cell.groove_timing.as_ref();
        let context = cell.groove_context.as_ref();
        if density.is_some_and(|d| d.issued_at != issue || d.evaluated_at != cell.evaluation_at)
            || timing.is_some_and(|t| {
                t.issued_at != issue
                    || t.evaluated_at != cell.evaluation_at
                    || t.group != cell.group
            })
            || context.is_some_and(|c| c.issued_at != issue || c.evaluated_at != cell.evaluation_at)
        {
            return None;
        }
        let raw = std::array::from_fn(|i| match i {
            0..42 => timing.map_or(Feature::Unsupported, |t| t.values[i / 14][i % 14]),
            42..50 => density.map_or(Feature::Unsupported, |d| d.values[i - 42]),
            _ => context.map_or(Feature::Unsupported, |c| c.values[i - 50]),
        });
        self.score(cell.group, cell.evaluation_at, raw)
    }

    fn score(&self, group: Handle, end: u64, raw: [Feature; 54]) -> Option<CandidateHeads> {
        let selected = self
            .snapshot
            .groups
            .iter()
            .flatten()
            .find(|g| g.group == group)?;
        let local = LocalPrediction::new(&self.heads.config, &raw.map(Feature::value));
        let mut outputs = [None; 2];
        let mut errors = [None; 2];
        for (i, head) in [&self.heads.config.groove, &self.heads.config.desire]
            .into_iter()
            .enumerate()
        {
            let rows = self.snapshot.groups.map(|g| ratings::WeightedRating {
                weight: g.map_or(0., |g| g.acoustic_weight),
                log_probabilities: g.and_then(|g| {
                    if g.group == group {
                        local.log_probabilities[i]
                    } else {
                        g.local.log_probabilities[i]
                    }
                }),
            });
            let coverage = self.snapshot.observed_coverage;
            match ratings::project_rating(&rows, coverage, head.prior, head.temperature) {
                Some(p) => outputs[i] = Some(p.into()),
                None => errors[i] = Some("rating mixture or calibration overflow"),
            }
        }
        Some(CandidateHeads {
            scope: "conditional_acoustic_base_heads",
            issue_window: self.snapshot.window,
            evaluated_at: end,
            selected_group: group,
            selected_group_weight: selected.acoustic_weight,
            observed_inputs: raw
                .iter()
                .filter(|v| matches!(v, Feature::Observed(_)))
                .count(),
            projected_inputs: raw
                .iter()
                .filter(|v| matches!(v, Feature::Projected(_)))
                .count(),
            unsupported_inputs: raw
                .iter()
                .filter(|v| matches!(v, Feature::Unsupported))
                .count(),
            selected_prediction: local,
            groove: outputs[0],
            desire: outputs[1],
            errors,
        })
    }
}

#[cfg(test)]
mod tests;
