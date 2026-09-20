//! Expected-segment comparisons on immutable observed and conditional histories.

use super::*;

mod accents;

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct AccentDensity {
    pub(in crate::temporal_cognition) value: Feature,
    pub observed_samples: u64,
    pub projected_samples: u64,
    pub observed_weight: f64,
    pub projected_weight: f64,
    pub projected_accents: usize,
    pub latest_projected_interval: Option<[u64; 2]>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct ResidualProjection {
    pub(in crate::temporal_cognition) value: Feature,
    pub observed_samples: u64,
    pub projected_samples: u64,
    pub observed_mean: Option<f64>,
    pub projected_mean: Option<f64>,
    pub future_reference: Option<recall::Prediction>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct ConditionalOrdinal {
    pub path_index: usize,
    pub issue_mass: f64,
    pub foreground_start: Option<u64>,
    pub(in crate::temporal_cognition) survival: Feature,
    pub categories: Option<[f64; 5]>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct IssueGroupHead {
    pub group: Handle,
    pub acoustic_weight: f64,
    pub known_mass: f64,
    pub categories: [Option<[f64; 5]>; 2],
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct IssueHeads {
    pub issued_at: u64,
    pub bus: u8,
    pub epoch: u64,
    pub observed_coverage: f64,
    pub priors: [[f64; 5]; 2],
    pub groups: [Option<IssueGroupHead>; 7],
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct RawOrdinalMixture {
    pub categories: [f64; 5],
    pub expected_rating: f64,
    pub observed_coverage: f64,
    pub supported_mass: f64,
    pub reported_support_mass: f64,
    pub selected_supported_mass: f64,
    pub held_supported_mass: f64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct RawHeadMixture {
    pub issued_at: u64,
    pub selected_group_weight: f64,
    pub issue_known_mass: f64,
    pub closure: RawOrdinalMixture,
    pub continuation: RawOrdinalMixture,
}

impl Phrase {
    pub(in crate::temporal_cognition) fn issue_heads(&self) -> IssueHeads {
        IssueHeads {
            issued_at: self.end,
            bus: self.bus,
            epoch: self.epoch,
            observed_coverage: self.snapshot.closure.observed_coverage,
            priors: [self.config.closure.prior, self.config.continuation.prior],
            groups: self.snapshot.groups.map(|group| {
                group.map(|g| IssueGroupHead {
                    group: g.group,
                    acoustic_weight: g.acoustic_weight,
                    known_mass: 1. - g.unknown,
                    categories: [g.closure, g.continuation],
                })
            }),
        }
    }
}

impl IssueHeads {
    pub(in crate::temporal_cognition) fn project(
        &self,
        cell: &super::super::action_profiles::Cell,
    ) -> Option<RawHeadMixture> {
        use super::super::ratings::{WeightedRating, project_rating};
        if cell.issued_at != self.issued_at
            || cell.group.bus != self.bus
            || cell.group.epoch != self.epoch
        {
            return None;
        }
        let selected = self
            .groups
            .iter()
            .flatten()
            .find(|g| g.group == cell.group)?;
        let path_mass: f64 = cell
            .continuation_paths
            .iter()
            .flatten()
            .map(|p| p.issue_mass)
            .sum();
        if (path_mass - selected.known_mass).abs() > 1e-12
            || (path_mass + cell.issue_phrase_unknown - 1.).abs() > 1e-12
        {
            return None;
        }
        // Raw base-head aggregation only; no invented shared posterior or score delta.
        let mut heads = [None; 2];
        for (head, output) in heads.iter_mut().enumerate() {
            let rows: [WeightedRating; 7 + PATHS] = std::array::from_fn(|index| {
                if index < 7 {
                    let held = self.groups[index].filter(|g| g.group != cell.group);
                    WeightedRating {
                        weight: held.map_or(0., |g| g.acoustic_weight * g.known_mass),
                        log_probabilities: held
                            .and_then(|g| g.categories[head].map(|p| p.map(f64::ln))),
                    }
                } else {
                    let path = cell.continuation_paths[index - 7];
                    WeightedRating {
                        weight: path.map_or(0., |p| selected.acoustic_weight * p.issue_mass),
                        log_probabilities: path.and_then(|p| {
                            if head == 0 {
                                cell.closure
                            } else {
                                p.categories
                            }
                            .map(|p| p.map(f64::ln))
                        }),
                    }
                }
            });
            let projection = project_rating(&rows, self.observed_coverage, self.priors[head], 1.)?;
            let categories = projection.log_probabilities.map(f64::exp);
            *output = Some(RawOrdinalMixture {
                categories,
                expected_rating: categories
                    .iter()
                    .enumerate()
                    .map(|(i, p)| i as f64 / 4. * p)
                    .sum(),
                observed_coverage: projection.observed_coverage,
                supported_mass: projection.supported_mass,
                reported_support_mass: projection.reported_support_mass,
                selected_supported_mass: rows[7..]
                    .iter()
                    .filter(|r| r.log_probabilities.is_some())
                    .map(|r| r.weight)
                    .sum(),
                held_supported_mass: rows[..7]
                    .iter()
                    .filter(|r| r.log_probabilities.is_some())
                    .map(|r| r.weight)
                    .sum(),
            });
        }
        Some(RawHeadMixture {
            issued_at: self.issued_at,
            selected_group_weight: selected.acoustic_weight,
            issue_known_mass: selected.known_mass,
            closure: heads[0]?,
            continuation: heads[1]?,
        })
    }
}

impl Group {
    pub(super) fn accent_density(
        &self,
        start: u64,
        end: u64,
        issued_at: u64,
        rate: u32,
    ) -> Option<f64> {
        let samples: u64 = self
            .history
            .iter()
            .filter(|s| s.raw.available_end <= issued_at)
            .map(|s| s.raw.end.min(end).saturating_sub(s.raw.start.max(start)))
            .sum();
        (samples > 0
            && samples as f64 >= 0.9 * (end - start) as f64
            && self.evicted_accent.is_none_or(|t| t < start))
        .then(|| {
            self.accents
                .iter()
                .filter(|a| a.event_end >= start && a.event_end <= end && a.at_cut(issued_at))
                .map(|a| a.weight)
                .sum::<f64>()
                * f64::from(rate)
                / samples as f64
        })
    }

    pub(super) fn projected_residual(
        &self,
        start: u64,
        end: u64,
        issued_at: u64,
        frames: impl Iterator<Item = window::Frame>,
    ) -> Result<ResidualProjection, &'static str> {
        if start > end || end < issued_at {
            return Err("invalid residual projection window");
        }
        let mut sum = 0.;
        let mut observed_samples = 0;
        let mut projected_samples = 0;
        for sample in &self.history {
            let Some(value) = sample.residual else {
                continue;
            };
            let n = sample
                .residual_interval
                .1
                .min(end)
                .saturating_sub(sample.residual_interval.0.max(start));
            if n == 0 {
                continue;
            }
            if sample.raw.end > issued_at
                || sample.raw.source_end < sample.raw.end
                || sample.raw.source_end > sample.raw.available_end
                || sample.raw.available_end > issued_at
                || sample.residual_interval.0 < sample.raw.start
                || sample.residual_interval.1 > sample.raw.end
                || !value.is_finite()
                || value < 0.
            {
                return Err("noncausal residual prefix");
            }
            sum += value * n as f64;
            observed_samples += n;
        }
        let reference = self.cue.filter(|cue| {
            cue.group == self.handle
                && cue.issued_at <= issued_at
                && cue.source_start <= cue.source_end
                && cue.source_end <= cue.available
                && cue.available <= cue.issued_at
                && cue.expected_start < cue.expected_end
        });
        let observed_sum = sum;
        let mut projected_sum = 0.;
        if let Some(cue) = reference {
            let mut previous_end = issued_at;
            for frame in frames {
                if frame.start >= end {
                    break;
                }
                if frame.start < previous_end || frame.start >= frame.end {
                    return Err("unordered residual projection frames");
                }
                previous_end = frame.end;
                let lo = start.max(frame.start).max(cue.expected_start);
                let hi = end.min(frame.end).min(cue.expected_end);
                if lo >= hi {
                    continue;
                }
                if frame.raw.iter().any(|v| matches!(v, Feature::Observed(_))) {
                    return Err("future residual input labeled observed");
                }
                if let Some(value) = cue.normalized_residual(frame.raw.map(Feature::value)) {
                    projected_sum += value * (hi - lo) as f64;
                    projected_samples += hi - lo;
                }
            }
        }
        let count = observed_samples + projected_samples;
        let sum = observed_sum + projected_sum;
        let value = if count == 0 {
            Feature::Unsupported
        } else {
            let value = sum / count as f64;
            if !value.is_finite() {
                return Err("residual window accumulation overflow");
            }
            if projected_samples > 0 {
                Feature::Projected(value)
            } else {
                Feature::Observed(value)
            }
        };
        Ok(ResidualProjection {
            value,
            observed_samples,
            projected_samples,
            observed_mean: (observed_samples > 0).then(|| observed_sum / observed_samples as f64),
            projected_mean: (projected_samples > 0)
                .then(|| projected_sum / projected_samples as f64),
            future_reference: reference.filter(|_| projected_samples > 0),
        })
    }
}

#[cfg(test)]
mod tests;
