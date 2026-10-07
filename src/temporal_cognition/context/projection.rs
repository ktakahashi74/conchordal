//! Hypothetical candidate windows over the immutable observed body context.

use super::*;
use crate::temporal_cognition::feature_projection::{Feature, window};
use crate::temporal_cognition::gesture;

impl Group {
    pub(super) fn frames(&self) -> impl DoubleEndedIterator<Item = window::Frame> + '_ {
        self.history.iter().map(|s| window::Frame {
            start: s.raw.start,
            end: s.raw.end,
            source_end: s.raw.source_end,
            available: s.raw.available_end,
            raw: s
                .raw
                .values
                .map(|v| v.map_or(Feature::Unsupported, Feature::Observed)),
            energy: s.energy.map_or(Feature::Unsupported, Feature::Observed),
        })
    }

    fn grouping_window(&self, start: u64, end: u64, available: u64) -> (Option<f64>, f64) {
        let mut sum = 0.;
        let mut weight = 0.;
        let mut known = 0_u64;
        for sample in &self.history {
            if sample.raw.available_end > available {
                continue;
            }
            if let Some(value) = sample.grouping {
                let n = sample
                    .raw
                    .end
                    .min(end)
                    .saturating_sub(sample.raw.start.max(start));
                sum += n as f64 * sample.alpha * value;
                weight += n as f64 * sample.alpha;
                known += n;
            }
        }
        let coverage = if end > start {
            known as f64 / (end - start) as f64
        } else {
            0.
        };
        (
            (weight > 0. && known as f64 >= 0.9 * (end - start) as f64).then(|| sum / weight),
            coverage,
        )
    }
}

impl Context {
    /// Project one candidate action window from the observed prefix and a conditional body
    /// profile. Hypothetical frames never enter the observed history.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::temporal_cognition) fn project_action_window(
        &self,
        profiles: &crate::temporal_cognition::action_profiles::Profiles,
        prototype: usize,
        class: crate::life::action_candidates::Class,
        action_offset: u64,
        evaluation_at: u64,
        handle: Handle,
        rms_reference: f64,
        articulation: Option<&gesture::Gesture>,
        arrival_issue: Option<&crate::temporal_cognition::arrival::Frozen>,
        projection_cache: Option<(usize, &mut gesture::ProjectionCache)>,
    ) -> Option<crate::temporal_cognition::action_profiles::Cell> {
        if self.rate != 48_000
            || self.hop != 512
            || handle.bus != self.bus
            || handle.epoch != self.epoch
            || evaluation_at < self.end
            || evaluation_at > self.end.checked_add(192_000)?
        {
            return None;
        }
        let group = self.groups.iter().flatten().find(|g| g.handle == handle)?;
        let previous = group.frames().next_back().filter(|f| {
            f.start < f.end
                && f.end == self.end
                && f.source_end >= f.end
                && f.source_end <= f.available
                && f.available <= self.end
        });
        let background = previous
            .and_then(|f| f.energy.value().zip(f.raw[6].value()))
            .and_then(|(energy, share)| {
                (share > 0. && share <= 1.).then(|| (energy / share - energy).max(0.))
            })
            .filter(|value| value.is_finite());
        let frames = || {
            profiles.frames(
                prototype,
                class,
                action_offset,
                self.end,
                previous,
                background,
            )
        };
        let _ = frames()?;
        let start = group
            .born
            .max(evaluation_at.saturating_sub(u64::from(self.rate) * 2));
        // Profiles have validated, issue-relative 512-sample frames. Keep both clipped edges.
        let projected_end = (evaluation_at - self.end).div_ceil(self.hop) as usize;
        let short_start = group
            .born
            .max(evaluation_at.saturating_sub(u64::from(self.rate) / 4));
        let features = window::summarize(
            start,
            evaluation_at,
            self.end,
            self.rate,
            rms_reference,
            group.frames().chain(
                frames()?
                    .take(projected_end)
                    .skip((start.saturating_sub(self.end) / self.hop) as usize),
            ),
        )
        .ok()?;
        let short = window::summarize(
            short_start,
            evaluation_at,
            self.end,
            self.rate,
            rms_reference,
            group.frames().chain(
                frames()?
                    .take(projected_end)
                    .skip((short_start.saturating_sub(self.end) / self.hop) as usize),
            ),
        )
        .ok()?;
        let mut short_features = [Feature::Unsupported; 4];
        short_features.copy_from_slice(&short.values[..4]);
        let (grouping, grouping_observed_fraction) =
            group.grouping_window(start, evaluation_at, self.end);
        let grouping = grouping.map_or(Feature::Unsupported, Feature::Observed);
        let articulation = articulation.and_then(|model| {
            model.project_states(handle, self.end, evaluation_at, frames()?, projection_cache)
        });
        let mut arrival = arrival_issue
            .filter(|f| f.group == handle && f.issued_at == self.end && f.sample_rate == self.rate)
            .and_then(|f| f.project(evaluation_at));
        let accent_density = group
            .projected_accent_density(
                [start, evaluation_at],
                self.end,
                self.rate,
                (profiles.accent_means, profiles.accent_deviations),
                arrival.as_mut(),
                frames()?,
            )
            .ok()?;
        let arrival = arrival.and_then(|a| a.finish());
        let context_features = [
            if evaluation_at == self.end {
                group
                    .arrival_probability
                    .map_or(Feature::Unsupported, Feature::Observed)
            } else {
                arrival
                    .filter(|a| a.horizon_end - a.evaluated_at == u64::from(self.rate))
                    .map_or(Feature::Unsupported, |a| a.probability)
            },
            accent_density.value,
            grouping,
        ];
        Some(crate::temporal_cognition::action_profiles::Cell {
            prototype,
            class,
            group: handle,
            issued_at: self.end,
            action_at: self.end.checked_add(action_offset)?,
            evaluation_at,
            window_start: start,
            held_background_energy: background,
            issue_log_rms: previous
                .and_then(|f| f.raw[2].value())
                .filter(|v| v.is_finite()),
            features,
            articulation,
            short_features,
            short_observed_fraction: std::array::from_fn(|i| short.observed_fraction[i]),
            short_projected_fraction: std::array::from_fn(|i| short.projected_fraction[i]),
            grouping,
            grouping_observed_fraction,
            arrival,
            accent_density,
            context_features,
        })
    }
}
