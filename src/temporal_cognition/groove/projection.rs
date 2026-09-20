//! Conditional density inputs preserve acoustic and hypothetical support separately.

use super::{Estimator, Group, WINDOWS};
use crate::temporal_cognition::feature_projection::Feature;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct DensityProjection {
    pub issued_at: u64,
    pub evaluated_at: u64,
    pub starts: [u64; 8],
    pub(in crate::temporal_cognition) values: [Feature; 8],
    pub observed_samples: [u64; 8],
    pub projected_samples: [u64; 8],
    pub observed_weight: [f64; 8],
    pub projected_weight: [f64; 8],
    pub retention_supported: [bool; 8],
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct ContextProjection {
    pub issued_at: u64,
    pub evaluated_at: u64,
    pub window_start: u64,
    /// Only the original observed prefix supplies these four inputs.
    pub(in crate::temporal_cognition) values: [Feature; 4],
    pub word_coverage: f64,
    pub grouping_coverage: f64,
    pub word_probabilities: Option<[f64; 9]>,
    pub retained_pairs: usize,
    pub surprise_capacity_supported: bool,
}

impl Group {
    pub(in crate::temporal_cognition) fn project_context(
        &self,
        estimator: &Estimator,
        end: u64,
    ) -> Option<ContextProjection> {
        if end < self.last_end || end - self.last_end > 4 * u64::from(self.rate) {
            return None;
        }
        let summary = self.summarize(estimator, end);
        Some(ContextProjection {
            issued_at: self.last_end,
            evaluated_at: end,
            window_start: end
                .saturating_sub(8 * u64::from(self.rate))
                .max(self.origin),
            values: std::array::from_fn(|i| {
                summary.raw[50 + i].map_or(Feature::Unsupported, Feature::Observed)
            }),
            word_coverage: summary.word_coverage,
            grouping_coverage: summary.grouping_coverage,
            word_probabilities: summary.word_probabilities,
            retained_pairs: summary.retained_pairs,
            surprise_capacity_supported: summary.surprise_capacity_supported,
        })
    }

    pub(in crate::temporal_cognition) fn project_density(
        &self,
        estimator: &Estimator,
        end: u64,
    ) -> Option<DensityProjection> {
        if end < self.last_end {
            return None;
        }
        let starts = WINDOWS.map(|seconds| {
            end.saturating_sub((seconds * f64::from(self.rate)).round() as u64)
                .max(self.origin)
        });
        let mut result = DensityProjection {
            issued_at: self.last_end,
            evaluated_at: end,
            starts,
            values: [Feature::Unsupported; 8],
            observed_samples: [0; 8],
            projected_samples: [0; 8],
            observed_weight: [0.; 8],
            projected_weight: [0.; 8],
            retention_supported: [false; 8],
        };
        for frame in self.frames.iter().filter(|f| f.known) {
            for (i, start) in starts.into_iter().enumerate() {
                result.observed_samples[i] += frame.end.saturating_sub(frame.start.max(start));
            }
        }
        for (i, start) in starts.into_iter().enumerate() {
            let weight = if start > self.last_end {
                Some(0.)
            } else {
                estimator.accent_weight(start, self.last_end)
            };
            result.retention_supported[i] = weight.is_some();
            result.observed_weight[i] = weight.unwrap_or(0.);
        }
        Some(result)
    }
}

impl DensityProjection {
    pub(in crate::temporal_cognition) fn receipt(
        &mut self,
        interval: [u64; 2],
        weight: Option<f64>,
    ) {
        for i in 0..8 {
            self.projected_samples[i] += interval[1]
                .min(self.evaluated_at)
                .saturating_sub(interval[0].max(self.starts[i]).max(self.issued_at));
            // A center ending at issue may need a hypothetical right neighbor for admission.
            if interval[1] >= self.starts[i] && interval[1] <= self.evaluated_at {
                self.projected_weight[i] += weight.unwrap_or(0.);
            }
        }
    }

    pub(in crate::temporal_cognition) fn finish(&mut self, rate: u32) {
        for i in 0..8 {
            let samples = self.observed_samples[i] + self.projected_samples[i];
            if self.retention_supported[i]
                && samples > 0
                && samples as f64 >= 0.9 * (self.evaluated_at - self.starts[i]) as f64
            {
                let density = (self.observed_weight[i] + self.projected_weight[i])
                    * f64::from(rate)
                    / samples as f64;
                self.values[i] = if self.projected_samples[i] > 0 || self.projected_weight[i] > 0. {
                    Feature::Projected(density)
                } else {
                    Feature::Observed(density)
                };
            }
        }
    }
}
