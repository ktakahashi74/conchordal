//! Explicit acoustic base heads, before any joint-path or candidate reweighting.

use super::Summary;
use crate::config::TemporalGrooveConfig;
use crate::temporal_cognition::{ratings, ridge::Handle};
use std::collections::VecDeque;

mod candidate;
pub(crate) use candidate::CandidateHeads;
pub(in crate::temporal_cognition) use candidate::Issue;

pub(crate) fn validate(config: &TemporalGrooveConfig) -> Result<(), &'static str> {
    if config.means.iter().any(|x| !x.is_finite())
        || config.deviations.iter().any(|x| !x.is_finite() || *x < 0.)
    {
        return Err("groove requires finite shared means and nonnegative finite deviations");
    }
    for head in [&config.groove, &config.desire] {
        if head
            .coefficients
            .iter()
            .chain(&head.cutpoints)
            .any(|x| !x.is_finite())
            || head.cutpoints.windows(2).any(|w| w[0] >= w[1])
            || head.prior.iter().any(|x| !x.is_finite() || *x <= 0.)
            || (head.prior.iter().sum::<f64>() - 1.).abs() > 1e-12
            || !head.temperature.is_finite()
            || head.temperature <= 0.
        {
            return Err(
                "groove/desire heads require finite coefficients, ordered cuts, positive normalized prior and positive finite temperature",
            );
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Span {
    start: u64,
    end: u64,
    known: bool,
}

pub(in crate::temporal_cognition) struct Heads {
    config: Box<TemporalGrooveConfig>,
    rate: u32,
    hop: u64,
    origin: u64,
    cut: u64,
    history: VecDeque<Span>,
    capacity: usize,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct GroupPrediction {
    pub group: Handle,
    pub acoustic_weight: f64,
    pub observed_inputs: usize,
    #[serde(flatten)]
    pub local: LocalPrediction,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct LocalPrediction {
    /// Head order is groove, desire; probabilities precede temperature and backoff.
    pub predictors: [Option<f64>; 2],
    pub log_probabilities: [Option<[f64; 5]>; 2],
    pub errors: [Option<&'static str>; 2],
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Rating {
    pub log_probabilities: [f64; 5],
    pub categories: [f64; 5],
    pub observed_coverage: f64,
    pub supported_mass: f64,
    pub reported_support_mass: f64,
    pub expected_rating: f64,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Snapshot {
    pub scope: &'static str,
    pub window: [u64; 2],
    pub observed_coverage: f64,
    pub groups: [Option<GroupPrediction>; 7],
    pub groove: Option<Rating>,
    pub desire: Option<Rating>,
    pub errors: [Option<&'static str>; 2],
    pub owned_bytes: usize,
}

impl Heads {
    pub(in crate::temporal_cognition) fn new(
        config: TemporalGrooveConfig,
        rate: u32,
        hop: u64,
        origin: u64,
    ) -> Result<Self, &'static str> {
        validate(&config)?;
        assert!(rate > 0 && hop > 0);
        let capacity = (8 * u64::from(rate)).div_ceil(hop) as usize + 1;
        Ok(Self {
            config: Box::new(config),
            rate,
            hop,
            origin,
            cut: origin,
            history: VecDeque::with_capacity(capacity),
            capacity,
        })
    }

    pub(in crate::temporal_cognition) fn advance(&mut self, end: u64, known: bool) {
        assert!(end > self.cut);
        let start = end.saturating_sub(self.hop).max(self.cut);
        self.cut = end;
        let window_start = end
            .saturating_sub(8 * u64::from(self.rate))
            .max(self.origin);
        while self.history.front().is_some_and(|s| s.end <= window_start) {
            self.history.pop_front();
        }
        if self.history.len() == self.capacity {
            self.history.pop_front();
        }
        self.history.push_back(Span { start, end, known });
    }

    // Keep rating scratch separate from the recurrence and observation stack frames.
    #[inline(never)]
    pub(in crate::temporal_cognition) fn score(
        &self,
        inputs: [Option<(Handle, &Summary)>; 7],
    ) -> Snapshot {
        let start = self
            .cut
            .saturating_sub(8 * u64::from(self.rate))
            .max(self.origin);
        let observed: u64 = self
            .history
            .iter()
            .filter(|s| s.known)
            .map(|s| s.end.saturating_sub(s.start.max(start)))
            .sum();
        let coverage = if self.cut > start {
            observed as f64 / (self.cut - start) as f64
        } else {
            0.
        };
        let inputs = inputs.map(|g| {
            g.filter(|(_, s)| s.end_sample == self.cut && s.assignment_sample_weight > 0.)
        });
        let total: f64 = inputs
            .iter()
            .flatten()
            .map(|(_, s)| s.assignment_sample_weight)
            .sum();
        let mut groups = [None; 7];
        for (index, input) in inputs.iter().enumerate() {
            let Some((group, raw)) = input else { continue };
            let result = GroupPrediction {
                group: *group,
                acoustic_weight: raw.assignment_sample_weight / total,
                observed_inputs: raw.raw.iter().flatten().count(),
                local: LocalPrediction::new(&self.config, &raw.raw),
            };
            groups[index] = Some(result);
        }
        let mut outputs = [None; 2];
        let mut errors = [None; 2];
        for (i, head) in [&self.config.groove, &self.config.desire]
            .into_iter()
            .enumerate()
        {
            let predictions = groups.map(|g| ratings::WeightedRating {
                weight: g.map_or(0., |g| g.acoustic_weight),
                log_probabilities: g.and_then(|g| g.local.log_probabilities[i]),
            });
            if let Some(p) =
                ratings::project_rating(&predictions, coverage, head.prior, head.temperature)
            {
                outputs[i] = Some(p.into());
            } else {
                errors[i] = Some("rating mixture or calibration overflow");
            }
        }
        Snapshot {
            scope: "acoustic_base_heads",
            window: [start, self.cut],
            observed_coverage: coverage,
            groups,
            groove: outputs[0],
            desire: outputs[1],
            errors,
            owned_bytes: std::mem::size_of::<Self>()
                + std::mem::size_of::<TemporalGrooveConfig>()
                + self.history.capacity() * std::mem::size_of::<Span>(),
        }
    }
}

impl LocalPrediction {
    fn new(config: &TemporalGrooveConfig, raw: &[Option<f64>; 54]) -> Self {
        let mut result = Self {
            predictors: [None; 2],
            log_probabilities: [None; 2],
            errors: [None; 2],
        };
        if raw.iter().all(Option::is_none) {
            return result;
        }
        let Some(coordinates) = ratings::groove_features(raw, &config.means, &config.deviations)
        else {
            result.errors = [Some("groove feature standardization overflow"); 2];
            return result;
        };
        for (i, head) in [&config.groove, &config.desire].into_iter().enumerate() {
            let eta: f64 = coordinates
                .iter()
                .zip(&head.coefficients)
                .map(|(x, b)| x * b)
                .sum();
            result.predictors[i] = eta.is_finite().then_some(eta);
            result.log_probabilities[i] = ratings::ordinal_log_probabilities(eta, head.cutpoints);
            if result.log_probabilities[i].is_none() {
                result.errors[i] = Some("ordinal predictor overflow");
            }
        }
        result
    }
}

impl From<ratings::RatingProjection> for Rating {
    fn from(p: ratings::RatingProjection) -> Self {
        let categories = p.log_probabilities.map(f64::exp);
        Self {
            log_probabilities: p.log_probabilities,
            categories,
            observed_coverage: p.observed_coverage,
            supported_mass: p.supported_mass,
            reported_support_mass: p.reported_support_mass,
            expected_rating: categories
                .iter()
                .enumerate()
                .map(|(j, p)| j as f64 * p / 4.)
                .sum(),
        }
    }
}

#[cfg(test)]
mod tests;
