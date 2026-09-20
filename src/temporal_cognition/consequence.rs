//! Numerical candidate previews. Inputs are frozen hypotheses, never observed updates.

#[cfg(test)]
use super::body_model::{Assignment, Descriptor, nearest};

use super::ratings::{RatingProjection, WeightedRating, project_rating};

const ALTERNATIVES: usize = 128;

#[derive(Clone, Copy)]
pub(super) struct Alternative {
    pub weight: f64,
    pub delta_log_score: Option<f64>,
    pub heads: [Option<WeightedRating>; 3],
}

pub(super) struct Projection {
    pub conditional_weights: [f64; ALTERNATIVES],
    pub retained_mass: f64,
    pub unknown_mass: f64,
    pub original_unknown_mass: f64,
    pub unsupported_projection_mass: f64,
    pub resolved_count: usize,
    pub entropy: f64,
    pub uncertainty: f64,
    pub heads: [RatingProjection; 3],
}

pub(super) fn project(
    alternatives: &[Alternative],
    issue_coverage: [f64; 3],
    priors: [[f64; 5]; 3],
    temperatures: [f64; 3],
) -> Option<Projection> {
    if alternatives.len() > ALTERNATIVES
        || alternatives.iter().any(|row| {
            !row.weight.is_finite()
                || !(0.0..=1.0).contains(&row.weight)
                || row.delta_log_score.is_some_and(|score| !score.is_finite())
                || row.heads.iter().flatten().any(|head| {
                    !head.weight.is_finite()
                        || !(0.0..=1.0).contains(&head.weight)
                        || head.log_probabilities.is_some_and(|p| {
                            p.iter().any(|p| p.is_nan() || *p > 0.)
                                || (p.iter().map(|p| p.exp()).sum::<f64>() - 1.).abs() > 1e-12
                        })
                })
        })
        || alternatives.iter().map(|row| row.weight).sum::<f64>() > 1. + 1e-12
    {
        return None;
    }
    let original_mass = alternatives
        .iter()
        .map(|row| row.weight)
        .sum::<f64>()
        .min(1.);
    let mut log_weights = [f64::NEG_INFINITY; ALTERNATIVES];
    let mut retained_mass = 0.;
    let mut resolved_count = 0;
    let maximum_delta = alternatives
        .iter()
        .filter(|row| row.weight > 0.)
        .filter_map(|row| row.delta_log_score)
        .fold(f64::NEG_INFINITY, f64::max);
    for (index, row) in alternatives.iter().enumerate() {
        if let Some(delta) = row.delta_log_score.filter(|_| row.weight > 0.) {
            // Subtract the common score before adding log(q); a large offset must not erase q.
            log_weights[index] = row.weight.ln() + (delta - maximum_delta);
            retained_mass += row.weight;
            resolved_count += 1;
        }
    }
    // Roundoff in a normalized joint measure cannot create negative unknown mass.
    retained_mass = retained_mass.min(1.);
    let unknown_mass = 1. - retained_mass;
    let mut conditional_weights = [0.; ALTERNATIVES];
    let mut entropy = 0.;
    if resolved_count > 0 {
        let (pivot, maximum) = log_weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap();
        let remainder: f64 = log_weights
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != pivot)
            .map(|(_, value)| (value - maximum).exp())
            .sum();
        let log_sum = remainder.ln_1p();
        for (index, weight) in conditional_weights.iter_mut().enumerate() {
            let log_p = (log_weights[index] - maximum) - log_sum;
            *weight = log_p.exp();
            if *weight > 0. {
                entropy -= *weight * log_p;
            }
        }
    }
    let uncertainty = (unknown_mass
        + if resolved_count > 1 {
            retained_mass * entropy / (resolved_count as f64).ln()
        } else {
            0.
        })
    .clamp(0., 1.);
    let mut heads = [None, None, None];
    for head in 0..3 {
        let rows: [WeightedRating; ALTERNATIVES] = std::array::from_fn(|index| {
            let mixture = alternatives.get(index).and_then(|row| row.heads[head]);
            WeightedRating {
                weight: retained_mass
                    * conditional_weights[index]
                    * mixture.map_or(0., |m| m.weight),
                log_probabilities: mixture.and_then(|m| m.log_probabilities),
            }
        });
        heads[head] = Some(project_rating(
            &rows[..alternatives.len()],
            issue_coverage[head],
            priors[head],
            temperatures[head],
        )?);
    }
    Some(Projection {
        conditional_weights,
        retained_mass,
        unknown_mass,
        original_unknown_mass: 1. - original_mass,
        unsupported_projection_mass: (original_mass - retained_mass).max(0.),
        resolved_count,
        entropy,
        uncertainty,
        heads: heads.map(Option::unwrap),
    })
}

/// Uniform absolute-sample grid, including both ends. Midpoint ties choose earlier.
pub(super) fn cell_index(origin: u64, end: u64, cells: usize, target: u64) -> Option<usize> {
    if cells < 2 || end <= origin || target < origin || target > end {
        return None;
    }
    let numerator = u128::from(target - origin) * (cells - 1) as u128;
    let denominator = u128::from(end - origin);
    let lower = numerator / denominator;
    let later = numerator % denominator > denominator / 2;
    Some(lower as usize + usize::from(later))
}

pub(super) struct Absolute {
    pub relation: Option<f64>,
    pub ratings: [Option<f64>; 3],
    pub uncertainty: Option<f64>,
}

impl Projection {
    pub(super) fn absolute(&self, relation: Option<f64>) -> Absolute {
        Absolute {
            relation,
            ratings: std::array::from_fn(|i| {
                let head = &self.heads[i];
                (head.reported_support_mass > 0.).then(|| {
                    head.log_probabilities
                        .iter()
                        .enumerate()
                        .map(|(category, p)| category as f64 / 4. * p.exp())
                        .sum()
                })
            }),
            uncertainty: (self.retained_mass > 0.).then_some(self.uncertainty),
        }
    }
}

#[derive(Debug, PartialEq)]
pub(super) struct Differences {
    pub relation: Option<f64>,
    pub ratings: [Option<f64>; 3],
    pub uncertainty: Option<f64>,
}

pub(super) fn differences(
    candidate: &Absolute,
    body_default: &Absolute,
    relation_scale: f64,
) -> Option<Differences> {
    if !relation_scale.is_finite()
        || relation_scale < 0.
        || [candidate, body_default].into_iter().any(|value| {
            value.relation.is_some_and(|v| !v.is_finite())
                || value
                    .ratings
                    .iter()
                    .chain(std::iter::once(&value.uncertainty))
                    .flatten()
                    .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        })
    {
        return None;
    }
    Some(Differences {
        relation: candidate
            .relation
            .zip(body_default.relation)
            .map(|(a, b)| ((a - b) / relation_scale.max(1e-6)).tanh()),
        ratings: std::array::from_fn(|i| {
            candidate.ratings[i]
                .zip(body_default.ratings[i])
                .map(|(a, b)| a - b)
        }),
        uncertainty: candidate
            .uncertainty
            .zip(body_default.uncertainty)
            .map(|(a, b)| a - b),
    })
}

#[cfg(test)]
mod tests;
