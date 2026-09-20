//! Model-owned descriptors, query-wide search, and frozen MR1 comparison controls.

use super::matcher::{
    self, Anchor, AnchorDiagnostic, CAPACITY, CoarseConfig, Config, Error, Knot, Output,
};
use super::transport::{Identity, Relation};

mod index;

#[cfg(test)]
pub(super) const EPISODES: usize = 256;
pub(super) const MAX_EPISODES: usize = 16384;
pub(super) const CANDIDATES: usize = 16;
pub(super) const MAX_CANDIDATES: usize = 1024;
#[cfg(test)]
pub(super) const MATCHES: usize = CANDIDATES * 4;

#[cfg_attr(test, derive(Clone))]
pub(super) struct Descriptor {
    pub knots: Vec<Knot>,
    // Acquired local intervals stay attached to values before a permutation.
    #[cfg(test)]
    pub local_intervals: Vec<Option<f64>>,
}

#[cfg_attr(test, derive(Clone))]
pub(super) struct Episode {
    pub identity: Identity,
    pub epoch: u64,
    pub available_end: f64,
    pub first_observed_end: f64,
    pub scales: [f64; 10],
    pub descriptor: Descriptor,
}

pub(super) struct Query<'a> {
    pub descriptor: &'a Descriptor,
    pub epoch: u64,
    pub end: f64,
    pub observed_end: f64,
    pub scales: [f64; 10],
}

#[derive(Clone, Copy, Debug)]
pub(super) struct CoarseEntry {
    pub identity: Identity,
    pub cost: Option<f64>,
    pub similarity: Option<f64>,
    pub approximate: bool,
}

#[derive(Debug)]
pub(super) struct Match {
    pub relation: Relation,
    pub cost: Option<f64>,
    pub applied: [f64; 2],
    pub path: Vec<[i16; 3]>,
    pub anchor: Option<usize>,
    pub residuals: Option<Residuals>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub(crate) struct Residuals {
    pub coordinate_squared_error: [f64; 10],
    pub coordinate_count: [u32; 10],
    pub motion_squared_error: f64,
    pub motion_count: u32,
    pub interval_squared_error: f64,
    pub interval_count: u32,
    pub observed: u32,
    pub matched: u32,
    pub missing: u32,
    pub inserted: u32,
    pub deleted: u32,
}

pub(super) struct Report {
    pub search_covered: bool,
    pub coarse: Vec<CoarseEntry>,
    pub matches: Vec<Match>,
    pub cutoff_tie: bool,
    pub dp_cells: u64,
    pub pruned_candidates: usize,
    pub pruned_ties: usize,
}

#[derive(Clone, Copy)]
struct Candidate {
    episode: usize,
    anchor: usize,
    transform: Anchor,
}

#[cfg(test)]
pub(super) fn ordered(query: &Query<'_>, episodes: &[Episode]) -> Result<Report, Error> {
    ordered_index(query, episodes, true, CANDIDATES)
}

#[cfg(test)]
pub(super) fn ordered_prefix(query: &Query<'_>, episodes: &[Episode]) -> Result<Report, Error> {
    ordered_index(query, episodes, false, CANDIDATES)
}

pub(super) fn ordered_index(
    query: &Query<'_>,
    episodes: &[Episode],
    whole_query: bool,
    candidate_limit: usize,
) -> Result<Report, Error> {
    assert!((1..=MAX_CANDIDATES).contains(&candidate_limit));
    let mut report = Report {
        search_covered: false,
        coarse: Vec::with_capacity(episodes.len()),
        matches: Vec::with_capacity(candidate_limit * 4),
        cutoff_tie: false,
        dp_cells: 0,
        pruned_candidates: 0,
        pruned_ties: 0,
    };
    validate_query(query, episodes)?;
    let samples = index::samples(&query.descriptor.knots);
    let mut candidates = Vec::with_capacity(episodes.len());
    let mut anchors = [Anchor::default(); CAPACITY];
    let mut diagnostic = AnchorDiagnostic::default();
    let config = CoarseConfig {
        spacing: 4,
        limit: 32,
        bounds: [2.0; 2],
        grid: 1.0 / 16.0,
        scales: query.scales,
    };
    for (index, episode) in episodes.iter().enumerate() {
        if episode.epoch != query.epoch
            || episode.available_end > query.observed_end
            || episode.first_observed_end >= query.end
        {
            continue;
        }
        matcher::validate(&episode.descriptor.knots, episode.available_end)?;
        if episode
            .descriptor
            .knots
            .last()
            .is_some_and(|k| k.end > episode.first_observed_end)
        {
            return Err(Error::InvalidInput);
        }
        if episode.scales != query.scales {
            return Err(Error::InvalidInput);
        }
        matcher::coarse(
            &query.descriptor.knots,
            &episode.descriptor.knots,
            &config,
            &mut anchors,
            &mut diagnostic,
        )?;
        let selected = if whole_query {
            index::select(query, &episode.descriptor.knots, &anchors, &samples)?
        } else {
            (diagnostic.index >= 0).then(|| {
                (
                    diagnostic.index as usize * 4,
                    anchors[diagnostic.index as usize],
                )
            })
        };
        let best = selected.map(|(_, t)| t);
        report.coarse.push(CoarseEntry {
            identity: episode.identity,
            cost: best.map(|row| row.cost),
            similarity: best.map(|row| (-row.cost).exp()),
            approximate: episode
                .descriptor
                .knots
                .len()
                .div_ceil(config.spacing as usize)
                > config.limit as usize
                || diagnostic.bound_anchors > 0
                || best.is_none_or(|row| row.pitch_samples == 0 || row.interval_samples == 0),
        });
        if let Some((anchor, transform)) = selected {
            if !transform.cost.is_finite() {
                return Err(Error::NumericalRange);
            }
            candidates.push(Candidate {
                episode: index,
                anchor,
                transform,
            });
        }
    }
    candidates.sort_unstable_by(|a, b| {
        a.transform.cost.total_cmp(&b.transform.cost).then(
            episodes[a.episode]
                .identity
                .id
                .cmp(&episodes[b.episode].identity.id),
        )
    });
    assert!(candidate_limit > 0);
    report.cutoff_tie = candidates.len() > candidate_limit
        && candidates[candidate_limit - 1].transform.cost
            == candidates[candidate_limit].transform.cost;
    let mut retained = candidates.len().min(candidate_limit);
    if whole_query && report.cutoff_tie {
        let cost = candidates[candidate_limit].transform.cost;
        retained = candidates.partition_point(|c| c.transform.cost < cost);
        report.pruned_ties = candidates
            .iter()
            .filter(|c| c.transform.cost == cost)
            .count();
        // Keep complete score groups; no identity arbitrarily wins a boundary tie.
        report.cutoff_tie = false;
    }
    report.pruned_candidates = candidates.len() - retained;
    report.search_covered = !report.cutoff_tie
        && report.pruned_candidates == 0
        && report
            .coarse
            .iter()
            .all(|c| c.cost.is_some() && !c.approximate);
    let mut output = Output::default();
    for candidate in candidates.iter().take(retained) {
        let episode = &episodes[candidate.episode];
        let transform = candidate.transform;
        let choices = refinements(&transform);
        let mut trials = Vec::with_capacity(4);
        for pitch in &choices[0] {
            for tempo in &choices[1] {
                if pitch.abs() >= 2.0 || tempo.abs() >= 2.0 {
                    continue;
                }
                let mut total = 0.0;
                let mut valid = 0;
                for (a, b) in query
                    .descriptor
                    .knots
                    .iter()
                    .zip(episode.descriptor.knots[candidate.anchor..].iter())
                    .take(8)
                {
                    let mut residual = 0.0;
                    for d in 0..10 {
                        if a.mask & b.mask & (1 << d) != 0 {
                            let shift = if d == 0 { *pitch } else { 0.0 };
                            let delta = (a.values[d] - b.values[d] - shift) / query.scales[d];
                            residual += delta * delta;
                            valid += 1;
                        }
                    }
                    total += residual;
                }
                if valid > 0 {
                    let cost = total / valid as f64;
                    if !cost.is_finite() {
                        return Err(Error::NumericalRange);
                    }
                    trials.push((cost, [*pitch, *tempo]));
                }
            }
        }
        trials.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut target_covered = false;
        for (_, applied) in trials {
            let config = Config {
                anchor: candidate.anchor as u32,
                band: 16,
                shift: applied[0],
                ratio: 2.0_f64.powf(applied[1]),
                tempo_shift: applied[1],
                scales: query.scales,
                insertion: 1.0,
                deletion: 1.0,
            };
            matcher::dtw(
                &query.descriptor.knots,
                &episode.descriptor.knots,
                &config,
                &mut output,
            )?;
            let supported =
                output.total.is_finite() && output.observed > 0 && output.valid_coordinates > 0;
            target_covered |= supported && output.matched > 0 && output.band_edge == 0;
            report.dp_cells += u64::from(output.cells);
            report.matches.push(Match {
                relation: Relation {
                    identity: episode.identity,
                    supported,
                    ambiguous: report.cutoff_tie || output.band_edge != 0,
                    transformation: [
                        (transform.pitch_samples > 0).then_some(applied[0]),
                        (transform.interval_samples > 0).then_some(applied[1]),
                    ],
                },
                cost: supported.then(|| output.total / f64::from(output.observed)),
                applied,
                path: output.path[..output.path_len as usize].to_vec(),
                anchor: Some(candidate.anchor),
                residuals: Some(Residuals {
                    coordinate_squared_error: output.coordinate_error,
                    coordinate_count: output.coordinate_count,
                    motion_squared_error: output.motion_error[..output.motion_count as usize]
                        .iter()
                        .sum(),
                    motion_count: output.motion_count,
                    interval_squared_error: output.interval_error[..output.interval_count as usize]
                        .iter()
                        .sum(),
                    interval_count: output.interval_count,
                    observed: output.observed,
                    matched: output.matched,
                    missing: output.missing,
                    inserted: output.inserted,
                    deleted: output.deleted,
                }),
            });
        }
        report.search_covered &= target_covered;
    }
    report.matches.sort_unstable_by(|a, b| {
        a.cost
            .unwrap_or(f64::INFINITY)
            .total_cmp(&b.cost.unwrap_or(f64::INFINITY))
            .then(a.relation.identity.id.cmp(&b.relation.identity.id))
            .then(a.applied[0].total_cmp(&b.applied[0]))
            .then(a.applied[1].total_cmp(&b.applied[1]))
    });
    Ok(report)
}

// Both callers implement the registered four bracketing transforms.
fn refinements(transform: &Anchor) -> [Vec<f64>; 2] {
    std::array::from_fn(|d| {
        if [transform.pitch_samples, transform.interval_samples][d] == 0 {
            return vec![0.0];
        }
        let x = transform.unrounded[d] * 64.0;
        let lo = x.floor() / 64.0;
        let hi = x.ceil() / 64.0;
        if lo == hi { vec![lo] } else { vec![lo, hi] }
    })
}

fn validate_query(query: &Query<'_>, episodes: &[Episode]) -> Result<(), Error> {
    if !query.end.is_finite()
        || query.end > query.observed_end
        || episodes.len() > MAX_EPISODES
        || query.scales.iter().any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err(Error::InvalidInput);
    }
    matcher::validate(&query.descriptor.knots, query.observed_end)?;
    if query
        .descriptor
        .knots
        .last()
        .is_some_and(|k| k.end > query.end)
    {
        return Err(Error::InvalidInput);
    }
    for (i, episode) in episodes.iter().enumerate() {
        if episode.identity.id == 0
            || episodes[..i]
                .iter()
                .any(|other| other.identity.id == episode.identity.id)
            || !episode.available_end.is_finite()
            || !episode.first_observed_end.is_finite()
            || episode.first_observed_end > episode.available_end
        {
            return Err(Error::InvalidInput);
        }
    }
    Ok(())
}

#[derive(Debug)]
#[cfg(test)]
struct Bag {
    mean: [Option<f64>; 10],
    deviation: [f64; 10],
    medians: [Option<f64>; 2],
}

#[cfg(test)]
fn bag(descriptor: &Descriptor) -> Result<Bag, Error> {
    if descriptor.local_intervals.len() != descriptor.knots.len() {
        return Err(Error::InvalidInput);
    }
    let mut bag = Bag {
        mean: [None; 10],
        deviation: [0.0; 10],
        medians: [None; 2],
    };
    let mut values = Vec::with_capacity(CAPACITY);
    for d in 0..11 {
        values.clear();
        if d < 10 {
            values.extend(
                descriptor
                    .knots
                    .iter()
                    .filter(|k| k.mask & (1 << d) != 0)
                    .map(|k| k.values[d]),
            );
        } else {
            for value in descriptor.local_intervals.iter().flatten() {
                if !value.is_finite() || *value <= 0.0 {
                    return Err(Error::InvalidInput);
                }
                values.push(value.log2());
            }
        }
        if values.is_empty() {
            continue;
        }
        // Canonical content order makes floating-point bag sums permutation invariant.
        values.sort_unstable_by(f64::total_cmp);
        if d < 10 {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
            if !mean.is_finite() || !variance.is_finite() {
                return Err(Error::NumericalRange);
            }
            bag.mean[d] = Some(mean);
            bag.deviation[d] = variance.sqrt();
        }
        if d == 0 || d == 10 {
            let mid = values.len() / 2;
            bag.medians[usize::from(d == 10)] = Some(if values.len() % 2 == 1 {
                values[mid]
            } else {
                (values[mid - 1] + values[mid]) / 2.0
            });
        }
    }
    Ok(bag)
}

#[cfg(test)]
pub(super) fn orderless(query: &Query<'_>, episodes: &[Episode]) -> Result<Report, Error> {
    validate_query(query, episodes)?;
    let cue = bag(query.descriptor)?;
    let mut report = Report {
        search_covered: false,
        coarse: Vec::with_capacity(episodes.len()),
        matches: Vec::with_capacity(MATCHES),
        cutoff_tie: false,
        dp_cells: 0,
        pruned_candidates: 0,
        pruned_ties: 0,
    };
    let mut candidates = Vec::with_capacity(episodes.len());
    for episode in episodes {
        if episode.epoch != query.epoch
            || episode.available_end > query.observed_end
            || episode.first_observed_end >= query.end
        {
            continue;
        }
        matcher::validate(&episode.descriptor.knots, episode.available_end)?;
        if episode
            .descriptor
            .knots
            .last()
            .is_some_and(|k| k.end > episode.first_observed_end)
        {
            return Err(Error::InvalidInput);
        }
        let reference = bag(&episode.descriptor)?;
        if episode.scales != query.scales {
            return Err(Error::InvalidInput);
        }
        let mut transform = Anchor::default();
        for d in 0..2 {
            if let (Some(a), Some(b)) = (cue.medians[d], reference.medians[d]) {
                let value = if d == 0 { a - b } else { b - a };
                if !value.is_finite() {
                    return Err(Error::NumericalRange);
                }
                let floor = (value * 16.0).floor();
                let rounded = (if value * 16.0 - floor <= 0.5 {
                    floor
                } else {
                    floor + 1.0
                }) / 16.0;
                transform.unrounded[d] = value;
                transform.applied[d] = rounded;
                transform.bound_hit |= u32::from(value.abs() >= 2.0 || rounded.abs() >= 2.0);
                if d == 0 {
                    transform.pitch_samples = 1;
                } else {
                    transform.interval_samples = 1;
                }
            }
        }
        let cost = if transform.bound_hit != 0 {
            None
        } else {
            bag_cost(&cue, &reference, transform.applied[0], &query.scales)?
        };
        report.coarse.push(CoarseEntry {
            identity: episode.identity,
            cost,
            similarity: cost.map(|value| (-value).exp()),
            approximate: transform.bound_hit != 0
                || transform.pitch_samples == 0
                || transform.interval_samples == 0,
        });
        if let Some(cost) = cost {
            candidates.push((cost, episode.identity, transform, reference));
        }
    }
    candidates.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.id.cmp(&b.1.id)));
    report.cutoff_tie =
        candidates.len() > CANDIDATES && candidates[CANDIDATES - 1].0 == candidates[CANDIDATES].0;
    for (_, identity, transform, reference) in candidates.iter().take(CANDIDATES) {
        let choices = refinements(transform);
        for pitch in &choices[0] {
            for tempo in &choices[1] {
                if pitch.abs() >= 2.0 || tempo.abs() >= 2.0 {
                    continue;
                }
                let cost = bag_cost(&cue, reference, *pitch, &query.scales)?;
                report.matches.push(Match {
                    relation: Relation {
                        identity: *identity,
                        supported: cost.is_some(),
                        ambiguous: report.cutoff_tie,
                        transformation: [
                            (transform.pitch_samples > 0).then_some(*pitch),
                            (transform.interval_samples > 0).then_some(*tempo),
                        ],
                    },
                    cost,
                    applied: [*pitch, *tempo],
                    path: Vec::new(),
                    anchor: None,
                    residuals: None,
                });
            }
        }
    }
    report.matches.sort_unstable_by(|a, b| {
        a.cost
            .unwrap_or(f64::INFINITY)
            .total_cmp(&b.cost.unwrap_or(f64::INFINITY))
            .then(a.relation.identity.id.cmp(&b.relation.identity.id))
            .then(a.applied[0].total_cmp(&b.applied[0]))
            .then(a.applied[1].total_cmp(&b.applied[1]))
    });
    Ok(report)
}

#[cfg(test)]
fn bag_cost(
    cue: &Bag,
    reference: &Bag,
    shift: f64,
    scales: &[f64; 10],
) -> Result<Option<f64>, Error> {
    let mut total = 0.0;
    let mut count = 0;
    for (d, scale) in scales.iter().enumerate() {
        if let (Some(a), Some(b)) = (cue.mean[d], reference.mean[d]) {
            let delta = (a - b - if d == 0 { shift } else { 0.0 }) / scale;
            let deviation = (cue.deviation[d] - reference.deviation[d]) / scale;
            total += delta * delta + deviation * deviation;
            count += 2;
        }
    }
    if !total.is_finite() {
        return Err(Error::NumericalRange);
    }
    Ok((count > 0).then(|| total / count as f64))
}

#[cfg(test)]
mod reference;
