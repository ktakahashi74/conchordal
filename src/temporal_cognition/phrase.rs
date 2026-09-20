//! Bounded base phrase hypotheses and independent, uncalibrated ordinal diagnostics.

use super::feature_projection::{Feature, window};
use super::observables::WindowDescriptor;
use super::{
    features::RawDescriptor, gesture, hazard::integrate, proposals::frontend, recall, ridge::Handle,
};
use crate::config::{TemporalOrdinalConfig, TemporalPhraseConfig};
use serde::Serialize;
use std::collections::VecDeque;

pub(in crate::temporal_cognition) mod interpretation;
use interpretation::Interpretation;

mod projection;
#[cfg(test)]
pub(crate) use projection::RawOrdinalMixture;
pub(crate) use projection::{
    AccentDensity, ConditionalOrdinal, IssueHeads, RawHeadMixture, ResidualProjection,
};

const PATHS: usize = 15;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Exit {
    New,
    Overlap,
    Reinterpret,
    Inactive,
}
const EXITS: [Exit; 4] = [Exit::New, Exit::Overlap, Exit::Reinterpret, Exit::Inactive];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct Foreground {
    pub id: u64,
    // Shared by reinterpreted ongoing prefixes; not a unique sealed-endpoint identity.
    pub credit: u64,
    pub start: u64,
    pub heard_end: u64,
}
#[derive(Clone, Copy)]
struct Path {
    parent_index: usize,
    state: Interpretation,
    mass: f64,
}
#[derive(Clone, Copy)]
struct Proposal {
    parent: usize,
    kind: Option<Exit>,
    mass: f64,
}
#[derive(Clone, Copy)]
struct Sample {
    raw: RawDescriptor,
    energy: Option<f64>,
    spectral_shape_supported: bool,
    alpha: f64,
    grouping: Option<f64>,
    residual: Option<f64>,
    residual_interval: (u64, u64),
}
struct Group {
    handle: Handle,
    born: u64,
    end: u64,
    paths: Vec<Path>,
    scratch: Vec<Path>,
    proposals: Vec<Proposal>,
    unknown: f64,
    history: VecDeque<Sample>,
    context: [Option<f64>; 12],
    arrival_probability: Option<f64>,
    states: [f64; 5],
    query: Option<recall::ResultSnapshot>,
    cue: Option<recall::Prediction>,
    cue_id: u64,
    accents: VecDeque<super::features::Accent>,
    last_accent: Option<u64>,
    evicted_accent: Option<u64>,
    snapshot: GroupSnapshot,
}
#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Candidate {
    pub parent_index: usize,
    pub completed_foreground: Option<Foreground>,
    pub completed_ending: Option<WindowDescriptor>,
    pub foreground: Option<Foreground>,
    pub mass: f64,
    pub event: Option<(Exit, u64)>,
    pub unresolved_links: usize,
    pub right_censored_links: usize,
    pub oldest_link_start: Option<u64>,
    pub lost_links: u64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Ordinal {
    pub categories: [f64; 5],
    pub expected_rating: f64,
    pub supported_expected_rating: Option<f64>,
    pub supported_mass: f64,
    pub observed_coverage: f64,
    pub unknown: f64,
}
#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Forecast {
    pub issued_at: u64,
    pub horizon_end: u64,
    pub source_start: u64,
    pub source_end: u64,
    pub available: u64,
    pub survival: f64,
    pub exits: [f64; 4],
    pub unknown: f64,
}
#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct GroupSnapshot {
    pub group: Handle,
    pub previous_end_sample: u64,
    pub acoustic_weight: f64,
    pub recent_energy: Option<RecentEnergy>,
    pub candidates: [Option<Candidate>; PATHS],
    pub unknown: f64,
    pub forecast: Option<Forecast>,
    pub phrase_values: [Option<f64>; 12],
    pub articulation_mass: [f64; 5],
    pub closure_values: [Option<f64>; 10],
    pub closure: Option<[f64; 5]>,
    pub continuation: Option<[f64; 5]>,
    pub cue: Option<recall::Prediction>,
    pub prediction_comparisons: u64,
    pub source_start: u64,
    pub source_end: u64,
    pub available: u64,
}
#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Snapshot {
    pub version: u64,
    pub epoch_start_sample: u64,
    pub sample_rate: u32,
    pub end_sample: u64,
    pub groups: [Option<GroupSnapshot>; 7],
    pub closure: Ordinal,
    pub continuation: Ordinal,
    pub censored: bool,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct RecentEnergy {
    pub start_sample: u64,
    pub end_sample: u64,
    pub values: [Option<f64>; 3],
    pub coverage: [f64; 3],
}
pub(crate) struct Phrase {
    config: TemporalPhraseConfig,
    bus: u8,
    epoch: u64,
    rate: u32,
    hop: u64,
    start: u64,
    end: u64,
    next_span: u64,
    groups: [Option<Group>; 7],
    observed: VecDeque<(u64, u64)>,
    snapshot: Snapshot,
}

fn ordinal(values: &[Option<f64>; 14], cfg: TemporalOrdinalConfig) -> Option<[f64; 5]> {
    if values.iter().all(Option::is_none) {
        return None;
    }
    let mut z = cfg.coefficients[0];
    for (i, value) in values.iter().enumerate() {
        z += value.map_or(0., |v| (v - cfg.means[i]) / cfg.deviations[i].max(1e-6))
            * cfg.coefficients[1 + i]
            + f64::from(value.is_none()) * cfg.coefficients[15 + i];
    }
    if !z.is_finite() {
        return None;
    }
    let mut last = 0.;
    let mut probabilities = [0.; 5];
    for (i, cut) in cfg.cutpoints.iter().enumerate() {
        let x = *cut - z;
        let cdf = if x >= 0. {
            1. / (1. + (-x).exp())
        } else {
            x.exp() / (1. + x.exp())
        };
        probabilities[i] = cdf - last;
        last = cdf;
    }
    probabilities[4] = 1. - last;
    Some(probabilities)
}

fn vector(
    values: [Option<f64>; 12],
    state: usize,
    elapsed: f64,
    cfg: TemporalPhraseConfig,
) -> [f64; 26] {
    let mut x = [0.; 26];
    x[0] = 1.;
    x[1] = elapsed.ln_1p();
    for i in 0..12 {
        let v = if i < 4 {
            (state < 4).then_some(f64::from(state == i))
        } else {
            values[i]
        };
        x[2 + i] = v.map_or(0., |v| (v - cfg.means[i]) / cfg.deviations[i].max(1e-6));
        x[14 + i] = f64::from(v.is_none());
    }
    x
}

pub(super) struct ConditionalLaw {
    pub survival: f64,
    // Keep the unnormalized scores so ordinary mixtures preserve multiplication order.
    pub exits: Option<([f64; 4], f64)>,
}

pub(super) fn conditional_law<const WITH_EXITS: bool>(
    values: [Option<f64>; 12],
    state: Option<gesture::State>,
    [lo, hi]: [f64; 2],
    cfg: TemporalPhraseConfig,
) -> Option<ConditionalLaw> {
    let mut x = vector(values, state.map_or(4, |s| s as usize), 0., cfg);
    let intercept = x.iter().zip(cfg.hazard).map(|(a, b)| a * b).sum();
    let h = integrate(intercept, cfg.hazard[1], lo, hi)?;
    let survival = (-h).exp();
    let exits = if WITH_EXITS {
        x[1] = hi.ln_1p();
        let logits = cfg
            .exits
            .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum::<f64>());
        if logits.iter().any(|v| !v.is_finite()) {
            None
        } else {
            let max = logits.into_iter().fold(f64::NEG_INFINITY, f64::max);
            let scores = logits.map(|v| (v - max).exp());
            Some((scores, scores.iter().sum()))
        }
    } else {
        None
    };
    Some(ConditionalLaw { survival, exits })
}

// Marginalize state-specific laws, never pass averaged articulation indicators through nonlinear heads.
fn law<const WITH_EXITS: bool>(
    values: [Option<f64>; 12],
    states: [f64; 5],
    lo: f64,
    hi: f64,
    low: bool,
    cfg: TemporalPhraseConfig,
) -> ([f64; 6], Option<f64>) {
    let mut result = [0.; 6];
    let mut survival_supported = true;
    for (state, weight) in states.into_iter().enumerate().filter(|(_, w)| *w > 0.) {
        let state = [
            Some(gesture::State::Attack),
            Some(gesture::State::Continuation),
            Some(gesture::State::Release),
            Some(gesture::State::Gap),
            None,
        ][state];
        let Some(law) = conditional_law::<WITH_EXITS>(values, state, [lo, hi], cfg) else {
            result[5] += weight;
            survival_supported = false;
            continue;
        };
        let survival = law.survival;
        result[0] += weight * survival;
        // Continuation needs survival alone; exit logits cannot change it.
        if !WITH_EXITS {
            continue;
        }
        let Some((scores, total)) = law.exits else {
            result[5] += weight * (1. - survival);
            continue;
        };
        for i in 0..4 {
            result[if i == 3 && !low { 5 } else { i + 1 }] +=
                weight * (1. - survival) * scores[i] / total;
        }
    }
    (result, survival_supported.then_some(result[0]))
}

impl Group {
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

    fn feature_window(
        &self,
        start: u64,
        end: u64,
        rate: u32,
        rms_reference: f64,
    ) -> window::Summary {
        window::summarize(
            start,
            end,
            end,
            rate,
            rms_reference,
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
            }),
        )
        .expect("validated observed phrase window")
    }

    fn recent_energy(&self, start: u64, end: u64, rate: u32, rms_reference: f64) -> RecentEnergy {
        let summary = self.feature_window(start, end, rate, rms_reference);
        RecentEnergy {
            start_sample: start,
            end_sample: end,
            values: std::array::from_fn(|i| summary.values[4 + i].value()),
            coverage: std::array::from_fn(|i| summary.observed_fraction[4 + i]),
        }
    }

    fn ending(&self, span: Foreground, rate: u32) -> WindowDescriptor {
        let end = span.heard_end;
        let start = span
            .start
            .max(self.born)
            .max(end.saturating_sub(2 * u64::from(rate)));
        super::observables::summarize(
            self.handle,
            (start, end),
            rate,
            self.history
                .iter()
                .map(|s| (s.raw, s.energy, s.spectral_shape_supported)),
            self.accents.iter().copied(),
            self.evicted_accent,
        )
    }

    fn step(
        &mut self,
        end: u64,
        observed: bool,
        low: bool,
        rate: u32,
        cfg: TemporalPhraseConfig,
        next: &mut u64,
    ) {
        if end <= self.end {
            return;
        }
        self.proposals.clear();
        let mut unknown = self.unknown;
        let keep = (-((end - self.end) as f64 / f64::from(rate)) / 30.).exp();
        for (parent, path) in self.paths.iter().enumerate() {
            let Some(f) = path.state.foreground else {
                let reentry = if observed && !low { self.states[0] } else { 0. };
                self.proposals.push(Proposal {
                    parent,
                    kind: None,
                    mass: path.mass * (1. - reentry),
                });
                self.proposals.push(Proposal {
                    parent,
                    kind: Some(Exit::New),
                    mass: path.mass * reentry,
                });
                continue;
            };
            let scores = law::<true>(
                self.context,
                self.states,
                (self.end - f.start) as f64 / f64::from(rate),
                (end - f.start) as f64 / f64::from(rate),
                low,
                cfg,
            )
            .0;
            if !observed {
                self.proposals.push(Proposal {
                    parent,
                    kind: None,
                    mass: path.mass * scores[0],
                });
                unknown += path.mass * (1. - scores[0]);
                continue;
            }
            unknown += path.mass * (1. - keep + keep * scores[5]);
            self.proposals.push(Proposal {
                parent,
                kind: None,
                mass: path.mass * keep * scores[0],
            });
            for (i, kind) in EXITS.into_iter().enumerate() {
                self.proposals.push(Proposal {
                    parent,
                    kind: Some(kind),
                    mass: path.mass * keep * scores[i + 1],
                });
            }
        }
        self.proposals.sort_by(|a, b| {
            b.mass
                .total_cmp(&a.mass)
                .then(a.parent.cmp(&b.parent))
                .then((a.kind.map(|v| v as u8)).cmp(&b.kind.map(|v| v as u8)))
        });
        self.scratch.clear();
        for (index, p) in self.proposals.iter().filter(|p| p.mass > 0.).enumerate() {
            if index >= PATHS {
                unknown += p.mass;
                continue;
            }
            // Copy continuation links only after pruning.
            let mut child = self.paths[p.parent];
            child.mass = p.mass;
            let id = p.kind.map(|_| {
                *next += 1;
                *next
            });
            child
                .state
                .advance(p.kind, [self.end, end], observed, id)
                .expect("validated phrase transition");
            self.scratch.push(child);
        }
        std::mem::swap(&mut self.paths, &mut self.scratch);
        self.unknown = unknown;
        self.end = end;
    }
}

impl Phrase {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::temporal_cognition) fn project_action_window(
        &self,
        profiles: &super::action_profiles::Profiles,
        prototype: usize,
        class: crate::life::action_candidates::Class,
        action_offset: u64,
        evaluation_at: u64,
        handle: Handle,
        rms_reference: f64,
        articulation: Option<&gesture::Gesture>,
        arrival_issue: Option<&super::arrival::Frozen>,
        projection_cache: Option<(usize, &mut gesture::ProjectionCache)>,
        mut groove_density: Option<super::groove::DensityProjection>,
        mut groove_timing: Option<super::auditory_timing::Projection<'_>>,
    ) -> Option<super::action_profiles::Cell> {
        #[cfg(test)]
        let mut cost_clock = tests::projection_clock();
        if self.rate != 48_000
            || self.hop != 512
            || handle.bus != self.bus
            || handle.epoch != self.epoch
            || evaluation_at < self.end
            || evaluation_at > self.end.checked_add(192_000)?
            || groove_density
                .as_ref()
                .is_some_and(|d| d.issued_at != self.end || d.evaluated_at != evaluation_at)
        {
            return None;
        }
        let group = self.groups.iter().flatten().find(|g| g.handle == handle)?;
        let observed = || {
            group.history.iter().map(|s| window::Frame {
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
        };
        let previous = observed().next_back().filter(|f| {
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
        let projected = profiles.frames(
            prototype,
            class,
            action_offset,
            self.end,
            previous,
            background,
        )?;
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
            observed().chain(
                projected
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
            observed().chain(
                profiles
                    .frames(
                        prototype,
                        class,
                        action_offset,
                        self.end,
                        previous,
                        background,
                    )?
                    .take(projected_end)
                    .skip((short_start.saturating_sub(self.end) / self.hop) as usize),
            ),
        )
        .ok()?;
        #[cfg(test)]
        tests::record_projection_cost(&mut cost_clock, 0);
        let mut phrase_features = [Feature::Unsupported; 8];
        phrase_features[..4].copy_from_slice(&short.values[..4]);
        let (grouping, grouping_observed_fraction) =
            group.grouping_window(start, evaluation_at, self.end);
        phrase_features[4] = grouping.map_or(Feature::Unsupported, Feature::Observed);
        if evaluation_at == self.end {
            phrase_features[5] = group.context[9].map_or(Feature::Unsupported, Feature::Observed);
        }
        #[cfg(test)]
        tests::record_projection_cost(&mut cost_clock, 1);
        let articulation = articulation.and_then(|model| {
            model.project_states(
                handle,
                self.end,
                evaluation_at,
                profiles.frames(
                    prototype,
                    class,
                    action_offset,
                    self.end,
                    previous,
                    background,
                )?,
                projection_cache,
            )
        });
        #[cfg(test)]
        tests::record_projection_cost(&mut cost_clock, 2);
        let residual = group
            .projected_residual(
                start,
                evaluation_at,
                self.end,
                profiles.frames(
                    prototype,
                    class,
                    action_offset,
                    self.end,
                    previous,
                    background,
                )?,
            )
            .ok()?;
        let mut closure_inputs = [None; 14];
        closure_inputs[0] = residual.value.value();
        let closure = ordinal(&closure_inputs, self.config.closure);
        #[cfg(test)]
        tests::record_projection_cost(&mut cost_clock, 3);
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
                groove_density.as_mut(),
                groove_timing.as_mut(),
                profiles.frames(
                    prototype,
                    class,
                    action_offset,
                    self.end,
                    previous,
                    background,
                )?,
            )
            .ok()?;
        let arrival = arrival.and_then(|a| a.finish());
        if let Some(density) = &mut groove_density {
            density.finish(self.rate);
        }
        let groove_timing = groove_timing.map(|timing| timing.finish());
        let continuation_context = [
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
            phrase_features[4],
        ];
        closure_inputs[10] = continuation_context[0].value();
        closure_inputs[12] = continuation_context[1].value();
        closure_inputs[13] = continuation_context[2].value();
        #[cfg(test)]
        tests::record_projection_cost(&mut cost_clock, 4);
        let mut phrase_values = [None; 12];
        for (out, value) in phrase_values[4..].iter_mut().zip(phrase_features) {
            *out = value.value();
        }
        let mut continuation_paths: [Option<ConditionalOrdinal>; PATHS] = [None; PATHS];
        for (index, path) in group.paths.iter().take(PATHS).enumerate() {
            let foreground_start = path.state.foreground.map(|f| f.start);
            // This cell shares all head inputs except foreground duration across paths.
            if let Some(prior) = continuation_paths[..index]
                .iter()
                .flatten()
                .find(|prior| prior.foreground_start == foreground_start)
                .copied()
            {
                continuation_paths[index] = Some(ConditionalOrdinal {
                    path_index: index,
                    issue_mass: path.mass,
                    ..prior
                });
                continue;
            }
            let survival = if let Some(foreground) = path.state.foreground {
                articulation.and_then(|a| {
                    let states = [
                        a.states[0],
                        a.states[1],
                        a.states[2],
                        a.states[3],
                        a.unknown,
                    ];
                    let elapsed = (evaluation_at - foreground.start) as f64 / f64::from(self.rate);
                    law::<false>(
                        phrase_values,
                        states,
                        elapsed,
                        elapsed + 1.,
                        false,
                        self.config,
                    )
                    .1
                })
            } else {
                Some(1.)
            };
            closure_inputs[11] = survival;
            continuation_paths[index] = Some(ConditionalOrdinal {
                path_index: index,
                issue_mass: path.mass,
                foreground_start,
                survival: survival.map_or(Feature::Unsupported, |value| {
                    if evaluation_at == self.end {
                        Feature::Observed(value)
                    } else {
                        Feature::Projected(value)
                    }
                }),
                categories: ordinal(&closure_inputs, self.config.continuation),
            });
        }

        #[cfg(test)]
        tests::record_projection_cost(&mut cost_clock, 5);
        Some(super::action_profiles::Cell {
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
            phrase_features,
            phrase_observed_fraction: std::array::from_fn(|i| short.observed_fraction[i]),
            phrase_projected_fraction: std::array::from_fn(|i| short.projected_fraction[i]),
            grouping_observed_fraction,
            residual,
            accent_density,
            groove_density,
            groove_timing,
            groove_context: None,
            groove_heads: None,
            arrival,
            closure,
            continuation_context,
            continuation_paths,
            issue_phrase_unknown: group.unknown,
            raw_unreweighted_heads: None,
        })
    }

    pub(crate) fn validate(cfg: TemporalPhraseConfig) -> Result<(), &'static str> {
        if cfg
            .means
            .iter()
            .chain(&cfg.hazard)
            .chain(cfg.exits.iter().flatten())
            .any(|x| !x.is_finite())
            || cfg.deviations.iter().any(|x| !x.is_finite() || *x < 0.)
        {
            return Err("invalid phrase coefficients or scales");
        }
        for c in [cfg.closure, cfg.continuation] {
            if c.means
                .iter()
                .chain(&c.coefficients)
                .chain(&c.cutpoints)
                .any(|x| !x.is_finite())
                || c.deviations.iter().any(|x| !x.is_finite() || *x < 0.)
                || c.cutpoints.windows(2).any(|w| w[0] >= w[1])
                || c.prior.iter().any(|p| !p.is_finite() || *p <= 0.)
                || (c.prior.iter().sum::<f64>() - 1.).abs() > 1e-12
            {
                return Err("invalid ordinal coefficients, cutpoints or frozen prior");
            }
        }
        if cfg.closure.coefficients[11..15]
            .iter()
            .chain(&cfg.closure.coefficients[25..29])
            .any(|x| *x != 0.)
        {
            return Err("closure cannot consume continuation-only coordinates");
        }
        Ok(())
    }
    pub(crate) fn new(
        bus: u8,
        epoch: u64,
        start: u64,
        rate: u32,
        hop: u64,
        config: TemporalPhraseConfig,
    ) -> Result<Box<Self>, &'static str> {
        Self::validate(config)?;
        if bus > 1 || rate == 0 || hop == 0 {
            return Err("invalid phrase stream");
        }
        let empty = |prior| Ordinal {
            supported_expected_rating: None,
            categories: prior,
            expected_rating: prior
                .into_iter()
                .enumerate()
                .map(|(i, p)| i as f64 * p / 4.)
                .sum(),
            supported_mass: 0.,
            observed_coverage: 0.,
            unknown: 1.,
        };
        Ok(Box::new(Self {
            config,
            bus,
            epoch,
            rate,
            hop,
            start,
            end: start,
            next_span: 0,
            groups: std::array::from_fn(|_| None),
            observed: VecDeque::with_capacity((2 * u64::from(rate)).div_ceil(hop) as usize + 1),
            snapshot: Snapshot {
                version: 5,
                epoch_start_sample: start,
                sample_rate: rate,
                end_sample: start,
                groups: [None; 7],
                closure: empty(config.closure.prior),
                continuation: empty(config.continuation.prior),
                censored: false,
            },
        }))
    }
    pub(crate) fn snapshot(&self) -> Snapshot {
        self.snapshot
    }

    pub(crate) fn body_descriptors(&self) -> [Option<WindowDescriptor>; 7] {
        std::array::from_fn(|i| {
            self.groups[i].as_ref().map(|g| {
                // Body matching uses the group lifetime, not any inferred foreground boundary.
                super::observables::summarize(
                    g.handle,
                    (
                        g.born
                            .max(self.start)
                            .max(self.end.saturating_sub(2 * u64::from(self.rate))),
                        self.end,
                    ),
                    self.rate,
                    g.history
                        .iter()
                        .map(|s| (s.raw, s.energy, s.spectral_shape_supported)),
                    g.accents.iter().copied(),
                    g.evicted_accent,
                )
            })
        })
    }

    pub(in crate::temporal_cognition) fn advance(
        &mut self,
        acoustic: &frontend::Snapshot,
        articulation: &gesture::Gesture,
        memory: Option<recall::Snapshot>,
        period: Option<frontend::recurrence::Snapshot>,
        end: u64,
    ) -> Result<(), &'static str> {
        if end <= self.end {
            return Err("backward phrase observation");
        }
        let rate = u64::from(self.rate);
        let window = end.saturating_sub(2 * rate).max(self.start);
        for slot in &mut self.groups {
            if slot
                .as_ref()
                .is_some_and(|g| !acoustic.retained_groups.contains(&Some(g.handle)))
            {
                *slot = None;
            } else if let Some(g) = slot {
                g.snapshot.previous_end_sample = g.end;
                for (index, path) in g.paths.iter_mut().enumerate() {
                    path.parent_index = index;
                    path.state.completed_foreground = None;
                    path.state.completed_ending = None;
                }
            }
        }
        let denominator: f64 = acoustic
            .assignment
            .rows
            .iter()
            .flatten()
            .flat_map(|r| r.weights)
            .sum();
        if acoustic.spectral_shape_supported && denominator > 0. {
            self.observed.push_back((end - self.hop, end));
        }
        while self.observed.front().is_some_and(|s| s.1 <= window) {
            self.observed.pop_front();
        }
        for (index, update) in acoustic.features[..7]
            .iter()
            .enumerate()
            .filter_map(|(i, u)| u.map(|u| (i, u)))
        {
            let raw = update.raw;
            if raw.group.bus != self.bus
                || raw.group.epoch != self.epoch
                || raw.end != end
                || raw.available_end > end
                || raw.source_end > raw.available_end
                || raw.start >= raw.end
            {
                return Err("noncausal phrase input");
            }
            if !acoustic.retained_groups.contains(&Some(raw.group)) {
                continue;
            }
            let observed = acoustic.eligible[index]
                && raw.known_samples == self.hop
                && raw.end - raw.start == self.hop;
            let slot = if let Some(i) = self
                .groups
                .iter()
                .position(|g| g.as_ref().is_some_and(|g| g.handle == raw.group))
            {
                i
            } else {
                if !observed {
                    continue;
                }
                let i = self
                    .groups
                    .iter()
                    .position(Option::is_none)
                    .ok_or("phrase group capacity")?;
                self.next_span += 1;
                let path = Path {
                    parent_index: 0,
                    state: Interpretation::new(self.next_span, [raw.start, end])
                        .expect("validated phrase admission"),
                    mass: 1.,
                };
                let mut paths = Vec::with_capacity(PATHS);
                paths.push(path);
                self.groups[i] = Some(Group {
                    handle: raw.group,
                    born: raw.start,
                    end: raw.start,
                    paths,
                    scratch: Vec::with_capacity(PATHS),
                    proposals: Vec::with_capacity(PATHS * 5),
                    unknown: 0.,
                    history: VecDeque::with_capacity((2 * rate).div_ceil(self.hop) as usize + 1),
                    context: [None; 12],
                    arrival_probability: None,
                    states: [0., 0., 0., 0., 1.],
                    query: None,
                    cue: None,
                    cue_id: 0,
                    accents: VecDeque::with_capacity(128),
                    last_accent: None,
                    evicted_accent: None,
                    snapshot: GroupSnapshot {
                        group: raw.group,
                        previous_end_sample: raw.start,
                        acoustic_weight: 0.,
                        recent_energy: None,
                        candidates: [None; PATHS],
                        unknown: 0.,
                        forecast: None,
                        phrase_values: [None; 12],
                        articulation_mass: [0., 0., 0., 0., 1.],
                        closure_values: [None; 10],
                        closure: None,
                        continuation: None,
                        cue: None,
                        prediction_comparisons: 0,
                        source_start: raw.source_start,
                        source_end: raw.source_end,
                        available: raw.available_end,
                    },
                });
                i
            };
            let g = self.groups[slot].as_mut().unwrap();
            if raw.start > g.end {
                g.step(
                    raw.start,
                    false,
                    false,
                    self.rate,
                    self.config,
                    &mut self.next_span,
                );
            }
            let low =
                raw.values[2].is_some_and(|r| 2_f64.powf(r) <= 0.01 * articulation.rms_reference());
            g.step(
                end,
                observed,
                observed && low,
                self.rate,
                self.config,
                &mut self.next_span,
            );
            let states = articulation.current_states(raw.group);
            let pg = period.and_then(|p| {
                p.groups
                    .into_iter()
                    .flatten()
                    .find(|g| g.ledger.group == raw.group && g.active)
            });
            let grouping = pg
                .and_then(|p| p.grouping)
                .and_then(|g| g.admission_support);
            let alpha = if observed && acoustic.spectral_shape_supported && denominator > 0. {
                acoustic
                    .assignment
                    .rows
                    .iter()
                    .flatten()
                    .map(|r| r.weights[index])
                    .sum::<f64>()
                    / denominator
            } else {
                0.
            };
            if let Some(q) = memory.and_then(|m| m.latest_for(g.handle)).filter(|q| {
                q.group == g.handle
                    && q.available_at <= q.issued_at
                    && q.issued_at <= q.completed_at
                    && q.completed_at <= q.received_at
                    && q.received_at <= end
                    && q.deadline >= end
                    && q.supporting_audio_end
                        .is_some_and(|t| t <= q.issued_at && end - t < rate / 2)
            }) {
                g.query = Some(q);
                if q.query_id > g.cue_id {
                    g.cue_id = q.query_id;
                    if let Some(c) = q.prediction.filter(|c| {
                        c.group == g.handle
                            && c.query_id == q.query_id
                            && c.issued_at == q.issued_at
                            && c.expected_end > raw.start
                            && c.expected_start < c.expected_end
                            && c.issued_at <= raw.start
                            && c.source_start <= c.source_end
                            && c.source_end <= c.available
                            && c.available <= c.issued_at
                    }) {
                        g.snapshot.source_start = g.snapshot.source_start.min(c.source_start);
                        g.cue = Some(c);
                    }
                }
            }
            let mut residual = None;
            let mut residual_interval = (0, 0);
            if observed
                && let Some(c) = g
                    .cue
                    .filter(|c| c.issued_at <= raw.start && c.available <= c.issued_at)
            {
                residual_interval = (raw.start.max(c.expected_start), raw.end.min(c.expected_end));
                if residual_interval.1 > residual_interval.0 {
                    residual = c.normalized_residual(raw.values);
                    if residual.is_some() {
                        g.snapshot.prediction_comparisons += 1;
                    }
                }
            }
            if observed {
                g.history.push_back(Sample {
                    raw,
                    energy: acoustic.energy.map(|energies| energies[index]),
                    spectral_shape_supported: acoustic.spectral_shape_supported,
                    alpha,
                    grouping,
                    residual,
                    residual_interval,
                });
            }
            while g.history.front().is_some_and(|s| s.raw.end <= window) {
                g.history.pop_front();
            }
            if let Some(a) = update
                .detector
                .and_then(|d| d.accent)
                .filter(|a| a.at_cut(end) && Some(a.event_end) != g.last_accent)
            {
                if g.accents.len() == 128 {
                    g.evicted_accent = g.accents.pop_front().map(|a| a.event_end);
                }
                g.accents.push_back(a);
                g.last_accent = Some(a.event_end);
            }
            while g.accents.front().is_some_and(|a| a.event_end < window) {
                g.accents.pop_front();
            }
            if observed {
                for i in 0..g.paths.len() {
                    if g.paths[i].state.completed_ending.is_none()
                        && let Some(f) = g.paths[i]
                            .state
                            .completed_foreground
                            .filter(|f| f.start < f.heard_end && f.heard_end == end)
                    {
                        g.paths[i].state.completed_ending = Some(g.ending(f, self.rate));
                    }
                    if let Some(f) = g.paths[i]
                        .state
                        .foreground
                        .filter(|f| f.start < f.heard_end)
                    {
                        let start = f
                            .start
                            .max(g.born)
                            .max(f.heard_end.saturating_sub(2 * rate));
                        let cached = g.paths[..i].iter().find_map(|p| {
                            p.state
                                .ending
                                .filter(|e| e.start == start && e.end == f.heard_end)
                        });
                        g.paths[i].state.ending = cached.or_else(|| Some(g.ending(f, self.rate)));
                    }
                }
            }
            let mut values = [None; 12];
            let short = g.feature_window(
                end.saturating_sub(rate / 4).max(g.born),
                end,
                self.rate,
                articulation.rms_reference(),
            );
            for i in 0..4 {
                values[4 + i] = short.values[i].value();
            }
            let start = window.max(g.born);
            g.snapshot.recent_energy =
                Some(g.recent_energy(start, end, self.rate, articulation.rms_reference()));
            values[8] = g.grouping_window(start, end, end).0;
            if let Some(q) = g.query.filter(|q| {
                q.deadline >= end
                    && q.supporting_audio_end
                        .is_some_and(|t| t <= end && end - t < rate / 2)
            }) {
                values[9] = q
                    .best
                    .filter(|b| !b.ambiguous && b.available_at <= end)
                    .map(|b| b.cost.ln_1p());
            }
            g.context = values;
            g.states = states;
            let mut closure_values = [None; 14];
            let mut sum = 0.;
            let mut count = 0;
            for s in &g.history {
                if let Some(r) = s.residual {
                    let n = s
                        .residual_interval
                        .1
                        .saturating_sub(s.residual_interval.0.max(start));
                    sum += r * n as f64;
                    count += n;
                }
            }
            if count > 0 {
                closure_values[0] = Some(sum / count as f64);
            }
            g.snapshot.source_start = g
                .history
                .iter()
                .map(|s| s.raw.source_start)
                .min()
                .unwrap_or(g.snapshot.source_start)
                .min(g.snapshot.source_start);
            g.snapshot.source_end = g
                .history
                .iter()
                .map(|s| s.raw.source_end)
                .max()
                .unwrap_or(g.snapshot.source_end);
            g.snapshot.available = g
                .history
                .iter()
                .map(|s| s.raw.available_end)
                .max()
                .unwrap_or(g.snapshot.available);
            if let Some(q) = g.query.filter(|_| values[9].is_some()) {
                g.snapshot.source_start = g.snapshot.source_start.min(q.source_start_sample);
                if let Some(b) = q.best {
                    g.snapshot.source_start = g.snapshot.source_start.min(b.source_start_sample);
                }
            }
            if let Some(c) = g.cue {
                g.snapshot.source_start = g.snapshot.source_start.min(c.source_start);
            }
            let mut forecast = Forecast {
                issued_at: end,
                horizon_end: end.checked_add(rate).ok_or("phrase horizon overflow")?,
                source_start: g.snapshot.source_start,
                source_end: g.snapshot.source_end,
                available: g.snapshot.available,
                survival: 0.,
                exits: [0.; 4],
                unknown: g.unknown,
            };
            for path in &g.paths {
                if let Some(f) = path.state.foreground {
                    forecast.source_start = forecast.source_start.min(f.start);
                    let d = (end - f.start) as f64 / f64::from(self.rate);
                    let p = law::<true>(values, states, d, d + 1., observed && low, self.config).0;
                    forecast.survival += path.mass * p[0];
                    for i in 0..4 {
                        forecast.exits[i] += path.mass * p[i + 1];
                    }
                    forecast.unknown += path.mass * p[5];
                } else {
                    forecast.survival += path.mass;
                }
            }
            g.snapshot.forecast = Some(forecast);
            g.snapshot.phrase_values = values;
            g.snapshot.articulation_mass = states;
            g.snapshot
                .closure_values
                .copy_from_slice(&closure_values[..10]);
            g.snapshot.closure = ordinal(&closure_values, self.config.closure);
            let arrival = pg
                .and_then(|p| p.forecast)
                .filter(|f| f.issued_at == end && f.horizon_end - end == rate && !f.reset_unknown)
                .and_then(|f| f.probability)
                .filter(|p| p[0] == p[1])
                .map(|p| p[0]);
            g.arrival_probability = arrival;
            closure_values[10] = arrival;
            let known = g.paths.iter().map(|p| p.mass).sum::<f64>();
            // Evaluate duration-dependent ordinal inputs for each retained phrase path.
            let mut continuation = [0.; 5];
            let mut supported = 0.;
            closure_values[12] = g.accent_density(start, end, end, self.rate);
            closure_values[13] = values[8];
            for p in &g.paths {
                closure_values[11] = if let Some(f) = p.state.foreground {
                    let d = (end - f.start) as f64 / f64::from(self.rate);
                    let l = law::<false>(values, states, d, d + 1., observed && low, self.config);
                    l.1
                } else {
                    Some(1.)
                };
                if let Some(c) = ordinal(&closure_values, self.config.continuation) {
                    for i in 0..5 {
                        continuation[i] += p.mass * c[i];
                    }
                    supported += p.mass;
                }
            }
            g.snapshot.continuation =
                (supported > 0. && known > 0.).then(|| continuation.map(|p| p / supported));
            g.snapshot.unknown = g.unknown;
            g.snapshot.cue = g.cue;
        }
        for g in self.groups.iter_mut().flatten() {
            if g.end < end {
                g.step(
                    end,
                    false,
                    false,
                    self.rate,
                    self.config,
                    &mut self.next_span,
                );
                g.snapshot.forecast = None;
                g.snapshot.closure = None;
                g.snapshot.continuation = None;
                g.snapshot.recent_energy = None;
                g.snapshot.unknown = g.unknown;
            }
        }
        self.end = end;
        self.refresh(false);
        Ok(())
    }

    fn refresh(&mut self, censored: bool) {
        let start = self
            .end
            .saturating_sub(2 * u64::from(self.rate))
            .max(self.start);
        let observed: u64 = self
            .observed
            .iter()
            .map(|&(a, b)| b.saturating_sub(a.max(start)))
            .sum();
        let coverage = if self.end > start {
            observed as f64 / (self.end - start) as f64
        } else {
            0.
        };
        for g in self.groups.iter_mut().flatten() {
            g.snapshot.unknown = g.unknown;
            g.snapshot.candidates = std::array::from_fn(|i| {
                g.paths.get(i).map(|p| Candidate {
                    parent_index: p.parent_index,
                    completed_foreground: p.state.completed_foreground,
                    completed_ending: p.state.completed_ending,
                    foreground: p.state.foreground,
                    mass: p.mass,
                    event: p.state.event,
                    unresolved_links: p.state.links.iter().flatten().count(),
                    right_censored_links: p
                        .state
                        .links
                        .iter()
                        .flatten()
                        .filter(|l| l.right_censored)
                        .count(),
                    oldest_link_start: p.state.links.iter().flatten().map(|l| l.span.start).min(),
                    lost_links: p.state.lost_links,
                })
            });
        }
        let mut distributions = [[0.; 5]; 2];
        let mut supports = [0.; 2];
        for g in self.groups.iter_mut().flatten() {
            let weight = if observed > 0 {
                g.history
                    .iter()
                    .map(|s| s.alpha * s.raw.end.saturating_sub(s.raw.start.max(start)) as f64)
                    .sum::<f64>()
                    / observed as f64
            } else {
                0.
            };
            g.snapshot.acoustic_weight = weight;
            for (h, prediction) in [g.snapshot.closure, g.snapshot.continuation]
                .into_iter()
                .enumerate()
            {
                if let Some(p) = prediction {
                    let w = weight * (1. - g.unknown);
                    supports[h] += w;
                    for (i, p) in p.into_iter().enumerate() {
                        distributions[h][i] += w * p;
                    }
                }
            }
        }
        let heads = std::array::from_fn::<_, 2, _>(|h| {
            let prior = if h == 0 {
                self.config.closure.prior
            } else {
                self.config.continuation.prior
            };
            let r = (coverage * supports[h]).clamp(0., 1.);
            let categories = std::array::from_fn(|i| {
                if supports[h] > 0. {
                    r * distributions[h][i] / supports[h] + (1. - r) * prior[i]
                } else {
                    prior[i]
                }
            });
            Ordinal {
                supported_expected_rating: (supports[h] > 0.).then(|| {
                    distributions[h]
                        .iter()
                        .enumerate()
                        .map(|(i, p)| i as f64 * p / (4. * supports[h]))
                        .sum()
                }),
                categories,
                expected_rating: categories
                    .into_iter()
                    .enumerate()
                    .map(|(i, p)| i as f64 * p / 4.)
                    .sum(),
                supported_mass: supports[h],
                observed_coverage: coverage,
                unknown: 1. - r,
            }
        });
        self.snapshot = Snapshot {
            version: 5,
            epoch_start_sample: self.start,
            sample_rate: self.rate,
            end_sample: self.end,
            groups: std::array::from_fn(|i| self.groups[i].as_ref().map(|g| g.snapshot)),
            closure: heads[0],
            continuation: heads[1],
            censored,
        };
    }
    pub(crate) fn finish(&mut self, end: u64) -> Result<(), &'static str> {
        if end < self.end {
            return Err("backward phrase EOF");
        }
        for g in self.groups.iter_mut().flatten() {
            if end > self.end {
                g.snapshot.previous_end_sample = g.end;
                for (index, path) in g.paths.iter_mut().enumerate() {
                    path.parent_index = index;
                    path.state.completed_foreground = None;
                    path.state.completed_ending = None;
                }
                g.step(
                    end,
                    false,
                    false,
                    self.rate,
                    self.config,
                    &mut self.next_span,
                );
                g.snapshot.forecast = None;
                g.snapshot.closure = None;
                g.snapshot.continuation = None;
                g.snapshot.unknown = g.unknown;
            }
        }
        self.end = end;
        self.refresh(true);
        Ok(())
    }
}

#[cfg(test)]
pub(in crate::temporal_cognition) mod tests;
