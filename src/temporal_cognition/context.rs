//! Per-group observed acoustic context: a bounded two-second history of raw descriptors,
//! energies and admitted accents, for body matching and candidate-window projection.

use super::feature_projection::{Feature, window};
use super::observables::WindowDescriptor;
use super::{features::RawDescriptor, gesture, proposals::frontend, ridge::Handle};
use serde::Serialize;
use std::collections::VecDeque;

mod accents;
pub(crate) use accents::AccentDensity;

#[derive(Clone, Copy)]
struct Sample {
    raw: RawDescriptor,
    energy: Option<f64>,
    spectral_shape_supported: bool,
    alpha: f64,
    grouping: Option<f64>,
}

pub(crate) struct Group {
    handle: Handle,
    born: u64,
    history: VecDeque<Sample>,
    arrival_probability: Option<f64>,
    accents: VecDeque<super::features::Accent>,
    last_accent: Option<u64>,
    evicted_accent: Option<u64>,
    acoustic_weight: f64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct GroupSnapshot {
    pub group: Handle,
    pub born: u64,
    pub acoustic_weight: f64,
    pub history_hops: usize,
    pub accents: usize,
    pub arrival_probability: Option<f64>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Snapshot {
    pub version: u64,
    pub epoch_start_sample: u64,
    pub sample_rate: u32,
    pub end_sample: u64,
    pub observed_coverage: f64,
    pub groups: [Option<GroupSnapshot>; 7],
    pub censored: bool,
}

pub(crate) struct Context {
    bus: u8,
    epoch: u64,
    rate: u32,
    hop: u64,
    start: u64,
    end: u64,
    groups: [Option<Group>; 7],
    observed: VecDeque<(u64, u64)>,
    snapshot: Snapshot,
}

impl Group {
    fn frames(&self) -> impl DoubleEndedIterator<Item = window::Frame> + '_ {
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
    pub(crate) fn new(
        bus: u8,
        epoch: u64,
        start: u64,
        rate: u32,
        hop: u64,
    ) -> Result<Box<Self>, &'static str> {
        if bus > 1 || rate == 0 || hop == 0 {
            return Err("invalid context stream");
        }
        Ok(Box::new(Self {
            bus,
            epoch,
            rate,
            hop,
            start,
            end: start,
            groups: std::array::from_fn(|_| None),
            observed: VecDeque::with_capacity((2 * u64::from(rate)).div_ceil(hop) as usize + 1),
            snapshot: Snapshot {
                version: 1,
                epoch_start_sample: start,
                sample_rate: rate,
                end_sample: start,
                observed_coverage: 0.,
                groups: [None; 7],
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
                // Body matching uses the group lifetime, never an inferred phrase boundary.
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
        period: Option<frontend::recurrence::Snapshot>,
        end: u64,
    ) -> Result<(), &'static str> {
        if end <= self.end {
            return Err("backward context observation");
        }
        let rate = u64::from(self.rate);
        let window = end.saturating_sub(2 * rate).max(self.start);
        for slot in &mut self.groups {
            if slot
                .as_ref()
                .is_some_and(|g| !acoustic.retained_groups.contains(&Some(g.handle)))
            {
                *slot = None;
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
                return Err("noncausal context input");
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
                    .ok_or("context group capacity")?;
                self.groups[i] = Some(Group {
                    handle: raw.group,
                    born: raw.start,
                    history: VecDeque::with_capacity((2 * rate).div_ceil(self.hop) as usize + 1),
                    arrival_probability: None,
                    accents: VecDeque::with_capacity(128),
                    last_accent: None,
                    evicted_accent: None,
                    acoustic_weight: 0.,
                });
                i
            };
            let g = self.groups[slot].as_mut().unwrap();
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
            if observed {
                g.history.push_back(Sample {
                    raw,
                    energy: acoustic.energy.map(|energies| energies[index]),
                    spectral_shape_supported: acoustic.spectral_shape_supported,
                    alpha,
                    grouping,
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
            g.arrival_probability = pg
                .and_then(|p| p.forecast)
                .filter(|f| f.issued_at == end && f.horizon_end - end == rate && !f.reset_unknown)
                .and_then(|f| f.probability)
                .filter(|p| p[0] == p[1])
                .map(|p| p[0]);
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
            g.acoustic_weight = if observed > 0 {
                g.history
                    .iter()
                    .map(|s| s.alpha * s.raw.end.saturating_sub(s.raw.start.max(start)) as f64)
                    .sum::<f64>()
                    / observed as f64
            } else {
                0.
            };
        }
        self.snapshot = Snapshot {
            version: 1,
            epoch_start_sample: self.start,
            sample_rate: self.rate,
            end_sample: self.end,
            observed_coverage: coverage,
            groups: std::array::from_fn(|i| {
                self.groups[i].as_ref().map(|g| GroupSnapshot {
                    group: g.handle,
                    born: g.born,
                    acoustic_weight: g.acoustic_weight,
                    history_hops: g.history.len(),
                    accents: g.accents.len(),
                    arrival_probability: g.arrival_probability,
                })
            }),
            censored,
        };
    }

    pub(crate) fn finish(&mut self, end: u64) -> Result<(), &'static str> {
        if end < self.end {
            return Err("backward context EOF");
        }
        self.end = end;
        self.refresh(true);
        Ok(())
    }

    /// Project one candidate action window from the observed prefix and a conditional body
    /// profile. Hypothetical frames never enter the observed history.
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
    ) -> Option<super::action_profiles::Cell> {
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

#[cfg(test)]
pub(in crate::temporal_cognition) mod tests;
