//! Physical acoustic windows without interpretation weights or marginal labels.

use crate::temporal_cognition::{
    feature_projection::{Feature, window},
    features::{Accent, RawDescriptor},
    observables,
    phrase::{Foreground, interpretation::Interpretation},
    proposals::frontend,
    ridge::Handle,
    section::input::Observation,
};
use std::collections::VecDeque;

#[derive(Clone, Copy)]
struct Sample {
    raw: RawDescriptor,
    energy: Option<f64>,
    spectral: bool,
}

pub(super) struct Cache {
    group: Handle,
    born: u64,
    end: u64,
    rate: u32,
    hop: u64,
    receipt: Option<Observation>,
    history: VecDeque<Sample>,
    accents: VecDeque<Accent>,
    last_accent: Option<u64>,
    evicted_accent: Option<u64>,
}

impl Cache {
    pub fn new(group: Handle, start: u64, rate: u32, hop: u64) -> Result<Self, &'static str> {
        if group.bus > 1 || group.generation == 0 || rate == 0 || hop == 0 {
            return Err("invalid joint acoustic cache owner or clock");
        }
        Ok(Self {
            group,
            born: start,
            end: start,
            rate,
            hop,
            receipt: None,
            history: VecDeque::with_capacity((2 * u64::from(rate)).div_ceil(hop) as usize + 1),
            accents: VecDeque::with_capacity(128),
            last_accent: None,
            evicted_accent: None,
        })
    }

    pub fn advance(
        &mut self,
        acoustic: &frontend::Snapshot,
        interval: [u64; 2],
    ) -> Result<Observation, &'static str> {
        if interval[0] != self.end
            || interval[1] <= interval[0]
            || interval[1] - interval[0] < self.hop
        {
            return Err("noncausal joint acoustic cache interval");
        }
        let observation = Observation::new(acoustic, self.group, interval, self.rate, None)?;
        let raw = observation.raw.filter(|_| observation.observed);
        let index = acoustic
            .group_handles
            .iter()
            .position(|g| *g == Some(self.group));
        let energy = index.and_then(|i| acoustic.energy.map(|e| e[i]));
        if raw.is_some_and(|u| {
            interval[1] - interval[0] != self.hop
                || !acoustic.retained_groups.contains(&Some(self.group))
                || u.raw.group != self.group
                || u.raw.source_start > u.raw.start
                || u.raw.source_end < u.raw.end
                || u.raw.source_end > u.raw.available_end
                || u.raw.values.iter().flatten().any(|v| !v.is_finite())
                || [1, 3, 4, 5, 6]
                    .iter()
                    .any(|&i| u.raw.values[i].is_some_and(|v| v < 0.))
        }) || energy.is_some_and(|e| !e.is_finite() || e < 0.)
            || !observation.delta.assignment_seconds.is_finite()
            || observation.delta.assignment_seconds < 0.
            || observation.delta.assignment_seconds > observation.delta.physical_window_seconds
        {
            return Err("invalid joint acoustic cache evidence");
        }
        let accent = observation
            .raw
            .and_then(|u| u.detector)
            .and_then(|d| d.accent)
            .filter(|a| a.at_cut(interval[1]) && Some(a.event_end) != self.last_accent);
        if accent.is_some_and(|a| {
            a.group != self.group
                || a.event_start >= a.event_end
                || a.event_end > a.source_end
                || a.source_end > a.available_end
                || !a.weight.is_finite()
                || a.weight <= 0.
                || self.last_accent.is_some_and(|t| a.event_end < t)
        }) {
            return Err("invalid joint acoustic accent provenance");
        }
        let window = interval[1].saturating_sub(2 * u64::from(self.rate));
        while self.history.front().is_some_and(|s| s.raw.end <= window) {
            self.history.pop_front();
        }
        if let Some(u) = raw {
            assert!(self.history.len() < self.history.capacity());
            self.history.push_back(Sample {
                raw: u.raw,
                energy,
                spectral: acoustic.spectral_shape_supported,
            });
        }
        while self.accents.front().is_some_and(|a| a.event_end < window) {
            self.accents.pop_front();
        }
        if let Some(a) = accent {
            if self.accents.len() == 128 {
                self.evicted_accent = self.accents.pop_front().map(|a| a.event_end);
            }
            self.accents.push_back(a);
            self.last_accent = Some(a.event_end);
        }
        self.end = interval[1];
        self.receipt = Some(observation);
        Ok(observation)
    }

    pub fn matches(&self, observation: &Observation) -> bool {
        let mut observation = *observation;
        // Canonical delivery ownership is validated by the section accumulator.
        observation.delivery = None;
        self.receipt == Some(observation)
    }

    pub(super) fn assignment(
        &self,
        observation: &Observation,
    ) -> Result<Option<f64>, &'static str> {
        if !self.matches(observation) {
            return Err("phrase assignment differs from original observation");
        }
        Ok(self
            .history
            .back()
            .filter(|s| observation.observed && s.spectral && s.raw.end == observation.interval[1])
            .map(|s| {
                observation.delta.assignment_seconds * f64::from(self.rate)
                    / (s.raw.end - s.raw.start) as f64
            }))
    }

    pub fn short(&self, cut: u64, rms_reference: f64) -> Result<window::Summary, &'static str> {
        if cut > self.end || cut < self.born {
            return Err("noncausal joint short-window request");
        }
        window::summarize(
            cut.saturating_sub(u64::from(self.rate) / 4).max(self.born),
            cut,
            cut,
            self.rate,
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
    }

    pub fn ending(&self, span: Foreground) -> Result<observables::WindowDescriptor, &'static str> {
        if span.start > span.heard_end || span.heard_end > self.end {
            return Err("noncausal joint phrase ending request");
        }
        Ok(observables::summarize(
            self.group,
            (
                span.start
                    .max(self.born)
                    .max(span.heard_end.saturating_sub(2 * u64::from(self.rate))),
                span.heard_end,
            ),
            self.rate,
            self.history.iter().map(|s| (s.raw, s.energy, s.spectral)),
            self.accents.iter().copied(),
            self.evicted_accent,
        ))
    }

    pub fn project(&self, phrase: &mut Interpretation, observed: bool) -> Result<(), &'static str> {
        if !observed {
            return Ok(());
        }
        if phrase.completed_ending.is_none()
            && let Some(f) = phrase
                .completed_foreground
                .filter(|f| f.start < f.heard_end && f.heard_end == self.end)
        {
            phrase.completed_ending = Some(self.ending(f)?);
        }
        if let Some(f) = phrase.foreground.filter(|f| f.start < f.heard_end) {
            phrase.ending = Some(self.ending(f)?);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
