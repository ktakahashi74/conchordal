//! Shared six-coordinate acoustic window; private body and group inputs stay separate.

use super::{
    features::{Accent, RawDescriptor},
    ridge::Handle,
};

pub(crate) fn standardize(
    values: [Option<f64>; 6],
    scales: crate::config::TemporalBodyConfig,
) -> [Option<f64>; 6] {
    std::array::from_fn(|i| {
        values[i]
            .map(|v| (v - scales.means[i]) / scales.deviations[i].max(1e-6))
            .filter(|v| v.is_finite())
    })
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct WindowDescriptor {
    pub group: Handle,
    pub start: u64,
    pub end: u64,
    pub raw_values: [Option<f64>; 6],
    pub coverage: [f64; 6],
    pub source_start: u64,
    pub available: u64,
}
pub(super) fn summarize(
    group: Handle,
    (start, end): (u64, u64),
    rate: u32,
    samples: impl Iterator<Item = (RawDescriptor, Option<f64>, bool)>,
    accents: impl Iterator<Item = Accent>,
    evicted_accent: Option<u64>,
) -> WindowDescriptor {
    let mut result = WindowDescriptor {
        group,
        start,
        end,
        raw_values: [None; 6],
        coverage: [0.; 6],
        source_start: end,
        available: 0,
    };
    if end <= start {
        return result;
    }
    let mut known = [0u64; 6];
    let mut numerators = [0.; 6];
    let mut spectral_mass = 0.;
    let mut centroid = 0.;
    let mut spread_mass = 0.;
    for (raw, energy, spectral_shape_supported) in samples {
        if raw.end > end || raw.available_end > end || raw.source_end > end {
            continue;
        }
        let samples = raw.end.saturating_sub(raw.start.max(start));
        if samples == 0 {
            continue;
        }
        result.source_start = result.source_start.min(raw.source_start);
        result.available = result.available.max(raw.available_end);
        known[5] += samples;
        if let Some(energy) = energy {
            known[2] += samples;
            numerators[2] += energy * samples as f64;
            if spectral_shape_supported {
                if energy == 0. {
                    known[0] += samples;
                    known[1] += samples;
                } else if let (Some(center), Some(spread)) = (raw.values[0], raw.values[1]) {
                    known[0] += samples;
                    known[1] += samples;
                    let weight = samples as f64 * energy;
                    let total = spectral_mass + weight;
                    let delta = center - centroid;
                    spread_mass +=
                        weight * spread * spread + spectral_mass / total * weight * delta * delta;
                    centroid += weight / total * delta;
                    spectral_mass = total;
                }
            }
        }
        for (i, coordinate) in [(3, 3), (4, 5)] {
            if let Some(value) = raw.values[coordinate] {
                numerators[i] += value * samples as f64;
                known[i] += samples;
            }
        }
    }
    result.coverage = known.map(|n| n as f64 / (end - start) as f64);
    if spectral_mass > 0. && result.coverage[0] >= 0.9 {
        result.raw_values[0] = Some(centroid);
        result.raw_values[1] = Some((spread_mass / spectral_mass).max(0.).sqrt());
    }
    if result.coverage[2] >= 0.9 {
        result.raw_values[2] = Some((numerators[2] / known[2] as f64).sqrt().max(1e-6).log2());
    }
    for i in [3, 4] {
        if result.coverage[i] >= 0.9 {
            result.raw_values[i] = Some(numerators[i] / known[i] as f64);
        }
    }
    if result.coverage[5] >= 0.9 && evicted_accent.is_none_or(|t| t < start) {
        let events: f64 = accents
            .filter(|a| a.at_cut(end) && start <= a.event_end && a.event_end <= end)
            .map(|a| a.weight)
            .sum();
        result.raw_values[5] = Some((events * f64::from(rate) / known[5] as f64).ln_1p());
    }
    for value in &mut result.raw_values {
        if value.is_some_and(|v| !v.is_finite()) {
            *value = None;
        }
    }
    result
}
