//! Per-group raw observables and four-hop accent admission with original support.

use super::feature_projection::{self, Feature, Frame, Spectrum};
use super::ridge::Handle;
use crate::core::log2space::Log2Space;

#[derive(Clone, Copy, Debug)]
pub(super) struct Config {
    pub means: [f64; 2],
    pub deviations: [f64; 2],
    pub threshold: f64,
    pub rms_floor: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Stamp {
    pub group: Handle,
    pub association: Option<u64>,
    pub grid_id: u64,
    pub start: u64,
    pub end: u64,
    pub source_start: u64,
    pub source_end: u64,
    pub available_end: u64,
    pub known_samples: u64,
    pub observed: bool,
}

pub(super) struct Input<'a> {
    pub stamp: Stamp,
    pub energy: Option<f64>,
    pub bus_energy: Option<f64>,
    pub energy_scan: Option<&'a [f64]>,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct RawDescriptor {
    pub group: Handle,
    pub start: u64,
    pub end: u64,
    pub known_samples: u64,
    pub values: [Option<f64>; 10],
    pub source_start: u64,
    pub source_end: u64,
    pub available_end: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct Accent {
    pub group: Handle,
    pub event_start: u64,
    pub event_end: u64,
    pub raw_intervals: [(u64, u64); 4],
    pub source_start: u64,
    pub source_end: u64,
    pub available_end: u64,
    pub weight: f64,
    pub observed_prefix: u64,
}

impl Accent {
    pub fn at_cut(&self, cut: u64) -> bool {
        self.source_end <= cut && self.available_end <= cut
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) enum Status {
    Unsupported,
    BelowThreshold,
    NotLocalPeak,
    Admitted,
}

pub(super) fn accent_salience(components: [f64; 2], means: [f64; 2], deviations: [f64; 2]) -> f64 {
    (0..2)
        .map(|i| ((components[i] - means[i]) / deviations[i].max(1e-6)).max(0.))
        .sum::<f64>()
        / 2.
}

pub(super) fn accent_status([left, middle, right]: [f64; 3], threshold: f64) -> Status {
    if middle <= threshold {
        Status::BelowThreshold
    } else if middle <= left || middle < right {
        Status::NotLocalPeak
    } else {
        Status::Admitted
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct Detection {
    pub acquisition_coverage: f64,
    pub detector_coverage: bool,
    pub saliences: Option<[f64; 3]>,
    pub status: Status,
    pub accent: Option<Accent>,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct Update {
    pub raw: RawDescriptor,
    pub detector: Option<Detection>,
}

#[derive(Clone, Copy)]
struct Saved {
    stamp: Stamp,
    energy: Option<f64>,
    bus_energy: Option<f64>,
    has_spectrum: bool,
    salience: Option<f64>,
    observed_prefix: u64,
}

pub(super) struct Stream {
    config: Config,
    history: [Option<Saved>; 3],
    len: usize,
    previous_scan: Vec<f64>,
    observed_prefix: u64,
}

impl Stream {
    pub fn new(bins: usize, config: Config) -> Result<Self, &'static str> {
        if bins == 0
            || config.means.iter().any(|x| !x.is_finite())
            || config.deviations.iter().any(|x| !x.is_finite() || *x < 0.)
            || !config.threshold.is_finite()
            || config.threshold <= 0.
            || !config.rms_floor.is_finite()
            || config.rms_floor <= 0.
            || !(config.rms_floor * config.rms_floor).is_finite()
            || config.rms_floor * config.rms_floor == 0.
        {
            return Err("invalid raw-feature grid or frozen accent scales");
        }
        Ok(Self {
            config,
            history: [None; 3],
            len: 0,
            previous_scan: vec![0.; bins],
            observed_prefix: 0,
        })
    }

    pub fn clear(&mut self) {
        self.history = [None; 3];
        self.len = 0;
        self.observed_prefix = 0;
    }

    pub fn push(
        &mut self,
        space: &Log2Space,
        input: Input<'_>,
        cut: u64,
    ) -> Result<Option<Update>, &'static str> {
        space.assert_scan_len_named(&self.previous_scan, "previous_group_feature_scan");
        if let Some(scan) = input.energy_scan {
            space.assert_scan_len_named(scan, "raw_group_energy_scan");
        }
        let s = input.stamp;
        if s.group.bus > 1
            || s.group.generation == 0
            || s.start >= s.end
            || s.source_start > s.start
            || s.source_end < s.end
            || s.available_end < s.source_end
            || s.known_samples > s.end - s.start
        {
            return Err("invalid raw-feature source support");
        }
        if s.available_end > cut {
            return Ok(None);
        }
        let prior = self.len.checked_sub(1).and_then(|i| self.history[i]);
        if let Some(p) = prior {
            let duplicate = p.stamp == s
                && p.energy == input.energy
                && p.bus_energy == input.bus_energy
                && p.has_spectrum == input.energy_scan.is_some()
                && input
                    .energy_scan
                    .is_none_or(|scan| scan == self.previous_scan);
            if duplicate {
                return Ok(None);
            }
            if s.start < p.stamp.end {
                return Err("stale or conflicting raw-feature delivery");
            }
        }
        if [input.energy, input.bus_energy]
            .into_iter()
            .flatten()
            .any(|x| !x.is_finite() || x < 0.)
            || input
                .energy
                .zip(input.bus_energy)
                .is_some_and(|(e, b)| e > b + 1e-12)
            || input
                .energy_scan
                .is_some_and(|scan| scan.iter().any(|x| !x.is_finite() || *x < 0.))
        {
            return Err("invalid assigned raw-feature energy");
        }
        let known = s.known_samples == s.end - s.start && s.association.is_some();
        let observed_prefix = if prior.is_some_and(|p| p.stamp.group == s.group) {
            self.observed_prefix
        } else {
            0
        } + if s.observed && s.association.is_some() {
            s.known_samples
        } else {
            0
        };
        let previous = prior.filter(|p| {
            known
                && p.stamp.end == s.start
                && p.stamp.observed
                && p.stamp.known_samples == p.stamp.end - p.stamp.start
                && p.stamp.association.is_some()
                && p.stamp.group == s.group
                && p.stamp.association == s.association
                && p.stamp.grid_id == s.grid_id
                && p.stamp.available_end <= cut
        });
        let projection = feature_projection::evaluate(
            space,
            Frame {
                energy: input
                    .energy
                    .filter(|_| known)
                    .map_or(Feature::Unsupported, Feature::Observed),
                bus_energy: input
                    .bus_energy
                    .filter(|_| known)
                    .map_or(Feature::Unsupported, Feature::Observed),
                spectrum: input
                    .energy_scan
                    .filter(|_| known)
                    .map(|energy_scan| Spectrum {
                        energy_scan,
                        projected: false,
                    }),
            },
            previous.map(|p| Frame {
                energy: p.energy.map_or(Feature::Unsupported, Feature::Observed),
                bus_energy: Feature::Unsupported,
                spectrum: p.has_spectrum.then_some(Spectrum {
                    energy_scan: &self.previous_scan,
                    projected: false,
                }),
            }),
            self.config.rms_floor,
        )?;
        let mut raw = RawDescriptor {
            group: s.group,
            start: s.start,
            end: s.end,
            known_samples: s.known_samples,
            values: projection.values.map(Feature::value),
            source_start: s.source_start,
            source_end: s.source_end,
            available_end: s.available_end,
        };
        if let Some(p) = previous
            && raw.values[3..6].iter().any(Option::is_some)
        {
            raw.source_start = raw.source_start.min(p.stamp.source_start);
            raw.source_end = raw.source_end.max(p.stamp.source_end);
            raw.available_end = raw.available_end.max(p.stamp.available_end);
        }
        // Accent floor sensitivity must not redefine the registered descriptor coordinates.
        let salience = raw.values[3].zip(raw.values[5]).map(|_| {
            let rise = (input
                .energy
                .unwrap()
                .sqrt()
                .max(self.config.rms_floor)
                .log2()
                - previous
                    .unwrap()
                    .energy
                    .unwrap()
                    .sqrt()
                    .max(self.config.rms_floor)
                    .log2())
            .max(0.);
            let components = [
                rise,
                projection.accent_flux / self.previous_scan.len() as f64,
            ];
            accent_salience(components, self.config.means, self.config.deviations)
        });
        if salience.is_some_and(|x| !x.is_finite()) {
            return Err("accent standardization overflow");
        }
        let detector = if self.len == 3 {
            let old = self.history.map(Option::unwrap);
            let stamps = [old[0].stamp, old[1].stamp, old[2].stamp, s];
            let coverage = stamps.iter().map(|s| s.known_samples).sum::<u64>() as f64
                / (s.end - stamps[0].start) as f64;
            let mut detection = Detection {
                acquisition_coverage: coverage,
                detector_coverage: false,
                saliences: None,
                status: Status::Unsupported,
                accent: None,
            };
            if stamps.iter().all(|t| {
                t.observed
                    && t.available_end <= cut
                    && t.known_samples == t.end - t.start
                    && t.group == s.group
                    && t.association == s.association
                    && t.association.is_some()
                    && t.grid_id == s.grid_id
            }) && stamps.windows(2).all(|p| p[0].end == p[1].start)
                && let (Some(left), Some(middle), Some(right)) =
                    (old[1].salience, old[2].salience, salience)
            {
                detection.detector_coverage = true;
                detection.saliences = Some([left, middle, right]);
                detection.status = accent_status([left, middle, right], self.config.threshold);
                if detection.status == Status::Admitted {
                    detection.accent = Some(Accent {
                        group: s.group,
                        event_start: old[2].stamp.start,
                        event_end: old[2].stamp.end,
                        raw_intervals: stamps.map(|t| (t.start, t.end)),
                        source_start: stamps.iter().map(|t| t.source_start).min().unwrap(),
                        source_end: stamps.iter().map(|t| t.source_end).max().unwrap(),
                        available_end: stamps.iter().map(|t| t.available_end).max().unwrap(),
                        weight: (middle - self.config.threshold).clamp(0., 1.),
                        observed_prefix: old[2].observed_prefix,
                    });
                }
            }
            Some(detection)
        } else {
            None
        };
        let saved = Saved {
            stamp: s,
            energy: input.energy,
            bus_energy: input.bus_energy,
            has_spectrum: input.energy_scan.is_some(),
            salience,
            observed_prefix,
        };
        if self.len == 3 {
            self.history.rotate_left(1);
            self.history[2] = Some(saved);
        } else {
            self.history[self.len] = Some(saved);
            self.len += 1;
        }
        if let Some(scan) = input.energy_scan {
            self.previous_scan.copy_from_slice(scan);
        }
        self.observed_prefix = observed_prefix;
        Ok(Some(Update { raw, detector }))
    }
}

#[cfg(test)]
mod tests;
