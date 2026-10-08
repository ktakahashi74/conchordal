//! Author rules: fixed first-hop mass fractions, signed means, and direct peaks.
use super::{CHANNELS, Frame, centers_hz, validate_config};
use crate::config::TemporalT2Config;
use crate::core::log2space::Log2Space;
use crate::temporal_cognition::features::{Accent, Detection, Stamp, Status, accent_status};
use crate::temporal_cognition::ridge::Handle;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Diagnostic {
    pub known: bool,
    pub unknown: Option<&'static str>,
    pub values: Option<[f64; 4]>,
}

#[derive(Clone, Copy)]
struct Saved {
    stamp: Stamp,
    frame: Frame,
    weights: [f64; CHANNELS],
    prefix: u64,
}

#[derive(Default)]
struct Stream {
    history: [Option<Saved>; 3],
    owner: Option<Handle>,
    prefix: u64,
}

pub(crate) struct Adapter {
    config: TemporalT2Config,
    native_bin: Vec<Option<usize>>,
    weights: [[f64; CHANNELS]; 7],
    streams: Box<[Stream; 7]>,
}

impl Adapter {
    pub(crate) fn new(space: &Log2Space, config: TemporalT2Config) -> Result<Self, &'static str> {
        validate_config(config)?;
        let centers = centers_hz();
        let erb = centers.map(|f| 9.2645 * (1.0 + f * 0.00437).ln());
        let edges: [f64; CHANNELS - 1] = std::array::from_fn(|i| (erb[i] + erb[i + 1]) / 2.0);
        let native_bin = space
            .centers_log2
            .iter()
            .map(|&x| {
                let f = f64::from(x).exp2();
                (80.0..=8000.0).contains(&f).then(|| {
                    let coordinate = 9.2645 * (1.0 + f * 0.00437).ln();
                    edges.partition_point(|&edge| edge <= coordinate)
                })
            })
            .collect();
        Ok(Self {
            config,
            native_bin,
            weights: [[0.0; CHANNELS]; 7],
            streams: Box::new(std::array::from_fn(|_| Stream::default())),
        })
    }

    pub(crate) fn clear(&mut self, index: usize) {
        self.streams[index] = Stream::default();
    }

    pub(crate) fn prepare(&mut self, space: &Log2Space, energy_scans: &[Vec<f64>; 8]) {
        space.assert_scan_len_named(&self.native_bin, "T2_native_mapping_scan");
        for scan in energy_scans {
            space.assert_scan_len_named(scan, "T2_group_mass_scan");
        }
        let mut mass = [[0.0; CHANNELS]; 8];
        for (bin, native) in self.native_bin.iter().enumerate() {
            if let Some(native) = native {
                for (group, scan) in mass.iter_mut().zip(energy_scans) {
                    group[*native] += scan[bin];
                }
            }
        }
        for band in 0..CHANNELS {
            let bus_mass: f64 = mass.iter().map(|group| group[band]).sum();
            for (group, weights) in self.weights.iter_mut().enumerate() {
                weights[band] = if bus_mass > 0.0 {
                    mass[group][band] / bus_mass
                } else {
                    0.0
                };
            }
        }
    }

    pub(crate) fn push(
        &mut self,
        index: usize,
        stamp: Stamp,
        frame: Option<&Frame>,
        cut: u64,
    ) -> Result<(Detection, Diagnostic), &'static str> {
        assert!(index < 7);
        self.streams[index].push(self.config, stamp, frame, self.weights[index], cut)
    }
}

impl Stream {
    fn push(
        &mut self,
        config: TemporalT2Config,
        stamp: Stamp,
        frame: Option<&Frame>,
        weights: [f64; CHANNELS],
        cut: u64,
    ) -> Result<(Detection, Diagnostic), &'static str> {
        let mut detection = Detection {
            acquisition_coverage: 0.0,
            detector_coverage: false,
            saliences: None,
            status: Status::Unsupported,
            accent: None,
        };
        if self.owner != Some(stamp.group) {
            self.history = [None; 3];
            self.prefix = 0;
            self.owner = Some(stamp.group);
        }
        if stamp.observed && stamp.association.is_some() {
            self.prefix += stamp.known_samples;
        }
        let reason = if !stamp.observed || stamp.known_samples != stamp.end - stamp.start {
            Some("t2_observation_missing")
        } else if stamp.association.is_none() {
            Some("t2_owner_association_unknown")
        } else if frame.is_none() {
            Some("t2_pcm_missing")
        } else if frame.is_some_and(|f| {
            f.start != stamp.start || f.end != stamp.end || f.epoch_start > f.start || f.end > cut
        }) {
            Some("t2_pcm_clock_mismatch")
        } else if !weights.iter().any(|&w| w > 0.0) {
            Some("t2_native_group_mass_absent")
        } else {
            None
        };
        if let Some(reason) = reason {
            self.history = [None; 3];
            return Ok((
                detection,
                Diagnostic {
                    known: false,
                    unknown: Some(reason),
                    values: None,
                },
            ));
        }
        let frame = *frame.unwrap();
        if weights
            .iter()
            .any(|w| !w.is_finite() || !(0.0..=1.0).contains(w))
            || frame.means.iter().flatten().any(|x| !x.is_finite())
        {
            return Err("invalid T2 group values or fractions");
        }
        if self.history[2].is_some_and(|s| {
            s.stamp.end != stamp.start
                || s.stamp.association != stamp.association
                || s.frame.epoch_start != frame.epoch_start
        }) {
            self.history = [None; 3];
        }
        let current = Saved {
            stamp,
            frame,
            weights,
            prefix: self.prefix,
        };
        let values =
            std::array::from_fn(|k| (0..CHANNELS).map(|b| weights[b] * frame.means[b][k]).sum());
        let mut diagnostic = Diagnostic {
            known: true,
            unknown: None,
            values: Some(values),
        };
        if let [Some(first), Some(left), Some(middle)] = self.history {
            let saved = [first, left, middle, current];
            let stamps = saved.map(|x| x.stamp);
            if !stamps.windows(2).all(|s| s[0].end == s[1].start)
                || !stamps.iter().all(|s| {
                    s.group == stamp.group
                        && s.association == stamp.association
                        && s.grid_id == stamp.grid_id
                        && s.available_end <= cut
                })
                || !saved
                    .iter()
                    .all(|s| s.frame.epoch_start == frame.epoch_start)
            {
                diagnostic.unknown = Some("t2_four_hop_support_missing");
            } else {
                // Freeze attribution across the peak window; moving mass cannot create a rise.
                let grouped: [[f64; 4]; 4] = saved.map(|s| {
                    std::array::from_fn(|k| {
                        (0..CHANNELS)
                            .map(|b| first.weights[b] * s.frame.means[b][k])
                            .sum()
                    })
                });
                let saliences: [f64; 3] = std::array::from_fn(|j| {
                    config.gain
                        * (0..4)
                            .map(|k| {
                                config.component_weights[k]
                                    * (grouped[j + 1][k] - grouped[j][k]).max(0.0)
                            })
                            .sum::<f64>()
                });
                if saliences.iter().any(|s| !s.is_finite()) {
                    return Err("T2 event combination overflow");
                }
                detection.acquisition_coverage = 1.0;
                detection.detector_coverage = true;
                detection.saliences = Some(saliences);
                detection.status = accent_status(saliences, config.threshold);
                if detection.status == Status::Admitted {
                    detection.accent = Some(Accent {
                        group: stamp.group,
                        event_start: middle.stamp.start,
                        event_end: middle.stamp.end,
                        raw_intervals: stamps.map(|s| (s.start, s.end)),
                        source_start: saved
                            .iter()
                            .map(|s| s.stamp.source_start.min(s.frame.epoch_start))
                            .min()
                            .unwrap(),
                        source_end: stamps.iter().map(|s| s.source_end).max().unwrap(),
                        available_end: stamps.iter().map(|s| s.available_end).max().unwrap(),
                        weight: ((saliences[1] - config.threshold) * config.weight_gain)
                            .clamp(0.0, 1.0),
                        observed_prefix: middle.prefix,
                    });
                }
            }
        } else {
            diagnostic.unknown = Some("t2_four_hop_support_warming");
        }
        self.history.rotate_left(1);
        self.history[2] = Some(current);
        Ok((detection, diagnostic))
    }
}

#[cfg(test)]
mod tests;
