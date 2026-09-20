//! One acoustic hop through trajectory, ridge, assignment, grouping, and lifecycle.

use super::{Candidate, lifecycle, producer};
use crate::core::log2space::Log2Space;
use crate::temporal_cognition::{features, group, grouping, ridge, trajectory};

#[derive(Clone, Copy)]
pub(super) struct Config {
    pub group: group::Config,
    pub ridge: ridge::Config,
    pub epoch_start: u64,
    pub group_retirement_samples: u64,
    pub inactive_energy_max: f64,
    pub correlation_window_samples: u64,
    pub min_pairs: usize,
    pub min_coverage: f64,
    pub persistence_hops: u8,
    pub features: features::Config,
}

#[derive(Clone, Copy)]
pub(in crate::temporal_cognition) struct Observation<'a> {
    pub power_scan: &'a [f32],
    pub mono_energy: f64,
    pub source_start: u64,
    pub source_end: u64,
    pub available_end: u64,
}

pub(crate) struct Frontend {
    space: Log2Space,
    config: Config,
    ridge: ridge::Tracker,
    window: grouping::Window,
    producer: producer::Generator,
    lifecycle: lifecycle::Lifecycle,
    energy_scans: [Vec<f64>; 8],
    energy_handles: [Option<ridge::Handle>; 8],
    energy_eligible: [bool; 8],
    feature_streams: [features::Stream; 8],
    shape_supported: bool,
    last_end: u64,
    failure: Option<&'static str>,
}

pub(in crate::temporal_cognition) struct Output {
    pub correlation_window: [u64; 2],
    pub partition: Option<trajectory::TrajectoryFrame>,
    pub ridges: ridge::Update,
    pub(super) proposals: producer::Output,
    pub(super) groups: lifecycle::Update,
    pub energy: Option<group::Energy>,
    pub features: [Option<features::Update>; 8],
    pub feature_gaps: [Option<features::RawDescriptor>; 8],
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Snapshot {
    pub assignment: group::Assignment,
    pub retained_groups: [Option<ridge::Handle>; 7],
    pub correlation_window: [u64; 2],
    pub group_handles: [Option<ridge::Handle>; 8],
    pub eligible: [bool; 8],
    pub energy: Option<[f64; 8]>,
    pub spectral_shape_supported: bool,
    pub features: [Option<features::Update>; 8],
    pub feature_gaps: [Option<features::RawDescriptor>; 8],
    pub admissions: usize,
    pub admissions_by_kind: [usize; 3],
    pub rejections: usize,
    pub retired: [Option<RetirementSnapshot>; 7],
    pub superseded: usize,
    pub pending_proposals: usize,
    pub proposal_conflicts: usize,
    pub candidate_count: usize,
    pub continued_candidates: usize,
    pub bundle_count: usize,
    pub correlation_reads: usize,
    pub cross_pair_reads: usize,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct RetirementSnapshot {
    pub handle: ridge::Handle,
    pub capacity: bool,
    pub last_active_end: u64,
    pub inactive_samples: u64,
}

#[cfg(test)]
pub(super) struct EnergySnapshot<'a> {
    pub end_sample: u64,
    pub group_handles: [Option<ridge::Handle>; 8],
    pub eligible: [bool; 8],
    pub scans: &'a [Vec<f64>; 8],
}

impl Frontend {
    pub(crate) fn configured(
        space: Log2Space,
        bus: u8,
        (epoch, epoch_start): (u64, u64),
        sample_rate: u32,
        hop: u64,
        ridge: crate::config::TemporalRidgeConfig,
        acoustic: crate::config::TemporalAcousticConfig,
    ) -> Result<Self, &'static str> {
        let window = (acoustic.correlation_window_sec * f64::from(sample_rate)).ceil();
        let retirement = (acoustic.group_retirement_sec * f64::from(sample_rate)).ceil();
        if [window, retirement]
            .iter()
            .any(|v| !v.is_finite() || *v < 1.0 || *v >= u64::MAX as f64)
        {
            return Err("invalid acoustic diagnostic durations");
        }
        Self::new(
            space,
            Config {
                group: group::Config {
                    bus,
                    epoch,
                    sample_rate,
                    hop,
                    means: acoustic.group_means,
                    deviations: acoustic.group_deviations,
                    distance_limit: 1.0,
                    residual_raw: (-1.0_f64).exp(),
                },
                ridge: ridge::Config {
                    means: ridge.means,
                    deviations: ridge.deviations,
                    distance_limit: 1.0,
                    retirement_sec: 0.25,
                },
                epoch_start,
                group_retirement_samples: retirement as u64,
                inactive_energy_max: acoustic.inactive_energy_max,
                correlation_window_samples: window as u64,
                min_pairs: acoustic.min_pairs,
                min_coverage: acoustic.min_coverage,
                persistence_hops: acoustic.persistence_hops,
                features: features::Config {
                    means: acoustic.accent_means,
                    deviations: acoustic.accent_deviations,
                    threshold: 1.0,
                    rms_floor: 1e-6,
                },
            },
        )
    }

    pub(in crate::temporal_cognition) fn snapshot(&self, out: &Output) -> Snapshot {
        let mut admissions_by_kind = [0; 3];
        for admission in out
            .groups
            .admissions
            .iter()
            .flatten()
            .filter(|a| a.rejection.is_none())
        {
            let index = match admission.key.kind {
                super::Kind::Birth => 0,
                super::Kind::Split => 1,
                super::Kind::Merge => 2,
            };
            admissions_by_kind[index] += 1;
        }
        Snapshot {
            assignment: out.groups.assignment,
            retained_groups: self.lifecycle.retained_handles(),
            correlation_window: out.correlation_window,
            group_handles: self.energy_handles,
            eligible: self.energy_eligible,
            energy: out.energy.as_ref().map(|e| e.group_energy),
            spectral_shape_supported: self.shape_supported,
            features: out.features,
            feature_gaps: out.feature_gaps,
            admissions: out
                .groups
                .admissions
                .iter()
                .flatten()
                .filter(|a| a.rejection.is_none())
                .count(),
            admissions_by_kind,
            rejections: out
                .groups
                .admissions
                .iter()
                .flatten()
                .filter(|a| a.rejection.is_some())
                .count(),
            retired: out.groups.retired.map(|r| {
                r.map(|r| RetirementSnapshot {
                    handle: r.group.handle,
                    capacity: r.reason == lifecycle::Retirement::Capacity,
                    last_active_end: r.last_active_end,
                    inactive_samples: r.inactive_samples,
                })
            }),
            superseded: out.groups.superseded.iter().flatten().count(),
            pending_proposals: out.proposals.proposals.pending,
            proposal_conflicts: out.proposals.proposals.conflicts,
            candidate_count: out.proposals.candidates.iter().flatten().count(),
            continued_candidates: out.proposals.continued,
            bundle_count: out.proposals.bundles.count,
            correlation_reads: out.proposals.bundles.correlation_reads,
            cross_pair_reads: out.proposals.cross_pair_reads,
        }
    }

    pub(super) fn new(space: Log2Space, config: Config) -> Result<Self, &'static str> {
        let g = config.group;
        let ridge = ridge::Tracker::new(g.bus, g.epoch, g.sample_rate, g.hop, config.ridge)?;
        let window = grouping::Window::new(
            g.bus,
            g.epoch,
            config.epoch_start,
            g.hop,
            config.correlation_window_samples,
            config.min_pairs,
            config.min_coverage,
        )?;
        let producer = producer::Generator::new(
            g.bus,
            g.epoch,
            config.epoch_start,
            g.hop,
            config.persistence_hops,
        )?;
        let lifecycle = lifecycle::Lifecycle::new(
            g.bus,
            g.epoch,
            config.epoch_start,
            g.hop,
            config.group_retirement_samples,
            config.inactive_energy_max,
        )?;
        // Validate the frozen group scales even before any observed frame arrives.
        let mut empty = [None; 7];
        let first_end = config
            .epoch_start
            .checked_add(g.hop)
            .ok_or("acoustic source clock exhausted")?;
        group::assign_and_refresh(&g, first_end, &[None; 8], &mut empty)?;
        let energy_scans = std::array::from_fn(|_| vec![0.0; space.n_bins()]);
        let mut feature_streams = Vec::with_capacity(8);
        for _ in 0..8 {
            feature_streams.push(features::Stream::new(space.n_bins(), config.features)?);
        }
        let feature_streams = feature_streams
            .try_into()
            .unwrap_or_else(|_| unreachable!());
        Ok(Self {
            space,
            config,
            ridge,
            window,
            producer,
            lifecycle,
            energy_scans,
            energy_handles: [None; 8],
            energy_eligible: [false; 8],
            feature_streams,
            shape_supported: false,
            last_end: config.epoch_start,
            failure: None,
        })
    }

    pub(in crate::temporal_cognition) fn advance(
        &mut self,
        end: u64,
        observation: Option<Observation<'_>>,
    ) -> Result<Output, &'static str> {
        if self.failure.is_some() {
            return Err("acoustic epoch has failed");
        }
        self.shape_supported = false;
        let result = self.process(end, observation);
        if let Err(error) = result {
            self.failure = Some(error);
        }
        result
    }

    fn process(
        &mut self,
        end: u64,
        observation: Option<Observation<'_>>,
    ) -> Result<Output, &'static str> {
        if end <= self.last_end || !end.is_multiple_of(self.config.group.hop) {
            return Err("noncausal acoustic endpoint");
        }
        let start = end - self.config.group.hop;
        if observation.is_some_and(|o| {
            o.source_start > start || o.source_end < end || o.available_end < o.source_end
        }) {
            return Err("invalid acoustic source support or availability");
        }
        let partition = observation
            .map(|o| {
                trajectory::partition(&self.space, o.power_scan, o.mono_energy)
                    .ok_or("invalid acoustic power or mono energy")
            })
            .transpose()?;
        let mut points = [ridge::Point {
            frequency_log2: None,
            log_envelope: None,
        }; 7];
        let mut peak_count = 0;
        if let Some(frame) = partition {
            for (i, bin) in frame
                .peak_bins
                .iter()
                .enumerate()
                .filter_map(|(i, p)| p.map(|p| (i, p)))
            {
                points[i] = ridge::Point {
                    frequency_log2: Some(f64::from(self.space.centers_log2[bin])),
                    log_envelope: Some(frame.log_envelope[i]),
                };
                peak_count += 1;
            }
        }
        let ridges = self.ridge.advance(
            end,
            partition
                .as_ref()
                .filter(|p| p.spectral_shape_supported)
                .map(|_| &points[..peak_count]),
        )?;
        let mut current: [Option<group::Trajectory>; 8] = [None; 8];
        let mut envelopes = [None; 8];
        let mut eligible = [None; 8];
        if let Some(frame) = partition {
            for (i, ridge) in ridges
                .current
                .iter()
                .enumerate()
                .filter_map(|(i, r)| r.map(|r| (i, r)))
            {
                current[i] = Some(group::Trajectory {
                    handle: ridge.handle,
                    point: ridge.point,
                    slopes: ridge.slopes,
                });
                envelopes[i] = Some(grouping::Envelope {
                    handle: ridge.handle,
                    log_envelope: frame.log_envelope[i],
                });
                if frame.energy[i] > 0.0 {
                    eligible[i] = Some(ridge.handle);
                }
            }
            // Ridge allocation checks increment before issuance and cannot issue u64::MAX.
            let residual = ridge::Handle {
                bus: self.config.group.bus,
                epoch: self.config.group.epoch,
                generation: u64::MAX,
            };
            current[7] = Some(group::Trajectory {
                handle: residual,
                point: ridge::Point {
                    frequency_log2: None,
                    log_envelope: Some(frame.log_envelope[7]),
                },
                slopes: [None; 2],
            });
            envelopes[7] = Some(grouping::Envelope {
                handle: residual,
                log_envelope: frame.log_envelope[7],
            });
            if frame.energy[7] > 0.0 {
                eligible[7] = Some(residual);
            }
        }
        self.window.push(end, envelopes)?;
        let mut handles = [ridge::Handle {
            bus: 0,
            epoch: 0,
            generation: 0,
        }; 8];
        let mut count = 0;
        for &h in eligible.iter().flatten() {
            handles[count] = h;
            count += 1;
        }
        let correlations = self.window.correlations(&handles[..count])?;
        let prepared = self.lifecycle.prepare(&self.config.group, end, &current)?;
        let energy = if let (Some(o), Some(frame)) = (observation, partition) {
            group::allocate_energy(
                &self.space,
                o.power_scan,
                Some(o.mono_energy),
                &frame.peak_bins,
                &prepared.assignment,
                &mut self.energy_scans,
            )?
        } else {
            None
        };
        let mut parents = [ridge::Handle {
            bus: 0,
            epoch: 0,
            generation: 0,
        }; 7];
        let mut parent_count = 0;
        for h in self.lifecycle.eligible_handles().into_iter().flatten() {
            parents[parent_count] = h;
            parent_count += 1;
        }
        let proposals = self.producer.advance(
            observation.is_some(),
            &correlations,
            &prepared.assignment,
            &parents[..parent_count],
        )?;
        let mut accepted: [Option<Candidate>; 8] = [None; 8];
        let mut accepted_count = 0;
        for c in proposals.proposals.accepted.iter().flatten() {
            accepted[accepted_count] = Some(*c);
            accepted_count += 1;
        }
        let group_energy = std::array::from_fn(|i| {
            energy
                .as_ref()
                .filter(|e| e.spectral_shape_supported)
                .map(|e| e.group_energy[i])
        });
        let groups = if accepted_count == 0 {
            self.lifecycle.commit(prepared, group_energy, &[])?
        } else {
            let mut packed = [accepted[0].unwrap(); 8];
            for (dst, c) in packed
                .iter_mut()
                .zip(accepted[..accepted_count].iter().flatten())
            {
                *dst = *c;
            }
            self.lifecycle
                .commit(prepared, group_energy, &packed[..accepted_count])?
        };
        self.energy_handles[..7].copy_from_slice(&groups.assignment.group_handles);
        self.energy_handles[7] = Some(ridge::Handle {
            bus: self.config.group.bus,
            epoch: self.config.group.epoch,
            generation: 1,
        });
        self.shape_supported = energy.as_ref().is_some_and(|e| e.spectral_shape_supported);
        let mut features = std::array::from_fn(|_| None);
        let mut feature_gaps = [None; 8];
        for (i, stream) in self.feature_streams.iter_mut().enumerate() {
            let Some(handle) = self.energy_handles[i] else {
                stream.clear();
                self.energy_eligible[i] = false;
                continue;
            };
            self.energy_eligible[i] = i == 7 || parents[..parent_count].contains(&handle);
            let eligible = self.energy_eligible[i];
            // A dropped span is one missing record, never fabricated silent hops.
            if observation.is_some() && self.last_end < start {
                feature_gaps[i] = stream
                    .push(
                        &self.space,
                        features::Input {
                            stamp: features::Stamp {
                                group: handle,
                                association: eligible.then_some(handle.generation),
                                grid_id: 1,
                                start: self.last_end,
                                end: start,
                                source_start: self.last_end,
                                source_end: start,
                                available_end: start,
                                known_samples: 0,
                                observed: false,
                            },
                            energy: None,
                            bus_energy: None,
                            energy_scan: None,
                        },
                        start,
                    )?
                    .map(|u| u.raw);
            }
            let feature_start = if observation.is_some() {
                start
            } else {
                self.last_end
            };
            let (source_start, source_end, available_end) = observation
                .map_or((feature_start, end, end), |o| {
                    (o.source_start, o.source_end, o.available_end)
                });
            features[i] = stream.push(
                &self.space,
                features::Input {
                    stamp: features::Stamp {
                        group: handle,
                        association: (eligible && (self.shape_supported || i == 7))
                            .then_some(handle.generation),
                        grid_id: 1,
                        start: feature_start,
                        end,
                        source_start,
                        source_end,
                        available_end,
                        known_samples: if observation.is_some() {
                            self.config.group.hop
                        } else {
                            0
                        },
                        observed: observation.is_some(),
                    },
                    energy: energy
                        .as_ref()
                        .filter(|e| e.spectral_shape_supported || i == 7)
                        .map(|e| e.group_energy[i]),
                    bus_energy: observation.map(|o| o.mono_energy),
                    energy_scan: self
                        .shape_supported
                        .then_some(self.energy_scans[i].as_slice()),
                },
                available_end,
            )?;
        }
        self.last_end = end;
        Ok(Output {
            correlation_window: [correlations.window_start, correlations.window_end],
            partition,
            ridges,
            proposals,
            groups,
            energy,
            features,
            feature_gaps,
        })
    }

    #[cfg(test)]
    pub(super) fn energy_snapshot(&self) -> Option<EnergySnapshot<'_>> {
        if self.failure.is_some() || !self.shape_supported {
            return None;
        }
        for scan in &self.energy_scans {
            self.space
                .assert_scan_len_named(scan, "acoustic_group_energy_scan");
        }
        Some(EnergySnapshot {
            end_sample: self.last_end,
            group_handles: self.energy_handles,
            eligible: self.energy_eligible,
            scans: &self.energy_scans,
        })
    }
}

pub(crate) mod recurrence;
#[cfg(test)]
mod tests;
