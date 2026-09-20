//! Immutable conditional descriptor profiles. No hypothetical frame enters observation memory.

use super::feature_projection::{Feature, window};
use super::{body_model, context, ridge::Handle};
use crate::config::{AppConfig, TemporalActionProfilesConfig};
use crate::life::action_candidates::Class;
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::io::Read;

pub(crate) mod consumer;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Cell {
    pub prototype: usize,
    pub class: Class,
    pub group: Handle,
    pub issued_at: u64,
    pub action_at: u64,
    pub evaluation_at: u64,
    pub window_start: u64,
    pub held_background_energy: Option<f64>,
    pub issue_log_rms: Option<f64>,
    pub(super) features: window::Summary,
    pub articulation: Option<super::gesture::Projection>,
    /// Quarter-second window of the first four raw descriptor coordinates.
    pub(super) short_features: [Feature; 4],
    pub short_observed_fraction: [f64; 4],
    pub short_projected_fraction: [f64; 4],
    pub(super) grouping: Feature,
    pub grouping_observed_fraction: f64,
    pub arrival: Option<super::arrival::Projection>,
    pub accent_density: super::context::AccentDensity,
    /// Arrival probability, accent density and grouping support at the evaluated time.
    pub(super) context_features: [Feature; 3],
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Snapshot {
    pub profile_sha256: [u8; 32],
    pub body_model_version: [u8; 32],
    pub routed_prototypes: u8,
    pub bus: u8,
    pub epoch: u64,
    pub issued_at: u64,
    pub assignment_end: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation_delay_samples: Option<u64>,
    pub groups: [Option<Handle>; 8],
    pub issue_observed_coverage: f64,
    pub arrival_issues: [Option<super::arrival::Frozen>; 7],
    pub profile_bytes: usize,
    pub table_bytes: usize,
    pub cells: usize,
    pub supported_coordinates: usize,
    pub projected_coordinates: usize,
    pub articulation_supported_cells: usize,
    pub projected_accent_density_cells: usize,
    pub projected_arrival_cells: usize,
    pub latest: [Option<Cell>; 7],
}

pub(crate) struct Table {
    cells: Vec<Option<Cell>>,
    snapshot: Box<Option<Snapshot>>,
    last_build: Option<([u8; 32], u8, u64, u64, Option<u64>)>,
    resources: Resources,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(crate) struct Resources {
    pub refresh_calls: u64,
    pub rebuilds: u64,
    pub builds_by_assigned_prototypes: [u64; 9],
    pub invalidations: u64,
    pub deferred_calls: u64,
    pub rejected_inputs: u64,
    pub attempted_cells: u64,
    pub evaluation_unavailable_cells: u64,
    pub projection_calls: u64,
    pub reused_cells: u64,
    pub articulation_frames: u64,
    pub reused_articulation_frames: u64,
    pub last_build_sample: Option<u64>,
    pub last_build_us: u64,
    pub max_build_us: u64,
    pub total_build_us: u64,
}

impl Table {
    // Keep the large empty snapshot's initialization out of the worker's lasting stack frame.
    #[inline(never)]
    pub(crate) fn new() -> Self {
        Self {
            cells: vec![None; 8 * 7 * 32],
            snapshot: Box::new(None),
            last_build: None,
            resources: Resources::default(),
        }
    }

    pub(crate) fn resources(&self) -> Resources {
        self.resources
    }

    pub(crate) fn refresh(
        &mut self,
        profiles: &Profiles,
        context: &context::Context,
        shared: &body_model::Shared,
        articulation: &super::gesture::Gesture,
        recurrence: Option<&super::proposals::frontend::recurrence::Recurrence>,
    ) -> Option<Snapshot> {
        let started = std::time::Instant::now();
        self.resources.refresh_calls += 1;
        let current = context.snapshot();
        let same_clock = self.last_build.is_some_and(|(profile, bus, epoch, _, _)| {
            profile == profiles.sha256 && bus == shared.bus && epoch == shared.epoch
        });
        if current.sample_rate != 48_000
            || profiles.body_model_version != shared.model_version
            || shared.end_sample > current.end_sample
            || (same_clock && self.last_build.unwrap().3 > current.end_sample)
        {
            self.resources.rejected_inputs += 1;
            if self.snapshot.take().is_some() {
                self.resources.invalidations += 1;
                self.cells.fill(None);
            }
            return None;
        }
        let groups = shared.assignments.map(|a| {
            a.map(|a| Handle {
                bus: shared.bus,
                epoch: a.key.0,
                generation: a.key.1,
            })
        });
        if self.snapshot.is_some_and(|s| {
            s.profile_sha256 == profiles.sha256
                && s.evaluation_delay_samples == profiles.evaluation_delay_samples
                && s.bus == shared.bus
                && s.epoch == shared.epoch
                && s.assignment_end == shared.end_sample
                && s.groups == groups
                && s.issued_at <= current.end_sample
        }) {
            return *self.snapshot;
        }
        // Retirement invalidates the whole frozen table, not just its selected group.
        if let Some(previous) = self.snapshot.take() {
            self.resources.invalidations += u64::from(
                previous.groups != groups
                    || previous.evaluation_delay_samples != profiles.evaluation_delay_samples
                    || previous.profile_sha256 != profiles.sha256
                    || previous.bus != shared.bus
                    || previous.epoch != shared.epoch,
            );
            self.cells.fill(None);
        }
        if same_clock
            && self.last_build.unwrap().4 == profiles.evaluation_delay_samples
            && current.end_sample - self.last_build.unwrap().3
                < u64::from(current.sample_rate).div_ceil(10)
        {
            self.resources.deferred_calls += 1;
            return None;
        }
        self.last_build = Some((
            profiles.sha256,
            shared.bus,
            shared.epoch,
            current.end_sample,
            profiles.evaluation_delay_samples,
        ));
        self.resources.rebuilds += 1;
        let assigned = groups.iter().take(profiles.count()).flatten().count();
        self.resources.builds_by_assigned_prototypes[assigned] += 1;
        self.resources.last_build_sample = Some(current.end_sample);
        let mut snapshot = Snapshot {
            profile_sha256: profiles.sha256,
            body_model_version: profiles.body_model_version,
            routed_prototypes: profiles
                .profiles
                .iter()
                .enumerate()
                .fold(0, |mask, (i, p)| mask | (u8::from(p.source_routed) << i)),
            bus: shared.bus,
            epoch: shared.epoch,
            issued_at: current.end_sample,
            assignment_end: shared.end_sample,
            evaluation_delay_samples: profiles.evaluation_delay_samples,
            groups,
            issue_observed_coverage: current.observed_coverage,
            arrival_issues: recurrence.map_or([None; 7], |r| r.arrival_issues(current.end_sample)),
            profile_bytes: profiles.stored_bytes(),
            table_bytes: std::mem::size_of::<Self>()
                + std::mem::size_of::<Option<Snapshot>>()
                + self.cells.capacity() * std::mem::size_of::<Option<Cell>>(),
            cells: 0,
            supported_coordinates: 0,
            projected_coordinates: 0,
            articulation_supported_cells: 0,
            projected_accent_density_cells: 0,
            projected_arrival_cells: 0,
            latest: [None; 7],
        };
        let mut projection_cache = super::gesture::ProjectionCache::default();
        for (prototype, group) in groups.iter().enumerate().take(profiles.count()) {
            let Some(group) = *group else { continue };
            for (class_index, class) in CLASSES.into_iter().enumerate() {
                for (index, offset) in profiles.offsets().iter().copied().enumerate() {
                    self.resources.attempted_cells += 1;
                    let Some(evaluation_at) = profiles.evaluation_at(current.end_sample, offset)
                    else {
                        self.resources.evaluation_unavailable_cells += 1;
                        continue;
                    };
                    // Only identical whole trajectories at the same group and time share work.
                    // Earlier classes belong to this build; no result survives issue invalidation.
                    let trajectory = profiles.profiles[prototype].trajectories[class_index][index];
                    let reused = trajectory.and_then(|trajectory| {
                        (0..class_index).find_map(|earlier| {
                            (profiles.profiles[prototype].trajectories[earlier][index]
                                == Some(trajectory))
                            .then(|| self.cells[(prototype * 7 + earlier) * 32 + index])
                            .flatten()
                        })
                    });
                    let cell = if let Some(mut cell) = reused {
                        self.resources.reused_cells += 1;
                        cell.class = class;
                        Some(cell)
                    } else {
                        self.resources.projection_calls += 1;
                        #[cfg(test)]
                        let mut cost_clock = super::context::tests::projection_clock();
                        if let Some(trajectory) = trajectory {
                            projection_cache.retag_matching_prefix(
                                trajectory,
                                current.end_sample + offset,
                                |old, frames| {
                                    let count = frames as usize;
                                    let a = &profiles.frames[old * FRAMES..][..count];
                                    let b = &profiles.frames[trajectory * FRAMES..][..count];
                                    a.iter().zip(b).all(|(a, b)| {
                                        a.mask == b.mask
                                            && a.values
                                                .iter()
                                                .zip(b.values)
                                                .all(|(a, b)| a.to_bits() == b.to_bits())
                                    })
                                },
                            );
                        }
                        #[cfg(test)]
                        super::context::tests::record_projection_cost(&mut cost_clock, 6);
                        context.project_action_window(
                            profiles,
                            prototype,
                            class,
                            offset,
                            evaluation_at,
                            group,
                            articulation.rms_reference(),
                            Some(articulation),
                            snapshot
                                .arrival_issues
                                .iter()
                                .flatten()
                                .find(|f| f.group == group),
                            trajectory.map(|id| (id, &mut projection_cache)),
                        )
                    };
                    let Some(cell) = cell else {
                        continue;
                    };
                    self.cells[(prototype * 7 + class_index) * 32 + index] = Some(cell);
                    snapshot.cells += 1;
                    snapshot.projected_arrival_cells +=
                        usize::from(matches!(cell.context_features[0], Feature::Projected(_)));
                    snapshot.projected_accent_density_cells +=
                        usize::from(matches!(cell.accent_density.value, Feature::Projected(_)));
                    snapshot.articulation_supported_cells += usize::from(
                        cell.articulation
                            .is_some_and(|p| p.states.iter().sum::<f64>() > 0.),
                    );
                    snapshot.supported_coordinates += cell
                        .features
                        .values
                        .iter()
                        .filter(|v| v.value().is_some())
                        .count();
                    snapshot.projected_coordinates += cell
                        .features
                        .values
                        .iter()
                        .filter(|v| matches!(v, Feature::Projected(_)))
                        .count();
                    snapshot.latest[class_index] = Some(cell);
                }
            }
        }
        self.resources.articulation_frames += projection_cache.advanced_frames;
        snapshot.table_bytes = std::mem::size_of::<Self>()
            + std::mem::size_of::<Option<Snapshot>>()
            + self.cells.capacity() * std::mem::size_of::<Option<Cell>>();
        self.resources.reused_articulation_frames += projection_cache.reused_frames;
        self.resources.last_build_us =
            started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        self.resources.max_build_us = self
            .resources
            .max_build_us
            .max(self.resources.last_build_us);
        self.resources.total_build_us += self.resources.last_build_us;
        *self.snapshot = Some(snapshot);
        *self.snapshot
    }
}

pub(crate) const CLASSES: [Class; 7] = [
    Class::OnsetNow,
    Class::DelayedOnset,
    Class::Wait,
    Class::Skip,
    Class::Continue,
    Class::Release,
    Class::Gap,
];
const NAMES: [&str; 7] = [
    "onset_now",
    "delayed_onset",
    "wait",
    "skip",
    "continue",
    "release",
    "gap",
];
const HORIZON: u64 = 192_000;
const FRAMES: usize = 375;
const MAX_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone, Copy)]
struct Frame {
    mask: u32,
    values: [f64; 11],
}

struct Profile {
    source_routed: bool,
    trajectories: [[Option<usize>; 32]; 7],
}

pub(crate) struct Profiles {
    pub(crate) sha256: [u8; 32],
    pub(crate) body_model_version: [u8; 32],
    pub(super) accent_means: [f64; 2],
    pub(super) accent_deviations: [f64; 2],
    evaluation_delay_samples: Option<u64>,
    offsets: [u64; 32],
    profiles: Vec<Profile>,
    frames: Box<[Frame]>,
}

pub(crate) fn validate_config(config: &AppConfig) -> Result<()> {
    if let Some(profile) = &config.temporal_action_profiles {
        ensure!(
            profile.evaluation_delay_ms.is_none_or(|ms| ms <= 4000),
            "action profile evaluation delay must be within 0..=4000 ms"
        );
        ensure!(
            !profile.file.is_empty()
                && profile.sha256.len() == 64
                && profile
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "temporal_action_profiles requires a file and lowercase SHA-256"
        );
        ensure!(
            config.temporal_body_prototypes.is_some(),
            "temporal_action_profiles requires body prototypes"
        );
    }
    Ok(())
}

impl Profiles {
    pub(crate) fn load(config: &AppConfig, spec: &TemporalActionProfilesConfig) -> Result<Self> {
        let mut bytes = Vec::new();
        std::fs::File::open(&spec.file)
            .with_context(|| format!("open action profiles {}", spec.file))?
            .take((MAX_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() <= MAX_BYTES,
            "action profile file exceeds 32 MiB"
        );
        Self::decode(config, spec, &bytes)
    }

    fn decode(
        config: &AppConfig,
        spec: &TemporalActionProfilesConfig,
        bytes: &[u8],
    ) -> Result<Self> {
        body_model::validate(config)?;
        ensure!(
            spec.evaluation_delay_ms.is_none_or(|ms| ms <= 4000),
            "action profile evaluation delay must be within 0..=4000 ms"
        );
        let sha256: [u8; 32] = Sha256::digest(bytes).into();
        ensure!(
            format!("{:x}", Sha256::digest(bytes)) == spec.sha256,
            "action profile SHA-256 mismatch"
        );
        ensure!(
            bytes.len() >= 16 && &bytes[..8] == b"I10AP001",
            "invalid action profile header"
        );
        let header_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let trajectories = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        ensure!(
            header_len <= 1024 * 1024
                && (1..=8 * 129).contains(&trajectories)
                && bytes.len() == 16 + header_len + trajectories * FRAMES * 96,
            "invalid action profile dimensions"
        );
        let header: serde_json::Value = serde_json::from_slice(&bytes[16..16 + header_len])?;
        ensure!(
            header.get("routing_projection").is_none(),
            "route-conditioned research profiles require an actual-body routing consumer"
        );
        let model = config
            .temporal_body_prototypes
            .as_ref()
            .context("action profiles need body prototypes")?;
        ensure!(
            header["schema"] == "i10-action-profiles-v1"
                && header["body_model_version"] == model.model_version,
            "action profile body model mismatch"
        );
        ensure!(
            header["sample_rate"] == config.audio.sample_rate
                && header["nfft"] == config.analysis.nfft
                && header["hop_samples"] == config.analysis.hop_size
                && config.audio.sample_rate == 48000
                && config.analysis.nfft == 2048
                && config.analysis.hop_size == 512
                && header["horizon_samples"] == HORIZON
                && header["frames_per_trajectory"] == FRAMES
                && header["trajectory_count"] == trajectories,
            "action profile physical grid mismatch"
        );
        ensure!(
            header["classes"] == serde_json::json!(NAMES),
            "action profile class order mismatch"
        );
        let offsets = std::array::from_fn(|i| (i as u64 * HORIZON + 15) / 31);
        ensure!(
            header["action_offsets"] == serde_json::json!(offsets),
            "action profile offset grid mismatch"
        );
        let profiles = header["profiles"]
            .as_array()
            .context("missing action profiles")?;
        ensure!(
            profiles.len() == model.medoids.len() && (1..=8).contains(&profiles.len()),
            "action profile medoid count mismatch"
        );
        let mut bindings = Vec::with_capacity(profiles.len());
        for (profile, medoid) in profiles.iter().zip(&model.medoids) {
            ensure!(
                profile["record_id"] == medoid.record_id,
                "action profile medoid order mismatch"
            );
            let descriptor = &profile["source_descriptor"];
            let source_bus = profile["source_bus"]
                .as_u64()
                .filter(|bus| *bus < 2)
                .context("invalid action profile source bus")?;
            ensure!(
                descriptor["bus"].as_u64() == Some(source_bus),
                "action profile descriptor source bus mismatch"
            );
            let source_routed = match profile["recipe"]["routing"].as_str() {
                Some("both") => true,
                Some("habitat") => source_bus == 0,
                Some("presentation") => source_bus == 1,
                _ => anyhow::bail!("invalid action profile source routing"),
            };
            ensure!(
                descriptor["mask"] == medoid.mask
                    && descriptor["raw_values"]
                        .as_array()
                        .is_some_and(|values| values.len() == 6
                            && values
                                .iter()
                                .zip(medoid.raw_values)
                                .all(|(actual, expected)| actual
                                    .as_f64()
                                    .is_some_and(|v| v.to_bits() == expected.to_bits()))),
                "action profile medoid descriptor mismatch"
            );
            let classes = profile["trajectories"]
                .as_array()
                .context("missing class trajectories")?;
            ensure!(classes.len() == 7, "wrong action profile class count");
            let mut slots = [[None; 32]; 7];
            for (class, rows) in classes.iter().enumerate() {
                let rows = rows.as_array().context("missing action-time rows")?;
                ensure!(rows.len() == 32, "wrong action-time row count");
                for (cell, row) in rows.iter().enumerate() {
                    if row.is_null() {
                        continue;
                    }
                    let index = row.as_u64().context("invalid action trajectory index")?;
                    ensure!(
                        index < trajectories as u64,
                        "action trajectory index out of range"
                    );
                    let expected = match class {
                        0 | 3 | 4 => cell == 0,
                        1 | 2 => cell > 0,
                        _ => true,
                    };
                    ensure!(expected, "illegal class/time profile binding");
                    slots[class][cell] = Some(index as usize);
                }
            }
            bindings.push(Profile {
                source_routed,
                trajectories: slots,
            });
        }
        let mut frames = Vec::with_capacity(trajectories * FRAMES);
        for row in bytes[16 + header_len..].chunks_exact(96) {
            let mask = u32::from_le_bytes(row[..4].try_into().unwrap());
            ensure!(
                mask < 2048 && row[4..8] == [0; 4],
                "invalid action frame mask/reserved bits"
            );
            let values = std::array::from_fn(|i| {
                f64::from_le_bytes(row[8 + i * 8..16 + i * 8].try_into().unwrap())
            });
            ensure!(
                values.iter().enumerate().all(|(i, v)| v.is_finite()
                    && (mask & (1 << i) != 0 || *v == 0.)
                    && (i == 2 || *v >= 0.)),
                "invalid action frame value"
            );
            ensure!(
                (7..=9).all(|i| values[i] <= 1.) && values[6] <= 1.,
                "invalid action frame fraction"
            );
            frames.push(Frame { mask, values });
        }
        let body_model_version = std::array::from_fn(|i| {
            u8::from_str_radix(&model.model_version[i * 2..i * 2 + 2], 16)
                .expect("validated model version")
        });
        Ok(Self {
            sha256,
            body_model_version,
            accent_means: model.accent_means,
            accent_deviations: model.accent_deviations,
            evaluation_delay_samples: spec.evaluation_delay_ms.map(|ms| u64::from(ms) * 48),
            offsets,
            profiles: bindings,
            frames: frames.into_boxed_slice(),
        })
    }

    pub(crate) fn count(&self) -> usize {
        self.profiles.len()
    }

    fn evaluation_at(&self, issued_at: u64, action_offset: u64) -> Option<u64> {
        let offset = match self.evaluation_delay_samples {
            None => action_offset,
            Some(delay) => action_offset
                .checked_add(delay)?
                .div_ceil(512)
                .checked_mul(512)?,
        };
        // Never clip to the profile horizon or invent an unrecorded tail.
        (offset <= HORIZON).then_some(())?;
        issued_at.checked_add(offset)
    }
    pub(crate) fn stored_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.frames.len() * std::mem::size_of::<Frame>()
            + self.profiles.capacity() * std::mem::size_of::<Profile>()
    }
    pub(crate) fn offsets(&self) -> &[u64; 32] {
        &self.offsets
    }

    pub(super) fn frames(
        &self,
        prototype: usize,
        class: Class,
        action_offset: u64,
        issued_at: u64,
        previous: Option<window::Frame>,
        background_energy: Option<f64>,
    ) -> Option<impl Iterator<Item = window::Frame> + '_> {
        issued_at.checked_add(HORIZON)?;
        if background_energy.is_some_and(|v| !v.is_finite() || v < 0.) {
            return None;
        }
        let cell = self.offsets.binary_search(&action_offset).ok()?;
        let class = CLASSES.iter().position(|c| *c == class)?;
        let index = self.profiles.get(prototype)?.trajectories[class][cell]?;
        let previous = previous
            .filter(|p| {
                p.end == issued_at
                    && p.start < p.end
                    && p.source_end >= p.end
                    && p.source_end <= p.available
                    && p.available <= issued_at
            })
            .and_then(|p| match p.raw[2] {
                Feature::Observed(v) if v.is_finite() => Some(v),
                _ => None,
            });
        Some(
            self.frames[index * FRAMES..(index + 1) * FRAMES]
                .iter()
                .enumerate()
                .map(move |(i, frame)| {
                    let mut raw = std::array::from_fn(|j| {
                        if frame.mask & (1 << j) != 0 {
                            Feature::Projected(frame.values[j])
                        } else {
                            Feature::Unsupported
                        }
                    });
                    let energy = (frame.mask & (1 << 10) != 0).then_some(frame.values[10]);
                    raw[6] = match (energy, background_energy) {
                        (Some(own), Some(background)) if (own + background).is_finite() => {
                            Feature::Projected(if own + background > 0. {
                                own / (own + background)
                            } else {
                                0.
                            })
                        }
                        _ => Feature::Unsupported,
                    };
                    if i == 0 {
                        raw[5] = Feature::Unsupported;
                        let difference = raw[2]
                            .value()
                            .zip(previous)
                            .map(|(now, before)| now - before);
                        raw[3] = difference
                            .map_or(Feature::Unsupported, |d| Feature::Projected(d.max(0.)));
                        raw[4] = difference
                            .map_or(Feature::Unsupported, |d| Feature::Projected((-d).max(0.)));
                    }
                    let start = issued_at + i as u64 * 512;
                    window::Frame {
                        start,
                        end: start + 512,
                        source_end: start + 512,
                        available: start + 512,
                        raw,
                        energy: energy.map_or(Feature::Unsupported, Feature::Projected),
                    }
                }),
        )
    }
}

#[cfg(test)]
pub(super) mod tests;
