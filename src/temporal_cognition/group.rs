//! Fractional acoustic assignment against saved member sets, before lifecycle.

use super::ridge::{Handle, Point, continuity_distance};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Trajectory {
    pub handle: Handle,
    pub point: Point,
    pub slopes: [Option<f64>; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Member {
    pub trajectory: Trajectory,
    pub end_sample: u64,
    pub weight: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Group {
    pub handle: Handle,
    pub eligible: bool,
    pub members: [Option<Member>; 8],
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Config {
    pub bus: u8,
    pub epoch: u64,
    pub sample_rate: u32,
    pub hop: u64,
    pub means: [f64; 3],
    pub deviations: [f64; 3],
    pub distance_limit: f64,
    pub residual_raw: f64,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Match {
    pub member: Handle,
    pub end_sample: u64,
    pub slope_index: usize,
    pub distance: f64,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Row {
    pub trajectory: Handle,
    pub weights: [f64; 8],
    pub matched_members: [Option<Match>; 7],
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Assignment {
    pub end_sample: u64,
    pub group_handles: [Option<Handle>; 7],
    pub rows: [Option<Row>; 8],
    pub distance_evaluations: usize,
}

impl Assignment {
    #[cfg(test)]
    pub fn unassigned_fraction(&self, member_mask: u8) -> Option<f64> {
        let mut residual = 0.0;
        let mut total = 0.0;
        for (index, row) in self
            .rows
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.map(|r| (i, r)))
        {
            if member_mask & (1 << index) != 0 {
                residual += row.weights[7];
                total += row.weights.iter().sum::<f64>();
            }
        }
        (total > 0.0).then(|| residual / total)
    }
}

impl Group {
    pub fn seed(
        handle: Handle,
        end_sample: u64,
        members: &[Trajectory],
    ) -> Result<Self, &'static str> {
        if handle.bus > 1
            || handle.generation == 0
            || end_sample == 0
            || members.is_empty()
            || members.len() > 8
            || members.iter().enumerate().any(|(i, m)| {
                m.handle.bus != handle.bus
                    || m.handle.epoch != handle.epoch
                    || m.handle.generation == 0
                    || members[..i]
                        .iter()
                        .any(|previous| previous.handle == m.handle)
                    || [m.point.frequency_log2, m.point.log_envelope]
                        .iter()
                        .all(Option::is_none)
                    || [m.point.frequency_log2, m.point.log_envelope]
                        .into_iter()
                        .chain(m.slopes)
                        .flatten()
                        .any(|x| !x.is_finite())
            })
        {
            return Err("invalid acoustic group seed");
        }
        let mut out = Self {
            handle,
            eligible: true,
            members: [None; 8],
        };
        for (slot, &trajectory) in out.members.iter_mut().zip(members) {
            *slot = Some(Member {
                trajectory,
                end_sample,
                weight: 1.0,
            });
        }
        Ok(out)
    }
}

pub(super) fn assign_and_refresh(
    config: &Config,
    end_sample: u64,
    current: &[Option<Trajectory>; 8],
    groups: &mut [Option<Group>; 7],
) -> Result<Assignment, &'static str> {
    if config.bus > 1
        || config.sample_rate == 0
        || config.hop == 0
        || end_sample == 0
        || !end_sample.is_multiple_of(config.hop)
        || config.means.iter().any(|x| !x.is_finite())
        || config.deviations.iter().any(|x| !x.is_finite() || *x < 0.0)
        || !config.distance_limit.is_finite()
        || config.distance_limit <= 0.0
        || !config.residual_raw.is_finite()
        || config.residual_raw <= 0.0
    {
        return Err("invalid frozen group configuration or frame time");
    }
    let valid_handle =
        |h: Handle| h.bus == config.bus && h.epoch == config.epoch && h.generation > 0;
    let valid_trajectory = |t: Trajectory| {
        valid_handle(t.handle)
            && [t.point.frequency_log2, t.point.log_envelope]
                .into_iter()
                .chain(t.slopes)
                .flatten()
                .all(f64::is_finite)
    };
    for (index, trajectory) in current
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.map(|t| (i, t)))
    {
        if !valid_trajectory(trajectory)
            || current[..index]
                .iter()
                .flatten()
                .any(|t| t.handle == trajectory.handle)
        {
            return Err("invalid or duplicate current trajectory");
        }
    }
    for (index, group) in groups
        .iter()
        .enumerate()
        .filter_map(|(i, g)| g.as_ref().map(|g| (i, g)))
    {
        if !valid_handle(group.handle)
            || groups[..index]
                .iter()
                .flatten()
                .any(|g| g.handle == group.handle)
        {
            return Err("invalid or duplicate group identity");
        }
        for (member_index, member) in group
            .members
            .iter()
            .enumerate()
            .filter_map(|(i, m)| m.map(|m| (i, m)))
        {
            if !valid_trajectory(member.trajectory)
                || member.end_sample == 0
                || member.end_sample >= end_sample
                || !member.end_sample.is_multiple_of(config.hop)
                || !member.weight.is_finite()
                || !(0.0..=1.0).contains(&member.weight)
                || group.members[..member_index]
                    .iter()
                    .flatten()
                    .any(|m| m.trajectory.handle == member.trajectory.handle)
            {
                return Err("invalid, repeated or noncausal group member");
            }
        }
    }
    let mut output = Assignment {
        end_sample,
        group_handles: std::array::from_fn(|i| groups[i].as_ref().map(|g| g.handle)),
        rows: [None; 8],
        distance_evaluations: 0,
    };
    for (index, trajectory) in current
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.map(|t| (i, t)))
    {
        if trajectory.point.frequency_log2.is_none() && trajectory.point.log_envelope.is_none() {
            continue;
        }
        let mut raw = [0.0; 8];
        raw[7] = config.residual_raw;
        let mut matched_members: [Option<Match>; 7] = [None; 7];
        for (slot, group) in groups
            .iter()
            .enumerate()
            .filter_map(|(i, g)| g.as_ref().filter(|g| g.eligible).map(|g| (i, g)))
        {
            for member in group.members.iter().flatten().filter(|m| m.weight > 0.0) {
                let old = member.trajectory;
                let interval = end_sample - member.end_sample;
                let dt = interval as f64 / f64::from(config.sample_rate);
                let mut best = None;
                for slope_index in 0..if old.slopes[1].is_some() { 2 } else { 1 } {
                    output.distance_evaluations += 1;
                    if let Some((distance, _)) = continuity_distance(
                        old.point,
                        old.slopes[slope_index],
                        trajectory.point,
                        dt,
                        interval == config.hop,
                        &config.means,
                        &config.deviations,
                    ) && distance <= config.distance_limit
                        && best.is_none_or(|(_, d)| distance < d)
                    {
                        best = Some((slope_index, distance));
                    }
                }
                if let Some((slope_index, distance)) = best {
                    let score = member.weight * (-distance * distance).exp();
                    if score > raw[slot]
                        || (score == raw[slot]
                            && matched_members[slot].is_none_or(|m| {
                                (old.handle, slope_index) < (m.member, m.slope_index)
                            }))
                    {
                        raw[slot] = score;
                        matched_members[slot] = Some(Match {
                            member: old.handle,
                            end_sample: member.end_sample,
                            slope_index,
                            distance,
                        });
                    }
                }
            }
        }
        let denominator: f64 = raw.iter().sum();
        if !denominator.is_finite() {
            return Err("group normalization overflow");
        }
        output.rows[index] = Some(Row {
            trajectory: trajectory.handle,
            weights: raw.map(|w| w / denominator),
            matched_members,
        });
    }
    // Refresh only after every assignment has used the same saved reference set.
    for (slot, group) in groups
        .iter_mut()
        .enumerate()
        .filter_map(|(i, g)| g.as_mut().filter(|g| g.eligible).map(|g| (i, g)))
    {
        let mut members = [None; 8];
        let mut count = 0;
        for (index, row) in output
            .rows
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.map(|r| (i, r)))
        {
            if row.weights[slot] > 0.0 {
                members[count] = Some(Member {
                    trajectory: current[index].unwrap(),
                    end_sample,
                    weight: row.weights[slot],
                });
                count += 1;
            }
        }
        if count > 0 {
            group.members = members;
        }
    }
    Ok(output)
}

#[derive(Debug)]
pub(super) struct Energy {
    pub group_energy: [f64; 8],
    pub spectral_shape_supported: bool,
}

pub(super) fn allocate_energy(
    space: &crate::core::log2space::Log2Space,
    power_scan: &[f32],
    mono_energy: Option<f64>,
    peak_bins: &[Option<usize>; 7],
    assignment: &Assignment,
    group_energy_scans: &mut [Vec<f64>; 8],
) -> Result<Option<Energy>, &'static str> {
    space.assert_scan_len_named(power_scan, "group_power_scan");
    for scan in group_energy_scans.iter() {
        space.assert_scan_len_named(scan, "group_energy_scan");
    }
    if power_scan.iter().any(|v| !v.is_finite() || *v < 0.0)
        || mono_energy.is_some_and(|v| !v.is_finite() || v < 0.0)
        || peak_bins.iter().enumerate().any(|(i, p)| {
            p.is_some_and(|p| p >= space.n_bins() || peak_bins[..i].contains(&Some(p)))
        })
        || assignment.rows.iter().flatten().any(|r| {
            r.weights.iter().any(|v| !v.is_finite() || *v < 0.0)
                || (r.weights.iter().sum::<f64>() - 1.0).abs() > 1e-12
        })
    {
        return Err("invalid acoustic energy inputs");
    }
    let Some(mono_energy) = mono_energy else {
        return Ok(None);
    };
    let total: f64 = power_scan.iter().map(|&x| f64::from(x)).sum();
    if mono_energy > 0.0
        && total > 0.0
        && power_scan.iter().enumerate().any(|(bin, &p)| {
            p > 0.0 && assignment.rows[super::trajectory::owner(peak_bins, bin)].is_none()
        })
    {
        return Ok(None);
    }
    for scan in group_energy_scans.iter_mut() {
        scan.fill(0.0);
    }
    let mut group_energy = [0.0; 8];
    if total > 0.0 && mono_energy > 0.0 {
        for (bin, &power) in power_scan.iter().enumerate().filter(|(_, p)| **p > 0.0) {
            let row = assignment.rows[super::trajectory::owner(peak_bins, bin)].unwrap();
            let energy = mono_energy * (f64::from(power) / total);
            for (group, weight) in row.weights.into_iter().enumerate() {
                let value = energy * weight;
                group_energy_scans[group][bin] = value;
                group_energy[group] += value;
            }
        }
    } else if total == 0.0 {
        // Spectral shape is unavailable; the specified fallback retains the scalar in residual.
        group_energy[7] = mono_energy;
    }
    Ok(Some(Energy {
        group_energy,
        spectral_shape_supported: total > 0.0 || mono_energy == 0.0,
    }))
}

#[cfg(test)]
mod tests;
