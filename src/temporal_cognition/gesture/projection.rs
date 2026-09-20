//! Conditional state paths without synthetic evidence endpoints or unknown-parent admission.

use super::*;
use crate::temporal_cognition::feature_projection::{Feature, window::Frame};

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Projection {
    pub issued_at: u64,
    pub evaluation_at: u64,
    pub projected: bool,
    pub states: [f64; 4],
    pub original_unknown: f64,
    pub pruned_mass: f64,
    pub unknown: f64,
    pub retained_paths: usize,
}

#[derive(Clone, Copy)]
struct ConditionalPath {
    state: State,
    entered: u64,
    mass: f64,
}

#[derive(Clone, Copy)]
struct Prefix {
    key: (ridge::Handle, u64, usize),
    end: u64,
    paths: [ConditionalPath; 15],
    count: usize,
    pruned_mass: f64,
    frames: u64,
}

/// Scratch for immutable profile trajectories within one shared-table build.
#[derive(Default)]
pub(in crate::temporal_cognition) struct ProjectionCache {
    prefix: Option<Prefix>,
    branch_prefix: Option<Prefix>,
    branch_at: Option<u64>,
    pub advanced_frames: u64,
    pub reused_frames: u64,
}

impl ProjectionCache {
    // The caller proves raw-prefix identity; project_states still checks group, issue and end.
    pub(in crate::temporal_cognition) fn retag_matching_prefix(
        &mut self,
        trajectory: usize,
        branch_at: u64,
        mut matches: impl FnMut(usize, u64) -> bool,
    ) {
        self.branch_at = Some(branch_at);
        // The endpoint may contain a changed suffix while the pre-action prefix still matches.
        let mut matched = None;
        for prefix in [&mut self.prefix, &mut self.branch_prefix]
            .into_iter()
            .flatten()
        {
            if prefix.key.2 == trajectory {
                continue;
            }
            if matched == Some((prefix.key, prefix.frames, prefix.end))
                || matches(prefix.key.2, prefix.frames)
            {
                matched = Some((prefix.key, prefix.frames, prefix.end));
                prefix.key.2 = trajectory;
            }
        }
    }
}

impl Gesture {
    pub(in crate::temporal_cognition) fn project_states(
        &self,
        handle: ridge::Handle,
        issued_at: u64,
        evaluation_at: u64,
        frames: impl Iterator<Item = Frame>,
        cache: Option<(usize, &mut ProjectionCache)>,
    ) -> Option<Projection> {
        if handle.bus != self.bus
            || handle.epoch != self.epoch
            || self.last != Some(issued_at)
            || evaluation_at < issued_at
            || evaluation_at > issued_at.checked_add(u64::from(self.rate) * 4)?
        {
            return None;
        }
        let group = self
            .groups
            .iter()
            .flatten()
            .find(|g| g.handle == handle && g.end == issued_at)?;
        let empty = ConditionalPath {
            state: State::Attack,
            entered: 0,
            mass: 0.,
        };
        let mut paths = [empty; 15];
        if group.paths.len() > paths.len() {
            return None;
        }
        let mut count = group.paths.len();
        for (out, path) in paths.iter_mut().zip(&group.paths) {
            *out = ConditionalPath {
                state: path.state,
                entered: path.entered,
                mass: path.mass,
            };
        }
        let mut scratch = [empty; 60];
        let mut cursor = issued_at;
        let mut pruned_mass = 0.;
        let mut complete_frames = 0;
        let (trajectory, mut cache) = match cache {
            Some((trajectory, cache)) => (trajectory, Some(cache)),
            None => (0, None),
        };
        let key = (handle, issued_at, trajectory);
        if let Some(prefix) = cache.as_ref().and_then(|c| {
            [c.prefix, c.branch_prefix]
                .into_iter()
                .flatten()
                .filter(|p| p.key == key && p.end <= evaluation_at)
                .max_by_key(|p| p.end)
        }) {
            paths = prefix.paths;
            count = prefix.count;
            cursor = prefix.end;
            pruned_mass = prefix.pruned_mass;
            complete_frames = prefix.frames;
            let cache = cache.as_deref_mut().unwrap();
            cache.reused_frames += complete_frames;
            if cache.branch_at.is_some_and(|cut| prefix.end <= cut) {
                cache.branch_prefix = Some(prefix);
            }
        } else if let Some(cache) = cache.as_deref_mut() {
            cache.prefix = None;
            cache.branch_prefix = None;
        }
        let reused_until = cursor;
        for frame in frames {
            if cursor == evaluation_at {
                break;
            }
            if reused_until > issued_at && frame.end <= reused_until {
                continue;
            }
            if frame.start != cursor
                || frame.end <= frame.start
                || frame.raw.iter().any(|v| matches!(v, Feature::Observed(_)))
                || frame
                    .raw
                    .iter()
                    .filter_map(|v| v.value())
                    .any(|v| !v.is_finite())
            {
                return None;
            }
            let log_rms = frame.raw[2].value()?;
            let low = log_rms.exp2() <= 0.01 * self.config.rms_reference;
            let end = frame.end.min(evaluation_at);
            let dt = (end - cursor) as f64 / f64::from(self.rate);
            let mut expanded = 0;
            let rate_features = rate_features(
                [
                    frame.raw[3].value(),
                    frame.raw[4].value(),
                    frame.raw[5].value(),
                    None,
                ],
                &self.config,
            );
            #[cfg(test)]
            let mut cost_clock = crate::temporal_cognition::context::tests::projection_clock();
            for path in &paths[..count] {
                let elapsed = cursor.checked_sub(path.entered)? as f64 / f64::from(self.rate);
                let transition = transition(
                    path.state,
                    rates(path.state, &rate_features, elapsed.ln_1p(), &self.config).ok()?,
                    dt,
                    low,
                );
                if transition.iter().all(|p| *p == 0.) {
                    return None;
                }
                for (state, probability) in STATES.into_iter().zip(transition) {
                    let mass = path.mass * probability;
                    if mass > 0. {
                        scratch[expanded] = ConditionalPath {
                            state,
                            entered: if state == path.state {
                                path.entered
                            } else {
                                end
                            },
                            mass,
                        };
                        expanded += 1;
                    }
                }
            }
            #[cfg(test)]
            crate::temporal_cognition::context::tests::record_projection_cost(&mut cost_clock, 8);
            scratch[..expanded].sort_unstable_by_key(|p| (p.state, p.entered));
            let mut merged = 0;
            for index in 0..expanded {
                let path = scratch[index];
                if merged > 0
                    && (scratch[merged - 1].state, scratch[merged - 1].entered)
                        == (path.state, path.entered)
                {
                    scratch[merged - 1].mass += path.mass;
                } else {
                    scratch[merged] = path;
                    merged += 1;
                }
            }
            #[cfg(test)]
            crate::temporal_cognition::context::tests::record_projection_cost(&mut cost_clock, 9);
            scratch[..merged].sort_unstable_by(|a, b| {
                b.mass
                    .total_cmp(&a.mass)
                    .then((a.state, a.entered).cmp(&(b.state, b.entered)))
            });
            count = merged.min(paths.len());
            pruned_mass += scratch[count..merged].iter().map(|p| p.mass).sum::<f64>();
            paths[..count].copy_from_slice(&scratch[..count]);
            #[cfg(test)]
            crate::temporal_cognition::context::tests::record_projection_cost(&mut cost_clock, 10);
            cursor = end;
            if let Some(cache) = cache.as_deref_mut() {
                cache.advanced_frames += 1;
                // A clipped last hop is recomputed from the full-hop prefix next time.
                if end == frame.end {
                    complete_frames += 1;
                    let prefix = Prefix {
                        key,
                        end,
                        paths,
                        count,
                        pruned_mass,
                        frames: complete_frames,
                    };
                    cache.prefix = Some(prefix);
                    if cache.branch_at.is_some_and(|cut| end <= cut) {
                        cache.branch_prefix = Some(prefix);
                    }
                }
            }
        }
        if cursor != evaluation_at {
            return None;
        }
        let mut states = [0.; 4];
        for path in &paths[..count] {
            states[path.state as usize] += path.mass;
        }
        let unknown = (group.unknown + pruned_mass).min(1.);
        if (states.iter().sum::<f64>() + unknown - 1.).abs() > 1e-10 {
            return None;
        }
        Some(Projection {
            issued_at,
            evaluation_at,
            projected: evaluation_at > issued_at,
            states,
            original_unknown: group.unknown,
            pruned_mass,
            unknown,
            retained_paths: count,
        })
    }
}

#[cfg(test)]
mod tests;
