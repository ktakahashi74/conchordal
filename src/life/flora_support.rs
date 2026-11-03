//! Joint lane trajectories over a published allocation window.
use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct SupportWindow {
    pub start: u64,
    pub end: Option<u64>,
    pub hop_start: u64,
    pub kind: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(super) struct SupportChoice {
    pub action: &'static str,
    pub generation: u64,
    pub position: f32,
    pub at: u64,
    pub end: Option<u64>,
    pub opportunity: &'static str,
    pub reference_end: u64,
    pub baseline: [f64; 2],
    pub candidate: [f64; 2],
    pub release: Option<[f64; 2]>,
    pub delta: [f64; 2],
    pub cells: u64,
    pub retiring: usize,
}

#[derive(Clone, Debug)]
struct CachedMean {
    cells: u64,
    excluded: Option<usize>,
    candidate: Option<Lane>,
    key: Option<Lane>,
    value: [f64; 2],
}

#[derive(Clone, Debug)]
struct Projection {
    r: f32,
    level: f32,
    position: f32,
    bin: usize,
    scale: f32,
}
impl Default for Projection {
    fn default() -> Self {
        Self {
            r: f32::NAN,
            level: f32::NAN,
            position: f32::NAN,
            bin: 0,
            scale: 0.,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct SupportScratch {
    pub cells: u64,
    at: u64,
    target_scale: f32,
    tail_at: Option<(u64, u64)>,
    score_window: Option<(u64, u64)>,
    score_reference: Option<std::sync::Arc<crate::core::harmonic_support::Snapshot>>,
    score_template: Option<std::sync::Arc<[(isize, f32)]>>,
    score_lanes: Vec<Lane>,
    scores: Vec<CachedMean>,
    tail_keys: Vec<(f32, f32)>,
    tails: Vec<f32>,
    lanes: Vec<Lane>,
    density: Vec<f64>,
    summaries: Vec<[f64; 3]>,
    gains: Vec<f32>,
    touched: Vec<usize>,
    point_rows: Vec<usize>,
    parts: Vec<(usize, f32)>,
    last_parts: Vec<(usize, f32)>,
    projections: Vec<Projection>,
}

impl SupportScratch {
    pub(super) fn reset_scores(&mut self) {
        self.scores.clear();
    }

    pub fn reserve(&mut self, hop: usize, bins: usize, lanes: usize) {
        self.cells = 0;
        self.score_window = None;
        self.scores.clear();
        self.scores
            .reserve((lanes + 3) * (hop.max(1).ilog2() as usize + 2));
        self.score_lanes
            .reserve(lanes.saturating_sub(self.score_lanes.len()));
        self.density.resize(hop * bins, 0.);
        self.summaries.resize(hop, [0.; 3]);
        self.gains.resize(hop, 0.);
        self.point_rows.resize(hop, 0);
        self.parts.resize(lanes, (0, 0.));
        self.last_parts.resize(lanes, (0, 0.));
        self.projections.resize(lanes + 1, Projection::default());
        self.touched
            .reserve(bins.saturating_sub(self.touched.len()));
        self.tail_at = None;
        self.tail_keys.resize(lanes, (f32::NAN, f32::NAN));
        self.tails.resize(lanes * hop, 0.);
        self.lanes.reserve(lanes.saturating_sub(self.lanes.len()));
    }
}

impl Flora {
    pub(super) fn support_width(&self) -> Option<u64> {
        let window = self.support_window?;
        let end = window.end?;
        let (snapshot, _) = self.support_action.as_ref()?;
        // The next hop may change targets, pace, or ramps; it is not an issued plan.
        if end <= window.start
            || end > window.hop_start + self.hop as u64
            || snapshot.end_sample > window.start
            || self.spec.amp <= 0.
            // Allocation can create residual noise after tonal selection, even from silence.
            || self.spec.residual
            || (!self.spec.leave_self_out
                && self.lanes.iter().any(|l| l.mode != Mode::Free && l.r > 0.))
            || self
                .lanes
                .iter()
                .any(|l| !l.tonal && l.mode != Mode::Free && (l.r > 0. || l.target > 0.))
        {
            return None;
        }
        Some(end - window.start)
    }

    fn support_phase(&self, excluded: Option<usize>, candidate: Option<&Lane>) -> Option<u64> {
        let width = self.support_width()?;
        let mut shortest = width;
        let target_scale = self
            .lanes
            .iter()
            .enumerate()
            .filter(|(i, l)| Some(*i) != excluded && l.mode != Mode::Free)
            .map(|(_, l)| l.target.powi(2))
            .chain(candidate.map(|l| l.target.powi(2)))
            .sum::<f32>()
            .max(1.)
            .sqrt()
            .recip();
        for lane in self
            .lanes
            .iter()
            .enumerate()
            .filter(|(i, l)| Some(*i) != excluded && l.mode != Mode::Free)
            .map(|(_, l)| l)
            .chain(candidate)
        {
            let phase = if lane.mode == Mode::Retiring {
                if !(0. ..1.).contains(&lane.retirement_decay) || lane.retirement_decay == 0. {
                    return None;
                }
                (DEAD_MASS.sqrt() / (lane.r + DEAD_MASS.sqrt())).ln() / lane.retirement_decay.ln()
            } else {
                let (current, target) = if lane.noisy {
                    (lane.q, (lane.target * target_scale).powi(2))
                } else {
                    (lane.r, lane.target * target_scale)
                };
                if target == current {
                    width as f32
                } else {
                    (target - current).abs() * SLEW_SEC * self.fs
                }
            };
            shortest = shortest.min((phase.ceil() as u64).max(1));
        }
        Some(shortest)
    }

    pub(super) fn support_action_mean(
        &self,
        excluded: Option<usize>,
        candidate: Option<&Lane>,
        cells: u64,
    ) -> Option<[f64; 2]> {
        let width = self.support_width()?;
        let window = self.support_window?;
        let (snapshot, _) = self.support_action.as_ref()?;
        let bins = self.space.n_bins();
        let mut scratch = self.support_scratch.borrow_mut();
        if cells == 0 || cells > width || width as usize > scratch.summaries.len() {
            return None;
        }
        let window_key = (window.start, width);
        if scratch.score_window != Some(window_key)
            || scratch.score_lanes != self.lanes
            || scratch
                .score_reference
                .as_ref()
                .is_none_or(|s| !std::sync::Arc::ptr_eq(s, snapshot))
            || scratch
                .score_template
                .as_ref()
                .is_none_or(|t| !std::sync::Arc::ptr_eq(t, &self.support_template))
        {
            scratch.cells = 0;
            scratch.reset_scores();
            scratch.score_window = Some(window_key);
            scratch.score_reference = Some(snapshot.clone());
            scratch.score_template = Some(self.support_template.clone());
            scratch.score_lanes.clone_from(&self.lanes);
        }
        if let Some(found) = scratch.scores.iter().find(|s| {
            s.cells == cells && s.excluded == excluded && s.candidate.as_ref() == candidate
        }) {
            return Some(found.value);
        }
        let mut key = candidate.cloned();
        if let Some(lane) = &mut key {
            let start = (window.start - window.hop_start) as usize;
            let bin_at = |t| {
                let position = project(lane, t, start, self.hop, self.fs, 1., Some(lane.r)).1;
                self.space.nearest_index(position.exp2())
            };
            let bin = bin_at(0);
            let mut next = (start / 64 + 1) * 64;
            let mut constant = true;
            while next < start + width as usize {
                if bin_at(next - start) != bin {
                    constant = false;
                    break;
                }
                next += 64;
            }
            if constant {
                let position = self.space.centers_log2[bin];
                lane.position = position;
                lane.target_position = position;
                lane.sounding_log2 = position;
            }
        }
        if let Some(found) = scratch
            .scores
            .iter()
            .find(|s| s.cells == cells && s.excluded == excluded && s.key == key)
        {
            return Some(found.value);
        }
        let target_scale = self
            .lanes
            .iter()
            .enumerate()
            .filter(|(i, l)| Some(*i) != excluded && l.mode != Mode::Free)
            .map(|(_, l)| l.target.powi(2))
            .chain(candidate.map(|l| l.target.powi(2)))
            .sum::<f32>()
            .max(1.)
            .sqrt()
            .recip();
        let new_window = scratch.tail_at != Some(window_key);
        if new_window {
            scratch.projections.fill(Projection::default());
            let mut amp = self.amp.clone();
            let mut gain = self.gain.clone();
            for level in &mut scratch.gains[..width as usize] {
                *level = amp.tick() * gain.tick();
            }
        }
        for (index, lane) in self.lanes.iter().enumerate() {
            let key = (lane.r, lane.retirement_decay);
            if new_window || scratch.tail_keys[index] != key {
                if lane.r == 0. {
                    scratch.tails[index * self.hop..index * self.hop + width as usize].fill(0.);
                } else {
                    let mut tail = lane.clone();
                    tail.retire();
                    for t in 0..width as usize {
                        tail.slew(self.fs);
                        scratch.tails[index * self.hop + t] = tail.r;
                    }
                }
                scratch.tail_keys[index] = key;
            }
        }
        scratch.tail_at = Some(window_key);
        let contribution = |lane: &Lane,
                            t: usize,
                            tail: Option<f32>,
                            level: f32,
                            cached: &mut Projection|
         -> Option<(usize, f32)> {
            let (r, position) = project(
                lane,
                t,
                (window.start - window.hop_start) as usize,
                self.hop,
                self.fs,
                target_scale,
                tail,
            );
            if r == 0. {
                return Some((0, 0.));
            }
            if cached.r != r || cached.level != level {
                let scale = (level * r / self.spec.amp).powf(2. * self.support_loudness_exp);
                if !scale.is_finite() || scale < 0. {
                    return None;
                }
                cached.r = r;
                cached.level = level;
                cached.scale = scale;
            }
            if cached.position != position {
                cached.position = position;
                cached.bin = self.space.nearest_index(position.exp2());
            }
            Some((cached.bin, cached.scale))
        };
        if scratch.cells != cells
            || scratch.at != window.start
            || scratch.target_scale != target_scale
            || scratch.lanes != self.lanes
        {
            scratch.density[..cells as usize * bins].fill(0.);
            scratch.summaries[..cells as usize].fill([0.; 3]);
            let first = self.support_template.first().map_or(0, |p| p.0);
            let contiguous = self
                .support_template
                .last()
                .is_some_and(|p| p.0 - first + 1 == self.support_template.len() as isize);
            for point in 0..cells as usize {
                let t = (u128::from(width) * (2 * point as u128 + 1) / (2 * u128::from(cells)))
                    as usize;
                for (index, lane) in self.lanes.iter().enumerate() {
                    scratch.parts[index] = if lane.mode == Mode::Free {
                        (0, 0.)
                    } else {
                        let tail = scratch.tails[index * self.hop + t];
                        let level = scratch.gains[t];
                        contribution(lane, t, Some(tail), level, &mut scratch.projections[index])?
                    };
                }
                if point > 0 && scratch.parts == scratch.last_parts {
                    scratch.point_rows[point] = scratch.point_rows[point - 1];
                    continue;
                }
                scratch.point_rows[point] = point;
                for index in 0..self.lanes.len() {
                    scratch.last_parts[index] = scratch.parts[index];
                }
                scratch.touched.clear();
                let mut lower = bins;
                let mut upper = 0;
                for index in 0..self.lanes.len() {
                    let (bin, scale) = scratch.parts[index];
                    if scale == 0. {
                        continue;
                    }
                    if contiguous {
                        let start = bin as isize + first;
                        let lo = start.max(0);
                        let hi = (start + self.support_template.len() as isize).min(bins as isize);
                        if lo >= hi {
                            continue;
                        }
                        let (lo, hi) = (lo as usize, hi as usize);
                        let from = (lo as isize - start) as usize;
                        for (out, &(_, q)) in scratch.density[point * bins + lo..point * bins + hi]
                            .iter_mut()
                            .zip(&self.support_template[from..from + hi - lo])
                        {
                            *out += f64::from(q * scale);
                        }
                        lower = lower.min(lo);
                        upper = upper.max(hi);
                    } else {
                        for &(offset, q) in self.support_template.iter() {
                            let i = bin as isize + offset;
                            if i < 0 || i >= bins as isize {
                                continue;
                            }
                            let value = f64::from(q * scale);
                            if value == 0. {
                                continue;
                            }
                            let i = i as usize;
                            if scratch.density[point * bins + i] == 0. {
                                scratch.touched.push(i);
                            }
                            scratch.density[point * bins + i] += value;
                        }
                    }
                }
                let summary = if contiguous {
                    if lower < upper {
                        snapshot.joint_density_slice(
                            lower,
                            &scratch.density[point * bins + lower..point * bins + upper],
                        )
                    } else {
                        [0.; 3]
                    }
                } else {
                    snapshot.joint_density_summary(
                        scratch
                            .touched
                            .iter()
                            .map(|&i| (i, scratch.density[point * bins + i])),
                    )
                };
                scratch.summaries[point] = summary;
            }
            scratch.cells = cells;
            scratch.at = window.start;
            scratch.target_scale = target_scale;
            scratch.lanes.clone_from(&self.lanes);
        }
        let mut result = [0.; 2];
        let mut previous = None;
        let mut previous_value = [0.; 2];
        for point in 0..cells as usize {
            let t =
                (u128::from(width) * (2 * point as u128 + 1) / (2 * u128::from(cells))) as usize;
            let removed = if let Some(index) = excluded {
                let tail = scratch.tails[index * self.hop + t];
                let level = scratch.gains[t];
                contribution(
                    &self.lanes[index],
                    t,
                    Some(tail),
                    level,
                    &mut scratch.projections[index],
                )?
            } else {
                (0, 0.)
            };
            let added = if let Some(lane) = candidate {
                let tail = excluded
                    .filter(|&i| {
                        (self.lanes[i].r, self.lanes[i].retirement_decay)
                            == (lane.r, lane.retirement_decay)
                    })
                    .map(|i| scratch.tails[i * self.hop + t]);
                let level = scratch.gains[t];
                contribution(
                    lane,
                    t,
                    tail,
                    level,
                    &mut scratch.projections[self.lanes.len()],
                )?
            } else {
                (0, 0.)
            };
            let row = scratch.point_rows[point];
            let query = (row, removed, added);
            let value = if previous == Some(query) {
                previous_value
            } else {
                let summary = scratch.summaries[row];
                let summary = snapshot.update_joint_density(
                    &mut scratch.density[row * bins..(row + 1) * bins],
                    summary,
                    &self.support_template,
                    removed,
                    added,
                    false,
                );
                let mass = snapshot.observed_mass() + summary[2];
                [
                    summary[0],
                    if mass > 0. {
                        4. * summary[1].max(0.) / mass.powi(2)
                    } else {
                        0.
                    },
                ]
            };
            result[0] += value[0] / cells as f64;
            result[1] += value[1] / cells as f64;
            previous = Some(query);
            previous_value = value;
        }
        if scratch.scores.len() < scratch.scores.capacity() {
            scratch.scores.push(CachedMean {
                cells,
                excluded,
                candidate: candidate.cloned(),
                key,
                value: result,
            });
        }
        Some(result)
    }

    pub(super) fn support_cells(
        &self,
        excluded: Option<usize>,
        candidate: Option<&Lane>,
        resolution: i32,
    ) -> Option<u64> {
        let width = self.support_width()?;
        let base =
            crate::life::support_action::cells(width, self.support_phase(excluded, candidate)?);
        Some(crate::life::support_action::resolution_cells(
            base, width, resolution,
        ))
    }

    pub(super) fn record_support_choice(&self, choice: SupportChoice) {
        tracing::trace!(flora_id = self.id, ?choice, "support Flora joint action");
        #[cfg(test)]
        if let Some(records) = self.support_records.borrow_mut().as_mut() {
            records.push(choice);
        }
    }
}

/// Jump the renderer's fixed-target slew deadline to an integration sample.
pub(super) fn project(
    lane: &Lane,
    t: usize,
    start: usize,
    hop: usize,
    fs: f32,
    target_scale: f32,
    retired_r: Option<f32>,
) -> (f32, f32) {
    let r = if lane.mode == Mode::Retiring {
        retired_r.unwrap_or_else(|| {
            let mut tail = lane.clone();
            for _ in 0..=t {
                tail.slew(fs);
            }
            tail.r
        })
    } else {
        let target = lane.target * target_scale;
        let (target, current) = if lane.noisy {
            (target * target, lane.q)
        } else {
            (target, lane.r)
        };
        let duration = SLEW_SEC * fs;
        let left = if target != lane.slew_target || (lane.slew_left == 0 && current != target) {
            ((target - current).abs() * duration).ceil() as usize
        } else {
            lane.slew_left
        };
        let left = left.saturating_sub(t + 1);
        let value = if left == 0 {
            target
        } else {
            target - (target - current).signum() * (1. / duration) * left as f32
        };
        if lane.noisy { value.sqrt() } else { value }
    };
    let last = (start + t) / 64 * 64;
    let update = if last >= start {
        Some(last)
    } else if lane.age_samples == 0 && lane.mode != Mode::Free {
        Some(start)
    } else {
        None
    };
    let position = update.map_or(lane.sounding_log2, |offset| {
        lane.position + (lane.target_position - lane.position) * (offset + 1) as f32 / hop as f32
    });
    (r, position)
}
