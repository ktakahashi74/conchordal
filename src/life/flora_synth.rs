use super::*;
use fearless_simd::{Level, Simd, dispatch, f32x8, prelude::*};
use std::f32::consts::TAU;

pub(super) fn random_unit(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)) as f32
}

pub(super) fn noise_draw(state: &mut u64) -> [f32; 2] {
    let scale = 1.5_f32.sqrt();
    [
        (2.0 * random_unit(state) - 1.0) * scale,
        (2.0 * random_unit(state) - 1.0) * scale,
    ]
}

impl Lane {
    pub(super) fn selected_target(
        &self,
        peak: &Peak,
        mode: PitchApplyMode,
        hold: bool,
    ) -> (f32, f32) {
        (
            peak.mass.sqrt(),
            if mode == PitchApplyMode::Rearticulate || hold {
                self.position
            } else {
                peak.position
            },
        )
    }

    pub(super) fn new(noisy: bool, position: f32) -> Self {
        Self {
            harmonic_sounding: false,
            harmonic_hold: false,
            mode: Mode::Free,
            tonal: true,
            noisy,
            noise_index: 0,
            cell: None,
            position,
            target_position: position,
            sounding_log2: position,
            sigma: 1.0,
            r: 0.0,
            q: 0.0,
            target: 0.0,
            power_sum: 0.0,
            age_samples: 0,
            generation: 0,
            window: 0..0,
            pending_since: None,
            forced_reason: None,
            retirement_decay: 0.0,
            slew_target: 0.0,
            slew_left: 0,
            slew_step: 0.0,
            z: [1.0, 0.0],
            w: [1.0, 0.0],
            e: [1.0, 0.0],
            a: 1.0,
            b: 0.0,
        }
    }

    pub(super) fn retire(&mut self) {
        if self.mode != Mode::Free {
            self.mode = Mode::Retiring;
            self.target = 0.0;
            self.target_position = self.position;
            self.cell = None;
        }
    }

    /// Samples and per-sample step of a new linear slew. A birth keeps the fast
    /// full-scale onset; a sounding lane spreads each change over CHANGE_SEC.
    pub(super) fn slew_plan(&self, target: f32, current: f32, fs: f32) -> (usize, f32) {
        let birth = SLEW_SEC * fs;
        if (self.age_samples as f32) < birth {
            (
                ((target - current).abs() * birth).ceil() as usize,
                1.0 / birth,
            )
        } else {
            let left = (CHANGE_SEC * fs).ceil() as usize;
            (left, (target - current).abs() / left as f32)
        }
    }

    pub(super) fn slew(&mut self, fs: f32) {
        if self.mode == Mode::Retiring && self.retirement_decay > 0.0 {
            // Shift the metabolic amplitude exponential by the existing dead-mass
            // floor, giving a continuous finite tail and a reusable pool slot.
            let floor = DEAD_MASS.sqrt();
            self.r = ((self.r + floor) * self.retirement_decay - floor).max(0.0);
            self.q = self.r * self.r;
            if self.r == 0.0 {
                self.mode = Mode::Free;
            }
            return;
        }
        let target = if self.noisy {
            self.target * self.target
        } else {
            self.target
        };
        let current = if self.noisy { self.q } else { self.r };
        if target != self.slew_target || (self.slew_left == 0 && current != target) {
            self.slew_target = target;
            (self.slew_left, self.slew_step) = self.slew_plan(target, current, fs);
        }
        // Pin the deadline rather than letting accumulated rounding add one sample.
        let value = if self.slew_left > 0 {
            self.slew_left -= 1;
            if self.slew_left == 0 {
                target
            } else {
                target - (target - current).signum() * self.slew_step * self.slew_left as f32
            }
        } else {
            target
        };
        if self.noisy {
            self.q = value;
            self.r = value.sqrt();
        } else {
            self.r = value;
            self.q = value * value;
        }
        if self.mode == Mode::Retiring && self.r == 0.0 {
            self.mode = Mode::Free;
        }
    }

    pub(super) fn atom(&mut self, n: [f32; 2]) -> f32 {
        let x = self.w[0] * self.z[0] - self.w[1] * self.z[1];
        self.z[1] = self.w[1] * self.z[0] + self.w[0] * self.z[1];
        self.z[0] = x;
        self.e[0] = self.a * self.e[0] + self.b * n[0];
        self.e[1] = self.a * self.e[1] + self.b * n[1];
        self.e[0] * self.z[0] - self.e[1] * self.z[1]
    }

    pub(super) fn project_support_density(
        &self,
        amp: f32,
        fs: f32,
        space: &Log2Space,
        ref_power: f32,
        loudness_exp: f32,
    ) -> Vec<(isize, f32)> {
        let hz = self.sounding_log2.exp2();
        let mut amplitudes = vec![0.; space.n_bins()];
        if !self.noisy {
            crate::life::sound::spectral::add_log2_energy(&mut amplitudes, space, hz, amp * self.r);
        } else {
            // Integrate the stationary AR(1) sidebands over each log-frequency bin.
            // Each quadrature's innovation has variance 1/2; the real atom has
            // mean-square power 1/2, like the unit sine atom.
            let a = f64::from(self.a);
            let w0 = std::f64::consts::TAU * f64::from(hz / fs);
            let primitive =
                |w: f64| 2. * ((1. + a) * (w / 2.).sin()).atan2((1. - a) * (w / 2.).cos());
            let half = f64::from(space.step()) / 2.;
            for (i, amplitude) in amplitudes.iter_mut().enumerate() {
                let center = f64::from(space.centers_log2[i]);
                let lo = std::f64::consts::TAU * (center - half).exp2() / f64::from(fs);
                let hi = std::f64::consts::TAU * (center + half).exp2() / f64::from(fs);
                let power = (primitive(hi - w0) - primitive(lo - w0) + primitive(hi + w0)
                    - primitive(lo + w0))
                    / (4. * std::f64::consts::PI);
                *amplitude = amp * self.r * (2. * power.max(0.)).sqrt() as f32;
            }
        }
        crate::core::harmonic_support::body_density::project(
            &amplitudes,
            hz,
            space,
            ref_power,
            loudness_exp,
        )
    }
}

impl Flora {
    pub(super) fn update_tones(&mut self) {
        self.order.clear();
        self.order.extend(
            (0..self.lanes.len())
                .filter(|&i| self.lanes[i].tonal && self.lanes[i].mode == Mode::Current),
        );
        self.order.sort_unstable_by(|&a, &b| {
            self.lanes[b]
                .target
                .total_cmp(&self.lanes[a].target)
                .then(a.cmp(&b))
        });
        for index in 0..self.order.len() {
            let i = self.order[index];
            let lane = &self.lanes[i];
            let mut nearest = self
                .cells
                .iter()
                .enumerate()
                .filter_map(|(c, cell)| {
                    let peak = cell.peak.as_ref()?;
                    let distance = (peak.position - lane.position).abs();
                    (peak.stay && peak.claimed.is_none() && distance <= 50.0 / 1200.0)
                        .then_some((c, distance, peak.bin))
                })
                .min_by(|a, b| a.1.total_cmp(&b.1).then(a.2.cmp(&b.2)));
            self.support.scratch.reset_scores();
            let mut selected_target = None;
            let mut support_hold = false;
            let mut support_release = false;
            let mut support_selected = false;
            if self.harmonic.act_now
                && self.support.action.is_some()
                && let Some((c, _, _)) = nearest
            {
                if let Some(use_relations) = self.harmonic.consumer.as_mut() {
                    use_relations.set_source(Some(
                        crate::temporal_cognition::harmonic_relations::Source {
                            kind: 1,
                            id: self.id,
                            generation: lane.generation,
                        },
                    ));
                }
                let lane_position = lane.position;
                let lane_generation = lane.generation;
                let peak = self.cells[c].peak.as_ref().unwrap();
                let baseline_position = peak.position;
                let baseline_distance = -f64::from((peak.position - lane_position).abs());
                let mut follow = lane.clone();
                (follow.target, follow.target_position) =
                    lane.selected_target(peak, self.spec.pitch_apply_mode, false);
                let mut hold = follow.clone();
                hold.target_position = lane_position;
                let mut release = lane.clone();
                release.retire();
                // Temporal evidence is fixed across quadrature refinements.
                let temporal = |pitch, retain| {
                    self.harmonic.consumer.as_ref().map_or(0., |u| {
                        u.action_difference(pitch, baseline_position, retain)
                    })
                };
                for cell in &self.cells {
                    let Some(peak) = &cell.peak else { continue };
                    if peak.stay
                        && peak.claimed.is_none()
                        && (peak.position - lane_position).abs() <= 50. / 1200.
                    {
                        self.support.native_scores[peak.bin] = temporal(Some(peak.position), false);
                    }
                }
                let native_hold = temporal(Some(lane_position), true);
                let native_release = temporal(None, true);
                if let Some((selected, (target, base, chosen, release_value, cells), _)) =
                    crate::life::support_action::resolved_choice(
                        self.support_width().unwrap_or(0),
                        |resolution| {
                            let mut cells = self
                                .support_cells(Some(i), Some(&release), resolution)?
                                .max(self.support_cells(Some(i), Some(&follow), resolution)?);
                            for cell in &self.cells {
                                let Some(peak) = &cell.peak else { continue };
                                if peak.stay
                                    && peak.claimed.is_none()
                                    && (peak.position - lane_position).abs() <= 50. / 1200.
                                {
                                    let mut candidate = follow.clone();
                                    candidate.target = peak.mass.sqrt();
                                    cells = cells.max(self.support_cells(
                                        Some(i),
                                        Some(&candidate),
                                        resolution,
                                    )?);
                                }
                            }
                            let base = self.support_action_mean(Some(i), Some(&follow), cells)?;
                            let release_value =
                                self.support_action_mean(Some(i), Some(&release), cells)?;
                            let weights = self.support.action.as_ref()?.1;
                            let weighted = |value: [f64; 2]| {
                                weights.value([value[0] - base[0], 0., value[1] - base[1]])
                            };
                            let mut chosen = base;

                            let mut best = c;
                            let mut score = 0.;
                            let mut target = (follow.target, follow.target_position);
                            for j in 0..self.cells.len() {
                                let Some(peak) = &self.cells[j].peak else {
                                    continue;
                                };
                                if !peak.stay
                                    || peak.claimed.is_some()
                                    || (peak.position - lane_position).abs() > 50. / 1200.
                                {
                                    continue;
                                }
                                let mut candidate = follow.clone();
                                (candidate.target, candidate.target_position) = self.lanes[i]
                                    .selected_target(peak, self.spec.pitch_apply_mode, false);
                                let peak_distance = (peak.position - lane_position).abs();
                                let bin = peak.bin;
                                let candidate_target =
                                    (candidate.target, candidate.target_position);
                                let value =
                                    self.support_action_mean(Some(i), Some(&candidate), cells)?;
                                let delta =
                                    weighted(value) - f64::from(peak_distance) - baseline_distance
                                        + self.support.native_scores[bin];
                                if delta > score {
                                    best = j;
                                    score = delta;
                                    target = candidate_target;
                                    chosen = value;
                                }
                            }
                            for (j, candidate, native) in [
                                (self.cells.len(), hold.clone(), native_hold),
                                (self.cells.len() + 1, release.clone(), native_release),
                            ] {
                                let candidate_target =
                                    (candidate.target, candidate.target_position);
                                let value =
                                    self.support_action_mean(Some(i), Some(&candidate), cells)?;
                                let delta = weighted(value) + native;
                                if delta > score {
                                    best = j;
                                    score = delta;
                                    target = candidate_target;
                                    chosen = value;
                                }
                            }
                            Some((best, (target, base, chosen, release_value, cells)))
                        },
                    )
                {
                    let window = self.support.window.unwrap();
                    self.record_support_choice(SupportChoice {
                        action: if selected == self.cells.len() + 1 {
                            "release"
                        } else if selected == self.cells.len() {
                            "hold"
                        } else {
                            "follow"
                        },
                        generation: lane_generation,
                        position: lane_position,
                        at: window.start,
                        end: window.end,
                        opportunity: window.kind,
                        reference_end: self.support.action.as_ref().unwrap().0.end_sample,
                        baseline: base,
                        candidate: chosen,
                        release: Some(release_value),
                        delta: [chosen[0] - base[0], chosen[1] - base[1]],
                        cells,
                        retiring: self
                            .lanes
                            .iter()
                            .filter(|l| l.mode == Mode::Retiring)
                            .count(),
                    });
                    support_selected = true;
                    selected_target = Some(target);
                    support_hold = selected == self.cells.len();
                    support_release = selected == self.cells.len() + 1;
                    if selected < self.cells.len() {
                        let peak = self.cells[selected].peak.as_ref().unwrap();
                        nearest = Some((selected, (peak.position - lane_position).abs(), peak.bin));
                    }
                }
            }
            let lane = &mut self.lanes[i];
            lane.harmonic_hold = !self.harmonic.act_now || support_hold;
            if support_release {
                lane.retire();
                self.retirements += 1;
                continue;
            }
            if self.harmonic.act_now
                && !support_selected
                && let Some(use_relations) = self.harmonic.consumer.as_mut().filter(|u| u.active())
                && let Some((baseline, _, _)) = nearest
            {
                use_relations.set_source(Some(
                    crate::temporal_cognition::harmonic_relations::Source {
                        kind: 1,
                        id: self.id,
                        generation: lane.generation,
                    },
                ));
                self.harmonic.cells.clear();
                self.harmonic
                    .cells
                    .extend(self.cells.iter().enumerate().filter_map(|(c, cell)| {
                        let peak = cell.peak.as_ref()?;
                        (peak.stay
                            && peak.claimed.is_none()
                            && (peak.position - lane.position).abs() <= 50. / 1200.)
                            .then_some(c)
                    }));
                let base = self
                    .harmonic
                    .cells
                    .iter()
                    .position(|&c| c == baseline)
                    .unwrap();
                let count = self.harmonic.cells.len();
                let baseline_distance = -f64::from(
                    (self.cells[baseline].peak.as_ref().unwrap().position - lane.position).abs(),
                );
                let selected = use_relations.choose_scored(count + 1, base, |j| {
                    if j == count {
                        (
                            Some(lane.position),
                            use_relations.source(),
                            baseline_distance,
                            true,
                        )
                    } else {
                        let peak = self.cells[self.harmonic.cells[j]].peak.as_ref().unwrap();
                        // The existing continuation selector minimizes displacement.
                        (
                            Some(peak.position),
                            use_relations.source(),
                            -f64::from((peak.position - lane.position).abs()),
                            false,
                        )
                    }
                });
                if selected == count {
                    lane.harmonic_hold = true;
                } else {
                    let c = self.harmonic.cells[selected];
                    let peak = self.cells[c].peak.as_ref().unwrap();
                    nearest = Some((c, (peak.position - lane.position).abs(), peak.bin));
                }
            }
            if let Some((c, _, _)) = nearest {
                self.cells[c].peak.as_mut().unwrap().claimed = Some(i);
                lane.cell = Some(c);
                let peak = self.cells[c].peak.as_ref().unwrap();
                (lane.target, lane.target_position) = selected_target.unwrap_or_else(|| {
                    lane.selected_target(peak, self.spec.pitch_apply_mode, lane.harmonic_hold)
                });
            } else {
                self.retirements += 1;
                self.flapping += u64::from(lane.age_samples as f32 <= SLEW_SEC * self.fs);
                lane.retire();
            }
        }
        // Resolve adjacent conflicts in position order, independent of cell traversal.
        loop {
            self.order.clear();
            self.order.extend(
                (0..self.lanes.len())
                    .filter(|&i| self.lanes[i].tonal && self.lanes[i].mode == Mode::Current),
            );
            self.order.sort_unstable_by(|&a, &b| {
                let pa = self.cells[self.lanes[a].cell.unwrap()]
                    .peak
                    .as_ref()
                    .unwrap();
                let pb = self.cells[self.lanes[b].cell.unwrap()]
                    .peak
                    .as_ref()
                    .unwrap();
                pa.position.total_cmp(&pb.position).then(a.cmp(&b))
            });
            let conflict = self.order.windows(2).find_map(|pair| {
                let a = pair[0];
                let b = pair[1];
                let pa = self.cells[self.lanes[a].cell.unwrap()]
                    .peak
                    .as_ref()
                    .unwrap();
                let pb = self.cells[self.lanes[b].cell.unwrap()]
                    .peak
                    .as_ref()
                    .unwrap();
                if hz_to_erb(pb.position.exp2()) - hz_to_erb(pa.position.exp2()) >= 0.5 {
                    return None;
                }
                let (va, vb) = if self.support.active {
                    (
                        f64::from(pa.mass).ln() + self.support.choice_scan[pa.bin],
                        f64::from(pb.mass).ln() + self.support.choice_scan[pb.bin],
                    )
                } else {
                    (f64::from(pa.mass), f64::from(pb.mass))
                };
                let loser = if va < vb || (va == vb && a > b) { a } else { b };
                if self.harmonic.act_now
                    && let Some(use_relations) =
                        self.harmonic.consumer.as_ref().filter(|u| u.active())
                {
                    let candidates = [a.min(b), a.max(b)];
                    let baseline = usize::from(candidates[0] == loser);
                    let keep = use_relations.choose_scored(2, baseline, |j| {
                        let lane = &self.lanes[candidates[j]];
                        (
                            Some(
                                self.cells[lane.cell.unwrap()]
                                    .peak
                                    .as_ref()
                                    .unwrap()
                                    .position,
                            ),
                            Some(crate::temporal_cognition::harmonic_relations::Source {
                                kind: 1,
                                id: self.id,
                                generation: lane.generation,
                            }),
                            if candidates[j] == a { va } else { vb },
                            false,
                        )
                    });
                    Some(candidates[1 - keep])
                } else {
                    Some(loser)
                }
            });
            let Some(i) = conflict else {
                break;
            };
            let peak = self.cells[self.lanes[i].cell.unwrap()]
                .peak
                .as_mut()
                .unwrap();
            peak.claimed = None;
            peak.suppressed = true;
            self.retirements += 1;
            self.flapping += u64::from(self.lanes[i].age_samples as f32 <= SLEW_SEC * self.fs);
            self.lanes[i].retire();
        }
        self.support.scratch.reset_scores();
        loop {
            let best = self
                .cells
                .iter()
                .enumerate()
                .filter_map(|(c, cell)| {
                    let peak = cell.peak.as_ref()?;
                    (peak.enter && peak.claimed.is_none() && !peak.suppressed && !peak.visited)
                        .then_some(c)
                })
                .max_by(|&a, &b| {
                    let pa = self.cells[a].peak.as_ref().unwrap();
                    let pb = self.cells[b].peak.as_ref().unwrap();
                    if self.support.active {
                        (f64::from(pa.mass).ln() + self.support.choice_scan[pa.bin])
                            .total_cmp(
                                &(f64::from(pb.mass).ln() + self.support.choice_scan[pb.bin]),
                            )
                            .then(pb.bin.cmp(&pa.bin))
                    } else {
                        pa.mass.total_cmp(&pb.mass).then(pb.bin.cmp(&pa.bin))
                    }
                });
            if !self.harmonic.act_now {
                break;
            }
            let Some(mut c) = best else {
                break;
            };
            let mut temporal_no_spawn = false;
            if let Some(use_relations) = self.harmonic.consumer.as_mut().filter(|u| u.active()) {
                use_relations.set_source(None);
                self.harmonic.cells.clear();
                self.harmonic
                    .cells
                    .extend(self.cells.iter().enumerate().filter_map(|(c, cell)| {
                        let peak = cell.peak.as_ref()?;
                        (peak.enter && peak.claimed.is_none() && !peak.suppressed && !peak.visited)
                            .then_some(c)
                    }));
                let base = self
                    .harmonic
                    .cells
                    .iter()
                    .position(|&candidate| candidate == c)
                    .unwrap();
                let count = self.harmonic.cells.len();
                let base_peak = self.cells[self.harmonic.cells[base]].peak.as_ref().unwrap();
                let base_score = if self.support.active {
                    f64::from(base_peak.mass).ln() + self.support.choice_scan[base_peak.bin]
                } else {
                    f64::from(base_peak.mass)
                };
                let chosen = use_relations.choose_scored(count + 1, base, |j| {
                    if j == count {
                        (None, None, base_score, true)
                    } else {
                        let peak = self.cells[self.harmonic.cells[j]].peak.as_ref().unwrap();
                        let score = if self.support.active {
                            f64::from(peak.mass).ln() + self.support.choice_scan[peak.bin]
                        } else {
                            f64::from(peak.mass)
                        };
                        (Some(peak.position), None, score, false)
                    }
                });
                if chosen == count {
                    if self.support.action.is_none() {
                        break;
                    }
                    temporal_no_spawn = true;
                } else {
                    c = self.harmonic.cells[chosen];
                }
            }
            let position = self.cells[c].peak.as_ref().unwrap().position;
            self.cells[c].peak.as_mut().unwrap().visited = true;
            let blocked = self.lanes.iter().any(|lane| {
                lane.tonal
                    && lane.mode == Mode::Current
                    && (hz_to_erb(position.exp2())
                        - hz_to_erb(
                            self.cells[lane.cell.unwrap()]
                                .peak
                                .as_ref()
                                .unwrap()
                                .position
                                .exp2(),
                        ))
                    .abs()
                        < 0.5
            });
            if blocked {
                self.cells[c].peak.as_mut().unwrap().suppressed = true;
                continue;
            }
            let Some(i) = self.lanes[..self.tonal_len]
                .iter()
                .position(|lane| lane.mode == Mode::Free)
            else {
                break;
            };
            let mut lane = Lane::new(self.spec.tone == FloraTone::Resonant, position);
            lane.noise_index = self.lanes[i].noise_index;
            lane.retirement_decay = (-0.5 * GROWTH_RATE * self.pace * METABOLISM / self.fs).exp();
            lane.mode = Mode::Current;
            lane.cell = Some(c);
            (lane.target, lane.target_position) = lane.selected_target(
                self.cells[c].peak.as_ref().unwrap(),
                self.spec.pitch_apply_mode,
                false,
            );
            let no_add = if self.harmonic.weight != 0. {
                self.harmonic
                    .consumer
                    .as_ref()
                    .map_or(0., |u| u.no_add_difference(position))
            } else {
                0.
            };
            let declined = crate::life::support_action::resolved_choice(
                self.support_width().unwrap_or(0),
                |resolution| {
                    let cells = self.support_cells(None, Some(&lane), resolution)?;
                    let base = self.support_action_mean(None, None, cells)?;
                    let added = self.support_action_mean(None, Some(&lane), cells)?;
                    let value = self.support.action.as_ref()?.1.value([
                        added[0] - base[0],
                        0.,
                        added[1] - base[1],
                    ]);
                    Some((usize::from(no_add > value), (value, base, added, cells)))
                },
            );
            if let Some((selected, (_, base, added, cells), _)) = declined {
                let window = self.support.window.unwrap();
                self.record_support_choice(SupportChoice {
                    action: if selected == 1 { "no_spawn" } else { "birth" },
                    generation: self.births + 1,
                    position,
                    at: window.start,
                    end: window.end,
                    opportunity: window.kind,
                    reference_end: self.support.action.as_ref().unwrap().0.end_sample,
                    baseline: base,
                    candidate: added,
                    release: None,
                    delta: [added[0] - base[0], added[1] - base[1]],
                    cells,
                    retiring: self
                        .lanes
                        .iter()
                        .filter(|l| l.mode == Mode::Retiring)
                        .count(),
                });
            }
            if declined.map_or(temporal_no_spawn, |(selected, _, _)| selected == 1) {
                tracing::trace!(
                    flora_id = self.id,
                    position,
                    ?declined,
                    "support action no-spawn"
                );
                continue;
            }
            let angle = TAU * random_unit(&mut self.rng);
            let (sin, cos) = angle.sin_cos();
            lane.z = [cos, sin];
            if lane.noisy {
                lane.e = noise_draw(&mut self.rng);
            }
            lane.generation = self.births + 1;
            self.lanes[i] = lane;
            self.births += 1;
            self.cells[c].peak.as_mut().unwrap().claimed = Some(i);
        }
        for lane in &mut self.lanes {
            if lane.tonal && lane.mode == Mode::Current {
                let peak = self.cells[lane.cell.unwrap()].peak.as_ref().unwrap();
                lane.sigma = peak.sigma;
                lane.window = peak.window.clone();
                lane.pending_since = None;
            }
        }
    }

    pub(crate) fn render_add(&mut self, presentation: &mut [f32], habitat: &mut [f32]) {
        self.render::<true>(presentation, habitat);
    }

    pub(super) fn render<const SIMD: bool>(
        &mut self,
        presentation: &mut [f32],
        habitat: &mut [f32],
    ) {
        self.assert_scans();
        let before = (
            self.births,
            self.retirements,
            self.lanes.iter().filter(|l| l.mode != Mode::Free).count(),
        );
        assert_eq!(presentation.len(), self.hop);
        assert_eq!(habitat.len(), self.hop);
        let now = self.temporal.as_ref().unwrap().now;
        if let Some(gap) = &mut self.span_gap {
            gap.advance(now, self.releasing);
            if self.releasing {
                gap.finish(now);
            }
        }
        self.metric_allocate(false, 0);
        let decay = (-0.5 * GROWTH_RATE * self.pace * METABOLISM / self.fs).exp();
        for lane in &mut self.lanes {
            lane.retirement_decay = decay;
            lane.power_sum = 0.0;
            let norm = lane.z[0].hypot(lane.z[1]);
            lane.z[0] /= norm;
            lane.z[1] /= norm;
            if lane.noisy && lane.mode != Mode::Retiring {
                let f = lane.target_position.exp2();
                let b = if lane.tonal {
                    (f * ((lane.sigma / self.space.bins_per_oct as f32).exp2() - 1.0))
                        .clamp(0.5, 0.5 * erb_bw_hz(f))
                } else {
                    erb_bw_hz(f)
                };
                lane.a = (-PI * b / self.fs).exp();
                lane.b = (1.0 - lane.a * lane.a).sqrt();
            }
        }
        // Include free noisy lanes, so lane indices define the stream even during turnover.
        for pair in self.noise.as_chunks_mut::<2>().0 {
            pair.copy_from_slice(&noise_draw(&mut self.rng));
        }
        self.habitat_pcm.fill(0.0);
        for t in 0..self.hop {
            let at = now + t as u64;
            if let Some(gap) = &mut self.span_gap {
                gap.advance(at, self.releasing);
            }
            let m = self.temporal.as_ref().unwrap();
            let crossing = m.crossings.iter().any(|c| c.tick == at);
            let active = m.active;
            let beat_crossing = if m.opportunity_ratio == 1 {
                crossing
            } else {
                m.beat_crossings.iter().any(|c| c.tick == at)
            };
            let transition = beat_crossing && !self.releasing && self.span_gap_opportunity(at);
            let allocation = if let Some(ticks) = &self.harmonic.ticks {
                ticks.binary_search(&at).is_ok()
            } else {
                crossing && (active || transition)
            };
            if self.harmonic.ticks.is_some() && transition {
                self.metric_allocate(false, t);
            }
            if allocation && !self.releasing && !self.range.is_empty() {
                self.metric_allocate(true, t);
                for lane in &mut self.lanes {
                    lane.retirement_decay = decay;
                    if lane.noisy && lane.mode == Mode::Current {
                        let f = lane.target_position.exp2();
                        let b = if lane.tonal {
                            (f * ((lane.sigma / self.space.bins_per_oct as f32).exp2() - 1.0))
                                .clamp(0.5, 0.5 * erb_bw_hz(f))
                        } else {
                            erb_bw_hz(f)
                        };
                        lane.a = (-PI * b / self.fs).exp();
                        lane.b = (1.0 - lane.a * lane.a).sqrt();
                    }
                }
            }
            let amp = self.amp.tick() * self.gain.tick();
            let amp = if self.range.is_empty() { 0.0 } else { amp };
            for (lane_index, lane) in self.lanes.iter_mut().enumerate() {
                if self.releasing && self.gain.value == 0.0 {
                    lane.mode = Mode::Free;
                    lane.r = 0.0;
                    lane.q = 0.0;
                    lane.target = 0.0;
                } else {
                    lane.slew(self.fs);
                }
                if lane.mode != Mode::Free {
                    lane.age_samples += 1;
                }
                lane.power_sum += (amp * lane.r).powi(2);
                if t % 64 == 0 || lane.age_samples == 1 {
                    let log2 = lane.position
                        + (lane.target_position - lane.position) * (t + 1) as f32 / self.hop as f32;
                    lane.sounding_log2 = log2;
                    let (sin, cos) = (TAU * log2.exp2() / self.fs).sin_cos();
                    lane.w = [cos, sin];
                }
                if let Some(recorder) = &self.harmonic.recorder
                    && self.spec.routing.to_habitat
                    && (lane.tonal || self.spec.residual)
                {
                    use crate::temporal_cognition::harmonic_relations::{
                        OFFSET, ONSET, Source, Trajectory, Value, runtime::Record,
                    };
                    let sounding = lane.mode != Mode::Free && amp * lane.r != 0.;
                    let flags = if sounding != lane.harmonic_sounding {
                        if sounding { ONSET } else { OFFSET }
                    } else {
                        0
                    };
                    if flags != 0 || (sounding && t + 1 == self.hop) {
                        let source = Source {
                            kind: if lane.tonal { 1 } else { 2 },
                            id: self.id,
                            generation: if lane.tonal {
                                lane.generation
                            } else {
                                lane_index as u64
                            },
                        };
                        recorder.record(Record {
                            sample: if flags != 0 {
                                at
                            } else {
                                now + self.hop as u64
                            },
                            source: Some(source),
                            flags,
                            trajectory: Some(Trajectory {
                                source,
                                pitch: Value {
                                    valid: lane.tonal,
                                    ..Value::exact(f64::from(lane.sounding_log2))
                                },
                                amplitude: Value::exact(f64::from(amp * lane.r)),
                                timbre: Value::default(),
                                articulation: source.generation,
                                sounding,
                            }),
                        });
                    }
                    lane.harmonic_sounding = sounding;
                }
            }
            self.observe_sounding_span_gap(at, amp);
            let noise = &self.noise[t * 2 * self.noisy_len..(t + 1) * 2 * self.noisy_len];
            let out = synthesize::<SIMD>(&mut self.lanes, noise, amp);
            if self.spec.routing.to_presentation {
                presentation[t] += out;
            }
            if self.spec.routing.to_habitat {
                habitat[t] += out;
                self.habitat_pcm[t] = out;
            }
        }
        if let Some(gap) = &mut self.span_gap {
            gap.advance(now + self.hop as u64, self.releasing);
        }
        for lane in &mut self.lanes {
            lane.position = lane.target_position;
            self.energy[usize::from(!lane.tonal)] += lane.power_sum as f64 / self.fs as f64;
        }
        self.lifecycle_changed = before
            != (
                self.births,
                self.retirements,
                self.lanes.iter().filter(|l| l.mode != Mode::Free).count(),
            );
        self.rendered_samples += self.hop as u64;
    }
}

/// Sums every lane's atom. With `SIMD`, whole blocks of eight lanes use the
/// compile-time baseline level (SSE2, NEON); without `mul_add` they match the scalar
/// `Lane::atom` bit for bit.
fn synthesize<const SIMD: bool>(lanes: &mut [Lane], noise: &[f32], amp: f32) -> f32 {
    let (first, mut out) = if SIMD {
        dispatch!(Level::baseline(), simd => synthesize_blocks(simd, lanes, noise, amp))
    } else {
        (0, 0.0)
    };
    for lane in &mut lanes[first..] {
        let n = if lane.noisy {
            [noise[2 * lane.noise_index], noise[2 * lane.noise_index + 1]]
        } else {
            [0.0; 2]
        };
        let signal = lane.atom(n);
        out += amp * lane.r * signal;
    }
    out
}

/// Returns the first lane past the whole blocks and the blocks' sum.
#[inline(always)]
fn synthesize_blocks<S: Simd>(
    simd: S,
    lanes: &mut [Lane],
    noise: &[f32],
    amp: f32,
) -> (usize, f32) {
    let mut out = 0.0;
    let mut first = 0;
    while first + 8 <= lanes.len() {
        let block = &mut lanes[first..first + 8];
        let x = f32x8::load_array(simd, std::array::from_fn(|i| block[i].z[0]));
        let y = f32x8::load_array(simd, std::array::from_fn(|i| block[i].z[1]));
        let c = f32x8::load_array(simd, std::array::from_fn(|i| block[i].w[0]));
        let s = f32x8::load_array(simd, std::array::from_fn(|i| block[i].w[1]));
        let er = f32x8::load_array(simd, std::array::from_fn(|i| block[i].e[0]));
        let ei = f32x8::load_array(simd, std::array::from_fn(|i| block[i].e[1]));
        let a = f32x8::load_array(simd, std::array::from_fn(|i| block[i].a));
        let b = f32x8::load_array(simd, std::array::from_fn(|i| block[i].b));
        let noise_part = |part: usize| {
            f32x8::load_array(
                simd,
                std::array::from_fn(|i| {
                    if block[i].noisy {
                        noise[2 * block[i].noise_index + part]
                    } else {
                        0.0
                    }
                }),
            )
        };
        let (nr, ni) = (noise_part(0), noise_part(1));
        let xr = c * x - s * y;
        let yi = s * x + c * y;
        let er = a * er + b * nr;
        let ei = a * ei + b * ni;
        let signal = (er * xr - ei * yi).to_array();
        let (xr, yi, er, ei) = (xr.to_array(), yi.to_array(), er.to_array(), ei.to_array());
        for i in 0..8 {
            block[i].z = [xr[i], yi[i]];
            block[i].e = [er[i], ei[i]];
            out += amp * block[i].r * signal[i];
        }
        first += 8;
    }
    (first, out)
}
