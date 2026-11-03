use super::Community;
use crate::core::float::unit_gaussian;
use crate::core::landscape::Landscape;
use crate::life::voice::SoundBody;
use crate::scenario::{FieldSampling, FieldTarget, SpawnStrategy};
use rand::{Rng, RngExt, distr::Distribution, distr::weighted::WeightedIndex};
use std::sync::Arc;

/// Width (fraction of the range's field_score span) of the Gaussian that weights
/// peaks near the tension target in density placement.
const TENSION_LEVEL_SIGMA_FRAC: f32 = 0.15;

/// Peak score for a field target at bin `i` (higher = better extremum).
fn field_peak_score(target: FieldTarget, landscape: &Landscape, i: usize) -> f32 {
    match target {
        FieldTarget::Consonance => landscape
            .consonance_field_level_eff
            .get(i)
            .copied()
            .unwrap_or(f32::MIN),
        FieldTarget::Dissonance => -landscape
            .consonance_field_level_eff
            .get(i)
            .copied()
            .unwrap_or(f32::MAX),
        FieldTarget::Edge => {
            let v = landscape
                .consonance_field_level_eff
                .get(i)
                .copied()
                .unwrap_or(f32::MAX);
            -(v - 0.5).abs()
        }
        FieldTarget::Gap => -landscape
            .subjective_intensity
            .get(i)
            .copied()
            .unwrap_or(f32::MAX),
        FieldTarget::Uniform => 0.0,
    }
}

/// Non-negative density mass for a field target at bin `i`. `gap_ref` is the
/// loudest in-range intensity, used only by `Gap`. Shared by spawn placement and Rain.
pub(crate) fn field_density_mass(
    target: FieldTarget,
    landscape: &Landscape,
    i: usize,
    gap_ref: f32,
) -> f32 {
    match target {
        FieldTarget::Consonance => landscape
            .consonance_density_mass_eff
            .get(i)
            .copied()
            .unwrap_or(0.0),
        FieldTarget::Dissonance => {
            // Missing bins default to fully consonant so they get zero mass.
            let v = landscape
                .consonance_field_level_eff
                .get(i)
                .copied()
                .unwrap_or(1.0);
            (1.0 - v).max(0.0)
        }
        FieldTarget::Edge => {
            let v = landscape
                .consonance_field_level_eff
                .get(i)
                .copied()
                .unwrap_or(0.0);
            (1.0 - 2.0 * (v - 0.5).abs()).max(0.0)
        }
        FieldTarget::Gap => {
            let e = landscape
                .subjective_intensity
                .get(i)
                .copied()
                .unwrap_or(gap_ref);
            (gap_ref - e).max(0.0)
        }
        FieldTarget::Uniform => 1.0,
    }
}

impl Community {
    pub(super) fn spawn_body_template(
        &self,
        control: Option<&crate::scenario::control::VoiceControl>,
        landscape: &Landscape,
        nominal_hz: f32,
    ) -> Option<Arc<[(isize, f32)]>> {
        control.and_then(|control| {
            use rand::SeedableRng;
            let mut preview_rng = rand::rngs::SmallRng::seed_from_u64(self.seed);
            let mut body = crate::life::voice::sound_body::build_sound_body_from_control(
                control,
                nominal_hz,
                self.time.fs,
                Some(landscape),
                &mut preview_rng,
            );
            let signal = crate::life::voice::articulation_core::ArticulationSignal {
                amplitude: 1.,
                is_active: true,
                ..Default::default()
            };
            body.project_support_density(
                &landscape.space,
                landscape.support_ref_power,
                landscape.support_loudness_exp,
                &signal,
            )
            .map(Arc::from)
        })
    }

    pub(super) fn refresh_spawn_relations(
        &self,
        landscape: &Landscape,
        control: Option<&crate::scenario::control::VoiceControl>,
        template: Option<Arc<[(isize, f32)]>>,
    ) -> bool {
        if landscape.harmonic_relations_weight == 0. && landscape.harmonic_continuation_weight == 0.
        {
            return false;
        }
        let density = landscape.support_single_density(
            control.map_or(crate::scenario::control::BodyControl::default().amp, |c| {
                c.body.amp
            }),
        );
        let mut slot = self.harmonic_spawn.borrow_mut();
        crate::temporal_cognition::harmonic_relations::consumer::Use::refresh(
            &mut slot,
            landscape,
            None,
            crate::core::harmonic_support::preview::Observer::Mixed,
            landscape.support.clone(),
            template,
            density,
            self.time.fs,
            landscape.harmonic_now_sample,
        );
        slot.as_ref().is_some_and(|u| u.active())
    }

    pub(super) fn harmonic_spawn_allowed(
        &self,
        landscape: &Landscape,
        control: &crate::scenario::control::VoiceControl,
        frequency: f32,
    ) -> bool {
        let temporal_allowed = (|| {
            if landscape.harmonic_relations_weight == 0.
                && landscape.harmonic_continuation_weight == 0.
            {
                return true;
            }
            crate::temporal_cognition::harmonic_relations::runtime::record_opportunity(
                landscape,
                None,
                landscape.harmonic_now_sample,
            );
            if landscape.harmonic_relations.is_none() {
                return true;
            }
            let template = self.spawn_body_template(Some(control), landscape, frequency);
            if !self.refresh_spawn_relations(landscape, Some(control), template) {
                return true;
            }
            self.harmonic_spawn
                .borrow()
                .as_ref()
                .unwrap()
                .choose_scored(2, 0, |i| {
                    ((i == 0).then_some(frequency.log2()), None, 0., i != 0)
                })
                == 0
        })();
        // Birth has no issued onset/participation plan yet. Its first hop cannot
        // stand in for the later sound; C/N remains neutral until that opportunity.
        temporal_allowed
    }

    pub(super) fn mix_spawn_weights<R: Rng + Clone>(
        &self,
        frequencies: &[f32],
        weights: &mut [f32],
        rng: &R,
    ) {
        let active = self.harmonic_spawn.borrow();
        let Some(active) = active.as_ref().filter(|u| u.active()) else {
            return;
        };
        let mut scratch = self.harmonic_spawn_weights.borrow_mut();
        scratch.resize(weights.len(), 0.);
        for (out, weight) in scratch.iter_mut().zip(weights.iter()) {
            *out = f64::from(*weight);
        }
        let baseline = WeightedIndex::new(&*weights).map_or_else(
            |_| {
                weights
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map_or(0, |(i, _)| i)
            },
            |dist| dist.sample(&mut rng.clone()),
        );
        active.mix(&mut scratch, baseline, |i| Some(frequencies[i].log2()));
        for (out, weight) in weights.iter_mut().zip(scratch.iter()) {
            *out = *weight as f32;
        }
    }

    pub(super) fn choose_spawn_candidate(
        &self,
        frequencies: &[f32],
        baseline: usize,
        mut score: impl FnMut(usize) -> f64,
    ) -> usize {
        let active = self.harmonic_spawn.borrow();
        let Some(active) = active.as_ref().filter(|u| u.active()) else {
            return baseline;
        };
        let last = frequencies.len() - 1;
        // Respawn's max_by keeps the last candidate on a tie.
        last - active.choose_scored(frequencies.len(), last - baseline, |i| {
            (
                Some(frequencies[last - i].log2()),
                None,
                score(last - i),
                false,
            )
        })
    }

    #[cfg(test)]
    pub(crate) fn aud_m1_range_mass(landscape: &Landscape, range: (f32, f32)) -> Vec<f32> {
        let lo = landscape.space.nearest_index(range.0);
        let hi = landscape.space.nearest_index(range.1);
        // Empty population, zero tension: every in-range bin is unoccupied.
        let mut weights: Vec<_> = (lo..=hi)
            .map(|i| field_density_mass(FieldTarget::Consonance, landscape, i, 0.0))
            .collect();
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        } else {
            let uniform = 1.0 / weights.len() as f32;
            weights.fill(uniform);
        }
        weights
    }

    /// Returns true if `freq_hz` is within `min_dist_erb` (ERB scale) of any existing
    /// voice's base frequency, or of any `reserved` frequency claimed earlier in the
    /// same spawn batch.
    fn is_range_occupied_with(&self, freq_hz: f32, min_dist_erb: f32, reserved: &[f32]) -> bool {
        if !freq_hz.is_finite() || min_dist_erb <= 0.0 {
            return false;
        }
        let target_erb = crate::core::erb::hz_to_erb(freq_hz.max(1e-6));
        for voice in &self.voices {
            let base_hz = voice.body.base_freq_hz();
            if !base_hz.is_finite() {
                continue;
            }
            let d_erb = (crate::core::erb::hz_to_erb(base_hz.max(1e-6)) - target_erb).abs();
            if d_erb < min_dist_erb {
                return true;
            }
        }
        for &freq in reserved {
            if !freq.is_finite() {
                continue;
            }
            let d_erb = (crate::core::erb::hz_to_erb(freq.max(1e-6)) - target_erb).abs();
            if d_erb < min_dist_erb {
                return true;
            }
        }
        false
    }

    #[cfg(test)]
    fn decide_frequency<R: Rng + Clone>(
        &self,
        strategy: &SpawnStrategy,
        landscape: &Landscape,
        rng: &mut R,
        reserved: &[f32],
        control: Option<&crate::scenario::control::VoiceControl>,
    ) -> f32 {
        self.decide_frequency_with_relations(strategy, landscape, rng, reserved, control, true)
    }

    fn decide_frequency_with_relations<R: Rng + Clone>(
        &self,
        strategy: &SpawnStrategy,
        landscape: &Landscape,
        rng: &mut R,
        reserved: &[f32],
        control: Option<&crate::scenario::control::VoiceControl>,
        condition_temporal: bool,
    ) -> f32 {
        let space = &landscape.space;
        let n_bins = space.n_bins();
        let (min_freq, max_freq) = strategy.freq_range_hz();
        let (min_freq, max_freq) = (min_freq.min(max_freq), min_freq.max(max_freq));
        let min_dist_erb = strategy.min_dist_erb();
        if landscape.support_weights.active()
            && landscape.support.is_some()
            && matches!(strategy, SpawnStrategy::Field { target, .. } if *target != FieldTarget::Uniform)
            && min_freq <= space.fmax
            && max_freq >= space.fmin
        {
            return self.decide_support_frequency(
                strategy,
                landscape,
                rng,
                reserved,
                control,
                condition_temporal,
            );
        }

        let sample_range = |lo: f32, hi: f32, rng: &mut R| -> f32 {
            let (lo_log2, hi_log2) = (lo.log2(), hi.log2());
            if lo_log2 >= hi_log2 {
                return lo;
            }
            2.0f32
                .powf(rng.random_range(lo_log2..hi_log2))
                .clamp(lo, hi)
        };
        // Without field evidence, keep the uniform fallback inside the requested
        // range. Uniform placement itself does not depend on the analysis space.
        if n_bins == 0
            || max_freq < space.fmin
            || min_freq > space.fmax
            || matches!(
                strategy,
                SpawnStrategy::Field {
                    target: FieldTarget::Uniform,
                    ..
                }
            )
        {
            for _ in 0..32 {
                let f = sample_range(min_freq, max_freq, rng);
                if !self.is_range_occupied_with(f, min_dist_erb, reserved) {
                    return f;
                }
            }
            return sample_range(min_freq, max_freq, rng);
        }

        let min_freq = min_freq.max(space.fmin);
        let max_freq = max_freq.min(space.fmax);
        let idx_min = space.nearest_index(min_freq);
        let idx_max = space.nearest_index(max_freq);
        let bin_freq = |idx| space.freq_of_index(idx).clamp(min_freq, max_freq);
        let temporal = if condition_temporal
            && (landscape.harmonic_relations_weight != 0.
                || landscape.harmonic_continuation_weight != 0.)
            && landscape.harmonic_relations.is_some()
        {
            self.refresh_spawn_relations(
                landscape,
                control,
                self.spawn_body_template(control, landscape, (min_freq * max_freq).sqrt()),
            )
        } else {
            false
        };

        // Tension (Consonance only): aim at a metastable step below the strongest
        // peak — target = L_max - tension*(L_max - L_min) in field_score over the
        // range. None when tension == 0 (plain strongest-consonance behaviour).
        let tension = match strategy {
            SpawnStrategy::Field {
                target: FieldTarget::Consonance,
                tension,
                ..
            } => (*tension).clamp(0.0, 1.0),
            _ => 0.0,
        };
        let (target_score, tension_sigma) = if tension > 0.0 {
            let mut lmax = f32::MIN;
            let mut lmin = f32::MAX;
            for i in idx_min..=idx_max {
                let s = landscape
                    .consonance_field_score_eff
                    .get(i)
                    .copied()
                    .unwrap_or(f32::NAN);
                if s.is_finite() {
                    lmax = lmax.max(s);
                    lmin = lmin.min(s);
                }
            }
            if lmax > lmin {
                let sigma = ((lmax - lmin) * TENSION_LEVEL_SIGMA_FRAC).max(1e-6);
                (Some(lmax - tension * (lmax - lmin)), sigma)
            } else {
                (None, 1.0)
            }
        } else {
            (None, 1.0)
        };

        let jitter_bin = |idx: usize, rng: &mut R| -> f32 {
            let center = space.freq_of_index(idx);
            let half = space.step() * 0.5;
            let center_log2 = center.log2();
            let lo = if idx == 0 {
                min_freq
            } else {
                2.0f32.powf(center_log2 - half).max(min_freq)
            };
            let hi = if idx == n_bins - 1 {
                max_freq
            } else {
                2.0f32.powf(center_log2 + half).min(max_freq)
            };
            if lo >= hi {
                return bin_freq(idx);
            }
            sample_range(lo, hi, rng)
        };

        let jitter_free_bin = |idx: usize, rng: &mut R| -> f32 {
            // Try a few times to jitter within the bin while avoiding occupied bands.
            for _ in 0..16 {
                let f = jitter_bin(idx, rng);
                if !self.is_range_occupied_with(f, min_dist_erb, reserved) {
                    return f;
                }
            }
            bin_freq(idx)
        };

        let mut body_workspace = self.birth_surrogate.borrow_mut();
        let body_cache = if let (Some(work), Some(control)) = (body_workspace.as_mut(), control) {
            match strategy {
                SpawnStrategy::Field {
                    target,
                    sampling: FieldSampling::Peak,
                    ..
                } => work.candidates(
                    control,
                    landscape,
                    self.current_frame,
                    (min_freq, max_freq),
                    |i| {
                        if target_score.is_some() {
                            landscape.consonance_field_score_eff[i]
                        } else {
                            field_peak_score(*target, landscape, i)
                        }
                    },
                ),
                SpawnStrategy::Field { target, .. } => {
                    let gap_ref = if *target == FieldTarget::Gap {
                        (idx_min..=idx_max)
                            .map(|i| landscape.subjective_intensity[i])
                            .fold(0.0f32, f32::max)
                    } else {
                        0.0
                    };
                    work.candidates(
                        control,
                        landscape,
                        self.current_frame,
                        (min_freq, max_freq),
                        |i| field_density_mass(*target, landscape, i, gap_ref),
                    )
                }
                SpawnStrategy::Linear { .. } => None,
            }
        } else {
            None
        };

        let pick_idx = match strategy {
            // Deterministic extremum of the target (higher score = better).
            SpawnStrategy::Field {
                target,
                sampling: FieldSampling::Peak,
                ..
            } => {
                let mut best_free = None;
                let mut best_any = (idx_min, f32::MIN);
                let native_score = |i: usize| {
                    match target_score {
                        Some(t) => {
                            let s = if let Some(slot) = body_cache {
                                body_workspace.as_ref().unwrap().candidate_mass(slot, i)
                            } else {
                                landscape
                                    .consonance_field_score_eff
                                    .get(i)
                                    .copied()
                                    .unwrap_or(f32::MIN)
                            };
                            -(s - t).abs() // nearest to the tension target = best
                        }
                        None => {
                            if let Some(slot) = body_cache {
                                body_workspace.as_ref().unwrap().candidate_mass(slot, i)
                            } else {
                                field_peak_score(*target, landscape, i)
                            }
                        }
                    }
                };
                for i in idx_min..=idx_max {
                    let score = native_score(i);
                    if score > best_any.1 {
                        best_any = (i, score);
                    }
                    let f = bin_freq(i);
                    if !self.is_range_occupied_with(f, min_dist_erb, reserved)
                        && score > best_free.map_or(f32::MIN, |(_, v)| v)
                    {
                        best_free = Some((i, score));
                    }
                }
                let baseline = best_free.unwrap_or(best_any).0;
                if temporal {
                    let has_free = best_free.is_some();
                    let mut candidates = self.harmonic_spawn_candidates.borrow_mut();
                    candidates.clear();
                    candidates.extend((idx_min..=idx_max).filter(|&i| {
                        !has_free
                            || !self.is_range_occupied_with(bin_freq(i), min_dist_erb, reserved)
                    }));
                    let baseline = candidates.iter().position(|&i| i == baseline).unwrap();
                    let chosen = self
                        .harmonic_spawn
                        .borrow()
                        .as_ref()
                        .unwrap()
                        .choose_scored(candidates.len(), baseline, |i| {
                            (
                                Some(bin_freq(candidates[i]).log2()),
                                None,
                                f64::from(native_score(candidates[i])),
                                false,
                            )
                        });
                    candidates[chosen]
                } else {
                    baseline
                }
            }
            // Stochastic cloud weighted by the target mass.
            SpawnStrategy::Field { target, .. } => {
                let range_len = idx_max - idx_min + 1;
                // Gap mass is measured relative to the loudest bin in range.
                let gap_ref = if *target == FieldTarget::Gap {
                    (idx_min..=idx_max)
                        .filter_map(|i| landscape.subjective_intensity.get(i).copied())
                        .fold(0.0f32, f32::max)
                } else {
                    0.0
                };
                let mut weights = if body_cache.is_some() {
                    std::mem::take(&mut body_workspace.as_mut().unwrap().selection_weights)
                } else {
                    Vec::with_capacity(range_len)
                };
                weights.clear();
                let mut has_unoccupied = false;
                let mut sum = 0.0f32;
                for i in idx_min..=idx_max {
                    let f = bin_freq(i);
                    let occupied = self.is_range_occupied_with(f, min_dist_erb, reserved);
                    if !occupied {
                        has_unoccupied = true;
                    }
                    let raw = match target_score {
                        Some(t) => {
                            let s = landscape
                                .consonance_field_score_eff
                                .get(i)
                                .copied()
                                .unwrap_or(f32::MIN);
                            let base = if let Some(slot) = body_cache {
                                body_workspace.as_ref().unwrap().candidate_mass(slot, i)
                            } else {
                                field_density_mass(*target, landscape, i, gap_ref)
                            };
                            base * unit_gaussian(s - t, tension_sigma)
                        }
                        None => {
                            if let Some(slot) = body_cache {
                                body_workspace.as_ref().unwrap().candidate_mass(slot, i)
                            } else {
                                field_density_mass(*target, landscape, i, gap_ref)
                            }
                        }
                    };
                    let w = if occupied { 0.0 } else { raw.max(0.0) };
                    let w = if w.is_finite() { w } else { 0.0 };
                    weights.push((w, occupied));
                    sum += w;
                }
                // Fallback: if mass sums to zero, use uniform over unoccupied bins.
                if !(sum > 0.0 && sum.is_finite()) {
                    for (w, occupied) in &mut weights {
                        *w = if *occupied && has_unoccupied {
                            0.0
                        } else {
                            1.0
                        };
                    }
                }
                if temporal {
                    let mut conditioned = self.harmonic_spawn_weights.borrow_mut();
                    conditioned.resize(weights.len(), 0.);
                    for (out, (weight, _)) in conditioned.iter_mut().zip(&weights) {
                        *out = f64::from(*weight);
                    }
                    let mut shadow = rng.clone();
                    let baseline = if body_cache.is_some() {
                        let total = weights.iter().map(|(w, _)| *w).sum::<f32>();
                        let draw = shadow.random_range(0.0..total);
                        let mut cumulative = 0.;
                        weights
                            .iter()
                            .position(|(w, _)| {
                                cumulative += *w;
                                draw < cumulative
                            })
                            .unwrap_or(range_len - 1)
                    } else {
                        WeightedIndex::new(weights.iter().map(|(w, _)| *w))
                            .map_or_else(|_| 0, |d| d.sample(&mut shadow))
                    };
                    self.harmonic_spawn.borrow().as_ref().unwrap().mix(
                        &mut conditioned,
                        baseline,
                        |i| Some(bin_freq(idx_min + i).log2()),
                    );
                    for ((weight, _), value) in weights.iter_mut().zip(conditioned.iter()) {
                        *weight = *value as f32;
                    }
                }
                if body_cache.is_some() {
                    let total = weights.iter().map(|(w, _)| *w).sum::<f32>();
                    let draw = rng.random_range(0.0..total);
                    let mut cumulative = 0.0;
                    let offset = weights
                        .iter()
                        .position(|(w, _)| {
                            cumulative += *w;
                            draw < cumulative
                        })
                        .unwrap_or(range_len - 1);
                    body_workspace.as_mut().unwrap().selection_weights = weights;
                    idx_min + offset
                } else {
                    let ws: Vec<f32> = weights.iter().map(|(w, _)| *w).collect();
                    if let Ok(dist) = WeightedIndex::new(&ws) {
                        idx_min + dist.sample(rng)
                    } else {
                        idx_min + rng.random_range(0..range_len)
                    }
                }
            }
            SpawnStrategy::Linear { .. } => idx_min,
        };

        jitter_free_bin(pick_idx, rng)
    }

    fn decide_support_frequency<R: Rng + Clone>(
        &self,
        strategy: &SpawnStrategy,
        landscape: &Landscape,
        rng: &mut R,
        reserved: &[f32],
        control: Option<&crate::scenario::control::VoiceControl>,
        condition_temporal: bool,
    ) -> f32 {
        let SpawnStrategy::Field {
            target,
            sampling,
            tension,
            ..
        } = strategy
        else {
            unreachable!()
        };
        let snapshot = landscape.support.as_ref().unwrap();
        let space = &landscape.space;
        let (a, b) = strategy.freq_range_hz();
        let lo_hz = a.min(b).max(space.fmin);
        let hi_hz = a.max(b).min(space.fmax);
        let lo = space.nearest_index(lo_hz);
        let hi = space.nearest_index(hi_hz);
        let density = landscape.support_single_density(
            control.map_or(crate::scenario::control::BodyControl::default().amp, |c| {
                c.body.amp
            }),
        );
        let mut scratch = self.support_scratch.borrow_mut();
        let [scores, masses] = &mut *scratch;
        assert_eq!(
            scores.len(),
            space.n_bins(),
            "support spawn scratch must be prepared before running"
        );
        let weights = landscape.support_weights;
        // Prepare one self-reported body before scanning candidate positions.
        let template = self.spawn_body_template(control, landscape, (lo_hz * hi_hz).sqrt());
        let temporal = condition_temporal
            && self.refresh_spawn_relations(landscape, control, template.clone());
        let mut minimum = f64::MAX;
        let mut maximum = -f64::MAX;
        let gap_ref = landscape.subjective_intensity[lo..=hi]
            .iter()
            .copied()
            .fold(0., f32::max);
        let choice_base = |i: usize| match target {
            FieldTarget::Consonance => landscape.consonance_field_score_eff[i],
            FieldTarget::Dissonance => -landscape.consonance_field_score_eff[i],
            _ => field_peak_score(*target, landscape, i),
        };
        let mut body_workspace = self.birth_surrogate.borrow_mut();
        let body_cache = if let (Some(work), Some(control)) = (body_workspace.as_mut(), control) {
            work.candidates(
                control,
                landscape,
                self.current_frame,
                (lo_hz, hi_hz),
                |i| {
                    if *sampling == FieldSampling::Peak {
                        choice_base(i)
                    } else {
                        field_density_mass(*target, landscape, i, gap_ref)
                    }
                },
            )
        } else {
            None
        };
        for i in lo..=hi {
            let body_base =
                body_cache.map(|slot| body_workspace.as_ref().unwrap().candidate_mass(slot, i));
            let base = if *sampling == FieldSampling::Peak {
                body_base.unwrap_or_else(|| choice_base(i))
            } else {
                choice_base(i)
            };
            let value = weights.value(template.as_ref().map_or_else(
                || snapshot.candidate(&[(i, density)]),
                |t| snapshot.template_features(i, t),
            ));
            scores[i] = (f64::from(base) + value).clamp(-f64::MAX, f64::MAX);
            minimum = minimum.min(scores[i]);
            maximum = maximum.max(scores[i]);
            let base_mass = if *sampling == FieldSampling::Density {
                body_base.unwrap_or_else(|| field_density_mass(*target, landscape, i, gap_ref))
            } else {
                field_density_mass(*target, landscape, i, gap_ref)
            };
            masses[i] = if base_mass > 0. {
                (f64::from(base_mass).ln()
                    + (f64::from(landscape.consonance_representation.beta) * value)
                        .clamp(-f64::MAX, f64::MAX))
                .clamp(-f64::MAX, f64::MAX)
            } else {
                f64::NEG_INFINITY
            };
        }
        let tension = if *target == FieldTarget::Consonance {
            tension.clamp(0., 1.) as f64
        } else {
            0.
        };
        if tension > 0. && maximum > minimum {
            let target = (1. - tension) * maximum + tension * minimum;
            let sigma = ((maximum / 2. - minimum / 2.)
                * (2. * f64::from(TENSION_LEVEL_SIGMA_FRAC)))
            .max(1e-6);
            for i in lo..=hi {
                let difference = (scores[i] - target).abs();
                masses[i] -= 0.5 * (difference / sigma).powi(2);
                scores[i] = -difference;
            }
        }
        let free = |i| {
            !self.is_range_occupied_with(space.freq_of_index(i), strategy.min_dist_erb(), reserved)
        };
        let has_free = (lo..=hi).any(free);
        let selected = if *sampling == FieldSampling::Peak {
            let baseline = (lo..=hi)
                .filter(|&i| !has_free || free(i))
                .max_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(b.cmp(&a)))
                .unwrap();
            if temporal {
                let mut candidates = self.harmonic_spawn_candidates.borrow_mut();
                candidates.clear();
                candidates.extend((lo..=hi).filter(|&i| !has_free || free(i)));
                let baseline = candidates.iter().position(|&i| i == baseline).unwrap();
                let chosen = self
                    .harmonic_spawn
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .choose_scored(candidates.len(), baseline, |i| {
                        (
                            Some(space.centers_log2[candidates[i]]),
                            None,
                            scores[candidates[i]],
                            false,
                        )
                    });
                candidates[chosen]
            } else {
                baseline
            }
        } else {
            let max_log = (lo..=hi)
                .filter(|&i| !has_free || free(i))
                .map(|i| masses[i])
                .fold(f64::NEG_INFINITY, f64::max);
            let mut total = 0.;
            for i in lo..=hi {
                masses[i] = if has_free && !free(i) {
                    0.
                } else if max_log.is_finite() {
                    (masses[i] - max_log).exp()
                } else {
                    1.
                };
                total += masses[i];
            }
            if temporal {
                let draw = rng.clone().random_range(0. ..total);
                let mut cumulative = 0.;
                let baseline = (lo..=hi)
                    .find(|&i| {
                        cumulative += masses[i];
                        draw < cumulative
                    })
                    .unwrap_or(hi)
                    - lo;
                self.harmonic_spawn.borrow().as_ref().unwrap().mix(
                    &mut masses[lo..=hi],
                    baseline,
                    |i| Some(space.centers_log2[lo + i]),
                );
            }
            let draw = rng.random_range(0. ..total);
            let mut cumulative = 0.;
            (lo..=hi)
                .find(|&i| {
                    cumulative += masses[i];
                    draw < cumulative
                })
                .unwrap_or(hi)
        };
        let center = space.centers_log2[selected];
        tracing::debug!(
            support_evaluation = if template.is_some() {
                "body_partials"
            } else {
                "single_partial"
            },
            selected_bin = selected,
            "support birth choice"
        );
        #[cfg(test)]
        if std::env::var_os("L1_SAMPLE").is_some() {
            println!(
                "HV6_SUPPORT_BIRTH source={} bin={selected}",
                if template.is_some() {
                    "body_partials"
                } else {
                    "single_partial"
                }
            );
        }
        let low = (center - space.step() * 0.5).max(lo_hz.log2());
        let high = (center + space.step() * 0.5).min(hi_hz.log2());
        if low < high {
            for _ in 0..16 {
                let hz = rng.random_range(low..high).exp2();
                if !self.is_range_occupied_with(hz, strategy.min_dist_erb(), reserved) {
                    return hz;
                }
            }
        }
        space.freq_of_index(selected).clamp(lo_hz, hi_hz)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn resolve_strategy_frequency<R: Rng + Clone>(
        &self,
        strategy: &SpawnStrategy,
        landscape: &Landscape,
        rng: &mut R,
        reserved: &[f32],
        members: (usize, usize),
        control: &crate::scenario::control::VoiceControl,
        condition_temporal: bool,
    ) -> f32 {
        let (member_idx, member_count) = members;
        match strategy {
            SpawnStrategy::Linear {
                start_freq,
                end_freq,
            } => {
                if member_count <= 1 {
                    *start_freq
                } else {
                    let t = member_idx as f32 / (member_count - 1) as f32;
                    start_freq + (end_freq - start_freq) * t
                }
            }
            _ => self.decide_frequency_with_relations(
                strategy,
                landscape,
                rng,
                reserved,
                Some(control),
                condition_temporal,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::test_pop;
    use super::*;
    use crate::core::log2space::Log2Space;
    use rand::SeedableRng;
    use std::collections::HashSet;

    #[test]
    fn support_body_partials_change_peak_density_and_tension_choices() {
        use crate::core::harmonic_support::{Geometry, Snapshot};
        use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
        use crate::scenario::control::{BodyMethod, VoiceControl};
        let space = Log2Space::new(55., 3520., 24);
        let mut landscape = Landscape::new(space.clone());
        landscape.consonance_density_mass_eff.fill(1.);
        let mut pop = test_pop();
        pop.prepare_support(space.n_bins());
        let geometry = Geometry::new(
            &space,
            &HarmonicityKernel::new(&space, HarmonicityParams::default()),
        );
        let mut snapshot = Snapshot::new(geometry);
        let mut e = vec![0.; space.n_bins()];
        e[60] = 1.;
        e[84] = 0.7;
        snapshot.observe(&e, 1e-12, 512);
        landscape.support = Some(std::sync::Arc::new(snapshot));
        for method in [BodyMethod::Sine, BodyMethod::Harmonic, BodyMethod::Modal] {
            let mut control = VoiceControl::default();
            control.body.method = method;
            control.body.amp = 0.2;
            for (sampling, tension) in [
                (FieldSampling::Peak, 0.),
                (FieldSampling::Density, 0.),
                (FieldSampling::Peak, 0.75),
            ] {
                let strategy = SpawnStrategy::Field {
                    target: FieldTarget::Consonance,
                    sampling,
                    min_freq: 110.,
                    max_freq: 880.,
                    min_dist_erb: 0.,
                    tension,
                };
                let mut different = false;
                for seed in 0..8 {
                    let mut a = rand::rngs::SmallRng::seed_from_u64(seed);
                    let mut b = rand::rngs::SmallRng::seed_from_u64(seed);
                    landscape.support_weights.0 = [1000., 0., 0.];
                    let positive =
                        pop.decide_frequency(&strategy, &landscape, &mut a, &[], Some(&control));
                    landscape.support_weights.0 = [-1000., 0., 0.];
                    let negative =
                        pop.decide_frequency(&strategy, &landscape, &mut b, &[], Some(&control));
                    different |= space.nearest_index(positive) != space.nearest_index(negative);
                }
                assert!(different, "{method:?} {sampling:?} tension={tension}");
            }
        }
    }

    #[test]
    fn support_spawn_zero_preserves_rng_and_each_axis_changes_peak_choice() {
        use crate::core::harmonic_support::{Geometry, Snapshot};
        use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
        let space = Log2Space::new(55., 1760., 24);
        let mut landscape = Landscape::new(space.clone());
        landscape.consonance_density_mass_eff.fill(1.);
        let mut pop = test_pop();
        pop.prepare_support(space.n_bins());
        let geometry = Geometry::new(
            &space,
            &HarmonicityKernel::new(&space, HarmonicityParams::default()),
        );
        let mut snapshot = Snapshot::new(geometry);
        let mut density = vec![0.; space.n_bins()];
        density[72] = 1.;
        snapshot.observe(&density, 1e-12, 512);
        let mut strategy = SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Density,
            min_freq: space.fmin,
            max_freq: space.fmax,
            min_dist_erb: 0.,
            tension: 0.,
        };
        let mut a = rand::rngs::SmallRng::seed_from_u64(735);
        let mut b = rand::rngs::SmallRng::seed_from_u64(735);
        let original = pop.decide_frequency(&strategy, &landscape, &mut a, &[], None);
        landscape.support = Some(std::sync::Arc::new(snapshot));
        assert_eq!(
            original.to_bits(),
            pop.decide_frequency(&strategy, &landscape, &mut b, &[], None)
                .to_bits()
        );
        assert_eq!(a.random::<u64>(), b.random::<u64>());
        if let SpawnStrategy::Field { sampling, .. } = &mut strategy {
            *sampling = FieldSampling::Peak;
        }
        for axis in 0..3 {
            landscape.support_weights = Default::default();
            let mut a = rand::rngs::SmallRng::seed_from_u64(735);
            let mut b = rand::rngs::SmallRng::seed_from_u64(735);
            let neutral = pop.decide_frequency(&strategy, &landscape, &mut a, &[], None);
            landscape.support_weights.set(axis, 1.);
            let changed = pop.decide_frequency(&strategy, &landscape, &mut b, &[], None);
            assert_ne!(
                space.nearest_index(neutral),
                space.nearest_index(changed),
                "axis {axis}"
            );
        }
        landscape.support_weights.0 = [f64::MAX; 3];
        if let SpawnStrategy::Field { sampling, .. } = &mut strategy {
            *sampling = FieldSampling::Density;
        }
        let only = 60;
        landscape.consonance_density_mass_eff.fill(0.);
        landscape.consonance_density_mass_eff[only] = 1.;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(10);
        for _ in 0..16 {
            assert_eq!(
                space.nearest_index(pop.decide_frequency(
                    &strategy,
                    &landscape,
                    &mut rng,
                    &[],
                    None
                )),
                only
            );
        }
    }

    #[test]
    fn harmonic_relations_spawn_conditions_peak_density_and_no_birth() {
        use crate::core::harmonic_support::preview::{Observer, Preview};
        use crate::temporal_cognition::harmonic_relations::{Acoustic, Bin, Value, consumer};
        let (mut landscape, pitches) = consumer::test_landscape(1., false);
        let mut pop = test_pop();
        pop.prepare_support(landscape.space.n_bins());
        landscape.consonance_field_level_eff.fill(0.);
        landscape.consonance_field_level_eff[80] = 0.1;
        landscape.consonance_density_mass_eff.fill(0.);
        landscape.consonance_density_mass_eff[20] = 1.;
        landscape.consonance_density_mass_eff[80] = 1.;
        let mut preview = Preview::new(landscape.support.as_ref().unwrap());
        let density =
            landscape.support_single_density(crate::scenario::control::BodyControl::default().amp);
        landscape.support.as_ref().unwrap().preview(
            Observer::Mixed,
            pitches[0],
            &[(0, density)],
            &mut preview,
        );
        let mut strategy = SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Peak,
            min_freq: 2.0f32.powf(pitches[0]),
            max_freq: 2.0f32.powf(pitches[1]),
            min_dist_erb: 0.,
            tension: 0.,
        };
        let mut baseline_rng = rand::rngs::SmallRng::seed_from_u64(7);
        let baseline = pop.decide_frequency(&strategy, &landscape, &mut baseline_rng, &[], None);
        assert_eq!(landscape.space.nearest_index(baseline), 80);
        let decision = Arc::make_mut(landscape.harmonic_relations.as_mut().unwrap());
        decision.prediction.unknown = 0.;
        decision.prediction.len = 1;
        let atom = &mut decision.prediction.atoms[0];
        atom.weight = 1.;
        atom.acoustic = Acoustic {
            mass: Value::exact(preview.mass),
            focus: Value::exact(preview.focus),
            competition: Value::exact(preview.conflict),
            ..Default::default()
        };
        atom.shape_valid = true;
        atom.bins = preview
            .p
            .iter()
            .enumerate()
            .filter_map(|(index, &p)| {
                (p > 0.).then_some(Bin {
                    index,
                    mass: p as f32,
                })
            })
            .collect();
        let chosen = pop.decide_frequency(&strategy, &landscape, &mut baseline_rng, &[], None);
        assert_eq!(landscape.space.nearest_index(chosen), 20);
        if let SpawnStrategy::Field { sampling, .. } = &mut strategy {
            *sampling = FieldSampling::Density;
        }
        let mut changed = false;
        for seed in 0..32 {
            let mut a = rand::rngs::SmallRng::seed_from_u64(seed);
            let mut b = rand::rngs::SmallRng::seed_from_u64(seed);
            let old = pop.decide_frequency_with_relations(
                &strategy,
                &landscape,
                &mut a,
                &[],
                None,
                false,
            );
            let conditioned = pop.decide_frequency(&strategy, &landscape, &mut b, &[], None);
            changed |= old.to_bits() != conditioned.to_bits();
            assert_eq!(a.random::<u64>(), b.random::<u64>());
        }
        assert!(changed);
        let decision = Arc::make_mut(landscape.harmonic_relations.as_mut().unwrap());
        decision.prediction.atoms[0].shape_valid = false;
        decision.prediction.atoms[0].acoustic = Acoustic {
            mass: Value::exact(landscape.support.as_ref().unwrap().mass),
            ..Default::default()
        };
        let control = crate::scenario::control::VoiceControl::default();
        assert!(pop.harmonic_spawn_allowed(&landscape, &control, baseline));
        landscape.harmonic_continuation_weight = 1.;
        assert!(!pop.harmonic_spawn_allowed(&landscape, &control, baseline));
        landscape.harmonic_relations_weight = 0.;
        landscape.harmonic_continuation_weight = 0.;
        assert!(pop.harmonic_spawn_allowed(&landscape, &control, baseline));
    }

    #[test]
    fn harmonic_relations_zero_preserves_spawn_choice_and_rng_with_vertical_axes() {
        use crate::core::harmonic_support::{Geometry, Snapshot};
        use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
        use crate::scenario::control::VoiceControl;
        let space = Log2Space::new(55., 1760., 24);
        let mut landscape = Landscape::new(space.clone());
        landscape.consonance_density_mass_eff.fill(1.);
        let geometry = Geometry::new(
            &space,
            &HarmonicityKernel::new(&space, HarmonicityParams::default()),
        );
        let mut snapshot = Snapshot::new(geometry);
        let mut density = vec![0.; space.n_bins()];
        density[72] = 1.;
        snapshot.observe(&density, 1e-12, 512);
        landscape.support = Some(Arc::new(snapshot));
        let mut pop = test_pop();
        pop.prepare_support(space.n_bins());
        let control = VoiceControl::default();
        for weights in [[0.; 3], [0.3, 0.2, 0.4]] {
            landscape.support_weights.0 = weights;
            for (sampling, tension) in [
                (FieldSampling::Density, 0.),
                (FieldSampling::Peak, 0.),
                (FieldSampling::Peak, 0.75),
            ] {
                let strategy = SpawnStrategy::Field {
                    target: FieldTarget::Consonance,
                    sampling,
                    min_freq: 110.,
                    max_freq: 880.,
                    min_dist_erb: 0.,
                    tension,
                };
                for seed in 0..8 {
                    let mut a = rand::rngs::SmallRng::seed_from_u64(seed);
                    let mut b = rand::rngs::SmallRng::seed_from_u64(seed);
                    let old = pop.decide_frequency_with_relations(
                        &strategy,
                        &landscape,
                        &mut a,
                        &[],
                        Some(&control),
                        false,
                    );
                    let zero = pop.decide_frequency_with_relations(
                        &strategy,
                        &landscape,
                        &mut b,
                        &[],
                        Some(&control),
                        true,
                    );
                    assert_eq!(old.to_bits(), zero.to_bits());
                    assert!(pop.harmonic_spawn_allowed(&landscape, &control, zero));
                    assert_eq!(a.random::<u64>(), b.random::<u64>());
                    assert!(pop.harmonic_spawn.borrow().is_none());
                }
            }
        }
    }

    #[test]
    fn birth_surrogate_keeps_occupancy_spacing_and_zero_mass_fallback_in_range() {
        use crate::scenario::control::VoiceControl;
        let space = Log2Space::new(100.0, 800.0, 24);
        let mut landscape = Landscape::new(space.clone());
        let lo = 200.0;
        let hi = 400.0;
        let occupied_hz = space.freq_of_index(space.nearest_index(220.0));
        let mut pop = test_pop();
        pop.enable_birth_surrogate(0.23, 1e-4, Default::default(), false);
        let control = VoiceControl::default();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(73);
        let strategy = SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Density,
            min_freq: lo,
            max_freq: hi,
            min_dist_erb: 0.5,
            tension: 0.0,
        };
        for mass in [0.0, 1.0] {
            landscape.consonance_density_mass_eff.fill(mass);
            for _ in 0..64 {
                let f = pop.decide_frequency(
                    &strategy,
                    &landscape,
                    &mut rng,
                    &[occupied_hz],
                    Some(&control),
                );
                assert!((lo..=hi).contains(&f));
                assert!(!pop.is_range_occupied_with(f, 0.5, &[occupied_hz]));
            }
        }
        let reserved: Vec<f32> = (space.nearest_index(lo)..=space.nearest_index(hi))
            .map(|i| space.freq_of_index(i))
            .collect();
        landscape.consonance_density_mass_eff.fill(0.0);
        for _ in 0..64 {
            let f =
                pop.decide_frequency(&strategy, &landscape, &mut rng, &reserved, Some(&control));
            assert!((lo..=hi).contains(&f));
        }
    }

    #[test]
    fn excluded_body_uses_exact_legacy_selection_and_rng() {
        use crate::scenario::control::{BodyMethod, VoiceControl};
        let mut landscape = Landscape::new(Log2Space::new(100.0, 800.0, 24));
        for i in 0..landscape.space.n_bins() {
            landscape.consonance_density_mass_eff[i] = (i % 11) as f32 / 11.0;
            landscape.consonance_field_level_eff[i] = (i % 7) as f32 / 7.0;
        }
        let legacy = test_pop();
        let mut enabled = test_pop();
        enabled.enable_birth_surrogate(0.23, 1e-4, Default::default(), false);
        let mut control = VoiceControl::default();
        control.body.method = BodyMethod::Harmonic;
        control.body.timbre.motion = 0.9;
        let mut old_rng = rand::rngs::SmallRng::seed_from_u64(4);
        let mut new_rng = rand::rngs::SmallRng::seed_from_u64(4);
        for sampling in [FieldSampling::Density, FieldSampling::Peak] {
            let strategy = SpawnStrategy::Field {
                target: FieldTarget::Consonance,
                sampling,
                min_freq: 200.0,
                max_freq: 600.0,
                min_dist_erb: 0.3,
                tension: 0.0,
            };
            for _ in 0..32 {
                let old = legacy.decide_frequency(
                    &strategy,
                    &landscape,
                    &mut old_rng,
                    &[220.0],
                    Some(&control),
                );
                let new = enabled.decide_frequency(
                    &strategy,
                    &landscape,
                    &mut new_rng,
                    &[220.0],
                    Some(&control),
                );
                assert_eq!(old.to_bits(), new.to_bits());
                assert_eq!(old_rng.next_u64(), new_rng.next_u64());
            }
        }
    }

    #[test]
    fn harmonic_birth_reads_upper_partials_instead_of_only_the_fundamental() {
        use crate::core::mode_pattern::ModePattern;
        use crate::scenario::control::{BodyMethod, VoiceControl};
        let mut landscape = Landscape::new(Log2Space::new(100.0, 1600.0, 24));
        let upper = landscape.space.nearest_index(440.0);
        landscape.consonance_density_mass_eff[upper] = 1.0;
        let mut control = VoiceControl::default();
        control.body.method = BodyMethod::Harmonic;
        control.body.timbre.spread = 0.0;
        control.body.timbre.unison = 1;
        control.body.modes = Some(ModePattern::custom_modes(vec![1.0, 2.0]));
        let strategy = SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Density,
            min_freq: 200.0,
            max_freq: 240.0,
            min_dist_erb: 0.0,
            tension: 0.0,
        };
        for renderer_phase3 in [false, true] {
            let mut pop = test_pop();
            pop.enable_birth_surrogate(0.23, 1e-4, Default::default(), renderer_phase3);
            let mut rng = rand::rngs::SmallRng::seed_from_u64(9);
            for _ in 0..32 {
                let f = pop.decide_frequency(&strategy, &landscape, &mut rng, &[], Some(&control));
                assert_eq!(
                    landscape.space.nearest_index(f),
                    landscape.space.nearest_index(220.0)
                );
            }
        }
    }

    #[test]
    #[ignore = "release-only diagnostic cost; never a real-time acceptance gate"]
    // Preserve the runtime profile guard so debug builds still compile this ignored assay.
    #[allow(clippy::assertions_on_constants)]
    fn birth_surrogate_release_cost() {
        use crate::scenario::control::BodyMethod;
        use std::time::Instant;
        assert!(!cfg!(debug_assertions), "release profile required");
        let mut landscape = Landscape::new(Log2Space::new(55.0, 8000.0, 96));
        landscape.consonance_density_mass_eff.fill(0.5);
        for (name, count, distinct) in [
            ("one", 1, false),
            ("ten_same", 10, false),
            ("ten_distinct", 10, true),
        ] {
            let mut samples = Vec::new();
            for _ in 0..5 {
                let mut pop = super::super::Community::new(crate::core::timebase::Timebase {
                    fs: 48000.0,
                    hop: 512,
                });
                pop.enable_birth_surrogate(0.23, 1e-4, Default::default(), false);
                let mut spec = super::super::tests::spawn_spec_with_freq(440.0);
                spec.control.body.method = BodyMethod::Harmonic;
                spec.control.body.timbre.spread = 0.0;
                spec.control.body.timbre.unison = 1;
                let strategy = SpawnStrategy::Field {
                    target: FieldTarget::Consonance,
                    sampling: FieldSampling::Density,
                    min_freq: 55.0,
                    max_freq: 8000.0,
                    min_dist_erb: 0.0,
                    tension: 0.0,
                };
                let mut actions = Vec::new();
                if distinct {
                    for i in 0..count {
                        let mut child = spec.clone();
                        child.control.body.timbre.brightness = i as f32 / count as f32;
                        actions.push((vec![i as u64 + 1], child));
                    }
                } else {
                    actions.push(((1..=count as u64).collect(), spec));
                }
                let start = Instant::now();
                for (ids, spec) in actions {
                    pop.apply_action(
                        crate::scenario::Action::Spawn {
                            population_id: 1,
                            ids: ids.into(),
                            spec,
                            strategy: Some(strategy.clone()),
                        },
                        &landscape,
                    );
                }
                samples.push(start.elapsed().as_secs_f64() * 1000.0);
                assert_eq!(pop.voices.len(), count);
            }
            println!(
                "{}",
                serde_json::json!({"schema":"b4-birth-cost", "case":name,
                "births":count, "milliseconds":samples,
                "scope":"legacy harmonic actual spawn path; scratch allocated before timer; diagnostic only"})
            );
        }
    }

    #[test]
    fn field_placement_keeps_narrow_and_clipped_ranges_in_hz() {
        let space = Log2Space::new(100.0, 410.0, 24);
        let landscape = Landscape::new(space.clone());
        let pop = test_pop();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
        for target in [
            FieldTarget::Consonance,
            FieldTarget::Dissonance,
            FieldTarget::Edge,
            FieldTarget::Gap,
        ] {
            for sampling in [FieldSampling::Peak, FieldSampling::Density] {
                for (min_freq, max_freq) in [
                    (220.0f32, 220.1f32),
                    (220.1, 220.0),
                    (220.0, 220.0),
                    (90.0, 100.1),
                    (408.0, 420.0),
                ] {
                    let lo = min_freq.min(max_freq).max(space.fmin);
                    let hi = min_freq.max(max_freq).min(space.fmax);
                    for min_dist_erb in [0.0, 100.0] {
                        let strategy = SpawnStrategy::Field {
                            target,
                            sampling,
                            min_freq,
                            max_freq,
                            min_dist_erb,
                            tension: 0.0,
                        };
                        // The large spacing forces the all-occupied fallback.
                        for _ in 0..32 {
                            let freq =
                                pop.decide_frequency(&strategy, &landscape, &mut rng, &[lo], None);
                            assert!(
                                (lo..=hi).contains(&freq),
                                "{target:?}/{sampling:?}: {freq} Hz outside [{lo}, {hi}]"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn field_placement_without_analysis_overlap_stays_in_requested_range() {
        let landscape = Landscape::new(Log2Space::new(100.0, 400.0, 24));
        let pop = test_pop();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(2);
        for target in [
            FieldTarget::Consonance,
            FieldTarget::Dissonance,
            FieldTarget::Edge,
            FieldTarget::Gap,
            FieldTarget::Uniform,
        ] {
            for (min_freq, max_freq) in [(50.0f32, 60.0f32), (600.0, 500.0)] {
                let strategy = SpawnStrategy::Field {
                    target,
                    sampling: FieldSampling::Density,
                    min_freq,
                    max_freq,
                    min_dist_erb: 0.0,
                    tension: 0.0,
                };
                let lo = min_freq.min(max_freq);
                let hi = min_freq.max(max_freq);
                let mut seen = HashSet::new();
                for _ in 0..32 {
                    let freq = pop.decide_frequency(&strategy, &landscape, &mut rng, &[], None);
                    assert!((lo..=hi).contains(&freq), "{target:?}: {freq} Hz");
                    seen.insert(freq.to_bits());
                }
                assert!(seen.len() > 1, "fallback must sample the requested range");
            }
        }
    }

    #[test]
    fn field_placement_checks_spacing_at_clipped_bin_frequency() {
        let space = Log2Space::new(100.0, 400.0, 24);
        let mut landscape = Landscape::new(space.clone());
        let min_freq = 219.0;
        let idx_min = space.nearest_index(min_freq);
        let next_freq = space.freq_of_index(idx_min + 1);
        landscape.consonance_field_level_eff[idx_min] = 1.0;
        landscape.consonance_density_mass_eff[idx_min] = 1.0;
        landscape.consonance_field_level_eff[idx_min + 1] = 0.5;
        landscape.consonance_density_mass_eff[idx_min + 1] = 0.5;
        let pop = test_pop();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(3);
        for sampling in [FieldSampling::Peak, FieldSampling::Density] {
            let strategy = SpawnStrategy::Field {
                target: FieldTarget::Consonance,
                sampling,
                min_freq,
                max_freq: next_freq,
                min_dist_erb: 0.01,
                tension: 0.0,
            };
            for _ in 0..32 {
                let freq = pop.decide_frequency(&strategy, &landscape, &mut rng, &[min_freq], None);
                assert!((min_freq..=next_freq).contains(&freq));
                assert_eq!(space.nearest_index(freq), idx_min + 1);
                assert!(!pop.is_range_occupied_with(freq, 0.01, &[min_freq]));
            }
        }
    }

    #[test]
    fn decide_frequency_uses_consonance_field_level() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let mut landscape = Landscape::new(space.clone());
        landscape.consonance_field_score.fill(-10.0);
        landscape.consonance_field_score_eff.fill(-10.0);
        landscape.consonance_field_level.fill(0.0);
        landscape.consonance_field_level_eff.fill(0.0);

        let idx_high = space.index_of_freq(200.0).expect("idx");
        let idx_raw = space.index_of_freq(300.0).expect("idx");
        landscape.consonance_field_level[idx_high] = 1.0;
        landscape.consonance_field_level_eff[idx_high] = 1.0;
        landscape.consonance_field_score[idx_raw] = 10.0;
        landscape.consonance_field_score_eff[idx_raw] = 10.0;

        let pop = test_pop();
        let strategy = SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Peak,
            min_freq: 100.0,
            max_freq: 400.0,
            min_dist_erb: 0.0,
            tension: 0.0,
        };
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        let freq = pop.decide_frequency(&strategy, &landscape, &mut rng, &[], None);
        let picked_idx = space.index_of_freq(freq).expect("picked idx");
        assert_eq!(picked_idx, idx_high);
    }

    #[test]
    fn consonance_density_sampling_uses_density_pmf() {
        let space = Log2Space::new(100.0, 400.0, 24);
        let mut landscape = Landscape::new(space.clone());
        landscape.consonance_density_mass.fill(0.0);
        landscape.consonance_density_mass_eff.fill(0.0);
        let idx_target = space.index_of_freq(220.0).expect("idx target");
        landscape.consonance_density_mass[idx_target] = 1.0;
        landscape.consonance_density_mass_eff[idx_target] = 1.0;

        let pop = test_pop();
        let strategy = SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Density,
            min_freq: space.fmin,
            max_freq: space.fmax,
            min_dist_erb: 0.0,
            tension: 0.0,
        };
        let mut rng = rand::rngs::SmallRng::seed_from_u64(1234);

        for _ in 0..64 {
            let freq = pop.decide_frequency(&strategy, &landscape, &mut rng, &[], None);
            let picked_idx = space.index_of_freq(freq).expect("picked idx");
            assert_eq!(picked_idx, idx_target);
        }
    }

    #[test]
    fn consonance_tension_peak_targets_a_lower_step() {
        let space = Log2Space::new(100.0, 800.0, 48);
        let mut landscape = Landscape::new(space.clone());
        // Background well below the three steps so L_min is the background.
        for s in landscape.consonance_field_score.iter_mut() {
            *s = -1.0;
        }
        for s in landscape.consonance_field_score_eff.iter_mut() {
            *s = -1.0;
        }
        let idx_strong = space.index_of_freq(200.0).expect("strong");
        let idx_mid = space.index_of_freq(300.0).expect("mid");
        let idx_weak = space.index_of_freq(500.0).expect("weak");
        landscape.consonance_field_score[idx_strong] = 1.0; // L_max
        landscape.consonance_field_score[idx_mid] = 0.5;
        landscape.consonance_field_score[idx_weak] = 0.0;
        landscape.consonance_field_score_eff[idx_strong] = 1.0; // L_max
        landscape.consonance_field_score_eff[idx_mid] = 0.5;
        landscape.consonance_field_score_eff[idx_weak] = 0.0;

        let pop = test_pop();
        let mk = |t: f32| SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Peak,
            min_freq: 100.0,
            max_freq: 800.0,
            min_dist_erb: 0.0,
            tension: t,
        };
        let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
        // L_max=1, L_min=-1: target = 1 - 2*tension.
        // tension=0.25 -> 0.5 (mid step); tension=0.5 -> 0.0 (weak step).
        let f_mid = pop.decide_frequency(&mk(0.25), &landscape, &mut rng, &[], None);
        assert_eq!(space.index_of_freq(f_mid).expect("idx"), idx_mid);
        let f_weak = pop.decide_frequency(&mk(0.5), &landscape, &mut rng, &[], None);
        assert_eq!(space.index_of_freq(f_weak).expect("idx"), idx_weak);
    }

    #[test]
    fn consonance_density_range_zero_weights_fallback_is_range_uniform() {
        let space = Log2Space::new(100.0, 400.0, 24);
        let mut landscape = Landscape::new(space.clone());
        landscape.consonance_density_mass.fill(1.0);
        landscape.consonance_density_mass_eff.fill(1.0);

        let idx_min = 6usize;
        let idx_max = 12usize;
        for i in idx_min..=idx_max {
            landscape.consonance_density_mass[i] = 0.0;
            landscape.consonance_density_mass_eff[i] = 0.0;
        }

        let pop = test_pop();
        let strategy = SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Density,
            min_freq: space.freq_of_index(idx_min),
            max_freq: space.freq_of_index(idx_max),
            min_dist_erb: 0.0,
            tension: 0.0,
        };
        let mut rng = rand::rngs::SmallRng::seed_from_u64(11);
        let mut seen = HashSet::new();

        for _ in 0..64 {
            let freq = pop.decide_frequency(&strategy, &landscape, &mut rng, &[], None);
            assert!((space.freq_of_index(idx_min)..=space.freq_of_index(idx_max)).contains(&freq));
            let picked_idx = space.index_of_freq(freq).expect("picked idx");
            assert!(
                (idx_min..=idx_max).contains(&picked_idx),
                "picked_idx={picked_idx}, expected in [{idx_min},{idx_max}]"
            );
            seen.insert(picked_idx);
        }

        assert!(
            seen.len() > 1,
            "range fallback should not collapse to a single fixed index"
        );
    }

    #[test]
    fn consonance_density_range_all_occupied_fallback_does_not_panic() {
        let space = Log2Space::new(100.0, 400.0, 24);
        let mut landscape = Landscape::new(space.clone());
        landscape.consonance_density_mass.fill(1.0);
        landscape.consonance_density_mass_eff.fill(1.0);

        let idx_min = 8usize;
        let idx_max = 14usize;
        let reserved: Vec<f32> = (idx_min..=idx_max)
            .map(|i| space.freq_of_index(i))
            .collect();

        let pop = test_pop();
        let strategy = SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Density,
            min_freq: space.freq_of_index(idx_min),
            max_freq: space.freq_of_index(idx_max),
            min_dist_erb: 1e-4,
            tension: 0.0,
        };
        let mut rng = rand::rngs::SmallRng::seed_from_u64(12);

        for _ in 0..64 {
            let freq = pop.decide_frequency(&strategy, &landscape, &mut rng, &reserved, None);
            assert!((space.freq_of_index(idx_min)..=space.freq_of_index(idx_max)).contains(&freq));
            let picked_idx = space.index_of_freq(freq).expect("picked idx");
            assert!(
                (idx_min..=idx_max).contains(&picked_idx),
                "picked_idx={picked_idx}, expected in [{idx_min},{idx_max}]"
            );
        }
    }

    #[test]
    fn consonance_density_avoids_occupied_when_unoccupied_exists() {
        let space = Log2Space::new(100.0, 400.0, 24);
        let mut landscape = Landscape::new(space.clone());
        landscape.consonance_density_mass.fill(1.0);
        landscape.consonance_density_mass_eff.fill(1.0);

        let idx_min = 5usize;
        let idx_max = 11usize;
        let idx_occupied = 8usize;
        let reserved = vec![space.freq_of_index(idx_occupied)];

        let pop = test_pop();
        let strategy = SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Density,
            min_freq: space.freq_of_index(idx_min),
            max_freq: space.freq_of_index(idx_max),
            min_dist_erb: 1e-4,
            tension: 0.0,
        };
        let mut rng = rand::rngs::SmallRng::seed_from_u64(13);

        for _ in 0..100 {
            let freq = pop.decide_frequency(&strategy, &landscape, &mut rng, &reserved, None);
            let picked_idx = space.index_of_freq(freq).expect("picked idx");
            assert!(
                (idx_min..=idx_max).contains(&picked_idx),
                "picked_idx={picked_idx}, expected in [{idx_min},{idx_max}]"
            );
            assert_ne!(
                picked_idx, idx_occupied,
                "occupied index should not be chosen when unoccupied bins exist"
            );
        }
    }

    #[test]
    fn consonance_density_zero_sum_fallback_still_avoids_occupied() {
        let space = Log2Space::new(100.0, 400.0, 24);
        let mut landscape = Landscape::new(space.clone());
        landscape.consonance_density_mass.fill(1.0);
        landscape.consonance_density_mass_eff.fill(1.0);

        let idx_min = 5usize;
        let idx_max = 11usize;
        for i in idx_min..=idx_max {
            landscape.consonance_density_mass[i] = 0.0;
            landscape.consonance_density_mass_eff[i] = 0.0;
        }
        let idx_occupied = 8usize;
        let reserved = vec![space.freq_of_index(idx_occupied)];

        let pop = test_pop();
        let strategy = SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Density,
            min_freq: space.freq_of_index(idx_min),
            max_freq: space.freq_of_index(idx_max),
            min_dist_erb: 1e-4,
            tension: 0.0,
        };
        let mut rng = rand::rngs::SmallRng::seed_from_u64(14);

        for _ in 0..100 {
            let freq = pop.decide_frequency(&strategy, &landscape, &mut rng, &reserved, None);
            let picked_idx = space.index_of_freq(freq).expect("picked idx");
            assert!(
                (idx_min..=idx_max).contains(&picked_idx),
                "picked_idx={picked_idx}, expected in [{idx_min},{idx_max}]"
            );
            assert_ne!(
                picked_idx, idx_occupied,
                "occupied index should not be chosen in zero-sum fallback"
            );
        }
    }

    #[test]
    fn consonance_density_reversed_range_is_handled_safely() {
        let space = Log2Space::new(100.0, 400.0, 24);
        let mut landscape = Landscape::new(space.clone());
        landscape.consonance_density_mass.fill(0.0);
        landscape.consonance_density_mass_eff.fill(0.0);

        let idx_low = 6usize;
        let idx_high = 12usize;
        let idx_target = 9usize;
        landscape.consonance_density_mass[idx_target] = 1.0;
        landscape.consonance_density_mass_eff[idx_target] = 1.0;

        let pop = test_pop();
        let strategy = SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Density,
            // Intentionally reversed order to emulate Rhai-side input mistakes.
            min_freq: space.freq_of_index(idx_high),
            max_freq: space.freq_of_index(idx_low),
            min_dist_erb: 0.0,
            tension: 0.0,
        };
        let mut rng = rand::rngs::SmallRng::seed_from_u64(15);

        for _ in 0..64 {
            let freq = pop.decide_frequency(&strategy, &landscape, &mut rng, &[], None);
            let picked_idx = space.index_of_freq(freq).expect("picked idx");
            assert!(
                (idx_low..=idx_high).contains(&picked_idx),
                "picked_idx={picked_idx}, expected in [{idx_low},{idx_high}]"
            );
            assert_eq!(picked_idx, idx_target);
        }
    }
}
