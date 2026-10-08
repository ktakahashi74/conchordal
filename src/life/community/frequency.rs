use super::Community;
use crate::core::float::unit_gaussian;
use crate::core::landscape::LandscapeFrame;
use crate::life::voice::SoundBody;
use crate::scenario::{FieldSampling, FieldTarget, SpawnStrategy};
use rand::{Rng, RngExt, distr::Distribution, distr::weighted::WeightedIndex};

/// Width (fraction of the range's field_score span) of the Gaussian that weights
/// peaks near the tension target in density placement.
const TENSION_LEVEL_SIGMA_FRAC: f32 = 0.15;

/// Peak score for a field target at bin `i` (higher = better extremum).
fn field_peak_score(target: FieldTarget, landscape: &LandscapeFrame, i: usize) -> f32 {
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
/// loudest in-range intensity, used only by `Gap`.
fn field_density_mass(
    target: FieldTarget,
    landscape: &LandscapeFrame,
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

    fn decide_frequency<R: Rng + ?Sized>(
        &self,
        strategy: &SpawnStrategy,
        landscape: &LandscapeFrame,
        rng: &mut R,
        reserved: &[f32],
        control: Option<&crate::scenario::control::VoiceControl>,
    ) -> f32 {
        let space = &landscape.space;
        let n_bins = space.n_bins();
        let (min_freq, max_freq) = strategy.freq_range_hz();
        let (min_freq, max_freq) = (min_freq.min(max_freq), min_freq.max(max_freq));
        let min_dist_erb = strategy.min_dist_erb();
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
                for i in idx_min..=idx_max {
                    let score = match target_score {
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
                    };
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
                best_free.unwrap_or(best_any).0
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

    pub(super) fn resolve_strategy_frequency<R: Rng + ?Sized>(
        &self,
        strategy: &SpawnStrategy,
        landscape: &LandscapeFrame,
        rng: &mut R,
        reserved: &[f32],
        members: (usize, usize),
        control: &crate::scenario::control::VoiceControl,
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
            _ => self.decide_frequency(strategy, landscape, rng, reserved, Some(control)),
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
    fn birth_surrogate_keeps_occupancy_spacing_and_zero_mass_fallback_in_range() {
        use crate::scenario::control::VoiceControl;
        let space = Log2Space::new(100.0, 800.0, 24);
        let mut landscape = LandscapeFrame::new(space.clone());
        let lo = 200.0;
        let hi = 400.0;
        let occupied_hz = space.freq_of_index(space.nearest_index(220.0));
        let mut pop = test_pop();
        pop.enable_birth_surrogate(0.23, 1e-4);
        let control = VoiceControl::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(73);
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
        let mut landscape = LandscapeFrame::new(Log2Space::new(100.0, 800.0, 24));
        for i in 0..landscape.space.n_bins() {
            landscape.consonance_density_mass_eff[i] = (i % 11) as f32 / 11.0;
            landscape.consonance_field_level_eff[i] = (i % 7) as f32 / 7.0;
        }
        let legacy = test_pop();
        let mut enabled = test_pop();
        enabled.enable_birth_surrogate(0.23, 1e-4);
        let mut control = VoiceControl::default();
        control.body.method = BodyMethod::Harmonic;
        control.body.timbre.motion = 0.9;
        let mut old_rng = rand::rngs::StdRng::seed_from_u64(4);
        let mut new_rng = rand::rngs::StdRng::seed_from_u64(4);
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
        let mut landscape = LandscapeFrame::new(Log2Space::new(100.0, 1600.0, 24));
        let upper = landscape.space.nearest_index(440.0);
        landscape.consonance_density_mass_eff[upper] = 1.0;
        let mut pop = test_pop();
        pop.enable_birth_surrogate(0.23, 1e-4);
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
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        for _ in 0..32 {
            let f = pop.decide_frequency(&strategy, &landscape, &mut rng, &[], Some(&control));
            assert_eq!(
                landscape.space.nearest_index(f),
                landscape.space.nearest_index(220.0)
            );
        }
    }

    #[test]
    #[ignore = "release-only diagnostic cost; never a real-time acceptance gate"]
    fn birth_surrogate_release_cost() {
        use crate::scenario::control::BodyMethod;
        use std::time::Instant;
        assert!(!cfg!(debug_assertions), "release profile required");
        let mut landscape = LandscapeFrame::new(Log2Space::new(55.0, 8000.0, 96));
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
                pop.enable_birth_surrogate(0.23, 1e-4);
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
                            ids,
                            spec,
                            strategy: Some(strategy.clone()),
                        },
                        &landscape,
                        None,
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
        let landscape = LandscapeFrame::new(space.clone());
        let pop = test_pop();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
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
        let landscape = LandscapeFrame::new(Log2Space::new(100.0, 400.0, 24));
        let pop = test_pop();
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
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
        let mut landscape = LandscapeFrame::new(space.clone());
        let min_freq = 219.0;
        let idx_min = space.nearest_index(min_freq);
        let next_freq = space.freq_of_index(idx_min + 1);
        landscape.consonance_field_level_eff[idx_min] = 1.0;
        landscape.consonance_density_mass_eff[idx_min] = 1.0;
        landscape.consonance_field_level_eff[idx_min + 1] = 0.5;
        landscape.consonance_density_mass_eff[idx_min + 1] = 0.5;
        let pop = test_pop();
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
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
        let mut landscape = LandscapeFrame::new(space.clone());
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
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let freq = pop.decide_frequency(&strategy, &landscape, &mut rng, &[], None);
        let picked_idx = space.index_of_freq(freq).expect("picked idx");
        assert_eq!(picked_idx, idx_high);
    }

    #[test]
    fn consonance_density_sampling_uses_density_pmf() {
        let space = Log2Space::new(100.0, 400.0, 24);
        let mut landscape = LandscapeFrame::new(space.clone());
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
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);

        for _ in 0..64 {
            let freq = pop.decide_frequency(&strategy, &landscape, &mut rng, &[], None);
            let picked_idx = space.index_of_freq(freq).expect("picked idx");
            assert_eq!(picked_idx, idx_target);
        }
    }

    #[test]
    fn consonance_tension_peak_targets_a_lower_step() {
        let space = Log2Space::new(100.0, 800.0, 48);
        let mut landscape = LandscapeFrame::new(space.clone());
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
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
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
        let mut landscape = LandscapeFrame::new(space.clone());
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
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
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
        let mut landscape = LandscapeFrame::new(space.clone());
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
        let mut rng = rand::rngs::StdRng::seed_from_u64(12);

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
        let mut landscape = LandscapeFrame::new(space.clone());
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
        let mut rng = rand::rngs::StdRng::seed_from_u64(13);

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
        let mut landscape = LandscapeFrame::new(space.clone());
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
        let mut rng = rand::rngs::StdRng::seed_from_u64(14);

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
        let mut landscape = LandscapeFrame::new(space.clone());
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
        let mut rng = rand::rngs::StdRng::seed_from_u64(15);

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
