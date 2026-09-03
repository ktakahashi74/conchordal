use super::{Community, ParentCandidate, RuntimePopulationState, SpawnParams, SpawnReason};
use crate::core::float::unit_gaussian;
use crate::core::landscape::LandscapeFrame;
use crate::core::peak_extraction::{PeakExtractConfig, extract_peaks_density};
use crate::life::voice::{AnyArticulationCore, SoundBody};
use crate::scenario::control::{MAX_FREQ_HZ, MIN_FREQ_HZ};
use crate::scenario::{RespawnPeakBiasConfig, RespawnPolicy, SpawnStrategy};
use rand::{
    Rng, RngExt, SeedableRng, distr::Distribution, distr::weighted::WeightedIndex, rngs::SmallRng,
};
use std::collections::BTreeMap;

const RESPAWN_CANDIDATE_COUNT: usize = 16;

pub(super) fn weighted_parent_select<R: Rng + ?Sized>(
    parents: &[ParentCandidate],
    rng: &mut R,
) -> usize {
    debug_assert!(!parents.is_empty());
    let total: f32 = parents.iter().map(|p| p.energy.max(0.0)).sum();
    if total > 0.0 && total.is_finite() {
        let mut threshold = rng.random_range(0.0..total);
        for (i, p) in parents.iter().enumerate() {
            threshold -= p.energy.max(0.0);
            if threshold <= 0.0 {
                return i;
            }
        }
        parents.len() - 1
    } else {
        rng.random_range(0..parents.len())
    }
}

pub(super) fn peak_bias_gaussian_weight(delta_st: f32, sigma_st: f32) -> f32 {
    let sigma_st = if sigma_st.is_finite() {
        sigma_st.max(1e-3)
    } else {
        9.0
    };
    unit_gaussian(delta_st, sigma_st)
}

pub(super) fn peak_bias_same_band(
    parent_freq_hz: f32,
    candidate_freq_hz: f32,
    window_cents: f32,
) -> bool {
    if !parent_freq_hz.is_finite()
        || parent_freq_hz <= 0.0
        || !candidate_freq_hz.is_finite()
        || candidate_freq_hz <= 0.0
    {
        return false;
    }
    let window_cents = window_cents.max(0.0);
    (1200.0 * (candidate_freq_hz / parent_freq_hz).log2()).abs() <= window_cents
}

pub(super) fn peak_bias_parent_octave(
    parent_freq_hz: f32,
    candidate_freq_hz: f32,
    window_cents: f32,
) -> bool {
    if !parent_freq_hz.is_finite()
        || parent_freq_hz <= 0.0
        || !candidate_freq_hz.is_finite()
        || candidate_freq_hz <= 0.0
    {
        return false;
    }
    let delta_cents = 1200.0 * (candidate_freq_hz / parent_freq_hz).log2();
    let nearest_octave = (delta_cents / 1200.0).round();
    nearest_octave.abs() >= 1.0
        && (delta_cents - nearest_octave * 1200.0).abs() <= window_cents.max(0.0)
}

pub(super) fn peak_bias_candidate_bins(
    landscape: &LandscapeFrame,
    min_hz: f32,
    max_hz: f32,
    candidate_count: usize,
) -> Vec<usize> {
    let candidate_count = candidate_count.max(1);
    let min_hz = min_hz
        .min(max_hz)
        .clamp(landscape.space.fmin.max(1e-6), landscape.space.fmax);
    let max_hz = max_hz
        .max(min_hz)
        .clamp(min_hz, landscape.space.fmax.max(min_hz));
    let mut weights = vec![0.0f32; landscape.space.n_bins()];
    for (idx, &freq_hz) in landscape.space.centers_hz.iter().enumerate() {
        if !freq_hz.is_finite() || freq_hz < min_hz || freq_hz > max_hz {
            continue;
        }
        let weight = landscape.consonance_field_score_eff[idx].max(0.0);
        if weight.is_finite() {
            weights[idx] = weight;
        }
    }

    let mut cfg = PeakExtractConfig::normal();
    cfg.max_peaks = Some(candidate_count.saturating_mul(2));
    cfg.min_prominence_db_power = 0.5;
    cfg.min_sep_erb = 0.10;
    let mut bins: Vec<usize> = extract_peaks_density(&weights, &landscape.space, &cfg)
        .into_iter()
        .filter_map(|peak| (weights[peak.bin_idx] > 0.0).then_some(peak.bin_idx))
        .collect();

    bins.sort_by(|a, b| {
        weights[*b]
            .partial_cmp(&weights[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if bins.len() < candidate_count {
        let mut ranked_bins: Vec<usize> = weights
            .iter()
            .enumerate()
            .filter_map(|(idx, weight)| (*weight > 0.0).then_some(idx))
            .collect();
        ranked_bins.sort_by(|a, b| {
            weights[*b]
                .partial_cmp(&weights[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for idx in ranked_bins {
            if bins.contains(&idx) {
                continue;
            }
            bins.push(idx);
            if bins.len() >= candidate_count {
                break;
            }
        }
    }

    bins.truncate(candidate_count);
    bins
}

pub(super) fn peak_bias_local_search_frequency(
    landscape: &LandscapeFrame,
    center_hz: f32,
    min_hz: f32,
    max_hz: f32,
    config: RespawnPeakBiasConfig,
) -> f32 {
    let lo = min_hz.min(max_hz).max(MIN_FREQ_HZ);
    let hi = max_hz.max(min_hz).clamp(lo, MAX_FREQ_HZ);
    let center_hz = center_hz.clamp(lo, hi);
    let radius_log2 = (config.local_search_radius_st.max(0.0)) / 12.0;
    let step_log2 = (if config.local_search_step_st.is_finite() {
        config.local_search_step_st.max(1e-3)
    } else {
        0.05
    }) / 12.0;
    if radius_log2 <= 0.0 {
        return center_hz;
    }
    let center_log2 = center_hz.log2();
    let min_log2 = lo.log2().max(center_log2 - radius_log2);
    let max_log2 = hi.log2().min(center_log2 + radius_log2);
    let mut best_freq = center_hz;
    let mut best_score = landscape.evaluate_pitch_score(center_hz);
    let mut cur = min_log2;
    while cur <= max_log2 + 1e-6 {
        let freq_hz = 2.0f32.powf(cur).clamp(lo, hi);
        let score = landscape.evaluate_pitch_score(freq_hz);
        if score.is_finite() && (!best_score.is_finite() || score > best_score) {
            best_score = score;
            best_freq = freq_hz;
        }
        cur += step_log2;
    }
    best_freq
}

pub(super) fn choose_candidate_by_scene_score<R: Rng + ?Sized>(
    landscape: &LandscapeFrame,
    candidates: &[f32],
    rng: &mut R,
) -> Option<f32> {
    if candidates.is_empty() {
        return None;
    }

    let mut scene_scores = Vec::with_capacity(candidates.len());
    let mut selection_weights = Vec::with_capacity(candidates.len());
    for &freq_hz in candidates {
        let scene_score = landscape.evaluate_pitch_score(freq_hz);
        scene_scores.push(scene_score);
        selection_weights.push(if scene_score.is_finite() {
            scene_score.max(0.0)
        } else {
            0.0
        });
    }

    let chosen_idx = if selection_weights
        .iter()
        .any(|weight| *weight > 0.0 && weight.is_finite())
    {
        if let Ok(dist) = WeightedIndex::new(&selection_weights) {
            dist.sample(rng)
        } else {
            selection_weights
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }
    } else {
        scene_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    };

    Some(candidates[chosen_idx])
}

impl Community {
    fn random_respawn_frequency<R: Rng + ?Sized>(
        &self,
        population: &RuntimePopulationState,
        landscape: &LandscapeFrame,
        rng: &mut R,
        member_idx: usize,
    ) -> f32 {
        if let Some(strategy) = population.strategy.as_ref() {
            let linear_idx = member_idx % population.spawn_count_hint.max(1);
            self.resolve_strategy_frequency(
                strategy,
                landscape,
                rng,
                &[],
                linear_idx,
                population.spawn_count_hint.max(1),
            )
            .max(MIN_FREQ_HZ)
        } else {
            population
                .template
                .control
                .pitch
                .freq
                .clamp(MIN_FREQ_HZ, MAX_FREQ_HZ)
        }
    }

    fn peak_biased_respawn_candidate<R: Rng + ?Sized>(
        &self,
        population: &RuntimePopulationState,
        selected_parent: Option<ParentCandidate>,
        landscape: &LandscapeFrame,
        rng: &mut R,
        member_idx: usize,
        config: RespawnPeakBiasConfig,
    ) -> Option<(f32, Option<u64>, Option<u32>)> {
        let (min_hz, max_hz) = population
            .strategy
            .as_ref()
            .map(SpawnStrategy::freq_range_hz)
            .unwrap_or_else(|| landscape.freq_bounds());
        let lo = min_hz.clamp(MIN_FREQ_HZ, MAX_FREQ_HZ);
        let hi = max_hz.clamp(lo, MAX_FREQ_HZ);
        let candidate_count = RESPAWN_CANDIDATE_COUNT;
        let candidate_bins = peak_bias_candidate_bins(landscape, lo, hi, candidate_count);

        let fallback_freq = selected_parent
            .map(|parent| parent.freq_hz.clamp(lo, hi))
            .unwrap_or_else(|| {
                self.random_respawn_frequency(population, landscape, rng, member_idx)
            });

        let chosen_freq = if candidate_bins.is_empty() {
            fallback_freq
        } else {
            let scene_exp = if config.scene_score_exponent.is_finite() {
                config.scene_score_exponent.max(0.0)
            } else {
                0.35
            };
            let parent_freq_hz = selected_parent
                .map(|parent| parent.freq_hz.max(MIN_FREQ_HZ))
                .filter(|freq_hz| freq_hz.is_finite() && *freq_hz > 0.0);
            let mut scene_weights = Vec::with_capacity(candidate_bins.len());
            let mut final_weights = Vec::with_capacity(candidate_bins.len());
            for &bin_idx in &candidate_bins {
                let center_hz = landscape.space.centers_hz[bin_idx].clamp(lo, hi);
                let mut scene_weight = landscape.consonance_field_score_eff[bin_idx].max(0.0);
                if scene_exp > 0.0 {
                    scene_weight = scene_weight.powf(scene_exp);
                }
                scene_weights.push(scene_weight);

                let mut final_weight = scene_weight;
                if let Some(parent_freq_hz) = parent_freq_hz {
                    let delta_st = 12.0 * (center_hz / parent_freq_hz).log2();
                    final_weight *= peak_bias_gaussian_weight(delta_st, config.proposal_sigma_st);
                    if peak_bias_same_band(parent_freq_hz, center_hz, config.same_band_window_cents)
                    {
                        final_weight *= config.same_band_discount.clamp(0.0, 1.0);
                    }
                    if peak_bias_parent_octave(
                        parent_freq_hz,
                        center_hz,
                        config.octave_window_cents,
                    ) {
                        final_weight *= config.octave_discount.clamp(0.0, 1.0);
                    }
                }
                final_weights.push(final_weight.max(0.0));
            }

            if final_weights
                .iter()
                .all(|weight| !weight.is_finite() || *weight <= 0.0)
            {
                final_weights.clone_from(&scene_weights);
            }

            let chosen_idx = if let Ok(dist) = WeightedIndex::new(&final_weights) {
                dist.sample(rng)
            } else {
                scene_weights
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            };
            let center_hz = landscape.space.centers_hz[candidate_bins[chosen_idx]].clamp(lo, hi);
            peak_bias_local_search_frequency(landscape, center_hz, lo, hi, config)
        };

        if let Some(min_c_level) = population.respawn_min_c_level
            && landscape.evaluate_pitch_level(chosen_freq) < min_c_level
        {
            return None;
        }

        let (parent_id, parent_gen) = match selected_parent {
            Some(parent) => (Some(parent.id), Some(parent.generation)),
            None => (None, None),
        };
        Some((chosen_freq, parent_id, parent_gen))
    }

    fn pick_respawn_candidate<R: Rng + ?Sized>(
        &self,
        population_id: u64,
        population: &RuntimePopulationState,
        alive_by_population: &BTreeMap<u64, Vec<ParentCandidate>>,
        landscape: &LandscapeFrame,
        rng: &mut R,
        member_idx: usize,
    ) -> Option<(f32, Option<u64>, Option<u32>)> {
        if let RespawnPolicy::PeakBiased { config } = population.respawn_policy {
            let pool = alive_by_population
                .get(&population_id)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let selected_parent = if pool.is_empty() {
                None
            } else {
                Some(pool[weighted_parent_select(pool, rng)])
            };
            return self.peak_biased_respawn_candidate(
                population,
                selected_parent,
                landscape,
                rng,
                member_idx,
                config,
            );
        }

        let candidate_count = RESPAWN_CANDIDATE_COUNT;

        // Step 1: Select parent ONCE before candidate generation
        let selected_parent: Option<ParentCandidate> = match population.respawn_policy {
            RespawnPolicy::None => return None,
            RespawnPolicy::Random => None,
            RespawnPolicy::Hereditary { .. } => {
                let pool = alive_by_population
                    .get(&population_id)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                if pool.is_empty() {
                    None
                } else {
                    Some(pool[weighted_parent_select(pool, rng)])
                }
            }
            RespawnPolicy::PeakBiased { .. } => unreachable!("handled above"),
        };

        // Step 2: Generate candidates (all share same parent lineage)
        let mut candidates = Vec::with_capacity(candidate_count);
        for idx in 0..candidate_count {
            let freq = match (idx, population.respawn_settle_strategy.as_ref()) {
                (1.., Some(strategy)) => self
                    .resolve_strategy_frequency(
                        strategy,
                        landscape,
                        rng,
                        &[],
                        member_idx + idx,
                        candidate_count,
                    )
                    .max(MIN_FREQ_HZ),
                _ => match population.respawn_policy {
                    RespawnPolicy::None => return None,
                    RespawnPolicy::Random => {
                        self.random_respawn_frequency(population, landscape, rng, member_idx + idx)
                    }
                    RespawnPolicy::Hereditary { sigma_oct } => {
                        if let Some(ref parent) = selected_parent {
                            let parent_log2 = parent.freq_hz.max(MIN_FREQ_HZ).log2();
                            let noise = Self::normal_sample(rng) * sigma_oct.max(0.0);
                            let child_log2 = parent_log2 + noise;
                            let (min_hz, max_hz) = population
                                .strategy
                                .as_ref()
                                .map(SpawnStrategy::freq_range_hz)
                                .unwrap_or_else(|| landscape.freq_bounds());
                            let lo = min_hz.clamp(MIN_FREQ_HZ, MAX_FREQ_HZ);
                            let hi = max_hz.clamp(lo, MAX_FREQ_HZ);
                            2.0f32.powf(child_log2).clamp(lo, hi)
                        } else {
                            self.random_respawn_frequency(
                                population,
                                landscape,
                                rng,
                                member_idx + idx,
                            )
                        }
                    }
                    RespawnPolicy::PeakBiased { .. } => unreachable!("handled above"),
                },
            };
            candidates.push(freq);
        }

        let chosen_freq = match population.respawn_policy {
            RespawnPolicy::Random => choose_candidate_by_scene_score(landscape, &candidates, rng)?,
            _ => *candidates.iter().max_by(|a, b| {
                landscape
                    .evaluate_pitch_level(**a)
                    .partial_cmp(&landscape.evaluate_pitch_level(**b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?,
        };

        if let Some(min_c_level) = population.respawn_min_c_level
            && landscape.evaluate_pitch_level(chosen_freq) < min_c_level
        {
            return None;
        }

        // All candidates share same parent lineage
        let (parent_id, parent_gen) = match selected_parent {
            Some(p) => (Some(p.id), Some(p.generation)),
            None => (None, None),
        };
        Some((chosen_freq, parent_id, parent_gen))
    }

    pub(super) fn respawn_on_new_deaths(
        &mut self,
        scenario_finished: bool,
        landscape: &LandscapeFrame,
    ) {
        if scenario_finished || self.abort_requested {
            return;
        }

        let mut statuses = Vec::with_capacity(self.voices.len());
        let mut alive_by_population: BTreeMap<u64, Vec<ParentCandidate>> = BTreeMap::new();
        for voice in &self.voices {
            let alive = voice.is_alive();
            let population_id = voice.metadata.population_id;
            let id = voice.id();
            statuses.push((id, population_id, alive));
            if alive {
                let energy = match &voice.articulation.core {
                    AnyArticulationCore::Entrain(core) => core.energy.max(0.0),
                    _ => 0.0,
                };
                alive_by_population
                    .entry(population_id)
                    .or_default()
                    .push(ParentCandidate {
                        id,
                        freq_hz: voice.body.base_freq_hz().clamp(MIN_FREQ_HZ, MAX_FREQ_HZ),
                        energy,
                        generation: voice.metadata.generation,
                    });
            }
        }

        let mut dead_candidates = Vec::new();
        for (id, population_id, alive) in statuses {
            if alive {
                self.death_observed.remove(&id);
                continue;
            }
            if self.death_observed.insert(id) {
                dead_candidates.push((id, population_id));
            }
        }

        let mut projected_alive: BTreeMap<u64, usize> = alive_by_population
            .iter()
            .map(|(population_id, members)| (*population_id, members.len()))
            .collect();

        for (_dead_id, population_id) in dead_candidates {
            let Some(population) = self.populations.get(&population_id).cloned() else {
                continue;
            };
            if population.released {
                continue;
            }
            let alive_count = projected_alive.get(&population_id).copied().unwrap_or(0);
            if alive_count >= population.respawn_capacity {
                continue;
            }

            let member_idx = population.next_member_idx;
            let spawn_seq = self.spawn_counter;
            self.spawn_counter = self.spawn_counter.wrapping_add(1);
            let seed = self.spawn_seed(population_id, 1, spawn_seq);
            let mut rng = SmallRng::seed_from_u64(seed);

            let Some((freq_hz, parent_id, parent_generation)) = self.pick_respawn_candidate(
                population_id,
                &population,
                &alive_by_population,
                landscape,
                &mut rng,
                member_idx,
            ) else {
                continue;
            };

            let id = self.allocate_runtime_id();
            self.spawn_one(
                SpawnParams {
                    id,
                    population_id,
                    member_idx,
                    resolved_freq_hz: freq_hz,
                    parent_id,
                    parent_generation,
                    reason: SpawnReason::Respawn,
                },
                &population.template,
                landscape,
            );

            if let Some(state) = self.populations.get_mut(&population_id) {
                state.next_member_idx = state.next_member_idx.saturating_add(1);
            }
            *projected_alive.entry(population_id).or_default() += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{
        decay_spawn_spec_with_freq, force_dead, peak_bias_landscape, runtime_landscape,
        spawn_spec_with_freq, step_population, sustain_spawn_spec_with_freq, test_pop,
    };
    use super::*;
    use crate::core::log2space::Log2Space;
    use crate::scenario::Action;
    use crate::scenario::control::{ControlUpdate, PitchMode};

    #[test]
    fn respawn_none_keeps_current_behavior() {
        let mut pop = test_pop();
        pop.set_seed(7);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                population_id: 7,
                ids: vec![1],
                spec: decay_spawn_spec_with_freq(220.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );

        for frame in 0..300 {
            step_population(&mut pop, frame, 0.01, &landscape);
            if pop.voices.is_empty() {
                break;
            }
        }

        assert!(pop.voices.is_empty());
    }

    #[test]
    fn respawn_random_maintains_population() {
        let mut pop = test_pop();
        pop.set_seed(11);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                population_id: 8,
                ids: vec![10],
                spec: decay_spawn_spec_with_freq(220.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 8,
                policy: RespawnPolicy::Random,
                settle_strategy: None,
                capacity: 1,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );

        let mut saw_respawned = false;
        for frame in 0..300 {
            step_population(&mut pop, frame, 0.01, &landscape);
            if pop.voices.iter().any(|a| a.id() != 10) {
                saw_respawned = true;
                break;
            }
        }

        assert!(saw_respawned, "expected at least one respawned member");
        assert!(
            !pop.voices.is_empty(),
            "population should not collapse with random respawn"
        );
    }

    #[test]
    fn respawn_capacity_limits_living_members() {
        let mut pop = test_pop();
        pop.set_seed(12);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                population_id: 82,
                ids: vec![820, 821],
                spec: sustain_spawn_spec_with_freq(220.0),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 82,
                policy: RespawnPolicy::Random,
                settle_strategy: None,
                capacity: 1,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );

        force_dead(&mut pop, 820);
        pop.cleanup_dead(0, 0.01, false, &landscape);

        assert_eq!(pop.voices.len(), 1);
        assert_eq!(pop.voices[0].id(), 821);
    }

    #[test]
    fn background_turnover_replaces_member_via_respawn() {
        let mut pop = test_pop();
        pop.set_seed(61);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                population_id: 81,
                ids: vec![810],
                spec: sustain_spawn_spec_with_freq(220.0),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 81,
                policy: RespawnPolicy::Random,
                settle_strategy: None,
                capacity: 1,
                min_c_level: None,
                background_death_rate_per_sec: 10_000.0,
            },
            &landscape,
            None,
        );

        step_population(&mut pop, 0, 0.01, &landscape);

        assert_eq!(
            pop.voices.len(),
            1,
            "respawn should preserve population size"
        );
        assert_ne!(
            pop.voices[0].id(),
            810,
            "background turnover should replace the member"
        );
    }

    #[test]
    fn respawn_hereditary_maintains_population() {
        let mut pop = test_pop();
        pop.set_seed(13);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                population_id: 9,
                ids: vec![20],
                spec: decay_spawn_spec_with_freq(330.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 9,
                policy: RespawnPolicy::Hereditary { sigma_oct: 0.01 },
                settle_strategy: None,
                capacity: 1,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );

        let mut saw_respawned = false;
        for frame in 0..300 {
            step_population(&mut pop, frame, 0.01, &landscape);
            if pop.voices.iter().any(|a| a.id() != 20) {
                saw_respawned = true;
                break;
            }
        }

        assert!(saw_respawned, "expected at least one respawned member");
        assert!(
            !pop.voices.is_empty(),
            "population should not collapse with hereditary respawn"
        );
    }

    #[test]
    fn hereditary_respawn_without_strategy_uses_parent_pitch_regression() {
        let mut pop = test_pop();
        pop.set_seed(31);
        let landscape = runtime_landscape();

        let mut spec = decay_spawn_spec_with_freq(220.0, 0.02);
        spec.control.pitch.mode = PitchMode::Lock;
        pop.apply_action(
            Action::Spawn {
                population_id: 90,
                ids: vec![900, 901],
                spec,
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 90,
                policy: RespawnPolicy::Hereditary { sigma_oct: 0.002 },
                settle_strategy: None,
                capacity: 2,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );

        let parent_target_hz: f32 = 440.0;
        if let Some(parent) = pop.voices.iter_mut().find(|v| v.id() == 901) {
            parent.force_set_pitch_log2(parent_target_hz.log2());
        }
        force_dead(&mut pop, 900);
        pop.cleanup_dead(0, 0.01, false, &landscape);

        let child = pop
            .voices
            .iter()
            .find(|v| v.id() != 901)
            .expect("child exists");
        let child_log2 = child.body.base_freq_hz().log2();
        let parent_log2 = parent_target_hz.log2();
        let spec_log2 = 220.0f32.log2();
        let to_parent = (child_log2 - parent_log2).abs();
        let to_spec = (child_log2 - spec_log2).abs();
        assert!(to_parent < 0.05, "child should be close to live parent");
        assert!(
            to_parent < to_spec,
            "regression: child should follow parent, not template frequency"
        );
    }

    #[test]
    fn release_reaches_respawned_members() {
        let mut pop = test_pop();
        pop.set_seed(17);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                population_id: 10,
                ids: vec![30],
                spec: decay_spawn_spec_with_freq(220.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 10,
                policy: RespawnPolicy::Random,
                settle_strategy: None,
                capacity: 1,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );

        let mut respawned_id = None;
        for frame in 0..300 {
            step_population(&mut pop, frame, 0.01, &landscape);
            if let Some(id) = pop
                .voices
                .iter()
                .find(|v| v.metadata.population_id == 10 && v.id() != 30)
                .map(|v| v.id())
            {
                respawned_id = Some(id);
                break;
            }
        }
        let respawned_id = respawned_id.expect("respawned member should exist");

        pop.apply_action(
            Action::ReleasePopulation {
                population_id: 10,
                fade_sec: 0.05,
            },
            &landscape,
            None,
        );

        let respawned = pop
            .voices
            .iter()
            .find(|v| v.id() == respawned_id)
            .expect("respawned member");
        assert!(respawned.remove_pending);
    }

    #[test]
    fn live_update_reaches_respawned_members() {
        let mut pop = test_pop();
        pop.set_seed(23);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                population_id: 11,
                ids: vec![40],
                spec: decay_spawn_spec_with_freq(220.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 11,
                policy: RespawnPolicy::Random,
                settle_strategy: None,
                capacity: 1,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );

        let mut respawned_id = None;
        for frame in 0..300 {
            step_population(&mut pop, frame, 0.01, &landscape);
            if let Some(id) = pop
                .voices
                .iter()
                .find(|v| v.metadata.population_id == 11 && v.id() != 40)
                .map(|v| v.id())
            {
                respawned_id = Some(id);
                break;
            }
        }
        let respawned_id = respawned_id.expect("respawned member should exist");

        pop.apply_action(
            Action::UpdatePopulation {
                population_id: 11,
                patch: ControlUpdate {
                    amp: Some(0.17),
                    ..ControlUpdate::default()
                },
            },
            &landscape,
            None,
        );

        let respawned = pop
            .voices
            .iter()
            .find(|v| v.id() == respawned_id)
            .expect("respawned member");
        assert!((respawned.effective_control.body.amp - 0.17).abs() <= 1e-6);
    }

    #[test]
    fn live_landscape_weight_update_is_inherited_by_respawn() {
        let mut pop = test_pop();
        pop.set_seed(41);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                population_id: 91,
                ids: vec![910, 911],
                spec: decay_spawn_spec_with_freq(220.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 91,
                policy: RespawnPolicy::Random,
                settle_strategy: None,
                capacity: 2,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::UpdatePopulation {
                population_id: 91,
                patch: ControlUpdate {
                    landscape_weight: Some(0.73),
                    ..ControlUpdate::default()
                },
            },
            &landscape,
            None,
        );

        for member in pop.voices.iter().filter(|v| v.metadata.population_id == 91) {
            assert!((member.effective_control.pitch.landscape_weight - 0.73).abs() <= 1e-6);
        }

        force_dead(&mut pop, 910);
        pop.cleanup_dead(0, 0.01, false, &landscape);

        let child = pop
            .voices
            .iter()
            .find(|v| v.id() != 911)
            .expect("child exists");
        assert!((child.effective_control.pitch.landscape_weight - 0.73).abs() <= 1e-6);
    }

    #[test]
    fn release_disables_future_respawns() {
        let mut pop = test_pop();
        pop.set_seed(47);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                population_id: 92,
                ids: vec![920],
                spec: decay_spawn_spec_with_freq(220.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 92,
                policy: RespawnPolicy::Random,
                settle_strategy: None,
                capacity: 1,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::ReleasePopulation {
                population_id: 92,
                fade_sec: 0.01,
            },
            &landscape,
            None,
        );

        let mut saw_new_id = false;
        for frame in 0..400 {
            step_population(&mut pop, frame, 0.01, &landscape);
            if pop.voices.iter().any(|v| v.id() != 920) {
                saw_new_id = true;
                break;
            }
            if pop.voices.is_empty() {
                break;
            }
        }

        assert!(!saw_new_id, "release must disable future respawns");
        assert!(
            pop.voices.is_empty(),
            "released population should drain without repopulation"
        );
    }

    #[test]
    fn hereditary_respawn_child_stays_near_parent() {
        let mut pop = test_pop();
        pop.set_seed(29);
        let landscape = runtime_landscape();

        let mut spec = spawn_spec_with_freq(220.0);
        spec.control.pitch.mode = PitchMode::Lock;
        pop.apply_action(
            Action::Spawn {
                population_id: 12,
                ids: vec![100, 101],
                spec,
                strategy: Some(SpawnStrategy::Linear {
                    start_freq: 220.0,
                    end_freq: 330.0,
                }),
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 12,
                policy: RespawnPolicy::Hereditary { sigma_oct: 0.005 },
                settle_strategy: None,
                capacity: 2,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );

        let parent_freq = pop
            .voices
            .iter()
            .find(|v| v.id() == 101)
            .map(|v| v.body.base_freq_hz())
            .expect("parent exists");

        if let Some(dying) = pop.voices.iter_mut().find(|v| v.id() == 100) {
            dying.release_gain = 0.0;
            dying.release_pending = true;
        }
        pop.cleanup_dead(0, 0.01, false, &landscape);

        let child = pop
            .voices
            .iter()
            .find(|v| v.id() != 101)
            .expect("child exists");
        let delta_oct = (child.body.base_freq_hz().log2() - parent_freq.log2()).abs();
        assert!(
            delta_oct < 0.05,
            "child should stay near parent in log2 space"
        );
    }

    #[test]
    fn peak_biased_respawn_prefers_parent_nearby_peak_family() {
        let mut pop = test_pop();
        pop.set_seed(59);
        let landscape = peak_bias_landscape();

        let mut spec = spawn_spec_with_freq(220.0);
        spec.control.pitch.mode = PitchMode::Lock;
        pop.apply_action(
            Action::Spawn {
                population_id: 13,
                ids: vec![130, 131],
                spec,
                strategy: Some(SpawnStrategy::Linear {
                    start_freq: 250.0,
                    end_freq: 700.0,
                }),
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 13,
                policy: RespawnPolicy::PeakBiased {
                    config: RespawnPeakBiasConfig::default(),
                },
                settle_strategy: None,
                capacity: 8,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );

        if let Some(parent) = pop.voices.iter_mut().find(|v| v.id() == 131) {
            parent.force_set_pitch_log2(300.0f32.log2());
        }
        force_dead(&mut pop, 130);
        pop.cleanup_dead(0, 0.01, false, &landscape);

        let child = pop
            .voices
            .iter()
            .find(|v| v.id() != 131)
            .expect("child exists");
        let child_freq = child.body.base_freq_hz();
        let near_parent_peak = (child_freq.log2() - 330.0f32.log2()).abs();
        let far_peak = (child_freq.log2() - 660.0f32.log2()).abs();
        assert!(
            near_parent_peak < far_peak,
            "child should stay closer to the parent-aligned peak family"
        );
    }

    #[test]
    fn random_respawn_selection_uses_weighted_scene_scores() {
        let mut landscape = LandscapeFrame::new(Log2Space::new(220.0, 880.0, 96));
        let candidate_bins = [12usize, 36usize, 60usize];
        let candidate_freqs = candidate_bins.map(|idx| landscape.space.centers_hz[idx]);
        let candidate_scores = [0.0f32, 0.5, 2.0];

        for (bin_idx, score) in candidate_bins.into_iter().zip(candidate_scores) {
            landscape.consonance_field_score[bin_idx] = score;
            landscape.consonance_field_level[bin_idx] = score.clamp(0.0, 1.0);
            landscape.consonance_field_score_eff[bin_idx] = score;
            landscape.consonance_field_level_eff[bin_idx] = score.clamp(0.0, 1.0);
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(20260331);
        let mut counts = [0usize; 3];
        for _ in 0..4096 {
            let chosen = choose_candidate_by_scene_score(&landscape, &candidate_freqs, &mut rng)
                .expect("candidate should be selected");
            let idx = candidate_freqs
                .iter()
                .position(|freq_hz| (*freq_hz - chosen).abs() <= 1e-6)
                .expect("chosen candidate should come from the candidate list");
            counts[idx] += 1;
        }

        assert_eq!(
            counts[0], 0,
            "zero-score candidates should not be sampled when positive weights exist"
        );
        assert!(
            counts[1] > 0,
            "lower-score positive candidates should remain reachable"
        );
        assert!(
            counts[2] > counts[1],
            "higher scene scores should win more often than lower ones"
        );
    }
}
