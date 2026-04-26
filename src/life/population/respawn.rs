use super::ParentCandidate;
use crate::core::landscape::LandscapeFrame;
use crate::core::peak_extraction::{PeakExtractConfig, extract_peaks_density};
use crate::life::control::{MAX_FREQ_HZ, MIN_FREQ_HZ};
use crate::life::scenario::RespawnPeakBiasConfig;
use rand::{Rng, distr::Distribution, distr::weighted::WeightedIndex};

pub(super) fn semitone_distance_from_anchor(freq_hz: f32, anchor_hz: f32) -> f32 {
    if !freq_hz.is_finite() || freq_hz <= 0.0 || !anchor_hz.is_finite() || anchor_hz <= 0.0 {
        return 0.0;
    }
    12.0 * (freq_hz / anchor_hz).log2()
}

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
    (-0.5 * (delta_st / sigma_st).powi(2)).exp()
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
        let weight = landscape.consonance_field_score[idx].max(0.0);
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

pub(super) fn is_rejected_target(
    freq_hz: f32,
    anchor_hz: f32,
    targets_st: &[f32],
    exclusion_st: f32,
) -> bool {
    let semitone_abs = semitone_distance_from_anchor(freq_hz, anchor_hz).abs();
    targets_st.iter().any(|target| {
        target.is_finite()
            && exclusion_st.is_finite()
            && (semitone_abs - target.abs()).abs() <= exclusion_st.max(0.0)
    })
}
