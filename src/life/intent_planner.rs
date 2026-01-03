use crate::core::log2space::{Log2Space, sample_scan_linear_log2};
use crate::core::timebase::Tick;
use crate::life::intent::Intent;

#[derive(Clone, Copy, Debug)]
pub struct GestureChoiceTf {
    pub onset: Tick,
    pub freq_hz: f32,
    pub score: f32,
}

pub fn choose_onset_by_density(candidates: &[Tick], intents: &[Intent], eps: Tick) -> Option<Tick> {
    if candidates.is_empty() {
        return None;
    }
    let mut best_tick = candidates[0];
    let mut best_score = 0usize;
    for &cand in candidates {
        let min = cand.saturating_sub(eps);
        let max = cand.saturating_add(eps);
        let mut score = 0usize;
        for intent in intents {
            if intent.onset >= min && intent.onset <= max {
                score += 1;
            }
        }
        if score > best_score || (score == best_score && cand < best_tick) {
            best_score = score;
            best_tick = cand;
        }
    }
    Some(best_tick)
}

pub fn choose_freq_by_consonance(
    candidates_hz: &[f32],
    neighbor_freqs_hz: &[f32],
    base_freq_hz: f32,
) -> Option<f32> {
    // Placeholder: consonance proxy using simple ratio fit to nearby intents.
    // This is not R/H consonance (no roughness term, no spectrum interactions).
    let mut valid_candidates: Vec<f32> = candidates_hz
        .iter()
        .copied()
        .filter(|f| f.is_finite() && *f > 0.0)
        .collect();
    if valid_candidates.is_empty() {
        return None;
    }
    if neighbor_freqs_hz.is_empty() {
        let base = if base_freq_hz.is_finite() && base_freq_hz > 0.0 {
            base_freq_hz
        } else {
            valid_candidates[0]
        };
        valid_candidates.sort_by(|a, b| {
            let da = (a - base).abs();
            let db = (b - base).abs();
            da.partial_cmp(&db)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        });
        return valid_candidates.first().copied();
    }

    let ratios = crate::core::harmonic_ratios::HARMONIC_RATIOS;
    let neighbors: Vec<f32> = neighbor_freqs_hz
        .iter()
        .copied()
        .filter(|f| f.is_finite() && *f > 0.0)
        .collect();
    if neighbors.is_empty() {
        return valid_candidates.first().copied();
    }

    let base = base_freq_hz;
    let mut best = valid_candidates[0];
    let mut best_score = f32::NEG_INFINITY;
    for cand in valid_candidates {
        let mut score = 0.0f32;
        for nf in &neighbors {
            let ratio = if cand >= *nf { cand / *nf } else { *nf / cand };
            let mut best_cents = f32::INFINITY;
            for &target in ratios {
                let target_ratio = crate::core::harmonic_ratios::ratio_to_f32(target);
                let cents = 1200.0 * (ratio / target_ratio).log2();
                let abs_cents = cents.abs();
                if abs_cents < best_cents {
                    best_cents = abs_cents;
                }
            }
            score += -best_cents;
        }

        let tie = (score - best_score).abs() <= 1e-3;
        if score > best_score {
            best_score = score;
            best = cand;
        } else if tie {
            let base = if base.is_finite() && base > 0.0 {
                base
            } else {
                best
            };
            let cand_dist = (cand - base).abs();
            let best_dist = (best - base).abs();
            if cand_dist < best_dist || (cand_dist == best_dist && cand < best) {
                best = cand;
            }
        }
    }
    Some(best)
}

pub fn choose_best_freq_by_pred_c(
    space: &Log2Space,
    pred_c_statepm1_scan: &[f32],
    freq_candidates_hz: &[f32],
    base_freq_hz: f32,
) -> Option<(f32, f32)> {
    if pred_c_statepm1_scan.is_empty() || freq_candidates_hz.is_empty() {
        return None;
    }
    if pred_c_statepm1_scan.len() != space.n_bins() {
        return None;
    }

    let mut best_freq = 0.0f32;
    let mut best_score = f32::NEG_INFINITY;
    let mut best_abs_log2 = f32::INFINITY;

    for &cand in freq_candidates_hz {
        if !cand.is_finite() || cand <= 0.0 {
            continue;
        }
        let score = sample_scan_linear_log2(space, pred_c_statepm1_scan, cand);
        if !score.is_finite() || score == f32::NEG_INFINITY {
            continue;
        }
        let base = if base_freq_hz.is_finite() && base_freq_hz > 0.0 {
            base_freq_hz
        } else {
            cand
        };
        let abs_log2 = (cand / base).log2().abs();
        let score_better = score > best_score;
        let score_tie = (score - best_score).abs() <= 1e-6;
        let dist_better = abs_log2 < best_abs_log2 - 1e-6;
        let dist_tie = (abs_log2 - best_abs_log2).abs() <= 1e-6;
        if score_better
            || (score_tie
                && (dist_better || (dist_tie && (cand < best_freq || best_score.is_infinite()))))
        {
            best_score = score;
            best_freq = cand;
            best_abs_log2 = abs_log2;
        }
    }

    if best_score.is_finite() {
        Some((best_freq, best_score))
    } else {
        None
    }
}

pub fn choose_best_gesture_tf_by_pred_c(
    space: &Log2Space,
    onset_candidates: &[Tick],
    base_freq_hz: f32,
    mut make_freq_candidates: impl FnMut(Tick) -> Vec<f32>,
    mut pred_c_scan_at: impl FnMut(Tick) -> Option<std::sync::Arc<[f32]>>,
) -> Option<GestureChoiceTf> {
    if onset_candidates.is_empty() {
        return None;
    }

    let mut best_choice: Option<GestureChoiceTf> = None;
    let mut best_abs_log2 = f32::INFINITY;

    for &onset in onset_candidates {
        let pred_c_scan = match pred_c_scan_at(onset) {
            Some(scan) if !scan.is_empty() => scan,
            _ => continue,
        };
        let mut candidates = make_freq_candidates(onset);
        if candidates.is_empty() && base_freq_hz.is_finite() && base_freq_hz > 0.0 {
            candidates.push(base_freq_hz);
        }
        let (freq_hz, score) = match choose_best_freq_by_pred_c(
            space,
            pred_c_scan.as_ref(),
            &candidates,
            base_freq_hz,
        ) {
            Some(choice) => choice,
            None => continue,
        };
        let base = if base_freq_hz.is_finite() && base_freq_hz > 0.0 {
            base_freq_hz
        } else {
            freq_hz
        };
        let abs_log2 = (freq_hz / base).log2().abs();
        let choice = GestureChoiceTf {
            onset,
            freq_hz,
            score,
        };

        let is_better = match best_choice {
            None => true,
            Some(best) => {
                let score_better = choice.score > best.score + 1e-6;
                let score_tie = (choice.score - best.score).abs() <= 1e-6;
                let onset_better = choice.onset < best.onset;
                let onset_tie = choice.onset == best.onset;
                let dist_better = abs_log2 < best_abs_log2 - 1e-6;
                let dist_tie = (abs_log2 - best_abs_log2).abs() <= 1e-6;
                let freq_better = choice.freq_hz < best.freq_hz;
                score_better
                    || (score_tie
                        && (onset_better
                            || (onset_tie && (dist_better || (dist_tie && freq_better)))))
            }
        };
        if is_better {
            best_abs_log2 = abs_log2;
            best_choice = Some(choice);
        }
    }

    best_choice
}
