use crate::core::timebase::Tick;
use crate::life::intent::Intent;

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
    // Roughness/energy interactions are not modeled here yet.
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

    let ratios = [
        1.0,
        2.0,
        1.5,
        4.0 / 3.0,
        5.0 / 4.0,
        6.0 / 5.0,
        5.0 / 3.0,
        8.0 / 5.0,
    ];
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
            for &target in &ratios {
                let cents = 1200.0 * (ratio / target).log2();
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
