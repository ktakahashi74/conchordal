use crate::core::harmonic_ratios::{HARMONIC_RATIOS, ratio_to_f32};
use crate::core::landscape::Landscape;
use rand::{Rng, RngExt};

pub(super) const DEFAULT_LOCAL_WINDOW_CENTS: f32 = 240.0;
pub(super) const DEFAULT_LOCAL_TOP_K: usize = 10;
pub(super) const DEFAULT_RANDOM_CANDIDATES: usize = 3;
pub(super) const DEFAULT_RANDOM_SIGMA_CENTS: f32 = 30.0;

pub(super) fn cents_to_log2(cents: f32) -> f32 {
    cents / 1200.0
}

pub(super) fn window_bins_from_cents(landscape: &Landscape, cents: f32) -> usize {
    let step = landscape.space.step().max(1e-6);
    let span_log2 = cents_to_log2(cents.max(0.0));
    (span_log2 / step).ceil() as usize
}

#[allow(clippy::too_many_arguments)]
pub(super) fn top_local_peaks(
    landscape: &Landscape,
    current_target_log2: f32,
    window_bins: usize,
    top_k: usize,
    scorer: &mut impl FnMut(f32) -> f32,
    scored: &mut Vec<(f32, f32)>,
    out: &mut Vec<f32>,
) {
    let (fmin, fmax) = landscape.space.voice_pitch_bounds_log2();
    let n_bins = landscape.space.n_bins();
    if n_bins == 0 {
        return;
    }
    let clamped_target = current_target_log2.clamp(fmin, fmax);
    let idx0 = landscape.space.index_of_log2(clamped_target).unwrap_or(0);
    let start = idx0.saturating_sub(window_bins);
    let end = idx0.saturating_add(window_bins).min(n_bins - 1);
    scored.clear();
    for idx in start..=end {
        let pitch = landscape.space.centers_log2[idx];
        if pitch < fmin || pitch > fmax {
            continue;
        }
        let score = scorer(pitch);
        if score.is_finite() {
            scored.push((score, pitch));
        }
    }
    // Equal scores retain the ascending bin order of the former stable sort.
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.total_cmp(&b.1)));
    let limit = top_k.max(1).min(scored.len());
    out.extend(scored.iter().take(limit).map(|&(_, pitch)| pitch));
}

pub(super) fn top_global_peaks(
    landscape: &Landscape,
    top_k: usize,
    min_sep_log2: f32,
    scorer: &mut impl FnMut(f32) -> f32,
    scored: &mut Vec<(f32, f32)>,
    out: &mut Vec<f32>,
) {
    let n_bins = landscape.space.n_bins();
    if n_bins == 0 || top_k == 0 {
        return;
    }
    let (fmin, fmax) = landscape.space.voice_pitch_bounds_log2();
    scored.clear();
    for &pitch in &landscape.space.centers_log2 {
        if pitch < fmin || pitch > fmax {
            continue;
        }
        let score = scorer(pitch);
        if score.is_finite() {
            scored.push((score, pitch));
        }
    }
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.total_cmp(&b.1)));
    let start = out.len();
    for &(_, pitch) in scored.iter() {
        if min_sep_log2 > 0.0
            && out[start..]
                .iter()
                .any(|p| (pitch - *p).abs() < min_sep_log2)
        {
            continue;
        }
        out.push(pitch);
        if out.len() - start >= top_k {
            break;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn push_runtime_ratio_candidates(
    candidates: &mut Vec<f32>,
    generated: &mut Vec<f32>,
    current_target_log2: f32,
    current_pitch_log2: f32,
    neighbor_pitch_log2: &[f32],
    max_count: usize,
    min_log2: f32,
    max_log2: f32,
) {
    if max_count == 0 {
        return;
    }
    let mut anchors = [current_target_log2, current_pitch_log2, f32::NAN];
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for &neighbor in neighbor_pitch_log2 {
        if neighbor.is_finite() {
            sum += neighbor;
            count += 1;
        }
    }
    if count > 0 {
        anchors[2] = sum / count as f32;
    }

    generated.clear();
    for anchor in anchors {
        if !anchor.is_finite() {
            continue;
        }
        for &ratio in HARMONIC_RATIOS {
            let ratio_f = ratio_to_f32(ratio);
            if !ratio_f.is_finite() || ratio_f <= 0.0 {
                continue;
            }
            let ratio_log2 = ratio_f.log2();
            generated.push((anchor + ratio_log2).clamp(min_log2, max_log2));
            generated.push((anchor - ratio_log2).clamp(min_log2, max_log2));
        }
    }
    generated.retain(|x| x.is_finite());
    dedupe_log2_by_cents(generated, 2.0);
    generated.sort_unstable_by(|a, b| {
        (a - current_target_log2)
            .abs()
            .total_cmp(&(b - current_target_log2).abs())
            .then(a.total_cmp(b))
    });
    candidates.extend(generated.iter().copied().take(max_count));
}

pub(super) fn seed_step_candidates(current_target_log2: f32, neighbor_step_log2: f32) -> [f32; 3] {
    [
        current_target_log2,
        current_target_log2 + neighbor_step_log2,
        current_target_log2 - neighbor_step_log2,
    ]
}

pub(super) fn score_candidates_with(
    candidates: Vec<f32>,
    fmin: f32,
    fmax: f32,
    mut score_fn: impl FnMut(f32) -> f32,
) -> Vec<(f32, f32)> {
    let mut scored = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let pitch = candidate.clamp(fmin, fmax);
        let score = score_fn(pitch);
        if score.is_finite() {
            scored.push((pitch, score));
        }
    }
    scored
}

pub(super) fn push_gaussian_candidates<R: Rng + Clone>(
    candidates: &mut Vec<f32>,
    center_log2: f32,
    min_log2: f32,
    max_log2: f32,
    sigma_cents: f32,
    count: usize,
    rng: &mut R,
) {
    if count == 0 || sigma_cents <= 0.0 {
        return;
    }
    let sigma_log2 = cents_to_log2(sigma_cents);
    for _ in 0..count {
        let z = sample_standard_normal(rng);
        let candidate = (center_log2 + z * sigma_log2).clamp(min_log2, max_log2);
        candidates.push(candidate);
    }
}

fn sample_standard_normal<R: Rng + Clone>(rng: &mut R) -> f32 {
    let u1 = rng.random_range(f32::EPSILON..1.0);
    let u2 = rng.random_range(0.0..1.0);
    let mag = (-2.0 * u1.ln()).sqrt();
    let angle = core::f32::consts::TAU * u2;
    mag * angle.cos()
}

pub(super) fn sample_softmax_candidate<R: Rng + Clone>(
    scored: &[(f32, f32)],
    temperature: f32,
    rng: &mut R,
) -> f32 {
    sample_softmax_with_relations(scored, temperature, rng, None, None)
}

pub(super) fn sample_softmax_with_relations<R: Rng + Clone>(
    scored: &[(f32, f32)],
    temperature: f32,
    rng: &mut R,
    temporal: Option<&crate::temporal_cognition::harmonic_relations::consumer::Use>,
    current: Option<f32>,
) -> f32 {
    if scored.is_empty() {
        return 0.0;
    }
    let best = scored
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(pitch, _)| *pitch)
        .unwrap_or(scored[0].0);

    let max_score = scored
        .iter()
        .map(|(_, score)| *score)
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(0.0);
    let temp = temperature.max(1e-4);

    let mut weights = Vec::with_capacity(scored.len());
    let mut total = 0.0f32;
    for &(_, score) in scored {
        let weight = ((score - max_score) / temp).exp();
        let weight = if weight.is_finite() && weight > 0.0 {
            weight
        } else {
            0.0
        };
        total += weight;
        weights.push(weight);
    }

    if total <= 0.0 {
        return best;
    }

    let mut conditioned = false;
    let mut conditional_draw = None;
    if let Some(temporal) = temporal.filter(|u| u.active()) {
        let mut shadow = rng.clone();
        let reference = sample_softmax_candidate(scored, temperature, &mut shadow);
        let baseline = scored.iter().position(|(p, _)| *p == reference).unwrap();
        if temporal.continuation == 0. {
            let retain = current == Some(reference);
            let mut class_mass = 0.;
            let mut excluded_before = 0.;
            for (i, ((pitch, _), &mass)) in scored.iter().zip(&weights).enumerate() {
                if (current == Some(*pitch)) == retain {
                    class_mass += mass;
                } else if i < baseline {
                    excluded_before += mass;
                }
            }
            conditional_draw = Some((excluded_before, class_mass));
        }
        conditioned = temporal.softmax_scores(&mut weights, baseline, f64::from(temp), |i| {
            (
                Some(scored[i].0),
                temporal.source(),
                f64::from(scored[i].1),
                current == Some(scored[i].0),
            )
        });
    }

    let mut draw = rng.random_range(0.0..total);
    if conditioned && let Some((excluded_before, class_mass)) = conditional_draw {
        // Reuse the native quantile within its action class, without another draw.
        draw = ((draw - excluded_before).max(0.) / class_mass * total).min(total);
    }
    for ((pitch, _), weight) in scored.iter().zip(weights.iter()) {
        if *weight <= 0.0 {
            continue;
        }
        if draw <= *weight {
            return *pitch;
        }
        draw -= *weight;
    }

    if conditioned {
        // Float accumulation may leave a tail; stay inside the conditioned support.
        weights
            .iter()
            .rposition(|w| *w > 0.)
            .map_or(best, |i| scored[i].0)
    } else {
        best
    }
}

pub(super) fn dedupe_log2_by_cents(values_log2: &mut Vec<f32>, dedupe_cents: f32) {
    let min_log2 = cents_to_log2(dedupe_cents.max(0.0));
    values_log2.retain(|x| x.is_finite());
    values_log2.sort_unstable_by(|a, b| a.total_cmp(b));
    values_log2.dedup_by(|value, previous| (*value - *previous).abs() < min_log2);
}
