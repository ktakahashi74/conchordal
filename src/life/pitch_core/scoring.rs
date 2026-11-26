use super::candidates::cents_to_log2;
#[cfg(feature = "loo-diagnostic")]
use super::diagnostic;
use crate::core::float::unit_gaussian;
#[cfg(any(test, feature = "loo-diagnostic"))]
use crate::core::harmonicity_kernel::HarmonicityKernel;
use crate::core::landscape::Landscape;
use crate::core::log2space::Log2Space;
#[cfg(any(test, feature = "loo-diagnostic"))]
use crate::core::psycho_state::roughness_ratio_to_state01;
#[cfg(any(test, feature = "loo-diagnostic"))]
use crate::core::roughness_kernel::RoughnessKernel;
#[cfg(any(test, feature = "loo-diagnostic"))]
use crate::core::stream::potentials::PotentialScratch;
use crate::life::adaptation::AdaptationContext;
use std::sync::Arc;

// Historical comparators remain available only to tests and shadow diagnostics.
#[derive(Clone, Copy, Debug)]
pub(super) enum ReferenceLooMode {
    ApproxHarmonics,
    #[cfg(any(test, feature = "loo-diagnostic"))]
    ExactScan,
}

pub(super) const DEFAULT_MOVE_COST_COEFF: f32 = 0.5;
pub(super) const DEFAULT_MOVE_COST_EXP: u8 = 1;
#[cfg(any(test, feature = "loo-diagnostic"))]
const DEFAULT_APPROX_LOO_SIGMA_CENTS: f32 = 24.0;
pub(super) const DEFAULT_LOO_HARMONICS: u8 = 1;
const DEFAULT_CROWDING_PAIR_SPLIT_EPS_FRAC: f32 = 0.25;
#[cfg(any(test, feature = "loo-diagnostic"))]
pub(super) const EXACT_LOO_ROUGHNESS_ERB_STEP: f32 = 0.005;

/// Single fundamental-occupancy contribution of one neighbor at `candidate_log2`:
/// a Gaussian on octave (log2) distance with the pairwise split bias that keeps
/// two voices at the same f0 from drifting in lockstep. This is the one shared
/// definition of "how occupied is this fundamental", used by both the crowding
/// penalty and the adaptation occupancy feed (`PitchController`).
#[inline]
pub(crate) fn occupancy_contribution(
    candidate_log2: f32,
    neighbor_log2: f32,
    split_sign: f32,
    sigma_log2: f32,
) -> f32 {
    let sigma = sigma_log2.max(1e-9);
    let eps = sigma * DEFAULT_CROWDING_PAIR_SPLIT_EPS_FRAC;
    let split = split_sign.clamp(-1.0, 1.0);
    let d = candidate_log2 - (neighbor_log2 - split * eps);
    unit_gaussian(d, sigma)
}

/// Fill `out` with the fundamental-occupancy scan (the leave-self-out spatial
/// slice of `F`): deposit each visible neighbor's `occupancy_contribution` across
/// the Log2Space. The Gaussian on raw f0 only reaches a local window around the
/// neighbor, so each deposit touches O(window) bins.
pub(crate) fn fill_occupancy_scan(
    out: &mut Vec<f32>,
    space: &Log2Space,
    neighbors: &[f32],
    neighbor_salience: &[f32],
    sigma_cents: f32,
) {
    let n = space.n_bins();
    out.clear();
    out.resize(n, 0.0);
    if neighbors.is_empty() || n == 0 {
        return;
    }
    let sigma_log2 = (sigma_cents.max(1e-3)) / 1200.0;
    let margin = 4.0 * sigma_log2;
    let win = (margin * space.bins_per_oct as f32).ceil() as isize;
    let n_isize = n as isize;
    for (idx, &nb) in neighbors.iter().enumerate() {
        let Some(center) = space.index_of_log2(nb) else {
            continue;
        };
        let split_sign = neighbor_salience.get(idx).copied().unwrap_or(0.0);
        let center_bin = center as isize;
        let lo = (center_bin - win).max(0);
        let hi = (center_bin + win).min(n_isize - 1);
        if hi < lo {
            continue;
        }
        for i in lo..=hi {
            out[i as usize] +=
                occupancy_contribution(space.centers_log2[i as usize], nb, split_sign, sigma_log2);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn adjusted_pitch_score_impl(
    pitch_log2: f32,
    current_pitch_log2: f32,
    integration_window: f32,
    tessitura_center: f32,
    tessitura_gravity: f32,
    landscape_weight: f32,
    move_cost_coeff: f32,
    move_cost_exp: u8,
    landscape: &Landscape,
    perceptual: &AdaptationContext,
    leave_self_out: bool,
    reference_loo_mode: ReferenceLooMode,
    reference_harmonics: u8,
    exact_loo_scan: Option<&[f32]>,
    body_ratios: Option<&[f32]>,
    crowding_strength: f32,
    crowding_sigma_cents: f32,
    neighbor_pitch_log2: &[f32],
    neighbor_salience: &[f32],
) -> f32 {
    let (fmin, fmax) = landscape.acoustic.space.voice_pitch_bounds_log2();
    let clamped = pitch_log2.clamp(fmin, fmax);
    let score = sample_consonance_score_with_loo(
        clamped,
        current_pitch_log2,
        landscape,
        leave_self_out,
        reference_loo_mode,
        reference_harmonics,
        exact_loo_scan,
        body_ratios,
    );
    let distance_oct = (clamped - current_pitch_log2).abs();
    let dist_cost = if move_cost_exp == 2 {
        distance_oct * distance_oct
    } else {
        distance_oct
    };
    let penalty = dist_cost * integration_window * move_cost_coeff.max(0.0);
    let dist = clamped - tessitura_center;
    let gravity_penalty = dist * dist * tessitura_gravity;
    let weighted_score =
        landscape_weight.max(0.0) * landscape.pitch_objective_mode.apply_consonance_score(score);
    // Stage 2 of the occupancy redesign: crowding samples the same fundamental-
    // occupancy kernel as the adaptation feed (Log2 Gaussian of width
    // crowding_sigma_cents, self already excluded upstream).
    // See docs/design-notes/voice-movement-redesign.md.
    let crowding_penalty = if crowding_strength > 0.0 && !neighbor_pitch_log2.is_empty() {
        let sigma_log2 = cents_to_log2(crowding_sigma_cents.max(1e-3));
        let mut sum = 0.0f32;
        for (idx, &neighbor_log2) in neighbor_pitch_log2.iter().enumerate() {
            if !neighbor_log2.is_finite() {
                continue;
            }
            let split_sign = neighbor_salience.get(idx).copied().unwrap_or(0.0);
            sum += occupancy_contribution(clamped, neighbor_log2, split_sign, sigma_log2);
        }
        crowding_strength * sum
    } else {
        0.0
    };
    let base = weighted_score - penalty - gravity_penalty - crowding_penalty;
    let idx = landscape.acoustic.space.index_of_log2(clamped).unwrap_or(0);
    let result = base + perceptual.score_adjustment(idx);
    #[cfg(feature = "loo-diagnostic")]
    diagnostic::record_score(clamped, score, result);
    result
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sample_consonance_score_with_loo(
    pitch_log2: f32,
    current_pitch_log2: f32,
    landscape: &Landscape,
    leave_self_out: bool,
    reference_loo_mode: ReferenceLooMode,
    reference_harmonics: u8,
    exact_loo_scan: Option<&[f32]>,
    body_ratios: Option<&[f32]>,
) -> f32 {
    #[cfg(feature = "loo-diagnostic")]
    if diagnostic::variant() == "legacy" {
        return legacy_sample_consonance_score_with_loo(
            pitch_log2,
            current_pitch_log2,
            landscape,
            leave_self_out,
            reference_loo_mode,
            reference_harmonics,
            exact_loo_scan,
            body_ratios,
        );
    }
    #[cfg(any(test, feature = "loo-diagnostic"))]
    if leave_self_out
        && matches!(reference_loo_mode, ReferenceLooMode::ExactScan)
        && exact_loo_scan.is_none()
    {
        return legacy_sample_consonance_score_with_loo(
            pitch_log2,
            current_pitch_log2,
            landscape,
            leave_self_out,
            reference_loo_mode,
            reference_harmonics,
            exact_loo_scan,
            body_ratios,
        );
    }
    let _ = (
        current_pitch_log2,
        reference_loo_mode,
        reference_harmonics,
        body_ratios,
    );
    sample_ownership_score(
        landscape,
        pitch_log2,
        exact_loo_scan.filter(|_| leave_self_out),
    )
}

pub(super) fn sample_ownership_score(
    landscape: &Landscape,
    pitch_log2: f32,
    scan: Option<&[f32]>,
) -> f32 {
    if let Some(scan) = scan {
        landscape
            .acoustic
            .space
            .assert_scan_len_named(scan, "loo_score_scan");
        let raw = landscape.sample_linear_log2(scan, pitch_log2);
        let h = landscape
            .sample_linear_log2(&landscape.acoustic.perc_habituation_state_scan, pitch_log2);
        return crate::core::habituation::erode_score(raw, h, landscape.acoustic.consonance_theta);
    }
    // Without an ownership estimate, preserve the observed field.
    landscape.evaluate_pitch_score_log2(pitch_log2)
}

#[cfg(any(test, feature = "loo-diagnostic"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn legacy_sample_consonance_score_with_loo(
    pitch_log2: f32,
    current_pitch_log2: f32,
    landscape: &Landscape,
    leave_self_out: bool,
    reference_loo_mode: ReferenceLooMode,
    reference_harmonics: u8,
    exact_loo_scan: Option<&[f32]>,
    body_ratios: Option<&[f32]>,
) -> f32 {
    #[cfg(any(test, feature = "loo-diagnostic"))]
    if leave_self_out
        && matches!(reference_loo_mode, ReferenceLooMode::ExactScan)
        && let Some(scan) = exact_loo_scan
    {
        let raw = landscape.sample_linear_log2(scan, pitch_log2);
        let h = landscape
            .sample_linear_log2(&landscape.acoustic.perc_habituation_state_scan, pitch_log2);
        return crate::core::habituation::erode_score(raw, h, landscape.acoustic.consonance_theta);
    }
    let _ = (reference_loo_mode, exact_loo_scan);
    let (fmin, fmax) = landscape.freq_bounds_log2();
    let clamped = pitch_log2.clamp(fmin, fmax);
    let mut score = landscape.evaluate_pitch_score_log2(clamped);
    if leave_self_out {
        let current_clamped = current_pitch_log2.clamp(fmin, fmax);
        let sigma = cents_to_log2(DEFAULT_APPROX_LOO_SIGMA_CENTS).max(1e-6);
        if let Some(ratios) = body_ratios {
            // Perceive with your own body: subtract the voice's actual partials.
            // Position is ratio -> log2 offset from the fundamental; weight follows
            // the same per-index 1/n law as the generic series (index 0 -> 1/1).
            for (idx, &ratio) in ratios.iter().enumerate() {
                if !ratio.is_finite() || ratio <= 0.0 {
                    continue;
                }
                // Match the render-side ratio clamp so subtraction lands where
                // the partial actually sounds (see harmonic_ratio/partial_ratio).
                let ratio = ratio.max(0.1);
                let partial_log2 = current_clamped + ratio.log2();
                if partial_log2 < fmin || partial_log2 > fmax {
                    continue;
                }
                let self_score = landscape.evaluate_pitch_score_log2(partial_log2);
                if !self_score.is_finite() || self_score <= 0.0 {
                    continue;
                }
                let d = (clamped - partial_log2).abs();
                let weight = 1.0 / (idx as f32 + 1.0);
                score -= weight * self_score * (-d / sigma).exp();
            }
        } else {
            // Generic assumed harmonic series (bodies without ratios, e.g. Sine).
            let harmonics = reference_harmonics.max(1);
            for harmonic in 1..=harmonics {
                let harmonic_f = harmonic as f32;
                let harmonic_log2 = current_clamped + harmonic_f.log2();
                if harmonic_log2 < fmin || harmonic_log2 > fmax {
                    continue;
                }
                let self_score = landscape.evaluate_pitch_score_log2(harmonic_log2);
                if !self_score.is_finite() || self_score <= 0.0 {
                    continue;
                }
                let d = (clamped - harmonic_log2).abs();
                let harmonic_weight = 1.0 / harmonic_f.max(1.0);
                score -= harmonic_weight * self_score * (-d / sigma).exp();
            }
        }
    }
    score
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LooInput {
    pub(crate) relations: Option<crate::temporal_cognition::harmonic_relations::consumer::Use>,
    pub(crate) relation_source: crate::temporal_cognition::harmonic_relations::Source,
    pub(crate) support_template: Option<Arc<[(isize, f32)]>>,
    pub(crate) support: Option<Arc<crate::core::harmonic_support::Snapshot>>,
    pub(crate) support_single_density: f32,
    pub(crate) support_warned: bool,
    pub(crate) tonal_context: Option<Arc<crate::core::tonal_context::Snapshot>>,
    pub(crate) exact_score: Option<Arc<[f32]>>,
    /// Own-ID attack history without own PCM, for `respond_to_onsets`.
    pub(crate) onset_habituation_state_scan: Option<Arc<[f32]>>,
    pub(crate) onset_habituation_end: u64,
    pub(crate) onset_habituation_known: bool,
    pub(in crate::life::voice) rearticulation: Option<RearticulationView>,
    pub(super) rearticulation_score: Vec<f32>,
    #[cfg(test)]
    pub(crate) own_power_scan: Vec<f32>,
    #[cfg(test)]
    pub(crate) other_power_scan: Vec<f64>,
    #[cfg(test)]
    pub(crate) computed_score: Option<Option<Arc<[f32]>>>,
}

/// A candidate action context; shared observation scans remain untouched.
#[derive(Clone, Copy, Debug)]
pub(in crate::life::voice) struct RearticulationView {
    pub weight: f32,
    pub id: u64,
    pub generation: u32,
    pub reference_log2: f32,
    pub threshold_cents: f32,
    pub previous_target: f32,
    pub bounds: (f32, f32),
}

impl LooInput {
    pub(super) fn candidate_scan<'a>(
        &'a self,
        space: &Log2Space,
        pitch_log2: f32,
        external: Option<&'a [f32]>,
    ) -> Option<&'a [f32]> {
        let Some(action) = self.rearticulation else {
            return external;
        };
        let bounds = if action.bounds.0 <= action.bounds.1 {
            action.bounds
        } else {
            space.voice_pitch_bounds_log2()
        };
        let target = pitch_log2.clamp(bounds.0, bounds.1);
        if external.is_some()
            && !self.rearticulation_score.is_empty()
            && action.previous_target > 0.0
            && target != action.previous_target
            && !(action.threshold_cents > 0.0
                && super::super::onset_distance_cents(space, action.reference_log2, target)
                    <= f64::from(action.threshold_cents))
        {
            Some(&self.rearticulation_score)
        } else {
            external
        }
    }
}

pub(super) fn loo_score_scan(landscape: &Landscape, input: &mut LooInput) -> Option<Arc<[f32]>> {
    input.rearticulation_score.clear();
    if let Some(action) = input.rearticulation
        && action.weight > 0.0
        && let Some(observation) = landscape
            .ecology
            .loo
            .iter()
            .find(|s| s.id == action.id && s.generation == action.generation)
    {
        landscape
            .acoustic
            .space
            .assert_scan_len_named(&observation.h_scan, "loo_h_scan");
        landscape
            .acoustic
            .space
            .assert_scan_len_named(&landscape.acoustic.roughness01, "mixed_r_scan");
        // Keep the accepted weight-one arithmetic unchanged.
        if action.weight == 1.0 {
            input.rearticulation_score.extend(
                observation
                    .h_scan
                    .iter()
                    .zip(&landscape.acoustic.roughness01)
                    .map(|(&h, &r)| landscape.acoustic.consonance_kernel.score(h, r)),
            );
        } else {
            landscape
                .acoustic
                .space
                .assert_scan_len_named(&observation.r_scan, "loo_r_scan");
            input.rearticulation_score.extend(
                observation
                    .h_scan
                    .iter()
                    .zip(&observation.r_scan)
                    .zip(&landscape.acoustic.roughness01)
                    .map(|((&h, &external), &mixed)| {
                        let r = (1.0 - action.weight) * external + action.weight * mixed;
                        landscape.acoustic.consonance_kernel.score(h, r)
                    }),
            );
        }
    }
    if let Some(scan) = &input.exact_score {
        landscape
            .acoustic
            .space
            .assert_scan_len_named(scan, "loo_score_scan");
    }
    input.exact_score.clone()
}

#[cfg(any(test, feature = "loo-diagnostic"))]
#[derive(Clone, Debug)]
pub(super) struct LooScratch {
    space: Log2Space,
    harmonicity: HarmonicityKernel,
    pub(super) roughness: RoughnessKernel,
    pub(super) potentials: PotentialScratch,
    pub(super) h: Vec<f32>,
    pub(super) r: Vec<f32>,
    #[cfg(test)]
    merge_ranges: Vec<(usize, usize)>,
    #[cfg(test)]
    suppression_ranges: Vec<(usize, usize)>,
    #[cfg(test)]
    erb: Vec<f32>,
    #[cfg(test)]
    raw_density: Vec<f32>,
    #[cfg(test)]
    peak_scratch: crate::core::peak_extraction::PeakScratch,
}

#[cfg(any(test, feature = "loo-diagnostic"))]
pub(super) fn exact_loo_consonance_score_scan(
    landscape: &Landscape,
    current_pitch_log2: f32,
    enabled: bool,
    mode: ReferenceLooMode,
    scratch: &mut Option<Box<LooScratch>>,
) -> Option<Vec<f32>> {
    if !enabled || !matches!(mode, ReferenceLooMode::ExactScan) {
        return None;
    }
    let n = landscape.acoustic.space.n_bins();
    // F2: a misaligned landscape is a bug, but the exact-LOO path degrades to the
    // approximate score rather than panicking mid-proposal. Flag it in dev builds,
    // one scan at a time so the offender is identifiable.
    debug_assert_eq!(
        landscape.acoustic.subjective_intensity.len(),
        n,
        "exact-LOO scan length mismatch: subjective_intensity"
    );
    debug_assert_eq!(
        landscape.acoustic.roughness01.len(),
        n,
        "exact-LOO scan length mismatch: roughness01"
    );
    debug_assert_eq!(
        landscape.acoustic.harmonicity01.len(),
        n,
        "exact-LOO scan length mismatch: harmonicity01"
    );
    if n == 0
        || landscape.acoustic.subjective_intensity.len() != n
        || landscape.acoustic.roughness01.len() != n
        || landscape.acoustic.harmonicity01.len() != n
    {
        return None;
    }
    let current_idx = landscape.acoustic.space.index_of_log2(current_pitch_log2)?;
    let current_mass = landscape
        .acoustic
        .subjective_intensity
        .get(current_idx)
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)?;
    let mut density_loo = landscape.acoustic.subjective_intensity.clone();
    density_loo[current_idx] = (density_loo[current_idx] - current_mass).max(0.0);

    Some(recompute_loo_score_scan(landscape, &density_loo, scratch))
}

#[cfg(any(test, feature = "loo-diagnostic"))]
pub(super) fn prepare_loo_scratch<'a>(
    landscape: &Landscape,
    scratch: &'a mut Option<Box<LooScratch>>,
) -> &'a mut LooScratch {
    let n = landscape.acoustic.space.n_bins();
    let stale = scratch.as_ref().is_none_or(|cache| {
        cache.space.fmin != landscape.acoustic.space.fmin
            || cache.space.fmax != landscape.acoustic.space.fmax
            || cache.space.bins_per_oct != landscape.acoustic.space.bins_per_oct
            || cache.space.step_log2 != landscape.acoustic.space.step_log2
            || cache.space.centers_hz != landscape.acoustic.space.centers_hz
            || cache.space.centers_log2 != landscape.acoustic.space.centers_log2
            || cache.harmonicity.params != landscape.acoustic.harmonicity_params
            || cache.roughness.params != landscape.acoustic.roughness_kernel_params
    });
    if stale {
        let harmonicity = HarmonicityKernel::new(
            &landscape.acoustic.space,
            landscape.acoustic.harmonicity_params,
        );
        let roughness = RoughnessKernel::new(
            landscape.acoustic.roughness_kernel_params,
            EXACT_LOO_ROUGHNESS_ERB_STEP,
        );
        let potentials = PotentialScratch::new(&landscape.acoustic.space, &harmonicity, &roughness);
        *scratch = Some(Box::new(LooScratch {
            space: landscape.acoustic.space.clone(),
            harmonicity,
            roughness,
            potentials,
            h: vec![0.; n],
            r: vec![0.; n],
            #[cfg(test)]
            merge_ranges: vec![(0, 0); n],
            #[cfg(test)]
            suppression_ranges: {
                let (erb, _) = crate::core::roughness_kernel::erb_grid(&landscape.acoustic.space);
                let sep =
                    crate::core::peak_extraction::PeakExtractConfig::nsgt_default().min_sep_erb;
                erb.iter()
                    .map(|&u| {
                        (
                            erb.partition_point(|&v| v <= u - sep),
                            erb.partition_point(|&v| v < u + sep),
                        )
                    })
                    .collect()
            },
            #[cfg(test)]
            erb: crate::core::roughness_kernel::erb_grid(&landscape.acoustic.space).0,
            #[cfg(test)]
            raw_density: vec![0.; n],
            #[cfg(test)]
            peak_scratch: crate::core::peak_extraction::PeakScratch::new(n),
        }));
    }
    scratch.as_mut().unwrap()
}

#[cfg(any(test, feature = "loo-diagnostic"))]
pub(super) fn recompute_loo_score_scan(
    landscape: &Landscape,
    density_loo: &[f32],
    scratch: &mut Option<Box<LooScratch>>,
) -> Vec<f32> {
    let n = landscape.acoustic.space.n_bins();
    let cache = prepare_loo_scratch(landscape, scratch);
    let eps = landscape.acoustic.roughness_ref_eps.max(1e-12);
    cache.potentials.compute_shape(
        density_loo,
        &landscape.acoustic.space,
        (&cache.harmonicity, &cache.roughness),
        eps,
        &mut cache.h,
        &mut cache.r,
    );
    let roughness_ref_peak = landscape.acoustic.roughness_ref_peak.max(eps);
    let roughness_k = landscape.acoustic.roughness_k.max(1e-6);
    let mut out = Vec::with_capacity(n);
    for (i, r_shape) in cache.r.iter().enumerate().take(n) {
        let h01 = cache.h.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let r01 = roughness_ratio_to_state01(*r_shape / roughness_ref_peak, roughness_k);
        out.push(landscape.acoustic.consonance_kernel.score(h01, r01));
    }
    out
}

#[cfg(test)]
#[path = "scoring/test_support.rs"]
pub(super) mod test_support;
