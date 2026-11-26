use super::*;

pub(in super::super) fn approx_loo_score_scan(
    landscape: &Landscape,
    input: &mut LooInput,
    scratch: &mut Option<Box<LooScratch>>,
) -> Option<Arc<[f32]>> {
    // Both empty means no Community ownership snapshot has been supplied.
    if input.own_power_scan.is_empty() && input.other_power_scan.is_empty() {
        return None;
    }
    let space = &landscape.space;
    space.assert_scan_len_named(&input.own_power_scan, "loo_own_power_scan");
    space.assert_scan_len_named(&input.other_power_scan, "loo_other_power_scan");
    space.assert_scan_len_named(&landscape.subjective_intensity, "subjective_intensity_scan");
    if let Some(score) = &input.computed_score {
        return score.clone();
    }

    if !input
        .own_power_scan
        .iter()
        .zip(&landscape.subjective_intensity)
        .any(|(&own, &observed)| own > 0. && own.is_finite() && observed > 0.)
    {
        input.computed_score = Some(None);
        return None;
    }
    let cache = prepare_loo_scratch(landscape, scratch);
    let mut density = landscape.subjective_intensity.clone();
    let removed = redistribute_loo_density(landscape, input, cache, &mut density);
    if !removed {
        input.computed_score = Some(None);
        return None;
    }
    let score: Arc<[f32]> = recompute_loo_score_scan(landscape, &density, scratch).into();
    input.computed_score = Some(Some(score.clone()));
    Some(score)
}

// The extractor assigns all admitted mass to the nearest selected peak, then
// redirects pruned peaks. Its actual assignment supplies the redistribution basin.
// Historical smoothed peaks absent from the current frame use the suppression span.
pub(in super::super) fn redistribute_loo_density(
    landscape: &Landscape,
    input: &LooInput,
    cache: &mut LooScratch,
    density: &mut [f32],
) -> bool {
    landscape
        .space
        .assert_scan_len_named(density, "loo_residual_density_scan");
    let space = &landscape.space;
    space.assert_scan_len_named(&landscape.nsgt_power, "nsgt_power_scan");
    let cfg = crate::core::peak_extraction::PeakExtractConfig::nsgt_default();
    for (i, raw) in cache.raw_density.iter_mut().enumerate() {
        *raw = (landscape.nsgt_power[i] / cache.potentials.du[i].max(1e-12)).max(0.);
    }
    cache.merge_ranges.clone_from(&cache.suppression_ranges);
    let peaks = crate::core::peak_extraction::extract_peaks_density_with_grid_reuse(
        &cache.raw_density,
        &cache.erb,
        &cache.potentials.du,
        &cfg,
        &mut cache.peak_scratch,
    );
    for peak in peaks {
        cache.merge_ranges[peak.bin_idx] = (space.n_bins(), 0);
    }
    for (j, &u) in cache.erb.iter().enumerate() {
        if let Some(i) = cache.peak_scratch.mass_owner_bin(u) {
            let range = &mut cache.merge_ranges[i];
            range.0 = range.0.min(j);
            range.1 = range.1.max(j + 1);
        }
    }
    for (i, value) in density.iter_mut().enumerate() {
        if input.own_power_scan[i] > 0.0 && input.own_power_scan[i].is_finite() {
            *value = 0.0;
        }
    }
    let mut removed = false;
    for (i, &own) in input.own_power_scan.iter().enumerate() {
        let observed = landscape.subjective_intensity[i];
        if own <= 0.0 || !own.is_finite() || observed <= 0.0 {
            continue;
        }
        let (lo, hi) = cache.merge_ranges[i];
        let other: f64 = input.other_power_scan[lo..hi].iter().sum();
        let own: f64 = input.own_power_scan[lo..hi].iter().map(|&v| v as f64).sum();
        // Form the residual directly: 1 - own / total loses quiet neighbours.
        let retained =
            (observed as f64 * (other / (own + other)).powf(landscape.loudness_exp as f64)) as f32;
        if other > 0.0 {
            let mass = retained as f64 * cache.potentials.du[i] as f64;
            for (j, &power) in input.other_power_scan.iter().enumerate().take(hi).skip(lo) {
                density[j] += (mass * (power / other) / cache.potentials.du[j] as f64) as f32;
            }
        } else {
            density[i] += retained;
        }
        removed = true;
    }
    removed
}
