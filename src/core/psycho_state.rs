use crate::core::density;
use crate::core::landscape::LandscapeParams;
use crate::core::log2space::Log2Space;
use crate::core::roughness_kernel::erb_grid;

#[derive(Clone, Copy, Debug)]
pub struct RoughnessRef {
    pub total: f32,
    pub peak: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct HarmonicityRef {
    pub max: f32,
}

#[inline]
pub fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

#[inline]
pub fn clamp_pm1(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

/// Normalize a density curve for potential scans, returning (normalized, mass).
pub fn normalize_density(density_vals: &[f32], du: &[f32], eps: f32) -> (Vec<f32>, f32) {
    let mass = density::density_to_mass(density_vals, du);
    if mass <= eps {
        return (vec![0.0; density_vals.len()], mass);
    }
    let inv = 1.0 / (mass + eps);
    let norm = density_vals.iter().map(|&v| v * inv).collect();
    (norm, mass)
}

/// Convert roughness ratio (relative to reference) into normalized state [0, 1].
pub fn roughness_ratio_to_state01(ratio: f32, k: f32) -> f32 {
    if ratio.is_nan() {
        return 0.0;
    }
    if ratio.is_infinite() {
        return if ratio.is_sign_positive() { 1.0 } else { 0.0 };
    }
    let ratio = ratio.max(0.0);
    let k = if k.is_finite() { k.max(1e-6) } else { 1e-6 };
    let r_ref = 1.0 / (1.0 + k);
    if ratio <= 0.0 {
        0.0
    } else if ratio < 1.0 {
        (ratio * r_ref).clamp(0.0, 1.0)
    } else {
        let denom = ratio + k;
        if denom <= 0.0 {
            return 1.0;
        }
        (1.0 - k / denom).clamp(0.0, 1.0)
    }
}

pub fn r_pot_scan_to_r_state01_scan(r_pot_scan: &[f32], r_ref_peak: f32, k: f32, out: &mut [f32]) {
    debug_assert_eq!(out.len(), r_pot_scan.len());
    let denom = if r_ref_peak.is_finite() && r_ref_peak > 0.0 {
        r_ref_peak
    } else {
        1.0
    };
    let len = out.len().min(r_pot_scan.len());
    for i in 0..len {
        let ratio = r_pot_scan[i] / denom;
        out[i] = roughness_ratio_to_state01(ratio, k);
    }
}

pub fn h_pot_scan_to_h_state01_scan(h_pot_scan: &[f32], h_ref_max: f32, out: &mut [f32]) {
    debug_assert_eq!(out.len(), h_pot_scan.len());
    let denom = if h_ref_max.is_finite() && h_ref_max > 0.0 {
        h_ref_max
    } else {
        1.0
    };
    let len = out.len().min(h_pot_scan.len());
    for i in 0..len {
        out[i] = clamp01(h_pot_scan[i] / denom);
    }
}

fn sanitize01(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else if x.is_infinite() {
        if x.is_sign_positive() { 1.0 } else { 0.0 }
    } else {
        0.0
    }
}

fn sanitize_nonneg(x: f32) -> f32 {
    if x.is_finite() { x.max(0.0) } else { 0.0 }
}

pub fn compose_c_statepm1(
    h_state01: f32,
    r_state01: f32,
    alpha: f32,
    w0: f32,
    w1: f32,
) -> (f32, f32) {
    let h01 = sanitize01(h_state01);
    let r01 = sanitize01(r_state01);
    let alpha = sanitize_nonneg(alpha);
    let w0 = sanitize_nonneg(w0);
    let w1 = sanitize_nonneg(w1);
    let dh = 1.0 - h01;
    let w = w0 + w1 * dh;
    let d = alpha * dh + w * r01;
    let c01 = if d.is_finite() {
        let denom = 1.0 + d.max(0.0);
        if denom <= 0.0 { 1.0 } else { 1.0 / denom }
    } else if d.is_sign_negative() {
        1.0
    } else {
        0.0
    };
    let c_signed = clamp_pm1(2.0 * c01 - 1.0);
    (c_signed, c01)
}

pub fn compose_c_statepm1_scan(
    h_state01: &[f32],
    r_state01: &[f32],
    alpha: f32,
    w0: f32,
    w1: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(h_state01.len(), r_state01.len());
    debug_assert_eq!(out.len(), h_state01.len());
    let len = out.len().min(h_state01.len()).min(r_state01.len());
    for i in 0..len {
        let (c_signed, _) = compose_c_statepm1(h_state01[i], r_state01[i], alpha, w0, w1);
        out[i] = c_signed;
    }
}

pub fn roughness_ref_from_r_pot_scan(r_pot_scan: &[f32], du: &[f32]) -> RoughnessRef {
    debug_assert_eq!(r_pot_scan.len(), du.len());
    let peak = r_pot_scan.iter().copied().fold(0.0f32, f32::max);
    let total = density::density_to_mass(r_pot_scan, du);
    RoughnessRef { total, peak }
}

pub fn build_roughness_reference_density(params: &LandscapeParams, space: &Log2Space) -> Vec<f32> {
    let (erb, du) = erb_grid(space);
    if erb.is_empty() {
        return Vec::new();
    }
    let f0 = params.roughness_ref_f0_hz.max(1.0);
    let a = nearest_bin_by_freq(space, f0);
    let target_erb = erb[a] + params.roughness_ref_sep_erb;
    let mut b = nearest_bin_by_erb(&erb, target_erb);
    if b == a && erb.len() > 1 {
        b = if a + 1 < erb.len() {
            a + 1
        } else {
            a.saturating_sub(1)
        };
    }

    let mut ref_density = vec![0.0f32; erb.len()];
    let m_a = params.roughness_ref_mass_split.clamp(0.0, 1.0);
    let m_b = 1.0 - m_a;
    if du[a] > 0.0 {
        ref_density[a] += m_a / du[a];
    }
    if b != a && du[b] > 0.0 {
        ref_density[b] += m_b / du[b];
    } else if b == a && du[a] > 0.0 {
        ref_density[a] += m_b / du[a];
    }

    ref_density
}

pub fn compute_roughness_reference(params: &LandscapeParams, space: &Log2Space) -> RoughnessRef {
    let eps = params.roughness_ref_eps.max(1e-12);
    let ref_density = build_roughness_reference_density(params, space);
    let (r_pot_scan, r_total) = params
        .roughness_kernel
        .potential_r_from_log2_spectrum_density(&ref_density, space);
    let (_erb, du) = erb_grid(space);
    let mut ref_vals = roughness_ref_from_r_pot_scan(&r_pot_scan, &du);
    ref_vals.total = r_total.max(eps);
    ref_vals.peak = ref_vals.peak.max(eps);
    ref_vals
}

fn nearest_bin_by_freq(space: &Log2Space, f0: f32) -> usize {
    let mut best = 0;
    let mut best_diff = f32::MAX;
    for (i, &f) in space.centers_hz.iter().enumerate() {
        let diff = (f - f0).abs();
        if diff < best_diff {
            best_diff = diff;
            best = i;
        }
    }
    best
}

fn nearest_bin_by_erb(erb: &[f32], target: f32) -> usize {
    let mut best = 0;
    let mut best_diff = f32::MAX;
    for (i, &u) in erb.iter().enumerate() {
        let diff = (u - target).abs();
        if diff < best_diff {
            best_diff = diff;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
    use crate::core::landscape::RoughnessScalarMode;
    use crate::core::log2space::Log2Space;
    use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};

    fn build_params(space: &Log2Space) -> LandscapeParams {
        LandscapeParams {
            fs: 48_000.0,
            max_hist_cols: 1,
            alpha: 0.0,
            roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005),
            harmonicity_kernel: HarmonicityKernel::new(space, HarmonicityParams::default()),
            roughness_scalar_mode: RoughnessScalarMode::Total,
            roughness_half: 0.1,
            consonance_harmonicity_deficit_weight: 1.0,
            consonance_roughness_weight_floor: 0.35,
            consonance_roughness_weight: 0.5,
            loudness_exp: 1.0,
            ref_power: 1.0,
            tau_ms: 1.0,
            roughness_k: 1.0,
            roughness_ref_f0_hz: 1000.0,
            roughness_ref_sep_erb: 0.25,
            roughness_ref_mass_split: 0.5,
            roughness_ref_eps: 1e-12,
        }
    }

    #[test]
    fn roughness_ref_matches_reference_scan() {
        let space = Log2Space::new(80.0, 8000.0, 96);
        let params = build_params(&space);
        let ref_density = build_roughness_reference_density(&params, &space);
        let (r_pot_scan, r_total) = params
            .roughness_kernel
            .potential_r_from_log2_spectrum_density(&ref_density, &space);
        let (_erb, du) = erb_grid(&space);
        let ref_vals = roughness_ref_from_r_pot_scan(&r_pot_scan, &du);
        let legacy_total = density::density_to_mass(&r_pot_scan, &du);
        let legacy_peak = r_pot_scan.iter().copied().fold(0.0f32, f32::max);
        assert!((ref_vals.total - legacy_total).abs() < 1e-6);
        assert!((ref_vals.peak - legacy_peak).abs() < 1e-6);
        let ref_from_kernel = compute_roughness_reference(&params, &space);
        assert!((ref_from_kernel.total - r_total).abs() < 1e-6);
    }

    #[test]
    fn roughness_ratio_state01_matches() {
        let k: f32 = 0.3;
        let legacy = |ratio: f32| {
            if ratio.is_nan() {
                return 0.0;
            }
            if ratio.is_infinite() {
                return if ratio.is_sign_positive() { 1.0 } else { 0.0 };
            }
            let ratio = ratio.max(0.0);
            let k = k.max(1e-6);
            let r_ref = 1.0 / (1.0 + k);
            if ratio <= 0.0 {
                0.0
            } else if ratio < 1.0 {
                (ratio * r_ref).clamp(0.0, 1.0)
            } else {
                let denom = ratio + k;
                if denom <= 0.0 {
                    return 1.0;
                }
                (1.0 - k / denom).clamp(0.0, 1.0)
            }
        };
        for &ratio in &[0.0, 0.5, 1.0, 10.0, f32::INFINITY] {
            let new_val = roughness_ratio_to_state01(ratio, k);
            let old_val = legacy(ratio);
            assert!((new_val - old_val).abs() < 1e-6);
        }
    }

    #[test]
    fn compose_c_statepm1_scan_matches_scalar() {
        let h = vec![0.0, 0.2, 0.8, 1.0];
        let r = vec![0.0, 0.1, 0.5, 1.0];
        let mut out = vec![0.0; h.len()];
        let w0 = 0.35;
        let w1 = 0.75;
        for &alpha in &[1.0, 2.0] {
            compose_c_statepm1_scan(&h, &r, alpha, w0, w1, &mut out);
            for i in 0..h.len() {
                let h01 = h[i].clamp(0.0, 1.0);
                let r01 = r[i].clamp(0.0, 1.0);
                let dh = 1.0 - h01;
                let w = w0 + w1 * dh;
                let d = alpha * dh + w * r01;
                let c01_expected = 1.0 / (1.0 + d.max(0.0));
                let cpm1_expected = (2.0 * c01_expected - 1.0).clamp(-1.0, 1.0);

                let (cpm1, c01) = compose_c_statepm1(h[i], r[i], alpha, w0, w1);
                assert!(
                    (c01 - c01_expected).abs() < 1e-6,
                    "alpha={alpha} i={i} c01={c01} expected={c01_expected}"
                );
                assert!(
                    (cpm1 - cpm1_expected).abs() < 1e-6,
                    "alpha={alpha} i={i} cpm1={cpm1} expected={cpm1_expected}"
                );
                assert!(
                    (out[i] - cpm1_expected).abs() < 1e-6,
                    "alpha={alpha} i={i} scan={scan} expected={cpm1_expected}",
                    scan = out[i]
                );
            }
        }
    }

    #[test]
    fn alpha_increases_penalty_when_h_low() {
        let h = 0.2;
        let r = 0.2;
        let w0 = 0.35;
        let w1 = 0.75;
        let (_cpm1_low, c01_low) = compose_c_statepm1(h, r, 0.5, w0, w1);
        let (_cpm1_high, c01_high) = compose_c_statepm1(h, r, 2.0, w0, w1);
        assert!(c01_high < c01_low, "expected higher alpha to reduce C01");
    }

    #[test]
    fn w0_floor_preserves_roughness_penalty_at_h1() {
        let h = 1.0;
        let r = 0.4;
        let alpha = 1.0;
        let w0 = 0.35;
        let w1 = 0.75;
        let (_cpm1, c01) = compose_c_statepm1(h, r, alpha, w0, w1);
        assert!(c01 < 1.0, "expected w0 to reduce C01 when h=1 and r>0");
    }
}
