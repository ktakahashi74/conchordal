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

pub fn w_of_h(w0: f32, w1: f32, h01: f32) -> f32 {
    let h01 = sanitize01(h01);
    let w0 = sanitize_nonneg(w0);
    let w1 = sanitize_nonneg(w1);
    let dh = 1.0 - h01;
    w0 + w1 * dh
}

pub fn compose_c_score(alpha: f32, w0: f32, w1: f32, h01: f32, r01: f32) -> f32 {
    let h01 = sanitize01(h01);
    let r01 = sanitize01(r01);
    let alpha = sanitize_nonneg(alpha);
    let w = w_of_h(w0, w1, h01);
    let c_score = alpha * h01 - w * r01;
    if c_score.is_finite() { c_score } else { 0.0 }
}

pub fn sigmoid01(x: f32) -> f32 {
    if x.is_nan() {
        return 0.0;
    }
    let x = x.clamp(-80.0, 80.0);
    1.0 / (1.0 + (-x).exp())
}

pub fn compose_c_state(beta: f32, theta: f32, c_score: f32) -> f32 {
    let beta = sanitize_nonneg(beta);
    let theta = if theta.is_finite() { theta } else { 0.0 };
    let c_score = if c_score.is_finite() { c_score } else { 0.0 };
    sigmoid01(beta * (c_score - theta))
}

pub fn compute_c01_from_hr(h01: f32, r01: f32, wr: f32) -> f32 {
    compute_c01_from_hr_params(h01, r01, wr, 1.0, 0.35, 0.5, 2.0, 0.0)
}

#[allow(clippy::too_many_arguments)]
pub fn compute_c01_from_hr_params(
    h01: f32,
    r01: f32,
    wr: f32,
    alpha: f32,
    base_w0: f32,
    base_w1: f32,
    beta: f32,
    theta: f32,
) -> f32 {
    let wr = sanitize01(wr);
    let w0 = sanitize_nonneg(base_w0) * wr;
    let w1 = sanitize_nonneg(base_w1) * wr;
    let c_score = compose_c_score(alpha, w0, w1, h01, r01);
    compose_c_state(beta, theta, c_score)
}

#[allow(clippy::too_many_arguments)]
pub fn compose_c_state01_scan(
    h_state01: &[f32],
    r_state01: &[f32],
    alpha: f32,
    w0: f32,
    w1: f32,
    beta: f32,
    theta: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(h_state01.len(), r_state01.len());
    debug_assert_eq!(out.len(), h_state01.len());
    let len = out.len().min(h_state01.len()).min(r_state01.len());
    for i in 0..len {
        let c_score = compose_c_score(alpha, w0, w1, h_state01[i], r_state01[i]);
        out[i] = compose_c_state(beta, theta, c_score);
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
            consonance_harmonicity_weight: 1.0,
            consonance_roughness_weight_floor: 0.35,
            consonance_roughness_weight: 0.5,
            c_state_beta: 2.0,
            c_state_theta: 0.0,
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
    fn compose_c_state01_scan_matches_scalar() {
        let h = vec![0.0, 0.2, 0.8, 1.0];
        let r = vec![0.0, 0.1, 0.5, 1.0];
        let mut out = vec![0.0; h.len()];
        let w0 = 0.35;
        let w1 = 0.75;
        let beta = 2.0;
        let theta = 0.1;
        for &alpha in &[0.8, 1.6] {
            compose_c_state01_scan(&h, &r, alpha, w0, w1, beta, theta, &mut out);
            for i in 0..h.len() {
                let c_score = compose_c_score(alpha, w0, w1, h[i], r[i]);
                let c_state = compose_c_state(beta, theta, c_score);
                assert!(
                    (out[i] - c_state).abs() < 1e-6,
                    "alpha={alpha} i={i} out={out_val} expected={c_state}",
                    out_val = out[i]
                );
            }
        }
    }

    #[test]
    fn compose_c_score_matches_extremes() {
        let alpha = 1.5;
        let w0 = 0.35;
        let w1 = 0.65;
        let c_hi = compose_c_score(alpha, w0, w1, 1.0, 0.0);
        let c_lo = compose_c_score(alpha, w0, w1, 0.0, 1.0);
        assert!((c_hi - alpha).abs() < 1e-6, "c_hi={c_hi} alpha={alpha}");
        assert!(
            (c_lo + (w0 + w1)).abs() < 1e-6,
            "c_lo={c_lo} expected={}",
            -(w0 + w1)
        );
    }

    #[test]
    fn compose_c_state_midpoint_and_extremes() {
        let beta = 2.0;
        let theta = 0.25;
        let mid = compose_c_state(beta, theta, theta);
        assert!((mid - 0.5).abs() < 1e-6, "mid={mid}");

        let hi = compose_c_state(beta, theta, theta + 10.0);
        let lo = compose_c_state(beta, theta, theta - 10.0);
        assert!(hi > 0.999, "hi={hi}");
        assert!(lo < 0.001, "lo={lo}");

        let a = compose_c_state(beta, theta, theta - 0.5);
        let b = compose_c_state(beta, theta, theta);
        let c = compose_c_state(beta, theta, theta + 0.5);
        assert!(a < b && b < c, "monotonicity failed: {a} {b} {c}");
    }

    #[test]
    fn compose_c_state_sanitizes_negative_beta() {
        let beta = -3.0;
        let theta = 0.1;
        let low = compose_c_state(beta, theta, -1.0);
        let high = compose_c_state(beta, theta, 1.0);
        assert!((low - 0.5).abs() < 1e-6, "low={low}");
        assert!((high - 0.5).abs() < 1e-6, "high={high}");
        assert!(high >= low - 1e-6, "monotonicity failed: {low} {high}");
    }

    #[test]
    fn sigmoid01_saturates_extremes() {
        let hi = sigmoid01(1000.0);
        let lo = sigmoid01(-1000.0);
        assert!(hi > 0.999, "hi={hi}");
        assert!(lo < 0.001, "lo={lo}");
    }

    #[test]
    fn compute_c01_from_hr_is_monotonic_in_wr() {
        let h01 = 0.62;
        let r01 = 0.88;
        let c1 = compute_c01_from_hr(h01, r01, 1.0);
        let c05 = compute_c01_from_hr(h01, r01, 0.5);
        let c0 = compute_c01_from_hr(h01, r01, 0.0);
        assert!(
            c0 >= c05 && c05 >= c1,
            "expected c01(wr=0)>=c01(wr=0.5)>=c01(wr=1), got {c0}, {c05}, {c1}"
        );
    }

    #[test]
    fn compute_c01_from_hr_wr_zero_disables_roughness_term() {
        let h01 = 0.41;
        let c_r0 = compute_c01_from_hr(h01, 0.0, 0.0);
        let c_r1 = compute_c01_from_hr(h01, 1.0, 0.0);
        assert!(
            (c_r0 - c_r1).abs() < 1e-6,
            "wr=0 should remove roughness effect: c_r0={c_r0}, c_r1={c_r1}"
        );
    }
}
