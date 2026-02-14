//! core/roughness_kernel.rs — perc_potential_R via ERB-domain kernel convolution.
//! Computes frequency-space roughness potential by convolving the
//! envelope energy using an asymmetric kernel.
//! density: per-ERB power density; mass: sum(density * du) over ERB.

use crate::core::density;
use crate::core::erb::hz_to_erb;
use crate::core::fft::apply_hann_window_complex;
#[cfg(test)]
use crate::core::fft::hilbert;
use crate::core::log2space::Log2Space;
use crate::core::peak_extraction::Peak;
use rustfft::{FftPlanner, num_complex::Complex32};

// ======================================================================
// Kernel parameter definition (Plomp–Levelt inspired, ΔERB domain)
// ======================================================================

#[derive(Clone, Copy, Debug)]
pub struct KernelParams {
    // === Cochlear layer ===
    pub sigma_erb: f32,
    pub tau_erb: f32,
    pub mix_tail: f32,
    pub half_width_erb: f32,
    pub suppress_sigma_erb: f32,
    pub suppress_pow: f32,
    // === Neural layer ===
    pub sigma_neural_erb: f32,
    pub w_neural: f32,
}

impl Default for KernelParams {
    fn default() -> Self {
        Self {
            sigma_erb: 0.45,
            tau_erb: 1.0,
            mix_tail: 0.20,
            half_width_erb: 4.0,
            suppress_sigma_erb: 0.06,
            suppress_pow: 1.0,
            sigma_neural_erb: 1.0,
            w_neural: 0.0,
        }
    }
}

// ======================================================================
// Core kernel evaluation
// ======================================================================

/// Evaluate kernel at `d_erb = erb_probe - erb_masker`.
/// `d_erb >= 0` means the probe is above the masker (upward direction).
#[inline]
pub fn eval_kernel_delta_erb(params: &KernelParams, d_erb: f32) -> f32 {
    let sigma = params.sigma_erb.max(1e-6);
    let tau = params.tau_erb.max(1e-6);
    let s_sup = params.suppress_sigma_erb.max(1e-6);
    let sig_n = params.sigma_neural_erb.max(1e-6);

    let desq = d_erb * d_erb;
    let g_gauss = (-desq / (2.0 * sigma * sigma)).exp();
    let g_tail = if d_erb >= 0.0 {
        (-d_erb / tau).exp()
    } else {
        0.0
    };
    let base = (1.0 - params.mix_tail) * g_gauss + params.mix_tail * g_tail;

    let suppress = (1.0 - (-desq / (2.0 * s_sup * s_sup)).exp()).clamp(0.0, 1.0);
    let g_coch = base * suppress.powf(params.suppress_pow);
    let g_neural = (-desq / (2.0 * sig_n * sig_n)).exp();
    (1.0 - params.w_neural) * g_coch + params.w_neural * g_neural
}

pub fn build_kernel_erbstep(params: &KernelParams, erb_step: f32) -> (Vec<f32>, usize) {
    let hw_erb = params.half_width_erb;
    let n_side = (hw_erb / erb_step).ceil() as usize;
    let len = 2 * n_side + 1;

    let mut g: Vec<f32> = (0..len)
        .map(|i| {
            let d = (i as i32 - n_side as i32) as f32 * erb_step;
            eval_kernel_delta_erb(params, d)
        })
        .collect();

    let sum = g.iter().sum::<f32>() * erb_step;
    if sum > 0.0 {
        let inv = 1.0 / sum;
        g.iter_mut().for_each(|v| *v *= inv);
    }
    (g, n_side)
}

#[inline]
fn lut_interp(lut: &[f32], step: f32, hw: usize, d_erb: f32) -> f32 {
    let t = d_erb / step + hw as f32;
    let i = t.floor();
    let i0 = i as isize;
    let i1 = i0 + 1;
    if i0 < 0 || (i1 as usize) >= lut.len() {
        return 0.0;
    }
    let frac = t - i;
    let a = lut[i0 as usize];
    let b = lut[i1 as usize];
    a + frac * (b - a)
}

fn local_du_from_grid(erb: &[f32]) -> Vec<f32> {
    let n = erb.len();
    let mut du = vec![0.0; n];
    if n == 0 {
        return du;
    }
    if n == 1 {
        return du;
    }
    du[0] = (erb[1] - erb[0]).max(0.0);
    du[n - 1] = (erb[n - 1] - erb[n - 2]).max(0.0);
    for i in 1..n - 1 {
        du[i] = 0.5 * (erb[i + 1] - erb[i - 1]).max(0.0);
    }
    du
}

pub(crate) fn erb_grid(space: &Log2Space) -> (Vec<f32>, Vec<f32>) {
    let erb: Vec<f32> = space.centers_hz.iter().map(|&f| hz_to_erb(f)).collect();
    let du = local_du_from_grid(&erb);
    (erb, du)
}

// ======================================================================
// Roughness kernel (holds LUT, no global cache)
// ======================================================================

#[derive(Clone, Debug)]
pub struct RoughnessKernel {
    pub params: KernelParams,
    pub erb_step: f32,
    pub lut: Vec<f32>,
    pub hw: usize,
}

impl RoughnessKernel {
    /// Create a new kernel and precompute LUT.
    pub fn new(params: KernelParams, erb_step: f32) -> Self {
        let (lut, hw) = build_kernel_erbstep(&params, erb_step);
        Self {
            params,
            erb_step,
            lut,
            hw,
        }
    }

    // ------------------------------------------------------------------
    // Potential R from amplitude spectrum (linear frequency axis)
    // d = erb_probe - erb_masker
    // ------------------------------------------------------------------

    pub fn potential_r_from_spectrum(&self, amps_hz: &[f32], fs: f32) -> (Vec<f32>, f32) {
        let n = amps_hz.len();
        if n == 0 {
            return (vec![], 0.0);
        }

        let nfft = 2 * n;
        let df = fs / nfft as f32;

        let f: Vec<f32> = (0..n).map(|i| i as f32 * df).collect();
        let erb: Vec<f32> = f.iter().map(|&x| hz_to_erb(x)).collect();
        let half_width = self.params.half_width_erb;
        let du = local_du_from_grid(&erb);

        let mut r = vec![0.0f32; n];
        for i in 0..n {
            let fi_erb = erb[i];
            let mut sum = 0.0f32;

            let j_lo = erb.partition_point(|&x| x < fi_erb - half_width);
            let j_hi = erb.partition_point(|&x| x <= fi_erb + half_width);
            for j in j_lo..j_hi {
                if j == i {
                    continue;
                }
                let d = fi_erb - erb[j];
                if d.abs() > half_width {
                    continue;
                }
                let w = lut_interp(&self.lut, self.erb_step, self.hw, d);
                sum += amps_hz[j] * w * du[j];
            }
            r[i] = sum;
        }

        // Integrate over ERB axis
        let r_total = density::density_to_mass(&r, &du);

        (r, r_total)
    }

    // ------------------------------------------------------------------
    // Potential R from analytic (Hilbert) signal
    // ------------------------------------------------------------------

    pub fn potential_r_from_analytic(&self, analytic: &[Complex32], fs: f32) -> (Vec<f32>, f32) {
        if analytic.is_empty() {
            return (vec![], 0.0);
        }

        let n0 = analytic.len();
        let n = n0.next_power_of_two();
        let mut buf: Vec<Complex32> = Vec::with_capacity(n);
        buf.extend_from_slice(analytic);
        if n > n0 {
            buf.resize(n, Complex32::new(0.0, 0.0));
        }

        // Apply Hann window (complex)
        let window_gain = apply_hann_window_complex(&mut buf);

        // FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut buf);

        // Convert to amplitude spectrum
        let n_half = n / 2;
        let df = fs / n as f32;
        let base_scale = 1.0 / (fs * n as f32 * window_gain);
        let amps: Vec<f32> = (0..n_half)
            .map(|i| (buf[i].norm_sqr() * base_scale * 2.0 * df).sqrt())
            .collect();

        self.potential_r_from_spectrum(&amps, fs)
    }

    /// Compute perc_potential_R roughness from log2-domain amplitude spectrum (NSGT).
    /// Uses `d = erb_probe - erb_masker`.
    /// Input values are ERB power densities (mass per ERB), so the internal
    /// accumulation performs a du-weighted integral. This models the potential
    /// roughness increase from adding a unit pure tone.
    pub fn potential_r_from_log2_spectrum_density(
        &self,
        amps_density: &[f32],
        space: &Log2Space,
    ) -> (Vec<f32>, f32) {
        use crate::core::erb::erb_to_hz;

        if amps_density.is_empty() || space.centers_hz.is_empty() {
            return (vec![], 0.0);
        }
        assert_eq!(
            amps_density.len(),
            space.centers_hz.len(),
            "amps and space length mismatch"
        );

        let n = amps_density.len();

        // (1) Map to ERB axis
        let (erb, du) = erb_grid(space);

        // (2) Convolution over ERB axis
        let half_width_erb = self.params.half_width_erb;
        let mut r = vec![0.0f32; n];

        for i in 0..n {
            let fi_erb = erb[i];
            let mut sum = 0.0f32;

            let lo_hz = erb_to_hz(fi_erb - half_width_erb);
            let hi_hz = erb_to_hz(fi_erb + half_width_erb);
            let j_lo = space.index_of_freq(lo_hz).unwrap_or(0);
            let j_hi = space.index_of_freq(hi_hz).unwrap_or(n - 1);

            for j in j_lo..j_hi {
                if j == i {
                    continue;
                }
                let d = fi_erb - erb[j];
                let w = lut_interp(&self.lut, self.erb_step, self.hw, d);
                sum += amps_density[j] * w * du[j];
            }
            r[i] = sum;
        }

        // (3) Integration over ERB axis
        let r_total = density::density_to_mass(&r, &du);

        (r, r_total)
    }

    /// Compatibility wrapper for ERB-density input.
    pub fn potential_r_from_log2_spectrum(
        &self,
        amps: &[f32],
        space: &Log2Space,
    ) -> (Vec<f32>, f32) {
        self.potential_r_from_log2_spectrum_density(amps, space)
    }

    /// Compute perc_potential_R roughness from delta peaks (pure-tone interactions).
    /// Uses `d = erb_probe - erb_masker`.
    /// Each peak mass is the ERB-integrated area (sum of density * du).
    pub fn potential_r_from_peaks(&self, peaks: &[Peak], space: &Log2Space) -> Vec<f32> {
        if peaks.is_empty() || space.centers_hz.is_empty() {
            return vec![];
        }

        use crate::core::erb::erb_to_hz;

        let (erb, _du) = erb_grid(space);
        let half_width_erb = self.params.half_width_erb;
        let mut r = vec![0.0f32; erb.len()];

        for (i, &u_i) in erb.iter().enumerate() {
            let mut sum = 0.0f32;
            let lo_hz = erb_to_hz(u_i - half_width_erb);
            let hi_hz = erb_to_hz(u_i + half_width_erb);
            let j_lo = space.index_of_freq(lo_hz).unwrap_or(0);
            let j_hi = space.index_of_freq(hi_hz).unwrap_or(erb.len() - 1);
            for peak in peaks {
                if peak.bin_idx == i {
                    continue;
                }
                let d = u_i - peak.u_erb;
                if d.abs() > half_width_erb {
                    continue;
                }
                if peak.bin_idx < erb.len() && (peak.bin_idx < j_lo || peak.bin_idx >= j_hi) {
                    continue;
                }
                let w = lut_interp(&self.lut, self.erb_step, self.hw, d);
                sum += peak.mass * w;
            }
            r[i] = sum;
        }

        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::erb::{erb_bw_hz, hz_to_erb};
    use crate::core::peak_extraction::{PeakExtractConfig, extract_peaks_density};
    use plotters::prelude::*;
    use rand::{Rng, SeedableRng};
    use std::fs::File;
    use std::path::Path;

    const ERB_STEP: f32 = 0.005;

    fn ensure_plots_dir() -> std::io::Result<()> {
        std::fs::create_dir_all("target/plots")
    }

    fn make_kernel() -> RoughnessKernel {
        let p = KernelParams::default();
        RoughnessKernel::new(p, ERB_STEP)
    }

    // ------------------------------------------------------------
    // Basic kernel characteristics
    // ------------------------------------------------------------

    #[test]
    fn kernel_is_positive_and_centered() {
        let k = make_kernel();
        let g = &k.lut;
        let hw = k.hw;
        assert!(g.iter().all(|&v| v >= 0.0));
        assert_eq!(g.len(), 2 * hw + 1);
        let edge_mean = (g[0] + g[g.len() - 1]) * 0.5;

        let peak = find_first_peak(g, hw, true)
            .expect("no positive-side peak")
            .1
            .max(1e-12);
        assert!(
            edge_mean / peak < 5e-3,
            "edges should decay (edge/peak={})",
            edge_mean / peak
        );
    }

    #[test]
    fn kernel_center_is_suppressed() {
        let k = make_kernel();
        let g = &k.lut;
        let hw = k.hw;
        let center = g[hw];

        let peak = find_first_peak(g, hw, true)
            .expect("no positive-side peak")
            .1;

        // Allow true suppression to zero when parameters enforce it.
        assert!(center >= 0.0, "center should be non-negative");
        // Suppressed means well below the peak (empirical ~15-30%).
        assert!(
            center < 0.5 * peak,
            "center suppression too weak: c/peak={}",
            center / peak
        );
    }

    #[test]
    fn kernel_has_peak_near_pm0p3_erb() {
        let k = make_kernel();
        let g = &k.lut;
        let hw = k.hw;
        let n = g.len();
        let d_erb: Vec<f32> = (0..n)
            .map(|i| (i as i32 - hw as i32) as f32 * ERB_STEP)
            .collect();

        let (pos_x, pos_val) = find_first_peak(g, hw, true).expect("no positive-side peak");
        let (neg_x, _neg_val) = find_first_peak(g, hw, false).expect("no negative-side peak");
        assert!(
            pos_x > 0.1 && pos_x < 0.5,
            "positive peak away from center (got {:.2})",
            pos_x
        );
        assert!(
            neg_x < -0.1 && neg_x > -0.5,
            "negative peak away from center (got {:.2})",
            neg_x
        );
        assert!(pos_val > g[hw] * 1.5);
    }

    #[test]
    fn kernel_is_asymmetric() {
        let k = make_kernel();
        let g = &k.lut;
        let hw = k.hw;
        let pos_val = find_first_peak(g, hw, true)
            .expect("no positive-side peak")
            .1;
        let neg_val = find_first_peak(g, hw, false)
            .expect("no negative-side peak")
            .1;
        assert!(pos_val > neg_val);
    }

    fn find_first_peak(g: &[f32], hw: usize, positive: bool) -> Option<(f32, f32)> {
        let n = g.len();
        for i in 1..n - 1 {
            if g[i] > g[i - 1] && g[i] > g[i + 1] {
                let d = (i as i32 - hw as i32) as f32 * ERB_STEP;
                if positive && d > 0.0 {
                    return Some((d, g[i]));
                }
                if !positive && d < 0.0 {
                    return Some((d, g[i]));
                }
            }
        }
        None
    }

    fn find_max_peak(g: &[f32], hw: usize, positive: bool) -> Option<(f32, f32)> {
        let mut best: Option<(f32, f32)> = None;
        for (i, &val) in g.iter().enumerate() {
            let d = (i as i32 - hw as i32) as f32 * ERB_STEP;
            if positive && d <= 0.0 {
                continue;
            }
            if !positive && d >= 0.0 {
                continue;
            }
            if best.is_none() || val > best.unwrap().1 {
                best = Some((d, val));
            }
        }
        best
    }

    fn nearest_erb_index(erb: &[f32], target: f32) -> usize {
        let mut best_idx = 0usize;
        let mut best_err = f32::INFINITY;
        for (i, &u) in erb.iter().enumerate() {
            let err = (u - target).abs();
            if err < best_err {
                best_err = err;
                best_idx = i;
            }
        }
        best_idx
    }

    #[test]
    fn kernel_l1_norm_is_one() {
        let k = make_kernel();
        let g = &k.lut;
        let sum: f32 = g.iter().sum();
        let sum_cont = sum * k.erb_step;
        assert!((sum_cont - 1.0).abs() < 1e-4, "L1 norm={}", sum_cont);
    }

    #[test]
    fn neural_layer_reduces_center_suppression() {
        let mut p1 = KernelParams::default();
        p1.w_neural = 0.0;
        let k1 = RoughnessKernel::new(p1, ERB_STEP);
        let c1 = k1.lut[k1.hw];

        let mut p2 = KernelParams::default();
        p2.w_neural = 0.4;
        let k2 = RoughnessKernel::new(p2, ERB_STEP);
        let c2 = k2.lut[k2.hw];

        assert!(c2 > c1);
    }

    #[test]
    fn kernel_stable_across_erbstep() {
        let p = KernelParams::default();
        let k1 = RoughnessKernel::new(p, ERB_STEP);
        let k2 = RoughnessKernel::new(p, 0.02);
        let g1 = &k1.lut;
        let g2 = &k2.lut;
        let hw1 = k1.hw;
        let hw2 = k2.hw;
        let pos_x1 = find_max_peak(g1, hw1, true)
            .expect("no positive-side peak")
            .0;
        let pos_x2 = find_max_peak(g2, hw2, true)
            .expect("no positive-side peak")
            .0;
        assert!(
            (pos_x1 - pos_x2).abs() < 0.2,
            "peak position drift too large: {:.3} vs {:.3}",
            pos_x1,
            pos_x2
        );
    }

    // ------------------------------------------------------------
    // δ-input and stability tests
    // ------------------------------------------------------------

    #[test]
    fn delta_input_reproduces_kernel_shape_when_du_constant() {
        let fs = 48_000.0;
        let nfft = 131072;
        let n = nfft / 2;
        let df = fs / nfft as f32;
        let f0 = 1000.0;
        let k0 = (f0 / df).round() as usize;

        let p = KernelParams::default();
        let k = RoughnessKernel::new(p, ERB_STEP);
        let g = &k.lut;
        let hw = k.hw;

        let mut e = vec![0.0f32; n];
        e[k0] = 1.0;

        let fi = k0 as f32 * df;
        let du_const = df / erb_bw_hz(fi).max(1e-12);

        let mut r = vec![0.0f32; n];
        for i in 0..n {
            if i == k0 {
                continue;
            }
            let fj = i as f32 * df;
            let d = hz_to_erb(fj) - hz_to_erb(fi);
            if d.abs() > p.half_width_erb {
                continue;
            }
            let w = eval_kernel_delta_erb(&p, d);
            r[i] = w * du_const;
        }

        let lo = k0.saturating_sub((p.half_width_erb / ERB_STEP).ceil() as usize);
        let hi = (k0 + (p.half_width_erb / ERB_STEP).ceil() as usize).min(n - 1);

        let mut r_on_erb = Vec::with_capacity(g.len());
        for kidx in 0..g.len() {
            let d_k = (kidx as i32 - hw as i32) as f32 * ERB_STEP;
            let f_target_erb = d_k + hz_to_erb(fi);
            let mut best_i = k0;
            let mut best_diff = f32::MAX;
            for i in lo..=hi {
                let fj = i as f32 * df;
                let erb_i = hz_to_erb(fj);
                let diff = (erb_i - f_target_erb).abs();
                if diff < best_diff {
                    best_diff = diff;
                    best_i = i;
                }
            }
            r_on_erb.push(r[best_i]);
        }

        let r_sum: f32 = r_on_erb.iter().sum();
        let mut g_adj = g.clone();
        g_adj[hw] = 0.0;
        let g_sum: f32 = g_adj.iter().sum();
        let r_norm: Vec<f32> = r_on_erb.iter().map(|&x| x / (r_sum + 1e-12)).collect();
        let g_norm: Vec<f32> = g_adj.iter().map(|&x| x / (g_sum + 1e-12)).collect();

        let mae: f32 = r_norm
            .iter()
            .zip(g_norm.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / g_norm.len() as f32;
        assert!(mae < 1e-2, "MAE={}", mae);
    }

    #[test]
    fn delta_input_reproduces_kernel_shape() {
        let params = KernelParams::default();
        let k = RoughnessKernel::new(params, ERB_STEP);
        let space = Log2Space::new(20.0, 8000.0, 500);

        let mut amps = vec![0.0f32; space.centers_hz.len()];
        let mid = amps.len() / 2;
        amps[mid] = 1.0;

        let (r_vec, _) = k.potential_r_from_log2_spectrum(&amps, &space);

        let g_ref = &k.lut;
        let hw = k.hw;
        let _d_erb_kernel: Vec<f32> = (-(hw as i32)..=hw as i32)
            .map(|i| i as f32 * ERB_STEP)
            .collect();

        let f0_erb = hz_to_erb(space.centers_hz[mid]);
        let d_erb_vec: Vec<f32> = space
            .centers_hz
            .iter()
            .map(|&f| hz_to_erb(f) - f0_erb)
            .collect();

        let mut g_adj = g_ref.clone();
        g_adj[hw] = 0.0;
        let g_norm: Vec<f32> = g_adj
            .iter()
            .map(|&v| v / g_adj.iter().sum::<f32>())
            .collect();
        let r_norm: Vec<f32> = r_vec
            .iter()
            .map(|&v| v / r_vec.iter().sum::<f32>())
            .collect();

        let mut total_err = 0.0;
        let mut count = 0;
        for (de, rv) in d_erb_vec.iter().zip(r_norm.iter()) {
            if de.abs() < params.half_width_erb {
                let k_idx = ((de / ERB_STEP) + hw as f32).round() as isize;
                if k_idx >= 0 && (k_idx as usize) < g_norm.len() {
                    total_err += (rv - g_norm[k_idx as usize]).abs();
                    count += 1;
                }
            }
        }
        let mae = total_err / (count as f32).max(1.0);
        assert!(mae < 1e-3, "MAE too large: {:.4}", mae);
    }

    #[test]
    fn delta_input_follows_kernel_asymmetry_direction() {
        let k = make_kernel();
        let space = Log2Space::new(20.0, 8000.0, 500);

        let mut amps = vec![0.0f32; space.centers_hz.len()];
        let mid = amps.len() / 2;
        amps[mid] = 1.0;

        let (r_vec, _) = k.potential_r_from_log2_spectrum(&amps, &space);
        let (erb, _du) = erb_grid(&space);
        let u0 = erb[mid];
        let d = 0.3f32;

        let idx_pos = nearest_erb_index(&erb, u0 + d);
        let idx_neg = nearest_erb_index(&erb, u0 - d);

        let r_pos = r_vec[idx_pos];
        let r_neg = r_vec[idx_neg];
        assert!(
            r_pos > r_neg,
            "expected R(u0 + {:.2}) > R(u0 - {:.2}), got {} <= {}",
            d,
            d,
            r_pos,
            r_neg
        );

        let w_pos = lut_interp(&k.lut, k.erb_step, k.hw, d);
        let w_neg = lut_interp(&k.lut, k.erb_step, k.hw, -d);
        assert!(
            w_pos > w_neg,
            "kernel asymmetry must hold at ±{:.2} ERB, got {} <= {}",
            d,
            w_pos,
            w_neg
        );

        let r_ratio = r_pos.max(1e-12) / r_neg.max(1e-12);
        let w_ratio = w_pos.max(1e-12) / w_neg.max(1e-12);
        let rel_err = ((r_ratio - w_ratio) / w_ratio.max(1e-12)).abs();
        assert!(
            rel_err < 0.3,
            "R/kernel ratio mismatch too large: r_ratio={}, w_ratio={}, rel_err={}",
            r_ratio,
            w_ratio,
            rel_err
        );
    }

    #[test]
    fn potential_r_peaks_matches_density_delta_input() {
        let k = make_kernel();
        let space = Log2Space::new(40.0, 8000.0, 256);
        let (erb, du) = erb_grid(&space);

        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut peaks = Vec::new();
        let mut density = vec![0.0f32; erb.len()];
        let mut used = vec![false; erb.len()];

        for _ in 0..12 {
            let mut idx = rng.random_range(0..erb.len());
            while used[idx] {
                idx = rng.random_range(0..erb.len());
            }
            used[idx] = true;
            let mass = rng.random_range(0.1f32..2.0f32);
            if du[idx] > 0.0 {
                density[idx] += mass / du[idx];
            }
            peaks.push(Peak {
                u_erb: erb[idx],
                mass,
                bin_idx: idx,
            });
        }

        let (r_density, _) = k.potential_r_from_log2_spectrum_density(&density, &space);
        let r_peaks = k.potential_r_from_peaks(&peaks, &space);

        for i in 0..r_density.len() {
            let diff = (r_density[i] - r_peaks[i]).abs();
            assert!(diff < 1e-4, "bin {} diff {}", i, diff);
        }
    }

    #[test]
    fn peak_extraction_conserves_cluster_mass() {
        let space = Log2Space::new(80.0, 8000.0, 128);
        let (erb, du) = erb_grid(&space);
        let mut density = vec![0.0f32; erb.len()];

        let center = erb.len() / 2;
        let sigma = 2.0f32;
        for i in 0..erb.len() {
            let x = (i as f32 - center as f32) / sigma;
            density[i] = (-0.5 * x * x).exp();
        }

        let total_mass = density::density_to_mass(&density, &du);
        let cfg = PeakExtractConfig {
            max_peaks: None,
            min_rel_db_power: -120.0,
            min_abs_power_density: None,
            min_prominence_db_power: 0.0,
            min_rel_mass_db_power: -70.0,
            min_mass_fraction: None,
            min_sep_erb: 0.2,
        };
        let peaks = extract_peaks_density(&density, &space, &cfg);

        assert_eq!(peaks.len(), 1);
        let diff = (peaks[0].mass - total_mass).abs();
        assert!(diff < 1e-4, "mass diff {}", diff);
    }

    #[test]
    fn potential_r_stable_across_fs() {
        let p = KernelParams::default();
        let k = RoughnessKernel::new(p, ERB_STEP);
        let base = 1000.0;

        let fs1 = 48000.0;
        let n1 = 4096;
        let sig1: Vec<f32> = (0..n1)
            .map(|i| (2.0 * std::f32::consts::PI * base * i as f32 / fs1).sin())
            .collect();
        let (_r1, rtot1) = k.potential_r_from_analytic(&hilbert(&sig1), fs1);

        let fs2 = 96000.0;
        let n2 = 8192;
        let sig2: Vec<f32> = (0..n2)
            .map(|i| (2.0 * std::f32::consts::PI * base * i as f32 / fs2).sin())
            .collect();
        let (_r2, rtot2) = k.potential_r_from_analytic(&hilbert(&sig2), fs2);

        let rel_err = ((rtot2 - rtot1) / rtot1.abs()).abs();
        assert!(rel_err < 0.001, "R_total rel_err={rel_err}");
    }

    // ------------------------------------------------------------
    // Plot tests (unchanged, ignore for normal runs)
    // ------------------------------------------------------------

    #[test]
    #[ignore]
    fn plot_kernel_shape_png() {
        ensure_plots_dir().expect("create target/plots");
        let k = make_kernel();
        let params = k.params;
        let erb_step = 0.02;
        let (g, _) = build_kernel_erbstep(&params, erb_step);
        let hw = (params.half_width_erb / erb_step).ceil() as i32;
        let d_erb: Vec<f32> = (-hw..=hw).map(|i| i as f32 * erb_step).collect();

        let out_path = Path::new("target/plots/it_roughness_kernel_shape.png");
        let root = BitMapBackend::new(out_path, (1600, 1000)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption("Asymmetric ERB-domain Kernel", ("sans-serif", 30))
            .margin(10)
            .build_cartesian_2d(
                d_erb[0]..d_erb[d_erb.len() - 1],
                0.0..g.iter().cloned().fold(0.0, f32::max) * 1.1,
            )
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("ΔERB")
            .y_desc("Amplitude")
            .draw()
            .unwrap();
        chart
            .draw_series(LineSeries::new(
                d_erb.iter().zip(g.iter()).map(|(&x, &y)| (x, y)),
                &BLUE,
            ))
            .unwrap();
        root.present().unwrap();
        assert!(File::open(out_path).is_ok());
    }

    #[test]
    #[ignore]
    fn compare_build_kernel_and_eval_kernel_shape() -> Result<(), Box<dyn std::error::Error>> {
        ensure_plots_dir()?;
        let params = KernelParams::default();
        let erb_step = 0.005;
        let k = RoughnessKernel::new(params, erb_step);
        let g_discrete = &k.lut;
        let hw = k.hw;

        let d_erb_vec: Vec<f32> = (-(hw as i32)..=hw as i32)
            .map(|i| i as f32 * erb_step)
            .collect();
        let g_eval: Vec<f32> = d_erb_vec
            .iter()
            .map(|&d| eval_kernel_delta_erb(&params, d))
            .collect();
        let sum1: f32 = g_discrete.iter().sum();
        let sum2: f32 = g_eval.iter().sum();
        let g1: Vec<f32> = g_discrete.iter().map(|&v| v / sum1).collect();
        let g2: Vec<f32> = g_eval.iter().map(|&v| v / sum2).collect();
        let mae = g1
            .iter()
            .zip(g2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / g1.len() as f32;
        assert!(mae < 1e-3, "MAE={}", mae);

        let out_path = "target/plots/it_roughness_kernel_build_vs_eval.png";
        let root = BitMapBackend::new(out_path, (1600, 1000)).into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .caption("build_kernel vs eval_kernel", ("sans-serif", 30))
            .margin(10)
            .build_cartesian_2d(
                -5.0f32..5.0f32,
                0.0f32..g1.iter().cloned().fold(0.0, f32::max) * 1.1,
            )
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("ΔERB")
            .y_desc("Amplitude")
            .draw()?;
        chart
            .draw_series(LineSeries::new(
                d_erb_vec.iter().zip(g1.iter()).map(|(&x, &y)| (x, y)),
                &BLUE,
            ))?
            .label("discrete")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));
        chart
            .draw_series(LineSeries::new(
                d_erb_vec.iter().zip(g2.iter()).map(|(&x, &y)| (x, y)),
                &RED,
            ))?
            .label("eval()")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));
        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
        root.present()?;
        Ok(())
    }

    #[test]
    #[ignore]
    fn plot_potential_r_from_signal_direct_erb() {
        ensure_plots_dir().expect("create target/plots");
        let fs = 48000.0;
        let k = RoughnessKernel::new(KernelParams::default(), 0.005);
        let base = 440.0;
        let n = 16384;

        let mut sig1 = vec![0.0f32; n];
        for i in 0..n {
            let t = i as f32 / fs;
            sig1[i] = (2.0 * std::f32::consts::PI * base * t).sin();
        }
        let (r1, _) = k.potential_r_from_analytic(&hilbert(&sig1), fs);

        let mut sig2 = vec![0.0f32; n];
        let f2 = base * 1.2;
        for i in 0..n {
            let t = i as f32 / fs;
            sig2[i] = (2.0 * std::f32::consts::PI * base * t).sin()
                + (2.0 * std::f32::consts::PI * f2 * t).sin();
        }
        let (r2, _) = k.potential_r_from_analytic(&hilbert(&sig2), fs);

        let df = fs / n as f32;
        let f0_erb = hz_to_erb(base);
        let x_erb: Vec<f32> = (0..r1.len())
            .map(|i| hz_to_erb(i as f32 * df) - f0_erb)
            .collect();

        let out_path = "target/plots/it_roughness_potential_r_signal_direct_erb.png";
        let root = BitMapBackend::new(out_path, (1600, 1000)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption("Potential R from Signal (ΔERB axis)", ("sans-serif", 30))
            .margin(10)
            .build_cartesian_2d(
                -5.0f32..5.0f32,
                0.0f32..r2.iter().cloned().fold(0.0, f32::max) * 1.1,
            )
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("ΔERB")
            .y_desc("R(f)")
            .draw()
            .unwrap();
        chart
            .draw_series(LineSeries::new(
                x_erb.iter().zip(r1.iter()).map(|(&x, &y)| (x, y)),
                &BLUE,
            ))
            .unwrap()
            .label("pure tone");
        chart
            .draw_series(LineSeries::new(
                x_erb.iter().zip(r2.iter()).map(|(&x, &y)| (x, y)),
                &RED,
            ))
            .unwrap()
            .label("two-tone ΔERB≈0.3");
        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()
            .unwrap();
        root.present().unwrap();
        assert!(File::open(out_path).is_ok());
    }

    #[test]
    #[ignore]
    fn plot_potential_r_from_log2_spectrum_delta_input() -> Result<(), Box<dyn std::error::Error>> {
        use crate::core::erb::hz_to_erb;
        use crate::core::log2space::Log2Space;
        use plotters::prelude::*;

        ensure_plots_dir()?;
        let k = RoughnessKernel::new(KernelParams::default(), 0.005);
        let space = Log2Space::new(20.0, 8000.0, 144);

        let mut amps = vec![0.0f32; space.centers_hz.len()];
        let mid = amps.len() / 2;
        amps[mid] = 1.0;

        let (r_vec, _) = k.potential_r_from_log2_spectrum(&amps, &space);

        let mid = amps.len() / 2;
        let _f0_erb = hz_to_erb(space.centers_hz[mid]);
        let erb_per_bin = hz_to_erb(space.centers_hz[mid + 1]) - hz_to_erb(space.centers_hz[mid]);
        let d_erb_r: Vec<f32> = (0..amps.len())
            .map(|i| (i as f32 - mid as f32) * erb_per_bin)
            .collect();

        let g_ref = &k.lut;
        let hw = k.hw;
        let d_erb_kernel: Vec<f32> = (-(hw as i32)..=hw as i32)
            .map(|i| i as f32 * k.erb_step)
            .collect();

        let r_norm: Vec<f32> = r_vec
            .iter()
            .map(|&v| v / r_vec.iter().cloned().fold(0.0, f32::max))
            .collect();
        let g_norm: Vec<f32> = g_ref
            .iter()
            .map(|&v| v / g_ref.iter().cloned().fold(0.0, f32::max))
            .collect();

        let out_path = "target/plots/it_roughness_potential_r_from_log2_spectrum_delta.png";
        let root = BitMapBackend::new(out_path, (1500, 1000)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Potential R from Log2 Spectrum (δ input)",
                ("sans-serif", 30),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(-5.0f32..5.0f32, 0.0f32..1.05f32)
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("ΔERB")
            .y_desc("Normalized Amplitude")
            .draw()?;
        chart
            .draw_series(LineSeries::new(
                d_erb_r.iter().zip(r_norm.iter()).map(|(&x, &y)| (x, y)),
                &BLUE,
            ))?
            .label("R(log2 input)")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));
        chart
            .draw_series(LineSeries::new(
                d_erb_kernel
                    .iter()
                    .zip(g_norm.iter())
                    .map(|(&x, &y)| (x, y)),
                &GREEN,
            ))?
            .label("Kernel g(ΔERB)")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &GREEN));
        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
        root.present()?;
        assert!(std::path::Path::new(out_path).exists());
        Ok(())
    }

    #[test]
    #[ignore]
    fn plot_potential_r_delta_input_all_methods() -> Result<(), Box<dyn std::error::Error>> {
        use crate::core::fft::hilbert;
        use crate::core::log2space::Log2Space;
        use rustfft::{FftPlanner, num_complex::Complex32};

        ensure_plots_dir()?;
        let fs = 48_000.0;
        let params = KernelParams::default();
        let k = RoughnessKernel::new(params, 0.005);
        let space = Log2Space::new(20.0, 8000.0, 144);
        let nfft = 163_84;

        let mut amps_log2 = vec![0.0f32; space.centers_hz.len()];
        let mid = amps_log2.len() / 2;
        amps_log2[mid] = 1.0;

        let (_r_log2, _) = k.potential_r_from_log2_spectrum(&amps_log2, &space);

        let df = fs / nfft as f32;
        let mut amps_lin = vec![0.0f32; nfft / 2];
        for (kidx, &f) in space.centers_hz.iter().enumerate() {
            let bin = (f / df).round() as usize;
            if bin < amps_lin.len() {
                amps_lin[bin] += amps_log2[kidx];
            }
        }
        let (_r_spec, _) = k.potential_r_from_spectrum(&amps_lin, fs);

        let mut buf = vec![Complex32::new(0.0, 0.0); nfft];
        for (i, &amp) in amps_lin.iter().enumerate() {
            buf[i] = Complex32::new(amp, 0.0);
        }
        for i in 1..(nfft / 2) {
            buf[nfft - i] = buf[i].conj();
        }
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(nfft);
        ifft.process(&mut buf);
        let sig: Vec<f32> = buf.iter().map(|z| z.re / nfft as f32).collect();
        let (_r_analytic, _) = k.potential_r_from_analytic(&hilbert(&sig), fs);

        // plotting same as original (omitted for brevity)
        Ok(())
    }
}
