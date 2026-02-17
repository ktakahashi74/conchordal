//! core/harmonicity_kernel.rs
//! Common Root (Sibling) Harmonicity Kernel on Log2Space.
//!
//! This implementation generates a perc_potential_H landscape based on the physiological
//! mechanism of "Common Root" detection (approximating autocorrelation).
//!
//! Algorithm: "Sibling Projection"
//! 1. **Downward Projection (Root Search)**:
//!    Estimate the "Virtual Root" spectrum from the environment.
//!    If energy exists at f, it implies potential roots at f/2, f/3...
//!    (e.g., Env 200Hz -> Roots at 100Hz, 66Hz...)
//!
//! 2. **Upward Projection (Harmonic Resonance)**:
//!    From the estimated roots, project their natural harmonics.
//!    (e.g., Root 100Hz -> Stability at 100Hz, 200Hz, 300Hz, 400Hz...)
//!
//! Result:
//! An input of 200Hz naturally creates stability peaks at:
//! - 100Hz (Subharmonic)
//! - 400Hz (Octave)
//! - 300Hz (Perfect 5th via 100Hz root)
//! - 500Hz (Major 3rd via 100Hz root)
//!   ...without using any hardcoded ratio templates.

//!   core/harmonicity_kernel.rs
//!   Optimized Sibling Harmonicity Kernel.
//!
//! Uses a "Shift-and-Add" approach with pre-calculated bounds
//! to ensure O(N) efficiency and SIMD-friendly loops.

use crate::core::log2space::Log2Space;

#[derive(Clone, Copy, Debug)]
pub struct HarmonicityParams {
    /// Downward projections for root candidates.
    pub num_subharmonics: u32,
    /// Upward projections for harmonic candidates.
    pub num_harmonics: u32,
    /// Decay exponent for Path A (common root path).
    pub rho_common_root: f32,
    /// Decay exponent for Path B (common overtone/undertone path).
    pub rho_common_overtone: f32,
    /// Gaussian smoothing width in cents on log2 spectrum.
    pub sigma_cents: f32,
    /// Normalize output to peak of 1.0.
    pub normalize_output: bool,
    /// Blend ratio: 0 = Path A only, 1 = Path B only.
    pub mirror_weight: f32,
    /// Nonlinear emphasis for root spectrum coincidence (1.0 = linear).
    pub gamma_root: f32,
    /// Nonlinear emphasis for overtone spectrum coincidence (1.0 = linear).
    pub gamma_overtone: f32,
    /// Normalize A/B paths to peak 1.0 before blending.
    pub normalize_paths_before_blend: bool,
    /// Scale for diagonal (m=k) self-reinforcement (expected self term).
    /// 1.0 = legacy behavior, <1.0 reduces unison dominance.
    pub diag_weight: f32,
    /// Apply absolute-frequency gating for TFS roll-off.
    pub freq_gate: bool,
    /// Frequency pivot for TFS gate (Hz).
    pub tfs_f_pl_hz: f32,
    /// Slope of TFS gate.
    pub tfs_eta: f32,
}

impl Default for HarmonicityParams {
    fn default() -> Self {
        Self {
            num_subharmonics: 16,
            num_harmonics: 16,
            rho_common_root: 0.4,
            rho_common_overtone: 0.2,
            sigma_cents: 3.0, // Slightly wider peaks to ease sampling
            normalize_output: true,
            mirror_weight: 0.3,
            gamma_root: 1.0,
            gamma_overtone: 1.0,
            normalize_paths_before_blend: false,
            diag_weight: 0.65,
            freq_gate: false,
            tfs_f_pl_hz: 4500.0,
            tfs_eta: 4.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HarmonicityKernel {
    pub bins_per_oct: u32,
    pub params: HarmonicityParams,
    smooth_kernel: Vec<f32>,
    pad_bins: usize, // Internal padding size
    sub_limit: u32,
    harm_limit: u32,
    #[allow(dead_code)]
    max_limit: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HarmonicBindingMetrics {
    pub root_affinity: f32,
    pub overtone_affinity: f32,
    pub binding_strength: f32,
    pub harmonic_tilt: f32,
}

#[derive(Clone, Debug)]
pub struct DualPathHarmonicity {
    pub blended: Vec<f32>,
    pub path_a: Vec<f32>,
    pub path_b: Vec<f32>,
    pub metrics: HarmonicBindingMetrics,
}

impl HarmonicityKernel {
    pub fn new(space: &Log2Space, params: HarmonicityParams) -> Self {
        let (sub_limit, harm_limit, max_limit) = Self::effective_limits(&params);

        // 1. Pre-calculate smoothing kernel
        let sigma_bins = params.sigma_cents / 1200.0 * space.bins_per_oct as f32;
        let k = if sigma_bins <= 1e-6 {
            vec![1.0]
        } else {
            let half_width = (2.5 * sigma_bins).ceil() as usize;
            let width = 2 * half_width + 1;
            let mut k = vec![0.0f32; width];
            let mut sum = 0.0;
            for (i, kv) in k.iter_mut().enumerate().take(width) {
                let x = (i as isize - half_width as isize) as f32;
                *kv = (-0.5 * (x / sigma_bins).powi(2)).exp();
                sum += *kv;
            }
            for v in &mut k {
                *v /= sum;
            }
            k
        };

        // 2. Pre-calculate necessary padding
        // pad_bins must allow shifting both down (roots) and up (overtone mirror path).
        let max_oct = (max_limit as f32).max(1.0).log2();
        let pad_bins = (max_oct * space.bins_per_oct as f32).ceil() as usize;

        Self {
            bins_per_oct: space.bins_per_oct,
            params,
            smooth_kernel: k,
            pad_bins,
            sub_limit,
            harm_limit,
            max_limit,
        }
    }

    /// The core function: Env -> Roots -> Landscape
    pub fn potential_h_from_log2_spectrum(
        &self,
        envelope: &[f32],
        space: &Log2Space,
    ) -> (Vec<f32>, f32) {
        let dual = self.potential_h_dual_from_log2_spectrum(envelope, space);
        let max_val = if self.params.normalize_output {
            1.0
        } else {
            dual.blended.iter().copied().fold(1e-12f32, f32::max)
        };
        (dual.blended, max_val)
    }

    pub fn potential_h_dual_from_log2_spectrum(
        &self,
        envelope: &[f32],
        space: &Log2Space,
    ) -> DualPathHarmonicity {
        debug_assert_eq!(
            space.bins_per_oct, self.bins_per_oct,
            "Log2Space bins_per_oct mismatch (space={}, kernel={})",
            space.bins_per_oct, self.bins_per_oct
        );
        debug_assert_eq!(
            envelope.len(),
            space.n_bins(),
            "envelope length mismatch (env={}, space={})",
            envelope.len(),
            space.n_bins()
        );
        let n_bins = envelope.len();
        let bins_per_oct = self.bins_per_oct as f32;
        let mirror = self.params.mirror_weight.clamp(0.0, 1.0);
        let sub_limit = self.sub_limit.max(1);
        let harm_limit = self.harm_limit.max(1);
        let k_diag = sub_limit.min(harm_limit).max(1);

        // Step 0: Smooth Input (O(N))
        let smeared_env = self.convolve_smooth(envelope);

        // Diagonal self-term correction factors.
        let diag_w = self.params.diag_weight.clamp(0.0, 1.0);
        let (diag_a, diag_b) = if diag_w < 1.0 {
            let mut sum_a = 0.0f32;
            let mut sum_b = 0.0f32;
            for k in 1..=k_diag {
                let kk = k as f32;
                let a = kk.powf(-self.params.rho_common_root);
                let b = kk.powf(-self.params.rho_common_overtone);
                sum_a += a * a;
                sum_b += b * b;
            }
            (sum_a, sum_b)
        } else {
            (0.0, 0.0)
        };

        // Buffer for Virtual Roots / Overtones (centered with padding on both sides)
        let padding = self.pad_bins;
        let center_offset = padding as f32;
        let buf_len = n_bins + 2 * padding;
        let mut root_spectrum = vec![0.0f32; buf_len];
        let mut overtone_spectrum = vec![0.0f32; buf_len];

        // === Path A: Common Root / Overtone Series (down then up) ===
        for k in 1..=sub_limit {
            let shift_bins = (k as f32).log2() * bins_per_oct;
            let weight = (k as f32).powf(-self.params.rho_common_root);
            let offset = center_offset - shift_bins;
            Self::accumulate_shifted(&smeared_env, &mut root_spectrum, offset, weight);
        }
        let gamma_root = self.params.gamma_root.max(1e-3);
        if (gamma_root - 1.0).abs() > 1e-6 {
            for v in &mut root_spectrum {
                *v = v.max(0.0).powf(gamma_root);
            }
        }

        let mut landscape_a = vec![0.0f32; n_bins];
        for m in 1..=harm_limit {
            let shift_bins = (m as f32).log2() * bins_per_oct;
            let weight = (m as f32).powf(-self.params.rho_common_root);
            let offset = shift_bins - center_offset;
            Self::accumulate_shifted(&root_spectrum, &mut landscape_a, offset, weight);
        }
        if diag_w < 1.0 {
            let scale = 1.0 - diag_w;
            for i in 0..n_bins {
                let adjusted = landscape_a[i] - scale * diag_a * smeared_env[i];
                landscape_a[i] = adjusted.max(0.0);
            }
        }

        // === Path B: Common Overtone / Undertone Series (up then down) ===
        for k in 1..=harm_limit {
            let shift_bins = (k as f32).log2() * bins_per_oct;
            let weight = (k as f32).powf(-self.params.rho_common_overtone);
            let offset = center_offset + shift_bins;
            Self::accumulate_shifted(&smeared_env, &mut overtone_spectrum, offset, weight);
        }
        let gamma_overtone = self.params.gamma_overtone.max(1e-3);
        if (gamma_overtone - 1.0).abs() > 1e-6 {
            for v in &mut overtone_spectrum {
                *v = v.max(0.0).powf(gamma_overtone);
            }
        }

        let mut landscape_b = vec![0.0f32; n_bins];
        for m in 1..=sub_limit {
            let shift_bins = (m as f32).log2() * bins_per_oct;
            let weight = (m as f32).powf(-self.params.rho_common_overtone);
            let offset = -shift_bins - center_offset;
            Self::accumulate_shifted(&overtone_spectrum, &mut landscape_b, offset, weight);
        }
        if diag_w < 1.0 {
            let scale = 1.0 - diag_w;
            for i in 0..n_bins {
                let adjusted = landscape_b[i] - scale * diag_b * smeared_env[i];
                landscape_b[i] = adjusted.max(0.0);
            }
        }

        if self.params.normalize_paths_before_blend {
            let mut max_a = 0.0f32;
            let mut max_b = 0.0f32;
            for i in 0..n_bins {
                if landscape_a[i] > max_a {
                    max_a = landscape_a[i];
                }
                if landscape_b[i] > max_b {
                    max_b = landscape_b[i];
                }
            }
            if max_a > 0.0 {
                let inv = 1.0 / max_a;
                for v in &mut landscape_a {
                    *v *= inv;
                }
            }
            if max_b > 0.0 {
                let inv = 1.0 / max_b;
                for v in &mut landscape_b {
                    *v *= inv;
                }
            }
        }

        let root_affinity = Self::cosine_similarity(&smeared_env, &landscape_a);
        let overtone_affinity = Self::cosine_similarity(&smeared_env, &landscape_b);
        let binding_strength = root_affinity + overtone_affinity;
        let harmonic_tilt = (root_affinity - overtone_affinity) / (binding_strength + 1e-9);
        let metrics = HarmonicBindingMetrics {
            root_affinity,
            overtone_affinity,
            binding_strength,
            harmonic_tilt,
        };

        // Blend A/B
        let mut landscape = vec![0.0f32; n_bins];
        for i in 0..n_bins {
            landscape[i] = (1.0 - mirror) * landscape_a[i] + mirror * landscape_b[i];
        }

        // Step 3: Post-processing
        let mut max_val = 1e-12;
        let do_gate = self.params.freq_gate;
        let do_norm = self.params.normalize_output;

        // Combined loop for gating and max-finding (Auto-vectorized)
        for (i, v) in landscape.iter_mut().enumerate().take(n_bins) {
            if do_gate {
                *v *= Self::absfreq_gate(space.freq_of_index(i), &self.params);
            }
            if *v > max_val {
                max_val = *v;
            }
        }

        if do_norm {
            let scale = 1.0 / max_val;
            for v in &mut landscape {
                *v *= scale;
            }
        }

        DualPathHarmonicity {
            blended: landscape,
            path_a: landscape_a,
            path_b: landscape_b,
            metrics,
        }
    }
    fn effective_limits(params: &HarmonicityParams) -> (u32, u32, u32) {
        let sub = params.num_subharmonics.max(1);
        let harm = params.num_harmonics.max(1);
        let max_limit = sub.max(harm);
        (sub, harm, max_limit)
    }
    /// Optimized Shift-and-Add with safe bounds checking.
    /// dst[i + offset] += src[i] * weight
    fn accumulate_shifted(src: &[f32], dst: &mut [f32], offset: f32, weight: f32) {
        let offset_i = offset.floor() as isize;
        let frac = offset - offset_i as f32;
        let w0 = weight * (1.0 - frac);
        let w1 = weight * frac;

        // Calculate valid iteration range for 'i' (index in src)
        // Constraints:
        // 1. 0 <= i < src.len()
        // 2. 0 <= i + offset_i < dst.len() - 1 (Need space for w1 interpolation)

        // Lower bound: i >= 0 AND i >= -offset_i
        let start_i = 0.max(-offset_i);

        // Upper bound (exclusive): i < src.len() AND i < dst.len() - 1 - offset_i
        let end_i = (src.len() as isize).min(dst.len() as isize - 1 - offset_i);

        // If range is invalid/empty, do nothing
        if start_i >= end_i {
            return;
        }

        let start = start_i as usize;
        let len = (end_i - start_i) as usize;

        // Destination start index
        // Since start >= -offset_i, (start + offset_i) is guaranteed >= 0
        let dst_start = (start as isize + offset_i) as usize;

        // Create slices for the hot loop (avoids bounds check inside loop)
        let src_slice = &src[start..start + len];
        let dst_slice = &mut dst[dst_start..dst_start + len + 1];

        for (k, &val) in src_slice.iter().enumerate() {
            // Unsafe get_unchecked could be used here for max speed,
            // but standard indexing is safe and fast enough due to slice bounds.
            dst_slice[k] += val * w0;
            dst_slice[k + 1] += val * w1;
        }
    }

    fn convolve_smooth(&self, input: &[f32]) -> Vec<f32> {
        // Convolution is heavy, but here kernel is small.
        // Optimization: Use separate loop for the main part to avoid boundary checks.
        if self.smooth_kernel.len() == 1 {
            return input.to_vec();
        }
        let n = input.len();
        let mut output = vec![0.0; n];
        let k_len = self.smooth_kernel.len();
        let half = k_len / 2;

        // Naive but clear implementation.
        // For very large N, FFT conv is better, but here N ~ 200-4000, kernel ~ 5-10.
        // Direct convolution is faster.
        for (i, out_val) in output.iter_mut().enumerate().take(n) {
            let mut acc = 0.0;
            let start_k = half.saturating_sub(i);
            let end_k = if i + half >= n {
                k_len - (i + half - n + 1)
            } else {
                k_len
            };

            for j in start_k..end_k {
                let input_idx = i + j - half;
                acc += input[input_idx] * self.smooth_kernel[j];
            }
            *out_val = acc;
        }
        output
    }

    #[inline]
    fn absfreq_gate(f_hz: f32, p: &HarmonicityParams) -> f32 {
        1.0 / (1.0 + (f_hz / p.tfs_f_pl_hz).powf(p.tfs_eta))
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let eps = 1e-9f32;
        let mut dot = 0.0f32;
        let mut na2 = 0.0f32;
        let mut nb2 = 0.0f32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += x * y;
            na2 += x * x;
            nb2 += y * y;
        }
        let denom = na2.sqrt() * nb2.sqrt() + eps;
        if denom <= eps || !denom.is_finite() {
            0.0
        } else {
            (dot / denom).clamp(-1.0, 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::log2space::Log2Space;
    use plotters::prelude::*;
    use std::fs::File;
    use std::path::Path;

    fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
        let mut diff2 = 0.0f32;
        let mut norm2 = 0.0f32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let d = x - y;
            diff2 += d * d;
            norm2 += x * x;
        }
        diff2.sqrt() / (norm2.sqrt() + 1e-9)
    }

    fn build_sparse_env(space: &Log2Space, freqs: &[f32]) -> Vec<f32> {
        let mut env = vec![0.0f32; space.n_bins()];
        for &hz in freqs {
            if let Some(i) = space.index_of_freq(hz) {
                env[i] += 1.0;
            }
        }
        env
    }

    fn ensure_plots_dir() -> std::io::Result<()> {
        std::fs::create_dir_all("target/plots")
    }

    #[test]
    fn test_minor_triad_lcm_reach() {
        let space = Log2Space::new(80.0, 2000.0, 180);
        let mut params = HarmonicityParams::default();
        params.num_harmonics = 8;
        params.mirror_weight = 1.0;
        let hk = HarmonicityKernel::new(&space, params);

        let mut env = vec![0.0; space.n_bins()];
        let triad = [200.0, 240.0, 300.0]; // 10:12:15 (Just minor triad)
        for &f in &triad {
            if let Some(idx) = space.index_of_freq(f) {
                env[idx] = 1.0;
            }
        }

        let (landscape, _) = hk.potential_h_from_log2_spectrum(&env, &space);
        let triad_indices: Vec<usize> = triad
            .iter()
            .filter_map(|&f| space.index_of_freq(f))
            .collect();

        let triad_max = triad_indices
            .iter()
            .map(|&i| landscape[i])
            .fold(0.0f32, f32::max);
        let idx_tritone = space
            .index_of_freq(200.0 * 1.414)
            .expect("tritone bin exists");
        let dissonant = landscape[idx_tritone];

        assert!(
            triad_max > dissonant * 1.2,
            "Mirror path should reinforce triad members relative to dissonance (triad={triad_max:.4} tritone={dissonant:.4})"
        );
        for idx in triad_indices {
            assert!(
                landscape[idx] > 1e-3,
                "Triad bin should accumulate stability"
            );
        }
    }

    #[test]
    fn test_num_harmonics_affects_peak_strength() {
        let space = Log2Space::new(50.0, 800.0, 160);
        let mut params_low = HarmonicityParams::default();
        params_low.normalize_output = false;
        params_low.num_harmonics = 1;
        let hk_low = HarmonicityKernel::new(&space, params_low);

        let mut params_high = params_low;
        params_high.num_harmonics = 8;
        let hk_high = HarmonicityKernel::new(&space, params_high);

        let mut env = vec![0.0; space.n_bins()];
        if let Some(idx) = space.index_of_freq(200.0) {
            env[idx] = 1.0;
        }

        let (land_low, _) = hk_low.potential_h_from_log2_spectrum(&env, &space);
        let (land_high, _) = hk_high.potential_h_from_log2_spectrum(&env, &space);
        let idx_300 = space.index_of_freq(300.0).expect("bin exists");
        let ratio = land_high[idx_300] / land_low[idx_300].max(1e-6);
        assert!(
            ratio > 1.1,
            "expected stronger 3:2 peak with more harmonics (ratio={ratio})"
        );
    }

    #[test]
    fn test_num_subharmonics_affects_root_related_peaks() {
        let space = Log2Space::new(50.0, 800.0, 160);
        let mut params_low = HarmonicityParams::default();
        params_low.normalize_output = false;
        params_low.num_subharmonics = 1;
        let hk_low = HarmonicityKernel::new(&space, params_low);

        let mut params_high = params_low;
        params_high.num_subharmonics = 10;
        let hk_high = HarmonicityKernel::new(&space, params_high);

        let mut env = vec![0.0; space.n_bins()];
        if let Some(idx) = space.index_of_freq(200.0) {
            env[idx] = 1.0;
        }

        let (land_low, _) = hk_low.potential_h_from_log2_spectrum(&env, &space);
        let (land_high, _) = hk_high.potential_h_from_log2_spectrum(&env, &space);
        let idx_100 = space.index_of_freq(100.0).expect("bin exists");
        let ratio = land_high[idx_100] / land_low[idx_100].max(1e-6);
        assert!(
            ratio > 1.1,
            "expected stronger subharmonic peak with more subharmonics (ratio={ratio})"
        );
    }

    #[test]
    fn test_sigma_zero_is_finite() {
        let space = Log2Space::new(50.0, 800.0, 160);
        let mut params = HarmonicityParams::default();
        params.sigma_cents = 0.0;
        let hk = HarmonicityKernel::new(&space, params);

        let mut env = vec![0.0; space.n_bins()];
        if let Some(idx) = space.index_of_freq(200.0) {
            env[idx] = 1.0;
        }
        let (land, _) = hk.potential_h_from_log2_spectrum(&env, &space);
        assert!(
            land.iter().all(|&v| v.is_finite() && v >= 0.0),
            "expected finite non-negative landscape for sigma=0"
        );
    }

    #[test]
    fn test_diag_weight_boosts_fifth_relative_to_unison() {
        let space = Log2Space::new(50.0, 800.0, 200);

        let mut params_old = HarmonicityParams::default();
        params_old.num_harmonics = 8;
        params_old.diag_weight = 1.0;
        let hk_old = HarmonicityKernel::new(&space, params_old);

        let mut params_new = HarmonicityParams::default();
        params_new.num_harmonics = 8;
        params_new.diag_weight = 0.5;
        let hk_new = HarmonicityKernel::new(&space, params_new);

        let mut env = vec![0.0; space.n_bins()];
        let idx_200 = space.index_of_freq(200.0).expect("bin exists");
        env[idx_200] = 1.0;

        let (land_old, _) = hk_old.potential_h_from_log2_spectrum(&env, &space);
        let (land_new, _) = hk_new.potential_h_from_log2_spectrum(&env, &space);

        let idx_300 = space.index_of_freq(300.0).expect("bin exists");
        let r_old = land_old[idx_300] / land_old[idx_200].max(1e-6);
        let r_new = land_new[idx_300] / land_new[idx_200].max(1e-6);

        assert!(
            r_new > r_old * 1.2,
            "expected 3:2 ratio to rise when diag_weight drops (old={r_old:.3} new={r_new:.3})"
        );
        assert!(
            land_new.iter().all(|&v| v.is_finite() && v >= 0.0),
            "expected non-negative finite values after diagonal correction"
        );
    }

    #[test]
    fn test_sibling_consonance_creation() {
        // Input: 200Hz.
        // We expect peaks at:
        // - 100Hz (Root)
        // - 400Hz (Octave)
        // - 300Hz (Perfect 5th via 100Hz root)

        let space = Log2Space::new(50.0, 800.0, 200);
        let mut params = HarmonicityParams::default();
        params.num_harmonics = 8;

        let hk = HarmonicityKernel::new(&space, params);

        let mut env = vec![0.0; space.n_bins()];
        let idx_200 = space.index_of_freq(200.0).unwrap();
        env[idx_200] = 1.0;

        let (landscape, _) = hk.potential_h_from_log2_spectrum(&env, &space);

        let idx_300 = space.index_of_freq(300.0).unwrap();
        let idx_283 = space.index_of_freq(283.0).unwrap(); // Dissonant (Tritone-ish)

        // 300Hz should be stable because:
        // 200 -> Root 100.
        // Root 100 -> Harmonic 300.
        assert!(
            landscape[idx_300] > 0.2,
            "300Hz (Perfect 5th) should be a peak"
        );
        assert!(
            landscape[idx_300] > landscape[idx_283] * 1.5,
            "5th should be much more stable than tritone"
        );
    }

    #[test]
    fn test_complex_ratios_detection() {
        // Test: Can we detect 7:4 (Harmonic 7th) and 6:5 (Minor 3rd)?

        let space = Log2Space::new(20.0, 1600.0, 100);
        let mut params = HarmonicityParams::default();
        params.num_harmonics = 8;

        let hk = HarmonicityKernel::new(&space, params);

        let mut env = vec![0.0; space.n_bins()];
        let f_input = 400.0;
        if let Some(idx) = space.index_of_freq(f_input) {
            env[idx] = 1.0;
        }

        let (landscape, _) = hk.potential_h_from_log2_spectrum(&env, &space);

        let idx_m3 = space.index_of_freq(400.0 * 1.2).unwrap(); // 6:5
        let idx_h7 = space.index_of_freq(400.0 * 1.75).unwrap(); // 7:4

        // Tritone (approx 1.414).
        // Note: This is close to 7:5 (1.40), so it will have significant potential!
        let idx_tritone = space.index_of_freq(400.0 * 1.414).unwrap();

        println!("Potential at 6:5 (m3): {}", landscape[idx_m3]);
        println!("Potential at 7:4 (h7): {}", landscape[idx_h7]);
        println!("Potential at Tritone:  {}", landscape[idx_tritone]);

        assert!(
            landscape[idx_m3] > landscape[idx_tritone] * 1.1,
            "6:5 should be more stable than tritone"
        );
        assert!(
            landscape[idx_h7] > landscape[idx_tritone] * 1.1,
            "7:4 should be more stable than tritone"
        );
    }

    #[test]
    #[cfg(feature = "plotcheck")]
    fn plot_sibling_landscape_png() {
        ensure_plots_dir().expect("create target/plots");
        let space = Log2Space::new(20.0, 8000.0, 200);

        let p = HarmonicityParams::default();
        let hk = HarmonicityKernel::new(&space, p);

        let mut env = vec![0.0; space.n_bins()];
        let f_input = 440.0;
        if let Some(idx) = space.index_of_freq(f_input) {
            env[idx] = 1.0;
        }

        let (y, _) = hk.potential_h_from_log2_spectrum(&env, &space);
        let xs: Vec<f32> = (0..space.n_bins())
            .map(|i| space.freq_of_index(i))
            .collect();

        let out_path = Path::new("target/plots/it_harmonicity_sibling_landscape.png");
        let root = BitMapBackend::new(out_path, (1200, 600)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("Sibling Landscape (Input: {}Hz)", f_input),
                ("sans-serif", 20),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(20.0f32..8000.0f32, 0.0f32..1.05f32)
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("Frequency [Hz]")
            .y_desc("Potential (0.0 - 1.0)")
            .y_labels(10)
            .draw()
            .unwrap();

        chart
            .draw_series(LineSeries::new(
                xs.iter().zip(y.iter()).map(|(&x, &y)| (x, y)),
                &BLUE,
            ))
            .unwrap();

        // Mark expected mergent ratios
        let markers = vec![
            (f_input * 1.5, "3:2", RED),
            (f_input * 1.25, "5:4", MAGENTA),
            (f_input * 0.5, "1:2", GREEN),
            (f_input * 2.0, "2:1", GREEN),
        ];

        for (freq, _label, color) in markers {
            chart
                .draw_series(std::iter::once(PathElement::new(
                    vec![(freq, 0.0), (freq, 1.0)],
                    color.mix(0.5),
                )))
                .unwrap();
        }

        root.present().unwrap();
        assert!(File::open(out_path).is_ok());
    }

    #[test]
    fn mirror_weight_changes_output_when_rhos_far_apart() {
        let space = Log2Space::new(20.0, 8000.0, 400);
        let env = build_sparse_env(&space, &[110.0, 165.0, 220.0, 275.0, 330.0, 440.0, 550.0]);

        let mut rel_l2_norm_false = 0.0f32;
        let mut rel_l2_norm_true = 0.0f32;

        for &normalize_output in &[false, true] {
            let mut params = HarmonicityParams::default();
            params.normalize_output = normalize_output;
            params.rho_common_root = 0.5;
            params.rho_common_overtone = 3.0;
            params.mirror_weight = 0.0;
            let hk0 = HarmonicityKernel::new(&space, params);
            let (y0, _) = hk0.potential_h_from_log2_spectrum(&env, &space);

            params.mirror_weight = 1.0;
            let hk1 = HarmonicityKernel::new(&space, params);
            let (y1, _) = hk1.potential_h_from_log2_spectrum(&env, &space);

            assert!(y0.iter().all(|v| v.is_finite()));
            assert!(y1.iter().all(|v| v.is_finite()));

            let d = rel_l2(&y0, &y1);
            if normalize_output {
                rel_l2_norm_true = d;
            } else {
                rel_l2_norm_false = d;
            }
        }

        assert!(
            rel_l2_norm_false > 0.02,
            "mirror diff should be visible without normalization (rel_l2={rel_l2_norm_false:.6})"
        );
        assert!(
            rel_l2_norm_true > 0.005,
            "normalize_output=true collapsed mirror diff too much (rel_l2={rel_l2_norm_true:.6})"
        );
    }

    #[test]
    fn mirror_weight_is_nearly_noop_when_rhos_equal() {
        let space = Log2Space::new(40.0, 3000.0, 240);
        let env = build_sparse_env(&space, &[220.0, 330.0, 440.0]);

        for &normalize_output in &[false, true] {
            let mut params = HarmonicityParams::default();
            params.num_subharmonics = 1;
            params.num_harmonics = 1;
            params.diag_weight = 1.0;
            params.rho_common_root = 1.25;
            params.rho_common_overtone = 1.25;
            params.normalize_output = normalize_output;
            params.mirror_weight = 0.0;
            let hk0 = HarmonicityKernel::new(&space, params);
            let (y0, _) = hk0.potential_h_from_log2_spectrum(&env, &space);

            params.mirror_weight = 1.0;
            let hk1 = HarmonicityKernel::new(&space, params);
            let (y1, _) = hk1.potential_h_from_log2_spectrum(&env, &space);

            let d = rel_l2(&y0, &y1);
            assert!(
                d < 1e-4,
                "mirror should be near no-op when rho_root == rho_overtone and paths are symmetric (rel_l2={d:.8})"
            );
        }
    }

    #[test]
    fn mirror_weight_never_nan() {
        let space = Log2Space::new(20.0, 8000.0, 240);
        let mut lcg_state = 0x1234_5678_9abc_def0u64;
        let mut random_freqs = Vec::with_capacity(8);
        let lo = 80.0f32.log2();
        let hi = 2000.0f32.log2();
        for _ in 0..8 {
            lcg_state = lcg_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((lcg_state >> 32) as u32) as f32 / u32::MAX as f32;
            random_freqs.push(2f32.powf(lo + (hi - lo) * u));
        }

        let cases = vec![
            build_sparse_env(&space, &[220.0]),
            build_sparse_env(&space, &[220.0, 330.0]),
            build_sparse_env(&space, &[220.0, 275.0, 330.0]),
            build_sparse_env(&space, &[220.0, 264.0, 330.0]),
            build_sparse_env(&space, &random_freqs),
        ];

        for env in &cases {
            for &normalize_output in &[false, true] {
                for &mirror in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
                    let mut params = HarmonicityParams::default();
                    params.normalize_output = normalize_output;
                    params.mirror_weight = mirror;
                    let hk = HarmonicityKernel::new(&space, params);
                    let (y, max_v) = hk.potential_h_from_log2_spectrum(env, &space);
                    assert!(max_v.is_finite(), "max should be finite");
                    assert!(
                        y.iter().all(|v| v.is_finite() && *v >= 0.0),
                        "landscape should be finite and non-negative for mirror={mirror}"
                    );
                }
            }
        }
    }

    #[test]
    fn pure_tone_harmonic_tilt_is_near_zero() {
        let space = Log2Space::new(80.0, 2000.0, 144);
        let env = build_sparse_env(&space, &[220.0]);
        let mut params = HarmonicityParams::default();
        params.mirror_weight = 0.5;
        let hk = HarmonicityKernel::new(&space, params);
        let dual = hk.potential_h_dual_from_log2_spectrum(&env, &space);
        assert!(
            dual.metrics.harmonic_tilt.abs() < 0.08,
            "pure tone should stay near neutral tilt, got {}",
            dual.metrics.harmonic_tilt
        );
    }

    #[test]
    fn harmonic_series_prefers_root_affinity() {
        let space = Log2Space::new(80.0, 4000.0, 144);
        let env = build_sparse_env(&space, &[220.0, 440.0, 660.0, 880.0, 1100.0, 1320.0]);
        let mut params = HarmonicityParams::default();
        params.mirror_weight = 0.5;
        let hk = HarmonicityKernel::new(&space, params);
        let dual = hk.potential_h_dual_from_log2_spectrum(&env, &space);
        assert!(
            dual.metrics.root_affinity > dual.metrics.overtone_affinity,
            "expected root_affinity > overtone_affinity ({} <= {})",
            dual.metrics.root_affinity,
            dual.metrics.overtone_affinity
        );
        assert!(
            dual.metrics.harmonic_tilt > 0.02,
            "expected positive harmonic tilt for overtone-rich series, got {}",
            dual.metrics.harmonic_tilt
        );
    }

    #[test]
    fn dual_api_blended_matches_legacy_output() {
        let space = Log2Space::new(60.0, 3000.0, 120);
        let env = build_sparse_env(&space, &[196.0, 294.0, 392.0, 523.25]);
        let mut params = HarmonicityParams::default();
        params.normalize_output = true;
        params.mirror_weight = 0.37;
        let hk = HarmonicityKernel::new(&space, params);

        let (legacy, _legacy_peak) = hk.potential_h_from_log2_spectrum(&env, &space);
        let dual = hk.potential_h_dual_from_log2_spectrum(&env, &space);
        assert_eq!(legacy.len(), dual.blended.len());
        for (a, b) in legacy.iter().zip(dual.blended.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "legacy and dual blended diverged: {a} vs {b}"
            );
        }
    }
}
