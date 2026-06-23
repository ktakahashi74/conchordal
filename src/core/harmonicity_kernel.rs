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
    /// Decay exponent for the common-root path.
    pub rho_common_root: f32,
    /// Gaussian smoothing width in cents on log2 spectrum.
    pub sigma_cents: f32,
    /// Normalize output to peak of 1.0.
    pub normalize_output: bool,
    /// Nonlinear emphasis for root spectrum coincidence (1.0 = linear).
    pub gamma_root: f32,
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
            sigma_cents: 3.0, // Slightly wider peaks to ease sampling
            normalize_output: true,
            gamma_root: 1.0,
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

impl HarmonicityKernel {
    pub fn new(space: &Log2Space, params: HarmonicityParams) -> Self {
        let params = Self::sanitize_params(params);
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

    /// The core function: Env -> common-root series -> harmonicity landscape.
    /// Returns the landscape and its reporting max (1.0 when normalized).
    pub fn potential_h_from_log2_spectrum(
        &self,
        envelope: &[f32],
        space: &Log2Space,
    ) -> (Vec<f32>, f32) {
        assert_eq!(
            space.bins_per_oct, self.bins_per_oct,
            "Log2Space bins_per_oct mismatch (space={}, kernel={})",
            space.bins_per_oct, self.bins_per_oct
        );
        assert_eq!(
            envelope.len(),
            space.n_bins(),
            "envelope length mismatch (env={}, space={})",
            envelope.len(),
            space.n_bins()
        );
        let n_bins = envelope.len();
        let bins_per_oct = self.bins_per_oct as f32;
        let sub_limit = self.sub_limit.max(1);
        let harm_limit = self.harm_limit.max(1);
        let k_diag = sub_limit.min(harm_limit).max(1);

        // Step 0: Smooth Input (O(N))
        let smeared_env = self.convolve_smooth(envelope);

        // Diagonal self-term correction factor.
        let diag_w = self.params.diag_weight.clamp(0.0, 1.0);
        let diag_a = if diag_w < 1.0 {
            let mut sum_a = 0.0f32;
            for k in 1..=k_diag {
                let a = (k as f32).powf(-self.params.rho_common_root);
                sum_a += a * a;
            }
            sum_a
        } else {
            0.0
        };

        // Buffer for virtual roots (centered with padding on both sides).
        let padding = self.pad_bins;
        let center_offset = padding as f32;
        let buf_len = n_bins + 2 * padding;
        let mut root_spectrum = vec![0.0f32; buf_len];

        // Common-root series: project the spectrum down to virtual roots,
        // then back up to harmonics.
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

        let mut landscape = vec![0.0f32; n_bins];
        for m in 1..=harm_limit {
            let shift_bins = (m as f32).log2() * bins_per_oct;
            let weight = (m as f32).powf(-self.params.rho_common_root);
            let offset = shift_bins - center_offset;
            Self::accumulate_shifted(&root_spectrum, &mut landscape, offset, weight);
        }
        if diag_w < 1.0 {
            let scale = 1.0 - diag_w;
            for i in 0..n_bins {
                let adjusted = landscape[i] - scale * diag_a * smeared_env[i];
                landscape[i] = adjusted.max(0.0);
            }
        }

        // Step 3: Post-processing (optional gating, then peak find / normalize).
        let mut max_val = 1e-12;
        let do_gate = self.params.freq_gate;
        let do_norm = self.params.normalize_output;
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

        let report_max = if do_norm { 1.0 } else { max_val };
        (landscape, report_max)
    }
    fn sanitize_params(mut params: HarmonicityParams) -> HarmonicityParams {
        let defaults = HarmonicityParams::default();
        params.rho_common_root = finite_or(params.rho_common_root, defaults.rho_common_root);
        params.sigma_cents = finite_or(params.sigma_cents, defaults.sigma_cents).max(0.0);
        params.gamma_root = finite_or(params.gamma_root, defaults.gamma_root).max(1e-3);
        params.diag_weight = finite_or(params.diag_weight, defaults.diag_weight).clamp(0.0, 1.0);
        params.tfs_f_pl_hz = finite_or(params.tfs_f_pl_hz, defaults.tfs_f_pl_hz).max(1e-6);
        params.tfs_eta = finite_or(params.tfs_eta, defaults.tfs_eta).max(1e-6);
        params
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
            return input
                .iter()
                .map(|&v| if v.is_finite() { v.max(0.0) } else { 0.0 })
                .collect();
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
                let x = if input[input_idx].is_finite() {
                    input[input_idx].max(0.0)
                } else {
                    0.0
                };
                acc += x * self.smooth_kernel[j];
            }
            *out_val = acc;
        }
        output
    }

    #[inline]
    fn absfreq_gate(f_hz: f32, p: &HarmonicityParams) -> f32 {
        1.0 / (1.0 + (f_hz / p.tfs_f_pl_hz).powf(p.tfs_eta))
    }
}

#[inline]
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::log2space::Log2Space;
    #[cfg(feature = "plotcheck")]
    use crate::core::utils::ensure_plots_dir;

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
        use plotters::prelude::*;
        use std::fs::File;
        use std::path::Path;

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
    fn constructor_sanitizes_non_finite_params() {
        let space = Log2Space::new(80.0, 2000.0, 96);
        let params = HarmonicityParams {
            rho_common_root: f32::NAN,
            sigma_cents: f32::NAN,
            gamma_root: f32::NAN,
            diag_weight: f32::NAN,
            tfs_f_pl_hz: f32::NAN,
            tfs_eta: f32::NAN,
            ..HarmonicityParams::default()
        };
        let hk = HarmonicityKernel::new(&space, params);
        let env = vec![f32::NAN; space.n_bins()];
        let (landscape, _) = hk.potential_h_from_log2_spectrum(&env, &space);

        assert!(hk.params.rho_common_root.is_finite());
        assert!(
            landscape.iter().all(|v| v.is_finite() && *v >= 0.0),
            "harmonicity output must stay finite"
        );
    }
}
