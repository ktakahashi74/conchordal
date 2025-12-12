//! core/harmonicity_kernel.rs
//! Common Root (Sibling) Harmonicity Kernel on Log2Space.
//!
//! This implementation generates a consonance landscape based on the physiological
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
    pub num_subharmonics: u32,
    pub num_harmonics: u32,
    pub rho_sub: f32,
    pub rho_harm: f32,
    pub sigma_cents: f32,
    pub normalize_output: bool,
    pub freq_gate: bool,
    pub tfs_f_pl_hz: f32,
    pub tfs_eta: f32,
}

impl Default for HarmonicityParams {
    fn default() -> Self {
        Self {
            num_subharmonics: 12,
            num_harmonics: 12,
            rho_sub: 0.75,
            rho_harm: 0.5,
            sigma_cents: 2.0, // Sharp precision for phase locking
            normalize_output: true,
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
}

impl HarmonicityKernel {
    pub fn new(space: &Log2Space, params: HarmonicityParams) -> Self {
        // 1. Pre-calculate smoothing kernel
        let sigma_bins = params.sigma_cents / 1200.0 * space.bins_per_oct as f32;
        let half_width = (2.5 * sigma_bins).ceil() as usize;
        let width = 2 * half_width + 1;
        let mut k = vec![0.0f32; width];
        let mut sum = 0.0;
        for i in 0..width {
            let x = (i as isize - half_width as isize) as f32;
            k[i] = (-0.5 * (x / sigma_bins).powi(2)).exp();
            sum += k[i];
        }
        for v in &mut k {
            *v /= sum;
        }

        // 2. Pre-calculate necessary padding
        // How many bins down do we need to store the deepest root?
        let max_sub_oct = (params.num_subharmonics as f32).log2();
        let pad_bins = (max_sub_oct * space.bins_per_oct as f32).ceil() as usize;

        Self {
            bins_per_oct: space.bins_per_oct,
            params,
            smooth_kernel: k,
            pad_bins,
        }
    }

    /// The core function: Env -> Roots -> Landscape
    pub fn potential_h_from_log2_spectrum(
        &self,
        envelope: &[f32],
        space: &Log2Space,
    ) -> (Vec<f32>, f32) {
        let n_bins = envelope.len();
        let bins_per_oct = self.bins_per_oct as f32;

        // Step 0: Smooth Input (O(N))
        let smeared_env = self.convolve_smooth(envelope);

        // Buffer for Virtual Roots
        // Size = N + Padding (to hold roots below f_min)
        let mut root_spectrum = vec![0.0f32; n_bins + self.pad_bins];

        // Step 1: Downward Projection (Scatter Env into Roots)
        // Env[i] implies Root at Env[i - shift]
        // Since Root buffer has padding, Env[0] maps to Root[pad_bins].
        // So Env[i] maps to Root[pad_bins + i - shift].
        // Target Offset = pad_bins - shift.
        for k in 1..=self.params.num_subharmonics {
            let shift_bins = (k as f32).log2() * bins_per_oct;
            let weight = 1.0 / (k as f32).powf(self.params.rho_sub);

            // "Add smeared_env into root_spectrum at offset"
            let offset = self.pad_bins as f32 - shift_bins;
            Self::accumulate_shifted(&smeared_env, &mut root_spectrum, offset, weight);
        }

        // Step 2: Upward Projection (Gather Roots into Landscape)
        // Root[j] implies Harmonic at Root[j + shift]
        // Landscape[i] corresponds to Root[pad_bins + i].
        // So Landscape[i] gets energy from Root[pad_bins + i - shift].
        // Read Offset (in Root) = pad_bins - shift.
        // Or view it as: Shift Root buffer RIGHT by `shift`, then map to Landscape.
        // Let's use the Scatter view again for consistency:
        // We take the WHOLE root_spectrum and add it to Landscape? No, sizes differ.
        // It's cleaner to say: Landscape[i] += Root[pad_bins + i - shift].
        // This is equivalent to: Add `root_spectrum` into `landscape` with offset `- (pad_bins - shift)`.
        // Wait, easier: "Add Root into Landscape with offset `shift - pad_bins`".

        let mut landscape = vec![0.0f32; n_bins];

        for m in 1..=self.params.num_harmonics {
            let shift_bins = (m as f32).log2() * bins_per_oct;
            let weight = 1.0 / (m as f32).powf(self.params.rho_harm);

            // Mapping: Root[x] -> Land[x + shift - pad_bins]
            // We want to add `root_spectrum` into `landscape`.
            // The 0-th element of root_spectrum corresponds to frequency f_min / 2^pad.
            // The 0-th element of landscape corresponds to f_min.
            // The relative shift is: `shift - pad_bins`.
            let offset = shift_bins - self.pad_bins as f32;
            Self::accumulate_shifted(&root_spectrum, &mut landscape, offset, weight);
        }

        // Step 3: Post-processing
        let mut max_val = 1e-12;
        let do_gate = self.params.freq_gate;
        let do_norm = self.params.normalize_output;

        // Combined loop for gating and max-finding (Auto-vectorized)
        for i in 0..n_bins {
            let v = &mut landscape[i];
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
            max_val = 1.0;
        }

        (landscape, max_val)
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
        let n = input.len();
        let mut output = vec![0.0; n];
        let k_len = self.smooth_kernel.len();
        let half = k_len / 2;

        // Naive but clear implementation.
        // For very large N, FFT conv is better, but here N ~ 200-4000, kernel ~ 5-10.
        // Direct convolution is faster.
        for i in 0..n {
            let mut acc = 0.0;
            let start_k = if i < half { half - i } else { 0 };
            let end_k = if i + half >= n {
                k_len - (i + half - n + 1)
            } else {
                k_len
            };

            for j in start_k..end_k {
                let input_idx = i + j - half;
                acc += input[input_idx] * self.smooth_kernel[j];
            }
            output[i] = acc;
        }
        output
    }

    #[inline]
    fn absfreq_gate(f_hz: f32, p: &HarmonicityParams) -> f32 {
        1.0 / (1.0 + (f_hz / p.tfs_f_pl_hz).powf(p.tfs_eta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::log2space::Log2Space;
    use plotters::prelude::*;
    use std::fs::File;
    use std::path::Path;

    #[test]
    fn test_sibling_consonance_creation() {
        // Input: 200Hz.
        // We expect peaks at:
        // - 100Hz (Root)
        // - 400Hz (Octave)
        // - 300Hz (Perfect 5th via 100Hz root)

        let space = Log2Space::new(50.0, 800.0, 200);
        let params = HarmonicityParams::default();

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
        let params = HarmonicityParams::default();

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
    #[ignore]
    fn plot_sibling_landscape_png() {
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

        let out_path = Path::new("target/test_sibling_landscape.png");
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
}
