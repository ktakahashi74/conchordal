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
//! ...without using any hardcoded ratio templates.

use crate::core::log2space::Log2Space;

#[inline]
fn log2f(x: f32) -> f32 {
    x.log2()
}

/// Parameters for the Sibling Projection algorithm.
#[derive(Clone, Copy, Debug)]
pub struct HarmonicityParams {
    /// Step 1: How far down to look for roots? (e.g., 8).
    /// Corresponds to the denominator of perceptible ratios.
    pub num_subharmonics: u32,

    /// Step 2: How far up to project harmonics from roots? (e.g., 8).
    /// Corresponds to the numerator of perceptible ratios.
    pub num_harmonics: u32,

    /// Weight decay for root search (1/k^rho).
    pub rho_sub: f32,

    /// Weight decay for harmonic projection (1/m^rho).
    pub rho_harm: f32,

    /// Smoothing width in cents (tolerance/auditory filter width).
    /// Applied to the input envelope before processing.
    pub sigma_cents: f32,

    /// Normalize the final landscape to max=1.0.
    pub normalize_output: bool,

    /// Apply absolute-frequency gating (Phase-locking roll-off).
    pub freq_gate: bool,
    pub tfs_f_pl_hz: f32,
    pub tfs_eta: f32,
}

impl Default for HarmonicityParams {
    fn default() -> Self {
        Self {
            num_subharmonics: 12,
            num_harmonics: 12,
            rho_sub: 0.6,
            rho_harm: 0.6,
            sigma_cents: 10.0,
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

    /// Pre-calculated smoothing kernel (Gaussian FIR).
    smooth_kernel: Vec<f32>,
    smooth_offset: usize,
}

impl HarmonicityKernel {
    pub fn new(space: &Log2Space, params: HarmonicityParams) -> Self {
        // Build Gaussian smoothing kernel
        // sigma_bins = sigma_cents / 1200 * bins_per_oct
        let sigma_bins = params.sigma_cents / 1200.0 * space.bins_per_oct as f32;

        // Kernel width ~ 4*sigma (covers 99.9%)
        let half_width = (2.5 * sigma_bins).ceil() as usize;
        let width = 2 * half_width + 1;

        let mut k = vec![0.0f32; width];
        let mut sum = 0.0;
        for i in 0..width {
            let x = (i as isize - half_width as isize) as f32;
            k[i] = (-0.5 * (x / sigma_bins).powi(2)).exp();
            sum += k[i];
        }
        // Normalize
        for v in &mut k {
            *v /= sum;
        }

        Self {
            bins_per_oct: space.bins_per_oct,
            params,
            smooth_kernel: k,
            smooth_offset: half_width,
        }
    }

    #[inline]
    fn absfreq_gate(f_hz: f32, p: &HarmonicityParams) -> f32 {
        let g_pl = 1.0 / (1.0 + (f_hz / p.tfs_f_pl_hz).powf(p.tfs_eta.max(0.1)));
        g_pl.clamp(0.0, 1.0)
    }

    /// Calculate the Harmonicity Landscape.
    /// Returns (landscape_vector, normalization_factor).
    /// High values indicate high consonance/stability.
    pub fn potential_h_from_log2_spectrum(
        &self,
        envelope: &[f32],
        space: &Log2Space,
    ) -> (Vec<f32>, f32) {
        let n_bins = envelope.len();

        // 0. Pre-process: Smooth the input
        let smeared_env = self.convolve_smooth(envelope);
        let bins_per_oct = self.bins_per_oct as f32;

        // --- Step 1: Find Virtual Roots (Downward) ---
        // Buffer to hold the strength of potential roots
        let mut root_spectrum = vec![0.0f32; n_bins];

        // Loop k: 1 (self), 2 (1/2), 3 (1/3)...
        for k in 1..=self.params.num_subharmonics {
            let shift_oct = (k as f32).log2();
            let shift_bins = shift_oct * bins_per_oct;
            let weight = 1.0 / (k as f32).powf(self.params.rho_sub);

            // "Look UP" to find energy that supports this root
            // root[i] += env[i + shift]
            self.accumulate_look_upper(&smeared_env, &mut root_spectrum, shift_bins, weight);
        }

        // --- Step 2: Project Harmonics (Upward from Roots) ---
        // Buffer for the final landscape
        let mut landscape = vec![0.0f32; n_bins];

        // Loop m: 1 (root itself), 2 (x2), 3 (x3)...
        for m in 1..=self.params.num_harmonics {
            let shift_oct = (m as f32).log2();
            let shift_bins = shift_oct * bins_per_oct;
            let weight = 1.0 / (m as f32).powf(self.params.rho_harm);

            // "Look DOWN" to find roots that project to this harmonic
            // landscape[i] += root[i - shift]
            self.accumulate_look_lower(&root_spectrum, &mut landscape, shift_bins, weight);
        }

        // --- 3. Post-processing ---
        let mut max_val = 1e-12;

        // Apply frequency gating (Phase locking limit)
        if self.params.freq_gate {
            for (i, v) in landscape.iter_mut().enumerate() {
                let gate = Self::absfreq_gate(space.freq_of_index(i), &self.params);
                *v *= gate;
                if *v > max_val {
                    max_val = *v;
                }
            }
        } else {
            for v in &landscape {
                if *v > max_val {
                    max_val = *v;
                }
            }
        }

        // Normalize
        if self.params.normalize_output {
            let scale = 1.0 / max_val;
            for v in &mut landscape {
                *v *= scale;
            }
            max_val = 1.0;
        }

        (landscape, max_val)
    }

    /// Apply Gaussian smoothing.
    fn convolve_smooth(&self, input: &[f32]) -> Vec<f32> {
        let n = input.len();
        let mut output = vec![0.0; n];
        let k_len = self.smooth_kernel.len();
        let half = self.smooth_offset;

        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..k_len {
                let input_idx = (i as isize - half as isize) + j as isize;
                if input_idx >= 0 && input_idx < n as isize {
                    acc += input[input_idx as usize] * self.smooth_kernel[j];
                }
            }
            output[i] = acc;
        }
        output
    }

    /// Look "UP" (Higher Freq) in src to add to dest.
    /// Used for finding Roots: root[i] += env[i + shift]
    /// (If root is i, its harmonic is at i+shift)
    fn accumulate_look_upper(&self, src: &[f32], dest: &mut [f32], shift_bins: f32, weight: f32) {
        let n = src.len();
        let shift_int = shift_bins.floor() as usize;
        let shift_frac = shift_bins - shift_int as f32;
        let w_lower = weight * (1.0 - shift_frac);
        let w_upper = weight * shift_frac;

        // Bounds check: we need src[i + shift_int + 1] to exist
        // i + shift_int + 1 < n  =>  i < n - shift_int - 1
        let max_i = if n > shift_int + 1 {
            n - shift_int - 1
        } else {
            0
        };

        for i in 0..max_i {
            let idx = i + shift_int;
            dest[i] += src[idx] * w_lower + src[idx + 1] * w_upper;
        }

        // Edge case: last valid bin
        if shift_int < n && max_i < n {
            let i = n - shift_int - 1;
            dest[i] += src[n - 1] * w_lower;
        }
    }

    /// Look "DOWN" (Lower Freq) in src to add to dest.
    /// Used for projecting Harmonics: dest[i] += root[i - shift]
    /// (If harmonic is i, its root is at i-shift)
    fn accumulate_look_lower(&self, src: &[f32], dest: &mut [f32], shift_bins: f32, weight: f32) {
        let n = src.len();
        let shift_int = shift_bins.floor() as usize;
        let shift_frac = shift_bins - shift_int as f32;
        let w_lower = weight * (1.0 - shift_frac);
        let w_upper = weight * shift_frac;

        // We need src[i - shift]
        // i - shift_int >= 0 => i >= shift_int
        // Because of interpolation (using i-shift-1 and i-shift), we need slightly more care.
        // Target index in src is `i - shift_bins`.
        // Let `idx` = i - shift_bins.
        // We read src[floor(idx)] and src[ceil(idx)].

        // Start loop where i - shift_bins >= 0
        let start_i = (shift_bins).ceil() as usize;

        for i in start_i..n {
            let target_idx = i as f32 - shift_bins;

            // Linear interpolation manual
            let idx_int = target_idx.floor() as usize;
            let idx_frac = target_idx - idx_int as f32;

            // We need src[idx_int] and src[idx_int+1]
            if idx_int + 1 < n {
                let val = src[idx_int] * (1.0 - idx_frac) + src[idx_int + 1] * idx_frac;
                dest[i] += val * weight;
            } else if idx_int < n {
                dest[i] += src[idx_int] * weight;
            }
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

    #[test]
    fn test_sibling_consonance_creation() {
        // Input: 200Hz.
        // We expect peaks at:
        // - 100Hz (Root)
        // - 400Hz (Octave)
        // - 300Hz (Perfect 5th via 100Hz root)

        let space = Log2Space::new(50.0, 800.0, 200);
        let mut params = HarmonicityParams::default();
        params.num_subharmonics = 6;
        params.num_harmonics = 6;
        params.rho_sub = 0.5;
        params.rho_harm = 0.5;

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
        
        params.num_subharmonics = 12; 
        params.num_harmonics = 12;
        params.rho_sub = 0.4; 
        params.rho_harm = 0.4;
        params.sigma_cents = 15.0;
        
        let hk = HarmonicityKernel::new(&space, params);
        
        let mut env = vec![0.0; space.n_bins()];
        let f_input = 400.0;
        if let Some(idx) = space.index_of_freq(f_input) {
            env[idx] = 1.0;
        }

        let (landscape, _) = hk.potential_h_from_log2_spectrum(&env, &space);

        let idx_m3 = space.index_of_freq(400.0 * 1.2).unwrap();      // 6:5
        let idx_h7 = space.index_of_freq(400.0 * 1.75).unwrap();     // 7:4
        
        // Tritone (approx 1.414). 
        // Note: This is close to 7:5 (1.40), so it will have significant potential!
        let idx_tritone = space.index_of_freq(400.0 * 1.414).unwrap();

        println!("Potential at 6:5 (m3): {}", landscape[idx_m3]);
        println!("Potential at 7:4 (h7): {}", landscape[idx_h7]);
        println!("Potential at Tritone:  {}", landscape[idx_tritone]);

        assert!(landscape[idx_m3] > landscape[idx_tritone] * 1.1, "6:5 should be more stable than tritone");
        assert!(landscape[idx_h7] > landscape[idx_tritone] * 1.1, "7:4 should be more stable than tritone");
    }
    
    #[test]
    #[ignore]
    fn plot_sibling_landscape_png() {
        let space = Log2Space::new(20.0, 1600.0, 200);
        let mut p = HarmonicityParams::default();
        p.num_subharmonics = 12;
        p.num_harmonics = 12;
        p.rho_sub = 0.5;
        p.rho_harm = 0.5;
        p.sigma_cents = 10.0;

        let hk = HarmonicityKernel::new(&space, p);

        // Input: 200Hz (G3 approx)
        let mut env = vec![0.0; space.n_bins()];
        let f_input = 200.0;
        env[space.index_of_freq(f_input).unwrap()] = 1.0;

        let (y, _) = hk.potential_h_from_log2_spectrum(&env, &space);
        let xs: Vec<f32> = (0..space.n_bins())
            .map(|i| space.freq_of_index(i))
            .collect();

        let out_path = Path::new("target/test_sibling_landscape.png");
        let root = BitMapBackend::new(out_path, (1200, 600)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let max_y = y.iter().cloned().fold(0.0, f32::max) * 1.1;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("Sibling Landscape (Input: {}Hz)", f_input),
                ("sans-serif", 20),
            )
            .margin(10)
            .build_cartesian_2d(20.0f32..2000.0f32, 0.0f32..max_y)
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("Frequency [Hz]")
            .y_desc("Potential (Stability)")
            .draw()
            .unwrap();

        chart
            .draw_series(LineSeries::new(
                xs.iter().zip(y.iter()).map(|(&x, &y)| (x, y)),
                &BLUE,
            ))
            .unwrap();

        // Mark expected emergent ratios
        let markers = vec![
            (f_input * 1.5, "3:2", RED),
            (f_input * 1.25, "5:4", MAGENTA),
            (f_input * 0.5, "1:2", GREEN),
            (f_input * 2.0, "2:1", GREEN),
        ];

        for (freq, label, color) in markers {
            chart
                .draw_series(std::iter::once(PathElement::new(
                    vec![(freq, 0.0), (freq, max_y)],
                    color.mix(0.5),
                )))
                .unwrap();

            // Simple label positioning
            // (Note: Text positioning in plotters is absolute or relative to coord,
            // here just drawing lines for verification)
        }

        root.present().unwrap();
        assert!(File::open(out_path).is_ok());
    }
}
