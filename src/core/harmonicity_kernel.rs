//! core/harmonicity_kernel.rs
//! Harmonicity H on Log2Space using an integer-ratio comb kernel.
//!
//! - Kernel K(Δlog2) is 1-octave periodic → circular convolution on one octave.
//! - Input: NSGT-RT amplitude envelope per frame with optional amplitude compression.
//! - Integer ratios: all (m,n) with m≤max_num, n≤max_den, reduced, mapped to [1,2), de-duplicated.
//! - FFT path via rustfft for O(B log B) circular convolution on one octave.

use crate::core::log2::Log2Space;
use rustfft::{FftPlanner, num_complex::Complex32};

#[inline]
fn wrap_unit(u: f32) -> f32 {
    let y = u - u.floor(); // [0, 1)
    if y >= 0.5 { y - 1.0 } else { y }
}

#[inline]
fn dist1(x: f32) -> f32 {
    wrap_unit(x)
}

#[inline]
fn log2f(x: f32) -> f32 {
    x.log2()
}

#[inline]
fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Parameters for the harmonicity kernel.
#[derive(Clone, Copy, Debug)]
pub struct HarmonicityParams {
    pub sigma_cents: f32,
    pub gamma_ratio: f32,
    pub normalize_kernel: bool,
    pub max_num: u32,
    pub max_den: u32,
    pub max_complexity: Option<u32>,
}

impl Default for HarmonicityParams {
    fn default() -> Self {
        Self {
            sigma_cents: 7.0,
            gamma_ratio: 1.2,
            normalize_kernel: true,
            max_num: 16,
            max_den: 16,
            max_complexity: Some(15),
        }
    }
}

/// Integer-ratio comb kernel on Log2Space.
#[derive(Clone, Debug)]
pub struct HarmonicityKernel {
    pub bins_per_oct: u32,
    pub params: HarmonicityParams,
    kernel_one_oct: Vec<f32>,
    ratio_centers_log2: Vec<f32>,
    ratio_weights: Vec<f32>,
}

impl HarmonicityKernel {
    pub fn new(space: &Log2Space, params: HarmonicityParams) -> Self {
        let bpo = space.bins_per_oct;
        let (ratio_centers_log2, ratio_weights) = Self::build_ratio_set(&params);
        let kernel_one_oct = Self::build_kernel_one_oct(
            bpo,
            params.sigma_cents,
            &ratio_centers_log2,
            &ratio_weights,
            params.normalize_kernel,
        );
        Self {
            bins_per_oct: bpo,
            params,
            kernel_one_oct,
            ratio_centers_log2,
            ratio_weights,
        }
    }

    fn build_ratio_set(params: &HarmonicityParams) -> (Vec<f32>, Vec<f32>) {
        use std::collections::HashMap;
        let max_m = params.max_num;
        let max_n = params.max_den;
        let gamma = params.gamma_ratio;
        let max_cmplx = params.max_complexity;
        let quant = 1200.0_f32 * 1000.0_f32;

        let mut chosen: HashMap<i32, (usize, u32)> = HashMap::new();
        let mut centers_log2 = Vec::new();
        let mut weights = Vec::new();

        for m in 1..=max_m {
            for n in 1..=max_n {
                let g = gcd_u32(m, n);
                let (rm, rn) = (m / g, n / g);
                let cmplx = rm + rn;
                if let Some(limit) = max_cmplx {
                    if cmplx > limit {
                        continue;
                    }
                }
                let mut r = (rm as f32) / (rn as f32);
                while r < 1.0 {
                    r *= 2.0;
                }
                while r >= 2.0 {
                    r /= 2.0;
                }
                let mu = r.log2();
                let code = (mu * quant).round() as i32;
                let w = 1.0 / ((rm + rn) as f32).powf(gamma);
                if let Some((idx, best_c)) = chosen.get(&code).cloned() {
                    if cmplx < best_c {
                        centers_log2[idx] = mu;
                        weights[idx] = w;
                        chosen.insert(code, (idx, cmplx));
                    }
                } else {
                    let idx = centers_log2.len();
                    centers_log2.push(mu);
                    weights.push(w);
                    chosen.insert(code, (idx, cmplx));
                }
            }
        }
        if centers_log2.iter().all(|&v| v != 0.0) {
            centers_log2.push(0.0);
            weights.push(1.0);
        }
        (centers_log2, weights)
    }

    fn build_kernel_one_oct(
        bins_per_oct: u32,
        sigma_cents: f32,
        centers_log2: &[f32],
        weights: &[f32],
        normalize_to_one: bool,
    ) -> Vec<f32> {
        let b = bins_per_oct as usize;
        let step_log2 = 1.0 / bins_per_oct as f32;
        let sigma = sigma_cents / 1200.0;
        let two_s2 = 2.0 * sigma * sigma;
        let mut k = vec![0.0; b];
        for d in 0..b {
            let dx = d as f32 * step_log2;
            let mut acc = 0.0;
            for (j, &mu) in centers_log2.iter().enumerate() {
                let w = weights[j];
                let dd = dist1(dx - mu);
                acc += w * (-(dd * dd) / two_s2).exp();
            }
            k[d] = acc;
        }
        if normalize_to_one {
            if let Some(maxv) = k.iter().cloned().reduce(f32::max) {
                if maxv > 0.0 {
                    for v in &mut k {
                        *v /= maxv;
                    }
                }
            }
        }
        k
    }

    fn circ_convolve(a: &[f32], k: &[f32]) -> Vec<f32> {
        let b = a.len();
        assert_eq!(b, k.len());
        let mut y = vec![0.0; b];
        for i in 0..b {
            let mut acc = 0.0;
            for d in 0..b {
                let j = (i + b - d) % b;
                acc += a[j] * k[d];
            }
            y[i] = acc;
        }
        y
    }

    fn circ_convolve_fft(a: &[f32], k: &[f32]) -> Vec<f32> {
        let n = a.len();
        assert_eq!(n, k.len());
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);
        let mut a_spec: Vec<Complex32> = a.iter().map(|&x| Complex32::new(x, 0.0)).collect();
        let mut k_spec: Vec<Complex32> = k.iter().map(|&x| Complex32::new(x, 0.0)).collect();
        fft.process(&mut a_spec);
        fft.process(&mut k_spec);
        for i in 0..n {
            a_spec[i] *= k_spec[i];
        }
        ifft.process(&mut a_spec);
        let scale = 1.0 / (n as f32);
        a_spec.into_iter().map(|c| c.re * scale).collect()
    }

    fn fold_to_chroma(weights_full: &[f32], bins_per_oct: u32) -> Vec<f32> {
        let b = bins_per_oct as usize;
        let mut chroma = vec![0.0; b];
        for (i, &w) in weights_full.iter().enumerate() {
            chroma[i % b] += w.max(0.0);
        }
        chroma
    }

    /// Harmonicity landscape (probe response) from an NSGT-RT amplitude envelope.
    pub fn potential_h_from_log2_spectrum(
        &self,
        envelope: &[f32],
        space: &Log2Space,
        gamma_amp: f32,
    ) -> (Vec<f32>, f32) {
        assert_eq!(space.bins_per_oct, self.bins_per_oct);
        assert_eq!(space.n_bins(), envelope.len());
        let weights_full: Vec<f32> = if (gamma_amp - 1.0).abs() < 1e-12 {
            envelope.iter().map(|&a| a.max(0.0)).collect()
        } else {
            envelope
                .iter()
                .map(|&a| a.max(0.0).powf(gamma_amp))
                .collect()
        };
        let chroma = Self::fold_to_chroma(&weights_full, self.bins_per_oct);
        let denom: f32 = chroma.iter().sum::<f32>().max(1e-12);
        let num = Self::circ_convolve_fft(&chroma, &self.kernel_one_oct);
        let b = self.bins_per_oct as usize;
        let mut h_field = vec![0.0; space.n_bins()];
        for i in 0..space.n_bins() {
            let r = i % b;
            h_field[i] = num[r] / denom;
        }
        (h_field, denom)
    }

    pub fn kernel_one_oct(&self) -> &[f32] {
        &self.kernel_one_oct
    }

    pub fn ratios_debug(&self) -> (&[f32], &[f32]) {
        (&self.ratio_centers_log2, &self.ratio_weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::log2::Log2Space;
    use plotters::prelude::*;
    use std::fs::File;
    use std::path::Path;

    #[test]
    fn kernel_length_and_max_norm() {
        let space = Log2Space::new(27.5, 880.0, 24);
        let hk = HarmonicityKernel::new(&space, HarmonicityParams::default());
        assert_eq!(hk.kernel_one_oct().len(), space.bins_per_oct as usize);
        let mx = hk.kernel_one_oct().iter().cloned().fold(0.0f32, f32::max);
        assert!((mx - 1.0).abs() < 1e-3);
        assert!(hk.kernel_one_oct().iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn ratio_set_contains_common_intervals() {
        let space = Log2Space::new(55.0, 3520.0, 48);
        let hk = HarmonicityKernel::new(&space, HarmonicityParams::default());
        let (centers, _w) = hk.ratios_debug();
        let targets = [1.0, 3.0 / 2.0, 4.0 / 3.0, 5.0 / 4.0, 6.0 / 5.0, 5.0 / 3.0];
        for r in targets {
            let mut v: f32 = r;
            while v < 1.0 {
                v *= 2.0;
            }
            while v >= 2.0 {
                v /= 2.0;
            }
            let mu = v.log2();
            let ok = centers.iter().any(|&c| (wrap_unit(c - mu)).abs() < 1e-6);
            assert!(ok, "missing ratio near log2({:.5})", r);
        }
    }

    #[test]
    fn fft_matches_naive() {
        let b = 64usize;
        let a: Vec<f32> = (0..b)
            .map(|i| ((i as f32 * 0.13).sin() + 1.0) * 0.5)
            .collect();
        let k: Vec<f32> = (0..b)
            .map(|i| ((i as f32 * 0.07).cos() + 1.0) * 0.5)
            .collect();
        let y_naive = HarmonicityKernel::circ_convolve(&a, &k);
        let y_fft = HarmonicityKernel::circ_convolve_fft(&a, &k);
        for (u, v) in y_naive.iter().zip(y_fft.iter()) {
            assert!((u - v).abs() < 1e-5);
        }
    }

    #[test]
    fn delta_input_reproduces_kernel_shift() {
        let space = Log2Space::new(110.0, 3520.0, 48);
        let hk = HarmonicityKernel::new(&space, HarmonicityParams::default());
        let mut env = vec![0.0f32; space.n_bins()];
        let i0 = space.index_of_freq(440.0).unwrap();
        env[i0] = 1.0;
        let (h_field, denom) = hk.potential_h_from_log2_spectrum(&env, &space, 1.0);
        assert!((denom - 1.0).abs() < 1e-6);
        let chroma = HarmonicityKernel::fold_to_chroma(&env, space.bins_per_oct);
        let num = HarmonicityKernel::circ_convolve_fft(&chroma, hk.kernel_one_oct());
        let b = space.bins_per_oct as usize;
        for i in 0..space.n_bins() {
            let r = i % b;
            assert!((h_field[i] - num[r]).abs() < 1e-5);
        }
    }
}
