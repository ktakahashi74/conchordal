//! core/consonance_kernel.rs
//! Consonance H on Log2Space using an integer‑ratio comb kernel.
//!
//! - Kernel K(Δlog2) is 1‑octave periodic → circular convolution on one octave.
//! - Input: NSGT‑RT amplitude envelope per frame with optional amplitude compression.
//! - Integer ratios: all (m,n) with m≤max_num, n≤max_den, reduced, mapped to [1,2), de‑duplicated.
//! - FFT path via rustfft for O(B log B) circular convolution on one octave.

use std::collections::HashMap;

use crate::core::log2::Log2Space;
use rustfft::{FftPlanner, num_complex::Complex32};

#[inline]
fn wrap_unit(u: f32) -> f32 {
    // Fold to [-0.5, 0.5)
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
    // Euclidean algorithm
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Parameters for the consonance kernel.
#[derive(Clone, Copy, Debug)]
pub struct ConsonanceParams {
    /// Gaussian width in cents.
    pub sigma_cents: f32, // e.g., 20.0
    /// Ratio weight: w_{mn} ∝ (m+n)^(-gamma_ratio).
    pub gamma_ratio: f32, // e.g., 1.5
    /// Normalize kernel to max=1.0.
    pub normalize_kernel: bool,
    /// Integer ratio search bounds (numerator/denominator).
    pub max_num: u32,
    pub max_den: u32,
    pub max_complexity: Option<u32>,
}

impl Default for ConsonanceParams {
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

/// Integer‑ratio comb kernel on Log2Space.
#[derive(Clone, Debug)]
pub struct ConsonanceKernel {
    pub bins_per_oct: u32,
    pub params: ConsonanceParams,
    // One‑octave discrete kernel (length = bins_per_oct).
    kernel_one_oct: Vec<f32>,
    // Debug: accepted ratio centers (log2) and weights.
    ratio_centers_log2: Vec<f32>,
    ratio_weights: Vec<f32>,
}

impl ConsonanceKernel {
    /// Build for a given Log2Space.
    pub fn new(space: &Log2Space, params: ConsonanceParams) -> Self {
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

    /// Rebuild for another Log2Space (requires same bins_per_oct).
    pub fn rebuild_for_space(&mut self, space: &Log2Space) {
        self.bins_per_oct = space.bins_per_oct;
        self.kernel_one_oct = Self::build_kernel_one_oct(
            self.bins_per_oct,
            self.params.sigma_cents,
            &self.ratio_centers_log2,
            &self.ratio_weights,
            self.params.normalize_kernel,
        );
    }

    /// Build the ratio set:
    /// - reduce (m,n) by gcd
    /// - normalize r=m/n to [1,2)
    /// - quantize log2(r) (micro‑cent) and deduplicate; prefer smaller m+n
    fn build_ratio_set(params: &ConsonanceParams) -> (Vec<f32>, Vec<f32>) {
        use std::collections::HashMap;
        let max_m = params.max_num;
        let max_n = params.max_den;
        let gamma = params.gamma_ratio;
        let max_cmplx = params.max_complexity;

        // micro-cent quantization
        let quant = 1200.0_f32 * 1000.0_f32;

        let mut chosen: HashMap<i32, (usize, u32)> = HashMap::new();
        let mut centers_log2: Vec<f32> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();

        for m in 1..=max_m {
            for n in 1..=max_n {
                let g = gcd_u32(m, n);
                let (rm, rn) = (m / g, n / g);
                let cmplx = rm + rn;

                // (1) perceptual complexity cutoff
                if let Some(limit) = max_cmplx {
                    if cmplx > limit {
                        continue;
                    }
                }

                // (2) normalize to [1,2)
                let mut r = (rm as f32) / (rn as f32);
                while r < 1.0 {
                    r *= 2.0;
                }
                while r >= 2.0 {
                    r /= 2.0;
                }

                let mu = r.log2();
                let code = (mu * quant).round() as i32;

                // (3) weight by (m+n)^(-γ)
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

        // (4) ensure 1/1 exists
        if centers_log2.iter().all(|&v| v != 0.0) {
            centers_log2.push(0.0);
            weights.push(1.0);
        }

        (centers_log2, weights)
    }

    /// Build a 1‑octave kernel (length = bins_per_oct).
    fn build_kernel_one_oct(
        bins_per_oct: u32,
        sigma_cents: f32,
        centers_log2: &[f32],
        weights: &[f32],
        normalize_to_one: bool,
    ) -> Vec<f32> {
        let b = bins_per_oct as usize;
        let step_log2 = 1.0 / bins_per_oct as f32;
        let sigma = sigma_cents / 1200.0; // cents → log2
        let two_s2 = 2.0 * sigma * sigma;

        let mut k = vec![0.0_f32; b];
        for d in 0..b {
            let dx = d as f32 * step_log2; // [0,1)
            let mut acc = 0.0_f32;
            for (j, &mu) in centers_log2.iter().enumerate() {
                let w = weights[j];
                let dd = dist1(dx - mu);
                acc += w * (-(dd * dd) / two_s2).exp();
            }
            k[d] = acc;
        }

        if normalize_to_one {
            if let Some(maxv) = k
                .iter()
                .cloned()
                .fold(None, |m, v| Some(m.map_or(v, |u: f32| u.max(v))))
            {
                if maxv > 0.0 {
                    for v in &mut k {
                        *v /= maxv;
                    }
                }
            }
        }
        k
    }

    /// Naive circular convolution on one octave (O(B^2)).
    fn circ_convolve(a: &[f32], k: &[f32]) -> Vec<f32> {
        let b = a.len();
        assert_eq!(b, k.len());
        let mut y = vec![0.0f32; b];
        for i in 0..b {
            let mut acc = 0.0f32;
            for d in 0..b {
                let j = (i + b - d) % b; // a[j] * k[d]
                acc += a[j] * k[d];
            }
            y[i] = acc;
        }
        y
    }

    /// Circular convolution on one octave using FFT (O(B log B)).
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

    /// Fold full‑band weights to 1‑octave chroma.
    fn fold_to_chroma(weights_full: &[f32], bins_per_oct: u32) -> Vec<f32> {
        let b = bins_per_oct as usize;
        let mut chroma = vec![0.0f32; b];
        for (i, &w) in weights_full.iter().enumerate() {
            chroma[i % b] += w.max(0.0);
        }
        chroma
    }

    /// Consonance landscape (probe response) from an NSGT‑RT amplitude envelope.
    ///
    /// Steps:
    ///  1) weights_full[i] = envelope[i]^gamma_amp
    ///  2) fold to chroma
    ///  3) chroma ⊛ kernel_one_oct (FFT)
    ///  4) expand to full bins and normalize by sum(chroma)
    pub fn potential_c_from_log2_spectrum(
        &self,
        envelope: &[f32],
        space: &Log2Space,
        gamma_amp: f32,
    ) -> (Vec<f32>, f32) {
        assert_eq!(space.bins_per_oct, self.bins_per_oct);
        assert_eq!(space.n_bins(), envelope.len());

        // 1) amplitude compression
        let weights_full: Vec<f32> = if (gamma_amp - 1.0).abs() < 1e-12 {
            envelope.iter().map(|&a| a.max(0.0)).collect()
        } else {
            envelope
                .iter()
                .map(|&a| a.max(0.0).powf(gamma_amp))
                .collect()
        };

        // 2) chroma fold
        let chroma = Self::fold_to_chroma(&weights_full, self.bins_per_oct);

        // denom for level‑invariance
        let denom: f32 = chroma.iter().sum::<f32>().max(1e-12);

        // 3) circular convolution (FFT)
        let num = Self::circ_convolve_fft(&chroma, &self.kernel_one_oct);

        // 4) expand to full bins
        let b = self.bins_per_oct as usize;
        let mut h_field = vec![0.0f32; space.n_bins()];
        for i in 0..space.n_bins() {
            let r = i % b;
            h_field[i] = num[r] / denom;
        }
        (h_field, denom)
    }

    /// 1‑octave kernel (for debug/visualization).
    pub fn kernel_one_oct(&self) -> &[f32] {
        &self.kernel_one_oct
    }

    /// Ratio centers (log2) and weights (debug).
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

    // ---------------------------------------------
    // Kernel construction & sanity checks
    // ---------------------------------------------

    #[test]
    fn kernel_length_and_max_norm() {
        let space = Log2Space::new(27.5, 880.0, 24);
        let ck = ConsonanceKernel::new(&space, ConsonanceParams::default());
        assert_eq!(ck.kernel_one_oct().len(), space.bins_per_oct as usize);
        let mx = ck.kernel_one_oct().iter().cloned().fold(0.0f32, f32::max);
        assert!(
            (mx - 1.0).abs() < 1e-3,
            "kernel max should be ~1.0 when normalized"
        );
        assert!(ck.kernel_one_oct().iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn ratio_set_contains_common_intervals() {
        let space = Log2Space::new(55.0, 3520.0, 48);
        let ck = ConsonanceKernel::new(&space, ConsonanceParams::default());
        let (centers, _w) = ck.ratios_debug();

        let targets: [f32; 6] = [1.0, 3.0 / 2.0, 4.0 / 3.0, 5.0 / 4.0, 6.0 / 5.0, 5.0 / 3.0];
        for r in targets {
            let mu: f32 = (r.fract() == 0.0).then(|| 0.0_f32).unwrap_or_else(|| {
                let mut v: f32 = r;
                while v < 1.0 {
                    v *= 2.0;
                }
                while v >= 2.0 {
                    v /= 2.0;
                }
                v.log2()
            });
            let ok = centers
                .iter()
                .any(|&c: &f32| (wrap_unit(c - mu)).abs() < 1e-6);
            assert!(ok, "missing ratio near log2({:.5})", r);
        }
    }

    // ---------------------------------------------
    // FFT vs naive circular convolution
    // ---------------------------------------------

    #[test]
    fn fft_matches_naive() {
        let b = 64usize;
        let a: Vec<f32> = (0..b)
            .map(|i| ((i as f32 * 0.13).sin() + 1.0) * 0.5)
            .collect();
        let k: Vec<f32> = (0..b)
            .map(|i| ((i as f32 * 0.07).cos() + 1.0) * 0.5)
            .collect();

        let y_naive = ConsonanceKernel::circ_convolve(&a, &k);
        let y_fft = ConsonanceKernel::circ_convolve_fft(&a, &k);

        for (u, v) in y_naive.iter().zip(y_fft.iter()) {
            assert!((u - v).abs() < 1e-5, "FFT and naive should match");
        }
    }

    // ---------------------------------------------
    // Field property: δ input → shifted kernel on chroma
    // ---------------------------------------------

    #[test]
    fn delta_input_reproduces_kernel_shift() {
        let space = Log2Space::new(110.0, 3520.0, 48); // multi‑octave
        let ck = ConsonanceKernel::new(&space, ConsonanceParams::default());

        // delta at i0
        let mut env = vec![0.0f32; space.n_bins()];
        let i0 = space.index_of_freq(440.0).unwrap();
        env[i0] = 1.0;

        let (h_field, denom) = ck.potential_c_from_log2_spectrum(&env, &space, 1.0);
        assert!(
            (denom - 1.0).abs() < 1e-6,
            "denom=1 for single delta after fold"
        );

        // Build chroma and direct conv to get the expected circular kernel shift.
        let chroma = ConsonanceKernel::fold_to_chroma(&env, space.bins_per_oct);
        let num = ConsonanceKernel::circ_convolve_fft(&chroma, ck.kernel_one_oct());

        let b = space.bins_per_oct as usize;
        for i in 0..space.n_bins() {
            let r = i % b;
            let exp = num[r]; // denom = 1
            assert!((h_field[i] - exp).abs() < 1e-5, "mismatch at bin {}", i);
        }
    }

    // ---------------------------------------------
    // Plot: kernel shape (ignored by default)
    // ---------------------------------------------

    #[test]
    #[ignore]
    fn plot_kernel_one_oct_png() {
        let space = Log2Space::new(27.5, 3520.0, 500);
        let ck = ConsonanceKernel::new(&space, ConsonanceParams::default());
        let k = ck.kernel_one_oct();
        let b = k.len();

        let xs: Vec<f32> = (0..b).map(|i| i as f32 / b as f32).collect();

        let out_path = Path::new("target/test_consonance_kernel_one_oct.png");
        let root = BitMapBackend::new(out_path, (1600, 1000)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let ymax = k.iter().cloned().fold(0.0, f32::max) * 1.05;
        let mut chart = ChartBuilder::on(&root)
            .caption("Consonance Kernel (1 Octave, Δlog2)", ("sans-serif", 30))
            .margin(10)
            .build_cartesian_2d(0.0f32..1.0f32, 0.0f32..ymax)
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("Δlog2 (oct)")
            .y_desc("Amplitude")
            .draw()
            .unwrap();
        chart
            .draw_series(LineSeries::new(
                xs.iter().zip(k.iter()).map(|(&x, &y)| (x, y)),
                &BLUE,
            ))
            .unwrap();

        root.present().unwrap();
        assert!(File::open(out_path).is_ok());
    }

    // ---------------------------------------------
    // Plot: two‑tone sweep vs base (ignored by default)
    // ---------------------------------------------

    #[test]
    #[ignore]
    fn plot_consonance_two_tone_sweep_png() {
        let space = Log2Space::new(110.0, 3520.0, 500);
        let ck = ConsonanceKernel::new(&space, ConsonanceParams::default());

        let base = 440.0;
        let i_base = space.index_of_freq(base).unwrap();
        let b = space.bins_per_oct as usize;

        let mut ratios = Vec::new();
        let mut vals = Vec::new();

        // Sweep r in [1,2)
        let steps = 5000;
        for s in 0..steps {
            let r = 1.0 + (s as f32) / (steps as f32); // 1.0 .. 2.0
            let f2 = base * r;
            if f2 >= space.fmin && f2 <= space.fmax {
                let mut env = vec![0.0f32; space.n_bins()];
                env[i_base] = 1.0;
                if let Some(i2) = space.index_of_freq(f2) {
                    env[i2] = 1.0;
                } else {
                    continue;
                }
                let (h, _denom) = ck.potential_c_from_log2_spectrum(&env, &space, 1.0);
                vals.push(h[i_base]); // probe at base pitch class
                ratios.push(r);
            }
        }

        let out_path = Path::new("target/test_consonance_two_tone_sweep.png");
        let root = BitMapBackend::new(out_path, (1600, 1000)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let ymax = vals.iter().cloned().fold(0.0, f32::max) * 1.05;
        let mut chart = ChartBuilder::on(&root)
            .caption("Consonance at Base vs Two‑tone Ratio r", ("sans-serif", 30))
            .margin(10)
            .build_cartesian_2d(1.0f32..2.0f32, 0.0f32..ymax)
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("r = f2/f1")
            .y_desc("H(base)")
            .draw()
            .unwrap();
        chart
            .draw_series(LineSeries::new(
                ratios.iter().zip(vals.iter()).map(|(&x, &y)| (x, y)),
                &RED,
            ))
            .unwrap();

        // Optional: mark simple ratios
        let marks = [
            (3.0 / 2.0, "3/2"),
            (4.0 / 3.0, "4/3"),
            (5.0 / 4.0, "5/4"),
            (6.0 / 5.0, "6/5"),
        ];
        for (rx, lbl) in marks {
            chart
                .draw_series(std::iter::once(PathElement::new(
                    [(rx, 0.0), (rx, ymax)],
                    &BLACK,
                )))
                .unwrap();
            chart
                .draw_series(std::iter::once(Text::new(
                    lbl,
                    (rx, ymax * 0.95),
                    ("sans-serif", 18).into_font(),
                )))
                .unwrap();
        }

        root.present().unwrap();
        assert!(File::open(out_path).is_ok());
        assert!(b > 0); // silence unused var warning if any
    }
}
