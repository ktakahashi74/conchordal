//! core/harmonicity_kernel.rs
//! Multi-octave harmonicity kernel on Log2Space.
//!
//! - Build a stationary kernel K(Δlog2) over ±L octaves from integer ratios.
//! - Weights favor small integers and decay with octave distance.
//! - Per-frame result is linear convolution (FFT) on the full log2 axis.
//! - Optional post gate for absolute-frequency physiology (phase locking, ERB).

use crate::core::erb::erb_bw_hz;
use crate::core::fft::linear_convolve_fft;
use crate::core::log2::Log2Space;

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

/// Parameters for the multi-octave kernel.
#[derive(Clone, Copy, Debug)]
pub struct HarmonicityParams {
    /// Base Gaussian width per ratio [cents].
    pub sigma_cents: f32,
    /// Weight ~ (m+n)^(-gamma).
    pub gamma_ratio: f32,
    /// Normalize kernel to max=1.
    pub normalize_kernel: bool,
    /// Max numerator for ratio set.
    pub max_num: u32,
    /// Max denominator for ratio set.
    pub max_den: u32,
    /// Optional (m+n) limit.
    pub max_complexity: Option<u32>,
    /// Weight for 1/1 (0..1)
    pub identity_weight: f32,

    /// Kernel half span in octaves (support is [-span, +span]).
    pub span_octaves: f32,
    /// Exponential decay by octave distance: exp(-alpha*|μ|).
    pub alpha_oct_decay: f32,
    /// Extra weight for harmonic side (m>=n): m^-p_harm.
    pub p_harm: f32,
    /// Extra weight for subharmonic side (n>m): n^-q_sub.
    pub q_sub: f32,
    /// Broadening by order: σ = σ0 * sqrt(1 + max(m,n)/k0).
    pub k0_for_sigma: u32,
    /// If false, drop subharmonics (n>m).
    pub allow_subharmonics: bool,

    /// Apply abs-frequency gating
    pub freq_gate: bool,
    /// Phase-locking roll-off pivot [Hz].
    pub tfs_f_pl_hz: f32,
    /// Phase-locking roll-off steepness.
    pub tfs_eta: f32,
    ///// ERB gate exponent; gain *= (1/ERB(f))^rho.
    //pub erb_rho: f32,
}

impl Default for HarmonicityParams {
    fn default() -> Self {
        Self {
            sigma_cents: 7.0,
            gamma_ratio: 0.9,
            normalize_kernel: true,
            max_num: 16,
            max_den: 16,
            max_complexity: Some(15),
            identity_weight: 0.5,

            span_octaves: 5.0,
            alpha_oct_decay: 0.35,
            p_harm: 0.5,
            q_sub: 1.0,
            k0_for_sigma: 4,
            allow_subharmonics: true,

            freq_gate: false,
            tfs_f_pl_hz: 4500.0,
            tfs_eta: 4.0,
        }
    }
}

/// Multi-octave integer-ratio kernel for harmonicity.
#[derive(Clone, Debug)]
pub struct HarmonicityKernel {
    pub bins_per_oct: u32,
    pub params: HarmonicityParams,
    /// Centered kernel samples over Δlog2 ∈ [-span, +span].
    kernel: Vec<f32>,
    /// Index of Δ=0 inside `kernel`.
    center_idx: usize,
    /// Debug info.
    ratio_mu_log2: Vec<f32>,
    ratio_weights: Vec<f32>,
    ratio_sigmas_log2: Vec<f32>,
}

impl HarmonicityKernel {
    pub fn new(space: &Log2Space, params: HarmonicityParams) -> Self {
        let (ratio_mu_log2, ratio_weights, ratio_sigmas_log2) =
            Self::build_ratio_set_multi_oct(&params);
        let (kernel, center_idx) = Self::build_kernel_multi_oct(
            space.bins_per_oct,
            params.span_octaves,
            &ratio_mu_log2,
            &ratio_weights,
            &ratio_sigmas_log2,
            params.normalize_kernel,
        );
        Self {
            bins_per_oct: space.bins_per_oct,
            params,
            kernel,
            center_idx,
            ratio_mu_log2,
            ratio_weights,
            ratio_sigmas_log2,
        }
    }

    /// Build integer-ratio set without folding to [1,2).
    /// μ = log2(m/n) kept as-is within ±span.
    fn build_ratio_set_multi_oct(params: &HarmonicityParams) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        use std::collections::HashMap;

        let mut mu = Vec::new();
        let mut w = Vec::new();
        let mut sg = Vec::new();

        // Micro-cent quantization for dedup.
        let quant = 1200.0_f32 * 1000.0_f32; // 0.001 cent
        let mut chosen: HashMap<i64, (usize, u32)> = HashMap::new();

        for m in 1..=params.max_num {
            for n in 1..=params.max_den {
                if !params.allow_subharmonics && n > m {
                    continue;
                }
                let g = gcd_u32(m, n);
                let (rm, rn) = (m / g, n / g);
                let cmplx = rm + rn;
                if let Some(limit) = params.max_complexity {
                    if cmplx > limit {
                        continue;
                    }
                }
                let mu0 = log2f(rm as f32) - log2f(rn as f32);
                if mu0.abs() > params.span_octaves {
                    continue;
                }

                // Weights.
                let base = (cmplx as f32).powf(-params.gamma_ratio);
                let dir = if rm >= rn {
                    (rm as f32).powf(-params.p_harm)
                } else {
                    (rn as f32).powf(-params.q_sub)
                };
                let dist = (-params.alpha_oct_decay * mu0.abs()).exp();
                let is_identity = rm == 1 && rn == 1;
                let mut w_all = base * dir * dist;
                if is_identity {
                    w_all *= params.identity_weight;
                }

                // Width (log2).
                let k = rm.max(rn) as f32;
                let sig_cents = params.sigma_cents * (1.0 + k / params.k0_for_sigma as f32).sqrt();
                let sig_log2 = sig_cents / 1200.0;

                // Dedup by μ; prefer smaller complexity.
                let code = (mu0 * quant).round() as i64;
                if let Some((idx, best_c)) = chosen.get(&code).cloned() {
                    if cmplx < best_c {
                        mu[idx] = mu0;
                        w[idx] = w_all;
                        sg[idx] = sig_log2;
                        chosen.insert(code, (idx, cmplx));
                    }
                } else {
                    let idx = mu.len();
                    mu.push(mu0);
                    w.push(w_all);
                    sg.push(sig_log2);
                    chosen.insert(code, (idx, cmplx));
                }
            }
        }

        (mu, w, sg)
    }

    /// Build centered kernel samples over Δlog2 grid.
    fn build_kernel_multi_oct(
        bins_per_oct: u32,
        span_oct: f32,
        mu: &[f32],
        w: &[f32],
        sig: &[f32],
        normalize: bool,
    ) -> (Vec<f32>, usize) {
        let step = 1.0 / bins_per_oct as f32;
        let m = (2.0 * span_oct / step).round() as usize + 1;
        let center = m / 2;

        let mut k = vec![0.0f32; m];
        for i in 0..m {
            let d = (i as isize - center as isize) as f32 * step; // Δlog2
            let mut acc = 0.0f32;
            for j in 0..mu.len() {
                let dd = d - mu[j];
                let two_s2 = 2.0 * sig[j] * sig[j];
                acc += w[j] * (-(dd * dd) / two_s2).exp();
            }
            k[i] = acc.max(0.0);
        }

        if normalize {
            if let Some(mx) = k.iter().cloned().reduce(f32::max) {
                if mx > 0.0 {
                    for v in &mut k {
                        *v /= mx;
                    }
                }
            }
        }
        (k, center)
    }

    /// Absolute-frequency gate (phase locking, ERB).
    #[inline]
    fn absfreq_gate(f_hz: f32, p: &HarmonicityParams) -> f32 {
        // TFS roll-off
        let g_pl = 1.0 / (1.0 + (f_hz / p.tfs_f_pl_hz).powf(p.tfs_eta.max(0.1)));
        g_pl.clamp(0.0, 1.0)
    }

    pub fn potential_h_from_log2_spectrum(
        &self,
        envelope: &[f32],
        space: &Log2Space,
    ) -> (Vec<f32>, f32) {
        assert_eq!(space.bins_per_oct, self.bins_per_oct);
        assert_eq!(space.n_bins(), envelope.len());

        let y_full = crate::core::fft::linear_convolve_fft(envelope, &self.kernel);
        let start = self.center_idx;
        let end = start + envelope.len();
        assert!(
            y_full.len() >= end,
            "linear_convolve_fft must return full length (N+M-1)"
        );

        let mut y = y_full[start..end].to_vec();

        // level invariance
        let denom = envelope.iter().sum::<f32>().max(1e-12);
        let norm = 1.0 / denom;
        let p = self.params;
        if p.freq_gate {
            for (i, v) in y.iter_mut().enumerate() {
                let gate = Self::absfreq_gate(space.freq_of_index(i), &p);
                *v *= norm * gate;
            }
        } else {
            for v in &mut y {
                *v *= norm;
            }
        }
        (y, denom)
    }

    /// Kernel accessor.
    pub fn kernel(&self) -> &[f32] {
        &self.kernel
    }

    /// Debug: (μ_log2, weights, σ_log2).
    pub fn ratios_debug(&self) -> (&[f32], &[f32], &[f32]) {
        (
            &self.ratio_mu_log2,
            &self.ratio_weights,
            &self.ratio_sigmas_log2,
        )
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
    fn kernel_is_centered_and_nonnegative() {
        let space = Log2Space::new(110.0, 3520.0, 48);
        let hk = HarmonicityKernel::new(&space, HarmonicityParams::default());
        let k = hk.kernel();
        assert!(k.len() % 2 == 1);
        assert!(k.iter().all(|&v| v >= 0.0));
        if hk.params.normalize_kernel {
            let mx = k.iter().fold(0.0, |a: f32, &b| a.max(b));
            assert!((mx - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn delta_input_reproduces_kernel_shape_same_mode() {
        let space = Log2Space::new(110.0, 3520.0, 48);
        let mut p = HarmonicityParams::default();
        p.freq_gate = false;
        let hk = HarmonicityKernel::new(&space, p);
        let mut env = vec![0.0f32; space.n_bins()];
        let mid = space.n_bins() / 2;
        env[mid] = 1.0;

        let (y, denom) = hk.potential_h_from_log2_spectrum(&env, &space);
        assert!((denom - 1.0f32).abs() < 1e-6);

        // Around the impulse, y should look like the centered kernel.
        let k = hk.kernel();
        let c = hk.center_idx;
        let win = (k.len() / 2).min(32);
        for d in 0..win {
            let i = mid + d;
            let j = mid - d;
            if i < y.len() {
                assert!((y[i] - k[c + d]).abs() < 1e-3);
            }
            if j < y.len() {
                assert!((y[j] - k[c - d]).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn ratio_set_contains_common_intervals() {
        let space = Log2Space::new(55.0, 3520.0, 48);
        let hk = HarmonicityKernel::new(&space, HarmonicityParams::default());
        let (mu, _w, _s) = hk.ratios_debug();
        let targets = [1.0, 3.0 / 2.0, 4.0 / 3.0, 5.0 / 4.0, 6.0 / 5.0, 5.0 / 3.0];
        for r in targets {
            let m = log2f(r);
            let ok = mu
                .iter()
                .any(|&u| (u - m).abs() < 1e-6 || (u + m).abs() < 1e-6);
            assert!(ok, "missing ratio for log2({:.5})", r);
        }
    }

    #[test]
    fn tfs_gate_expected_values() {
        let mut p = HarmonicityParams::default();
        p.freq_gate = true;
        p.tfs_f_pl_hz = 4500.0;
        p.tfs_eta = 4.0;

        let g1 = super::HarmonicityKernel::absfreq_gate(4500.0, &p);
        let g2 = super::HarmonicityKernel::absfreq_gate(9000.0, &p);

        assert!((g1 - 0.5).abs() < 1e-6); // knee = 0.5
        assert!(g2 < 0.07); // 9kHz≈1/(1+16)=0.0588
    }

    #[test]
    fn freq_gate_reduces_high_freq_when_enabled() {
        let mut p = HarmonicityParams::default();
        p.freq_gate = true;
        let space = Log2Space::new(50.0, 8000.0, 48);
        let hk = HarmonicityKernel::new(&space, p);

        let mut lo = vec![0.0f32; space.n_bins()];
        lo[space.index_of_freq(440.0).unwrap()] = 1.0;
        let peak_lo = hk
            .potential_h_from_log2_spectrum(&lo, &space)
            .0
            .iter()
            .cloned()
            .fold(0.0, f32::max);

        let mut hi = vec![0.0f32; space.n_bins()];
        hi[space.index_of_freq(5000.0).unwrap()] = 1.0;
        let peak_hi = hk
            .potential_h_from_log2_spectrum(&hi, &space)
            .0
            .iter()
            .cloned()
            .fold(0.0, f32::max);

        assert!(peak_hi < peak_lo);
    }

    // ---------- Plot tests (ignored by default) ----------

    #[test]
    #[ignore]
    fn plot_kernel_shape_png() {
        // Plot K(Δlog2) over the designed span.
        let space = Log2Space::new(20.0, 8000.0, 200);
        let hk = HarmonicityKernel::new(&space, HarmonicityParams::default());
        let k = hk.kernel().to_vec();
        let step = 1.0 / hk.bins_per_oct as f32;
        let d_log2: Vec<f32> = (0..k.len())
            .map(|i| (i as i32 - hk.center_idx as i32) as f32 * step)
            .collect();

        let out_path = Path::new("target/test_h_kernel_shape.png");
        let root = BitMapBackend::new(out_path, (1600, 1000)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let maxy = k.iter().cloned().fold(0.0, f32::max) * 1.1;
        let mut chart = ChartBuilder::on(&root)
            .caption("Harmonicity Kernel K(Δlog2)", ("sans-serif", 30))
            .margin(10)
            .build_cartesian_2d(d_log2[0]..d_log2[d_log2.len() - 1], 0.0f32..maxy)
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("Δlog2 (octaves)")
            .y_desc("Amplitude")
            .draw()
            .unwrap();

        chart
            .draw_series(LineSeries::new(
                d_log2.iter().zip(k.iter()).map(|(&x, &y)| (x, y)),
                &BLUE,
            ))
            .unwrap();

        root.present().unwrap();
        assert!(File::open(out_path).is_ok());
    }

    #[test]
    #[ignore]
    fn compare_build_kernel_and_eval_kernel_shape() -> Result<(), Box<dyn std::error::Error>> {
        // Compare discrete samples with analytic sum of Gaussians from (μ,w,σ).
        let space = Log2Space::new(20.0, 8000.0, 200);
        let hk = HarmonicityKernel::new(&space, HarmonicityParams::default());
        let k_discrete = hk.kernel().to_vec();
        let (mu, w, sig) = hk.ratios_debug();
        let step = 1.0 / hk.bins_per_oct as f32;

        let d_log2: Vec<f32> = (0..k_discrete.len())
            .map(|i| (i as i32 - hk.center_idx as i32) as f32 * step)
            .collect();
        let mut k_eval = vec![0.0f32; d_log2.len()];
        for (i, &x) in d_log2.iter().enumerate() {
            let mut acc = 0.0f32;
            for j in 0..mu.len() {
                let dd = x - mu[j];
                let two_s2 = 2.0 * sig[j] * sig[j];
                acc += w[j] * (-(dd * dd) / two_s2).exp();
            }
            k_eval[i] = acc;
        }

        // Normalize and compute MAE.
        let s1: f32 = k_discrete.iter().sum();
        let s2: f32 = k_eval.iter().sum();
        let g1: Vec<f32> = k_discrete.iter().map(|&v| v / s1).collect();
        let g2: Vec<f32> = k_eval.iter().map(|&v| v / s2).collect();
        let mae = g1
            .iter()
            .zip(g2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / g1.len() as f32;
        assert!(mae < 1e-6, "MAE={}", mae);

        // Plot overlay.
        let out_path = "target/test_h_kernel_build_vs_eval.png";
        let root = BitMapBackend::new(out_path, (1600, 1000)).into_drawing_area();
        root.fill(&WHITE)?;
        let maxy = g1
            .iter()
            .cloned()
            .fold(0.0, f32::max)
            .max(g2.iter().cloned().fold(0.0, f32::max))
            * 1.1;
        let mut chart = ChartBuilder::on(&root)
            .caption("Discrete K vs Analytic Sum", ("sans-serif", 30))
            .margin(10)
            .build_cartesian_2d(
                d_log2.first().copied().unwrap()..d_log2.last().copied().unwrap(),
                0.0f32..maxy,
            )?;
        chart
            .configure_mesh()
            .x_desc("Δlog2 (octaves)")
            .y_desc("Normalized amplitude")
            .draw()?;
        chart
            .draw_series(LineSeries::new(
                d_log2.iter().zip(g1.iter()).map(|(&x, &y)| (x, y)),
                &BLUE,
            ))?
            .label("discrete")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));
        chart
            .draw_series(LineSeries::new(
                d_log2.iter().zip(g2.iter()).map(|(&x, &y)| (x, y)),
                &RED,
            ))?
            .label("analytic")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));
        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
        root.present()?;
        assert!(Path::new(out_path).exists());
        Ok(())
    }

    #[test]
    #[ignore]
    fn plot_freq_gate_gain_curve_png() -> Result<(), Box<dyn std::error::Error>> {
        use plotters::prelude::*;
        use std::fs::File;
        use std::path::Path;

        let space = Log2Space::new(20.0, 10_000.0, 96);

        // gate OFF
        let mut p_off = HarmonicityParams::default();
        p_off.freq_gate = false;
        let hk_off = HarmonicityKernel::new(&space, p_off);

        // gate ON
        let mut p_on = HarmonicityParams::default();
        p_on.freq_gate = true;
        let hk_on = HarmonicityKernel::new(&space, p_on);

        let n = space.n_bins();
        let mut env = vec![0.0f32; n];

        let mut xs_hz = Vec::with_capacity(n);
        let mut y_off = Vec::with_capacity(n);
        let mut y_on = Vec::with_capacity(n);

        for i in 0..n {
            env.iter_mut().for_each(|x| *x = 0.0);
            env[i] = 1.0;

            let (h0, _) = hk_off.potential_h_from_log2_spectrum(&env, &space);
            let (h1, _) = hk_on.potential_h_from_log2_spectrum(&env, &space);

            xs_hz.push(space.freq_of_index(i));
            y_off.push(h0[i]);
            y_on.push(h1[i]);
        }

        let out_path = "target/test_h_freq_gate_curve.png";
        let root = BitMapBackend::new(out_path, (1600, 1000)).into_drawing_area();
        root.fill(&WHITE)?;
        let max_y = y_off.iter().chain(y_on.iter()).copied().fold(0.0, f32::max) * 1.1;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Abs-frequency Gate Curve (center response)",
                ("sans-serif", 30),
            )
            .margin(10)
            .build_cartesian_2d(xs_hz[0]..xs_hz[xs_hz.len() - 1], 0.0f32..max_y)?;

        chart
            .configure_mesh()
            .x_desc("Frequency [Hz]")
            .y_desc("Center response")
            .draw()?;

        chart
            .draw_series(LineSeries::new(
                xs_hz.iter().zip(y_off.iter()).map(|(&x, &y)| (x, y)),
                &BLUE,
            ))?
            .label("freq_gate = false")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

        chart
            .draw_series(LineSeries::new(
                xs_hz.iter().zip(y_on.iter()).map(|(&x, &y)| (x, y)),
                &RED,
            ))?
            .label("freq_gate = true")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
        root.present()?;
        assert!(File::open(out_path).is_ok());
        Ok(())
    }

    #[test]
    #[ignore]
    fn plot_potential_h_from_log2_spectrum_delta_input() -> Result<(), Box<dyn std::error::Error>> {
        // δ input should reproduce the kernel shape in "same" mode.
        let space = Log2Space::new(20.0, 8000.0, 200);
        let mut p = HarmonicityParams::default();
        p.freq_gate = false; // ensure no gating
        let hk = HarmonicityKernel::new(&space, p);
        let mut env = vec![0.0f32; space.n_bins()];
        let mid = env.len() / 2;
        env[mid] = 1.0;

        let (h_vec, _) = hk.potential_h_from_log2_spectrum(&env, &space);

        let step = 1.0 / hk.bins_per_oct as f32;
        let d_log2_field: Vec<f32> = (0..env.len())
            .map(|i| (i as i32 - mid as i32) as f32 * step)
            .collect();

        // Normalize to compare shapes.
        let max_h = h_vec.iter().cloned().fold(0.0, f32::max);
        let h_norm: Vec<f32> = h_vec.iter().map(|&v| v / max_h).collect();

        let k = hk.kernel();
        let d_log2_kern: Vec<f32> = (0..k.len())
            .map(|i| (i as i32 - hk.center_idx as i32) as f32 * step)
            .collect();
        let max_k = k.iter().cloned().fold(0.0, f32::max);
        let k_norm: Vec<f32> = k.iter().map(|&v| v / max_k).collect();

        let out_path = "target/test_h_potential_from_log2_delta.png";
        let root = BitMapBackend::new(out_path, (1600, 1000)).into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Potential H from Log2 Spectrum (δ input)",
                ("sans-serif", 30),
            )
            .margin(10)
            .build_cartesian_2d(
                d_log2_field[d_log2_field.len() / 2 - 800]
                    ..d_log2_field[d_log2_field.len() / 2 + 800],
                0.0f32..1.05f32,
            )?;
        chart
            .configure_mesh()
            .x_desc("Δlog2 (octaves)")
            .y_desc("Normalized amplitude")
            .draw()?;
        chart
            .draw_series(LineSeries::new(
                d_log2_field
                    .iter()
                    .zip(h_norm.iter())
                    .map(|(&x, &y)| (x, y)),
                &BLUE,
            ))?
            .label("H(log2 δ)")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));
        chart
            .draw_series(LineSeries::new(
                d_log2_kern.iter().zip(k_norm.iter()).map(|(&x, &y)| (x, y)),
                &GREEN,
            ))?
            .label("Kernel K(Δlog2)")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &GREEN));
        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
        root.present()?;
        assert!(Path::new(out_path).exists());
        Ok(())
    }
}
