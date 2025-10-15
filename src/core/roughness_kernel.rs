//! core/roughness_kernel.rs — Roughness R via ERB-domain kernel convolution.
//! Computes frequency-space roughness landscape by convolving the cochlear
//! envelope energy along the ERB axis using an asymmetric kernel.
//!
//! Designed for biologically-plausible spatial interference modeling rather than
//! time-domain envelope modulation.  Used in Landscape::process_block() for
//! RVariant::KernelConv.

use crate::core::erb::ErbSpace;
use rustfft::{FftPlanner, num_complex::Complex32};
use std::sync::OnceLock;

// ======================================================================
// Kernel definition (Plomp–Levelt inspired, biologically extended)
// ======================================================================

/// Parameters defining two-layer asymmetric ERB-domain roughness kernel g(ΔERB).
/// Layer 1 = Cochlear-level (narrow, with center suppression)
/// Layer 2 = Neural-level (broader lateral inhibition)
#[derive(Clone, Copy, Debug)]
pub struct KernelParams {
    // === Layer 1: Cochlear ===
    /// Gaussian width (low-side spread)
    pub sigma_erb: f32,
    /// High-side exponential tail constant
    pub tau_erb: f32,
    /// Mix ratio between Gaussian core and exponential tail [0..1]
    pub mix_tail: f32,
    /// Kernel radius in ERB units
    pub half_width_erb: f32,
    /// Width of central suppression zone (ΔERB for fusion)
    pub suppress_sigma_erb: f32,
    /// Exponent controlling steepness of suppression rise
    pub suppress_pow: f32,

    // === Layer 2: Neural ===
    /// Broad Gaussian width of neural lateral inhibition (ΔERB scale)
    pub sigma_neural_erb: f32,
    /// Weight of neural-level contribution [0..1]
    pub w_neural: f32,
}

impl Default for KernelParams {
    fn default() -> Self {
        Self {
            // --- Cochlear layer ---
            sigma_erb: 0.45,
            tau_erb: 1.0,
            mix_tail: 0.20,
            half_width_erb: 4.0,
            suppress_sigma_erb: 0.10,
            suppress_pow: 2.0,
            // --- Neural layer ---
            sigma_neural_erb: 1.0,
            w_neural: 0.3,
        }
    }
}

/// Build two-layer asymmetric ERB-domain kernel g(ΔERB).
/// Returns (kernel vector, half-width in bins).
pub fn build_kernel(params: &KernelParams, erb_step: f32) -> (Vec<f32>, usize) {
    let hw = (params.half_width_erb / erb_step).ceil() as i32;
    let len = (2 * hw + 1) as usize;
    let mut g = vec![0.0f32; len];

    // --- Precompute constants ---
    let sigma = params.sigma_erb.max(1e-6);
    let tau = params.tau_erb.max(1e-6);
    let s_sup = params.suppress_sigma_erb.max(1e-6);
    let sigma_neural = params.sigma_neural_erb.max(1e-6);

    // --- Layer 1: cochlear narrow asymmetric kernel ---
    let mut g_coch = vec![0.0f32; len];
    for i in 0..len {
        let d_idx = i as i32 - hw;
        let d_erb = d_idx as f32 * erb_step;

        // Gaussian + exponential tail
        let g_gauss = (-(d_erb * d_erb) / (2.0 * sigma * sigma)).exp();
        let g_tail = if d_erb >= 0.0 {
            (-d_erb / tau).exp()
        } else {
            0.0
        };
        let base = (1.0 - params.mix_tail) * g_gauss + params.mix_tail * g_tail;

        // Center suppression (fusion zone)
        // Normalized for erb_step independence
        let base_suppress = 1.0 - (-(d_erb * d_erb) / (2.0 * s_sup * s_sup)).exp();
        let norm_suppress = 1.0 - (-(erb_step * erb_step) / (2.0 * s_sup * s_sup)).exp();
        let suppress = (base_suppress / norm_suppress).powf(params.suppress_pow);

        //let suppress =
        //            (1.0 - (-(d_erb * d_erb) / (2.0 * s_sup * s_sup)).exp()).powf(params.suppress_pow);

        g_coch[i] = base * suppress;
    }
    let sum_coch: f32 = g_coch.iter().sum();
    if sum_coch > 0.0 {
        for v in &mut g_coch {
            *v /= sum_coch;
        }
    }

    // --- Layer 2: neural broad Gaussian ---
    let mut g_neural = vec![0.0f32; len];
    for i in 0..len {
        let d_idx = i as i32 - hw;
        let d_erb = d_idx as f32 * erb_step;
        g_neural[i] = (-(d_erb * d_erb) / (2.0 * sigma_neural * sigma_neural)).exp();
    }
    let sum_neural: f32 = g_neural.iter().sum();
    if sum_neural > 0.0 {
        for v in &mut g_neural {
            *v /= sum_neural;
        }
    }

    // --- Weighted combination ---
    for i in 0..len {
        g[i] = (1.0 - params.w_neural) * g_coch[i] + params.w_neural * g_neural[i];
    }

    // --- Normalize total ---
    let sum: f32 = g.iter().sum();
    if sum > 0.0 {
        for v in &mut g {
            *v /= sum;
        }
    }

    (g, hw as usize)
}

// ======================================================================
// FFT convolution utilities
// ======================================================================

/// Convolution (same size) on ERB grid using FFT (real-only).
fn fft_convolve_same(x: &[f32], h: &[f32]) -> Vec<f32> {
    let n = x.len();
    let m = h.len();
    let n_conv = n + m - 1;
    let n_fft = n_conv.next_power_of_two();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    let ifft = planner.plan_fft_inverse(n_fft);

    let mut X = vec![Complex32::new(0.0, 0.0); n_fft];
    let mut H = vec![Complex32::new(0.0, 0.0); n_fft];

    for (i, &v) in x.iter().enumerate() {
        X[i].re = v;
    }
    for (i, &v) in h.iter().enumerate() {
        H[i].re = v;
    }

    fft.process(&mut X);
    fft.process(&mut H);
    for i in 0..n_fft {
        X[i] *= H[i];
    }
    ifft.process(&mut X);

    // --- scale (rustfft inverse is unnormalized)
    let scale = 1.0 / n_fft as f32;
    for xi in &mut X {
        *xi *= scale;
    }

    // --- Centered “same”: take indices [ (m-1)/2 .. (m-1)/2 + n )
    let offset = (m - 1) / 2;
    (0..n).map(|i| X[i + offset].re).collect()
}

// ======================================================================
// Public API
// ======================================================================

/// Compute roughness R via ERB-domain kernel convolution.
///
/// - `e_ch`: per-channel envelope energy (≥0) on an ERB-uniform grid.
/// - `erb_space`: ErbSpace defining the grid
/// - `params`: kernel parameters
///
/// Returns per-channel R values (roughness potential).
pub fn compute_r_kernelconv(e_ch: &[f32], erb_space: &ErbSpace, params: &KernelParams) -> Vec<f32> {
    let n_ch = e_ch.len();
    if n_ch == 0 {
        return vec![];
    }

    // 1. Build kernel (ΔERB domain)
    let (g, _) = build_kernel(params, erb_space.erb_step);

    // 2. Convolve envelope along ERB axis
    let r_conv = fft_convolve_same(e_ch, &g);

    // 3. Local weighting (sqrt of energy, approximating loudness compression)
    let mut r_ch = Vec::with_capacity(n_ch);
    for (r, &e) in r_conv.iter().zip(e_ch) {
        r_ch.push(r * (e.abs() + 1e-12).powf(0.5)); // gamma=0.5 ≈ Fechner
    }
    r_ch
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::erb::ErbSpace;

    /// Shared ERB step for all tests (numerical resolution)
    const ERB_STEP: f32 = 0.005;

    /// Helper: build kernel with shared ERB step
    fn make_kernel() -> (Vec<f32>, usize) {
        let p = KernelParams::default();
        build_kernel(&p, ERB_STEP)
    }

    /// 1. Kernel must be positive, centered, and decay to zero at edges
    #[test]
    fn kernel_is_positive_and_centered() {
        let (g, hw) = make_kernel();
        assert!(g.iter().all(|&v| v >= 0.0), "kernel must be non-negative");
        assert_eq!(g.len(), 2 * hw + 1, "length = 2*hw+1");
        let edge_mean = (g[0] + g[g.len() - 1]) * 0.5;
        assert!(edge_mean < 1e-4, "edges should decay toward zero");
    }

    /// 2. Center (ΔERB=0) should be strongly suppressed but nonzero
    #[test]
    fn kernel_center_is_suppressed_but_not_zero() {
        let (g, hw) = make_kernel();
        let center = g[hw];
        assert!(
            center > 0.0 && center < 0.005,
            "center should be small but >0"
        );
    }

    /// 3. Kernel should have peaks around ±0.3 ERB
    #[test]
    fn kernel_has_peak_near_pm0p3_erb() {
        let (g, hw) = make_kernel();
        let n = g.len();
        let d_erb: Vec<f32> = (0..n)
            .map(|i| (i as i32 - hw as i32) as f32 * ERB_STEP)
            .collect();

        // Find local maxima
        let mut pos_peak = None;
        let mut neg_peak = None;
        for i in 1..n - 1 {
            if g[i] > g[i - 1] && g[i] > g[i + 1] {
                if d_erb[i] > 0.0 && pos_peak.is_none() {
                    pos_peak = Some((d_erb[i], g[i]));
                }
                if d_erb[i] < 0.0 && neg_peak.is_none() {
                    neg_peak = Some((d_erb[i], g[i]));
                }
            }
        }

        let (pos_x, pos_val) = pos_peak.expect("no positive-side peak found");
        let (neg_x, neg_val) = neg_peak.expect("no negative-side peak found");

        assert!(
            pos_x > 0.2 && pos_x < 0.4,
            "positive peak should be near +0.3 ERB (got {:.2})",
            pos_x
        );
        assert!(
            neg_x < -0.2 && neg_x > -0.4,
            "negative peak should be near -0.3 ERB (got {:.2})",
            neg_x
        );
        assert!(
            pos_val > g[hw] * 1.5,
            "positive peak should exceed center by ≥1.5x"
        );
    }

    /// 4. Positive side (ΔERB>0) should be higher than negative side
    #[test]
    fn kernel_is_asymmetric() {
        let (g, hw) = make_kernel();
        let pos = (hw as f32 + 0.3 / ERB_STEP).round() as usize;
        let neg = (hw as f32 - 0.3 / ERB_STEP).round() as usize;
        assert!(
            g[pos] > g[neg],
            "positive side should be higher than negative"
        );
    }

    /// 5. Kernel must be L1-normalized (area ≈ 1)
    #[test]
    fn kernel_l1_norm_is_one() {
        let (g, _) = make_kernel();
        let sum: f32 = g.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "L1 norm must be ~1, got {}", sum);
    }

    /// 6. Neural layer should reduce center suppression
    #[test]
    fn neural_layer_reduces_center_suppression() {
        let mut p1 = KernelParams::default();
        p1.w_neural = 0.0;
        let (g1, hw) = build_kernel(&p1, ERB_STEP);
        let c1 = g1[hw];

        let mut p2 = KernelParams::default();
        p2.w_neural = 0.4;
        let (g2, _) = build_kernel(&p2, ERB_STEP);
        let c2 = g2[hw];

        assert!(c2 > c1, "neural layer should reduce center suppression");
    }

    /// 7. Kernel shape should remain stable across ERB steps
    #[test]
    fn kernel_stable_across_erbstep() {
        let p = KernelParams::default();
        let (g1, _) = build_kernel(&p, ERB_STEP);
        let (g2, _) = build_kernel(&p, 0.02);
        let hw1 = g1.len() / 2;
        let hw2 = g2.len() / 2;
        let ratio1 = g1[(hw1 as f32 + 0.3 / ERB_STEP).round() as usize] / g1[hw1];
        let ratio2 = g2[(hw2 as f32 + 0.3 / 0.02).round() as usize] / g2[hw2];
        assert!(
            (ratio1 - ratio2).abs() < 0.1,
            "kernel shape should be stable across ERB step"
        );
    }

    #[test]
    #[ignore]
    fn plot_kernel_shape_png() {
        use plotters::prelude::*;
        use std::fs::File;
        use std::path::Path;

        // --- Build kernel
        let params = KernelParams::default();
        let erb_step = 0.02;
        let (g, _) = super::build_kernel(&params, erb_step);
        let hw = (params.half_width_erb / erb_step).ceil() as i32;
        let d_erb: Vec<f32> = (-hw..=hw).map(|i| i as f32 * erb_step).collect();

        // --- Output path
        let out_path = Path::new("target/test_kernel_shape.png");
        let root = BitMapBackend::new(out_path, (1280, 960)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .caption("Asymmetric ERB-domain Roughness Kernel", ("sans-serif", 20))
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(
                d_erb[0]..d_erb[d_erb.len() - 1],
                -0.0f32..g.iter().cloned().fold(0.0, f32::max) * 1.1,
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
            .unwrap()
            .label("g(ΔERB)")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .background_style(&WHITE.mix(0.8))
            .draw()
            .unwrap();

        // --- finalize
        root.present().unwrap();
        assert!(
            File::open(out_path).is_ok(),
            "kernel shape PNG was not created"
        );
    }
}
