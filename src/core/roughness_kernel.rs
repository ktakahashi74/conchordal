//! core/roughness/kernelconv.rs — Roughness R via ERB-domain kernel convolution.
//! Computes frequency-space roughness landscape by convolving the cochlear
//! envelope energy along the ERB axis using an asymmetric kernel.
//!
//! Designed for biologically-plausible spatial interference modeling rather than
//! time-domain envelope modulation.  Used in Landscape::process_block() for
//! RVariant::KernelConv.

use crate::core::erb::ErbSpace;

use rustfft::{FftPlanner, num_complex::Complex32};

/// Parameters defining asymmetric ERB-domain kernel g(ΔERB).
#[derive(Clone, Copy, Debug)]
pub struct KernelParams {
    /// Gaussian width (low-side spread)
    pub sigma_erb: f32,
    /// High-side exponential tail constant
    pub tau_erb: f32,
    /// Mix ratio between Gaussian core and exponential tail [0..1]
    pub mix_tail: f32,
    /// Kernel radius in ERB units
    pub half_width_erb: f32,

    pub suppress_sigma_erb: f32,
    pub suppress_pow: f32,
}

impl Default for KernelParams {
    fn default() -> Self {
        Self {
            sigma_erb: 0.6,
            tau_erb: 1.2,
            mix_tail: 0.25,
            half_width_erb: 4.0,
            suppress_sigma_erb: 0.15,
            suppress_pow: 1.5,
        }
    }
}
fn build_kernel(params: &KernelParams, erb_step: f32) -> (Vec<f32>, usize) {
    let hw = (params.half_width_erb / erb_step).ceil() as i32;
    let len = (2 * hw + 1) as usize;
    let mut g = vec![0.0f32; len];

    for i in 0..len {
        let d_idx = i as i32 - hw;
        let d_erb = d_idx as f32 * erb_step;
        let g_gauss = (-(d_erb * d_erb) / (params.sigma_erb * params.sigma_erb)).exp();
        let g_tail = if d_erb >= 0.0 {
            (-d_erb / params.tau_erb).exp()
        } else {
            0.0
        };
        let base = (1.0 - params.mix_tail) * g_gauss + params.mix_tail * g_tail;

        let s = params.suppress_sigma_erb.max(1e-6);
        let suppress = 1.0 - (-(d_erb * d_erb) / (s * s)).exp(); // 0..1
        let suppress = suppress.powf(params.suppress_pow); // 中心ゼロを強める
        g[i] = base * suppress;
    }

    // L1 normalize
    let sum: f32 = g.iter().sum();
    if sum > 0.0 {
        for v in &mut g {
            *v /= sum;
        }
    }
    (g, hw as usize)
}

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

    // scale by n_fft (rustfft inverse is unnormalized)
    let scale = 1.0 / n_fft as f32;
    for xi in &mut X {
        *xi *= scale;
    }

    // Centered “same”: take indices [ (m-1)/2 .. (m-1)/2 + n )
    let offset = (m - 1) / 2;
    (0..n).map(|i| X[i + offset].re).collect()
}

/// Compute roughness R via ERB-domain kernel convolution.
///
/// - `e_ch`: per-channel envelope energy (≥0) on an ERB-uniform grid.
/// - `erb_space`: ErbSpace defining the grid
/// - `params`: kernel parameters
///
/// Returns per-channel R values.
pub fn compute_r_kernelconv(e_ch: &[f32], erb_space: &ErbSpace, params: &KernelParams) -> Vec<f32> {
    let n_ch = e_ch.len();
    if n_ch == 0 {
        return vec![];
    }

    // 1. Build kernel (ΔERB domain)
    let (g, _) = build_kernel(params, erb_space.erb_step);

    // 2. Convolve envelope along ERB axis
    let r_conv = fft_convolve_same(e_ch, &g);

    // 3. Local weighting by sqrt(energy)
    let mut r_ch = Vec::with_capacity(n_ch);
    for (r, &e) in r_conv.iter().zip(e_ch) {
        r_ch.push(r * (e.abs() + 1e-12).sqrt());
    }
    r_ch
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::erb::ErbSpace;

    #[test]
    fn kernel_is_l1_normalized() {
        let (g, _) = build_kernel(&KernelParams::default(), 0.1);
        let sum: f32 = g.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "kernel sum = {sum}");
    }

    #[test]
    fn kernel_has_center_zero_and_offcenter_peak() {
        let (g, _) = build_kernel(&KernelParams::default(), 0.05);
        let mid = g.len() / 2;

        // 中心はゼロ近傍（中心抑圧）
        assert!(g[mid] < 1e-4, "center not ~zero: {}", g[mid]);

        // オフセンタにピークが存在（中心から2..25binのどこか）
        let search_lo = mid.saturating_sub(25);
        let search_hi = (mid + 25).min(g.len() - 1);
        let (imax, vmax) = (search_lo..=search_hi)
            .filter(|&i| i != mid)
            .map(|i| (i, g[i]))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        assert!(imax != mid, "peak at center unexpectedly");
        assert!(vmax > g[mid.saturating_sub(1)] && vmax > g[(mid + 1).min(g.len() - 1)]);
    }

    #[test]
    fn impulse_response_has_center_zero_not_peak() {
        let n = 129;
        let mut x = vec![0.0f32; n];
        x[n / 2] = 1.0;
        let (g, _) = build_kernel(&KernelParams::default(), 0.05);
        let y = fft_convolve_same(&x, &g);
        let mid = n / 2;

        // 畳み込み出力はカーネル形（中心ゼロ）
        assert!(y[mid].abs() < 1e-4, "conv center not ~zero: {}", y[mid]);

        // 中心の近傍でピークは立たない（ゼロ抑圧が効いている）
        assert!(y[mid] < y[mid.saturating_sub(2)]);
        assert!(y[mid] < y[(mid + 2).min(y.len() - 1)]);
    }

    #[test]
    fn zero_energy_returns_zero() {
        let erb_space = ErbSpace::new(100.0, 5000.0, 0.25);
        let e = vec![0.0; erb_space.n_bins()];
        let r = compute_r_kernelconv(&e, &erb_space, &KernelParams::default());
        assert!(r.iter().all(|&v| v.abs() < 1e-9));
    }

    #[test]
    fn narrowband_energy_yields_local_peak() {
        // Energy between 400–600 Hz → local peak expected near that band
        let erb_space = ErbSpace::new(100.0, 5000.0, 0.25);
        let e: Vec<f32> = erb_space
            .freqs_hz()
            .iter()
            .map(|&f| if (f > 400.0 && f < 600.0) { 1.0 } else { 0.0 })
            .collect();
        let r = compute_r_kernelconv(&e, &erb_space, &KernelParams::default());
        let imax = r
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;
        let fmax = erb_space.freqs_hz()[imax];
        assert!((fmax > 300.0) && (fmax < 700.0));
    }

    #[test]
    fn stable_under_small_perturbations() {
        // Small random noise shouldn't cause large changes
        let erb_space = ErbSpace::new(100.0, 5000.0, 0.25);
        let mut e: Vec<f32> = erb_space
            .freqs_hz()
            .iter()
            .map(|&f| (f > 500.0 && f < 1500.0) as i32 as f32)
            .collect();
        let r1 = compute_r_kernelconv(&e, &erb_space, &KernelParams::default());
        for v in &mut e {
            *v += (rand::random::<f32>() - 0.5) * 1e-3;
        }
        let r2 = compute_r_kernelconv(&e, &erb_space, &KernelParams::default());
        let diff: f32 =
            r1.iter().zip(&r2).map(|(a, b)| (a - b).abs()).sum::<f32>() / r1.len() as f32;
        assert!(diff < 1e-2, "unstable output diff={diff}");
    }

    #[test]
    fn kernel_zero_at_center() {
        let (g, _) = build_kernel(&KernelParams::default(), 0.05);
        let mid = g.len() / 2;
        // 中心ゼロ性
        assert!(g[mid] < 1e-4, "center not zero enough: {}", g[mid]);

        // “ピーク値が十分大きいか”は絶対値でなく相対で判定
        let vmax = g.iter().cloned().fold(f32::MIN, f32::max);
        let vmed = {
            let mut t = g.clone();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            t[t.len() / 2]
        };
        assert!(
            vmax > 5.0 * vmed.max(1e-6),
            "peak too flat: vmax={} vmed={}",
            vmax,
            vmed
        );
    }
}
