//! core/roughness_kernel.rs — Roughness R via ERB-domain kernel convolution.
//! Computes frequency-space roughness landscape by convolving the cochlear
//! envelope energy along the ERB axis using an asymmetric kernel.
//!
//! Designed for biologically-plausible spatial interference modeling rather than
//! time-domain envelope modulation.  Used in Landscape::process_block() for
//! RVariant::KernelConv.

use crate::core::erb::{ErbSpace, erb_bw_hz, hz_to_erb};
use crate::core::fft::fft_convolve_same;
use rustfft::{FftPlanner, num_complex::Complex32};

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

/// two-layer asymmetric ERB-domain kernel g(ΔERB).
#[inline]
fn eval_kernel_delta_erb(params: &KernelParams, d_erb: f32) -> f32 {
    // Continuous version of two-layer asymmetric kernel (no normalization here)
    let de = d_erb;
    let sigma = params.sigma_erb.max(1e-6);
    let tau = params.tau_erb.max(1e-6);
    let s_sup = params.suppress_sigma_erb.max(1e-6);
    let sig_n = params.sigma_neural_erb.max(1e-6);

    // === Cochlear base: Gaussian + (tail on *positive* side) ===
    // ΔERB > 0 → 高域側（f_j > f_i）では tail
    let g_gauss = (-(de * de) / (2.0 * sigma * sigma)).exp();
    let g_tail = if de >= 0.0 { (-de / tau).exp() } else { 0.0 };
    let base = (1.0 - params.mix_tail) * g_gauss + params.mix_tail * g_tail;

    // Center suppression (fusion zone)
    let suppress = 1.0 - (-(de * de) / (2.0 * s_sup * s_sup)).exp();
    let g_coch = base * suppress.powf(params.suppress_pow);

    // Neural broad Gaussian
    let g_neural = (-(de * de) / (2.0 * sig_n * sig_n)).exp();

    (1.0 - params.w_neural) * g_coch + params.w_neural * g_neural
}

/// Build discrete kernel g[ΔERB] sampled at given ERB step size.
pub fn build_kernel_erbstep(params: &KernelParams, erb_step: f32) -> (Vec<f32>, usize) {
    let half_width = params.half_width_erb;
    let n_side = (half_width / erb_step).ceil() as usize;
    let len = 2 * n_side + 1;

    // Sample continuous g(ΔERB)
    let mut g: Vec<f32> = (0..len)
        .map(|i| {
            let d_erb = (i as i32 - n_side as i32) as f32 * erb_step;
            eval_kernel_delta_erb(params, d_erb)
        })
        .collect();

    // L1 normalization: ∫g dΔERB ≈ Σg * step = 1
    let sum: f32 = g.iter().sum();
    if sum > 0.0 {
        let scale = 1.0 / sum;
        for v in &mut g {
            *v *= scale;
        }
    }

    (g, n_side)
}

#[inline]
fn lut_interp(lut: &[f32], step: f32, hw: usize, d_erb: f32) -> f32 {
    let t = d_erb / step + hw as f32; // ΔERB=0 が中心
    let i = t.floor() as isize;
    if i < 0 || i as usize >= lut.len() - 1 {
        return 0.0;
    }
    let frac = t - i as f32;
    lut[i as usize] * (1.0 - frac) + lut[i as usize + 1] * frac
}

#[inline]
fn delta_erb(f1: f32, f2: f32) -> f32 {
    let f_center = 0.5 * (f1 + f2);
    let bw = erb_bw_hz(f_center);
    (f2 - f1).abs() / bw
}

/// Direct potential-R from PSD bins (no ERB resampling, no FFT).
/// psd_hz: power spectrum on linear frequency bins 0..fs/2 (rfft length).
/// fs: sampling rate.
/// Returns (R per bin aligned to psd bins, R_total scalar).
pub fn potential_r_from_psd_direct(
    psd_hz: &[f32],
    fs: f32,
    kparams: &KernelParams,
    gamma: f32,
    alpha: f32,
) -> (Vec<f32>, f32) {
    let n = psd_hz.len();
    if n == 0 {
        return (vec![], 0.0);
    }

    // rfft 由来の片側 N/2 本（Nyquist除外）を想定 ⇒ nfft = 2*n
    let nfft = 2 * n;
    let df = fs / nfft as f32;

    // === 周波数・圧縮エネルギー ===
    let mut f = vec![0.0f32; n];
    let mut e = vec![0.0f32; n];
    let eps = 1e-12f32;
    for i in 0..n {
        let fi = i as f32 * df;
        f[i] = fi;
        e[i] = (psd_hz[i] + eps).powf(gamma);
    }

    // === ERB-number 累積座標（厳密な ΔERB 積分用） ===
    let mut u = vec![0.0f32; n];
    for i in 1..n {
        let fmid = 0.5 * (f[i] + f[i - 1]);
        let du = (f[i] - f[i - 1]) / erb_bw_hz(fmid);
        u[i] = u[i - 1] + du;
    }

    // === LUT ===
    let half_width_erb = kparams.half_width_erb.max(0.5);
    let lut_step: f32 = 0.001;
    let (lut, hw) = build_kernel_erbstep(kparams, lut_step);

    // === 近傍積分（O(N·K)） ===
    let mut r = vec![0.0f32; n];
    for i in 1..n {
        let fi = f[i];
        let ei = e[i];

        for j in 0..n {
            if i == j {
                continue;
            }
            let fj = f[j];
            let bw = erb_bw_hz(0.5 * (fi + fj));
            let d = (fj - fi) / bw; // 符号付き ΔERB
            if d.abs() > half_width_erb {
                continue;
            }

            let w = lut_interp(&lut, lut_step, hw, d);
            // ★ ERB 数の微小幅：df を含める（PSD は 1/Hz スケール）
            let du = df / erb_bw_hz(fj).max(1e-12);
            r[i] += e[j] * w * du;
        }

        // 自己項除去（同じく df を含める）
        let g0 = lut[hw];
        r[i] = (r[i] - ei * g0 * (df / erb_bw_hz(fi).max(1e-12))).max(0.0);

        // 局所サリエンス重み
        r[i] *= ei.powf(alpha);
    }

    // === R_total：ERB-number 軸で台形積分（df と整合済みなのでそのまま）
    let mut r_total = 0.0f32;
    for i in 1..n {
        let du = (u[i] - u[i - 1]).max(0.0);
        r_total += 0.5 * (r[i - 1] + r[i]) * du;
    }

    (r, r_total)
}

/// Computes potential roughness directly from a mono waveform.
/// Performs: window → FFT → |X|² → ΔERB convolution via potential_r_from_psd_direct().
///
/// # Arguments
/// * `signal` - mono input signal
/// * `fs` - sampling rate [Hz]
/// * `params` - kernel parameters
/// * `gamma` - loudness compression exponent (typically 0.5)
/// * `alpha` - local salience weighting exponent (typically 0.5)
///
/// # Returns
/// (R_per_bin, R_total)
pub fn potential_r_from_signal_direct(
    signal: &[f32],
    fs: f32,
    params: &KernelParams,
    gamma: f32,
    alpha: f32,
) -> (Vec<f32>, f32) {
    if signal.is_empty() {
        return (vec![], 0.0);
    }

    // 1. zero-pad to next power of two
    let n = signal.len().next_power_of_two();
    let mut buf: Vec<Complex32> = signal.iter().map(|&x| Complex32::new(x, 0.0)).collect();
    buf.resize(n, Complex32::new(0.0, 0.0));

    // 2. apply Hann window  （窓掛けと同時に U=mean(w^2) を計算）
    let mut w2_sum = 0.0f32;
    for (i, x) in buf.iter_mut().enumerate() {
        let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos();
        x.re *= w;
        w2_sum += w * w;
    }
    let U = w2_sum / n as f32; // mean(w^2)

    // 3. FFT
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buf);

    // 4. One-sided PSD (1/Hz) per periodogram
    let nfft = n as f32;
    let n_half = n / 2; // Nyquist は含めない（片側 N/2 本）
    let scale = 1.0 / (fs * nfft * U); // Welchスケール
    let mut psd = vec![0.0f32; n_half];
    for i in 0..n_half {
        let mag2 = buf[i].re * buf[i].re + buf[i].im * buf[i].im;
        // 片側化: DCを除き2倍（Nyquistは配列に含めていないので気にしない）
        let two_sided_to_one = if i == 0 { 1.0 } else { 2.0 };
        psd[i] = mag2 * scale * two_sided_to_one;
    }

    // 5. pass to ΔERB convolution
    potential_r_from_psd_direct(&psd, fs, params, gamma, alpha)
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Shared ERB step for all tests (numerical resolution)
    const ERB_STEP: f32 = 0.005;

    /// Helper: build kernel with shared ERB step
    fn make_kernel() -> (Vec<f32>, usize) {
        let p = KernelParams::default();
        build_kernel_erbstep(&p, ERB_STEP)
    }

    /// 1. Kernel must be positive, centered, and decay to zero at edges
    #[test]
    fn kernel_is_positive_and_centered() {
        let (g, hw) = make_kernel();
        assert!(g.iter().all(|&v| v >= 0.0), "kernel must be non-negative");
        assert_eq!(g.len(), 2 * hw + 1, "length = 2*hw+1");
        let edge_mean = (g[0] + g[g.len() - 1]) * 0.5;
        eprintln!("edge mean: {}", edge_mean);
        assert!(edge_mean < 1e-4, "edges should decay toward zero");
    }

    /// 2. Center (ΔERB=0) should be strongly suppressed but nonzero
    #[test]
    fn kernel_center_is_suppressed_but_not_zero() {
        let (g, hw) = make_kernel();
        let center = g[hw];
        eprintln!("center value: {}", center);
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
        let (g1, hw) = build_kernel_erbstep(&p1, ERB_STEP);
        let c1 = g1[hw];

        let mut p2 = KernelParams::default();
        p2.w_neural = 0.4;
        let (g2, _) = build_kernel_erbstep(&p2, ERB_STEP);
        let c2 = g2[hw];

        assert!(c2 > c1, "neural layer should reduce center suppression");
    }

    /// 7. Kernel shape should remain stable across ERB steps
    #[test]
    fn kernel_stable_across_erbstep() {
        let p = KernelParams::default();
        let (g1, _) = build_kernel_erbstep(&p, ERB_STEP);
        let (g2, _) = build_kernel_erbstep(&p, 0.02);
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
        let (g, _) = super::build_kernel_erbstep(&params, erb_step);
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

    #[test]
    fn pure_tone_psd_returns_kernel_shape() {
        use crate::core::roughness_kernel::{
            KernelParams, build_kernel_erbstep, hz_to_erb, potential_r_from_psd_direct,
        };

        let fs = 48000.0;
        let n = 2048;
        let erb_step = 0.002;
        let params = KernelParams::default();
        let (g, hw) = build_kernel_erbstep(&params, erb_step);

        // --- narrow Gaussian PSD near 1 kHz
        let mut psd = vec![0.0f32; n / 2];
        let f_tone = 1000.0;
        let df = fs / (2 * (n - 1)) as f32;
        for i in 0..psd.len() {
            let fi = i as f32 * df;
            let d_erb = (hz_to_erb(fi) - hz_to_erb(f_tone)).abs();
            psd[i] = (-0.5 * (d_erb / 0.1).powi(2)).exp();
        }

        let (r_bins, _r_total) = potential_r_from_psd_direct(&psd, fs, &params, 1.0, 0.0);

        // --- normalize
        let g_norm: Vec<f32> = g.iter().map(|x| x / g.iter().sum::<f32>()).collect();
        let r_norm: Vec<f32> = r_bins
            .iter()
            .map(|x| x / r_bins.iter().sum::<f32>())
            .collect();

        // --- center alignment at peak
        let peak_idx = r_norm
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let lo = peak_idx.saturating_sub(hw);
        let hi = (peak_idx + hw).min(r_norm.len() - 1);
        let r_slice = &r_norm[lo..=hi];

        // --- mean absolute error
        let mae: f32 = r_slice
            .iter()
            .zip(&g_norm)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / g_norm.len() as f32;
        assert!(
            mae < 1e-3,
            "PSD pure tone should reproduce kernel (MAE={})",
            mae
        );
    }

    /// 純音入力で potential_r が単位カーネル形状を再現する（signal 経由）
    #[test]
    fn pure_tone_signal_returns_kernel_shape() {
        use crate::core::roughness_kernel::{
            KernelParams, build_kernel_erbstep, potential_r_from_signal_direct,
        };

        let fs = 48000.0;
        let f_tone = 1000.0;
        let n = 4096;
        let erb_step = 0.002;
        let params = KernelParams::default();
        let (g, hw) = build_kernel_erbstep(&params, erb_step);

        // --- 純音信号生成
        let mut sig = vec![0.0f32; n];
        for i in 0..n {
            let t = i as f32 / fs;
            sig[i] = (2.0 * std::f32::consts::PI * f_tone * t).sin();
        }

        // --- ΔERB 畳み込み実行
        let (r_bins, _r_total) = potential_r_from_signal_direct(&sig, fs, &params, 0.5, 0.0);

        // --- 正規化と比較
        let g_norm: Vec<f32> = g.iter().map(|x| x / g.iter().sum::<f32>()).collect();
        let r_norm: Vec<f32> = r_bins
            .iter()
            .map(|x| x / r_bins.iter().sum::<f32>())
            .collect();

        // --- 中心近傍抽出（最大値周辺）
        let peak_idx = r_norm
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let lo = peak_idx.saturating_sub(hw);
        let hi = (peak_idx + hw).min(r_norm.len() - 1);
        let r_slice = &r_norm[lo..=hi];

        // --- 相関誤差
        let mae: f32 = r_slice
            .iter()
            .zip(&g_norm)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / g_norm.len() as f32;
        assert!(
            mae < 1e-3,
            "Signal pure tone should reproduce kernel (MAE={})",
            mae
        );
    }

    /// 2. Two-tone ΔERB sweep → maximum near ΔERB≈0.25–0.35
    #[test]
    fn two_tone_peak_near_0p3_erb() {
        use crate::core::erb::ErbSpace;
        use crate::core::roughness_kernel::{KernelParams, potential_r_from_signal_direct};

        let fs = 48000.0;
        let base = 440.0;
        let n = 4096;
        let erb = ErbSpace::new(20.0, 8000.0, 0.005);
        let p = KernelParams::default();

        let mut d_erb_vec = Vec::new();
        let mut r_vals = Vec::new();

        for ratio in (100..200).map(|k| k as f32 / 100.0) {
            let f2 = base * ratio;
            let d_erb = (crate::core::roughness_kernel::hz_to_erb(f2)
                - crate::core::roughness_kernel::hz_to_erb(base))
            .abs();
            let mut sig = vec![0.0f32; n];
            for i in 0..n {
                let t = i as f32 / fs;
                sig[i] = (2.0 * std::f32::consts::PI * base * t).sin()
                    + (2.0 * std::f32::consts::PI * f2 * t).sin();
            }
            let (_r, r_total) = potential_r_from_signal_direct(&sig, fs, &p, 0.5, 0.5);
            d_erb_vec.push(d_erb);
            r_vals.push(r_total);
        }

        // --- Find ΔERB of max roughness ---
        let (i_max, _max_val) = r_vals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let d_erb_peak = d_erb_vec[i_max];
        assert!(
            d_erb_peak > 0.2 && d_erb_peak < 0.4,
            "expected ΔERB peak ≈ 0.25–0.35, got {d_erb_peak}"
        );
    }

    /// 3. Scale stability: doubling fs should yield similar total R
    #[test]
    fn potential_r_stable_across_fs() {
        use crate::core::roughness_kernel::{KernelParams, potential_r_from_signal_direct};

        let p = KernelParams::default();
        let base = 1000.0;

        // fs=48k
        let fs1 = 48000.0;
        let n1 = 4096;
        let mut sig1 = vec![0.0f32; n1];
        for i in 0..n1 {
            let t = i as f32 / fs1;
            sig1[i] = (2.0 * std::f32::consts::PI * base * t).sin();
        }
        let (_r1, rtot1) = potential_r_from_signal_direct(&sig1, fs1, &p, 0.5, 0.5);

        // fs=96k
        let fs2 = 96000.0;
        let n2 = 8192;
        let mut sig2 = vec![0.0f32; n2];
        for i in 0..n2 {
            let t = i as f32 / fs2;
            sig2[i] = (2.0 * std::f32::consts::PI * base * t).sin();
        }
        let (_r2, rtot2) = potential_r_from_signal_direct(&sig2, fs2, &p, 0.5, 0.5);

        let rel_err = ((rtot2 - rtot1) / rtot1.abs()).abs();
        assert!(
            rel_err < 0.001,
            "R_total should be stable across fs (rel_err={rel_err})"
        );
    }

    #[test]
    #[ignore]
    fn plot_potential_r_pure_tone_png() {
        use crate::core::roughness_kernel::{
            KernelParams, build_kernel_erbstep, potential_r_from_psd_direct,
            potential_r_from_signal_direct,
        };
        use plotters::prelude::*;
        use std::fs::File;
        use std::path::Path;

        // === 条件設定 ===
        let fs = 48000.0;
        let f_tone = 1000.0;
        let n = 4096;
        let params = KernelParams::default();
        let erb_step = 0.005;

        // === Kernel 参照 ===
        let (g, hw) = build_kernel_erbstep(&params, erb_step);
        let d_erb: Vec<f32> = (-(hw as i32)..=hw as i32)
            .map(|i| i as f32 * erb_step)
            .collect();

        // === 純音信号生成 ===
        let mut sig = vec![0.0f32; n];
        for i in 0..n {
            let t = i as f32 / fs;
            sig[i] = (2.0 * std::f32::consts::PI * f_tone * t).sin();
        }

        // === 1. Signal 経由 ===
        let (r_sig, _r_total_sig) = potential_r_from_signal_direct(&sig, fs, &params, 0.5, 0.0);

        // === 2. PSD 直接 ===
        let n_half = n / 2;
        let mut psd = vec![0.0f32; n_half];
        let df = fs / n as f32;
        let idx = (f_tone / df).round() as usize;
        if idx < n_half {
            psd[idx] = 1.0;
        }
        let (r_psd, _r_total_psd) = potential_r_from_psd_direct(&psd, fs, &params, 1.0, 0.0);

        // === 正規化 ===
        let max_sig = r_sig.iter().cloned().fold(0.0, f32::max);
        let max_psd = r_psd.iter().cloned().fold(0.0, f32::max);
        let r_sig_norm: Vec<f32> = r_sig.iter().map(|x| x / max_sig).collect();
        let r_psd_norm: Vec<f32> = r_psd.iter().map(|x| x / max_psd).collect();
        let g_norm: Vec<f32> = g
            .iter()
            .map(|x| x / g.iter().cloned().fold(0.0, f32::max))
            .collect();

        // === 出力パス ===
        let out_path = Path::new("target/test_potential_r_pure_tone.png");
        let root = BitMapBackend::new(out_path, (1280, 960)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        // === Chart 設定 ===
        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .caption(
                "Potential R for Pure Tone (Signal vs PSD)",
                ("sans-serif", 20),
            )
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0f32..r_sig_norm.len() as f32, 0.0f32..1.1f32)
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("frequency-bin index")
            .y_desc("Normalized R amplitude")
            .draw()
            .unwrap();

        // --- signal 経由
        chart
            .draw_series(LineSeries::new(
                (0..r_sig_norm.len()).map(|i| (i as f32, r_sig_norm[i])),
                &BLUE,
            ))
            .unwrap()
            .label("R(signal)")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

        // --- psd 直接
        chart
            .draw_series(LineSeries::new(
                (0..r_psd_norm.len()).map(|i| (i as f32, r_psd_norm[i])),
                &RED,
            ))
            .unwrap()
            .label("R(psd)")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

        // --- kernel 基準
        let df = fs / n as f32;
        let erb_to_bin = 100.0 / df;
        chart
            .draw_series(LineSeries::new(
                d_erb
                    .iter()
                    .zip(g_norm.iter())
                    .map(|(&d, &y)| (100.0 + d * erb_to_bin, y)),
                &GREEN,
            ))
            .unwrap()
            .label("Kernel g(ΔERB)")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &GREEN));

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .background_style(&WHITE.mix(0.8))
            .draw()
            .unwrap();

        // === finalize ===
        root.present().unwrap();
        assert!(File::open(out_path).is_ok(), "plot image was not created");
    }

    #[test]
    #[ignore]
    fn plot_potential_r_from_signal_direct_debug() {
        use crate::core::roughness_kernel::{KernelParams, potential_r_from_signal_direct};
        use plotters::prelude::*;

        let fs = 48000.0;
        let p = KernelParams::default();
        let base = 440.0;
        let n = 8192;

        // === Pure tone ===
        let mut sig1 = vec![0.0f32; n];
        for i in 0..n {
            let t = i as f32 / fs;
            sig1[i] = (2.0 * std::f32::consts::PI * base * t).sin();
        }
        let (r1, _) = potential_r_from_signal_direct(&sig1, fs, &p, 0.5, 0.5);

        // === Two-tone (ΔERB≈0.3) ===
        let mut sig2 = vec![0.0f32; n];
        let f2 = 440.0 * 1.2; // ≈ ΔERB 0.3
        for i in 0..n {
            let t = i as f32 / fs;
            sig2[i] = (2.0 * std::f32::consts::PI * base * t).sin()
                + (2.0 * std::f32::consts::PI * f2 * t).sin();
        }
        let (r2, _) = potential_r_from_signal_direct(&sig2, fs, &p, 0.5, 0.5);

        // === Plot ===
        let out_path = "target/test_potential_r_from_signal_direct_debug.png";
        let root = BitMapBackend::new(out_path, (1280, 960)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root)
            .caption("Potential R from Signal (direct ΔERB)", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(
                0.0f32..r1.len() as f32,
                0.0f32..r2.iter().cloned().fold(0.0, f32::max) * 1.1,
            )
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("frequency-bin index")
            .y_desc("R(f)")
            .draw()
            .unwrap();

        chart
            .draw_series(LineSeries::new(
                (0..r1.len()).map(|i| (i as f32, r1[i])),
                &BLUE,
            ))
            .unwrap()
            .label("pure tone")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

        chart
            .draw_series(LineSeries::new(
                (0..r2.len()).map(|i| (i as f32, r2[i])),
                &RED,
            ))
            .unwrap()
            .label("two-tone ΔERB≈0.3")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .background_style(&WHITE.mix(0.8))
            .draw()
            .unwrap();

        root.present().unwrap();
        assert!(
            std::fs::File::open(out_path).is_ok(),
            "failed to create debug plot"
        );
    }

    #[test]
    #[ignore]
    fn plot_potential_r_from_signal_direct_two_tone() {
        use crate::core::roughness_kernel::{KernelParams, potential_r_from_signal_direct};
        use plotters::prelude::*;

        let fs = 48000.0;
        let p = KernelParams::default();

        let base = 440.0;
        let mut ratios = Vec::new();
        let mut r_vals = Vec::new();

        for ratio in (100..200).map(|k| k as f32 / 100.0) {
            let f2 = base * ratio;
            let n = 8192;
            let mut sig = vec![0.0f32; n];
            for i in 0..n {
                let t = i as f32 / fs;
                sig[i] = (2.0 * std::f32::consts::PI * base * t).sin()
                    + (2.0 * std::f32::consts::PI * f2 * t).sin();
            }
            let (_r_bins, r_total) = potential_r_from_signal_direct(&sig, fs, &p, 0.5, 0.5);
            ratios.push(ratio);
            r_vals.push(r_total);
        }

        // === 可視化 ===
        let out_path = "target/test_potential_r_signal_direct_two_tone.png";
        let root = BitMapBackend::new(out_path, (1280, 960)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let max_y = r_vals.iter().cloned().fold(0.0, f32::max) * 1.2;
        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Potential R from Signal (direct ΔERB convolution)",
                ("sans-serif", 20),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(1.0f32..2.0f32, 0.0f32..max_y)
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("Frequency ratio f2/f1")
            .y_desc("R_total")
            .draw()
            .unwrap();

        chart
            .draw_series(LineSeries::new(
                ratios.iter().zip(r_vals.iter()).map(|(&x, &y)| (x, y)),
                &RED,
            ))
            .unwrap()
            .label("R_total")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .background_style(&WHITE.mix(0.8))
            .draw()
            .unwrap();

        root.present().unwrap();
    }

    #[test]
    #[ignore] // 実行時: cargo test -- --ignored
    fn compare_build_kernel_and_eval_kernel_shape() -> Result<(), Box<dyn std::error::Error>> {
        use crate::core::roughness_kernel::{
            KernelParams, build_kernel_erbstep, eval_kernel_delta_erb,
        };
        use plotters::prelude::*;

        // --- パラメータとステップ設定 ---
        let params = KernelParams::default();
        let erb_step = 0.005;
        let (g_discrete, hw) = build_kernel_erbstep(&params, erb_step);

        // ΔERB 軸
        let d_erb_vec: Vec<f32> = (-(hw as i32)..=hw as i32)
            .map(|i| i as f32 * erb_step)
            .collect();

        // eval_kernel_delta_erb で連続的にサンプリング
        let g_eval: Vec<f32> = d_erb_vec
            .iter()
            .map(|&d| eval_kernel_delta_erb(&params, d))
            .collect();

        // 正規化（L1一致のため）
        let sum1: f32 = g_discrete.iter().sum();
        let sum2: f32 = g_eval.iter().sum();
        let g1: Vec<f32> = g_discrete.iter().map(|&v| v / sum1).collect();
        let g2: Vec<f32> = g_eval.iter().map(|&v| v / sum2).collect();

        // --- 数値比較 ---
        let mut mae = 0.0;
        for i in 0..g1.len() {
            mae += (g1[i] - g2[i]).abs();
        }
        mae /= g1.len() as f32;

        println!(
            "MAE between build_kernel and eval_kernel_delta_erb = {}",
            mae
        );
        assert!(mae < 1e-3, "Kernel shapes differ (MAE={})", mae);

        // --- 可視化（確認用） ---
        let out_path = "target/test_kernel_build_vs_eval.png";
        let root = BitMapBackend::new(out_path, (1280, 960)).into_drawing_area();
        root.fill(&WHITE)?;

        let y_max = g1.iter().cloned().fold(0.0, f32::max) * 1.2;
        let mut chart = ChartBuilder::on(&root)
            .caption(
                "build_kernel_erbstep() vs eval_kernel_delta_erb()",
                ("sans-serif", 20),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(
                *d_erb_vec.first().unwrap()..*d_erb_vec.last().unwrap(),
                0.0f32..y_max,
            )?;

        chart
            .configure_mesh()
            .x_desc("ΔERB")
            .y_desc("Amplitude (normalized)")
            .draw()?;

        chart
            .draw_series(LineSeries::new(
                d_erb_vec.iter().zip(g1.iter()).map(|(&x, &y)| (x, y)),
                &BLUE,
            ))?
            .label("build_kernel (discrete)")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

        chart
            .draw_series(LineSeries::new(
                d_erb_vec.iter().zip(g2.iter()).map(|(&x, &y)| (x, y)),
                &RED,
            ))?
            .label("eval_kernel_delta_erb (continuous)")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .background_style(&WHITE.mix(0.8))
            .draw()?;

        root.present()?;
        println!("Output: {}", out_path);
        Ok(())
    }
}
