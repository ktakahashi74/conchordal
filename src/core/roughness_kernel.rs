//! core/roughness_kernel.rs — Roughness R via ERB-domain kernel convolution.
//! Computes frequency-space roughness landscape by convolving the cochlear
//! envelope energy along the ERB axis using an asymmetric kernel.
//!
//! This version keeps the kernel defined in ΔERB space (biologically grounded),
//! but allows reuse from NSGT/log2-domain analysis by mapping frequencies
//! through hz→ERB before weighting.
//!
//! Used in Landscape::process_frame() for RVariant::NsgtKernel.

use crate::core::erb::{erb_bw_hz, hz_to_erb};
use crate::core::fft::{apply_hann_window_complex, hilbert};
use crate::core::log2::Log2Space;
use crate::core::nsgt::{NsgtLog2, NsgtLog2Config};
use rustfft::{FftPlanner, num_complex::Complex32};
use std::sync::OnceLock;

// ======================================================================
// Kernel definition (Plomp–Levelt inspired, ΔERB domain)
// ======================================================================

#[derive(Clone, Copy, Debug)]
pub struct KernelParams {
    // === Cochlear layer ===
    pub sigma_erb: f32,
    pub tau_erb: f32,
    pub mix_tail: f32,
    pub half_width_erb: f32,
    pub suppress_sigma_erb: f32,
    pub suppress_pow: f32,
    // === Neural layer ===
    pub sigma_neural_erb: f32,
    pub w_neural: f32,
}

impl Default for KernelParams {
    fn default() -> Self {
        Self {
            sigma_erb: 0.45,
            tau_erb: 1.0,
            mix_tail: 0.20,
            half_width_erb: 4.0,
            suppress_sigma_erb: 0.10,
            suppress_pow: 2.0,
            sigma_neural_erb: 1.0,
            w_neural: 0.3,
        }
    }
}

// ======================================================================
// Kernel LUT
// ======================================================================

static KERNEL_CACHE: OnceLock<(Vec<f32>, usize, f32)> = OnceLock::new();

fn get_cached_kernel(kparams: &KernelParams, erb_step: f32) -> &'static (Vec<f32>, usize, f32) {
    KERNEL_CACHE.get_or_init(|| {
        let (lut, hw) = build_kernel_erbstep(kparams, erb_step);
        (lut, hw, erb_step)
    })
}

#[inline]
pub fn eval_kernel_delta_erb(params: &KernelParams, d_erb: f32) -> f32 {
    let sigma = params.sigma_erb.max(1e-6);
    let tau = params.tau_erb.max(1e-6);
    let s_sup = params.suppress_sigma_erb.max(1e-6);
    let sig_n = params.sigma_neural_erb.max(1e-6);

    let desq = d_erb * d_erb;
    let g_gauss = (-desq / (2.0 * sigma * sigma)).exp();
    let g_tail = if d_erb >= 0.0 {
        (-d_erb / tau).exp()
    } else {
        0.0
    };
    let base = (1.0 - params.mix_tail) * g_gauss + params.mix_tail * g_tail;

    let suppress = 1.0 - (-desq / (2.0 * s_sup * s_sup)).exp();
    let g_coch = base * suppress.powf(params.suppress_pow);
    let g_neural = (-desq / (2.0 * sig_n * sig_n)).exp();
    (1.0 - params.w_neural) * g_coch + params.w_neural * g_neural
}

pub fn build_kernel_erbstep(params: &KernelParams, erb_step: f32) -> (Vec<f32>, usize) {
    let hw_erb = params.half_width_erb;
    let n_side = (hw_erb / erb_step).ceil() as usize;
    let len = 2 * n_side + 1;

    let mut g: Vec<f32> = (0..len)
        .map(|i| {
            let d = (i as i32 - n_side as i32) as f32 * erb_step;
            eval_kernel_delta_erb(params, d)
        })
        .collect();

    let sum = g.iter().sum::<f32>();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        g.iter_mut().for_each(|v| *v *= inv);
    }
    (g, n_side)
}

#[inline]
fn lut_interp(lut: &[f32], step: f32, hw: usize, d_erb: f32) -> f32 {
    let t = d_erb / step + hw as f32;
    let i = t.floor();
    let i0 = i as isize;
    let i1 = i0 + 1;
    if i0 < 0 || (i1 as usize) >= lut.len() {
        return 0.0;
    }
    let frac = t - i;
    let a = lut[i0 as usize];
    let b = lut[i1 as usize];
    a + frac * (b - a)
}

// ======================================================================
// Roughness from NSGT/log2-space amplitudes (ΔERB LUT reused)
// ======================================================================

#[derive(Clone, Debug)]
pub struct RoughnessKernel {
    pub params: KernelParams,
    pub erb_step: f32,
}

impl RoughnessKernel {
    pub fn new(params: KernelParams, erb_step: f32) -> Self {
        Self { params, erb_step }
    }
}

// ======================================================================
// Existing (legacy) direct / analytic versions
// ======================================================================

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
    let nfft = 2 * n;
    let df = fs / nfft as f32;
    let e: Vec<f32> = psd_hz.iter().map(|&x| (x + 1e-12).powf(gamma)).collect();
    let f: Vec<f32> = (0..n).map(|i| i as f32 * df).collect();
    let erb: Vec<f32> = f.iter().map(|&x| hz_to_erb(x)).collect();
    let bw: Vec<f32> = f.iter().map(|&x| erb_bw_hz(x).max(1e-12)).collect();
    let du_hz: Vec<f32> = bw.iter().map(|&b| df / b).collect();
    let (lut, hw) = build_kernel_erbstep(kparams, 0.005);
    let half_width = kparams.half_width_erb;

    let mut r = vec![0.0f32; n];
    for i in 0..n {
        let fi_erb = erb[i];
        let ei = e[i];
        let mut sum = 0.0f32;
        let j_lo = erb.partition_point(|&x| x < fi_erb - half_width);
        let j_hi = erb.partition_point(|&x| x <= fi_erb + half_width);
        for j in j_lo..j_hi {
            let d = erb[j] - fi_erb;
            if d.abs() > half_width || i == j {
                continue;
            }
            let w = lut_interp(&lut, 0.005, hw, d);
            sum += e[j] * w * du_hz[j];
        }
        r[i] = sum * ei.powf(alpha);
    }
    let mut r_total = 0.0f32;
    for i in 1..n {
        let du = (erb[i] - erb[i - 1]).max(0.0);
        r_total += 0.5 * (r[i - 1] + r[i]) * du;
    }
    (r, r_total)
}

pub fn potential_r_from_analytic(
    analytic: &[Complex32],
    fs: f32,
    params: &KernelParams,
    gamma: f32,
    alpha: f32,
) -> (Vec<f32>, f32) {
    if analytic.is_empty() {
        return (vec![], 0.0);
    }
    let n0 = analytic.len();
    let n = n0.next_power_of_two();
    let mut buf: Vec<Complex32> = Vec::with_capacity(n);
    buf.extend_from_slice(analytic);
    if n > n0 {
        buf.resize(n, Complex32::new(0.0, 0.0));
    }
    let U = apply_hann_window_complex(&mut buf);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buf);
    let n_half = n / 2;
    let base_scale = 1.0 / (fs * n as f32 * U);
    let mut psd = vec![0.0f32; n_half];
    psd[0] = buf[0].norm_sqr() * base_scale;
    for i in 1..n_half {
        psd[i] = buf[i].norm_sqr() * base_scale * 2.0;
    }
    potential_r_from_psd_direct(&psd, fs, params, gamma, alpha)
}

/// Potential-R from log2-domain envelope spectrum.
///
/// Input:
/// - `amps` : envelope amplitudes per log2-bin (from NSGT)
/// - `freqs_hz` : corresponding center frequencies [Hz]
///
/// The kernel remains defined in ΔERB space.
/// Suitable for `Landscape::process_frame()` where NSGT is precomputed.

/// Potential-R from log2-domain envelope spectrum.
/// Uses `Log2Space` to obtain center frequencies.
/// Potential-R from log2-domain envelope spectrum (Neighbor integration, ±4 ERB).
///
/// Biological approximation of ΔERB kernel convolution using local integration
/// of neighboring bands within ±4 ERB distance on the cochlear (ERB) axis.
/// Faster, low-memory alternative to kernel convolution.
/// Potential-R from log2-domain envelope spectrum.
/// Uses `Log2Space` to obtain center frequencies.
pub fn potential_r_from_log2_spectrum(
    amps: &[f32],
    space: &Log2Space,
    params: &KernelParams,
    gamma: f32,
    alpha: f32,
) -> (Vec<f32>, f32) {
    use crate::core::erb::erb_to_hz;
    use crate::core::erb::hz_to_erb;

    if amps.is_empty() || space.centers_hz.is_empty() {
        return (vec![], 0.0);
    }
    assert_eq!(
        amps.len(),
        space.centers_hz.len(),
        "amps and space length mismatch"
    );

    // --- ERB mapping ---
    let erb_vals: Vec<f32> = space.centers_hz.iter().map(|&f| hz_to_erb(f)).collect();

    // --- Envelope compression ---
    let e: Vec<f32> = amps.iter().map(|&x| (x + 1e-9).powf(gamma)).collect();

    // --- Build ΔERB kernel ---
    let erb_step = 0.005;
    let (lut, hw) = build_kernel_erbstep(params, erb_step);
    let half_width_erb = params.half_width_erb;

    // --- Convolution ---
    let n = e.len();
    let mut r = vec![0.0f32; n];
    for i in 0..n {
        let fi_erb = erb_vals[i];
        let ei = e[i];
        let mut sum = 0.0f32;

        // --- determine ERB-range in Hz, then index range on log2-axis ---
        let lo_hz = erb_to_hz(fi_erb - half_width_erb);
        let hi_hz = erb_to_hz(fi_erb + half_width_erb);
        let j_lo = space.index_of_freq(lo_hz).unwrap_or(0);
        let j_hi = space.index_of_freq(hi_hz).unwrap_or(n - 1);

        for j in j_lo..j_hi {
            let d = erb_vals[j] - fi_erb;
            if d.abs() > half_width_erb || i == j {
                continue;
            }
            let w = lut_interp(&lut, erb_step, hw, d);
            sum += e[j] * w;
        }
        r[i] = sum * ei.powf(alpha);
    }

    // --- Integration over ERB axis ---
    let mut r_total = 0.0;
    for i in 1..n {
        let du = (erb_vals[i] - erb_vals[i - 1]).max(0.0);
        r_total += 0.5 * (r[i - 1] + r[i]) * du;
    }

    (r, r_total)
}

// ======================================================================
// Tests (Full restored version)
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::erb::{ErbSpace, erb_bw_hz, hz_to_erb};
    use plotters::prelude::*;
    use std::fs::File;
    use std::path::Path;

    const ERB_STEP: f32 = 0.005;

    fn make_kernel() -> (Vec<f32>, usize) {
        let p = KernelParams::default();
        build_kernel_erbstep(&p, ERB_STEP)
    }

    // ------------------------------------------------------------
    // 基本特性テスト
    // ------------------------------------------------------------

    #[test]
    fn kernel_is_positive_and_centered() {
        let (g, hw) = make_kernel();
        assert!(g.iter().all(|&v| v >= 0.0));
        assert_eq!(g.len(), 2 * hw + 1);
        let edge_mean = (g[0] + g[g.len() - 1]) * 0.5;
        assert!(edge_mean < 1e-4, "edges should decay toward zero");
    }

    #[test]
    fn kernel_center_is_suppressed_but_not_zero() {
        let (g, hw) = make_kernel();
        let center = g[hw];
        assert!(center > 0.0 && center < 0.005, "center suppression wrong");
    }

    #[test]
    fn kernel_has_peak_near_pm0p3_erb() {
        let (g, hw) = make_kernel();
        let n = g.len();
        let d_erb: Vec<f32> = (0..n)
            .map(|i| (i as i32 - hw as i32) as f32 * ERB_STEP)
            .collect();

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

        let (pos_x, pos_val) = pos_peak.expect("no positive-side peak");
        let (neg_x, _neg_val) = neg_peak.expect("no negative-side peak");
        assert!(
            pos_x > 0.2 && pos_x < 0.4,
            "positive peak near +0.3 ERB (got {:.2})",
            pos_x
        );
        assert!(
            neg_x < -0.2 && neg_x > -0.4,
            "negative peak near -0.3 ERB (got {:.2})",
            neg_x
        );
        assert!(pos_val > g[hw] * 1.5);
    }

    #[test]
    fn kernel_is_asymmetric() {
        let (g, hw) = make_kernel();
        let pos = (hw as f32 + 0.3 / ERB_STEP).round() as usize;
        let neg = (hw as f32 - 0.3 / ERB_STEP).round() as usize;
        assert!(g[pos] > g[neg]);
    }

    #[test]
    fn kernel_l1_norm_is_one() {
        let (g, _) = make_kernel();
        let sum: f32 = g.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "L1 norm={}", sum);
    }

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

        assert!(c2 > c1);
    }

    #[test]
    fn kernel_stable_across_erbstep() {
        let p = KernelParams::default();
        let (g1, _) = build_kernel_erbstep(&p, ERB_STEP);
        let (g2, _) = build_kernel_erbstep(&p, 0.02);
        let hw1 = g1.len() / 2;
        let hw2 = g2.len() / 2;
        let ratio1 = g1[(hw1 as f32 + 0.3 / ERB_STEP).round() as usize] / g1[hw1];
        let ratio2 = g2[(hw2 as f32 + 0.3 / 0.02).round() as usize] / g2[hw2];
        assert!((ratio1 - ratio2).abs() < 0.1);
    }

    // ------------------------------------------------------------
    // δ入力・NSGT・安定性
    // ------------------------------------------------------------

    #[test]
    fn delta_input_reproduces_kernel_shape_when_du_constant() {
        let fs = 48_000.0;
        let nfft = 131072;
        let n = nfft / 2;
        let df = fs / nfft as f32;
        let f0 = 1000.0;
        let k0 = (f0 / df).round() as usize;

        let p = KernelParams::default();
        let erb_step = 0.005;
        let (g, hw) = build_kernel_erbstep(&p, erb_step);

        let mut e = vec![0.0f32; n];
        e[k0] = 1.0;

        let fi = k0 as f32 * df;
        let du_const = df / erb_bw_hz(fi).max(1e-12);

        let mut r = vec![0.0f32; n];
        for i in 0..n {
            if i == k0 {
                continue;
            }
            let fj = i as f32 * df;
            let d = hz_to_erb(fj) - hz_to_erb(fi);
            if d.abs() > p.half_width_erb {
                continue;
            }
            let w = eval_kernel_delta_erb(&p, d);
            r[i] = w * du_const;
        }

        let lo = k0.saturating_sub((p.half_width_erb / erb_step).ceil() as usize);
        let hi = (k0 + (p.half_width_erb / erb_step).ceil() as usize).min(n - 1);

        let mut r_on_erb = Vec::with_capacity(g.len());
        for k in 0..g.len() {
            let d_k = (k as i32 - hw as i32) as f32 * erb_step;
            let f_target_erb = d_k + hz_to_erb(fi);
            let mut best_i = k0;
            let mut best_diff = f32::MAX;
            for i in lo..=hi {
                let fj = i as f32 * df;
                let erb_i = hz_to_erb(fj);
                let diff = (erb_i - f_target_erb).abs();
                if diff < best_diff {
                    best_diff = diff;
                    best_i = i;
                }
            }
            r_on_erb.push(r[best_i]);
        }

        let r_sum: f32 = r_on_erb.iter().sum();
        let g_sum: f32 = g.iter().sum();
        let r_norm: Vec<f32> = r_on_erb.iter().map(|&x| x / (r_sum + 1e-12)).collect();
        let g_norm: Vec<f32> = g.iter().map(|&x| x / (g_sum + 1e-12)).collect();

        let mae: f32 = r_norm
            .iter()
            .zip(g_norm.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / g_norm.len() as f32;
        assert!(mae < 1e-2, "MAE={}", mae);
    }

    #[test]
    fn delta_input_reproduces_kernel_shape() {
        let params = KernelParams::default();
        let erb_step = 0.005;

        // === 1. log2軸空間を生成 ===
        let space = Log2Space::new(20.0, 8000.0, 48); // 約8 octaves, 48 bins/oct

        // === 2. δ入力 ===
        let mut amps = vec![0.0f32; space.centers_hz.len()];
        let mid = amps.len() / 2;
        amps[mid] = 1.0;

        // === 3. potential R 計算 ===
        let (r_vec, _) = potential_r_from_log2_spectrum(&amps, &space, &params, 1.0, 0.0);

        // === 4. 参照kernel ===
        let (g_ref, hw) = build_kernel_erbstep(&params, erb_step);
        let d_erb_kernel: Vec<f32> = (-(hw as i32)..=hw as i32)
            .map(|i| i as f32 * erb_step)
            .collect();

        // === 5. ΔERB軸マッピング ===
        let f0_erb = hz_to_erb(space.centers_hz[mid]);
        let d_erb_vec: Vec<f32> = space
            .centers_hz
            .iter()
            .map(|&f| hz_to_erb(f) - f0_erb)
            .collect();

        // === 6. 正規化・MAE計算 ===
        let g_norm: Vec<f32> = g_ref
            .iter()
            .map(|&v| v / g_ref.iter().sum::<f32>())
            .collect();
        let r_norm: Vec<f32> = r_vec
            .iter()
            .map(|&v| v / r_vec.iter().sum::<f32>())
            .collect();

        let mut total_err = 0.0;
        let mut count = 0;
        for (de, rv) in d_erb_vec.iter().zip(r_norm.iter()) {
            if de.abs() < params.half_width_erb {
                let k_idx = ((de / erb_step) + hw as f32).round() as isize;
                if k_idx >= 0 && (k_idx as usize) < g_norm.len() {
                    total_err += (rv - g_norm[k_idx as usize]).abs();
                    count += 1;
                }
            }
        }
        let mae = total_err / (count as f32).max(1.0);
        assert!(
            mae < 1e-2,
            "MAE too large: {:.4} (should reproduce kernel shape)",
            mae
        );
    }

    #[test]
    fn two_tone_peak_near_0p3_erb() {
        let fs = 48000.0;
        let base = 440.0;
        let n = 4096;
        let p = KernelParams::default();

        let mut d_erb_vec = Vec::new();
        let mut r_vals = Vec::new();

        for ratio in (100..200).map(|k| k as f32 / 100.0) {
            let f2 = base * ratio;
            let d_erb = (hz_to_erb(f2) - hz_to_erb(base)).abs();
            let mut sig = vec![0.0f32; n];
            for i in 0..n {
                let t = i as f32 / fs;
                sig[i] = (2.0 * std::f32::consts::PI * base * t).sin()
                    + (2.0 * std::f32::consts::PI * f2 * t).sin();
            }
            let (_r, r_total) = potential_r_from_analytic(&hilbert(&sig), fs, &p, 0.5, 0.5);
            d_erb_vec.push(d_erb);
            r_vals.push(r_total);
        }

        let (i_max, _) = r_vals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let d_erb_peak = d_erb_vec[i_max];
        assert!(
            d_erb_peak > 0.2 && d_erb_peak < 0.4,
            "ΔERB peak ≈0.25–0.35, got {d_erb_peak}"
        );
    }

    #[test]
    fn potential_r_stable_across_fs() {
        let p = KernelParams::default();
        let base = 1000.0;

        let fs1 = 48000.0;
        let n1 = 4096;
        let sig1: Vec<f32> = (0..n1)
            .map(|i| (2.0 * std::f32::consts::PI * base * i as f32 / fs1).sin())
            .collect();
        let (_r1, rtot1) = potential_r_from_analytic(&hilbert(&sig1), fs1, &p, 0.5, 0.5);

        let fs2 = 96000.0;
        let n2 = 8192;
        let sig2: Vec<f32> = (0..n2)
            .map(|i| (2.0 * std::f32::consts::PI * base * i as f32 / fs2).sin())
            .collect();
        let (_r2, rtot2) = potential_r_from_analytic(&hilbert(&sig2), fs2, &p, 0.5, 0.5);

        let rel_err = ((rtot2 - rtot1) / rtot1.abs()).abs();
        assert!(rel_err < 0.001, "R_total rel_err={rel_err}");
    }

    // ------------------------------------------------------------
    // Plot系テスト (ignore)
    // ------------------------------------------------------------

    #[test]
    #[ignore]
    fn plot_kernel_shape_png() {
        let params = KernelParams::default();
        let erb_step = 0.02;
        let (g, _) = build_kernel_erbstep(&params, erb_step);
        let hw = (params.half_width_erb / erb_step).ceil() as i32;
        let d_erb: Vec<f32> = (-hw..=hw).map(|i| i as f32 * erb_step).collect();

        let out_path = Path::new("target/test_kernel_shape.png");
        let root = BitMapBackend::new(out_path, (1600, 1000)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption("Asymmetric ERB-domain Kernel", ("sans-serif", 20))
            .margin(10)
            .build_cartesian_2d(
                d_erb[0]..d_erb[d_erb.len() - 1],
                0.0..g.iter().cloned().fold(0.0, f32::max) * 1.1,
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
            .unwrap();
        root.present().unwrap();
        assert!(File::open(out_path).is_ok());
    }

    #[test]
    #[ignore]
    fn compare_build_kernel_and_eval_kernel_shape() -> Result<(), Box<dyn std::error::Error>> {
        let params = KernelParams::default();
        let erb_step = 0.005;
        let (g_discrete, hw) = build_kernel_erbstep(&params, erb_step);
        let d_erb_vec: Vec<f32> = (-(hw as i32)..=hw as i32)
            .map(|i| i as f32 * erb_step)
            .collect();
        let g_eval: Vec<f32> = d_erb_vec
            .iter()
            .map(|&d| eval_kernel_delta_erb(&params, d))
            .collect();
        let sum1: f32 = g_discrete.iter().sum();
        let sum2: f32 = g_eval.iter().sum();
        let g1: Vec<f32> = g_discrete.iter().map(|&v| v / sum1).collect();
        let g2: Vec<f32> = g_eval.iter().map(|&v| v / sum2).collect();
        let mae = g1
            .iter()
            .zip(g2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / g1.len() as f32;
        assert!(mae < 1e-3, "MAE={}", mae);

        let out_path = "target/test_kernel_build_vs_eval.png";
        let root = BitMapBackend::new(out_path, (1600, 1000)).into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .caption("build_kernel vs eval_kernel", ("sans-serif", 20))
            .margin(10)
            .build_cartesian_2d(
                -5.0f32..5.0f32,
                0.0f32..g1.iter().cloned().fold(0.0, f32::max) * 1.1,
            )
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("ΔERB")
            .y_desc("Amplitude")
            .draw()?;
        chart
            .draw_series(LineSeries::new(
                d_erb_vec.iter().zip(g1.iter()).map(|(&x, &y)| (x, y)),
                &BLUE,
            ))?
            .label("discrete")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));
        chart
            .draw_series(LineSeries::new(
                d_erb_vec.iter().zip(g2.iter()).map(|(&x, &y)| (x, y)),
                &RED,
            ))?
            .label("eval()")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));
        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
        root.present()?;
        Ok(())
    }
    #[test]
    #[ignore]
    fn plot_potential_r_pure_tone_png() {
        use crate::core::erb::hz_to_erb;

        let fs = 48000.0;
        let f_tone = 440.0;
        let n = 16384;
        let params = KernelParams::default();
        let erb_step = 0.005;

        // --- Kernel reference ---
        let (g, hw) = build_kernel_erbstep(&params, erb_step);
        let d_erb_kernel: Vec<f32> = (-(hw as i32)..=hw as i32)
            .map(|i| i as f32 * erb_step)
            .collect();

        // --- Generate pure tone signal ---
        let sig: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f_tone * i as f32 / fs).sin())
            .collect();

        // --- Compute potential R ---
        let (r_bins, _) = potential_r_from_analytic(&hilbert(&sig), fs, &params, 0.5, 0.0);

        // --- ΔERB axis for R ---
        let df = fs / n as f32;
        let f0_erb = hz_to_erb(f_tone);
        let d_erb_r: Vec<f32> = (0..r_bins.len())
            .map(|i| hz_to_erb(i as f32 * df) - f0_erb)
            .collect();

        // --- Normalize for overlay ---
        let g_norm: Vec<f32> = g
            .iter()
            .map(|&v| v / g.iter().cloned().fold(0.0, f32::max))
            .collect();
        let r_norm: Vec<f32> = r_bins
            .iter()
            .map(|&v| v / r_bins.iter().cloned().fold(0.0, f32::max))
            .collect();

        // --- Plot ---
        let out_path = Path::new("target/test_potential_r_pure_tone.png");
        let root = BitMapBackend::new(out_path, (1280, 960)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let y_max = 1.05f32;
        let mut chart = ChartBuilder::on(&root)
            .caption("Potential R for Pure Tone (ΔERB axis)", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(-5.0f64..5.0f64, 0.0f64..y_max as f64)
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("ΔERB")
            .y_desc("Normalized Amplitude")
            .x_labels(11)
            .x_label_formatter(&|v| format!("{:+.1}", v))
            .draw()
            .unwrap();

        // --- Kernel reference ---
        chart
            .draw_series(LineSeries::new(
                d_erb_kernel
                    .iter()
                    .zip(g_norm.iter())
                    .map(|(&x, &y)| (x as f64, y as f64)),
                &GREEN,
            ))
            .unwrap()
            .label("Kernel g(ΔERB)");

        // --- R result ---
        chart
            .draw_series(LineSeries::new(
                d_erb_r
                    .iter()
                    .zip(r_norm.iter())
                    .map(|(&x, &y)| (x as f64, y as f64)),
                &BLUE,
            ))
            .unwrap()
            .label("R(signal)");

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .background_style(&WHITE.mix(0.8))
            .draw()
            .unwrap();

        root.present().unwrap();
        assert!(File::open(out_path).is_ok(), "plot image was not created");
    }

    #[test]
    #[ignore]
    fn plot_potential_r_from_signal_direct_two_tone() {
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
            let (_r_bins, r_total) = potential_r_from_analytic(&hilbert(&sig), fs, &p, 0.5, 0.5);
            ratios.push(ratio);
            r_vals.push(r_total);
        }

        let out_path = "target/test_potential_r_signal_two_tone.png";
        let root = BitMapBackend::new(out_path, (1600, 1000)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let max_y = r_vals.iter().cloned().fold(0.0, f32::max) * 1.2;
        let mut chart = ChartBuilder::on(&root)
            .caption("Potential R (Two-tone ΔERB sweep)", ("sans-serif", 20))
            .margin(10)
            .build_cartesian_2d(1.0f32..2.0f32, 0.0f32..max_y)
            .unwrap();
        chart
            .configure_mesh()
            .x_desc("f2/f1")
            .y_desc("R_total")
            .draw()
            .unwrap();
        chart
            .draw_series(LineSeries::new(
                ratios.iter().zip(r_vals.iter()).map(|(&x, &y)| (x, y)),
                &RED,
            ))
            .unwrap();
        root.present().unwrap();
        assert!(File::open(out_path).is_ok());
    }

    #[test]
    #[ignore]
    fn plot_potential_r_from_signal_direct_erb() {
        let fs = 48000.0;
        let p = KernelParams::default();
        let base = 440.0;
        let n = 16384;

        let mut sig1 = vec![0.0f32; n];
        for i in 0..n {
            let t = i as f32 / fs;
            sig1[i] = (2.0 * std::f32::consts::PI * base * t).sin();
        }
        let (r1, _) = potential_r_from_analytic(&hilbert(&sig1), fs, &p, 0.5, 0.0);

        let mut sig2 = vec![0.0f32; n];
        let f2 = base * 1.2;
        for i in 0..n {
            let t = i as f32 / fs;
            sig2[i] = (2.0 * std::f32::consts::PI * base * t).sin()
                + (2.0 * std::f32::consts::PI * f2 * t).sin();
        }
        let (r2, _) = potential_r_from_analytic(&hilbert(&sig2), fs, &p, 0.5, 0.0);

        let df = fs / n as f32;
        let f0_erb = hz_to_erb(base);
        let x_erb: Vec<f32> = (0..r1.len())
            .map(|i| hz_to_erb(i as f32 * df) - f0_erb)
            .collect();

        let out_path = "target/test_potential_r_signal_direct_erb.png";
        let root = BitMapBackend::new(out_path, (1280, 960)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption("Potential R from Signal (ΔERB axis)", ("sans-serif", 20))
            .margin(10)
            .build_cartesian_2d(
                -5.0f32..5.0f32,
                0.0f32..r2.iter().cloned().fold(0.0, f32::max) * 1.1,
            )
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("ΔERB")
            .y_desc("R(f)")
            .draw()
            .unwrap();
        chart
            .draw_series(LineSeries::new(
                x_erb.iter().zip(r1.iter()).map(|(&x, &y)| (x, y)),
                &BLUE,
            ))
            .unwrap()
            .label("pure tone");
        chart
            .draw_series(LineSeries::new(
                x_erb.iter().zip(r2.iter()).map(|(&x, &y)| (x, y)),
                &RED,
            ))
            .unwrap()
            .label("two-tone ΔERB≈0.3");
        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()
            .unwrap();
        root.present().unwrap();
        assert!(File::open(out_path).is_ok());
    }

    #[test]
    #[ignore]
    fn plot_potential_r_from_log2_spectrum_delta_input() -> Result<(), Box<dyn std::error::Error>> {
        use crate::core::erb::hz_to_erb;
        use crate::core::log2::Log2Space;
        use plotters::prelude::*;

        let params = KernelParams::default();
        let erb_step = 0.005;
        let space = Log2Space::new(20.0, 8000.0, 144);

        // === δ入力 ===
        let mut amps = vec![0.0f32; space.centers_hz.len()];
        let mid = amps.len() / 2;
        amps[mid] = 1.0;

        // === potential R ===
        let (r_vec, _) = potential_r_from_log2_spectrum(&amps, &space, &params, 1.0, 0.0);

        // === ΔERB軸構築 ===
        let f0_erb = hz_to_erb(space.centers_hz[mid]);
        let d_erb_r: Vec<f32> = space
            .centers_hz
            .iter()
            .map(|&f| hz_to_erb(f) - f0_erb)
            .collect();

        // === Kernel参照 ===
        let (g_ref, hw) = build_kernel_erbstep(&params, erb_step);
        let d_erb_kernel: Vec<f32> = (-(hw as i32)..=hw as i32)
            .map(|i| i as f32 * erb_step)
            .collect();

        // === 正規化 ===
        let r_norm: Vec<f32> = r_vec
            .iter()
            .map(|&v| v / r_vec.iter().cloned().fold(0.0, f32::max))
            .collect();
        let g_norm: Vec<f32> = g_ref
            .iter()
            .map(|&v| v / g_ref.iter().cloned().fold(0.0, f32::max))
            .collect();

        // === Plot ===
        let out_path = "target/test_potential_r_from_log2_spectrum_delta.png";
        let root = BitMapBackend::new(out_path, (1280, 960)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Potential R from Log2 Spectrum (δ input)",
                ("sans-serif", 20),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(-5.0f32..5.0f32, 0.0f32..1.05f32)
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("ΔERB")
            .y_desc("Normalized Amplitude")
            .draw()?;

        chart
            .draw_series(LineSeries::new(
                d_erb_r.iter().zip(r_norm.iter()).map(|(&x, &y)| (x, y)),
                &BLUE,
            ))?
            .label("R(log2 input)")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

        chart
            .draw_series(LineSeries::new(
                d_erb_kernel
                    .iter()
                    .zip(g_norm.iter())
                    .map(|(&x, &y)| (x, y)),
                &GREEN,
            ))?
            .label("Kernel g(ΔERB)")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &GREEN));

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
        root.present()?;
        assert!(std::path::Path::new(out_path).exists());
        Ok(())
    }
}
