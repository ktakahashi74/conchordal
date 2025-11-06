//! core/roughness_kernel.rs — Roughness R via ERB-domain kernel convolution.
//! Computes frequency-space roughness landscape by convolving the cochlear
//! envelope energy along the ERB axis using an asymmetric kernel.
//!
//! This version keeps the kernel defined in ΔERB space (biologically grounded),
//! but allows reuse from NSGT/log2-domain analysis by mapping frequencies
//! through hz→ERB before weighting.

use crate::core::erb::{erb_bw_hz, hz_to_erb};
use crate::core::fft::{apply_hann_window_complex, hilbert};
use crate::core::log2::Log2Space;
use rustfft::{FftPlanner, num_complex::Complex32};

// ======================================================================
// Kernel parameter definition (Plomp–Levelt inspired, ΔERB domain)
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
// Core kernel evaluation
// ======================================================================

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

    let suppress = (1.0 - (-desq / (2.0 * s_sup * s_sup)).exp()).clamp(0.0, 1.0);
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
// Roughness kernel (holds LUT, no global cache)
// ======================================================================

#[derive(Clone, Debug)]
pub struct RoughnessKernel {
    pub params: KernelParams,
    pub erb_step: f32,
    pub lut: Vec<f32>,
    pub hw: usize,
}

impl RoughnessKernel {
    /// Create a new kernel and precompute LUT.
    pub fn new(params: KernelParams, erb_step: f32) -> Self {
        let (lut, hw) = build_kernel_erbstep(&params, erb_step);
        Self {
            params,
            erb_step,
            lut,
            hw,
        }
    }

    // ------------------------------------------------------------------
    // Potential R from amplitude spectrum (linear frequency axis)
    // ------------------------------------------------------------------

    pub fn potential_r_from_spectrum(
        &self,
        amps_hz: &[f32],
        fs: f32,
        gamma: f32,
        alpha: f32,
    ) -> (Vec<f32>, f32) {
        let n = amps_hz.len();
        if n == 0 {
            return (vec![], 0.0);
        }

        let nfft = 2 * n;
        let df = fs / nfft as f32;

        // Envelope compression
        let e: Vec<f32> = amps_hz.iter().map(|&x| (x + 1e-12).powf(gamma)).collect();
        let f: Vec<f32> = (0..n).map(|i| i as f32 * df).collect();
        let erb: Vec<f32> = f.iter().map(|&x| hz_to_erb(x)).collect();
        let bw: Vec<f32> = f.iter().map(|&x| erb_bw_hz(x).max(1e-12)).collect();
        let du_hz: Vec<f32> = bw.iter().map(|&b| df / b).collect();
        let half_width = self.params.half_width_erb;

        let mut r = vec![0.0f32; n];
        for i in 0..n {
            let fi_erb = erb[i];
            let ei = e[i];
            let mut sum = 0.0f32;

            let j_lo = erb.partition_point(|&x| x < fi_erb - half_width);
            let j_hi = erb.partition_point(|&x| x <= fi_erb + half_width);
            for j in j_lo..j_hi {
                let d = erb[j] - fi_erb;
                if d.abs() > half_width {
                    continue;
                }
                let w = lut_interp(&self.lut, self.erb_step, self.hw, d);
                sum += e[j] * w * du_hz[j];
            }
            r[i] = sum * ei.powf(alpha);
        }

        // Integrate over ERB axis
        let mut r_total = 0.0;
        for i in 1..n {
            let du = (erb[i] - erb[i - 1]).max(0.0);
            r_total += 0.5 * (r[i - 1] + r[i]) * du;
        }
        (r, r_total)
    }

    // ------------------------------------------------------------------
    // Potential R from analytic (Hilbert) signal
    // ------------------------------------------------------------------

    pub fn potential_r_from_analytic(
        &self,
        analytic: &[Complex32],
        fs: f32,
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

        // Apply Hann window (complex)
        let U = apply_hann_window_complex(&mut buf);

        // FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut buf);

        // Convert to amplitude spectrum
        let n_half = n / 2;
        let df = fs / n as f32;
        let base_scale = 1.0 / (fs * n as f32 * U);
        let amps: Vec<f32> = (0..n_half)
            .map(|i| (buf[i].norm_sqr() * base_scale * 2.0 * df).sqrt())
            .collect();

        self.potential_r_from_spectrum(&amps, fs, gamma, alpha)
    }

    // ------------------------------------------------------------------
    // Potential R from log2-domain amplitude spectrum (NSGT)
    // ------------------------------------------------------------------

    pub fn potential_r_from_log2_spectrum(
        &self,
        amps: &[f32],
        space: &Log2Space,
        gamma: f32,
        alpha: f32,
    ) -> (Vec<f32>, f32) {
        use crate::core::erb::{erb_bw_hz, erb_to_hz, hz_to_erb};

        if amps.is_empty() || space.centers_hz.is_empty() {
            return (vec![], 0.0);
        }
        assert_eq!(
            amps.len(),
            space.centers_hz.len(),
            "amps and space length mismatch"
        );

        let n = amps.len();
        let bpo = space.bins_per_oct as f32;
        let delta_log2 = 1.0 / bpo;

        // (1) Map to ERB axis
        let erb_vals: Vec<f32> = space.centers_hz.iter().map(|&f| hz_to_erb(f)).collect();

        // (2) Envelope compression
        let e: Vec<f32> = amps.iter().map(|&x| (x + 1e-9).powf(gamma)).collect();

        // (3) Jacobian correction (Δf / BW_ERB)
        let jacobian: Vec<f32> = space
            .centers_hz
            .iter()
            .map(|&f| {
                let bw = erb_bw_hz(f).max(1e-12);
                let df = f * (2f32.powf(delta_log2 * 0.5) - 2f32.powf(-delta_log2 * 0.5));
                df / bw
            })
            .collect();

        // (4) Convolution over ERB axis
        let half_width_erb = self.params.half_width_erb;
        let mut r = vec![0.0f32; n];
        for i in 0..n {
            let fi_erb = erb_vals[i];
            let ei = e[i];
            let mut sum = 0.0f32;

            let lo_hz = erb_to_hz(fi_erb - half_width_erb);
            let hi_hz = erb_to_hz(fi_erb + half_width_erb);
            let j_lo = space.index_of_freq(lo_hz).unwrap_or(0);
            let j_hi = space.index_of_freq(hi_hz).unwrap_or(n - 1);

            for j in j_lo..=j_hi {
                let d = erb_vals[j] - fi_erb;
                let w = lut_interp(&self.lut, self.erb_step, self.hw, d);
                sum += e[j] * w * jacobian[j];
            }
            r[i] = sum * ei.powf(alpha);
        }

        // (5) Integration over ERB axis
        let mut r_total = 0.0;
        for i in 1..n {
            let du = (erb_vals[i] - erb_vals[i - 1]).max(0.0);
            r_total += 0.5 * (r[i - 1] + r[i]) * du;
        }

        (r, r_total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::erb::{ErbSpace, erb_bw_hz, hz_to_erb};
    use plotters::prelude::*;
    use std::fs::File;
    use std::path::Path;

    const ERB_STEP: f32 = 0.005;

    fn make_kernel() -> RoughnessKernel {
        let p = KernelParams::default();
        RoughnessKernel::new(p, ERB_STEP)
    }

    // ------------------------------------------------------------
    // Basic kernel characteristics
    // ------------------------------------------------------------

    #[test]
    fn kernel_is_positive_and_centered() {
        let k = make_kernel();
        let g = &k.lut;
        let hw = k.hw;
        assert!(g.iter().all(|&v| v >= 0.0));
        assert_eq!(g.len(), 2 * hw + 1);
        let edge_mean = (g[0] + g[g.len() - 1]) * 0.5;
        assert!(edge_mean < 1e-4, "edges should decay toward zero");
    }

    #[test]
    fn kernel_center_is_suppressed_but_not_zero() {
        let k = make_kernel();
        let g = &k.lut;
        let hw = k.hw;
        let center = g[hw];
        assert!(center > 0.0 && center < 0.005, "center suppression wrong");
    }

    #[test]
    fn kernel_has_peak_near_pm0p3_erb() {
        let k = make_kernel();
        let g = &k.lut;
        let hw = k.hw;
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
        let k = make_kernel();
        let g = &k.lut;
        let hw = k.hw;
        let pos = (hw as f32 + 0.3 / ERB_STEP).round() as usize;
        let neg = (hw as f32 - 0.3 / ERB_STEP).round() as usize;
        assert!(g[pos] > g[neg]);
    }

    #[test]
    fn kernel_l1_norm_is_one() {
        let k = make_kernel();
        let g = &k.lut;
        let sum: f32 = g.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "L1 norm={}", sum);
    }

    #[test]
    fn neural_layer_reduces_center_suppression() {
        let mut p1 = KernelParams::default();
        p1.w_neural = 0.0;
        let k1 = RoughnessKernel::new(p1, ERB_STEP);
        let c1 = k1.lut[k1.hw];

        let mut p2 = KernelParams::default();
        p2.w_neural = 0.4;
        let k2 = RoughnessKernel::new(p2, ERB_STEP);
        let c2 = k2.lut[k2.hw];

        assert!(c2 > c1);
    }

    #[test]
    fn kernel_stable_across_erbstep() {
        let p = KernelParams::default();
        let k1 = RoughnessKernel::new(p, ERB_STEP);
        let k2 = RoughnessKernel::new(p, 0.02);
        let g1 = &k1.lut;
        let g2 = &k2.lut;
        let hw1 = k1.hw;
        let hw2 = k2.hw;
        let ratio1 = g1[(hw1 as f32 + 0.3 / ERB_STEP).round() as usize] / g1[hw1];
        let ratio2 = g2[(hw2 as f32 + 0.3 / 0.02).round() as usize] / g2[hw2];
        assert!((ratio1 - ratio2).abs() < 0.1);
    }

    // ------------------------------------------------------------
    // δ-input and stability tests
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
        let k = RoughnessKernel::new(p, ERB_STEP);
        let g = &k.lut;
        let hw = k.hw;

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

        let lo = k0.saturating_sub((p.half_width_erb / ERB_STEP).ceil() as usize);
        let hi = (k0 + (p.half_width_erb / ERB_STEP).ceil() as usize).min(n - 1);

        let mut r_on_erb = Vec::with_capacity(g.len());
        for kidx in 0..g.len() {
            let d_k = (kidx as i32 - hw as i32) as f32 * ERB_STEP;
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
        let k = RoughnessKernel::new(params, ERB_STEP);
        let space = Log2Space::new(20.0, 8000.0, 500);

        let mut amps = vec![0.0f32; space.centers_hz.len()];
        let mid = amps.len() / 2;
        amps[mid] = 1.0;

        let (r_vec, _) = k.potential_r_from_log2_spectrum(&amps, &space, 1.0, 0.0);

        let g_ref = &k.lut;
        let hw = k.hw;
        let d_erb_kernel: Vec<f32> = (-(hw as i32)..=hw as i32)
            .map(|i| i as f32 * ERB_STEP)
            .collect();

        let f0_erb = hz_to_erb(space.centers_hz[mid]);
        let d_erb_vec: Vec<f32> = space
            .centers_hz
            .iter()
            .map(|&f| hz_to_erb(f) - f0_erb)
            .collect();

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
                let k_idx = ((de / ERB_STEP) + hw as f32).round() as isize;
                if k_idx >= 0 && (k_idx as usize) < g_norm.len() {
                    total_err += (rv - g_norm[k_idx as usize]).abs();
                    count += 1;
                }
            }
        }
        let mae = total_err / (count as f32).max(1.0);
        assert!(mae < 1e-3, "MAE too large: {:.4}", mae);
    }

    #[test]
    fn potential_r_stable_across_fs() {
        let p = KernelParams::default();
        let k = RoughnessKernel::new(p, ERB_STEP);
        let base = 1000.0;

        let fs1 = 48000.0;
        let n1 = 4096;
        let sig1: Vec<f32> = (0..n1)
            .map(|i| (2.0 * std::f32::consts::PI * base * i as f32 / fs1).sin())
            .collect();
        let (_r1, rtot1) = k.potential_r_from_analytic(&hilbert(&sig1), fs1, 0.5, 0.5);

        let fs2 = 96000.0;
        let n2 = 8192;
        let sig2: Vec<f32> = (0..n2)
            .map(|i| (2.0 * std::f32::consts::PI * base * i as f32 / fs2).sin())
            .collect();
        let (_r2, rtot2) = k.potential_r_from_analytic(&hilbert(&sig2), fs2, 0.5, 0.5);

        let rel_err = ((rtot2 - rtot1) / rtot1.abs()).abs();
        assert!(rel_err < 0.001, "R_total rel_err={rel_err}");
    }

    // ------------------------------------------------------------
    // Plot tests (unchanged, ignore for normal runs)
    // ------------------------------------------------------------

    #[test]
    #[ignore]
    fn plot_kernel_shape_png() {
        let k = make_kernel();
        let params = k.params;
        let erb_step = 0.02;
        let (g, _) = build_kernel_erbstep(&params, erb_step);
        let hw = (params.half_width_erb / erb_step).ceil() as i32;
        let d_erb: Vec<f32> = (-hw..=hw).map(|i| i as f32 * erb_step).collect();

        let out_path = Path::new("target/test_kernel_shape.png");
        let root = BitMapBackend::new(out_path, (1600, 1000)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption("Asymmetric ERB-domain Kernel", ("sans-serif", 30))
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
        let k = RoughnessKernel::new(params, erb_step);
        let g_discrete = &k.lut;
        let hw = k.hw;

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
            .caption("build_kernel vs eval_kernel", ("sans-serif", 30))
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
    fn plot_potential_r_from_signal_direct_two_tone() {
        let fs = 48000.0;
        let k = RoughnessKernel::new(KernelParams::default(), 0.005);
        let base = 440.0;
        let mut ratios = Vec::new();
        let mut r_vals = Vec::new();

        for ratio in (100..200).map(|k| k as f32 / 100.0) {
            let f2 = base * ratio;
            let n = 4096;
            let mut sig = vec![0.0f32; n];
            for i in 0..n {
                let t = i as f32 / fs;
                sig[i] = (2.0 * std::f32::consts::PI * base * t).sin()
                    + (2.0 * std::f32::consts::PI * f2 * t).sin();
            }
            let (_r_bins, r_total) = k.potential_r_from_analytic(&hilbert(&sig), fs, 1.0, 0.0);
            ratios.push(ratio);
            r_vals.push(r_total);
        }

        let out_path = "target/test_potential_r_signal_two_tone.png";
        let root = BitMapBackend::new(out_path, (1600, 1000)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let max_y = r_vals.iter().cloned().fold(0.0, f32::max) * 1.2;
        let mut chart = ChartBuilder::on(&root)
            .caption("Potential R (Two-tone ΔERB sweep)", ("sans-serif", 30))
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
        let k = RoughnessKernel::new(KernelParams::default(), 0.005);
        let base = 440.0;
        let n = 16384;

        let mut sig1 = vec![0.0f32; n];
        for i in 0..n {
            let t = i as f32 / fs;
            sig1[i] = (2.0 * std::f32::consts::PI * base * t).sin();
        }
        let (r1, _) = k.potential_r_from_analytic(&hilbert(&sig1), fs, 1.0, 0.0);

        let mut sig2 = vec![0.0f32; n];
        let f2 = base * 1.2;
        for i in 0..n {
            let t = i as f32 / fs;
            sig2[i] = (2.0 * std::f32::consts::PI * base * t).sin()
                + (2.0 * std::f32::consts::PI * f2 * t).sin();
        }
        let (r2, _) = k.potential_r_from_analytic(&hilbert(&sig2), fs, 1.0, 0.0);

        let df = fs / n as f32;
        let f0_erb = hz_to_erb(base);
        let x_erb: Vec<f32> = (0..r1.len())
            .map(|i| hz_to_erb(i as f32 * df) - f0_erb)
            .collect();

        let out_path = "target/test_potential_r_signal_direct_erb.png";
        let root = BitMapBackend::new(out_path, (1600, 1000)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption("Potential R from Signal (ΔERB axis)", ("sans-serif", 30))
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

        let k = RoughnessKernel::new(KernelParams::default(), 0.005);
        let space = Log2Space::new(20.0, 8000.0, 144);

        let mut amps = vec![0.0f32; space.centers_hz.len()];
        let mid = amps.len() / 2;
        amps[mid] = 1.0;

        let (r_vec, _) = k.potential_r_from_log2_spectrum(&amps, &space, 1.0, 0.0);

        let mid = amps.len() / 2;
        let f0_erb = hz_to_erb(space.centers_hz[mid]);
        let erb_per_bin = hz_to_erb(space.centers_hz[mid + 1]) - hz_to_erb(space.centers_hz[mid]);
        let d_erb_r: Vec<f32> = (0..amps.len())
            .map(|i| (i as f32 - mid as f32) * erb_per_bin)
            .collect();

        let g_ref = &k.lut;
        let hw = k.hw;
        let d_erb_kernel: Vec<f32> = (-(hw as i32)..=hw as i32)
            .map(|i| i as f32 * k.erb_step)
            .collect();

        let r_norm: Vec<f32> = r_vec
            .iter()
            .map(|&v| v / r_vec.iter().cloned().fold(0.0, f32::max))
            .collect();
        let g_norm: Vec<f32> = g_ref
            .iter()
            .map(|&v| v / g_ref.iter().cloned().fold(0.0, f32::max))
            .collect();

        let out_path = "target/test_potential_r_from_log2_spectrum_delta.png";
        let root = BitMapBackend::new(out_path, (1500, 1000)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Potential R from Log2 Spectrum (δ input)",
                ("sans-serif", 30),
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
                    .map(|(&x, &y)| (-x, y)),
                &GREEN,
            ))?
            .label("Kernel g(ΔERB), flipped")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &GREEN));
        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
        root.present()?;
        assert!(std::path::Path::new(out_path).exists());
        Ok(())
    }

    #[test]
    #[ignore]
    fn plot_potential_r_delta_input_all_methods() -> Result<(), Box<dyn std::error::Error>> {
        use crate::core::erb::hz_to_erb;
        use crate::core::fft::hilbert;
        use crate::core::log2::Log2Space;
        use plotters::prelude::*;
        use rustfft::{FftPlanner, num_complex::Complex32};
        use std::f32::consts::PI;

        let fs = 48_000.0;
        let params = KernelParams::default();
        let k = RoughnessKernel::new(params, 0.005);
        let space = Log2Space::new(20.0, 8000.0, 144);
        let nfft = 163_84;

        let mut amps_log2 = vec![0.0f32; space.centers_hz.len()];
        let mid = amps_log2.len() / 2;
        amps_log2[mid] = 1.0;

        let (r_log2, _) = k.potential_r_from_log2_spectrum(&amps_log2, &space, 1.0, 0.0);

        let df = fs / nfft as f32;
        let mut amps_lin = vec![0.0f32; nfft / 2];
        for (kidx, &f) in space.centers_hz.iter().enumerate() {
            let bin = (f / df).round() as usize;
            if bin < amps_lin.len() {
                amps_lin[bin] += amps_log2[kidx];
            }
        }
        let (r_spec, _) = k.potential_r_from_spectrum(&amps_lin, fs, 1.0, 0.0);

        let mut buf = vec![Complex32::new(0.0, 0.0); nfft];
        for (i, &amp) in amps_lin.iter().enumerate() {
            buf[i] = Complex32::new(amp, 0.0);
        }
        for i in 1..(nfft / 2) {
            buf[nfft - i] = buf[i].conj();
        }
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(nfft);
        ifft.process(&mut buf);
        let sig: Vec<f32> = buf.iter().map(|z| z.re / nfft as f32).collect();
        let (r_analytic, _) = k.potential_r_from_analytic(&hilbert(&sig), fs, 1.0, 0.0);

        // plotting same as original (omitted for brevity)
        Ok(())
    }
}
