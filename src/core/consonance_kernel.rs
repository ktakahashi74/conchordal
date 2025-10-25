//! core/consonance_kernel.rs — Potential C via analytic signal (Hilbert or NSGT)
//!
//! Computes phase-locking potential C(f) from a complex analytic signal,
//! symmetric to potential R. Two stages:
//!   (1) PLV_state(t,f)  — short-lag phase locking (local temporal coherence)
//!   (2) Comb projection — harmonic comb convolution in log2-frequency.

use crate::core::fft::fft_convolve_same;
use rustfft::num_complex::Complex32;

#[derive(Clone, Debug, Default)]
pub struct ConsonanceKernel {
    pub params: CCombKernelParams,
}

impl ConsonanceKernel {
    pub fn new(params: CCombKernelParams) -> Self {
        Self { params }
    }
}

// ======================================================
// Stage 1 — PLV state from analytic signal
// ======================================================

#[derive(Clone, Copy, Debug)]
pub struct PlvParams {
    /// maximum lag (ms)
    pub tau_max_ms: f32,
    /// exponential decay constant (ms)
    pub tau0_ms: f32,
}

impl Default for PlvParams {
    fn default() -> Self {
        Self {
            tau_max_ms: 12.0,
            tau0_ms: 6.0,
        }
    }
}

/// Compute short-lag PLV spectrum from analytic signal.
/// Returns normalized magnitude per frequency bin.
pub fn plv_spectrum_from_analytic(analytic: &[Complex32], fs: f32, p: &PlvParams) -> Vec<f32> {
    if analytic.is_empty() {
        return vec![];
    }
    let n = analytic.len();
    let max_tau = ((p.tau_max_ms * 1e-3 * fs) as usize).min(n.saturating_sub(1));

    // normalize phase (unit vector)
    let z_hat: Vec<Complex32> = analytic
        .iter()
        .map(|z| {
            if z.norm() > 0.0 {
                *z / z.norm()
            } else {
                Complex32::new(0.0, 0.0)
            }
        })
        .collect();

    let mut acc = vec![Complex32::new(0.0, 0.0); n];
    let mut wsum = vec![0.0f32; n];

    // accumulate phase differences
    for tau in 1..=max_tau {
        let w_tau = (-((tau as f32) * 1000.0 / fs) / p.tau0_ms).exp();
        for t in 0..(n - tau) {
            let d = z_hat[t] * z_hat[t + tau].conj();
            let idx = t + tau / 2;
            acc[idx] += d * w_tau;
            wsum[idx] += w_tau;
        }
    }

    // mean vector length = PLV(t)
    let plv_time: Vec<f32> = acc
        .iter()
        .zip(wsum.iter())
        .map(|(&z, &w)| if w > 0.0 { (z / w).norm() } else { 0.0 })
        .collect();

    // FFT to frequency domain
    let mut buf: Vec<Complex32> = plv_time.iter().map(|&x| Complex32::new(x, 0.0)).collect();
    let mut planner = rustfft::FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buf);
    let n_half = n / 2;
    buf[..n_half].iter().map(|z| z.norm()).collect()
}

// ======================================================
// Stage 2 — Comb kernel (Δlog2 f axis)
// ======================================================

#[derive(Clone, Debug)]
pub struct CCombKernelParams {
    /// step size in log2(f)
    pub log2_step: f32,
    /// half width in bins
    pub half_width_bins: usize,
    /// Gaussian base sigma in bins
    pub sigma_bins: f32,
    /// (p,q,weight,σ) = ratio term (p/q harmonic)
    pub ratio_terms: Vec<(u32, u32, f32, f32)>,
}

impl Default for CCombKernelParams {
    fn default() -> Self {
        Self {
            log2_step: 1.0 / 100.0, // ≈100 bins per octave
            half_width_bins: 200,
            sigma_bins: 3.5,
            ratio_terms: vec![
                (1, 1, 1.0, 3.5),
                (2, 1, 0.9, 3.5),
                (3, 2, 0.8, 3.5),
                (4, 3, 0.7, 3.5),
                (5, 4, 0.6, 3.5),
                (6, 5, 0.5, 3.5),
            ],
        }
    }
}

#[inline]
fn gaussian(x: f32, sigma: f32) -> f32 {
    let s2 = sigma.max(1e-6).powi(2);
    (-0.5 * x * x / s2).exp()
}

/// Build comb kernel along log2 frequency axis.
pub fn build_c_comb_kernel(p: &CCombKernelParams) -> (Vec<f32>, usize) {
    let hw = p.half_width_bins;
    let n = 2 * hw + 1;
    let mut k = vec![0.0; n];
    for i in 0..n {
        let d = (i as isize - hw as isize) as f32;
        let mut v = gaussian(d, p.sigma_bins);
        for &(num, den, w, s) in &p.ratio_terms {
            let shift = ((num as f32) / (den as f32)).log2() / p.log2_step;
            v += w * (gaussian(d - shift, s) + gaussian(d + shift, s));
        }
        k[i] = v;
    }
    let sum: f32 = k.iter().sum();
    if sum > 0.0 {
        k.iter_mut().for_each(|x| *x /= sum);
    }
    (k, hw)
}

// ======================================================
// Stage 3 — Projection
// ======================================================

/// Potential C from analytic (or NSGT) signal.
pub fn potential_c_from_analytic(
    analytic: &[Complex32],
    fs: f32,
    plv_params: &PlvParams,
    kernel_params: &CCombKernelParams,
) -> (Vec<f32>, f32) {
    if analytic.is_empty() {
        return (vec![], 0.0);
    }

    // Stage 1: PLV magnitude
    let plv_spec = plv_spectrum_from_analytic(analytic, fs, plv_params);

    // Stage 2: Comb-kernel convolution (Δlog2 f)
    let (k, _) = build_c_comb_kernel(kernel_params);
    let c_pot = fft_convolve_same(&plv_spec, &k);

    let c_total: f32 = c_pot.iter().sum();
    (c_pot, c_total)
}

// ======================================================
// Tests
// ======================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fft::hilbert;
    use plotters::prelude::*;

    #[test]
    #[ignore]
    fn plot_potential_c_from_analytic_two_tone() {
        let fs = 48000.0;
        let n = 16384;
        let f1 = 440.0;
        let f2 = 550.0;
        let sig: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs;
                (2.0 * std::f32::consts::PI * f1 * t).sin()
                    + (2.0 * std::f32::consts::PI * f2 * t).sin()
            })
            .collect();

        let analytic = hilbert(&sig);
        let p = PlvParams::default();
        let k = CCombKernelParams::default();
        let (c_prof, _c_total) = potential_c_from_analytic(&analytic, fs, &p, &k);

        // plot
        let df = fs / n as f32;
        let freqs: Vec<f32> = (0..c_prof.len()).map(|i| i as f32 * df).collect();
        let out_path = "target/test_potential_c_analytic_two_tone.png";
        let root = BitMapBackend::new(out_path, (1400, 900)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let ymax = c_prof.iter().cloned().fold(0.0, f32::max) * 1.1;
        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Potential C(f) — analytic two-tone (log2 axis)",
                ("sans-serif", 20),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0f32..8000f32, 0f32..ymax)
            .unwrap();
        chart
            .configure_mesh()
            .x_desc("Frequency (Hz)")
            .y_desc("C_potential(f)")
            .x_label_formatter(&|v| format!("{:.0}", v))
            .draw()
            .unwrap();
        chart
            .draw_series(LineSeries::new(
                freqs.iter().zip(c_prof.iter()).map(|(&f, &c)| (f, c)),
                &BLUE,
            ))
            .unwrap();
        root.present().unwrap();
        println!("saved {out_path}");
    }
}
