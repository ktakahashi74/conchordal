//! core/roughness.rs — Potential roughness (pre-cochlear) computation.
//!
//! Computes “potential roughness” R_pot directly from the pre-cochlear spectrum,
//! using the same biologically grounded ERB-domain kernel as `roughness_kernel.rs`.
//!
//! - pure tone → reproduces kernel shape (central dip and side peak)
//! - complex spectrum → yields Plomp–Levelt type roughness map
//!
//! This module reuses the same `KernelParams`, `build_kernel`, and
//! `fft_convolve_same()` definitions from `core::roughness_kernel`.

use crate::core::erb::ErbSpace;
use crate::core::roughness_kernel::{KernelParams, build_kernel, fft_convolve_same};
use rustfft::{FftPlanner, num_complex::Complex32};

// ======================================================================
// Utility functions
// ======================================================================

/// Linear interpolation in a uniformly spaced 1D array (index domain).
#[inline]
fn interp_linear(arr: &[f32], idx: f32) -> f32 {
    if arr.is_empty() {
        return 0.0;
    }
    let n = arr.len();
    if idx <= 0.0 {
        return arr[0];
    }
    let imax = (n - 1) as f32;
    if idx >= imax {
        return arr[n - 1];
    }
    let i0 = idx.floor() as usize;
    let t = idx - i0 as f32;
    let i1 = i0 + 1;
    arr[i0] * (1.0 - t) + arr[i1] * t
}

// ======================================================================
// ERB resampling and core computation
// ======================================================================

/// Resample a linear power spectrum (`|X(f)|²` up to Nyquist) onto the ERB grid.
/// - `power_lin[0..n_fft/2]` assumed, uniform Δf = fs/(2*n_bins)
pub fn resample_linear_power_to_erb(power_lin: &[f32], fs: f32, erb_space: &ErbSpace) -> Vec<f32> {
    let n_bins = power_lin.len().max(1);
    let df = (fs * 0.5) / (n_bins as f32);
    let mut e_erb = Vec::with_capacity(erb_space.freqs_hz.len());
    for &f_hz in &erb_space.freqs_hz {
        let idx = (f_hz / df).clamp(0.0, (n_bins - 1) as f32);
        e_erb.push(interp_linear(power_lin, idx));
    }
    e_erb
}

/// Compute potential roughness R_pot from ERB-sampled energy array.
///
/// This is the *physically defined* interference potential, independent of
/// cochlear filtering.  For a pure tone, the output equals the kernel shape.
pub fn compute_potential_r_from_erb_energy(
    e_erb: &[f32],
    erb_space: &ErbSpace,
    params: &KernelParams,
) -> Vec<f32> {
    if e_erb.is_empty() {
        return vec![];
    }

    let (g, _) = build_kernel(params, erb_space.erb_step);
    let r = fft_convolve_same(e_erb, &g);

    // Optional local weighting (Fechner-like loudness compression)
    r.iter()
        .zip(e_erb.iter())
        .map(|(ri, &ei)| ri * (ei.abs() + 1e-12).powf(0.5))
        .collect()
}

/// Compute potential roughness from a linear power spectrum (|X(f)|²).
pub fn compute_potential_r_from_linear_power(
    power_lin: &[f32],
    fs: f32,
    erb_space: &ErbSpace,
    params: &KernelParams,
) -> Vec<f32> {
    let e_erb = resample_linear_power_to_erb(power_lin, fs, erb_space);
    compute_potential_r_from_erb_energy(&e_erb, erb_space, params)
}

/// Compute potential roughness directly from a time-domain signal (mono).
/// Internally computes |FFT|², resamples to ERB, then applies the kernel.
pub fn compute_potential_r_from_signal(
    x: &[f32],
    fs: f32,
    erb_space: &ErbSpace,
    params: &KernelParams,
) -> Vec<f32> {
    if x.is_empty() {
        return vec![];
    }

    // zero-pad to next power of two
    let mut n = 1usize;
    while n < x.len() {
        n <<= 1;
    }

    let mut buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];
    for (i, &xi) in x.iter().enumerate() {
        buf[i].re = xi;
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buf);

    // one-sided power spectrum
    let nh = n / 2;
    let mut power_lin = vec![0.0f32; nh + 1];
    for k in 0..=nh {
        let c = buf[k];
        power_lin[k] = c.re * c.re + c.im * c.im;
    }

    compute_potential_r_from_linear_power(&power_lin, fs, erb_space, params)
}

// ======================================================================
// Post-cochlear variant (for comparison)
// ======================================================================

/// Compute cochlear roughness (post-filter envelope convolution).
/// This version uses the same kernel but assumes cochlear envelope inputs.
/// Not a “potential R” but often correlates with perceptual roughness.
pub fn compute_cochlear_r_from_envelope(
    e_ch: &[f32],
    erb_space: &ErbSpace,
    params: &KernelParams,
) -> Vec<f32> {
    if e_ch.is_empty() {
        return vec![];
    }
    let (g, _) = build_kernel(params, erb_space.erb_step);
    let r = fft_convolve_same(e_ch, &g);
    r.iter()
        .zip(e_ch.iter())
        .map(|(ri, &ei)| ri * (ei.abs() + 1e-12).powf(0.5))
        .collect()
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::roughness_kernel::KernelParams;

    #[test]
    fn pure_tone_reproduces_kernel_shape() {
        // build a mock ERB energy vector (single spike)
        let erb_space = ErbSpace::new(100.0, 8000.0, 0.005);
        let p = KernelParams::default();
        let (g, hw) = build_kernel(&p, erb_space.erb_step);
        let n_ch = 801;
        let center = n_ch / 2;
        let mut e = vec![0.0f32; n_ch];
        e[center] = 1.0;

        let r = compute_potential_r_from_erb_energy(&e, &erb_space, &p);

        // --- safe slice bounds
        let lo = center.saturating_sub(hw);
        let hi = (center + hw).min(n_ch - 1);

        let mut mae = 0.0;
        let mut n = 0;
        for (ri, &gi) in r[lo..=hi].iter().zip(g.iter()) {
            mae += (ri - gi).abs();
            n += 1;
        }
        mae /= n as f32;
        assert!(mae < 1e-3, "MAE too large: {}", mae);
    }

    #[test]
    #[ignore] // 実行時: cargo test -- --ignored
    fn plot_kernel_shape_for_pure_tone_png() -> Result<(), Box<dyn std::error::Error>> {
        use plotters::prelude::*;

        // === 定数設定 ===
        let erb_step = 0.005_f32;
        let n_ch = 1201usize;
        let center = n_ch / 2;

        let p = KernelParams::default();
        let (g, hw) = build_kernel(&p, erb_step);

        // --- 純音（ERB上のスパイク入力） ---
        let mut e = vec![0.0f32; n_ch];
        e[center] = 1.0;

        // === R_pot の計算（中心合わせあり） ===
        let r = crate::core::roughness_kernel::fft_convolve_same(&e, &g);

        // === 表示領域の切り出し ===
        let lo = center.saturating_sub(hw);
        let hi = (center + hw).min(n_ch - 1);

        let view_len = hi - lo + 1;
        let xs: Vec<f32> = (0..view_len)
            .map(|i| (i as f32 - hw as f32) * erb_step)
            .collect();
        let g_view = g.clone();
        let r_view = r[lo..=hi].to_vec();

        // === 正規化 ===
        let g_max = g_view.iter().cloned().fold(0.0, f32::max).max(1e-9);
        let r_max = r_view.iter().cloned().fold(0.0, f32::max).max(1e-9);
        let g_norm: Vec<f32> = g_view.iter().map(|&v| v / g_max).collect();
        let r_norm: Vec<f32> = r_view.iter().map(|&v| v / r_max).collect();

        // === プロット領域設定 ===
        let path = std::path::Path::new("target/pure_tone_kernel_shape.png");
        let size = (1400, 600);
        let root = BitMapBackend::new(path, size).into_drawing_area();
        root.fill(&WHITE)?;
        let plot = root.margin(40, 40, 40, 40);

        // 軸と枠
        let (w, h) = (size.0 as i32 - 80, size.1 as i32 - 80);
        let left = 0;
        let bottom = h;
        let right = w;
        let top = 0;

        // スケール変換
        let x_min = -(hw as f32) * erb_step;
        let x_max = (hw as f32) * erb_step;
        let y_min = 0.0;
        let y_max = 1.05;
        let sx = (right - left) as f32 / (x_max - x_min);
        let sy = (bottom - top) as f32 / (y_max - y_min);
        let to_px = |x: f32, y: f32| -> (i32, i32) {
            let px = left as f32 + (x - x_min) * sx;
            let py = bottom as f32 - (y - y_min) * sy;
            (px.round() as i32, py.round() as i32)
        };

        // 枠と基準線
        plot.draw(&PathElement::new(
            vec![(left, bottom), (right, bottom)],
            &BLACK,
        ))?;
        plot.draw(&PathElement::new(vec![(left, bottom), (left, top)], &BLACK))?;
        let (x0_px, _) = to_px(0.0, 0.0);
        plot.draw(&PathElement::new(
            vec![(x0_px, bottom), (x0_px, top)],
            &BLACK.mix(0.2),
        ))?;

        // === カーネル曲線（青） ===
        let g_points: Vec<(i32, i32)> = xs
            .iter()
            .zip(g_norm.iter())
            .map(|(&x, &y)| to_px(x, y))
            .collect();
        plot.draw(&PathElement::new(
            g_points,
            ShapeStyle::from(&BLUE).stroke_width(2),
        ))?;

        // === R_pot 曲線（赤、中心揃い） ===
        let r_points: Vec<(i32, i32)> = xs
            .iter()
            .zip(r_norm.iter())
            .map(|(&x, &y)| to_px(x, y))
            .collect();
        plot.draw(&PathElement::new(
            r_points,
            ShapeStyle::from(&RED.mix(0.6)).stroke_width(1),
        ))?;

        // 簡易目盛り（ΔERB ±0.25, ±0.5）
        for &tick in &[-0.5f32, -0.25, 0.25, 0.5] {
            let (tx, _) = to_px(tick, 0.0);
            plot.draw(&PathElement::new(
                vec![(tx, bottom), (tx, bottom - 8)],
                &BLACK.mix(0.4),
            ))?;
        }

        root.present()?;
        eprintln!("Saved kernel plot to {}", path.display());
        Ok(())
    }
}
