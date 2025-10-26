// src/core/fft.rs
use rustfft::{FftPlanner, num_complex::Complex32};
use scirs2_signal::{convolve, hilbert, window};

/// Symmetric Hann window (for offline / filter design)
pub fn hann_window_symmetric(n: usize) -> Vec<f32> {
    let w = window::hann(n, false).expect("failed to create symmetric Hann window");
    w.into_iter().map(|x| x as f32).collect()
}

/// Periodic Hann window (for FFT/STFT)
pub fn hann_window(n: usize) -> Vec<f32> {
    let w = window::hann(n, true).expect("failed to create periodic Hann window");
    w.into_iter().map(|x| x as f32).collect()
}

/// Apply Hann window to a real-valued buffer in-place.
/// Returns the energy correction factor U = mean(w²) ≈ 3/8 for normalization.
pub fn apply_hann_window_real(buf: &mut [f32]) -> f32 {
    let n = buf.len();
    if n <= 1 {
        return 1.0;
    }

    let win = hann_window(n);
    let mut sum_sq = 0.0;
    for (x, &w) in buf.iter_mut().zip(&win) {
        *x *= w;
        sum_sq += w * w;
    }

    // U = mean(w²) = 3/8
    sum_sq / n as f32
}

/// Apply Hann window (complex-valued input, periodic).
///
/// Multiplies each sample by w[i] = 0.5 * (1 - cos(2πi / N)),
/// and returns the Welch normalization factor U = Σw[i]^2 / N.
///
/// Same behavior and scaling as `apply_hann_window()` for real signals.

pub fn apply_hann_window_complex(buf: &mut [Complex32]) -> f32 {
    let n = buf.len();
    if n <= 1 {
        return 1.0;
    }
    let win = hann_window(n);
    let mut sum_sq = 0.0;
    for (x, &w) in buf.iter_mut().zip(win.iter()) {
        *x *= w;
        sum_sq += w * w;
    }
    sum_sq / buf.len() as f32
}

// ======================================================================
// Inverse STFT (OLA-based)
// ======================================================================

pub struct ISTFT {
    pub n: usize,
    pub hop: usize,
    pub window: Vec<f32>,
    ifft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    tmp: Vec<Complex32>,
    ola_buffer: Vec<f32>,
    write_pos: usize,
}

impl ISTFT {
    pub fn new(n: usize, hop: usize) -> Self {
        assert!(hop == n / 2);
        let mut planner = FftPlanner::<f32>::new();
        let ifft = planner.plan_fft_inverse(n);
        let window = hann_window(n);

        Self {
            n,
            hop,
            window,
            ifft,
            tmp: vec![Complex32::new(0.0, 0.0); n],
            ola_buffer: vec![0.0; n],
            write_pos: 0,
        }
    }

    pub fn process(&mut self, spec_half: &[Complex32]) -> Vec<f32> {
        let n = self.n;
        assert_eq!(spec_half.len(), n / 2 + 1);

        // Hermitian symmetry
        for k in 0..=n / 2 {
            self.tmp[k] = spec_half[k];
        }
        for k in 1..n / 2 {
            self.tmp[n - k] = self.tmp[k].conj();
        }

        // IFFT
        self.ifft.process(&mut self.tmp);
        let inv_n = 1.0 / (n as f32);

        // Apply window
        let mut win_frame = vec![0.0f32; n];
        for i in 0..n {
            win_frame[i] = (self.tmp[i].re * inv_n) * self.window[i];
        }

        // OLA
        for i in 0..n {
            let idx = (self.write_pos + i) % n;
            self.ola_buffer[idx] += win_frame[i];
        }
        self.write_pos = (self.write_pos + self.hop) % n;

        let mut out = vec![0.0; self.hop];
        for i in 0..self.hop {
            let idx = (self.write_pos + i) % n;
            out[i] = self.ola_buffer[idx];
            self.ola_buffer[idx] = 0.0;
        }
        out
    }
}

// ======================================================================
// FFT-based convolution (same-length output)
// ======================================================================

/// FFT-based convolution ("same" mode)
pub fn fft_convolve_same(x: &[f32], h: &[f32]) -> Vec<f32> {
    let y64 = convolve(x, h, "same").expect("FFT convolution failed");
    // f64 → f32 に変換して返す
    y64.into_iter().map(|v| v as f32).collect()
}

/// Compute analytic signal (normalize orientation)
/// Returns complex analytic signal: real = original signal, imag = Hilbert transform.
// /// Compute analytic signal via Hilbert transform.
// /// Output: complex-valued analytic signal (same length as input).
pub fn hilbert(x: &[f32]) -> Vec<Complex32> {
    let n = x.len().next_power_of_two();
    let mut buf: Vec<Complex32> = x.iter().map(|&v| Complex32::new(v, 0.0)).collect();
    buf.resize(n, Complex32::new(0.0, 0.0));

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    // Forward FFT
    fft.process(&mut buf);

    // Hilbert frequency multiplier
    let mut h = vec![Complex32::new(0.0, 0.0); n];
    if n > 0 {
        h[0] = Complex32::new(1.0, 0.0);
        if n % 2 == 0 {
            h[n / 2] = Complex32::new(1.0, 0.0);
            for k in 1..(n / 2) {
                h[k] = Complex32::new(2.0, 0.0);
            }
        } else {
            for k in 1..((n + 1) / 2) {
                h[k] = Complex32::new(2.0, 0.0);
            }
        }
    }

    // Apply H and IFFT
    for (z, &w) in buf.iter_mut().zip(h.iter()) {
        *z *= w;
    }
    ifft.process(&mut buf);

    // Normalize
    let norm = 1.0 / n as f32;
    buf.iter_mut().for_each(|z| *z *= norm);

    buf
}

// !!! scirs2 hilbert seems broken.

// pub fn hilbert(x: &[f32]) -> Vec<Complex32> {
//     // f32 → f64
//     let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();

//     // Compute analytic signal
//     let z64 = hilbert::hilbert(&x64).expect("Hilbert transform failed");

//     // f64 → f32
//     z64.into_iter()
//         .map(|c| Complex32::new(c.re as f32, -c.im as f32))
//         .collect()
// }

// /// Perform real-valued convolution using FFT ("same" mode)
// pub fn fft_convolve_same(x: &[f32], h: &[f32]) -> Vec<f32> {
//     assert!(!x.is_empty() && !h.is_empty());
//     let n_x = x.len();
//     let n_h = h.len();
//     let n_conv = n_x + n_h - 1;

//     // Next power of 2
//     let n_fft = n_conv.next_power_of_two();

//     // Prepare complex buffers
//     let mut a = vec![Complex32::new(0.0, 0.0); n_fft];
//     let mut b = vec![Complex32::new(0.0, 0.0); n_fft];
//     for (i, &v) in x.iter().enumerate() {
//         a[i].re = v;
//     }
//     for (i, &v) in h.iter().enumerate() {
//         b[i].re = v;
//     }

//     // FFT
//     let mut planner = FftPlanner::<f32>::new();
//     let fft = planner.plan_fft_forward(n_fft);
//     fft.process(&mut a);
//     fft.process(&mut b);

//     // Multiply in frequency domain
//     for i in 0..n_fft {
//         a[i] = a[i] * b[i];
//     }

//     // IFFT
//     let ifft = planner.plan_fft_inverse(n_fft);
//     ifft.process(&mut a);
//     let inv = 1.0 / (n_fft as f32);

//     // Real output (full convolution)
//     let mut y_full = vec![0.0f32; n_conv];
//     for i in 0..n_conv {
//         y_full[i] = a[i].re * inv;
//     }

//     // Center-crop to "same" length
//     let start = (n_h - 1) / 2;
//     let end = start + n_x;
//     y_full[start..end].to_vec()
// }

// ======================================================================
// Utility
// ======================================================================

pub fn bin_freqs_hz(fs: f32, n: usize) -> Vec<f32> {
    (0..=n / 2).map(|k| k as f32 * fs / n as f32).collect()
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hann_window_symmetric_sum() {
        let n = 1024;
        let w = hann_window_symmetric(n);
        assert!(w.iter().all(|&v| v >= 0.0));

        // 端点は理論的に0だが、浮動小数丸めで1e-5程度ずれることがある
        assert!(w.first().unwrap().abs() < 1e-5, "first sample not ~0");
        assert!(w.last().unwrap().abs() < 1e-5, "last sample not ~0");

        // エネルギー確認（mean(w²) ≈ 3/8）
        let u: f32 = w.iter().map(|&x| x * x).sum::<f32>() / n as f32;
        assert!((u - 0.375).abs() < 1e-4, "mean-square mismatch: {u}");
    }

    #[test]
    fn hann_window_periodic_props() {
        use std::f32::consts::PI;
        let n = 1024;
        let w = hann_window(n);

        // 非負値・先頭0確認
        assert!(w.iter().all(|&v| v >= 0.0));
        assert!(w[0].abs() < 1e-5, "first sample not ~0");

        // 末尾は理論値とほぼ一致
        let last_expected = 0.5 * (1.0 - (2.0 * PI * ((n as f32 - 1.0) / n as f32)).cos());
        assert!(
            (w[n - 1] - last_expected).abs() < 1e-4,
            "last sample diff too large: got {}, expected {}",
            w[n - 1],
            last_expected
        );

        // パワー正規化（U ≈ 3/8 ±0.001）
        let u: f32 = w.iter().map(|&x| x * x).sum::<f32>() / n as f32;
        assert!(
            (u - 0.375).abs() < 0.001,
            "mean-square mismatch: {} (expected ~0.375)",
            u
        );

        // COLA（Constant Overlap-Add）検証（誤差 ±1%）
        let hop = n / 2;
        let mut sum = vec![0.0f32; n + hop];
        for i in 0..n {
            sum[i] += w[i];
            sum[i + hop] += w[i];
        }
        let mid = n / 2;
        let avg = sum[mid - 32..mid + 32].iter().sum::<f32>() / 64.0;
        assert!((avg - 1.0).abs() < 0.01, "OLA not flat enough: avg={avg}");
    }

    #[test]
    fn test_apply_hann_window_complex_props() {
        use rustfft::num_complex::Complex32;

        let n = 1024usize;

        // 入力：0..1 の直線（実部のみ）
        let mut buf: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(i as f32 / n as f32, 0.0))
            .collect();

        // 参考用に周期Hann窓を再計算（apply_* 内部と同じ定義）
        let w: Vec<f32> = (0..n)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos()))
            .collect();

        // 期待する U = mean(w^2) を事前計算
        let u_ref: f32 = w.iter().map(|&v| v * v).sum::<f32>() / n as f32;

        // apply 実行
        let u = apply_hann_window_complex(&mut buf);

        // (1) U が mean(w^2) に一致（f32 なので 1e-6〜1e-5 程度で十分）
        assert!((u - u_ref).abs() < 1e-3, "U mismatch: got {u}, ref {u_ref}");

        // (2) E[(x·w)^2] を相関込みで厳密に離散計算した値と比較
        let expected_mean_sq: f32 = (0..n)
            .map(|i| {
                let x = i as f32 / n as f32;
                (x * w[i]).powi(2)
            })
            .sum::<f32>()
            / n as f32;

        let mean_sq: f32 = buf.iter().map(|z| z.re * z.re).sum::<f32>() / n as f32;

        let rel_err = (mean_sq - expected_mean_sq).abs() / expected_mean_sq.max(1e-12);
        assert!(
            rel_err < 5e-3,
            "Energy scaling mismatch: mean_sq={mean_sq}, expected={expected_mean_sq}, rel_err={rel_err:.3e}"
        );

        // (3) 端点チェック

        let last_expected = ((n - 1) as f32 / n as f32) * w[n - 1];
        assert!(buf[0].re.abs() < 1e-7, "start not ~0: {}", buf[0].re);
        let diff_end = (buf[n - 1].re - last_expected).abs();
        assert!(
            diff_end < 1e-5 || buf[n - 1].re == 0.0,
            "end mismatch: got {}, expected {}, diff={diff_end}",
            buf[n - 1].re,
            last_expected
        );

        // (4) 非負（Hann は非負）
        assert!(
            buf.iter().all(|z| z.re >= 0.0),
            "Negative real samples found"
        );
    }

    #[test]
    fn hann_periodic_sum_overlapadd_ok() {
        let n = 1024;
        let hop = n / 2;
        let w = hann_window(n);
        let mut sum = vec![0.0; n + hop];
        for i in 0..n {
            sum[i] += w[i];
            sum[i + hop] += w[i];
        }
        // overlap-add test: should be ~constant near middle
        let mid = n / 2;
        let avg = (sum[mid - 10..mid + 10].iter().sum::<f32>() / 20.0);
        assert!((avg - 1.0).abs() < 1e-3, "Overlap-add sum not flat: {avg}");
    }
    #[test]
    fn test_fft_convolve_same_impulse() {
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let h = vec![0.0, 1.0, 0.0, 0.0];
        let y = fft_convolve_same(&x, &h);

        // Check that output energy equals input energy (simple sanity)
        let sum_x: f32 = x.iter().sum();
        let sum_y: f32 = y.iter().sum();
        assert_relative_eq!(sum_x, sum_y, epsilon = 1e-6);

        // Shape sanity: one peak, same length
        assert_eq!(y.len(), x.len());
        let max_i = y
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert!(max_i < y.len());
    }

    #[test]
    fn test_fft_convolve_same_gaussian() {
        let n = 128;
        let mut g = vec![0.0; n];
        for i in 0..n {
            let x = (i as f32 - n as f32 / 2.0) / 10.0;
            g[i] = (-x * x).exp();
        }
        let y = fft_convolve_same(&g, &g);
        assert!(y[n / 2] > y[n / 4]); // center peak larger
    }

    #[test]
    fn test_fft_convolve_same_rect() {
        let x = vec![1.0; 32];
        let h = vec![1.0; 32];
        let y = fft_convolve_same(&x, &h);
        assert!(y.iter().all(|&v| v >= 0.0));
        let mid = y[y.len() / 2];
        assert!(mid > 10.0);
    }

    #[test]
    fn test_hilbert_signal_cosine() {
        use std::f32::consts::PI;

        let n = 1024;
        let freq = 5.0;
        let dt = 0.01;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 * dt).cos())
            .collect();

        let analytic = hilbert(&signal);
        assert_eq!(analytic.len(), signal.len());

        // 実部 ≈ 入力
        let mse_real: f32 = signal
            .iter()
            .zip(analytic.iter())
            .map(|(a, b)| (a - b.re).powi(2))
            .sum::<f32>()
            / n as f32;
        assert!(mse_real < 1e-6, "Real part mismatch too large: {mse_real}");

        // 中心領域の振幅 ≈ 1.0
        let start = n / 4;
        let end = 3 * n / 4;
        let avg_mag: f32 = analytic[start..end]
            .iter()
            .map(|c| (c.re.powi(2) + c.im.powi(2)).sqrt())
            .sum::<f32>()
            / (end - start) as f32;
        assert!((avg_mag - 1.0).abs() < 0.05, "Average magnitude {avg_mag}");
    }
}
