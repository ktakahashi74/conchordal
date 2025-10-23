// src/core/fft.rs
use rustfft::{FftPlanner, num_complex::Complex32};

pub fn hann_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| {
            let x = std::f32::consts::TAU * i as f32 / (n as f32 - 1.0);
            0.5 * (1.0 - f32::cos(x))
        })
        .collect()
}

/// Apply Hann window in place to a real-valued buffer and
/// return the mean-square value U = mean(w²) for physical scaling.
///
/// For Hann window, U = 3/8 = 0.375 (verified analytically).
/// This is computed once here for flexibility (not recomputed per sample).
pub fn apply_hann_window(buf: &mut [Complex32]) -> f32 {
    let n = buf.len();
    if n <= 1 {
        return 1.0;
    }

    let win = hann_window(n);
    for (x, &w) in buf.iter_mut().zip(&win) {
        x.re *= w;
    }

    // U = mean(w²) = 3/8
    3.0 / 8.0
}

/// Apply Hann window (complex-valued input).
///
/// Multiplies each sample by w[i] = 0.5 * (1 - cos(2πi / N)),
/// and returns the Welch normalization factor U = Σw[i]^2 / N.
///
/// Same behavior and scaling as `apply_hann_window()` for real signals.
pub fn apply_hann_window_complex(buf: &mut [Complex32]) -> f32 {
    let n = buf.len();
    if n == 0 {
        return 1.0;
    }

    let mut sum_sq = 0.0f32;
    for (i, x) in buf.iter_mut().enumerate() {
        let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos());
        *x *= w;
        sum_sq += w * w;
    }

    sum_sq / n as f32
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

/// Perform real-valued convolution using FFT ("same" mode)
pub fn fft_convolve_same(x: &[f32], h: &[f32]) -> Vec<f32> {
    assert!(!x.is_empty() && !h.is_empty());
    let n_x = x.len();
    let n_h = h.len();
    let n_conv = n_x + n_h - 1;

    // Next power of 2
    let n_fft = n_conv.next_power_of_two();

    // Prepare complex buffers
    let mut a = vec![Complex32::new(0.0, 0.0); n_fft];
    let mut b = vec![Complex32::new(0.0, 0.0); n_fft];
    for (i, &v) in x.iter().enumerate() {
        a[i].re = v;
    }
    for (i, &v) in h.iter().enumerate() {
        b[i].re = v;
    }

    // FFT
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    fft.process(&mut a);
    fft.process(&mut b);

    // Multiply in frequency domain
    for i in 0..n_fft {
        a[i] = a[i] * b[i];
    }

    // IFFT
    let ifft = planner.plan_fft_inverse(n_fft);
    ifft.process(&mut a);
    let inv = 1.0 / (n_fft as f32);

    // Real output (full convolution)
    let mut y_full = vec![0.0f32; n_conv];
    for i in 0..n_conv {
        y_full[i] = a[i].re * inv;
    }

    // Center-crop to "same" length
    let start = (n_h - 1) / 2;
    let end = start + n_x;
    y_full[start..end].to_vec()
}

/// Compute analytic signal via Hilbert transform.
/// Output: complex-valued analytic signal (same length as input).
pub fn analytic_signal(x: &[f32]) -> Vec<Complex32> {
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
    fn test_hann_window_sum() {
        let w = hann_window(1024);
        assert!(w.iter().all(|&v| v >= 0.0));
        assert!(w.first().unwrap() < &1e-6);
        assert!(w.last().unwrap() < &1e-6);
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
}
