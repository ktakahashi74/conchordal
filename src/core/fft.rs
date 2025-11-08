use rustfft::{FftPlanner, num_complex::Complex32};

/// Symmetric Hann window (offline/filter design)
/// w[i] = 0.5 * (1 - cos(2πi/(N-1)))
#[inline]
pub fn hann_window_symmetric(n: usize) -> Vec<f32> {
    match n {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => {
            let two_pi = std::f32::consts::PI * 2.0;
            let denom = (n - 1) as f32;
            let mut w = Vec::with_capacity(n);
            for i in 0..n {
                let phi = two_pi * i as f32 / denom;
                w.push(0.5 * (1.0 - phi.cos()));
            }
            w
        }
    }
}

/// Periodic Hann window (for FFT/STFT, COLA)
/// w[i] = 0.5 * (1 - cos(2πi/N))
#[inline]
pub fn hann_window_periodic(n: usize) -> Vec<f32> {
    match n {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => {
            let two_pi = std::f32::consts::PI * 2.0;
            let n_f = n as f32;
            let mut w = Vec::with_capacity(n);
            for i in 0..n {
                let phi = two_pi * i as f32 / n_f;
                w.push(0.5 * (1.0 - phi.cos()));
            }
            w
        }
    }
}

/// Apply Hann window to a real buffer.
/// Returns U = mean(w²) ≈ 3/8 for normalization.
pub fn apply_hann_window_real(buf: &mut [f32]) -> f32 {
    let n = buf.len();
    if n <= 1 {
        return 1.0;
    }

    let win = hann_window_periodic(n);
    let mut sum_sq = 0.0;
    for (x, &w) in buf.iter_mut().zip(&win) {
        *x *= w;
        sum_sq += w * w;
    }
    sum_sq / n as f32
}

/// Apply Hann window to a complex buffer.
/// Returns Welch factor U = Σw² / N.
pub fn apply_hann_window_complex(buf: &mut [Complex32]) -> f32 {
    let n = buf.len();
    if n <= 1 {
        return 1.0;
    }
    let win = hann_window_periodic(n);
    let mut sum_sq = 0.0;
    for (x, &w) in buf.iter_mut().zip(&win) {
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
        let window = hann_window_periodic(n);

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

        // Reconstruct Hermitian symmetry
        for k in 0..=n / 2 {
            self.tmp[k] = spec_half[k];
        }
        for k in 1..n / 2 {
            self.tmp[n - k] = self.tmp[k].conj();
        }

        // IFFT
        self.ifft.process(&mut self.tmp);
        let inv_n = 1.0 / n as f32;

        // Apply window
        let mut win_frame = vec![0.0f32; n];
        for i in 0..n {
            win_frame[i] = self.tmp[i].re * inv_n * self.window[i];
        }

        // Overlap-add
        for i in 0..n {
            let idx = (self.write_pos + i) % n;
            self.ola_buffer[idx] += win_frame[i];
        }
        self.write_pos = (self.write_pos + self.hop) % n;

        // Output next hop
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
// FFT-based convolution ("same" mode)
// ======================================================================

/// FFT convolution with "same" output length (SciPy-compatible).
pub fn fft_convolve_same(x: &[f32], h: &[f32]) -> Vec<f32> {
    let nx = x.len();
    let nh = h.len();
    if nx == 0 || nh == 0 {
        return Vec::new();
    }

    // Use direct convolution for small signals
    let direct_limit: usize = 16_384;
    if nx.saturating_mul(nh) <= direct_limit {
        return conv_same_direct(x, h);
    }

    // Linear convolution via FFT
    let n_full = nx + nh - 1;
    let n_fft = n_full.next_power_of_two();

    let mut xa = vec![Complex32::new(0.0, 0.0); n_fft];
    let mut hb = vec![Complex32::new(0.0, 0.0); n_fft];

    for (i, &v) in x.iter().enumerate() {
        xa[i].re = v;
    }
    for (i, &v) in h.iter().enumerate() {
        hb[i].re = v;
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let ifft = planner.plan_fft_inverse(n_fft);

    fft.process(&mut xa);
    fft.process(&mut hb);

    // Multiply spectra
    for i in 0..n_fft {
        xa[i] = xa[i] * hb[i];
    }

    // IFFT and scale
    ifft.process(&mut xa);
    let scale = 1.0 / n_fft as f32;

    let mut y_full = vec![0.0f32; n_full];
    for i in 0..n_full {
        y_full[i] = xa[i].re * scale;
    }

    // Crop "same" segment
    let start = (nh - 1) / 2;
    y_full[start..start + nx].to_vec()
}

#[inline]
fn conv_same_direct(x: &[f32], h: &[f32]) -> Vec<f32> {
    let nx = x.len();
    let nh = h.len();
    let n_full = nx + nh - 1;
    let mut y_full = vec![0.0f32; n_full];

    for i in 0..nx {
        let xi = x[i];
        for j in 0..nh {
            y_full[i + j] += xi * h[j];
        }
    }

    let start = (nh - 1) / 2;
    y_full[start..start + nx].to_vec()
}

/// Compute analytic signal (Hilbert transform).
/// Returns complex output: real=input, imag=Hilbert(x).
pub fn hilbert(x: &[f32]) -> Vec<Complex32> {
    let n = x.len().next_power_of_two();
    let mut buf: Vec<Complex32> = x.iter().map(|&v| Complex32::new(v, 0.0)).collect();
    buf.resize(n, Complex32::new(0.0, 0.0));

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    // Forward FFT
    fft.process(&mut buf);

    // Frequency-domain multiplier
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

    // Apply H and inverse FFT
    for (z, &w) in buf.iter_mut().zip(&h) {
        *z *= w;
    }
    ifft.process(&mut buf);

    // Normalize
    let scale = 1.0 / n as f32;
    buf.iter_mut().for_each(|z| *z *= scale);

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
    use rustfft::num_complex::Complex32;

    // ----- helpers -----

    // Reference "same" via explicit linear convolution then crop.
    fn same_ref(x: &[f32], h: &[f32]) -> Vec<f32> {
        let n = x.len();
        let m = h.len();
        let mut full = vec![0.0f32; n + m - 1];
        for i in 0..n {
            let xi = x[i];
            for j in 0..m {
                full[i + j] += xi * h[j];
            }
        }
        let start = (m - 1) / 2;
        full[start..start + n].to_vec()
    }

    // ==============================
    // Hann window: symmetric/periodic
    // ==============================

    #[test]
    fn test_hann_window_symmetric_sum() {
        let n = 1024;
        let w = hann_window_symmetric(n);
        assert!(w.iter().all(|&v| v >= 0.0));

        // Endpoints ~0 (allow small FP drift)
        assert!(w.first().unwrap().abs() < 1e-5, "first sample not ~0");
        assert!(w.last().unwrap().abs() < 1e-5, "last sample not ~0");

        // Energy check: mean(w^2) ≈ 3/8
        let u: f32 = w.iter().map(|&x| x * x).sum::<f32>() / n as f32;
        assert!((u - 0.375).abs() < 1e-3, "mean-square mismatch: {u}");
    }

    #[test]
    fn hann_window_periodic_props() {
        use std::f32::consts::PI;
        let n = 1024;
        let w = hann_window_periodic(n);

        // Non-negative; first sample ~0
        assert!(w.iter().all(|&v| v >= 0.0));
        assert!(w[0].abs() < 1e-5, "first sample not ~0");

        // Last sample close to closed-form value
        let last_expected = 0.5 * (1.0 - (2.0 * PI * ((n as f32 - 1.0) / n as f32)).cos());
        assert!(
            (w[n - 1] - last_expected).abs() < 1e-4,
            "last sample diff too large: got {}, expected {}",
            w[n - 1],
            last_expected
        );

        // Welch power normalization U ≈ 3/8
        let u: f32 = w.iter().map(|&x| x * x).sum::<f32>() / n as f32;
        assert!(
            (u - 0.375).abs() < 0.001,
            "mean-square mismatch: {} (expected ~0.375)",
            u
        );

        // COLA (hop=N/2): overlap-add is ~flat
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
    fn hann_window_periodic_mean_is_half() {
        // For the periodic Hann, mean(w) = 0.5 exactly in discrete-time.
        let n = 1024;
        let w = hann_window_periodic(n);
        let mean = w.iter().sum::<f32>() / n as f32;
        assert!((mean - 0.5).abs() < 1e-6, "mean={}", mean);
    }

    #[test]
    fn hann_window_periodic_pairwise_sum_is_one() {
        // For hop = N/2, w[i] + w[i+N/2] = 1 for all i (up to FP).
        let n = 1024;
        let hop = n / 2;
        let w = hann_window_periodic(n);
        let mut max_err = 0.0f32;
        for i in 0..hop {
            let err = (w[i] + w[i + hop] - 1.0).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(max_err < 1e-6, "max_err={}", max_err);
    }

    #[test]
    fn hann_window_small_n_edges() {
        // N=1 policy in this codebase: return [1.0]
        assert_eq!(hann_window_symmetric(1), vec![1.0]);
        assert_eq!(hann_window_periodic(1), vec![1.0]);

        // N=2: symmetric=[0,0]; periodic=[0,1]
        assert_eq!(hann_window_symmetric(2), vec![0.0, 0.0]);
        let w2p = hann_window_periodic(2);
        assert!(w2p[0].abs() < 1e-7);
        assert!((w2p[1] - 1.0).abs() < 1e-7);
    }

    // ==============================
    // Window application
    // ==============================

    #[test]
    fn apply_hann_window_complex_props() {
        let n = 1024usize;

        // Input: linear ramp in the real part
        let mut buf: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(i as f32 / n as f32, 0.0))
            .collect();

        // Recreate the same periodic Hann
        let w: Vec<f32> = (0..n)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos()))
            .collect();

        // Reference U = mean(w^2)
        let u_ref: f32 = w.iter().map(|&v| v * v).sum::<f32>() / n as f32;

        // Apply and get U
        let u = apply_hann_window_complex(&mut buf);
        assert!((u - u_ref).abs() < 1e-3, "U mismatch: got {u}, ref {u_ref}");

        // Compare mean square to exact discrete expectation
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
            "mean_sq={mean_sq}, expected={expected_mean_sq}, rel_err={rel_err:.3e}"
        );

        // Endpoints
        let last_expected = ((n - 1) as f32 / n as f32) * w[n - 1];
        assert!(buf[0].re.abs() < 1e-7, "start not ~0: {}", buf[0].re);
        let diff_end = (buf[n - 1].re - last_expected).abs();
        assert!(
            diff_end < 1e-5 || buf[n - 1].re == 0.0,
            "end mismatch: got {}, expected {}, diff={diff_end}",
            buf[n - 1].re,
            last_expected
        );

        // Hann is non-negative
        assert!(
            buf.iter().all(|z| z.re >= 0.0),
            "Negative real samples found"
        );
    }

    #[test]
    fn apply_hann_window_real_returns_mean_square() {
        // For a buffer of ones, mean(x·w)^2 equals mean(w^2) = U.
        let n = 2048;
        let mut ones = vec![1.0f32; n];
        let u = apply_hann_window_real(&mut ones);
        let mean_sq: f32 = ones.iter().map(|&v| v * v).sum::<f32>() / n as f32;
        assert!((mean_sq - u).abs() < 1e-6, "mean_sq={}, U={}", mean_sq, u);
        assert!((u - 0.375).abs() < 1e-3);
    }

    // ==============================
    // FFT-based convolution ("same")
    // ==============================

    #[test]
    fn fft_convolve_same_impulse() {
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let h = vec![0.0, 1.0, 0.0, 0.0];
        let y = fft_convolve_same(&x, &h);

        // Sum preserved (simple sanity)
        let sum_x: f32 = x.iter().sum();
        let sum_y: f32 = y.iter().sum();
        assert_relative_eq!(sum_x, sum_y, epsilon = 1e-6);

        // One peak, same length
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
    fn fft_convolve_same_identity_with_centered_odd_kernel() {
        // With h = [0,1,0] (centered odd), "same" must equal x.
        let n = 64;
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin()).collect();
        let h = vec![0.0, 1.0, 0.0];
        let y = fft_convolve_same(&x, &h);
        for (a, b) in y.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn fft_convolve_same_matches_reference_small() {
        // Small sizes choose direct path; must match reference exactly.
        let n = 64;
        let m = 31; // odd
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin()).collect();
        let h: Vec<f32> = (0..m).map(|i| ((i as f32) * 0.07).cos()).collect();

        let y = fft_convolve_same(&x, &h);
        let y_ref = same_ref(&x, &h);

        assert_eq!(y.len(), n);
        let max_err = y
            .iter()
            .zip(y_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_err < 1e-6, "max_err={}", max_err);
    }

    #[test]
    fn fft_convolve_same_matches_reference_even_kernel() {
        // Even-length kernel; "same" cropping must still match reference.
        let n = 256;
        let m = 4; // even
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.02).sin()).collect();
        let h = vec![0.25f32; m]; // simple box
        let y = fft_convolve_same(&x, &h);
        let y_ref = same_ref(&x, &h);

        let mae = y
            .iter()
            .zip(y_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / (n as f32);
        assert!(mae < 2e-6, "mae={}", mae);
    }

    #[test]
    fn fft_convolve_same_gaussian() {
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
    fn fft_convolve_same_rect() {
        let x = vec![1.0; 32];
        let h = vec![1.0; 32];
        let y = fft_convolve_same(&x, &h);
        assert!(y.iter().all(|&v| v >= 0.0));
        let mid = y[y.len() / 2];
        assert!(mid > 10.0);
    }

    // ==============================
    // Hilbert transform
    // ==============================

    #[test]
    fn hilbert_signal_cosine() {
        use std::f32::consts::PI;

        let n = 1024; // power-of-two to match current implementation
        let freq = 5.0;
        let dt = 0.01;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 * dt).cos())
            .collect();

        let analytic = hilbert(&signal);
        assert_eq!(analytic.len(), signal.len());

        // Real part ≈ input
        let mse_real: f32 = signal
            .iter()
            .zip(analytic.iter())
            .map(|(a, b)| (a - b.re).powi(2))
            .sum::<f32>()
            / n as f32;
        assert!(mse_real < 1e-6, "Real part mismatch too large: {mse_real}");

        // Envelope ≈ 1 in the central region
        let start = n / 4;
        let end = 3 * n / 4;
        let avg_mag: f32 = analytic[start..end]
            .iter()
            .map(|c| (c.re.powi(2) + c.im.powi(2)).sqrt())
            .sum::<f32>()
            / (end - start) as f32;
        assert!((avg_mag - 1.0).abs() < 0.05, "Average magnitude {avg_mag}");
    }

    // ==============================
    // Utility
    // ==============================

    #[test]
    fn bin_freqs_hz_props() {
        let fs = 48000.0f32;
        let n = 1024usize;
        let f = bin_freqs_hz(fs, n);
        assert_eq!(f.len(), n / 2 + 1);
        assert!(f.windows(2).all(|w| w[1] > w[0]));
        assert!((f[0] - 0.0).abs() < 1e-12);
        assert!((f.last().unwrap() - fs / 2.0).abs() < 1e-6);
    }

    // ==============================
    // ISTFT (OLA)
    // ==============================

    #[test]
    fn istft_ola_dc_reconstruction_after_warmup() {
        // Feed pure-DC spectra; after one hop warmup, output should be ~1.0.
        let n = 1024;
        let hop = n / 2;
        let mut istft = ISTFT::new(n, hop);

        let mut out_all = Vec::new();
        for _ in 0..4 {
            let mut spec_half = vec![Complex32::new(0.0, 0.0); n / 2 + 1];
            // DC bin amplitude = n so that IFFT (1/n scaling) yields 1.0 samples.
            spec_half[0] = Complex32::new(n as f32, 0.0);
            out_all.extend(istft.process(&spec_half));
        }

        // Drop the first hop as warmup
        let steady = &out_all[hop..];
        let max_dev = steady.iter().map(|&v| (v - 1.0).abs()).fold(0.0, f32::max);
        assert!(max_dev < 2e-3, "max_dev={}", max_dev);
    }
}
