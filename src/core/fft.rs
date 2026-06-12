/// Symmetric 4-term Blackman-Harris window for peak detection.
/// Side lobes reach about -92 dB, which is excellent for reducing false peaks,
/// but the main lobe is wider than a Hann window.
#[inline]
pub fn blackman_harris_window_symmetric(n: usize) -> Vec<f32> {
    match n {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => {
            let two_pi = std::f64::consts::PI * 2.0;
            let denom = (n - 1) as f64;
            let a0 = 0.35875_f64;
            let a1 = 0.48829_f64;
            let a2 = 0.14128_f64;
            let a3 = 0.01168_f64;
            let mut w = Vec::with_capacity(n);
            for i in 0..n {
                let phi = two_pi * i as f64 / denom;
                let w_i = a0 - a1 * phi.cos() + a2 * (2.0 * phi).cos() - a3 * (3.0 * phi).cos();
                w.push(w_i.max(0.0) as f32);
            }
            w
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfft::{FftPlanner, num_complex::Complex32};

    #[test]
    fn blackman_harris_window_symmetric_props() {
        let n = 1024;
        let w = blackman_harris_window_symmetric(n);
        assert_eq!(w.len(), n);
        assert!(w.iter().all(|&v| v >= -1e-6));

        // Endpoints ~0 (allow small FP drift for coefficient rounding)
        assert!(w.first().unwrap().abs() < 1e-4, "first sample not ~0");
        assert!(w.last().unwrap().abs() < 1e-4, "last sample not ~0");

        // Symmetry
        let max_err = (0..n / 2)
            .map(|i| (w[i] - w[n - 1 - i]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-6, "symmetry max_err={max_err}");
    }

    #[test]
    fn blackman_harris_window_small_n_edges() {
        assert_eq!(blackman_harris_window_symmetric(1), vec![1.0]);
        let w2 = blackman_harris_window_symmetric(2);
        assert_eq!(w2.len(), 2);
    }

    #[test]
    fn blackman_harris_window_sidelobe_level() {
        let n = 1024;
        let nfft = 16_384;
        let w = blackman_harris_window_symmetric(n);

        let mut buf = vec![Complex32::new(0.0, 0.0); nfft];
        for (i, &v) in w.iter().enumerate() {
            buf[i].re = v;
        }
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(nfft);
        fft.process(&mut buf);

        let half = nfft / 2;
        let mags: Vec<f32> = buf[..half].iter().map(|z| z.norm()).collect();
        let peak = mags.iter().copied().fold(0.0f32, f32::max).max(1e-20);

        let thr = peak * 1e-4;
        let mut k = 1usize;
        while k < half && mags[k] >= thr {
            k += 1;
        }
        assert!(k < half, "main lobe extends too far");

        let side = mags[k..].iter().copied().fold(0.0f32, f32::max);
        if side > 0.0 {
            let side_db = 20.0 * (side / peak).log10();
            assert!(side_db < -70.0, "side lobe too high: {side_db:.2} dB");
        }
    }
}
