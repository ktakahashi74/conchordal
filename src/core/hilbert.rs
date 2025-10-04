use std::f32::consts::PI;

use rustfft::{FftPlanner, num_complex::Complex, num_complex::Complex32, num_traits::Zero};

/// Analytic-like signal by quadrature demodulation
/// signal: input mono samples
/// fs: sampling rate (Hz)
/// fc: center frequency (Hz)
/// lp_len: length of moving-average LPF (samples)
pub fn hilbert_demod(signal: &[f32], fs: f32, fc: f32, lp_len: usize) -> Vec<Complex32> {
    let n = signal.len();
    let mut i_comp = Vec::with_capacity(n);
    let mut q_comp = Vec::with_capacity(n);

    // Generate quadrature carriers and multiply
    for (t, &x) in signal.iter().enumerate() {
        let phase = 2.0 * PI * fc * (t as f32) / fs;
        i_comp.push(x * phase.cos());
        q_comp.push(-x * phase.sin()); // minus for analytic convention
    }

    // Low-pass filter (simple moving average)
    let mut i_filt = vec![0.0; n];
    let mut q_filt = vec![0.0; n];
    if lp_len > 1 {
        let mut acc_i = 0.0;
        let mut acc_q = 0.0;
        for t in 0..n {
            acc_i += i_comp[t];
            acc_q += q_comp[t];
            if t >= lp_len {
                acc_i -= i_comp[t - lp_len];
                acc_q -= q_comp[t - lp_len];
            }
            let denom = if t + 1 < lp_len { t + 1 } else { lp_len };
            i_filt[t] = acc_i / denom as f32;
            q_filt[t] = acc_q / denom as f32;
        }
    } else {
        i_filt.copy_from_slice(&i_comp);
        q_filt.copy_from_slice(&q_comp);
    }

    // Combine into complex baseband signal
    i_filt
        .into_iter()
        .zip(q_filt.into_iter())
        .map(|(i, q)| Complex32::new(i, q))
        .collect()
}

/// Convenience: get envelope directly
pub fn hilbert_demod_envelope(signal: &[f32], fs: f32, fc: f32, lp_len: usize) -> Vec<f32> {
    hilbert_demod(signal, fs, fc, lp_len)
        .iter()
        .map(|c| c.norm())
        .collect()
}

/// Hilbert envelope (magnitude only)
pub fn hilbert_envelope(input: &[f32]) -> Vec<f32> {
    let z = hilbert_envelope_complex(input);
    z.iter().map(|c| c.norm()).collect()
}

/// Hilbert analytic signal (FFT-based)
/// Returns complex time series (same length as input)
pub fn hilbert_envelope_complex(input: &[f32]) -> Vec<Complex32> {
    let n0 = input.len();
    let n = n0.next_power_of_two(); // zero-padding for better Hilbert

    // --- forward FFT ---
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut x: Vec<Complex32> = input.iter().map(|&v| Complex32::new(v, 0.0)).collect();
    x.resize(n, Complex32::zero());

    fft.process(&mut x);

    // --- construct analytic spectrum ---
    for i in 0..n {
        if i == 0 {
            // DC keep
        } else if i < n / 2 {
            // positive freqs doubled
            x[i] *= Complex32::new(2.0, 0.0);
        } else if i == n / 2 {
            // Nyquist (even n) keep
        } else {
            // negative freqs zeroed
            x[i] = Complex32::zero();
        }
    }

    // --- inverse FFT ---
    ifft.process(&mut x);

    // normalize (rustfft does not scale ifft)
    let scale = 1.0 / n as f32;
    for xi in x.iter_mut() {
        *xi *= scale;
    }

    // return only original length
    x[..n0].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(fs: f32, f: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f * (i as f32) / fs).sin())
            .collect()
    }

    #[test]
    fn hilbert_on_sine_wave_gives_flat_envelope() {
        let fs = 16000.0;
        let f = 440.0;
        let n = 1024;
        let x = sine(fs, f, n);

        let env = hilbert_envelope(&x);
        let mean = env.iter().sum::<f32>() / (n as f32);

        assert!((mean - 1.0).abs() < 0.2, "mean envelope = {}", mean);
    }

    #[test]
    fn hilbert_on_dc_signal() {
        let x = vec![1.0f32; 512];
        let env = hilbert_envelope(&x);
        let mean = env.iter().sum::<f32>() / env.len() as f32;

        assert!((mean - 1.0).abs() < 0.01, "mean envelope = {}", mean);
    }

    #[test]
    fn hilbert_on_impulse_gives_flat_envelope() {
        let mut x = vec![0.0f32; 256];
        x[0] = 1.0;
        let env = hilbert_envelope(&x);

        // mean should be close to 0
        let mean = env.iter().sum::<f32>() / env.len() as f32;
        assert!(mean.abs() < 0.1, "mean envelope = {}", mean);

        // max should be significantly above 0.5
        let max_val = env.iter().cloned().fold(0.0, f32::max);
        assert!(max_val > 0.5, "max envelope = {}", max_val);
    }

    #[test]
    fn hilbert_on_two_tone_gives_beating_envelope() {
        let fs = 16000.0;
        let f1 = 440.0;
        let f2 = 445.0;
        let n = 4096;
        let mut x = vec![0.0f32; n];
        for i in 0..n {
            let t = i as f32 / fs;
            x[i] = (2.0 * std::f32::consts::PI * f1 * t).sin()
                + (2.0 * std::f32::consts::PI * f2 * t).sin();
        }

        let env = hilbert_envelope(&x);
        let mean = env.iter().sum::<f32>() / env.len() as f32;

        // envelope should fluctuate significantly
        let max_val = env.iter().cloned().fold(f32::MIN, f32::max);
        let min_val = env.iter().cloned().fold(f32::MAX, f32::min);

        assert!(max_val - min_val > 0.2 * mean, "beating not visible");
    }
}
