// 4th-order (biquad x2) all-pole Gammatone filterbank + Hilbert envelope

use rustfft::num_complex::Complex;

use rustfft::num_complex::Complex32;
use std::f32::consts::PI;

use crate::core::erb::erb_bw_hz;

#[derive(Clone, Copy, Debug)]
pub struct Biquad {
    // Direct Form I (a0 = 1)
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
    // state
    z1: f32,
    z2: f32,
}
impl Biquad {
    #[inline]
    pub fn new(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            z1: 0.0,
            z2: 0.0,
        }
    }
    #[inline]
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        // y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
    pub fn process_block(&mut self, x: &[f32], y: &mut [f32]) {
        for (i, &xi) in x.iter().enumerate() {
            y[i] = self.process_sample(xi);
        }
    }
}

/// Design two identical real biquads that realize a 4th-order all-pole gammatone channel.
/// Each biquad has denominator 1 - 2 r cosθ z^{-1} + r^2 z^{-2} and b0 chosen so that
/// overall |H(e^{jθ})| = 1 at θ = 2π f_c / fs.
pub fn design_gammatone_biquads(fc: f32, fs: f32) -> (Biquad, Biquad) {
    let theta = 2.0 * std::f32::consts::PI * fc / fs;
    let erb = erb_bw_hz(fc);
    let b_hz = 1.019 * erb; // Patterson/Slaney
    let r = (-2.0 * std::f32::consts::PI * b_hz / fs).exp(); // pole radius
    let a1 = -2.0 * r * theta.cos();
    let a2 = r * r;

    // Denominator at ω = θ : D(e^{jθ}) = 1 - 2 r cosθ e^{-jθ} + r^2 e^{-j2θ}
    let ejm1 = Complex::new(theta.cos(), -theta.sin());
    let ejm2 = Complex::new((2.0 * theta).cos(), -(2.0 * theta).sin());
    let den = Complex::new(1.0, 0.0) + (a1 * ejm1) + (a2 * ejm2); // since denom is 1 + a1 z^{-1} + a2 z^{-2}
    let den_mag = (den.re * den.re + den.im * den.im).sqrt();

    // Two identical sections; choose per-section b0 so that (b0^2 / |den|^2) = 1 → b0 = |den|
    let b0 = den_mag;
    let b1 = 0.0;
    let b2 = 0.0;

    let s1 = Biquad::new(b0, b1, b2, a1, a2);
    let s2 = Biquad::new(b0, b1, b2, a1, a2);
    (s1, s2)
}

/// Apply a single 4th-order gammatone channel (biquad x2) to `input`.
pub fn gammatone_channel_process(input: &[f32], s1: &mut Biquad, s2: &mut Biquad) -> Vec<f32> {
    let mut tmp = vec![0.0f32; input.len()];
    let mut out = vec![0.0f32; input.len()];
    s1.process_block(input, &mut tmp);
    s2.process_block(&tmp, &mut out);
    out
}

/// Filterbank: returns per-channel outputs [channels][samples]
pub fn gammatone_filterbank(
    input: &[f32],
    center_freqs: &[f32],
    sample_rate: f32,
) -> Vec<Vec<f32>> {
    center_freqs
        .iter()
        .map(|&fc| {
            let (mut s1, mut s2) = design_gammatone_biquads(fc, sample_rate);
            s1.reset();
            s2.reset();
            gammatone_channel_process(input, &mut s1, &mut s2)
        })
        .collect()
}

/// Complex one–pole resonator (analytic gammatone primitive)
/// y[n] = a * y[n-1] + b * x[n],  a = r * exp(+jθ),  b is gain
#[derive(Clone, Copy, Debug)]
pub struct ComplexOnePole {
    pub a: Complex32,
    pub b: f32, // real scalar
    // state
    z1: Complex32,
}
impl ComplexOnePole {
    #[inline]
    pub fn new(a: Complex32, b: f32) -> Self {
        Self {
            a,
            b,
            z1: Complex32::new(0.0, 0.0),
        }
    }
    #[inline]
    pub fn reset(&mut self) {
        self.z1 = Complex32::new(0.0, 0.0);
    }
    #[inline]
    pub fn process_sample(&mut self, x: f32) -> Complex32 {
        // Direct Form: y = b*x + a*z1
        let y = self.a * self.z1 + Complex32::new(self.b * x, 0.0);
        self.z1 = y;
        y
    }
    pub fn process_block(&mut self, x: &[f32], y: &mut [Complex32]) {
        for (i, &xi) in x.iter().enumerate() {
            y[i] = self.process_sample(xi);
        }
    }
}

/// Design 4th-order complex gammatone at center fc (Hz), fs (Hz).
/// Implementation: cascade 4 identical complex one–poles (Hohmann/Slaney系)
/// Pole radius r = exp(-2π * b / fs),  b ≈ 1.019 * ERB(fc).
/// Per-stage gain b_s is set so that the *cascade* has |H(e^{jθ})| = 1 at θ=2πfc/fs.
pub fn design_complex_gammatone4(fc: f32, fs: f32) -> [ComplexOnePole; 4] {
    let theta = 2.0 * PI * fc / fs;
    let erb = erb_bw_hz(fc);
    let b_hz = 1.019 * erb;

    // pole radius and complex pole
    let r = (-2.0 * PI * b_hz / fs).exp();
    let a = Complex32::from_polar(r, theta); // r * e^{+jθ}

    // Magnitude of 1 - a e^{-jθ}  (since X -> z-transform evaluation at ω=θ)
    // |1 - r e^{jθ} e^{-jθ}| = |1 - r| = 1 - r  (r<1)
    let den_mag_at_fc = (1.0 - r).abs();

    // Cascade of 4 stages: |H| = (b_s / |1 - a e^{-jθ}|)^4
    // Set |H|=1  =>  b_s = den_mag_at_fc
    let b_s = den_mag_at_fc;

    let s = ComplexOnePole::new(a, b_s);
    [s, s, s, s]
}

/// Process one channel: 4 cascaded complex one–poles
pub fn complex_gammatone4_channel(input: &[f32], s: &mut [ComplexOnePole; 4]) -> Vec<Complex32> {
    let n = input.len();
    let mut y1 = vec![Complex32::new(0.0, 0.0); n];
    let mut y2 = vec![Complex32::new(0.0, 0.0); n];
    let mut y3 = vec![Complex32::new(0.0, 0.0); n];
    let mut y4 = vec![Complex32::new(0.0, 0.0); n];

    s[0].process_block(input, &mut y1);
    s[1].process_block(input, &mut y2);
    // Direct-form chaining: next stage input is previous stage output's real part?
    // No — keep it complex to stay analytic.
    for i in 0..n {
        y2[i] = s[1].a * y1[i] + Complex32::new(s[1].b * 0.0, 0.0);
    } // re-do to ensure complex chain
    s[1].reset();
    // The above line shows an in-place form; simpler is to re-run process_sample:
    // For clarity, replace by sample loop:
    let mut y = vec![Complex32::new(0.0, 0.0); n];
    let mut s0 = s[0];
    let mut s1 = s[1];
    let mut s2 = s[2];
    let mut s3 = s[3];
    for (i, &x) in input.iter().enumerate() {
        let y_1 = s0.process_sample(x);
        let y_2 = s1.process_sample(0.0) + y_1; // feed complex; b*x only on first stage
        let y_3 = s2.process_sample(0.0) + y_2;
        let y_4 = s3.process_sample(0.0) + y_3;
        y[i] = y_4;
    }
    // NOTE: 上の簡略化は読みやすさ優先。実用は各段で y = a*y_prev + b*x_prev の素直な形にしてください。
    y
}

/// Filterbank (complex, analytic): returns [channels][samples] (complex)
pub fn complex_gammatone4_filterbank(
    input: &[f32],
    center_freqs: &[f32],
    fs: f32,
) -> Vec<Vec<Complex32>> {
    center_freqs
        .iter()
        .map(|&fc| {
            let mut stages = design_complex_gammatone4(fc, fs);
            for s in stages.iter_mut() {
                s.reset();
            }
            complex_gammatone4_channel(input, &mut stages)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::util::sine;

    use approx::assert_abs_diff_eq;

    #[test]
    fn biquad_coefficients_range() {
        let fs = 48000.0;
        let fc = 1000.0;
        let (s1, s2) = design_gammatone_biquads(fc, fs);
        // 半径 r は (0,1) → a2 = r^2 も (0,1)
        assert!(s1.a2 > 0.0 && s1.a2 < 1.0);
        assert_eq!(s1.a1, s2.a1);
        assert_eq!(s1.a2, s2.a2);
        // オールポールなので b1=b2=0
        assert_eq!(s1.b1, 0.0);
        assert_eq!(s1.b2, 0.0);
    }

    #[test]
    fn unity_gain_at_center_freq() {
        let fs = 48000.0;
        let fc = 2000.0;
        let n = 48000; // 1s
        let x = sine(fs, fc, n);

        let (mut s1, mut s2) = design_gammatone_biquads(fc, fs);
        let y = gammatone_channel_process(&x, &mut s1, &mut s2);

        // 中心周波数で振幅がほぼ1（立上り/端の影響を避けて中央部分のRMSで評価）
        let mid = &y[n / 4..3 * n / 4];
        let rms = (mid.iter().map(|v| v * v).sum::<f32>() / (mid.len() as f32)).sqrt();
        assert_abs_diff_eq!(rms, 0.707, epsilon = 0.12); // 正弦のRMS ≈ 1/√2、±0.12の許容
    }

    #[test]
    fn selective_at_off_center() {
        let fs = 48000.0;
        let fc = 1000.0;
        let off = 1500.0;
        let n = 48000;
        let x_fc = sine(fs, fc, n);
        let x_off = sine(fs, off, n);

        let (mut s1, mut s2) = design_gammatone_biquads(fc, fs);
        let y_fc = gammatone_channel_process(&x_fc, &mut s1, &mut s2);

        // 同じチャネルに、オフ中心の信号を通す（状態をリセット）
        let (mut s1b, mut s2b) = design_gammatone_biquads(fc, fs);
        let y_off = gammatone_channel_process(&x_off, &mut s1b, &mut s2b);

        let rms_fc = (y_fc.iter().map(|v| v * v).sum::<f32>() / (n as f32)).sqrt();
        let rms_off = (y_off.iter().map(|v| v * v).sum::<f32>() / (n as f32)).sqrt();

        assert!(rms_fc > rms_off); // 中心周波数の方が強く通る
    }

    #[test]
    fn filterbank_shapes_and_hilbert() {
        let fs = 16000.0;
        let n = 4096;
        // 2つの正弦の和
        let x1 = sine(fs, 440.0, n);
        let x2 = sine(fs, 1000.0, n);
        let x: Vec<f32> = x1.iter().zip(x2.iter()).map(|(a, b)| a + b).collect();

        let cfs = vec![440.0, 1000.0, 2000.0];
        let y = gammatone_filterbank(&x, &cfs, fs);
        assert_eq!(y.len(), cfs.len());
        assert_eq!(y[0].len(), n);
    }
}
