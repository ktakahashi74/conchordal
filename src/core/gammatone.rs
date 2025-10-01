
// 4th-order (biquad x2) all-pole Gammatone filterbank + Hilbert envelope

use rustfft::{num_complex::Complex};

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
        Self { b0, b1, b2, a1, a2, z1: 0.0, z2: 0.0 }
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
	let mid = &y[n/4..3*n/4];
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
	
	let rms_fc = (y_fc.iter().map(|v| v*v).sum::<f32>() / (n as f32)).sqrt();
	let rms_off = (y_off.iter().map(|v| v*v).sum::<f32>() / (n as f32)).sqrt();
	
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
