use rustfft::{num_complex::Complex, num_traits::Zero, FftPlanner};

/// Hilbert envelope (FFT-based)
pub fn hilbert_envelope(input: &[f32]) -> Vec<f32> {
    let n0 = input.len();
    let n = n0.next_power_of_two();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut x: Vec<Complex<f32>> = input.iter().map(|&v| Complex{re: v, im: 0.0}).collect();
    x.resize(n, Complex::zero());

    fft.process(&mut x);

    // analytic signal spectrum window
    for i in 0..n {
        if i == 0 {
            // DC
            // leave as is
        } else if i < n/2 {
            x[i] *= Complex::new(2.0, 0.0);
        } else if i == n/2 {
            // Nyquist (even n)
            // leave as is
        } else {
            x[i] = Complex::zero();
        }
    }

    ifft.process(&mut x);

    
    let scale = 1.0 / (n as f32);
    x.iter_mut().for_each(|c| { c.re *= scale; c.im *= scale; });

    
    x[..n0].iter().map(|c| (c.re * c.re + c.im * c.im).sqrt()).collect()
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

}
