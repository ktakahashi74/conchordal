use ndarray::prelude::*;

#[derive(Clone, Debug)]
pub struct LandscapeParams {
    pub alpha: f32, // K = alpha*C - beta*R (here C=0)
    pub beta: f32,
    pub fmin: f32,
    pub fmax: f32,
}

#[derive(Clone, Debug, Default)]
pub struct LandscapeFrame {
    pub freqs_hz: Vec<f32>,
    pub r: Vec<f32>,
    pub c: Vec<f32>,
    pub k: Vec<f32>,
}

/// Compute a very simplified R_v1-like roughness over frequency bins from amplitude spectrum.
/// Here, C is zero (dummy), and K = alpha*C - beta*R.
pub fn compute_landscape(amps: &[f32], fs: f32, n_fft: usize, p: &LandscapeParams) -> LandscapeFrame {
    let n_bins = amps.len();
    let freqs: Vec<f32> = (0..n_bins).map(|k| k as f32 * fs / n_fft as f32).collect();

    // Roughness kernel: simple exponential of normalized distance (placeholder).
    // R(f_k) = sum_j a_k * a_j * exp(-|f_k - f_j| / bw(f_k))
    // Use a crude ERB-like bandwidth bw = 3.5 + 0.075 f (Hz). (toy model)
    let a = Array1::from_vec(amps.to_vec());
    let mut r = vec![0.0f32; n_bins];

    for k in 0..n_bins {
        let fk = freqs[k];
        let bw = 3.5 + 0.075 * fk;
        let mut acc = 0.0f32;
        for j in 0..n_bins {
            if j == k { continue; }
            let fj = freqs[j];
            let w = (-(fk - fj).abs() / bw).exp();
            acc += a[k] * a[j] * w;
        }
        r[k] = acc;
    }

    // Consonance dummy = zeros
    let c = vec![0.0f32; n_bins];

    // K = alpha*C - beta*R
    let k: Vec<f32> = r.iter().zip(c.iter()).map(|(rr, cc)| p.alpha * *cc - p.beta * *rr).collect();

    LandscapeFrame { freqs_hz: freqs, r, c, k }
}
