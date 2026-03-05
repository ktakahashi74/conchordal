//! core/nsgt.rs — Log2-axis Non-Stationary Gabor (analysis-only, Goertzel-based)
//!
//! Overview
//! --------
//! - Uses [`Log2Space`] grid between [fmin, fmax] with `bins_per_oct` resolution.
//! - Band-dependent window length: L_k = ceil(Q * fs / f_k), Q = 1/(2^(1/B)-1).
//! - Analysis via windowed Goertzel per band & frame.
//!
//! Notes
//! -----
//! - Analysis-only (no dual frames yet).
//! - Designed for Conchordal: downstream modules use log2-axis envelopes or analytic signals.
//! - Window: periodic Hann, overlap ≥ 50% recommended.
//!
//! Performance
//! -----------
//! - Goertzel avoids per-frame FFT overhead, efficient for sparse-band constant-Q.
//! - Dense or long-window use cases may prefer batched FFT (TODO).

use crate::core::log2space::Log2Space;
use rustfft::num_complex::Complex32;

// =====================================================
// Config structures
// =====================================================

/// NSGT configuration (log2-axis, analysis-only).
#[derive(Clone, Copy, Debug)]
pub struct NsgtLog2Config {
    /// Sampling rate [Hz]
    pub fs: f32,
    /// Overlap ratio in [0, 0.95). 0.5 = 50% overlap (default-good)
    pub overlap: f32,
}

impl Default for NsgtLog2Config {
    fn default() -> Self {
        Self {
            fs: 48_000.0,
            overlap: 0.5,
        }
    }
}

/// Band descriptor on log2 axis.
#[derive(Clone, Debug)]
pub struct NsgtBand {
    pub f_hz: f32,
    pub win_len: usize,
    pub hop: usize,
    pub window: Vec<f32>,
    pub log2_hz: f32,
    pub q: f32,
}

/// Analysis result for one band.
#[derive(Clone, Debug)]
pub struct BandCoeffs {
    pub coeffs: Vec<Complex32>,
    pub t_sec: Vec<f32>,
    pub f_hz: f32,
    pub log2_hz: f32,
    pub win_len: usize,
    pub hop: usize,
}

// =====================================================
// Analyzer core
// =====================================================

#[derive(Clone, Debug)]
pub struct NsgtLog2 {
    pub cfg: NsgtLog2Config,
    pub bands: Vec<NsgtBand>,
    pub space: Log2Space,
}

impl NsgtLog2 {
    /// Construct analyzer with log2-spaced bands and band-dependent windows.
    pub fn new(cfg: NsgtLog2Config, space: Log2Space) -> Self {
        assert!(
            cfg.overlap >= 0.0 && cfg.overlap < 0.95,
            "Overlap must be in [0,0.95)"
        );

        let fs = cfg.fs;
        let bpo = space.bins_per_oct as f32;
        let q = 1.0 / (2f32.powf(1.0 / bpo) - 1.0);

        // --- use given log2-space to define band centers
        let bands: Vec<NsgtBand> = space
            .centers_hz
            .iter()
            .zip(space.centers_log2.iter())
            .map(|(&f, &log2_f)| {
                let mut win_len = (q * fs / f).ceil() as usize;
                win_len = win_len.max(16);
                if win_len.is_multiple_of(2) {
                    win_len += 1;
                }
                let hop = ((1.0 - cfg.overlap) * win_len as f32).round().max(1.0) as usize;
                let window = crate::core::fft::hann_window_periodic(win_len);
                NsgtBand {
                    f_hz: f,
                    win_len,
                    hop,
                    window,
                    log2_hz: log2_f,
                    q,
                }
            })
            .collect();

        Self {
            cfg,
            bands,
            space: space.clone(),
        }
    }

    pub fn bands(&self) -> &[NsgtBand] {
        &self.bands
    }

    pub fn space(&self) -> &Log2Space {
        &self.space
    }

    pub fn freqs_hz(&self) -> Vec<f32> {
        self.space.centers_hz.clone()
    }

    pub fn hop_s(&self) -> f32 {
        let mean_hop =
            self.bands.iter().map(|b| b.hop as f32).sum::<f32>() / self.bands.len().max(1) as f32;
        mean_hop / self.cfg.fs
    }

    /// Full NSGT analysis returning complex coefficients per band.
    pub fn analyze(&self, signal: &[f32]) -> Vec<BandCoeffs> {
        let n = signal.len();
        if n == 0 {
            return Vec::new();
        }
        let fs = self.cfg.fs;
        let mut out = Vec::with_capacity(self.bands.len());
        for b in &self.bands {
            let (coeffs, t_idx) = analyze_band_goertzel(signal, fs, b);
            let t_sec = t_idx.into_iter().map(|i| i as f32 / fs).collect();
            out.push(BandCoeffs {
                coeffs,
                t_sec,
                f_hz: b.f_hz,
                log2_hz: b.log2_hz,
                win_len: b.win_len,
                hop: b.hop,
            });
        }
        out
    }

    /// Mean envelope magnitude per band (for perc_potential_R roughness kernels).
    pub fn analyze_envelope(&self, signal: &[f32]) -> Vec<f32> {
        let bands = self.analyze(signal);
        let amps: Vec<f32> = bands
            .iter()
            .map(|b| {
                if b.coeffs.is_empty() {
                    0.0
                } else {
                    b.coeffs.iter().map(|z| z.norm()).sum::<f32>() / b.coeffs.len() as f32
                }
            })
            .collect();
        amps
    }

    /// Representative analytic vector per band (for perc_potential_H kernels).
    pub fn analyze_flattened(&self, signal: &[f32]) -> Vec<Complex32> {
        self.analyze(signal)
            .iter()
            .filter_map(|b| b.coeffs.get(b.coeffs.len() / 2).copied())
            .collect()
    }

    /// Compute PSD-like estimate from an existing `analyze()` result.
    pub fn analyze_psd_from_bands(&self, bands: &[BandCoeffs]) -> Vec<f32> {
        if bands.is_empty() {
            return Vec::new();
        }

        let mut psd_vals = Vec::with_capacity(bands.len());

        for b in bands {
            if b.coeffs.is_empty() {
                psd_vals.push(0.0);
                continue;
            }

            // (1) mean power
            let mean_pow =
                b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>() / (b.coeffs.len().max(1) as f32);

            // (2) bandwidth
            let bw_hz = self.space.bandwidth_hz(b.f_hz);

            // (3) moderate correction → multiply by sqrt(Δf)
            let psd_adj = mean_pow * bw_hz.sqrt();

            psd_vals.push(psd_adj);
        }

        psd_vals
    }

    /// Convenience version: run full analysis and return PSD directly.
    pub fn analyze_psd(&self, signal: &[f32]) -> Vec<f32> {
        let bands = self.analyze(signal);
        self.analyze_psd_from_bands(&bands)
    }
}

// =====================================================
// Internal utilities
// =====================================================

// /// Windowed Goertzel per band across time.
fn analyze_band_goertzel(x: &[f32], fs: f32, band: &NsgtBand) -> (Vec<Complex32>, Vec<usize>) {
    let n = x.len();
    let win_len = band.win_len;
    let h = band.hop;
    let f = band.f_hz;
    let omega = 2.0 * std::f32::consts::PI * f / fs;

    let mut centers = Vec::new();
    let mut c = win_len / 2;
    while c + win_len / 2 <= n + (win_len / 2) {
        centers.push(c);
        c += h;
        if c == 0 {
            break;
        }
    }

    let mut coeffs = Vec::with_capacity(centers.len());
    for &c in &centers {
        let start = c.saturating_sub(win_len / 2);
        let mut acc_re = 0.0;
        let mut acc_im = 0.0;
        for i in 0..win_len {
            let xi = if start + i < n { x[start + i] } else { 0.0 };
            let w = band.window[i];
            let ph = omega * (start + i) as f32;
            acc_re += xi * w * ph.cos();
            acc_im -= xi * w * ph.sin();
        }
        let norm = (2.0 / win_len as f32).sqrt();
        coeffs.push(Complex32::new(acc_re * norm, acc_im * norm));
    }
    (coeffs, centers)
}

// =====================================================
// Tests
// =====================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::log2space::Log2Space;
    use approx::assert_relative_eq;

    fn mk_sine(fs: f32, f: f32, secs: f32) -> Vec<f32> {
        let n = (fs * secs).round() as usize;
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f * (i as f32) / fs).sin())
            .collect()
    }

    #[test]
    fn pure_tone_hits_right_band() {
        let fs = 48_000.0;
        let nsgt = NsgtLog2::new(
            NsgtLog2Config { fs, overlap: 0.5 },
            Log2Space::new(20.0, 8000.0, 48),
        );
        let sig = mk_sine(fs, 440.0, 1.0);
        let bands = nsgt.analyze(&sig);

        let (mut best_f, mut best_val) = (0.0, 0.0);
        for b in &bands {
            let p =
                b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>() / (b.coeffs.len().max(1) as f32);
            if p > best_val {
                best_val = p;
                best_f = b.f_hz;
            }
        }
        let cents = 1200.0 * ((best_f / 440.0).log2().abs());
        assert!(
            cents < 60.0,
            "440Hz peak band off by {:.1} cents (f={:.2})",
            cents,
            best_f
        );
    }

    #[test]
    fn window_len_monotonic_vs_freq() {
        let nsgt = NsgtLog2::new(NsgtLog2Config::default(), Log2Space::new(20.0, 8000.0, 48));
        for w in nsgt.bands.windows(2) {
            if w[0].f_hz < w[1].f_hz {
                assert!(w[0].win_len >= w[1].win_len);
            }
        }
    }

    #[test]
    fn empty_signal_returns_empty() {
        let nsgt = NsgtLog2::new(NsgtLog2Config::default(), Log2Space::new(20.0, 8000.0, 48));
        let out = nsgt.analyze(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn low_vs_high_freq_energy_scaling() {
        let fs = 48000.0;
        let nsgt = NsgtLog2::new(NsgtLog2Config::default(), Log2Space::new(20.0, 8000.0, 96));
        let sig_low = mk_sine(fs, 220.0, 1.0);
        let sig_high = mk_sine(fs, 1760.0, 1.0);
        let p_low = nsgt
            .analyze(&sig_low)
            .iter()
            .map(|b| b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>())
            .sum::<f32>();
        let p_high = nsgt
            .analyze(&sig_high)
            .iter()
            .map(|b| b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>())
            .sum::<f32>();
        let ratio = p_high / p_low;
        assert_relative_eq!(ratio, 1.0, epsilon = 0.3, max_relative = 0.3);
    }

    #[test]
    fn amplitude_linearity() {
        let fs = 48000.0;
        let nsgt = NsgtLog2::new(NsgtLog2Config::default(), Log2Space::new(20.0, 8000.0, 96));
        let sig1 = mk_sine(fs, 440.0, 1.0);
        let sig2: Vec<f32> = sig1.iter().map(|v| v * 2.0).collect();

        let e1 = nsgt
            .analyze(&sig1)
            .iter()
            .map(|b| b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>())
            .sum::<f32>();
        let e2 = nsgt
            .analyze(&sig2)
            .iter()
            .map(|b| b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>())
            .sum::<f32>();
        assert_relative_eq!(e2 / e1, 4.0, epsilon = 0.1, max_relative = 0.1);
    }
    #[test]
    fn time_invariance_of_magnitude() {
        let fs = 48000.0;
        let nsgt = NsgtLog2::new(NsgtLog2Config::default(), Log2Space::new(20.0, 8000.0, 96));
        let sig = mk_sine(fs, 440.0, 1.0);
        let shift = (0.1 * fs) as usize;
        let mut sig_shifted = vec![0.0; shift];
        sig_shifted.extend_from_slice(&sig);

        let amp1 = nsgt.analyze_envelope(&sig);
        let amp2 = nsgt.analyze_envelope(&sig_shifted);
        let corr: f32 = amp1.iter().zip(&amp2).map(|(a, b)| a * b).sum::<f32>()
            / (amp1.iter().map(|a| a * a).sum::<f32>().sqrt()
                * amp2.iter().map(|b| b * b).sum::<f32>().sqrt());
        assert!(
            corr > 0.95,
            "Magnitude envelope correlation should be high (corr={corr:.3})"
        );
    }

    #[test]
    fn hop_size_stability() {
        let fs = 48000.0;
        let space = Log2Space::new(20.0, 8000.0, 96);
        let sig = mk_sine(fs, 440.0, 1.0);

        let e_mean = |overlap: f32| {
            let nsgt = NsgtLog2::new(NsgtLog2Config { fs, overlap }, space.clone());
            let out = nsgt.analyze(&sig);
            let sum_e: f32 = out
                .iter()
                .map(|b| b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>())
                .sum();
            let n_frames: usize = out.iter().map(|b| b.coeffs.len()).max().unwrap_or(1);
            sum_e / n_frames as f32
        };

        let e_half = e_mean(0.5);
        let e_75 = e_mean(0.75);

        assert_relative_eq!(e_half, e_75, epsilon = 0.3, max_relative = 0.3);
    }

    #[test]
    fn band_count_matches_space() {
        let space = Log2Space::new(20.0, 8000.0, 96);
        let nsgt = NsgtLog2::new(NsgtLog2Config::default(), space.clone());
        assert_eq!(
            nsgt.bands().len(),
            space.centers_hz.len(),
            "Band count must match Log2Space bin count"
        );
    }
}
