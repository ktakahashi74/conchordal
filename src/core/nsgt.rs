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

use crate::core::log2::Log2Space;
use rustfft::num_complex::Complex32;

// =====================================================
// Config structures
// =====================================================

/// NSGT configuration (log2-axis, analysis-only).
#[derive(Clone, Copy, Debug)]
pub struct NsgtLog2Config {
    /// Sampling rate [Hz]
    pub fs: f32,
    /// Minimum frequency [Hz] (inclusive)
    pub fmin: f32,
    /// Maximum frequency [Hz] (inclusive)
    pub fmax: f32,
    /// Bins per octave (e.g., 12=semitone, 24=quarter-tone)
    pub bins_per_oct: u32,
    /// Overlap ratio in [0, 0.95). 0.5 = 50% overlap (default-good)
    pub overlap: f32,
}

impl Default for NsgtLog2Config {
    fn default() -> Self {
        Self {
            fs: 48_000.0,
            fmin: 27.5,    // A0
            fmax: 8_000.0, // upper harmonic limit
            bins_per_oct: 96,
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
    cfg: NsgtLog2Config,
    bands: Vec<NsgtBand>,
    space: Log2Space,
}

impl NsgtLog2 {
    /// Construct analyzer with log2-spaced bands and band-dependent windows.
    pub fn new(cfg: NsgtLog2Config) -> Self {
        assert!(
            cfg.fmin > 0.0 && cfg.fmax > cfg.fmin,
            "Invalid frequency range"
        );
        assert!(
            cfg.overlap >= 0.0 && cfg.overlap < 0.95,
            "Overlap must be in [0,0.95)"
        );

        let fs = cfg.fs;
        let bpo = cfg.bins_per_oct as f32;
        let q = 1.0 / (2f32.powf(1.0 / bpo) - 1.0);

        // --- build log2 frequency grid
        let space = Log2Space::new(cfg.fmin, cfg.fmax, cfg.bins_per_oct);

        let bands: Vec<NsgtBand> = space
            .centers_hz
            .iter()
            .zip(space.centers_log2.iter())
            .map(|(&f, &log2_f)| {
                let mut win_len = (q * fs / f).ceil() as usize;
                win_len = win_len.max(16);
                if win_len % 2 == 0 {
                    win_len += 1;
                }
                let hop = ((1.0 - cfg.overlap) * win_len as f32).round().max(1.0) as usize;
                let window = hann_periodic(win_len);
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

        Self { cfg, bands, space }
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

    /// Mean envelope magnitude per band (for potential-R roughness kernels).
    pub fn analyze_envelope(&self, signal: &[f32]) -> (Vec<f32>, Vec<f32>) {
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
        let freqs_log2 = bands.iter().map(|b| b.log2_hz).collect();
        (amps, freqs_log2)
    }

    /// Representative analytic vector per band (for potential-C consonance kernels).
    pub fn analyze_flattened(&self, signal: &[f32]) -> Vec<Complex32> {
        self.analyze(signal)
            .iter()
            .filter_map(|b| b.coeffs.get(b.coeffs.len() / 2).copied())
            .collect()
    }
}

// =====================================================
// Internal utilities
// =====================================================

fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * (i as f32) / n as f32).cos()))
        .collect()
}

/// Windowed Goertzel per band across time.
fn analyze_band_goertzel(x: &[f32], fs: f32, band: &NsgtBand) -> (Vec<Complex32>, Vec<usize>) {
    let n = x.len();
    let L = band.win_len;
    let h = band.hop;
    let f = band.f_hz;
    let omega = 2.0 * std::f32::consts::PI * f / fs;

    let mut centers = Vec::new();
    let mut c = L / 2;
    while c + L / 2 <= n + (L / 2) {
        centers.push(c);
        c += h;
        if c == 0 {
            break;
        }
    }

    let mut coeffs = Vec::with_capacity(centers.len());
    for &c in &centers {
        let start = c.saturating_sub(L / 2);
        let mut acc_re = 0.0;
        let mut acc_im = 0.0;
        for i in 0..L {
            let xi = if start + i < n { x[start + i] } else { 0.0 };
            let w = band.window[i];
            let ph = omega * (start + i) as f32;
            acc_re += xi * w * ph.cos();
            acc_im -= xi * w * ph.sin();
        }
        let norm = (2.0 / L as f32).sqrt();
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
        let nsgt = NsgtLog2::new(NsgtLog2Config {
            fs,
            fmin: 55.0,
            fmax: 4000.0,
            bins_per_oct: 24,
            overlap: 0.5,
        });
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
        let nsgt = NsgtLog2::new(NsgtLog2Config::default());
        for w in nsgt.bands.windows(2) {
            if w[0].f_hz < w[1].f_hz {
                assert!(w[0].win_len >= w[1].win_len);
            }
        }
    }

    #[test]
    fn empty_signal_returns_empty() {
        let nsgt = NsgtLog2::new(NsgtLog2Config::default());
        let out = nsgt.analyze(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn high_frequency_shorter_window() {
        let nsgt = NsgtLog2::new(NsgtLog2Config::default());
        let low = &nsgt.bands[0];
        let high = nsgt.bands.last().unwrap();
        assert!(low.win_len > high.win_len);
    }

    #[test]
    fn power_consistency_between_nearby_bins() {
        let fs = 48000.0;
        let nsgt = NsgtLog2::new(NsgtLog2Config {
            fs,
            fmin: 100.0,
            fmax: 1000.0,
            bins_per_oct: 12,
            overlap: 0.5,
        });
        let sig = mk_sine(fs, 440.0, 0.5);
        let bands = nsgt.analyze(&sig);

        let mut powers: Vec<(f32, f32)> = bands
            .iter()
            .map(|b| {
                let p = b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>()
                    / (b.coeffs.len().max(1) as f32);
                (b.f_hz, p)
            })
            .collect();

        powers.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let idx = powers.iter().position(|(f, _)| *f > 440.0).unwrap_or(1);
        let (f1, p1) = powers[idx - 1];
        let (f2, p2) = powers[idx];
        assert!(p1 > 0.0 && p2 > 0.0);
        assert_relative_eq!(p1, p2, epsilon = 0.5, max_relative = 0.5);
    }
}
