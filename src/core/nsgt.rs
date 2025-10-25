//! src/core/nsgt.rs — Log2-axis Non-Stationary Gabor (analysis-only, Goertzel-based)
//!
//! Overview
//! --------
//! - Log2(f) grid between [fmin, fmax] with `bins_per_oct` resolution.
//! - Band-dependent window length: L_k = ceil(Q * fs / f_k), Q = 1/(2^(1/B)-1).
//! - Analysis via windowed Goertzel (single-bin DFT) per band & frame.
//! - Output = per-band time series of complex coefficients (variable rates).
//!
//! Notes
//! -----
//! - This is *analysis-only*; painless synthesis/dual frames are TODO.
//! - Designed for conchordal: downstream uses log2-axis landscapes (R/C/K).
//! - Windows: Hann (periodic) with >=50% overlap recommended.
//!
//! Perf
//! ----
//! - Goertzel avoids per-frame FFTs and is cache-friendly for single-bin eval.
//! - For dense bands or very long windows, consider batching or FFT later.
//!
//! Tests
//! -----
//! - `pure_tone_hits_right_band`: ensures 440 Hz peaks near the expected bin.

use rustfft::num_complex::Complex32;

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
            fmax: 8_000.0, // voice/instrument upper partials
            bins_per_oct: 24,
            overlap: 0.5,
        }
    }
}

/// Band descriptor on log2 axis.
#[derive(Clone, Debug)]
pub struct NsgtBand {
    /// Center frequency [Hz]
    pub f_hz: f32,
    /// Window length [samples]
    pub win_len: usize,
    /// Hop size [samples]
    pub hop: usize,
    /// Hann window (cached)
    pub window: Vec<f32>,
    /// Log2(f) coordinate
    pub log2_hz: f32,
    /// Constant-Q for this grid
    pub q: f32,
}

/// Analysis result for one band.
#[derive(Clone, Debug)]
pub struct BandCoeffs {
    /// Complex coefficients across time (frames)
    pub coeffs: Vec<Complex32>,
    /// Frame centers [seconds] aligned to input signal timebase
    pub t_sec: Vec<f32>,
    /// Band center frequency [Hz]
    pub f_hz: f32,
    /// Log2(f) coordinate
    pub log2_hz: f32,
    /// Window length used [samples]
    pub win_len: usize,
    /// Hop size used [samples]
    pub hop: usize,
}

/// Log2-axis NSGT analyzer (analysis-only).
#[derive(Clone, Debug)]
pub struct NsgtLog2 {
    cfg: NsgtLog2Config,
    bands: Vec<NsgtBand>,
}

impl NsgtLog2 {
    /// Build analyzer with log2-spaced bands and band-dependent windows.
    pub fn new(cfg: NsgtLog2Config) -> Self {
        assert!(cfg.fmin > 0.0 && cfg.fmax > cfg.fmin, "Invalid band edges");
        assert!(
            cfg.overlap >= 0.0 && cfg.overlap < 0.95,
            "overlap in [0,0.95)"
        );
        let fs = cfg.fs;
        let bpo = cfg.bins_per_oct as f32;

        let q = 1.0 / (2f32.powf(1.0 / bpo) - 1.0); // constant across grid
        let mut freqs = Vec::new();

        let log2_min = cfg.fmin.log2();
        let log2_max = cfg.fmax.log2();
        let n_steps = ((log2_max - log2_min) * bpo).ceil() as i32;

        for i in 0..=n_steps {
            let log2_f = log2_min + (i as f32) / bpo;
            let f = 2f32.powf(log2_f);
            if f >= cfg.fmin && f <= cfg.fmax {
                freqs.push((f, log2_f));
            }
        }

        let bands = freqs
            .into_iter()
            .map(|(f, log2_f)| {
                // L_k = ceil(Q * fs / f_k), enforce odd length >= 16
                let mut win_len = (q * fs / f).ceil() as usize;
                win_len = win_len.max(16);
                if win_len % 2 == 0 {
                    win_len += 1; // odd length centers nicely
                }

                // Hop based on overlap
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

        Self { cfg, bands }
    }

    /// Access bands (center freqs, windows, etc.).
    pub fn bands(&self) -> &[NsgtBand] {
        &self.bands
    }

    /// Analyze a real signal: returns per-band complex time series with their timestamps.
    ///
    /// For each band k, we slide a Hann window of length L_k with hop h_k,
    /// multiply the framed samples, and evaluate a single-bin DFT at f_k via Goertzel.
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
}

// ========================= Utilities ================================

/// Periodic Hann window (N points, last-to-first seamless for STFT).
fn hann_periodic(n: usize) -> Vec<f32> {
    // w[i] = 0.5 * (1 - cos(2π i / N))
    // "Periodic" form: denominator N (not N-1) to avoid duplicate endpoints.
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * (i as f32) / (n as f32)).cos()))
        .collect()
}

/// Analyze one band using windowed Goertzel across time.
/// Returns (coeffs, frame_center_sample_indices).
fn analyze_band_goertzel(x: &[f32], fs: f32, band: &NsgtBand) -> (Vec<Complex32>, Vec<usize>) {
    let n = x.len();
    let L = band.win_len;
    let h = band.hop;
    let f = band.f_hz;

    // Angular frequency per sample
    let omega = 2.0 * std::f32::consts::PI * f / fs;

    // Frame centers (sample indices)
    let mut centers = Vec::new();
    {
        let mut c = L / 2;
        while c + L / 2 <= n + (L / 2) {
            centers.push(c);
            if c + h <= c {
                break; // overflow safety
            }
            c += h;
            if c == 0 {
                break;
            }
        }
    }

    let mut coeffs = Vec::with_capacity(centers.len());

    // Temporary buffer with zero-padding fetch
    for &c in &centers {
        let start = c.saturating_sub(L / 2);
        let end_excl = start + L;

        // Accumulate windowed complex sinusoid projection at f
        let mut acc_re = 0.0f32;
        let mut acc_im = 0.0f32;

        for i in 0..L {
            let xi = if start + i < n { x[start + i] } else { 0.0 };
            let w = band.window[i];
            // phase at absolute sample index (start + i)
            let t = (start + i) as f32;
            let ph = omega * t;
            // multiply by e^{-j ph} to demodulate
            let cosph = ph.cos();
            let sinph = ph.sin();
            let v = xi * w;
            acc_re += v * cosph;
            acc_im -= v * sinph;
        }

        // Optional amplitude norm: √(2/L) matches STFT single-bin scaling.
        let norm = (2.0 / (L as f32)).sqrt();
        coeffs.push(Complex32::new(acc_re * norm, acc_im * norm));

        // Silence unused variable warning for end_excl (kept for clarity)
        let _ = end_excl;
    }

    (coeffs, centers)
}

// =========================== Tests ==================================

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_sine(fs: f32, f: f32, secs: f32) -> Vec<f32> {
        let n = (fs * secs).round() as usize;
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f * (i as f32) / fs).sin())
            .collect()
    }

    #[test]
    fn pure_tone_hits_right_band() {
        let fs = 48_000.0;
        let cfg = NsgtLog2Config {
            fs,
            fmin: 55.0,
            fmax: 4_000.0,
            bins_per_oct: 24,
            overlap: 0.5,
        };
        let nsgt = NsgtLog2::new(cfg);

        let sig = mk_sine(fs, 440.0, 1.0);

        let bands = nsgt.analyze(&sig);
        // Find max RMS band
        let mut best_idx = 0usize;
        let mut best_val = 0.0f32;
        for (i, b) in bands.iter().enumerate() {
            let p: f32 =
                b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>() / (b.coeffs.len().max(1) as f32);
            if p > best_val {
                best_val = p;
                best_idx = i;
            }
        }

        let f_peak = bands[best_idx].f_hz;
        // Expect within ~0.6 semitone for 24 bpo
        let cents = 1200.0 * ((f_peak / 440.0).log2().abs());
        assert!(
            cents < 60.0,
            "peak band off by {:.1} cents at f_peak={:.2} Hz",
            cents,
            f_peak
        );
    }

    #[test]
    fn window_len_monotonic_vs_freq() {
        let cfg = NsgtLog2Config::default();
        let nsgt = NsgtLog2::new(cfg);
        for w in nsgt.bands.windows(2) {
            if w[0].f_hz < w[1].f_hz {
                assert!(
                    w[0].win_len >= w[1].win_len,
                    "L_k should shrink as f increases"
                );
            }
        }
    }
}
