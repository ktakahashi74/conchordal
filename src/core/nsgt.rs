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

    /// Mean envelope magnitude per band (for potential-R roughness kernels).
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

    /// Representative analytic vector per band (for potential-C consonance kernels).
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

        let fs = self.cfg.fs;
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
            let bw_hz = self
                .space
                .delta_hz_at(b.f_hz)
                .unwrap_or(fs / (b.win_len as f32));

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

fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * (i as f32) / n as f32).cos()))
        .collect()
}

// /// Windowed Goertzel per band across time.
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
    use crate::core::log2::Log2Space;
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

    // #[test]
    // #[ignore]
    // fn plot_nsgt_spectrum() {
    //     use plotters::prelude::*;

    //     let fs = 48000.0;
    //     let nsgt = NsgtLog2::new(
    //         NsgtLog2Config { fs, overlap: 0.5 },
    //         Log2Space::new(20.0, 8000.0, 96),
    //     );

    //     // === 1. テスト信号（純音440Hz） ===
    //     let sig = mk_sine(fs, 440.0, 1.0);

    //     // === 2. NSGT解析 ===
    //     let bands = nsgt.analyze(&sig);

    //     // === 3. 各バンドの平均エネルギーをlog2軸で取得 ===
    //     let points: Vec<(f32, f32)> = bands
    //         .iter()
    //         .map(|b| {
    //             let p = b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>()
    //                 / (b.coeffs.len().max(1) as f32);
    //             (b.log2_hz, p)
    //         })
    //         .collect();

    //     // === 4. 出力先 ===
    //     let root = BitMapBackend::new("target/nsgt_spectrum.png", (1500, 1000)).into_drawing_area();
    //     root.fill(&WHITE).unwrap();
    //     let mut chart = ChartBuilder::on(&root)
    //         .caption("NSGT Spectrum (pure 440Hz)", ("sans-serif", 18))
    //         .margin(10)
    //         .x_label_area_size(40)
    //         .y_label_area_size(50)
    //         .build_cartesian_2d(
    //             (20f32.log2())..(8000f32.log2()),
    //             0f32..points.iter().map(|(_, p)| *p).fold(0.0f32, f32::max) * 1.1,
    //         )
    //         .unwrap();

    //     chart
    //         .configure_mesh()
    //         .x_desc("log2(frequency) [oct]")
    //         .y_desc("mean power")
    //         .x_label_formatter(&|v| format!("{:.0}", 2f32.powf(*v)))
    //         .draw()
    //         .unwrap();

    //     chart
    //         .draw_series(LineSeries::new(points.iter().cloned(), &BLUE))
    //         .unwrap()
    //         .label("power")
    //         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    //     chart
    //         .configure_series_labels()
    //         .border_style(&BLACK)
    //         .draw()
    //         .unwrap();
    //     root.present().unwrap();

    //     println!("Saved plot to target/nsgt_spectrum.png");
    // }

    // #[test]
    // #[ignore]
    // fn plot_nsgt_log2_noise_response() {
    //     use plotters::prelude::*;
    //     use rand::Rng;
    //     use scirs2_signal::waveforms::{brown_noise, pink_noise};

    //     let fs = 48_000.0;
    //     let secs = 4.0;
    //     let n = (fs * secs) as usize;

    //     // === 1. NSGT設定 ===
    //     let nsgt = NsgtLog2::new(
    //         NsgtLog2Config { fs, overlap: 0.5 },
    //         Log2Space::new(35.0, 24_000.0, 96),
    //     );

    //     // === 2. ノイズ生成 ===
    //     let mut rng = rand::rng();
    //     let white: Vec<f32> = (0..n).map(|_| rng.random_range(-1.0f32..1.0)).collect();
    //     let pink: Vec<f32> = pink_noise(n, Some(42))
    //         .unwrap()
    //         .iter()
    //         .map(|&v| v as f32)
    //         .collect();
    //     let brown: Vec<f32> = brown_noise(n, Some(42))
    //         .unwrap()
    //         .iter()
    //         .map(|&v| v as f32)
    //         .collect();

    //     // === 3. フル解析 ===
    //     let bands_w = nsgt.analyze(&white);
    //     let bands_p = nsgt.analyze(&pink);
    //     let bands_b = nsgt.analyze(&brown);

    //     // === 4. PSD正規化処理 (power per Hz) ===
    //     let psd_norm = |bands: &[BandCoeffs]| -> Vec<f32> {
    //         bands
    //             .iter()
    //             .map(|b| {
    //                 let mean_pow = b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>()
    //                     / (b.coeffs.len().max(1) as f32);
    //                 let bw = nsgt
    //                     .space()
    //                     .delta_hz_at(b.f_hz)
    //                     .unwrap_or(fs / (b.win_len as f32));
    //                 mean_pow / bw.max(1e-9)
    //             })
    //             .collect()
    //     };

    //     let psd_w = psd_norm(&bands_w);
    //     let psd_p = psd_norm(&bands_p);
    //     let psd_b = psd_norm(&bands_b);

    //     // === 5. 対数化 [dB re power/Hz] ===
    //     let to_db =
    //         |x: &[f32]| -> Vec<f32> { x.iter().map(|v| 10.0 * v.max(1e-20).log10()).collect() };
    //     let white_db = to_db(&psd_w);
    //     let pink_db = to_db(&psd_p);
    //     let brown_db = to_db(&psd_b);
    //     let log2x = nsgt.space().centers_log2.clone();

    //     // === 6. 出力 ===
    //     let root =
    //         BitMapBackend::new("target/nsgt_noise_psd_db.png", (1500, 1000)).into_drawing_area();
    //     root.fill(&WHITE).unwrap();

    //     let y_min = white_db
    //         .iter()
    //         .chain(&pink_db)
    //         .chain(&brown_db)
    //         .cloned()
    //         .fold(f32::INFINITY, f32::min);
    //     let y_max = white_db
    //         .iter()
    //         .chain(&pink_db)
    //         .chain(&brown_db)
    //         .cloned()
    //         .fold(f32::NEG_INFINITY, f32::max);

    //     let mut chart = ChartBuilder::on(&root)
    //         .caption("NSGT PSD (White / Pink / Brown Noise)", ("sans-serif", 18))
    //         .margin(10)
    //         .x_label_area_size(40)
    //         .y_label_area_size(60)
    //         .build_cartesian_2d(
    //             (20f32.log2())..(24_000f32.log2()),
    //             (y_min - 10.0)..(y_max + 10.0),
    //         )
    //         .unwrap();

    //     chart
    //         .configure_mesh()
    //         .x_desc("log2(frequency) [oct]")
    //         .y_desc("Power Spectral Density [dB re 1/Hz]")
    //         .x_label_formatter(&|v| format!("{:.0}", 2f32.powf(*v)))
    //         .draw()
    //         .unwrap();

    //     // === 7. プロット ===
    //     chart
    //         .draw_series(LineSeries::new(
    //             log2x.iter().cloned().zip(white_db.iter().cloned()),
    //             &BLUE,
    //         ))
    //         .unwrap()
    //         .label("White")
    //         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    //     chart
    //         .draw_series(LineSeries::new(
    //             log2x.iter().cloned().zip(pink_db.iter().cloned()),
    //             &RED,
    //         ))
    //         .unwrap()
    //         .label("Pink")
    //         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    //     chart
    //         .draw_series(LineSeries::new(
    //             log2x.iter().cloned().zip(brown_db.iter().cloned()),
    //             &GREEN,
    //         ))
    //         .unwrap()
    //         .label("Brown")
    //         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    //     chart
    //         .configure_series_labels()
    //         .border_style(&BLACK)
    //         .background_style(&WHITE.mix(0.8))
    //         .draw()
    //         .unwrap();

    //     root.present().unwrap();
    //     println!("Saved plot to target/nsgt_noise_psd_db.png");
    // }
}
