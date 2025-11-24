//! core/nsgt_fft.rs — Log2-axis NSGT (FFT-batched analysis version)
//!
//! Faster constant-Q analysis using FFT batches grouped by window length.
//! Compatible with `NsgtLog2` output format (for downstream roughness/consonance).

use crate::core::log2space::Log2Space;
use crate::core::nsgt::{BandCoeffs, NsgtLog2Config};
use rustfft::{FftPlanner, num_complex::Complex32};
use std::sync::Arc;

// =====================================================
// Internal structs
// =====================================================

#[derive(Clone, Debug)]
struct BandInfo {
    f_hz: f32,
    log2_hz: f32,
    bin: usize,
}

#[derive(Clone)]
struct BandGroup {
    win_len: usize,
    hop: usize,
    window: Vec<f32>,
    bands: Vec<BandInfo>,
    fft: Arc<dyn rustfft::Fft<f32>>,
}

// =====================================================
// Main struct
// =====================================================

#[derive(Clone)]
pub struct NsgtFftLog2 {
    cfg: NsgtLog2Config,
    space: Log2Space,
    groups: Vec<BandGroup>,
}

// =====================================================
// Implementation
// =====================================================

impl NsgtFftLog2 {
    /// Construct FFT-batched NSGT analyzer grouped by octave.
    /// Each group's window length L is decided at the group's f_min,
    /// so that f* = Q fs / L <= f_min and the inflection never falls inside.
    pub fn new(cfg: NsgtLog2Config, space: Log2Space) -> Self {
        let fs = cfg.fs;
        let bpo = space.bins_per_oct as f32;
        let q = 1.0 / (2f32.powf(1.0 / bpo) - 1.0);

        // ← 必要なら false にして非2冪L（さらに滑らか、やや遅い）
        let use_pow2 = true;

        use std::collections::BTreeMap;
        let mut by_oct: BTreeMap<i32, Vec<(f32, f32)>> = BTreeMap::new();
        for (&f, &log2f) in space.centers_hz.iter().zip(space.centers_log2.iter()) {
            by_oct
                .entry(log2f.floor() as i32)
                .or_default()
                .push((f, log2f));
        }

        let mut groups = Vec::new();
        for (_oct, mut bins) in by_oct {
            bins.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let f_min = bins.first().unwrap().0;

            // 決定式： L_g = ceil(Q fs / f_min)
            let mut Lg = (q * fs / f_min).ceil() as usize;
            Lg = Lg.max(32);
            if use_pow2 {
                Lg = Lg.next_power_of_two();
            }

            let hop = ((1.0 - cfg.overlap) * Lg as f32).round().max(1.0) as usize;
            let window = hann_periodic(Lg);

            let mut planner = FftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(Lg);

            let bands: Vec<_> = bins
                .into_iter()
                .map(|(f, log2f)| {
                    let bin = (f * Lg as f32 / fs).round() as usize;
                    BandInfo {
                        f_hz: f,
                        log2_hz: log2f,
                        bin,
                    }
                })
                .collect();

            groups.push(BandGroup {
                win_len: Lg,
                hop,
                window,
                bands,
                fft,
            });
        }

        Self { cfg, space, groups }
    }

    //impl NsgtFftLog2 {
    // pub fn new(cfg: NsgtLog2Config, space: Log2Space) -> Self {
    //     let fs = cfg.fs;
    //     let bpo = space.bins_per_oct as f32;
    //     let q = 1.0 / (2f32.powf(1.0 / bpo) - 1.0);

    //     let mut bands_all: Vec<(f32, f32, usize)> = space
    //         .centers_hz
    //         .iter()
    //         .zip(&space.centers_log2)
    //         .map(|(&f, &log2_f)| {
    //             let L = (q * fs / f).ceil() as usize;
    //             let L_pow2 = L.next_power_of_two().max(32);
    //             (f, log2_f, L_pow2)
    //         })
    //         .collect();

    //     use itertools::Itertools;
    //     let mut groups = Vec::new();
    //     for (L_pow2, subset) in &bands_all.into_iter().group_by(|(_, _, L)| *L) {
    //         let hop = ((1.0 - cfg.overlap) * L_pow2 as f32).round().max(1.0) as usize;
    //         let window = hann_periodic(L_pow2);
    //         let mut planner = FftPlanner::<f32>::new();
    //         let fft = planner.plan_fft_forward(L_pow2);
    //         let bands: Vec<_> = subset
    //             .map(|(f, log2_f, _)| {
    //                 let bin = (f * L_pow2 as f32 / fs).round() as usize;
    //                 BandInfo {
    //                     f_hz: f,
    //                     log2_hz: log2_f,
    //                     bin,
    //                 }
    //             })
    //             .collect();
    //         groups.push(BandGroup {
    //             win_len: L_pow2,
    //             hop,
    //             window,
    //             bands,
    //             fft,
    //         });
    //     }

    //     Self { cfg, space, groups }
    // }

    // pub fn new(cfg: NsgtLog2Config, space: Log2Space) -> Self {
    //     let fs = cfg.fs;
    //     let bpo = space.bins_per_oct as f32;
    //     let q = 1.0 / (2f32.powf(1.0 / bpo) - 1.0);

    //     let mut bands_all: Vec<(f32, f32, usize)> = space
    //         .centers_hz
    //         .iter()
    //         .zip(&space.centers_log2)
    //         .map(|(&f, &log2_f)| {
    //             let L = (q * fs / f).ceil() as usize;
    //             let L_pow2 = L.next_power_of_two().max(32);
    //             (f, log2_f, L_pow2)
    //         })
    //         .collect();

    //     use itertools::Itertools;
    //     let mut groups = Vec::new();
    //     for (L_pow2, subset) in &bands_all.into_iter().group_by(|(_, _, L)| *L) {
    //         let hop = ((1.0 - cfg.overlap) * L_pow2 as f32).round().max(1.0) as usize;
    //         let window = hann_periodic(L_pow2);
    //         let mut planner = FftPlanner::<f32>::new();
    //         let fft = planner.plan_fft_forward(L_pow2);
    //         let bands: Vec<_> = subset
    //             .map(|(f, log2_f, _)| {
    //                 let bin = (f * L_pow2 as f32 / fs).round() as usize;
    //                 BandInfo {
    //                     f_hz: f,
    //                     log2_hz: log2_f,
    //                     bin,
    //                 }
    //             })
    //             .collect();
    //         groups.push(BandGroup {
    //             win_len: L_pow2,
    //             hop,
    //             window,
    //             bands,
    //             fft,
    //         });
    //     }

    //     Self { cfg, space, groups }
    // }

    pub fn analyze(&self, x: &[f32]) -> Vec<BandCoeffs> {
        let fs = self.cfg.fs;
        let mut results = Vec::new();

        for g in &self.groups {
            let L = g.win_len;
            let hop = g.hop;
            if x.len() < L {
                continue;
            }

            let n_frames = (x.len() - L) / hop + 1;

            let U = (g.window.iter().map(|v| v * v).sum::<f32>()).sqrt();

            let mut buf = vec![Complex32::default(); L];
            let mut fft_out = vec![Complex32::default(); L];

            let mut group_results: Vec<BandCoeffs> = g
                .bands
                .iter()
                .map(|b| BandCoeffs {
                    coeffs: Vec::with_capacity(n_frames),
                    t_sec: Vec::with_capacity(n_frames),
                    f_hz: b.f_hz,
                    log2_hz: b.log2_hz,
                    win_len: L,
                    hop,
                })
                .collect();

            for frame_idx in 0..n_frames {
                let start = frame_idx * hop;
                for i in 0..L {
                    let xi = if start + i < x.len() {
                        x[start + i]
                    } else {
                        0.0
                    };
                    buf[i] = Complex32::new(xi * g.window[i] / U, 0.0);
                }

                g.fft.process(&mut buf);
                fft_out.copy_from_slice(&buf);

                for (b_idx, band) in g.bands.iter().enumerate() {
                    let bin = band.bin.min(L - 1);
                    let val = fft_out[bin];
                    group_results[b_idx].coeffs.push(val);
                    group_results[b_idx].t_sec.push((start + L / 2) as f32 / fs);
                }
            }

            results.extend(group_results);
        }

        results
    }

    //     pub fn analyze(&self, x: &[f32]) -> Vec<BandCoeffs> {
    //     let fs = self.cfg.fs;
    //     let mut results = Vec::new();

    //     for g in &self.groups {
    //         let L = g.win_len;
    //         let hop = g.hop;
    //         let U = (g.window.iter().map(|v| v * v).sum::<f32>()).sqrt();

    //         if x.len() < L {
    //             continue;
    //         }
    //         let n_frames = (x.len() - L) / hop + 1;
    //         let mut buf = vec![Complex32::default(); L];
    //         let mut fft_out = vec![Complex32::default(); L];

    //         let mut group_results: Vec<BandCoeffs> = g
    //             .bands
    //             .iter()
    //             .map(|b| BandCoeffs {
    //                 coeffs: Vec::with_capacity(n_frames),
    //                 t_sec: Vec::with_capacity(n_frames),
    //                 f_hz: b.f_hz,
    //                 log2_hz: b.log2_hz,
    //                 win_len: L,
    //                 hop,
    //             })
    //             .collect();

    //         for frame_idx in 0..n_frames {
    //             let start = frame_idx * hop;
    //             for i in 0..L {
    //                 let xi = if start + i < x.len() {
    //                     x[start + i]
    //                 } else {
    //                     0.0
    //                 };
    //                 buf[i] = Complex32::new(xi * g.window[i] / U, 0.0);
    //             }

    //             g.fft.process(&mut buf);
    //             fft_out.copy_from_slice(&buf);

    //             for (b_idx, band) in g.bands.iter().enumerate() {
    //                 let bin = band.bin.min(L - 1);
    //                 let val = fft_out[bin];
    //                 group_results[b_idx].coeffs.push(val);
    //                 group_results[b_idx].t_sec.push((start + L / 2) as f32 / fs);
    //             }
    //         }

    //         results.extend(group_results);
    //     }

    //     results
    // }

    pub fn analyze_envelope(&self, signal: &[f32]) -> Vec<f32> {
        let bands = self.analyze(signal);
        bands
            .iter()
            .map(|b| {
                if b.coeffs.is_empty() {
                    0.0
                } else {
                    b.coeffs.iter().map(|z| z.norm()).sum::<f32>() / b.coeffs.len() as f32
                }
            })
            .collect()
    }

    pub fn analyze_psd(&self, signal: &[f32]) -> Vec<f32> {
        let bands = self.analyze(signal);
        let fs = self.cfg.fs;
        bands
            .iter()
            .map(|b| {
                if b.coeffs.is_empty() {
                    0.0
                } else {
                    let mean_pow =
                        b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>() / b.coeffs.len() as f32;
                    let bw = b.f_hz * (2f32.powf(1.0 / self.space.bins_per_oct as f32) - 1.0);
                    mean_pow / bw.max(1e-9)
                }
            })
            .collect()
    }

    pub fn space(&self) -> &Log2Space {
        &self.space
    }
}

fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * (i as f32) / n as f32).cos()))
        .collect()
}

// =====================================================
// Tests (移植版)
// =====================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use plotters::prelude::*;
    use rand::Rng;
    use scirs2_signal::waveforms::{brown_noise, pink_noise};

    fn mk_sine(fs: f32, f: f32, secs: f32) -> Vec<f32> {
        let n = (fs * secs).round() as usize;
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f * (i as f32) / fs).sin())
            .collect()
    }

    #[test]
    fn pure_tone_hits_right_band() {
        let fs = 48_000.0;
        let nsgt = NsgtFftLog2::new(
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
        let cents = 1200.0 * (((best_f / 440.0_f32).log2()).abs());
        assert!(
            cents < 60.0,
            "440Hz peak band off by {:.1} cents (f={:.2})",
            cents,
            best_f
        );
    }

    #[test]
    fn window_len_monotonic_vs_freq() {
        let nsgt = NsgtFftLog2::new(NsgtLog2Config::default(), Log2Space::new(20.0, 8000.0, 48));
        let mut winlens = nsgt
            .groups
            .iter()
            .flat_map(|g| g.bands.iter().map(move |_| g.win_len))
            .collect::<Vec<_>>();
        for w in winlens.windows(2) {
            assert!(w[0] >= w[1]);
        }
    }

    #[test]
    fn empty_signal_returns_empty() {
        let nsgt = NsgtFftLog2::new(NsgtLog2Config::default(), Log2Space::new(20.0, 8000.0, 48));
        let out = nsgt.analyze(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn low_vs_high_freq_energy_scaling() {
        let fs = 48000.0;
        let nsgt = NsgtFftLog2::new(NsgtLog2Config::default(), Log2Space::new(20.0, 8000.0, 96));
        let sig_low = mk_sine(fs, 220.0, 10.0);
        let sig_high = mk_sine(fs, 1760.0, 10.0);
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
        let nsgt = NsgtFftLog2::new(NsgtLog2Config::default(), Log2Space::new(20.0, 8000.0, 96));
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
        let nsgt = NsgtFftLog2::new(NsgtLog2Config::default(), Log2Space::new(20.0, 8000.0, 96));
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
            let nsgt = NsgtFftLog2::new(NsgtLog2Config { fs, overlap }, space.clone());
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
        let nsgt = NsgtFftLog2::new(NsgtLog2Config::default(), space.clone());
        let count_space = space.centers_hz.len();
        let count_total: usize = nsgt.groups.iter().map(|g| g.bands.len()).sum();
        assert_eq!(count_space, count_total);
    }

    #[test]
    #[ignore]
    fn plot_nsgt_spectrum() {
        let fs = 48000.0;
        let nsgt = NsgtFftLog2::new(
            NsgtLog2Config { fs, overlap: 0.5 },
            Log2Space::new(20.0, 8000.0, 96),
        );
        let sig = mk_sine(fs, 440.0, 10.0);
        let bands = nsgt.analyze(&sig);
        let points: Vec<(f32, f32)> = bands
            .iter()
            .map(|b| {
                let p = b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>()
                    / (b.coeffs.len().max(1) as f32);
                (b.log2_hz, p)
            })
            .collect();
        let root =
            BitMapBackend::new("target/nsgt_fft_spectrum.png", (1500, 1000)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption("FFT NSGT Spectrum (pure 440Hz)", ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(
                (20f32.log2())..(8000f32.log2()),
                0f32..points.iter().map(|(_, p)| *p).fold(0.0f32, f32::max) * 1.1,
            )
            .unwrap();
        chart
            .configure_mesh()
            .x_desc("log2(frequency) [oct]")
            .y_desc("mean power")
            .x_label_formatter(&|v| format!("{:.0}", 2f32.powf(*v)))
            .draw()
            .unwrap();
        chart
            .draw_series(LineSeries::new(points.iter().cloned(), &BLUE))
            .unwrap();
        root.present().unwrap();
    }

    #[test]
    #[ignore]
    fn plot_nsgt_log2_noise_response() {
        let fs = 48_000.0;
        let secs = 10.0;
        let n = (fs * secs) as usize;
        let nsgt = NsgtFftLog2::new(
            NsgtLog2Config { fs, overlap: 0.5 },
            Log2Space::new(35.0, 24_000.0, 50),
        );
        let mut rng = rand::rng();
        let white: Vec<f32> = (0..n).map(|_| rng.random_range(-1.0f32..1.0)).collect();
        let pink: Vec<f32> = pink_noise(n, Some(42))
            .unwrap()
            .iter()
            .map(|&v| v as f32)
            .collect();
        let brown: Vec<f32> = brown_noise(n, Some(42))
            .unwrap()
            .iter()
            .map(|&v| v as f32)
            .collect();
        let psd_norm = |bands: &[BandCoeffs]| -> Vec<f32> {
            bands
                .iter()
                .map(|b| {
                    let mean_pow = b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>()
                        / (b.coeffs.len().max(1) as f32);
                    let bw = nsgt
                        .space()
                        .delta_hz_at(b.f_hz)
                        .unwrap_or(fs / (b.win_len as f32));
                    mean_pow / bw.max(1e-9)
                })
                .collect()
        };

        let bands_w = nsgt.analyze(&white);
        let bands_p = nsgt.analyze(&pink);
        let bands_b = nsgt.analyze(&brown);
        let to_db =
            |x: &[f32]| -> Vec<f32> { x.iter().map(|v| 10.0 * v.max(1e-20).log10()).collect() };
        let white_db = to_db(&psd_norm(&bands_w));
        let pink_db = to_db(&psd_norm(&bands_p));
        let brown_db = to_db(&psd_norm(&bands_b));
        let log2x = nsgt.space().centers_log2.clone();
        let root = BitMapBackend::new("target/nsgt_fft_noise_psd_db.png", (1500, 1000))
            .into_drawing_area();
        root.fill(&WHITE).unwrap();
        let y_min = white_db
            .iter()
            .chain(&pink_db)
            .chain(&brown_db)
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let y_max = white_db
            .iter()
            .chain(&pink_db)
            .chain(&brown_db)
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut chart = ChartBuilder::on(&root)
            .caption(
                "FFT NSGT PSD (White / Pink / Brown Noise)",
                ("sans-serif", 18),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(
                (20f32.log2())..(24_000f32.log2()),
                (y_min - 10.0)..(y_max + 10.0),
            )
            .unwrap();
        chart
            .configure_mesh()
            .x_desc("log2(frequency) [oct]")
            .y_desc("Power Spectral Density [dB re 1/Hz]")
            .x_label_formatter(&|v| format!("{:.0}", 2f32.powf(*v)))
            .draw()
            .unwrap();
        chart
            .draw_series(LineSeries::new(
                log2x.iter().cloned().zip(white_db.iter().cloned()),
                &BLUE,
            ))
            .unwrap()
            .label("White");
        chart
            .draw_series(LineSeries::new(
                log2x.iter().cloned().zip(pink_db.iter().cloned()),
                &RED,
            ))
            .unwrap()
            .label("Pink");
        chart
            .draw_series(LineSeries::new(
                log2x.iter().cloned().zip(brown_db.iter().cloned()),
                &GREEN,
            ))
            .unwrap()
            .label("Brown");
        root.present().unwrap();
    }
}
