//! core/nsgt_kernel.rs — Log2-axis NSGT (kernel-based, one-FFT-per-frame)
//!
//! Overview
//! -----
//! - For each band k, we precompute a time-domain kernel h_k[n] = w_k[n] * e^{-j 2π f_k n / fs},
//!   zero-pad it to length Nfft, center it, and take FFT → K_k[ν] (frequency-domain kernel).
//! - For each frame, compute X[ν] = FFT{x_frame} only once, and obtain C_k = (1/Nfft) Σ_ν X[ν] * conj(K_k[ν]).
//! - K_k is **sparsified** by thresholding: we store only (index, weight) of nonzero bins for fast accumulation.
//! - Normalization: window energy U_k = √Σ w_k^2 is already included (divided by U_k). We do NOT divide by L here
//!   (handled on the test/PSD side).
//!
//! Design Notes
//! -----
//! - Frame length is fixed to Nfft. hop is a single global value based on overlap (shared across all bands).
//! - Use a periodic Hann window for each band with length L_k. Normalization by /U_k is already applied.
//! - Complex linear interpolation is unnecessary (the kernel itself represents the continuous frequency).

use crate::core::log2::Log2Space;
use rustfft::{FftPlanner, num_complex::Complex32};
use std::sync::Arc;

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

#[derive(Clone, Debug)]
struct KernelBand {
    f_hz: f32,
    log2_hz: f32,
    win_len: usize, // L_k
    // frequency-domain kernel (sparse): store (bin index, conj(K_k[bin])) for computation (multiplication-only)
    spec_conj_sparse: Vec<(usize, Complex32)>,
}

#[derive(Clone)]
pub struct NsgtKernelLog2 {
    cfg: NsgtLog2Config,
    space: Log2Space,
    nfft: usize,
    hop: usize,
    fft: Arc<dyn rustfft::Fft<f32>>,
    bands: Vec<KernelBand>,
}

impl NsgtKernelLog2 {
    /// Kernel-based NSGT
    ///
    /// - Nfft is set to the next power of two of max(L_k) (multiplied by `zpad_pow2`, default=2).
    /// - hop is based on Nfft (shared across all bands): hop = round((1-overlap)*Nfft)
    pub fn new(cfg: NsgtLog2Config, space: Log2Space) -> Self {
        assert!(
            cfg.overlap >= 0.0 && cfg.overlap < 0.95,
            "Overlap must be in [0,0.95)"
        );

        let fs = cfg.fs;
        let bpo = space.bins_per_oct as f32;
        let q = 1.0 / (2f32.powf(1.0 / bpo) - 1.0);

        // Calculate L_k for each band
        let mut Lks: Vec<usize> = space
            .centers_hz
            .iter()
            .map(|&f| ((q * fs / f).round() as usize).max(16)) // or ceil?
            .collect();

        // Nfft determination: next power of two based on max L_k (zero-padding to sharpen the kernel)
        let zpad_pow2: usize = 2;
        let max_L = *Lks.iter().max().unwrap_or(&1024);
        let mut nfft = (max_L * zpad_pow2).next_power_of_two().max(1024);

        // upper limit for practical use (to avoid excessively large FFT)
        if nfft > 1 << 18 {
            nfft = 1 << 18;
        }

        let hop = ((1.0 - cfg.overlap) * nfft as f32).round().max(1.0) as usize;

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(nfft);

        // Precompute kernels for each band (frequency domain, conjugated and sparsified)
        let mut bands: Vec<KernelBand> = Vec::with_capacity(space.centers_hz.len());
        for ((&f, &log2_f), &Lk) in space
            .centers_hz
            .iter()
            .zip(space.centers_log2.iter())
            .zip(Lks.iter())
        {
            let Lk = if Lk % 2 == 0 { Lk + 1 } else { Lk }; // make L_k odd for symmetric window
            let window = hann_periodic(Lk);
            //let U = (window.iter().map(|v| v * v).sum::<f32>()).sqrt();
            let sum_w = window.iter().copied().sum::<f32>().max(1e-12);

            // circularly shifted kernel h_k[n] = w_k[n] * e^{-j 2π f_k n / fs}, zero-padded to Nfft
            let mut h = vec![Complex32::new(0.0, 0.0); nfft];
            let center = Lk / 2;
            for i in 0..Lk {
                let w = window[i] / sum_w;
                let ph = 2.0 * std::f32::consts::PI * f * (i as f32) / fs;
                let cplx = Complex32::new(ph.cos(), -ph.sin()) * w;
                let idx = (i + nfft - center) % nfft;
                h[idx] = cplx;
            }

            // K_k = FFT{h_k}
            let mut H = h.clone();
            fft.process(&mut H);

            // 疎化（最大値に対して相対しきい値を掛ける）
            let mut max_mag = 0.0f32;
            for z in &H {
                let m = z.norm_sqr();
                if m > max_mag {
                    max_mag = m;
                }
            }
            let tol = (1e-5f32).max(1e-6 * max_mag.sqrt()); // 実用的な下限
            let mut sparse: Vec<(usize, Complex32)> = Vec::new();
            for (idx, &z) in H.iter().enumerate() {
                if z.norm() >= tol {
                    // 後の内積時に X * conj(K) をそのまま足せるよう conj を先にしておく
                    sparse.push((idx, z.conj()));
                }
            }

            bands.push(KernelBand {
                f_hz: f,
                log2_hz: log2_f,
                win_len: Lk,
                spec_conj_sparse: sparse,
            });
        }

        Self {
            cfg,
            space,
            nfft,
            hop,
            fft,
            bands,
        }
    }

    pub fn space(&self) -> &Log2Space {
        &self.space
    }
    pub fn nfft(&self) -> usize {
        self.nfft
    }
    pub fn hop(&self) -> usize {
        self.hop
    }
    pub fn bands(&self) -> &[KernelBand] {
        &self.bands
    }

    /// 解析：各フレームで X を一回だけ計算し、全バンドの C_k を周波数領域の疎内積で同時計算
    pub fn analyze(&self, x: &[f32]) -> Vec<BandCoeffs> {
        let fs = self.cfg.fs;
        if x.is_empty() {
            return Vec::new();
        }

        let nfft = self.nfft;
        let hop = self.hop;

        let n_frames = if x.len() < nfft {
            1
        } else {
            (x.len() - nfft) / hop + 1
        };

        // 出力スロット
        let mut out: Vec<BandCoeffs> = self
            .bands
            .iter()
            .map(|b| BandCoeffs {
                coeffs: Vec::with_capacity(n_frames),
                t_sec: Vec::with_capacity(n_frames),
                f_hz: b.f_hz,
                log2_hz: b.log2_hz,
                win_len: b.win_len,
                hop,
            })
            .collect();

        let mut buf = vec![Complex32::new(0.0, 0.0); nfft];

        for frame_idx in 0..n_frames {
            // フレーミング（矩形。カーネル側に窓が含まれるためここで追加窓は不要）
            let start = frame_idx * hop;
            for i in 0..nfft {
                let xi = if start + i < x.len() {
                    x[start + i]
                } else {
                    0.0
                };
                buf[i] = Complex32::new(xi, 0.0);
            }

            // X = FFT{x}
            self.fft.process(&mut buf);

            // 各バンドの疎内積： C_k = (1/Nfft) Σ X[ν] * conj(K_k[ν])
            for (b_idx, b) in self.bands.iter().enumerate() {
                let mut acc = Complex32::new(0.0, 0.0);
                for &(k, w) in &b.spec_conj_sparse {
                    acc += buf[k] * w;
                }
                acc = acc / (nfft as f32); // 1/Nfft スケーリング（rustfftの正規化に対応）
                out[b_idx].coeffs.push(acc);
                out[b_idx].t_sec.push((start + nfft / 2) as f32 / fs);
            }
        }

        out
    }

    /// 平均振幅（envelope的）。/L・per-Hz正規化はしない（利用側で定義）
    pub fn analyze_envelope(&self, x: &[f32]) -> Vec<f32> {
        let bands = self.analyze(x);
        bands
            .into_iter()
            .map(|b| {
                let m = b.coeffs.len();
                if m == 0 {
                    0.0
                } else {
                    // 端部1フレームずつ除外（エッジ効果の影響を消す）
                    let (start, end) = if m > 2 { (1, m - 1) } else { (0, m) };
                    let slice = &b.coeffs[start..end];
                    if slice.is_empty() {
                        0.0
                    } else {
                        slice.iter().map(|z| z.norm()).sum::<f32>() / (slice.len() as f32)
                    }
                }
            })
            .collect()
    }

    /// Hann窓用 ENBW 補正を含む Power Spectral Density [power/Hz]
    pub fn analyze_psd(&self, x: &[f32]) -> Vec<f32> {
        let fs = self.cfg.fs;
        self.analyze(x)
            .iter()
            .map(|b| {
                let mean_pow = b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>()
                    / (b.coeffs.len().max(1) as f32);
                let enbw_hz = 1.5_f32 * fs / (b.win_len as f32);
                mean_pow / enbw_hz.max(1e-12)
            })
            .collect()
    }
}

// ===== util =====

fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * (i as f32) / n as f32).cos()))
        .collect()
}

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
    fn pure_tone_near_target() {
        let fs = 48_000.0;
        let nsgt = NsgtKernelLog2::new(
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
            "peak off by {:.1} cents @ {:.2} Hz",
            cents,
            best_f
        );
    }

    #[test]
    fn multi_tone_localization() {
        let fs = 48_000.0;
        let nsgt = NsgtKernelLog2::new(NsgtLog2Config::default(), Log2Space::new(20.0, 8000.0, 96));
        for f in [55.0, 110.0, 220.0, 440.0, 880.0, 1760.0] {
            let sig = mk_sine(fs, f, 1.0);
            let bands = nsgt.analyze(&sig);
            let (mut best_f, mut best_val) = (0.0, 0.0);
            for b in &bands {
                let p = b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>()
                    / (b.coeffs.len().max(1) as f32);
                if p > best_val {
                    best_val = p;
                    best_f = b.f_hz;
                }
            }
            let cents = 1200.0 * ((best_f / f).log2()).abs();
            assert!(
                cents < 60.0,
                "freq localization failed: input {f} → peak {best_f} ({cents:.1} cents)"
            );
        }
    }

    #[test]
    fn amplitude_linearity() {
        let fs = 48_000.0;
        let nsgt = NsgtKernelLog2::new(NsgtLog2Config::default(), Log2Space::new(20.0, 8000.0, 96));
        let a = mk_sine(fs, 440.0, 1.0);
        let b: Vec<f32> = a.iter().map(|v| v * 2.0).collect();
        let e1: f32 = nsgt.analyze_psd(&a).iter().sum();
        let e2: f32 = nsgt.analyze_psd(&b).iter().sum();
        assert_relative_eq!(e2 / e1, 4.0, epsilon = 0.15, max_relative = 0.15);
    }
    #[test]
    fn window_len_monotonic_vs_freq() {
        // L_k は f が上がるほど単調非増加のはず
        let nsgt = NsgtKernelLog2::new(NsgtLog2Config::default(), Log2Space::new(20.0, 8000.0, 48));
        let lens: Vec<usize> = nsgt.bands().iter().map(|b| b.win_len).collect();
        for w in lens.windows(2) {
            assert!(
                w[0] >= w[1],
                "win_len should be non-increasing with frequency"
            );
        }
    }

    #[test]
    fn empty_signal_returns_empty() {
        let nsgt = NsgtKernelLog2::new(NsgtLog2Config::default(), Log2Space::new(20.0, 8000.0, 48));
        let out = nsgt.analyze(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn low_vs_high_freq_energy_scaling() {
        // 周波数によらず全体のエネルギーが概ね一定であることを確認
        let fs = 48_000.0;
        let space = Log2Space::new(20.0, 8000.0, 96);
        let nsgt = NsgtKernelLog2::new(NsgtLog2Config::default(), space.clone());

        let sig_low = mk_sine(fs, 220.0, 1.0);
        let sig_high = mk_sine(fs, 1760.0, 1.0);

        let integrate_psd = |x: &[f32]| -> f32 {
            let psd = nsgt.analyze_psd(x);
            let freqs = &space.centers_hz;
            let mut total = 0.0;
            for i in 0..freqs.len() - 1 {
                let df = (freqs[i + 1] - freqs[i]).max(1e-9);
                total += psd[i] * df;
            }
            total
        };

        let p_low = integrate_psd(&sig_low);
        let p_high = integrate_psd(&sig_high);
        let ratio = p_high / p_low;

        assert_relative_eq!(ratio, 1.0, epsilon = 0.3, max_relative = 0.3);
    }

    #[test]
    fn energy_conservation() {
        // 440 Hz の正弦波を 1 秒間生成
        let fs = 48_000.0;
        let f0 = 440.0;
        let sig = mk_sine(fs, f0, 1.0);

        // log2軸空間（楽音帯域）
        let space = Log2Space::new(35.0, 8000.0, 200);
        let nsgt = NsgtKernelLog2::new(NsgtLog2Config::default(), space.clone());

        // === 1. 入力信号の平均パワー ===
        // 正弦波の平均パワーは 0.5 に近い（sin²の平均＝1/2）
        let energy_signal: f32 = sig.iter().map(|x| x * x).sum::<f32>() / sig.len() as f32;

        // === 2. NSGT出力の per-Hz パワースペクトル密度 ===
        let psd = nsgt.analyze_psd(&sig);

        // === 3. Δf [Hz] で積分して総パワーを算出 ===
        let freqs = &space.centers_hz;
        let mut energy_coeffs = 0.0;
        for i in 0..freqs.len() - 1 {
            let df = (freqs[i + 1] - freqs[i]).max(1e-9);
            energy_coeffs += psd[i] * df;
        }

        // === 4. 比率を比較 ===
        let ratio = 4.0 * energy_coeffs / energy_signal; // 4x hann window scale correction

        eprintln!(
            "energy_coeffs={:.6}, energy_signal={:.6}, ratio={:.3}",
            energy_coeffs, energy_signal, ratio
        );

        // 理想的には ratio ≈ 1.0 （±10% 誤差許容）
        assert!(
            (0.9..=1.1).contains(&ratio),
            "Energy mismatch: ratio={ratio:.3}, coeffs={energy_coeffs:.6}, signal={energy_signal:.6}"
        );
    }

    #[test]
    fn hop_size_stability() {
        let fs = 48_000.0;
        let space = Log2Space::new(20.0, 8000.0, 96);
        let sig = mk_sine(fs, 440.0, 1.0);

        let e_mean = |overlap: f32| {
            let nsgt = NsgtKernelLog2::new(NsgtLog2Config { fs, overlap }, space.clone());
            let out = nsgt.analyze(&sig);
            let sum_e: f32 = out
                .iter()
                .map(|b| b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>())
                .sum();
            let n_frames: usize = out.iter().map(|b| b.coeffs.len()).max().unwrap_or(1);
            if n_frames > 0 {
                sum_e / n_frames as f32
            } else {
                0.0
            }
        };

        let e_half = e_mean(0.5);
        let e_75 = e_mean(0.75);
        assert_relative_eq!(e_half, e_75, epsilon = 0.3, max_relative = 0.3);
    }

    #[test]
    fn band_count_matches_space() {
        let space = Log2Space::new(20.0, 8000.0, 96);
        let nsgt = NsgtKernelLog2::new(NsgtLog2Config::default(), space.clone());
        assert_eq!(
            nsgt.bands().len(),
            space.centers_hz.len(),
            "Band count must match Log2Space bin count"
        );
    }

    #[test]
    fn noise_response_slope() {
        use rand::Rng;
        use scirs2_signal::waveforms::{brown_noise, pink_noise};

        let fs = 48_000.0;
        let secs = 6.0; // 長時間で平均化
        let n = (fs * secs) as usize;

        let nsgt = NsgtKernelLog2::new(
            NsgtLog2Config { fs, overlap: 0.5 },
            Log2Space::new(35.0, 8000.0, 200),
        );

        // --- ノイズ生成 ---
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

        // --- ENBW補正付き PSD (per-Hz 正規化) ---
        let psd_enbw = |bands: &[BandCoeffs]| -> Vec<f32> {
            bands
                .iter()
                .map(|b| {
                    let mean_pow = b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>()
                        / (b.coeffs.len().max(1) as f32);
                    let enbw_hz = 1.5_f32 * fs / (b.win_len as f32);
                    mean_pow / enbw_hz.max(1e-12)
                })
                .collect()
        };

        // --- per-Hz正規化済み PSD を取得 ---
        let w_psd = nsgt.analyze_psd(&white);
        let p_psd = nsgt.analyze_psd(&pink);
        let b_psd = nsgt.analyze_psd(&brown);

        // --- 傾き算出（dB/oct） ---
        let slope_db_per_oct = |psd: &[f32], space: &Log2Space| -> f32 {
            let logp: Vec<f32> = psd.iter().map(|v| 10.0 * v.max(1e-20).log10()).collect();
            let xs: &Vec<f32> = &space.centers_log2;
            let n = xs.len() as f32;
            let mx = xs.iter().sum::<f32>() / n;
            let my = logp.iter().sum::<f32>() / n;
            let num = xs
                .iter()
                .zip(&logp)
                .map(|(x, y)| (x - mx) * (y - my))
                .sum::<f32>();
            let den = xs.iter().map(|x| (x - mx).powi(2)).sum::<f32>();
            num / den
        };

        let s_white = slope_db_per_oct(&w_psd, nsgt.space());
        let s_pink = slope_db_per_oct(&p_psd, nsgt.space());
        let s_brown = slope_db_per_oct(&b_psd, nsgt.space());

        eprintln!("white slope(dB/oct) = {:.2}", s_white);
        eprintln!("pink  slope(dB/oct) = {:.2}", s_pink);
        eprintln!("brown slope(dB/oct) = {:.2}", s_brown);

        // --- 検証 ---
        assert!(s_white.abs() < 0.8, "white slope unexpected");
        assert!((s_pink + 3.0).abs() < 1.0, "pink slope unexpected");
        assert!((s_brown + 6.0).abs() < 1.5, "brown slope unexpected");
    }

    // ==============================
    // 可視化（プロット用）: cargo test -- --ignored
    // ==============================

    #[test]
    #[ignore]
    fn plot_nsgt_spectrum_kernel() {
        use plotters::prelude::*;

        let fs = 48_000.0;
        let nsgt = NsgtKernelLog2::new(
            NsgtLog2Config { fs, overlap: 0.5 },
            Log2Space::new(20.0, 8000.0, 96),
        );
        let sig = mk_sine(fs, 440.0, 1.0);
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
            BitMapBackend::new("target/nsgt_kernel_spectrum.png", (1500, 1000)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let y_max = points.iter().map(|(_, p)| *p).fold(0.0f32, f32::max) * 1.1;

        let mut chart = ChartBuilder::on(&root)
            .caption("NSGT Kernel Spectrum (pure 440Hz)", ("sans-serif", 18))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d((20f32.log2())..(8000f32.log2()), 0f32..y_max)
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
    fn plot_nsgt_log2_noise_response_kernel() {
        use plotters::prelude::*;
        use rand::Rng;
        use scirs2_signal::waveforms::{brown_noise, pink_noise};

        let fs = 48_000.0;
        let secs = 400.0;
        let n = (fs * secs) as usize;

        // 対象帯域を 35–8000Hz に制限（安定）
        let nsgt = NsgtKernelLog2::new(
            NsgtLog2Config { fs, overlap: 0.5 },
            Log2Space::new(35.0, 24_000.0, 200),
        );

        // --- ノイズ生成 ---
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

        // --- per-Hz正規化済み PSD を取得 ---
        let w_psd = nsgt.analyze_psd(&white);
        let p_psd = nsgt.analyze_psd(&pink);
        let b_psd = nsgt.analyze_psd(&brown);

        // dB換算
        let to_db =
            |v: &[f32]| -> Vec<f32> { v.iter().map(|x| 10.0 * x.max(1e-20).log10()).collect() };
        let w_db = to_db(&w_psd);
        let p_db = to_db(&p_psd);
        let b_db = to_db(&b_psd);

        let log2x = nsgt.space().centers_log2.clone();

        // --- 描画 ---
        let root = BitMapBackend::new("target/nsgt_kernel_noise_psd_db.png", (1500, 1000))
            .into_drawing_area();
        root.fill(&WHITE).unwrap();

        let y_min = w_db
            .iter()
            .chain(&p_db)
            .chain(&b_db)
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let y_max = w_db
            .iter()
            .chain(&p_db)
            .chain(&b_db)
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "NSGT Kernel PSD (White / Pink / Brown Noise)",
                ("sans-serif", 18),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(
                (35f32.log2())..(8_000f32.log2()),
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
                log2x.iter().cloned().zip(w_db.iter().cloned()),
                &BLUE,
            ))
            .unwrap()
            .label("White");

        chart
            .draw_series(LineSeries::new(
                log2x.iter().cloned().zip(p_db.iter().cloned()),
                &RED,
            ))
            .unwrap()
            .label("Pink");

        chart
            .draw_series(LineSeries::new(
                log2x.iter().cloned().zip(b_db.iter().cloned()),
                &GREEN,
            ))
            .unwrap()
            .label("Brown");

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()
            .unwrap();

        root.present().unwrap();
    }
}
