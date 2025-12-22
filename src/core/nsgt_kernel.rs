//! core/nsgt_kernel.rs — Log2-axis NSGT (kernel-based, one-FFT-per-frame)
//!
//! Overview
//! -----
//! - For each band k, we precompute a time-domain kernel h_k[n] = w_k[n] * exp(+j 2*pi f_k n / fs),
//!   zero-pad it to length Nfft, center it, and take FFT -> K_k[nu] (frequency-domain kernel).
//! - For each frame, compute X[nu] = FFT{x_frame} only once, and obtain
//!   C_k = (1/Nfft) * sum_nu X[nu] * conj(K_k[nu]) = sum_n x[n] * conj(h_k[n]),
//!   so conj(h_k[n]) contains the standard analysis factor exp(-j omega n).
//! - K_k is **sparsified** by thresholding: we store only (index, weight) of nonzero bins for fast accumulation.
//!
//! Design Notes
//! -----
//! - Frame length is fixed to Nfft. hop is a single global value based on overlap (shared across all bands).
//! - Use a periodic Hann window for each band with length L_k. Normalization by /U_k is already applied.
//! - Complex linear interpolation is unnecessary (the kernel itself represents the continuous frequency).

use crate::core::log2space::Log2Space;
use rustfft::{FftPlanner, num_complex::Complex32};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// =====================================================
// Config structures
// =====================================================
const MIN_WIN_LEN: usize = 16;

/// NSGT configuration (log2-axis, analysis-only).
#[derive(Clone, Copy, Debug)]
pub struct NsgtLog2Config {
    /// Sampling rate [Hz]
    pub fs: f32,
    /// Overlap ratio in [0, 0.95). 0.5 = 50% overlap (default-good)
    pub overlap: f32,
    pub nfft_override: Option<usize>,
    /// Kernel time alignment within the FFT frame.
    pub kernel_align: KernelAlign,
}

impl Default for NsgtLog2Config {
    fn default() -> Self {
        Self {
            fs: 48_000.0,
            overlap: 0.5,
            nfft_override: None,
            kernel_align: KernelAlign::Center,
        }
    }
}

/// Time placement of kernels within the FFT frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelAlign {
    Center,
    Right,
}

impl Default for KernelAlign {
    fn default() -> Self {
        Self::Center
    }
}

/// Frequency-dependent cap on analysis window length.
#[derive(Clone, Copy, Debug)]
pub struct WinLenCap {
    /// Apply cap at and above this frequency [Hz].
    pub pivot_hz: f32,
    /// Maximum window length above pivot [s].
    pub max_len_above_s: f32,
}

impl WinLenCap {
    /// Cap length in samples (rounded), clamped to < nfft and normalized.
    pub fn cap_samples(&self, fs: f32, nfft: usize) -> usize {
        let raw = (self.max_len_above_s * fs).round().max(1.0) as usize;
        finalize_win_len(raw, nfft)
    }
}

fn finalize_win_len(mut len: usize, nfft: usize) -> usize {
    let nfft_cap = nfft.saturating_sub(1).max(1);
    if len < MIN_WIN_LEN {
        len = MIN_WIN_LEN;
    }
    if len > nfft_cap {
        len = nfft_cap;
    }
    if len % 2 == 0 {
        len = len.saturating_sub(1);
    }
    if len < 3 {
        len = 3.min(nfft_cap);
    }
    len
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
pub struct KernelBand {
    pub f_hz: f32,
    pub log2_hz: f32,
    pub win_len: usize, // L_k
    // frequency-domain kernel (sparse): store (bin index, conj(K_k[bin])) for computation (multiplication-only)
    pub spec_conj_sparse: Vec<(usize, Complex32)>,
}

#[derive(Clone)]
pub struct NsgtKernelLog2 {
    pub cfg: NsgtLog2Config,
    pub space: Log2Space,
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
    /// - win_len_cap limits high-frequency window length without clamping Q, preserving
    ///   time resolution when bins_per_oct is increased.
    pub fn new(cfg: NsgtLog2Config, space: Log2Space, win_len_cap: Option<WinLenCap>) -> Self {
        assert!(
            cfg.overlap >= 0.0 && cfg.overlap < 0.99,
            "Overlap must be in [0,0.99)"
        );

        let fs = cfg.fs;
        let bpo = space.bins_per_oct as f32;
        let raw_q = 1.0 / (2f32.powf(1.0 / bpo) - 1.0);
        let q_min = 5.0;
        let q_max = 80.0;
        // With a frequency-dependent window cap, do not clamp q_max. This allows
        // higher Q (better peak separation) without increasing high-frequency window length.
        let q = if win_len_cap.is_some() {
            raw_q.max(q_min)
        } else {
            raw_q.clamp(q_min, q_max)
        };

        // Calculate L_k for each band. With win_len_cap, high-frequency windows are capped
        // to preserve time resolution while allowing higher Q via bins_per_oct.
        // With fixed nfft_override, low bands may still hit nfft-1.
        let win_lengths: Vec<usize> = space
            .centers_hz
            .iter()
            .map(|&f| {
                let mut len = (q * fs / f).round().max(1.0) as usize;
                if let Some(cap) = win_len_cap {
                    if f >= cap.pivot_hz {
                        let cap_raw = (cap.max_len_above_s * fs).round().max(1.0) as usize;
                        len = len.min(cap_raw);
                    }
                }
                len
            })
            .collect();

        // Nfft determination: next power of two based on max L_k (zero-padding to sharpen the kernel)
        let zpad_pow2: usize = 2;
        let max_win_len = *win_lengths.iter().max().unwrap_or(&1024);

        let mut nfft = if let Some(n) = cfg.nfft_override {
            n
        } else {
            (max_win_len * zpad_pow2).next_power_of_two().max(1024)
        };

        // upper limit for practical use (to avoid excessively large FFT)
        if nfft > 1 << 18 {
            nfft = 1 << 18;
        }

        let hop = ((1.0 - cfg.overlap) * nfft as f32).round().max(1.0) as usize;

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(nfft);

        // Precompute kernels for each band (frequency domain, conjugated and sparsified)
        let mut bands: Vec<KernelBand> = Vec::with_capacity(space.centers_hz.len());
        for ((&f, &log2_f), &win_len) in space
            .centers_hz
            .iter()
            .zip(space.centers_log2.iter())
            .zip(win_lengths.iter())
        {
            // Apply constraints: cap (if any), nfft limit, odd length, and minimum length.
            // With win_len_cap, the cap dominates high frequencies to preserve time resolution
            // even when bins_per_oct (Q) is increased.
            let win_len_req = finalize_win_len(win_len, nfft);

            let window = hann_periodic(win_len_req);
            let sum_w = window.iter().copied().sum::<f32>().max(1e-12);

            // Circularly shifted kernel h_k[n] = w[n]*exp(+j 2*pi f n/fs) (zero-padded to nfft).
            let mut h = vec![Complex32::new(0.0, 0.0); nfft];
            let center = win_len_req / 2;
            let shift = match cfg.kernel_align {
                KernelAlign::Center => (nfft / 2 + nfft - center) % nfft,
                KernelAlign::Right => nfft - win_len_req,
            };
            for (i, &win) in window.iter().enumerate().take(win_len_req) {
                let w = win / sum_w;
                let ph = 2.0 * std::f32::consts::PI * f * (i as f32) / fs;
                let cplx = Complex32::new(ph.cos(), ph.sin()) * w;
                let idx = (i + shift) % nfft;
                h[idx] = cplx;
            }

            // K_k = FFT{h_k}
            let mut kernel_freq = h.clone();
            fft.process(&mut kernel_freq);

            // Sparsify with a relative magnitude threshold.
            let mut max_mag = 0.0f32;
            for z in &kernel_freq {
                let m = z.norm_sqr();
                if m > max_mag {
                    max_mag = m;
                }
            }
            let tol = (1e-6 * max_mag.sqrt()).max(1e-20);
            let mut sparse: Vec<(usize, Complex32)> = Vec::new();
            for (k, &z) in kernel_freq.iter().enumerate() {
                if z.norm() >= tol {
                    // Keep time-domain alignment (center/right); do not undo the shift here.
                    sparse.push((k, z.conj()));
                }
            }

            bands.push(KernelBand {
                f_hz: f,
                log2_hz: log2_f,
                win_len: win_len_req,
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
    #[inline]
    pub fn time_ref_sample_in_frame(&self) -> usize {
        match self.cfg.kernel_align {
            KernelAlign::Center => self.nfft / 2,
            KernelAlign::Right => self.nfft.saturating_sub(1),
        }
    }

    /// Analysis: compute X once per frame and accumulate sparse inner products for all bands.
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

        // Output slots.
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
            // Rectangular framing; the kernel already contains the analysis window.
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

            // Sparse inner product per band: C_k = (1/Nfft) * sum X[nu] * conj(K_k[nu]) = sum x[n] * conj(h_k[n]).
            for (b_idx, b) in self.bands.iter().enumerate() {
                let mut acc = Complex32::new(0.0, 0.0);
                for &(k, w) in &b.spec_conj_sparse {
                    acc += buf[k] * w;
                }
                acc /= nfft as f32; // 1/Nfft scaling to match rustfft normalization.
                out[b_idx].coeffs.push(acc);
                let t_ref = self.time_ref_sample_in_frame();
                out[b_idx].t_sec.push((start + t_ref) as f32 / fs);
            }
        }

        out
    }

    /// Mean magnitude (envelope). No per-L or per-Hz normalization; caller defines it.
    pub fn analyze_envelope(&self, x: &[f32]) -> Vec<f32> {
        let bands = self.analyze(x);
        bands
            .into_iter()
            .map(|b| {
                let m = b.coeffs.len();
                if m == 0 {
                    0.0
                } else {
                    // Drop one frame at each edge to reduce boundary effects.
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

    /// Power spectral density [power/Hz] with ENBW correction for Hann window (**one-sided**).
    pub fn analyze_psd(&self, x: &[f32]) -> Vec<f32> {
        let fs = self.cfg.fs;
        self.analyze(x)
            .iter()
            .map(|b| {
                let mean_pow = b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>()
                    / (b.coeffs.len().max(1) as f32);
                let enbw_hz = 1.5_f32 * fs / (b.win_len as f32);
                // two-sided -> one-sided
                2.0 * (mean_pow / enbw_hz.max(1e-12))
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
    use crate::core::utils::{brown_noise, pink_noise, white_noise};
    use approx::assert_relative_eq;

    fn mk_sine(fs: f32, f: f32, secs: f32) -> Vec<f32> {
        let n = (fs * secs).round() as usize;
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f * (i as f32) / fs).sin())
            .collect()
    }

    fn mk_sine_len(fs: f32, f: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f * (i as f32) / fs).sin())
            .collect()
    }

    fn wrap_to_pi(mut x: f32) -> f32 {
        use core::f32::consts::PI;
        let two_pi = 2.0 * PI;
        while x <= -PI {
            x += two_pi;
        }
        while x > PI {
            x -= two_pi;
        }
        x
    }

    #[test]
    fn pure_tone_near_target() {
        let fs = 48_000.0;
        let nsgt = NsgtKernelLog2::new(
            NsgtLog2Config {
                fs,
                overlap: 0.5,
                ..Default::default()
            },
            Log2Space::new(20.0, 8000.0, 200),
            None,
        );
        let sig = mk_sine_len(fs, 440.0, nsgt.nfft());
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
        let nsgt = NsgtKernelLog2::new(
            NsgtLog2Config::default(),
            Log2Space::new(20.0, 8000.0, 96),
            None,
        );
        for f in [55.0, 110.0, 220.0, 440.0, 880.0, 1760.0] {
            let sig = mk_sine_len(fs, f, nsgt.nfft());
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
        let nsgt = NsgtKernelLog2::new(
            NsgtLog2Config::default(),
            Log2Space::new(20.0, 8000.0, 96),
            None,
        );
        let a = mk_sine(fs, 440.0, 1.0);
        let b: Vec<f32> = a.iter().map(|v| v * 2.0).collect();
        let e1: f32 = nsgt.analyze_psd(&a).iter().sum();
        let e2: f32 = nsgt.analyze_psd(&b).iter().sum();
        assert_relative_eq!(e2 / e1, 4.0, epsilon = 0.15, max_relative = 0.15);
    }

    #[test]
    fn win_len_decreases_with_freq() {
        let nsgt = NsgtKernelLog2::new(
            NsgtLog2Config::default(),
            Log2Space::new(20.0, 8000.0, 96),
            None,
        );
        let lens: Vec<usize> = nsgt.bands().iter().map(|b| b.win_len).collect();
        for w in lens.windows(2) {
            assert!(w[0] >= w[1], "win_len not monotonic: {} < {}", w[0], w[1]);
        }
    }

    #[test]
    fn log2_space_linear_check() {
        let space = Log2Space::new(20.0, 8000.0, 48);
        let diffs: Vec<f32> = space.centers_log2.windows(2).map(|x| x[1] - x[0]).collect();
        let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
        let var = diffs
            .iter()
            .map(|d| (d - mean).abs() / mean)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        assert!(var < 1e-3, "log2 spacing nonuniform >0.1%");
    }

    #[test]
    fn empty_signal_returns_empty() {
        let nsgt = NsgtKernelLog2::new(
            NsgtLog2Config::default(),
            Log2Space::new(20.0, 8000.0, 48),
            None,
        );
        let out = nsgt.analyze(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn impulse_peak_aligns_with_time_tag() {
        let fs = 48_000.0;
        let nsgt = NsgtKernelLog2::new(
            NsgtLog2Config {
                fs,
                overlap: 0.98,
                nfft_override: Some(256),
                ..Default::default()
            },
            Log2Space::new(4000.0, 8000.0, 12),
            None,
        );
        let mut sig = vec![0.0f32; 512];
        let imp = 200usize;
        sig[imp] = 1.0;

        let bands = nsgt.analyze(&sig);
        let band = bands.last().expect("no bands returned");
        let (i_peak, _) = band
            .coeffs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.norm_sqr().partial_cmp(&b.1.norm_sqr()).unwrap())
            .expect("no coefficients returned");

        let t_peak = band.t_sec[i_peak];
        let hop = nsgt.hop() as f32;
        let err = (t_peak * fs - imp as f32).abs();

        assert!(
            err <= hop,
            "impulse peak misaligned: err={err:.2} samples (hop={hop:.2})"
        );
    }

    #[test]
    fn impulse_time_alignment_center_vs_right() {
        let fs = 48_000.0;
        let nfft = 1024;
        let space = Log2Space::new(100.0, 5000.0, 96);
        let target_f = 1000.0;

        let cfg_center = NsgtLog2Config {
            fs,
            overlap: 0.5,
            nfft_override: Some(nfft),
            kernel_align: KernelAlign::Center,
        };
        let nsgt_center = NsgtKernelLog2::new(cfg_center, space.clone(), None);
        let bi = nsgt_center
            .bands()
            .iter()
            .enumerate()
            .min_by(|a, b| {
                let da = (a.1.f_hz - target_f).abs();
                let db = (b.1.f_hz - target_f).abs();
                da.partial_cmp(&db).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap();

        let hop = nsgt_center.hop();
        let frame_idx = 3usize;
        let start = frame_idx * hop;
        let len = start + nfft + 1;

        let mut sig_center = vec![0.0f32; len];
        let t_imp_center = start + nfft / 2;
        sig_center[t_imp_center] = 1.0;
        let bands_center = nsgt_center.analyze(&sig_center);
        let band_center = &bands_center[bi];
        let (i_peak_center, _) = band_center
            .coeffs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.norm_sqr().partial_cmp(&b.1.norm_sqr()).unwrap())
            .unwrap();
        assert_eq!(i_peak_center, frame_idx, "center peak frame mismatch");
        let expected_center = (start + nfft / 2) as f32 / fs;
        let got_center = band_center.t_sec[frame_idx];
        assert!(
            (got_center - expected_center).abs() < 1e-9,
            "center t_sec mismatch: got={}, expected={}",
            got_center,
            expected_center
        );

        let cfg_right = NsgtLog2Config {
            fs,
            overlap: 0.5,
            nfft_override: Some(nfft),
            kernel_align: KernelAlign::Right,
        };
        let nsgt_right = NsgtKernelLog2::new(cfg_right, space, None);
        let win_len = nsgt_right.bands()[bi].win_len;
        let center = win_len / 2;

        let mut sig_right = vec![0.0f32; len];
        let t_imp_right = start + (nfft - win_len + center);
        sig_right[t_imp_right] = 1.0;
        let bands_right = nsgt_right.analyze(&sig_right);
        let band_right = &bands_right[bi];
        let (i_peak_right, _) = band_right
            .coeffs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.norm_sqr().partial_cmp(&b.1.norm_sqr()).unwrap())
            .unwrap();
        assert_eq!(i_peak_right, frame_idx, "right peak frame mismatch");
        let expected_right = (start + nfft - 1) as f32 / fs;
        let got_right = band_right.t_sec[frame_idx];
        assert!(
            (got_right - expected_right).abs() < 1e-9,
            "right t_sec mismatch: got={}, expected={}",
            got_right,
            expected_right
        );
    }

    #[test]
    fn phase_shift_sign_matches_standard_analysis() {
        let fs = 48_000.0;
        let nsgt = NsgtKernelLog2::new(
            NsgtLog2Config {
                fs,
                overlap: 0.98,
                nfft_override: Some(512),
                ..Default::default()
            },
            Log2Space::new(4000.0, 8000.0, 12),
            None,
        );

        let bi = nsgt.bands().len() - 1;
        let f = nsgt.bands()[bi].f_hz;
        let omega = 2.0 * std::f32::consts::PI * f / fs;
        let len = nsgt.nfft() * 8;
        let delta = 1usize;
        let delta_f = delta as f32;

        let x0_cos: Vec<f32> = (0..len).map(|n| (omega * n as f32).cos()).collect();
        let x0_sin: Vec<f32> = (0..len).map(|n| (omega * n as f32).sin()).collect();
        let x1_cos: Vec<f32> = (0..len)
            .map(|n| (omega * (n as f32 - delta_f)).cos())
            .collect();
        let x1_sin: Vec<f32> = (0..len)
            .map(|n| (omega * (n as f32 - delta_f)).sin())
            .collect();

        let bands0_cos = nsgt.analyze(&x0_cos);
        let bands0_sin = nsgt.analyze(&x0_sin);
        let bands1_cos = nsgt.analyze(&x1_cos);
        let bands1_sin = nsgt.analyze(&x1_sin);

        let fi = bands0_cos[bi].coeffs.len() / 2;
        let c0_cos = bands0_cos[bi].coeffs[fi];
        let c0_sin = bands0_sin[bi].coeffs[fi];
        let c1_cos = bands1_cos[bi].coeffs[fi];
        let c1_sin = bands1_sin[bi].coeffs[fi];

        let j = Complex32::new(0.0, 1.0);
        let c0 = c0_cos + j * c0_sin;
        let c1 = c1_cos + j * c1_sin;

        assert!(c0.norm() > 1e-3, "base coefficient too small");

        let d = c1 * c0.conj();
        let phase = d.im.atan2(d.re);
        let expected = -omega * delta_f;
        let err = wrap_to_pi(phase - expected).abs();

        assert!(err < 1e-2, "phase error too large: {err}");
    }

    #[test]
    fn low_vs_high_freq_energy_scaling() {
        // Check that total energy is roughly frequency-independent.
        let fs = 48_000.0;
        let space = Log2Space::new(20.0, 8000.0, 96);
        let nsgt = NsgtKernelLog2::new(NsgtLog2Config::default(), space.clone(), None);

        let sig_low = mk_sine_len(fs, 220.0, nsgt.nfft());
        let sig_high = mk_sine_len(fs, 1760.0, nsgt.nfft());

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
    fn parseval_consistency() {
        let fs = 48_000.0;
        let f0 = 880.0;
        let space = Log2Space::new(20.0, 8000.0, 200);
        let nsgt = NsgtKernelLog2::new(NsgtLog2Config::default(), space.clone(), None);
        let sig = mk_sine_len(fs, f0, nsgt.nfft());

        let e_time = sig.iter().map(|x| x * x).sum::<f32>() / sig.len() as f32;
        let psd = nsgt.analyze_psd(&sig);

        let e_freq = psd
            .iter()
            .zip(space.centers_hz.windows(2))
            .map(|(p, f)| p * (f[1] - f[0]))
            .sum::<f32>();

        assert_relative_eq!(e_freq / e_time, 1.0, epsilon = 0.1);
    }

    #[test]
    fn hop_size_stability() {
        let fs = 48_000.0;
        let space = Log2Space::new(20.0, 8000.0, 96);
        let sig = mk_sine(fs, 440.0, 1.0);

        let e_mean = |overlap: f32| {
            let nsgt = NsgtKernelLog2::new(
                NsgtLog2Config {
                    fs,
                    overlap,
                    ..Default::default()
                },
                space.clone(),
                None,
            );
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
        let nsgt = NsgtKernelLog2::new(NsgtLog2Config::default(), space.clone(), None);
        assert_eq!(
            nsgt.bands().len(),
            space.centers_hz.len(),
            "Band count must match Log2Space bin count"
        );
    }

    #[test]
    fn single_tone_peak_shape() {
        let fs = 48_000.0;
        let f0 = 1000.0;
        let space = Log2Space::new(100.0, 5000.0, 200);
        let nsgt = NsgtKernelLog2::new(NsgtLog2Config::default(), space.clone(), None);
        let sig = mk_sine_len(fs, f0, nsgt.nfft());

        let psd = nsgt.analyze_psd(&sig);
        let freqs = &space.centers_hz;
        let (i_peak, &p_peak) = psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let f_peak = freqs[i_peak];

        // Peak location should be within +/- 1/12 octave.
        let cents = 1200.0 * ((f_peak / f0).log2()).abs();
        assert!(
            cents < 100.0,
            "peak off by {:.1} cents @ {:.2} Hz",
            cents,
            f_peak
        );

        // Evaluate flanks only outside the main lobe.
        // Hann first zero is roughly |df| = 2/T, T = win_len / fs.
        let win_len = nsgt.bands()[i_peak].win_len as f32; // Lk_req
        let period = win_len / fs;
        let df_zero = 2.0 / period; // [Hz]

        let mut flank_max = 0.0f32;
        let mut flank_count = 0usize;
        for (i, &p) in psd.iter().enumerate() {
            let df = (freqs[i] - f_peak).abs();
            if df >= df_zero {
                flank_max = flank_max.max(p);
                flank_count += 1;
            }
        }
        assert!(
            flank_count > 0,
            "no bins outside main lobe to evaluate flank"
        );

        // Outside the main lobe, expect at least ~10x drop (~10.8 dB).
        assert!(
            p_peak / flank_max > 10.0,
            "peak too broad or flat: peak={}, flank_max={}, df_zero={:.2} Hz, Lk={}",
            p_peak,
            flank_max,
            df_zero,
            win_len as usize
        );
    }

    #[test]
    fn amplitude_scaling_quadratic() {
        let fs = 48_000.0;
        let f0 = 440.0;
        let sig1: Vec<f32> = (0..(fs as usize))
            .map(|i| (2.0 * std::f32::consts::PI * f0 * i as f32 / fs).sin())
            .collect();
        let sig2: Vec<f32> = sig1.iter().map(|x| x * 2.0).collect();

        let space = Log2Space::new(20.0, 8000.0, 96);
        let nsgt = NsgtKernelLog2::new(NsgtLog2Config::default(), space, None);
        let e1: f32 = nsgt.analyze_psd(&sig1).iter().sum();
        let e2: f32 = nsgt.analyze_psd(&sig2).iter().sum();

        assert_relative_eq!(e2 / e1, 4.0, epsilon = 0.1);
    }

    #[test]
    fn overlap_independence() {
        let fs = 48_000.0;
        let f0 = 880.0;
        let sig = (0..(fs as usize))
            .map(|i| (2.0 * std::f32::consts::PI * f0 * i as f32 / fs).sin())
            .collect::<Vec<_>>();
        let space = Log2Space::new(20.0, 8000.0, 96);

        let energies: Vec<f32> = [0.25, 0.5, 0.75]
            .iter()
            .map(|&ov| {
                let nsgt = NsgtKernelLog2::new(
                    NsgtLog2Config {
                        fs,
                        overlap: ov,
                        ..Default::default()
                    },
                    space.clone(),
                    None,
                );
                nsgt.analyze_psd(&sig).iter().sum::<f32>()
            })
            .collect();

        let mean = energies.iter().sum::<f32>() / energies.len() as f32;
        for &e in &energies {
            assert_relative_eq!(e / mean, 1.0, epsilon = 0.2);
        }
    }

    #[test]
    fn noise_slope_accuracy() {
        let fs = 48_000.0;
        let n = (fs * 4.0) as usize;

        let white: Vec<f32> = white_noise(n, 1).iter().map(|&v| v as f32).collect();
        let pink: Vec<f32> = pink_noise(n, 1).iter().map(|&v| v as f32).collect();
        let brown: Vec<f32> = brown_noise(n, 1).iter().map(|&v| v as f32).collect();

        let space = Log2Space::new(50.0, 8000.0, 150);
        let nsgt = NsgtKernelLog2::new(NsgtLog2Config::default(), space.clone(), None);

        let psd_white = nsgt.analyze_psd(&white);
        let psd_pink = nsgt.analyze_psd(&pink);
        let psd_brown = nsgt.analyze_psd(&brown);

        let slope = |psd: &[f32]| {
            let y: Vec<f32> = psd.iter().map(|v| 10.0 * v.max(1e-20).log10()).collect();
            let x = &space.centers_log2;
            let (sx, sy, sxx, sxy, n) = x.iter().zip(&y).fold(
                (0.0, 0.0, 0.0, 0.0, 0.0),
                |(sx, sy, sxx, sxy, n), (xv, yv)| {
                    (
                        sx + *xv,
                        sy + *yv,
                        sxx + *xv * *xv,
                        sxy + *xv * *yv,
                        n + 1.0,
                    )
                },
            );
            (n * sxy - sx * sy) / (n * sxx - sx * sx)
        };

        let s_white = slope(&psd_white);
        let s_pink = slope(&psd_pink);
        let s_brown = slope(&psd_brown);

        assert!(s_white.abs() < 0.8, "white slope {s_white}");
        assert!((s_pink + 3.0).abs() < 1.0, "pink slope {s_pink}");
        assert!((s_brown + 6.0).abs() < 1.5, "brown slope {s_brown}");
    }

    #[test]
    fn win_len_cap_applies_above_pivot() {
        let fs = 48_000.0;
        let nfft = 4096;
        let pivot_hz = 2000.0;
        let cap = WinLenCap {
            pivot_hz,
            max_len_above_s: 256.0 / fs,
        };
        let space = Log2Space::new(200.0, 8000.0, 96);
        let nsgt = NsgtKernelLog2::new(
            NsgtLog2Config {
                fs,
                overlap: 0.9,
                nfft_override: Some(nfft),
                ..Default::default()
            },
            space,
            Some(cap),
        );
        let cap_eff = cap.cap_samples(fs, nfft);

        let mut max_above = 0usize;
        for b in nsgt.bands().iter().filter(|b| b.f_hz >= pivot_hz) {
            assert!(
                b.win_len <= cap_eff,
                "win_len exceeds cap: f_hz={}, win_len={}, cap={}",
                b.f_hz,
                b.win_len,
                cap_eff
            );
            max_above = max_above.max(b.win_len);
        }
        assert!(
            max_above == cap_eff,
            "cap not binding above pivot: max_above={}, cap={}",
            max_above,
            cap_eff
        );
    }

    #[test]
    fn win_len_cap_limits_growth_with_bins_per_oct() {
        let fs = 48_000.0;
        let nfft = 4096;
        let pivot_hz = 2000.0;
        let cap = WinLenCap {
            pivot_hz,
            max_len_above_s: 256.0 / fs,
        };
        let cap_eff = cap.cap_samples(fs, nfft);

        for bpo in [24u32, 96u32] {
            let space = Log2Space::new(200.0, 8000.0, bpo);
            let nsgt = NsgtKernelLog2::new(
                NsgtLog2Config {
                    fs,
                    overlap: 0.9,
                    nfft_override: Some(nfft),
                    ..Default::default()
                },
                space,
                Some(cap),
            );
            let max_above = nsgt
                .bands()
                .iter()
                .filter(|b| b.f_hz >= pivot_hz)
                .map(|b| b.win_len)
                .max()
                .unwrap_or(0);
            assert!(
                max_above == cap_eff,
                "cap not enforced for bpo={}: max_above={}, cap={}",
                bpo,
                max_above,
                cap_eff
            );
        }
    }

    #[test]
    fn win_len_cap_none_allows_longer_high_band() {
        let fs = 48_000.0;
        let nfft = 4096;
        let pivot_hz = 2000.0;
        let cap = WinLenCap {
            pivot_hz,
            max_len_above_s: 256.0 / fs,
        };
        let cap_eff = cap.cap_samples(fs, nfft);
        let space = Log2Space::new(200.0, 8000.0, 96);
        let nsgt = NsgtKernelLog2::new(
            NsgtLog2Config {
                fs,
                overlap: 0.9,
                nfft_override: Some(nfft),
                ..Default::default()
            },
            space,
            None,
        );
        let max_above = nsgt
            .bands()
            .iter()
            .filter(|b| b.f_hz >= pivot_hz)
            .map(|b| b.win_len)
            .max()
            .unwrap_or(0);

        assert!(
            max_above > cap_eff,
            "expected uncapped win_len above pivot: max_above={}, cap={}",
            max_above,
            cap_eff
        );
    }

    // ==============================
    // Visualization (plotting): cargo test -- --ignored
    // ==============================

    #[test]
    #[ignore]
    fn plot_nsgt_spectrum_kernel() {
        use plotters::prelude::*;

        let fs = 48_000.0;
        let nsgt = NsgtKernelLog2::new(
            NsgtLog2Config {
                fs,
                overlap: 0.5,
                nfft_override: Some(16_384),
                ..Default::default()
            },
            Log2Space::new(20.0, 8000.0, 200),
            None,
        );
        let sig = mk_sine_len(fs, 440.0, nsgt.nfft());
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

        let fs = 48_000.0;
        let secs = 40.0;
        let n = (fs * secs) as usize;

        let nsgt = NsgtKernelLog2::new(
            NsgtLog2Config {
                fs,
                overlap: 0.5,
                ..Default::default()
            },
            Log2Space::new(35.0, 24_000.0, 100),
            None,
        );

        // Noise generation.
        let white: Vec<f32> = white_noise(n, 42).iter().map(|&v| v as f32).collect();
        let pink: Vec<f32> = pink_noise(n, 42).iter().map(|&v| v as f32).collect();
        let brown: Vec<f32> = brown_noise(n, 42).iter().map(|&v| v as f32).collect();

        // Get per-Hz normalized PSD.
        let w_psd = nsgt.analyze_psd(&white);
        let p_psd = nsgt.analyze_psd(&pink);
        let b_psd = nsgt.analyze_psd(&brown);

        // Convert to dB.
        let to_db =
            |v: &[f32]| -> Vec<f32> { v.iter().map(|x| 10.0 * x.max(1e-20).log10()).collect() };
        let w_db = to_db(&w_psd);
        let p_db = to_db(&p_psd);
        let b_db = to_db(&b_psd);

        let log2x = nsgt.space().centers_log2.clone();

        // Plot.
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
                (35f32.log2())..(24_000f32.log2()),
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
