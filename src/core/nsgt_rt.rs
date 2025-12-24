//! core/nsgt_rt.rs — Real-time NSGT with per-hop exponential smoothing
//!
//! Design
//! -----
//! - 1 hop = 1 FFT update. A ring buffer (length = nfft) holds the latest samples.
//! - Uses NsgtKernelLog2’s sparse frequency-domain kernels (no per-hop heap alloc).
//! - Per-band exponential integrator: y[n] = (1−α)x[n] + α y[n−1], α = exp(−dt/τ(f)).
//! - τ(f) is mapped low→slow, high→fast via simple f-dependent rule.
//!
//! Notes
//! -----
//! - Instantaneous measure is raw power (coherent or incoherent, fixed at kernel creation).
//! - `process_hop()` accepts ≤ hop samples (short reads are zero-padded) or
//!   > hop (multiple hops processed); returns the last envelope slice.
//! - No per-hop allocations; FFT and scratch buffers are reused.

use crate::core::log2space::Log2Space;
use crate::core::nsgt_kernel::{NsgtKernelLog2, PowerMode};
use rustfft::{FftPlanner, num_complex::Complex32};
use std::sync::Arc;

/// Smoothing configuration.
#[derive(Clone, Copy, Debug)]
pub struct RtConfig {
    /// τ at high frequencies [s].
    pub tau_min: f32,
    /// τ at low frequencies [s].
    pub tau_max: f32,
    /// Reference frequency for mapping τ(f) [Hz].
    pub f_ref: f32,
}

impl Default for RtConfig {
    fn default() -> Self {
        Self {
            tau_min: 0.005,
            tau_max: 0.020,
            f_ref: 200.0,
        }
    }
}

/// Per-band persistent state.
#[derive(Clone, Debug)]
pub struct BandState {
    /// Center frequency [Hz].
    pub f_hz: f32,
    /// Time constant [s].
    pub tau: f32,
    /// α = exp(−dt/τ).
    pub alpha: f32,
    /// ENBW [Hz] (Hann ≈ 1.5*fs/Lk).
    pub enbw_hz: f32,
    /// Smoothed band power (running state).
    pub smooth: f32,
}

/// Real-time kernel analyzer with per-hop update and leaky integration.
#[derive(Clone)]
pub struct RtNsgtKernelLog2 {
    // analysis core
    nsgt: NsgtKernelLog2,
    fs: f32,
    nfft: usize,
    hop: usize,
    dt: f32,

    // runtime buffers (no per-hop alloc)
    ring: Vec<f32>,
    write_pos: usize, // next position to write
    fft: Arc<dyn rustfft::Fft<f32>>,
    fft_buf: Vec<Complex32>,

    // per-band states & cached meta
    bands_state: Vec<BandState>,
    out_env: Vec<f32>,

    // settings
    power_mode: PowerMode,
}

impl RtNsgtKernelLog2 {
    /// Construct with default RtConfig.
    pub fn new(nsgt: NsgtKernelLog2) -> Self {
        Self::with_config(nsgt, RtConfig::default())
    }

    /// Construct with explicit RtConfig.
    pub fn with_config(nsgt: NsgtKernelLog2, cfg: RtConfig) -> Self {
        let fs = nsgt.cfg.fs;
        let nfft = nsgt.nfft();
        let hop = nsgt.hop();
        let dt = hop as f32 / fs;
        let power_mode = nsgt.power_mode;

        // Our own FFT plan (no per-hop allocation; same size as nsgt).
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(nfft);

        // Build band states (τ mapping and ENBW precompute).
        let bands_state = nsgt
            .bands()
            .iter()
            .map(|b| {
                let f = b.f_hz.max(1e-6); // guard for f=0
                let ratio = (cfg.f_ref / f).clamp(0.0, 1.0);
                let mut tau = cfg.tau_min + (cfg.tau_max - cfg.tau_min) * ratio;
                tau = tau.clamp(cfg.tau_min, cfg.tau_max);
                let alpha = (-dt / tau).exp();
                let enbw_hz = 1.5_f32 * fs / (b.win_len as f32);
                BandState {
                    f_hz: b.f_hz,
                    tau,
                    alpha,
                    enbw_hz,
                    smooth: 0.0,
                }
            })
            .collect::<Vec<_>>();

        Self {
            nsgt,
            fs,
            nfft,
            hop,
            dt,
            ring: vec![0.0; nfft],
            write_pos: 0,
            fft,
            fft_buf: vec![Complex32::new(0.0, 0.0); nfft],
            out_env: vec![0.0; bands_state.len()],
            bands_state,
            power_mode,
        }
    }

    /// Push one hop of audio and return the smoothed band power/PSD slice.
    ///
    /// - If `hop_in.len() == hop()`: normal per-hop update.
    /// - If shorter: zero-padded and still one update.
    /// - If longer: processes multiple hops internally; returns the last envelope.
    pub fn process_hop(&mut self, hop_in: &[f32]) -> &[f32] {
        if hop_in.len() <= self.hop {
            self.write_hop_zero_padded(hop_in);
            self.analyze_one_and_update();
            &self.out_env
        } else {
            // Process multiple hops
            let mut i = 0usize;
            while i + self.hop <= hop_in.len() {
                self.write_hop(&hop_in[i..i + self.hop]);
                self.analyze_one_and_update();
                i += self.hop;
            }
            if i < hop_in.len() {
                self.write_hop_zero_padded(&hop_in[i..]);
                self.analyze_one_and_update();
            }
            &self.out_env
        }
    }

    /// Process an arbitrary block and emit per-hop band power/PSD slices via callback.
    pub fn process_block_emit<F: FnMut(&[f32])>(&mut self, block: &[f32], mut emit: F) {
        let mut i = 0usize;
        while i + self.hop <= block.len() {
            self.write_hop(&block[i..i + self.hop]);
            self.analyze_one_and_update();
            emit(&self.out_env);
            i += self.hop;
        }
        if i < block.len() {
            self.write_hop_zero_padded(&block[i..]);
            self.analyze_one_and_update();
            emit(&self.out_env);
        }
    }

    /// Current smoothed band power/PSD without processing new samples.
    #[inline]
    pub fn current_envelope(&self) -> &[f32] {
        &self.out_env
    }

    /// Accessors
    #[inline]
    pub fn hop(&self) -> usize {
        self.hop
    }
    #[inline]
    pub fn dt(&self) -> f32 {
        self.dt
    }
    #[inline]
    pub fn fs(&self) -> f32 {
        self.fs
    }
    #[inline]
    pub fn nfft(&self) -> usize {
        self.nfft
    }
    /// Center frequencies [Hz].
    pub fn freqs(&self) -> Vec<f32> {
        self.nsgt.bands().iter().map(|b| b.f_hz).collect()
    }
    /// Underlying log2 space.
    #[inline]
    pub fn space(&self) -> &Log2Space {
        self.nsgt.space()
    }

    /// Reconfigure τ mapping and recompute α (no allocation).
    pub fn reconfigure_smoothing(&mut self, tau_min: f32, tau_max: f32, f_ref: f32) {
        let tau_min = tau_min.max(1e-6);
        let tau_max = tau_max.max(tau_min);
        for (b, st) in self.nsgt.bands().iter().zip(self.bands_state.iter_mut()) {
            let f = b.f_hz.max(1e-6);
            let ratio = (f_ref / f).clamp(0.0, 1.0);
            let mut tau = tau_min + (tau_max - tau_min) * ratio;
            tau = tau.clamp(tau_min, tau_max);
            st.tau = tau;
            st.alpha = (-self.dt / tau).exp();
        }
    }

    /// Reset smoothing states and ring buffer.
    pub fn reset(&mut self) {
        for st in &mut self.bands_state {
            st.smooth = 0.0;
        }
        for x in &mut self.ring {
            *x = 0.0;
        }
        self.write_pos = 0;
        for z in &mut self.fft_buf {
            *z = Complex32::new(0.0, 0.0);
        }
        for y in &mut self.out_env {
            *y = 0.0;
        }
    }

    // ---- internal helpers ----

    #[inline]
    fn write_hop(&mut self, hop_in: &[f32]) {
        debug_assert_eq!(hop_in.len(), self.hop);
        let n = self.nfft;
        let mut wp = self.write_pos;
        // write hop samples into the ring
        for &s in hop_in {
            self.ring[wp] = s;
            wp += 1;
            if wp == n {
                wp = 0;
            }
        }
        self.write_pos = wp;
    }

    #[inline]
    fn write_hop_zero_padded(&mut self, hop_in: &[f32]) {
        let n = self.nfft;
        let mut wp = self.write_pos;
        let mut i = 0usize;
        // copy provided samples
        while i < hop_in.len() {
            self.ring[wp] = hop_in[i];
            wp += 1;
            if wp == n {
                wp = 0;
            }
            i += 1;
        }
        // zero-pad the remainder
        while i < self.hop {
            self.ring[wp] = 0.0;
            wp += 1;
            if wp == n {
                wp = 0;
            }
            i += 1;
        }
        self.write_pos = wp;
    }

    /// Build contiguous FFT frame from ring, run FFT, accumulate bands, and update smoothing.
    fn analyze_one_and_update(&mut self) {
        // Reassemble latest nfft samples: [write_pos..end) then [0..write_pos)
        let n = self.nfft;
        let left = n - self.write_pos;

        // Fill fft_buf with real input (imag=0), then FFT.
        for i in 0..left {
            let s = self.ring[self.write_pos + i];
            self.fft_buf[i] = Complex32::new(s, 0.0);
        }
        for i in 0..self.write_pos {
            let s = self.ring[i];
            self.fft_buf[left + i] = Complex32::new(s, 0.0);
        }

        self.fft.process(&mut self.fft_buf);

        // Sparse inner products (same math as NsgtKernelLog2::analyze)
        let bands = self.nsgt.bands();
        let inv_nfft = 1.0 / n as f32;
        let inv_nfft2 = inv_nfft * inv_nfft;
        for (bi, band) in bands.iter().enumerate() {
            // Instantaneous measure (power mode fixed at kernel creation time)
            let p = match self.power_mode {
                PowerMode::Coherent => {
                    let sparse = band
                        .spec_conj_sparse
                        .as_ref()
                        .expect("coherent kernel required for RT");
                    let mut acc = Complex32::new(0.0, 0.0);
                    // Σ X[k] * conj(K_k[k]) (already conj in spec_conj_sparse)
                    for &(k, w) in sparse {
                        // Safety: kernels are built for the same nfft.
                        debug_assert!(k < self.fft_buf.len());
                        acc += self.fft_buf[k] * w;
                    }
                    acc *= inv_nfft;
                    acc.norm_sqr()
                }
                PowerMode::Incoherent => {
                    let sparse = band
                        .spec_pow_sparse
                        .as_ref()
                        .expect("incoherent kernel required for RT");
                    let mut sum = 0.0f32;
                    for &(k, w_pow) in sparse {
                        debug_assert!(k < self.fft_buf.len());
                        sum += self.fft_buf[k].norm_sqr() * w_pow;
                    }
                    sum * inv_nfft2
                }
            };

            // Exponential smoothing
            let st = &mut self.bands_state[bi];
            st.smooth = (1.0 - st.alpha) * p + st.alpha * st.smooth;
            self.out_env[bi] = st.smooth;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::log2space::Log2Space;
    use crate::core::nsgt_kernel::NsgtLog2Config;
    use core::f32::consts::PI;

    fn mk_sine(len: usize, f_hz: f32, fs: f32, amp: f32) -> Vec<f32> {
        let w = 2.0 * PI * f_hz / fs;
        (0..len).map(|i| amp * (w * i as f32).cos()).collect()
    }

    fn mk_two_sine(len: usize, f1_hz: f32, f2_hz: f32, fs: f32, amp: f32) -> Vec<f32> {
        let w1 = 2.0 * PI * f1_hz / fs;
        let w2 = 2.0 * PI * f2_hz / fs;
        (0..len)
            .map(|i| {
                let t = i as f32;
                amp * (w1 * t).cos() + amp * (w2 * t).cos()
            })
            .collect()
    }

    fn std_dev(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let mut var = 0.0f32;
        for &v in data {
            let d = v - mean;
            var += d * d;
        }
        (var / data.len() as f32).sqrt()
    }

    #[test]
    fn rt_raw_power_scales_quadratically() {
        let fs = 48_000.0;
        let space = Log2Space::new(200.0, 4000.0, 12);
        let nsgt = NsgtKernelLog2::new(
            NsgtLog2Config {
                fs,
                overlap: 0.5,
                nfft_override: Some(256),
                ..Default::default()
            },
            space,
            None,
            PowerMode::Coherent,
        );
        let cfg = RtConfig {
            tau_min: 1e-6,
            tau_max: 1e-6,
            f_ref: 200.0,
        };

        let mut rt = RtNsgtKernelLog2::with_config(nsgt.clone(), cfg);
        let f = rt.freqs()[rt.freqs().len() / 2];
        let len = rt.nfft() * 4;
        let sig1 = mk_sine(len, f, fs, 1.0);
        let sig2 = mk_sine(len, f, fs, 2.0);

        let mut env1 = Vec::new();
        rt.process_block_emit(&sig1, |env| env1 = env.to_vec());

        let mut rt2 = RtNsgtKernelLog2::with_config(nsgt, cfg);
        let mut env2 = Vec::new();
        rt2.process_block_emit(&sig2, |env| env2 = env.to_vec());

        let (imax, p1) = env1
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert!(*p1 > 1e-6, "unexpected near-zero band power");
        let ratio = env2[imax] / *p1;
        assert!(
            (ratio - 4.0).abs() < 0.1,
            "expected ~4x power scaling, got {ratio:.3}"
        );
    }

    #[test]
    fn rt_incoherent_power_reduces_beating_variance() {
        let fs = 48_000.0;
        let space = Log2Space::new(200.0, 4000.0, 12);
        let nsgt_coh = NsgtKernelLog2::new(
            NsgtLog2Config {
                fs,
                overlap: 0.5,
                nfft_override: Some(512),
                ..Default::default()
            },
            space.clone(),
            None,
            PowerMode::Coherent,
        );
        let nsgt_incoh = NsgtKernelLog2::new(
            NsgtLog2Config {
                fs,
                overlap: 0.5,
                nfft_override: Some(512),
                ..Default::default()
            },
            space,
            None,
            PowerMode::Incoherent,
        );
        let cfg = RtConfig {
            tau_min: 1e-6,
            tau_max: 1e-6,
            f_ref: 200.0,
        };
        let f1 = 1000.0;
        let f2 = 1007.0;
        let len = nsgt_coh.nfft() * 16;
        let sig = mk_two_sine(len, f1, f2, fs, 0.5);

        let mut rt_coh = RtNsgtKernelLog2::with_config(nsgt_coh, cfg);
        let mut rt_incoh = RtNsgtKernelLog2::with_config(nsgt_incoh, cfg);

        let band_idx = rt_coh
            .freqs()
            .iter()
            .enumerate()
            .min_by(|a, b| {
                (a.1 - f1)
                    .abs()
                    .partial_cmp(&(b.1 - f1).abs())
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut series_coh = Vec::new();
        rt_coh.process_block_emit(&sig, |env| series_coh.push(env[band_idx]));
        let mut series_incoh = Vec::new();
        rt_incoh.process_block_emit(&sig, |env| series_incoh.push(env[band_idx]));

        let std_coh = std_dev(&series_coh);
        let std_incoh = std_dev(&series_incoh);
        assert!(
            std_incoh < std_coh * 0.8,
            "expected incoherent variance to drop: std_incoh={std_incoh:.6} std_coh={std_coh:.6}"
        );
    }

    #[test]
    fn rt_incoherent_power_scales_quadratically() {
        let fs = 48_000.0;
        let space = Log2Space::new(200.0, 4000.0, 12);
        let nsgt = NsgtKernelLog2::new(
            NsgtLog2Config {
                fs,
                overlap: 0.5,
                nfft_override: Some(256),
                ..Default::default()
            },
            space,
            None,
            PowerMode::Incoherent,
        );
        let cfg = RtConfig {
            tau_min: 1e-6,
            tau_max: 1e-6,
            f_ref: 200.0,
        };

        let mut rt = RtNsgtKernelLog2::with_config(nsgt.clone(), cfg);
        let f = rt.freqs()[rt.freqs().len() / 2];
        let len = rt.nfft() * 4;
        let sig1 = mk_sine(len, f, fs, 1.0);
        let sig2 = mk_sine(len, f, fs, 2.0);

        let mut env1 = Vec::new();
        rt.process_block_emit(&sig1, |env| env1 = env.to_vec());

        let mut rt2 = RtNsgtKernelLog2::with_config(nsgt, cfg);
        let mut env2 = Vec::new();
        rt2.process_block_emit(&sig2, |env| env2 = env.to_vec());

        let (imax, p1) = env1
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert!(*p1 > 1e-6, "unexpected near-zero band power");
        let ratio = env2[imax] / *p1;
        assert!(
            (ratio - 4.0).abs() < 0.1,
            "expected ~4x power scaling, got {ratio:.3}"
        );
    }
}
