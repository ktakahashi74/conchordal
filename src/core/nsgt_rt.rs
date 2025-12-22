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
//! - Instantaneous measure can be raw power |C_k|^2 or one-sided PSD (~power/Hz).
//! - `process_hop()` accepts ≤ hop samples (short reads are zero-padded) or
//!   > hop (multiple hops processed); returns the last envelope slice.
//! - No per-hop allocations; FFT and scratch buffers are reused.

use crate::core::log2space::Log2Space;
use crate::core::nsgt_kernel::NsgtKernelLog2;
use rustfft::{FftPlanner, num_complex::Complex32};
use std::sync::Arc;

/// Instantaneous measure used before smoothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstMeasure {
    /// Instantaneous band power: |C_k|^2 (no per-Hz normalization).
    RawPower,
    /// One-sided PSD-like measure: 2 * (|C_k|^2 / ENBW), ENBW for periodic Hann ≈ 1.5 * fs / L_k.
    PsdOneSided,
}

/// Smoothing configuration and instantaneous measure selection.
#[derive(Clone, Copy, Debug)]
pub struct RtConfig {
    /// τ at high frequencies [s].
    pub tau_min: f32,
    /// τ at low frequencies [s].
    pub tau_max: f32,
    /// Reference frequency for mapping τ(f) [Hz].
    pub f_ref: f32,
    /// Instantaneous measure type.
    pub measure: InstMeasure,
}

impl Default for RtConfig {
    fn default() -> Self {
        Self {
            tau_min: 0.03,
            tau_max: 0.30,
            f_ref: 200.0,
            measure: InstMeasure::RawPower,
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
    /// ENBW [Hz] (periodic Hann ≈ 1.5*fs/Lk).
    pub enbw_hz: f32,
    /// Smoothed envelope (running state).
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
    measure: InstMeasure,
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
            measure: cfg.measure,
        }
    }

    /// Push one hop of audio and return the smoothed envelope slice.
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

    /// Process an arbitrary block and emit per-hop envelopes via callback.
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

    /// Current smoothed envelope without processing new samples.
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

    /// Change measure (raw power / PSD) on the fly.
    pub fn set_measure(&mut self, m: InstMeasure) {
        self.measure = m;
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
        for (bi, band) in bands.iter().enumerate() {
            let mut acc = Complex32::new(0.0, 0.0);
            // Σ X[k] * conj(K_k[k]) (already conj in spec_conj_sparse)
            for &(k, w) in &band.spec_conj_sparse {
                // Safety: kernels are built for the same nfft.
                debug_assert!(k < self.fft_buf.len());
                acc += self.fft_buf[k] * w;
            }
            acc /= n as f32;

            // Instantaneous measure
            let p = match self.measure {
                InstMeasure::RawPower => acc.norm_sqr(),
                InstMeasure::PsdOneSided => {
                    // 2 * power / ENBW (one-sided)
                    let enbw = self.bands_state[bi].enbw_hz.max(1e-12);
                    2.0 * (acc.norm_sqr() / enbw)
                }
            };

            // Exponential smoothing
            let st = &mut self.bands_state[bi];
            st.smooth = (1.0 - st.alpha) * p + st.alpha * st.smooth;
            self.out_env[bi] = st.smooth;
        }
    }
}
