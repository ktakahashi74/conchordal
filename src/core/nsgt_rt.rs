//! core/nsgt_rt.rs — Real-time NSGT with exponential temporal integration
//!
//! Overview
//! -----
//! - Wraps `NsgtKernelLog2` to provide frame-by-frame real-time analysis.
//! - Each band keeps a leaky integrator state (`smooth_pow`) updated every hop.
//! - Time constant τ(f) determines smoothing per band (low bands → slower).
//!
//! Typical usage
//! -----
//! ```ignore
//! let base = NsgtKernelLog2::new(cfg, space);
//! let mut rt = RtNsgtKernelLog2::new(base);
//! for frame in input_stream.chunks(rt.hop()) {
//!     let smoothed = rt.process_frame(frame);
//!     landscape.update(&smoothed);
//! }
//! ```
//!
//! Design Notes
//! -----
//! - FFT and sparse kernels are reused from `NsgtKernelLog2`.
//! - Temporal integration is exponential: y[t] = (1−α)x[t] + αy[t−1], α=exp(−Δt/τ).
//! - τ(f) is frequency-dependent, e.g. τ = τ_min + (τ_max−τ_min)*(f_ref/f).

use crate::core::log2::Log2Space;
use crate::core::nsgt_kernel::{BandCoeffs, NsgtKernelLog2};
use rustfft::num_complex::Complex32;

/// Per-band temporal state for real-time NSGT.
#[derive(Clone, Debug)]
pub struct BandState {
    /// Band center frequency [Hz]
    pub f_hz: f32,
    /// Time constant [s]
    pub tau: f32,
    /// Leaky coefficient α = exp(−Δt/τ)
    pub alpha: f32,
    /// Smoothed power (persistent state)
    pub smooth_pow: f32,
}

/// Real-time NSGT kernel analyzer with leaky temporal integration.
///
/// - Maintains state across frames.
/// - Suitable for 10–20 ms updates in real-time audio streams.
#[derive(Clone)]
pub struct RtNsgtKernelLog2 {
    pub nsgt: NsgtKernelLog2,
    pub bands_state: Vec<BandState>,
    pub fs: f32,
    pub hop: usize,
    pub dt: f32,
}

impl RtNsgtKernelLog2 {
    /// Create a real-time NSGT wrapper from an existing `NsgtKernelLog2`.
    ///
    /// τ(f) is assigned automatically as:
    /// τ = τ_min + (τ_max − τ_min) * (f_ref / f)
    /// clamped to [τ_min, τ_max].
    pub fn new(nsgt: NsgtKernelLog2) -> Self {
        let fs = nsgt.cfg.fs; // 修正：Log2Spaceにはfsがないのでcfgから取得
        let hop = nsgt.hop();
        let dt = hop as f32 / fs;

        let tau_min: f32 = 0.03; // [s] high frequencies
        let tau_max: f32 = 0.30; // [s] low frequencies
        let f_ref: f32 = 200.0; // [Hz]

        let bands_state = nsgt
            .bands()
            .iter()
            .map(|b| {
                let f_hz = b.f_hz; // 修正：private対策 → pub化済み or getter前提
                let mut tau: f32 = tau_min + (tau_max - tau_min) * (f_ref / f_hz).clamp(0.0, 1.0);
                tau = tau.clamp(tau_min, tau_max);
                let alpha = (-dt / tau).exp();
                BandState {
                    f_hz,
                    tau,
                    alpha,
                    smooth_pow: 0.0,
                }
            })
            .collect();

        Self {
            nsgt,
            bands_state,
            fs,
            hop,
            dt,
        }
    }

    /// Process one frame of audio (length ≥ nfft).
    /// Returns smoothed band powers.
    pub fn process_frame(&mut self, frame: &[f32]) -> Vec<f32> {
        let bands = self.nsgt.analyze(frame);
        self.update_states(&bands);
        self.bands_state.iter().map(|b| b.smooth_pow).collect()
    }

    /// Internal update: exponential temporal integration for each band.
    fn update_states(&mut self, bands: &[BandCoeffs]) {
        for (b, st) in bands.iter().zip(self.bands_state.iter_mut()) {
            let inst_pow = if b.coeffs.is_empty() {
                0.0
            } else {
                b.coeffs.iter().map(|z| z.norm_sqr()).sum::<f32>() / (b.coeffs.len().max(1) as f32)
            };
            st.smooth_pow = (1.0 - st.alpha) * inst_pow + st.alpha * st.smooth_pow;
        }
    }

    /// Reset all internal states (e.g. on stream start)
    pub fn reset(&mut self) {
        for st in &mut self.bands_state {
            st.smooth_pow = 0.0;
        }
    }

    /// Return current smoothed powers without processing a new frame.
    pub fn current_envelope(&self) -> Vec<f32> {
        self.bands_state.iter().map(|b| b.smooth_pow).collect()
    }

    /// Accessor: hop size [samples]
    pub fn hop(&self) -> usize {
        self.hop
    }

    /// Accessor: frame interval [s]
    pub fn dt(&self) -> f32 {
        self.dt
    }

    /// Accessor: band frequencies [Hz]
    pub fn freqs(&self) -> Vec<f32> {
        self.bands_state.iter().map(|b| b.f_hz).collect()
    }

    /// Accessor: reference to inner Log2Space
    pub fn space(&self) -> &Log2Space {
        self.nsgt.space()
    }
}

// =====================================================
// Tests (basic functionality, not real-time audio thread)
// =====================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::log2::Log2Space;
    use crate::core::nsgt_kernel::NsgtLog2Config;

    fn mk_sine(fs: f32, f: f32, secs: f32) -> Vec<f32> {
        let n = (fs * secs).round() as usize;
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f * (i as f32) / fs).sin())
            .collect()
    }

    #[test]
    fn rt_smoothing_converges() {
        let fs = 48_000.0;
        let secs = 0.5;
        let sig = mk_sine(fs, 440.0, secs);
        let nsgt = NsgtKernelLog2::new(
            NsgtLog2Config {
                fs,
                overlap: 0.5,
                ..Default::default()
            },
            Log2Space::new(200.0, 4000.0, 96),
        );
        let mut rt = RtNsgtKernelLog2::new(nsgt);
        let hop = rt.hop();

        let mut last_mean = 0.0;
        for chunk in sig.chunks(hop) {
            let env = rt.process_frame(chunk);
            let mean = env.iter().sum::<f32>() / env.len() as f32;
            last_mean = mean;
        }

        assert!(last_mean > 0.0, "no envelope response");
        assert!(last_mean.is_finite(), "invalid envelope value (NaN or inf)");
    }

    #[test]
    fn reset_works() {
        let fs = 48_000.0;
        let nsgt =
            NsgtKernelLog2::new(NsgtLog2Config::default(), Log2Space::new(200.0, 4000.0, 96));
        let mut rt = RtNsgtKernelLog2::new(nsgt);
        for st in &mut rt.bands_state {
            st.smooth_pow = 1.0;
        }
        rt.reset();
        assert!(rt.bands_state.iter().all(|b| b.smooth_pow == 0.0));
    }
}
