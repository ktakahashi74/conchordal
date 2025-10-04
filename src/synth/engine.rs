use rustfft::num_complex::Complex32;

use crate::core::fft::{ISTFT, bin_freqs_hz};

#[derive(Clone, Debug)]
pub struct SynthConfig {
    pub fs: f32,
    pub fft_size: usize,
    pub hop: usize,
    pub n_bins: usize,
}

pub struct SynthEngine {
    cfg: SynthConfig,
    istft: ISTFT,
    phase: Vec<f32>, // phase accumulator per bin
    bin_freqs: Vec<f32>,
}

impl SynthEngine {
    pub fn new(cfg: SynthConfig) -> Self {
        let istft = ISTFT::new(cfg.fft_size, cfg.hop);
        let phase = vec![0.0f32; cfg.n_bins];
        let bin_freqs = bin_freqs_hz(cfg.fs, cfg.fft_size);
        Self {
            cfg,
            istft,
            phase,
            bin_freqs,
        }
    }

    pub fn bin_freqs_hz(&self) -> Vec<f32> {
        self.bin_freqs.clone()
    }

    /// Render one hop of audio from magnitude spectrum (amps per bin).
    pub fn render_hop(&mut self, amps: &[f32]) -> Vec<f32> {
        let n_half = self.cfg.n_bins;
        let hop_t = self.cfg.hop as f32 / self.cfg.fs;

        let mut half_spec = vec![Complex32::new(0.0, 0.0); n_half];
        for k in 0..n_half {
            // advance phase
            let omega = 2.0 * std::f32::consts::PI * self.bin_freqs[k];
            self.phase[k] = (self.phase[k] + omega * hop_t) % (2.0 * std::f32::consts::PI);

            // magnitude to complex using current phase
            let (s, c) = self.phase[k].sin_cos();
            half_spec[k] = Complex32::new(c * amps[k], s * amps[k]);
        }
        // iSTFT overlap-add
        self.istft.process(&half_spec)
    }
}
