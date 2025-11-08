//! core/landscape.rs — Landscape computed on log2(NSGT-RT) domain.
//! Each incoming frame updates the potential roughness (R) and consonance (C)
//! using a real-time NSGT analyzer. A lightweight psychoacoustic normalization
//! (subjective intensity + leaky integration) runs before kernels.

use crate::core::consonance_kernel::ConsonanceKernel;
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::core::roughness_kernel::RoughnessKernel;

#[derive(Clone, Debug)]
pub struct LandscapeParams {
    pub fs: f32,
    pub max_hist_cols: usize,
    pub gamma: f32,
    pub alpha: f32,
    pub roughness_kernel: RoughnessKernel,
    pub consonance_kernel: ConsonanceKernel,
    // --- Added: normalization params ---
    /// Exponent for subjective intensity (≈ specific loudness). Typical: 0.23
    pub loudness_exp: f32,
    /// Reference power for normalization. Tune to your signal scale.
    pub ref_power: f32,
    /// Leaky integration time constant [ms]. Typical: 60–120 ms.
    pub tau_ms: f32,
}

#[derive(Clone, Debug, Default)]
pub struct LandscapeFrame {
    pub fs: f32,
    pub freqs_hz: Vec<f32>,
    pub r_last: Vec<f32>,
    pub c_last: Vec<f32>,
    pub k_last: Vec<f32>,
    /// Preprocessed subjective intensity (after normalize()) on log2 bins.
    pub amps_last: Vec<f32>,
}

pub struct Landscape {
    nsgt_rt: RtNsgtKernelLog2,
    params: LandscapeParams,
    last_r: Vec<f32>,
    last_c: Vec<f32>,
    last_k: Vec<f32>,
    /// State of the leaky integrator per bin (subjective intensity domain).
    norm_state: Vec<f32>,
    /// Last preprocessed amplitudes (subjective intensity).
    amps_last: Vec<f32>,
}

impl Landscape {
    pub fn new(params: LandscapeParams, nsgt_rt: RtNsgtKernelLog2) -> Self {
        let n_ch = nsgt_rt.space().n_bins();
        Self {
            nsgt_rt,
            params,
            last_r: vec![0.0; n_ch],
            last_c: vec![0.0; n_ch],
            last_k: vec![0.0; n_ch],
            norm_state: vec![0.0; n_ch],
            amps_last: vec![0.0; n_ch],
        }
    }

    /// Map NSGT magnitude to subjective intensity (≈ specific loudness) and
    /// apply a first-order leaky integrator (attack/decay via single tau).
    /// This keeps values non-negative, compresses dynamic range, and stabilizes
    /// short-term fluctuations. Output aligns R/C scales better.
    fn normalize(&mut self, envelope: &[f32], dt_sec: f32) -> Vec<f32> {
        let eps = 0.0; //1e-12_f32;
        let exp = self.params.loudness_exp; //.max(0.01);
        let tau_s = (self.params.tau_ms.max(1.0)) * 1e-3;
        let a = (-dt_sec / tau_s).exp(); // leaky factor in [0,1)

        let mut out = vec![0.0f32; envelope.len()];
        for (i, &mag) in envelope.iter().enumerate() {
            // 1) magnitude → power
            let pow = mag * mag;
            // 2) subjective intensity (power law compression)
            let subj = ((pow / self.params.ref_power) + eps).powf(exp);
            // 3) leaky integration
            let y_prev = self.norm_state[i];
            let y = a * y_prev + (1.0 - a) * subj;
            self.norm_state[i] = y;
            out[i] = y;
        }
        out
    }

    /// Process one streaming frame (e.g. hop-size worth of samples)
    pub fn process_frame(&mut self, x_frame: &[f32]) -> LandscapeFrame {
        let fs = self.params.fs;

        // === 1. NSGT-RT analysis (streaming) ===
        let envelope: Vec<f32> = self.nsgt_rt.process_frame(x_frame);

        // === 2. Psychoacoustic normalization (subjective intensity + leaky) ===
        let dt_sec = (x_frame.len() as f32) / fs;
        let norm_env = self.normalize(&envelope, dt_sec);
        //let norm_env = envelope.clone();
        self.amps_last.clone_from(&norm_env); // keep the preprocessed view

        // Frequency space (log2 bins)
        let space = self.nsgt_rt.space();

        // === 3. potential R ===
        let (r, _r_total) = self
            .params
            .roughness_kernel
            .potential_r_from_log2_spectrum(&norm_env, space);

        // === 4. potential C ===
        let (c, _norm) = self
            .params
            .consonance_kernel
            .potential_c_from_log2_spectrum(&norm_env, space, self.params.gamma);

        // === 5. potential K (example: gate-like fusion) ===
        let k: Vec<f32> = r.iter().zip(&c).map(|(ri, ci)| ci * (1.0 - ri)).collect();

        // Update internal state
        self.last_r = r;
        self.last_c = c;
        self.last_k = k;

        LandscapeFrame {
            fs,
            freqs_hz: space.centers_hz.clone(),
            r_last: self.last_r.clone(),
            c_last: self.last_c.clone(),
            k_last: self.last_k.clone(),
            amps_last: self.amps_last.clone(), // preprocessed subjective intensity
        }
    }

    pub fn snapshot(&self) -> LandscapeFrame {
        LandscapeFrame {
            fs: self.params.fs,
            freqs_hz: self.nsgt_rt.space().centers_hz.clone(),
            r_last: self.last_r.clone(),
            c_last: self.last_c.clone(),
            k_last: self.last_k.clone(),
            amps_last: self.amps_last.clone(), // preprocessed subjective intensity
        }
    }

    pub fn params(&self) -> &LandscapeParams {
        &self.params
    }
}
