//! core/landscape.rs — Landscape computed on log2(NSGT-RT) domain.
//!
//! Each incoming hop updates the potential roughness (R) and harmonicity (H)
//! using a real-time NSGT analyzer with leaky temporal integration.
//! Psychoacoustic normalization (subjective intensity + leaky smoothing)
//! runs before applying the R/H kernels.

use crate::core::harmonicity_kernel::HarmonicityKernel;
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::core::roughness_kernel::RoughnessKernel;

#[derive(Clone, Debug)]
pub struct LandscapeParams {
    pub fs: f32,
    pub max_hist_cols: usize,
    pub alpha: f32,
    pub roughness_kernel: RoughnessKernel,
    pub harmonicity_kernel: HarmonicityKernel,

    /// Exponent for subjective intensity (≈ specific loudness). Typical: 0.23
    pub loudness_exp: f32,
    /// Reference power for normalization. Tune to your signal scale.
    pub ref_power: f32,
    /// Leaky integration time constant [ms]. Typical: 60–120 ms.
    pub tau_ms: f32,

    /// Roughness normalization constant (k).
    /// Controls the saturation curve: D_index = R / (R_total + k).
    pub roughness_k: f32,
}

#[derive(Clone, Debug, Default)]
pub struct LandscapeFrame {
    pub fs: f32,
    pub freqs_hz: Vec<f32>,
    pub r_last: Vec<f32>,
    pub h_last: Vec<f32>,
    pub c_last: Vec<f32>,
    /// Preprocessed subjective intensity (after normalize()) on log2 bins.
    pub amps_last: Vec<f32>,
}

/// Maintains real-time psychoacoustic landscape (R/C/K) driven by NSGT-RT.
pub struct Landscape {
    nsgt_rt: RtNsgtKernelLog2,
    params: LandscapeParams,
    last_r: Vec<f32>,
    last_h: Vec<f32>,
    last_c: Vec<f32>,
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
            last_h: vec![0.0; n_ch],
            last_c: vec![0.0; n_ch],
            norm_state: vec![0.0; n_ch],
            amps_last: vec![0.0; n_ch],
        }
    }

    /// Convert magnitude envelope → subjective intensity (power law)
    /// and apply a leaky integrator to stabilize dynamics.
    fn normalize(&mut self, envelope: &[f32], dt_sec: f32) -> Vec<f32> {
        let exp = self.params.loudness_exp.max(0.01);
        let tau_s = (self.params.tau_ms.max(1.0)) * 1e-3;
        let a = (-dt_sec / tau_s).exp();
        let mut out = vec![0.0f32; envelope.len()];

        for (i, &mag) in envelope.iter().enumerate() {
            let pow = mag * mag;
            let subj = (pow / self.params.ref_power).powf(exp);
            let y_prev = self.norm_state[i];
            let y = a * y_prev + (1.0 - a) * subj;
            self.norm_state[i] = y;
            out[i] = y;
        }
        out
    }

    /// Process one hop of samples (length = hop). Real-time streaming entry point.
    pub fn process_frame(&mut self, x_frame: &[f32]) -> LandscapeFrame {
        let fs = self.params.fs;
        let dt_sec = (x_frame.len() as f32) / fs;

        // === 1. NSGT-RT analysis (streaming hop) ===
        let envelope: Vec<f32> = self.nsgt_rt.process_hop(x_frame).to_vec();

        // === 2. Psychoacoustic normalization ===
        let norm_env = self.normalize(&envelope, dt_sec);
        self.amps_last.clone_from(&norm_env);

        // === 3. Roughness potential R ===
        let space = self.nsgt_rt.space();
        let (r, r_total) = self
            .params
            .roughness_kernel
            .potential_r_from_log2_spectrum(&norm_env, space);

        println!("{r_total}");

        // === 4. Harmonicity potential C ===
        let (h, _norm) = self
            .params
            .harmonicity_kernel
            .potential_h_from_log2_spectrum(&norm_env, space);
        //.potential_h_with_freq_gate(&norm_env, space);

        // === 5. Combined potential K ===
        //let c: Vec<f32> = r.iter().zip(&h).map(|(ri, hi)| hi * (1.0 - ri)).collect();
        //        let c: Vec<f32> = r.iter().zip(&h).map(|(ri, hi)| hi - ri).collect();

        let k = self.params.roughness_k.max(1e-6);
        let denom_inv = 1.0 / (r_total + k);

        let c: Vec<f32> = r
            .iter()
            .zip(&h)
            .map(|(ri, hi)| {
                let d_index = ri * denom_inv;
                //hi * (1.0 - d_index)
                hi - d_index
            })
            .collect();

        // === 6. Update state ===
        self.last_r.clone_from(&r);
        self.last_h.clone_from(&h);
        self.last_c.clone_from(&c);

        LandscapeFrame {
            fs,
            freqs_hz: space.centers_hz.clone(),
            r_last: r,
            h_last: h,
            c_last: c,
            amps_last: self.amps_last.clone(),
        }
    }

    pub fn snapshot(&self) -> LandscapeFrame {
        LandscapeFrame {
            fs: self.params.fs,
            freqs_hz: self.nsgt_rt.space().centers_hz.clone(),
            r_last: self.last_r.clone(),
            h_last: self.last_h.clone(),
            c_last: self.last_c.clone(),
            amps_last: self.amps_last.clone(),
        }
    }

    pub fn params(&self) -> &LandscapeParams {
        &self.params
    }

    pub fn reset(&mut self) {
        self.nsgt_rt.reset();
        for x in &mut self.norm_state {
            *x = 0.0;
        }
        for x in &mut self.amps_last {
            *x = 0.0;
        }
        for x in &mut self.last_r {
            *x = 0.0;
        }
        for x in &mut self.last_h {
            *x = 0.0;
        }
        for x in &mut self.last_c {
            *x = 0.0;
        }
    }
}
