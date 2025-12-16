//! core/landscape.rs — Landscape computed on log2(NSGT-RT) domain.
//!
//! Each incoming hop updates the potential roughness (R) and harmonicity (H)
//! using a real-time NSGT analyzer with leaky temporal integration.
//! Psychoacoustic normalization (subjective intensity + leaky smoothing)
//! runs before applying the R/H kernels.

use crate::core::harmonicity_kernel::HarmonicityKernel;
use crate::core::log2space::Log2Space;
use crate::core::modulation::{ModulationBank, NeuralRhythms};
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::core::roughness_kernel::RoughnessKernel;

#[derive(Clone, Debug)]
pub struct LandscapeParams {
    pub fs: f32,
    pub max_hist_cols: usize,
    pub alpha: f32,
    pub roughness_kernel: RoughnessKernel,
    pub harmonicity_kernel: HarmonicityKernel,
    /// Habituation integration time constant [s].
    pub habituation_tau: f32,
    /// Weight of habituation penalty in consonance.
    pub habituation_weight: f32,
    /// Maximum depth of habituation penalty.
    pub habituation_max_depth: f32,

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

#[derive(Clone, Copy, Debug, Default)]
pub struct LandscapeUpdate {
    pub mirror: Option<f32>,
    pub limit: Option<u32>,
    pub habituation_weight: Option<f32>,
    pub habituation_tau: Option<f32>,
    pub habituation_max_depth: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct LandscapeFrame {
    pub fs: f32,
    pub space: Log2Space,
    pub r_last: Vec<f32>,
    pub h_last: Vec<f32>,
    pub c_last: Vec<f32>,
    pub rhythm: NeuralRhythms,
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
    /// Habituation state per bin (boredom integration).
    habituation_state: Vec<f32>,
    /// A-weighting gains per bin for loudness correction.
    loudness_weights: Vec<f32>,
    pub rhythm: NeuralRhythms,
    modulation_bank: Option<ModulationBank>,
    lp_200_state: f32,
    lp_2k_state: f32,
    modulation_vitality: f32,
}

impl Landscape {
    pub fn new(params: LandscapeParams, nsgt_rt: RtNsgtKernelLog2) -> Self {
        let n_ch = nsgt_rt.space().n_bins();
        let loudness_weights = nsgt_rt
            .space()
            .centers_hz
            .iter()
            .map(|&f| crate::core::utils::a_weighting_gain(f))
            .collect();
        Self {
            nsgt_rt,
            params,
            last_r: vec![0.0; n_ch],
            last_h: vec![0.0; n_ch],
            last_c: vec![0.0; n_ch],
            norm_state: vec![0.0; n_ch],
            habituation_state: vec![0.0; n_ch],
            amps_last: vec![0.0; n_ch],
            loudness_weights,
            rhythm: NeuralRhythms::default(),
            modulation_bank: None,
            lp_200_state: 0.0,
            lp_2k_state: 0.0,
            modulation_vitality: 0.1,
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
            let weighted_mag = mag * self.loudness_weights[i];
            let pow = weighted_mag * weighted_mag;
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
        self.update_spectrum(x_frame);
        self.compute_potentials();
        self.snapshot()
    }

    /// Update spectrum state (NSGT + normalization) without computing potentials.
    pub fn update_spectrum(&mut self, x_frame: &[f32]) {
        let fs = self.params.fs;
        let dt_sec = (x_frame.len() as f32) / fs;

        // === 1. NSGT-RT analysis (streaming hop) ===
        let envelope: Vec<f32> = self.nsgt_rt.process_hop(x_frame).to_vec();

        // === 2. Psychoacoustic normalization ===
        let norm_env = self.normalize(&envelope, dt_sec);
        self.amps_last.clone_from(&norm_env);
    }

    /// Compute potentials from the current normalized spectrum.
    pub fn compute_potentials(&mut self) {
        if self.amps_last.is_empty() {
            return;
        }
        let space = self.nsgt_rt.space();
        if self.habituation_state.len() != self.amps_last.len() {
            self.habituation_state.resize(self.amps_last.len(), 0.0);
        }

        // Roughness potential R
        let (r, r_total) = self
            .params
            .roughness_kernel
            .potential_r_from_log2_spectrum(&self.amps_last, space);

        // Harmonicity potential C
        let (h, _norm) = self
            .params
            .harmonicity_kernel
            .potential_h_from_log2_spectrum(&self.amps_last, space);

        // Habituation (boredom): leaky integration of recent amplitudes.
        let tau = self.params.habituation_tau.max(1e-3);
        let dt = self.nsgt_rt.dt();
        let a = (-dt / tau).exp();
        let max_depth = self.params.habituation_max_depth.max(0.0);
        for (state, &amp) in self.habituation_state.iter_mut().zip(&self.amps_last) {
            let y = a * *state + (1.0 - a) * amp;
            *state = y.min(max_depth);
        }

        // Combined potential K (here: C - D_index)
        let k = self.params.roughness_k.max(1e-6);
        let denom_inv = 1.0 / (r_total + k);
        let weight = self.params.habituation_weight.max(0.0);
        let c: Vec<f32> = r
            .iter()
            .zip(&h)
            .zip(&self.habituation_state)
            .map(|((ri, hi), hab)| {
                let d_index = ri * denom_inv;
                hi - d_index - weight * hab
            })
            .collect();

        self.last_r.clone_from(&r);
        self.last_h.clone_from(&h);
        self.last_c.clone_from(&c);
    }

    /// Process a precomputed spectral body (already in frequency domain).
    /// Expects `spectrum_body` to represent linear-frequency magnitudes (nfft/2+1).
    pub fn process_precomputed_spectrum(&mut self, spectrum_body: &[f32]) -> LandscapeFrame {
        let fs = self.params.fs;
        let n_bins_lin = spectrum_body.len();
        let nfft = (n_bins_lin.saturating_sub(1)) * 2;
        let df = if nfft > 0 { fs / nfft as f32 } else { 0.0 };

        let space = self.nsgt_rt.space().clone();
        let mut log_env = vec![0.0f32; space.n_bins()];

        for (i, &mag) in spectrum_body.iter().enumerate() {
            let f = i as f32 * df;
            if let Some(idx) = space.index_of_freq(f) {
                log_env[idx] += mag;
            }
        }

        let dt_sec = self.nsgt_rt.dt();

        // Psychoacoustic normalization
        let norm_env = self.normalize(&log_env, dt_sec);
        self.amps_last.clone_from(&norm_env);

        // Roughness
        let (r, r_total) = self
            .params
            .roughness_kernel
            .potential_r_from_log2_spectrum(&norm_env, &space);

        // Harmonicity
        let (h, _norm) = self
            .params
            .harmonicity_kernel
            .potential_h_from_log2_spectrum(&norm_env, &space);

        if self.habituation_state.len() != self.amps_last.len() {
            self.habituation_state.resize(self.amps_last.len(), 0.0);
        }

        // Habituation (boredom): leaky integration of recent amplitudes.
        let tau = self.params.habituation_tau.max(1e-3);
        let dt = self.nsgt_rt.dt();
        let a = (-dt / tau).exp();
        let max_depth = self.params.habituation_max_depth.max(0.0);
        for (state, &amp) in self.habituation_state.iter_mut().zip(&self.amps_last) {
            let y = a * *state + (1.0 - a) * amp;
            *state = y.min(max_depth);
        }

        let k = self.params.roughness_k.max(1e-6);
        let denom_inv = 1.0 / (r_total + k);

        let c: Vec<f32> = r
            .iter()
            .zip(&h)
            .zip(&self.habituation_state)
            .map(|((ri, hi), hab)| {
                let d_index = ri * denom_inv;
                hi - d_index - self.params.habituation_weight.max(0.0) * hab
            })
            .collect();

        self.last_r.clone_from(&r);
        self.last_h.clone_from(&h);
        self.last_c.clone_from(&c);

        LandscapeFrame {
            fs,
            space,
            r_last: r,
            h_last: h,
            c_last: c,
            rhythm: self.rhythm,
            amps_last: self.amps_last.clone(),
        }
    }

    pub fn snapshot(&self) -> LandscapeFrame {
        LandscapeFrame {
            fs: self.params.fs,
            space: self.nsgt_rt.space().clone(),
            r_last: self.last_r.clone(),
            h_last: self.last_h.clone(),
            c_last: self.last_c.clone(),
            rhythm: self.rhythm,
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
        for x in &mut self.habituation_state {
            *x = 0.0;
        }
        self.rhythm = NeuralRhythms::default();
        self.lp_200_state = 0.0;
        self.lp_2k_state = 0.0;
        self.modulation_bank = None;
        self.modulation_vitality = 0.1;
    }

    /// Lightweight rhythm extractor driven by time-domain audio.
    pub fn update_rhythm(&mut self, audio_chunk: &[f32]) {
        if audio_chunk.is_empty() {
            return;
        }
        let fs = self.params.fs;
        let hop_len = audio_chunk.len() as f32;
        if self.modulation_bank.is_none() {
            let env_rate = fs / hop_len.max(1.0);
            let mut bank = ModulationBank::new(env_rate);
            bank.set_vitality(self.modulation_vitality);
            self.modulation_bank = Some(bank);
        }

        let dt = 1.0 / fs;
        let alpha_200 = 1.0 - (-2.0 * std::f32::consts::PI * 200.0 * dt).exp();
        let alpha_2k = 1.0 - (-2.0 * std::f32::consts::PI * 2000.0 * dt).exp();

        let mut energy_sum = 0.0f32;
        for &s in audio_chunk {
            let x2 = s * s;
            energy_sum += x2;
            // Lowpass at 200 Hz on energy
            self.lp_200_state += alpha_200 * (x2 - self.lp_200_state);
            // Lowpass at 2 kHz on energy
            self.lp_2k_state += alpha_2k * (x2 - self.lp_2k_state);
        }

        let mean_energy = energy_sum / hop_len.max(1.0);
        let low_energy = self.lp_200_state.max(0.0);
        let mid_energy = (self.lp_2k_state - self.lp_200_state).max(0.0);
        let high_energy = (mean_energy - self.lp_2k_state).max(0.0);

        let low = low_energy.sqrt();
        let mid = mid_energy.sqrt();
        let high = high_energy.sqrt();

        if let Some(bank) = &mut self.modulation_bank {
            self.rhythm = bank.update(low, mid, high);
        }
    }

    pub fn consonance_at(&self, freq_hz: f32) -> f32 {
        if let Some(idx) = self.nsgt_rt.space().index_of_freq(freq_hz) {
            self.last_c.get(idx).copied().unwrap_or(0.0)
        } else {
            0.0
        }
    }

    /// Inject externally computed potentials (e.g., from analysis thread).
    pub fn apply_frame(&mut self, frame: &LandscapeFrame) {
        self.last_r.clone_from(&frame.r_last);
        self.last_h.clone_from(&frame.h_last);
        self.last_c.clone_from(&frame.c_last);
        self.amps_last.clone_from(&frame.amps_last);
        if self.habituation_state.len() != self.amps_last.len() {
            self.habituation_state.resize(self.amps_last.len(), 0.0);
        }
    }

    pub fn set_vitality(&mut self, v: f32) {
        self.modulation_vitality = v;
        if let Some(bank) = &mut self.modulation_bank {
            bank.set_vitality(v);
        }
    }

    pub fn set_roughness_k(&mut self, k: f32) {
        self.params.roughness_k = k.max(1e-6);
    }

    pub fn update_harmonicity_params(&mut self, mirror: Option<f32>, limit: Option<u32>) {
        let mut params = self.params.harmonicity_kernel.params;
        if let Some(m) = mirror {
            params.mirror_weight = m;
        }
        if let Some(l) = limit {
            params.param_limit = l;
        }
        let space = self.nsgt_rt.space().clone();
        self.params.harmonicity_kernel = HarmonicityKernel::new(&space, params);
    }

    pub fn update_habituation_params(&mut self, weight: f32, tau: f32, max_depth: f32) {
        self.params.habituation_weight = weight.max(0.0);
        self.params.habituation_tau = tau.max(1e-3);
        self.params.habituation_max_depth = max_depth.max(0.0);
    }

    pub fn apply_update(&mut self, upd: LandscapeUpdate) {
        if upd.mirror.is_some() || upd.limit.is_some() {
            self.update_harmonicity_params(upd.mirror, upd.limit);
        }
        if upd.habituation_weight.is_some()
            || upd.habituation_tau.is_some()
            || upd.habituation_max_depth.is_some()
        {
            let w = upd
                .habituation_weight
                .unwrap_or(self.params.habituation_weight);
            let tau = upd.habituation_tau.unwrap_or(self.params.habituation_tau);
            let max_d = upd
                .habituation_max_depth
                .unwrap_or(self.params.habituation_max_depth);
            self.update_habituation_params(w, tau, max_d);
        }
    }

    pub fn evaluate_pitch(&self, freq_hz: f32) -> f32 {
        if self.last_c.is_empty() {
            return -2.0;
        }
        let space = self.nsgt_rt.space();
        if freq_hz < space.fmin || freq_hz > space.fmax {
            return -2.0;
        }
        let l = freq_hz.log2();
        let step = space.step();
        let base = space.centers_log2[0];
        let pos = (l - base) / step;
        let idx = pos.floor() as usize;
        let frac = pos - pos.floor();
        let idx0 = idx.min(self.last_c.len().saturating_sub(1));
        let idx1 = (idx0 + 1).min(self.last_c.len().saturating_sub(1));
        let c0 = self.last_c.get(idx0).copied().unwrap_or(0.0);
        let c1 = self.last_c.get(idx1).copied().unwrap_or(c0);
        c0 + (c1 - c0) * frac as f32
    }

    pub fn freq_bounds(&self) -> (f32, f32) {
        let space = self.nsgt_rt.space();
        (space.fmin, space.fmax)
    }
}

impl Default for LandscapeFrame {
    fn default() -> Self {
        let space = Log2Space::new(1.0, 2.0, 1);
        let n = space.n_bins();
        Self {
            fs: 0.0,
            space,
            r_last: vec![0.0; n],
            h_last: vec![0.0; n],
            c_last: vec![0.0; n],
            rhythm: NeuralRhythms::default(),
            amps_last: vec![0.0; n],
        }
    }
}
