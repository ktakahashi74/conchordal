//! core/landscape.rs — Landscape computed on a log2-frequency domain.
//!
//! This module intentionally separates the sources of the two perceptual axes:
//! - Roughness (R): derived from the actual audio input via NSGT-RT.
//! - Harmonicity (H): derived from an externally provided "spectrum body"
//!   (e.g. collective agent intent provided by `AnalysisWorker`).
//!
//! The `LandscapeFrame` snapshot can therefore contain an R value coming from
//! the latest audio hop and an H value coming from the latest body update.

use crate::core::harmonicity_kernel::HarmonicityKernel;
use crate::core::log2space::Log2Space;
use crate::core::modulation::{NeuralRhythms, RhythmDynamics};
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

/// DTO for UI visualization.
#[derive(Clone, Debug)]
pub struct LandscapeFrame {
    pub fs: f32,
    pub space: Log2Space,
    /// Roughness profile derived from the audio stream.
    pub r_scan: Vec<f32>,
    /// Harmonicity profile derived from the agent/body stream.
    pub h_scan: Vec<f32>,
    /// Integrated consonance profile (H ⊖ R ⊖ habituation).
    pub c_scan: Vec<f32>,
    pub rhythm: NeuralRhythms,
    /// Audio-derived spectrum used for visualization.
    pub amps: Vec<f32>,
}

/// Maintains real-time psychoacoustic landscape (R/C/K) driven by NSGT-RT.
pub struct Landscape {
    nsgt_rt: RtNsgtKernelLog2,
    params: LandscapeParams,
    // === Stream A: Audio reality ===
    /// Roughness profile derived from the audio stream.
    r_scan: Vec<f32>,
    /// Audio-derived (psychoacoustically normalized) spectrum.
    pub amps_audio: Vec<f32>,
    /// Leaky integration state for audio loudness normalization.
    norm_state_audio: Vec<f32>,
    /// Habituation state per bin (boredom integration), driven by audio.
    habituation_state: Vec<f32>,

    // === Stream B: Agent intention (body) ===
    /// Harmonicity profile derived from the agent/body stream.
    h_scan: Vec<f32>,
    /// Body-derived spectrum mapped to log2 bins (no temporal smoothing).
    pub amps_body: Vec<f32>,

    // === Integrated ===
    /// Integrated consonance profile.
    c_scan: Vec<f32>,

    // Cache: local ERB integration weights for summations.
    du_cache: Vec<f32>,
    /// A-weighting gains per bin for loudness correction.
    loudness_weights: Vec<f32>,
    pub rhythm: NeuralRhythms,
    rhythm_dynamics: RhythmDynamics,
    lp_200_state: f32,
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

        let erb_coords: Vec<f32> = nsgt_rt
            .space()
            .centers_hz
            .iter()
            .map(|&f| crate::core::erb::hz_to_erb(f))
            .collect();
        let du_cache = local_du_from_grid(&erb_coords);
        Self {
            nsgt_rt,
            params,
            r_scan: vec![0.0; n_ch],
            amps_audio: vec![0.0; n_ch],
            norm_state_audio: vec![0.0; n_ch],
            habituation_state: vec![0.0; n_ch],
            h_scan: vec![0.0; n_ch],
            amps_body: vec![0.0; n_ch],
            c_scan: vec![0.0; n_ch],
            du_cache,
            loudness_weights,
            rhythm: NeuralRhythms::default(),
            rhythm_dynamics: RhythmDynamics::default(),
            lp_200_state: 0.0,
        }
    }

    /// Backwards-compatible alias for the audio thread entry point.
    pub fn process_frame(&mut self, x_frame: &[f32]) -> LandscapeFrame {
        self.process_audio_frame(x_frame)
    }

    /// Stream A: audio thread entry point.
    ///
    /// Runs NSGT-RT → loudness normalization → roughness (R) → habituation → consonance (C).
    /// Harmonicity (H) is not updated here; it is provided asynchronously via `apply_body_analysis`.
    pub fn process_audio_frame(&mut self, audio_chunk: &[f32]) -> LandscapeFrame {
        let dt_sec = (audio_chunk.len() as f32) / self.params.fs.max(1.0);

        // 1. NSGT-RT analysis (streaming hop)
        let raw_env: Vec<f32> = self.nsgt_rt.process_hop(audio_chunk).to_vec();

        // 2. Psychoacoustic normalization (audio)
        self.normalize_audio(&raw_env, dt_sec);

        // 3. Roughness profile from audio
        let (r, _) = self
            .params
            .roughness_kernel
            .potential_r_from_log2_spectrum(&self.amps_audio, self.nsgt_rt.space());
        self.r_scan.clone_from(&r);

        // 4. Habituation from audio
        self.update_habituation(dt_sec);

        // 5. Integrate consonance from current R/H
        self.update_consonance();

        self.snapshot()
    }

    // ============================================================================================
    // Stream B: Body analysis integration (async)
    // ============================================================================================

    /// Called when the analysis worker returns a new H scan.
    pub fn apply_body_analysis(&mut self, h_scan: Vec<f32>, body_spectrum: Vec<f32>) {
        if h_scan.len() == self.h_scan.len() {
            self.h_scan = h_scan;
        }
        if body_spectrum.len() == self.amps_body.len() {
            self.amps_body = body_spectrum;
        }
        self.update_consonance();
    }

    /// Helper for the analysis worker to calculate H without affecting the main thread state.
    /// Returns `(h_scan, amps_body_log2)`.
    pub fn calculate_h_from_linear(&self, spectrum_lin: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let fs = self.params.fs;
        let n_bins_lin = spectrum_lin.len();
        let nfft = (n_bins_lin.saturating_sub(1)) * 2;
        let df = if nfft > 0 { fs / nfft as f32 } else { 0.0 };

        let space = self.nsgt_rt.space();
        let mut amps_log = vec![0.0f32; space.n_bins()];

        // Map linear → log2 bins
        for (i, &mag) in spectrum_lin.iter().enumerate() {
            let f = i as f32 * df;
            if let Some(idx) = space.index_of_freq(f) {
                amps_log[idx] += mag;
            }
        }

        let (h, _) = self
            .params
            .harmonicity_kernel
            .potential_h_from_log2_spectrum(&amps_log, space);

        (h, amps_log)
    }

    pub fn snapshot(&self) -> LandscapeFrame {
        LandscapeFrame {
            fs: self.params.fs,
            space: self.nsgt_rt.space().clone(),
            r_scan: self.r_scan.clone(),
            h_scan: self.h_scan.clone(),
            c_scan: self.c_scan.clone(),
            rhythm: self.rhythm,
            amps: self.amps_audio.clone(),
        }
    }

    pub fn params(&self) -> &LandscapeParams {
        &self.params
    }

    pub fn reset(&mut self) {
        self.nsgt_rt.reset();
        self.r_scan.fill(0.0);
        self.h_scan.fill(0.0);
        self.c_scan.fill(0.0);
        self.amps_audio.fill(0.0);
        self.amps_body.fill(0.0);
        self.norm_state_audio.fill(0.0);
        self.habituation_state.fill(0.0);
        self.rhythm = NeuralRhythms::default();
        self.rhythm_dynamics = RhythmDynamics::default();
        self.lp_200_state = 0.0;
    }

    /// Lightweight rhythm extractor driven by time-domain audio.
    pub fn update_rhythm(&mut self, audio_chunk: &[f32]) {
        if audio_chunk.is_empty() {
            return;
        }
        let fs = self.params.fs;
        let hop_len = audio_chunk.len() as f32;
        let dt = hop_len / fs.max(1.0);
        let alpha_200 = 1.0 - (-2.0 * std::f32::consts::PI * 200.0 * (1.0 / fs)).exp();

        let mut energy_sum = 0.0f32;
        for &s in audio_chunk {
            let x2 = s * s;
            energy_sum += x2;
            self.lp_200_state += alpha_200 * (x2 - self.lp_200_state);
        }

        let current_energy = energy_sum / hop_len.max(1.0);
        let flux = (current_energy - self.lp_200_state).max(0.0);

        self.rhythm = self.rhythm_dynamics.update(dt, flux);
    }

    pub fn consonance_at(&self, freq_hz: f32) -> f32 {
        if let Some(idx) = self.nsgt_rt.space().index_of_freq(freq_hz) {
            self.c_scan.get(idx).copied().unwrap_or(0.0)
        } else {
            0.0
        }
    }

    /// Inject a full snapshot (used by tests).
    pub fn apply_frame(&mut self, frame: &LandscapeFrame) {
        self.r_scan.clone_from(&frame.r_scan);
        self.h_scan.clone_from(&frame.h_scan);
        self.c_scan.clone_from(&frame.c_scan);
        self.amps_audio.clone_from(&frame.amps);
    }

    pub fn set_vitality(&mut self, v: f32) {
        self.rhythm_dynamics.set_vitality(v);
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
        if self.c_scan.is_empty() {
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
        let idx0 = idx.min(self.c_scan.len().saturating_sub(1));
        let idx1 = (idx0 + 1).min(self.c_scan.len().saturating_sub(1));
        let c0 = self.c_scan.get(idx0).copied().unwrap_or(0.0);
        let c1 = self.c_scan.get(idx1).copied().unwrap_or(c0);
        c0 + (c1 - c0) * frac as f32
    }

    pub fn freq_bounds(&self) -> (f32, f32) {
        let space = self.nsgt_rt.space();
        (space.fmin, space.fmax)
    }

    pub fn get_habituation_at(&self, freq_hz: f32) -> f32 {
        sample_linear(&self.habituation_state, self.nsgt_rt.space(), freq_hz)
    }

    pub fn get_crowding_at(&self, freq_hz: f32) -> f32 {
        sample_linear(&self.amps_body, self.nsgt_rt.space(), freq_hz)
    }

    pub fn get_spectral_satiety(&self, freq_hz: f32) -> f32 {
        if freq_hz <= 0.0 {
            return 0.0;
        }
        // Spectral tilt pressure: higher bands reach "full" with less energy (1/f-like).
        let amp = sample_linear(&self.amps_audio, self.nsgt_rt.space(), freq_hz).max(0.0);
        let k = 1000.0f32; // reference frequency for satiety scaling
        let weight = freq_hz / k;
        amp * weight
    }
}

impl Default for LandscapeFrame {
    fn default() -> Self {
        let space = Log2Space::new(1.0, 2.0, 1);
        let n = space.n_bins();
        Self {
            fs: 0.0,
            space,
            r_scan: vec![0.0; n],
            h_scan: vec![0.0; n],
            c_scan: vec![0.0; n],
            rhythm: NeuralRhythms::default(),
            amps: vec![0.0; n],
        }
    }
}

impl Landscape {
    fn normalize_audio(&mut self, raw_env: &[f32], dt_sec: f32) {
        let exp = self.params.loudness_exp.max(0.01);
        let tau_s = (self.params.tau_ms.max(1.0)) * 1e-3;
        let a = (-dt_sec / tau_s).exp();

        for (i, &mag) in raw_env.iter().enumerate() {
            let weighted_mag = mag * self.loudness_weights[i];
            let pow = weighted_mag * weighted_mag;
            let subj = (pow / self.params.ref_power).powf(exp);
            let prev = self.norm_state_audio[i];
            let next = a * prev + (1.0 - a) * subj;
            self.norm_state_audio[i] = next;
            self.amps_audio[i] = next;
        }
    }

    fn update_habituation(&mut self, dt_sec: f32) {
        let tau = self.params.habituation_tau.max(1e-3);
        let a = (-dt_sec / tau).exp();
        let max_depth = self.params.habituation_max_depth.max(0.0);

        for (state, &amp) in self.habituation_state.iter_mut().zip(&self.amps_audio) {
            let next = a * *state + (1.0 - a) * amp;
            *state = next.min(max_depth);
        }
    }

    fn update_consonance(&mut self) {
        // C = H - D_index - habituation
        // D_index = R / (R_total + k)
        if self.r_scan.is_empty() || self.h_scan.is_empty() {
            return;
        }

        let r_total: f32 = self
            .r_scan
            .iter()
            .zip(&self.du_cache)
            .map(|(r, du)| r * du)
            .sum();

        let k = self.params.roughness_k.max(1e-6);
        let denom_inv = 1.0 / (r_total + k);
        let w_hab = self.params.habituation_weight.max(0.0);

        for i in 0..self.c_scan.len() {
            let r_val = self.r_scan[i];
            let h_val = self.h_scan[i];
            let hab = self.habituation_state[i];
            let d_index = r_val * denom_inv;
            self.c_scan[i] = h_val - d_index - w_hab * hab;
        }
    }
}

fn sample_linear(data: &[f32], space: &Log2Space, freq_hz: f32) -> f32 {
    if data.is_empty() || freq_hz < space.fmin || freq_hz > space.fmax {
        return 0.0;
    }
    let l = freq_hz.log2();
    let step = space.step();
    let base = space.centers_log2[0];
    let pos = (l - base) / step;
    let idx = pos.floor() as usize;
    let frac = pos - pos.floor();
    let idx0 = idx.min(data.len().saturating_sub(1));
    let idx1 = (idx0 + 1).min(data.len().saturating_sub(1));
    let v0 = data.get(idx0).copied().unwrap_or(0.0);
    let v1 = data.get(idx1).copied().unwrap_or(v0);
    v0 + (v1 - v0) * frac as f32
}

fn local_du_from_grid(erb: &[f32]) -> Vec<f32> {
    let n = erb.len();
    let mut du = vec![0.0; n];
    if n < 2 {
        return du;
    }
    du[0] = (erb[1] - erb[0]).max(0.0);
    du[n - 1] = (erb[n - 1] - erb[n - 2]).max(0.0);
    for i in 1..n - 1 {
        du[i] = 0.5 * (erb[i + 1] - erb[i - 1]).max(0.0);
    }
    du
}
