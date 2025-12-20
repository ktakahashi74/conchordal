//! core/landscape.rs — Data container for the psychoacoustic map (The Map).
//!
//! This module now holds only the data structures and lightweight helpers.
//! Processing lives in the Dorsal (rhythm), Roughness, and Harmonicity streams.

use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;

#[derive(Clone, Debug)]
pub struct LandscapeParams {
    pub fs: f32,
    pub max_hist_cols: usize,
    pub alpha: f32,
    pub roughness_kernel: crate::core::roughness_kernel::RoughnessKernel,
    pub harmonicity_kernel: crate::core::harmonicity_kernel::HarmonicityKernel,
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
    pub roughness_k: Option<f32>,
    pub habituation_weight: Option<f32>,
    pub habituation_tau: Option<f32>,
    pub habituation_max_depth: Option<f32>,
}

/// Pure data snapshot for UI and agent evaluation.
#[derive(Clone, Debug)]
pub struct Landscape {
    pub space: Log2Space,
    pub roughness: Vec<f32>,
    pub harmonicity: Vec<f32>,
    pub consonance: Vec<f32>,
    pub subjective_intensity: Vec<f32>,
    pub habituation: Vec<f32>,
    pub roughness_total: f32,
    pub rhythm: NeuralRhythms,
}

/// Alias retained for existing call sites.
pub type LandscapeFrame = Landscape;

impl Landscape {
    pub fn new(space: Log2Space) -> Self {
        let n = space.n_bins();
        Self {
            space,
            roughness: vec![0.0; n],
            harmonicity: vec![0.0; n],
            consonance: vec![0.0; n],
            subjective_intensity: vec![0.0; n],
            habituation: vec![0.0; n],
            roughness_total: 0.0,
            rhythm: NeuralRhythms::default(),
        }
    }

    pub fn evaluate_pitch(&self, freq_hz: f32) -> f32 {
        self.sample_linear(&self.consonance, freq_hz)
    }

    pub fn evaluate_pitch_log2(&self, log_freq: f32) -> f32 {
        self.sample_linear_log2(&self.consonance, log_freq)
    }

    pub fn consonance_at(&self, freq_hz: f32) -> f32 {
        self.evaluate_pitch(freq_hz)
    }

    pub fn get_crowding_at(&self, freq_hz: f32) -> f32 {
        self.sample_linear(&self.subjective_intensity, freq_hz)
    }

    pub fn get_habituation_at(&self, freq_hz: f32) -> f32 {
        self.sample_linear(&self.habituation, freq_hz)
    }

    pub fn get_audio_level_at(&self, freq_hz: f32) -> f32 {
        self.get_crowding_at(freq_hz)
    }

    pub fn get_spectral_satiety(&self, freq_hz: f32) -> f32 {
        if freq_hz <= 0.0 {
            return 0.0;
        }
        let amp = self.get_crowding_at(freq_hz).max(0.0);
        let k = 1000.0f32; // reference frequency for satiety scaling
        let weight = freq_hz / k;
        amp * weight
    }

    pub fn freq_bounds(&self) -> (f32, f32) {
        (self.space.fmin, self.space.fmax)
    }

    pub fn freq_bounds_log2(&self) -> (f32, f32) {
        (self.space.fmin.log2(), self.space.fmax.log2())
    }

    pub fn recompute_consonance(&mut self, params: &LandscapeParams) {
        let k = params.roughness_k.max(1e-6);
        let denom_inv = 1.0 / (self.roughness_total + k);
        let w_hab = params.habituation_weight.max(0.0);
        for i in 0..self.consonance.len() {
            let r = *self.roughness.get(i).unwrap_or(&0.0);
            let h = *self.harmonicity.get(i).unwrap_or(&0.0);
            let hab = *self.habituation.get(i).unwrap_or(&0.0);
            self.consonance[i] = h - r * denom_inv - w_hab * hab;
        }
    }

    fn sample_linear(&self, data: &[f32], freq_hz: f32) -> f32 {
        if data.is_empty() || freq_hz < self.space.fmin || freq_hz > self.space.fmax {
            return 0.0;
        }
        let l = freq_hz.log2();
        self.sample_linear_log2(data, l)
    }

    fn sample_linear_log2(&self, data: &[f32], log_freq: f32) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        let step = self.space.step();
        let base = self.space.centers_log2[0];
        let pos = (log_freq - base) / step;
        let idx = pos.floor() as usize;
        let frac = pos - pos.floor();
        let idx0 = idx.min(data.len().saturating_sub(1));
        let idx1 = (idx0 + 1).min(data.len().saturating_sub(1));
        let v0 = data.get(idx0).copied().unwrap_or(0.0);
        let v1 = data.get(idx1).copied().unwrap_or(v0);
        v0 + (v1 - v0) * frac as f32
    }
}

impl Default for Landscape {
    fn default() -> Self {
        Self::new(Log2Space::new(1.0, 2.0, 1))
    }
}
