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
    /// Scalar roughness summary used for normalization and diagnostics.
    pub roughness_scalar_mode: RoughnessScalarMode,
    /// Half-saturation point for roughness normalization (R_norm).
    pub roughness_half: f32,
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

#[derive(Clone, Copy, Debug)]
pub enum RoughnessScalarMode {
    Total,
    Max,
    P95,
}

/// Pure data snapshot for UI and agent evaluation.
#[derive(Clone, Debug)]
pub struct Landscape {
    pub space: Log2Space,
    pub roughness: Vec<f32>,
    pub roughness01: Vec<f32>,
    pub harmonicity: Vec<f32>,
    pub harmonicity01: Vec<f32>,
    pub consonance: Vec<f32>,
    pub consonance01: Vec<f32>,
    pub subjective_intensity: Vec<f32>,
    pub nsgt_power: Vec<f32>,
    pub habituation: Vec<f32>,
    pub roughness_total: f32,
    pub roughness_max: f32,
    pub roughness_p95: f32,
    pub roughness_scalar_raw: f32,
    pub roughness_norm: f32,
    pub roughness01_scalar: f32,
    pub loudness_mass: f32,
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
            roughness01: vec![0.0; n],
            harmonicity: vec![0.0; n],
            harmonicity01: vec![0.0; n],
            consonance: vec![0.0; n],
            consonance01: vec![0.0; n],
            subjective_intensity: vec![0.0; n],
            nsgt_power: vec![0.0; n],
            habituation: vec![0.0; n],
            roughness_total: 0.0,
            roughness_max: 0.0,
            roughness_p95: 0.0,
            roughness_scalar_raw: 0.0,
            roughness_norm: 0.0,
            roughness01_scalar: 0.0,
            loudness_mass: 0.0,
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
        // Current raw consonance uses roughness_total as the scalar normalizer.
        let k = params.roughness_k.max(1e-6);
        let denom_inv = 1.0 / (self.roughness_total + k);
        let w_hab = params.habituation_weight.max(0.0);
        if self.harmonicity01.len() != self.harmonicity.len() {
            self.harmonicity01 = vec![0.0; self.harmonicity.len()];
        }
        if self.consonance01.len() != self.consonance.len() {
            self.consonance01 = vec![0.0; self.consonance.len()];
        }
        if self.roughness01.len() != self.roughness.len() {
            self.roughness01 = vec![0.0; self.roughness.len()];
        }
        for i in 0..self.consonance.len() {
            let r = *self.roughness.get(i).unwrap_or(&0.0);
            let h = *self.harmonicity.get(i).unwrap_or(&0.0);
            let hab = *self.habituation.get(i).unwrap_or(&0.0);
            self.consonance[i] = h - r * denom_inv - w_hab * hab;
            let h01 = h.clamp(0.0, 1.0);
            let r01 = self.roughness01[i].clamp(0.0, 1.0);
            let c01 = h01 * (1.0 - r01) - w_hab * hab;
            self.harmonicity01[i] = h01;
            self.consonance01[i] = c01.clamp(0.0, 1.0);
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

pub fn map_roughness01(r_norm: f32, r_half: f32) -> f32 {
    let half = r_half.max(1e-12);
    let denom = r_norm + half;
    if denom <= 0.0 {
        0.0
    } else {
        (r_norm / denom).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roughness01_in_range_and_halfpoint() {
        let r_half = 0.2;
        let r0 = map_roughness01(0.0, r_half);
        let r_half_out = map_roughness01(r_half, r_half);
        let r_big = map_roughness01(10.0, r_half);
        assert!(r0 >= 0.0 && r0 <= 1.0);
        assert!((r_half_out - 0.5).abs() < 1e-6);
        assert!(r_big >= 0.0 && r_big <= 1.0);
    }

    #[test]
    fn roughness01_invariant_to_joint_scaling() {
        let r_half = 0.3;
        let r_raw = 2.0;
        let loudness = 5.0;
        let r_norm = r_raw / (loudness + 1e-12);
        let r01 = map_roughness01(r_norm, r_half);
        let scale = 4.0;
        let r_norm2 = (r_raw * scale) / (loudness * scale + 1e-12);
        let r012 = map_roughness01(r_norm2, r_half);
        assert!((r01 - r012).abs() < 1e-6);
    }

    #[test]
    fn c01_stays_in_range() {
        let h01 = [0.0f32, 0.5, 1.0];
        let r01 = [0.0f32, 0.4, 1.0];
        for &h in &h01 {
            for &r in &r01 {
                let c = (h * (1.0 - r)).clamp(0.0, 1.0);
                assert!(c >= 0.0 && c <= 1.0);
            }
        }
    }
}

impl Default for Landscape {
    fn default() -> Self {
        Self::new(Log2Space::new(1.0, 2.0, 1))
    }
}
