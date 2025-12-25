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
    /// Half-saturation point for roughness range compression (legacy).
    pub roughness_half: f32,
    /// Habituation integration time constant [s].
    pub habituation_tau: f32,
    /// Weight of habituation penalty in consonance.
    pub habituation_weight: f32,
    /// Maximum depth of habituation penalty.
    pub habituation_max_depth: f32,
    /// Roughness penalty weight in consonance01.
    pub consonance_roughness_weight: f32,

    /// Exponent for subjective intensity (≈ specific loudness). Typical: 0.23
    pub loudness_exp: f32,
    /// Reference power for normalization. Tune to your signal scale.
    pub ref_power: f32,
    /// Leaky integration time constant [ms]. Typical: 60–120 ms.
    pub tau_ms: f32,

    /// Roughness normalization constant (k).
    /// Controls the saturation curve: D_index = R / (R_total + k).
    pub roughness_k: f32,

    /// Reference f0 (Hz) for roughness shape normalization.
    pub roughness_ref_f0_hz: f32,
    /// ERB separation for the 2-peak reference stimulus.
    pub roughness_ref_sep_erb: f32,
    /// Mass split between the two reference peaks (peak A mass).
    pub roughness_ref_mass_split: f32,
    /// Epsilon for roughness normalization and mass checks.
    pub roughness_ref_eps: f32,
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
    pub roughness_shape_raw: Vec<f32>,
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
            roughness_shape_raw: vec![0.0; n],
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

    pub fn resize_to_space(&mut self, space: Log2Space) {
        let n = space.n_bins();
        self.space = space;
        self.roughness.resize(n, 0.0);
        self.roughness_shape_raw.resize(n, 0.0);
        self.roughness01.resize(n, 0.0);
        self.harmonicity.resize(n, 0.0);
        self.harmonicity01.resize(n, 0.0);
        self.consonance.resize(n, 0.0);
        self.consonance01.resize(n, 0.0);
        self.subjective_intensity.resize(n, 0.0);
        self.nsgt_power.resize(n, 0.0);
        self.habituation.resize(n, 0.0);
    }

    /// Signed consonance [-1, 1]. Prefer `evaluate_pitch01` for normalized usage.
    pub fn evaluate_pitch(&self, freq_hz: f32) -> f32 {
        self.sample_linear(&self.consonance, freq_hz)
    }

    /// Signed consonance [-1, 1]. Prefer `evaluate_pitch01_log2` for normalized usage.
    pub fn evaluate_pitch_log2(&self, log_freq: f32) -> f32 {
        self.sample_linear_log2(&self.consonance, log_freq)
    }

    pub fn evaluate_pitch01(&self, freq_hz: f32) -> f32 {
        self.sample_linear(&self.consonance01, freq_hz)
            .clamp(0.0, 1.0)
    }

    pub fn evaluate_pitch01_log2(&self, log_freq: f32) -> f32 {
        self.sample_linear_log2(&self.consonance01, log_freq)
            .clamp(0.0, 1.0)
    }

    pub fn consonance_at(&self, freq_hz: f32) -> f32 {
        self.evaluate_pitch(freq_hz)
    }

    pub fn consonance01_at(&self, freq_hz: f32) -> f32 {
        self.evaluate_pitch01(freq_hz)
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
        let w_hab = params.habituation_weight.max(0.0);
        let w_r = params.consonance_roughness_weight.max(0.0);
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
            let h = *self.harmonicity.get(i).unwrap_or(&0.0);
            let hab = *self.habituation.get(i).unwrap_or(&0.0);
            let h01 = h.clamp(0.0, 1.0);
            let r01 = self.roughness01[i].clamp(0.0, 1.0);
            let c_signed_raw = h01 - w_r * r01 - w_hab * hab;
            let c_signed = c_signed_raw.clamp(-1.0, 1.0);
            let c01 = (c_signed + 1.0) * 0.5;
            self.harmonicity01[i] = h01;
            self.consonance[i] = c_signed;
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
        v0 + (v1 - v0) * frac
    }
}

pub fn map_roughness01(r_norm: f32, r_half: f32) -> f32 {
    // Range compression helper (not a psychoacoustic normalization).
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
    use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
    use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};

    fn build_params(space: &Log2Space) -> LandscapeParams {
        LandscapeParams {
            fs: 48_000.0,
            max_hist_cols: 1,
            alpha: 0.0,
            roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005),
            harmonicity_kernel: HarmonicityKernel::new(space, HarmonicityParams::default()),
            roughness_scalar_mode: RoughnessScalarMode::Total,
            roughness_half: 0.1,
            habituation_tau: 1.0,
            habituation_weight: 0.2,
            habituation_max_depth: 1.0,
            consonance_roughness_weight: 0.5,
            loudness_exp: 1.0,
            ref_power: 1.0,
            tau_ms: 1.0,
            roughness_k: 1.0,
            roughness_ref_f0_hz: 1000.0,
            roughness_ref_sep_erb: 0.25,
            roughness_ref_mass_split: 0.5,
            roughness_ref_eps: 1e-12,
        }
    }

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
                let c_signed = (h - r).clamp(-1.0, 1.0);
                let c = ((c_signed + 1.0) * 0.5).clamp(0.0, 1.0);
                assert!(c >= 0.0 && c <= 1.0);
            }
        }
    }

    #[test]
    fn evaluate_pitch01_uses_consonance01() {
        let mut landscape = Landscape::new(Log2Space::new(100.0, 400.0, 12));
        landscape.consonance.fill(10.0);
        landscape.consonance01.fill(0.3);
        let val = landscape.evaluate_pitch01(200.0);
        assert!((val - 0.3).abs() < 1e-6, "val={val}");
    }

    #[test]
    fn evaluate_pitch01_is_clamped() {
        let mut landscape = Landscape::new(Log2Space::new(100.0, 400.0, 12));
        landscape.consonance01.fill(1.2);
        let val = landscape.evaluate_pitch01(200.0);
        assert!((val - 1.0).abs() < 1e-6, "val={val}");
    }

    #[test]
    fn consonance01_matches_formula_after_updates() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let params = build_params(&space);
        let mut landscape = Landscape::new(space);
        let n = landscape.roughness01.len();
        landscape.harmonicity = vec![0.25; n];
        landscape.roughness = vec![0.0; n];
        landscape.roughness01 = vec![0.4; n];
        landscape.habituation = vec![0.3; n];
        landscape.recompute_consonance(&params);

        let w_hab = params.habituation_weight;
        let w_r = params.consonance_roughness_weight;
        for i in 0..n {
            let h = landscape.harmonicity[i].clamp(0.0, 1.0);
            let r = landscape.roughness01[i].clamp(0.0, 1.0);
            let hab = landscape.habituation[i];
            let c_signed = (h - w_r * r - w_hab * hab).clamp(-1.0, 1.0);
            let c_pred = ((c_signed + 1.0) * 0.5).clamp(0.0, 1.0);
            let c = landscape.consonance01[i];
            assert!((c - c_pred).abs() < 1e-6, "i={i} c={c} c_pred={c_pred}");
        }
    }

    #[test]
    fn consonance01_independent_of_update_order() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let params = build_params(&space);
        let mut a = Landscape::new(space.clone());
        let n = a.roughness01.len();
        a.harmonicity = vec![0.75; n];
        a.roughness01 = vec![0.35; n];
        a.habituation = vec![0.1; n];
        a.recompute_consonance(&params);
        let c_a = a.consonance01.clone();

        let mut b = Landscape::new(space);
        b.roughness01 = vec![0.35; n];
        b.harmonicity = vec![0.75; n];
        b.habituation = vec![0.1; n];
        b.recompute_consonance(&params);
        for i in 0..n {
            let da = c_a[i];
            let db = b.consonance01[i];
            assert!((da - db).abs() < 1e-6, "i={i} da={da} db={db}");
        }
    }

    #[test]
    fn consonance01_decreases_linearly_with_roughness() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let mut params = build_params(&space);
        params.habituation_weight = 0.0;
        params.consonance_roughness_weight = 1.0;

        let mut landscape = Landscape::new(space);
        let n = landscape.roughness01.len();
        landscape.harmonicity = vec![1.0; n];
        landscape.roughness = vec![0.0; n];
        landscape.habituation = vec![0.0; n];

        landscape.roughness01 = vec![0.0; n];
        landscape.recompute_consonance(&params);
        assert!((landscape.consonance01[0] - 1.0).abs() < 1e-6);

        landscape.roughness01 = vec![0.5; n];
        landscape.recompute_consonance(&params);
        assert!((landscape.consonance01[0] - 0.75).abs() < 1e-6);

        landscape.roughness01 = vec![1.0; n];
        landscape.recompute_consonance(&params);
        assert!((landscape.consonance01[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn resize_to_space_keeps_lengths_and_allows_recompute() {
        let space_a = Log2Space::new(100.0, 400.0, 12);
        let space_b = Log2Space::new(80.0, 800.0, 24);
        let mut landscape = Landscape::new(space_a);
        landscape.harmonicity.fill(0.6);
        landscape.roughness01.fill(0.2);
        landscape.habituation.fill(0.1);

        landscape.resize_to_space(space_b.clone());
        let n = space_b.n_bins();
        assert_eq!(landscape.roughness.len(), n);
        assert_eq!(landscape.roughness01.len(), n);
        assert_eq!(landscape.harmonicity.len(), n);
        assert_eq!(landscape.harmonicity01.len(), n);
        assert_eq!(landscape.consonance.len(), n);
        assert_eq!(landscape.consonance01.len(), n);
        assert_eq!(landscape.subjective_intensity.len(), n);
        assert_eq!(landscape.nsgt_power.len(), n);
        assert_eq!(landscape.habituation.len(), n);

        let params = build_params(&space_b);
        landscape.recompute_consonance(&params);
    }
}

impl Default for Landscape {
    fn default() -> Self {
        Self::new(Log2Space::new(1.0, 2.0, 1))
    }
}
