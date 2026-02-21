//! core/landscape.rs — Data container for the psychoacoustic map (The Map).
//!
//! This module now holds only the data structures and lightweight helpers.
//! Processing lives in the Dorsal (rhythm) and Roughness/Harmonicity streams.

use crate::core::consonance_kernel::{ConsonanceKernel, ConsonanceRepresentationParams};
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;

#[derive(Clone, Debug)]
pub struct LandscapeParams {
    pub fs: f32,
    pub max_hist_cols: usize,
    pub roughness_kernel: crate::core::roughness_kernel::RoughnessKernel,
    pub harmonicity_kernel: crate::core::harmonicity_kernel::HarmonicityKernel,
    pub consonance_kernel: ConsonanceKernel,
    pub consonance_representation: ConsonanceRepresentationParams,

    /// Scalar roughness summary used for normalization and diagnostics.
    pub roughness_scalar_mode: RoughnessScalarMode,
    /// Half-saturation point for roughness range compression (legacy helper).
    pub roughness_half: f32,

    /// Exponent for subjective intensity (≈ specific loudness). Typical: 0.23
    pub loudness_exp: f32,
    /// Reference power for normalization. Tune to your signal scale.
    pub ref_power: f32,
    /// Leaky integration time constant [ms]. Typical: 60–120 ms.
    pub tau_ms: f32,

    /// Roughness normalization constant (k) for physiological saturation.
    /// Reference-normalized ratio x=1 maps to roughness01=1/(1+k).
    /// Larger k reduces roughness01 for the same reference ratio.
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
    pub roughness_k: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
pub enum RoughnessScalarMode {
    Total,
    Max,
    P95,
}

/// Pure data snapshot for UI and agent evaluation.
/// Values here are perceptual (`perc_*`) unless stated otherwise.
#[derive(Clone, Debug)]
pub struct Landscape {
    pub space: Log2Space,
    /// perc_potential_R over log2 frequency.
    pub roughness: Vec<f32>,
    /// Raw shape for perc_potential_R normalization.
    pub roughness_shape_raw: Vec<f32>,
    /// Normalized perc_potential_R [0, 1].
    pub roughness01: Vec<f32>,
    /// perc_potential_H over log2 frequency.
    pub harmonicity: Vec<f32>,
    /// Root-binding path (Path A) before blend.
    pub harmonicity_path_a: Vec<f32>,
    /// Overtone/Ceiling-binding path (Path B) before blend.
    pub harmonicity_path_b: Vec<f32>,
    /// Normalized perc_potential_H [0, 1].
    pub harmonicity01: Vec<f32>,
    /// Layer 1: kernel output (unbounded).
    pub consonance_score: Vec<f32>,
    /// Layer 2: normalized level in [0, 1].
    pub consonance_level01: Vec<f32>,
    /// Layer 2: non-negative pre-normalized weight.
    pub consonance_weight: Vec<f32>,
    /// Layer 2: normalized density (PMF).
    pub consonance_density: Vec<f32>,
    /// Layer 2: minimization energy.
    pub consonance_energy: Vec<f32>,
    pub subjective_intensity: Vec<f32>,
    pub nsgt_power: Vec<f32>,
    /// perc_state_R (summary statistics).
    pub roughness_total: f32,
    pub roughness_max: f32,
    pub roughness_p95: f32,
    pub roughness_scalar_raw: f32,
    pub roughness_norm: f32,
    pub roughness01_scalar: f32,
    pub loudness_mass: f32,
    pub root_affinity: f32,
    pub overtone_affinity: f32,
    pub binding_strength: f32,
    pub harmonic_tilt: f32,
    pub harmonicity_mirror_weight: f32,
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
            harmonicity_path_a: vec![0.0; n],
            harmonicity_path_b: vec![0.0; n],
            harmonicity01: vec![0.0; n],
            consonance_score: vec![0.0; n],
            consonance_level01: vec![0.0; n],
            consonance_weight: vec![0.0; n],
            consonance_density: vec![0.0; n],
            consonance_energy: vec![0.0; n],
            subjective_intensity: vec![0.0; n],
            nsgt_power: vec![0.0; n],
            roughness_total: 0.0,
            roughness_max: 0.0,
            roughness_p95: 0.0,
            roughness_scalar_raw: 0.0,
            roughness_norm: 0.0,
            roughness01_scalar: 0.0,
            loudness_mass: 0.0,
            root_affinity: 0.0,
            overtone_affinity: 0.0,
            binding_strength: 0.0,
            harmonic_tilt: 0.0,
            harmonicity_mirror_weight: 0.0,
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
        self.harmonicity_path_a.resize(n, 0.0);
        self.harmonicity_path_b.resize(n, 0.0);
        self.harmonicity01.resize(n, 0.0);
        self.consonance_score.resize(n, 0.0);
        self.consonance_level01.resize(n, 0.0);
        self.consonance_weight.resize(n, 0.0);
        self.consonance_density.resize(n, 0.0);
        self.consonance_energy.resize(n, 0.0);
        self.subjective_intensity.resize(n, 0.0);
        self.nsgt_power.resize(n, 0.0);
    }

    fn assert_scan_lengths(&self) {
        self.space
            .assert_scan_len_named(&self.roughness, "roughness");
        self.space
            .assert_scan_len_named(&self.roughness_shape_raw, "roughness_shape_raw");
        self.space
            .assert_scan_len_named(&self.roughness01, "roughness01");
        self.space
            .assert_scan_len_named(&self.harmonicity, "harmonicity");
        self.space
            .assert_scan_len_named(&self.harmonicity_path_a, "harmonicity_path_a");
        self.space
            .assert_scan_len_named(&self.harmonicity_path_b, "harmonicity_path_b");
        self.space
            .assert_scan_len_named(&self.harmonicity01, "harmonicity01");
        self.space
            .assert_scan_len_named(&self.consonance_score, "consonance_score");
        self.space
            .assert_scan_len_named(&self.consonance_level01, "consonance_level01");
        self.space
            .assert_scan_len_named(&self.consonance_weight, "consonance_weight");
        self.space
            .assert_scan_len_named(&self.consonance_density, "consonance_density");
        self.space
            .assert_scan_len_named(&self.consonance_energy, "consonance_energy");
        self.space
            .assert_scan_len_named(&self.subjective_intensity, "subjective_intensity");
        self.space
            .assert_scan_len_named(&self.nsgt_power, "nsgt_power");
    }

    /// Layer 1 score. Prefer `evaluate_pitch_level01` for normalized usage.
    pub fn evaluate_pitch_score(&self, freq_hz: f32) -> f32 {
        self.assert_scan_lengths();
        self.sample_linear(&self.consonance_score, freq_hz)
    }

    /// Layer 1 score. Prefer `evaluate_pitch_level01_log2` for normalized usage.
    pub fn evaluate_pitch_score_log2(&self, log_freq: f32) -> f32 {
        self.assert_scan_lengths();
        self.sample_linear_log2(&self.consonance_score, log_freq)
    }

    pub fn evaluate_pitch_level01(&self, freq_hz: f32) -> f32 {
        self.assert_scan_lengths();
        self.sample_linear(&self.consonance_level01, freq_hz)
            .clamp(0.0, 1.0)
    }

    pub fn evaluate_pitch_level01_log2(&self, log_freq: f32) -> f32 {
        self.assert_scan_lengths();
        self.sample_linear_log2(&self.consonance_level01, log_freq)
            .clamp(0.0, 1.0)
    }

    pub fn consonance_score_at(&self, freq_hz: f32) -> f32 {
        self.assert_scan_lengths();
        self.evaluate_pitch_score(freq_hz)
    }

    pub fn consonance_level01_at(&self, freq_hz: f32) -> f32 {
        self.assert_scan_lengths();
        self.evaluate_pitch_level01(freq_hz)
    }

    pub fn freq_bounds(&self) -> (f32, f32) {
        (self.space.fmin, self.space.fmax)
    }

    pub fn freq_bounds_log2(&self) -> (f32, f32) {
        (self.space.fmin.log2(), self.space.fmax.log2())
    }

    pub fn recompute_consonance(&mut self, params: &LandscapeParams) {
        self.assert_scan_lengths();

        if self.harmonicity01.len() != self.harmonicity.len() {
            self.harmonicity01 = vec![0.0; self.harmonicity.len()];
        }
        if self.roughness01.len() != self.roughness.len() {
            self.roughness01 = vec![0.0; self.roughness.len()];
        }

        let n = self.consonance_score.len();
        if self.consonance_level01.len() != n {
            self.consonance_level01 = vec![0.0; n];
        }
        if self.consonance_weight.len() != n {
            self.consonance_weight = vec![0.0; n];
        }
        if self.consonance_density.len() != n {
            self.consonance_density = vec![0.0; n];
        }
        if self.consonance_energy.len() != n {
            self.consonance_energy = vec![0.0; n];
        }

        let perc_h_pot_scan = &self.harmonicity;
        crate::core::psycho_state::h_pot_scan_to_h_state01_scan(
            perc_h_pot_scan,
            1.0,
            &mut self.harmonicity01,
        );

        debug_assert_eq!(self.harmonicity01.len(), n);
        debug_assert_eq!(self.roughness01.len(), n);

        for i in 0..n {
            let h01 = self.harmonicity01[i].clamp(0.0, 1.0);
            let r01 = self.roughness01[i].clamp(0.0, 1.0);
            let score = params.consonance_kernel.score(h01, r01);
            self.consonance_score[i] = score;
            self.consonance_level01[i] = params.consonance_representation.level01(score);
            self.consonance_weight[i] = params.consonance_representation.weight(score).max(0.0);
            self.consonance_energy[i] = params.consonance_representation.energy(score);
        }
        params
            .consonance_representation
            .normalize_density(&self.consonance_weight, &mut self.consonance_density);
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
    use crate::core::consonance_kernel::{ConsonanceKernel, ConsonanceRepresentationParams};
    use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
    use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};

    fn build_params(space: &Log2Space) -> LandscapeParams {
        LandscapeParams {
            fs: 48_000.0,
            max_hist_cols: 1,
            roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005),
            harmonicity_kernel: HarmonicityKernel::new(space, HarmonicityParams::default()),
            consonance_kernel: ConsonanceKernel::default(),
            consonance_representation: ConsonanceRepresentationParams::default(),
            roughness_scalar_mode: RoughnessScalarMode::Total,
            roughness_half: 0.1,
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
        assert!((0.0..=1.0).contains(&r0));
        assert!((r_half_out - 0.5).abs() < 1e-6);
        assert!((0.0..=1.0).contains(&r_big));
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
    fn consonance_level01_stays_in_range() {
        let params = build_params(&Log2Space::new(100.0, 400.0, 12));
        let h01 = [0.0f32, 0.5, 1.0];
        let r01 = [0.0f32, 0.4, 1.0];
        for &h in &h01 {
            for &r in &r01 {
                let score = params.consonance_kernel.score(h, r);
                let level = params.consonance_representation.level01(score);
                assert!((0.0..=1.0).contains(&level));
            }
        }
    }

    #[test]
    fn evaluate_pitch_level01_uses_consonance_level01() {
        let mut landscape = Landscape::new(Log2Space::new(100.0, 400.0, 12));
        landscape.consonance_score.fill(10.0);
        landscape.consonance_level01.fill(0.3);
        let val = landscape.evaluate_pitch_level01(200.0);
        assert!((val - 0.3).abs() < 1e-6, "val={val}");
    }

    #[test]
    fn evaluate_pitch_level01_is_clamped() {
        let mut landscape = Landscape::new(Log2Space::new(100.0, 400.0, 12));
        landscape.consonance_level01.fill(1.2);
        let val = landscape.evaluate_pitch_level01(200.0);
        assert!((val - 1.0).abs() < 1e-6, "val={val}");
    }

    #[test]
    fn consonance_level_matches_kernel_and_representation() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let params = build_params(&space);
        let mut landscape = Landscape::new(space);
        let n = landscape.roughness01.len();
        landscape.harmonicity = vec![0.25; n];
        landscape.roughness = vec![0.0; n];
        landscape.roughness01 = vec![0.4; n];
        landscape.recompute_consonance(&params);

        for i in 0..n {
            let h = landscape.harmonicity01[i].clamp(0.0, 1.0);
            let r = landscape.roughness01[i].clamp(0.0, 1.0);
            let score = params.consonance_kernel.score(h, r);
            let expected = params.consonance_representation.level01(score);
            let got = landscape.consonance_level01[i];
            assert!(
                (got - expected).abs() < 1e-6,
                "i={i} got={got} expected={expected}"
            );
        }
    }

    #[test]
    fn consonance_level01_independent_of_update_order() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let params = build_params(&space);
        let mut a = Landscape::new(space.clone());
        let n = a.roughness01.len();
        a.harmonicity = vec![0.75; n];
        a.roughness01 = vec![0.35; n];
        a.recompute_consonance(&params);
        let c_a = a.consonance_level01.clone();

        let mut b = Landscape::new(space);
        b.roughness01 = vec![0.35; n];
        b.harmonicity = vec![0.75; n];
        b.recompute_consonance(&params);
        for i in 0..n {
            assert!((c_a[i] - b.consonance_level01[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn consonance_score_decreases_with_roughness() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let mut params = build_params(&space);
        params.consonance_kernel = ConsonanceKernel {
            a: 1.0,
            b: -1.0,
            c: 0.0,
            d: 0.0,
        };

        let mut landscape = Landscape::new(space);
        let n = landscape.roughness01.len();
        landscape.harmonicity = vec![1.0; n];
        landscape.roughness = vec![0.0; n];

        landscape.roughness01 = vec![0.0; n];
        landscape.recompute_consonance(&params);
        let c0 = landscape.consonance_score[0];

        landscape.roughness01 = vec![0.5; n];
        landscape.recompute_consonance(&params);
        let c1 = landscape.consonance_score[0];

        landscape.roughness01 = vec![1.0; n];
        landscape.recompute_consonance(&params);
        let c2 = landscape.consonance_score[0];

        assert!(
            c0 > c1 && c1 > c2,
            "expected monotonic decrease: {c0} > {c1} > {c2}"
        );
    }

    #[test]
    fn consonance_weight_non_negative_and_density_normalized() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let params = build_params(&space);
        let mut landscape = Landscape::new(space);
        let n = landscape.roughness01.len();
        landscape.harmonicity = vec![0.5; n];
        landscape.roughness01 = vec![0.2; n];
        landscape.recompute_consonance(&params);

        for &w in &landscape.consonance_weight {
            assert!(w >= 0.0, "weight should be non-negative: {w}");
        }
        let sum: f32 = landscape.consonance_density.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "density sum={sum}");
    }

    #[test]
    fn recompute_consonance_sets_energy_and_density_consistently() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let params = build_params(&space);
        let mut landscape = Landscape::new(space);
        let n = landscape.roughness01.len();
        landscape.harmonicity01 = vec![0.7; n];
        landscape.roughness01 = vec![0.3; n];
        landscape.recompute_consonance(&params);

        for i in 0..n {
            let score = landscape.consonance_score[i];
            let energy = landscape.consonance_energy[i];
            assert!(
                (energy + score).abs() < 1e-6,
                "energy must be -score at i={i}: score={score} energy={energy}"
            );
        }
        let density_sum: f32 = landscape.consonance_density.iter().sum();
        assert!(
            (density_sum - 1.0).abs() < 1e-5,
            "consonance_density must sum to 1, got {density_sum}"
        );
    }

    #[test]
    fn recompute_consonance_keeps_level01_finite_and_bounded() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let params = build_params(&space);
        let mut landscape = Landscape::new(space);
        let n = landscape.roughness01.len();
        landscape.harmonicity01 = vec![0.7; n];
        landscape.roughness01 = vec![0.3; n];
        landscape.recompute_consonance(&params);

        for (i, &level) in landscape.consonance_level01.iter().enumerate() {
            assert!(
                level.is_finite(),
                "level01 must be finite at i={i}: {level}"
            );
            assert!(
                (0.0..=1.0).contains(&level),
                "level01 must be within [0,1] at i={i}: {level}"
            );
        }
    }

    #[test]
    fn resize_to_space_keeps_lengths_and_allows_recompute() {
        let space_a = Log2Space::new(100.0, 400.0, 12);
        let space_b = Log2Space::new(80.0, 800.0, 24);
        let mut landscape = Landscape::new(space_a);
        landscape.harmonicity.fill(0.6);
        landscape.roughness01.fill(0.2);

        landscape.resize_to_space(space_b.clone());
        let n = space_b.n_bins();
        assert_eq!(landscape.roughness.len(), n);
        assert_eq!(landscape.roughness01.len(), n);
        assert_eq!(landscape.harmonicity.len(), n);
        assert_eq!(landscape.harmonicity01.len(), n);
        assert_eq!(landscape.consonance_score.len(), n);
        assert_eq!(landscape.consonance_level01.len(), n);
        assert_eq!(landscape.consonance_weight.len(), n);
        assert_eq!(landscape.consonance_density.len(), n);
        assert_eq!(landscape.consonance_energy.len(), n);
        assert_eq!(landscape.subjective_intensity.len(), n);
        assert_eq!(landscape.nsgt_power.len(), n);

        let params = build_params(&space_b);
        landscape.recompute_consonance(&params);
    }
}

impl Default for Landscape {
    fn default() -> Self {
        Self::new(Log2Space::new(1.0, 2.0, 1))
    }
}
