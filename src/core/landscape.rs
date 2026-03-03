//! core/landscape.rs — Data container for the psychoacoustic map (The Map).
//!
//! This module now holds only the data structures and lightweight helpers.
//! Processing lives in the Dorsal (rhythm) and Roughness/Harmonicity streams.

use crate::core::consonance_kernel::{ConsonanceKernel, ConsonanceRepresentationParams};
use crate::core::log2space::{Log2Space, sample_scan_linear_at_pos};
use crate::core::modulation::NeuralRhythms;
use crate::core::psycho_state::sanitize01;

#[derive(Clone, Debug)]
pub struct LandscapeParams {
    pub fs: f32,
    pub max_hist_cols: usize,
    pub roughness_kernel: crate::core::roughness_kernel::RoughnessKernel,
    pub harmonicity_kernel: crate::core::harmonicity_kernel::HarmonicityKernel,
    pub consonance_kernel: ConsonanceKernel,
    pub consonance_representation: ConsonanceRepresentationParams,
    pub consonance_density_roughness_gain: f32,

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
    /// Runtime copy of roughness-kernel center suppression width (ERB).
    /// Used by behavior-side crowding when sigma is synced to roughness settings.
    pub roughness_suppress_sigma_erb: f32,
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
    pub consonance_field_score: Vec<f32>,
    /// Layer 2: normalized level in [0, 1].
    pub consonance_field_level: Vec<f32>,
    /// Density axis: raw spawn weight before occupancy/mask.
    /// Computed as max(0, H01 * (1 - rho * R01)) via ConsonanceKernel::density_with_rho.
    pub consonance_density_mass: Vec<f32>,
    /// Density axis: normalized PMF for no-mask/default view.
    pub consonance_density_pmf: Vec<f32>,
    /// Field energy cache (`-score`) kept for diagnostics/tests/plots.
    pub consonance_field_energy: Vec<f32>,
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
            roughness_suppress_sigma_erb: 0.06,
            roughness: vec![0.0; n],
            roughness_shape_raw: vec![0.0; n],
            roughness01: vec![0.0; n],
            harmonicity: vec![0.0; n],
            harmonicity_path_a: vec![0.0; n],
            harmonicity_path_b: vec![0.0; n],
            harmonicity01: vec![0.0; n],
            consonance_field_score: vec![0.0; n],
            consonance_field_level: vec![0.0; n],
            consonance_density_mass: vec![0.0; n],
            consonance_density_pmf: vec![0.0; n],
            consonance_field_energy: vec![0.0; n],
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
        self.consonance_field_score.resize(n, 0.0);
        self.consonance_field_level.resize(n, 0.0);
        self.consonance_density_mass.resize(n, 0.0);
        self.consonance_density_pmf.resize(n, 0.0);
        self.consonance_field_energy.resize(n, 0.0);
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
            .assert_scan_len_named(&self.consonance_field_score, "consonance_field_score");
        self.space
            .assert_scan_len_named(&self.consonance_field_level, "consonance_field_level");
        self.space
            .assert_scan_len_named(&self.consonance_density_mass, "consonance_density_mass");
        self.space
            .assert_scan_len_named(&self.consonance_density_pmf, "consonance_density_pmf");
        self.space
            .assert_scan_len_named(&self.consonance_field_energy, "consonance_field_energy");
        self.space
            .assert_scan_len_named(&self.subjective_intensity, "subjective_intensity");
        self.space
            .assert_scan_len_named(&self.nsgt_power, "nsgt_power");
    }

    /// Layer 1 score. Prefer `evaluate_pitch_level` for normalized usage.
    pub fn evaluate_pitch_score(&self, freq_hz: f32) -> f32 {
        self.assert_scan_lengths();
        self.sample_linear(&self.consonance_field_score, freq_hz)
    }

    /// Layer 1 score. Prefer `evaluate_pitch_level_log2` for normalized usage.
    pub fn evaluate_pitch_score_log2(&self, log_freq: f32) -> f32 {
        self.assert_scan_lengths();
        self.sample_linear_log2(&self.consonance_field_score, log_freq)
    }

    pub fn evaluate_pitch_level(&self, freq_hz: f32) -> f32 {
        self.assert_scan_lengths();
        self.sample_linear(&self.consonance_field_level, freq_hz)
            .clamp(0.0, 1.0)
    }

    pub fn evaluate_pitch_level_log2(&self, log_freq: f32) -> f32 {
        self.assert_scan_lengths();
        self.sample_linear_log2(&self.consonance_field_level, log_freq)
            .clamp(0.0, 1.0)
    }

    pub fn consonance_field_score_at(&self, freq_hz: f32) -> f32 {
        self.assert_scan_lengths();
        self.evaluate_pitch_score(freq_hz)
    }

    pub fn consonance_field_level_at(&self, freq_hz: f32) -> f32 {
        self.assert_scan_lengths();
        self.evaluate_pitch_level(freq_hz)
    }

    pub fn freq_bounds(&self) -> (f32, f32) {
        (self.space.fmin, self.space.fmax)
    }

    pub fn freq_bounds_log2(&self) -> (f32, f32) {
        (self.space.fmin.log2(), self.space.fmax.log2())
    }

    pub fn recompute_consonance(&mut self, params: &LandscapeParams) {
        self.assert_scan_lengths();

        reset_if_len_mismatch(&mut self.harmonicity01, self.harmonicity.len());
        reset_if_len_mismatch(&mut self.roughness01, self.roughness.len());

        let n = self.consonance_field_score.len();
        reset_if_len_mismatch(&mut self.consonance_field_level, n);
        reset_if_len_mismatch(&mut self.consonance_density_mass, n);
        reset_if_len_mismatch(&mut self.consonance_density_pmf, n);
        reset_if_len_mismatch(&mut self.consonance_field_energy, n);

        let perc_h_pot_scan = &self.harmonicity;
        crate::core::psycho_state::h_pot_scan_to_h_state01_scan(
            perc_h_pot_scan,
            1.0,
            &mut self.harmonicity01,
        );

        debug_assert_eq!(self.harmonicity01.len(), n);
        debug_assert_eq!(self.roughness01.len(), n);

        self.recompute_consonance_field(params);
        self.recompute_consonance_density_raw(params);
        for i in 0..n {
            self.consonance_density_pmf[i] = self.consonance_density_mass[i];
        }
        normalize_or_uniform(&mut self.consonance_density_pmf[..n]);
    }

    pub fn recompute_consonance_field(&mut self, params: &LandscapeParams) {
        self.assert_scan_lengths();
        let n = self.consonance_field_score.len();
        for i in 0..n {
            let h01 = sanitize01(self.harmonicity01[i]);
            let r01 = sanitize01(self.roughness01[i]);
            let score = params.consonance_kernel.score(h01, r01);
            self.consonance_field_score[i] = score;
            self.consonance_field_level[i] = params.consonance_representation.level(score);
            self.consonance_field_energy[i] = params.consonance_representation.energy(score);
        }
    }

    pub fn recompute_consonance_density_raw(&mut self, params: &LandscapeParams) {
        self.assert_scan_lengths();
        let density_kernel =
            ConsonanceKernel::density_with_rho(params.consonance_density_roughness_gain);
        let n = self.consonance_density_mass.len();
        for i in 0..n {
            let h01 = sanitize01(self.harmonicity01[i]);
            let r01 = sanitize01(self.roughness01[i]);
            let raw = density_kernel.score(h01, r01).max(0.0);
            self.consonance_density_mass[i] = if raw.is_finite() { raw } else { 0.0 };
        }
    }

    pub fn build_consonance_density_pmf(&self, occupied: &[bool], out: &mut [f32]) {
        self.assert_scan_lengths();
        debug_assert_eq!(occupied.len(), self.consonance_density_mass.len());
        debug_assert_eq!(out.len(), self.consonance_density_mass.len());
        if out.is_empty() {
            return;
        }
        let n = out
            .len()
            .min(self.consonance_density_mass.len())
            .min(occupied.len());

        let occupied_n = &occupied[..n];
        let out_n = &mut out[..n];
        let mut unoccupied_count = 0usize;
        for i in 0..n {
            let is_occupied = occupied_n[i];
            if !is_occupied {
                unoccupied_count += 1;
            }
            out_n[i] = if is_occupied {
                0.0
            } else {
                self.consonance_density_mass[i]
            };
        }
        normalize_or_uniform_masked(out_n, occupied_n, unoccupied_count);
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
        sample_scan_linear_at_pos(data, pos)
    }
}

#[inline]
fn reset_if_len_mismatch(scan: &mut Vec<f32>, expected_len: usize) {
    if scan.len() != expected_len {
        *scan = vec![0.0; expected_len];
    }
}

#[inline]
fn sanitize_nonnegative_finite(x: f32) -> f32 {
    if x.is_finite() { x.max(0.0) } else { 0.0 }
}

fn normalize_or_uniform(out: &mut [f32]) {
    let mut sum = 0.0f32;
    for v in out.iter_mut() {
        *v = sanitize_nonnegative_finite(*v);
        sum += *v;
    }
    if sum > 0.0 && sum.is_finite() {
        let inv = 1.0 / sum;
        for v in out.iter_mut() {
            *v *= inv;
        }
    } else if !out.is_empty() {
        out.fill(1.0 / out.len() as f32);
    }
}

fn normalize_or_uniform_masked(out: &mut [f32], occupied: &[bool], unoccupied_count: usize) {
    debug_assert_eq!(out.len(), occupied.len());
    let mut sum = 0.0f32;
    for v in out.iter_mut() {
        *v = sanitize_nonnegative_finite(*v);
        sum += *v;
    }
    if sum > 0.0 && sum.is_finite() {
        let inv = 1.0 / sum;
        for v in out.iter_mut() {
            *v *= inv;
        }
        return;
    }

    if unoccupied_count > 0 {
        let uniform = 1.0 / unoccupied_count as f32;
        for (i, v) in out.iter_mut().enumerate() {
            *v = if occupied[i] { 0.0 } else { uniform };
        }
    } else if !out.is_empty() {
        out.fill(1.0 / out.len() as f32);
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
            consonance_density_roughness_gain: 1.0,
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
    fn consonance_field_level_stays_in_range() {
        let params = build_params(&Log2Space::new(100.0, 400.0, 12));
        let h01 = [0.0f32, 0.5, 1.0];
        let r01 = [0.0f32, 0.4, 1.0];
        for &h in &h01 {
            for &r in &r01 {
                let score = params.consonance_kernel.score(h, r);
                let level = params.consonance_representation.level(score);
                assert!((0.0..=1.0).contains(&level));
            }
        }
    }

    #[test]
    fn evaluate_pitch_level_uses_consonance_field_level() {
        let mut landscape = Landscape::new(Log2Space::new(100.0, 400.0, 12));
        landscape.consonance_field_score.fill(10.0);
        landscape.consonance_field_level.fill(0.3);
        let val = landscape.evaluate_pitch_level(200.0);
        assert!((val - 0.3).abs() < 1e-6, "val={val}");
    }

    #[test]
    fn evaluate_pitch_level_is_clamped() {
        let mut landscape = Landscape::new(Log2Space::new(100.0, 400.0, 12));
        landscape.consonance_field_level.fill(1.2);
        let val = landscape.evaluate_pitch_level(200.0);
        assert!((val - 1.0).abs() < 1e-6, "val={val}");
    }

    #[test]
    fn evaluate_pitch_score_log2_outside_space_uses_edge_bins() {
        let mut landscape = Landscape::new(Log2Space::new(100.0, 400.0, 12));
        let n = landscape.consonance_field_score.len();
        landscape.consonance_field_score.fill(0.0);
        landscape.consonance_field_score[0] = 0.25;
        landscape.consonance_field_score[n - 1] = 0.75;

        let lo = landscape.evaluate_pitch_score_log2(landscape.space.fmin.log2() - 10.0);
        let hi = landscape.evaluate_pitch_score_log2(landscape.space.fmax.log2() + 10.0);

        assert!((lo - 0.25).abs() < 1e-6, "lo={lo}");
        assert!((hi - 0.75).abs() < 1e-6, "hi={hi}");
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
            let expected = params.consonance_representation.level(score);
            let got = landscape.consonance_field_level[i];
            assert!(
                (got - expected).abs() < 1e-6,
                "i={i} got={got} expected={expected}"
            );
        }
    }

    #[test]
    fn consonance_field_level_independent_of_update_order() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let params = build_params(&space);
        let mut a = Landscape::new(space.clone());
        let n = a.roughness01.len();
        a.harmonicity = vec![0.75; n];
        a.roughness01 = vec![0.35; n];
        a.recompute_consonance(&params);
        let c_a = a.consonance_field_level.clone();

        let mut b = Landscape::new(space);
        b.roughness01 = vec![0.35; n];
        b.harmonicity = vec![0.75; n];
        b.recompute_consonance(&params);
        for i in 0..n {
            assert!((c_a[i] - b.consonance_field_level[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn consonance_field_score_decreases_with_roughness() {
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
        let c0 = landscape.consonance_field_score[0];

        landscape.roughness01 = vec![0.5; n];
        landscape.recompute_consonance(&params);
        let c1 = landscape.consonance_field_score[0];

        landscape.roughness01 = vec![1.0; n];
        landscape.recompute_consonance(&params);
        let c2 = landscape.consonance_field_score[0];

        assert!(
            c0 > c1 && c1 > c2,
            "expected monotonic decrease: {c0} > {c1} > {c2}"
        );
    }

    #[test]
    fn density_definition_matches_rho_one_kernel() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let params = build_params(&space);
        let mut landscape = Landscape::new(space);
        let n = landscape.roughness01.len();
        landscape.harmonicity = vec![0.5; n];
        landscape.roughness01 = vec![0.2; n];
        landscape.recompute_consonance(&params);

        for i in 0..n {
            let h = landscape.harmonicity01[i].clamp(0.0, 1.0);
            let r = landscape.roughness01[i].clamp(0.0, 1.0);
            let expected = ConsonanceKernel::density_with_rho(1.0).score(h, r).max(0.0);
            let got = landscape.consonance_density_mass[i];
            assert!(
                (got - expected).abs() < 1e-6,
                "density raw mismatch at i={i}: got={got} expected={expected}"
            );
            assert!(got.is_finite(), "density raw must be finite");
            assert!(
                (0.0..=1.0).contains(&got),
                "density raw out of [0,1]: {got}"
            );
        }
    }

    #[test]
    fn density_raw_edge_cases_match_definition() {
        let space = Log2Space::new(100.0, 400.0, 3);
        let params = build_params(&space);
        let mut landscape = Landscape::new(space);
        let n = landscape.roughness01.len();
        landscape.harmonicity = vec![0.0; n];
        landscape.roughness01 = vec![0.4; n];
        landscape.harmonicity[1] = 1.0;
        landscape.roughness01[1] = 1.0;
        landscape.harmonicity[2] = 1.0;
        landscape.roughness01[2] = 0.0;
        landscape.recompute_consonance(&params);
        assert_eq!(landscape.consonance_density_mass[0], 0.0); // H=0
        assert_eq!(landscape.consonance_density_mass[1], 0.0); // R=1
        assert_eq!(landscape.consonance_density_mass[2], 1.0); // H=1,R=0
    }

    #[test]
    fn density_raw_respects_rho_zero() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let mut params = build_params(&space);
        params.consonance_density_roughness_gain = 0.0;
        let mut landscape = Landscape::new(space);
        let n = landscape.roughness01.len();
        landscape.harmonicity01 = vec![0.7; n];
        landscape.roughness01 = vec![0.6; n];
        landscape.recompute_consonance(&params);

        for i in 0..n {
            let expected = landscape.harmonicity01[i].clamp(0.0, 1.0);
            let got = landscape.consonance_density_mass[i];
            assert!(
                (got - expected).abs() < 1e-6,
                "i={i} got={got} expected={expected}"
            );
        }
    }

    #[test]
    fn density_raw_respects_rho_two_with_clamp() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let mut params = build_params(&space);
        params.consonance_density_roughness_gain = 2.0;
        let mut landscape = Landscape::new(space);
        let n = landscape.roughness01.len();
        landscape.harmonicity01 = vec![1.0; n];
        landscape.roughness01 = vec![0.6; n];
        landscape.recompute_consonance(&params);

        for i in 0..n {
            let expected = (1.0 - 2.0 * 0.6f32).max(0.0);
            let got = landscape.consonance_density_mass[i];
            assert!(
                (got - expected).abs() < 1e-6,
                "i={i} got={got} expected={expected}"
            );
        }
    }

    #[test]
    fn density_raw_sanitizes_non_finite_and_negative_rho() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let mut params = build_params(&space);
        let mut landscape = Landscape::new(space);
        let n = landscape.roughness01.len();
        landscape.harmonicity01 = vec![0.8; n];
        landscape.roughness01 = vec![0.25; n];

        params.consonance_density_roughness_gain = f32::NAN;
        landscape.recompute_consonance(&params);
        for &raw in &landscape.consonance_density_mass {
            assert!(raw.is_finite());
            assert!(raw >= 0.0);
        }

        params.consonance_density_roughness_gain = -1.0;
        landscape.recompute_consonance(&params);
        for i in 0..n {
            let expected = landscape.harmonicity01[i].clamp(0.0, 1.0);
            let got = landscape.consonance_density_mass[i];
            assert!(
                (got - expected).abs() < 1e-6,
                "i={i} got={got} expected={expected}"
            );
        }
    }

    #[test]
    fn density_raw_stays_finite_for_extreme_positive_rho() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let mut params = build_params(&space);
        params.consonance_density_roughness_gain = 1e30;
        let mut landscape = Landscape::new(space);
        let n = landscape.roughness01.len();
        landscape.harmonicity01 = vec![0.9; n];
        landscape.roughness01 = vec![0.4; n];
        landscape.recompute_consonance(&params);

        for (i, &raw) in landscape.consonance_density_mass.iter().enumerate() {
            assert!(raw.is_finite(), "raw must be finite at i={i}");
            assert!(raw >= 0.0, "raw must be non-negative at i={i}: {raw}");
        }
    }

    #[test]
    fn density_normalization_and_fallbacks_work() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let params = build_params(&space);
        let mut landscape = Landscape::new(space);
        let n = landscape.roughness01.len();
        landscape.harmonicity01 = vec![0.8; n];
        landscape.roughness01 = vec![0.25; n];
        landscape.recompute_consonance(&params);

        let mut pmf = vec![0.0f32; n];
        let occupied_none = vec![false; n];
        landscape.build_consonance_density_pmf(&occupied_none, &mut pmf);
        let density_sum: f32 = pmf.iter().sum();
        assert!(
            (density_sum - 1.0).abs() < 1e-5,
            "consonance_density_pmf must sum to 1, got {density_sum}"
        );

        let mut occupied_some = vec![false; n];
        for i in (0..n).step_by(2) {
            occupied_some[i] = true;
        }
        landscape.build_consonance_density_pmf(&occupied_some, &mut pmf);
        let density_sum: f32 = pmf.iter().sum();
        assert!(
            (density_sum - 1.0).abs() < 1e-5,
            "sum with mask={density_sum}"
        );
        for i in 0..n {
            if occupied_some[i] {
                assert_eq!(pmf[i], 0.0, "occupied bin must be zero");
            }
        }

        landscape.consonance_density_mass.fill(0.0);
        landscape.build_consonance_density_pmf(&occupied_none, &mut pmf);
        let density_sum: f32 = pmf.iter().sum();
        assert!(
            (density_sum - 1.0).abs() < 1e-5,
            "zero-raw fallback sum={density_sum}"
        );

        let occupied_all = vec![true; n];
        landscape.build_consonance_density_pmf(&occupied_all, &mut pmf);
        let density_sum: f32 = pmf.iter().sum();
        assert!(
            (density_sum - 1.0).abs() < 1e-5,
            "all-occupied fallback sum={density_sum}"
        );
    }

    #[test]
    fn recompute_consonance_sets_field_energy_consistently() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let params = build_params(&space);
        let mut landscape = Landscape::new(space);
        let n = landscape.roughness01.len();
        landscape.harmonicity01 = vec![0.7; n];
        landscape.roughness01 = vec![0.3; n];
        landscape.recompute_consonance(&params);
        for i in 0..n {
            let score = landscape.consonance_field_score[i];
            let energy = landscape.consonance_field_energy[i];
            assert!(
                (energy + score).abs() < 1e-6,
                "energy must be -score at i={i}: score={score} energy={energy}"
            );
        }
    }

    #[test]
    fn recompute_consonance_normalizes_density_pmf_from_raw() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let params = build_params(&space);
        let mut landscape = Landscape::new(space);
        let n = landscape.roughness01.len();
        landscape.harmonicity = (0..n).map(|i| (i as f32 + 1.0) / n as f32).collect();
        landscape.roughness01 = vec![0.25; n];
        landscape.recompute_consonance(&params);

        let sum_raw: f32 = landscape
            .consonance_density_mass
            .iter()
            .map(|v| if v.is_finite() { v.max(0.0) } else { 0.0 })
            .sum();
        assert!(sum_raw > 0.0 && sum_raw.is_finite());

        for i in 0..n {
            let expected = landscape.consonance_density_mass[i].max(0.0) / sum_raw;
            let got = landscape.consonance_density_pmf[i];
            assert!(
                (got - expected).abs() < 1e-6,
                "density pmf mismatch at i={i}: got={got} expected={expected}"
            );
        }

        let pmf_sum: f32 = landscape.consonance_density_pmf.iter().sum();
        assert!((pmf_sum - 1.0).abs() < 1e-6, "pmf sum={pmf_sum}");
    }

    #[test]
    fn recompute_consonance_density_pmf_falls_back_to_uniform_on_all_zero_raw() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let params = build_params(&space);
        let mut landscape = Landscape::new(space);
        let n = landscape.roughness01.len();
        landscape.harmonicity.fill(0.0);
        landscape.roughness01.fill(1.0);
        landscape.recompute_consonance(&params);

        let uniform = 1.0 / n as f32;
        for (i, &p) in landscape.consonance_density_pmf.iter().enumerate() {
            assert!((p - uniform).abs() < 1e-6, "i={i} p={p} uniform={uniform}");
        }
    }

    #[test]
    fn recompute_consonance_keeps_level_finite_and_bounded() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let params = build_params(&space);
        let mut landscape = Landscape::new(space);
        let n = landscape.roughness01.len();
        landscape.harmonicity01 = vec![0.7; n];
        landscape.roughness01 = vec![0.3; n];
        landscape.recompute_consonance(&params);

        for (i, &level) in landscape.consonance_field_level.iter().enumerate() {
            assert!(level.is_finite(), "level must be finite at i={i}: {level}");
            assert!(
                (0.0..=1.0).contains(&level),
                "level must be within [0,1] at i={i}: {level}"
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
        assert_eq!(landscape.consonance_field_score.len(), n);
        assert_eq!(landscape.consonance_field_level.len(), n);
        assert_eq!(landscape.consonance_density_mass.len(), n);
        assert_eq!(landscape.consonance_density_pmf.len(), n);
        assert_eq!(landscape.consonance_field_energy.len(), n);
        assert_eq!(landscape.subjective_intensity.len(), n);
        assert_eq!(landscape.nsgt_power.len(), n);

        let params = build_params(&space_b);
        landscape.recompute_consonance(&params);
    }

    #[test]
    #[cfg(feature = "plotcheck")]
    fn plot_consonance_variants_scan_png() -> Result<(), Box<dyn std::error::Error>> {
        use plotters::prelude::*;

        std::fs::create_dir_all("target/plots")?;

        let space = Log2Space::new(20.0, 8000.0, 96);
        let mut params = build_params(&space);
        // Keep the E1 scan style kernel coefficients used in the paper tooling.
        params.consonance_kernel = ConsonanceKernel {
            a: 1.0,
            b: -0.85,
            c: 0.5,
            d: 0.0,
        };
        params.consonance_representation = ConsonanceRepresentationParams {
            beta: 2.0,
            theta: 0.0,
        };

        let n = space.n_bins();
        let anchor_hz = 440.0f32;
        let anchor_idx = space.index_of_freq(anchor_hz).unwrap_or(n / 2);
        let (_erb, du) = crate::core::roughness_kernel::erb_grid(&space);

        let mut env_scan = vec![0.0f32; n];
        env_scan[anchor_idx] = 1.0;
        let mut density_scan = vec![0.0f32; n];
        density_scan[anchor_idx] = 1.0 / du[anchor_idx].max(1e-12);

        let (h_pot_scan, _) = params
            .harmonicity_kernel
            .potential_h_from_log2_spectrum(&env_scan, &space);
        let (r_pot_scan, _) = params
            .roughness_kernel
            .potential_r_from_log2_spectrum_density(&density_scan, &space);

        let h_ref_max = h_pot_scan.iter().copied().fold(0.0f32, f32::max).max(1e-12);
        let mut h01_scan = vec![0.0f32; n];
        crate::core::psycho_state::h_pot_scan_to_h_state01_scan(
            &h_pot_scan,
            h_ref_max,
            &mut h01_scan,
        );

        let r_ref = crate::core::psycho_state::compute_roughness_reference(&params, &space);
        let mut r01_scan = vec![0.0f32; n];
        crate::core::psycho_state::r_pot_scan_to_r_state01_scan(
            &r_pot_scan,
            r_ref.peak,
            params.roughness_k,
            &mut r01_scan,
        );

        let mut c_score_scan = vec![0.0f32; n];
        let mut c_level_scan = vec![0.0f32; n];
        let mut c_density_mass_scan = vec![0.0f32; n];
        let mut c_energy_scan = vec![0.0f32; n];
        let density_kernel =
            ConsonanceKernel::density_with_rho(params.consonance_density_roughness_gain);
        for i in 0..n {
            let h = h01_scan[i].clamp(0.0, 1.0);
            let r = r01_scan[i].clamp(0.0, 1.0);
            let score = params.consonance_kernel.score(h01_scan[i], r01_scan[i]);
            c_score_scan[i] = score;
            c_level_scan[i] = params.consonance_representation.level(score);
            c_density_mass_scan[i] = density_kernel.score(h, r).max(0.0);
            c_energy_scan[i] = params.consonance_representation.energy(score);
        }
        let sum_raw: f32 = c_density_mass_scan.iter().sum();
        let mut c_density_scan = vec![0.0f32; n];
        if sum_raw > 0.0 && sum_raw.is_finite() {
            for i in 0..n {
                c_density_scan[i] = c_density_mass_scan[i] / sum_raw;
            }
        } else {
            let uniform = 1.0 / n as f32;
            c_density_scan.fill(uniform);
        }

        let density_sum: f32 = c_density_scan.iter().sum();
        assert!(
            (density_sum - 1.0).abs() < 1e-5,
            "density sum={density_sum}"
        );

        for i in 0..n {
            let score = c_score_scan[i];
            let level = c_level_scan[i];
            let density_raw = c_density_mass_scan[i];
            let density = c_density_scan[i];
            let energy = c_energy_scan[i];
            assert!(score.is_finite(), "score not finite at {i}");
            assert!(level.is_finite(), "level not finite at {i}");
            assert!(density_raw.is_finite(), "density raw not finite at {i}");
            assert!(density.is_finite(), "density not finite at {i}");
            assert!(energy.is_finite(), "energy not finite at {i}");
            assert!(
                (energy + score).abs() < 1e-6,
                "energy must be -score at i={i}: score={score} energy={energy}"
            );
            assert!((0.0..=1.0).contains(&level), "level out of range at {i}");
            assert!(density_raw >= 0.0, "density raw negative at {i}");
            assert!(density >= 0.0, "density negative at {i}");
        }

        let csv_path = "target/plots/it_consonance_e1_stack.csv";
        let mut csv = String::from(
            "freq_hz,h01,r01,c_field_score,c_field_level,c_density_mass,c_density_pmf,c_field_energy\n",
        );
        for i in 0..n {
            csv.push_str(&format!(
                "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                space.centers_hz[i],
                h01_scan[i],
                r01_scan[i],
                c_score_scan[i],
                c_level_scan[i],
                c_density_mass_scan[i],
                c_density_scan[i],
                c_energy_scan[i],
            ));
        }
        std::fs::write(csv_path, csv)?;

        let png_path = "target/plots/it_consonance_e1_stack.png";
        let root = BitMapBackend::new(png_path, (1600, 2100)).into_drawing_area();
        root.fill(&WHITE)?;
        let areas = root.split_evenly((7, 1));

        let x_min_hz = 20.0f32;
        let x_max_hz = 8000.0f32;
        let x_min = x_min_hz.log2();
        let x_max = x_max_hz.log2();
        let x_log2_scan: Vec<f32> = space
            .centers_hz
            .iter()
            .copied()
            .map(|hz| hz.clamp(x_min_hz, x_max_hz).log2())
            .collect();

        let mut chart_h = ChartBuilder::on(&areas[0])
            .caption("E1-like Stack: H01", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(75)
            .build_cartesian_2d(x_min..x_max, 0.0f32..1.0f32)?;
        chart_h
            .configure_mesh()
            .x_desc("frequency (Hz, log2 axis)")
            .x_label_formatter(&|x| format!("{:.0}", 2.0f32.powf(*x)))
            .y_desc("H01")
            .draw()?;
        chart_h.draw_series(LineSeries::new(
            x_log2_scan.iter().copied().zip(h01_scan.iter().copied()),
            &GREEN,
        ))?;

        let mut chart_r = ChartBuilder::on(&areas[1])
            .caption("E1-like Stack: R01", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(75)
            .build_cartesian_2d(x_min..x_max, 0.0f32..1.0f32)?;
        chart_r
            .configure_mesh()
            .x_desc("frequency (Hz, log2 axis)")
            .x_label_formatter(&|x| format!("{:.0}", 2.0f32.powf(*x)))
            .y_desc("R01")
            .draw()?;
        chart_r.draw_series(LineSeries::new(
            x_log2_scan.iter().copied().zip(r01_scan.iter().copied()),
            &RED,
        ))?;

        let c_score_min = c_score_scan.iter().copied().fold(f32::INFINITY, f32::min);
        let c_score_max = c_score_scan
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let c_score_pad = ((c_score_max - c_score_min).abs() * 0.1).max(1e-3);
        let mut chart_score = ChartBuilder::on(&areas[2])
            .caption("E1-like Stack: C_field_score", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(75)
            .build_cartesian_2d(
                x_min..x_max,
                (c_score_min - c_score_pad)..(c_score_max + c_score_pad),
            )?;
        chart_score
            .configure_mesh()
            .x_desc("frequency (Hz, log2 axis)")
            .x_label_formatter(&|x| format!("{:.0}", 2.0f32.powf(*x)))
            .y_desc("score")
            .draw()?;
        chart_score.draw_series(LineSeries::new(
            x_log2_scan
                .iter()
                .copied()
                .zip(c_score_scan.iter().copied()),
            &BLUE,
        ))?;

        let mut chart_level = ChartBuilder::on(&areas[3])
            .caption("E1-like Stack: C_field_level", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(75)
            .build_cartesian_2d(x_min..x_max, 0.0f32..1.0f32)?;
        chart_level
            .configure_mesh()
            .x_desc("frequency (Hz, log2 axis)")
            .x_label_formatter(&|x| format!("{:.0}", 2.0f32.powf(*x)))
            .y_desc("level")
            .draw()?;
        chart_level.draw_series(LineSeries::new(
            x_log2_scan
                .iter()
                .copied()
                .zip(c_level_scan.iter().copied()),
            &MAGENTA,
        ))?;

        let c_weight_max = c_density_mass_scan
            .iter()
            .copied()
            .fold(0.0f32, f32::max)
            .max(1e-12);
        let mut chart_weight = ChartBuilder::on(&areas[4])
            .caption("E1-like Stack: C_density_mass", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(75)
            .build_cartesian_2d(x_min..x_max, 0.0f32..(c_weight_max * 1.05))?;
        chart_weight
            .configure_mesh()
            .x_desc("frequency (Hz, log2 axis)")
            .x_label_formatter(&|x| format!("{:.0}", 2.0f32.powf(*x)))
            .y_desc("density weight raw")
            .draw()?;
        chart_weight.draw_series(LineSeries::new(
            x_log2_scan
                .iter()
                .copied()
                .zip(c_density_mass_scan.iter().copied()),
            &CYAN,
        ))?;

        let c_density_max = c_density_scan
            .iter()
            .copied()
            .fold(0.0f32, f32::max)
            .max(1e-12);
        let mut chart_density = ChartBuilder::on(&areas[5])
            .caption("E1-like Stack: C_density_pmf", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(75)
            .build_cartesian_2d(x_min..x_max, 0.0f32..(c_density_max * 1.05))?;
        chart_density
            .configure_mesh()
            .x_desc("frequency (Hz, log2 axis)")
            .x_label_formatter(&|x| format!("{:.0}", 2.0f32.powf(*x)))
            .y_desc("density pmf")
            .draw()?;
        chart_density.draw_series(LineSeries::new(
            x_log2_scan
                .iter()
                .copied()
                .zip(c_density_scan.iter().copied()),
            &BLACK,
        ))?;

        let c_energy_min = c_energy_scan.iter().copied().fold(f32::INFINITY, f32::min);
        let c_energy_max = c_energy_scan
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let c_energy_pad = ((c_energy_max - c_energy_min).abs() * 0.1).max(1e-3);
        let mut chart_energy = ChartBuilder::on(&areas[6])
            .caption("E1-like Stack: C_field_energy", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(75)
            .build_cartesian_2d(
                x_min..x_max,
                (c_energy_min - c_energy_pad)..(c_energy_max + c_energy_pad),
            )?;
        chart_energy
            .configure_mesh()
            .x_desc("frequency (Hz, log2 axis)")
            .x_label_formatter(&|x| format!("{:.0}", 2.0f32.powf(*x)))
            .y_desc("energy")
            .draw()?;
        chart_energy.draw_series(LineSeries::new(
            x_log2_scan
                .iter()
                .copied()
                .zip(c_energy_scan.iter().copied()),
            &BLUE,
        ))?;

        root.present()?;
        assert!(std::path::Path::new(png_path).exists());
        assert!(std::path::Path::new(csv_path).exists());
        Ok(())
    }
}

impl Default for Landscape {
    fn default() -> Self {
        Self::new(Log2Space::new(1.0, 2.0, 1))
    }
}
