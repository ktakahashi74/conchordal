use crate::core::landscape::RoughnessScalarMode;
use crate::core::landscape::{Landscape, LandscapeParams, LandscapeUpdate};
use crate::core::landscape_spectral::SpectralFrontEnd;
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::core::roughness_kernel::erb_grid;

/// Roughness Stream (formerly Ventral).
/// Handles slow spectral analysis focused on roughness.
pub struct RoughnessStream {
    nsgt_rt: RtNsgtKernelLog2,
    params: LandscapeParams,
    spectral_frontend: SpectralFrontEnd,
    roughness_ref_total: f32,
    roughness_ref_peak: f32,

    // Last computed state (roughness-side of the landscape)
    last_landscape: Landscape,
}

impl RoughnessStream {
    pub fn new(params: LandscapeParams, nsgt_rt: RtNsgtKernelLog2) -> Self {
        let spectral_frontend = SpectralFrontEnd::new(nsgt_rt.space().clone(), &params);
        let (roughness_ref_total, roughness_ref_peak) =
            compute_roughness_reference(&params, nsgt_rt.space());

        Self {
            nsgt_rt: nsgt_rt.clone(),
            params,
            spectral_frontend,
            roughness_ref_total,
            roughness_ref_peak,
            last_landscape: Landscape::new(nsgt_rt.space().clone()),
        }
    }

    /// Access the most recent landscape snapshot without processing.
    pub fn last(&self) -> &Landscape {
        &self.last_landscape
    }

    /// Process audio chunk asynchronously.
    /// Returns the updated Landscape (roughness only).
    pub fn process(&mut self, audio: &[f32]) -> Landscape {
        if audio.is_empty() {
            return self.last_landscape.clone();
        }

        // 1. Update Spectrum (NSGT + Normalization)
        let envelope: Vec<f32> = {
            let env = self.nsgt_rt.process_hop(audio);
            env.to_vec()
        };
        self.last_landscape.nsgt_power = envelope.clone();
        let dt_sec = audio.len() as f32 / self.params.fs;
        let spectral_frame =
            self.spectral_frontend
                .process_nsgt_power(&envelope, dt_sec, &self.params);

        // 2. Compute Roughness
        self.compute_potentials(
            &spectral_frame.subjective_intensity,
            spectral_frame.loudness_mass,
        );

        // 3. Return snapshot
        self.last_landscape.clone()
    }

    fn compute_potentials(&mut self, density: &[f32], loudness_mass: f32) {
        let space = self.nsgt_rt.space();
        let (_erb, du) = erb_grid(space);
        let eps = self.params.roughness_ref_eps.max(1e-12);
        let roughness_k = self.params.roughness_k.max(1e-6);

        // Roughness strength (level-dependent).
        let (r_strength, r_total) = self
            .params
            .roughness_kernel
            .potential_r_from_log2_spectrum(density, space);
        let r_max = r_strength.iter().cloned().fold(0.0f32, f32::max);
        let r_p95 = percentile_95(&r_strength);
        let r_scalar_raw = match self.params.roughness_scalar_mode {
            RoughnessScalarMode::Total => r_total,
            RoughnessScalarMode::Max => r_max,
            RoughnessScalarMode::P95 => r_p95,
        };
        let r_norm = r_scalar_raw / (loudness_mass + eps);

        // Roughness shape (level-invariant).
        let (p_density, mass) = normalize_density(density, &du, eps);
        let (r_shape_raw, r_shape_total) = if mass > eps {
            self.params
                .roughness_kernel
                .potential_r_from_log2_spectrum(&p_density, space)
        } else {
            (vec![0.0; r_strength.len()], 0.0)
        };

        let r_ref_peak = self.roughness_ref_peak.max(eps);
        let r_ref_total = self.roughness_ref_total.max(eps);
        let r01 = r_shape_raw
            .iter()
            .map(|&ri| roughness_ratio_to_01(ri / r_ref_peak, roughness_k))
            .collect::<Vec<f32>>();
        let r01_scalar = roughness_ratio_to_01(r_shape_total / r_ref_total, roughness_k);

        self.last_landscape.roughness = r_strength;
        self.last_landscape.roughness_shape_raw = r_shape_raw;
        self.last_landscape.roughness01 = r01;
        self.last_landscape.roughness_total = r_total;
        self.last_landscape.roughness_max = r_max;
        self.last_landscape.roughness_p95 = r_p95;
        self.last_landscape.roughness_scalar_raw = r_scalar_raw;
        self.last_landscape.roughness_norm = r_norm;
        self.last_landscape.roughness01_scalar = r01_scalar;
        self.last_landscape.loudness_mass = loudness_mass;
        self.last_landscape.subjective_intensity = density.to_vec();
    }

    pub fn reset(&mut self) {
        self.nsgt_rt.reset();
        self.spectral_frontend.reset();
        self.last_landscape = Landscape::new(self.nsgt_rt.space().clone());
    }

    pub fn apply_update(&mut self, upd: LandscapeUpdate) {
        if let Some(k) = upd.roughness_k {
            self.params.roughness_k = k.max(1e-6);
        }
    }
}

fn roughness_ratio_to_01(x: f32, k: f32) -> f32 {
    if x.is_nan() {
        return 0.0;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() { 1.0 } else { 0.0 };
    }
    let x = x.max(0.0);
    let k = if k.is_finite() { k.max(1e-6) } else { 1e-6 };
    let r_ref = 1.0 / (1.0 + k);
    if x <= 0.0 {
        0.0
    } else if x < 1.0 {
        (x * r_ref).clamp(0.0, 1.0)
    } else {
        let denom = x + k;
        if denom <= 0.0 {
            return 1.0;
        }
        (1.0 - k / denom).clamp(0.0, 1.0)
    }
}

fn normalize_density(density: &[f32], du: &[f32], eps: f32) -> (Vec<f32>, f32) {
    let mass = crate::core::density::density_to_mass(density, du);
    if mass <= eps {
        return (vec![0.0; density.len()], mass);
    }
    let inv = 1.0 / (mass + eps);
    let p_density = density.iter().map(|&v| v * inv).collect();
    (p_density, mass)
}

fn compute_roughness_reference(
    params: &LandscapeParams,
    space: &crate::core::log2space::Log2Space,
) -> (f32, f32) {
    let eps = params.roughness_ref_eps.max(1e-12);
    let ref_density = build_reference_density(params, space);
    let (r, r_total) = params
        .roughness_kernel
        .potential_r_from_log2_spectrum_density(&ref_density, space);
    let r_peak = r.iter().copied().fold(0.0f32, f32::max).max(eps);
    (r_total.max(eps), r_peak)
}

fn build_reference_density(
    params: &LandscapeParams,
    space: &crate::core::log2space::Log2Space,
) -> Vec<f32> {
    let (erb, du) = erb_grid(space);
    if erb.is_empty() {
        return Vec::new();
    }
    let f0 = params.roughness_ref_f0_hz.max(1.0);
    let a = nearest_bin_by_freq(space, f0);
    let target_erb = erb[a] + params.roughness_ref_sep_erb;
    let mut b = nearest_bin_by_erb(&erb, target_erb);
    if b == a && erb.len() > 1 {
        b = if a + 1 < erb.len() {
            a + 1
        } else {
            a.saturating_sub(1)
        };
    }

    let mut ref_density = vec![0.0f32; erb.len()];
    let m_a = params.roughness_ref_mass_split.clamp(0.0, 1.0);
    let m_b = 1.0 - m_a;
    if du[a] > 0.0 {
        ref_density[a] += m_a / du[a];
    }
    if b != a && du[b] > 0.0 {
        ref_density[b] += m_b / du[b];
    } else if b == a && du[a] > 0.0 {
        ref_density[a] += m_b / du[a];
    }

    ref_density
}

fn nearest_bin_by_freq(space: &crate::core::log2space::Log2Space, f0: f32) -> usize {
    let mut best = 0;
    let mut best_diff = f32::MAX;
    for (i, &f) in space.centers_hz.iter().enumerate() {
        let diff = (f - f0).abs();
        if diff < best_diff {
            best_diff = diff;
            best = i;
        }
    }
    best
}

fn nearest_bin_by_erb(erb: &[f32], target: f32) -> usize {
    let mut best = 0;
    let mut best_diff = f32::MAX;
    for (i, &u) in erb.iter().enumerate() {
        let diff = (u - target).abs();
        if diff < best_diff {
            best_diff = diff;
            best = i;
        }
    }
    best
}

fn percentile_95(vals: &[f32]) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut buf = vals.to_vec();
    let idx = ((buf.len() - 1) as f32 * 0.95).round() as usize;
    let (_, v, _) = buf.select_nth_unstable_by(idx, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    *v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
    use crate::core::landscape::{LandscapeParams, RoughnessScalarMode};
    use crate::core::log2space::Log2Space;
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
    fn roughness_shape_invariant_to_level() {
        let space = Log2Space::new(80.0, 8000.0, 96);
        let params = build_params(&space);
        let (_erb, du) = erb_grid(&space);

        let mut density = vec![0.0f32; space.n_bins()];
        let mid = density.len() / 2;
        density[mid] = 0.8;
        if mid + 8 < density.len() {
            density[mid + 8] = 0.5;
        }

        let (p1, _) = normalize_density(&density, &du, params.roughness_ref_eps);
        let (r1, _) = params
            .roughness_kernel
            .potential_r_from_log2_spectrum(&p1, &space);

        let scaled: Vec<f32> = density.iter().map(|&v| v * 5.0).collect();
        let (p2, _) = normalize_density(&scaled, &du, params.roughness_ref_eps);
        let (r2, _) = params
            .roughness_kernel
            .potential_r_from_log2_spectrum(&p2, &space);

        for (a, b) in r1.iter().zip(r2.iter()) {
            assert!((a - b).abs() < 1e-6, "shape changed: {a} vs {b}");
        }
    }

    #[test]
    fn reference_maps_to_expected() {
        let space = Log2Space::new(80.0, 8000.0, 96);
        let params = build_params(&space);
        let ref_density = build_reference_density(&params, &space);
        let (r_shape, r_total) = params
            .roughness_kernel
            .potential_r_from_log2_spectrum_density(&ref_density, &space);
        let (r_ref_total, r_ref_peak) = compute_roughness_reference(&params, &space);

        let peak = r_shape.iter().copied().fold(0.0f32, f32::max);
        let r01_peak = roughness_ratio_to_01(peak / r_ref_peak, params.roughness_k);
        let r01_scalar = roughness_ratio_to_01(r_total / r_ref_total, params.roughness_k);
        let expected = 1.0 / (1.0 + params.roughness_k.max(1e-6));

        assert!(
            (r01_scalar - expected).abs() < 1e-5,
            "scalar {r01_scalar}"
        );
        assert!(
            (r01_peak - expected).abs() < 1e-5,
            "peak {r01_peak}"
        );
    }

    #[test]
    fn roughness_ratio_handles_nan_and_inf() {
        let k = 0.3;
        assert_eq!(roughness_ratio_to_01(f32::NAN, k), 0.0);
        assert_eq!(roughness_ratio_to_01(f32::INFINITY, k), 1.0);
        assert_eq!(roughness_ratio_to_01(f32::NEG_INFINITY, k), 0.0);
    }

    #[test]
    fn single_peak_has_zero_self_roughness() {
        let space = Log2Space::new(80.0, 8000.0, 96);
        let params = build_params(&space);
        let (_erb, du) = erb_grid(&space);

        let mut density = vec![0.0f32; space.n_bins()];
        let mid = density.len() / 2;
        density[mid] = 1.0;

        let (p_density, _) = normalize_density(&density, &du, params.roughness_ref_eps);
        let (r_shape, _r_total) = params
            .roughness_kernel
            .potential_r_from_log2_spectrum(&p_density, &space);

        assert!(r_shape[mid].abs() < 1e-6, "self roughness {}", r_shape[mid]);
    }
}
