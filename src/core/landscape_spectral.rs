//! Shared spectral front-end for landscape processing.
//! NSGT power -> peak extraction -> delta + loudness -> leaky normalization.

use crate::core::a_weighting;
use crate::core::density;
use crate::core::landscape::LandscapeParams;
use crate::core::log2space::Log2Space;
use crate::core::peak_extraction::{Peak, PeakExtractConfig, extract_peaks_density_with_grid};
use crate::core::roughness_kernel::erb_grid;

#[derive(Clone, Debug)]
pub struct SpectralFrame {
    pub subjective_intensity: Vec<f32>,
    pub loudness_mass: f32,
    pub peaks_raw: Vec<Peak>,
}

pub struct SpectralFrontEnd {
    space: Log2Space,
    erb: Vec<f32>,
    du: Vec<f32>,
    loudness_weights_pow: Vec<f32>,
    norm_state: Vec<f32>,
    peak_cfg: PeakExtractConfig,
}

impl SpectralFrontEnd {
    pub fn new(space: Log2Space, _params: &LandscapeParams) -> Self {
        let (erb, du) = erb_grid(&space);
        let loudness_weights_pow = space
            .centers_hz
            .iter()
            .map(|&f| a_weighting::a_weighting_gain_pow(f))
            .collect::<Vec<f32>>();
        let n = space.n_bins();
        Self {
            space,
            erb,
            du,
            loudness_weights_pow,
            norm_state: vec![0.0; n],
            peak_cfg: PeakExtractConfig::nsgt_default(),
        }
    }

    pub fn reset(&mut self) {
        self.norm_state.fill(0.0);
    }

    pub fn process_nsgt_power(
        &mut self,
        nsgt_power: &[f32],
        dt_sec: f32,
        params: &LandscapeParams,
    ) -> SpectralFrame {
        assert_eq!(nsgt_power.len(), self.du.len());

        let exp = params.loudness_exp.max(0.01);
        let tau_s = params.tau_ms.max(1.0) * 1e-3;
        let a = (-dt_sec / tau_s).exp();
        let ref_power = params.ref_power.max(1e-12);

        let mut raw_density = vec![0.0f32; nsgt_power.len()];
        for (i, &pow) in nsgt_power.iter().enumerate() {
            let dui = self.du[i].max(1e-12);
            let density = (pow / dui).max(0.0);
            raw_density[i] = density;
        }

        let peaks = extract_peaks_density_with_grid(
            &raw_density,
            &self.erb,
            &self.du,
            &self.peak_cfg,
        );

        let mut subj_density_delta = vec![0.0f32; nsgt_power.len()];
        for peak in &peaks {
            if peak.bin_idx >= self.du.len() {
                continue;
            }
            let dui = self.du[peak.bin_idx];
            if dui <= 0.0 {
                continue;
            }
            let weight_pow = self.loudness_weights_pow[peak.bin_idx];
            let subj_mass = ((peak.mass * weight_pow) / ref_power).powf(exp);
            subj_density_delta[peak.bin_idx] += subj_mass / dui;
        }

        let mut subjective_intensity = vec![0.0f32; nsgt_power.len()];
        for i in 0..nsgt_power.len() {
            let y = a * self.norm_state[i] + (1.0 - a) * subj_density_delta[i];
            self.norm_state[i] = y;
            subjective_intensity[i] = y;
        }

        let loudness_mass = density::density_to_mass(&subjective_intensity, &self.du);

        SpectralFrame {
            subjective_intensity,
            loudness_mass,
            peaks_raw: peaks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
    use crate::core::log2space::Log2Space;
    use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};
    use crate::core::landscape::{LandscapeParams, RoughnessScalarMode};

    fn build_params(fs: f32, space: &Log2Space) -> LandscapeParams {
        LandscapeParams {
            fs,
            max_hist_cols: 1,
            alpha: 0.0,
            roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005),
            harmonicity_kernel: HarmonicityKernel::new(space, HarmonicityParams::default()),
            roughness_scalar_mode: RoughnessScalarMode::Total,
            roughness_half: 0.1,
            habituation_tau: 1.0,
            habituation_weight: 0.0,
            habituation_max_depth: 1.0,
            loudness_exp: 1.0,
            ref_power: 1.0,
            tau_ms: 1.0,
            roughness_k: 1.0,
        }
    }

    fn build_frontend(fs: f32) -> (SpectralFrontEnd, LandscapeParams) {
        let space = Log2Space::new(200.0, 4000.0, 12);
        let params = build_params(fs, &space);
        let frontend = SpectralFrontEnd::new(space, &params);
        (frontend, params)
    }

    #[test]
    fn loudness_scales_with_power_input() {
        let fs = 48_000.0;
        let (mut frontend, params) = build_frontend(fs);
        let n = frontend.space.n_bins();
        let mut base = vec![0.0f32; n];
        let peak = n / 2;
        base[peak] = 1.0;
        let scaled: Vec<f32> = base.iter().map(|v| v * 4.0).collect();

        let out1 = frontend.process_nsgt_power(&base, 1.0, &params);
        let mut frontend2 = SpectralFrontEnd::new(frontend.space.clone(), &params);
        let out2 = frontend2.process_nsgt_power(&scaled, 1.0, &params);

        let v1 = out1.subjective_intensity[peak];
        assert!(v1 > 1e-6, "unexpected near-zero normalized power");
        let ratio = out2.subjective_intensity[peak] / v1;
        assert!(
            (ratio - 4.0).abs() < 0.1,
            "expected ~4x scaling for power input, got {ratio:.3}"
        );
    }

    #[test]
    fn a_weighting_uses_power_gain() {
        let fs = 48_000.0;
        let (frontend, _params) = build_frontend(fs);
        let i = frontend.loudness_weights_pow.len() / 2;
        let f = frontend.space.centers_hz[i];
        let expected = a_weighting::a_weighting_gain_pow(f);
        let got = frontend.loudness_weights_pow[i];
        assert!(
            (got - expected).abs() < 1e-6,
            "expected A-weighting power gain, got {got:.6} vs {expected:.6}"
        );
    }
}
