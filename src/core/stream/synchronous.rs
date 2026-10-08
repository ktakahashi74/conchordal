//! Borrowed synchronous analysis, with the same numerical law as AnalysisStream.

use crate::core::landscape::{Landscape, LandscapeParams};
use crate::core::landscape_spectral::{SpectralFrameView, SpectralFrontEnd};
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::core::psycho_state::{
    compute_roughness_reference, r_pot_scan_to_r_state01_scan, roughness_ratio_to_state01,
};
use crate::core::roughness_kernel::erb_grid;

#[derive(Clone)]
pub(crate) struct SynchronousAnalysis {
    nsgt: RtNsgtKernelLog2,
    frontend: SpectralFrontEnd,
    pub landscape: Landscape,
    pub du: Vec<f32>,
    erb: Vec<f32>,
    normalized: Vec<f32>,
    smeared: Vec<f32>,
    roots: Vec<f32>,
    ref_total: f32,
    ref_peak: f32,
}

impl SynchronousAnalysis {
    pub(crate) fn new(nsgt: RtNsgtKernelLog2, params: &LandscapeParams) -> Self {
        let space = nsgt.space();
        let bins = space.n_bins();
        let reference = compute_roughness_reference(params, space);
        let (erb, du) = erb_grid(space);
        let mut landscape = Landscape::new(space.clone());
        landscape.recompute_consonance(params);
        Self {
            frontend: SpectralFrontEnd::new(space.clone(), params),
            landscape,
            erb,
            du,
            normalized: vec![0.; bins],
            smeared: vec![0.; bins],
            roots: vec![0.; params.harmonicity_kernel.root_buffer_len(space)],
            ref_total: reference.total,
            ref_peak: reference.peak,
            nsgt,
        }
    }

    pub(crate) fn reset_density(&mut self) {
        self.nsgt.reset();
        self.frontend.reset();
    }

    pub(crate) fn density<'a>(
        &'a mut self,
        audio: &[f32],
        params: &LandscapeParams,
    ) -> SpectralFrameView<'a> {
        assert_eq!(audio.len(), self.nsgt.hop(), "one complete analysis hop");
        let power = self.nsgt.process_hop(audio);
        self.frontend
            .process_nsgt_power_reuse(power, audio.len() as f32 / params.fs, params)
    }

    pub(crate) fn process(&mut self, audio: &[f32], params: &LandscapeParams) {
        self.process_gap(audio, 1, params);
    }

    pub(crate) fn skip(&mut self, audio: &[f32]) {
        self.nsgt.push_without_analysis(audio);
    }

    pub(crate) fn process_gap(&mut self, audio: &[f32], gap: i32, params: &LandscapeParams) {
        assert_eq!(audio.len(), self.nsgt.hop(), "one complete analysis hop");
        let power = self.nsgt.process_hop_gap(audio, gap);
        self.landscape.nsgt_power.copy_from_slice(power);
        let spectral = self.frontend.process_nsgt_power_reuse(
            power,
            audio.len() as f32 / params.fs * gap as f32,
            params,
        );
        let density = spectral.subjective_intensity;
        let space = &self.landscape.space;
        space.assert_scan_len_named(density, "subjective_intensity_scan");
        let eps = params.roughness_ref_eps.max(1e-12);
        let roughness_k = params.roughness_k.max(1e-6);
        params.harmonicity_kernel.potential_h_into(
            density,
            space,
            &mut self.smeared,
            &mut self.roots,
            &mut self.landscape.harmonicity,
        );
        let total = params.roughness_kernel.potential_r_into(
            density,
            space,
            &self.erb,
            &self.du,
            &mut self.landscape.roughness,
        );
        let mass = density
            .iter()
            .zip(&self.du)
            .map(|(&value, &width)| {
                let value = if value.is_finite() { value.max(0.) } else { 0. };
                let width = if width.is_finite() { width.max(0.) } else { 0. };
                value * width
            })
            .sum::<f32>();
        self.normalized.fill(0.);
        if mass.is_finite() && mass > eps {
            let inv = 1. / (mass + eps);
            for (out, &value) in self.normalized.iter_mut().zip(density) {
                *out = if value.is_finite() {
                    value.max(0.) * inv
                } else {
                    0.
                };
            }
        }
        let shape_total = if mass > eps {
            params.roughness_kernel.potential_r_into(
                &self.normalized,
                space,
                &self.erb,
                &self.du,
                &mut self.landscape.roughness_shape_raw,
            )
        } else {
            self.landscape.roughness_shape_raw.fill(0.);
            0.
        };
        r_pot_scan_to_r_state01_scan(
            &self.landscape.roughness_shape_raw,
            self.ref_peak.max(eps),
            roughness_k,
            &mut self.landscape.roughness01,
        );
        self.landscape.subjective_intensity.copy_from_slice(density);
        self.landscape.loudness_mass = spectral.loudness_mass;
        self.landscape.roughness_total = total;
        self.landscape.roughness_scalar_raw = total;
        self.landscape.roughness_norm = total / (spectral.loudness_mass + eps);
        self.landscape.roughness01_scalar =
            roughness_ratio_to_state01(shape_total / self.ref_total.max(eps), roughness_k);
        self.landscape.roughness_suppress_sigma_erb =
            params.roughness_kernel.params.suppress_sigma_erb.max(1e-6);
        self.landscape.roughness_kernel_params = params.roughness_kernel.params;
        self.landscape.harmonicity_params = params.harmonicity_kernel.params;
        self.landscape.roughness_k = params.roughness_k;
        self.landscape.roughness_ref_peak = self.ref_peak;
        self.landscape.roughness_ref_eps = params.roughness_ref_eps;
        self.landscape.recompute_consonance(params);
    }
}
