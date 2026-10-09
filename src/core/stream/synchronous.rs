//! Borrowed synchronous analysis, with the same numerical law as AnalysisStream.

use crate::core::landscape::{Landscape, LandscapeParams};
use crate::core::landscape_spectral::{SpectralFrameView, SpectralFrontEnd};
use crate::core::nsgt_rt::RtNsgtKernelLog2;
#[cfg(test)]
use crate::core::psycho_state::roughness_ratio_to_state01;
use crate::core::psycho_state::{
    compute_roughness_reference, h_pot_scan_to_h_state01_scan, r_pot_scan_to_r_state01_scan,
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
    #[cfg(test)]
    ref_total: f32,
    ref_peak: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_only_analysis_and_birth_state_match_full_oracle_bits() {
        for gap in [1, 2, 4, 8, 18] {
            let (nsgt, mut params) = crate::runtime::body_metabolism_test_core();
            let mut full_shared = SynchronousAnalysis::new(nsgt, &params);
            let mut density_shared = full_shared.clone();
            let mut full = full_shared.clone();
            let mut body = density_shared.clone();
            let mut audio = vec![0.; full.nsgt.hop()];
            let mut habituation = vec![0.; full.landscape.space.n_bins()];
            let mut last = 0;
            for frame in 0..(2 * gap + 5) {
                let hop = audio.len();
                for (i, sample) in audio.iter_mut().enumerate() {
                    let t = (frame as usize * hop + i) as f32 / params.fs;
                    *sample = if frame < 2 {
                        0.
                    } else {
                        0.13 * (t * std::f32::consts::TAU * 440.).sin()
                            + 0.09 * (t * std::f32::consts::TAU * 603.).sin()
                    };
                }
                if frame == 3 {
                    params.loudness_exp = 0.71;
                    params.ref_power *= 1.3;
                    params.tau_ms *= 0.8;
                }
                full_shared.process(&audio, &params);
                density_shared.density(&audio, &params);
                if frame == 3 {
                    // Newborn environments inherit continuous analysis state.
                    full = full_shared.clone();
                    body = density_shared.clone();
                    last = frame + 1;
                }
                let end = frame + 1;
                if end - last >= gap || frame == 0 {
                    full.process_gap(&audio, (end - last).max(1), &params);
                    body.process_body_gap(&audio, (end - last).max(1), &params);
                    last = end;
                    assert_eq!(
                        full.landscape
                            .harmonicity
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        body.landscape
                            .harmonicity
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                    );
                    assert_eq!(
                        full.landscape
                            .roughness_shape_raw
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        body.landscape
                            .roughness_shape_raw
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                    );
                    assert_eq!(
                        full.landscape
                            .roughness01
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        body.landscape
                            .roughness01
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                    );
                    for enabled in [false, true] {
                        params.consonance_kernel.a = 0.8 + frame as f32 * 0.01;
                        params.consonance_representation.theta = 0.12;
                        for (i, value) in habituation.iter_mut().enumerate() {
                            *value = (i % 7) as f32 * 0.2;
                        }
                        full.landscape.recompute_consonance(&params);
                        if enabled {
                            full.landscape.apply_habituation(
                                &habituation,
                                params.consonance_representation.theta,
                                &params.consonance_representation,
                            );
                        }
                        body.recompute_body_score(
                            &params,
                            enabled.then_some(habituation.as_slice()),
                        );
                        for (reference, actual) in [
                            (&full.landscape.harmonicity01, &body.landscape.harmonicity01),
                            (
                                &full.landscape.consonance_field_score_eff,
                                &body.landscape.consonance_field_score_eff,
                            ),
                        ] {
                            assert_eq!(
                                reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                            );
                        }
                        let reference = crate::life::body_fitness::evaluate(
                            &full.landscape.space,
                            &full.normalized,
                            &full.du,
                            &full.landscape.consonance_field_score_eff,
                            &params.consonance_representation,
                        );
                        let actual = crate::life::body_fitness::evaluate(
                            &body.landscape.space,
                            &body.normalized,
                            &body.du,
                            &body.landscape.consonance_field_score_eff,
                            &params.consonance_representation,
                        );
                        match (reference, actual) {
                            (Ok(a), Ok(b)) => assert_eq!(
                                [
                                    a.score.to_bits(),
                                    a.level.to_bits(),
                                    a.in_band_mass.to_bits()
                                ],
                                [
                                    b.score.to_bits(),
                                    b.level.to_bits(),
                                    b.in_band_mass.to_bits()
                                ]
                            ),
                            (Err(a), Err(b)) => assert_eq!(a, b),
                            _ => panic!("body fitness availability changed"),
                        }
                    }
                } else {
                    full.skip(&audio);
                    body.skip(&audio);
                }
            }
        }
    }
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
            #[cfg(test)]
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

    #[cfg(test)]
    pub(crate) fn process(&mut self, audio: &[f32], params: &LandscapeParams) {
        self.process_gap(audio, 1, params);
    }

    pub(crate) fn skip(&mut self, audio: &[f32]) {
        self.nsgt.push_without_analysis(audio);
    }

    pub(crate) fn process_body_gap(&mut self, audio: &[f32], gap: i32, params: &LandscapeParams) {
        assert_eq!(audio.len(), self.nsgt.hop(), "one complete analysis hop");
        let power = self.nsgt.process_hop_gap(audio, gap);
        let spectral = self.frontend.process_nsgt_power_reuse(
            power,
            audio.len() as f32 / params.fs * gap as f32,
            params,
        );
        let density = spectral.subjective_intensity;
        let space = &self.landscape.space;
        space.assert_scan_len_named(density, "subjective_intensity_scan");
        params.harmonicity_kernel.potential_h_into(
            density,
            space,
            &mut self.smeared,
            &mut self.roots,
            &mut self.landscape.harmonicity,
        );
        let eps = params.roughness_ref_eps.max(1e-12);
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
        if mass > eps {
            params.roughness_kernel.potential_r_into(
                &self.normalized,
                space,
                &self.erb,
                &self.du,
                &mut self.landscape.roughness_shape_raw,
            );
        } else {
            self.landscape.roughness_shape_raw.fill(0.);
        }
        r_pot_scan_to_r_state01_scan(
            &self.landscape.roughness_shape_raw,
            self.ref_peak.max(eps),
            params.roughness_k.max(1e-6),
            &mut self.landscape.roughness01,
        );
    }

    pub(crate) fn recompute_body_score(
        &mut self,
        params: &LandscapeParams,
        habituation: Option<&[f32]>,
    ) {
        let landscape = &mut self.landscape;
        landscape
            .space
            .assert_scan_len_named(&landscape.harmonicity, "h_pot_scan");
        landscape
            .space
            .assert_scan_len_named(&landscape.roughness01, "r_state01_scan");
        landscape
            .space
            .assert_scan_len_named(&landscape.consonance_field_score, "c_score_scan");
        landscape
            .space
            .assert_scan_len_named(&landscape.consonance_field_score_eff, "c_score_eff_scan");
        if let Some(h) = habituation {
            landscape
                .space
                .assert_scan_len_named(h, "habituation_state_scan");
        }
        h_pot_scan_to_h_state01_scan(&landscape.harmonicity, 1., &mut landscape.harmonicity01);
        let kernel = params.consonance_field_kernel();
        for i in 0..landscape.space.n_bins() {
            let h01 = crate::core::float::sanitize01(landscape.harmonicity01[i]);
            let r01 = crate::core::float::sanitize01(landscape.roughness01[i]);
            let score = kernel.score(h01, r01);
            landscape.consonance_field_score[i] = score;
            landscape.consonance_field_score_eff[i] = match habituation {
                Some(h) => crate::core::habituation::erode_score(
                    score,
                    h[i],
                    params.consonance_representation.theta,
                ),
                None => score,
            };
        }
    }

    // Keep the pre-optimization numerical path as the test oracle.
    #[cfg(test)]
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
