//! Borrowed synchronous analysis, with the same numerical law as AnalysisStream.

use super::potentials::PotentialScratch;
use crate::core::landscape::{Landscape, LandscapeParams};
use crate::core::landscape_spectral::{SpectralFrameView, SpectralFrontEnd};
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::core::psycho_state::roughness_ratio_to_state01;
use crate::core::psycho_state::{
    compute_roughness_reference, h_pot_scan_to_h_state01_scan, r_pot_scan_to_r_state01_scan,
};
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct SynchronousAnalysis {
    pub(super) nsgt: RtNsgtKernelLog2,
    frontend: SpectralFrontEnd,
    pub landscape: Landscape,
    pub du: Arc<[f32]>,
    potentials: PotentialScratch,
    ref_total: f32,
    ref_peak: f32,
}

impl SynchronousAnalysis {
    pub(crate) fn new(nsgt: RtNsgtKernelLog2, params: &LandscapeParams) -> Self {
        let space = nsgt.space();
        let reference = compute_roughness_reference(params, space);
        let potentials =
            PotentialScratch::new(space, &params.harmonicity_kernel, &params.roughness_kernel);
        let mut landscape = Landscape::new(space.clone());
        landscape.set_analysis_metadata(
            params.roughness_kernel.params,
            params.harmonicity_kernel.params,
            params.consonance_field_kernel(),
            params.roughness_k,
            reference.peak,
            params.roughness_ref_eps,
        );
        landscape.recompute_consonance(params);
        Self {
            frontend: SpectralFrontEnd::new(space.clone()),
            landscape,
            du: Arc::clone(&potentials.du),
            potentials,
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
            .process_nsgt_power(power, audio.len() as f32 / params.fs, params)
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
        self.process_spectral(audio, gap, params, false);
    }

    pub(super) fn process_spectral(
        &mut self,
        audio: &[f32],
        gap: i32,
        params: &LandscapeParams,
        full: bool,
    ) {
        let power = if gap == 1 {
            self.nsgt.process_hop(audio)
        } else {
            self.nsgt.process_hop_gap(audio, gap)
        };
        if full {
            self.landscape.nsgt_power.copy_from_slice(power);
        }
        let spectral = self.frontend.process_nsgt_power(
            power,
            audio.len() as f32 / params.fs * gap as f32,
            params,
        );
        let density = spectral.subjective_intensity;
        let space = &self.landscape.space;
        let eps = params.roughness_ref_eps.max(1e-12);
        let roughness_k = params.roughness_k.max(1e-6);
        let shape_total = self.potentials.compute_shape(
            density,
            space,
            (&params.harmonicity_kernel, &params.roughness_kernel),
            eps,
            &mut self.landscape.harmonicity,
            &mut self.landscape.roughness_shape_raw,
        );
        r_pot_scan_to_r_state01_scan(
            &self.landscape.roughness_shape_raw,
            self.ref_peak.max(eps),
            roughness_k,
            &mut self.landscape.roughness01,
        );
        if full {
            self.potentials.geometry.potential_r_into(
                density,
                space,
                &self.du,
                &mut self.landscape.roughness,
            );
            self.landscape.loudness_exp = params.loudness_exp.max(0.01);
            self.landscape.subjective_intensity.copy_from_slice(density);
            self.landscape.loudness_mass = crate::core::density::density_to_mass(density, &self.du);
            self.landscape.roughness01_scalar =
                roughness_ratio_to_state01(shape_total / self.ref_total.max(eps), roughness_k);
        }
        self.landscape.set_analysis_metadata(
            params.roughness_kernel.params,
            params.harmonicity_kernel.params,
            params.consonance_field_kernel(),
            params.roughness_k,
            self.ref_peak,
            params.roughness_ref_eps,
        );
    }

    pub(super) fn reset(&mut self, params: &LandscapeParams) {
        self.reset_density();
        let mode = self.landscape.pitch_objective_mode;
        self.landscape = Landscape::new(self.nsgt.space().clone());
        self.landscape.pitch_objective_mode = mode;
        self.landscape.set_analysis_metadata(
            params.roughness_kernel.params,
            params.harmonicity_kernel.params,
            params.consonance_field_kernel(),
            params.roughness_k,
            self.ref_peak,
            params.roughness_ref_eps,
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

    #[cfg(test)]
    pub(crate) fn process_gap(&mut self, audio: &[f32], gap: i32, params: &LandscapeParams) {
        assert_eq!(audio.len(), self.nsgt.hop(), "one complete analysis hop");
        self.process_spectral(audio, gap, params, true);
        self.landscape.recompute_consonance(params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rt_and_body_frontend_match_bits() {
        let (nsgt, params) = crate::runtime::body_metabolism_test_core();
        let mut rt =
            crate::core::stream::analysis::AnalysisStream::new(params.clone(), nsgt.clone());
        let mut body = SynchronousAnalysis::new(nsgt.clone(), &params);
        let mut body_only = SynchronousAnalysis::new(nsgt, &params);
        let mut audio = vec![0.; body.nsgt.hop()];
        for frame in 0..24 {
            let hop = audio.len();
            for (i, sample) in audio.iter_mut().enumerate() {
                let t = (frame * hop + i) as f32 / params.fs;
                *sample = if !(3..=18).contains(&frame) {
                    0.
                } else {
                    0.13 * (t * std::f32::consts::TAU * 440.).sin()
                        + 0.09 * (t * std::f32::consts::TAU * 603.).sin()
                };
            }
            let actual = rt.process(&audio);
            body.process_gap(&audio, 1, &params);
            body_only.process_body_gap(&audio, 1, &params);
            for (a, b) in [
                (&actual.nsgt_power, &body.landscape.nsgt_power),
                (
                    &actual.subjective_intensity,
                    &body.landscape.subjective_intensity,
                ),
                (&actual.harmonicity, &body.landscape.harmonicity),
                (&actual.roughness, &body.landscape.roughness),
                (
                    &actual.roughness_shape_raw,
                    &body.landscape.roughness_shape_raw,
                ),
                (&actual.roughness01, &body.landscape.roughness01),
                (&actual.harmonicity, &body_only.landscape.harmonicity),
                (
                    &actual.roughness_shape_raw,
                    &body_only.landscape.roughness_shape_raw,
                ),
                (&actual.roughness01, &body_only.landscape.roughness01),
            ] {
                assert_eq!(
                    a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "frame {frame}"
                );
            }
            assert_eq!(
                actual.roughness01_scalar.to_bits(),
                body.landscape.roughness01_scalar.to_bits()
            );
            assert_eq!(
                actual.loudness_mass.to_bits(),
                body.landscape.loudness_mass.to_bits()
            );
        }
    }

    #[test]
    fn body_only_analysis_and_birth_state_match_full_bits() {
        for gap in [1, 2, 4, 8, 18] {
            let (nsgt, mut params) = crate::runtime::body_metabolism_test_core();
            let mut full_shared = SynchronousAnalysis::new(nsgt, &params);
            let mut density_shared = full_shared.clone();
            assert!(Arc::ptr_eq(
                &full_shared.potentials.geometry,
                &density_shared.potentials.geometry
            ));
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
                if frame == 5 {
                    params.roughness_kernel = crate::core::roughness_kernel::RoughnessKernel::new(
                        crate::core::roughness_kernel::KernelParams {
                            half_width_erb: 2.5,
                            mix_tail: 0.2,
                            ..Default::default()
                        },
                        0.005,
                    );
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
                            &full.potentials.normalized,
                            &full.du,
                            &full.landscape.consonance_field_score_eff,
                            &params.consonance_representation,
                        );
                        let actual = crate::life::body_fitness::evaluate(
                            &body.landscape.space,
                            &body.potentials.normalized,
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
