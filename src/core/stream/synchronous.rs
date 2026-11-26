//! Borrowed synchronous analysis, with the same numerical law as AnalysisStream.

#[cfg(any(test, feature = "research"))]
#[path = "synchronous/body_probe.rs"]
mod body_probe;

use super::potentials::PotentialScratch;
#[cfg(not(test))]
use crate::core::auditory_frontend::AuditoryFrontEnd;
#[cfg(test)]
use crate::core::auditory_frontend::ShadowFrontEnd as AuditoryFrontEnd;
use crate::core::landscape::{Landscape, LandscapeParams};
#[cfg(test)]
use crate::core::landscape_spectral::SpectralFrontEnd;
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::core::psycho_state::roughness_ratio_to_state01;
use crate::core::psycho_state::{compute_roughness_reference, r_pot_scan_to_r_state01_scan};
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct SynchronousAnalysis {
    pub(super) nsgt: RtNsgtKernelLog2,
    pub(super) frontend: AuditoryFrontEnd,
    pub(super) nsgt_enabled: bool,
    #[cfg(test)]
    pub(super) legacy: Option<SpectralFrontEnd>,
    pub landscape: Landscape,
    pub du: Arc<[f32]>,
    pub(super) potentials: PotentialScratch,
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
            frontend: AuditoryFrontEnd::selected(space.clone(), params.fs, nsgt.hop(), nsgt.nfft()),
            nsgt_enabled: true,
            #[cfg(test)]
            legacy: None,
            landscape,
            du: Arc::clone(&potentials.du),
            potentials,
            ref_total: reference.total,
            ref_peak: reference.peak,
            nsgt,
        }
    }

    #[cfg(test)]
    pub(crate) fn legacy_new(nsgt: RtNsgtKernelLog2, params: &LandscapeParams) -> Self {
        let frontend = SpectralFrontEnd::new(nsgt.space().clone());
        let mut analysis = Self::new(nsgt, params);
        analysis.legacy = Some(frontend);
        analysis
    }

    pub(crate) fn reset_density(&mut self) {
        self.nsgt.reset();
        self.frontend.reset();
        #[cfg(test)]
        if let Some(legacy) = &mut self.legacy {
            legacy.reset();
        }
    }

    #[cfg(test)]
    pub(crate) fn process(&mut self, audio: &[f32], params: &LandscapeParams) {
        self.process_gap(audio, 1, params);
    }

    pub(super) fn process_spectral(
        &mut self,
        audio: &[f32],
        gap: i32,
        params: &LandscapeParams,
        full: bool,
    ) {
        if full && self.nsgt_enabled {
            let power = if gap == 1 {
                self.nsgt.process_hop(audio)
            } else {
                self.nsgt.process_hop_gap(audio, gap)
            };
            self.landscape.acoustic.nsgt_power.copy_from_slice(power);
        }
        #[cfg(test)]
        let density = if let Some(legacy) = &mut self.legacy {
            let power = if full {
                &self.landscape.acoustic.nsgt_power
            } else {
                self.nsgt.process_hop_gap(audio, gap)
            };
            legacy
                .process_nsgt_power(power, audio.len() as f32 / params.fs * gap as f32, params)
                .subjective_intensity
        } else {
            if full && gap == 1 {
                self.frontend.process(audio, params);
            } else {
                self.frontend.process_body_gap(audio, gap as usize, params);
            }
            &self.frontend.density_scan
        };
        #[cfg(not(test))]
        let density = {
            if full && gap == 1 {
                self.frontend.process(audio, params);
            } else {
                self.frontend.process_body_gap(audio, gap as usize, params);
            }
            &self.frontend.density_scan
        };
        let space = &self.landscape.acoustic.space;
        let eps = params.roughness_ref_eps.max(1e-12);
        let roughness_k = params.roughness_k.max(1e-6);
        let shape_total = self.potentials.compute_shape(
            density,
            space,
            (&params.harmonicity_kernel, &params.roughness_kernel),
            eps,
            &mut self.landscape.acoustic.harmonicity,
            &mut self.landscape.acoustic.roughness_shape_raw,
        );
        r_pot_scan_to_r_state01_scan(
            &self.landscape.acoustic.roughness_shape_raw,
            self.ref_peak.max(eps),
            roughness_k,
            &mut self.landscape.acoustic.roughness01,
        );
        if full {
            self.potentials.geometry.potential_r_into(
                density,
                space,
                &self.du,
                &mut self.landscape.acoustic.roughness,
            );
            self.landscape.acoustic.loudness_exp = params.loudness_exp.max(0.01);
            self.landscape
                .acoustic
                .subjective_intensity
                .copy_from_slice(density);
            self.landscape.acoustic.loudness_mass =
                crate::core::density::density_to_mass(density, &self.du);
            self.landscape.acoustic.roughness01_scalar =
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

    /// Take `process_spectral`'s perception from a stream that analyzed bit-identical
    /// PCM with the same front-end history. The own front end does not advance, so the
    /// caller adopts the source's front end before the inputs differ.
    pub(super) fn copy_spectral(&mut self, source: &Self, params: &LandscapeParams) {
        let (to, from) = (&mut self.landscape, &source.landscape);
        to.acoustic
            .harmonicity
            .copy_from_slice(&from.acoustic.harmonicity);
        to.acoustic
            .roughness_shape_raw
            .copy_from_slice(&from.acoustic.roughness_shape_raw);
        to.acoustic
            .roughness01
            .copy_from_slice(&from.acoustic.roughness01);
        to.acoustic
            .roughness
            .copy_from_slice(&from.acoustic.roughness);
        to.acoustic
            .subjective_intensity
            .copy_from_slice(&from.acoustic.subjective_intensity);
        to.acoustic.loudness_exp = params.loudness_exp.max(0.01);
        to.acoustic.loudness_mass = from.acoustic.loudness_mass;
        to.acoustic.roughness01_scalar = from.acoustic.roughness01_scalar;
        to.set_analysis_metadata(
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
    fn auditory_frontend_preserves_nsgt_and_selected_history() {
        let (params, kernel, hop) =
            crate::runtime::analysis_runtime_test_core(&crate::config::AppConfig::default(), 48000);
        let mut baseline = SynchronousAnalysis::legacy_new(kernel.clone(), &params);
        let mut selected = crate::core::auditory_shadow::with_selected_test_frontend(|| {
            SynchronousAnalysis::new(kernel.clone(), &params)
        });
        let mut direct = crate::core::auditory_frontend::AuditoryFrontEnd::selected(
            kernel.space().clone(),
            params.fs,
            hop,
            kernel.nfft(),
        );
        let assert_density_bits = |actual: &[f32], expected: &[f32]| {
            assert!(
                actual
                    .iter()
                    .map(|density| density.to_bits())
                    .eq(expected.iter().map(|density| density.to_bits()))
            );
        };
        for frame in 0..16 {
            let pcm: Vec<_> = (0..hop)
                .map(|i| {
                    let t = (frame * hop + i) as f32 / params.fs;
                    0.1 * (std::f32::consts::TAU * 110.0 * t).cos()
                        + 0.05 * (std::f32::consts::TAU * 150.0 * t).cos()
                })
                .collect();
            baseline.process(&pcm, &params);
            selected.process(&pcm, &params);
            direct.process(&pcm, &params);
            assert_density_bits(
                &selected.landscape.acoustic.subjective_intensity,
                &direct.density_scan,
            );
            assert_eq!(
                selected.landscape.acoustic.nsgt_power,
                baseline.landscape.acoustic.nsgt_power
            );
        }
        assert_ne!(
            selected.landscape.acoustic.subjective_intensity,
            baseline.landscape.acoustic.subjective_intensity
        );
        let mut copy = selected.clone();
        let silence = vec![0.0; hop];
        selected.process(&silence, &params);
        copy.process(&silence, &params);
        assert_eq!(
            selected.landscape.acoustic.subjective_intensity,
            copy.landscape.acoustic.subjective_intensity
        );

        selected.reset_density();
        direct.reset();
        let pcm: Vec<_> = (0..hop).map(|i| 0.1 * (i as f32 * 0.13).sin()).collect();
        for frame in 0..kernel.nfft().div_ceil(hop) + 4 {
            if frame % 4 != 3 {
                selected.skip(&pcm);
                direct.skip(&pcm);
                continue;
            }
            assert_eq!(
                selected.frontend.process_body_gap(&pcm, 4, &params),
                direct.process_body_gap(&pcm, 4, &params)
            );
            assert_density_bits(&selected.frontend.density_scan, &direct.density_scan);
        }
    }

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
                (
                    &actual.acoustic.nsgt_power,
                    &body.landscape.acoustic.nsgt_power,
                ),
                (
                    &actual.acoustic.subjective_intensity,
                    &body.landscape.acoustic.subjective_intensity,
                ),
                (
                    &actual.acoustic.harmonicity,
                    &body.landscape.acoustic.harmonicity,
                ),
                (
                    &actual.acoustic.roughness,
                    &body.landscape.acoustic.roughness,
                ),
                (
                    &actual.acoustic.roughness_shape_raw,
                    &body.landscape.acoustic.roughness_shape_raw,
                ),
                (
                    &actual.acoustic.roughness01,
                    &body.landscape.acoustic.roughness01,
                ),
                (
                    &actual.acoustic.harmonicity,
                    &body_only.landscape.acoustic.harmonicity,
                ),
                (
                    &actual.acoustic.roughness_shape_raw,
                    &body_only.landscape.acoustic.roughness_shape_raw,
                ),
                (
                    &actual.acoustic.roughness01,
                    &body_only.landscape.acoustic.roughness01,
                ),
            ] {
                assert_eq!(
                    a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "frame {frame}"
                );
            }
            assert_eq!(
                actual.acoustic.roughness01_scalar.to_bits(),
                body.landscape.acoustic.roughness01_scalar.to_bits()
            );
            assert_eq!(
                actual.acoustic.loudness_mass.to_bits(),
                body.landscape.acoustic.loudness_mass.to_bits()
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
            let mut habituation = vec![0.; full.landscape.acoustic.space.n_bins()];
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
                            .acoustic
                            .harmonicity
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        body.landscape
                            .acoustic
                            .harmonicity
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                    );
                    assert_eq!(
                        full.landscape
                            .acoustic
                            .roughness_shape_raw
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        body.landscape
                            .acoustic
                            .roughness_shape_raw
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                    );
                    assert_eq!(
                        full.landscape
                            .acoustic
                            .roughness01
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        body.landscape
                            .acoustic
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
                            (
                                &full.landscape.acoustic.harmonicity01,
                                &body.landscape.acoustic.harmonicity01,
                            ),
                            (
                                &full.landscape.acoustic.consonance_field_score_eff,
                                &body.landscape.acoustic.consonance_field_score_eff,
                            ),
                        ] {
                            assert_eq!(
                                reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                            );
                        }
                        let reference = crate::life::body_fitness::evaluate(
                            &full.landscape.acoustic.space,
                            &full.potentials.normalized,
                            &full.du,
                            &full.landscape.acoustic.consonance_field_score_eff,
                            &params.consonance_representation,
                        );
                        let actual = crate::life::body_fitness::evaluate(
                            &body.landscape.acoustic.space,
                            &body.potentials.normalized,
                            &body.du,
                            &body.landscape.acoustic.consonance_field_score_eff,
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
