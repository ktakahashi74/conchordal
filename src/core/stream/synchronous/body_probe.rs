use super::*;
use crate::core::auditory_frontend::SpectralFrameView;
use crate::core::psycho_state::h_pot_scan_to_h_state01_scan;

impl SynchronousAnalysis {
    pub(crate) fn density<'a>(
        &'a mut self,
        audio: &[f32],
        params: &LandscapeParams,
    ) -> SpectralFrameView<'a> {
        assert_eq!(audio.len(), self.nsgt.hop(), "one complete analysis hop");
        #[cfg(test)]
        if let Some(legacy) = &mut self.legacy {
            return legacy.process_nsgt_power(
                self.nsgt.process_hop(audio),
                audio.len() as f32 / params.fs,
                params,
            );
        }
        self.frontend.process(audio, params);
        SpectralFrameView {
            subjective_intensity: &self.frontend.density_scan,
        }
    }

    pub(crate) fn skip(&mut self, audio: &[f32]) {
        #[cfg(test)]
        if self.legacy.is_some() {
            self.nsgt.push_without_analysis(audio);
            return;
        }
        self.frontend.skip(audio);
    }

    pub(crate) fn process_body_gap(&mut self, audio: &[f32], gap: i32, params: &LandscapeParams) {
        assert!(gap > 0);
        assert_eq!(audio.len(), self.nsgt.hop(), "one complete analysis hop");
        self.process_spectral(audio, gap, params, false);
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
}
