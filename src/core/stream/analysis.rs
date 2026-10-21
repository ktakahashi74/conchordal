use super::synchronous::SynchronousAnalysis;
use crate::core::landscape::{Landscape, LandscapeParams, LandscapeUpdate};
use crate::core::nsgt_rt::RtNsgtKernelLog2;

/// Spectral analysis shared with the body's synchronous perception path.
#[cfg_attr(test, derive(Clone))]
pub struct AnalysisStream {
    analysis: SynchronousAnalysis,
    params: LandscapeParams,
}

impl AnalysisStream {
    pub fn new(params: LandscapeParams, nsgt_rt: RtNsgtKernelLog2) -> Self {
        let mut analysis = SynchronousAnalysis::new(nsgt_rt, &params);
        // Live analysis publishes raw perception; the worker derives consonance.
        let ref_peak = analysis.landscape.roughness_ref_peak;
        analysis.landscape = Landscape::new(analysis.nsgt.space().clone());
        analysis.landscape.set_analysis_metadata(
            params.roughness_kernel.params,
            params.harmonicity_kernel.params,
            params.consonance_field_kernel(),
            params.roughness_k,
            ref_peak,
            params.roughness_ref_eps,
        );
        Self { analysis, params }
    }

    pub(crate) fn fresh(&self) -> Self {
        let mut nsgt = self.analysis.nsgt.clone();
        nsgt.reset();
        Self::new(self.params.clone(), nsgt)
    }

    // Invert loudness on the observed ERB mass, retaining its peak and history.
    // Coherent NSGT power is A^2/4 for a centered sinusoid; Voice projection uses A^2.
    // Inverting a smoothed nonlinear density yields equivalent power, not raw power.
    pub(crate) fn observed_loo_power(&self) -> Vec<f64> {
        let frame = self.last();
        let du = &self.analysis.du;
        frame
            .subjective_intensity
            .iter()
            .enumerate()
            .map(|(i, &density)| {
                4. * (density as f64 * du[i] as f64)
                    .powf(1. / self.params.loudness_exp.max(0.01) as f64)
                    * self.params.ref_power.max(1e-12) as f64
                    / crate::core::utils::a_weighting_gain_pow(frame.space.centers_hz[i]) as f64
            })
            .collect()
    }

    pub fn last(&self) -> &Landscape {
        &self.analysis.landscape
    }
    pub(crate) fn window_samples(&self) -> usize {
        self.analysis.nsgt.nfft()
    }
    pub(crate) fn hop_samples(&self) -> usize {
        self.analysis.nsgt.hop()
    }
    pub(crate) fn sample_rate(&self) -> f32 {
        self.params.fs
    }

    /// Borrow the current frame; callers clone only when publishing it.
    pub fn process(&mut self, audio: &[f32]) -> &Landscape {
        if !audio.is_empty() {
            self.analysis.process_spectral(audio, 1, &self.params, true);
        }
        &self.analysis.landscape
    }

    pub fn reset(&mut self) {
        self.analysis.reset(&self.params);
    }

    pub fn apply_update(&mut self, upd: LandscapeUpdate) {
        if let Some(w) = upd.roughness_aversion {
            self.params.set_roughness_aversion(w);
        }
        if let Some(mode) = upd.pitch_objective_mode {
            self.analysis.landscape.pitch_objective_mode = mode;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_and_variable_chunks_preserve_analysis_contract() {
        let (mut nsgt, mut params) = crate::runtime::body_metabolism_test_core();
        nsgt.process_hop(&vec![0.1; nsgt.hop()]);
        let mut reference = nsgt.clone();
        let hop = nsgt.hop();
        let mut stream = AnalysisStream::new(params.clone(), nsgt);
        for len in [hop / 2, hop, 2 * hop + 1] {
            let audio: Vec<_> = (0..len).map(|i| 0.1 * (i as f32 * 0.13).sin()).collect();
            let expected = reference.process_hop(&audio);
            let frame = stream.process(&audio);
            assert_eq!(
                frame
                    .nsgt_power
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        let previous = stream.last().subjective_intensity.clone();
        assert_eq!(stream.process(&[]).subjective_intensity, previous);
        let update = LandscapeUpdate {
            roughness_aversion: Some(0.3),
            ..Default::default()
        };
        stream.apply_update(update);
        params.set_roughness_aversion(0.3);
        stream.reset();
        let (nsgt, _) = crate::runtime::body_metabolism_test_core();
        let mut fresh = AnalysisStream::new(params.clone(), nsgt);
        let audio: Vec<_> = (0..hop).map(|i| 0.1 * (i as f32 * 0.11).sin()).collect();
        let actual = stream.process(&audio);
        let expected = fresh.process(&audio);
        for (a, b) in [
            (&actual.subjective_intensity, &expected.subjective_intensity),
            (&actual.harmonicity, &expected.harmonicity),
            (&actual.roughness01, &expected.roughness01),
        ] {
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        assert_eq!(
            actual.consonance_kernel.score(0.4, 0.6).to_bits(),
            params.consonance_field_kernel().score(0.4, 0.6).to_bits()
        );
    }
}
