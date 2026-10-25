use super::synchronous::SynchronousAnalysis;
use crate::core::landscape::{Landscape, LandscapeParams, LandscapeUpdate};
use crate::core::nsgt_rt::RtNsgtKernelLog2;

/// Spectral analysis shared with the body's synchronous perception path.
#[derive(Clone)]
pub struct AnalysisStream {
    analysis: SynchronousAnalysis,
    params: LandscapeParams,
    onset_habituation: Option<crate::core::onset_habituation::OnsetHabituation>,
    next_sample: u64,
}

impl AnalysisStream {
    #[cfg(test)]
    pub(crate) fn aud_m1_shadow_diagnostic(&self) -> serde_json::Value {
        self.analysis.frontend.diagnostic()
    }

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
        Self {
            analysis,
            params,
            onset_habituation: None,
            next_sample: 0,
        }
    }

    pub(crate) fn enable_onset_habituation(&mut self, period: f64) {
        let mut state = crate::core::onset_habituation::OnsetHabituation::new(
            &self.analysis.landscape.space,
            self.analysis.frontend.band_energy().0,
            self.params.habituation,
            period,
            self.params.fs,
        );
        state.settling_samples = self.analysis.frontend.band_settling_samples();
        self.onset_habituation = Some(state);
        self.analysis.landscape.onset_habituation_state_scan =
            vec![0.0; self.analysis.landscape.space.n_bins()].into();
    }

    pub(crate) fn gap_to(&mut self, sample: u64) {
        let window = self.window_samples();
        let hop = self.hop_samples();
        if let Some(state) = &mut self.onset_habituation {
            state.missing(sample.saturating_sub(self.next_sample));
            state.restart_frontend(window, self.analysis.frontend.band_hop_counts(), hop);
        }
        self.analysis.reset(&self.params);
        self.next_sample = sample;
        self.analysis.landscape.onset_habituation_known = false;
    }

    pub(crate) fn enable_resource_power(&mut self) {
        self.analysis.nsgt.enable_resource_power();
    }

    pub(crate) fn with_nsgt_power(mut self, enabled: bool) -> Self {
        self.analysis.nsgt_enabled = enabled;
        self
    }

    pub(crate) fn nsgt_power_enabled(&self) -> bool {
        self.analysis.nsgt_enabled
    }

    pub(crate) fn voice_observer(&self) -> Self {
        let mut observer = Self {
            analysis: self.analysis.clone(),
            params: self.params.clone(),
            onset_habituation: None,
            next_sample: self.next_sample,
        };
        observer.analysis.landscape.onset_habituation_state_scan = std::sync::Arc::from([]);
        observer.analysis.landscape.onset_habituation_known = false;
        // Voice observers consume H/R only, even when the mixed stream feeds Flora.
        observer.analysis.nsgt.disable_resource_power();
        observer.analysis.nsgt_enabled = false;
        #[cfg(test)]
        if self.analysis.legacy.is_some() {
            observer.analysis.nsgt_enabled = true;
        }
        observer
    }

    pub(crate) fn resource_observer(&mut self) -> RtNsgtKernelLog2 {
        self.analysis.nsgt.enable_resource_power();
        self.analysis.nsgt.clone()
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

    /// Process one complete hop, or hold state on empty input.
    /// Callers clone the borrowed frame only when publishing it.
    pub fn process(&mut self, audio: &[f32]) -> &Landscape {
        if !audio.is_empty() {
            assert_eq!(audio.len(), self.hop_samples(), "one complete analysis hop");
            self.analysis.process_spectral(audio, 1, &self.params, true);
            self.next_sample = self.next_sample.saturating_add(audio.len() as u64);
            if let Some(state) = &mut self.onset_habituation {
                state.process(
                    self.analysis.frontend.band_energy().1,
                    audio.len(),
                    self.next_sample,
                );
                let frame = &mut self.analysis.landscape;
                if frame.onset_habituation_state_scan.len() != state.state.len() {
                    frame.onset_habituation_state_scan = state.state.clone().into();
                } else {
                    std::sync::Arc::make_mut(&mut frame.onset_habituation_state_scan)
                        .copy_from_slice(&state.state);
                }
                frame.onset_habituation_end = self.next_sample;
                frame.onset_habituation_known = state.known();
            }
        }
        &self.analysis.landscape
    }

    pub fn reset(&mut self) {
        let window = self.window_samples();
        let hop = self.hop_samples();
        if let Some(state) = &mut self.onset_habituation {
            state.reset();
            state.restart_frontend(window, self.analysis.frontend.band_hop_counts(), hop);
        }
        self.analysis.reset(&self.params);
        self.next_sample = 0;
        self.analysis.landscape.onset_habituation_known = false;
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
    fn onset_habituation_has_one_optional_owner_and_reset_invalidates_publication() {
        let (nsgt, params) = crate::runtime::body_metabolism_test_core();
        let mut stream = AnalysisStream::new(params, nsgt).with_nsgt_power(false);
        assert!(stream.onset_habituation.is_none());
        assert!(stream.last().onset_habituation_state_scan.is_empty());
        stream.enable_onset_habituation(0.5);
        let silence = vec![0.0; stream.hop_samples()];
        assert!(!stream.last().onset_habituation_known);
        stream.process(&silence);
        assert!(stream.last().onset_habituation_known);
        let observer = stream.voice_observer();
        assert!(observer.onset_habituation.is_none());
        assert!(observer.last().onset_habituation_state_scan.is_empty());
        assert!(!observer.last().onset_habituation_known);
        stream.reset();
        assert!(!stream.last().onset_habituation_known);
        assert!(
            stream
                .onset_habituation
                .as_ref()
                .unwrap()
                .state
                .iter()
                .all(|&x| x == 0.0)
        );
        stream.process(&silence);
        assert!(!stream.last().onset_habituation_known);
    }

    #[test]
    fn disabling_nsgt_preserves_perception_and_does_not_advance_its_state() {
        let (nsgt, params) = crate::runtime::body_metabolism_test_core();
        let mut frozen = nsgt.clone();
        let mut enabled = AnalysisStream::new(params.clone(), nsgt.clone());
        let mut disabled = AnalysisStream::new(params.clone(), nsgt).with_nsgt_power(false);
        let mut last_audio = Vec::new();
        for frame in 0..80 {
            let audio: Vec<_> = (0..enabled.hop_samples())
                .map(|n| {
                    let t = (frame * enabled.hop_samples() + n) as f32 / params.fs;
                    if (8..64).contains(&frame) {
                        0.1 * (std::f32::consts::TAU * 110.0 * t).cos()
                            + 0.07 * (std::f32::consts::TAU * 150.0 * t).sin()
                    } else {
                        0.0
                    }
                })
                .collect();
            let mut expected = enabled.process(&audio).clone();
            let mut actual = disabled.process(&audio).clone();
            expected.recompute_consonance(&params);
            actual.recompute_consonance(&params);
            for (a, b) in [
                (&actual.subjective_intensity, &expected.subjective_intensity),
                (&actual.harmonicity, &expected.harmonicity),
                (&actual.roughness, &expected.roughness),
                (&actual.roughness01, &expected.roughness01),
                (
                    &actual.consonance_field_score,
                    &expected.consonance_field_score,
                ),
                (
                    &actual.consonance_density_mass,
                    &expected.consonance_density_mass,
                ),
            ] {
                assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
            }
            assert!(actual.nsgt_power.iter().all(|&p| p == 0.0));
            assert_eq!(
                crate::life::community::Community::aud_m1_respawn(&actual),
                crate::life::community::Community::aud_m1_respawn(&expected)
            );
            for range in [(55.0, 8000.0), (110.0, 880.0), (440.0, 3520.0)] {
                assert_eq!(
                    crate::life::community::Community::aud_m1_range_mass(&actual, range),
                    crate::life::community::Community::aud_m1_range_mass(&expected, range)
                );
            }
            let anchors = [110.0_f32.log2(), 150.0_f32.log2(), 220.0_f32.log2()];
            for &a in &anchors {
                let (chosen, _) =
                    crate::life::voice::pitch_core::aud_m1_hill_choice(&actual, a, &anchors);
                let (reference, _) =
                    crate::life::voice::pitch_core::aud_m1_hill_choice(&expected, a, &anchors);
                assert_eq!(chosen.to_bits(), reference.to_bits());
            }
            last_audio = audio;
        }
        // A hidden push_without_analysis would make this next NSGT output differ.
        assert_eq!(
            disabled.analysis.nsgt.process_hop(&last_audio),
            frozen.process_hop(&last_audio)
        );
    }

    #[test]
    fn voice_observers_drop_resource_work_without_changing_perception() {
        let (nsgt, params) = crate::runtime::body_metabolism_test_core();
        let mut mixed = AnalysisStream::new(params, nsgt);
        mixed.enable_resource_power();
        let audio: Vec<_> = (0..mixed.hop_samples())
            .map(|i| 0.03 * (i as f32 * 0.11).sin())
            .collect();
        for _ in 0..40 {
            mixed.process(&audio);
        }
        assert!(mixed.analysis.nsgt.resource_power().is_some());
        let mut voice = mixed.voice_observer();
        assert!(voice.analysis.nsgt.resource_power().is_none());
        let retained_power = voice.last().nsgt_power.clone();
        for _ in 0..40 {
            assert!(!voice.analysis.nsgt_enabled);
            let expected = mixed.process(&audio);
            let actual = voice.process(&audio);
            assert_eq!(actual.nsgt_power, retained_power);
            assert_eq!(actual.harmonicity, expected.harmonicity);
            assert_eq!(actual.roughness01, expected.roughness01);
        }
    }

    #[test]
    fn incomplete_hops_are_rejected_before_state_changes() {
        let (nsgt, params) = crate::runtime::body_metabolism_test_core();
        let mut stream = AnalysisStream::new(params, nsgt);
        let hop = stream.hop_samples();
        for len in [hop / 2, 2 * hop + 1] {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    stream.process(&vec![0.1; len]);
                }))
                .is_err()
            );
        }
        let mut fresh = stream.clone();
        stream.reset();
        let audio = vec![0.03; hop];
        assert_eq!(
            stream.process(&audio).subjective_intensity,
            fresh.process(&audio).subjective_intensity
        );
    }

    #[test]
    fn complete_hops_reset_and_empty_input_preserve_analysis_contract() {
        let (mut nsgt, mut params) = crate::runtime::body_metabolism_test_core();
        nsgt.process_hop(&vec![0.1; nsgt.hop()]);
        let mut reference = nsgt.clone();
        let hop = nsgt.hop();
        let mut stream = AnalysisStream::new(params.clone(), nsgt);
        for len in [hop, hop, hop] {
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

#[cfg(test)]
mod onset_habituation_tests {
    use super::*;
    use crate::core::onset_habituation::OnsetHabituation;

    #[test]
    fn short_worker_window_and_band_boxcars_exclude_reset_transients() {
        use crate::core::nsgt_kernel::{NsgtKernelLog2, NsgtLog2Config, PowerMode};
        let mut config = crate::config::AppConfig::default();
        config.analysis.nfft = 512;
        config.analysis.hop_size = 512;
        config.validate().unwrap();
        let (_, mut params) = crate::runtime::body_metabolism_test_core();
        params.habituation.enabled = true;
        let space = crate::core::log2space::Log2Space::new(55.0, 8000.0, 96);
        let kernel = NsgtKernelLog2::new(
            NsgtLog2Config {
                fs: params.fs,
                overlap: 0.0,
                nfft_override: Some(config.analysis.nfft),
                ..Default::default()
            },
            space,
            None,
            PowerMode::Coherent,
        );
        let mut stream = AnalysisStream::new(params.clone(), RtNsgtKernelLog2::new(kernel))
            .with_nsgt_power(false);
        let hop = stream.hop_samples();
        assert_eq!(hop, 512);
        assert_eq!(stream.window_samples(), config.analysis.nfft);
        let mut observer = OnsetHabituation::new(
            &stream.last().space,
            stream.analysis.frontend.band_energy().0,
            params.habituation,
            0.5,
            params.fs,
        );
        let mut audio = vec![0.0; hop];
        observer.settling_samples = stream.analysis.frontend.band_settling_samples();
        let mut attacks = Vec::new();
        for frame in 0..1100 {
            if (1000..1010).contains(&frame) {
                continue;
            }
            if frame == 1010 {
                observer.missing(10 * hop as u64);
                stream.reset();
                observer.restart_frontend(
                    stream.window_samples(),
                    stream.analysis.frontend.band_hop_counts(),
                    hop,
                );
            }
            for (i, value) in audio.iter_mut().enumerate() {
                let sample = frame * hop as u64 + i as u64;
                *value = (0.1
                    * (std::f64::consts::TAU * 440.0 * sample as f64 / f64::from(params.fs)).sin())
                    as f32;
            }
            stream.process(&audio);
            let end = (frame + 1) * hop as u64;
            observer.process(stream.analysis.frontend.band_energy().1, hop, end);
            if frame >= 1010 {
                for channel in observer.attacks_at(end) {
                    let recovered_hops = frame - 1009;
                    let hz = stream.analysis.frontend.band_energy().0[channel];
                    let boxcar_hops = stream.analysis.frontend.band_hop_counts()[channel];
                    println!(
                        "short-window attack channel={channel} hz={hz} recovered_hops={recovered_hops} boxcar_hops={boxcar_hops} window_samples={}",
                        stream.window_samples()
                    );
                    attacks.push((channel, recovered_hops));
                }
            }
        }
        assert!(
            attacks.is_empty(),
            "supported short window retained reset transients: {attacks:?}"
        );
    }

    // Reusing the worker window excludes reconstruction transients from evidence.
    #[test]
    fn continuous_pcm_after_gap_does_not_create_an_attack() {
        let (nsgt, mut params) = crate::runtime::body_metabolism_test_core();
        params.habituation.enabled = true;
        let fs = params.fs;
        let mut stream = AnalysisStream::new(params.clone(), nsgt).with_nsgt_power(false);
        let mut observer = OnsetHabituation::new(
            &stream.last().space,
            stream.analysis.frontend.band_energy().0,
            params.habituation,
            0.5,
            fs,
        );
        let hop = stream.hop_samples();
        let mut audio = vec![0.0; hop];
        observer.settling_samples = stream.analysis.frontend.band_settling_samples();
        let make_audio = |audio: &mut [f32], start: u64| {
            for (i, value) in audio.iter_mut().enumerate() {
                *value = (0.1
                    * (std::f64::consts::TAU * 440.0 * (start + i as u64) as f64 / f64::from(fs))
                        .sin()) as f32;
            }
        };
        for frame in 0..1000 {
            make_audio(&mut audio, frame * hop as u64);
            stream.process(&audio);
            observer.process(
                stream.analysis.frontend.band_energy().1,
                hop,
                (frame + 1) * hop as u64,
            );
        }
        let before = observer.state.clone();
        let mut continuous = stream.clone();
        let mut continuous_observer = observer.clone();
        for frame in 1000..1010 {
            make_audio(&mut audio, frame * hop as u64);
            continuous.process(&audio);
        }
        continuous_observer.missing(10 * hop as u64);
        observer.missing(10 * hop as u64);
        stream.reset();
        observer.restart_frontend(
            stream.window_samples(),
            stream.analysis.frontend.band_hop_counts(),
            hop,
        );
        continuous_observer.restart_frontend(
            continuous.window_samples(),
            continuous.analysis.frontend.band_hop_counts(),
            hop,
        );
        assert_eq!(before, observer.state);
        let mut rises = Vec::new();
        for frame in 1010..1210 {
            make_audio(&mut audio, frame * hop as u64);
            stream.process(&audio);
            continuous.process(&audio);
            observer.process(
                stream.analysis.frontend.band_energy().1,
                hop,
                (frame + 1) * hop as u64,
            );
            continuous_observer.process(
                continuous.analysis.frontend.band_energy().1,
                hop,
                (frame + 1) * hop as u64,
            );
            assert!(
                continuous_observer
                    .state
                    .iter()
                    .zip(&before)
                    .all(|(a, b)| a <= b)
            );
            if (frame - 1010) * (hop as u64) < stream.window_samples() as u64 {
                assert_eq!(
                    observer.state, before,
                    "recovery window must freeze the scan"
                );
            }
            for channel in observer.attacks_at((frame + 1) * hop as u64) {
                println!(
                    "unexpected attack channel={channel} hz={} recovered_hops={} window_samples={} boxcar_hops={}",
                    stream.analysis.frontend.band_energy().0[channel],
                    frame - 1009,
                    stream.window_samples(),
                    stream.analysis.frontend.band_hop_counts()[channel]
                );
            }
            let center = stream
                .analysis
                .frontend
                .band_energy()
                .0
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| (*a - 440.0).abs().total_cmp(&(*b - 440.0).abs()))
                .unwrap()
                .0;
            println!(
                "band_hz={} reset_energy={} continuous_energy={}",
                stream.analysis.frontend.band_energy().0[center],
                stream.analysis.frontend.band_energy().1[center],
                continuous.analysis.frontend.band_energy().1[center]
            );
            let largest_rise = observer
                .state
                .iter()
                .zip(&before)
                .map(|(a, b)| a - b)
                .fold(0.0_f32, f32::max);
            println!(
                "gap recovery frame={frame} end={} largest_rise={largest_rise}",
                (frame + 1) * hop as u64
            );
            rises.push(largest_rise);
        }
        assert_eq!(
            rises[0], 0.0,
            "the first post-gap hop must not use pre-gap energy"
        );
        assert!(
            rises.iter().all(|&rise| rise == 0.0),
            "unchanged stationary PCM gained attack habituation after a gap"
        );
    }
}
