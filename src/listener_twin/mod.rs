use crate::core::float::sanitize_nonnegative_finite;
use crate::core::landscape::LandscapeFrame;
use crate::core::log2space::Log2Space;
use crate::core::meter::{MeterNetwork, MeterState};
use crate::core::roughness_kernel::erb_grid;
use crate::core::stream::dorsal::{DorsalMetrics, DorsalStream};

mod contour;
pub(crate) use contour::ContourEvent;
use contour::ContourObserver;

const ATTENTION_ATTACK_TAU_SEC: f32 = 0.04;
const ATTENTION_RELEASE_TAU_SEC: f32 = 0.18;
const AUDIBLE_EVIDENCE_EPS: f32 = 1e-7;
const NEUTRAL_STABILITY_LEVEL: f32 = 0.5;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ListenerState {
    pub(crate) time_sec: f32,
    pub(crate) generated_frame_id: u64,
    pub(crate) analysis_frame_id: u64,
    pub(crate) analysis_lag_frames: u64,
    pub(crate) stability_level: f32,
    pub(crate) resolvability_level: f32,
    pub(crate) tension_level: f32,
    pub(crate) attention_level: f32,
    pub(crate) beat_hz: f32,
    pub(crate) beat_phase: f32,
    pub(crate) beat_confidence: f32,
    pub(crate) subdivision_ratio: u8,
    pub(crate) subdivision_confidence: f32,
    pub(crate) measure_hz: f32,
    pub(crate) measure_ratio: u8,
    pub(crate) measure_confidence: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ListenerFastState {
    pub(crate) time_sec: f32,
    pub(crate) generated_frame_id: u64,
    pub(crate) attention_level: f32,
    pub(crate) attention_metrics: DorsalMetrics,
    pub(crate) meter_state: MeterState,
    pub(crate) has_state: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ListenerTwinConfig {
    pub(crate) reachable_cents: f32,
    pub(crate) movement_cost_per_oct: f32,
    pub(crate) gain_scale: f32,
}

impl Default for ListenerTwinConfig {
    fn default() -> Self {
        Self {
            reachable_cents: 220.0,
            movement_cost_per_oct: 0.45,
            gain_scale: 0.20,
        }
    }
}

pub(crate) struct ListenerTwin {
    config: ListenerTwinConfig,
    fast_stream: DorsalStream,
    meter: MeterNetwork,
    fast_state: ListenerFastState,
    erb_grid_key: Option<(u32, u32, u32, usize)>,
    erb_du_scan: Vec<f32>,
    contour: Option<ContourObserver>,
    pub(crate) spectral_history:
        Option<std::sync::Arc<crate::core::spectral_history::SpectralHistorySnapshot>>,
}

impl ListenerTwin {
    pub(crate) fn new(config: ListenerTwinConfig) -> Self {
        Self::with_sample_rate(48_000.0, config)
    }

    pub(crate) fn with_sample_rate(fs: f32, config: ListenerTwinConfig) -> Self {
        let fast_stream = DorsalStream::new(fs);
        Self {
            config,
            fast_stream,
            meter: MeterNetwork::new(),
            fast_state: ListenerFastState::default(),
            erb_grid_key: None,
            erb_du_scan: Vec::new(),
            contour: None,
            spectral_history: None,
        }
    }

    pub(crate) fn enable_contour_reporting(&mut self, fs: f32) -> bool {
        self.contour = ContourObserver::new(fs);
        self.contour.is_some()
    }

    pub(crate) fn observe_contour_audio(
        &mut self,
        start_sample: u64,
        audio: &[f32],
        emit: impl FnMut(ContourEvent),
    ) {
        if let Some(observer) = self.contour.as_mut() {
            observer.process(start_sample, audio, emit);
        }
    }

    pub(crate) fn observe_presentation_audio(
        &mut self,
        time_sec: f32,
        generated_frame_id: u64,
        audio: &[f32],
    ) -> ListenerFastState {
        self.fast_stream.process(audio);
        let attention_metrics = self.fast_stream.last_metrics();
        let salience_level = bottom_up_salience_level_from_metrics(attention_metrics);
        let dt_sec = if self.fast_state.has_state {
            (time_sec - self.fast_state.time_sec).max(0.0)
        } else {
            0.0
        };
        let attention_level = if self.fast_state.has_state {
            smooth_attention_level(self.fast_state.attention_level, salience_level, dt_sec)
        } else {
            salience_level
        };
        let meter_state = self.meter.process(dt_sec, salience_level);
        self.fast_state = ListenerFastState {
            time_sec,
            generated_frame_id,
            attention_level,
            attention_metrics,
            meter_state,
            has_state: true,
        };
        self.fast_state
    }

    pub(crate) fn observe_presentation_landscape(
        &mut self,
        time_sec: f32,
        generated_frame_id: u64,
        analysis_frame_id: u64,
        landscape: &LandscapeFrame,
    ) -> ListenerState {
        self.spectral_history = landscape.spectral_history.clone();
        let space = &landscape.space;
        space.assert_scan_len_named(&landscape.subjective_intensity, "subjective_intensity");
        space.assert_scan_len_named(
            &landscape.consonance_field_level_eff,
            "consonance_field_level_eff",
        );
        let key = (
            space.fmin.to_bits(),
            space.fmax.to_bits(),
            space.bins_per_oct,
            space.n_bins(),
        );
        if self.erb_grid_key != Some(key) {
            // The front end stores mass per ERB, so quadrature must use its grid widths.
            self.erb_du_scan = erb_grid(space).1;
            self.erb_grid_key = Some(key);
        }
        space.assert_scan_len_named(&self.erb_du_scan, "listener_erb_du_scan");
        let (stability_level, resolvability_level, tension_level) =
            if has_audible_evidence(landscape, &self.erb_du_scan) {
                let stability_level = weighted_stability_level(landscape, &self.erb_du_scan);
                let resolution_gain =
                    weighted_resolution_gain(landscape, &self.erb_du_scan, &self.config);
                let resolvability_level =
                    (resolution_gain / self.config.gain_scale.max(1e-6)).clamp(0.0, 1.0);
                let tension_level = ((1.0 - stability_level) * resolvability_level).clamp(0.0, 1.0);
                (stability_level, resolvability_level, tension_level)
            } else {
                (NEUTRAL_STABILITY_LEVEL, 0.0, 0.0)
            };

        ListenerState {
            time_sec,
            generated_frame_id,
            analysis_frame_id,
            analysis_lag_frames: generated_frame_id.saturating_sub(analysis_frame_id),
            stability_level,
            resolvability_level,
            tension_level,
            attention_level: self.fast_state.attention_level,
            beat_hz: sanitize_nonnegative_finite(self.fast_state.meter_state.beat.freq_hz),
            beat_phase: self.fast_state.meter_state.beat.phase,
            beat_confidence: self.fast_state.meter_state.beat.confidence.clamp(0.0, 1.0),
            subdivision_ratio: self.fast_state.meter_state.subdivision_ratio,
            subdivision_confidence: self
                .fast_state
                .meter_state
                .subdivision
                .confidence
                .clamp(0.0, 1.0),
            measure_hz: sanitize_nonnegative_finite(self.fast_state.meter_state.measure.freq_hz),
            measure_ratio: self.fast_state.meter_state.measure_ratio,
            measure_confidence: self
                .fast_state
                .meter_state
                .measure
                .confidence
                .clamp(0.0, 1.0),
        }
    }
}

impl Default for ListenerTwin {
    fn default() -> Self {
        Self::new(ListenerTwinConfig::default())
    }
}

fn has_audible_evidence(landscape: &LandscapeFrame, erb_du: &[f32]) -> bool {
    landscape
        .subjective_intensity
        .iter()
        .zip(erb_du)
        .map(|(&density, &du)| sanitize_nonnegative_finite(density * du))
        .sum::<f32>()
        > AUDIBLE_EVIDENCE_EPS
}

fn weighted_stability_level(landscape: &LandscapeFrame, erb_du: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut weight_sum = 0.0f32;
    for ((&value, &density), &du) in landscape
        .consonance_field_level_eff
        .iter()
        .zip(&landscape.subjective_intensity)
        .zip(erb_du)
    {
        let w = sanitize_nonnegative_finite(density * du);
        if w <= 0.0 {
            continue;
        }
        sum += w * value.clamp(0.0, 1.0);
        weight_sum += w;
    }

    if weight_sum > 0.0 {
        (sum / weight_sum).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn weighted_resolution_gain(
    landscape: &LandscapeFrame,
    erb_du: &[f32],
    config: &ListenerTwinConfig,
) -> f32 {
    let levels = &landscape.consonance_field_level_eff;
    let weights = &landscape.subjective_intensity;

    let window_bins = reachable_window_bins(&landscape.space, config.reachable_cents);
    let mut gain_sum = 0.0f32;
    let mut weight_sum = 0.0f32;

    for (idx, ((&level, &density), &du)) in
        levels.iter().zip(weights.iter()).zip(erb_du).enumerate()
    {
        let w = sanitize_nonnegative_finite(density * du);
        if w <= 0.0 {
            continue;
        }

        let current = level.clamp(0.0, 1.0);
        let lo = idx.saturating_sub(window_bins);
        let hi = (idx + window_bins + 1).min(levels.len());
        let mut best_gain = 0.0f32;

        for (offset, &candidate) in levels[lo..hi].iter().enumerate() {
            let target_idx = lo + offset;
            let dist_oct = target_idx.abs_diff(idx) as f32 / landscape.space.bins_per_oct as f32;
            let movement_cost = config.movement_cost_per_oct.max(0.0) * dist_oct;
            let gain = candidate.clamp(0.0, 1.0) - current - movement_cost;
            if gain.is_finite() {
                best_gain = best_gain.max(gain);
            }
        }

        gain_sum += w * best_gain.max(0.0);
        weight_sum += w;
    }

    if weight_sum > 0.0 {
        gain_sum / weight_sum
    } else {
        0.0
    }
}

fn reachable_window_bins(space: &Log2Space, reachable_cents: f32) -> usize {
    let cents_per_bin = 1200.0 / space.bins_per_oct.max(1) as f32;
    (reachable_cents.max(0.0) / cents_per_bin).ceil().max(1.0) as usize
}

fn bottom_up_salience_level_from_metrics(metrics: DorsalMetrics) -> f32 {
    (sanitize_nonnegative_finite(metrics.flux) * 500.0)
        .tanh()
        .clamp(0.0, 1.0)
}

fn smooth_attention_level(prev: f32, target: f32, dt_sec: f32) -> f32 {
    if !dt_sec.is_finite() || dt_sec <= 0.0 {
        return target.clamp(0.0, 1.0);
    }
    let tau = if target > prev {
        ATTENTION_ATTACK_TAU_SEC
    } else {
        ATTENTION_RELEASE_TAU_SEC
    };
    let alpha = 1.0 - (-dt_sec / tau).exp();
    (prev + (target - prev) * alpha).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::landscape::Landscape;

    fn test_landscape(default_level: f32, active: &[(usize, f32, f32)]) -> Landscape {
        test_landscape_with_space(Log2Space::new(100.0, 400.0, 12), default_level, active)
    }

    fn test_landscape_with_space(
        space: Log2Space,
        default_level: f32,
        active: &[(usize, f32, f32)],
    ) -> Landscape {
        let (_, du) = erb_grid(&space);
        let mut landscape = Landscape::new(space);
        landscape.consonance_field_level.fill(default_level);
        landscape.consonance_field_level_eff.fill(default_level);
        landscape.consonance_field_score.fill(default_level);
        landscape.subjective_intensity.fill(0.0);
        for &(idx, level, mass) in active {
            landscape.consonance_field_level[idx] = level;
            landscape.consonance_field_level_eff[idx] = level;
            landscape.consonance_field_score[idx] = level;
            landscape.subjective_intensity[idx] = mass / du[idx];
        }
        landscape
    }

    #[test]
    #[should_panic(expected = "scan length mismatch: subjective_intensity")]
    fn observation_rejects_density_scan_length_mismatch() {
        let mut landscape = test_landscape(0.0, &[]);
        landscape.subjective_intensity.pop();
        ListenerTwin::default().observe_presentation_landscape(0.0, 0, 0, &landscape);
    }

    #[test]
    #[should_panic(expected = "scan length mismatch: consonance_field_level_eff")]
    fn silent_observation_rejects_level_scan_length_mismatch() {
        let mut landscape = test_landscape(0.0, &[]);
        landscape.consonance_field_level_eff.pop();
        ListenerTwin::default().observe_presentation_landscape(0.0, 0, 0, &landscape);
    }

    #[test]
    #[should_panic(expected = "scan length mismatch: listener_erb_du_scan")]
    fn observation_rejects_cached_grid_width_length_mismatch() {
        let landscape = test_landscape(0.0, &[]);
        let mut twin = ListenerTwin::default();
        twin.observe_presentation_landscape(0.0, 0, 0, &landscape);
        twin.erb_du_scan.pop();
        twin.observe_presentation_landscape(0.01, 1, 1, &landscape);
    }

    #[test]
    fn equal_mass_at_different_frequencies_contributes_equally_to_stability() {
        let landscape = test_landscape(0.0, &[(2, 0.2, 1.0), (20, 0.8, 1.0)]);
        assert_ne!(
            landscape.subjective_intensity[2],
            landscape.subjective_intensity[20]
        );
        let state = ListenerTwin::default().observe_presentation_landscape(0.0, 0, 0, &landscape);

        assert!((state.stability_level - 0.5).abs() < 1e-6);
    }

    #[test]
    fn equal_mass_at_different_frequencies_contributes_equally_to_resolution() {
        let landscape = test_landscape(
            0.0,
            &[(2, 0.0, 1.0), (3, 0.2, 0.0), (20, 0.0, 1.0), (21, 0.8, 0.0)],
        );
        let mut twin = ListenerTwin::new(ListenerTwinConfig {
            reachable_cents: 100.0,
            movement_cost_per_oct: 0.0,
            gain_scale: 1.0,
        });
        let state = twin.observe_presentation_landscape(0.0, 0, 0, &landscape);

        assert!((state.resolvability_level - 0.5).abs() < 1e-6);
    }

    #[test]
    fn audible_evidence_threshold_uses_mass_instead_of_density_sum() {
        let below = test_landscape(0.2, &[(2, 0.2, AUDIBLE_EVIDENCE_EPS * 0.5)]);
        assert!(below.subjective_intensity[2] > AUDIBLE_EVIDENCE_EPS);
        let mut twin = ListenerTwin::default();
        let silent = twin.observe_presentation_landscape(0.0, 0, 0, &below);
        assert_eq!(silent.stability_level, NEUTRAL_STABILITY_LEVEL);

        let above = test_landscape_with_space(
            Log2Space::new(1000.0, 16_000.0, 1),
            0.2,
            &[(2, 0.2, AUDIBLE_EVIDENCE_EPS * 2.0)],
        );
        assert!(above.subjective_intensity[2] < AUDIBLE_EVIDENCE_EPS);
        let audible = twin.observe_presentation_landscape(0.01, 1, 1, &above);
        assert!((audible.stability_level - 0.2).abs() < 1e-6);
    }

    #[test]
    fn grid_width_cache_updates_for_same_length_space_with_different_frequencies() {
        let low = test_landscape(0.0, &[(2, 0.2, 1.0), (20, 0.8, 1.0)]);
        let high = test_landscape_with_space(
            Log2Space::new(400.0, 1600.0, 12),
            0.0,
            &[(2, 0.2, 1.0), (20, 0.8, 1.0)],
        );
        assert_eq!(low.space.n_bins(), high.space.n_bins());
        let mut twin = ListenerTwin::default();
        twin.observe_presentation_landscape(0.0, 0, 0, &low);
        let old_widths = twin.erb_du_scan.clone();
        let state = twin.observe_presentation_landscape(0.01, 1, 1, &high);

        assert_ne!(twin.erb_du_scan, old_widths);
        assert!((state.stability_level - 0.5).abs() < 1e-6);
        let cached_ptr = twin.erb_du_scan.as_ptr();
        twin.observe_presentation_landscape(0.02, 2, 2, &high);
        assert_eq!(twin.erb_du_scan.as_ptr(), cached_ptr);
    }

    #[test]
    fn stable_presentation_has_low_tension() {
        let landscape = test_landscape(0.90, &[(12, 0.90, 1.0)]);
        let mut twin = ListenerTwin::default();

        let state = twin.observe_presentation_landscape(0.0, 10, 8, &landscape);

        assert!(state.stability_level > 0.89);
        assert!(state.resolvability_level < 0.05);
        assert!(state.tension_level < 0.05);
        assert_eq!(state.analysis_lag_frames, 2);
    }

    #[test]
    fn silent_presentation_has_neutral_stability_and_no_tension() {
        let landscape = test_landscape(0.0, &[]);
        let mut twin = ListenerTwin::default();

        let state = twin.observe_presentation_landscape(0.0, 0, 0, &landscape);

        assert_eq!(state.stability_level, NEUTRAL_STABILITY_LEVEL);
        assert_eq!(state.resolvability_level, 0.0);
        assert_eq!(state.tension_level, 0.0);
    }

    #[test]
    fn nearby_stable_target_increases_resolvability() {
        let landscape = test_landscape(0.20, &[(12, 0.20, 1.0), (14, 0.90, 0.0)]);
        let mut twin = ListenerTwin::new(ListenerTwinConfig {
            reachable_cents: 240.0,
            movement_cost_per_oct: 0.05,
            gain_scale: 0.20,
        });

        let state = twin.observe_presentation_landscape(0.0, 0, 0, &landscape);

        assert!(state.stability_level < 0.25);
        assert!(state.resolvability_level > 0.9);
        assert!(state.tension_level > 0.7);
    }

    #[test]
    fn unstable_without_reachable_target_is_not_tension() {
        let landscape = test_landscape(0.20, &[(12, 0.20, 1.0), (20, 0.90, 0.0)]);
        let mut twin = ListenerTwin::new(ListenerTwinConfig {
            reachable_cents: 120.0,
            movement_cost_per_oct: 0.05,
            gain_scale: 0.20,
        });

        let state = twin.observe_presentation_landscape(0.0, 0, 0, &landscape);

        assert!(state.stability_level < 0.25);
        assert!(state.resolvability_level < 0.05);
        assert!(state.tension_level < 0.05);
    }

    #[test]
    fn presentation_audio_transient_raises_attention() {
        let mut twin = ListenerTwin::with_sample_rate(48_000.0, ListenerTwinConfig::default());
        let quiet = vec![0.0f32; 256];
        let quiet_state = twin.observe_presentation_audio(0.0, 0, &quiet);

        let mut transient = vec![0.0f32; 256];
        transient[0] = 1.0;
        let active_state = twin.observe_presentation_audio(0.01, 1, &transient);

        assert!(quiet_state.attention_level < 0.01);
        assert!(active_state.attention_level > quiet_state.attention_level);
        assert!(active_state.has_state);
    }

    #[test]
    fn presentation_attention_decays_smoothly_after_transient() {
        let mut twin = ListenerTwin::with_sample_rate(48_000.0, ListenerTwinConfig::default());
        let quiet = vec![0.0f32; 256];
        twin.observe_presentation_audio(0.0, 0, &quiet);

        let mut transient = vec![0.0f32; 256];
        transient[0] = 1.0;
        let active_state = twin.observe_presentation_audio(0.01, 1, &transient);
        let decay_state = twin.observe_presentation_audio(0.02, 2, &quiet);

        assert!(decay_state.attention_level > 0.0);
        assert!(decay_state.attention_level < active_state.attention_level);
    }

    #[test]
    fn listener_rhythm_uses_internal_attention_vitality() {
        let mut twin = ListenerTwin::with_sample_rate(48_000.0, ListenerTwinConfig::default());
        let quiet = vec![0.0f32; 256];
        let state = twin.observe_presentation_audio(0.0, 0, &quiet);

        assert!(state.attention_level < 0.01);
        assert!(state.has_state);
    }
}
