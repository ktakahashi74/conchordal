use crate::core::landscape::LandscapeFrame;
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::core::stream::dorsal::{DorsalMetrics, DorsalStream};
use serde::Serialize;

const ATTENTION_ATTACK_TAU_SEC: f32 = 0.04;
const ATTENTION_RELEASE_TAU_SEC: f32 = 0.18;
const AUDIBLE_EVIDENCE_EPS: f32 = 1e-7;
const NEUTRAL_STABILITY_LEVEL: f32 = 0.5;

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct ListenerState {
    pub(crate) time_sec: f32,
    pub(crate) generated_frame_id: u64,
    pub(crate) analysis_frame_id: u64,
    pub(crate) analysis_lag_frames: u64,
    pub(crate) stability_level: f32,
    pub(crate) resolvability_level: f32,
    pub(crate) tension_level: f32,
    pub(crate) attention_level: f32,
    pub(crate) theta_hz: f32,
    pub(crate) theta_mag: f32,
    pub(crate) theta_alpha: f32,
    pub(crate) delta_hz: f32,
    pub(crate) delta_mag: f32,
    pub(crate) delta_alpha: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ListenerFastState {
    pub(crate) time_sec: f32,
    pub(crate) generated_frame_id: u64,
    pub(crate) attention_level: f32,
    pub(crate) attention_metrics: DorsalMetrics,
    pub(crate) neural_rhythms: NeuralRhythms,
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
    fast_state: ListenerFastState,
}

impl ListenerTwin {
    pub(crate) fn new(config: ListenerTwinConfig) -> Self {
        Self::with_sample_rate(48_000.0, config)
    }

    pub(crate) fn with_sample_rate(fs: f32, config: ListenerTwinConfig) -> Self {
        let mut fast_stream = DorsalStream::new(fs);
        fast_stream.set_vitality(0.0);
        Self {
            config,
            fast_stream,
            fast_state: ListenerFastState::default(),
        }
    }

    pub(crate) fn observe_presentation_audio(
        &mut self,
        time_sec: f32,
        generated_frame_id: u64,
        audio: &[f32],
    ) -> ListenerFastState {
        let listener_rhythm_vitality = if self.fast_state.has_state {
            self.fast_state.attention_level
        } else {
            0.0
        };
        self.fast_stream.set_vitality(listener_rhythm_vitality);
        let neural_rhythms = self.fast_stream.process(audio);
        let attention_metrics = self.fast_stream.last_metrics();
        let salience_level = bottom_up_salience_level_from_metrics(attention_metrics);
        let attention_level = if self.fast_state.has_state {
            let dt_sec = (time_sec - self.fast_state.time_sec).max(0.0);
            smooth_attention_level(self.fast_state.attention_level, salience_level, dt_sec)
        } else {
            salience_level
        };
        self.fast_state = ListenerFastState {
            time_sec,
            generated_frame_id,
            attention_level,
            attention_metrics,
            neural_rhythms,
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
        let (stability_level, resolvability_level, tension_level) =
            if has_audible_evidence(landscape) {
                let stability_level = weighted_stability_level(landscape);
                let resolution_gain = weighted_resolution_gain(landscape, &self.config);
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
            theta_hz: finite_positive(self.fast_state.neural_rhythms.theta.freq_hz),
            theta_mag: self.fast_state.neural_rhythms.theta.mag.clamp(0.0, 1.0),
            theta_alpha: self.fast_state.neural_rhythms.theta.alpha.clamp(0.0, 1.0),
            delta_hz: finite_positive(self.fast_state.neural_rhythms.delta.freq_hz),
            delta_mag: self.fast_state.neural_rhythms.delta.mag.clamp(0.0, 1.0),
            delta_alpha: self.fast_state.neural_rhythms.delta.alpha.clamp(0.0, 1.0),
        }
    }
}

impl Default for ListenerTwin {
    fn default() -> Self {
        Self::new(ListenerTwinConfig::default())
    }
}

fn has_audible_evidence(landscape: &LandscapeFrame) -> bool {
    if landscape.subjective_intensity.len() != landscape.space.n_bins() {
        return false;
    }
    landscape
        .subjective_intensity
        .iter()
        .map(|&weight| finite_nonnegative(weight))
        .sum::<f32>()
        > AUDIBLE_EVIDENCE_EPS
}

fn weighted_stability_level(landscape: &LandscapeFrame) -> f32 {
    weighted_scan_mean(
        &landscape.consonance_field_level,
        &landscape.subjective_intensity,
    )
}

fn weighted_resolution_gain(landscape: &LandscapeFrame, config: &ListenerTwinConfig) -> f32 {
    let levels = &landscape.consonance_field_level;
    let weights = &landscape.subjective_intensity;
    if levels.is_empty()
        || levels.len() != weights.len()
        || levels.len() != landscape.space.n_bins()
    {
        return 0.0;
    }

    let window_bins = reachable_window_bins(&landscape.space, config.reachable_cents);
    let mut gain_sum = 0.0f32;
    let mut weight_sum = 0.0f32;

    for (idx, (&level, &weight)) in levels.iter().zip(weights.iter()).enumerate() {
        let w = finite_nonnegative(weight);
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

fn weighted_scan_mean(values: &[f32], weights: &[f32]) -> f32 {
    if values.is_empty() || values.len() != weights.len() {
        return 0.0;
    }

    let mut sum = 0.0f32;
    let mut weight_sum = 0.0f32;
    for (&value, &weight) in values.iter().zip(weights.iter()) {
        let w = finite_nonnegative(weight);
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

fn reachable_window_bins(space: &Log2Space, reachable_cents: f32) -> usize {
    let cents_per_bin = 1200.0 / space.bins_per_oct.max(1) as f32;
    (reachable_cents.max(0.0) / cents_per_bin).ceil().max(1.0) as usize
}

fn finite_nonnegative(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

fn finite_positive(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

fn bottom_up_salience_level_from_metrics(metrics: DorsalMetrics) -> f32 {
    (finite_nonnegative(metrics.flux) * 500.0)
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
        let space = Log2Space::new(100.0, 400.0, 12);
        let mut landscape = Landscape::new(space);
        landscape.consonance_field_level.fill(default_level);
        landscape.consonance_field_score.fill(default_level);
        landscape.subjective_intensity.fill(0.0);
        for &(idx, level, weight) in active {
            landscape.consonance_field_level[idx] = level;
            landscape.consonance_field_score[idx] = level;
            landscape.subjective_intensity[idx] = weight;
        }
        landscape
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
