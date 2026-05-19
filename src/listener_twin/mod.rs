use crate::core::landscape::LandscapeFrame;
use crate::core::log2space::Log2Space;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct ListenerState {
    pub(crate) time_sec: f32,
    pub(crate) generated_frame_id: u64,
    pub(crate) analysis_frame_id: u64,
    pub(crate) analysis_lag_frames: u64,
    pub(crate) stability_level: f32,
    pub(crate) resolvability_level: f32,
    pub(crate) tension_level: f32,
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

#[derive(Clone, Debug)]
pub(crate) struct ListenerTwin {
    config: ListenerTwinConfig,
}

impl ListenerTwin {
    pub(crate) fn new(config: ListenerTwinConfig) -> Self {
        Self { config }
    }

    pub(crate) fn observe_presentation_landscape(
        &mut self,
        time_sec: f32,
        generated_frame_id: u64,
        analysis_frame_id: u64,
        landscape: &LandscapeFrame,
    ) -> ListenerState {
        let stability_level = weighted_stability_level(landscape);
        let resolution_gain = weighted_resolution_gain(landscape, &self.config);
        let resolvability_level =
            (resolution_gain / self.config.gain_scale.max(1e-6)).clamp(0.0, 1.0);
        let tension_level = ((1.0 - stability_level) * resolvability_level).clamp(0.0, 1.0);

        ListenerState {
            time_sec,
            generated_frame_id,
            analysis_frame_id,
            analysis_lag_frames: generated_frame_id.saturating_sub(analysis_frame_id),
            stability_level,
            resolvability_level,
            tension_level,
        }
    }
}

impl Default for ListenerTwin {
    fn default() -> Self {
        Self::new(ListenerTwinConfig::default())
    }
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
}
