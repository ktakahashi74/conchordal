use crate::config::DccConfig;
use crate::listener_twin::ListenerState;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ListenerPressure {
    pub(crate) tension_pressure: f32,
    pub(crate) temperature_bonus: f32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DccCoupler {
    coupling_strength: f32,
    max_temperature_bonus: f32,
}

impl DccCoupler {
    pub(crate) fn new(config: DccConfig) -> Self {
        Self {
            coupling_strength: sanitize_unit(config.coupling_strength),
            max_temperature_bonus: sanitize_nonnegative(config.max_temperature_bonus, 0.10),
        }
    }

    pub(crate) fn coupling_strength(&self) -> f32 {
        self.coupling_strength
    }

    pub(crate) fn pressure(&self, state: Option<ListenerState>) -> ListenerPressure {
        let Some(state) = state else {
            return ListenerPressure::default();
        };
        let tension_pressure = sanitize_unit(state.tension_level)
            * sanitize_unit(state.resolvability_level)
            * self.coupling_strength;
        ListenerPressure {
            tension_pressure,
            temperature_bonus: tension_pressure * self.max_temperature_bonus,
        }
    }
}

fn sanitize_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn sanitize_nonnegative(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn listener_state(tension_level: f32, resolvability_level: f32) -> ListenerState {
        ListenerState {
            time_sec: 0.0,
            generated_frame_id: 0,
            analysis_frame_id: 0,
            analysis_lag_frames: 0,
            stability_level: 0.5,
            resolvability_level,
            tension_level,
            attention_level: 0.0,
            beat_hz: 0.0,
            beat_phase: 0.0,
            beat_confidence: 0.0,
            subdivision_ratio: 0,
            subdivision_confidence: 0.0,
            measure_hz: 0.0,
            measure_ratio: 0,
            measure_confidence: 0.0,
        }
    }

    #[test]
    fn default_coupling_is_report_only() {
        let coupler = DccCoupler::new(DccConfig::default());
        let pressure = coupler.pressure(Some(listener_state(1.0, 1.0)));

        assert_eq!(pressure, ListenerPressure::default());
    }

    #[test]
    fn pressure_requires_tension_and_resolvability() {
        let coupler = DccCoupler::new(DccConfig {
            coupling_strength: 0.5,
            max_temperature_bonus: 0.2,
        });

        let pressure = coupler.pressure(Some(listener_state(0.8, 0.25)));

        assert!((pressure.tension_pressure - 0.1).abs() < 1e-6);
        assert!((pressure.temperature_bonus - 0.02).abs() < 1e-6);
    }

    #[test]
    fn pressure_sanitizes_inputs() {
        let coupler = DccCoupler::new(DccConfig {
            coupling_strength: 2.0,
            max_temperature_bonus: f32::NAN,
        });

        let pressure = coupler.pressure(Some(listener_state(2.0, -1.0)));

        assert_eq!(coupler.coupling_strength(), 1.0);
        assert_eq!(pressure.tension_pressure, 0.0);
        assert_eq!(pressure.temperature_bonus, 0.0);
    }
}
