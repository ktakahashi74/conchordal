use super::pitch_core::{AnyPitchCore, PitchCore};
use crate::core::landscape::Landscape;
use crate::core::modulation::NeuralRhythms;
use crate::life::perceptual::{FeaturesNow, PerceptualContext};
use rand::rngs::SmallRng;

#[derive(Debug)]
pub struct PitchController {
    core: AnyPitchCore,
    perceptual: PerceptualContext,
    target_pitch_log2: f32,
    integration_window: f32,
    accumulated_time: f32,
    last_theta_phase: f32,
    theta_phase_initialized: bool,
    last_target_salience: f32,
    rng: SmallRng,
}

impl PitchController {
    pub fn new(
        core: AnyPitchCore,
        perceptual: PerceptualContext,
        target_pitch_log2: f32,
        integration_window: f32,
        rng: SmallRng,
    ) -> Self {
        Self {
            core,
            perceptual,
            target_pitch_log2,
            integration_window,
            accumulated_time: 0.0,
            last_theta_phase: 0.0,
            theta_phase_initialized: false,
            last_target_salience: 0.0,
            rng,
        }
    }

    pub fn target_pitch_log2(&self) -> f32 {
        self.target_pitch_log2
    }

    pub fn integration_window(&self) -> f32 {
        self.integration_window
    }

    pub fn last_target_salience(&self) -> f32 {
        self.last_target_salience
    }

    pub fn core_mut(&mut self) -> &mut AnyPitchCore {
        &mut self.core
    }

    pub fn force_set_target_pitch_log2(&mut self, log_freq: f32) {
        self.target_pitch_log2 = log_freq.max(0.0);
        self.accumulated_time = 0.0;
        self.last_theta_phase = 0.0;
        self.theta_phase_initialized = false;
        self.last_target_salience = 0.0;
    }

    /// Update pitch targets at control rate (hop-sized steps).
    pub fn update_pitch_target(
        &mut self,
        current_freq_hz: f32,
        rhythms: &NeuralRhythms,
        dt_sec: f32,
        landscape: &Landscape,
    ) {
        let dt_sec = dt_sec.max(0.0);
        let current_freq = current_freq_hz.max(1.0);
        let current_pitch_log2 = current_freq.log2();
        let mut target_pitch_log2 = if self.target_pitch_log2 <= 0.0 {
            current_pitch_log2
        } else {
            self.target_pitch_log2
        };
        self.integration_window = 2.0 + 10.0 / current_freq.max(1.0);
        self.accumulated_time += dt_sec;

        // Detect theta wrap to avoid missing zero-crossings at control rate.
        let theta_phase = rhythms.theta.phase;
        let theta_cross = if theta_phase.is_finite() && self.last_theta_phase.is_finite() {
            let wrapped = self.theta_phase_initialized && theta_phase < self.last_theta_phase;
            wrapped && rhythms.theta.mag.is_finite() && rhythms.theta.mag > 0.0
        } else {
            false
        };
        self.last_theta_phase = if theta_phase.is_finite() {
            self.theta_phase_initialized = true;
            theta_phase
        } else {
            self.theta_phase_initialized = false;
            0.0
        };

        if theta_cross && self.accumulated_time >= self.integration_window {
            let elapsed = self.accumulated_time;
            self.accumulated_time = 0.0;
            let features = FeaturesNow::from_subjective_intensity(&landscape.subjective_intensity);
            debug_assert_eq!(features.distribution.len(), landscape.space.n_bins());
            self.perceptual.ensure_len(features.distribution.len());
            let proposal = self.core.propose_target(
                current_pitch_log2,
                target_pitch_log2,
                current_freq,
                self.integration_window,
                landscape,
                &self.perceptual,
                &features,
                &mut self.rng,
            );
            target_pitch_log2 = proposal.target_pitch_log2;
            self.last_target_salience = proposal.salience;
            if let Some(idx) = landscape.space.index_of_log2(target_pitch_log2) {
                self.perceptual.update(idx, &features, elapsed);
            }
        }

        let (fmin, fmax) = landscape.freq_bounds_log2();
        target_pitch_log2 = target_pitch_log2.clamp(fmin, fmax);
        self.target_pitch_log2 = target_pitch_log2;
    }
}

#[cfg(test)]
impl PitchController {
    pub(crate) fn set_accumulated_time_for_test(&mut self, value: f32) {
        self.accumulated_time = value;
    }

    pub(crate) fn accumulated_time_for_test(&self) -> f32 {
        self.accumulated_time
    }

    pub(crate) fn set_theta_phase_state_for_test(&mut self, last_phase: f32, initialized: bool) {
        self.last_theta_phase = last_phase;
        self.theta_phase_initialized = initialized;
    }
}
