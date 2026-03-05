const RANGE_OCT_MAX: f32 = 6.0;
pub const MIN_FREQ_HZ: f32 = 1.0;
pub const MAX_FREQ_HZ: f32 = 20_000.0;
const DEFAULT_CROWDING_SIGMA_CENTS: f32 = 60.0;
const DEFAULT_ANNEAL_TEMP: f32 = 0.0;
const DEFAULT_MOVE_COST_COEFF: f32 = 0.5;
const DEFAULT_IMPROVEMENT_THRESHOLD: f32 = 0.1;
const DEFAULT_GLOBAL_PEAK_MIN_SEP_CENTS: f32 = 0.0;
const DEFAULT_PITCH_GLIDE_TAU_SEC: f32 = 0.0;

use crate::core::mode_pattern::ModePattern;

#[derive(Debug, Clone, Default)]
pub struct WorldControl {}

#[derive(Debug, Clone, Default)]
pub struct AgentControl {
    pub body: BodyControl,
    pub pitch: PitchControl,
    pub utterance: UtteranceControl,
    pub perceptual: PerceptualControl,
}

impl AgentControl {
    pub fn validate(&self) -> Result<(), String> {
        let freq = self.pitch.freq;
        if !freq.is_finite() || freq < MIN_FREQ_HZ {
            return Err("pitch.freq must be finite and > 0".to_string());
        }
        Ok(())
    }

    #[inline]
    pub fn set_amp_clamped(&mut self, amp: f32) {
        self.body.amp = amp.clamp(0.0, 1.0);
    }

    #[inline]
    pub fn set_freq_lock_clamped(&mut self, freq: f32) {
        self.pitch.freq = freq.clamp(MIN_FREQ_HZ, MAX_FREQ_HZ);
        self.pitch.mode = PitchMode::Lock;
    }

    #[inline]
    pub fn set_timbre_brightness_clamped(&mut self, brightness: f32) {
        self.body.timbre.brightness = brightness.clamp(0.0, 1.0);
    }

    #[inline]
    pub fn set_timbre_inharmonic_clamped(&mut self, inharmonic: f32) {
        self.body.timbre.inharmonic = inharmonic.clamp(0.0, 1.0);
    }

    #[inline]
    pub fn set_timbre_width_clamped(&mut self, width: f32) {
        self.body.timbre.width = width.clamp(0.0, 1.0);
    }

    #[inline]
    pub fn set_timbre_motion_clamped(&mut self, motion: f32) {
        self.body.timbre.motion = motion.clamp(0.0, 1.0);
    }

    #[inline]
    pub fn set_landscape_weight_clamped(&mut self, weight: f32) {
        self.pitch.landscape_weight = if weight.is_finite() {
            weight.max(0.0)
        } else {
            1.0
        };
    }

    #[inline]
    pub fn set_exploration_clamped(&mut self, value: f32) {
        self.pitch.exploration = value.clamp(0.0, 1.0);
    }

    #[inline]
    pub fn set_persistence_clamped(&mut self, value: f32) {
        self.pitch.persistence = value.clamp(0.0, 1.0);
    }

    #[inline]
    pub fn set_crowding_strength_clamped(&mut self, value: f32) {
        self.pitch.crowding_strength = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
    }

    #[inline]
    pub fn set_crowding_sigma_cents_clamped(&mut self, value: f32) {
        self.pitch.crowding_sigma_cents = if value.is_finite() {
            value.max(1e-3)
        } else {
            DEFAULT_CROWDING_SIGMA_CENTS
        };
        self.pitch.crowding_sigma_from_roughness = false;
    }

    #[inline]
    pub fn set_crowding_sigma_from_roughness(&mut self, enabled: bool) {
        self.pitch.crowding_sigma_from_roughness = enabled;
    }

    #[inline]
    pub fn set_leave_self_out(&mut self, enabled: bool) {
        self.pitch.leave_self_out = enabled;
    }

    #[inline]
    pub fn set_continuous_drive_clamped(&mut self, value: f32) {
        self.body.continuous_drive = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
    }

    #[inline]
    pub fn set_pitch_smooth_tau_clamped(&mut self, value: f32) {
        self.body.pitch_smooth_tau = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
    }

    #[inline]
    pub fn set_anneal_temp_clamped(&mut self, value: f32) {
        self.pitch.anneal_temp = if value.is_finite() {
            value.max(0.0)
        } else {
            DEFAULT_ANNEAL_TEMP
        };
    }

    #[inline]
    pub fn set_move_cost_coeff_clamped(&mut self, value: f32) {
        self.pitch.move_cost_coeff = if value.is_finite() {
            value.max(0.0)
        } else {
            DEFAULT_MOVE_COST_COEFF
        };
    }

    #[inline]
    pub fn set_improvement_threshold_clamped(&mut self, value: f32) {
        self.pitch.improvement_threshold = if value.is_finite() {
            value.max(0.0)
        } else {
            DEFAULT_IMPROVEMENT_THRESHOLD
        };
    }

    #[inline]
    pub fn set_proposal_interval_sec_clamped(&mut self, value: f32) {
        self.pitch.proposal_interval_sec = if value.is_finite() && value > 0.0 {
            Some(value)
        } else {
            None
        };
    }

    #[inline]
    pub fn set_global_peak_count_clamped(&mut self, value: i64) {
        self.pitch.global_peak_count = value.max(0) as usize;
    }

    #[inline]
    pub fn set_global_peak_min_sep_cents_clamped(&mut self, value: f32) {
        self.pitch.global_peak_min_sep_cents = if value.is_finite() {
            value.max(0.0)
        } else {
            DEFAULT_GLOBAL_PEAK_MIN_SEP_CENTS
        };
    }

    #[inline]
    pub fn set_use_ratio_candidates(&mut self, enabled: bool) {
        self.pitch.use_ratio_candidates = enabled;
    }

    #[inline]
    pub fn set_ratio_candidate_count_clamped(&mut self, value: i64) {
        self.pitch.ratio_candidate_count = value.max(0) as usize;
    }

    #[inline]
    pub fn set_move_cost_time_scale(&mut self, value: MoveCostTimeScale) {
        self.pitch.move_cost_time_scale = value;
    }

    #[inline]
    pub fn set_leave_self_out_harmonics_clamped(&mut self, value: i64) {
        self.pitch.leave_self_out_harmonics = value.clamp(1, i64::from(u8::MAX)) as u8;
    }

    #[inline]
    pub fn set_pitch_apply_mode(&mut self, value: PitchApplyMode) {
        self.pitch.pitch_apply_mode = value;
    }

    #[inline]
    pub fn set_pitch_glide_tau_sec_clamped(&mut self, value: f32) {
        self.pitch.pitch_glide_tau_sec = if value.is_finite() {
            value.max(0.0)
        } else {
            DEFAULT_PITCH_GLIDE_TAU_SEC
        };
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BodyMethod {
    #[default]
    Sine,
    Harmonic,
    Modal,
}

#[derive(Debug, Clone)]
pub struct TimbreControl {
    pub brightness: f32,
    pub inharmonic: f32,
    pub width: f32,
    pub motion: f32,
}

impl Default for TimbreControl {
    fn default() -> Self {
        Self {
            brightness: 0.6,
            inharmonic: 0.0,
            width: 0.0,
            motion: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BodyControl {
    pub method: BodyMethod,
    pub amp: f32,
    pub timbre: TimbreControl,
    pub modes: Option<ModePattern>,
    pub continuous_drive: f32,
    pub pitch_smooth_tau: f32,
}

impl Default for BodyControl {
    fn default() -> Self {
        Self {
            method: BodyMethod::default(),
            amp: 0.18,
            timbre: TimbreControl::default(),
            modes: None,
            continuous_drive: 0.0,
            pitch_smooth_tau: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PitchMode {
    #[default]
    Free,
    Lock,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PitchCoreKind {
    #[default]
    HillClimb,
    PeakSampler,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MoveCostTimeScale {
    #[default]
    LegacyIntegrationWindow,
    ProposalInterval,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PitchApplyMode {
    #[default]
    GateSnap,
    Glide,
}

#[derive(Debug, Clone)]
pub struct PitchControl {
    pub mode: PitchMode,
    pub core_kind: PitchCoreKind,
    /// Frequency in Hz: center for free mode, fixed output for lock.
    pub freq: f32,
    pub range_oct: f32,
    pub gravity: f32,
    pub landscape_weight: f32,
    pub exploration: f32,
    pub persistence: f32,
    pub crowding_strength: f32,
    pub crowding_sigma_cents: f32,
    pub crowding_sigma_from_roughness: bool,
    pub leave_self_out: bool,
    pub anneal_temp: f32,
    pub move_cost_coeff: f32,
    pub improvement_threshold: f32,
    pub proposal_interval_sec: Option<f32>,
    pub global_peak_count: usize,
    pub global_peak_min_sep_cents: f32,
    pub use_ratio_candidates: bool,
    pub ratio_candidate_count: usize,
    pub move_cost_time_scale: MoveCostTimeScale,
    pub leave_self_out_harmonics: u8,
    pub pitch_apply_mode: PitchApplyMode,
    pub pitch_glide_tau_sec: f32,
}

impl Default for PitchControl {
    fn default() -> Self {
        Self {
            mode: PitchMode::Free,
            core_kind: PitchCoreKind::HillClimb,
            freq: 220.0,
            range_oct: RANGE_OCT_MAX,
            gravity: 0.5,
            landscape_weight: 1.0,
            exploration: 0.0,
            persistence: 0.5,
            crowding_strength: 0.0,
            crowding_sigma_cents: DEFAULT_CROWDING_SIGMA_CENTS,
            crowding_sigma_from_roughness: true,
            leave_self_out: false,
            anneal_temp: DEFAULT_ANNEAL_TEMP,
            move_cost_coeff: DEFAULT_MOVE_COST_COEFF,
            improvement_threshold: DEFAULT_IMPROVEMENT_THRESHOLD,
            proposal_interval_sec: None,
            global_peak_count: 0,
            global_peak_min_sep_cents: DEFAULT_GLOBAL_PEAK_MIN_SEP_CENTS,
            use_ratio_candidates: false,
            ratio_candidate_count: 0,
            move_cost_time_scale: MoveCostTimeScale::LegacyIntegrationWindow,
            leave_self_out_harmonics: 1,
            pitch_apply_mode: PitchApplyMode::GateSnap,
            pitch_glide_tau_sec: DEFAULT_PITCH_GLIDE_TAU_SEC,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct UtteranceControl {
    pub spec: crate::life::scenario::UtteranceSpec,
}

#[derive(Debug, Clone)]
pub struct PerceptualControl {
    pub enabled: bool,
    pub adaptation: f32,
    pub novelty_bias: f32,
    pub self_focus: f32,
}

impl Default for PerceptualControl {
    fn default() -> Self {
        Self {
            enabled: true,
            adaptation: 0.5,
            novelty_bias: 1.0,
            self_focus: 0.15,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ControlUpdate {
    pub amp: Option<f32>,
    pub freq: Option<f32>,
    pub landscape_weight: Option<f32>,
    pub exploration: Option<f32>,
    pub persistence: Option<f32>,
    pub crowding_strength: Option<f32>,
    pub crowding_sigma_cents: Option<f32>,
    pub crowding_sigma_from_roughness: Option<bool>,
    pub leave_self_out: Option<bool>,
    pub anneal_temp: Option<f32>,
    pub timbre_brightness: Option<f32>,
    pub timbre_inharmonic: Option<f32>,
    pub timbre_width: Option<f32>,
    pub timbre_motion: Option<f32>,
    pub continuous_drive: Option<f32>,
    pub pitch_smooth_tau: Option<f32>,
    pub move_cost_coeff: Option<f32>,
    pub improvement_threshold: Option<f32>,
    pub proposal_interval_sec: Option<f32>,
    pub global_peak_count: Option<i64>,
    pub global_peak_min_sep_cents: Option<f32>,
    pub use_ratio_candidates: Option<bool>,
    pub ratio_candidate_count: Option<i64>,
    pub move_cost_time_scale: Option<MoveCostTimeScale>,
    pub leave_self_out_harmonics: Option<i64>,
    pub pitch_apply_mode: Option<PitchApplyMode>,
    pub pitch_glide_tau_sec: Option<f32>,
}

impl AgentControl {
    pub fn apply_update(&mut self, update: &ControlUpdate) {
        if let Some(amp) = update.amp {
            self.set_amp_clamped(amp);
        }
        if let Some(freq) = update.freq {
            self.set_freq_lock_clamped(freq);
        }
        if let Some(weight) = update.landscape_weight {
            self.set_landscape_weight_clamped(weight);
        }
        if let Some(exploration) = update.exploration {
            self.set_exploration_clamped(exploration);
        }
        if let Some(persistence) = update.persistence {
            self.set_persistence_clamped(persistence);
        }
        if let Some(strength) = update.crowding_strength {
            self.set_crowding_strength_clamped(strength);
        }
        if let Some(sigma) = update.crowding_sigma_cents {
            self.set_crowding_sigma_cents_clamped(sigma);
        }
        if let Some(enabled) = update.crowding_sigma_from_roughness {
            self.set_crowding_sigma_from_roughness(enabled);
        }
        if let Some(enabled) = update.leave_self_out {
            self.set_leave_self_out(enabled);
        }
        if let Some(anneal_temp) = update.anneal_temp {
            self.set_anneal_temp_clamped(anneal_temp);
        }
        if let Some(brightness) = update.timbre_brightness {
            self.set_timbre_brightness_clamped(brightness);
        }
        if let Some(inharmonic) = update.timbre_inharmonic {
            self.set_timbre_inharmonic_clamped(inharmonic);
        }
        if let Some(width) = update.timbre_width {
            self.set_timbre_width_clamped(width);
        }
        if let Some(motion) = update.timbre_motion {
            self.set_timbre_motion_clamped(motion);
        }
        if let Some(drive) = update.continuous_drive {
            self.set_continuous_drive_clamped(drive);
        }
        if let Some(tau) = update.pitch_smooth_tau {
            self.set_pitch_smooth_tau_clamped(tau);
        }
        if let Some(coeff) = update.move_cost_coeff {
            self.set_move_cost_coeff_clamped(coeff);
        }
        if let Some(threshold) = update.improvement_threshold {
            self.set_improvement_threshold_clamped(threshold);
        }
        if let Some(interval) = update.proposal_interval_sec {
            self.set_proposal_interval_sec_clamped(interval);
        }
        if let Some(count) = update.global_peak_count {
            self.set_global_peak_count_clamped(count);
        }
        if let Some(min_sep) = update.global_peak_min_sep_cents {
            self.set_global_peak_min_sep_cents_clamped(min_sep);
        }
        if let Some(enabled) = update.use_ratio_candidates {
            self.set_use_ratio_candidates(enabled);
        }
        if let Some(count) = update.ratio_candidate_count {
            self.set_ratio_candidate_count_clamped(count);
        }
        if let Some(scale) = update.move_cost_time_scale {
            self.set_move_cost_time_scale(scale);
        }
        if let Some(harmonics) = update.leave_self_out_harmonics {
            self.set_leave_self_out_harmonics_clamped(harmonics);
        }
        if let Some(mode) = update.pitch_apply_mode {
            self.set_pitch_apply_mode(mode);
        }
        if let Some(tau) = update.pitch_glide_tau_sec {
            self.set_pitch_glide_tau_sec_clamped(tau);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_update_clamps_and_locks_freq() {
        let mut control = AgentControl::default();
        control.pitch.mode = PitchMode::Free;
        let update = ControlUpdate {
            amp: Some(1.2),
            freq: Some(99_999.0),
            landscape_weight: Some(-2.0),
            exploration: Some(2.0),
            persistence: Some(-1.0),
            crowding_strength: Some(-4.0),
            crowding_sigma_cents: Some(-3.0),
            crowding_sigma_from_roughness: Some(false),
            leave_self_out: Some(true),
            anneal_temp: Some(-0.5),
            timbre_brightness: Some(-0.1),
            timbre_inharmonic: Some(2.0),
            timbre_width: Some(0.25),
            timbre_motion: Some(0.5),
            continuous_drive: None,
            pitch_smooth_tau: None,
            move_cost_coeff: Some(-0.2),
            improvement_threshold: Some(-0.5),
            proposal_interval_sec: Some(-1.0),
            global_peak_count: Some(8),
            global_peak_min_sep_cents: Some(-9.0),
            use_ratio_candidates: Some(true),
            ratio_candidate_count: Some(6),
            move_cost_time_scale: Some(MoveCostTimeScale::ProposalInterval),
            leave_self_out_harmonics: Some(0),
            pitch_apply_mode: Some(PitchApplyMode::Glide),
            pitch_glide_tau_sec: Some(-0.2),
        };
        control.apply_update(&update);

        assert!((control.body.amp - 1.0).abs() <= 1e-6);
        assert_eq!(control.pitch.mode, PitchMode::Lock);
        assert!((control.pitch.freq - MAX_FREQ_HZ).abs() <= 1e-6);
        assert!((control.pitch.landscape_weight - 0.0).abs() <= 1e-6);
        assert!((control.pitch.exploration - 1.0).abs() <= 1e-6);
        assert!((control.pitch.persistence - 0.0).abs() <= 1e-6);
        assert!((control.pitch.crowding_strength - 0.0).abs() <= 1e-6);
        assert!((control.pitch.crowding_sigma_cents - 1e-3).abs() <= 1e-6);
        assert!(!control.pitch.crowding_sigma_from_roughness);
        assert!(control.pitch.leave_self_out);
        assert!((control.pitch.anneal_temp - 0.0).abs() <= 1e-6);
        assert!((control.pitch.move_cost_coeff - 0.0).abs() <= 1e-6);
        assert!((control.pitch.improvement_threshold - 0.0).abs() <= 1e-6);
        assert_eq!(control.pitch.proposal_interval_sec, None);
        assert_eq!(control.pitch.global_peak_count, 8);
        assert!((control.pitch.global_peak_min_sep_cents - 0.0).abs() <= 1e-6);
        assert!(control.pitch.use_ratio_candidates);
        assert_eq!(control.pitch.ratio_candidate_count, 6);
        assert_eq!(
            control.pitch.move_cost_time_scale,
            MoveCostTimeScale::ProposalInterval
        );
        assert_eq!(control.pitch.leave_self_out_harmonics, 1);
        assert_eq!(control.pitch.pitch_apply_mode, PitchApplyMode::Glide);
        assert!((control.pitch.pitch_glide_tau_sec - 0.0).abs() <= 1e-6);
        assert!((control.body.timbre.brightness - 0.0).abs() <= 1e-6);
        assert!((control.body.timbre.inharmonic - 1.0).abs() <= 1e-6);
        assert!((control.body.timbre.width - 0.25).abs() <= 1e-6);
        assert!((control.body.timbre.motion - 0.5).abs() <= 1e-6);
    }

    #[test]
    fn set_freq_lock_uses_min_bound() {
        let mut control = AgentControl::default();
        control.set_freq_lock_clamped(-10.0);
        assert_eq!(control.pitch.mode, PitchMode::Lock);
        assert!((control.pitch.freq - MIN_FREQ_HZ).abs() <= 1e-6);
    }
}
