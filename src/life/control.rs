const RANGE_OCT_MAX: f32 = 6.0;
pub const MIN_FREQ_HZ: f32 = 1.0;
pub const MAX_FREQ_HZ: f32 = 20_000.0;

#[derive(Debug, Clone, Default)]
pub struct WorldControl {}

#[derive(Debug, Clone, Default)]
pub struct AgentControl {
    pub body: BodyControl,
    pub pitch: PitchControl,
    pub phonation: PhonationControl,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BodyMethod {
    #[default]
    Sine,
    Harmonic,
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
}

impl Default for BodyControl {
    fn default() -> Self {
        Self {
            method: BodyMethod::default(),
            amp: 0.18,
            timbre: TimbreControl::default(),
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

#[derive(Debug, Clone)]
pub struct PitchControl {
    pub mode: PitchMode,
    pub core_kind: PitchCoreKind,
    /// Frequency in Hz: center for free mode, fixed output for lock.
    pub freq: f32,
    pub range_oct: f32,
    pub gravity: f32,
    pub exploration: f32,
    pub persistence: f32,
}

impl Default for PitchControl {
    fn default() -> Self {
        Self {
            mode: PitchMode::Free,
            core_kind: PitchCoreKind::HillClimb,
            freq: 220.0,
            range_oct: RANGE_OCT_MAX,
            gravity: 0.5,
            exploration: 0.0,
            persistence: 0.5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhonationType {
    #[default]
    Interval,
    Clock,
    Field,
    /// Sustain once per lifecycle; ignores density/sync/legato.
    Hold,
    None,
}

#[derive(Debug, Clone)]
pub struct PhonationControl {
    pub r#type: PhonationType,
    pub density: f32,
    pub sync: f32,
    pub legato: f32,
    pub sociality: f32,
}

impl Default for PhonationControl {
    fn default() -> Self {
        Self {
            r#type: PhonationType::default(),
            density: 0.5,
            sync: 0.5,
            legato: 0.5,
            sociality: 0.0,
        }
    }
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

#[derive(Debug, Clone, Default)]
pub struct ControlUpdate {
    pub amp: Option<f32>,
    pub freq: Option<f32>,
    pub timbre_brightness: Option<f32>,
    pub timbre_inharmonic: Option<f32>,
    pub timbre_width: Option<f32>,
    pub timbre_motion: Option<f32>,
}

impl ControlUpdate {
    pub fn is_empty(&self) -> bool {
        self.amp.is_none()
            && self.freq.is_none()
            && self.timbre_brightness.is_none()
            && self.timbre_inharmonic.is_none()
            && self.timbre_width.is_none()
            && self.timbre_motion.is_none()
    }
}

impl AgentControl {
    pub fn apply_update(&mut self, update: &ControlUpdate) {
        if let Some(amp) = update.amp {
            self.set_amp_clamped(amp);
        }
        if let Some(freq) = update.freq {
            self.set_freq_lock_clamped(freq);
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
            timbre_brightness: Some(-0.1),
            timbre_inharmonic: Some(2.0),
            timbre_width: Some(0.25),
            timbre_motion: Some(0.5),
        };
        control.apply_update(&update);

        assert!((control.body.amp - 1.0).abs() <= 1e-6);
        assert_eq!(control.pitch.mode, PitchMode::Lock);
        assert!((control.pitch.freq - MAX_FREQ_HZ).abs() <= 1e-6);
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
