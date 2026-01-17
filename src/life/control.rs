const RANGE_OCT_MAX: f32 = 6.0;

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
        if !freq.is_finite() || freq <= 0.0 {
            return Err("pitch.freq must be finite and > 0".to_string());
        }
        Ok(())
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

#[derive(Debug, Clone)]
pub struct PitchControl {
    pub mode: PitchMode,
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
