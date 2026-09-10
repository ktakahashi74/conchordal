//! Acoustic onset evidence shared by temporal observers.

#[derive(Clone, Copy, Default)]
pub(crate) struct Onset {
    pub(crate) fired: bool,
    pub(crate) frac: f32,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct OnsetDetector {
    baseline: f32,
    variance: f32,
    refractory_left: f32,
    previous_drive: f32,
}

impl OnsetDetector {
    pub(crate) fn process(&mut self, dt: f32, drive: f32) -> Onset {
        let a = (-dt / 1.0).exp();
        self.baseline = a * self.baseline + (1.0 - a) * drive;
        let dev = drive - self.baseline;
        self.variance = a * self.variance + (1.0 - a) * dev * dev;
        let std = self.variance.max(1e-6).sqrt();
        let threshold = (self.baseline + 2.0 * std).clamp(0.01, 0.95);
        self.refractory_left = (self.refractory_left - dt).max(0.0);
        let mut onset = Onset::default();
        if self.refractory_left <= 0.0 && self.previous_drive < threshold && drive >= threshold {
            let denominator = (drive - self.previous_drive).max(1e-6);
            onset.frac = ((threshold - self.previous_drive) / denominator).clamp(0.0, 1.0);
            onset.fired = true;
            self.refractory_left = 0.06;
        }
        self.previous_drive = drive;
        onset
    }
}
