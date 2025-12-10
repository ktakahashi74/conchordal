use std::f32::consts::PI;

#[derive(Clone, Copy, Debug, Default)]
pub struct NeuralRhythm {
    pub mag: f32,
    pub phase: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NeuralRhythms {
    pub delta: NeuralRhythm, // 0.5–2 Hz
    pub theta: NeuralRhythm, // 4–8 Hz
    pub alpha: NeuralRhythm, // 8–12 Hz
    pub beta: NeuralRhythm,  // 15–30 Hz
}

struct Resonator {
    rho: f32,
    cos_t: f32,
    sin_t: f32,
    state_re: f32,
    state_im: f32,
    scale: f32,
}

impl Resonator {
    fn new(center_hz: f32, q: f32, sample_rate: f32) -> Self {
        let omega = 2.0 * PI * center_hz / sample_rate;
        let cos_t = omega.cos();
        let sin_t = omega.sin();
        // Pole radius from Q; tighter q -> slower decay.
        let rho = (-omega / (2.0 * q.max(0.1))).exp().min(0.9999);
        Self {
            rho,
            cos_t,
            sin_t,
            state_re: 0.0,
            state_im: 0.0,
            scale: 1.0 - rho,
        }
    }

    fn process(&mut self, x: f32) -> NeuralRhythm {
        // Rotate previous state
        let re_rot = self.state_re * self.cos_t - self.state_im * self.sin_t;
        let im_rot = self.state_re * self.sin_t + self.state_im * self.cos_t;
        // Decay and inject new energy
        let re = self.rho * re_rot + self.scale * x;
        let im = self.rho * im_rot;
        self.state_re = re;
        self.state_im = im;
        let mag = (re * re + im * im).sqrt();
        let phase = im.atan2(re);
        NeuralRhythm { mag, phase }
    }
}

/// Extracts neural-band modulations from 3-band energy envelopes.
pub struct ModulationBank {
    bands: [Resonator; 4],
    last: NeuralRhythms,
}

impl ModulationBank {
    pub fn new(envelope_rate_hz: f32) -> Self {
        // Center frequencies for the neural bands.
        let centers = [1.0, 6.0, 10.0, 22.0];
        let q = 2.5;
        let bands = [
            Resonator::new(centers[0], q, envelope_rate_hz),
            Resonator::new(centers[1], q, envelope_rate_hz),
            Resonator::new(centers[2], q, envelope_rate_hz),
            Resonator::new(centers[3], q, envelope_rate_hz),
        ];
        Self {
            bands,
            last: NeuralRhythms::default(),
        }
    }

    /// Update the modulation bank with band energies (low, mid, high).
    pub fn update(&mut self, low: f32, mid: f32, high: f32) -> NeuralRhythms {
        // Sum energy across bands; a more elaborate model could weight bands differently.
        let x = (low.max(0.0) + mid.max(0.0) + high.max(0.0)).sqrt(); // compress
        self.last.delta = self.bands[0].process(x);
        self.last.theta = self.bands[1].process(x);
        self.last.alpha = self.bands[2].process(x);
        self.last.beta = self.bands[3].process(x);
        self.last
    }

    pub fn last(&self) -> NeuralRhythms {
        self.last
    }
}
