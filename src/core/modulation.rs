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
    vitality: f32,
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
            vitality: 0.0,
        }
    }

    fn process(&mut self, x: f32) -> NeuralRhythm {
        let mag_sq = self.state_re * self.state_re + self.state_im * self.state_im;
        let effective_rho = self.rho + self.vitality * (1.0 - mag_sq);
        // Rotate previous state
        let re_rot = self.state_re * self.cos_t - self.state_im * self.sin_t;
        let im_rot = self.state_re * self.sin_t + self.state_im * self.cos_t;
        // Decay and inject new energy
        let re = effective_rho * re_rot + self.scale * x;
        let im = effective_rho * im_rot;
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
    long_term_avg: f32,
    vitality: f32,
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
            long_term_avg: 0.0,
            vitality: 0.0,
        }
    }

    pub fn set_vitality(&mut self, v: f32) {
        self.vitality = v;
        for band in &mut self.bands {
            band.vitality = v;
        }
    }

    /// Update the modulation bank with band energies (low, mid, high).
    pub fn update(&mut self, low: f32, mid: f32, high: f32) -> NeuralRhythms {
        // Sum energy across bands; a more elaborate model could weight bands differently.
        let x_raw = (low.max(0.0) + mid.max(0.0) + high.max(0.0)).sqrt(); // compress
        // DC reject: track slow average and feed AC component to the resonators.
        self.long_term_avg = 0.995 * self.long_term_avg + 0.005 * x_raw;
        let x_ac = x_raw - self.long_term_avg;

        self.last.delta = self.bands[0].process(x_ac);
        self.last.theta = self.bands[1].process(x_ac);
        self.last.alpha = self.bands[2].process(x_ac);
        self.last.beta = self.bands[3].process(x_ac);
        self.last
    }

    pub fn last(&self) -> NeuralRhythms {
        self.last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_oscillation() {
        let mut res = Resonator::new(2.0, 5.0, 100.0);
        res.vitality = 0.5;

        // Kick the resonator into motion.
        for _ in 0..16 {
            res.process(0.6);
        }

        // With positive vitality, energy should settle into a limit cycle even with no further input.
        let mut steady_mag = 0.0;
        for _ in 0..200 {
            steady_mag = res.process(0.0).mag;
        }

        // Linear version should decay toward zero under the same conditions.
        let mut linear = Resonator::new(2.0, 5.0, 100.0);
        for _ in 0..16 {
            linear.process(0.6);
        }
        let mut decayed_mag = 0.0;
        for _ in 0..200 {
            decayed_mag = linear.process(0.0).mag;
        }

        assert!(
            steady_mag > 0.3,
            "expected sustained oscillation; got mag={steady_mag}"
        );
        assert!(
            decayed_mag < 0.05,
            "expected linear resonator to decay; got mag={decayed_mag}"
        );
    }
}
