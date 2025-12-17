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

fn wrap_phase(phase: &mut f32) {
    let tau = 2.0 * PI;
    while *phase >= tau {
        *phase -= tau;
    }
    while *phase < 0.0 {
        *phase += tau;
    }
}

/// Predictive-coding rhythm dynamics driven by transient flux.
#[derive(Clone, Debug)]
pub struct RhythmDynamics {
    pub theta_phase: f32,
    pub delta_phase: f32,
    pub alpha_phase: f32,
    pub beta_phase: f32,
    pub beta_energy: f32,
    pub alpha_energy: f32,
    pub last_flux: f32,
    theta_freq: f32,
    coupling_strength: f32,
    beta_tau: f32,
    alpha_tau: f32,
    last: NeuralRhythms,
}

impl Default for RhythmDynamics {
    fn default() -> Self {
        Self {
            theta_phase: 0.0,
            delta_phase: 0.0,
            alpha_phase: 0.0,
            beta_phase: 0.0,
            beta_energy: 0.0,
            alpha_energy: 0.0,
            last_flux: 0.0,
            theta_freq: 6.0,
            coupling_strength: 20.0,
            beta_tau: 0.15, // ~150 ms for smoother surprise
            alpha_tau: 0.2, // ~200 ms stability build-up
            last: NeuralRhythms::default(),
        }
    }
}

impl RhythmDynamics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_vitality(&mut self, v: f32) {
        // Map vitality into flux scaling to keep response bounded.
        self.last_flux *= v.max(0.0);
    }

    pub fn update(&mut self, dt: f32, flux: f32) -> NeuralRhythms {
        let dt = dt.max(1e-4);
        // Light smoothing on flux to avoid impulsive jitter.
        let flux_env = 0.8 * self.last_flux + 0.2 * flux.max(0.0);
        self.last_flux = flux_env;

        // Theta (beat) oscillator with PRC coupling toward phase 0 on flux.
        let omega = 2.0 * PI * self.theta_freq;
        let d_theta = (omega + self.coupling_strength * flux_env * (-self.theta_phase).sin()) * dt;

        // Harmonic phase grid
        self.theta_phase = (self.theta_phase + d_theta).rem_euclid(2.0 * PI);
        self.delta_phase = (self.delta_phase + d_theta * 0.25).rem_euclid(2.0 * PI);
        self.alpha_phase = (self.alpha_phase + d_theta * 2.0).rem_euclid(2.0 * PI);
        self.beta_phase = (self.beta_phase + d_theta * 4.0).rem_euclid(2.0 * PI);

        // Predictive coding magnitudes
        let alignment = self.theta_phase.cos();
        let error = flux_env * (1.0 - alignment).max(0.0);
        let stability = flux_env * alignment.max(0.0);

        let a_beta = (-dt / self.beta_tau).exp();
        self.beta_energy = a_beta * self.beta_energy + (1.0 - a_beta) * error;
        self.beta_energy = self.beta_energy.clamp(0.0, 1.0);

        let a_alpha = (-dt / self.alpha_tau).exp();
        let target_alpha = (stability + 0.1).min(1.0);
        self.alpha_energy = a_alpha * self.alpha_energy + (1.0 - a_alpha) * target_alpha;
        self.alpha_energy = self.alpha_energy.clamp(0.0, 1.0);

        // Presence follows the smoothed flux to keep beat/measure magnitudes informative.
        let presence = (self.last_flux * 2.0).clamp(0.1, 1.0);
        let theta_mag = presence;
        let delta_mag = presence;

        self.last = NeuralRhythms {
            delta: NeuralRhythm {
                mag: delta_mag,
                phase: self.delta_phase,
            },
            theta: NeuralRhythm {
                mag: theta_mag,
                phase: self.theta_phase,
            },
            alpha: NeuralRhythm {
                mag: self.alpha_energy,
                phase: self.alpha_phase,
            },
            beta: NeuralRhythm {
                mag: self.beta_energy,
                phase: self.beta_phase,
            },
        };

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
    fn beta_tracks_prediction_error() {
        let mut dyn_model = RhythmDynamics::default();
        let mut beta = 0.0;
        for _ in 0..100 {
            let rhythms = dyn_model.update(0.01, 1.0);
            beta = rhythms.beta.mag;
        }
        assert!(beta > 0.001, "beta should rise under persistent error");
    }
}
