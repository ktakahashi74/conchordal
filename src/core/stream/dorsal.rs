use crate::core::modulation::{NeuralRhythms, RhythmDynamics};

/// Dorsal Stream (Where/How Pathway)
/// Handles fast temporal processing, rhythm extraction, and motor synchronization.
///
/// Features:
/// - Low-latency synchronous processing.
/// - 3-Band Crossover Flux detection (Low/Mid/High).
/// - Non-linear sensitivity boost for ambient signals.
pub struct DorsalStream {
    dynamics: RhythmDynamics,
    // IIR Filter States
    lp_low_state: f32, // ~200Hz
    lp_mid_state: f32, // ~3000Hz
    prev_band_energy: [f32; 3],
    last_metrics: DorsalMetrics,
    fs: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DorsalMetrics {
    pub e_low: f32,
    pub e_mid: f32,
    pub e_high: f32,
    pub flux: f32,
}

impl DorsalStream {
    pub fn new(fs: f32) -> Self {
        Self {
            dynamics: RhythmDynamics::default(),
            lp_low_state: 0.0,
            lp_mid_state: 0.0,
            prev_band_energy: [0.0; 3],
            last_metrics: DorsalMetrics::default(),
            fs,
        }
    }

    /// Process audio chunk synchronously.
    /// Returns the updated NeuralRhythms immediately.
    pub fn process(&mut self, audio: &[f32]) -> NeuralRhythms {
        if audio.is_empty() {
            return self.dynamics.last();
        }

        // --- 1. Sub-band Flux Calculation ---

        // Coefficients for 1-pole filters
        // Low crossover: 200 Hz
        let alpha_low = 1.0 - (-2.0 * std::f32::consts::PI * 200.0 / self.fs).exp();
        // Mid/High crossover: 3000 Hz
        let alpha_mid = 1.0 - (-2.0 * std::f32::consts::PI * 3000.0 / self.fs).exp();

        let mut e_low = 0.0;
        let mut e_mid = 0.0;
        let mut e_high = 0.0;

        for &x in audio {
            // Low Band
            self.lp_low_state += alpha_low * (x - self.lp_low_state);
            let l = self.lp_low_state;

            // Mid Band boundary
            self.lp_mid_state += alpha_mid * (x - self.lp_mid_state);
            let mh_boundary = self.lp_mid_state;

            let band_l = l;
            let band_m = mh_boundary - l;
            let band_h = x - mh_boundary;

            e_low += band_l * band_l;
            e_mid += band_m * band_m;
            e_high += band_h * band_h;
        }

        // Normalize energy by chunk length
        let inv_len = 1.0 / audio.len() as f32;
        let currents = [e_low * inv_len, e_mid * inv_len, e_high * inv_len];

        // Sum positive flux across bands
        let mut raw_flux = 0.0;
        for i in 0..3 {
            let diff = currents[i] - self.prev_band_energy[i];
            if diff > 0.0 {
                raw_flux += diff;
            }
            self.prev_band_energy[i] = currents[i];
        }
        self.last_metrics = DorsalMetrics {
            e_low: currents[0],
            e_mid: currents[1],
            e_high: currents[2],
            flux: raw_flux,
        };

        // --- 2. Non-linear Boost (Simulating Neural Activation) ---
        // High gain (500.0) + tanh saturation to detect ambient shifts
        let drive = (raw_flux * 500.0).tanh();

        // --- 3. Update Oscillators ---
        let dt = audio.len() as f32 / self.fs;
        self.dynamics.update(dt, drive)
    }

    pub fn reset(&mut self) {
        self.dynamics = RhythmDynamics::default();
        self.lp_low_state = 0.0;
        self.lp_mid_state = 0.0;
        self.prev_band_energy = [0.0; 3];
        self.last_metrics = DorsalMetrics::default();
    }

    pub fn set_vitality(&mut self, v: f32) {
        self.dynamics.set_vitality(v);
    }

    pub fn last_metrics(&self) -> DorsalMetrics {
        self.last_metrics
    }
}
