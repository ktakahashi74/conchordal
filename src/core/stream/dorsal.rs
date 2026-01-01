use crate::core::modulation::{NeuralRhythms, RhythmEngine};
use tracing::debug;

/// Dorsal Stream (Where/How Pathway)
/// Handles fast articulation processing, rhythm extraction, and motor synchronization.
///
/// Features:
/// - Low-latency synchronous processing.
/// - 3-Band Crossover Flux detection (Low/Mid/High).
/// - Non-linear sensitivity boost for ambient signals.
pub struct DorsalStream {
    dynamics: RhythmEngine,
    // IIR Filter States
    lp_low_state: f32, // ~200Hz
    lp_mid_state: f32, // ~3000Hz
    prev_band_energy: [f32; 3],
    delta_env: f32,
    last_metrics: DorsalMetrics,
    fs: f32,
    vitality: f32,
    debug_timer: f32,
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
            dynamics: RhythmEngine::default(),
            lp_low_state: 0.0,
            lp_mid_state: 0.0,
            prev_band_energy: [0.0; 3],
            delta_env: 0.0,
            last_metrics: DorsalMetrics::default(),
            fs,
            vitality: 1.0,
            debug_timer: 0.0,
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
        for (i, &cur) in currents.iter().enumerate() {
            let diff = cur - self.prev_band_energy[i];
            if diff > 0.0 {
                raw_flux += diff;
            }
            self.prev_band_energy[i] = cur;
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
        let u_theta = drive.clamp(0.0, 1.0);

        // --- 3. Update Oscillators ---
        let dt = audio.len() as f32 / self.fs;
        let tau_delta = 0.6;
        let alpha = 1.0 - (-dt / tau_delta).exp();
        self.delta_env += alpha * (u_theta - self.delta_env);
        let u_delta = self.delta_env.clamp(0.0, 1.0);

        self.debug_timer += dt;
        if self.debug_timer >= 1.0 {
            debug!(
                target: "rhythm::dorsal",
                raw_flux,
                drive,
                u_theta,
                u_delta,
                dt,
                vitality = self.vitality
            );
            self.debug_timer = 0.0;
        }

        self.dynamics.update(dt, u_theta, u_delta, self.vitality)
    }

    pub fn reset(&mut self) {
        self.dynamics = RhythmEngine::default();
        self.lp_low_state = 0.0;
        self.lp_mid_state = 0.0;
        self.prev_band_energy = [0.0; 3];
        self.delta_env = 0.0;
        self.last_metrics = DorsalMetrics::default();
        self.vitality = 1.0;
        self.debug_timer = 0.0;
    }

    pub fn set_vitality(&mut self, v: f32) {
        self.vitality = v.clamp(0.0, 1.0);
    }

    pub fn last_metrics(&self) -> DorsalMetrics {
        self.last_metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dorsal_detects_click_train() {
        let fs = 44_100.0;
        let hop = 512;
        let dur_sec = 2.5;
        let total = (fs * dur_sec) as usize;
        let mut sig = vec![0.0f32; total];
        let period = 1.0 / 6.0;
        let click_len = 0.003;
        let freq = 3000.0;
        for i in 0..total {
            let t = i as f32 / fs;
            let phase = t % period;
            if phase < click_len {
                sig[i] = (std::f32::consts::TAU * freq * t).sin() * 0.8;
            }
        }

        let mut dorsal = DorsalStream::new(fs);
        let mut max_u_theta: f32 = 0.0;
        let mut last = NeuralRhythms::default();
        let mut idx = 0;
        while idx < total {
            let end = (idx + hop).min(total);
            last = dorsal.process(&sig[idx..end]);
            let raw_flux = dorsal.last_metrics().flux;
            let drive = (raw_flux * 500.0).tanh();
            max_u_theta = max_u_theta.max(drive);
            idx = end;
        }

        assert!(
            max_u_theta > 0.05,
            "expected u_theta to rise above 0.05, got {max_u_theta}"
        );
        assert!(
            last.theta.mag > 0.04,
            "expected theta.mag to rise, got {}",
            last.theta.mag
        );
        assert!(
            (last.theta.freq_hz - 6.0).abs() < 1.5,
            "expected theta.freq_hz near 6Hz, got {}",
            last.theta.freq_hz
        );
    }
}
