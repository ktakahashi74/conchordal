use tracing::debug;

/// Dorsal Stream (Where/How Pathway)
/// Fast articulation front-end: computes 3-band spectral flux used as the
/// metric drive for downstream meter networks.
///
/// Features:
/// - Low-latency synchronous processing.
/// - 3-Band Crossover Flux detection (Low/Mid/High).
pub struct DorsalStream {
    // IIR Filter States
    lp_low_state: f32, // ~200Hz
    lp_mid_state: f32, // ~3000Hz
    prev_band_energy: [f32; 3],
    last_metrics: DorsalMetrics,
    fs: f32,
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
            lp_low_state: 0.0,
            lp_mid_state: 0.0,
            prev_band_energy: [0.0; 3],
            last_metrics: DorsalMetrics::default(),
            fs,
            debug_timer: 0.0,
        }
    }

    /// Process an audio chunk synchronously, updating the 3-band flux metrics.
    pub fn process(&mut self, audio: &[f32]) {
        if audio.is_empty() {
            return;
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

        let dt = audio.len() as f32 / self.fs;
        self.debug_timer += dt;
        if self.debug_timer >= 1.0 {
            debug!(target: "rhythm::dorsal", raw_flux, dt);
            self.debug_timer = 0.0;
        }
    }

    pub fn reset(&mut self) {
        self.lp_low_state = 0.0;
        self.lp_mid_state = 0.0;
        self.prev_band_energy = [0.0; 3];
        self.last_metrics = DorsalMetrics::default();
        self.debug_timer = 0.0;
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
        let mut max_drive: f32 = 0.0;
        let mut idx = 0;
        while idx < total {
            let end = (idx + hop).min(total);
            dorsal.process(&sig[idx..end]);
            let drive = (dorsal.last_metrics().flux * 500.0).tanh();
            max_drive = max_drive.max(drive);
            idx = end;
        }

        assert!(
            max_drive > 0.05,
            "expected flux drive to rise above 0.05, got {max_drive}"
        );
    }
}
