use super::lifecycle::Lifecycle;

/// Hybrid synthesis agents render both time-domain audio and a spectral "body".
pub trait AudioAgent: Send + Sync + 'static {
    fn id(&self) -> u64;
    /// Generate audio samples (mixing into the buffer).
    /// Handles continuous phase to prevent clicks.
    fn render_wave(&mut self, buffer: &mut [f32], fs: f32, current_frame: u64, dt_sec: f32);

    /// Project the agent's "body" (energy) onto the spectral bins.
    /// Used for Landscape analysis (Roughness/Harmonicity).
    fn render_spectrum(
        &mut self,
        amps: &mut [f32],
        fs: f32,
        nfft: usize,
        current_frame: u64,
        dt_sec: f32,
    );

    fn is_alive(&self) -> bool;
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

#[derive(Debug)]
pub struct PureToneAgent {
    pub id: u64,
    pub freq_hz: f32,
    pub amp: f32,
    pub start_frame: u64,
    pub lifecycle: Box<dyn Lifecycle>,
    phase: f32,
    last_gain: f32,
}

impl PureToneAgent {
    pub fn new(
        id: u64,
        freq_hz: f32,
        amp: f32,
        start_frame: u64,
        lifecycle: Box<dyn super::lifecycle::Lifecycle>,
    ) -> Self {
        Self {
            id,
            freq_hz,
            amp,
            start_frame,
            lifecycle,
            phase: 0.0,
            last_gain: 1.0,
        }
    }

    pub fn set_amp(&mut self, amp: f32) {
        self.amp = amp;
    }

    pub fn set_freq(&mut self, freq: f32) {
        self.freq_hz = freq;
    }

    pub fn set_phase(&mut self, phase: f32) {
        self.phase = phase;
    }

    pub fn kill(&mut self) {
        // Lifecycle decides liveness; no-op here.
    }
}

impl AudioAgent for PureToneAgent {
    fn id(&self) -> u64 {
        self.id
    }

    fn render_wave(&mut self, buffer: &mut [f32], fs: f32, current_frame: u64, dt_sec: f32) {
        let age = current_frame.saturating_sub(self.start_frame) as f32 * dt_sec;
        let gain = self.lifecycle.process(dt_sec, age);
        self.last_gain = gain;
        let omega = 2.0 * std::f32::consts::PI * self.freq_hz / fs;
        for s in buffer.iter_mut() {
            let sin = self.phase.sin();
            *s += self.amp * gain * sin;
            self.phase = self.phase + omega;
            if self.phase > std::f32::consts::TAU {
                self.phase -= std::f32::consts::TAU;
            }
        }
    }

    fn render_spectrum(
        &mut self,
        amps: &mut [f32],
        fs: f32,
        nfft: usize,
        _current_frame: u64,
        _dt_sec: f32,
    ) {
        // Use the gain computed during render_wave to avoid double decay per hop.
        let gain = self.last_gain;

        let bin_f = self.freq_hz * nfft as f32 / fs;
        let k = bin_f.round() as isize;
        if k >= 0 && (k as usize) < amps.len() {
            amps[k as usize] += self.amp * gain;
        }
    }

    fn is_alive(&self) -> bool {
        self.lifecycle.is_alive()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::lifecycle::LifecycleConfig;

    fn approx_eq(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "expected {b}, got {a}, tol {tol}");
    }

    #[test]
    fn decay_lifecycle_drops_gain_per_hop() {
        let lifecycle = LifecycleConfig::Decay {
            initial_energy: 1.0,
            half_life_sec: 0.2,
        }
        .create_lifecycle();
        let mut agent = PureToneAgent::new(1, 100.0, 1.0, 0, lifecycle);
        agent.set_phase(std::f32::consts::FRAC_PI_2); // sin = 1 at phase π/2

        let fs = 1000.0;
        let dt_sec = 0.1; // hop = 0.1 s

        let mut buf = vec![0.0f32; 1];
        agent.render_wave(&mut buf, fs, 0, dt_sec);
        // gain should be sqrt(0.5) ≈ 0.707
        approx_eq(buf[0], 0.707, 1e-3);

        agent.set_phase(std::f32::consts::FRAC_PI_2);
        let mut buf = vec![0.0f32; 1];
        agent.render_wave(&mut buf, fs, 1, dt_sec);
        approx_eq(buf[0], 0.5, 1e-3);
    }
}
