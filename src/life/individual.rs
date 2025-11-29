use super::scenario::EnvelopeConfig;

/// Hybrid synthesis agents render both time-domain audio and a spectral "body".
pub trait AudioAgent: Send + Sync + 'static {
    fn id(&self) -> u64;
    /// Generate audio samples (mixing into the buffer).
    /// Handles continuous phase to prevent clicks.
    fn render_wave(&mut self, buffer: &mut [f32], fs: f32);

    /// Project the agent's "body" (energy) onto the spectral bins.
    /// Used for Landscape analysis (Roughness/Harmonicity).
    fn render_body(&self, spectrum: &mut [f32], nfft: usize, fs: f32, current_frame: u64);

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
    pub envelope: EnvelopeConfig,
    phase: f32,
    alive: bool,
}

impl PureToneAgent {
    pub fn new(
        id: u64,
        freq_hz: f32,
        amp: f32,
        start_frame: u64,
        envelope: Option<EnvelopeConfig>,
    ) -> Self {
        Self {
            id,
            freq_hz,
            amp,
            start_frame,
            envelope: envelope.unwrap_or(EnvelopeConfig {
                attack_sec: 0.01,
                decay_sec: 0.2,
                sustain_level: 0.2,
            }),
            phase: 0.0,
            alive: true,
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
        self.alive = false;
    }
}

impl AudioAgent for PureToneAgent {
    fn id(&self) -> u64 {
        self.id
    }

    fn render_wave(&mut self, buffer: &mut [f32], fs: f32) {
        if !self.alive {
            return;
        }
        let omega = 2.0 * std::f32::consts::PI * self.freq_hz / fs;
        for s in buffer.iter_mut() {
            let sin = self.phase.sin();
            *s += self.amp * sin;
            self.phase = self.phase + omega;
            if self.phase > std::f32::consts::TAU {
                self.phase -= std::f32::consts::TAU;
            }
        }
    }

    fn render_body(&self, spectrum: &mut [f32], nfft: usize, fs: f32, current_frame: u64) {
        if !self.alive {
            return;
        }
        let elapsed_frames = current_frame.saturating_sub(self.start_frame);
        let t_sec = elapsed_frames as f32 * (nfft as f32 / fs);

        let env = &self.envelope;
        let attack = env.attack_sec.max(1e-6);
        let decay = env.decay_sec.max(1e-6);
        let sustain = env.sustain_level.clamp(0.0, 1.0);

        let gain = if t_sec < attack {
            (t_sec / attack).min(1.0)
        } else {
            let t_decay = t_sec - attack;
            sustain + (1.0 - sustain) * (-t_decay / decay).exp()
        };

        let bin_f = self.freq_hz * nfft as f32 / fs;
        let k = bin_f.round() as isize;
        if k >= 0 && (k as usize) < spectrum.len() {
            spectrum[k as usize] += self.amp * gain;
        }
    }

    fn is_alive(&self) -> bool {
        self.alive
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
    use crate::life::scenario::EnvelopeConfig;

    fn approx_eq(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "expected {b}, got {a}, tol {tol}");
    }

    #[test]
    fn envelope_applies_attack_and_decay() {
        let env = EnvelopeConfig {
            attack_sec: 0.1,
            decay_sec: 0.2,
            sustain_level: 0.5,
        };
        let agent = PureToneAgent::new(1, 100.0, 1.0, 10, Some(env));
        let fs = 1000.0;
        let nfft = 100; // treated as hop for envelope timing here
        let mut body = vec![0.0f32; 64];

        // At spawn frame: gain should be ~0
        agent.render_body(&mut body, nfft, fs, 10);
        approx_eq(body[10], 0.0, 1e-4);

        // One frame later (0.1s): end of attack -> gain ~1
        let mut body = vec![0.0f32; 64];
        agent.render_body(&mut body, nfft, fs, 11);
        approx_eq(body[10], 1.0, 1e-3);

        // After entering decay: should be between sustain and 1.0
        let mut body = vec![0.0f32; 64];
        agent.render_body(&mut body, nfft, fs, 12);
        // expected sustain + (1-sustain)*exp(-0.1/0.2) ≈ 0.803
        approx_eq(body[10], 0.803, 5e-3);
    }
}
