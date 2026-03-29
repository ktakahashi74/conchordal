use crate::core::log2space::Log2Space;
use crate::life::sound::control::ToneControlBlock;
use crate::life::sound::spectral::add_log2_energy;
use crate::life::voice::ArticulationSignal;
use crate::synth::SynthError;
use std::f32::consts::TAU;

#[derive(Debug, Clone)]
pub struct SineOscBackend {
    fs: f32,
    phase_rad: f32,
    last_pitch_hz: f32,
}

impl SineOscBackend {
    pub fn new(fs: f32) -> Result<Self, SynthError> {
        if !fs.is_finite() || fs <= 0.0 {
            return Err(SynthError::InvalidSampleRate);
        }
        Ok(Self {
            fs,
            phase_rad: 0.0,
            last_pitch_hz: 440.0,
        })
    }

    pub fn seed_phase(&mut self, seed: u64) {
        self.phase_rad = splitmix64_unit_f32(seed) * TAU;
    }

    pub fn render_block(&mut self, _drive: &[f32], ctrl: ToneControlBlock, out: &mut [f32]) {
        if out.is_empty() {
            return;
        }
        for (idx, y) in out.iter_mut().enumerate() {
            let pitch_hz = (ctrl.pitch_hz.start + ctrl.pitch_hz.step * idx as f32).max(1.0);
            let amp = (ctrl.amp.start + ctrl.amp.step * idx as f32).max(0.0);
            let dphi = TAU * pitch_hz / self.fs;
            self.phase_rad = (self.phase_rad + dphi).rem_euclid(TAU);
            self.last_pitch_hz = pitch_hz;
            *y += amp * self.phase_rad.sin();
        }
    }

    pub fn project_spectral(
        &mut self,
        amps: &mut [f32],
        space: &Log2Space,
        signal: &ArticulationSignal,
    ) {
        debug_assert_eq!(amps.len(), space.n_bins());
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        if self.last_pitch_hz.is_finite() && self.last_pitch_hz > 0.0 {
            add_log2_energy(amps, space, self.last_pitch_hz, signal.amplitude.max(0.0));
        }
    }
}

fn splitmix64_unit_f32(seed: u64) -> f32 {
    const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let bits = z >> 11;
    (bits as f64 * SCALE) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::sound::control::ControlRamp;

    #[test]
    fn sine_backend_renders_finite_signal() {
        let mut backend = SineOscBackend::new(48_000.0).expect("backend");
        backend.seed_phase(123);
        let mut out = [0.0f32; 8];
        backend.render_block(
            &[0.0; 8],
            ToneControlBlock {
                pitch_hz: ControlRamp {
                    start: 440.0,
                    step: 0.0,
                },
                amp: ControlRamp {
                    start: 0.5,
                    step: 0.0,
                },
            },
            &mut out,
        );
        assert!(out.iter().all(|s| s.is_finite()));
        assert!(out.iter().any(|s| s.abs() > 1e-6));
    }
}
