use crate::core::log2space::Log2Space;
use crate::life::audio::control::VoiceControlBlock;
use crate::life::audio::modal_engine::{ModalEngine, ModeShape};
use crate::life::individual::ArticulationSignal;
use crate::synth::SynthError;

#[derive(Debug, Clone)]
pub enum AnyBackend {
    Modal(ModalEngine),
}

impl AnyBackend {
    pub fn from_shape(fs: f32, shape: ModeShape) -> Result<Self, SynthError> {
        Ok(Self::Modal(ModalEngine::new(fs, shape)?))
    }

    pub fn render_block(&mut self, drive: &[f32], ctrl: VoiceControlBlock, out: &mut [f32]) {
        match self {
            AnyBackend::Modal(engine) => engine.render_block(drive, ctrl, out),
        }
    }

    pub fn project_spectral(
        &mut self,
        amps: &mut [f32],
        space: &Log2Space,
        signal: &ArticulationSignal,
    ) {
        match self {
            AnyBackend::Modal(engine) => engine.project_spectral(amps, space, signal),
        }
    }

    #[cfg(test)]
    pub(crate) fn debug_last_modes_len(&self) -> usize {
        match self {
            AnyBackend::Modal(engine) => engine.last_modes_len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::audio::control::{ControlRamp, VoiceControlBlock};

    #[test]
    fn any_backend_delegates_render_block() {
        let fs = 48_000.0;
        let shape = ModeShape::Sine {
            t60_s: 0.3,
            out_gain: 1.0,
            in_gain: 1.0,
        };
        let mut backend = AnyBackend::from_shape(fs, shape).expect("backend");
        let drive = [1.0, 0.0, 0.0, 0.0];
        let ctrl = VoiceControlBlock {
            pitch_hz: ControlRamp {
                start: 440.0,
                step: 0.0,
            },
            amp: ControlRamp {
                start: 1.0,
                step: 0.0,
            },
        };
        let mut out = [0.0f32; 4];
        backend.render_block(&drive, ctrl, &mut out);
        assert!(out.iter().all(|s| s.is_finite()));
        assert!(out.iter().any(|s| s.abs() > 1e-6));
    }
}
