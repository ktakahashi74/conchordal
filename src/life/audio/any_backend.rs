use crate::core::log2space::Log2Space;
use crate::life::audio::backend::Backend;
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
}

impl Backend for AnyBackend {
    fn render_block(&mut self, drive: &[f32], ctrl: VoiceControlBlock, out: &mut [f32]) {
        match self {
            AnyBackend::Modal(engine) => engine.render_block(drive, ctrl, out),
        }
    }

    fn project_spectral(
        &mut self,
        amps: &mut [f32],
        space: &Log2Space,
        signal: &ArticulationSignal,
    ) {
        match self {
            AnyBackend::Modal(engine) => engine.project_spectral(amps, space, signal),
        }
    }
}
