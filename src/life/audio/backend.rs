use crate::core::log2space::Log2Space;
use crate::life::audio::control::VoiceControlBlock;
use crate::life::individual::ArticulationSignal;

pub trait Backend {
    fn render_block(&mut self, drive: &[f32], ctrl: VoiceControlBlock, out: &mut [f32]);
    fn project_spectral(
        &mut self,
        amps: &mut [f32],
        space: &Log2Space,
        signal: &ArticulationSignal,
    );
}
