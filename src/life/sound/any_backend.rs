use crate::core::log2space::Log2Space;
use crate::life::individual::ArticulationSignal;
use crate::life::sound::control::VoiceControlBlock;
use crate::life::sound::harmonic_resonator_backend::HarmonicResonatorBackend;
use crate::life::sound::modal_engine::{ModalEngine, ModeShape};
use crate::life::sound::mode_utils::{
    cluster_spread_cents_from_public, modal_modes_from_ratios, modal_tilt_from_brightness,
};
use crate::life::sound::sine_osc_backend::SineOscBackend;
use crate::life::sound::{BodyKind, BodySnapshot};
use crate::synth::SynthError;

#[derive(Debug, Clone)]
pub enum AnyBackend {
    Sine(SineOscBackend),
    Harmonic(HarmonicResonatorBackend),
    Modal(ModalEngine),
}

impl AnyBackend {
    pub fn from_snapshot(fs: f32, snapshot: &BodySnapshot) -> Result<Self, SynthError> {
        match snapshot.kind {
            BodyKind::Sine => Ok(Self::Sine(SineOscBackend::new(fs)?)),
            BodyKind::Harmonic => Ok(Self::Harmonic(HarmonicResonatorBackend::from_snapshot(
                fs, snapshot,
            )?)),
            BodyKind::Modal => {
                let shape = modal_shape_from_snapshot(snapshot);
                Ok(Self::Modal(ModalEngine::new(fs, shape)?))
            }
        }
    }

    pub fn is_sine(&self) -> bool {
        matches!(self, Self::Sine(_))
    }

    pub fn supports_continuous_drive(&self) -> bool {
        !self.is_sine()
    }

    pub fn seed_modal_phases(&mut self, seed: u64) {
        match self {
            AnyBackend::Sine(backend) => backend.seed_phase(seed),
            AnyBackend::Harmonic(backend) => backend.seed_phases(seed),
            AnyBackend::Modal(engine) => engine.seed_modal_phases(seed),
        }
    }

    pub fn render_block(&mut self, drive: &[f32], ctrl: VoiceControlBlock, out: &mut [f32]) {
        match self {
            AnyBackend::Sine(backend) => backend.render_block(drive, ctrl, out),
            AnyBackend::Harmonic(backend) => backend.render_block(drive, ctrl, out),
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
            AnyBackend::Sine(backend) => backend.project_spectral(amps, space, signal),
            AnyBackend::Harmonic(backend) => backend.project_spectral(amps, space, signal),
            AnyBackend::Modal(engine) => engine.project_spectral(amps, space, signal),
        }
    }
}

fn modal_shape_from_snapshot(snapshot: &BodySnapshot) -> ModeShape {
    let fallback = [1.0f32];
    let ratios = snapshot.ratios.as_deref().unwrap_or(&fallback);
    let modal_tilt = modal_tilt_from_brightness(snapshot.brightness);
    let cluster_spread_cents = cluster_spread_cents_from_public(snapshot.spread);
    ModeShape::Modal {
        modes: modal_modes_from_ratios(ratios, modal_tilt, cluster_spread_cents, snapshot.voices),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::sound::control::{ControlRamp, VoiceControlBlock};

    #[test]
    fn any_backend_delegates_render_block() {
        let fs = 48_000.0;
        let snapshot = BodySnapshot {
            kind: BodyKind::Harmonic,
            amp_scale: 1.0,
            brightness: 0.6,
            spread: 0.0,
            voices: 1,
            noise_mix: 0.0,
            ratios: None,
        };
        let mut backend = AnyBackend::from_snapshot(fs, &snapshot).expect("backend");
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
