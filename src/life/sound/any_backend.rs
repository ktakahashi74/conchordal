use crate::core::log2space::Log2Space;
use crate::life::individual::ArticulationSignal;
use crate::life::sound::control::VoiceControlBlock;
use crate::life::sound::modal_engine::{ModalEngine, ModeShape};
use crate::life::sound::mode_utils::{
    cluster_spread_cents_from_public, modal_modes_from_ratios, modal_tilt_from_brightness,
};
use crate::life::sound::oscillator_bank::OscillatorBank;
use crate::life::sound::{BodyKind, BodySnapshot};
use crate::synth::SynthError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriveMode {
    None,
    Deterministic,
    Noisy,
}

#[derive(Debug, Clone)]
pub enum AnyBackend {
    Oscillator(OscillatorBank),
    Resonator(ModalEngine),
}

impl AnyBackend {
    pub fn from_snapshot(fs: f32, snapshot: &BodySnapshot) -> Result<Self, SynthError> {
        match snapshot.kind {
            BodyKind::Sine | BodyKind::Harmonic => Ok(Self::Oscillator(
                OscillatorBank::from_snapshot(fs, snapshot)?,
            )),
            BodyKind::Modal => {
                let shape = modal_shape_from_snapshot(snapshot);
                Ok(Self::Resonator(ModalEngine::new(fs, shape)?))
            }
        }
    }

    pub fn is_sine(&self) -> bool {
        matches!(self, Self::Oscillator(backend) if backend.is_sine())
    }

    pub fn drive_mode(&self) -> DriveMode {
        match self {
            AnyBackend::Oscillator(backend) if backend.is_sine() => DriveMode::None,
            AnyBackend::Oscillator(_) => DriveMode::Deterministic,
            AnyBackend::Resonator(_) => DriveMode::Noisy,
        }
    }

    pub fn supports_continuous_drive(&self) -> bool {
        !matches!(self.drive_mode(), DriveMode::None)
    }

    pub fn seed_modal_phases(&mut self, seed: u64) {
        match self {
            AnyBackend::Oscillator(backend) => backend.seed_phases(seed),
            AnyBackend::Resonator(engine) => engine.seed_modal_phases(seed),
        }
    }

    pub fn render_block(&mut self, drive: &[f32], ctrl: VoiceControlBlock, out: &mut [f32]) {
        match self {
            AnyBackend::Oscillator(backend) => backend.render_block(drive, ctrl, out),
            AnyBackend::Resonator(engine) => engine.render_block(drive, ctrl, out),
        }
    }

    pub fn project_spectral(
        &mut self,
        amps: &mut [f32],
        space: &Log2Space,
        signal: &ArticulationSignal,
    ) {
        match self {
            AnyBackend::Oscillator(backend) => backend.project_spectral(amps, space, signal),
            AnyBackend::Resonator(engine) => engine.project_spectral(amps, space, signal),
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
            inharmonic: 0.0,
            spread: 0.0,
            voices: 1,
            motion: 0.0,
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
