use crate::core::log2space::Log2Space;
use crate::core::mode_pattern::ModePattern;
use crate::life::individual::ArticulationSignal;
use crate::life::individual::sound_body::{
    AnySoundBody, SoundBody, SoundBodyBuildInput, SoundBodyFactory, register_sound_body_factory,
};
use crate::life::sound::control::{ControlRamp, VoiceControlBlock};
use crate::life::sound::mode_utils::{modal_modes_from_ratios, sanitize_mode_ratios};
use crate::life::sound::{BodyKind, BodySnapshot, ModalEngine, ModeShape};
use rand::rngs::SmallRng;
use std::sync::{Arc, Once};

#[derive(Debug)]
pub struct ModalBody {
    base_freq_hz: f32,
    amp: f32,
    fs: f32,
    brightness: f32,
    width: f32,
    base_ratios: Arc<[f32]>,
    engine: ModalEngine,
}

impl ModalBody {
    const TIMBRE_EPS: f32 = 1.0e-4;

    fn new(
        fs: f32,
        base_freq_hz: f32,
        amp: f32,
        ratios: Vec<f32>,
        brightness: f32,
        width: f32,
    ) -> Self {
        let fs = if fs.is_finite() && fs > 0.0 {
            fs
        } else {
            48_000.0
        };
        let brightness = brightness.clamp(0.0, 1.0);
        let width = width.clamp(0.0, 1.0);
        let base_ratios = Arc::<[f32]>::from(sanitize_mode_ratios(ratios));
        let modes = modal_modes_from_ratios(&base_ratios, brightness, width);
        let shape = ModeShape::Modal { modes };
        let engine = ModalEngine::new(fs, shape).unwrap_or_else(|_| {
            ModalEngine::new(
                fs,
                ModeShape::Sine {
                    t60_s: 0.8,
                    out_gain: 1.0,
                    in_gain: 1.0,
                },
            )
            .expect("modal sine fallback")
        });

        Self {
            base_freq_hz: base_freq_hz.max(1.0),
            amp,
            fs,
            brightness,
            width,
            base_ratios,
            engine,
        }
    }

    fn control_block(&self, amp: f32) -> VoiceControlBlock {
        VoiceControlBlock {
            pitch_hz: ControlRamp {
                start: self.base_freq_hz.max(1.0),
                step: 0.0,
            },
            amp: ControlRamp {
                start: amp.max(0.0),
                step: 0.0,
            },
        }
    }

    fn rebuild_engine_shape(&mut self) {
        let modes = modal_modes_from_ratios(&self.base_ratios, self.brightness, self.width);
        if let Ok(engine) = ModalEngine::new(self.fs, ModeShape::Modal { modes }) {
            self.engine = engine;
        }
    }

    fn seed_modal_phases(&mut self, seed: u64) {
        self.engine.seed_modal_phases(seed);
    }
}

impl SoundBody for ModalBody {
    fn base_freq_hz(&self) -> f32 {
        self.base_freq_hz
    }

    fn set_freq(&mut self, freq: f32) {
        self.base_freq_hz = freq.max(1.0);
    }

    fn set_pitch_log2(&mut self, log_freq: f32) {
        self.base_freq_hz = 2.0f32.powf(log_freq).max(1.0);
    }

    fn set_amp(&mut self, amp: f32) {
        self.amp = amp;
    }

    fn amp(&self) -> f32 {
        self.amp
    }

    fn articulate_wave(
        &mut self,
        sample: &mut f32,
        _fs: f32,
        _dt: f32,
        signal: &ArticulationSignal,
    ) {
        if !signal.is_active || signal.amplitude <= 0.0 || self.amp <= 0.0 {
            return;
        }
        let drive = [signal.amplitude.max(0.0)];
        let mut out = [0.0f32; 1];
        self.engine
            .render_block(&drive, self.control_block(self.amp), &mut out);
        *sample += out[0];
    }

    fn project_spectral_body(
        &mut self,
        amps: &mut [f32],
        space: &Log2Space,
        signal: &ArticulationSignal,
    ) {
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        // Keep the internal bank's last pitch synchronized before projection.
        let mut scratch = [0.0f32; 1];
        self.engine
            .render_block(&[0.0], self.control_block(0.0), &mut scratch);
        self.engine.project_spectral(amps, space, signal);
    }

    fn apply_timbre_controls(
        &mut self,
        brightness: f32,
        _inharmonic: f32,
        width: f32,
        _motion: f32,
    ) {
        let next_brightness = brightness.clamp(0.0, 1.0);
        let next_width = width.clamp(0.0, 1.0);
        if (self.brightness - next_brightness).abs() <= Self::TIMBRE_EPS
            && (self.width - next_width).abs() <= Self::TIMBRE_EPS
        {
            return;
        }
        self.brightness = next_brightness;
        self.width = next_width;
        self.rebuild_engine_shape();
    }

    fn snapshot(&self) -> BodySnapshot {
        BodySnapshot {
            kind: BodyKind::Modal,
            amp_scale: 1.0,
            brightness: self.brightness,
            width: self.width,
            noise_mix: 0.0,
            ratios: Some(self.base_ratios.clone()),
        }
    }
}

#[derive(Debug)]
struct ModalBodyFactory;

impl SoundBodyFactory for ModalBodyFactory {
    fn build(&self, input: &SoundBodyBuildInput<'_>, rng: &mut SmallRng) -> AnySoundBody {
        let fallback_space;
        let eval_space = if let Some(frame) = input.landscape {
            &frame.space
        } else {
            fallback_space = Log2Space::new(55.0, 8000.0, 96);
            &fallback_space
        };

        let pattern = input
            .control
            .body
            .modes
            .clone()
            .unwrap_or_else(ModePattern::harmonic_modes);
        let ratios = pattern.eval(input.base_freq_hz, eval_space, input.landscape, rng);
        let timbre = &input.control.body.timbre;
        let body = ModalBody::new(
            input.fs,
            input.base_freq_hz,
            input.control.body.amp,
            ratios,
            timbre.brightness,
            timbre.width,
        );
        let mut body = body;
        body.seed_modal_phases(rand::RngCore::next_u64(rng));
        AnySoundBody::from_dyn(Box::new(body))
    }
}

pub fn register_modal() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        register_sound_body_factory("modal", Arc::new(ModalBodyFactory));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modal_snapshot_keeps_base_ratios_without_detune() {
        let body = ModalBody::new(48_000.0, 220.0, 0.2, vec![1.0, 2.756, 5.404], 0.62, 0.35);
        let snapshot = body.snapshot();
        let ratios = snapshot.ratios.expect("ratios");
        assert_eq!(snapshot.kind, BodyKind::Modal);
        assert!((snapshot.width - 0.35).abs() <= 1.0e-6);
        assert!((snapshot.noise_mix - 0.0).abs() <= 1.0e-6);
        assert!((ratios[0] - 1.0).abs() <= 1.0e-6);
        assert!((ratios[1] - 2.756).abs() <= 1.0e-6);
        assert!((ratios[2] - 5.404).abs() <= 1.0e-6);
    }
}
