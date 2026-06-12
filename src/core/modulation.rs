use std::f32::consts::TAU;

use crate::core::meter::MeterState;
use crate::core::phase::wrap_pm_pi;

fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    if hi <= lo {
        return 0.0;
    }
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RhythmBand {
    pub phase: f32,   // wrapped [-pi, pi]
    pub freq_hz: f32, // tracked tempo
    pub mag: f32,     // presence
    pub alpha: f32,   // precision
    pub beta: f32,    // prediction error
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NeuralRhythms {
    pub theta: RhythmBand,
    pub delta: RhythmBand,
    pub env_open: f32,
    pub env_level: f32,
}

fn compute_env_open(delta_phase: f32, delta_mag: f32, delta_alpha: f32) -> f32 {
    let env_wave = 0.5 + 0.5 * delta_phase.cos();
    let delta_conf = (delta_mag * delta_alpha).clamp(0.0, 1.0);
    let min_depth = 0.15;
    let mag_scale = smoothstep(0.02, 0.08, delta_mag);
    let depth = (min_depth + (1.0 - min_depth) * delta_conf) * (0.2 + 0.8 * mag_scale);
    (1.0 - depth * (1.0 - env_wave)).clamp(0.0, 1.0)
}

impl NeuralRhythms {
    /// Derive the production rhythm from the meter core's beat structure.
    ///
    /// The beat (tactus) drives the slow `delta` band that gates the
    /// articulation envelope, and its entrained subdivision is the note-rate
    /// `theta` band the motor oscillator locks to. Entrainment confidence sets
    /// precision (`alpha`) and its complement is prediction error (`beta`), so a
    /// confidently entrained beat sharpens the envelope toward the downbeat
    /// while a free-running beat leaves the gate open.
    pub fn from_meter_state(state: &MeterState) -> Self {
        let beat = state.beat;
        let sub = state.subdivision;
        let conf = beat.confidence.clamp(0.0, 1.0);
        let theta = RhythmBand {
            phase: sub.phase,
            freq_hz: sub.freq_hz,
            mag: conf,
            alpha: conf,
            beta: (1.0 - conf).clamp(0.0, 1.0),
        };
        let delta = RhythmBand {
            phase: beat.phase,
            freq_hz: beat.freq_hz,
            mag: conf,
            alpha: conf,
            beta: (1.0 - conf).clamp(0.0, 1.0),
        };
        let env_open = compute_env_open(delta.phase, delta.mag, delta.alpha);
        NeuralRhythms {
            theta,
            delta,
            env_open,
            env_level: conf,
        }
    }

    pub fn advance_in_place(&mut self, dt: f32) {
        if dt <= 0.0 {
            return;
        }
        self.theta.phase = wrap_pm_pi(self.theta.phase + TAU * self.theta.freq_hz * dt);
        self.delta.phase = wrap_pm_pi(self.delta.phase + TAU * self.delta.freq_hz * dt);
        self.env_open = compute_env_open(self.delta.phase, self.delta.mag, self.delta.alpha);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neuralrhythms_advance_wraps_phase() {
        let mut rhythms = NeuralRhythms {
            theta: RhythmBand {
                phase: 0.0,
                freq_hz: 1.0,
                mag: 1.0,
                alpha: 1.0,
                beta: 0.0,
            },
            delta: RhythmBand {
                phase: 0.0,
                freq_hz: 1.0,
                mag: 1.0,
                alpha: 1.0,
                beta: 0.0,
            },
            env_open: 1.0,
            env_level: 0.5,
        };
        for _ in 0..4 {
            rhythms.advance_in_place(0.25);
        }
        assert!(
            rhythms.theta.phase.abs() < 1e-4,
            "theta phase should wrap to 0"
        );
        assert!(
            rhythms.delta.phase.abs() < 1e-4,
            "delta phase should wrap to 0"
        );
    }

    #[test]
    fn neuralrhythms_env_open_updates_with_delta_phase() {
        let mut rhythms = NeuralRhythms {
            theta: RhythmBand {
                phase: 0.0,
                freq_hz: 1.0,
                mag: 1.0,
                alpha: 1.0,
                beta: 0.0,
            },
            delta: RhythmBand {
                phase: 0.0,
                freq_hz: 1.0,
                mag: 1.0,
                alpha: 1.0,
                beta: 0.0,
            },
            env_open: 1.0,
            env_level: 0.5,
        };
        let env_before = rhythms.env_open;
        rhythms.advance_in_place(0.1);
        let env_after = rhythms.env_open;
        assert!(env_after.is_finite());
        assert!((0.0..=1.0).contains(&env_after));
        assert!(
            (env_before - env_after).abs() > 1e-4,
            "env_open should respond to delta phase advance"
        );
    }
}
