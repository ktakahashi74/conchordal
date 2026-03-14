use crate::core::modulation::NeuralRhythms;
use crate::life::articulation_envelope::step_attack_decay_sustain_envelope;
use crate::life::individual::{ArticulationSignal, ArticulationState};
use crate::life::phonation_engine::OnsetKick;

#[derive(Clone, Debug)]
pub enum RenderModulatorSpec {
    EntrainPulse {
        attack_step: f32,
        decay_rate: f32,
        sustain_level: f32,
        initial_state: RenderModulatorStateKind,
        initial_env_level: f32,
        alpha_gain: f32,
        beta_gain: f32,
        autonomous_pulse: Option<AutonomousPulseSpec>,
    },
    SeqGate {
        duration_sec: f32,
    },
    DroneSway {
        phase: f32,
        sway_rate: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderModulatorStateKind {
    Idle,
    Attack,
    Decay,
}

impl From<ArticulationState> for RenderModulatorStateKind {
    fn from(value: ArticulationState) -> Self {
        match value {
            ArticulationState::Idle => Self::Idle,
            ArticulationState::Attack => Self::Attack,
            ArticulationState::Decay => Self::Decay,
        }
    }
}

impl From<RenderModulatorStateKind> for ArticulationState {
    fn from(value: RenderModulatorStateKind) -> Self {
        match value {
            RenderModulatorStateKind::Idle => Self::Idle,
            RenderModulatorStateKind::Attack => Self::Attack,
            RenderModulatorStateKind::Decay => Self::Decay,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AutonomousPulseSpec {
    pub rate_hz: f32,
    pub phase_0_1: f32,
    pub retrigger: bool,
    pub env_open_threshold: f32,
    pub mag_threshold: f32,
    pub alpha_threshold: f32,
}

#[derive(Clone, Debug)]
struct AutonomousPulseRuntime {
    rate_hz: f32,
    phase_0_1: f32,
    retrigger: bool,
    env_open_threshold: f32,
    mag_threshold: f32,
    alpha_threshold: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct EntrainPulseModulator {
    attack_step: f32,
    decay_rate: f32,
    sustain_level: f32,
    state: ArticulationState,
    env_level: f32,
    alpha_gain: f32,
    beta_gain: f32,
    autonomous_pulse: Option<AutonomousPulseRuntime>,
}

#[derive(Clone, Debug)]
pub(crate) struct SeqGateModulator {
    timer: f32,
    duration_sec: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct DroneSwayModulator {
    phase: f32,
    sway_rate: f32,
}

#[derive(Clone, Debug)]
pub(crate) enum RenderModulator {
    EntrainPulse(EntrainPulseModulator),
    SeqGate(SeqGateModulator),
    DroneSway(DroneSwayModulator),
}

impl RenderModulator {
    pub fn from_spec(spec: RenderModulatorSpec) -> Self {
        match spec {
            RenderModulatorSpec::EntrainPulse {
                attack_step,
                decay_rate,
                sustain_level,
                initial_state,
                initial_env_level,
                alpha_gain,
                beta_gain,
                autonomous_pulse,
            } => Self::EntrainPulse(EntrainPulseModulator {
                attack_step,
                decay_rate,
                sustain_level: sustain_level.clamp(0.0, 1.0),
                state: initial_state.into(),
                env_level: initial_env_level.clamp(0.0, 1.0),
                alpha_gain,
                beta_gain,
                autonomous_pulse: autonomous_pulse.map(|pulse| AutonomousPulseRuntime {
                    rate_hz: pulse.rate_hz.max(0.01),
                    phase_0_1: pulse.phase_0_1.rem_euclid(1.0),
                    retrigger: pulse.retrigger,
                    env_open_threshold: pulse.env_open_threshold,
                    mag_threshold: pulse.mag_threshold,
                    alpha_threshold: pulse.alpha_threshold,
                }),
            }),
            RenderModulatorSpec::SeqGate { duration_sec } => Self::SeqGate(SeqGateModulator {
                timer: duration_sec.max(0.0),
                duration_sec: duration_sec.max(0.0),
            }),
            RenderModulatorSpec::DroneSway { phase, sway_rate } => {
                Self::DroneSway(DroneSwayModulator {
                    phase,
                    sway_rate: sway_rate.max(0.01),
                })
            }
        }
    }

    pub fn kick_planned(&mut self, kick: OnsetKick) {
        let strength = kick.strength.clamp(0.0, 1.0);
        if strength <= 0.0 {
            return;
        }
        match self {
            Self::EntrainPulse(modulator) => modulator.begin_attack(),
            Self::SeqGate(modulator) => modulator.timer = 0.0,
            Self::DroneSway(_) => {}
        }
    }

    pub fn process(&mut self, rhythms: &NeuralRhythms, dt: f32) -> ArticulationSignal {
        match self {
            Self::EntrainPulse(modulator) => modulator.process(rhythms, dt),
            Self::SeqGate(modulator) => modulator.process(rhythms, dt),
            Self::DroneSway(modulator) => modulator.process(rhythms, dt),
        }
    }
}

impl EntrainPulseModulator {
    fn begin_attack(&mut self) {
        self.env_level = 0.0;
        self.state = ArticulationState::Attack;
    }

    fn process(&mut self, rhythms: &NeuralRhythms, dt: f32) -> ArticulationSignal {
        let mut wrapped = 0u32;
        if let Some(pulse) = self.autonomous_pulse.as_mut() {
            pulse.phase_0_1 += pulse.rate_hz * dt.max(0.0);
            while pulse.phase_0_1 >= 1.0 {
                pulse.phase_0_1 -= 1.0;
                wrapped += u32::from(pulse.retrigger);
            }
        }
        if wrapped > 0 {
            for _ in 0..wrapped {
                let gate_ok = self.autonomous_pulse.as_ref().is_some_and(|pulse| {
                    rhythms.env_open > pulse.env_open_threshold
                        && rhythms.theta.mag > pulse.mag_threshold
                        && rhythms.theta.alpha > pulse.alpha_threshold
                });
                if self.state == ArticulationState::Idle && gate_ok {
                    self.begin_attack();
                }
            }
        }
        step_attack_decay_sustain_envelope(
            &mut self.state,
            &mut self.env_level,
            self.attack_step,
            self.decay_rate,
            self.sustain_level,
            dt,
        );
        ArticulationSignal {
            amplitude: self.env_level,
            is_active: self.state != ArticulationState::Idle && self.env_level > 1e-6,
            relaxation: rhythms.theta.alpha * self.alpha_gain,
            tension: rhythms.theta.beta * self.beta_gain,
        }
    }
}

impl SeqGateModulator {
    fn process(&mut self, rhythms: &NeuralRhythms, dt: f32) -> ArticulationSignal {
        self.timer += dt.max(0.0);
        let is_active = self.timer < self.duration_sec;
        ArticulationSignal {
            amplitude: if is_active { 1.0 } else { 0.0 },
            is_active,
            relaxation: rhythms.theta.alpha,
            tension: 0.0,
        }
    }
}

impl DroneSwayModulator {
    fn process(&mut self, rhythms: &NeuralRhythms, dt: f32) -> ArticulationSignal {
        let omega = std::f32::consts::TAU * self.sway_rate.max(0.01);
        self.phase = (self.phase + omega * dt.max(0.0)).rem_euclid(std::f32::consts::TAU);
        let lfo = 0.5 * (self.phase.sin() + 1.0);
        let relax_boost = 1.0 + rhythms.theta.alpha * 0.5;
        let amplitude = ((0.3 + 0.7 * lfo) * relax_boost).clamp(0.0, 1.0);
        ArticulationSignal {
            amplitude,
            is_active: true,
            relaxation: rhythms.theta.alpha,
            tension: rhythms.theta.beta * 0.25,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_rhythms() -> NeuralRhythms {
        let mut rhythms = NeuralRhythms::default();
        rhythms.theta.alpha = 0.7;
        rhythms.theta.beta = 0.4;
        rhythms
    }

    #[test]
    fn entrain_pulse_kick_starts_attack_decay() {
        let spec = RenderModulatorSpec::EntrainPulse {
            attack_step: 100.0,
            decay_rate: 10.0,
            sustain_level: 0.0,
            initial_state: RenderModulatorStateKind::Idle,
            initial_env_level: 0.0,
            alpha_gain: 0.5,
            beta_gain: 0.25,
            autonomous_pulse: None,
        };
        let mut modulator = RenderModulator::from_spec(spec);
        modulator.kick_planned(OnsetKick { strength: 1.0 });
        let signal = modulator.process(&default_rhythms(), 1.0 / 48_000.0);
        assert!(signal.amplitude > 0.0);
        assert!(signal.is_active);
    }

    #[test]
    fn entrain_pulse_hold_autonomous_phase_retriggers_when_idle() {
        let spec = RenderModulatorSpec::EntrainPulse {
            attack_step: 1000.0,
            decay_rate: 2000.0,
            sustain_level: 0.0,
            initial_state: RenderModulatorStateKind::Idle,
            initial_env_level: 0.0,
            alpha_gain: 0.5,
            beta_gain: 0.25,
            autonomous_pulse: Some(AutonomousPulseSpec {
                rate_hz: 5.0,
                phase_0_1: 0.99,
                retrigger: true,
                env_open_threshold: 0.1,
                mag_threshold: 0.1,
                alpha_threshold: 0.1,
            }),
        };
        let mut modulator = RenderModulator::from_spec(spec);
        let mut rhythms = default_rhythms();
        rhythms.env_open = 1.0;
        rhythms.theta.mag = 1.0;
        rhythms.theta.alpha = 1.0;
        let signal = modulator.process(&rhythms, 0.01);
        assert!(signal.is_active);
        assert!(signal.amplitude > 0.0);
    }

    #[test]
    fn entrain_pulse_uses_alpha_beta_gains_from_spec() {
        let spec = RenderModulatorSpec::EntrainPulse {
            attack_step: 100.0,
            decay_rate: 10.0,
            sustain_level: 0.0,
            initial_state: RenderModulatorStateKind::Attack,
            initial_env_level: 0.5,
            alpha_gain: 0.25,
            beta_gain: 0.75,
            autonomous_pulse: None,
        };
        let mut modulator = RenderModulator::from_spec(spec);
        let rhythms = default_rhythms();
        let signal = modulator.process(&rhythms, 0.0);
        assert!((signal.relaxation - rhythms.theta.alpha * 0.25).abs() < 1e-6);
        assert!((signal.tension - rhythms.theta.beta * 0.75).abs() < 1e-6);
    }

    #[test]
    fn seq_gate_restarts_on_kick() {
        let spec = RenderModulatorSpec::SeqGate { duration_sec: 0.1 };
        let mut modulator = RenderModulator::from_spec(spec);
        let rhythms = default_rhythms();
        let signal = modulator.process(&rhythms, 0.2);
        assert!(!signal.is_active);
        modulator.kick_planned(OnsetKick { strength: 1.0 });
        let signal = modulator.process(&rhythms, 0.0);
        assert!(signal.is_active);
    }

    #[test]
    fn drone_sway_remains_active() {
        let spec = RenderModulatorSpec::DroneSway {
            phase: 0.0,
            sway_rate: 0.1,
        };
        let mut modulator = RenderModulator::from_spec(spec);
        let signal = modulator.process(&default_rhythms(), 0.01);
        assert!(signal.is_active);
        assert!(signal.amplitude > 0.0);
    }

    #[test]
    fn entrain_pulse_hold_gate_blocks_retrigger_when_mag_below_threshold() {
        let spec = RenderModulatorSpec::EntrainPulse {
            attack_step: 1000.0,
            decay_rate: 2000.0,
            sustain_level: 0.0,
            initial_state: RenderModulatorStateKind::Idle,
            initial_env_level: 0.0,
            alpha_gain: 0.5,
            beta_gain: 0.25,
            autonomous_pulse: Some(AutonomousPulseSpec {
                rate_hz: 2.0,
                phase_0_1: 0.0,
                retrigger: true,
                env_open_threshold: 0.1,
                mag_threshold: 0.5,
                alpha_threshold: 0.1,
            }),
        };
        let mut modulator = RenderModulator::from_spec(spec);
        let mut rhythms = default_rhythms();
        rhythms.env_open = 1.0;
        rhythms.theta.mag = 0.1;
        rhythms.theta.alpha = 1.0;
        let mut rises = 0;
        let mut prev_active = false;
        for _ in 0..2_000 {
            let signal = modulator.process(&rhythms, 0.001);
            if signal.is_active && !prev_active {
                rises += 1;
            }
            prev_active = signal.is_active;
        }
        assert_eq!(rises, 0);
    }

    #[test]
    fn entrain_pulse_hold_gate_blocks_retrigger_when_alpha_below_threshold() {
        let spec = RenderModulatorSpec::EntrainPulse {
            attack_step: 1000.0,
            decay_rate: 2000.0,
            sustain_level: 0.0,
            initial_state: RenderModulatorStateKind::Idle,
            initial_env_level: 0.0,
            alpha_gain: 0.5,
            beta_gain: 0.25,
            autonomous_pulse: Some(AutonomousPulseSpec {
                rate_hz: 2.0,
                phase_0_1: 0.0,
                retrigger: true,
                env_open_threshold: 0.1,
                mag_threshold: 0.1,
                alpha_threshold: 0.5,
            }),
        };
        let mut modulator = RenderModulator::from_spec(spec);
        let mut rhythms = default_rhythms();
        rhythms.env_open = 1.0;
        rhythms.theta.mag = 1.0;
        rhythms.theta.alpha = 0.1;
        let mut rises = 0;
        let mut prev_active = false;
        for _ in 0..2_000 {
            let signal = modulator.process(&rhythms, 0.001);
            if signal.is_active && !prev_active {
                rises += 1;
            }
            prev_active = signal.is_active;
        }
        assert_eq!(rises, 0);
    }

    #[test]
    fn entrain_pulse_hold_gate_allows_retrigger_when_thresholds_satisfied() {
        let spec = RenderModulatorSpec::EntrainPulse {
            attack_step: 1000.0,
            decay_rate: 2000.0,
            sustain_level: 0.0,
            initial_state: RenderModulatorStateKind::Idle,
            initial_env_level: 0.0,
            alpha_gain: 0.5,
            beta_gain: 0.25,
            autonomous_pulse: Some(AutonomousPulseSpec {
                rate_hz: 2.0,
                phase_0_1: 0.0,
                retrigger: true,
                env_open_threshold: 0.1,
                mag_threshold: 0.1,
                alpha_threshold: 0.1,
            }),
        };
        let mut modulator = RenderModulator::from_spec(spec);
        let mut rhythms = default_rhythms();
        rhythms.env_open = 1.0;
        rhythms.theta.mag = 1.0;
        rhythms.theta.alpha = 1.0;
        let mut rises = 0;
        let mut prev_active = false;
        for _ in 0..2_000 {
            let signal = modulator.process(&rhythms, 0.001);
            if signal.is_active && !prev_active {
                rises += 1;
            }
            prev_active = signal.is_active;
        }
        assert!(rises >= 2, "expected repeated gated rises, got {rises}");
    }
}
