use crate::life::adaptation::AdaptationConfig;
use crate::life::control::AdaptationControl;
use crate::scenario::{
    DurationConfig, DurationSpec, OnsetConfig, PhonationClockConfig, PhonationConfig,
    PhonationMode, PhonationSpec, PhonationTiming, RhythmRole,
};

pub(crate) fn adaptation_config_from_control(control: &AdaptationControl) -> AdaptationConfig {
    let adaptation = control.adaptation.clamp(0.0, 1.0);
    let tau_fast = 0.1 + (1.0 - adaptation) * 0.8;
    let tau_slow = 5.0 + (1.0 - adaptation) * 30.0;
    let (w_boredom, w_familiarity, rho_self) = if control.enabled {
        (
            control.novelty_bias,
            0.2,
            control.self_focus.clamp(0.0, 1.0),
        )
    } else {
        (0.0, 0.0, 0.0)
    };
    AdaptationConfig {
        tau_fast: Some(tau_fast),
        tau_slow: Some(tau_slow),
        w_boredom: Some(w_boredom),
        w_familiarity: Some(w_familiarity),
        rho_self: Some(rho_self),
        boredom_gamma: Some(0.5),
        self_smoothing_radius: Some(1),
        silence_mass_epsilon: Some(1e-6),
    }
}

fn duration_config_from_spec(duration: &DurationSpec) -> DurationConfig {
    match duration {
        DurationSpec::Field(f) => DurationConfig::Field {
            hold_min_theta: f.hold_min_theta,
            hold_max_theta: f.hold_max_theta,
            curve_k: f.curve_k,
            curve_x0: f.curve_x0,
            drop_gain: f.drop_gain,
        },
        DurationSpec::Gates(n) => DurationConfig::FixedGate { length_gates: *n },
        DurationSpec::WhileAlive => DurationConfig::FixedGate { length_gates: 1 },
    }
}

/// Scale a gate-counted hold by `mult` so a dense fixed grid does not force notes
/// to be extremely short (clicky). Field holds (theta-fraction semantics) are
/// left unchanged.
fn scale_gate_duration(duration: DurationConfig, mult: u32) -> DurationConfig {
    let mult = mult.max(1);
    match duration {
        DurationConfig::FixedGate { length_gates } => DurationConfig::FixedGate {
            length_gates: length_gates.saturating_mul(mult),
        },
        other => other,
    }
}

/// Per-onset emphasis for a rhythm role. An accent drives the shared meter
/// harder so a recurring downbeat can seed an emergent measure; texture and
/// subdivision sit below a plain beat.
fn onset_strength_for_role(role: RhythmRole) -> f32 {
    match role {
        RhythmRole::Beat => 1.0,
        RhythmRole::Subdivision => 0.7,
        RhythmRole::Accent => 2.5,
        RhythmRole::Texture => 0.85,
    }
}

pub(crate) fn phonation_config_from_spec(spec: &PhonationSpec) -> PhonationConfig {
    let duration = duration_config_from_spec(&spec.duration);

    // The rhythm families are one coupling continuum on the shared production
    // meter: each voice is a phase oscillator that entrains its onset phase to
    // the emergent beat with a per-voice coupling strength. There is no
    // externally imposed grid -- coherence (or its absence) emerges from how
    // tightly each voice locks to the meter the population itself drives.
    if let PhonationTiming::Coupled(coupled) = spec.timing {
        let coupled = coupled.sanitized();
        // The low-coupling (flow) renewal droplets are short; scale the
        // gate-counted hold so they are substantial rather than clicky.
        const FLOW_HOLD_MULT: u32 = 3;
        let duration = if coupled.flow_depth > 0.0 {
            scale_gate_duration(duration, FLOW_HOLD_MULT)
        } else {
            duration
        };
        return PhonationConfig {
            mode: PhonationMode::Gated,
            onset: OnsetConfig::Always {
                strength: onset_strength_for_role(coupled.role),
            },
            duration,
            clock: PhonationClockConfig::Coupling {
                coupling: coupled.coupling,
                base_rate_hz: coupled.base_rate_hz,
                flow_depth: coupled.flow_depth,
                microtiming: coupled.microtiming,
            },
        };
    }

    match spec.timing {
        PhonationTiming::Once => match &spec.duration {
            // once() + while_alive() = sustain: Hold mode, NoteOff on death.
            DurationSpec::WhileAlive => PhonationConfig {
                mode: PhonationMode::Hold,
                onset: OnsetConfig::None,
                duration,
                clock: PhonationClockConfig::ThetaGate,
            },
            // once() + cycles(n) / adaptive_duration(): fire immediately, never repeat.
            _ => PhonationConfig {
                mode: PhonationMode::Gated,
                onset: OnsetConfig::Accumulator {
                    rate: 1e6,
                    refractory: u32::MAX,
                },
                duration,
                clock: PhonationClockConfig::ThetaGate,
            },
        },
        // `sync` does not shape onset timing; it only mixes the predictive gate
        // gain (`prediction_sync`) read by the population layer.
        PhonationTiming::Pulse { rate_hz, .. } => PhonationConfig {
            mode: PhonationMode::Gated,
            onset: OnsetConfig::Accumulator {
                rate: rate_hz.max(0.01),
                refractory: 1,
            },
            duration,
            clock: PhonationClockConfig::ThetaGate,
        },
        PhonationTiming::Coupled(_) => unreachable!("handled above"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::{CoupledTimingSpec, PhonationTiming};

    #[test]
    fn low_coupling_flow_uses_renewal_clock() {
        let mut spec = PhonationSpec::default();
        spec.timing = PhonationTiming::Coupled(CoupledTimingSpec {
            coupling: 0.05,
            base_rate_hz: 2.5,
            flow_depth: 0.9,
            role: RhythmRole::Texture,
            ..Default::default()
        });

        let config = phonation_config_from_spec(&spec);
        assert!(
            matches!(
                config.clock,
                PhonationClockConfig::Coupling {
                    coupling,
                    base_rate_hz,
                    flow_depth,
                    ..
                } if coupling < 0.2 && base_rate_hz > 0.0 && flow_depth > 0.0
            ),
            "expected low coupling to use a near-zero-coupling renewal clock, got {:?}",
            config.clock
        );
        assert!(matches!(config.onset, OnsetConfig::Always { .. }));
    }

    #[test]
    fn high_coupling_metric_uses_deep_attractor_clock() {
        let mut spec = PhonationSpec::default();
        spec.timing = PhonationTiming::Coupled(CoupledTimingSpec {
            coupling: 0.95,
            base_rate_hz: 2.0,
            ..Default::default()
        });

        let config = phonation_config_from_spec(&spec);
        assert!(
            matches!(
                config.clock,
                PhonationClockConfig::Coupling {
                    coupling,
                    flow_depth,
                    ..
                } if coupling > 0.8 && flow_depth == 0.0
            ),
            "expected high coupling to use a deep-attractor clock, got {:?}",
            config.clock
        );
        assert!(matches!(config.onset, OnsetConfig::Always { .. }));
    }

    #[test]
    fn accent_role_emits_stronger_onset_than_beat() {
        let beat = phonation_config_from_spec(&PhonationSpec {
            timing: PhonationTiming::Coupled(CoupledTimingSpec {
                role: RhythmRole::Beat,
                ..Default::default()
            }),
            ..PhonationSpec::default()
        });
        let accent = phonation_config_from_spec(&PhonationSpec {
            timing: PhonationTiming::Coupled(CoupledTimingSpec {
                role: RhythmRole::Accent,
                ..Default::default()
            }),
            ..PhonationSpec::default()
        });
        let strength = |c: &PhonationConfig| match c.onset {
            OnsetConfig::Always { strength } => strength,
            _ => panic!("expected Always onset"),
        };
        assert!(
            strength(&accent) > strength(&beat),
            "accent {} should exceed beat {}",
            strength(&accent),
            strength(&beat)
        );
    }
}
