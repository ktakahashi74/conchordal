use crate::life::adaptation::AdaptationConfig;
use crate::life::control::AdaptationControl;
use crate::scenario::{
    DurationConfig, DurationSpec, OnsetConfig, PhonationClockConfig, PhonationConfig,
    PhonationMode, PhonationSpec, PhonationTiming, SubThetaModConfig, SubdivisionClockConfig,
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

pub(crate) fn phonation_config_from_spec(spec: &PhonationSpec) -> PhonationConfig {
    let duration = duration_config_from_spec(&spec.duration);

    match spec.timing {
        PhonationTiming::MetricBeat(metric) => {
            let metric = metric.sanitized();
            return PhonationConfig {
                mode: PhonationMode::Gated,
                onset: OnsetConfig::Accumulator {
                    rate: metric.rate_hz,
                    refractory: 1,
                },
                duration,
                clock: PhonationClockConfig::ThetaGate,
                sub_theta_mod: if metric.accent > 0.0 {
                    SubThetaModConfig::Cosine {
                        n: 1,
                        depth: metric.accent,
                        phase0: 0.0,
                    }
                } else {
                    SubThetaModConfig::None
                },
            };
        }
        PhonationTiming::EntrainedBeat(entrained) => {
            let entrained = entrained.sanitized();
            return PhonationConfig {
                mode: PhonationMode::Gated,
                onset: OnsetConfig::Accumulator {
                    rate: entrained.rate_hz,
                    refractory: 1,
                },
                duration,
                clock: PhonationClockConfig::ThetaGate,
                sub_theta_mod: SubThetaModConfig::None,
            };
        }
        PhonationTiming::FlowTiming(flow) => {
            let flow = flow.sanitized();
            return PhonationConfig {
                mode: PhonationMode::Gated,
                onset: OnsetConfig::Flow {
                    mean_rate: flow.mean_rate_hz,
                    depth: flow.depth,
                },
                duration,
                clock: PhonationClockConfig::Composite {
                    subdivision: Some(SubdivisionClockConfig {
                        divisions: vec![5, 7, 11],
                    }),
                    internal_phase: None,
                },
                sub_theta_mod: SubThetaModConfig::None,
            };
        }
        PhonationTiming::Once | PhonationTiming::Pulse { .. } => {}
    }

    match spec.timing {
        PhonationTiming::Once => match &spec.duration {
            // once() + while_alive() = sustain: Hold mode, NoteOff on death.
            DurationSpec::WhileAlive => PhonationConfig {
                mode: PhonationMode::Hold,
                onset: OnsetConfig::None,
                duration,
                clock: PhonationClockConfig::ThetaGate,
                sub_theta_mod: SubThetaModConfig::None,
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
                sub_theta_mod: SubThetaModConfig::None,
            },
        },
        PhonationTiming::Pulse {
            rate_hz,
            sync,
            social: _,
        } => {
            let onset = OnsetConfig::Accumulator {
                rate: rate_hz.max(0.01),
                refractory: 1,
            };
            let sub_theta_mod = if sync > 0.0 {
                SubThetaModConfig::Cosine {
                    n: 1,
                    depth: sync.clamp(0.0, 1.0),
                    phase0: 0.0,
                }
            } else {
                SubThetaModConfig::None
            };
            PhonationConfig {
                mode: PhonationMode::Gated,
                onset,
                duration,
                clock: PhonationClockConfig::ThetaGate,
                sub_theta_mod,
            }
        }
        PhonationTiming::MetricBeat(_)
        | PhonationTiming::EntrainedBeat(_)
        | PhonationTiming::FlowTiming(_) => unreachable!("handled above"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::{FlowTimingSpec, PhonationTiming};

    #[test]
    fn flow_timing_uses_dense_non_binary_clock() {
        let mut spec = PhonationSpec::default();
        spec.timing = PhonationTiming::FlowTiming(FlowTimingSpec {
            mean_rate_hz: 2.5,
            depth: 0.9,
        });

        let config = phonation_config_from_spec(&spec);
        let PhonationClockConfig::Composite {
            subdivision: Some(subdivision),
            internal_phase: None,
        } = config.clock
        else {
            panic!("expected flow timing to use a composite clock");
        };
        assert_eq!(subdivision.divisions, vec![5, 7, 11]);
    }
}
