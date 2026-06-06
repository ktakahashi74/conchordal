use crate::life::adaptation::AdaptationConfig;
use crate::life::control::AdaptationControl;
use crate::scenario::{
    DurationConfig, DurationSpec, OnsetConfig, PhonationClockConfig, PhonationConfig,
    PhonationMode, PhonationSpec, PhonationTiming, SubThetaModConfig,
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

pub(crate) fn phonation_config_from_spec(spec: &PhonationSpec) -> PhonationConfig {
    let duration = duration_config_from_spec(&spec.duration);

    match spec.timing {
        PhonationTiming::MetricBeat(metric) => {
            let metric = metric.sanitized();
            // Metric beat rides a fixed wall-clock grid (absolute tick 0 anchor),
            // not the adaptive theta, so the pulse is isochronous and every
            // metric voice at the same rate shares the beat phase. The grid is
            // subdivided MULTx for fine duration resolution, and the onset rate
            // matches the grid with a MULT-1 refractory so the accumulator fires
            // on its first gate (regardless of its random seed phase) and then
            // once per beat: all voices land on the beat together instead of at
            // independent random offsets.
            const METRIC_GRID_MULT: u32 = 4;
            let grid_rate = metric.rate_hz * METRIC_GRID_MULT as f32;
            // `accent`/`beat_strength` has no engine effect on a one-onset-per-
            // beat grid: a within-gate cosine only modulates sub-beat candidates,
            // and metric emits onsets only on beat boundaries. A meter-level
            // accent (emphasis every Nth beat) is the right home for it and is
            // not yet implemented, so leave the sub-theta modulation off.
            return PhonationConfig {
                mode: PhonationMode::Gated,
                onset: OnsetConfig::Accumulator {
                    rate: grid_rate,
                    refractory: METRIC_GRID_MULT - 1,
                },
                duration,
                clock: PhonationClockConfig::FixedRate {
                    rate_hz: grid_rate,
                    // Shared absolute grid: all metric voices land on the same beat.
                    shared_grid: true,
                },
                sub_theta_mod: SubThetaModConfig::None,
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
            // Flow fires at real-time renewal IOIs on a fixed fine grid instead
            // of theta subdivisions, so it does not inherit the adaptive theta
            // wobble and reads as non-metric. The grid is fine enough (~21 ms) to
            // resolve clustered IOIs that the old theta subdivisions reached.
            const FLOW_CLOCK_HZ: f32 = 48.0;
            // The dense grid would otherwise make holds extremely short and
            // clicky; scale them to a discrete-but-substantial droplet length
            // (between the too-long legato and the too-short click extremes).
            const FLOW_HOLD_MULT: u32 = 3;
            return PhonationConfig {
                mode: PhonationMode::Gated,
                onset: OnsetConfig::Flow {
                    mean_rate: flow.mean_rate_hz,
                    depth: flow.depth,
                },
                duration: scale_gate_duration(duration, FLOW_HOLD_MULT),
                clock: PhonationClockConfig::FixedRate {
                    rate_hz: FLOW_CLOCK_HZ,
                    // Per-voice grid: droplets keep independent phase (no shared
                    // lattice that would make them lock to one period).
                    shared_grid: false,
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
    fn flow_timing_uses_fixed_rate_clock() {
        let mut spec = PhonationSpec::default();
        spec.timing = PhonationTiming::FlowTiming(FlowTimingSpec {
            mean_rate_hz: 2.5,
            depth: 0.9,
        });

        let config = phonation_config_from_spec(&spec);
        assert!(
            matches!(
                config.clock,
                PhonationClockConfig::FixedRate {
                    rate_hz,
                    shared_grid: false
                } if rate_hz > 0.0
            ),
            "expected flow timing to use a per-voice fixed-rate clock, got {:?}",
            config.clock
        );
        assert!(matches!(config.onset, OnsetConfig::Flow { .. }));
    }
}
