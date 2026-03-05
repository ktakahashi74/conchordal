use crate::life::control::{PerceptualControl, PitchControl, PitchCoreKind};
use crate::life::perceptual::PerceptualConfig;
use crate::life::scenario::{
    DurationSpec, PhonationClockConfig, PhonationConfig, PhonationConnectConfig,
    PhonationIntervalConfig, PhonationMode, PitchCoreConfig, SocialConfig, SubThetaModConfig,
    UtteranceSpec, WhenSpec,
};

pub(crate) fn tessitura_gravity_from_control(gravity: f32) -> f32 {
    gravity.clamp(0.0, 1.0) * 0.2
}

pub(crate) fn pitch_core_config_from_control(pitch: &PitchControl) -> PitchCoreConfig {
    match pitch.core_kind {
        PitchCoreKind::HillClimb => PitchCoreConfig::PitchHillClimb {
            neighbor_step_cents: None,
            tessitura_gravity: Some(tessitura_gravity_from_control(pitch.gravity)),
            move_cost_coeff: Some(pitch.move_cost_coeff),
            move_cost_exp: None,
            improvement_threshold: Some(pitch.improvement_threshold),
            exploration: Some(pitch.exploration),
            persistence: Some(pitch.persistence),
            leave_self_out: Some(pitch.leave_self_out),
            anneal_temp: Some(pitch.anneal_temp),
            global_peak_count: Some(pitch.global_peak_count),
            global_peak_min_sep_cents: Some(pitch.global_peak_min_sep_cents),
            use_ratio_candidates: Some(pitch.use_ratio_candidates),
            ratio_candidate_count: Some(pitch.ratio_candidate_count),
            move_cost_time_scale: Some(pitch.move_cost_time_scale),
            leave_self_out_harmonics: Some(pitch.leave_self_out_harmonics),
        },
        PitchCoreKind::PeakSampler => PitchCoreConfig::PitchPeakSampler {
            neighbor_step_cents: None,
            window_cents: None,
            top_k: None,
            temperature: None,
            sigma_cents: None,
            random_candidates: None,
            tessitura_gravity: Some(tessitura_gravity_from_control(pitch.gravity)),
            exploration: Some(pitch.exploration),
            persistence: Some(pitch.persistence),
            leave_self_out: Some(pitch.leave_self_out),
        },
    }
}

pub(crate) fn perceptual_config_from_control(control: &PerceptualControl) -> PerceptualConfig {
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
    PerceptualConfig {
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

fn connect_from_duration(duration: &DurationSpec) -> PhonationConnectConfig {
    match duration {
        DurationSpec::Field(f) => PhonationConnectConfig::Field {
            hold_min_theta: f.hold_min_theta,
            hold_max_theta: f.hold_max_theta,
            curve_k: f.curve_k,
            curve_x0: f.curve_x0,
            drop_gain: f.drop_gain,
        },
        DurationSpec::Gates(n) => PhonationConnectConfig::FixedGate { length_gates: *n },
        DurationSpec::WhileAlive => PhonationConnectConfig::FixedGate { length_gates: 1 },
    }
}

pub(crate) fn phonation_config_from_utterance(spec: &UtteranceSpec) -> PhonationConfig {
    let connect = connect_from_duration(&spec.duration);

    match &spec.when {
        WhenSpec::Once => match &spec.duration {
            // once() + while_alive() = sustain: Hold mode, NoteOff on death.
            DurationSpec::WhileAlive => PhonationConfig {
                mode: PhonationMode::Hold,
                interval: PhonationIntervalConfig::None,
                connect,
                clock: PhonationClockConfig::ThetaGate,
                sub_theta_mod: SubThetaModConfig::None,
                social: SocialConfig::default(),
            },
            // once() + gates(n) / field(): fire immediately, never repeat.
            _ => PhonationConfig {
                mode: PhonationMode::Gated,
                interval: PhonationIntervalConfig::Accumulator {
                    rate: 1e6,
                    refractory: u32::MAX,
                },
                connect,
                clock: PhonationClockConfig::ThetaGate,
                sub_theta_mod: SubThetaModConfig::None,
                social: SocialConfig::default(),
            },
        },
        WhenSpec::Pulse { rate, sync, social } => {
            let interval = PhonationIntervalConfig::Accumulator {
                rate: rate.max(0.01),
                refractory: 1,
            };
            let sub_theta_mod = if *sync > 0.0 {
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
                interval,
                connect,
                clock: PhonationClockConfig::ThetaGate,
                sub_theta_mod,
                social: SocialConfig {
                    coupling: social.clamp(0.0, 1.0),
                    bin_ticks: 0,
                    smooth: 0.0,
                },
            }
        }
    }
}
