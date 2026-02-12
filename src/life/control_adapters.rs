use crate::life::control::{
    BodyControl, BodyMethod, PerceptualControl, PhonationControl, PhonationType, PitchControl,
    PitchCoreKind,
};
use crate::life::perceptual::PerceptualConfig;
use crate::life::scenario::{
    HarmonicMode, PhonationClockConfig, PhonationConfig, PhonationConnectConfig,
    PhonationIntervalConfig, PhonationMode, PitchCoreConfig, SocialConfig, SoundBodyConfig,
    SubThetaModConfig, TimbreGenotype,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct PerceptualParams {
    pub(crate) enabled: bool,
    pub(crate) tau_fast: f32,
    pub(crate) tau_slow: f32,
    pub(crate) w_boredom: f32,
    pub(crate) w_familiarity: f32,
    pub(crate) rho_self: f32,
    pub(crate) boredom_gamma: f32,
    pub(crate) self_smoothing_radius: usize,
    pub(crate) silence_mass_epsilon: f32,
}

pub(crate) fn perceptual_params_from_control(control: &PerceptualControl) -> PerceptualParams {
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
    PerceptualParams {
        enabled: control.enabled,
        tau_fast,
        tau_slow,
        w_boredom,
        w_familiarity,
        rho_self,
        boredom_gamma: 0.5,
        self_smoothing_radius: 1,
        silence_mass_epsilon: 1e-6,
    }
}

pub(crate) fn tessitura_gravity_from_control(gravity: f32) -> f32 {
    gravity.clamp(0.0, 1.0) * 0.2
}

pub(crate) fn pitch_core_config_from_control(pitch: &PitchControl) -> PitchCoreConfig {
    match pitch.core_kind {
        PitchCoreKind::HillClimb => PitchCoreConfig::PitchHillClimb {
            neighbor_step_cents: None,
            tessitura_gravity: Some(tessitura_gravity_from_control(pitch.gravity)),
            move_cost_coeff: None,
            move_cost_exp: None,
            improvement_threshold: None,
            exploration: Some(pitch.exploration),
            persistence: Some(pitch.persistence),
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
        },
    }
}

pub(crate) fn perceptual_config_from_control(control: &PerceptualControl) -> PerceptualConfig {
    let params = perceptual_params_from_control(control);
    PerceptualConfig {
        tau_fast: Some(params.tau_fast),
        tau_slow: Some(params.tau_slow),
        w_boredom: Some(params.w_boredom),
        w_familiarity: Some(params.w_familiarity),
        rho_self: Some(params.rho_self),
        boredom_gamma: Some(params.boredom_gamma),
        self_smoothing_radius: Some(params.self_smoothing_radius),
        silence_mass_epsilon: Some(params.silence_mass_epsilon),
    }
}

pub(crate) fn sound_body_config_from_control(body: &BodyControl) -> SoundBodyConfig {
    match body.method {
        BodyMethod::Sine => SoundBodyConfig::Sine { phase: None },
        BodyMethod::Harmonic => {
            let timbre = &body.timbre;
            let genotype = TimbreGenotype {
                mode: HarmonicMode::Harmonic,
                stiffness: timbre.inharmonic,
                brightness: timbre.brightness,
                comb: 0.0,
                damping: 0.5,
                vibrato_rate: 5.0,
                vibrato_depth: timbre.motion * 0.02,
                jitter: timbre.motion,
                unison: timbre.width,
            };
            SoundBodyConfig::Harmonic {
                genotype,
                partials: None,
            }
        }
    }
}

pub(crate) fn phonation_config_from_control(control: &PhonationControl) -> PhonationConfig {
    // Hold ignores density/sync/legato; it is purely lifecycle-driven.
    if matches!(control.r#type, PhonationType::Hold) {
        let social = SocialConfig {
            coupling: control.sociality.clamp(0.0, 1.0),
            bin_ticks: 0,
            smooth: 0.0,
        };
        return PhonationConfig {
            mode: PhonationMode::Hold,
            interval: PhonationIntervalConfig::None,
            connect: PhonationConnectConfig::FixedGate { length_gates: 1 },
            clock: PhonationClockConfig::ThetaGate,
            sub_theta_mod: SubThetaModConfig::None,
            social,
        };
    }
    let density = control.density.clamp(0.0, 1.0);
    let rate = 0.5 + density * 3.5;
    let interval = match control.r#type {
        PhonationType::None => PhonationIntervalConfig::None,
        _ => PhonationIntervalConfig::Accumulator {
            rate,
            refractory: 1,
        },
    };
    let legato = control.legato.clamp(0.0, 1.0);
    let length_gates = (1.0 + legato * 8.0).round().max(1.0) as u32;
    let connect = match control.r#type {
        PhonationType::Field => PhonationConnectConfig::Field {
            hold_min_theta: 0.1 + legato * 0.2,
            hold_max_theta: 0.6 + legato * 0.4,
            curve_k: 2.0,
            curve_x0: 0.5,
            drop_gain: (1.0 - legato).clamp(0.0, 1.0),
        },
        _ => PhonationConnectConfig::FixedGate { length_gates },
    };
    let sub_theta_mod = if control.sync > 0.0 {
        SubThetaModConfig::Cosine {
            n: 1,
            depth: control.sync.clamp(0.0, 1.0),
            phase0: 0.0,
        }
    } else {
        SubThetaModConfig::None
    };
    let social = SocialConfig {
        coupling: control.sociality.clamp(0.0, 1.0),
        bin_ticks: 0,
        smooth: 0.0,
    };
    PhonationConfig {
        mode: PhonationMode::Gated,
        interval,
        connect,
        clock: PhonationClockConfig::ThetaGate,
        sub_theta_mod,
        social,
    }
}
