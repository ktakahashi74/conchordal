use crate::life::control::{
    PerceptualControl, PhonationControl, PhonationType, PitchControl, PitchCoreKind,
};
use crate::life::perceptual::PerceptualConfig;
use crate::life::scenario::{
    PhonationClockConfig, PhonationConfig, PhonationConnectConfig, PhonationIntervalConfig,
    PhonationMode, PitchCoreConfig, SocialConfig, SubThetaModConfig,
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
