use crate::core::landscape::Landscape;
use crate::life::adaptation::{AdaptationContext, FeaturesNow};
use crate::scenario::control::{PitchControl, PitchCoreKind};
use rand::Rng;
use std::sync::Arc;

#[path = "pitch_core/candidates.rs"]
mod candidates;
#[path = "pitch_core/hill_climb.rs"]
mod hill_climb;
#[path = "pitch_core/peak_sampler.rs"]
mod peak_sampler;
#[path = "pitch_core/scoring.rs"]
mod scoring;

pub use hill_climb::PitchHillClimbPitchCore;
#[cfg(test)]
pub(crate) use hill_climb::test_support::aud_m1_hill_choice;
pub use peak_sampler::PitchPeakSamplerCore;
pub(super) use scoring::RearticulationView;
pub(crate) use scoring::{LooInput, fill_occupancy_scan};

use candidates::{
    DEFAULT_LOCAL_TOP_K, DEFAULT_LOCAL_WINDOW_CENTS, DEFAULT_RANDOM_CANDIDATES,
    DEFAULT_RANDOM_SIGMA_CENTS,
};
use scoring::{loo_score_scan, sample_ownership_score};

#[derive(Debug, Clone, Copy)]
pub struct TargetProposal {
    pub target_pitch_log2: f32,
    pub salience: f32,
}

#[allow(clippy::too_many_arguments)]
pub trait PitchCore {
    fn propose_target<R: Rng + Clone>(
        &mut self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        current_freq_hz: f32,
        integration_window: f32,
        landscape: &Landscape,
        perceptual: &AdaptationContext,
        _features: &FeaturesNow,
        neighbor_pitch_log2: &[f32],
        rng: &mut R,
    ) -> TargetProposal;

    #[allow(clippy::too_many_arguments)]
    fn propose_target_with_crowding_salience<R: Rng + Clone>(
        &mut self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        current_freq_hz: f32,
        integration_window: f32,
        landscape: &Landscape,
        perceptual: &AdaptationContext,
        features: &FeaturesNow,
        neighbor_pitch_log2: &[f32],
        neighbor_salience: &[f32],
        rng: &mut R,
    ) -> TargetProposal {
        let _ = neighbor_salience;
        self.propose_target(
            current_pitch_log2,
            current_target_log2,
            current_freq_hz,
            integration_window,
            landscape,
            perceptual,
            features,
            neighbor_pitch_log2,
            rng,
        )
    }
}

#[derive(Debug, Clone)]
// Fixed snapshots avoid allocating on each sequence-enabled pitch decision.
#[allow(clippy::large_enum_variant)]
pub enum AnyPitchCore {
    PitchHillClimb(PitchHillClimbPitchCore),
    PitchPeakSampler(PitchPeakSamplerCore),
}

impl PitchCore for AnyPitchCore {
    fn propose_target<R: Rng + Clone>(
        &mut self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        current_freq_hz: f32,
        integration_window: f32,
        landscape: &Landscape,
        perceptual: &AdaptationContext,
        features: &FeaturesNow,
        neighbor_pitch_log2: &[f32],
        rng: &mut R,
    ) -> TargetProposal {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.propose_target(
                current_pitch_log2,
                current_target_log2,
                current_freq_hz,
                integration_window,
                landscape,
                perceptual,
                features,
                neighbor_pitch_log2,
                rng,
            ),
            AnyPitchCore::PitchPeakSampler(core) => core.propose_target(
                current_pitch_log2,
                current_target_log2,
                current_freq_hz,
                integration_window,
                landscape,
                perceptual,
                features,
                neighbor_pitch_log2,
                rng,
            ),
        }
    }

    fn propose_target_with_crowding_salience<R: Rng + Clone>(
        &mut self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        current_freq_hz: f32,
        integration_window: f32,
        landscape: &Landscape,
        perceptual: &AdaptationContext,
        features: &FeaturesNow,
        neighbor_pitch_log2: &[f32],
        neighbor_salience: &[f32],
        rng: &mut R,
    ) -> TargetProposal {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.propose_target_with_crowding_salience(
                current_pitch_log2,
                current_target_log2,
                current_freq_hz,
                integration_window,
                landscape,
                perceptual,
                features,
                neighbor_pitch_log2,
                neighbor_salience,
                rng,
            ),
            AnyPitchCore::PitchPeakSampler(core) => core.propose_target_with_crowding_salience(
                current_pitch_log2,
                current_target_log2,
                current_freq_hz,
                integration_window,
                landscape,
                perceptual,
                features,
                neighbor_pitch_log2,
                neighbor_salience,
                rng,
            ),
        }
    }
}

impl AnyPitchCore {
    pub(crate) fn from_control(pitch: &PitchControl, initial_pitch_log2: f32) -> Self {
        let mut core = match pitch.core_kind {
            PitchCoreKind::HillClimb => AnyPitchCore::PitchHillClimb(PitchHillClimbPitchCore::new(
                pitch.neighbor_step_cents.unwrap_or(200.0),
                initial_pitch_log2,
                pitch.resolved_tessitura_gravity(),
            )),
            PitchCoreKind::PeakSampler => {
                AnyPitchCore::PitchPeakSampler(PitchPeakSamplerCore::new(
                    pitch.neighbor_step_cents.unwrap_or(160.0),
                    initial_pitch_log2,
                    pitch.resolved_tessitura_gravity(),
                    pitch.window_cents.unwrap_or(DEFAULT_LOCAL_WINDOW_CENTS),
                    pitch.top_k.unwrap_or(DEFAULT_LOCAL_TOP_K),
                    pitch
                        .temperature
                        .unwrap_or(PitchCoreKind::PeakSampler.default_temperature()),
                    pitch.sigma_cents.unwrap_or(DEFAULT_RANDOM_SIGMA_CENTS),
                    pitch.random_candidates.unwrap_or(DEFAULT_RANDOM_CANDIDATES),
                ))
            }
        };
        core.apply_control(pitch);
        core
    }

    pub(crate) fn dwell_level(
        &self,
        landscape: &Landscape,
        freq_hz: f32,
        leave_self_out: bool,
    ) -> (f32, bool) {
        let scan = self
            .loo_input()
            .exact_score
            .as_deref()
            .filter(|_| leave_self_out);
        let score_scan = scan.unwrap_or(&landscape.acoustic.consonance_field_score_eff);
        landscape
            .acoustic
            .space
            .assert_scan_len_named(score_scan, "dwell_score_scan");
        // Ordinary pitch sampling sanitizes invalid input to zero. Dwell needs
        // neutral level instead, including when the configured theta is nonzero.
        let log2 = freq_hz.max(1.0).log2();
        let sample_hz = log2
            .exp2()
            .clamp(landscape.acoustic.space.fmin, landscape.acoustic.space.fmax);
        let (lo, hi) = landscape.freq_bounds_log2();
        if (lo..=hi).contains(&log2) {
            let pos = (sample_hz.log2() - landscape.acoustic.space.centers_log2[0])
                / landscape.acoustic.space.step_log2;
            let raw = crate::core::log2space::sample_scan_linear_at_pos(score_scan, pos);
            let invalid_habituation = scan.is_some() && {
                landscape.acoustic.space.assert_scan_len_named(
                    &landscape.acoustic.perc_habituation_state_scan,
                    "dwell_habituation_scan",
                );
                !crate::core::log2space::sample_scan_linear_at_pos(
                    &landscape.acoustic.perc_habituation_state_scan,
                    pos,
                )
                .is_finite()
            };
            if !raw.is_finite() || invalid_habituation {
                return (0.5, scan.is_some());
            }
        }
        let score = sample_ownership_score(landscape, log2, scan);
        let score = if let Self::PitchHillClimb(core) = self {
            core.tonal
                .as_ref()
                .and_then(|t| t.query(freq_hz).map(|q| t.score(score, q)))
                .unwrap_or(score)
        } else {
            score
        };
        let level = if score.is_finite() {
            landscape.acoustic.consonance_representation.level(score)
        } else {
            0.5
        };
        (if level.is_finite() { level } else { 0.5 }, scan.is_some())
    }

    pub(crate) fn environment_score(&mut self, landscape: &Landscape, freq_hz: f32) -> f32 {
        let input = self.loo_input_mut();
        let scan = loo_score_scan(landscape, input);
        sample_ownership_score(landscape, freq_hz.max(1.0).log2(), scan.as_deref())
    }

    pub(crate) fn loo_input(&self) -> &LooInput {
        match self {
            Self::PitchHillClimb(core) => &core.loo_input,
            Self::PitchPeakSampler(core) => &core.loo_input,
        }
    }

    pub(crate) fn loo_input_mut(&mut self) -> &mut LooInput {
        match self {
            Self::PitchHillClimb(core) => &mut core.loo_input,
            Self::PitchPeakSampler(core) => &mut core.loo_input,
        }
    }

    pub(crate) fn apply_control(&mut self, pitch: &PitchControl) {
        match self {
            AnyPitchCore::PitchHillClimb(core) => apply_hill_climb_control(core, pitch),
            AnyPitchCore::PitchPeakSampler(core) => apply_peak_sampler_control(core, pitch),
        }
    }

    pub(crate) fn set_temperature(&mut self, value: f32) {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.set_temperature(value),
            AnyPitchCore::PitchPeakSampler(core) => core.set_temperature(value),
        }
    }

    pub(crate) fn set_proposal_interval_sec(&mut self, value: Option<f32>) {
        if let AnyPitchCore::PitchHillClimb(core) = self {
            core.set_proposal_interval_sec(value);
        }
    }

    pub(crate) fn set_body_ratios(&mut self, ratios: Option<Arc<[f32]>>) {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.set_body_ratios(ratios),
            AnyPitchCore::PitchPeakSampler(core) => core.set_body_ratios(ratios),
        }
    }
}

fn apply_hill_climb_control(core: &mut PitchHillClimbPitchCore, pitch: &PitchControl) {
    core.set_tessitura_center(pitch.freq.max(1.0).log2());
    core.set_tessitura_gravity(pitch.resolved_tessitura_gravity());
    core.set_landscape_weight(pitch.landscape_weight);
    core.set_temperature(
        pitch
            .temperature
            .unwrap_or(PitchCoreKind::HillClimb.default_temperature()),
    );
    core.set_crowding(pitch.crowding_strength, pitch.crowding_sigma_cents);
    core.set_leave_self_out(pitch.leave_self_out);
    if let Some(cents) = pitch.neighbor_step_cents {
        core.set_neighbor_step_cents(cents);
    }
    core.set_move_cost_coeff(pitch.move_cost_coeff);
    if let Some(exp) = pitch.move_cost_exp {
        core.set_move_cost_exp(exp);
    }
    core.set_global_peaks(pitch.global_peak_count, pitch.global_peak_min_sep_cents);
    core.set_ratio_candidates(pitch.use_ratio_candidates, pitch.ratio_candidate_count);
    core.set_move_cost_time_scale(pitch.move_cost_time_scale);
    core.set_proposal_interval_sec(pitch.proposal_interval_sec);
}

fn apply_peak_sampler_control(core: &mut PitchPeakSamplerCore, pitch: &PitchControl) {
    core.set_tessitura_center(pitch.freq.max(1.0).log2());
    core.set_tessitura_gravity(pitch.resolved_tessitura_gravity());
    core.set_landscape_weight(pitch.landscape_weight);
    core.set_temperature(
        pitch
            .temperature
            .unwrap_or(PitchCoreKind::PeakSampler.default_temperature()),
    );
    core.set_crowding(pitch.crowding_strength, pitch.crowding_sigma_cents);
    core.set_leave_self_out(pitch.leave_self_out);
    if let Some(cents) = pitch.neighbor_step_cents {
        core.set_neighbor_step_cents(cents);
    }
    if let Some(cents) = pitch.window_cents {
        core.set_window_cents(cents);
    }
    if let Some(top_k) = pitch.top_k {
        core.set_top_k(top_k);
    }
    if let Some(cents) = pitch.sigma_cents {
        core.set_sigma_cents(cents);
    }
    if let Some(count) = pitch.random_candidates {
        core.set_random_candidates(count);
    }
}

#[cfg(test)]
#[path = "pitch_core/tests.rs"]
mod tests;

#[cfg(feature = "loo-diagnostic")]
#[path = "pitch_loo_diagnostic.rs"]
pub(crate) mod diagnostic;

#[cfg(test)]
#[path = "pitch_loo_tests.rs"]
mod loo_tests;

#[cfg(test)]
#[path = "pitch_loo_conformance_tests.rs"]
mod loo_conformance_tests;
