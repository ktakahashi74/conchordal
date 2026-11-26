use super::candidates::{
    DEFAULT_LOCAL_WINDOW_CENTS, cents_to_log2, push_gaussian_candidates, sample_softmax_candidate,
    sample_softmax_with_relations, score_candidates_with, seed_step_candidates, top_local_peaks,
    window_bins_from_cents,
};
use super::scoring::{
    DEFAULT_LOO_HARMONICS, DEFAULT_MOVE_COST_COEFF, DEFAULT_MOVE_COST_EXP, LooInput,
    ReferenceLooMode, adjusted_pitch_score_impl, loo_score_scan,
};
use super::{PitchCore, TargetProposal};
use crate::core::landscape::Landscape;
use crate::life::adaptation::{AdaptationContext, FeaturesNow};
use rand::Rng;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct PitchPeakSamplerCore {
    neighbor_step_log2: f32,
    window_cents: f32,
    top_k: usize,
    pub(super) temperature: f32,
    sigma_cents: f32,
    random_candidates: usize,
    tessitura_center: f32,
    tessitura_gravity: f32,
    landscape_weight: f32,
    pub(super) crowding_strength: f32,
    pub(super) crowding_sigma_cents: f32,
    pub(super) leave_self_out: bool,
    /// Ratios retained for historical test controls and diagnostic replay.
    body_ratios: Option<Arc<[f32]>>,
    pub(in crate::life::voice) loo_input: LooInput,
}

impl PitchPeakSamplerCore {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        neighbor_step_cents: f32,
        tessitura_center: f32,
        tessitura_gravity: f32,
        window_cents: f32,
        top_k: usize,
        temperature: f32,
        sigma_cents: f32,
        random_candidates: usize,
    ) -> Self {
        Self {
            neighbor_step_log2: cents_to_log2(neighbor_step_cents.max(0.0)),
            window_cents: window_cents.max(1.0),
            top_k: top_k.max(1),
            temperature: if temperature.is_finite() {
                temperature.max(0.0)
            } else {
                0.0
            },
            sigma_cents: sigma_cents.max(0.0),
            random_candidates,
            tessitura_center,
            tessitura_gravity,
            landscape_weight: 1.0,
            crowding_strength: 0.0,
            crowding_sigma_cents: 60.0,
            leave_self_out: false,
            body_ratios: None,
            loo_input: LooInput::default(),
        }
    }

    pub fn set_body_ratios(&mut self, ratios: Option<Arc<[f32]>>) {
        self.body_ratios = ratios;
    }

    pub fn set_crowding(&mut self, strength: f32, sigma_cents: f32) {
        self.crowding_strength = if strength.is_finite() {
            strength.max(0.0)
        } else {
            0.0
        };
        self.crowding_sigma_cents = if sigma_cents.is_finite() {
            sigma_cents.max(1e-3)
        } else {
            60.0
        };
    }

    pub fn set_leave_self_out(&mut self, enabled: bool) {
        self.leave_self_out = enabled;
    }

    pub fn set_window_cents(&mut self, value: f32) {
        self.window_cents = if value.is_finite() {
            value.max(1.0)
        } else {
            DEFAULT_LOCAL_WINDOW_CENTS
        };
    }

    pub fn set_top_k(&mut self, value: usize) {
        self.top_k = value.max(1);
    }

    pub fn set_temperature(&mut self, value: f32) {
        self.temperature = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
    }

    pub fn set_sigma_cents(&mut self, value: f32) {
        self.sigma_cents = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
    }

    pub fn set_random_candidates(&mut self, value: usize) {
        self.random_candidates = value;
    }

    pub fn set_neighbor_step_cents(&mut self, value: f32) {
        let cents = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
        self.neighbor_step_log2 = cents_to_log2(cents);
    }

    pub fn set_tessitura_center(&mut self, value: f32) {
        self.tessitura_center = value;
    }

    pub fn set_tessitura_gravity(&mut self, value: f32) {
        self.tessitura_gravity = value;
    }

    pub fn set_landscape_weight(&mut self, value: f32) {
        self.landscape_weight = if value.is_finite() {
            value.max(0.0)
        } else {
            1.0
        };
    }
}

impl PitchCore for PitchPeakSamplerCore {
    fn propose_target<R: Rng + Clone>(
        &mut self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        _current_freq_hz: f32,
        integration_window: f32,
        landscape: &Landscape,
        perceptual: &AdaptationContext,
        _features: &FeaturesNow,
        neighbor_pitch_log2: &[f32],
        rng: &mut R,
    ) -> TargetProposal {
        self.propose_target_with_crowding_salience(
            current_pitch_log2,
            current_target_log2,
            _current_freq_hz,
            integration_window,
            landscape,
            perceptual,
            _features,
            neighbor_pitch_log2,
            &[],
            rng,
        )
    }

    fn propose_target_with_crowding_salience<R: Rng + Clone>(
        &mut self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        _current_freq_hz: f32,
        integration_window: f32,
        landscape: &Landscape,
        perceptual: &AdaptationContext,
        _features: &FeaturesNow,
        neighbor_pitch_log2: &[f32],
        neighbor_salience: &[f32],
        rng: &mut R,
    ) -> TargetProposal {
        let (fmin, fmax) = landscape.acoustic.space.voice_pitch_bounds_log2();
        let current_target_log2 = current_target_log2.clamp(fmin, fmax);
        let window_bins = window_bins_from_cents(landscape, self.window_cents);
        let tessitura_center = self.tessitura_center;
        let tessitura_gravity = self.tessitura_gravity;
        let landscape_weight = self.landscape_weight;
        let leave_self_out = self.leave_self_out;
        let crowding_strength = self.crowding_strength;
        let crowding_sigma_cents = self.crowding_sigma_cents;
        let body_ratios = self.body_ratios.clone();
        let loo_scan = if leave_self_out {
            loo_score_scan(landscape, &mut self.loo_input)
        } else {
            None
        };
        let support = self
            .loo_input
            .support
            .clone()
            .filter(|_| leave_self_out && landscape.harmonic.support_weights.active());
        let single_density = self.loo_input.support_single_density;
        let support_template = self.loo_input.support_template.clone();
        let mut scorer = |pitch_log2: f32| -> f32 {
            let base = adjusted_pitch_score_impl(
                pitch_log2,
                current_pitch_log2,
                integration_window,
                tessitura_center,
                tessitura_gravity,
                landscape_weight,
                DEFAULT_MOVE_COST_COEFF,
                DEFAULT_MOVE_COST_EXP,
                landscape,
                perceptual,
                leave_self_out,
                ReferenceLooMode::ApproxHarmonics,
                DEFAULT_LOO_HARMONICS,
                self.loo_input.candidate_scan(
                    &landscape.acoustic.space,
                    pitch_log2,
                    loo_scan.as_deref(),
                ),
                body_ratios.as_deref(),
                crowding_strength,
                crowding_sigma_cents,
                neighbor_pitch_log2,
                neighbor_salience,
            );
            support.as_ref().map_or(base, |s| {
                (f64::from(base)
                    + support_template.as_ref().map_or_else(
                        || {
                            s.single_partial_value(
                                pitch_log2,
                                single_density,
                                landscape.harmonic.support_weights,
                            )
                        },
                        |template| {
                            s.template_value(
                                pitch_log2,
                                template,
                                landscape.harmonic.support_weights,
                            )
                        },
                    ))
                .clamp(-f64::from(f32::MAX), f64::from(f32::MAX)) as f32
            })
        };

        let mut candidates =
            seed_step_candidates(current_target_log2, self.neighbor_step_log2).to_vec();
        top_local_peaks(
            landscape,
            current_target_log2,
            window_bins,
            self.top_k,
            &mut scorer,
            &mut Vec::new(),
            &mut candidates,
        );
        push_gaussian_candidates(
            &mut candidates,
            current_target_log2,
            fmin,
            fmax,
            self.sigma_cents,
            self.random_candidates,
            rng,
        );

        let scored = score_candidates_with(candidates, fmin, fmax, &mut scorer);
        if scored.is_empty() {
            return TargetProposal {
                target_pitch_log2: current_target_log2,
                salience: 0.0,
            };
        }

        let current_adjusted = scorer(current_target_log2);
        let mut best_score = current_adjusted;
        for &(_, score) in &scored {
            if score > best_score {
                best_score = score;
            }
        }
        let improvement = best_score - current_adjusted;

        let temporal = self.loo_input.relations.as_ref().filter(|u| u.active());
        let target_pitch_log2 = if self.temperature > 0.0 {
            if let Some(temporal) = temporal {
                sample_softmax_with_relations(
                    &scored,
                    self.temperature,
                    rng,
                    Some(temporal),
                    Some(current_target_log2),
                )
                .clamp(fmin, fmax)
            } else {
                sample_softmax_candidate(&scored, self.temperature, rng).clamp(fmin, fmax)
            }
        } else {
            // Greedy: settle on the highest-scoring candidate (current target is
            // always among `scored`, so this never moves downhill).
            let baseline = scored
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.1.total_cmp(&b.1))
                .map(|(i, _)| i)
                .unwrap();
            let selected = temporal.map_or(baseline, |u| {
                // max_by chooses the last tie, so retain that candidate order.
                scored.len()
                    - 1
                    - u.choose_scored(scored.len(), scored.len() - 1 - baseline, |i| {
                        let (p, s) = scored[scored.len() - 1 - i];
                        (Some(p), u.source(), f64::from(s), p == current_target_log2)
                    })
            });
            scored[selected].0.clamp(fmin, fmax)
        };

        TargetProposal {
            target_pitch_log2,
            salience: (improvement / 0.2).clamp(0.0, 1.0),
        }
    }
}
