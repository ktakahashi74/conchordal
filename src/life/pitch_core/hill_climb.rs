use super::candidates::{
    DEFAULT_LOCAL_TOP_K, DEFAULT_LOCAL_WINDOW_CENTS, DEFAULT_RANDOM_CANDIDATES,
    DEFAULT_RANDOM_SIGMA_CENTS, cents_to_log2, dedupe_log2_by_cents, push_gaussian_candidates,
    push_runtime_ratio_candidates, seed_step_candidates, top_global_peaks, top_local_peaks,
    window_bins_from_cents,
};
#[cfg(feature = "loo-diagnostic")]
use super::diagnostic;
#[cfg(test)]
use super::scoring::sample_ownership_score;
use super::scoring::{
    DEFAULT_LOO_HARMONICS, DEFAULT_MOVE_COST_COEFF, DEFAULT_MOVE_COST_EXP, LooInput,
    ReferenceLooMode, adjusted_pitch_score_impl, loo_score_scan,
};
#[cfg(any(test, feature = "loo-diagnostic"))]
use super::scoring::{LooScratch, exact_loo_consonance_score_scan};
use super::{PitchCore, TargetProposal};
use crate::core::landscape::Landscape;
use crate::life::adaptation::{AdaptationContext, FeaturesNow};
use crate::scenario::control::MoveCostTimeScale;
use rand::Rng;
use std::sync::Arc;

#[path = "hill_climb/selection.rs"]
mod selection;

const DEFAULT_GLOBAL_PEAK_COUNT: usize = 0;
const DEFAULT_GLOBAL_PEAK_MIN_SEP_CENTS: f32 = 0.0;
const DEFAULT_RUNTIME_RATIO_CANDIDATE_COUNT: usize = 0;

#[derive(Debug, Clone)]
pub struct PitchHillClimbPitchCore {
    neighbor_step_log2: f32,
    tessitura_center: f32,
    tessitura_gravity: f32,
    landscape_weight: f32,
    pub(super) move_cost_coeff: f32,
    move_cost_exp: u8,
    pub(super) move_cost_time_scale: MoveCostTimeScale,
    pub(super) proposal_interval_sec: Option<f32>,
    pub(super) temperature: f32,
    pub(super) crowding_strength: f32,
    pub(super) crowding_sigma_cents: f32,
    pub(super) leave_self_out: bool,
    #[cfg(any(test, feature = "loo-diagnostic"))]
    reference_loo_mode: ReferenceLooMode,
    pub(super) global_peak_count: usize,
    global_peak_min_sep_log2: f32,
    pub(super) use_ratio_candidates: bool,
    pub(super) ratio_candidate_count: usize,
    /// Ratios retained for historical test controls and diagnostic replay.
    body_ratios: Option<Arc<[f32]>>,
    #[cfg(any(test, feature = "loo-diagnostic"))]
    loo_scratch: Option<Box<LooScratch>>,
    pub(in crate::life::voice) loo_input: LooInput,
    #[cfg(feature = "loo-diagnostic")]
    pub(super) diagnostic_variant: Option<&'static str>,
    pub(crate) melodic: Option<crate::life::melodic_continuity::Snapshot>,
    pub(crate) melodic_trace: bool,
    pub(crate) melodic_decision:
        std::cell::RefCell<Option<crate::life::melodic_continuity::Decision>>,
    pub(crate) sequence: Option<crate::life::own_sequence::SequenceSnapshot>,
    pub(crate) sequence_enabled: bool,
    pub(crate) tonal: Option<crate::core::tonal_context::Use>,
    pub(crate) pitch_set: Option<crate::core::tonal_context::pitch_set::Use>,
    pub(crate) sequence_trace: bool,
    pub(crate) sequence_decision:
        std::cell::RefCell<Option<crate::life::own_sequence::SequenceDecision>>,
    pub(super) scratch: std::cell::RefCell<ProposalScratch>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct ProposalScratch {
    candidates: Vec<f32>,
    peaks: Vec<(f32, f32)>,
    pub(super) scored: Vec<(f32, f32)>,
    ratios: Vec<f32>,
    counterfactual: Vec<(f32, f32)>,
}

impl PitchHillClimbPitchCore {
    pub fn new(neighbor_step_cents: f32, tessitura_center: f32, tessitura_gravity: f32) -> Self {
        let mut neighbor_step_cents = neighbor_step_cents;
        if !neighbor_step_cents.is_finite() {
            neighbor_step_cents = 0.0;
        }
        neighbor_step_cents = neighbor_step_cents.max(0.0);
        Self {
            neighbor_step_log2: cents_to_log2(neighbor_step_cents),
            tessitura_center,
            tessitura_gravity,
            landscape_weight: 1.0,
            move_cost_coeff: DEFAULT_MOVE_COST_COEFF,
            move_cost_exp: DEFAULT_MOVE_COST_EXP,
            move_cost_time_scale: MoveCostTimeScale::LegacyIntegrationWindow,
            proposal_interval_sec: None,
            temperature: 0.0,
            crowding_strength: 0.0,
            crowding_sigma_cents: 60.0,
            leave_self_out: false,
            #[cfg(any(test, feature = "loo-diagnostic"))]
            reference_loo_mode: ReferenceLooMode::ApproxHarmonics,
            global_peak_count: DEFAULT_GLOBAL_PEAK_COUNT,
            global_peak_min_sep_log2: cents_to_log2(DEFAULT_GLOBAL_PEAK_MIN_SEP_CENTS),
            use_ratio_candidates: false,
            ratio_candidate_count: DEFAULT_RUNTIME_RATIO_CANDIDATE_COUNT,
            body_ratios: None,
            #[cfg(any(test, feature = "loo-diagnostic"))]
            loo_scratch: None,
            loo_input: LooInput::default(),
            #[cfg(feature = "loo-diagnostic")]
            diagnostic_variant: None,
            melodic: None,
            melodic_trace: false,
            melodic_decision: Default::default(),
            sequence: None,
            sequence_enabled: false,
            tonal: None,
            pitch_set: None,
            sequence_trace: false,
            sequence_decision: Default::default(),
            scratch: Default::default(),
        }
    }

    pub fn set_body_ratios(&mut self, ratios: Option<Arc<[f32]>>) {
        self.body_ratios = ratios;
    }

    pub fn set_temperature(&mut self, value: f32) {
        self.temperature = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
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

    #[cfg(test)]
    pub(super) fn set_reference_loo_mode(&mut self, mode: ReferenceLooMode) {
        self.reference_loo_mode = mode;
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

    pub fn set_move_cost_coeff(&mut self, value: f32) {
        let coeff = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
        self.move_cost_coeff = coeff;
    }

    pub fn set_move_cost_exp(&mut self, value: u8) {
        self.move_cost_exp = if value == 2 { 2 } else { 1 };
    }

    pub fn set_move_cost_time_scale(&mut self, value: MoveCostTimeScale) {
        self.move_cost_time_scale = value;
    }

    pub fn set_proposal_interval_sec(&mut self, value: Option<f32>) {
        self.proposal_interval_sec = value.filter(|v| v.is_finite() && *v > 0.0);
    }

    pub fn set_global_peaks(&mut self, count: usize, min_sep_cents: f32) {
        self.global_peak_count = count;
        let min_sep = if min_sep_cents.is_finite() {
            min_sep_cents.max(0.0)
        } else {
            DEFAULT_GLOBAL_PEAK_MIN_SEP_CENTS
        };
        self.global_peak_min_sep_log2 = cents_to_log2(min_sep);
    }

    pub fn set_ratio_candidates(&mut self, enabled: bool, count: usize) {
        self.use_ratio_candidates = enabled;
        self.ratio_candidate_count = count;
    }

    fn move_cost_time_sec(&self, integration_window: f32) -> f32 {
        match self.move_cost_time_scale {
            MoveCostTimeScale::LegacyIntegrationWindow => integration_window.max(0.0),
            MoveCostTimeScale::ProposalInterval => self
                .proposal_interval_sec
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(integration_window)
                .max(0.0),
        }
    }

    /// Propose a target pitch using a custom scorer.
    ///
    /// The scorer maps `pitch_log2 → score` and is used for both peak extraction
    /// and candidate evaluation. Callers control the objective (e.g. sign-flip
    /// for dissonance maximisation) by supplying their own closure.
    ///
    /// The scorer must be deterministic: repeated calls with the same `pitch_log2`
    /// must return the same value. It is invoked during local/global peak scanning,
    /// candidate rescoring, and the current-position baseline, so a stateful scorer
    /// that returns different values per call will produce inconsistent proposals.
    pub fn propose_with_scorer<R: Rng + Clone>(
        &self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        landscape: &Landscape,
        neighbor_pitch_log2: &[f32],
        rng: &mut R,
        mut scorer: impl FnMut(f32) -> f32,
    ) -> TargetProposal {
        self.propose_with_melodic_scorer(
            current_pitch_log2,
            current_target_log2,
            landscape,
            neighbor_pitch_log2,
            rng,
            None,
            &mut scorer,
        )
        .expect("legacy scorer is infallible")
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn propose_with_melodic_scorer<R: Rng + Clone>(
        &self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        landscape: &Landscape,
        neighbor_pitch_log2: &[f32],
        rng: &mut R,
        melodic: Option<crate::life::melodic_continuity::Snapshot>,
        mut scorer: impl FnMut(f32) -> f32,
    ) -> Result<TargetProposal, &'static str> {
        let (fmin, fmax) = landscape.space.voice_pitch_bounds_log2();
        let current_target_log2 = current_target_log2.clamp(fmin, fmax);
        let melodic_bounds = melodic.map(|m| (m.bounds.0.max(fmin), m.bounds.1.min(fmax)));
        if melodic_bounds.is_some_and(|(lo, hi)| lo > hi) {
            return Err("empty_legal_range");
        }

        #[cfg(feature = "loo-diagnostic")]
        let _ = scorer(current_pitch_log2);
        let adaptive_window_cents =
            (self.neighbor_step_log2.abs() * 1200.0 * 4.0).max(DEFAULT_LOCAL_WINDOW_CENTS);
        let window_bins = window_bins_from_cents(landscape, adaptive_window_cents);
        let mut scratch = self.scratch.borrow_mut();
        let ProposalScratch {
            candidates,
            peaks,
            scored,
            ratios,
            counterfactual,
        } = &mut *scratch;
        candidates.clear();
        candidates.extend(seed_step_candidates(
            current_target_log2,
            self.neighbor_step_log2,
        ));
        top_local_peaks(
            landscape,
            current_target_log2,
            window_bins,
            DEFAULT_LOCAL_TOP_K,
            &mut scorer,
            peaks,
            candidates,
        );
        if self.global_peak_count > 0 {
            top_global_peaks(
                landscape,
                self.global_peak_count,
                self.global_peak_min_sep_log2,
                &mut scorer,
                peaks,
                candidates,
            );
        }
        if self.use_ratio_candidates && self.ratio_candidate_count > 0 {
            push_runtime_ratio_candidates(
                candidates,
                ratios,
                current_target_log2,
                current_pitch_log2,
                neighbor_pitch_log2,
                self.ratio_candidate_count,
                fmin,
                fmax,
            );
        }
        push_gaussian_candidates(
            candidates,
            current_target_log2,
            fmin,
            fmax,
            DEFAULT_RANDOM_SIGMA_CENTS,
            DEFAULT_RANDOM_CANDIDATES,
            rng,
        );
        candidates.retain(|x| x.is_finite());
        dedupe_log2_by_cents(candidates, 1.0);
        if let Some(ps) = &self.pitch_set {
            ps.added.borrow_mut().clear();
            ps.tc_failed_added.set(false);
            for candidate in candidates.iter_mut() {
                *candidate = candidate.clamp(ps.bounds.0, ps.bounds.1);
            }
            candidates.dedup_by(|a, b| a.exp2().to_bits() == b.exp2().to_bits());
            if let Some(memory) = &ps.snapshot {
                for candidate in memory
                    .candidates(current_target_log2, ps.bounds)
                    .into_iter()
                    .flatten()
                {
                    if !candidates
                        .iter()
                        .any(|p| p.exp2().to_bits() == candidate.exp2().to_bits())
                    {
                        candidates.push(candidate);
                        if ps.trace {
                            ps.added.borrow_mut().push(candidate);
                        }
                    }
                }
            }
        }
        #[cfg(all(test, feature = "native", feature = "loo-diagnostic"))]
        diagnostic::restore_room_candidates(candidates);
        scored.clear();
        for &candidate in candidates.iter() {
            let mut pitch = candidate.clamp(fmin, fmax);
            if let Some(snapshot) = self
                .sequence
                .as_ref()
                .filter(|s| s.bounds_log2.0.max(fmin) <= s.bounds_log2.1.min(fmax))
            {
                pitch = pitch.clamp(
                    snapshot.bounds_log2.0.max(fmin),
                    snapshot.bounds_log2.1.min(fmax),
                );
            }
            if let Some((lo, hi)) = melodic_bounds {
                pitch = pitch.clamp(lo, hi);
            }
            if (melodic.is_some() && scored.last().is_some_and(|&(x, _)| x == pitch))
                || (self.pitch_set.is_some()
                    && scored
                        .iter()
                        .any(|&(x, _)| x.exp2().to_bits() == pitch.exp2().to_bits()))
            {
                continue;
            }
            let score = scorer(pitch);
            if melodic.is_some() && !score.is_finite() {
                return Err("nonfinite_melodic_score");
            }
            if score.is_finite() {
                scored.push((pitch, score));
            }
        }
        if scored.is_empty() {
            if melodic.is_some() {
                return Err("empty_candidates");
            }
            return Ok(TargetProposal {
                target_pitch_log2: current_target_log2,
                salience: 0.0,
            });
        }

        #[cfg(feature = "loo-diagnostic")]
        diagnostic::record_candidates(scored);
        let current_target_log2 = self
            .sequence
            .as_ref()
            .filter(|s| s.bounds_log2.0.max(fmin) <= s.bounds_log2.1.min(fmax))
            .map_or(current_target_log2, |snapshot| {
                current_target_log2.clamp(
                    snapshot.bounds_log2.0.max(fmin),
                    snapshot.bounds_log2.1.min(fmax),
                )
            });
        let current_target_log2 = melodic_bounds.map_or(current_target_log2, |(lo, hi)| {
            current_target_log2.clamp(lo, hi)
        });
        let current_target_log2 = self.pitch_set.as_ref().map_or(current_target_log2, |ps| {
            current_target_log2.clamp(ps.bounds.0, ps.bounds.1)
        });
        let current_adjusted = scorer(current_target_log2);
        #[cfg(test)]
        crate::core::tonal_context::acquisition::decision(
            scored,
            current_target_log2,
            current_adjusted,
            sample_ownership_score(
                landscape,
                current_pitch_log2,
                self.loo_input
                    .exact_score
                    .as_deref()
                    .filter(|_| self.leave_self_out),
            ),
            landscape.consonance_representation,
            self.leave_self_out,
        );
        let mut current_target_log2 = current_target_log2;
        let mut current_adjusted = current_adjusted;
        if let Some(tonal) = &self.tonal {
            counterfactual.clear();
            let baseline = current_target_log2.clamp(tonal.bounds.0, tonal.bounds.1);
            let mut known = tonal.query(baseline.exp2());
            for &(candidate, _) in scored.iter() {
                let candidate = candidate.clamp(tonal.bounds.0, tonal.bounds.1);
                if let Some(q) = tonal.query(candidate.exp2()) {
                    counterfactual.push((candidate, tonal.score(scorer(candidate), q)));
                } else {
                    if known.is_some()
                        && let Some(ps) = &self.pitch_set
                    {
                        ps.tc_failed_added.set(
                            ps.added
                                .borrow()
                                .iter()
                                .any(|p| p.exp2().to_bits() == candidate.exp2().to_bits()),
                        );
                    }
                    known = None;
                    break;
                }
            }
            tonal.selection_known.set(Some(known.is_some()));
            if let Some(q) = known {
                current_target_log2 = baseline;
                current_adjusted = tonal.score(scorer(baseline), q);
                scored.copy_from_slice(counterfactual);
            }
        }
        if let Some(snapshot) = melodic {
            if scored.len() > if self.pitch_set.is_some() { 66 } else { 64 } {
                return Err("candidate_resource_limit");
            }
            if !current_adjusted.is_finite() {
                return Err("nonfinite_melodic_score");
            }
            for (pitch, score) in scored.iter_mut() {
                let term = snapshot
                    .contribution(*pitch, current_target_log2)
                    .ok_or("nonfinite_melodic_score")?;
                let combined = (*score as f64 + term) as f32;
                if !combined.is_finite() {
                    return Err("nonfinite_melodic_score");
                }
                *score = combined;
            }
        }
        Ok(self.select_with_sequence(
            current_target_log2,
            current_adjusted,
            scored,
            counterfactual,
            rng,
        ))
    }
}

impl PitchCore for PitchHillClimbPitchCore {
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
        let move_cost_time_sec = self.move_cost_time_sec(integration_window);
        let tessitura_center = self.tessitura_center;
        let tessitura_gravity = self.tessitura_gravity;
        let landscape_weight = self.landscape_weight;
        let move_cost_coeff = self.move_cost_coeff;
        let move_cost_exp = self.move_cost_exp;
        let leave_self_out = self.leave_self_out;
        #[cfg(any(test, feature = "loo-diagnostic"))]
        let reference_loo_mode = self.reference_loo_mode;
        #[cfg(not(any(test, feature = "loo-diagnostic")))]
        let reference_loo_mode = ReferenceLooMode::ApproxHarmonics;
        let reference_harmonics = DEFAULT_LOO_HARMONICS;
        let crowding_strength = self.crowding_strength;
        let crowding_sigma_cents = self.crowding_sigma_cents;
        let body_ratios = self.body_ratios.clone();
        #[cfg(feature = "loo-diagnostic")]
        let variant = self
            .diagnostic_variant
            .unwrap_or_else(diagnostic::runtime_variant);
        #[cfg(all(test, not(feature = "loo-diagnostic")))]
        let variant = "corrected";
        #[cfg(feature = "loo-diagnostic")]
        diagnostic::set_variant(variant);
        let loo_scan = if leave_self_out {
            loo_score_scan(landscape, &mut self.loo_input)
        } else {
            None
        };
        #[cfg(any(test, feature = "loo-diagnostic"))]
        let reference_loo_mode = if variant == "exact" {
            ReferenceLooMode::ExactScan
        } else {
            reference_loo_mode
        };
        #[cfg(any(test, feature = "loo-diagnostic"))]
        let loo_scan: Option<Arc<[f32]>> = if !leave_self_out || variant == "legacy" {
            None
        } else if matches!(reference_loo_mode, ReferenceLooMode::ExactScan) || variant == "exact" {
            exact_loo_consonance_score_scan(
                landscape,
                current_pitch_log2,
                true,
                ReferenceLooMode::ExactScan,
                &mut self.loo_scratch,
            )
            .map(Arc::from)
        } else {
            loo_scan
        };
        let support = self
            .loo_input
            .support
            .clone()
            .filter(|_| leave_self_out && landscape.harmonic.support_weights.active());
        let single_density = self.loo_input.support_single_density;
        let support_template = self.loo_input.support_template.clone();
        let scorer = |pitch_log2, replace| {
            let base = adjusted_pitch_score_impl(
                pitch_log2,
                current_pitch_log2,
                move_cost_time_sec,
                tessitura_center,
                if replace { 0.0 } else { tessitura_gravity },
                landscape_weight,
                if replace { 0.0 } else { move_cost_coeff },
                move_cost_exp,
                landscape,
                perceptual,
                leave_self_out,
                reference_loo_mode,
                reference_harmonics,
                self.loo_input
                    .candidate_scan(&landscape.space, pitch_log2, loo_scan.as_deref()),
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
        let snapshot = self.melodic;
        let original_rng = snapshot.map(|_| rng.clone());
        let attempt = self.propose_with_melodic_scorer(
            current_pitch_log2,
            current_target_log2,
            landscape,
            neighbor_pitch_log2,
            rng,
            snapshot,
            |p| scorer(p, snapshot.is_some()),
        );
        let reason = attempt.as_ref().err().copied();
        let proposal = match attempt {
            Ok(proposal) => proposal,
            Err(_) => {
                *rng = original_rng.as_ref().expect("ME attempt RNG").clone();
                self.propose_with_scorer(
                    current_pitch_log2,
                    current_target_log2,
                    landscape,
                    neighbor_pitch_log2,
                    rng,
                    |p| scorer(p, false),
                )
            }
        };
        let mut without = proposal.target_pitch_log2;
        if self.melodic_trace
            && reason.is_none()
            && let Some(mut counter_rng) = original_rng
        {
            let ps_record = self
                .pitch_set
                .as_ref()
                .and_then(|p| p.decision.borrow_mut().take());
            let sequence_record = self.sequence_decision.borrow_mut().take();
            let tonal_known = self.tonal.as_ref().and_then(|t| t.selection_known.get());
            without = self
                .propose_with_scorer(
                    current_pitch_log2,
                    current_target_log2,
                    landscape,
                    neighbor_pitch_log2,
                    &mut counter_rng,
                    |p| scorer(p, false),
                )
                .target_pitch_log2;
            *self.sequence_decision.borrow_mut() = sequence_record;
            if let Some(ps) = &self.pitch_set {
                *ps.decision.borrow_mut() = ps_record;
            }
            if let Some(tonal) = &self.tonal {
                tonal.selection_known.set(tonal_known);
            }
        }
        if snapshot.is_some() || self.melodic_trace {
            *self.melodic_decision.borrow_mut() = Some(crate::life::melodic_continuity::Decision {
                at: 0,
                snapshot,
                reason,
                old_terms_replaced: snapshot.is_some() && reason.is_none(),
                selected_log2: proposal.target_pitch_log2,
                without_melodic_log2: without,
                changed: proposal.target_pitch_log2.to_bits() != without.to_bits(),
            });
        } else {
            *self.melodic_decision.borrow_mut() = None;
        }
        proposal
    }
}

#[cfg(test)]
pub(crate) fn aud_m1_hill_choice(
    landscape: &Landscape,
    anchor: f32,
    neighbors: &[f32],
) -> (f32, f32) {
    use rand::SeedableRng;
    let mut core = PitchHillClimbPitchCore::new(landscape.space.step() * 1200.0, anchor, 0.0);
    core.set_global_peaks(8, 20.0);
    core.set_ratio_candidates(true, 20);
    core.move_cost_coeff = 0.0;
    let choice = core
        .propose_with_scorer(
            anchor,
            anchor,
            landscape,
            neighbors,
            &mut rand::rngs::SmallRng::seed_from_u64(719),
            |p| landscape.evaluate_pitch_score_log2(p),
        )
        .target_pitch_log2;
    let mut ranked = core.scratch.borrow().scored.clone();
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    let margin = if ranked.len() > 1 {
        ranked[0].1 - ranked[1].1
    } else {
        f32::INFINITY
    };
    (choice, margin)
}
