use crate::core::harmonic_ratios::{HARMONIC_RATIOS, ratio_to_f32};
use crate::core::harmonicity_kernel::HarmonicityKernel;
use crate::core::landscape::Landscape;
use crate::core::log2space::Log2Space;
use crate::core::psycho_state::{normalize_density, roughness_ratio_to_state01};
use crate::core::roughness_kernel::{RoughnessKernel, erb_grid};
use crate::life::adaptation::{AdaptationContext, FeaturesNow};
use crate::life::control::{LeaveSelfOutMode, MoveCostTimeScale, PitchControl, PitchCoreKind};
use rand::{Rng, RngExt};

const DEFAULT_LOCAL_WINDOW_CENTS: f32 = 240.0;
const DEFAULT_LOCAL_TOP_K: usize = 10;
const DEFAULT_RANDOM_CANDIDATES: usize = 3;
const DEFAULT_RANDOM_SIGMA_CENTS: f32 = 30.0;
const DEFAULT_MOVE_COST_COEFF: f32 = 0.5;
const DEFAULT_MOVE_COST_EXP: u8 = 1;
const DEFAULT_APPROX_LOO_SIGMA_CENTS: f32 = 24.0;
const DEFAULT_GLOBAL_PEAK_COUNT: usize = 0;
const DEFAULT_GLOBAL_PEAK_MIN_SEP_CENTS: f32 = 0.0;
const DEFAULT_RUNTIME_RATIO_CANDIDATE_COUNT: usize = 0;
const DEFAULT_LOO_HARMONICS: u8 = 1;
const DEFAULT_CROWDING_PAIR_SPLIT_EPS_FRAC: f32 = 0.25;
const EXACT_LOO_ROUGHNESS_ERB_STEP: f32 = 0.005;
/// Minimum field improvement required to accept a greedy hill-climb move.
const IMPROVEMENT_THRESHOLD: f32 = 0.1;

#[derive(Debug, Clone, Copy)]
pub struct TargetProposal {
    pub target_pitch_log2: f32,
    pub salience: f32,
}

#[allow(clippy::too_many_arguments)]
pub trait PitchCore {
    fn propose_target<R: Rng + ?Sized>(
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
    fn propose_target_with_crowding_salience<R: Rng + ?Sized>(
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
pub struct PitchHillClimbPitchCore {
    neighbor_step_log2: f32,
    tessitura_center: f32,
    tessitura_gravity: f32,
    landscape_weight: f32,
    move_cost_coeff: f32,
    move_cost_exp: u8,
    move_cost_time_scale: MoveCostTimeScale,
    proposal_interval_sec: Option<f32>,
    temperature: f32,
    crowding_strength: f32,
    crowding_sigma_cents: f32,
    octave_avoidance: f32,
    leave_self_out: bool,
    leave_self_out_mode: LeaveSelfOutMode,
    leave_self_out_harmonics: u8,
    global_peak_count: usize,
    global_peak_min_sep_log2: f32,
    use_ratio_candidates: bool,
    ratio_candidate_count: usize,
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
            octave_avoidance: 0.0,
            leave_self_out: false,
            leave_self_out_mode: LeaveSelfOutMode::ApproxHarmonics,
            leave_self_out_harmonics: DEFAULT_LOO_HARMONICS,
            global_peak_count: DEFAULT_GLOBAL_PEAK_COUNT,
            global_peak_min_sep_log2: cents_to_log2(DEFAULT_GLOBAL_PEAK_MIN_SEP_CENTS),
            use_ratio_candidates: false,
            ratio_candidate_count: DEFAULT_RUNTIME_RATIO_CANDIDATE_COUNT,
        }
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

    /// Octave-equivalence (chroma) weight for the occupancy field: 0 leaves
    /// octaves free, > 0 repels octave-doubling toward distinct chord tones.
    pub fn set_octave_avoidance(&mut self, value: f32) {
        self.octave_avoidance = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
    }

    pub fn set_leave_self_out(&mut self, enabled: bool) {
        self.leave_self_out = enabled;
    }

    pub fn set_leave_self_out_mode(&mut self, mode: LeaveSelfOutMode) {
        self.leave_self_out_mode = mode;
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

    pub fn set_leave_self_out_harmonics(&mut self, harmonics: u8) {
        self.leave_self_out_harmonics = harmonics.max(1);
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
    pub fn propose_with_scorer<R: Rng + ?Sized>(
        &self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        landscape: &Landscape,
        neighbor_pitch_log2: &[f32],
        rng: &mut R,
        mut scorer: impl FnMut(f32) -> f32,
    ) -> TargetProposal {
        let (fmin, fmax) = landscape.freq_bounds_log2();
        let current_target_log2 = current_target_log2.clamp(fmin, fmax);

        let adaptive_window_cents =
            (self.neighbor_step_log2.abs() * 1200.0 * 4.0).max(DEFAULT_LOCAL_WINDOW_CENTS);
        let window_bins = window_bins_from_cents(landscape, adaptive_window_cents);
        let mut candidates = seed_step_candidates(current_target_log2, self.neighbor_step_log2);
        candidates.extend(top_local_peaks(
            landscape,
            current_target_log2,
            window_bins,
            DEFAULT_LOCAL_TOP_K,
            &mut scorer,
        ));
        if self.global_peak_count > 0 {
            candidates.extend(top_global_peaks(
                landscape,
                self.global_peak_count,
                self.global_peak_min_sep_log2,
                &mut scorer,
            ));
        }
        if self.use_ratio_candidates && self.ratio_candidate_count > 0 {
            push_runtime_ratio_candidates(
                &mut candidates,
                current_target_log2,
                current_pitch_log2,
                neighbor_pitch_log2,
                self.ratio_candidate_count,
                fmin,
                fmax,
            );
        }
        push_gaussian_candidates(
            &mut candidates,
            current_target_log2,
            fmin,
            fmax,
            DEFAULT_RANDOM_SIGMA_CENTS,
            DEFAULT_RANDOM_CANDIDATES,
            rng,
        );
        candidates.retain(|x| x.is_finite());
        candidates = dedupe_log2_by_cents(candidates, 1.0);

        let scored = score_candidates_with(candidates, fmin, fmax, &mut scorer);
        if scored.is_empty() {
            return TargetProposal {
                target_pitch_log2: current_target_log2,
                salience: 0.0,
            };
        }

        let mut best_pitch = current_target_log2;
        let mut best_score = f32::MIN;
        let mut best_distance = f32::INFINITY;
        for &(pitch, score) in &scored {
            let distance = (pitch - current_target_log2).abs();
            let better_score = score > best_score + 1e-6;
            let same_score = (score - best_score).abs() <= 1e-6;
            let closer = distance < best_distance - 1e-6;
            let same_distance = (distance - best_distance).abs() <= 1e-6;
            let random_tie_break = same_score && same_distance && rng.random_range(0.0..1.0) < 0.5;
            if better_score || (same_score && closer) || random_tie_break {
                best_score = score;
                best_pitch = pitch;
                best_distance = distance;
            }
        }

        let current_adjusted = scorer(current_target_log2);
        let improvement = best_score - current_adjusted;
        let mut target_pitch_log2 = current_target_log2;

        if improvement > IMPROVEMENT_THRESHOLD {
            target_pitch_log2 = best_pitch;
        } else if self.temperature > 0.0 {
            // Metropolis step: pick a random movable candidate and accept it if it
            // improves, or with Boltzmann probability if it is downhill.
            let mut movable: Vec<(f32, f32)> = scored
                .iter()
                .copied()
                .filter(|(pitch, _)| (*pitch - current_target_log2).abs() > 1e-9)
                .collect();
            if !movable.is_empty() {
                let pick = rng.random_range(0..movable.len());
                let (candidate_pitch, candidate_score) = movable.swap_remove(pick);
                let delta = candidate_score - current_adjusted;
                if delta >= 0.0 || rng.random_range(0.0..1.0) < (delta / self.temperature).exp() {
                    target_pitch_log2 = candidate_pitch;
                }
            }
        }
        // else: temperature == 0, settle (greedy hill-climb stays put).

        TargetProposal {
            target_pitch_log2,
            salience: (improvement / 0.2).clamp(0.0, 1.0),
        }
    }
}

impl PitchCore for PitchHillClimbPitchCore {
    fn propose_target<R: Rng + ?Sized>(
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

    fn propose_target_with_crowding_salience<R: Rng + ?Sized>(
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
        let leave_self_out_mode = self.leave_self_out_mode;
        let leave_self_out_harmonics = self.leave_self_out_harmonics;
        let crowding_strength = self.crowding_strength;
        let crowding_sigma_cents = self.crowding_sigma_cents;
        let octave_avoidance = self.octave_avoidance;
        let exact_loo_scan =
            exact_loo_consonance_score_scan(landscape, current_pitch_log2, leave_self_out_mode);
        self.propose_with_scorer(
            current_pitch_log2,
            current_target_log2,
            landscape,
            neighbor_pitch_log2,
            rng,
            |pitch_log2| {
                adjusted_pitch_score_impl(
                    pitch_log2,
                    current_pitch_log2,
                    move_cost_time_sec,
                    tessitura_center,
                    tessitura_gravity,
                    landscape_weight,
                    move_cost_coeff,
                    move_cost_exp,
                    landscape,
                    perceptual,
                    leave_self_out,
                    leave_self_out_mode,
                    leave_self_out_harmonics,
                    exact_loo_scan.as_deref(),
                    crowding_strength,
                    crowding_sigma_cents,
                    octave_avoidance,
                    neighbor_pitch_log2,
                    neighbor_salience,
                )
            },
        )
    }
}

#[derive(Debug, Clone)]
pub struct PitchPeakSamplerCore {
    neighbor_step_log2: f32,
    window_cents: f32,
    top_k: usize,
    temperature: f32,
    sigma_cents: f32,
    random_candidates: usize,
    tessitura_center: f32,
    tessitura_gravity: f32,
    landscape_weight: f32,
    crowding_strength: f32,
    crowding_sigma_cents: f32,
    octave_avoidance: f32,
    leave_self_out: bool,
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
            octave_avoidance: 0.0,
            leave_self_out: false,
        }
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

    /// Octave-equivalence (chroma) weight for the occupancy field: 0 leaves
    /// octaves free, > 0 repels octave-doubling toward distinct chord tones.
    pub fn set_octave_avoidance(&mut self, value: f32) {
        self.octave_avoidance = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
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
    fn propose_target<R: Rng + ?Sized>(
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

    fn propose_target_with_crowding_salience<R: Rng + ?Sized>(
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
        let (fmin, fmax) = landscape.freq_bounds_log2();
        let current_target_log2 = current_target_log2.clamp(fmin, fmax);
        let window_bins = window_bins_from_cents(landscape, self.window_cents);
        let tessitura_center = self.tessitura_center;
        let tessitura_gravity = self.tessitura_gravity;
        let landscape_weight = self.landscape_weight;
        let leave_self_out = self.leave_self_out;
        let crowding_strength = self.crowding_strength;
        let crowding_sigma_cents = self.crowding_sigma_cents;
        let octave_avoidance = self.octave_avoidance;
        let mut scorer = |pitch_log2: f32| -> f32 {
            adjusted_pitch_score_impl(
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
                LeaveSelfOutMode::ApproxHarmonics,
                DEFAULT_LOO_HARMONICS,
                None,
                crowding_strength,
                crowding_sigma_cents,
                octave_avoidance,
                neighbor_pitch_log2,
                neighbor_salience,
            )
        };

        let mut candidates = seed_step_candidates(current_target_log2, self.neighbor_step_log2);
        candidates.extend(top_local_peaks(
            landscape,
            current_target_log2,
            window_bins,
            self.top_k,
            &mut scorer,
        ));
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

        let target_pitch_log2 = if self.temperature > 0.0 {
            sample_softmax_candidate(&scored, self.temperature, rng).clamp(fmin, fmax)
        } else {
            // Greedy: settle on the highest-scoring candidate (current target is
            // always among `scored`, so this never moves downhill).
            scored
                .iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(pitch, _)| *pitch)
                .unwrap_or(current_target_log2)
                .clamp(fmin, fmax)
        };

        TargetProposal {
            target_pitch_log2,
            salience: (improvement / 0.2).clamp(0.0, 1.0),
        }
    }
}

fn cents_to_log2(cents: f32) -> f32 {
    cents / 1200.0
}

fn window_bins_from_cents(landscape: &Landscape, cents: f32) -> usize {
    let step = landscape.space.step().max(1e-6);
    let span_log2 = cents_to_log2(cents.max(0.0));
    (span_log2 / step).ceil() as usize
}

/// Single fundamental-occupancy contribution of one neighbor at `candidate_log2`:
/// a Gaussian on octave (log2) distance with the pairwise split bias that keeps
/// two voices at the same f0 from drifting in lockstep. This is the one shared
/// definition of "how occupied is this fundamental", used by both the crowding
/// penalty and the adaptation occupancy feed (`PitchController`).
///
/// `octave_avoidance` (>= 0) adds an octave-equivalence (chroma) term: distance is
/// folded into one octave so unison *and* octaves count as occupied. `0` leaves
/// octaves free (they may converge); `> 0` repels octave-doubling toward distinct
/// chord tones. See docs/design-notes/voice-movement-redesign.md.
#[inline]
pub(crate) fn occupancy_contribution(
    candidate_log2: f32,
    neighbor_log2: f32,
    split_sign: f32,
    sigma_log2: f32,
    octave_avoidance: f32,
) -> f32 {
    let sigma = sigma_log2.max(1e-9);
    let eps = sigma * DEFAULT_CROWDING_PAIR_SPLIT_EPS_FRAC;
    let split = split_sign.clamp(-1.0, 1.0);
    let d = candidate_log2 - (neighbor_log2 - split * eps);
    let raw = (-0.5 * (d / sigma).powi(2)).exp();
    if octave_avoidance > 0.0 {
        let d_chroma = d - d.round(); // fold to [-0.5, 0.5] octave
        raw + octave_avoidance * (-0.5 * (d_chroma / sigma).powi(2)).exp()
    } else {
        raw
    }
}

/// Fill `out` with the fundamental-occupancy scan (the leave-self-out spatial
/// slice of `F`): deposit each visible neighbor's `occupancy_contribution` across
/// the Log2Space. The Gaussian on raw f0 only reaches a local window around the
/// neighbor; with octave avoidance the chroma term is octave-periodic, so deposits
/// land in a window around each octave multiple of the neighbor. Skipping the
/// negligible gaps between octaves makes this equivalent to a full scan but
/// O(octaves x window) instead of O(n_bins) per neighbor.
pub(crate) fn fill_occupancy_scan(
    out: &mut Vec<f32>,
    space: &Log2Space,
    neighbors: &[f32],
    neighbor_salience: &[f32],
    sigma_cents: f32,
    octave_avoidance: f32,
) {
    let n = space.n_bins();
    out.clear();
    out.resize(n, 0.0);
    if neighbors.is_empty() || n == 0 {
        return;
    }
    let sigma_log2 = (sigma_cents.max(1e-3)) / 1200.0;
    let margin = 4.0 * sigma_log2;
    let win = (margin * space.bins_per_oct as f32).ceil() as isize;
    let base = space.centers_log2[0];
    let top = space.centers_log2[n - 1];
    let bins_per_oct = space.bins_per_oct as f32;
    let n_isize = n as isize;
    for (idx, &nb) in neighbors.iter().enumerate() {
        let Some(center) = space.index_of_log2(nb) else {
            continue;
        };
        let split_sign = neighbor_salience.get(idx).copied().unwrap_or(0.0);
        // Octave offsets whose window intersects the space (k = 0 only without
        // octave avoidance, since the chroma term is then absent).
        let (k_min, k_max) = if octave_avoidance > 0.0 {
            (
                (base - margin - nb).ceil() as isize,
                (top + margin - nb).floor() as isize,
            )
        } else {
            (0, 0)
        };
        let mut prev_hi: isize = -1;
        for k in k_min..=k_max {
            let center_bin = if k == 0 {
                center as isize
            } else {
                ((nb + k as f32 - base) * bins_per_oct).round() as isize
            };
            let lo = (center_bin - win).max(0).max(prev_hi + 1);
            let hi = (center_bin + win).min(n_isize - 1);
            if hi < lo {
                continue;
            }
            for i in lo..=hi {
                out[i as usize] += occupancy_contribution(
                    space.centers_log2[i as usize],
                    nb,
                    split_sign,
                    sigma_log2,
                    octave_avoidance,
                );
            }
            prev_hi = hi;
        }
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn adjusted_pitch_score(
    pitch_log2: f32,
    current_pitch_log2: f32,
    integration_window: f32,
    tessitura_center: f32,
    tessitura_gravity: f32,
    landscape_weight: f32,
    move_cost_coeff: f32,
    move_cost_exp: u8,
    landscape: &Landscape,
    perceptual: &AdaptationContext,
    leave_self_out: bool,
    crowding_strength: f32,
    crowding_sigma_cents: f32,
    neighbor_pitch_log2: &[f32],
    neighbor_salience: &[f32],
) -> f32 {
    adjusted_pitch_score_impl(
        pitch_log2,
        current_pitch_log2,
        integration_window,
        tessitura_center,
        tessitura_gravity,
        landscape_weight,
        move_cost_coeff,
        move_cost_exp,
        landscape,
        perceptual,
        leave_self_out,
        LeaveSelfOutMode::ApproxHarmonics,
        DEFAULT_LOO_HARMONICS,
        None,
        crowding_strength,
        crowding_sigma_cents,
        0.0,
        neighbor_pitch_log2,
        neighbor_salience,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn adjusted_pitch_score_with_loo_harmonics(
    pitch_log2: f32,
    current_pitch_log2: f32,
    integration_window: f32,
    tessitura_center: f32,
    tessitura_gravity: f32,
    landscape_weight: f32,
    move_cost_coeff: f32,
    move_cost_exp: u8,
    landscape: &Landscape,
    perceptual: &AdaptationContext,
    leave_self_out: bool,
    leave_self_out_harmonics: u8,
    crowding_strength: f32,
    crowding_sigma_cents: f32,
    neighbor_pitch_log2: &[f32],
    neighbor_salience: &[f32],
) -> f32 {
    adjusted_pitch_score_impl(
        pitch_log2,
        current_pitch_log2,
        integration_window,
        tessitura_center,
        tessitura_gravity,
        landscape_weight,
        move_cost_coeff,
        move_cost_exp,
        landscape,
        perceptual,
        leave_self_out,
        LeaveSelfOutMode::ApproxHarmonics,
        leave_self_out_harmonics,
        None,
        crowding_strength,
        crowding_sigma_cents,
        0.0,
        neighbor_pitch_log2,
        neighbor_salience,
    )
}

#[allow(clippy::too_many_arguments)]
fn adjusted_pitch_score_impl(
    pitch_log2: f32,
    current_pitch_log2: f32,
    integration_window: f32,
    tessitura_center: f32,
    tessitura_gravity: f32,
    landscape_weight: f32,
    move_cost_coeff: f32,
    move_cost_exp: u8,
    landscape: &Landscape,
    perceptual: &AdaptationContext,
    leave_self_out: bool,
    leave_self_out_mode: LeaveSelfOutMode,
    leave_self_out_harmonics: u8,
    exact_loo_scan: Option<&[f32]>,
    crowding_strength: f32,
    crowding_sigma_cents: f32,
    octave_avoidance: f32,
    neighbor_pitch_log2: &[f32],
    neighbor_salience: &[f32],
) -> f32 {
    let (fmin, fmax) = landscape.freq_bounds_log2();
    let clamped = pitch_log2.clamp(fmin, fmax);
    let score = sample_consonance_score_with_loo(
        clamped,
        current_pitch_log2,
        landscape,
        leave_self_out,
        leave_self_out_mode,
        leave_self_out_harmonics,
        exact_loo_scan,
    );
    let distance_oct = (clamped - current_pitch_log2).abs();
    let dist_cost = if move_cost_exp == 2 {
        distance_oct * distance_oct
    } else {
        distance_oct
    };
    let penalty = dist_cost * integration_window * move_cost_coeff.max(0.0);
    let dist = clamped - tessitura_center;
    let gravity_penalty = dist * dist * tessitura_gravity;
    let weighted_score =
        landscape_weight.max(0.0) * landscape.pitch_objective_mode.apply_consonance_score(score);
    // Stage 2 of the occupancy redesign: crowding samples the same fundamental-
    // occupancy kernel as the adaptation feed (Log2 Gaussian of width
    // crowding_sigma_cents, self already excluded upstream).
    // See docs/design-notes/voice-movement-redesign.md.
    let crowding_penalty = if crowding_strength > 0.0 && !neighbor_pitch_log2.is_empty() {
        let sigma_log2 = cents_to_log2(crowding_sigma_cents.max(1e-3));
        let mut sum = 0.0f32;
        for (idx, &neighbor_log2) in neighbor_pitch_log2.iter().enumerate() {
            if !neighbor_log2.is_finite() {
                continue;
            }
            let split_sign = neighbor_salience.get(idx).copied().unwrap_or(0.0);
            sum += occupancy_contribution(
                clamped,
                neighbor_log2,
                split_sign,
                sigma_log2,
                octave_avoidance,
            );
        }
        crowding_strength * sum
    } else {
        0.0
    };
    let base = weighted_score - penalty - gravity_penalty - crowding_penalty;
    let idx = landscape.space.index_of_log2(clamped).unwrap_or(0);
    base + perceptual.score_adjustment(idx)
}

fn sample_consonance_score_with_loo(
    pitch_log2: f32,
    current_pitch_log2: f32,
    landscape: &Landscape,
    leave_self_out: bool,
    leave_self_out_mode: LeaveSelfOutMode,
    leave_self_out_harmonics: u8,
    exact_loo_scan: Option<&[f32]>,
) -> f32 {
    if leave_self_out
        && matches!(leave_self_out_mode, LeaveSelfOutMode::ExactScan)
        && let Some(scan) = exact_loo_scan
    {
        return landscape.sample_linear_log2(scan, pitch_log2);
    }
    let (fmin, fmax) = landscape.freq_bounds_log2();
    let clamped = pitch_log2.clamp(fmin, fmax);
    let mut score = landscape.evaluate_pitch_score_log2(clamped);
    if leave_self_out {
        let current_clamped = current_pitch_log2.clamp(fmin, fmax);
        let harmonics = leave_self_out_harmonics.max(1);
        let sigma = cents_to_log2(DEFAULT_APPROX_LOO_SIGMA_CENTS).max(1e-6);
        for harmonic in 1..=harmonics {
            let harmonic_f = harmonic as f32;
            let harmonic_log2 = current_clamped + harmonic_f.log2();
            if harmonic_log2 < fmin || harmonic_log2 > fmax {
                continue;
            }
            let self_score = landscape.evaluate_pitch_score_log2(harmonic_log2);
            if !self_score.is_finite() || self_score <= 0.0 {
                continue;
            }
            let d = (clamped - harmonic_log2).abs();
            let harmonic_weight = 1.0 / harmonic_f.max(1.0);
            score -= harmonic_weight * self_score * (-d / sigma).exp();
        }
    }
    score
}

pub(crate) fn approx_loo_pitch_score(landscape: &Landscape, freq_hz: f32, harmonics: u8) -> f32 {
    let pitch_log2 = freq_hz.max(1.0).log2();
    sample_consonance_score_with_loo(
        pitch_log2,
        pitch_log2,
        landscape,
        true,
        LeaveSelfOutMode::ApproxHarmonics,
        harmonics.max(1),
        None,
    )
}

fn exact_loo_consonance_score_scan(
    landscape: &Landscape,
    current_pitch_log2: f32,
    mode: LeaveSelfOutMode,
) -> Option<Vec<f32>> {
    if !matches!(mode, LeaveSelfOutMode::ExactScan) {
        return None;
    }
    let n = landscape.space.n_bins();
    if n == 0
        || landscape.subjective_intensity.len() != n
        || landscape.roughness01.len() != n
        || landscape.harmonicity01.len() != n
    {
        return None;
    }
    let current_idx = landscape.space.index_of_log2(current_pitch_log2)?;
    let current_mass = landscape
        .subjective_intensity
        .get(current_idx)
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)?;
    let mut density_loo = landscape.subjective_intensity.clone();
    density_loo[current_idx] = (density_loo[current_idx] - current_mass).max(0.0);

    let harmonicity_kernel = HarmonicityKernel::new(&landscape.space, landscape.harmonicity_params);
    let h_dual =
        harmonicity_kernel.potential_h_dual_from_log2_spectrum(&density_loo, &landscape.space);
    let roughness_kernel = RoughnessKernel::new(
        landscape.roughness_kernel_params,
        EXACT_LOO_ROUGHNESS_ERB_STEP,
    );
    let (_erb, du) = erb_grid(&landscape.space);
    let eps = landscape.roughness_ref_eps.max(1e-12);
    let (p_density, mass) = normalize_density(&density_loo, &du, eps);
    let r_shape_raw = if mass > eps {
        roughness_kernel
            .potential_r_from_log2_spectrum_density(&p_density, &landscape.space)
            .0
    } else {
        vec![0.0; n]
    };
    let roughness_ref_peak = landscape.roughness_ref_peak.max(eps);
    let roughness_k = landscape.roughness_k.max(1e-6);
    let mut out = Vec::with_capacity(n);
    for (i, r_shape) in r_shape_raw.iter().enumerate().take(n) {
        let h01 = h_dual
            .blended
            .get(i)
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let r01 = roughness_ratio_to_state01(*r_shape / roughness_ref_peak, roughness_k);
        out.push(landscape.consonance_kernel.score(h01, r01));
    }
    Some(out)
}

fn top_local_peaks(
    landscape: &Landscape,
    current_target_log2: f32,
    window_bins: usize,
    top_k: usize,
    scorer: &mut impl FnMut(f32) -> f32,
) -> Vec<f32> {
    let (fmin, fmax) = landscape.freq_bounds_log2();
    let n_bins = landscape.space.n_bins();
    if n_bins == 0 {
        return Vec::new();
    }
    let clamped_target = current_target_log2.clamp(fmin, fmax);
    let idx0 = landscape.space.index_of_log2(clamped_target).unwrap_or(0);
    let start = idx0.saturating_sub(window_bins);
    let end = idx0.saturating_add(window_bins).min(n_bins - 1);

    let mut scored = Vec::with_capacity(end.saturating_sub(start) + 1);
    for idx in start..=end {
        let pitch_log2 = landscape.space.centers_log2[idx];
        let score = scorer(pitch_log2);
        if score.is_finite() {
            scored.push((score, pitch_log2));
        }
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));

    let limit = top_k.max(1).min(scored.len());
    scored
        .into_iter()
        .take(limit)
        .map(|(_, pitch)| pitch)
        .collect()
}

fn top_global_peaks(
    landscape: &Landscape,
    top_k: usize,
    min_sep_log2: f32,
    scorer: &mut impl FnMut(f32) -> f32,
) -> Vec<f32> {
    let n_bins = landscape.space.n_bins();
    if n_bins == 0 || top_k == 0 {
        return Vec::new();
    }
    let mut scored = Vec::with_capacity(n_bins);
    for &pitch_log2 in &landscape.space.centers_log2 {
        let score = scorer(pitch_log2);
        if score.is_finite() {
            scored.push((score, pitch_log2));
        }
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut out: Vec<f32> = Vec::with_capacity(top_k.min(scored.len()));
    for (_, pitch_log2) in scored {
        if min_sep_log2 > 0.0
            && out
                .iter()
                .any(|selected| (pitch_log2 - *selected).abs() < min_sep_log2)
        {
            continue;
        }
        out.push(pitch_log2);
        if out.len() >= top_k {
            break;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn push_runtime_ratio_candidates(
    candidates: &mut Vec<f32>,
    current_target_log2: f32,
    current_pitch_log2: f32,
    neighbor_pitch_log2: &[f32],
    max_count: usize,
    min_log2: f32,
    max_log2: f32,
) {
    if max_count == 0 {
        return;
    }
    let mut anchors = vec![current_target_log2, current_pitch_log2];
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for &neighbor in neighbor_pitch_log2 {
        if neighbor.is_finite() {
            sum += neighbor;
            count += 1;
        }
    }
    if count > 0 {
        anchors.push(sum / count as f32);
    }

    let mut generated = Vec::with_capacity(anchors.len() * HARMONIC_RATIOS.len() * 2);
    for anchor in anchors {
        if !anchor.is_finite() {
            continue;
        }
        for &ratio in HARMONIC_RATIOS {
            let ratio_f = ratio_to_f32(ratio);
            if !ratio_f.is_finite() || ratio_f <= 0.0 {
                continue;
            }
            let ratio_log2 = ratio_f.log2();
            generated.push((anchor + ratio_log2).clamp(min_log2, max_log2));
            generated.push((anchor - ratio_log2).clamp(min_log2, max_log2));
        }
    }
    generated.retain(|x| x.is_finite());
    generated = dedupe_log2_by_cents(generated, 2.0);
    generated.sort_by(|a, b| {
        (a - current_target_log2)
            .abs()
            .total_cmp(&(b - current_target_log2).abs())
    });
    candidates.extend(generated.into_iter().take(max_count));
}

fn seed_step_candidates(current_target_log2: f32, neighbor_step_log2: f32) -> Vec<f32> {
    vec![
        current_target_log2,
        current_target_log2 + neighbor_step_log2,
        current_target_log2 - neighbor_step_log2,
    ]
}

fn score_candidates_with(
    candidates: Vec<f32>,
    fmin: f32,
    fmax: f32,
    mut score_fn: impl FnMut(f32) -> f32,
) -> Vec<(f32, f32)> {
    let mut scored = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let pitch = candidate.clamp(fmin, fmax);
        let score = score_fn(pitch);
        if score.is_finite() {
            scored.push((pitch, score));
        }
    }
    scored
}

fn push_gaussian_candidates<R: Rng + ?Sized>(
    candidates: &mut Vec<f32>,
    center_log2: f32,
    min_log2: f32,
    max_log2: f32,
    sigma_cents: f32,
    count: usize,
    rng: &mut R,
) {
    if count == 0 || sigma_cents <= 0.0 {
        return;
    }
    let sigma_log2 = cents_to_log2(sigma_cents);
    for _ in 0..count {
        let z = sample_standard_normal(rng);
        let candidate = (center_log2 + z * sigma_log2).clamp(min_log2, max_log2);
        candidates.push(candidate);
    }
}

fn sample_standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let u1 = rng.random_range(f32::EPSILON..1.0);
    let u2 = rng.random_range(0.0..1.0);
    let mag = (-2.0 * u1.ln()).sqrt();
    let angle = core::f32::consts::TAU * u2;
    mag * angle.cos()
}

fn sample_softmax_candidate<R: Rng + ?Sized>(
    scored: &[(f32, f32)],
    temperature: f32,
    rng: &mut R,
) -> f32 {
    if scored.is_empty() {
        return 0.0;
    }
    let best = scored
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(pitch, _)| *pitch)
        .unwrap_or(scored[0].0);

    let max_score = scored
        .iter()
        .map(|(_, score)| *score)
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(0.0);
    let temp = temperature.max(1e-4);

    let mut weights = Vec::with_capacity(scored.len());
    let mut total = 0.0f32;
    for &(_, score) in scored {
        let weight = ((score - max_score) / temp).exp();
        let weight = if weight.is_finite() && weight > 0.0 {
            weight
        } else {
            0.0
        };
        total += weight;
        weights.push(weight);
    }

    if total <= 0.0 {
        return best;
    }

    let mut draw = rng.random_range(0.0..total);
    for ((pitch, _), weight) in scored.iter().zip(weights.iter()) {
        if *weight <= 0.0 {
            continue;
        }
        if draw <= *weight {
            return *pitch;
        }
        draw -= *weight;
    }

    best
}

fn dedupe_log2_by_cents(mut values_log2: Vec<f32>, dedupe_cents: f32) -> Vec<f32> {
    if values_log2.is_empty() {
        return values_log2;
    }
    let min_log2 = cents_to_log2(dedupe_cents.max(0.0));
    values_log2.retain(|x| x.is_finite());
    values_log2.sort_by(|a, b| a.total_cmp(b));
    let mut out = Vec::with_capacity(values_log2.len());
    let mut last: Option<f32> = None;
    for value in values_log2 {
        if let Some(prev) = last
            && (value - prev).abs() < min_log2
        {
            continue;
        }
        last = Some(value);
        out.push(value);
    }
    out
}

#[derive(Debug, Clone)]
pub enum AnyPitchCore {
    PitchHillClimb(PitchHillClimbPitchCore),
    PitchPeakSampler(PitchPeakSamplerCore),
}

impl PitchCore for AnyPitchCore {
    fn propose_target<R: Rng + ?Sized>(
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

    fn propose_target_with_crowding_salience<R: Rng + ?Sized>(
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
                    pitch.temperature.unwrap_or(0.08),
                    pitch.sigma_cents.unwrap_or(DEFAULT_RANDOM_SIGMA_CENTS),
                    pitch.random_candidates.unwrap_or(DEFAULT_RANDOM_CANDIDATES),
                ))
            }
        };
        core.apply_control(pitch);
        core
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
}

fn apply_hill_climb_control(core: &mut PitchHillClimbPitchCore, pitch: &PitchControl) {
    core.set_tessitura_center(pitch.freq.max(1.0).log2());
    core.set_tessitura_gravity(pitch.resolved_tessitura_gravity());
    core.set_landscape_weight(pitch.landscape_weight);
    core.set_temperature(pitch.temperature.unwrap_or(0.0));
    core.set_crowding(pitch.crowding_strength, pitch.crowding_sigma_cents);
    core.set_octave_avoidance(pitch.octave_avoidance);
    core.set_leave_self_out(pitch.leave_self_out);
    core.set_leave_self_out_mode(pitch.leave_self_out_mode);
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
    core.set_leave_self_out_harmonics(pitch.leave_self_out_harmonics);
    core.set_proposal_interval_sec(pitch.proposal_interval_sec);
}

fn apply_peak_sampler_control(core: &mut PitchPeakSamplerCore, pitch: &PitchControl) {
    core.set_tessitura_center(pitch.freq.max(1.0).log2());
    core.set_tessitura_gravity(pitch.resolved_tessitura_gravity());
    core.set_landscape_weight(pitch.landscape_weight);
    core.set_temperature(pitch.temperature.unwrap_or(0.08));
    core.set_crowding(pitch.crowding_strength, pitch.crowding_sigma_cents);
    core.set_octave_avoidance(pitch.octave_avoidance);
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
impl AnyPitchCore {
    pub(crate) fn leave_self_out_for_test(&self) -> bool {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.leave_self_out,
            AnyPitchCore::PitchPeakSampler(core) => core.leave_self_out,
        }
    }

    pub(crate) fn leave_self_out_mode_for_test(&self) -> LeaveSelfOutMode {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.leave_self_out_mode,
            AnyPitchCore::PitchPeakSampler(_) => LeaveSelfOutMode::ApproxHarmonics,
        }
    }

    pub(crate) fn move_cost_coeff_for_test(&self) -> f32 {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.move_cost_coeff,
            AnyPitchCore::PitchPeakSampler(_) => 0.0,
        }
    }

    pub(crate) fn move_cost_exp_for_test(&self) -> u8 {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.move_cost_exp,
            AnyPitchCore::PitchPeakSampler(_) => DEFAULT_MOVE_COST_EXP,
        }
    }

    pub(crate) fn proposal_interval_sec_for_test(&self) -> Option<f32> {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.proposal_interval_sec,
            AnyPitchCore::PitchPeakSampler(_) => None,
        }
    }

    pub(crate) fn global_peak_count_for_test(&self) -> usize {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.global_peak_count,
            AnyPitchCore::PitchPeakSampler(_) => 0,
        }
    }

    pub(crate) fn ratio_candidate_count_for_test(&self) -> usize {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.ratio_candidate_count,
            AnyPitchCore::PitchPeakSampler(_) => 0,
        }
    }

    pub(crate) fn use_ratio_candidates_for_test(&self) -> bool {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.use_ratio_candidates,
            AnyPitchCore::PitchPeakSampler(_) => false,
        }
    }

    pub(crate) fn move_cost_time_scale_for_test(&self) -> MoveCostTimeScale {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.move_cost_time_scale,
            AnyPitchCore::PitchPeakSampler(_) => MoveCostTimeScale::LegacyIntegrationWindow,
        }
    }

    pub(crate) fn leave_self_out_harmonics_for_test(&self) -> u8 {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.leave_self_out_harmonics,
            AnyPitchCore::PitchPeakSampler(_) => DEFAULT_LOO_HARMONICS,
        }
    }

    pub(crate) fn crowding_strength_for_test(&self) -> f32 {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.crowding_strength,
            AnyPitchCore::PitchPeakSampler(core) => core.crowding_strength,
        }
    }

    pub(crate) fn crowding_sigma_cents_for_test(&self) -> f32 {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.crowding_sigma_cents,
            AnyPitchCore::PitchPeakSampler(core) => core.crowding_sigma_cents,
        }
    }

    pub(crate) fn neighbor_step_cents_for_test(&self) -> f32 {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.neighbor_step_log2 * 1200.0,
            AnyPitchCore::PitchPeakSampler(core) => core.neighbor_step_log2 * 1200.0,
        }
    }

    pub(crate) fn tessitura_gravity_for_test(&self) -> f32 {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.tessitura_gravity,
            AnyPitchCore::PitchPeakSampler(core) => core.tessitura_gravity,
        }
    }

    pub(crate) fn window_cents_for_test(&self) -> f32 {
        match self {
            AnyPitchCore::PitchHillClimb(_) => DEFAULT_LOCAL_WINDOW_CENTS,
            AnyPitchCore::PitchPeakSampler(core) => core.window_cents,
        }
    }

    pub(crate) fn top_k_for_test(&self) -> usize {
        match self {
            AnyPitchCore::PitchHillClimb(_) => DEFAULT_LOCAL_TOP_K,
            AnyPitchCore::PitchPeakSampler(core) => core.top_k,
        }
    }

    pub(crate) fn temperature_for_test(&self) -> f32 {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.temperature,
            AnyPitchCore::PitchPeakSampler(core) => core.temperature,
        }
    }

    pub(crate) fn sigma_cents_for_test(&self) -> f32 {
        match self {
            AnyPitchCore::PitchHillClimb(_) => DEFAULT_RANDOM_SIGMA_CENTS,
            AnyPitchCore::PitchPeakSampler(core) => core.sigma_cents,
        }
    }

    pub(crate) fn random_candidates_for_test(&self) -> usize {
        match self {
            AnyPitchCore::PitchHillClimb(_) => DEFAULT_RANDOM_CANDIDATES,
            AnyPitchCore::PitchPeakSampler(core) => core.random_candidates,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::log2space::Log2Space;
    use crate::life::adaptation::AdaptationConfig;
    use rand::{SeedableRng, rngs::SmallRng};

    fn test_landscape(peaks_hz: &[(f32, f32)]) -> Landscape {
        let mut landscape = Landscape::new(Log2Space::new(110.0, 880.0, 96));
        let width_cents = 40.0f32;
        for (idx, &bin_log2) in landscape.space.centers_log2.iter().enumerate() {
            let mut score = 0.0f32;
            for &(freq_hz, gain) in peaks_hz {
                let center = freq_hz.max(1.0).log2();
                let d_cents = (bin_log2 - center).abs() * 1200.0;
                let g = (-(d_cents * d_cents) / (2.0 * width_cents * width_cents)).exp();
                score += gain * g;
            }
            landscape.consonance_field_score[idx] = score;
            landscape.consonance_field_level[idx] = score.clamp(0.0, 1.0);
        }
        landscape
    }

    fn test_adaptation(n_bins: usize) -> AdaptationContext {
        AdaptationContext::from_config(&AdaptationConfig::default(), n_bins)
    }

    #[test]
    fn approx_loo_pitch_score_reduces_positive_self_peak_score() {
        let landscape = test_landscape(&[(220.0, 1.0)]);
        let raw = landscape.evaluate_pitch_score(220.0);
        let loo = approx_loo_pitch_score(&landscape, 220.0, 1);

        assert!(raw > 0.0);
        assert!(loo < raw);
    }

    #[test]
    fn adjusted_score_landscape_weight_zero_disables_landscape_term() {
        let mut landscape = Landscape::new(Log2Space::new(110.0, 880.0, 96));
        let idx = landscape.space.n_bins() / 2;
        let pitch_log2 = landscape.space.centers_log2[idx];
        landscape.consonance_field_score[idx] = 1.75;
        let perceptual = test_adaptation(landscape.space.n_bins());

        let weighted = adjusted_pitch_score(
            pitch_log2,
            pitch_log2,
            0.0,
            pitch_log2,
            0.0,
            1.0,
            0.0,
            1,
            &landscape,
            &perceptual,
            false,
            0.0,
            0.0,
            &[],
            &[],
        );
        let without_landscape = adjusted_pitch_score(
            pitch_log2,
            pitch_log2,
            0.0,
            pitch_log2,
            0.0,
            0.0,
            0.0,
            1,
            &landscape,
            &perceptual,
            false,
            0.0,
            0.0,
            &[],
            &[],
        );

        assert!((weighted - 1.75).abs() <= 1e-6);
        assert!(without_landscape.abs() <= 1e-6);
    }

    #[test]
    fn occupancy_octave_avoidance_repels_octave_but_not_fifth() {
        let sigma = 60.0 / 1200.0;
        let unison = 0.0_f32;
        let octave = 1.0_f32; // log2 distance of one octave
        let fifth = (3.0_f32 / 2.0).log2();
        // Without octave avoidance the octave is invisible (far in raw f0).
        assert!(occupancy_contribution(octave, unison, 0.0, sigma, 0.0) < 1e-3);
        // With octave avoidance the octave is repelled (chroma folds to unison).
        assert!(occupancy_contribution(octave, unison, 0.0, sigma, 1.0) > 0.5);
        // A fifth stays free even with octave avoidance (chroma ~415c away).
        assert!(occupancy_contribution(fifth, unison, 0.0, sigma, 1.0) < 0.1);
        // Unison is always occupied.
        assert!(occupancy_contribution(unison, unison, 0.0, sigma, 0.0) > 0.9);
    }

    #[test]
    fn fill_occupancy_scan_octave_windows_match_full_scan() {
        let space = Log2Space::new(110.0, 880.0, 48);
        let n = space.n_bins();
        let neighbors = [220.0_f32.log2(), 330.0_f32.log2()];
        let salience = [1.0_f32, -1.0];
        let sigma_cents = 60.0;
        let octave_avoidance = 1.0_f32;

        let mut windowed = Vec::new();
        fill_occupancy_scan(
            &mut windowed,
            &space,
            &neighbors,
            &salience,
            sigma_cents,
            octave_avoidance,
        );

        // Reference: brute-force full scan over every bin.
        let sigma_log2 = sigma_cents / 1200.0;
        let mut full = vec![0.0f32; n];
        for (idx, &nb) in neighbors.iter().enumerate() {
            let split = salience[idx];
            for (i, slot) in full.iter_mut().enumerate() {
                *slot += occupancy_contribution(
                    space.centers_log2[i],
                    nb,
                    split,
                    sigma_log2,
                    octave_avoidance,
                );
            }
        }

        let max_diff = windowed
            .iter()
            .zip(full.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "octave-windowed occupancy must match the full scan (max_diff={max_diff})"
        );
        // And it must actually be octave-aware: the octave of 220 (=440) is occupied.
        let oct_bin = space.index_of_log2(440.0f32.log2()).unwrap();
        assert!(
            windowed[oct_bin] > 0.1,
            "a neighbor's octave should be occupied"
        );
    }

    #[test]
    fn adjusted_score_crowding_penalizes_close_neighbor() {
        let mut landscape = Landscape::new(Log2Space::new(110.0, 880.0, 96));
        let idx = landscape.space.n_bins() / 2;
        let pitch_log2 = landscape.space.centers_log2[idx];
        landscape.consonance_field_score[idx] = 1.0;
        let perceptual = test_adaptation(landscape.space.n_bins());

        let without_crowding = adjusted_pitch_score(
            pitch_log2,
            pitch_log2,
            0.0,
            pitch_log2,
            0.0,
            1.0,
            0.0,
            1,
            &landscape,
            &perceptual,
            false,
            1.0,
            20.0,
            &[],
            &[],
        );
        let with_close_neighbor = adjusted_pitch_score(
            pitch_log2,
            pitch_log2,
            0.0,
            pitch_log2,
            0.0,
            1.0,
            0.0,
            1,
            &landscape,
            &perceptual,
            false,
            1.0,
            20.0,
            &[pitch_log2],
            &[1.0],
        );

        assert!(
            with_close_neighbor < without_crowding,
            "crowding should lower score when neighbor is near"
        );
    }

    #[test]
    fn crowding_pairwise_split_biases_agents_in_opposite_directions() {
        let mut landscape = Landscape::new(Log2Space::new(110.0, 880.0, 128));
        landscape.consonance_field_score.fill(1.0);
        let perceptual = test_adaptation(landscape.space.n_bins());
        let center = landscape.space.centers_log2[landscape.space.n_bins() / 2];
        let left = center - cents_to_log2(6.0);
        let right = center + cents_to_log2(6.0);
        let neighbor = [center];

        let left_pos = adjusted_pitch_score(
            left,
            center,
            0.0,
            center,
            0.0,
            1.0,
            0.0,
            1,
            &landscape,
            &perceptual,
            false,
            4.0,
            60.0,
            &neighbor,
            &[1.0],
        );
        let right_pos = adjusted_pitch_score(
            right,
            center,
            0.0,
            center,
            0.0,
            1.0,
            0.0,
            1,
            &landscape,
            &perceptual,
            false,
            4.0,
            60.0,
            &neighbor,
            &[1.0],
        );
        let left_neg = adjusted_pitch_score(
            left,
            center,
            0.0,
            center,
            0.0,
            1.0,
            0.0,
            1,
            &landscape,
            &perceptual,
            false,
            4.0,
            60.0,
            &neighbor,
            &[-1.0],
        );
        let right_neg = adjusted_pitch_score(
            right,
            center,
            0.0,
            center,
            0.0,
            1.0,
            0.0,
            1,
            &landscape,
            &perceptual,
            false,
            4.0,
            60.0,
            &neighbor,
            &[-1.0],
        );

        let dir_pos = right_pos - left_pos;
        let dir_neg = right_neg - left_neg;
        assert!(
            dir_pos * dir_neg < 0.0,
            "pairwise split sign should induce opposite directional preference (dir_pos={dir_pos}, dir_neg={dir_neg})"
        );
    }

    #[test]
    fn crowding_samples_sigma_cents_gaussian_not_roughness() {
        let mut landscape = Landscape::new(Log2Space::new(110.0, 880.0, 96));
        let idx = landscape.space.n_bins() / 2;
        let pitch_log2 = landscape.space.centers_log2[idx];
        let neighbor_log2 = pitch_log2 + cents_to_log2(40.0);
        landscape.consonance_field_score[idx] = 1.0;
        let perceptual = test_adaptation(landscape.space.n_bins());

        landscape.roughness_suppress_sigma_erb = 0.04;
        landscape.roughness_kernel_params.suppress_sigma_erb = 0.04;
        let narrow = adjusted_pitch_score(
            pitch_log2,
            pitch_log2,
            0.0,
            pitch_log2,
            0.0,
            1.0,
            0.0,
            1,
            &landscape,
            &perceptual,
            false,
            1.0,
            20.0,
            &[neighbor_log2],
            &[1.0],
        );

        landscape.roughness_suppress_sigma_erb = 0.10;
        landscape.roughness_kernel_params.suppress_sigma_erb = 0.10;
        let wide = adjusted_pitch_score(
            pitch_log2,
            pitch_log2,
            0.0,
            pitch_log2,
            0.0,
            1.0,
            0.0,
            1,
            &landscape,
            &perceptual,
            false,
            1.0,
            20.0,
            &[neighbor_log2],
            &[1.0],
        );
        assert!(
            (wide - narrow).abs() <= 1e-6,
            "crowding samples a fixed sigma_cents Gaussian and ignores roughness sigma"
        );
    }

    #[test]
    fn loo_off_preserves_legacy_score_formula() {
        let mut landscape = Landscape::new(Log2Space::new(110.0, 880.0, 96));
        let idx = landscape.space.n_bins() / 2;
        let pitch_log2 = landscape.space.centers_log2[idx];
        landscape.consonance_field_score[idx] = 1.23;
        let perceptual = test_adaptation(landscape.space.n_bins());

        let score = adjusted_pitch_score(
            pitch_log2,
            pitch_log2,
            0.0,
            pitch_log2,
            0.0,
            1.0,
            0.0,
            1,
            &landscape,
            &perceptual,
            false,
            0.0,
            0.0,
            &[],
            &[],
        );
        assert!((score - 1.23).abs() <= 1e-6);
    }

    #[test]
    fn approximate_loo_weakens_self_locking() {
        let mut landscape = Landscape::new(Log2Space::new(110.0, 880.0, 96));
        let idx = landscape.space.n_bins() / 2;
        let current = landscape.space.centers_log2[idx];
        let nearby = landscape.space.centers_log2[idx + 1];
        landscape.consonance_field_score[idx] = 1.0;
        landscape.consonance_field_score[idx + 1] = 0.7;
        let perceptual = test_adaptation(landscape.space.n_bins());

        let current_no_loo = adjusted_pitch_score(
            current,
            current,
            0.0,
            current,
            0.0,
            1.0,
            0.0,
            1,
            &landscape,
            &perceptual,
            false,
            0.0,
            0.0,
            &[],
            &[],
        );
        let nearby_no_loo = adjusted_pitch_score(
            nearby,
            current,
            0.0,
            current,
            0.0,
            1.0,
            0.0,
            1,
            &landscape,
            &perceptual,
            false,
            0.0,
            0.0,
            &[],
            &[],
        );
        assert!(current_no_loo > nearby_no_loo);

        let current_with_loo = adjusted_pitch_score(
            current,
            current,
            0.0,
            current,
            0.0,
            1.0,
            0.0,
            1,
            &landscape,
            &perceptual,
            true,
            0.0,
            0.0,
            &[],
            &[],
        );
        let nearby_with_loo = adjusted_pitch_score(
            nearby,
            current,
            0.0,
            current,
            0.0,
            1.0,
            0.0,
            1,
            &landscape,
            &perceptual,
            true,
            0.0,
            0.0,
            &[],
            &[],
        );
        assert!(nearby_with_loo > current_with_loo);
    }

    #[test]
    fn crowding_strength_zero_keeps_behavior_with_neighbors() {
        let landscape = test_landscape(&[(330.0, 1.0)]);
        let perceptual = test_adaptation(landscape.space.n_bins());
        let features = FeaturesNow::from_occupancy_scan(&landscape.subjective_intensity);
        let mut core = PitchHillClimbPitchCore::new(120.0, 330.0f32.log2(), 0.0);
        core.set_crowding(0.0, 20.0);

        let mut rng_a = SmallRng::seed_from_u64(77);
        let mut rng_b = SmallRng::seed_from_u64(77);
        let base = core.propose_target(
            330.0f32.log2(),
            330.0f32.log2(),
            330.0,
            1.0,
            &landscape,
            &perceptual,
            &features,
            &[],
            &mut rng_a,
        );
        let with_neighbors = core.propose_target(
            330.0f32.log2(),
            330.0f32.log2(),
            330.0,
            1.0,
            &landscape,
            &perceptual,
            &features,
            &[330.0f32.log2(), 440.0f32.log2()],
            &mut rng_b,
        );

        assert!((base.target_pitch_log2 - with_neighbors.target_pitch_log2).abs() <= 1e-6);
        assert!((base.salience - with_neighbors.salience).abs() <= 1e-6);
    }

    #[test]
    fn peak_sampler_crowding_zero_ignores_neighbors() {
        let landscape = test_landscape(&[(330.0, 1.0)]);
        let perceptual = test_adaptation(landscape.space.n_bins());
        let features = FeaturesNow::from_occupancy_scan(&landscape.subjective_intensity);
        let mut core =
            PitchPeakSamplerCore::new(120.0, 330.0f32.log2(), 0.0, 700.0, 16, 0.03, 10.0, 2);
        core.set_crowding(0.0, 20.0);

        let mut rng_a = SmallRng::seed_from_u64(71);
        let mut rng_b = SmallRng::seed_from_u64(71);
        let base = core.propose_target(
            330.0f32.log2(),
            330.0f32.log2(),
            330.0,
            1.0,
            &landscape,
            &perceptual,
            &features,
            &[],
            &mut rng_a,
        );
        let with_neighbors = core.propose_target(
            330.0f32.log2(),
            330.0f32.log2(),
            330.0,
            1.0,
            &landscape,
            &perceptual,
            &features,
            &[330.0f32.log2(), 440.0f32.log2()],
            &mut rng_b,
        );

        assert!((base.target_pitch_log2 - with_neighbors.target_pitch_log2).abs() <= 1e-6);
        assert!((base.salience - with_neighbors.salience).abs() <= 1e-6);
    }

    #[test]
    fn peak_sampler_crowding_penalizes_close_neighbor_selection() {
        let landscape = test_landscape(&[(330.0, 1.0)]);
        let perceptual = test_adaptation(landscape.space.n_bins());
        let features = FeaturesNow::from_occupancy_scan(&landscape.subjective_intensity);
        let mut no_crowding =
            PitchPeakSamplerCore::new(24.0, 330.0f32.log2(), 0.0, 700.0, 16, 0.001, 0.0, 0);
        let mut with_crowding = no_crowding.clone();
        no_crowding.set_crowding(0.0, 8.0);
        with_crowding.set_crowding(3.0, 8.0);
        let neighbor = 330.0f32.log2();

        let mut rng_a = SmallRng::seed_from_u64(99);
        let mut rng_b = SmallRng::seed_from_u64(99);
        let off = no_crowding.propose_target(
            neighbor,
            neighbor,
            330.0,
            1.0,
            &landscape,
            &perceptual,
            &features,
            &[neighbor],
            &mut rng_a,
        );
        let on = with_crowding.propose_target(
            neighbor,
            neighbor,
            330.0,
            1.0,
            &landscape,
            &perceptual,
            &features,
            &[neighbor],
            &mut rng_b,
        );

        let dist_off = (off.target_pitch_log2 - neighbor).abs();
        let dist_on = (on.target_pitch_log2 - neighbor).abs();
        assert!(
            dist_on > dist_off + 1e-6,
            "crowding should bias peak-sampler away from close neighbor"
        );
    }

    #[test]
    fn hillclimb_temperature_zero_makes_no_downhill_move() {
        // On a plateau (no improvement above threshold) a zero-temperature
        // hill-climb must stay put: greedy, never downhill.
        let mut landscape = Landscape::new(Log2Space::new(220.0, 440.0, 256));
        let idx_center = landscape.space.n_bins() / 2;
        let current = landscape.space.centers_log2[idx_center];
        let width_cents = 40.0f32;
        for (idx, &bin_log2) in landscape.space.centers_log2.iter().enumerate() {
            let d_cents = (bin_log2 - current).abs() * 1200.0;
            let score = (-(d_cents * d_cents) / (2.0 * width_cents * width_cents)).exp();
            landscape.consonance_field_score[idx] = score;
            landscape.consonance_field_level[idx] = score.clamp(0.0, 1.0);
        }
        let perceptual = test_adaptation(landscape.space.n_bins());
        let features = FeaturesNow::from_occupancy_scan(&landscape.subjective_intensity);

        for seed in 0..256u64 {
            let mut core = PitchHillClimbPitchCore::new(30.0, current, 0.0);
            let mut rng = SmallRng::seed_from_u64(5_000 + seed);
            let p = core.propose_target(
                current,
                current,
                2.0f32.powf(current),
                1.0,
                &landscape,
                &perceptual,
                &features,
                &[],
                &mut rng,
            );
            assert!(
                (p.target_pitch_log2 - current).abs() <= 1e-6,
                "temperature=0 should keep greedy hill-climb behavior (no downhill move)"
            );
        }
    }

    #[test]
    fn hillclimb_temperature_accepts_downhill_moves_probabilistically() {
        let mut landscape = Landscape::new(Log2Space::new(220.0, 440.0, 256));
        let idx_center = landscape.space.n_bins() / 2;
        let current = landscape.space.centers_log2[idx_center];
        let width_cents = 40.0f32;
        for (idx, &bin_log2) in landscape.space.centers_log2.iter().enumerate() {
            let d_cents = (bin_log2 - current).abs() * 1200.0;
            let score = (-(d_cents * d_cents) / (2.0 * width_cents * width_cents)).exp();
            landscape.consonance_field_score[idx] = score;
            landscape.consonance_field_level[idx] = score.clamp(0.0, 1.0);
        }
        let perceptual = test_adaptation(landscape.space.n_bins());
        let features = FeaturesNow::from_occupancy_scan(&landscape.subjective_intensity);
        let trials = 256u64;
        let mut moved_with_temp = 0usize;
        let mut moved_without_temp = 0usize;

        for seed in 0..trials {
            let mut cold = PitchHillClimbPitchCore::new(30.0, current, 0.0);
            cold.set_temperature(0.0);
            let mut hot = PitchHillClimbPitchCore::new(30.0, current, 0.0);
            hot.set_temperature(0.2);

            let mut rng_a = SmallRng::seed_from_u64(5_000 + seed);
            let mut rng_b = SmallRng::seed_from_u64(5_000 + seed);
            let p0 = cold.propose_target(
                current,
                current,
                2.0f32.powf(current),
                1.0,
                &landscape,
                &perceptual,
                &features,
                &[],
                &mut rng_a,
            );
            let p1 = hot.propose_target(
                current,
                current,
                2.0f32.powf(current),
                1.0,
                &landscape,
                &perceptual,
                &features,
                &[],
                &mut rng_b,
            );
            if (p0.target_pitch_log2 - current).abs() > 1e-6 {
                moved_without_temp += 1;
            }
            if (p1.target_pitch_log2 - current).abs() > 1e-6 {
                moved_with_temp += 1;
            }
        }

        assert_eq!(
            moved_without_temp, 0,
            "temperature=0 should keep greedy hill-climb behavior"
        );
        assert!(
            moved_with_temp > 0 && moved_with_temp < trials as usize,
            "temperature should probabilistically accept downhill moves"
        );
    }

    #[test]
    fn hillclimb_crowding_moves_away_from_close_neighbor() {
        let mut landscape = Landscape::new(Log2Space::new(220.0, 440.0, 256));
        let center_log2 = 330.0f32.log2();
        let width_cents = 20.0f32;
        for (idx, &bin_log2) in landscape.space.centers_log2.iter().enumerate() {
            let d_cents = (bin_log2 - center_log2).abs() * 1200.0;
            let score = (-(d_cents * d_cents) / (2.0 * width_cents * width_cents)).exp();
            landscape.consonance_field_score[idx] = score;
            landscape.consonance_field_level[idx] = score.clamp(0.0, 1.0);
        }
        let perceptual = test_adaptation(landscape.space.n_bins());
        let features = FeaturesNow::from_occupancy_scan(&landscape.subjective_intensity);

        let mut without_crowding = PitchHillClimbPitchCore::new(24.0, center_log2, 0.0);
        let mut with_crowding = without_crowding.clone();
        without_crowding.set_crowding(0.0, 60.0);
        with_crowding.set_crowding(6.0, 60.0);

        let mut rng_a = SmallRng::seed_from_u64(6001);
        let mut rng_b = SmallRng::seed_from_u64(6001);
        let off = without_crowding.propose_target(
            center_log2,
            center_log2,
            330.0,
            1.0,
            &landscape,
            &perceptual,
            &features,
            &[center_log2],
            &mut rng_a,
        );
        let on = with_crowding.propose_target(
            center_log2,
            center_log2,
            330.0,
            1.0,
            &landscape,
            &perceptual,
            &features,
            &[center_log2],
            &mut rng_b,
        );
        let dist_off = (off.target_pitch_log2 - center_log2).abs();
        let dist_on = (on.target_pitch_log2 - center_log2).abs();
        assert!(
            dist_on > dist_off + 1e-6,
            "crowding should bias hill-climb away from a close neighbor"
        );
    }

    #[test]
    fn hillclimb_global_peaks_enable_distant_peak_escape() {
        let current_hz = 220.0;
        let distant_hz = 330.0;
        let landscape = test_landscape(&[(current_hz, 0.4), (distant_hz, 1.0)]);
        let perceptual = test_adaptation(landscape.space.n_bins());
        let features = FeaturesNow::from_occupancy_scan(&landscape.subjective_intensity);
        let current_log2 = current_hz.log2();

        let mut local_only = PitchHillClimbPitchCore::new(120.0, current_log2, 0.0);
        local_only.set_global_peaks(0, 0.0);
        let mut with_global = local_only.clone();
        with_global.set_global_peaks(8, 20.0);

        let mut rng_a = SmallRng::seed_from_u64(3001);
        let mut rng_b = SmallRng::seed_from_u64(3001);
        let local = local_only.propose_target(
            current_log2,
            current_log2,
            current_hz,
            1.0,
            &landscape,
            &perceptual,
            &features,
            &[],
            &mut rng_a,
        );
        let global = with_global.propose_target(
            current_log2,
            current_log2,
            current_hz,
            1.0,
            &landscape,
            &perceptual,
            &features,
            &[],
            &mut rng_b,
        );

        let local_dist_cents = (local.target_pitch_log2 - current_log2).abs() * 1200.0;
        let global_dist_cents = (global.target_pitch_log2 - current_log2).abs() * 1200.0;
        assert!(
            global_dist_cents > local_dist_cents + 80.0,
            "global peaks should enable distant move (local={local_dist_cents:.1}c, global={global_dist_cents:.1}c)"
        );
    }

    #[test]
    fn peak_sampler_converges_to_single_peak() {
        let peak_hz = 330.0;
        let landscape = test_landscape(&[(peak_hz, 1.0)]);
        let perceptual = test_adaptation(landscape.space.n_bins());
        let features = FeaturesNow::from_occupancy_scan(&landscape.subjective_intensity);
        let mut rng = SmallRng::seed_from_u64(17);

        let mut core =
            PitchPeakSamplerCore::new(120.0, 300.0f32.log2(), 0.0, 700.0, 16, 0.03, 10.0, 2);

        let mut target_log2 = 300.0f32.log2();
        for _ in 0..24 {
            let proposal = core.propose_target(
                target_log2,
                target_log2,
                2.0f32.powf(target_log2),
                1.0,
                &landscape,
                &perceptual,
                &features,
                &[],
                &mut rng,
            );
            target_log2 = proposal.target_pitch_log2;
        }

        let target_hz = 2.0f32.powf(target_log2);
        assert!(
            (target_hz - peak_hz).abs() < 25.0,
            "target {target_hz} Hz did not approach {peak_hz} Hz"
        );
    }

    fn right_peak_ratio(
        core: &mut PitchPeakSamplerCore,
        landscape: &Landscape,
        seed: u64,
        left_peak_hz: f32,
        right_peak_hz: f32,
    ) -> f32 {
        let perceptual = test_adaptation(landscape.space.n_bins());
        let features = FeaturesNow::from_occupancy_scan(&landscape.subjective_intensity);
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut right_hits = 0u32;
        let trials = 512u32;
        let start_log2 = 330.0f32.log2();

        for _ in 0..trials {
            let proposal = core.propose_target(
                start_log2,
                start_log2,
                330.0,
                1.0,
                landscape,
                &perceptual,
                &features,
                &[],
                &mut rng,
            );
            let hz = 2.0f32.powf(proposal.target_pitch_log2);
            let d_left = (hz - left_peak_hz).abs();
            let d_right = (hz - right_peak_hz).abs();
            if d_right < d_left {
                right_hits = right_hits.saturating_add(1);
            }
        }

        right_hits as f32 / trials as f32
    }

    #[test]
    fn peak_sampler_temperature_and_noise_change_bimodal_mix() {
        let left_peak_hz = 300.0;
        let right_peak_hz = 360.0;
        let landscape = test_landscape(&[(left_peak_hz, 1.0), (right_peak_hz, 0.95)]);

        let mut cold =
            PitchPeakSamplerCore::new(90.0, 330.0f32.log2(), 0.0, 900.0, 18, 0.01, 0.0, 0);
        let mut hot =
            PitchPeakSamplerCore::new(90.0, 330.0f32.log2(), 0.0, 900.0, 18, 0.25, 40.0, 3);

        let cold_right = right_peak_ratio(&mut cold, &landscape, 99, left_peak_hz, right_peak_hz);
        let hot_right = right_peak_ratio(&mut hot, &landscape, 99, left_peak_hz, right_peak_hz);

        assert!(
            hot_right > cold_right + 0.08,
            "expected higher right-peak ratio with hotter sampler (cold={cold_right:.3}, hot={hot_right:.3})"
        );
    }

    #[test]
    fn source_has_no_fixed_twelve_tet_candidate_patterns() {
        let src = include_str!("pitch_core.rs");
        let compact: String = src.chars().filter(|c| !c.is_ascii_whitespace()).collect();

        let twelfth_patterns = [
            ["3", ".0", "/", "12", ".0"].concat(),
            ["4", ".0", "/", "12", ".0"].concat(),
            ["7", ".0", "/", "12", ".0"].concat(),
        ];
        for pattern in &twelfth_patterns {
            assert!(!compact.contains(pattern));
        }

        let semitone_tokens = [
            ["-", "12", ".0"].concat(),
            ["-", "9", ".0"].concat(),
            ["-", "7", ".0"].concat(),
        ];
        assert!(!contains_ordered_tokens_with_gap(
            &compact,
            &semitone_tokens,
            3
        ));
    }

    #[test]
    fn non_tet_peak_is_reached_by_hillclimb_and_peaksampler() {
        let peak_hz = 347.0;
        let landscape = test_landscape(&[(peak_hz, 1.0)]);
        let perceptual = test_adaptation(landscape.space.n_bins());
        let features = FeaturesNow::from_occupancy_scan(&landscape.subjective_intensity);

        let mut hill_rng = SmallRng::seed_from_u64(123);
        let mut hill = PitchHillClimbPitchCore::new(120.0, 330.0f32.log2(), 0.0);
        let mut hill_target = 330.0f32.log2();
        for _ in 0..32 {
            let proposal = hill.propose_target(
                hill_target,
                hill_target,
                2.0f32.powf(hill_target),
                1.0,
                &landscape,
                &perceptual,
                &features,
                &[],
                &mut hill_rng,
            );
            hill_target = proposal.target_pitch_log2;
        }
        let hill_hz = 2.0f32.powf(hill_target);
        assert!(
            (hill_hz - peak_hz).abs() < 20.0,
            "hill climb did not converge to non-12TET peak ({hill_hz} vs {peak_hz})"
        );

        let mut sampler_rng = SmallRng::seed_from_u64(456);
        let mut sampler =
            PitchPeakSamplerCore::new(120.0, 330.0f32.log2(), 0.0, 700.0, 16, 0.03, 10.0, 2);
        let mut sampler_target = 330.0f32.log2();
        for _ in 0..32 {
            let proposal = sampler.propose_target(
                sampler_target,
                sampler_target,
                2.0f32.powf(sampler_target),
                1.0,
                &landscape,
                &perceptual,
                &features,
                &[],
                &mut sampler_rng,
            );
            sampler_target = proposal.target_pitch_log2;
        }
        let sampler_hz = 2.0f32.powf(sampler_target);
        assert!(
            (sampler_hz - peak_hz).abs() < 20.0,
            "peak sampler did not converge to non-12TET peak ({sampler_hz} vs {peak_hz})"
        );
    }

    fn contains_ordered_tokens_with_gap(text: &str, tokens: &[String], max_gap: usize) -> bool {
        if tokens.is_empty() {
            return true;
        }
        let first = tokens[0].as_str();
        let mut start_at = 0usize;
        while let Some(pos0) = text[start_at..].find(first) {
            let mut prev_end = start_at + pos0 + first.len();
            let mut ok = true;
            for token in &tokens[1..] {
                let search_to = prev_end.saturating_add(max_gap);
                let Some(window) = text.get(prev_end..search_to.min(text.len())) else {
                    ok = false;
                    break;
                };
                if let Some(rel) = window.find(token.as_str()) {
                    prev_end += rel + token.len();
                } else {
                    ok = false;
                    break;
                }
            }
            if ok {
                return true;
            }
            start_at += pos0 + 1;
        }
        false
    }

    #[test]
    fn peak_sampler_scorer_matches_adjusted_pitch_score() {
        let landscape = test_landscape(&[(330.0, 1.0), (440.0, 0.6)]);
        let perceptual = test_adaptation(landscape.space.n_bins());
        let current = 330.0f32.log2();
        let neighbor = [440.0f32.log2()];
        let salience = [1.0f32];

        for &loo in &[false, true] {
            let via_wrapper = adjusted_pitch_score(
                current,
                current,
                1.0,
                current,
                0.1,
                1.0,
                DEFAULT_MOVE_COST_COEFF,
                DEFAULT_MOVE_COST_EXP,
                &landscape,
                &perceptual,
                loo,
                0.0,
                0.0,
                &neighbor,
                &salience,
            );
            let via_direct = adjusted_pitch_score_with_loo_harmonics(
                current,
                current,
                1.0,
                current,
                0.1,
                1.0,
                DEFAULT_MOVE_COST_COEFF,
                DEFAULT_MOVE_COST_EXP,
                &landscape,
                &perceptual,
                loo,
                DEFAULT_LOO_HARMONICS,
                0.0,
                0.0,
                &neighbor,
                &salience,
            );
            assert!(
                (via_wrapper - via_direct).abs() < 1e-9,
                "adjusted_pitch_score must equal adjusted_pitch_score_with_loo_harmonics(\
                 DEFAULT_LOO_HARMONICS) when leave_self_out={loo}"
            );
        }
    }

    #[test]
    fn peak_sampler_proposal_unchanged_after_refactor() {
        let landscape = test_landscape(&[(330.0, 1.0)]);
        let perceptual = test_adaptation(landscape.space.n_bins());
        let features = FeaturesNow::from_occupancy_scan(&landscape.subjective_intensity);
        let mut core_a =
            PitchPeakSamplerCore::new(120.0, 330.0f32.log2(), 0.0, 700.0, 16, 0.03, 10.0, 2);
        core_a.set_leave_self_out(true);
        let mut core_b = core_a.clone();

        let mut rng_a = SmallRng::seed_from_u64(42);
        let mut rng_b = SmallRng::seed_from_u64(42);
        let pa = core_a.propose_target(
            330.0f32.log2(),
            330.0f32.log2(),
            330.0,
            1.0,
            &landscape,
            &perceptual,
            &features,
            &[],
            &mut rng_a,
        );
        let pb = core_b.propose_target(
            330.0f32.log2(),
            330.0f32.log2(),
            330.0,
            1.0,
            &landscape,
            &perceptual,
            &features,
            &[],
            &mut rng_b,
        );
        assert!(
            (pa.target_pitch_log2 - pb.target_pitch_log2).abs() < 1e-9,
            "same core, same seed must produce same proposal"
        );
    }

    #[test]
    fn propose_with_scorer_default_matches_trait_method() {
        let landscape = test_landscape(&[(330.0, 1.0), (440.0, 0.6)]);
        let perceptual = test_adaptation(landscape.space.n_bins());
        let features = FeaturesNow::from_occupancy_scan(&landscape.subjective_intensity);
        let core = PitchHillClimbPitchCore::new(120.0, 330.0f32.log2(), 0.0);
        let current = 330.0f32.log2();

        let mut rng_a = SmallRng::seed_from_u64(99);
        let via_scorer = core.propose_with_scorer(
            current,
            current,
            &landscape,
            &[],
            &mut rng_a,
            |pitch_log2| {
                adjusted_pitch_score_with_loo_harmonics(
                    pitch_log2,
                    current,
                    1.0,
                    current,
                    0.0,
                    1.0,
                    DEFAULT_MOVE_COST_COEFF,
                    DEFAULT_MOVE_COST_EXP,
                    &landscape,
                    &perceptual,
                    false,
                    DEFAULT_LOO_HARMONICS,
                    0.0,
                    0.0,
                    &[],
                    &[],
                )
            },
        );

        let mut rng_b = SmallRng::seed_from_u64(99);
        let mut core2 = core.clone();
        let via_trait = core2.propose_target(
            current,
            current,
            330.0,
            1.0,
            &landscape,
            &perceptual,
            &features,
            &[],
            &mut rng_b,
        );

        assert!(
            (via_scorer.target_pitch_log2 - via_trait.target_pitch_log2).abs() < 1e-9,
            "propose_with_scorer with default scorer must match propose_target"
        );
    }

    #[test]
    fn propose_with_scorer_negated_prefers_different_peak() {
        let landscape = test_landscape(&[(330.0, 1.0), (550.0, 0.3)]);
        let _perceptual = test_adaptation(landscape.space.n_bins());
        let mut core = PitchHillClimbPitchCore::new(120.0, 440.0f32.log2(), 0.0);
        core.set_global_peaks(3, 0.0);
        let current = 440.0f32.log2();

        let mut rng_pos = SmallRng::seed_from_u64(200);
        let positive = core.propose_with_scorer(
            current,
            current,
            &landscape,
            &[],
            &mut rng_pos,
            |pitch_log2| landscape.evaluate_pitch_score_log2(pitch_log2),
        );

        let mut rng_neg = SmallRng::seed_from_u64(200);
        let negated = core.propose_with_scorer(
            current,
            current,
            &landscape,
            &[],
            &mut rng_neg,
            |pitch_log2| -landscape.evaluate_pitch_score_log2(pitch_log2),
        );

        assert!(
            (positive.target_pitch_log2 - negated.target_pitch_log2).abs() > 1e-3,
            "negated scorer should produce a different proposal than positive scorer \
             (pos={}, neg={})",
            positive.target_pitch_log2,
            negated.target_pitch_log2,
        );
    }

    #[test]
    fn propose_with_scorer_non_exploratory_plateau_stays_put() {
        let landscape = test_landscape(&[(330.0, 1.0), (550.0, 0.3)]);
        let mut core = PitchHillClimbPitchCore::new(120.0, 440.0f32.log2(), 0.0);
        core.set_global_peaks(3, 0.0);
        let current = 440.0f32.log2();

        let mut rng = SmallRng::seed_from_u64(7);
        let proposal =
            core.propose_with_scorer(current, current, &landscape, &[], &mut rng, |_| 0.0);

        assert!(
            (proposal.target_pitch_log2 - current).abs() < 1e-9,
            "non-exploratory plateau should stay at current target (current={}, proposed={})",
            current,
            proposal.target_pitch_log2,
        );
    }
}
