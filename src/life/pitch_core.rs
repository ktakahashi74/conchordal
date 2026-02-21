use crate::core::harmonic_ratios::{HARMONIC_RATIOS, fold_to_octave_near, ratio_to_f32};
use crate::core::landscape::Landscape;
use crate::life::perceptual::{FeaturesNow, PerceptualContext};
use crate::life::scenario::PitchCoreConfig;
use rand::Rng;
use std::sync::OnceLock;

const DEFAULT_LOCAL_WINDOW_CENTS: f32 = 240.0;
const DEFAULT_LOCAL_TOP_K: usize = 10;
const DEFAULT_RANDOM_CANDIDATES: usize = 3;
const DEFAULT_RANDOM_SIGMA_CENTS: f32 = 30.0;
const DEFAULT_FALLBACK_RATIO_ORDER: u16 = 12;
const DEFAULT_MOVE_COST_COEFF: f32 = 0.5;
const DEFAULT_MOVE_COST_EXP: u8 = 1;
static FALLBACK_REDUCED_RATIOS: OnceLock<Vec<(u16, u16)>> = OnceLock::new();

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
        perceptual: &PerceptualContext,
        _features: &FeaturesNow,
        rng: &mut R,
    ) -> TargetProposal;

    fn propose_freqs_hz(&mut self, base_freq_hz: f32, k: usize) -> Vec<f32> {
        self.propose_freqs_hz_with_neighbors(base_freq_hz, &[], k, k.clamp(1, 8), 12.0)
    }

    fn propose_freqs_hz_with_neighbors(
        &mut self,
        base_freq_hz: f32,
        neighbor_freqs_hz: &[f32],
        max_candidates: usize,
        min_candidates: usize,
        dedupe_cents: f32,
    ) -> Vec<f32>;
}

#[derive(Debug, Clone)]
pub struct PitchHillClimbPitchCore {
    neighbor_step_log2: f32,
    tessitura_center: f32,
    tessitura_gravity: f32,
    move_cost_coeff: f32,
    move_cost_exp: u8,
    improvement_threshold: f32,
    exploration: f32,
    persistence: f32,
}

impl PitchHillClimbPitchCore {
    pub fn new(
        neighbor_step_cents: f32,
        tessitura_center: f32,
        tessitura_gravity: f32,
        improvement_threshold: f32,
        exploration: f32,
        persistence: f32,
    ) -> Self {
        let mut neighbor_step_cents = neighbor_step_cents;
        if !neighbor_step_cents.is_finite() {
            neighbor_step_cents = 0.0;
        }
        neighbor_step_cents = neighbor_step_cents.max(0.0);
        Self {
            neighbor_step_log2: cents_to_log2(neighbor_step_cents),
            tessitura_center,
            tessitura_gravity,
            move_cost_coeff: DEFAULT_MOVE_COST_COEFF,
            move_cost_exp: DEFAULT_MOVE_COST_EXP,
            improvement_threshold,
            exploration: exploration.clamp(0.0, 1.0),
            persistence: persistence.clamp(0.0, 1.0),
        }
    }

    pub fn set_exploration(&mut self, value: f32) {
        self.exploration = value.clamp(0.0, 1.0);
    }

    pub fn set_persistence(&mut self, value: f32) {
        self.persistence = value.clamp(0.0, 1.0);
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
}

impl PitchCore for PitchHillClimbPitchCore {
    fn propose_target<R: Rng + ?Sized>(
        &mut self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        _current_freq_hz: f32,
        integration_window: f32,
        landscape: &Landscape,
        perceptual: &PerceptualContext,
        _features: &FeaturesNow,
        rng: &mut R,
    ) -> TargetProposal {
        let (fmin, fmax) = landscape.freq_bounds_log2();
        let current_target_log2 = current_target_log2.clamp(fmin, fmax);

        let adaptive_window_cents =
            (self.neighbor_step_log2.abs() * 1200.0 * 4.0).max(DEFAULT_LOCAL_WINDOW_CENTS);
        let window_bins = window_bins_from_cents(landscape, adaptive_window_cents);
        let mut candidates = vec![
            current_target_log2,
            current_target_log2 + self.neighbor_step_log2,
            current_target_log2 - self.neighbor_step_log2,
        ];
        candidates.extend(top_local_candidates(
            landscape,
            current_target_log2,
            window_bins,
            DEFAULT_LOCAL_TOP_K,
            current_pitch_log2,
            integration_window,
            self.tessitura_center,
            self.tessitura_gravity,
            self.move_cost_coeff,
            self.move_cost_exp,
            perceptual,
        ));
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

        let adjusted_score = |pitch_log2: f32| -> f32 {
            adjusted_pitch_score(
                pitch_log2,
                current_pitch_log2,
                integration_window,
                self.tessitura_center,
                self.tessitura_gravity,
                self.move_cost_coeff,
                self.move_cost_exp,
                landscape,
                perceptual,
            )
        };

        let mut best_pitch = current_target_log2;
        let mut best_score = f32::MIN;
        for p in candidates {
            let clamped = p.clamp(fmin, fmax);
            let adjusted = adjusted_score(clamped);
            if adjusted > best_score {
                best_score = adjusted;
                best_pitch = clamped;
            }
        }

        let current_adjusted = adjusted_score(current_target_log2);
        let improvement = best_score - current_adjusted;
        let mut target_pitch_log2 = current_target_log2;

        if improvement > self.improvement_threshold {
            target_pitch_log2 = best_pitch;
        } else {
            let satisfaction = ((current_adjusted + 1.0) * 0.5).clamp(0.0, 1.0);
            let mut stay_prob = self.persistence.clamp(0.0, 1.0) * satisfaction;
            stay_prob = stay_prob.clamp(0.0, 1.0);
            let exploration = self.exploration.clamp(0.0, 1.0);
            stay_prob = (stay_prob * (1.0 - exploration)).clamp(0.0, 1.0);
            if rng.random_range(0.0..1.0) > stay_prob {
                target_pitch_log2 = best_pitch;
            }
        }

        TargetProposal {
            target_pitch_log2,
            salience: (improvement / 0.2).clamp(0.0, 1.0),
        }
    }

    fn propose_freqs_hz_with_neighbors(
        &mut self,
        base_freq_hz: f32,
        neighbor_freqs_hz: &[f32],
        max_candidates: usize,
        min_candidates: usize,
        dedupe_cents: f32,
    ) -> Vec<f32> {
        propose_ratio_candidates(
            base_freq_hz,
            neighbor_freqs_hz,
            max_candidates,
            min_candidates,
            dedupe_cents,
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
    exploration: f32,
    persistence: f32,
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
        exploration: f32,
        persistence: f32,
    ) -> Self {
        Self {
            neighbor_step_log2: cents_to_log2(neighbor_step_cents.max(0.0)),
            window_cents: window_cents.max(1.0),
            top_k: top_k.max(1),
            temperature: temperature.max(1e-3),
            sigma_cents: sigma_cents.max(0.0),
            random_candidates,
            tessitura_center,
            tessitura_gravity,
            exploration: exploration.clamp(0.0, 1.0),
            persistence: persistence.clamp(0.0, 1.0),
        }
    }

    pub fn set_exploration(&mut self, value: f32) {
        self.exploration = value.clamp(0.0, 1.0);
    }

    pub fn set_persistence(&mut self, value: f32) {
        self.persistence = value.clamp(0.0, 1.0);
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
}

impl PitchCore for PitchPeakSamplerCore {
    fn propose_target<R: Rng + ?Sized>(
        &mut self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        _current_freq_hz: f32,
        integration_window: f32,
        landscape: &Landscape,
        perceptual: &PerceptualContext,
        _features: &FeaturesNow,
        rng: &mut R,
    ) -> TargetProposal {
        let (fmin, fmax) = landscape.freq_bounds_log2();
        let current_target_log2 = current_target_log2.clamp(fmin, fmax);
        let window_bins = window_bins_from_cents(landscape, self.window_cents);

        let mut candidates = vec![
            current_target_log2,
            current_target_log2 + self.neighbor_step_log2,
            current_target_log2 - self.neighbor_step_log2,
        ];
        candidates.extend(top_local_candidates(
            landscape,
            current_target_log2,
            window_bins,
            self.top_k,
            current_pitch_log2,
            integration_window,
            self.tessitura_center,
            self.tessitura_gravity,
            DEFAULT_MOVE_COST_COEFF,
            DEFAULT_MOVE_COST_EXP,
            perceptual,
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

        let mut scored = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let pitch = candidate.clamp(fmin, fmax);
            let score = adjusted_pitch_score(
                pitch,
                current_pitch_log2,
                integration_window,
                self.tessitura_center,
                self.tessitura_gravity,
                DEFAULT_MOVE_COST_COEFF,
                DEFAULT_MOVE_COST_EXP,
                landscape,
                perceptual,
            );
            if score.is_finite() {
                scored.push((pitch, score));
            }
        }
        if scored.is_empty() {
            return TargetProposal {
                target_pitch_log2: current_target_log2,
                salience: 0.0,
            };
        }

        let current_adjusted = adjusted_pitch_score(
            current_target_log2,
            current_pitch_log2,
            integration_window,
            self.tessitura_center,
            self.tessitura_gravity,
            DEFAULT_MOVE_COST_COEFF,
            DEFAULT_MOVE_COST_EXP,
            landscape,
            perceptual,
        );
        let mut best_score = current_adjusted;
        for &(_, score) in &scored {
            if score > best_score {
                best_score = score;
            }
        }
        let improvement = best_score - current_adjusted;

        let satisfaction = ((current_adjusted + 1.0) * 0.5).clamp(0.0, 1.0);
        let mut stay_prob = self.persistence * satisfaction;
        stay_prob = (stay_prob * (1.0 - self.exploration)).clamp(0.0, 1.0);

        let target_pitch_log2 = if rng.random_range(0.0..1.0) < stay_prob {
            current_target_log2
        } else {
            let effective_temperature = self.temperature * (1.0 + 2.0 * self.exploration);
            sample_softmax_candidate(&scored, effective_temperature, rng).clamp(fmin, fmax)
        };

        TargetProposal {
            target_pitch_log2,
            salience: (improvement / 0.2).clamp(0.0, 1.0),
        }
    }

    fn propose_freqs_hz_with_neighbors(
        &mut self,
        base_freq_hz: f32,
        neighbor_freqs_hz: &[f32],
        max_candidates: usize,
        min_candidates: usize,
        dedupe_cents: f32,
    ) -> Vec<f32> {
        propose_ratio_candidates(
            base_freq_hz,
            neighbor_freqs_hz,
            max_candidates,
            min_candidates,
            dedupe_cents,
        )
    }
}

fn propose_ratio_candidates(
    base_freq_hz: f32,
    neighbor_freqs_hz: &[f32],
    max_candidates: usize,
    min_candidates: usize,
    dedupe_cents: f32,
) -> Vec<f32> {
    if max_candidates == 0 || !base_freq_hz.is_finite() || base_freq_hz <= 0.0 {
        return Vec::new();
    }
    let mut candidates = Vec::new();
    let lo = base_freq_hz * 0.5;
    let hi = base_freq_hz * 2.0;

    if !neighbor_freqs_hz.is_empty() {
        for &neighbor in neighbor_freqs_hz {
            if !neighbor.is_finite() || neighbor <= 0.0 {
                continue;
            }
            for &ratio in HARMONIC_RATIOS {
                let r = ratio_to_f32(ratio);
                let forward = fold_to_octave_near(neighbor * r, base_freq_hz, lo, hi);
                let inverse = fold_to_octave_near(neighbor / r, base_freq_hz, lo, hi);
                candidates.push(forward);
                candidates.push(inverse);
            }
        }
    }

    candidates = dedupe_by_cents(candidates, dedupe_cents);
    candidates.sort_by(|a, b| cmp_by_base(*a, *b, base_freq_hz));

    if candidates.len() < min_candidates {
        let fill_limit = candidates
            .len()
            .saturating_add(min_candidates.saturating_mul(8).max(32));
        push_ratio_family_candidates(&mut candidates, base_freq_hz, lo, hi, fill_limit);
        candidates = dedupe_by_cents(candidates, dedupe_cents);
        candidates.sort_by(|a, b| cmp_by_base(*a, *b, base_freq_hz));
    }

    if candidates.len() < min_candidates {
        push_log_grid_candidates(&mut candidates, base_freq_hz, lo, hi, min_candidates.max(4));
        candidates = dedupe_by_cents(candidates, dedupe_cents);
        candidates.sort_by(|a, b| cmp_by_base(*a, *b, base_freq_hz));
    }

    candidates
        .into_iter()
        .filter(|f| f.is_finite() && *f > 0.0 && *f >= 20.0 && *f <= 20_000.0)
        .take(max_candidates)
        .collect()
}

fn push_ratio_family_candidates(
    candidates: &mut Vec<f32>,
    base_freq_hz: f32,
    lo: f32,
    hi: f32,
    fill_limit: usize,
) {
    for &(num, den) in fallback_reduced_ratios() {
        if candidates.len() >= fill_limit {
            break;
        }
        let ratio = num as f32 / den as f32;
        let freq = fold_to_octave_near(base_freq_hz * ratio, base_freq_hz, lo, hi);
        candidates.push(freq);
    }
}

fn fallback_reduced_ratios() -> &'static [(u16, u16)] {
    FALLBACK_REDUCED_RATIOS
        .get_or_init(|| {
            let max_order = DEFAULT_FALLBACK_RATIO_ORDER;
            let mut out =
                Vec::with_capacity((max_order as usize).saturating_mul(max_order as usize));
            for num in 1..=max_order {
                for den in 1..=max_order {
                    if gcd_u16(num, den) == 1 {
                        out.push((num, den));
                    }
                }
            }
            out
        })
        .as_slice()
}

fn push_log_grid_candidates(
    candidates: &mut Vec<f32>,
    base_freq_hz: f32,
    lo: f32,
    hi: f32,
    count: usize,
) {
    if count == 0 {
        return;
    }
    if count == 1 {
        candidates.push(base_freq_hz);
        return;
    }
    for i in 0..count {
        let t = i as f32 / (count.saturating_sub(1)) as f32;
        let log_offset = (2.0 * t) - 1.0;
        let freq = base_freq_hz * 2.0f32.powf(log_offset);
        candidates.push(fold_to_octave_near(freq, base_freq_hz, lo, hi));
    }
}

fn gcd_u16(mut a: u16, mut b: u16) -> u16 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a.max(1)
}

fn cents_to_log2(cents: f32) -> f32 {
    cents / 1200.0
}

fn window_bins_from_cents(landscape: &Landscape, cents: f32) -> usize {
    let step = landscape.space.step().max(1e-6);
    let span_log2 = cents_to_log2(cents.max(0.0));
    (span_log2 / step).ceil() as usize
}

#[allow(clippy::too_many_arguments)]
fn adjusted_pitch_score(
    pitch_log2: f32,
    current_pitch_log2: f32,
    integration_window: f32,
    tessitura_center: f32,
    tessitura_gravity: f32,
    move_cost_coeff: f32,
    move_cost_exp: u8,
    landscape: &Landscape,
    perceptual: &PerceptualContext,
) -> f32 {
    let (fmin, fmax) = landscape.freq_bounds_log2();
    let clamped = pitch_log2.clamp(fmin, fmax);
    let score = landscape.evaluate_pitch_score_log2(clamped);
    let distance_oct = (clamped - current_pitch_log2).abs();
    let dist_cost = if move_cost_exp == 2 {
        distance_oct * distance_oct
    } else {
        distance_oct
    };
    let penalty = dist_cost * integration_window * move_cost_coeff.max(0.0);
    let dist = clamped - tessitura_center;
    let gravity_penalty = dist * dist * tessitura_gravity;
    let base = score - penalty - gravity_penalty;
    let idx = landscape.space.index_of_log2(clamped).unwrap_or(0);
    base + perceptual.score_adjustment(idx)
}

#[allow(clippy::too_many_arguments)]
fn top_local_candidates(
    landscape: &Landscape,
    current_target_log2: f32,
    window_bins: usize,
    top_k: usize,
    current_pitch_log2: f32,
    integration_window: f32,
    tessitura_center: f32,
    tessitura_gravity: f32,
    move_cost_coeff: f32,
    move_cost_exp: u8,
    perceptual: &PerceptualContext,
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
        let score = adjusted_pitch_score(
            pitch_log2,
            current_pitch_log2,
            integration_window,
            tessitura_center,
            tessitura_gravity,
            move_cost_coeff,
            move_cost_exp,
            landscape,
            perceptual,
        );
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

fn dedupe_by_cents(mut freqs: Vec<f32>, dedupe_cents: f32) -> Vec<f32> {
    if freqs.is_empty() {
        return freqs;
    }
    freqs.retain(|f| f.is_finite() && *f > 0.0);
    freqs.sort_by(|a, b| a.total_cmp(b));
    let mut out = Vec::with_capacity(freqs.len());
    let mut last: Option<f32> = None;
    for f in freqs {
        if let Some(prev) = last {
            let cents = 1200.0f32 * (f / prev).log2().abs();
            if cents < dedupe_cents {
                continue;
            }
        }
        last = Some(f);
        out.push(f);
    }
    out
}

fn cmp_by_base(a: f32, b: f32, base: f32) -> std::cmp::Ordering {
    let base = if base.is_finite() && base > 0.0 {
        base
    } else {
        1.0
    };
    let da = (a / base).log2().abs();
    let db = (b / base).log2().abs();
    da.total_cmp(&db).then_with(|| a.total_cmp(&b))
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
        perceptual: &PerceptualContext,
        features: &FeaturesNow,
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
                rng,
            ),
        }
    }

    fn propose_freqs_hz_with_neighbors(
        &mut self,
        base_freq_hz: f32,
        neighbor_freqs_hz: &[f32],
        max_candidates: usize,
        min_candidates: usize,
        dedupe_cents: f32,
    ) -> Vec<f32> {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.propose_freqs_hz_with_neighbors(
                base_freq_hz,
                neighbor_freqs_hz,
                max_candidates,
                min_candidates,
                dedupe_cents,
            ),
            AnyPitchCore::PitchPeakSampler(core) => core.propose_freqs_hz_with_neighbors(
                base_freq_hz,
                neighbor_freqs_hz,
                max_candidates,
                min_candidates,
                dedupe_cents,
            ),
        }
    }
}

impl AnyPitchCore {
    pub fn from_config<R: Rng + ?Sized>(
        config: &PitchCoreConfig,
        initial_pitch_log2: f32,
        _rng: &mut R,
    ) -> Self {
        match config {
            PitchCoreConfig::PitchHillClimb {
                neighbor_step_cents,
                tessitura_gravity,
                move_cost_coeff,
                move_cost_exp,
                improvement_threshold,
                exploration,
                persistence,
            } => {
                let neighbor_step_cents = neighbor_step_cents.unwrap_or(200.0);
                let tessitura_gravity = tessitura_gravity.unwrap_or(0.1);
                let move_cost_coeff = move_cost_coeff.unwrap_or(DEFAULT_MOVE_COST_COEFF);
                let move_cost_exp = move_cost_exp.unwrap_or(DEFAULT_MOVE_COST_EXP);
                let improvement_threshold = improvement_threshold.unwrap_or(0.1);
                let exploration = exploration.unwrap_or(0.0);
                let persistence = persistence.unwrap_or(0.5);
                let mut core = PitchHillClimbPitchCore::new(
                    neighbor_step_cents,
                    initial_pitch_log2,
                    tessitura_gravity,
                    improvement_threshold,
                    exploration,
                    persistence,
                );
                core.set_move_cost_coeff(move_cost_coeff);
                core.set_move_cost_exp(move_cost_exp);
                AnyPitchCore::PitchHillClimb(core)
            }
            PitchCoreConfig::PitchPeakSampler {
                neighbor_step_cents,
                window_cents,
                top_k,
                temperature,
                sigma_cents,
                random_candidates,
                tessitura_gravity,
                exploration,
                persistence,
            } => {
                let neighbor_step_cents = neighbor_step_cents.unwrap_or(160.0);
                let window_cents = window_cents.unwrap_or(DEFAULT_LOCAL_WINDOW_CENTS);
                let top_k = top_k.unwrap_or(DEFAULT_LOCAL_TOP_K);
                let temperature = temperature.unwrap_or(0.08);
                let sigma_cents = sigma_cents.unwrap_or(DEFAULT_RANDOM_SIGMA_CENTS);
                let random_candidates = random_candidates.unwrap_or(DEFAULT_RANDOM_CANDIDATES);
                let tessitura_gravity = tessitura_gravity.unwrap_or(0.1);
                let exploration = exploration.unwrap_or(0.2);
                let persistence = persistence.unwrap_or(0.35);
                AnyPitchCore::PitchPeakSampler(PitchPeakSamplerCore::new(
                    neighbor_step_cents,
                    initial_pitch_log2,
                    tessitura_gravity,
                    window_cents,
                    top_k,
                    temperature,
                    sigma_cents,
                    random_candidates,
                    exploration,
                    persistence,
                ))
            }
        }
    }

    pub fn set_exploration(&mut self, value: f32) {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.set_exploration(value),
            AnyPitchCore::PitchPeakSampler(core) => core.set_exploration(value),
        }
    }

    pub fn set_neighbor_step_cents(&mut self, value: f32) {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.set_neighbor_step_cents(value),
            AnyPitchCore::PitchPeakSampler(core) => core.set_neighbor_step_cents(value),
        }
    }

    pub fn set_persistence(&mut self, value: f32) {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.set_persistence(value),
            AnyPitchCore::PitchPeakSampler(core) => core.set_persistence(value),
        }
    }

    pub fn set_tessitura_center(&mut self, value: f32) {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.set_tessitura_center(value),
            AnyPitchCore::PitchPeakSampler(core) => core.set_tessitura_center(value),
        }
    }

    pub fn set_tessitura_gravity(&mut self, value: f32) {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.set_tessitura_gravity(value),
            AnyPitchCore::PitchPeakSampler(core) => core.set_tessitura_gravity(value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::log2space::Log2Space;
    use crate::life::perceptual::PerceptualConfig;
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
            landscape.consonance_score[idx] = score;
            landscape.consonance_level01[idx] = score.clamp(0.0, 1.0);
        }
        landscape
    }

    fn test_perceptual(n_bins: usize) -> PerceptualContext {
        PerceptualContext::from_config(&PerceptualConfig::default(), n_bins)
    }

    #[test]
    fn peak_sampler_converges_to_single_peak() {
        let peak_hz = 330.0;
        let landscape = test_landscape(&[(peak_hz, 1.0)]);
        let perceptual = test_perceptual(landscape.space.n_bins());
        let features = FeaturesNow::from_subjective_intensity(&landscape.subjective_intensity);
        let mut rng = SmallRng::seed_from_u64(17);

        let mut core = PitchPeakSamplerCore::new(
            120.0,
            300.0f32.log2(),
            0.0,
            700.0,
            16,
            0.03,
            10.0,
            2,
            1.0,
            0.0,
        );

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
        let perceptual = test_perceptual(landscape.space.n_bins());
        let features = FeaturesNow::from_subjective_intensity(&landscape.subjective_intensity);
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

        let mut cold = PitchPeakSamplerCore::new(
            90.0,
            330.0f32.log2(),
            0.0,
            900.0,
            18,
            0.01,
            0.0,
            0,
            1.0,
            0.0,
        );
        let mut hot = PitchPeakSamplerCore::new(
            90.0,
            330.0f32.log2(),
            0.0,
            900.0,
            18,
            0.25,
            40.0,
            3,
            1.0,
            0.0,
        );

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
        let perceptual = test_perceptual(landscape.space.n_bins());
        let features = FeaturesNow::from_subjective_intensity(&landscape.subjective_intensity);

        let mut hill_rng = SmallRng::seed_from_u64(123);
        let mut hill = PitchHillClimbPitchCore::new(120.0, 330.0f32.log2(), 0.0, 0.0, 0.0, 0.0);
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
        let mut sampler = PitchPeakSamplerCore::new(
            120.0,
            330.0f32.log2(),
            0.0,
            700.0,
            16,
            0.03,
            10.0,
            2,
            1.0,
            0.0,
        );
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
}
