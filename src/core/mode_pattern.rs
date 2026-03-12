use crate::core::erb::hz_to_erb;
use crate::core::landscape::LandscapeFrame;
use crate::core::log2space::Log2Space;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use tracing::warn;

pub(crate) const DEFAULT_MODE_COUNT: usize = 16;
const DEFAULT_MIN_MUL: f32 = 1.0;
const DEFAULT_MAX_MUL: f32 = 4.0;
const DEFAULT_MIN_DIST_ERB: f32 = 1.0;

const MODAL_TABLE_UNIFORM_ALUMINUM_BAR: &[f32] = &[1.0, 2.756, 5.404, 8.933, 13.34, 18.64];
const MODAL_TABLE_UNIFORM_WOODEN_BAR: &[f32] = &[1.0, 2.572, 4.644, 6.984, 9.723, 12.0];
const MODAL_TABLE_XYLOPHONE: &[f32] = &[1.0, 3.932, 9.538, 16.688, 24.566, 31.147];
const MODAL_TABLE_VIBRAPHONE_1: &[f32] = &[1.0, 3.984, 10.668, 17.979, 23.679];
const MODAL_TABLE_VIBRAPHONE_2: &[f32] = &[1.0, 3.997, 9.469, 15.566, 20.863];
const MODAL_TABLE_WINE_GLASS: &[f32] = &[1.0, 2.32, 4.25, 6.63, 9.38];

#[derive(Clone, Debug, PartialEq)]
pub enum ModePatternKind {
    Harmonic,
    Odd,
    PowerLaw { beta: f32 },
    StiffString { stiffness: f32 },
    Custom { ratios: Vec<f32> },
    ModalTable { name: String, ratios: Vec<f32> },
    LandscapeDensity,
    LandscapePeaks,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModePattern {
    pub kind: ModePatternKind,
    pub count: usize,
    pub min_mul: f32,
    pub max_mul: f32,
    pub min_dist_erb: f32,
    pub gamma: f32,
    pub jitter_cents: Option<f32>,
    pub seed: Option<u64>,
}

impl ModePattern {
    pub fn harmonic_modes() -> Self {
        Self::new(ModePatternKind::Harmonic)
    }

    pub fn odd_modes() -> Self {
        Self::new(ModePatternKind::Odd)
    }

    pub fn power_modes(beta: f32) -> Self {
        Self::new(ModePatternKind::PowerLaw { beta })
    }

    pub fn stiff_string_modes(stiffness: f32) -> Self {
        Self::new(ModePatternKind::StiffString { stiffness })
    }

    pub fn custom_modes(ratios: Vec<f32>) -> Self {
        let ratios = sanitize_ratio_list(ratios);
        Self::from_ratio_pattern_defaults(
            ModePatternKind::Custom {
                ratios: ratios.clone(),
            },
            &ratios,
        )
    }

    pub fn modal_table(name: &str) -> Option<Self> {
        let key = name.trim().to_ascii_lowercase();
        let ratios = modal_table_ratios(&key)?;
        let ratios = ratios.to_vec();
        Some(Self::from_ratio_pattern_defaults(
            ModePatternKind::ModalTable {
                name: key,
                ratios: ratios.clone(),
            },
            &ratios,
        ))
    }

    pub fn landscape_density_modes() -> Self {
        Self::new(ModePatternKind::LandscapeDensity)
    }

    pub fn landscape_peaks_modes() -> Self {
        Self::new(ModePatternKind::LandscapePeaks)
    }

    fn new(kind: ModePatternKind) -> Self {
        Self {
            kind,
            count: DEFAULT_MODE_COUNT,
            min_mul: DEFAULT_MIN_MUL,
            max_mul: DEFAULT_MAX_MUL,
            min_dist_erb: DEFAULT_MIN_DIST_ERB,
            gamma: 1.0,
            jitter_cents: None,
            seed: None,
        }
    }

    fn from_ratio_pattern_defaults(kind: ModePatternKind, ratios: &[f32]) -> Self {
        let mut pattern = Self::new(kind);
        let ratios = sanitize_ratio_list(ratios.to_vec());
        if ratios.is_empty() {
            return pattern;
        }

        let max_ratio = ratios
            .iter()
            .copied()
            .fold(1.0f32, |acc, ratio| acc.max(ratio));
        let min_ratio = ratios
            .iter()
            .copied()
            .fold(f32::INFINITY, |acc, ratio| acc.min(ratio));
        pattern.count = ratios.len().max(1);
        pattern.min_mul = if min_ratio.is_finite() {
            min_ratio.max(1.0e-6)
        } else {
            DEFAULT_MIN_MUL
        };
        pattern.max_mul = max_ratio.max(pattern.min_mul);
        pattern
    }

    pub fn with_count(mut self, count: usize) -> Self {
        self.count = count.max(1);
        self
    }

    pub fn with_range(mut self, min_mul: f32, max_mul: f32) -> Self {
        self.min_mul = min_mul;
        self.max_mul = max_mul;
        self
    }

    pub fn with_min_dist_erb(mut self, min_dist_erb: f32) -> Self {
        self.min_dist_erb = min_dist_erb;
        self
    }

    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    pub fn with_jitter_cents(mut self, jitter_cents: f32) -> Self {
        self.jitter_cents = Some(jitter_cents);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub(crate) fn supports_range(&self) -> bool {
        matches!(
            &self.kind,
            ModePatternKind::LandscapeDensity | ModePatternKind::LandscapePeaks
        )
    }

    pub(crate) fn supports_min_dist_erb(&self) -> bool {
        self.supports_range()
    }

    pub(crate) fn supports_gamma(&self) -> bool {
        matches!(&self.kind, ModePatternKind::LandscapeDensity)
    }

    pub fn eval(
        &self,
        base_hz: f32,
        space: &Log2Space,
        landscape: Option<&LandscapeFrame>,
        rng: &mut SmallRng,
    ) -> Vec<f32> {
        if let Some(seed) = self.seed {
            let base_mix = (base_hz.max(1.0).to_bits() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut local_rng = SmallRng::seed_from_u64(seed ^ base_mix);
            self.eval_with_rng(base_hz, space, landscape, &mut local_rng)
        } else {
            self.eval_with_rng(base_hz, space, landscape, rng)
        }
    }

    fn eval_with_rng<R: Rng + ?Sized>(
        &self,
        base_hz: f32,
        space: &Log2Space,
        landscape: Option<&LandscapeFrame>,
        rng: &mut R,
    ) -> Vec<f32> {
        if !base_hz.is_finite() || base_hz <= 0.0 {
            return vec![1.0];
        }

        let count = self.count.max(1);

        let mut ratios = match &self.kind {
            ModePatternKind::Harmonic => self.eval_harmonic_like(count, 1.0),
            ModePatternKind::Odd => self.eval_harmonic_like(count, 2.0),
            ModePatternKind::PowerLaw { beta } => {
                let beta = sanitize_positive_finite(*beta, 1.0).max(1.0e-3);
                self.eval_formula(count, |k| k.powf(beta))
            }
            ModePatternKind::StiffString { stiffness } => {
                let stiffness = sanitize_nonnegative_finite(*stiffness);
                self.eval_formula(count, |k| k * (1.0 + stiffness * k * k).sqrt())
            }
            ModePatternKind::Custom { ratios } | ModePatternKind::ModalTable { ratios, .. } => {
                ratios.clone()
            }
            ModePatternKind::LandscapeDensity => {
                let (min_mul, max_mul) = sanitized_range(self.min_mul, self.max_mul);
                self.eval_landscape_density(base_hz, space, landscape, min_mul, max_mul)
            }
            ModePatternKind::LandscapePeaks => {
                let (min_mul, max_mul) = sanitized_range(self.min_mul, self.max_mul);
                self.eval_landscape_peaks(base_hz, space, landscape, min_mul, max_mul)
            }
        };

        ratios.retain(|r| r.is_finite() && *r > 0.0);

        let mut jitter_filter = None;
        if let Some(cents) = self.jitter_cents {
            let cents = sanitize_nonnegative_finite(cents);
            if cents > 0.0 {
                if self.supports_range() {
                    let (min_mul, max_mul) = sanitized_range(self.min_mul, self.max_mul);
                    let jitter_mul = 2.0f32.powf(cents / 1200.0);
                    if jitter_mul.is_finite() && jitter_mul > 0.0 {
                        jitter_filter = Some((min_mul / jitter_mul, max_mul * jitter_mul));
                    } else {
                        jitter_filter = Some((0.0, f32::INFINITY));
                    }
                }
                for ratio in &mut ratios {
                    let detune = rng.random_range(-cents..=cents);
                    let mul = 2.0f32.powf(detune / 1200.0);
                    *ratio *= mul;
                }
            }
        }

        ratios.retain(|r| r.is_finite() && *r > 0.0);
        if let Some((jitter_filter_min, jitter_filter_max)) = jitter_filter {
            ratios.retain(|r| *r >= jitter_filter_min && *r <= jitter_filter_max);
        }
        if ratios.is_empty() {
            return vec![1.0];
        }
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ratios.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-4);
        if ratios.len() > count {
            ratios.truncate(count);
        }
        if ratios.is_empty() {
            return vec![1.0];
        }
        if ratios.len() < count {
            warn!(
                "mode pattern {} produced {} ratios below requested count {}; source data or search constraints kept fewer modes",
                self.kind_label(),
                ratios.len(),
                count
            );
        }
        ratios
    }

    fn kind_label(&self) -> &'static str {
        match &self.kind {
            ModePatternKind::Harmonic => "harmonic_modes",
            ModePatternKind::Odd => "odd_modes",
            ModePatternKind::PowerLaw { .. } => "power_modes",
            ModePatternKind::StiffString { .. } => "stiff_string_modes",
            ModePatternKind::Custom { .. } => "custom_modes",
            ModePatternKind::ModalTable { .. } => "modal_table",
            ModePatternKind::LandscapeDensity => "landscape_density_modes",
            ModePatternKind::LandscapePeaks => "landscape_peaks_modes",
        }
    }

    fn eval_harmonic_like(&self, count: usize, step: f32) -> Vec<f32> {
        let mut ratios = Vec::with_capacity(count);
        let mut k = 1usize;
        while ratios.len() < count {
            let ratio = if step <= 1.0 {
                k as f32
            } else {
                (2 * k - 1) as f32
            };
            ratios.push(ratio);
            k = k.saturating_add(1);
        }
        if ratios.is_empty() {
            ratios.push(1.0);
        }
        ratios
    }

    fn eval_harmonic_like_in_range(
        &self,
        min_mul: f32,
        max_mul: f32,
        count: usize,
        step: f32,
    ) -> Vec<f32> {
        let mut ratios = Vec::with_capacity(count);
        let mut k = 1usize;
        let mut guard = 0usize;
        while ratios.len() < count && guard < count.saturating_mul(64) {
            let ratio = if step <= 1.0 {
                k as f32
            } else {
                (2 * k - 1) as f32
            };
            if ratio >= min_mul && ratio <= max_mul {
                ratios.push(ratio);
            }
            k = k.saturating_add(1);
            guard = guard.saturating_add(1);
        }
        if ratios.is_empty() {
            ratios.push(min_mul.max(1.0));
        }
        ratios
    }

    fn eval_formula<F>(&self, count: usize, mut f: F) -> Vec<f32>
    where
        F: FnMut(f32) -> f32,
    {
        let mut ratios = Vec::with_capacity(count);
        let mut k = 1usize;
        let mut guard = 0usize;
        while ratios.len() < count && guard < count.saturating_mul(128) {
            let raw = f(k as f32);
            if raw.is_finite() && raw > 0.0 {
                ratios.push(raw);
            }
            k = k.saturating_add(1);
            guard = guard.saturating_add(1);
        }
        if ratios.is_empty() {
            ratios.push(1.0);
        }
        ratios
    }

    fn eval_landscape_density(
        &self,
        base_hz: f32,
        _space: &Log2Space,
        landscape: Option<&LandscapeFrame>,
        min_mul: f32,
        max_mul: f32,
    ) -> Vec<f32> {
        let Some(landscape) = landscape else {
            return self.eval_harmonic_like_in_range(min_mul, max_mul, self.count.max(1), 1.0);
        };
        let eval_space = &landscape.space;
        let Some((idx_min, idx_max)) =
            freq_range_to_bins(eval_space, base_hz * min_mul, base_hz * max_mul)
        else {
            return self.eval_harmonic_like_in_range(min_mul, max_mul, self.count.max(1), 1.0);
        };
        if landscape.consonance_density_mass.len() != eval_space.n_bins() {
            return self.eval_harmonic_like_in_range(min_mul, max_mul, self.count.max(1), 1.0);
        }
        let gamma = sanitize_positive_finite(self.gamma, 1.0);
        let mut weights = Vec::with_capacity(idx_max - idx_min + 1);
        for idx in idx_min..=idx_max {
            let raw = sanitize_nonnegative_finite(landscape.consonance_density_mass[idx]);
            weights.push(raw.powf(gamma));
        }

        let mut selected: Vec<f32> = Vec::with_capacity(self.count);
        let min_dist_erb = sanitize_nonnegative_finite(self.min_dist_erb);
        for _ in 0..self.count.max(1) {
            let mut best_offset = None;
            let mut best_weight = f32::NEG_INFINITY;
            for (offset, &weight) in weights.iter().enumerate() {
                if weight > best_weight {
                    best_weight = weight;
                    best_offset = Some(offset);
                }
            }
            let Some(offset) = best_offset else { break };
            if !best_weight.is_finite() || best_weight <= 0.0 {
                break;
            }
            let idx = idx_min + offset;
            let freq = eval_space.freq_of_index(idx);
            selected.push(freq / base_hz);
            suppress_neighbor_weights(
                eval_space,
                idx_min,
                idx_max,
                idx,
                min_dist_erb,
                &mut weights,
            );
        }

        if selected.is_empty() {
            self.eval_harmonic_like_in_range(min_mul, max_mul, self.count.max(1), 1.0)
        } else {
            selected
        }
    }

    fn eval_landscape_peaks(
        &self,
        base_hz: f32,
        _space: &Log2Space,
        landscape: Option<&LandscapeFrame>,
        min_mul: f32,
        max_mul: f32,
    ) -> Vec<f32> {
        let Some(landscape) = landscape else {
            return self.eval_harmonic_like_in_range(min_mul, max_mul, self.count.max(1), 1.0);
        };
        let eval_space = &landscape.space;
        let Some((idx_min, idx_max)) =
            freq_range_to_bins(eval_space, base_hz * min_mul, base_hz * max_mul)
        else {
            return self.eval_harmonic_like_in_range(min_mul, max_mul, self.count.max(1), 1.0);
        };
        if landscape.consonance_field_level.len() != eval_space.n_bins() {
            return self.eval_harmonic_like_in_range(min_mul, max_mul, self.count.max(1), 1.0);
        }
        let scan = &landscape.consonance_field_level;
        let mut candidates = Vec::new();
        for idx in idx_min..=idx_max {
            let center = sanitize_nonnegative_finite(scan[idx]);
            if center <= 0.0 {
                continue;
            }
            let left = if idx > 0 {
                sanitize_nonnegative_finite(scan[idx - 1])
            } else {
                center
            };
            let right = if idx + 1 < scan.len() {
                sanitize_nonnegative_finite(scan[idx + 1])
            } else {
                center
            };
            if center >= left && center >= right {
                candidates.push((idx, center));
            }
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected: Vec<f32> = Vec::with_capacity(self.count.max(1));
        let min_dist_erb = sanitize_nonnegative_finite(self.min_dist_erb);
        for (idx, _) in candidates {
            let freq = eval_space.freq_of_index(idx);
            let freq_erb = hz_to_erb(freq.max(1.0e-6));
            let too_close = selected.iter().any(|&chosen_freq_hz| {
                let chosen_erb = hz_to_erb(chosen_freq_hz.max(1.0e-6));
                (chosen_erb - freq_erb).abs() < min_dist_erb
            });
            if too_close {
                continue;
            }
            selected.push(freq);
            if selected.len() >= self.count.max(1) {
                break;
            }
        }

        if selected.is_empty() {
            self.eval_harmonic_like_in_range(min_mul, max_mul, self.count.max(1), 1.0)
        } else {
            selected.into_iter().map(|freq| freq / base_hz).collect()
        }
    }
}

pub fn modal_table_ratios(name: &str) -> Option<&'static [f32]> {
    let key = name.trim().to_ascii_lowercase();
    match key.as_str() {
        "uniform_aluminum_bar" => Some(MODAL_TABLE_UNIFORM_ALUMINUM_BAR),
        "uniform_wooden_bar" => Some(MODAL_TABLE_UNIFORM_WOODEN_BAR),
        "xylophone" => Some(MODAL_TABLE_XYLOPHONE),
        "vibraphone_1" => Some(MODAL_TABLE_VIBRAPHONE_1),
        "vibraphone_2" => Some(MODAL_TABLE_VIBRAPHONE_2),
        "wine_glass" => Some(MODAL_TABLE_WINE_GLASS),
        _ => None,
    }
}

fn sanitized_range(min_mul: f32, max_mul: f32) -> (f32, f32) {
    let mut min_mul = sanitize_positive_finite(min_mul, DEFAULT_MIN_MUL);
    let mut max_mul = sanitize_positive_finite(max_mul, DEFAULT_MAX_MUL);
    if min_mul > max_mul {
        std::mem::swap(&mut min_mul, &mut max_mul);
    }
    (min_mul, max_mul)
}

fn sanitize_nonnegative_finite(x: f32) -> f32 {
    if x.is_finite() { x.max(0.0) } else { 0.0 }
}

fn sanitize_positive_finite(x: f32, fallback: f32) -> f32 {
    if x.is_finite() && x > 0.0 {
        x
    } else {
        fallback
    }
}

fn sanitize_ratio_list(mut ratios: Vec<f32>) -> Vec<f32> {
    ratios.retain(|ratio| ratio.is_finite() && *ratio > 0.0);
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ratios.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-4);
    ratios
}

fn freq_range_to_bins(space: &Log2Space, min_hz: f32, max_hz: f32) -> Option<(usize, usize)> {
    if space.n_bins() == 0 {
        return None;
    }
    let min_hz = min_hz.max(space.fmin).min(space.fmax);
    let max_hz = max_hz.max(space.fmin).min(space.fmax);
    let mut idx_min = space.index_of_freq(min_hz).unwrap_or(0);
    let mut idx_max = space
        .index_of_freq(max_hz)
        .unwrap_or_else(|| space.n_bins().saturating_sub(1));
    if idx_min > idx_max {
        std::mem::swap(&mut idx_min, &mut idx_max);
    }
    if idx_min >= space.n_bins() {
        return None;
    }
    idx_max = idx_max.min(space.n_bins().saturating_sub(1));
    if idx_min > idx_max {
        None
    } else {
        Some((idx_min, idx_max))
    }
}

fn suppress_neighbor_weights(
    space: &Log2Space,
    idx_min: usize,
    idx_max: usize,
    chosen_idx: usize,
    min_dist_erb: f32,
    weights: &mut [f32],
) {
    if weights.is_empty() || chosen_idx < idx_min || chosen_idx > idx_max {
        return;
    }
    if min_dist_erb <= 0.0 {
        let offset = chosen_idx - idx_min;
        if let Some(weight) = weights.get_mut(offset) {
            *weight = 0.0;
        }
        return;
    }
    let chosen_freq = space.freq_of_index(chosen_idx);
    let chosen_erb = hz_to_erb(chosen_freq.max(1.0e-6));
    for idx in idx_min..=idx_max {
        let freq = space.freq_of_index(idx);
        let dist = (hz_to_erb(freq.max(1.0e-6)) - chosen_erb).abs();
        if dist < min_dist_erb {
            let offset = idx - idx_min;
            if let Some(weight) = weights.get_mut(offset) {
                *weight = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::landscape::Landscape;

    #[test]
    fn harmonic_modes_default_count_is_16() {
        let pattern = ModePattern::harmonic_modes();
        let space = Log2Space::new(55.0, 8800.0, 48);
        let mut rng = SmallRng::seed_from_u64(1);
        let ratios = pattern.eval(220.0, &space, None, &mut rng);
        assert_eq!(ratios.len(), 16);
        assert!((ratios[0] - 1.0).abs() <= 1e-6);
        assert!((ratios[1] - 2.0).abs() <= 1e-6);
    }

    #[test]
    fn harmonic_modes_ignore_range_constraints() {
        let pattern = ModePattern::harmonic_modes()
            .with_count(16)
            .with_range(1.0, 4.0);
        let space = Log2Space::new(55.0, 8800.0, 48);
        let mut rng = SmallRng::seed_from_u64(3);
        let ratios = pattern.eval(220.0, &space, None, &mut rng);
        assert_eq!(ratios.len(), 16);
        assert!((ratios[15] - 16.0).abs() <= 1.0e-6);
    }

    #[test]
    fn modal_table_lookup_is_case_insensitive() {
        let pattern = ModePattern::modal_table("Uniform_Aluminum_Bar").expect("table");
        let space = Log2Space::new(55.0, 8800.0, 48);
        let mut rng = SmallRng::seed_from_u64(1);
        let ratios = pattern.with_count(3).eval(220.0, &space, None, &mut rng);
        assert_eq!(ratios.len(), 3);
        assert!((ratios[0] - 1.0).abs() <= 1e-6);
    }

    #[test]
    fn modal_table_default_range_keeps_full_table() {
        let pattern = ModePattern::modal_table("uniform_aluminum_bar").expect("table");
        let space = Log2Space::new(55.0, 8800.0, 48);
        let mut rng = SmallRng::seed_from_u64(4);
        let ratios = pattern.eval(220.0, &space, None, &mut rng);
        assert!(ratios.len() >= 6);
        assert!(ratios.iter().any(|ratio| (ratio - 18.64).abs() <= 1.0e-3));
    }

    #[test]
    fn modal_table_jitter_preserves_table_len() {
        let space = Log2Space::new(55.0, 8800.0, 48);
        for seed in [1u64, 2, 3, 4, 5] {
            let pattern = ModePattern::modal_table("uniform_aluminum_bar")
                .expect("table")
                .with_jitter_cents(12.0)
                .with_seed(seed);
            let mut rng = SmallRng::seed_from_u64(77);
            let ratios = pattern.eval(220.0, &space, None, &mut rng);
            assert_eq!(ratios.len(), 6);
        }
    }

    #[test]
    fn custom_modes_jitter_preserves_len() {
        let pattern = ModePattern::custom_modes(vec![1.0, 2.0, 3.0])
            .with_jitter_cents(12.0)
            .with_seed(123);
        let space = Log2Space::new(55.0, 8800.0, 48);
        let mut rng = SmallRng::seed_from_u64(9);
        let ratios = pattern.eval(220.0, &space, None, &mut rng);
        assert_eq!(ratios.len(), 3);
    }

    #[test]
    fn landscape_density_falls_back_to_harmonic_without_landscape() {
        let pattern = ModePattern::landscape_density_modes().with_count(4);
        let space = Log2Space::new(55.0, 8800.0, 48);
        let mut rng = SmallRng::seed_from_u64(1);
        let ratios = pattern.eval(220.0, &space, None, &mut rng);
        assert_eq!(ratios, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn landscape_density_applies_erb_inhibition() {
        let space = Log2Space::new(55.0, 8800.0, 48);
        let mut landscape = Landscape::new(space.clone());
        let idx_a = space.index_of_freq(220.0).expect("idx a");
        let idx_b = space.index_of_freq(233.0).expect("idx b");
        let idx_c = space.index_of_freq(440.0).expect("idx c");
        landscape.consonance_density_mass[idx_a] = 1.0;
        landscape.consonance_density_mass[idx_b] = 0.95;
        landscape.consonance_density_mass[idx_c] = 0.8;

        let pattern = ModePattern::landscape_density_modes()
            .with_count(2)
            .with_range(1.0, 2.5)
            .with_min_dist_erb(0.9);
        let mut rng = SmallRng::seed_from_u64(1);
        let ratios = pattern.eval(220.0, &space, Some(&landscape), &mut rng);
        assert_eq!(ratios.len(), 2);
        // 233 Hz should be inhibited by 220 Hz under the ERB distance.
        assert!(ratios.iter().all(|r| (r - (233.0 / 220.0)).abs() > 1.0e-3));
    }
}
