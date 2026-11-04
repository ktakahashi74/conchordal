use crate::core::log2space::Log2Space;

#[derive(Debug, Clone, Default)]
pub struct AdaptationConfig {
    pub tau_fast: Option<f32>,
    pub tau_slow: Option<f32>,
    pub w_boredom: Option<f32>,
    pub w_familiarity: Option<f32>,
    pub environment_weight: Option<f32>,
    pub boredom_gamma: Option<f32>,
    pub candidate_smoothing_radius: Option<usize>,
    pub silence_mass_epsilon: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct FeaturesNow {
    pub density: Vec<f32>,
    pub mass: f32,
}

impl FeaturesNow {
    /// Build a normalized density (PMF over Log2Space bins) from a non-negative
    /// fundamental-occupancy scan. `mass` is the pre-normalization total, used to
    /// detect silence. The input must be on fundamentals, not the full spectrum,
    /// so that shared harmonics of consonant intervals are not counted as occupied.
    ///
    /// `space` is taken only to enforce the F2 scan-alignment boundary.
    pub fn from_occupancy_scan(space: &Log2Space, raw: &[f32]) -> Self {
        space.assert_scan_len_named(raw, "occupancy_scan");
        let mut density = Vec::with_capacity(raw.len());
        let mut sum = 0.0f32;
        for &v in raw {
            let c = v.max(0.0).ln_1p();
            sum += c;
            density.push(c);
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for v in &mut density {
                *v *= inv;
            }
        }
        Self { density, mass: sum }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptationContext {
    enabled: bool,
    pub tau_fast: f32,
    pub tau_slow: f32,
    pub w_boredom: f32,
    pub w_familiarity: f32,
    pub environment_weight: f32,
    pub boredom_gamma: f32,
    pub candidate_smoothing_radius: usize,
    pub silence_mass_epsilon: f32,
    h_fast: Vec<f32>,
    h_slow: Vec<f32>,
}

impl AdaptationContext {
    pub fn from_config(config: &AdaptationConfig, n_bins: usize) -> Self {
        let tau_fast = config.tau_fast.unwrap_or(0.5).max(1e-3);
        let tau_slow = config.tau_slow.unwrap_or(20.0).max(tau_fast + 1e-3);
        let w_boredom = config.w_boredom.unwrap_or(1.0).max(0.0);
        let w_familiarity = config.w_familiarity.unwrap_or(0.2).max(0.0);
        let environment_weight = config
            .environment_weight
            .unwrap_or(1.0 - 0.15)
            .clamp(0.0, 1.0);
        let boredom_gamma = config.boredom_gamma.unwrap_or(0.5).clamp(0.1, 1.0);
        let candidate_smoothing_radius = config.candidate_smoothing_radius.unwrap_or(1);
        let silence_mass_epsilon = config.silence_mass_epsilon.unwrap_or(1e-6).max(0.0);
        Self {
            enabled: true,
            tau_fast,
            tau_slow,
            w_boredom,
            w_familiarity,
            environment_weight,
            boredom_gamma,
            candidate_smoothing_radius,
            silence_mass_epsilon,
            h_fast: vec![0.0; n_bins],
            h_slow: vec![0.0; n_bins],
        }
    }

    pub fn ensure_len(&mut self, n_bins: usize) {
        if self.h_fast.len() == n_bins {
            return;
        }
        self.h_fast.resize(n_bins, 0.0);
        self.h_slow.resize(n_bins, 0.0);
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn score_adjustment(&self, candidate_idx: usize) -> f32 {
        if !self.enabled {
            return 0.0;
        }
        if candidate_idx >= self.h_fast.len() {
            return 0.0;
        }
        let radius = self
            .candidate_smoothing_radius
            .min(self.h_fast.len().saturating_sub(1));
        let mut boredom = 0.0f32;
        let mut familiarity = 0.0f32;
        for_each_candidate_weight(candidate_idx, self.h_fast.len(), radius, |idx, w| {
            boredom += self.h_fast[idx] * w;
            familiarity += self.h_slow[idx] * w;
        });
        let curved_boredom = boredom.max(0.0).powf(self.boredom_gamma);
        (self.w_familiarity * familiarity) - (self.w_boredom * curved_boredom)
    }

    pub fn update(&mut self, features: &FeaturesNow, dt: f32) {
        if !self.enabled {
            return;
        }
        if self.h_fast.is_empty() || self.h_slow.is_empty() {
            return;
        }
        debug_assert_eq!(features.density.len(), self.h_fast.len());
        let dt = dt.max(0.0);
        let a_f = (-dt / self.tau_fast.max(1e-3)).exp();
        let a_s = (-dt / self.tau_slow.max(1e-3)).exp();
        let one_minus_a_f = 1.0 - a_f;
        let one_minus_a_s = 1.0 - a_s;

        let env_present = features.mass > self.silence_mass_epsilon;
        let env_weight = if env_present {
            self.environment_weight
        } else {
            0.0
        };

        for (i, (h_fast, h_slow)) in self
            .h_fast
            .iter_mut()
            .zip(self.h_slow.iter_mut())
            .enumerate()
        {
            let x_env = if env_present && i < features.density.len() {
                features.density[i]
            } else {
                0.0
            };
            let x_env_scaled = env_weight * x_env;
            *h_fast = a_f * *h_fast + one_minus_a_f * x_env_scaled;
            *h_slow = a_s * *h_slow + one_minus_a_s * x_env_scaled;
        }
    }
}

fn for_each_candidate_weight(
    candidate_idx: usize,
    n_bins: usize,
    radius: usize,
    mut f: impl FnMut(usize, f32),
) {
    if candidate_idx >= n_bins || n_bins == 0 {
        return;
    }
    let radius = radius.min(n_bins.saturating_sub(1));
    if radius == 0 {
        f(candidate_idx, 1.0);
        return;
    }

    let mut sum = 0.0f32;
    for offset in -(radius as isize)..=(radius as isize) {
        let idx = candidate_idx as isize + offset;
        if idx < 0 || idx >= n_bins as isize {
            continue;
        }
        let weight = candidate_weight(offset, radius);
        if weight > 0.0 {
            sum += weight;
        }
    }

    if sum <= 0.0 {
        return;
    }
    let inv_sum = 1.0 / sum;
    for offset in -(radius as isize)..=(radius as isize) {
        let idx = candidate_idx as isize + offset;
        if idx < 0 || idx >= n_bins as isize {
            continue;
        }
        let weight = candidate_weight(offset, radius);
        if weight > 0.0 {
            f(idx as usize, weight * inv_sum);
        }
    }
}

#[inline]
fn candidate_weight(offset: isize, radius: usize) -> f32 {
    if radius == 1 {
        if offset == 0 { 0.5 } else { 0.25 }
    } else {
        let dist = offset.unsigned_abs();
        1.0 - (dist as f32 / (radius as f32 + 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::{AdaptationConfig, AdaptationContext, FeaturesNow, for_each_candidate_weight};

    #[test]
    fn candidate_weights_normalize_and_stay_in_range() {
        let mut sum = 0.0f32;
        let mut max_idx = 0usize;
        for_each_candidate_weight(0, 8, 1, |idx, w| {
            sum += w;
            max_idx = max_idx.max(idx);
        });
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(max_idx < 8);
    }

    #[test]
    fn candidate_weights_preserve_ratio_at_edges() {
        let mut w0 = 0.0f32;
        let mut w1 = 0.0f32;
        for_each_candidate_weight(0, 8, 1, |idx, w| {
            if idx == 0 {
                w0 = w;
            } else if idx == 1 {
                w1 = w;
            }
        });
        assert!(w1 > 0.0);
        assert!((w0 / w1 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn candidate_weights_radius_is_clamped() {
        let mut sum = 0.0f32;
        let mut max_idx = 0usize;
        for_each_candidate_weight(2, 5, 999, |idx, w| {
            sum += w;
            max_idx = max_idx.max(idx);
        });
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(max_idx < 5);
    }

    #[test]
    fn no_neighbors_never_add_private_self_exposure() {
        let mut ctx = AdaptationContext::from_config(&AdaptationConfig::default(), 8);
        let silent = FeaturesNow {
            density: vec![0.0; 8],
            mass: 0.0,
        };
        for _ in 0..1000 {
            ctx.update(&silent, 0.2);
            for i in 0..8 {
                assert_eq!(ctx.score_adjustment(i), 0.0);
            }
        }
        assert_eq!(ctx.h_fast, vec![0.0; 8]);
        assert_eq!(ctx.h_slow, vec![0.0; 8]);
    }

    #[test]
    fn environment_history_and_scores_match_previous_environment_term() {
        use crate::life::control_adapters::adaptation_config_from_control;
        use crate::scenario::control::AdaptationControl;
        for adaptation in [0.0_f32, 0.5, 1.0] {
            for old_self_focus in [0.0_f32, 0.15, 0.7, 1.0] {
                let config = adaptation_config_from_control(&AdaptationControl {
                    adaptation,
                    environment_weight: 1.0 - old_self_focus,
                    ..Default::default()
                });
                let mut ctx = AdaptationContext::from_config(&config, 8);
                let mut fast = [0.0_f32; 8];
                let mut slow = [0.0_f32; 8];
                for step in 0..120 {
                    let dt = [0.01_f32, 0.2, 2.0][step % 3];
                    let present = step % 5 != 0;
                    let now = FeaturesNow {
                        density: (0..8)
                            .map(|i| if i == step % 8 { 1.0 } else { 0.0 })
                            .collect(),
                        mass: if present { 1.0 } else { 0.0 },
                    };
                    let af = (-dt / (0.1 + (1.0 - adaptation) * 0.8)).exp();
                    let aslow = (-dt / (5.0 + (1.0 - adaptation) * 30.0)).exp();
                    let weight = if present { 1.0 - old_self_focus } else { 0.0 };
                    for i in 0..8 {
                        let input = weight * if present { now.density[i] } else { 0.0 };
                        fast[i] = af * fast[i] + (1.0 - af) * input;
                        slow[i] = aslow * slow[i] + (1.0 - aslow) * input;
                    }
                    ctx.update(&now, dt);
                    assert_eq!(ctx.h_fast, fast);
                    assert_eq!(ctx.h_slow, slow);
                    for i in 0..8 {
                        // Independent spelling of the original radius-one reader, including edges.
                        let (boredom, familiarity) = if i == 0 {
                            (
                                fast[0] * (0.5 * (1.0 / 0.75)) + fast[1] * (0.25 * (1.0 / 0.75)),
                                slow[0] * (0.5 * (1.0 / 0.75)) + slow[1] * (0.25 * (1.0 / 0.75)),
                            )
                        } else if i == 7 {
                            (
                                fast[6] * (0.25 * (1.0 / 0.75)) + fast[7] * (0.5 * (1.0 / 0.75)),
                                slow[6] * (0.25 * (1.0 / 0.75)) + slow[7] * (0.5 * (1.0 / 0.75)),
                            )
                        } else {
                            (
                                fast[i - 1] * 0.25 + fast[i] * 0.5 + fast[i + 1] * 0.25,
                                slow[i - 1] * 0.25 + slow[i] * 0.5 + slow[i + 1] * 0.25,
                            )
                        };
                        // Keep powf's exponent dynamic, as in the old configurable scorer.
                        let expected = 0.2 * familiarity
                            - boredom.max(0.0).powf(std::hint::black_box(0.5_f32));
                        assert_eq!(ctx.score_adjustment(i).to_bits(), expected.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn disabled_context_forces_zero_score_adjustment() {
        let mut cfg = AdaptationConfig::default();
        cfg.w_boredom = Some(1.0);
        cfg.w_familiarity = Some(0.0);
        cfg.environment_weight = Some(1.0);
        let mut ctx = AdaptationContext::from_config(&cfg, 8);

        let mut features = vec![0.0f32; 8];
        features[3] = 1.0;
        let now = FeaturesNow {
            density: features,
            mass: 1.0,
        };
        for _ in 0..8 {
            ctx.update(&now, 0.2);
        }
        let before_disable = ctx.score_adjustment(3);
        assert!(before_disable.abs() > 1e-6);

        ctx.set_enabled(false);
        assert!(ctx.score_adjustment(3).abs() <= 1e-6);
    }
}
