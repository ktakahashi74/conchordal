use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PerceptualConfig {
    pub tau_fast: Option<f32>,
    pub tau_slow: Option<f32>,
    pub w_boredom: Option<f32>,
    pub w_familiarity: Option<f32>,
    pub rho_self: Option<f32>,
    pub boredom_gamma: Option<f32>,
    pub self_smoothing_radius: Option<usize>,
    pub silence_mass_epsilon: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct FeaturesNow {
    pub distribution: Vec<f32>,
    pub mass: f32,
}

impl FeaturesNow {
    pub fn from_subjective_intensity(raw: &[f32]) -> Self {
        let mut distribution = Vec::with_capacity(raw.len());
        let mut sum = 0.0f32;
        for &v in raw {
            let c = v.max(0.0).ln_1p();
            sum += c;
            distribution.push(c);
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for v in &mut distribution {
                *v *= inv;
            }
        }
        Self {
            distribution,
            mass: sum,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerceptualContext {
    pub tau_fast: f32,
    pub tau_slow: f32,
    pub w_boredom: f32,
    pub w_familiarity: f32,
    pub rho_self: f32,
    pub boredom_gamma: f32,
    pub self_smoothing_radius: usize,
    pub silence_mass_epsilon: f32,
    h_fast: Vec<f32>,
    h_slow: Vec<f32>,
}

impl PerceptualContext {
    pub fn from_config(config: &PerceptualConfig, n_bins: usize) -> Self {
        let tau_fast = config.tau_fast.unwrap_or(0.5).max(1e-3);
        let tau_slow = config.tau_slow.unwrap_or(20.0).max(tau_fast + 1e-3);
        let w_boredom = config.w_boredom.unwrap_or(1.0).max(0.0);
        let w_familiarity = config.w_familiarity.unwrap_or(0.2).max(0.0);
        let rho_self = config.rho_self.unwrap_or(0.15).clamp(0.0, 1.0);
        let boredom_gamma = config.boredom_gamma.unwrap_or(0.5).clamp(0.1, 1.0);
        let self_smoothing_radius = config.self_smoothing_radius.unwrap_or(1);
        let silence_mass_epsilon = config.silence_mass_epsilon.unwrap_or(1e-6).max(0.0);
        Self {
            tau_fast,
            tau_slow,
            w_boredom,
            w_familiarity,
            rho_self,
            boredom_gamma,
            self_smoothing_radius,
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

    pub fn score_adjustment(&self, candidate_idx: usize) -> f32 {
        if candidate_idx >= self.h_fast.len() {
            return 0.0;
        }
        let radius = self
            .self_smoothing_radius
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

    pub fn update(&mut self, candidate_idx: usize, features: &FeaturesNow, dt: f32) {
        if self.h_fast.is_empty() || self.h_slow.is_empty() {
            return;
        }
        debug_assert_eq!(features.distribution.len(), self.h_fast.len());
        let dt = dt.max(0.0);
        let a_f = (-dt / self.tau_fast.max(1e-3)).exp();
        let a_s = (-dt / self.tau_slow.max(1e-3)).exp();
        let one_minus_a_f = 1.0 - a_f;
        let one_minus_a_s = 1.0 - a_s;

        let n_bins = self.h_fast.len();
        let env_present = features.mass > self.silence_mass_epsilon;
        let (env_weight, self_weight) = if env_present {
            (1.0 - self.rho_self, self.rho_self)
        } else {
            (0.0, 1.0)
        };

        for (i, (h_fast, h_slow)) in self
            .h_fast
            .iter_mut()
            .zip(self.h_slow.iter_mut())
            .enumerate()
        {
            let x_env = if env_present && i < features.distribution.len() {
                features.distribution[i]
            } else {
                0.0
            };
            let x_env_scaled = env_weight * x_env;
            *h_fast = a_f * *h_fast + one_minus_a_f * x_env_scaled;
            *h_slow = a_s * *h_slow + one_minus_a_s * x_env_scaled;
        }

        if self_weight > 0.0 && candidate_idx < n_bins {
            let (h_fast, h_slow) = (&mut self.h_fast, &mut self.h_slow);
            let radius = self.self_smoothing_radius.min(n_bins.saturating_sub(1));
            for_each_candidate_weight(candidate_idx, n_bins, radius, |idx, w| {
                h_fast[idx] += one_minus_a_f * self_weight * w;
                h_slow[idx] += one_minus_a_s * self_weight * w;
            });
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
        let dist = offset.unsigned_abs();
        let weight = if radius == 1 {
            if offset == 0 { 0.5 } else { 0.25 }
        } else {
            1.0 - (dist as f32 / (radius as f32 + 1.0))
        };
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
        let dist = offset.unsigned_abs();
        let weight = if radius == 1 {
            if offset == 0 { 0.5 } else { 0.25 }
        } else {
            1.0 - (dist as f32 / (radius as f32 + 1.0))
        };
        if weight > 0.0 {
            f(idx as usize, weight * inv_sum);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::for_each_candidate_weight;

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
}
