//! Issued whole-source energy windows, scored before residual learning.

use super::{INPUTS, Input};

pub(crate) const WINDOWS: usize = 16;
const FEATURES: usize = INPUTS + 1;
const SCALE: f64 = 1e-6;
pub(super) type Weights = [[[f64; FEATURES]; WINDOWS]; 2];

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct RetainedEnergy {
    pub complete: bool,
    pub scanned_entries: usize,
    pub included_tones: usize,
    pub fixed_energy: [f64; WINDOWS],
    pub control_supported: [bool; WINDOWS],
}

impl Default for RetainedEnergy {
    fn default() -> Self {
        Self {
            complete: true,
            scanned_entries: 0,
            included_tones: 0,
            fixed_energy: [0.; WINDOWS],
            control_supported: [true; WINDOWS],
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct EnergyStats {
    pub ratio_previews: u64,
    pub ratio_supported: u64,
    pub ratio_unknown: u64,
    pub scored_windows: u64,
    pub updated: u64,
    pub unsupported_windows: u64,
    pub retired: u64,
    pub squared_error_sum: [f64; 3],
    pub log_squared_error_sum: [f64; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct EnergyForecast {
    pub model: &'static str,
    pub command_fixed_energy: [Option<f64>; WINDOWS],
    pub retained: RetainedEnergy,
    pub incoherent_fixed_energy: [Option<f64>; WINDOWS],
    pub coherent_sine_energy: [Option<f64>; WINDOWS],
    pub windows: [[u64; 2]; WINDOWS],
    /// Fixed sine-envelope proxy, action-conditioned residual, action-independent residual.
    pub predictions: [[Option<f64>; WINDOWS]; 3],
    pub target: [Option<f64>; WINDOWS],
    pub squared_error: [[Option<f64>; WINDOWS]; 3],
    pub updates_at_issue: u64,
    pub learned: bool,
    #[serde(skip)]
    features: [f64; FEATURES],
}

impl EnergyForecast {
    pub(super) fn issue(
        input: &Input,
        interval: [u64; 2],
        routed: bool,
        bus: usize,
        features: [f64; INPUTS],
        weights: &Weights,
        updates: u64,
    ) -> Self {
        let retained = input.retained_energy[bus];
        let [start, end] = interval;
        let width = end - start;
        let windows = std::array::from_fn(|k| {
            [k, k + 1].map(|edge| {
                start + (u128::from(width) * edge as u128).div_ceil(WINDOWS as u128) as u64
            })
        });
        let mut x = [0.; FEATURES];
        x[..INPUTS].copy_from_slice(&features);
        x[INPUTS] = f64::from(routed);
        let mut blind = x;
        blind[1..6].fill(0.);
        blind[INPUTS] = 0.;
        let mut predictions = [[None; WINDOWS]; 3];
        let mut command_fixed_energy = [None; WINDOWS];
        let mut incoherent_fixed_energy = [None; WINDOWS];
        let maximum_log = (f64::from(f32::MAX).powi(2) / SCALE).ln_1p();
        for (k, [left, right]) in windows.iter().copied().enumerate() {
            if right == left {
                continue;
            }
            let tick = left + (right - left) / 2;
            let fixed = if routed {
                super::ToneEnergy {
                    amplitude: input.amplitude,
                    envelope: input.envelope,
                    control: input.control,
                    sine: input.sine,
                    scheduled_release: input.scheduled_release,
                }
                .at(tick, None)
            } else {
                Some(0.)
            };
            command_fixed_energy[k] = fixed;
            let Some(fixed) = fixed else {
                continue;
            };
            if !retained.complete || !retained.control_supported[k] {
                continue;
            }
            let fixed = fixed + retained.fixed_energy[k];
            incoherent_fixed_energy[k] = Some(fixed);
            let fixed = input.coherent_energy[bus][k].unwrap_or(fixed);
            predictions[0][k] = Some(fixed);
            for (model, features) in [x, blind].iter().enumerate() {
                let prior = if model == 0 { fixed } else { 1e-4 };
                let log = (prior / SCALE).ln_1p()
                    + weights[model][k]
                        .iter()
                        .zip(features)
                        .map(|(w, x)| w * x)
                        .sum::<f64>();
                predictions[model + 1][k] = Some(log.clamp(0., maximum_log).exp_m1() * SCALE);
            }
        }
        Self {
            model: "source_energy_log1p_residual_v7",
            command_fixed_energy,
            retained,
            incoherent_fixed_energy,
            coherent_sine_energy: input.coherent_energy[bus],
            windows,
            predictions,
            target: [None; WINDOWS],
            squared_error: [[None; WINDOWS]; 3],
            updates_at_issue: updates,
            learned: false,
            features: x,
        }
    }

    pub(super) fn observe(
        &mut self,
        model: Option<(&mut Weights, &mut u64)>,
        stats: &mut EnergyStats,
    ) {
        let mut model = model;
        let mut scored = false;
        for k in 0..WINDOWS {
            if self.windows[k][0] == self.windows[k][1] {
                continue;
            }
            if self.predictions.iter().any(|p| p[k].is_none()) {
                stats.unsupported_windows += 1;
                continue;
            }
            let Some(target) = self.target[k] else {
                stats.unsupported_windows += 1;
                continue;
            };
            scored = true;
            stats.scored_windows += 1;
            let target_log = (target / SCALE).ln_1p();
            for comparison in 0..3 {
                let prediction = self.predictions[comparison][k].unwrap();
                let squared = (target - prediction).powi(2);
                self.squared_error[comparison][k] = Some(squared);
                stats.squared_error_sum[comparison] += squared;
                stats.log_squared_error_sum[comparison] +=
                    (target_log - (prediction / SCALE).ln_1p()).powi(2);
            }
            if let Some((weights, _)) = model.as_mut() {
                for (comparison, model) in weights.iter_mut().enumerate() {
                    let mut x = self.features;
                    if comparison == 1 {
                        x[1..6].fill(0.);
                        x[INPUTS] = 0.;
                    }
                    let error =
                        target_log - (self.predictions[comparison + 1][k].unwrap() / SCALE).ln_1p();
                    let norm = 1. + x.iter().map(|v| v * v).sum::<f64>();
                    for (weight, value) in model[k].iter_mut().zip(x) {
                        *weight = (*weight + 0.1 * error * value / norm).clamp(-8., 8.);
                    }
                }
            }
        }
        if scored {
            if let Some((_, updates)) = model {
                *updates += 1;
                self.learned = true;
                stats.updated += 1;
            } else {
                stats.retired += 1;
            }
        }
    }
}
