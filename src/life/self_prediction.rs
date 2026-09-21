//! Issued private command predictions, prequential scores, and bounded residual learning.

use super::action_observation::{ActionKind, Outcome};

const INPUTS: usize = 18;
const OUTPUTS: usize = 5;
const CAPACITY: usize = 64;

mod descriptor;
pub use descriptor::{DescriptorOutcome, DescriptorStats};
mod energy;
mod projection;
pub(crate) use projection::{CoherentWindow, ToneEnergy};
mod ratios;
pub(crate) use energy::WINDOWS as ENERGY_WINDOWS;
pub use energy::{EnergyForecast, EnergyStats, RetainedEnergy};
pub use ratios::EnergyRatioPreview;

pub(crate) struct Input {
    pub body_generation: u32,
    pub descriptors: [[Option<f64>; 6]; 2],
    pub descriptor_support: [Option<[u64; 3]>; 2],
    pub frequency_hz: f32,
    pub amplitude: f32,
    pub envelope: super::sound::envelope::Envelope,
    pub scheduled_release: Option<ScheduledRelease>,
    pub control: Option<super::sound::control_forecast::ControlForecast>,
    pub sine: Option<super::sound::sine_forecast::SineForecast>,
    pub bank: Option<super::sound::bank_forecast::BankForecast>,
    pub retained_energy: [RetainedEnergy; 2],
    pub coherent_energy: [[Option<f64>; ENERGY_WINDOWS]; 2],
    pub descriptor_target_end: u64,
    pub descriptor_slot: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct ScheduledRelease {
    pub apply_at_sample: u64,
    pub off_sample: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct BusForecast {
    /// Rows: fixed body, action-conditioned, action-independent. Columns are registered in I10.
    pub predictions: [[f64; OUTPUTS]; 3],
    pub target: [Option<f64>; OUTPUTS],
    pub squared_error: [[Option<f64>; OUTPUTS]; 3],
    pub updates_at_issue: u64,
    pub learned: bool,
    #[serde(skip)]
    input: [f64; INPUTS],
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Forecast {
    pub energy_ratio: Option<EnergyRatioPreview>,
    pub source_energy: [EnergyForecast; 2],
    /// Dimensionless renderer envelope; excludes backend energy and modulation.
    pub envelope: super::sound::envelope::Envelope,
    /// Issued plan, conditional on no intervening control or scheduler discontinuity.
    pub scheduled_release: Option<ScheduledRelease>,
    pub control: Option<super::sound::control_forecast::ControlForecast>,
    pub sine: Option<super::sound::sine_forecast::SineForecast>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bank: Option<super::sound::bank_forecast::BankForecast>,
    pub body_generation: u32,
    pub issued_at_sample: u64,
    pub buses: [Option<BusForecast>; 2],
    #[serde(skip)]
    slot: usize,
    #[serde(skip)]
    pub(crate) source_slot: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct Stats {
    pub energy: EnergyStats,
    pub descriptor: DescriptorStats,
    pub issued: u64,
    pub matched: u64,
    pub updated: u64,
    pub unsupported: u64,
    pub evicted: u64,
    pub retired: u64,
    pub squared_error_sum: [f64; 3],
    pub scored_coordinates: u64,
}

struct Model {
    energy_weights: [energy::Weights; 2],
    energy_updates: [u64; 2],
    owner: (u64, u32, u32),
    last_issue: u64,
    weights: [[[[f64; INPUTS]; OUTPUTS]; 2]; 2],
    updates: [u64; 2],
    descriptor_weights: [[[[f64; INPUTS]; 6]; 2]; 2],
    descriptor_updates: [u64; 2],
}

pub(crate) struct ModelBank {
    models: Vec<Option<Model>>,
    pub stats: Stats,
    descriptor_pending: Vec<Option<DescriptorOutcome>>,
    descriptor_ready: Vec<DescriptorOutcome>,
}

impl ModelBank {
    pub(crate) fn new() -> Self {
        Self {
            models: (0..CAPACITY).map(|_| None).collect(),
            stats: Stats::default(),
            descriptor_pending: (0..super::action_observation::CAPACITY)
                .map(|_| None)
                .collect(),
            descriptor_ready: Vec::with_capacity(super::action_observation::CAPACITY),
        }
    }

    pub(crate) fn issue(&mut self, outcome: &Outcome, input: Input) -> Option<Forecast> {
        if outcome.command_status != "accepted" || outcome.action == ActionKind::Shutdown {
            return None;
        }
        let owner = (
            outcome.source_id,
            outcome.source_generation,
            input.body_generation,
        );
        let existing = self
            .models
            .iter()
            .position(|m| m.as_ref().is_some_and(|m| m.owner == owner));
        let slot = existing
            .or_else(|| {
                self.models
                    .iter()
                    .position(|m| m.as_ref().is_some_and(|m| m.owner.0 == owner.0))
            })
            .or_else(|| self.models.iter().position(Option::is_none))
            .or_else(|| {
                self.models
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, m)| m.as_ref().unwrap().last_issue)
                    .map(|(i, _)| i)
            })
            .expect("nonempty model bank");
        if existing.is_none() {
            self.stats.evicted += u64::from(self.models[slot].is_some());
            self.models[slot] = Some(Model {
                energy_weights: [[[[0.; INPUTS + 1]; ENERGY_WINDOWS]; 2]; 2],
                energy_updates: [0; 2],
                owner,
                last_issue: outcome.command_id,
                weights: [[[[0.; INPUTS]; OUTPUTS]; 2]; 2],
                updates: [0; 2],
                descriptor_weights: [[[[0.; INPUTS]; 6]; 2]; 2],
                descriptor_updates: [0; 2],
            });
        }
        let model = self.models[slot].as_mut().unwrap();
        model.last_issue = outcome.command_id;
        let start = outcome.scheduled_action_sample?;
        let width = (outcome.window_end_sample? - start) as f64;
        let terminal_envelope = input.scheduled_release.map_or(input.envelope, |release| {
            input.envelope.with_release(release.off_sample)
        });
        let end = terminal_envelope.release_end.saturating_sub(start) as f64 / width;
        let mut buses = [None; 2];
        let mut source_energy = [None; 2];
        for (bus, result) in buses.iter_mut().enumerate() {
            let fixed = [
                if input.amplitude > 0. { 0.99 } else { 0.01 },
                1. / width,
                (f64::from(input.amplitude).max(0.) / 2_f64.sqrt())
                    .max(1e-6)
                    .log2()
                    / 20.,
                if end < 1. { 0.99 } else { 0.01 },
                end.clamp(0., 1.),
            ];
            let mut x = [0.; INPUTS];
            x[..6].copy_from_slice(&[
                1.,
                f64::from(outcome.action == ActionKind::Onset),
                f64::from(outcome.action == ActionKind::Release),
                f64::from(input.frequency_hz).max(1.).log2() / 16.,
                f64::from(input.amplitude).max(1e-6).log2() / 20.,
                end.clamp(0., 1.),
            ]);
            for (i, value) in input.descriptors[bus].iter().enumerate() {
                if let Some(value) = value.filter(|v| v.is_finite()) {
                    x[6 + i] = (value / 4.).tanh();
                    x[12 + i] = 1.;
                }
            }
            source_energy[bus] = Some(EnergyForecast::issue(
                &input,
                [start, outcome.window_end_sample.unwrap()],
                outcome.buses[bus].routed,
                bus,
                x,
                &model.energy_weights[bus],
                model.energy_updates[bus],
            ));
            if !outcome.buses[bus].routed {
                continue;
            }
            let mut blind = x;
            blind[1..6].fill(0.);
            let mut predictions = [fixed, fixed, [0.5, 0.5, -0.5, 0.5, 0.5]];
            for learned in 0..2 {
                let features = if learned == 0 { x } else { blind };
                for (coordinate, value) in predictions[learned + 1].iter_mut().enumerate() {
                    *value += model.weights[bus][learned][coordinate]
                        .iter()
                        .zip(features)
                        .map(|(w, x)| w * x)
                        .sum::<f64>();
                    *value = if coordinate == 2 {
                        value.clamp(-8., 8.)
                    } else {
                        value.clamp(0., 1.)
                    };
                }
            }
            *result = Some(BusForecast {
                predictions,
                target: [None; OUTPUTS],
                squared_error: [[None; OUTPUTS]; 3],
                updates_at_issue: model.updates[bus],
                learned: false,
                input: x,
            });
        }
        let forecast = Forecast {
            energy_ratio: None,
            source_energy: source_energy.map(Option::unwrap),
            source_slot: input.descriptor_slot,
            envelope: input.envelope,
            scheduled_release: input.scheduled_release,
            control: input.control,
            sine: input.sine,
            bank: input.bank,
            body_generation: input.body_generation,
            issued_at_sample: outcome.issued_at_sample,
            buses,
            slot,
        };
        self.stats.issued += 1;
        self.issue_descriptor(outcome, &input, &forecast);
        Some(forecast)
    }

    pub(crate) fn observe(&mut self, outcome: &mut Outcome) {
        let Some(forecast) = outcome.prediction.as_mut() else {
            return;
        };
        let model = self.models[forecast.slot].as_mut().filter(|m| {
            m.owner
                == (
                    outcome.source_id,
                    outcome.source_generation,
                    forecast.body_generation,
                )
        });
        let mut model = model;
        for (bus, energy) in forecast.source_energy.iter_mut().enumerate() {
            if outcome.command_status != "accepted" {
                energy.target.fill(None);
            }
            energy.observe(
                model
                    .as_deref_mut()
                    .map(|m| (&mut m.energy_weights[bus], &mut m.energy_updates[bus])),
                &mut self.stats.energy,
            );
        }
        let start = outcome.scheduled_action_sample.unwrap();
        let width = (outcome.window_end_sample.unwrap() - start) as f64;
        for (bus, predicted) in forecast.buses.iter_mut().enumerate() {
            let Some(predicted) = predicted else {
                continue;
            };
            let heard = outcome.buses[bus];
            if outcome.command_status != "accepted"
                || !matches!(heard.status, "activity" | "below_threshold" | "silence")
            {
                self.stats.unsupported += 1;
                continue;
            }
            let active = heard.first_activity_sample.is_some();
            let ended = outcome.renderer_end_sample;
            predicted.target = [
                Some(f64::from(active)),
                heard
                    .first_activity_sample
                    .map(|t| (t - start) as f64 / width),
                active.then(|| heard.rms.unwrap().max(1e-6).log2() / 20.),
                Some(f64::from(ended.is_some())),
                ended.map(|t| (t - start) as f64 / width),
            ];
            self.stats.matched += 1;
            for (coordinate, target) in predicted.target.iter().enumerate() {
                let Some(target) = target else {
                    continue;
                };
                self.stats.scored_coordinates += 1;
                for comparison in 0..3 {
                    let error = target - predicted.predictions[comparison][coordinate];
                    predicted.squared_error[comparison][coordinate] = Some(error * error);
                    self.stats.squared_error_sum[comparison] += error * error;
                }
            }
            let Some(model) = model.as_deref_mut() else {
                self.stats.retired += 1;
                continue;
            };
            for learned in 0..2 {
                let mut x = predicted.input;
                if learned == 1 {
                    x[1..6].fill(0.);
                }
                let norm = 1. + x.iter().map(|v| v * v).sum::<f64>();
                for (coordinate, target) in predicted.target.iter().enumerate() {
                    let Some(target) = target else {
                        continue;
                    };
                    let error = target - predicted.predictions[learned + 1][coordinate];
                    for (weight, x) in model.weights[bus][learned][coordinate].iter_mut().zip(x) {
                        *weight = (*weight + 0.1 * error * x / norm).clamp(-8., 8.);
                    }
                }
            }
            model.updates[bus] += 1;
            predicted.learned = true;
            self.stats.updated += 1;
        }
    }
}

#[cfg(test)]
mod tests;
