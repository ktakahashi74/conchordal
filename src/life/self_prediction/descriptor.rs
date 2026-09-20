use super::*;
use crate::temporal_cognition::body::{Record, Snapshot};

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct DescriptorStats {
    pub issued: u64,
    pub matched: u64,
    pub updated: u64,
    pub unsupported: u64,
    pub retired: u64,
    pub capacity_dropped: u64,
    pub output_dropped: u64,
    pub pending: usize,
    pub scored_coordinates: u64,
    pub squared_error_sum: [f64; 3],
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct DescriptorBus {
    pub predictions: [[f64; 6]; 3],
    pub target: [Option<f64>; 6],
    pub squared_error: [[Option<f64>; 6]; 3],
    pub updates_at_issue: u64,
    pub status: &'static str,
    pub learned: bool,
    pub observed: Option<Record>,
    /// Input window start, end, and actual availability at issue time.
    pub input_support: Option<[u64; 3]>,
    pub input_mask: u8,
    #[serde(skip)]
    input: [f64; INPUTS],
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct DescriptorOutcome {
    pub command_id: u64,
    pub action: ActionKind,
    pub source_id: u64,
    pub source_generation: u32,
    pub body_generation: u32,
    pub issued_at_sample: u64,
    pub target_end_sample: u64,
    pub available_at_sample: u64,
    pub buses: [Option<DescriptorBus>; 2],
    #[serde(skip)]
    slot: usize,
    #[serde(skip)]
    body_slot: usize,
}

impl ModelBank {
    pub(super) fn issue_descriptor(&mut self, out: &Outcome, input: &Input, forecast: &Forecast) {
        let Some(pending) = self.descriptor_pending.iter_mut().find(|p| p.is_none()) else {
            self.stats.descriptor.capacity_dropped += 1;
            return;
        };
        let model = self.models[forecast.slot].as_ref().unwrap();
        let mut result = DescriptorOutcome {
            command_id: out.command_id,
            action: out.action,
            source_id: out.source_id,
            source_generation: out.source_generation,
            body_generation: input.body_generation,
            issued_at_sample: out.issued_at_sample,
            target_end_sample: input.descriptor_target_end,
            available_at_sample: out.issued_at_sample,
            buses: [None; 2],
            slot: forecast.slot,
            body_slot: input.descriptor_slot,
        };
        for (bus, scalar) in forecast.buses.iter().enumerate() {
            let Some(scalar) = scalar else { continue };
            let fixed = input.descriptors[bus]
                .map(|v| v.filter(|x| x.is_finite()).map_or(0., |v| (v / 4.).tanh()));
            let mut predictions = [fixed, fixed, [0.; 6]];
            for learned in 0..2 {
                let mut x = scalar.input;
                if learned == 1 {
                    x[1..6].fill(0.);
                }
                for (coordinate, value) in predictions[learned + 1].iter_mut().enumerate() {
                    *value = (*value
                        + model.descriptor_weights[bus][learned][coordinate]
                            .iter()
                            .zip(x)
                            .map(|(w, x)| w * x)
                            .sum::<f64>())
                    .clamp(-1., 1.);
                }
            }
            result.buses[bus] = Some(DescriptorBus {
                predictions,
                target: [None; 6],
                squared_error: [[None; 6]; 3],
                updates_at_issue: model.descriptor_updates[bus],
                status: "pending",
                learned: false,
                observed: None,
                input_support: input.descriptor_support[bus],
                input_mask: input.descriptors[bus]
                    .iter()
                    .enumerate()
                    .fold(0, |mask, (i, v)| {
                        mask | if v.is_some_and(|x| x.is_finite()) {
                            1 << i
                        } else {
                            0
                        }
                    }),
                input: scalar.input,
            });
        }
        *pending = Some(result);
        self.stats.descriptor.issued += 1;
        self.stats.descriptor.pending += 1;
    }

    pub(crate) fn intervene(&mut self, source_id: u64, generation: u32, at: u64) {
        for pending in self.descriptor_pending.iter_mut().flatten() {
            if pending.source_id == source_id
                && pending.source_generation == generation
                && at < pending.target_end_sample
            {
                for bus in pending.buses.iter_mut().flatten() {
                    bus.status = "intervening_command";
                }
            }
        }
    }

    pub(crate) fn cancel_descriptor(&mut self, command_id: u64) {
        for pending in self.descriptor_pending.iter_mut().flatten() {
            if pending.command_id == command_id {
                for bus in pending.buses.iter_mut().flatten() {
                    bus.status = "cancelled";
                }
            }
        }
    }

    pub(crate) fn observe_body(&mut self, snapshot: &Snapshot) {
        for pending in &mut self.descriptor_pending {
            let Some(out) = pending.as_ref() else {
                continue;
            };
            if snapshot.input_end < out.target_end_sample && !snapshot.finished {
                continue;
            }
            let mut out = pending.take().unwrap();
            self.stats.descriptor.pending -= 1;
            if self.descriptor_ready.len() == self.descriptor_ready.capacity() {
                self.stats.descriptor.output_dropped += 1;
                continue;
            }
            out.available_at_sample = snapshot.input_end;
            let mut model = self.models[out.slot]
                .as_mut()
                .filter(|m| m.owner == (out.source_id, out.source_generation, out.body_generation));
            for (bus, forecast) in out.buses.iter_mut().enumerate() {
                let Some(forecast) = forecast else { continue };
                if forecast.status != "pending" {
                    self.stats.descriptor.unsupported += 1;
                    continue;
                }
                let record = Some(&snapshot.records[out.body_slot * 2 + bus]).filter(|r| {
                    r.active
                        && r.source_id == out.source_id
                        && r.source_generation == out.source_generation
                        && r.body_generation == out.body_generation
                        && r.bus as usize == bus
                        && r.end == out.target_end_sample
                });
                let Some(record) = record else {
                    forecast.status = if snapshot.input_end < out.target_end_sample {
                        "unfinished"
                    } else {
                        "target_unavailable"
                    };
                    self.stats.descriptor.unsupported += 1;
                    continue;
                };
                forecast.observed = Some(*record);
                out.available_at_sample = out.available_at_sample.max(record.available);
                forecast.target = record
                    .standardized(snapshot.config)
                    .map(|v| v.map(|v| (v / 4.).tanh()));
                if forecast.target.iter().all(Option::is_none) {
                    forecast.status = "unsupported_coordinates";
                    self.stats.descriptor.unsupported += 1;
                    continue;
                }
                forecast.status = "matched";
                self.stats.descriptor.matched += 1;
                for (coordinate, target) in forecast.target.iter().enumerate() {
                    let Some(target) = target else { continue };
                    self.stats.descriptor.scored_coordinates += 1;
                    for comparison in 0..3 {
                        let error = target - forecast.predictions[comparison][coordinate];
                        forecast.squared_error[comparison][coordinate] = Some(error * error);
                        self.stats.descriptor.squared_error_sum[comparison] += error * error;
                    }
                }
                let Some(model) = model.as_deref_mut() else {
                    self.stats.descriptor.retired += 1;
                    continue;
                };
                for learned in 0..2 {
                    let mut x = forecast.input;
                    if learned == 1 {
                        x[1..6].fill(0.);
                    }
                    let norm = 1. + x.iter().map(|v| v * v).sum::<f64>();
                    for (coordinate, target) in forecast.target.iter().enumerate() {
                        let Some(target) = target else { continue };
                        let error = target - forecast.predictions[learned + 1][coordinate];
                        for (weight, x) in model.descriptor_weights[bus][learned][coordinate]
                            .iter_mut()
                            .zip(x)
                        {
                            *weight = (*weight + 0.1 * error * x / norm).clamp(-8., 8.);
                        }
                    }
                }
                model.descriptor_updates[bus] += 1;
                forecast.learned = true;
                self.stats.descriptor.updated += 1;
            }
            self.descriptor_ready.push(out);
        }
    }

    pub(crate) fn drain_descriptors(&mut self) -> impl Iterator<Item = DescriptorOutcome> + '_ {
        self.descriptor_ready.drain(..)
    }
}
