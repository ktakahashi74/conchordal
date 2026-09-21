//! Private command-to-audio observations and issued prediction diagnostics.

use super::voice::PhonationBatch;

pub(crate) const CAPACITY: usize = 256;
pub(crate) const ACTIVITY_THRESHOLD: f32 = 1e-5;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    Onset,
    Release,
    Shutdown,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct BusOutcome {
    pub routed: bool,
    pub status: &'static str,
    pub first_activity_sample: Option<u64>,
    pub last_activity_sample: Option<u64>,
    pub peak: f32,
    pub rms: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Outcome {
    pub prediction: Option<super::self_prediction::Forecast>,
    pub command_id: u64,
    pub action: ActionKind,
    pub renderer_end_sample: Option<u64>,
    pub source_id: u64,
    pub source_generation: u32,
    pub tone_id: u64,
    pub issued_at_sample: u64,
    pub scheduled_action_sample: Option<u64>,
    pub window_end_sample: Option<u64>,
    pub available_at_sample: u64,
    pub observed_samples: u64,
    pub contiguous_observed_end_sample: Option<u64>,
    pub command_status: &'static str,
    /// Habitat, then presentation; measurements precede mixdown and output guards.
    pub buses: [BusOutcome; 2],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct Snapshot {
    pub body_defaults: Option<super::action_candidates::live::Stats>,
    pub participation_trace: Option<super::participation_trace::Stats>,
    pub prediction: Option<super::self_prediction::Stats>,
    pub sample_rate: u32,
    pub window_samples: u64,
    pub release_window_samples: u64,
    pub activity_threshold: f32,
    pub commands: u64,
    pub completed: u64,
    pub rejected: u64,
    pub pending: usize,
    pub capacity_dropped: u64,
    pub output_dropped: u64,
    pub discontinuities: u64,
    pub latest: Option<Outcome>,
}

struct Pending {
    outcome: Outcome,
    energy: [f64; 2],
    invalid: [bool; 2],
    source_energy: [[f64; super::self_prediction::ENERGY_WINDOWS]; 2],
    source_samples: [[u64; super::self_prediction::ENERGY_WINDOWS]; 2],
}

struct EnergyContext {
    source_id: u64,
    source_generation: u32,
    body_generation: u32,
    issued_at: u64,
    forecast: crate::core::temporal_expectation::TemporalForecast,
}

pub(crate) struct Observer {
    body_defaults: Option<super::action_candidates::live::Bank>,
    trace: Option<Box<super::participation_trace::Bank>>,
    predictor: Option<Box<super::self_prediction::ModelBank>>,
    energy_contexts: Vec<EnergyContext>,
    slots: Vec<Option<Pending>>,
    ready: Vec<Outcome>,
    pub(crate) snapshot: Snapshot,
    next_sample: u64,
    observing_hop: bool,
}

impl Observer {
    pub(crate) fn new(sample_rate: u32) -> Self {
        assert!(sample_rate > 0);
        Self {
            body_defaults: None,
            trace: None,
            predictor: None,
            energy_contexts: Vec::new(),
            slots: (0..CAPACITY).map(|_| None).collect(),
            ready: Vec::with_capacity(CAPACITY),
            snapshot: Snapshot {
                sample_rate,
                window_samples: (sample_rate as u64).div_ceil(50),
                release_window_samples: u64::from(sample_rate) * 2,
                activity_threshold: ACTIVITY_THRESHOLD,
                ..Snapshot::default()
            },
            next_sample: 0,
            observing_hop: true,
        }
    }

    pub(crate) fn enable_predictions(&mut self) {
        self.body_defaults = Some(super::action_candidates::live::Bank::new(
            self.snapshot.sample_rate,
        ));
        self.snapshot.body_defaults = Some(Default::default());
        self.predictor = Some(Box::new(super::self_prediction::ModelBank::new()));
        self.energy_contexts = Vec::with_capacity(crate::temporal_cognition::body::VOICES);
        self.snapshot.prediction = Some(Default::default());
    }

    pub(crate) fn refresh_energy_contexts(
        &mut self,
        now: u64,
        contexts: impl Iterator<
            Item = (
                u64,
                u32,
                u32,
                crate::core::temporal_expectation::TemporalForecast,
            ),
        >,
    ) {
        self.energy_contexts.clear();
        for (source_id, source_generation, body_generation, forecast) in
            contexts.take(self.energy_contexts.capacity())
        {
            self.energy_contexts.push(EnergyContext {
                source_id,
                source_generation,
                body_generation,
                issued_at: now,
                forecast,
            });
        }
    }

    pub(crate) fn candidate_energy_due(&self, owner: (u64, u32), now: u64) -> bool {
        now >= self.next_sample
            && self
                .body_defaults
                .as_ref()
                .is_some_and(|bank| bank.sample_due(owner, now))
    }

    pub(crate) fn freeze_energy_context(
        &self,
        owner: (u64, u32, Option<u32>),
        now: u64,
    ) -> Option<crate::core::temporal_expectation::TemporalForecast> {
        self.energy_contexts
            .iter()
            .find(|context| {
                context.source_id == owner.0
                    && context.source_generation == owner.1
                    && Some(context.body_generation) == owner.2
                    && context.issued_at == now
            })
            .map(|context| context.forecast)
    }

    pub(crate) fn enable_trace(&mut self, config: crate::config::TemporalPrivateTraceConfig) {
        self.trace = Some(Box::new(super::participation_trace::Bank::new(
            self.snapshot.sample_rate,
            config,
        )));
        self.snapshot.participation_trace = Some(Default::default());
    }

    pub(crate) fn trace_context(
        &mut self,
        context: crate::temporal_cognition::reference_inventory::Context,
        now: u64,
    ) {
        if let Some(trace) = self.trace.as_mut() {
            trace.context(context, now);
            self.snapshot.participation_trace = Some(trace.stats);
        }
    }

    pub(crate) fn drain_traces(
        &mut self,
    ) -> impl Iterator<Item = super::participation_trace::Record> + '_ {
        self.trace.iter_mut().flat_map(|t| t.drain())
    }

    pub(crate) fn predict<'a>(
        &mut self,
        slot: (usize, u64),
        mut input: super::self_prediction::Input,
        retained: impl Iterator<
            Item = Option<(
                &'a super::sound::Tone,
                crate::scenario::control::Routing,
                Option<super::self_prediction::ScheduledRelease>,
                super::sound::control_forecast::ControlForecast,
            )>,
        >,
    ) {
        if let Some(predictor) = self.predictor.as_mut()
            && let Some(pending) = self.slots[slot.0].as_mut()
            && pending.outcome.command_id == slot.1
            && pending.outcome.prediction.is_none()
        {
            let start = pending.outcome.scheduled_action_sample.unwrap();
            let width = pending.outcome.window_end_sample.unwrap() - start;
            let mut coherent = [[super::self_prediction::CoherentWindow::new(); 16]; 2];
            let command = super::self_prediction::ToneEnergy {
                amplitude: input.amplitude,
                envelope: input.envelope,
                control: input.control,
                scheduled_release: input.scheduled_release,
                sine: input.sine,
                // Lane products are quadratic in carriers; this runs on the hop path, so
                // multi-lane bodies keep the explicit incoherent prior here.
                bank: None,
            };
            for (bus, windows) in coherent.iter_mut().enumerate() {
                if pending.outcome.buses[bus].routed {
                    for (k, window) in windows.iter_mut().enumerate() {
                        let [left, right] = [k, k + 1].map(|edge| {
                            start + (u128::from(width) * edge as u128).div_ceil(16) as u64
                        });
                        if left < right {
                            let tick = left + (right - left) / 2;
                            command.add_carriers(window, tick, None, command.at(tick, None));
                        }
                    }
                }
            }
            for (index, entry) in retained.take(65).enumerate() {
                if index == 64 {
                    for bus in &mut input.retained_energy {
                        bus.complete = false;
                        bus.fixed_energy.fill(0.);
                    }
                    break;
                }
                for bus in &mut input.retained_energy {
                    bus.scanned_entries += 1;
                }
                let Some((tone, routing, plan, control)) = entry else {
                    continue;
                };
                let (_, amplitude, envelope) = tone.prediction_parameters(None);
                let projected = super::self_prediction::ToneEnergy {
                    amplitude,
                    envelope,
                    control: Some(control),
                    scheduled_release: plan,
                    sine: tone.prediction_sine(control.issued_at),
                    bank: None,
                };
                for (bus, routed) in [routing.to_habitat, routing.to_presentation]
                    .into_iter()
                    .enumerate()
                {
                    if !routed {
                        continue;
                    }
                    input.retained_energy[bus].included_tones += 1;
                    for k in 0..super::self_prediction::ENERGY_WINDOWS {
                        let [left, right] = [k, k + 1].map(|edge| {
                            start
                                + (u128::from(width) * edge as u128)
                                    .div_ceil(super::self_prediction::ENERGY_WINDOWS as u128)
                                    as u64
                        });
                        if left == right {
                            continue;
                        }
                        let tick = left + (right - left) / 2;
                        projected.add_carriers(
                            &mut coherent[bus][k],
                            tick,
                            None,
                            projected.at(tick, None),
                        );
                        if let Some(energy) = projected.at(tick, None) {
                            input.retained_energy[bus].fixed_energy[k] += energy;
                        } else {
                            input.retained_energy[bus].control_supported[k] = false;
                        }
                    }
                }
            }
            for (bus, windows) in coherent.iter().enumerate() {
                if input.retained_energy[bus].complete {
                    for (k, window) in windows.iter().enumerate() {
                        let [left, right] = [k, k + 1].map(|edge| {
                            start + (u128::from(width) * edge as u128).div_ceil(16) as u64
                        });
                        input.coherent_energy[bus][k] =
                            window.mean(left, right, left + (right - left) / 2);
                    }
                }
            }
            pending.outcome.prediction = predictor.issue(&pending.outcome, input);
            if let Some(prediction) = pending.outcome.prediction.as_mut() {
                let external = self.energy_contexts.iter().find(|context| {
                    context.source_id == pending.outcome.source_id
                        && context.source_generation == pending.outcome.source_generation
                        && context.body_generation == prediction.body_generation
                        && context.issued_at == pending.outcome.issued_at_sample
                });
                let ratio = prediction.source_energy[0].preview_ratio(
                    external.map(|context| &context.forecast),
                    self.snapshot.sample_rate,
                );
                predictor.stats.energy.ratio_previews += 1;
                if ratio.status == "supported" {
                    predictor.stats.energy.ratio_supported += 1;
                } else {
                    predictor.stats.energy.ratio_unknown += 1;
                }
                prediction.energy_ratio = Some(ratio);
            }
            if let Some(trace) = self.trace.as_mut() {
                trace.preview(&pending.outcome);
                self.snapshot.participation_trace = Some(trace.stats);
            }
            self.snapshot.prediction = Some(predictor.stats);
        }
    }

    pub(crate) fn begin_hop(&mut self, now: u64) {
        if now != self.next_sample {
            self.snapshot.discontinuities += 1;
        }
        // Backward/replayed input must not add evidence twice.
        self.observing_hop = now >= self.next_sample;
    }

    pub(crate) fn body_defaults(&mut self) -> Option<&mut super::action_candidates::live::Bank> {
        self.body_defaults.as_mut().filter(|_| self.observing_hop)
    }

    pub(crate) fn drain_body_defaults(
        &mut self,
    ) -> impl Iterator<Item = super::action_candidates::live::Record> + '_ {
        self.body_defaults.iter_mut().flat_map(|bank| bank.drain())
    }

    pub(crate) fn drain_candidate_energy(
        &mut self,
    ) -> impl Iterator<Item = super::action_candidates::energy::Record> + '_ {
        self.body_defaults
            .iter_mut()
            .flat_map(|bank| bank.energy.drain())
    }

    pub(crate) fn freeze_onset_trace(
        &self,
        slot: (usize, u64),
    ) -> Option<super::participation_trace::Frozen> {
        let p = self.slots.get(slot.0)?.as_ref()?;
        let o = &p.outcome;
        if o.command_id != slot.1
            || o.action != ActionKind::Onset
            || o.command_status != "accepted"
            || !o.buses[0].routed
        {
            return None;
        }
        self.trace.as_ref()?.freeze_onset(
            o.command_id,
            (o.source_id, o.source_generation),
            o.issued_at_sample,
        )
    }

    pub(crate) fn freeze_release_trace(
        &self,
        source: (u64, u32),
        issued: u64,
        period: Option<f64>,
    ) -> Option<super::participation_trace::Frozen> {
        self.trace.as_ref()?.freeze_release(source, issued, period)
    }

    pub(crate) fn observe_body(&mut self, snapshot: &crate::temporal_cognition::body::Snapshot) {
        if let Some(bank) = &mut self.body_defaults {
            bank.bindings.clear();
            bank.bindings
                .extend(crate::temporal_cognition::action_profiles::consumer::bindings(snapshot));
        }
        if let Some(predictor) = self.predictor.as_mut() {
            predictor.observe_body(snapshot);
            self.snapshot.prediction = Some(predictor.stats);
        }
    }

    pub(crate) fn shared_candidates(
        &mut self,
        shared: [Option<std::sync::Arc<crate::temporal_cognition::action_profiles::consumer::Publication>>;
            2],
    ) {
        if let Some(bank) = &mut self.body_defaults {
            bank.shared = shared;
        }
    }

    pub(crate) fn drain_descriptor_predictions(
        &mut self,
    ) -> impl Iterator<Item = super::self_prediction::DescriptorOutcome> + '_ {
        self.predictor
            .iter_mut()
            .flat_map(|p| p.drain_descriptors())
    }

    pub(crate) fn command(
        &mut self,
        batch: &PhonationBatch,
        tone_id: u64,
        scheduled: Option<u64>,
        now: u64,
        status: &'static str,
        action: ActionKind,
    ) -> Option<(usize, u64)> {
        self.snapshot.commands += 1;
        let window = if action == ActionKind::Onset {
            self.snapshot.window_samples
        } else {
            self.snapshot.release_window_samples
        };
        let mut outcome = Outcome {
            prediction: None,
            command_id: self.snapshot.commands,
            action,
            renderer_end_sample: None,
            source_id: batch.source_id,
            source_generation: batch.source_generation,
            tone_id,
            issued_at_sample: now,
            scheduled_action_sample: scheduled,
            window_end_sample: scheduled.map(|start| start.saturating_add(window)),
            available_at_sample: now,
            observed_samples: 0,
            contiguous_observed_end_sample: scheduled,
            command_status: status,
            buses: [batch.routing.to_habitat, batch.routing.to_presentation].map(|routed| {
                BusOutcome {
                    routed,
                    status: if routed { "unobserved" } else { "not_routed" },
                    first_activity_sample: None,
                    last_activity_sample: None,
                    peak: 0.0,
                    rms: None,
                }
            }),
        };
        if status != "accepted" {
            self.snapshot.rejected += 1;
            self.publish(outcome);
            return None;
        }
        let Some(slot) = self.slots.iter().position(Option::is_none) else {
            self.snapshot.capacity_dropped += 1;
            for bus in &mut outcome.buses {
                if bus.routed {
                    bus.status = "capacity_dropped";
                }
            }
            self.publish(outcome);
            return None;
        };
        if let Some(predictor) = self.predictor.as_mut() {
            predictor.intervene(
                batch.source_id,
                batch.source_generation,
                scheduled.unwrap_or(now),
            );
        }
        if let Some(trace) = self.trace.as_mut() {
            trace.issue(&outcome, batch.intrinsic_period_sec);
            self.snapshot.participation_trace = Some(trace.stats);
        }
        self.slots[slot] = Some(Pending {
            outcome,
            energy: [0.0; 2],
            invalid: [false; 2],
            source_energy: [[0.; super::self_prediction::ENERGY_WINDOWS]; 2],
            source_samples: [[0; super::self_prediction::ENERGY_WINDOWS]; 2],
        });
        self.snapshot.pending += 1;
        Some((slot, self.snapshot.commands))
    }

    /// Return false when a renderer's old slot no longer owns this command.
    pub(crate) fn sample(
        &mut self,
        slot: (usize, u64),
        owner: (u64, u32, u64),
        at: u64,
        value: f32,
        ended: bool,
    ) -> bool {
        let Some(pending) = self.slots[slot.0].as_mut() else {
            return false;
        };
        let out = &mut pending.outcome;
        if out.command_id != slot.1 || (out.source_id, out.source_generation, out.tone_id) != owner
        {
            return false;
        }
        if !self.observing_hop
            || at < out.scheduled_action_sample.unwrap()
            || at >= out.window_end_sample.unwrap()
        {
            return true;
        }
        if ended && out.renderer_end_sample.is_none() {
            out.renderer_end_sample = Some(at);
        }
        for (i, bus) in out.buses.iter_mut().enumerate() {
            if !bus.routed {
                continue;
            }
            if !value.is_finite() {
                pending.invalid[i] = true;
                continue;
            }
            pending.energy[i] += f64::from(value).powi(2);
            bus.peak = bus.peak.max(value.abs());
            if value.abs() > ACTIVITY_THRESHOLD {
                bus.first_activity_sample.get_or_insert(at);
                bus.last_activity_sample = Some(at);
            }
        }
        true
    }

    pub(crate) fn end_hop(&mut self, start: u64, end: u64) {
        assert!(end >= start);
        if !self.observing_hop {
            return;
        }
        for i in 0..self.slots.len() {
            let Some(p) = self.slots[i].as_mut() else {
                continue;
            };
            let left = start.max(p.outcome.scheduled_action_sample.unwrap());
            let right = end.min(p.outcome.window_end_sample.unwrap());
            // An absent/finished tone contributes known zero in a rendered interval.
            p.outcome.observed_samples += right.saturating_sub(left);
            if let Some(prefix) = p.outcome.contiguous_observed_end_sample.as_mut()
                && start <= *prefix
                && right > *prefix
            {
                *prefix = right;
            }
            if end >= p.outcome.window_end_sample.unwrap() {
                self.complete(i, end);
            }
        }
        self.next_sample = end;
        if let Some(bank) = self.body_defaults.as_mut() {
            bank.end_hop();
            self.snapshot.body_defaults = Some(bank.stats);
        }
        if let Some(trace) = self.trace.as_mut() {
            let watermark = self
                .slots
                .iter()
                .flatten()
                .filter_map(|p| p.outcome.scheduled_action_sample)
                .min()
                .unwrap_or(end);
            trace.seal(watermark.min(end), end);
            self.snapshot.participation_trace = Some(trace.stats);
        }
    }

    pub(crate) fn observe_source_energy(
        &mut self,
        capture: &crate::temporal_cognition::body::Capture,
    ) {
        if !self.observing_hop {
            return;
        }
        for pending in self.slots.iter_mut().flatten() {
            let Some(forecast) = pending.outcome.prediction.as_ref() else {
                continue;
            };
            let Some((now, audio)) = capture.source_audio(
                forecast.source_slot,
                (
                    pending.outcome.source_id,
                    pending.outcome.source_generation,
                    forecast.body_generation,
                ),
            ) else {
                continue;
            };
            let start = pending.outcome.scheduled_action_sample.unwrap();
            let end = pending.outcome.window_end_sample.unwrap();
            for (bus, samples) in audio.into_iter().enumerate() {
                let Some(samples) = samples else { continue };
                let left = start.max(now);
                let right = end.min(now.saturating_add(samples.len() as u64));
                if left >= right {
                    continue;
                }
                for (offset, value) in samples[(left - now) as usize..(right - now) as usize]
                    .iter()
                    .enumerate()
                {
                    if value.is_finite() {
                        let bin = ((left - start + offset as u64)
                            * super::self_prediction::ENERGY_WINDOWS as u64
                            / (end - start)) as usize;
                        pending.source_energy[bus][bin] += f64::from(*value).powi(2);
                        pending.source_samples[bus][bin] += 1;
                    }
                }
            }
        }
    }

    pub(crate) fn finish(&mut self) {
        if let Some(bank) = self.body_defaults.as_mut() {
            bank.energy.finish();
            bank.stats.candidate_energy = bank.energy.stats;
            self.snapshot.body_defaults = Some(bank.stats);
        }
        for i in 0..self.slots.len() {
            if self.slots[i].is_some() {
                self.complete(i, self.next_sample);
            }
        }
        if let Some(trace) = self.trace.as_mut() {
            trace.seal(self.next_sample, self.next_sample);
            self.snapshot.participation_trace = Some(trace.stats);
        }
    }

    fn complete(&mut self, slot: usize, available: u64) {
        let mut p = self.slots[slot].take().unwrap();
        p.outcome.available_at_sample = available;
        let window =
            p.outcome.window_end_sample.unwrap() - p.outcome.scheduled_action_sample.unwrap();
        let complete =
            p.outcome.observed_samples == window && p.outcome.command_status == "accepted";
        for (i, bus) in p.outcome.buses.iter_mut().enumerate() {
            if !bus.routed {
                continue;
            }
            bus.status = if p.invalid[i] {
                "invalid_sample"
            } else if !complete {
                "incomplete"
            } else if bus.first_activity_sample.is_some() {
                "activity"
            } else if bus.peak > 0.0 {
                "below_threshold"
            } else {
                "silence"
            };
            if complete && !p.invalid[i] {
                bus.rms = Some((p.energy[i] / window as f64).sqrt());
            }
        }
        if let Some(predictor) = self.predictor.as_mut() {
            if let Some(forecast) = p.outcome.prediction.as_mut() {
                for (bus, energy) in forecast.source_energy.iter_mut().enumerate() {
                    for (k, [left, right]) in energy.windows.iter().copied().enumerate() {
                        let width = right - left;
                        if width > 0
                            && p.source_samples[bus][k] == width
                            && p.outcome.command_status == "accepted"
                        {
                            energy.target[k] = Some(p.source_energy[bus][k] / width as f64);
                        }
                    }
                }
            }
            if p.outcome.command_status != "accepted" {
                predictor.cancel_descriptor(p.outcome.command_id);
            }
            predictor.observe(&mut p.outcome);
            self.snapshot.prediction = Some(predictor.stats);
        }
        self.snapshot.pending -= 1;
        self.snapshot.completed += 1;
        self.publish(p.outcome);
    }

    pub(crate) fn interrupt(
        &mut self,
        slot: (usize, u64),
        owner: (u64, u32, u64),
        now: u64,
        status: &'static str,
    ) {
        if let Some(pending) = self.slots[slot.0].as_mut()
            && pending.outcome.command_id == slot.1
            && (
                pending.outcome.source_id,
                pending.outcome.source_generation,
                pending.outcome.tone_id,
            ) == owner
        {
            pending.outcome.command_status = status;
            self.complete(slot.0, now);
        }
    }

    fn publish(&mut self, outcome: Outcome) {
        if let Some(trace) = self.trace.as_mut() {
            trace.complete(outcome);
        }
        self.snapshot.latest = Some(outcome);
        if self.ready.len() < self.ready.capacity() {
            self.ready.push(outcome);
        } else {
            self.snapshot.output_dropped += 1;
        }
    }

    pub(crate) fn drain(&mut self) -> impl Iterator<Item = Outcome> + '_ {
        self.ready.drain(..)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emitted_samples_determine_activity_and_energy_not_the_command_time() {
        let mut observer = Observer::new(1000);
        let batch = PhonationBatch {
            source_id: 9,
            source_generation: 4,
            ..Default::default()
        };
        let slot = observer
            .command(&batch, 1, Some(3), 0, "accepted", ActionKind::Onset)
            .unwrap();
        observer.begin_hop(0);
        observer.sample(slot, (9, 4, 1), 7, 0.4, false);
        observer.sample(slot, (9, 4, 1), 8, -0.3, false);
        observer.end_hop(0, 10);
        assert_eq!(observer.snapshot.completed, 0);
        observer.begin_hop(10);
        observer.end_hop(10, 30);
        let out = observer.drain().next().unwrap();
        assert_eq!(
            (
                out.issued_at_sample,
                out.scheduled_action_sample,
                out.window_end_sample
            ),
            (0, Some(3), Some(23))
        );
        assert_eq!(
            (
                out.observed_samples,
                out.available_at_sample,
                out.source_generation
            ),
            (20, 30, 4)
        );
        for bus in out.buses {
            assert_eq!(bus.status, "activity");
            assert_eq!(bus.first_activity_sample, Some(7));
            assert!((bus.rms.unwrap() - (0.25_f64 / 20.).sqrt()).abs() < 1e-8);
        }
    }

    #[test]
    fn silence_below_threshold_gaps_and_unrendered_tail_remain_distinct() {
        let batch = PhonationBatch::default();
        for (value, gap, eof, status, coverage) in [
            (0., false, false, "silence", 20),
            (1e-6, false, false, "below_threshold", 20),
            (0., true, false, "incomplete", 15),
            (0., false, true, "incomplete", 5),
            (f32::NAN, false, false, "invalid_sample", 20),
        ] {
            let mut observer = Observer::new(1000);
            let slot = observer
                .command(&batch, 1, Some(0), 0, "accepted", ActionKind::Onset)
                .unwrap();
            observer.begin_hop(0);
            observer.sample(slot, (0, 0, 1), 0, value, false);
            observer.end_hop(0, 5);
            if !eof {
                let start = if gap { 10 } else { 5 };
                observer.begin_hop(start);
                observer.end_hop(start, 20);
            }
            observer.finish();
            let out = observer.drain().next().unwrap();
            assert_eq!(out.observed_samples, coverage);
            assert_eq!(out.buses[0].status, status);
            assert_eq!(out.buses[0].first_activity_sample, None);
            assert_eq!(
                out.buses[0].rms.is_some(),
                !gap && !eof && value.is_finite()
            );
            assert_eq!(observer.snapshot.pending, 0);
        }
    }

    #[test]
    fn slot_reuse_replay_and_routing_cannot_borrow_another_commands_evidence() {
        let mut observer = Observer::new(1000);
        let mut batch = PhonationBatch {
            source_id: 1,
            ..Default::default()
        };
        let slot = observer
            .command(&batch, 1, Some(0), 0, "accepted", ActionKind::Onset)
            .unwrap();
        observer.begin_hop(0);
        observer.end_hop(0, 20);
        observer.drain().for_each(drop);
        batch.source_id = 2;
        batch.source_generation = 7;
        batch.routing.to_habitat = false;
        let next = observer
            .command(&batch, 1, Some(20), 20, "accepted", ActionKind::Onset)
            .unwrap();
        assert_eq!(next.0, slot.0);
        assert_ne!(next.1, slot.1);
        observer.begin_hop(0);
        observer.sample(slot, (2, 7, 1), 20, 0.9, false);
        observer.end_hop(0, 20);
        observer.begin_hop(20);
        assert!(!observer.sample(slot, (1, 0, 1), 20, 0.9, false));
        observer.sample(next, (2, 7, 1), 21, 0.2, false);
        observer.end_hop(20, 40);
        let out = observer.drain().next().unwrap();
        assert_eq!(out.source_generation, 7);
        assert_eq!(out.buses[0].status, "not_routed");
        assert_eq!(out.buses[0].rms, None);
        assert_eq!(out.buses[1].first_activity_sample, Some(21));
        assert_eq!(out.buses[1].peak, 0.2);
        assert_eq!(out.observed_samples, 20);
        assert_eq!(observer.snapshot.discontinuities, 1);
    }

    #[test]
    fn release_windows_preserve_unknown_tail_and_reused_slot_identity() {
        let batch = PhonationBatch::default();
        for gap in [false, true] {
            let mut observer = Observer::new(1000);
            let old = observer
                .command(&batch, 1, Some(0), 0, "accepted", ActionKind::Onset)
                .unwrap();
            observer.begin_hop(0);
            observer.end_hop(0, 20);
            observer.drain().for_each(drop);
            let release = observer
                .command(&batch, 1, Some(20), 20, "accepted", ActionKind::Release)
                .unwrap();
            assert_eq!(old.0, release.0);
            assert_ne!(old.1, release.1);
            observer.begin_hop(20);
            assert!(!observer.sample(old, (0, 0, 1), 20, 100., false));
            assert!(!observer.sample(release, (0, 1, 1), 20, 100., false));
            observer.sample(release, (0, 0, 1), 21, 0.5, false);
            observer.end_hop(20, 30);
            observer.begin_hop(if gap { 40 } else { 30 });
            observer.sample(release, (0, 0, 1), 2019, 0.25, false);
            observer.end_hop(if gap { 40 } else { 30 }, 2020);
            let out = observer.drain().next().unwrap();
            assert_eq!(out.renderer_end_sample, None);
            assert_eq!(out.buses[0].last_activity_sample, Some(2019));
            assert_eq!(out.buses[0].peak, 0.5);
            assert_eq!(
                out.buses[0].status,
                if gap { "incomplete" } else { "activity" }
            );
            assert_eq!(out.buses[0].rms.is_some(), !gap);
        }
    }

    #[test]
    fn rejection_and_bounded_capacity_never_create_observation_credit() {
        let mut observer = Observer::new(1000);
        let batch = PhonationBatch::default();
        let slots = observer.slots.as_ptr();
        let output = observer.ready.as_ptr();
        observer.command(&batch, 1, None, 0, "missing_spec", ActionKind::Onset);
        assert_eq!(observer.snapshot.pending, 0);
        assert_eq!(observer.drain().next().unwrap().observed_samples, 0);
        for id in 0..CAPACITY + 1 {
            observer.command(&batch, id as u64, Some(0), 0, "accepted", ActionKind::Onset);
        }
        assert_eq!(observer.snapshot.capacity_dropped, 1);
        observer.begin_hop(0);
        observer.end_hop(0, 20);
        assert_eq!(observer.snapshot.completed, CAPACITY as u64);
        assert_eq!(observer.snapshot.output_dropped, 1);
        assert_eq!(
            observer.snapshot.commands,
            observer.snapshot.completed
                + observer.snapshot.rejected
                + observer.snapshot.capacity_dropped
        );
        assert_eq!(observer.slots.as_ptr(), slots);
        assert_eq!(observer.ready.as_ptr(), output);
        assert_eq!(observer.drain().count(), CAPACITY);
    }
}
