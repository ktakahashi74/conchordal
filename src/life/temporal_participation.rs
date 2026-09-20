//! Intrinsic pace, acoustic cadence, and memory of executed actions.

use crate::core::temporal_expectation::{OwnSoundHistory, TemporalForecast};
use crate::core::temporal_history::AuditoryHistorySnapshot;
use crate::life::sound::ToneAdsr;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::collections::VecDeque;

const CONTEXT_DELAY_SEC: f64 = 0.08;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct ParticipationContextPrediction {
    pub(crate) external_energy_footprint:
        Option<crate::core::temporal_expectation::ExternalEnergyFootprint>,
    pub(crate) onset_frame: u64,
    pub(crate) forecast_observed_frame: u64,
    pub(crate) decision_external_history: Option<AuditoryHistorySnapshot>,
    pub(crate) energy_prediction_model: &'static str,
    pub(crate) target_start_frames: [u64; 2],
    pub(crate) target_end_frames: [u64; 2],
    pub(crate) pred_external_band_energy: [[f32; 3]; 2],
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct ParticipationContextUpdate {
    #[serde(flatten)]
    pub(crate) prediction: ParticipationContextPrediction,
    pub(crate) observed_through_frame: u64,
    pub(crate) status: &'static str,
    pub(crate) observed_external_band_energy: Option<[[f32; 3]; 2]>,
    pub(crate) memory_external_band_energy: Option<[[f32; 3]; 2]>,
}

#[cfg(test)]
#[path = "temporal_participation_assay.rs"]
mod acoustic_assay;

pub(crate) struct TemporalParticipation {
    fs: f64,
    intrinsic_period_frames: f64,
    period_frames: f64,
    last_reference_frame: Option<u64>,
    due_frame: f64,
    planned: Option<(f64, Option<ParticipationContextPrediction>)>,
    pending_contexts: VecDeque<ParticipationContextPrediction>,
    memory: Option<[[f32; 3]; 2]>,
    last_onset: Option<u64>,
    coupling: f32,
    own_band_energy: [f32; 3],
    overlap_sensitivity: f32,
    flow_depth: f32,
    cluster_remaining: u8,
    skipped_cycles: u32,
    sound_hold_sec: f32,
    sound_adsr: Option<ToneAdsr>,
    rng: SmallRng,
}

impl TemporalParticipation {
    pub(crate) fn new(
        fs: u32,
        base_rate_hz: f32,
        coupling: f32,
        start: u64,
        seed: u64,
        hop_frames: u64,
    ) -> Self {
        assert!(fs > 0 && base_rate_hz.is_finite() && base_rate_hz > 0.0);
        let window = (fs as f64 / 100.0).round().max(1.0) as u64;
        let min_onset =
            (fs as f64 / crate::life::phonation_engine::MAX_ONSET_RATE_HZ as f64).ceil() as u64;
        // A hop can schedule all its onsets before any resulting audio is observed.
        let pending_span = hop_frames + (CONTEXT_DELAY_SEC * fs as f64).ceil() as u64 + 2 * window;
        let pending_capacity = pending_span.div_ceil(min_onset) as usize + 1;
        let period = fs as f64 / base_rate_hz as f64;
        let mut rng = SmallRng::seed_from_u64(seed);
        Self {
            fs: fs as f64,
            intrinsic_period_frames: period,
            period_frames: period,
            last_reference_frame: None,
            due_frame: start as f64 + rng.random_range(0.0..1.0) * period,
            planned: None,
            pending_contexts: VecDeque::with_capacity(pending_capacity),
            memory: None,
            last_onset: None,
            coupling: coupling.clamp(0.0, 1.0),
            own_band_energy: [0.0; 3],
            overlap_sensitivity: 0.0,
            flow_depth: 0.0,
            cluster_remaining: 0,
            skipped_cycles: 0,
            sound_hold_sec: 0.08,
            sound_adsr: None,
            rng,
        }
    }

    pub(crate) fn needs_forecast(&self, now: u64, end: u64) -> bool {
        self.coupling > 0.0
            && self.planned.is_none_or(|(at, _)| (at.round() as u64) < now)
            && self.due_frame - (end as f64) < (0.2 * self.period_frames).min(0.06 * self.fs)
    }

    pub(crate) fn opportunity(&self, now: u64) -> Option<super::action_candidates::Opportunity> {
        use super::action_candidates::{Opportunity, OpportunityBasis};
        let (at, basis) = match self.planned {
            Some((at, _)) => (at, OpportunityBasis::ParticipationPlanned),
            None => (self.due_frame, OpportunityBasis::ParticipationDue),
        };
        // Stale state must be advanced by the real policy, never by this observer.
        (at.is_finite() && at >= 0.0 && at.round() < u64::MAX as f64)
            .then_some(at.round() as u64)
            .filter(|at| *at >= now)
            .map(|at| Opportunity {
                issued_at: now,
                at,
                basis,
            })
    }

    pub(crate) fn intrinsic_due_for_selected(&self, tick: u64) -> Option<u64> {
        let (planned, _) = self.planned?;
        (planned.round() as u64 == tick
            && self.due_frame.is_finite()
            && self.due_frame >= 0.0
            && self.due_frame.round() < u64::MAX as f64)
            .then_some(self.due_frame.round() as u64)
    }

    pub(crate) fn set_sound_duration(&mut self, hold_sec: f32, adsr: Option<ToneAdsr>) {
        self.sound_hold_sec = hold_sec.max(1.0 / self.fs as f32);
        self.sound_adsr = adsr;
    }

    /// Periodic participation follows recurrence; renewal keeps its intrinsic scale.
    pub(crate) fn update_reference(&mut self, now: u64, forecast: Option<&TemporalForecast>) {
        if self.coupling == 0.0 || self.flow_depth > 0.0 || self.planned.is_some() {
            return;
        }
        let forecast = forecast.filter(|f| f.contrast_at(now as f64).is_some());
        let frame = forecast.map_or(now, TemporalForecast::observed_frame);
        if self.last_reference_frame.is_some_and(|last| frame <= last) {
            return;
        }
        let elapsed = self
            .last_reference_frame
            .map_or(self.intrinsic_period_frames, |last| {
                frame.saturating_sub(last) as f64
            });
        self.last_reference_frame = Some(frame);
        let (target, support) = forecast
            .and_then(|f| f.period_reference((self.intrinsic_period_frames / self.fs) as f32))
            .map_or((self.intrinsic_period_frames, 1.0), |(sec, support)| {
                (sec as f64 * self.fs, support as f64)
            });
        let alpha = -(-elapsed * self.coupling as f64 * support
            / (2.0 * self.intrinsic_period_frames))
            .exp_m1();
        let previous = self.period_frames;
        self.period_frames += alpha * (target - previous);
        self.period_frames = self
            .period_frames
            .max(self.fs / crate::life::phonation_engine::MAX_ONSET_RATE_HZ as f64);
        // Rescale remaining cycle time; there is no common phase to reset toward.
        if self.due_frame >= now as f64 {
            self.due_frame =
                now as f64 + (self.due_frame - now as f64) * self.period_frames / previous;
        }
    }

    /// Positive sensitivity avoids energetic overlap; negative explicitly seeks it.
    pub(crate) fn set_overlap(&mut self, sensitivity: f32, own_band_energy: [f32; 3]) {
        assert!(own_band_energy.iter().all(|e| e.is_finite() && *e >= 0.0));
        assert!(sensitivity.is_finite());
        self.overlap_sensitivity = sensitivity.clamp(-1.0, 1.0);
        self.own_band_energy = own_band_energy;
    }

    /// Selecting a candidate does not establish that it was sounded.
    pub(crate) fn candidate(
        &mut self,
        now: u64,
        end: u64,
        onset_allowed: bool,
        forecast: Option<&TemporalForecast>,
    ) -> Option<u64> {
        if end <= now {
            return None;
        }
        if let Some((planned, _)) = self.planned
            && (planned.round() as u64) < now
        {
            self.planned = None;
            self.due_frame = planned + self.period_frames;
        }
        while self.planned.is_none() {
            if self.due_frame < now as f64 {
                let missed = ((now as f64 - self.due_frame) / self.period_frames).ceil();
                self.due_frame += missed * self.period_frames;
            }
            let width = (0.2 * self.period_frames).min(0.06 * self.fs);
            if self.due_frame - end as f64 >= width {
                return None;
            }
            let mut best = self.due_frame;
            let mut best_context = None;
            let mut best_cost = f32::INFINITY;
            // Missing forecast coverage must not make an unscored candidate cheaper.
            let forecast = forecast.filter(|f| {
                f.energy_window_after((self.due_frame - width).max(now as f64).round() as u64)
                    .is_some()
                    && f.energy_window_after(
                        (self.due_frame + self.period_frames).round() as u64
                            + (CONTEXT_DELAY_SEC * self.fs).round() as u64,
                    )
                    .is_some()
            });
            // Integrate the planned hold and release. This is an envelope proxy;
            // the acoustic body and render modulator can add their own decay.
            let release = self.sound_adsr.map_or(0.0, |a| a.release_sec.max(0.0));
            let duration = self.sound_hold_sec + release;
            let footprint: [(f64, f32); 64] = std::array::from_fn(|i| {
                let age = (i as f32 + 0.5) * duration / 64.0;
                let gain = self.sound_adsr.map_or(1.0, |a| {
                    let attack = a
                        .attack_sec
                        .max(1.0 / self.fs as f32)
                        .min(self.sound_hold_sec);
                    let before_release = if age < attack {
                        age / attack
                    } else if a.decay_sec > 0.0 {
                        a.sustain_level
                            + (1.0 - a.sustain_level)
                                * (-6.908 * (age - attack) / a.decay_sec).exp()
                    } else {
                        a.sustain_level
                    };
                    before_release
                        * if age < self.sound_hold_sec {
                            1.0
                        } else {
                            ((duration - age) / release.max(1e-12)).clamp(0.0, 1.0)
                        }
                });
                (age as f64 * self.fs, gain * gain)
            });
            let own_total: f32 = self.own_band_energy.iter().sum();
            let footprint_mass: f32 = footprint.iter().map(|(_, power)| power).sum();
            let earliest = self.last_onset.map_or(now as f64, |last| {
                (last as f64
                    + (self.fs / crate::life::phonation_engine::MAX_ONSET_RATE_HZ as f64).ceil())
                .max(now as f64)
            });
            // The exact bodily due time is a candidate; search spacing is not a beat grid.
            for offset in -2..=20 {
                let shift = if offset < 0 {
                    offset as f64 * width / 2.0
                } else {
                    offset as f64 * self.period_frames / 20.0
                };
                let at = if offset == 0 {
                    self.due_frame.max(earliest)
                } else {
                    self.due_frame + shift
                };
                if at < earliest {
                    continue;
                }
                let displacement = (at - self.due_frame) / self.period_frames;
                let context = forecast.and_then(|f| {
                    let onset = at.round() as u64;
                    let (start, end, present) = f.energy_window_after(onset)?;
                    let (after_start, after_end, after) = f.energy_window_after(
                        onset + (CONTEXT_DELAY_SEC * self.fs).round() as u64,
                    )?;
                    Some(ParticipationContextPrediction {
                        external_energy_footprint: None,
                        onset_frame: onset,
                        forecast_observed_frame: f.observed_frame(),
                        decision_external_history: f.observed_history,
                        energy_prediction_model: f.energy_prediction_model,
                        target_start_frames: [start, after_start],
                        target_end_frames: [end, after_end],
                        pred_external_band_energy: [present, after],
                    })
                });
                let mut cost = (displacement * displacement) as f32;
                if onset_allowed && let (Some(memory), Some(context)) = (self.memory, context) {
                    let predicted = context.pred_external_band_energy;
                    let a_mass: f32 = memory.iter().flatten().sum();
                    let b_mass: f32 = predicted.iter().flatten().sum();
                    // Squared Hellinger distance between energy shapes, not confidence.
                    // Observed silence has its own context rather than a uniform spectrum.
                    let distance = if a_mass <= 1e-12 || b_mass <= 1e-12 {
                        u8::from((a_mass > 1e-12) != (b_mass > 1e-12)) as f32
                    } else {
                        0.5 * memory
                            .iter()
                            .flatten()
                            .zip(predicted.iter().flatten())
                            .map(|(a, b)| ((a / a_mass).sqrt() - (b / b_mass).sqrt()).powi(2))
                            .sum::<f32>()
                    };
                    cost += self.coupling * distance;
                }
                if onset_allowed
                    && self.overlap_sensitivity != 0.0
                    && let Some(forecast) = forecast
                {
                    let mut overlap = 0.0;
                    for (delay, power) in footprint {
                        if let Some(energy) = forecast.sustained_energy_at(at + delay) {
                            for (own, other) in self.own_band_energy.into_iter().zip(energy) {
                                let own = own * power;
                                overlap += own * other / (own + other + 1e-12);
                            }
                        }
                    }
                    // The caller forecasts the observed external waveform's energy.
                    cost += self.coupling * 6.0 * self.overlap_sensitivity * overlap
                        / (own_total * footprint_mass).max(1e-12);
                }
                if cost < best_cost {
                    best_cost = cost;
                    best = at;
                    best_context = context;
                }
            }
            // Skipping is a participation decision, not an executed sound or a reward.
            // Its increasing cost prevents energetic overlap from prescribing silence.
            if onset_allowed
                && forecast.is_some()
                && own_total > 1e-12
                && self.overlap_sensitivity > 0.0
                && best_cost > 1.0 + self.skipped_cycles as f32
            {
                self.skipped_cycles = self.skipped_cycles.saturating_add(1);
                self.due_frame += self.next_interval();
                continue;
            }
            if let (Some(context), Some(forecast)) = (&mut best_context, forecast) {
                context.external_energy_footprint = Some(
                    forecast.external_footprint(
                        [
                            context.onset_frame,
                            context
                                .onset_frame
                                .saturating_add((f64::from(duration) * self.fs).ceil() as u64),
                        ],
                        forecast
                            .observed_frame()
                            .saturating_add((4. * self.fs) as u64),
                    ),
                );
            }
            self.planned = Some((best, best_context));
        }
        let (planned, _) = self.planned?;
        let onset = planned.round() as u64;
        if onset >= end {
            return None;
        }
        Some(onset.max(now))
    }

    #[cfg(test)]
    pub(crate) fn period_frames(&self) -> f64 {
        self.period_frames
    }

    pub(crate) fn set_flow_depth(&mut self, depth: f32) {
        self.flow_depth = depth.clamp(0.0, 1.0);
        if self.flow_depth > 0.0 {
            // Keep pending actions; only subsequent renewal intervals change scale.
            self.period_frames = self.intrinsic_period_frames;
            self.last_reference_frame = None;
        }
    }

    pub(crate) fn update_parameters(&mut self, base_rate_hz: f32, coupling: f32, flow_depth: f32) {
        let intrinsic = self.fs / base_rate_hz as f64;
        if intrinsic != self.intrinsic_period_frames || coupling == 0.0 {
            self.period_frames = intrinsic;
        }
        self.intrinsic_period_frames = intrinsic;
        self.coupling = coupling;
        self.set_flow_depth(flow_depth);
    }

    pub(crate) fn resolve(&mut self, tick: u64, sounded: bool) {
        let (planned, context) = self
            .planned
            .take()
            .expect("resolve requires a selected candidate");
        assert_eq!(
            planned.round() as u64,
            tick,
            "resolve must match the selected onset"
        );
        let onset = tick;
        if sounded {
            self.skipped_cycles = 0;
            // A chosen delay changes participation, not the body's tempo prior.
            self.last_onset = Some(onset);
            if let Some(context) = context {
                assert!(
                    self.pending_contexts.len() < self.pending_contexts.capacity(),
                    "context backlog exceeds the configured hop and onset rate"
                );
                self.pending_contexts.push_back(context);
            }
        }
        self.due_frame = planned + self.next_interval();
    }

    pub(crate) fn observe_context(
        &mut self,
        own: &OwnSoundHistory,
        mut emit: impl FnMut(ParticipationContextUpdate),
    ) {
        let observed_through_frame = own.observed_through_frame();
        while self
            .pending_contexts
            .front()
            .is_some_and(|p| p.target_end_frames[1] <= observed_through_frame)
        {
            let prediction = self.pending_contexts.pop_front().unwrap();
            let observed = own
                .observed_external_energy(
                    prediction.target_start_frames[0],
                    prediction.target_end_frames[0],
                )
                .zip(own.observed_external_energy(
                    prediction.target_start_frames[1],
                    prediction.target_end_frames[1],
                ))
                .map(|(present, after)| [present, after]);
            if let Some(context) = observed {
                match &mut self.memory {
                    Some(memory) => {
                        for (m, c) in memory
                            .iter_mut()
                            .flatten()
                            .zip(context.into_iter().flatten())
                        {
                            *m += 0.1 * (c - *m);
                        }
                    }
                    None => self.memory = Some(context),
                }
            }
            emit(ParticipationContextUpdate {
                prediction,
                observed_through_frame,
                status: if observed.is_some() {
                    "observed"
                } else {
                    "outside_observed_history"
                },
                observed_external_band_energy: observed,
                memory_external_band_energy: self.memory,
            });
        }
    }

    fn next_interval(&mut self) -> f64 {
        let interval = if self.flow_depth > 0.0 {
            crate::life::phonation_engine::OnsetRule::flow_next_ioi(
                (self.fs / self.period_frames) as f32,
                self.flow_depth,
                &mut self.cluster_remaining,
                &mut self.rng,
            ) as f64
                * self.fs
        } else {
            self.period_frames
        };
        interval.max(1.0)
    }

    #[cfg(test)]
    pub(crate) fn advance(
        &mut self,
        now: u64,
        end: u64,
        allowed: bool,
        forecast: Option<&TemporalForecast>,
    ) -> Option<u64> {
        if self.needs_forecast(now, end) {
            self.update_reference(now, forecast);
        }
        let tick = self.candidate(now, end, allowed, forecast)?;
        self.resolve(tick, allowed);
        allowed.then_some(tick)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::temporal_expectation::AcousticTemporalExpectation;

    #[test]
    fn frozen_opportunity_preserves_due_plan_and_renewal_state() {
        use crate::life::action_candidates::OpportunityBasis;
        let mut policy = TemporalParticipation::new(1000, 2.0, 0.0, 0, 59, 16);
        let mut control = TemporalParticipation::new(1000, 2.0, 0.0, 0, 59, 16);
        policy.set_flow_depth(0.8);
        control.set_flow_depth(0.8);
        for _ in 0..20 {
            let issue = policy.last_onset.map_or(0, |last| last + 1);
            let due = policy.opportunity(issue).unwrap();
            assert_eq!(policy.intrinsic_due_for_selected(due.at), None);
            assert_eq!(due.basis, OpportunityBasis::ParticipationDue);
            assert_eq!(policy.opportunity(issue), Some(due));
            let end = due.at + 1000;
            let actual = policy.candidate(issue, end, true, None).unwrap();
            let expected = control.candidate(issue, end, true, None).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(actual, due.at);
            let planned = policy.opportunity(issue).unwrap();
            assert_eq!(planned.at, actual);
            assert_eq!(planned.basis, OpportunityBasis::ParticipationPlanned);
            assert_eq!(policy.intrinsic_due_for_selected(actual), Some(due.at));
            assert_eq!(policy.intrinsic_due_for_selected(actual + 1), None);
            assert!(policy.opportunity(actual + 1).is_none());
            policy.resolve(actual, true);
            assert_eq!(policy.intrinsic_due_for_selected(actual), None);
            control.resolve(expected, true);
        }
    }

    #[test]
    fn learned_acoustic_continuations_change_participation_after_equal_silence() {
        let fs = 24_000;
        let window = 240;
        let mut choices = Vec::new();
        let mut uncoupled = Vec::new();
        let mut recurrence_choices = Vec::new();
        let mut predictions = Vec::new();
        for final_order in [false, true] {
            let mut observer = AcousticTemporalExpectation::new(fs).unwrap();
            let mut own = OwnSoundHistory::new(&observer, 0);
            let mut rng = SmallRng::seed_from_u64(7301);
            for episode in 0..49 {
                let order = if episode == 48 {
                    final_order
                } else {
                    rng.random::<bool>()
                };
                let windows = if episode == 48 { 80 } else { 150 };
                for index in 0..windows {
                    let start = (episode * 150 + index) * window;
                    let audio: [f32; 240] = std::array::from_fn(|i| {
                        let t = (index * window + i) as f64 / fs as f64;
                        let mut signal = 0.0;
                        for (at, duration, high) in
                            [(0.05, 0.2, order), (0.25, 0.2, !order), (1.0, 0.15, order)]
                        {
                            let age = t - at;
                            if age >= 0.0 && age < duration {
                                let gain = (age / 0.005).min((duration - age) / 0.005).min(1.0);
                                let hz = if high { 1800.0 } else { 180.0 };
                                signal += 0.3 * gain * (std::f64::consts::TAU * hz * age).sin();
                            }
                        }
                        signal as f32
                    });
                    own.process(start as u64, &[0.0; 240], &[0.0; 240], &audio, |_, _, _| {});
                    observer.process(start as u64, &audio, |_| {});
                    if (index + 1) % 10 == 0 {
                        let mut forecast = observer.forecast().unwrap();
                        observer.use_external_energy(&mut forecast, &mut own.external);
                    }
                }
            }
            let now = ((48 * 150 + 80) * window) as u64;
            assert_eq!(
                own.observed_external_energy(now - window as u64, now),
                Some([0.0; 3])
            );
            let mut forecast = observer.forecast().unwrap();
            let recurrence = forecast;
            observer.use_external_energy(&mut forecast, &mut own.external);
            predictions.push(forecast.energy_window_after(now + fs as u64 / 5).unwrap().2);
            let mut selected = Vec::new();
            let mut intrinsic = Vec::new();
            let mut baseline = Vec::new();
            // Cross body pace, duration, band and readiness rather than force every body to react.
            for rate in [0.8, 1.5, 3.0] {
                for hold in [0.04, 0.16, 0.4] {
                    for band in 0..3 {
                        for ready in [0.04, 0.16, 0.3] {
                            for (target, coupling, input) in [
                                (&mut intrinsic, 0.0, &forecast),
                                (&mut baseline, 1.0, &recurrence),
                                (&mut selected, 1.0, &forecast),
                            ] {
                                let mut voice = TemporalParticipation::new(
                                    fs,
                                    rate,
                                    coupling,
                                    now,
                                    21,
                                    window as u64,
                                );
                                voice.due_frame = now as f64 + ready * fs as f64;
                                voice.set_sound_duration(hold, None);
                                let mut energy = [0.0; 3];
                                energy[band] = 0.03;
                                voice.set_overlap(1.0, energy);
                                target.push(voice.candidate(
                                    now,
                                    now + 2 * fs as u64,
                                    true,
                                    Some(input),
                                ));
                            }
                        }
                    }
                }
            }
            choices.push(selected);
            uncoupled.push(intrinsic);
            recurrence_choices.push(baseline);
        }
        assert_eq!(uncoupled[0], uncoupled[1]);
        assert_ne!(predictions[0], predictions[1]);
        let differing = choices[0]
            .iter()
            .zip(&choices[1])
            .filter(|(a, b)| a != b)
            .count();
        let newly_differing = (0..choices[0].len())
            .filter(|&i| {
                choices[0][i] != choices[1][i]
                    && recurrence_choices[0][i] == recurrence_choices[1][i]
            })
            .count();
        println!(
            "paired bodies={} different={differing} new relative to recurrence={newly_differing}",
            choices[0].len()
        );
        assert!(newly_differing > 0, "predictions={predictions:?}");
    }

    #[test]
    fn shared_acoustic_cadence_preserves_distinct_action_positions() {
        let fs = 48_000;
        let mut observer = AcousticTemporalExpectation::new(fs).unwrap();
        let mut voices =
            [1, 21, 42].map(|seed| TemporalParticipation::new(fs, 2.0, 0.8, 0, seed, fs as u64));
        // Controlled starting positions test collapse; production assigns no seats.
        for (voice, fraction) in voices.iter_mut().zip([0.1, 0.35, 0.8]) {
            voice.due_frame = fraction * voice.period_frames;
        }
        let mut control = TemporalParticipation::new(fs, 2.0, 0.0, 0, 21, fs as u64);
        let mut onsets: [Vec<u64>; 3] = std::array::from_fn(|_| Vec::new());
        let initial_separation = (voices[0].due_frame - voices[1].due_frame).abs();
        for hop in 0..4000 {
            let now = hop * 480;
            let forecast = observer.forecast();
            for (voice, track) in voices.iter_mut().zip(&mut onsets) {
                if voice.needs_forecast(now, now + 480) {
                    voice.update_reference(now, forecast.as_ref());
                }
                // Isolate cadence from placement scoring against the same sound.
                if let Some(tick) = voice.candidate(now, now + 480, true, None) {
                    voice.resolve(tick, true);
                    track.push(tick);
                }
            }
            control.update_reference(now, forecast.as_ref());
            let audio: [f32; 480] = std::array::from_fn(|i| {
                let age = ((now + i as u64) % 28_800) as f32 / fs as f32;
                0.5 * (std::f32::consts::TAU * 180.0 * age).sin() * (-age / 0.012).exp()
            });
            observer.process(now, &audio, |_| {});
        }
        assert_eq!(control.period_frames(), 24_000.0);
        for voice in &voices {
            assert!((voice.period_frames() - 28_800.0).abs() < 96.0);
        }
        assert!(initial_separation > 480.0);
        for i in 0..3 {
            let late: Vec<_> = onsets[i]
                .iter()
                .copied()
                .filter(|t| *t >= 30 * fs as u64)
                .collect();
            assert!(late.len() >= 15);
            assert!(late.windows(2).all(|w| w[1].abs_diff(w[0] + 28_800) < 96));
            for other in onsets.iter().skip(i + 1) {
                let separation = late
                    .iter()
                    .map(|t| other.iter().map(|o| o.abs_diff(*t)).min().unwrap())
                    .min()
                    .unwrap();
                assert!(separation > 480, "positions collapsed: {separation}");
            }
        }
    }

    #[test]
    fn reference_updates_preserve_cycle_fraction_and_committed_actions() {
        let fs = 48_000;
        let mut observer = AcousticTemporalExpectation::new(fs).unwrap();
        for hop in 0..2000 {
            let now = hop * 480;
            let audio: [f32; 480] = std::array::from_fn(|i| {
                let age = ((now + i) % 28_800) as f32 / fs as f32;
                0.5 * (std::f32::consts::TAU * 180.0 * age).sin() * (-age / 0.012).exp()
            });
            observer.process(now as u64, &audio, |_| {});
        }
        let now = 20 * fs as u64;
        let forecast = observer.forecast().unwrap();
        let mut voice = TemporalParticipation::new(fs, 2.0, 0.8, now, 21, fs as u64);
        let before = (voice.due_frame - now as f64) / voice.period_frames;
        voice.update_reference(now - 1, Some(&forecast));
        assert_eq!(
            voice.period_frames, 24_000.0,
            "future input must not be used"
        );
        voice.update_reference(now, Some(&forecast));
        assert!(voice.period_frames > 24_000.0);
        let after = (voice.due_frame - now as f64) / voice.period_frames;
        assert!((after - before).abs() < 1e-12);
        let period = voice.period_frames;
        let due = voice.due_frame;
        voice.update_reference(now + 480, Some(&forecast));
        assert_eq!(
            voice.period_frames, period,
            "cached evidence must be applied once"
        );
        assert_eq!(voice.due_frame, due);
        voice.candidate(now, now + fs as u64, true, None).unwrap();
        let planned = voice.planned;
        voice.update_reference(now + 960, None);
        assert_eq!(voice.planned, planned);
        assert_eq!(voice.period_frames, period);
        voice.resolve(planned.unwrap().0.round() as u64, true);
        voice.update_reference(now + 2 * fs as u64, None);
        assert!(
            voice.period_frames < period,
            "missing reference relaxes toward the bodily prior"
        );
    }

    #[test]
    fn waiting_for_sound_to_clear_does_not_retrain_the_bodily_tempo() {
        let mut voice = TemporalParticipation::new(1000, 2.0, 0.8, 10_000, 21, 1000 as u64);
        voice.due_frame = 10_000.0;
        assert_eq!(voice.advance(10_000, 10_001, true, None), Some(10_000));
        voice.set_overlap(0.8, [1.0, 0.0, 0.0]);
        let occupied = TemporalForecast::energy_fixture(1000, 10_480, |sec| {
            if sec < 0.13 {
                [10.0, 0.0, 0.0]
            } else {
                [0.0; 3]
            }
        });
        let onset = voice
            .candidate(10_480, 11_000, true, Some(&occupied))
            .unwrap();
        assert!(onset > 10_500 && onset < 10_750, "{onset}");
        assert_eq!(voice.intrinsic_due_for_selected(onset), Some(10_500));
        assert_ne!(voice.intrinsic_due_for_selected(onset), Some(onset));
        voice.resolve(onset, true);
        assert_eq!(voice.intrinsic_due_for_selected(onset), None);
        assert_eq!(voice.period_frames(), 500.0);
        voice.update_parameters(4.0, 0.8, 0.0);
        assert_eq!(voice.period_frames(), 250.0);
    }

    #[test]
    fn flow_keeps_committed_actions_and_still_responds_to_acoustic_overlap() {
        let mut voice = TemporalParticipation::new(1000, 2.0, 0.8, 10_000, 21, 1000 as u64);
        voice.period_frames = 600.0;
        voice.last_reference_frame = Some(9000);
        voice.due_frame = 10_000.0;
        let onset = voice.candidate(10_000, 10_001, true, None).unwrap();
        let planned = voice.planned;
        voice.update_parameters(2.0, 0.8, 0.65);
        assert_eq!(voice.planned, planned);
        assert_eq!(voice.period_frames(), 500.0);
        assert!(voice.last_reference_frame.is_none());
        voice.resolve(onset, true);

        voice.due_frame = 11_000.0;
        voice.set_overlap(0.8, [1.0, 0.0, 0.0]);
        let occupied = TemporalForecast::energy_fixture(1000, 11_000, |_| [10.0, 0.0, 0.0]);
        assert_eq!(voice.advance(11_000, 11_001, true, Some(&occupied)), None);
        assert!(voice.skipped_cycles > 0);
        assert_eq!(voice.period_frames(), 500.0);
        let next = voice.advance(11_001, 15_001, true, Some(&occupied));
        assert!(next.is_some_and(|tick| tick > 11_000));
        assert_eq!(voice.skipped_cycles, 0);
    }

    #[test]
    fn acoustic_advance_cannot_exceed_the_maximum_onset_rate() {
        for fs in [48_000, 48_001] {
            let now = 10 * fs as u64;
            let mut voice = TemporalParticipation::new(fs, 200.0, 0.3, now, 21, fs as u64);
            voice.due_frame = now as f64;
            let min_gap = (fs as f64 / 200.0).ceil() as u64;
            voice.last_onset = Some(now - min_gap);
            voice.set_sound_duration(0.001, None);
            voice.set_overlap(0.8, [1.0, 0.0, 0.0]);
            let forecast = TemporalForecast::energy_fixture(fs, now - fs as u64 / 100, |sec| {
                [sec as f32 * 100.0, 0.0, 0.0]
            });
            let mut unconstrained = TemporalParticipation::new(fs, 200.0, 0.3, now, 21, fs as u64);
            unconstrained.due_frame = now as f64;
            unconstrained.set_sound_duration(0.001, None);
            unconstrained.set_overlap(0.8, [1.0, 0.0, 0.0]);
            assert!(
                unconstrained
                    .candidate(now - 48, now + 480, true, Some(&forecast))
                    .unwrap()
                    < now,
                "the fixture must favor an early onset"
            );
            let onset = voice
                .candidate(now - 48, now + 480, true, Some(&forecast))
                .unwrap();
            assert!(onset - voice.last_onset.unwrap() >= min_gap);
        }
    }

    #[test]
    fn overlap_can_skip_a_cycle_but_cannot_prescribe_permanent_silence() {
        let mut voice = TemporalParticipation::new(1000, 2.0, 0.8, 10_000, 21, 1000 as u64);
        voice.set_overlap(0.8, [1.0, 0.0, 0.0]);
        voice.due_frame = 10_000.0;
        let occupied = TemporalForecast::energy_fixture(1000, 10_000, |_| [10.0, 0.0, 0.0]);
        assert_eq!(voice.candidate(10_000, 10_001, true, Some(&occupied)), None);
        assert_eq!(voice.skipped_cycles, 1);
        assert!(voice.memory.is_none() && voice.last_onset.is_none());
        let tick = voice
            .candidate(10_001, 12_001, true, Some(&occupied))
            .unwrap();
        assert!((11_000..=12_000).contains(&tick), "{tick}");
        voice.resolve(tick, true);
        assert_eq!(voice.skipped_cycles, 0);

        for (coupling, profile) in [(0.0, [1.0, 0.0, 0.0]), (0.8, [0.0, 1.0, 0.0])] {
            let mut control =
                TemporalParticipation::new(1000, 2.0, coupling, 10_000, 21, 1000 as u64);
            control.due_frame = 10_000.0;
            control.set_overlap(0.8, profile);
            assert_eq!(
                control.candidate(10_000, 10_001, true, Some(&occupied)),
                Some(10_000)
            );
        }
    }

    #[test]
    fn duration_changes_the_action_and_withdrawal_releases_waiting() {
        let occupied = TemporalForecast::energy_fixture(1000, 10_000, |sec| {
            if (0.2..0.7).contains(&sec) {
                [10.0, 0.0, 0.0]
            } else {
                [0.0; 3]
            }
        });
        let mut short = TemporalParticipation::new(1000, 2.0, 0.8, 10_000, 21, 1000 as u64);
        let mut long = TemporalParticipation::new(1000, 2.0, 0.8, 10_000, 21, 1000 as u64);
        for voice in [&mut short, &mut long] {
            voice.due_frame = 10_000.0;
            voice.set_overlap(0.8, [1.0, 0.0, 0.0]);
        }
        short.set_sound_duration(0.08, None);
        long.set_sound_duration(1.5, None);
        assert_eq!(
            short.candidate(10_000, 10_001, true, Some(&occupied)),
            Some(10_000)
        );
        assert_eq!(long.candidate(10_000, 10_001, true, Some(&occupied)), None);
        assert!(long.skipped_cycles > 0 || long.planned.unwrap().0 > 10_000.0);
        // New evidence affects the next uncommitted decision; no retrospective replay.
        long.planned = None;
        long.due_frame = 11_000.0;
        let silent = TemporalForecast::energy_fixture(1000, 11_000, |_| [0.0; 3]);
        assert_eq!(
            long.candidate(11_000, 11_001, true, Some(&silent)),
            Some(11_000)
        );
    }

    #[test]
    fn no_evidence_preserves_bodily_pace_without_a_shared_phase() {
        let mut voices = [1, 21, 42]
            .map(|seed| TemporalParticipation::new(48_000, 2.0, 0.7, 0, seed, 48_000 as u64));
        let mut onsets: [Vec<u64>; 3] = std::array::from_fn(|_| Vec::new());
        for now in (0..48_000 * 15).step_by(480) {
            for (voice, track) in voices.iter_mut().zip(&mut onsets) {
                if let Some(onset) = voice.advance(now, now + 480, true, None) {
                    track.push(onset);
                }
            }
        }
        for track in &onsets {
            assert!(track.len() >= 29);
            assert!(track.windows(2).all(|w| w[1] - w[0] == 24_000));
        }
        assert!(onsets[0][0].abs_diff(onsets[1][0]) > 480);
        assert!(onsets[1][0].abs_diff(onsets[2][0]) > 480);
    }

    #[test]
    fn shifting_the_audio_clock_origin_does_not_change_bodily_actions() {
        let shift = 12_345;
        let mut original = TemporalParticipation::new(48_000, 2.0, 0.7, 0, 21, 48_000 as u64);
        let mut shifted = TemporalParticipation::new(48_000, 2.0, 0.7, shift, 21, 48_000 as u64);
        for now in (0..48_000 * 15).step_by(480) {
            let a = original.advance(now, now + 480, true, None);
            let b = shifted.advance(now + shift, now + shift + 480, true, None);
            assert_eq!(a.map(|tick| tick + shift), b);
        }
    }

    #[test]
    fn inaudible_actions_do_not_learn_a_relation_or_reset_every_body() {
        let mut voice = TemporalParticipation::new(48_000, 2.0, 0.7, 0, 21, 48_000 as u64);
        for now in (0..48_000 * 12).step_by(480) {
            assert!(voice.advance(now, now + 480, false, None).is_none());
        }
        assert!(voice.memory.is_none());
        assert!(voice.last_onset.is_none());
        let due = voice.due_frame.round() as u64;
        let mut resumed = Vec::new();
        for now in (48_000 * 12..48_000 * 14).step_by(480) {
            if let Some(onset) = voice.advance(now, now + 480, true, None) {
                resumed.push(onset);
            }
        }
        assert_eq!(resumed[0], due);
    }

    #[test]
    fn selection_and_rejected_execution_do_not_update_action_memory() {
        let mut observer = AcousticTemporalExpectation::new(48_000).unwrap();
        for now in (0..480_000).step_by(480) {
            let audio: [f32; 480] = std::array::from_fn(|i| {
                let age = ((now + i) % 24_000) as f32 / 48_000.0;
                (std::f32::consts::TAU * 180.0 * age).sin() * (-age / 0.012).exp()
            });
            observer.process(now as u64, &audio, |_| {});
        }
        let forecast = observer.forecast().unwrap();
        let mut voice = TemporalParticipation::new(48_000, 10.0, 0.7, 480_000, 21, 48_000 as u64);
        let tick = voice
            .candidate(480_000, 486_000, true, Some(&forecast))
            .unwrap();
        assert!(
            voice
                .planned
                .unwrap()
                .1
                .unwrap()
                .pred_external_band_energy
                .iter()
                .flatten()
                .map(|v| v * v)
                .sum::<f32>()
                > 0.0
        );
        assert!(voice.memory.is_none());
        assert!(voice.last_onset.is_none());
        voice.resolve(tick, false);
        assert!(voice.memory.is_none());
        assert!(voice.last_onset.is_none());
        let rejected_tick = tick;
        let mut voice = TemporalParticipation::new(48_000, 10.0, 0.7, 480_000, 21, 48_000 as u64);
        let tick = voice
            .candidate(480_000, 486_000, true, Some(&forecast))
            .unwrap();
        assert_eq!(tick, rejected_tick);
        voice.resolve(tick, true);
        assert!(
            voice.memory.is_none(),
            "execution has not supplied acoustic results yet"
        );
        assert_eq!(voice.pending_contexts.len(), 1);
        let pending = voice.pending_contexts.clone();
        assert_eq!(voice.last_onset, Some(tick));
        let due = voice.due_frame;
        voice.update_parameters(8.0, 0.4, 0.8);
        assert!(voice.memory.is_none());
        assert_eq!(voice.pending_contexts, pending);
        assert_eq!(voice.last_onset, Some(tick));
        assert_eq!(voice.due_frame, due);
    }

    #[test]
    fn actual_context_waits_for_complete_windows_and_changes_the_next_choice() {
        let fs = 48_000;
        let mut choices = Vec::new();
        for frequency in [40.0, 14_000.0] {
            let observer = AcousticTemporalExpectation::new(fs).unwrap();
            let mut own = OwnSoundHistory::new(&observer, 0);
            let forecast = TemporalForecast::energy_fixture(fs, 0, |_| [0.0, 1.0, 0.0]);
            let mut voice = TemporalParticipation::new(fs, 2.0, 0.7, 0, 21, 512);
            voice.due_frame = 0.0;
            let onset = voice.candidate(0, 512, true, Some(&forecast)).unwrap();
            voice.resolve(onset, true);
            assert!(voice.memory.is_none());
            let issued = *voice.pending_contexts.front().unwrap();
            let footprint = issued.external_energy_footprint.unwrap();
            assert_eq!(footprint.requested[0], onset);
            assert!(
                footprint
                    .points
                    .iter()
                    .flatten()
                    .all(|point| point.band_energy_sum == Some(1.))
            );
            let end = issued.target_end_frames[1] as usize;
            let audio: Vec<f32> = (0..end)
                .map(|i| 0.2 * (std::f32::consts::TAU * frequency * i as f32 / fs as f32).sin())
                .collect();
            let silent = vec![0.0; end];
            let mut updates = Vec::new();
            own.process(
                0,
                &silent[..end - 1],
                &silent[..end - 1],
                &audio[..end - 1],
                |_, _, _| {},
            );
            voice.observe_context(&own, |u| updates.push(u));
            assert!(voice.memory.is_none() && updates.is_empty());
            own.process(
                (end - 1) as u64,
                &silent[end - 1..],
                &silent[end - 1..],
                &audio[end - 1..],
                |_, _, _| {},
            );
            voice.observe_context(&own, |u| updates.push(u));
            assert_eq!(updates.len(), 1);
            assert_eq!(updates[0].prediction, issued);
            assert_eq!(updates[0].status, "observed");
            let expected = std::array::from_fn(|i| {
                own.observed_external_energy(
                    issued.target_start_frames[i],
                    issued.target_end_frames[i],
                )
                .unwrap()
            });
            assert_eq!(voice.memory, Some(expected));
            assert_ne!(voice.memory, Some(issued.pred_external_band_energy));
            voice.observe_context(&own, |u| updates.push(u));
            assert_eq!(updates.len(), 1, "results must not train twice");
            let next = TemporalForecast::energy_fixture(fs, end as u64, |sec| {
                if sec < 0.55 {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 0.0, 1.0]
                }
            });
            choices.push(
                voice
                    .candidate(end as u64, fs as u64, true, Some(&next))
                    .unwrap(),
            );
        }
        assert!(
            choices[1] > choices[0],
            "observed contexts should choose different positions: {choices:?}"
        );
    }

    #[test]
    fn unavailable_context_is_discarded_without_learning_silence() {
        let fs = 1000;
        let mut voice = TemporalParticipation::new(fs, 2.0, 0.7, 0, 21, 512);
        voice.due_frame = 0.0;
        let forecast = TemporalForecast::energy_fixture(fs, 0, |_| [1.0; 3]);
        let onset = voice.candidate(0, 1, true, Some(&forecast)).unwrap();
        voice.resolve(onset, true);
        let mut observer = AcousticTemporalExpectation::new(fs).unwrap();
        observer.process(1000, &[0.0; 10], |_| {});
        let own = OwnSoundHistory::new(&observer, 1010);
        let mut updates = Vec::new();
        voice.observe_context(&own, |u| updates.push(u));
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].status, "outside_observed_history");
        assert!(updates[0].observed_external_band_energy.is_none());
        assert!(voice.memory.is_none() && voice.pending_contexts.is_empty());
    }

    #[cfg(feature = "profile-alloc")]
    #[test]
    fn maximum_rate_context_backlog_updates_without_steady_state_allocations() {
        let fs = 48_001;
        let hop = 16_384;
        let observer = AcousticTemporalExpectation::new(fs).unwrap();
        let mut own = OwnSoundHistory::new(&observer, 0);
        let mut voice = TemporalParticipation::new(fs, 200.0, 0.7, 0, 21, hop);
        let capacity = voice.pending_contexts.capacity();
        let silent = vec![0.0; hop as usize];
        let external = vec![0.125; hop as usize];
        let mut updates = 0;
        let mut peak_pending = 0;
        crate::runtime_profile::begin_allocations();
        for i in 0..40 {
            let now = i * hop;
            let forecast =
                TemporalForecast::energy_fixture(fs, own.observed_through_frame(), |_| [0.1; 3]);
            let mut cursor = now;
            while let Some(onset) = voice.candidate(cursor, now + hop, true, Some(&forecast)) {
                voice.resolve(onset, true);
                cursor = onset + 1;
            }
            peak_pending = peak_pending.max(voice.pending_contexts.len());
            own.process(now, &silent, &silent, &external, |_, _, _| {});
            voice.observe_context(&own, |_| updates += 1);
        }
        let allocations = crate::runtime_profile::finish_allocations().unwrap();
        assert_eq!(allocations.count, 0);
        assert_eq!(allocations.bytes, 0);
        assert_eq!(voice.pending_contexts.capacity(), capacity);
        assert!(peak_pending > 60 && peak_pending <= capacity);
        assert!(updates > 2000);
    }

    #[test]
    fn observed_context_changes_action_while_zero_coupling_remains_intrinsic() {
        let fs = 48_000;
        let mut observer = AcousticTemporalExpectation::new(fs).unwrap();
        let mut own = OwnSoundHistory::new(&observer, 0);
        let mut voice = TemporalParticipation::new(fs, 2.0, 0.7, 0, 21, fs as u64);
        let mut control = TemporalParticipation::new(fs, 2.0, 0.0, 0, 21, fs as u64);
        let mut actions = Vec::new();
        let mut controls = Vec::new();
        let mut forecast = None;
        for hop in 0..4000 {
            let now = hop * 480;
            if let Some(onset) = voice.advance(now, now + 480, true, forecast.as_ref()) {
                actions.push(onset);
            }
            if let Some(onset) = control.advance(now, now + 480, true, forecast.as_ref()) {
                controls.push(onset);
            }
            let audio: [f32; 480] = std::array::from_fn(|i| {
                let t = (now + i as u64) as f32 / fs as f32;
                let phase = if t < 20.0 { t % 0.5 } else { (t - 20.0) % 0.54 };
                let a = (std::f32::consts::TAU * 180.0 * phase).sin() * (-phase / 0.012).exp();
                let age = (phase - 0.31).rem_euclid(if t < 20.0 { 0.5 } else { 0.54 });
                a * 0.6 + (std::f32::consts::TAU * 1800.0 * age).sin() * (-age / 0.012).exp() * 0.5
            });
            own.process(now, &[0.0; 480], &[0.0; 480], &audio, |_, _, _| {});
            voice.observe_context(&own, |_| {});
            observer.process(now, &audio, |_| {});
            forecast = observer.forecast();
        }
        assert!(voice.memory.is_some());
        assert_ne!(actions, controls);
        assert!(controls.windows(2).all(|w| w[1] - w[0] == 24_000));
        assert!(actions.windows(2).all(|w| w[1] - w[0] > 12_000));
        let differing = actions
            .iter()
            .zip(&controls)
            .filter(|(a, b)| a.abs_diff(**b) > 480)
            .count();
        assert!(
            differing > 10,
            "only {differing} actions responded to observed context"
        );
    }

    #[test]
    #[ignore = "offline action-policy observation; set CONCHORDAL_TEMPORAL_ASSAY_DIR"]
    fn export_participation_assay() -> anyhow::Result<()> {
        use std::fs::{self, OpenOptions};
        use std::io::{BufWriter, Write};
        use std::path::Path;
        let root = std::env::var_os("CONCHORDAL_TEMPORAL_ASSAY_DIR")
            .ok_or_else(|| anyhow::anyhow!("CONCHORDAL_TEMPORAL_ASSAY_DIR is required"))?;
        let mut cases = fs::read_dir(Path::new(&root))?
            .map(|e| e.map(|e| e.path()))
            .collect::<Result<Vec<_>, _>>()?;
        cases.retain(|p| p.is_dir() && p.join("audio.wav").is_file());
        cases.sort();
        anyhow::ensure!(!cases.is_empty(), "no audio.wav inputs");
        for case in cases {
            let mut reader = hound::WavReader::open(case.join("audio.wav"))?;
            let spec = reader.spec();
            anyhow::ensure!(
                spec.channels == 1
                    && spec.bits_per_sample == 16
                    && spec.sample_format == hound::SampleFormat::Int,
                "expected mono PCM16"
            );
            let audio = reader
                .samples::<i16>()
                .map(|s| s.map(|v| v as f32 / 32768.0))
                .collect::<Result<Vec<_>, _>>()?;
            let mut observer = AcousticTemporalExpectation::new(spec.sample_rate)
                .ok_or_else(|| anyhow::anyhow!("unsupported observation rate"))?;
            let mut own = OwnSoundHistory::new(&observer, 0);
            let silent = [0.0; 512];
            let mut voices = [0.0, 0.7].map(|coupling| {
                [1, 21, 42, 7, 57, 113, 307, 911].map(|seed| {
                    TemporalParticipation::new(
                        spec.sample_rate,
                        2.0,
                        coupling,
                        0,
                        seed,
                        spec.sample_rate as u64,
                    )
                })
            });
            let mut out = BufWriter::new(
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(case.join("participation.csv"))?,
            );
            writeln!(
                out,
                "condition,seed,onset_frame,time_sec,body_period_sec,context_learned"
            )?;
            let mut forecast = None;
            for (index, chunk) in audio.chunks(512).enumerate() {
                let now = (index * 512) as u64;
                for (condition, cohort) in voices.iter_mut().enumerate() {
                    for (seed, voice) in [1, 21, 42, 7, 57, 113, 307, 911].into_iter().zip(cohort) {
                        if let Some(onset) =
                            voice.advance(now, now + chunk.len() as u64, true, forecast.as_ref())
                        {
                            writeln!(
                                out,
                                "{condition},{seed},{onset},{:.9},{:.9},{}",
                                onset as f64 / spec.sample_rate as f64,
                                voice.period_frames / spec.sample_rate as f64,
                                voice.memory.is_some()
                            )?;
                        }
                    }
                }
                // Counterfactual actions never enter this acoustic evidence stream.
                own.process(
                    now,
                    &silent[..chunk.len()],
                    &silent[..chunk.len()],
                    chunk,
                    |_, _, _| {},
                );
                for voice in voices.iter_mut().flatten() {
                    voice.observe_context(&own, |_| {});
                }
                observer.process(now, chunk, |_| {});
                forecast = observer.forecast();
            }
            out.flush()?;
            println!("observed action policy {}", case.display());
        }
        Ok(())
    }
}
