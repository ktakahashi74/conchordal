use super::action_candidates::{Class, live::Transition};
use super::action_observation::ActionKind;
use crate::core::modulation::NeuralRhythms;
use crate::core::temporal_expectation::{AcousticTemporalExpectation, OwnSoundHistory};
use crate::core::timebase::{Tick, Timebase};
use crate::life::phonation_engine::ToneCmd;
use crate::life::sound::Tone;
use crate::life::voice::PhonationBatch;
use crate::scenario::control::Routing;
use std::collections::BTreeMap;
use std::time::Instant;
use tracing::debug;

// Ordered so the per-tick mixdown sums tones in a fixed (source_id, tone_id)
// order. HashMap iteration order is process-randomized, which makes the
// non-associative float accumulation below differ run-to-run. The habitat sum
// feeds NSGT analysis -> landscape -> pitch decisions, so that drift would make
// the pre-synth ALIFE/landscape computation non-deterministic across renders.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ToneKey {
    source_id: u64,
    tone_id: u64,
}

struct RoutedTone {
    tone: Tone,
    routing: Routing,
    self_slot: Option<usize>,
    outcome_slots: [Option<(usize, u64)>; 2],
    source_generation: u32,
    body_slot: Option<(usize, u32)>,
    scheduled_release: Option<super::self_prediction::ScheduledRelease>,
}

struct SelfSound {
    source_id: u64,
    used: bool,
    body: Vec<f32>,
    habitat: Vec<f32>,
    history: OwnSoundHistory,
}

pub struct RenderFrame<'a> {
    pub presentation: &'a [f32],
    pub habitat: &'a [f32],
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(crate) struct RenderProfile {
    setup_us: f64,
    commands_us: f64,
    samples_us: f64,
    history_us: f64,
    observation_us: f64,
    capture_delivery_us: f64,
}

fn profile_lap(checkpoint: &mut Option<Instant>) -> f64 {
    checkpoint.as_mut().map_or(0.0, |previous| {
        let now = Instant::now();
        let elapsed = now.duration_since(*previous).as_secs_f64() * 1_000_000.0;
        *previous = now;
        elapsed
    })
}

pub struct ScheduleRenderer {
    time: Timebase,
    buf_presentation: Vec<f32>,
    buf_habitat: Vec<f32>,
    tones: BTreeMap<ToneKey, RoutedTone>,
    cutoff_tick: Option<Tick>,
    self_sound: Vec<Option<Box<SelfSound>>>,
    pub(crate) body_capture: Option<crate::temporal_cognition::body::Capture>,
    pub(crate) action_observer: Option<Box<super::action_observation::Observer>>,
    pub(crate) profile: Option<RenderProfile>,
}

pub type SoundRenderer = ScheduleRenderer;

impl ScheduleRenderer {
    pub fn new(time: Timebase) -> Self {
        Self {
            time,
            buf_presentation: vec![0.0; time.hop],
            buf_habitat: vec![0.0; time.hop],
            tones: BTreeMap::new(),
            cutoff_tick: None,
            self_sound: Vec::new(),
            action_observer: None,
            body_capture: None,
            profile: None,
        }
    }

    /// Copy only one actor's current sounding state for an offline forward model.
    /// The caller must resume at the next unrendered tick. Future commands and rhythm
    /// changes are supplied explicitly, never borrowed from other actors' state.
    /// Used by examples/action_prediction.rs; cloning allocates and is not an RT path.
    pub fn fork_source(&self, source_id: u64) -> Self {
        let mut fork = Self::new(self.time);
        fork.cutoff_tick = self.cutoff_tick;
        fork.tones = self
            .tones
            .iter()
            .filter(|(key, _)| key.source_id == source_id)
            .map(|(key, routed)| {
                (
                    *key,
                    RoutedTone {
                        tone: routed.tone.clone(),
                        routing: routed.routing,
                        self_slot: None,
                        outcome_slots: [None; 2],
                        source_generation: routed.source_generation,
                        body_slot: None,
                        scheduled_release: routed.scheduled_release,
                    },
                )
            })
            .collect();
        fork
    }

    #[cfg(test)]
    pub(crate) fn source_envelopes(
        &self,
        source_id: u64,
        source_generation: u32,
    ) -> impl Iterator<Item = (u64, crate::life::sound::envelope::Envelope)> + '_ {
        self.tones
            .iter()
            .filter(move |(key, tone)| {
                key.source_id == source_id && tone.source_generation == source_generation
            })
            .map(|(key, tone)| (key.tone_id, tone.tone.prediction_parameters(None).2))
    }

    #[cfg(test)]
    pub(crate) fn source_energy_models(
        &self,
        source_id: u64,
        source_generation: u32,
        now: u64,
        rhythms: &NeuralRhythms,
    ) -> impl Iterator<Item = (u64, [bool; 2], super::self_prediction::ToneEnergy)> + '_ {
        let rhythms = *rhythms;
        self.tones
            .iter()
            .filter(move |(key, rt)| {
                key.source_id == source_id && rt.source_generation == source_generation
            })
            .map(move |(key, rt)| {
                let (_, amplitude, envelope) = rt.tone.prediction_parameters(None);
                (
                    key.tone_id,
                    [rt.routing.to_habitat, rt.routing.to_presentation],
                    super::self_prediction::ToneEnergy {
                        amplitude,
                        envelope,
                        control: Some(rt.tone.prediction_control(now, &rhythms)),
                        scheduled_release: rt.scheduled_release,
                        sine: rt.tone.prediction_sine(now),
                        bank: rt.tone.prediction_bank(now),
                    },
                )
            })
    }

    pub(crate) fn prepare_self_sound(
        &mut self,
        ids: impl Iterator<Item = u64>,
        now: Tick,
        observer: &AcousticTemporalExpectation,
    ) {
        if self.time.fs < 50.0 {
            return;
        }
        for sound in self.self_sound.iter_mut().flatten() {
            sound.used = false;
        }
        for id in ids {
            if let Some(sound) = self
                .self_sound
                .iter_mut()
                .flatten()
                .find(|sound| sound.source_id == id)
            {
                sound.used = true;
                continue;
            }
            debug_assert!(
                self.tones.keys().all(|key| key.source_id != id),
                "self tracking must start before this source sounds"
            );
            let sound = Box::new(SelfSound {
                source_id: id,
                used: true,
                body: vec![0.0; self.time.hop],
                habitat: vec![0.0; self.time.hop],
                history: OwnSoundHistory::new(observer, now),
            });
            let slot = if let Some(slot) = self.self_sound.iter().position(Option::is_none) {
                self.self_sound[slot] = Some(sound);
                slot
            } else {
                self.self_sound.push(Some(sound));
                self.self_sound.len() - 1
            };
            for (key, tone) in &mut self.tones {
                if key.source_id == id {
                    tone.self_slot = Some(slot);
                }
            }
        }
        for (slot, sound) in self.self_sound.iter_mut().enumerate() {
            if sound.as_ref().is_some_and(|sound| !sound.used) {
                for tone in self.tones.values_mut() {
                    if tone.self_slot == Some(slot) {
                        tone.self_slot = None;
                    }
                }
                *sound = None;
            }
        }
    }

    pub(crate) fn self_sound_history(&mut self, source_id: u64) -> Option<&mut OwnSoundHistory> {
        self.self_sound
            .iter_mut()
            .flatten()
            .find(|sound| sound.source_id == source_id)
            .map(|sound| &mut sound.history)
    }

    pub(crate) fn prepare_energy_contexts(
        &mut self,
        acoustic: &AcousticTemporalExpectation,
        batches: &[PhonationBatch],
        now: Tick,
    ) {
        let (Some(capture), Some(observer)) =
            (self.body_capture.as_ref(), self.action_observer.as_mut())
        else {
            return;
        };
        let sampled: [bool; crate::temporal_cognition::body::VOICES] = std::array::from_fn(|i| {
            self.self_sound
                .get(i)
                .and_then(Option::as_ref)
                .is_some_and(|sound| {
                    batches.iter().any(|batch| {
                        batch.source_id == sound.source_id
                            && observer.candidate_energy_due(
                                (batch.source_id, batch.source_generation),
                                now,
                            )
                    })
                })
        });
        let contexts = self.self_sound.iter().enumerate().filter_map(|(i, sound)| {
            let sound = sound.as_ref()?;
            let batch = batches.iter().find(|batch| {
                batch.source_id == sound.source_id
                    && (sampled.get(i) == Some(&true)
                        || batch
                            .cmds
                            .iter()
                            .any(|cmd| matches!(cmd, ToneCmd::On { .. } | ToneCmd::Off { .. })))
            })?;
            let (_, body_generation) = capture.token(batch.source_id, batch.source_generation)?;
            let forecast = acoustic.preview_external_energy(&sound.history.external)?;
            Some((
                batch.source_id,
                batch.source_generation,
                body_generation,
                forecast,
            ))
        });
        observer.refresh_energy_contexts(now, contexts);
    }

    pub(crate) fn drain_prediction_errors(
        &mut self,
        mut emit: impl FnMut(
            u64,
            u64,
            u64,
            usize,
            &crate::core::history_prediction::PredictionErrorTotals,
        ),
    ) {
        for sound in self.self_sound.iter_mut().flatten() {
            if let Some((from, through, window, errors)) = sound.history.take_prediction_errors() {
                emit(sound.source_id, from, through, window, &errors);
            }
        }
    }

    pub fn render(
        &mut self,
        phonation_batches: &[PhonationBatch],
        now: Tick,
        rhythms: &NeuralRhythms,
    ) -> RenderFrame<'_> {
        self.render_with_prediction_matches(phonation_batches, now, rhythms, |_, _, _, _| {})
    }

    pub(crate) fn render_with_prediction_matches(
        &mut self,
        phonation_batches: &[PhonationBatch],
        now: Tick,
        rhythms: &NeuralRhythms,
        mut emit: impl FnMut(u64, u64, usize, &crate::core::history_prediction::PredictionMatch<'_>),
    ) -> RenderFrame<'_> {
        let mut checkpoint = self.profile.as_ref().map(|_| Instant::now());
        self.profile = self.profile.map(|_| RenderProfile::default());
        let hop = self.time.hop;
        if let Some(capture) = self.body_capture.as_mut() {
            capture.begin(now);
        }
        if let Some(observer) = self.action_observer.as_mut() {
            observer.begin_hop(now);
            if let Some(bank) = observer.body_defaults() {
                bank.begin_hop(phonation_batches);
            }
        }
        if self.buf_presentation.len() != hop {
            self.buf_presentation.resize(hop, 0.0);
        }
        if self.buf_habitat.len() != hop {
            self.buf_habitat.resize(hop, 0.0);
        }
        self.buf_presentation.fill(0.0);
        self.buf_habitat.fill(0.0);
        for sound in self.self_sound.iter_mut().flatten() {
            sound.body.fill(0.0);
            sound.habitat.fill(0.0);
        }

        let fs = self.time.fs;
        if fs <= 0.0 {
            return RenderFrame {
                presentation: &self.buf_presentation,
                habitat: &self.buf_habitat,
            };
        }

        self.tones.retain(|key, rt| {
            if !rt.tone.is_done(now) {
                return true;
            }
            if let Some(observer) = self.action_observer.as_mut() {
                for slot in rt.outcome_slots.into_iter().flatten() {
                    observer.sample(
                        slot,
                        (key.source_id, rt.source_generation, key.tone_id),
                        now,
                        0.0,
                        true,
                    );
                }
            }
            false
        });

        let end = now.saturating_add(hop as Tick);
        let dt = 1.0 / fs;
        let mut rhythms = *rhythms;
        let setup_us = profile_lap(&mut checkpoint);
        self.apply_phonation_batches(phonation_batches, now, &rhythms, dt);
        let commands_us = profile_lap(&mut checkpoint);
        for tick in now..end {
            let idx = (tick - now) as usize;
            let mut acc_presentation = 0.0f32;
            let mut acc_habitat = 0.0f32;
            for (key, rt) in &mut self.tones {
                rt.tone.apply_updates_if_due(tick);
                rt.tone.kick_planned_if_due(tick);
                let sample = rt.tone.render_tick(tick, fs, dt, &rhythms);
                if let Some(observer) = self.action_observer.as_mut() {
                    for slot in &mut rt.outcome_slots {
                        if let Some(index) = *slot
                            && !observer.sample(
                                index,
                                (key.source_id, rt.source_generation, key.tone_id),
                                tick,
                                sample,
                                rt.tone.is_done(tick),
                            )
                        {
                            *slot = None;
                        }
                    }
                }
                if let (Some(capture), Some(slot)) = (self.body_capture.as_mut(), rt.body_slot) {
                    capture.sample(slot, idx, sample, rt.routing);
                }
                if let Some(slot) = rt.self_slot {
                    let sound = self.self_sound[slot]
                        .as_mut()
                        .expect("active self-sound slot");
                    sound.body[idx] += sample;
                    if rt.routing.to_habitat {
                        sound.habitat[idx] += sample;
                    }
                }
                if rt.routing.to_presentation {
                    acc_presentation += sample;
                }
                if rt.routing.to_habitat {
                    acc_habitat += sample;
                }
            }
            self.buf_presentation[idx] = acc_presentation;
            self.buf_habitat[idx] = acc_habitat;
            rhythms.advance_in_place(dt);
        }
        let samples_us = profile_lap(&mut checkpoint);

        for sound in self.self_sound.iter_mut().flatten() {
            sound.history.process(
                now,
                &sound.body,
                &sound.habitat,
                &self.buf_habitat,
                |start, window, matched| emit(sound.source_id, start, window, matched),
            );
        }
        let history_us = profile_lap(&mut checkpoint);
        if let Some(observer) = self.action_observer.as_mut() {
            if let Some(capture) = self.body_capture.as_ref() {
                observer.observe_source_energy(capture);
            }
            observer.end_hop(now, end);
        }
        let observation_us = profile_lap(&mut checkpoint);

        if let Some(capture) = self.body_capture.as_mut() {
            capture.end();
        }
        let capture_delivery_us = profile_lap(&mut checkpoint);
        if let Some(profile) = self.profile.as_mut() {
            *profile = RenderProfile {
                setup_us,
                commands_us,
                samples_us,
                history_us,
                observation_us,
                capture_delivery_us,
            };
        }
        RenderFrame {
            presentation: &self.buf_presentation,
            habitat: &self.buf_habitat,
        }
    }

    pub fn is_idle(&self) -> bool {
        self.tones.is_empty()
    }

    pub(crate) fn active_tone_count(&self) -> usize {
        self.tones.len()
    }

    pub fn shutdown_at(&mut self, tick: Tick) {
        self.cutoff_tick = Some(tick);
        self.tones.retain(|key, rt| {
            if rt.tone.onset() <= tick {
                return true;
            }
            if let Some(observer) = self.action_observer.as_mut() {
                for slot in rt.outcome_slots.into_iter().flatten() {
                    observer.interrupt(
                        slot,
                        (key.source_id, rt.source_generation, key.tone_id),
                        tick,
                        "cancelled",
                    );
                }
            }
            false
        });
        for (key, rt) in &mut self.tones {
            if let Some(observer) = self.action_observer.as_mut() {
                let owner = PhonationBatch {
                    intrinsic_period_sec: None,
                    source_id: key.source_id,
                    source_generation: rt.source_generation,
                    routing: rt.routing,
                    ..Default::default()
                };
                let status = if rt.tone.is_done(tick) {
                    "already_ended"
                } else {
                    "accepted"
                };
                if status == "accepted" {
                    if let Some(slot) = rt.outcome_slots[1].take() {
                        observer.interrupt(
                            slot,
                            (key.source_id, rt.source_generation, key.tone_id),
                            tick,
                            "superseded",
                        );
                    }
                    rt.outcome_slots[1] = observer.command(
                        &owner,
                        key.tone_id,
                        Some(tick),
                        tick,
                        status,
                        ActionKind::Shutdown,
                    );
                }
            }
            rt.tone.note_off(tick);
        }
    }

    fn apply_phonation_batches(
        &mut self,
        phonation_batches: &[PhonationBatch],
        now: Tick,
        rhythms: &NeuralRhythms,
        _dt: f32,
    ) {
        let default_hold_ticks = max_phonation_hold_ticks(self.time);
        for batch in phonation_batches {
            let body_generation = self
                .body_capture
                .as_ref()
                .and_then(|c| c.token(batch.source_id, batch.source_generation))
                .map(|t| t.1);
            if let Some(bank) = self
                .action_observer
                .as_mut()
                .and_then(|o| o.body_defaults())
                && let Some(record) = bank.begin_source(batch, now, body_generation)
            {
                let from = ToneKey {
                    source_id: batch.source_id,
                    tone_id: 0,
                };
                let through = ToneKey {
                    source_id: batch.source_id,
                    tone_id: u64::MAX,
                };
                for (_, rt) in self.tones.range(from..=through) {
                    if rt.source_generation != batch.source_generation {
                        record.other_source_generation_tones += 1;
                        continue;
                    }
                    if body_generation.is_none() || rt.body_slot.map(|t| t.1) != body_generation {
                        record.unmatched_body_tones += 1;
                    }
                    if rt.tone.onset() > now {
                        record.queued_tones += 1;
                        record.next_queued_onset = Some(
                            record
                                .next_queued_onset
                                .map_or(rt.tone.onset(), |at| at.min(rt.tone.onset())),
                        );
                    } else if !rt.tone.is_done(now) {
                        record.active_tones += 1;
                        record.active_bus_tones[0] += usize::from(rt.routing.to_habitat);
                        record.active_bus_tones[1] += usize::from(rt.routing.to_presentation);
                    }
                }
                record.finish_facts(self.time.fs as u32);
                if (record.active_tones > 0 || record.queued_tones > 0)
                    && record.policy.is_some_and(|p| p.is_alive)
                    && let Some(body_generation) = body_generation
                {
                    let request = super::action_candidates::energy::ScheduledRequest {
                        source_id: batch.source_id,
                        source_generation: batch.source_generation,
                        body_generation,
                        issued_at: now,
                        sample_rate: self.time.fs as u32,
                        hop: self.time.hop as u64,
                        period: record
                            .intrinsic_period_sec
                            .map(|p| (p * f64::from(self.time.fs)).round())
                            .filter(|p| p.is_finite() && *p >= 1. && *p < u64::MAX as f64)
                            .map(|p| p as u64),
                    };
                    if let Some(mut packet) = bank.energy.acquire() {
                        packet.scheduled = Some(request);
                        packet.shared = bank.shared.clone();
                        packet.bindings = std::array::from_fn(|bus| {
                            bank.bindings.iter().copied().find(|b| {
                                b.source_id == batch.source_id
                                    && b.source_generation == batch.source_generation
                                    && b.body_generation == body_generation
                                    && b.bus as usize == bus
                            })
                        });
                        bank.pending.push_back(packet);
                    }
                }
            }
            for cmd in &batch.cmds {
                let mut prediction = None;
                let mut candidate_packet = None;
                match *cmd {
                    ToneCmd::On { tone_id, kick } => {
                        let key = ToneKey {
                            source_id: batch.source_id,
                            tone_id,
                        };
                        let spec = batch.tones.iter().find(|t| t.tone_id == tone_id);
                        if self.tones.contains_key(&key) {
                            if let Some(observer) = self.action_observer.as_mut() {
                                observer.command(
                                    batch,
                                    tone_id,
                                    spec.map(|s| s.onset),
                                    now,
                                    "duplicate",
                                    ActionKind::Onset,
                                );
                            }
                            continue;
                        }
                        let Some(spec) = spec else {
                            if let Some(observer) = self.action_observer.as_mut() {
                                observer.command(
                                    batch,
                                    tone_id,
                                    None,
                                    now,
                                    "missing_spec",
                                    ActionKind::Onset,
                                );
                            }
                            continue;
                        };
                        if let Some(cutoff) = self.cutoff_tick
                            && spec.onset >= cutoff
                        {
                            if let Some(observer) = self.action_observer.as_mut() {
                                observer.command(
                                    batch,
                                    tone_id,
                                    Some(spec.onset),
                                    now,
                                    "cutoff",
                                    ActionKind::Onset,
                                );
                            }
                            continue;
                        }
                        let hold_ticks = spec.hold_ticks.unwrap_or(default_hold_ticks);
                        if let Some(mut tone) = Tone::from_parts(
                            self.time,
                            spec.onset,
                            hold_ticks,
                            spec.freq_hz,
                            spec.amp,
                            Some(spec.body.clone()),
                            Some(spec.render_modulator.clone()),
                            spec.adsr,
                        ) {
                            tone.seed_modal_phases(modal_phase_seed(
                                batch.source_id,
                                spec.onset,
                                tone_id,
                            ));
                            tone.set_smoothing_tau_sec(spec.smoothing_tau_sec);
                            tone.note_on(spec.onset);
                            tone.schedule_planned_kick(kick);
                            tone.arm_onset_trigger(kick.strength.max(0.0));
                            let mut scheduled_release = None;
                            let opportunity = spec.opportunity.filter(|receipt| {
                                receipt.issued_at == now
                                    && receipt.at == spec.onset
                                    && receipt.at >= now
                                    && batch.body_policy.is_some_and(|policy| {
                                        policy.at == now
                                            && policy.is_alive
                                            && policy.gate_allows_onset
                                    })
                            });
                            if let Some(receipt) = opportunity
                                && let Some(off) =
                                    receipt.planned_release_at.filter(|off| *off >= spec.onset)
                            {
                                // Off changes attack clipping when its batch is applied.
                                // Freeze that hop switch without changing the live Tone.
                                scheduled_release =
                                    Some(super::self_prediction::ScheduledRelease {
                                        apply_at_sample: now
                                            + (off - now) / self.time.hop as u64
                                                * self.time.hop as u64,
                                        off_sample: off,
                                    });
                            }

                            if let Some(receipt) = opportunity
                                && let Some(bank) = self
                                    .action_observer
                                    .as_mut()
                                    .and_then(|o| o.body_defaults())
                                && bank
                                    .sampled_record((batch.source_id, batch.source_generation), now)
                                    .is_some()
                            {
                                let existing = bank.pending.iter().position(|p| {
                                    p.request
                                        .map(|r| (r.source_id, r.source_generation))
                                        .or(p.scheduled.map(|r| (r.source_id, r.source_generation)))
                                        == Some((batch.source_id, batch.source_generation))
                                });
                                let packet = match existing {
                                    Some(i) if bank.pending[i].request.is_none() => {
                                        bank.pending.remove(i)
                                    }
                                    Some(_) => None,
                                    None => bank.energy.acquire(),
                                };
                                if let Some(mut packet) = packet {
                                    packet.scheduled = None;
                                    packet.shared = bank.shared.clone();
                                    packet.bindings = std::array::from_fn(|bus| {
                                        bank.bindings.iter().copied().find(|b| {
                                            b.source_id == batch.source_id
                                                && b.source_generation == batch.source_generation
                                                && Some(b.body_generation) == body_generation
                                                && b.bus as usize == bus
                                        })
                                    });
                                    let (_, amplitude, envelope) = tone.prediction_parameters(None);
                                    packet.request =
                                        Some(super::action_candidates::energy::Request {
                                            source_id: batch.source_id,
                                            source_generation: batch.source_generation,
                                            body_generation,
                                            tone_id,
                                            issued_at: now,
                                            sample_rate: self.time.fs as u32,
                                            hop: self.time.hop as u64,
                                            opportunity: receipt,
                                            routed: [
                                                batch.routing.to_habitat,
                                                batch.routing.to_presentation,
                                            ],
                                            recipe: super::self_prediction::ToneEnergy {
                                                amplitude,
                                                envelope,
                                                scheduled_release,
                                                control: Some(
                                                    tone.prediction_control(now, rhythms),
                                                ),
                                                sine: tone.prediction_sine(now),
                                                bank: tone.prediction_bank(now),
                                            },
                                        });
                                    candidate_packet = Some(packet);
                                }
                            }
                            let outcome_slot = self.action_observer.as_mut().and_then(|observer| {
                                if let Some(bank) = observer.body_defaults() {
                                    bank.transition(
                                        (batch.source_id, batch.source_generation),
                                        now,
                                        Transition {
                                            tone_id,
                                            at: spec.onset,
                                            class: if spec.onset == now {
                                                Class::OnsetNow
                                            } else {
                                                Class::DelayedOnset
                                            },
                                            active_at_candidate: false,
                                            body_generation,
                                            routed: [
                                                batch.routing.to_habitat,
                                                batch.routing.to_presentation,
                                            ],
                                        },
                                    );
                                }
                                observer.command(
                                    batch,
                                    tone_id,
                                    Some(spec.onset),
                                    now,
                                    "accepted",
                                    ActionKind::Onset,
                                )
                            });
                            if let Some(packet) = candidate_packet.as_mut()
                                && packet.request.is_some_and(|r| r.tone_id == tone_id)
                                && let Some(observer) = self.action_observer.as_mut()
                            {
                                packet.trace =
                                    outcome_slot.and_then(|slot| observer.freeze_onset_trace(slot));
                            }
                            if let (Some(_), Some(slot), Some(capture)) = (
                                self.action_observer.as_mut(),
                                outcome_slot,
                                self.body_capture.as_ref(),
                            ) && let Some(token) =
                                capture.token(batch.source_id, batch.source_generation)
                                && let Some(mut input) = capture.prediction_input(
                                    token,
                                    now,
                                    spec.onset,
                                    tone.prediction_parameters(None),
                                )
                            {
                                input.scheduled_release = scheduled_release;
                                prediction = Some((slot, input, key));
                            }
                            debug!(
                                target: "phonation::tone_on",
                                source_id = batch.source_id,
                                tone_id,
                                onset = spec.onset,
                                freq_hz = spec.freq_hz,
                                amp = spec.amp
                            );
                            self.tones.insert(
                                key,
                                RoutedTone {
                                    tone,
                                    scheduled_release,
                                    routing: batch.routing,
                                    self_slot: self.self_sound.iter().position(|sound| {
                                        sound
                                            .as_ref()
                                            .is_some_and(|sound| sound.source_id == batch.source_id)
                                    }),
                                    outcome_slots: [outcome_slot, None],
                                    source_generation: batch.source_generation,
                                    body_slot: self.body_capture.as_ref().and_then(|c| {
                                        c.token(batch.source_id, batch.source_generation)
                                    }),
                                },
                            );
                        } else if let Some(observer) = self.action_observer.as_mut() {
                            observer.command(
                                batch,
                                tone_id,
                                Some(spec.onset),
                                now,
                                "invalid_tone",
                                ActionKind::Onset,
                            );
                        }
                    }
                    ToneCmd::Off { tone_id, off_tick } => {
                        let key = ToneKey {
                            source_id: batch.source_id,
                            tone_id,
                        };
                        if let Some(rt) = self.tones.get_mut(&key) {
                            if let Some(observer) = self.action_observer.as_mut() {
                                let status = if rt.source_generation != batch.source_generation {
                                    "generation_mismatch"
                                } else if rt.tone.is_done(off_tick) {
                                    "already_ended"
                                } else {
                                    "accepted"
                                };
                                if status == "accepted"
                                    && let Some(bank) = observer.body_defaults()
                                {
                                    bank.transition(
                                        (batch.source_id, batch.source_generation),
                                        now,
                                        Transition {
                                            tone_id,
                                            at: off_tick,
                                            class: Class::Release,
                                            active_at_candidate: rt.tone.onset() <= off_tick
                                                && !rt.tone.is_done(off_tick),
                                            body_generation: rt.body_slot.map(|t| t.1),
                                            routed: [
                                                rt.routing.to_habitat,
                                                rt.routing.to_presentation,
                                            ],
                                        },
                                    );
                                }
                                // Routing belongs to the sounding tone, not this command batch.
                                let owner = PhonationBatch {
                                    intrinsic_period_sec: batch.intrinsic_period_sec,
                                    source_id: key.source_id,
                                    source_generation: rt.source_generation,
                                    routing: rt.routing,
                                    ..Default::default()
                                };
                                if status == "accepted"
                                    && let Some(slot) = rt.outcome_slots[1].take()
                                {
                                    observer.interrupt(
                                        slot,
                                        (key.source_id, rt.source_generation, tone_id),
                                        now,
                                        "superseded",
                                    );
                                }
                                let slot = observer.command(
                                    &owner,
                                    tone_id,
                                    Some(off_tick),
                                    now,
                                    status,
                                    ActionKind::Release,
                                );
                                if let Some(slot) = slot {
                                    rt.outcome_slots[1] = Some(slot);
                                    if let (Some(capture), Some(token)) =
                                        (self.body_capture.as_ref(), rt.body_slot)
                                        && let Some(input) = capture.prediction_input(
                                            token,
                                            now,
                                            off_tick,
                                            rt.tone.prediction_parameters(Some(off_tick)),
                                        )
                                    {
                                        prediction = Some((slot, input, key));
                                    }
                                }
                            }
                            if rt.source_generation == batch.source_generation {
                                rt.tone.note_off(off_tick);
                            }
                        } else if let Some(observer) = self.action_observer.as_mut() {
                            observer.command(
                                batch,
                                tone_id,
                                Some(off_tick),
                                now,
                                "missing_tone",
                                ActionKind::Release,
                            );
                        }
                    }
                    ToneCmd::Update { .. } => {}
                }
                if let Some((slot, mut input, command_key)) = prediction {
                    input.control = self
                        .tones
                        .get(&command_key)
                        .map(|rt| rt.tone.prediction_control(now, rhythms));
                    input.sine = self
                        .tones
                        .get(&command_key)
                        .and_then(|rt| rt.tone.prediction_sine(now));
                    let source_id = command_key.source_id;
                    let retained = self
                        .tones
                        .range(
                            ToneKey {
                                source_id,
                                tone_id: 0,
                            }..=ToneKey {
                                source_id,
                                tone_id: u64::MAX,
                            },
                        )
                        .map(|(key, rt)| {
                            (*key != command_key && rt.source_generation == batch.source_generation)
                                .then(|| {
                                    (
                                        &rt.tone,
                                        rt.routing,
                                        rt.scheduled_release,
                                        rt.tone.prediction_control(now, rhythms),
                                    )
                                })
                        });
                    self.action_observer
                        .as_mut()
                        .unwrap()
                        .predict(slot, input, retained);
                }
                if let Some(packet) = candidate_packet {
                    self.action_observer
                        .as_mut()
                        .unwrap()
                        .body_defaults()
                        .unwrap()
                        .pending
                        .push_back(packet);
                }
            }
            for cmd in &batch.cmds {
                let ToneCmd::Update {
                    tone_id,
                    at_tick,
                    update,
                } = *cmd
                else {
                    continue;
                };
                let key = ToneKey {
                    source_id: batch.source_id,
                    tone_id,
                };
                let Some(rt) = self.tones.get_mut(&key) else {
                    continue;
                };
                let tick = at_tick.unwrap_or(now);
                rt.tone.schedule_update(tick, update);
            }
        }
        if let Some(observer) = self.action_observer.as_mut() {
            while let Some(mut packet) =
                observer.body_defaults().and_then(|b| b.pending.pop_front())
            {
                let (source_id, source_generation, body_generation) = packet
                    .request
                    .map(|r| (r.source_id, r.source_generation, r.body_generation))
                    .or(packet
                        .scheduled
                        .map(|r| (r.source_id, r.source_generation, Some(r.body_generation))))
                    .expect("pending candidate has one source");
                packet.external = observer
                    .freeze_energy_context((source_id, source_generation, body_generation), now);
                let record = observer
                    .body_defaults()
                    .and_then(|bank| bank.sampled_record((source_id, source_generation), now))
                    .expect("sampled default survives command application");
                packet.release_trace = observer.freeze_release_trace(
                    (source_id, source_generation),
                    now,
                    record.intrinsic_period_sec,
                );
                let bank = observer
                    .body_defaults()
                    .expect("pending candidate has a bank");
                let mut schedule = super::action_candidates::energy::DefaultSchedule {
                    accepted_transitions: record.accepted_transitions,
                    first_transition: record.default_input,
                    ..Default::default()
                };
                let from = ToneKey {
                    source_id,
                    tone_id: 0,
                };
                let through = ToneKey {
                    source_id,
                    tone_id: u64::MAX,
                };
                let mut supported = true;
                let mut found = packet.request.is_none();
                let mut active = false;
                let mut queued = false;
                for (scanned, (key, rt)) in self.tones.range(from..=through).enumerate() {
                    if scanned == 64
                        || rt.source_generation != source_generation
                        || rt.body_slot.map(|slot| slot.1) != body_generation
                    {
                        supported = false;
                        break;
                    }
                    let (_, amplitude, envelope) = rt.tone.prediction_parameters(None);
                    let planned = rt
                        .scheduled_release
                        .map_or(envelope, |r| envelope.with_release(r.off_sample));
                    active |= planned.onset <= now && now < planned.release_end;
                    let control = Some(rt.tone.prediction_control(now, rhythms));
                    schedule.scheduled_tones += 1;
                    schedule.queued_tones += usize::from(rt.tone.onset() > now);
                    schedule.unsupported_control_tones +=
                        usize::from(control.is_none_or(|c| c.valid_until.is_some()));
                    schedule.amplitude_smoothing_tones += usize::from(control.is_some_and(|c| {
                        c.amplitude_smoothing.is_some_and(|s| s.current != s.target)
                    }));
                    schedule.scheduled_amplitude_tones +=
                        usize::from(control.is_some_and(|c| c.amplitude_updates.is_some()));
                    let energy = super::self_prediction::ToneEnergy {
                        amplitude,
                        envelope,
                        scheduled_release: rt.scheduled_release,
                        control,
                        sine: rt.tone.prediction_sine(now),
                        bank: rt.tone.prediction_bank(now),
                    };
                    queued |= envelope.onset > now
                        && energy
                            .renderer_end_after(now, None)
                            .is_some_and(|end| envelope.onset < end);
                    let routed = [rt.routing.to_habitat, rt.routing.to_presentation];
                    if let Some(request) =
                        packet.request.as_mut().filter(|r| key.tone_id == r.tone_id)
                    {
                        request.recipe = energy;
                        request.routed = routed;
                        found = true;
                    } else {
                        packet.retained.push((key.tone_id, routed, energy));
                    }
                }
                packet.default_schedule = Some(schedule);
                let has_recipe = packet.request.is_some();
                bank.energy.submit(
                    packet,
                    supported && found && (active || queued || has_recipe),
                );
            }
        }
    }
}

fn max_phonation_hold_ticks(time: Timebase) -> Tick {
    let max_sec = 60.0;
    let ticks = time.sec_to_tick(max_sec);
    ticks.max(1)
}

pub(crate) fn modal_phase_seed(a: u64, b: u64, c: u64) -> u64 {
    let mut x = a ^ b.rotate_left(21) ^ c.rotate_left(42) ^ 0x9E37_79B9_7F4A_7C15;
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::phonation_engine::{OnsetKick, ToneUpdate};
    use crate::life::sound::{BodyKind, BodySnapshot, RenderModulatorSpec, default_release_ticks};
    use crate::life::voice::{PhonationBatch, ToneSpec};

    fn outcome_batch(kind: BodyKind) -> PhonationBatch {
        PhonationBatch {
            body_policy: None,
            body_opportunity: None,
            intrinsic_period_sec: None,
            source_id: 2,
            source_generation: 7,
            routing: Routing::default(),
            cmds: vec![ToneCmd::On {
                tone_id: 1,
                kick: OnsetKick { strength: 1. },
            }],
            tones: vec![ToneSpec {
                opportunity: None,
                tone_id: 1,
                onset: 13,
                hold_ticks: Some(80),
                freq_hz: 337.,
                amp: 0.2,
                smoothing_tau_sec: 0.,
                body: BodySnapshot {
                    kind,
                    amp_scale: 1.,
                    brightness: 0.4,
                    inharmonic: 0.,
                    spread: 0.,
                    unison: 1,
                    motion: 0.,
                    ratios: None,
                },
                render_modulator: RenderModulatorSpec::SeqGate { duration_sec: 0.1 },
                adsr: None,
            }],
            onsets: Vec::new(),
        }
    }

    #[test]
    fn conditional_body_energy_keeps_audio_and_learning_unchanged() {
        use crate::life::action_candidates::{OnsetOpportunity, PolicySnapshot};
        use crate::life::action_observation::Observer;
        let time = Timebase {
            fs: 8000.,
            hop: 400,
        };
        for (compound, pending_update, smoothing) in [
            (false, false, false),
            (true, false, false),
            (false, true, false),
            (false, false, true),
        ] {
            let mut tracked = ScheduleRenderer::new(time);
            let mut control = ScheduleRenderer::new(time);
            for renderer in [&mut tracked, &mut control] {
                let mut observer = Observer::new(8000);
                observer.enable_predictions();
                observer.enable_trace(crate::config::TemporalPrivateTraceConfig {
                    tau_sec: 10.,
                    kappa: 2.,
                    strength_max: 3.,
                });
                renderer.action_observer = Some(Box::new(observer));
            }
            for now in (0..4800).step_by(400) {
                let mut batch = outcome_batch(BodyKind::Sine);
                batch.body_policy = Some(PolicySnapshot {
                    at: now,
                    is_alive: true,
                    gate_allows_onset: true,
                });
                batch.tones[0].onset = now;
                batch.tones[0].hold_ticks = Some(4000);
                if pending_update || smoothing {
                    batch.tones[0].render_modulator =
                        RenderModulatorSpec::SeqGate { duration_sec: 2. };
                }
                if smoothing {
                    batch.tones[0].smoothing_tau_sec = 0.5;
                }
                if now == 800 {
                    batch.cmds[0] = ToneCmd::On {
                        tone_id: 2,
                        kick: OnsetKick { strength: 1. },
                    };
                    batch.tones[0].tone_id = 2;
                    batch.tones[0].opportunity = Some(OnsetOpportunity {
                        issued_at: now,
                        at: now,
                        gate: 3,
                        intrinsic_due_at: Some(now),
                        intrinsic_period_ticks: Some(4000),
                        planned_release_at: None,
                    });
                    if compound {
                        batch.cmds.push(ToneCmd::Off {
                            tone_id: 1,
                            off_tick: now,
                        });
                    }
                    if pending_update {
                        batch.cmds.push(ToneCmd::Update {
                            tone_id: 2,
                            at_tick: Some(3000),
                            update: ToneUpdate {
                                target_freq_hz: None,
                                target_amp: Some(0.1),
                                continuous_drive: None,
                            },
                        });
                    }
                } else if now == 400 && smoothing {
                    batch.tones.clear();
                    batch.cmds = vec![ToneCmd::Update {
                        tone_id: 1,
                        at_tick: Some(now),
                        update: ToneUpdate {
                            target_freq_hz: None,
                            target_amp: Some(0.05),
                            continuous_drive: None,
                        },
                    }];
                } else if now != 0 {
                    batch.cmds.clear();
                    batch.tones.clear();
                }
                let mut without_receipt = batch.clone();
                for spec in &mut without_receipt.tones {
                    spec.opportunity = None;
                }
                let audio =
                    tracked.render(std::slice::from_ref(&batch), now, &NeuralRhythms::default());
                let expected = control.render(
                    std::slice::from_ref(&without_receipt),
                    now,
                    &NeuralRhythms::default(),
                );
                assert_eq!(audio.habitat, expected.habitat);
                assert_eq!(audio.presentation, expected.presentation);
                if smoothing && now == 800 {
                    let retained = tracked
                        .tones
                        .iter()
                        .find(|(key, _)| key.tone_id == 1)
                        .unwrap()
                        .1;
                    assert!(
                        retained
                            .tone
                            .prediction_control(now + 400, &NeuralRhythms::default())
                            .amplitude_smoothing
                            .is_some()
                    );
                }
            }
            let observer = tracked.action_observer.as_mut().unwrap();
            let control = control.action_observer.as_mut().unwrap();
            observer.finish();
            control.finish();
            assert_eq!(observer.snapshot.commands, if compound { 3 } else { 2 });
            assert_eq!(observer.snapshot.prediction, control.snapshot.prediction);
            assert_eq!(
                observer.snapshot.participation_trace,
                control.snapshot.participation_trace
            );
            assert_eq!(
                observer.drain().collect::<Vec<_>>(),
                control.drain().collect::<Vec<_>>()
            );
            let stats = observer.snapshot.body_defaults.unwrap().candidate_energy;
            assert_eq!(stats.unsupported, 0);
            assert_eq!(stats.completed, 1);
            assert_eq!(stats.worker_unsupported, 0);
            assert!(!stats.worker_failed);
            let records: Vec<_> = observer.drain_candidate_energy().collect();
            assert_eq!(records.len(), 1);
            let r = &records[0];
            let schedule = r.default_schedule.unwrap();
            assert_eq!(schedule.accepted_transitions, if compound { 2 } else { 1 });
            assert_eq!(schedule.scheduled_tones, 2);
            assert_eq!(schedule.queued_tones, 0);
            assert_eq!(schedule.unsupported_control_tones, 0);
            assert_eq!(
                schedule.scheduled_amplitude_tones,
                usize::from(pending_update)
            );
            assert_eq!(schedule.amplitude_smoothing_tones, usize::from(smoothing));
            if smoothing || pending_update {
                assert!(
                    r.candidates.iter().all(|c| c
                        .buses
                        .iter()
                        .flatten()
                        .flatten()
                        .all(|w| w.mean.is_some()))
                );
            }
            assert_eq!(
                (
                    r.source_id,
                    r.source_generation,
                    r.tone_id,
                    r.issued_at,
                    r.decision_at
                ),
                (2, 7, Some(2), 800, 800)
            );
            assert_eq!(r.retained_tones, 1);
            let origin = r.onset_trace_origin.unwrap();
            assert_eq!(
                (
                    origin.command_id,
                    origin.source_id,
                    origin.source_generation,
                    origin.issued_at_sample,
                    origin.bus
                ),
                (Some(2), 2, 7, 800, 0)
            );
            assert_eq!(r.onset_trace_default_sample, Some(801));
            for c in &r.candidates {
                assert_eq!(c.onset_trace.is_some(), c.input.excitation_at.is_some());
                if let Some(trace) = c.onset_trace {
                    assert_eq!(trace.event_sample, c.input.excitation_at.unwrap() + 1);
                    assert!(
                        trace
                            .fits
                            .iter()
                            .all(|fit| fit.difference.is_none() && fit.unassigned == 1.)
                    );
                }
            }
            for class in [
                Class::OnsetNow,
                Class::DelayedOnset,
                Class::Wait,
                Class::Skip,
                Class::Continue,
                Class::Release,
                Class::Gap,
            ] {
                assert!(r.candidates.iter().any(|c| c.input.class == class));
            }
            assert!(
                r.candidates[0]
                    .buses
                    .iter()
                    .flatten()
                    .flatten()
                    .all(|w| w.difference == w.mean.map(|_| 0.))
            );
            if pending_update {
                assert!(r.candidates[0].buses[0][0].unwrap().mean.unwrap() > 0.);
                assert!(r.candidates[0].buses[0][1].unwrap().mean.is_some());
            }
            assert_eq!(control.drain_candidate_energy().count(), 0);
        }
    }

    /// The control keeps observation and learning but strips the policy facts and
    /// opportunities that make candidates due, so any effect of evaluating them would show.
    #[test]
    fn candidate_evaluation_leaves_audio_and_learning_unchanged_for_every_body() {
        use crate::life::action_candidates::{OnsetOpportunity, PolicySnapshot};
        use crate::life::action_observation::Observer;
        let time = Timebase { fs: 8000., hop: 64 };
        for kind in [BodyKind::Sine, BodyKind::Harmonic, BodyKind::Modal] {
            let mut tracked = ScheduleRenderer::new(time);
            let mut control = ScheduleRenderer::new(time);
            for renderer in [&mut tracked, &mut control] {
                let mut observer = Observer::new(8000);
                observer.enable_predictions();
                observer.enable_trace(crate::config::TemporalPrivateTraceConfig {
                    tau_sec: 10.,
                    kappa: 2.,
                    strength_max: 3.,
                });
                renderer.action_observer = Some(Box::new(observer));
                let kernel = crate::core::nsgt_kernel::NsgtKernelLog2::new(
                    crate::core::nsgt_kernel::NsgtLog2Config {
                        fs: 8000.,
                        overlap: 0.75,
                        nfft_override: Some(256),
                        ..Default::default()
                    },
                    crate::core::log2space::Log2Space::new(100., 2000., 12),
                    None,
                    crate::core::nsgt_kernel::PowerMode::Coherent,
                );
                let mut capture = crate::temporal_cognition::body::Capture::spawn(
                    crate::core::nsgt_rt::RtNsgtKernelLog2::new(kernel),
                    crate::config::TemporalBodyConfig {
                        means: [0.; 6],
                        deviations: [1.; 6],
                        accent_means: [0.; 2],
                        accent_deviations: [1.; 2],
                    },
                    true,
                    None,
                );
                capture.prepare(
                    std::iter::once((2, 7, outcome_batch(kind).tones[0].body.clone())),
                    0,
                );
                renderer.body_capture = Some(capture);
            }
            for now in (0..4800).step_by(64) {
                let mut batch = outcome_batch(kind);
                batch.body_policy = Some(PolicySnapshot {
                    at: now,
                    is_alive: true,
                    gate_allows_onset: true,
                });
                batch.intrinsic_period_sec = Some(0.5);
                batch.tones[0].hold_ticks = Some(1600);
                if now > 0 {
                    batch.cmds.clear();
                    batch.tones.clear();
                }
                if now == 896 {
                    let mut tone = outcome_batch(kind).tones.remove(0);
                    tone.tone_id = 2;
                    tone.onset = 896;
                    tone.hold_ticks = Some(896);
                    tone.opportunity = Some(OnsetOpportunity {
                        issued_at: 896,
                        at: 896,
                        gate: 1,
                        intrinsic_due_at: Some(896),
                        intrinsic_period_ticks: Some(4000),
                        planned_release_at: None,
                    });
                    batch.tones.push(tone);
                    batch.cmds.push(ToneCmd::On {
                        tone_id: 2,
                        kick: OnsetKick { strength: 1. },
                    });
                }
                let mut reference_batch = batch.clone();
                reference_batch.body_policy = None;
                for tone in &mut reference_batch.tones {
                    tone.opportunity = None;
                }
                let actual = tracked.render(&[batch], now, &NeuralRhythms::default());
                let reference = control.render(&[reference_batch], now, &NeuralRhythms::default());
                assert_eq!(actual.habitat, reference.habitat, "{kind:?} at {now}");
                assert_eq!(
                    actual.presentation, reference.presentation,
                    "{kind:?} at {now}"
                );
            }
            let observer = tracked.action_observer.as_mut().unwrap();
            let reference = control.action_observer.as_mut().unwrap();
            observer.finish();
            reference.finish();
            assert_eq!(
                observer.snapshot.prediction, reference.snapshot.prediction,
                "{kind:?}"
            );
            assert_eq!(
                observer.snapshot.participation_trace, reference.snapshot.participation_trace,
                "{kind:?}"
            );
            let learned: Vec<_> = observer.drain().collect();
            assert!(!learned.is_empty(), "{kind:?}");
            assert_eq!(learned, reference.drain().collect::<Vec<_>>(), "{kind:?}");
            assert_eq!(reference.drain_candidate_energy().count(), 0);
            let records: Vec<_> = observer.drain_candidate_energy().collect();
            let onset = records
                .iter()
                .find(|r| r.candidates.iter().any(|c| c.input.excitation_at.is_some()))
                .unwrap_or_else(|| panic!("{kind:?}: no onset candidates in {}", records.len()));
            let window = onset.candidates[0].buses[0][0].unwrap();
            assert!(window.coherent_bins > 0, "{kind:?}");
        }
    }

    #[test]
    fn idle_queued_candidates_preserve_audio_learning_and_owned_schedule() {
        use crate::life::action_candidates::{OnsetOpportunity, PolicySnapshot};
        use crate::life::action_observation::Observer;
        use crate::life::sound::ToneAdsr;
        let time = Timebase { fs: 8000., hop: 64 };
        for case in [
            "keep",
            "cancel",
            "update",
            "upgrade",
            "source_generation",
            "body_generation",
            "capacity",
        ] {
            for routing in [
                Routing {
                    to_habitat: true,
                    to_presentation: false,
                },
                Routing {
                    to_habitat: false,
                    to_presentation: true,
                },
                Routing::default(),
            ] {
                let mut tracked = ScheduleRenderer::new(time);
                let mut control = ScheduleRenderer::new(time);
                for renderer in [&mut tracked, &mut control] {
                    let mut observer = Observer::new(8000);
                    observer.enable_predictions();
                    observer.enable_trace(crate::config::TemporalPrivateTraceConfig {
                        tau_sec: 10.,
                        kappa: 2.,
                        strength_max: 3.,
                    });
                    renderer.action_observer = Some(Box::new(observer));
                    let kernel = crate::core::nsgt_kernel::NsgtKernelLog2::new(
                        crate::core::nsgt_kernel::NsgtLog2Config {
                            fs: 8000.,
                            overlap: 0.75,
                            nfft_override: Some(256),
                            ..Default::default()
                        },
                        crate::core::log2space::Log2Space::new(100., 2000., 12),
                        None,
                        crate::core::nsgt_kernel::PowerMode::Coherent,
                    );
                    let mut capture = crate::temporal_cognition::body::Capture::spawn(
                        crate::core::nsgt_rt::RtNsgtKernelLog2::new(kernel),
                        crate::config::TemporalBodyConfig {
                            means: [0.; 6],
                            deviations: [1.; 6],
                            accent_means: [0.; 2],
                            accent_deviations: [1.; 2],
                        },
                        true,
                        None,
                    );
                    capture.prepare(
                        std::iter::once((
                            2,
                            7,
                            outcome_batch(BodyKind::Sine).tones[0].body.clone(),
                        )),
                        0,
                    );
                    renderer.body_capture = Some(capture);
                }
                for now in (0..3200).step_by(64) {
                    let mut batch = outcome_batch(BodyKind::Sine);
                    batch.routing = routing;
                    batch.body_policy = Some(PolicySnapshot {
                        at: now,
                        is_alive: true,
                        gate_allows_onset: true,
                    });
                    batch.intrinsic_period_sec = Some(0.5);
                    batch.tones[0].onset = 1600;
                    batch.tones[0].hold_ticks = Some(896);
                    batch.tones[0].adsr = Some(ToneAdsr {
                        attack_sec: 0.,
                        decay_sec: 0.,
                        sustain_level: 1.,
                        release_sec: 0.05,
                    });
                    if now == 0 && case == "capacity" {
                        for id in 2..=65 {
                            let mut tone = batch.tones[0].clone();
                            tone.tone_id = id;
                            batch.tones.push(tone);
                            batch.cmds.push(ToneCmd::On {
                                tone_id: id,
                                kick: OnsetKick { strength: 1. },
                            });
                        }
                    }
                    if now > 0 {
                        batch.cmds.clear();
                        batch.tones.clear();
                    }
                    if now == 448 && case == "cancel" {
                        batch.cmds.push(ToneCmd::Off {
                            tone_id: 1,
                            off_tick: 448,
                        });
                    }
                    if now == 448 && case == "update" {
                        batch.cmds.push(ToneCmd::Update {
                            tone_id: 1,
                            at_tick: Some(2000),
                            update: ToneUpdate {
                                target_freq_hz: None,
                                target_amp: Some(0.1),
                                continuous_drive: None,
                            },
                        });
                    }
                    if now == 896 && case == "upgrade" {
                        let mut tone = outcome_batch(BodyKind::Sine).tones.remove(0);
                        tone.tone_id = 2;
                        tone.onset = 896;
                        tone.hold_ticks = Some(896);
                        tone.opportunity = Some(OnsetOpportunity {
                            issued_at: 896,
                            at: 896,
                            gate: 1,
                            intrinsic_due_at: Some(896),
                            intrinsic_period_ticks: Some(4000),
                            planned_release_at: None,
                        });
                        batch.tones.push(tone);
                        batch.cmds.push(ToneCmd::On {
                            tone_id: 2,
                            kick: OnsetKick { strength: 1. },
                        });
                    }
                    let mut reference_batch = batch.clone();
                    reference_batch.body_policy = None;
                    for tone in &mut reference_batch.tones {
                        tone.opportunity = None;
                    }
                    let actual = tracked.render(&[batch], now, &NeuralRhythms::default());
                    let reference =
                        control.render(&[reference_batch], now, &NeuralRhythms::default());
                    assert_eq!(actual.habitat, reference.habitat, "{case} at {now}");
                    assert_eq!(
                        actual.presentation, reference.presentation,
                        "{case} at {now}"
                    );
                    if now == 0 {
                        for renderer in [&mut tracked, &mut control] {
                            let rt = renderer.tones.values_mut().next().unwrap();
                            if case == "source_generation" {
                                rt.source_generation += 1;
                            }
                            if case == "body_generation" {
                                rt.body_slot = Some((0, 123));
                            }
                        }
                    }
                }
                let observer = tracked.action_observer.as_mut().unwrap();
                let reference = control.action_observer.as_mut().unwrap();
                observer.finish();
                reference.finish();
                assert_eq!(
                    observer.snapshot.prediction, reference.snapshot.prediction,
                    "{case}"
                );
                assert_eq!(
                    observer.snapshot.participation_trace, reference.snapshot.participation_trace,
                    "{case}"
                );
                assert_eq!(
                    observer.drain().collect::<Vec<_>>(),
                    reference.drain().collect::<Vec<_>>(),
                    "{case}"
                );
                assert_eq!(reference.drain_candidate_energy().count(), 0);
                let stats = observer.snapshot.body_defaults.unwrap().candidate_energy;
                assert_eq!(stats.worker_unsupported, 0, "{case}");
                assert_eq!(stats.capacity_dropped + stats.output_dropped, 0, "{case}");
                let records: Vec<_> = observer.drain_candidate_energy().collect();
                if matches!(
                    case,
                    "cancel" | "source_generation" | "body_generation" | "capacity"
                ) {
                    assert!(records.is_empty(), "{case}");
                    if case == "source_generation" {
                        assert_eq!(stats.submitted + stats.unsupported, 0);
                    } else {
                        assert!(stats.unsupported > 0, "{case}");
                    }
                    continue;
                }
                assert_eq!(
                    records.iter().filter(|r| r.issued_at == 448).count(),
                    1,
                    "{case}: {stats:?}"
                );
                let first = records.iter().find(|r| r.issued_at == 448).unwrap();
                assert_eq!(first.default_basis, "accepted_queued_onset");
                assert_eq!(first.default_input.at, 1600);
                assert_eq!(first.default_schedule.unwrap().queued_tones, 1);
                assert_eq!(first.retained_tones, 1);
                for bus in 0..2 {
                    let routed = [routing.to_habitat, routing.to_presentation][bus];
                    let mean = first.candidates[0].buses[bus][0].unwrap().mean;
                    if !routed {
                        assert_eq!(mean, Some(0.));
                    } else {
                        assert!(mean.unwrap() > 0.);
                    }
                }
                if case == "upgrade" {
                    let upgraded: Vec<_> = records.iter().filter(|r| r.issued_at == 896).collect();
                    assert_eq!(upgraded.len(), 1);
                    assert_eq!(upgraded[0].default_basis, "accepted_onset_recipe");
                    assert_eq!(upgraded[0].tone_id, Some(2));
                    assert_eq!(upgraded[0].retained_tones, 1);
                }
                assert!(
                    records
                        .iter()
                        .any(|r| r.issued_at == 1792 && r.default_basis == "active_body")
                );
            }
        }
    }

    #[test]
    fn compound_default_freezes_after_commands_with_queued_routes_and_unknown_controls() {
        use crate::life::action_candidates::{OnsetOpportunity, PolicySnapshot};
        use crate::life::action_observation::Observer;
        use crate::life::sound::ToneAdsr;
        let time = Timebase {
            fs: 8000.,
            hop: 400,
        };
        for case in [
            "off_first",
            "on_first",
            "cancel",
            "update",
            "pitch_update",
            "source_generation",
            "body_generation",
            "capacity",
        ] {
            let mut tracked = ScheduleRenderer::new(time);
            let mut control = ScheduleRenderer::new(time);
            let mut observer = Observer::new(8000);
            observer.enable_predictions();
            tracked.action_observer = Some(Box::new(observer));
            let mut batch = outcome_batch(BodyKind::Sine);
            batch.routing = Routing {
                to_habitat: false,
                to_presentation: true,
            };
            batch.tones[0].onset = 0;
            batch.tones[0].hold_ticks = Some(8000);
            batch.tones[0].render_modulator = RenderModulatorSpec::SeqGate { duration_sec: 2. };
            batch.tones[0].adsr = Some(ToneAdsr {
                attack_sec: 0.,
                decay_sec: 0.,
                sustain_level: 1.,
                release_sec: 0.05,
            });
            for renderer in [&mut tracked, &mut control] {
                renderer.render(std::slice::from_ref(&batch), 0, &NeuralRhythms::default());
                let old = renderer.tones.values_mut().next().unwrap();
                if case == "source_generation" {
                    old.source_generation += 1;
                }
                if case == "body_generation" {
                    old.body_slot = Some((0, 123));
                }
            }
            batch.body_policy = Some(PolicySnapshot {
                at: 400,
                is_alive: true,
                gate_allows_onset: true,
            });
            batch.intrinsic_period_sec = Some(0.5);
            batch.routing = Routing {
                to_habitat: true,
                to_presentation: false,
            };
            batch.tones[0].tone_id = 2;
            batch.tones[0].onset = 400;
            batch.tones[0].adsr.as_mut().unwrap().release_sec = 0.;
            batch.tones[0].opportunity = Some(OnsetOpportunity {
                issued_at: 400,
                at: 400,
                gate: 1,
                intrinsic_due_at: Some(400),
                intrinsic_period_ticks: Some(4000),
                planned_release_at: None,
            });
            let mut queued = batch.tones[0].clone();
            queued.tone_id = 3;
            queued.onset = 2800;
            queued.opportunity.as_mut().unwrap().at = 2800;
            batch.tones.push(queued);
            batch.cmds = vec![
                ToneCmd::Off {
                    tone_id: 1,
                    off_tick: 400,
                },
                ToneCmd::On {
                    tone_id: 2,
                    kick: OnsetKick { strength: 1. },
                },
                ToneCmd::On {
                    tone_id: 3,
                    kick: OnsetKick { strength: 1. },
                },
            ];
            if case == "on_first" {
                batch.cmds.swap(0, 1);
            }
            if case == "cancel" {
                batch.cmds.push(ToneCmd::Off {
                    tone_id: 2,
                    off_tick: 400,
                });
            }
            if matches!(case, "update" | "pitch_update") {
                batch.cmds.push(ToneCmd::Update {
                    tone_id: 2,
                    at_tick: Some(800),
                    update: ToneUpdate {
                        target_freq_hz: (case == "pitch_update").then_some(440.),
                        target_amp: Some(0.1),
                        continuous_drive: None,
                    },
                });
            }
            if case == "capacity" {
                for id in 4..=65 {
                    let mut extra = batch.tones[1].clone();
                    extra.tone_id = id;
                    extra.opportunity = None;
                    batch.tones.push(extra);
                    batch.cmds.push(ToneCmd::On {
                        tone_id: id,
                        kick: OnsetKick { strength: 1. },
                    });
                }
            }
            let actual =
                tracked.render(std::slice::from_ref(&batch), 400, &NeuralRhythms::default());
            let reference =
                control.render(std::slice::from_ref(&batch), 400, &NeuralRhythms::default());
            assert_eq!(actual.habitat, reference.habitat, "{case}");
            assert_eq!(actual.presentation, reference.presentation, "{case}");
            let observer = tracked.action_observer.as_mut().unwrap();
            observer.finish();
            let stats = observer.snapshot.body_defaults.unwrap().candidate_energy;
            let records: Vec<_> = observer.drain_candidate_energy().collect();
            if matches!(case, "source_generation" | "body_generation" | "capacity") {
                assert_eq!(stats.unsupported, 1, "{case}");
                assert!(records.is_empty(), "{case}");
                continue;
            }
            assert_eq!(stats.submitted, 1);
            assert_eq!(records.len(), 1);
            let record = &records[0];
            assert_eq!(record.default_input.class, Class::OnsetNow);
            assert_eq!(record.default_routed, [true, true]);
            assert_eq!(record.retained_tones, 2);
            let schedule = record.default_schedule.unwrap();
            assert_eq!(
                schedule.first_transition.unwrap().class,
                if case == "on_first" {
                    Class::OnsetNow
                } else {
                    Class::Release
                }
            );
            assert_eq!(
                schedule.accepted_transitions,
                if case == "cancel" { 4 } else { 3 }
            );
            assert_eq!(schedule.scheduled_tones, 3);
            assert_eq!(schedule.queued_tones, 1);
            assert_eq!(
                schedule.unsupported_control_tones,
                usize::from(case == "pitch_update")
            );
            assert_eq!(
                schedule.scheduled_amplitude_tones,
                usize::from(case == "update")
            );
            let default = &record.candidates[0];
            let habitat = default.buses[0][0].unwrap();
            match case {
                // A tone released at its onset still renders a brief attack-release sliver.
                "cancel" => assert!(habitat.mean.unwrap() < 1e-6),
                "pitch_update" => assert_eq!(habitat.mean, None),
                _ => assert!(habitat.mean.unwrap() > 0.),
            }
            assert!(default.buses[1][0].unwrap().mean.unwrap() > 0.);
            assert_eq!(habitat.difference, habitat.mean.map(|_| 0.));
        }
    }

    #[test]
    fn hop_default_uses_accepted_recipe_and_later_batches_without_extra_learning() {
        use crate::life::action_candidates::{OnsetOpportunity, PolicySnapshot};
        use crate::life::action_observation::Observer;
        use crate::life::sound::ToneAdsr;
        let time = Timebase {
            fs: 8000.,
            hop: 400,
        };
        for case in ["duplicate", "invalid", "later_release", "later_update"] {
            let mut tracked = ScheduleRenderer::new(time);
            let mut control = ScheduleRenderer::new(time);
            for r in [&mut tracked, &mut control] {
                let mut observer = Observer::new(8000);
                observer.enable_predictions();
                observer.enable_trace(crate::config::TemporalPrivateTraceConfig {
                    tau_sec: 10.,
                    kappa: 2.,
                    strength_max: 3.,
                });
                r.action_observer = Some(Box::new(observer));
            }
            for now in (0..2800).step_by(400) {
                let mut initial = outcome_batch(BodyKind::Sine);
                initial.tones[0].onset = 0;
                initial.tones[0].hold_ticks = Some(8000);
                initial.tones[0].adsr = Some(ToneAdsr {
                    attack_sec: 0.,
                    decay_sec: 0.,
                    sustain_level: 1.,
                    release_sec: 0.,
                });
                initial.routing = Routing {
                    to_habitat: false,
                    to_presentation: true,
                };
                let batches = if now == 0 {
                    vec![initial.clone()]
                } else if now == 800 {
                    let mut issue = initial.clone();
                    issue.routing = Routing {
                        to_habitat: true,
                        to_presentation: false,
                    };
                    issue.body_policy = Some(PolicySnapshot {
                        at: now,
                        is_alive: true,
                        gate_allows_onset: true,
                    });
                    issue.tones[0].tone_id = if case == "duplicate" { 1 } else { 2 };
                    issue.tones[0].amp = if case == "invalid" { 0. } else { 0.2 };
                    issue.tones[0].onset = now;
                    issue.tones[0].opportunity = Some(OnsetOpportunity {
                        issued_at: now,
                        at: now,
                        gate: 2,
                        intrinsic_due_at: Some(now),
                        intrinsic_period_ticks: Some(4000),
                        planned_release_at: None,
                    });
                    issue.cmds = vec![ToneCmd::On {
                        tone_id: issue.tones[0].tone_id,
                        kick: OnsetKick { strength: 1. },
                    }];
                    let mut later = issue.clone();
                    if matches!(case, "duplicate" | "invalid") {
                        later.tones[0].tone_id = 3;
                        later.tones[0].amp = 0.2;
                        later.cmds = vec![ToneCmd::On {
                            tone_id: 3,
                            kick: OnsetKick { strength: 1. },
                        }];
                    } else {
                        later.tones.clear();
                        later.cmds = vec![if case == "later_release" {
                            ToneCmd::Off {
                                tone_id: 2,
                                off_tick: now,
                            }
                        } else {
                            ToneCmd::Update {
                                tone_id: 2,
                                at_tick: Some(1200),
                                update: ToneUpdate {
                                    target_freq_hz: None,
                                    target_amp: Some(0.1),
                                    continuous_drive: None,
                                },
                            }
                        }];
                    }
                    vec![issue, later]
                } else {
                    vec![]
                };
                let mut without_receipts = batches.clone();
                for b in &mut without_receipts {
                    for spec in &mut b.tones {
                        spec.opportunity = None;
                    }
                }
                let actual = tracked.render(&batches, now, &NeuralRhythms::default());
                let reference = control.render(&without_receipts, now, &NeuralRhythms::default());
                assert_eq!(actual.habitat, reference.habitat, "{case}");
                assert_eq!(actual.presentation, reference.presentation, "{case}");
            }
            let observer = tracked.action_observer.as_mut().unwrap();
            let control = control.action_observer.as_mut().unwrap();
            observer.finish();
            control.finish();
            assert_eq!(
                observer.snapshot.prediction, control.snapshot.prediction,
                "{case}"
            );
            assert_eq!(
                observer.snapshot.participation_trace, control.snapshot.participation_trace,
                "{case}"
            );
            assert_eq!(
                observer.drain().collect::<Vec<_>>(),
                control.drain().collect::<Vec<_>>(),
                "{case}"
            );
            let stats = observer.snapshot.body_defaults.unwrap().candidate_energy;
            assert_eq!(stats.submitted, 1, "{case}");
            assert_eq!(stats.completed, 1, "{case}");
            let defaults: Vec<_> = observer.drain_body_defaults().collect();
            let issued = defaults.iter().find(|r| r.issued_at == 800).unwrap();
            assert_eq!(
                issued.onset_recipe_count,
                if matches!(case, "duplicate" | "invalid") {
                    2
                } else {
                    1
                }
            );
            assert_eq!(
                issued.onset_recipe_tone_id,
                Some(if case == "duplicate" { 1 } else { 2 })
            );
            let records: Vec<_> = observer.drain_candidate_energy().collect();
            assert_eq!(records.len(), 1);
            let r = &records[0];
            assert_eq!(
                r.tone_id,
                Some(if matches!(case, "duplicate" | "invalid") {
                    3
                } else {
                    2
                })
            );
            assert!(r.onset_trace_origin.is_some());
            let schedule = r.default_schedule.unwrap();
            assert_eq!(
                schedule.accepted_transitions,
                if case == "later_release" { 2 } else { 1 }
            );
            assert_eq!(
                schedule.scheduled_amplitude_tones,
                usize::from(case == "later_update")
            );
            match case {
                "later_release" => {
                    assert!(r.candidates[0].buses[0][0].unwrap().mean.unwrap() < 1e-6)
                }
                _ => assert!(r.candidates[0].buses[0][0].unwrap().mean.unwrap() > 0.),
            }
        }
    }

    #[test]
    fn live_body_defaults_follow_accepted_order_and_owned_tones_without_changing_audio() {
        use crate::life::action_candidates::PolicySnapshot;
        use crate::life::action_observation::Observer;
        let time = Timebase {
            fs: 8000.,
            hop: 400,
        };
        let mut renderer = ScheduleRenderer::new(time);
        let mut observer = Observer::new(8000);
        observer.enable_predictions();
        renderer.action_observer = Some(Box::new(observer));
        let mut control = ScheduleRenderer::new(time);
        let mut actor = outcome_batch(BodyKind::Modal);
        actor.tones[0].onset = 800;
        actor.tones[0].hold_ticks = Some(4000);
        let mut records = Vec::new();
        for now in (0..2400).step_by(400) {
            let mut batch = actor.clone();
            batch.body_policy = Some(PolicySnapshot {
                at: now,
                is_alive: true,
                gate_allows_onset: true,
            });
            if now > 0 {
                batch.cmds.clear();
                batch.tones.clear();
            }
            // Commands may carry changed routing; an existing body's buses do not change.
            if now > 0 {
                batch.routing.to_habitat = false;
            }
            if now == 1200 {
                batch.cmds = vec![
                    ToneCmd::On {
                        tone_id: 1,
                        kick: OnsetKick { strength: 1. },
                    },
                    ToneCmd::Off {
                        tone_id: 99,
                        off_tick: 1200,
                    },
                    ToneCmd::Off {
                        tone_id: 1,
                        off_tick: 1250,
                    },
                    ToneCmd::On {
                        tone_id: 2,
                        kick: OnsetKick { strength: 1. },
                    },
                ];
                let mut spec = actor.tones[0].clone();
                spec.tone_id = 2;
                spec.onset = 1300;
                batch.tones.push(spec);
            }
            if now == 1600 {
                batch.source_generation += 1;
            }
            let batches = if now == 2000 {
                &[][..]
            } else {
                std::slice::from_ref(&batch)
            };
            let actual = renderer.render(batches, now, &NeuralRhythms::default());
            let expected = control.render(batches, now, &NeuralRhythms::default());
            assert_eq!(actual.habitat, expected.habitat);
            assert_eq!(actual.presentation, expected.presentation);
            records.extend(
                renderer
                    .action_observer
                    .as_mut()
                    .unwrap()
                    .drain_body_defaults(),
            );
        }
        assert_eq!(records.len(), 5);
        assert_eq!(records[0].default_input.unwrap().class, Class::DelayedOnset);
        assert_eq!(records[0].default_input.unwrap().at, 800);
        assert_eq!(records[1].queued_tones, 1);
        assert_eq!(records[1].status, "queued_onset");
        assert_eq!(records[2].active_bus_tones, [1, 1]);
        assert_eq!(records[2].default_input.unwrap().class, Class::Continue);
        assert_eq!(records[3].default_input.unwrap().class, Class::Release);
        assert_eq!(records[3].default_input.unwrap().at, 1250);
        assert_eq!(records[3].default_tone_id, Some(1));
        assert_eq!(records[3].default_routed, Some([true, true]));
        assert_eq!(records[3].accepted_transitions, 2);
        assert_eq!(records[4].active_tones, 0);
        assert!(records[4].other_source_generation_tones > 0);
        assert_eq!(records[4].default_input, None);
        assert!(records.iter().all(|r| r.policy.unwrap().at == r.issued_at));
        let observer = renderer.action_observer.as_mut().unwrap();
        assert_eq!(observer.snapshot.body_defaults.unwrap().retired, 2);
        assert_eq!(observer.snapshot.body_defaults.unwrap().latest, None);
        renderer.render(std::slice::from_ref(&actor), 0, &NeuralRhythms::default());
        assert_eq!(
            renderer
                .action_observer
                .as_mut()
                .unwrap()
                .drain_body_defaults()
                .count(),
            0
        );
    }

    #[test]
    fn live_body_default_does_not_turn_pre_onset_cancellation_into_supported_release() {
        use crate::life::action_observation::Observer;
        let mut renderer = ScheduleRenderer::new(Timebase {
            fs: 8000.,
            hop: 400,
        });
        let mut observer = Observer::new(8000);
        observer.enable_predictions();
        renderer.action_observer = Some(Box::new(observer));
        let mut actor = outcome_batch(BodyKind::Sine);
        actor.tones[0].onset = 2000;
        renderer.render(std::slice::from_ref(&actor), 0, &NeuralRhythms::default());
        renderer
            .action_observer
            .as_mut()
            .unwrap()
            .drain_body_defaults()
            .for_each(drop);
        actor.cmds = vec![ToneCmd::Off {
            tone_id: 1,
            off_tick: 500,
        }];
        renderer.render(std::slice::from_ref(&actor), 400, &NeuralRhythms::default());
        let record = renderer
            .action_observer
            .as_mut()
            .unwrap()
            .drain_body_defaults()
            .next()
            .unwrap();
        assert_eq!(record.queued_tones, 1);
        assert_eq!(record.accepted_transitions, 1);
        assert_eq!(record.default_input, None);
        assert_eq!(record.status, "accepted_transition_unmapped");
    }

    #[test]
    fn onset_outcomes_match_private_rendered_audio_without_changing_either_mix() {
        use super::super::action_observation::{ACTIVITY_THRESHOLD, Observer};
        let time = Timebase { fs: 8000., hop: 32 };
        for kind in [BodyKind::Sine, BodyKind::Harmonic, BodyKind::Modal] {
            for routing in [(true, true), (true, false), (false, true), (false, false)] {
                let mut actor = outcome_batch(kind);
                actor.routing.to_habitat = routing.0;
                actor.routing.to_presentation = routing.1;
                let mut other = outcome_batch(BodyKind::Sine);
                other.source_id = 99;
                other.tones[0].onset = 0;
                other.tones[0].freq_hz = 800.;
                other.tones[0].amp = 0.8;
                let all = [actor.clone(), other];
                let mut renderer = ScheduleRenderer::new(time);
                renderer.action_observer = Some(Box::new(Observer::new(8000)));
                let mut control = ScheduleRenderer::new(time);
                let mut isolated = ScheduleRenderer::new(time);
                let mut private = [Vec::new(), Vec::new()];
                let mut outcomes = Vec::new();
                for now in (0..256).step_by(32) {
                    let batches = if now == 0 { &all[..] } else { &[] };
                    let actual = renderer.render(batches, now, &NeuralRhythms::default());
                    let reference = control.render(batches, now, &NeuralRhythms::default());
                    assert_eq!(actual.habitat, reference.habitat);
                    assert_eq!(actual.presentation, reference.presentation);
                    let frame = isolated.render(
                        if now == 0 {
                            std::slice::from_ref(&actor)
                        } else {
                            &[]
                        },
                        now,
                        &NeuralRhythms::default(),
                    );
                    private[0].extend_from_slice(frame.habitat);
                    private[1].extend_from_slice(frame.presentation);
                    outcomes.extend(renderer.action_observer.as_mut().unwrap().drain());
                }
                let result = outcomes.iter().find(|o| o.source_id == 2).unwrap();
                assert_eq!(result.source_generation, 7);
                assert_eq!(result.observed_samples, 160);
                assert_eq!(result.available_at_sample, 192);
                for (bus, pcm) in result.buses.iter().zip(private) {
                    let pcm = &pcm[13..173];
                    let expected = pcm
                        .iter()
                        .position(|v| v.abs() > ACTIVITY_THRESHOLD)
                        .map(|p| p as u64 + 13);
                    assert_eq!(bus.first_activity_sample, expected);
                    if bus.routed {
                        let rms =
                            (pcm.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / 160.).sqrt();
                        assert!((bus.rms.unwrap() - rms).abs() < 1e-12);
                        assert!(bus.first_activity_sample.unwrap() >= 13);
                    } else {
                        assert_eq!(bus.status, "not_routed");
                        assert_eq!(bus.rms, None);
                    }
                }
            }
        }
    }

    #[test]
    fn release_outcomes_match_isolated_tail_and_preserve_both_mixes() {
        use super::super::action_observation::{ACTIVITY_THRESHOLD, Observer};
        use crate::life::sound::ToneAdsr;
        let time = Timebase { fs: 8000., hop: 32 };
        for kind in [BodyKind::Sine, BodyKind::Harmonic, BodyKind::Modal] {
            for routing in [(true, true), (true, false), (false, true), (false, false)] {
                let mut actor = outcome_batch(kind);
                actor.tones[0].hold_ticks = Some(24000);
                actor.tones[0].render_modulator =
                    RenderModulatorSpec::SeqGate { duration_sec: 10. };
                actor.tones[0].adsr = Some(ToneAdsr {
                    attack_sec: 0.001,
                    decay_sec: 0.,
                    sustain_level: 1.,
                    release_sec: 0.05,
                });
                actor.routing.to_habitat = routing.0;
                actor.routing.to_presentation = routing.1;
                let mut other = outcome_batch(BodyKind::Sine);
                other.source_id = 99;
                let release = PhonationBatch {
                    intrinsic_period_sec: None,
                    source_id: actor.source_id,
                    source_generation: actor.source_generation,
                    // Deliberately different: release must use the original tone's routing.
                    routing: Routing {
                        to_habitat: !routing.0,
                        to_presentation: !routing.1,
                    },
                    cmds: vec![ToneCmd::Off {
                        tone_id: actor.tones[0].tone_id,
                        off_tick: 64,
                    }],
                    ..Default::default()
                };
                let all = [actor.clone(), other];
                let mut tracked = ScheduleRenderer::new(time);
                tracked.action_observer = Some(Box::new(Observer::new(8000)));
                let mut control = ScheduleRenderer::new(time);
                let mut isolated = ScheduleRenderer::new(time);
                let mut pcm = [Vec::new(), Vec::new()];
                let mut outcomes = Vec::new();
                let mut predicted_end = None;
                for now in (0..16064).step_by(32) {
                    let batches = if now == 0 {
                        &all[..]
                    } else if now == 32 {
                        std::slice::from_ref(&release)
                    } else {
                        &[]
                    };
                    let actual = tracked.render(batches, now, &NeuralRhythms::default());
                    let reference = control.render(batches, now, &NeuralRhythms::default());
                    assert_eq!(actual.habitat, reference.habitat);
                    assert_eq!(actual.presentation, reference.presentation);
                    if now == 0 {
                        let owned = tracked
                            .tones
                            .get(&ToneKey {
                                source_id: actor.source_id,
                                tone_id: actor.tones[0].tone_id,
                            })
                            .unwrap();
                        let (_, amplitude, envelope) = owned.tone.prediction_parameters(None);
                        let frozen = crate::life::self_prediction::ToneEnergy {
                            amplitude,
                            envelope,
                            control: None,
                            scheduled_release: None,
                            sine: None,
                            bank: None,
                        };
                        predicted_end = frozen.renderer_end_after(
                            32,
                            Some(crate::life::self_prediction::ScheduledRelease {
                                apply_at_sample: 32,
                                off_sample: 64,
                            }),
                        );
                    }
                    let private_batches = if now == 0 {
                        std::slice::from_ref(&actor)
                    } else if now == 32 {
                        std::slice::from_ref(&release)
                    } else {
                        &[]
                    };
                    let private = isolated.render(private_batches, now, &NeuralRhythms::default());
                    pcm[0].extend_from_slice(private.habitat);
                    pcm[1].extend_from_slice(private.presentation);
                    outcomes.extend(tracked.action_observer.as_mut().unwrap().drain());
                }
                let result = outcomes
                    .iter()
                    .find(|o| o.action == ActionKind::Release)
                    .unwrap();
                assert_eq!(result.command_status, "accepted");
                assert_eq!(result.issued_at_sample, 32);
                assert_eq!(result.scheduled_action_sample, Some(64));
                assert_eq!(result.renderer_end_sample, Some(464));
                assert_eq!(result.renderer_end_sample, predicted_end);
                assert_eq!(result.observed_samples, 16000);
                assert_eq!(result.available_at_sample, 16064);
                for (bus, pcm) in result.buses.iter().zip(pcm) {
                    let tail = &pcm[64..16064];
                    let first = tail
                        .iter()
                        .position(|v| v.abs() > ACTIVITY_THRESHOLD)
                        .map(|i| i as u64 + 64);
                    let last = tail
                        .iter()
                        .rposition(|v| v.abs() > ACTIVITY_THRESHOLD)
                        .map(|i| i as u64 + 64);
                    assert_eq!(bus.first_activity_sample, first);
                    assert_eq!(bus.last_activity_sample, last);
                    if bus.routed {
                        let rms = (tail.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>()
                            / 16000.)
                            .sqrt();
                        assert!((bus.rms.unwrap() - rms).abs() < 1e-12);
                        assert!(last.unwrap() > 64 && last.unwrap() < 464);
                    } else {
                        assert_eq!(bus.status, "not_routed");
                        assert_eq!(bus.rms, None);
                    }
                }
                assert_eq!(
                    tracked.action_observer.as_ref().unwrap().snapshot.pending,
                    0
                );
            }
        }
    }

    #[test]
    fn repeated_release_missing_tones_generations_and_shutdown_are_explicit() {
        use super::super::action_observation::Observer;
        let time = Timebase { fs: 8000., hop: 32 };
        let mut renderer = ScheduleRenderer::new(time);
        renderer.action_observer = Some(Box::new(Observer::new(8000)));
        let mut batch = outcome_batch(BodyKind::Sine);
        batch.tones[0].hold_ticks = Some(8000);
        batch.tones[0].adsr = Some(crate::life::sound::ToneAdsr {
            attack_sec: 0.001,
            decay_sec: 0.,
            sustain_level: 1.,
            release_sec: 0.5,
        });
        renderer.render(std::slice::from_ref(&batch), 0, &NeuralRhythms::default());
        let tone_id = batch.tones[0].tone_id;
        batch.tones.clear();
        batch.cmds = vec![ToneCmd::Off {
            tone_id,
            off_tick: 96,
        }];
        renderer.render(std::slice::from_ref(&batch), 32, &NeuralRhythms::default());
        batch.cmds = vec![ToneCmd::Off {
            tone_id,
            off_tick: 80,
        }];
        renderer.render(std::slice::from_ref(&batch), 64, &NeuralRhythms::default());
        batch.source_generation += 1;
        renderer.render(std::slice::from_ref(&batch), 96, &NeuralRhythms::default());
        batch.cmds = vec![ToneCmd::Off {
            tone_id: 999,
            off_tick: 128,
        }];
        renderer.render(std::slice::from_ref(&batch), 128, &NeuralRhythms::default());
        renderer.shutdown_at(160);
        renderer.render(&[], 160, &NeuralRhythms::default());
        let observer = renderer.action_observer.as_mut().unwrap();
        observer.finish();
        let outcomes: Vec<_> = observer.drain().collect();
        assert_eq!(
            outcomes
                .iter()
                .filter(|o| o.command_status == "superseded")
                .count(),
            2
        );
        assert!(
            outcomes
                .iter()
                .any(|o| o.command_status == "generation_mismatch" && o.observed_samples == 0)
        );
        assert!(
            outcomes
                .iter()
                .any(|o| o.command_status == "missing_tone" && o.observed_samples == 0)
        );
        let shutdown = outcomes
            .iter()
            .find(|o| o.action == ActionKind::Shutdown)
            .unwrap();
        assert_eq!(shutdown.buses[0].status, "incomplete");
        assert_eq!(shutdown.observed_samples, 32);
        assert_eq!(shutdown.renderer_end_sample, None);
        assert_eq!(observer.snapshot.pending, 0);
    }

    #[test]
    fn refused_commands_and_eof_never_claim_confirmed_audio() {
        use super::super::action_observation::Observer;
        for status in [
            "duplicate",
            "missing_spec",
            "cutoff",
            "invalid_tone",
            "accepted",
        ] {
            let time = Timebase { fs: 8000., hop: 32 };
            let mut renderer = ScheduleRenderer::new(time);
            renderer.action_observer = Some(Box::new(Observer::new(8000)));
            let mut batch = outcome_batch(BodyKind::Sine);
            match status {
                "duplicate" => batch.cmds.push(batch.cmds[0]),
                "missing_spec" => batch.tones.clear(),
                "cutoff" => renderer.shutdown_at(0),
                "invalid_tone" => batch.tones[0].amp = 0.,
                _ => {}
            }
            renderer.render(&[batch], 0, &NeuralRhythms::default());
            let observer = renderer.action_observer.as_mut().unwrap();
            observer.finish();
            let records: Vec<_> = observer.drain().collect();
            let outcome = records.iter().find(|o| o.command_status == status).unwrap();
            if status == "accepted" {
                assert_eq!(outcome.buses[0].status, "incomplete");
                assert_eq!(outcome.observed_samples, 19);
            } else {
                assert_eq!(outcome.observed_samples, 0);
                assert_eq!(outcome.buses[0].status, "unobserved");
                assert_eq!(outcome.buses[0].first_activity_sample, None);
            }
            assert_eq!(outcome.buses[0].rms, None);
        }
    }

    #[test]
    fn owned_fork_preserves_ringing_pending_controls_and_rng_without_other_sources() {
        let time = Timebase {
            fs: 8000.0,
            hop: 32,
        };
        for kind in [BodyKind::Sine, BodyKind::Harmonic, BodyKind::Modal] {
            for onset in [0, 64] {
                let actor = PhonationBatch {
                    body_policy: None,
                    body_opportunity: None,
                    intrinsic_period_sec: None,
                    source_id: 2,
                    source_generation: 0,
                    routing: Routing {
                        to_presentation: false,
                        to_habitat: true,
                    },
                    cmds: vec![
                        ToneCmd::On {
                            tone_id: 1,
                            kick: OnsetKick { strength: 1.0 },
                        },
                        ToneCmd::Update {
                            tone_id: 1,
                            at_tick: Some(80),
                            update: ToneUpdate {
                                target_freq_hz: Some(451.0),
                                target_amp: Some(0.14),
                                continuous_drive: Some(0.02),
                            },
                        },
                    ],
                    tones: vec![ToneSpec {
                        opportunity: None,
                        tone_id: 1,
                        onset,
                        hold_ticks: Some(256),
                        freq_hz: 337.0,
                        amp: 0.2,
                        smoothing_tau_sec: 0.01,
                        body: BodySnapshot {
                            kind,
                            amp_scale: 1.0,
                            brightness: 0.4,
                            inharmonic: 0.0,
                            spread: 0.0,
                            unison: 1,
                            motion: 0.1,
                            ratios: Some(std::sync::Arc::from([1.0, 1.7, 2.8])),
                        },
                        render_modulator: RenderModulatorSpec::SeqGate { duration_sec: 0.1 },
                        adsr: None,
                    }],
                    onsets: Vec::new(),
                };
                let mut other = actor.clone();
                other.source_id = 99;
                other.tones[0].freq_hz = 677.0;
                let mut live = ScheduleRenderer::new(time);
                let mut reference = ScheduleRenderer::new(time);
                let mut owned_reference = ScheduleRenderer::new(time);
                let mut rhythms = NeuralRhythms::default();
                for now in [0, 32] {
                    let all = if now == 0 {
                        vec![actor.clone(), other.clone()]
                    } else {
                        Vec::new()
                    };
                    assert_eq!(
                        live.render(&all, now, &rhythms).habitat,
                        reference.render(&all, now, &rhythms).habitat
                    );
                    owned_reference.render(
                        if now == 0 {
                            std::slice::from_ref(&actor)
                        } else {
                            &[]
                        },
                        now,
                        &rhythms,
                    );
                    for _ in 0..time.hop {
                        rhythms.advance_in_place(1.0 / time.fs);
                    }
                }
                let mut fork = live.fork_source(2);
                let mut absent = live.fork_source(200);
                let mut future_rhythms = rhythms;
                for now in [64, 96, 128] {
                    let result = fork.render(&[], now, &future_rhythms);
                    assert_eq!(
                        result.habitat,
                        owned_reference.render(&[], now, &future_rhythms).habitat
                    );
                    assert!(result.presentation.iter().all(|x| *x == 0.0));
                    assert!(
                        absent
                            .render(&[], now, &future_rhythms)
                            .habitat
                            .iter()
                            .all(|x| *x == 0.0)
                    );
                    for _ in 0..time.hop {
                        future_rhythms.advance_in_place(1.0 / time.fs);
                    }
                }
                // Advancing the fork must not consume live phases, noise, or queued updates.
                for now in [64, 96, 128] {
                    assert_eq!(
                        live.render(&[], now, &rhythms).habitat,
                        reference.render(&[], now, &rhythms).habitat
                    );
                    for _ in 0..time.hop {
                        rhythms.advance_in_place(1.0 / time.fs);
                    }
                }
            }
        }
    }

    #[test]
    fn update_command_applies_to_tone() {
        let tb = Timebase { fs: 1000.0, hop: 4 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let tone_id = 1;
        let batch = PhonationBatch {
            body_policy: None,
            body_opportunity: None,
            intrinsic_period_sec: None,
            source_id: 2,
            source_generation: 0,
            routing: crate::scenario::control::Routing::default(),
            cmds: vec![
                ToneCmd::Update {
                    tone_id,
                    at_tick: Some(0),
                    update: ToneUpdate {
                        target_freq_hz: Some(440.0),
                        target_amp: Some(0.25),
                        continuous_drive: None,
                    },
                },
                ToneCmd::On {
                    tone_id,
                    kick: OnsetKick { strength: 1.0 },
                },
            ],
            tones: vec![ToneSpec {
                opportunity: None,
                tone_id,
                onset: 0,
                hold_ticks: Some(8),
                freq_hz: 220.0,
                amp: 0.5,
                smoothing_tau_sec: 0.0,
                body: BodySnapshot {
                    kind: BodyKind::Sine,
                    amp_scale: 1.0,
                    brightness: 0.0,
                    inharmonic: 0.0,
                    spread: 0.0,
                    unison: 1,
                    motion: 0.0,
                    ratios: None,
                },
                render_modulator: RenderModulatorSpec::SeqGate { duration_sec: 0.1 },
                adsr: None,
            }],
            onsets: Vec::new(),
        };
        renderer.render(&[batch], 0, &rhythms);

        let key = ToneKey {
            source_id: 2,
            tone_id,
        };
        let rt = renderer.tones.get(&key).expect("tone");
        assert!((rt.tone.debug_current_freq_hz() - 440.0).abs() < 1e-6);
        assert!((rt.tone.debug_current_amp() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn self_sound_tracking_preserves_mix_and_retires_without_inheriting_old_tails() {
        let tb = Timebase {
            fs: 48_000.0,
            hop: 512,
        };
        let rhythms = NeuralRhythms::default();
        for to_habitat in [false, true] {
            let mut tracked = ScheduleRenderer::new(tb);
            let mut observer = AcousticTemporalExpectation::new(48_000).unwrap();
            let mut control = ScheduleRenderer::new(tb);
            let batch = PhonationBatch {
                body_policy: None,
                body_opportunity: None,
                intrinsic_period_sec: None,
                source_id: 2,
                source_generation: 0,
                routing: crate::scenario::control::Routing {
                    to_presentation: true,
                    to_habitat,
                },
                cmds: vec![ToneCmd::On {
                    tone_id: 1,
                    kick: OnsetKick { strength: 1.0 },
                }],
                tones: vec![ToneSpec {
                    opportunity: None,
                    tone_id: 1,
                    onset: 0,
                    hold_ticks: Some(48_000),
                    freq_hz: 440.0,
                    amp: 0.5,
                    smoothing_tau_sec: 0.0,
                    body: BodySnapshot {
                        kind: BodyKind::Sine,
                        amp_scale: 1.0,
                        brightness: 0.0,
                        inharmonic: 0.0,
                        spread: 0.0,
                        unison: 1,
                        motion: 0.0,
                        ratios: None,
                    },
                    render_modulator: RenderModulatorSpec::SeqGate { duration_sec: 1.0 },
                    adsr: None,
                }],
                onsets: Vec::new(),
            };
            for hop in 0..20 {
                let now = hop * 512;
                // Keep the original sound alive after its participant retires.
                let id = if hop < 5 { 2 } else { hop + 10 };
                tracked.prepare_self_sound(std::iter::once(id), now, &observer);
                let batches = if hop == 0 {
                    std::slice::from_ref(&batch)
                } else {
                    &[]
                };
                let actual = tracked.render(batches, now, &rhythms);
                let expected = control.render(batches, now, &rhythms);
                assert_eq!(actual.presentation, expected.presentation);
                assert_eq!(actual.habitat, expected.habitat);
                observer.process(now, expected.habitat, |_| {});
                let sound = tracked
                    .self_sound
                    .iter()
                    .flatten()
                    .find(|s| s.source_id == id)
                    .unwrap();
                if hop < 5 {
                    assert_eq!(sound.body, expected.presentation);
                    assert_eq!(sound.habitat, expected.habitat);
                    assert!(sound.history.profile.iter().sum::<f32>() > 0.0);
                } else {
                    assert!(expected.presentation.iter().any(|s| *s != 0.0));
                    assert!(sound.body.iter().all(|s| *s == 0.0));
                    assert_eq!(sound.history.profile, [0.0; 3]);
                }
                assert!(tracked.self_sound.len() <= 2);
            }
        }
    }

    #[test]
    fn update_commands_same_tick_last_wins() {
        let tb = Timebase { fs: 1000.0, hop: 4 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let tone_id = 1;
        let batch = PhonationBatch {
            body_policy: None,
            body_opportunity: None,
            intrinsic_period_sec: None,
            source_id: 2,
            source_generation: 0,
            routing: crate::scenario::control::Routing::default(),
            cmds: vec![
                ToneCmd::Update {
                    tone_id,
                    at_tick: Some(0),
                    update: ToneUpdate {
                        target_freq_hz: Some(330.0),
                        target_amp: None,
                        continuous_drive: None,
                    },
                },
                ToneCmd::Update {
                    tone_id,
                    at_tick: Some(0),
                    update: ToneUpdate {
                        target_freq_hz: Some(440.0),
                        target_amp: None,
                        continuous_drive: None,
                    },
                },
                ToneCmd::On {
                    tone_id,
                    kick: OnsetKick { strength: 1.0 },
                },
            ],
            tones: vec![ToneSpec {
                opportunity: None,
                tone_id,
                onset: 0,
                hold_ticks: Some(8),
                freq_hz: 220.0,
                amp: 0.5,
                smoothing_tau_sec: 0.0,
                body: BodySnapshot {
                    kind: BodyKind::Sine,
                    amp_scale: 1.0,
                    brightness: 0.0,
                    inharmonic: 0.0,
                    spread: 0.0,
                    unison: 1,
                    motion: 0.0,
                    ratios: None,
                },
                render_modulator: RenderModulatorSpec::SeqGate { duration_sec: 0.1 },
                adsr: None,
            }],
            onsets: Vec::new(),
        };
        renderer.render(&[batch], 0, &rhythms);

        let key = ToneKey {
            source_id: 2,
            tone_id,
        };
        let rt = renderer.tones.get(&key).expect("tone");
        assert!((rt.tone.debug_target_freq_hz() - 440.0).abs() < 1e-6);
    }

    #[test]
    fn shutdown_releases_tones_and_becomes_idle() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let tone_id = 1;
        let batch = PhonationBatch {
            body_policy: None,
            body_opportunity: None,
            intrinsic_period_sec: None,
            source_id: 1,
            source_generation: 0,
            routing: crate::scenario::control::Routing::default(),
            cmds: vec![ToneCmd::On {
                tone_id,
                kick: OnsetKick { strength: 1.0 },
            }],
            tones: vec![ToneSpec {
                opportunity: None,
                tone_id,
                onset: 0,
                hold_ticks: Some(Tick::MAX),
                freq_hz: 220.0,
                amp: 0.4,
                smoothing_tau_sec: 0.0,
                body: BodySnapshot {
                    kind: BodyKind::Sine,
                    amp_scale: 1.0,
                    brightness: 0.0,
                    inharmonic: 0.0,
                    spread: 0.0,
                    unison: 1,
                    motion: 0.0,
                    ratios: None,
                },
                render_modulator: RenderModulatorSpec::SeqGate { duration_sec: 0.1 },
                adsr: None,
            }],
            onsets: Vec::new(),
        };
        renderer.render(&[batch], 0, &rhythms);
        assert!(!renderer.is_idle());
        renderer.shutdown_at(0);
        let done_at = default_release_ticks(tb) + 2;
        renderer.render(&[], done_at, &rhythms);
        assert!(renderer.is_idle());
    }
}
