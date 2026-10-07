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
    scheduled_release: Option<super::tone_energy::ScheduledRelease>,
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
    ) -> impl Iterator<Item = (u64, [bool; 2], super::tone_energy::ToneEnergy)> + '_ {
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
                    super::tone_energy::ToneEnergy {
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
        let contexts = self.self_sound.iter().filter_map(|sound| {
            let sound = sound.as_ref()?;
            let batch = batches.iter().find(|batch| {
                batch.source_id == sound.source_id
                    && batch
                        .cmds
                        .iter()
                        .any(|cmd| matches!(cmd, ToneCmd::On { .. } | ToneCmd::Off { .. }))
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
            for cmd in &batch.cmds {
                let mut prediction = None;
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
                                scheduled_release = Some(super::tone_energy::ScheduledRelease {
                                    apply_at_sample: now
                                        + (off - now) / self.time.hop as u64 * self.time.hop as u64,
                                    off_sample: off,
                                });
                            }
                            let outcome_slot = self.action_observer.as_mut().and_then(|observer| {
                                observer.command(
                                    batch,
                                    tone_id,
                                    Some(spec.onset),
                                    now,
                                    "accepted",
                                    ActionKind::Onset,
                                )
                            });

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
                if rt.source_generation != batch.source_generation {
                    continue;
                }
                let tick = at_tick.unwrap_or(now);
                rt.tone.schedule_update(tick, update);
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

    /// The control keeps observation and learning but strips the policy facts and

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
                        let frozen = crate::life::tone_energy::ToneEnergy {
                            amplitude,
                            envelope,
                            control: None,
                            scheduled_release: None,
                            sine: None,
                            bank: None,
                        };
                        predicted_end = frozen.renderer_end_after(
                            32,
                            Some(crate::life::tone_energy::ScheduledRelease {
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
    fn update_commands_ignore_other_source_generations() {
        let time = Timebase { fs: 8000., hop: 32 };
        let rhythms = NeuralRhythms::default();
        for at_tick in [None, Some(96)] {
            for generation in [6, 7, 8] {
                for update in [
                    ToneUpdate {
                        target_freq_hz: Some(220.),
                        target_amp: None,
                        continuous_drive: None,
                    },
                    ToneUpdate {
                        target_freq_hz: None,
                        target_amp: Some(0.05),
                        continuous_drive: None,
                    },
                    ToneUpdate {
                        target_freq_hz: None,
                        target_amp: None,
                        continuous_drive: Some(2.0),
                    },
                ] {
                    let mut renderer = ScheduleRenderer::new(time);
                    let mut reference = ScheduleRenderer::new(time);
                    let mut batch = outcome_batch(BodyKind::Harmonic);
                    batch.tones[0].hold_ticks = Some(8000);
                    let tone_id = batch.tones[0].tone_id;
                    let key = ToneKey {
                        source_id: batch.source_id,
                        tone_id,
                    };
                    let initial = renderer.render(std::slice::from_ref(&batch), 0, &rhythms);
                    let baseline = reference.render(std::slice::from_ref(&batch), 0, &rhythms);
                    assert!(initial.habitat.iter().any(|sample| *sample != 0.));
                    assert_eq!(initial.habitat, baseline.habitat);
                    assert_eq!(initial.presentation, baseline.presentation);
                    batch.tones.clear();
                    batch.source_generation = generation;
                    batch.cmds = vec![ToneCmd::Update {
                        tone_id,
                        at_tick,
                        update,
                    }];
                    for now in [32, 64, 96, 128] {
                        let batches = if now == 32 {
                            std::slice::from_ref(&batch)
                        } else {
                            &[]
                        };
                        let actual = renderer.render(batches, now, &rhythms);
                        let unchanged = reference.render(&[], now, &rhythms);
                        let applied = generation == 7 && now >= at_tick.unwrap_or(32);
                        if applied {
                            assert_ne!(actual.habitat, unchanged.habitat);
                            assert_ne!(actual.presentation, unchanged.presentation);
                        } else {
                            assert_eq!(actual.habitat, unchanged.habitat);
                            assert_eq!(actual.presentation, unchanged.presentation);
                        }
                        let tone = &renderer.tones[&key].tone;
                        assert_eq!(
                            tone.debug_current_freq_hz(),
                            if applied {
                                update.target_freq_hz.unwrap_or(337.)
                            } else {
                                337.
                            }
                        );
                        assert_eq!(
                            tone.debug_current_amp(),
                            if applied {
                                update.target_amp.unwrap_or(0.2)
                            } else {
                                0.2
                            }
                        );
                    }
                }
            }
        }
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
