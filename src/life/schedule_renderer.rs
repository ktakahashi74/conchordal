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
    source_generation: u32,
    body_slot: Option<(usize, u32)>,
    scheduled_release: Option<super::tone_energy::ScheduledRelease>,
    source_pcm_slot: Option<usize>,
}

/// Caller-owned habitat scratch, separated by Voice identity and generation.
pub(crate) struct SourcePcm {
    pub id: u64,
    pub generation: u32,
    pub habitat: Vec<f32>,
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
                        source_generation: routed.source_generation,
                        body_slot: None,
                        scheduled_release: routed.scheduled_release,
                        source_pcm_slot: None,
                    },
                )
            })
            .collect();
        fork
    }

    #[cfg(test)]
    pub(crate) fn fork_all_without_capture(&self) -> Self {
        let mut fork = Self::new(self.time);
        fork.cutoff_tick = self.cutoff_tick;
        fork.tones = self
            .tones
            .iter()
            .map(|(key, routed)| {
                (
                    *key,
                    RoutedTone {
                        tone: routed.tone.clone(),
                        routing: routed.routing,
                        self_slot: None,
                        source_generation: routed.source_generation,
                        body_slot: None,
                        scheduled_release: routed.scheduled_release,
                        source_pcm_slot: None,
                    },
                )
            })
            .collect();
        fork
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
        emit: impl FnMut(u64, u64, usize, &crate::core::history_prediction::PredictionMatch<'_>),
    ) -> RenderFrame<'_> {
        self.render_with_source_pcm(phonation_batches, now, rhythms, &mut [], emit)
    }

    pub(crate) fn render_with_source_pcm(
        &mut self,
        phonation_batches: &[PhonationBatch],
        now: Tick,
        rhythms: &NeuralRhythms,
        sources: &mut [SourcePcm],
        mut emit: impl FnMut(u64, u64, usize, &crate::core::history_prediction::PredictionMatch<'_>),
    ) -> RenderFrame<'_> {
        for source in sources.iter_mut() {
            assert_eq!(source.habitat.len(), self.time.hop, "source PCM hop length");
            source.habitat.fill(0.);
        }
        for (index, source) in sources.iter().enumerate() {
            assert!(
                !sources[..index]
                    .iter()
                    .any(|previous| (previous.id, previous.generation)
                        == (source.id, source.generation)),
                "duplicate source PCM identity"
            );
        }
        let mut checkpoint = self.profile.as_ref().map(|_| Instant::now());
        self.profile = self.profile.map(|_| RenderProfile::default());
        let hop = self.time.hop;
        if let Some(capture) = self.body_capture.as_mut() {
            capture.begin(now);
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

        self.tones.retain(|_, rt| {
            if !rt.tone.is_done(now) {
                return true;
            }

            false
        });

        let end = now.saturating_add(hop as Tick);
        let dt = 1.0 / fs;
        let mut rhythms = *rhythms;
        let setup_us = profile_lap(&mut checkpoint);
        self.apply_phonation_batches(phonation_batches, now);
        for (key, tone) in &mut self.tones {
            tone.source_pcm_slot = sources.iter().position(|source| {
                source.id == key.source_id && source.generation == tone.source_generation
            });
        }
        let commands_us = profile_lap(&mut checkpoint);
        for tick in now..end {
            let idx = (tick - now) as usize;
            let mut acc_presentation = 0.0f32;
            let mut acc_habitat = 0.0f32;
            for rt in self.tones.values_mut() {
                rt.tone.apply_updates_if_due(tick);
                rt.tone.kick_planned_if_due(tick);
                let sample = rt.tone.render_tick(tick, fs, dt, &rhythms);

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
                    if let Some(slot) = rt.source_pcm_slot {
                        sources[slot].habitat[idx] += sample;
                    }
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
        self.tones.retain(|_, rt| {
            if rt.tone.onset() <= tick {
                return true;
            }

            false
        });
        for rt in self.tones.values_mut() {
            rt.tone.note_off(tick);
        }
    }

    fn apply_phonation_batches(&mut self, phonation_batches: &[PhonationBatch], now: Tick) {
        let default_hold_ticks = max_phonation_hold_ticks(self.time);
        for batch in phonation_batches {
            for cmd in &batch.cmds {
                match *cmd {
                    ToneCmd::On { tone_id, kick } => {
                        let key = ToneKey {
                            source_id: batch.source_id,
                            tone_id,
                        };
                        let spec = batch.tones.iter().find(|t| t.tone_id == tone_id);
                        if self.tones.contains_key(&key) {
                            continue;
                        }
                        let Some(spec) = spec else {
                            continue;
                        };
                        if let Some(cutoff) = self.cutoff_tick
                            && spec.onset >= cutoff
                        {
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
                                    source_pcm_slot: None,
                                    routing: batch.routing,
                                    self_slot: self.self_sound.iter().position(|sound| {
                                        sound
                                            .as_ref()
                                            .is_some_and(|sound| sound.source_id == batch.source_id)
                                    }),
                                    source_generation: batch.source_generation,
                                    body_slot: self.body_capture.as_ref().and_then(|c| {
                                        c.token(batch.source_id, batch.source_generation)
                                    }),
                                },
                            );
                        }
                    }
                    ToneCmd::Off { tone_id, off_tick } => {
                        let key = ToneKey {
                            source_id: batch.source_id,
                            tone_id,
                        };
                        if let Some(rt) = self.tones.get_mut(&key)
                            && rt.source_generation == batch.source_generation
                        {
                            rt.tone.note_off(off_tick);
                        }
                    }
                    ToneCmd::Update { .. } => {}
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
    fn source_pcm_preserves_mix_and_separates_routing_and_generation() {
        let time = Timebase { fs: 8000., hop: 64 };
        let rhythms = NeuralRhythms::default();
        for kind in [BodyKind::Sine, BodyKind::Harmonic, BodyKind::Modal] {
            let own = outcome_batch(kind);
            let mut other = own.clone();
            other.source_id = 3;
            other.tones[0].freq_hz = 466.;
            let mut decor = own.clone();
            decor.tones[0].tone_id = 2;
            decor.cmds[0] = ToneCmd::On {
                tone_id: 2,
                kick: OnsetKick { strength: 1. },
            };
            decor.routing.to_habitat = false;
            let batches = [own.clone(), other, decor];
            let mut captured = ScheduleRenderer::new(time);
            let mut ordinary = ScheduleRenderer::new(time);
            let mut own_reference = ScheduleRenderer::new(time);
            let mut scratch = [
                SourcePcm {
                    id: 2,
                    generation: 7,
                    habitat: vec![0.; time.hop],
                },
                SourcePcm {
                    id: 2,
                    generation: 8,
                    habitat: vec![0.; time.hop],
                },
                SourcePcm {
                    id: 999,
                    generation: 0,
                    habitat: vec![0.; time.hop],
                },
            ];
            for frame in 0..20 {
                let now = frame * time.hop as Tick;
                let commands = if frame == 0 { &batches[..] } else { &[] };
                let actual = captured.render_with_source_pcm(
                    commands,
                    now,
                    &rhythms,
                    &mut scratch,
                    |_, _, _, _| {},
                );
                let expected = ordinary.render(commands, now, &rhythms);
                assert_eq!(actual.habitat, expected.habitat);
                assert_eq!(actual.presentation, expected.presentation);
                let expected_own = own_reference.render(
                    if frame == 0 {
                        std::slice::from_ref(&own)
                    } else {
                        &[]
                    },
                    now,
                    &rhythms,
                );
                assert_eq!(scratch[0].habitat, expected_own.habitat);
                assert!(
                    scratch[1]
                        .habitat
                        .iter()
                        .chain(&scratch[2].habitat)
                        .all(|value| *value == 0.)
                );
            }
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
