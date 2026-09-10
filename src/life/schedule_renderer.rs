use crate::core::modulation::NeuralRhythms;
use crate::core::temporal_expectation::{AcousticTemporalExpectation, OwnSoundHistory};
use crate::core::timebase::{Tick, Timebase};
use crate::life::phonation_engine::ToneCmd;
use crate::life::sound::Tone;
use crate::life::voice::PhonationBatch;
use crate::scenario::control::Routing;
use std::collections::BTreeMap;
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

pub struct ScheduleRenderer {
    time: Timebase,
    buf_presentation: Vec<f32>,
    buf_habitat: Vec<f32>,
    tones: BTreeMap<ToneKey, RoutedTone>,
    cutoff_tick: Option<Tick>,
    self_sound: Vec<Option<Box<SelfSound>>>,
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
        mut emit: impl FnMut(u64, u64, usize, &crate::core::history_prediction::PredictionMatch<'_>),
    ) -> RenderFrame<'_> {
        let hop = self.time.hop;
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

        self.tones.retain(|_, rt| !rt.tone.is_done(now));

        let end = now.saturating_add(hop as Tick);
        let dt = 1.0 / fs;
        let mut rhythms = *rhythms;
        self.apply_phonation_batches(phonation_batches, now, &rhythms, dt);
        for tick in now..end {
            let idx = (tick - now) as usize;
            let mut acc_presentation = 0.0f32;
            let mut acc_habitat = 0.0f32;
            for rt in self.tones.values_mut() {
                rt.tone.apply_updates_if_due(tick);
                rt.tone.kick_planned_if_due(tick);
                let sample = rt.tone.render_tick(tick, fs, dt, &rhythms);
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

        for sound in self.self_sound.iter_mut().flatten() {
            sound.history.process(
                now,
                &sound.body,
                &sound.habitat,
                &self.buf_habitat,
                |start, window, matched| emit(sound.source_id, start, window, matched),
            );
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
        self.tones.retain(|_, rt| rt.tone.onset() <= tick);
        for rt in self.tones.values_mut() {
            rt.tone.note_off(tick);
        }
    }

    fn apply_phonation_batches(
        &mut self,
        phonation_batches: &[PhonationBatch],
        now: Tick,
        _rhythms: &NeuralRhythms,
        _dt: f32,
    ) {
        let default_hold_ticks = max_phonation_hold_ticks(self.time);
        for batch in phonation_batches {
            for cmd in &batch.cmds {
                match *cmd {
                    ToneCmd::On { tone_id, kick } => {
                        let key = ToneKey {
                            source_id: batch.source_id,
                            tone_id,
                        };
                        if self.tones.contains_key(&key) {
                            continue;
                        }
                        let spec = batch.tones.iter().find(|t| t.tone_id == tone_id);
                        let Some(spec) = spec else { continue };
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
                                    routing: batch.routing,
                                    self_slot: self.self_sound.iter().position(|sound| {
                                        sound
                                            .as_ref()
                                            .is_some_and(|sound| sound.source_id == batch.source_id)
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
                        if let Some(rt) = self.tones.get_mut(&key) {
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

fn modal_phase_seed(a: u64, b: u64, c: u64) -> u64 {
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

    #[test]
    fn owned_fork_preserves_ringing_pending_controls_and_rng_without_other_sources() {
        let time = Timebase {
            fs: 8000.0,
            hop: 32,
        };
        for kind in [BodyKind::Sine, BodyKind::Harmonic, BodyKind::Modal] {
            for onset in [0, 64] {
                let actor = PhonationBatch {
                    source_id: 2,
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
            source_id: 2,
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
                source_id: 2,
                routing: crate::scenario::control::Routing {
                    to_presentation: true,
                    to_habitat,
                },
                cmds: vec![ToneCmd::On {
                    tone_id: 1,
                    kick: OnsetKick { strength: 1.0 },
                }],
                tones: vec![ToneSpec {
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
            source_id: 2,
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
            source_id: 1,
            routing: crate::scenario::control::Routing::default(),
            cmds: vec![ToneCmd::On {
                tone_id,
                kick: OnsetKick { strength: 1.0 },
            }],
            tones: vec![ToneSpec {
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
