use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::individual::PhonationBatch;
use crate::life::intent::{Intent, IntentBoard};
use crate::life::phonation_engine::PhonationCmd;
use crate::life::sound::{AudioCommand, Voice, VoiceTarget, default_release_ticks};
use std::collections::{HashMap, HashSet};
use tracing::debug;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum VoiceKind {
    Intent,
    Phonation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct VoiceKey {
    source_id: u64,
    note_id: u64,
    kind: VoiceKind,
}

pub struct ScheduleRenderer {
    time: Timebase,
    buf: Vec<f32>,
    voices: HashMap<VoiceKey, Voice>,
    agent_voices: HashMap<u64, Voice>,
    add_future_ticks: Tick,
    add_past_ticks: Tick,
    did_full_resync: bool,
    cutoff_tick: Option<Tick>,
    agent_ids_scratch: HashSet<u64>,
}

pub type SoundRenderer = ScheduleRenderer;

impl ScheduleRenderer {
    pub fn new(time: Timebase) -> Self {
        let add_future_ticks = (time.hop as Tick).saturating_mul(4).max(1);
        let add_past_ticks = schedule_add_past_ticks(time);
        Self {
            time,
            buf: vec![0.0; time.hop],
            voices: HashMap::new(),
            agent_voices: HashMap::new(),
            add_future_ticks,
            add_past_ticks,
            did_full_resync: false,
            cutoff_tick: None,
            agent_ids_scratch: HashSet::new(),
        }
    }

    pub fn render(
        &mut self,
        board: &IntentBoard,
        phonation_batches: &[PhonationBatch],
        now: Tick,
        rhythms: &NeuralRhythms,
        voice_targets: &[VoiceTarget],
        audio_cmds: &[AudioCommand],
    ) -> &[f32] {
        let hop = self.time.hop;
        if self.buf.len() != hop {
            self.buf.resize(hop, 0.0);
        }
        self.buf.fill(0.0);

        let fs = self.time.fs;
        if fs <= 0.0 {
            return &self.buf;
        }

        self.agent_ids_scratch.clear();
        self.agent_ids_scratch.reserve(voice_targets.len());
        for state in voice_targets {
            self.agent_ids_scratch.insert(state.id);
        }
        for (id, voice) in self.agent_voices.iter_mut() {
            if !self.agent_ids_scratch.contains(id) {
                voice.note_off(now);
            }
        }
        self.agent_voices
            .retain(|id, voice| self.agent_ids_scratch.contains(id) || !voice.is_done(now));
        self.voices.retain(|_, voice| !voice.is_done(now));

        let past = if self.did_full_resync {
            board.retention_past.min(self.add_past_ticks)
        } else {
            self.did_full_resync = true;
            board.retention_past
        };
        let future = board.horizon_future.min(self.add_future_ticks);
        // Order is intentional: enqueue new voices first, then kick+render for this tick.
        // This assumes add_voice_if_needed accepts onset<=now so same-tick onsets are available.
        for intent in board.snapshot(now, past, future) {
            if let Some(cutoff) = self.cutoff_tick
                && intent.onset >= cutoff
            {
                continue;
            }
            self.add_voice_if_needed(intent, now);
        }
        for cmd in audio_cmds {
            if let AudioCommand::EnsureVoice {
                id,
                body,
                pitch_hz,
                amp,
            } = cmd
            {
                if !self.agent_ids_scratch.contains(id) {
                    continue;
                }
                if self.agent_voices.contains_key(id) {
                    continue;
                }
                let intent = Intent {
                    source_id: *id,
                    intent_id: 0,
                    onset: now,
                    duration: Tick::MAX,
                    freq_hz: *pitch_hz,
                    amp: *amp,
                    tag: None,
                    confidence: 1.0,
                    body: Some(body.clone()),
                    articulation: None,
                };
                if let Some(voice) = Voice::from_intent(self.time, intent) {
                    self.agent_voices.insert(*id, voice);
                }
            }
        }
        for state in voice_targets {
            if let Some(voice) = self.agent_voices.get_mut(&state.id) {
                voice.set_target(state.pitch_hz, state.amp, 0.0);
            }
        }
        for cmd in audio_cmds {
            if let AudioCommand::Impulse { id, energy } = cmd {
                if !self.agent_ids_scratch.contains(id) {
                    continue;
                }
                if let Some(voice) = self.agent_voices.get_mut(id) {
                    voice.trigger_impulse(*energy);
                }
            }
        }

        let end = now.saturating_add(hop as Tick);
        let dt = 1.0 / fs;
        let mut rhythms = *rhythms;
        self.apply_phonation_batches(phonation_batches, now, &rhythms, dt);
        // For each tick: apply due updates, then render the sample.
        for tick in now..end {
            let idx = (tick - now) as usize;
            let mut acc = 0.0f32;
            for (_key, voice) in self.voices.iter_mut() {
                voice.apply_updates_if_due(tick);
                voice.kick_planned_if_due(tick, &rhythms, dt);
                acc += voice.render_tick(tick, fs, dt, &rhythms);
            }
            for voice in self.agent_voices.values_mut() {
                voice.apply_updates_if_due(tick);
                voice.kick_planned_if_due(tick, &rhythms, dt);
                acc += voice.render_tick(tick, fs, dt, &rhythms);
            }
            self.buf[idx] = acc;
            rhythms.advance_in_place(dt);
        }

        self.apply_limiter();
        &self.buf
    }

    pub fn is_idle(&self) -> bool {
        self.voices.is_empty() && self.agent_voices.is_empty()
    }

    pub fn set_cutoff_tick(&mut self, cutoff: Option<Tick>) {
        self.cutoff_tick = cutoff;
    }

    pub fn shutdown_at(&mut self, tick: Tick) {
        self.cutoff_tick = Some(tick);
        self.voices.retain(|_, voice| voice.onset() <= tick);
        for voice in self.voices.values_mut() {
            voice.note_off(tick);
        }
        self.agent_voices.retain(|_, voice| voice.onset() <= tick);
        for voice in self.agent_voices.values_mut() {
            voice.note_off(tick);
        }
    }

    fn add_voice_if_needed(&mut self, intent: Intent, now: Tick) {
        if intent.duration == 0 || intent.freq_hz <= 0.0 {
            return;
        }
        if intent.amp == 0.0 {
            return;
        }
        let end_tick = intent
            .onset
            .saturating_add(intent.duration)
            .saturating_add(default_release_ticks(self.time));
        if end_tick <= now {
            return;
        }
        let key = VoiceKey {
            source_id: intent.source_id,
            note_id: intent.intent_id,
            kind: VoiceKind::Intent,
        };
        if self.voices.contains_key(&key) {
            return;
        }
        let onset = intent.onset;
        if let Some(mut voice) = Voice::from_intent(self.time, intent) {
            voice.note_on(onset);
            voice.arm_onset_trigger(1.0);
            self.voices.insert(key, voice);
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
                    PhonationCmd::NoteOn { note_id, kick } => {
                        let key = VoiceKey {
                            source_id: batch.source_id,
                            note_id,
                            kind: VoiceKind::Phonation,
                        };
                        if self.voices.contains_key(&key) {
                            continue;
                        }
                        let spec = batch.notes.iter().find(|note| note.note_id == note_id);
                        let Some(spec) = spec else { continue };
                        if let Some(cutoff) = self.cutoff_tick
                            && spec.onset >= cutoff
                        {
                            continue;
                        }
                        let hold_ticks = spec.hold_ticks.unwrap_or(default_hold_ticks);
                        let intent = Intent {
                            source_id: batch.source_id,
                            intent_id: note_id,
                            onset: spec.onset,
                            duration: hold_ticks,
                            freq_hz: spec.freq_hz,
                            amp: spec.amp,
                            tag: None,
                            confidence: 1.0,
                            body: Some(spec.body.clone()),
                            articulation: Some(spec.articulation.clone()),
                        };
                        if let Some(mut voice) = Voice::from_intent(self.time, intent) {
                            voice.set_smoothing_tau_sec(spec.smoothing_tau_sec);
                            voice.note_on(spec.onset);
                            voice.schedule_planned_kick(kick);
                            voice.arm_onset_trigger(kick.strength().max(0.0));
                            debug!(
                                target: "phonation::note_on",
                                source_id = batch.source_id,
                                note_id,
                                onset = spec.onset,
                                freq_hz = spec.freq_hz,
                                amp = spec.amp
                            );
                            self.voices.insert(key, voice);
                        }
                    }
                    PhonationCmd::NoteOff { note_id, off_tick } => {
                        let key = VoiceKey {
                            source_id: batch.source_id,
                            note_id,
                            kind: VoiceKind::Phonation,
                        };
                        if let Some(voice) = self.voices.get_mut(&key) {
                            voice.note_off(off_tick);
                        }
                    }
                    PhonationCmd::Update { .. } => {}
                }
            }
            for cmd in &batch.cmds {
                let PhonationCmd::Update {
                    note_id,
                    at_tick,
                    update,
                } = *cmd
                else {
                    continue;
                };
                let key = VoiceKey {
                    source_id: batch.source_id,
                    note_id,
                    kind: VoiceKind::Phonation,
                };
                let Some(voice) = self.voices.get_mut(&key) else {
                    continue;
                };
                let tick = at_tick.unwrap_or(now);
                voice.schedule_update(tick, update);
            }
        }
    }

    fn apply_limiter(&mut self) {
        let mut peak = 0.0f32;
        for &s in &self.buf {
            if s.is_finite() {
                peak = peak.max(s.abs());
            }
        }
        let target = 0.98f32;
        if peak > target && peak > 0.0 {
            let g = target / peak;
            if g.is_finite() {
                for s in &mut self.buf {
                    *s *= g;
                }
            }
        }
    }
}

fn schedule_add_past_ticks(time: Timebase) -> Tick {
    let hop_window = (time.hop as Tick).saturating_mul(2);
    default_release_ticks(time).max(hop_window).max(1)
}

fn max_phonation_hold_ticks(time: Timebase) -> Tick {
    let max_sec = 60.0;
    let ticks = time.sec_to_tick(max_sec);
    ticks.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::individual::{
        AnyArticulationCore, ArticulationWrapper, PhonationBatch, PhonationNoteSpec, SequencedCore,
    };
    use crate::life::intent::BodySnapshot;
    use crate::life::phonation_engine::{PhonationKick, PhonationUpdate};
    use crate::life::sound::{AudioCommand, VoiceTarget, default_release_ticks};

    #[test]
    fn update_command_applies_to_voice() {
        let tb = Timebase { fs: 1000.0, hop: 4 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let board = IntentBoard::new(1, 1);
        let articulation = ArticulationWrapper::new(
            AnyArticulationCore::Seq(SequencedCore {
                timer: 0.0,
                duration: 0.1,
                env_level: 0.0,
            }),
            0.0,
        );
        let note_id = 1;
        let batch = PhonationBatch {
            source_id: 2,
            cmds: vec![
                PhonationCmd::Update {
                    note_id,
                    at_tick: Some(0),
                    update: PhonationUpdate {
                        target_freq_hz: Some(440.0),
                        target_amp: Some(0.25),
                    },
                },
                PhonationCmd::NoteOn {
                    note_id,
                    kick: PhonationKick::Planned { strength: 1.0 },
                },
            ],
            notes: vec![PhonationNoteSpec {
                note_id,
                onset: 0,
                hold_ticks: Some(8),
                freq_hz: 220.0,
                amp: 0.5,
                smoothing_tau_sec: 0.0,
                body: BodySnapshot {
                    kind: "sine".to_string(),
                    amp_scale: 1.0,
                    brightness: 0.0,
                    noise_mix: 0.0,
                },
                articulation,
            }],
            onsets: Vec::new(),
        };
        renderer.render(&board, &[batch], 0, &rhythms, &[], &[]);

        let key = VoiceKey {
            source_id: 2,
            note_id,
            kind: VoiceKind::Phonation,
        };
        let voice = renderer.voices.get(&key).expect("voice");
        assert!((voice.debug_current_freq_hz() - 440.0).abs() < 1e-6);
        assert!((voice.debug_current_amp() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn update_commands_same_tick_last_wins() {
        let tb = Timebase { fs: 1000.0, hop: 4 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let board = IntentBoard::new(1, 1);
        let articulation = ArticulationWrapper::new(
            AnyArticulationCore::Seq(SequencedCore {
                timer: 0.0,
                duration: 0.1,
                env_level: 0.0,
            }),
            0.0,
        );
        let note_id = 1;
        let batch = PhonationBatch {
            source_id: 2,
            cmds: vec![
                PhonationCmd::Update {
                    note_id,
                    at_tick: Some(0),
                    update: PhonationUpdate {
                        target_freq_hz: Some(330.0),
                        target_amp: None,
                    },
                },
                // Same-tick updates are applied in command order; later update wins.
                PhonationCmd::Update {
                    note_id,
                    at_tick: Some(0),
                    update: PhonationUpdate {
                        target_freq_hz: Some(440.0),
                        target_amp: None,
                    },
                },
                PhonationCmd::NoteOn {
                    note_id,
                    kick: PhonationKick::Planned { strength: 1.0 },
                },
            ],
            notes: vec![PhonationNoteSpec {
                note_id,
                onset: 0,
                hold_ticks: Some(8),
                freq_hz: 220.0,
                amp: 0.5,
                smoothing_tau_sec: 0.0,
                body: BodySnapshot {
                    kind: "sine".to_string(),
                    amp_scale: 1.0,
                    brightness: 0.0,
                    noise_mix: 0.0,
                },
                articulation,
            }],
            onsets: Vec::new(),
        };
        renderer.render(&board, &[batch], 0, &rhythms, &[], &[]);

        let key = VoiceKey {
            source_id: 2,
            note_id,
            kind: VoiceKind::Phonation,
        };
        let voice = renderer.voices.get(&key).expect("voice");
        assert!((voice.debug_target_freq_hz() - 440.0).abs() < 1e-6);
    }

    #[test]
    fn ensure_voice_without_impulse_is_silent() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let body = BodySnapshot {
            kind: "sine".to_string(),
            amp_scale: 1.0,
            brightness: 0.0,
            noise_mix: 0.0,
        };
        let voice_targets = [VoiceTarget {
            id: 11,
            pitch_hz: 220.0,
            amp: 0.4,
        }];
        let cmds = [AudioCommand::EnsureVoice {
            id: 11,
            body,
            pitch_hz: 220.0,
            amp: 0.4,
        }];
        let out = renderer.render(
            &IntentBoard::new(0, 0),
            &[],
            0,
            &rhythms,
            &voice_targets,
            &cmds,
        );
        assert!(out.iter().all(|s| s.abs() <= 1e-6));
    }

    #[test]
    fn ensure_voice_then_impulse_emits_audio() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let body = BodySnapshot {
            kind: "harmonic".to_string(),
            amp_scale: 1.0,
            brightness: 0.6,
            noise_mix: 0.2,
        };
        let voice_targets = [VoiceTarget {
            id: 12,
            pitch_hz: 220.0,
            amp: 0.4,
        }];
        let cmds = [
            AudioCommand::EnsureVoice {
                id: 12,
                body,
                pitch_hz: 220.0,
                amp: 0.4,
            },
            AudioCommand::Impulse {
                id: 12,
                energy: 1.0,
            },
        ];
        let out = renderer.render(
            &IntentBoard::new(0, 0),
            &[],
            0,
            &rhythms,
            &voice_targets,
            &cmds,
        );
        assert!(out.iter().any(|s| s.abs() > 1e-6));
    }

    #[test]
    fn impulse_unknown_id_is_silent() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let cmds = [AudioCommand::Impulse {
            id: 99,
            energy: 1.0,
        }];
        let out = renderer.render(&IntentBoard::new(0, 0), &[], 0, &rhythms, &[], &cmds);
        assert!(out.iter().all(|s| s.abs() <= 1e-6));
    }

    #[test]
    fn ensure_voice_for_unknown_id_is_silent_and_not_created() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let body = BodySnapshot {
            kind: "sine".to_string(),
            amp_scale: 1.0,
            brightness: 0.0,
            noise_mix: 0.0,
        };
        let cmds = [
            AudioCommand::EnsureVoice {
                id: 21,
                body,
                pitch_hz: 220.0,
                amp: 0.4,
            },
            AudioCommand::Impulse {
                id: 21,
                energy: 1.0,
            },
        ];
        let out = renderer.render(&IntentBoard::new(0, 0), &[], 0, &rhythms, &[], &cmds);
        assert!(out.iter().all(|s| s.abs() <= 1e-6));
        assert!(renderer.agent_voices.is_empty());
    }

    #[test]
    fn impulse_before_ensure_voice_still_emits_audio() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let body = BodySnapshot {
            kind: "sine".to_string(),
            amp_scale: 1.0,
            brightness: 0.0,
            noise_mix: 0.0,
        };
        let voice_targets = [VoiceTarget {
            id: 13,
            pitch_hz: 220.0,
            amp: 0.4,
        }];
        let cmds = [
            AudioCommand::Impulse {
                id: 13,
                energy: 1.0,
            },
            AudioCommand::EnsureVoice {
                id: 13,
                body,
                pitch_hz: 220.0,
                amp: 0.4,
            },
        ];
        let out = renderer.render(
            &IntentBoard::new(0, 0),
            &[],
            0,
            &rhythms,
            &voice_targets,
            &cmds,
        );
        assert!(out.iter().any(|s| s.abs() > 1e-6));
    }

    #[test]
    fn ensure_voice_is_idempotent() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let body = BodySnapshot {
            kind: "harmonic".to_string(),
            amp_scale: 1.0,
            brightness: 0.7,
            noise_mix: 0.2,
        };
        let voice_targets = [VoiceTarget {
            id: 14,
            pitch_hz: 220.0,
            amp: 0.4,
        }];
        let cmds = [
            AudioCommand::EnsureVoice {
                id: 14,
                body: body.clone(),
                pitch_hz: 220.0,
                amp: 0.4,
            },
            AudioCommand::EnsureVoice {
                id: 14,
                body,
                pitch_hz: 220.0,
                amp: 0.4,
            },
            AudioCommand::Impulse {
                id: 14,
                energy: 1.0,
            },
        ];
        let out = renderer
            .render(
                &IntentBoard::new(0, 0),
                &[],
                0,
                &rhythms,
                &voice_targets,
                &cmds,
            )
            .to_vec();
        assert_eq!(renderer.agent_voices.len(), 1);
        assert!(out.iter().any(|s| s.abs() > 1e-6));
    }

    #[test]
    fn ensure_voice_created_in_same_render_receives_target_update() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let body = BodySnapshot {
            kind: "sine".to_string(),
            amp_scale: 1.0,
            brightness: 0.0,
            noise_mix: 0.0,
        };
        let voice_targets = [VoiceTarget {
            id: 16,
            pitch_hz: 440.0,
            amp: 0.8,
        }];
        let cmds = [AudioCommand::EnsureVoice {
            id: 16,
            body,
            pitch_hz: 110.0,
            amp: 0.1,
        }];
        let _ = renderer
            .render(
                &IntentBoard::new(0, 0),
                &[],
                0,
                &rhythms,
                &voice_targets,
                &cmds,
            )
            .to_vec();
        let voice = renderer.agent_voices.get(&16).expect("voice");
        assert!((voice.debug_target_freq_hz() - 440.0).abs() < 1e-6);
        assert!((voice.debug_target_amp() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn shutdown_releases_agent_voice_and_becomes_idle() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let body = BodySnapshot {
            kind: "sine".to_string(),
            amp_scale: 1.0,
            brightness: 0.0,
            noise_mix: 0.0,
        };
        let voice_targets = [VoiceTarget {
            id: 15,
            pitch_hz: 220.0,
            amp: 0.4,
        }];
        let cmds = [
            AudioCommand::EnsureVoice {
                id: 15,
                body,
                pitch_hz: 220.0,
                amp: 0.4,
            },
            AudioCommand::Impulse {
                id: 15,
                energy: 1.0,
            },
        ];
        renderer.render(
            &IntentBoard::new(0, 0),
            &[],
            0,
            &rhythms,
            &voice_targets,
            &cmds,
        );
        renderer.shutdown_at(0);
        let now = default_release_ticks(tb) + 2;
        renderer.render(&IntentBoard::new(0, 0), &[], now, &rhythms, &[], &[]);
        assert!(renderer.is_idle());
    }

    #[test]
    fn agent_voice_releases_after_agent_disappears() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let body = BodySnapshot {
            kind: "sine".to_string(),
            amp_scale: 1.0,
            brightness: 0.0,
            noise_mix: 0.0,
        };
        let voice_targets = [VoiceTarget {
            id: 22,
            pitch_hz: 220.0,
            amp: 0.4,
        }];
        let cmds = [
            AudioCommand::EnsureVoice {
                id: 22,
                body,
                pitch_hz: 220.0,
                amp: 0.4,
            },
            AudioCommand::Impulse {
                id: 22,
                energy: 1.0,
            },
        ];
        renderer.render(
            &IntentBoard::new(0, 0),
            &[],
            0,
            &rhythms,
            &voice_targets,
            &cmds,
        );
        let now = tb.hop as Tick;
        renderer.render(&IntentBoard::new(0, 0), &[], now, &rhythms, &[], &[]);
        assert_eq!(renderer.agent_voices.len(), 1);
        let done_at = now.saturating_add(default_release_ticks(tb) + 2);
        renderer.render(&IntentBoard::new(0, 0), &[], done_at, &rhythms, &[], &[]);
        assert!(renderer.is_idle());
    }
}
