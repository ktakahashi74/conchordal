use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::audio::{AudioAgentState, AudioCommand, Voice, default_release_ticks};
use crate::life::individual::PhonationBatch;
use crate::life::intent::{BodySnapshot, Intent, IntentBoard};
use crate::life::phonation_engine::PhonationCmd;
use std::collections::{HashMap, HashSet};
use tracing::debug;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum VoiceKind {
    Intent,
    Phonation,
    Agent,
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
    add_future_ticks: Tick,
    add_past_ticks: Tick,
    did_full_resync: bool,
    cutoff_tick: Option<Tick>,
    agent_ids_scratch: HashSet<u64>,
    agent_lookup_scratch: HashMap<u64, usize>,
    agent_body_cache: HashMap<u64, BodySnapshot>,
    body_dirty: HashSet<u64>,
}

impl ScheduleRenderer {
    pub fn new(time: Timebase) -> Self {
        let add_future_ticks = (time.hop as Tick).saturating_mul(4).max(1);
        let add_past_ticks = schedule_add_past_ticks(time);
        Self {
            time,
            buf: vec![0.0; time.hop],
            voices: HashMap::new(),
            add_future_ticks,
            add_past_ticks,
            did_full_resync: false,
            cutoff_tick: None,
            agent_ids_scratch: HashSet::new(),
            agent_lookup_scratch: HashMap::new(),
            agent_body_cache: HashMap::new(),
            body_dirty: HashSet::new(),
        }
    }

    pub fn render(
        &mut self,
        board: &IntentBoard,
        phonation_batches: &[PhonationBatch],
        now: Tick,
        rhythms: &NeuralRhythms,
        agent_states: &[AudioAgentState],
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
        self.agent_lookup_scratch.clear();
        self.agent_ids_scratch.reserve(agent_states.len());
        self.agent_lookup_scratch.reserve(agent_states.len());
        for (idx, state) in agent_states.iter().enumerate() {
            self.agent_ids_scratch.insert(state.id);
            self.agent_lookup_scratch.insert(state.id, idx);
        }
        self.agent_body_cache
            .retain(|id, _| self.agent_ids_scratch.contains(id));
        self.body_dirty
            .retain(|id| self.agent_ids_scratch.contains(id));
        self.voices.retain(|key, voice| match key.kind {
            VoiceKind::Agent => self.agent_ids_scratch.contains(&key.source_id),
            _ => !voice.is_done(now),
        });

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
        for state in agent_states {
            let key = VoiceKey {
                source_id: state.id,
                note_id: 0,
                kind: VoiceKind::Agent,
            };
            if let Some(voice) = self.voices.get_mut(&key) {
                voice.set_target(state.pitch_hz, state.amp, 0.0);
            }
        }
        for cmd in audio_cmds {
            match cmd {
                AudioCommand::SetBody { id, body } => {
                    self.agent_body_cache.insert(*id, body.clone());
                    self.body_dirty.insert(*id);
                }
                AudioCommand::Trigger { id, ev, body } => {
                    let mut force_recreate = false;
                    if let Some(snapshot) = body.as_ref() {
                        self.agent_body_cache.insert(*id, snapshot.clone());
                        force_recreate = true;
                    }
                    if self.body_dirty.remove(id) {
                        force_recreate = true;
                    }
                    let Some(state_idx) = self.agent_lookup_scratch.get(id) else {
                        continue;
                    };
                    let state = &agent_states[*state_idx];
                    let key = VoiceKey {
                        source_id: *id,
                        note_id: 0,
                        kind: VoiceKind::Agent,
                    };
                    let is_done = self
                        .voices
                        .get(&key)
                        .is_some_and(|voice| voice.is_done(now));
                    if is_done || force_recreate {
                        self.voices.remove(&key);
                    }
                    if let Some(voice) = self.voices.get_mut(&key) {
                        voice.trigger(*ev);
                        continue;
                    }
                    let body_snapshot = body
                        .as_ref()
                        .cloned()
                        .or_else(|| self.agent_body_cache.get(id).cloned());
                    let intent = Intent {
                        source_id: *id,
                        intent_id: 0,
                        onset: now,
                        duration: Tick::MAX,
                        freq_hz: state.pitch_hz,
                        amp: state.amp,
                        tag: None,
                        confidence: 1.0,
                        body: body_snapshot,
                        articulation: None,
                    };
                    if let Some(mut voice) = Voice::from_intent(self.time, intent) {
                        voice.note_on(now);
                        voice.trigger(*ev);
                        self.voices.insert(key, voice);
                    }
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
            self.buf[idx] = acc;
            rhythms.advance_in_place(dt);
        }

        self.apply_limiter();
        &self.buf
    }

    pub fn is_idle(&self) -> bool {
        self.voices.is_empty()
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
    use crate::life::audio::{
        AudioAgentState, AudioCommand, AudioEvent, Voice, default_release_ticks,
    };
    use crate::life::individual::{
        AnyArticulationCore, ArticulationWrapper, PhonationBatch, PhonationNoteSpec, SequencedCore,
    };
    use crate::life::intent::BodySnapshot;
    use crate::life::phonation_engine::{PhonationKick, PhonationUpdate};

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
    fn trigger_recreates_agent_voice_after_done() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let mut renderer = ScheduleRenderer::new(tb);
        let intent = Intent {
            source_id: 1,
            intent_id: 0,
            onset: 0,
            duration: 1,
            freq_hz: 440.0,
            amp: 0.5,
            tag: None,
            confidence: 1.0,
            body: None,
            articulation: None,
        };
        let mut voice = Voice::from_intent(tb, intent).expect("voice");
        voice.note_on(0);
        let done_tick = 1 + default_release_ticks(tb) + 1;
        renderer.voices.insert(
            VoiceKey {
                source_id: 1,
                note_id: 0,
                kind: VoiceKind::Agent,
            },
            voice,
        );

        let agent_states = [AudioAgentState {
            id: 1,
            pitch_hz: 440.0,
            amp: 0.5,
        }];
        let cmds = [AudioCommand::Trigger {
            id: 1,
            ev: AudioEvent::Impulse { energy: 1.0 },
            body: Some(BodySnapshot {
                kind: "sine".to_string(),
                amp_scale: 1.0,
                brightness: 0.0,
                noise_mix: 0.0,
            }),
        }];
        let rhythms = NeuralRhythms::default();
        let out = renderer.render(
            &IntentBoard::new(0, 0),
            &[],
            done_tick,
            &rhythms,
            &agent_states,
            &cmds,
        );
        assert!(out.iter().any(|s| s.abs() > 1e-6));
    }

    #[test]
    fn trigger_uses_body_snapshot_when_state_has_none() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let body = BodySnapshot {
            kind: "harmonic".to_string(),
            amp_scale: 1.0,
            brightness: 0.6,
            noise_mix: 0.2,
        };
        let agent_states = [AudioAgentState {
            id: 3,
            pitch_hz: 220.0,
            amp: 0.4,
        }];
        let cmds = [AudioCommand::Trigger {
            id: 3,
            ev: AudioEvent::Impulse { energy: 1.0 },
            body: Some(body.clone()),
        }];
        let out = renderer.render(
            &IntentBoard::new(0, 0),
            &[],
            0,
            &rhythms,
            &agent_states,
            &cmds,
        );
        assert!(out.iter().any(|s| s.abs() > 1e-6));
        let cached = renderer.agent_body_cache.get(&3).expect("body cache");
        assert_eq!(cached.kind, body.kind);
        assert!((cached.amp_scale - body.amp_scale).abs() < 1e-6);
        assert!((cached.brightness - body.brightness).abs() < 1e-6);
        assert!((cached.noise_mix - body.noise_mix).abs() < 1e-6);
    }

    #[test]
    fn body_changed_updates_cache_for_followup_trigger() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let body = BodySnapshot {
            kind: "harmonic".to_string(),
            amp_scale: 1.0,
            brightness: 0.9,
            noise_mix: 0.1,
        };
        let agent_states = [AudioAgentState {
            id: 7,
            pitch_hz: 330.0,
            amp: 0.3,
        }];
        let cmds = [
            AudioCommand::SetBody {
                id: 7,
                body: body.clone(),
            },
            AudioCommand::Trigger {
                id: 7,
                ev: AudioEvent::Impulse { energy: 1.0 },
                body: None,
            },
        ];
        let out = renderer.render(
            &IntentBoard::new(0, 0),
            &[],
            0,
            &rhythms,
            &agent_states,
            &cmds,
        );
        assert!(out.iter().any(|s| s.abs() > 1e-6));
        let cached = renderer.agent_body_cache.get(&7).expect("body cache");
        assert_eq!(cached.kind, body.kind);
        assert!((cached.amp_scale - body.amp_scale).abs() < 1e-6);
        assert!((cached.brightness - body.brightness).abs() < 1e-6);
        assert!((cached.noise_mix - body.noise_mix).abs() < 1e-6);
        assert!(!renderer.body_dirty.contains(&7));
    }
}
