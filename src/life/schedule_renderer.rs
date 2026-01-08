use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::individual::PhonationBatch;
use crate::life::intent::{Intent, IntentBoard};
use crate::life::phonation_engine::PhonationCmd;
use crate::life::sound_voice::{SoundVoice, default_release_ticks};
use std::collections::HashMap;
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
    voices: HashMap<VoiceKey, SoundVoice>,
    add_future_ticks: Tick,
    add_past_ticks: Tick,
    did_full_resync: bool,
    cutoff_tick: Option<Tick>,
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
        }
    }

    pub fn render(
        &mut self,
        board: &IntentBoard,
        phonation_batches: &[PhonationBatch],
        now: Tick,
        rhythms: &NeuralRhythms,
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

        let end = now.saturating_add(hop as Tick);
        let dt = 1.0 / fs;
        let mut rhythms = *rhythms;
        self.apply_phonation_batches(phonation_batches, now, &rhythms, dt);
        // For each tick: kick due births first, then render the sample.
        for tick in now..end {
            let idx = (tick - now) as usize;
            let mut acc = 0.0f32;
            for (key, voice) in self.voices.iter_mut() {
                voice.apply_updates_if_due(tick);
                let kicked = voice.kick_if_due(tick, &rhythms, dt);
                if kicked {
                    debug!(
                        target: "phonation::birth_fire",
                        "id={} onset={} kick_applied=true kind=BirthOnce",
                        key.source_id,
                        voice.onset()
                    );
                }
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
        if intent.amp == 0.0 && intent.kind != crate::life::intent::IntentKind::BirthOnce {
            return;
        }
        if intent.kind == crate::life::intent::IntentKind::BirthOnce
            && intent.articulation.is_none()
        {
            debug!(
                target: "phonation::birth_skip",
                source_id = intent.source_id,
                intent_id = intent.intent_id,
                "BirthOnce intent without articulation; skipping voice creation"
            );
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
        if let Some(mut voice) = SoundVoice::from_intent(self.time, intent) {
            voice.note_on(onset);
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
                            kind: crate::life::intent::IntentKind::Normal,
                            onset: spec.onset,
                            duration: hold_ticks,
                            freq_hz: spec.freq_hz,
                            amp: spec.amp,
                            tag: None,
                            confidence: 1.0,
                            body: Some(spec.body.clone()),
                            articulation: Some(spec.articulation.clone()),
                        };
                        if let Some(mut voice) = SoundVoice::from_intent(self.time, intent) {
                            voice.note_on(spec.onset);
                            voice.schedule_planned_kick(kick);
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
    use crate::core::modulation::NeuralRhythms;
    use crate::life::individual::{
        AnyArticulationCore, ArticulationWrapper, PhonationBatch, PhonationNoteSpec, SequencedCore,
    };
    use crate::life::intent::BodySnapshot;
    use crate::life::phonation_engine::{PhonationKick, PhonationUpdate};

    #[test]
    fn birth_kick_consumed_on_onset_tick() {
        let tb = Timebase {
            fs: 48_000.0,
            hop: 8,
        };
        let mut board = IntentBoard::new(1, 1);
        let articulation = ArticulationWrapper::new(
            AnyArticulationCore::Seq(SequencedCore {
                timer: 0.0,
                duration: 0.1,
                env_level: 0.0,
            }),
            0.0,
        );
        board.publish(Intent {
            source_id: 1,
            intent_id: 0,
            kind: crate::life::intent::IntentKind::BirthOnce,
            onset: 0,
            duration: 16,
            freq_hz: 440.0,
            amp: 0.0,
            tag: None,
            confidence: 1.0,
            body: None,
            articulation: Some(articulation),
        });

        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        renderer.render(&board, &[], 0, &rhythms);

        assert!(
            renderer
                .voices
                .values()
                .all(|voice| !voice.birth_kick_pending()),
            "expected birth kick to be consumed on onset tick"
        );
    }

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
                        freq_hz: Some(440.0),
                        amp: Some(0.25),
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
        renderer.render(&board, &[batch], 0, &rhythms);

        let key = VoiceKey {
            source_id: 2,
            note_id,
            kind: VoiceKind::Phonation,
        };
        let voice = renderer.voices.get(&key).expect("voice");
        assert!((voice.debug_body_freq_hz() - 440.0).abs() < 1e-6);
        assert!((voice.debug_body_amp() - 0.25).abs() < 1e-6);
    }
}
