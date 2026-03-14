use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::individual::PhonationBatch;
use crate::life::phonation_engine::NoteCmd;
use crate::life::sound::Voice;
use std::collections::HashMap;
use tracing::debug;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct VoiceKey {
    source_id: u64,
    note_id: u64,
}

pub struct ScheduleRenderer {
    time: Timebase,
    buf: Vec<f32>,
    voices: HashMap<VoiceKey, Voice>,
    cutoff_tick: Option<Tick>,
}

pub type SoundRenderer = ScheduleRenderer;

impl ScheduleRenderer {
    pub fn new(time: Timebase) -> Self {
        Self {
            time,
            buf: vec![0.0; time.hop],
            voices: HashMap::new(),
            cutoff_tick: None,
        }
    }

    pub fn render(
        &mut self,
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

        let end = now.saturating_add(hop as Tick);
        let dt = 1.0 / fs;
        let mut rhythms = *rhythms;
        self.apply_phonation_batches(phonation_batches, now, &rhythms, dt);
        for tick in now..end {
            let idx = (tick - now) as usize;
            let mut acc = 0.0f32;
            for (_key, voice) in self.voices.iter_mut() {
                voice.apply_updates_if_due(tick);
                voice.kick_planned_if_due(tick);
                acc += voice.render_tick(tick, fs, dt, &rhythms);
            }
            self.buf[idx] = acc;
            rhythms.advance_in_place(dt);
        }

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
                    NoteCmd::NoteOn { note_id, kick } => {
                        let key = VoiceKey {
                            source_id: batch.source_id,
                            note_id,
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
                        if let Some(mut voice) = Voice::from_parts(
                            self.time,
                            spec.onset,
                            hold_ticks,
                            spec.freq_hz,
                            spec.amp,
                            Some(spec.body.clone()),
                            Some(spec.render_modulator.clone()),
                            spec.adsr,
                        ) {
                            voice.seed_modal_phases(modal_phase_seed(
                                batch.source_id,
                                spec.onset,
                                note_id,
                            ));
                            voice.set_smoothing_tau_sec(spec.smoothing_tau_sec);
                            voice.note_on(spec.onset);
                            voice.schedule_planned_kick(kick);
                            voice.arm_onset_trigger(kick.strength.max(0.0));
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
                    NoteCmd::NoteOff { note_id, off_tick } => {
                        let key = VoiceKey {
                            source_id: batch.source_id,
                            note_id,
                        };
                        if let Some(voice) = self.voices.get_mut(&key) {
                            voice.note_off(off_tick);
                        }
                    }
                    NoteCmd::Update { .. } => {}
                }
            }
            for cmd in &batch.cmds {
                let NoteCmd::Update {
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
                };
                let Some(voice) = self.voices.get_mut(&key) else {
                    continue;
                };
                let tick = at_tick.unwrap_or(now);
                voice.schedule_update(tick, update);
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
    use crate::life::individual::{NoteSpec, PhonationBatch};
    use crate::life::phonation_engine::{NoteUpdate, OnsetKick};
    use crate::life::sound::{BodyKind, BodySnapshot, RenderModulatorSpec, default_release_ticks};

    #[test]
    fn update_command_applies_to_voice() {
        let tb = Timebase { fs: 1000.0, hop: 4 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let note_id = 1;
        let batch = PhonationBatch {
            source_id: 2,
            cmds: vec![
                NoteCmd::Update {
                    note_id,
                    at_tick: Some(0),
                    update: NoteUpdate {
                        target_freq_hz: Some(440.0),
                        target_amp: Some(0.25),
                        continuous_drive: None,
                    },
                },
                NoteCmd::NoteOn {
                    note_id,
                    kick: OnsetKick { strength: 1.0 },
                },
            ],
            notes: vec![NoteSpec {
                note_id,
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
                    voices: 1,
                    motion: 0.0,
                    ratios: None,
                },
                render_modulator: RenderModulatorSpec::SeqGate { duration_sec: 0.1 },
                adsr: None,
            }],
            onsets: Vec::new(),
        };
        renderer.render(&[batch], 0, &rhythms);

        let key = VoiceKey {
            source_id: 2,
            note_id,
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
        let note_id = 1;
        let batch = PhonationBatch {
            source_id: 2,
            cmds: vec![
                NoteCmd::Update {
                    note_id,
                    at_tick: Some(0),
                    update: NoteUpdate {
                        target_freq_hz: Some(330.0),
                        target_amp: None,
                        continuous_drive: None,
                    },
                },
                NoteCmd::Update {
                    note_id,
                    at_tick: Some(0),
                    update: NoteUpdate {
                        target_freq_hz: Some(440.0),
                        target_amp: None,
                        continuous_drive: None,
                    },
                },
                NoteCmd::NoteOn {
                    note_id,
                    kick: OnsetKick { strength: 1.0 },
                },
            ],
            notes: vec![NoteSpec {
                note_id,
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
                    voices: 1,
                    motion: 0.0,
                    ratios: None,
                },
                render_modulator: RenderModulatorSpec::SeqGate { duration_sec: 0.1 },
                adsr: None,
            }],
            onsets: Vec::new(),
        };
        renderer.render(&[batch], 0, &rhythms);

        let key = VoiceKey {
            source_id: 2,
            note_id,
        };
        let voice = renderer.voices.get(&key).expect("voice");
        assert!((voice.debug_target_freq_hz() - 440.0).abs() < 1e-6);
    }

    #[test]
    fn shutdown_releases_voices_and_becomes_idle() {
        let tb = Timebase { fs: 1000.0, hop: 8 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let note_id = 1;
        let batch = PhonationBatch {
            source_id: 1,
            cmds: vec![NoteCmd::NoteOn {
                note_id,
                kick: OnsetKick { strength: 1.0 },
            }],
            notes: vec![NoteSpec {
                note_id,
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
                    voices: 1,
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
