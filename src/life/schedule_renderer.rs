use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::control::Routing;
use crate::life::phonation_engine::ToneCmd;
use crate::life::sound::Tone;
use crate::life::voice::PhonationBatch;
use std::collections::HashMap;
use tracing::debug;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ToneKey {
    source_id: u64,
    tone_id: u64,
}

struct RoutedTone {
    tone: Tone,
    routing: Routing,
}

pub struct RenderFrame<'a> {
    pub listener: &'a [f32],
    pub perceptual: &'a [f32],
}

pub struct ScheduleRenderer {
    time: Timebase,
    buf_listener: Vec<f32>,
    buf_perceptual: Vec<f32>,
    tones: HashMap<ToneKey, RoutedTone>,
    cutoff_tick: Option<Tick>,
}

pub type SoundRenderer = ScheduleRenderer;

impl ScheduleRenderer {
    pub fn new(time: Timebase) -> Self {
        Self {
            time,
            buf_listener: vec![0.0; time.hop],
            buf_perceptual: vec![0.0; time.hop],
            tones: HashMap::new(),
            cutoff_tick: None,
        }
    }

    pub fn render(
        &mut self,
        phonation_batches: &[PhonationBatch],
        now: Tick,
        rhythms: &NeuralRhythms,
    ) -> RenderFrame<'_> {
        let hop = self.time.hop;
        if self.buf_listener.len() != hop {
            self.buf_listener.resize(hop, 0.0);
        }
        if self.buf_perceptual.len() != hop {
            self.buf_perceptual.resize(hop, 0.0);
        }
        self.buf_listener.fill(0.0);
        self.buf_perceptual.fill(0.0);

        let fs = self.time.fs;
        if fs <= 0.0 {
            return RenderFrame {
                listener: &self.buf_listener,
                perceptual: &self.buf_perceptual,
            };
        }

        self.tones.retain(|_, rt| !rt.tone.is_done(now));

        let end = now.saturating_add(hop as Tick);
        let dt = 1.0 / fs;
        let mut rhythms = *rhythms;
        self.apply_phonation_batches(phonation_batches, now, &rhythms, dt);
        for tick in now..end {
            let idx = (tick - now) as usize;
            let mut acc_listener = 0.0f32;
            let mut acc_perceptual = 0.0f32;
            for (_key, rt) in self.tones.iter_mut() {
                rt.tone.apply_updates_if_due(tick);
                rt.tone.kick_planned_if_due(tick);
                let sample = rt.tone.render_tick(tick, fs, dt, &rhythms);
                if rt.routing.to_listener {
                    acc_listener += sample;
                }
                if rt.routing.to_voices {
                    acc_perceptual += sample;
                }
            }
            self.buf_listener[idx] = acc_listener;
            self.buf_perceptual[idx] = acc_perceptual;
            rhythms.advance_in_place(dt);
        }

        RenderFrame {
            listener: &self.buf_listener,
            perceptual: &self.buf_perceptual,
        }
    }

    pub fn is_idle(&self) -> bool {
        self.tones.is_empty()
    }

    pub fn set_cutoff_tick(&mut self, cutoff: Option<Tick>) {
        self.cutoff_tick = cutoff;
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
    fn update_command_applies_to_tone() {
        let tb = Timebase { fs: 1000.0, hop: 4 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let tone_id = 1;
        let batch = PhonationBatch {
            source_id: 2,
            routing: crate::life::control::Routing::default(),
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
    fn update_commands_same_tick_last_wins() {
        let tb = Timebase { fs: 1000.0, hop: 4 };
        let mut renderer = ScheduleRenderer::new(tb);
        let rhythms = NeuralRhythms::default();
        let tone_id = 1;
        let batch = PhonationBatch {
            source_id: 2,
            routing: crate::life::control::Routing::default(),
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
            routing: crate::life::control::Routing::default(),
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
