use crate::core::landscape::Landscape;
use crate::core::log2space::{Log2Space, sample_scan_linear_log2};
use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::intent::{BodySnapshot, Intent};
use crate::life::intent_planner::{choose_best_gesture_tf_by_pred_c, choose_onset_by_density};
use crate::life::perceptual::{FeaturesNow, PerceptualContext};
use rand::rngs::SmallRng;
use std::collections::VecDeque;
use tracing::debug;

#[path = "articulation_core.rs"]
pub mod articulation_core;
#[path = "pitch_core.rs"]
pub mod pitch_core;
#[path = "sound_body.rs"]
pub mod sound_body;

pub use articulation_core::{
    AnyArticulationCore, ArticulationCore, ArticulationSignal, ArticulationState, ArticulationStep,
    ArticulationWrapper, DroneCore, ErrorState, KuramotoCore, PinkNoise, PlannedGate, PlannedPitch,
    Sensitivity, SequencedCore,
};
pub use pitch_core::{AnyPitchCore, PitchCore, PitchHillClimbPitchCore, TargetProposal};
pub use sound_body::{AnySoundBody, HarmonicBody, SineBody, SoundBody};

pub trait AudioAgent {
    fn id(&self) -> u64;
    fn metadata(&self) -> &AgentMetadata;
    fn render_wave(
        &mut self,
        buffer: &mut [f32],
        fs: f32,
        current_frame: u64,
        dt_sec: f32,
        landscape: &Landscape,
        global_coupling: f32,
    );
    fn render_spectrum(&mut self, amps: &mut [f32], space: &Log2Space);
    fn is_alive(&self) -> bool;
}

#[derive(Debug, Clone, Default)]
pub struct AgentMetadata {
    pub id: u64,
    pub tag: Option<String>,
    pub group_idx: usize,
    pub member_idx: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AudioSample {
    pub sample: f32,
}

#[derive(Debug, Clone)]
pub struct Individual {
    pub id: u64,
    pub metadata: AgentMetadata,
    pub articulation: ArticulationWrapper,
    pub pitch: AnyPitchCore,
    pub perceptual: PerceptualContext,
    pub body: AnySoundBody,
    pub last_signal: ArticulationSignal,
    pub release_gain: f32,
    pub release_sec: f32,
    pub release_pending: bool,
    pub target_pitch_log2: f32,
    pub integration_window: f32,
    pub accumulated_time: f32,
    pub last_theta_sample: f32,
    pub last_target_salience: f32,
    pub last_error_state: ErrorState,
    pub last_error_cents: f32,
    pub error_initialized: bool,
    pub last_chosen_freq_hz: f32,
    pub next_intent_tick: Tick,
    pub intent_seq: u64,
    pub self_confidence: f32,
    pub pred_intent_records: VecDeque<PredIntentRecord>,
    pub pred_intent_records_cap: usize,
    pub rng: SmallRng,
}

#[derive(Clone, Debug)]
pub struct PredIntentRecord {
    pub intent_id: u64,
    pub onset: Tick,
    pub end: Tick,
    pub freq_hz: f32,
    pub pred_c_statepm1: f32,
    pub created_at: Tick,
    pub eval_tick: Tick,
}

impl Individual {
    pub fn metadata(&self) -> &AgentMetadata {
        &self.metadata
    }

    pub fn force_set_pitch_log2(&mut self, log_freq: f32) {
        let log_freq = log_freq.max(0.0);
        self.body.set_pitch_log2(log_freq);
        self.target_pitch_log2 = log_freq;
        self.last_chosen_freq_hz = self.body.base_freq_hz();
        self.articulation.set_gate(1.0);
        self.accumulated_time = 0.0;
        self.last_theta_sample = 0.0;
        self.last_target_salience = 0.0;
        self.last_error_state = ErrorState::default();
        self.last_error_cents = 0.0;
        self.error_initialized = false;
    }

    pub fn update_pitch_target(&mut self, rhythms: &NeuralRhythms, dt: f32, landscape: &Landscape) {
        let dt = dt.max(0.0);
        let current_freq = self.body.base_freq_hz().max(1.0);
        let current_pitch_log2 = current_freq.log2();
        if self.target_pitch_log2 <= 0.0 {
            self.target_pitch_log2 = current_pitch_log2;
        }
        self.integration_window = 2.0 + 10.0 / current_freq.max(1.0);
        self.accumulated_time += dt;

        let theta_signal = rhythms.theta.mag * rhythms.theta.phase.sin();
        let theta_cross = self.last_theta_sample <= 0.0 && theta_signal > 0.0;
        self.last_theta_sample = theta_signal;

        if theta_cross && self.accumulated_time >= self.integration_window {
            let elapsed = self.accumulated_time;
            self.accumulated_time = 0.0;
            let features = FeaturesNow::from_subjective_intensity(&landscape.subjective_intensity);
            debug_assert_eq!(features.distribution.len(), landscape.space.n_bins());
            self.perceptual.ensure_len(features.distribution.len());
            let proposal = self.pitch.propose_target(
                current_pitch_log2,
                self.target_pitch_log2,
                current_freq,
                self.integration_window,
                landscape,
                &self.perceptual,
                &features,
                &mut self.rng,
            );
            self.target_pitch_log2 = proposal.target_pitch_log2;
            self.last_target_salience = proposal.salience;
            if let Some(idx) = landscape.space.index_of_log2(self.target_pitch_log2) {
                self.perceptual.update(idx, &features, elapsed);
            }
        }

        let (fmin, fmax) = landscape.freq_bounds_log2();
        self.target_pitch_log2 = self.target_pitch_log2.clamp(fmin, fmax);
    }

    pub fn start_release(&mut self, release_sec: f32) {
        if self.release_pending {
            return;
        }
        self.release_pending = true;
        self.release_sec = release_sec.max(1e-4);
        self.release_gain = self.release_gain.clamp(0.0, 1.0);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn plan_intents(
        &mut self,
        tb: &Timebase,
        now: Tick,
        perc_tick: Tick,
        hop: usize,
        landscape: &Landscape,
        intents: &[Intent],
        pred_rhythm: Option<&crate::life::predictive_rhythm::PredictiveRhythmBank>,
        pred_c_scan_at: &mut dyn FnMut(Tick) -> Option<std::sync::Arc<[f32]>>,
        agents_pitch: bool,
    ) -> Vec<Intent> {
        self.update_self_confidence_from_perc(&landscape.space, landscape, perc_tick);
        let hop_tick = hop as Tick;
        if self.next_intent_tick != 0 && now < self.next_intent_tick {
            return Vec::new();
        }

        let mut horizon = tb.sec_to_tick(2.0);
        if horizon == 0 {
            horizon = 1;
        }
        horizon = horizon.max(hop_tick.saturating_mul(4).max(1));
        let mut candidates = self.articulation.propose_onsets(tb, now, horizon);
        if candidates.is_empty() {
            return Vec::new();
        }
        if let Some(bank) = pred_rhythm {
            if bank.is_informative(0.05) {
                let mut scored: Vec<(Tick, f32)> = candidates
                    .iter()
                    .map(|&tick| (tick, bank.prior01_at_tick(tb, tick)))
                    .collect();
                scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                if cfg!(debug_assertions) && self.intent_seq.is_multiple_of(32) {
                    let top_prior = scored.first().map(|(_, prior)| *prior).unwrap_or(0.0);
                    let mut top_entries = String::new();
                    for (i, (tick, prior)) in scored.iter().take(5).enumerate() {
                        if i > 0 {
                            top_entries.push_str(", ");
                        }
                        top_entries.push_str(&format!("{tick}:{prior:.3}"));
                    }
                    debug!(
                        target: "rhythm::prior",
                        "agent={} candidates={} top_prior={:.3} top5=[{}]",
                        self.id,
                        scored.len(),
                        top_prior,
                        top_entries
                    );
                }
                let max_keep = 16usize;
                let mut filtered: Vec<Tick> = scored
                    .into_iter()
                    .take(max_keep)
                    .map(|(tick, _)| tick)
                    .collect();
                if filtered.is_empty() {
                    filtered.push(candidates[0]);
                }
                filtered.sort_unstable();
                candidates = filtered;
            }
        }
        let mut eps = tb.sec_to_tick(0.01);
        if eps == 0 {
            eps = 1;
        }
        let fallback_onset = choose_onset_by_density(&candidates, intents, eps)
            .or_else(|| candidates.first().copied());
        let Some(fallback_onset) = fallback_onset else {
            return Vec::new();
        };
        let mut dur_tick = tb.sec_to_tick(0.08);
        if dur_tick == 0 {
            dur_tick = 1;
        }
        let amp = 1.0;
        let base_freq_hz = if self.last_chosen_freq_hz > 0.0 && self.last_chosen_freq_hz.is_finite()
        {
            self.last_chosen_freq_hz
        } else {
            self.body.base_freq_hz()
        };
        let mut chosen = fallback_onset;
        let mut freq_hz = base_freq_hz;
        let mut chosen_score: Option<f32> = None;
        if agents_pitch {
            let mut freq_eps = tb.sec_to_tick(0.01);
            if freq_eps == 0 {
                freq_eps = 1;
            }
            let intent_refs = intents;
            let mut make_freq_candidates = |onset: Tick| {
                let min = onset.saturating_sub(freq_eps);
                let max = onset.saturating_add(freq_eps);
                let neighbors: Vec<f32> = intent_refs
                    .iter()
                    .filter(|intent| intent.onset >= min && intent.onset <= max)
                    .filter_map(|intent| {
                        if intent.freq_hz.is_finite() && intent.freq_hz > 0.0 {
                            Some(intent.freq_hz)
                        } else {
                            None
                        }
                    })
                    .collect();
                self.pitch
                    .propose_freqs_hz_with_neighbors(base_freq_hz, &neighbors, 16, 8, 12.0)
            };
            if let Some(choice) = choose_best_gesture_tf_by_pred_c(
                &landscape.space,
                &candidates,
                base_freq_hz,
                &mut make_freq_candidates,
                &mut *pred_c_scan_at,
            ) {
                chosen = choice.onset;
                freq_hz = choice.freq_hz.clamp(20.0, 20_000.0);
                chosen_score = Some(choice.score);
            }
        }
        self.last_chosen_freq_hz = freq_hz;
        if cfg!(debug_assertions) && self.intent_seq.is_multiple_of(32) {
            let min = chosen.saturating_sub(eps);
            let max = chosen.saturating_add(eps);
            let density = intents
                .iter()
                .filter(|intent| intent.onset >= min && intent.onset <= max)
                .count();
            let score = chosen_score.unwrap_or(f32::NAN);
            debug!(
                target: "intent::plan",
                "agent={} candidates={} chosen={} density={} pred_c={:.3}",
                self.id,
                candidates.len(),
                chosen,
                density,
                score
            );
        }
        let snapshot = self.body_snapshot();
        let kind = snapshot.kind.clone();
        let intent = Intent {
            source_id: self.id,
            intent_id: self.intent_seq,
            onset: chosen,
            duration: dur_tick,
            freq_hz,
            amp,
            tag: Some(format!("agent:{} {}", self.id, kind)),
            confidence: 1.0,
            body: Some(snapshot),
        };
        self.record_pred_intent(&intent, pred_c_scan_at, now, landscape);
        self.intent_seq = self.intent_seq.wrapping_add(1);
        let min_interval = hop_tick.max(dur_tick);
        self.next_intent_tick = chosen.saturating_add(min_interval);
        let intents = vec![intent];
        intents
    }

    pub fn update_self_confidence_from_perc(
        &mut self,
        space: &Log2Space,
        perc_landscape: &Landscape,
        perc_tick: Tick,
    ) {
        if self.pred_intent_records.is_empty() {
            return;
        }
        let mut next_records = VecDeque::with_capacity(self.pred_intent_records.len());
        for record in self.pred_intent_records.drain(..) {
            if perc_tick >= record.end {
                continue;
            }
            if record.eval_tick <= perc_tick && perc_tick < record.end {
                let perc_c =
                    sample_scan_linear_log2(space, &perc_landscape.consonance, record.freq_hz);
                if !perc_c.is_finite() || !record.pred_c_statepm1.is_finite() {
                    continue;
                }
                let err_c = perc_c - record.pred_c_statepm1;
                if !err_c.is_finite() {
                    continue;
                }
                let agreement01 = 1.0 - (err_c.abs() / 2.0).clamp(0.0, 1.0);
                let lr = 0.05;
                self.self_confidence = lerp(self.self_confidence, agreement01, lr).clamp(0.0, 1.0);
                continue;
            }
            next_records.push_back(record);
        }
        self.pred_intent_records = next_records;
    }

    fn record_pred_intent(
        &mut self,
        intent: &Intent,
        pred_c_scan_at: &mut dyn FnMut(Tick) -> Option<std::sync::Arc<[f32]>>,
        now: Tick,
        landscape: &Landscape,
    ) {
        let eval_tick = intent.onset.saturating_add((intent.duration / 2).max(1));
        let pred_c_statepm1 = pred_c_scan_at(eval_tick)
            .map(|scan| sample_scan_linear_log2(&landscape.space, scan.as_ref(), intent.freq_hz))
            .unwrap_or(0.0);
        let pred_c_statepm1 = if pred_c_statepm1.is_finite() {
            pred_c_statepm1
        } else {
            0.0
        };
        let record = PredIntentRecord {
            intent_id: intent.intent_id,
            onset: intent.onset,
            end: intent.onset.saturating_add(intent.duration),
            freq_hz: intent.freq_hz,
            pred_c_statepm1,
            created_at: now,
            eval_tick,
        };
        if self.pred_intent_records.len() >= self.pred_intent_records_cap {
            let _ = self.pred_intent_records.pop_front();
        }
        self.pred_intent_records.push_back(record);
    }

    fn body_snapshot(&self) -> BodySnapshot {
        match &self.body {
            AnySoundBody::Sine(body) => BodySnapshot {
                kind: "sine".to_string(),
                amp_scale: body.amp.clamp(0.0, 1.0),
                brightness: 0.0,
                noise_mix: 0.0,
            },
            AnySoundBody::Harmonic(body) => BodySnapshot {
                kind: "harmonic".to_string(),
                amp_scale: body.amp.clamp(0.0, 1.0),
                brightness: body.genotype.brightness.clamp(0.0, 1.0),
                noise_mix: body.genotype.jitter.clamp(0.0, 1.0),
            },
        }
    }
}

#[cfg(test)]
impl Individual {
    pub fn debug_last_error_state(&self) -> ErrorState {
        self.last_error_state
    }
}

impl AudioAgent for Individual {
    fn id(&self) -> u64 {
        self.id
    }

    fn metadata(&self) -> &AgentMetadata {
        self.metadata()
    }

    fn render_wave(
        &mut self,
        buffer: &mut [f32],
        fs: f32,
        _current_frame: u64,
        dt_sec: f32,
        landscape: &Landscape,
        global_coupling: f32,
    ) {
        if buffer.is_empty() {
            return;
        }
        let dt_per_sample = dt_sec / buffer.len() as f32;
        let mut rhythms = landscape.rhythm;
        for sample in buffer.iter_mut() {
            self.update_pitch_target(&rhythms, dt_per_sample, landscape);
            let current_freq = self.body.base_freq_hz().max(1.0);
            let current_pitch_log2 = current_freq.log2();
            let planned = PlannedPitch {
                target_pitch_log2: self.target_pitch_log2,
                jump_cents_abs: 1200.0 * (self.target_pitch_log2 - current_pitch_log2).abs(),
                salience: self.last_target_salience,
            };
            let pitch_error_cents = 1200.0 * (planned.target_pitch_log2 - current_pitch_log2);
            let d_pitch_error_cents_per_sec = if self.error_initialized && dt_per_sample > 0.0 {
                (pitch_error_cents - self.last_error_cents) / dt_per_sample
            } else {
                0.0
            };
            self.last_error_state = ErrorState {
                pitch_error_cents,
                abs_pitch_error_cents: pitch_error_cents.abs(),
                d_pitch_error_cents_per_sec,
            };
            self.last_error_cents = pitch_error_cents;
            self.error_initialized = true;
            let apply_planned_pitch = self.articulation.update_gate(
                &planned,
                &self.last_error_state,
                &rhythms,
                dt_per_sample,
            );
            if apply_planned_pitch {
                self.body.set_pitch_log2(planned.target_pitch_log2);
            }
            let consonance = landscape.evaluate_pitch01(self.body.base_freq_hz());
            let step: ArticulationStep = self.articulation.process(
                consonance,
                &rhythms,
                dt_per_sample,
                global_coupling,
                &planned,
                &self.last_error_state,
            );
            debug_assert_eq!(step.apply_planned_pitch, apply_planned_pitch);
            let mut signal = step.signal;
            signal.amplitude *= self.articulation.gate();
            if self.release_pending {
                let step = dt_per_sample / self.release_sec.max(1e-6);
                self.release_gain = (self.release_gain - step).max(0.0);
            }
            signal.amplitude *= self.release_gain;
            signal.is_active = signal.is_active && signal.amplitude > 0.0;
            self.last_signal = signal;
            if signal.is_active {
                self.body
                    .articulate_wave(sample, fs, dt_per_sample, &signal);
            }
            rhythms.advance_in_place(dt_per_sample);
        }
    }

    fn render_spectrum(&mut self, amps: &mut [f32], space: &Log2Space) {
        let signal = self.last_signal;
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        self.body.project_spectral_body(amps, space, &signal);
    }

    fn is_alive(&self) -> bool {
        self.articulation.is_alive() && self.release_gain > 0.0
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
