use crate::core::landscape::Landscape;
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::intent::{BodySnapshot, Intent};
use crate::life::perceptual::{FeaturesNow, PerceptualContext};
use rand::rngs::SmallRng;

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
    pub next_intent_tick: Tick,
    pub intent_seq: u64,
    pub rng: SmallRng,
}

impl Individual {
    pub fn metadata(&self) -> &AgentMetadata {
        &self.metadata
    }

    pub fn force_set_pitch_log2(&mut self, log_freq: f32) {
        let log_freq = log_freq.max(0.0);
        self.body.set_pitch_log2(log_freq);
        self.target_pitch_log2 = log_freq;
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

    pub fn plan_intents(
        &mut self,
        tb: &Timebase,
        now: Tick,
        hop: usize,
        _landscape: &Landscape,
    ) -> Vec<Intent> {
        let hop_tick = hop as Tick;
        let offset = (hop_tick / 2).max(1);
        let mut next = self.next_intent_tick;
        if next == 0 {
            next = now.saturating_add(offset);
        }

        let period = hop_tick.saturating_mul(4).max(1);
        let min_next = now.saturating_add(offset);
        if next < min_next {
            next = min_next;
        }
        let window_end = now.saturating_add(hop_tick);
        let mut intents = Vec::new();
        while next < window_end {
            let mut dur_tick = tb.sec_to_tick(0.08);
            if dur_tick == 0 {
                dur_tick = 1;
            }
            let amp = 1.0;
            let freq_hz = self.body.base_freq_hz();
            let snapshot = self.body_snapshot();
            let kind = snapshot.kind.clone();
            intents.push(Intent {
                source_id: self.id,
                intent_id: self.intent_seq,
                onset: next,
                duration: dur_tick,
                freq_hz,
                amp,
                tag: Some(format!("agent:{} {}", self.id, kind)),
                confidence: 1.0,
                body: Some(snapshot),
            });
            self.intent_seq = self.intent_seq.wrapping_add(1);
            next = next.saturating_add(period);
        }
        self.next_intent_tick = next;
        intents
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
