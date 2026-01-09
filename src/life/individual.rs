use crate::core::landscape::Landscape;
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::intent::BodySnapshot;
use crate::life::perceptual::{FeaturesNow, PerceptualContext};
use crate::life::phonation_engine::{
    CoreState, CoreTickCtx, NoteId, OnsetEvent, PhonationCmd, PhonationEngine, PhonationNoteEvent,
};
use crate::life::scenario::PhonationConfig;
use crate::life::social_density::SocialDensityTrace;
use rand::rngs::SmallRng;

#[path = "articulation_core.rs"]
pub mod articulation_core;
#[path = "pitch_core.rs"]
pub mod pitch_core;
#[path = "sound_body.rs"]
pub mod sound_body;

pub use articulation_core::{
    AnyArticulationCore, ArticulationCore, ArticulationSignal, ArticulationState,
    ArticulationWrapper, DroneCore, KuramotoCore, PinkNoise, PlannedGate, PlannedPitch,
    Sensitivity, SequencedCore,
};
pub use pitch_core::{AnyPitchCore, PitchCore, PitchHillClimbPitchCore, TargetProposal};
pub use sound_body::{AnySoundBody, HarmonicBody, SineBody, SoundBody};

#[derive(Debug, Clone, Default)]
pub struct AgentMetadata {
    pub id: u64,
    pub tag: Option<String>,
    pub group_idx: usize,
    pub member_idx: usize,
}

#[derive(Debug)]
pub struct Individual {
    pub id: u64,
    pub metadata: AgentMetadata,
    pub articulation: ArticulationWrapper,
    pub pitch: AnyPitchCore,
    pub perceptual: PerceptualContext,
    pub phonation: PhonationConfig,
    pub phonation_engine: PhonationEngine,
    pub body: AnySoundBody,
    pub last_signal: ArticulationSignal,
    pub release_gain: f32,
    pub release_sec: f32,
    pub release_pending: bool,
    pub target_pitch_log2: f32,
    pub integration_window: f32,
    pub accumulated_time: f32,
    pub last_theta_phase: f32,
    pub theta_phase_initialized: bool,
    pub last_target_salience: f32,
    pub rng: SmallRng,
}

#[derive(Clone, Debug)]
pub struct PhonationNoteSpec {
    pub note_id: NoteId,
    pub onset: Tick,
    pub hold_ticks: Option<Tick>,
    pub freq_hz: f32,
    pub amp: f32,
    pub smoothing_tau_sec: f32,
    pub body: BodySnapshot,
    pub articulation: ArticulationWrapper,
}

#[derive(Clone, Debug, Default)]
pub struct PhonationBatch {
    pub source_id: u64,
    pub cmds: Vec<PhonationCmd>,
    pub notes: Vec<PhonationNoteSpec>,
    pub onsets: Vec<OnsetEvent>,
}

impl Individual {
    const AMP_EPS: f32 = 1e-6;

    pub fn should_retain(&self) -> bool {
        self.is_alive() || self.phonation_engine.has_active_notes()
    }
    pub fn metadata(&self) -> &AgentMetadata {
        &self.metadata
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn force_set_pitch_log2(&mut self, log_freq: f32) {
        let log_freq = log_freq.max(0.0);
        self.body.set_pitch_log2(log_freq);
        self.target_pitch_log2 = log_freq;
        self.articulation.set_gate(1.0);
        self.accumulated_time = 0.0;
        self.last_theta_phase = 0.0;
        self.theta_phase_initialized = false;
        self.last_target_salience = 0.0;
    }

    /// Update pitch targets at control rate (hop-sized steps).
    pub fn update_pitch_target(
        &mut self,
        rhythms: &NeuralRhythms,
        dt_sec: f32,
        landscape: &Landscape,
    ) {
        let dt_sec = dt_sec.max(0.0);
        let current_freq = self.body.base_freq_hz().max(1.0);
        let current_pitch_log2 = current_freq.log2();
        if self.target_pitch_log2 <= 0.0 {
            self.target_pitch_log2 = current_pitch_log2;
        }
        self.integration_window = 2.0 + 10.0 / current_freq.max(1.0);
        self.accumulated_time += dt_sec;

        // Detect theta wrap to avoid missing zero-crossings at control rate.
        let theta_phase = rhythms.theta.phase;
        let theta_cross = if theta_phase.is_finite() && self.last_theta_phase.is_finite() {
            let wrapped = self.theta_phase_initialized && theta_phase < self.last_theta_phase;
            wrapped && rhythms.theta.mag.is_finite() && rhythms.theta.mag > 0.0
        } else {
            false
        };
        self.last_theta_phase = if theta_phase.is_finite() {
            self.theta_phase_initialized = true;
            theta_phase
        } else {
            self.theta_phase_initialized = false;
            0.0
        };

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

    /// Update articulation at control rate (hop-sized steps).
    pub fn update_articulation(
        &mut self,
        dt_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        global_coupling: f32,
    ) -> ArticulationSignal {
        let current_freq = self.body.base_freq_hz().max(1.0);
        let current_pitch_log2 = current_freq.log2();
        let planned = PlannedPitch {
            target_pitch_log2: self.target_pitch_log2,
            jump_cents_abs: 1200.0 * (self.target_pitch_log2 - current_pitch_log2).abs(),
            salience: self.last_target_salience,
        };
        let apply_planned_pitch = self.articulation.update_gate(&planned, rhythms, dt_sec);
        if apply_planned_pitch {
            self.body.set_pitch_log2(planned.target_pitch_log2);
        }
        let consonance = landscape.evaluate_pitch01(self.body.base_freq_hz());
        let mut signal = self
            .articulation
            .process(consonance, rhythms, dt_sec, global_coupling);
        signal.amplitude *= self.articulation.gate();
        if self.release_pending {
            let step = dt_sec / self.release_sec.max(1e-6);
            self.release_gain = (self.release_gain - step).max(0.0);
        }
        signal.amplitude *= self.release_gain;
        signal.is_active = signal.is_active && signal.amplitude > 0.0;
        self.last_signal = signal;
        signal
    }

    pub fn start_release(&mut self, release_sec: f32) {
        if self.release_pending {
            return;
        }
        self.release_pending = true;
        self.release_sec = release_sec.max(1e-4);
        self.release_gain = self.release_gain.clamp(0.0, 1.0);
    }

    pub fn tick_phonation(
        &mut self,
        tb: &Timebase,
        now: Tick,
        rhythms: &NeuralRhythms,
        social: Option<&SocialDensityTrace>,
        social_coupling: f32,
    ) -> PhonationBatch {
        let hop_tick = (tb.hop as Tick).max(1);
        let frame_end = now.saturating_add(hop_tick);
        let ctx = CoreTickCtx {
            now_tick: now,
            frame_end,
            fs: tb.fs,
            rhythms: *rhythms,
        };
        let state = CoreState {
            is_alive: self.is_alive(),
        };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        self.phonation_engine.tick(
            &ctx,
            &state,
            social,
            social_coupling,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        let mut notes = Vec::new();
        for event in events {
            if let Some(note) = self.build_phonation_note_spec(event, None) {
                notes.push(note);
            }
        }
        PhonationBatch {
            source_id: self.id,
            cmds,
            notes,
            onsets,
        }
    }

    fn build_phonation_note_spec(
        &mut self,
        event: PhonationNoteEvent,
        hold_ticks: Option<Tick>,
    ) -> Option<PhonationNoteSpec> {
        let amp = self.compute_target_amp();
        if amp <= Self::AMP_EPS {
            return None;
        }
        let freq_hz = self.body.base_freq_hz();
        if !freq_hz.is_finite() || freq_hz <= 0.0 {
            return None;
        }
        let articulation = self.articulation_snapshot_for_render();
        let smoothing_tau_sec = 0.0;
        Some(PhonationNoteSpec {
            note_id: event.note_id,
            onset: event.onset_tick,
            hold_ticks,
            freq_hz,
            amp,
            smoothing_tau_sec,
            body: self.body_snapshot(),
            articulation,
        })
    }

    pub(crate) fn compute_target_amp(&self) -> f32 {
        let release_gain = self.release_gain.clamp(0.0, 1.0);
        // Include articulation gate in the final target amp.
        let gate = self.articulation.gate().clamp(0.0, 1.0);
        let mut amp = self.body.amp() * release_gain * gate;
        if !amp.is_finite() {
            amp = 0.0;
        }
        amp.max(0.0)
    }

    /// Gate is baked into amp, so render-side gate is fixed to 1.0 while other articulation state is preserved.
    fn articulation_snapshot_for_render(&self) -> ArticulationWrapper {
        let mut articulation = self.articulation.clone();
        // Normalize render gate to avoid double-applying the gate.
        articulation.set_gate(1.0);
        articulation
    }

    fn body_snapshot(&self) -> BodySnapshot {
        match &self.body {
            AnySoundBody::Sine(_body) => BodySnapshot {
                kind: "sine".to_string(),
                // Target amp already includes body gain; keep snapshot scale neutral.
                amp_scale: 1.0,
                brightness: 0.0,
                noise_mix: 0.0,
            },
            AnySoundBody::Harmonic(body) => BodySnapshot {
                kind: "harmonic".to_string(),
                // Target amp already includes body gain; keep snapshot scale neutral.
                amp_scale: 1.0,
                brightness: body.genotype.brightness.clamp(0.0, 1.0),
                noise_mix: body.genotype.jitter.clamp(0.0, 1.0),
            },
        }
    }

    pub fn render_spectrum(&mut self, amps: &mut [f32], space: &Log2Space) {
        let signal = self.last_signal;
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        self.body.project_spectral_body(amps, space, &signal);
    }

    pub fn is_alive(&self) -> bool {
        self.articulation.is_alive() && self.release_gain > 0.0
    }
}
