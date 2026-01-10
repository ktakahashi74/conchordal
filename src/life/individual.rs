use crate::core::landscape::Landscape;
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::core::phase::wrap_0_tau;
use crate::core::timebase::{Tick, Timebase};
use crate::life::gate_clock::next_gate_tick;
use crate::life::intent::{BodySnapshot, Intent};
use crate::life::perceptual::{FeaturesNow, PerceptualContext};
use crate::life::phonation_engine::{
    CoreState, CoreTickCtx, NoteId, OnsetEvent, PhonationCmd, PhonationEngine, PhonationNoteEvent,
};
use crate::life::scenario::{
    BehaviorConfig, MotionAutonomy, PhonationConfig, VoiceControl, VoiceOnSpawn,
};
use crate::life::social_density::SocialDensityTrace;
use rand::rngs::SmallRng;
use std::f32::consts::TAU;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StartLatch {
    Pending,
    None,
    WaitForOnset,
    WaitForGateExitThenOnset,
    Disabled,
}

#[derive(Clone, Debug)]
pub struct VoiceRuntime {
    pub voice_enabled: bool,
    pub on_spawn: VoiceOnSpawn,
    pub control: VoiceControl,
    pub start_latch: StartLatch,
}

impl VoiceRuntime {
    pub(crate) fn from_behavior(behavior: &crate::life::scenario::VoiceBehavior) -> Self {
        let voice_enabled = !matches!(behavior.on_spawn, VoiceOnSpawn::Disabled);
        let start_latch = if !voice_enabled {
            StartLatch::Disabled
        } else {
            match behavior.on_spawn {
                VoiceOnSpawn::Immediate => StartLatch::None,
                VoiceOnSpawn::NextRhythm => StartLatch::Pending,
                VoiceOnSpawn::Disabled => StartLatch::Disabled,
            }
        };
        Self {
            voice_enabled,
            on_spawn: behavior.on_spawn,
            control: behavior.control,
            start_latch,
        }
    }

    fn init_on_spawn_latch(&mut self, rhythms: &NeuralRhythms) {
        if self.start_latch != StartLatch::Pending {
            return;
        }
        if !self.voice_enabled {
            self.start_latch = StartLatch::Disabled;
            return;
        }
        match self.on_spawn {
            VoiceOnSpawn::Immediate => {
                self.start_latch = StartLatch::None;
            }
            VoiceOnSpawn::NextRhythm => {
                let phase_in_gate = wrap_0_tau(rhythms.theta.phase) / TAU;
                let in_gate = phase_in_gate > 1e-4;
                self.start_latch = if in_gate {
                    StartLatch::WaitForGateExitThenOnset
                } else {
                    StartLatch::WaitForOnset
                };
            }
            VoiceOnSpawn::Disabled => {
                self.voice_enabled = false;
                self.start_latch = StartLatch::Disabled;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct MotionRuntime {
    pub motion_enabled: bool,
    pub motion_policy: MotionAutonomy,
}

impl MotionRuntime {
    pub(crate) fn from_behavior(behavior: &crate::life::scenario::AutonomyBehavior) -> Self {
        let motion_enabled = matches!(behavior.motion, MotionAutonomy::Immediate);
        Self {
            motion_enabled,
            motion_policy: behavior.motion,
        }
    }

    pub(crate) fn note_first_utterance(&mut self) {
        if matches!(self.motion_policy, MotionAutonomy::AfterFirstUtterance) {
            self.motion_enabled = true;
        }
    }
}

#[derive(Debug)]
pub struct Individual {
    pub id: u64,
    pub metadata: AgentMetadata,
    pub behavior: BehaviorConfig,
    pub voice_runtime: VoiceRuntime,
    pub motion_runtime: MotionRuntime,
    pub articulation: ArticulationWrapper,
    pub pitch: AnyPitchCore,
    pub perceptual: PerceptualContext,
    pub phonation: PhonationConfig,
    pub phonation_engine: PhonationEngine,
    pub body: AnySoundBody,
    pub last_signal: ArticulationSignal,
    pub(crate) release_gain: f32,
    pub(crate) release_sec: f32,
    pub(crate) release_pending: bool,
    pub(crate) target_pitch_log2: Option<f32>,
    pub(crate) integration_window: f32,
    pub(crate) accumulated_time: f32,
    pub(crate) last_theta_phase: f32,
    pub(crate) theta_phase_initialized: bool,
    pub(crate) last_target_salience: f32,
    pub rng: SmallRng,
    pub(crate) birth_once_pending: bool,
    pub(crate) birth_frame: u64,
    pub(crate) birth_once_duration_sec: Option<f32>,
    pub(crate) phonation_scratch: PhonationScratch,
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

#[derive(Debug, Default)]
pub(crate) struct PhonationScratch {
    events: Vec<PhonationNoteEvent>,
}

#[derive(Clone, Debug, Default)]
pub struct PhonationBatch {
    pub source_id: u64,
    pub cmds: Vec<PhonationCmd>,
    pub notes: Vec<PhonationNoteSpec>,
    pub onsets: Vec<OnsetEvent>,
}

impl PhonationBatch {
    pub(crate) fn clear(&mut self) {
        self.cmds.clear();
        self.notes.clear();
        self.onsets.clear();
    }
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

    pub fn target_pitch_log2(&self) -> f32 {
        self.target_pitch_log2
            .unwrap_or_else(|| self.body.base_freq_hz().max(1.0).log2())
    }

    pub fn integration_window(&self) -> f32 {
        self.integration_window
    }

    pub fn release_gain(&self) -> f32 {
        self.release_gain
    }

    pub(crate) fn birth_once_duration_sec(&self) -> Option<f32> {
        self.birth_once_duration_sec
    }

    #[cfg(test)]
    pub(crate) fn set_accumulated_time(&mut self, value: f32) {
        self.accumulated_time = value;
    }

    #[cfg(test)]
    pub(crate) fn accumulated_time(&self) -> f32 {
        self.accumulated_time
    }

    #[cfg(test)]
    pub(crate) fn set_theta_phase_state(&mut self, last_phase: f32, initialized: bool) {
        self.last_theta_phase = last_phase;
        self.theta_phase_initialized = initialized;
    }

    pub fn force_set_pitch_log2(&mut self, log_freq: f32) {
        let log_freq = log_freq.max(0.0);
        self.body.set_pitch_log2(log_freq);
        self.target_pitch_log2 = Some(log_freq);
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
        let mut target_pitch_log2 = self.target_pitch_log2.unwrap_or(current_pitch_log2);
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
                target_pitch_log2,
                current_freq,
                self.integration_window,
                landscape,
                &self.perceptual,
                &features,
                &mut self.rng,
            );
            target_pitch_log2 = proposal.target_pitch_log2;
            self.last_target_salience = proposal.salience;
            if let Some(idx) = landscape.space.index_of_log2(target_pitch_log2) {
                self.perceptual.update(idx, &features, elapsed);
            }
        }

        let (fmin, fmax) = landscape.freq_bounds_log2();
        target_pitch_log2 = target_pitch_log2.clamp(fmin, fmax);
        self.target_pitch_log2 = Some(target_pitch_log2);
    }

    /// Control-rate entry point for pitch + articulation updates.
    pub fn tick_control(
        &mut self,
        dt_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        global_coupling: f32,
    ) -> ArticulationSignal {
        self.update_articulation(dt_sec, rhythms, landscape, global_coupling)
    }

    /// Update articulation at control rate (hop-sized steps).
    pub fn update_articulation(
        &mut self,
        dt_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        global_coupling: f32,
    ) -> ArticulationSignal {
        if self.motion_runtime.motion_enabled {
            self.update_pitch_target(rhythms, dt_sec, landscape);
            self.update_articulation_autonomous(dt_sec, rhythms);
        }
        self.tick_articulation_lifecycle(dt_sec, rhythms, landscape, global_coupling)
    }

    pub fn update_articulation_autonomous(&mut self, dt_sec: f32, rhythms: &NeuralRhythms) {
        let current_freq = self.body.base_freq_hz().max(1.0);
        let current_pitch_log2 = current_freq.log2();
        let target_pitch_log2 = self.target_pitch_log2();
        let planned = PlannedPitch {
            target_pitch_log2,
            jump_cents_abs: 1200.0 * (target_pitch_log2 - current_pitch_log2).abs(),
            salience: self.last_target_salience,
        };
        let apply_planned_pitch = self.articulation.update_gate(&planned, rhythms, dt_sec);
        if apply_planned_pitch {
            self.body.set_pitch_log2(planned.target_pitch_log2);
        }
    }

    pub fn tick_articulation_lifecycle(
        &mut self,
        dt_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        global_coupling: f32,
    ) -> ArticulationSignal {
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
        let mut batch = PhonationBatch::default();
        self.tick_phonation_into(tb, now, rhythms, social, social_coupling, &mut batch);
        batch
    }

    pub fn tick_phonation_into(
        &mut self,
        tb: &Timebase,
        now: Tick,
        rhythms: &NeuralRhythms,
        social: Option<&SocialDensityTrace>,
        social_coupling: f32,
        out: &mut PhonationBatch,
    ) {
        out.source_id = self.id;
        out.clear();
        self.phonation_scratch.events.clear();
        self.voice_runtime.init_on_spawn_latch(rhythms);
        if !self.voice_runtime.voice_enabled {
            return;
        }
        if matches!(self.voice_runtime.control, VoiceControl::Scripted) {
            return;
        }
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
        let mut gate_exit_tick = None;
        if matches!(
            self.voice_runtime.start_latch,
            StartLatch::WaitForGateExitThenOnset
        ) {
            gate_exit_tick = next_gate_tick(now, tb.fs, rhythms.theta, 0.0);
            if gate_exit_tick.is_none() {
                self.voice_runtime.start_latch = StartLatch::WaitForOnset;
            }
        }
        let min_allowed_onset_tick = match self.voice_runtime.start_latch {
            StartLatch::WaitForGateExitThenOnset => gate_exit_tick.or(Some(Tick::MAX)),
            _ => None,
        };
        self.phonation_engine.tick(
            &ctx,
            &state,
            social,
            social_coupling,
            min_allowed_onset_tick,
            &mut out.cmds,
            &mut self.phonation_scratch.events,
            &mut out.onsets,
        );
        let had_note_on = out
            .cmds
            .iter()
            .any(|cmd| matches!(cmd, PhonationCmd::NoteOn { .. }));
        if had_note_on {
            self.voice_runtime.start_latch = StartLatch::None;
            self.motion_runtime.note_first_utterance();
        } else if matches!(
            self.voice_runtime.start_latch,
            StartLatch::WaitForGateExitThenOnset
        ) {
            if let Some(exit_tick) = gate_exit_tick
                && exit_tick < frame_end
            {
                self.voice_runtime.start_latch = StartLatch::WaitForOnset;
            }
        }
        if self.phonation_scratch.events.is_empty() {
            debug_assert!(
                !had_note_on,
                "NoteOn emitted without note specs (no note events)"
            );
            return;
        }
        let amp = self.compute_target_amp();
        if amp <= Self::AMP_EPS {
            debug_assert!(
                !had_note_on,
                "NoteOn emitted but amp invalid => no note specs"
            );
            self.phonation_scratch.events.clear();
            return;
        }
        let freq_hz = self.body.base_freq_hz();
        if !freq_hz.is_finite() || freq_hz <= 0.0 {
            debug_assert!(
                !had_note_on,
                "NoteOn emitted but freq invalid => no note specs"
            );
            self.phonation_scratch.events.clear();
            return;
        }
        let articulation = self.articulation_snapshot_for_render();
        let body = self.body_snapshot();
        let smoothing_tau_sec = 0.0;
        for event in self.phonation_scratch.events.drain(..) {
            out.notes.push(PhonationNoteSpec {
                note_id: event.note_id,
                onset: event.onset_tick,
                hold_ticks: None,
                freq_hz,
                amp,
                smoothing_tau_sec,
                body: body.clone(),
                articulation: articulation.clone(),
            });
        }
        debug_assert!(
            !had_note_on || !out.notes.is_empty(),
            "NoteOn emitted without note specs"
        );
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

    pub(crate) fn take_birth_intent(
        &mut self,
        tb: &Timebase,
        now: Tick,
        intent_id: u64,
        duration_sec: f32,
    ) -> Option<Intent> {
        if !self.birth_once_pending {
            return None;
        }
        if matches!(self.behavior.voice.on_spawn, VoiceOnSpawn::Disabled) {
            self.birth_once_pending = false;
            return None;
        }

        let onset = tb.frame_start_tick(self.birth_frame);
        if onset > now {
            return None;
        }
        let mut duration = tb.sec_to_tick(duration_sec.max(0.0));
        if duration == 0 && duration_sec > 0.0 {
            duration = 1;
        }
        if duration == 0 {
            self.birth_once_pending = false;
            return None;
        }

        let amp = self.compute_target_amp();
        if amp <= Self::AMP_EPS {
            self.birth_once_pending = false;
            return None;
        }
        let freq_hz = self.body.base_freq_hz();
        if !freq_hz.is_finite() || freq_hz <= 0.0 {
            self.birth_once_pending = false;
            return None;
        }

        self.birth_once_pending = false;
        Some(Intent {
            source_id: self.id,
            intent_id,
            onset,
            duration,
            freq_hz,
            amp,
            tag: self.metadata.tag.clone(),
            confidence: 1.0,
            body: Some(self.body_snapshot()),
            articulation: None,
        })
    }

    pub fn is_alive(&self) -> bool {
        self.articulation.is_alive() && self.release_gain > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::modulation::RhythmBand;
    use crate::life::phonation_engine::{
        CandidatePoint, ClockSource, FixedGateConnect, PhonationClock, PhonationCmd,
        PhonationInterval, PhonationKick,
    };

    struct AlwaysInterval;

    impl PhonationInterval for AlwaysInterval {
        fn on_candidate(
            &mut self,
            _c: &crate::life::phonation_engine::IntervalInput,
            _state: &crate::life::phonation_engine::CoreState,
        ) -> Option<PhonationKick> {
            Some(PhonationKick::Planned { strength: 1.0 })
        }
    }

    struct TestClock {
        points: Vec<CandidatePoint>,
    }

    impl PhonationClock for TestClock {
        fn gather_candidates(
            &mut self,
            ctx: &crate::life::phonation_engine::CoreTickCtx,
            out: &mut Vec<CandidatePoint>,
        ) {
            out.extend(
                self.points
                    .iter()
                    .cloned()
                    .filter(|point| point.tick >= ctx.now_tick && point.tick < ctx.frame_end),
            );
        }
    }

    #[test]
    fn next_rhythm_blocks_onsets_and_note_offs_before_gate_exit() {
        let mut life = crate::life::scenario::LifeConfig::default();
        life.behavior.voice.on_spawn = VoiceOnSpawn::NextRhythm;
        life.behavior.voice.control = VoiceControl::Autonomous;
        let cfg = crate::life::scenario::IndividualConfig {
            freq: 440.0,
            amp: 0.5,
            life,
            tag: None,
        };
        let meta = AgentMetadata {
            id: 1,
            tag: None,
            group_idx: 0,
            member_idx: 0,
        };
        let mut agent = cfg.spawn(1, 0, meta, 40.0, 0);
        agent.phonation_engine.interval = Box::new(AlwaysInterval);
        agent.phonation_engine.connect = Box::new(FixedGateConnect::new(1));
        agent.phonation_engine.clock = Box::new(TestClock {
            points: vec![
                CandidatePoint {
                    tick: 4,
                    gate: 0,
                    theta_pos: 0.5,
                    phase_in_gate: 0.5,
                    sources: vec![ClockSource::Subdivision { n: 2 }],
                },
                CandidatePoint {
                    tick: 10,
                    gate: 1,
                    theta_pos: 1.0,
                    phase_in_gate: 0.0,
                    sources: vec![ClockSource::GateBoundary],
                },
                CandidatePoint {
                    tick: 20,
                    gate: 2,
                    theta_pos: 2.0,
                    phase_in_gate: 0.0,
                    sources: vec![ClockSource::GateBoundary],
                },
            ],
        });
        let tb = Timebase { fs: 40.0, hop: 8 };
        let mut rhythms = NeuralRhythms {
            theta: RhythmBand {
                phase: -std::f32::consts::FRAC_PI_2,
                freq_hz: 1.0,
                mag: 1.0,
                alpha: 1.0,
                beta: 0.0,
            },
            env_open: 1.0,
            ..Default::default()
        };
        let batch0 = agent.tick_phonation(&tb, 0, &rhythms, None, 0.0);
        assert!(
            !batch0
                .cmds
                .iter()
                .any(|cmd| matches!(cmd, PhonationCmd::NoteOn { .. }))
        );
        assert!(
            !batch0
                .cmds
                .iter()
                .any(|cmd| matches!(cmd, PhonationCmd::NoteOff { .. }))
        );

        rhythms.advance_in_place(tb.hop as f32 / tb.fs);
        let batch1 = agent.tick_phonation(&tb, 8, &rhythms, None, 0.0);
        let onsets: Vec<Tick> = batch1.notes.iter().map(|note| note.onset).collect();
        assert_eq!(onsets, vec![10]);
        assert!(
            !batch1
                .cmds
                .iter()
                .any(|cmd| matches!(cmd, PhonationCmd::NoteOff { .. }))
        );
    }

    #[test]
    fn next_rhythm_falls_back_when_gate_exit_is_unavailable() {
        let mut life = crate::life::scenario::LifeConfig::default();
        life.behavior.voice.on_spawn = VoiceOnSpawn::NextRhythm;
        life.behavior.voice.control = VoiceControl::Autonomous;
        let cfg = crate::life::scenario::IndividualConfig {
            freq: 440.0,
            amp: 0.5,
            life,
            tag: None,
        };
        let meta = AgentMetadata {
            id: 2,
            tag: None,
            group_idx: 0,
            member_idx: 0,
        };
        let mut agent = cfg.spawn(2, 0, meta, 40.0, 0);
        agent.phonation_engine.interval = Box::new(AlwaysInterval);
        agent.phonation_engine.connect = Box::new(FixedGateConnect::new(1));
        agent.phonation_engine.clock = Box::new(TestClock {
            points: vec![CandidatePoint {
                tick: 4,
                gate: 0,
                theta_pos: 0.5,
                phase_in_gate: 0.5,
                sources: vec![ClockSource::Subdivision { n: 2 }],
            }],
        });
        let tb = Timebase { fs: 40.0, hop: 8 };
        let rhythms = NeuralRhythms {
            theta: RhythmBand {
                phase: 1.0,
                freq_hz: 0.0,
                mag: 1.0,
                alpha: 1.0,
                beta: 0.0,
            },
            env_open: 1.0,
            ..Default::default()
        };
        let batch0 = agent.tick_phonation(&tb, 0, &rhythms, None, 0.0);
        assert!(
            batch0
                .cmds
                .iter()
                .any(|cmd| matches!(cmd, PhonationCmd::NoteOn { .. }))
        );
    }

    #[test]
    fn motion_after_first_utterance_enables_on_note_on() {
        let mut life = crate::life::scenario::LifeConfig::default();
        life.behavior.voice.on_spawn = VoiceOnSpawn::Immediate;
        life.behavior.voice.control = VoiceControl::Autonomous;
        life.behavior.autonomy.motion = MotionAutonomy::AfterFirstUtterance;
        let cfg = crate::life::scenario::IndividualConfig {
            freq: 440.0,
            amp: 0.5,
            life,
            tag: None,
        };
        let meta = AgentMetadata {
            id: 3,
            tag: None,
            group_idx: 0,
            member_idx: 0,
        };
        let mut agent = cfg.spawn(3, 0, meta, 40.0, 0);
        agent.phonation_engine.interval = Box::new(AlwaysInterval);
        agent.phonation_engine.connect = Box::new(FixedGateConnect::new(1));
        agent.phonation_engine.clock = Box::new(TestClock {
            points: vec![CandidatePoint {
                tick: 4,
                gate: 0,
                theta_pos: 0.5,
                phase_in_gate: 0.5,
                sources: vec![ClockSource::Subdivision { n: 2 }],
            }],
        });
        let tb = Timebase { fs: 40.0, hop: 8 };
        let rhythms = NeuralRhythms {
            theta: RhythmBand {
                phase: 0.0,
                freq_hz: 1.0,
                mag: 1.0,
                alpha: 1.0,
                beta: 0.0,
            },
            env_open: 1.0,
            ..Default::default()
        };
        assert!(!agent.motion_runtime.motion_enabled);
        let batch0 = agent.tick_phonation(&tb, 0, &rhythms, None, 0.0);
        assert!(
            batch0
                .cmds
                .iter()
                .any(|cmd| matches!(cmd, PhonationCmd::NoteOn { .. }))
        );
        assert!(agent.motion_runtime.motion_enabled);
    }

    #[test]
    fn voice_disabled_suppresses_note_on_across_hops() {
        let mut life = crate::life::scenario::LifeConfig::default();
        life.behavior.voice.on_spawn = VoiceOnSpawn::Disabled;
        life.behavior.voice.control = VoiceControl::Autonomous;
        let cfg = crate::life::scenario::IndividualConfig {
            freq: 440.0,
            amp: 0.5,
            life,
            tag: None,
        };
        let meta = AgentMetadata {
            id: 4,
            tag: None,
            group_idx: 0,
            member_idx: 0,
        };
        let mut agent = cfg.spawn(4, 0, meta, 40.0, 0);
        agent.phonation_engine.interval = Box::new(AlwaysInterval);
        agent.phonation_engine.connect = Box::new(FixedGateConnect::new(1));
        agent.phonation_engine.clock = Box::new(TestClock {
            points: vec![
                CandidatePoint {
                    tick: 4,
                    gate: 0,
                    theta_pos: 0.5,
                    phase_in_gate: 0.5,
                    sources: vec![ClockSource::Subdivision { n: 2 }],
                },
                CandidatePoint {
                    tick: 10,
                    gate: 1,
                    theta_pos: 1.0,
                    phase_in_gate: 0.0,
                    sources: vec![ClockSource::GateBoundary],
                },
            ],
        });
        let tb = Timebase { fs: 40.0, hop: 8 };
        let mut rhythms = NeuralRhythms {
            theta: RhythmBand {
                phase: 0.0,
                freq_hz: 1.0,
                mag: 1.0,
                alpha: 1.0,
                beta: 0.0,
            },
            env_open: 1.0,
            ..Default::default()
        };
        let batch0 = agent.tick_phonation(&tb, 0, &rhythms, None, 0.0);
        assert!(
            !batch0
                .cmds
                .iter()
                .any(|cmd| matches!(cmd, PhonationCmd::NoteOn { .. }))
        );
        rhythms.advance_in_place(tb.hop as f32 / tb.fs);
        let batch1 = agent.tick_phonation(&tb, 8, &rhythms, None, 0.0);
        assert!(
            !batch1
                .cmds
                .iter()
                .any(|cmd| matches!(cmd, PhonationCmd::NoteOn { .. }))
        );
    }
}
