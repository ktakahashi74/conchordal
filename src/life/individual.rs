use crate::core::landscape::Landscape;
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::control::{
    AgentControl, AgentPatch, BodyControl, BodyMethod, PerceptualControl, PhonationControl,
    PhonationType, PitchConstraintMode, PitchControl, merge_json, remove_json_path,
};
use crate::life::intent::{BodySnapshot, Intent};
use crate::life::lifecycle::LifecycleConfig;
use crate::life::perceptual::PerceptualConfig;
use crate::life::phonation_engine::{
    CoreState, CoreTickCtx, NoteId, OnsetEvent, PhonationCmd, PhonationEngine, PhonationNoteEvent,
};
use crate::life::scenario::{
    ArticulationCoreConfig, HarmonicMode, PhonationClockConfig, PhonationConfig,
    PhonationConnectConfig, PhonationIntervalConfig, PhonationMode, PitchCoreConfig, SocialConfig,
    SoundBodyConfig, SubThetaModConfig, TimbreGenotype,
};
use crate::life::social_density::SocialDensityTrace;
use rand::SeedableRng;

#[path = "articulation_core.rs"]
pub mod articulation_core;
#[path = "pitch_controller.rs"]
pub mod pitch_controller;
#[path = "pitch_core.rs"]
pub mod pitch_core;
#[path = "sound_body.rs"]
pub mod sound_body;

pub use articulation_core::{
    AnyArticulationCore, ArticulationCore, ArticulationSignal, ArticulationState,
    ArticulationWrapper, DroneCore, KuramotoCore, PinkNoise, PlannedGate, PlannedPitch,
    Sensitivity, SequencedCore,
};
pub use pitch_controller::PitchController;
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
    pub(crate) seed: u64,
    pub id: u64,
    pub metadata: AgentMetadata,
    pub base_control: AgentControl,
    pub override_json: serde_json::Value,
    pub effective_control: AgentControl,
    pub articulation: ArticulationWrapper,
    pub(crate) pitch_ctl: PitchController,
    pub phonation_engine: PhonationEngine,
    pub phonation_social: SocialConfig,
    pub body: AnySoundBody,
    pub last_signal: ArticulationSignal,
    pub(crate) release_gain: f32,
    pub(crate) release_sec: f32,
    pub(crate) release_pending: bool,
    pub(crate) remove_pending: bool,
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

#[derive(Clone, Copy, Debug)]
struct BodyRuntime {
    amp: f32,
    brightness: f32,
    inharmonic: f32,
    width: f32,
    motion: f32,
}

impl BodyRuntime {
    fn from_control(body: &BodyControl) -> Self {
        let timbre = &body.timbre;
        Self {
            amp: body.amp,
            brightness: timbre.brightness,
            inharmonic: timbre.inharmonic,
            width: timbre.width,
            motion: timbre.motion,
        }
    }
}

impl Individual {
    const AMP_EPS: f32 = 1e-6;

    pub fn spawn_from_control(
        control: AgentControl,
        assigned_id: u64,
        start_frame: u64,
        mut metadata: AgentMetadata,
        fs: f32,
        seed_offset: u64,
    ) -> Self {
        metadata.id = assigned_id;
        let seed = seed_offset ^ assigned_id ^ start_frame.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);

        let mut effective_control = control.clone();
        let constraint = &effective_control.pitch.constraint;
        if matches!(constraint.mode, PitchConstraintMode::Lock)
            && let Some(freq) = constraint.freq_hz
        {
            effective_control.pitch.center_hz = freq.max(1.0);
        }

        let target_freq = effective_control.pitch.center_hz.max(1.0);
        let target_pitch_log2 = target_freq.log2();
        let integration_window = 2.0 + 10.0 / target_freq.max(1.0);

        let articulation_config = ArticulationCoreConfig::default();
        let core =
            AnyArticulationCore::from_config(&articulation_config, fs, assigned_id, &mut rng);

        let phonation_config = phonation_config_from_control(&effective_control.phonation);
        let phonation_engine = PhonationEngine::from_config(&phonation_config, seed);
        let phonation_social = phonation_config.social;

        let pitch_config = pitch_core_config_from_control(&effective_control.pitch);
        let pitch = AnyPitchCore::from_config(&pitch_config, target_pitch_log2, &mut rng);
        let perceptual_config = perceptual_config_from_control(&effective_control.perceptual);
        let perceptual =
            crate::life::perceptual::PerceptualContext::from_config(&perceptual_config, 0);

        let body_config = sound_body_config_from_control(&effective_control.body);
        let body = AnySoundBody::from_config(
            &body_config,
            target_freq,
            effective_control.body.amp,
            &mut rng,
        );

        let pitch_ctl = PitchController::new(
            pitch,
            perceptual,
            target_pitch_log2,
            integration_window,
            rng,
        );

        let (articulation_core, lifecycle_label, default_by_articulation, breath_gain_init) =
            match &articulation_config {
                ArticulationCoreConfig::Entrain {
                    lifecycle,
                    breath_gain_init,
                    ..
                } => {
                    let life_label = match lifecycle {
                        LifecycleConfig::Decay { .. } => "decay",
                        LifecycleConfig::Sustain { .. } => "sustain",
                    };
                    ("entrain", life_label, 1.0, *breath_gain_init)
                }
                ArticulationCoreConfig::Seq {
                    breath_gain_init, ..
                } => ("seq", "none", 1.0, *breath_gain_init),
                ArticulationCoreConfig::Drone {
                    breath_gain_init, ..
                } => ("drone", "none", 0.0, *breath_gain_init),
            };
        let breath_gain = breath_gain_init
            .unwrap_or(default_by_articulation)
            .clamp(0.0, 1.0);
        tracing::debug!(
            target: "rhythm::spawn",
            id = assigned_id,
            tag = ?metadata.tag,
            articulation = articulation_core,
            lifecycle = lifecycle_label,
            breath_gain_init,
            breath_gain
        );

        let mut agent = Individual {
            seed,
            id: assigned_id,
            metadata,
            base_control: control,
            override_json: serde_json::Value::Object(serde_json::Map::new()),
            effective_control,
            articulation: ArticulationWrapper::new(core, breath_gain),
            pitch_ctl,
            phonation_engine,
            phonation_social,
            body,
            last_signal: Default::default(),
            release_gain: 1.0,
            release_sec: 0.03,
            release_pending: false,
            remove_pending: false,
            birth_once_pending: true,
            birth_frame: start_frame,
            birth_once_duration_sec: None,
            phonation_scratch: Default::default(),
        };
        agent.apply_body_runtime();
        agent.apply_perceptual_control();
        agent.apply_pitch_control();
        agent
    }

    fn apply_body_runtime(&mut self) {
        let runtime = BodyRuntime::from_control(&self.effective_control.body);
        self.body.set_amp(runtime.amp);
        match &mut self.body {
            AnySoundBody::Sine(_body) => {}
            AnySoundBody::Harmonic(body) => {
                body.genotype.brightness = runtime.brightness;
                body.genotype.stiffness = runtime.inharmonic;
                body.genotype.unison = runtime.width;
                body.genotype.jitter = runtime.motion;
                body.genotype.vibrato_depth = runtime.motion * 0.02;
            }
        }
    }

    fn apply_perceptual_control(&mut self) {
        let params = perceptual_params_from_control(&self.effective_control.perceptual);
        let perceptual = self.pitch_ctl.perceptual_mut();
        perceptual.tau_fast = params.tau_fast;
        perceptual.tau_slow = params.tau_slow;
        perceptual.w_boredom = params.w_boredom;
        perceptual.w_familiarity = params.w_familiarity;
        perceptual.rho_self = params.rho_self;
        perceptual.boredom_gamma = params.boredom_gamma;
        perceptual.self_smoothing_radius = params.self_smoothing_radius;
        perceptual.silence_mass_epsilon = params.silence_mass_epsilon;
        self.pitch_ctl.set_perceptual_enabled(params.enabled);
    }

    fn apply_pitch_control(&mut self) {
        let pitch = &self.effective_control.pitch;
        let center_log2 = pitch.center_hz.max(1.0).log2();
        let gravity = tessitura_gravity_from_control(pitch.gravity);
        let core = self.pitch_ctl.core_mut();
        core.set_tessitura_center(center_log2);
        core.set_tessitura_gravity(gravity);
        core.set_exploration(pitch.exploration);
        core.set_persistence(pitch.persistence);
    }

    fn apply_phonation_control(&mut self) {
        let config = phonation_config_from_control(&self.effective_control.phonation);
        self.phonation_engine = PhonationEngine::from_config(&config, self.seed);
        self.phonation_social = config.social;
    }

    fn apply_effective_control(&mut self, control: AgentControl) {
        self.effective_control = control;
        self.apply_body_runtime();
        self.apply_perceptual_control();
        self.apply_pitch_control();
        self.apply_phonation_control();
    }

    pub fn should_retain(&self) -> bool {
        if self.remove_pending && self.release_gain <= 0.0 {
            return false;
        }
        if self.remove_pending {
            return self.is_alive();
        }
        self.is_alive() || self.phonation_engine.has_active_notes()
    }
    pub fn metadata(&self) -> &AgentMetadata {
        &self.metadata
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn target_pitch_log2(&self) -> f32 {
        self.pitch_ctl.target_pitch_log2()
    }

    pub fn integration_window(&self) -> f32 {
        self.pitch_ctl.integration_window()
    }

    pub fn release_gain(&self) -> f32 {
        self.release_gain
    }

    pub(crate) fn birth_once_duration_sec(&self) -> Option<f32> {
        self.birth_once_duration_sec
    }

    pub fn apply_control_patch(&mut self, patch: serde_json::Value) -> Result<(), String> {
        let patch_struct: AgentPatch =
            serde_json::from_value(patch.clone()).map_err(|e| format!("parse AgentPatch: {e}"))?;
        if patch_struct.contains_type_switch() {
            return Err(
                "set() cannot change body.method or phonation.type; use spawn() for type selection"
                    .to_string(),
            );
        }
        let merged_override = merge_json(self.override_json.clone(), patch);
        let base_json = self.base_control.to_json()?;
        let effective_json = merge_json(base_json, merged_override.clone());
        let effective = AgentControl::from_json(effective_json)?;
        self.override_json = merged_override;
        self.apply_effective_control(effective);
        Ok(())
    }

    pub fn apply_unset_path(&mut self, path: &str) -> Result<bool, String> {
        let mut override_json = self.override_json.clone();
        let removed = remove_json_path(&mut override_json, path);
        if !removed {
            return Ok(false);
        }
        let base_json = self.base_control.to_json()?;
        let effective_json = merge_json(base_json, override_json.clone());
        let effective = AgentControl::from_json(effective_json)?;
        self.override_json = override_json;
        self.apply_effective_control(effective);
        Ok(true)
    }

    #[cfg(test)]
    pub(crate) fn set_accumulated_time_for_test(&mut self, value: f32) {
        self.pitch_ctl.set_accumulated_time_for_test(value);
    }

    #[cfg(test)]
    pub(crate) fn accumulated_time_for_test(&self) -> f32 {
        self.pitch_ctl.accumulated_time_for_test()
    }

    #[cfg(test)]
    pub(crate) fn set_theta_phase_state_for_test(&mut self, last_phase: f32, initialized: bool) {
        self.pitch_ctl
            .set_theta_phase_state_for_test(last_phase, initialized);
    }

    pub fn force_set_pitch_log2(&mut self, log_freq: f32) {
        let log_freq = log_freq.max(0.0);
        self.body.set_pitch_log2(log_freq);
        self.articulation.set_gate(1.0);
        self.pitch_ctl.force_set_target_pitch_log2(log_freq);
    }

    /// Update pitch targets at control rate (hop-sized steps).
    pub fn update_pitch_target(
        &mut self,
        rhythms: &NeuralRhythms,
        dt_sec: f32,
        landscape: &Landscape,
    ) {
        let current_freq = self.body.base_freq_hz();
        self.pitch_ctl.update_pitch_target(
            current_freq,
            rhythms,
            dt_sec,
            landscape,
            &self.effective_control.pitch,
        );
    }

    /// Control-rate entry point for pitch + articulation updates.
    pub fn tick_control(
        &mut self,
        dt_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        global_coupling: f32,
    ) -> ArticulationSignal {
        self.apply_body_runtime();
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
        self.update_pitch_target(rhythms, dt_sec, landscape);
        self.update_articulation_autonomous(dt_sec, rhythms);
        self.tick_articulation_lifecycle(dt_sec, rhythms, landscape, global_coupling)
    }

    pub fn update_articulation_autonomous(&mut self, dt_sec: f32, rhythms: &NeuralRhythms) {
        let current_freq = self.body.base_freq_hz().max(1.0);
        let current_pitch_log2 = current_freq.log2();
        let target_pitch_log2 = self.target_pitch_log2();
        let planned = PlannedPitch {
            target_pitch_log2,
            jump_cents_abs: 1200.0 * (target_pitch_log2 - current_pitch_log2).abs(),
            salience: self.pitch_ctl.last_target_salience(),
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

    pub fn start_remove_fade(&mut self, fade_sec: f32) {
        if self.remove_pending {
            return;
        }
        self.remove_pending = true;
        self.release_pending = true;
        self.release_sec = fade_sec.max(1e-4);
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
        if matches!(self.effective_control.phonation.r#type, PhonationType::None) {
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
            is_alive: self.is_alive() && !self.remove_pending,
        };
        self.phonation_engine.tick(
            &ctx,
            &state,
            social,
            social_coupling,
            None,
            &mut out.cmds,
            &mut self.phonation_scratch.events,
            &mut out.onsets,
        );
        let had_note_on = out
            .cmds
            .iter()
            .any(|cmd| matches!(cmd, PhonationCmd::NoteOn { .. }));
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
        if self.remove_pending
            || matches!(self.effective_control.phonation.r#type, PhonationType::None)
        {
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
        if self.remove_pending && self.release_gain <= 0.0 {
            return false;
        }
        self.articulation.is_alive() && self.release_gain > 0.0
    }
}

#[derive(Clone, Copy, Debug)]
struct PerceptualParams {
    enabled: bool,
    tau_fast: f32,
    tau_slow: f32,
    w_boredom: f32,
    w_familiarity: f32,
    rho_self: f32,
    boredom_gamma: f32,
    self_smoothing_radius: usize,
    silence_mass_epsilon: f32,
}

fn perceptual_params_from_control(control: &PerceptualControl) -> PerceptualParams {
    let adaptation = control.adaptation.clamp(0.0, 1.0);
    let tau_fast = 0.1 + (1.0 - adaptation) * 0.8;
    let tau_slow = 5.0 + (1.0 - adaptation) * 30.0;
    let (w_boredom, w_familiarity, rho_self) = if control.enabled {
        (
            control.novelty_bias,
            0.2,
            control.self_focus.clamp(0.0, 1.0),
        )
    } else {
        (0.0, 0.0, 0.0)
    };
    PerceptualParams {
        enabled: control.enabled,
        tau_fast,
        tau_slow,
        w_boredom,
        w_familiarity,
        rho_self,
        boredom_gamma: 0.5,
        self_smoothing_radius: 1,
        silence_mass_epsilon: 1e-6,
    }
}

fn tessitura_gravity_from_control(gravity: f32) -> f32 {
    gravity.clamp(0.0, 1.0) * 0.2
}

fn pitch_core_config_from_control(pitch: &PitchControl) -> PitchCoreConfig {
    PitchCoreConfig::PitchHillClimb {
        neighbor_step_cents: None,
        tessitura_gravity: Some(tessitura_gravity_from_control(pitch.gravity)),
        improvement_threshold: None,
        exploration: Some(pitch.exploration),
        persistence: Some(pitch.persistence),
    }
}

fn perceptual_config_from_control(control: &PerceptualControl) -> PerceptualConfig {
    let params = perceptual_params_from_control(control);
    PerceptualConfig {
        tau_fast: Some(params.tau_fast),
        tau_slow: Some(params.tau_slow),
        w_boredom: Some(params.w_boredom),
        w_familiarity: Some(params.w_familiarity),
        rho_self: Some(params.rho_self),
        boredom_gamma: Some(params.boredom_gamma),
        self_smoothing_radius: Some(params.self_smoothing_radius),
        silence_mass_epsilon: Some(params.silence_mass_epsilon),
    }
}

fn sound_body_config_from_control(body: &BodyControl) -> SoundBodyConfig {
    match body.method {
        BodyMethod::Sine => SoundBodyConfig::Sine { phase: None },
        BodyMethod::Harmonic => {
            let timbre = &body.timbre;
            let genotype = TimbreGenotype {
                mode: HarmonicMode::Harmonic,
                stiffness: timbre.inharmonic,
                brightness: timbre.brightness,
                comb: 0.0,
                damping: 0.5,
                vibrato_rate: 5.0,
                vibrato_depth: timbre.motion * 0.02,
                jitter: timbre.motion,
                unison: timbre.width,
            };
            SoundBodyConfig::Harmonic {
                genotype,
                partials: None,
            }
        }
    }
}

fn phonation_config_from_control(control: &PhonationControl) -> PhonationConfig {
    // Hold ignores density/sync/legato; it is purely lifecycle-driven.
    if matches!(control.r#type, PhonationType::Hold) {
        let social = SocialConfig {
            coupling: control.sociality.clamp(0.0, 1.0),
            bin_ticks: 0,
            smooth: 0.0,
        };
        return PhonationConfig {
            mode: PhonationMode::Hold,
            interval: PhonationIntervalConfig::None,
            connect: PhonationConnectConfig::FixedGate { length_gates: 1 },
            clock: PhonationClockConfig::ThetaGate,
            sub_theta_mod: SubThetaModConfig::None,
            social,
        };
    }
    let density = control.density.clamp(0.0, 1.0);
    let rate = 0.5 + density * 3.5;
    let interval = match control.r#type {
        PhonationType::None => PhonationIntervalConfig::None,
        _ => PhonationIntervalConfig::Accumulator {
            rate,
            refractory: 1,
        },
    };
    let legato = control.legato.clamp(0.0, 1.0);
    let length_gates = (1.0 + legato * 8.0).round().max(1.0) as u32;
    let connect = match control.r#type {
        PhonationType::Field => PhonationConnectConfig::Field {
            hold_min_theta: 0.1 + legato * 0.2,
            hold_max_theta: 0.6 + legato * 0.4,
            curve_k: 2.0,
            curve_x0: 0.5,
            drop_gain: (1.0 - legato).clamp(0.0, 1.0),
        },
        _ => PhonationConnectConfig::FixedGate { length_gates },
    };
    let sub_theta_mod = if control.sync > 0.0 {
        SubThetaModConfig::Cosine {
            n: 1,
            depth: control.sync.clamp(0.0, 1.0),
            phase0: 0.0,
        }
    } else {
        SubThetaModConfig::None
    };
    let social = SocialConfig {
        coupling: control.sociality.clamp(0.0, 1.0),
        bin_ticks: 0,
        smooth: 0.0,
    };
    PhonationConfig {
        mode: PhonationMode::Gated,
        interval,
        connect,
        clock: PhonationClockConfig::ThetaGate,
        sub_theta_mod,
        social,
    }
}
