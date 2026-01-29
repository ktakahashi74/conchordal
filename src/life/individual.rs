use crate::core::landscape::Landscape;
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::control::{
    AgentControl, BodyControl, BodyMethod, ControlUpdate, PhonationType, PitchMode,
};
use crate::life::control_adapters::{
    perceptual_config_from_control, perceptual_params_from_control, phonation_config_from_control,
    pitch_core_config_from_control, sound_body_config_from_control, tessitura_gravity_from_control,
};
use crate::life::lifecycle::LifecycleConfig;
use crate::life::phonation_engine::{
    CoreState, CoreTickCtx, NoteId, OnsetEvent, PhonationCmd, PhonationEngine, PhonationNoteOn,
};
use crate::life::scenario::ArticulationCoreConfig;
use crate::life::social_density::SocialDensityTrace;
use crate::life::sound::BodySnapshot;
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
    pub group_id: u64,
    pub member_idx: usize,
}

#[derive(Debug)]
pub struct Individual {
    pub id: u64,
    pub metadata: AgentMetadata,
    fixed_body_method: BodyMethod,
    fixed_phonation_type: PhonationType,
    pub base_control: AgentControl,
    pub effective_control: AgentControl,
    pub articulation: ArticulationWrapper,
    pub(crate) pitch_ctl: PitchController,
    pub phonation_engine: PhonationEngine,
    pub(crate) phonation_coupling: f32,
    pub body: AnySoundBody,
    pub last_signal: ArticulationSignal,
    last_consonance01: f32,
    pub(crate) release_gain: f32,
    pub(crate) release_sec: f32,
    pub(crate) release_pending: bool,
    pub(crate) remove_pending: bool,
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
    events: Vec<PhonationNoteOn>,
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

#[derive(Clone, Copy, Debug, Default)]
struct Dirty {
    body: bool,
    pitch: bool,
    phonation: bool,
    perceptual: bool,
}

impl Dirty {
    fn from_update(update: &ControlUpdate) -> Self {
        let body = update.amp.is_some()
            || update.timbre_brightness.is_some()
            || update.timbre_inharmonic.is_some()
            || update.timbre_width.is_some()
            || update.timbre_motion.is_some();
        let pitch = update.freq.is_some();
        Self {
            body,
            pitch,
            phonation: false,
            perceptual: false,
        }
    }
}

impl Individual {
    const AMP_EPS: f32 = 1e-6;

    pub fn spawn_from_control(
        control: AgentControl,
        articulation_config: ArticulationCoreConfig,
        assigned_id: u64,
        start_frame: u64,
        metadata: AgentMetadata,
        fs: f32,
        seed_offset: u64,
    ) -> Self {
        let seed = seed_offset ^ assigned_id ^ start_frame.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);

        let effective_control = control.clone();
        let target_freq = effective_control.pitch.freq.max(1.0);
        let target_pitch_log2 = target_freq.log2();
        let integration_window = PitchController::integration_window_for_freq(target_freq);

        let core =
            AnyArticulationCore::from_config(&articulation_config, fs, assigned_id, &mut rng);

        let phonation_config = phonation_config_from_control(&effective_control.phonation);
        let phonation_engine = PhonationEngine::from_config(&phonation_config, seed);
        let phonation_coupling = phonation_config.social.coupling;

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

        let mut pitch_ctl = PitchController::new(
            pitch,
            perceptual,
            target_pitch_log2,
            integration_window,
            rng,
        );
        pitch_ctl.set_perceptual_enabled(effective_control.perceptual.enabled);

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
            group_id = metadata.group_id,
            member_idx = metadata.member_idx,
            articulation = articulation_core,
            lifecycle = lifecycle_label,
            breath_gain_init,
            breath_gain
        );

        Individual {
            id: assigned_id,
            metadata,
            fixed_body_method: effective_control.body.method,
            fixed_phonation_type: effective_control.phonation.r#type,
            base_control: control,
            effective_control,
            articulation: ArticulationWrapper::new(core, breath_gain),
            pitch_ctl,
            phonation_engine,
            phonation_coupling,
            body,
            last_signal: Default::default(),
            last_consonance01: 0.0,
            release_gain: 1.0,
            release_sec: 0.03,
            release_pending: false,
            remove_pending: false,
            phonation_scratch: Default::default(),
        }
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
        let center_log2 = pitch.freq.max(1.0).log2();
        let gravity = tessitura_gravity_from_control(pitch.gravity);
        let core = self.pitch_ctl.core_mut();
        core.set_tessitura_center(center_log2);
        core.set_tessitura_gravity(gravity);
        core.set_exploration(pitch.exploration);
        core.set_persistence(pitch.persistence);
    }

    fn apply_phonation_control(&mut self) {
        let config = phonation_config_from_control(&self.effective_control.phonation);
        self.phonation_engine.update_from_config(&config);
        self.phonation_coupling = config.social.coupling;
    }

    fn apply_effective_control(&mut self, control: AgentControl, dirty: Dirty) {
        self.effective_control = control;
        if dirty.body {
            self.apply_body_runtime();
        }
        if dirty.perceptual {
            self.apply_perceptual_control();
        }
        if dirty.pitch {
            self.apply_pitch_control();
        }
        if dirty.phonation {
            self.apply_phonation_control();
        }
    }

    fn ensure_fixed_kinds(&self, control: &AgentControl) -> Result<(), String> {
        if control.body.method != self.fixed_body_method
            || control.phonation.r#type != self.fixed_phonation_type
        {
            return Err(
                "update cannot change body.method or phonation.type; use spawn() for type selection"
                    .to_string(),
            );
        }
        Ok(())
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

    pub fn apply_update(&mut self, update: &ControlUpdate) -> Result<(), String> {
        if update.is_empty() {
            return Ok(());
        }
        let mut control = self.effective_control.clone();
        if let Some(amp) = update.amp {
            control.body.amp = amp.clamp(0.0, 1.0);
        }
        if let Some(freq) = update.freq {
            control.pitch.freq = freq.clamp(1.0, 20_000.0);
            control.pitch.mode = PitchMode::Lock;
        }
        if let Some(brightness) = update.timbre_brightness {
            control.body.timbre.brightness = brightness.clamp(0.0, 1.0);
        }
        if let Some(inharmonic) = update.timbre_inharmonic {
            control.body.timbre.inharmonic = inharmonic.clamp(0.0, 1.0);
        }
        if let Some(width) = update.timbre_width {
            control.body.timbre.width = width.clamp(0.0, 1.0);
        }
        if let Some(motion) = update.timbre_motion {
            control.body.timbre.motion = motion.clamp(0.0, 1.0);
        }
        self.ensure_fixed_kinds(&control)?;
        let dirty = Dirty::from_update(update);
        self.base_control = control.clone();
        self.apply_effective_control(control, dirty);
        Ok(())
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
        if matches!(self.effective_control.pitch.mode, PitchMode::Lock) {
            self.articulation.set_gate(1.0);
            self.body.set_pitch_log2(planned.target_pitch_log2);
            return;
        }
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
        self.last_consonance01 = consonance;
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
        extra_gate_gain: f32,
    ) -> PhonationBatch {
        let mut batch = PhonationBatch::default();
        self.tick_phonation_into(
            tb,
            now,
            rhythms,
            social,
            social_coupling,
            extra_gate_gain,
            &mut batch,
        );
        batch
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tick_phonation_into(
        &mut self,
        tb: &Timebase,
        now: Tick,
        rhythms: &NeuralRhythms,
        social: Option<&SocialDensityTrace>,
        social_coupling: f32,
        extra_gate_gain: f32,
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
            extra_gate_gain,
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

    pub(crate) fn body_snapshot(&self) -> BodySnapshot {
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

    pub fn last_consonance01(&self) -> f32 {
        self.last_consonance01
    }

    pub fn is_alive(&self) -> bool {
        if self.remove_pending && self.release_gain <= 0.0 {
            return false;
        }
        self.articulation.is_alive() && self.release_gain > 0.0
    }
}
