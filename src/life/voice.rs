use crate::core::landscape::{Landscape, LandscapeFrame};
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::control::{
    BodyControl, BodyMethod, ControlUpdate, PitchApplyMode, PitchMode, VoiceControl,
};
use crate::life::control_adapters::{
    adaptation_config_from_control, pitch_core_config_from_control, tessitura_gravity_from_control,
};
use crate::life::lifecycle::LifecycleConfig;
use crate::life::phonation_engine::{
    CoreState, CoreTickCtx, OnsetEvent, PhonationEngine, ToneCmd, ToneId, ToneOnEvent, ToneUpdate,
};
use crate::life::scenario::WhenSpec;
use crate::life::scenario::{ArticulationCoreConfig, PhonationMode};
use crate::life::social_density::SocialDensityTrace;
use crate::life::sound::{BodySnapshot, RenderModulatorSpec};
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
pub use pitch_core::{
    AnyPitchCore, PitchCore, PitchHillClimbPitchCore, PitchPeakSamplerCore, TargetProposal,
};
pub use sound_body::{AnySoundBody, HarmonicBody, SineBody, SoundBody};

#[derive(Debug, Clone, Default)]
pub struct VoiceMetadata {
    pub group_id: u64,
    pub member_idx: usize,
    pub generation: u32,
    pub parent_id: Option<u64>,
}

#[derive(Debug)]
pub struct Voice {
    pub id: u64,
    pub metadata: VoiceMetadata,
    fixed_body_method: BodyMethod,
    fixed_phonation_when: std::mem::Discriminant<WhenSpec>,
    pub base_control: VoiceControl,
    pub effective_control: VoiceControl,
    pub articulation: ArticulationWrapper,
    pub(crate) pitch_ctl: PitchController,
    pub phonation_engine: PhonationEngine,
    pub(crate) social_coupling: f32,
    pub body: AnySoundBody,
    pub last_signal: ArticulationSignal,
    pub(crate) release_gain: f32,
    pub(crate) release_sec: f32,
    pub(crate) release_pending: bool,
    pub(crate) remove_pending: bool,
    pub(crate) phonation_scratch: PhonationScratch,
    active_render_notes: Vec<TrackedRenderNote>,
    pub(crate) life_accumulator: Option<super::telemetry::LifeAccumulator>,
    voice_adsr: Option<super::sound::ToneAdsr>,
}

#[derive(Clone, Debug)]
pub struct ToneSpec {
    pub tone_id: ToneId,
    pub onset: Tick,
    pub hold_ticks: Option<Tick>,
    pub freq_hz: f32,
    pub amp: f32,
    pub smoothing_tau_sec: f32,
    pub body: BodySnapshot,
    pub render_modulator: RenderModulatorSpec,
    pub adsr: Option<super::sound::ToneAdsr>,
}

#[derive(Debug, Default)]
pub(crate) struct PhonationScratch {
    events: Vec<ToneOnEvent>,
}

#[derive(Clone, Copy, Debug)]
struct TrackedRenderNote {
    tone_id: ToneId,
    last_target_amp: f32,
    last_target_freq_hz: f32,
    last_continuous_drive: f32,
}

#[derive(Clone, Debug, Default)]
pub struct PhonationBatch {
    pub source_id: u64,
    pub cmds: Vec<ToneCmd>,
    pub tones: Vec<ToneSpec>,
    pub onsets: Vec<OnsetEvent>,
}

impl PhonationBatch {
    pub(crate) fn clear(&mut self) {
        self.cmds.clear();
        self.tones.clear();
        self.onsets.clear();
    }
}

#[derive(Clone, Copy, Debug)]
struct BodyRuntime {
    amp: f32,
    brightness: f32,
    inharmonic: f32,
    motion: f32,
    spread: f32,
    unison: usize,
}

impl BodyRuntime {
    fn from_control(body: &BodyControl) -> Self {
        let timbre = &body.timbre;
        Self {
            amp: body.amp,
            brightness: timbre.brightness,
            inharmonic: timbre.inharmonic,
            motion: timbre.motion,
            spread: timbre.spread,
            unison: timbre.unison,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Dirty {
    body: bool,
    pitch: bool,
    phonation: bool,
    adaptation: bool,
}

impl Dirty {
    fn from_update(update: &ControlUpdate) -> Self {
        let body = update.amp.is_some()
            || update.timbre_brightness.is_some()
            || update.timbre_inharmonic.is_some()
            || update.timbre_motion.is_some()
            || update.timbre_spread.is_some()
            || update.timbre_unison.is_some();
        let pitch = update.freq.is_some()
            || update.neighbor_step_cents.is_some()
            || update.tessitura_gravity.is_some()
            || update.landscape_weight.is_some()
            || update.exploration.is_some()
            || update.persistence.is_some()
            || update.crowding_strength.is_some()
            || update.crowding_sigma_cents.is_some()
            || update.crowding_sigma_from_roughness.is_some()
            || update.leave_self_out.is_some()
            || update.leave_self_out_mode.is_some()
            || update.anneal_temp.is_some()
            || update.move_cost_coeff.is_some()
            || update.move_cost_exp.is_some()
            || update.improvement_threshold.is_some()
            || update.proposal_interval_sec.is_some()
            || update.global_peak_count.is_some()
            || update.global_peak_min_sep_cents.is_some()
            || update.use_ratio_candidates.is_some()
            || update.ratio_candidate_count.is_some()
            || update.window_cents.is_some()
            || update.top_k.is_some()
            || update.temperature.is_some()
            || update.sigma_cents.is_some()
            || update.random_candidates.is_some()
            || update.move_cost_time_scale.is_some()
            || update.leave_self_out_harmonics.is_some()
            || update.pitch_apply_mode.is_some()
            || update.pitch_glide_tau_sec.is_some();
        Self {
            body,
            pitch,
            phonation: false,
            adaptation: false,
        }
    }
}

impl Voice {
    const AMP_EPS: f32 = 1e-6;
    const PHONATION_AMP_UPDATE_EPS: f32 = 0.01;
    const PHONATION_UPDATE_SMOOTH_TAU_SEC: f32 = 0.02;

    #[allow(clippy::too_many_arguments)]
    pub fn spawn_from_control(
        control: VoiceControl,
        articulation_config: ArticulationCoreConfig,
        assigned_id: u64,
        start_frame: u64,
        metadata: VoiceMetadata,
        fs: f32,
        landscape: Option<&LandscapeFrame>,
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

        let phonation_engine = PhonationEngine::from_spec(&effective_control.phonation.spec, seed);
        let social_coupling = effective_control.phonation.spec.social_coupling();

        let pitch_config = pitch_core_config_from_control(&effective_control.pitch);
        let pitch = AnyPitchCore::from_config(&pitch_config, target_pitch_log2, &mut rng);
        let adaptation_config = adaptation_config_from_control(&effective_control.adaptation);
        let adaptation =
            crate::life::adaptation::AdaptationContext::from_config(&adaptation_config, 0);

        let body = sound_body::build_sound_body_from_control(
            &effective_control,
            target_freq,
            fs,
            landscape,
            &mut rng,
        );

        let mut pitch_ctl = PitchController::new(
            pitch,
            adaptation,
            target_pitch_log2,
            integration_window,
            rng,
        );
        pitch_ctl
            .core_mut()
            .set_landscape_weight(effective_control.pitch.landscape_weight);
        pitch_ctl
            .core_mut()
            .set_exploration(effective_control.pitch.exploration);
        pitch_ctl
            .core_mut()
            .set_persistence(effective_control.pitch.persistence);
        pitch_ctl.core_mut().set_crowding(
            effective_control.pitch.crowding_strength,
            effective_control.pitch.crowding_sigma_cents,
            effective_control.pitch.crowding_sigma_from_roughness,
        );
        pitch_ctl
            .core_mut()
            .set_leave_self_out(effective_control.pitch.leave_self_out);
        pitch_ctl
            .core_mut()
            .set_leave_self_out_mode(effective_control.pitch.leave_self_out_mode);
        if let Some(cents) = effective_control.pitch.neighbor_step_cents {
            pitch_ctl.core_mut().set_neighbor_step_cents(cents);
        }
        pitch_ctl
            .core_mut()
            .set_anneal_temp(effective_control.pitch.anneal_temp);
        pitch_ctl
            .core_mut()
            .set_move_cost_coeff(effective_control.pitch.move_cost_coeff);
        if let Some(exp) = effective_control.pitch.move_cost_exp {
            pitch_ctl.core_mut().set_move_cost_exp(exp);
        }
        pitch_ctl
            .core_mut()
            .set_improvement_threshold(effective_control.pitch.improvement_threshold);
        pitch_ctl.core_mut().set_global_peaks(
            effective_control.pitch.global_peak_count,
            effective_control.pitch.global_peak_min_sep_cents,
        );
        pitch_ctl.core_mut().set_ratio_candidates(
            effective_control.pitch.use_ratio_candidates,
            effective_control.pitch.ratio_candidate_count,
        );
        pitch_ctl
            .core_mut()
            .set_move_cost_time_scale(effective_control.pitch.move_cost_time_scale);
        pitch_ctl
            .core_mut()
            .set_leave_self_out_harmonics(effective_control.pitch.leave_self_out_harmonics);
        pitch_ctl
            .core_mut()
            .set_proposal_interval_sec(effective_control.pitch.proposal_interval_sec);
        if let Some(cents) = effective_control.pitch.window_cents {
            pitch_ctl.core_mut().set_window_cents(cents);
        }
        if let Some(top_k) = effective_control.pitch.top_k {
            pitch_ctl.core_mut().set_top_k(top_k);
        }
        if let Some(temperature) = effective_control.pitch.temperature {
            pitch_ctl.core_mut().set_temperature(temperature);
        }
        if let Some(cents) = effective_control.pitch.sigma_cents {
            pitch_ctl.core_mut().set_sigma_cents(cents);
        }
        if let Some(count) = effective_control.pitch.random_candidates {
            pitch_ctl.core_mut().set_random_candidates(count);
        }
        pitch_ctl.set_adaptation_enabled(effective_control.adaptation.enabled);

        let (
            articulation_core,
            lifecycle_label,
            default_by_articulation,
            breath_gain_init,
            voice_adsr,
        ) = match &articulation_config {
            ArticulationCoreConfig::Entrain {
                lifecycle,
                breath_gain_init,
                ..
            } => {
                let life_label = match lifecycle {
                    LifecycleConfig::Decay { .. } => "decay",
                    LifecycleConfig::Sustain { .. } => "sustain",
                };
                ("entrain", life_label, 1.0, *breath_gain_init, None)
            }
            ArticulationCoreConfig::Seq {
                breath_gain_init, ..
            } => ("seq", "none", 1.0, *breath_gain_init, None),
            ArticulationCoreConfig::Drone {
                breath_gain_init,
                envelope,
                ..
            } => {
                let adsr = envelope.as_ref().map(|env| super::sound::ToneAdsr {
                    attack_sec: env.attack_sec,
                    decay_sec: env.decay_sec,
                    sustain_level: env.sustain_level,
                    release_sec: env.release_sec,
                });
                ("drone", "none", 1.0, *breath_gain_init, adsr)
            }
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

        Voice {
            id: assigned_id,
            metadata,
            fixed_body_method: effective_control.body.method,
            fixed_phonation_when: std::mem::discriminant(&effective_control.phonation.spec.when),
            base_control: control,
            effective_control,
            articulation: ArticulationWrapper::new(
                core,
                breath_gain,
                matches!(phonation_engine.mode, PhonationMode::Hold),
            ),
            pitch_ctl,
            phonation_engine,
            social_coupling,
            body,
            last_signal: Default::default(),
            release_gain: 1.0,
            release_sec: 0.03,
            release_pending: false,
            remove_pending: false,
            phonation_scratch: Default::default(),
            active_render_notes: Vec::new(),
            life_accumulator: None,
            voice_adsr,
        }
    }

    fn apply_body_runtime(&mut self) {
        let runtime = BodyRuntime::from_control(&self.effective_control.body);
        self.body.set_amp(runtime.amp);
        self.body.apply_timbre_controls(
            runtime.brightness,
            runtime.inharmonic,
            runtime.motion,
            runtime.spread,
            runtime.unison,
        );
    }

    fn apply_adaptation_control(&mut self) {
        let config = adaptation_config_from_control(&self.effective_control.adaptation);
        let adaptation = self.pitch_ctl.adaptation_mut();
        adaptation.tau_fast = config.tau_fast.unwrap_or(0.5).max(1e-3);
        adaptation.tau_slow = config
            .tau_slow
            .unwrap_or(20.0)
            .max(adaptation.tau_fast + 1e-3);
        adaptation.w_boredom = config.w_boredom.unwrap_or(1.0).max(0.0);
        adaptation.w_familiarity = config.w_familiarity.unwrap_or(0.2).max(0.0);
        adaptation.rho_self = config.rho_self.unwrap_or(0.15).clamp(0.0, 1.0);
        adaptation.boredom_gamma = config.boredom_gamma.unwrap_or(0.5).clamp(0.1, 1.0);
        adaptation.self_smoothing_radius = config.self_smoothing_radius.unwrap_or(1);
        adaptation.silence_mass_epsilon = config.silence_mass_epsilon.unwrap_or(1e-6).max(0.0);
        self.pitch_ctl
            .set_adaptation_enabled(self.effective_control.adaptation.enabled);
    }

    fn apply_pitch_control(&mut self) {
        let pitch = &self.effective_control.pitch;
        let center_log2 = pitch.freq.max(1.0).log2();
        let gravity = pitch
            .tessitura_gravity
            .unwrap_or_else(|| tessitura_gravity_from_control(pitch.gravity));
        let core = self.pitch_ctl.core_mut();
        core.set_tessitura_center(center_log2);
        core.set_tessitura_gravity(gravity);
        if let Some(cents) = pitch.neighbor_step_cents {
            core.set_neighbor_step_cents(cents);
        }
        core.set_landscape_weight(pitch.landscape_weight);
        core.set_exploration(pitch.exploration);
        core.set_persistence(pitch.persistence);
        core.set_crowding(
            pitch.crowding_strength,
            pitch.crowding_sigma_cents,
            pitch.crowding_sigma_from_roughness,
        );
        core.set_leave_self_out(pitch.leave_self_out);
        core.set_leave_self_out_mode(pitch.leave_self_out_mode);
        core.set_anneal_temp(pitch.anneal_temp);
        core.set_move_cost_coeff(pitch.move_cost_coeff);
        if let Some(exp) = pitch.move_cost_exp {
            core.set_move_cost_exp(exp);
        }
        core.set_improvement_threshold(pitch.improvement_threshold);
        core.set_global_peaks(pitch.global_peak_count, pitch.global_peak_min_sep_cents);
        core.set_ratio_candidates(pitch.use_ratio_candidates, pitch.ratio_candidate_count);
        if let Some(cents) = pitch.window_cents {
            core.set_window_cents(cents);
        }
        if let Some(top_k) = pitch.top_k {
            core.set_top_k(top_k);
        }
        if let Some(temperature) = pitch.temperature {
            core.set_temperature(temperature);
        }
        if let Some(cents) = pitch.sigma_cents {
            core.set_sigma_cents(cents);
        }
        if let Some(count) = pitch.random_candidates {
            core.set_random_candidates(count);
        }
        core.set_move_cost_time_scale(pitch.move_cost_time_scale);
        core.set_leave_self_out_harmonics(pitch.leave_self_out_harmonics);
        core.set_proposal_interval_sec(pitch.proposal_interval_sec);
    }

    fn apply_phonation_control(&mut self) {
        self.phonation_engine
            .update_from_spec(&self.effective_control.phonation.spec);
        self.articulation.set_autonomous_attack_enabled(matches!(
            self.phonation_engine.mode,
            PhonationMode::Hold
        ));
        self.social_coupling = self.effective_control.phonation.spec.social_coupling();
    }

    fn apply_effective_control(&mut self, control: VoiceControl, dirty: Dirty) {
        self.effective_control = control;
        if dirty.body {
            self.apply_body_runtime();
        }
        if dirty.adaptation {
            self.apply_adaptation_control();
        }
        if dirty.pitch {
            self.apply_pitch_control();
        }
        if dirty.phonation {
            self.apply_phonation_control();
        }
    }

    fn ensure_fixed_kinds(&self, control: &VoiceControl) -> Result<(), String> {
        if control.body.method != self.fixed_body_method
            || std::mem::discriminant(&control.phonation.spec.when) != self.fixed_phonation_when
        {
            return Err(
                "update cannot change body.method or phonation.when kind; use spawn() for type selection"
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
    pub fn metadata(&self) -> &VoiceMetadata {
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

    pub fn apply_patch(&mut self, patch: &ControlUpdate) -> Result<(), String> {
        if patch == &ControlUpdate::default() {
            return Ok(());
        }
        let mut control = self.effective_control.clone();
        control.apply_update(patch);
        self.ensure_fixed_kinds(&control)?;
        let dirty = Dirty::from_update(patch);
        self.base_control = control.clone();
        self.apply_effective_control(control, dirty);
        Ok(())
    }

    pub fn apply_update(&mut self, update: &ControlUpdate) -> Result<(), String> {
        self.apply_patch(update)
    }

    #[cfg(test)]
    pub(crate) fn set_accumulated_time_for_test(&mut self, value: f32) {
        self.pitch_ctl.set_accumulated_time_for_test(value);
    }

    #[cfg(test)]
    pub(crate) fn pitch_core_for_test(&self) -> &AnyPitchCore {
        self.pitch_ctl.core_for_test()
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

    pub fn set_neighbor_step_cents(&mut self, value: f32) {
        self.pitch_ctl.core_mut().set_neighbor_step_cents(value);
    }

    /// Update pitch targets at control rate (hop-sized steps).
    pub fn update_pitch_target(
        &mut self,
        rhythms: &NeuralRhythms,
        dt_sec: f32,
        landscape: &Landscape,
        neighbor_pitch_log2: &[f32],
        neighbor_salience: &[f32],
    ) {
        let current_freq = self.body.base_freq_hz();
        self.pitch_ctl.update_pitch_target(
            current_freq,
            rhythms,
            dt_sec,
            landscape,
            &self.effective_control.pitch,
            neighbor_pitch_log2,
            neighbor_salience,
        );
    }

    /// Control-rate entry point for pitch + articulation updates.
    #[allow(clippy::too_many_arguments)]
    pub fn tick_control(
        &mut self,
        dt_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        neighbor_pitch_log2: &[f32],
        neighbor_salience: &[f32],
        global_coupling: f32,
    ) -> ArticulationSignal {
        self.decide_pitch_target(
            dt_sec,
            rhythms,
            landscape,
            neighbor_pitch_log2,
            neighbor_salience,
        );
        self.commit_decided_control(dt_sec, rhythms, landscape, global_coupling)
    }

    /// Decide phase.
    ///
    /// Contract: this must not mutate body/articulation/lifecycle fields that
    /// other agents can observe in the same substep. Only local pitch-controller
    /// decision state (target/adaptation memory) may change here.
    #[allow(clippy::too_many_arguments)]
    pub fn decide_pitch_target(
        &mut self,
        dt_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        neighbor_pitch_log2: &[f32],
        neighbor_salience: &[f32],
    ) {
        self.update_pitch_target(
            rhythms,
            dt_sec,
            landscape,
            neighbor_pitch_log2,
            neighbor_salience,
        );
    }

    /// Commit phase: applies articulation/body/lifecycle using already-decided targets.
    pub fn commit_decided_control(
        &mut self,
        dt_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        global_coupling: f32,
    ) -> ArticulationSignal {
        self.update_articulation_autonomous(dt_sec, rhythms);
        self.tick_articulation_lifecycle(dt_sec, rhythms, landscape, global_coupling)
    }

    /// Update articulation at control rate (hop-sized steps).
    #[allow(clippy::too_many_arguments)]
    pub fn update_articulation(
        &mut self,
        dt_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        neighbor_pitch_log2: &[f32],
        neighbor_salience: &[f32],
        global_coupling: f32,
    ) -> ArticulationSignal {
        self.decide_pitch_target(
            dt_sec,
            rhythms,
            landscape,
            neighbor_pitch_log2,
            neighbor_salience,
        );
        self.commit_decided_control(dt_sec, rhythms, landscape, global_coupling)
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
        match self.effective_control.pitch.pitch_apply_mode {
            PitchApplyMode::GateSnap => {
                if apply_planned_pitch {
                    self.body.set_pitch_log2(planned.target_pitch_log2);
                }
            }
            PitchApplyMode::Glide => {
                self.articulation.set_gate(1.0);
                let tau = self.effective_control.pitch.pitch_glide_tau_sec.max(0.0);
                if tau <= 1e-6 {
                    self.body.set_pitch_log2(planned.target_pitch_log2);
                } else {
                    let alpha = 1.0 - (-dt_sec.max(0.0) / tau).exp();
                    let next_pitch_log2 = current_pitch_log2
                        + (planned.target_pitch_log2 - current_pitch_log2) * alpha.clamp(0.0, 1.0);
                    self.body.set_pitch_log2(next_pitch_log2);
                }
            }
        }
    }

    pub fn tick_articulation_lifecycle(
        &mut self,
        dt_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        global_coupling: f32,
    ) -> ArticulationSignal {
        let consonance_level = landscape.evaluate_pitch_level(self.body.base_freq_hz());
        let selection_score = landscape.evaluate_pitch_score(self.body.base_freq_hz());
        if let Some(ref mut acc) = self.life_accumulator {
            acc.accumulate_tick(consonance_level);
        }
        let mut signal = self.articulation.process(
            consonance_level,
            selection_score,
            rhythms,
            dt_sec,
            global_coupling,
        );
        if self.release_pending {
            let step = dt_sec / self.release_sec.max(1e-6);
            self.release_gain = (self.release_gain - step).max(0.0);
        }
        signal.amplitude *= self.compute_output_gain();
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

    #[allow(clippy::too_many_arguments)]
    pub fn tick_phonation(
        &mut self,
        tb: &Timebase,
        now: Tick,
        rhythms: &NeuralRhythms,
        social: Option<&SocialDensityTrace>,
        social_coupling: f32,
        extra_gate_gain: f32,
        consonance: f32,
    ) -> PhonationBatch {
        let mut batch = PhonationBatch::default();
        self.tick_phonation_into(
            tb,
            now,
            rhythms,
            social,
            social_coupling,
            extra_gate_gain,
            consonance,
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
        consonance: f32,
        out: &mut PhonationBatch,
    ) {
        out.source_id = self.id;
        out.clear();
        self.phonation_scratch.events.clear();
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
        let phonation_mode = self.phonation_engine.mode;
        let had_tone_on = out.cmds.iter().any(|cmd| matches!(cmd, ToneCmd::On { .. }));
        if matches!(phonation_mode, PhonationMode::Gated) {
            for onset in &out.onsets {
                self.articulation
                    .apply_phonation_onset(consonance, onset.strength);
            }
        }
        let target_amp = self.compute_target_amp();
        let freq_hz = self.body.base_freq_hz();
        let continuous_drive = self.effective_control.body.continuous_drive;
        self.emit_phonation_updates(now, target_amp, freq_hz, continuous_drive, &mut out.cmds);
        self.prune_tracked_render_notes(&out.cmds);
        if self.phonation_scratch.events.is_empty() {
            if had_tone_on {
                Self::discard_pending_render_tone_ons(&mut out.cmds);
            }
            return;
        }
        if !freq_hz.is_finite() || freq_hz <= 0.0 {
            if had_tone_on {
                Self::discard_pending_render_tone_ons(&mut out.cmds);
            }
            self.phonation_scratch.events.clear();
            return;
        }
        if target_amp <= Self::AMP_EPS {
            if had_tone_on {
                Self::discard_pending_render_tone_ons(&mut out.cmds);
            }
            self.phonation_scratch.events.clear();
            return;
        }
        let render_modulator = self.articulation.render_modulator_spec(phonation_mode);
        let body = self.body_snapshot();
        let smoothing_tau_sec = Self::PHONATION_UPDATE_SMOOTH_TAU_SEC;
        let hold_ticks = if matches!(phonation_mode, PhonationMode::Hold) {
            Some(Tick::MAX)
        } else {
            None
        };
        for event in self.phonation_scratch.events.drain(..) {
            out.tones.push(ToneSpec {
                tone_id: event.tone_id,
                onset: event.onset_tick,
                hold_ticks,
                freq_hz,
                amp: target_amp,
                smoothing_tau_sec,
                body: body.clone(),
                render_modulator: render_modulator.clone(),
                adsr: self.voice_adsr,
            });
        }
        self.track_new_render_notes(&out.tones);
        debug_assert!(
            !out.cmds.iter().any(|cmd| matches!(cmd, ToneCmd::On { .. })) || !out.tones.is_empty(),
            "ToneCmd::On emitted without tone specs"
        );
    }

    fn discard_pending_render_tone_ons(out_cmds: &mut Vec<ToneCmd>) {
        // Keep Off/Update commands intact when a new render tone cannot be materialized.
        out_cmds.retain(|cmd| !matches!(cmd, ToneCmd::On { .. }));
    }

    fn emit_phonation_updates(
        &mut self,
        now: Tick,
        target_amp: f32,
        freq_hz: f32,
        continuous_drive: f32,
        out_cmds: &mut Vec<ToneCmd>,
    ) {
        for tracked in &mut self.active_render_notes {
            let amp_changed =
                (target_amp - tracked.last_target_amp).abs() >= Self::PHONATION_AMP_UPDATE_EPS;
            let freq_changed = freq_hz.is_finite()
                && freq_hz > 0.0
                && (freq_hz - tracked.last_target_freq_hz).abs() > 0.01;
            let drive_changed = (continuous_drive - tracked.last_continuous_drive).abs() > 1e-4;
            if !amp_changed && !freq_changed && !drive_changed {
                continue;
            }
            out_cmds.push(ToneCmd::Update {
                tone_id: tracked.tone_id,
                at_tick: Some(now),
                update: ToneUpdate {
                    target_freq_hz: if freq_changed { Some(freq_hz) } else { None },
                    target_amp: if amp_changed { Some(target_amp) } else { None },
                    continuous_drive: if drive_changed {
                        Some(continuous_drive)
                    } else {
                        None
                    },
                },
            });
            if amp_changed {
                tracked.last_target_amp = target_amp;
            }
            if freq_changed {
                tracked.last_target_freq_hz = freq_hz;
            }
            if drive_changed {
                tracked.last_continuous_drive = continuous_drive;
            }
        }
    }

    fn prune_tracked_render_notes(&mut self, cmds: &[ToneCmd]) {
        let off_tone_ids: Vec<ToneId> = cmds
            .iter()
            .filter_map(|cmd| match cmd {
                ToneCmd::Off { tone_id, .. } => Some(*tone_id),
                _ => None,
            })
            .collect();
        if off_tone_ids.is_empty() {
            return;
        }
        self.active_render_notes
            .retain(|tracked| !off_tone_ids.contains(&tracked.tone_id));
    }

    fn track_new_render_notes(&mut self, tones: &[ToneSpec]) {
        self.active_render_notes
            .extend(tones.iter().map(|tone| TrackedRenderNote {
                tone_id: tone.tone_id,
                last_target_amp: tone.amp,
                last_target_freq_hz: tone.freq_hz,
                last_continuous_drive: 0.0,
            }));
    }

    pub(crate) fn compute_target_amp(&self) -> f32 {
        let mut amp =
            self.body.amp() * self.compute_output_gain() * self.articulation.vitality_scalar();
        if !amp.is_finite() {
            amp = 0.0;
        }
        amp.max(0.0)
    }

    pub(crate) fn compute_output_gain(&self) -> f32 {
        let gate = self.articulation.gate().clamp(0.0, 1.0);
        let release = self.release_gain.clamp(0.0, 1.0);
        let mut gain = gate * release;
        if !gain.is_finite() {
            gain = 0.0;
        }
        gain.max(0.0)
    }

    pub(crate) fn body_snapshot(&self) -> BodySnapshot {
        self.body.snapshot()
    }

    pub fn render_spectrum(&mut self, amps: &mut [f32], space: &Log2Space) {
        let signal = self.last_signal;
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        self.body.project_spectral_body(amps, space, &signal);
    }

    pub fn is_alive(&self) -> bool {
        if self.remove_pending && self.release_gain <= 0.0 {
            return false;
        }
        self.articulation.is_alive() && self.release_gain > 0.0
    }
}
