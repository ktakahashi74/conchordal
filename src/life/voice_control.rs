use super::*;
use crate::scenario::control::ControlUpdate;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Dirty {
    pub(super) body: bool,
    pub(super) pitch: bool,
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
            || update.crowding_strength.is_some()
            || update.crowding_sigma_cents.is_some()
            || update.leave_self_out.is_some()
            || update.tail_roughness_aversion.is_some()
            || update.move_cost_coeff.is_some()
            || update.move_cost_exp.is_some()
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
            || update.pitch_apply_mode.is_some()
            || update.pitch_glide_tau_sec.is_some()
            || update.rearticulate_above_cents.is_some();
        Self { body, pitch }
    }
}

impl Voice {
    pub(super) fn autonomous_attack_enabled(
        mode: PhonationMode,
        application: PitchApplyMode,
        participating: bool,
    ) -> bool {
        matches!(mode, PhonationMode::Hold)
            && application != PitchApplyMode::Rearticulate
            && !participating
    }

    pub(crate) fn set_control_tick(&mut self, tick: Tick) {
        self.control_tick = Some(tick);
    }

    pub(super) fn apply_pitch_control(&mut self) {
        let next = select_pitch_application(&self.effective_control.pitch, None);
        if next != self.participation_pitch.application {
            if let Some(shape) = self.phonation_engine.clock.shape_mut() {
                shape.state = crate::life::shape_policy::Shape::Base;
                shape.veto_plan = false;
            }
            self.participation_pitch.rearticulations.clear();
            self.participation_pitch.steps.clear();
            self.participation_pitch.pending = None;
            self.participation_pitch.glide_target = None;
            if let Some(gap) = &mut self.phonation_engine.span_gap {
                gap.pending_pitch = None;
                gap.glide_target = None;
            }
            self.participation_pitch.adopted = false;
            self.participation_pitch.pulse_update_pending |= (next == PitchApplyMode::Rearticulate)
                != (self.participation_pitch.application == PitchApplyMode::Rearticulate);
            self.participation_pitch.application = next;
            self.participation_pitch.adoption_application = next;
            self.articulation
                .set_autonomous_attack_enabled(Self::autonomous_attack_enabled(
                    self.phonation_engine.mode,
                    next,
                    self.sustain_participation,
                ));
        }
        let pitch = &self.effective_control.pitch;
        self.pitch_ctl.apply_control(pitch);
    }

    pub(super) fn apply_effective_control(&mut self, mut control: VoiceControl, dirty: Dirty) {
        if control.pitch.follow_tonal_context > 0. && !control.supports_own_sequence() {
            tracing::warn!("follow_tonal_context disabled by unsupported live control");
            control.pitch.follow_tonal_context = 0.;
        }
        if self.sequence_control.is_some()
            && (control.supports_own_sequence() != self.effective_control.supports_own_sequence()
                || control.pitch.freq != self.effective_control.pitch.freq
                || control.pitch.range_oct != self.effective_control.pitch.range_oct)
        {
            let was_supported = self
                .pitch_ctl
                .own_sequence
                .as_ref()
                .is_some_and(|s| s.supported);
            if let Some(s) = &mut self.pitch_ctl.own_sequence {
                s.supported = false;
            }
            if let Some(s) = &mut self.pitch_ctl.melodic {
                s.supported = false;
            }
            self.flush_sequence_controls();
            if let Some(s) = &mut self.pitch_ctl.own_sequence {
                s.supported = was_supported;
            }
            let center = control.pitch.freq.log2();
            let half = control.pitch.range_oct.clamp(0.0, 6.0) * 0.5;
            if let Some(me) = &mut self.pitch_ctl.melodic {
                me.reconfigure(
                    control.supports_own_sequence(),
                    (center - half, center + half),
                );
            }
            if let Some(sequence) = &mut self.pitch_ctl.own_sequence {
                sequence.reconfigure(
                    control.supports_own_sequence(),
                    (center - half, center + half),
                );
            }
        }
        self.effective_control = control;
        if dirty.body {
            self.apply_body_runtime();
        }
        if dirty.pitch {
            self.apply_pitch_control();
        }
    }

    fn ensure_fixed_kinds(&self, control: &VoiceControl) -> Result<(), String> {
        if control.body.method != self.fixed_body_method
            || control.phonation.spec.timing_update_kind() != self.fixed_phonation_timing
        {
            return Err(
                "update cannot change body.method or phonation timing kind; use spawn() for type selection"
                    .to_string(),
            );
        }
        Ok(())
    }

    pub fn should_retain(&self) -> bool {
        if self.release.remove_pending && self.release.gain <= 0.0 {
            return !self.active_render_notes.is_empty();
        }
        if self.release.remove_pending {
            return self.is_alive() || !self.active_render_notes.is_empty();
        }
        self.is_alive() || self.phonation_engine.has_active_notes()
    }

    pub fn apply_patch(&mut self, patch: &ControlUpdate) -> Result<(), String> {
        if patch == &ControlUpdate::default() {
            return Ok(());
        }
        let mut control = self.effective_control.clone();
        control.apply_update(patch);
        if self.harmonic.member && !control.supports_harmonic_rhythm() {
            control.pitch.pitch_apply_mode = self.effective_control.pitch.pitch_apply_mode;
            control.pitch.rearticulate_above_cents =
                self.effective_control.pitch.rearticulate_above_cents;
            control.pitch.mode = self.effective_control.pitch.mode;
            control.pitch.freq = self.effective_control.pitch.freq;
        }
        self.ensure_fixed_kinds(&control)?;
        control.validate_pitch_application()?;
        let dirty = Dirty::from_update(patch);
        self.base_control = control.clone();
        self.apply_effective_control(control, dirty);
        Ok(())
    }

    pub fn apply_update(&mut self, update: &ControlUpdate) -> Result<(), String> {
        self.apply_patch(update)
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
        self.tick_control_with_listener_pressure(
            dt_sec,
            rhythms,
            landscape,
            neighbor_pitch_log2,
            neighbor_salience,
            global_coupling,
            ListenerPressure::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tick_control_with_listener_pressure(
        &mut self,
        dt_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        neighbor_pitch_log2: &[f32],
        neighbor_salience: &[f32],
        global_coupling: f32,
        listener_pressure: ListenerPressure,
    ) -> ArticulationSignal {
        self.decide_pitch_target_with_listener_pressure(
            dt_sec,
            rhythms,
            landscape,
            neighbor_pitch_log2,
            neighbor_salience,
            listener_pressure,
        );
        self.commit_decided_control(dt_sec, rhythms, landscape, global_coupling)
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
        self.update_articulation_with_listener_pressure(
            dt_sec,
            rhythms,
            landscape,
            neighbor_pitch_log2,
            neighbor_salience,
            global_coupling,
            ListenerPressure::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update_articulation_with_listener_pressure(
        &mut self,
        dt_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        neighbor_pitch_log2: &[f32],
        neighbor_salience: &[f32],
        global_coupling: f32,
        listener_pressure: ListenerPressure,
    ) -> ArticulationSignal {
        self.decide_pitch_target_with_listener_pressure(
            dt_sec,
            rhythms,
            landscape,
            neighbor_pitch_log2,
            neighbor_salience,
            listener_pressure,
        );
        self.commit_decided_control(dt_sec, rhythms, landscape, global_coupling)
    }

    pub fn update_articulation_autonomous(&mut self, dt_sec: f32, rhythms: &NeuralRhythms) {
        if self.harmonic.member {
            self.articulation.set_gate(1.0);
            return;
        }
        let configured = select_pitch_application(&self.effective_control.pitch, None);
        if configured != self.participation_pitch.application {
            self.participation_pitch.adoption_application = configured;
            self.participation_pitch.application = configured;
        }
        let current_freq = self.body.base_freq_hz().max(1.0);
        if self.sustain_participation {
            // Release grows the buffer if the birth capacity is insufficient.
            debug_assert!(
                self.participation_pitch.steps.len() < self.participation_pitch.steps.capacity()
            );
            debug_assert!(self.control_tick.is_some(), "control sample");
            // Release skips an untimed control sample and continues the control update.
            if let Some(at) = self.control_tick {
                self.participation_pitch.steps.push(ParticipationPitchStep {
                    at,
                    before_log2: current_freq.log2(),
                    target_log2: self.target_pitch_log2(),
                    dt: dt_sec,
                });
            }
        }
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
        if self.defers_participation_pitch() {
            self.articulation.set_gate(1.0);
            return;
        }
        let application = if self.participation_pitch.application == PitchApplyMode::Rearticulate {
            self.participation_pitch.adoption_application
        } else {
            self.participation_pitch.application
        };
        match application {
            PitchApplyMode::GateSnap => {
                if apply_planned_pitch {
                    self.body.set_pitch_log2(planned.target_pitch_log2);
                }
            }
            PitchApplyMode::Rearticulate => {
                self.articulation.set_gate(1.0);
                if self.participation_pitch.adopted {
                    self.body.set_pitch_log2(planned.target_pitch_log2);
                }
            }
            PitchApplyMode::Glide => {
                self.articulation.set_gate(1.0);
                let tau = self.effective_control.pitch.pitch_glide_tau_sec.max(0.0);
                if tau <= 1e-6 {
                    self.body.set_pitch_log2(planned.target_pitch_log2);
                } else {
                    let next_pitch_log2 = Self::glide_pitch(
                        current_pitch_log2,
                        planned.target_pitch_log2,
                        dt_sec,
                        tau,
                    );
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
        self.tick_articulation_lifecycle_with_body(
            dt_sec,
            rhythms,
            landscape,
            global_coupling,
            None,
        )
    }

    pub(in crate::life) fn tick_articulation_lifecycle_with_body(
        &mut self,
        dt_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        global_coupling: f32,
        body: Option<(f32, f32)>,
    ) -> ArticulationSignal {
        let consonance_level = landscape.evaluate_pitch_level(self.body.base_freq_hz());
        let selection_score = if self.selection_environment {
            self.pitch_ctl
                .environment_score(landscape, self.body.base_freq_hz())
        } else {
            landscape.evaluate_pitch_score(self.body.base_freq_hz())
        };
        let (level, score) = body.unwrap_or((consonance_level, selection_score));
        let mut signal = self
            .articulation
            .process(level, score, rhythms, dt_sec, global_coupling);
        let energy_depleted = matches!(
            &self.articulation.core,
            AnyArticulationCore::Entrain(core) if core.energy <= 0.0
        );
        if let Some(ref mut acc) = self.life_accumulator {
            acc.accumulate_tick(consonance_level, energy_depleted, dt_sec);
        }
        if self.release.pending {
            let step = dt_sec / self.release.sec.max(1e-6);
            self.release.gain = (self.release.gain - step).max(0.0);
        }
        let gain = self.compute_output_gain();
        signal.amplitude *= gain;
        signal.strike_gain *= gain;
        signal.is_active = signal.is_active && signal.amplitude > 0.0;
        self.last_signal = signal;
        signal
    }

    pub fn start_remove_fade(&mut self, fade_sec: f32) {
        if self.release.remove_pending {
            return;
        }
        self.release.remove_pending = true;
        self.release.pending = true;
        self.release.sec = fade_sec.max(1e-4);
        self.release.gain = self.release.gain.clamp(0.0, 1.0);
    }

    pub(crate) fn compute_output_gain(&self) -> f32 {
        let gate = self.articulation.gate().clamp(0.0, 1.0);
        let release = self.release.gain.clamp(0.0, 1.0);
        let mut gain = gate * release;
        if !gain.is_finite() {
            gain = 0.0;
        }
        gain.max(0.0)
    }

    pub fn is_alive(&self) -> bool {
        if self.release.remove_pending && self.release.gain <= 0.0 {
            return false;
        }
        self.articulation.is_alive() && self.release.gain > 0.0
    }
}
