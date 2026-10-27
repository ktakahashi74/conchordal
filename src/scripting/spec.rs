use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) enum BrainKind {
    Entrain,
    Seq,
    Drone,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum PhonationKind {
    Sustain,
    Repeat,
}

#[derive(Clone, Debug)]
pub(super) struct PopulationSpec {
    pub(super) control: VoiceControl,
    pub(super) respawn_policy: RespawnPolicy,
    pub(super) respawn_settle_strategy: Option<SpawnStrategy>,
    pub(super) respawn_capacity: Option<usize>,
    pub(super) respawn_min_c_level: Option<f32>,
    pub(super) respawn_background_death_rate_per_sec: f32,
    pub(super) crowding_target_same: bool,
    pub(super) crowding_target_other: bool,
    pub(super) brain: BrainKind,
    pub(super) phonation_spec: PhonationSpec,
    pub(super) phonation_gate: PhonationGate,
    pub(super) field_duration_spec: FieldDurationSpec,
    pub(super) finite_duration: Option<DurationSpec>,
    pub(super) rest_periods: Option<f32>,
    pub(super) rest_sustain: bool,
    pub(super) pulse_sync: Option<f32>,
    pub(super) social_coupling: Option<f32>,
    pub(super) entrainment: Option<f32>,
    pub(super) intrinsic_rate: Option<FLOAT>,
    pub(super) rhythm_role: Option<RhythmRole>,
    pub(super) microtiming: Option<f32>,
    pub(super) measure_accent: f32,
    pub(super) endurance_sec: Option<f32>,
    pub(super) recovery_sec: Option<f32>,
    pub(super) attack_cost_fraction: Option<f32>,
    pub(super) attack_recharge_fraction: Option<f32>,
    pub(super) continuous_recharge_score_low: Option<f32>,
    pub(super) continuous_recharge_score_high: Option<f32>,
    pub(super) selection_environment: bool,
    pub(super) dissonance_penalty: Option<f32>,
    pub(super) adsr_user_set: bool,
    pub(super) rhythm_coupling: RhythmCouplingMode,
    pub(super) rhythm_reward: Option<MetabolismRhythmReward>,
    pub(super) rhythm_reward_user_enabled: bool,
    pub(super) rhythm_freq: Option<f32>,
    pub(super) consonance_movement: bool,
    pub(super) pitch_apply_mode_user_set: bool,
    pub(super) pitch_core_user_set: bool,
    pub(super) glide_user_set: bool,
    pub(super) viability_scope_user_set: bool,
    pub(super) specified: BTreeMap<&'static str, Position>,
    pub(super) diagnostics: Arc<Mutex<Vec<ScriptWarning>>>,
    pub(super) position: Position,
    pub(super) tie_user_set: bool,
    pub(super) explicit_sustain: bool,
    pub(super) explicit_metric: bool,
}

impl PopulationSpec {
    pub(super) fn record_call(
        &mut self,
        name: &'static str,
        position: Position,
        diagnostics: &ScriptWarnings,
    ) {
        self.diagnostics = diagnostics.clone();
        self.position = position;
        self.specified.insert(name, position);
    }

    pub(super) fn warn(&self, message: String) {
        self.diagnostics
            .lock()
            .expect("lock script warnings")
            .push(ScriptWarning {
                message,
                position: self.position,
            });
    }

    pub(super) fn preset(body: BodyMethod) -> Self {
        let mut control = VoiceControl::default();
        control.body.method = body;
        control.body.envelope = EnvelopeConfig::for_body_method(body);
        Self {
            control,
            respawn_policy: RespawnPolicy::None,
            respawn_settle_strategy: None,
            respawn_capacity: None,
            respawn_min_c_level: None,
            respawn_background_death_rate_per_sec: 0.0,
            crowding_target_same: true,
            crowding_target_other: false,
            brain: BrainKind::Entrain,
            phonation_spec: PhonationSpec::default(),
            phonation_gate: PhonationGate::default(),
            field_duration_spec: FieldDurationSpec::default(),
            finite_duration: None,
            rest_periods: None,
            rest_sustain: false,
            pulse_sync: None,
            social_coupling: None,
            entrainment: None,
            intrinsic_rate: None,
            rhythm_role: None,
            microtiming: None,
            measure_accent: 0.0,
            endurance_sec: None,
            recovery_sec: None,
            attack_cost_fraction: None,
            attack_recharge_fraction: None,
            continuous_recharge_score_low: None,
            continuous_recharge_score_high: None,
            selection_environment: false,
            dissonance_penalty: None,
            adsr_user_set: false,
            rhythm_coupling: RhythmCouplingMode::TemporalOnly,
            rhythm_reward: None,
            rhythm_reward_user_enabled: false,
            rhythm_freq: None,
            consonance_movement: false,
            pitch_apply_mode_user_set: false,
            pitch_core_user_set: false,
            glide_user_set: false,
            viability_scope_user_set: false,
            specified: BTreeMap::new(),
            diagnostics: Arc::default(),
            position: Position::NONE,
            tie_user_set: false,
            explicit_sustain: false,
            explicit_metric: false,
        }
    }

    pub(super) fn release_sec(&self) -> f32 {
        self.control.body.envelope.release_sec.max(0.0)
    }

    pub(super) fn pulse_sync_or(&self, default: f32) -> f32 {
        self.pulse_sync.unwrap_or(default).clamp(0.0, 1.0)
    }

    pub(super) fn social_coupling_or(&self, default: f32) -> f32 {
        self.social_coupling.unwrap_or(default).clamp(0.0, 1.0)
    }

    pub(super) fn entrainment_or(&self, default: f32) -> f32 {
        self.entrainment.unwrap_or(default).clamp(0.0, 1.0)
    }

    pub(super) fn rhythm_role_or(&self, default: RhythmRole) -> RhythmRole {
        self.rhythm_role.unwrap_or(default)
    }

    pub(super) fn microtiming_or(&self, default: f32) -> f32 {
        // set_microtiming already clamps the stored value and the built spec is
        // sanitized before use, so no clamp is needed here.
        self.microtiming.unwrap_or(default)
    }

    /// Get the current coupled-timing spec, creating a default one (medium
    /// coupling, plain beat) if the voice is not already on the continuum. Lets
    /// a bare `entrainment(..)` / `rhythm_role(..)` / `microtiming(..)` work
    /// without first naming a preset.
    pub(super) fn coupled_spec_mut(&mut self) -> &mut CoupledTimingSpec {
        if !matches!(self.phonation_spec.timing, PhonationTiming::Coupled(_)) {
            self.phonation_spec.timing = PhonationTiming::Coupled(CoupledTimingSpec::default());
            self.ensure_gated_duration(DEFAULT_GATE_COUNT);
        }
        match &mut self.phonation_spec.timing {
            PhonationTiming::Coupled(spec) => spec,
            _ => unreachable!("just ensured Coupled"),
        }
    }

    pub(super) fn apply_field_duration_spec(&mut self) {
        if let Some(DurationSpec::Field(field)) = &mut self.finite_duration {
            *field = self.field_duration_spec.clone();
        }
        if let DurationSpec::Field(field) = &mut self.phonation_spec.duration {
            *field = self.field_duration_spec.clone();
        }
    }

    pub(super) fn clear_rhythm_preset_controls(&mut self) {
        self.rhythm_freq = None;
        self.rhythm_coupling = RhythmCouplingMode::TemporalOnly;
        self.rhythm_reward = None;
    }

    pub(super) fn lifecycle_envelope(&self) -> EnvelopeConfig {
        let env = &self.control.body.envelope;
        EnvelopeConfig {
            attack_sec: env.attack_sec.max(0.0),
            decay_sec: env.decay_sec.max(0.0),
            sustain_level: env.sustain_level.clamp(0.0, 1.0),
            release_sec: env.release_sec.max(0.0),
        }
    }

    pub(super) fn lifecycle_config(&self) -> LifecycleConfig {
        let selection_environment = if self.viability_scope_user_set {
            self.selection_environment
        } else {
            self.continuous_recharge_score_low.is_some()
        };
        let energy_knobs = self.endurance_sec.is_some()
            || self.recovery_sec.is_some()
            || self.attack_cost_fraction.is_some()
            || self.attack_recharge_fraction.is_some()
            || self.continuous_recharge_score_low.is_some()
            || self.continuous_recharge_score_high.is_some()
            || selection_environment
            || self.dissonance_penalty.is_some();
        if energy_knobs || self.adsr_user_set {
            // An explicit ADSR is a timbral statement, not a mortality
            // statement: without endurance the voice has no basal drain and
            // sings until released.
            LifecycleConfig::Sustain {
                endurance_sec: self.endurance_sec,
                recovery_sec: self.recovery_sec,
                attack_cost_fraction: self.attack_cost_fraction,
                attack_recharge_fraction: self.attack_recharge_fraction,
                continuous_recharge_score_low: self.continuous_recharge_score_low,
                continuous_recharge_score_high: self.continuous_recharge_score_high,
                selection_environment,
                dissonance_penalty: self.dissonance_penalty.unwrap_or(0.0),
                envelope: self.lifecycle_envelope(),
            }
        } else {
            LifecycleConfig::default()
        }
    }

    pub(super) fn articulation_config(&self) -> ArticulationCoreConfig {
        match self.brain {
            BrainKind::Entrain => ArticulationCoreConfig::Entrain {
                lifecycle: self.lifecycle_config(),
                rhythm_freq: self.rhythm_freq,
                rhythm_coupling: self.rhythm_coupling,
                rhythm_reward: self.rhythm_reward,
                breath_gain_init: None,
            },
            BrainKind::Seq => ArticulationCoreConfig::Seq {
                duration: DEFAULT_SEQ_DURATION_SEC,
                breath_gain_init: None,
            },
            BrainKind::Drone => ArticulationCoreConfig::Drone {
                sway: None,
                breath_gain_init: None,
            },
        }
    }

    pub(super) fn spawn_spec(&self) -> VoiceSpec {
        let mut control = self.control.clone();
        if self.consonance_movement {
            if !self.pitch_core_user_set {
                control.pitch.core_kind = PitchCoreKind::HillClimb;
            }
            if !self.glide_user_set {
                control
                    .pitch
                    .set_pitch_glide_tau_sec_clamped(DEFAULT_CONSONANCE_MOVEMENT_GLIDE_TAU_SEC);
            }
        }
        control.phonation.spec = self.phonation_spec.clone();
        control.phonation.gate = self.phonation_gate;
        if self.rest_periods.is_some() {
            control.phonation.finite_duration = self.finite_duration.clone();
            control.phonation.rest_periods = self.rest_periods;
            if self.rest_sustain {
                control.phonation.spec.duration = DurationSpec::WhileAlive;
                control.phonation.spec.sustain_participation = true;
            }
        }
        // Resolve how seek_consonance() pitch decisions land, unless the script
        // chose explicitly: re-attacking voices snap at onsets, sustained voices
        // glide. Order-independent (resolved at placement, not at call time).
        if !self.pitch_apply_mode_user_set
            && (self.consonance_movement || control.phonation.spec.is_sustained_participation())
        {
            let apply = if control.phonation.spec.is_sustained_participation() {
                PitchApplyMode::Glide
            } else {
                match self.phonation_spec.timing {
                    PhonationTiming::Once => PitchApplyMode::Glide,
                    PhonationTiming::Pulse { .. } | PhonationTiming::Coupled(_) => {
                        PitchApplyMode::GateSnap
                    }
                }
            };
            control.pitch.set_pitch_apply_mode(apply);
        }
        if !self.tie_user_set
            && self.explicit_metric
            && control.pitch.mode == PitchMode::Free
            && control.phonation.spec.is_sustained_participation()
            && matches!(
                control.phonation.spec.timing,
                PhonationTiming::Coupled(CoupledTimingSpec {
                    relation: RhythmRelation::Synchronized,
                    ..
                })
            )
        {
            // Keep the existing policy dormant in Glide for a later mode switch.
            control.phonation.tie = 1.0;
        }
        VoiceSpec {
            control,
            articulation: self.articulation_config(),
        }
    }

    pub(super) fn set_amp(&mut self, amp: f32) {
        self.control.set_amp_clamped(amp);
    }

    pub(super) fn set_freq(&mut self, freq: f32) {
        self.control.pitch.set_freq_lock_clamped(freq);
    }

    pub(super) fn set_landscape_weight(&mut self, value: f32) {
        self.control.pitch.set_landscape_weight_clamped(value);
    }

    pub(super) fn set_neighbor_step_cents(&mut self, value: f32) {
        self.control.pitch.set_neighbor_step_cents_clamped(value);
    }

    pub(super) fn set_tessitura_gravity(&mut self, value: f32) {
        self.control.pitch.set_tessitura_gravity_clamped(value);
    }

    pub(super) fn set_continuous_drive(&mut self, value: f32) {
        self.control.set_continuous_drive_clamped(value);
    }

    pub(super) fn set_crowding(&mut self, strength: f32, sigma_cents: f32) {
        self.control.pitch.set_crowding_strength_clamped(strength);
        self.control
            .pitch
            .set_crowding_sigma_cents_clamped(sigma_cents);
    }

    pub(super) fn set_crowding_strength_only(&mut self, strength: f32) {
        self.control.pitch.set_crowding_strength_clamped(strength);
    }

    pub(super) fn set_routing(&mut self, routing: Routing) {
        self.control.body.routing = routing;
    }

    pub(super) fn set_crowding_target(
        &mut self,
        same_population_visible: bool,
        other_population_visible: bool,
    ) {
        self.crowding_target_same = same_population_visible;
        self.crowding_target_other = other_population_visible;
    }

    pub(super) fn set_leave_self_out(&mut self, enabled: bool) {
        self.control.pitch.set_leave_self_out(enabled);
    }

    pub(super) fn set_move_cost_coeff(&mut self, value: f32) {
        self.control.pitch.set_move_cost_coeff_clamped(value);
    }

    pub(super) fn set_move_cost_exp(&mut self, value: f32) {
        self.control
            .pitch
            .set_move_cost_exp_clamped(value.round() as i64);
    }

    pub(super) fn set_phrase_boundary(&mut self, value: FLOAT) {
        self.control.phonation.phrase_boundary = if value.is_finite() && value >= 0.0 {
            value
        } else {
            self.warn("phrase_boundary() expects a finite nonnegative weight; using 0".into());
            0.0
        };
    }

    pub(super) fn set_follow_own_sequence(&mut self, value: FLOAT) {
        self.control.pitch.follow_own_sequence = if value.is_finite() && value >= 0.0 {
            value
        } else {
            self.warn("follow_own_sequence() expects a finite nonnegative weight; using 0".into());
            0.0
        };
    }

    pub(super) fn set_melodic_continuity(&mut self, value: FLOAT) {
        self.control.pitch.melodic_continuity = if value.is_finite() && value >= 0.0 {
            value
        } else {
            self.warn("melodic_continuity() expects a finite nonnegative weight; using 0".into());
            0.0
        };
    }

    pub(super) fn set_follow_tonal_context(&mut self, value: FLOAT) {
        self.control.pitch.follow_tonal_context = if value.is_finite() && value >= 0.0 {
            value
        } else {
            self.warn("follow_tonal_context() expects a finite nonnegative weight; using 0".into());
            0.0
        };
    }

    pub(super) fn set_dwell(&mut self, value: FLOAT) {
        if !(value as f32).is_finite() || value < 0.0 {
            self.warn(
                "dwell() expects a finite nonnegative f32 weight; clamped (non-finite becomes 0)"
                    .into(),
            );
        }
        self.control.pitch.dwell = if (value as f32).is_finite() {
            value.max(0.0) as f32
        } else {
            0.0
        };
    }

    pub(super) fn set_proposal_interval_sec(&mut self, value: f32) {
        self.control.pitch.set_proposal_interval_sec_clamped(value);
    }

    pub(super) fn set_global_peaks(&mut self, count: i64, min_sep_cents: f32) {
        self.control.pitch.set_global_peak_count_clamped(count);
        self.control
            .pitch
            .set_global_peak_min_sep_cents_clamped(min_sep_cents);
    }

    pub(super) fn set_ratio_candidates(&mut self, count: i64) {
        self.control.pitch.set_ratio_candidate_count_clamped(count);
        self.control.pitch.set_use_ratio_candidates(count > 0);
    }

    pub(super) fn set_window_cents(&mut self, value: f32) {
        self.control.pitch.set_window_cents_clamped(value);
    }

    pub(super) fn set_top_k(&mut self, value: INT) {
        self.control.pitch.set_top_k_clamped(value);
    }

    pub(super) fn set_temperature(&mut self, value: f32) {
        self.control.pitch.set_temperature_clamped(value);
    }

    pub(super) fn set_sigma_cents(&mut self, value: f32) {
        self.control.pitch.set_sigma_cents_clamped(value);
    }

    pub(super) fn set_random_candidates(&mut self, value: INT) {
        self.control.pitch.set_random_candidates_clamped(value);
    }

    pub(super) fn set_move_cost_time_scale(
        &mut self,
        name: &str,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        let value = parse_enum(
            "move_cost_time_scale",
            name,
            position,
            &[
                ("legacy", MoveCostTimeScale::LegacyIntegrationWindow),
                ("proposal_interval", MoveCostTimeScale::ProposalInterval),
            ],
        )?;
        self.control.pitch.set_move_cost_time_scale(value);
        Ok(())
    }

    pub(super) fn set_pitch_apply_mode(
        &mut self,
        name: &str,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        let value = parse_enum(
            "pitch_apply_mode",
            name,
            position,
            &[
                ("gate_snap", PitchApplyMode::GateSnap),
                ("glide", PitchApplyMode::Glide),
                ("rearticulate", PitchApplyMode::Rearticulate),
            ],
        )?;
        self.set_pitch_apply_mode_resolved(value);
        Ok(())
    }

    pub(super) fn set_pitch_apply_mode_resolved(&mut self, mode: PitchApplyMode) {
        self.pitch_apply_mode_user_set = true;
        self.control.pitch.set_pitch_apply_mode(mode);
    }

    pub(super) fn set_tie(&mut self, value: f32) {
        self.tie_user_set = true;
        self.control.phonation.tie = crate::core::float::sanitize01(value);
    }

    pub(super) fn set_articulation_persistence(&mut self, value: f32) {
        self.control.phonation.articulation_persistence = crate::core::float::sanitize01(value);
    }

    pub(super) fn set_rearticulate_above(&mut self, value: f32) {
        self.control
            .pitch
            .set_rearticulate_above_cents_clamped(value);
    }

    pub(super) fn set_pitch_glide_tau_sec(&mut self, value: f32) {
        self.glide_user_set = true;
        self.control.pitch.set_pitch_glide_tau_sec_clamped(value);
    }

    pub(super) fn set_consonance_movement(&mut self) {
        self.consonance_movement = true;
        self.control.pitch.mode = PitchMode::Free;
    }

    pub(super) fn set_anchor(&mut self) {
        self.consonance_movement = false;
        self.control.pitch.mode = PitchMode::Lock;
    }

    pub(super) fn set_pitch_core(
        &mut self,
        name: &str,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        let value = parse_enum(
            "pitch_core",
            name,
            position,
            &[
                ("hill_climb", PitchCoreKind::HillClimb),
                ("peak_sampler", PitchCoreKind::PeakSampler),
            ],
        )?;
        self.pitch_core_user_set = true;
        self.control.pitch.core_kind = value;
        Ok(())
    }

    pub(super) fn warn_timbre_noop_if_needed(&self, label: &str) {
        if matches!(self.control.body.method, BodyMethod::Sine) {
            self.warn(format!("{label}() is a no-op for sine bodies"));
        }
    }

    pub(super) fn set_brightness(&mut self, brightness: f32) {
        self.control.set_timbre_brightness_clamped(brightness);
        self.warn_timbre_noop_if_needed("brightness");
    }

    pub(super) fn set_spread(&mut self, spread: f32) {
        self.control.set_timbre_spread_clamped(spread);
        self.warn_timbre_noop_if_needed("spread");
    }

    pub(super) fn set_unison(&mut self, value: INT) {
        self.control.set_timbre_unison_clamped(value);
        self.warn_timbre_noop_if_needed("unison");
    }

    pub(super) fn set_modes(&mut self, pattern: ModePattern) {
        self.control.body.modes = Some(pattern);
    }

    pub(super) fn set_brain(
        &mut self,
        name: &str,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        let value = parse_enum(
            "brain",
            name,
            position,
            &[
                ("entrain", BrainKind::Entrain),
                ("seq", BrainKind::Seq),
                ("drone", BrainKind::Drone),
            ],
        )?;
        self.brain = value;
        Ok(())
    }

    pub(super) fn set_phonation(&mut self, kind: PhonationKind) {
        match kind {
            PhonationKind::Sustain => {
                self.rest_sustain = true;
                self.explicit_sustain = true;
                if matches!(
                    self.phonation_spec.timing,
                    PhonationTiming::Coupled(CoupledTimingSpec {
                        relation: crate::scenario::RhythmRelation::Participating,
                        ..
                    })
                ) || (self.explicit_metric
                    && matches!(self.phonation_spec.timing, PhonationTiming::Coupled(_)))
                {
                    self.phonation_spec.duration = DurationSpec::WhileAlive;
                    self.phonation_spec.sustain_participation = true;
                } else {
                    self.clear_rhythm_preset_controls();
                    self.phonation_spec = PhonationSpec::default();
                }
            }
            PhonationKind::Repeat => {
                self.explicit_metric = false;
                self.rest_sustain = false;
                self.explicit_sustain = false;
                self.clear_rhythm_preset_controls();
                self.phonation_spec = PhonationSpec {
                    sustain_participation: false,
                    timing: PhonationTiming::Pulse {
                        rate_hz: DEFAULT_PULSE_RATE,
                        sync: self.pulse_sync_or(DEFAULT_PULSE_SYNC),
                        social: self.social_coupling_or(0.0),
                    },
                    duration: DurationSpec::Gates(DEFAULT_GATE_COUNT),
                };
            }
        }
    }

    pub(super) fn ensure_gated_duration(&mut self, default_gates: u32) {
        if matches!(self.phonation_spec.duration, DurationSpec::WhileAlive) {
            self.phonation_spec.duration = DurationSpec::Gates(default_gates.max(1));
        }
    }

    /// Metric preset: a deep attractor on the shared beat. Locks tightly and
    /// reads as a stable pulse. No Hz argument -- the tempo region is the
    /// director's `temporal_basin`; this only sets how hard the voice locks.
    pub(super) fn set_metric(&mut self) {
        self.explicit_metric = true;
        self.phonation_spec.sustain_participation = self.explicit_sustain;
        self.clear_rhythm_preset_controls();
        self.phonation_spec.timing = PhonationTiming::Coupled(
            CoupledTimingSpec {
                coupling: self.entrainment_or(0.95),
                role: self.rhythm_role_or(RhythmRole::Beat),
                microtiming: self.microtiming_or(0.0),
                measure_accent: self.measure_accent,
                ..CoupledTimingSpec::default()
            }
            .sanitized(),
        );
        if self.explicit_sustain {
            self.phonation_spec.duration = DurationSpec::WhileAlive;
        } else {
            self.ensure_gated_duration(DEFAULT_GATE_COUNT);
        }
    }

    /// Acoustic participation; legacy vitality and attack reward remain for comparison.
    pub(super) fn set_entrained(&mut self) {
        self.explicit_metric = false;
        self.clear_rhythm_preset_controls();
        let spec = CoupledTimingSpec {
            relation: crate::scenario::RhythmRelation::Participating,
            coupling: self.entrainment_or(0.5),
            role: self.rhythm_role_or(RhythmRole::Beat),
            microtiming: self.microtiming_or(0.0),
            measure_accent: self.measure_accent,
            social: self.social_coupling_or(0.6),
            vitality_lambda: 0.8,
            vitality_floor: 0.35,
            reward: 0.6,
            ..CoupledTimingSpec::default()
        }
        .sanitized();
        self.phonation_spec.timing = PhonationTiming::Coupled(spec);
        if self.explicit_sustain {
            self.phonation_spec.duration = DurationSpec::WhileAlive;
            self.phonation_spec.sustain_participation = true;
        } else {
            self.ensure_gated_duration(DEFAULT_GATE_COUNT);
        }
        self.rhythm_coupling = RhythmCouplingMode::TemporalTimesVitality {
            lambda_v: spec.vitality_lambda,
            v_floor: spec.vitality_floor,
        };
        self.rhythm_reward = Some(MetabolismRhythmReward {
            rho_t: spec.reward,
            metric: RhythmRewardMetric::AttackPhaseMatch,
        });
    }

    /// Preserve private opportunities while using acoustic context for sound or skip.
    pub(super) fn set_steady(&mut self) {
        self.set_entrained();
        if let PhonationTiming::Coupled(timing) = &mut self.phonation_spec.timing {
            timing.steady = true;
        }
    }

    /// Flow preset: near-zero coupling, a free renewal process that ignores the
    /// beat and fires at clustered, non-metric IOIs.
    pub(super) fn set_flow(&mut self) {
        self.explicit_metric = false;
        self.clear_rhythm_preset_controls();
        self.phonation_spec.timing = PhonationTiming::Coupled(
            CoupledTimingSpec {
                relation: crate::scenario::RhythmRelation::Participating,
                coupling: self.entrainment_or(0.05),
                flow_depth: 0.65,
                role: self.rhythm_role_or(RhythmRole::Texture),
                microtiming: self.microtiming_or(0.0),
                measure_accent: self.measure_accent,
                ..CoupledTimingSpec::default()
            }
            .sanitized(),
        );
        if self.explicit_sustain {
            self.phonation_spec.duration = DurationSpec::WhileAlive;
            self.phonation_spec.sustain_participation = true;
        } else {
            self.ensure_gated_duration(1);
        }
    }

    pub(super) fn set_entrainment(&mut self, strength: f32) {
        let strength = strength.clamp(0.0, 1.0);
        self.entrainment = Some(strength);
        self.coupled_spec_mut().coupling = strength;
    }

    pub(super) fn set_rhythm_role(
        &mut self,
        name: &str,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        let value = parse_enum(
            "rhythm_role",
            name,
            position,
            &[
                ("beat", RhythmRole::Beat),
                ("subdivision", RhythmRole::Subdivision),
                ("accent", RhythmRole::Accent),
                ("texture", RhythmRole::Texture),
            ],
        )?;
        self.rhythm_role = Some(value);
        self.coupled_spec_mut().role = value;
        Ok(())
    }

    pub(super) fn set_microtiming(&mut self, amount: f32) {
        let amount = if amount.is_finite() {
            amount.clamp(-0.5, 0.5)
        } else {
            0.0
        };
        self.microtiming = Some(amount);
        self.coupled_spec_mut().microtiming = amount;
    }

    pub(super) fn set_measure_accent(&mut self, amount: f32) {
        let amount = sanitize01(amount);
        self.measure_accent = amount;
        self.coupled_spec_mut().measure_accent = amount;
    }

    pub(super) fn set_when_once(&mut self) {
        self.explicit_metric = false;
        self.rest_sustain = false;
        self.explicit_sustain = false;
        self.phonation_spec.sustain_participation = false;
        self.clear_rhythm_preset_controls();
        self.phonation_spec.timing = PhonationTiming::Once;
    }

    pub(super) fn set_when_pulse(&mut self, rate: f32) {
        self.explicit_metric = false;
        self.rest_sustain = false;
        self.explicit_sustain = false;
        self.phonation_spec.sustain_participation = false;
        let rate = rate.max(0.01);
        self.clear_rhythm_preset_controls();
        match &mut self.phonation_spec.timing {
            PhonationTiming::Pulse { rate_hz, .. } => *rate_hz = rate,
            _ => {
                self.phonation_spec.timing = PhonationTiming::Pulse {
                    rate_hz: rate,
                    sync: self.pulse_sync_or(DEFAULT_PULSE_SYNC),
                    social: self.social_coupling_or(0.0),
                };
            }
        }
        if let PhonationTiming::Pulse { sync, social, .. } = &mut self.phonation_spec.timing {
            if let Some(value) = self.pulse_sync {
                *sync = value.clamp(0.0, 1.0);
            }
            if let Some(value) = self.social_coupling {
                *social = value.clamp(0.0, 1.0);
            }
        }
        self.ensure_gated_duration(1);
    }

    pub(super) fn set_duration_while_alive(&mut self) {
        self.explicit_metric = false;
        self.finite_duration = None;
        self.rest_sustain = false;
        self.explicit_sustain = false;
        self.clear_rhythm_preset_controls();
        self.phonation_spec = PhonationSpec::default();
    }

    /// Withhold the voice's first onset until perceived consonance reaches the
    /// low bound of its consonance-viability window (`consonance_viability()`,
    /// or 0.0 -- always viable -- if that window was never set), then latch
    /// open for good.
    pub(super) fn set_phonate_when_viable(&mut self) {
        self.phonation_gate = PhonationGate::WhenViable;
    }

    pub(super) fn set_duration_cycles(&mut self, n: u32) {
        self.finite_duration = Some(DurationSpec::Gates(n.max(1)));
        self.explicit_sustain = false;
        self.phonation_spec.sustain_participation = false;
        self.phonation_spec.duration = DurationSpec::Gates(n.max(1));
    }

    pub(super) fn set_adaptive_duration(&mut self) {
        self.finite_duration = Some(DurationSpec::Field(self.field_duration_spec.clone()));
        self.explicit_sustain = false;
        self.phonation_spec.sustain_participation = false;
        self.phonation_spec.duration = DurationSpec::Field(self.field_duration_spec.clone());
    }

    pub(super) fn set_rest_on_habituation(&mut self, value: f32) {
        self.control.phonation.rest_on_habituation = value;
    }

    pub(super) fn set_rest(&mut self, periods: f32) {
        self.rest_periods = Some(crate::core::float::sanitize_nonnegative_finite(periods));
    }

    pub(super) fn set_pulse_lock(&mut self, depth: f32) {
        let depth = depth.clamp(0.0, 1.0);
        self.pulse_sync = Some(depth);
        if let PhonationTiming::Pulse { sync, .. } = &mut self.phonation_spec.timing {
            *sync = depth;
        }
    }

    pub(super) fn set_social(&mut self, coupling: f32) {
        let coupling = coupling.clamp(0.0, 1.0);
        self.social_coupling = Some(coupling);
        match &mut self.phonation_spec.timing {
            PhonationTiming::Coupled(spec) => spec.social = coupling,
            PhonationTiming::Pulse { social, .. } => *social = coupling,
            PhonationTiming::Once => {}
        }
    }

    pub(super) fn set_duration_range(&mut self, min: f32, max: f32) {
        self.field_duration_spec.hold_min_theta = min.clamp(0.0, 1.0);
        self.field_duration_spec.hold_max_theta = max.clamp(0.0, 1.0);
        self.apply_field_duration_spec();
    }

    pub(super) fn set_duration_curve(&mut self, k: f32, x0: f32) {
        self.field_duration_spec.curve_k = k;
        self.field_duration_spec.curve_x0 = x0;
        self.apply_field_duration_spec();
    }

    pub(super) fn set_shorten_on_drop(&mut self, gain: f32) {
        self.field_duration_spec.drop_gain = gain.max(0.0);
        self.apply_field_duration_spec();
    }

    pub(super) fn set_endurance(&mut self, sec: f32) {
        if sec.is_finite() && sec > 0.0 {
            self.endurance_sec = Some(sec);
        } else {
            self.warn("endurance() expects finite seconds greater than zero".to_string());
        }
    }

    pub(super) fn set_recovery(&mut self, sec: f32) {
        if sec.is_finite() && sec > 0.0 {
            self.recovery_sec = Some(sec);
        } else {
            self.warn("recovery() expects finite seconds greater than zero".to_string());
        }
    }

    pub(super) fn set_attack_cost_fraction(&mut self, value: f32) {
        if value.is_finite() && value >= 0.0 {
            self.attack_cost_fraction = Some(value);
        } else {
            self.warn("attack_cost_fraction() expects a finite non-negative value".to_string());
        }
    }

    pub(super) fn set_attack_recharge_fraction(&mut self, value: f32) {
        if value.is_finite() && value >= 0.0 {
            self.attack_recharge_fraction = Some(value);
        } else {
            self.warn("attack_recharge_fraction() expects a finite non-negative value".to_string());
        }
    }

    pub(super) fn set_consonance_viability(&mut self, low: f32, high: f32) {
        if !low.is_finite() || !high.is_finite() {
            self.warn("consonance_viability() expects finite thresholds".to_string());
            return;
        }
        self.continuous_recharge_score_low = Some(low);
        self.continuous_recharge_score_high = Some(high);
    }

    pub(super) fn set_viability_scope(
        &mut self,
        name: &str,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        let value = parse_enum(
            "viability_scope",
            name,
            position,
            &[("environment", true), ("total", false)],
        )?;
        self.viability_scope_user_set = true;
        self.selection_environment = value;
        Ok(())
    }

    pub(super) fn set_dissonance_penalty(&mut self, value: f32) {
        if value.is_finite() && value >= 0.0 {
            self.dissonance_penalty = Some(value);
        } else {
            self.warn("dissonance_penalty() expects a finite non-negative value".to_string());
        }
    }

    pub(super) fn set_adsr(&mut self, a: f32, d: f32, s: f32, r: f32) {
        self.control.body.envelope = EnvelopeConfig {
            attack_sec: a.max(0.0),
            decay_sec: d.max(0.0),
            sustain_level: s.clamp(0.0, 1.0),
            release_sec: r.max(0.0),
        };
        self.adsr_user_set = true;
    }

    pub(super) fn set_respawn_random(&mut self) {
        self.respawn_policy = RespawnPolicy::Random;
    }

    pub(super) fn set_respawn_hereditary(&mut self, sigma_oct: f32) {
        let sigma_oct = if sigma_oct.is_finite() {
            sigma_oct.max(0.0)
        } else {
            0.0
        };
        self.respawn_policy = RespawnPolicy::Hereditary { sigma_oct };
    }

    pub(super) fn set_respawn_consonance(&mut self) {
        self.respawn_policy = RespawnPolicy::PeakBiased {
            config: RespawnPeakBiasConfig::default(),
        };
    }

    pub(super) fn set_respawn_settle_strategy(&mut self, strategy: SpawnStrategy) {
        self.respawn_settle_strategy = Some(strategy);
    }

    pub(super) fn set_respawn_capacity(&mut self, value: INT) {
        self.respawn_capacity = Some(value.max(1) as usize);
    }

    pub(super) fn set_respawn_min_c_level(&mut self, value: f32) {
        self.respawn_min_c_level = Some(value.clamp(0.0, 1.0));
    }

    pub(super) fn set_respawn_background_death_rate(&mut self, value: f32) {
        self.respawn_background_death_rate_per_sec = value.max(0.0);
    }

    pub(super) fn set_rhythm_coupling_vitality(&mut self, lambda_v: f32, v_floor: f32) {
        self.rhythm_coupling = RhythmCouplingMode::TemporalTimesVitality { lambda_v, v_floor };
    }

    pub(super) fn set_rhythm_freq(&mut self, value: f32) {
        self.rhythm_freq = Some(value.max(0.0));
    }

    pub(super) fn set_rhythm_reward(
        &mut self,
        rho_t: f32,
        metric: &str,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        let enabled = parse_enum(
            "rhythm_reward",
            metric,
            position,
            &[("attack_phase_match", true), ("none", false)],
        )?;
        self.rhythm_reward_user_enabled = enabled;
        self.rhythm_reward = enabled.then_some(MetabolismRhythmReward {
            rho_t,
            metric: RhythmRewardMetric::AttackPhaseMatch,
        });
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct PopulationSpecHandle {
    pub(super) spec: PopulationSpec,
}

#[derive(Clone, Debug)]
pub struct PopulationHandle {
    pub(super) id: u64,
}

#[derive(Clone, Debug)]
pub(super) struct FloraSpecHandle {
    pub(super) spec: FloraSpec,
}

#[derive(Clone, Debug)]
pub(super) struct FloraHandle {
    pub(super) id: u64,
}

pub(super) fn validate_flora_numeric(
    name: &str,
    value: FLOAT,
    position: Position,
    diagnostics: &ScriptWarnings,
) -> Result<f32, Box<EvalAltResult>> {
    if name == "rest_on_habituation" {
        let converted = value as f32;
        if !converted.is_finite() || value < 0.0 {
            diagnostics
                .lock()
                .expect("lock script warnings")
                .push(ScriptWarning {
                    message:
                        "Flora rest_on_habituation expects a finite nonnegative f32 weight; using 0"
                            .into(),
                    position,
                });
            return Ok(0.0);
        }
        return Ok(converted);
    }
    if name == "pace" {
        let min = crate::scenario::FLORA_PACE_MIN as FLOAT;
        let max = f32::MAX as FLOAT;
        let clamped = if value.is_nan() {
            min
        } else {
            value.clamp(min, max)
        };
        if value != clamped {
            diagnostics
                .lock()
                .expect("lock script warnings")
                .push(ScriptWarning {
                    message: format!(
                        "Flora pace {value} clamped to {clamped}; expected a finite f32 value >= {min}"
                    ),
                    position,
                });
        }
        return Ok(clamped as f32);
    }
    let (min, max) = match name {
        "amp" => (0.0, 1.0),
        "fade" => (0.0, 60.0),
        _ => unreachable!("registered Flora parameter"),
    };
    if !value.is_finite() {
        return Err(Box::new(EvalAltResult::ErrorRuntime(
            format!("Flora {name} must be finite before and after f32 conversion").into(),
            position,
        )));
    }
    if !(min..=max).contains(&value) {
        return Err(Box::new(EvalAltResult::ErrorRuntime(
            format!("Flora {name} must be within [{min}, {max}]").into(),
            position,
        )));
    }
    let converted = value as f32;
    if !converted.is_finite() {
        return Err(Box::new(EvalAltResult::ErrorRuntime(
            format!("Flora {name} must be finite before and after f32 conversion").into(),
            position,
        )));
    }
    Ok(converted)
}

impl FloraSpecHandle {
    pub(super) fn with_numeric(
        mut self,
        name: &str,
        value: FLOAT,
        position: Position,
        diagnostics: &ScriptWarnings,
    ) -> Result<Self, Box<EvalAltResult>> {
        let value = validate_flora_numeric(name, value, position, diagnostics)?;
        match name {
            "pace" => self.spec.pace = value,
            "amp" => self.spec.amp = value,
            "fade" => self.spec.fade_sec = value,
            "rest_on_habituation" => self.spec.rest_on_habituation = value,
            _ => unreachable!("registered Flora spec parameter"),
        }
        Ok(self)
    }
}

pub(super) fn flora_pitch_apply_mode(
    name: &str,
    position: Position,
    diagnostics: &ScriptWarnings,
) -> Option<PitchApplyMode> {
    match name {
        "glide" => Some(PitchApplyMode::Glide),
        "rearticulate" => Some(PitchApplyMode::Rearticulate),
        _ => {
            diagnostics
                .lock()
                .expect("lock script warnings")
                .push(ScriptWarning {
                    message: format!(
                        "Flora pitch_apply_mode expects glide or rearticulate; '{name}' ignored"
                    ),
                    position,
                });
            None
        }
    }
}
