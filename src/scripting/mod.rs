use std::collections::BTreeMap;
use std::fs;
use std::sync::{Arc, Mutex};

use rand::random;
use rhai::{Array, Dynamic, Engine, EvalAltResult, FLOAT, FnPtr, INT, NativeCallContext, Position};
use tracing::warn;

use crate::core::landscape::PitchObjectiveMode;
use crate::core::mode_pattern::ModePattern;

use crate::core::meter::MeterShaping;
use crate::life::control::{
    BodyMethod, ControlUpdate, LeaveSelfOutMode, MoveCostTimeScale, PitchApplyMode, PitchCoreKind,
    PitchMode, Routing, VoiceControl,
};
use crate::life::lifecycle::LifecycleConfig;
use crate::scenario::PhonationTiming;
use crate::scenario::{
    Action, ArticulationCoreConfig, ControlUpdateMode, CoupledTimingSpec, DurationSpec,
    EnvelopeConfig, FieldDurationSpec, MetabolismRhythmReward, PhonationSpec,
    RespawnPeakBiasConfig, RespawnPolicy, RhythmCouplingMode, RhythmRewardMetric, RhythmRole,
    ScaffoldConfig, Scenario, SceneMarker, SpawnStrategy, TimedEvent, VoiceSpec,
};

const DEFAULT_SEQ_DURATION_SEC: f32 = 1.0;
const DEFAULT_PULSE_RATE: f32 = 2.25;
const DEFAULT_PULSE_SYNC: f32 = 0.5;
const DEFAULT_GATE_COUNT: u32 = 5;
const DEFAULT_CONSONANCE_MOVEMENT_GLIDE_TAU_SEC: f32 = 0.30;

fn rhai_array_to_f32(values: Array, label: &str) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        if let Some(v) = value.clone().try_cast::<FLOAT>() {
            out.push(v as f32);
            continue;
        }
        if let Some(v) = value.try_cast::<INT>() {
            out.push(v as f32);
            continue;
        }
        warn!("{label} expects numeric array elements");
    }
    out
}

fn parse_leave_self_out_mode_name(current: LeaveSelfOutMode, name: &str) -> LeaveSelfOutMode {
    match name.trim().to_ascii_lowercase().as_str() {
        "approx" | "approx_harmonics" | "harmonics" => LeaveSelfOutMode::ApproxHarmonics,
        "exact" | "exact_scan" | "scan" => LeaveSelfOutMode::ExactScan,
        other => {
            warn!(
                "leave_self_out_mode() expects 'approx' or 'exact', got '{}'",
                other
            );
            current
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum BrainKind {
    Entrain,
    Seq,
    Drone,
}

#[derive(Clone, Copy, Debug)]
enum PhonationKind {
    Sustain,
    Repeat,
}

#[derive(Clone, Debug)]
struct SpeciesSpec {
    control: VoiceControl,
    respawn_policy: RespawnPolicy,
    respawn_settle_strategy: Option<SpawnStrategy>,
    respawn_capacity: usize,
    respawn_min_c_level: Option<f32>,
    respawn_background_death_rate_per_sec: f32,
    crowding_target_same: bool,
    crowding_target_other: bool,
    brain: BrainKind,
    phonation_spec: PhonationSpec,
    field_duration_spec: FieldDurationSpec,
    pulse_sync: Option<f32>,
    social_coupling: Option<f32>,
    entrainment: Option<f32>,
    rhythm_role: Option<RhythmRole>,
    microtiming: Option<f32>,
    metabolism_rate: Option<f32>,
    initial_energy: Option<f32>,
    recharge_rate: Option<f32>,
    action_cost: Option<f32>,
    viability_rate: Option<f32>,
    continuous_recharge_score_low: Option<f32>,
    continuous_recharge_score_high: Option<f32>,
    selection_approx_loo: bool,
    dissonance_cost: Option<f32>,
    adsr_user_set: bool,
    rhythm_coupling: RhythmCouplingMode,
    rhythm_reward: Option<MetabolismRhythmReward>,
    rhythm_freq: Option<f32>,
    energy_cap: Option<f32>,
    consonance_movement: bool,
    pitch_apply_mode_user_set: bool,
}

impl SpeciesSpec {
    fn preset(body: BodyMethod) -> Self {
        let mut control = VoiceControl::default();
        control.body.method = body;
        control.body.envelope = EnvelopeConfig::for_body_method(body);
        Self {
            control,
            respawn_policy: RespawnPolicy::None,
            respawn_settle_strategy: None,
            respawn_capacity: 1,
            respawn_min_c_level: None,
            respawn_background_death_rate_per_sec: 0.0,
            crowding_target_same: true,
            crowding_target_other: false,
            brain: BrainKind::Entrain,
            phonation_spec: PhonationSpec::default(),
            field_duration_spec: FieldDurationSpec::default(),
            pulse_sync: None,
            social_coupling: None,
            entrainment: None,
            rhythm_role: None,
            microtiming: None,
            metabolism_rate: None,
            initial_energy: None,
            recharge_rate: None,
            action_cost: None,
            viability_rate: None,
            continuous_recharge_score_low: None,
            continuous_recharge_score_high: None,
            selection_approx_loo: false,
            dissonance_cost: None,
            adsr_user_set: false,
            rhythm_coupling: RhythmCouplingMode::TemporalOnly,
            rhythm_reward: None,
            rhythm_freq: None,
            energy_cap: None,
            consonance_movement: false,
            pitch_apply_mode_user_set: false,
        }
    }

    fn release_sec(&self) -> f32 {
        self.control.body.envelope.release_sec.max(0.0)
    }

    fn pulse_sync_or(&self, default: f32) -> f32 {
        self.pulse_sync.unwrap_or(default).clamp(0.0, 1.0)
    }

    fn social_coupling_or(&self, default: f32) -> f32 {
        self.social_coupling.unwrap_or(default).clamp(0.0, 1.0)
    }

    fn entrainment_or(&self, default: f32) -> f32 {
        self.entrainment.unwrap_or(default).clamp(0.0, 1.0)
    }

    fn rhythm_role_or(&self, default: RhythmRole) -> RhythmRole {
        self.rhythm_role.unwrap_or(default)
    }

    fn microtiming_or(&self, default: f32) -> f32 {
        // set_microtiming already clamps the stored value and the built spec is
        // sanitized before use, so no clamp is needed here.
        self.microtiming.unwrap_or(default)
    }

    /// Get the current coupled-timing spec, creating a default one (medium
    /// coupling, plain beat) if the voice is not already on the continuum. Lets
    /// a bare `entrainment(..)` / `rhythm_role(..)` / `microtiming(..)` work
    /// without first naming a preset.
    fn coupled_spec_mut(&mut self) -> &mut CoupledTimingSpec {
        if !matches!(self.phonation_spec.timing, PhonationTiming::Coupled(_)) {
            self.phonation_spec.timing = PhonationTiming::Coupled(CoupledTimingSpec::default());
            self.ensure_gated_duration(DEFAULT_GATE_COUNT);
        }
        match &mut self.phonation_spec.timing {
            PhonationTiming::Coupled(spec) => spec,
            _ => unreachable!("just ensured Coupled"),
        }
    }

    fn apply_field_duration_spec(&mut self) {
        if let DurationSpec::Field(field) = &mut self.phonation_spec.duration {
            *field = self.field_duration_spec.clone();
        }
    }

    fn clear_rhythm_preset_controls(&mut self) {
        self.rhythm_freq = None;
        self.rhythm_coupling = RhythmCouplingMode::TemporalOnly;
        self.rhythm_reward = None;
    }

    fn lifecycle_envelope(&self) -> EnvelopeConfig {
        let env = &self.control.body.envelope;
        EnvelopeConfig {
            attack_sec: env.attack_sec.max(0.0),
            decay_sec: env.decay_sec.max(0.0),
            sustain_level: env.sustain_level.clamp(0.0, 1.0),
            release_sec: env.release_sec.max(0.0),
        }
    }

    fn lifecycle_config(&self) -> LifecycleConfig {
        let energy_knobs = self.metabolism_rate.is_some()
            || self.initial_energy.is_some()
            || self.recharge_rate.is_some()
            || self.action_cost.is_some()
            || self.viability_rate.is_some()
            || self.continuous_recharge_score_low.is_some()
            || self.continuous_recharge_score_high.is_some()
            || self.selection_approx_loo
            || self.dissonance_cost.is_some();
        if energy_knobs || self.adsr_user_set {
            // An explicit ADSR is a timbral statement, not a mortality
            // statement: without any energy knob the voice has no basal drain
            // and sings until released. Declaring an energy economy opts into
            // the starving default.
            let default_metabolism = if energy_knobs { 0.5 } else { 0.0 };
            let metabolism_rate = self.metabolism_rate.unwrap_or(default_metabolism).max(1e-6);
            LifecycleConfig::Sustain {
                initial_energy: self.initial_energy.unwrap_or(1.0).max(0.0),
                metabolism_rate,
                recharge_rate: self.recharge_rate.map(|value| value.max(0.0)),
                action_cost: self.action_cost.map(|value| value.max(0.0)),
                continuous_recharge_rate: self.viability_rate.map(|value| value.max(0.0)),
                continuous_recharge_score_low: self.continuous_recharge_score_low,
                continuous_recharge_score_high: self.continuous_recharge_score_high,
                selection_approx_loo: self.selection_approx_loo,
                dissonance_cost: self.dissonance_cost,
                envelope: self.lifecycle_envelope(),
            }
        } else {
            LifecycleConfig::default()
        }
    }

    fn articulation_config(&self) -> ArticulationCoreConfig {
        match self.brain {
            BrainKind::Entrain => ArticulationCoreConfig::Entrain {
                lifecycle: self.lifecycle_config(),
                rhythm_freq: self.rhythm_freq,
                rhythm_coupling: self.rhythm_coupling,
                rhythm_reward: self.rhythm_reward,
                breath_gain_init: None,
                energy_cap: self.energy_cap,
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

    fn spawn_spec(&self) -> VoiceSpec {
        let mut control = self.control.clone();
        control.phonation.spec = self.phonation_spec.clone();
        // Resolve how seek_consonance() pitch decisions land, unless the script
        // chose explicitly: re-attacking voices snap at onsets, sustained voices
        // glide. Order-independent (resolved at commit, not at call time).
        if self.consonance_movement && !self.pitch_apply_mode_user_set {
            let apply = match self.phonation_spec.timing {
                PhonationTiming::Once => PitchApplyMode::Glide,
                PhonationTiming::Pulse { .. } | PhonationTiming::Coupled(_) => {
                    PitchApplyMode::GateSnap
                }
            };
            control.pitch.set_pitch_apply_mode(apply);
        }
        VoiceSpec {
            control,
            articulation: self.articulation_config(),
        }
    }

    fn set_amp(&mut self, amp: f32) {
        self.control.set_amp_clamped(amp);
    }

    fn set_freq(&mut self, freq: f32) {
        self.control.pitch.set_freq_lock_clamped(freq);
    }

    fn set_landscape_weight(&mut self, value: f32) {
        self.control.pitch.set_landscape_weight_clamped(value);
    }

    fn set_neighbor_step_cents(&mut self, value: f32) {
        self.control.pitch.set_neighbor_step_cents_clamped(value);
    }

    fn set_tessitura_gravity(&mut self, value: f32) {
        self.control.pitch.set_tessitura_gravity_clamped(value);
    }

    fn set_continuous_drive(&mut self, value: f32) {
        self.control.set_continuous_drive_clamped(value);
    }

    fn set_pitch_smooth_tau(&mut self, value: f32) {
        self.control.set_pitch_smooth_tau_clamped(value);
    }

    fn set_exploration(&mut self, value: f32) {
        self.control.pitch.set_exploration_clamped(value);
    }

    fn set_persistence(&mut self, value: f32) {
        self.control.pitch.set_persistence_clamped(value);
    }

    fn set_crowding(&mut self, strength: f32, sigma_cents: f32) {
        self.control.pitch.set_crowding_strength_clamped(strength);
        self.control
            .pitch
            .set_crowding_sigma_cents_clamped(sigma_cents);
    }

    fn set_crowding_auto_sigma(&mut self, strength: f32) {
        self.control.pitch.set_crowding_strength_clamped(strength);
    }

    fn set_octave_avoidance(&mut self, weight: f32) {
        self.control.pitch.set_octave_avoidance_clamped(weight);
    }

    fn set_routing(&mut self, routing: Routing) {
        self.control.body.routing = routing;
    }

    fn set_crowding_target(&mut self, same_group_visible: bool, other_group_visible: bool) {
        self.crowding_target_same = same_group_visible;
        self.crowding_target_other = other_group_visible;
    }

    fn set_leave_self_out(&mut self, enabled: bool) {
        self.control.pitch.set_leave_self_out(enabled);
    }

    fn set_leave_self_out_mode(&mut self, name: &str) {
        let mode = parse_leave_self_out_mode_name(self.control.pitch.leave_self_out_mode, name);
        self.control.pitch.set_leave_self_out_mode(mode);
    }

    fn set_anneal_temp(&mut self, value: f32) {
        self.control.pitch.set_anneal_temp_clamped(value);
    }

    fn set_move_cost_coeff(&mut self, value: f32) {
        self.control.pitch.set_move_cost_coeff_clamped(value);
    }

    fn set_move_cost_exp(&mut self, value: f32) {
        self.control
            .pitch
            .set_move_cost_exp_clamped(value.round() as i64);
    }

    fn set_improvement_threshold(&mut self, value: f32) {
        self.control.pitch.set_improvement_threshold_clamped(value);
    }

    fn set_proposal_interval_sec(&mut self, value: f32) {
        self.control.pitch.set_proposal_interval_sec_clamped(value);
    }

    fn set_global_peaks(&mut self, count: i64, min_sep_cents: f32) {
        self.control.pitch.set_global_peak_count_clamped(count);
        self.control
            .pitch
            .set_global_peak_min_sep_cents_clamped(min_sep_cents);
    }

    fn set_ratio_candidates(&mut self, count: i64) {
        self.control.pitch.set_ratio_candidate_count_clamped(count);
        self.control.pitch.set_use_ratio_candidates(count > 0);
    }

    fn set_window_cents(&mut self, value: f32) {
        self.control.pitch.set_window_cents_clamped(value);
    }

    fn set_top_k(&mut self, value: f32) {
        self.control.pitch.set_top_k_clamped(value.round() as i64);
    }

    fn set_temperature(&mut self, value: f32) {
        self.control.pitch.set_temperature_clamped(value);
    }

    fn set_sigma_cents(&mut self, value: f32) {
        self.control.pitch.set_sigma_cents_clamped(value);
    }

    fn set_random_candidates(&mut self, value: f32) {
        self.control
            .pitch
            .set_random_candidates_clamped(value.round() as i64);
    }

    fn set_move_cost_time_scale(&mut self, name: &str) {
        let lowered = name.trim().to_ascii_lowercase();
        let value = match lowered.as_str() {
            "legacy" | "integration" | "integration_window" => {
                MoveCostTimeScale::LegacyIntegrationWindow
            }
            "proposal" | "proposal_interval" => MoveCostTimeScale::ProposalInterval,
            other => {
                warn!(
                    "move_cost_time_scale() expects 'legacy' or 'proposal_interval', got '{}'",
                    other
                );
                self.control.pitch.move_cost_time_scale
            }
        };
        self.control.pitch.set_move_cost_time_scale(value);
    }

    fn set_leave_self_out_harmonics(&mut self, value: i64) {
        self.control
            .pitch
            .set_leave_self_out_harmonics_clamped(value);
    }

    fn set_pitch_apply_mode(&mut self, name: &str) {
        let lowered = name.trim().to_ascii_lowercase();
        let mode = match lowered.as_str() {
            "gate_snap" | "gatesnap" | "snap" => PitchApplyMode::GateSnap,
            "glide" | "gliss" | "glissando" => PitchApplyMode::Glide,
            other => {
                warn!(
                    "pitch_apply_mode() expects 'gate_snap' or 'glide', got '{}'",
                    other
                );
                return;
            }
        };
        self.set_pitch_apply_mode_resolved(mode);
    }

    fn set_pitch_apply_mode_resolved(&mut self, mode: PitchApplyMode) {
        self.pitch_apply_mode_user_set = true;
        self.control.pitch.set_pitch_apply_mode(mode);
    }

    fn set_pitch_glide_tau_sec(&mut self, value: f32) {
        self.control.pitch.set_pitch_glide_tau_sec_clamped(value);
    }

    fn set_consonance_movement(&mut self) {
        self.consonance_movement = true;
        self.control.pitch.mode = PitchMode::Free;
        self.control.pitch.core_kind = PitchCoreKind::HillClimb;
        self.control
            .pitch
            .set_pitch_glide_tau_sec_clamped(DEFAULT_CONSONANCE_MOVEMENT_GLIDE_TAU_SEC);
    }

    fn set_anchor(&mut self) {
        self.consonance_movement = false;
        self.control.pitch.mode = PitchMode::Lock;
    }

    fn set_pitch_core(&mut self, name: &str) {
        let lowered = name.trim().to_ascii_lowercase();
        self.control.pitch.core_kind = match lowered.as_str() {
            "hill_climb" | "hillclimb" | "hill" => PitchCoreKind::HillClimb,
            "peak_sampler" | "peaksampler" | "peak" => PitchCoreKind::PeakSampler,
            other => {
                warn!(
                    "pitch_core() expects 'hill_climb' or 'peak_sampler', got '{}'",
                    other
                );
                self.control.pitch.core_kind
            }
        };
    }

    #[cfg(debug_assertions)]
    fn warn_timbre_noop_if_needed(&self, label: &str) {
        if matches!(self.control.body.method, BodyMethod::Sine) {
            warn!("{label}() is a no-op for sine bodies");
        }
    }

    #[cfg(not(debug_assertions))]
    fn warn_timbre_noop_if_needed(&self, _label: &str) {}

    fn set_brightness(&mut self, brightness: f32) {
        self.control.set_timbre_brightness_clamped(brightness);
        self.warn_timbre_noop_if_needed("brightness");
    }

    fn set_spread(&mut self, spread: f32) {
        self.control.set_timbre_spread_clamped(spread);
        self.warn_timbre_noop_if_needed("spread");
    }

    fn set_unison(&mut self, unison: f32) {
        let unison = if unison.is_finite() {
            unison.round() as i64
        } else {
            1
        };
        self.control.set_timbre_unison_clamped(unison);
        self.warn_timbre_noop_if_needed("unison");
    }

    fn set_modes(&mut self, pattern: ModePattern) {
        self.control.body.modes = Some(pattern);
    }

    fn set_brain(&mut self, name: &str) {
        let lowered = name.trim().to_ascii_lowercase();
        self.brain = match lowered.as_str() {
            "drone" => BrainKind::Drone,
            "seq" => BrainKind::Seq,
            "entrain" => BrainKind::Entrain,
            other => {
                warn!("brain '{}' not supported yet", other);
                self.brain
            }
        };
    }

    fn set_phonation(&mut self, kind: PhonationKind) {
        match kind {
            PhonationKind::Sustain => self.set_duration_while_alive(),
            PhonationKind::Repeat => {
                self.clear_rhythm_preset_controls();
                self.phonation_spec = PhonationSpec {
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

    fn ensure_gated_duration(&mut self, default_gates: u32) {
        if matches!(self.phonation_spec.duration, DurationSpec::WhileAlive) {
            self.phonation_spec.duration = DurationSpec::Gates(default_gates.max(1));
        }
    }

    /// Metric preset: a deep attractor on the shared beat. Locks tightly and
    /// reads as a stable pulse. No Hz argument -- the tempo region is the
    /// director's `temporal_basin`; this only sets how hard the voice locks.
    fn set_metric(&mut self) {
        self.clear_rhythm_preset_controls();
        self.phonation_spec.timing = PhonationTiming::Coupled(
            CoupledTimingSpec {
                coupling: self.entrainment_or(0.95),
                role: self.rhythm_role_or(RhythmRole::Beat),
                microtiming: self.microtiming_or(0.0),
                ..CoupledTimingSpec::default()
            }
            .sanitized(),
        );
        self.ensure_gated_duration(DEFAULT_GATE_COUNT);
    }

    /// Entrained preset: medium coupling, so synchronization emerges over time
    /// while timing still feeds the life cycle (vitality + attack reward).
    fn set_entrained(&mut self) {
        self.clear_rhythm_preset_controls();
        let spec = CoupledTimingSpec {
            coupling: self.entrainment_or(0.5),
            role: self.rhythm_role_or(RhythmRole::Beat),
            microtiming: self.microtiming_or(0.0),
            social: self.social_coupling_or(0.6),
            vitality_lambda: 0.8,
            vitality_floor: 0.35,
            reward: 0.6,
            ..CoupledTimingSpec::default()
        }
        .sanitized();
        self.phonation_spec.timing = PhonationTiming::Coupled(spec);
        self.ensure_gated_duration(DEFAULT_GATE_COUNT);
        self.rhythm_coupling = RhythmCouplingMode::TemporalTimesVitality {
            lambda_v: spec.vitality_lambda,
            v_floor: spec.vitality_floor,
        };
        self.rhythm_reward = Some(MetabolismRhythmReward {
            rho_t: spec.reward,
            metric: RhythmRewardMetric::AttackPhaseMatch,
        });
    }

    /// Flow preset: near-zero coupling, a free renewal process that ignores the
    /// beat and fires at clustered, non-metric IOIs.
    fn set_flow(&mut self) {
        self.clear_rhythm_preset_controls();
        self.phonation_spec.timing = PhonationTiming::Coupled(
            CoupledTimingSpec {
                coupling: self.entrainment_or(0.05),
                flow_depth: 0.65,
                role: self.rhythm_role_or(RhythmRole::Texture),
                microtiming: self.microtiming_or(0.0),
                ..CoupledTimingSpec::default()
            }
            .sanitized(),
        );
        self.ensure_gated_duration(1);
    }

    fn set_entrainment(&mut self, strength: f32) {
        let strength = strength.clamp(0.0, 1.0);
        self.entrainment = Some(strength);
        self.coupled_spec_mut().coupling = strength;
    }

    fn set_rhythm_role(&mut self, name: &str) {
        let role = match name {
            "beat" => RhythmRole::Beat,
            "subdivision" => RhythmRole::Subdivision,
            "accent" => RhythmRole::Accent,
            "texture" => RhythmRole::Texture,
            other => {
                warn!("unknown rhythm_role '{other}', ignoring");
                return;
            }
        };
        self.rhythm_role = Some(role);
        self.coupled_spec_mut().role = role;
    }

    fn set_microtiming(&mut self, amount: f32) {
        let amount = if amount.is_finite() {
            amount.clamp(-0.5, 0.5)
        } else {
            0.0
        };
        self.microtiming = Some(amount);
        self.coupled_spec_mut().microtiming = amount;
    }

    fn set_when_once(&mut self) {
        self.clear_rhythm_preset_controls();
        self.phonation_spec.timing = PhonationTiming::Once;
    }

    fn set_when_pulse(&mut self, rate: f32) {
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

    fn set_duration_while_alive(&mut self) {
        self.clear_rhythm_preset_controls();
        self.phonation_spec = PhonationSpec {
            timing: PhonationTiming::Once,
            duration: DurationSpec::WhileAlive,
        };
    }

    fn set_duration_cycles(&mut self, n: u32) {
        self.phonation_spec.duration = DurationSpec::Gates(n.max(1));
    }

    fn set_adaptive_duration(&mut self) {
        self.phonation_spec.duration = DurationSpec::Field(self.field_duration_spec.clone());
    }

    fn set_pulse_lock(&mut self, depth: f32) {
        let depth = depth.clamp(0.0, 1.0);
        self.pulse_sync = Some(depth);
        if let PhonationTiming::Pulse { sync, .. } = &mut self.phonation_spec.timing {
            *sync = depth;
        }
    }

    fn set_social(&mut self, coupling: f32) {
        let coupling = coupling.clamp(0.0, 1.0);
        self.social_coupling = Some(coupling);
        match &mut self.phonation_spec.timing {
            PhonationTiming::Coupled(spec) => spec.social = coupling,
            PhonationTiming::Pulse { social, .. } => *social = coupling,
            PhonationTiming::Once => {}
        }
    }

    fn set_duration_range(&mut self, min: f32, max: f32) {
        self.field_duration_spec.hold_min_theta = min.clamp(0.0, 1.0);
        self.field_duration_spec.hold_max_theta = max.clamp(0.0, 1.0);
        self.apply_field_duration_spec();
    }

    fn set_duration_curve(&mut self, k: f32, x0: f32) {
        self.field_duration_spec.curve_k = k;
        self.field_duration_spec.curve_x0 = x0;
        self.apply_field_duration_spec();
    }

    fn set_shorten_on_drop(&mut self, gain: f32) {
        self.field_duration_spec.drop_gain = gain.max(0.0);
        self.apply_field_duration_spec();
    }

    fn set_metabolism(&mut self, rate: f32) {
        self.metabolism_rate = Some(rate.max(0.0));
    }

    fn set_initial_energy(&mut self, value: f32) {
        self.initial_energy = Some(value.max(0.0));
    }

    fn set_recharge_rate(&mut self, value: f32) {
        self.recharge_rate = Some(value.max(0.0));
    }

    fn set_action_cost(&mut self, value: f32) {
        self.action_cost = Some(value.max(0.0));
    }

    fn set_viability_rate(&mut self, value: f32) {
        self.viability_rate = Some(value.max(0.0));
    }

    fn set_consonance_viability(&mut self, low: f32, high: f32) {
        if !low.is_finite() || !high.is_finite() {
            warn!("consonance_viability() expects finite thresholds");
            return;
        }
        self.continuous_recharge_score_low = Some(low);
        self.continuous_recharge_score_high = Some(high);
        self.selection_approx_loo = true;
    }

    fn set_viability_scope(&mut self, name: &str) {
        match name.trim().to_ascii_lowercase().as_str() {
            "environment" => self.selection_approx_loo = true,
            "total" => self.selection_approx_loo = false,
            other => warn!(
                "viability_scope() expects 'environment' or 'total', got '{}'",
                other
            ),
        }
    }

    fn set_selection_approx_loo(&mut self, enabled: bool) {
        self.selection_approx_loo = enabled;
    }

    fn set_dissonance_cost(&mut self, value: f32) {
        self.dissonance_cost = Some(value.max(0.0));
    }

    fn set_energy_cap(&mut self, value: f32) {
        self.energy_cap = Some(value.max(0.0));
    }

    fn set_adsr(&mut self, a: f32, d: f32, s: f32, r: f32) {
        self.control.body.envelope = EnvelopeConfig {
            attack_sec: a.max(0.0),
            decay_sec: d.max(0.0),
            sustain_level: s.clamp(0.0, 1.0),
            release_sec: r.max(0.0),
        };
        self.adsr_user_set = true;
    }

    fn set_respawn_random(&mut self) {
        self.respawn_policy = RespawnPolicy::Random;
    }

    fn set_respawn_hereditary(&mut self, sigma_oct: f32) {
        let sigma_oct = if sigma_oct.is_finite() {
            sigma_oct.max(0.0)
        } else {
            0.0
        };
        self.respawn_policy = RespawnPolicy::Hereditary { sigma_oct };
    }

    fn set_respawn_consonance(&mut self) {
        self.respawn_policy = RespawnPolicy::PeakBiased {
            config: RespawnPeakBiasConfig::default(),
        };
    }

    fn set_respawn_settle_strategy(&mut self, strategy: SpawnStrategy) {
        self.respawn_settle_strategy = Some(strategy);
    }

    fn set_respawn_capacity(&mut self, value: f32) {
        let rounded = if value.is_finite() {
            value.round()
        } else {
            1.0
        };
        self.respawn_capacity = rounded.max(1.0) as usize;
    }

    fn set_respawn_min_c_level(&mut self, value: f32) {
        self.respawn_min_c_level = Some(value.clamp(0.0, 1.0));
    }

    fn set_respawn_background_death_rate(&mut self, value: f32) {
        self.respawn_background_death_rate_per_sec = value.max(0.0);
    }

    fn set_rhythm_coupling_vitality(&mut self, lambda_v: f32, v_floor: f32) {
        self.rhythm_coupling = RhythmCouplingMode::TemporalTimesVitality { lambda_v, v_floor };
    }

    fn set_rhythm_freq(&mut self, value: f32) {
        self.rhythm_freq = Some(value.max(0.0));
    }

    fn set_rhythm_reward(&mut self, rho_t: f32, metric: &str) {
        let lowered = metric.trim().to_ascii_lowercase();
        self.rhythm_reward = match lowered.as_str() {
            "attack_phase_match" | "attackphasematch" | "phase_match" => {
                Some(MetabolismRhythmReward {
                    rho_t,
                    metric: RhythmRewardMetric::AttackPhaseMatch,
                })
            }
            "none" | "off" | "disabled" => None,
            other => {
                warn!(
                    "rhythm_reward() expects 'attack_phase_match' or 'none', got '{}'",
                    other
                );
                self.rhythm_reward
            }
        };
    }
}

#[derive(Clone, Debug)]
pub struct SpeciesHandle {
    spec: SpeciesSpec,
}

#[derive(Clone, Debug)]
pub struct GroupHandle {
    id: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Bus {
    routing: Routing,
}

impl Bus {
    pub fn habitat() -> Self {
        Self {
            routing: Routing {
                to_presentation: false,
                to_habitat: true,
            },
        }
    }

    pub fn presentation() -> Self {
        Self {
            routing: Routing {
                to_presentation: true,
                to_habitat: false,
            },
        }
    }

    fn set(self) -> BusSet {
        BusSet {
            routing: self.routing,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BusSet {
    routing: Routing,
}

impl BusSet {
    fn combine(self, other: Self) -> Self {
        Self {
            routing: Routing {
                to_presentation: self.routing.to_presentation || other.routing.to_presentation,
                to_habitat: self.routing.to_habitat || other.routing.to_habitat,
            },
        }
    }

    fn routing(self) -> Routing {
        self.routing
    }
}

#[derive(Clone, Debug)]
pub struct Placement {
    count: usize,
    strategy: Option<SpawnStrategy>,
    freq_hz: Option<f32>,
}

impl Placement {
    fn at(freq_hz: f32) -> Self {
        Self {
            count: 1,
            strategy: None,
            freq_hz: Some(freq_hz),
        }
    }

    fn density(min_freq: f32, max_freq: f32) -> Self {
        Self {
            count: 1,
            strategy: Some(SpawnStrategy::ConsonanceDensity {
                min_freq,
                max_freq,
                min_dist_erb: 1.0,
            }),
            freq_hz: None,
        }
    }

    fn peaks(root_freq: f32) -> Self {
        Self {
            count: 1,
            strategy: Some(SpawnStrategy::Consonance {
                root_freq,
                min_mul: 1.0,
                max_mul: 4.0,
                min_dist_erb: 1.0,
            }),
            freq_hz: None,
        }
    }

    fn random(min_freq: f32, max_freq: f32) -> Self {
        Self {
            count: 1,
            strategy: Some(SpawnStrategy::RandomLog { min_freq, max_freq }),
            freq_hz: None,
        }
    }

    fn line(start_freq: f32, end_freq: f32) -> Self {
        Self {
            count: 1,
            strategy: Some(SpawnStrategy::Linear {
                start_freq,
                end_freq,
            }),
            freq_hz: None,
        }
    }

    fn with_count(mut self, count: i64) -> Self {
        self.count = count.max(0) as usize;
        self
    }

    fn with_range(mut self, min_mul: f32, max_mul: f32) -> Self {
        if let Some(SpawnStrategy::Consonance {
            root_freq,
            min_dist_erb,
            ..
        }) = self.strategy
        {
            self.strategy = Some(SpawnStrategy::Consonance {
                root_freq,
                min_mul,
                max_mul,
                min_dist_erb,
            });
        } else {
            warn!("range() is only supported for peaks(); ignored");
        }
        self
    }

    fn with_spacing(mut self, spacing: f32) -> Self {
        self.strategy = self.strategy.map(|strategy| match strategy {
            SpawnStrategy::Consonance {
                root_freq,
                min_mul,
                max_mul,
                ..
            } => SpawnStrategy::Consonance {
                root_freq,
                min_mul,
                max_mul,
                min_dist_erb: spacing,
            },
            SpawnStrategy::ConsonanceDensity {
                min_freq, max_freq, ..
            } => SpawnStrategy::ConsonanceDensity {
                min_freq,
                max_freq,
                min_dist_erb: spacing,
            },
            other => {
                warn!("spacing() is only supported for peaks() and density(); ignored");
                other
            }
        });
        self
    }

    fn with_reject_targets(
        mut self,
        anchor_hz: f32,
        targets_st: Vec<f32>,
        exclusion_st: f32,
        max_tries: i64,
    ) -> Self {
        if let Some(base) = self.strategy.take() {
            self.strategy = Some(SpawnStrategy::RejectTargets {
                base: Box::new(base),
                anchor_hz,
                targets_st,
                exclusion_st,
                max_tries: max_tries.max(1) as usize,
            });
        } else {
            warn!("reject_targets() requires density(), peaks(), random(), or line()");
        }
        self
    }

    fn strategy(&self) -> Option<SpawnStrategy> {
        self.strategy.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum GroupStatus {
    Draft,
    Live,
    Released,
    Dropped,
}

#[derive(Clone, Debug)]
struct GroupState {
    id: u64,
    count: usize,
    spec: SpeciesSpec,
    respawn_policy: RespawnPolicy,
    crowding_target_same: bool,
    crowding_target_other: bool,
    strategy: Option<SpawnStrategy>,
    status: GroupStatus,
    live_ids: Vec<u64>,
    pending_patch: ControlUpdate,
    pending_crowding_target: Option<(bool, bool)>,
    pending_release: bool,
}

#[derive(Clone, Debug, Default)]
struct ScriptWarnings {
    draft_dropped: u32,
}

#[derive(Debug, Clone)]
struct ScopeFrame {
    created_groups: Vec<u64>,
}

#[derive(Debug, Clone)]
pub struct ScriptContext {
    pub cursor: f32,
    pub scenario: Scenario,
    pub seed: u64,
    pub next_event_order: u64,
    next_group_id: u64,
    next_voice_id: u64,
    groups: BTreeMap<u64, GroupState>,
    scopes: Vec<ScopeFrame>,
    warnings: ScriptWarnings,
}

impl Default for ScriptContext {
    fn default() -> Self {
        let seed = random::<u64>();
        Self {
            cursor: 0.0,
            scenario: Scenario {
                seed,
                control_update_mode: ControlUpdateMode::SnapshotPhased,
                scaffold: ScaffoldConfig::Off,
                meter_shaping: MeterShaping::default(),
                scene_markers: Vec::new(),
                events: Vec::new(),
                duration_sec: 0.0,
            },
            seed,
            next_event_order: 1,
            next_group_id: 1,
            next_voice_id: 1,
            groups: BTreeMap::new(),
            scopes: Vec::new(),
            warnings: ScriptWarnings::default(),
        }
    }
}

impl ScriptContext {
    fn push_event(&mut self, time_sec: f32, actions: Vec<Action>) {
        if actions.is_empty() {
            return;
        }
        let order = self.next_event_order;
        self.next_event_order += 1;
        self.scenario.events.push(TimedEvent {
            time: time_sec,
            order,
            actions,
        });
    }

    fn push_scene_marker(&mut self, name: &str) {
        let order = self.next_event_order;
        self.next_event_order += 1;
        self.scenario.scene_markers.push(SceneMarker {
            name: name.to_string(),
            time: self.cursor,
            order,
        });
    }

    fn push_scope(&mut self) {
        self.scopes.push(ScopeFrame {
            created_groups: Vec::new(),
        });
    }

    fn pop_scope(&mut self) {
        let Some(scope) = self.scopes.pop() else {
            return;
        };
        let mut releases = Vec::new();
        for group_id in scope.created_groups {
            let Some(group) = self.groups.get_mut(&group_id) else {
                continue;
            };
            match group.status {
                GroupStatus::Draft => {
                    warn!("scope ended with draft group {group_id} (spawn skipped)");
                    self.warnings.draft_dropped += 1;
                    group.status = GroupStatus::Dropped;
                }
                GroupStatus::Live => {
                    if !group.live_ids.is_empty() {
                        releases.push(Action::ReleaseGroup {
                            group_id,
                            fade_sec: group.spec.release_sec(),
                        });
                    }
                    group.pending_patch = ControlUpdate::default();
                    group.pending_crowding_target = None;
                    group.pending_release = false;
                    group.status = GroupStatus::Released;
                }
                GroupStatus::Released | GroupStatus::Dropped => {}
            }
        }
        if !releases.is_empty() {
            self.push_event(self.cursor, releases);
        }
    }

    fn drop_remaining_drafts(&mut self) {
        for (group_id, group) in self.groups.iter_mut() {
            if matches!(group.status, GroupStatus::Draft) {
                warn!("script ended with draft group {group_id} (spawn skipped)");
                self.warnings.draft_dropped += 1;
                group.status = GroupStatus::Dropped;
            }
        }
    }

    fn finalize_duration(&mut self) -> f32 {
        let mut duration = self.cursor.max(0.0);
        if duration <= 0.0 {
            duration = 0.01;
            self.cursor = duration;
        }
        self.scenario.duration_sec = duration;
        duration
    }

    fn max_release_tail(&self) -> f32 {
        let mut max_end: f32 = 0.0;
        for event in &self.scenario.events {
            for action in &event.actions {
                if let Action::ReleaseGroup { fade_sec, .. } = action {
                    let fade = fade_sec.max(0.0);
                    max_end = max_end.max(event.time + fade);
                }
            }
        }
        max_end
    }

    fn commit(&mut self, include_drafts: bool) {
        let mut spawn_actions = Vec::new();
        if include_drafts {
            for group in self.groups.values_mut() {
                if !matches!(group.status, GroupStatus::Draft) {
                    continue;
                }
                let mut ids = Vec::with_capacity(group.count);
                for _ in 0..group.count {
                    let id = self.next_voice_id;
                    self.next_voice_id = self.next_voice_id.wrapping_add(1);
                    ids.push(id);
                }
                group.live_ids = ids.clone();
                group.status = GroupStatus::Live;
                if !ids.is_empty() {
                    spawn_actions.push(Action::Spawn {
                        group_id: group.id,
                        ids,
                        spec: group.spec.spawn_spec(),
                        strategy: group.strategy.clone(),
                    });
                    spawn_actions.push(Action::SetGroupCrowdingTarget {
                        group_id: group.id,
                        same_group_visible: group.crowding_target_same,
                        other_group_visible: group.crowding_target_other,
                    });
                    if !matches!(group.respawn_policy, RespawnPolicy::None) {
                        spawn_actions.push(Action::SetRespawnPolicy {
                            group_id: group.id,
                            policy: group.respawn_policy,
                            settle_strategy: group.spec.respawn_settle_strategy.clone(),
                            capacity: group.spec.respawn_capacity.max(1),
                            min_c_level: group.spec.respawn_min_c_level,
                            background_death_rate_per_sec: group
                                .spec
                                .respawn_background_death_rate_per_sec,
                        });
                    }
                }
            }
        }

        let mut update_actions = Vec::new();
        let mut crowding_target_actions = Vec::new();
        for group in self.groups.values_mut() {
            if !matches!(group.status, GroupStatus::Live) {
                continue;
            }
            if group.pending_patch != ControlUpdate::default() {
                if !group.live_ids.is_empty() {
                    update_actions.push(Action::UpdateGroup {
                        group_id: group.id,
                        patch: group.pending_patch.clone(),
                    });
                }
                group.pending_patch = ControlUpdate::default();
            }
            if let Some((same_group_visible, other_group_visible)) = group.pending_crowding_target {
                crowding_target_actions.push(Action::SetGroupCrowdingTarget {
                    group_id: group.id,
                    same_group_visible,
                    other_group_visible,
                });
                group.pending_crowding_target = None;
            }
        }

        let mut release_actions = Vec::new();
        for group in self.groups.values_mut() {
            if !matches!(group.status, GroupStatus::Live) {
                continue;
            }
            if group.pending_release {
                if !group.live_ids.is_empty() {
                    release_actions.push(Action::ReleaseGroup {
                        group_id: group.id,
                        fade_sec: group.spec.release_sec(),
                    });
                }
                group.pending_release = false;
                group.status = GroupStatus::Released;
            }
        }

        if !spawn_actions.is_empty()
            || !update_actions.is_empty()
            || !crowding_target_actions.is_empty()
            || !release_actions.is_empty()
        {
            let mut actions = Vec::with_capacity(
                spawn_actions.len()
                    + update_actions.len()
                    + crowding_target_actions.len()
                    + release_actions.len(),
            );
            actions.extend(spawn_actions);
            actions.extend(update_actions);
            actions.extend(crowding_target_actions);
            actions.extend(release_actions);
            self.push_event(self.cursor, actions);
        }
    }

    fn finish(&mut self) {
        self.commit(false);
        self.drop_remaining_drafts();
        let mut end_time = self.finalize_duration();
        let release_tail = self.max_release_tail();
        if release_tail > end_time {
            end_time = release_tail;
            self.scenario.duration_sec = end_time;
            self.cursor = end_time;
        }
        self.push_event(end_time, vec![Action::Finish]);
    }

    fn create_group(
        &mut self,
        species: SpeciesHandle,
        count: i64,
        position: Position,
    ) -> Result<GroupHandle, Box<EvalAltResult>> {
        if count < 0 {
            return Err(Box::new(EvalAltResult::ErrorRuntime(
                "create count must be non-negative".into(),
                position,
            )));
        }
        let count = count as usize;
        let id = self.next_group_id;
        self.next_group_id = self.next_group_id.wrapping_add(1);
        let group = GroupState {
            id,
            count,
            respawn_policy: species.spec.respawn_policy,
            crowding_target_same: species.spec.crowding_target_same,
            crowding_target_other: species.spec.crowding_target_other,
            spec: species.spec,
            strategy: None,
            status: GroupStatus::Draft,
            live_ids: Vec::new(),
            pending_patch: ControlUpdate::default(),
            pending_crowding_target: None,
            pending_release: false,
        };
        self.groups.insert(id, group);
        if let Some(scope) = self.scopes.last_mut() {
            scope.created_groups.push(id);
        }
        Ok(GroupHandle { id })
    }

    fn place_material(
        &mut self,
        mut species: SpeciesHandle,
        placement: Placement,
        position: Position,
    ) -> Result<GroupHandle, Box<EvalAltResult>> {
        if let Some(freq_hz) = placement.freq_hz {
            species.spec.set_freq(freq_hz);
        }
        let handle = self.create_group(species, placement.count as i64, position)?;
        if let Some(group) = self.groups.get_mut(&handle.id) {
            group.strategy = placement.strategy;
        }
        Ok(handle)
    }

    fn set_seed(&mut self, seed: i64, position: Position) -> Result<(), Box<EvalAltResult>> {
        if seed < 0 {
            return Err(Box::new(EvalAltResult::ErrorRuntime(
                "seed must be >= 0".into(),
                position,
            )));
        }
        let seed = seed as u64;
        self.seed = seed;
        self.scenario.seed = seed;
        Ok(())
    }

    fn wait(&mut self, sec: f32) {
        self.commit(true);
        self.cursor += sec.max(0.0);
    }

    fn flush(&mut self) {
        self.commit(true);
    }

    fn release_group(&mut self, group_id: u64) {
        let Some(group) = self.groups.get_mut(&group_id) else {
            warn!("release on unknown group {group_id}");
            return;
        };
        match group.status {
            GroupStatus::Draft => {
                warn!("release on draft group {group_id} (spawn skipped)");
                self.warnings.draft_dropped += 1;
                group.status = GroupStatus::Dropped;
            }
            GroupStatus::Live => {
                group.pending_release = true;
            }
            GroupStatus::Released | GroupStatus::Dropped => {
                warn!("release ignored for inactive group {group_id}");
            }
        }
    }

    fn warn_live_builder(&self, group_id: u64, label: &str) {
        warn!("{label} ignored for live group {group_id}");
    }
}

type SpeciesNumericSetter = fn(&mut SpeciesSpec, f32);
type SpeciesPairNumericSetter = fn(&mut SpeciesSpec, f32, f32);
type GroupSpecNumericSetter = fn(&mut SpeciesSpec, f32);
type GroupSpecPairNumericSetter = fn(&mut SpeciesSpec, f32, f32);
type GroupPatchNumericSetter = fn(&mut ControlUpdate, f32);
type GroupDraftHook = fn(&mut GroupState);
type SpeciesNumericMethod = (&'static str, SpeciesNumericSetter);
type SpeciesPairNumericMethod = (&'static str, SpeciesPairNumericSetter);
type GroupNumericMethod = (
    &'static str,
    GroupSpecNumericSetter,
    GroupPatchNumericSetter,
    Option<GroupDraftHook>,
);
type GroupDraftNumericMethod = (&'static str, GroupSpecNumericSetter);
type GroupDraftPairNumericMethod = (&'static str, GroupSpecPairNumericSetter);

fn register_species_numeric_overloads(
    engine: &mut Engine,
    name: &'static str,
    setter: SpeciesNumericSetter,
) {
    engine.register_fn(name, move |mut species: SpeciesHandle, value: FLOAT| {
        setter(&mut species.spec, value as f32);
        species
    });
    engine.register_fn(name, move |mut species: SpeciesHandle, value: INT| {
        setter(&mut species.spec, value as f32);
        species
    });
}

fn register_species_pair_numeric_overloads(
    engine: &mut Engine,
    name: &'static str,
    setter: SpeciesPairNumericSetter,
) {
    engine.register_fn(
        name,
        move |mut species: SpeciesHandle, first: FLOAT, second: FLOAT| {
            setter(&mut species.spec, first as f32, second as f32);
            species
        },
    );
    engine.register_fn(
        name,
        move |mut species: SpeciesHandle, first: INT, second: FLOAT| {
            setter(&mut species.spec, first as f32, second as f32);
            species
        },
    );
    engine.register_fn(
        name,
        move |mut species: SpeciesHandle, first: FLOAT, second: INT| {
            setter(&mut species.spec, first as f32, second as f32);
            species
        },
    );
    engine.register_fn(
        name,
        move |mut species: SpeciesHandle, first: INT, second: INT| {
            setter(&mut species.spec, first as f32, second as f32);
            species
        },
    );
}

fn register_species_numeric_methods(engine: &mut Engine, methods: &[SpeciesNumericMethod]) {
    for &(name, setter) in methods {
        register_species_numeric_overloads(engine, name, setter);
    }
}

fn register_species_pair_numeric_methods(
    engine: &mut Engine,
    methods: &[SpeciesPairNumericMethod],
) {
    for &(name, setter) in methods {
        register_species_pair_numeric_overloads(engine, name, setter);
    }
}

fn apply_group_numeric_patch(
    ctx_arc: &Arc<Mutex<ScriptContext>>,
    handle: GroupHandle,
    label: &'static str,
    value: f32,
    spec_setter: GroupSpecNumericSetter,
    patch_setter: GroupPatchNumericSetter,
    draft_hook: Option<GroupDraftHook>,
) -> Result<GroupHandle, Box<EvalAltResult>> {
    let mut ctx = ctx_arc.lock().expect("lock script context");
    let Some(group) = ctx.groups.get_mut(&handle.id) else {
        warn!("{label} ignored for unknown group {}", handle.id);
        return Ok(handle);
    };
    match group.status {
        GroupStatus::Draft => {
            if let Some(hook) = draft_hook {
                hook(group);
            }
            spec_setter(&mut group.spec, value);
        }
        GroupStatus::Live => {
            spec_setter(&mut group.spec, value);
            patch_setter(&mut group.pending_patch, value);
        }
        GroupStatus::Released | GroupStatus::Dropped => {
            warn!("{label} ignored for inactive group {}", handle.id);
        }
    }
    Ok(handle)
}

fn register_group_numeric_overloads(
    engine: &mut Engine,
    ctx: Arc<Mutex<ScriptContext>>,
    name: &'static str,
    spec_setter: GroupSpecNumericSetter,
    patch_setter: GroupPatchNumericSetter,
    draft_hook: Option<GroupDraftHook>,
) {
    let ctx_float = ctx.clone();
    engine.register_fn(
        name,
        move |handle: GroupHandle, value: FLOAT| -> Result<GroupHandle, Box<EvalAltResult>> {
            apply_group_numeric_patch(
                &ctx_float,
                handle,
                name,
                value as f32,
                spec_setter,
                patch_setter,
                draft_hook,
            )
        },
    );
    engine.register_fn(
        name,
        move |handle: GroupHandle, value: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
            apply_group_numeric_patch(
                &ctx,
                handle,
                name,
                value as f32,
                spec_setter,
                patch_setter,
                draft_hook,
            )
        },
    );
}

fn register_group_numeric_methods(
    engine: &mut Engine,
    ctx: Arc<Mutex<ScriptContext>>,
    methods: &[GroupNumericMethod],
) {
    for &(name, spec_setter, patch_setter, draft_hook) in methods {
        register_group_numeric_overloads(
            engine,
            ctx.clone(),
            name,
            spec_setter,
            patch_setter,
            draft_hook,
        );
    }
}

fn register_group_draft_numeric_overloads(
    engine: &mut Engine,
    ctx: Arc<Mutex<ScriptContext>>,
    name: &'static str,
    spec_setter: GroupSpecNumericSetter,
) {
    let ctx_float = ctx.clone();
    engine.register_fn(
        name,
        move |handle: GroupHandle, value: FLOAT| -> Result<GroupHandle, Box<EvalAltResult>> {
            let mut ctx = ctx_float.lock().expect("lock script context");
            let Some(group) = ctx.groups.get_mut(&handle.id) else {
                warn!("{name} ignored for unknown group {}", handle.id);
                return Ok(handle);
            };
            match group.status {
                GroupStatus::Draft => spec_setter(&mut group.spec, value as f32),
                _ => ctx.warn_live_builder(handle.id, name),
            }
            Ok(handle)
        },
    );
    engine.register_fn(
        name,
        move |handle: GroupHandle, value: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
            let mut ctx = ctx.lock().expect("lock script context");
            let Some(group) = ctx.groups.get_mut(&handle.id) else {
                warn!("{name} ignored for unknown group {}", handle.id);
                return Ok(handle);
            };
            match group.status {
                GroupStatus::Draft => spec_setter(&mut group.spec, value as f32),
                _ => ctx.warn_live_builder(handle.id, name),
            }
            Ok(handle)
        },
    );
}

fn register_group_draft_numeric_methods(
    engine: &mut Engine,
    ctx: Arc<Mutex<ScriptContext>>,
    methods: &[GroupDraftNumericMethod],
) {
    for &(name, spec_setter) in methods {
        register_group_draft_numeric_overloads(engine, ctx.clone(), name, spec_setter);
    }
}

fn register_group_draft_pair_numeric_overloads(
    engine: &mut Engine,
    ctx: Arc<Mutex<ScriptContext>>,
    name: &'static str,
    spec_setter: GroupSpecPairNumericSetter,
) {
    let ctx_ff = ctx.clone();
    engine.register_fn(
        name,
        move |handle: GroupHandle,
              first: FLOAT,
              second: FLOAT|
              -> Result<GroupHandle, Box<EvalAltResult>> {
            let mut ctx = ctx_ff.lock().expect("lock script context");
            let Some(group) = ctx.groups.get_mut(&handle.id) else {
                warn!("{name} ignored for unknown group {}", handle.id);
                return Ok(handle);
            };
            match group.status {
                GroupStatus::Draft => spec_setter(&mut group.spec, first as f32, second as f32),
                _ => ctx.warn_live_builder(handle.id, name),
            }
            Ok(handle)
        },
    );
    let ctx_if = ctx.clone();
    engine.register_fn(
        name,
        move |handle: GroupHandle,
              first: INT,
              second: FLOAT|
              -> Result<GroupHandle, Box<EvalAltResult>> {
            let mut ctx = ctx_if.lock().expect("lock script context");
            let Some(group) = ctx.groups.get_mut(&handle.id) else {
                warn!("{name} ignored for unknown group {}", handle.id);
                return Ok(handle);
            };
            match group.status {
                GroupStatus::Draft => spec_setter(&mut group.spec, first as f32, second as f32),
                _ => ctx.warn_live_builder(handle.id, name),
            }
            Ok(handle)
        },
    );
    let ctx_fi = ctx.clone();
    engine.register_fn(
        name,
        move |handle: GroupHandle,
              first: FLOAT,
              second: INT|
              -> Result<GroupHandle, Box<EvalAltResult>> {
            let mut ctx = ctx_fi.lock().expect("lock script context");
            let Some(group) = ctx.groups.get_mut(&handle.id) else {
                warn!("{name} ignored for unknown group {}", handle.id);
                return Ok(handle);
            };
            match group.status {
                GroupStatus::Draft => spec_setter(&mut group.spec, first as f32, second as f32),
                _ => ctx.warn_live_builder(handle.id, name),
            }
            Ok(handle)
        },
    );
    engine.register_fn(
        name,
        move |handle: GroupHandle,
              first: INT,
              second: INT|
              -> Result<GroupHandle, Box<EvalAltResult>> {
            let mut ctx = ctx.lock().expect("lock script context");
            let Some(group) = ctx.groups.get_mut(&handle.id) else {
                warn!("{name} ignored for unknown group {}", handle.id);
                return Ok(handle);
            };
            match group.status {
                GroupStatus::Draft => spec_setter(&mut group.spec, first as f32, second as f32),
                _ => ctx.warn_live_builder(handle.id, name),
            }
            Ok(handle)
        },
    );
}

fn register_group_draft_pair_numeric_methods(
    engine: &mut Engine,
    ctx: Arc<Mutex<ScriptContext>>,
    methods: &[GroupDraftPairNumericMethod],
) {
    for &(name, spec_setter) in methods {
        register_group_draft_pair_numeric_overloads(engine, ctx.clone(), name, spec_setter);
    }
}

fn apply_group_crowding(
    ctx_arc: &Arc<Mutex<ScriptContext>>,
    handle: GroupHandle,
    label: &'static str,
    strength: f32,
    sigma_cents: Option<f32>,
) -> Result<GroupHandle, Box<EvalAltResult>> {
    let mut ctx = ctx_arc.lock().expect("lock script context");
    let Some(group) = ctx.groups.get_mut(&handle.id) else {
        warn!("{label} ignored for unknown group {}", handle.id);
        return Ok(handle);
    };
    match group.status {
        GroupStatus::Draft => match sigma_cents {
            Some(sigma) => group.spec.set_crowding(strength, sigma),
            None => group.spec.set_crowding_auto_sigma(strength),
        },
        GroupStatus::Live => match sigma_cents {
            Some(sigma) => {
                group.spec.set_crowding(strength, sigma);
                group.pending_patch.crowding_strength = Some(strength);
                group.pending_patch.crowding_sigma_cents = Some(sigma);
            }
            None => {
                group.spec.set_crowding_auto_sigma(strength);
                group.pending_patch.crowding_strength = Some(strength);
            }
        },
        _ => ctx.warn_live_builder(handle.id, label),
    }
    Ok(handle)
}

fn register_group_crowding_overloads(
    engine: &mut Engine,
    ctx: Arc<Mutex<ScriptContext>>,
    name: &'static str,
) {
    let ctx_float_float = ctx.clone();
    engine.register_fn(
        name,
        move |handle: GroupHandle,
              strength: FLOAT,
              sigma_cents: FLOAT|
              -> Result<GroupHandle, Box<EvalAltResult>> {
            apply_group_crowding(
                &ctx_float_float,
                handle,
                name,
                strength as f32,
                Some(sigma_cents as f32),
            )
        },
    );
    let ctx_int_float = ctx.clone();
    engine.register_fn(
        name,
        move |handle: GroupHandle,
              strength: INT,
              sigma_cents: FLOAT|
              -> Result<GroupHandle, Box<EvalAltResult>> {
            apply_group_crowding(
                &ctx_int_float,
                handle,
                name,
                strength as f32,
                Some(sigma_cents as f32),
            )
        },
    );
    let ctx_float_int = ctx.clone();
    engine.register_fn(
        name,
        move |handle: GroupHandle,
              strength: FLOAT,
              sigma_cents: INT|
              -> Result<GroupHandle, Box<EvalAltResult>> {
            apply_group_crowding(
                &ctx_float_int,
                handle,
                name,
                strength as f32,
                Some(sigma_cents as f32),
            )
        },
    );
    let ctx_int_int = ctx.clone();
    engine.register_fn(
        name,
        move |handle: GroupHandle,
              strength: INT,
              sigma_cents: INT|
              -> Result<GroupHandle, Box<EvalAltResult>> {
            apply_group_crowding(
                &ctx_int_int,
                handle,
                name,
                strength as f32,
                Some(sigma_cents as f32),
            )
        },
    );
    let ctx_float = ctx.clone();
    engine.register_fn(
        name,
        move |handle: GroupHandle, strength: FLOAT| -> Result<GroupHandle, Box<EvalAltResult>> {
            apply_group_crowding(&ctx_float, handle, name, strength as f32, None)
        },
    );
    engine.register_fn(
        name,
        move |handle: GroupHandle, strength: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
            apply_group_crowding(&ctx, handle, name, strength as f32, None)
        },
    );
}

fn patch_amp(update: &mut ControlUpdate, value: f32) {
    update.amp = Some(value);
}

fn patch_freq(update: &mut ControlUpdate, value: f32) {
    update.freq = Some(value);
}

fn patch_landscape_weight(update: &mut ControlUpdate, value: f32) {
    update.landscape_weight = Some(value);
}

fn patch_neighbor_step_cents(update: &mut ControlUpdate, value: f32) {
    update.neighbor_step_cents = Some(value);
}

fn patch_tessitura_gravity(update: &mut ControlUpdate, value: f32) {
    update.tessitura_gravity = Some(value);
}

fn patch_continuous_drive(update: &mut ControlUpdate, value: f32) {
    update.continuous_drive = Some(value);
}

fn patch_pitch_smooth_tau(update: &mut ControlUpdate, value: f32) {
    update.pitch_smooth_tau = Some(value);
}

fn patch_exploration(update: &mut ControlUpdate, value: f32) {
    update.exploration = Some(value);
}

fn patch_persistence(update: &mut ControlUpdate, value: f32) {
    update.persistence = Some(value);
}

fn patch_leave_self_out_mode(update: &mut ControlUpdate, mode: LeaveSelfOutMode) {
    update.leave_self_out_mode = Some(mode);
}

fn patch_anneal_temp(update: &mut ControlUpdate, value: f32) {
    update.anneal_temp = Some(value);
}

fn patch_move_cost_coeff(update: &mut ControlUpdate, value: f32) {
    update.move_cost_coeff = Some(value);
}

fn patch_move_cost_exp(update: &mut ControlUpdate, value: f32) {
    update.move_cost_exp = Some(value.round() as i64);
}

fn patch_improvement_threshold(update: &mut ControlUpdate, value: f32) {
    update.improvement_threshold = Some(value);
}

fn patch_proposal_interval(update: &mut ControlUpdate, value: f32) {
    update.proposal_interval_sec = Some(value);
}

fn patch_window_cents(update: &mut ControlUpdate, value: f32) {
    update.window_cents = Some(value);
}

fn patch_top_k(update: &mut ControlUpdate, value: f32) {
    update.top_k = Some(value.round() as i64);
}

fn patch_temperature(update: &mut ControlUpdate, value: f32) {
    update.temperature = Some(value);
}

fn patch_sigma_cents(update: &mut ControlUpdate, value: f32) {
    update.sigma_cents = Some(value);
}

fn patch_random_candidates(update: &mut ControlUpdate, value: f32) {
    update.random_candidates = Some(value.round() as i64);
}

fn patch_pitch_glide_tau(update: &mut ControlUpdate, value: f32) {
    update.pitch_glide_tau_sec = Some(value);
}

fn patch_timbre_brightness(update: &mut ControlUpdate, value: f32) {
    update.timbre_brightness = Some(value);
}

fn patch_timbre_spread(update: &mut ControlUpdate, value: f32) {
    update.timbre_spread = Some(value);
}

fn patch_timbre_unison(update: &mut ControlUpdate, value: f32) {
    update.timbre_unison = Some(value.round() as i64);
}

fn draft_clear_strategy(group: &mut GroupState) {
    group.strategy = None;
}

pub struct ScriptHost;

pub mod defs_gen;
pub mod docs;
mod engine;

#[derive(Debug, Clone)]
pub struct ScriptError {
    pub message: String,
    pub position: Option<Position>,
}

impl ScriptError {
    pub fn new(message: impl Into<String>, position: Option<Position>) -> Self {
        Self {
            message: message.into(),
            position,
        }
    }

    pub fn with_context(mut self, context: impl AsRef<str>) -> Self {
        let ctx = context.as_ref();
        self.message = format!("{ctx}: {}", self.message);
        self
    }

    pub fn from_eval(err: Box<EvalAltResult>, context: Option<&str>) -> Self {
        let pos = err.position();
        let position = if pos == Position::NONE {
            None
        } else {
            Some(pos)
        };
        let mut err = ScriptError::new(err.to_string(), position);
        if let Some(ctx) = context {
            err = err.with_context(ctx);
        }
        err
    }
}

impl std::fmt::Display for ScriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(pos) = self.position {
            let line = pos.line().unwrap_or(0);
            write!(f, "{} (line {line})", self.message)
        } else {
            write!(f, "{}", self.message)
        }
    }
}

impl std::error::Error for ScriptError {}

#[cfg(test)]
mod tests;
