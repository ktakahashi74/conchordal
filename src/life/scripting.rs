use std::collections::BTreeMap;
use std::fs;
use std::sync::{Arc, Mutex};

use rand::random;
use rhai::{Array, Dynamic, Engine, EvalAltResult, FLOAT, FnPtr, INT, NativeCallContext, Position};
use tracing::warn;

use crate::core::landscape::PitchObjectiveMode;
use crate::core::mode_pattern::ModePattern;

use super::control::{
    AgentControl, BodyMethod, ControlUpdate, LeaveSelfOutMode, MoveCostTimeScale, PitchApplyMode,
    PitchCoreKind, PitchMode,
};
use super::lifecycle::LifecycleConfig;
use super::scenario::{
    Action, ArticulationCoreConfig, DurationSpec, EnvelopeConfig, FieldDurationSpec,
    MetabolismRhythmReward, PhonationSpec, RespawnPolicy, RhythmCouplingMode, RhythmRewardMetric,
    Scenario, SceneMarker, SpawnSpec, SpawnStrategy, TimedEvent, WhenSpec,
};

const DEFAULT_RELEASE_SEC: f32 = 0.05;
const DEFAULT_SEQ_DURATION_SEC: f32 = 1.0;
const DEFAULT_PULSE_RATE: f32 = 2.25;
const DEFAULT_PULSE_SYNC: f32 = 0.5;
const DEFAULT_GATE_COUNT: u32 = 5;

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

#[derive(Clone, Copy, Debug)]
struct AdsrSpec {
    attack_sec: f32,
    decay_sec: f32,
    sustain_level: f32,
    release_sec: f32,
}

#[derive(Clone, Debug)]
struct SpeciesSpec {
    control: AgentControl,
    respawn_policy: RespawnPolicy,
    crowding_target_same: bool,
    crowding_target_other: bool,
    brain: BrainKind,
    phonation_spec: PhonationSpec,
    metabolism_rate: Option<f32>,
    adsr: Option<AdsrSpec>,
    rhythm_coupling: RhythmCouplingMode,
    rhythm_reward: Option<MetabolismRhythmReward>,
    telemetry_first_k: Option<u32>,
    plv_window: Option<usize>,
}

impl SpeciesSpec {
    fn preset(body: BodyMethod) -> Self {
        let mut control = AgentControl::default();
        control.body.method = body;
        Self {
            control,
            respawn_policy: RespawnPolicy::None,
            crowding_target_same: true,
            crowding_target_other: false,
            brain: BrainKind::Entrain,
            phonation_spec: PhonationSpec::default(),
            metabolism_rate: None,
            adsr: None,
            rhythm_coupling: RhythmCouplingMode::TemporalOnly,
            rhythm_reward: None,
            telemetry_first_k: None,
            plv_window: None,
        }
    }

    fn release_sec(&self) -> f32 {
        self.adsr
            .map(|adsr| adsr.release_sec)
            .unwrap_or(DEFAULT_RELEASE_SEC)
            .max(0.0)
    }

    fn envelope_from_adsr(&self) -> EnvelopeConfig {
        if let Some(adsr) = self.adsr {
            EnvelopeConfig {
                attack_sec: adsr.attack_sec.max(0.0),
                decay_sec: adsr.decay_sec.max(0.0),
                sustain_level: adsr.sustain_level.clamp(0.0, 1.0),
            }
        } else {
            EnvelopeConfig::default()
        }
    }

    fn lifecycle_config(&self) -> LifecycleConfig {
        if self.metabolism_rate.is_some() || self.adsr.is_some() {
            let metabolism_rate = self.metabolism_rate.unwrap_or(0.5).max(1e-6);
            LifecycleConfig::Sustain {
                initial_energy: 1.0,
                metabolism_rate,
                recharge_rate: None,
                action_cost: None,
                envelope: self.envelope_from_adsr(),
            }
        } else {
            LifecycleConfig::default()
        }
    }

    fn articulation_config(&self) -> ArticulationCoreConfig {
        match self.brain {
            BrainKind::Entrain => ArticulationCoreConfig::Entrain {
                lifecycle: self.lifecycle_config(),
                rhythm_freq: None,
                rhythm_sensitivity: None,
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

    fn spawn_spec(&self) -> SpawnSpec {
        let mut control = self.control.clone();
        control.phonation.spec = self.phonation_spec.clone();
        SpawnSpec {
            control,
            articulation: self.articulation_config(),
        }
    }

    fn set_amp(&mut self, amp: f32) {
        self.control.set_amp_clamped(amp);
    }

    fn set_freq(&mut self, freq: f32) {
        self.control.set_freq_lock_clamped(freq);
    }

    fn set_landscape_weight(&mut self, value: f32) {
        self.control.set_landscape_weight_clamped(value);
    }

    fn set_neighbor_step_cents(&mut self, value: f32) {
        self.control.set_neighbor_step_cents_clamped(value);
    }

    fn set_tessitura_gravity(&mut self, value: f32) {
        self.control.set_tessitura_gravity_clamped(value);
    }

    fn set_continuous_drive(&mut self, value: f32) {
        self.control.set_continuous_drive_clamped(value);
    }

    fn set_pitch_smooth_tau(&mut self, value: f32) {
        self.control.set_pitch_smooth_tau_clamped(value);
    }

    fn set_exploration(&mut self, value: f32) {
        self.control.set_exploration_clamped(value);
    }

    fn set_persistence(&mut self, value: f32) {
        self.control.set_persistence_clamped(value);
    }

    fn set_crowding(&mut self, strength: f32, sigma_cents: f32) {
        self.control.set_crowding_strength_clamped(strength);
        self.control.set_crowding_sigma_cents_clamped(sigma_cents);
    }

    fn set_crowding_auto_sigma(&mut self, strength: f32) {
        self.control.set_crowding_strength_clamped(strength);
        self.control.set_crowding_sigma_from_roughness(true);
    }

    fn set_crowding_target(&mut self, same_group_visible: bool, other_group_visible: bool) {
        self.crowding_target_same = same_group_visible;
        self.crowding_target_other = other_group_visible;
    }

    fn set_leave_self_out(&mut self, enabled: bool) {
        self.control.set_leave_self_out(enabled);
    }

    fn set_leave_self_out_mode(&mut self, name: &str) {
        let mode = parse_leave_self_out_mode_name(self.control.pitch.leave_self_out_mode, name);
        self.control.set_leave_self_out_mode(mode);
    }

    fn set_anneal_temp(&mut self, value: f32) {
        self.control.set_anneal_temp_clamped(value);
    }

    fn set_move_cost_coeff(&mut self, value: f32) {
        self.control.set_move_cost_coeff_clamped(value);
    }

    fn set_move_cost_exp(&mut self, value: f32) {
        self.control.set_move_cost_exp_clamped(value.round() as i64);
    }

    fn set_improvement_threshold(&mut self, value: f32) {
        self.control.set_improvement_threshold_clamped(value);
    }

    fn set_proposal_interval_sec(&mut self, value: f32) {
        self.control.set_proposal_interval_sec_clamped(value);
    }

    fn set_global_peaks(&mut self, count: i64, min_sep_cents: f32) {
        self.control.set_global_peak_count_clamped(count);
        self.control
            .set_global_peak_min_sep_cents_clamped(min_sep_cents);
    }

    fn set_ratio_candidates(&mut self, count: i64) {
        self.control.set_ratio_candidate_count_clamped(count);
        self.control.set_use_ratio_candidates(count > 0);
    }

    fn set_window_cents(&mut self, value: f32) {
        self.control.set_window_cents_clamped(value);
    }

    fn set_top_k(&mut self, value: f32) {
        self.control.set_top_k_clamped(value.round() as i64);
    }

    fn set_temperature(&mut self, value: f32) {
        self.control.set_temperature_clamped(value);
    }

    fn set_sigma_cents(&mut self, value: f32) {
        self.control.set_sigma_cents_clamped(value);
    }

    fn set_random_candidates(&mut self, value: f32) {
        self.control
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
        self.control.set_move_cost_time_scale(value);
    }

    fn set_leave_self_out_harmonics(&mut self, value: i64) {
        self.control.set_leave_self_out_harmonics_clamped(value);
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
                self.control.pitch.pitch_apply_mode
            }
        };
        self.control.set_pitch_apply_mode(mode);
    }

    fn set_pitch_glide_tau_sec(&mut self, value: f32) {
        self.control.set_pitch_glide_tau_sec_clamped(value);
    }

    fn set_pitch_mode(&mut self, name: &str) {
        let lowered = name.trim().to_ascii_lowercase();
        self.control.pitch.mode = match lowered.as_str() {
            "free" => PitchMode::Free,
            "lock" => PitchMode::Lock,
            other => {
                warn!("pitch_mode() expects 'free' or 'lock', got '{}'", other);
                self.control.pitch.mode
            }
        };
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
        self.phonation_spec = match kind {
            PhonationKind::Sustain => PhonationSpec::default(),
            PhonationKind::Repeat => PhonationSpec {
                when: WhenSpec::Pulse {
                    rate: DEFAULT_PULSE_RATE,
                    sync: DEFAULT_PULSE_SYNC,
                    social: 0.0,
                },
                duration: DurationSpec::Gates(DEFAULT_GATE_COUNT),
            },
        };
    }

    fn set_when_once(&mut self) {
        self.phonation_spec.when = WhenSpec::Once;
    }

    fn set_when_pulse(&mut self, rate: f32) {
        let rate = rate.max(0.01);
        match &mut self.phonation_spec.when {
            WhenSpec::Pulse { rate: r, .. } => *r = rate,
            _ => {
                self.phonation_spec.when = WhenSpec::Pulse {
                    rate,
                    sync: DEFAULT_PULSE_SYNC,
                    social: 0.0,
                };
            }
        }
    }

    fn set_duration_while_alive(&mut self) {
        self.phonation_spec.duration = DurationSpec::WhileAlive;
    }

    fn set_duration_gates(&mut self, n: u32) {
        self.phonation_spec.duration = DurationSpec::Gates(n.max(1));
    }

    fn set_duration_field(&mut self) {
        self.phonation_spec.duration = DurationSpec::Field(FieldDurationSpec::default());
    }

    fn set_sync(&mut self, depth: f32) {
        match &mut self.phonation_spec.when {
            WhenSpec::Pulse { sync, .. } => *sync = depth.clamp(0.0, 1.0),
            _ => warn!("sync() requires pulse(); ignored"),
        }
    }

    fn set_social(&mut self, coupling: f32) {
        match &mut self.phonation_spec.when {
            WhenSpec::Pulse { social, .. } => *social = coupling.clamp(0.0, 1.0),
            _ => warn!("social() requires pulse(); ignored"),
        }
    }

    fn set_field_window(&mut self, min: f32, max: f32) {
        match &mut self.phonation_spec.duration {
            DurationSpec::Field(f) => {
                f.hold_min_theta = min.clamp(0.0, 1.0);
                f.hold_max_theta = max.clamp(0.0, 1.0);
            }
            _ => warn!("field_window() requires field(); ignored"),
        }
    }

    fn set_field_curve(&mut self, k: f32, x0: f32) {
        match &mut self.phonation_spec.duration {
            DurationSpec::Field(f) => {
                f.curve_k = k;
                f.curve_x0 = x0;
            }
            _ => warn!("field_curve() requires field(); ignored"),
        }
    }

    fn set_field_drop(&mut self, gain: f32) {
        match &mut self.phonation_spec.duration {
            DurationSpec::Field(f) => f.drop_gain = gain.max(0.0),
            _ => warn!("field_drop() requires field(); ignored"),
        }
    }

    fn set_metabolism(&mut self, rate: f32) {
        self.metabolism_rate = Some(rate.max(0.0));
    }

    fn set_adsr(&mut self, a: f32, d: f32, s: f32, r: f32) {
        self.adsr = Some(AdsrSpec {
            attack_sec: a.max(0.0),
            decay_sec: d.max(0.0),
            sustain_level: s.clamp(0.0, 1.0),
            release_sec: r.max(0.0),
        });
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

    fn set_rhythm_coupling(&mut self, mode: &str) {
        let lowered = mode.trim().to_ascii_lowercase();
        self.rhythm_coupling = match lowered.as_str() {
            "temporal" | "temporal_only" | "temporalonly" => RhythmCouplingMode::TemporalOnly,
            other => {
                warn!(
                    "rhythm_coupling() expects 'temporal'; use rhythm_coupling_vitality() for vitality modulation, got '{}'",
                    other
                );
                self.rhythm_coupling
            }
        };
    }

    fn set_rhythm_coupling_vitality(&mut self, lambda_v: f32, v_floor: f32) {
        self.rhythm_coupling = RhythmCouplingMode::TemporalTimesVitality { lambda_v, v_floor };
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
    next_agent_id: u64,
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
                scene_markers: Vec::new(),
                events: Vec::new(),
                duration_sec: 0.0,
            },
            seed,
            next_event_order: 1,
            next_group_id: 1,
            next_agent_id: 1,
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
                    let id = self.next_agent_id;
                    self.next_agent_id = self.next_agent_id.wrapping_add(1);
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
                        });
                    }
                    if let Some(first_k) = group.spec.telemetry_first_k {
                        spawn_actions.push(Action::EnableTelemetry {
                            group_id: group.id,
                            first_k,
                        });
                    }
                    if let Some(window) = group.spec.plv_window {
                        spawn_actions.push(Action::EnablePlv {
                            group_id: group.id,
                            window,
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
type GroupPatchNumericSetter = fn(&mut ControlUpdate, f32);
type GroupDraftHook = fn(&mut GroupState);

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
                group.pending_patch.crowding_sigma_from_roughness = Some(false);
            }
            None => {
                group.spec.set_crowding_auto_sigma(strength);
                group.pending_patch.crowding_strength = Some(strength);
                group.pending_patch.crowding_sigma_from_roughness = Some(true);
            }
        },
        _ => ctx.warn_live_builder(handle.id, label),
    }
    Ok(handle)
}

fn register_group_crowding_overloads(engine: &mut Engine, ctx: Arc<Mutex<ScriptContext>>) {
    let ctx_float_float = ctx.clone();
    engine.register_fn(
        "crowding",
        move |handle: GroupHandle,
              strength: FLOAT,
              sigma_cents: FLOAT|
              -> Result<GroupHandle, Box<EvalAltResult>> {
            apply_group_crowding(
                &ctx_float_float,
                handle,
                "crowding",
                strength as f32,
                Some(sigma_cents as f32),
            )
        },
    );
    let ctx_int_float = ctx.clone();
    engine.register_fn(
        "crowding",
        move |handle: GroupHandle,
              strength: INT,
              sigma_cents: FLOAT|
              -> Result<GroupHandle, Box<EvalAltResult>> {
            apply_group_crowding(
                &ctx_int_float,
                handle,
                "crowding",
                strength as f32,
                Some(sigma_cents as f32),
            )
        },
    );
    let ctx_float_int = ctx.clone();
    engine.register_fn(
        "crowding",
        move |handle: GroupHandle,
              strength: FLOAT,
              sigma_cents: INT|
              -> Result<GroupHandle, Box<EvalAltResult>> {
            apply_group_crowding(
                &ctx_float_int,
                handle,
                "crowding",
                strength as f32,
                Some(sigma_cents as f32),
            )
        },
    );
    let ctx_int_int = ctx.clone();
    engine.register_fn(
        "crowding",
        move |handle: GroupHandle,
              strength: INT,
              sigma_cents: INT|
              -> Result<GroupHandle, Box<EvalAltResult>> {
            apply_group_crowding(
                &ctx_int_int,
                handle,
                "crowding",
                strength as f32,
                Some(sigma_cents as f32),
            )
        },
    );
    let ctx_float = ctx.clone();
    engine.register_fn(
        "crowding",
        move |handle: GroupHandle, strength: FLOAT| -> Result<GroupHandle, Box<EvalAltResult>> {
            apply_group_crowding(&ctx_float, handle, "crowding", strength as f32, None)
        },
    );
    engine.register_fn(
        "crowding",
        move |handle: GroupHandle, strength: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
            apply_group_crowding(&ctx, handle, "crowding", strength as f32, None)
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

fn draft_clear_strategy(group: &mut GroupState) {
    group.strategy = None;
}

pub struct ScriptHost;

impl ScriptHost {
    fn create_engine(ctx: Arc<Mutex<ScriptContext>>) -> Engine {
        let mut engine = Engine::new();
        engine.on_print(|msg| println!("[rhai] {msg}"));

        engine.register_type_with_name::<SpeciesHandle>("SpeciesHandle");
        engine.register_type_with_name::<GroupHandle>("GroupHandle");
        engine.register_type_with_name::<SpawnStrategy>("SpawnStrategy");
        engine.register_type_with_name::<ModePattern>("ModePattern");

        let mut presets = rhai::Module::new();
        presets.set_var(
            "sine",
            SpeciesHandle {
                spec: SpeciesSpec::preset(BodyMethod::Sine),
            },
        );
        presets.set_var(
            "harmonic",
            SpeciesHandle {
                spec: SpeciesSpec::preset(BodyMethod::Harmonic),
            },
        );
        presets.set_var(
            "saw",
            SpeciesHandle {
                spec: {
                    let mut spec = SpeciesSpec::preset(BodyMethod::Harmonic);
                    spec.control.body.timbre.brightness = 0.85;
                    spec
                },
            },
        );
        presets.set_var(
            "square",
            SpeciesHandle {
                spec: {
                    let mut spec = SpeciesSpec::preset(BodyMethod::Harmonic);
                    spec.control.body.timbre.brightness = 0.65;
                    spec
                },
            },
        );
        presets.set_var(
            "noise",
            SpeciesHandle {
                spec: {
                    let mut spec = SpeciesSpec::preset(BodyMethod::Harmonic);
                    spec.control.body.timbre.brightness = 1.0;
                    spec.control.body.timbre.motion = 1.0;
                    spec
                },
            },
        );
        presets.set_var(
            "modal",
            SpeciesHandle {
                spec: SpeciesSpec::preset(BodyMethod::Modal),
            },
        );
        engine.register_global_module(presets.into());

        engine.register_fn("derive", |parent: SpeciesHandle| parent);

        register_species_numeric_overloads(&mut engine, "amp", SpeciesSpec::set_amp);
        register_species_numeric_overloads(&mut engine, "freq", SpeciesSpec::set_freq);
        register_species_numeric_overloads(
            &mut engine,
            "landscape_weight",
            SpeciesSpec::set_landscape_weight,
        );
        register_species_numeric_overloads(
            &mut engine,
            "neighbor_step_cents",
            SpeciesSpec::set_neighbor_step_cents,
        );
        register_species_numeric_overloads(
            &mut engine,
            "tessitura_gravity",
            SpeciesSpec::set_tessitura_gravity,
        );
        register_species_numeric_overloads(
            &mut engine,
            "sustain_drive",
            SpeciesSpec::set_continuous_drive,
        );
        register_species_numeric_overloads(
            &mut engine,
            "pitch_smooth",
            SpeciesSpec::set_pitch_smooth_tau,
        );
        register_species_numeric_overloads(
            &mut engine,
            "exploration",
            SpeciesSpec::set_exploration,
        );
        register_species_numeric_overloads(
            &mut engine,
            "persistence",
            SpeciesSpec::set_persistence,
        );
        register_species_pair_numeric_overloads(&mut engine, "crowding", SpeciesSpec::set_crowding);
        engine.register_fn("crowding", |mut species: SpeciesHandle, strength: FLOAT| {
            species.spec.set_crowding_auto_sigma(strength as f32);
            species
        });
        engine.register_fn("crowding", |mut species: SpeciesHandle, strength: INT| {
            species.spec.set_crowding_auto_sigma(strength as f32);
            species
        });
        engine.register_fn(
            "crowding_target",
            |mut species: SpeciesHandle, same_group_visible: bool, other_group_visible: bool| {
                species
                    .spec
                    .set_crowding_target(same_group_visible, other_group_visible);
                species
            },
        );
        engine.register_fn(
            "leave_self_out",
            |mut species: SpeciesHandle, enabled: bool| {
                species.spec.set_leave_self_out(enabled);
                species
            },
        );
        engine.register_fn(
            "leave_self_out_mode",
            |mut species: SpeciesHandle, name: &str| {
                species.spec.set_leave_self_out_mode(name);
                species
            },
        );
        register_species_numeric_overloads(
            &mut engine,
            "anneal_temp",
            SpeciesSpec::set_anneal_temp,
        );
        register_species_numeric_overloads(
            &mut engine,
            "move_cost",
            SpeciesSpec::set_move_cost_coeff,
        );
        register_species_numeric_overloads(
            &mut engine,
            "move_cost_exp",
            SpeciesSpec::set_move_cost_exp,
        );
        register_species_numeric_overloads(
            &mut engine,
            "improvement_threshold",
            SpeciesSpec::set_improvement_threshold,
        );
        register_species_numeric_overloads(
            &mut engine,
            "proposal_interval",
            SpeciesSpec::set_proposal_interval_sec,
        );
        engine.register_fn("global_peaks", |mut species: SpeciesHandle, count: INT| {
            species.spec.set_global_peaks(count, 0.0);
            species
        });
        engine.register_fn(
            "global_peaks",
            |mut species: SpeciesHandle, count: INT, min_sep_cents: FLOAT| {
                species.spec.set_global_peaks(count, min_sep_cents as f32);
                species
            },
        );
        engine.register_fn(
            "global_peaks",
            |mut species: SpeciesHandle, count: INT, min_sep_cents: INT| {
                species.spec.set_global_peaks(count, min_sep_cents as f32);
                species
            },
        );
        engine.register_fn(
            "ratio_candidates",
            |mut species: SpeciesHandle, count: INT| {
                species.spec.set_ratio_candidates(count);
                species
            },
        );
        register_species_numeric_overloads(
            &mut engine,
            "window_cents",
            SpeciesSpec::set_window_cents,
        );
        register_species_numeric_overloads(&mut engine, "top_k", SpeciesSpec::set_top_k);
        register_species_numeric_overloads(
            &mut engine,
            "temperature",
            SpeciesSpec::set_temperature,
        );
        register_species_numeric_overloads(
            &mut engine,
            "sigma_cents",
            SpeciesSpec::set_sigma_cents,
        );
        register_species_numeric_overloads(
            &mut engine,
            "random_candidates",
            SpeciesSpec::set_random_candidates,
        );
        engine.register_fn(
            "move_cost_time_scale",
            |mut species: SpeciesHandle, name: &str| {
                species.spec.set_move_cost_time_scale(name);
                species
            },
        );
        engine.register_fn(
            "leave_self_out_harmonics",
            |mut species: SpeciesHandle, value: INT| {
                species.spec.set_leave_self_out_harmonics(value);
                species
            },
        );
        engine.register_fn(
            "pitch_apply_mode",
            |mut species: SpeciesHandle, name: &str| {
                species.spec.set_pitch_apply_mode(name);
                species
            },
        );
        engine.register_fn("pitch_apply", |mut species: SpeciesHandle, name: &str| {
            species.spec.set_pitch_apply_mode(name);
            species
        });
        register_species_numeric_overloads(
            &mut engine,
            "pitch_glide",
            SpeciesSpec::set_pitch_glide_tau_sec,
        );
        engine.register_fn("pitch_mode", |mut species: SpeciesHandle, name: &str| {
            species.spec.set_pitch_mode(name);
            species
        });
        engine.register_fn("mode", |mut species: SpeciesHandle, name: &str| {
            species.spec.set_pitch_mode(name);
            species
        });
        engine.register_fn("pitch_core", |mut species: SpeciesHandle, name: &str| {
            species.spec.set_pitch_core(name);
            species
        });
        engine.register_fn("brain", |mut species: SpeciesHandle, name: &str| {
            species.spec.set_brain(name);
            species
        });
        engine.register_fn("sustain", |mut species: SpeciesHandle| {
            species.spec.set_phonation(PhonationKind::Sustain);
            species
        });
        engine.register_fn("repeat", |mut species: SpeciesHandle| {
            species.spec.set_phonation(PhonationKind::Repeat);
            species
        });
        // Tier 2: explicit when/duration
        engine.register_fn("once", |mut species: SpeciesHandle| {
            species.spec.set_when_once();
            species
        });
        engine.register_fn("pulse", |mut species: SpeciesHandle, rate: FLOAT| {
            species.spec.set_when_pulse(rate as f32);
            species
        });
        engine.register_fn("while_alive", |mut species: SpeciesHandle| {
            species.spec.set_duration_while_alive();
            species
        });
        engine.register_fn("gates", |mut species: SpeciesHandle, n: INT| {
            species.spec.set_duration_gates(n.max(1) as u32);
            species
        });
        engine.register_fn("field", |mut species: SpeciesHandle| {
            species.spec.set_duration_field();
            species
        });
        // Tier 3: expert tuning
        engine.register_fn("sync", |mut species: SpeciesHandle, depth: FLOAT| {
            species.spec.set_sync(depth as f32);
            species
        });
        engine.register_fn("social", |mut species: SpeciesHandle, coupling: FLOAT| {
            species.spec.set_social(coupling as f32);
            species
        });
        engine.register_fn(
            "field_window",
            |mut species: SpeciesHandle, min: FLOAT, max: FLOAT| {
                species.spec.set_field_window(min as f32, max as f32);
                species
            },
        );
        engine.register_fn(
            "field_curve",
            |mut species: SpeciesHandle, k: FLOAT, x0: FLOAT| {
                species.spec.set_field_curve(k as f32, x0 as f32);
                species
            },
        );
        engine.register_fn("field_drop", |mut species: SpeciesHandle, gain: FLOAT| {
            species.spec.set_field_drop(gain as f32);
            species
        });
        register_species_numeric_overloads(&mut engine, "energy", SpeciesSpec::set_amp);
        register_species_numeric_overloads(&mut engine, "brightness", SpeciesSpec::set_brightness);
        engine.register_fn(
            "modes",
            |mut species: SpeciesHandle, pattern: ModePattern| {
                species.spec.set_modes(pattern);
                species
            },
        );
        engine.register_fn("metabolism", |mut species: SpeciesHandle, rate: FLOAT| {
            species.spec.set_metabolism(rate as f32);
            species
        });
        engine.register_fn(
            "adsr",
            |mut species: SpeciesHandle, a: FLOAT, d: FLOAT, s: FLOAT, r: FLOAT| {
                species
                    .spec
                    .set_adsr(a as f32, d as f32, s as f32, r as f32);
                species
            },
        );
        engine.register_fn(
            "rhythm_coupling",
            |mut species: SpeciesHandle, mode: &str| {
                species.spec.set_rhythm_coupling(mode);
                species
            },
        );
        engine.register_fn(
            "rhythm_coupling_vitality",
            |mut species: SpeciesHandle, lambda_v: FLOAT, v_floor: FLOAT| {
                species
                    .spec
                    .set_rhythm_coupling_vitality(lambda_v as f32, v_floor as f32);
                species
            },
        );
        engine.register_fn(
            "rhythm_reward",
            |mut species: SpeciesHandle, rho_t: FLOAT, metric: &str| {
                species.spec.set_rhythm_reward(rho_t as f32, metric);
                species
            },
        );
        engine.register_fn("respawn_random", |mut species: SpeciesHandle| {
            species.spec.set_respawn_random();
            species
        });
        engine.register_fn(
            "respawn_hereditary",
            |mut species: SpeciesHandle, sigma_oct: FLOAT| {
                species.spec.set_respawn_hereditary(sigma_oct as f32);
                species
            },
        );
        engine.register_fn(
            "respawn_hereditary",
            |mut species: SpeciesHandle, sigma_oct: INT| {
                species.spec.set_respawn_hereditary(sigma_oct as f32);
                species
            },
        );
        engine.register_fn(
            "enable_telemetry",
            |mut species: SpeciesHandle, first_k: INT| {
                species.spec.telemetry_first_k = Some(first_k.max(0) as u32);
                species
            },
        );
        engine.register_fn("enable_plv", |mut species: SpeciesHandle, window: INT| {
            species.spec.plv_window = Some(window.max(1) as usize);
            species
        });

        let ctx_for_create = ctx.clone();
        engine.register_fn(
            "create",
            move |call_ctx: NativeCallContext, species: SpeciesHandle, count: INT| {
                let mut ctx = ctx_for_create.lock().expect("lock script context");
                ctx.create_group(species, count, call_ctx.call_position())
            },
        );

        let ctx_for_wait = ctx.clone();
        engine.register_fn(
            "wait",
            move |_call_ctx: NativeCallContext, sec: FLOAT| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_wait.lock().expect("lock script context");
                ctx.wait(sec as f32);
                Ok(())
            },
        );
        let ctx_for_wait_int = ctx.clone();
        engine.register_fn(
            "wait",
            move |_call_ctx: NativeCallContext, sec: INT| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_wait_int.lock().expect("lock script context");
                ctx.wait(sec as f32);
                Ok(())
            },
        );

        let ctx_for_flush = ctx.clone();
        engine.register_fn(
            "flush",
            move |_call_ctx: NativeCallContext| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_flush.lock().expect("lock script context");
                ctx.flush();
                Ok(())
            },
        );

        let ctx_for_seed = ctx.clone();
        engine.register_fn(
            "seed",
            move |call_ctx: NativeCallContext, seed: INT| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_seed.lock().expect("lock script context");
                ctx.set_seed(seed, call_ctx.call_position())
            },
        );

        let ctx_for_release = ctx.clone();
        engine.register_fn(
            "release",
            move |_call_ctx: NativeCallContext, handle: GroupHandle| {
                let mut ctx = ctx_for_release.lock().expect("lock script context");
                ctx.release_group(handle.id);
            },
        );

        let ctx_for_scene = ctx.clone();
        engine.register_fn(
            "scene",
            move |call_ctx: NativeCallContext, name: &str, callback: FnPtr| {
                {
                    let mut ctx = ctx_for_scene.lock().expect("lock script context");
                    ctx.push_scene_marker(name);
                    ctx.push_scope();
                }
                let result = callback.call_within_context::<Dynamic>(&call_ctx, ());
                let mut ctx = ctx_for_scene.lock().expect("lock script context");
                ctx.pop_scope();
                result.map(|_| ())
            },
        );

        let ctx_for_play = ctx.clone();
        engine.register_fn(
            "play",
            move |call_ctx: NativeCallContext, callback: FnPtr| {
                {
                    let mut ctx = ctx_for_play.lock().expect("lock script context");
                    ctx.push_scope();
                }
                let result = callback.call_within_context::<Dynamic>(&call_ctx, ());
                let mut ctx = ctx_for_play.lock().expect("lock script context");
                ctx.pop_scope();
                result.map(|_| ())
            },
        );
        let ctx_for_play1 = ctx.clone();
        engine.register_fn(
            "play",
            move |call_ctx: NativeCallContext, callback: FnPtr, arg1: Dynamic| {
                {
                    let mut ctx = ctx_for_play1.lock().expect("lock script context");
                    ctx.push_scope();
                }
                let result = callback.call_within_context::<Dynamic>(&call_ctx, (arg1,));
                let mut ctx = ctx_for_play1.lock().expect("lock script context");
                ctx.pop_scope();
                result.map(|_| ())
            },
        );
        let ctx_for_play2 = ctx.clone();
        engine.register_fn(
            "play",
            move |call_ctx: NativeCallContext, callback: FnPtr, arg1: Dynamic, arg2: Dynamic| {
                {
                    let mut ctx = ctx_for_play2.lock().expect("lock script context");
                    ctx.push_scope();
                }
                let result = callback.call_within_context::<Dynamic>(&call_ctx, (arg1, arg2));
                let mut ctx = ctx_for_play2.lock().expect("lock script context");
                ctx.pop_scope();
                result.map(|_| ())
            },
        );
        let ctx_for_play3 = ctx.clone();
        engine.register_fn(
            "play",
            move |call_ctx: NativeCallContext,
                  callback: FnPtr,
                  arg1: Dynamic,
                  arg2: Dynamic,
                  arg3: Dynamic| {
                {
                    let mut ctx = ctx_for_play3.lock().expect("lock script context");
                    ctx.push_scope();
                }
                let result = callback.call_within_context::<Dynamic>(&call_ctx, (arg1, arg2, arg3));
                let mut ctx = ctx_for_play3.lock().expect("lock script context");
                ctx.pop_scope();
                result.map(|_| ())
            },
        );
        let ctx_for_play_args = ctx.clone();
        engine.register_fn(
            "play",
            move |call_ctx: NativeCallContext, callback: FnPtr, args: Array| {
                {
                    let mut ctx = ctx_for_play_args.lock().expect("lock script context");
                    ctx.push_scope();
                }
                let result = callback.call_within_context::<Dynamic>(&call_ctx, args);
                let mut ctx = ctx_for_play_args.lock().expect("lock script context");
                ctx.pop_scope();
                result.map(|_| ())
            },
        );

        let ctx_for_parallel = ctx.clone();
        engine.register_fn(
            "parallel",
            move |call_ctx: NativeCallContext,
                  callbacks: Array|
                  -> Result<(), Box<EvalAltResult>> {
                let start_time = {
                    let ctx = ctx_for_parallel.lock().expect("lock script context");
                    ctx.cursor
                };
                let mut max_end = start_time;
                for (idx, callback) in callbacks.into_iter().enumerate() {
                    let Some(fn_ptr) = callback.try_cast::<FnPtr>() else {
                        return Err(Box::new(EvalAltResult::ErrorRuntime(
                            format!("parallel expects closures (index {idx})").into(),
                            call_ctx.call_position(),
                        )));
                    };
                    {
                        let mut ctx = ctx_for_parallel.lock().expect("lock script context");
                        ctx.cursor = start_time;
                        ctx.push_scope();
                    }
                    let result = fn_ptr.call_within_context::<Dynamic>(&call_ctx, ());
                    let mut ctx = ctx_for_parallel.lock().expect("lock script context");
                    let end_time = ctx.cursor;
                    ctx.pop_scope();
                    max_end = max_end.max(end_time);
                    let _ = result?;
                }
                let mut ctx = ctx_for_parallel.lock().expect("lock script context");
                ctx.cursor = max_end;
                Ok(())
            },
        );

        engine.register_fn("harmonic_modes", ModePattern::harmonic_modes);
        engine.register_fn("odd_modes", ModePattern::odd_modes);
        engine.register_fn("power_modes", |beta: FLOAT| {
            ModePattern::power_modes(beta as f32)
        });
        engine.register_fn("stiff_string_modes", |stiffness: FLOAT| {
            ModePattern::stiff_string_modes(stiffness as f32)
        });
        engine.register_fn("custom_modes", |ratios: Array| {
            ModePattern::custom_modes(rhai_array_to_f32(ratios, "custom_modes"))
        });
        engine.register_fn("modal_table", |name: &str| {
            if let Some(pattern) = ModePattern::modal_table(name) {
                pattern
            } else {
                warn!(
                    "modal_table('{}') not found; falling back to harmonic_modes()",
                    name
                );
                ModePattern::harmonic_modes()
            }
        });
        engine.register_fn(
            "landscape_density_modes",
            ModePattern::landscape_density_modes,
        );
        engine.register_fn("landscape_peaks_modes", ModePattern::landscape_peaks_modes);
        engine.register_fn("count", |pattern: ModePattern, n: INT| {
            pattern.with_count((n as usize).max(1))
        });
        engine.register_fn(
            "range",
            |pattern: ModePattern, min_mul: FLOAT, max_mul: FLOAT| {
                if pattern.supports_range() {
                    pattern.with_range(min_mul as f32, max_mul as f32)
                } else {
                    warn!("range() is only supported for landscape_*_modes(); ignored");
                    pattern
                }
            },
        );
        engine.register_fn("min_dist", |pattern: ModePattern, min_dist: FLOAT| {
            if pattern.supports_min_dist_erb() {
                pattern.with_min_dist_erb(min_dist as f32)
            } else {
                warn!("min_dist() is only supported for landscape_*_modes(); ignored");
                pattern
            }
        });
        engine.register_fn("gamma", |pattern: ModePattern, gamma: FLOAT| {
            if pattern.supports_gamma() {
                pattern.with_gamma(gamma as f32)
            } else {
                warn!("gamma() is only supported for landscape_density_modes(); ignored");
                pattern
            }
        });
        engine.register_fn("jitter", |pattern: ModePattern, cents: FLOAT| {
            pattern.with_jitter_cents(cents as f32)
        });
        engine.register_fn("seed", |pattern: ModePattern, seed: INT| {
            if seed < 0 {
                warn!("seed() expects >= 0");
                pattern
            } else {
                pattern.with_seed(seed as u64)
            }
        });

        engine.register_fn("consonance", |root_freq: FLOAT| SpawnStrategy::Consonance {
            root_freq: root_freq as f32,
            min_mul: 1.0,
            max_mul: 4.0,
            min_dist_erb: 1.0,
        });
        engine.register_fn(
            "range",
            |strategy: SpawnStrategy, min_mul: FLOAT, max_mul: FLOAT| match strategy {
                SpawnStrategy::Consonance {
                    root_freq,
                    min_dist_erb,
                    ..
                } => SpawnStrategy::Consonance {
                    root_freq,
                    min_mul: min_mul as f32,
                    max_mul: max_mul as f32,
                    min_dist_erb,
                },
                other => {
                    warn!("range() ignored for non-consonance strategy");
                    other
                }
            },
        );
        engine.register_fn(
            "min_dist",
            |strategy: SpawnStrategy, min_dist: FLOAT| match strategy {
                SpawnStrategy::Consonance {
                    root_freq,
                    min_mul,
                    max_mul,
                    ..
                } => SpawnStrategy::Consonance {
                    root_freq,
                    min_mul,
                    max_mul,
                    min_dist_erb: min_dist as f32,
                },
                SpawnStrategy::ConsonanceDensity {
                    min_freq, max_freq, ..
                } => SpawnStrategy::ConsonanceDensity {
                    min_freq,
                    max_freq,
                    min_dist_erb: min_dist as f32,
                },
                other => {
                    warn!("min_dist() ignored for non-consonance strategy");
                    other
                }
            },
        );
        engine.register_fn(
            "consonance_density_pmf",
            |min_freq: FLOAT, max_freq: FLOAT| SpawnStrategy::ConsonanceDensity {
                min_freq: min_freq as f32,
                max_freq: max_freq as f32,
                min_dist_erb: 1.0,
            },
        );
        engine.register_fn("random_log", |min_freq: FLOAT, max_freq: FLOAT| {
            SpawnStrategy::RandomLog {
                min_freq: min_freq as f32,
                max_freq: max_freq as f32,
            }
        });
        engine.register_fn("linear", |start: FLOAT, end: FLOAT| SpawnStrategy::Linear {
            start_freq: start as f32,
            end_freq: end as f32,
        });
        engine.register_fn(
            "reject_targets",
            |strategy: SpawnStrategy,
             anchor_hz: FLOAT,
             targets_st: Array,
             exclusion_st: FLOAT,
             max_tries: INT| {
                SpawnStrategy::RejectTargets {
                    base: Box::new(strategy),
                    anchor_hz: anchor_hz as f32,
                    targets_st: rhai_array_to_f32(targets_st, "reject_targets"),
                    exclusion_st: exclusion_st as f32,
                    max_tries: max_tries.max(1) as usize,
                }
            },
        );
        engine.register_fn(
            "reject_targets",
            |strategy: SpawnStrategy,
             anchor_hz: INT,
             targets_st: Array,
             exclusion_st: FLOAT,
             max_tries: INT| {
                SpawnStrategy::RejectTargets {
                    base: Box::new(strategy),
                    anchor_hz: anchor_hz as f32,
                    targets_st: rhai_array_to_f32(targets_st, "reject_targets"),
                    exclusion_st: exclusion_st as f32,
                    max_tries: max_tries.max(1) as usize,
                }
            },
        );
        engine.register_fn(
            "reject_targets",
            |strategy: SpawnStrategy,
             anchor_hz: FLOAT,
             targets_st: Array,
             exclusion_st: INT,
             max_tries: INT| {
                SpawnStrategy::RejectTargets {
                    base: Box::new(strategy),
                    anchor_hz: anchor_hz as f32,
                    targets_st: rhai_array_to_f32(targets_st, "reject_targets"),
                    exclusion_st: exclusion_st as f32,
                    max_tries: max_tries.max(1) as usize,
                }
            },
        );
        engine.register_fn(
            "reject_targets",
            |strategy: SpawnStrategy,
             anchor_hz: INT,
             targets_st: Array,
             exclusion_st: INT,
             max_tries: INT| {
                SpawnStrategy::RejectTargets {
                    base: Box::new(strategy),
                    anchor_hz: anchor_hz as f32,
                    targets_st: rhai_array_to_f32(targets_st, "reject_targets"),
                    exclusion_st: exclusion_st as f32,
                    max_tries: max_tries.max(1) as usize,
                }
            },
        );

        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "amp",
            SpeciesSpec::set_amp,
            patch_amp,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "energy",
            SpeciesSpec::set_amp,
            patch_amp,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "freq",
            SpeciesSpec::set_freq,
            patch_freq,
            Some(draft_clear_strategy),
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "landscape_weight",
            SpeciesSpec::set_landscape_weight,
            patch_landscape_weight,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "neighbor_step_cents",
            SpeciesSpec::set_neighbor_step_cents,
            patch_neighbor_step_cents,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "tessitura_gravity",
            SpeciesSpec::set_tessitura_gravity,
            patch_tessitura_gravity,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "sustain_drive",
            SpeciesSpec::set_continuous_drive,
            patch_continuous_drive,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "pitch_smooth",
            SpeciesSpec::set_pitch_smooth_tau,
            patch_pitch_smooth_tau,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "exploration",
            SpeciesSpec::set_exploration,
            patch_exploration,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "persistence",
            SpeciesSpec::set_persistence,
            patch_persistence,
            None,
        );
        register_group_crowding_overloads(&mut engine, ctx.clone());
        let ctx_for_group_crowding_target = ctx.clone();
        engine.register_fn(
            "crowding_target",
            move |handle: GroupHandle,
                  same_group_visible: bool,
                  other_group_visible: bool|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_crowding_target
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("crowding_target ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => {
                        group.crowding_target_same = same_group_visible;
                        group.crowding_target_other = other_group_visible;
                        group
                            .spec
                            .set_crowding_target(same_group_visible, other_group_visible);
                    }
                    GroupStatus::Live => {
                        group.crowding_target_same = same_group_visible;
                        group.crowding_target_other = other_group_visible;
                        group
                            .spec
                            .set_crowding_target(same_group_visible, other_group_visible);
                        group.pending_crowding_target =
                            Some((same_group_visible, other_group_visible));
                    }
                    _ => ctx.warn_live_builder(handle.id, "crowding_target"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_leave_self_out = ctx.clone();
        engine.register_fn(
            "leave_self_out",
            move |handle: GroupHandle, enabled: bool| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_leave_self_out
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("leave_self_out ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_leave_self_out(enabled),
                    GroupStatus::Live => {
                        group.spec.set_leave_self_out(enabled);
                        group.pending_patch.leave_self_out = Some(enabled);
                    }
                    _ => ctx.warn_live_builder(handle.id, "leave_self_out"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_leave_self_out_mode = ctx.clone();
        engine.register_fn(
            "leave_self_out_mode",
            move |handle: GroupHandle, name: &str| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_leave_self_out_mode
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!(
                        "leave_self_out_mode ignored for unknown group {}",
                        handle.id
                    );
                    return Ok(handle);
                };
                let mode = parse_leave_self_out_mode_name(
                    group.spec.control.pitch.leave_self_out_mode,
                    name,
                );
                match group.status {
                    GroupStatus::Draft => group.spec.control.set_leave_self_out_mode(mode),
                    GroupStatus::Live => {
                        group.spec.control.set_leave_self_out_mode(mode);
                        patch_leave_self_out_mode(&mut group.pending_patch, mode);
                    }
                    _ => ctx.warn_live_builder(handle.id, "leave_self_out_mode"),
                }
                Ok(handle)
            },
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "anneal_temp",
            SpeciesSpec::set_anneal_temp,
            patch_anneal_temp,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "move_cost",
            SpeciesSpec::set_move_cost_coeff,
            patch_move_cost_coeff,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "move_cost_exp",
            SpeciesSpec::set_move_cost_exp,
            patch_move_cost_exp,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "improvement_threshold",
            SpeciesSpec::set_improvement_threshold,
            patch_improvement_threshold,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "proposal_interval",
            SpeciesSpec::set_proposal_interval_sec,
            patch_proposal_interval,
            None,
        );
        let ctx_for_group_global_peaks = ctx.clone();
        engine.register_fn(
            "global_peaks",
            move |handle: GroupHandle, count: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_global_peaks
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("global_peaks ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_global_peaks(count, 0.0),
                    GroupStatus::Live => {
                        group.spec.set_global_peaks(count, 0.0);
                        group.pending_patch.global_peak_count = Some(count);
                        group.pending_patch.global_peak_min_sep_cents = Some(0.0);
                    }
                    _ => ctx.warn_live_builder(handle.id, "global_peaks"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_global_peaks_sep = ctx.clone();
        engine.register_fn(
            "global_peaks",
            move |handle: GroupHandle,
                  count: INT,
                  min_sep_cents: FLOAT|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_global_peaks_sep
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("global_peaks ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let min_sep = min_sep_cents as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_global_peaks(count, min_sep),
                    GroupStatus::Live => {
                        group.spec.set_global_peaks(count, min_sep);
                        group.pending_patch.global_peak_count = Some(count);
                        group.pending_patch.global_peak_min_sep_cents = Some(min_sep);
                    }
                    _ => ctx.warn_live_builder(handle.id, "global_peaks"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_global_peaks_sep_int = ctx.clone();
        engine.register_fn(
            "global_peaks",
            move |handle: GroupHandle,
                  count: INT,
                  min_sep_cents: INT|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_global_peaks_sep_int
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("global_peaks ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let min_sep = min_sep_cents as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_global_peaks(count, min_sep),
                    GroupStatus::Live => {
                        group.spec.set_global_peaks(count, min_sep);
                        group.pending_patch.global_peak_count = Some(count);
                        group.pending_patch.global_peak_min_sep_cents = Some(min_sep);
                    }
                    _ => ctx.warn_live_builder(handle.id, "global_peaks"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_ratio_candidates = ctx.clone();
        engine.register_fn(
            "ratio_candidates",
            move |handle: GroupHandle, count: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_ratio_candidates
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("ratio_candidates ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_ratio_candidates(count),
                    GroupStatus::Live => {
                        group.spec.set_ratio_candidates(count);
                        group.pending_patch.ratio_candidate_count = Some(count);
                        group.pending_patch.use_ratio_candidates = Some(count > 0);
                    }
                    _ => ctx.warn_live_builder(handle.id, "ratio_candidates"),
                }
                Ok(handle)
            },
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "window_cents",
            SpeciesSpec::set_window_cents,
            patch_window_cents,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "top_k",
            SpeciesSpec::set_top_k,
            patch_top_k,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "temperature",
            SpeciesSpec::set_temperature,
            patch_temperature,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "sigma_cents",
            SpeciesSpec::set_sigma_cents,
            patch_sigma_cents,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "random_candidates",
            SpeciesSpec::set_random_candidates,
            patch_random_candidates,
            None,
        );
        let ctx_for_group_move_cost_time_scale = ctx.clone();
        engine.register_fn(
            "move_cost_time_scale",
            move |handle: GroupHandle, name: &str| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_move_cost_time_scale
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!(
                        "move_cost_time_scale ignored for unknown group {}",
                        handle.id
                    );
                    return Ok(handle);
                };
                let lowered = name.trim().to_ascii_lowercase();
                let value = match lowered.as_str() {
                    "legacy" | "integration" | "integration_window" => {
                        MoveCostTimeScale::LegacyIntegrationWindow
                    }
                    "proposal" | "proposal_interval" => MoveCostTimeScale::ProposalInterval,
                    _ => {
                        ctx.warn_live_builder(handle.id, "move_cost_time_scale");
                        return Ok(handle);
                    }
                };
                match group.status {
                    GroupStatus::Draft => group.spec.control.set_move_cost_time_scale(value),
                    GroupStatus::Live => {
                        group.spec.control.set_move_cost_time_scale(value);
                        group.pending_patch.move_cost_time_scale = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "move_cost_time_scale"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_loo_harmonics = ctx.clone();
        engine.register_fn(
            "leave_self_out_harmonics",
            move |handle: GroupHandle, value: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_loo_harmonics
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!(
                        "leave_self_out_harmonics ignored for unknown group {}",
                        handle.id
                    );
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_leave_self_out_harmonics(value),
                    GroupStatus::Live => {
                        group.spec.set_leave_self_out_harmonics(value);
                        group.pending_patch.leave_self_out_harmonics = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "leave_self_out_harmonics"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_pitch_apply_mode = ctx.clone();
        engine.register_fn(
            "pitch_apply_mode",
            move |handle: GroupHandle, name: &str| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_pitch_apply_mode
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("pitch_apply_mode ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let lowered = name.trim().to_ascii_lowercase();
                let mode = match lowered.as_str() {
                    "gate_snap" | "gatesnap" | "snap" => PitchApplyMode::GateSnap,
                    "glide" | "gliss" | "glissando" => PitchApplyMode::Glide,
                    _ => {
                        ctx.warn_live_builder(handle.id, "pitch_apply_mode");
                        return Ok(handle);
                    }
                };
                match group.status {
                    GroupStatus::Draft => group.spec.control.set_pitch_apply_mode(mode),
                    GroupStatus::Live => {
                        group.spec.control.set_pitch_apply_mode(mode);
                        group.pending_patch.pitch_apply_mode = Some(mode);
                    }
                    _ => ctx.warn_live_builder(handle.id, "pitch_apply_mode"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_pitch_apply = ctx.clone();
        engine.register_fn(
            "pitch_apply",
            move |handle: GroupHandle, name: &str| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_pitch_apply
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("pitch_apply ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let lowered = name.trim().to_ascii_lowercase();
                let mode = match lowered.as_str() {
                    "gate_snap" | "gatesnap" | "snap" => PitchApplyMode::GateSnap,
                    "glide" | "gliss" | "glissando" => PitchApplyMode::Glide,
                    _ => {
                        ctx.warn_live_builder(handle.id, "pitch_apply");
                        return Ok(handle);
                    }
                };
                match group.status {
                    GroupStatus::Draft => group.spec.control.set_pitch_apply_mode(mode),
                    GroupStatus::Live => {
                        group.spec.control.set_pitch_apply_mode(mode);
                        group.pending_patch.pitch_apply_mode = Some(mode);
                    }
                    _ => ctx.warn_live_builder(handle.id, "pitch_apply"),
                }
                Ok(handle)
            },
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "pitch_glide",
            SpeciesSpec::set_pitch_glide_tau_sec,
            patch_pitch_glide_tau,
            None,
        );
        let ctx_for_group_brain = ctx.clone();
        engine.register_fn(
            "brain",
            move |handle: GroupHandle, name: &str| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_brain.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("brain ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_brain(name),
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "brain"),
                    _ => ctx.warn_live_builder(handle.id, "brain"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_pitch_mode = ctx.clone();
        engine.register_fn(
            "pitch_mode",
            move |handle: GroupHandle, name: &str| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_pitch_mode
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("pitch_mode ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_pitch_mode(name),
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "pitch_mode"),
                    _ => ctx.warn_live_builder(handle.id, "pitch_mode"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_mode = ctx.clone();
        engine.register_fn(
            "mode",
            move |handle: GroupHandle, name: &str| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_mode.lock().map_err(|err| {
                    let msg = format!("script context lock poisoned: {err}");
                    Box::new(EvalAltResult::ErrorSystem(
                        msg.clone(),
                        Box::new(std::io::Error::other(msg)),
                    ))
                })?;
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("mode ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_pitch_mode(name),
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "mode"),
                    _ => ctx.warn_live_builder(handle.id, "mode"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_pitch_core = ctx.clone();
        engine.register_fn(
            "pitch_core",
            move |handle: GroupHandle, name: &str| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_pitch_core
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("pitch_core ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_pitch_core(name),
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "pitch_core"),
                    _ => ctx.warn_live_builder(handle.id, "pitch_core"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_sustain = ctx.clone();
        engine.register_fn(
            "sustain",
            move |handle: GroupHandle| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_sustain.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("sustain ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => {
                        group.spec.set_phonation(PhonationKind::Sustain);
                    }
                    _ => ctx.warn_live_builder(handle.id, "sustain"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_repeat = ctx.clone();
        engine.register_fn(
            "repeat",
            move |handle: GroupHandle| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_repeat.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("repeat ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => {
                        group.spec.set_phonation(PhonationKind::Repeat);
                    }
                    _ => ctx.warn_live_builder(handle.id, "repeat"),
                }
                Ok(handle)
            },
        );
        // Tier 2: explicit when/duration (group, draft-only)
        macro_rules! register_group_draft_fn {
            ($name:expr, $ctx:expr, $engine:expr, |$spec:ident| $body:expr) => {{
                let ctx_clone = $ctx.clone();
                $engine.register_fn(
                    $name,
                    move |handle: GroupHandle| -> Result<GroupHandle, Box<EvalAltResult>> {
                        let mut ctx = ctx_clone.lock().expect("lock script context");
                        let Some(group) = ctx.groups.get_mut(&handle.id) else {
                            warn!("{} ignored for unknown group {}", $name, handle.id);
                            return Ok(handle);
                        };
                        match group.status {
                            GroupStatus::Draft => {
                                let $spec = &mut group.spec;
                                $body;
                            }
                            _ => ctx.warn_live_builder(handle.id, $name),
                        }
                        Ok(handle)
                    },
                );
            }};
        }
        macro_rules! register_group_draft_fn1 {
            ($name:expr, $ctx:expr, $engine:expr, |$spec:ident, $a:ident: $at:ty| $body:expr) => {{
                let ctx_clone = $ctx.clone();
                $engine.register_fn(
                    $name,
                    move |handle: GroupHandle,
                          $a: $at|
                          -> Result<GroupHandle, Box<EvalAltResult>> {
                        let mut ctx = ctx_clone.lock().expect("lock script context");
                        let Some(group) = ctx.groups.get_mut(&handle.id) else {
                            warn!("{} ignored for unknown group {}", $name, handle.id);
                            return Ok(handle);
                        };
                        match group.status {
                            GroupStatus::Draft => {
                                let $spec = &mut group.spec;
                                $body;
                            }
                            _ => ctx.warn_live_builder(handle.id, $name),
                        }
                        Ok(handle)
                    },
                );
            }};
        }
        macro_rules! register_group_draft_fn2 {
            ($name:expr, $ctx:expr, $engine:expr, |$spec:ident, $a:ident: $at:ty, $b:ident: $bt:ty| $body:expr) => {{
                let ctx_clone = $ctx.clone();
                $engine.register_fn(
                    $name,
                    move |handle: GroupHandle,
                          $a: $at,
                          $b: $bt|
                          -> Result<GroupHandle, Box<EvalAltResult>> {
                        let mut ctx = ctx_clone.lock().expect("lock script context");
                        let Some(group) = ctx.groups.get_mut(&handle.id) else {
                            warn!("{} ignored for unknown group {}", $name, handle.id);
                            return Ok(handle);
                        };
                        match group.status {
                            GroupStatus::Draft => {
                                let $spec = &mut group.spec;
                                $body;
                            }
                            _ => ctx.warn_live_builder(handle.id, $name),
                        }
                        Ok(handle)
                    },
                );
            }};
        }
        register_group_draft_fn!("once", ctx, engine, |s| s.set_when_once());
        register_group_draft_fn1!("pulse", ctx, engine, |s, rate: FLOAT| s
            .set_when_pulse(rate as f32));
        register_group_draft_fn!("while_alive", ctx, engine, |s| s.set_duration_while_alive());
        register_group_draft_fn1!("gates", ctx, engine, |s, n: INT| s
            .set_duration_gates(n.max(1) as u32));
        register_group_draft_fn!("field", ctx, engine, |s| s.set_duration_field());
        register_group_draft_fn1!("sync", ctx, engine, |s, depth: FLOAT| s
            .set_sync(depth as f32));
        register_group_draft_fn1!("social", ctx, engine, |s, coupling: FLOAT| s
            .set_social(coupling as f32));
        register_group_draft_fn2!("field_window", ctx, engine, |s, min: FLOAT, max: FLOAT| s
            .set_field_window(min as f32, max as f32));
        register_group_draft_fn2!("field_curve", ctx, engine, |s, k: FLOAT, x0: FLOAT| s
            .set_field_curve(k as f32, x0 as f32));
        register_group_draft_fn1!("field_drop", ctx, engine, |s, gain: FLOAT| s
            .set_field_drop(gain as f32));
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "brightness",
            SpeciesSpec::set_brightness,
            patch_timbre_brightness,
            None,
        );
        let ctx_for_group_modes = ctx.clone();
        engine.register_fn(
            "modes",
            move |handle: GroupHandle,
                  pattern: ModePattern|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_modes.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("modes ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_modes(pattern),
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "modes"),
                    _ => ctx.warn_live_builder(handle.id, "modes"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_metabolism = ctx.clone();
        engine.register_fn(
            "metabolism",
            move |handle: GroupHandle, rate: FLOAT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_metabolism
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("metabolism ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_metabolism(rate as f32),
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "metabolism"),
                    _ => ctx.warn_live_builder(handle.id, "metabolism"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_adsr = ctx.clone();
        engine.register_fn(
            "adsr",
            move |handle: GroupHandle,
                  a: FLOAT,
                  d: FLOAT,
                  s: FLOAT,
                  r: FLOAT|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_adsr.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("adsr ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => {
                        group.spec.set_adsr(a as f32, d as f32, s as f32, r as f32);
                    }
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "adsr"),
                    _ => ctx.warn_live_builder(handle.id, "adsr"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_rhythm_coupling = ctx.clone();
        engine.register_fn(
            "rhythm_coupling",
            move |handle: GroupHandle, mode: &str| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_rhythm_coupling
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("rhythm_coupling ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_rhythm_coupling(mode),
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "rhythm_coupling"),
                    _ => ctx.warn_live_builder(handle.id, "rhythm_coupling"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_rhythm_coupling_vitality = ctx.clone();
        engine.register_fn(
            "rhythm_coupling_vitality",
            move |handle: GroupHandle,
                  lambda_v: FLOAT,
                  v_floor: FLOAT|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_rhythm_coupling_vitality
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!(
                        "rhythm_coupling_vitality ignored for unknown group {}",
                        handle.id
                    );
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group
                        .spec
                        .set_rhythm_coupling_vitality(lambda_v as f32, v_floor as f32),
                    GroupStatus::Live => {
                        ctx.warn_live_builder(handle.id, "rhythm_coupling_vitality")
                    }
                    _ => ctx.warn_live_builder(handle.id, "rhythm_coupling_vitality"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_rhythm_reward = ctx.clone();
        engine.register_fn(
            "rhythm_reward",
            move |handle: GroupHandle,
                  rho_t: FLOAT,
                  metric: &str|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_rhythm_reward
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("rhythm_reward ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_rhythm_reward(rho_t as f32, metric),
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "rhythm_reward"),
                    _ => ctx.warn_live_builder(handle.id, "rhythm_reward"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_respawn_random = ctx.clone();
        engine.register_fn(
            "respawn_random",
            move |handle: GroupHandle| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_respawn_random
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("respawn_random ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => {
                        group.respawn_policy = RespawnPolicy::Random;
                        group.spec.respawn_policy = RespawnPolicy::Random;
                    }
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "respawn_random"),
                    _ => ctx.warn_live_builder(handle.id, "respawn_random"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_respawn_hereditary = ctx.clone();
        engine.register_fn(
            "respawn_hereditary",
            move |handle: GroupHandle,
                  sigma_oct: FLOAT|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_respawn_hereditary
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("respawn_hereditary ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let sigma_oct = sigma_oct as f32;
                let sigma_oct = if sigma_oct.is_finite() {
                    sigma_oct.max(0.0)
                } else {
                    0.0
                };
                match group.status {
                    GroupStatus::Draft => {
                        let policy = RespawnPolicy::Hereditary { sigma_oct };
                        group.respawn_policy = policy;
                        group.spec.respawn_policy = policy;
                    }
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "respawn_hereditary"),
                    _ => ctx.warn_live_builder(handle.id, "respawn_hereditary"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_respawn_hereditary_int = ctx.clone();
        engine.register_fn(
            "respawn_hereditary",
            move |handle: GroupHandle, sigma_oct: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_respawn_hereditary_int
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("respawn_hereditary ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let sigma_oct = (sigma_oct as f32).max(0.0);
                match group.status {
                    GroupStatus::Draft => {
                        let policy = RespawnPolicy::Hereditary { sigma_oct };
                        group.respawn_policy = policy;
                        group.spec.respawn_policy = policy;
                    }
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "respawn_hereditary"),
                    _ => ctx.warn_live_builder(handle.id, "respawn_hereditary"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_place = ctx.clone();
        engine.register_fn(
            "place",
            move |handle: GroupHandle,
                  strategy: SpawnStrategy|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_place.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("place ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => {
                        group.spec.control.pitch.mode = PitchMode::Lock;
                        group.strategy = Some(strategy);
                    }
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "place"),
                    _ => ctx.warn_live_builder(handle.id, "place"),
                }
                Ok(handle)
            },
        );

        let ctx_for_set_harmonicity_mirror_weight = ctx.clone();
        engine.register_fn(
            "set_harmonicity_mirror_weight",
            move |_call_ctx: NativeCallContext, mirror: FLOAT| {
                let mut ctx = ctx_for_set_harmonicity_mirror_weight
                    .lock()
                    .expect("lock script context");
                let update = crate::core::landscape::LandscapeUpdate {
                    mirror: Some(mirror as f32),
                    ..crate::core::landscape::LandscapeUpdate::default()
                };
                let cursor = ctx.cursor;
                ctx.push_event(cursor, vec![Action::SetHarmonicityParams { update }]);
            },
        );
        let ctx_for_set_pitch_objective = ctx.clone();
        engine.register_fn(
            "set_pitch_objective",
            move |_call_ctx: NativeCallContext, name: &str| {
                let mut ctx = ctx_for_set_pitch_objective
                    .lock()
                    .expect("lock script context");
                let lowered = name.trim().to_ascii_lowercase();
                let mode = match lowered.as_str() {
                    "consonance" | "positive" | "pos" => PitchObjectiveMode::Consonance,
                    "negative_consonance" | "negative" | "neg" | "dissonance" => {
                        PitchObjectiveMode::NegativeConsonance
                    }
                    other => {
                        warn!(
                            "set_pitch_objective() expects 'consonance' or 'negative_consonance', got '{}'",
                            other
                        );
                        return;
                    }
                };
                let cursor = ctx.cursor;
                let update = crate::core::landscape::LandscapeUpdate {
                    pitch_objective_mode: Some(mode),
                    ..crate::core::landscape::LandscapeUpdate::default()
                };
                ctx.push_event(cursor, vec![Action::SetHarmonicityParams { update }]);
            },
        );

        let ctx_for_set_global_coupling = ctx.clone();
        engine.register_fn(
            "set_global_coupling",
            move |_call_ctx: NativeCallContext, value: FLOAT| {
                let mut ctx = ctx_for_set_global_coupling
                    .lock()
                    .expect("lock script context");
                let cursor = ctx.cursor;
                ctx.push_event(
                    cursor,
                    vec![Action::SetGlobalCoupling {
                        value: value as f32,
                    }],
                );
            },
        );

        let ctx_for_set_roughness_k = ctx.clone();
        engine.register_fn(
            "set_roughness_k",
            move |_call_ctx: NativeCallContext, value: FLOAT| {
                let mut ctx = ctx_for_set_roughness_k.lock().expect("lock script context");
                let cursor = ctx.cursor;
                ctx.push_event(
                    cursor,
                    vec![Action::SetRoughnessTolerance {
                        value: value as f32,
                    }],
                );
            },
        );

        engine
    }

    pub fn load_script(path: &str) -> Result<Scenario, ScriptError> {
        let src = fs::read_to_string(path)
            .map_err(|err| ScriptError::new(format!("read script {path}: {err}"), None))?;
        let ctx = Arc::new(Mutex::new(ScriptContext::default()));
        let engine = ScriptHost::create_engine(ctx.clone());

        if let Err(e) = engine.eval::<()>(&src) {
            println!("Debug script error: {:?}", e);
            return Err(ScriptError::from_eval(
                e,
                Some(&format!("execute script {path}")),
            ));
        }

        let mut ctx_out = ctx.lock().expect("lock script context");
        ctx_out.finish();
        Ok(ctx_out.scenario.clone())
    }
}

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
mod tests {
    use super::*;
    use crate::core::landscape::LandscapeFrame;
    use crate::core::timebase::Timebase;
    use crate::life::individual::AnyArticulationCore;
    use crate::life::individual::sound_body::SoundBody;
    use crate::life::population::Population;
    use crate::life::scenario::RhythmCouplingMode;
    use rand::SeedableRng;
    use std::collections::HashMap;

    fn run_script(src: &str) -> (Scenario, ScriptWarnings) {
        let ctx = Arc::new(Mutex::new(ScriptContext::default()));
        let engine = ScriptHost::create_engine(ctx.clone());
        let _ = engine.eval::<Dynamic>(&src).expect("script runs");
        let mut ctx_out = ctx.lock().expect("lock script context");
        ctx_out.finish();
        (ctx_out.scenario.clone(), ctx_out.warnings.clone())
    }

    fn run_script_err(src: &str) -> ScriptError {
        let ctx = Arc::new(Mutex::new(ScriptContext::default()));
        let engine = ScriptHost::create_engine(ctx);
        let err = engine
            .eval::<Dynamic>(&src)
            .expect_err("script should fail");
        ScriptError::from_eval(err, None)
    }

    fn action_times<'a>(scenario: &'a Scenario) -> Vec<(f32, &'a Action)> {
        let mut out = Vec::new();
        for ev in &scenario.events {
            for action in &ev.actions {
                out.push((ev.time, action));
            }
        }
        out
    }

    const E4_STEP_RESPONSE_SCRIPT: &str = r#"
        let anchor = derive(harmonic).pitch_mode("lock");
        let probe = derive(harmonic).pitch_mode("free").pitch_core("peak_sampler");

        scene("E4 Step Response Test", || {
            let base = create(anchor, 1).freq(196.0);
            flush();

            let probes = create(probe, 12)
                .place(consonance(196.0).range(0.8, 2.5).min_dist(0.9))
                .mode("free");
            flush();

            set_harmonicity_mirror_weight(0.0);
            flush();
            set_harmonicity_mirror_weight(0.5);
            flush();
            set_harmonicity_mirror_weight(1.0);
            flush();

            release(probes);
            release(base);
            flush();
        });
    "#;

    const E4_BETWEEN_RUNS_SCRIPT: &str = r#"
        let anchor = derive(harmonic).pitch_mode("lock");
        let probe = derive(harmonic).pitch_mode("free").pitch_core("peak_sampler");

        scene("E4 Between Runs Test", || {
            let weights = [0.0, 0.5, 1.0];
            for w in weights {
                set_harmonicity_mirror_weight(w);
                flush();

                let base = create(anchor, 1).freq(196.0);
                let probes = create(probe, 8)
                    .place(consonance(196.0).range(0.8, 2.5).min_dist(0.9))
                    .mode("free");
                flush();

                release(probes);
                release(base);
                flush();
            }
        });
    "#;

    #[test]
    fn draft_group_without_commit_is_dropped() {
        let (scenario, warnings) = run_script(
            r#"
            create(sine, 1);
        "#,
        );
        let has_spawn = scenario
            .events
            .iter()
            .any(|ev| ev.actions.iter().any(|a| matches!(a, Action::Spawn { .. })));
        assert!(!has_spawn, "draft should not spawn without wait/flush");
        assert_eq!(warnings.draft_dropped, 1);
    }

    #[test]
    fn flush_spawns_without_advancing_time() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine, 1);
            flush();
            wait(1.0);
        "#,
        );
        let mut spawn_time = None;
        let mut finish_time = None;
        for (time, action) in action_times(&scenario) {
            match action {
                Action::Spawn { .. } => spawn_time = Some(time),
                Action::Finish => finish_time = Some(time),
                _ => {}
            }
        }
        assert_eq!(spawn_time, Some(0.0));
        assert_eq!(finish_time, Some(1.0));
    }

    #[test]
    fn scene_scope_releases_live_groups() {
        let (scenario, _warnings) = run_script(
            r#"
            scene("alpha", || {
                let g = create(sine, 1);
                flush();
                wait(0.5);
            });
        "#,
        );
        let mut release_time = None;
        for (time, action) in action_times(&scenario) {
            if matches!(action, Action::ReleaseGroup { .. }) {
                release_time = Some(time);
            }
        }
        assert_eq!(release_time, Some(0.5));
    }

    #[test]
    fn parallel_advances_to_max_child_end() {
        let (scenario, _warnings) = run_script(
            r#"
            parallel([
                || { create(sine, 1); wait(0.5); },
                || { create(sine, 1); wait(1.0); }
            ]);
        "#,
        );
        let finish_time = scenario
            .events
            .iter()
            .find(|ev| ev.actions.iter().any(|a| matches!(a, Action::Finish)))
            .map(|ev| ev.time);
        let mut release_tail: f32 = 0.0;
        for event in &scenario.events {
            for action in &event.actions {
                if let Action::ReleaseGroup { fade_sec, .. } = action {
                    release_tail = release_tail.max(event.time + fade_sec);
                }
            }
        }
        let expected = release_tail.max(1.0);
        assert!(matches!(finish_time, Some(t) if (t - expected).abs() <= 1e-6));
    }

    #[test]
    fn scope_drop_warns_on_draft() {
        let (_scenario, warnings) = run_script(
            r#"
            scene("alpha", || { create(sine, 1); });
        "#,
        );
        assert_eq!(warnings.draft_dropped, 1);
    }

    #[test]
    fn spawn_order_is_group_id_order() {
        let (scenario, _warnings) = run_script(
            r#"
            let a = create(sine, 1);
            let b = create(sine, 1);
            flush();
        "#,
        );
        let mut spawns = Vec::new();
        for event in &scenario.events {
            for action in &event.actions {
                if let Action::Spawn { group_id, ids, .. } = action {
                    spawns.push((event.time, *group_id, ids.clone()));
                }
            }
        }
        assert_eq!(spawns.len(), 2);
        assert_eq!(spawns[0].0, 0.0);
        assert_eq!(spawns[0].1, 1);
        assert_eq!(spawns[0].2, vec![1]);
        assert_eq!(spawns[1].0, 0.0);
        assert_eq!(spawns[1].1, 2);
        assert_eq!(spawns[1].2, vec![2]);
    }

    #[test]
    fn flush_events_have_increasing_order_at_same_time() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine, 1);
            flush();
            create(sine, 1);
            flush();
        "#,
        );
        let mut orders = Vec::new();
        for event in &scenario.events {
            if (event.time - 0.0).abs() <= f32::EPSILON {
                orders.push(event.order);
            }
        }
        assert!(orders.len() >= 2);
        for pair in orders.windows(2) {
            assert!(pair[0] < pair[1]);
        }
    }

    #[test]
    fn place_then_freq_clears_strategy() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine, 4).place(consonance(220.0)).freq(330.0);
            flush();
        "#,
        );
        let strategy = scenario
            .events
            .iter()
            .flat_map(|event| &event.actions)
            .find_map(|action| match action {
                Action::Spawn { strategy, .. } => Some(strategy.clone()),
                _ => None,
            })
            .expect("spawn action");
        assert!(strategy.is_none());
    }

    #[test]
    fn freq_then_place_sets_strategy() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine, 4).freq(330.0).place(consonance(220.0));
            flush();
        "#,
        );
        let strategy = scenario
            .events
            .iter()
            .flat_map(|event| &event.actions)
            .find_map(|action| match action {
                Action::Spawn { strategy, .. } => Some(strategy.clone()),
                _ => None,
            })
            .expect("spawn action");
        assert!(matches!(strategy, Some(SpawnStrategy::Consonance { .. })));
    }

    #[test]
    fn place_then_pitch_mode_free_overrides_lock() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine, 4).place(consonance(220.0)).pitch_mode("free");
            flush();
        "#,
        );
        let (strategy, mode) = scenario
            .events
            .iter()
            .flat_map(|event| &event.actions)
            .find_map(|action| match action {
                Action::Spawn { strategy, spec, .. } => {
                    Some((strategy.clone(), spec.control.pitch.mode))
                }
                _ => None,
            })
            .expect("spawn action");
        assert!(matches!(strategy, Some(SpawnStrategy::Consonance { .. })));
        assert_eq!(mode, PitchMode::Free);
    }

    #[test]
    fn place_then_mode_free_survives_spawn() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine, 2).place(consonance(220.0)).mode("free");
            flush();
        "#,
        );
        let spawn_action = scenario
            .events
            .iter()
            .flat_map(|event| &event.actions)
            .find_map(|action| match action {
                Action::Spawn { .. } => Some(action.clone()),
                _ => None,
            })
            .expect("spawn action");

        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        pop.apply_action(spawn_action, &landscape, None);

        let agent = pop.individuals.first().expect("spawned");
        assert_eq!(agent.base_control.pitch.mode, PitchMode::Free);
        assert_eq!(agent.effective_control.pitch.mode, PitchMode::Free);
    }

    #[test]
    fn place_defaults_to_lock_mode() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine, 4).place(consonance(220.0));
            flush();
        "#,
        );
        let mode = scenario
            .events
            .iter()
            .flat_map(|event| &event.actions)
            .find_map(|action| match action {
                Action::Spawn { spec, .. } => Some(spec.control.pitch.mode),
                _ => None,
            })
            .expect("spawn action");
        assert_eq!(mode, PitchMode::Lock);
    }

    #[test]
    fn pitch_mode_free_then_place_restores_lock() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine, 4).pitch_mode("free").place(consonance(220.0));
            flush();
        "#,
        );
        let mode = scenario
            .events
            .iter()
            .flat_map(|event| &event.actions)
            .find_map(|action| match action {
                Action::Spawn { spec, .. } => Some(spec.control.pitch.mode),
                _ => None,
            })
            .expect("spawn action");
        assert_eq!(mode, PitchMode::Lock);
    }

    #[test]
    fn species_pitch_mode_sets_spawn_mode() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine.pitch_mode("lock"), 1);
            flush();
        "#,
        );
        let mode = scenario
            .events
            .iter()
            .flat_map(|event| &event.actions)
            .find_map(|action| match action {
                Action::Spawn { spec, .. } => Some(spec.control.pitch.mode),
                _ => None,
            })
            .expect("spawn action");
        assert_eq!(mode, PitchMode::Lock);
    }

    #[test]
    fn species_pitch_core_sets_spawn_core_kind() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine.pitch_core("peak_sampler"), 1);
            flush();
        "#,
        );
        let core_kind = scenario
            .events
            .iter()
            .flat_map(|event| &event.actions)
            .find_map(|action| match action {
                Action::Spawn { spec, .. } => Some(spec.control.pitch.core_kind),
                _ => None,
            })
            .expect("spawn action");
        assert_eq!(core_kind, PitchCoreKind::PeakSampler);
    }

    #[test]
    fn species_landscape_weight_sets_spawn_control() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine.landscape_weight(0.25), 1);
            flush();
        "#,
        );
        let weight = scenario
            .events
            .iter()
            .flat_map(|event| &event.actions)
            .find_map(|action| match action {
                Action::Spawn { spec, .. } => Some(spec.control.pitch.landscape_weight),
                _ => None,
            })
            .expect("spawn action");
        assert!((weight - 0.25).abs() <= 1e-6);
    }

    #[test]
    fn species_landscape_weight_reaches_spawned_individual() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine.landscape_weight(0.3), 1);
            flush();
        "#,
        );
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            if let Action::Spawn { .. } = action {
                pop.apply_action(action.clone(), &landscape, None);
            }
        }
        let agent = pop.individuals.first().expect("spawned");
        assert!((agent.effective_control.pitch.landscape_weight - 0.3).abs() <= 1e-6);
    }

    #[test]
    fn species_exploration_persistence_reach_spawned_core() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine.exploration(0.8).persistence(0.2), 1);
            flush();
        "#,
        );
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            if let Action::Spawn { .. } = action {
                pop.apply_action(action.clone(), &landscape, None);
            }
        }
        let agent = pop.individuals.first().expect("spawned");
        assert!((agent.pitch_core_for_test().exploration_for_test() - 0.8).abs() <= 1e-6);
        assert!((agent.pitch_core_for_test().persistence_for_test() - 0.2).abs() <= 1e-6);
    }

    #[test]
    fn group_landscape_weight_emits_live_update() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(sine, 1);
            flush();
            let g = g.landscape_weight(0.4);
            flush();
        "#,
        );
        let mut saw_update = false;
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            if let Action::UpdateGroup { patch, .. } = action
                && patch.landscape_weight == Some(0.4)
            {
                saw_update = true;
                break;
            }
        }
        assert!(saw_update, "expected landscape_weight live update");
    }

    #[test]
    fn group_landscape_weight_live_update_reaches_individual() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(sine, 1);
            flush();
            let g = g.landscape_weight(0.6);
            flush();
        "#,
        );
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            match action {
                Action::Spawn { .. } | Action::UpdateGroup { .. } => {
                    pop.apply_action(action.clone(), &landscape, None);
                }
                _ => {}
            }
        }
        let agent = pop.individuals.first().expect("spawned");
        assert!((agent.effective_control.pitch.landscape_weight - 0.6).abs() <= 1e-6);
    }

    #[test]
    fn group_amp_live_update_preserves_member_pitch_centers() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(sine, 4).place(consonance(196.0).range(0.8, 2.5).min_dist(0.9));
            flush();
            let g = g.amp(0.33);
            flush();
        "#,
        );
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        let mut before_update: Vec<(u64, f32)> = Vec::new();
        let mut after_update: Vec<(u64, f32)> = Vec::new();

        for event in &scenario.events {
            for action in &event.actions {
                pop.apply_action(action.clone(), &landscape, None);
            }
            if event
                .actions
                .iter()
                .any(|action| matches!(action, Action::Spawn { .. }))
            {
                for (idx, agent) in pop.individuals.iter_mut().enumerate() {
                    let freq_hz = 220.0 * (idx as f32 + 1.0);
                    agent.force_set_pitch_log2(freq_hz.log2());
                }
                before_update = pop
                    .individuals
                    .iter()
                    .map(|agent| (agent.id(), agent.body.base_freq_hz()))
                    .collect();
                before_update.sort_by_key(|(id, _)| *id);
            }
            if event
                .actions
                .iter()
                .any(|action| matches!(action, Action::UpdateGroup { .. }))
            {
                after_update = pop
                    .individuals
                    .iter()
                    .map(|agent| (agent.id(), agent.body.base_freq_hz()))
                    .collect();
                after_update.sort_by_key(|(id, _)| *id);
            }
        }

        assert_eq!(before_update.len(), 4);
        assert_eq!(before_update.len(), after_update.len());

        for ((id_a, freq_a), (id_b, freq_b)) in before_update.iter().zip(after_update.iter()) {
            assert_eq!(*id_a, *id_b);
            assert!(
                (freq_a - freq_b).abs() <= 1e-6,
                "amp-only live update must not modify member pitch center"
            );
        }
    }

    #[test]
    fn group_live_update_last_write_wins_within_flush() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(sine, 1);
            flush();
            let g = g.amp(0.2).amp(0.8);
            flush();
            let g = g.crowding(1.0, 35.0).crowding(1.0);
            flush();
            let g = g.crowding(1.0).crowding(1.0, 35.0);
            flush();
        "#,
        );

        let updates = scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
            .filter_map(|action| match action {
                Action::UpdateGroup { patch, .. } => Some(patch.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(updates.len(), 3, "expected three live update flushes");
        assert_eq!(updates[0].amp, Some(0.8));
        assert_eq!(updates[1].crowding_strength, Some(1.0));
        assert_eq!(updates[1].crowding_sigma_from_roughness, Some(true));
        assert_eq!(updates[2].crowding_strength, Some(1.0));
        assert_eq!(updates[2].crowding_sigma_from_roughness, Some(false));
        assert_eq!(updates[2].crowding_sigma_cents, Some(35.0));
    }

    #[test]
    fn flush_emits_update_before_release_for_same_group() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(sine, 1);
            flush();
            let g = g.amp(0.25);
            release(g);
            flush();
        "#,
        );

        let event = scenario
            .events
            .iter()
            .find(|event| {
                event
                    .actions
                    .iter()
                    .any(|action| matches!(action, Action::UpdateGroup { .. }))
                    && event
                        .actions
                        .iter()
                        .any(|action| matches!(action, Action::ReleaseGroup { .. }))
            })
            .expect("event with both update and release");

        let update_idx = event
            .actions
            .iter()
            .position(|action| matches!(action, Action::UpdateGroup { .. }))
            .expect("update action");
        let release_idx = event
            .actions
            .iter()
            .position(|action| matches!(action, Action::ReleaseGroup { .. }))
            .expect("release action");
        assert!(
            update_idx < release_idx,
            "flush must emit update before release within same event"
        );
    }

    #[test]
    fn group_exploration_persistence_live_update_reaches_individual() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(sine, 1);
            flush();
            let g = g.exploration(0.75).persistence(0.1);
            flush();
        "#,
        );
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            match action {
                Action::Spawn { .. } | Action::UpdateGroup { .. } => {
                    pop.apply_action(action.clone(), &landscape, None);
                }
                _ => {}
            }
        }
        let agent = pop.individuals.first().expect("spawned");
        assert!((agent.pitch_core_for_test().exploration_for_test() - 0.75).abs() <= 1e-6);
        assert!((agent.pitch_core_for_test().persistence_for_test() - 0.1).abs() <= 1e-6);
    }

    #[test]
    fn species_crowding_sets_spawn_control() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine.crowding(1.2, 35.0), 1);
            flush();
        "#,
        );
        let (strength, sigma_cents) = scenario
            .events
            .iter()
            .flat_map(|event| &event.actions)
            .find_map(|action| match action {
                Action::Spawn { spec, .. } => Some((
                    spec.control.pitch.crowding_strength,
                    spec.control.pitch.crowding_sigma_cents,
                )),
                _ => None,
            })
            .expect("spawn action");
        assert!((strength - 1.2).abs() <= 1e-6);
        assert!((sigma_cents - 35.0).abs() <= 1e-6);
    }

    #[test]
    fn species_crowding_reaches_spawned_core() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine.crowding(1.2, 35.0), 1);
            flush();
        "#,
        );
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            if let Action::Spawn { .. } = action {
                pop.apply_action(action.clone(), &landscape, None);
            }
        }
        let agent = pop.individuals.first().expect("spawned");
        assert!((agent.pitch_core_for_test().crowding_strength_for_test() - 1.2).abs() <= 1e-6);
        assert!((agent.pitch_core_for_test().crowding_sigma_cents_for_test() - 35.0).abs() <= 1e-3);
    }

    #[test]
    fn species_crowding_single_arg_uses_default_sigma() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine.crowding(0.8), 1);
            flush();
        "#,
        );
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            if let Action::Spawn { .. } = action {
                pop.apply_action(action.clone(), &landscape, None);
            }
        }
        let agent = pop.individuals.first().expect("spawned");
        assert!((agent.pitch_core_for_test().crowding_strength_for_test() - 0.8).abs() <= 1e-6);
        assert!((agent.pitch_core_for_test().crowding_sigma_cents_for_test() - 60.0).abs() <= 1e-3);
        assert!(
            agent
                .pitch_core_for_test()
                .crowding_sigma_from_roughness_for_test()
        );
    }

    #[test]
    fn species_crowding_mixed_numeric_overloads_work() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine.crowding(1, 35.0), 1);
            create(sine.crowding(1.0, 35), 1);
            flush();
        "#,
        );
        let mut seen = 0usize;
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            if let Action::Spawn { spec, .. } = action {
                assert!((spec.control.pitch.crowding_strength - 1.0).abs() <= 1e-6);
                assert!((spec.control.pitch.crowding_sigma_cents - 35.0).abs() <= 1e-6);
                seen += 1;
            }
        }
        assert_eq!(seen, 2);
    }

    #[test]
    fn group_crowding_live_update_reaches_individual_core() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(sine, 1);
            flush();
            let g = g.crowding(0.8, 25.0);
            flush();
        "#,
        );
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            match action {
                Action::Spawn { .. } | Action::UpdateGroup { .. } => {
                    pop.apply_action(action.clone(), &landscape, None);
                }
                _ => {}
            }
        }
        let agent = pop.individuals.first().expect("spawned");
        assert!((agent.pitch_core_for_test().crowding_strength_for_test() - 0.8).abs() <= 1e-6);
        assert!((agent.pitch_core_for_test().crowding_sigma_cents_for_test() - 25.0).abs() <= 1e-3);
    }

    #[test]
    fn group_crowding_mixed_numeric_overloads_work() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(sine, 1);
            flush();
            let g = g.crowding(1, 35.0);
            flush();
            let g = g.crowding(1.0, 35);
            flush();
        "#,
        );
        let mut updates = 0usize;
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            if let Action::UpdateGroup { patch, .. } = action
                && let (Some(strength), Some(sigma)) =
                    (patch.crowding_strength, patch.crowding_sigma_cents)
            {
                assert!((strength - 1.0).abs() <= 1e-6);
                assert!((sigma - 35.0).abs() <= 1e-6);
                updates += 1;
            }
        }
        assert_eq!(updates, 2);
    }

    #[test]
    fn group_crowding_target_emits_actions_for_draft_and_live_updates() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(sine.crowding_target(true, false), 1);
            flush();
            let g = g.crowding_target(true, true);
            flush();
        "#,
        );
        let mut saw_spawn_target = false;
        let mut saw_live_target = false;
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            if let Action::SetGroupCrowdingTarget {
                same_group_visible,
                other_group_visible,
                ..
            } = action
            {
                if *same_group_visible && !*other_group_visible {
                    saw_spawn_target = true;
                }
                if *same_group_visible && *other_group_visible {
                    saw_live_target = true;
                }
            }
        }
        assert!(
            saw_spawn_target,
            "expected draft crowding_target to be emitted"
        );
        assert!(
            saw_live_target,
            "expected live crowding_target update to be emitted"
        );
    }

    #[test]
    fn species_leave_self_out_and_anneal_reach_spawned_core() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine.leave_self_out(true).anneal_temp(0.12), 1);
            flush();
        "#,
        );
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            if let Action::Spawn { .. } = action {
                pop.apply_action(action.clone(), &landscape, None);
            }
        }
        let agent = pop.individuals.first().expect("spawned");
        assert!(agent.pitch_core_for_test().leave_self_out_for_test());
        assert!((agent.pitch_core_for_test().anneal_temp_for_test() - 0.12).abs() <= 1e-6);
    }

    #[test]
    fn group_leave_self_out_and_anneal_live_update_reaches_individual() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(sine, 1);
            flush();
            let g = g.leave_self_out(true).anneal_temp(0.2);
            flush();
        "#,
        );
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            match action {
                Action::Spawn { .. } | Action::UpdateGroup { .. } => {
                    pop.apply_action(action.clone(), &landscape, None);
                }
                _ => {}
            }
        }
        let agent = pop.individuals.first().expect("spawned");
        assert!(agent.pitch_core_for_test().leave_self_out_for_test());
        assert!((agent.pitch_core_for_test().anneal_temp_for_test() - 0.2).abs() <= 1e-6);
    }

    #[test]
    fn species_move_cost_and_improvement_threshold_reach_spawned_core() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine.move_cost(0.9).improvement_threshold(0.07), 1);
            flush();
        "#,
        );
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            if let Action::Spawn { .. } = action {
                pop.apply_action(action.clone(), &landscape, None);
            }
        }
        let agent = pop.individuals.first().expect("spawned");
        assert!((agent.pitch_core_for_test().move_cost_coeff_for_test() - 0.9).abs() <= 1e-6);
        assert!(
            (agent.pitch_core_for_test().improvement_threshold_for_test() - 0.07).abs() <= 1e-6
        );
    }

    #[test]
    fn group_move_cost_and_improvement_threshold_live_update_reaches_individual() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(sine, 1);
            flush();
            let g = g.move_cost(0.8).improvement_threshold(0.05);
            flush();
        "#,
        );
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            match action {
                Action::Spawn { .. } | Action::UpdateGroup { .. } => {
                    pop.apply_action(action.clone(), &landscape, None);
                }
                _ => {}
            }
        }
        let agent = pop.individuals.first().expect("spawned");
        assert!((agent.pitch_core_for_test().move_cost_coeff_for_test() - 0.8).abs() <= 1e-6);
        assert!(
            (agent.pitch_core_for_test().improvement_threshold_for_test() - 0.05).abs() <= 1e-6
        );
    }

    #[test]
    fn group_hill_climb_knobs_and_exact_loo_live_update_reach_individual() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(sine, 1);
            flush();
            let g = g
                .neighbor_step_cents(25)
                .tessitura_gravity(0.12)
                .move_cost_exp(2)
                .leave_self_out(true)
                .leave_self_out_mode("exact");
            flush();
        "#,
        );
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            match action {
                Action::Spawn { .. } | Action::UpdateGroup { .. } => {
                    pop.apply_action(action.clone(), &landscape, None);
                }
                _ => {}
            }
        }
        let agent = pop.individuals.first().expect("spawned");
        let core = agent.pitch_core_for_test();
        assert!((core.neighbor_step_cents_for_test() - 25.0).abs() <= 1e-6);
        assert!((core.tessitura_gravity_for_test() - 0.12).abs() <= 1e-6);
        assert_eq!(core.move_cost_exp_for_test(), 2);
        assert!(core.leave_self_out_for_test());
        assert_eq!(
            core.leave_self_out_mode_for_test(),
            LeaveSelfOutMode::ExactScan
        );
        assert_eq!(
            agent.effective_control.pitch.leave_self_out_mode,
            LeaveSelfOutMode::ExactScan
        );
    }

    #[test]
    fn species_peak_sampler_knobs_reach_spawned_core() {
        let (scenario, _warnings) = run_script(
            r#"
            create(
                sine
                    .pitch_core("peak_sampler")
                    .neighbor_step_cents(30)
                    .tessitura_gravity(0.14)
                    .window_cents(320)
                    .top_k(7)
                    .temperature(0.0)
                    .sigma_cents(18)
                    .random_candidates(5),
                1
            );
            flush();
        "#,
        );
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            if let Action::Spawn { .. } = action {
                pop.apply_action(action.clone(), &landscape, None);
            }
        }
        let agent = pop.individuals.first().expect("spawned");
        let core = agent.pitch_core_for_test();
        assert!((core.neighbor_step_cents_for_test() - 30.0).abs() <= 1e-6);
        assert!((core.tessitura_gravity_for_test() - 0.14).abs() <= 1e-6);
        assert!((core.window_cents_for_test() - 320.0).abs() <= 1e-6);
        assert_eq!(core.top_k_for_test(), 7);
        assert!((core.temperature_for_test() - 0.0).abs() <= 1e-6);
        assert!((core.sigma_cents_for_test() - 18.0).abs() <= 1e-6);
        assert_eq!(core.random_candidates_for_test(), 5);
    }

    #[test]
    fn species_advanced_pitch_knobs_reach_spawned_core() {
        let (scenario, _warnings) = run_script(
            r#"
            create(
                sine
                    .proposal_interval(0.3)
                    .global_peaks(12, 40.0)
                    .ratio_candidates(5)
                    .move_cost_time_scale("proposal_interval")
                    .leave_self_out_harmonics(4)
                    .pitch_apply_mode("glide")
                    .pitch_glide(0.08),
                1
            );
            flush();
        "#,
        );
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            if let Action::Spawn { .. } = action {
                pop.apply_action(action.clone(), &landscape, None);
            }
        }
        let agent = pop.individuals.first().expect("spawned");
        let core = agent.pitch_core_for_test();
        assert!((core.proposal_interval_sec_for_test().unwrap_or(0.0) - 0.3).abs() <= 1e-6);
        assert_eq!(core.global_peak_count_for_test(), 12);
        assert_eq!(core.ratio_candidate_count_for_test(), 5);
        assert!(core.use_ratio_candidates_for_test());
        assert_eq!(
            core.move_cost_time_scale_for_test(),
            MoveCostTimeScale::ProposalInterval
        );
        assert_eq!(core.leave_self_out_harmonics_for_test(), 4);
        assert_eq!(
            agent.effective_control.pitch.pitch_apply_mode,
            PitchApplyMode::Glide
        );
        assert!((agent.effective_control.pitch.pitch_glide_tau_sec - 0.08).abs() <= 1e-6);
    }

    #[test]
    fn group_advanced_pitch_knobs_live_update_reaches_individual() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(sine, 1);
            flush();
            let g = g
                .proposal_interval(0.25)
                .global_peaks(10, 30)
                .ratio_candidates(4)
                .move_cost_time_scale("proposal_interval")
                .leave_self_out_harmonics(3)
                .pitch_apply("glide")
                .pitch_glide(0.05);
            flush();
        "#,
        );
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            match action {
                Action::Spawn { .. } | Action::UpdateGroup { .. } => {
                    pop.apply_action(action.clone(), &landscape, None);
                }
                _ => {}
            }
        }
        let agent = pop.individuals.first().expect("spawned");
        let core = agent.pitch_core_for_test();
        assert!((core.proposal_interval_sec_for_test().unwrap_or(0.0) - 0.25).abs() <= 1e-6);
        assert_eq!(core.global_peak_count_for_test(), 10);
        assert_eq!(core.ratio_candidate_count_for_test(), 4);
        assert!(core.use_ratio_candidates_for_test());
        assert_eq!(
            core.move_cost_time_scale_for_test(),
            MoveCostTimeScale::ProposalInterval
        );
        assert_eq!(core.leave_self_out_harmonics_for_test(), 3);
        assert_eq!(
            agent.effective_control.pitch.pitch_apply_mode,
            PitchApplyMode::Glide
        );
        assert!((agent.effective_control.pitch.pitch_glide_tau_sec - 0.05).abs() <= 1e-6);
    }

    #[test]
    fn set_pitch_objective_emits_landscape_update() {
        let (scenario, _warnings) = run_script(
            r#"
            set_pitch_objective("negative_consonance");
            wait(0.1);
            set_pitch_objective("consonance");
        "#,
        );
        let modes: Vec<PitchObjectiveMode> = scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
            .filter_map(|action| match action {
                Action::SetHarmonicityParams { update } => update.pitch_objective_mode,
                _ => None,
            })
            .collect();
        assert_eq!(
            modes,
            vec![
                PitchObjectiveMode::NegativeConsonance,
                PitchObjectiveMode::Consonance
            ]
        );
    }

    #[test]
    fn reject_targets_wraps_spawn_strategy() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(sine, 1)
                .place(reject_targets(random_log(200.0, 400.0), 220, [0, 7, 12], 0.35, 16));
            flush();
        "#,
        );
        let strategy = scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
            .find_map(|action| match action {
                Action::Spawn {
                    strategy: Some(strategy),
                    ..
                } => Some(strategy.clone()),
                _ => None,
            })
            .expect("spawn strategy");
        match strategy {
            SpawnStrategy::RejectTargets {
                base,
                anchor_hz,
                targets_st,
                exclusion_st,
                max_tries,
            } => {
                assert!(matches!(*base, SpawnStrategy::RandomLog { .. }));
                assert!((anchor_hz - 220.0).abs() <= 1e-6);
                assert_eq!(targets_st, vec![0.0, 7.0, 12.0]);
                assert!((exclusion_st - 0.35).abs() <= 1e-6);
                assert_eq!(max_tries, 16);
            }
            other => panic!("expected RejectTargets strategy, got {other:?}"),
        }
    }

    #[test]
    fn group_draft_landscape_weight_sets_spawn_control() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(sine, 1).landscape_weight(0.4);
            flush();
        "#,
        );
        let weight = scenario
            .events
            .iter()
            .flat_map(|event| &event.actions)
            .find_map(|action| match action {
                Action::Spawn { spec, .. } => Some(spec.control.pitch.landscape_weight),
                _ => None,
            })
            .expect("spawn action");
        assert!((weight - 0.4).abs() <= 1e-6);
    }

    #[test]
    fn species_respawn_policy_emits_runtime_action() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine.respawn_hereditary(0.03), 1);
            flush();
        "#,
        );
        let mut saw_policy = false;
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            if let Action::SetRespawnPolicy { group_id, policy } = action {
                assert_eq!(*group_id, 1);
                assert_eq!(*policy, RespawnPolicy::Hereditary { sigma_oct: 0.03 });
                saw_policy = true;
            }
        }
        assert!(saw_policy, "expected SetRespawnPolicy action");
    }

    #[test]
    fn group_draft_respawn_random_emits_runtime_action() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(sine, 1).respawn_random();
            flush();
        "#,
        );
        let mut saw_policy = false;
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            if let Action::SetRespawnPolicy { group_id, policy } = action {
                assert_eq!(*group_id, 1);
                assert_eq!(*policy, RespawnPolicy::Random);
                saw_policy = true;
            }
        }
        assert!(saw_policy, "expected SetRespawnPolicy(Random)");
    }

    #[test]
    fn group_draft_respawn_hereditary_emits_runtime_action() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(sine, 1).respawn_hereditary(0.03);
            flush();
        "#,
        );
        let mut saw_policy = false;
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            if let Action::SetRespawnPolicy { group_id, policy } = action {
                assert_eq!(*group_id, 1);
                assert_eq!(*policy, RespawnPolicy::Hereditary { sigma_oct: 0.03 });
                saw_policy = true;
            }
        }
        assert!(saw_policy, "expected SetRespawnPolicy(Hereditary)");
    }

    #[test]
    fn spawn_payload_preserves_species_control_fields() {
        let (scenario, _warnings) = run_script(
            r#"
            create(harmonic, 1)
                .amp(0.33)
                .freq(330.0)
                .brightness(0.7);
            flush();
        "#,
        );
        let control = scenario
            .events
            .iter()
            .flat_map(|event| &event.actions)
            .find_map(|action| match action {
                Action::Spawn { spec, strategy, .. } => {
                    assert!(strategy.is_none());
                    Some(spec.control.clone())
                }
                _ => None,
            })
            .expect("spawn action");
        assert!((control.body.amp - 0.33).abs() <= 1e-6);
        assert!((control.pitch.freq - 330.0).abs() <= 1e-6);
        assert_eq!(control.pitch.mode, PitchMode::Lock);
        assert!((control.body.timbre.brightness - 0.7).abs() <= 1e-6);
    }

    #[test]
    fn live_group_brightness_emits_timbre_patch() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(harmonic, 1);
            flush();
            g.brightness(0.25);
            flush();
        "#,
        );
        let patch = scenario
            .events
            .iter()
            .flat_map(|event| &event.actions)
            .find_map(|action| match action {
                Action::UpdateGroup { patch, .. } => Some(patch.clone()),
                _ => None,
            })
            .expect("update action");
        assert_eq!(patch.timbre_brightness, Some(0.25));
    }

    #[test]
    fn timbre_method_is_not_registered() {
        let err = run_script_err(
            r#"
            create(harmonic, 1).timbre(0.7, 0.2);
        "#,
        );
        assert!(err.message.contains("timbre"));
    }

    #[test]
    fn width_method_is_not_registered() {
        let err = run_script_err(
            r#"
            create(harmonic, 1).width(0.4);
        "#,
        );
        assert!(err.message.contains("width"));
    }

    #[test]
    fn rhythm_modulators_are_sanitized_at_core_boundary() {
        let (scenario, _warnings) = run_script(
            r#"
            create(
                sine
                    .rhythm_coupling_vitality(-3.0, 2.0)
                    .rhythm_reward(-2.0, "attack_phase_match"),
                1
            );
            flush();
        "#,
        );
        let spawn = scenario
            .events
            .iter()
            .flat_map(|event| &event.actions)
            .find_map(|action| match action {
                Action::Spawn { spec, .. } => Some(spec.clone()),
                _ => None,
            })
            .expect("spawn action");
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let core = AnyArticulationCore::from_config(&spawn.articulation, 48_000.0, 9, &mut rng);
        let AnyArticulationCore::Entrain(core) = core else {
            panic!("expected entrain core");
        };

        assert_eq!(
            core.rhythm_coupling,
            RhythmCouplingMode::TemporalTimesVitality {
                lambda_v: 0.0,
                v_floor: 0.999
            }
        );
        let reward = core.rhythm_reward.expect("expected rhythm reward");
        assert_eq!(reward.rho_t, 0.0);
    }

    #[test]
    fn e4_step_response_script_has_fixed_population_spawns() {
        let script = E4_STEP_RESPONSE_SCRIPT;
        assert!(
            script.contains("let anchor = derive(harmonic)")
                && script.contains("let probe = derive(harmonic)"),
            "E4 step response must use harmonic bodies"
        );
        assert!(
            script.contains(".pitch_mode(\"lock\")"),
            "E4 step response anchor must lock pitch mode"
        );
        let (scenario, _warnings) = run_script(script);

        let mut spawn_actions = 0usize;
        let mut spawned_agents = 0usize;
        let mut mirror_updates = 0usize;
        for action in scenario.events.iter().flat_map(|ev| ev.actions.iter()) {
            match action {
                Action::Spawn { ids, .. } => {
                    spawn_actions += 1;
                    spawned_agents += ids.len();
                }
                Action::SetHarmonicityParams { update } => {
                    if update.mirror.is_some() {
                        mirror_updates += 1;
                    }
                }
                _ => {}
            }
        }

        // One anchor group + one probe group should spawn once each.
        assert_eq!(spawn_actions, 2);
        assert_eq!(spawned_agents, 13);
        assert!(spawn_actions < mirror_updates);
    }

    #[test]
    fn e4_between_runs_script_pairs_spawns_and_releases_per_weight() {
        let script = E4_BETWEEN_RUNS_SCRIPT;
        assert!(
            script.contains("let anchor = derive(harmonic)")
                && script.contains("let probe = derive(harmonic)"),
            "E4 between-runs must use harmonic bodies"
        );
        assert!(
            script.contains(".pitch_mode(\"lock\")"),
            "E4 between-runs anchor must lock pitch mode"
        );
        let (scenario, _warnings) = run_script(script);

        let mut mirror_updates = 0usize;
        let mut spawn_by_group: HashMap<u64, usize> = HashMap::new();
        let mut release_by_group: HashMap<u64, usize> = HashMap::new();

        for action in scenario.events.iter().flat_map(|ev| ev.actions.iter()) {
            match action {
                Action::SetHarmonicityParams { update } => {
                    if update.mirror.is_some() {
                        mirror_updates += 1;
                    }
                }
                Action::Spawn { group_id, .. } => {
                    *spawn_by_group.entry(*group_id).or_insert(0) += 1;
                }
                Action::ReleaseGroup { group_id, .. } => {
                    *release_by_group.entry(*group_id).or_insert(0) += 1;
                }
                _ => {}
            }
        }

        assert_eq!(spawn_by_group.len(), mirror_updates * 2);
        assert_eq!(release_by_group.len(), mirror_updates * 2);
        for (group_id, spawn_count) in &spawn_by_group {
            assert_eq!(*spawn_count, 1, "group {group_id} spawned more than once");
            assert_eq!(
                release_by_group.get(group_id).copied().unwrap_or(0),
                1,
                "group {group_id} does not have exactly one matching release"
            );
        }
    }
}
