use std::collections::BTreeMap;
use std::fs;
use std::sync::{Arc, Mutex};

use rand::random;
use rhai::{Array, Dynamic, Engine, EvalAltResult, FLOAT, FnPtr, INT, NativeCallContext, Position};
use tracing::warn;

use crate::core::mode_pattern::ModePattern;

use super::control::{
    AgentControl, BodyMethod, ControlUpdate, MoveCostTimeScale, PhonationType, PitchApplyMode,
    PitchCoreKind, PitchMode,
};
use super::lifecycle::LifecycleConfig;
use super::scenario::{
    Action, ArticulationCoreConfig, EnvelopeConfig, MetabolismRhythmReward, RespawnPolicy,
    RhythmCouplingMode, RhythmRewardMetric, Scenario, SceneMarker, SpawnSpec, SpawnStrategy,
    TimedEvent,
};

const DEFAULT_RELEASE_SEC: f32 = 0.05;
const DEFAULT_SEQ_DURATION_SEC: f32 = 1.0;

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

#[derive(Clone, Copy, Debug)]
enum BrainKind {
    Entrain,
    Seq,
    Drone,
}

#[derive(Clone, Copy, Debug)]
enum PhonationKind {
    Hold,
    Decay,
    Grain,
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
    brain: BrainKind,
    phonation: Option<PhonationKind>,
    metabolism_rate: Option<f32>,
    adsr: Option<AdsrSpec>,
    rhythm_coupling: RhythmCouplingMode,
    rhythm_reward: Option<MetabolismRhythmReward>,
}

impl SpeciesSpec {
    fn preset(body: BodyMethod) -> Self {
        let mut control = AgentControl::default();
        control.body.method = body;
        Self {
            control,
            respawn_policy: RespawnPolicy::None,
            brain: BrainKind::Entrain,
            phonation: None,
            metabolism_rate: None,
            adsr: None,
            rhythm_coupling: RhythmCouplingMode::TemporalOnly,
            rhythm_reward: None,
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
        if let Some(phonation) = self.phonation {
            control.phonation.r#type = match phonation {
                PhonationKind::Hold => PhonationType::Hold,
                PhonationKind::Decay => PhonationType::Interval,
                PhonationKind::Grain => PhonationType::Field,
            };
        }
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

    fn set_repulsion(&mut self, strength: f32, sigma_cents: f32) {
        self.control.set_repulsion_strength_clamped(strength);
        self.control.set_repulsion_sigma_cents_clamped(sigma_cents);
    }

    fn set_leave_self_out(&mut self, enabled: bool) {
        self.control.set_leave_self_out(enabled);
    }

    fn set_anneal_temp(&mut self, value: f32) {
        self.control.set_anneal_temp_clamped(value);
    }

    fn set_move_cost_coeff(&mut self, value: f32) {
        self.control.set_move_cost_coeff_clamped(value);
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

    fn set_timbre(&mut self, brightness: f32, width: f32) {
        self.control.set_timbre_brightness_clamped(brightness);
        self.control.set_timbre_width_clamped(width);
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

    fn set_phonation(&mut self, name: &str) {
        let lowered = name.trim().to_ascii_lowercase();
        self.phonation = match lowered.as_str() {
            "hold" => Some(PhonationKind::Hold),
            "decay" => Some(PhonationKind::Decay),
            "grain" => Some(PhonationKind::Grain),
            other => {
                warn!("phonation '{}' not supported yet", other);
                self.phonation
            }
        };
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
    strategy: Option<SpawnStrategy>,
    status: GroupStatus,
    live_ids: Vec<u64>,
    pending_update: ControlUpdate,
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
                    let ids = group.live_ids.clone();
                    if !ids.is_empty() {
                        releases.push(Action::Release {
                            group_id,
                            ids,
                            fade_sec: group.spec.release_sec(),
                        });
                    }
                    group.pending_update = ControlUpdate::default();
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
                if let Action::Release { fade_sec, .. } = action {
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
                    if !matches!(group.respawn_policy, RespawnPolicy::None) {
                        spawn_actions.push(Action::SetRespawnPolicy {
                            group_id: group.id,
                            policy: group.respawn_policy,
                        });
                    }
                }
            }
        }

        let mut update_actions = Vec::new();
        for group in self.groups.values_mut() {
            if !matches!(group.status, GroupStatus::Live) {
                continue;
            }
            if !group.pending_update.is_empty() {
                if !group.live_ids.is_empty() {
                    update_actions.push(Action::Update {
                        group_id: group.id,
                        ids: group.live_ids.clone(),
                        update: group.pending_update.clone(),
                    });
                }
                group.pending_update = ControlUpdate::default();
            }
        }

        let mut release_actions = Vec::new();
        for group in self.groups.values_mut() {
            if !matches!(group.status, GroupStatus::Live) {
                continue;
            }
            if group.pending_release {
                if !group.live_ids.is_empty() {
                    release_actions.push(Action::Release {
                        group_id: group.id,
                        ids: group.live_ids.clone(),
                        fade_sec: group.spec.release_sec(),
                    });
                }
                group.pending_release = false;
                group.status = GroupStatus::Released;
            }
        }

        if !spawn_actions.is_empty() || !update_actions.is_empty() || !release_actions.is_empty() {
            let mut actions = Vec::with_capacity(
                spawn_actions.len() + update_actions.len() + release_actions.len(),
            );
            actions.extend(spawn_actions);
            actions.extend(update_actions);
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
            spec: species.spec,
            strategy: None,
            status: GroupStatus::Draft,
            live_ids: Vec::new(),
            pending_update: ControlUpdate::default(),
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
                    spec.control.body.timbre.width = 0.2;
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
                    spec.control.body.timbre.width = 0.1;
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
                    spec.control.body.timbre.width = 0.35;
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

        engine.register_fn("amp", |mut species: SpeciesHandle, value: FLOAT| {
            species.spec.set_amp(value as f32);
            species
        });
        engine.register_fn("amp", |mut species: SpeciesHandle, value: INT| {
            species.spec.set_amp(value as f32);
            species
        });
        engine.register_fn("freq", |mut species: SpeciesHandle, value: FLOAT| {
            species.spec.set_freq(value as f32);
            species
        });
        engine.register_fn("freq", |mut species: SpeciesHandle, value: INT| {
            species.spec.set_freq(value as f32);
            species
        });
        engine.register_fn(
            "landscape_weight",
            |mut species: SpeciesHandle, value: FLOAT| {
                species.spec.set_landscape_weight(value as f32);
                species
            },
        );
        engine.register_fn(
            "landscape_weight",
            |mut species: SpeciesHandle, value: INT| {
                species.spec.set_landscape_weight(value as f32);
                species
            },
        );
        engine.register_fn(
            "sustain_drive",
            |mut species: SpeciesHandle, value: FLOAT| {
                species.spec.set_continuous_drive(value as f32);
                species
            },
        );
        engine.register_fn("sustain_drive", |mut species: SpeciesHandle, value: INT| {
            species.spec.set_continuous_drive(value as f32);
            species
        });
        engine.register_fn(
            "pitch_smooth",
            |mut species: SpeciesHandle, value: FLOAT| {
                species.spec.set_pitch_smooth_tau(value as f32);
                species
            },
        );
        engine.register_fn("pitch_smooth", |mut species: SpeciesHandle, value: INT| {
            species.spec.set_pitch_smooth_tau(value as f32);
            species
        });
        engine.register_fn("exploration", |mut species: SpeciesHandle, value: FLOAT| {
            species.spec.set_exploration(value as f32);
            species
        });
        engine.register_fn("exploration", |mut species: SpeciesHandle, value: INT| {
            species.spec.set_exploration(value as f32);
            species
        });
        engine.register_fn("persistence", |mut species: SpeciesHandle, value: FLOAT| {
            species.spec.set_persistence(value as f32);
            species
        });
        engine.register_fn("persistence", |mut species: SpeciesHandle, value: INT| {
            species.spec.set_persistence(value as f32);
            species
        });
        engine.register_fn(
            "repulsion",
            |mut species: SpeciesHandle, strength: FLOAT, sigma_cents: FLOAT| {
                species
                    .spec
                    .set_repulsion(strength as f32, sigma_cents as f32);
                species
            },
        );
        engine.register_fn(
            "repulsion",
            |mut species: SpeciesHandle, strength: INT, sigma_cents: FLOAT| {
                species
                    .spec
                    .set_repulsion(strength as f32, sigma_cents as f32);
                species
            },
        );
        engine.register_fn(
            "repulsion",
            |mut species: SpeciesHandle, strength: FLOAT, sigma_cents: INT| {
                species
                    .spec
                    .set_repulsion(strength as f32, sigma_cents as f32);
                species
            },
        );
        engine.register_fn(
            "repulsion",
            |mut species: SpeciesHandle, strength: INT, sigma_cents: INT| {
                species
                    .spec
                    .set_repulsion(strength as f32, sigma_cents as f32);
                species
            },
        );
        engine.register_fn(
            "repulsion",
            |mut species: SpeciesHandle, strength: FLOAT| {
                species.spec.set_repulsion(strength as f32, 60.0);
                species
            },
        );
        engine.register_fn("repulsion", |mut species: SpeciesHandle, strength: INT| {
            species.spec.set_repulsion(strength as f32, 60.0);
            species
        });
        engine.register_fn(
            "leave_self_out",
            |mut species: SpeciesHandle, enabled: bool| {
                species.spec.set_leave_self_out(enabled);
                species
            },
        );
        engine.register_fn("loo", |mut species: SpeciesHandle, enabled: bool| {
            species.spec.set_leave_self_out(enabled);
            species
        });
        engine.register_fn("anneal_temp", |mut species: SpeciesHandle, value: FLOAT| {
            species.spec.set_anneal_temp(value as f32);
            species
        });
        engine.register_fn("anneal_temp", |mut species: SpeciesHandle, value: INT| {
            species.spec.set_anneal_temp(value as f32);
            species
        });
        engine.register_fn("move_cost", |mut species: SpeciesHandle, value: FLOAT| {
            species.spec.set_move_cost_coeff(value as f32);
            species
        });
        engine.register_fn("move_cost", |mut species: SpeciesHandle, value: INT| {
            species.spec.set_move_cost_coeff(value as f32);
            species
        });
        engine.register_fn(
            "improvement_threshold",
            |mut species: SpeciesHandle, value: FLOAT| {
                species.spec.set_improvement_threshold(value as f32);
                species
            },
        );
        engine.register_fn(
            "improvement_threshold",
            |mut species: SpeciesHandle, value: INT| {
                species.spec.set_improvement_threshold(value as f32);
                species
            },
        );
        engine.register_fn(
            "proposal_interval",
            |mut species: SpeciesHandle, value: FLOAT| {
                species.spec.set_proposal_interval_sec(value as f32);
                species
            },
        );
        engine.register_fn(
            "proposal_interval",
            |mut species: SpeciesHandle, value: INT| {
                species.spec.set_proposal_interval_sec(value as f32);
                species
            },
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
        engine.register_fn("pitch_glide", |mut species: SpeciesHandle, value: FLOAT| {
            species.spec.set_pitch_glide_tau_sec(value as f32);
            species
        });
        engine.register_fn("pitch_glide", |mut species: SpeciesHandle, value: INT| {
            species.spec.set_pitch_glide_tau_sec(value as f32);
            species
        });
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
        engine.register_fn("phonation", |mut species: SpeciesHandle, name: &str| {
            species.spec.set_phonation(name);
            species
        });
        engine.register_fn(
            "timbre",
            |mut species: SpeciesHandle, brightness: FLOAT, width: FLOAT| {
                species.spec.set_timbre(brightness as f32, width as f32);
                species
            },
        );
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
                pattern.with_range(min_mul as f32, max_mul as f32)
            },
        );
        engine.register_fn("min_dist", |pattern: ModePattern, min_dist: FLOAT| {
            pattern.with_min_dist_erb(min_dist as f32)
        });
        engine.register_fn("gamma", |pattern: ModePattern, gamma: FLOAT| {
            pattern.with_gamma(gamma as f32)
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

        let ctx_for_group_amp = ctx.clone();
        engine.register_fn(
            "amp",
            move |handle: GroupHandle, value: FLOAT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_amp.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("amp ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => {
                        group.spec.set_amp(value);
                    }
                    GroupStatus::Live => {
                        group.spec.set_amp(value);
                        group.pending_update.amp = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "amp"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_amp_int = ctx.clone();
        engine.register_fn(
            "amp",
            move |handle: GroupHandle, value: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_amp_int.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("amp ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => {
                        group.spec.set_amp(value);
                    }
                    GroupStatus::Live => {
                        group.spec.set_amp(value);
                        group.pending_update.amp = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "amp"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_freq = ctx.clone();
        engine.register_fn(
            "freq",
            move |handle: GroupHandle, value: FLOAT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_freq.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("freq ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => {
                        group.strategy = None;
                        group.spec.set_freq(value);
                    }
                    GroupStatus::Live => {
                        group.spec.set_freq(value);
                        group.pending_update.freq = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "freq"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_freq_int = ctx.clone();
        engine.register_fn(
            "freq",
            move |handle: GroupHandle, value: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_freq_int.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("freq ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => {
                        group.strategy = None;
                        group.spec.set_freq(value);
                    }
                    GroupStatus::Live => {
                        group.spec.set_freq(value);
                        group.pending_update.freq = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "freq"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_landscape_weight = ctx.clone();
        engine.register_fn(
            "landscape_weight",
            move |handle: GroupHandle, value: FLOAT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_landscape_weight
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("landscape_weight ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => {
                        group.spec.set_landscape_weight(value);
                    }
                    GroupStatus::Live => {
                        group.spec.set_landscape_weight(value);
                        group.pending_update.landscape_weight = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "landscape_weight"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_landscape_weight_int = ctx.clone();
        engine.register_fn(
            "landscape_weight",
            move |handle: GroupHandle, value: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_landscape_weight_int
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("landscape_weight ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => {
                        group.spec.set_landscape_weight(value);
                    }
                    GroupStatus::Live => {
                        group.spec.set_landscape_weight(value);
                        group.pending_update.landscape_weight = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "landscape_weight"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_sustain_drive = ctx.clone();
        engine.register_fn(
            "sustain_drive",
            move |handle: GroupHandle, value: FLOAT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_sustain_drive
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("sustain_drive ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => {
                        group.spec.set_continuous_drive(value);
                    }
                    GroupStatus::Live => {
                        group.spec.set_continuous_drive(value);
                        group.pending_update.continuous_drive = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "sustain_drive"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_sustain_drive_int = ctx.clone();
        engine.register_fn(
            "sustain_drive",
            move |handle: GroupHandle, value: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_sustain_drive_int
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("sustain_drive ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => {
                        group.spec.set_continuous_drive(value);
                    }
                    GroupStatus::Live => {
                        group.spec.set_continuous_drive(value);
                        group.pending_update.continuous_drive = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "sustain_drive"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_pitch_smooth = ctx.clone();
        engine.register_fn(
            "pitch_smooth",
            move |handle: GroupHandle, value: FLOAT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_pitch_smooth
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("pitch_smooth ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => {
                        group.spec.set_pitch_smooth_tau(value);
                    }
                    GroupStatus::Live => {
                        group.spec.set_pitch_smooth_tau(value);
                        group.pending_update.pitch_smooth_tau = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "pitch_smooth"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_pitch_smooth_int = ctx.clone();
        engine.register_fn(
            "pitch_smooth",
            move |handle: GroupHandle, value: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_pitch_smooth_int
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("pitch_smooth ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => {
                        group.spec.set_pitch_smooth_tau(value);
                    }
                    GroupStatus::Live => {
                        group.spec.set_pitch_smooth_tau(value);
                        group.pending_update.pitch_smooth_tau = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "pitch_smooth"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_exploration = ctx.clone();
        engine.register_fn(
            "exploration",
            move |handle: GroupHandle, value: FLOAT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_exploration
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("exploration ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_exploration(value),
                    GroupStatus::Live => {
                        group.spec.set_exploration(value);
                        group.pending_update.exploration = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "exploration"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_exploration_int = ctx.clone();
        engine.register_fn(
            "exploration",
            move |handle: GroupHandle, value: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_exploration_int
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("exploration ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_exploration(value),
                    GroupStatus::Live => {
                        group.spec.set_exploration(value);
                        group.pending_update.exploration = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "exploration"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_persistence = ctx.clone();
        engine.register_fn(
            "persistence",
            move |handle: GroupHandle, value: FLOAT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_persistence
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("persistence ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_persistence(value),
                    GroupStatus::Live => {
                        group.spec.set_persistence(value);
                        group.pending_update.persistence = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "persistence"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_persistence_int = ctx.clone();
        engine.register_fn(
            "persistence",
            move |handle: GroupHandle, value: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_persistence_int
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("persistence ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_persistence(value),
                    GroupStatus::Live => {
                        group.spec.set_persistence(value);
                        group.pending_update.persistence = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "persistence"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_repulsion = ctx.clone();
        engine.register_fn(
            "repulsion",
            move |handle: GroupHandle,
                  strength: FLOAT,
                  sigma_cents: FLOAT|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_repulsion.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("repulsion ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let strength = strength as f32;
                let sigma_cents = sigma_cents as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_repulsion(strength, sigma_cents),
                    GroupStatus::Live => {
                        group.spec.set_repulsion(strength, sigma_cents);
                        group.pending_update.repulsion_strength = Some(strength);
                        group.pending_update.repulsion_sigma_cents = Some(sigma_cents);
                    }
                    _ => ctx.warn_live_builder(handle.id, "repulsion"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_repulsion_int_float = ctx.clone();
        engine.register_fn(
            "repulsion",
            move |handle: GroupHandle,
                  strength: INT,
                  sigma_cents: FLOAT|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_repulsion_int_float
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("repulsion ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let strength = strength as f32;
                let sigma_cents = sigma_cents as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_repulsion(strength, sigma_cents),
                    GroupStatus::Live => {
                        group.spec.set_repulsion(strength, sigma_cents);
                        group.pending_update.repulsion_strength = Some(strength);
                        group.pending_update.repulsion_sigma_cents = Some(sigma_cents);
                    }
                    _ => ctx.warn_live_builder(handle.id, "repulsion"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_repulsion_float_int = ctx.clone();
        engine.register_fn(
            "repulsion",
            move |handle: GroupHandle,
                  strength: FLOAT,
                  sigma_cents: INT|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_repulsion_float_int
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("repulsion ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let strength = strength as f32;
                let sigma_cents = sigma_cents as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_repulsion(strength, sigma_cents),
                    GroupStatus::Live => {
                        group.spec.set_repulsion(strength, sigma_cents);
                        group.pending_update.repulsion_strength = Some(strength);
                        group.pending_update.repulsion_sigma_cents = Some(sigma_cents);
                    }
                    _ => ctx.warn_live_builder(handle.id, "repulsion"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_repulsion_default = ctx.clone();
        engine.register_fn(
            "repulsion",
            move |handle: GroupHandle,
                  strength: FLOAT|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_repulsion_default
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("repulsion ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let strength = strength as f32;
                let sigma_cents = 60.0;
                match group.status {
                    GroupStatus::Draft => group.spec.set_repulsion(strength, sigma_cents),
                    GroupStatus::Live => {
                        group.spec.set_repulsion(strength, sigma_cents);
                        group.pending_update.repulsion_strength = Some(strength);
                        group.pending_update.repulsion_sigma_cents = Some(sigma_cents);
                    }
                    _ => ctx.warn_live_builder(handle.id, "repulsion"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_repulsion_int = ctx.clone();
        engine.register_fn(
            "repulsion",
            move |handle: GroupHandle,
                  strength: INT,
                  sigma_cents: INT|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_repulsion_int
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("repulsion ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let strength = strength as f32;
                let sigma_cents = sigma_cents as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_repulsion(strength, sigma_cents),
                    GroupStatus::Live => {
                        group.spec.set_repulsion(strength, sigma_cents);
                        group.pending_update.repulsion_strength = Some(strength);
                        group.pending_update.repulsion_sigma_cents = Some(sigma_cents);
                    }
                    _ => ctx.warn_live_builder(handle.id, "repulsion"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_repulsion_default_int = ctx.clone();
        engine.register_fn(
            "repulsion",
            move |handle: GroupHandle, strength: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_repulsion_default_int
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("repulsion ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let strength = strength as f32;
                let sigma_cents = 60.0;
                match group.status {
                    GroupStatus::Draft => group.spec.set_repulsion(strength, sigma_cents),
                    GroupStatus::Live => {
                        group.spec.set_repulsion(strength, sigma_cents);
                        group.pending_update.repulsion_strength = Some(strength);
                        group.pending_update.repulsion_sigma_cents = Some(sigma_cents);
                    }
                    _ => ctx.warn_live_builder(handle.id, "repulsion"),
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
                        group.pending_update.leave_self_out = Some(enabled);
                    }
                    _ => ctx.warn_live_builder(handle.id, "leave_self_out"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_loo = ctx.clone();
        engine.register_fn(
            "loo",
            move |handle: GroupHandle, enabled: bool| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_loo.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("loo ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_leave_self_out(enabled),
                    GroupStatus::Live => {
                        group.spec.set_leave_self_out(enabled);
                        group.pending_update.leave_self_out = Some(enabled);
                    }
                    _ => ctx.warn_live_builder(handle.id, "loo"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_anneal_temp = ctx.clone();
        engine.register_fn(
            "anneal_temp",
            move |handle: GroupHandle, value: FLOAT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_anneal_temp
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("anneal_temp ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_anneal_temp(value),
                    GroupStatus::Live => {
                        group.spec.set_anneal_temp(value);
                        group.pending_update.anneal_temp = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "anneal_temp"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_anneal_temp_int = ctx.clone();
        engine.register_fn(
            "anneal_temp",
            move |handle: GroupHandle, value: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_anneal_temp_int
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("anneal_temp ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_anneal_temp(value),
                    GroupStatus::Live => {
                        group.spec.set_anneal_temp(value);
                        group.pending_update.anneal_temp = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "anneal_temp"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_move_cost = ctx.clone();
        engine.register_fn(
            "move_cost",
            move |handle: GroupHandle, value: FLOAT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_move_cost.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("move_cost ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_move_cost_coeff(value),
                    GroupStatus::Live => {
                        group.spec.set_move_cost_coeff(value);
                        group.pending_update.move_cost_coeff = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "move_cost"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_move_cost_int = ctx.clone();
        engine.register_fn(
            "move_cost",
            move |handle: GroupHandle, value: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_move_cost_int
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("move_cost ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_move_cost_coeff(value),
                    GroupStatus::Live => {
                        group.spec.set_move_cost_coeff(value);
                        group.pending_update.move_cost_coeff = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "move_cost"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_imp_thresh = ctx.clone();
        engine.register_fn(
            "improvement_threshold",
            move |handle: GroupHandle, value: FLOAT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_imp_thresh
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!(
                        "improvement_threshold ignored for unknown group {}",
                        handle.id
                    );
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_improvement_threshold(value),
                    GroupStatus::Live => {
                        group.spec.set_improvement_threshold(value);
                        group.pending_update.improvement_threshold = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "improvement_threshold"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_imp_thresh_int = ctx.clone();
        engine.register_fn(
            "improvement_threshold",
            move |handle: GroupHandle, value: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_imp_thresh_int
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!(
                        "improvement_threshold ignored for unknown group {}",
                        handle.id
                    );
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_improvement_threshold(value),
                    GroupStatus::Live => {
                        group.spec.set_improvement_threshold(value);
                        group.pending_update.improvement_threshold = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "improvement_threshold"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_proposal_interval = ctx.clone();
        engine.register_fn(
            "proposal_interval",
            move |handle: GroupHandle, value: FLOAT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_proposal_interval
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("proposal_interval ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_proposal_interval_sec(value),
                    GroupStatus::Live => {
                        group.spec.set_proposal_interval_sec(value);
                        group.pending_update.proposal_interval_sec = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "proposal_interval"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_proposal_interval_int = ctx.clone();
        engine.register_fn(
            "proposal_interval",
            move |handle: GroupHandle, value: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_proposal_interval_int
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("proposal_interval ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_proposal_interval_sec(value),
                    GroupStatus::Live => {
                        group.spec.set_proposal_interval_sec(value);
                        group.pending_update.proposal_interval_sec = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "proposal_interval"),
                }
                Ok(handle)
            },
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
                        group.pending_update.global_peak_count = Some(count);
                        group.pending_update.global_peak_min_sep_cents = Some(0.0);
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
                        group.pending_update.global_peak_count = Some(count);
                        group.pending_update.global_peak_min_sep_cents = Some(min_sep);
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
                        group.pending_update.global_peak_count = Some(count);
                        group.pending_update.global_peak_min_sep_cents = Some(min_sep);
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
                        group.pending_update.ratio_candidate_count = Some(count);
                        group.pending_update.use_ratio_candidates = Some(count > 0);
                    }
                    _ => ctx.warn_live_builder(handle.id, "ratio_candidates"),
                }
                Ok(handle)
            },
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
                        group.pending_update.move_cost_time_scale = Some(value);
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
                        group.pending_update.leave_self_out_harmonics = Some(value);
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
                        group.pending_update.pitch_apply_mode = Some(mode);
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
                        group.pending_update.pitch_apply_mode = Some(mode);
                    }
                    _ => ctx.warn_live_builder(handle.id, "pitch_apply"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_pitch_glide = ctx.clone();
        engine.register_fn(
            "pitch_glide",
            move |handle: GroupHandle, value: FLOAT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_pitch_glide
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("pitch_glide ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_pitch_glide_tau_sec(value),
                    GroupStatus::Live => {
                        group.spec.set_pitch_glide_tau_sec(value);
                        group.pending_update.pitch_glide_tau_sec = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "pitch_glide"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_pitch_glide_int = ctx.clone();
        engine.register_fn(
            "pitch_glide",
            move |handle: GroupHandle, value: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_pitch_glide_int
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("pitch_glide ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let value = value as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_pitch_glide_tau_sec(value),
                    GroupStatus::Live => {
                        group.spec.set_pitch_glide_tau_sec(value);
                        group.pending_update.pitch_glide_tau_sec = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "pitch_glide"),
                }
                Ok(handle)
            },
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
        let ctx_for_group_phonation = ctx.clone();
        engine.register_fn(
            "phonation",
            move |handle: GroupHandle, name: &str| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_phonation.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("phonation ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_phonation(name),
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "phonation"),
                    _ => ctx.warn_live_builder(handle.id, "phonation"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_timbre = ctx.clone();
        engine.register_fn(
            "timbre",
            move |handle: GroupHandle,
                  brightness: FLOAT,
                  width: FLOAT|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_timbre.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("timbre ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let brightness = brightness as f32;
                let width = width as f32;
                match group.status {
                    GroupStatus::Draft => {
                        group.spec.set_timbre(brightness, width);
                    }
                    GroupStatus::Live => {
                        group.spec.set_timbre(brightness, width);
                        group.pending_update.timbre_brightness = Some(brightness);
                        group.pending_update.timbre_width = Some(width);
                    }
                    _ => ctx.warn_live_builder(handle.id, "timbre"),
                }
                Ok(handle)
            },
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
            if matches!(action, Action::Release { .. }) {
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
                if let Action::Release { fade_sec, .. } = action {
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
            if let Action::Update { update, .. } = action
                && update.landscape_weight == Some(0.4)
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
                Action::Spawn { .. } | Action::Update { .. } => {
                    pop.apply_action(action.clone(), &landscape, None);
                }
                _ => {}
            }
        }
        let agent = pop.individuals.first().expect("spawned");
        assert!((agent.effective_control.pitch.landscape_weight - 0.6).abs() <= 1e-6);
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
                Action::Spawn { .. } | Action::Update { .. } => {
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
    fn species_repulsion_sets_spawn_control() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine.repulsion(1.2, 35.0), 1);
            flush();
        "#,
        );
        let (strength, sigma_cents) = scenario
            .events
            .iter()
            .flat_map(|event| &event.actions)
            .find_map(|action| match action {
                Action::Spawn { spec, .. } => Some((
                    spec.control.pitch.repulsion_strength,
                    spec.control.pitch.repulsion_sigma_cents,
                )),
                _ => None,
            })
            .expect("spawn action");
        assert!((strength - 1.2).abs() <= 1e-6);
        assert!((sigma_cents - 35.0).abs() <= 1e-6);
    }

    #[test]
    fn species_repulsion_reaches_spawned_core() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine.repulsion(1.2, 35.0), 1);
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
        assert!((agent.pitch_core_for_test().repulsion_strength_for_test() - 1.2).abs() <= 1e-6);
        assert!(
            (agent.pitch_core_for_test().repulsion_sigma_cents_for_test() - 35.0).abs() <= 1e-3
        );
    }

    #[test]
    fn species_repulsion_single_arg_uses_default_sigma() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine.repulsion(0.8), 1);
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
        assert!((agent.pitch_core_for_test().repulsion_strength_for_test() - 0.8).abs() <= 1e-6);
        assert!(
            (agent.pitch_core_for_test().repulsion_sigma_cents_for_test() - 60.0).abs() <= 1e-3
        );
    }

    #[test]
    fn species_repulsion_mixed_numeric_overloads_work() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine.repulsion(1, 35.0), 1);
            create(sine.repulsion(1.0, 35), 1);
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
                assert!((spec.control.pitch.repulsion_strength - 1.0).abs() <= 1e-6);
                assert!((spec.control.pitch.repulsion_sigma_cents - 35.0).abs() <= 1e-6);
                seen += 1;
            }
        }
        assert_eq!(seen, 2);
    }

    #[test]
    fn group_repulsion_live_update_reaches_individual_core() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(sine, 1);
            flush();
            let g = g.repulsion(0.8, 25.0);
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
                Action::Spawn { .. } | Action::Update { .. } => {
                    pop.apply_action(action.clone(), &landscape, None);
                }
                _ => {}
            }
        }
        let agent = pop.individuals.first().expect("spawned");
        assert!((agent.pitch_core_for_test().repulsion_strength_for_test() - 0.8).abs() <= 1e-6);
        assert!(
            (agent.pitch_core_for_test().repulsion_sigma_cents_for_test() - 25.0).abs() <= 1e-3
        );
    }

    #[test]
    fn group_repulsion_mixed_numeric_overloads_work() {
        let (scenario, _warnings) = run_script(
            r#"
            let g = create(sine, 1);
            flush();
            let g = g.repulsion(1, 35.0);
            flush();
            let g = g.repulsion(1.0, 35);
            flush();
        "#,
        );
        let mut updates = 0usize;
        for action in scenario
            .events
            .iter()
            .flat_map(|event| event.actions.iter())
        {
            if let Action::Update { update, .. } = action
                && let (Some(strength), Some(sigma)) =
                    (update.repulsion_strength, update.repulsion_sigma_cents)
            {
                assert!((strength - 1.0).abs() <= 1e-6);
                assert!((sigma - 35.0).abs() <= 1e-6);
                updates += 1;
            }
        }
        assert_eq!(updates, 2);
    }

    #[test]
    fn species_loo_and_anneal_reach_spawned_core() {
        let (scenario, _warnings) = run_script(
            r#"
            create(sine.loo(true).anneal_temp(0.12), 1);
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
    fn group_loo_and_anneal_live_update_reaches_individual() {
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
                Action::Spawn { .. } | Action::Update { .. } => {
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
                Action::Spawn { .. } | Action::Update { .. } => {
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
                Action::Spawn { .. } | Action::Update { .. } => {
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
                .timbre(0.7, 0.2);
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
        assert!((control.body.timbre.width - 0.2).abs() <= 1e-6);
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
                Action::Release { group_id, .. } => {
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
