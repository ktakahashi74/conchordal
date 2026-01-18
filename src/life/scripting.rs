use std::collections::BTreeMap;
use std::fs;
use std::sync::{Arc, Mutex};

use rand::random;
use rhai::{Array, Dynamic, Engine, EvalAltResult, FLOAT, FnPtr, INT, NativeCallContext, Position};
use tracing::warn;

use super::control::{AgentControl, BodyMethod, ControlUpdate, PhonationType, PitchMode};
use super::lifecycle::LifecycleConfig;
use super::scenario::{
    Action, ArticulationCoreConfig, EnvelopeConfig, Scenario, SceneMarker, SpawnSpec,
    SpawnStrategy, TimedEvent,
};

const DEFAULT_RELEASE_SEC: f32 = 0.05;
const DEFAULT_SEQ_DURATION_SEC: f32 = 1.0;

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
    brain: BrainKind,
    phonation: Option<PhonationKind>,
    metabolism_rate: Option<f32>,
    adsr: Option<AdsrSpec>,
}

impl SpeciesSpec {
    fn preset(body: BodyMethod) -> Self {
        let mut control = AgentControl::default();
        control.body.method = body;
        Self {
            control,
            brain: BrainKind::Entrain,
            phonation: None,
            metabolism_rate: None,
            adsr: None,
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
        self.control.body.amp = amp.clamp(0.0, 1.0);
    }

    fn set_freq(&mut self, freq: f32) {
        self.control.pitch.freq = freq.clamp(1.0, 20_000.0);
        self.control.pitch.mode = PitchMode::Lock;
    }

    fn set_timbre(&mut self, brightness: f32, width: f32) {
        self.control.body.timbre.brightness = brightness.clamp(0.0, 1.0);
        self.control.body.timbre.width = width.clamp(0.0, 1.0);
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

        engine.register_fn("harmonicity", |root_freq: FLOAT| {
            SpawnStrategy::Harmonicity {
                root_freq: root_freq as f32,
                min_mul: 1.0,
                max_mul: 4.0,
                min_dist_erb: 1.0,
            }
        });
        engine.register_fn(
            "range",
            |strategy: SpawnStrategy, min_mul: FLOAT, max_mul: FLOAT| match strategy {
                SpawnStrategy::Harmonicity {
                    root_freq,
                    min_dist_erb,
                    ..
                } => SpawnStrategy::Harmonicity {
                    root_freq,
                    min_mul: min_mul as f32,
                    max_mul: max_mul as f32,
                    min_dist_erb,
                },
                other => {
                    warn!("range() ignored for non-harmonicity strategy");
                    other
                }
            },
        );
        engine.register_fn(
            "min_dist",
            |strategy: SpawnStrategy, min_dist: FLOAT| match strategy {
                SpawnStrategy::Harmonicity {
                    root_freq,
                    min_mul,
                    max_mul,
                    ..
                } => SpawnStrategy::Harmonicity {
                    root_freq,
                    min_mul,
                    max_mul,
                    min_dist_erb: min_dist as f32,
                },
                other => {
                    warn!("min_dist() ignored for non-harmonicity strategy");
                    other
                }
            },
        );
        engine.register_fn("harmonic_density", |min_freq: FLOAT, max_freq: FLOAT| {
            SpawnStrategy::HarmonicDensity {
                min_freq: min_freq as f32,
                max_freq: max_freq as f32,
                min_dist_erb: 1.0,
            }
        });
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

        let ctx_for_set_harmonicity = ctx.clone();
        engine.register_fn(
            "set_harmonicity",
            move |_call_ctx: NativeCallContext, mirror: FLOAT| {
                let mut ctx = ctx_for_set_harmonicity.lock().expect("lock script context");
                let update = crate::core::landscape::LandscapeUpdate {
                    mirror: Some(mirror as f32),
                    ..crate::core::landscape::LandscapeUpdate::default()
                };
                let cursor = ctx.cursor;
                ctx.push_event(cursor, vec![Action::SetHarmonicity { update }]);
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

        let ctx_for_set_roughness_tolerance = ctx.clone();
        engine.register_fn(
            "set_roughness_tolerance",
            move |_call_ctx: NativeCallContext, value: FLOAT| {
                let mut ctx = ctx_for_set_roughness_tolerance
                    .lock()
                    .expect("lock script context");
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
            create(sine, 4).place(harmonicity(220.0)).freq(330.0);
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
            create(sine, 4).freq(330.0).place(harmonicity(220.0));
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
        assert!(matches!(strategy, Some(SpawnStrategy::Harmonicity { .. })));
    }
}
