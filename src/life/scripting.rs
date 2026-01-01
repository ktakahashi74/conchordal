use std::fs;
use std::sync::{Arc, Mutex};

use anyhow::{Context, anyhow};
use rhai::{
    Dynamic, Engine, EvalAltResult, EvalContext, Expression, FLOAT, Map, NativeCallContext,
    Position,
};
use serde::Serialize;

use super::api::script_api as api;
use super::api::{
    add_agent_at, set_agent_at, set_amp_agent, set_amp_cohort, set_amp_tag, set_cohort_at,
    set_commitment_agent, set_commitment_cohort, set_commitment_tag, set_drift_agent,
    set_drift_cohort, set_drift_tag, set_freq_agent, set_freq_cohort, set_freq_tag, set_tag_at,
    set_tag_str_at, spawn_agents_at, spawn_min_at, spawn_with_opts_at,
};
use super::scenario::{
    Action, AgentHandle, CohortHandle, IndividualConfig, LifeConfig, Scenario, SceneMarker,
    SpawnMethod, TagSelector, TargetRef, TimedEvent,
};

const SCRIPT_PRELUDE: &str = r#"
fn parallel(callback) {
    after(0.0, callback);
}

fn after(dt, callback) {
    push_time();
    wait(dt);
    callback.call();
    pop_time();
}

fn at(t, callback) {
    push_time();
    set_time(t);
    callback.call();
    pop_time();
}

fn repeat(n, callback) {
    let i = 0;
    while i < n {
        callback.call(i);
        i += 1;
    }
}

fn every(interval, count, callback) {
    let i = 0;
    while i < count {
        push_time();
        wait(i * interval);
        callback.call(i);
        pop_time();
        i += 1;
    }
}

fn spawn_every(tag, count, interval, opts) {
    let i = 0;
    while i < count {
        push_time();
        wait(i * interval);
        spawn(tag, 1, opts);
        pop_time();
        i += 1;
    }
}

fn spawn_every(tag, count, interval) {
    spawn_every(tag, count, interval, #{});
}
"#;

#[derive(Debug, Clone)]
pub struct ScriptContext {
    pub cursor: f32,
    pub time_stack: Vec<f32>,
    pub scenario: Scenario,
    pub next_group_id: u64,
    pub next_agent_id: u64,
    pub next_event_order: u64,
}

impl Default for ScriptContext {
    fn default() -> Self {
        Self {
            cursor: 0.0,
            time_stack: Vec::new(),
            scenario: Scenario {
                scene_markers: Vec::new(),
                events: Vec::new(),
                duration_sec: 0.0,
            },
            next_group_id: 1,
            next_agent_id: 1,
            next_event_order: 1,
        }
    }
}

impl ScriptContext {
    pub fn scene(&mut self, name: &str) {
        let order = self.next_event_order;
        self.next_event_order += 1;
        self.scenario.scene_markers.push(SceneMarker {
            name: name.to_string(),
            time: self.cursor,
            order,
        });
    }

    pub fn wait(&mut self, sec: f32) {
        self.cursor += sec;
    }

    pub fn set_time(&mut self, time_sec: f32) {
        self.cursor = time_sec;
    }

    pub fn push_time(&mut self) {
        self.time_stack.push(self.cursor);
    }

    pub fn pop_time(&mut self) {
        if let Some(t) = self.time_stack.pop() {
            self.cursor = t;
        }
    }

    fn push_event(&mut self, time_sec: f32, actions: Vec<Action>) {
        let order = self.next_event_order;
        self.next_event_order += 1;
        self.scenario.events.push(TimedEvent {
            time: time_sec,
            order,
            actions,
        });
    }

    pub fn spawn(
        &mut self,
        tag: &str,
        method_map: Map,
        life_map: Map,
        count: i64,
        amp: f32,
        position: Position,
    ) -> Result<CohortHandle, Box<EvalAltResult>> {
        let method = Self::from_map::<SpawnMethod>(method_map, "SpawnMethod", position)?;
        let life = Self::from_map_patch::<LifeConfig>(
            life_map,
            "LifeConfig",
            &LifeConfig::default(),
            position,
        )?;
        let c = count.max(0).min(u32::MAX as i64) as u32;
        let group_id = self.next_group_id;
        self.next_group_id += 1;
        let base_id = self.next_agent_id;
        self.next_agent_id += u64::from(c);
        let action = Action::SpawnAgents {
            group_id,
            base_id,
            method,
            count: c,
            amp,
            life,
            tag: Some(tag.to_string()),
        };
        self.push_event(self.cursor, vec![action]);
        Ok(CohortHandle {
            tag: tag.to_string(),
            group_id,
            base_id,
            count: c,
        })
    }

    pub fn spawn_default(
        &mut self,
        tag: &str,
        count: i64,
    ) -> Result<CohortHandle, Box<EvalAltResult>> {
        let method = SpawnMethod::default();
        let life = LifeConfig::default();
        let c = count.max(0).min(u32::MAX as i64) as u32;
        let amp = 0.18;
        let group_id = self.next_group_id;
        self.next_group_id += 1;
        let base_id = self.next_agent_id;
        self.next_agent_id += u64::from(c);
        let action = Action::SpawnAgents {
            group_id,
            base_id,
            method,
            count: c,
            amp,
            life,
            tag: Some(tag.to_string()),
        };
        self.push_event(self.cursor, vec![action]);
        Ok(CohortHandle {
            tag: tag.to_string(),
            group_id,
            base_id,
            count: c,
        })
    }

    pub fn spawn_opts(
        &mut self,
        tag: &str,
        count: i64,
        opts: Map,
        position: Position,
    ) -> Result<CohortHandle, Box<EvalAltResult>> {
        let mut unknown_keys = Vec::new();
        for key in opts.keys() {
            match key.as_str() {
                "amp" | "method" | "life" => {}
                _ => unknown_keys.push(key.to_string()),
            }
        }
        if !unknown_keys.is_empty() {
            unknown_keys.sort();
            let msg = format!(
                "spawn opts has unknown keys: [{}] (allowed: amp, method, life)",
                unknown_keys.join(", ")
            );
            return Err(Box::new(EvalAltResult::ErrorRuntime(msg.into(), position)));
        }

        let mut amp = 0.18;
        if let Some(value) = opts.get("amp") {
            if let Ok(f) = value.as_float() {
                amp = f as f32;
            } else if let Ok(i) = value.as_int() {
                amp = i as f32;
            } else {
                return Err(Box::new(EvalAltResult::ErrorRuntime(
                    "spawn opts amp must be a number".into(),
                    position,
                )));
            }
        }

        let method = if let Some(value) = opts.get("method") {
            let map = value.clone().try_cast().ok_or_else(|| {
                Box::new(EvalAltResult::ErrorRuntime(
                    "spawn opts method must be a map".into(),
                    position,
                ))
            })?;
            Self::from_map::<SpawnMethod>(map, "SpawnMethod", position)?
        } else {
            SpawnMethod::default()
        };

        let life = if let Some(value) = opts.get("life") {
            let map = value.clone().try_cast().ok_or_else(|| {
                Box::new(EvalAltResult::ErrorRuntime(
                    "spawn opts life must be a map".into(),
                    position,
                ))
            })?;
            Self::from_map_patch::<LifeConfig>(map, "LifeConfig", &LifeConfig::default(), position)?
        } else {
            LifeConfig::default()
        };
        let c = count.max(0).min(u32::MAX as i64) as u32;
        let group_id = self.next_group_id;
        self.next_group_id += 1;
        let base_id = self.next_agent_id;
        self.next_agent_id += u64::from(c);
        let action = Action::SpawnAgents {
            group_id,
            base_id,
            method,
            count: c,
            amp,
            life,
            tag: Some(tag.to_string()),
        };
        self.push_event(self.cursor, vec![action]);
        Ok(CohortHandle {
            tag: tag.to_string(),
            group_id,
            base_id,
            count: c,
        })
    }

    pub fn add_agent(
        &mut self,
        tag: &str,
        freq: f32,
        amp: f32,
        life_map: Map,
        position: Position,
    ) -> Result<AgentHandle, Box<EvalAltResult>> {
        let life = Self::from_map::<LifeConfig>(life_map, "LifeConfig", position)?;
        let agent = IndividualConfig {
            freq,
            amp,
            life,
            tag: Some(tag.to_string()),
        };
        let id = self.next_agent_id;
        self.next_agent_id += 1;
        let action = Action::AddAgent { id, agent };
        self.push_event(self.cursor, vec![action]);
        Ok(AgentHandle {
            id,
            tag: Some(tag.to_string()),
        })
    }

    pub fn set_freq(&mut self, target: TargetRef, freq: f32) {
        self.push_event(
            self.cursor,
            vec![Action::SetFreq {
                target,
                freq_hz: freq,
            }],
        );
    }

    pub fn set_amp(&mut self, target: TargetRef, amp: f32) {
        self.push_event(self.cursor, vec![Action::SetAmp { target, amp }]);
    }

    pub fn set_drift(&mut self, target: TargetRef, value: f32) {
        self.push_event(self.cursor, vec![Action::SetDrift { target, value }]);
    }

    pub fn set_commitment(&mut self, target: TargetRef, value: f32) {
        self.push_event(self.cursor, vec![Action::SetCommitment { target, value }]);
    }

    pub fn set_rhythm_vitality(&mut self, value: f32) {
        self.push_event(self.cursor, vec![Action::SetRhythmVitality { value }]);
    }

    pub fn set_global_coupling(&mut self, value: f32) {
        self.push_event(self.cursor, vec![Action::SetGlobalCoupling { value }]);
    }

    pub fn set_roughness_tolerance(&mut self, value: f32) {
        self.push_event(self.cursor, vec![Action::SetRoughnessTolerance { value }]);
    }

    pub fn set_harmonicity(&mut self, map: Map) -> Result<(), Box<EvalAltResult>> {
        let mirror = map
            .get("mirror")
            .and_then(|v| v.as_float().ok())
            .map(|v| v as f32);
        let limit = map
            .get("limit")
            .and_then(|v| v.as_int().ok())
            .map(|v| v.max(0) as u32);
        self.push_event(self.cursor, vec![Action::SetHarmonicity { mirror, limit }]);
        Ok(())
    }

    pub fn set_patch(
        &mut self,
        target: TargetRef,
        patch: Map,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        let mut unknown_keys = Vec::new();
        for key in patch.keys() {
            match key.as_str() {
                "amp" | "freq" | "drift" | "commitment" => {}
                _ => unknown_keys.push(key.to_string()),
            }
        }
        if !unknown_keys.is_empty() {
            unknown_keys.sort();
            let msg = format!(
                "set() patch has unknown keys: [{}] (allowed: amp, freq, drift, commitment)",
                unknown_keys.join(", ")
            );
            return Err(Box::new(EvalAltResult::ErrorRuntime(msg.into(), position)));
        }

        let mut actions = Vec::new();
        if let Some(value) = patch.get("amp") {
            let amp = value
                .as_float()
                .ok()
                .map(|v| v as f32)
                .or_else(|| value.as_int().ok().map(|v| v as f32))
                .ok_or_else(|| {
                    Box::new(EvalAltResult::ErrorRuntime(
                        "set() patch amp must be a number".into(),
                        position,
                    ))
                })?;
            actions.push(Action::SetAmp {
                target: target.clone(),
                amp,
            });
        }
        if let Some(value) = patch.get("freq") {
            let freq_hz = value
                .as_float()
                .ok()
                .map(|v| v as f32)
                .or_else(|| value.as_int().ok().map(|v| v as f32))
                .ok_or_else(|| {
                    Box::new(EvalAltResult::ErrorRuntime(
                        "set() patch freq must be a number".into(),
                        position,
                    ))
                })?;
            actions.push(Action::SetFreq {
                target: target.clone(),
                freq_hz,
            });
        }
        if let Some(value) = patch.get("drift") {
            let drift = value
                .as_float()
                .ok()
                .map(|v| v as f32)
                .or_else(|| value.as_int().ok().map(|v| v as f32))
                .ok_or_else(|| {
                    Box::new(EvalAltResult::ErrorRuntime(
                        "set() patch drift must be a number".into(),
                        position,
                    ))
                })?;
            actions.push(Action::SetDrift {
                target: target.clone(),
                value: drift,
            });
        }
        if let Some(value) = patch.get("commitment") {
            let commitment = value
                .as_float()
                .ok()
                .map(|v| v as f32)
                .or_else(|| value.as_int().ok().map(|v| v as f32))
                .ok_or_else(|| {
                    Box::new(EvalAltResult::ErrorRuntime(
                        "set() patch commitment must be a number".into(),
                        position,
                    ))
                })?;
            actions.push(Action::SetCommitment {
                target: target.clone(),
                value: commitment,
            });
        }
        if actions.is_empty() {
            return Err(Box::new(EvalAltResult::ErrorRuntime(
                "set() patch has no recognized keys".into(),
                position,
            )));
        }
        self.push_event(self.cursor, actions);
        Ok(())
    }

    pub fn remove(&mut self, target: TargetRef) {
        self.push_event(self.cursor, vec![Action::RemoveAgent { target }]);
    }

    pub fn release(&mut self, target: TargetRef, sec: f32) {
        self.push_event(
            self.cursor,
            vec![Action::ReleaseAgent {
                target,
                release_sec: sec,
            }],
        );
    }

    pub fn finish(&mut self) {
        self.push_event(self.cursor, vec![Action::Finish]);
        self.scenario.duration_sec = self.scenario.duration_sec.max(self.cursor);
    }

    pub fn run(&mut self, sec: f32) {
        let end = (self.cursor + sec.max(0.0)).max(self.cursor);
        self.scenario.duration_sec = self.scenario.duration_sec.max(end);
        self.push_event(end, vec![Action::Finish]);
    }

    fn from_map<T: serde::de::DeserializeOwned>(
        map: Map,
        name: &str,
        position: Position,
    ) -> Result<T, Box<EvalAltResult>> {
        serde_json::to_value(&map)
            .map_err(|e| {
                Box::new(EvalAltResult::ErrorRuntime(
                    format!("Error serializing {name}: {e}").into(),
                    position,
                ))
            })
            .and_then(|v| {
                serde_json::from_value::<T>(v).map_err(|e| {
                    let debug_map = format!("{:?}", map);
                    Box::new(EvalAltResult::ErrorRuntime(
                        format!("Error parsing {name}: {e} (input: {debug_map})").into(),
                        position,
                    ))
                })
            })
    }

    fn from_map_patch<T: serde::de::DeserializeOwned + Serialize>(
        patch: Map,
        name: &str,
        base: &T,
        position: Position,
    ) -> Result<T, Box<EvalAltResult>> {
        let base_val = serde_json::to_value(base).map_err(|e| {
            Box::new(EvalAltResult::ErrorRuntime(
                format!("Error serializing base {name}: {e}").into(),
                position,
            ))
        })?;
        let patch_val = serde_json::to_value(&patch).map_err(|e| {
            Box::new(EvalAltResult::ErrorRuntime(
                format!("Error serializing patch {name}: {e}").into(),
                position,
            ))
        })?;
        let merged = merge_json(base_val, patch_val);
        serde_json::from_value::<T>(merged).map_err(|e| {
            let debug_map = format!("{:?}", patch);
            Box::new(EvalAltResult::ErrorRuntime(
                format!("Error parsing {name}: {e} (input: {debug_map})").into(),
                position,
            ))
        })
    }
}

fn merge_json(base: serde_json::Value, patch: serde_json::Value) -> serde_json::Value {
    match (base, patch) {
        (serde_json::Value::Object(mut base_map), serde_json::Value::Object(patch_map)) => {
            if patch_map.contains_key("core") || patch_map.contains_key("type") {
                return serde_json::Value::Object(patch_map);
            }
            for (k, v) in patch_map {
                let base_val = base_map.remove(&k).unwrap_or(serde_json::Value::Null);
                base_map.insert(k, merge_json(base_val, v));
            }
            serde_json::Value::Object(base_map)
        }
        (_, patch_val) => patch_val,
    }
}

pub struct ScriptHost;

impl ScriptHost {
    fn create_engine(ctx: Arc<Mutex<ScriptContext>>) -> Engine {
        let mut engine = Engine::new();
        engine.on_print(|msg| println!("[rhai] {msg}"));
        engine.register_type_with_name::<CohortHandle>("CohortHandle");
        engine.register_type_with_name::<AgentHandle>("AgentHandle");
        engine.register_type_with_name::<TagSelector>("TagSelector");
        engine.register_iterator::<CohortHandle>();
        engine.register_fn("len", |cohort: &mut CohortHandle| -> i64 {
            cohort.len() as i64
        });
        engine.register_fn("is_empty", |cohort: &mut CohortHandle| -> bool {
            cohort.is_empty()
        });
        engine.register_indexer_get(
            |cohort: &mut CohortHandle, index: i64| -> Result<AgentHandle, Box<EvalAltResult>> {
                if index < 0 {
                    let msg = format!(
                        "GroupHandle index out of range: tag={} index={} count={}",
                        cohort.tag, index, cohort.count
                    );
                    return Err(Box::new(EvalAltResult::ErrorRuntime(
                        msg.into(),
                        Position::NONE,
                    )));
                }
                let idx = index as u32;
                if idx >= cohort.count {
                    let msg = format!(
                        "GroupHandle index out of range: tag={} index={} count={}",
                        cohort.tag, index, cohort.count
                    );
                    return Err(Box::new(EvalAltResult::ErrorRuntime(
                        msg.into(),
                        Position::NONE,
                    )));
                }
                Ok(AgentHandle {
                    id: cohort.base_id + u64::from(idx),
                    tag: Some(cohort.tag.clone()),
                })
            },
        );

        let ctx_for_spawn_syntax = ctx.clone();
        engine.register_custom_syntax_with_state_raw(
            "spawn",
            |symbols, look_ahead, _state| {
                let next = match symbols.len() {
                    0 => Some("spawn"),
                    1 => Some("("),
                    2 => Some("$expr$"),
                    3 => Some(","),
                    4 => Some("$expr$"),
                    5 => match look_ahead {
                        "," => Some(","),
                        ")" => Some(")"),
                        _ => {
                            return Err(rhai::LexError::ImproperSymbol(
                                look_ahead.to_string(),
                                "Expected ',' or ')' for spawn".to_string(),
                            )
                            .into_err(Position::NONE));
                        }
                    },
                    6 => {
                        let last = symbols.last().map(|s| s.as_str()).unwrap_or("");
                        if last == ")" { None } else { Some("$expr$") }
                    }
                    7 => Some(")"),
                    _ => None,
                };
                Ok(next.map(Into::into))
            },
            false,
            move |eval_ctx: &mut EvalContext, exprs: &[Expression], _state| {
                let pos_tag = exprs[0].position();
                let pos_count = exprs[1].position();
                let pos_opts = exprs.get(2).map(|e| e.position()).unwrap_or(pos_tag);
                let tag_dyn = eval_ctx.eval_expression_tree(&exprs[0])?;
                let count_dyn = eval_ctx.eval_expression_tree(&exprs[1])?;
                let tag = tag_dyn.try_cast::<String>().ok_or_else(|| {
                    Box::new(EvalAltResult::ErrorRuntime(
                        "spawn tag must be a string".into(),
                        pos_tag,
                    ))
                })?;
                let count = count_dyn.try_cast::<i64>().ok_or_else(|| {
                    Box::new(EvalAltResult::ErrorRuntime(
                        "spawn count must be an integer".into(),
                        pos_count,
                    ))
                })?;
                if count < 0 {
                    return Err(Box::new(EvalAltResult::ErrorRuntime(
                        "spawn count must be non-negative".into(),
                        pos_count,
                    )));
                }
                let mut ctx = ctx_for_spawn_syntax.lock().expect("lock script context");
                let handle = if exprs.len() >= 3 {
                    let opts_dyn = eval_ctx.eval_expression_tree(&exprs[2])?;
                    let opts = opts_dyn.try_cast::<Map>().ok_or_else(|| {
                        Box::new(EvalAltResult::ErrorRuntime(
                            "spawn opts must be a map".into(),
                            pos_opts,
                        ))
                    })?;
                    spawn_with_opts_at(&mut ctx, &tag, count, opts, pos_opts)?
                } else {
                    spawn_min_at(&mut ctx, &tag, count, pos_tag)?
                };
                Ok(Dynamic::from(handle))
            },
        );

        let ctx_for_scene = ctx.clone();
        engine.register_fn("scene", move |name: &str| {
            let mut ctx = ctx_for_scene.lock().expect("lock script context");
            api::scene(&mut ctx, name);
        });

        let ctx_for_wait = ctx.clone();
        engine.register_fn("wait", move |sec: FLOAT| {
            let mut ctx = ctx_for_wait.lock().expect("lock script context");
            api::wait(&mut ctx, sec);
        });
        let ctx_for_wait_int = ctx.clone();
        engine.register_fn("wait", move |sec: i64| {
            let mut ctx = ctx_for_wait_int.lock().expect("lock script context");
            api::wait_int(&mut ctx, sec);
        });

        let ctx_for_push_time = ctx.clone();
        engine.register_fn("push_time", move || {
            let mut ctx = ctx_for_push_time.lock().expect("lock script context");
            api::push_time(&mut ctx);
        });

        let ctx_for_pop_time = ctx.clone();
        engine.register_fn("pop_time", move || {
            let mut ctx = ctx_for_pop_time.lock().expect("lock script context");
            api::pop_time(&mut ctx);
        });
        let ctx_for_set_time = ctx.clone();
        engine.register_fn("set_time", move |sec: FLOAT| {
            let mut ctx = ctx_for_set_time.lock().expect("lock script context");
            api::set_time(&mut ctx, sec);
        });

        let ctx_for_spawn_agents = ctx.clone();
        engine.register_fn(
            "spawn_agents",
            move |call_ctx: NativeCallContext,
                  tag: &str,
                  method_map: Map,
                  life_map: Map,
                  count: i64,
                  amp: FLOAT|
                  -> Result<CohortHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_spawn_agents.lock().expect("lock script context");
                spawn_agents_at(
                    &mut ctx,
                    tag,
                    method_map,
                    life_map,
                    count,
                    amp,
                    call_ctx.call_position(),
                )
            },
        );
        let ctx_for_spawn_default = ctx.clone();
        engine.register_fn(
            "spawn_default",
            move |tag: &str, count: i64| -> Result<CohortHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_spawn_default.lock().expect("lock script context");
                ctx.spawn_default(tag, count)
            },
        );

        engine.register_fn("random", api::random_method);
        engine.register_fn("random", api::random_method_opts);
        engine.register_fn("harmonicity", api::harmonicity);
        engine.register_fn("harmonicity", api::harmonicity_opts);
        engine.register_fn("spectral_gap", api::spectral_gap);
        engine.register_fn("spectral_gap", api::spectral_gap_opts);
        engine.register_fn("harmonic_density", api::harmonic_density);
        engine.register_fn("harmonic_density", api::harmonic_density_opts);
        engine.register_fn("life", api::life);
        let ctx_for_add_agent = ctx.clone();
        engine.register_fn(
            "add_agent",
            move |call_ctx: NativeCallContext,
                  tag: &str,
                  freq: FLOAT,
                  amp: FLOAT,
                  life_map: Map|
                  -> Result<AgentHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_add_agent.lock().expect("lock script context");
                add_agent_at(&mut ctx, tag, freq, amp, life_map, call_ctx.call_position())
            },
        );

        let ctx_for_tag = ctx.clone();
        engine.register_fn("tag", move |name: &str| {
            let mut ctx = ctx_for_tag.lock().expect("lock script context");
            api::tag(&mut ctx, name)
        });

        let ctx_for_set_freq_agent = ctx.clone();
        engine.register_fn("set_freq", move |target: AgentHandle, freq: FLOAT| {
            let mut ctx = ctx_for_set_freq_agent.lock().expect("lock script context");
            set_freq_agent(&mut ctx, target, freq);
        });
        let ctx_for_set_freq_cohort = ctx.clone();
        engine.register_fn("set_freq", move |target: CohortHandle, freq: FLOAT| {
            let mut ctx = ctx_for_set_freq_cohort.lock().expect("lock script context");
            set_freq_cohort(&mut ctx, target, freq);
        });
        let ctx_for_set_freq_tag = ctx.clone();
        engine.register_fn("set_freq", move |target: TagSelector, freq: FLOAT| {
            let mut ctx = ctx_for_set_freq_tag.lock().expect("lock script context");
            set_freq_tag(&mut ctx, target, freq);
        });
        let ctx_for_set_tag_str = ctx.clone();
        engine.register_fn(
            "set",
            move |call_ctx: NativeCallContext, tag: &str, patch: Map| {
                let mut ctx = ctx_for_set_tag_str.lock().expect("lock script context");
                set_tag_str_at(&mut ctx, tag, patch, call_ctx.call_position())
            },
        );
        let ctx_for_set_agent = ctx.clone();
        engine.register_fn(
            "set",
            move |call_ctx: NativeCallContext, target: AgentHandle, patch: Map| {
                let mut ctx = ctx_for_set_agent.lock().expect("lock script context");
                set_agent_at(&mut ctx, target, patch, call_ctx.call_position())
            },
        );
        let ctx_for_set_cohort = ctx.clone();
        engine.register_fn(
            "set",
            move |call_ctx: NativeCallContext, target: CohortHandle, patch: Map| {
                let mut ctx = ctx_for_set_cohort.lock().expect("lock script context");
                set_cohort_at(&mut ctx, target, patch, call_ctx.call_position())
            },
        );
        let ctx_for_set_tag = ctx.clone();
        engine.register_fn(
            "set",
            move |call_ctx: NativeCallContext, target: TagSelector, patch: Map| {
                let mut ctx = ctx_for_set_tag.lock().expect("lock script context");
                set_tag_at(&mut ctx, target, patch, call_ctx.call_position())
            },
        );

        let ctx_for_set_amp_agent = ctx.clone();
        engine.register_fn("set_amp", move |target: AgentHandle, amp: FLOAT| {
            let mut ctx = ctx_for_set_amp_agent.lock().expect("lock script context");
            set_amp_agent(&mut ctx, target, amp);
        });
        let ctx_for_set_amp_cohort = ctx.clone();
        engine.register_fn("set_amp", move |target: CohortHandle, amp: FLOAT| {
            let mut ctx = ctx_for_set_amp_cohort.lock().expect("lock script context");
            set_amp_cohort(&mut ctx, target, amp);
        });
        let ctx_for_set_amp_tag = ctx.clone();
        engine.register_fn("set_amp", move |target: TagSelector, amp: FLOAT| {
            let mut ctx = ctx_for_set_amp_tag.lock().expect("lock script context");
            set_amp_tag(&mut ctx, target, amp);
        });

        let ctx_for_set_drift_agent = ctx.clone();
        engine.register_fn("set_drift", move |target: AgentHandle, value: FLOAT| {
            let mut ctx = ctx_for_set_drift_agent.lock().expect("lock script context");
            set_drift_agent(&mut ctx, target, value);
        });
        let ctx_for_set_drift_cohort = ctx.clone();
        engine.register_fn("set_drift", move |target: CohortHandle, value: FLOAT| {
            let mut ctx = ctx_for_set_drift_cohort
                .lock()
                .expect("lock script context");
            set_drift_cohort(&mut ctx, target, value);
        });
        let ctx_for_set_drift_tag = ctx.clone();
        engine.register_fn("set_drift", move |target: TagSelector, value: FLOAT| {
            let mut ctx = ctx_for_set_drift_tag.lock().expect("lock script context");
            set_drift_tag(&mut ctx, target, value);
        });

        let ctx_for_set_commitment_agent = ctx.clone();
        engine.register_fn(
            "set_commitment",
            move |target: AgentHandle, value: FLOAT| {
                let mut ctx = ctx_for_set_commitment_agent
                    .lock()
                    .expect("lock script context");
                set_commitment_agent(&mut ctx, target, value);
            },
        );
        let ctx_for_set_commitment_cohort = ctx.clone();
        engine.register_fn(
            "set_commitment",
            move |target: CohortHandle, value: FLOAT| {
                let mut ctx = ctx_for_set_commitment_cohort
                    .lock()
                    .expect("lock script context");
                set_commitment_cohort(&mut ctx, target, value);
            },
        );
        let ctx_for_set_commitment_tag = ctx.clone();
        engine.register_fn(
            "set_commitment",
            move |target: TagSelector, value: FLOAT| {
                let mut ctx = ctx_for_set_commitment_tag
                    .lock()
                    .expect("lock script context");
                set_commitment_tag(&mut ctx, target, value);
            },
        );

        let ctx_for_set_vitality = ctx.clone();
        engine.register_fn("set_rhythm_vitality", move |value: FLOAT| {
            let mut ctx = ctx_for_set_vitality.lock().expect("lock script context");
            api::set_rhythm_vitality(&mut ctx, value);
        });

        let ctx_for_set_coupling = ctx.clone();
        engine.register_fn("set_global_coupling", move |value: FLOAT| {
            let mut ctx = ctx_for_set_coupling.lock().expect("lock script context");
            api::set_global_coupling(&mut ctx, value);
        });

        let ctx_for_set_roughness = ctx.clone();
        engine.register_fn("set_roughness_tolerance", move |value: FLOAT| {
            let mut ctx = ctx_for_set_roughness.lock().expect("lock script context");
            api::set_roughness_tolerance(&mut ctx, value);
        });

        let ctx_for_set_harmonicity = ctx.clone();
        engine.register_fn(
            "set_harmonicity",
            move |map: Map| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_set_harmonicity.lock().expect("lock script context");
                api::set_harmonicity(&mut ctx, map)
            },
        );

        let ctx_for_remove_agent = ctx.clone();
        engine.register_fn("remove", move |target: AgentHandle| {
            let mut ctx = ctx_for_remove_agent.lock().expect("lock script context");
            api::remove_agent(&mut ctx, target);
        });
        let ctx_for_remove_cohort = ctx.clone();
        engine.register_fn("remove", move |target: CohortHandle| {
            let mut ctx = ctx_for_remove_cohort.lock().expect("lock script context");
            api::remove_cohort(&mut ctx, target);
        });
        let ctx_for_remove_tag = ctx.clone();
        engine.register_fn("remove", move |target: TagSelector| {
            let mut ctx = ctx_for_remove_tag.lock().expect("lock script context");
            api::remove_tag(&mut ctx, target);
        });
        let ctx_for_remove_tag_str = ctx.clone();
        engine.register_fn("remove", move |tag: &str| {
            let mut ctx = ctx_for_remove_tag_str.lock().expect("lock script context");
            api::remove_tag_str(&mut ctx, tag);
        });

        let ctx_for_release_agent = ctx.clone();
        engine.register_fn("release", move |target: AgentHandle, sec: FLOAT| {
            let mut ctx = ctx_for_release_agent.lock().expect("lock script context");
            api::release_agent(&mut ctx, target, sec);
        });
        let ctx_for_release_cohort = ctx.clone();
        engine.register_fn("release", move |target: CohortHandle, sec: FLOAT| {
            let mut ctx = ctx_for_release_cohort.lock().expect("lock script context");
            api::release_cohort(&mut ctx, target, sec);
        });
        let ctx_for_release_tag = ctx.clone();
        engine.register_fn("release", move |target: TagSelector, sec: FLOAT| {
            let mut ctx = ctx_for_release_tag.lock().expect("lock script context");
            api::release_tag(&mut ctx, target, sec);
        });
        let ctx_for_release_tag_str = ctx.clone();
        engine.register_fn("release", move |tag: &str, sec: FLOAT| {
            let mut ctx = ctx_for_release_tag_str.lock().expect("lock script context");
            api::release_tag_str(&mut ctx, tag, sec);
        });

        let ctx_for_finish = ctx.clone();
        engine.register_fn("finish", move || {
            let mut ctx = ctx_for_finish.lock().expect("lock script context");
            api::finish(&mut ctx);
        });

        let ctx_for_run = ctx.clone();
        engine.register_fn("run", move |sec: FLOAT| {
            let mut ctx = ctx_for_run.lock().expect("lock script context");
            api::run_float(&mut ctx, sec);
        });
        let ctx_for_run_int = ctx.clone();
        engine.register_fn("run", move |sec: i64| {
            let mut ctx = ctx_for_run_int.lock().expect("lock script context");
            api::run_int(&mut ctx, sec);
        });

        engine
    }

    pub fn load_script(path: &str) -> anyhow::Result<Scenario> {
        let src = fs::read_to_string(path).map_err(|err| anyhow!("read script {path}: {err}"))?;
        let ctx = Arc::new(Mutex::new(ScriptContext::default()));
        let engine = ScriptHost::create_engine(ctx.clone());
        let script_src = format!("{SCRIPT_PRELUDE}\n{src}");

        if let Err(e) = engine.eval::<()>(&script_src) {
            // Print structured error to help diagnose script issues (e.g., type mismatches).
            println!("Debug script error: {:?}", e);
            return Err(anyhow!(e.to_string())).with_context(|| format!("execute script {path}"));
        }

        let ctx_out = ctx.lock().expect("lock script context");
        Ok(ctx_out.scenario.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhai::Position;

    fn run_script(src: &str) -> Scenario {
        let ctx = Arc::new(Mutex::new(ScriptContext::default()));
        let engine = ScriptHost::create_engine(ctx.clone());
        let script_src = combined_script(src);
        engine.eval::<()>(&script_src).expect("script runs");
        ctx.lock().expect("lock ctx").scenario.clone()
    }

    fn eval_script_error_position(src: &str) -> Position {
        let ctx = Arc::new(Mutex::new(ScriptContext::default()));
        let engine = ScriptHost::create_engine(ctx.clone());
        let script_src = combined_script(src);
        let err = engine
            .eval::<()>(&script_src)
            .expect_err("script should error");
        err.position()
    }

    fn combined_script(src: &str) -> String {
        format!("{SCRIPT_PRELUDE}\n\n{src}")
    }

    fn expected_line(full: &str, needle: &str) -> usize {
        full.lines()
            .enumerate()
            .find_map(|(idx, line)| {
                if line.contains(needle) {
                    Some(idx + 1)
                } else {
                    None
                }
            })
            .unwrap_or_else(|| panic!("expected marker not found: {}", needle))
    }

    fn assert_time_close(actual: f32, expected: f32) {
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-5,
            "time mismatch: expected {expected}, got {actual}"
        );
    }

    #[test]
    fn scene_sets_start_and_relative_times() {
        let scenario = run_script(
            r#"
            scene("intro");
            let life = #{
                body: #{ core: "sine" },
                articulation: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.5 },
                pitch: #{ core: "pitch_hill_climb" },
                perceptual: #{ tau_fast: 0.5, tau_slow: 6.0, w_boredom: 0.8, w_familiarity: 0.2 }
            };
            add_agent("lead", 440.0, 0.2, life);
            wait(1.0);
            scene("break");
            let hit_life = #{
                body: #{ core: "sine" },
                articulation: #{ core: "entrain", type: "decay", initial_energy: 0.8, half_life_sec: 0.2 },
                pitch: #{ core: "pitch_hill_climb" },
                perceptual: #{ tau_fast: 0.5, tau_slow: 6.0, w_boredom: 0.8, w_familiarity: 0.2 }
            };
            add_agent("hit", 880.0, 0.1, hit_life);
            wait(0.5);
            finish();
        "#,
        );

        assert_eq!(scenario.scene_markers.len(), 2);
        assert_eq!(scenario.scene_markers[0].name, "intro");
        assert_time_close(scenario.scene_markers[0].time, 0.0);
        assert_eq!(scenario.scene_markers[1].name, "break");
        assert_time_close(scenario.scene_markers[1].time, 1.0);

        let mut has_hit = false;
        let mut has_finish = false;
        for ev in &scenario.events {
            for action in &ev.actions {
                match action {
                    Action::AddAgent { agent, .. } => {
                        if agent.tag.as_deref() == Some("hit") {
                            assert_time_close(ev.time, 1.0);
                            has_hit = true;
                        }
                    }
                    Action::Finish => {
                        assert_time_close(ev.time, 1.5);
                        has_finish = true;
                    }
                    _ => {}
                }
            }
        }
        assert!(has_hit, "expected add_agent event for hit");
        assert!(has_finish, "expected finish event");
    }

    #[test]
    fn parallel_restores_time_after_block() {
        let scenario = run_script(
            r#"
            scene("intro");
            wait(0.1);
            parallel(|| {
                wait(0.5);
                let life = #{
                    body: #{ core: "sine" },
                    articulation: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.3 },
                    pitch: #{ core: "pitch_hill_climb" },
                    perceptual: #{ tau_fast: 0.5, tau_slow: 6.0, w_boredom: 0.8, w_familiarity: 0.2 }
                };
                add_agent("pad", 200.0, 0.1, life);
            });
            wait(0.2);
            let after_life = #{
                body: #{ core: "sine" },
                articulation: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.3 },
                pitch: #{ core: "pitch_hill_climb" },
                perceptual: #{ tau_fast: 0.5, tau_slow: 6.0, w_boredom: 0.8, w_familiarity: 0.2 }
            };
            add_agent("after", 300.0, 0.1, after_life);
            finish();
        "#,
        );

        let mut pad_time = None;
        let mut after_time = None;
        let mut finish_time = None;
        for ev in &scenario.events {
            for action in &ev.actions {
                match action {
                    Action::AddAgent { agent, .. } => match agent.tag.as_deref() {
                        Some("pad") => pad_time = Some(ev.time),
                        Some("after") => after_time = Some(ev.time),
                        _ => {}
                    },
                    Action::Finish => finish_time = Some(ev.time),
                    _ => {}
                }
            }
        }

        assert_time_close(pad_time.expect("pad time"), 0.6);
        assert_time_close(after_time.expect("after time"), 0.3);
        assert_time_close(finish_time.expect("finish time"), 0.3);
    }

    #[test]
    fn spawn_uses_relative_scene_time() {
        let scenario = run_script(
            r#"
            scene("alpha");
            wait(1.2);
            let method = #{ mode: "random_log_uniform", min_freq: 100.0, max_freq: 200.0 };
            let life = #{
                body: #{ core: "sine" },
                articulation: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.5 },
                pitch: #{ core: "pitch_hill_climb" },
                perceptual: #{ tau_fast: 0.5, tau_slow: 6.0, w_boredom: 0.8, w_familiarity: 0.2 }
            };
            spawn_agents("tag", method, life, 3, 0.25);
            finish();
        "#,
        );

        assert_eq!(scenario.scene_markers.len(), 1);
        assert_eq!(scenario.scene_markers[0].name, "alpha");
        assert_time_close(scenario.scene_markers[0].time, 0.0);

        let ev = scenario
            .events
            .iter()
            .find(|ev| {
                ev.actions
                    .iter()
                    .any(|a| matches!(a, Action::SpawnAgents { .. }))
            })
            .expect("spawn event");
        assert_time_close(ev.time, 1.2);
        assert_eq!(ev.actions.len(), 1);
        match &ev.actions[0] {
            Action::SpawnAgents {
                group_id: _,
                base_id: _,
                count,
                amp,
                life,
                tag,
                method:
                    SpawnMethod::RandomLogUniform {
                        min_freq,
                        max_freq,
                        min_dist_erb: None,
                    },
            } => {
                assert_eq!(*count, 3);
                assert_time_close(*amp, 0.25);
                assert_eq!(*min_freq, 100.0);
                assert_eq!(*max_freq, 200.0);
                assert!(matches!(
                    life.articulation,
                    crate::life::scenario::ArticulationCoreConfig::Entrain { .. }
                ));
                assert_eq!(tag.as_deref(), Some("tag"));
            }
            other => panic!("unexpected action: {:?}", other),
        }
    }

    #[test]
    fn scene_created_when_scene_absent() {
        let scenario = run_script(
            r#"
            let life = #{
                body: #{ core: "sine" },
                articulation: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.5 },
                pitch: #{ core: "pitch_hill_climb" },
                perceptual: #{ tau_fast: 0.5, tau_slow: 6.0, w_boredom: 0.8, w_familiarity: 0.2 }
            };
            add_agent("init", 330.0, 0.2, life);
            wait(0.3);
            finish();
        "#,
        );

        assert!(scenario.scene_markers.is_empty());
        assert_eq!(scenario.events.len(), 2);
        assert_time_close(scenario.events[0].time, 0.0);
        assert_time_close(scenario.events[1].time, 0.3);
    }

    #[test]
    fn sample_script_file_executes() {
        let scenario = ScriptHost::load_script("samples/01_fundamentals/spawn_basics.rhai")
            .expect("sample script should run");
        assert!(!scenario.events.is_empty());
    }

    #[test]
    fn minimal_spawn_run_executes() {
        let scenario = ScriptHost::load_script("tests/scripts/minimal_spawn_run.rhai")
            .expect("minimal script should run");
        let mut has_spawn = false;
        for ev in &scenario.events {
            for action in &ev.actions {
                if let Action::SpawnAgents {
                    group_id,
                    base_id,
                    count,
                    ..
                } = action
                {
                    assert!(*group_id > 0);
                    assert!(*base_id > 0);
                    assert!(*count > 0);
                    has_spawn = true;
                }
            }
        }
        let has_finish = scenario
            .events
            .iter()
            .any(|ev| ev.actions.iter().any(|a| matches!(a, Action::Finish)));
        assert!(has_spawn, "expected spawn event");
        assert!(has_finish, "expected finish event");
    }

    #[test]
    fn spawn_opts_unknown_key_has_position() {
        let pos = eval_script_error_position(
            r#"
            spawn("d", 1, #{ amm: 0.1 });
        "#,
        );
        assert_ne!(pos, Position::NONE);
    }

    #[test]
    fn spawn_opts_bad_type_has_position() {
        let pos = eval_script_error_position(
            r#"
            spawn("d", 1, #{ amp: "0.1" });
        "#,
        );
        assert_ne!(pos, Position::NONE);
    }

    #[test]
    fn set_patch_unknown_key_has_position() {
        let pos = eval_script_error_position(
            r#"
            spawn("d", 1);
            set("d", #{ amp: 0.1, ampp: 0.2 });
        "#,
        );
        assert_ne!(pos, Position::NONE);
    }

    #[test]
    fn set_patch_bad_type_has_position() {
        let pos = eval_script_error_position(
            r#"
            spawn("d", 1);
            set("d", #{ amp: "x" });
        "#,
        );
        assert_ne!(pos, Position::NONE);
    }

    #[test]
    fn spawn_count_error_points_to_count_arg() {
        let src = r#"
            spawn(
                "d",
                "BAD_COUNT_123",
                #{}
            );
        "#;
        let full = combined_script(src);
        let pos = eval_script_error_position(src);
        let expected_line = expected_line(&full, "BAD_COUNT_123");
        assert_eq!(pos.line().unwrap(), expected_line);
    }

    #[test]
    fn spawn_opts_error_points_to_opts_arg() {
        let src = r#"
            spawn(
                "d",
                1,
                "BAD_OPTS_123"
            );
        "#;
        let full = combined_script(src);
        let pos = eval_script_error_position(src);
        let expected_line = expected_line(&full, "BAD_OPTS_123");
        assert_eq!(pos.line().unwrap(), expected_line);
    }

    #[test]
    fn spawn_opts_unknown_key_points_to_opts_arg() {
        let src = r#"
            spawn(
                "d",
                1,
                #{ amm_BADKEY_123: 0.1 }
            );
        "#;
        let full = combined_script(src);
        let pos = eval_script_error_position(src);
        let expected_line = expected_line(&full, "amm_BADKEY_123");
        assert_eq!(pos.line().unwrap(), expected_line);
    }

    #[test]
    fn spawn_count_negative_points_to_count_arg() {
        let src = r#"
            spawn(
                "d",
                -123
            );
        "#;
        let full = combined_script(src);
        let pos = eval_script_error_position(src);
        let expected_line = expected_line(&full, "-123");
        assert_eq!(pos.line().unwrap(), expected_line);
    }

    #[test]
    fn empty_life_map_executes() {
        let scenario = ScriptHost::load_script("tests/scripts/empty_life_map_ok.rhai")
            .expect("empty life map should run");
        let mut has_add = false;
        for ev in &scenario.events {
            for action in &ev.actions {
                if let Action::AddAgent { id, .. } = action {
                    assert!(*id > 0);
                    has_add = true;
                }
            }
        }
        let has_finish = scenario
            .events
            .iter()
            .any(|ev| ev.actions.iter().any(|a| matches!(a, Action::Finish)));
        assert!(has_add, "expected add_agent event");
        assert!(has_finish, "expected finish event");
    }

    #[test]
    fn handle_index_and_iter_executes() {
        let scenario = ScriptHost::load_script("tests/scripts/handle_index_and_iter.rhai")
            .expect("handle iteration script should run");
        assert!(!scenario.events.is_empty());
    }

    #[test]
    fn tag_selector_ops_executes() {
        let scenario = ScriptHost::load_script("tests/scripts/tag_selector_ops.rhai")
            .expect("tag selector script should run");
        assert!(!scenario.events.is_empty());
    }

    #[test]
    fn stable_order_same_time_is_monotonic() {
        let scenario = ScriptHost::load_script("tests/scripts/stable_order_same_time.rhai")
            .expect("stable order script should run");
        let mut last_order = 0;
        let mut freqs = Vec::new();
        for ev in &scenario.events {
            assert!(ev.order > last_order);
            last_order = ev.order;
            for action in &ev.actions {
                if let Action::SetFreq { freq_hz, .. } = action {
                    freqs.push(*freq_hz);
                }
            }
        }
        assert_eq!(freqs, vec![200.0, 300.0]);
    }

    #[test]
    fn all_sample_scripts_parse() {
        let mut stack = vec![std::path::PathBuf::from("samples")];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).expect("samples dir exists") {
                let path = entry.expect("dir entry").path();
                let name = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default()
                    .to_string();
                if name.starts_with('#') || name.starts_with('.') || name.ends_with('~') {
                    continue;
                }
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().and_then(|s| s.to_str()) != Some("rhai") {
                    continue;
                }
                ScriptHost::load_script(path.to_str().expect("path str"))
                    .unwrap_or_else(|e| panic!("script {name} should parse: {e}"));
            }
        }
    }
}
