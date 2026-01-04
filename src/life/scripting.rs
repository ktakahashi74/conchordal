use std::fs;
use std::sync::{Arc, Mutex};

use rhai::{
    Dynamic, Engine, EvalAltResult, EvalContext, Expression, FLOAT, INT, Map, NativeCallContext,
    Position,
};
use serde::Serialize;

use super::api::{
    pop_time, push_time, set_agent_at, set_cohort_at, set_tag_str_at, set_time, spawn_min_at,
    spawn_with_opts_at,
};
use super::scenario::{
    Action, AgentHandle, CohortHandle, IndividualConfig, LifeConfig, Scenario, SceneMarker,
    SpawnMethod, TargetRef, TimedEvent,
};

const SCRIPT_PRELUDE: &str = r#"
fn parallel(callback) {
    after(0.0, callback);
}

fn after(dt, callback) {
    __internal_push_time();
    wait(dt);
    callback.call();
    __internal_pop_time();
}

fn at(t, callback) {
    __internal_push_time();
    __internal_set_time(t);
    callback.call();
    __internal_pop_time();
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
        __internal_push_time();
        wait(i * interval);
        callback.call(i);
        __internal_pop_time();
        i += 1;
    }
}

fn spawn_every(tag, count, interval, opts) {
    let i = 0;
    while i < count {
        __internal_push_time();
        wait(i * interval);
        spawn(tag, 1, opts);
        __internal_pop_time();
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
    pub seed: u64,
    pub next_group_id: u64,
    pub next_agent_id: u64,
    pub next_event_order: u64,
    pub ended: bool,
}

impl Default for ScriptContext {
    fn default() -> Self {
        let seed = rand::random::<u64>();
        Self {
            cursor: 0.0,
            time_stack: Vec::new(),
            scenario: Scenario {
                seed,
                scene_markers: Vec::new(),
                events: Vec::new(),
                duration_sec: 0.0,
            },
            seed,
            next_group_id: 1,
            next_agent_id: 1,
            next_event_order: 1,
            ended: false,
        }
    }
}

impl ScriptContext {
    fn ensure_not_ended(&self, position: Position) -> Result<(), Box<EvalAltResult>> {
        if self.ended {
            return Err(Box::new(EvalAltResult::ErrorRuntime(
                "script already ended".into(),
                position,
            )));
        }
        Ok(())
    }

    pub fn scene(&mut self, name: &str, position: Position) -> Result<(), Box<EvalAltResult>> {
        self.ensure_not_ended(position)?;
        let order = self.next_event_order;
        self.next_event_order += 1;
        self.scenario.scene_markers.push(SceneMarker {
            name: name.to_string(),
            time: self.cursor,
            order,
        });
        Ok(())
    }

    pub fn wait(&mut self, sec: f32, position: Position) -> Result<(), Box<EvalAltResult>> {
        self.ensure_not_ended(position)?;
        self.cursor += sec.max(0.0);
        Ok(())
    }

    pub fn set_time(
        &mut self,
        time_sec: f32,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        self.ensure_not_ended(position)?;
        self.cursor = time_sec.max(0.0);
        Ok(())
    }

    pub fn set_seed(&mut self, seed: i64, position: Position) -> Result<(), Box<EvalAltResult>> {
        self.ensure_not_ended(position)?;
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

    pub fn push_time(&mut self, position: Position) -> Result<(), Box<EvalAltResult>> {
        self.ensure_not_ended(position)?;
        self.time_stack.push(self.cursor);
        Ok(())
    }

    pub fn pop_time(&mut self, position: Position) -> Result<(), Box<EvalAltResult>> {
        self.ensure_not_ended(position)?;
        if let Some(t) = self.time_stack.pop() {
            self.cursor = t;
        }
        Ok(())
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
        self.ensure_not_ended(position)?;
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
        self.ensure_not_ended(position)?;
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

    pub fn intent(
        &mut self,
        freq_hz: f32,
        dt: f32,
        duration_sec: f32,
        amp: f32,
        opts: Option<Map>,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        self.ensure_not_ended(position)?;

        let mut source_id: u64 = 0;
        let mut tag: Option<String> = None;
        let mut confidence: f32 = 1.0;

        if let Some(opts) = opts {
            let mut unknown_keys = Vec::new();
            for key in opts.keys() {
                match key.as_str() {
                    "source_id" | "tag" | "confidence" => {}
                    _ => unknown_keys.push(key.to_string()),
                }
            }
            if !unknown_keys.is_empty() {
                unknown_keys.sort();
                let msg = format!(
                    "intent opts has unknown keys: [{}] (allowed: source_id, tag, confidence)",
                    unknown_keys.join(", ")
                );
                return Err(Box::new(EvalAltResult::ErrorRuntime(msg.into(), position)));
            }

            if let Some(value) = opts.get("source_id") {
                let id = value
                    .as_int()
                    .ok()
                    .or_else(|| value.as_float().ok().map(|v| v as i64))
                    .ok_or_else(|| {
                        Box::new(EvalAltResult::ErrorRuntime(
                            "intent opts source_id must be an integer".into(),
                            position,
                        ))
                    })?;
                if id < 0 {
                    return Err(Box::new(EvalAltResult::ErrorRuntime(
                        "intent opts source_id must be non-negative".into(),
                        position,
                    )));
                }
                source_id = id as u64;
            }

            if let Some(value) = opts.get("tag") {
                let tag_value = value.clone().try_cast::<String>().ok_or_else(|| {
                    Box::new(EvalAltResult::ErrorRuntime(
                        "intent opts tag must be a string".into(),
                        position,
                    ))
                })?;
                tag = Some(tag_value);
            }

            if let Some(value) = opts.get("confidence") {
                confidence = value
                    .as_float()
                    .ok()
                    .map(|v| v as f32)
                    .or_else(|| value.as_int().ok().map(|v| v as f32))
                    .ok_or_else(|| {
                        Box::new(EvalAltResult::ErrorRuntime(
                            "intent opts confidence must be a number".into(),
                            position,
                        ))
                    })?;
            }
        }

        let onset_sec = (self.cursor + dt).max(0.0);
        let duration_sec = duration_sec.max(0.0);
        let action = Action::PostIntent {
            source_id,
            onset_sec,
            duration_sec,
            freq_hz,
            amp,
            tag,
            confidence,
        };
        self.push_event(self.cursor, vec![action]);
        Ok(())
    }

    pub fn add_agent(
        &mut self,
        tag: &str,
        freq: f32,
        amp: f32,
        life_map: Map,
        position: Position,
    ) -> Result<AgentHandle, Box<EvalAltResult>> {
        self.ensure_not_ended(position)?;
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

    pub fn set_plan_rate_source(&mut self, source_id: i64, rate: f32) {
        // Avoid accidentally targeting agent 0 on invalid input.
        if source_id < 0 {
            return;
        }
        let id = source_id as u64;
        self.push_event(
            self.cursor,
            vec![Action::SetPlanRate {
                target: TargetRef::AgentId { id },
                plan_rate: rate,
            }],
        );
    }

    pub fn set_plan_rate_tag(&mut self, tag: &str, rate: f32) {
        self.push_event(
            self.cursor,
            vec![Action::SetPlanRate {
                target: TargetRef::Tag {
                    tag: tag.to_string(),
                },
                plan_rate: rate,
            }],
        );
    }

    pub fn set_rhythm_vitality(
        &mut self,
        value: f32,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        self.ensure_not_ended(position)?;
        self.push_event(self.cursor, vec![Action::SetRhythmVitality { value }]);
        Ok(())
    }

    pub fn set_global_coupling(
        &mut self,
        value: f32,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        self.ensure_not_ended(position)?;
        self.push_event(self.cursor, vec![Action::SetGlobalCoupling { value }]);
        Ok(())
    }

    pub fn set_roughness_tolerance(
        &mut self,
        value: f32,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        self.ensure_not_ended(position)?;
        self.push_event(self.cursor, vec![Action::SetRoughnessTolerance { value }]);
        Ok(())
    }

    pub fn set_harmonicity(
        &mut self,
        map: Map,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        self.ensure_not_ended(position)?;
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
        self.ensure_not_ended(position)?;
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

    pub fn remove(
        &mut self,
        target: TargetRef,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        self.ensure_not_ended(position)?;
        self.push_event(self.cursor, vec![Action::RemoveAgent { target }]);
        Ok(())
    }

    pub fn release(
        &mut self,
        target: TargetRef,
        sec: f32,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        self.ensure_not_ended(position)?;
        self.push_event(
            self.cursor,
            vec![Action::ReleaseAgent {
                target,
                release_sec: sec,
            }],
        );
        Ok(())
    }

    pub fn end(&mut self, position: Position) -> Result<(), Box<EvalAltResult>> {
        self.end_at(self.cursor, position)
    }

    pub fn end_at(&mut self, t_abs: f32, position: Position) -> Result<(), Box<EvalAltResult>> {
        if t_abs < 0.0 {
            return Err(Box::new(EvalAltResult::ErrorRuntime(
                "end_at time must be non-negative".into(),
                position,
            )));
        }
        self.ensure_not_ended(position)?;
        let end = t_abs.max(0.0);
        self.scenario.duration_sec = self.scenario.duration_sec.max(end);
        self.push_event(end, vec![Action::Finish]);
        self.ended = true;
        Ok(())
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
    fn last_non_empty_line(full: &str) -> u16 {
        let mut last_non_empty = None;
        let mut count = 0usize;
        for (idx, line) in full.lines().enumerate() {
            count = idx + 1;
            if !line.trim().is_empty() {
                last_non_empty = Some(idx + 1);
            }
        }
        let line = last_non_empty.unwrap_or(count.max(1));
        line as u16
    }

    fn combined_script(src: &str) -> String {
        format!("{SCRIPT_PRELUDE}\n\n{src}")
    }

    fn create_engine(ctx: Arc<Mutex<ScriptContext>>) -> Engine {
        let mut engine = Engine::new();
        engine.on_print(|msg| println!("[rhai] {msg}"));
        engine.register_type_with_name::<CohortHandle>("CohortHandle");
        engine.register_type_with_name::<AgentHandle>("AgentHandle");
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

        let ctx_for_intent = ctx.clone();
        engine.register_fn(
            "intent",
            move |call_ctx: NativeCallContext,
                  freq_hz: FLOAT,
                  dt: FLOAT,
                  dur: FLOAT,
                  amp: FLOAT|
                  -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_intent.lock().expect("lock script context");
                ctx.intent(
                    freq_hz as f32,
                    dt as f32,
                    dur as f32,
                    amp as f32,
                    None,
                    call_ctx.call_position(),
                )
            },
        );
        let ctx_for_intent_opts = ctx.clone();
        engine.register_fn(
            "intent",
            move |call_ctx: NativeCallContext,
                  freq_hz: FLOAT,
                  dt: FLOAT,
                  dur: FLOAT,
                  amp: FLOAT,
                  opts: Map|
                  -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_intent_opts.lock().expect("lock script context");
                ctx.intent(
                    freq_hz as f32,
                    dt as f32,
                    dur as f32,
                    amp as f32,
                    Some(opts),
                    call_ctx.call_position(),
                )
            },
        );

        let ctx_for_scene = ctx.clone();
        engine.register_fn(
            "scene",
            move |call_ctx: NativeCallContext, name: &str| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_scene.lock().expect("lock script context");
                ctx.scene(name, call_ctx.call_position())
            },
        );

        let ctx_for_wait = ctx.clone();
        engine.register_fn(
            "wait",
            move |call_ctx: NativeCallContext, sec: FLOAT| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_wait.lock().expect("lock script context");
                ctx.wait(sec as f32, call_ctx.call_position())
            },
        );
        let ctx_for_wait_int = ctx.clone();
        engine.register_fn(
            "wait",
            move |call_ctx: NativeCallContext, sec: i64| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_wait_int.lock().expect("lock script context");
                ctx.wait(sec as f32, call_ctx.call_position())
            },
        );

        let ctx_for_push_time = ctx.clone();
        engine.register_fn(
            "__internal_push_time",
            move |call_ctx: NativeCallContext| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_push_time.lock().expect("lock script context");
                push_time(&mut ctx, call_ctx.call_position())
            },
        );

        let ctx_for_pop_time = ctx.clone();
        engine.register_fn(
            "__internal_pop_time",
            move |call_ctx: NativeCallContext| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_pop_time.lock().expect("lock script context");
                pop_time(&mut ctx, call_ctx.call_position())
            },
        );
        let ctx_for_set_time = ctx.clone();
        engine.register_fn(
            "__internal_set_time",
            move |call_ctx: NativeCallContext, sec: FLOAT| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_set_time.lock().expect("lock script context");
                set_time(&mut ctx, sec, call_ctx.call_position())
            },
        );

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

        let ctx_for_set_vitality = ctx.clone();
        engine.register_fn(
            "set_rhythm_vitality",
            move |call_ctx: NativeCallContext, value: FLOAT| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_set_vitality.lock().expect("lock script context");
                ctx.set_rhythm_vitality(value as f32, call_ctx.call_position())
            },
        );

        let ctx_for_set_coupling = ctx.clone();
        engine.register_fn(
            "set_global_coupling",
            move |call_ctx: NativeCallContext, value: FLOAT| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_set_coupling.lock().expect("lock script context");
                ctx.set_global_coupling(value as f32, call_ctx.call_position())
            },
        );

        let ctx_for_set_roughness = ctx.clone();
        engine.register_fn(
            "set_roughness_tolerance",
            move |call_ctx: NativeCallContext, value: FLOAT| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_set_roughness.lock().expect("lock script context");
                ctx.set_roughness_tolerance(value as f32, call_ctx.call_position())
            },
        );

        let ctx_for_set_harmonicity = ctx.clone();
        engine.register_fn(
            "set_harmonicity",
            move |call_ctx: NativeCallContext, map: Map| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_set_harmonicity.lock().expect("lock script context");
                ctx.set_harmonicity(map, call_ctx.call_position())
            },
        );

        let ctx_for_set_seed = ctx.clone();
        engine.register_fn(
            "set_seed",
            move |call_ctx: NativeCallContext, seed: INT| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_set_seed.lock().expect("lock script context");
                ctx.set_seed(seed, call_ctx.call_position())
            },
        );

        let ctx_for_set_plan_rate_source = ctx.clone();
        engine.register_fn(
            "set_plan_rate_source",
            move |source_id: INT, rate: FLOAT| {
                let mut ctx = ctx_for_set_plan_rate_source
                    .lock()
                    .expect("lock script context");
                ctx.set_plan_rate_source(source_id, rate as f32);
            },
        );

        let ctx_for_set_plan_rate_tag = ctx.clone();
        engine.register_fn("set_plan_rate_tag", move |tag: &str, rate: FLOAT| {
            let mut ctx = ctx_for_set_plan_rate_tag
                .lock()
                .expect("lock script context");
            ctx.set_plan_rate_tag(tag, rate as f32);
        });

        let ctx_for_remove_agent = ctx.clone();
        engine.register_fn(
            "remove",
            move |call_ctx: NativeCallContext, target: AgentHandle| {
                let mut ctx = ctx_for_remove_agent.lock().expect("lock script context");
                ctx.remove(
                    TargetRef::AgentId { id: target.id },
                    call_ctx.call_position(),
                )
            },
        );
        let ctx_for_remove_cohort = ctx.clone();
        engine.register_fn(
            "remove",
            move |call_ctx: NativeCallContext, target: CohortHandle| {
                let mut ctx = ctx_for_remove_cohort.lock().expect("lock script context");
                ctx.remove(
                    TargetRef::Range {
                        base_id: target.base_id,
                        count: target.count,
                    },
                    call_ctx.call_position(),
                )
            },
        );
        let ctx_for_remove_tag_str = ctx.clone();
        engine.register_fn("remove", move |call_ctx: NativeCallContext, tag: &str| {
            let mut ctx = ctx_for_remove_tag_str.lock().expect("lock script context");
            ctx.remove(
                TargetRef::Tag {
                    tag: tag.to_string(),
                },
                call_ctx.call_position(),
            )
        });

        let ctx_for_release_agent = ctx.clone();
        engine.register_fn(
            "release",
            move |call_ctx: NativeCallContext, target: AgentHandle, sec: FLOAT| {
                let mut ctx = ctx_for_release_agent.lock().expect("lock script context");
                ctx.release(
                    TargetRef::AgentId { id: target.id },
                    sec as f32,
                    call_ctx.call_position(),
                )
            },
        );
        let ctx_for_release_cohort = ctx.clone();
        engine.register_fn(
            "release",
            move |call_ctx: NativeCallContext, target: CohortHandle, sec: FLOAT| {
                let mut ctx = ctx_for_release_cohort.lock().expect("lock script context");
                ctx.release(
                    TargetRef::Range {
                        base_id: target.base_id,
                        count: target.count,
                    },
                    sec as f32,
                    call_ctx.call_position(),
                )
            },
        );
        let ctx_for_release_tag_str = ctx.clone();
        engine.register_fn(
            "release",
            move |call_ctx: NativeCallContext, tag: &str, sec: FLOAT| {
                let mut ctx = ctx_for_release_tag_str.lock().expect("lock script context");
                ctx.release(
                    TargetRef::Tag {
                        tag: tag.to_string(),
                    },
                    sec as f32,
                    call_ctx.call_position(),
                )
            },
        );

        let ctx_for_end = ctx.clone();
        engine.register_fn(
            "end",
            move |call_ctx: NativeCallContext| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_end.lock().expect("lock script context");
                ctx.end(call_ctx.call_position())
            },
        );
        let ctx_for_end_at = ctx.clone();
        engine.register_fn(
            "end_at",
            move |call_ctx: NativeCallContext, sec: FLOAT| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_end_at.lock().expect("lock script context");
                ctx.end_at(sec as f32, call_ctx.call_position())
            },
        );
        let ctx_for_end_at_int = ctx.clone();
        engine.register_fn(
            "end_at",
            move |call_ctx: NativeCallContext, sec: i64| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_end_at_int.lock().expect("lock script context");
                ctx.end_at(sec as f32, call_ctx.call_position())
            },
        );

        engine
    }

    pub fn load_script(path: &str) -> Result<Scenario, ScriptError> {
        let src = fs::read_to_string(path)
            .map_err(|err| ScriptError::new(format!("read script {path}: {err}"), None))?;
        let ctx = Arc::new(Mutex::new(ScriptContext::default()));
        let engine = ScriptHost::create_engine(ctx.clone());
        let script_src = ScriptHost::combined_script(&src);

        if let Err(e) = engine.eval::<()>(&script_src) {
            // Print structured error to help diagnose script issues (e.g., type mismatches).
            println!("Debug script error: {:?}", e);
            return Err(ScriptError::from_eval(
                e,
                Some(&format!("execute script {path}")),
            ));
        }

        let ctx_out = ctx.lock().expect("lock script context");
        if !ctx_out.ended {
            let line = ScriptHost::last_non_empty_line(&script_src);
            let pos = Position::new(line, 1);
            let msg = "script must call end() or end_at(t) to specify duration";
            return Err(
                ScriptError::new(msg, Some(pos)).with_context(format!("execute script {path}"))
            );
        }
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
    use rhai::Position;

    fn run_script(src: &str) -> Scenario {
        let ctx = Arc::new(Mutex::new(ScriptContext::default()));
        let engine = ScriptHost::create_engine(ctx.clone());
        let script_src = ScriptHost::combined_script(src);
        engine.eval::<()>(&script_src).expect("script runs");
        ctx.lock().expect("lock ctx").scenario.clone()
    }

    fn eval_script_error_position(src: &str) -> Position {
        let ctx = Arc::new(Mutex::new(ScriptContext::default()));
        let engine = ScriptHost::create_engine(ctx.clone());
        let script_src = ScriptHost::combined_script(src);
        match engine.eval::<()>(&script_src) {
            Ok(_) => {
                if !ctx.lock().expect("lock ctx").ended {
                    let line = ScriptHost::last_non_empty_line(&script_src);
                    Position::new(line, 1)
                } else {
                    panic!("script should error");
                }
            }
            Err(err) => err.position(),
        }
    }

    fn eval_script_error(src: &str) -> Box<EvalAltResult> {
        let ctx = Arc::new(Mutex::new(ScriptContext::default()));
        let engine = ScriptHost::create_engine(ctx.clone());
        let script_src = ScriptHost::combined_script(src);
        match engine.eval::<Dynamic>(&script_src) {
            Ok(_) => {
                if !ctx.lock().expect("lock ctx").ended {
                    let line = ScriptHost::last_non_empty_line(&script_src);
                    Box::new(EvalAltResult::ErrorRuntime(
                        "script must call end() or end_at(t) to specify duration".into(),
                        Position::new(line, 1),
                    ))
                } else {
                    panic!("script should error");
                }
            }
            Err(err) => err,
        }
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
            let lead_method = #{ mode: "random_log_uniform", min_freq: 440.0, max_freq: 440.0 };
            spawn("lead", 1, #{ amp: 0.2, method: lead_method, life: life });
            wait(1.0);
            scene("break");
            let hit_life = #{
                body: #{ core: "sine" },
                articulation: #{ core: "entrain", type: "decay", initial_energy: 0.8, half_life_sec: 0.2 },
                pitch: #{ core: "pitch_hill_climb" },
                perceptual: #{ tau_fast: 0.5, tau_slow: 6.0, w_boredom: 0.8, w_familiarity: 0.2 }
            };
            let hit_method = #{ mode: "random_log_uniform", min_freq: 880.0, max_freq: 880.0 };
            spawn("hit", 1, #{ amp: 0.1, method: hit_method, life: hit_life });
            wait(0.5);
            end();
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
                    Action::SpawnAgents { tag, .. } => {
                        if tag.as_deref() == Some("hit") {
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
        assert!(has_hit, "expected spawn event for hit");
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
                let pad_method = #{ mode: "random_log_uniform", min_freq: 200.0, max_freq: 200.0 };
                spawn("pad", 1, #{ amp: 0.1, method: pad_method, life: life });
            });
            wait(0.2);
            let after_life = #{
                body: #{ core: "sine" },
                articulation: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.3 },
                pitch: #{ core: "pitch_hill_climb" },
                perceptual: #{ tau_fast: 0.5, tau_slow: 6.0, w_boredom: 0.8, w_familiarity: 0.2 }
            };
            let after_method = #{ mode: "random_log_uniform", min_freq: 300.0, max_freq: 300.0 };
            spawn("after", 1, #{ amp: 0.1, method: after_method, life: after_life });
            end();
        "#,
        );

        let mut pad_time = None;
        let mut after_time = None;
        let mut finish_time = None;
        for ev in &scenario.events {
            for action in &ev.actions {
                match action {
                    Action::SpawnAgents { tag, .. } => match tag.as_deref() {
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
            spawn("tag", 3, #{ amp: 0.25, method: method, life: life });
            end();
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
            let init_method = #{ mode: "random_log_uniform", min_freq: 330.0, max_freq: 330.0 };
            spawn("init", 1, #{ amp: 0.2, method: init_method, life: life });
            wait(0.3);
            end();
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
    fn end_at_sets_duration() {
        let scenario = run_script(
            r#"
            let method = #{ mode: "random_log_uniform", min_freq: 200.0, max_freq: 200.0 };
            spawn("d", 1, #{ amp: 0.2, method: method, life: #{} });
            end_at(20);
        "#,
        );
        assert_time_close(scenario.duration_sec, 20.0);
        let has_finish = scenario
            .events
            .iter()
            .any(|ev| ev.time == 20.0 && ev.actions.iter().any(|a| matches!(a, Action::Finish)));
        assert!(has_finish, "expected finish at end_at time");
    }

    #[test]
    fn end_sets_duration_from_cursor() {
        let scenario = run_script(
            r#"
            wait(2);
            end();
        "#,
        );
        assert_time_close(scenario.duration_sec, 2.0);
    }

    #[test]
    fn end_at_negative_has_position() {
        let src = r#"
            end_at(-1);
        "#;
        let full = ScriptHost::combined_script(src);
        let pos = eval_script_error_position(src);
        let expected_line = expected_line(&full, "end_at(-1");
        assert_eq!(pos.line().unwrap(), expected_line);
    }

    #[test]
    fn end_is_terminal() {
        let src = r#"
            end_at(1);
            spawn("AFTER_END_123", 1);
        "#;
        let full = ScriptHost::combined_script(src);
        let pos = eval_script_error_position(src);
        let expected_line = expected_line(&full, "AFTER_END_123");
        assert_eq!(pos.line().unwrap(), expected_line);
    }

    #[test]
    fn end_is_required() {
        let src = r#"
            spawn("d", 1);
        "#;
        let err = eval_script_error(src);
        let msg = err.to_string();
        assert!(
            msg.contains("end()") || msg.contains("end_at"),
            "message: {msg}"
        );
        let pos = err.position();
        assert_ne!(pos, Position::NONE);
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
        let full = ScriptHost::combined_script(src);
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
        let full = ScriptHost::combined_script(src);
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
        let full = ScriptHost::combined_script(src);
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
        let full = ScriptHost::combined_script(src);
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
                if let Action::SpawnAgents { group_id, .. } = action {
                    assert!(*group_id > 0);
                    has_add = true;
                }
            }
        }
        let has_finish = scenario
            .events
            .iter()
            .any(|ev| ev.actions.iter().any(|a| matches!(a, Action::Finish)));
        assert!(has_add, "expected spawn event");
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
